#!/usr/bin/env python3
"""
Main entry point for Cooperative Vehicular Networks simulation.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from src.simulation_controller import SimulationController
from src.ml.traffic_predictor import TrafficPredictor
from src.sumo_integration.sumo_manager import SUMOManager
from src.ns3_integration.ns3_manager import NS3Manager

def check_environment():
    """Check if required environment variables are set."""
    missing_vars = []
    
    # Check for SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        missing_vars.append('SUMO_HOME')
        
    # Check for NS3_DIR
    if 'NS3_DIR' not in os.environ:
        missing_vars.append('NS3_DIR')
        
    if missing_vars:
        print("Error: Required environment variables not set:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables. Example:")
        print("  export SUMO_HOME=/path/to/sumo")
        print("  export NS3_DIR=/path/to/ns-3")
        return False
        
    return True

def check_executables():
    """Check if required executables exist and are in PATH."""
    missing_execs = []
    
    # Check for sumo
    try:
        subprocess.run(["sumo", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        missing_execs.append("sumo")
        
    if missing_execs:
        print("Error: Required executables not found in PATH:")
        for exe in missing_execs:
            print(f"  - {exe}")
        return False
        
    return True

def train_model(args):
    """Train the traffic prediction model."""
    print("Training traffic prediction model...")
    
    # Check if training data exists
    if not os.path.exists(args.data):
        print(f"Error: Training data file not found: {args.data}")
        return False
        
    # Create directories if they don't exist
    model_dir = os.path.dirname(args.output)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Initialize predictor
    predictor = TrafficPredictor(
        sequence_length=args.seq_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Load and preprocess data
    import pandas as pd
    data = pd.read_csv(args.data)
    scaled_data = predictor.preprocess_data(data, args.column)
    
    # Create datasets and dataloaders
    train_loader, val_loader, test_loader = predictor.create_datasets(
        scaled_data,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=args.batch_size
    )
    
    # Train model
    predictor.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        gpus=args.gpus,
        save_path=args.output
    )
    
    # Evaluate model
    test_results = predictor.evaluate(test_loader)
    print(f"Test loss: {test_results['test_loss']:.4f}")
    
    print(f"Model saved to: {args.output}")
    return True

def run_simulation(args):
    """Run the integrated simulation."""
    print("Starting Cooperative Vehicular Networks simulation...")
    
    # Check if config file exists
    if not os.path.exists(args.sumo_config):
        print(f"Error: SUMO configuration file not found: {args.sumo_config}")
        return False
        
    # Check if model exists if specified
    if args.model and not os.path.exists(args.model):
        print(f"Error: ML model file not found: {args.model}")
        return False
        
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Get NS-3 directory
    ns3_dir = args.ns3_dir or os.environ.get('NS3_DIR')
    
    # Initialize simulation controller
    controller = SimulationController(
        sumo_config=args.sumo_config,
        ns3_dir=ns3_dir,
        ml_model_path=args.model,
        output_dir=args.output_dir,
        tx_power_min=args.tx_power_min,
        tx_power_max=args.tx_power_max,
        log_level=args.log_level
    )
    
    try:
        # Set up simulation
        controller.setup_simulation(
            duration_seconds=args.duration,
            area_dimensions=(args.area_width, args.area_height)
        )
        
        # Run simulation
        controller.run_simulation(
            duration_seconds=args.duration,
            step_size=args.step_size
        )
        
        print(f"Simulation completed. Results saved to: {args.output_dir}")
        return True
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        return False
        
    finally:
        # Clean up resources
        controller.cleanup()

def main():
    """Parse command line arguments and run the appropriate function."""
    parser = argparse.ArgumentParser(
        description="Cooperative Vehicular Networks Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train model subcommand
    train_parser = subparsers.add_parser("train", help="Train traffic prediction model")
    train_parser.add_argument("--data", required=True, help="Path to training data CSV file")
    train_parser.add_argument("--output", required=True, help="Path to save trained model")
    train_parser.add_argument("--column", default="vehicles_per_minute", help="Column to use for prediction")
    train_parser.add_argument("--seq-length", type=int, default=12, help="Input sequence length")
    train_parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    train_parser.add_argument("--num-layers", type=int, default=4, help="Number of LSTM layers")
    train_parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=600, help="Number of training epochs")
    train_parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use (0 for CPU)")
    
    # Run simulation subcommand
    sim_parser = subparsers.add_parser("simulate", help="Run integrated simulation")
    sim_parser.add_argument("--sumo-config", required=True, help="Path to SUMO configuration file")
    sim_parser.add_argument("--model", help="Path to pre-trained ML model (optional)")
    sim_parser.add_argument("--ns3-dir", help="Path to NS-3 installation directory (defaults to NS3_DIR env var)")
    sim_parser.add_argument("--output-dir", default="results", help="Directory for output files")
    sim_parser.add_argument("--duration", type=int, default=3600, help="Simulation duration in seconds")
    sim_parser.add_argument("--step-size", type=float, default=1.0, help="Simulation step size in seconds")
    sim_parser.add_argument("--area-width", type=float, default=1500.0, help="Simulation area width in meters")
    sim_parser.add_argument("--area-height", type=float, default=750.0, help="Simulation area height in meters")
    sim_parser.add_argument("--tx-power-min", type=float, default=5.0, help="Minimum transmit power in dBm")
    sim_parser.add_argument("--tx-power-max", type=float, default=35.0, help="Maximum transmit power in dBm")
    sim_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                          help="Logging level")
    
    args = parser.parse_args()
    
    # Check if command is specified
    if not args.command:
        parser.print_help()
        return 1
        
    # Check environment
    if not check_environment() or not check_executables():
        return 1
        
    # Run appropriate function
    if args.command == "train":
        success = train_model(args)
    elif args.command == "simulate":
        success = run_simulation(args)
    else:
        parser.print_help()
        return 1
        
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 