import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

# Import project modules
from src.sumo_integration.sumo_manager import SUMOManager
from src.ns3_integration.ns3_manager import NS3Manager
from src.ml.traffic_predictor import TrafficPredictor
from src.relay_selection.optimal_stopping import AdaptiveRelaySelector

class SimulationController:
    """
    Controller class for integrated simulation of Cooperative Vehicular Networks.
    Coordinates SUMO, NS-3, ML prediction, and relay selection.
    """
    
    def __init__(self, 
                 sumo_config: str,
                 ns3_dir: str,
                 ml_model_path: Optional[str] = None,
                 output_dir: str = 'results',
                 tx_power_min: float = 5.0,
                 tx_power_max: float = 35.0,
                 log_level: str = 'INFO'):
        """
        Initialize the simulation controller.
        
        Args:
            sumo_config: Path to SUMO configuration file
            ns3_dir: Path to NS-3 installation directory
            ml_model_path: Path to pre-trained ML model (optional)
            output_dir: Directory for output files
            tx_power_min: Minimum transmit power in dBm
            tx_power_max: Maximum transmit power in dBm
            log_level: Logging level
        """
        # Set up logging
        self._setup_logging(log_level)
        
        # Store parameters
        self.output_dir = output_dir
        self.tx_power_min = tx_power_min
        self.tx_power_max = tx_power_max
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize SUMO
        self.logger.info("Initializing SUMO manager...")
        self.sumo = SUMOManager(sumo_config, sim_step=0.05, gui=False)
        
        # Initialize NS-3
        self.logger.info("Initializing NS-3 manager...")
        self.ns3 = NS3Manager(ns3_dir, sim_step=0.05)
        
        # Initialize ML predictor
        self.logger.info("Initializing traffic predictor...")
        self.predictor = TrafficPredictor(model_path=ml_model_path)
        
        # Initialize relay selector
        self.logger.info("Initializing relay selector...")
        self.relay_selector = AdaptiveRelaySelector()
        
        # Simulation state
        self.is_running = False
        self.current_time = 0.0
        self.vehicle_density_history = []
        self.rsu_positions = []
        
    def _setup_logging(self, log_level: str):
        """Set up logging configuration."""
        # Configure logger
        logging_level = getattr(logging, log_level.upper())
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("SimulationController")
        
    def setup_simulation(self, duration_seconds: int = 43200, area_dimensions: Tuple[float, float] = (1500.0, 750.0)):
        """
        Set up the simulation environment.
        
        Args:
            duration_seconds: Duration of simulation in seconds
            area_dimensions: Dimensions of simulation area (width, height)
        """
        # Start SUMO
        self.logger.info("Starting SUMO simulation...")
        self.sumo.start_simulation()
        
        # Create NS-3 simulation
        self.logger.info("Creating NS-3 simulation...")
        self.ns3.create_simulation(duration_seconds, area_dimensions)
        
        # Set up WAVE devices
        self.logger.info("Setting up WAVE devices...")
        self.ns3.setup_wave_devices(tx_power_min=self.tx_power_min, tx_power_max=self.tx_power_max)
        
        # Place RSUs using K-means clustering
        self.logger.info("Placing RSUs...")
        self._place_rsus(num_rsus=3)
        
        self.logger.info("Simulation setup complete.")
        
    def _place_rsus(self, num_rsus: int = 3):
        """
        Place RSUs in the simulation area using K-means clustering.
        
        Args:
            num_rsus: Number of RSUs to place
        """
        # In a real implementation, we would run K-means clustering on historical data
        # For this example, we'll use predefined positions from the paper
        rsu_positions = [
            (200, 150, 5),  # (x, y, z) position in meters
            (700, 250, 5),
            (1200, 500, 5)
        ]
        
        # Add RSUs to NS-3 simulation
        for i, pos in enumerate(rsu_positions):
            rsu_id = self.ns3.add_rsu(position=pos, tx_power=self.tx_power_min)
            self.rsu_positions.append((rsu_id, pos))
            self.logger.info(f"Added RSU {rsu_id} at position {pos}")
            
    def run_simulation(self, duration_seconds: int, step_size: float = 1.0):
        """
        Run the integrated simulation.
        
        Args:
            duration_seconds: Duration to run in seconds
            step_size: Step size in seconds
        """
        if not self.sumo._is_running or not self.ns3._is_running:
            self.logger.error("Simulation not set up. Call setup_simulation() first.")
            return
            
        self.is_running = True
        start_time = time.time()
        
        # Run for specified duration
        num_steps = int(duration_seconds / step_size)
        
        self.logger.info(f"Starting simulation for {duration_seconds} seconds with {num_steps} steps...")
        
        for step in range(num_steps):
            self.current_time = step * step_size
            
            # Display progress
            if step % 100 == 0:
                self.logger.info(f"Simulation time: {self.current_time:.1f}s ({(step/num_steps)*100:.1f}%)")
                
            # Step SUMO
            num_vehicles = self.sumo.step()
            
            # Store vehicle density
            vehicles_per_minute = self.sumo.get_vehicles_per_minute()
            self.vehicle_density_history.append((self.current_time, vehicles_per_minute))
            
            # Every 5 seconds, predict traffic and update RSU power
            if step % 5 == 0 and len(self.vehicle_density_history) >= 12:
                self._predict_and_adapt()
                
            # Every step, update vehicle positions in NS-3
            self._update_vehicle_positions()
            
            # Every step, simulate packet transmissions
            self._simulate_transmissions()
            
            # Sleep to slow down simulation if needed (for real-time visualization)
            # time.sleep(0.001)
            
        # End of simulation
        end_time = time.time()
        self.is_running = False
        
        self.logger.info(f"Simulation completed in {end_time - start_time:.2f} seconds wall clock time.")
        
        # Export results
        self._export_results()
        
    def _predict_and_adapt(self):
        """Predict traffic density and adapt RSU transmit power."""
        # Extract density history for prediction
        recent_density = [d[1] for d in self.vehicle_density_history[-12:]]
        density_array = np.array(recent_density).reshape(-1, 1)
        
        # Predict future density
        try:
            predicted_density = self.predictor.predict(density_array)
            self.logger.debug(f"Predicted vehicle density: {predicted_density:.2f} vehicles/min")
            
            # Adapt RSU transmit power based on prediction
            # Higher density -> more power
            for rsu_id, _ in self.rsu_positions:
                # Linear interpolation between min and max power based on density
                # Assume max density is 30 vehicles/min
                norm_density = min(1.0, predicted_density / 30.0)
                power_range = self.tx_power_max - self.tx_power_min
                new_power = self.tx_power_min + norm_density * power_range
                
                # Update RSU power
                self.ns3.update_rsu_tx_power(rsu_id, new_power)
                self.logger.debug(f"Updated RSU {rsu_id} power to {new_power:.2f} dBm")
                
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            
    def _update_vehicle_positions(self):
        """Update vehicle positions in NS-3 based on SUMO data."""
        try:
            # Get vehicles and their positions from SUMO
            vehicle_positions = {}
            vehicle_ids = self.sumo.vit['vehicle_id'].unique()
            
            for v_id in vehicle_ids:
                vehicle_data = self.sumo.vit[self.sumo.vit['vehicle_id'] == v_id].iloc[-1]
                pos = (vehicle_data['position_x'], vehicle_data['position_y'], 1.5)  # 1.5m height
                vehicle_positions[v_id] = pos
                
            # Update positions in NS-3
            self.ns3.update_vehicle_positions(vehicle_positions)
            
        except Exception as e:
            self.logger.error(f"Error updating vehicle positions: {e}")
            
    def _simulate_transmissions(self):
        """Simulate packet transmissions between vehicles and RSUs."""
        try:
            # Get active vehicles
            vehicle_ids = list(self.ns3.vehicles.keys())
            
            # Skip if no vehicles
            if not vehicle_ids:
                return
                
            # Randomly select source and destination vehicles
            if len(vehicle_ids) >= 2:
                source_id = np.random.choice(vehicle_ids)
                dest_id = np.random.choice([v for v in vehicle_ids if v != source_id])
                
                # Calculate direct link SNR
                direct_link_snr = self.ns3.calculate_snr(source_id, dest_id)
                
                # Prepare candidate relays (both RSUs and other vehicles)
                candidates = []
                
                # Add RSUs as candidates
                for rsu_id in self.ns3.rsus:
                    # Calculate SNR from source to RSU and from RSU to destination
                    snr_src_to_rsu = self.ns3.calculate_snr(source_id, rsu_id)
                    snr_rsu_to_dest = self.ns3.calculate_snr(rsu_id, dest_id)
                    
                    # End-to-end SNR is the minimum of the two links
                    snr = min(snr_src_to_rsu, snr_rsu_to_dest)
                    
                    candidates.append({
                        'id': rsu_id,
                        'type': 'rsu',
                        'snr': snr
                    })
                    
                # Add other vehicles as candidates
                for v_id in vehicle_ids:
                    if v_id != source_id and v_id != dest_id:
                        # Calculate SNR from source to vehicle and from vehicle to destination
                        snr_src_to_veh = self.ns3.calculate_snr(source_id, v_id)
                        snr_veh_to_dest = self.ns3.calculate_snr(v_id, dest_id)
                        
                        # End-to-end SNR is the minimum of the two links
                        snr = min(snr_src_to_veh, snr_veh_to_dest)
                        
                        candidates.append({
                            'id': v_id,
                            'type': 'vehicle',
                            'snr': snr
                        })
                
                # Select relay using optimal stopping
                relay, use_relay, changed = self.relay_selector.update_relay(candidates, direct_link_snr)
                
                # Simulate transmission
                if use_relay:
                    # Use relay: source -> relay -> destination
                    relay_id = relay['id']
                    
                    # First hop: source to relay
                    res1 = self.ns3.simulate_packet_transmission(source_id, relay_id)
                    
                    # Second hop: relay to destination (only if first hop succeeded)
                    if res1['received']:
                        res2 = self.ns3.simulate_packet_transmission(relay_id, dest_id)
                        success = res2['received']
                        delay = res1['delay'] + res2['delay']
                    else:
                        success = False
                        delay = res1['delay']
                        
                    self.logger.debug(f"Transmission via relay {relay_id}: {success}")
                    
                else:
                    # Direct transmission
                    res = self.ns3.simulate_packet_transmission(source_id, dest_id)
                    success = res['received']
                    delay = res['delay']
                    
                    self.logger.debug(f"Direct transmission: {success}")
                    
        except Exception as e:
            self.logger.error(f"Error in packet transmission: {e}")
            
    def _export_results(self):
        """Export simulation results to CSV files."""
        try:
            # Create results directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Export vehicle density history
            density_df = pd.DataFrame(self.vehicle_density_history, columns=['time', 'vehicles_per_minute'])
            density_df.to_csv(os.path.join(self.output_dir, 'vehicle_density.csv'), index=False)
            
            # Export packet traces from NS-3
            self.ns3.export_packet_traces(os.path.join(self.output_dir, 'packet_traces.csv'))
            
            # Export vehicle information table from SUMO
            self.sumo.export_vit(os.path.join(self.output_dir, 'vehicle_info.csv'))
            
            self.logger.info(f"Results exported to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            
    def cleanup(self):
        """Clean up simulation resources."""
        try:
            # Close SUMO
            if hasattr(self, 'sumo'):
                self.sumo.close()
                
            # Reset NS-3
            if hasattr(self, 'ns3'):
                self.ns3.reset_simulation()
                
            self.logger.info("Simulation resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up: {e}")
            
    def __del__(self):
        """Clean up on object destruction."""
        self.cleanup() 