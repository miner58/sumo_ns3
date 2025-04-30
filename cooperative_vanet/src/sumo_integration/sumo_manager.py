import os
import sys
import traci
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class SUMOManager:
    """
    Manager class for SUMO traffic simulation integration.
    Handles vehicle movement, data collection, and TraCI communication.
    """
    
    def __init__(self, 
                config_file: str,
                sim_step: float = 0.05,
                port: int = 8813,
                gui: bool = False):
        """
        Initialize the SUMO manager.
        
        Args:
            config_file: Path to SUMO configuration file (.sumocfg)
            sim_step: Simulation step size in seconds
            port: Port for TraCI connection
            gui: Whether to show SUMO GUI or run in headless mode
        """
        self.config_file = config_file
        self.sim_step = sim_step
        self.port = port
        self.gui = gui
        self.vehicle_data = {}
        self._is_running = False
        
        # Vehicle Information Table (VIT) as mentioned in the paper
        self.vit = pd.DataFrame(columns=[
            'vehicle_id', 'position_x', 'position_y', 'speed', 
            'acceleration', 'heading', 'timestamp'
        ])
        
    def start_simulation(self):
        """Start the SUMO simulation with TraCI connection."""
        if self._is_running:
            print("Simulation is already running.")
            return
            
        # Prepare SUMO command
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.config_file,
            "--step-length", str(self.sim_step),
            "--start",
            "--quit-on-end"
        ]
        
        # Start TraCI connection
        traci.start(sumo_cmd, port=self.port)
        self._is_running = True
        print(f"SUMO simulation started with config: {self.config_file}")
        
    def step(self) -> int:
        """
        Advance the simulation by one step.
        
        Returns:
            Number of vehicles currently in the simulation
        """
        if not self._is_running:
            raise RuntimeError("Simulation is not running. Call start_simulation() first.")
        
        traci.simulationStep()
        self._update_vehicle_data()
        
        return len(traci.vehicle.getIDList())
    
    def _update_vehicle_data(self):
        """Update the vehicle information table with current vehicle data."""
        current_time = traci.simulation.getTime()
        vehicle_ids = traci.vehicle.getIDList()
        
        # Prepare data for VIT update
        new_data = []
        for v_id in vehicle_ids:
            pos = traci.vehicle.getPosition(v_id)
            speed = traci.vehicle.getSpeed(v_id)
            accel = traci.vehicle.getAcceleration(v_id)
            angle = traci.vehicle.getAngle(v_id)
            
            new_data.append({
                'vehicle_id': v_id,
                'position_x': pos[0],
                'position_y': pos[1],
                'speed': speed,
                'acceleration': accel,
                'heading': angle,
                'timestamp': current_time
            })
        
        # Update VIT
        new_df = pd.DataFrame(new_data)
        if not new_df.empty:
            self.vit = pd.concat([self.vit, new_df], ignore_index=True)
            # Keep only recent data (last 5 minutes)
            self.vit = self.vit[self.vit['timestamp'] > current_time - 300]
    
    def get_vehicle_density_map(self, grid_size: Tuple[int, int] = (10, 5)) -> np.ndarray:
        """
        Create a vehicle density heatmap based on current vehicle positions.
        
        Args:
            grid_size: Size of the density grid (rows, cols)
            
        Returns:
            numpy array representing vehicle density grid
        """
        # Get network boundaries
        net_bounds = traci.simulation.getNetBoundary()
        x_min, y_min = net_bounds[0]
        x_max, y_max = net_bounds[1]
        
        # Initialize density grid
        density = np.zeros(grid_size)
        
        # Get current vehicles
        vehicles = traci.vehicle.getIDList()
        
        # Calculate grid cell sizes
        cell_width = (x_max - x_min) / grid_size[1]
        cell_height = (y_max - y_min) / grid_size[0]
        
        # Fill density grid
        for v_id in vehicles:
            pos = traci.vehicle.getPosition(v_id)
            grid_x = min(int((pos[0] - x_min) / cell_width), grid_size[1] - 1)
            grid_y = min(int((pos[1] - y_min) / cell_height), grid_size[0] - 1)
            density[grid_y, grid_x] += 1
            
        return density
    
    def get_vehicles_per_minute(self, interval: int = 60) -> int:
        """
        Calculate vehicles per minute based on the VIT data.
        
        Args:
            interval: Time interval in seconds to consider
            
        Returns:
            Number of unique vehicles in the last interval
        """
        current_time = traci.simulation.getTime()
        recent_vit = self.vit[self.vit['timestamp'] > current_time - interval]
        return len(recent_vit['vehicle_id'].unique())
    
    def close(self):
        """Close the TraCI connection and clean up."""
        if self._is_running:
            traci.close()
            self._is_running = False
            print("SUMO simulation stopped")
    
    def export_vit(self, output_path: str):
        """
        Export the Vehicle Information Table to CSV.
        
        Args:
            output_path: Path to save the CSV file
        """
        self.vit.to_csv(output_path, index=False)
        print(f"Vehicle Information Table saved to: {output_path}")
    
    def __del__(self):
        """Clean up on object destruction."""
        self.close() 