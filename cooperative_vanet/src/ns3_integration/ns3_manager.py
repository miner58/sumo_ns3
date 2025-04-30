import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable

class NS3Manager:
    """
    Manager class for NS-3 network simulation integration.
    Handles network simulation, packet transmission, and data collection.
    """
    
    def __init__(self,
                 ns3_dir: str,
                 sim_step: float = 0.05,
                 use_wave: bool = True):
        """
        Initialize the NS-3 manager.
        
        Args:
            ns3_dir: Path to NS-3 installation directory
            sim_step: Simulation step size in seconds
            use_wave: Whether to use IEEE 802.11p WAVE module
        """
        self.ns3_dir = ns3_dir
        self.sim_step = sim_step
        self.use_wave = use_wave
        self._is_running = False
        
        # Set up NS-3 environment
        self._setup_ns3_environment()
        
        # Initialize packet trace data
        self.packet_traces = pd.DataFrame(columns=[
            'timestamp', 'source_id', 'dest_id', 'packet_size', 'tx_power',
            'snr', 'delay', 'received'
        ])
        
        # RSU information
        self.rsus = {}
        
    def _setup_ns3_environment(self):
        """Set up the NS-3 environment for Python integration."""
        try:
            # Check for PYTHONPATH setup
            if not self.ns3_dir:
                if 'NS3_DIR' in os.environ:
                    self.ns3_dir = os.environ['NS3_DIR']
                else:
                    raise ValueError("NS3_DIR not set. Please provide ns3_dir or set NS3_DIR environment variable.")
            
            # Append NS-3 paths to Python path
            python_binding_path = os.path.join(self.ns3_dir, 'build', 'bindings', 'python')
            if python_binding_path not in sys.path:
                sys.path.append(python_binding_path)
                
            # Import NS-3 modules
            import ns.core
            import ns.network
            import ns.internet
            import ns.mobility
            
            if self.use_wave:
                import ns.wave
                
            # Store modules for later use
            self.ns = {
                'core': ns.core,
                'network': ns.network,
                'internet': ns.internet,
                'mobility': ns.mobility
            }
            
            if self.use_wave:
                self.ns['wave'] = ns.wave
                
            print("NS-3 environment setup successful")
            
        except ImportError as e:
            print(f"Failed to import NS-3 modules: {e}")
            print("Make sure NS-3 is built with Python bindings enabled.")
            raise
            
    def create_simulation(self, 
                        duration_seconds: int, 
                        area_dimensions: Tuple[float, float] = (1500.0, 750.0)):
        """
        Create a new NS-3 simulation.
        
        Args:
            duration_seconds: Duration of simulation in seconds
            area_dimensions: Dimensions of simulation area (width, height)
        """
        # Set simulation time resolution
        self.ns['core'].Time.SetResolution(self.ns['core'].Time.NS)
        
        # Create simulation helper
        self.cmd_line = self.ns['core'].CommandLine()
        self.cmd_line.Parse([])
        
        # Configure simulation duration
        self.sim_stop_time = self.ns['core'].Seconds(duration_seconds)
        
        # Set up simulation area
        self.area_width, self.area_height = area_dimensions
        
        print(f"NS-3 simulation created with duration: {duration_seconds}s, "
              f"area: {self.area_width}x{self.area_height}m")
              
        self._is_running = True
        
    def setup_wave_devices(self, tx_power_min: float = 5.0, tx_power_max: float = 35.0):
        """
        Set up IEEE 802.11p WAVE devices for V2V and V2I communication.
        
        Args:
            tx_power_min: Minimum transmit power in dBm
            tx_power_max: Maximum transmit power in dBm
        """
        if not self.use_wave:
            raise RuntimeError("WAVE module not enabled. Initialize with use_wave=True")
            
        if not self._is_running:
            raise RuntimeError("Simulation not created. Call create_simulation() first.")
            
        # Store power parameters
        self.tx_power_min = tx_power_min
        self.tx_power_max = tx_power_max
        
        # Create PHY and MAC helpers
        self.wave_helper = self.ns['wave'].YansWaveHelper()
        
        # Configure PHY
        self.wave_phy = self.ns['wave'].YansWifiPhyHelper().Default()
        
        # Configure channel
        self.channel = self.ns['wave'].YansWifiChannelHelper.Default()
        self.wave_phy.SetChannel(self.channel.Create())
        
        # Set default TX power to minimum
        self.wave_phy.Set("TxPowerStart", self.ns['core'].DoubleValue(tx_power_min))
        self.wave_phy.Set("TxPowerEnd", self.ns['core'].DoubleValue(tx_power_min))
        
        # Configure MAC
        self.wave_mac = self.ns['wave'].NqosWaveMacHelper.Default()
        
        # Create WAVE devices
        self.wave_devices = self.wave_helper.Install(self.wave_phy, self.wave_mac, 
                                                     self.ns['network'].NodeContainer())
        
        print(f"WAVE devices set up with power range: {tx_power_min}-{tx_power_max} dBm")
        
    def add_rsu(self, 
               position: Tuple[float, float, float], 
               tx_power: float = None,
               rsu_id: str = None):
        """
        Add a Road-Side Unit (RSU) to the simulation.
        
        Args:
            position: (x, y, z) position in meters
            tx_power: Transmit power in dBm (default: tx_power_min)
            rsu_id: ID for the RSU (default: auto-generated)
            
        Returns:
            The ID of the created RSU
        """
        if not self._is_running:
            raise RuntimeError("Simulation not created. Call create_simulation() first.")
            
        # Create a node for the RSU
        rsu_node = self.ns['network'].Node()
        
        # Set position
        mobility = self.ns['mobility'].ConstantPositionMobilityModel()
        mobility.SetPosition(self.ns['core'].Vector3D(position[0], position[1], position[2]))
        rsu_node.AggregateObject(mobility)
        
        # Create device for RSU
        rsu_device = self.wave_helper.Install(self.wave_phy, self.wave_mac, 
                                              self.ns['network'].NodeContainer().Add(rsu_node)).Get(0)
        
        # Set transmit power
        if tx_power is None:
            tx_power = self.tx_power_min
            
        # Ensure power is within limits
        tx_power = min(max(tx_power, self.tx_power_min), self.tx_power_max)
        
        # Apply power setting
        rsu_device.GetPhy().Set("TxPowerStart", self.ns['core'].DoubleValue(tx_power))
        rsu_device.GetPhy().Set("TxPowerEnd", self.ns['core'].DoubleValue(tx_power))
        
        # Generate RSU ID if not provided
        if rsu_id is None:
            rsu_id = f"RSU_{len(self.rsus)}"
            
        # Store RSU information
        self.rsus[rsu_id] = {
            'node': rsu_node,
            'device': rsu_device,
            'position': position,
            'tx_power': tx_power
        }
        
        print(f"Added RSU {rsu_id} at position {position} with power {tx_power} dBm")
        return rsu_id
        
    def add_vehicle(self, 
                  position: Tuple[float, float, float],
                  velocity: Tuple[float, float, float] = (0, 0, 0),
                  vehicle_id: str = None):
        """
        Add a vehicle to the simulation.
        
        Args:
            position: (x, y, z) position in meters
            velocity: (vx, vy, vz) velocity in m/s
            vehicle_id: ID for the vehicle (default: auto-generated)
            
        Returns:
            The ID of the created vehicle
        """
        if not self._is_running:
            raise RuntimeError("Simulation not created. Call create_simulation() first.")
            
        # Create a node for the vehicle
        vehicle_node = self.ns['network'].Node()
        
        # Create mobility model for the vehicle
        mobility = self.ns['mobility'].ConstantVelocityMobilityModel()
        mobility.SetPosition(self.ns['core'].Vector3D(position[0], position[1], position[2]))
        mobility.SetVelocity(self.ns['core'].Vector3D(velocity[0], velocity[1], velocity[2]))
        vehicle_node.AggregateObject(mobility)
        
        # Create device for vehicle
        vehicle_device = self.wave_helper.Install(self.wave_phy, self.wave_mac, 
                                                 self.ns['network'].NodeContainer().Add(vehicle_node)).Get(0)
        
        # Set default power
        vehicle_device.GetPhy().Set("TxPowerStart", self.ns['core'].DoubleValue(self.tx_power_min))
        vehicle_device.GetPhy().Set("TxPowerEnd", self.ns['core'].DoubleValue(self.tx_power_min))
        
        # Generate vehicle ID if not provided
        if vehicle_id is None:
            vehicle_id = f"VEH_{vehicle_node.GetId()}"
            
        # Store vehicle information
        if not hasattr(self, 'vehicles'):
            self.vehicles = {}
            
        self.vehicles[vehicle_id] = {
            'node': vehicle_node,
            'device': vehicle_device,
            'position': position,
            'velocity': velocity,
            'tx_power': self.tx_power_min
        }
        
        return vehicle_id
        
    def update_vehicle_positions(self, vehicle_positions: Dict[str, Tuple[float, float, float]]):
        """
        Update the positions of vehicles in the simulation.
        
        Args:
            vehicle_positions: Dictionary mapping vehicle IDs to (x, y, z) positions
        """
        if not self._is_running:
            raise RuntimeError("Simulation not created. Call create_simulation() first.")
            
        for vehicle_id, position in vehicle_positions.items():
            if vehicle_id in self.vehicles:
                # Get mobility model
                mobility = self.vehicles[vehicle_id]['node'].GetObject(self.ns['mobility'].MobilityModel.GetTypeId())
                # Update position
                mobility.SetPosition(self.ns['core'].Vector3D(position[0], position[1], position[2]))
                # Update stored position
                self.vehicles[vehicle_id]['position'] = position
            else:
                # Create new vehicle if it doesn't exist
                self.add_vehicle(position, (0, 0, 0), vehicle_id)
                
    def update_rsu_tx_power(self, rsu_id: str, tx_power: float):
        """
        Update the transmit power of an RSU.
        
        Args:
            rsu_id: ID of the RSU
            tx_power: New transmit power in dBm
        """
        if not self._is_running:
            raise RuntimeError("Simulation not created. Call create_simulation() first.")
            
        if rsu_id not in self.rsus:
            raise ValueError(f"RSU with ID {rsu_id} not found")
            
        # Ensure power is within limits
        tx_power = min(max(tx_power, self.tx_power_min), self.tx_power_max)
        
        # Apply power setting
        rsu_device = self.rsus[rsu_id]['device']
        rsu_device.GetPhy().Set("TxPowerStart", self.ns['core'].DoubleValue(tx_power))
        rsu_device.GetPhy().Set("TxPowerEnd", self.ns['core'].DoubleValue(tx_power))
        
        # Update stored power
        self.rsus[rsu_id]['tx_power'] = tx_power
        
    def calculate_snr(self, source_id: str, dest_id: str) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR) between two nodes.
        
        Args:
            source_id: ID of source node (vehicle or RSU)
            dest_id: ID of destination node (vehicle or RSU)
            
        Returns:
            SNR value in dB
        """
        # Get source node
        if source_id in self.vehicles:
            source = self.vehicles[source_id]
        elif source_id in self.rsus:
            source = self.rsus[source_id]
        else:
            raise ValueError(f"Source node {source_id} not found")
            
        # Get destination node
        if dest_id in self.vehicles:
            dest = self.vehicles[dest_id]
        elif dest_id in self.rsus:
            dest = self.rsus[dest_id]
        else:
            raise ValueError(f"Destination node {dest_id} not found")
            
        # Get positions
        source_pos = source['position']
        dest_pos = dest['position']
        
        # Calculate distance
        dx = source_pos[0] - dest_pos[0]
        dy = source_pos[1] - dest_pos[1]
        dz = source_pos[2] - dest_pos[2]
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Get transmit power
        tx_power = source['tx_power']
        
        # Simple path loss model (free space)
        # SNR = Tx Power - Path Loss - Noise Floor
        # Path Loss (dB) = 20 * log10(distance) + 20 * log10(frequency) - 147.55
        frequency_ghz = 5.9  # 5.9 GHz for WAVE
        path_loss = 20 * np.log10(distance) + 20 * np.log10(frequency_ghz * 1e9) - 147.55
        noise_floor_dbm = -96  # Typical value for outdoor environment
        
        snr = tx_power - path_loss - noise_floor_dbm
        
        return snr
        
    def simulate_packet_transmission(self, 
                                   source_id: str, 
                                   dest_id: str, 
                                   packet_size: int = 1000) -> Dict[str, Any]:
        """
        Simulate transmission of a packet between two nodes.
        
        Args:
            source_id: ID of source node (vehicle or RSU)
            dest_id: ID of destination node (vehicle or RSU)
            packet_size: Size of packet in bytes
            
        Returns:
            Dictionary with transmission results
        """
        # Calculate SNR
        snr = self.calculate_snr(source_id, dest_id)
        
        # Get source information
        if source_id in self.vehicles:
            source = self.vehicles[source_id]
        elif source_id in self.rsus:
            source = self.rsus[source_id]
        else:
            raise ValueError(f"Source node {source_id} not found")
            
        # Simple packet error model based on SNR
        # PDR = 1 - exp(-SNR/SNR_threshold)
        snr_threshold = 10.0  # 10 dB threshold for reliable communication
        pdr = 1.0 - np.exp(-snr / snr_threshold) if snr > 0 else 0.0
        pdr = min(max(pdr, 0.0), 1.0)  # Ensure PDR is between 0 and 1
        
        # Determine if packet is received
        is_received = np.random.random() < pdr
        
        # Calculate delay (simplified model)
        # Delay = Propagation Delay + Transmission Delay + Processing Delay
        prop_speed = 3e8  # Speed of light in m/s
        
        # Get positions
        if source_id in self.vehicles:
            source_pos = self.vehicles[source_id]['position']
        else:
            source_pos = self.rsus[source_id]['position']
            
        if dest_id in self.vehicles:
            dest_pos = self.vehicles[dest_id]['position']
        else:
            dest_pos = self.rsus[dest_id]['position']
            
        # Calculate distance
        dx = source_pos[0] - dest_pos[0]
        dy = source_pos[1] - dest_pos[1]
        dz = source_pos[2] - dest_pos[2]
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Propagation delay
        prop_delay = distance / prop_speed
        
        # Transmission delay (using 6 Mbps as base rate for IEEE 802.11p)
        tx_rate = 6e6  # 6 Mbps
        tx_delay = (packet_size * 8) / tx_rate
        
        # Processing delay (simplified)
        proc_delay = 0.001  # 1 ms
        
        # Total delay
        delay = prop_delay + tx_delay + proc_delay
        
        # Record packet trace
        current_time = self.ns['core'].Simulator.Now().GetSeconds()
        
        trace_entry = pd.DataFrame([{
            'timestamp': current_time,
            'source_id': source_id,
            'dest_id': dest_id,
            'packet_size': packet_size,
            'tx_power': source['tx_power'],
            'snr': snr,
            'delay': delay,
            'received': is_received
        }])
        
        self.packet_traces = pd.concat([self.packet_traces, trace_entry], ignore_index=True)
        
        # Return transmission result
        return {
            'source_id': source_id,
            'dest_id': dest_id,
            'packet_size': packet_size,
            'snr': snr,
            'pdr': pdr,
            'delay': delay,
            'received': is_received
        }
        
    def run_simulation(self, duration: float):
        """
        Run the simulation for a specified duration.
        
        Args:
            duration: Duration to run the simulation in seconds
        """
        if not self._is_running:
            raise RuntimeError("Simulation not created. Call create_simulation() first.")
            
        # Schedule simulation stop
        self.ns['core'].Simulator.Stop(self.ns['core'].Seconds(duration))
        
        # Run simulation
        self.ns['core'].Simulator.Run()
        
        print(f"NS-3 simulation completed for {duration} seconds")
        
    def reset_simulation(self):
        """Reset the simulation."""
        if self._is_running:
            self.ns['core'].Simulator.Destroy()
            self._is_running = False
            
            # Reset data
            self.packet_traces = pd.DataFrame(columns=[
                'timestamp', 'source_id', 'dest_id', 'packet_size', 'tx_power',
                'snr', 'delay', 'received'
            ])
            
            print("NS-3 simulation reset")
            
    def export_packet_traces(self, output_path: str):
        """
        Export packet trace data to CSV.
        
        Args:
            output_path: Path to save the CSV file
        """
        self.packet_traces.to_csv(output_path, index=False)
        print(f"Packet traces saved to: {output_path}")
        
    def __del__(self):
        """Clean up on object destruction."""
        self.reset_simulation() 