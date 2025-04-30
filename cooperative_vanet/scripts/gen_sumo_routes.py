#!/usr/bin/env python3
"""
Route Generator for SUMO Traffic Simulation
Generates vehicle flows for SUMO simulation based on configuration parameters.
"""

import os
import sys
import random
import argparse
import numpy as np
from typing import Dict, List, Tuple

try:
    # Check if SUMO_HOME is defined
    if 'SUMO_HOME' in os.environ:
        tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools_path)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
        
    from sumolib import checkBinary
    import sumolib
except ImportError:
    sys.exit("Error: SUMO tools not found. Please set SUMO_HOME correctly.")

def create_route_file(
    net_file: str,
    output_file: str,
    num_vehicles: int = 100,
    begin_time: int = 0,
    end_time: int = 43200,  # 12 hours in seconds
    seed: int = 42,
    vehicle_type: str = "passenger",
    max_speed: float = 13.0  # m/s, ~= 47 km/h
) -> None:
    """
    Generate a SUMO route file with random trips.
    
    Args:
        net_file: Path to SUMO network file (.net.xml)
        output_file: Path to output route file (.rou.xml)
        num_vehicles: Total number of vehicles to generate
        begin_time: Start time of simulation
        end_time: End time of simulation
        seed: Random seed for reproducibility
        vehicle_type: Vehicle type (passenger, truck, etc.)
        max_speed: Maximum speed of vehicles in m/s
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Load the network
    net = sumolib.net.readNet(net_file)
    
    # Get all edges that allow the specified vehicle type
    edges = [edge.getID() for edge in net.getEdges() if edge.allows(vehicle_type)]
    
    # If no valid edges found, exit
    if not edges:
        sys.exit(f"Error: No edges found that allow vehicle type '{vehicle_type}'")
    
    # Generate vehicle departure times (Poisson distribution)
    duration = end_time - begin_time
    avg_interval = duration / num_vehicles
    departure_times = []
    
    # Generate non-uniform vehicle distribution as mentioned in the paper
    # First 4 hours: sparse traffic
    # Middle 4 hours: medium traffic 
    # Last 4 hours: dense traffic
    for phase in range(3):
        # Adjust lambda for each phase to create sparse -> dense transition
        if phase == 0:  # Sparse
            lam = 0.5 * num_vehicles / 3 / (duration / 3)
        elif phase == 1:  # Medium
            lam = num_vehicles / 3 / (duration / 3)
        else:  # Dense
            lam = 1.5 * num_vehicles / 3 / (duration / 3)
            
        phase_begin = begin_time + phase * (duration / 3)
        phase_end = begin_time + (phase + 1) * (duration / 3)
        
        # Generate Poisson arrivals for this phase
        t = phase_begin
        while t < phase_end:
            dt = np.random.exponential(1.0 / lam)
            t += dt
            if t < phase_end:
                departure_times.append(t)
    
    # Sort departure times
    departure_times.sort()
    
    # Limit to the requested number of vehicles
    if len(departure_times) > num_vehicles:
        departure_times = departure_times[:num_vehicles]
    
    # Write the routes file
    with open(output_file, 'w') as routes:
        print("""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Vehicle types -->
    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="{}" color="1,1,0" carFollowModel="Krauss"/>
    
    <!-- Vehicle flows -->""".format(max_speed), file=routes)
        
        # Generate individual vehicles with random routes
        for i, depart_time in enumerate(departure_times):
            # Select random source and destination edges
            source_edge = random.choice(edges)
            dest_edge = random.choice(edges)
            
            # Ensure source and destination are different
            while source_edge == dest_edge:
                dest_edge = random.choice(edges)
            
            # Write vehicle
            print(f'    <vehicle id="veh{i}" type="passenger" depart="{depart_time:.2f}" departLane="best" departSpeed="max">', file=routes)
            print(f'        <route edges="{source_edge} {dest_edge}"/>', file=routes)
            print('    </vehicle>', file=routes)
        
        # Close routes file
        print('</routes>', file=routes)
    
    print(f"Generated {len(departure_times)} vehicles in route file: {output_file}")

def main():
    """Parse command line arguments and generate route file."""
    parser = argparse.ArgumentParser(description='Generate random routes for SUMO simulation')
    parser.add_argument('--map', '-m', required=True, help='Path to SUMO network file (.net.xml)')
    parser.add_argument('--flows', '-f', type=int, default=100, help='Number of vehicle flows to generate')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--duration', '-d', type=int, default=43200, help='Simulation duration in seconds')
    parser.add_argument('--max-speed', type=float, default=13.0, help='Maximum vehicle speed in m/s')
    parser.add_argument('--output', '-o', required=True, help='Output route file (.rou.xml)')
    
    args = parser.parse_args()
    
    create_route_file(
        net_file=args.map,
        output_file=args.output,
        num_vehicles=args.flows,
        end_time=args.duration,
        seed=args.seed,
        max_speed=args.max_speed
    )

if __name__ == "__main__":
    main() 