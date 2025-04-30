#!/usr/bin/env python3
"""
Create SUMO network files for urban grid scenario.
Generates a rectangular grid network with the dimensions specified in the paper.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_sumo_home():
    """Check if SUMO_HOME environment variable is set."""
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME environment variable not set.")
        print("Please set it to your SUMO installation directory.")
        return False
    return True

def create_grid_network(
    output_dir: str,
    prefix: str = "urban",
    grid_size: tuple = (5, 10),  # 5 rows, 10 columns
    grid_length: int = 150,  # meters between intersections
    lanes: int = 2,
    speed_limit: float = 13.0  # m/s (~ 47 km/h)
):
    """
    Create a grid network for SUMO.
    
    Args:
        output_dir: Output directory
        prefix: Prefix for output files
        grid_size: Grid size as (rows, columns)
        grid_length: Length of each grid cell in meters
        lanes: Number of lanes per direction
        speed_limit: Speed limit in m/s
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    node_file = os.path.join(output_dir, f"{prefix}.nod.xml")
    edge_file = os.path.join(output_dir, f"{prefix}.edg.xml")
    net_file = os.path.join(output_dir, f"{prefix}.net.xml")
    
    # Create node file
    with open(node_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">\n')
        
        # Generate grid nodes
        rows, cols = grid_size
        for r in range(rows):
            for c in range(cols):
                node_id = f"n{r}_{c}"
                x = c * grid_length
                y = r * grid_length
                f.write(f'    <node id="{node_id}" x="{x}" y="{y}" type="traffic_light"/>\n')
        
        f.write('</nodes>\n')
    
    # Create edge file
    with open(edge_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">\n')
        
        # Generate horizontal edges
        rows, cols = grid_size
        for r in range(rows):
            for c in range(cols - 1):
                from_node = f"n{r}_{c}"
                to_node = f"n{r}_{c+1}"
                
                # Forward edge
                edge_id = f"e_{from_node}_{to_node}"
                f.write(f'    <edge id="{edge_id}" from="{from_node}" to="{to_node}" numLanes="{lanes}" '
                        f'speed="{speed_limit}" priority="1"/>\n')
                
                # Backward edge
                edge_id = f"e_{to_node}_{from_node}"
                f.write(f'    <edge id="{edge_id}" from="{to_node}" to="{from_node}" numLanes="{lanes}" '
                        f'speed="{speed_limit}" priority="1"/>\n')
        
        # Generate vertical edges
        for r in range(rows - 1):
            for c in range(cols):
                from_node = f"n{r}_{c}"
                to_node = f"n{r+1}_{c}"
                
                # Forward edge
                edge_id = f"e_{from_node}_{to_node}"
                f.write(f'    <edge id="{edge_id}" from="{from_node}" to="{to_node}" numLanes="{lanes}" '
                        f'speed="{speed_limit}" priority="1"/>\n')
                
                # Backward edge
                edge_id = f"e_{to_node}_{from_node}"
                f.write(f'    <edge id="{edge_id}" from="{to_node}" to="{from_node}" numLanes="{lanes}" '
                        f'speed="{speed_limit}" priority="1"/>\n')
        
        f.write('</edges>\n')
    
    # Create network file using SUMO netconvert
    netconvert_cmd = [
        "netconvert",
        "--node-files", node_file,
        "--edge-files", edge_file,
        "--output-file", net_file,
        "--tls.green.time", "20",
        "--tls.yellow.time", "4",
        "--tls.cycle.time", "60"
    ]
    
    print(f"Running: {' '.join(netconvert_cmd)}")
    try:
        subprocess.run(netconvert_cmd, check=True)
        print(f"Network file created: {net_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running netconvert: {e}")
        sys.exit(1)
    
    return net_file

def create_configuration(
    output_dir: str,
    net_file: str,
    route_file: str = None,
    prefix: str = "urban",
    begin_time: int = 0,
    end_time: int = 3600
):
    """
    Create a SUMO configuration file.
    
    Args:
        output_dir: Output directory
        net_file: Path to network file
        route_file: Path to route file (optional)
        prefix: Prefix for output files
        begin_time: Begin time in seconds
        end_time: End time in seconds
    """
    # Config file path
    config_file = os.path.join(output_dir, f"{prefix}.sumocfg")
    
    # Create configuration file
    with open(config_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n')
        
        # Input files
        f.write('    <input>\n')
        f.write(f'        <net-file value="{os.path.basename(net_file)}"/>\n')
        if route_file:
            f.write(f'        <route-files value="{os.path.basename(route_file)}"/>\n')
        f.write('    </input>\n')
        
        # Time settings
        f.write('    <time>\n')
        f.write(f'        <begin value="{begin_time}"/>\n')
        f.write(f'        <end value="{end_time}"/>\n')
        f.write('    </time>\n')
        
        # Output settings
        f.write('    <output>\n')
        f.write(f'        <tripinfo-output value="{prefix}_tripinfo.xml"/>\n')
        f.write(f'        <summary-output value="{prefix}_summary.xml"/>\n')
        f.write('    </output>\n')
        
        # Processing settings
        f.write('    <processing>\n')
        f.write('        <ignore-route-errors value="true"/>\n')
        f.write('        <time-to-teleport value="300"/>\n')
        f.write('    </processing>\n')
        
        # GUI settings
        f.write('    <gui_only>\n')
        f.write('        <gui-settings-file value="gui-settings.cfg"/>\n')
        f.write('    </gui_only>\n')
        
        f.write('</configuration>\n')
    
    print(f"Configuration file created: {config_file}")
    return config_file

def main():
    """Parse command line arguments and create SUMO network files."""
    parser = argparse.ArgumentParser(
        description="Create SUMO network files for urban grid scenario",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--output-dir", default="configs", help="Output directory")
    parser.add_argument("--prefix", default="urban", help="Prefix for output files")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows in grid")
    parser.add_argument("--cols", type=int, default=10, help="Number of columns in grid")
    parser.add_argument("--grid-length", type=int, default=150, help="Length of each grid cell in meters")
    parser.add_argument("--lanes", type=int, default=2, help="Number of lanes per direction")
    parser.add_argument("--speed", type=float, default=13.0, help="Speed limit in m/s")
    
    args = parser.parse_args()
    
    # Check if SUMO_HOME is set
    if not check_sumo_home():
        return 1
    
    # Create grid network
    net_file = create_grid_network(
        output_dir=args.output_dir,
        prefix=args.prefix,
        grid_size=(args.rows, args.cols),
        grid_length=args.grid_length,
        lanes=args.lanes,
        speed_limit=args.speed
    )
    
    # Create SUMO configuration file
    config_file = create_configuration(
        output_dir=args.output_dir,
        net_file=net_file,
        prefix=args.prefix
    )
    
    # Create GUI settings file
    gui_file = os.path.join(args.output_dir, "gui-settings.cfg")
    with open(gui_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<viewsettings>\n')
        f.write('    <scheme name="real world"/>\n')
        f.write('    <delay value="50"/>\n')
        f.write('</viewsettings>\n')
    
    print(f"GUI settings file created: {gui_file}")
    print("\nFiles created successfully. Next steps:")
    print(f"1. Generate routes: ./scripts/gen_sumo_routes.py --map {net_file} --flows 100 --output {args.output_dir}/{args.prefix}.rou.xml")
    print(f"2. Run simulation: sumo-gui -c {config_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 