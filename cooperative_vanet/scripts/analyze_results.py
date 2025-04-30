#!/usr/bin/env python3
"""
Analyze simulation results and generate plots and statistics.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

def load_data(results_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load data from CSV files in the results directory.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Dictionary of DataFrames with loaded data
    """
    data = {}
    
    # Load vehicle density data
    vehicle_density_path = os.path.join(results_dir, 'vehicle_density.csv')
    if os.path.exists(vehicle_density_path):
        data['vehicle_density'] = pd.read_csv(vehicle_density_path)
        print(f"Loaded vehicle density data: {len(data['vehicle_density'])} records")
    
    # Load packet traces
    packet_traces_path = os.path.join(results_dir, 'packet_traces.csv')
    if os.path.exists(packet_traces_path):
        data['packet_traces'] = pd.read_csv(packet_traces_path)
        print(f"Loaded packet traces: {len(data['packet_traces'])} records")
    
    # Load vehicle information
    vehicle_info_path = os.path.join(results_dir, 'vehicle_info.csv')
    if os.path.exists(vehicle_info_path):
        data['vehicle_info'] = pd.read_csv(vehicle_info_path)
        print(f"Loaded vehicle information: {len(data['vehicle_info'])} records")
    
    if not data:
        print("No data files found in the specified directory.")
        
    return data

def analyze_vehicle_density(df: pd.DataFrame, output_dir: str):
    """
    Analyze vehicle density over time.
    
    Args:
        df: DataFrame with vehicle density data
        output_dir: Directory to save plots
    """
    if df.empty:
        print("Vehicle density data is empty.")
        return
    
    # Create time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['vehicles_per_minute'])
    plt.title('Vehicle Density Over Time')
    plt.xlabel('Simulation Time (seconds)')
    plt.ylabel('Vehicles per Minute')
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'vehicle_density_time_series.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved vehicle density time series plot to {plot_path}")
    
    # Calculate statistics
    stats = {
        'mean': df['vehicles_per_minute'].mean(),
        'median': df['vehicles_per_minute'].median(),
        'min': df['vehicles_per_minute'].min(),
        'max': df['vehicles_per_minute'].max(),
        'std': df['vehicles_per_minute'].std()
    }
    
    # Save statistics
    stats_df = pd.DataFrame([stats])
    stats_path = os.path.join(output_dir, 'vehicle_density_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved vehicle density statistics to {stats_path}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['vehicles_per_minute'], bins=20, edgecolor='black')
    plt.title('Distribution of Vehicle Density')
    plt.xlabel('Vehicles per Minute')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Save histogram
    hist_path = os.path.join(output_dir, 'vehicle_density_histogram.png')
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved vehicle density histogram to {hist_path}")

def analyze_packet_traces(df: pd.DataFrame, output_dir: str):
    """
    Analyze packet traces.
    
    Args:
        df: DataFrame with packet trace data
        output_dir: Directory to save plots
    """
    if df.empty:
        print("Packet trace data is empty.")
        return
    
    # Calculate packet delivery ratio
    pdr = df['received'].mean() * 100
    print(f"Overall Packet Delivery Ratio: {pdr:.2f}%")
    
    # Group by source-destination pairs
    df['pair'] = df['source_id'] + '-' + df['dest_id']
    pair_stats = df.groupby('pair').agg({
        'received': 'mean',
        'snr': 'mean',
        'delay': 'mean',
        'packet_size': 'mean'
    }).reset_index()
    pair_stats['pdr'] = pair_stats['received'] * 100
    
    # Save pair statistics
    pair_stats_path = os.path.join(output_dir, 'pair_statistics.csv')
    pair_stats.to_csv(pair_stats_path, index=False)
    print(f"Saved pair statistics to {pair_stats_path}")
    
    # Plot SNR vs. PDR
    plt.figure(figsize=(10, 6))
    plt.scatter(pair_stats['snr'], pair_stats['pdr'], alpha=0.6)
    plt.title('SNR vs. Packet Delivery Ratio')
    plt.xlabel('Signal-to-Noise Ratio (dB)')
    plt.ylabel('Packet Delivery Ratio (%)')
    plt.grid(True)
    
    # Add trend line
    if len(pair_stats) > 1:
        z = np.polyfit(pair_stats['snr'], pair_stats['pdr'], 1)
        p = np.poly1d(z)
        plt.plot(pair_stats['snr'], p(pair_stats['snr']), "r--", alpha=0.7)
    
    # Save plot
    snr_pdr_path = os.path.join(output_dir, 'snr_vs_pdr.png')
    plt.savefig(snr_pdr_path)
    plt.close()
    print(f"Saved SNR vs. PDR plot to {snr_pdr_path}")
    
    # Plot delay histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['delay'] * 1000, bins=20, edgecolor='black')  # Convert to ms
    plt.title('Distribution of Packet Delay')
    plt.xlabel('Delay (ms)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Save histogram
    delay_hist_path = os.path.join(output_dir, 'delay_histogram.png')
    plt.savefig(delay_hist_path)
    plt.close()
    print(f"Saved delay histogram to {delay_hist_path}")
    
    # Plot PDR over time
    time_windows = pd.cut(df['timestamp'], bins=20)
    pdr_over_time = df.groupby(time_windows)['received'].mean() * 100
    
    plt.figure(figsize=(12, 6))
    pdr_over_time.plot(kind='line', marker='o')
    plt.title('Packet Delivery Ratio Over Time')
    plt.xlabel('Simulation Time')
    plt.ylabel('Packet Delivery Ratio (%)')
    plt.grid(True)
    
    # Save plot
    pdr_time_path = os.path.join(output_dir, 'pdr_over_time.png')
    plt.savefig(pdr_time_path)
    plt.close()
    print(f"Saved PDR over time plot to {pdr_time_path}")

def analyze_vehicle_info(df: pd.DataFrame, output_dir: str):
    """
    Analyze vehicle information.
    
    Args:
        df: DataFrame with vehicle information
        output_dir: Directory to save plots
    """
    if df.empty:
        print("Vehicle information data is empty.")
        return
    
    # Calculate vehicle speed statistics
    speed_stats = df.groupby('vehicle_id')['speed'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
    
    # Save speed statistics
    speed_stats_path = os.path.join(output_dir, 'vehicle_speed_stats.csv')
    speed_stats.to_csv(speed_stats_path, index=False)
    print(f"Saved vehicle speed statistics to {speed_stats_path}")
    
    # Plot speed distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['speed'], kde=True, bins=20)
    plt.title('Distribution of Vehicle Speeds')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Save plot
    speed_dist_path = os.path.join(output_dir, 'speed_distribution.png')
    plt.savefig(speed_dist_path)
    plt.close()
    print(f"Saved speed distribution plot to {speed_dist_path}")
    
    # Calculate vehicle positions at different time points
    time_points = np.linspace(df['timestamp'].min(), df['timestamp'].max(), 4)
    
    for i, t in enumerate(time_points):
        # Get the closest timestamp
        closest_time = df['timestamp'].iloc[(df['timestamp'] - t).abs().argsort()[0]]
        
        # Get vehicle positions at that time
        positions = df[df['timestamp'] == closest_time][['vehicle_id', 'position_x', 'position_y']]
        
        # Plot vehicle positions
        plt.figure(figsize=(10, 8))
        plt.scatter(positions['position_x'], positions['position_y'], alpha=0.7)
        plt.title(f'Vehicle Positions at Time {closest_time:.1f}s')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.grid(True)
        
        # Save plot
        pos_path = os.path.join(output_dir, f'vehicle_positions_time_{i}.png')
        plt.savefig(pos_path)
        plt.close()
        print(f"Saved vehicle positions plot to {pos_path}")

def generate_summary_report(data: Dict[str, pd.DataFrame], output_dir: str):
    """
    Generate a summary report of the simulation results.
    
    Args:
        data: Dictionary of DataFrames with loaded data
        output_dir: Directory to save the report
    """
    report_lines = ["# Simulation Results Summary", ""]
    
    # Vehicle density summary
    if 'vehicle_density' in data:
        vd = data['vehicle_density']
        report_lines.extend([
            "## Vehicle Density",
            f"- Average: {vd['vehicles_per_minute'].mean():.2f} vehicles/minute",
            f"- Peak: {vd['vehicles_per_minute'].max():.2f} vehicles/minute",
            f"- Simulation duration: {vd['time'].max() - vd['time'].min():.1f} seconds",
            ""
        ])
    
    # Packet trace summary
    if 'packet_traces' in data:
        pt = data['packet_traces']
        report_lines.extend([
            "## Network Performance",
            f"- Total packets: {len(pt)}",
            f"- Packet Delivery Ratio: {pt['received'].mean() * 100:.2f}%",
            f"- Average SNR: {pt['snr'].mean():.2f} dB",
            f"- Average delay: {pt['delay'].mean() * 1000:.2f} ms",
            f"- Average packet size: {pt['packet_size'].mean():.2f} bytes",
            ""
        ])
    
    # Vehicle info summary
    if 'vehicle_info' in data:
        vi = data['vehicle_info']
        report_lines.extend([
            "## Vehicle Statistics",
            f"- Total unique vehicles: {vi['vehicle_id'].nunique()}",
            f"- Average speed: {vi['speed'].mean():.2f} m/s",
            f"- Maximum speed: {vi['speed'].max():.2f} m/s",
            f"- Average acceleration: {vi['acceleration'].mean():.2f} m/s²",
            ""
        ])
    
    # Write report to file
    report_path = os.path.join(output_dir, 'summary_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved summary report to {report_path}")

def main():
    """Parse command line arguments and analyze results."""
    parser = argparse.ArgumentParser(
        description="Analyze simulation results and generate plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--results-dir", required=True, help="Directory containing result files")
    parser.add_argument("--output-dir", help="Directory to save analysis results (default: same as results-dir)")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or args.results_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_data(args.results_dir)
    
    if not data:
        return 1
    
    # Analyze data
    if 'vehicle_density' in data:
        analyze_vehicle_density(data['vehicle_density'], output_dir)
    
    if 'packet_traces' in data:
        analyze_packet_traces(data['packet_traces'], output_dir)
    
    if 'vehicle_info' in data:
        analyze_vehicle_info(data['vehicle_info'], output_dir)
    
    # Generate summary report
    generate_summary_report(data, output_dir)
    
    print("\nAnalysis completed. Check output directory for results.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 