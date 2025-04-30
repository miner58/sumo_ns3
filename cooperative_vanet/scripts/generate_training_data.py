#!/usr/bin/env python3
"""
Generate synthetic training data for the traffic prediction model.
This script creates time-series data of vehicle density with
realistic patterns for training the LSTM model.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_time_series(
    duration_hours: int = 24,
    time_step_minutes: int = 1,
    base_density: int = 10,
    peak_density: int = 30,
    noise_level: float = 0.1,
    num_days: int = 7
) -> pd.DataFrame:
    """
    Generate a synthetic time series of vehicle density.
    
    Args:
        duration_hours: Duration of a single day in hours
        time_step_minutes: Time step in minutes
        base_density: Base vehicle density (vehicles/minute)
        peak_density: Peak vehicle density during rush hours
        noise_level: Level of random noise (0.0-1.0)
        num_days: Number of days to generate
        
    Returns:
        DataFrame with timestamp and vehicle density
    """
    # Calculate number of time steps
    steps_per_day = int(duration_hours * 60 / time_step_minutes)
    total_steps = steps_per_day * num_days
    
    # Create time index
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i*time_step_minutes) for i in range(total_steps)]
    
    # Initialize density array
    density = np.zeros(total_steps)
    
    for day in range(num_days):
        # Base pattern for each day
        day_start = day * steps_per_day
        day_end = (day + 1) * steps_per_day
        
        # Generate daily pattern with morning and evening rush hours
        t = np.linspace(0, 2*np.pi, steps_per_day)
        
        # Base density
        daily_pattern = np.ones(steps_per_day) * base_density
        
        # Add morning rush hour (around 8 AM)
        morning_peak = peak_density * np.exp(-((t - np.pi/3)**2) / 0.1)
        
        # Add evening rush hour (around 5-6 PM)
        evening_peak = peak_density * np.exp(-((t - 2*np.pi/3)**2) / 0.1)
        
        # Combine patterns
        daily_pattern += morning_peak + evening_peak
        
        # Add random noise
        noise = np.random.normal(0, noise_level * base_density, steps_per_day)
        daily_pattern += noise
        
        # Ensure non-negative values
        daily_pattern = np.maximum(daily_pattern, 0)
        
        # Add weekly patterns (higher on weekdays, lower on weekends)
        if day % 7 >= 5:  # Weekend (Saturday and Sunday)
            daily_pattern *= 0.7
        
        # Add to overall density
        density[day_start:day_end] = daily_pattern
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'vehicles_per_minute': density
    })
    
    # Add derived time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df

def add_trend(df: pd.DataFrame, trend_type: str = 'linear', trend_strength: float = 0.2) -> pd.DataFrame:
    """
    Add a trend to the time series.
    
    Args:
        df: Input DataFrame
        trend_type: Type of trend ('linear', 'quadratic', 'exponential')
        trend_strength: Strength of the trend (0.0-1.0)
        
    Returns:
        DataFrame with trend added
    """
    n = len(df)
    x = np.arange(n)
    
    if trend_type == 'linear':
        trend = x * trend_strength
    elif trend_type == 'quadratic':
        trend = (x**2) * (trend_strength / n)
    elif trend_type == 'exponential':
        trend = np.exp(x * trend_strength / n) - 1
    else:
        raise ValueError(f"Unknown trend type: {trend_type}")
    
    # Normalize trend
    trend = trend * (df['vehicles_per_minute'].mean() * trend_strength)
    
    # Add trend to vehicle density
    df['vehicles_per_minute'] += trend
    
    return df

def add_anomalies(df: pd.DataFrame, anomaly_ratio: float = 0.01, magnitude: float = 2.0) -> pd.DataFrame:
    """
    Add anomalies to the time series.
    
    Args:
        df: Input DataFrame
        anomaly_ratio: Ratio of anomalies to add
        magnitude: Magnitude of anomalies relative to mean
        
    Returns:
        DataFrame with anomalies added
    """
    n = len(df)
    num_anomalies = int(n * anomaly_ratio)
    
    # Randomly select indices for anomalies
    anomaly_indices = np.random.choice(n, num_anomalies, replace=False)
    
    # Calculate anomaly values
    mean = df['vehicles_per_minute'].mean()
    std = df['vehicles_per_minute'].std()
    
    for idx in anomaly_indices:
        # Add or subtract a random magnitude
        if np.random.random() > 0.5:
            df.loc[idx, 'vehicles_per_minute'] += magnitude * mean
        else:
            df.loc[idx, 'vehicles_per_minute'] -= min(magnitude * mean, df.loc[idx, 'vehicles_per_minute'] * 0.8)
    
    return df

def plot_time_series(df: pd.DataFrame, output_file: str = None):
    """
    Plot the generated time series.
    
    Args:
        df: DataFrame with time series data
        output_file: Path to save the plot (if None, display instead)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['vehicles_per_minute'])
    plt.title('Synthetic Vehicle Density Time Series')
    plt.xlabel('Time')
    plt.ylabel('Vehicles per Minute')
    plt.grid(True)
    
    # Add day boundaries
    day_boundaries = df[df['timestamp'].dt.hour == 0]['timestamp']
    for day in day_boundaries:
        plt.axvline(x=day, color='r', linestyle='--', alpha=0.3)
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    """Parse command line arguments and generate training data."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for traffic prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--output", required=True, help="Path to save CSV file")
    parser.add_argument("--plot", help="Path to save plot (optional)")
    parser.add_argument("--days", type=int, default=7, help="Number of days to generate")
    parser.add_argument("--step", type=int, default=1, help="Time step in minutes")
    parser.add_argument("--base", type=float, default=10, help="Base vehicle density")
    parser.add_argument("--peak", type=float, default=30, help="Peak vehicle density")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise level (0.0-1.0)")
    parser.add_argument("--trend", choices=["none", "linear", "quadratic", "exponential"],
                      default="none", help="Type of trend to add")
    parser.add_argument("--trend-strength", type=float, default=0.2, help="Trend strength")
    parser.add_argument("--anomalies", type=float, default=0.01, 
                      help="Ratio of anomalies to add (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate time series
    print("Generating synthetic time series data...")
    df = generate_time_series(
        time_step_minutes=args.step,
        base_density=args.base,
        peak_density=args.peak,
        noise_level=args.noise,
        num_days=args.days
    )
    
    # Add trend if specified
    if args.trend != "none":
        print(f"Adding {args.trend} trend...")
        df = add_trend(df, args.trend, args.trend_strength)
    
    # Add anomalies
    if args.anomalies > 0:
        print(f"Adding anomalies (ratio: {args.anomalies})...")
        df = add_anomalies(df, args.anomalies)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")
    
    # Plot if requested
    if args.plot:
        print("Generating plot...")
        plot_time_series(df, args.plot)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 