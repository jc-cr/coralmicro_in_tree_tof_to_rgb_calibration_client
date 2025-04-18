#!/usr/bin/env python3

import numpy as np
import json
import os
import glob
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

import matplotlib
matplotlib.use('Agg')


plt.rcParams['figure.figsize'] = (12, 8)  # Larger default figure size
plt.rcParams["figure.autolayout"]  = True
plt.rcParams["figure.dpi"] = 300



# Set font with fallbacks to system fonts
plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.size'] = 14  # Base font size
plt.rcParams['axes.titlesize'] = 18  # Title font size
plt.rcParams['axes.labelsize'] = 16  # Axes labels font size
plt.rcParams['xtick.labelsize'] = 14  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 14  # Legend font size
plt.rcParams['figure.titlesize'] = 20  # Figure title size


def load_data_from_directory(base_dir):
    """
    Load data from the directory structure with ground truth and sample files
    """
    data = {}
    
    # Find all directories with samples
    sample_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for sample_dir in sample_dirs:
        dir_path = os.path.join(base_dir, sample_dir)
        
        # Read ground truth value
        gt_file = os.path.join(dir_path, 'gt.txt')
        if not os.path.exists(gt_file):
            print(f"Warning: No ground truth file found in {dir_path}, skipping.")
            continue
            
        with open(gt_file, 'r') as f:
            gt_text = f.read().strip()
            # Extract just the number from something like "404 mm"
            ground_truth = int(gt_text.split()[0])
        
        # Find all sample files
        sample_files = sorted(glob.glob(os.path.join(dir_path, 'sample_*.json')))
        
        if not sample_files:
            print(f"Warning: No sample files found in {dir_path}, skipping.")
            continue
        
        # Store all sensor readings for this ground truth distance
        all_readings = []
        
        for sample_file in sample_files:
            with open(sample_file, 'r') as f:
                sample_data = json.load(f)
                
                # Extract ToF sensor readings
                if isinstance(sample_data, list) and len(sample_data) == 16:
                    readings = sample_data
                elif isinstance(sample_data, dict) and 'tof_data' in sample_data:
                    readings = sample_data['tof_data']
                else:
                    # Try to find any array of 16 integers in the JSON
                    for key, value in sample_data.items():
                        if isinstance(value, list) and len(value) == 16 and all(isinstance(x, (int, float)) for x in value):
                            readings = value
                            break
                    else:
                        print(f"Warning: Could not find ToF data in {sample_file}, skipping.")
                        continue
                
                all_readings.append(readings)
        
        if all_readings:
            # Average the readings from all samples at this distance
            avg_readings = np.mean(all_readings, axis=0).tolist()
            data[ground_truth] = avg_readings
    
    return data

def analyze_tof_sensors(data):
    """
    Analyze ToF sensor data across multiple ground truth distances.
    """
    # Convert data to arrays for easier manipulation
    ground_truths = np.array(list(data.keys()))
    
    # Create a 2D array: rows = different distances, columns = different sensors
    readings = np.array([data[gt] for gt in ground_truths])
    
    # Number of sensors (should be 16 for a 4x4 grid)
    num_sensors = readings.shape[1]
    
    # Calculate errors (measured - ground_truth)
    errors = np.zeros_like(readings)
    for i, gt in enumerate(ground_truths):
        errors[i] = readings[i] - gt
    
    # Calculate error metrics for each sensor
    mean_errors = np.mean(errors, axis=0)  # Average error (bias)
    abs_errors = np.abs(errors)
    mean_abs_errors = np.mean(abs_errors, axis=0)  # Mean absolute error
    rmse = np.sqrt(np.mean(errors**2, axis=0))  # Root mean square error
    
    # Generate weights based on inverse of RMSE (higher weight = more accurate)
    weights_rmse = 1 / (rmse + 1e-10)  # Add small epsilon to avoid division by zero
    weights_rmse = weights_rmse / np.sum(weights_rmse)
    
    # Results dictionary
    results = {
        'mean_errors': mean_errors,
        'mean_abs_errors': mean_abs_errors,
        'rmse': rmse,
        'weights': weights_rmse
    }
    
    return results

def plot_error_grid(error_data, title, output_path, cmap='RdYlGn_r', grid_size=(4, 4)):
    """Plot error data on a 4x4 grid with color mapping"""
    # Reshape to 4x4 grid (assuming sensor indices are in row-major order)
    grid_data = error_data.reshape(grid_size)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid_data, cmap=plt.get_cmap(cmap))
    
    # Add colorbar
    cbar = plt.colorbar(im)
    
    # Add text annotations with values and sensor IDs
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            sensor_id = i * grid_size[1] + j  # Calculate sensor ID (0-15)
            text = ax.text(j, i, f"{sensor_id}\n{grid_data[i, j]:.2f}",
                          ha="center", va="center", color="black")
    
    ax.set_title(title)
    ax.set_xticks(np.arange(grid_size[1]))
    ax.set_yticks(np.arange(grid_size[0]))
    
    # Set grid lines
    ax.set_xticks(np.arange(-.5, grid_size[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_error_vs_distance(data, output_path, grid_size=(4, 4)):
    """Create a large plot showing sensor errors vs distance"""
    # Sort ground truth distances
    ground_truths = np.array(sorted(list(data.keys())))
    readings = np.array([data[gt] for gt in ground_truths])
    num_sensors = readings.shape[1]
    
    # Calculate absolute errors
    errors = np.zeros_like(readings)
    for i, gt in enumerate(ground_truths):
        errors[i] = readings[i] - gt
    abs_errors = np.abs(errors)
    
    # Create a large figure
    plt.figure(figsize=(16, 10))
    
    # Use different line styles and markers to help distinguish sensors
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', '*', 'x', '+']
    
    # Plot each sensor with a unique style combination
    for i in range(num_sensors):
        sensor_label = f"cell {i}"  # Use the sensor ID (0-15)
        line_style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        plt.plot(ground_truths, abs_errors[:, i], 
                 linestyle=line_style, marker=marker,
                 linewidth=2, markersize=8,
                 alpha=0.8, label=sensor_label)
    
    plt.xlabel('Ground Truth Distance (mm)', fontsize=14)
    plt.ylabel('Absolute Error (mm)', fontsize=14)
    plt.title('Cell Absolute Error vs Distance', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_results_to_file(results, output_file):
    """Save the analysis results to a JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    # Add a grid representation of the weights
    weights_grid = results['weights'].reshape(4, 4).tolist()
    serializable_results['weights_grid'] = weights_grid
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze ToF sensor data')
    parser.add_argument('--input', default='saved_samples', help='Directory containing sample data')
    parser.add_argument('--output', default='tof_error_results', help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load data from directory structure
    print(f"Loading data from {args.input}...")
    data = load_data_from_directory(args.input)
    
    if not data:
        print("Error: No valid data found. Please check the directory structure and file formats.")
        exit(1)
    
    print(f"Loaded data for {len(data)} different ground truth distances.")
    
    # Run the analysis
    print("Analyzing sensor data...")
    results = analyze_tof_sensors(data)
    
    # Save results to file
    output_file = os.path.join(args.output, "tof_sensor_weights.json")
    save_results_to_file(results, output_file)
    
    # Save summary to a text file
    summary_file = os.path.join(args.output, "tof_analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"ToF Sensor Analysis Summary\n")
        f.write(f"==========================\n\n")
        f.write(f"Data analyzed from: {args.input}\n")
        f.write(f"Number of distances: {len(data)}\n")
        f.write(f"Distance range: {min(data.keys())}mm to {max(data.keys())}mm\n\n")
        
        f.write("Most accurate cells (by RMSE):\n")
        sorted_indices = np.argsort(results['rmse'])
        for i, idx in enumerate(sorted_indices[:3]):
            f.write(f"  {i+1}. Sensor {idx}: RMSE = {results['rmse'][idx]:.2f}mm\n")
        
        f.write("\nLeast accurate cells (by RMSE):\n")
        for i, idx in enumerate(sorted_indices[-3:]):
            f.write(f"  {i+1}. Sensor {idx}: RMSE = {results['rmse'][idx]:.2f}mm\n")
        
        f.write("\nRecommended weights (based on RMSE):\n")
        weights = results['weights'].reshape(4, 4)
        for row in weights:
            f.write("  " + " ".join([f"{w:.4f}" for w in row]) + "\n")
    
    # Generate key plots
    print("\nGenerating plots...")
    
    # 1. Error grid
    rmse_grid_path = os.path.join(args.output, "rmse_grid.png")
    plot_error_grid(results['rmse'], 'RMSE (mm)', rmse_grid_path, 'YlOrRd')
    
    # 2. Weights grid 
    weights_grid_path = os.path.join(args.output, "sensor_weights_grid.png")
    plot_error_grid(results['weights'], 'Cell Weights', weights_grid_path, 'RdYlGn')
    
    # 3. Error vs distance plot
    error_vs_distance_path = os.path.join(args.output, "sensor_error_vs_distance.png")
    plot_error_vs_distance(data, error_vs_distance_path)
    
    # Print summary
    print("\nMost accurate cell (by RMSE):")
    for i, idx in enumerate(sorted_indices[:3]):
        print(f"  {i+1}. Cell {idx}: RMSE = {results['rmse'][idx]:.2f}mm")
    
    print("\nLeast accurate cell (by RMSE):")
    for i, idx in enumerate(sorted_indices[-3:]):
        print(f"  {i+1}. Cell {idx}: RMSE = {results['rmse'][idx]:.2f}mm")
    
    print(f"\nAnalysis complete! Results saved to {args.output}/")

if __name__ == "__main__":
    main()