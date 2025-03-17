import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_avg_path_interpolation(csv_files, folder_path):
    # Store all x and y data in lists
    all_x_data = []
    all_y_data = []

    # Extract data from all csv files and determine the maximum length
    max_length = 0
    for file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        all_x_data.append(df['position_x'].tolist())
        all_y_data.append(df['position_y'].tolist())
        if len(df) > max_length:
            max_length = len(df)

    # Interpolate data for paths shorter than the maximum length
    for i in range(len(all_x_data)):
        if len(all_x_data[i]) < max_length:
            old_indices = np.linspace(0, 1, len(all_x_data[i]))
            new_indices = np.linspace(0, 1, max_length)
            all_x_data[i] = np.interp(new_indices, old_indices, all_x_data[i])
            all_y_data[i] = np.interp(new_indices, old_indices, all_y_data[i])

    # Compute average for each position
    avg_x = [sum(x) / len(x) for x in zip(*all_x_data)]
    avg_y = [sum(y) / len(y) for y in zip(*all_y_data)]

    return avg_x, avg_y

def plot_average_path_with_baseline(avg_x, avg_y, baseline_csv):
    # Read the baseline data
    baseline_df = pd.read_csv(baseline_csv)
    baseline_x = baseline_df['position_x'].tolist()
    baseline_y = baseline_df['position_y'].tolist()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(avg_x, avg_y, '-g', linewidth=3, label='Average Path')
    plt.plot(baseline_x, baseline_y, '--b', linewidth=3, label='Baseline Path')
    plt.title('Average Odom Path Visualization with Baseline')
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Paths
folder_path = "/Users/hakosaki/Desktop/DataProcessing/right_day3_morning/successed"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
baseline_csv = "/Users/hakosaki/Desktop/DataProcessing/right_day3_morning/baseline.csv"

# Compute and plot
avg_x, avg_y = compute_avg_path_interpolation(csv_files, folder_path)
plot_average_path_with_baseline(avg_x, avg_y, baseline_csv)
