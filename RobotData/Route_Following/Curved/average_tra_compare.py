import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_avg_path_with_range_interpolation(csv_files, folder_path):
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
    
    max_x = [max(x) for x in zip(*all_x_data)]
    max_y = [max(y) for y in zip(*all_y_data)]
    
    min_x = [min(x) for x in zip(*all_x_data)]
    min_y = [min(y) for y in zip(*all_y_data)]

    return avg_x, avg_y, max_x, max_y, min_x, min_y

def plot_paths_with_baseline_and_range(avg_x1, avg_y1, max_x1, max_y1, min_x1, min_y1,
                                       avg_x2, avg_y2, max_x2, max_y2, min_x2, min_y2,
                                       avg_x3, avg_y3, max_x3, max_y3, min_x3, min_y3,
                                       baseline_csv):
    # Read the baseline data
    baseline_df = pd.read_csv(baseline_csv)
    baseline_x = baseline_df['position_x'].tolist()
    baseline_y = baseline_df['position_y'].tolist()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.xlim([-0.5, 2.5])
    plt.ylim([-1.5, 1.5])
    
    plt.fill_between(avg_x1, min_y1, max_y1, color='lightgreen', alpha=0.3, label='Range with all lights on')
    plt.plot(avg_x1, avg_y1, '-g', linewidth=3, label='Average Route with all lights on')
    
    plt.fill_between(avg_x2, min_y2, max_y2, color='lightblue', alpha=0.3, label='Range with only left light on')
    plt.plot(avg_x2, avg_y2, '--c', linewidth=3, label='Average Route with only left light on')
    
    plt.fill_between(avg_x3, min_y3, max_y3, color='lightcoral', alpha=0.3, label='Range with only right light on')
    plt.plot(avg_x3, avg_y3, ':r', linewidth=3, label='Average Route with only right light on')
    
    plt.plot(baseline_x, baseline_y, '-b', linewidth=3, label='Baseline Path')
    
    plt.title('Path Visualization with Baseline and Range')
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Paths
folder_path1 = "/Users/hakosaki/Desktop/DataProcessing/right_day2_afternoon/successed"  # Original experiment data
folder_path2 = "/Users/hakosaki/Desktop/DataProcessing/right_day1_afternoon_light/right/processed"  # Additional experiment data
folder_path3 = "/Users/hakosaki/Desktop/DataProcessing/right_day1_afternoon_light/left/successed"  # Third experiment data

csv_files1 = [f for f in os.listdir(folder_path1) if f.endswith('.csv')]
csv_files2 = [f for f in os.listdir(folder_path2) if f.endswith('.csv')]
csv_files3 = [f for f in os.listdir(folder_path3) if f.endswith('.csv')]

baseline_csv = "/Users/hakosaki/Desktop/DataProcessing/right_day1_afternoon_light/left/baseline.csv"

# Compute and plot
avg_x1, avg_y1, max_x1, max_y1, min_x1, min_y1 = compute_avg_path_with_range_interpolation(csv_files1, folder_path1)
avg_x2, avg_y2, max_x2, max_y2, min_x2, min_y2 = compute_avg_path_with_range_interpolation(csv_files2, folder_path2)
avg_x3, avg_y3, max_x3, max_y3, min_x3, min_y3 = compute_avg_path_with_range_interpolation(csv_files3, folder_path3)

plot_paths_with_baseline_and_range(avg_x1, avg_y1, max_x1, max_y1, min_x1, min_y1,
                                   avg_x2, avg_y2, max_x2, max_y2, min_x2, min_y2,
                                   avg_x3, avg_y3, max_x3, max_y3, min_x3, min_y3,
                                   baseline_csv)