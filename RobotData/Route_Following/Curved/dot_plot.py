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
    
    return avg_x, avg_y

def point2segments(point, segment_points):
    """
    Computes the shortest distance between a point and a line segment.
    """
    if segment_points.shape[0] < 2:
        return np.linalg.norm(point - segment_points[0])
    
    segment_diff = segment_points[1:] - segment_points[:-1]
    segment_length = np.linalg.norm(segment_diff, axis=1)
    segment_diff = (segment_diff.T / segment_length).T
    
    to_point = point - segment_points[:-1]
    proj_on_segment = np.sum(to_point * segment_diff, axis=1)
    
    distance = np.zeros_like(proj_on_segment)
    distance[proj_on_segment < 0] = np.linalg.norm(to_point[proj_on_segment < 0], axis=1)
    distance[proj_on_segment > segment_length] = np.linalg.norm(to_point[proj_on_segment > segment_length] - segment_diff[proj_on_segment > segment_length], axis=1)
    
    on_segment_idx = np.logical_and(proj_on_segment >= 0, proj_on_segment <= segment_length)
    if np.any(on_segment_idx):
        point_on_segment = segment_points[:-1] + (segment_diff.T * proj_on_segment).T
        distance[on_segment_idx] = np.linalg.norm(to_point[on_segment_idx] - point_on_segment[on_segment_idx], axis=1)
        
    return np.min(distance)

def segpath2path(segment_points, path_nodes):
    segment_length = np.linalg.norm(segment_points[1:] - segment_points[:-1], axis=1)
    segment_length = np.insert(segment_length, [0, segment_length.size], 0)
    segpath_totallen = np.sum(segment_length)
    weighted_dist = 0
    for pidx, p in enumerate(segment_points):
        p2path_dist = point2segments(p, path_nodes)
        p_weight = segment_length[pidx] + segment_length[pidx + 1]
        weighted_dist += p2path_dist * p_weight
    return weighted_dist / segpath_totallen / 2

# Paths to data
folder_paths = [
    "/Users/hakosaki/Desktop/DataProcessing/right_day1_afternoon_light/left/successed",
    "/Users/hakosaki/Desktop/DataProcessing/right_day1_afternoon_light/right/processed",
    "/Users/hakosaki/Desktop/DataProcessing/right_day3_afternoon_nowheel/successed",
    "/Users/hakosaki/Desktop/DataProcessing/right_day2_afternoon/successed"
]

baseline_csv = "/Users/hakosaki/Desktop/DataProcessing/right_day2_afternoon/baseline.csv"
baseline_df = pd.read_csv(baseline_csv)
baseline_x = baseline_df['position_x'].tolist()
baseline_y = baseline_df['position_y'].tolist()

distances = []
similarities = []

for folder_path in folder_paths:
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    avg_x, avg_y = compute_avg_path_with_range_interpolation(csv_files, folder_path)
    
    # Compute distance from the last point of avg_path to the last point of baseline
    distance = np.linalg.norm(np.array([avg_x[-1], avg_y[-1]]) - np.array([baseline_x[-1], baseline_y[-1]]))
    distances.append(distance)
    
    # Compute similarity
    similarity = segpath2path(np.array(list(zip(avg_x, avg_y))), np.array(list(zip(baseline_x, baseline_y))))
    similarities.append(similarity)

# Plotting
plt.figure(figsize=(10, 6))
colors = ['g', 'c', 'm', 'y']
labels = ['Experiment 1', 'Experiment 2', 'Experiment 3', 'Experiment 4']

for i, (distance, similarity) in enumerate(zip(distances, similarities)):
    plt.scatter(distance, similarity, c=colors[i], label=labels[i], s=100)

plt.title('Comparison of Average Paths to Baseline')
plt.xlabel('Distance to Baseline Endpoint (m)')
plt.ylabel('Similarity to Baseline Path')
plt.legend()
plt.grid(True)
plt.show()
