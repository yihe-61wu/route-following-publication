import pandas as pd
import os

def truncate_test_paths(baseline_csv_path, test_folder_path):
    # Read the baseline csv and extract its final position
    baseline_df = pd.read_csv(baseline_csv_path)
    baseline_end_x = baseline_df.iloc[-1]['position_x']
    baseline_end_y = baseline_df.iloc[-1]['position_y']

    # Go through all test csv files in the test_folder_path
    for test_file in os.listdir(test_folder_path):
        if test_file.startswith("test") and test_file.endswith(".csv"):
            test_csv_path = os.path.join(test_folder_path, test_file)
            test_df = pd.read_csv(test_csv_path)

            # Compute the distance from each point in the test path to the baseline's end point
            test_df['distance_to_end'] = ((test_df['position_x'] - baseline_end_x)**2 + (test_df['position_y'] - baseline_end_y)**2)**0.5

            # Find the index of the closest point
            closest_idx = test_df['distance_to_end'].idxmin()

            # Drop all rows after the closest point
            truncated_df = test_df.iloc[:closest_idx+1]
            truncated_df.drop(columns=['distance_to_end'], inplace=True)

            # Save the truncated dataframe to a new csv
            output_path = os.path.join(test_folder_path, f"processed_{test_file}")
            truncated_df.to_csv(output_path, index=False)

# Provide the paths to the baseline csv and the folder containing all test csv files
baseline_csv_path = '/Users/hakosaki/Desktop/DataProcessing/right_day3_morning/baseline.csv'
test_folder_path = '/Users/hakosaki/Desktop/DataProcessing/right_day3_morning/original_csv'

# Call the function to process the csv files
truncate_test_paths(baseline_csv_path, test_folder_path)
