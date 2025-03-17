import pandas as pd
import matplotlib.pyplot as plt
import os

current_folder_path = os.path.dirname(os.path.abspath(__file__))

def visualize_odom_from_csv(baseline_csv_path, test_folder_path):
    # 提取odom数据的函数
    def extract_odom_data(csv_path):
        data = pd.read_csv(csv_path)
        x_data = data["position_x"].values
        y_data = data["position_y"].values
        return x_data, y_data

    plt.figure(figsize=(10, 6))

    # baseline
    x_base, y_base = extract_odom_data(baseline_csv_path)
    plt.plot(x_base, y_base, '-b', linewidth=8, label='Baseline')

    # test
    first_test = True
    for test_file in os.listdir(test_folder_path):
        if test_file.startswith("processed") and test_file.endswith(".csv"):
            x_test, y_test = extract_odom_data(os.path.join(test_folder_path, test_file))
            if first_test:
                plt.plot(x_test, y_test, '-c', linewidth=2, label='Test Route')
                first_test = False
            else:
                plt.plot(x_test, y_test, '-c', linewidth=2, label='_no_legend_')
    plt.xlim(-0.5, 3.5)
    plt.ylim(-1.8, 1.8) 
    plt.title('Odom Path Visualization')
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

baseline_csv_path = '/Users/hakosaki/Desktop/DataProcessing/right_day2_afternoon/baseline.csv'
test_folder_path = current_folder_path  
visualize_odom_from_csv(baseline_csv_path, test_folder_path)
