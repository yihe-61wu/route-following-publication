import rosbag
import matplotlib.pyplot as plt
import numpy as np
import os

current_folder_path = os.path.dirname(os.path.abspath(__file__))

def visualize_odom_from_rosbag(baseline_bag_path, failed_folder_path, test_folder_path):
# def visualize_odom_from_rosbag(baseline_bag_path, test_folder_path):

    # 提取odom数据的函数
    def extract_odom_data(bag_path):
        x_data = []
        y_data = []
        bag = rosbag.Bag(bag_path)
        for _, msg, _ in bag.read_messages(topics=['/odom']):
            x_data.append(msg.pose.pose.position.x)
            y_data.append(msg.pose.pose.position.y)
        bag.close()
        return x_data, y_data

    plt.figure(figsize=(10, 6))

    # baseline
    x_base, y_base = extract_odom_data(baseline_bag_path)
    plt.plot(x_base, y_base, '-b', linewidth=8, label='Baseline')

    # failed
    first_failed = True
    for failed_file in os.listdir(failed_folder_path):
        if failed_file.startswith("failed") and failed_file.endswith(".bag"):
            x_failed, y_failed = extract_odom_data(os.path.join(failed_folder_path, failed_file))
            if first_failed:
                plt.plot(x_failed, y_failed, '-.r', linewidth=3, label='Failed Test Route')
                first_failed = False
            else:
                plt.plot(x_failed, y_failed, '-.r', linewidth=3, label='_no_legend_')

    # test
    first_test = True
    for test_file in os.listdir(test_folder_path):
        if test_file.startswith("test") and test_file.endswith(".bag"):
            x_test, y_test = extract_odom_data(os.path.join(test_folder_path, test_file))
            if first_test:
                plt.plot(x_test, y_test, '--c', linewidth=3, label='Successed Test Route')
                first_test = False
            else:
                plt.plot(x_test, y_test, '--c', linewidth=3, label='_no_legend_')

    plt.axis('equal')
    plt.title('Odom Path Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

failed_folder_path = current_folder_path 
baseline_bag_path = '/home/hakosaki/Desktop/InsectRouteFollowing/Data/Trajectory/route/right/baseline.bag'
test_folder_path = current_folder_path  
visualize_odom_from_rosbag(baseline_bag_path, failed_folder_path, test_folder_path)
# visualize_odom_from_rosbag(baseline_bag_path, test_folder_path)

