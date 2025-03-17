import rosbag
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_odom_from_rosbag(baseline_bag_path, test_folder_path):
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

    x_base, y_base = extract_odom_data(baseline_bag_path)
  
    plt.plot(x_base, y_base, '-b', linewidth=8, label='Baseline Path')  

    x_base, y_base = extract_odom_data(manual_bag_path)

    plt.plot(x_base, y_base, '--r', linewidth=8, label='Manual Path') 


    for test_file in os.listdir(test_folder_path):
        if test_file.startswith("test") and test_file.endswith(".bag"):
            x_test, y_test = extract_odom_data(os.path.join(test_folder_path, test_file))
            plt.plot(x_test, y_test, '--c', linewidth=2, label=f'{test_file} Path')  

    plt.axis('equal')
    plt.title('Odom Path Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

manual_bag_path = '/home/hakosaki/Desktop/InsectRouteFollowing/Data/Trajectory/straight/rightwheel/manual.bag'
baseline_bag_path = '/home/hakosaki/Desktop/InsectRouteFollowing/Data/Trajectory/straight/baseline.bag'
test_folder_path = '/home/hakosaki/Desktop/InsectRouteFollowing/Data/Trajectory/straight/rightwheel/'  
visualize_odom_from_rosbag(baseline_bag_path, test_folder_path)