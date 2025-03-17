import rosbag
import csv
import os

def bag_to_csv(bag_file):
    # 获取文件名，不带扩展名
    base_name = os.path.splitext(bag_file)[0]
    csv_file = base_name + '.csv'

    with rosbag.Bag(bag_file, 'r') as bag:
        with open(csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile)

            # 写入CSV文件的标题
            writer.writerow([
                "timestamp", 
                "position_x", "position_y", "position_z", 
                "orientation_x", "orientation_y", "orientation_z", "orientation_w", 
                "linear_velocity_x", "linear_velocity_y", "linear_velocity_z", 
                "angular_velocity_x", "angular_velocity_y", "angular_velocity_z"
            ])

            for topic, msg, t in bag.read_messages(topics=['/odom']):
                writer.writerow([
                    t.to_sec(), 
                    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, 
                    msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, 
                    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z, 
                    msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z
                ])

    print(f"Converted {bag_file} to {csv_file}")

def process_all_bags_in_directory(directory_path):
    # 列出目录下的所有.bag文件
    bag_files = [f for f in os.listdir(directory_path) if f.endswith('.bag')]

    for bag_file in bag_files:
        full_path = os.path.join(directory_path, bag_file)
        bag_to_csv(full_path)

directory_path = '/home/hakosaki/Desktop/InsectRouteFollowing/Data/Trajectory/route/right_day1_afternoon/light'
process_all_bags_in_directory(directory_path)
