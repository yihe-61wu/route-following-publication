import rosbag
import matplotlib.pyplot as plt

# 路径到你的rosbag文件
bag_path = '/home/hakosaki/Desktop/InsectRouteFollowing/Data/angle/test2.bag'

# 初始化空列表来存储数据
time_stamps = []
orientation_x = []

# 打开rosbag文件
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=['/imu']):
        time_stamps.append(t.to_sec())
        orientation_x.append(msg.orientation.x)

# 使用matplotlib绘制数据
plt.figure()
plt.plot(time_stamps, orientation_x, label='Orientation X')
plt.xlabel('Time (s)')
plt.ylabel('Orientation X Value')
plt.title('IMU Orientation X over Time')
plt.legend()
plt.grid(True)
plt.show()
