import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data_path = "/home/hakosaki/Desktop/InsectRouteFollowing/Data/fam_cal/familiarity_data.txt"  # 请替换为你的文件路径
data = np.loadtxt(data_path, delimiter=',')

# 分别获取左眼和右眼的数据
fam_left = data[:, 0]
fam_right = data[:, 1]

# 为数据点设置时间轴
time_interval = 0.38  # 时间间隔为0.43秒
times = np.arange(0, len(fam_left) * time_interval, time_interval)

# 绘制fam_left的图形
plt.figure(figsize=(10, 6))
plt.plot(times, fam_left, label='Left View', color='blue', marker='o', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Familiarity')
plt.title('Familiarity for Left View over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制fam_right的图形
plt.figure(figsize=(10, 6))
plt.plot(times, fam_right, label='Right View', color='red', marker='x', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Familiarity')
plt.title('Familiarity for Right View over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
