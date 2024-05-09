import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# 定义一个函数来计算每一行数据的均值，同时排除异常值
def calculate_mean(row):
    # 移除异常值，这里简单地使用了第一列的数据作为阈值
    threshold = np.mean(row) - np.std(row)
    filtered_row = [cell for cell in row if cell >= threshold]
    return np.mean(filtered_row)

# 定义一个函数来处理单个CSV文件
def process_csv_file(file_path):
    # 读取CSV文件并解析数据
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    # 提取除第一行和第一列外的数据
    values = np.array([[float(cell) for cell in row[1:]] for row in data[1:]])

    # 计算每一行数据的均值
    means = [calculate_mean(row) for row in values]
    return means

# 指定文件夹路径
folder_path = '/home/data/cyx/autodl-tmp/DiffPose/experiments/deepfluoro/eval_single_view'

# 存储所有文件的均值结果
all_means = []
file_names = []

# 遍历文件夹下的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        means = process_csv_file(file_path)
        all_means.append(means)
        file_names.append(filename[:-4])  # 去掉文件名的.csv扩展名

x = np.arange(0, 1001, 50)

plt.figure(figsize=(10, 6))
for means, filename in zip(all_means, file_names):
    plt.plot(x, means, label=filename)
    print(means)

plt.xlabel('Epoch')
plt.ylabel('Mean Value')
plt.xticks(np.arange(0, 1001, 50))  # 设置 x 轴标签
plt.title('Change in Mean Value over Epochs')
plt.legend()
plt.grid(True)  # 添加网格线
plt.tight_layout()  # 调整布局，避免图像被截断

save_path = '/home/data/cyx/autodl-tmp/DiffPose/experiments/deepfluoro/eval_single_view'
plt.savefig(save_path)
print('ok')