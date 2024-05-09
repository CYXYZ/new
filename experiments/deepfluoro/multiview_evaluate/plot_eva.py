# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # 文件夹路径
# folder_path = '/home/data/cyx/autodl-tmp/DiffPose/experiments/deepfluoro/evaluations'

# # 列出文件夹中的所有文件
# files = os.listdir(folder_path)

# # 筛选出CSV文件
# csv_files = [f for f in files if f.endswith('.csv')]

# # 遍历每个CSV文件并保存
# for csv_file in csv_files:

#     print(csv_files)
#     print(csv_file)

#     # 读取CSV文件到DataFrame
#     df = pd.read_csv(os.path.join(folder_path, csv_file))
    
#     print('df',df)
#     # # 遍历DataFrame的每一列，并绘制变化图
#     # for column in df.columns:
#     #     if column != '0':
#     #         plt.plot(df['0'], df[column], label=column)

    
#     # 删除第一行和第一列
#     df = df.iloc[1:, 1:]
    

#     print(df)
    
#     # 去除异常值（例如，可以使用3倍标准差方法）
#     std = df_cumsum.std(axis=1)
#     mean = df_cumsum.mean(axis=1)
#     threshold = 3 * std + mean
#     df_cleaned = df_cumsum[df_cumsum.lt(threshold, axis=0)]
    
#     # 计算平均值
#     mean_values = df_cleaned.mean(axis=1)

#     print(mean_values)
    
#     # 绘制变化图
#     plt.plot(df_data.columns, mean_values, label="Mean Cumulative Sum")
    
    




#     # 添加标题和标签
#     plt.title('Column Changes Over Time')
#     plt.xlabel('0')
#     plt.ylabel('Values')

#     # # 添加图例
#     # plt.legend()
#     # # 构建保存图片的路径
#     # save_path = os.path.join(folder_path, csv_file.split('.')[0] + 'per_epoch.png')

#     # # 保存图表
#     # plt.savefig(save_path)

#     # 清除当前图形以准备绘制下一个图表
#     plt.clf()

# print("Plots saved to folder:", folder_path)


# import csv
# import numpy as np
# import matplotlib.pyplot as plt

# # 读取CSV文件并解析数据
# data = []
# with open('/home/data/cyx/autodl-tmp/DiffPose/experiments/deepfluoro/evaluations/subject1.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         data.append(row)

# # 提取除第一行和第一列外的数据
# values = np.array([[float(cell) for cell in row[1:]] for row in data[1:]])

# # 定义一个函数来计算每一行数据的均值，同时排除异常值
# def calculate_mean(row):
#     # 移除异常值，这里简单地使用了第一列的数据作为阈值
#     threshold = np.mean(row) - np.std(row)
#     filtered_row = [cell for cell in row if cell >= threshold]
#     return np.mean(filtered_row)

# # 计算每一行数据的均值
# means = [calculate_mean(row) for row in values]
# print(means)

# # 绘制均值的变化曲线
# plt.plot(means)
# plt.xlabel('Epoch')
# plt.ylabel('Mean Value')
# plt.title('Change in Mean Value over Epochs')
# plt.show()


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
folder_path = '/home/data/cyx/autodl-tmp/DiffPose_copy/experiments/deepfluoro/multiview_evaluate'

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

save_path = '/home/data/cyx/autodl-tmp/DiffPose_copy/experiments/deepfluoro/multiview_evaluate/mean_value_plot.png'
plt.savefig(save_path)
print('ok')