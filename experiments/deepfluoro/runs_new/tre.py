import os
import pandas as pd

# 指定文件夹路径
folder_path = "/home/data/cyx/autodl-tmp/DiffPose_copy/experiments/deepfluoro/runs_new"

# 初始化一个字典，用于存储每个类别的最后一个fiducial值的和和计数
class_sum = {}
class_count = {}

# 获取文件夹下所有文件
file_list = os.listdir(folder_path)

# 循环遍历每个文件
for file_name in file_list:
    # 确保文件以.csv结尾
    if file_name.endswith('.csv'):
        # 提取文件名中的两位数字作为分类标签
        class_label = file_name.split('_')[0][-2:]
        
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 获取fiducial列的最后一个值
        last_fiducial_value = df["fiducial"].iloc[-1]
        
        # 更新对应类别的sum和count
        if class_label not in class_sum:
            class_sum[class_label] = last_fiducial_value
            class_count[class_label] = 1
        else:
            class_sum[class_label] += last_fiducial_value
            class_count[class_label] += 1

# 计算每个类别的最后一个fiducial值的均值
class_avg = {class_label: class_sum[class_label] / class_count[class_label] for class_label in class_sum}

print("每个类别的最后一个fiducial值的均值:")
for class_label, avg_value in class_avg.items():
    print(f"类别 {class_label}: {avg_value}")
