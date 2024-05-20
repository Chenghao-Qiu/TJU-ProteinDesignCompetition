import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据集
data_file_path = 'data/GFP_data_with_full_sequences.csv'
data = pd.read_csv(data_file_path)

# 划分数据集为训练集和测试集，比例为80%训练集和20%测试集
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# 保存训练集和测试集为 CSV 文件
train_set.to_csv('data/train.csv', index=False)
test_set.to_csv('data/test.csv', index=False)

print("训练集和测试集已经被保存为 'data/train.csv' 和 'data/test.csv'")
