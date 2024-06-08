import os
import pandas as pd
from tensorflow import keras
from proteinbert import load_pretrained_model, InputEncoder

# 指定数据集的路径和名称
PREDICTION_DIR = './predict/predict_data'
DATASET_NAME = 'acc'

# 加载数据集
test_set_file_path = os.path.join(PREDICTION_DIR, f'{DATASET_NAME}.predict.csv')
test_set = pd.read_csv(test_set_file_path)

# 加载预训练模型和输入编码器
pretrained_model_generator, input_encoder = load_pretrained_model()

# 加载模型
model = keras.models.load_model('./saved_model')

# 编码测试集序列
X_test_encoded = input_encoder.encode_X(test_set['Full Sequence'].values, seq_len=241)

# 使用模型进行预测
predictions = model.predict(X_test_encoded, batch_size=32).flatten()

# 定义多个阈值来判断预测的准确性
thresholds = [0.025, 0.05, 0.10]

# 对每个阈值计算准确率
for relative_threshold in thresholds:
    accurate_predictions = sum(1 for pred, actual in zip(predictions, test_set['Brightness'].values)
                               if abs(pred - actual) / actual <= relative_threshold)
    accuracy_percentage = (accurate_predictions / len(predictions)) * 100
    print(f'阈值为 {relative_threshold} 下的模型准确率为: {accuracy_percentage:.2f}%')


