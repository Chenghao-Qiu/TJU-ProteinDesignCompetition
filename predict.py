import os
import pandas as pd
from tensorflow import keras
from proteinbert import load_pretrained_model, InputEncoder

# 指定数据集的路径和名称
PREDICTION_DIR = './predict/predict_data'
PREDICTION_RESULT_DIR = './predict/predict_result'
DATASET_NAME = 'seq'

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

# 输出预测结果
for i, prediction in enumerate(predictions):
    print(f'序列 {i+1} 的预测发光强度: {prediction}')

# 创建一个DataFrame来保存序列和预测的发光强度
results_df = pd.DataFrame({
    'Full Sequence': test_set['Full Sequence'].values,
    'Predicted Brightness': predictions
})

# 将结果保存到CSV文件
results_df.to_csv(os.path.join(PREDICTION_RESULT_DIR, 'predictions.csv'), index=False)

print(f'预测结果已保存到 {os.path.join(PREDICTION_RESULT_DIR, "predictions.csv")}')
