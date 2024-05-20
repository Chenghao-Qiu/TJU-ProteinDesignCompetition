import os
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

BENCHMARKS_DIR = './data'
# 您的数据集名称
DATASET_NAME = 'test'

# 连续值输出
OUTPUT_TYPE = OutputType(False, 'numeric')
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, None)  # 连续输出没有唯一标签

# 加载数据集
train_set_file_path = os.path.join('./data', '%s.train.csv' % DATASET_NAME)
train_set = pd.read_csv(train_set_file_path)
train_set, valid_set = train_test_split(train_set, test_size = 0.1, random_state = 0)

test_set_file_path = os.path.join('./data', '%s.test.csv' % DATASET_NAME)
test_set = pd.read_csv(test_set_file_path)

print(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')

# 加载预训练模型并在加载的数据集上微调
pretrained_model_generator, input_encoder = load_pretrained_model()
model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, dropout_rate = 0.5)

# 编码验证集序列
X_val_encoded = input_encoder.encode_X(valid_set['Full Sequence'].values, seq_len=240)

# 验证集的亮度分数
y_val = valid_set['Brightness'].values

class PerformanceCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(X_val_encoded, batch_size=32).flatten()
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        print(f'Epoch {epoch+1}: Mean Squared Error (MSE) on validation set: {mse}')
        print(f'Epoch {epoch+1}: Mean Absolute Error (MAE) on validation set: {mae}')

# 训练回调
training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
    keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
]

# 训练回调列表
training_callbacks.append(PerformanceCallback())

# 微调模型
finetuned_model = finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['Full Sequence'], train_set['Brightness'], valid_set['Full Sequence'], valid_set['Brightness'],
         seq_len = 240, batch_size = 32, max_epochs_per_stage = 1, lr = 1e-04, begin_with_frozen_pretrained_layers = True,
         lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = 1, final_seq_len = 240, final_lr = 1e-05, callbacks = training_callbacks)

# 保存微调后的模型
finetuned_model.save('./saved_model')
