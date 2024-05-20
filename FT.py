import os
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

BENCHMARKS_DIR = './data'
DATASET_NAME = 'GFP_data_with_full_sequences'

OUTPUT_TYPE = OutputType(False, 'numeric')
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, None)

train_set_file_path = os.path.join('./data', '%s.train.csv' % DATASET_NAME)
train_set = pd.read_csv(train_set_file_path)
train_set, valid_set = train_test_split(train_set, test_size=0.1, random_state=0)

test_set_file_path = os.path.join('./data', '%s.test.csv' % DATASET_NAME)
test_set = pd.read_csv(test_set_file_path)

print(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')

pretrained_model_generator, input_encoder = load_pretrained_model()
model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, dropout_rate=0.5)

X_val_encoded = input_encoder.encode_X(valid_set['Full Sequence'].values, seq_len=241)
y_val = valid_set['Brightness'].values

class PerformanceCallback(keras.callbacks.Callback):
    def __init__(self, X_val, y_val, log_dir='./logs'):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.log_file = os.path.join(log_dir, 'training_log.txt')
        self.mse_plot_file = os.path.join(log_dir, 'mse_plot.png')
        self.mae_plot_file = os.path.join(log_dir, 'mae_plot.png')
        self.plot_file = os.path.join(log_dir, 'training_plot.png')
        self.mse = []
        self.mae = []
        os.makedirs(log_dir, exist_ok=True)
        self.initialize_log_file()

    def initialize_log_file(self):
        with open(self.log_file, 'w') as f:
            f.write(f'New Training Session\n{"="*20}\n')

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, batch_size=32).flatten()
        mse = mean_squared_error(self.y_val, y_pred)
        mae = mean_absolute_error(self.y_val, y_pred)
        self.mse.append(mse)
        self.mae.append(mae)
        with open(self.log_file, 'a') as f:
            f.write(f'Epoch {epoch + 1}: MSE = {mse}, MAE = {mae}\n')
        self.plot_mse()
        self.plot_mae()
        self.plot_metrics()

    def plot_mse(self):
        plt.figure()
        plt.plot(range(1, len(self.mse) + 1), self.mse, label='MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Mean Squared Error over epochs')
        plt.legend()
        plt.savefig(self.mse_plot_file)
        plt.close()

    def plot_mae(self):
        plt.figure()
        plt.plot(range(1, len(self.mae) + 1), self.mae, label='MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error over epochs')
        plt.legend()
        plt.savefig(self.mae_plot_file)
        plt.close()

    def plot_metrics(self):
        plt.figure()
        plt.plot(range(1, len(self.mse) + 1), self.mse, label='MSE')
        plt.plot(range(1, len(self.mae) + 1), self.mae, label='MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MSE and MAE over epochs')
        plt.legend()
        plt.savefig(self.plot_file)
        plt.close()

training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.25, min_lr=1e-05, verbose=1),
    keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
    PerformanceCallback(X_val_encoded, y_val)
]

finetuned_model = finetune(
    model_generator,
    input_encoder,
    OUTPUT_SPEC,
    train_set['Full Sequence'],
    train_set['Brightness'],
    valid_set['Full Sequence'],
    valid_set['Brightness'],
    seq_len=241,
    batch_size=32,
    max_epochs_per_stage=2,
    lr=1e-04,
    begin_with_frozen_pretrained_layers=True,
    lr_with_frozen_pretrained_layers=1e-02,
    n_final_epochs=1,
    final_seq_len=241,
    final_lr=1e-05,
    callbacks=training_callbacks
)

finetuned_model.save('./saved_model')
