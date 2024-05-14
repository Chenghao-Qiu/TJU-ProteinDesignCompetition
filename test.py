import tensorflow as tf

# 检查TensorFlow是否能够识别GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 如果上面的命令输出显示有可用的GPU，那么您的CUDA和cuDNN设置应该是正确的
