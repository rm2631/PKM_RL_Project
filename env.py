# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

import torch
print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
