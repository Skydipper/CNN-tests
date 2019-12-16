"""Utilities to download and preprocess the MNIST data."""


from tensorflow import keras
import tensorflow as tf
from keras.utils import np_utils
import numpy as np

def load_data():
  """Loads data into preprocessed (train_x, train_y, eval_x, eval_y) dataframes.

  Returns:
    A tuple (train_x, train_y, eval_x, eval_y), where train_x and eval_x are
    numpy arrays with features for training and train_y and eval_y are
    numpy arrays with the corresponding labels.
  """
  # Load image data from MNIST.
  (train_x, train_y),(eval_x, eval_y) = keras.datasets.mnist.load_data()

  # We convert the input data to (60000, 28, 28, 1), float32 and normalize our data values to the range [0, 1].
  train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
  eval_x = eval_x.reshape(eval_x.shape[0], eval_x.shape[1], eval_x.shape[2], 1)

  train_x = train_x.astype('float32')
  eval_x = eval_x.astype('float32')
  train_x /= 255
  eval_x /= 255

  # Preprocess class labels 
  train_y = train_y.astype(np.int32)
  eval_y = eval_y.astype(np.int32)

  train_y = np_utils.to_categorical(train_y, 10)
  eval_y = np_utils.to_categorical(eval_y, 10)

  return train_x, train_y, eval_x, eval_y
