"""Defines a Keras model and input function for training."""

import tensorflow as tf

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten # Keras core layes
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D # Keras CNN layers


def input_fn(features, labels, shuffle, num_epochs, batch_size):
  """Generates an input function to be used for model training.

  Args:
    features: numpy array of features used for training or inference
    labels: numpy array of labels for each example
    shuffle: boolean for whether to shuffle the data or not (set True for
      training, False for evaluation)
    num_epochs: number of epochs to provide the data for
    batch_size: batch size for training

  Returns:
    A tf.data.Dataset that can provide data to the Keras model for training or
      evaluation
  """
  if labels is None:
    inputs = features
  else:
    inputs = (features, labels)
  dataset = tf.data.Dataset.from_tensor_slices(inputs)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=len(features))

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset

def create_keras_model(inputShape, learning_rate):
  # Model input
  inputs = Input(shape=inputShape)
   
  # Convolutional layers    
  x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
  x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
  x = MaxPooling2D(pool_size=(2,2))(x)
  x = Dropout(0.25)(x)
  
  # Fully connected Dense layers
  x = Flatten()(x)
  x = Dense(128, activation='relu')(x)
  x = Dropout(0.5)(x)
  
  outputs = Dense(10, activation='softmax')(x)
  
  model = Model(inputs=inputs, outputs=outputs, name='myModel')
  
  # Custom Optimizer:
  # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
  optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)

  # Compile Keras model
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer, metrics=['accuracy'])
      
  return model  

