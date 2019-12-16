"""Defines a Keras model for training."""

import tensorflow as tf

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten # Keras core layes
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D # Keras CNN layers

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