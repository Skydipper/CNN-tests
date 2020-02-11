"""DeepVel model.

DeepVel: Deep learning for the estimation of horizontal
velocities at the solar surface
https://www.aanda.org/articles/aa/pdf/2017/08/aa30783-17.pdf

"""

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Add
from tensorflow.python.keras.layers.core import Layer, Activation, Reshape
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D


def residual(inputs, filter_size, kernel):
    x = Conv2D(filter_size, kernel, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter_size, kernel, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, inputs])  
    return x

def create_keras_model(inputShape, nClasses, output_activation='softmax'):
    """
    DeepVel model
    ----------
    inputShape : tuple
        Tuple with the dimensions of the input data (ny, nx, nBands). 
    nClasses : int
            Number of classes.
    """

    filter_size = 64
    kernel = (3, 3)        
    n_residual_layers = 5   

    inputs = Input(shape=inputShape, name='image')  
    conv = Conv2D(filter_size, kernel, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)   
    x = residual(conv, filter_size, kernel)
    for i in range(n_residual_layers):
        x = residual(x, filter_size, kernel)    
    x = Conv2D(filter_size, kernel, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Add()([x, conv])    
    outputs = Conv2D(nClasses, (1, 1), activation=output_activation, padding='same', kernel_initializer='he_normal', name= 'output')(x) 
    model = Model(inputs=inputs, outputs=outputs, name='deepvel')

    return model

if __name__ == '__main__':
    model = create_keras_model((256,256,6), 4)
    model.summary()