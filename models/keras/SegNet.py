"""SegNet model.

SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
https://arxiv.org/pdf/1511.00561.pdf

"""

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Layer, Activation, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D


def segnet(inputShape, nClasses):
    """
    SegNet model
    ----------
    inputShape : tuple
        Tuple with the dimensions of the input data (ny, nx, nBands). 
    nClasses : int
         Number of classes.
    """

    filter_size = 64
    kernel = (3, 3)        
    pad = (1, 1)
    pool_size = (2, 2)
        

    inputs = Input(shape=inputShape)
        
    # Encoder
    x = Conv2D(64, kernel, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
            
    x = Conv2D(128, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
            
    x = Conv2D(256, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
            
    x = Conv2D(512, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
            
            
    # Decoder
    x = Conv2D(512, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=pool_size)(x)
            
    x = Conv2D(256, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=pool_size)(x)
            
    x = Conv2D(128, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=pool_size)(x)
            
    x = Conv2D(64, kernel, padding='same')(x)
    x = BatchNormalization()(x)
            
    x = Conv2D(nClasses, (1, 1), padding='valid')(x)
    
    outputs = Activation('softmax')(x)
        
    model = Model(inputs=inputs, outputs=outputs, name='segnet')
        
    return model

if __name__ == '__main__':
    model = segnet((496,496,6), 4)
    model.summary()
    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='SegNet.png')