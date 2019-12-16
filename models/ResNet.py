"""ResNet model.

ResNet: Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf

Adapted from code contributed by François Chollet.
https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
"""

from keras import layers
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Add
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras_applications.imagenet_utils import _obtain_input_shape
import keras.backend as K


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    The identity block is the block that has no conv layer at shortcut.
    ----------
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    
    # when using TensorFlow, for best performance you should set:
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    ----------
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
        Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        And the shortcut should have strides=(2,2) as well
    """

    filters1, filters2, filters3 = filters

    # when using TensorFlow, for best performance you should set:
    bn_axis = 3
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet(inputShape, nClasses, nLayers=101):
    """
    ResNet model
    ----------
    # Arguments
        inputShape: tuple
            Tuple with the dimensions of the input data (ny, nx, nBands). 
        nClasses: int
            Number of classes.
        nLayers: int
            50 or 101 for ResNet50 and ResNet101, respectively.
            
    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `nLayers`,
            or invalid input shape.
    """
    if nLayers not in {50, 101}:
        raise ValueError('The `nLayers` argument should be either 50 (ResNet50) or 101 (ResNet101).')
                         
    # when using TensorFlow, for best performance you should set:
    bn_axis = 3
    
    # Determine proper input shape
    input_shape = _obtain_input_shape(inputShape, default_size=224, min_size=197,
                                      data_format=K.image_data_format(), require_flatten=True)

    inputs = Input(shape=input_shape)
    
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    if nLayers is 50:
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        for i in range(1,6):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))
    elif nLayers is 101:
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        for i in range(1,23):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    
    outputs = Dense(nClasses, activation='softmax')(x)

    #outputs = GlobalAveragePooling2D()(x)
    #outputs = GlobalMaxPooling2D()(x)

    model = Model(inputs, outputs, name='resnet50')

    return model


if __name__ == '__main__':
    model = resnet((512,512,3), 4, nLayers=50)
    model.summary()
    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='ResNet.png')