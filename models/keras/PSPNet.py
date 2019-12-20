"""PSPNet model.

PSPNet: Pyramid Scene Parsing Network
https://arxiv.org/pdf/1612.01105.pdf

Adapted from code contributed by ykamikawa.
https://github.com/ykamikawa/PSPNet
"""

import keras.backend as K
from keras.models import Model
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.engine.topology import Layer
from keras.layers import Input, Reshape, Permute, Dense, Activation, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import ZeroPadding2D, Lambda, Conv2DTranspose
from keras.layers import merge, multiply, Add, concatenate, BatchNormalization


# squeeze and excitation function
def squeeze_excite_block(input, filters, k=1, name=None):
    init = input
    if K.image_data_format() == 'channels_last':
        se_shape = (1, 1, filters * k)
    else:
        se_shape = (filters * k, 1, 1)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense((filters * k) // 16,activation='relu', kernel_initializer='he_normal', use_bias=False, name=name+'_fc1')(se)
    se = Dense(filters * k, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,name=name+'_fc2')(se)
    
    return se

def identity_block(input_tensor, kernel_size, filters, stage, block, dilation_rate=1, multigrid=[1, 2, 1], use_se=True):
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
    
    # conv filters
    filters1, filters2, filters3 = filters

    # compute dataformat
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # layer names
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # dilated rate
    if dilation_rate < 2:
        multigrid = [1, 1, 1]

    # forward
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a', dilation_rate=dilation_rate*multigrid[0])(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b', dilation_rate=dilation_rate*multigrid[1])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3,(1, 1),name=conv_name_base + '2c', dilation_rate=dilation_rate*multigrid[2])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # stage 5 after squeeze and excinttation layer
    if use_se and stage < 5:
        se = squeeze_excite_block(x, filters3, k=1, name=conv_name_base+'_se')
        x = multiply([x, se])
        
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

# residual module
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation_rate=1, multigrid=[1, 2, 1], use_se=True):
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
    
    # conv filters
    filters1, filters2, filters3 = filters

    # compute dataformat
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # dilated rate
    if dilation_rate > 1:
        strides = (1, 1)
    else:
        multigrid = [1, 1, 1]

    # forward
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a', dilation_rate=dilation_rate*multigrid[0])(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b', dilation_rate=dilation_rate*multigrid[1])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', dilation_rate=dilation_rate*multigrid[2])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    # stage after 5 squeeze and excittation
    if use_se and stage < 5:
        se = squeeze_excite_block(x, filters3, k=1, name=conv_name_base+'_se')
        x = multiply([x, se])
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def ResNet(input_tensor, nLayers=101, output_stride=8, multigrid=[1, 1, 1], use_se=True):
    """
    ResNet model
    ----------
    # Arguments
        input_tensor: input tensor
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
        
        
    # compute input shape
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), use_se=use_se)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', use_se=use_se)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', use_se=use_se)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', use_se=use_se)

    if output_stride == 8:
        rate_scale = 2
    elif output_stride == 16:
        rate_scale = 1

    if nLayers is 50:
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', dilation_rate=1*rate_scale, multigrid=multigrid, use_se=use_se)
        for i in range(1,6):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i), dilation_rate=1*rate_scale, multigrid=multigrid, use_se=use_se)
    elif nLayers is 101:
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', dilation_rate=1*rate_scale, multigrid=multigrid, use_se=use_se)
        for i in range(1,23):
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i), dilation_rate=1*rate_scale, multigrid=multigrid, use_se=use_se)

    init_rate = 2
    
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', dilation_rate=init_rate * rate_scale, multigrid=multigrid, use_se=use_se)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dilation_rate=init_rate * rate_scale, multigrid=multigrid, use_se=use_se)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dilation_rate=init_rate * rate_scale, multigrid=multigrid, use_se=use_se)

    return x


def conv(**conv_params):
    # conv params
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault('dilation_rate', (1, 1))
    kernel_initializer = conv_params.setdefault(
            "kernel_initializer",
            "he_normal")
    padding = conv_params.setdefault("padding", "same")

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      dilation_rate=dilation_rate, kernel_initializer=kernel_initializer, activation='linear')(input)
        return conv
    return f

def Interp(x, shape):
    ''' interpolation '''
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [int(new_height), int(new_width)], align_corners=True)
    
    return resized

def interp_block(x, num_filters=512, level=1, input_shape=(512, 512, 3), output_stride=16):
    ''' interpolation block '''
    feature_map_shape = (input_shape[0] / output_stride, input_shape[1] / output_stride)

    # compute dataformat
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    if output_stride == 16:
        scale = 5
    elif output_stride == 8:
        scale = 10

    kernel = (level*scale, level*scale)
    strides = (level*scale, level*scale)
    global_feat = AveragePooling2D(kernel, strides=strides, name='pool_level_%s_%s' % (level, output_stride))(x)
    global_feat = conv(filters=num_filters, kernel_size=(1, 1), padding='same', name='conv_level_%s_%s' % (level, output_stride))(global_feat)
    global_feat = BatchNormalization(axis=bn_axis, name='bn_level_%s_%s' % (level, output_stride))(global_feat)
    global_feat = Lambda(Interp, arguments={'shape': feature_map_shape})(global_feat)

    return global_feat

def pyramid_pooling_module(x, num_filters, input_shape, output_stride, levels):
    ''' pyramid pooling function '''

    # compute data format
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    pyramid_pooling_blocks = [x]
    for level in levels:
        pyramid_pooling_blocks.append(
            interp_block(x, num_filters=num_filters, level=level, input_shape=input_shape,
                         output_stride=output_stride))

    y = concatenate(pyramid_pooling_blocks)
    y = conv(filters=num_filters, kernel_size=(3, 3), padding='same', block='pyramid_out_%s' % output_stride)(y)
    y = BatchNormalization(axis=bn_axis, name='bn_pyramid_out_%s' % output_stride)(y)
    y = Activation('relu')(y)

    return y

def pspnet(inputShape, nClasses, nLayers=101, output_stride=16, levels=[6, 3, 2, 1]):
    """
    PSPNet model
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
    
    # Input shape
    inputs = Input(shape=inputShape)

    # ResNet
    x = ResNet(inputs, nLayers=nLayers, output_stride=output_stride)
    
    # Pyramid Pooling Module
    x = pyramid_pooling_module(x, num_filters=512, input_shape=inputShape, output_stride=output_stride, levels=levels)
    
    # Upsampling
    x = Conv2DTranspose(filters=nClasses, kernel_size=(output_stride*2, output_stride*2), strides=(output_stride, output_stride), 
                          padding='same', kernel_initializer='he_normal', kernel_regularizer=None, use_bias=False, name='upscore_{}'.format('out'))(x)
    
    outputs = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == '__main__':
    model = pspnet(inputShape=(512, 512, 6), nClasses=4, nLayers=50)
    model.summary()
    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='PSPNet.png')