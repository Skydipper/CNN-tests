"""EDSR model.

Enhanced Deep Residual Networks for Single Image Super-Resolution
https://arxiv.org/pdf/1707.02921.pdf

"""

from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model
import tensorflow as tf

def create_keras_model(inputShape, nClasses, scale=2, n_filters=64, depth=8, residual_scaling=None):
    
    def residual(inputs, n_filters, scaling):
        x = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(inputs)
        x = Conv2D(n_filters, (3, 3), padding='same')(x)
        if scaling:
            x = Lambda(lambda t: t * scaling)(x)
        x = Add()([inputs, x])

        return x

    def upsample(inputs, n_filters, scale):
        x = Conv2D(n_filters * (scale ** 2), (3, 3), padding='same')(inputs)
        x = Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x)
        return x

    inputs = Input(shape=inputShape)

    x0 = Conv2D(n_filters, (3, 3), padding='same')(inputs)

    x = residual(x0, n_filters, residual_scaling)

    for i in range(depth-1):
        x = residual(x, n_filters, residual_scaling)

    x = Conv2D(n_filters, (3, 3), padding='same')(x)
    x = Add()([x, x0])

    # Upsampling for super-resolution
    if scale == 2:
        x = upsample(x, n_filters, scale)
    elif scale == 3:
        x = upsample(x, n_filters, scale)
    elif scale == 4:
        x = upsample(x, n_filters, (scale-2))
        x = upsample(x, n_filters, (scale-2))

    outputs = Conv2D(nClasses, (3, 3), padding='same')(x)

    model = Model(inputs, outputs, name="edsr")

    return model

if __name__ == '__main__':
    model = create_keras_model((256,256,3), 6, 3)
    model.summary()