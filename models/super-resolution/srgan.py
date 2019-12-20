"""SRGAN model.

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
https://arxiv.org/pdf/1609.04802.pdf

"""

from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model
import tensorflow as tf

def create_keras_model(inputShape, nClasses, scale=2, n_filters=64, depth=16):

    def residual(inputs, n_filters, momentum=0.8):
        x = Conv2D(n_filters, kernel_size=3, padding='same')(inputs)
        x = BatchNormalization(momentum=momentum)(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(n_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = Add()([inputs, x])
        return x

    def upsample(inputs, n_filters, scale):
        x = Conv2D(n_filters * (scale ** 2), kernel_size=3, padding='same')(inputs)
        x = Lambda(lambda x: tf.nn.depth_to_space(x, scale))(x)
        x = PReLU(shared_axes=[1, 2])(x)
        return x

    inputs = Input(shape=inputShape)

    x = Conv2D(n_filters, kernel_size=9, padding='same')(inputs)
    x0 = PReLU(shared_axes=[1, 2])(x)

    x = residual(x0, n_filters)
    for i in range(depth-1):
        x = residual(x, n_filters)

    x = Conv2D(n_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x0, x])

    # Upsampling for super-resolution
    if scale == 2:
        x = upsample(x, n_filters, scale)
    elif scale == 3:
        x = upsample(x, n_filters, scale)
    elif scale == 4:
        x = upsample(x, n_filters, (scale-2))
        x = upsample(x, n_filters, (scale-2))

    outputs = Conv2D(nClasses, kernel_size=9, padding='same', activation='sigmoid')(x)

    model = Model(inputs, outputs, name="srgan")

    return model

if __name__ == '__main__':
    model = create_keras_model((256,256,3), 6, 3)
    model.summary()
