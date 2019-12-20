"""WDSR model.

Wide Activation for Efficient and Accurate Image Super-Resolution
https://arxiv.org/pdf/1808.08718.pdf

"""

from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model
from tensorflow_addons.layers import WeightNormalization
import tensorflow as tf

def create_keras_model(inputShape, nClasses, scale=2, n_filters=32, depth=8, residual_expansion=6, residual_scaling=None):

    def residual(inputs, n_filters, expansion, kernel_size, scaling):
        linear = 0.8
        x = WeightNormalization(Conv2D(n_filters * expansion, (1, 1), padding='same', activation='relu'))(inputs)
        x = WeightNormalization(Conv2D(int(n_filters * linear), (1, 1), padding='same'))(x)
        x = WeightNormalization(Conv2D(n_filters, kernel_size, padding='same'))(x)
        if scaling:
            x = Lambda(lambda t: t * scaling)(x)
        x = Add()([inputs, x])
        return x

    inputs = Input(shape=inputShape)

    # main branch
    xm = WeightNormalization(Conv2D(n_filters, (3, 3), padding='same'))(inputs)
    for i in range(depth):
        xm = residual(xm, n_filters, residual_expansion, kernel_size=3, scaling=residual_scaling)
    xm = WeightNormalization(Conv2D(nClasses * scale ** 2, (3, 3), padding='same'))(xm)
    xm = Lambda(lambda x: tf.nn.depth_to_space(x, scale))(xm)

    # skip branch
    xs = WeightNormalization(Conv2D(nClasses * scale ** 2, (5, 5), padding='same'))(inputs)
    xs = Lambda(lambda x: tf.nn.depth_to_space(x, scale))(xs)

    outputs = Add()([xm, xs])

    model = Model(inputs, outputs, name="wdsr")

    return model

if __name__ == '__main__':
    model = create_keras_model((256,256,3), 6, 3)
    model.summary()
