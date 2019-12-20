"""Enhance model.

Enhancing SDO/HMI images using deep learning
https://arxiv.org/pdf/1706.02933.pdf

"""

from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, GaussianNoise, Add, UpSampling2D
from tensorflow.python.keras.layers.core import Layer, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf

# To deactivate warnings: https://stackoverflow.com/questions/54685134/warning-from-tensorflow-when-creating-vgg16
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1))):
    """Pads the 2nd and 3rd dimensions of a 4D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 tuples, padding pattern.
    # Returns
        A padded 4D tensor.
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2

    pattern = [[0, 0],
               list(padding[0]), list(padding[1]),
               [0, 0]]

    return tf.pad(x, pattern, "REFLECT")

class ReflectionPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).
    This layer can add rows and columns or zeros
    at the top, bottom, left and right side of an image tensor.
    # Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to width and height.
            - If tuple of 2 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints:
                interpreted as
                `((top_pad, bottom_pad), (left_pad, right_pad))`
    # Input shape
        4D tensor with shape:
            `(batch, rows, cols, channels)`
    # Output shape
        4D tensor with shape:
            `(batch, padded_rows, padded_cols, channels)`
    """
    
    def __init__(self,
                 padding=(1, 1),
                 **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        
        if input_shape[1] is not None:
            rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
        else:
            rows = None
        if input_shape[2] is not None:
            cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
        else:
            cols = None
        return (input_shape[0],
                rows,
                cols,
                input_shape[3])

    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs,
                                    padding=self.padding)

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def create_keras_model(inputShape, nClasses, scale=2, noise=1e-3, depth=5, activation='relu', n_filters=64, l2_reg=1e-4):
    """
    Deep residual network that keeps the size of the input throughout the whole network
    """

    def residual(inputs, n_filters):
        x = ReflectionPadding2D()(inputs)
        x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = ReflectionPadding2D()(x)
        x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Add()([x, inputs])

        return x

    inputs = Input(shape=inputShape)
    x = GaussianNoise(noise)(inputs)

    x = ReflectionPadding2D()(x)
    x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
    x0 = Activation(activation)(x)

    x = residual(x0, n_filters)

    for i in range(depth-1):
        x = residual(x, n_filters)

    x = ReflectionPadding2D()(x)
    x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])

    # Upsampling for super-resolution
    x = UpSampling2D(size=(scale, scale))(x)

    x = ReflectionPadding2D()(x)
    x = Conv2D(n_filters, (3, 3), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)
    x = Activation(activation)(x)

    outputs = Conv2D(nClasses, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x)

    model = Model(inputs=inputs, outputs=outputs, name='enhance')

    return model

if __name__ == '__main__':
    model = create_keras_model((256,256,3), 6, 3)
    model.summary()