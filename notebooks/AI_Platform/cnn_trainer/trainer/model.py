"""U-Net model.

U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/pdf/1505.04597.pdf

"""


from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras import models


# A variant of the UNET model.

def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)

    return decoder

def create_keras_model(inputShape, nClasses, output_activation='linear'):
    """
    U-Net model
    ----------
    inputShape : tuple
        Tuple with the dimensions of the input data (ny, nx, nBands). 
    nClasses : int
            Number of classes.
    """
    inputs = Input(shape=inputShape, name= 'image')
    encoder0_pool, encoder0 = encoder_block(inputs, 32) 
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) 
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) 
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) 
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) 
    center = conv_block(encoder4_pool, 1024) 
    decoder4 = decoder_block(center, encoder4, 512) 
    decoder3 = decoder_block(decoder4, encoder3, 256) 
    decoder2 = decoder_block(decoder3, encoder2, 128) 
    decoder1 = decoder_block(decoder2, encoder1, 64) 
    decoder0 = decoder_block(decoder1, encoder0, 32) 
    outputs = layers.Conv2D(nClasses, (1, 1), activation=output_activation)(decoder0)

    model = models.Model(inputs=inputs, outputs=outputs, name='unet')   
    return model

if __name__ == '__main__':
    model = create_keras_model((256,256,6), 4)
    model.summary()