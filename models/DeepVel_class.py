from keras.models import Model
from keras.layers import Input, Add
from keras.layers.core import Layer, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D


class deepvel(object):
    
    def __init__(self, inputShape, nClasses):
        """
        DeepVel model
        ----------
        inputShape : tuple
            Tuple with the dimensions of the input data (ny, nx, nBands). 
        nClasses : int
             Number of classes.
        """
    
        self.inputShape = inputShape
        self.nClasses = nClasses
        
        self.filter_size = 64
        self.kernel = (3, 3)        
        self.n_residual_layers = 5  

    def residual(self,inputs):
        x = Conv2D(self.filter_size, self.kernel, padding='same', kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.filter_size, self.kernel, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Add()([x, inputs])

        return x

    def network(self):
            
        inputs = Input(shape=self.inputShape)

        conv = Conv2D(self.filter_size, self.kernel, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        x = self.residual(conv)
        for i in range(self.n_residual_layers):
            x = self.residual(x)

        x = Conv2D(self.filter_size, self.kernel, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Add()([x, conv])
    
        x = Conv2D(self.nClasses, (1, 1), padding='valid')(x)

        outputs = Activation('softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        
        return model

if __name__ == '__main__':
    model = deepvel((128,128,6), 4).network()
    model.summary()
    from keras.utils import plot_model
    plot_model(model, show_shapes=True , to_file='DeepVel.png')