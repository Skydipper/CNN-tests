"""Sequential model.
"""

from tensorflow.python.keras import Model # Keras model module
from tensorflow.python.keras.layers import Input, Dense, Dropout, Activation 

def create_keras_model(inputShape, nClasses, output_activation='linear'):
    
    inputs = Input(shape=inputShape, name='vector')
 
    x = Dense(32, input_shape=inputShape, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nClasses)(x)
    
    outputs = Activation(output_activation, name= 'output')(x)
        
    model = Model(inputs=inputs, outputs=outputs, name='sequential1')
        
    return model

if __name__ == '__main__':
    model = create_keras_model((1,1,6), 4)
    model.summary()