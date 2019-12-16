from keras.models import model_from_json
from keras.models import load_model
from keras.optimizers import Adam 
from keras.callbacks import Callback, ModelCheckpoint
import h5py
import json
import argparse
import numpy as np
from models import SegNet, DeepVel, PSPNet, DeepLabv3plus

class LossHistory(Callback):
    def __init__(self, root_out, losses):
        self.root_out = root_out        
        self.losses = losses

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs)
        with open("{0}_loss.json".format(self.root_out), 'w') as f:
            json.dump(self.losses, f)

    def finalize(self):
        pass
        
class train(object):
    
    def __init__(self, root_in, root_out, nClasses, model_name, option):
        """
        Class used to train models
        Parameters
        ----------
        root_in : string
            Path of the input files.
        root_out : string
            Path of the output files.
        nClasses : int
            Number of classes.
        model_name : string
            Model name.
        option : string
            Indicates what needs to be done (start or continue).
        """
        
        self.root_in = root_in
        self.root_out = root_out
        self.nClasses = nClasses
        self.option = option
        self.model_name = model_name
        
        self.batch_size = 16
        
        self.models = {'segnet': SegNet.segnet, 'deepvel': DeepVel.deepvel, 
                       'pspnet': PSPNet.pspnet, 'deeplabv3plus': DeepLabv3plus.deeplabv3plus}
        
        self.input_x_train = self.root_in + "x_train.hdf5"
        self.input_y_train = self.root_in + "y_train.hdf5"

        self.input_x_validation = self.root_in + "x_validation.hdf5"
        self.input_y_validation = self.root_in + "y_validation.hdf5"
        
        tmp = np.load(self.root_in + 'normalization_values.npz')
        self.min_v, self.max_v = tmp['arr_0'], tmp['arr_1']

        f = h5py.File(self.input_x_train, 'r')
        self.n_train_orig, self.ny, self.nx, self.nBands = f.get("x_train").shape        
        f.close()

        f = h5py.File(self.input_y_validation, 'r')
        self.n_validation_orig, _, _, _ = f.get("y_validation").shape        
        f.close()
        
        self.batchs_per_epoch_training = int(self.n_train_orig / self.batch_size)
        self.batchs_per_epoch_validation = int(self.n_validation_orig / self.batch_size)

        self.n_training = self.batchs_per_epoch_training * self.batch_size
        self.n_validation = self.batchs_per_epoch_validation * self.batch_size

        print("Original training set size: {0}".format(self.n_train_orig))
        print("   - Final training set size: {0}".format(self.n_training))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_training))

        print("Original validation set size: {0}".format(self.n_validation_orig))
        print("   - Final validation set size: {0}".format(self.n_validation))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.batchs_per_epoch_validation))
        
        print("Number of Bands: {0}".format(self.nBands))
        print("Number of Classes: {0}".format(self.nClasses))
        
    def training_generator(self):
        f_x = h5py.File(self.input_x_train, 'r')
        x = f_x.get(list(f_x.keys())[0])

        f_y = h5py.File(self.input_y_train, 'r')
        y = f_y.get(list(f_y.keys())[0])

        while True:        
            for i in range(self.batchs_per_epoch_training):

                input_train = x[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('float32')            
                output_train = y[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('uint8')
                
                # Normalize input
                for n in range(len(self.min_v)):
                    input_train[:,:,:,n] = (input_train[:,:,:,n]-self.min_v[n])/(self.max_v[n]-self.min_v[n])
                
                yield input_train, output_train

        f_x.close()
        f_y.close()
        
    def validation_generator(self):
        f_x = h5py.File(self.input_x_validation, 'r')
        x = f_x.get(list(f_x.keys())[0])

        f_y = h5py.File(self.input_y_validation, 'r')
        y = f_y.get(list(f_y.keys())[0])
        
        while True:        
            for i in range(self.batchs_per_epoch_validation):
                
                input_validation = x[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('float32')            
                output_validation = y[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('uint8')
                
                # Normalize input
                for n in range(len(self.min_v)):
                    input_validation[:,:,:,n] = (input_validation[:,:,:,n]-self.min_v[n])/(self.max_v[n]-self.min_v[n])

                yield input_validation, output_validation

        f_x.close()
        f_y.close()

    def define_network(self):
        print("Setting up network...")
    
        model = self.models[self.model_name]

        self.model = model((self.ny, self.nx, self.nBands), self.nClasses)
        
        # Save model
        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root_out+self.model_name), 'w')
        f.write(json_string)
        f.close()

    def read_network(self):
        print("Reading previous network...")
                
        self.model = load_model("{0}_weights.hdf5".format(self.root_out+self.model_name))
        
    def compile_network(self):        
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-4))

    def train_network(self, nEpochs):
        print("Training "+self.model_name+"...")        
        
        # Recover losses from previous run or set and empty one
        if (self.option == 'continue'):
            with open("{0}_loss.json".format(self.root_out+self.model_name), 'r') as f:
                losses = json.load(f)
        else:
            losses = []
  
        # To saves the model weights after each epoch if the validation loss decreased
        self.checkpointer = ModelCheckpoint(filepath="{0}_weights.hdf5".format(self.root_out+self.model_name), verbose=1, save_best_only=True)
        # To save a list of losses over each batch 
        self.history = LossHistory(self.root_out, losses) # saving a list of losses over each batch 
    
        # Train the network
        self.metrics = self.model.fit_generator(self.training_generator(), steps_per_epoch=self.n_training, epochs=nEpochs, 
            callbacks=[self.checkpointer, self.history], validation_data=self.validation_generator(), validation_steps=self.n_validation)
        
        self.history.finalize()

        

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train SegNet')
    parser.add_argument('-i','--in', help='Input files path')
    parser.add_argument('-o','--out', help='Output files path')
    parser.add_argument('-c','--classes', help='Number of classes', default=4)
    parser.add_argument('-e','--epochs', help='Number of epochs', default=10)
    parser.add_argument('-m','--model_name', help='Output files path', default = "")
    parser.add_argument('-a','--action', help='Action', choices=['start', 'continue'], required=True)
    parsed = vars(parser.parse_args())

    root_in = str(parsed['in'])
    root_out = str(parsed['out'])
    nClasses = int(parsed['classes'])
    nEpochs = int(parsed['epochs'])
    model_name = str(parsed['model_name'])
    option = parsed['action']

    out = train(root_in, root_out, nClasses, model_name, option)

    if (option == 'start'):           
        out.define_network()        
        
    if (option == 'continue'):
        out.read_network()
        out.train_network(nEpochs)

    if (option == 'start'):
        out.compile_network()
        out.train_network(nEpochs)


