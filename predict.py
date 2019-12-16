from keras.models import model_from_json
from keras.optimizers import Adam 
from keras.callbacks import Callback, ModelCheckpoint
import h5py
import json
import time
import argparse
import numpy as np
from models import SegNet, DeepVel, PSPNet, DeepLabv3plus
       
class predict(object):
    
    def __init__(self, root_in, output, nClasses, model_name):
        """
        Class used to predict using different models
        Parameters
        ----------
        root_in : string
            Path of the input files.
        output : string
            Filename where the output is saved
        input : int
            Number of classes
        model_name : string
            Model name.
        """
        
        self.root_in = root_in
        self.output = output
        self.nClasses = nClasses
        self.model_name = model_name
        
        self.batch_size = 1
        
        self.models = {'segnet': SegNet.segnet, 'deepvel': DeepVel.deepvel, 
                       'pspnet': PSPNet.pspnet, 'deeplabv3plus': DeepLabv3plus.deeplabv3plus}
        
        self.input = self.root_in + "data_x.hdf5"
        
        tmp = np.load(self.root_in + 'normalization_values.npz')
        self.min_v, self.max_v = tmp['arr_0'], tmp['arr_1']
        
        f = h5py.File(self.input, 'r')
        self.n_frames, self.ny, self.nx, self.nBands = f.get(list(f.keys())[0]).shape        
        f.close()

        print("Image size: {0}x{1}".format(self.ny, self.nx))
        print("Number of images: {0}".format(self.n_frames))
        
    def prediction_generator(self):
        f = h5py.File(self.input, 'r')
        x = f.get(list(f.keys())[0])

        while True:        
            for i in range(self.n_frames):

                input_predict = x[i*self.batch_size:(i+1)*self.batch_size,:,:,:].astype('float32')            
                
                # Normalize input
                for n in range(len(self.min_v)):
                    input_predict[:,:,:,n] = (input_predict[:,:,:,n]-self.min_v[n])/(self.max_v[n]-self.min_v[n])
                
                yield input_predict

        f.close()
        
    def define_network(self):
        print("Setting up network...")
        
        model = self.models[self.model_name]

        self.model = model((self.ny, self.nx, self.nBands), self.nClasses)
        
        self.model.load_weights("{0}_weights.hdf5".format('./networks/'+self.model_name))
        

    def predict(self):
        print("Predicting with "+self.model_name+"...")        
        
        start = time.time()

        output = self.model.predict_generator(self.prediction_generator(), steps=self.n_frames, max_queue_size=1)

        end = time.time()
        print("Prediction took {0:3.2} seconds...".format(end-start))
        
        print("Saving data...")
        f = h5py.File(self.output, 'w')
        f.create_dataset('output', data=output)     
        f.close()   

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('-i','--in', help='Input files path')
    parser.add_argument('-o','--out', help='Output file')
    parser.add_argument('-c','--classes', help='Number of classes')
    parser.add_argument('-m','--model_name', help='Output files path', default = "")
    parsed = vars(parser.parse_args())

    root_in = str(parsed['in'])
    root_out = str(parsed['out'])
    nClasses = int(parsed['classes'])
    model_name = str(parsed['model_name'])
    
    prediction = predict(root_in, root_out, nClasses, model_name)
    prediction.define_network()
    out = prediction.predict()


