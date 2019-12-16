from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import Adam 
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint
import h5py
import json
import os
import argparse
import numpy as np
from models.tensorflow import SegNet
from google.cloud import storage
import tensorflow as tf


def parse_function(proto):
    
    # Define your tfrecord 
    bands_input = ['B2', 'B3', 'B4', 'B8', 'ndvi', 'ndwi']
    columns_input = [tf.FixedLenFeature([256,256,1], tf.float32) for i in bands_input]

    bands_output = ['cropland']
    columns_output = [tf.FixedLenFeature([], tf.string) for i in bands_output]

    bands = bands_input + bands_output
    columns = columns_input + columns_output
    features = dict(zip(bands, columns))
    
    # Load one example
    parsed_features = tf.parse_single_example(proto, features)
    
    # Separate the output images from the input images
    label = parsed_features.pop('cropland')
    image = tf.concat([parsed_features[i] for i in bands_input], axis=2)
    
    # Turn your saved image string into an array
    label = tf.decode_raw(label, tf.uint8)
    
    # Normalize
    image = tf.divide(image, 255.0)
    
    # Bring your picture back in shape
    label = tf.reshape(label, [256, 256])
    
    # Create a one hot array for your labels
    label = tf.one_hot(label, 4)
    
    return image, label

def create_dataset(filepath, batchSize, nEpochs, nRecords):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath, compression_type='GZIP')
    
    # This dataset will go on forever
    dataset = dataset.repeat(nEpochs)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(parse_function, num_parallel_calls=8)
    
    # Set the number of datapoints you want to load and shuffle 
    dataset = dataset.shuffle(10 * batchSize)
    
    # Set the batchsize
    dataset = dataset.batch(batchSize)
        
    return dataset.make_one_shot_iterator()
    

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
    
    def __init__(self, root_out, nClasses, model_name, option):
        """
        Class used to train models
        Parameters
        ----------
        root_out : string
            Path of the output files.
        nClasses : int
            Number of classes.
        model_name : string
            Model name.
        option : string
            Indicates what needs to be done (start or continue).
        """
        
        self.root_out = root_out
        self.nClasses = nClasses
        self.option = option
        self.model_name = model_name
        
        self.batch_size = 8
        
        self.models = {'segnet': SegNet.segnet}
        

        client = storage.Client()
        self.bucketname = 'skydipper_materials'
        bucket = client.get_bucket(self.bucketname)
        
        self.folder = 'gee_data_TFRecords/'
        self.file_type = 'tfrecord.gz'
        
        ## Get file list
        filelist = []
        blobs = bucket.list_blobs(prefix=self.folder )
        for blob in blobs:
            filelist.append(blob.name)   

        self.files = [i for i in filelist if self.file_type in i]
        self.files = ['gs://'+self.bucketname+'/'+i for i in self.files]
        
        self.files_train = self.files[:22]
        self.files_val = self.files[22:]
        
        ## Get number of records and patch size
        print("Reading number of records and patch size...")
        self.files_json = [i for i in filelist if 'json' in i]
        
        self.files_json_train = self.files_json[:11]
        self.files_json_val = self.files_json[11:]

        self.nRecords_train = 0
        for file in self.files_json_train:
            blob = bucket.get_blob(file)
            blob.download_to_filename('data.json')
    
            with open('data.json') as f:
                data = json.load(f)
        
            self.nRecords_train += data.get('totalPatches')
            
        self.nRecords_val = 0
        for file in self.files_json_val:
            blob = bucket.get_blob(file)
            blob.download_to_filename('data.json')
    
            with open('data.json') as f:
                data = json.load(f)
        
            self.nRecords_val += data.get('totalPatches')
    
        self.patchSize = data.get('patchDimensions')

        # Remove file
        os.remove('data.json')
        
        [self.ny, self.nx] = self.patchSize

        self.nBands = 6
        
        #self.n_train = int(0.5 * self.nRecords)
        #self.n_val = self.nRecords - self.n_train
        
        self.n_train = self.nRecords_train
        self.n_val = self.nRecords_val
     
        self.nStep_train = int(self.n_train/self.batch_size)
        self.nStep_val = int(self.n_val/self.batch_size)
        
        
        print("Number of training records: {0}".format(self.n_train))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.nStep_train))
        
        print("Number of validation records: {0}".format(self.n_val))
        print("   - Batch size: {0}".format(self.batch_size))
        print("   - Batches per epoch: {0}".format(self.nStep_val))
        
        print("Number of Bands: {0}".format(self.nBands))
        print("Number of Classes: {0}".format(self.nClasses))
        
        
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
        
        # Read dataset
        dataset_train= create_dataset(filepath=self.files_train, batchSize=self.batch_size, 
                                      nEpochs=nEpochs, nRecords=self.n_train)
        
        dataset_val= create_dataset(filepath=self.files_val, batchSize=self.batch_size, 
                                      nEpochs=nEpochs, nRecords=self.n_val)
        
        
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
        self.metrics = self.model.fit(dataset_train, steps_per_epoch=self.nStep_train, epochs=nEpochs, 
                                      validation_data=dataset_val, validation_steps=self.nStep_val, 
                                      callbacks=[self.checkpointer, self.history])

        
        self.history.finalize()

        

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='Train SegNet')
    parser.add_argument('-o','--out', help='Output files path')
    parser.add_argument('-c','--classes', help='Number of classes', default=4)
    parser.add_argument('-e','--epochs', help='Number of epochs', default=10)
    parser.add_argument('-m','--model_name', help='Output files path', default = "")
    parser.add_argument('-a','--action', help='Action', choices=['start', 'continue'], required=True)
    parsed = vars(parser.parse_args())

    root_out = str(parsed['out'])
    nClasses = int(parsed['classes'])
    nEpochs = int(parsed['epochs'])
    model_name = str(parsed['model_name'])
    option = parsed['action']

    out = train(root_out, nClasses, model_name, option)

    if (option == 'start'):
        out.define_network()        
        
    if (option == 'continue'):
        out.read_network()
        out.train_network(nEpochs)

    if (option == 'start'):
        out.compile_network()
        out.train_network(nEpochs)


