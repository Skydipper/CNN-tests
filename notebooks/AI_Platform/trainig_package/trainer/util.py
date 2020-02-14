"""Utilities to download and preprocess the data."""

import tensorflow as tf
import json
from google.cloud import storage
from google.cloud.storage import blob

class Util():

    def __init__(self, path):
        self.path = path
        
        self.client = storage.Client(project='skydipper-196010')
        self.bucket = self.client.get_bucket('geo-ai')
        self.blob = self.bucket.blob(self.path)
        self.config = json.loads(self.blob.download_as_string(client=self.client).decode('utf-8'))
        
    def parse_function(self, proto):
        """The parsing function.
        Read a serialized example into the structure defined by features_dict.
        Args:
          example_proto: a serialized Example.
        Returns: 
          A dictionary of tensors, keyed by feature name.
        """
        
        # Define your tfrecord 
        features = self.config.get('in_bands') + self.config.get('out_bands')
        
        # Specify the size and shape of patches expected by the model.
        kernel_shape = [self.config.get('kernel_size'), self.config.get('kernel_size')]
        columns = [
          tf.io.FixedLenFeature(shape=kernel_shape, dtype=tf.float32) for k in features
        ]
        features_dict = dict(zip(features, columns))
        
        # Load one example
        parsed_features = tf.io.parse_single_example(proto, features_dict)
    
        # Convert a dictionary of tensors to a tuple of (inputs, outputs)
        inputs_list = [parsed_features.get(key) for key in features]
        stacked = tf.stack(inputs_list, axis=0)
        
        # Convert the tensors into a stack in HWC shape
        stacked = tf.transpose(stacked, [1, 2, 0])
        
        return stacked[:,:,:len(self.config.get('in_bands'))], stacked[:,:,len(self.config.get('in_bands')):]
    
    def get_dataset(self, glob):
        """Get the preprocessed training dataset
        Returns: 
        A tf.data.Dataset of training data.
        """
        glob = tf.compat.v1.io.gfile.glob(glob)
        
        dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
        dataset = dataset.map(self.parse_function, num_parallel_calls=5)
        
        return dataset
    
    
    def get_training_dataset(self):
        """Get the preprocessed training dataset
        Returns: 
        A tf.data.Dataset of training data.
        """
        glob = self.config.get('data_dir') + '/' + self.config.get('base_names')[0] + '*'
        dataset = self.get_dataset(glob)
        dataset = dataset.shuffle(self.config.get('shuffle_size')).batch(self.config.get('batch_size')).repeat()
        return dataset
    
    def get_validation_dataset(self):
        """Get the preprocessed validation dataset
        Returns: 
          A tf.data.Dataset of validation data.
        """
        glob = self.config.get('data_dir') + '/' + self.config.get('base_names')[1] + '*'
        dataset = self.get_dataset(glob)
        dataset = dataset.batch(1).repeat()
        return dataset
