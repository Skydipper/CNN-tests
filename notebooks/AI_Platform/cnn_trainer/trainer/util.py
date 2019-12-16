"""Utilities to download and preprocess the data."""

from . import config
import tensorflow as tf

def parse_function(proto):
    """The parsing function.
    Read a serialized example into the structure defined by features_dict.
    Args:
      example_proto: a serialized Example.
    Returns: 
      A dictionary of tensors, keyed by feature name.
    """
    
    # Define your tfrecord 
    features = config.in_bands + config.out_bands
    
    # Specify the size and shape of patches expected by the model.
    kernel_shape = [config.kernel_size, config.kernel_size]
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
    
    return stacked[:,:,:len(config.in_bands)], stacked[:,:,len(config.in_bands):]

def get_dataset(glob):
    """Get the preprocessed training dataset
    Returns: 
    A tf.data.Dataset of training data.
    """
    glob = tf.compat.v1.io.gfile.glob(glob)
    
    dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
    dataset = dataset.map(parse_function, num_parallel_calls=5)
    
    return dataset


def get_training_dataset():
    """Get the preprocessed training dataset
    Returns: 
    A tf.data.Dataset of training data.
    """
    glob = 'gs://' + config.bucket + '/' + config.folder + '/' + config.base_names[0] + '*'
    dataset = get_dataset(glob)
    dataset = dataset.shuffle(config.shuffle_size).batch(config.batch_size).repeat()
    return dataset

def get_evaluation_dataset():
    """Get the preprocessed evaluation dataset
    Returns: 
      A tf.data.Dataset of evaluation data.
    """
    glob = 'gs://' + config.bucket + '/' + config.folder + '/' + config.base_names[1] + '*'
    dataset = get_dataset(glob)
    dataset = dataset.batch(1).repeat()
    return dataset