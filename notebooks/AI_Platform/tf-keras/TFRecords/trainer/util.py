"""Utilities to download and preprocess the MNIST data."""


import tensorflow as tf

def parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.  
    features = {'image': tf.FixedLenFeature([28, 28, 1], tf.float32),
                'label': tf.FixedLenFeature([], tf.int64)}
    
    # Load one example
    parsed_features = tf.parse_single_example(proto, features)
    
    # Turn your saved image string into an array
    image = parsed_features['image']
    
    # Normalize
    image = tf.divide(image, 255.0)
    
    # Create a one hot array for your labels
    label = tf.one_hot(parsed_features['label'], 10)
    
    return image, label

def create_dataset(filepath, shuffle_size, num_epochs, batch_size):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath, compression_type='GZIP')
    
    # Maps the preprocessing function. You can set the number of parallel loaders here
    dataset = dataset.map(parse_function, num_parallel_calls=8)
    
    # This dataset will go on forever
    dataset = dataset.repeat()
    
    # Set the number of datapoints you want to load and shuffle 
    dataset = dataset.shuffle(shuffle_size)
    
    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    
    # Set the batchsize
    dataset = dataset.batch(batch_size)
    
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    return iterator