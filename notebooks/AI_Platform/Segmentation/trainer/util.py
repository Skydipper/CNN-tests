"""Utilities to download and preprocess the data."""


import tensorflow as tf
from google.cloud import storage

def parse_function(proto):
    
    # Define your tfrecord again. Remember that you saved your image as a string.
    bands_input = ['vis-blue', 'vis-green', 'vis-red']
    columns_input = [tf.FixedLenFeature([], tf.string) for i in bands_input]
    
    bands_output = ['cropland']
    columns_output = [tf.FixedLenFeature([], tf.string) for i in bands_output]
    
    bands = bands_input + bands_output
    columns = columns_input + columns_output
    features = dict(zip(bands, columns))
    
    # Load one example
    parsed_features = tf.parse_single_example(proto, features)
    
    ## Turn your saved input image string into an array
    # Decode images (turn your saved image string into an array) and reshape them
    for i in bands_input:
        parsed_features[i] = tf.decode_raw(parsed_features[i], tf.uint8)
        parsed_features[i] = tf.reshape(parsed_features[i], [128, 128, 1])
    
    # Merge the input bands into a sigle image
    parsed_features['image'] = tf.concat([parsed_features[i] for i in bands_input], axis=2)
    
    # Normalize image
    parsed_features['image'] = tf.divide(parsed_features['image'], 255)
    
    ## Decode image (turn your saved output image string into an array) 
    parsed_features['label'] = tf.decode_raw(parsed_features['cropland'], tf.uint8)
    #
    ## Reshape image
    parsed_features['label'] = tf.reshape(parsed_features['label'], [128, 128])
    #
    ## Create a one hot array for your labels
    parsed_features['label'] = tf.one_hot(parsed_features['label'], 4)
    
    return parsed_features['image'], parsed_features['label']

def create_dataset(filepath, shuffle_size, num_epochs, batch_size):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath, compression_type='GZIP')
    
    # Maps the preprocessing function. You can set the number of parallel loaders here
    dataset = dataset.map(parse_function, num_parallel_calls=8)
    
    # Set the number of datapoints you want to load and shuffle 
    dataset = dataset.shuffle(shuffle_size)

    # This dataset will go on forever
    dataset = dataset.repeat()
    
    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    #dataset = dataset.repeat(num_epochs)
    
    # Set the batchsize
    dataset = dataset.batch(batch_size)
    
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    return iterator

def get_files(bucketname, privatekey_path, folder, file_type):
    client = storage.Client()#.from_service_account_json(privatekey_path)
    bucket = client.get_bucket(bucketname)

    ## Get file list
    filelist = []
    blobs = bucket.list_blobs(prefix=folder)
    for blob in blobs:
        filelist.append(blob.name) 
    
    files = [i for i in filelist if file_type in i]
    files = ['gs://'+bucketname+'/'+i for i in files]
    
    return files