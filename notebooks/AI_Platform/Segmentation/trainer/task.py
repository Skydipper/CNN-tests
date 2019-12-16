"""Trains a Keras model to predict handwritten numbers from MNIST data."""

import argparse
import os

from . import model
from . import util
from .models import SegNet, DeepVel, DeepLabv3plus, ResNet

import tensorflow as tf

def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting models')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=4,
        help='number of classes to classify, default=4')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=1e-4,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--model-name',
        default='segnet',
        type=str,
        help='Name of the model that we want to use, default=segnet')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.

    Args:
      args: dictionary of arguments - see get_args() for details
    """
    

    bucketname = 'skydipper_materials'
    privatekey_path = "/Users/ikersanchez/Vizzuality/Keys/Skydipper/skydipper-562ee3e31bb2.json"
    folder = 'gee_data_TFRecords/'
    file_type = 'tfrecord.gz'

    models = {'segnet': SegNet.segnet, 'deepvel': DeepVel.deepvel, 
    'deeplabv3plus': DeepLabv3plus.deeplabv3plus, 'resnet': ResNet.resnet}

    files = util.get_files(bucketname, privatekey_path, folder, file_type)

    nTrain = int(len(files)*0.75)
    nValidation = int(len(files)*0.25)
    training_filepath = files[:nTrain]
    validation_filepath = files[nTrain:nTrain+nValidation]
    
    # Dimensions
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    num_train_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filepath, options=options)) for filepath in training_filepath)
    num_eval_examples = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filepath, options=options)) for filepath in validation_filepath)

    # Create the Keras Model
    model = models[args.model_name]
    keras_model = model(
        inputShape = (128,128,3), nClasses = args.num_classes)

    ## Compile Keras model
    #keras_model.compile(loss='mse', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
    # Custom Optimizer:
    # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
    optimizer = tf.keras.optimizers.RMSprop(lr=args.learning_rate)

    # Compile Keras model
    keras_model.compile(loss='categorical_crossentropy',
                optimizer=optimizer, metrics=['accuracy'])

    # Pass a tfrecord
    training_dataset = util.create_dataset(
        filepath = training_filepath, 
        shuffle_size = num_train_examples, 
        num_epochs = args.num_epochs, 
        batch_size = args.batch_size)

    # Pass a tfrecord
    validation_dataset = util.create_dataset(
        filepath = validation_filepath, 
        shuffle_size = num_eval_examples, 
        num_epochs = args.num_epochs, 
        batch_size = args.batch_size)

    # Setup Learning Rate decay.
    lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: args.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
        verbose=True)

     # Setup TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir, 'keras_tensorboard'),
        histogram_freq=1)

    # Train model
    keras_model.fit(
        training_dataset,
        steps_per_epoch=int(num_train_examples / args.batch_size),
        epochs=args.num_epochs,
        validation_data=validation_dataset,
        validation_steps=1,
        verbose=1,
        callbacks=[lr_decay_cb, tensorboard_cb])

    export_path = tf.contrib.saved_model.save_keras_model(
        keras_model, os.path.join(args.job_dir, 'keras_export'))
    export_path = export_path.decode('utf-8')
    print('Model exported to: ', export_path)


    return keras_model

if __name__ == '__main__':
    args = get_args()
    tf.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)