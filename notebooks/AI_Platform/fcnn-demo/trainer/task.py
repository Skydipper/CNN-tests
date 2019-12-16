"""Trains a Keras model"""

import argparse
import os

from . import model
from . import util

import tensorflow as tf
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizers

# Sizes of the training and evaluation datasets.
TRAIN_SIZE = 32000
EVAL_SIZE = 8000

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
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--shuffle_size',
        default=2000,
        type=int,
        help='number of records to be shuffled during each training step, default=2000')
    parser.add_argument(
        '--batch-size',
        default=16,
        type=int,
        help='number of records to read during each training step, default=16')
    parser.add_argument(
        '--learning-rate',
        default=1e-3,
        type=float,
        help='learning rate for gradient descent, default=1e-3')
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
    # Specify inputs (Landsat bands) to the model and the response variable.
    opticalBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    thermalBands = ['B10', 'B11']
    BANDS = opticalBands + thermalBands

    # Create the Keras Model
    keras_model = model.create_keras_model(inputShape = (None, None, len(BANDS)))

    ## Compile Keras model
    #keras_model.compile(loss='mse', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
    # Custom Optimizer:
    # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
    #optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    #optimizer = tf.keras.optimizers.SGD(lr=args.learning_rate)
    #optimizer = tf.train.AdagradOptimizer(learning_rate=args.learning_rate)
    #optimizer = tf.keras.optimizers.RMSprop(lr=args.learning_rate)
    optimizer = tf.keras.optimizers.SGD(lr=args.learning_rate)

    # Compile Keras model
    keras_model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


    # Pass a tfrecord
    training_dataset = util.get_training_dataset(
        shuffle_size = args.shuffle_size,  
        batch_size = args.batch_size)

    # Pass a tfrecord
    validation_dataset = util.get_eval_dataset()

    # Setup Learning Rate decay.
    #lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
    #    lambda epoch: args.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
    #    verbose=True)
#
    # Setup TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir, 'keras_tensorboard'),
        histogram_freq=1)

    # Train model
    keras_model.fit(
        x=training_dataset,
        steps_per_epoch=int(TRAIN_SIZE / args.batch_size),
        epochs=args.num_epochs,
        validation_data=validation_dataset,
        validation_steps=int(EVAL_SIZE / args.batch_size),
        verbose=1,
        callbacks=[tensorboard_cb])

    export_path = tf.contrib.saved_model.save_keras_model(
        keras_model, os.path.join(args.job_dir, 'keras_export'))
    export_path = export_path.decode('utf-8')
    print('Model exported to: ', export_path)


if __name__ == '__main__':
    args = get_args()
    tf.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)