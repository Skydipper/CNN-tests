"""Trains a Keras model"""

import os
import json
import argparse

import tensorflow as tf
from google.cloud import storage
from google.cloud.storage import blob

from .util import Util
from . import model

def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--params-file',
        type=str,
        required=True,
        help='GCS location where we have saved the training_params.json file')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args

def train_and_validation(args):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.
    """
    
    # Read training parameters from GCS
    client = storage.Client(project='skydipper-196010')
    bucket = client.get_bucket('geo-ai')
    blob = bucket.blob(args.params_file)
    config = json.loads(blob.download_as_string(client=client).decode('utf-8'))

    # Create the Keras Model
    selected_model = model.select_model(args.params_file)

    if not config.get('output_activation'):
        keras_model = selected_model(inputShape = (None, None, len(config.get('in_bands'))), nClasses = len(config.get('out_bands')))
    else:
        keras_model = selected_model(inputShape = (None, None, len(config.get('in_bands'))), nClasses = len(config.get('out_bands')), output_activation = config.get('output_activation'))

    # Compile Keras model
    optimizer = tf.keras.optimizers.Adam(lr=config.get('learning_rate'))
    keras_model.compile(loss=config.get('loss'), optimizer=optimizer, metrics=config.get('metrics'))


    # Pass a tfrecord
    util = Util(path = args.params_file) 
    training_dataset = util.get_training_dataset()
    validation_dataset = util.get_validation_dataset()

    # Setup TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(os.path.join(config.get('job_dir'), 'logs'), histogram_freq=1)

    # Train model
    keras_model.fit(
        x=training_dataset,
        steps_per_epoch=int(config.get('training_size') / config.get('batch_size')),
        epochs=config.get('epochs'),
        validation_data=validation_dataset,
        validation_steps=int(config.get('validation_size') / config.get('batch_size')),
        verbose=1,
        callbacks=[tensorboard_cb])

    tf.keras.models.save_model(keras_model, os.path.join(config.get('job_dir'), 'model'), save_format="tf")

if __name__ == '__main__':
    args = get_args()
    tf.logging.set_verbosity('INFO')
    train_and_validation(args)
