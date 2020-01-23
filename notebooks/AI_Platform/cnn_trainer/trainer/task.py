"""Trains a Keras model"""

import os
import json

import tensorflow as tf

from . import config
from . import util
from . import model

#with open('training_params.json') as json_file:
#    config = json.loads(json.load(json_file))
          
def train_and_evaluate():
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.
    """

    # Create the Keras Model
    if not config.config.get('output_activation'):
        keras_model = model.create_keras_model(inputShape = (None, None, len(config.config.get('in_bands'))), nClasses = len(config.config.get('out_bands')))
    else:
        keras_model = model.create_keras_model(inputShape = (None, None, len(config.config.get('in_bands'))), nClasses = len(config.config.get('out_bands')), output_activation = config.config.get('output_activation'))

    # Compile Keras model
    optimizer = tf.keras.optimizers.SGD(lr=config.config.get('learning_rate'))
    keras_model.compile(loss=config.config.get('loss'), optimizer=optimizer, metrics=config.config.get('metrics'))


    # Pass a tfrecord
    training_dataset = util.get_training_dataset()
    evaluation_dataset = util.get_evaluation_dataset()

    # Setup TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(os.path.join(config.config.get('job_dir'), 'logs'))

    # Train model
    keras_model.fit(
        x=training_dataset,
        steps_per_epoch=int(config.config.get('train_size') / config.config.get('batch_size')),
        epochs=config.config.get('epochs'),
        validation_data=evaluation_dataset,
        validation_steps=int(config.config.get('eval_size') / config.config.get('batch_size')),
        verbose=1,
        callbacks=[tensorboard_cb])

    tf.contrib.saved_model.save_keras_model(keras_model, os.path.join(config.config.get('job_dir'), 'model'))

if __name__ == '__main__':
    tf.logging.set_verbosity('INFO')
    train_and_evaluate()
