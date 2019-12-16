"""Trains a Keras model"""

from . import config
from . import model
from . import util

import os
import time
import tensorflow as tf


def train_and_evaluate():
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.
    """

    # Create the Keras Model
    keras_model = model.create_keras_model(inputShape = (None, None, len(config.in_bands)), nClasses = len(config.out_bands))

    # Compile Keras model
    keras_model.compile(loss=config.loss, optimizer=config.optimizer, metrics=config.metrics)


    # Pass a tfrecord
    training_dataset = util.get_training_dataset()
    evaluation_dataset = util.get_evaluation_dataset()

    # Setup TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(config.logs_dir)

    # Train model
    keras_model.fit(
        x=training_dataset,
        steps_per_epoch=int(config.train_size / config.batch_size),
        epochs=config.epochs,
        validation_data=evaluation_dataset,
        validation_steps=int(config.eval_size / config.batch_size),
        verbose=1,
        callbacks=[tensorboard_cb])

    tf.contrib.saved_model.save_keras_model(keras_model, os.path.join(config.model_dir, str(int(time.time()))))

if __name__ == '__main__':
    tf.logging.set_verbosity('INFO')
    train_and_evaluate()