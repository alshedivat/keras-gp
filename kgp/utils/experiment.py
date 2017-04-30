"""
Utility functions for running experiments, saving results, etc.
"""
import os
import sys
import warnings

import numpy as np

from keras.callbacks import ModelCheckpoint

from kgp.metrics import root_mean_squared_error as RMSE


def train(model, data,
          epochs=100,
          batch_size=128,
          callbacks=None,
          checkpoint=None,
          checkpoint_monitor='val_loss',
          verbose=1,
          **fit_kwargs):
    """Train the model on the data.

    Arguments:
    ----------
        model : Model
            Assumes the model has been already compiled.
        data : dict
        epochs : uint (default: 100)
        batch_size : uint (default: 128)
        callbacks : list (default: None)
        checkpoint : str (default: None)
        verbose : uint (default: 1)

    Returns:
    --------
        history : training history
    """
    X_train, y_train = data['train']
    X_test, y_test = data['test']
    validation_data = data['valid'] if 'valid' in data else None
    callbacks = callbacks or []

    # Make sure the checkpoints directory exists
    if checkpoint is not None:
        if not os.path.isdir('checkpoints/'):
            os.makedirs('checkpoints/')

    # Update list of callbacks
    if checkpoint is not None:
        callbacks += [
            ModelCheckpoint('checkpoints/%s.h5' % checkpoint,
                            monitor=checkpoint_monitor,
                            save_weights_only=True,
                            save_best_only=True)
        ]

    # Train the model
    if verbose:
        sys.stdout.write("Training...\n")
        sys.stdout.flush()

    history = model.fit(X_train, y_train, validation_data=validation_data,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks, verbose=verbose,
                        **fit_kwargs)

    if verbose:
        sys.stdout.write('Done.\n')

    # Test the model
    if checkpoint is not None:
        if os.path.isfile('checkpoints/%s.h5' % checkpoint):
            model.load_weights('checkpoints/%s.h5' % checkpoint)
        else:
            warnings.warn('Checkpoint file was specified, but no models were '
                          'saved by the monitor. Make sure the validation '
                          'dataset is specified and the monitoring channel '
                          'is set correctly.')

    return history

