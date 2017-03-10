"""
Gaussian Process callbacks for Keras.
"""
from __future__ import absolute_import, division

import warnings

import numpy as np

from timeit import default_timer

from keras.callbacks import Callback

from .metrics import mean_squared_error as MSE
from .utils.execute import elapsed_timer


class UpdateGP(Callback):
    '''Performs GP updates at the beginning of each epoch during training.
    '''
    def __init__(self, ins, val_ins=None, batch_size=128, gp_n_iter=1):
        self.training_data = ins
        self.validation_data = val_ins
        self.batch_size = batch_size
        self.gp_n_iter = gp_n_iter

    def set_params(self, params):
        self.params = params
        if self.validation_data is not None:
            self.params['metrics'] += ['val_' + m for m in params['metrics']]

    def on_epoch_begin(self, epoch, logs={}):
        X, Y = self.training_data

        # Do forward pass
        H = self.model.transform(X, self.batch_size)

        # Update GPs
        gp_update_elapsed = []
        for gp, h, y in zip(self.model.gp_output_layers, H, Y):
            # Update GP data (and grid if necessary)
            gp.backend.update_data('tr', h, y)
            if gp.update_grid and (epoch % gp.update_grid == 0):
                gp.backend.update_grid('tr')

            # Train GP & get derivatives
            with elapsed_timer() as elapsed:
                gp.hyp = gp.backend.train(self.gp_n_iter)
                gp.dlik_dh = gp.backend.get_dlik_dx('tr')
            gp_update_elapsed.append(elapsed())

            # Compute MSE and NLML
            gp.mse = MSE(y, gp.backend.predict(h))
            gp.nlml = gp.backend.evaluate('tr')
        logs['gp_update_elapsed'] = np.mean(gp_update_elapsed)

    def on_batch_begin(self, batch, logs={}):
        # Update the batch ids for the current batch
        for gp in self.model.gp_output_layers:
            gp.batch_sz = int(logs['size'])
            pad_size = self.batch_size - logs['size']
            gp.batch_ids = np.pad(logs['ids'], (0, pad_size), 'constant')

    def on_epoch_end(self, epoch, logs={}):
        # Do validation if necessary
        if self.validation_data is not None:
            X_val, Y_val = self.validation_data
            H_val = self.model.transform(X_val, self.batch_size)
            for gp, h, y in zip(self.model.gp_output_layers, H_val, Y_val):
                logs['val_nlml'] = gp.backend.evaluate('val', h, y)
                logs['val_mse'] = MSE(y, gp.backend.predict(h))


class Timer(Callback):
    '''Simply records time for each batch and epoch.
    '''
    def on_epoch_begin(self, epoch, logs={}):
        self._epoch_start = default_timer()

    def on_batch_begin(self, batch, logs={}):
        self._batch_start = default_timer()
        self._batch_elapsed = []

    def on_batch_end(self, batch, logs={}):
        batch_elapsed = default_timer() - self._batch_start
        self._batch_elapsed.append(batch_elapsed)

    def on_epoch_end(self, epoch, logs={}):
        logs['epoch_elapsed'] = default_timer() - self._epoch_start
        logs['batch_elapsed_avg'] = np.mean(self._batch_elapsed)
