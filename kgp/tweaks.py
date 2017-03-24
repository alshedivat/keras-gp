"""
Tweaks for KGP and original Keras classes or methods.
To keep the primary KGP code clean, all tweaks are kept in a separate file.
"""
from __future__ import absolute_import

import numpy as np

import keras.callbacks as cbks
from keras.engine.training import _batch_shuffle
from keras.engine.training import _make_batches
from keras.engine.training import _slice_arrays

from .models import Model


def _fit_loop(self, f, ins, out_labels=None, batch_size=32,
              epochs=100, verbose=1, callbacks=None,
              val_f=None, val_ins=None, shuffle=True,
              callback_metrics=None, initial_epoch=0):
    """Abstract fit function for f(ins).
    Assume that f returns a list, labeled by out_labels.

    # Arguments
        f: Keras function returning a list of tensors
        ins: list of tensors to be fed to `f`
        out_labels: list of strings, display names of
            the outputs of `f`
        batch_size: integer batch size
        epochs: number of times to iterate over the data
        verbose: verbosity mode, 0, 1 or 2
        callbacks: list of callbacks to be called during training
        val_f: Keras function to call for validation
        val_ins: list of tensors to be fed to `val_f`
        shuffle: whether to shuffle the data at the beginning of each epoch
        callback_metrics: list of strings, the display names of the metrics
            passed to the callbacks. They should be the
            concatenation of list the display names of the outputs of
             `f` and the list of display names of the outputs of `f_val`.
        initial_epoch: epoch at which to start training
            (useful for resuming a previous training run)

    # Returns
        `History` object.

    [A tweaked version.]
    """
    do_validation = False
    if val_f and val_ins:
        do_validation = True
        if verbose:
            print('Train on %d samples, validate on %d samples' %
                  (ins[0].shape[0], val_ins[0].shape[0]))

    if ins and hasattr(ins[0], 'shape'):
        num_train_samples = ins[0].shape[0]
    else:
        # May happen if we are running `fit` without Numpy input data,
        # i.e. if all inputs to the models are data tensors
        # instead of placeholders.
        # In that case we will run `fit` over a single batch.
        num_train_samples = batch_size
        verbose = 2
    index_array = np.arange(num_train_samples)

    self.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
    if verbose:
        callbacks += [cbks.ProgbarLogger()]
    callbacks = cbks.CallbackList(callbacks)
    out_labels = out_labels or []

    # it's possible to callback a different model than self
    # (used by Sequential models)
    if hasattr(self, 'callback_model') and self.callback_model:
        callback_model = self.callback_model
    else:
        callback_model = self

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'samples': num_train_samples,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics or [],
    })
    callbacks.on_train_begin()
    callback_model.stop_training = False
    # for cbk in callbacks:
    #     cbk.validation_data = val_ins

    for epoch in range(initial_epoch, epochs):
        callbacks.on_epoch_begin(epoch)
        if shuffle == 'batch':
            index_array = _batch_shuffle(index_array, batch_size)
        elif shuffle:
            np.random.shuffle(index_array)

        batches = _make_batches(num_train_samples, batch_size)
        epoch_logs = {}
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            try:
                if isinstance(ins[-1], float):
                    # do not slice the training phase flag
                    ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
                else:
                    ins_batch = _slice_arrays(ins, batch_ids)
            except TypeError:
                raise TypeError('TypeError while preparing batch. '
                                'If using HDF5 input data, '
                                'pass shuffle="batch".')
            batch_logs = {}
            batch_logs['batch'] = batch_index
            batch_logs['size'] = len(batch_ids)
            batch_logs['ids'] = batch_ids
            callbacks.on_batch_begin(batch_index, batch_logs)
            outs = f(ins_batch)
            if not isinstance(outs, list):
                outs = [outs]
            for l, o in zip(out_labels, outs[1:]):
                batch_logs[l] = o

            callbacks.on_batch_end(batch_index, batch_logs)

            if batch_index == len(batches) - 1:  # last batch
                # validation
                if do_validation:
                    # replace with self._evaluate
                    val_outs = self._test_loop(val_f, val_ins,
                                               batch_size=batch_size,
                                               verbose=0)
                    if not isinstance(val_outs, list):
                        val_outs = [val_outs]
                    # same labels assumed
                    for l, o in zip(out_labels, val_outs[1:]):
                        epoch_logs['val_' + l] = o
        callbacks.on_epoch_end(epoch, epoch_logs)
        if callback_model.stop_training:
            break
    callbacks.on_train_end()
    return self.history

# Monkey-patch kgp.models.Model
Model._fit_loop = _fit_loop


def _on_batch_end(self, batch, logs=None):
    logs = logs if logs is not None else {}
    batch_size = logs.get('size', 0)
    self.seen += batch_size

    for k, v in logs.items():
        if k == 'ids': continue
        if k in self.totals:
            self.totals[k] += v * batch_size
        else:
            self.totals[k] = v * batch_size

# Monkey-patch keras.callbacks.BaseLogger
cbks.BaseLogger.on_batch_end = _on_batch_end
