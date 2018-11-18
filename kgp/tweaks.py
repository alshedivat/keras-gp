"""
Tweaks for KGP and original Keras classes or methods.
To keep the primary KGP code clean, all tweaks are kept in a separate file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import keras.callbacks as cbks
from keras.engine.training import _batch_shuffle
from keras.engine.training import _make_batches
from keras.engine.training import _slice_arrays
from keras.engine.training import _standardize_input_data

from keras.models import Model

from keras import backend as K


def _fit_loop(self, f, ins, out_labels=None, batch_size=32,
              epochs=100, verbose=1, callbacks=None,
              val_f=None, val_ins=None, shuffle=True,
              callback_metrics=None, initial_epoch=0,
              steps_per_epoch=None, validation_steps=None):
    """Abstract fit function for f(ins).
    Assume that f returns a list, labeled by out_labels.

    # Arguments
        f: Keras function returning a list of tensors
        ins: List of tensors to be fed to `f`
        out_labels: List of strings, display names of
            the outputs of `f`
        batch_size: Integer batch size or None if unknown.
        epochs: Number of times to iterate over the data
        verbose: Verbosity mode, 0, 1 or 2
        callbacks: List of callbacks to be called during training
        val_f: Keras function to call for validation
        val_ins: List of tensors to be fed to `val_f`
        shuffle: Whether to shuffle the data at the beginning of each epoch
        callback_metrics: List of strings, the display names of the metrics
            passed to the callbacks. They should be the
            concatenation of list the display names of the outputs of
             `f` and the list of display names of the outputs of `f_val`.
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)
        steps_per_epoch: Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. Ignored with the default value of `None`.
        validation_steps: Number of steps to run validation for
            (only if doing validation from data tensors).
            Ignored with the default value of `None`.

    # Returns
        `History` object.

    [A tweaked version.]
    """
    do_validation = False
    if val_f and val_ins:
        do_validation = True
        if verbose and ins and hasattr(ins[0], 'shape') and hasattr(val_ins[0], 'shape'):
            print('Train on %d samples, validate on %d samples' %
                  (ins[0].shape[0], val_ins[0].shape[0]))
    if validation_steps:
        do_validation = True
        if steps_per_epoch is None:
            raise ValueError('Can only use `validation_steps` '
                             'when doing step-wise '
                             'training, i.e. `steps_per_epoch` '
                             'must be set.')

    num_train_samples = self._check_num_samples(ins, batch_size,
                                                steps_per_epoch,
                                                'steps_per_epoch')
    if num_train_samples is not None:
        index_array = np.arange(num_train_samples)

    self.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
    if verbose:
        if steps_per_epoch is not None:
            count_mode = 'steps'
        else:
            count_mode = 'samples'
        callbacks += [cbks.ProgbarLogger(count_mode)]
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
        'steps': steps_per_epoch,
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
        epoch_logs = {}
        if steps_per_epoch is not None:
            for step_index in range(steps_per_epoch):
                batch_logs = {}
                batch_logs['batch'] = step_index
                batch_logs['size'] = 1
                callbacks.on_batch_begin(step_index, batch_logs)
                outs = f(ins)

                if not isinstance(outs, list):
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(step_index, batch_logs)
                if callback_model.stop_training:
                    break

            if do_validation:
                val_outs = self._test_loop(val_f, val_ins,
                                           batch_size=batch_size,
                                           steps=validation_steps,
                                           verbose=0)
                if not isinstance(val_outs, list):
                    val_outs = [val_outs]
                # Same labels assumed.
                for l, o in zip(out_labels, val_outs):
                    epoch_logs['val_' + l] = o
        else:
            if shuffle == 'batch':
                index_array = _batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = _make_batches(num_train_samples, batch_size)
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
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)
                if callback_model.stop_training:
                    break

                if batch_index == len(batches) - 1:  # last batch.
                    if do_validation:
                        val_outs = self._test_loop(val_f, val_ins,
                                                   batch_size=batch_size,
                                                   verbose=0)
                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]
                        # same labels assumed
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o
        callbacks.on_epoch_end(epoch, epoch_logs)
        if callback_model.stop_training:
            break
    callbacks.on_train_end()
    return self.history


def predict(self, x,
            batch_size=None,
            learning_phase=0.,
            verbose=0,
            steps=None):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            x: the input data, as a Numpy array
                (or list of Numpy arrays if the model has multiple outputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.

        # Returns
            Numpy array(s) of predictions.

        # Raises
            ValueError: In case of mismatch between the provided
                input data and the model's expectations,
                or in case a stateful model receives a number of samples
                that is not a multiple of the batch size.

        [A tweaked version.]
        """
        # Backwards compatibility.
        if batch_size is None and steps is None:
            batch_size = 32
        if x is None and steps is None:
            raise ValueError('If predicting from data tensors, '
                             'you should specify the `steps` '
                             'argument.')
        # validate user data
        x = _standardize_input_data(x, self._feed_input_names,
                                    self._feed_input_shapes,
                                    check_batch_axis=False)
        if self.stateful:
            if x[0].shape[0] > batch_size and x[0].shape[0] % batch_size != 0:
                raise ValueError('In a stateful network, '
                                 'you should only pass inputs with '
                                 'a number of samples that can be '
                                 'divided by the batch size. Found: ' +
                                 str(x[0].shape[0]) + ' samples. '
                                 'Batch size: ' + str(batch_size) + '.')

        # prepare inputs, delegate logic to _predict_loop
        if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
            ins = x + [learning_phase]
        else:
            ins = x
        self._make_predict_function()
        f = self.predict_function
        return self._predict_loop(f, ins,
                                  batch_size=batch_size, verbose=verbose)

# Monkey-patch keras.models.Model
Model._fit_loop = _fit_loop
Model.predict = predict


def _on_batch_end(self, batch, logs=None):
    logs = logs if logs is not None else {}
    batch_size = logs.get('size', 0)
    self.seen += batch_size

    for k, v in logs.items():
        if k == 'ids':
            continue
        if k in self.totals:
            self.totals[k] += v * batch_size
        else:
            self.totals[k] = v * batch_size

# Monkey-patch keras.callbacks.BaseLogger
cbks.BaseLogger.on_batch_end = _on_batch_end
