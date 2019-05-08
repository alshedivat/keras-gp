"""Gaussian Process layers for Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.engine import InputSpec
from keras.engine import Layer

# Backend for neural networks
from keras import backend as K

# Backend for Gaussian processes
from .backend import GP_BACKEND


class GP(Layer):
    """Gaussian Process layer.

    Encapsulates GP backend functionality and provides a Keras-like interface
    for working with Gaussian processes.

    Arguments:
    ----------
        hyp : dict
            GP kernel hyper-parameters.
        batch_size : uint

        nb_train_samples : uint
            The size of the training data. GP needs to know this a priory.
        opt : dict (default: {})
            GP inference / training options.
        inf : str (default: 'infExact')
            Inference function name (should match one from GPML).
        lik : str (default: 'likGauss')
            Likelihood name (should match one from GPML).
        mean : str (default: 'meanZero')
            Mean function name (should match one from GPML).
        cov : str (default: 'covSEiso')
            Covariance function name (should match one from GPML).
        dlik : str (default: 'dlikExact')
            Derivative of the likelihood w.r.t. covariance kernel.
        update_grid : uint (default: 0)
            The frequency of grid updates in epochs.
            0 means the grid is being created once and fixed.
        gpml_path : str or None (default: None)
            Path to GPML. If None, the backend will attempt to use `GPML_PATH`
            environment variable and raises ValueError if couldn't find.
    """
    def __init__(self, hyp, batch_size, nb_train_samples, opt=None,
                 inf='infExact', lik='likGauss', dlik='dlikExact',
                 mean='meanZero', cov='covSEiso',
                 grid_kwargs=None, 
                 cov_args=None, mean_args=None,
                 update_grid=0,
                 engine=None, engine_kwargs=None,
                 gpml_path=None, verbose=1):
        self.hyp = hyp
        self.batch_size = batch_size
        self.nb_train_samples = nb_train_samples
        self.backend = GP_BACKEND(engine, engine_kwargs, gpml_path)
        self.backend_config = {
            'opt': opt or {},
            'inf': inf,
            'lik': lik,
            'dlik': dlik,
            'mean': mean,
            'cov': cov,
            'cov_args': cov_args, 
            'mean_args': mean_args,
            'grid_kwargs': grid_kwargs,
            'verbose': verbose,
        }
        self.update_grid = update_grid
        super(GP, self).__init__(trainable=False)

    @property
    def hyp(self):
        return self._hyp

    @hyp.setter
    def hyp(self, value):
        self._hyp = value

    @property
    def dlik_dh(self):
        return self._dlik_dh

    @dlik_dh.setter
    def dlik_dh(self, value):
        K.set_value(self._dlik_dh, value)

    @property
    def batch_ids(self):
        return self._batch_ids

    @batch_ids.setter
    def batch_ids(self, value):
        K.set_value(self._batch_ids, value)

    @property
    def batch_sz(self):
        return self._batch_sz

    @batch_sz.setter
    def batch_sz(self, value):
        K.set_value(self._batch_sz, value)

    @property
    def nlml(self):
        return self._nlml

    @nlml.setter
    def nlml(self, value):
        K.set_value(self._nlml, value)

    @property
    def mse(self):
        return self._mse

    @mse.setter
    def mse(self, value):
        K.set_value(self._mse, value)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def build(self, input_shape):
        """Create the internal variables for communication with GP backend.

        Arguments:
        ----------
            input_shape: Keras tensor (future input to layer)
                or list/tuple of Keras tensors to reference
                for weight shape computations.
        """
        assert len(input_shape) == 2
        input_dim = input_shape[-1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        # Configure GP backend
        self.backend.configure(input_dim, self.hyp, **self.backend_config)

        # Internal shared variables
        self._dlik_dh = K.zeros((self.nb_train_samples, input_dim))
        self._batch_ids = K.variable(np.zeros(self.batch_size), dtype='int32')
        self._batch_sz = K.variable(self.batch_size, dtype='int32')

        # Internal metrics
        self._nlml = K.variable(0.)
        self._mse = K.variable(0.)

        self.built = True
