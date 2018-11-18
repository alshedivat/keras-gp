"""Gaussian Process models for Keras 2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Model as KerasModel
from keras.engine.topology import _to_list
from keras.engine.training import _standardize_input_data

from .callbacks import UpdateGP


class Model(KerasModel):
    """Model that supports arbitrary structure with GP output layers.

    This class extends `keras.models.Model` and allows using Gaussian Processes
    as output layers. The model completely inherits the function interface
    of Keras and is constructed in a standard way.

    On training, GPs are optimized using empirical Bayes (log marginal
    likelihood maximization) using semi-stochastic alternating scheme with
    delayed kernel matrix updates [1]. Non-GP output layers can use one the
    standard Keras objectives, e.g., the mean squared error.
    """
    def __init__(self, inputs, outputs, name=None):
        super(Model, self).__init__(inputs, outputs, name)

        # List all output GP layers
        self.output_gp_layers = [layer for layer in self.output_layers
                                 if layer.name.startswith('gp')]

    def compile(self, optimizer, loss, metrics=None, loss_weights=None,
                sample_weight_mode=None, **kwargs):
        super(Model, self).compile(optimizer, loss, metrics, loss_weights,
                                   sample_weight_mode, **kwargs)

        # Remove the metrics meaningless for GP output layers
        self.metrics_tensors = [
            mt for mt, mn in zip(self.metrics_tensors, self.metrics_names[1:])
            if not (mn.startswith('gp') and mn.endswith('loss'))
        ]
        self.metrics_names = [
            mn for mn in self.metrics_names
            if not (mn.startswith('gp') and mn.endswith('loss'))
        ]

        # Add MSE and NLML metrics for each output GP
        for gp in self.output_gp_layers:
            self.metrics_tensors.extend([gp.mse, gp.nlml])
            self.metrics_names.extend([gp.name + '_mse', gp.name + '_nlml'])

        # Add cumulative MSE & NLML metrics
        self.mse = sum([gp.mse for gp in self.output_gp_layers])
        self.nlml = sum([gp.nlml for gp in self.output_gp_layers])
        self.metrics_tensors.extend([self.mse, self.nlml])
        self.metrics_names.extend(['mse', 'nlml'])

    def transform(self, x, batch_size=32, learning_phase=0., verbose=0):
        h = super(Model, self).predict(x, batch_size, learning_phase, verbose)
        return _to_list(h)

    def fit(self, X, Y,
            batch_size=32,
            epochs=1,
            gp_n_iter=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            **kwargs):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        For argument details, refer to `keras.engine.training.Model.fit`.

        Notes:
            The following arguments are currently unsupported by models with GP
            output layers:
            - validation_split
            - class_weight
            - sample_weight
        """
        # Validate user data
        X, Y, _ = self._standardize_user_data(
            X, Y,
            sample_weight=None,
            class_weight=None,
            check_batch_axis=False,
            batch_size=batch_size)
        if validation_data is not None:
            X_val, Y_val, _ = self._standardize_user_data(
                *validation_data,
                sample_weight=None,
                class_weight=None,
                check_batch_axis=False,
                batch_size=batch_size)
            validation_data = (X_val, Y_val)

        # Setup GP updates
        update_gp = UpdateGP(ins=(X, Y),
                             val_ins=validation_data,
                             batch_size=batch_size,
                             gp_n_iter=gp_n_iter,
                             verbose=verbose)
        callbacks = [update_gp] + (callbacks or [])

        return super(Model, self).fit(
            X, Y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            shuffle=shuffle,
            initial_epoch=initial_epoch,
            **kwargs)

    def finetune(self, X, Y, batch_size=32, gp_n_iter=1, verbose=1):
        """Finetune the output GP layers assuming the network is pre-trained.

        Arguments:
        ----------
            X : np.ndarray or list of np.ndarrays
            Y : np.ndarray or list of np.ndarrays
            batch_size : uint (default: 128)
                Batch size used for data streaming through the network.
            gp_n_iter : uint (default: 100)
                Number of iterations for GP training.
            verbose : uint (default: 1)
                Verbosity mode, 0 or 1.
        """
        # Validate user data
        X = _standardize_input_data(
            X, self._feed_input_names, self._feed_input_shapes,
            check_batch_axis=False, exception_prefix='input')

        H = self.transform(X, batch_size=batch_size)

        if verbose:
            print("Finetuning output GPs...")

        for gp, h, y in zip(self.output_gp_layers, H, Y):
            # Update GP data (and grid if necessary)
            gp.backend.update_data('tr', h, y)
            if gp.update_grid:
                gp.backend.update_grid('tr')

            # Train GP
            gp.hyp = gp.backend.train(gp_n_iter, verbose=verbose)

        if verbose:
            print("Done.")

    def evaluate(self, X, Y, batch_size=32, verbose=0):
        """Compute NLML on the given data.

        Arguments:
        ----------
            X : np.ndarray or list of np.ndarrays
            Y : np.ndarray or list of np.ndarrays
            batch_size : uint (default: 128)
            verbose : uint (default: 0)
                Verbosity mode, 0 or 1.

        Returns:
        --------
            nlml : float
        """
        # Validate user data
        X, Y, _ = self._standardize_user_data(
            X, Y,
            sample_weight=None,
            class_weight=None,
            check_batch_axis=False,
            batch_size=batch_size)

        H = self.transform(X, batch_size=batch_size)

        nlml = 0.
        for gp, h, y in zip(self.output_gp_layers, H, Y):
            nlml += gp.backend.evaluate('tmp', h, y)

        return nlml

    def predict(self, X, X_tr=None, Y_tr=None,
                batch_size=32, return_var=False, verbose=0):
        """Generate output predictions for the input samples batch by batch.

        Arguments:
        ----------
            X : np.ndarray or list of np.ndarrays
            batch_size : uint (default: 128)
            return_var : bool (default: False)
                Whether predictive variance is returned.
            verbose : uint (default: 0)
                Verbosity mode, 0 or 1.

        Returns:
        --------
            preds : a list or a tuple of lists
                Lists of output predictions and variance estimates.
        """
        # Update GP data if provided (and grid if necessary)
        if X_tr is not None and Y_tr is not None:
            X_tr, Y_tr, _ = self._standardize_user_data(
                X_tr, Y_tr,
                sample_weight=None,
                class_weight=None,
                check_batch_axis=False,
                batch_size=batch_size)
            H_tr = self.transform(X_tr, batch_size=batch_size)
            for gp, h, y in zip(self.output_gp_layers, H_tr, Y_tr):
                gp.backend.update_data('tr', h, y)
                if gp.update_grid:
                    gp.backend.update_grid('tr')

        # Validate user data
        X = _standardize_input_data(
            X, self._feed_input_names, self._feed_input_shapes,
            check_batch_axis=False, exception_prefix='input')

        H = self.transform(X, batch_size=batch_size)

        preds = []
        for gp, h in zip(self.output_gp_layers, H):
            preds.append(gp.backend.predict(h, return_var=return_var))

        if return_var:
            preds = map(list, zip(*preds))

        return preds


# Apply tweaks
from . import tweaks
