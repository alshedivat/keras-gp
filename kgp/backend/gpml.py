"""GPML backend for Gaussian processes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from pprint import pprint

# MATLAB scripts
_gp_train_epoch = """
hyp = minimize(hyp, @gp, -{n_iter:d}, {inf}, {mean}, {cov}, {lik}, X_tr, y_tr);
"""
_gp_evaluate = """
[nlZ dnlZ post     ] = gp(hyp, {inf}, {mean}, {cov}, {lik}, {X}, {y});
"""
_gp_predict = """
[ymu ys2 fmu fs2   ] = gp(hyp, {inf}, {mean}, {cov}, {lik}, X_tr, y_tr, X_tst);
"""
_gp_predict_grid = """
[post nlZ dnlZ     ] = infGrid(hyp, {mean}, {cov}, {lik}, X_tr, y_tr, opt);
[fmu fs2 ymu ys2   ] = post.predict(X_tst);
"""
_gp_dlik = """
[dlik_dx           ] = dlik(hyp, {mean}, {cov}, {lik}, {dcov}, {X}, {y});
"""
_gp_create_grid = """
xg = covGrid('create', {X}, eq, k);
"""


class GPML(object):
    """Class that implements backend functionality for Gaussian processes.

    Arguments:
    ----------
        engine : str (either 'matlab' or 'octave')
        gpml_path : str or None
    """
    def __init__(self, engine=None, engine_kwargs=None, gpml_path=None):
        if engine is None:
            if 'GP_ENGINE' in os.environ:
                engine = os.environ['GP_ENGINE']
            else:
                raise ValueError("GP_ENGINE is neither provided nor available "
                                 "in the environment.")
        if engine == 'matlab':
            from .engines import MATLABEngine as Engine
        elif engine == 'octave':
            from .engines import OctaveEngine as Engine
        else:
            raise ValueError('Unknown GP_ENGINE: %s' % engine)

        if gpml_path is None:
            if 'GPML_PATH' in os.environ:
                gpml_path = os.environ['GPML_PATH']
            else:
                current_dir = os.path.dirname(os.path.realpath(__file__))
                gpml_path = os.path.join(current_dir, 'gpml')
                if not os.path.isfile(os.path.join(gpml_path, 'startup.m')):
                    raise ValueError(
                        "Neither GPML_PATH is provided nor GPML library is "
                        "available directly from keras-gp. "
                        "Please make sure you cloned keras-gp *recursively*.")

        engine_kwargs = engine_kwargs or {}
        self.eng = Engine(**engine_kwargs)
        self.eng.addpath(gpml_path)
        self.eng.eval('startup', verbose=0)

        utils_path = os.path.join(os.path.dirname(__file__), 'utils')
        self.eng.addpath(utils_path)

    def configure(self, input_dim, hyp, opt, inf, mean, cov, lik, dlik,
                  grid_kwargs=None, cov_args=None, mean_args=None, verbose=1):
        """Configure GPML-based Guassian process.

        Arguments:
        ----------
            input_dim : uint
                The dimension of the GP inputs.
            hyp : dict
                A dictionary of GP hyperparameters.
            opt : dict
                GPML inference/training options (see GPML doc for details).
            inf : str
                Name of the inference method (see GPML doc for details).
            mean : str
                Name of the mean function (see GPML doc for details).
            cov : str
                Name of the covariance function (see GPML doc for details).
            lik : str
                Name of the likelihood function (see GPML doc for details).
            dlik : str
                Name of the function that computes dlik/dx.
            grid_kwargs : dict
                'eq' : uint
                    Whether to enforce an equispaced grid.
                'k' : uint or float in (0, 1]
                    Number of inducing points per dimension.
                'xg' : list
                    Manually specified grid. Must be represented in the form of
                    a list of np.ndarrays that specify the grid points for each
                    dimension.
        """
        self.config = {}
        self.config['lik']  = "{@%s}" % lik

        if mean_args is None:
            self.config['mean'] = "{@%s}" % mean
        else:
            self.config['mean'] = '{@%s, %s}' % (
                mean, ', '.join(str(e) for e in mean_args))
        
        self.config['inf']  = "{@(varargin) %s(varargin{:}, opt)}" % inf
        self.config['dlik'] = "@(varargin) %s(varargin{:}, opt)" % dlik

        if inf == 'infGrid':
            assert grid_kwargs is not None, (
                "GPML: No arguments provided for grid generation for infGrid.")
            self.eng.push('k', float(grid_kwargs['k']))
            self.eng.push('eq', float(grid_kwargs['eq']))
            if 'xg' in grid_kwargs:
                self.eng.push('xg', grid_kwargs['xg'])
            if cov_args is None:
                cov = ','.join(input_dim * ['{@%s}' % cov])
            else:
                cov = ','.join(input_dim * [
                    '{@%s, %s}' % (cov, ', '.join(str(e) for e in cov_args))])
            if input_dim > 1:
                cov = '{' + cov + '}'
            hyp['cov'] = np.tile(hyp['cov'], (1, input_dim))
            self.config['cov'] = "{@covGrid, %s, xg}" % cov
            self.config['dcov'] = "[]"
            self.using_grid = True
        else:
            hyp['cov'] = np.asarray(hyp['cov'])
            self.config['cov'] = "{@%s}" % cov
            self.config['dcov'] = "@d%s" % cov
            self.using_grid = False

        self.eng.push('hyp', hyp)
        self.eng.push('opt', opt)
        self.eng.eval("dlik = %s;" % self.config['dlik'], verbose=0)

        if verbose:
            print("GP configuration:")
            pprint(self.config)

    def update_data(self, which_set, X, y=None, verbose=0):
        """Update data in GP backend.
        """
        assert which_set in {'tr', 'tst', 'val', 'tmp'}
        self.eng.push('X_' + which_set, X)
        if y is not None:
            self.eng.push('y_' + which_set, y)

    def update_grid(self, which_set, verbose=0):
        """Update grid for grid-based GP inference.
        """
        assert which_set in {'tr', 'tst', 'val', 'tmp'}
        self.config.update({'X': 'X_' + which_set, 'y': None})
        self.eng.eval(_gp_create_grid.format(**self.config), verbose=verbose)

    def evaluate(self, which_set, X=None, y=None, verbose=0):
        """Evaluate GP for given X and y.
        Return negative log marginal likelihood.
        """
        assert which_set in {'tr', 'tst', 'val', 'tmp'}
        if X is not None and y is not None:
            self.update_data(which_set, X, y)
        X_name, y_name = 'X_' + which_set, 'y_' + which_set
        self.config.update({'X': X_name, 'y': y_name})
        self.eng.eval(_gp_evaluate.format(**self.config), verbose=verbose)
        nlZ = self.eng.pull('nlZ')
        return nlZ

    def predict(self, X, X_tr=None, y_tr=None, return_var=False, verbose=0):
        """Predict ymu and ys2 for a given X. Return ymu and ys2.
        """
        self.update_data('tst', X)
        if X_tr is not None and y_tr is not None:
            self.update_data('tr', X_tr, y_tr)
        if self.using_grid:
            self.eng.eval(_gp_predict_grid.format(**self.config),
                          verbose=verbose)
        else:
            self.eng.eval(_gp_predict.format(**self.config),
                          verbose=verbose)
        preds = self.eng.pull('ymu')
        if return_var:
            preds = (preds, self.eng.pull('ys2'))
        return preds

    def train(self, n_iter, X_tr=None, y_tr=None, verbose=0):
        """Train GP for `n_iter` iterations. Return a dict of hyperparameters.
        """
        if X_tr is not None and y_tr is not None:
            self.update_data('tr', X_tr, y_tr)
        self.config.update({'n_iter': n_iter})
        self.eng.eval(_gp_train_epoch.format(**self.config), verbose=verbose)
        hyp = self.eng.pull('hyp')
        return hyp

    def get_dlik_dx(self, which_set, verbose=0):
        """Get derivative of the log marginal likelihood w.r.t. the kernel.
        """
        assert which_set in {'tr', 'tst', 'val', 'tmp'}
        X_name, y_name = 'X_' + which_set, 'y_' + which_set
        self.config.update({'X': X_name, 'y': y_name})
        self.eng.eval(_gp_dlik.format(**self.config), verbose=verbose)
        dlik_dx = self.eng.pull('dlik_dx')
        return dlik_dx

    def eval_predict(self, X, X_tr=None, y_tr=None, verbose=0):
        """Use the grid-specific fast evaluation and prediction.
        """
        self.update_data('tst', X)
        if X_tr is not None and y_tr is not None:
            self.update_data('tr', X_tr, y_tr)
        if self.using_grid:
            self.eng.eval(_gp_predict_grid.format(**self.config),
                          verbose=verbose)
        else:
            self.eng.eval(_gp_evaluate.format(**self.config), verbose=verbose)
            self.eng.eval(_gp_predict.format(**self.config), verbose=verbose)
        return self.eng.pull('nlZ'), self.eng.pull('ymu')
