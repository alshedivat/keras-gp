"""
Interface for the data from GEF 2012 wind forecasting Kaggle competition.
"""
import os
import sys

import numpy as np

from six.moves import cPickle as pkl

def load_data(start=0., stop=100., t_step=1, wf=[1], t_decay=None, verbose=1):
    """Load the GEF-wind data.

    Arguments:
    ----------
        t_step : uint
            Take data points t_step apart from each other in time.
        start : float in [0., 100.)
        stop : float in (0., 100.]
        wf : list of uints in [1, 7]
            The list of wind farm ids to use data from.
        t_decay : float in (0., 1.] or None (default: None)
            At each time step `t`, we are given 4 forecasts of the wind
            parameters. Each of these forecasts was made at some point
            `t_forecast` during the past 48 hours with a step of 12 hours
            between the forecasts. We average these forecasts with coefficients
            proportional to `t_dacay^forecast_n` so that older forecasts
            contribute less than the fresh ones. If None, all forecasts are
            used as independent features (no averaging is done).
        verbose : uint (default: 1)
    """
    if 'DATA_PATH' not in os.environ:
        raise Exception("Cannot find DATA_PATH variable in the environment. "
                        "DATA_PATH should be the folder that contains "
                        "`GEF/wind/` directory with the GEF data. "
                        "Please export DATA_PATH before loading the data.")

    hours_path = os.path.join(os.environ['DATA_PATH'],
                              'GEF', 'wind', 'hours.npy')
    targets_path = os.path.join(os.environ['DATA_PATH'],
                                'GEF', 'wind', 'targets.npy')
    forecast_paths = [
        os.path.join(os.environ['DATA_PATH'], 'GEF', 'wind',
                     'windforecasts_wf%d.npy' % i)
        for i in wf]

    if not os.path.exists(hours_path):
        raise Exception("Cannot find data: %s" % hours_path)
    if not os.path.exists(targets_path):
        raise Exception("Cannot find data: %s" % targets_path)
    for path in forecast_paths:
        if not os.path.exists(path):
            raise Exception("Cannot find data: %s" % path)

    if verbose:
        sys.stdout.write('Loading data...')
        sys.stdout.flush()

    hours = np.hstack(len(wf) * [np.load(hours_path)])
    targets = np.load(targets_path)[:, [i - 1 for i in wf]]
    forecasts = np.vstack([np.load(path) for path in forecast_paths])

    if t_decay is not None:
        decay = t_decay**np.arange(forecasts.shape[1])
        decay /= decay.sum()
        forecasts = (forecasts * decay).sum(axis=1)
    else:
        forecasts = forecasts.reshape(len(forecasts), -1)

    X = np.hstack([hours[:, None], forecasts])
    Y = targets.T.reshape(-1, 1)

    # Select the data points
    start = int((start/100.) * len(X))
    stop = int((stop/100.) * len(X))

    X = X[start:stop:t_step,:]
    Y = Y[start:stop:t_step,:]

    if verbose:
        sys.stdout.write('Done.\n')
        print('# of loaded points: %d' % len(X))

    return X, Y
