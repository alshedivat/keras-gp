"""
Interface for the data from GEF 2012 power forecasting Kaggle competition.
"""
import os
import sys

import numpy as np

from six.moves import cPickle as pkl

def load_data(start=0., stop=100., t_step=1, average_load=True, verbose=1):
    """Load the GEF-power data.

    Arguments:
    ----------
        t_step : uint
            Take data points t_step apart from each other in time.
        start : float in [0., 100.)
        stop : float in (0., 100.]
        average_load : bool (default: True)
            Whether to use hourly power load averaged across 20 stations, or
            separate power loads for each station.
        verbose : uint (default: 1)
    """
    if 'DATA_PATH' not in os.environ:
        raise Exception("Cannot find DATA_PATH variable in the environment. "
                        "DATA_PATH should be the folder that contains "
                        "`GEF/power/` directory with the GEF data. "
                        "Please export DATA_PATH before loading the data.")

    dataset_path = os.path.join(os.environ['DATA_PATH'], 'GEF', 'power',
                                'temperature_load_history.pkl')
    if not os.path.exists(dataset_path):
        raise Exception("Cannot find data: %s" % dataset_path)

    if verbose:
        sys.stdout.write('Loading data from %s...' %
                         os.path.basename(dataset_path))
        sys.stdout.flush()

    with open(dataset_path) as fp:
        data = pkl.load(fp)

    # Select the data points
    start = int((start/100.) * len(data['X']))
    stop = int((stop/100.) * len(data['X']))

    X = data['X'][start:stop:t_step,:]
    Y = data['Y'][start:stop:t_step,:]

    if average_load:
        Y = np.mean(Y, axis=1, keepdims=True)

    if verbose:
        sys.stdout.write('Done.\n')
        print('# of loaded points: %d' % len(X))

    return X, Y
