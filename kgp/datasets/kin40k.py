"""
Interface for kin40k UCI dataset.
"""
import os
import sys
import numpy as np


def load_data(start=0., stop=100., verbose=1):
    """Load the Kin40k data.

    Arguments:
    ----------
        start : float in [0., 100.) (default: 0.)
        stop : float in (0., 100.] (default: 100.)
        verbose : uint (default: 1)
    """
    if 'DATA_PATH' not in os.environ:
        raise Exception("Cannot find DATA_PATH variable in the environment. "
                        "DATA_PATH should be the folder that contains "
                        "`kin40k/` directory with the data. "
                        "Please export DATA_PATH before loading the data.")

    dataset_path = os.path.join(os.environ['DATA_PATH'],'kin40k','kin40k.npz')
    if not os.path.exists(dataset_path):
        raise Exception("Cannot find data: %s" % dataset_path)

    if verbose:
        sys.stdout.write('Loading data...')
        sys.stdout.flush()

    data = np.load(dataset_path)

    start = int((start/100.) * len(data['X']))
    stop = int((stop/100.) * len(data['X']))

    X = data['X'][start:stop,:]
    Y = data['Y'][start:stop,:]

    if verbose:
        sys.stdout.write('Done.\n')
        print('# of loaded points: %d' % len(X))

    return X, Y
