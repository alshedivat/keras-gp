"""
Interface for system identification data (Actuator and Drives).
"""
import os
import sys
import numpy as np

from six.moves import cPickle as pkl

def load_data(name, t_step=1, start=0., stop=100.,
              use_targets=True, batch_size=None, verbose=1):
    '''Load the system identification data.

    Arguments:
    ----------
        t_step : uint (default: 1)
            Take data points t_step apart from each other in time.
        start : float in [0., 100.) (default: 0.)
        stop : float in (0., 100.] (default: 100.)
        use_targets : bool (default: True)
        batch_size : uint or None (default: None)
        verbose : uint (default: 1)
    '''
    if 'DATA_PATH' not in os.environ:
        raise Exception("Cannot find DATA_PATH variable in the environment. "
                        "DATA_PATH should be the folder that contains "
                        "`sysid/` directory with the data. "
                        "Please export DATA_PATH before loading the data.")

    dataset_path = os.path.join(os.environ['DATA_PATH'],
                                'sysid', name + '.pkl')
    if not os.path.exists(dataset_path):
        raise Exception("Cannot find data: %s" % dataset_path)

    if verbose:
        sys.stdout.write('Loading data...')
        sys.stdout.flush()

    with open(dataset_path) as fp:
        data = pkl.load(fp)

    start = int((start/100.) * len(data['X']))
    stop = int((stop/100.) * len(data['X']))

    X = data['X'][start:stop:t_step,:]
    Y = data['Y'][start:stop:t_step,:]

    if use_targets:
        X = np.hstack([X, Y])

    if batch_size:
        nb_examples = (len(X) // batch_size) * batch_size
        X = X[:nb_examples]
        Y = Y[:nb_examples]

    if verbose:
        sys.stdout.write('Done.\n')
        print('# of loaded points: %d' % len(X))

    return X, Y
