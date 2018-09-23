"""
Interface for system identification data (Actuator and Drives).
"""
from __future__ import print_function

import os
import sys
import zipfile
import warnings

import numpy as np
import scipy.io as sio

from six.moves import urllib
from six.moves import cPickle as pkl

SOURCE_URLS = {
    'actuator': 'https://www.cs.cmu.edu/~mshediva/assets/data/actuator.mat',
    'drives': 'https://www.cs.cmu.edu/~mshediva/assets/data/NonlinearData.zip',
}


def maybe_download(data_path, dataset_name, verbose=1):
    source_url = SOURCE_URLS[dataset_name]
    datadir_path = os.path.join(data_path, 'sysid')
    dataset_path = os.path.join(datadir_path, dataset_name + '.mat')

    # Create directories (if necessary)
    if not os.path.isdir(datadir_path):
        os.makedirs(datadir_path)

    # Download & extract the data (if necessary)
    if not os.path.isfile(dataset_path):
        if dataset_name == 'actuator':
            urllib.request.urlretrieve(source_url, dataset_path)
        if dataset_name == 'drives':
            assert source_url.endswith('.zip')
            archive_path = os.path.join(datadir_path, 'tmp.zip')
            urllib.request.urlretrieve(source_url, archive_path)
            with zipfile.ZipFile(archive_path, 'r') as zfp:
                zfp.extract('DATAPRBS.MAT', datadir_path)
            os.rename(os.path.join(datadir_path, 'DATAPRBS.MAT'), dataset_path)
            os.remove(archive_path)
        if verbose:
            print("Successfully downloaded `%s` dataset from %s." %
                  (dataset_name, source_url))

    return dataset_path


def load_data(dataset_name, t_step=1, start=0., stop=100.,
              use_targets=True, batch_size=None, verbose=1):
    """Load the system identification data.

    Arguments:
    ----------
        t_step : uint (default: 1)
            Take data points t_step apart from each other in time.
        start : float in [0., 100.) (default: 0.)
        stop : float in (0., 100.] (default: 100.)
        use_targets : bool (default: True)
        batch_size : uint or None (default: None)
        verbose : uint (default: 1)
    """
    if dataset_name not in {'actuator', 'drives'}:
        raise ValueError("Unknown dataset: %s" % dataset_name)

    if 'DATA_PATH' not in os.environ:
        warnings.warn("Cannot find DATA_PATH variable in the environment. "
                      "Using <current_working_directory>/data/ instead.")
        DATA_PATH = os.path.join(os.getcwd(), 'data')
    else:
        DATA_PATH = os.environ['DATA_PATH']

    dataset_path = maybe_download(DATA_PATH, dataset_name, verbose=verbose)
    if not os.path.exists(dataset_path):
        raise Exception("Cannot find data: %s" % dataset_path)

    if verbose:
        sys.stdout.write('Loading data...')
        sys.stdout.flush()

    data_mat = sio.loadmat(dataset_path)
    if dataset_name == 'actuator':
        X, Y = data_mat['u'], data_mat['p']
    if dataset_name == 'drives':
        X, Y = data_mat['u1'], data_mat['z1']

    start = int((start/100.) * len(X))
    stop = int((stop/100.) * len(X))

    X = X[start:stop:t_step,:]
    Y = Y[start:stop:t_step,:]

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
