"""
Tests for datasets and their utility functions.
"""
import numpy as np

from six.moves import xrange

from kgp.datasets.data_utils import *


X_dim = 10
Y_dim = 3

t_lag = 5
t_future_shift = 1
t_future_steps = 2
t_sw_step = 1


def test_data_to_seq():
    N = 10
    X = np.arange(N * X_dim).reshape(N, X_dim)
    Y = np.arange(N * Y_dim).reshape(N, Y_dim)
    X_seq, Y_seq = data_to_seq(X, Y,
                               t_lag=t_lag,
                               t_future_shift=t_future_shift,
                               t_future_steps=t_future_steps,
                               t_sw_step=t_sw_step)

    t_future = t_future_shift + t_future_steps - 1
    expected_num_seq = N - t_future
    assert X_seq.shape == (expected_num_seq, t_lag + t_future, X_dim)
    assert Y_seq.shape == (expected_num_seq, t_future_steps, Y_dim)


def test_preprocess_data():
    N_train, N_test, N_valid = 20, 7, 10

    X_train = np.arange(N_train * X_dim).reshape(N_train, X_dim)
    Y_train = np.arange(N_train * Y_dim).reshape(N_train, Y_dim)
    X_test = np.arange(N_test * X_dim).reshape(N_test, X_dim)
    Y_test = np.arange(N_test * Y_dim).reshape(N_test, Y_dim)
    X_valid = np.arange(N_valid * X_dim).reshape(N_valid, X_dim)
    Y_valid = np.arange(N_valid * Y_dim).reshape(N_valid, Y_dim)

    data = {
        'train': (X_train, Y_train),
        'test': (X_test, Y_test),
        'valid': (X_valid, Y_valid),
    }

    # Make sure zero padding works properly
    new_data1 = preprocess_data(data,
                                standardize=False,
                                t_lag=t_lag,
                                t_future_shift=t_future_shift,
                                t_future_steps=t_future_steps,
                                t_sw_step=t_sw_step,
                                seq_restart=True)
    for t in xrange(t_lag):
        assert np.isclose(new_data1['train'][0][t][:t_lag - t - 1].sum(), 0.0)
        assert np.isclose(new_data1['test'][0][t][:t_lag - t - 1].sum(), 0.0)
        assert np.isclose(new_data1['valid'][0][t][:t_lag - t - 1].sum(), 0.0)

    # Make sure padding from previous sequence works properly
    new_data1 = preprocess_data(data,
                                standardize=False,
                                t_lag=t_lag,
                                t_future_shift=t_future_shift,
                                t_future_steps=t_future_steps,
                                t_sw_step=t_sw_step,
                                seq_restart=False)
    for t in xrange(t_lag - 1):
        assert np.allclose(new_data1['test'][0][t][:t_lag - t - 1],
                           X_train[-(t_lag - t - 1):])
        assert np.allclose(new_data1['valid'][0][t][:t_lag - t - 1],
                           X_test[-(t_lag - t - 1):])
