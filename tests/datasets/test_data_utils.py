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
    # assert X_seq.shape == (expected_num_seq, t_lag + t_future, X_dim)
    assert X_seq.shape == (expected_num_seq, t_lag, X_dim)
    assert Y_seq.shape == (expected_num_seq, t_future_steps, Y_dim)
