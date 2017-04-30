"""
Data utility functions.
"""
import numpy as np

from six.moves import xrange


def standardize_data(data):
    if type(data) is not dict:
        raise ValueError("Data should be a dict.")
    if 'train' not in data:
        raise ValueError("Cannot find training set in the data.")

    # Compute mean and std on the train set
    X, y = data['train']
    X_mean = X.reshape((-1, X.shape[-1])).mean(axis=0)
    X_std = X.reshape((-1, X.shape[-1])).std(axis=0)
    y_mean = y.reshape((-1, y.shape[-1])).mean(axis=0)
    y_std = y.reshape((-1, y.shape[-1])).std(axis=0)

    # Standardize the data
    for set_name, (X, y) in data.iteritems():
        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std
        if len(y.shape) == 1:
            y = y[:, None]
        data[set_name] = [X, y]

    return data


def data_to_seq(X, Y,
                t_lag=8,
                t_future_shift=1,
                t_future_steps=1,
                t_sw_step=1,
                X_pad_with=None):
    """Slice X and Y into sequences using a sliding window.

    Arguments:
    ----------
        X : np.ndarray with ndim == 2
        Y : np.ndarray with ndim == 2
        t_sw_step : uint (default: 1)
            Time step of the sliding window.
        t_lag : uint (default: 8)
            (t_lag - 1) past time steps used to construct a sequence of inputs.
        t_future_shift : uint (default: 0)
            How far in the future predictions are supposed to be made.
        t_future_steps : uint (default: 1)
            How many steps to be predicted from t + t_future_shift.
            The sequences are constructed in a way that the model can be
            trained to predict Y[t_future:t_future+t_future_steps]
            from X[t-t_lag:t] where t_future = t + t_future_shift.
    """
    # Assume that provided X and Y are matrices and are aligned in time
    assert X.ndim == 2 and Y.ndim == 2
    assert len(X) == len(Y)

    # Pad X sequence from the beginning
    X_padding_left = np.zeros((t_lag - 1, X.shape[1]))
    X = np.vstack([X_padding_left, X])

    # The future steps of X should be skipped, hence padded with zeros
    # X_padding_right = np.zeros((t_future_shift+t_future_steps-1, X.shape[1]))

    nb_t_steps = 1 + len(X) - (t_future_shift + (t_future_steps - 1))
    X_seq, Y_seq = [], []
    for t in xrange(t_lag, nb_t_steps, t_sw_step):
        t_past = t - t_lag
        t_future = t_past + t_future_shift
        # X_seq.append(np.vstack([X[t_past:t], X_padding_right]))
        X_seq.append(X[t_past:t])
        Y_seq.append(Y[t_future:t_future+t_future_steps])
    X_seq = np.asarray(X_seq)
    Y_seq = np.asarray(Y_seq)

    return [X_seq, Y_seq]
