"""
Data utility functions.
"""
import numpy as np

from six.moves import xrange


def standardize_data(X, X_mean=None, X_std=None):
    if X is not None:
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if X_mean is None:
            X_mean = X.mean(axis=0)
        if X_std is None:
            X_std = X.std(axis=0)
        X = (X - X_mean) / X_std
    return X, X_mean, X_std


def data_to_seq(X, Y,
                t_lag=8,
                t_future_shift=1,
                t_future_steps=1,
                t_sw_step=1,
                X_pad_with=None):
    '''Slice X and Y into sequences using a sliding window.

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
    '''
    # Assume that provided X and Y are matrices and are aligned in time
    assert X.ndim == 2 and Y.ndim == 2
    assert len(X) == len(Y)

    # Pad X sequence from the beginning
    if X_pad_with is not None:
        X_padding_left = X_pad_with[-(t_lag - 1):]
        if t_lag > len(X_pad_with) + 1:
            # if X_pad_with is not long enough, pad the rest with zeros
            X_padding_left = np.vstack([
                np.zeros((t_lag - 1 - len(X_pad_with), X.shape[1])),
                X_padding_left
            ])
    else:
        X_padding_left = np.zeros((t_lag - 1, X.shape[1]))
    X = np.vstack([X_padding_left, X])

    # The future steps of X should be skipped, hence padded with zeros
    X_padding_right = np.zeros((t_future_shift+t_future_steps-1, X.shape[1]))

    nb_t_steps = 1 + len(X) - (t_future_shift + (t_future_steps - 1))
    X_seq, Y_seq = [], []
    for t in xrange(t_lag, nb_t_steps, t_sw_step):
        t_past = t - t_lag
        t_future = t_past + t_future_shift
        X_seq.append(np.vstack([X[t_past:t], X_padding_right]))
        Y_seq.append(Y[t_future:t_future+t_future_steps])
    X_seq = np.asarray(X_seq)
    Y_seq = np.asarray(Y_seq)

    return [X_seq, Y_seq]


def preprocess_data(data,
                    standardize=True,
                    multiple_outputs=False,
                    t_lag=32,
                    t_future_shift=1,
                    t_future_steps=1,
                    t_sw_step=1,
                    seq_restart=True):
    '''Pre-process by standardizing it and slicing into sequences.

    Arguments:
    ----------
        data : dict
            Must have 'train' and 'test' keys; 'valid' is optional.
        standardize : bool (default: True)
        multiple_outputs : bool (default: False)
            Whether outputs are represented as a list of 1-dimensional vectors.
        t_lag : uint (default: 32)
        t_future_shift : uint (default: 1)
        t_future_steps : uint (default: 1)
        t_sw_step : uint (default: 1)
        seq_restart : bool (default: bool)
            Whether to start sequences for train, test, and valid from zero
            or use the whole data as a continuous sequence. In the latter case,
            the assumed order is `train -> test -> valid`.
    '''
    X_train, Y_train = data['train']
    X_test, Y_test = data['test']

    # Center and standardize the data
    if standardize:
        X_train, X_mean, X_std = standardize_data(X_train)
        Y_train, Y_mean, Y_std = standardize_data(Y_train)
        X_test, _, _ = standardize_data(X_test, X_mean, X_std)
        Y_test, _, _ = standardize_data(Y_test, Y_mean, Y_std)

    preproc_data = {}

    # Prepare train and test data
    preproc_data['train'] = data_to_seq(X_train, Y_train,
                                        t_lag=t_lag,
                                        t_future_shift=t_future_shift,
                                        t_future_steps=t_future_steps,
                                        t_sw_step=t_sw_step)

    X_test_padding = None if seq_restart else X_train
    preproc_data['test']  = data_to_seq(X_test, Y_test,
                                        t_lag=t_lag,
                                        t_future_shift=t_future_shift,
                                        t_future_steps=t_future_steps,
                                        t_sw_step=t_sw_step,
                                        X_pad_with=X_test_padding)

    # Prepare validation data (if given)
    if 'valid' in data:
        X_valid, Y_valid = data['valid']

        if standardize:
            X_valid, _, _ = standardize_data(X_valid, X_mean, X_std)
            Y_valid, _, _ = standardize_data(Y_valid, Y_mean, Y_std)

        X_valid_padding = None if seq_restart else np.vstack([X_train, X_test])
        preproc_data['valid'] = data_to_seq(X_valid, Y_valid,
                                            t_lag=t_lag,
                                            t_future_shift=t_future_shift,
                                            t_future_steps=t_future_steps,
                                            t_sw_step=t_sw_step,
                                            X_pad_with=X_valid_padding)

    # Convert multi-dim targets into a list of multiple 1-dim targets
    if multiple_outputs:
        for set_name in ['train', 'valid', 'test']:
            Y = preproc_data[set_name][1]
            Y = Y.reshape((-1, 1, np.prod(Y.shape[1:])))
            preproc_data[set_name][1] = [Y[:,:,i] for i in xrange(Y.shape[2])]

    return preproc_data
