"""
LSTM regression on Actuator data.
"""
from __future__ import print_function

import numpy as np
np.random.seed(42)

# Keras
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping

# Dataset interfaces
from kgp.datasets.sysid import load_data
from kgp.datasets.data_utils import data_to_seq, standardize_data

# Model assembling and executing
from kgp.utils.assemble import load_NN_configs, assemble
from kgp.utils.experiment import train

# Metrics
from kgp.metrics import root_mean_squared_error as RMSE


def main():
    # Load data
    X, y = load_data('actuator', use_targets=False)
    X_seq, y_seq = data_to_seq(X, y,
        t_lag=32, t_future_shift=1, t_future_steps=1, t_sw_step=1)

    # Split
    train_end = int((45. / 100.) * len(X_seq))
    test_end = int((90. / 100.) * len(X_seq))
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_test, y_test = X_seq[train_end:test_end], y_seq[train_end:test_end]
    X_valid, y_valid = X_seq[test_end:], y_seq[test_end:]

    data = {
        'train': [X_train, y_train],
        'valid': [X_valid, y_valid],
        'test': [X_test, y_test],
    }

    # Model & training parameters
    input_shape = list(data['train'][0].shape[1:])
    output_shape = list(data['train'][1].shape[1:])
    batch_size = 16
    epochs = 5

    nn_params = {
        'H_dim': 32,
        'H_activation': 'tanh',
        'dropout': 0.1,
    }

    # Retrieve model config
    configs = load_NN_configs(filename='lstm.yaml',
                              input_shape=input_shape,
                              output_shape=output_shape,
                              params=nn_params)

    # Construct & compile the model
    model = assemble('LSTM', configs['1H'])
    model.compile(optimizer=Adam(1e-1), loss='mse')

    # Callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

    # Train the model
    history = train(model, data, callbacks=callbacks,
                    checkpoint='lstm', checkpoint_monitor='val_loss',
                    epochs=epochs, batch_size=batch_size, verbose=2)

    # Test the model
    X_test, y_test = data['test']
    y_preds = model.predict(X_test)
    rmse_predict = RMSE(y_test, y_preds)
    print('Test RMSE:', rmse_predict)


if __name__ == '__main__':
    main()
