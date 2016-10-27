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
from kgp.datasets.data_utils import preprocess_data

# Model assembling and executing
from kgp.utils.assemble import load_NN_configs, assemble
from kgp.utils.execute import train

# Metrics
from kgp.metrics import root_mean_squared_error as RMSE


def main():
    # Load data
    X_train, y_train = load_data('actuator', stop=45.)
    X_test, y_test = load_data('actuator', start=45., stop=90.)
    X_valid, y_valid = load_data('actuator', start=90.)
    data = {
        'train': (X_train, y_train),
        'valid': (X_valid, y_valid),
        'test': (X_test, y_test),
    }

    data = preprocess_data(data,
                           standardize=True,
                           t_lag=10,
                           t_future_shift=1,
                           t_future_steps=1,
                           t_sw_step=1,
                           seq_restart=False)

    # Model & training parameters
    input_shape = data['train'][0].shape[1:]
    output_shape = data['train'][1].shape[1:]
    batch_size = 16
    nb_epoch = 3

    # Retrieve model config
    configs = load_NN_configs(filename='lstm.yaml',
                              input_shape=input_shape,
                              output_shape=output_shape,
                              H_dim=128, H_activation='tanh',
                              dropout=0.1)

    # Construct & compile the model
    model = assemble('LSTM', configs['1H'])
    model.compile(optimizer=Adam(1e-1), loss='mse')

    # Callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

    # Train the model
    history = train(model, data, callbacks=callbacks,
                    checkpoint='lstm', checkpoint_monitor='val_loss',
                    nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)

    # Test the model
    X_test, y_test = data['test']
    y_preds = model.predict(X_test)
    rmse_predict = RMSE(y_test, y_preds)
    print('Test RMSE:', rmse_predict)


if __name__ == '__main__':
    main()
