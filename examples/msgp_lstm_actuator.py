"""
MSGP-LSTM regression on Actuator data.
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
from kgp.utils.assemble import load_NN_configs, load_GP_configs, assemble
from kgp.utils.execute import train

# Metrics & losses
from kgp.losses import gen_gp_loss
from kgp.metrics import root_mean_squared_error as RMSE


def main():
    # Load data
    X_train, y_train = load_data('actuator', stop=45.)
    X_valid, y_valid = load_data('actuator', start=45., stop=55.)
    X_test, y_test = load_data('actuator', start=55.)
    data = {
        'train': (X_train, y_train),
        'valid': (X_valid, y_valid),
        'test': (X_test, y_test),
    }

    data = preprocess_data(data,
                           standardize=True,
                           multiple_outputs=True,
                           t_lag=10,
                           t_future_shift=1,
                           t_future_steps=1,
                           t_sw_step=1)

    # Model & training parameters
    nb_train_samples = data['train'][0].shape[0]
    input_shape = data['train'][0].shape[1:]
    nb_outputs = len(data['train'][1])
    gp_input_shape = (1,)
    batch_size = 128
    epochs = 5

    nn_params = {
        'H_dim': 16,
        'H_activation': 'tanh',
        'dropout': 0.1,
    }
    gp_params = {
        'cov': 'SEiso',
        'hyp_lik': -2.0,
        'hyp_cov': [[-0.7], [0.0]],
        'opt': {'cg_maxit': 500, 'cg_tol': 1e-4},
        'grid_kwargs': {'eq': 1, 'k': 1e2},
        'update_grid': True,
    }

    # Retrieve model config
    nn_configs = load_NN_configs(filename='lstm.yaml',
                                 input_shape=input_shape,
                                 output_shape=gp_input_shape,
                                 params=nn_params)
    gp_configs = load_GP_configs(filename='gp.yaml',
                                 nb_outputs=nb_outputs,
                                 batch_size=batch_size,
                                 nb_train_samples=nb_train_samples,
                                 params=gp_params)

    # Construct & compile the model
    model = assemble('GP-LSTM', [nn_configs['1H'], gp_configs['MSGP']])
    loss = [gen_gp_loss(gp) for gp in model.output_layers]
    model.compile(optimizer=Adam(1e-2), loss=loss)

    # Callbacks
    callbacks = [EarlyStopping(monitor='val_mse', patience=10)]

    # Train the model
    history = train(model, data, callbacks=callbacks, gp_n_iter=5,
                    checkpoint='lstm', checkpoint_monitor='val_mse',
                    epochs=epochs, batch_size=batch_size, verbose=2)

    # Finetune the model
    model.finetune(*data['train'],
                   batch_size=batch_size,
                   gp_n_iter=100,
                   verbose=0)

    # Test the model
    X_test, y_test = data['test']
    y_preds = model.predict(X_test)
    rmse_predict = RMSE(y_test, y_preds)
    print('Test predict RMSE:', rmse_predict)


if __name__ == '__main__':
    main()
