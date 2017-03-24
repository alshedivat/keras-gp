import numpy as np

from six.moves import xrange

from keras.layers import Input, Dense, LSTM

from kgp.models import Model
from kgp.layers import GP
from kgp.callbacks import *
from kgp.losses import gen_gp_loss

N = 256
batch_size = 64
input_shape = (32, 10)
lstm_dim = 16
dense_dim = 1
optimizer = 'adam'

gp_test_config = {
    'hyp': {'lik': -2.0, 'cov': np.array([[-0.7], [0.0]])},
    'nb_train_samples': N,
    'batch_size': batch_size,
    'opt': {},
    'inf': 'infExact',
    'mean': 'meanZero',
    'cov': 'covSEiso',
    'lik': 'likGauss',
    'dlik': 'dlikExact',
    'verbose': 0,
}


def build_model(nb_outputs=2):
    inputs = Input(shape=input_shape)

    # Neural transformations
    lstm = LSTM(lstm_dim)(inputs)
    dense = Dense(dense_dim)(lstm)

    # GP outputs
    outputs = [GP(**gp_test_config)(dense) for _ in xrange(nb_outputs)]

    # Build the model
    model = Model(inputs=inputs, outputs=outputs)

    return model


def test_update_gp(seed=42):
    rng = np.random.RandomState(seed)

    for nb_outputs in [1, 2]:
        # Generate dummy data
        X_tr = rng.normal(size=(N, input_shape[0], input_shape[1]))
        Y_tr = [rng.normal(size=(N, 1)) for _ in xrange(nb_outputs)]
        X_val = rng.normal(size=(N, input_shape[0], input_shape[1]))
        Y_val = [rng.normal(size=(N, 1)) for _ in xrange(nb_outputs)]

        # Build & compile the model
        model = build_model(nb_outputs)
        loss = [gen_gp_loss(gp) for gp in model.output_gp_layers]
        model.compile(optimizer=optimizer, loss=loss)

        # Setup the callback
        update_gp_callback = UpdateGP((X_tr, Y_tr),
                                      val_ins=(X_val, Y_val),
                                      batch_size=batch_size)
        update_gp_callback.set_model(model)

        # Test the callback
        epoch_logs, batch_logs = {}, {}
        batch_logs['size'] = batch_size
        batch_logs['ids'] = np.arange(batch_size)
        update_gp_callback.on_epoch_begin(1, epoch_logs)
        update_gp_callback.on_batch_begin(1, batch_logs)
        update_gp_callback.on_epoch_end(1, epoch_logs)

        assert 'gp_update_elapsed' in epoch_logs
        assert 'val_nlml' in epoch_logs
        assert 'val_mse' in epoch_logs


def test_timer():
    # Setup the callback
    timer_callback = Timer()

    # Test the callback
    epoch_logs= {}
    timer_callback.on_epoch_begin(1, epoch_logs)
    timer_callback.on_batch_begin(1)
    timer_callback.on_batch_end(1)
    timer_callback.on_epoch_end(1, epoch_logs)

    assert 'epoch_elapsed' in epoch_logs
    assert 'batch_elapsed_avg' in epoch_logs
