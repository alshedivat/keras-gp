import numpy as np

from six.moves import xrange

from keras.layers import Input, Dense, LSTM

from kgp.models import Model
from kgp.layers import GP
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


def test_compile():
    model = build_model()

    # Generate losses for GP outputs
    loss = [gen_gp_loss(gp) for gp in model.output_gp_layers]

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss)


def test_fit(epochs=10, seed=42):
    rng = np.random.RandomState(seed)

    for nb_outputs in [1, 2]:
        # Generate dummy data
        X_tr = rng.normal(size=(N, input_shape[0], input_shape[1]))
        Y_tr = [rng.normal(size=(N, 1)) for _ in xrange(nb_outputs)]

        # Build & compile the model
        model = build_model(nb_outputs)
        loss = [gen_gp_loss(gp) for gp in model.output_gp_layers]
        model.compile(optimizer=optimizer, loss=loss)

        # Train the model
        model.fit(X_tr, Y_tr,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=2)


def test_finetune(gp_n_iter=10, seed=42):
    rng = np.random.RandomState(seed)

    for nb_outputs in [1, 2]:
        # Generate dummy data
        X_tr = rng.normal(size=(N, input_shape[0], input_shape[1]))
        Y_tr = [rng.normal(size=(N, 1)) for _ in xrange(nb_outputs)]

        # Build & compile the model
        model = build_model(nb_outputs)
        loss = [gen_gp_loss(gp) for gp in model.output_gp_layers]
        model.compile(optimizer=optimizer, loss=loss)

        # Finetune the model
        model.finetune(X_tr, Y_tr,
                       batch_size=batch_size,
                       gp_n_iter=gp_n_iter,
                       verbose=0)


def test_evaluate(seed=42):
    rng = np.random.RandomState(seed)

    for nb_outputs in [1, 2]:
        # Generate dummy data
        X_ts = rng.normal(size=(N, input_shape[0], input_shape[1]))
        Y_ts = [rng.normal(size=(N, 1)) for _ in xrange(nb_outputs)]

        # Build & compile the model
        model = build_model(nb_outputs)
        loss = [gen_gp_loss(gp) for gp in model.output_gp_layers]
        model.compile(optimizer=optimizer, loss=loss)

        # Evaluate the model
        nlml = model.evaluate(X_ts, Y_ts, batch_size=batch_size, verbose=0)


def test_predict(seed=42):
    rng = np.random.RandomState(seed)

    for nb_outputs in [1, 2]:
        # Generate dummy data
        X_tr = rng.normal(size=(N, input_shape[0], input_shape[1]))
        Y_tr = [rng.normal(size=(N, 1)) for _ in xrange(nb_outputs)]
        X_ts = rng.normal(size=(N, input_shape[0], input_shape[1]))
        Y_ts = [rng.normal(size=(N, 1)) for _ in xrange(nb_outputs)]

        # Build & compile the model
        model = build_model(nb_outputs)
        loss = [gen_gp_loss(gp) for gp in model.output_gp_layers]
        model.compile(optimizer=optimizer, loss=loss)

        # Predict
        Y_pr = model.predict(X_ts, X_tr, Y_tr,
                             batch_size=batch_size, verbose=0)
        assert type(Y_pr) is list
        assert len(Y_pr) == len(Y_ts)
        assert np.all([(yp.shape == yt.shape) for yp, yt in zip(Y_pr, Y_ts)])
