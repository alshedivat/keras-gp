from six.moves import xrange

import numpy as np

import kgp
from kgp.losses import gen_gp_loss
from kgp.utils.assemble import *

# Test parameters
N = 256
batch_size = 64
input_shape = (32, 10)
output_shape = (2, )
optimizer = 'rmsprop'

# Load configs
narx_configs = load_NN_configs(filename='narx.yaml',
                               input_shape=input_shape,
                               output_shape=output_shape)
lstm_configs = load_NN_configs(filename='lstm.yaml',
                               input_shape=input_shape,
                               output_shape=output_shape)
rnn_configs = load_NN_configs(filename='rnn.yaml',
                              input_shape=input_shape,
                              output_shape=output_shape)
gru_configs = load_NN_configs(filename='gru.yaml',
                              input_shape=input_shape,
                              output_shape=output_shape)
gp_configs = load_GP_configs(filename='gp.yaml',
                             nb_outputs=np.prod(output_shape),
                             batch_size=batch_size,
                             nb_train_samples=N)

def test_assemble_narx():
    for i in xrange(3):
        model = assemble('NARX', narx_configs[str(i) + 'H'])
        model.compile(optimizer=optimizer, loss='mse')
        assert model.built

def test_assemble_gpnarx():
    for gp_type in ['GP', 'MSGP']:
        model = assemble('GP-NARX', [narx_configs['1H'], gp_configs[gp_type]])
        loss = [gen_gp_loss(gp) for gp in model.output_layers]
        model.compile(optimizer=optimizer, loss=loss)
        assert model.built

def test_assemble_rnn():
    for i in xrange(1, 3):
        model = assemble('RNN', rnn_configs[str(i) + 'H'])
        model.compile(optimizer=optimizer, loss='mse')
        assert model.built

def test_assemble_gprnn():
    for gp_type in ['GP', 'MSGP']:
        model = assemble('GP-RNN', [rnn_configs['1H'], gp_configs[gp_type]])
        loss = [gen_gp_loss(gp) for gp in model.output_layers]
        model.compile(optimizer=optimizer, loss=loss)
        assert model.built

def test_assemble_lstm():
    for i in xrange(1, 3):
        model = assemble('LSTM', lstm_configs[str(i) + 'H'])
        model.compile(optimizer=optimizer, loss='mse')
        assert model.built

def test_assemble_gplstm():
    for gp_type in ['GP', 'MSGP']:
        model = assemble('GP-LSTM', [lstm_configs['1H'], gp_configs[gp_type]])
        loss = [gen_gp_loss(gp) for gp in model.output_layers]
        model.compile(optimizer=optimizer, loss=loss)
        assert model.built

def test_assemble_gru():
    for i in xrange(1, 3):
        model = assemble('GRU', gru_configs[str(i) + 'H'])
        model.compile(optimizer=optimizer, loss='mse')
        assert model.built

def test_assemble_gpgru():
    for gp_type in ['GP', 'MSGP']:
        model = assemble('GP-GRU', [gru_configs['1H'], gp_configs[gp_type]])
        loss = [gen_gp_loss(gp) for gp in model.output_layers]
        model.compile(optimizer=optimizer, loss=loss)
        assert model.built
