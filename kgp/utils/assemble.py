"""
Utility functions for assembling Keras and KGP models.
"""
import os
import sys
import yaml

import numpy as np

from six.moves import xrange

import keras
from keras.models import Model as kModel
from keras.layers import Input, Dense, Dropout, Masking, Reshape
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils.generic_utils import get_from_module

import kgp
from kgp.models import Model as kgpModel
from kgp.layers import GP


def load_NN_configs(filename, input_shape, output_shape,
                    H_dim=32,
                    H_activation='relu',
                    dropout=0.5,
                    dropout_W=0.0,
                    dropout_U=0.0,
                    batch_norm='null',
                    verbose=0):
    path = os.path.join(kgp.__path__[0], 'configs', filename)
    with open(path) as fp:
        config_templates = fp.read()
    configs = config_templates.format(input_shape=list(input_shape),
                                      output_shape=list(output_shape),
                                      H_dim=H_dim,
                                      H_activation=H_activation,
                                      dropout=dropout,
                                      dropout_W=dropout_W,
                                      dropout_U=dropout_U,
                                      batch_norm=batch_norm)
    if verbose:
        print(configs)
    return yaml.load(configs)


def load_GP_configs(filename, nb_outputs, batch_size, nb_train_samples,
                    cov='SEiso',
                    hyp_lik=-2.0,
                    hyp_cov=[[-0.7], [0.0]],
                    opt={},
                    grid_kwargs={'eq': 1, 'k': 1e2},
                    update_grid=True,
                    verbose=0):
    path = os.path.join(kgp.__path__[0], 'configs', filename)
    with open(path) as fp:
        config_templates = fp.read()
    configs = config_templates.format(nb_outputs=nb_outputs,
                                      batch_size=batch_size,
                                      nb_train_samples=nb_train_samples,
                                      hyp_lik=hyp_lik,
                                      hyp_cov=hyp_cov,
                                      opt=opt,
                                      cov=cov,
                                      grid_kwargs=grid_kwargs,
                                      update_grid=update_grid)
    if verbose:
        print(configs)
    return yaml.load(configs)


def assemble(name, params):
    if name == 'NARX':
        return assemble_narx(params)
    elif name == 'GP-NARX':
        return assemble_gpnarx(*params)
    elif name == 'RNN':
        return assemble_rnn(params)
    elif name == 'GP-RNN':
        return assemble_gprnn(*params)
    elif name == 'LSTM':
        return assemble_rnn(params)
    elif name == 'GP-LSTM':
        return assemble_gprnn(*params)
    elif name == 'GRU':
        return assemble_rnn(params)
    elif name == 'GP-GRU':
        return assemble_gprnn(*params)
    else:
        raise ValueError("Unknown name of the model: %s." % name)


def assemble_narx(params, final_reshape=True):
    '''Construct a NARX model of the form: X-[H1-H2-...-HN]-Y.
    All the H-layers are Dense and optional, i.e., depend on whether they are
    specified in the params dictionary. Here, X is a sequence.
    '''
    # Input layer
    input_shape = params['input_shape']
    inputs = Input(shape=input_shape)

    # Flatten the time dimension
    target_shape = (np.prod(input_shape), )
    previous = Reshape(target_shape)(inputs)

    # Hidden layers
    for layer in params['hidden_layers']:
        Layer = get_from_module(layer['name'], globals(), 'assemble')
        previous = Layer(**layer['config'])(previous)
        if 'dropout' in layer and layer['dropout'] is not None:
            previous = Dropout(layer['dropout'])(previous)
        if 'batch_norm' in layer and layer['batch_norm'] is not None:
            previous = BatchNormalization(**layer['batch_norm'])(previous)

    # Output layer
    output_shape = params['output_shape']
    output_dim = np.prod(output_shape)
    outputs = Dense(output_dim)(previous)

    if final_reshape:
        outputs = Reshape(output_shape)(outputs)

    return kModel(input=inputs, output=outputs)


def assemble_gpnarx(nn_params, gp_params):
    '''Construct an GP-NARX model of the form: X-[H1-H2-...-HN]-GP-Y.
    '''
    # Input layer
    input_shape = nn_params['input_shape']
    inputs = Input(shape=input_shape)

    # NARX transformation
    narx = assemble_narx(nn_params, final_reshape=False)(inputs)

    # Output layer
    outputs = [GP(**gp_params['config'])(narx)
               for _ in xrange(gp_params['nb_outputs'])]

    return kgpModel(input=inputs, output=outputs)


def assemble_rnn(params, final_reshape=True):
    '''Construct an RNN/LSTM/GRU model of the form: X-[H1-H2-...-HN]-Y.
    All the H-layers are optional recurrent layers and depend on whether they
    are specified in the params dictionary.
    '''
    # Input layer
    input_shape = params['input_shape']
    inputs = Input(shape=input_shape)

    # Masking layer
    previous = Masking(mask_value=0.0)(inputs)

    # Hidden layers
    for layer in params['hidden_layers']:
        Layer = get_from_module(layer['name'], globals(), 'assemble')
        previous = Layer(**layer['config'])(previous)
        if 'dropout' in layer and layer['dropout'] is not None:
            previous = Dropout(layer['dropout'])(previous)
        if 'batch_norm' in layer and layer['batch_norm'] is not None:
            previous = BatchNormalization(**layer['batch_norm'])(previous)

    # Output layer
    output_shape = params['output_shape']
    output_dim = np.prod(output_shape)
    outputs = Dense(output_dim)(previous)

    if final_reshape:
        outputs = Reshape(output_shape)(outputs)

    return kModel(input=inputs, output=outputs)


def assemble_gprnn(nn_params, gp_params):
    '''Construct an GP-RNN/LSTM/GRU model of the form: X-[H1-H2-...-HN]-GP-Y
    '''
    # Input layer
    input_shape = nn_params['input_shape']
    inputs = Input(shape=input_shape)

    # NARX transformation
    rnn = assemble_rnn(nn_params, final_reshape=False)(inputs)

    # Output layer
    outputs = [GP(**gp_params['config'])(rnn)
               for _ in xrange(gp_params['nb_outputs'])]

    return kgpModel(input=inputs, output=outputs)
