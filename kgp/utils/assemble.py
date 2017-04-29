"""
Utility functions for assembling Keras and KGP models.
"""
import os
import sys
import yaml

import numpy as np

from six.moves import xrange

import keras
from keras import layers, regularizers
from keras.models import Model as KerasModel

import kgp
from kgp.models import Model
from kgp.layers import GP


NN_DEFAULT_PARAMS = {
    'H_dim': 32,
    'H_activation': 'relu',
    'dropout': 0.5,
    'dropout_W': 0.0,
    'dropout_U': 0.0,
    'batch_norm': 'null',
    'stateful': False,
}


GP_DEFAULT_PARAMS = {
    'cov': 'SEiso',
    'hyp_lik': -2.0,
    'hyp_cov': [[-0.7], [0.0]],
    'opt': {},
    'grid_kwargs': {'eq': 1, 'k': 1e2},
    'update_grid': True,
}


def load_NN_configs(filename, input_shape, output_shape,
                    params=None, verbose=0):
    path = os.path.join(kgp.__path__[0], 'configs', filename)
    with open(path) as fp:
        config_templates = fp.read()
    if params is None:
        params = NN_DEFAULT_PARAMS
    else:
        temp = NN_DEFAULT_PARAMS.copy()
        temp.update(params)
        params = temp
    configs = config_templates.format(input_shape=list(input_shape),
                                      output_shape=list(output_shape),
                                      **params)
    if verbose:
        print(configs)
    return yaml.load(configs)


def load_GP_configs(filename, nb_outputs, batch_size, nb_train_samples,
                    params=None, verbose=0):
    path = os.path.join(kgp.__path__[0], 'configs', filename)
    with open(path) as fp:
        config_templates = fp.read()
    if params is None:
        params = GP_DEFAULT_PARAMS
    else:
        temp = GP_DEFAULT_PARAMS.copy()
        temp.update(params)
        params = temp
    configs = config_templates.format(nb_outputs=nb_outputs,
                                      batch_size=batch_size,
                                      nb_train_samples=nb_train_samples,
                                      **params)
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
    """Construct a NARX model of the form: X-[H1-H2-...-HN]-Y.
    All the H-layers are Dense and optional, i.e., depend on whether they are
    specified in the params dictionary. Here, X is a sequence.
    """
    # Input layer
    input_shape = params['input_shape']
    inputs = layers.Input(shape=input_shape)

    # Flatten the time dimension
    target_shape = (np.prod(input_shape), )
    previous = layers.Reshape(target_shape)(inputs)

    # Hidden layers
    for layer in params['hidden_layers']:
        Layer = layers.deserialize(
            {'class_name': layer['name'], 'config': layer['config']})
        previous = Layer(previous)
        if 'dropout' in layer and layer['dropout'] is not None:
            previous = layers.Dropout(layer['dropout'])(previous)
        if 'batch_norm' in layer and layer['batch_norm'] is not None:
            previous = layers.BatchNormalization(**layer['batch_norm'])(previous)

    # Output layer
    output_shape = params['output_shape']
    output_dim = np.prod(output_shape)
    outputs = layers.Dense(output_dim)(previous)

    if final_reshape:
        outputs = layers.Reshape(output_shape)(outputs)

    return KerasModel(inputs=inputs, outputs=outputs)


def assemble_gpnarx(nn_params, gp_params):
    """Construct an GP-NARX model of the form: X-[H1-H2-...-HN]-GP-Y.
    """
    # Assemble NARX
    NARX = assemble_narx(nn_params, final_reshape=False)

    # Inputs and NARX layer
    inputs = NARX.inputs
    narx = NARX(inputs)

    # Output GP layers
    outputs = [GP(**gp_params['config'])(narx)
               for _ in xrange(gp_params['nb_outputs'])]

    return Model(inputs=inputs, outputs=outputs)


def assemble_rnn(params, final_reshape=True):
    """Construct an RNN/LSTM/GRU model of the form: X-[H1-H2-...-HN]-Y.
    All the H-layers are optional recurrent layers and depend on whether they
    are specified in the params dictionary.
    """
    # Input layer
    input_shape = params['input_shape']
    inputs = layers.Input(shape=input_shape)
    # inputs = layers.Input(batch_shape=[20] + list(input_shape))

    # Masking layer
    previous = layers.Masking(mask_value=0.0)(inputs)

    # Hidden layers
    for layer in params['hidden_layers']:
        Layer = layers.deserialize(
            {'class_name': layer['name'], 'config': layer['config']})
        previous = Layer(previous)
        if 'dropout' in layer and layer['dropout'] is not None:
            previous = layers.Dropout(layer['dropout'])(previous)
        if 'batch_norm' in layer and layer['batch_norm'] is not None:
            previous = layers.BatchNormalization(**layer['batch_norm'])(previous)

    # Output layer
    output_shape = params['output_shape']
    output_dim = np.prod(output_shape)
    outputs = layers.Dense(output_dim)(previous)

    if final_reshape:
        outputs = layers.Reshape(output_shape)(outputs)

    return KerasModel(inputs=inputs, outputs=outputs)


def assemble_gprnn(nn_params, gp_params):
    """Construct an GP-RNN/LSTM/GRU model of the form: X-[H1-H2-...-HN]-GP-Y
    """
    # Assemble RNN
    RNN = assemble_rnn(nn_params, final_reshape=False)

    # Inputs and RNN layer
    inputs = RNN.inputs
    rnn = RNN(inputs)

    # Output GP layers
    outputs = [GP(**gp_params['config'])(rnn)
               for _ in xrange(gp_params['nb_outputs'])]

    return Model(inputs=inputs, outputs=outputs)
