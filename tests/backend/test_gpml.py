"""
Tests for GPML backend.
"""
import os
import sys
import pytest
import numpy as np

from kgp.backend import GP_BACKEND

AVAILABLE_ENGINES = set()
try:
    import oct2py
    AVAILABLE_ENGINES.add('octave')
except ImportError:
    pass
try:
    import matlab.engine
    AVAILABLE_ENGINES.add('matlab')
except ImportError:
    pass
if not AVAILABLE_ENGINES:
    sys.stderr.write("No computational engines is available. "
                     "GPML tests will be skipped.\n")

gp_test_config = {
    'input_dim': 1,
    'hyp': {'lik': -2.0, 'mean': [], 'cov': np.array([[-0.7], [0.0]])},
    'opt': {},
    'inf': 'infExact',
    'mean': 'meanZero',
    'cov': 'covSEiso',
    'lik': 'likGauss',
    'dlik': 'dlikExact',
    'verbose': 0,
}

msgp_test_config = {
    'input_dim': 1,
    'hyp': {'lik': -2.0, 'mean': [], 'cov': np.array([[-0.7], [0.0]])},
    'opt': {'cg_maxit': 500, 'cg_tol': 1e-4},
    'inf': 'infGrid',
    'mean': 'meanZero',
    'cov': 'covSEiso',
    'lik': 'likGauss',
    'dlik': 'dlikGrid',
    'grid_kwargs': {'eq': 1, 'k': 100},
    'verbose': 0,
}

def configure(engine):
    gp = GP_BACKEND(engine)

    gp.configure(**gp_test_config)
    gp.configure(**msgp_test_config)

@pytest.mark.parametrize('engine', AVAILABLE_ENGINES)
def test_update_data(engine, N=100, D=1, seed=42):
    gp = GP_BACKEND(engine)

    rng = np.random.RandomState(seed)
    X = rng.normal(size=(N, D))
    y = rng.normal(size=(N, D))

    # Test updating data
    gp.configure(**gp_test_config)
    gp.update_data('tmp', X, y)
    gp.eng.eval("assert(exist('X_tmp','var') ~= 0)")
    gp.eng.eval("assert(exist('y_tmp','var') ~= 0)")

@pytest.mark.parametrize('engine', AVAILABLE_ENGINES)
def test_update_grid(engine, N=100, D=1, seed=42):
    gp = GP_BACKEND(engine)

    rng = np.random.RandomState(seed)
    X = rng.normal(size=(N, D))

    # Test updating the grid
    gp.configure(**msgp_test_config)
    gp.update_data('tmp', X)
    gp.update_grid('tmp')
    gp.eng.eval("assert(exist('xg','var') ~= 0)")

@pytest.mark.parametrize('engine', AVAILABLE_ENGINES)
def test_evaluate(engine, N=100, D=1, seed=42):
    gp = GP_BACKEND(engine)

    rng = np.random.RandomState(seed)
    X = rng.normal(size=(N, D))
    y = rng.normal(size=(N, D))
    gp.update_data('tmp', X, y)

    # Evaluate GP
    gp.configure(**gp_test_config)
    nlZ = gp.evaluate('tmp')

    # Evaluate MSGP
    gp.configure(**msgp_test_config)
    gp.update_grid('tmp')
    nlZ = gp.evaluate('tmp')

@pytest.mark.parametrize('engine', AVAILABLE_ENGINES)
def test_predict(engine, N=100, D=1, seed=42):
    gp = GP_BACKEND(engine)

    rng = np.random.RandomState(seed)
    X_tr = rng.normal(size=(N, D))
    y_tr = rng.normal(size=(N, D))
    X_tst = rng.normal(size=(N, D))
    gp.update_data('tr', X_tr, y_tr)

    # Predict using GP
    gp.configure(**gp_test_config)
    ymu, ys2 = gp.predict(X_tst, return_var=True)

    assert type(ymu) is np.ndarray and type(ys2) is np.ndarray
    assert ymu.shape == ys2.shape

    # Predict using MSGP
    gp.configure(**msgp_test_config)
    gp.update_grid('tr')
    ymu, ys2 = gp.predict(X_tst, return_var=True)

    assert type(ymu) is np.ndarray and type(ys2) is np.ndarray
    assert ymu.shape == ys2.shape

@pytest.mark.parametrize('engine', AVAILABLE_ENGINES)
def test_train(engine, N=100, D=1, seed=42):
    gp = GP_BACKEND(engine)

    rng = np.random.RandomState(seed)
    X_tr = rng.normal(size=(N, D))
    y_tr = rng.normal(size=(N, D))

    # Train a GP
    gp.configure(**gp_test_config)
    hyp = gp.train(5, X_tr, y_tr)

    assert type(hyp) is dict
    assert set(hyp.keys()) == set(gp_test_config['hyp'].keys())

    # Train an MSGP
    gp.configure(**msgp_test_config)
    gp.update_data('tr', X_tr, y_tr)
    gp.update_grid('tr')
    hyp = gp.train(5)

    assert type(hyp) is dict
    assert set(hyp.keys()) == set(gp_test_config['hyp'].keys())

@pytest.mark.parametrize('engine', AVAILABLE_ENGINES)
def test_get_dlik_dx(engine, N=100, D=1, seed=42):
    gp = GP_BACKEND(engine)

    rng = np.random.RandomState(seed)
    X = rng.normal(size=(N, D))
    y = rng.normal(size=(N, D))
    gp.update_data('tmp', X, y)

    # GP
    gp.configure(**gp_test_config)
    dlik_dx = gp.get_dlik_dx('tmp')
    assert type(dlik_dx) is np.ndarray
    assert dlik_dx.shape == (N, D)

    # MSGP
    gp.configure(**msgp_test_config)
    gp.update_grid('tmp')
    dlik_dx = gp.get_dlik_dx('tmp')
    assert type(dlik_dx) is np.ndarray
    assert dlik_dx.shape == (N, D)

@pytest.mark.parametrize('engine', AVAILABLE_ENGINES)
def test_grad_checks(engine):
    gp = GP_BACKEND(engine)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    grad_checks_path = os.path.join(dir_path, 'matlab')
    gp.eng.addpath(grad_checks_path)

    gp.eng.eval('grad_check_covSEiso')
    gp.eng.eval('grad_check_covSEard')
    gp.eng.eval('grad_check_likExact')
    gp.eng.eval('grad_check_likGrid')
