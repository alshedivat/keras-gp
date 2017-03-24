"""
Computational engines used by GP backends.
"""
import numpy as np


class Engine(object):
    """The base class for computational engines.
    """
    def addpath(self, path):
        self._eng.addpath(path)

    def eval(self, expr, verbose):
        """Evaluate an expression.
        """
        raise NotImplementedError

    def push(self, name, var):
        """Push a variable into the engine session under the given name.
        """
        raise NotImplementedError

    def pull(self, name):
        """Pull a variable from the engine session.
        """
        raise NotImplementedError


class MATLABEngine(Engine):
    def __init__(self):
        import matlab.engine
        from matlab import double as matdouble

        from StringIO import StringIO

        self._matarray = matdouble
        self._eng = matlab.engine.start_matlab()
        self._devnull = StringIO()

    def push(self, name, var):
        # Convert np.ndarrays into matlab.doubles and push into the workspace
        if type(var) is np.ndarray:
            self._eng.workspace[name] = self._matarray(var.tolist())
        elif type(var) is dict:
            var_copy = var.copy()
            for k, v in var_copy.iteritems():
                if type(v) is np.ndarray:
                    var_copy[k] = self._matarray(v.tolist())
            self._eng.workspace[name] = var_copy
        elif type(var) in {list, int, float}:
            self._eng.workspace[name] = var
        else:
            raise ValueError("Unknown type (%s) variable being pushed "
                             "into the MATLAB session." % type(var))

    def pull(self, name):
        var = self._eng.workspace[name]
        if type(var) is self._matarray:
            var = np.asarray(var)
        elif type(var) is dict:
            for k, v in var.iteritems():
                if type(v) is self._matarray:
                    var[k] = np.asarray(v)
        return var

    def eval(self, expr, verbose=0):
        assert type(expr) is str
        stdout = None if verbose else self._devnull
        self._eng.eval(expr, nargout=0, stdout=stdout)


class OctaveEngine(Engine):
    def __init__(self, jit_enable=True):
        from oct2py import Oct2Py
        from oct2py import Struct

        self._struct = Struct
        self._eng = Oct2Py()
        if jit_enable:
            self._eng.eval('jit_enable(1)', verbose=0)
        self._eng.eval('pkg load statistics', verbose=0)

    def push(self, name, var):
        if type(var) is np.ndarray and var.dtype == 'float32':
            # Octave does not support `sparse matrix * dense matrix` operations
            # for float32 type, hence we cast `var` to float64 before pushing
            # into the Octave session
            var = var.astype('float64')
        self._eng.push(name, var)

    def pull(self, name):
        var = self._eng.pull(name)
        if type(var) is self._struct:
            var = dict(var)
        return var

    def eval(self, expr, verbose=0):
        assert type(expr) is str
        self._eng.eval(expr, verbose=verbose)

