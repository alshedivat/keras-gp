from __future__ import absolute_import

import os
import sys

_BACKEND = 'gpml'
_ENGINE = 'octave'

if 'GP_BACKEND' in os.environ:
    _BACKEND = os.environ['GP_BACKEND']
    assert _BACKEND in {'gpml'}
else:
    os.environ['GP_BACKEND'] = _BACKEND

if 'GP_ENGINE' in os.environ:
    _ENGINE = os.environ['GP_ENGINE']
    assert _ENGINE in {'matlab', 'octave'}
else:
    os.environ['GP_ENGINE'] = _ENGINE

if _ENGINE == 'matlab':
    if sys.version_info[0] > 2:
        raise ImportError('MATLAB engine does not support Python 3.x. '
                          'Please switch to Octave engine or use Python 2.x.')

if _BACKEND == 'gpml':
    sys.stderr.write('Using GPML backend with ')
    sys.stderr.write('MATLAB ' if _ENGINE == 'matlab' else 'Octave ')
    sys.stderr.write('engine as the default one.\n')
    from .gpml import GPML as GP_BACKEND
else:
    raise Exception('Unknown backend: ' + str(_BACKEND))
