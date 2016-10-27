import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='kgp',
    version='0.3',
    packages=find_packages(),
    description='GP models for Keras.',
    long_description=read('README.md'),
    author='Maruan Al-Shedivat',
    author_email='maruan@alshedivat.com',
    url='https://github.com/alshedivat/kgp',
    license='MIT',
    dependency_links=['git+http://github.com/Theano/Theano.git#egg=Theano'],
    install_requires=['numpy>=1.5', 'keras>=1.0', 'theano', 'pyyaml', 'six'],
    extras_require = {
        'matlab_engine':  ['matlab'],
        'octave_engine':  ['oct2py'],
    },
    package_data = {
        '': ['*.yaml', '*.m'],
    }
)
