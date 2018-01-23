import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='kgp',
    version='0.3.2',
    packages=find_packages(),
    description='GP models for Keras.',
    long_description=read('README.md'),
    author='Maruan Al-Shedivat',
    author_email='maruan@alshedivat.com',
    url='https://github.com/alshedivat/keras-gp',
    license='MIT',
    install_requires=[
        'numpy>=1.11', 'keras==2.1.3', 'tensorflow>=1.0', 'pyyaml', 'six',
    ],
    extras_require = {
        'matlab_engine':  ['matlab'],
        'octave_engine':  ['oct2py==3.9.2'],
    },
    package_data = {
        '': ['*.yaml', '*.m'],
    }
)
