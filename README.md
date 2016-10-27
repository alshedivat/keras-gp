Gaussian Processes for Keras
----------------------------

KGP extends [Keras](https://github.com/fchollet/keras/) with Gaussian Process (GP) layers.
It allows to build flexible GP models with kernels structured with [deep](https://arxiv.org/abs/1511.02222) and [recurrent]() networks built with Keras.
The structured part of the model (the neural net) runs on [Theano](http://deeplearning.net/software/theano/) or [Tensorflow](https://www.tensorflow.org/).
The GP layers use a custom backend based on [GPML 4.0](http://www.gaussianprocess.org/gpml/code/matlab/doc/) library.
The models can be trained in stages or jointly, using full-batch or semi-stochastic optimization approaches (see [the original paper]()).

KGP is compatible with: Python **2.7-3.5**.

## Getting started

KGP allows to build models in the same fashion as Keras, using the [functional API](https://keras.io/getting-started/functional-api-guide/).
For example, a simple GP-RNN model can be built and compiled in just a few lines of code:

```python
from keras.layers import Input, SimpleRNN
from keras.optimizers import Adam

from kgp.layers import GP
from kgp.models import Model
from kgp.objectives import gen_gp_loss

input_shape = (10, 2)  # 10 time steps, 2 dimensions
batch_size = 32
nb_train_samples = 512
gp_hypers = {'lik': -2.0, 'cov': [[-0.7], [0.0]]}

# Build the model
inputs = Input(shape=input_shape)
rnn = SimpleRNN(32)(inputs)
gp = GP(gp_hypers,
        batch_size=batch_size,
        nb_train_samples=nb_train_samples)
outputs = [gp(rnn)]
model = Model(input=inputs, output=outputs)

# Compile the model
loss = [gen_gp_loss(gp) for gp in model.output_layers]
model.compile(optimizer=Adam(1e-2), loss=loss)
```

Note that KGP models support arbitrary off-the-shelf optimizers from Keras.

Further resources:
- [Tutorials](https://github.com/alshedivat/kgp/tutorials) (*to be added soon*).
- A couple of [examples](https://github.com/alshedivat/kgp/examples).


## Installation

KGP depends on [Keras](https://github.com/fchollet/keras/) and requires either Theano or Tensorflow being installed.
The GPML backend requires either MATLAB or Octave and a corresponding Python interface package: [Oct2Py](https://blink1073.github.io/oct2py/) for Octave or the [MATLAB engine for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).

The requirements can be installed via [pip](https://pypi.python.org/pypi/pip) as follows (use `sudo` if necessary):

```bash
$ pip install -r requirements.txt
```

To install the package, clone the repository and run `setup.py` as follows:

```bash
$ git clone --recursive https://github.com/alshedivat/kgp
$ cd kgp
$ python setup.py develop [--user]
```

The `--user` flag (optional) will install the package for a given user only.
Note that recursive clone is required to get GPML library as a submodule.

## Contribution

Contributions are sought. Bug reports are welcome.

## Citation

```bibtex
@article{alshedivat2016srk,
  title={Learning Scalable Deep Kernels with Recurrent Structure},
  author={Al-Shedivat, Maruan and Wilson, Andrew G and Saatchi, Yunus and Hu, Zhiting and Xing, Eric P},
  year={2016}
}
```

## License

MIT (for details, please refer to [LICENSE](https://github.com/alshedivat/kgp/blob/master/LICENSE))

Copyright (c) 2016 Maruan Al-Shedivat
