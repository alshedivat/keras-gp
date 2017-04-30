Quick tutorial
==============

KGP extends [Keras 2](https://github.com/fchollet/keras) API by adding a [GP layer](https://github.com/alshedivat/kgp/blob/examples/kgp/layers.py), extending the `Model` class.
This tutorial walks you through the main components of the KGP library using GP-MLP as an example.

### Constructing GP-MLP

We start with constructing a simple 2 hidden layer MLP model using [Keras functional API](https://keras.io/getting-started/functional-api-guide/):

```python
# Keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout

# Define MLP
inputs = Input(shape=input_shape)
hidden = Dense(512, activation='relu')(hidden)
hidden = Dropout(0.5)(hidden)
hidden = Dense(64, activation='relu')(hidden)
hidden = Dense(2, activation='relu')(hidden)
```

At this point, `hidden` is a symbolic representation of our MLP-embedding of the inputs into the hidden space.
Now, we add a GP-layer and construct the model:
```python
# KGP
from kgp.models import Model
from kgp.layers import GP

MSGP = GP(
    hyp={
        'lik': np.log(0.3),
        'mean': [],
        'cov': [[0.5], [1.0]],
    },
    inf='infGrid', dlik='dlikGrid',
    opt={'cg_maxit': 2000, 'cg_tol': 1e-6},
    mean='meanZero', cov='covSEiso',
    update_grid=1,
    grid_kwargs={'eq': 1, 'k': 100.},
    batch_size=128,
    nb_train_samples=1024)

gp = MSGP(hidden)
model = Model(inputs=inputs, outputs=[gp])
```
At this stage, there is literally no difference between constructing a GP-based model and the standard Keras functional API.
The `GP` layer has a number of parameters that you can look up in the documentation.

**Notes:**
- Instead of `keras.models.Model`, we used `kgp.models.Model` class that extends the former with GP-related functionality.
- Here, we used a 2-layer MLP as an example. It is straightforward to change it to your favorite `ResNet-xx` or `DenseNet-xx`.


### The GP layer under the hood

Before we proceed to compiling and training the model, here are a few things to note.
First, the current implementation of the `GP` layer is based on the [GPML library](https://github.com/alshedivat/gpml), i.e., whenever we need to compute the negative log marginal likelihood (NLML) loss, make predictions, optimize GP-hypers, or get the derivatives of the loss w.r.t. to the GP inputs, we call GPML.
Calls to GPML library are abstracted away by the `GP_BACKEND` (you can look up how it is implemented [here](https://github.com/alshedivat/kgp/blob/master/kgp/backend/gpml.py)).
Importantly, since GP functionality is implemented by an external library that runs on MATLAB or Octave, they are effectively *not* a part of the computation graph constructed by Keras.

So, how do we optimize the model then? The trick is that we create placeholder variables in the computation graph for the loss and the gradient of the loss w.r.t. to inputs to the GP (denoted `dlik_dh`).
On the forward pass, we run the GP backend, compute the necessary quantities, and update values for the loss and the gradient placeholders.
This is done by the [UpdateGP callback](https://github.com/alshedivat/kgp/blob/master/kgp/callbacks.py) that is automatically added to the list of callbacks in `kgp.models.Model.fit`.
After these updates, Keras takes care of the backprop through the neural part of the model.
To make this work, we define a specific loss function for Keras to use for backprop:

```python
# The function is defined in `kgp.losses`
def gen_gp_loss(gp):
    """Generate an internal objective, `dlik_dh * H`, for a given GP layer.
    """
    def loss(_, H):
        dlik_dh_times_H = H * K.gather(gp.dlik_dh,      gp.batch_ids[:gp.batch_sz])
        return K.sum(dlik_dh_times_H, axis=1, keepdims=True)
    return loss
```

Note that this loss is meaningless, because it is constructed as an intermediate step of the backprop on the likelihood objective: `dlik_dx = dlik_dh dh_dx`, where `dlik_dh` is returned by GPML and `dh_dx` is computed by Keras via backprop.
Anyway, by using this trick, we can seamlessly use Keras optimizers and train the model end-to-end.


### Compiling & training

Now, we are ready to define the loss function and compile GP-MLP:

```python
# Metrics & losses
from kgp.losses import gen_gp_loss
from keras.optimizers import Adam

loss = [gen_gp_loss(gp) for gp in model.output_layers]
model.compile(optimizer=Adam(1e-4), loss=loss)
```

After the model is compiled, we can train it as any regular Keras model:

```python
model.fit(X_train, y_train,
          validation_data=validation_data,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          verbose=verbose)
```

**Note:** When the `batch_size` is smaller than the number of samples in the training set, the model is automatically trained in the semi-stochastic regime (for details, see [our paper](https://arxiv.org/abs/1610.08936)).
Otherwise, the model is trained in the full-batch fashion.

### Further resources

KGP provides additional wrapper scripts for running experiments in `kgp.utils.experiment`.
More examples are given in the [examples folder](https://github.com/alshedivat/kgp/tree/master/examples).
