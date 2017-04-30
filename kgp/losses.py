"""
Objectives for Gaussian Process layers.
"""
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def gen_gp_loss(gp):
    """Generate an internal objective, `dlik_dh * H`, for a given GP layer.
    """
    def loss(_, H):
        dlik_dh_times_H = H * K.gather(gp.dlik_dh, gp.batch_ids[:gp.batch_sz])
        return K.sum(dlik_dh_times_H, axis=1, keepdims=True)
    return loss


# Aliases

rmse = RMSE = root_mean_squared_error
