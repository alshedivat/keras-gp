"""
Objectives for Gaussian Process layers.
"""
from keras import backend as K

def gen_gp_loss(gp):
    def loss(_, H):
        objective = K.sum(H * K.gather(gp.dlik_dh, gp.batch_ids[:gp.batch_sz]),
                          axis=1)
        return K.reshape(objective, (-1, 1))
    return loss
