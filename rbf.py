import numpy as np

def _pow3(x):
    return x*x*x

def _dpow3(x):
    return 3*x*x

def _ddpow3(x):
    return 6*x

pow3 = np.vectorize(_pow3)
dpow3 = np.vectorize(_dpow3)
ddpow3 = np.vectorize(_ddpow3)
