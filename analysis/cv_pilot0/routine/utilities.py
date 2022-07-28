import numpy as np


def norm(a):
    amax, amin = np.nanmax(a), np.nanmin(a)
    return (a - amin) / (amax - amin)
