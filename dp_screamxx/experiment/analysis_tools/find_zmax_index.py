import numpy as np

def find_zmax_index(z, zmax):
    return int(np.argmin(np.abs(z - zmax)))