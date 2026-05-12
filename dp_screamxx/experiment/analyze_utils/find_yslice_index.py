import numpy as np

def find_yslice_index(y, yslice):
    return int(np.argmin(np.abs(y - yslice)))