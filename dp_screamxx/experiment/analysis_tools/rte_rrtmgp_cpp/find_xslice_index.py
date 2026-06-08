import numpy as np

def find_xslice_index(x, xslice):
    return int(np.argmin(np.abs(x - xslice)))