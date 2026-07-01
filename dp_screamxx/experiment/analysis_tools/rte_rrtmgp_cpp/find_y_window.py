# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.numeric import NP_SMALL

"""
Find the y-indices for a window based on the maximum of a given field.
"""
def find_y_window(y: NP_ARRAY[NP_REAL], field: NP_ARRAY[NP_REAL], window_width: NP_REAL) -> NP_ARRAY[NP_INT]:
    domain_width: NP_REAL = NP_REAL(y.max() - y.min())
    bounds: NP_ARRAY[NP_REAL]
    bound_indices: NP_ARRAY[NP_INT]
    if window_width > domain_width:
        bounds = np.array([y.min(), y.max()])
        bound_indices = np.array([0, -1], dtype = NP_INT)
    else:
        field_max_index: NP_ARRAY[NP_INT] = np.unravel_index(np.argmax(field[ll]), field[ll].shape) # [lay_index, y_index] of maximal field
        max_loc: NP_REAL = y[field_max_index[1]] # Y-location of maximal field [km]
        bounds = np.array([max_loc - window_width / 2., max_loc + window_width / 2.], dtype = NP_REAL)

        # Don't have to worry about shifting window past the edge after this block
        # because the window is not as wide as the domain
        if bounds[0] < y.min():
            bounds[:] += (y.min() - bounds[0])

        if bounds[1] > y.max():
            bounds[:] += (y.max() - bounds[1])

        bound_indices[0] = NP_INT(np.max(np.where(y - bounds[0] <= NP_SMALL)[0]))
        bound_indices[1] = NP_INT(np.min(np.where(bounds[1] - y <= NP_SMALL)[0]) + 1) # To include endpoint, add 1

    return [bounds, bound_indices]