# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.numeric import NP_SMALL

"""
Find the y-slices for a slice based on the maximum of a given field.
"""
def find_y_slice(y: NP_ARRAY[NP_REAL], field: XR_DATAARRAY, slice_width: NP_REAL) -> NP_ARRAY[NP_INT]:
    # slice-width given in [km], need to convert to [m]
    domain_width: NP_REAL = NP_REAL(y.max() - y.min())

    y_slice_width: NP_REAL = slice_width * 1.e3 # [km] => [m]
    
    y_slice_max: NP_REAL # [m]
    y_slice_min: NP_REAL # [m]
    bound_indices: NP_ARRAY[NP_INT]
    if y_slice_width > domain_width:
        y_slice_min = NP_REAL(y.min())
        y_slice_max = NP_REAL(y.max())
    else:
        field_max_index: NP_ARRAY[NP_INT] = np.unravel_index(np.argmax(field.to_numpy()), field.to_numpy().shape) # [lay_index, y_index] of maximal field
        max_loc: NP_REAL = field["y"][field_max_index[1]] # Y-location of maximal field [m]
        y_slice_min = NP_REAL(max_loc - y_slice_width / 2.)
        y_slice_max = NP_REAL(max_loc + y_slice_width / 2.)

        # Don't have to worry about shifting window past the edge after this block
        # because the window is not as wide as the domain
        if y_slice_min < y.min():
            y_slice_min += NP_REAL(y.min() - y_slice_min)
            y_slice_max += NP_REAL(y.min() - y_slice_min)

        if y_slice_max > y.max():
            y_slice_min += NP_REAL(y.max() - y_slice_max)
            y_slice_max += NP_REAL(y.max() - y_slice_max)

    return slice(y_slice_min, y_slice_max)