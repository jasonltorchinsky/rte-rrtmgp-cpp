# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.numeric import NP_SMALL

"""
Find the y-islices for a slice based on the maximum of a given field.
"""
def find_y_islice(y: NP_ARRAY[NP_REAL], field: XR_DATAARRAY, slice_width: NP_REAL) -> NP_ARRAY[NP_INT]:
    # slice-width given in [km], need to convert to [m]
    domain_width: NP_REAL = NP_REAL(y.max() - y.min())

    y_slice_width: NP_REAL = slice_width * 1.e3 # [km] => [m]
    
    y_slice_end: NP_REAL # [m]
    y_slice_start: NP_REAL # [m]
    bound_indices: NP_ARRAY[NP_INT]
    if y_slice_width > domain_width:
        y_slice_start = NP_REAL(y.min())
        y_slice_end = NP_REAL(y.max())
    else:
        field_max_index: NP_ARRAY[NP_INT] = np.unravel_index(np.argmax(field.to_numpy()), field.to_numpy().shape) # [lay_index, y_index] of maximal field
        max_loc: NP_REAL = field["y"][field_max_index[1]] # Y-location of maximal field [m]
        y_slice_start = NP_REAL(max_loc - y_slice_width / 2.)
        y_slice_end = NP_REAL(max_loc + y_slice_width / 2.)

        # Don't have to worry about shifting window past the edge after this block
        # because the window is not as wide as the domain
        if y_slice_start < y.min():
            y_slice_start += NP_REAL(y.min() - y_slice_start)
            y_slice_end += NP_REAL(y.min() - y_slice_start)

        if y_slice_end > y.max():
            y_slice_start += NP_REAL(y.max() - y_slice_end)
            y_slice_end += NP_REAL(y.max() - y_slice_end)

    y_islice_start: NP_INT = NP_INT(np.argmin(np.abs(NP_REAL(y.to_numpy()) - y_slice_start)))
    y_islice_end: NP_INT = NP_INT(np.argmin(np.abs(NP_REAL(y.to_numpy()) - y_slice_end))) + 1

    return slice(y_islice_start, y_islice_end)