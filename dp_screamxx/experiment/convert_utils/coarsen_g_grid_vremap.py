# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY

def coarsen_g_grid_vremap(g_grid: dict, coarse_factor: NP_INT) -> dict:
    #---------------------------------------------------------------------------
    # Coarsen x-grid
    #---------------------------------------------------------------------------
    nx_fine: NP_INT = g_grid["nx"]
    nx_coarse: NP_INT = nx_fine // coarse_factor

    xh_min: NP_REAL = g_grid["x"].min() - g_grid["dx"] / 2.
    xh_max: NP_REAL = g_grid["x"].max() + g_grid["dx"] / 2.

    xh_coarse: NP_ARRAY[NP_REAL] = np.linspace(xh_min, xh_max, nx_coarse + 1,
        dtype = NP_REAL)
    x_coarse: NP_ARRAY[NP_REAL] = (xh_coarse[:-1] + xh_coarse[1:]) / 2.
    dx_coarse: NP_REAL = x_coarse[1] - x_coarse[0]

    #---------------------------------------------------------------------------
    # Coarsen y-grid
    #---------------------------------------------------------------------------
    ny_fine: NP_INT = g_grid["ny"]
    ny_coarse: NP_INT = ny_fine // coarse_factor
    
    yh_min: NP_REAL = g_grid["y"].min() - g_grid["dy"] / 2.
    yh_max: NP_REAL = g_grid["y"].max() + g_grid["dy"] / 2.

    yh_coarse: NP_ARRAY[NP_REAL] = np.linspace(yh_min, yh_max, ny_coarse + 1,
        dtype = NP_REAL)
    y_coarse: NP_ARRAY[NP_REAL] = (yh_coarse[:-1] + yh_coarse[1:]) / 2.
    dy_coarse: NP_REAL = y_coarse[1] - y_coarse[0]

    # NOTE: z-grid does not get coarsened.
    #---------------------------------------------------------------------------
    # Collect grid information into dict
    #---------------------------------------------------------------------------
    g_grid_coarse: dict = dict(
        nx = nx_coarse,
        dx = dx_coarse,
        x = x_coarse,
        ny = ny_coarse,
        dy = dy_coarse,
        y = y_coarse,
        nlay = g_grid["nlay"],
        z_lay = g_grid["z_lay"],
        nlev = g_grid["nlev"],
        z_lev = g_grid["z_lev"],
        nz = g_grid["nz"],
        dz = g_grid["dz"],
        z = g_grid["z"]
    )

    return g_grid_coarse