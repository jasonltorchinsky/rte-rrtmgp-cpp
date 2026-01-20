# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY
from utils.rte_rrtmgp_cpp_fields import grid_descriptions, grid_units

def coarsen_g_grid(g_grid: dict, coarse_factor: NP_INT) -> dict:
    nx_fine: NP_INT = g_grid["nx"]
    nx_coarse: NP_INT = nx_fine // coarse_factor

    xh_min: NP_REAL = g_grid["xh"].min()
    xh_max: NP_REAL = g_grid["xh"].max()

    xh_coarse: NP_ARRAY[NP_REAL] = np.linspace(xh_min, xh_max, nx_coarse + 1,
        dtype = NP_REAL)
    x_coarse: NP_ARRAY[NP_REAL] = (xh_coarse[:-1] + xh_coarse[1:]) / 2.

    ny_fine: NP_INT = g_grid["ny"]
    ny_coarse: NP_INT = ny_fine // coarse_factor
    
    yh_min: NP_REAL = g_grid["yh"].min()
    yh_max: NP_REAL = g_grid["yh"].max()

    yh_coarse: NP_ARRAY[NP_REAL] = np.linspace(yh_min, yh_max, ny_coarse + 1,
        dtype = NP_REAL)
    y_coarse: NP_ARRAY[NP_REAL] = (yh_coarse[:-1] + yh_coarse[1:]) / 2.

    ## Spatial RTE-RRTMGP-CPP coords
    g_grid_coarse: dict = dict(
        nx = nx_coarse,
        x = x_coarse,
        xh = xh_coarse,
        ny = ny_coarse,
        y = y_coarse,
        yh = yh_coarse,
        nlay = g_grid["nlay"],
        nlev = g_grid["nlev"],
        z_lay = g_grid["z_lay"],
        z_lev = g_grid["z_lev"]
    )

    return g_grid_coarse