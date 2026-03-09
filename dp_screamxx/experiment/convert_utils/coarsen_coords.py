# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY
from consts.rte_rrtmgp_cpp_fields import grid_descriptions, grid_units

def coarsen_coords(coords: dict, coarse_factor: NP_INT) -> dict:
    nx_fine: NP_INT = NP_INT(coords["x"][1].size)
    nx_coarse: NP_INT = nx_fine // coarse_factor
    ngrid_x_coarse: NP_INT = NP_INT(np.ceil(nx_coarse / 10))
    xh_min: NP_REAL = coords["xh"][1].min()
    xh_max: NP_REAL = coords["xh"][1].max()
    xh_coarse: NP_ARRAY[NP_REAL] = np.linspace(xh_min, xh_max, nx_coarse + 1,
        dtype = NP_REAL)
    x_coarse: NP_ARRAY[NP_REAL] = (xh_coarse[:-1] + xh_coarse[1:]) / 2.

    ny_fine: NP_INT = NP_INT(coords["y"][1].size)
    ny_coarse: NP_INT = ny_fine // coarse_factor
    ngrid_y_coarse: NP_INT = NP_INT(np.ceil(ny_coarse / 10))
    yh_min: NP_REAL = coords["yh"][1].min()
    yh_max: NP_REAL = coords["yh"][1].max()
    yh_coarse: NP_ARRAY[NP_REAL] = np.linspace(yh_min, yh_max, ny_coarse + 1,
        dtype = NP_REAL)
    y_coarse: NP_ARRAY[NP_REAL] = (yh_coarse[:-1] + yh_coarse[1:]) / 2.

    ## Spatial RTE-RRTMGP-CPP coords
    coords_coarse: dict = dict(
        x = ("x", x_coarse, dict(description = grid_descriptions["x"], units = grid_units["x"])),
        xh = ("xh", xh_coarse, dict(description = grid_descriptions["xh"], units = grid_units["xh"])),
        y = ("y", y_coarse, dict(description = grid_descriptions["y"], units = grid_units["y"])),
        yh = ("yh", yh_coarse, dict(description = grid_descriptions["yh"], units = grid_units["yh"])),
        ngrid_x = ((), ngrid_x_coarse, dict(description = grid_descriptions["ngrid_x"], units = grid_units["ngrid_x"])),
        ngrid_y = ((), ngrid_y_coarse, dict(description = grid_descriptions["ngrid_y"], units = grid_units["ngrid_y"])),
    )

    return {**coords, **coords_coarse}