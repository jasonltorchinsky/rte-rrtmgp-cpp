# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, MPI_COMM, MPI_ROOT
from consts.rte_rrtmgp_cpp_fields import grid_descriptions, grid_units

def grids_to_coords(g_grids: dict, time: NP_ARRAY[NP_REAL], coarse_factors: NP_ARRAY[NP_REAL], comm: MPI_COMM) -> Optional[dict]:
    #---------------------------------------------------------------------------
    # Get MPI communicator information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    g_coords: Optional[dict] = None
    if l_rank == MPI_ROOT:
        g_coords = dict()
        for coarse_factor in coarse_factors:
            coarse_str: str = "{:02}".format(coarse_factor)
            #-------------------------------------------------------------------
            # Derive additional grids for RTE-RRTMGP-CPP input
            #-------------------------------------------------------------------
            x: NP_ARRAY[NP_REAL] = g_grids[coarse_str]["x"]
            dx: NP_REAL = x[1] - x[0]
            xh: NP_ARRAY[NP_REAL] = np.concatenate([x - dx / 2., [x[-1] + dx / 2.]]) # Column interfaces

            y: NP_ARRAY[NP_REAL] = g_grids[coarse_str]["y"]
            dy: NP_REAL = y[1] - y[0]
            yh: NP_ARRAY[NP_REAL] = np.concatenate([y - dy / 2., [y[-1] + dy / 2.]])

            g_coords[coarse_str]: dict = dict(
                time = ("time", time, dict(description = grid_descriptions["time"], units = grid_units["time"])),
                x = ("x", x, dict(description = grid_descriptions["x"], units = grid_units["x"])),
                xh = ("xh", xh, dict(description = grid_descriptions["xh"], units = grid_units["xh"])),
                y = ("y", y, dict(description = grid_descriptions["y"], units = grid_units["y"])),
                yh = ("yh", yh, dict(description = grid_descriptions["yh"], units = grid_units["yh"])),
                z = ("z", g_grids[coarse_str]["z_lay"], dict(description = grid_descriptions["z"], units = grid_units["z"])),
                zh = ("zh", g_grids[coarse_str]["z_lev"], dict(description = grid_descriptions["zh"], units = grid_units["zh"])),
                z_lay = ("z_lay", g_grids[coarse_str]["z_lay"], dict(description = grid_descriptions["z_lay"], units = grid_units["z_lay"])),
                z_lev = ("z_lev", g_grids[coarse_str]["z_lev"], dict(description = grid_descriptions["z_lev"], units = grid_units["z_lev"])),
            )

    return g_coords