# Standard Library Imports
from typing import Optional

# Third-Party Library Imports

# Local Library Imports
from consts.consts import NP_INT, XR_DATASET, MPI_COMM, MPI_ROOT
from consts.rte_rrtmgp_cpp_fields import grid_descriptions, grid_units

def grids_to_coords(xr_dpscream: XR_DATASET, g_grids: dict, comm: MPI_COMM) -> Optional[dict]:
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    g_coords: Optional[dict]
    if l_rank == MPI_ROOT:
        g_coords = dict()
        for coarse_str in g_grids.keys():
            ## Spatial RTE-RRTMGP-CPP coords
            g_coords[coarse_str]: dict = dict(
                x = ("x", g_grids[coarse_str]["x"], dict(description = grid_descriptions["x"], units = grid_units["x"])),
                xh = ("xh", g_grids[coarse_str]["xh"], dict(description = grid_descriptions["xh"], units = grid_units["xh"])),
                y = ("y", g_grids[coarse_str]["y"], dict(description = grid_descriptions["y"], units = grid_units["y"])),
                yh = ("yh", g_grids[coarse_str]["yh"], dict(description = grid_descriptions["yh"], units = grid_units["yh"])),
                z = ("z", g_grids[coarse_str]["z_lay"], dict(description = grid_descriptions["z"], units = grid_units["z"])),
                zh = ("zh", g_grids[coarse_str]["z_lev"], dict(description = grid_descriptions["zh"], units = grid_units["zh"])),
                z_lay = ("z_lay", g_grids[coarse_str]["z_lay"], dict(description = grid_descriptions["z_lay"], units = grid_units["z_lay"])),
                z_lev = ("z_lev", g_grids[coarse_str]["z_lev"], dict(description = grid_descriptions["z_lev"], units = grid_units["z_lev"])),
            )
    else:
        g_coords = None

    return g_coords