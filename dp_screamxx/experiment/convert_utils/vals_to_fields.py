# Standard Library Imports
from typing import Optional

# Third-Party Library Imports

# Local Library Imports
from consts.consts import NP_INT, XR_DATASET, MPI_COMM, MPI_ROOT
from consts.rte_rrtmgp_cpp_fields import grid_descriptions, grid_units, \
    fields_dimensions, fields_descriptions, fields_units

def vals_to_fields(g_vals: dict, comm: MPI_COMM) -> Optional[dict]:
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    g_fields: Optional[dict]
    if l_rank == MPI_ROOT:
        g_fields = dict()
        for coarse_str in g_vals.keys():
            g_fields[coarse_str]: dict = dict()
            ngrid_vars: list[str] = ["ngrid_x", "ngrid_y", "ngrid_z"]
            for grid_key in ngrid_vars:
                if grid_key in g_vals[coarse_str].keys():
                    g_fields[coarse_str][grid_key] = \
                        ((), g_vals[coarse_str][grid_key], 
                         dict(description = grid_descriptions[grid_key], units = grid_units[grid_key]))
            for val_key in g_vals[coarse_str].keys():
                if val_key not in ngrid_vars and val_key in g_vals[coarse_str].keys():
                    g_fields[coarse_str][val_key] = \
                        (fields_dimensions[val_key], g_vals[coarse_str][val_key], 
                         dict(description = fields_descriptions[val_key], units = fields_units[val_key]))
    else:
        g_fields = None

    return g_fields