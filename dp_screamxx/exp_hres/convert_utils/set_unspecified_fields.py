# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, \
    MPI_COMM, MPI_ROOT
from utils.rte_rrtmgp_cpp_fields import grid_descriptions, grid_units, \
    fields_dimensions, fields_descriptions, fields_units

def set_unspecified_fields(xr_dpscream: XR_DATASET, g_grids: dict, comm: MPI_COMM) -> dict:
    """
    Set fields not specified by the DP-SCREAM output.
    """
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    ## Wavelength info
    n_bnd_sw: NP_INT = NP_INT(xr_dpscream.sizes["swband"])
    n_bnd_lw: NP_INT = NP_INT(xr_dpscream.sizes["lwband"])

    fields_out: dict = {}

    if l_rank == MPI_ROOT:
        coarse_str: str
        for coarse_str in g_grids.keys():
            nx: NP_INT = g_grids[coarse_str]["nx"]
            ny: NP_INT = g_grids[coarse_str]["ny"]
            nlay: NP_INT = g_grids[coarse_str]["nlay"]

            ### NOTE: The number of points in the acceleration grid "should"
            ### be between 1/10 and 1/20 of nx, ny, nlay
            ngrid_x: NP_INT = NP_INT(np.ceil(nx / 10))
            ngrid_y: NP_INT = NP_INT(np.ceil(ny / 10))
            ngrid_z: NP_INT = NP_INT(np.ceil(nlay / 10))

            ## Longwave boundary conditions
            emis_sfc: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx, n_bnd_lw), dtype = NP_REAL)
            
            sfc_alb_dir: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx, n_bnd_sw), dtype = NP_REAL) * 0.07 
            sfc_alb_dif: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx, n_bnd_sw), dtype = NP_REAL) * 0.07

            tsi: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx), dtype = NP_REAL) * 551.58

            azi: NP_ARRAY[NP_REAL] = \
                np.ones((ny, nx), dtype = NP_REAL) * 0.0 
            
            fields_out[coarse_str]: dict = dict(
                ngrid_x = ((), ngrid_x, dict(description = grid_descriptions["ngrid_x"], units = grid_units["ngrid_x"])),
                ngrid_y = ((), ngrid_y, dict(description = grid_descriptions["ngrid_y"], units = grid_units["ngrid_y"])),
                ngrid_z = ((), ngrid_z, dict(description = grid_descriptions["ngrid_z"], units = grid_units["ngrid_z"])),
                emis_sfc = (fields_dimensions["emis_sfc"], emis_sfc, dict(description = fields_descriptions["emis_sfc"], units = fields_units["emis_sfc"])),
                sfc_alb_dir = (fields_dimensions["sfc_alb_dir"], sfc_alb_dir, dict(description = fields_descriptions["sfc_alb_dir"], units = fields_units["sfc_alb_dir"])),
                sfc_alb_dif = (fields_dimensions["sfc_alb_dif"], sfc_alb_dif, dict(description = fields_descriptions["sfc_alb_dif"], units = fields_units["sfc_alb_dif"])),
                tsi = (fields_dimensions["tsi"], tsi, dict(description = fields_descriptions["tsi"], units = fields_units["tsi"])),
                azi = (fields_dimensions["azi"], azi, dict(description = fields_descriptions["azi"], units = fields_units["azi"]))
            )

            ## Set quantities not expected to be set in the DP-SCREAM output
            unexpected_keys: list[str] = ["vmr_ccl4", "vmr_cfc11", "vmr_cfc12",
                "vmr_cfc22", "vmr_hfc143a", "vmr_hfc125", "vmr_hfc32", "vmr_hfc23",
                "vmr_hfc134a", "vmr_cf4", "vmr_no2", "aermr01", "aermr02",
                "aermr03", "aermr04", "aermr05", "aermr06", "aermr07", "aermr08",
                "aermr09", "aermr10", "aermr11"]
            
            field_vals: NP_ARRAY[NP_REAL] = np.zeros((nlay, ny, nx), dtype = NP_REAL)
            for key in unexpected_keys:
                fields_out[coarse_str][key] = \
                    (fields_dimensions[key], field_vals, 
                        dict(description = fields_descriptions[key], units = fields_units[key])
                    )
                
    return fields_out