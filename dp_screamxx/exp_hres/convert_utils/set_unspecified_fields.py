# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_COMM, MPI_ROOT
from utils.rte_rrtmgp_cpp_fields import fields_dimensions, fields_descriptions, fields_units

def set_unspecified_fields(coords: dict, comm: MPI_COMM) -> dict:
    """
    Set fields not specified by the DP-SCREAM output.
    """
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    fields_out: dict = {}

    if l_rank == MPI_ROOT:
        coarse_factor_str: str
        for coarse_factor_str in coords.keys():
            nx: NP_INT = NP_INT(coords[coarse_factor_str]["x"][1].size)
            ny: NP_INT = NP_INT(coords[coarse_factor_str]["y"][1].size)
            nlay: NP_INT = NP_INT(coords[coarse_factor_str]["z_lay"][1].size)
            n_bnd_sw: NP_INT = NP_INT(coords[coarse_factor_str]["n_bnd_sw"][1])
            n_bnd_lw: NP_INT = NP_INT(coords[coarse_factor_str]["n_bnd_lw"][1])

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
            
            fields_out[coarse_factor_str]: dict = dict(
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
                fields_out[coarse_factor_str][key] = \
                    (fields_dimensions[key], field_vals, 
                        dict(description = fields_descriptions[key], units = fields_units[key])
                    )
                
    return fields_out