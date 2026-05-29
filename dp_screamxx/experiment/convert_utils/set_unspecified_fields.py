# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, \
    MPI_COMM, MPI_ROOT

def set_unspecified_fields(xr_dp_scream: XR_DATASET, g_grids: dict,
    comm: MPI_COMM) -> Optional[dict]:
    #---------------------------------------------------------------------------
    # Get MPI communicator information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    #---------------------------------------------------------------------------
    # Set up dict for holding generally unspecified field values
    #---------------------------------------------------------------------------
    vals_out: Optional[dict] = None

    #---------------------------------------------------------------------------
    # Only root process actually generates these values
    #---------------------------------------------------------------------------
    if l_rank == MPI_ROOT:
        #-----------------------------------------------------------------------
        # Dimension info
        #-----------------------------------------------------------------------
        nt: NP_INT = NP_INT(xr_dp_scream.sizes["time"])
        n_bnd_sw: NP_INT = NP_INT(xr_dp_scream.sizes["swband"])
        n_bnd_lw: NP_INT = NP_INT(xr_dp_scream.sizes["lwband"])

        vals_out = dict()
        coarse_str: str
        for coarse_str in g_grids.keys():
            nx: NP_INT = g_grids[coarse_str]["nx"]
            ny: NP_INT = g_grids[coarse_str]["ny"]
            nlay: NP_INT = g_grids[coarse_str]["nlay"]

            #-------------------------------------------------------------------
            # Acceleration grid information
            #-------------------------------------------------------------------
            ### NOTE: The number of points in the acceleration grid "should"
            ### be between 1/10 and 1/20 of nx, ny, nlay
            ngrid_x: NP_INT = NP_INT(np.ceil(nx / 10))
            ngrid_y: NP_INT = NP_INT(np.ceil(ny / 10))
            ngrid_z: NP_INT = NP_INT(np.ceil(nlay / 10))

            #-------------------------------------------------------------------
            # Longwave boundary conditions
            #-------------------------------------------------------------------
            emis_sfc: NP_ARRAY[NP_REAL] = \
                np.ones((nt, ny, nx, n_bnd_lw), dtype = NP_REAL)
            
            #-------------------------------------------------------------------
            # Surface albedo
            #-------------------------------------------------------------------
            sfc_alb_dir: NP_ARRAY[NP_REAL] = \
                np.ones((nt, ny, nx, n_bnd_sw), dtype = NP_REAL) * 0.07
            sfc_alb_dif: NP_ARRAY[NP_REAL] = \
                np.ones((nt, ny, nx, n_bnd_sw), dtype = NP_REAL) * 0.07

            #-------------------------------------------------------------------
            # Incoming solar radiation information
            #-------------------------------------------------------------------
            tsi: NP_ARRAY[NP_REAL] = \
                np.ones((nt, ny, nx), dtype = NP_REAL) * 1361.841

            azi: NP_ARRAY[NP_REAL] = \
                np.ones((nt, ny, nx), dtype = NP_REAL) * 0.0
            
            #-------------------------------------------------------------------
            # Store values into a single dict
            #-------------------------------------------------------------------
            vals_out[coarse_str] = dict(
                ngrid_x = ngrid_x,
                ngrid_y = ngrid_y,
                ngrid_z = ngrid_z,
                emis_sfc = emis_sfc,
                sfc_alb_dir = sfc_alb_dir,
                sfc_alb_dif = sfc_alb_dif,
                tsi = tsi,
                azi = azi
            )

            #-------------------------------------------------------------------
            # The following quantities do not need to be specified for
            # RTE-RRTMGP-CPP input, although a warning will be generated
            #-------------------------------------------------------------------
            ## Set quantities not expected to be set in the DP-SCREAM output
#            unexpected_keys: list[str] = ["vmr_ccl4", "vmr_cfc11", "vmr_cfc12",
#                "vmr_cfc22", "vmr_hfc143a", "vmr_hfc125", "vmr_hfc32", "vmr_hfc23",
#                "vmr_hfc134a", "vmr_cf4", "vmr_no2", "aermr01", "aermr02",
#                "aermr03", "aermr04", "aermr05", "aermr06", "aermr07", "aermr08",
#                "aermr09", "aermr10", "aermr11"]
#            
#            unexpected_vals: NP_ARRAY[NP_REAL] = np.zeros((nlay, ny, nx), dtype = NP_REAL)
#            for key in unexpected_keys:
#                vals_out[coarse_str][key] = unexpected_vals
                
    return vals_out