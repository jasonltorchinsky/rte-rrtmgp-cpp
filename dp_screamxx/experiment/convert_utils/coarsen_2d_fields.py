# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np

from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET
from consts.rte_rrtmgp_cpp_fields import rte_2d_field_keys

def coarsen_2d_fields(xr_dp_scream: XR_DATASET, g_grids: dict, l_grid_src: dict, 
    l_grids_tgt: dict, comm: MPI_COMM, interp_method: str = "nearest") -> dict:
    #---------------------------------------------------------------------------
    # Get MPI communicator information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    #---------------------------------------------------------------------------
    # Common variables throuhgout script
    #---------------------------------------------------------------------------
    l_nt: NP_INT = NP_INT(xr_dp_scream["time"].size)

    #---------------------------------------------------------------------------
    # Set up dict for holding horizontally coarsened (i.e., tgt) field values
    #---------------------------------------------------------------------------
    l_fields_tgt: dict = {}
    for coarse_str in l_grids_tgt.keys():
        l_fields_tgt[coarse_str] = {}

    fields_tgt: Optional[dict] = None
    if l_rank == MPI_ROOT:
        fields_tgt = {}
        for coarse_str in l_grids_tgt.keys():
            fields_tgt[coarse_str] = {}

    #---------------------------------------------------------------------------
    # Get source grid information for horizontal coarsening
    #---------------------------------------------------------------------------
    l_x_src: NP_ARRAY[NP_REAL] = l_grid_src["x"]
    l_y_src: NP_ARRAY[NP_REAL] = l_grid_src["y"]

    #---------------------------------------------------------------------------
    # Loop through RTE-RRTMGP-CPP+RT fields
    #---------------------------------------------------------------------------
    for rad_tran_key in rte_2d_field_keys:
        if l_rank == MPI_ROOT:
            current_time = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Extracting DP-SCREAM field(s) for {}...".format(current_time, rad_tran_key)
            print(msg, flush = True)
        #-----------------------------------------------------------------------
        # Extract relevant fields from DP-SCREAM file
        #-----------------------------------------------------------------------
        if rad_tran_key == "t_sfc":
            dp_scream_key = "surf_radiative_T"
        elif rad_tran_key == "mu0":
            dp_scream_key = "cosine_solar_zenith_angle"

        l_field_src: NP_ARRAY[NP_REAL] = extract_dp_scream_field(xr_dp_scream, dp_scream_key) # [nt, l_nx, ny]

        #-----------------------------------------------------------------------
        # Coarsen field values horizontally
        #-----------------------------------------------------------------------
        l_horz_coarsener: list[RegularGridInterpolator] = \
            [RegularGridInterpolator((l_x_src, l_y_src), l_field_src[tt,...], method = interp_method)
                for tt in range(0, l_nt)]
        for coarse_str in l_grids_tgt.keys():
            if l_rank == MPI_ROOT:
                current_time = datetime.now().strftime("%H:%M:%S")
                msg: str = "[{}]: Coarsening {} to lr_{}...".format(current_time, rad_tran_key, coarse_str)
                print(msg, flush = True)
            #-------------------------------------------------------------------
            # Get target horizontal grid and communication parameters
            #-------------------------------------------------------------------
            l_nx_tgt: NP_INT = l_grids_tgt[coarse_str]["nx"]
            l_ny_tgt: NP_INT = l_grids_tgt[coarse_str]["ny"]

            l_x_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["x"]
            l_y_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["y"]

            l_XX_tgt, l_YY_tgt = \
                np.meshgrid(l_x_tgt, l_y_tgt, indexing = "ij")
            l_pts_tgt: NP_ARRAY[NP_REAL] = \
                np.stack([l_XX_tgt.flatten(), l_YY_tgt.flatten()], 
                    axis = 1)

            l_counts_x: NP_ARRAY[NP_INT] = l_grids_tgt[coarse_str]["l_counts_x"]
            l_displs_x: NP_ARRAY[NP_INT] = l_grids_tgt[coarse_str]["l_displs_x"]

            l_counts_tgt: NP_ARRAY[NP_INT] = l_nt * l_ny_tgt * l_counts_x
            l_displs_tgt: NP_ARRAY[NP_INT] = l_nt * l_ny_tgt * l_displs_x

            #-------------------------------------------------------------------
            # Horizontally coarsen the field
            #-------------------------------------------------------------------
            l_field_tgt: NP_ARRAY[NP_REAL] = np.empty([l_nt, l_nx_tgt, l_ny_tgt], dtype = NP_REAL)
            for tt in range(0, l_nt):
                l_field_tgt[tt,...] = l_horz_coarsener[tt](l_pts_tgt).reshape(l_nx_tgt, l_ny_tgt)

            #-------------------------------------------------------------------
            # Store horizontally-coarsened field values
            #-------------------------------------------------------------------
            l_fields_tgt[coarse_str][rad_tran_key] = l_field_tgt

            #-------------------------------------------------------------------
            # Gatherv the whole field onto MPI_ROOT
            #-------------------------------------------------------------------
            # Reshape l_field_tgt for easier Gatherv
            l_field_tgt = np.ascontiguousarray(np.transpose(l_field_tgt, axes = [1, 0, 2])) # [nx, nt, ny]
            field_tgt: Optional[NP_ARRAY[NP_REAL]] = None
            if l_rank == MPI_ROOT:
                nt_tgt: NP_INT = l_nt
                nx_tgt: NP_INT = g_grids[coarse_str]["nx"]
                ny_tgt: NP_INT = l_ny_tgt
    
                field_tgt = np.empty(nt_tgt * nx_tgt * ny_tgt, dtype = NP_REAL)

            comm.Gatherv(l_field_tgt, 
                [field_tgt, l_counts_tgt, l_displs_tgt, MPI_REAL],
                root = MPI_ROOT)

            # At this point, field_tgt is a concatenation of comm_size
            # arrays of length nt * l_nx * ny * nz. 
            if l_rank == MPI_ROOT:
                field_tgt = np.ascontiguousarray(
                    np.transpose(field_tgt.reshape(nx_tgt, nt_tgt, ny_tgt), 
                        axes = [1, 2, 0])) # [nt, ny, nx]

                fields_tgt[coarse_str][rad_tran_key] = field_tgt

    return fields_tgt

def extract_dp_scream_field(xr_dp_scream: XR_DATASET, dp_scream_key: str) -> tuple[NP_ARRAY[NP_REAL]]:
    #---------------------------------------------------------------------------
    # Get field information
    #---------------------------------------------------------------------------
    field_src: NP_ARRAY[NP_REAL] = xr_dp_scream[dp_scream_key].to_numpy().astype(NP_REAL) # (nt, ny, nx)

    field_src = np.transpose(field_src, axes = [0, 2, 1]) # [nt, nx, ny]

    return field_src
