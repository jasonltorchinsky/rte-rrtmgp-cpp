# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET

def interp_2dfield(xr_dpscream: XR_DATASET, dpscream_field_key: str,
    rte_field_key: str, sort_mask: NP_ARRAY[NP_INT], g_grids: dict, 
    l_grid_src: dict, l_grids_tgt: dict, tt: int, comm: MPI_COMM, interp_method: str = "nearest") -> dict:
    field_out: dict = {}

    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    field_src: Optional[NP_ARRAY[NP_REAL]]
    field_min: Optional[NP_REAL]
    field_max: Optional[NP_REAL]
    # Root Rank reads input file, constructs full field and Scatterv
    if l_rank == MPI_ROOT:
        assert(dpscream_field_key in xr_dpscream.keys())
        field_src: NP_ARRAY[NP_REAL] = \
            xr_dpscream[dpscream_field_key].isel(time = tt, ncol = sort_mask).values.astype(NP_REAL) # Field; (ncol)
        
        np.nan_to_num(field_src, NP_REAL(0.))

        ## Exceptions - Do in serial for now
        if rte_field_key in ["t_sfc"]:
            field_min = NP_REAL(100.0) # Lowest temperature in mesosphere https://scied.ucar.edu/learning-zone/atmosphere/mesosphere [K]
            field_max = NP_REAL(329.817) # Hottest observed temperature https://www.ncei.noaa.gov/news/earths-hottest-temperature [K]
        elif rte_field_key in ["mu0"]: # Between -1.0 and 1.0
            field_min: NP_REAL = NP_REAL(-1.0)
            field_max: NP_REAL = NP_REAL(1.0)
        else:
            field_min = field_src.min()
            field_max = field_src.max()
            
        field_src[field_src > field_max] = field_max
        field_src[field_src < field_min] = field_min

        g_nx: NP_INT = g_grids["01"]["nx"]
        g_ny: NP_INT = g_grids["01"]["ny"]

        field_src = field_src.reshape(g_nx, g_ny)
    else:
        g_nx = None
        g_ny = None
        field_src = None
        field_min = None
        field_max = None

    g_nx = comm.bcast(g_nx, root = MPI_ROOT)
    g_ny = comm.bcast(g_ny, root = MPI_ROOT)

    # Scatterv the original field
    l_nx_src: NP_INT = l_grid_src["nx"]
    l_ny_src: NP_INT = g_ny

    l_counts_src: NP_ARRAY[NP_INT] = l_grid_src["l_counts_x"] * l_ny_src
    l_displs_src: NP_ARRAY[NP_INT] = l_grid_src["l_displs_x"] * l_ny_src

    l_field_src: NP_ARRAY[NP_REAL] = np.empty([l_nx_src, l_ny_src], dtype = NP_REAL)

    field_min = comm.bcast(field_min, root = MPI_ROOT)
    field_max = comm.bcast(field_max, root = MPI_ROOT)
    comm.Scatterv([field_src, l_counts_src, l_displs_src, MPI_REAL], l_field_src, root = MPI_ROOT)

    # Get source grid - points to interpolate from
    l_x_src: NP_ARRAY[NP_REAL] = l_grid_src["x"]
    g_y: Optional[NP_ARRAY[NP_REAL]] = None
    if l_rank == MPI_ROOT:
        g_y = g_grids["01"]["y"]
    l_y_src: NP_ARRAY[NP_REAL] = comm.bcast(g_y, root = MPI_ROOT)

    l_horz_interpolator: RegularGridInterpolator = \
        RegularGridInterpolator((l_x_src, l_y_src), l_field_src, method = interp_method)

    # Coarsen the field as necessary
    for coarse_str in l_grids_tgt.keys():
        field_out[coarse_str]: dict = {}
        # Get target layer grid - points to interpolate to
        l_nx_tgt: NP_INT = l_grids_tgt[coarse_str]["nx"]
        l_ny_tgt: NP_INT = l_grids_tgt[coarse_str]["ny"]

        l_counts_x: NP_ARRAY[NP_INT] = l_grids_tgt[coarse_str]["l_counts_x"]
        l_displs_x: NP_ARRAY[NP_INT] = l_grids_tgt[coarse_str]["l_displs_x"]
            
        l_counts_tgt: list[NP_INT] = l_counts_x * l_ny_tgt
        l_displs_tgt: list[NP_INT] = l_displs_x * l_ny_tgt

        l_x_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["x"]
        l_y_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["y"]

        l_XX_tgt, l_YY_tgt = np.meshgrid(l_x_tgt, l_y_tgt, indexing = "ij")
        l_pts_tgt: NP_ARRAY[NP_REAL] = \
            np.stack([l_XX_tgt.flatten(), l_YY_tgt.flatten()], axis = 1)

        ## Interpolate the values to regular vertical layers, and limit them
        l_field_tgt: NP_ARRAY[NP_REAL] = l_horz_interpolator(l_pts_tgt).reshape(l_nx_tgt, l_ny_tgt)
        l_field_tgt[l_field_tgt < field_min] = field_min
        l_field_tgt[l_field_tgt > field_max] = field_max

        # Reconstruct the full field
        field_tgt: Optional[NP_ARRAY[NP_REAL]] = None
        if l_rank == MPI_ROOT:
            nx_tgt: NP_INT = g_grids[coarse_str]["nx"]
            ny_tgt: NP_INT = g_grids[coarse_str]["ny"]

            field_tgt = np.empty([nx_tgt, ny_tgt], dtype = NP_REAL)

        comm.Gatherv(l_field_tgt, 
            [field_tgt, l_counts_tgt, l_displs_tgt, MPI_REAL],
            root = MPI_ROOT)

        if l_rank == MPI_ROOT:
            field_tgt = np.reshape(field_tgt, (nx_tgt, ny_tgt)) # (nx, ny)
            field_tgt = np.transpose(field_tgt, axes = (1, 0)) # (ny, nx)

        if l_rank == MPI_ROOT:
            field_out[coarse_str][rte_field_key] = field_tgt

    return field_out