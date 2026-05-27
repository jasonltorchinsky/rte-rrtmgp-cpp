# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT

def scatterv_g_grids(g_grids: Optional[dict], coarse_factors: NP_ARRAY[NP_INT], comm: MPI_COMM) -> [dict, dict]:
    #---------------------------------------------------------------------------
    # Get MPI communicator information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    #---------------------------------------------------------------------------
    # Construct the local source grid
    # We have overlap in the x-dimension, each rank contains all of y- and z-,
    # e.g., [0., 10., 20.], [20., 30.].
    #---------------------------------------------------------------------------
    l_grid_src: dict = dict()

    g_nx: Optional[NP_INT] = None
    if l_rank == MPI_ROOT:
        g_nx = g_grids["01"]["nx"]
    g_nx = comm.bcast(g_nx, root = MPI_ROOT)

    g_count_x: NP_INT = g_nx + (comm_size - 1) # Include overlap for each process
        
    l_counts: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
    l_displs: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
    l_counts[0] = g_count_x // comm_size + NP_INT(0 < (g_count_x % comm_size))

    ii: int
    for ii in range(1, comm_size):
        l_counts[ii] = g_count_x // comm_size + NP_INT(ii < (g_count_x % comm_size))
        l_displs[ii] = l_counts[ii - 1] + l_displs[ii - 1] - 1 ## NOTE: I think this might be wrong?

    # Global x-grid
    if l_rank == MPI_ROOT:
        g_x: NP_ARRAY[NP_REAL] = np.copy(g_grids["01"]["x"])
    else:
        g_x: NP_ARRAY[NP_REAL] = None
    l_x: NP_ARRAY[NP_REAL] = np.empty(l_counts[l_rank], dtype = NP_REAL) # Local x-grid

    comm.Scatterv([g_x, l_counts, l_displs, MPI_REAL], l_x, root = MPI_ROOT)
    l_nx: NP_INT = NP_INT(l_x.size)

    #---------------------------------------------------------------------------
    # Broadcast y-, z- source grid info, which is the same across all grid resolutions
    #---------------------------------------------------------------------------
    ny_src: Optional[NP_INT] = None
    y_src: Optional[NP_ARRAY[NP_REAL]] = None
    nlay: Optional[NP_INT] = None
    z_lay: Optional[NP_ARRAY[NP_REAL]] = None
    nlev: Optional[NP_INT] = None
    z_lev: Optional[NP_ARRAY[NP_REAL]] = None
    nz: Optional[NP_INT] = None
    z: Optional[NP_ARRAY[NP_REAL]] = None
    if l_rank == MPI_ROOT:
        ny_src = g_grids["01"]["ny"]
        y_src = np.copy(g_grids["01"]["y"])
        nlay = g_grids["01"]["nlay"]
        z_lay = np.copy(g_grids["01"]["z_lay"])
        nlev = g_grids["01"]["nlev"]
        z_lev = np.copy(g_grids["01"]["z_lev"])
        nz = g_grids["01"]["nz"]
        z = np.copy(g_grids["01"]["z"])
    ny_src = comm.bcast(ny_src, root = MPI_ROOT)
    y_src = comm.bcast(y_src, root = MPI_ROOT)
    nlay = comm.bcast(nlay, root = MPI_ROOT)
    z_lay = comm.bcast(z_lay, root = MPI_ROOT)
    nlev = comm.bcast(nlev, root = MPI_ROOT)
    z_lev = comm.bcast(z_lev, root = MPI_ROOT)
    nz = comm.bcast(nz, root = MPI_ROOT)
    z = comm.bcast(z, root = MPI_ROOT)

    #---------------------------------------------------------------------------
    # Store source grid info in dict
    #---------------------------------------------------------------------------
    l_grid_src: dict = dict()
    l_grid_src["nx"] = l_nx
    l_grid_src["x"] = l_x

    l_grid_src["ny"] = ny_src
    l_grid_src["y"] = y_src

    l_grid_src["nlay"] = nlay
    l_grid_src["z_lay"] = z_lay
    l_grid_src["nlev"] = nlev
    l_grid_src["z_lev"] = z_lev
    l_grid_src["nz"] = nz
    l_grid_src["z"] = z

    l_grid_src["l_counts_x"] = l_counts
    l_grid_src["l_displs_x"] = l_displs

    #---------------------------------------------------------------------------
    # Construct the local target grid, which is the coarsened grid on each
    # process based on the intersection of l_grids_src with the coarsened grid
    #---------------------------------------------------------------------------
    l_grids_tgt: dict = dict()
    
    #---------------------------------------------------------------------------
    # Get the coarsened l_grids based on the intersection of the finest l_grids
    # and the coarsened g_grid
    #---------------------------------------------------------------------------
    for coarse_factor in coarse_factors:
        coarse_str: str = "{:02}".format(coarse_factor)
        l_grids_tgt[coarse_str]: dict = dict()

        #-----------------------------------------------------------------------
        # Broadcast the global coarsened grid
        #-----------------------------------------------------------------------
        coarse_g_grid: Optional[NP_ARRAY[NP_REAL]] = None
        if l_rank == MPI_ROOT:
            coarse_g_grid = g_grids[coarse_str]
        coarse_g_grid = comm.bcast(coarse_g_grid, root = MPI_ROOT)

        #-----------------------------------------------------------------------
        # Separate the coarsened x-grid
        #-----------------------------------------------------------------------
        # Each rank gets its counts, displs, and coarsened l_grid, then
        # allgathervs the counts and displs
        if l_rank < comm_size - 1: # Use [x0, xf)
            coarse_idxs: NP_ARRAY[NP_INT] = np.where(
                (coarse_g_grid["x"] >= l_grid_src["x"].min())
                & (coarse_g_grid["x"] < l_grid_src["x"].max())
                )[0]
        else: # Use [x0, xf]
            coarse_idxs: NP_ARRAY[NP_INT] = np.where(
                (coarse_g_grid["x"] >= l_grid_src["x"].min())
                & (coarse_g_grid["x"] <= l_grid_src["x"].max())
                )[0]
        l_counts: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
        l_displs: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
        if coarse_idxs.size >= 1:
            l_counts[l_rank] = (coarse_idxs[-1] - coarse_idxs[0]) + NP_INT(1)
            l_displs[l_rank] = coarse_idxs[0]

        allgatherv_counts: NP_ARRAY[NP_INT] = np.ones(comm_size, dtype = NP_INT)
        allgatherv_displs: NP_ARRAY[NP_INT] = np.arange(comm_size, dtype = NP_INT)

        comm.Allgatherv(l_counts[l_rank],
            [l_counts, allgatherv_counts, allgatherv_displs, MPI_REAL])
        comm.Allgatherv(l_displs[l_rank],
            [l_displs, allgatherv_counts, allgatherv_displs, MPI_REAL])

        l_nx: NP_INT
        l_x: NP_ARRAY[NP_REAL]
        if coarse_idxs.size >= 1:
            l_nx: NP_INT = (coarse_idxs[-1] - coarse_idxs[0]) + NP_INT(1)
            l_x: NP_ARRAY[NP_REAL] = coarse_g_grid["x"][coarse_idxs[0]:coarse_idxs[-1] + 1]
        else:
            l_nx = NP_INT(0)
            l_x = np.empty(0, dtype = NP_REAL)

        #-----------------------------------------------------------------------
        # Broadcast y-grid info
        #-----------------------------------------------------------------------
        ny: Optional[NP_INT] = None
        y: Optional[NP_ARRAY[NP_REAL]] = None
        if l_rank == MPI_ROOT:
            ny = g_grids[coarse_str]["ny"]
            y = np.copy(g_grids[coarse_str]["y"])
        ny = comm.bcast(ny, root = MPI_ROOT)
        y = comm.bcast(y, root = MPI_ROOT)

        #-----------------------------------------------------------------------
        # Store l_grid_tgt info in dict
        #-----------------------------------------------------------------------
        l_grids_tgt[coarse_str]["nx"] = l_nx
        l_grids_tgt[coarse_str]["x"] = l_x

        l_grids_tgt[coarse_str]["ny"] = ny
        l_grids_tgt[coarse_str]["y"] = y

        l_grids_tgt[coarse_str]["nlay"] = nlay
        l_grids_tgt[coarse_str]["z_lay"] = z_lay
        l_grids_tgt[coarse_str]["nlev"] = nlev
        l_grids_tgt[coarse_str]["z_lev"] = z_lev
        l_grids_tgt[coarse_str]["nz"] = nz
        l_grids_tgt[coarse_str]["z"] = z

        # Store communication values in each local grid
        l_grids_tgt[coarse_str]["l_counts_x"] = l_counts
        l_grids_tgt[coarse_str]["l_displs_x"] = l_displs

    return [l_grid_src, l_grids_tgt]