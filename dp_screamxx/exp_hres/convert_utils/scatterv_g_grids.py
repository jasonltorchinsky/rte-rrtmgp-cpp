# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT

def scatterv_g_grids(g_grids: Optional[dict], comm: MPI_COMM) -> [dict, dict]:
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    # l_grid_src has the source x-grid with overlap on each process, e.g.,
    # [0., 10., 20.], [20., 30.]
    l_grid_src: dict = dict()

    g_nx: Optional[NP_INT] = None
    if l_rank == MPI_ROOT:
        g_nx = g_grids["01"]["nx"]
    g_nx = comm.bcast(g_nx, root = MPI_ROOT)

    g_count_x: NP_INT = g_nx + (comm_size - 1) # Include overlap for each process
        
    l_counts: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
    l_counts[0] = g_count_x // comm_size + NP_INT(0 < (g_count_x % comm_size))

    l_displs: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)

    ii: int
    for ii in range(1, comm_size):
        l_counts[ii] = g_count_x // comm_size + NP_INT(ii < (g_count_x % comm_size))
        l_displs[ii] = l_counts[ii - 1] + l_displs[ii - 1] - 1 ## NOTE: I think this might be wrong?

    dx: Optional[NP_REAL]
    g_x: NP_ARRAY[NP_REAL] = np.empty(g_nx, dtype = NP_REAL)
    l_x: NP_ARRAY[NP_REAL] = np.empty(l_counts[l_rank], dtype = NP_REAL)
    if l_rank == MPI_ROOT:
        g_x = np.copy(g_grids["01"]["x"])
        dx = g_x[1] - g_x[0]
    else:
        dx = None

    comm.Scatterv([g_x, l_counts, l_displs, MPI_REAL], l_x, root = MPI_ROOT)
    dx = comm.bcast(dx, root = MPI_ROOT)
    l_nx: NP_INT
    if l_x.size > 0:
        l_nx = NP_INT(l_x.size)
    else:
        l_nx = NP_INT(0)

    # Broadcast the other values
    l_grid_src: dict = dict()
    l_grid_src["nx"] = l_nx
    l_grid_src["x"] = l_x

    l_grid_src["l_counts_x"] = l_counts
    l_grid_src["l_displs_x"] = l_displs

    # l_grids_tgt has the coarsen grids on each process, without overlap,
    # based on the intersection of l_grids_src with the coarsened grid
    l_grids_tgt: dict = dict()

    # Broadcast coarse_strs to setup l_grids
    coarse_strs: Optional[list[str]] = None
    if l_rank == MPI_ROOT:
        coarse_strs = sorted(list(g_grids.keys()))
    coarse_strs = comm.bcast(coarse_strs, root = MPI_ROOT)

    # Broadcast vertical grid info, which is the same across all grid resolutions
    nlay: Optional[NP_INT] = None
    nlev: Optional[NP_INT] = None
    z_lay: Optional[NP_ARRAY[NP_REAL]] = None
    z_lev: Optional[NP_ARRAY[NP_REAL]] = None
    if l_rank == MPI_ROOT:
        nlay = g_grids["01"]["nlay"]
        nlev = g_grids["01"]["nlev"]
        z_lay = np.copy(g_grids["01"]["z_lay"])
        z_lev = np.copy(g_grids["01"]["z_lev"])
    nlay = comm.bcast(nlay, root = MPI_ROOT)
    nlev = comm.bcast(nlev, root = MPI_ROOT)
    z_lay = comm.bcast(z_lay, root = MPI_ROOT)
    z_lev = comm.bcast(z_lev, root = MPI_ROOT)
    

    # Get the coarsened l_grids based on the intersection of the finest l_grids
    # and the coarsened g_grid
    coarse_str: str
    for coarse_str in coarse_strs:
        l_grids_tgt[coarse_str]: dict = dict()

        # Broadcast the coarsened g_grid
        coarse_g_grid: Optional[NP_ARRAY[NP_REAL]] = None
        if l_rank == MPI_ROOT:
            coarse_g_grid = g_grids[coarse_str]
        coarse_g_grid = comm.bcast(coarse_g_grid, root = MPI_ROOT)

        # Each rank gets its counts, displs, and coarsened l_grid, then
        # Allgathervs the counts and displs
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

        # Broadcast the other values
        ny: Optional[NP_INT] = None
        y: Optional[NP_ARRAY[NP_REAL]] = None
        if l_rank == MPI_ROOT:
            ny = g_grids[coarse_str]["ny"]
            y = np.copy(g_grids[coarse_str]["y"])
        ny = comm.bcast(ny, root = MPI_ROOT)
        y = comm.bcast(y, root = MPI_ROOT)

        # Store the other values in l_grids
        l_grids_tgt[coarse_str]["nx"] = l_nx
        l_grids_tgt[coarse_str]["x"] = l_x

        l_grids_tgt[coarse_str]["ny"] = ny
        l_grids_tgt[coarse_str]["y"] = y

        l_grids_tgt[coarse_str]["nlay"] = nlay
        l_grids_tgt[coarse_str]["nlev"] = nlev
        l_grids_tgt[coarse_str]["z_lay"] = z_lay
        l_grids_tgt[coarse_str]["z_lev"] = z_lev

        # Store communication values in each local grid
        l_grids_tgt[coarse_str]["l_counts_x"] = l_counts
        l_grids_tgt[coarse_str]["l_displs_x"] = l_displs

    return [l_grid_src, l_grids_tgt]