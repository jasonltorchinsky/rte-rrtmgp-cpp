# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT

def bcast_coords(coords: Optional[dict], comm: MPI_COMM) -> dict:
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    # Broadcast vertical grid info, which is the same across all grids
    lay: Optional[NP_INT] = None
    lev: Optional[NP_INT] = None
    z_lay: Optional[NP_ARRAY[NP_REAL]] = None
    z_lev: Optional[NP_ARRAY[NP_REAL]] = None
    ngrid_z: Optional[NP_INT] = None
    if l_rank == MPI_ROOT:
        lay = NP_INT(coords["01"]["z_lay"][1].size)
        lev = NP_INT(coords["01"]["z_lev"][1].size)
        z_lay = np.copy(coords["01"]["z_lay"][1])
        z_lev = np.copy(coords["01"]["z_lev"][1])
        ngrid_z = coords["01"]["ngrid_z"][1]
    lay = comm.bcast(lay, root = MPI_ROOT)
    lev = comm.bcast(lev, root = MPI_ROOT)
    z_lay = comm.bcast(z_lay, root = MPI_ROOT)
    z_lev = comm.bcast(z_lev, root = MPI_ROOT)
    ngrid_z = comm.bcast(ngrid_z, root = MPI_ROOT)

    # Broadcast coarse_strs to setup l_coords
    coarse_strs: Optional[list[str]] = None
    if l_rank == MPI_ROOT:
        coarse_strs = list(coords.keys())
    coarse_strs = comm.bcast(coarse_strs, root = MPI_ROOT)

    l_grids: dict = {}
    coarse_str: str
    for coarse_str in coarse_strs:
        l_grids[coarse_str] = {}

        # Scatterv x-grid
        g_nx: Optional[NP_INT] = None
        if l_rank == MPI_ROOT:
            g_nx = NP_INT(coords[coarse_str]["x"][1].size)
        g_nx = comm.bcast(g_nx, root = MPI_ROOT)
        
        l_counts: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
        l_counts[0] = (g_nx // comm_size + int(0 < (g_nx % comm_size)))

        l_displs: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)

        ii: int
        for ii in range(1, comm_size):
            l_counts[ii] = g_nx // comm_size + int(ii < (g_nx % comm_size))
            l_displs[ii] = l_counts[ii - 1] + l_displs[ii - 1] - 1 ## NOTE: I think this might be wrong?

        dx: Optional[NP_REAL]
        g_x: NP_ARRAY[NP_REAL] = np.empty(g_nx, dtype = NP_REAL)
        l_x: NP_ARRAY[NP_REAL] = np.empty(l_counts[l_rank], dtype = NP_REAL)
        if l_rank == MPI_ROOT:
            g_x = np.copy(coords[coarse_str]["x"][1])
            dx = g_x[1] - g_x[0]
        else:
            dx = None

        comm.Scatterv([g_x, l_counts, l_displs, MPI_REAL], l_x, root = MPI_ROOT)
        dx = comm.bcast(dx, root = MPI_ROOT)
        l_nx: NP_INT
        l_xh: NP_ARRAY[NP_REAL]
        if l_x.size > 0:
            l_nx = NP_INT(l_x.size)
            l_xh = np.append(l_x - dx / 2., l_x[-1] + dx / 2.)
        else:
            l_nx = NP_INT(0)
            l_xh = np.array([], dtype = NP_REAL)

        # Broadcast the other values
        ny: Optional[NP_INT] = None
        y: Optional[NP_ARRAY[NP_REAL]] = None
        yh: Optional[NP_ARRAY[NP_REAL]] = None
        ngrid_y: Optional[NP_INT] = None
        if l_rank == MPI_ROOT:
            y = np.copy(coords[coarse_str]["y"][1])
            yh = np.copy(coords[coarse_str]["yh"][1])
            ny = NP_INT(y.size)
            ngrid_y = coords[coarse_str]["ngrid_y"][1]
        ny = comm.bcast(ny, root = MPI_ROOT)
        y = comm.bcast(y, root = MPI_ROOT)
        yh = comm.bcast(yh, root = MPI_ROOT)
        ngrid_y = comm.bcast(ngrid_y, root = MPI_ROOT)

        # Store the other values in l_grids
        l_grids[coarse_str]["nx"] = l_nx
        l_grids[coarse_str]["x"] = l_x
        l_grids[coarse_str]["xh"] = l_xh

        l_grids[coarse_str]["ny"] = ny
        l_grids[coarse_str]["y"] = y
        l_grids[coarse_str]["yh"] = yh

        l_grids[coarse_str]["lay"] = lay
        l_grids[coarse_str]["lev"] = lev
        l_grids[coarse_str]["z"] = z_lay
        l_grids[coarse_str]["zh"] = z_lev
        l_grids[coarse_str]["z_lay"] = z_lay
        l_grids[coarse_str]["z_lev"] = z_lev

        l_grids[coarse_str]["ngrid_y"] = ngrid_y
        l_grids[coarse_str]["ngrid_y"] = ngrid_y
        l_grids[coarse_str]["ngrid_z"] = ngrid_z

        # Store communication values in each local grid
        l_grids[coarse_str]["l_counts_x"] = l_counts
        l_grids[coarse_str]["l_displs_x"] = l_displs

    return l_grids