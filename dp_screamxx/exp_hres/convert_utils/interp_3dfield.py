# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
from scipy.interpolate import griddata

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET, \
    g

def interp_3dfield(xr_dpscream: XR_DATASET, dpscream_field_key: str,
    rte_field_key: str, sort_mask: NP_ARRAY[NP_INT], g_grids: dict, 
    l_grid_src: dict, l_grids_tgt: dict, tt: NP_INT, comm: MPI_COMM, 
    interp_method: str = "nearest") -> NP_ARRAY[NP_REAL]:
    field_out: dict = {}

    l_rank: NP_INT = NP_INT(comm.Get_rank())

    z_src: Optional[NP_ARRAY[NP_REAL]]
    field_src: Optional[NP_ARRAY[NP_REAL]]
    field_min: Optional[NP_REAL]
    field_max: Optional[NP_REAL]
    # Root Rank reads input file, constructs full field and Scatterv
    if l_rank == MPI_ROOT:
        ## NOTE: Only using DP-SCREAM level interface (RTE-RRTMGP-CPP+RT layer) values
        if dpscream_field_key in xr_dpscream.keys(): # Only have values at midpoints
            field_src: NP_ARRAY[NP_REAL] = \
                xr_dpscream[dpscream_field_key].isel(time = tt, ncol = sort_mask).values.astype(NP_REAL) # Field at layer midpoints; (ncol, n_lay_z)
                    
        else: # Should have values and midpoints and interfaces
            dpscream_field_key_mid: str = dpscream_field_key + "_mid"

            ## We should always have fields values at layer midpoints
            ## Unless we don't, then this needs to be fixed
            assert(dpscream_field_key_mid in xr_dpscream.keys())
            field_src: NP_ARRAY[NP_REAL] = \
                xr_dpscream[dpscream_field_key_mid].isel(time = tt, ncol = sort_mask).values.astype(NP_REAL) # Field at layer midpoints; (ncol, n_lay_z)
        
        z_src = xr_dpscream["z_mid"].isel(time = tt, ncol = sort_mask).values.astype(NP_REAL) # Layer midpoints [m]; (ncol, n_lay_z)

        ## Exceptions - Do in serial for now
        if rte_field_key in ["dei"]: # DP-SCREAM has rei, RTE-RRTMGP-CPP has dei
            field_src = 2. * field_src
        elif rte_field_key in ["lwp", "iwp"]: # Derived from multiple quantities
            p_int: NP_ARRAY[NP_REAL] = \
                xr_dpscream["p_int"].isel(time = tt, ncol = sort_mask).values.astype(NP_REAL) # Pressure at layer interfaces [Pa]; (ncol, n_lev_z)
            dp: NP_ARRAY[NP_REAL] = p_int[:,1:] - p_int[:,:-1] # Layer pressure thickness [Pa]; (ncol, n_lay_z)

            field_src = field_src * dp / g

        ## Get field min and max
        ## Exceptions
        if rte_field_key in ["rel"]: # Between 2.5 μm and 21.5 μm
            field_min = NP_REAL(2.5)
            field_max = NP_REAL(21.5)
        elif rte_field_key in ["dei"]: # Between 10. μm and 180. μm
            field_min = NP_REAL(10.)
            field_max = NP_REAL(180.)
        else:
            field_min = field_src.min()
            field_max = field_src.max()

        g_nx: NP_INT = g_grids["01"]["nx"]
        g_ny: NP_INT = g_grids["01"]["ny"]
        g_nlay: NP_INT = g_grids["01"]["nlay"]

        z_src = z_src.reshape(g_nx, g_ny, g_nlay)
        field_src = field_src.reshape(g_nx, g_ny, g_nlay)
    else:
        g_nx = None
        g_ny = None
        g_nlay = None
        z_src = None
        field_src = None
        field_min = None
        field_max = None

    g_nx = comm.bcast(g_nx, root = MPI_ROOT)
    g_ny = comm.bcast(g_ny, root = MPI_ROOT)
    g_nlay = comm.bcast(g_nlay, root = MPI_ROOT)

    # Scatterv the original field
    l_nx_src: NP_INT = l_grid_src["nx"]
    l_ny_src: NP_INT = g_ny
    l_nlay_src: NP_INT = g_nlay

    l_counts_src: NP_ARRAY[NP_INT] = l_grid_src["l_counts_x"] * l_ny_src * l_nlay_src
    l_displs_src: NP_ARRAY[NP_INT] = l_grid_src["l_displs_x"] * l_ny_src * l_nlay_src

    l_field_src: NP_ARRAY[NP_REAL] = np.empty([l_nx_src, l_ny_src, l_nlay_src], dtype = NP_REAL) # NOTE: ASSUME only using layer midpoint values for field
    l_z_src: NP_ARRAY[NP_REAL] = np.empty([l_nx_src, l_ny_src, l_nlay_src], dtype = NP_REAL) # NOTE: ASSUME only using layer midpoint values for field

    field_min = comm.bcast(field_min, root = MPI_ROOT)
    field_max = comm.bcast(field_max, root = MPI_ROOT)
    comm.Scatterv([field_src, l_counts_src, l_displs_src, MPI_REAL], l_field_src, root = MPI_ROOT)
    comm.Scatterv([z_src, l_counts_src, l_displs_src, MPI_REAL], l_z_src, root = MPI_ROOT)

    if np.any(np.isnan(l_field_src)):
        print("{}: {} has NaNs in _src".format(l_rank, rte_field_key), flush = True)

    # Get source grid - points to interpolate from
    l_x_src: NP_ARRAY[NP_REAL] = l_grid_src["x"]
    g_y: Optional[NP_ARRAY[NP_REAL]] = None
    if l_rank == MPI_ROOT:
        g_y = g_grids["01"]["y"]
    l_y_src: NP_ARRAY[NP_REAL] = comm.bcast(g_y, root = MPI_ROOT)

    l_XX_src: NP_ARRAY[NP_REAL]
    l_YY_src: NP_ARRAY[NP_REAL]
    l_XX_src, l_YY_src = np.meshgrid(l_x_src, l_y_src, indexing = "ij")
    l_XX_src = np.tile(np.expand_dims(l_XX_src, axis = 2), (1, 1, l_nlay_src))
    l_YY_src = np.tile(np.expand_dims(l_YY_src, axis = 2), (1, 1, l_nlay_src))

    l_pts_src: NP_ARRAY[NP_REAL] = \
        np.stack([l_XX_src.flatten(), l_YY_src.flatten(), l_z_src.flatten()],
            axis = 1)

    # Coarsen the field as necessary
    ### BUG SOMEWHERE IN HERE ###
    for coarse_str in l_grids_tgt.keys():
        field_out[coarse_str]: dict = {}
        # Get target layer grid - points to interpolate to
        l_ny_tgt: NP_INT = l_grids_tgt[coarse_str]["ny"]
        l_nlay_tgt: NP_INT = l_grids_tgt[coarse_str]["nlay"]

        l_counts_x: NP_ARRAY[NP_INT] = l_grids_tgt[coarse_str]["l_counts_x"]
        l_displs_x: NP_ARRAY[NP_INT] = l_grids_tgt[coarse_str]["l_displs_x"]
            
        l_counts_lay_tgt: NP_ARRAY[NP_INT] = l_counts_x * l_ny_tgt * l_nlay_tgt
        l_displs_lay_tgt: NP_ARRAY[NP_INT] = l_displs_x * l_ny_tgt * l_nlay_tgt

        l_x_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["x"]
        l_y_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["y"]
        l_z_lay_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["z_lay"]

        l_XX_lay_tgt, l_YY_lay_tgt, l_ZZ_lay_tgt = \
            np.meshgrid(l_x_tgt, l_y_tgt, l_z_lay_tgt, indexing = "ij")
        l_pts_lay_tgt: NP_ARRAY[NP_REAL] = \
            np.stack([l_XX_lay_tgt.flatten(), l_YY_lay_tgt.flatten(), l_ZZ_lay_tgt.flatten()], 
                axis = 1)

        ## Interpolate the values to regular vertical layers, and limit them
        l_field_lay_tgt: NP_ARRAY[NP_REAL] = \
            griddata(l_pts_src, l_field_src.flatten(), l_pts_lay_tgt,
                method = interp_method)
        if np.any(np.isnan(l_field_lay_tgt)):
            print("{}: {}, {} has NaNs in _lay_tgt".format(l_rank, coarse_str, rte_field_key), flush = True)
        l_field_lay_tgt[l_field_lay_tgt < field_min] = field_min
        l_field_lay_tgt[l_field_lay_tgt > field_max] = field_max

        # Reconstruct the full field
        field_lay_tgt: Optional[NP_ARRAY[NP_REAL]] = None
        if l_rank == MPI_ROOT:
            nx_tgt: NP_INT = g_grids[coarse_str]["nx"]
            ny_tgt: NP_INT = g_grids[coarse_str]["ny"]
            nlay_tgt: NP_INT = g_grids[coarse_str]["nlay"]

            field_lay_tgt = np.empty(nx_tgt * ny_tgt * nlay_tgt, dtype = NP_REAL)

        comm.Gatherv(l_field_lay_tgt,
            [field_lay_tgt, l_counts_lay_tgt, l_displs_lay_tgt, MPI_REAL],
            root = MPI_ROOT)

        if l_rank == MPI_ROOT:
            field_lay_tgt = np.reshape(field_lay_tgt, (nx_tgt, ny_tgt, nlay_tgt)) # (nx, ny, nlay)
            field_lay_tgt = np.transpose(field_lay_tgt, axes = (2, 1, 0)) # (nlay, ny, nx)

        ## Some fields need to be interpolated to regular vertical levels, too
        if dpscream_field_key in ["p", "T"]:
            # Get target level grid - points to interpolate to
            l_nlev_tgt: NP_INT = l_grids_tgt[coarse_str]["nlev"]
            l_z_lev_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["z_lev"]
            l_counts_lev_tgt: list[NP_INT] = l_counts_x * l_ny_tgt * l_nlev_tgt
            l_displs_lev_tgt: list[NP_INT] = l_displs_x * l_ny_tgt * l_nlev_tgt

            l_XX_lev_tgt, l_YY_lev_tgt, l_ZZ_lev_tgt = \
                np.meshgrid(l_x_tgt, l_y_tgt, l_z_lev_tgt, indexing = "ij")
            l_pts_lev_tgt: NP_ARRAY[NP_REAL] = \
                np.stack([l_XX_lev_tgt.flatten(), l_YY_lev_tgt.flatten(), l_ZZ_lev_tgt.flatten()], 
                    axis = 1)

            ## Interpolate the values to regular vertical levels, and limit them
            l_field_lev_tgt: NP_ARRAY[NP_REAL] = \
                griddata(l_pts_src, l_field_src.flatten(), l_pts_lev_tgt,
                    method = interp_method)
            if np.any(np.isnan(l_field_lev_tgt)):
                print("{}: {}, {} has NaNs in _lev_tgt".format(l_rank, coarse_str, rte_field_key), flush = True)
            l_field_lev_tgt[l_field_lev_tgt < field_min] = field_min
            l_field_lev_tgt[l_field_lev_tgt > field_max] = field_max

            # Reconstruct the full field
            field_lev_tgt: Optional[NP_ARRAY[NP_REAL]] = None
            if l_rank == MPI_ROOT:
                nlev_tgt: NP_INT = g_grids[coarse_str]["nlev"]
                field_lev_tgt = np.empty(nx_tgt * ny_tgt * nlev_tgt, dtype = NP_REAL)
            
            comm.Gatherv(l_field_lev_tgt,
                [field_lev_tgt, l_counts_lev_tgt, l_displs_lev_tgt, MPI_REAL],
                root = MPI_ROOT)

            if l_rank == MPI_ROOT:
                field_lev_tgt = np.reshape(field_lev_tgt, (nx_tgt, ny_tgt, nlev_tgt)) # (nx, ny, nlev)
                field_lev_tgt = np.transpose(field_lev_tgt, axes = (2, 1, 0)) # (nlev, ny, nx)

        if l_rank == MPI_ROOT:
            ## Exceptions
            if rte_field_key in ["rh", "q", "lwp", "iwp", "rel", "dei",
                "vmr_ch4", "vmr_co", "vmr_co2", "vmr_h2o", "vmr_n2", "vmr_n2o",
                "vmr_o2", "vmr_o3"]:
                field_out[coarse_str][rte_field_key] = field_lay_tgt
            else:
                rte_field_key_lay: str = rte_field_key + "_lay"
                field_out[coarse_str][rte_field_key_lay] = field_lay_tgt

                if dpscream_field_key in ["p", "T"]:
                    rte_field_key_lev: str = rte_field_key + "_lev"
                    field_out[coarse_str][rte_field_key_lev] = field_lev_tgt

    return field_out