# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET, \
    g
from consts.dp_screamxx_fields import dpscream_3dfield_keys
from consts.rte_rrtmgp_cpp_fields import rte_3dfield_keys

def interp_3dfield(dp_scream_file: str, time_idxs: NP_ARRAY[NP_INT],
    sort_mask: NP_ARRAY[NP_INT], l_grid_src: dict, l_grids_tgt: dict, 
    comm: MPI_COMM, interp_method: str = "nearest") -> dict:
    #---------------------------------------------------------------------------
    # Get MPI communicator information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    #---------------------------------------------------------------------------
    # Each rank opens the DP-SCREAM file and extract their relevant part
    #---------------------------------------------------------------------------
    l_x_src = l_grid_src["x"]
    with (xr.open_dataset(dp_scream_file, engine = "netcdf4", 
        decode_timedelta = False)
        .isel(time = time_idxs, ncol = sort_mask)
        .rename({"lat": "y", "lon": "x"})
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")
        .transpose(..., "y", "x")
        .sel(x = slice(l_x_src.min(), l_x_src.max()))) as xr_dp_scream:

        #-----------------------------------------------------------------------
        # Loop through RTE-RRTMGP-CPP+RT fields
        #-----------------------------------------------------------------------
        for rad_tran_key in rte_3dfield_keys:
            #-------------------------------------------------------------------
            # Extract relevant fields from DP-SCREAM file
            #-------------------------------------------------------------------
            # TO-DO: HANDLE LWP, IWP
            if rad_tran_key in ["p", "t", "rh", "rel", "dei", "vmr_ch4", "vmr_co", 
                    "vmr_co2", "vmr_h2o", "vmr_n2", "vmr_n2o", "vmr_o2", 
                    "vmr_o3"]:
                if rad_tran_key in ["p", "t"]:
                    dp_scream_key = rad_tran_key
                elif rad_tran_key == "rh":
                    dp_scream_key = "RelativeHumidity"
                elif rad_tran_key == "rel":
                    dp_scream_key = "eff_radius_qc"
                elif rad_tran_key == "dei":
                    dp_scream_key = "eff_radius_qi"
                elif "vmr_" in rad_tran_key:
                    dp_scream_key = rad_tran_key[4:] + "_volume_mix_ratio"

                [l_z_src, l_field_src] = extract_dp_scream_field(xr_dp_scream, dp_scream_key)

                # DP-SCREAM uses ice-water effective radius
                # RAT-RRTMGP-CPP+RT use ice-water effective diameter
                if rad_tran_key == "dei":
                    l_field_src *= 2.

            #-------------------------------------------------------------------
            # Obtain target vertical grid
            # ASSUME VERTICAL GRID IS SAME FOR SRC, TGT, AND ALL COARSENINGS
            #-------------------------------------------------------------------
            if rad_tran_key in ["p", "t"]:
                tgt_data_at_layers = True
                tgt_data_at_levels = True
            else:
                tgt_data_at_layers = True
                tgt_data_at_levels = False

            assert(tgt_data_at_layers or tgt_data_at_levels)
            l_nz_tgt: NP_INT
            z_tgt: NP_ARRAY[NP_REAL]
            if tgt_data_at_layers and tgt_data_at_levels:
                l_nz_tgt = l_grids_tgt["01"]["nz"]
                z_tgt = l_grids_tgt["01"]["z"]
            elif not tgt_data_at_layers and tgt_data_at_levels:
                l_nz_tgt = l_grids_tgt["01"]["nlev"]
                z_tgt = l_grids_tgt["01"]["z_lev"]
            elif tgt_data_at_layers and not tgt_data_at_levels:
                l_nz_tgt = l_grids_tgt["01"]["nlay"]
                z_tgt = l_grids_tgt["01"]["z_lay"]

            #-------------------------------------------------------------------
            # Remap field values vertically
            #-------------------------------------------------------------------
            # TO-DO: HANDLE LWP, IWP
            l_nx_src: NP_INT = l_grid_src["nx"]
            l_ny_src: NP_INT = l_grid_src["ny"]
            l_field_vremap: NP_ARRAY[NP_REAL] = np.empty([l_nx_src, l_ny_src, l_nz_tgt], dtype = NP_REAL)
            for ii in range(0, l_nx_src):
                for jj in range(0, l_ny_src):
                    l_field_vremap[ii, jj, :] = np.interp(z_tgt, l_z_src[ii, jj, :], l_field_src[ii, jj, :])
            
            if l_rank == MPI_ROOT:
                    breakpoint()
            comm.barrier()

            # CONTINUE FROM HERE
    #---------------------------------------------------------------------------
    # Open the relevant part of the DP-SCREAM file
    #---------------------------------------------------------------------------

    field_out: dict = {}

    src_data_at_layers: Optional[bool] = None
    src_data_at_levels: Optional[bool] = None
    tgt_data_at_layers: Optional[bool] = None
    tgt_data_at_levels: Optional[bool] = None

    g_nx: Optional[NP_INT] = None
    g_ny: Optional[NP_INT] = None
    g_nlay: Optional[NP_INT] = None
    g_nlev: Optional[NP_INT] = None
    g_nz: Optional[NP_INT] = None
    z_src: Optional[NP_ARRAY[NP_REAL]] = None
    z_tgt: Optional[NP_ARRAY[NP_REAL]] = None
    field_src: Optional[NP_ARRAY[NP_REAL]] = None
    field_min: Optional[NP_REAL] = None
    field_max: Optional[NP_REAL] = None
    # Root Rank reads input file, constructs full field and Scatterv
    if l_rank == MPI_ROOT:
        g_nx = g_grids["01"]["nx"]
        g_ny = g_grids["01"]["ny"]
        g_nlay = g_grids["01"]["nlay"]
        g_nlev = g_grids["01"]["nlev"]

        ncol: NP_INT = NP_INT(xr_dpscream.sizes["ncol"])

        # Get src_data information
        if dpscream_field_key in xr_dpscream.keys():
            src_data_at_layers = True
            src_data_at_levels = False

            z_lay_src: NP_ARRAY[NP_REAL] = \
                xr_dpscream["z_mid"].isel(time = tt, ncol = sort_mask, lev = slice(None, None, -1)).values.astype(NP_REAL) # Layer midpoints [m]; (ncol, n_lay_z)
            
            field_lay_src: NP_ARRAY[NP_REAL] = \
                xr_dpscream[dpscream_field_key].isel(time = tt, ncol = sort_mask, lev = slice(None, None, -1)).values.astype(NP_REAL) # Field at layer midpoints; (ncol, n_lay_z)
        else:
            dpscream_field_key_mid: str = dpscream_field_key + "_mid"
            dpscream_field_key_int: str = dpscream_field_key + "_int"

            assert((dpscream_field_key_mid in xr_dpscream.keys()) or 
                (dpscream_field_key_int in xr_dpscream.keys()))

            if dpscream_field_key_mid in xr_dpscream.keys():
                src_data_at_layers = True

                z_lay_src: NP_ARRAY[NP_REAL] = \
                    xr_dpscream["z_mid"].isel(time = tt, ncol = sort_mask, lev = slice(None, None, -1)).values.astype(NP_REAL) # Layer midpoints [m]; (ncol, n_lay_z)

                field_lay_src: NP_ARRAY[NP_REAL] = \
                    xr_dpscream[dpscream_field_key_mid].isel(time = tt, ncol = sort_mask, lev = slice(None, None, -1)).values.astype(NP_REAL) # Field at layer midpoints; (ncol, n_lay_z)
            else:
                src_data_at_layers = False
            if dpscream_field_key_int in xr_dpscream.keys():
                src_data_at_levels = True

                z_lev_src: NP_ARRAY[NP_REAL] = \
                    xr_dpscream["z_int"].isel(time = tt, ncol = sort_mask, ilev = slice(None, None, -1)).values.astype(NP_REAL) # Layer interfaces [m]; (ncol, n_lev_z)

                field_lev_src: NP_ARRAY[NP_REAL] = \
                    xr_dpscream[dpscream_field_key_int].isel(time = tt, ncol = sort_mask, ilev = slice(None, None, -1)).values.astype(NP_REAL) # Field at layer interfaces; (ncol, n_lev_z)
            else:
                src_data_at_levels = False

        assert(src_data_at_layers or src_data_at_levels)
        if src_data_at_layers and src_data_at_levels:
            g_nz = g_nlay + g_nlev

            z_src: NP_ARRAY[NP_REAL] = np.empty([ncol, g_nz], dtype = NP_REAL)
            z_src[:, 0::2] = z_lev_src
            z_src[:, 1::2] = z_lay_src

            field_src: NP_ARRAY[NP_REAL] = np.empty([ncol, g_nz], dtype = NP_REAL)
            field_src[:, 0::2] = field_lev_src
            field_src[:, 1::2] = field_lay_src
        elif src_data_at_layers and not src_data_at_levels:
            g_nz = g_nlay

            z_src: NP_ARRAY[NP_REAL] = z_lay_src
            field_src: NP_ARRAY[NP_REAL] = field_lay_src
        elif not src_data_at_layers and src_data_at_levels:
            g_nz = g_nlev

            z_src: NP_ARRAY[NP_REAL] = z_lev_src
            field_src: NP_ARRAY[NP_REAL] = field_lev_src

        np.nan_to_num(field_src, NP_REAL(0.))
        
        ## Exceptions - Do in serial on MPI_ROOT for now
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
        elif rte_field_key in ["t"]:
            field_min = NP_REAL(100.0) # Lowest temperature in mesosphere https://scied.ucar.edu/learning-zone/atmosphere/mesosphere [K]
            field_max = NP_REAL(329.817) # Hottest observed temperature https://www.ncei.noaa.gov/news/earths-hottest-temperature [K]
        elif rte_field_key in ["p"]:
            field_min = NP_REAL(0.1) # Atmospheric pressure near top of mesosphere. [Pa]
            field_max = NP_REAL(108480.0) # Highest recorded atmospheric pressure https://web.archive.org/web/20121017130834/http://wmo.asu.edu/highest-sea-lvl-air-pressure-above-700m [Pa]
        else:
            field_min = field_src.min()
            field_max = field_src.max()

        field_src[field_src > field_max] = field_max
        field_src[field_src < field_min] = field_min

        z_src = z_src.reshape(g_nx, g_ny, g_nz)
        field_src = field_src.reshape(g_nx, g_ny, g_nz)

        # Get tgt_data information
        if rte_field_key in ["p", "t"]:
            tgt_data_at_layers = True
            tgt_data_at_levels = True
        else:
            tgt_data_at_layers = True
            tgt_data_at_levels = False

        assert(tgt_data_at_layers or tgt_data_at_levels)
        z_tgt: NP_ARRAY[NP_REAL]
        if tgt_data_at_layers and tgt_data_at_levels:
            z_tgt = g_grids["01"]["z"] # ASSUME VERTICAL GRID IS SAME FOR SRC, TGT, AND ALL COARSENINGS
        elif tgt_data_at_layers and not tgt_data_at_levels:
            z_tgt = g_grids["01"]["z_lay"] # ASSUME VERTICAL GRID IS SAME FOR SRC, TGT, AND ALL COARSENINGS
        elif not tgt_data_at_layers and tgt_data_at_levels:
            z_tgt = g_grids["01"]["z_lev"] # ASSUME VERTICAL GRID IS SAME FOR SRC, TGT, AND ALL COARSENINGS

    g_nx = comm.bcast(g_nx, root = MPI_ROOT)
    g_ny = comm.bcast(g_ny, root = MPI_ROOT)
    g_nz = comm.bcast(g_nz, root = MPI_ROOT)
    z_tgt = comm.bcast(z_tgt, root = MPI_ROOT)
    field_min = comm.bcast(field_min, root = MPI_ROOT)
    field_max = comm.bcast(field_max, root = MPI_ROOT)

    # Scatterv the original field
    l_nx_src: NP_INT = l_grid_src["nx"]
    l_ny_src: NP_INT = g_ny
    l_nz_src: NP_INT = g_nz
    l_nz_tgt: NP_INT = z_tgt.size ## ASSUME: z_tgt constant for all coarsenings

    l_counts_src: NP_ARRAY[NP_INT] = l_grid_src["l_counts_x"] * l_ny_src * l_nz_src
    l_displs_src: NP_ARRAY[NP_INT] = l_grid_src["l_displs_x"] * l_ny_src * l_nz_src

    l_field_src: NP_ARRAY[NP_REAL] = np.empty([l_nx_src, l_ny_src, l_nz_src], dtype = NP_REAL)
    l_z_src: NP_ARRAY[NP_REAL] = np.empty([l_nx_src, l_ny_src, l_nz_src], dtype = NP_REAL)

    comm.Scatterv([field_src, l_counts_src, l_displs_src, MPI_REAL], l_field_src, root = MPI_ROOT)
    comm.Scatterv([z_src, l_counts_src, l_displs_src, MPI_REAL], l_z_src, root = MPI_ROOT)

    # Interpolate the field vertically on each process
    l_field_vinterp: NP_ARRAY[NP_REAL] = np.empty([l_nx_src, l_ny_src, l_nz_tgt], dtype = NP_REAL)
    for ii in range(0, l_nx_src):
        for jj in range(0, l_ny_src):
            l_field_vinterp[ii, jj, :] = np.interp(z_tgt, l_z_src[ii, jj, :], l_field_src[ii, jj, :])
    
    if np.any(np.isnan(l_field_vinterp)):
        print("{}: {} has NaNs in _vinterp".format(l_rank, rte_field_key), flush = True)

    # Get source grid - points to interpolate from
    l_x_src: NP_ARRAY[NP_REAL] = l_grid_src["x"]
    g_y: Optional[NP_ARRAY[NP_REAL]] = None
    if l_rank == MPI_ROOT:
        g_y = g_grids["01"]["y"]
    l_y_src: NP_ARRAY[NP_REAL] = comm.bcast(g_y, root = MPI_ROOT)

    # I THINK THE CODE FROM HERE AND LOWER IS RELEVANT TO THE RE-WORK ABOVE

    # Create an inteprolator to evaluate at target points
    l_horz_interpolator: list[RegularGridInterpolator] = \
        [RegularGridInterpolator((l_x_src, l_y_src), l_field_vinterp[:,:,ii], method = interp_method)
        for ii in range(0, l_nz_tgt)]

    # Coarsen the field as necessary
    for coarse_str in l_grids_tgt.keys():
        field_out[coarse_str]: dict = {}
        # Get target horizontal grid - points to interpolate to
        l_nx_tgt: NP_INT = l_grids_tgt[coarse_str]["nx"]
        l_ny_tgt: NP_INT = l_grids_tgt[coarse_str]["ny"]

        l_counts_x: NP_ARRAY[NP_INT] = l_grids_tgt[coarse_str]["l_counts_x"]
        l_displs_x: NP_ARRAY[NP_INT] = l_grids_tgt[coarse_str]["l_displs_x"]
            
        l_counts_tgt: NP_ARRAY[NP_INT] = l_counts_x * l_ny_tgt * l_nz_tgt
        l_displs_tgt: NP_ARRAY[NP_INT] = l_displs_x * l_ny_tgt * l_nz_tgt

        l_x_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["x"]
        l_y_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["y"]

        l_XX_tgt, l_YY_tgt = \
            np.meshgrid(l_x_tgt, l_y_tgt, indexing = "ij")
        l_pts_tgt: NP_ARRAY[NP_REAL] = \
            np.stack([l_XX_tgt.flatten(), l_YY_tgt.flatten()], 
                axis = 1)

        ## Interpolate the values to regular vertical layers, and limit them
        l_field_tgt: NP_ARRAY[NP_REAL] = \
            np.empty([l_nx_tgt, l_ny_tgt, l_nz_tgt], dtype = NP_REAL)
        for kk in range(0, l_nz_tgt):
            l_field_tgt[:,:,kk] = l_horz_interpolator[kk](l_pts_tgt).reshape(l_nx_tgt, l_ny_tgt)
        if np.any(np.isnan(l_field_tgt)):
            print("{}: {}, {} has NaNs in _tgt".format(l_rank, coarse_str, rte_field_key), flush = True)
        l_field_tgt[l_field_tgt < field_min] = field_min
        l_field_tgt[l_field_tgt > field_max] = field_max

        # Reconstruct the full field
        field_tgt: Optional[NP_ARRAY[NP_REAL]] = None
        if l_rank == MPI_ROOT:
            nx_tgt: NP_INT = g_grids[coarse_str]["nx"]
            ny_tgt: NP_INT = g_grids[coarse_str]["ny"]
            nz_tgt: NP_INT = z_tgt.size

            field_tgt = np.empty(nx_tgt * ny_tgt * nz_tgt, dtype = NP_REAL)

        comm.Gatherv(l_field_tgt,
            [field_tgt, l_counts_tgt, l_displs_tgt, MPI_REAL],
            root = MPI_ROOT)

        if l_rank == MPI_ROOT:
            field_tgt = np.reshape(field_tgt, (nx_tgt, ny_tgt, nz_tgt)) # (nx, ny, nz)
            field_tgt = np.transpose(field_tgt, axes = (2, 1, 0)) # (nz, ny, nx)
            ## At this point, some fields are interpolated on layers and levels
            ## Here, we separate them
            field_lay_tgt: Optional[NP_ARRAY[NP_REAL]] = None
            field_lev_tgt: Optional[NP_ARRAY[NP_REAL]] = None
            assert(tgt_data_at_layers or tgt_data_at_levels)
            if tgt_data_at_layers and tgt_data_at_levels:
                field_lay_tgt = field_tgt[1::2,...]
                field_lev_tgt = field_tgt[0::2,...]
            elif tgt_data_at_layers and not tgt_data_at_levels:
                field_lay_tgt = field_tgt
            elif not tgt_data_at_layers and tgt_data_at_levels:
                field_lev_tgt = field_tgt
        
            ## Exceptions in rte_field_key names
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

def extract_dp_scream_field(xr_dp_scream: XR_DATASET, dp_scream_key: str):
    #-------------------------------------------------------------------
    # Get field information - at levels (int), layers (mid), or both
    #-------------------------------------------------------------------
    src_data_at_layers: bool
    src_data_at_levels: bool
            
    # If dp_scream_key in dp_scream_file, then src_data only at layers
    # Else, then dp_scream_key needs to be parsed further
    if dp_scream_key in xr_dp_scream.keys():
        src_data_at_layers = True
        src_data_at_levels = False

        z_lay_src: NP_ARRAY[NP_REAL] = xr_dp_scream["z_mid"].to_numpy().astype(NP_REAL) # Layer midpoints [m]; (ncol, n_lay_z)
        field_lay_src: NP_ARRAY[NP_REAL] = xr_dp_scream[dp_scream_key].to_numpy().astype(NP_REAL) # Field at layer midpoints; (ncol, n_lay_z)
    else:
        # Default form of keys
        dp_scream_key_mid: str = dp_scream_key + "_mid"
        dp_scream_key_int: str = dp_scream_key + "_int"

        assert((dp_scream_key_mid in xr_dp_scream.keys()) or 
            (dp_scream_key_int in xr_dp_scream.keys()))

        # Field values at layers available
        if dp_scream_key_mid in xr_dp_scream.keys():
            src_data_at_layers = True

            z_lay_src: NP_ARRAY[NP_REAL] = xr_dp_scream["z_mid"].to_numpy().astype(NP_REAL) # Layer midpoints [m]; (ncol, n_lay_z)
            field_lay_src: NP_ARRAY[NP_REAL] = xr_dp_scream[dp_scream_key_mid].to_numpy().astype(NP_REAL) # Field at layer midpoints; (ncol, n_lay_z)
        else:
            src_data_at_layers = False

            # Field values at levels available
            if dp_scream_key_int in xr_dp_scream.keys():
                src_data_at_levels = True

                z_lev_src: NP_ARRAY[NP_REAL] = xr_dp_scream["z_int"].to_numpy().astype(NP_REAL) # Layer interfaces [m]; (ncol, n_lev_z)
                field_lev_src: NP_ARRAY[NP_REAL] = xr_dp_scream[dp_scream_key_int].to_numpy().astype(NP_REAL) # Field at layer interfaces; (ncol, n_lev_z)
            else:
                src_data_at_levels = False
    
    #---------------------------------------------------------------------------
    # Interleave field values
    #---------------------------------------------------------------------------
    assert(src_data_at_layers or src_data_at_levels)
    g_nt: NP_INT
    l_nx: NP_INT
    g_ny: NP_INT
    g_nlay: NP_INT
    g_nlev: NP_INT
    g_nz: NP_INT
    
    if src_data_at_layers and src_data_at_levels:
        [g_nt, g_nlay, g_ny, l_nx] = z_lay_src.shape
        [_,    g_nlev, _,    _]    = z_lev_src.shape
        g_nz = g_nlay + g_nlev

        z_src: NP_ARRAY[NP_REAL] = np.empty([g_nt, g_nz, g_ny, l_nx], dtype = NP_REAL)
        z_src[:, 0::2, :, :] = z_lev_src
        z_src[:, 1::2, :, :] = z_lay_src

        field_src: NP_ARRAY[NP_REAL] = np.empty([g_nt, g_nz, g_ny, l_nx], dtype = NP_REAL)
        field_src[:, 0::2, :, :] = field_lev_src
        field_src[:, 1::2, :, :] = field_lay_src
    elif src_data_at_layers and not src_data_at_levels:
        [g_nt, g_nz, g_ny, l_nx] = z_lay_src.shape

        z_src: NP_ARRAY[NP_REAL] = z_lay_src
        field_src: NP_ARRAY[NP_REAL] = field_lay_src
    elif not src_data_at_layers and src_data_at_levels:
        [g_nt, g_nz, g_ny, l_nx] = z_lev_src.shape

        z_src: NP_ARRAY[NP_REAL] = z_lev_src
        field_src: NP_ARRAY[NP_REAL] = field_lev_src

    np.nan_to_num(field_src, NP_REAL(0.))
    
    return [z_src, field_src]