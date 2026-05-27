# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np
import xarray as xr

from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET, \
    R_d, R_v, mu_d
from consts.dp_screamxx_fields import dpscream_3dfield_keys
from consts.rte_rrtmgp_cpp_fields import rte_3dfield_keys

def coarsen_3d_fields(dp_scream_file: str, time_idxs: NP_ARRAY[NP_INT],
    sort_mask: NP_ARRAY[NP_INT], g_grids: dict, l_grid_src: dict, l_grids_tgt: dict, 
    comm: MPI_COMM, interp_method: str = "nearest") -> dict:
    #---------------------------------------------------------------------------
    # Get MPI communicator information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    #---------------------------------------------------------------------------
    # Each rank opens the DP-SCREAM file and extract their relevant part
    #---------------------------------------------------------------------------
    if l_rank == MPI_ROOT:
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Opening and reformatting DP-SCREAM dataset...".format(current_time)
        print(msg, flush = True)

    l_x_src = l_grid_src["x"]
    fields_tgt: Optional[dict] = None
    if l_rank == MPI_ROOT:
        fields_tgt = {}
        for coarse_str in l_grids_tgt.keys():
            fields_tgt[coarse_str] = {}

    with (xr.open_dataset(dp_scream_file, engine = "netcdf4", 
        decode_timedelta = False)
        .isel(time = time_idxs, ncol = sort_mask)
        .rename({"lat": "y", "lon": "x"})
        .set_index(ncol = ["y", "x"])
        .unstack("ncol")
        .transpose(..., "y", "x")
        .sel(x = slice(l_x_src.min(), l_x_src.max()))) as xr_dp_scream:

        #-----------------------------------------------------------------------
        # Set up dict for holding vertically-remapped field values
        #-----------------------------------------------------------------------
        l_fields_vremap: dict = {}

        #-----------------------------------------------------------------------
        # Set up dict for holding horizontally coarsened (i.e., tgt) field values
        #-----------------------------------------------------------------------
        l_fields_tgt: dict = {}
        for coarse_str in l_grids_tgt.keys():
            l_fields_tgt[coarse_str] = {}

        #-----------------------------------------------------------------------
        # Get source grid information for horizontal coarsening
        #-----------------------------------------------------------------------
        l_x_src: NP_ARRAY[NP_REAL] = l_grid_src["x"]
        l_y_src: NP_ARRAY[NP_REAL] = l_grid_src["y"]

        # ASSUME: Constant spacing in x- and y-.
        l_dx_src: NP_REAL = l_x_src[1] - l_x_src[0]
        l_dy_src: NP_REAL = l_y_src[1] - l_y_src[0]

        #-----------------------------------------------------------------------
        # All mixing ratios are moist mixing ratios. 
        # To calculate LWP, IWP later, we want mass of cloud liquid/ice
        # water in each cell. For this, we need mass of moist air first.
        #-----------------------------------------------------------------------
        l_p_src: NP_ARRAY[NP_REAL] # Pressure [Pa], [nt, l_nx, ny, nlay]
        l_T_src: NP_ARRAY[NP_REAL] # Temperature [K], [nt, l_nx, ny, nlay]
        l_qv_src: NP_ARRAY[NP_REAL] # Specific Humidity [kg kg^{-1}], [nt, l_nx, ny, nlay]
        l_z_lev_src: NP_ARRAY[NP_REAL]  # Level height [m], [nt, l_nx, ny, nlev]
        [_, l_p_src] = extract_dp_scream_field(xr_dp_scream, "p_mid") 
        [_, l_T_src] = extract_dp_scream_field(xr_dp_scream, "T_mid")
        [_, l_qv_src] = extract_dp_scream_field(xr_dp_scream, "qv")
        [_, l_z_lev_src] = extract_dp_scream_field(xr_dp_scream, "z_int")
                
        l_dz_src: NP_ARRAY[NP_REAL] = l_z_lev_src[:,:,:,1:] - l_z_lev_src[:,:,:,:-1] # Layer thickness [m], [nt, l_nx, ny, nlay]
        l_V_src: NP_ARRAY[NP_REAL] = l_dx_src * l_dy_src * l_dz_src # Cell volume [m^{3}], [nt, l_nx, ny, nlay]

        l_M_src: NP_ARRAY[NP_REAL] = (l_p_src * l_V_src) / (R_d * l_T_src * (1. + l_qv_src * ((R_v / R_d) - 1.))) # Moist air mass [kg], [nt, l_nx, ny, nlay]
        l_nd_src: NP_ARRAY[NP_REAL] = l_M_src * (1. - l_qv_src) / mu_d # Moles of dry air [mol], [nt, l_nx, ny, nz]

        #-----------------------------------------------------------------------
        # Loop through RTE-RRTMGP-CPP+RT fields, remap verticaly
        #-----------------------------------------------------------------------
        for rad_tran_key in ["lwp"]:#rte_3dfield_keys:
            if l_rank == MPI_ROOT:
                current_time = datetime.now().strftime("%H:%M:%S")
                msg: str = "[{}]: Extracting DP-SCREAM field(s) for {}...".format(current_time, rad_tran_key)
                print(msg, flush = True)
            #-------------------------------------------------------------------
            # Extract relevant fields from DP-SCREAM file
            #-------------------------------------------------------------------
            if rad_tran_key in ["p", "t"]:
                dp_scream_key = rad_tran_key
            elif rad_tran_key == "rh":
                dp_scream_key = "RelativeHumidity"
            elif rad_tran_key == "rel":
                dp_scream_key = "eff_radius_qc"
            elif rad_tran_key == "dei":
                dp_scream_key = "eff_radius_qi"
            elif rad_tran_key == "lwp":
                dp_scream_key = "qc"
            elif rad_tran_key == "iwp":
                dp_scream_key = "qi"
            elif "vmr_" in rad_tran_key:
                dp_scream_key = rad_tran_key[4:] + "_volume_mix_ratio"

            l_z_src: NP_ARRAY[NP_REAL] # [nt, l_nx, ny, nz]
            l_field_src: NP_ARRAY[NP_REAL] # [nt, l_nx, ny, nz]
            [l_z_src, l_field_src] = extract_dp_scream_field(xr_dp_scream, dp_scream_key)

            # DP-SCREAM uses ice-water effective radius
            # RT-RRTMGP-CPP+RT use ice-water effective diameter
            if rad_tran_key == "dei":
                l_field_src *= 2.
            # Mixing ratios to masses/moles for proportional split
            elif rad_tran_key in ["lwp", "iwp"]:
                l_field_src = l_field_src * l_M_src # Mass [kg], [nt, l_nx, ny, nz]
            elif rad_tran_key in ["vmr_ch4", "vmr_co", "vmr_co2", "vmr_h2o",
                "vmr_n2", "vmr_n2o", "vmr_o2", "vmr_o3"]:
                l_field_src = l_field_src * l_nd_src # Moles of gas [mole], [nt, l_nx, ny, nz]

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
            z_lev_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt["01"]["z_lev"]

            #-------------------------------------------------------------------
            # Remap field values vertically
            #-------------------------------------------------------------------
            if l_rank == MPI_ROOT:
                current_time = datetime.now().strftime("%H:%M:%S")
                msg: str = "[{}]: Vertically remapping {}...".format(current_time, rad_tran_key)
                print(msg, flush = True)
            # TO-DO: HANDLE LWP, IWP, VMR
            # TO-DO: ENSURE THAT TIME_DIMENSION IS ACCOUNTED FOR
            l_nt: NP_INT
            l_nx_src: NP_INT
            l_ny_src: NP_INT
            l_nz_src: NP_INT
            [l_nt, l_nx_src, l_ny_src, l_nz_src] = NP_INT(l_field_src.shape)
            l_field_vremap: NP_ARRAY[NP_REAL] = np.empty([l_nt, l_nx_src, l_ny_src, l_nz_src], dtype = NP_REAL)
            if rad_tran_key in ["p", "t", "rh", "rel", "dei"]:
                for tt in range(0, l_nt):
                    for ii in range(0, l_nx_src):
                        for jj in range(0, l_ny_src):
                            l_field_vremap[tt, ii, jj, :] = np.interp(z_tgt, l_z_src[tt, ii, jj, :], l_field_src[tt, ii, jj, :])
            elif rad_tran_key in ["lwp", "iwp", "vmr_ch4", "vmr_co", "vmr_co2",
                "vmr_h2o", "vmr_n2", "vmr_n2o", "vmr_o2", "vmr_o3"]:
                if l_rank == MPI_ROOT:
                    l_field_vremap = remap_layer_mass(z_lev_tgt, l_z_lev_src, l_field_src)
                comm.barrier()
                
            #-------------------------------------------------------------------
            # Store vertically-remapped field values
            #-------------------------------------------------------------------
            l_fields_vremap[rad_tran_key] = l_field_vremap

            #-------------------------------------------------------------------
            # Coarsen field values horizontally
            #-------------------------------------------------------------------
            l_horz_coarsener: list[RegularGridInterpolator] = \
                [[RegularGridInterpolator((l_x_src, l_y_src), l_field_vremap[tt,:,:,kk], method = interp_method)
                    for kk in range(0, l_nz_src)] for tt in range(0, l_nt)]
            for coarse_str in l_grids_tgt.keys():
                if l_rank == MPI_ROOT:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    msg: str = "[{}]: Coarsening {} to lr_{}...".format(current_time, rad_tran_key, coarse_str)
                    print(msg, flush = True)
                #---------------------------------------------------------------
                # Get target horizontal grid and communication parameters
                #---------------------------------------------------------------
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

                l_counts_tgt: NP_ARRAY[NP_INT] = l_nt * l_nz_tgt * l_ny_tgt * l_counts_x
                l_displs_tgt: NP_ARRAY[NP_INT] = l_nt * l_nz_tgt * l_ny_tgt * l_displs_x

                #---------------------------------------------------------------
                # Horizontally coarsen the field
                #---------------------------------------------------------------
                l_field_tgt: NP_ARRAY[NP_REAL] = \
                    np.empty([l_nt, l_nx_tgt, l_ny_tgt, l_nz_tgt], dtype = NP_REAL)
                for tt in range(0, l_nt):
                    for kk in range(0, l_nz_src):
                        l_field_tgt[tt,:,:,kk] = l_horz_coarsener[tt][kk](l_pts_tgt).reshape(l_nx_tgt, l_ny_tgt)

                #---------------------------------------------------------------
                # Store horizontally-coarsened field values
                #---------------------------------------------------------------
                l_fields_tgt[coarse_str][rad_tran_key] = l_field_tgt

                #---------------------------------------------------------------
                # Gatherv the whole field onto MPI_ROOT
                #---------------------------------------------------------------
                # Reshape l_field_tgt for easier Gatherv
                l_field_tgt = np.ascontiguousarray(np.transpose(l_field_tgt, axes = [1, 0, 2, 3])) # [nx, nt, ny, nz]
                field_tgt: Optional[NP_ARRAY[NP_REAL]] = None
                if l_rank == MPI_ROOT:
                    nt_tgt: NP_INT = l_nt
                    nx_tgt: NP_INT = g_grids[coarse_str]["nx"]
                    ny_tgt: NP_INT = l_ny_tgt
                    nz_tgt: NP_INT = l_nz_tgt
        
                    field_tgt = np.empty(nt_tgt * nx_tgt * ny_tgt * nz_tgt, dtype = NP_REAL)

                comm.Gatherv(l_field_tgt, 
                    [field_tgt, l_counts_tgt, l_displs_tgt, MPI_REAL],
                    root = MPI_ROOT)

                # At this point, field_tgt is a concatenation of comm_size
                # arrays of length nt * l_nx * ny * nz. 
                if l_rank == MPI_ROOT:
                    field_tgt = np.ascontiguousarray(
                        np.transpose(field_tgt.reshape(nx_tgt, nt_tgt, ny_tgt, nz_tgt), 
                            axes = [1, 3, 2, 0])) # [nt, nz, ny, nx]

                    if rad_tran_key in ["p", "t"]:
                        fields_tgt[coarse_str][rad_tran_key + "_lay"] = field_tgt[:,:,:,1::2]
                        fields_tgt[coarse_str][rad_tran_key + "_lev"] = field_tgt[:,:,:,0::2]
                    else:
                        fields_tgt[coarse_str][rad_tran_key] = field_tgt

        return fields_tgt
"""
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
"""

def extract_dp_scream_field(xr_dp_scream: XR_DATASET, dp_scream_key: str) -> tuple[NP_ARRAY[NP_REAL]]:
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
        [g_nt, g_nlay, g_ny, l_nx] = NP_INT(z_lay_src.shape)
        [_,    g_nlev, _,    _]    = NP_INT(z_lev_src.shape)
        g_nz = g_nlay + g_nlev

        z_src: NP_ARRAY[NP_REAL] = np.empty([g_nt, g_nz, g_ny, l_nx], dtype = NP_REAL)
        z_src[:, 0::2, :, :] = z_lev_src
        z_src[:, 1::2, :, :] = z_lay_src

        field_src: NP_ARRAY[NP_REAL] = np.empty([g_nt, g_nz, g_ny, l_nx], dtype = NP_REAL)
        field_src[:, 0::2, :, :] = field_lev_src
        field_src[:, 1::2, :, :] = field_lay_src
    elif src_data_at_layers and not src_data_at_levels:
        [g_nt, g_nz, g_ny, l_nx] = NP_INT(z_lay_src.shape)

        z_src: NP_ARRAY[NP_REAL] = z_lay_src
        field_src: NP_ARRAY[NP_REAL] = field_lay_src
    elif not src_data_at_layers and src_data_at_levels:
        [g_nt, g_nz, g_ny, l_nx] = NP_INT(z_lev_src.shape)

        z_src: NP_ARRAY[NP_REAL] = z_lev_src
        field_src: NP_ARRAY[NP_REAL] = field_lev_src

    np.nan_to_num(field_src, NP_REAL(0.))

    z_src = np.transpose(z_src, axes = [0, 3, 2, 1]) # [nt, nx, ny, nz]
    field_src = np.transpose(field_src, axes = [0, 3, 2, 1]) # [nt, nx, ny, nz]

    return [z_src[:,:,:,::-1], field_src[:,:,:,::-1]] # Flip to increasing z

def remap_layer_mass(l_z_lev_tgt: NP_ARRAY[NP_REAL], l_z_lev_src: NP_ARRAY[NP_REAL],
    l_mass_src: NP_ARRAY[NP_REAL]) -> NP_ARRAY[NP_REAL]:

    #---------------------------------------------------------------------------
    # Get dimensions
    #---------------------------------------------------------------------------
    nt: NP_INT
    l_nx: NP_INT
    ny: NP_INT
    nlay: NP_INT
    [nt, l_nx, ny, nlay] = NP_INT(l_mass_src.shape)

    breakpoint()

    #---------------------------------------------------------------------------
    # Compute source/target layer lower/upper bounds, get lay thicknesses
    #---------------------------------------------------------------------------
    l_z_lo_src: NP_ARRAY[NP_REAL] = l_z_lev_src[:,:,:,:-1] # [nt, l_nx, ny, nlay]
    l_z_hi_src: NP_ARRAY[NP_REAL] = l_z_lev_src[:,:,:,1:] # [nt, l_nx, ny, nlay]

    l_z_lo_tgt: NP_ARRAY[NP_REAL] = l_z_lev_tgt[:-1] # [nlay]
    l_z_hi_tgt: NP_ARRAY[NP_REAL] = l_z_lev_tgt[1:]  # [nlay]

    l_dz_src: NP_ARRAY[NP_REAL] = l_z_hi_src - l_z_lo_src # [nt, l_nx, ny, nlay]

    #---------------------------------------------------------------------------
    # Expand source arrays for pairwise source-target overlap
    #---------------------------------------------------------------------------
    l_z_lo_src_2d: NP_ARRAY[NP_REAL] = l_z_lo_src[:,:,:,:,None] # [nt, nx, ny, nlay, 1]
    l_z_hi_src_2d: NP_ARRAY[NP_REAL] = l_z_hi_src[:,:,:,:,None] # [nt, nx, ny, nlay, 1]
    l_dz_src_2d: NP_ARRAY[NP_REAL] = l_dz_src[:,:,:,:,None]     # [nt, nx, ny, nlay, 1]

    #---------------------------------------------------------------------------
    # Compute overlap thickness between every source layer and target layer
    # overlap = max(0, min(src_hi, tgt_hi) - max(src_lo, tgt_lo))
    #---------------------------------------------------------------------------
    l_overlap_lo: NP_ARRAY[NP_REAL] = np.maximum(l_z_lo_src_2d, l_z_lo_tgt)
    l_overlap_hi: NP_ARRAY[NP_REAL] = np.minimum(l_z_hi_src_2d, l_z_hi_tgt)

    l_overlap: NP_ARRAY[NP_REAL] = np.maximum(NP_REAL(0.), l_overlap_hi - l_overlap_lo) # [nt, nx, ny, nlay, nlay]

    #---------------------------------------------------------------------------
    # Compute fraction of each source layer contributing to each target layer
    #---------------------------------------------------------------------------
    l_frac: NP_ARRAY[NP_REAL] = np.zeros([nt, l_nx, ny, nlay, nlay], dtype = NP_REAL)

    l_mask_nonzero: NP_ARRAY[NP_BOOL] = (l_dz_src_2d > NP_REAL(0.))
    l_frac[l_mask_nonzero] = l_overlap[l_mask_nonzero] / l_dz_src_2d[l_mask_nonzero]

    #---------------------------------------------------------------------------
    # Distribute source mass into target layers
    #---------------------------------------------------------------------------
    l_mass_contrib: NP_ARRAY[NP_REAL] = l_mass_src[:,:,:,:,None] * l_frac
    l_mass_tgt: NP_ARRAY[NP_REAL] = np.sum(l_mass_contrib, axis = 3) # [nt, l_nx, ny, nlay]

    return l_mass_tgt