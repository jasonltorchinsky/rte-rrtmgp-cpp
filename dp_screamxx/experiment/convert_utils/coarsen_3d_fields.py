# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import numpy as np

from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_BOOL, NP_ARRAY, \
    MPI_REAL, MPI_COMM, MPI_ROOT, XR_DATASET, \
    R_d, R_v, mu_d
from consts.rte_rrtmgp_cpp_fields import rte_3d_field_keys

def coarsen_3d_fields(xr_dp_scream: XR_DATASET, g_grids: dict, l_grid_src: dict, 
    l_grids_tgt: dict, comm: MPI_COMM, interp_method: str = "nearest") -> dict:
    #---------------------------------------------------------------------------
    # Get MPI communicator information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    #---------------------------------------------------------------------------
    # Common variables throuhgout script
    #---------------------------------------------------------------------------
    tgt_def_key: str = list(l_grids_tgt.keys())[0] # Default key to use for l_grids_tgt
    # ASSUME: All target grids have same vertical grid
    l_nt: NP_INT = NP_INT(xr_dp_scream["time"].size)

    #---------------------------------------------------------------------------
    # Set up dict for holding vertically-remapped field values
    #---------------------------------------------------------------------------
    l_fields_vremap: dict = {}

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
    # Get source grid information for vertical remapping and horizontal coarsening
    #---------------------------------------------------------------------------
    l_x_src: NP_ARRAY[NP_REAL] = l_grid_src["x"]
    l_y_src: NP_ARRAY[NP_REAL] = l_grid_src["y"]

    # ASSUME: Constant spacing in x- and y-.
    l_dx_src: NP_REAL = l_x_src[1] - l_x_src[0]
    l_dy_src: NP_REAL = l_y_src[1] - l_y_src[0]

    l_z_lev_src: NP_ARRAY[NP_REAL]  # Level height [m], [nt, l_nx, ny, nlev]
    [_, l_z_lev_src] = extract_dp_scream_field(xr_dp_scream, "z_int")
    z_lev_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[tgt_def_key]["z_lev"] # ASSUME: 

    #---------------------------------------------------------------------------
    # All mixing ratios are moist mixing ratios. 
    # To calculate LWP, IWP later, we want mass of cloud liquid/ice
    # water in each cell. For this, we need mass of moist air first.
    # To calculate volume mixing ratios later, we want number of moles of
    # dry air in each cell first. These also need to be vertically remapped
    # horizontally coarsened.
    #---------------------------------------------------------------------------
    if l_rank == MPI_ROOT:
        current_time = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Calculating moist air mass and number of dry air moles...".format(current_time)
        print(msg, flush = True)

    l_p_src: NP_ARRAY[NP_REAL] # Pressure [Pa], [nt, l_nx, ny, nlay]
    l_T_src: NP_ARRAY[NP_REAL] # Temperature [K], [nt, l_nx, ny, nlay]
    l_qv_src: NP_ARRAY[NP_REAL] # Specific Humidity [kg kg^{-1}], [nt, l_nx, ny, nlay]
    [_, l_p_src] = extract_dp_scream_field(xr_dp_scream, "p_mid") 
    [_, l_T_src] = extract_dp_scream_field(xr_dp_scream, "T_mid")
    [_, l_qv_src] = extract_dp_scream_field(xr_dp_scream, "qv")
            
    l_dz_src: NP_ARRAY[NP_REAL] = l_z_lev_src[...,1:] - l_z_lev_src[...,:-1] # Layer thickness [m], [nt, l_nx, ny, nlay]
    l_V_src: NP_ARRAY[NP_REAL] = l_dx_src * l_dy_src * l_dz_src # Cell volume [m^{3}], [nt, l_nx, ny, nlay]

    l_M_src: NP_ARRAY[NP_REAL] = (l_p_src * l_V_src) / (R_d * l_T_src * (1. + l_qv_src * ((R_v / R_d) - 1.))) # Moist air mass [kg], [nt, l_nx, ny, nlay]
    l_nd_src: NP_ARRAY[NP_REAL] = l_M_src * (1. - l_qv_src) / mu_d # Moles of dry air [mol], [nt, l_nx, ny, nz]

    l_M_vremap: NP_ARRAY[NP_REAL] = remap_layer_mass(z_lev_tgt, l_z_lev_src, l_M_src)
    l_nd_vremap: NP_ARRAY[NP_REAL] = remap_layer_mass(z_lev_tgt, l_z_lev_src, l_nd_src)

    l_M_tgts: dict = {}
    l_nd_tgts: dict = {}

    l_M_coarsener: list[RegularGridInterpolator] = \
            [[RegularGridInterpolator((l_x_src, l_y_src), l_M_vremap[tt,:,:,kk], method = interp_method)
                for kk in range(0, l_nz_src)] for tt in range(0, l_nt)]
    l_nd_coarsener: list[RegularGridInterpolator] = \
            [[RegularGridInterpolator((l_x_src, l_y_src), l_nd_vremap[tt,:,:,kk], method = interp_method)
                for kk in range(0, l_nz_src)] for tt in range(0, l_nt)]
    for coarse_str in l_grids_tgt.keys():
        #-----------------------------------------------------------------------
        # Get target grid information
        #-----------------------------------------------------------------------
        l_nx_tgt: NP_INT = l_grids_tgt[coarse_str]["nx"]
        l_ny_tgt: NP_INT = l_grids_tgt[coarse_str]["ny"]
        l_nz_tgt: NP_INT = l_grids_tgt[coarse_str]["nlay"]

        l_x_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["x"]
        l_y_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["y"]

        l_XX_tgt, l_YY_tgt = \
            np.meshgrid(l_x_tgt, l_y_tgt, indexing = "ij")
        l_pts_tgt: NP_ARRAY[NP_REAL] = \
            np.stack([l_XX_tgt.flatten(), l_YY_tgt.flatten()], 
                axis = 1)

        #-----------------------------------------------------------------------
        # Horizontally coarsen the moist air mass and number of dry air moles
        #-----------------------------------------------------------------------
        l_M_tgt: NP_ARRAY[NP_REAL] = np.empty([l_nt, l_nx_tgt, l_ny_tgt, l_nz_tgt], dtype = NP_REAL)
        for tt in range(0, l_nt):
            for kk in range(0, l_nz_src):
                l_M_tgt[tt,:,:,kk] = l_M_coarsener[tt][kk](l_pts_tgt).reshape(l_nx_tgt, l_ny_tgt)

        l_M_tgts[coarse_str] = l_M_tgt

        l_nd_tgt: NP_ARRAY[NP_REAL] = np.empty([l_nt, l_nx_tgt, l_ny_tgt, l_nz_tgt], dtype = NP_REAL)
        for tt in range(0, l_nt):
            for kk in range(0, l_nz_src):
                l_nd_tgt[tt,:,:,kk] = l_nd_coarsener[tt][kk](l_pts_tgt).reshape(l_nx_tgt, l_ny_tgt)

        l_nd_tgts[coarse_str] = l_nd_tgt

    #---------------------------------------------------------------------------
    # Loop through RTE-RRTMGP-CPP+RT fields
    #---------------------------------------------------------------------------
    for rad_tran_key in ["lwp"]:#rte_3dfield_keys:
        if l_rank == MPI_ROOT:
            current_time = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Extracting DP-SCREAM field(s) for {}...".format(current_time, rad_tran_key)
            print(msg, flush = True)
        #-----------------------------------------------------------------------
        # Extract relevant fields from DP-SCREAM file
        #-----------------------------------------------------------------------
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

        #-----------------------------------------------------------------------
        # Obtain target vertical grid
        # ASSUME VERTICAL GRID IS SAME FOR SRC, TGT, AND ALL COARSENINGS
        #-----------------------------------------------------------------------
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
            l_nz_tgt = l_grids_tgt[tgt_def_key]["nz"]
            z_tgt = l_grids_tgt[tgt_def_key]["z"]
        elif not tgt_data_at_layers and tgt_data_at_levels:
            l_nz_tgt = l_grids_tgt[tgt_def_key]["nlev"]
            z_tgt = l_grids_tgt[tgt_def_key]["z_lev"]
        elif tgt_data_at_layers and not tgt_data_at_levels:
            l_nz_tgt = l_grids_tgt[tgt_def_key]["nlay"]
            z_tgt = l_grids_tgt[tgt_def_key]["z_lay"]

        #-----------------------------------------------------------------------
        # Remap field values vertically
        #-----------------------------------------------------------------------
        if l_rank == MPI_ROOT:
            current_time = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Vertically remapping {}...".format(current_time, rad_tran_key)
            print(msg, flush = True)

        l_nx_src: NP_INT
        l_ny_src: NP_INT
        l_nz_src: NP_INT
        [_, l_nx_src, l_ny_src, l_nz_src] = NP_INT(l_field_src.shape)
        l_field_vremap: NP_ARRAY[NP_REAL] = np.empty([l_nt, l_nx_src, l_ny_src, l_nz_src], dtype = NP_REAL)
        if rad_tran_key in ["p", "t", "rh", "rel", "dei"]:
            # TO-DO: Consider vectorizing with SciPy instead of nested for loops
            # https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#tutorial-interpolate-1dsection
            for tt in range(0, l_nt):
                for ii in range(0, l_nx_src):
                    for jj in range(0, l_ny_src):
                        l_field_vremap[tt, ii, jj, :] = np.interp(z_tgt, l_z_src[tt, ii, jj, :], l_field_src[tt, ii, jj, :])
        elif rad_tran_key in ["lwp", "iwp", "vmr_ch4", "vmr_co", "vmr_co2",
            "vmr_h2o", "vmr_n2", "vmr_n2o", "vmr_o2", "vmr_o3"]:
            # This is currently mass or number of moles. Change after coarsening.
            l_field_vremap = remap_layer_mass(z_lev_tgt, l_z_lev_src, l_field_src) 
            
        #-----------------------------------------------------------------------
        # Store vertically-remapped field values
        #-----------------------------------------------------------------------
        l_fields_vremap[rad_tran_key] = l_field_vremap

        #-----------------------------------------------------------------------
        # Coarsen field values horizontally
        #-----------------------------------------------------------------------
        l_horz_coarsener: list[RegularGridInterpolator] = \
            [[RegularGridInterpolator((l_x_src, l_y_src), l_field_vremap[tt,:,:,kk], method = interp_method)
                for kk in range(0, l_nz_src)] for tt in range(0, l_nt)]
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

            l_counts_tgt: NP_ARRAY[NP_INT] = l_nt * l_nz_tgt * l_ny_tgt * l_counts_x
            l_displs_tgt: NP_ARRAY[NP_INT] = l_nt * l_nz_tgt * l_ny_tgt * l_displs_x

            #-------------------------------------------------------------------
            # Horizontally coarsen the field
            #-------------------------------------------------------------------
            l_field_tgt: NP_ARRAY[NP_REAL] = \
                np.empty([l_nt, l_nx_tgt, l_ny_tgt, l_nz_tgt], dtype = NP_REAL)
            for tt in range(0, l_nt):
                for kk in range(0, l_nz_src):
                    l_field_tgt[tt,:,:,kk] = l_horz_coarsener[tt][kk](l_pts_tgt).reshape(l_nx_tgt, l_ny_tgt)

            #-------------------------------------------------------------------
            # Final calculations on coarsened field
            #-------------------------------------------------------------------
            if rad_tran_key in ["lwp", "iwp"]:
                # Needs to be divided by moist air mass
                l_field_tgt = l_field_tgt / l_M_tgts[coarse_str]
            elif rad_tran_key in ["vmr_ch4", "vmr_co", "vmr_co2", "vmr_h2o", 
                "vmr_n2", "vmr_n2o", "vmr_o2", "vmr_o3"]:
                # Needs to be divided by number of dry air moles
                l_field_tgt = l_field_tgt / l_nd_tgts[coarse_str]
            #-------------------------------------------------------------------
            # Store horizontally-coarsened field values
            #-------------------------------------------------------------------
            l_fields_tgt[coarse_str][rad_tran_key] = l_field_tgt

            #-------------------------------------------------------------------
            # Gatherv the whole field onto MPI_ROOT
            #-------------------------------------------------------------------
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
                    fields_tgt[coarse_str][rad_tran_key + "_lay"] = field_tgt[...,1::2]
                    fields_tgt[coarse_str][rad_tran_key + "_lev"] = field_tgt[...,0::2]
                else:
                    fields_tgt[coarse_str][rad_tran_key] = field_tgt

    return fields_tgt

def extract_dp_scream_field(xr_dp_scream: XR_DATASET, dp_scream_key: str) -> tuple[NP_ARRAY[NP_REAL]]:
    #---------------------------------------------------------------------------
    # Get field information - at levels (int), layers (mid), or both
    #---------------------------------------------------------------------------
    src_data_at_layers: bool
    src_data_at_levels: bool
            
    # If dp_scream_key in dp_scream_file, then src_data only at layers
    # Else, then dp_scream_key needs to be parsed further
    if dp_scream_key in xr_dp_scream.keys():
        src_data_at_layers = True
        src_data_at_levels = False

        z_lay_src: NP_ARRAY[NP_REAL] = xr_dp_scream["z_mid"].to_numpy().astype(NP_REAL) # Layer midpoints [m]; (nt, nx, ny, n_lay_z)
        field_lay_src: NP_ARRAY[NP_REAL] = xr_dp_scream[dp_scream_key].to_numpy().astype(NP_REAL) # Field at layer midpoints; (nt, nx, ny, n_lay_z)
    else:
        # Default form of keys
        dp_scream_key_mid: str = dp_scream_key + "_mid"
        dp_scream_key_int: str = dp_scream_key + "_int"

        assert((dp_scream_key_mid in xr_dp_scream.keys()) or 
            (dp_scream_key_int in xr_dp_scream.keys()))

        # Field values at layers available
        if dp_scream_key_mid in xr_dp_scream.keys():
            src_data_at_layers = True

            z_lay_src: NP_ARRAY[NP_REAL] = xr_dp_scream["z_mid"].to_numpy().astype(NP_REAL) # Layer midpoints [m]; (nt, nx, ny, n_lay_z)
            field_lay_src: NP_ARRAY[NP_REAL] = xr_dp_scream[dp_scream_key_mid].to_numpy().astype(NP_REAL) # Field at layer midpoints; (nt, nx, ny, n_lay_z)
        else:
            src_data_at_layers = False

        # Field values at levels available
        if dp_scream_key_int in xr_dp_scream.keys():
            src_data_at_levels = True

            z_lev_src: NP_ARRAY[NP_REAL] = xr_dp_scream["z_int"].to_numpy().astype(NP_REAL) # Layer interfaces [m]; (nt, nx, ny, n_lev_z)
            field_lev_src: NP_ARRAY[NP_REAL] = xr_dp_scream[dp_scream_key_int].to_numpy().astype(NP_REAL) # Field at layer interfaces; (nt, nx, ny, n_lev_z)
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

    return [z_src[...,::-1], field_src[...,::-1]] # Flip to increasing z

def remap_layer_mass(z_lev_tgt: NP_ARRAY[NP_REAL], z_lev_src: NP_ARRAY[NP_REAL],
    mass_src: NP_ARRAY[NP_REAL]) -> NP_ARRAY[NP_REAL]:

    #---------------------------------------------------------------------------
    # Get dimensions
    #---------------------------------------------------------------------------
    nt: NP_INT
    nx: NP_INT
    ny: NP_INT
    nlay: NP_INT
    [nt, nx, ny, nlay] = NP_INT(mass_src.shape)

    #---------------------------------------------------------------------------
    # Compute source/target layer lower/upper bounds, get lay thicknesses
    #---------------------------------------------------------------------------
    z_lo_src: NP_ARRAY[NP_REAL] = z_lev_src[...,:-1] # Lower-bound of source intervals, [nt, nx, ny, nlay]
    z_hi_src: NP_ARRAY[NP_REAL] = z_lev_src[...,1:] # Upper-bound of source intervals, [nt, nx, ny, nlay]

    z_lo_tgt: NP_ARRAY[NP_REAL] = z_lev_tgt[:-1] # Lower-bound of target intervals, [nlay]
    z_hi_tgt: NP_ARRAY[NP_REAL] = z_lev_tgt[1:] # Upper-bound of target intervals,  [nlay]

    dz_src: NP_ARRAY[NP_REAL] = z_hi_src - z_lo_src # [nt, nx, ny, nlay]

    #---------------------------------------------------------------------------
    # Expand source arrays for pairwise source-target overlap
    # This allows us to construct matrices encoding overlap information
    #---------------------------------------------------------------------------
    z_lo_src_2d: NP_ARRAY[NP_REAL] = z_lo_src[...,None] # [nt, nx, ny, nlay, 1]
    z_hi_src_2d: NP_ARRAY[NP_REAL] = z_hi_src[...,None] # [nt, nx, ny, nlay, 1]
    dz_src_2d: NP_ARRAY[NP_REAL] = np.tile(dz_src[...,None], (1,nlay)) # [nt, nx, ny, nlay, nlay]

    #---------------------------------------------------------------------------
    # Compute overlap thickness between every source layer and target layer
    # overlap = max(0, min(src_hi, tgt_hi) - max(src_lo, tgt_lo))
    # lo_ovlp: The (n,m) entry is the lower bound of where the mth interval of the 
    # target grid overlaps the nth interval of the source grid, and similarly
    # for hi_ovlp. Where hi_ovlp - lo_ovlp is negative, there is no actual overlap
    #---------------------------------------------------------------------------
    lo_ovlp: NP_ARRAY[NP_REAL] = np.maximum(z_lo_src_2d, z_lo_tgt) # [nt, nx, ny, nlay, nlay]
    hi_ovlp: NP_ARRAY[NP_REAL] = np.minimum(z_hi_src_2d, z_hi_tgt) # [nt, nx, ny, nlay, nlay]

    ovlp: NP_ARRAY[NP_REAL] = np.maximum(NP_REAL(0.), hi_ovlp - lo_ovlp) # [nt, nx, ny, nlay, nlay]

    #---------------------------------------------------------------------------
    # Compute fraction of each source layer contributing to each target layer
    #---------------------------------------------------------------------------
    frac: NP_ARRAY[NP_REAL] = np.zeros([nt, nx, ny, nlay, nlay], dtype = NP_REAL)

    mask_nonzero: NP_ARRAY[NP_BOOL] = (dz_src_2d > NP_REAL(0.))
    frac[mask_nonzero] = ovlp[mask_nonzero] / dz_src_2d[mask_nonzero]

    #---------------------------------------------------------------------------
    # Distribute source mass into target layers
    #---------------------------------------------------------------------------
    mass_contrib: NP_ARRAY[NP_REAL] = mass_src[...,None] * frac
    mass_tgt: NP_ARRAY[NP_REAL] = np.sum(mass_contrib, axis = -2) # [nt, nx, ny, nlay]

    return mass_tgt