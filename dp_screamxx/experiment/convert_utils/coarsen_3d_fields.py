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

def coarsen_3d_fields(xr_dp_scream: XR_DATASET, g_grid_vremap: dict, l_grid_vremap: dict, 
    g_grids_tgt: dict, l_grids_tgt: dict, comm: MPI_COMM, interp_method: str = "nearest") -> dict:
    #---------------------------------------------------------------------------
    # Get MPI communicator information
    #---------------------------------------------------------------------------
    l_rank: NP_INT = NP_INT(comm.Get_rank())

    #---------------------------------------------------------------------------
    # Common variables throuhgout script
    #---------------------------------------------------------------------------
    l_nt: NP_INT = NP_INT(xr_dp_scream["time"].size)
    
    #---------------------------------------------------------------------------
    # Extract vremap grid information
    #---------------------------------------------------------------------------
    l_nx_vremap: NP_INT = l_grid_vremap["nx"]
    l_x_vremap: NP_ARRAY[NP_REAL] = l_grid_vremap["x"]

    l_ny_vremap: NP_INT = l_grid_vremap["ny"]
    l_y_vremap: NP_ARRAY[NP_REAL] = l_grid_vremap["y"]

    # The number of vertical layers/levels matches between source grid and DP-SCREAM grid
    l_nz_vremap: NP_INT = l_grid_vremap["nz"]

    #---------------------------------------------------------------------------
    # Extract DP-SCREAM grid information - that differs from source grid
    #---------------------------------------------------------------------------
    l_z_lev_src: NP_ARRAY[NP_REAL]  # Level height [m], [nt, l_nx, ny, nlev]
    [_, l_z_lev_src] = extract_dp_scream_field(xr_dp_scream, "z_int")

    #---------------------------------------------------------------------------
    # Extract vertical target grid infromation - ASSUME: Common across all target grids
    #---------------------------------------------------------------------------
    key: NP_INT = list(l_grids_tgt.keys())[0]
    z_lev_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[key]["z_lev"] # ASSUME: Same vertical grid target

    #---------------------------------------------------------------------------
    # Derive source/target grid information for vertical remapping and horizontal coarsening
    #---------------------------------------------------------------------------
    dz_tgt: NP_REAL = z_lev_tgt[1] - z_lev_tgt[0]

    #---------------------------------------------------------------------------
    # Calculate moist air mass and number of dry air moles.
    #---------------------------------------------------------------------------
    # All cloud water mixing ratios are moist mixing ratios.
    # All volume mixing ratios are dry molar mixing ratios.
    # To quanities involving mass/moles, we conservatively remap mass/moles.
    # To get mass/moles, we need to extract them from ratios
    # We need these for lots of quanities, so we calculate them once.
    if l_rank == MPI_ROOT:
        current_time: str = datetime.now().strftime("%H:%M:%S")
        msg: str = "[{}]: Calculating moist air mass and number of dry air moles...".format(current_time)
        print(msg, flush = True)

    l_M_src: NP_ARRAY[NP_REAL]
    l_M_tgts: dict
    l_nd_src: NP_ARRAY[NP_REAL]
    l_nd_tgts: dict

    [l_M_src, l_M_tgts, l_nd_src, l_nd_tgts] = calc_air_mass_moles(xr_dp_scream,
        l_grid_vremap, l_grids_tgt, interp_method)

    #---------------------------------------------------------------------------
    # Set up dict for holding horizontally coarsened (i.e., tgt) field values
    #---------------------------------------------------------------------------
    fields_tgt: Optional[dict] = None
    if l_rank == MPI_ROOT:
        fields_tgt = {}
        for coarse_str in l_grids_tgt.keys():
            fields_tgt[coarse_str] = {}

    #---------------------------------------------------------------------------
    # Loop through RTE-RRTMGP-CPP+RT fields
    #---------------------------------------------------------------------------
    for rad_tran_key in rte_3d_field_keys:
        #-----------------------------------------------------------------------
        # Extract relevant fields from DP-SCREAM for RTE-RRTMGP-CPP+RT field
        #-----------------------------------------------------------------------
        if l_rank == MPI_ROOT:
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Extracting DP-SCREAM field(s) for {}...".format(current_time, rad_tran_key)
            print(msg, flush = True)
        
        # Most fields can be extracted directly from the file and vertically
        # remapped, possibly with a key change and a minor modification
        l_z_src: NP_ARRAY[NP_REAL] # [nt, l_nx, ny, nz]
        l_field_src: list[NP_ARRAY[NP_REAL]] # [nt, l_nx, ny, nz]

        [l_z_src, l_field_src] = get_l_field_src(xr_dp_scream, rad_tran_key, l_M_src, l_nd_src)

        #-----------------------------------------------------------------------
        # Obtain vertical remap grid
        # ASSUME VERTICAL GRID IS SAME FOR ALL COARSENINGS
        #-----------------------------------------------------------------------
        l_nz_vremap: NP_INT
        l_z_vremap: NP_ARRAY[NP_REAL]
        [l_nz_vremap, l_z_vremap] = get_vertical_grid(l_grid_vremap, rad_tran_key)

        #-----------------------------------------------------------------------
        # Remap field values vertically
        #-----------------------------------------------------------------------
        if l_rank == MPI_ROOT:
            current_time: str = datetime.now().strftime("%H:%M:%S")
            msg: str = "[{}]: Vertically remapping {}...".format(current_time, rad_tran_key)
            print(msg, flush = True)

        l_field_vremap: list[NP_ARRAY[NP_REAL]]

        if rad_tran_key in ["p", "t", "rel", "dei"]:
            # TO-DO: Consider vectorizing with SciPy instead of nested for loops
            # https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#tutorial-interpolate-1dsection
            l_field_vremap = [np.empty([l_nt, l_nx_vremap, l_ny_vremap, l_nz_vremap], dtype = NP_REAL)]
            for tt in range(0, l_nt):
                for ii in range(0, l_nx_vremap):
                    for jj in range(0, l_ny_vremap):
                        l_field_vremap[0][tt, ii, jj, :] = np.interp(l_z_vremap, l_z_src[tt, ii, jj, :], l_field_src[0][tt, ii, jj, :])
        elif rad_tran_key in (["lwp", "iwp"]) or ("vmr_" in rad_tran_key):
            # This is currently mass or number of moles. Change after coarsening.
            l_field_vremap = [remap_layer_mass(l_z_vremap, l_z_lev_src, l_field_src[0])]
        elif rad_tran_key == "rh":
            # Remap water vapor mass and saturation water vapor mass. Change after coarsening
            l_field_vremap = [remap_layer_mass(l_z_vremap, l_z_lev_src, l_field_src[0]),
                remap_layer_mass(l_z_vremap, l_z_lev_src, l_field_src[1])]

        nfields: NP_INT = NP_INT(len(l_field_vremap))

        # Limit vremapped values - ASSUME: Piecewise linear interpolation,
        # so vremapped values stay between source values
        field_min: Optional[NP_REAL] = None
        field_max: Optional[NP_REAL] = None
        if rad_tran_key == "rel": # Between 2.5 μm and 21.5 μm
            field_min = NP_REAL(2.5)
            field_max = NP_REAL(21.5)
        elif rad_tran_key == "dei": # Between 10. μm and 180. μm
            field_min = NP_REAL(10.)
            field_max = NP_REAL(180.)
        elif rad_tran_key == "t":
            field_min = NP_REAL(100.0) # Lowest temperature in mesosphere https://scied.ucar.edu/learning-zone/atmosphere/mesosphere [K]
            field_max = NP_REAL(329.817) # Hottest observed temperature https://www.ncei.noaa.gov/news/earths-hottest-temperature [K]
        elif rad_tran_key == "p":
            field_min = NP_REAL(0.1) # Atmospheric pressure near top of mesosphere. [Pa]
            field_max = NP_REAL(108480.0) # Highest recorded atmospheric pressure https://web.archive.org/web/20121017130834/http://wmo.asu.edu/highest-sea-lvl-air-pressure-above-700m [Pa]
        elif (rad_tran_key in ["lwp", "iwp"]) or ("vmr_" in rad_tran_key):
            field_min = NP_REAL(0.0)

        for ll in range(0, nfields):
            if field_max is not None:
                l_field_vremap[ll][l_field_vremap[ll] > field_max] = field_max
            if field_min is not None:
                l_field_vremap[ll][l_field_vremap[ll] < field_min] = field_min

        #-----------------------------------------------------------------------
        # Coarsen field values horizontally
        #-----------------------------------------------------------------------
        l_horz_coarsener: list[RegularGridInterpolator] = \
            [[[RegularGridInterpolator((l_x_vremap, l_y_vremap), l_field_vremap[ll][tt,:,:,kk], method = interp_method)
                for kk in range(0, l_nz_vremap)] for tt in range(0, l_nt)] for ll in range(0, nfields)]
        for coarse_str in l_grids_tgt.keys():
            if l_rank == MPI_ROOT:
                current_time: str = datetime.now().strftime("%H:%M:%S")
                msg: str = "[{}]: Coarsening {} to lr_{}...".format(current_time, rad_tran_key, coarse_str)
                print(msg, flush = True)

            #-------------------------------------------------------------------
            # Get target grid information
            #-------------------------------------------------------------------
            l_nx_tgt: NP_INT = l_grids_tgt[coarse_str]["nx"]
            l_ny_tgt: NP_INT = l_grids_tgt[coarse_str]["ny"]

            l_x_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["x"]
            l_y_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["y"]

            l_dy_tgt: NP_REAL = l_y_tgt[1] - l_y_tgt[0] # [m]
            l_dx_tgt: NP_REAL = l_dy_tgt # ASSUME: Same spacing in x- and y-, but we are guaranteed to have y_tgt

            l_XX_tgt: NP_ARRAY[NP_REAL]
            l_YY_tgt: NP_ARRAY[NP_REAL]
            l_XX_tgt, l_YY_tgt = np.meshgrid(l_x_tgt, l_y_tgt, indexing = "ij")
            l_pts_tgt: NP_ARRAY[NP_REAL] = np.stack([l_XX_tgt.flatten(), l_YY_tgt.flatten()], axis = 1)

            [l_nz_tgt, _] = get_vertical_grid(l_grids_tgt[coarse_str], rad_tran_key)

            #-------------------------------------------------------------------
            # Calculate communication parameters
            #-------------------------------------------------------------------
            l_counts_x: NP_ARRAY[NP_INT] = l_grids_tgt[coarse_str]["l_counts_x"]
            l_displs_x: NP_ARRAY[NP_INT] = l_grids_tgt[coarse_str]["l_displs_x"]

            l_counts_tgt: NP_ARRAY[NP_INT] = l_nt * l_nz_tgt * l_ny_tgt * l_counts_x
            l_displs_tgt: NP_ARRAY[NP_INT] = l_nt * l_nz_tgt * l_ny_tgt * l_displs_x

            #-------------------------------------------------------------------
            # Horizontally coarsen the field
            #-------------------------------------------------------------------
            l_field_tgt_pre: list[NP_ARRAY[NP_REAL]]
            l_field_tgt_pre: NP_ARRAY[NP_REAL] = \
                [np.empty([l_nt, l_nx_tgt, l_ny_tgt, l_nz_tgt], dtype = NP_REAL) for _ in range(0, nfields)]
            for ll in range(0, nfields):
                for tt in range(0, l_nt):
                    for kk in range(0, l_nz_vremap):
                        l_field_tgt_pre[ll][tt,:,:,kk] = l_horz_coarsener[ll][tt][kk](l_pts_tgt).reshape(l_nx_tgt, l_ny_tgt)

            #-------------------------------------------------------------------
            # Final calculations on coarsened field
            #-------------------------------------------------------------------
            l_field_tgt: NP_ARRAY[NP_REAL]
            if rad_tran_key in ["p", "t", "rel", "dei"]:
                # Simply extract the field
                l_field_tgt = l_field_tgt_pre[0]
            elif rad_tran_key in ["lwp", "iwp"]:
                # Have cloud liquid/ice water mass, get density and integrate vertical for water path
                # which is equivalent to dz * (mass / (dx * dy * dz))
                l_field_tgt = l_field_tgt_pre[0] / (l_dx_tgt * l_dy_tgt)
            elif ("vmr_" in rad_tran_key):
                # Needs to be divided by number of dry air moles
                l_field_tgt = l_field_tgt_pre[0] / l_nd_tgts[coarse_str]
            elif rad_tran_key == "rh":
                # Needs to be derived from the two remapped fields
                eps: NP_REAL = R_d / R_v
                l_qv_tgt: NP_ARRAY[NP_REAL] = l_field_tgt_pre[0] / l_M_tgts[coarse_str]
                l_qvs_tgt: NP_ARRAY[NP_REAL] = l_field_tgt_pre[1] / l_M_tgts[coarse_str]

                term0: NP_ARRAY[NP_REAL] = (1. - eps) * l_qv_tgt * l_qvs_tgt
                l_field_tgt: NP_ARRAY[NP_REAL] = (term0 + eps * l_qv_tgt) / (term0 + eps * l_qvs_tgt)

            #-------------------------------------------------------------------
            # Gatherv the whole field onto MPI_ROOT
            #-------------------------------------------------------------------
            # Reshape l_field_tgt for easier Gatherv
            l_field_tgt = np.ascontiguousarray(np.transpose(l_field_tgt, axes = [1, 0, 2, 3])) # [nx, nt, ny, nz]
            field_tgt: Optional[NP_ARRAY[NP_REAL]] = None
            if l_rank == MPI_ROOT:
                nt_tgt: NP_INT = l_nt
                nx_tgt: NP_INT = g_grids_tgt[coarse_str]["nx"]
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
                    fields_tgt[coarse_str][rad_tran_key + "_lay"] = field_tgt[:,1::2,...]
                    fields_tgt[coarse_str][rad_tran_key + "_lev"] = field_tgt[:,0::2,...]
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
        z_src[:,0::2,...] = z_lev_src
        z_src[:,1::2,...] = z_lay_src

        field_src: NP_ARRAY[NP_REAL] = np.empty([g_nt, g_nz, g_ny, l_nx], dtype = NP_REAL)
        field_src[:,0::2,...] = field_lev_src
        field_src[:,1::2,...] = field_lay_src
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

def get_l_field_src(xr_dp_scream: XR_DATASET, rad_tran_key: str,
    l_M_src: NP_ARRAY[NP_REAL], l_nd_src: NP_ARRAY[NP_REAL]) -> list[NP_ARRAY[NP_REAL]]:

    l_z_src: NP_ARRAY[NP_REAL] # [nt, l_nx, ny, nz]
    l_field_src: list[NP_ARRAY[NP_REAL]] = [] # [nt, l_nx, ny, nz]

    if rad_tran_key in (["p", "t", "rel", "dei", "lwp", "iwp"]) or ("vmr_" in rad_tran_key):
        if rad_tran_key == "p":
            dp_scream_key = rad_tran_key
        elif rad_tran_key == "t":
            dp_scream_key = rad_tran_key.upper()
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

        [l_z_src, l_field_src_pre] = extract_dp_scream_field(xr_dp_scream, dp_scream_key)
        l_field_src += [l_field_src_pre]

        # DP-SCREAM uses ice-water effective radius
        # RT-RRTMGP-CPP+RT use ice-water effective diameter
        if rad_tran_key == "dei":
            l_field_src[0] *= 2.
        # Mixing ratios to masses/moles for proportional split
        elif rad_tran_key in ["lwp", "iwp"]:
            l_field_src[0] = l_field_src[0] * l_M_src # Liquid/Ice Cloud-Water Mass [kg], [nt, l_nx, ny, nz]
        elif "vmr_" in rad_tran_key:
            l_field_src[0] = l_field_src[0] * l_nd_src # Moles of gas [mole], [nt, l_nx, ny, nz]
    # For relative humidity, we will remap both the water vapor mass and the
    # and the saturation water vapor mass ratio as calculated by DP-SCREAM
    elif rad_tran_key in ["rh"]:
        if rad_tran_key == "rh":
            dp_scream_key = "RelativeHumidity"

        l_z_src: NP_ARRAY[NP_REAL] # [nt, l_nx, ny, nz]
        l_rh_src: NP_ARRAY[NP_REAL] # [nt, l_nx, ny, nz]
        l_qv_src: NP_ARRAY[NP_REAL] # [nt, l_nx, ny, nz]
        [l_z_src, l_rh_src] = extract_dp_scream_field(xr_dp_scream, "RelativeHumidity")
        [_, l_qv_src] = extract_dp_scream_field(xr_dp_scream, "qv")

        eps: NP_REAL = R_d / R_v
        l_qvs_src: NP_ARRAY[NP_REAL] = (eps * l_qv_src) /  (l_qv_src * (l_rh_src - 1.) * (1. - eps) + eps * l_rh_src)

        l_mv_src: NP_ARRAY[NP_REAL] = l_qv_src * l_M_src # Mass of water vapor [kg], [nt, l_nx, ny, nz]
        l_mvs_src: NP_ARRAY[NP_REAL] = l_qvs_src * l_M_src # Mass of saturation water vapor [kg], [nt, l_nx, ny, nz]

        l_field_src: list[NP_ARRAY[NP_REAL]] = [l_mv_src, l_mvs_src]

    return [l_z_src, l_field_src]

def get_vertical_grid(l_grid: dict, rad_tran_key: str) -> list[NP_INT, NP_ARRAY[NP_REAL]]:
    l_nz: NP_INT
    l_z: NP_ARRAY[NP_REAL]
    
    if rad_tran_key in ["p", "t"]:
        l_nz = l_grid["nz"]
        l_z = l_grid["z"]
    elif rad_tran_key in (["rh", "lwp", "iwp"]) or ("vmr_" in rad_tran_key):
        # Remapping masses requires levels and returns values at layers
        l_nz = l_grid["nlay"]
        l_z = l_grid["z_lev"]
    else:
        l_nz = l_grid["nlay"]
        l_z = l_grid["z_lay"]

    return [l_nz, l_z]

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

def calc_air_mass_moles(xr_dp_scream: XR_DATASET, l_grid_vremap: dict, l_grids_tgt: dict,
    interp_method: str = "nearest"):
    # Returns three quantities needed later:
    # 1) Moist air mass on source grid [kg]
    # 2) Moles of dry air on source grid [mole]
    # 3) Dry air moles on target grids [mole]

    #---------------------------------------------------------------------------
    # Common variables throuhgout function
    #---------------------------------------------------------------------------
    tgt_def_key: str = list(l_grids_tgt.keys())[0] # Default key to use for l_grids_tgt
    # ASSUME: All target grids have same vertical grid
    l_nt: NP_INT = NP_INT(xr_dp_scream["time"].size)

    #---------------------------------------------------------------------------
    # Extract source grid information - as derived from DP-SCREAM grid
    #---------------------------------------------------------------------------
    l_x_vremap: NP_ARRAY[NP_REAL] = l_grid_vremap["x"]
    l_y_vremap: NP_ARRAY[NP_REAL] = l_grid_vremap["y"]

    # The number of vertical layers/levels matches between source grid and DP-SCREAM grid
    l_nlay_src: NP_INT = l_grid_vremap["nlay"]

    #---------------------------------------------------------------------------
    # Extract DP-SCREAM grid information - that differs from source grid
    #---------------------------------------------------------------------------
    l_z_lev_src: NP_ARRAY[NP_REAL]  # Level height [m], [nt, l_nx, ny, nlev]
    [_, l_z_lev_src] = extract_dp_scream_field(xr_dp_scream, "z_int")

    #---------------------------------------------------------------------------
    # Extract vertical target grid infromation - ASSUME: Common across all target grids
    #---------------------------------------------------------------------------
    z_lev_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[tgt_def_key]["z_lev"] # ASSUME: Same vertical grid target

    #---------------------------------------------------------------------------
    # Derive source/target grid information for vertical remapping and horizontal coarsening
    #---------------------------------------------------------------------------
    # ASSUME: Constant spacing in x- and y-.
    l_dx_src: NP_REAL = l_x_vremap[1] - l_x_vremap[0]
    l_dy_src: NP_REAL = l_y_vremap[1] - l_y_vremap[0]

    #---------------------------------------------------------------------------
    # Get the soruce grid values and vertically remap moles of dry air
    #---------------------------------------------------------------------------
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
        [[RegularGridInterpolator((l_x_vremap, l_y_vremap), l_M_vremap[tt,:,:,kk], method = interp_method)
            for kk in range(0, l_nlay_src)] for tt in range(0, l_nt)]

    l_nd_coarsener: list[RegularGridInterpolator] = \
        [[RegularGridInterpolator((l_x_vremap, l_y_vremap), l_nd_vremap[tt,:,:,kk], method = interp_method)
            for kk in range(0, l_nlay_src)] for tt in range(0, l_nt)]
    for coarse_str in l_grids_tgt.keys():
        #-----------------------------------------------------------------------
        # Get target grid information
        #-----------------------------------------------------------------------
        l_nx_tgt: NP_INT = l_grids_tgt[coarse_str]["nx"]
        l_ny_tgt: NP_INT = l_grids_tgt[coarse_str]["ny"]
        l_nz_tgt: NP_INT = l_grids_tgt[coarse_str]["nlay"]

        l_x_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["x"]
        l_y_tgt: NP_ARRAY[NP_REAL] = l_grids_tgt[coarse_str]["y"]

        l_XX_tgt, l_YY_tgt = np.meshgrid(l_x_tgt, l_y_tgt, indexing = "ij")
        l_pts_tgt: NP_ARRAY[NP_REAL] = np.stack([l_XX_tgt.flatten(), l_YY_tgt.flatten()], axis = 1)

        #-----------------------------------------------------------------------
        # Horizontally coarsen moist air mass
        #-----------------------------------------------------------------------
        l_M_tgt: NP_ARRAY[NP_REAL] = np.empty([l_nt, l_nx_tgt, l_ny_tgt, l_nz_tgt], dtype = NP_REAL)
        for tt in range(0, l_nt):
            for kk in range(0, l_nlay_src):
                l_M_tgt[tt,:,:,kk] = l_M_coarsener[tt][kk](l_pts_tgt).reshape(l_nx_tgt, l_ny_tgt)

        l_M_tgts[coarse_str] = l_M_tgt

        #-----------------------------------------------------------------------
        # Horizontally coarsen dry air moles
        #-----------------------------------------------------------------------
        l_nd_tgt: NP_ARRAY[NP_REAL] = np.empty([l_nt, l_nx_tgt, l_ny_tgt, l_nz_tgt], dtype = NP_REAL)
        for tt in range(0, l_nt):
            for kk in range(0, l_nlay_src):
                l_nd_tgt[tt,:,:,kk] = l_nd_coarsener[tt][kk](l_pts_tgt).reshape(l_nx_tgt, l_ny_tgt)

        l_nd_tgts[coarse_str] = l_nd_tgt

    return [l_M_src, l_M_tgts, l_nd_src, l_nd_tgts]