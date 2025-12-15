"""
Converts output from DP-SCREAM into input for RTE-RRTMGP-CPP.

See the following reference for more information:
M. A. Veerman. Simulating sunshine on cloudy days (2023). doi: 10.18174/634325.
"""

# Standard Library Imports
import argparse
import ast
import os
import re
import sys

from typing import Optional

# Third-Party Library Imports
from mpi4py import MPI
import numpy as np
import netCDF4 as nc
from scipy.interpolate import interpn
import xarray as xr

# Local Library Imports
from utils.consts import NP_INT, NP_REAL, MPI_INT, MPI_REAL, NC_REAL, NC_INT, \
    MPI_COMM, NP_ARRAY, NC_VARIABLE, XR_DATASET, MPI_ROOT, g
from utils.rte_rrtmgp_cpp_fields import grid_dimensions, grid_descriptions, \
    grid_units, fields_dimensions, fields_descriptions, fields_units

# Type aliases

def main(argv):
    # Communicator info
    comm: MPI_COMM = MPI.COMM_WORLD
    l_rank: NP_INT = NP_INT(comm.Get_rank())
    comm_size: NP_INT = NP_INT(comm.Get_size())

    ## Parse command-line input
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog = "convert_dpscream_output",
        description = "Creates output from DP-SCREAM to input to RTE-RRTMGP-CPP.")
    
    parser.add_argument("--input_dir",
        action = "store",
        nargs = 1,
        type = str,
        required = True,
        help = "Path to RTE-RRTMGP-CPP+RT input.")
    
    parser.add_argument("--coarse_factors",
        action = "store",
        nargs = 1,
        type = str,
        required = False,
        default = [None],
        help = "Factors by which to coarsen the input horizontal grid.")
    
    args: argparse.Namespace = parser.parse_args()
    
    input_dir: str = os.path.normpath(args.input_dir[0])
    if args.coarse_factors[0] is None:
        coarse_factors: NP_ARRAY[NP_INT] = np.array([2], dtype = NP_INT)
    else:
        coarse_factors: NP_ARRAY[NP_INT] = np.array( \
            ast.literal_eval(args.coarse_factors[0]), dtype = NP_INT).flatten()

    ### RTE-RRTMGP-CPP+RT field keys
    fields: dict = {}
    rte_3dfield_keys: list = ["p_lay", "p_lev", "t_lay", "t_lev", "rh", "lwp",
        "iwp", "rel", "dei", "vmr_ch4", "vmr_co", "vmr_co2", "vmr_h2o",
        "vmr_n2", "vmr_n2o", "vmr_o2", "vmr_o3"]
    rte_2dfield_keys: list[str] = ["t_sfc", "mu0"]

    # Root rank opens the original RTE-RRTMGP-CPP+RT input file
    input_file_paths: list[str] = sorted([os.path.join(input_dir, file_path)
        for file_path in os.listdir(input_dir) if "lr" not in file_path])
    file_ext: re.Pattern = re.compile(".in.nc")
    input_file_path: str
    for input_file_path in input_file_paths:
        input_file_path_root: str = file_ext.sub("", input_file_path) # ASSUME: input_file_path ends with ".in.nc"

        coarse_factor: NP_INT
        for coarse_factor in coarse_factors:
            if l_rank == MPI_ROOT:
                xr_input: XR_DATASET = xr.open_dataset(input_file_path,
                    engine = "netcdf4", decode_timedelta = False)

                ## Read the original grids
                x: NP_ARRAY[NP_REAL] = xr_input["x"].values.astype(NP_REAL) # Column-center - x-dimension [m]; (n_col_x)
                y: NP_ARRAY[NP_REAL] = xr_input["y"].values.astype(NP_REAL) # Column-center - y-dimension [m]; (n_col_y)
                z_lay: NP_ARRAY[NP_REAL] = xr_input["z_lay"].values.astype(NP_REAL) # Layer altitude - z-dimension [m]; (n_lay_z)
                z_lev: NP_ARRAY[NP_REAL] = xr_input["z_lev"].values.astype(NP_REAL) # Level altitude - z-dimension [m]; (n_lay_z)

                xh: NP_ARRAY[NP_REAL] = xr_input["xh"].values.astype(NP_REAL) # Column-interfaces - x-dimension [m]; (n_col_x + 1)
                yh: NP_ARRAY[NP_REAL] = xr_input["yh"].values.astype(NP_REAL) # Column-interfaces - y-dimension [m]; (n_col_y + 1)

                n_col_x: Optional[NP_INT] = NP_INT(xr_input.sizes["x"])
                n_col_y: Optional[NP_INT] = NP_INT(xr_input.sizes["y"])
                n_lay_z: Optional[NP_INT] = NP_INT(xr_input.sizes["z"])
                n_lev_z: Optional[NP_INT] = NP_INT(xr_input.sizes["zh"])

                ## Coarsen the original grid - _lr := "low-resolution"
                n_col_x_lr: Optional[NP_INT] = n_col_x // coarse_factor
                n_col_y_lr: Optional[NP_INT] = n_col_y // coarse_factor

                x_min: NP_REAL = np.min(xh)
                x_max: NP_REAL = np.max(xh)
                xh_lr: NP_ARRAY[NP_REAL] = np.linspace(x_min, x_max, n_col_x_lr + 1,
                    dtype = NP_REAL)
                x_lr: Optional[NP_ARRAY[NP_REAL]] = (xh_lr[1:] + xh_lr[:-1]) / 2.

                y_min: NP_REAL = np.min(yh)
                y_max: NP_REAL = np.max(yh)
                yh_lr: NP_ARRAY[NP_REAL] = np.linspace(y_min, y_max, n_col_y_lr + 1,
                    dtype = NP_REAL)
                y_lr: Optional[NP_ARRAY[NP_REAL]] = (yh_lr[1:] + yh_lr[:-1]) / 2.

                ## Store spatial grid for outputting to RTE-RRTMGP-CPP input file
                grid: dict = {}

                ### NOTE: The number of points in the horizontal and vertical acceleration grids "should"
                ### be between 1/10 and 1/20 of n_col_x, n_col_y, n_col_z
                ### NOTE: These are the time-independent quantities
                ngrid_x_lr: NP_INT = NP_INT(np.ceil(n_col_x_lr / 10))
                ngrid_y_lr: NP_INT = NP_INT(np.ceil(n_col_y_lr / 10))
                ngrid_z: NP_INT = NP_INT(np.ceil(n_lay_z / 10))

                grid["x"] = x_lr
                grid["xh"] = xh_lr
                grid["z"] = z_lay

                grid["y"] = y_lr
                grid["yh"] = yh_lr
                grid["zh"] = z_lev

                grid["z_lay"] = z_lay
                grid["z_lev"] = z_lev

                grid["ngrid_x"] = ngrid_x_lr
                grid["ngrid_y"] = ngrid_y_lr
                grid["ngrid_z"] = ngrid_z

                ## Number of shortwave and longwave bands
                band_sw: NP_INT = NP_INT(xr_input.sizes["band_sw"])
                band_lw: NP_INT = NP_INT(xr_input.sizes["band_lw"])

                ## Fields that are not specified in the DP-SCREAM output
                set_unspecified_fields(n_col_x_lr, n_col_y_lr, n_lay_z, band_sw, band_lw,
                    fields)
            else:
                n_col_x: Optional[NP_INT] = None
                n_col_y: Optional[NP_INT] = None
                x: Optional[NP_ARRAY[NP_REAL]] = None
                y: Optional[NP_ARRAY[NP_REAL]] = None
                n_lay_z: Optional[NP_INT] = None
                n_lev_z: Optional[NP_INT] = None
                n_col_x_lr: Optional[NP_INT] = None
                n_col_y_lr: Optional[NP_INT] = None
                x_lr: Optional[NP_ARRAY[NP_REAL]] = None
                y_lr: Optional[NP_ARRAY[NP_REAL]] = None

            ## Broadcast info to non-root ranks
            n_col_x = comm.bcast(n_col_x, root = MPI_ROOT)
            n_col_y = comm.bcast(n_col_y, root = MPI_ROOT)
            x = comm.bcast(x, root = MPI_ROOT)
            y = comm.bcast(y, root = MPI_ROOT)
            n_lay_z = comm.bcast(n_lay_z, root = MPI_ROOT)
            n_lev_z = comm.bcast(n_lev_z, root = MPI_ROOT)
            n_col_x_lr = comm.bcast(n_col_x_lr, root = MPI_ROOT)
            n_col_y_lr = comm.bcast(n_col_y_lr, root = MPI_ROOT)
            x_lr = comm.bcast(x_lr, root = MPI_ROOT)
            y_lr = comm.bcast(y_lr, root = MPI_ROOT)

            mesh_x_lr: NP_ARRAY[NP_REAL]
            mesh_y_lr: NP_ARRAY[NP_REAL]
            mesh_y_lr, mesh_x_lr = np.meshgrid(y_lr, x_lr, indexing = "ij")
            mesh_pts_lr: NP_ARRAY[NP_REAL] = np.stack( \
                [mesh_y_lr.flatten(), mesh_x_lr.flatten()], axis = 1)

            ## Set up counts, displs split for scatterv
            g_nlays: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
            for ii in range(0, comm_size):
                g_nlays[ii] = n_lay_z // comm_size + int(ii < (n_lay_z % comm_size))
            l_nlay: int = g_nlays[l_rank]

            g_nlevs: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
            for ii in range(0, comm_size):
                g_nlevs[ii] = n_lev_z // comm_size + int(ii < (n_lev_z % comm_size))
            l_nlev: int = g_nlevs[l_rank]

            counts_lay: NP_ARRAY[NP_INT] = g_nlays * n_col_x * n_col_y
            displs_lay: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
            displs_lay[1:] += n_col_x * n_col_y * np.cumsum(g_nlays)[:-1]

            counts_lay_lr: NP_ARRAY[NP_INT] = g_nlays * n_col_x_lr * n_col_y_lr
            displs_lay_lr: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
            displs_lay_lr[1:] += n_col_x_lr * n_col_y_lr * np.cumsum(g_nlays)[:-1]

            counts_lev: NP_ARRAY[NP_INT] = g_nlevs * n_col_x * n_col_y
            displs_lev: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
            displs_lev[1:] += n_col_x * n_col_y * np.cumsum(g_nlevs)[:-1]

            counts_lev_lr: NP_ARRAY[NP_INT] = g_nlevs * n_col_x_lr * n_col_y_lr
            displs_lev_lr: NP_ARRAY[NP_INT] = np.zeros(comm_size, dtype = NP_INT)
            displs_lev_lr[1:] += n_col_x_lr * n_col_y_lr * np.cumsum(g_nlevs)[:-1]

            ## Read the 3D fields from the file, remap them to regular z-levels, and store them
            for ii in range(0, len(rte_3dfield_keys)):
                field: Optional[NP_ARRAY[NP_REAL]] = None
                field_min: Optional[NP_REAL] = None
                field_max: Optional[NP_REAL] = None

                rte_field_key: str = rte_3dfield_keys[ii]
                ## Assume field values at layer altitude unless specified at level
                if rte_field_key[-4:] == "_lev":
                    field_flag: str = "lev"
                else:
                    field_flag: str = "lay"
                # Root Rank reads input file and scattervs
                if l_rank == MPI_ROOT:
                    field: NP_ARRAY[NP_REAL] = xr_input[rte_field_key].values.astype(NP_REAL)

                    ## Get field min and max
                    ## Exceptions
                    if rte_field_key in ["rel"]: # Between 2.5 μm and 21.5 μm
                        field_min = NP_REAL(2.5)
                        field_max = NP_REAL(21.5)
                    elif rte_field_key in ["dei"]: # Between 10. μm and 180. μm
                        field_min = NP_REAL(10.)
                        field_max = NP_REAL(180.)
                    else:
                        field_min = field.min()
                        field_max = field.max()

                # scatterv the field
                field_min = comm.bcast(field_min, root = MPI_ROOT)
                field_max = comm.bcast(field_max, root = MPI_ROOT)

                if field_flag == "lev":
                    l_field: NP_ARRAY[NP_REAL] = np.empty([l_nlev, n_col_y, n_col_x], dtype = NP_REAL)
                    counts: NP_ARRAY[NP_INT] = counts_lev
                    displs: NP_ARRAY[NP_INT] = displs_lev
                else:
                    l_field: NP_ARRAY[NP_REAL] = np.empty([l_nlay, n_col_y, n_col_x], dtype = NP_REAL)
                    counts: NP_ARRAY[NP_INT] = counts_lay
                    displs: NP_ARRAY[NP_INT] = displs_lay

                comm.Scatterv([field, counts, displs, MPI_REAL], l_field, 
                    root = MPI_ROOT)

                ## Interpolate the field to the coarse grid
                if field_flag == "lev":
                    l_field_lr: NP_ARRAY[NP_REAL] = \
                            np.empty([l_nlev, n_col_y_lr, n_col_x_lr], dtype = NP_REAL)
                    for jj in range(0, l_nlev):
                        l_field_lr[jj,...] = np.reshape( \
                            interpn([y, x], l_field[jj,...], mesh_pts_lr),
                            [n_col_y_lr, n_col_x_lr])
                else:
                    l_field_lr: NP_ARRAY[NP_REAL] = \
                            np.empty([l_nlay, n_col_y_lr, n_col_x_lr], dtype = NP_REAL)
                    for jj in range(0, l_nlay):
                        l_field_lr[jj,...] = np.reshape( \
                            interpn([y, x], l_field[jj,...], mesh_pts_lr),
                            [n_col_y_lr, n_col_x_lr])

                ## Limit the interpolated (and extrapolated) field values
                ## Exceptions:
                l_field_lr[l_field_lr < field_min] = field_min
                l_field_lr[l_field_lr > field_max] = field_max

                # Reconstruct the full field
                field_lr: Optional[NP_ARRAY[NP_REAL]] = None
                if field_flag == "lev":
                    if l_rank == MPI_ROOT:
                        field_lr = np.empty([n_lev_z, n_col_y_lr, n_col_x_lr], dtype = NP_REAL)
                    counts: NP_ARRAY[NP_INT] = counts_lev_lr
                    displs: NP_ARRAY[NP_INT] = displs_lev_lr
                else:
                    if l_rank == MPI_ROOT:
                        field_lr = np.empty([n_lay_z, n_col_y_lr, n_col_x_lr], dtype = NP_REAL)
                    counts: NP_ARRAY[NP_INT] = counts_lay_lr
                    displs: NP_ARRAY[NP_INT] = displs_lay_lr

                comm.Gatherv(l_field_lr, [field_lr, counts, displs, MPI_REAL],
                    root = MPI_ROOT)

                if l_rank == MPI_ROOT:
                    fields[rte_field_key] = field_lr

                comm.Barrier()

            ## Extract 2-D fields - done in serial because we parallelize over layers
            for ii in range(0, len(rte_2dfield_keys)):
                if l_rank == MPI_ROOT:
                    rte_field_key: str = rte_2dfield_keys[ii]

                    field: NP_ARRAY[NP_REAL] = xr_input[rte_field_key].values.astype(NP_REAL) # 2-D field; (n_col_y, n_col_x)

                    ## Exceptions
                    if rte_field_key in ["t_sfc", "mu0"]: # In case fill values are unreasonable
                        if rte_field_key in ["t_sfc"]: # Between 0 K and 2300 K (max natural temperature on Earth)
                            field_min: NP_REAL = NP_REAL(0.0)
                            field_max: NP_REAL = NP_REAL(2300.0)
                        elif rte_field_key in ["mu0"]: # Between -1.0 and 1.0
                            field_min: NP_REAL = NP_REAL(-1.0)
                            field_max: NP_REAL = NP_REAL(1.0)

                    field_lr: NP_ARRAY[NP_REAL] = np.empty([n_col_y_lr, n_col_x_lr],
                        dtype = NP_REAL)

                    field_lr = np.reshape( \
                        interpn([y, x], field, mesh_pts_lr), [n_col_y_lr, n_col_x_lr])

                    field_lr[field_lr > field_max] = field_max
                    field_lr[field_lr < field_min] = field_min

                    fields[rte_field_key] = field_lr

            comm.Barrier()

            ## Write to RTE-RRTMGP-CPP input file, with the varying given solar zenith angles
            if l_rank == MPI_ROOT:
                output_file_path: str = input_file_path_root \
                        + ".lr_{:02.0f}".format(coarse_factor) + ".in.nc"

                nc_file: NC_DATASET = nc.Dataset(output_file_path, mode = "w",
                    datamodel = "NETCDF4", clobber = True)

                nc_file.createDimension("x", n_col_x_lr)
                nc_file.createDimension("y", n_col_y_lr)
                nc_file.createDimension("lay", n_lay_z)
                nc_file.createDimension("lev", n_lev_z)
                nc_file.createDimension("z", n_lay_z)
                nc_file.createDimension("xh", n_col_x_lr + 1)
                nc_file.createDimension("yh", n_col_y_lr + 1)
                nc_file.createDimension("zh", n_lev_z)
                nc_file.createDimension("band_lw", band_lw)
                nc_file.createDimension("band_sw", band_sw)

                ## Spatial grid
                for rte_grid_key in grid_dimensions.keys():
                    field: NP_ARRAY[NP_REAL] = grid[rte_grid_key]
                    field_dimensions: str | tuple[Optional[str]] = grid_dimensions[rte_grid_key]
                    field_description: str = grid_descriptions[rte_grid_key]
                    field_units: str = grid_units[rte_grid_key]

                    nc_field: NC_VARIABLE = nc_file.createVariable(rte_grid_key, NC_REAL, field_dimensions)
                    nc_field.description: str = field_description
                    nc_field.units: str = field_units
                    nc_field[...]: NP_ARRAY[NP_REAL] = field

                ## Fields
                for rte_field_key in fields_dimensions:
                    field: NP_ARRAY[NP_REAL] = fields[rte_field_key][:]

                    field_dimensions: tuple[str] = fields_dimensions[rte_field_key]
                    field_description: str = fields_descriptions[rte_field_key]
                    field_units: str = fields_units[rte_field_key]

                    nc_field: NC_VARIABLE = nc_file.createVariable(rte_field_key, NC_REAL, field_dimensions)
                    nc_field.description: str = field_description
                    nc_field.units: str = field_units
                    nc_field[...]: NP_ARRAY[NP_REAL] = field

                nc_file.close()

            comm.Barrier()

def set_unspecified_fields(n_col_x: NP_INT, n_col_y: NP_INT, n_lay_z: NP_INT,
    n_bnd_sw: NP_INT, n_bnd_lw: NP_INT, fields: dict) -> None:
    """
    Set fields not specified by the DP-SCREAM output.
    """

    ## Longwave boundary conditions
    fields["emis_sfc"]: NP_ARRAY[NP_REAL] = \
        np.ones((n_col_y, n_col_x, n_bnd_lw), dtype = NP_REAL) # Surface emissivity [N/A]

    ## Shortwave boundary conditions
    fields["sfc_alb_dir"]: NP_ARRAY[NP_REAL] = \
        np.ones((n_col_y, n_col_x, n_bnd_sw), dtype = NP_REAL) * 0.07 # Surface Albedo - Direct
    fields["sfc_alb_dif"]: NP_ARRAY[NP_REAL] = \
        np.ones((n_col_y, n_col_x, n_bnd_sw), dtype = NP_REAL) * 0.07 # Surface Albedo - Diffuse

    fields["tsi"]: NP_ARRAY[NP_REAL] = \
        np.ones((n_col_y, n_col_x), dtype = NP_REAL) * 551.58 # Total Solar Irradiance [W m^(-2)]

    fields["azi"]: NP_ARRAY[NP_REAL] = \
        np.ones((n_col_y, n_col_x), dtype = NP_REAL) * 0.0 # Azimuthal Angle [Radians]

    ## Set quantities not expected to be set in the DP-SCREAM output
    ### Gas volume mixing ratios
    fields["vmr_ccl4"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Carbon Tetrachloride [kg kg^(-1)]
    fields["vmr_cfc11"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Trichlorofluoromethane (CFC-11) [kg kg^(-1)]
    fields["vmr_cfc12"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Dichlorodifluoromethane (CFC-12) [kg kg^(-1)]
    fields["vmr_cfc22"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Chlorodifluoromethane (HCFC-22) [kg kg^(-1)]
    fields["vmr_hfc143a"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # 1,1,1-Trifluoroethane (HFC-143a) [kg kg^(-1)]
    fields["vmr_hfc125"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Pentafluoroethane (HFC-125) [kg kg^(-1)]
    fields["vmr_hfc32"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Difluoromethane (HFC-32) [kg kg^(-1)]
    fields["vmr_hfc23"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Trifluoromethane (HFC-23) [kg kg^(-1)]
    fields["vmr_hfc134a"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # 1,1,1,2-Tetrafluoroethane (HFC-134a) [kg kg^(-1)]
    fields["vmr_cf4"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Carbon Tetrafluoride (CF₄) [kg kg^(-1)]
    fields["vmr_no2"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Nitrogen Dioxide [kg kg^(-1)]

    ### Aerosol mixing ratios
    fields["aermr01"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Sea salt aerosol (0.03 - 0.5 µm)
    fields["aermr02"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Sea salt aerosol (0.5 - 5 µm)
    fields["aermr03"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Sea salt aerosol (5 - 20 µm)
    fields["aermr04"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Dust aerosol (0.03 - 0.55 µm)
    fields["aermr05"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Dust aerosol (0.55 - 0.9 µm)
    fields["aermr06"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Dust aerosol (0.9 - 20 µm)
    fields["aermr07"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Hydrophilic Organic Matter Aerosol
    fields["aermr08"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Hydrophobic Organic Matter Aerosol
    fields["aermr09"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Hydrophilic Black Carbon Aerosol
    fields["aermr10"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Hydrophobic Black Carbon Aerosol
    fields["aermr11"]: NP_ARRAY[NP_REAL] = \
        np.zeros((n_lay_z, n_col_y, n_col_x), dtype = NP_REAL) # Sulfate Aerosol

if __name__ == "__main__":
    main(sys.argv)
