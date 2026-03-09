# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET
from consts.rte_rrtmgp_cpp_fields import grid_descriptions, grid_units

def get_coords_01(xr_dpscream: XR_DATASET, sort_mask: NP_ARRAY[NP_INT]) -> dict:
    ## Construct a sorting mask for reordering "ncol" into x- and y-columns
    lon: NP_ARRAY[NP_REAL] = xr_dpscream["lon"].values.astype(NP_REAL) # Column-center - x-dimension [m]; (ncol)
    lat: NP_ARRAY[NP_REAL] = xr_dpscream["lat"].values.astype(NP_REAL) # Column center - y-dimension [m]; (ncol)

    nx: NP_INT = NP_INT(np.unique(lon).size) # No. columns in x
    ny: NP_INT = NP_INT(np.unique(lat).size) # No. columns in y
    cols: NP_ARRAY[NP_REAL] = np.stack((lon[sort_mask], lat[sort_mask]), axis = 1).reshape(nx, ny, 2)

    ## Construct the horizontal grids
    ### NOTE: Assume that horizontal grids are regularly-spaced.
    x: NP_ARRAY[NP_REAL] = (cols[:,:,0])[0,:] # x-midpoints of each column [m]; (nx)
    dx: NP_REAL = x[1] - x[0]
    xh: NP_ARRAY[NP_REAL] = np.append(x - (dx / 2.), x[-1] + (dx / 2.)) # x-interfaces of each column [m]; (nx + 1)

    y: NP_ARRAY[NP_REAL] = (cols[:,:,1])[:,0] # y-midpoints of each column [m]; (ny)
    dy: NP_REAL = y[1] - y[0]
    yh: NP_ARRAY[NP_REAL] = np.append(y - (dy / 2.), x[-1] + (dy / 2.)) # y-interfaces of each column [m]; (ny + 1)

    ### NOTE: The number of points in the horizontal acceleration grid "should"
    ### be between 1/10 and 1/20 of nx, ny
    ngrid_x: NP_INT = NP_INT(np.ceil(nx / 10))
    ngrid_y: NP_INT = NP_INT(np.ceil(ny / 10))

    # VERTICAL GRID
    # NOTE: Here we get the uniform, time-independent vertical grid that we will
    # remap values to
    nlay: NP_INT = NP_INT(xr_dpscream.sizes["lev"]) # No. DP-SCREAM levels (RTE layers)
    nlev: NP_INT = NP_INT(xr_dpscream.sizes["ilev"]) # No. DP-SCREAM level interfaces (RTE levels)

    z_min: NP_REAL = NP_REAL(xr_dpscream["z_mid"].isel(lev=[-1]).max()) # Lowest RTE level altitude on regular grid; [m]
    z_max: NP_REAL = NP_REAL(xr_dpscream["z_mid"].isel(lev=[0]).min()) # Highest RTE level altitude on regular grid; [m]

    z_lev: NP_ARRAY[NP_REAL] = np.linspace(z_min, z_max, nlev, dtype = NP_REAL) # Regularly-spaced RTE levels [m]; (nlev)
    z_lay: NP_ARRAY[NP_REAL] = (z_lev[1:] + z_lev[:-1]) / 2. # Regularly-spaced RTE layers [m]; (nlay)

    ### NOTE: The number of points in the vertical acceleration grid "should"
    ### be between 1/10 and 1/20 of nlay
    ngrid_z: NP_INT = NP_INT(np.ceil(nlay / 10))

    ## Wavelength info
    n_bnd_sw: NP_INT = NP_INT(xr_dpscream.sizes["swband"])
    n_bnd_lw: NP_INT = NP_INT(xr_dpscream.sizes["lwband"])

    ## Spatial RTE-RRTMGP-CPP coords
    coords: dict = dict(
        x = ("x", x, dict(description = grid_descriptions["x"], units = grid_units["x"])),
        xh = ("xh", xh, dict(description = grid_descriptions["xh"], units = grid_units["xh"])),
        y = ("y", y, dict(description = grid_descriptions["y"], units = grid_units["y"])),
        yh = ("yh", yh, dict(description = grid_descriptions["yh"], units = grid_units["yh"])),
        z = ("z", z_lay, dict(description = grid_descriptions["z"], units = grid_units["z"])),
        zh = ("zh", z_lev, dict(description = grid_descriptions["zh"], units = grid_units["zh"])),
        z_lay = ("z_lay", z_lay, dict(description = grid_descriptions["z_lay"], units = grid_units["z_lay"])),
        z_lev = ("z_lev", z_lev, dict(description = grid_descriptions["z_lev"], units = grid_units["z_lev"])),
        ngrid_x = ((), ngrid_x, dict(description = grid_descriptions["ngrid_x"], units = grid_units["ngrid_x"])),
        ngrid_y = ((), ngrid_y, dict(description = grid_descriptions["ngrid_y"], units = grid_units["ngrid_y"])),
        ngrid_z = ((), ngrid_z, dict(description = grid_descriptions["ngrid_z"], units = grid_units["ngrid_z"])),
        n_bnd_sw = ((), n_bnd_sw, dict(description = grid_descriptions["n_bnd_sw"], units = grid_units["n_bnd_sw"])),
        n_bnd_lw = ((), n_bnd_lw, dict(description = grid_descriptions["n_bnd_lw"], units = grid_units["n_bnd_lw"])),
    )

    return coords