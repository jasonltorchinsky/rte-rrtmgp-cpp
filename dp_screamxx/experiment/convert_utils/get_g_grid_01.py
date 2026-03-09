# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET

def get_g_grid_01(xr_dpscream: XR_DATASET, sort_mask: NP_ARRAY[NP_INT]) -> dict:
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

    # VERTICAL GRID
    # NOTE: Here we get the uniform, time-independent vertical grid that we will
    # remap values to
    nlay: NP_INT = NP_INT(xr_dpscream.sizes["lev"]) # No. DP-SCREAM levels (RTE layers)
    nlev: NP_INT = NP_INT(xr_dpscream.sizes["ilev"]) # No. DP-SCREAM level interfaces (RTE levels)

    z_min: NP_REAL = NP_REAL(xr_dpscream["z_mid"].isel(lev=[-1]).max()) # Lowest RTE level altitude on regular grid; [m]
    z_max: NP_REAL = NP_REAL(xr_dpscream["z_mid"].isel(lev=[0]).min()) # Highest RTE level altitude on regular grid; [m]

    z_lev: NP_ARRAY[NP_REAL] = np.linspace(z_min, z_max, nlev, dtype = NP_REAL) # Regularly-spaced RTE levels [m]; (nlev)
    z_lay: NP_ARRAY[NP_REAL] = (z_lev[1:] + z_lev[:-1]) / 2. # Regularly-spaced RTE layers [m]; (nlay)
    z: NP_ARRAY[NP_REAL] = np.empty(nlev + nlay, dtype = NP_REAL)
    z[0::2] = z_lev
    z[1::2] = z_lay

    ## Spatial RTE-RRTMGP-CPP grid
    g_grid: dict = dict(
        nx = nx,
        x = x,
        xh = xh,
        ny = ny,
        y = y,
        yh = yh,
        nlay = nlay,
        nlev = nlev,
        z_lay = z_lay,
        z_lev = z_lev,
        z = z
    )

    return g_grid