# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import xarray as xr

# Local Library Imports
from consts.consts import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def get_g_grid_01(dp_scream_file: str, sort_mask: NP_ARRAY[NP_INT]) -> dict:
    xr_dp_scream: XR_DATASET
    with xr.open_dataset(dp_scream_file, engine = "netcdf4", decode_timedelta = False) as xr_dp_scream:
        #-----------------------------------------------------------------------
        # Construct the horizontal grid
        #-----------------------------------------------------------------------
        lon: XR_DATAARRAY = xr_dp_scream["lon"] # Column-center - x-dimension [m]; (ncol)
        lat: XR_DATAARRAY = xr_dp_scream["lat"] # Column center - y-dimension [m]; (ncol)

        nx: NP_INT = NP_INT(np.unique(lon).size) # No. columns in x
        ny: NP_INT = NP_INT(np.unique(lat).size) # No. columns in y
        cols: NP_ARRAY[NP_REAL] = np.stack((lon[sort_mask], lat[sort_mask]), axis = 1).reshape(nx, ny, 2).astype(NP_REAL)

        x: NP_ARRAY[NP_REAL] = np.ascontiguousarray((cols[:,:,0])[0,:]) # x-midpoints of each column [m]; (nx)
        dx: NP_REAL = x[1] - x[0] # Spacing in x [m]
        y: NP_ARRAY[NP_REAL] = np.ascontiguousarray((cols[:,:,1])[:,0]) # y-midpoints of each column [m]; (ny)
        dy: NP_REAL = y[1] - y[0] # Spacing in y [m]

        #-----------------------------------------------------------------------
        # Construct the uniform vertical grid that DP-SCREAM values will be 
        # interpolated to
        #-----------------------------------------------------------------------
        nlay: NP_INT = NP_INT(xr_dp_scream.sizes["lev"]) # No. DP-SCREAM levels (RTE layers)
        nlev: NP_INT = nlay + 1 # No. DP-SCREAM level interfaces (RTE levels)
        nz: NP_INT = nlay + nlev

        z_min: NP_REAL = NP_REAL(xr_dp_scream["z_mid"].isel(lev=[-1]).max()) # Lowest RTE level altitude on regular grid; [m]
        z_max: NP_REAL = NP_REAL(xr_dp_scream["z_mid"].isel(lev=[0]).min()) # Highest RTE level altitude on regular grid; [m]

        z_lev: NP_ARRAY[NP_REAL] = np.linspace(z_min, z_max, nlev, dtype = NP_REAL) # Regularly-spaced RTE levels [m]; (nlev)
        z_lay: NP_ARRAY[NP_REAL] = (z_lev[1:] + z_lev[:-1]) / 2. # Regularly-spaced RTE layers [m]; (nlay)
        z: NP_ARRAY[NP_REAL] = np.empty(nlev + nlay, dtype = NP_REAL)
        z[0::2] = z_lev
        z[1::2] = z_lay

        dz: NP_REAL = z_lev[1] - z_lev[0] # Layer thickness [m]

        #-----------------------------------------------------------------------
        # Collect grid information into dict
        #-----------------------------------------------------------------------
        g_grid: dict = dict(
            nx = nx,
            dx = dx,
            x = x,
            ny = ny,
            dy = dy,
            y = y,
            nlay = nlay,
            z_lay = z_lay,
            nlev = nlev,
            z_lev = z_lev,
            nz = nz,
            dz = dz,
            z = z
        )

        return g_grid