# Standard Library Imports

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY
from consts.physical import g

def calc_mass_moist_air(rad_tran_infile: str, time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None) -> XR_DATAARRAY | list[NP_ARRAY[NP_REAL]]:
    
    #---------------------------------------------------------------------------
    # Extract relevant fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        p_lev: XR_DATAARRAY = xr_rad_tran["p_lev"] # Hydrostatic pressure at levels; [nt, lev, y, x]; [Pa]
        lay: XR_DATAARRAY = xr_rad_tran["lay"] # Geometric height at layers; [lay]; [m]

    #---------------------------------------------------------------------------
    # Select relevant times for fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    p_lev = p_lev.isel(time = time_indices) # [time, lev, y, x]

    #---------------------------------------------------------------------------
    # Calculate grid spacing - ASSUME: Same in x-, y- throughout domain
    #---------------------------------------------------------------------------
    dx: NP_REAL = NP_REAL((p_lev["x"][1] - p_lev["x"][0]).to_numpy()) # [m]
    dy: NP_REAL = dx # [m]

    #---------------------------------------------------------------------------
    # Calculate moist air mass
    #---------------------------------------------------------------------------
    if x_indices is None:
        pdel: XR_DATAARRAY = (p_lev.diff("lev")).rename({"lev" : "lay"}).assign_coords({"lay" : lay}) # Pressure-thickness; [time, lay, y]; [Pa]
        mass_moist_air: XR_DATAARRAY = (pdel * dx * dy) / g # From hydrostatic pressure definition; [kg]
        mass_moist_air = (mass_moist_air
            .assign_attrs({"units" : "kg", 
                           "long_name" : "layer moist air mass",
                           "standard_name" : "moist_air_mass"})
            .rename("mass_moist_air"))
    else:
        mass_moist_air: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            p_lev_x: XR_DATAARRAY = p_lev.isel(time = ii, x = x_indices[ii]) # [lev, ny]
            pdel_x: XR_DATAARRAY = (p_lev_x.diff("lev")).rename({"lev" : "lay"}).assign_coords({"lay" : lay}) # Pressure-thickness; [lay, ny]; [Pa]

            mass_moist_air[ii] = ((pdel_x * dx * dy) / g).to_numpy().astype(NP_REAL) # From hydrostatic pressure definition; [kg]

    return mass_moist_air