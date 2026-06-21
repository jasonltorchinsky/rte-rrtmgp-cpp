# Standard Library Imports

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def calc_cloud_wc(rad_tran_infile: str, time_indices: NP_ARRAY[NP_INT],
    x_indices: NP_ARRAY[NP_INT] = None, zmax_index: NP_INT = None, detailed_calc: bool = False) -> list[NP_ARRAY[NP_REAL]]:
    
    #---------------------------------------------------------------------------
    # Extract relevant fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        lwp: XR_DATAARRAY = xr_rad_tran["lwp"] # Cloud liquid water path; [nt, lay, y, x]; [g m^{2}]
        iwp: XR_DATAARRAY = xr_rad_tran["iwp"] # Cloud ice water path; [nt, lay, y, x]; [g m^{2}]
        z: XR_DATAARRAY = xr_rad_tran["z"] # Layer midpoints; [lay]; [m]

    #---------------------------------------------------------------------------
    # Select relevant times for fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    lwp = lwp.isel(time = time_indices) # [time, lay, y, x]; [g m^{-2}]
    iwp = iwp.isel(time = time_indices) # [time, lay, y, x]; [g m^{-2}]

    #---------------------------------------------------------------------------
    # Calculate grid spacing - ASSUME: Uniform vertical grid in time and space
    #---------------------------------------------------------------------------
    dz: NP_REAL = NP_REAL((z[1] - z[0]).to_numpy())

    #---------------------------------------------------------------------------
    # Calculate cloud water content
    #---------------------------------------------------------------------------
    if x_indices is None:
        cloud_wc: XR_DATARRAY = ((lwp + iwp) / dz) # [g m^{-3}]; [time, lay, y, x]
        cloud_wc = (cloud_wc
            .assign_attrs({"units" : "g m^{-3}", 
                           "long_name" : "midpoint cloud water content",
                           "standard_name" : "cloud_water_content"})
            .rename("cloud_water_content")
            .assign_coords({"lay" : z.to_numpy()}))
    else:
        cloud_wc: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            lwp_x: XR_DATAARRAY = lwp.isel(time = ii, x = x_indices[ii])
            iwp_x: XR_DATAARRAY = iwp.isel(time = ii, x = x_indices[ii])

            cloud_wc[ii] = ((lwp_x + iwp_x) / dz).to_numpy().astype(NP_REAL) * mass_moist_air_x # [g m^{-3}]; [lay, y]

    return cloud_wc
