# Standard Library Imports
from typing import Optional

# Third-Party Library Imports
import xarray as xr

# Local imports
from consts.dtypes import NP_INT, NP_REAL, NP_ARRAY, XR_DATASET, XR_DATAARRAY

def calc_cloud_wc(rad_tran_infile: str, time_indices: NP_ARRAY[NP_INT] = None,
    x_indices: NP_ARRAY[NP_INT] = None, zmax: Optional[NP_REAL] = None) -> list[NP_ARRAY[NP_REAL]]:
    #---------------------------------------------------------------------------
    # Get indexers for xarray data arrays
    #---------------------------------------------------------------------------
    indexers: dict = {}
    if ((time_indices is not None) and (x_indices is not None)):
        indexers["time"] = XR_DATAARRAY(time_indices, dims = "slice")
        indexers["x"] = XR_DATAARRAY(x_indices, dims = "slice")
    elif ((time_indices is not None) and (x_indices is None)):
        indexers["time"] = XR_DATAARRAY(time_indices, dims = "time")
    elif ((time_indices is None) and (x_indices is not None)):
        indexers["x"] = XR_DATAARRAY(x_indices, dims = "x")

    ## TO-DO: CONTINUE FROM HERE WITH Z SELECTING AND ALL THAT TO GET RID OF LIST-NUMPY WEIRDNESS

    #---------------------------------------------------------------------------
    # Extract relevant fields from RTE-RRTMGP-CPP file
    #---------------------------------------------------------------------------
    

    xr_rad_tran: XR_DATASET
    with xr.open_dataset(rad_tran_infile, engine = "netcdf4", decode_timedelta = False) as xr_rad_tran:
        breakpoint()
        lwp: XR_DATAARRAY = xr_rad_tran["lwp"].isel(time = time_indices).load() # Cloud liquid water path; [nt, lay, y, x]; [g m^{2}]
        iwp: XR_DATAARRAY = xr_rad_tran["iwp"].isel(time = time_indices).load() # Cloud ice water path; [nt, lay, y, x]; [g m^{2}]
        z: XR_DATAARRAY = xr_rad_tran["z"].load() # Layer midpoints; [lay]; [m]

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

        if zmax is not None:
            cloud_wc = cloud_wc.sel(lay = slice(0, zmax * 1.e3)) # zmax [km] => [m]
    else:
        cloud_wc: list[NP_ARRAY[NP_REAL]] = [[] for _ in range(0, 3)]
        for ii in range(0, 3): # Assume Morning-Noon-Night indices
            lwp_x: XR_DATAARRAY = lwp.isel(time = ii, x = x_indices[ii])
            iwp_x: XR_DATAARRAY = iwp.isel(time = ii, x = x_indices[ii])

            cloud_wc_x: XR_DATAARRAY = (lwp_x + iwp_x) / dz

            if zmax is not None:
                cloud_wc_x = cloud_wc_x.sel(lay = slice(0, zmax * 1.e3)) # zmax [km] => m

            cloud_wc[ii] = (cloud_wc_x).to_numpy().astype(NP_REAL) * mass_moist_air_x # [g m^{-3}]; [lay, y]

    return cloud_wc