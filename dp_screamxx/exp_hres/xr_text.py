import numpy as np
import xarray as xr

def main():
    np.random.seed(0)
    temperature = 15 + 8 * np.random.randn(2, 3, 4)
    precipitation = 10 * np.random.rand(2, 3, 4)
    lon = [-99.83, -99.32, -98.97]
    lat = [42.25, 42.21]
    instruments = ["manufac1", "manufac2", "manufac3", "manufac4"]

    data_vars: dict = dict(
        scalar = 5.,
        temperature = (["lat", "lon", "instrument"], temperature, dict(units = "K")),
        precipitation = (["lat", "lon", "instrument"], precipitation, dict(units = "mm", type = "rain")),
    )

    coords: dict = dict(
        lon = ("lon", lon, dict(units = "degrees: 0 - 360")),
        lat = ("lat", lat, dict(units = "degrees: 0 - 90")),
        instrument = instruments
    )

    attrs: dict = dict(
        description = "Weather related data."
    )

    ds = xr.Dataset(
        data_vars = data_vars,
        coords = coords,
        attrs = attrs,
    )

    breakpoint()

    ds.to_netcdf("dataset.nc")

if __name__ == "__main__":
    main()