# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Numeric precision
INT_PREC: int = 64
REAL_PREC: int = 64

assert(INT_PREC in [32, 64])
if INT_PREC == 32:
    from mpi4py.MPI import INT as MPI_INT
    from numpy import int32 as NP_INT
    NC_INT: str = "i4"
elif INT_PREC == 64:
    from mpi4py.MPI import LONG as MPI_INT
    from numpy import int64 as NP_INT
    NC_INT: str = "i8"

assert(REAL_PREC in [32, 64])
if REAL_PREC == 32:
    from mpi4py.MPI import FLOAT as MPI_REAL
    from numpy import float32 as NP_REAL
    NC_REAL: str = "f4"
elif REAL_PREC == 64:
    from mpi4py.MPI import DOUBLE as MPI_REAL
    from numpy import float64 as NP_REAL
    NC_REAL: str = "f8"

# Class aliases
from matplotlib.axes import Axes as MPL_AXES
from matplotlib.colorbar import Colorbar as MPL_COLORBAR
from matplotlib.figure import Figure as MPL_FIGURE
from matplotlib.contour import QuadContourSet as MPL_CONTOUR
from mpi4py.MPI import Intracomm as MPI_COMM
from netCDF4 import Dataset as NC_DATASET
from netCDF4._netCDF4 import Variable as NC_VARIABLE
from numpy import ndarray as NP_ARRAY
NP_DATETIME = np.dtype('datetime64[ns]')
from xarray.core.dataset import Dataset as XR_DATASET
from xarray.core.dataarray import DataArray as XR_DATAARRAY
NP_BOOL = np.bool_

# Numeric constants
NP_EPS: NP_REAL = np.finfo(NP_REAL).resolution
NP_INF: NP_REAL = np.finfo(NP_REAL).max

NP_SMALL: NP_REAL = np.sqrt(NP_EPS)
NP_LARGE: NP_REAL = np.sqrt(NP_INF)

# MPI constants
MPI_ROOT: NP_INT = NP_INT(0)

# Physical constants
mu_d: NP_REAL = NP_REAL(28.9467e-3) # Mean molar mass of dry air - https://www.engineeringtoolbox.com/molecular-mass-air-d_679.html [kg mol^(-1)]
mu_v: NP_REAL = NP_REAL(18.0153e-3) # Molar mass of water - https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185 [kg mol^(-1)]

R: NP_REAL = NP_REAL(8.314462619) # Molar gas constant - https://physics.nist.gov/cgi-bin/cuu/Value?r [J mol^(-1) K^(-1)]
R_d: NP_REAL = R / mu_d # Gas constant for dry air [J kg^(-1) K^(-1)]
R_v: NP_REAL = R / mu_v # Gas constant for water vapor [J kg^(-1) K^(-1)]

L_v: NP_REAL = NP_REAL(2.5009e6) # Latent heat of vaporization of water at 0.01C - https://www.engineeringtoolbox.com/water-properties-d_1573.html [J kg^(-1)]

g: NP_REAL = NP_REAL(9.80665) # Standard acceleration of gravity - https://physics.nist.gov/cgi-bin/cuu/Value?gn [m s^(-2)]
p_0: NP_REAL = NP_REAL(101325.0) # Standard atmospheric pressure - https://physics.nist.gov/cgi-bin/cuu/Value?stdatm [Pa]

