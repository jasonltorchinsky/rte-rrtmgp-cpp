# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Numeric constants
np_float: np.float64 = np.float64
np_EPS: np.float64 = np.finfo(np_float).resolution
np_INF: np.float64 = np.finfo(np_float).max

np_SMALL: np.float64 = np.sqrt(np_EPS)
np_LARGE: np.float64 = np.sqrt(np_INF)

# Physical constants
mu_d: float = 28.9467e-3 # Mean molar mass of dry air - https://www.engineeringtoolbox.com/molecular-mass-air-d_679.html [kg mol^(-1)]
mu_v: float = 18.0153e-3 # Molar mass of water - https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185 [kg mol^(-1)]

R: float = 8.314462619 # Molar gas constant - https://physics.nist.gov/cgi-bin/cuu/Value?r [J mol^(-1) K^(-1)]
R_d: float = R / mu_d # Gas constant for dry air [J kg^(-1) K^(-1)]
R_v: float = R / mu_v # Gas constant for water vapor [J kg^(-1) K^(-1)]

L_v: float = 2.5009e6 # Latent heat of vaporization of water at 0.01C - https://www.engineeringtoolbox.com/water-properties-d_1573.html [J kg^(-1)]

g: float = 9.80665 # Standard acceleration of gravity - https://physics.nist.gov/cgi-bin/cuu/Value?gn [m s^(-2)]
p_0: float = 101325.0 # Standard atmospheric pressure - https://physics.nist.gov/cgi-bin/cuu/Value?stdatm [Pa]