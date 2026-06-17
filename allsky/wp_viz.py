import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Hardcoded file path in the current directory
input_file = "rte_rrtmgp_input.nc"

# -----------------------------
# Read input data
# -----------------------------
with Dataset(input_file, "r") as ds_in:
    x = ds_in.variables["x"][:]
    y = ds_in.variables["y"][:]
    z = ds_in.variables["z"][:]
    mu0 = ds_in.variables["mu0"][:]  # (y, x)

    lwp = ds_in.variables["lwp"][:]   # (lay, y, x)
    iwp = ds_in.variables["iwp"][:]   # (lay, y, x)

# -----------------------------
# Derived quantities
# -----------------------------
jmid = len(y) // 2

# Total water path
total_water_path = lwp + iwp              # (lay, y, x)

# Central y xz-slice
total_water_path_xz = total_water_path[:, jmid, :]   # (lay, x)

# For plotting against z, use layer midpoints if needed
# Here we use z directly, assuming lay aligns with z spacing
# If you want exact layer-center coordinates, we can derive them from zh.
z_plot = z

# Solar zenith angle from mu0 = cos(sza)
mu0_mean = np.mean(mu0)
sza_deg = np.degrees(np.arccos(np.clip(mu0_mean, -1.0, 1.0)))
sza_rounded = int(np.rint(sza_deg))

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

im = ax.pcolormesh(
    x, z_plot, total_water_path_xz,
    shading="auto",
    cmap="viridis"
)
ax.set_title("Total Cloud Water Path (IWP + LWP), central y xz-slice")
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Water path [kg m$^{-2}$]")

fig.suptitle(f"RTE+RRTMGP Input, Solar Zenith Angle = {sza_deg:.2f}°", fontsize=16)

outfile = f"rte_rrtmgp_input_total_water_path_{sza_rounded}.png"
fig.savefig(outfile, dpi=200, bbox_inches="tight")
plt.show()