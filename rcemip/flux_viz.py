import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Hardcoded file paths in the current directory
input_file = "rte_rrtmgp_input.nc"
output_file = "rte_rrtmgp_output.nc"

# -----------------------------
# Read solar zenith angle info
# -----------------------------
with Dataset(input_file, "r") as ds_in:
    x = ds_in.variables["x"][:]
    y = ds_in.variables["y"][:]
    z = ds_in.variables["z"][:]
    mu0 = ds_in.variables["mu0"][:]  # (y, x)

# -----------------------------
# Read output data
# -----------------------------
with Dataset(output_file, "r") as ds_out:
    rt_flux_abs_dir = ds_out.variables["rt_flux_abs_dir"][:]   # (z, y, x)
    rt_flux_abs_dif = ds_out.variables["rt_flux_abs_dif"][:]   # (z, y, x)
    rt_flux_sfc_dir = ds_out.variables["rt_flux_sfc_dir"][:]   # (y, x)
    rt_flux_sfc_dif = ds_out.variables["rt_flux_sfc_dif"][:]   # (y, x)

# -----------------------------
# Derived quantities
# -----------------------------
jmid = len(y) // 2

# Total absorbed flux from ray tracer
absorbed_flux = rt_flux_abs_dir + rt_flux_abs_dif          # (z, y, x)
absorbed_flux_xz = absorbed_flux[:, jmid, :]               # xz slice at central y

# Total downwelling surface flux from ray tracer
surface_downwelling_flux = rt_flux_sfc_dir + rt_flux_sfc_dif  # (y, x)

# Solar zenith angle from mu0 = cos(sza)
mu0_mean = np.mean(mu0)
sza_deg = np.degrees(np.arccos(np.clip(mu0_mean, -1.0, 1.0)))
sza_rounded = int(np.rint(sza_deg))

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# Left: absorbed flux in central y xz-slice
im0 = axes[0].pcolormesh(
    x, z, absorbed_flux_xz,
    shading="auto",
    cmap="plasma"
)
axes[0].set_title("Absorbed flux (ray tracer, central y xz-slice)")
axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("z [m]")
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label("Absorbed flux [W m$^{-3}$]")

# Right: downwelling surface flux
im1 = axes[1].pcolormesh(
    x, y, surface_downwelling_flux,
    shading="auto",
    cmap="plasma"
)
axes[1].set_title("Downwelling surface flux")
axes[1].set_xlabel("x [m]")
axes[1].set_ylabel("y [m]")
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label("Surface flux [W m$^{-2}$]")

fig.suptitle(f"RTE+RRTMGP Output, Solar Zenith Angle = {sza_deg:.2f}°", fontsize=16)

outfile = f"rte_rrtmgp_output_{sza_rounded}.png"
fig.savefig(outfile, dpi=200, bbox_inches="tight")
plt.show()