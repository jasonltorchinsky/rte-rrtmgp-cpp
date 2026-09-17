g = 9.80665 # Acceleration due to gravity at equator at sea level [m s^{-2}]
R_d = 287.047 # Gas constant for dry air [J kg^{-1} K^{-1}]
rho_sw = 1.027e3 # Reference density of sea water [kg^{-1} m^{-3}]
cp_sw = 3986. # Reference specific heat at constant pressure of seawater [J kg^{-1} K^{-1}]
h_m = 19.753 # Approximate mixing layer depth of the GATEIII region in August [m]
cp_d = 1.0061e3 # Specific heat of dry air at constant pressure [J kg^{-1} K^{-1}]
cp_lw = 4184. # Specific heat of liquid water at constant pressure [J kg^{-1} K^{-1}]
cp_iw = 2093. # Specific heat of ice water at constant pressure [J kg^{-1} K^{-1}]
sec_per_day = 86400. # Seconds per day [s d^{-1}]

heating_cmap = "hot"
flux_cmap = "magma"
plot_colors = ["#332288", "#117733", "#44AA99", "#88CCEE", 
    "#DDCC77", "#CC6677", "#AA4499", "#882255"]