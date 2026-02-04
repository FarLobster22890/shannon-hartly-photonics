import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Integrated photonics setup
# -------------------------
# Units: 1 = 1 µm
resolution = 300  # pixels/µm (increase for accuracy, slower)

wvl = 1.55       # µm (telecom)
f0 = 1 / wvl     # Meep frequency

# Materials (approx)
n_si = 3.48
n_sio2 = 1.44

si = mp.Medium(index=n_si)
sio2 = mp.Medium(index=n_sio2)

# Waveguide geometry (2D slab waveguide approximation)
wg_w = 0.45      # µm (typical silicon photonics width)
wg_len = 16      # total cell length in x
pad_y = 3.0      # vertical padding around waveguide

sx = wg_len
sy = wg_w + 2*pad_y
cell = mp.Vector3(sx, sy, 0)

dpml = 1.0
pml_layers = [mp.PML(dpml)]

# Waveguide centered in y
wg = mp.Block(
    size=mp.Vector3(mp.inf, wg_w, mp.inf),
    center=mp.Vector3(0, 0),
    material=si
)

geometry = [wg]

# Source: launch the guided mode (TE-like Ez in 2D)
# Place it left of center, spanning the waveguide region
src_x = -sx/2 + dpml + 1.0

sources = [mp.EigenModeSource(
    src=mp.ContinuousSource(frequency=f0),
    center=mp.Vector3(src_x, 0),
    size=mp.Vector3(0, sy - 2*dpml),
    eig_match_freq=True,
    eig_band=1,
    direction=mp.NO_DIRECTION,
    eig_kpoint=mp.Vector3(1,0,0),  # propagate +x
    component=mp.Ez
)]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    default_material=sio2,   # cladding
    sources=sources,
    resolution=resolution
)

# -------------------------
# Mode monitor (transmission)
# -------------------------
# Place a mode monitor near the right side to measure how much power remains in the guided mode
mon_x = sx/2 - dpml - 1.0
mode_mon = sim.add_mode_monitor(
    f0, 0, 1,
    mp.ModeRegion(center=mp.Vector3(mon_x, 0), size=mp.Vector3(0, sy - 2*dpml)),
    eig_band=1,
    direction=mp.NO_DIRECTION,
    eig_kpoint=mp.Vector3(1,0,0)
)

# Optional: flux monitor too (total power crossing plane)
flux_mon = sim.add_flux(
    f0, 0, 1,
    mp.FluxRegion(center=mp.Vector3(mon_x, 0), size=mp.Vector3(0, sy - 2*dpml))
)

# Run until steady state
sim.run(until=200)

# Get transmitted power in the fundamental mode
mode_data = sim.get_eigenmode_coefficients(mode_mon, [1], eig_parity=mp.NO_PARITY)
# alpha[0,0,0] is forward amplitude; power ~ |alpha|^2
alpha_fwd = mode_data.alpha[0,0,0]
P_mode = np.abs(alpha_fwd)**2

P_flux = mp.get_fluxes(flux_mon)[0]

print("=== Straight Waveguide Results ===")
print(f"Wavelength: {wvl} µm  (f0={f0:.6f})")
print(f"Mode power (fundamental): {P_mode:.6e}")
print(f"Total flux at monitor:     {P_flux:.6e}")
print("Note: mode power should be close to flux if most power stays in guided mode.\n")

# Field snapshot
eps = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ez  = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

plt.figure(figsize=(10,4))
plt.imshow(eps.T, origin="lower", aspect="auto", cmap="gray")
plt.title("Dielectric profile (ε)")
plt.colorbar()
plt.show()

plt.figure(figsize=(10,4))
plt.imshow(np.real(ez).T, origin="lower", aspect="auto", cmap="RdBu")
plt.title("Ez field snapshot (real part)")
plt.colorbar()
plt.show()

