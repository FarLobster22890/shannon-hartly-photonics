"""
Diamond Waveguide - Proof of Concept
=====================================
2D slab waveguide simulation using diamond (n ≈ 2.4) as core material.
Demonstrates guided mode propagation at 637 nm (NV center emission wavelength).
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Physical parameters
# -------------------------
# Units: 1 Meep unit = 1 µm

# Diamond properties
n_diamond = 2.4       # refractive index at visible wavelengths
n_air = 1.0           # cladding (air-clad for simplicity)

# Wavelength: 637 nm (NV center zero-phonon line)
wvl = 0.637           # µm
f0 = 1 / wvl          # Meep frequency (~1.57)

# Waveguide geometry
wg_width = 0.4        # µm (typical for single-mode at 637 nm)
wg_length = 12        # µm (propagation length)

# Simulation domain
resolution = 40       # pixels per µm (increase for accuracy)
pad_y = 2.0           # vertical padding
dpml = 1.0            # PML thickness

sx = wg_length + 2*dpml
sy = wg_width + 2*pad_y + 2*dpml
cell = mp.Vector3(sx, sy, 0)

# -------------------------
# Materials and geometry
# -------------------------
diamond = mp.Medium(index=n_diamond)

geometry = [
    mp.Block(
        size=mp.Vector3(mp.inf, wg_width, mp.inf),
        center=mp.Vector3(0, 0),
        material=diamond
    )
]

# -------------------------
# Source: eigenmode launcher
# -------------------------
src_x = -sx/2 + dpml + 0.5

sources = [
    mp.EigenModeSource(
        src=mp.GaussianSource(f0, fwidth=0.2*f0),
        center=mp.Vector3(src_x, 0),
        size=mp.Vector3(0, sy - 2*dpml),
        eig_band=1,
        eig_match_freq=True,
        eig_kpoint=mp.Vector3(1, 0, 0),
        direction=mp.NO_DIRECTION,
    )
]

# -------------------------
# Simulation
# -------------------------
pml_layers = [mp.PML(dpml)]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
    default_material=mp.Medium(index=n_air),
)

# -------------------------
# Flux monitors
# -------------------------
# Input flux (near source)
in_x = src_x + 1.0
flux_in = sim.add_flux(
    f0, 0.4*f0, 50,
    mp.FluxRegion(center=mp.Vector3(in_x, 0), size=mp.Vector3(0, wg_width*2))
)

# Output flux (near end)
out_x = sx/2 - dpml - 1.0
flux_out = sim.add_flux(
    f0, 0.4*f0, 50,
    mp.FluxRegion(center=mp.Vector3(out_x, 0), size=mp.Vector3(0, wg_width*2))
)

# -------------------------
# Run simulation
# -------------------------
print(f"Diamond waveguide simulation")
print(f"  Wavelength: {wvl*1000:.0f} nm")
print(f"  Waveguide width: {wg_width*1000:.0f} nm")
print(f"  Diamond n = {n_diamond}")
print(f"  Resolution: {resolution} px/µm")
print()

# Run until fields decay
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(out_x, 0), 1e-6))

# -------------------------
# Results
# -------------------------
freqs = np.array(mp.get_flux_freqs(flux_out))
flux_in_data = np.array(mp.get_fluxes(flux_in))
flux_out_data = np.array(mp.get_fluxes(flux_out))

# Transmission
T = flux_out_data / (flux_in_data + 1e-30)
wavelengths = 1 / freqs  # convert back to µm

# Get field snapshot
eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

print(f"\nResults:")
print(f"  Peak transmission: {np.max(T):.3f}")
print(f"  Transmission at {wvl*1000:.0f} nm: {T[len(T)//2]:.3f}")

# -------------------------
# Plotting
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Dielectric profile
ax1 = axes[0, 0]
im1 = ax1.imshow(eps_data.T, origin='lower', cmap='gray',
                  extent=[-sx/2, sx/2, -sy/2, sy/2])
ax1.set_xlabel('x (µm)')
ax1.set_ylabel('y (µm)')
ax1.set_title('Dielectric profile (ε)')
plt.colorbar(im1, ax=ax1)

# Ez field
ax2 = axes[0, 1]
im2 = ax2.imshow(np.real(ez_data).T, origin='lower', cmap='RdBu',
                  extent=[-sx/2, sx/2, -sy/2, sy/2],
                  vmin=-np.max(np.abs(ez_data)), vmax=np.max(np.abs(ez_data)))
ax2.set_xlabel('x (µm)')
ax2.set_ylabel('y (µm)')
ax2.set_title('Ez field (real part)')
plt.colorbar(im2, ax=ax2)

# Transmission spectrum
ax3 = axes[1, 0]
ax3.plot(wavelengths * 1000, T)
ax3.axvline(wvl * 1000, color='r', linestyle='--', alpha=0.5, label=f'{wvl*1000:.0f} nm')
ax3.set_xlabel('Wavelength (nm)')
ax3.set_ylabel('Transmission')
ax3.set_title('Transmission spectrum')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Field cross-section at center
ax4 = axes[1, 1]
mid_x = ez_data.shape[0] // 2
y_coords = np.linspace(-sy/2, sy/2, ez_data.shape[1])
ax4.plot(y_coords, np.abs(ez_data[mid_x, :]))
ax4.axvline(-wg_width/2, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(wg_width/2, color='gray', linestyle='--', alpha=0.5, label='Waveguide edges')
ax4.set_xlabel('y (µm)')
ax4.set_ylabel('|Ez|')
ax4.set_title('Mode profile (x = center)')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('diamond_waveguide_results.png', dpi=150)
print(f"\nSaved: diamond_waveguide_results.png")
plt.show()
