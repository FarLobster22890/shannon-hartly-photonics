#!/usr/bin/env python3
"""
FDTDX: Visualization Demo
Generates plots from synthetic EM fields (for demo purposes)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("FDTDX Visualization Demo")
print("=" * 60)
print()

# Create synthetic field data (simulating FDTDX output)
print("Generating synthetic EM fields...", flush=True)

nx, ny, nz = 60, 50, 60

# Create a simple waveguide-like field pattern
x = np.linspace(-3, 3, nx)
y = np.linspace(-2, 2, ny)
z = np.linspace(-3, 3, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Gaussian profile (simulates propagating wave)
E_x = np.exp(-((Y)**2 + (Z)**2) / 0.5) * np.cos(2*np.pi*Z/0.5)
E_y = np.exp(-((X)**2 + (Z)**2) / 0.3) * 0.2
E_z = np.exp(-((X)**2 + (Y)**2) / 0.4) * np.sin(2*np.pi*Z/0.5) * 0.1

# Field magnitudes
E_mag = np.sqrt(E_x**2 + E_y**2 + E_z**2)
H_mag = E_mag * 0.8  # simplified H from E

print(f"Field shape: {E_x.shape}")
print(f"Max |E|: {np.max(E_mag):.4e}")
print(f"Max |H|: {np.max(H_mag):.4e}")
print()

# Extract slices
mid_y = E_mag.shape[1] // 2
E_xy = E_mag[:, :, mid_y]
E_xz = E_mag[:, mid_y, :]
E_yz = E_mag[nx//2, :, :]

# Create plots
print("Creating plots...", flush=True)

# Overview figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('FDTDX Electromagnetic Field Distribution', fontsize=14, fontweight='bold')

# E-field XY
ax = axes[0, 0]
im = ax.imshow(E_xy.T, cmap='viridis', origin='lower', aspect='auto')
ax.set_title('|E| - XY Plane (Cross-section)')
ax.set_xlabel('X (grid points)')
ax.set_ylabel('Y (grid points)')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('E-field (V/m)')

# E-field XZ
ax = axes[0, 1]
im = ax.imshow(E_xz.T, cmap='viridis', origin='lower', aspect='auto')
ax.set_title('|E| - XZ Plane (Propagation)')
ax.set_xlabel('X (grid points)')
ax.set_ylabel('Z (grid points)')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('E-field (V/m)')

# H-field
ax = axes[1, 0]
im = ax.imshow(H_mag[:, :, mid_y].T, cmap='plasma', origin='lower', aspect='auto')
ax.set_title('|H| - XY Plane')
ax.set_xlabel('X (grid points)')
ax.set_ylabel('Y (grid points)')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('H-field (A/m)')

# Statistics
ax = axes[1, 1]
ax.axis('off')
stats = f"""
SIMULATION SUMMARY

Grid: {nx} × {ny} × {nz}
  Total cells: {nx*ny*nz:,}

Field Statistics:
  Max |E|: {np.max(E_mag):.4e} V/m
  Max |H|: {np.max(H_mag):.4e} A/m
  Mean |E|: {np.mean(E_mag):.4e} V/m

Field Energy:
  Total E² integral: {np.sum(E_mag**2):.4e}

Mode Characteristics:
  Propagation: +Z direction
  Transverse profile: Gaussian
  Wavelength: ~0.5 µm

Status: ✓ Visualization pipeline working
"""
ax.text(0.05, 0.95, stats, fontsize=11, family='monospace',
        verticalalignment='top', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('fdtdx_results.png', dpi=120, bbox_inches='tight')
print("✓ Saved: fdtdx_results.png")

# Field components
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10))
fig2.suptitle('E-Field Vector Components', fontsize=14, fontweight='bold')

components = ['Ex', 'Ey', 'Ez']
comp_data = [E_x, E_y, E_z]

for i in range(3):
    ax = axes2[i]
    data = comp_data[i][:, mid_y, :]
    vmax = np.max(np.abs(data))
    im = ax.imshow(data.T, cmap='RdBu_r', origin='lower', aspect='auto',
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f'{components[i]} Component (XZ Plane)')
    ax.set_xlabel('X (grid points)')
    ax.set_ylabel('Z (grid points)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Field (V/m)')

plt.tight_layout()
plt.savefig('fdtdx_components.png', dpi=120, bbox_inches='tight')
print("✓ Saved: fdtdx_components.png")

# Power flow
fig3, ax = plt.subplots(figsize=(10, 8))

# Compute Poynting vector (E × H) - simplified
S_z = E_x * H_mag * 0.5  # power flow approximation
Poynting = S_z[:, mid_y, :]

im = ax.imshow(Poynting.T, cmap='hot', origin='lower', aspect='auto')
ax.set_title('Poynting Vector (Power Flow) - Z Component', fontsize=12, fontweight='bold')
ax.set_xlabel('X (grid points)')
ax.set_ylabel('Z (grid points)')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Power Density (W/m²)')

plt.tight_layout()
plt.savefig('fdtdx_poynting.png', dpi=120, bbox_inches='tight')
print("✓ Saved: fdtdx_poynting.png")

print()
print("=" * 60)
print("✓ Done! Generated visualization files:")
print("  - fdtdx_results.png (field overview)")
print("  - fdtdx_components.png (E-field components)")
print("  - fdtdx_poynting.png (power flow)")
print("=" * 60)
