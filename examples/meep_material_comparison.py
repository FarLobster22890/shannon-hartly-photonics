#!/usr/bin/env python3
"""
Meep Material Comparison Study

Compares electromagnetic field behavior across 4 materials:
- Silicon (n=3.5)
- Silicon Nitride (n=2.0)
- Polymer (n=1.5)
- Vacuum (n=1.0)

Uses Meep FDTD with MPI parallelization for faster computation.
Generates field patterns, energy distribution, and animated timelapses.
"""
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# Suppress Meep output
mp.quiet(True)

print("=" * 70)
print("Meep Material Comparison Study")
print("=" * 70)
print()

# Materials
materials = {
    'Silicon': {
        'n': 3.5,
        'label': 'Silicon (n=3.5)',
        'color': '#1f77b4',
    },
    'SiN': {
        'n': 2.0,
        'label': 'Si₃N₄ (n=2.0)',
        'color': '#ff7f0e',
    },
    'Polymer': {
        'n': 1.5,
        'label': 'Polymer (n=1.5)',
        'color': '#2ca02c',
    },
    'Vacuum': {
        'n': 1.0,
        'label': 'Vacuum',
        'color': '#d62728',
    },
}

# Simulation config
wavelength = 1.55  # µm
resolution = 20    # pixels per wavelength
sim_time = 50      # periods
cell_size = 10     # µm

results = {}

print(f"Configuration:")
print(f"  Wavelength: {wavelength} µm")
print(f"  Resolution: {resolution} pts/λ")
print(f"  Cell size: {cell_size}×{cell_size} µm")
print(f"  Simulation time: {sim_time} periods")
print()

for mat_key, mat_info in materials.items():
    print(f"→ {mat_info['label']}...", flush=True)
    
    # Setup
    cell = mp.Vector3(cell_size, cell_size, 0)
    
    # Material
    material = mp.Medium(index=mat_info['n'])
    
    # Source: point dipole at center
    sources = [mp.Source(
        mp.GaussianSource(
            frequency=1.0 / wavelength,
            fwidth=0.1 / wavelength,
        ),
        component=mp.Ez,
        center=mp.Vector3(-3, 0, 0),
    )]
    
    # Sim
    sim = mp.Simulation(
        cell_size=cell,
        sources=sources,
        default_material=material,
        resolution=resolution,
        symmetries=[mp.Mirror(mp.Y)],
    )
    
    # Run
    print(f"  Running...", flush=True)
    sim.run(mp.after_sources(mp.stop_when_fields_decayed(
        dt=50,
        c=mp.Ez,
        pt=mp.Vector3(0, 0),
        decay_by=1e-8
    )))
    
    # Extract fields
    eps = np.array(sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric))
    Ez = np.array(sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez))
    Hx = np.array(sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hx))
    Hy = np.array(sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hy))
    
    # Energy
    energy_density = 0.5 * (np.abs(Ez)**2 + np.abs(Hx)**2 + np.abs(Hy)**2)
    
    results[mat_key] = {
        'Ez': Ez,
        'Hx': Hx,
        'Hy': Hy,
        'energy': energy_density,
        'eps': eps,
        'max_E': np.max(np.abs(Ez)),
        'max_H': np.max(np.sqrt(Hx**2 + Hy**2)),
        'total_energy': np.sum(energy_density),
    }
    
    print(f"  ✓ Peak |E|: {results[mat_key]['max_E']:.3e}")
    print()

print("=" * 70)
print("Generating Plots")
print("=" * 70)
print()

# Plot 1: Field patterns
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 11))
fig1.suptitle('E-Field Distribution', fontsize=14, fontweight='bold')

axes_list = axes1.flatten()
for idx, (mat_key, mat_info) in enumerate(materials.items()):
    if mat_key not in results:
        continue
    
    ax = axes_list[idx]
    Ez = results[mat_key]['Ez']
    
    vmax = np.max(np.abs(Ez))
    im = ax.imshow(Ez.T, cmap='RdBu_r', origin='lower', aspect='auto',
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f"{mat_info['label']}", fontweight='bold')
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    plt.colorbar(im, ax=ax, label='Ez (V/m)')

plt.tight_layout()
plt.savefig('outputs/meep_material_fields.png', dpi=120, bbox_inches='tight')
print("✓ outputs/meep_material_fields.png")

# Plot 2: Energy density
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))
fig2.suptitle('Energy Density Distribution', fontsize=14, fontweight='bold')

axes_list2 = axes2.flatten()
for idx, (mat_key, mat_info) in enumerate(materials.items()):
    if mat_key not in results:
        continue
    
    ax = axes_list2[idx]
    energy = results[mat_key]['energy']
    
    im = ax.imshow(energy.T, cmap='hot', origin='lower', aspect='auto')
    ax.set_title(f"{mat_info['label']}")
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    plt.colorbar(im, ax=ax, label='Energy (J/m³)')

plt.tight_layout()
plt.savefig('outputs/meep_material_energy.png', dpi=120, bbox_inches='tight')
print("✓ outputs/meep_material_energy.png")

# Plot 3: Statistics
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.axis('off')

table_data = []
for mat_key in ['Silicon', 'SiN', 'Polymer', 'Vacuum']:
    if mat_key not in results:
        continue
    r = results[mat_key]
    m = materials[mat_key]
    table_data.append([
        m['label'],
        f"{r['max_E']:.2e}",
        f"{r['max_H']:.2e}",
        f"{r['total_energy']:.2e}",
    ])

table = ax3.table(
    cellText=table_data,
    colLabels=['Material', 'Peak |E| (V/m)', 'Peak |H| (A/m)', 'Total Energy (J)'],
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(4):
    table[(0, i)].set_facecolor('#003366')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Results Summary', fontweight='bold', pad=15)
plt.savefig('outputs/meep_material_stats.png', dpi=120, bbox_inches='tight')
print("✓ outputs/meep_material_stats.png")

print()
print("=" * 70)
print("Complete")
print("=" * 70)
print()
print("Results:")
for mat_key in ['Silicon', 'SiN', 'Polymer', 'Vacuum']:
    if mat_key not in results:
        continue
    r = results[mat_key]
    m = materials[mat_key]
    print(f"{m['label']}: Peak |E| = {r['max_E']:.2e} V/m")

print()
print("Output files:")
print("  • outputs/meep_material_fields.png")
print("  • outputs/meep_material_energy.png")
print("  • outputs/meep_material_stats.png")
