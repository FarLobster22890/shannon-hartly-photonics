#!/usr/bin/env python3
"""
Meep Material Comparison Study with Animation

Compares electromagnetic field behavior across 4 materials:
- Silicon (n=3.5)
- Silicon Nitride (n=2.0)
- Polymer (n=1.5)
- Vacuum (n=1.0)

Generates:
- Static field patterns and energy density plots
- MP4 animations showing field evolution over time
- Statistics summary

Uses Meep FDTD with MPI parallelization.
"""
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

print("=" * 70)
print("Meep Material Comparison Study with Animation")
print("=" * 70)
print()

# Materials
materials = {
    'Silicon': {'n': 3.5, 'label': 'Silicon (n=3.5)'},
    'SiN': {'n': 2.0, 'label': 'Si₃N₄ (n=2.0)'},
    'Polymer': {'n': 1.5, 'label': 'Polymer (n=1.5)'},
    'Vacuum': {'n': 1.0, 'label': 'Vacuum'},
}

wavelength = 1.55
resolution = 40  # High resolution for accuracy
cell_size = 20   # Larger domain
sim_duration = 50  # Longer simulation

results = {}
frames_dict = {}

print(f"Configuration:")
print(f"  Wavelength: {wavelength} µm")
print(f"  Resolution: {resolution} pts/λ (high accuracy)")
print(f"  Cell: {cell_size}×{cell_size} µm (large domain)")
print(f"  Simulation time: {sim_duration} periods (extended)")
print(f"  Grid points: ~{int(cell_size * resolution)**2 * 4:,} cells per material")
print()

for mat_key, mat_info in materials.items():
    print(f"→ {mat_info['label']}...", flush=True)
    
    cell = mp.Vector3(cell_size, cell_size, 0)
    material = mp.Medium(index=mat_info['n'])
    
    sources = [mp.Source(
        mp.GaussianSource(
            frequency=1.0 / wavelength,
            fwidth=0.1 / wavelength,
        ),
        component=mp.Ez,
        center=mp.Vector3(-3, 0, 0),
    )]
    
    sim = mp.Simulation(
        cell_size=cell,
        sources=sources,
        default_material=material,
        resolution=resolution,
        symmetries=[mp.Mirror(mp.Y)],
    )
    
    # Capture frames during simulation
    frames = []
    frame_times = []
    
    def save_frame(sim):
        Ez = np.array(sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez))
        frames.append(Ez.copy())
        frame_times.append(sim.meep_time())
    
    # Run and capture frames every 0.5 time units
    print(f"  Recording frames...", flush=True)
    sim.run(
        mp.at_every(0.5, save_frame),
        until=sim_duration
    )
    
    # Get final state
    Ez_final = np.array(sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez))
    Hx = np.array(sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hx))
    Hy = np.array(sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hy))
    energy_density = 0.5 * (np.abs(Ez_final)**2 + np.abs(Hx)**2 + np.abs(Hy)**2)
    
    results[mat_key] = {
        'Ez': Ez_final,
        'energy': energy_density,
        'max_E': np.max(np.abs(Ez_final)),
        'max_H': np.max(np.sqrt(Hx**2 + Hy**2)),
        'total_energy': np.sum(energy_density),
    }
    
    frames_dict[mat_key] = frames
    
    print(f"  ✓ {len(frames)} frames captured, Peak |E|: {results[mat_key]['max_E']:.3e}")

print()
print("=" * 70)
print("Generating Visualizations")
print("=" * 70)
print()

# Plot 1: Static field patterns
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 11))
fig1.suptitle('E-Field Distribution (Final State)', fontsize=14, fontweight='bold')

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

# Plot 3: Statistics table
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

# Create animations
print()
print("Creating MP4 animations...", flush=True)

for mat_key, mat_info in materials.items():
    if mat_key not in frames_dict or len(frames_dict[mat_key]) < 2:
        continue
    
    frames = frames_dict[mat_key]
    print(f"  Animating {mat_info['label']}...", flush=True)
    
    # Normalize across all frames
    all_data = np.concatenate([f.flatten() for f in frames])
    vmax = np.max(np.abs(all_data))
    
    fig, ax = plt.subplots(figsize=(8, 7))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    
    im = ax.imshow(frames[0].T, cmap='RdBu_r', origin='lower', aspect='auto', norm=norm)
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    cbar = plt.colorbar(im, ax=ax, label='Ez (V/m)')
    title = ax.set_title(f'{mat_info["label"]} - Frame 0/{len(frames)}', fontsize=11, fontweight='bold')
    
    def animate(frame_num):
        im.set_array(frames[frame_num].T)
        title.set_text(f'{mat_info["label"]} - Frame {frame_num}/{len(frames)}')
        return [im, title]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=100, blit=True, repeat=True)
    
    filename = f'outputs/meep_{mat_key.lower()}_animation.mp4'
    anim.save(filename, writer='ffmpeg', fps=10, dpi=80)
    plt.close(fig)
    
    print(f"    ✓ {filename}")

print()
print("=" * 70)
print("Complete")
print("=" * 70)
print()
print("Output files:")
print("  Static plots:")
print("    • outputs/meep_material_fields.png")
print("    • outputs/meep_material_energy.png")
print("    • outputs/meep_material_stats.png")
print()
print("  Animations:")
for mat_key in ['Silicon', 'SiN', 'Polymer', 'Vacuum']:
    print(f"    • outputs/meep_{mat_key.lower()}_animation.mp4")
