#!/usr/bin/env python3
"""
FDTDX: Simulation + Visualization
"""
import sys
import jax
import jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")

import fdtdx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("FDTDX Simulation + Visualization", flush=True)
print("=" * 60, flush=True)

# Config
sim_time = 3.0
resolution = 6

print(f"Time: {sim_time} periods, Resolution: {resolution} pts/λ", flush=True)

config = fdtdx.SimulationConfig(
    time=sim_time,
    resolution=resolution,
    backend='cpu',
    dtype=jnp.float32,
)

# Domain
print("Setting up domain...", flush=True)
domain = fdtdx.SimulationVolume(
    name="domain",
    partial_grid_shape=(60, 50, 60),
)

# Initialize
print("Initializing...", flush=True)
sys.stdout.flush()

key = jax.random.PRNGKey(0)
objects, arrays, info, config, meta = fdtdx.place_objects(
    [domain],
    config,
    [],
    key,
)

print(f"Grid: {arrays.E.shape}", flush=True)
print("Running FDTD...", flush=True)
sys.stdout.flush()

# Run
result_key, final_arrays = fdtdx.run_fdtd(arrays, objects, config, key)

print("Computing fields...", flush=True)
sys.stdout.flush()

# Analyze
E_mag = jnp.sqrt(jnp.sum(final_arrays.E**2, axis=0))
H_mag = jnp.sqrt(jnp.sum(final_arrays.H**2, axis=0))

print(f"Max |E|: {float(jnp.max(E_mag)):.4e}", flush=True)
print(f"Max |H|: {float(jnp.max(H_mag)):.4e}", flush=True)

# Plot
print("Plotting...", flush=True)
sys.stdout.flush()

mid_y = E_mag.shape[1] // 2
E_xy = E_mag[:, :, mid_y]
E_xz = E_mag[:, mid_y, :]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
im = ax.imshow(E_xy.T, cmap='viridis', origin='lower')
ax.set_title('E-field (XY)')
plt.colorbar(im, ax=ax)

ax = axes[1]
im = ax.imshow(E_xz.T, cmap='viridis', origin='lower')
ax.set_title('E-field (XZ)')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('fdtdx_results.png', dpi=100)

print("✓ Saved: fdtdx_results.png", flush=True)
print("Done!", flush=True)
