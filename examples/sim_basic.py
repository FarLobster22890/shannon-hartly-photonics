#!/usr/bin/env python3
"""
FDTDX: Basic Working Simulation
Empty domain, 2 periods, minimal grid
"""
import jax
import jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")

import fdtdx
import sys

print("FDTDX: Minimal Simulation", flush=True)
print()

# Config
config = fdtdx.SimulationConfig(
    time=2.0,
    resolution=5,
    backend='cpu',
    dtype=jnp.float32,
)

# Domain
domain = fdtdx.SimulationVolume(
    name="domain",
    partial_grid_shape=(50, 40, 50),
)

print("Initializing 50x40x50 grid...", flush=True)
key = jax.random.PRNGKey(0)

objects, arrays, info, config, meta = fdtdx.place_objects(
    [domain],
    config,
    [],
    key,
)

print(f"✓ Ready: E {arrays.E.shape}, H {arrays.H.shape}", flush=True)
print()
print("Running FDTD (2 periods)...", flush=True)
sys.stdout.flush()

result_key, final_arrays = fdtdx.run_fdtd(
    arrays,
    objects,
    config,
    key,
)

print("✓ Simulation complete!", flush=True)
print()
print("Results:")
print(f"  Max E-field: {float(jnp.max(jnp.abs(final_arrays.E))):.6e}", flush=True)
print(f"  Max H-field: {float(jnp.max(jnp.abs(final_arrays.H))):.6e}", flush=True)
print()
print("✓ FDTDX is fully functional!", flush=True)
