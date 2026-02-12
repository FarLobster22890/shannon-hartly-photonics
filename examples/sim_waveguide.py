#!/usr/bin/env python3
"""
FDTDX Minimal Working Example
"""
import jax
import jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")

import fdtdx

print("=" * 70)
print("FDTDX: Minimal Example")
print("=" * 70)
print()

# Config
config = fdtdx.SimulationConfig(
    time=5.0,
    resolution=8,
    backend='cpu',
    dtype=jnp.float32,
)

# Domain
domain = fdtdx.SimulationVolume(
    name="domain",
    partial_grid_shape=(100, 100, 100),
)

print("Domain: 100x100x100 grid")
print()

# Initialize
print("Initializing...")
key = jax.random.PRNGKey(0)

objects, arrays, info, config, meta = fdtdx.place_objects(
    [domain],
    config,
    [],
    key,
)

print(f"E-field shape: {arrays.E.shape}")
print(f"H-field shape: {arrays.H.shape}")
print(f"Data type: {arrays.E.dtype}")
print()

print("✓ FDTDX initialized on CPU!")
print()
print("Arrays available:")
print(f"  E (electric): {arrays.E.shape}")
print(f"  H (magnetic): {arrays.H.shape}")
print(f"  Permittivity inverse: {arrays.inv_permittivities.shape}")
print()
print("Ready to:")
print("  - Add sources")
print("  - Run fdtdx.run_fdtd(arrays, objects, config, key)")
print("  - Extract detector data")
