#!/usr/bin/env python3
"""
FDTDX test: CPU mode with correct API
"""
import jax
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import fdtdx

print("=" * 60)
print("FDTDX Test: CPU-Accelerated FDTD Simulation")
print("=" * 60)
print()

# Simulation parameters
wavelength = 1.55  # telecom band (microns)
resolution = 20  # points per wavelength (fine = expensive, coarse = fast)
sim_time = 100.0  # simulation time in periods

print(f"Wavelength: {wavelength} µm")
print(f"Resolution: {resolution} points/wavelength")
print(f"Simulation time: {sim_time} periods")
print()

# Create simulation config
config = fdtdx.SimulationConfig(
    time=sim_time,
    resolution=resolution,
    backend='cpu',  # Use CPU for now (no ROCm plugin needed)
    dtype=jnp.float32,
)

print("SimulationConfig created:")
print(f"  Time: {config.time}")
print(f"  Resolution: {config.resolution}")
print(f"  Backend: {config.backend}")
print()

# Create a simple material (silicon)
silicon = fdtdx.Material(
    permittivity=3.5**2,  # n=3.5, ε=n²
)

print(f"Silicon material: ε = {silicon.permittivity}")
print()

# Create a simple waveguide object
waveguide = fdtdx.UniformMaterialObject(
    name="waveguide",
    material=silicon,
    partial_real_shape=(0.5, 0.2, 10.0),  # width, height, length (microns)
    partial_real_position=(0, 0, 0),
)

print("Waveguide created")
print()

# Create a detector
field_det = fdtdx.FieldDetector(
    name="field_monitor",
)

print("Detector created")
print()

print("✓ FDTDX is ready on CPU!")
print()
print("Next steps:")
print("  1. Define a source (GaussianPlaneSource, ModePlaneSource, etc.)")
print("  2. Add objects and detectors to lists")
print("  3. Create SimulationState")
print("  4. Run with fdtdx.run_fdtd()")
print()
