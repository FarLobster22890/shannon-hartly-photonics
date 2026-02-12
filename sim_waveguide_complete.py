#!/usr/bin/env python3
"""
FDTDX: Complete Waveguide Simulation
Silicon waveguide with source, detectors, and analysis
Based on official FDTDX API
"""
import jax
import jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")

import fdtdx
import numpy as np

print("=" * 70)
print("FDTDX: Silicon Waveguide Simulation")
print("=" * 70)
print()

# ===== SIMULATION CONFIG =====
wavelength = 1.55  # telecom band (µm)
resolution = 12    # points per wavelength
sim_time = 20.0    # periods

config = fdtdx.SimulationConfig(
    time=sim_time,
    resolution=resolution,
    backend='cpu',
    dtype=jnp.float32,
)

print(f"Configuration:")
print(f"  Wavelength: {wavelength} µm")
print(f"  Resolution: {resolution} pts/wavelength")
print(f"  Simulation time: {sim_time} periods")
print()

# ===== DEFINE OBJECTS =====
# Background domain (REQUIRED)
domain = fdtdx.SimulationVolume(
    name="domain",
    partial_real_shape=(8.0, 4.0, 8.0),  # 8µm x 4µm x 8µm
)

# Silicon material (n=3.5 at 1.55µm)
silicon = fdtdx.Material(permittivity=3.5**2)

# Waveguide: standard SOI geometry
# 500nm wide, 220nm tall, 3µm long (centered at z=0)
waveguide = fdtdx.UniformMaterialObject(
    name="waveguide",
    material=silicon,
    partial_real_shape=(0.5, 0.22, 3.0),
    partial_real_position=(0, 0, 0),
)

print("Objects:")
print(f"  Domain: 8µm x 4µm x 8µm")
print(f"  Waveguide (Si): 500nm x 220nm x 3µm (centered)")
print()

# ===== DEFINE SOURCE =====
# Single-frequency Gaussian plane wave
# Excites the fundamental TE mode of the waveguide
temporal_profile = fdtdx.SingleFrequencyProfile(
    num_startup_periods=3,  # ramp-up to steady state
)

source = fdtdx.GaussianPlaneSource(
    name="input_source",
    temporal_profile=temporal_profile,
    partial_real_position=(0, 0, -2.0),  # 2µm before waveguide entrance
    direction='+',  # propagate in +z direction
    fixed_E_polarization_vector=(1, 0, 0),  # TE mode (E along x)
    radius=0.8,  # 800nm beam radius (wider than waveguide)
)

print("Source:")
print(f"  Type: Single-frequency Gaussian plane wave")
print(f"  Position: z = -2µm (before waveguide)")
print(f"  Polarization: x-linear (TE mode)")
print(f"  Beam radius: 800nm")
print()

# ===== DEFINE DETECTORS =====
# Monitor fields at waveguide exit
field_detector = fdtdx.FieldDetector(
    name="field_out",
    partial_real_position=(0, 0, 2.0),  # z = +2µm (after waveguide)
)

# Monitor power transmission
flux_detector = fdtdx.PoyntingFluxDetector(
    name="power_out",
    partial_real_position=(0, 0, 2.0),
)

print("Detectors:")
print(f"  Field monitor at z = +2µm")
print(f"  Poynting flux monitor at z = +2µm")
print()

# ===== INITIALIZE SIMULATION =====
print("Initializing simulation...")
key = jax.random.PRNGKey(42)

try:
    objects, arrays, info, config, meta = fdtdx.place_objects(
        [domain, waveguide, source, field_detector, flux_detector],
        config,
        [],  # constraints
        key,
    )
    
    print(f"✓ Initialized")
    print(f"  Grid shape: {arrays.E.shape}")
    print(f"  Total grid points: {int(np.prod(arrays.E.shape[1:]))}")
    print()
    
    # ===== RUN SIMULATION =====
    print("Running FDTD simulation...")
    print("(This may take 30-60 seconds for a 20-period simulation)")
    print()
    
    result_key, final_arrays = fdtdx.run_fdtd(
        arrays,
        objects,
        config,
        key,
    )
    
    print("✓ Simulation complete!")
    print()
    
    # ===== ANALYSIS =====
    print("Results:")
    print(f"  Max E-field: {float(jnp.max(jnp.abs(final_arrays.E))):.6e} V/m")
    print(f"  Max H-field: {float(jnp.max(jnp.abs(final_arrays.H))):.6e} A/m")
    print()
    
    # Compute energy
    energy = fdtdx.compute_energy(
        final_arrays.E,
        final_arrays.H,
        final_arrays.inv_permittivities,
        final_arrays.inv_permeabilities,
    )
    print(f"  Total energy: {float(jnp.sum(energy)):.6e} J")
    print()
    
    print("✓ FDTDX waveguide simulation successful!")
    print()
    print("Next steps:")
    print("  1. Extract detector data (field_out, power_out)")
    print("  2. Visualize field patterns")
    print("  3. Calculate transmission efficiency")
    print("  4. Compare with Meep validation")
    
except ValueError as e:
    print(f"Object placement error: {e}")
    print()
    print("Tip: Ensure all objects fit within domain and constraints are valid")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
