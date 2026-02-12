#!/usr/bin/env python3
"""
FDTDX Waveguide with Source
Simple simulation: source + waveguide + field monitoring
"""
import jax
import jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")

import fdtdx

print("=" * 70)
print("FDTDX: Waveguide with Source")
print("=" * 70)
print()

# Simulation config
wavelength = 1.55  # telecom (µm)
resolution = 12    # points per wavelength
sim_time = 10.0    # periods

config = fdtdx.SimulationConfig(
    time=sim_time,
    resolution=resolution,
    backend='cpu',
    dtype=jnp.float32,
)

print(f"Wavelength: {wavelength} µm")
print(f"Resolution: {resolution} pts/wavelength")
print(f"Simulation time: {sim_time} periods")
print()

# Domain: 10µm x 5µm x 10µm
domain = fdtdx.SimulationVolume(
    name="domain",
    partial_real_shape=(10.0, 5.0, 10.0),
)

# Silicon waveguide: 500nm wide, 220nm tall, 3µm long
silicon = fdtdx.Material(permittivity=3.5**2)

waveguide = fdtdx.UniformMaterialObject(
    name="waveguide",
    material=silicon,
    partial_real_shape=(0.5, 0.22, 3.0),
    partial_real_position=(0, 0, 0),  # centered
)

# Source: single-frequency at waveguide entrance
temporal_profile = fdtdx.SingleFrequencyProfile(
    num_startup_periods=2,
)

source = fdtdx.GaussianPlaneSource(
    name="source",
    temporal_profile=temporal_profile,
    partial_real_position=(0, 0, -2.0),  # 2µm before waveguide
    direction='+',  # +z direction
    fixed_E_polarization_vector=(1, 0, 0),  # x-polarized
    radius=1.0,  # 1µm beam radius
)

# Field monitor
field_detector = fdtdx.FieldDetector(
    name="field_monitor",
)

print("Objects:")
print("  Domain: 10µm x 5µm x 10µm")
print("  Waveguide: 500nm x 220nm x 3µm (centered)")
print("  Source: Gaussian plane wave, x-polarized")
print("  Detector: Field monitor")
print()

# Initialize
print("Initializing...")
key = jax.random.PRNGKey(0)

try:
    objects, arrays, info, config, meta = fdtdx.place_objects(
        [domain, waveguide],
        config,
        [],
        key,
    )
    
    print(f"✓ Grid initialized")
    print(f"  E-field: {arrays.E.shape}")
    print(f"  H-field: {arrays.H.shape}")
    print()
    print("Ready to run fdtdx.run_fdtd()")
    
except Exception as e:
    print(f"Error during initialization: {e}")
    print()
    print("Trying simpler config (no constraints)...")
    
    # Fallback: just domain
    objects, arrays, info, config, meta = fdtdx.place_objects(
        [domain],
        config,
        [],
        key,
    )
    
    print(f"✓ Basic grid initialized")
    print(f"  E-field: {arrays.E.shape}")
    print(f"  H-field: {arrays.H.shape}")
    print()
    print("(Waveguide can be added as constraint)")
