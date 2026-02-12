# FDTDX API Guide

Based on official documentation: https://fdtdx.readthedocs.io/en/latest/

## Core Concepts

### 1. Simulation Setup

**SimulationConfig** - Control simulation parameters
```python
config = fdtdx.SimulationConfig(
    time=20.0,              # simulation time in periods
    resolution=12,          # points per wavelength
    backend='cpu',          # or 'gpu'
    dtype=jnp.float32,      # float32 or float64
)
```

### 2. Objects

**SimulationVolume** (REQUIRED) - Background domain
```python
domain = fdtdx.SimulationVolume(
    name="domain",
    partial_real_shape=(8.0, 4.0, 8.0),  # x, y, z size in µm
)
```

**UniformMaterialObject** - Uniform material blocks
```python
silicon = fdtdx.Material(permittivity=3.5**2)  # n=3.5

waveguide = fdtdx.UniformMaterialObject(
    name="waveguide",
    material=silicon,
    partial_real_shape=(0.5, 0.22, 3.0),     # width, height, length
    partial_real_position=(0, 0, 0),         # centered at origin
)
```

Other shapes: `Cylinder`, `Sphere`, `ExtrudedPolygon`

### 3. Sources

**GaussianPlaneSource** - Gaussian beam (most common)
```python
temporal_profile = fdtdx.SingleFrequencyProfile(num_startup_periods=3)

source = fdtdx.GaussianPlaneSource(
    name="input",
    temporal_profile=temporal_profile,
    partial_real_position=(0, 0, -2.0),
    direction='+',  # '+' or '-' for ±z (±x, ±y also valid)
    fixed_E_polarization_vector=(1, 0, 0),  # x-polarized
    radius=0.8,  # beam radius in µm
)
```

**UniformPlaneSource** - Uniform plane wave (simpler)

**ModePlaneSource** - Fundamental mode of waveguide

**SingleFrequencyProfile** vs **GaussianPulseProfile**

### 4. Detectors

**FieldDetector** - Monitor E and H fields
```python
field_det = fdtdx.FieldDetector(
    name="field",
    partial_real_position=(0, 0, 2.0),
)
```

**PoyntingFluxDetector** - Monitor power flow (Poynting vector)

**EnergyDetector** - Total electromagnetic energy

**ModeOverlapDetector** - Overlap with a specific waveguide mode

### 5. Initialization

```python
key = jax.random.PRNGKey(0)

objects, arrays, info, config, meta = fdtdx.place_objects(
    [domain, waveguide, source, field_detector],
    config,
    [],  # constraints (optional)
    key,
)

# arrays contains:
# - arrays.E[3, nx, ny, nz]: electric field components
# - arrays.H[3, nx, ny, nz]: magnetic field components
# - arrays.inv_permittivities[nx, ny, nz]: inverse permittivity
```

### 6. Run Simulation

```python
result_key, final_arrays = fdtdx.run_fdtd(
    arrays,
    objects,
    config,
    key,
)
```

Returns:
- `result_key`: updated random key
- `final_arrays`: arrays after simulation (same structure as input)

### 7. Analysis

**Compute Energy**
```python
energy = fdtdx.compute_energy(
    final_arrays.E,
    final_arrays.H,
    final_arrays.inv_permittivities,
    final_arrays.inv_permeabilities,
)
print(f"Total energy: {jnp.sum(energy)}")
```

**Compute Poynting Flux**
```python
flux = fdtdx.compute_poynting_flux(final_arrays.E, final_arrays.H, axis=2)
```

**Normalize Fields**
```python
normalized_E, normalized_H = fdtdx.normalize_by_energy(
    final_arrays.E,
    final_arrays.H,
    final_arrays.inv_permittivities,
    final_arrays.inv_permeabilities,
)
```

### 8. Visualization

**Plot Setup** - Show material distribution
```python
fdtdx.plot_setup(config, objects)
```

**Plot Material**
```python
fdtdx.plot_material(config, arrays)
```

**Plot Field Slice**
```python
fdtdx.plot_field_slice(arrays.E, arrays.H)
```

## Key Parameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| `wavelength` | 1.55 µm | Telecom band |
| `resolution` | 12-20 pts/λ | Higher = more accurate, slower |
| `sim_time` | 10-50 periods | Depends on decay time |
| `domain_size` | 3-10λ | Buffer space around structure |
| `PML_width` | Auto-set | Absorbing boundary |
| `material_permittivity` | 12.25 (Si) | n² for material |

## Coordinate System

- **Real coordinates**: physical size in µm
  - Use `partial_real_shape` and `partial_real_position`
- **Grid coordinates**: simulation grid indices
  - Use `partial_grid_shape` and `partial_grid_position`

Mix both: FDTDX converts based on resolution.

## Common Workflows

### 1. Waveguide Transmission
1. Create waveguide with UniformMaterialObject
2. Add source at entrance (z = -L)
3. Add field detector at exit (z = +L)
4. Run simulation
5. Compare E-field amplitude at detector

### 2. Resonator Design
1. Create cavity (ring or box)
2. Add source at side
3. Add field detector inside cavity
4. Monitor field growth (Q factor)

### 3. Topology Optimization
1. Use Device object (parametric permittivity)
2. Define optimization metric (transmission, resonance)
3. Compute gradients with autodiff
4. Update structure with optimizer

## Autodiff + GPU

FDTDX is built on JAX → automatic differentiation works!

```python
def simulate_and_measure(permittivity_pattern):
    # Define structure based on pattern
    # Run FDTDX
    # Return efficiency
    return efficiency

# Compute gradients
grads = jax.grad(simulate_and_measure)(pattern)

# Optimize
new_pattern = pattern - learning_rate * grads
```

## Comparing with Meep

**Similar:**
- FDTD time-stepping
- PML boundaries
- Field monitors and flux detectors
- Maxwell equation solving

**Advantages (FDTDX):**
- GPU acceleration (when working)
- Automatic differentiation built-in
- JAX integration (functional, composable)
- More intuitive object positioning

**When to use Meep:**
- Validation (mature, battle-tested)
- Complex scenarios (better documentation)
- Reference implementation

**When to use FDTDX:**
- GPU acceleration (if it works)
- Optimization loops (autodiff)
- Research into new designs
- Fast prototyping

## Resources

- Official docs: https://fdtdx.readthedocs.io
- GitHub: https://github.com/ymahlau/fdtdx
- JAX docs: https://jax.readthedocs.io (autodiff, JIT)
- Maxwell equations: Any EM textbook (Griffiths, Jackson)
