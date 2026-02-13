# Shannon-Hartley Photonics Research

Research into integrated photonics combining computational electromagnetics (FDTD simulation) with information theory (Shannon-Hartley capacity analysis) and optimization.

## Overview

This project bridges two domains:

1. **Electromagnetic Simulation** — Using FDTDX (JAX-based FDTD) and Meep to extract frequency-dependent transmission |H(f)|² from photonic structures
2. **Information Theory** — Computing channel capacity using the Shannon-Hartley theorem and optimizing power allocation via water-filling
3. **Design Optimization** — Leveraging automatic differentiation (autodiff) to optimize photonic structures for capacity and efficiency

Rather than assuming idealized channel models, the frequency response and field distributions are derived directly from first-principles electromagnetic simulation.

## Quick Start

### For FDTDX (GPU-accelerated FDTD simulator)

```bash
cd ~/Programming/shannon-hartley-photonics
source venv/bin/activate

# Fast visualization (instant, no simulation)
python3 examples/demo_plot.py

# Run a real FDTD simulation (~2 min on CPU)
python3 examples/sim_basic.py

# Read the guide
cat docs/README_FDTDX.md
```

See [`docs/README_FDTDX.md`](docs/README_FDTDX.md) for complete FDTDX usage.

### For Shannon-Hartley Analysis (Meep reference)

```bash
# Water-filling optimization demo
python3 analysis/shanhartcalcV2.py

# Full FDTD → capacity pipeline
python3 analysis/hf_pipeline.py
```

## Project Structure

```
shannon-hartley-photonics/
├── README.md                           # This file

├── docs/                               # Documentation
│   ├── README_FDTDX.md                # FDTDX quick-start guide ⭐ START HERE
│   ├── FDTDX_GUIDE.md                 # Complete FDTDX API reference
│   └── FDTDX_SETUP.md                 # Installation & setup notes

├── examples/                           # Simulation scripts
│   ├── demo_plot.py                   # Instant visualizations
│   ├── sim_basic.py                   # Minimal FDTDX example
│   ├── sim_and_plot.py                # Full FDTDX + visualization
│   ├── sim_waveguide_complete.py      # Waveguide with sources
│   ├── test_fdtdx_cpu.py              # Verification
│   └── ...

├── outputs/                            # Simulation results
│   ├── fdtdx_results.png              # E/H field visualization
│   ├── fdtdx_components.png           # Field components
│   └── fdtdx_poynting.png             # Power flow

├── analysis/                           # Shannon-Hartley analysis
│   ├── hf_pipeline.py                 # FDTD → capacity pipeline
│   ├── shanhartcalc.py                # Basic calculator
│   └── shanhartcalcV2.py              # Water-filling optimization

├── simulations/                        # Meep reference simulations
│   ├── slab.py                        # Dielectric slab
│   └── wg_straight.py                 # Silicon waveguide (1.55 µm)

├── data/                               # Research data
│   └── meep_channel_physical.npz      # Simulation outputs

├── figures/                            # Publication figures
│   ├── H2_vs_frequency_THz.png
│   ├── capacity_vs_power.png
│   └── ...

├── venv/                               # Python virtual environment
└── .git/                               # Version control
```

## Key Capabilities

### FDTDX Electromagnetic Simulation

- **GPU Acceleration:** JAX-based, supports CUDA/ROCm (CPU functional)
- **Automatic Differentiation:** Built-in gradients for topology optimization
- **Field Analysis:** E-field, H-field, Poynting flux, energy, modes
- **Visualization:** Field slices, power flow, resonances

### Shannon-Hartley Channel Analysis

- **Frequency-Selective Channels:** Capacity calculation for non-flat responses
- **Water-Filling Optimization:** Optimal power allocation under constraints
- **Interference Modeling:** Support for multiple resonances and notches

### Research Workflow

1. Design photonic structure
2. Simulate with FDTDX (or validate with Meep)
3. Extract frequency response |H(f)|²
4. Compute Shannon capacity with water-filling
5. Optimize structure using autodiff
6. Repeat

## Theory

### Shannon-Hartley for Frequency-Selective Channels

For a channel with frequency-dependent gain |H(f)|² and noise PSD N(f):

```
C = ∫ log₂(1 + |H(f)|² · S(f) / N(f)) df
```

Subject to total power constraint: ∫S(f)df = P

### Water-Filling Solution

Optimal power allocation:

```
S(f) = max(0, μ - N(f)/|H(f)|²)
```

Power "fills" favorable frequencies, avoiding poor channels.

## Current Results

- **Baseline Capacity:** 1.23e9 bits/s (uniform power)
- **Water-Filled Capacity:** 1.41e9 bits/s (optimized power)
- **Capacity Gain:** 1.15× improvement

Resonator-induced channel responses show how microring geometries create frequency-selective transmission suitable for photonic filters.

## Technology Stack

### FDTDX (New)
- Python 3.12+
- JAX (automatic differentiation + JIT)
- NumPy, Matplotlib
- Status: ✅ Working (CPU mode verified)

### Meep (Reference)
- Python 3.8+
- MEEP/PyMEEP (MIT FDTD solver)
- NumPy, Matplotlib
- Status: ✅ Existing simulations functional

### Analysis
- Python 3.8+
- SciPy (optimization)
- Pandas (data handling)

## Installation

### FDTDX (Recommended for new work)

```bash
cd ~/Programming/shannon-hartley-photonics
source venv/bin/activate  # Already set up
python3 examples/sim_basic.py
```

See [`docs/FDTDX_SETUP.md`](docs/FDTDX_SETUP.md) for detailed setup.

### Meep (For reference/validation)

```bash
conda install -c conda-forge pymeep
# or
pip install meep
```

## Usage Examples

### Run FDTDX Simulation

```bash
python3 examples/sim_basic.py              # Minimal
python3 examples/sim_waveguide_complete.py # With source + detectors
python3 examples/sim_and_plot.py           # Full simulation + plots
```

### Visualize Results

```bash
python3 examples/demo_plot.py              # Instant (synthetic data)
# Output: fdtdx_results.png, fdtdx_components.png, fdtdx_poynting.png
```

### Analyze Channel Capacity

```bash
python3 analysis/shanhartcalcV2.py
# Output: Baseline capacity, water-filled capacity, gain factor
```

## Future Work

- [ ] Integrate FDTDX autodiff into optimization loop
- [ ] Topology optimization for max-capacity photonic structures
- [ ] Multi-resonator geometries (coupled rings, photonic crystals)
- [ ] WDM (wavelength-division multiplexing) optimization
- [ ] GPU acceleration (once ROCm versioning stabilizes)
- [ ] Comparison: FDTDX vs Meep validation across designs

## Development

### Make Changes

```bash
cd ~/Programming/shannon-hartley-photonics
source venv/bin/activate

# Create a new simulation
cp examples/sim_basic.py examples/my_experiment.py
vim examples/my_experiment.py
python3 examples/my_experiment.py
```

### Version Control

```bash
git add .
git commit -m "Add waveguide resonator optimization"
git push origin main
```

## References

- **FDTDX:** https://fdtdx.readthedocs.io | https://github.com/ymahlau/fdtdx
- **JAX:** https://jax.readthedocs.io
- **Meep:** https://meep.readthedocs.io
- **Shannon, C.E. (1948):** "A Mathematical Theory of Communication"
- **Boyd, S. & Vandenberghe, L. (2004):** "Convex Optimization" (water-filling)

## Status

✅ FDTDX installed and verified (CPU)  
✅ Visualization pipeline complete  
✅ Documentation comprehensive  
✅ Example simulations functional  
✅ Meep reference simulations available  
✅ Repository clean and organized  

**Research is unblocked.** Ready for photonics design and optimization.

## License

MIT

## Author

Connor — Ontario, Canada

---

**Last Updated:** 2026-02-12  
**FDTDX Version:** 0.5.0 (CPU mode)  
**Meep Version:** 1.20+
