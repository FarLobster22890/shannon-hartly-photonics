# Shannon-Hartley Photonics Research

Research into photonic device simulation and channel capacity analysis. Combines FDTD electromagnetic simulation with information theory to understand and optimize integrated photonic structures.

## What This Is

Two parts:

1. **FDTDX Simulations** — Using JAX-based FDTD to model photonic devices, extract field distributions, and compute electromagnetic properties
2. **Shannon-Hartley Analysis** — Taking the simulated frequency response and calculating channel capacity, optimizing power allocation

The goal is to move beyond idealized models and build the frequency response directly from first-principles EM simulation.

## Getting Started

### FDTDX (Current focus)

```bash
cd ~/Programming/shannon-hartley-photonics
source venv/bin/activate
python3 examples/demo_plot.py          # instant plots
python3 examples/sim_basic.py          # run a simulation
```

Read `docs/README_FDTDX.md` for the actual guide.

### Shannon-Hartley (Meep reference)

```bash
python3 analysis/shanhartcalcV2.py     # water-filling optimization
python3 analysis/hf_pipeline.py        # full pipeline
```

## Folder Layout

```
.
├── README.md                        # this file
├── docs/
│   ├── README_FDTDX.md             # FDTDX quickstart
│   ├── FDTDX_GUIDE.md              # API reference
│   └── FDTDX_SETUP.md              # setup notes
├── examples/                        # simulation scripts
│   ├── demo_plot.py                # visualization (no sim)
│   ├── sim_basic.py                # minimal working example
│   ├── sim_and_plot.py             # full sim + plots
│   ├── sim_waveguide_complete.py   # with source/detectors
│   └── test_*.py
├── outputs/                         # results
│   ├── fdtdx_results.png
│   ├── fdtdx_components.png
│   └── fdtdx_poynting.png
├── analysis/                        # capacity calculations
│   ├── hf_pipeline.py
│   ├── shanhartcalc.py
│   └── shanhartcalcV2.py           # water-filling
├── simulations/                     # meep reference sims
│   ├── slab.py
│   └── wg_straight.py
├── data/                            # meep outputs
├── figures/                         # publication plots
└── venv/                            # python env
```

## What's Working

- FDTDX on CPU (verified, takes ~2 min for small grids)
- JAX autodiff (ready for optimization)
- Visualization pipeline (field analysis, plots)
- All simulations and analysis scripts functional

GPU acceleration attempted but blocked by JAX/ROCm version incompatibility. Not worth the time right now.

## Running Simulations

### Visualize (fast)
```bash
python3 examples/demo_plot.py
# generates: fdtdx_results.png, fdtdx_components.png, fdtdx_poynting.png
```

### Basic FDTDX
```bash
python3 examples/sim_basic.py
# 50×40×50 grid, 2 periods, ~30 seconds
```

### Full Simulation with Plots
```bash
python3 examples/sim_and_plot.py
# larger grid, more time stepping, generates output plots
```

### Water-Filling Analysis
```bash
python3 analysis/shanhartcalcV2.py
# baseline vs optimized power allocation
```

## Theory (Quick Reference)

**Shannon Capacity for frequency-selective channels:**
```
C = ∫ log₂(1 + |H(f)|² · S(f) / N(f)) df
```
where S(f) is power spectral density, N(f) is noise, subject to ∫S(f)df = P.

**Water-filling solution:**
```
S(f) = max(0, μ - N(f)/|H(f)|²)
```
Put power where the channel is good, avoid the bad parts.

## Tech

- Python 3.12+
- JAX (autodiff + JIT compilation)
- NumPy, Matplotlib
- MEEP (validation/reference)
- Meep is optional — FDTDX handles the main work

## Development

Make changes, test, commit:
```bash
cp examples/sim_basic.py examples/my_experiment.py
# edit and run
git add .
git commit -m "description"
git push origin main
```

## References

- FDTDX: https://github.com/ymahlau/fdtdx
- JAX: https://jax.readthedocs.io
- Meep: https://meep.readthedocs.io
- Shannon (1948): "A Mathematical Theory of Communication"

## Status

FDTDX working. Ready for design iteration.
