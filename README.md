# Shannon-Hartley Photonics Research

Research into photonic device simulation and channel capacity analysis. Combines FDTD electromagnetic simulation (Meep) with information theory to understand and optimize integrated photonic structures.

## What This Is

Two parts:

1. **Meep Simulations** — Using Meep FDTD to model photonic devices, extract field distributions, and compute electromagnetic properties. MPI-parallelized for CPU performance.
2. **Shannon-Hartley Analysis** — Taking the simulated frequency response and calculating channel capacity, optimizing power allocation with water-filling.

The goal is to move beyond idealized models and build the frequency response directly from first-principles EM simulation.

## Getting Started

### Meep (Current focus)

```bash
cd ~/Programming/shannon-hartley-photonics
# Assuming Meep is installed in a conda env:
conda activate <your-meep-env>
python3 examples/meep_material_comparison.py
```

### Shannon-Hartley (Analysis)

```bash
python3 analysis/shanhartcalcV2.py     # water-filling optimization
python3 analysis/hf_pipeline.py        # full pipeline
```

## Folder Layout

```
.
├── README.md                        # this file
├── examples/
│   ├── meep_material_comparison.py  # main material study
│   └── (other scripts as needed)
├── outputs/                         # results
│   ├── meep_material_fields.png
│   ├── meep_material_energy.png
│   └── meep_material_stats.png
├── analysis/                        # capacity calculations
│   ├── hf_pipeline.py
│   ├── shanhartcalc.py
│   └── shanhartcalcV2.py           # water-filling
├── simulations/                     # Meep sims
├── data/                            # outputs
├── figures/                         # publication plots
└── venv/                            # python env (legacy)
```

## What's Working

- Meep FDTD on CPU with MPI parallelization
- Material comparison studies (Si, SiN, Polymer)
- Field analysis and visualization
- Shannon-Hartley capacity calculations
- Water-filling optimization

## Running Simulations

```bash
python3 examples/meep_material_comparison.py
# Outputs 3 PNG files to outputs/
```

## Theory (Quick Reference)

**Shannon Capacity for frequency-selective channels:**
```
C = ∫ log₂(1 + |H(f)|² · S(f) / N(f)) df
```

**Water-filling solution:**
```
S(f) = max(0, μ - N(f)/|H(f)|²)
```

## Tech Stack

- Python 3.10+
- Meep (FDTD simulator with MPI support)
- NumPy, Matplotlib, SciPy
- MPI for parallelization

## Development

Make changes, test, commit:
```bash
python3 examples/meep_material_comparison.py
git add .
git commit -m "description"
git push origin main
```

## References

- Meep: https://meep.readthedocs.io
- MPI: https://www.open-mpi.org/
- Shannon (1948): "A Mathematical Theory of Communication"

## Status

Meep working. Ready for design iteration.
