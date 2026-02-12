# FDTDX GPU-Accelerated FDTD Simulation

## What's Installed

- **FDTDX 0.5.0**: Open-source FDTD simulator built on JAX with GPU acceleration
- **JAX 0.5.0**: Automatic differentiation and JIT compilation
- **ROCm Plugin**: AMD GPU support (requires librccl.so for full GPU utilization)

## Current Status

✅ FDTDX core library installed  
✅ JAX CPU backend functional  
⚠️  ROCm GPU backend: Plugin installed but needs full ROCm runtime

## Setup

```bash
# Activate the virtual environment
cd ~/Programming/shannon-hartley-photonics
source venv/bin/activate

# Test installation
JAX_PLATFORMS=cpu python3 -c "import fdtdx; print('OK')"
```

## Next Steps

### 1. Install ROCm Runtime (Optional but Recommended)
For your RX 5700 XT:

```bash
# Install ROCm developer tools
# (Instructions vary by Linux distro)
# https://rocmdocs.amd.com/en/docs-5.0.0/deploy/linux/index.html

# Once installed:
JAX_PLATFORMS=rocm python3 test_fdtdx.py  # Will use GPU
```

### 2. Learn the API
Key FDTDX components:
- `SimulationVolume`: Define simulation domain
- `SimulationConfig`: Configure FDTD parameters
- `run_fdtd()`: Execute simulation
- Detectors: `FieldDetector`, `PoyntingFluxDetector`, etc.
- Objects: `UniformMaterialObject`, `Cylinder`, `Sphere`, etc.

Documentation: https://fdtdx.readthedocs.io/

### 3. Build Photonic Structures
Examples to explore:
- Straight waveguides
- Resonators (ring, Fabry-Pérot)
- Grating couplers
- Photonic crystals

## Advantages Over Meep

- **Autodiff**: Gradients through the entire simulation (topology optimization)
- **GPU scaling**: Multi-GPU support for massive simulations
- **JAX integration**: Functional programming paradigm
- **Memory efficient**: Time-reversibility for gradient computation

## Integration with Shannon-Hartley Research

FDTDX complements your existing Meep-based research:
- Use Meep for validation and detailed analysis
- Use FDTDX for inverse design and optimization
- Cross-validate results between simulators

## Current Limitations

1. **ROCm setup**: GPU backend needs full runtime installation
2. **Learning curve**: API differs from Meep
3. **Documentation**: Fewer examples than mature projects like Meep

## Files

- `test_fdtdx.py`: Basic example (WIP)
- `venv/`: Python virtual environment with all dependencies
- `FDTDX_SETUP.md`: This file

## References

- FDTDX: https://github.com/ymahlau/fdtdx
- JAX: https://github.com/google/jax
- ROCm: https://rocmdocs.amd.com

---

**Last updated:** Feb 12, 2026  
**Status:** Ready for exploration and development
