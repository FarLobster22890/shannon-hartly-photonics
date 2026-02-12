# FDTDX Photonics Research Setup

## Status: ✓ Working

FDTDX is fully installed and verified on CPU. GPU acceleration attempted but blocked by JAX/ROCm version incompatibility (acceptable tradeoff for research).

## Quick Start

### Run a Basic Simulation

```bash
cd ~/Programming/shannon-hartley-photonics
source venv/bin/activate
JAX_PLATFORMS=cpu python3 sim_basic.py
```

Output: E and H field arrays, energy statistics

### Generate Visualizations

```bash
# Fast: Demo plots with synthetic data
python3 demo_plot.py
# Output: fdtdx_results.png, fdtdx_components.png, fdtdx_poynting.png

# Slow: Full simulation + plots
python3 sim_and_plot.py
# (takes 5-10 min on CPU, generates same plots with real data)
```

### Study the API

Read `FDTDX_GUIDE.md` for:
- Core concepts (SimulationConfig, objects, sources, detectors)
- Key parameters and workflows
- Comparison with Meep
- Autodiff + GPU notes

## Files

### Simulations
- `sim_basic.py` — Minimal working example (50×40×50 grid)
- `sim_and_plot.py` — Full sim with visualization
- `sim_waveguide_complete.py` — Example with waveguide + source + detectors
- `test_fdtdx_cpu.py` — Initial test (working)

### Documentation
- `FDTDX_GUIDE.md` — Complete API reference + workflows
- `FDTDX_SETUP.md` — Installation notes

### Visualization
- `demo_plot.py` — Generate plots instantly (no FDTDX run needed)
- `fdtdx_results.png` — E-field, H-field, energy overview
- `fdtdx_components.png` — E-field vector components
- `fdtdx_poynting.png` — Power flow (Poynting vector)

## What Works

✅ FDTDX fully functional on CPU  
✅ JAX + autodiff available (JIT compilation functional)  
✅ Complete FDTD time-stepping pipeline  
✅ Field analysis (magnitude, components, energy, power flow)  
✅ Visualization pipeline complete  
✅ Code in git, ready for research

## Performance Notes

**CPU mode (current):**
- Small grids (50×40×50): ~2 min per 2-period simulation
- JAX JIT compilation on first run
- Autodiff available for optimization loops

**GPU mode (blocked):**
- JAX/ROCm version mismatch (not worth hours of debugging)
- Can revisit when versions stabilize

## Next Steps

1. **Validate against Meep** — Compare results from your existing Meep simulations
2. **Add structures** — Implement waveguides, resonators, photonic crystals
3. **Optimize designs** — Use autodiff for topology optimization
4. **Scale up** — Increase grid resolution when needed

## Resources

- **Official docs:** https://fdtdx.readthedocs.io
- **GitHub:** https://github.com/ymahlau/fdtdx
- **JAX docs:** https://jax.readthedocs.io
- **Your research:** `~/Programming/Shannon-Hartly-Research/` (Meep reference)

## Key Advantages Over Meep

- Automatic differentiation (built-in gradient computation)
- GPU scalability (once version issues resolve)
- Functional programming paradigm (JAX)
- Composable with other JAX code

## Known Limitations

- Full simulations slow on CPU (but functional)
- GPU not accessible (ROCm versioning)
- Complex object positioning needs trial-and-error (API learning curve)
- Fewer examples than Meep (but official docs are solid)

---

**Setup completed:** 2026-02-12 17:33 EST

Your research is unblocked. Start comparing FDTDX vs Meep.
