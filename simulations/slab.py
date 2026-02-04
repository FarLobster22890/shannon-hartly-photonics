import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Var:

resolution = 50 #PPD unit.

sx = 16 #cell size (x )
sy = 8 #cell size in y
cell = mp.Vector3(sx, sy, 0)
dpml = 1.0 #PML
pml_layers = [mp.PML(dpml)]

#Material (Glass-Ish?):

n_slab = 1.5
slab_eps = n_slab**2
slab_thickness = 1.0

slab = mp.Block(
        size=mp.Vector3(slab_thickness, mp.inf, mp.inf),
        center=mp.Vector3(0,0,0),
        material=mp.Medium(epsilon=slab_eps)
)

#Source (Currently: Broadband- Gaussian Pulse)
f0 = 0.15 #center frequency
df = 0.10 #frequenct width

src = mp.Source(
        mp.GaussianSource(frequency=f0, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(-5, 0, 0),
        size=mp.Vector3(0, sy-2*dpml, 0)
)

#Simulation Parameters

sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=[slab],
        sources=[src],
        resolution=resolution
)

#Power Measurement:

nfreq=200
tran_fr = sim.add_flux(f0, df, nfreq, mp.FluxRegion(center=mp.Vector3(5, 0, 0), size=mp.Vector3(0, sy-2*dpml, 0)))
refl_fr = sim.add_flux(f0, df, nfreq, mp.FluxRegion(center=mp.Vector3(-7, 0, 0), size=mp.Vector3(0, sy-2*dpml, 0)))

#Run #1: Slab
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(5, 0, 0), 1e-9))
tran_with = mp.get_fluxes(tran_fr)
refl_with = mp.get_fluxes(refl_fr)
freqs = mp.get_flux_freqs(tran_fr)

#Quicksave field
eps_data = sim.get_array(center=mp.Vector3(), size = cell, component=mp.Dielectric)
ez_data = sim.get_array(center=mp.Vector3(), size = cell, component=mp.Ez)

#Run #2: Empty cell for reference (no slab)

sim.reset_meep()
sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=[],
        sources=[src],
        resolution=resolution
)

tran_fr2 = sim.add_flux(f0, df, nfreq, mp.FluxRegion(center=mp.Vector3(5, 0, 0), size=mp.Vector3(0, sy-2*dpml, 0)))
refl_fr2 = sim.add_flux(f0, df, nfreq, mp.FluxRegion(center=mp.Vector3(-7, 0, 0), size=mp.Vector3(0, sy-2*dpml, 0)))

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(5, 0, 0), 1e-9))
tran_ref = mp.get_fluxes(tran_fr2)
refl_ref = mp.get_fluxes(refl_fr2)

#Normalize:
T = np.array(tran_with) / np.array(tran_ref)
R = -np.array(refl_with) / np.array(tran_ref) #Sign Conversion - Reflected flux is [[[usually]]] negative

#Plot
plt.figure()
plt.plot(freqs, T, label="Transmission")
plt.plot(freqs, R, label="Reflection")
plt.plot(freqs, T+R, label="T+R", linestyle="--")
plt.xlabel("Frequency (1/units)")
plt.ylabel("Power fraction")
plt.legend()
plt.title("Slab transmission/reflection (normalized)")
plt.grid(True)
plt.show()

plt.figure()
plt.imshow(eps_data.T, interpolation="spline36", cmap="gray")
plt.title("Dielectric (epsilon)")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(ez_data.T, interpolation="spline36", cmap="RdBu")
plt.title("Ez field snapshot (with slab)")
plt.colorbar()
plt.show()
