import numpy as np
import meep as mp

# ------------------------------------------------------------
# 0) PHYSICAL UNIT SETTING (THIS is what makes it "real units")
# ------------------------------------------------------------
# Choose what 1 Meep length-unit equals in meters.
# Common for photonics: 1 unit = 1 micron.
L0_m = 1.0e-6  # meters per Meep unit (change if you want)

c0 = 299_792_458.0  # m/s

def meep_freq_to_hz(f_meep):
    # Meep uses c=1 units, so f_Hz = f_meep * (c / L0)
    return f_meep * (c0 / L0_m)

def meep_freq_to_thz(f_meep):
    return meep_freq_to_hz(f_meep) / 1e12

def meep_freq_to_wavelength_m(f_meep):
    # wavelength = c / f_Hz = L0 / f_meep
    return (L0_m / f_meep)

def meep_freq_to_wavelength_um(f_meep):
    return meep_freq_to_wavelength_m(f_meep) * 1e6

# ------------------------------------------------------------
# 1) SIM PARAMETERS (Meep units, but interpreted via L0_m)
# ------------------------------------------------------------
resolution = 120          # pixels per Meep unit
dpml = 1.0               # PML thickness (Meep units)
sx, sy = 150, 80          # cell size (Meep units), excluding PML

# Source spectrum in Meep frequency units
fcen = 0.15
df   = 0.10
nfreq = 401

# Materials and geometry (Meep units -> physical via L0_m)
n_core = 3.4
wg_w = 1.0

disk_r = 1.5
disk_offset_y = 1.8

src_x = -0.5*sx + 2.0
in_flux_x  = -2.0
out_flux_x = +5.0

# ------------------------------------------------------------
# 2) GEOMETRY
# ------------------------------------------------------------
def make_geometry(include_resonator: bool):
    wg = mp.Block(
        size=mp.Vector3(mp.inf, wg_w, mp.inf),
        center=mp.Vector3(0, 0),
        material=mp.Medium(index=n_core)
    )
    geom = [wg]

    if include_resonator:
        disk = mp.Cylinder(
            radius=disk_r,
            height=mp.inf,
            center=mp.Vector3(0.0, disk_offset_y),
            material=mp.Medium(index=n_core)
        )
        geom.append(disk)

    return geom

# ------------------------------------------------------------
# 3) RUN ONCE: return freqs + (in_flux_spectrum, out_flux_spectrum)
# ------------------------------------------------------------
def run_transmission(include_resonator: bool):
    cell = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(dpml)]

    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(src_x, 0),
            size=mp.Vector3(0, wg_w)
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=make_geometry(include_resonator),
        sources=sources,
        resolution=resolution,
        split_chunks_evenly=False,
    )

    in_fr  = mp.FluxRegion(center=mp.Vector3(in_flux_x, 0),  size=mp.Vector3(0, wg_w*1.5))
    out_fr = mp.FluxRegion(center=mp.Vector3(out_flux_x, 0), size=mp.Vector3(0, wg_w*1.5))

    in_flux  = sim.add_flux(fcen, df, nfreq, in_fr)
    out_flux = sim.add_flux(fcen, df, nfreq, out_fr)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ez, mp.Vector3(out_flux_x, 0), 1e-9
        )
    )

    freqs = np.array(mp.get_flux_freqs(out_flux))
    in_spec  = np.array(mp.get_fluxes(in_flux))
    out_spec = np.array(mp.get_fluxes(out_flux))

    return freqs, in_spec, out_spec

# ------------------------------------------------------------
# 4) MAIN PIPELINE: baseline + device -> clean |H(f)|^2
# ------------------------------------------------------------
def main():
    # Baseline
    freqs0, in0, out0 = run_transmission(include_resonator=False)
    # Device
    freqs1, in1, out1 = run_transmission(include_resonator=True)

    if not np.allclose(freqs0, freqs1):
        raise RuntimeError("Frequency grids differ between runs. Keep sim settings identical.")

    eps = 1e-30

    # Clean normalization:
    # T0 = out0/in0, T1 = out1/in1, H2 = T1/T0
    T0 = (out0 + eps) / (in0 + eps)
    T1 = (out1 + eps) / (in1 + eps)
    H2 = (T1 + eps) / (T0 + eps)
    H2 = np.clip(H2, 0.0, None)

    # Convert to physical units
    f_meep = freqs0
    f_thz = meep_freq_to_thz(f_meep)
    lam_um = meep_freq_to_wavelength_um(f_meep)

    # Identify resonance-like minima (quick-and-dirty)
    idx_min = int(np.argmin(H2))
    fmin_thz = float(f_thz[idx_min])
    lmin_um = float(lam_um[idx_min])
    h2min = float(H2[idx_min])

    if mp.am_master():
        # Save everything for later use in Shannon/water-filling
        np.savez(
            "meep_channel_physical.npz",
            L0_m=L0_m,
            c0=c0,
            f_meep=f_meep,
            f_thz=f_thz,
            lam_um=lam_um,
            in0=in0, out0=out0, T0=T0,
            in1=in1, out1=out1, T1=T1,
            H2=H2
        )

        print("Saved: meep_channel_physical.npz")
        print(f"Min |H(f)|^2 ~ {h2min:.4g} at {fmin_thz:.4f} THz  (λ ≈ {lmin_um:.4f} µm)")
        print(f"Physical interpretation: 1 Meep length unit = {L0_m*1e6:.3f} µm")

        # ----------------------------
        # Plotting (master only)
        # ----------------------------
        import matplotlib.pyplot as plt

        # (A) Output flux spectra vs frequency (THz)
        plt.figure()
        plt.plot(f_thz, out0, label="Baseline out (no resonator)")
        plt.plot(f_thz, out1, label="Device out (with resonator)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Output Flux (arb. units)")
        plt.title("Output Spectra vs Frequency (Physical Units)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("spectra_out_vs_frequency_THz.png", dpi=200)
        plt.close()

        # (B) Clean normalized transmission |H(f)|^2 vs frequency (THz)
        plt.figure()
        plt.plot(f_thz, H2)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Normalized Transmission |H(f)|^2 (dimensionless)")
        plt.title("Normalized Channel Response vs Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("H2_vs_frequency_THz.png", dpi=200)
        plt.close()

        # (C) Normalized transmission vs wavelength (µm)
        # Wavelength axis is reversed/nonuniform; sort by wavelength for a nice plot.
        order = np.argsort(lam_um)
        plt.figure()
        plt.plot(lam_um[order], H2[order])
        plt.xlabel("Wavelength (µm)")
        plt.ylabel("Normalized Transmission |H(f)|^2 (dimensionless)")
        plt.title("Normalized Channel Response vs Wavelength")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("H2_vs_wavelength_um.png", dpi=200)
        plt.close()

        print("Saved plots:")
        print("  spectra_out_vs_frequency_THz.png")
        print("  H2_vs_frequency_THz.png")
        print("  H2_vs_wavelength_um.png")

if __name__ == "__main__":
    main()

