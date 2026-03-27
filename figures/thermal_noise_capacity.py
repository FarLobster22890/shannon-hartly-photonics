"""
Thermal Noise vs Power: Capacity Analysis
==========================================
Visualizes how increased transmit power leads to heating,
which increases thermal noise, creating diminishing returns
(and eventually capacity degradation) at high power levels.

Physics:
- Johnson-Nyquist noise: N = k_B * T * B
- Heating from absorption: T = T0 + η * P_absorbed
- P_absorbed = α * P_transmit (absorption coefficient)
- Combined: N(P) = k_B * (T0 + η*α*P) * B = N0 + β*P

Shannon-Hartley with power-dependent noise:
  C = B * log2(1 + P / N(P))
  C = B * log2(1 + P / (N0 + β*P))

At high P, this saturates to: C → B * log2(1 + 1/β)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -------------------------
# Physical constants
# -------------------------
k_B = 1.38e-23      # Boltzmann constant (J/K)
T0 = 300            # Ambient temperature (K)
B = 100e9           # Bandwidth (Hz) - 100 GHz for optical

# -------------------------
# Model parameters
# -------------------------
# Absorption: fraction of power converted to heat
alpha = 0.10        # 10% absorption (lossy waveguide or intentionally absorbing)

# Thermal resistance: K per Watt of absorbed power
# (how much temperature rises per watt of heat)
R_th = 5000         # K/W (poor heat sinking - small device, no substrate cooling)

# Derived: temperature coefficient
# T = T0 + R_th * alpha * P = T0 + eta * P
eta = R_th * alpha  # K/W of transmit power

# Base noise power
N0 = k_B * T0 * B   # Watts

# Thermal noise coefficient: β = k_B * η * B
beta = k_B * eta * B

print("=== Thermal Noise Model Parameters ===")
print(f"Ambient temperature: {T0} K")
print(f"Bandwidth: {B/1e9:.1f} GHz")
print(f"Absorption coefficient: {alpha*100:.1f}%")
print(f"Thermal resistance: {R_th} K/W")
print(f"Base noise power N0: {N0*1e12:.3f} pW")
print(f"Noise coefficient β: {beta:.3e} W/W")
print()

# -------------------------
# Power sweep
# -------------------------
P = np.logspace(-6, 0, 500)  # 1 µW to 1 W

# Temperature vs power
T = T0 + eta * P

# Noise power vs power (Johnson-Nyquist)
N = k_B * T * B  # = N0 + beta * P

# SNR
SNR = P / N

# Capacity: standard (constant noise)
C_standard = B * np.log2(1 + P / N0)

# Capacity: with thermal noise increase
C_thermal = B * np.log2(1 + P / N)

# Theoretical limit as P → ∞
C_limit = B * np.log2(1 + 1/beta) if beta > 0 else np.inf

# Find optimal power (where dC/dP ≈ 0 for thermal case)
# Derivative: dC/dP = B/ln(2) * d/dP[ln(1 + P/(N0+βP))]
# Setting to zero and solving... actually it's monotonic but saturates
# Let's find where we get 90% of the limit
idx_90 = np.argmin(np.abs(C_thermal - 0.9 * C_limit))
P_90 = P[idx_90]

print(f"=== Results ===")
print(f"Capacity limit (P→∞): {C_limit/1e9:.2f} Gb/s")
print(f"Power for 90% of limit: {P_90*1e3:.2f} mW")
print(f"At 1 mW: C_standard = {np.interp(1e-3, P, C_standard)/1e9:.2f} Gb/s, C_thermal = {np.interp(1e-3, P, C_thermal)/1e9:.2f} Gb/s")
print(f"At 100 mW: C_standard = {np.interp(0.1, P, C_standard)/1e9:.2f} Gb/s, C_thermal = {np.interp(0.1, P, C_thermal)/1e9:.2f} Gb/s")

# -------------------------
# Plotting
# -------------------------
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Color scheme
c_standard = '#2ecc71'  # green
c_thermal = '#e74c3c'   # red
c_temp = '#3498db'      # blue
c_noise = '#9b59b6'     # purple

# --- Plot 1: Temperature vs Power ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogx(P * 1e3, T, color=c_temp, linewidth=2)
ax1.axhline(T0, color='gray', linestyle='--', alpha=0.5, label=f'Ambient ({T0} K)')
ax1.set_xlabel('Transmit Power (mW)', fontsize=12)
ax1.set_ylabel('Temperature (K)', fontsize=12)
ax1.set_title('Heating: Temperature vs Power', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(P[0]*1e3, P[-1]*1e3)

# --- Plot 2: Noise Power vs Power ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.loglog(P * 1e3, N * 1e12, color=c_noise, linewidth=2, label='Thermal noise N(P)')
ax2.axhline(N0 * 1e12, color='gray', linestyle='--', alpha=0.5, label=f'Base noise N₀ ({N0*1e12:.2f} pW)')
ax2.set_xlabel('Transmit Power (mW)', fontsize=12)
ax2.set_ylabel('Noise Power (pW)', fontsize=12)
ax2.set_title('Thermal Noise: N = k_B · T(P) · B', fontsize=14)
ax2.grid(True, alpha=0.3, which='both')
ax2.legend()
ax2.set_xlim(P[0]*1e3, P[-1]*1e3)

# --- Plot 3: Capacity vs Power (main result) ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.semilogx(P * 1e3, C_standard / 1e9, color=c_standard, linewidth=2, label='Ideal (constant noise)')
ax3.semilogx(P * 1e3, C_thermal / 1e9, color=c_thermal, linewidth=2, label='With thermal noise')
ax3.axhline(C_limit / 1e9, color=c_thermal, linestyle=':', alpha=0.7, label=f'Thermal limit ({C_limit/1e9:.1f} Gb/s)')
ax3.axvline(P_90 * 1e3, color='gray', linestyle='--', alpha=0.5, label=f'90% of limit ({P_90*1e3:.1f} mW)')

ax3.fill_between(P * 1e3, C_thermal / 1e9, C_standard / 1e9, alpha=0.2, color='gray', label='Capacity loss')
ax3.set_xlabel('Transmit Power (mW)', fontsize=12)
ax3.set_ylabel('Channel Capacity (Gb/s)', fontsize=12)
ax3.set_title('Shannon-Hartley Capacity: Ideal vs Thermal', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='lower right')
ax3.set_xlim(P[0]*1e3, P[-1]*1e3)
ax3.set_ylim(0, None)

# --- Plot 4: Capacity per Watt (efficiency) ---
ax4 = fig.add_subplot(gs[1, 1])
efficiency_standard = C_standard / P  # bits/s per Watt = bits/Joule
efficiency_thermal = C_thermal / P

ax4.loglog(P * 1e3, efficiency_standard / 1e12, color=c_standard, linewidth=2, label='Ideal')
ax4.loglog(P * 1e3, efficiency_thermal / 1e12, color=c_thermal, linewidth=2, label='With thermal noise')

# Find max efficiency point for thermal case
idx_max_eff = np.argmax(efficiency_thermal)
P_max_eff = P[idx_max_eff]
ax4.axvline(P_max_eff * 1e3, color='gray', linestyle='--', alpha=0.5, 
            label=f'Max efficiency ({P_max_eff*1e6:.1f} µW)')

ax4.set_xlabel('Transmit Power (mW)', fontsize=12)
ax4.set_ylabel('Capacity per Watt (Tb/s/W)', fontsize=12)
ax4.set_title('Power Efficiency: Capacity / Power', fontsize=14)
ax4.grid(True, alpha=0.3, which='both')
ax4.legend()
ax4.set_xlim(P[0]*1e3, P[-1]*1e3)

plt.suptitle('Thermal Noise Impact on Channel Capacity\n(Power → Heat → Noise → Diminishing Returns)', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('thermal_noise_capacity.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: thermal_noise_capacity.png")
plt.close()

# -------------------------
# Bonus: Animated-style multi-curve plot
# -------------------------
fig2, ax = plt.subplots(figsize=(10, 7))

# Different thermal resistances (heat sinking quality)
R_th_values = [500, 2000, 5000, 20000, 50000]  # K/W
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(R_th_values)))

for R_th_i, color in zip(R_th_values, colors):
    eta_i = R_th_i * alpha
    beta_i = k_B * eta_i * B
    N_i = N0 + beta_i * P
    C_i = B * np.log2(1 + P / N_i)
    C_lim_i = B * np.log2(1 + 1/beta_i) if beta_i > 0 else np.inf
    ax.semilogx(P * 1e3, C_i / 1e9, color=color, linewidth=2, 
                label=f'R_th = {R_th_i} K/W (limit: {C_lim_i/1e9:.0f} Gb/s)')

# Ideal case
ax.semilogx(P * 1e3, C_standard / 1e9, 'g--', linewidth=2, label='Ideal (no heating)')

ax.set_xlabel('Transmit Power (mW)', fontsize=12)
ax.set_ylabel('Channel Capacity (Gb/s)', fontsize=12)
ax.set_title('Impact of Heat Sinking on Capacity\n(Lower R_th = Better Cooling)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right')
ax.set_xlim(P[0]*1e3, P[-1]*1e3)
ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig('thermal_heatsink_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved: thermal_heatsink_comparison.png")
plt.close()
