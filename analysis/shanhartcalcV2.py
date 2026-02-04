import numpy as np

import meep as mp
if mp.am_master():
    print("only once")


# Uniform PSD - Three givens:
# Band Edges (f1, f2)
#Noise PSD Level: N0 (in W/Hz)
#Total Transmit Power: P (In W)
# Bandwidth is defined as: B= f2 - f1 
# B should be in Hz (DO NOT MIX GHZ AND HZ)

freq1 = 1.00e9 #Hz = 1.00 GHz
freq2 = 1.10e9 #Hz = 1.10 GHz
N = 8192 
N0 = 1.0e-20 # W/Hz (flat noise PSD)
powertotal = 1.0e-12 # W (total power accross band)

bandwidth = freq2 - freq1
df = bandwidth / N
f  = np.linspace(freq1, freq2, N, endpoint=False)
#So, Sx(f) = P[Total in W] over B (Our N0). Therefore:
# Sx/N0 = Ptotal / N0(B)

#Build channel notch in |H(f)|^2
f0 = 1.05e9
sigma =2.0e6 #Notch width (Hz)
depth = 0.99 #Deep notch, but not zero

H2 = 1.0 - depth * np.exp(-0.5 * ((f- f0) / sigma)**2)
H2 = np.clip(H2, 1e-12, None) #Saftey, avoid division by 0

#Build complex noise (Base + interference spike)
fi = 1.07e9 #interference center
sigma_i = 1.0e6 #interference width
A = 200.0 # Spike amplitude multiplyer

Sn = N0 * (1.0 + A * np.exp(-0.5 * ((f - fi) / sigma_i)**2))

#PSD Allocation (baseline)

Sx_baseline = powertotal / bandwidth
Pk_baseline = Sx_baseline  * df #Power per bin
Nk = Sn * df #Noise power per bin

snr_baseline = (H2 * Pk_baseline) / Nk
C_baseline = np.sum(df * np.log2(1.0 + snr_baseline))

#WATER FILLING:
# PK = max(0, mu - eta_k), eta_k = Nk / H2

eta = Nk / H2

def allocate(mu):
    Pk = np.maximum(0.0, mu - eta)
    return Pk, np.sum(Pk)

#Bracket Mu
mu_low = np.min(eta) #Gives ~ 0 Allocated
mu_high = np.min(eta) + powertotal #Might be enough, bump later if needed

#if bracket fails, expand mu_high until it gets enough power
for _ in range(50):
    _, p_sum = allocate(mu_high)
    if p_sum >= powertotal:
        break
    mu_high *= 2.0

#Bisection on mu
tol = 1e-12
for _ in range (80):
    mu_mid = 0.5 * (mu_low + mu_high)
    Pk_mid, p_sum = allocate(mu_mid)

    if abs(p_sum - powertotal) <= tol:
        break
    if p_sum < powertotal:
        mu_low = mu_mid
    else:
        mu_high = mu_mid

#Renormalize:
Pk_wf = Pk_mid * (powertotal / np.sum(Pk_mid))

snr_wf = (H2 * Pk_wf) / Nk
C_wf = np.sum(df * np.log2(1.0 + snr_wf))

print("Baseline Capacity:", C_baseline)
print("Water Filled Capacity:", C_wf)
print("Capacity Gain Factor (WF/Baseline):", C_wf / C_baseline)

print("Baseline total power:", np.sum(np.full(N, Pk_baseline)))
print("Waterfilling Total Power:", np.sum(Pk_wf))
print("Waterfilling minimum power bin:", np.min(Pk_wf), "Waterfilling Max Power Bin:", np.max(Pk_wf))

#Checksum:
print("All Waterfilling powers >= 0?", np.all(Pk_wf >= -1e-15)) #Should be true, unless I'm an idiot
print("WF beats uniform?", C_wf >= C_baseline)
