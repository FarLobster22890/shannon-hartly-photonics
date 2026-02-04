import numpy as np

# Uniform PSD - Three givens:
# Band Edges (f1, f2)
#Noise PSD Level: N0 (in W/Hz)
#Total Transmit Power: P (In W)
# Bandwidth is defined as: B= f2 - f1 
# B should be in Hz (DO NOT MIX GHZ AND HZ)

freq1 = 1.00 #Hz = 1.00 GHz
freq2 = 1.10e9 #Hz = 1.10 GHz
N = 1000 
N0 = 1.0e-20 # W/Hz (flat noise PSD)
powertotal = 1.0e-3 # W (total power accross band)

bandwidth = freq2 - freq1
df = bandwidth / N
#So, Sx(f) = P[Total in W] over B (Our N0). Therefore:
# Sx/N0 = Ptotal / N0(B)
#Now Channel + noise over bins:
H2 = np.ones(N) #H^2 = 1 - noise is flat
Sn = np.full(N, N0)

#Continuous
Sx = powertotal / bandwidth
snr = Sx / N0
continuous_capacity = bandwidth * np.log2(1 + snr)

#Validate (Check)
Pk = Sx * df #W per bin
Nk = Sn * df #W noise per bin
snr_k = (H2 * Pk) / Nk #Dimensionless
continuous_checksum = np.sum(df * np.log2(1 + snr_k))

print("CHECKSUM AND CAPACITY ***SHOULD*** BE EXTREMLY CLOSE",)
print("Continuous Capacity=", continuous_capacity)
print("Continuous Checksum=", continuous_checksum)
print("Relative Error=", abs(continuous_checksum - continuous_capacity) / continuous_capacity)
