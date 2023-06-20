from thermo import *

eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=400., P=1E6)
eos.a_alpha_plot(show=True, plot=True, Tmin=1.0)
