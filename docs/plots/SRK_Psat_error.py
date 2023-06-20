from thermo import *

eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=400., P=1E6)
eos.Psat_errors(plot=True, show=True)
