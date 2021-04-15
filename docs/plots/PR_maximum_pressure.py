import matplotlib.pyplot as plt
import numpy as np
from thermo import *
from fluids.numerics import logspace
from math import log10

thing = PR(P=1e5, T=460, Tc=658.0, Pc=1820000.0, omega=0.9)

Ts = logspace(log10(1000), log10(2000000), 100)
Ps = []
for T in Ts:
    try:
        Ps.append(thing.to(T=T, V=thing.V_l).P)
    except ValueError:
        Ps.append(1e5)

plt.loglog(Ts, Ps)
plt.xlabel('Temperature [K]')
plt.ylabel('Pressure [Pa]')
plt.title("Constant-volume calculated pressure at V=%.8f m^3/mol" %(thing.V_l))
#plt.show()
