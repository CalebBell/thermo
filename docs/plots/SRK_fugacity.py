import matplotlib.pyplot as plt
import numpy as np
from thermo import *
from fluids.numerics import logspace
from math import log10

eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=400., P=1E6)
eos.saturation_prop_plot('fugacity', show=True, plot=True, Tmin=10, pts=1000, Tmax=eos.Tc*.9999999, both=True)
