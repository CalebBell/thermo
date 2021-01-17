import matplotlib.pyplot as plt
import numpy as np
from thermo import *
from fluids.numerics import logspace

water_psat = VaporPressure(Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, CASRN='7732-18-5')

water_mu = ViscosityLiquid(CASRN="7732-18-5", MW=18.01528, Tm=273.15, Tc=647.14, Pc=22048320.0, 
                      Vc=5.6e-05, omega=0.344,method="DIPPR_PERRY_8E", Psat=water_psat,
                      method_P="LUCAS")
try:
    water_mu.plot_TP_dependent_property(Tmin=400, Pmin=1e5, Pmax=1e8, methods_P=['COOLPROP','LUCAS'], pts=15, only_valid=False)
except:
    pass
