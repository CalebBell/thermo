import matplotlib.pyplot as plt
import numpy as np
from thermo import *
from fluids.numerics import logspace

ethanol_psat = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')
ethanol_psat.plot_T_dependent_property(Tmin=1, Tmax=300, methods=['VDI_TABULAR', 'DIPPR_PERRY_8E', 'COOLPROP'], pts=50, only_valid=False)
