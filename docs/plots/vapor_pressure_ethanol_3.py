import matplotlib.pyplot as plt
import numpy as np
from thermo import *
from fluids.numerics import logspace

ethanol_psat = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')
ethanol_psat.plot_T_dependent_property(Tmin=400, Tmax=500, methods=['BOILING_CRITICAL', 'SANJARI', 'LEE_KESLER_PSAT', 'VDI_TABULAR', 'DIPPR_PERRY_8E'], pts=50, order=1)
