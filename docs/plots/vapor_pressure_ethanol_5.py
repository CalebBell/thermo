from thermo import *

ethanol_psat = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')
ethanol_psat.plot_T_dependent_property(Tmin=230, Tmax=500, methods=['VDI_TABULAR', 'DIPPR_PERRY_8E', 'AMBROSE_WALTON', 'VDI_PPDS', 'WAGNER_MCGARRY'], pts=50, only_valid=False, order=1)
