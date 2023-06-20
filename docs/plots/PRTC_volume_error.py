from thermo import *

obj = PRTranslatedConsistent(Tc=190.564, Pc=4599000.0, omega=0.008, T=300., P=1e5)
obj.volume_errors(show=True, plot=True, pts=7)
