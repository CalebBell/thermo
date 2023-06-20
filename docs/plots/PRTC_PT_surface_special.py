
import matplotlib.pyplot as plt

from thermo import *

obj = PRTranslatedConsistent(Tc=190.564, Pc=4599000.0, omega=0.008, T=300., P=1e5)
fig = obj.PT_surface_special(show=False, pts=50)
plt.show()
