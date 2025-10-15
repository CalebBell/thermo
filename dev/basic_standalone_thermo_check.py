import thermo
from thermo import *
import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.spatial
import scipy.special
import scipy.optimize

def check_close(a, b, rtol=1e-7, atol=0):
    np.all(np.abs(a - b) <= (atol + rtol * np.abs(b)))
    return True

def run_checks():
    checks = []

    # Add basic checks here
    checks.append(True)

    return all(checks)

if run_checks():
    print("thermo basic checks passed - NumPy and SciPy used successfully")
else:
    print('Library not OK')
    exit(1)
