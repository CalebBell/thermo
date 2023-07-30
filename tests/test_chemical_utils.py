'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2023 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from math import *

import pytest
from chemicals.elements import periodic_table
from chemicals.identifiers import pubchem_db
from fluids.constants import R
from fluids.numerics import assert_close
from thermo.chemical_utils import standard_state_ideal_gas_formation

from thermo.chemical import Chemical

def test_standard_state_ideal_gas_formation_water():
    c = Chemical('water') # H2O
    Ts = [1e-10, 100, 200, 298.15, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000]
    Ts = Ts[1:]
    dHs_Janaf = [-238.921, -240.083, -240.9, -241.826, -241.844, -242.846, -243.826, -244.758, -245.632, -246.443, -247.185, -247.857, -248.46, -248.997, -249.473, -249.894, -250.265, -250.592, -250.881, -251.138, -251.368, -251.575, -251.762, -251.934, -252.092, -252.239, -252.379, -252.513, -252.643, -252.771, -252.897, -253.024, -253.152, -253.282, -253.416, -253.553, -253.696, -253.844, -253.997, -254.158, -254.326, -254.501, -254.684, -254.876, -255.078, -255.288, -255.508, -255.738, -255.978, -256.229, -256.491, -256.763, -257.046, -257.338, -257.639, -257.95, -258.268, -258.595, -258.93, -259.272, -259.621, -259.977]
    dHs_Janaf = [v*1000 for v in dHs_Janaf[1:]]

    for T, dH_janaf in zip(Ts, dHs_Janaf):
        dH_calc = standard_state_ideal_gas_formation(c, T)[0]
        if T < 1000:
            rtol = 1e-3
        elif T < 2000:
            rtol = 3e-3
        else:
            rtol = .2
        assert_close(dH_calc, dH_janaf, rtol=rtol)