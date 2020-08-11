# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
SOFTWARE.'''

from numpy.testing import assert_allclose
import pytest
import numpy as np
import pandas as pd
from fluids.numerics import assert_close, derivative, assert_close1d
from thermo.vapor_pressure import *
from thermo.vapor_pressure import VDI_TABULAR
from chemicals.identifiers import checkCAS
from math import *

### Main predictor
@pytest.mark.meta_T_dept
def test_VaporPressure():
    # Ethanol, test as many methods asa possible at once
    EtOH = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')
    EtOH.T_dependent_property(305.)
    methods = EtOH.sorted_valid_methods
    methods.remove(VDI_TABULAR)
    Psat_calcs = [(EtOH.set_user_methods(i), EtOH.T_dependent_property(305.))[1] for i in methods]
    Psat_exp = [11579.634014300127, 11698.02742876088, 11590.408779316374, 11659.154222044575, 11592.205263402893, 11593.661615921257, 11612.378633936816, 11350.156640503357, 12081.738947110121, 14088.453409816764, 9210.26200064024]
    assert_allclose(sorted(Psat_calcs), sorted(Psat_exp))
    
    assert_allclose(EtOH.calculate(305, VDI_TABULAR), 11690.81660829924, rtol=1E-4)

    # Use another chemical to get in ANTOINE_EXTENDED_POLING
    a = VaporPressure(CASRN='589-81-1')
    a.T_dependent_property(410)
    Psat_calcs = [(a.set_user_methods(i), a.T_dependent_property(410))[1] for i in a.sorted_valid_methods]
    Psat_exp = [162944.82134710113, 162870.44794192078, 162865.5380455795]
    assert_allclose(Psat_calcs, Psat_exp)

    # Test that methods return None
    EtOH = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')
    EtOH.T_dependent_property(298.15)
    Psat_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(5000))[1] for i in EtOH.sorted_valid_methods]
    assert [None]*11 == Psat_calcs

    # Test interpolation, extrapolation
    w = VaporPressure(Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, CASRN='7732-18-5')
    Ts = np.linspace(300, 350, 10)
    Ps = [3533.918074415897, 4865.419832056078, 6612.2351036034115, 8876.854141719203, 11780.097759775277, 15462.98385942125, 20088.570250257424, 25843.747665059742, 32940.95821687677, 41619.81654904555]
    w.set_tabular_data(Ts=Ts, properties=Ps)
    assert_allclose(w.T_dependent_property(305.), 4715.122890601165)
    w.tabular_extrapolation_permitted = True
    assert_allclose(w.T_dependent_property(200.), 0.5364148240126076)
    w.tabular_extrapolation_permitted = False
    assert_allclose(w.T_dependent_property(200.), 0.09934382362141778) # Fall back to ambrose-Walton


    # Get a check for Antoine Extended
    cycloheptane = VaporPressure(Tb=391.95, Tc=604.2, Pc=3820000.0, omega=0.2384, CASRN='291-64-5')
    cycloheptane.set_user_methods('ANTOINE_EXTENDED_POLING', forced=True)
    assert_allclose(cycloheptane.T_dependent_property(410), 161647.35219882353)
    assert None == cycloheptane.T_dependent_property(400)

    with pytest.raises(Exception):
        cycloheptane.test_method_validity(300, 'BADMETHOD')

def test_VaporPressure_fast_Psat_best_fit():
    corr = VaporPressure(best_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))
    # Low temperature values - up to 612 Pa
    assert_close(corr.solve_prop(1e-5), corr.solve_prop_best_fit(1e-5), rtol=1e-10)
    assert_close(corr.solve_prop(1), corr.solve_prop_best_fit(1), rtol=1e-10)
    assert_close(corr.solve_prop(100), corr.solve_prop_best_fit(100), rtol=1e-10)
    
    P_trans = exp(corr.best_fit_Tmin_value)
    assert_close(corr.solve_prop(P_trans), corr.solve_prop_best_fit(P_trans), rtol=1e-10)
    assert_close(corr.solve_prop(P_trans+1e-7), corr.solve_prop_best_fit(P_trans+1e-7), rtol=1e-10)
    
    # Solver region
    assert_close(corr.solve_prop(1e5), corr.solve_prop_best_fit(1e5), rtol=1e-10)
    assert_close(corr.solve_prop(1e7), corr.solve_prop_best_fit(1e7), rtol=1e-10)
    
    P_trans = exp(corr.best_fit_Tmax_value)
    assert_close(corr.solve_prop(P_trans), corr.solve_prop_best_fit(P_trans), rtol=1e-10)
    assert_close(corr.solve_prop(P_trans+1e-7), corr.solve_prop_best_fit(P_trans+1e-7), rtol=1e-10)
    
    # High T
    assert_close(corr.solve_prop(1e8), corr.solve_prop_best_fit(1e8), rtol=1e-10)
