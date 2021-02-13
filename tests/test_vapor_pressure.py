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

import pytest
import numpy as np
import pandas as pd
from math import isnan
from fluids.numerics import linspace, assert_close, derivative, assert_close1d
from thermo.vapor_pressure import *
from thermo.vapor_pressure import VDI_TABULAR, WAGNER_MCGARRY
from chemicals.identifiers import check_CAS
from math import *

### Main predictor
@pytest.mark.meta_T_dept
def test_VaporPressure():
    # Ethanol, test as many methods asa possible at once
    EtOH = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')
    methods = list(EtOH.all_methods)
    methods.remove(VDI_TABULAR)

    Psat_calcs = []
    for i in methods:
        EtOH.method = i
        Psat_calcs.append(EtOH.T_dependent_property(305.))

    Psat_exp = [11579.634014300127, 11698.02742876088, 11590.408779316374, 11659.154222044575, 11592.205263402893, 11593.661615921257, 11612.378633936816, 11350.156640503357, 12081.738947110121, 14088.453409816764, 9210.26200064024]
    assert_close1d(sorted(Psat_calcs), sorted(Psat_exp))

    assert_close(EtOH.calculate(305, VDI_TABULAR), 11690.81660829924, rtol=1E-4)

    s = EtOH.as_json()
    assert 'json_version' in s
    obj2 = VaporPressure.from_json(s)
    assert EtOH == obj2



    # Use another chemical to get in ANTOINE_EXTENDED_POLING
    a = VaporPressure(CASRN='589-81-1')

    Psat_calcs = []
    for i in list(a.all_methods):
        a.method = i
        Psat_calcs.append(a.T_dependent_property(410))


    Psat_exp = [162944.82134710113, 162870.44794192078, 162865.5380455795]
    assert_close1d(sorted(Psat_calcs), sorted(Psat_exp))

    s = a.as_json()
    obj2 = VaporPressure.from_json(s)
    assert a == obj2

    # Test that methods return None
    EtOH = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')
    EtOH.extrapolation = None
    for i in list(EtOH.all_methods):
        EtOH.method = i
        assert EtOH.T_dependent_property(5000) is None

    # Test interpolation, extrapolation
    w = VaporPressure(Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, CASRN='7732-18-5')
    Ts = linspace(300, 350, 10)
    Ps = [3533.918074415897, 4865.419832056078, 6612.2351036034115, 8876.854141719203, 11780.097759775277, 15462.98385942125, 20088.570250257424, 25843.747665059742, 32940.95821687677, 41619.81654904555]
    w.add_tabular_data(Ts=Ts, properties=Ps)
    assert_close(w.T_dependent_property(305.), 4715.122890601165)
    w.extrapolation = 'interp1d'
    assert_close(w.T_dependent_property(200.), 0.5364148240126076)


    # Get a check for Antoine Extended
    cycloheptane = VaporPressure(Tb=391.95, Tc=604.2, Pc=3820000.0, omega=0.2384, CASRN='291-64-5')
    cycloheptane.method = ('ANTOINE_EXTENDED_POLING')
    cycloheptane.extrapolation = None
    assert_close(cycloheptane.T_dependent_property(410), 161647.35219882353)
    assert None == cycloheptane.T_dependent_property(400)

    with pytest.raises(Exception):
        cycloheptane.test_method_validity(300, 'BADMETHOD')


def test_VaporPressure_no_isnan():
    assert not isnan(VaporPressure(CASRN='4390-04-9').Tmin)

def test_VaporPressure_linear_extrapolation_non_negative():
    ethanol_psat = VaporPressure(Tb=351.39, Tc=514.0, Pc=6137000.0, omega=0.635, CASRN='64-17-5')

    # Make sure the constants are set to guard against future changes to defaults
    ethanol_psat.method = WAGNER_MCGARRY
    ethanol_psat.interpolation_T = (lambda T: 1/T)
    ethanol_psat.interpolation_property = (lambda P: log(P))
    ethanol_psat.interpolation_property_inv = (lambda P: exp(P))
    ethanol_psat.extrapolation = 'linear'

    assert_close(ethanol_psat(700), 59005875.32878946, rtol=1e-4)
    assert_close(ethanol_psat(100), 1.0475828451230242e-11, rtol=1e-4)

    assert ethanol_psat.T_limits['WAGNER_MCGARRY'][0] == ethanol_psat.WAGNER_MCGARRY_Tmin
    assert ethanol_psat.T_limits['WAGNER_MCGARRY'][1] == ethanol_psat.WAGNER_MCGARRY_Tc

    assert_close(ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tc),
                 ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tc-1e-6))
    assert_close(ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tc),
                 ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tc+1e-6))

    assert_close(ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tmin),
                 ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tmin-1e-6))
    assert_close(ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tmin),
                 ethanol_psat.T_dependent_property(ethanol_psat.WAGNER_MCGARRY_Tmin+1e-6))



    Tmin = ethanol_psat.T_limits[ethanol_psat.method][0]
    Ts = linspace(0.7*Tmin, Tmin*(1-1e-10), 10)
    Ps = [ethanol_psat(T) for T in Ts]

    # Confirms it's linear
    # plt.plot(1/np.array(Ts), np.log(Ps))
    # plt.show()
    def rsquared(x, y):
        import scipy.stats
        _, _, r_value, _, _ = scipy.stats.linregress(x, y)
        return r_value*r_value

    assert_close(rsquared(1/np.array(Ts), np.log(Ps)), 1, atol=1e-5)

    # TODO make work with different interpolation methods
#    assert ethanol_psat == VaporPressure.from_json(ethanol_psat.as_json())

@pytest.mark.meta_T_dept
def test_VaporPressure_extrapolation_solve_prop():
    cycloheptane = VaporPressure(Tb=391.95, Tc=604.2, Pc=3820000.0, omega=0.2384, CASRN='291-64-5')
    cycloheptane.method = 'ANTOINE_EXTENDED_POLING'
    cycloheptane.extrapolation = 'AntoineAB|DIPPR101_ABC'
    cycloheptane.T_dependent_property(T=4000)

    assert_close(cycloheptane.solve_property(1), 187.25621087267422)

    assert_close(cycloheptane.solve_property(1e-20), 60.677576120119156)

    assert_close(cycloheptane.solve_property(1e5), 391.3576035137979)
    assert_close(cycloheptane.solve_property(1e6), 503.31772463155266)
    assert_close(cycloheptane.solve_property(1e7), 711.8060977006566)
    assert_close(cycloheptane.solve_property(3e7), 979.2319342086599)

def test_VaporPressure_bestfit_derivatives():
    obj = VaporPressure(poly_fit=(175.7, 512.49, [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10,
                                                -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708]))
    assert_close(obj.T_dependent_property(300), 18601.061401014867, rtol=1e-11)
    assert_close(obj.T_dependent_property_derivative(300), 954.1652489206775, rtol=1e-11)
    assert_close(obj.T_dependent_property_derivative(300, order=2), 41.8787546283273, rtol=1e-11)
    assert_close(derivative(obj.T_dependent_property, 300, dx=300*1e-7), obj.T_dependent_property_derivative(300))
    assert_close(derivative(obj.T_dependent_property_derivative, 300, dx=300*1e-7), obj.T_dependent_property_derivative(300, order=2))


@pytest.mark.meta_T_dept
def test_VaporPressure_extrapolation_AB():
    obj = VaporPressure(Tb=309.21, Tc=469.7, Pc=3370000.0, omega=0.251, CASRN='109-66-0', load_data=True, extrapolation='AntoineAB')
    obj.method = WAGNER_MCGARRY
    obj.calculate_derivative(300, WAGNER_MCGARRY)

    for extrapolation in ('AntoineAB', 'DIPPR101_ABC', 'AntoineAB|AntoineAB', 'DIPPR101_ABC|DIPPR101_ABC',
                          'DIPPR101_ABC|AntoineAB', 'AntoineAB|DIPPR101_ABC'):
        obj.extrapolation = extrapolation

        assert_close(obj.T_dependent_property(obj.WAGNER_MCGARRY_Tc),
                     obj.T_dependent_property(obj.WAGNER_MCGARRY_Tc-1e-6))
        assert_close(obj.T_dependent_property(obj.WAGNER_MCGARRY_Tc),
                     obj.T_dependent_property(obj.WAGNER_MCGARRY_Tc+1e-6))

        assert_close(obj.T_dependent_property(obj.WAGNER_MCGARRY_Tmin),
                     obj.T_dependent_property(obj.WAGNER_MCGARRY_Tmin-1e-6))
        assert_close(obj.T_dependent_property(obj.WAGNER_MCGARRY_Tmin),
                     obj.T_dependent_property(obj.WAGNER_MCGARRY_Tmin+1e-6))


def test_VaporPressure_fast_Psat_poly_fit():
    corr = VaporPressure(poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))
    # Low temperature values - up to 612 Pa
    assert_close(corr.solve_property(1e-5), corr.solve_prop_poly_fit(1e-5), rtol=1e-10)
    assert_close(corr.solve_property(1), corr.solve_prop_poly_fit(1), rtol=1e-10)
    assert_close(corr.solve_property(100), corr.solve_prop_poly_fit(100), rtol=1e-10)

    P_trans = exp(corr.poly_fit_Tmin_value)
    assert_close(corr.solve_property(P_trans), corr.solve_prop_poly_fit(P_trans), rtol=1e-10)
    assert_close(corr.solve_property(P_trans + 1e-7), corr.solve_prop_poly_fit(P_trans + 1e-7), rtol=1e-10)

    # Solver region
    assert_close(corr.solve_property(1e5), corr.solve_prop_poly_fit(1e5), rtol=1e-10)
    assert_close(corr.solve_property(1e7), corr.solve_prop_poly_fit(1e7), rtol=1e-10)

    P_trans = exp(corr.poly_fit_Tmax_value)
    assert_close(corr.solve_property(P_trans), corr.solve_prop_poly_fit(P_trans), rtol=1e-10)
    assert_close(corr.solve_property(P_trans + 1e-7), corr.solve_prop_poly_fit(P_trans + 1e-7), rtol=1e-10)

    # High T
    assert_close(corr.solve_property(1e8), corr.solve_prop_poly_fit(1e8), rtol=1e-10)

    # Extrapolation
    from thermo.vapor_pressure import BESTFIT, BEST_FIT_AB, BEST_FIT_ABC
    obj = VaporPressure(poly_fit=(178.01, 591.74, [-8.638045111752356e-20, 2.995512203611858e-16, -4.5148088801006036e-13, 3.8761537879200513e-10, -2.0856828984716705e-07, 7.279010846673517e-05, -0.01641020023565049, 2.2758331029405516, -146.04484159879843]))
    assert_close(obj.calculate(1000, BEST_FIT_AB), 78666155.90418352, rtol=1e-10)
    assert_close(obj.calculate(1000, BEST_FIT_ABC), 156467764.5930495, rtol=1e-10)

    assert_close(obj.calculate(400, BESTFIT), 157199.6909849476, rtol=1e-10)
    assert_close(obj.calculate(400, BEST_FIT_AB), 157199.6909849476, rtol=1e-10)
    assert_close(obj.calculate(400, BEST_FIT_ABC), 157199.6909849476, rtol=1e-10)

@pytest.mark.meta_T_dept
def test_VaporPressure_extrapolation_no_validation():
    N2 = VaporPressure(CASRN='7727-37-9', extrapolation='DIPPR101_ABC')
    N2.method = WAGNER_MCGARRY
    assert N2(298.15) is not None
    assert N2(1000.15) is not None


def test_VaporPressure_fast_Psat_poly_fit_extrapolation():
    obj = VaporPressure(poly_fit=(175.7, 512.49, [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10,
                                                -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708]))
    obj.extrapolation = 'AntoineAB|DIPPR101_ABC'
    assert_close(obj.solve_property(.0000000000001), 3.2040851644645945)
    assert_close(obj.solve_property(300), 237.7793675652309)
    assert_close(obj.solve_property(1e8), 661.6135315674736)
