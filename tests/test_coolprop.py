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
SOFTWARE.
'''

import pytest
from chemicals.identifiers import check_CAS
from fluids.numerics import assert_close, derivative

from thermo.coolprop import *
from thermo.coolprop import has_CoolProp


@pytest.mark.CoolProp
@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_fluid_props():
    has_CoolProp()
#    tots = [sum([getattr(f, prop) for f in coolprop_fluids.values()]) for prop in ['Tmin', 'Tmax', 'Pmax', 'Tc', 'Pc', 'Tt', 'omega']]
#    tots_exp = [18589.301, 71575.0, 31017000000.0, 45189.59849999997, 440791794.7987591, 18589.301, 30.90243968446593]


#    assert_close1d(tots_exp, tots)

    assert len(coolprop_fluids) == len(coolprop_dict)
    assert len(coolprop_dict) == 105
    assert all(check_CAS(i) for i in coolprop_dict)



@pytest.mark.skipif(not has_CoolProp(), reason='CoolProp is missing')
def test_CoolProp_T_dependent_property():
    # Below the boiling point
    rhow = CoolProp_T_dependent_property(298.15, '7732-18-5', 'D', 'l')
    assert_close(rhow, 997.0476367603451)

    rhow = CoolProp_T_dependent_property(298.15, '7732-18-5', 'D', 'g')
    assert_close(rhow, 0.0230748041827597)

    # Above the boiling point
    rhow = CoolProp_T_dependent_property(450, '7732-18-5', 'D', 'l')
    assert_close(rhow, 890.3412497616716)

    rhow = CoolProp_T_dependent_property(450, '7732-18-5', 'D', 'g')
    assert_close(rhow, 0.49104706182775576)

    with pytest.raises(Exception):
        CoolProp_T_dependent_property(298.15, 'BADCAS', 'D', 'l')

    with pytest.raises(Exception):
        CoolProp_T_dependent_property(298.15, '7732-18-5', 'BADKEY', 'l')

    with pytest.raises(Exception):
        CoolProp_T_dependent_property(298.15, '7732-18-5', 'D', 'badphase')

    # Above the critical point
    with pytest.raises(Exception):
        CoolProp_T_dependent_property(700, '7732-18-5', 'D', 'l')

    rhow = CoolProp_T_dependent_property(700, '7732-18-5', 'D', 'g')
    assert_close(rhow, 0.3139926976198761)


# This test was part of an earlier attempt which is no longer relevant
#@pytest.mark.CoolProp
#@pytest.mark.slow
#def test_CP_approximators():
#    from thermo.coolprop import coolprop_fluids, CP_approximators
#    for CAS in coolprop_fluids:
#        obj = CP_approximators[CAS]
#        props = ['DMOLAR', 'HMOLAR', 'SMOLAR', 'SPEED_OF_SOUND', 'CONDUCTIVITY',
#                 'VISCOSITY', 'CPMOLAR', 'CVMOLAR']
#        for prop in props:
#            if hasattr(obj, prop+'_g'):
#                obj.validate_prop(prop, 'g', evaluated_points=15)
#
#
#


def test_Helmholtz_A0_water():
    # Water
    A0_kwargs = {'IdealGasHelmholtzLead_a1': -8.3204464837497, 'IdealGasHelmholtzLead_a2': 6.6832105275932,
                 'IdealGasHelmholtzLogTau_a': 3.00632,
                 'IdealGasHelmholtzPlanckEinstein_ns': [0.012436, 0.97315, 1.2795, 0.96956, 0.24873],
                 'IdealGasHelmholtzPlanckEinstein_ts': [1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105]}
    T = 400
    Tc_A0 = 647.096
    tau = Tc_A0/T

    delta = 1.1
    A0 = Helmholtz_A0(tau, delta, **A0_kwargs)
    A0_expect = 4.027842100447844
    assert_close(A0, A0_expect, rtol=1e-13)

    A0_kwargs2 = A0_kwargs.copy()
    A0_kwargs2['delta'] = delta
    dA0_numerical = derivative(Helmholtz_A0, tau, dx=tau*8e-8, kwargs=A0_kwargs2)
    dA0 = Helmholtz_dA0_dtau(tau, delta, **A0_kwargs)
    dA0_expect = 8.5551727965326573866 # with sympy
    assert_close(dA0, dA0_expect, rtol=1e-13)
    assert_close(dA0, dA0_numerical, rtol=1e-8)

    d2A0_numerical = derivative(Helmholtz_dA0_dtau, tau, dx=tau*2e-7, kwargs=A0_kwargs2)
    d2A0 = Helmholtz_d2A0_dtau2(tau, delta, **A0_kwargs)
    d2A0_expect = -1.1924853534853863124 # with sympy
    assert_close(d2A0, d2A0_expect, rtol=1e-13)
    assert_close(d2A0, d2A0_numerical, rtol=1e-8)

    d3A0_numerical = derivative(Helmholtz_d2A0_dtau2, tau, dx=tau*2e-7, kwargs=A0_kwargs2)
    d3A0 = Helmholtz_d3A0_dtau3(tau, delta, **A0_kwargs)
    d3A0_expect = 1.5708896021584690672 # with sympy
    assert_close(d3A0, d3A0_expect, rtol=1e-13)
    assert_close(d3A0, d3A0_numerical, rtol=1e-8)


    # from sympy import symbols, diff, N, log, exp
    # tau_sym = symbols('tau')
    # thermo.coolprop.log = log
    # thermo.coolprop.exp = exp

    # A0_symbolic = Helmholtz_A0(tau_sym, delta, **A0_kwargs)
    # dA0 = N(diff(A0_symbolic, tau_sym, 3).subs(tau_sym, tau), 20)
    # print(dA0)


def test_Helmholtz_A0_CO2():
    A0_kwargs = {'IdealGasHelmholtzLead_a1': 8.37304456, 'IdealGasHelmholtzLead_a2': -3.70454304,
                 'IdealGasHelmholtzLogTau_a': 2.5,
                 'IdealGasHelmholtzPlanckEinstein_ns': [1.99427042, 0.62105248, 0.41195293, 1.04028922, 0.08327678],
                 'IdealGasHelmholtzPlanckEinstein_ts': [3.15163, 6.1119, 6.77708, 11.32384, 27.08792]}
    T = 400
    Tc_A0 = 304.1282
    tau = Tc_A0/T

    delta = 1.1
    A0 = Helmholtz_A0(tau, delta, **A0_kwargs)
    A0_expect =4.767709823206007
    assert_close(A0, A0_expect, rtol=1e-13)

def test_Helmholtz_A0_methanol():

    A0_kwargs = {'IdealGasHelmholtzLead_a1': 13.9864114647, 'IdealGasHelmholtzLead_a2': 3200.6369296,
                 'IdealGasHelmholtzLogTau_a': 3.1950423807804,
                 'IdealGasHelmholtzPower_ns': [-0.585735321498174, -0.06899642310301084, 0.008650264506162275],
                 'IdealGasHelmholtzPower_ts': [-1, -2, -3],
                 'IdealGasHelmholtzPlanckEinstein_ns': [4.70118076896145],
                 'IdealGasHelmholtzPlanckEinstein_ts': [3.7664265756]}
    T = 400
    Tc_A0 = 512.5
    tau = Tc_A0/T

    delta = 1.1
    A0 = Helmholtz_A0(tau, delta, **A0_kwargs)
    A0_expect = 4115.156699606922
    assert_close(A0, A0_expect, rtol=1e-13)


def test_Helmholtz_A0_fluorine():

    A0_kwargs = {'IdealGasHelmholtzPower_ns': [3.0717001e-06, -5.2985762e-05, -16.372517, 3.6884682e-05, 4.3887271],
                 'IdealGasHelmholtzPower_ts': [-4, -3, 1, 2, 0],
                 'IdealGasHelmholtzLogTau_a': 2.5011231,
                 'IdealGasHelmholtzPlanckEinsteinGeneralized_ns': [1.012767],
                 'IdealGasHelmholtzPlanckEinsteinGeneralized_ts': [-8.9057501],
                 'IdealGasHelmholtzPlanckEinsteinGeneralized_cs': [1],
                 'IdealGasHelmholtzPlanckEinsteinGeneralized_ds': [-1],
                 'IdealGasHelmholtzLead_a1': 0, 'IdealGasHelmholtzLead_a2': 0}
    T = 400
    Tc_A0 = 144.414
    tau = Tc_A0/T

    delta = 1.1
    A0 = Helmholtz_A0(tau, delta, **A0_kwargs)
    A0_expect = -4.017544722732779
    assert_close(A0, A0_expect, rtol=1e-13)

    A0_kwargs2 = A0_kwargs.copy()
    A0_kwargs2['delta'] = delta
    dA0_numerical = derivative(Helmholtz_A0, tau, dx=tau*8e-8, kwargs=A0_kwargs2)
    dA0 = Helmholtz_dA0_dtau(tau, delta, **A0_kwargs)
    dA0_expect = -9.0602725221550990398 # with sympy
    assert_close(dA0, dA0_expect, rtol=1e-13)
    assert_close(dA0, dA0_numerical, rtol=1e-8)


    d2A0_numerical = derivative(Helmholtz_dA0_dtau, tau, dx=tau*2e-7, kwargs=A0_kwargs2)
    d2A0 = Helmholtz_d2A0_dtau2(tau, delta, **A0_kwargs)
    d2A0_expect = -22.764047332314731875 # with sympy
    assert_close(d2A0, d2A0_expect, rtol=1e-13)
    assert_close(d2A0, d2A0_numerical, rtol=1e-8)

    d3A0_numerical = derivative(Helmholtz_d2A0_dtau2, tau, dx=tau*2e-7, kwargs=A0_kwargs2)
    d3A0 = Helmholtz_d3A0_dtau3(tau, delta, **A0_kwargs)
    d3A0_expect = 141.04704963538839024 # with sympy
    assert_close(d3A0, d3A0_expect, rtol=1e-13)
    assert_close(d3A0, d3A0_numerical, rtol=1e-8)

    # from sympy import symbols, diff, N, log, exp
    # tau_sym = symbols('tau')
    # thermo.coolprop.log = log
    # thermo.coolprop.exp = exp

    # A0_symbolic = Helmholtz_A0(tau_sym, delta, **A0_kwargs)
    # dA0 = N(diff(A0_symbolic, tau_sym, 3).subs(tau_sym, tau), 20)
    # print(dA0)

