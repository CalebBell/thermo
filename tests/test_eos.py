# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import numpy as np
import pytest
from thermo import eos
from thermo.eos import *
from thermo.eos import eos_2P_list
from thermo.utils import allclose_variable
from fluids.constants import R
from math import log, exp, sqrt, log10
from fluids.numerics import linspace, derivative, logspace, assert_close, assert_close1d, assert_close2d, assert_close3d


def main_derivatives_and_departures_slow(T, P, V, b, delta, epsilon, a_alpha,
                                    da_alpha_dT, d2a_alpha_dT2):
    dP_dT = R/(V - b) - da_alpha_dT/(V**2 + V*delta + epsilon)
    dP_dV = -R*T/(V - b)**2 - (-2*V - delta)*a_alpha/(V**2 + V*delta + epsilon)**2
    d2P_dT2 = -d2a_alpha_dT2/(V**2 + V*delta + epsilon)
    d2P_dV2 = 2*(R*T/(V - b)**3 - (2*V + delta)**2*a_alpha/(V**2 + V*delta + epsilon)**3 + a_alpha/(V**2 + V*delta + epsilon)**2)
    d2P_dTdV = -R/(V - b)**2 + (2*V + delta)*da_alpha_dT/(V**2 + V*delta + epsilon)**2
    H_dep = P*V - R*T + 2*(T*da_alpha_dT - a_alpha)*catanh((2*V + delta)/sqrt(delta**2 - 4*epsilon)).real/sqrt(delta**2 - 4*epsilon)
    S_dep = -R*log(V) + R*log(P*V/(R*T)) + R*log(V - b) + 2*da_alpha_dT*catanh((2*V + delta)/sqrt(delta**2 - 4*epsilon)).real/sqrt(delta**2 - 4*epsilon)
    Cv_dep = -T*(sqrt(1/(delta**2 - 4*epsilon))*log(V - delta**2*sqrt(1/(delta**2 - 4*epsilon))/2 + delta/2 + 2*epsilon*sqrt(1/(delta**2 - 4*epsilon))) - sqrt(1/(delta**2 - 4*epsilon))*log(V + delta**2*sqrt(1/(delta**2 - 4*epsilon))/2 + delta/2 - 2*epsilon*sqrt(1/(delta**2 - 4*epsilon))))*d2a_alpha_dT2
    return dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep

@pytest.mark.slow
@pytest.mark.sympy
def test_PR_with_sympy():
    return
    # Test with hexane
    from sympy import Rational, symbols, sqrt, solve, diff, integrate, N, nsolve

    P, T, V = symbols('P, T, V')
    Tc = Rational('507.6')
    Pc = 3025000
    omega = Rational('0.2975')

    X = (-1 + (6*sqrt(2)+8)**Rational(1,3) - (6*sqrt(2)-8)**Rational(1,3))/3
    c1 = (8*(5*X+1)/(49-37*X)) # 0.45724
    c2 = (X/(X+3)) # 0.07780


    R_sym = Rational(R)
    a = c1*R_sym**2*Tc**2/Pc
    b = c2*R_sym*Tc/Pc

    kappa = Rational('0.37464')+ Rational('1.54226')*omega - Rational('0.26992')*omega**2

    a_alpha = a*(1 + kappa*(1-sqrt(T/Tc)))**2
    PR_formula = R_sym*T/(V-b) - a_alpha/(V*(V+b)+b*(V-b)) - P



    # First test - volume, liquid

    T_l, P_l = 299, 1000000
    PR_obj_l = PR(T=T_l, P=P_l, Tc=507.6, Pc=3025000, omega=0.2975)
#    solns = solve(PR_formula.subs({T: T_l, P:P_l}))
#    solns = [N(i) for i in solns]
#    V_l_sympy = float([i for i in solns if i.is_real][0])

    base = PR_formula.subs({T: T_l, P:P_l})
    V_l_sympy = nsolve(base, V, (.000130, 1), solver='bisect', verify=False)
#    V_l_sympy = 0.00013022212513965863

    assert_close(PR_obj_l.V_l, V_l_sympy)

    def numeric_sub_l(expr):
        return float(expr.subs({T: T_l, P:P_l, V:PR_obj_l.V_l}))

    # First derivatives
    dP_dT = diff(PR_formula, T)
    assert_close(numeric_sub_l(dP_dT), PR_obj_l.dP_dT_l)

    dP_dV = diff(PR_formula, V)
    assert_close(numeric_sub_l(dP_dV), PR_obj_l.dP_dV_l)

    dV_dT = -diff(PR_formula, T)/diff(PR_formula, V)
    assert_close(numeric_sub_l(dV_dT), PR_obj_l.dV_dT_l)

    dV_dP = -dV_dT/diff(PR_formula, T)
    assert_close(numeric_sub_l(dV_dP), PR_obj_l.dV_dP_l)

    # Checks out with solve as well
    dT_dV = 1/dV_dT
    assert_close(numeric_sub_l(dT_dV), PR_obj_l.dT_dV_l)

    dT_dP = 1/dP_dT
    assert_close(numeric_sub_l(dT_dP), PR_obj_l.dT_dP_l)

    # Second derivatives of two variables, easy ones

    d2P_dTdV = diff(dP_dT, V)
    assert_close(numeric_sub_l(d2P_dTdV), PR_obj_l.d2P_dTdV_l)

    d2P_dTdV = diff(dP_dV, T)
    assert_close(numeric_sub_l(d2P_dTdV), PR_obj_l.d2P_dTdV_l)


    # Second derivatives of one variable, easy ones
    d2P_dT2 = diff(dP_dT, T)
    assert_close(numeric_sub_l(d2P_dT2), PR_obj_l.d2P_dT2_l)
    d2P_dT2_maple = -506.2012523140132
    assert_close(d2P_dT2_maple, PR_obj_l.d2P_dT2_l)

    d2P_dV2 = diff(dP_dV, V)
    assert_close(numeric_sub_l(d2P_dV2), PR_obj_l.d2P_dV2_l)
    d2P_dV2_maple = 4.4821628180979494e+17
    assert_close(d2P_dV2_maple, PR_obj_l.d2P_dV2_l)

    # Second derivatives of one variable, Hard ones - require a complicated identity
    d2V_dT2 = (-(d2P_dT2*dP_dV - dP_dT*d2P_dTdV)*dP_dV**-2
              +(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*dP_dV**-3*dP_dT)
    assert_close(numeric_sub_l(d2V_dT2), PR_obj_l.d2V_dT2_l)
    d2V_dT2_maple = 1.1688517647207985e-09
    assert_close(d2V_dT2_maple, PR_obj_l.d2V_dT2_l)

    d2V_dP2 = -d2P_dV2/dP_dV**3
    assert_close(numeric_sub_l(d2V_dP2), PR_obj_l.d2V_dP2_l)
    d2V_dP2_maple = 9.103364399605894e-21
    assert_close(d2V_dP2_maple, PR_obj_l.d2V_dP2_l)


    d2T_dP2 = -d2P_dT2*dP_dT**-3
    assert_close(numeric_sub_l(d2T_dP2), PR_obj_l.d2T_dP2_l)
    d2T_dP2_maple = 2.5646844439707823e-15
    assert_close(d2T_dP2_maple, PR_obj_l.d2T_dP2_l)

    d2T_dV2 = (-(d2P_dV2*dP_dT - dP_dV*d2P_dTdV)*dP_dT**-2
              +(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*dP_dT**-3*dP_dV)
    assert_close(numeric_sub_l(d2T_dV2), PR_obj_l.d2T_dV2_l)
    d2T_dV2_maple = -291578743623.6926
    assert_close(d2T_dV2_maple, PR_obj_l.d2T_dV2_l)


    # Second derivatives of two variable, Hard ones - require a complicated identity
    d2T_dPdV = -(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*dP_dT**-3
    assert_close(numeric_sub_l(d2T_dPdV), PR_obj_l.d2T_dPdV_l)
    d2T_dPdV_maple = 0.06994168125617044
    assert_close(d2T_dPdV_maple, PR_obj_l.d2T_dPdV_l)

    d2V_dPdT = -(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*dP_dV**-3
    assert_close(numeric_sub_l(d2V_dPdT), PR_obj_l.d2V_dPdT_l)
    d2V_dPdT_maple = -3.772509038556849e-15
    assert_close(d2V_dPdT_maple, PR_obj_l.d2V_dPdT_l)

    # Cv integral, real slow
    # The Cv integral is possible with a more general form, but not here
    # The S and H integrals don't work in Sympy at present


def test_PR_quick():
    # Test solution for molar volumes
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    Vs_fast = eos.volume_solutions_full(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(0.00013022212513965863+0j), (0.001123631313468268+0.0012926967234386068j), (0.001123631313468268-0.0012926967234386068j)]
    assert_close1d(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [3.801262003434438, -0.006647930535193546, 1.6930139095364687e-05]
    a_alphas_implemented = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_implemented)

    # PR back calculation for T
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00013022212513965863, P=1E6)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00013022212513965863)
    assert_close(T_slow, 299)


    diffs_1 = [582232.4757941114, -3665179372374.127, 1.5885511093471238e-07, -2.7283794281321846e-13, 6295044.54792763, 1.7175270043741416e-06]
    diffs_2 = [-506.2012523140132, 4.4821628180979494e+17, 1.1688517647207979e-09, 9.103364399605888e-21, -291578743623.6926, 2.5646844439707823e-15]
    diffs_mixed = [-3.7725090385568464e-15, -20523296734.825127, 0.06994168125617044]
    departures = [-31134.750843460362, -72.475619319576, 25.165386034971817]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]

    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        main_calcs = (main_calcs[0:6], main_calcs[6:12], main_calcs[12:15], main_calcs[15:])

        for i, j in zip(known_derivs_deps, main_calcs):
            assert_close1d(i, j)





    # Test Cp_Dep, Cv_dep
    assert_close(eos.Cv_dep_l, 25.165386034971814)
    assert_close(eos.Cp_dep_l, 44.505614171906245)

    # Exception tests
    a = GCEOS()
    with pytest.raises(Exception):
        a.a_alpha_and_derivatives_pure(T=300)

    with pytest.raises(Exception):
        PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299)


    # First third a_alpha T derivative - works fine, easy for pures anyway
    d3a_alpha_dT3_num = derivative(lambda T: eos.to(P=eos.P, T=T).d2a_alpha_dT2, eos.T, dx=eos.T*3e-8)
    assert_close(d3a_alpha_dT3_num, eos.d3a_alpha_dT3, rtol=1e-8)


    # Integration tests
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_close(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    # One gas phase property
    assert 'g' == PR(Tc=507.6, Pc=3025000, omega=0.2975, T=499.,P=1E5).phase

    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)

    B = eos.b*eos.P/R/eos.T
    A = eos.a_alpha*eos.P/(R*eos.T)**2
    D = -eos.T*eos.da_alpha_dT

    V = eos.V_l
    Z = eos.P*V/(R*eos.T)

    # Compare against some known  in Walas [2] functions
    phi_walas =  exp(Z - 1 - log(Z - B) - A/(2*2**0.5*B)*log((Z+(sqrt(2)+1)*B)/(Z-(sqrt(2)-1)*B)))
    phi_l_expect = 0.022212524527244346
    assert_close(phi_l_expect, eos.phi_l)
    assert_close(phi_walas, eos.phi_l)

    # The formula given in [2]_ must be incorrect!
#    S_dep_walas =  R*(-log(Z - B) + B*D/(2*2**0.5*A*eos.a_alpha)*log((Z+(sqrt(2)+1)*B)/(Z-(sqrt(2)-1)*B)))
#    S_dep_expect = -72.47559475426013
#    assert_close(-S_dep_walas, S_dep_expect)
#    assert_close(S_dep_expect, eos.S_dep_l)

    H_dep_walas = R*eos.T*(1 - Z + A/(2*2**0.5*B)*(1 + D/eos.a_alpha)*log((Z+(sqrt(2)+1)*B)/(Z-(sqrt(2)-1)*B)))
    H_dep_expect = -31134.750843460355
    assert_close(-H_dep_walas, H_dep_expect)
    assert_close(H_dep_expect, eos.H_dep_l)

    # Author's original H_dep, in [1]
    H_dep_orig = R*eos.T*(Z-1) + (eos.T*eos.da_alpha_dT-eos.a_alpha)/(2*2**0.5*eos.b)*log((Z+2.44*B)/(Z-0.414*B))
    assert_close(H_dep_orig, H_dep_expect, rtol=5E-3)

    # Author's correlation, with the correct constants this time
    H_dep_orig = R*eos.T*(Z-1) + (eos.T*eos.da_alpha_dT-eos.a_alpha)/(2*2**0.5*eos.b)*log((Z+(sqrt(2)+1)*B)/(Z-(sqrt(2)-1)*B))
    assert_close(H_dep_orig, H_dep_expect)

    # Test against Preos.xlsx
    # chethermo (Elliott, Richard and Lira, Carl T. - 2012 - Introductory Chemical Engineering Thermodynamics)
    # Propane
    e = PR(Tc=369.8, Pc=4.249E6, omega=0.152, T=298, P=1E5)
    assert_close(e.V_g, 0.024366034151169353)
    assert_close(e.V_l, 8.681724253858589e-05)


    # The following are imprecise as the approximate constants 2.414 etc were
    # used in chetherm
    assert_close(e.fugacity_g, 98364.041542871, rtol=1E-5)
    # not sure the problem with precision with the liquid
    assert_close(e.fugacity_l, 781433.379991859, rtol=1E-2)

    assert_close(e.H_dep_g, -111.99060081493053)
    assert_close(e.H_dep_l, -16112.7239108382, rtol=1E-5)

    assert_close(e.U_dep_g, -70.88415572220038)
    assert_close(e.U_dep_l, -13643.6966117489, rtol=1E-5)

    assert_close(e.S_dep_g, -0.23863903817819482)
    assert_close(e.S_dep_l, -71.158231517264, rtol=1E-6)

    # Volume solutions vs
    # Fallibility of analytic roots of cubic equations of state in low temperature region
    Tc = 464.80
    Pc = 35.60E5
    omega = 0.237
    # Props said to be from Reid et al

    b = PR(T=114.93, P=5.7E-6, Tc=Tc, Pc=Pc, omega=omega)
    V_max = max([V.real for V in b.raw_volumes])
    assert_close(V_max, 1.6764E8, rtol=1E-3)
    # Other two roots don't match

    # Example 05.11 Liquid Density using the Peng-Robinson EOS in Chemical Thermodynamics for Process Simulation
    V_l = PR(T=353.85, P=101325, Tc=553.60, Pc=40.750E5, omega=.2092).V_l
    assert_close(V_l, 0.00011087, atol=1e-8)
    # Matches to rounding

    # End pressure for methanol
    thing = PR(P=1e5, V=0.00014369237974317395, Tc=512.5, Pc=8084000.0, omega=0.559)
    assert_close(thing.P_max_at_V(thing.V), 2247487113.806047, rtol=1e-12)

    base = PR(Tc=367.6, Pc=302500000, omega=1.5, T=299., P=1E9)
    base.to(V=base.V_l, P=base.P_max_at_V(base.V_l)-1)
    with pytest.raises(Exception):
        base.to(V=base.V_l, P=base.P_max_at_V(base.V_l)+1)



    # solve_T quick original test
    eos = PR(Tc=658.0, Pc=1820000.0, omega=0.562, T=500., P=1e5)

    def PR_solve_T_analytical_orig(P, V, Tc, a, b, kappa):
         return Tc*(-2*a*kappa*sqrt((V - b)**3*(V**2 + 2*V*b - b**2)*(P*R*Tc*V**2 + 2*P*R*Tc*V*b - P*R*Tc*b**2 - P*V*a*kappa**2 + P*a*b*kappa**2 + R*Tc*a*kappa**2 + 2*R*Tc*a*kappa + R*Tc*a))*(kappa + 1)*(R*Tc*V**2 + 2*R*Tc*V*b - R*Tc*b**2 - V*a*kappa**2 + a*b*kappa**2)**2 + (V - b)*(R**2*Tc**2*V**4 + 4*R**2*Tc**2*V**3*b + 2*R**2*Tc**2*V**2*b**2 - 4*R**2*Tc**2*V*b**3 + R**2*Tc**2*b**4 - 2*R*Tc*V**3*a*kappa**2 - 2*R*Tc*V**2*a*b*kappa**2 + 6*R*Tc*V*a*b**2*kappa**2 - 2*R*Tc*a*b**3*kappa**2 + V**2*a**2*kappa**4 - 2*V*a**2*b*kappa**4 + a**2*b**2*kappa**4)*(P*R*Tc*V**4 + 4*P*R*Tc*V**3*b + 2*P*R*Tc*V**2*b**2 - 4*P*R*Tc*V*b**3 + P*R*Tc*b**4 - P*V**3*a*kappa**2 - P*V**2*a*b*kappa**2 + 3*P*V*a*b**2*kappa**2 - P*a*b**3*kappa**2 + R*Tc*V**2*a*kappa**2 + 2*R*Tc*V**2*a*kappa + R*Tc*V**2*a + 2*R*Tc*V*a*b*kappa**2 + 4*R*Tc*V*a*b*kappa + 2*R*Tc*V*a*b - R*Tc*a*b**2*kappa**2 - 2*R*Tc*a*b**2*kappa - R*Tc*a*b**2 + V*a**2*kappa**4 + 2*V*a**2*kappa**3 + V*a**2*kappa**2 - a**2*b*kappa**4 - 2*a**2*b*kappa**3 - a**2*b*kappa**2))/((R*Tc*V**2 + 2*R*Tc*V*b - R*Tc*b**2 - V*a*kappa**2 + a*b*kappa**2)**2*(R**2*Tc**2*V**4 + 4*R**2*Tc**2*V**3*b + 2*R**2*Tc**2*V**2*b**2 - 4*R**2*Tc**2*V*b**3 + R**2*Tc**2*b**4 - 2*R*Tc*V**3*a*kappa**2 - 2*R*Tc*V**2*a*b*kappa**2 + 6*R*Tc*V*a*b**2*kappa**2 - 2*R*Tc*a*b**3*kappa**2 + V**2*a**2*kappa**4 - 2*V*a**2*b*kappa**4 + a**2*b**2*kappa**4))
    T_analytical = PR_solve_T_analytical_orig(eos.P, eos.V_g, eos.Tc, eos.a, eos.b, eos.kappa)
    assert_close(T_analytical, eos.solve_T(P=eos.P, V=eos.V_g), rtol=1e-13)

def test_lnphi_l_low_TP():
    # Was failing because of an underflow
    eos = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=1.0, P=0.00013848863713938732)
    assert_close(eos.lnphi_l, -5820.389433322589, rtol=1e-10)

@pytest.mark.slow
@pytest.mark.fuzz
def test_PR_high_pressure_not_always():
    for T in linspace(10, 5000, 25):
        for P in logspace(log10(.01*4872000.0), log10(4872000.0*1000), 25):
            base = PR(Tc=305.32, Pc=4872000.0, omega=0.098, T=T, P=P)
            try:
                V = base.V_l
            except:
                V = base.V_g

            P_max = base.P_max_at_V(V)
            assert P_max is None
            PR(V=V, P=1e30, Tc=base.Tc, Pc=base.Pc, omega=base.omega)


def test_PR_second_partial_derivative_shims():
    # Check the shims for the multi variate derivatives
    T = 400
    P = 1E6
    crit_params = {'Tc': 507.6, 'Pc': 3025000.0, 'omega': 0.2975}
    eos = PR(T=T, P=P, **crit_params)

    assert_close(1.0/eos.V_l, eos.rho_l)
    assert_close(1.0/eos.V_g, eos.rho_g)

    assert_close(eos.d2V_dTdP_l, eos.d2V_dPdT_l)
    assert_close(eos.d2V_dTdP_g, eos.d2V_dPdT_g)

    assert_close(eos.d2P_dVdT_l, eos.d2P_dTdV_l)
    assert_close(eos.d2P_dVdT_g, eos.d2P_dTdV_g)

    assert_close(eos.d2T_dVdP_l, eos.d2T_dPdV_l)
    assert_close(eos.d2T_dVdP_g, eos.d2T_dPdV_g)


def test_PR_density_derivatives():
    # TODO speed up big time - takes 2 seconds!
    '''
    '''
    '''Sympy expressions:

    >>> f = 1/x
    >>> f
    1/x
    >>> diff(g(x), x)/diff(f, x)
    -x**2*Derivative(g(x), x)
    >>> diff(diff(g(x), x)/diff(f, x), x)/diff(f, x)
    -x**2*(-x**2*Derivative(g(x), x, x) - 2*x*Derivative(g(x), x))


    >>> diff(1/f(x), x)
    -Derivative(f(x), x)/f(x)**2
    >>> diff(diff(1/f(x), x), x)
    -Derivative(f(x), x, x)/f(x)**2 + 2*Derivative(f(x), x)**2/f(x)**3


    >>> diff(diff(1/f(x, y), x), y)
    -Derivative(f(x, y), x, y)/f(x, y)**2 + 2*Derivative(f(x, y), x)*Derivative(f(x, y), y)/f(x, y)**3
    '''
    # Test solution for molar volumes
    T = 400
    P = 1E6
    crit_params = {'Tc': 507.6, 'Pc': 3025000.0, 'omega': 0.2975}
    eos = PR(T=T, P=P, **crit_params)

    assert_close(eos.dP_drho_l, 16717.849183154118)
    assert_close(eos.dP_drho_g, 1086.6585648335506)

    assert_close(eos.drho_dP_g, 0.0009202522598744487)
    assert_close(eos.drho_dP_l, 5.981630705268346e-05)

    V = 'V_g'
    def d_rho_dP(P):
        return 1/getattr(PR(T=T, P=P, **crit_params), V)
    assert_close(eos.drho_dP_g, derivative(d_rho_dP, P, order=3))
    V = 'V_l'
    assert_close(eos.drho_dP_l, derivative(d_rho_dP, P), rtol=1e-5)

    # f = 1/x
    # >>> diff(diff(g(x), x)/diff(f, x), x)/diff(f, x)
    # -x**2*(-x**2*Derivative(g(x), x, x) - 2*x*Derivative(g(x), x))
    assert_close(eos.d2P_drho2_l, 22.670941865204274)
    assert_close(eos.d2P_drho2_g, -4.025199354171817)

    # Numerical tests
    d2rho_dP2_l = eos.d2rho_dP2_l
    d2rho_dP2_g = eos.d2rho_dP2_g
    assert_close(d2rho_dP2_l, -4.852084485170919e-12)
    assert_close(d2rho_dP2_g, 3.136953435966285e-09)

    def d_rho_dP_second(P, state):
        e = PR(T=T, P=P, **crit_params)
        if state == 'l':
            return e.drho_dP_l
        else:
            return e.drho_dP_g

    d2rho_dP2_l_num = derivative(d_rho_dP_second, 1e6, dx=100, order=3, args=('l'))
    d2rho_dP2_g_num = derivative(d_rho_dP_second, 1e6, dx=1, order=3, args=('g'))
    assert_close(d2rho_dP2_l_num, d2rho_dP2_l)
    assert_close(d2rho_dP2_g_num, d2rho_dP2_g)

    # dT_drho tests - analytical
    assert_close(eos.dT_drho_l, -0.05794715601744427)
    assert_close(eos.dT_drho_g, -0.2115805039516506)
    assert_close(eos.d2T_drho2_l, -2.7171270001507268e-05)
    assert_close(eos.d2T_drho2_g, 0.0019160984645184406)

    def dT_drho(rho):
        return PR(V=1.0/rho, P=P, **crit_params).T

    ans_numeric = derivative(dT_drho, 1/eos.V_l, n=1, dx=1e-2)
    assert_close(eos.dT_drho_l, ans_numeric, rtol=1e-5)

    ans_numeric = derivative(dT_drho, 1/eos.V_g, n=1, dx=1e-2)
    assert_close(eos.dT_drho_g, ans_numeric)

    def dT_drho_second(rho):
        e = PR(V=1.0/rho, P=P, **crit_params)
        try:
            return e.dT_drho_l
        except AttributeError:
            return e.dT_drho_g
    ans_numeric = derivative(dT_drho_second, 1.0/eos.V_l, n=1, dx=1, order=3)
    assert_close(eos.d2T_drho2_l, ans_numeric)

    ans_numeric = derivative(dT_drho_second, 1/eos.V_g, n=1, dx=.01, order=3)
    assert_close(eos.d2T_drho2_g, ans_numeric)

    # drho_dT tests - analytical
    def drho_dT(T, V='V_l'):
        return 1.0/getattr(PR(T=T, P=P, **crit_params), V)

    assert_close(eos.drho_dT_g, -4.726333387638189)
    assert_close(eos.drho_dT_l, -17.25710231057694)

    ans_numeric = derivative(drho_dT, eos.T, n=1, dx=1e-2, args=['V_l'])
    assert_close(ans_numeric, eos.drho_dT_l)

    ans_numeric = derivative(drho_dT, eos.T, n=1, dx=1e-2, args=['V_g'])
    assert_close(ans_numeric, eos.drho_dT_g)


    def drho_dT_second(T, state='l'):
        e = PR(T=T, P=P, **crit_params)
        if state == 'l':
            return e.drho_dT_l
        else:
            return e.drho_dT_g

    d2rho_dT2_l = eos.d2rho_dT2_l
    d2rho_dT2_g = eos.d2rho_dT2_g
    assert_close(d2rho_dT2_l, -0.13964119596352564)
    assert_close(d2rho_dT2_g, 0.20229767021600575)

    ans_numeric = derivative(drho_dT_second, eos.T, n=1, dx=3e-2, args=['l'])
    assert_close(d2rho_dT2_l, ans_numeric)
    ans_numeric = derivative(drho_dT_second, eos.T, n=1, dx=1e-3, args=['g'])
    assert_close(d2rho_dT2_g, ans_numeric)

    # Sympy and numerical derivative quite agree!
    # d2P_drho_dT
    def dP_drho_to_diff(T, rho):
        e = PR(T=T, V=1/rho, **crit_params)
        try:
            return e.dP_drho_l
        except AttributeError:
            return e.dP_drho_g

    ans_numerical = derivative(dP_drho_to_diff, eos.T, dx=eos.T*1e-6, args=(1.0/eos.V_l,))
    d2P_dTdrho_l = eos.d2P_dTdrho_l
    assert_close(d2P_dTdrho_l, 121.15490035272644)
    assert_close(d2P_dTdrho_l, ans_numerical)

    ans_numerical = derivative(dP_drho_to_diff, eos.T, dx=eos.T*1e-6, args=(1.0/eos.V_g,))
    d2P_dTdrho_g = eos.d2P_dTdrho_g
    assert_close(d2P_dTdrho_g, 13.513868196361557)
    assert_close(d2P_dTdrho_g, ans_numerical)

    # Numerical derivatives agree - d2T_dPdrho_g
    def dT_drho(rho, P):
        return PR(P=P, V=1.0/rho, **crit_params).T

    def d2T_drhodP(P, rho):
        def to_diff(P):
            e = PR(P=P, V=1.0/rho, **crit_params)
            try:
                return e.dT_drho_l
            except:
                return e.dT_drho_g
        return derivative(to_diff, P, n=1, dx=100, order=3)

    ans_numerical = d2T_drhodP(eos.P, 1/eos.V_l)
    assert_close(ans_numerical, eos.d2T_dPdrho_l, rtol=2e-8)
    assert_close(eos.d2T_dPdrho_l, -1.6195726310475797e-09)

    ans_numerical = d2T_drhodP(eos.P, 1/eos.V_g)
    assert_close(ans_numerical, eos.d2T_dPdrho_g)
    assert_close(eos.d2T_dPdrho_g, -5.29734819740022e-07)

    # drho_DT_dP derivatives
    def drho_dT(T, P, V='V_l'):
        return 1.0/getattr(PR(T=T, P=P, **crit_params), V)

    def drho_dT_dP(T, P, V='V_l'):
        if V == 'V_l':
            rho = eos.rho_l
        else:
            rho = eos.rho_g
        def to_dP(P):
            e = PR(P=P, T=T, **crit_params)
            if V == 'V_l':
                return e.drho_dT_l
            else:
                return e.drho_dT_g
        return derivative(to_dP, P, n=1, dx=30, order=3)

    ans_numerical = drho_dT_dP(eos.T, eos.P, 'V_l')
    assert_close(9.66343207820545e-07, eos.d2rho_dPdT_l)
    assert_close(ans_numerical, eos.d2rho_dPdT_l, rtol=1e-5)

    ans_numerical = drho_dT_dP(eos.T, eos.P, 'V_g')
    assert_close(ans_numerical, eos.d2rho_dPdT_g, rtol=1e-6)
    assert_close(-2.755552405262765e-05, eos.d2rho_dPdT_g)

def test_PR_Psat():
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    Cs_PR = [-3.3466262, -9.9145207E-02, 1.015969390, -1.032780679,
             0.2904927517, 1.64073501E-02, -9.67894565E-03, 1.74161549E-03,
             -1.56974110E-04, 5.87311295E-06]
    def Psat(T, Tc, Pc, omega):
        Tr = T/Tc
        e = PR(Tc=Tc, Pc=Pc, omega=omega, T=T, P=1E5)
        alpha = e.a_alpha/e.a
        tot = 0
        for k, Ck in enumerate(Cs_PR[0:4]):
            tot += Ck*(alpha/Tr-1)**((k+2)/2.)
        for k, Ck in enumerate(Cs_PR[4:]):
            tot += Ck*(alpha/Tr-1)**(k+3)
        P = exp(tot)*Tr*Pc
        return P

    Ts = linspace(507.6*0.32, 504)
    Psats_lit = [Psat(T, Tc=507.6, Pc=3025000, omega=0.2975) for T in Ts]
    Psats_eos = [eos.Psat(T) for T in Ts]
    assert_close1d(Psats_lit, Psats_eos, rtol=1e-4)

    # Check that fugacities exist for both phases
    for T, P in zip(Ts, Psats_eos):
        eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=T, P=P)
        assert_close(eos.fugacity_l, eos.fugacity_g, rtol=1e-7)



def test_PR78():
    eos = PR78(Tc=632, Pc=5350000, omega=0.734, T=299., P=1E6)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [8.35196289693885e-05, -63764.67109328409, -130.7371532254518]
    assert_close1d(three_props, expect_props)

    # Test the results are identical to PR or lower things
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    PR_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    eos = PR78(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    PR78_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    assert_close1d(PR_props, PR78_props)


def test_PRSV():
    eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.0001301269135543934, -31698.926746698795, -74.16751538228138]
    assert_close1d(three_props, expect_props)

    # Test of a_alphas
    a_alphas = [3.812985698311453, -0.006976903474851659, 2.0026560811043733e-05]

    a_alphas_implemented = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_implemented)
    T, Tc, a, kappa0, kappa1 = eos.T, eos.Tc, eos.a, eos.kappa0, eos.kappa1
    a_alpha = a*((kappa0 + kappa1*(sqrt(T/Tc) + 1)*(-T/Tc + 0.7))*(-sqrt(T/Tc) + 1) + 1)**2
    da_alpha_dT = a*((kappa0 + kappa1*(sqrt(T/Tc) + 1)*(-T/Tc + 0.7))*(-sqrt(T/Tc) + 1) + 1)*(2*(-sqrt(T/Tc) + 1)*(-kappa1*(sqrt(T/Tc) + 1)/Tc + kappa1*sqrt(T/Tc)*(-T/Tc + 0.7)/(2*T)) - sqrt(T/Tc)*(kappa0 + kappa1*(sqrt(T/Tc) + 1)*(-T/Tc + 0.7))/T)
    d2a_alpha_dT2 = a*((kappa1*(sqrt(T/Tc) - 1)*(20*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(10*T/Tc - 7)/T) - sqrt(T/Tc)*(10*kappa0 - kappa1*(sqrt(T/Tc) + 1)*(10*T/Tc - 7))/T)**2 - sqrt(T/Tc)*((10*kappa0 - kappa1*(sqrt(T/Tc) + 1)*(10*T/Tc - 7))*(sqrt(T/Tc) - 1) - 10)*(kappa1*(40/Tc - (10*T/Tc - 7)/T)*(sqrt(T/Tc) - 1) + 2*kappa1*(20*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(10*T/Tc - 7)/T) + (10*kappa0 - kappa1*(sqrt(T/Tc) + 1)*(10*T/Tc - 7))/T)/T)/200
    assert_close1d(a_alphas_implemented, (a_alpha, da_alpha_dT, d2a_alpha_dT2), rtol=1e-13)


    # PR back calculation for T
    eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, V=0.0001301269135543934, P=1E6, kappa1=0.05104)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001301269135543934)
    assert_close(T_slow, 299)


    # Test the bool to control its behavior
    eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=406.08, P=1E6, kappa1=0.05104)
    assert_close(eos.kappa, 0.7977689278061457)
    eos.kappa1_Tr_limit = True
    eos.__init__(Tc=507.6, Pc=3025000, omega=0.2975, T=406.08, P=1E6, kappa1=0.05104)
    assert_close(eos.kappa, 0.8074380841890093)

    # Test the limit is not enforced while under Tr =0.7
    eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=304.56, P=1E6, kappa1=0.05104)
    assert_close(eos.kappa, 0.8164956255888178)
    eos.kappa1_Tr_limit = True
    eos.__init__(Tc=507.6, Pc=3025000, omega=0.2975, T=304.56, P=1E6, kappa1=0.05104)
    assert_close(eos.kappa, 0.8164956255888178)

    with pytest.raises(Exception):
        PRSV(Tc=507.6, Pc=3025000, omega=0.2975, P=1E6, kappa1=0.05104)

    # One solve_T that did not work
    test = PRSV(P=1e16, V=0.3498789873827434, Tc=507.6, Pc=3025000.0, omega=0.2975)
    assert_close(test.T, 421177338800932.0)




def test_PRSV2():
    eos = PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104, kappa2=0.8634, kappa3=0.460)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.00013018825759153257, -31496.184168729033, -73.6152829631142]
    assert_close1d(three_props, expect_props)

    # Test of PRSV2 a_alphas
    a_alphas = [3.80542021117275, -0.006873163375791913, 2.3078023705053787e-05]

    a_alphas_implemented = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_implemented)


    T, Tc, a, kappa0, kappa1, kappa2, kappa3 = eos.T, eos.Tc, eos.a, eos.kappa0, eos.kappa1, eos.kappa2, eos.kappa3
    Tr = T/Tc
    sqrtTr = sqrt(Tr)
    kappa =  kappa0 + ((kappa1 + kappa2*(kappa3 - Tr)*(1.0 - sqrtTr))*(1.0 + sqrtTr)*(0.7 - Tr))
    a_alpha = a*(1 + kappa*(1-sqrt(T/Tc)))**2
    da_alpha_dT = a*((kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))*(-sqrt(T/Tc) + 1) + 1)*(2*(-sqrt(T/Tc) + 1)*((sqrt(T/Tc) + 1)*(-T/Tc + 7/10)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T)) - (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(-T/Tc + 7/10)/(2*T)) - sqrt(T/Tc)*(kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))/T)
    d2a_alpha_dT2 = a*((kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))*(-sqrt(T/Tc) + 1) + 1)*(2*(-sqrt(T/Tc) + 1)*((sqrt(T/Tc) + 1)*(-T/Tc + 7/10)*(kappa2*sqrt(T/Tc)/(T*Tc) + kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(4*T**2)) - 2*(sqrt(T/Tc) + 1)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T))/Tc + sqrt(T/Tc)*(-T/Tc + 7/10)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T))/T - sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))/(T*Tc) - sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(-T/Tc + 7/10)/(4*T**2)) - 2*sqrt(T/Tc)*((sqrt(T/Tc) + 1)*(-T/Tc + 7/10)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T)) - (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(-T/Tc + 7/10)/(2*T))/T + sqrt(T/Tc)*(kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))/(2*T**2)) + a*((-sqrt(T/Tc) + 1)*((sqrt(T/Tc) + 1)*(-T/Tc + 7/10)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T)) - (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(-T/Tc + 7/10)/(2*T)) - sqrt(T/Tc)*(kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))/(2*T))*(2*(-sqrt(T/Tc) + 1)*((sqrt(T/Tc) + 1)*(-T/Tc + 7/10)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T)) - (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(-T/Tc + 7/10)/(2*T)) - sqrt(T/Tc)*(kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))/T)
    assert_close1d(a_alphas_implemented, (a_alpha, da_alpha_dT, d2a_alpha_dT2), rtol=1e-13)

    # PSRV2 back calculation for T
    eos = PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00013018825759153257, P=1E6, kappa1=0.05104, kappa2=0.8634, kappa3=0.460)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00013018825759153257)
    assert_close(T_slow, 299)

    # Check this is the same as PRSV
    eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104)
    three_props_PRSV = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    eos = PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104)
    three_props_PRSV2 = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    assert_close1d(three_props_PRSV, three_props_PRSV2)

    with pytest.raises(Exception):
        PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, T=299.)


def test_VDW():
    eos = VDW(Tc=507.6, Pc=3025000, T=299., P=1E6)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.00022332985608164609, -13385.727374687076, -32.65923125080434]
    assert_close1d(three_props, expect_props)

    # Test of a_alphas
    a_alphas = [2.4841053385218554, 0.0, 0.0]
    a_alphas_fast = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_fast)

    # Back calculation for P
    eos = VDW(Tc=507.6, Pc=3025000, T=299, V=0.00022332985608164609)
    assert_close(eos.P, 1E6)

    # Back calculation for T
    eos = VDW(Tc=507.6, Pc=3025000, P=1E6, V=0.00022332985608164609)
    assert_close(eos.T, 299)

    with pytest.raises(Exception):
        VDW(Tc=507.6, Pc=3025000, P=1E6)


def test_VDW_Psat():
    eos = VDW(Tc=507.6, Pc=3025000,  T=299., P=1E6)
    Cs_VDW = [-2.9959015, -4.281688E-2, 0.47692435, -0.35939335, -2.7490208E-3,
              4.4205329E-2, -1.18597319E-2, 1.74962842E-3, -1.41793758E-4,
              4.93570180E-6]

    def Psat(T, Tc, Pc, omega):
        Tr = T/Tc
        e = VDW(Tc=Tc, Pc=Pc, T=T, P=1E5)
        alpha = e.a_alpha/e.a
        tot = 0
        for k, Ck in enumerate(Cs_VDW[0:4]):
            tot += Ck*(alpha/Tr-1)**((k+2)/2.)
        for k, Ck in enumerate(Cs_VDW[4:]):
            tot += Ck*(alpha/Tr-1)**(k+3)
        P = exp(tot)*Tr*Pc
        return P

    Ts = linspace(507.6*.32, 506)
    Psats_lit = [Psat(T, Tc=507.6, Pc=3025000, omega=0.2975) for T in Ts]
    Psats_eos = [eos.Psat(T) for T in Ts]
    assert_close1d(Psats_lit, Psats_eos, rtol=2E-5)

    # Check that fugacities exist for both phases
    for T, P in zip(Ts, Psats_eos):
        eos = VDW(Tc=507.6, Pc=3025000, T=T, P=P)
        assert_close(eos.fugacity_l, eos.fugacity_g, rtol=1E-8)


def test_RK_quick():
    # Test solution for molar volumes
    eos = RK(Tc=507.6, Pc=3025000, T=299., P=1E6)
    Vs_fast = eos.volume_solutions_full(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(0.00015189346878119082+0j), (0.0011670654270233137+0.001117116441729614j), (0.0011670654270233137-0.001117116441729614j)]
    assert_close1d(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [3.279649770989796, -0.005484364165534776, 2.7513532603017274e-05]

    a_alphas_fast = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_fast)

    # PR back calculation for T
    eos = RK(Tc=507.6, Pc=3025000,  V=0.00015189346878119082, P=1E6)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00015189346878119082)
    assert_close(T_slow, 299)


    diffs_1 = [400451.91036588483, -1773162956091.739, 2.258404446078257e-07, -5.63963958622347e-13, 4427904.849977206, 2.497178747596235e-06]
    diffs_2 = [-664.0592454189463, 1.5385254880211363e+17, 1.50351759964441e-09, 2.759680128116916e-20, -130527901462.75568, 1.034083761001255e-14]
    diffs_mixed = [-7.87047555851423e-15, -10000511760.82874, 0.08069819844971812]
    departures = [-26160.84248778514, -63.01313785205201, 39.84399938752266]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]

    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        main_calcs = (main_calcs[0:6], main_calcs[6:12], main_calcs[12:15], main_calcs[15:])
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_close1d(i, j)





    # Test Cp_Dep, Cv_dep
    assert_close(eos.Cv_dep_l, 39.8439993875226)
    assert_close(eos.Cp_dep_l, 58.57056977621366)

    # Integration tests
    eos = RK(Tc=507.6, Pc=3025000, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_close(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    # One gas phase property
    assert 'g' == RK(Tc=507.6, Pc=3025000, T=499.,P=1E5).phase



    # Compare against some known  in Walas [2] functions
    eos = RK(Tc=507.6, Pc=3025000, T=299., P=1E6)
    V = eos.V_l
    Z = eos.P*V/(R*eos.T)

    phi_walas = exp(Z - 1 - log(Z*(1 - eos.b/V)) - eos.a*eos.Tc**0.5/(eos.b*R*eos.T**1.5)*log(1 + eos.b/V))
    phi_l_expect = 0.052632270169019224
    assert_close(phi_l_expect, eos.phi_l)
    assert_close(phi_walas, eos.phi_l)

    S_dep_walas = -R*(log(Z*(1 - eos.b/V)) - eos.a*eos.Tc**0.5/(2*eos.b*R*eos.T**1.5)*log(1 + eos.b/V))
    S_dep_expect = -63.01313785205201
    assert_close(-S_dep_walas, S_dep_expect)
    assert_close(S_dep_expect, eos.S_dep_l)

    H_dep_walas = R*eos.T*(1 - Z + 1.5*eos.a*eos.Tc**0.5/(eos.b*R*eos.T**1.5)*log(1 + eos.b/V))
    H_dep_expect = -26160.84248778514
    assert_close(-H_dep_walas, H_dep_expect)
    assert_close(H_dep_expect, eos.H_dep_l)


    # Poling table 4.9 edition 5 - molar volumes of vapor and liquid bang on
    T = 300
    kwargs = dict(Tc=369.83, Pc=4248000.0, omega=0.152, )
    Psat = 997420
    base = RK(T=T, P=Psat, **kwargs)
    assert_close(base.V_g, 0.002085, atol=.000001)
    assert 0.0001014 == round(base.V_l, 7)

def test_RK_Psat():
    eos = RK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)

    Ts = linspace(507.6*0.32, 504, 15)
    Psats_eos = [eos.Psat(T) for T in Ts]
    fugacity_ls, fugacity_gs = [], []
    for T, P in zip(Ts, Psats_eos):
        eos = RK(Tc=507.6, Pc=3025000, omega=0.2975, T=T, P=P)
        fugacity_ls.append(eos.fugacity_l)
        fugacity_gs.append(eos.fugacity_g)

    # Fit is very good
    assert_close1d(fugacity_ls, fugacity_gs, rtol=1e-7)


def test_SRK_quick():
    # Test solution for molar volumes
    eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    Vs_fast = eos.volume_solutions_full(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [0.0001468210773547258, (0.0011696016227365465+0.001304089515440735j), (0.0011696016227365465-0.001304089515440735j)]
    assert_close1d(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [3.72718144448615, -0.007332994130304654, 1.9476133436500582e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_fast)

    # PR back calculation for T
    eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, V=0.0001468210773547258, P=1E6)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001468210773547258)
    assert_close(T_slow, 299)

    # Derivatives
    diffs_1 = [507071.3781579619, -2693848855910.7515, 1.882330469452149e-07, -3.7121607539555683e-13, 5312563.421932225, 1.97210894377967e-06]
    diffs_2 = [-495.5254299681785, 2.685151838840304e+17, 1.3462644444996599e-09, 1.3735648667748024e-20, -201856509533.585, 3.800656805086307e-15]
    diffs_mixed = [-4.991348993006751e-15, -14322101736.003756, 0.06594010907198579]
    departures = [-31754.66385964974, -74.3732720444703, 28.936530624645115]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]

    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        main_calcs = (main_calcs[0:6], main_calcs[6:12], main_calcs[12:15], main_calcs[15:])
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_close1d(i, j)


    # Integration tests
    eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_close(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]


    # Compare against some known  in Walas [2] functions
    eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    V = eos.V_l
    Z = eos.P*V/(R*eos.T)
    D = -eos.T*eos.da_alpha_dT

    S_dep_walas = R*(-log(Z*(1-eos.b/V)) + D/(eos.b*R*eos.T)*log(1 + eos.b/V))
    S_dep_expect = -74.3732720444703
    assert_close(-S_dep_walas, S_dep_expect)
    assert_close(S_dep_expect, eos.S_dep_l)

    H_dep_walas = eos.T*R*(1 - Z + 1/(eos.b*R*eos.T)*(eos.a_alpha+D)*log(1 + eos.b/V))
    H_dep_expect = -31754.663859649743
    assert_close(-H_dep_walas, H_dep_expect)
    assert_close(H_dep_expect, eos.H_dep_l)

    phi_walas = exp(Z - 1 - log(Z*(1 - eos.b/V)) - eos.a_alpha/(eos.b*R*eos.T)*log(1 + eos.b/V))
    phi_l_expect = 0.02174822767621325
    assert_close(phi_l_expect, eos.phi_l)
    assert_close(phi_walas, eos.phi_l)

def test_SRK_Psat():
    eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)

    # ERROR actually for RK not SRK
    Cs_SRK = [-3.0486334, -5.2157649E-2, 0.55002312, -0.44506984, 3.1735078E-2,
              4.1819219E-2, -1.18709865E-2, 1.79267167E-3, -1.47491666E-4,
              5.19352748E-6]

    def Psat(T, Tc, Pc, omega):
        Tr = T/Tc
        e = SRK(Tc=Tc, Pc=Pc, omega=omega, T=T, P=1E5)
        alpha = e.a_alpha/e.a
        tot = 0
        for k, Ck in enumerate(Cs_SRK[0:4]):
            tot += Ck*(alpha/Tr-1)**((k+2)/2.)
        for k, Ck in enumerate(Cs_SRK[4:]):
            tot += Ck*(alpha/Tr-1)**(k+3)
        P = exp(tot)*Tr*Pc
        return P

    Ts = linspace(160, 504, 30)
    Psats_lit = [Psat(T, Tc=507.6, Pc=3025000, omega=0.2975) for T in Ts]
    Psats_eos = [eos.Psat(T) for T in Ts]
    assert_close1d(Psats_lit, Psats_eos, rtol=8E-5)

    fugacity_ls, fugacity_gs = [], []
    for T, P in zip(Ts, Psats_eos):
        eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=T, P=P)
        fugacity_ls.append(eos.fugacity_l)
        fugacity_gs.append(eos.fugacity_g)

    assert_close1d(fugacity_ls, fugacity_gs, rtol=1e-9)


def test_APISRK_quick():
    # Test solution for molar volumes
    eos = APISRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    Vs_fast = eos.volume_solutions_full(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(0.00014681828835112518+0j), (0.0011696030172383468+0.0013042038361510636j), (0.0011696030172383468-0.0013042038361510636j)]
    assert_close1d(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [3.727476773890392, -0.007334914894987986, 1.948255305988373e-05]
    a_alphas_implemented = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_implemented)

    a, Tc, S1, S2, T = eos.a, eos.Tc, eos.S1, eos.S2, eos.T
    a_alpha = a*(S1*(-sqrt(T/Tc) + 1) + S2*(-sqrt(T/Tc) + 1)/sqrt(T/Tc) + 1)**2
    da_alpha_dT = a*((S1*(-sqrt(T/Tc) + 1) + S2*(-sqrt(T/Tc) + 1)/sqrt(T/Tc) + 1)*(-S1*sqrt(T/Tc)/T - S2/T - S2*(-sqrt(T/Tc) + 1)/(T*sqrt(T/Tc))))
    d2a_alpha_dT2 = a*(((S1*sqrt(T/Tc) + S2 - S2*(sqrt(T/Tc) - 1)/sqrt(T/Tc))**2 - (S1*sqrt(T/Tc) + 3*S2 - 3*S2*(sqrt(T/Tc) - 1)/sqrt(T/Tc))*(S1*(sqrt(T/Tc) - 1) + S2*(sqrt(T/Tc) - 1)/sqrt(T/Tc) - 1))/(2*T**2))
    assert_close1d((a_alpha, da_alpha_dT, d2a_alpha_dT2), a_alphas_implemented)

    # SRK back calculation for T
    eos = APISRK(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00014681828835112518, P=1E6)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00014681828835112518)
    assert_close(T_slow, 299)
    # with a S1 set
    eos = APISRK(Tc=514.0, Pc=6137000.0, S1=1.678665, S2=-0.216396, P=1E6, V=7.045695070282895e-05)
    assert_close(eos.T, 299)
    eos = APISRK(Tc=514.0, Pc=6137000.0, omega=0.635, S2=-0.216396, P=1E6, V=7.184693818446427e-05)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=7.184693818446427e-05)
    assert_close(T_slow, 299)


    eos = APISRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    # Derivatives
    diffs_1 = [507160.1972586132, -2694518622391.442, 1.882192214387065e-07, -3.7112380359519615e-13, 5312953.652428371, 1.9717635678142066e-06]
    diffs_2 = [-495.70334320516105, 2.6860475503881738e+17, 1.3462140892058852e-09, 1.3729987070697146e-20, -201893442624.3192, 3.8000241940176305e-15]
    diffs_mixed = [-4.990229443299593e-15, -14325363284.978655, 0.06593412205681573]
    departures = [-31759.40804708375, -74.3842308177361, 28.9464819026358]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]

    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        main_calcs = (main_calcs[0:6], main_calcs[6:12], main_calcs[12:15], main_calcs[15:])
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_close1d(i, j)

    # Test Cp_Dep, Cv_dep
    assert_close(eos.Cv_dep_l, 28.9464819026358)
    assert_close(eos.Cp_dep_l, 49.17375122882494)

    # Integration tests
    eos = APISRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_close(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    # Error checking
    with pytest.raises(Exception):
        APISRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299.)
    with pytest.raises(Exception):
        APISRK(Tc=507.6, Pc=3025000, P=1E6,  T=299.)

    with pytest.raises(Exception):
        # No T solution
        APISRK(Tc=512.5, Pc=8084000.0, omega=0.559, P=1e9, V=0.00017556778406251403)


def test_TWUPR_quick():
    # Test solution for molar volumes
    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    Vs_fast = eos.volume_solutions_full(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [0.0001301755417057077, (0.0011236546051852435+0.001294926236567151j), (0.0011236546051852435-0.001294926236567151j)]
    assert_close1d(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [3.8069848647566698, -0.006971714700883658, 2.366703486824857e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_fast)

    # back calculation for T
    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, V=0.0001301755417057077, P=1E6)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001301755417057077)
    assert_close(T_slow, 299)


    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    # Derivatives
    diffs_1 = [592877.7698667824, -3683684905961.741, 1.6094692814449423e-07, -2.7146730122915294e-13, 6213228.245662597, 1.6866883037707698e-06]
    diffs_2 = [-708.1014081968287, 4.512485403434166e+17, 1.1685466035091765e-09, 9.027518486599707e-21, -280283776931.3797, 3.3978167906790706e-15]
    diffs_mixed = [-3.823707450118526e-15, -20741136287.632187, 0.0715233066523022]
    departures = [-31652.73712017438, -74.1128504294285, 35.18913741045412]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]

    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        main_calcs = (main_calcs[0:6], main_calcs[6:12], main_calcs[12:15], main_calcs[15:])
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_close1d(i, j)

    # Test Cp_Dep, Cv_dep
    assert_close(eos.Cv_dep_l, 35.18913741045409)
    assert_close(eos.Cp_dep_l, 55.40580968404073)

    # Integration tests
    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_close(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    # Error checking
    with pytest.raises(Exception):
        TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=299.)

    # Superctitical test
    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=900., P=1E6)
    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, V=0.007371700581036866, P=1E6)
    assert_close(eos.T, 900)

    assert None == TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, V=0.0001301755417057077, P=1E6).P_max_at_V(.0001301755417057077)




def test_TWUSRK_quick():
    # Test solution for molar volumes
    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    Vs_fast = eos.volume_solutions_full(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_expected = [(0.00014689222296622437+0j), (0.001169566049930797+0.0013011782630948804j), (0.001169566049930797-0.0013011782630948804j)]
    assert_close1d(Vs_fast, Vs_expected)

    # Test of a_alphas
    a_alphas = [3.7196696151053654, -0.00726972623757774, 2.305590221826195e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_fast)

    # back calculation for T
    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00014689222296622437, P=1E6)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00014689222296622437)
    assert_close(T_slow, 299)


    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    # Derivatives
    diffs_1 = [504446.40946384973, -2676840643946.8457, 1.8844842729228471e-07, -3.7357472222386695e-13, 5306491.618786468, 1.982371132471433e-06]
    diffs_2 = [-586.1645169279949, 2.6624340439193766e+17, 1.308862239605917e-09, 1.388069796850075e-20, -195576372405.25787, 4.5664049232057565e-15]
    diffs_mixed = [-5.015405580586871e-15, -14235383353.785719, 0.0681656809901603]
    departures = [-31612.602587050424, -74.0229660932213, 34.24267346218354]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]

    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        main_calcs = (main_calcs[0:6], main_calcs[6:12], main_calcs[12:15], main_calcs[15:])
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_close1d(i, j)

    # Test Cp_Dep, Cv_dep
    assert_close(eos.Cv_dep_l, 34.242673462183554)
    assert_close(eos.Cp_dep_l, 54.35178846652431)

    # Integration tests
    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_close(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    # Error checking
    with pytest.raises(Exception):
        TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299.)
    from thermo.eos_alpha_functions import TWU_a_alpha_common
    with pytest.raises(Exception):
        TWU_a_alpha_common(299.0, 507.6, 0.2975, 2.5171086468571824, method='FAIL')

    # Superctitical test
    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=900., P=1E6)
    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, V=0.007422212960199866, P=1E6)
    assert_close(eos.T, 900)

    assert None == TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, V=0.0001301755417057077, P=1E6).P_max_at_V(.0001301755417057077)


def test_PRTranslatedConsistent():
    eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.0001310369170127519, -31475.233165751422, -73.54457346552005]
    assert_close1d(three_props, expect_props)

    # Test of a_alphas
    a_alphas = [3.805629668918672, -0.0068587409608788265, 2.1778830141804843e-05]

    a_alphas_fast = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_fast)

    # back calculation for T
    eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, V=0.0001310369170127519, P=1E6)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001310369170127519)
    assert_close(T_slow, 299)

    # TV solve for P
    eos = eos.to(T=eos.T, V=eos.V_l)
    assert_close(eos.P, 1e6, rtol=1e-12)


    # Two correlations
    # Test the bool to control its behavior results in the same conditions
    eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=406.08, P=1E6)
    assert_close(eos.c, -8.503625575609944e-07, rtol=1e-12)
    assert_close1d([0.27877755625, 0.8266271, 2.0], eos.alpha_coeffs, rtol=1e-12)

    # Test overwritting c
    c_force = 0.6390E-6
    eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, c=c_force)
    assert_close(eos.c, c_force, rtol=1e-12)
    assert_close(eos.to(T=eos.T, P=eos.P).c, c_force, rtol=1e-12)

    # Test overwtitting alphas
    alpha_force = (0.623166885722628, 0.831835883048979, 1.13487540699625)
    eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, c=c_force, alpha_coeffs=alpha_force)
    assert_close(eos.c, c_force, rtol=1e-12)
    eos_copy = eos.to(T=eos.T, P=eos.P)
    assert_close(eos_copy.c, c_force, rtol=1e-12)

    a_alphas_new = (3.8055614088179506, -0.0069721058918524054, 2.1747678188454976e-05)
    a_alphas_calc = (eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2)
    assert_close1d(a_alphas_new, a_alphas_calc, rtol=1e-9)
    a_alphas_calc_copy = (eos_copy.a_alpha, eos_copy.da_alpha_dT, eos_copy.d2a_alpha_dT2)
    assert_close1d(a_alphas_new, a_alphas_calc_copy, rtol=1e-9)

    assert eos.P_max_at_V(1) is None # No direct solution for P


def test_SRKTranslatedConsistent():
    eos = SRKTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.0001263530801579297, -31683.55352277101, -74.18061768230253]
    assert_close1d(three_props, expect_props)

    # Test of a_alphas
    a_alphas = [3.720329980107943, -0.007300826722636607, 2.1378814745831847e-05]

    a_alphas_fast = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_fast)

    # back calculation for T
    eos = SRKTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, V=0.0001263530801579297, P=1E6)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001263530801579297)
    assert_close(T_slow, 299)

    # TV solve for P
    eos = eos.to(T=eos.T, V=eos.V_l)
    assert_close(eos.P, 1e6, rtol=1e-12)


    # Two correlations
    # Test the bool to control its behavior results in the same conditions
    eos = SRKTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=406.08, P=1E6)
    assert_close(eos.c, 2.053287245221519e-05, rtol=1e-12)
    assert_close1d([0.363593791875, 0.8320110093749999, 2.0], eos.alpha_coeffs, rtol=1e-12)

    # Test overwritting c
    c_force = 22.3098E-6
    eos = SRKTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, c=c_force)
    assert_close(eos.c, c_force, rtol=1e-12)
    assert_close(eos.to(T=eos.T, P=eos.P).c, c_force, rtol=1e-12)

    # Test overwtitting alphas
    alpha_force = (0.623166885722628, 0.831835883048979, 1.13487540699625)
    eos = SRKTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, c=c_force, alpha_coeffs=alpha_force)
    assert_close(eos.c, c_force, rtol=1e-12)
    eos_copy = eos.to(T=eos.T, P=eos.P)
    assert_close(eos_copy.c, c_force, rtol=1e-12)

    a_alphas_new = (3.557908729514593, -0.0065183855286746915, 2.0332415052898068e-05)
    a_alphas_calc = (eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2)
    assert_close1d(a_alphas_new, a_alphas_calc, rtol=1e-9)
    a_alphas_calc_copy = (eos_copy.a_alpha, eos_copy.da_alpha_dT, eos_copy.d2a_alpha_dT2)
    assert_close1d(a_alphas_new, a_alphas_calc_copy, rtol=1e-9)


def test_PRTranslatedPPJP():
    eos = PRTranslatedPPJP(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.00013013535006092269, -31304.613873527414, -72.86609506697148]
    assert_close1d(three_props, expect_props)

    # Test of a_alphas
    a_alphas = [3.811942643891255, -0.006716261984061717, 1.714789853730881e-05]

    a_alphas_implemented = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_implemented)

    # back calculation for T
    eos = PRTranslatedPPJP(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00013013535006092269, P=1E6)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00013013535006092269)
    assert_close(T_slow, 299)

    # TV solve for P
    eos_TV = eos.to(T=eos.T, V=eos.V_l)
    assert_close(eos_TV.P, 1e6, rtol=1e-12)
    assert_close(eos_TV.kappa, eos.kappa)
    assert_close(eos.kappa, 0.8167473931515625, rtol=1e-12)

    # Test with c
    c = 0.6390E-6
    eos = PRTranslatedPPJP(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, c=c)
    assert_close(eos.c, c, rtol=1e-12)

    eos_copy = eos.to(T=eos.T, P=eos.P)
    assert_close(eos_copy.c, c, rtol=1e-12)

    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.00012949635006092268, -31305.252873527414, -72.86609506697148]
    assert_close1d(three_props, expect_props)

    a_alphas_new = (3.811942643891255, -0.006716261984061717, 1.714789853730881e-05)
    a_alphas_calc = (eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2)
    assert_close1d(a_alphas_new, a_alphas_calc, rtol=1e-9)

    eos_copy = eos.to(T=eos.T, P=eos.P)
    a_alphas_calc_copy = (eos_copy.a_alpha, eos_copy.da_alpha_dT, eos_copy.d2a_alpha_dT2)
    assert_close1d(a_alphas_new, a_alphas_calc_copy, rtol=1e-9)

    # A P_max_at_V example anyway; still not implemented for the volume translation case
    base = PRTranslatedPPJP(Tc=512.5, Pc=8084000.0, omega=0.559, c=0.0, T=736.5357142857143, P=91255255.16379045)
    P_max = base.P_max_at_V(base.V_l)
    base.to(P=P_max-1, V=base.V_l).phase
    with pytest.raises(Exception):
        base.to(P=P_max+100, V=base.V_l).phase


def test_SRKTranslatedPPJP():
    eos = SRKTranslatedPPJP(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.0001468182905372137, -31759.404328020573, -74.38422222691308]
    assert_close1d(three_props, expect_props)

    # Test of a_alphas
    a_alphas = [3.7274765423787573, -0.007334913389260811, 1.9482548027213383e-05]

    a_alphas_implemented = eos.a_alpha_and_derivatives_pure(299)
    assert_close1d(a_alphas, a_alphas_implemented)


    # back calculation for T
    eos = SRKTranslatedPPJP(Tc=507.6, Pc=3025000, omega=0.2975, V=0.0001468182905372137, P=1E6)
    assert_close(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001468182905372137)
    assert_close(T_slow, 299)

    # TV solve for P
    eos_TV = eos.to(T=eos.T, V=eos.V_l)
    assert_close(eos_TV.P, 1e6, rtol=1e-12)
    assert_close(eos_TV.m, eos.m)
    assert_close(eos.m, 0.9328950816515624, rtol=1e-12)

    # Test with c
    c = 22.3098E-6
    eos = SRKTranslatedPPJP(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, c=c)
    assert_close(eos.c, c, rtol=1e-12)
    eos_copy = eos.to(T=eos.T, P=eos.P)
    assert_close(eos_copy.c, c, rtol=1e-12)

    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.0001245084905372137, -31781.714128020576, -74.3842222269131]
    assert_close1d(three_props, expect_props)

    a_alphas_new = (3.7274765423787573, -0.007334913389260811, 1.9482548027213383e-05)
    a_alphas_calc = (eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2)
    assert_close1d(a_alphas_new, a_alphas_calc, rtol=1e-9)
    a_alphas_calc_copy = (eos_copy.a_alpha, eos_copy.da_alpha_dT, eos_copy.d2a_alpha_dT2)
    assert_close1d(a_alphas_new, a_alphas_calc_copy, rtol=1e-9)

def test_IG_hash():
    base = IG(T=300.0, P=1e5)
    assert base.state_hash() != IG(T=400.0, P=1e5).state_hash()
    assert base.state_hash() != IG(T=base.T, V=base.V_g).state_hash()

def test_IG():
    Ts = [1e-20, 1e-5, 1, 100, 1000, 1e4, 1e7, 1e30]
    Ps = [1e-20, 1e-5, 1, 1000, 1e6, 1e10, 1e30]

    zero_attrs = ['d2T_dV2_g', 'd2V_dT2_g', 'U_dep_g', 'A_dep_g', 'V_dep_g', 'Cp_dep_g', 'Cv_dep_g',
             'G_dep_g', 'H_dep_g', 'S_dep_g', 'd2P_dT2_g', 'd2T_dP2_g',
             'a_alpha', 'da_alpha_dT', 'd2a_alpha_dT2', 'dH_dep_dT_g',
             'dH_dep_dP_g', 'dS_dep_dT_g', 'dS_dep_dP_g',
              'dfugacity_dT_g', 'dphi_dT_g', 'dphi_dP_g']
    tol = 1e-15


    eos = IG(T=400., P=1E6)
    assert eos.phi_sat(300) == 1.0
    assert eos.dphi_sat_dT(300) == 0.0
    assert eos.d2phi_sat_dT2(300) == 0.0

    for T in Ts:
        for P in Ps:

            eos = IG(T=T, P=P)

            for attr in zero_attrs:
                assert getattr(eos, attr) == 0
            V = eos.V_g

            assert_close(eos.Cp_minus_Cv_g, R, rtol=tol)
            assert_close(eos.PIP_g, 1, rtol=tol)
            assert_close(eos.Z_g, 1, rtol=tol)
            assert_close(eos.phi_g, 1, rtol=tol)
            assert_close(eos.dfugacity_dP_g, 1, rtol=tol)
            assert_close(eos.fugacity_g, P, rtol=tol)
            assert_close(eos.V_g,  R*T/P, rtol=tol)

            # P derivatives - print(diff(R*T/V, V, V))
            assert_close(eos.dP_dT_g, R/V, rtol=tol)
            assert_close(eos.dP_dV_g, -R*T/V**2, rtol=tol)
            assert_close(eos.d2P_dTdV_g, -R/V**2, rtol=tol)
            assert_close(eos.d2P_dV2_g, 2*R*T/V**3 , rtol=tol)

            # T derivatives - print(diff(P*V/R, P, V))
            assert_close(eos.dT_dP_g, V/R, rtol=tol)
            assert_close(eos.dT_dV_g, P/R, rtol=tol)
            assert_close(eos.d2T_dPdV_g, 1/R, rtol=tol)

            # V derivatives - print(diff(R*T/P, P))
            assert_close(eos.dV_dP_g, -R*T/P**2, rtol=tol)
            assert_close(eos.dV_dT_g, R/P, rtol=tol)
            assert_close(eos.d2V_dP2_g, 2*R*T/P**3, rtol=tol)
            assert_close(eos.d2V_dPdT_g, -R/P**2, rtol=tol)

            # Misc
            assert_close(eos.beta_g, R/P/V, rtol=tol)
            assert_close(eos.kappa_g, R*T/P**2/V, rtol=tol)

@pytest.mark.slow
def test_fuzz_dV_dT_and_d2V_dT2_derivatives():
    dx = 2e-7
    order = 5
    Tc=507.6
    Pc=3025000.
    omega=0.2975
    rtol = 5e-6
    for eos in eos_2P_list:
        for T in linspace(10, 1000, 5):
            for P in logspace(log10(3E4), log10(1E6), 5):
                e = eos(Tc=Tc, Pc=Pc, omega=omega, T=T, P=P)
                c = {}
                def cto(T, P):
                    k = (T, P)
                    if k in c:
                        return c[k]
                    new = e.to(T=T, P=P)
                    c[k] = new
                    return new

                if 'l' in e.phase:
                    dV_dT_l_num = derivative(lambda T: cto(T=T, P=P).V_l, T, order=order, dx=T*dx)
                    assert_close(e.dV_dT_l, dV_dT_l_num, rtol=rtol)
                    d2V_dT2_l_num = derivative(lambda T: cto(T=T, P=P).dV_dT_l, T, order=order, dx=T*dx)
                    assert_close(e.d2V_dT2_l, d2V_dT2_l_num, rtol=rtol)
                if 'g' in e.phase:
                    dV_dT_g_num = derivative(lambda T: cto(T=T, P=P).V_g, T, order=order, dx=T*dx)
                    assert_close(e.dV_dT_g, dV_dT_g_num, rtol=rtol)
                    d2V_dT2_g_num = derivative(lambda T: cto(T=T, P=P).dV_dT_g, T, order=order, dx=T*dx)
                    assert_close(e.d2V_dT2_g, d2V_dT2_g_num, rtol=rtol)


#    from thermo import eos
#    phase_extensions = {True: '_l', False: '_g'}
#    derivative_bases_dV_dT = {0:'V', 1:'dV_dT', 2:'d2V_dT2'}
#
#    def dV_dT(T, P, eos, order=0, phase=True, Tc=507.6, Pc=3025000., omega=0.2975):
#        e = eos(Tc=Tc, Pc=Pc, omega=omega, T=T, P=P)
#        phase_base = phase_extensions[phase]
#        attr = derivative_bases_dV_dT[order]+phase_base
#        return getattr(e, attr)
#
#    x, y = [], []
#    for eos in eos_2P_list:
#        for T in linspace(.1, 1000, 50):
#            for P in logspace(log10(3E4), log10(1E6), 50):
#                T, P = float(T), float(P)
#                for phase in [True, False]:
#                    for order in [1, 2]:
#                        try:
#                            # If dV_dx_phase doesn't exist, will simply abort and continue the loop
#                            numer = derivative(dV_dT, T, dx=1E-4, args=(P, eos, order-1, phase))
#                            ana = dV_dT(T=T, P=P, eos=eos, order=order, phase=phase)
#                        except:
#                            continue
#                        x.append(numer)
#                        y.append(ana)
#    assert allclose_variable(x, y, limits=[.009, .05, .65, .93],rtols=[1E-5, 1E-6, 1E-9, 1E-10])
#

@pytest.mark.slow
@pytest.mark.fuzz
def test_fuzz_dV_dP_and_d2V_dP2_derivatives():
    eos_list2 = list(eos_list)
    eos_list2.remove(VDW)
    eos_list2.remove(IG)

    phase_extensions = {True: '_l', False: '_g'}
    derivative_bases_dV_dP = {0:'V', 1:'dV_dP', 2:'d2V_dP2'}

    def dV_dP(P, T, eos, order=0, phase=True, Tc=507.6, Pc=3025000., omega=0.2975):
        eos = eos(Tc=Tc, Pc=Pc, omega=omega, T=T, P=P)
        phase_base = phase_extensions[phase]
        attr = derivative_bases_dV_dP[order]+phase_base
        return getattr(eos, attr)


    for eos in eos_list2:
        for T in linspace(20, 1000, 5):
            x, y = [], []
            for P in logspace(log10(3E4), log10(1E6), 5):
                T, P = float(T), float(P)
                for phase in [True, False]:
                    for order in [1, 2]:
                        try:
                            # If dV_dx_phase doesn't exist, will simply abort and continue the loop
                            numer = derivative(dV_dP, P, dx=P*1e-4, order=11, args=(T, eos, order-1, phase))
                            ana = dV_dP(T=T, P=P, eos=eos, order=order, phase=phase)
                        except:
                            continue
                        x.append(numer)
                        y.append(ana)
            assert_close1d(x, y, rtol=3e-5)


@pytest.mark.slow
def test_fuzz_Psat():
    Tc = 507.6
    Pc = 3025000
    omega = 0.2975
    # Basic test
    e = PR(T=400, P=1E5, Tc=507.6, Pc=3025000, omega=0.2975)
    Psats_expect = [22284.73414685111, 466205.07373896183, 2717375.4955021995]
    assert_close1d([e.Psat(300), e.Psat(400), e.Psat(500)], Psats_expect)


    # Test the relative fugacity errors at the correlated Psat are small
    x = []
    for eos in eos_2P_list:
        for T in linspace(0.318*Tc, Tc*.99, 100):
            e = eos(Tc=Tc, Pc=Pc, omega=omega, T=T, P=1E5)
            Psat = e.Psat(T)
            e = e.to_TP(T, Psat)
            rerr = (e.fugacity_l - e.fugacity_g)/e.fugacity_g
            x.append(rerr)

    # Assert the average error is under 0.04%
    assert sum(abs(np.array(x)))/len(x) < 1E-4

    # Test Polish is working, and that its values are close to the polynomials
    Psats_solved = []
    Psats_poly = []
    for eos in eos_2P_list:
        for T in linspace(0.4*Tc, Tc*.99, 50):
            e = eos(Tc=Tc, Pc=Pc, omega=omega, T=T, P=1E5)
            Psats_poly.append(e.Psat(T))
            Psats_solved.append(e.Psat(T, polish=True))
    assert_close1d(Psats_solved, Psats_poly, rtol=1E-11)

def test_Psat_issues():
    e = PR(T=229.43458646616548, P=100000.0, Tc=708.0, Pc=1480000.0, omega=0.6897)
    assert_close(e.Psat(e.T), 0.00012715494309024902)

    eos = TWUPR(Tc=611.7, Pc=2110000.0, omega=0.49, T=298.15, P=101325.0)
    Tsat = eos.Tsat(1e-100)
    assert_close(Tsat, 44.196052244378244, rtol=1e-7)
    assert_close(eos.Psat(Tsat), 1e-100)

    # Ammonia
    eos = TWUPR(Tc=405.6, Pc=11277472.5, omega=0.25, T=298.15, P=101325.0)
    Tsat = eos.Tsat(1e-100)
    assert_close(Tsat, 22.50315376221732, rtol=1e-7)
    assert_close(eos.Psat(Tsat), 1e-100)

    eos = TWUSRK(Tc=405.6, Pc=11277472.5, omega=0.25, T=298.15, P=101325.0)
    Tsat = eos.Tsat(1e-100)
    assert_close(Tsat, 23.41595921544242, rtol=1e-7)
    assert_close(eos.Psat(Tsat), 1e-100)

    # Issue where the brenth solver does not converge to an appropriate ytol
    # and the newton fails; added code to start newton from the bounded solver
    # where it looks like a good solution
    eos = PR(Tc=540.2, Pc=2740000.0, omega=0.3457, T=298.15, P=101325.0)
    Tsat = eos.Tsat(2453124.6502311486, polish=False)
    assert_close(Tsat, 532.1131652558847, rtol=1e-7)

    # Case where y was evaluated just above Pc and so couldn't converge
    # The exact precision of the answer can only be obtained with mpmath
    eos = PR(Tc=647.086, Pc=22048320.0, omega=0.344, T=230.0, P=100000.0)
    assert_close(eos.Psat(eos.Tc*(1-1e-13), polish=True), 22048319.99998073, rtol=1e-10)
    
    e = PRTranslatedConsistent(Tc=512.5, Pc=8084000.0, omega=0.559, c=2.4079466437131265e-06, alpha_coeffs=(0.46559014900000006, 0.798056656, 2.0), T=298.15, P=101325.0)
    assert_close(e.Psat(26.5928527253961065, polish=True), 3.4793909343216283e-152)

    e = TWUSRK(Tc=512.5, Pc=8084000.0, omega=0.559, T=298.15, P=101325.0)
    assert_close(e.Psat(30.9101562500000000, True), 1.925983717076344e-158)

    # TWU hydrogen at low conditions had many issues
    # e = TWUPR(Tc=33.2, Pc=1296960.0, omega=-0.22, T=298.15, P=101325.0)
    # e.Psat(T=1.24005018079967879, polish=True)

    # e = TWUSRK(Tc=33.2, Pc=1296960.0, omega=-0.22, T=298.15, P=101325.0)
    # e.Psat(T=1.24005018079967879, polish=True)


def test_Tsat_issues():
    # This point should be easy to solve and should not require full evaluations
    # Cannot test that but this can be manually checked
    base = PRTranslatedConsistent(Tc=647.14, Pc=22048320.0, omega=0.344, c=5.2711e-06, alpha_coeffs=[0.3872, 0.87587208, 1.9668], T=298.15, P=101325.0)
    assert_close(base.Tsat(1e5), 371.95148202471137, rtol=1e-6)

    # Case was not solving with newton to the desired tolerance - increased xtol
    e = PR(Tc=304.2, Pc=7376460.0, omega=0.2252, T=298.15, P=101325.0)
    assert_close(e.Tsat(135368.0749152002), 189.8172487166094)


def test_Tsat_issues_extrapolation():
    P = 6100000.000000002
    e = PR(Tc=305.32, Pc=4872000.0, omega=0.098, T=298.15, P=101325.0)
    e.Tsat(P)

    e = PR(Tc=469.7, Pc=3370000.0, omega=0.251, T=298.15, P=101325.0)
    e.Tsat(P)



def test_fuzz_dPsat_dT():
    Tc = 507.6
    Pc = 3025000
    omega = 0.2975

    e = PR(T=400, P=1E5, Tc=507.6, Pc=3025000, omega=0.2975)
    dPsats_dT_expect = [938.8330442120659, 10288.110417535852, 38843.65395496486, 42138.50972119547]
    assert_close1d([e.dPsat_dT(300), e.dPsat_dT(400), e.dPsat_dT(500), e.dPsat_dT(507.59)], dPsats_dT_expect)

@pytest.mark.slow
def test_fuzz_dPsat_dT_full():
    # TODO - add specific points to separate test

    Tc = 507.6
    Pc = 3025000
    omega = 0.2975

    e = PR(T=400, P=1E5, Tc=507.6, Pc=3025000, omega=0.2975)
    # Hammer the derivatives for each EOS in a wide range; most are really
    # accurate. There's an error around the transition between polynomials
    # though - to be expected; the derivatives are discontinuous there.
    dPsats_derivative = []
    dPsats_analytical = []
    for eos in eos_2P_list:
        e = eos(Tc=Tc, Pc=Pc, omega=omega, T=298.15, P=1E5)
        for T in [.1*Tc, .2*Tc, .5*Tc, .7*Tc, .9*Tc, .99*Tc, .999*Tc]: # , 0.99999*Tc will fail
            anal = e.dPsat_dT(T)
            numer = e.dPsat_dT(T, polish=True)
            dPsats_analytical.append(anal)
            dPsats_derivative.append(numer)

    assert_close1d(dPsats_derivative, dPsats_analytical)



def test_Hvaps():
    eos_iter = list(eos_list)
    eos_iter.remove(IG)

    Tc = 507.6
    Pc = 3025000
    omega = 0.2975

    Hvaps = {}
    Hvaps_expect = {PR78: 31086.219936356967,
                 TWUPR: 31592.70583768895,
                 TWUSRK: 31562.954930111275,
                 SRK: 31711.070598361417,
                 PR: 31086.219936356967,
                 PRSV2: 31035.43366905585,
                 VDW: 13004.122697411925,
                 APISRK: 31715.843150141456,
                 SRKTranslatedConsistent: 31615.664668686026,
                 RK: 26010.12522947027,
                 SRKTranslatedPPJP: 31715.83940885055,
                 PRTranslatedConsistent: 31419.504797000944,
                 PRSV: 31035.43366905585,
                 PRTranslatedPPJP: 31257.15735072686,
                 MSRKTranslated: 31548.838206563854}

    for eos in eos_iter:
        e = eos(Tc=Tc, Pc=Pc, omega=omega, T=300, P=1E5)
        Hvap_calc = e.Hvap(300)
        Hvaps[eos] = Hvap_calc

    for eos in eos_iter:
        assert_close(Hvaps_expect[eos], Hvaps[eos], rtol=1e-7)



def test_V_l_sats():
    eos_iter = list(eos_list)
    eos_iter.remove(IG)

    Tc = 507.6
    Pc = 3025000
    omega = 0.2975

    V_l_sats = {}
    V_l_sats_expect = {PR78: 0.00013065657945228289,
                        TWUPR: 0.0001306108304312293,
                        TWUSRK: 0.00014745855640971024,
                        SRK: 0.00014738493904330595,
                        PR: 0.00013065657945228289,
                        PRSV2: 0.00013068338288565773,
                        VDW: 0.0002249691466839674,
                        APISRK: 0.00014738204694914915,
                        SRKTranslatedConsistent: 0.00012691945527035751,
                        RK: 0.00015267480918175063,
                        SRKTranslatedPPJP: 0.00014738204921603808,
                        PRTranslatedConsistent: 0.00013147170331755256,
                        PRSV: 0.00013068338288565773,
                        PRTranslatedPPJP: 0.00013056689289733488,
                        MSRKTranslated:0.00014744370993727}

    for eos in eos_iter:
        e = eos(Tc=Tc, Pc=Pc, omega=omega, T=300, P=1E5)
        V_l_sat_calc = e.V_l_sat(300)
        V_l_sats[eos] = V_l_sat_calc

    for eos in eos_iter:
        assert_close(V_l_sats_expect[eos], V_l_sats[eos], rtol=1e-7)


def test_V_g_sats():
    eos_iter = list(eos_list)
    eos_iter.remove(IG)

    Tc = 507.6
    Pc = 3025000
    omega = 0.2975

    V_g_sats = {}
    V_g_sats_expect = {PR78: 0.11050249932616305,
                        TWUPR: 0.11172398125036595,
                        TWUSRK: 0.11196919594301465,
                        SRK: 0.11367528462559041,
                        PR: 0.11050249932616305,
                        PRSV2: 0.10979545797759405,
                        VDW: 0.009465797766589491,
                        APISRK: 0.11374303933593632,
                        SRKTranslatedConsistent: 0.11209163965600458,
                        RK: 0.04604605088410411,
                        SRKTranslatedPPJP: 0.11374298620644632,
                        PRTranslatedConsistent: 0.11144255765526169,
                        PRSV: 0.10979545797759405,
                        PRTranslatedPPJP: 0.1129148079163081,
                        MSRKTranslated: 0.11231040560602985}



    for eos in eos_iter:
        e = eos(Tc=Tc, Pc=Pc, omega=omega, T=300, P=1E5)
        V_g_sat_calc = e.V_g_sat(300)
        V_g_sats[eos] = V_g_sat_calc

    for eos in eos_iter:
        assert_close(V_g_sats_expect[eos], V_g_sats[eos], rtol=1e-7)


def test_dfugacity_dT_l_dfugacity_dT_g():
    T = 400
    delta = 1e-5
    eos1 = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T, P=1E6)
    eos2 = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T +delta, P=1E6)
    numerical = (eos2.fugacity_l - eos1.fugacity_l)/delta
    analytical = eos1.dfugacity_dT_l
    assert_close(numerical, analytical)

    numerical = (eos2.fugacity_g - eos1.fugacity_g)/delta
    analytical = eos1.dfugacity_dT_g
    assert_close(numerical, analytical)

def test_dfugacity_dP_l_dfugacity_dP_g():
    T = 400
    delta = 1
    eos1 = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T, P=1E6)
    eos2 = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T, P=1E6+delta)
    numerical = (eos2.fugacity_l - eos1.fugacity_l)/delta
    analytical = eos1.dfugacity_dP_l
    assert_close(numerical, analytical)

    numerical = (eos2.fugacity_g - eos1.fugacity_g)/delta
    analytical = eos1.dfugacity_dP_g
    assert_close(numerical, analytical, rtol=1e-6)


def test_dphi_dT_l_g():
    '''from sympy import * # doctest:+SKIP
    P, T, R = symbols('P, T, R') # doctest:+SKIP
    H_dep, S_dep = symbols('H_dep, S_dep', cls=Function)

    G_dep = H_dep(T, P) - T*S_dep(T, P)
    phi = exp(G_dep/(R*T)) # doctest:+SKIP
    print(latex(diff(phi, T)))
    '''
    T = 400
    delta = 1e-7
    eos1 = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T, P=1E6)
    eos2 = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T + delta, P=1E6)
    numerical = (eos2.phi_l - eos1.phi_l)/delta
    analytical = eos1.dphi_dT_l
    assert_close(numerical, analytical, rtol=1e-5)

    numerical = (eos2.phi_g - eos1.phi_g)/delta
    analytical = eos1.dphi_dT_g
    assert_close(numerical, analytical, rtol=1e-5)


def test_dphi_dP_l_g():
    '''
    from sympy import * # doctest:+SKIP
    P, T, R = symbols('P, T, R') # doctest:+SKIP
    H_dep, S_dep = symbols('H_dep, S_dep', cls=Function)

    G_dep = H_dep(T, P) - T*S_dep(T, P)
    phi = exp(G_dep/(R*T)) # doctest:+SKIP
    print(latex(diff(phi, P)))
    '''
    T = 400
    delta = 1e-2
    eos1 = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T, P=1E6)
    eos2 = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T, P=1E6+delta)
    numerical = (eos2.phi_l - eos1.phi_l)/delta
    analytical = eos1.dphi_dP_l
    assert_close(numerical, analytical, rtol=1e-6)

    numerical = (eos2.phi_g - eos1.phi_g)/delta
    analytical = eos1.dphi_dP_g
    assert_close(numerical, analytical, rtol=1e-6)

def test_dbeta_dT():
    T = 400
    delta = 1e-7
    eos1 = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T, P=1E6)
    eos2 = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T + delta, P=1E6)
    numerical = (eos2.beta_l - eos1.beta_l)/delta
    analytical = eos1.dbeta_dT_l
    assert_close(numerical, analytical, rtol=1e-5)
    assert_close(analytical, 2.9048493082069868e-05, rtol=1e-9)

    numerical = (eos2.beta_g - eos1.beta_g)/delta
    analytical = eos1.dbeta_dT_g
    assert_close(numerical, analytical, rtol=1e-5)
    assert_close(analytical, -0.00033081702756075523, rtol=1e-9)




def test_dbeta_dP():
    T = 400
    delta = 10
    eos1 = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T, P=1E6)
    numerical = derivative(lambda P: eos1.to_TP(eos1.T, P).beta_l, eos1.P, order=15, dx=30)
    analytical = eos1.dbeta_dP_l
    assert_close(numerical, analytical, rtol=1e-8)

    numerical = derivative(lambda P: eos1.to_TP(eos1.T, P).beta_g, eos1.P, order=15, dx=30)
    analytical = eos1.dbeta_dP_g
    assert_close(numerical, analytical, rtol=1e-8)

def test_d2H_dep_dT2_P():
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E5)
    d2H_dep_dT2_g_num = derivative(lambda T: eos.to(P=eos.P, T=T).dH_dep_dT_g, eos.T, dx=eos.T*1e-8)
    assert_close(d2H_dep_dT2_g_num, eos.d2H_dep_dT2_g, rtol=1e-8)
    assert_close(eos.d2H_dep_dT2_g, -0.01886053682747742, rtol=1e-12)

    d2H_dep_dT2_l_num = derivative(lambda T: eos.to(P=eos.P, T=T).dH_dep_dT_l, eos.T, dx=eos.T*1e-8)
    assert_close(d2H_dep_dT2_l_num, eos.d2H_dep_dT2_l, rtol=1e-7)
    assert_close(eos.d2H_dep_dT2_l, 0.05566404509607853, rtol=1e-12)

def test_d2H_dep_dT2_V():
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E5)
    d2H_dep_dT2_g_V_num = derivative(lambda T: eos.to(V=eos.V_g, T=T).dH_dep_dT_g_V, eos.T, dx=eos.T*5e-7, order=5)
    assert_close(eos.d2H_dep_dT2_g_V, d2H_dep_dT2_g_V_num, rtol=1e-9)
    assert_close(eos.d2H_dep_dT2_g_V, -0.0010786001950969532)

    d2H_dep_dT2_l_V_num = derivative(lambda T: eos.to(V=eos.V_l, T=T).dH_dep_dT_l_V, eos.T, dx=eos.T*5e-7, order=5)
    assert_close(eos.d2H_dep_dT2_l_V, d2H_dep_dT2_l_V_num, rtol=1e-9)
    assert_close(eos.d2H_dep_dT2_l_V, -0.1078300228825107)

def test_d2S_dep_dT2_P():
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E5)
    d2S_dep_dT2_g_num = derivative(lambda T: eos.to(P=eos.P, T=T).dS_dep_dT_g, eos.T, dx=eos.T*1e-8)
    assert_close(eos.d2S_dep_dT2_g, -8.644953519864201e-05, rtol=1e-10)
    assert_close(eos.d2S_dep_dT2_g, d2S_dep_dT2_g_num)

    d2S_dep_dT2_l_num = derivative(lambda T: eos.to(P=eos.P, T=T).dS_dep_dT_l, eos.T, dx=eos.T*1e-8)
    assert_close(eos.d2S_dep_dT2_l, -0.00031525424335593154)
    assert_close(d2S_dep_dT2_l_num, eos.d2S_dep_dT2_l)

def test_d2S_dep_dT2_V():
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E5)
    d2S_dep_dT2_g_V_num = derivative(lambda T: eos.to(V=eos.V_g, T=T).dS_dep_dT_g_V, eos.T, dx=eos.T*5e-7, order=5)
    assert_close(eos.d2S_dep_dT2_g_V, d2S_dep_dT2_g_V_num, rtol=1e-8)
    assert_close(eos.d2S_dep_dT2_g_V, -2.6744188913248543e-05, rtol=1e-11)

    d2S_dep_dT2_l_V_num = derivative(lambda T: eos.to(V=eos.V_l, T=T).dS_dep_dT_l_V, eos.T, dx=eos.T*5e-7, order=5)
    assert_close(eos.d2S_dep_dT2_l_V, d2S_dep_dT2_l_V_num, rtol=1e-9)
    assert_close(eos.d2S_dep_dT2_l_V, -277.0161576452194, rtol=1e-11)

def test_d2H_dep_dTdP():
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E5)
    d2H_dep_dTdP_g_num = derivative(lambda P: eos.to(P=P, T=eos.T).dH_dep_dT_g, eos.P, dx=eos.P*3e-8)
    assert_close(eos.d2H_dep_dTdP_g, 2.4880842067857194e-05, rtol=1e-11)
    assert_close(eos.d2H_dep_dTdP_g, d2H_dep_dTdP_g_num, rtol=1e-7)

    d2H_dep_dTdP_l_num = derivative(lambda P: eos.to(P=P, T=eos.T).dH_dep_dT_l, eos.P, dx=eos.P*3e-7)
    assert_close(eos.d2H_dep_dTdP_l, -3.662334969933377e-07, rtol=1e-11)
    assert_close(eos.d2H_dep_dTdP_l, d2H_dep_dTdP_l_num, rtol=4e-6)

def test_d2S_dep_dTdP():
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E5)
    d2S_dep_dTdP_g_num = derivative(lambda P: eos.to(P=P, T=eos.T).dS_dep_dT_g, eos.P, dx=eos.P*4e-8)
    assert_close(eos.d2S_dep_dTdP_g, 8.321351862159589e-08, rtol=1e-11)
    assert_close(eos.d2S_dep_dTdP_g, d2S_dep_dTdP_g_num, rtol=1e-7)

    d2S_dep_dTdP_l_num = derivative(lambda P: eos.to(P=P, T=eos.T).dS_dep_dT_l, eos.P, dx=eos.P*2e-6, order=5)
    assert_close(eos.d2S_dep_dTdP_l, -1.2248611939576295e-09, rtol=1e-11)
    assert_close(eos.d2S_dep_dTdP_l, d2S_dep_dTdP_l_num, rtol=5e-7)

def test_d2P_dVdP():
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E5)
    num = derivative(lambda P: eos.to(P=P, T=eos.T).dP_dV_g, eos.P, dx=eos.P*1e-8)
    assert_close(num, eos.d2P_dVdP_g, rtol=1e-7)
    assert_close(eos.d2P_dVdP_g, -79.86820966180667, rtol=1e-11)

    num = derivative(lambda P: eos.to(P=P, T=eos.T).dP_dV_l, eos.P, dx=eos.P*2e-7, order=5)
    assert_close(num, eos.d2P_dVdP_l, rtol=1e-5)
    assert_close(eos.d2P_dVdP_l, -121536.72600389269, rtol=1e-11)

def test_d2P_dTdP():
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E5)
    d2P_dTdP_g_num = derivative(lambda P: eos.to(P=P, V=eos.V_g).dP_dT_g, eos.P, dx=eos.P*4e-6)
    assert_close(eos.d2P_dTdP_g, d2P_dTdP_g_num)
    assert_close(eos.d2P_dTdP_g, -8.314373652066897e-05, rtol=1e-11)

def test_d2P_dVdT_TP():
    '''SymPy source:
    from sympy import *
    P, T, R, b, delta, epsilon = symbols('P, T, R, b, delta, epsilon')
    a_alpha, V = symbols(r'a\alpha, V', cls=Function)
    Vconst = symbols('Vconst')
    CUBIC = R*T/(Vconst-b) - a_alpha(T)/(Vconst*Vconst + delta*Vconst + epsilon)
    dP_dV = diff(CUBIC, Vconst)
    diff(dP_dV.subs(Vconst, V(T)), T)
    '''
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E5)
    d2P_dVdT_TP_g_num = derivative(lambda T: eos.to(P=eos.P, T=T).dP_dV_g, eos.T, dx=eos.T*2e-8, order=3)
    assert_close(eos.d2P_dVdT_TP_g, 13116.743483603945, rtol=1e-11)
    assert_close(eos.d2P_dVdT_TP_g, d2P_dVdT_TP_g_num, rtol=1e-7)

    d2P_dVdT_TP_l_num = derivative(lambda T: eos.to(P=eos.P, T=T).dP_dV_l, eos.T, dx=eos.T*2e-8, order=3)
    assert_close(eos.d2P_dVdT_TP_l, 50040777323.75235, rtol=1e-11)
    assert_close(eos.d2P_dVdT_TP_l, d2P_dVdT_TP_l_num, rtol=1e-7)

def test_d2P_dT2_PV():
    '''
    from sympy import *
    P, T, R, b, delta, epsilon, Vconst = symbols('P, T, R, b, delta, epsilon, Vconst')
    a_alpha, V = symbols(r'a\alpha, V', cls=Function)
    CUBIC = R*T/(Vconst-b) - a_alpha(T)/(Vconst*Vconst + delta*Vconst + epsilon)
    dP_dT_V = diff(CUBIC, T).subs(Vconst, V(T))
    diff(dP_dT_V, T)
    '''
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E5)
    d2P_dT2_PV_g_num = derivative(lambda T: eos.to(P=eos.P, T=T).dP_dT_g, eos.T, dx=eos.T*2e-8, order=3)
    assert_close(eos.d2P_dT2_PV_g, -1.5428790190793502, rtol=1e-11)
    assert_close(eos.d2P_dT2_PV_g, d2P_dT2_PV_g_num, rtol=1e-7)

    d2P_dT2_PV_l_num = derivative(lambda T: eos.to(P=eos.P, T=T).dP_dT_l, eos.T, dx=eos.T*2e-8, order=3)
    assert_close(eos.d2P_dT2_PV_l, -3768.328311659964, rtol=1e-11)
    assert_close(eos.d2P_dT2_PV_l, d2P_dT2_PV_l_num, rtol=1e-7)

def test_d2P_dTdP():
    '''
    from sympy import *
    P, T, R, b, delta, epsilon, Vconst = symbols('P, T, R, b, delta, epsilon, Vconst')
    a_alpha, V = symbols(r'a\alpha, V', cls=Function)
    CUBIC = R*T/(Vconst-b) - a_alpha(T)/(Vconst*Vconst + delta*Vconst + epsilon)
    dP_dT_V = diff(CUBIC, T).subs(Vconst, V(P))
    diff(dP_dT_V, P)
    '''
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E5)
    d2P_dTdP_g_num = derivative(lambda P: eos.to(P=P, T=eos.T).dP_dT_g, eos.P, dx=eos.P*1e-8)
    assert_close(d2P_dTdP_g_num, eos.d2P_dTdP_g, rtol=1e-8)
    assert_close(eos.d2P_dTdP_g, 0.0040914288794245265, rtol=1e-12)

    d2P_dTdP_l_num = derivative(lambda P: eos.to(P=P, T=eos.T).dP_dT_l, eos.P, dx=eos.P*8e-6, order=7)
    assert_close(d2P_dTdP_l_num, eos.d2P_dTdP_l, rtol=1e-7)
    assert_close(eos.d2P_dTdP_l, 0.0056550655224853735, rtol=1e-12)


def test_dH_dep_dT_V():
    '''Equation obtained with:

    from sympy import *
    V, T, R, b, delta, epsilon = symbols('V, T, R, b, delta, epsilon')
    P, a_alpha, aalpha2 = symbols('P, a_alpha, a_alpha2', cls=Function)
    H_dep = P(T, V)*V - R*T +2/sqrt(delta**2 - 4*epsilon)*(T*Derivative(a_alpha(T), T) - a_alpha(T))*atanh((2*V+delta)/sqrt(delta**2-4*epsilon))
    print(diff(H_dep, T))

    '''
    kwargs = dict(Tc=507.6, Pc=3025000, omega=0.2975, T=299, P=1e5)
    eos = PR(**kwargs)
    expr = lambda T: eos.to(T=T, V=eos.V_l).H_dep_l
    assert_close(derivative(expr, eos.T, dx=1e-3), eos.dH_dep_dT_l_V, rtol=1e-8)

    expr = lambda T: eos.to(T=T, V=eos.V_g).H_dep_g
    assert_close(derivative(expr, eos.T, dx=1e-3), eos.dH_dep_dT_g_V, rtol=1e-8)

    eos = IG(**kwargs)
    expr = lambda T: eos.to(T=T, V=eos.V_g).H_dep_g
    assert_close(derivative(expr, eos.T, dx=1e-3), eos.dH_dep_dT_g_V, rtol=1e-8)


def test_da_alpha_dP_V():
    kwargs = dict(Tc=507.6, Pc=3025000, omega=0.2975, T=299, P=1e5)
    eos = PR(**kwargs)
    expr = lambda P: eos.to(P=P, V=eos.V_l).a_alpha
    assert_close(derivative(expr, eos.P, dx=eos.P*2e-6, order=11), eos.da_alpha_dP_l_V, rtol=1e-6)

    expr = lambda P: eos.to(P=P, V=eos.V_g).a_alpha
    assert_close(derivative(expr, eos.P, dx=eos.P*2e-6, order=11), eos.da_alpha_dP_g_V, rtol=1e-6)


def test_d2a_alpha_dTdP_V():
    kwargs = dict(Tc=507.6, Pc=3025000, omega=0.2975, T=299, P=1e5)
    eos = PR(**kwargs)
    expr = lambda P: eos.to(P=P, V=eos.V_l).da_alpha_dT
    assert_close(derivative(expr, eos.P, dx=eos.P*4e-6, order=11, n=1), eos.d2a_alpha_dTdP_l_V, rtol=1e-6)
    expr = lambda P: eos.to(P=P, V=eos.V_g).da_alpha_dT
    assert_close(derivative(expr, eos.P, dx=eos.P*4e-6, order=11, n=1), eos.d2a_alpha_dTdP_g_V, rtol=1e-6)


def test_dH_dep_dP_V():
    '''
    from sympy import *
    V, P, R, b, delta, epsilon = symbols('V, P, R, b, delta, epsilon')
    T, f0, f1 = symbols('T, f0, f1', cls=Function) # T, a_alpha and its derivative are functions of P now
    H_dep = P*V - R*T(P) +2/sqrt(delta**2 - 4*epsilon)*(T(P)*f1(P) - f0(P))*atanh((2*V+delta)/sqrt(delta**2-4*epsilon))
    fun = diff(H_dep, P)
    # print(cse(fun, optimizations='basic'))
    print(latex(fun))
    '''
    kwargs = dict(Tc=507.6, Pc=3025000, omega=0.2975, T=299, P=1e5)
    eos = PR(**kwargs)
    expr = lambda P: eos.to(P=P, V=eos.V_l).H_dep_l
    assert_close(derivative(expr, eos.P, dx=eos.P*1e-6, order=11), eos.dH_dep_dP_l_V, rtol=1e-6)

    expr = lambda P: eos.to(P=P, V=eos.V_g).H_dep_g
    assert_close(derivative(expr, eos.P, dx=eos.P*1e-6, order=11), eos.dH_dep_dP_g_V, rtol=1e-6)


def test_dH_dep_dV_g_T_and_P():
    kwargs = dict(Tc=507.6, Pc=3025000, omega=0.2975, T=299, P=1e5)
    eos = PR(**kwargs)

    expr = lambda V: eos.to(T=eos.T, V=V).H_dep_l
    assert_close(derivative(expr, eos.V_l, dx=eos.V_l*1e-6), eos.dH_dep_dV_l_T, rtol=1e-9)

    expr = lambda V: eos.to(T=eos.T, V=V).H_dep_g
    assert_close(derivative(expr, eos.V_g, dx=eos.V_g*1e-6), eos.dH_dep_dV_g_T, rtol=1e-9)

    expr = lambda V: eos.to(P=eos.P, V=V).H_dep_l
    assert_close(derivative(expr, eos.V_l, dx=eos.V_l*1e-6), eos.dH_dep_dV_l_P, rtol=1e-9)

    expr = lambda V: eos.to(P=eos.P, V=V).H_dep_g
    assert_close(derivative(expr, eos.V_g, dx=eos.V_g*1e-6), eos.dH_dep_dV_g_P, rtol=1e-9)


def test_dS_dep_dT_l_V():
    '''
    from sympy import *
    V, T, R, b, delta, epsilon = symbols('V, T, R, b, delta, epsilon')
    P, a_alpha, aalpha2 = symbols('P, a_alpha, a_alpha2', cls=Function)

    # dS_dT|V
    S_dep = R*log(P(T, V)*V/(R*T)) + R*log(V-b)+2*Derivative(a_alpha(T), T)*atanh((2*V+delta)/sqrt(delta**2-4*epsilon))/sqrt(delta**2-4*epsilon)-R*log(V)
    # (cse(diff(S_dep, T), optimizations='basic'))
    print(latex(diff(S_dep, T)))
    '''
    kwargs = dict(Tc=507.6, Pc=3025000, omega=0.2975, T=299, P=1e5)
    eos = PR(**kwargs)

    expr = lambda T: eos.to(T=T, V=eos.V_l).S_dep_l
    assert_close(derivative(expr, eos.T, dx=eos.T*3e-8), eos.dS_dep_dT_l_V, rtol=1e-8)

    expr = lambda T: eos.to(T=T, V=eos.V_g).S_dep_g
    assert_close(derivative(expr, eos.T, dx=eos.T*3e-8), eos.dS_dep_dT_g_V, rtol=1e-8)


def test_dS_dep_dP_V():
    kwargs = dict(Tc=507.6, Pc=3025000, omega=0.2975, T=299, P=1e5)
    eos = PR(**kwargs)

    expr = lambda P: eos.to(P=P, V=eos.V_l).S_dep_l
    assert_close(derivative(expr, eos.P, dx=eos.P*2e-7), eos.dS_dep_dP_l_V, rtol=1e-8)

    expr = lambda P: eos.to(P=P, V=eos.V_g).S_dep_g
    assert_close(derivative(expr, eos.P, dx=eos.P*2e-7), eos.dS_dep_dP_g_V, rtol=1e-8)

def test_dS_dep_dV_P():
    kwargs = dict(Tc=507.6, Pc=3025000, omega=0.2975, T=299, P=1e5)
    eos = PR(**kwargs)

    expr = lambda V: eos.to(P=eos.P, V=V).S_dep_l
    assert_close(derivative(expr, eos.V_l, dx=eos.V_l*2e-7), eos.dS_dep_dV_l_P, rtol=1e-8)

    expr = lambda V: eos.to(P=eos.P, V=V).S_dep_g
    assert_close(derivative(expr, eos.V_g, dx=eos.V_g*2e-7), eos.dS_dep_dV_g_P, rtol=1e-8)

    expr = lambda V: eos.to(T=eos.T, V=V).S_dep_l
    assert_close(derivative(expr, eos.V_l, dx=eos.V_l*1e-8), eos.dS_dep_dV_l_T, rtol=1e-8)

    expr = lambda V: eos.to(T=eos.T, V=V).S_dep_g
    assert_close(derivative(expr, eos.V_g, dx=eos.V_g*2e-8), eos.dS_dep_dV_g_T, rtol=1e-8)


#@pytest.mark.xfail
def test_failure_dP_dV_zero_division():
    Tc = 507.6
    Pc = 3025000.
    omega = 0.2975
    SRK(T=507.5999979692211, P=3024999.9164836705, Tc=Tc, Pc=Pc, omega=omega)

def test_Psat_correlations():
    for EOS in [PR, SRK, RK, VDW]:
        eos = EOS(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
        Ts = linspace(eos.Tc*0.32, eos.Tc, 10) + linspace(eos.Tc*.999, eos.Tc, 10)
        Psats_correlation = []
        Psats_numerical = []
        for T in Ts:
            Psats_correlation.append(eos.Psat(T, polish=False))
            Psats_numerical.append(eos.Psat(T, polish=True))

        # PR - tested up to 1 million points (many fits are much better than 1e-8)
        assert_close1d(Psats_correlation, Psats_numerical, rtol=1e-8)

    # Other EOSs are covered, not sure what points need 1e-6 but generally they are perfect
    for EOS in [PR78, PRSV, PRSV2, APISRK, TWUPR, TWUSRK]:
        eos = EOS(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
        Ts = linspace(eos.Tc*0.32, eos.Tc, 5) + linspace(eos.Tc*.999, eos.Tc, 10)
        Psats_correlation = []
        Psats_numerical = []
        for T in Ts:
            Psats_correlation.append(eos.Psat(T, polish=False))
            Psats_numerical.append(eos.Psat(T, polish=True))

        # PR - tested up to 1 million points
        assert_close1d(Psats_correlation, Psats_numerical, rtol=1e-6)

def test_phi_sat():
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    phi_exp = 0.9985054999720072
    phi_good = eos.phi_sat(250.0, polish=True)
    assert_close(phi_good, phi_exp)
    phi_approx = eos.phi_sat(250.0, polish=False)
    assert_close(phi_good, phi_approx, rtol=1e-6)

def test_dphi_sat_dT():
    T = 399.0
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=T, P=1E6)

    dphi_sat_dT_analytical = eos.dphi_sat_dT(T)
    dphi_sat_dT_num = derivative(lambda T: eos.phi_sat(T), T, order=3, dx=T*1e-5)
    assert_close(dphi_sat_dT_analytical, dphi_sat_dT_num, rtol=1e-7)
    assert_close(dphi_sat_dT_analytical, -0.0017161558583938723, rtol=1e-10)


def test_PRTranslatedTwu():
    from thermo.eos import PRTranslated, PRTranslatedTwu, PR
    from fluids.numerics import linspace
    alpha_coeffs = (0.694911381318495, 0.919907783415812, 1.70412689631515)

    kwargs = dict(Tc=512.5, Pc=8084000.0, omega=0.559, alpha_coeffs=alpha_coeffs, c=0.0)
    eos = PRTranslatedTwu(T=300, P=1e5, **kwargs)

    # Volumes
    assert_close(eos.V_g, 0.024313406330537947, rtol=1e-9)
    assert_close(eos.V_l, 4.791874890665185e-05, rtol=1e-9)

    # same params as PR
    kwargs_PR = dict(Tc=512.5, Pc=8084000.0, omega=0.559)
    eos_PR = PR(T=300, P=1e5, **kwargs_PR)
    assert_close1d([eos.delta, eos.b, eos.epsilon], [eos_PR.delta, eos_PR.b, eos_PR.epsilon], rtol=1e-12)

    # Vapor pressure test
    Ts = linspace(.35*eos.Tc, eos.Tc, 10)
    Ps = [eos.Psat(T, polish=True) for T in Ts]
    Ps_unpolished = [eos.Psat(T, polish=False) for T in Ts]
    assert_close1d(Ps, Ps_unpolished, rtol=1e-9)

    # First implementation of vapor pressure analytical derivative
    assert_close(eos.dPsat_dT(eos.T, polish=True), eos.dPsat_dT(eos.T), rtol=1e-9)

def test_PRTranslatedConsistent_misc():
    # root solution fails
    # volume has nasty issue
    base = PRTranslatedConsistent(Tc=768.0, Pc=1070000.0, omega=0.8805, T=8837.07874361444, P=216556124.0631852)
    V_expect = 0.0007383087409586962
    assert_close(base.V_l, V_expect, rtol=1e-11)


def test_MSRK():
    eos = MSRKTranslated(Tc=507.6, Pc=3025000, omega=0.2975, c=22.0561E-6, M=0.7446, N=0.2476, T=250., P=1E6)
    V_expect = 0.00011692764613229268
    assert_close(eos.V_l, V_expect, rtol=1e-9)
    a_alphas = (4.1104444077070035, -0.008754034661729146, 4.0493421990729605e-05)
    assert_close1d(eos.a_alpha_and_derivatives(eos.T), a_alphas)
    assert_close(eos.a_alpha_and_derivatives(eos.T, full=False), a_alphas[0])

    # Test copies
    TP_copy = eos.to(T=eos.T, P=eos.P)
    assert_close(eos.V_l, TP_copy.V_l, rtol=1e-14)
    assert_close1d(eos.alpha_coeffs, TP_copy.alpha_coeffs, rtol=1e-14)
    assert_close(eos.c, TP_copy.c, rtol=1e-14)

    PV_copy = eos.to(P=eos.P, V=eos.V_l)
    assert_close(PV_copy.T, eos.T, rtol=1e-7)


    # water estimation
    Zc, omega = .235, .344
    M = .4745 + 2.7349*omega*Zc + 6.0984*(omega*Zc )**2
    N = 0.0674 + 2.1031*omega*Zc + 3.9512*(omega*Zc)**2

    eos = MSRKTranslated(Tc=647.3, Pc=221.2e5, omega=0.344, T=299., P=1E6, M=M, N=N)

    # Values are pertty close to tabulated
    Ts = [460, 490, 520, 550, 580, 510]
    Psats_full = []
    Psats = []
    for T in Ts:
        Psats.append(eos.Psat(T, polish=False))
        Psats_full.append(eos.Psat(T, polish=True))
    assert_close1d(Psats_full, Psats, rtol=1e-7)

    # Test estimation
    eos_SRK = SRK(Tc=647.3, Pc=221.2e5, omega=0.344, T=299., P=1E6)
    eos = MSRKTranslated(Tc=647.3, Pc=221.2e5, omega=0.344, T=299., P=1E6)
    assert_close1d(eos.alpha_coeffs, [0.8456055026734339, 0.24705086824600675])
    assert_close(eos.Tsat(101325), eos_SRK.Tsat(101325))
    assert_close(eos.Tsat(1333.2236842105262), eos_SRK.Tsat(1333.2236842105262))


@pytest.mark.slow
def test_eos_P_limits():
    '''Test designed to take some volumes, push the EOS to those limits, and
    check that the EOS either recognizes the limit or solves fine
    '''
    Tcs = [507.6, 647.14, 190.56400000000002, 305.32, 611.7, 405.6, 126.2, 154.58, 512.5]
    Pcs = [3025000.0, 22048320.0, 4599000.0, 4872000.0, 2110000.0, 11277472.5, 3394387.5, 5042945.25, 8084000.0]
    omegas = [0.2975, 0.344, 0.008, 0.098, 0.49, 0.25, 0.04, 0.021, 0.559]
    P_real_high = 1e20
    for eos in eos_list:
        for ci in range(len(Tcs)): #
            Tc, Pc, omega = Tcs[ci], Pcs[ci], omegas[ci]
            Ts = linspace(.01*Tc, Tc*10, 15)
            Ps = logspace(log10(Pc), log10(Pc*100000), 20)
            for T in Ts:
                for P in Ps:
                    max_P = None
                    base = eos(T=T, P=P, Tc=Tc, Pc=Pc, omega=omega)
                    try:
                        V = base.V_l
                    except:
                        V = base.V_g

                    try:
                        e_new = eos(V=V, P=P_real_high, Tc=Tc, Pc=Pc, omega=omega)
                    except Exception as e:
                        Pmax = base.P_max_at_V(V)
                        if Pmax is not None and Pmax < P_real_high:
                            pass
                        else:
                            print(T, P, V, ci, eos.__name__, Tc, Pc, omega, e)
                            raise ValueError("Failed")


def test_T_discriminant_zeros_analytical():
    # VDW
    eos = VDW(Tc=647.14, Pc=22048320.0, omega=0.344, T=200., P=1E6)
    roots_valid = eos.T_discriminant_zeros_analytical(True)
    assert_close1d(roots_valid, [171.53074673774842, 549.7182388464873], rtol=1e-11)
    roots_all = eos.T_discriminant_zeros_analytical(False)
    roots_all_expect = (549.7182388464873, -186.23123149684938, 171.53074673774842)
    assert_close1d(roots_all, roots_all_expect, rtol=1e-11)


    # RK
    eos = RK(Tc=647.14, Pc=22048320.0, omega=0.344, T=200., P=1E6)
    roots_valid = eos.T_discriminant_zeros_analytical(True)
    assert_close1d(roots_valid, [226.54569586014907, 581.5258414845399, 6071.904717499858], rtol=1e-11)
    roots_all = eos.T_discriminant_zeros_analytical(False)
    roots_all_expect = [(-3039.5486755087554-5260.499733964365j), (-3039.5486755087554+5260.499733964365j), (6071.904717499858+0j), (-287.16659607284214-501.5089438353455j), (-287.16659607284214+501.5089438353455j), (65.79460205013443-221.4001851805135j), (65.79460205013443+221.4001851805135j), (581.5258414845399+0j), (-194.3253376956101+136.82750992885255j), (-194.3253376956101-136.82750992885255j), (226.54569586014907+0j)]
    assert_close1d(roots_all, roots_all_expect, rtol=1e-11)


def test_Psats_low_P():
    Tc = 190.564
    kwargs = dict(Tc=Tc, Pc=4599000.0, omega=0.008, T=300, P=1e5)
    Ps = [10**(-20*i) for i in range(5)]
    # Total of 40 points, but they require mpmath
    for eos in (VDW, SRK, PR, RK): # RK
        base = eos(**kwargs)
        for P in Ps:
            T_calc = base.Tsat(P, polish=True)
            assert_close(base.Psat(T_calc, polish=True), P, rtol=1e-9)


def test_dfugacity_dP_g_zero_low_P():
    eos = PR(Tc=647.086, Pc=22048320.0, omega=0.344, T=15, P=3.663117310917741e-199)
    assert eos.dfugacity_dP_g == 1.0

def test_dH_dep_dT_g_zero_low_P():
    eos = PR(Tc=647.086, Pc=22048320.0, omega=0.344, T=14.3, P=4.021583789475729e-210)
    assert_close(eos.dH_dep_dT_g, 0, atol=1e-20)

def test_misc_volume_issues():
    # Case where a_alpha becomes so small, reverts to ideal gas
    obj = PRTranslatedConsistent(Tc=126.2, Pc=3394387.5, omega=0.04, T=1e4, P=1e9)
    assert_close(obj.V_l, 0.00010895770362725658, rtol=1e-13)
    assert not hasattr(obj, 'V_g')

def test_a_alpha_pure():
    from thermo.eos import eos_list
    for e in eos_list:
        obj = e(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400., P=1E6)
        a_alpha0 = obj.a_alpha_pure(obj.T)
        a_alpha1 = obj.a_alpha_and_derivatives_pure(obj.T)[0]
        assert_close(a_alpha0, a_alpha1, rtol=1e-13)


def test_properties_removed_from_default():
    obj = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400., P=1E6)
    assert_close(obj.V_dep_l, -0.0031697118624756325, rtol=1e-10)
    assert_close(obj.V_dep_g, -0.001183908230519499, rtol=1e-10)
    assert_close(obj.U_dep_l, -22942.165709199802, rtol=1e-10)
    assert_close(obj.U_dep_g, -2365.3923474388685, rtol=1e-10)
    assert_close(obj.A_dep_l, 297.2134281175058, rtol=1e-10)
    assert_close(obj.A_dep_g, 210.38840980284795, rtol=1e-10)

    assert_close(obj.beta_l, 0.0026933709177837427, rtol=1e-10)
    assert_close(obj.beta_g, 0.010123223911174954, rtol=1e-10)

    assert_close(obj.kappa_l, 9.335721543829307e-09, rtol=1e-10)
    assert_close(obj.kappa_g, 1.9710669809793286e-06, rtol=1e-10)

    assert_close(obj.Cp_minus_Cv_l, 48.510162249729795, rtol=1e-10)
    assert_close(obj.Cp_minus_Cv_g, 44.54416112806537, rtol=1e-10)

    assert_close(obj.phi_g, 0.7462319487885894, rtol=1e-10)
    assert_close(obj.phi_l, 0.4215970078576569, rtol=1e-10)

    assert_close(obj.fugacity_l, 421597.0078576569, rtol=1e-10)
    assert_close(obj.fugacity_g, 746231.9487885894, rtol=1e-10)


    assert_close(obj.d2T_dPdV_l, 0.06648808797660258, rtol=1e-10)
    assert_close(obj.d2T_dPdV_g, 0.11547009948453864, rtol=1e-10)

    assert_close(obj.d2V_dPdT_l, -3.138778202711144e-14, rtol=1e-10)
    assert_close(obj.d2V_dPdT_g, 4.093861979513936e-11, rtol=1e-10)

    assert_close(obj.d2T_dP2_l, 9.807759281716453e-15, rtol=1e-10)
    assert_close(obj.d2T_dP2_g, 1.6022283800180742e-11, rtol=1e-10)

    assert_close(obj.d2T_dV2_l, -76277125910.67743, rtol=1e-10)
    assert_close(obj.d2T_dV2_g, 47976821.951772854, rtol=1e-10)

    assert_close(obj.d2V_dP2_l, 1.4539647915909343e-19, rtol=1e-10)
    assert_close(obj.d2V_dP2_g, 2.2516313983125958e-15, rtol=1e-10)

    assert_close(obj.d2V_dT2_l, 5.665884245566452e-09, rtol=1e-10)
    assert_close(obj.d2V_dT2_g, -4.890705089284328e-07, rtol=1e-10)

    assert_close(obj.d2P_dT2_g, -2.170585015391721, rtol=1e-10)
    assert_close(obj.d2P_dT2_l, -235.51286126983416, rtol=1e-10)


def test_model_hash_eos():
    # Iterate through all the basic EOSs, and check that the computed model_hash
    # is the same
    eos_iter = list(eos_list)

    Tc = 507.6
    Pc = 3025000
    omega = 0.2975
    for eos in eos_iter:
        e = eos(Tc=Tc, Pc=Pc, omega=omega, T=300, P=1E5)
        e2 = eos(Tc=Tc, Pc=Pc, omega=omega, V=1, P=1E5)
        assert e.model_hash() == e2.model_hash()
        e3 = e.to(V=10, T=30)
        assert e.model_hash() == e3.model_hash()


def test_model_encode_json_eos():
    eos_iter = list(eos_list)

    Tc = 507.6
    Pc = 3025000
    omega = 0.2975
    for eos in eos_iter:
        e = eos(Tc=Tc, Pc=Pc, omega=omega, T=300, P=1E5)
        s = e.as_json()
        assert 'json_version' in s
        assert type(s) is dict
        e1 = eos.from_json(s)
        assert e.__dict__ == e1.__dict__
        assert e == e1
        assert hash(e) == hash(e1)

def test_model_pickleable_eos():
    import pickle
    Tc = 507.6
    Pc = 3025000
    omega = 0.2975
    for eos in eos_list:
        e = eos(Tc=Tc, Pc=Pc, omega=omega, T=300, P=1E5)
        p = pickle.dumps(e)
        e2 = pickle.loads(p)
        assert e.__dict__ == e2.__dict__
        assert e == e2
        assert hash(e) == hash(e2)

def test_eos_does_not_set_double_root_when_same():
    eos = PRTranslatedConsistent(Tc=126.2, Pc=3394387.5, omega=0.04, c=-1.764477338312623e-06, alpha_coeffs=(0.1120624, 0.8782816, 2.0), T=3274.5491628777654, P=278255940.2207035)
    assert eos.phase != 'l/g'

    # Another case - alpha 1e-16
    eos = PRTranslatedConsistent(Tc=126.2, Pc=3394387.5, omega=0.04, c=-1.764477338312623e-06, alpha_coeffs=(0.1120624, 0.8782816, 2.0), T=3944.2060594377003, P=2.1544346900318865)
    assert eos.phase != 'l/g/'

    # Duplicate root case
    obj = PRTranslatedConsistent(Tc=126.2, Pc=3394387.5, omega=0.04, T=3204.081632653062, P=1e9)
    obj.phase != 'l/g'

    obj = PR(Tc=611.7, Pc=2110000.0, omega=0.49, T=613.5907273413233, P=9.326033468834164e-10)
    assert obj.phase != 'l/g'

def test_eos_lnphi():
    '''
    from sympy import *
    P, T, V, R, b, a, delta, epsilon = symbols('P, T, V, R, b, a, delta, epsilon')
    Tc, Pc, omega = symbols('Tc, Pc, omega')
    a_alpha = symbols('a_alpha')
    da_alpha_dT, d2a_alpha_dT2 = symbols('da_alpha_dT, d2a_alpha_dT2')

    CUBIC = R*T/(V-b) - a_alpha/(V*V + delta*V + epsilon) - P

    S_dep = R*log(V-b)+2*da_alpha_dT*atanh((2*V+delta)/sqrt(delta**2-4*epsilon))/sqrt(delta**2-4*epsilon)-R*log(V)
    S_dep += R*log(P*V/(R*T))

    H_dep = 2*atanh((2*V+delta)/sqrt(delta**2-4*epsilon))*(da_alpha_dT*T-a_alpha)/sqrt(delta**2-4*epsilon)
    H_dep += P*V - R*T

    G_dep = H_dep - T*S_dep
    lnphi = G_dep/(R*T)
    '''
    # The numerical issues remaining should be resolved by using 
    # doubledoubles to calculate Z - 1 and V - b very accurately
    # This might be worth doing throughout the code base for the extra 
    # accuracy anyway.
    from thermo.eos import eos_list
    for e in eos_list:
        if e is IG or e is VDW:
            continue
#         for T in linspace(1, 10000, 2):
#         for P in logspace(log10(1e-4), log10(1e10), 10):

        T, P = 400.0, 1e5
        eos = e(Tc=507.6, Pc=3025000.0, omega=0.2975, T=T, P=P)
        if hasattr(eos, 'V_l'):
            lnphi_calc = eos_lnphi(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
            assert_close(lnphi_calc, eos.lnphi_l, rtol=1e-14)
        if hasattr(eos, 'V_g'):
            lnphi_calc = eos_lnphi(eos.T, eos.P, eos.V_g, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
            assert_close(lnphi_calc, eos.lnphi_g, rtol=1e-14)


