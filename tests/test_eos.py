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
import numpy as np
import pytest
from thermo import eos
from thermo.eos import *
from thermo.utils import allclose_variable
from scipy.misc import derivative
from scipy.constants import R
from math import log, exp, sqrt


@pytest.mark.slow
@pytest.mark.sympy
def test_PR_with_sympy():
    # Test with hexane
    from sympy import  Rational, symbols, sqrt, solve, diff, integrate, N

    P, T, V = symbols('P, T, V')
    Tc = Rational('507.6')
    Pc = 3025000
    omega = Rational('0.2975')
    
    X = (-1 + (6*sqrt(2)+8)**Rational(1,3) - (6*sqrt(2)-8)**Rational(1,3))/3
    c1 = (8*(5*X+1)/(49-37*X)) # 0.45724
    c2 = (X/(X+3)) # 0.07780
    
    R = Rational('8.3144598')
    a = c1*R**2*Tc**2/Pc
    b = c2*R*Tc/Pc
    
    kappa = Rational('0.37464')+ Rational('1.54226')*omega - Rational('0.26992')*omega**2
    
    a_alpha = a*(1 + kappa*(1-sqrt(T/Tc)))**2
    PR_formula = R*T/(V-b) - a_alpha/(V*(V+b)+b*(V-b)) - P
    
    
    
    # First test - volume, liquid
    
    T_l, P_l = 299, 1000000
    PR_obj_l = PR(T=T_l, P=P_l, Tc=507.6, Pc=3025000, omega=0.2975)
    solns = solve(PR_formula.subs({T: T_l, P:P_l}))
    solns = [N(i) for i in solns]
    V_l_sympy = float([i for i in solns if i.is_real][0])
    V_l_sympy = 0.00013022208100139964

    assert_allclose(PR_obj_l.V_l, V_l_sympy)

    def numeric_sub_l(expr):
        return float(expr.subs({T: T_l, P:P_l, V:PR_obj_l.V_l}))

    # First derivatives
    dP_dT = diff(PR_formula, T)
    assert_allclose(numeric_sub_l(dP_dT), PR_obj_l.dP_dT_l)

    dP_dV = diff(PR_formula, V)
    assert_allclose(numeric_sub_l(dP_dV), PR_obj_l.dP_dV_l)
    
    dV_dT = -diff(PR_formula, T)/diff(PR_formula, V)
    assert_allclose(numeric_sub_l(dV_dT), PR_obj_l.dV_dT_l)
    
    dV_dP = -dV_dT/diff(PR_formula, T)
    assert_allclose(numeric_sub_l(dV_dP), PR_obj_l.dV_dP_l)
    
    # Checks out with solve as well
    dT_dV = 1/dV_dT
    assert_allclose(numeric_sub_l(dT_dV), PR_obj_l.dT_dV_l)
    
    dT_dP = 1/dP_dT
    assert_allclose(numeric_sub_l(dT_dP), PR_obj_l.dT_dP_l)
    
    # Second derivatives of two variables, easy ones
    
    d2P_dTdV = diff(dP_dT, V)
    assert_allclose(numeric_sub_l(d2P_dTdV), PR_obj_l.d2P_dTdV_l)
    
    d2P_dTdV = diff(dP_dV, T)
    assert_allclose(numeric_sub_l(d2P_dTdV), PR_obj_l.d2P_dTdV_l)
    
    
    # Second derivatives of one variable, easy ones
    d2P_dT2 = diff(dP_dT, T)
    assert_allclose(numeric_sub_l(d2P_dT2), PR_obj_l.d2P_dT2_l)
    d2P_dT2_maple = -506.20125231401374
    assert_allclose(d2P_dT2_maple, PR_obj_l.d2P_dT2_l)

    d2P_dV2 = diff(dP_dV, V)
    assert_allclose(numeric_sub_l(d2P_dV2), PR_obj_l.d2P_dV2_l)
    d2P_dV2_maple = 4.482165856520912834998e+17
    assert_allclose(d2P_dV2_maple, PR_obj_l.d2P_dV2_l)
        
    # Second derivatives of one variable, Hard ones - require a complicated identity
    d2V_dT2 = (-(d2P_dT2*dP_dV - dP_dT*d2P_dTdV)*dP_dV**-2
              +(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*dP_dV**-3*dP_dT)
    assert_allclose(numeric_sub_l(d2V_dT2), PR_obj_l.d2V_dT2_l)
    d2V_dT2_maple = 1.16885136854333385E-9
    assert_allclose(d2V_dT2_maple, PR_obj_l.d2V_dT2_l)
    
    d2V_dP2 = -d2P_dV2/dP_dV**3
    assert_allclose(numeric_sub_l(d2V_dP2), PR_obj_l.d2V_dP2_l)
    d2V_dP2_maple = 9.10336131405833680E-21
    assert_allclose(d2V_dP2_maple, PR_obj_l.d2V_dP2_l)


    d2T_dP2 = -d2P_dT2*dP_dT**-3
    assert_allclose(numeric_sub_l(d2T_dP2), PR_obj_l.d2T_dP2_l)
    d2T_dP2_maple = 2.564684443971313e-15
    assert_allclose(d2T_dP2_maple, PR_obj_l.d2T_dP2_l)
    
    d2T_dV2 = (-(d2P_dV2*dP_dT - dP_dV*d2P_dTdV)*dP_dT**-2
              +(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*dP_dT**-3*dP_dV)
    assert_allclose(numeric_sub_l(d2T_dV2), PR_obj_l.d2T_dV2_l)
    d2T_dV2_maple = -291578941281.8895
    assert_allclose(d2T_dV2_maple, PR_obj_l.d2T_dV2_l)
    
    
    # Second derivatives of two variable, Hard ones - require a complicated identity
    d2T_dPdV = -(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*dP_dT**-3
    assert_allclose(numeric_sub_l(d2T_dPdV), PR_obj_l.d2T_dPdV_l)
    d2T_dPdV_maple = 0.0699417049626260466429
    assert_allclose(d2T_dPdV_maple, PR_obj_l.d2T_dPdV_l)
    
    d2V_dPdT = -(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*dP_dV**-3
    assert_allclose(numeric_sub_l(d2V_dPdT), PR_obj_l.d2V_dPdT_l)
    d2V_dPdT_maple = -3.772507759880541967e-15
    assert_allclose(d2V_dPdT_maple, PR_obj_l.d2V_dPdT_l)
    
    # Cv integral, real slow
    # The Cv integral is possible with a more general form, but not here
    # The S and H integrals don't work in Sympy at present

    
    
def test_PR_quick():
    # Test solution for molar volumes
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    Vs_fast = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_slow = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha, quick=False)
    Vs_expected = [(0.00013022208100139953-0j), (0.001123630932618011+0.0012926962852843173j), (0.001123630932618011-0.0012926962852843173j)]
    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = [3.801259426590328, -0.006647926028616357, 1.6930127618563258e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives(299, quick=True)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)
    
    # PR back calculation for T
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00013022208100139953, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00013022208100139953, quick=False)
    assert_allclose(T_slow, 299)
    
    
    diffs_1 = [582232.4757941157, -3665180614672.2373, 1.588550570914177e-07, -2.7283785033590384e-13, 6295046.681608136, 1.717527004374129e-06]
    diffs_2 = [-506.2012523140166, 4.482165856521269e+17, 1.1688513685432287e-09, 9.103361314057314e-21, -291578941282.6521, 2.564684443970742e-15]
    diffs_mixed = [-3.772507759880179e-15, -20523303691.115646, 0.06994170496262654]
    departures = [-31134.740290463407, -72.47559475426019, 25.165377505266793]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]
    
    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_allclose(i, j)
    
    
    
    
    
    # Test Cp_Dep, Cv_dep
    assert_allclose(eos.Cv_dep_l, 25.165377505266747)
    assert_allclose(eos.Cp_dep_l, 44.50559908690951)
    
    # Exception tests
    a = GCEOS()        
    with pytest.raises(Exception):
        a.a_alpha_and_derivatives(T=300)
        
    with pytest.raises(Exception):
        PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299)
    
    # Integration tests
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_allclose(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

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
    phi_l_expect = 0.022212524527244357
    assert_allclose(phi_l_expect, eos.phi_l)
    assert_allclose(phi_walas, eos.phi_l)
    
    # The formula given in [2]_ must be incorrect!
#    S_dep_walas =  R*(-log(Z - B) + B*D/(2*2**0.5*A*eos.a_alpha)*log((Z+(sqrt(2)+1)*B)/(Z-(sqrt(2)-1)*B)))
#    S_dep_expect = -72.47559475426013
#    assert_allclose(-S_dep_walas, S_dep_expect)
#    assert_allclose(S_dep_expect, eos.S_dep_l)
    
    H_dep_walas = R*eos.T*(1 - Z + A/(2*2**0.5*B)*(1 + D/eos.a_alpha)*log((Z+(sqrt(2)+1)*B)/(Z-(sqrt(2)-1)*B)))
    H_dep_expect = -31134.740290463407
    assert_allclose(-H_dep_walas, H_dep_expect)
    assert_allclose(H_dep_expect, eos.H_dep_l)
    
    # Author's original H_dep, in [1]
    H_dep_orig = R*eos.T*(Z-1) + (eos.T*eos.da_alpha_dT-eos.a_alpha)/(2*2**0.5*eos.b)*log((Z+2.44*B)/(Z-0.414*B))
    assert_allclose(H_dep_orig, H_dep_expect, rtol=5E-3)
    
    # Author's correlation, with the correct constants this time
    H_dep_orig = R*eos.T*(Z-1) + (eos.T*eos.da_alpha_dT-eos.a_alpha)/(2*2**0.5*eos.b)*log((Z+(sqrt(2)+1)*B)/(Z-(sqrt(2)-1)*B))
    assert_allclose(H_dep_orig, H_dep_expect)
    
    # Test against Preos.xlsx
    # chethermo (Elliott, Richard and Lira, Carl T. - 2012 - Introductory Chemical Engineering Thermodynamics)
    # Propane
    e = PR(Tc=369.8, Pc=4.249E6, omega=0.152, T=298, P=1E5)
    assert_allclose(e.V_g, 0.0243660258924206)
    assert_allclose(e.V_l, 8.68172131076956e-05)
    
    
    # The following are imprecise as the approximate constants 2.414 etc were
    # used in chetherm
    assert_allclose(e.fugacity_g, 98364.041542871, rtol=1E-5)
    # not sure the problem with precision with the liquid
    assert_allclose(e.fugacity_l, 781433.379991859, rtol=1E-2)
    
    assert_allclose(e.H_dep_g, -111.990562846069)
    assert_allclose(e.H_dep_l, -16112.7239108382, rtol=1E-5)

    assert_allclose(e.U_dep_g, -70.8841316881251)
    assert_allclose(e.U_dep_l, -13643.6966117489, rtol=1E-5)
    
    assert_allclose(e.S_dep_g, -0.238638957275652)
    assert_allclose(e.S_dep_l, -71.158231517264, rtol=1E-6)
    


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
    
    Ts = np.linspace(507.6*0.32, 504)
    Psats_lit = [Psat(T, Tc=507.6, Pc=3025000, omega=0.2975) for T in Ts]
    Psats_eos = [eos.Psat(T) for T in Ts]
    assert_allclose(Psats_lit, Psats_eos, rtol=1.5E-3)
    
    # Check that fugacities exist for both phases    
    for T, P in zip(Ts, Psats_eos):
        eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=T, P=P)
        assert_allclose(eos.fugacity_l, eos.fugacity_g, rtol=2E-3)
        


def test_PR78():
    eos = PR78(Tc=632, Pc=5350000, omega=0.734, T=299., P=1E6)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [8.351960066075052e-05, -63764.64948050847, -130.737108912626]
    assert_allclose(three_props, expect_props)
    
    # Test the results are identical to PR or lower things
    eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    PR_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    eos = PR78(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    PR78_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    assert_allclose(PR_props, PR78_props)


def test_PRSV():
    eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.0001301268694484059, -31698.916002476708, -74.1674902435042]
    assert_allclose(three_props, expect_props)
    
    # Test of a_alphas
    a_alphas = [3.8129831135199463, -0.006976898745266429, 2.0026547235203598e-05]
    
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    
    a_alphas_fast = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # PR back calculation for T
    eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, V=0.0001301268694484059, P=1E6, kappa1=0.05104)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001301268694484059, quick=False)
    assert_allclose(T_slow, 299)
    
    
    # Test the bool to control its behavior
    eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=406.08, P=1E6, kappa1=0.05104)
    assert_allclose(eos.kappa, 0.7977689278061457)
    eos.kappa1_Tr_limit = True
    eos.__init__(Tc=507.6, Pc=3025000, omega=0.2975, T=406.08, P=1E6, kappa1=0.05104)
    assert_allclose(eos.kappa, 0.8074380841890093)
    
    # Test the limit is not enforced while under Tr =0.7
    eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=304.56, P=1E6, kappa1=0.05104)
    assert_allclose(eos.kappa, 0.8164956255888178)
    eos.kappa1_Tr_limit = True
    eos.__init__(Tc=507.6, Pc=3025000, omega=0.2975, T=304.56, P=1E6, kappa1=0.05104)
    assert_allclose(eos.kappa, 0.8164956255888178)

    with pytest.raises(Exception):
        PRSV(Tc=507.6, Pc=3025000, omega=0.2975, P=1E6, kappa1=0.05104)

def test_PRSV2():
    eos = PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104, kappa2=0.8634, kappa3=0.460)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.00013018821346475254, -31496.173493225753, -73.6152580115141]
    assert_allclose(three_props, expect_props)
    
    # Test of PRSV2 a_alphas
    a_alphas = [3.8054176315098256, -0.00687315871653124, 2.3078008060652167e-05]
    
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_fast = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # PSRV2 back calculation for T
    eos = PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00013018821346475254, P=1E6, kappa1=0.05104, kappa2=0.8634, kappa3=0.460)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00013018821346475254, quick=False)
    assert_allclose(T_slow, 299)

    # Check this is the same as PRSV
    eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104)
    three_props_PRSV = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    eos = PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104)
    three_props_PRSV2 = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    assert_allclose(three_props_PRSV, three_props_PRSV2)
    
    with pytest.raises(Exception):
        PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, T=299.) 


def test_VDW():
    eos = VDW(Tc=507.6, Pc=3025000, T=299., P=1E6)
    three_props = [eos.V_l, eos.H_dep_l, eos.S_dep_l]
    expect_props = [0.00022332978038490077, -13385.722837649315, -32.65922018109096]
    assert_allclose(three_props, expect_props)
    
    # Test of a_alphas
    a_alphas = [2.4841036545673676, 0, 0]
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # Back calculation for P
    eos = VDW(Tc=507.6, Pc=3025000, T=299, V=0.00022332978038490077)
    assert_allclose(eos.P, 1E6)
    
    # Back calculation for T
    eos = VDW(Tc=507.6, Pc=3025000, P=1E6, V=0.00022332978038490077)
    assert_allclose(eos.T, 299)

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
    
    Ts = np.linspace(507.6*.32, 506)
    Psats_lit = [Psat(T, Tc=507.6, Pc=3025000, omega=0.2975) for T in Ts]
    Psats_eos = [eos.Psat(T) for T in Ts]
    assert_allclose(Psats_lit, Psats_eos, rtol=2E-5)
    
    # Check that fugacities exist for both phases    
    for T, P in zip(Ts, Psats_eos):
        eos = VDW(Tc=507.6, Pc=3025000, T=T, P=P)
        assert_allclose(eos.fugacity_l, eos.fugacity_g, rtol=1E-6)

    
def test_RK_quick():
    # Test solution for molar volumes
    eos = RK(Tc=507.6, Pc=3025000, T=299., P=1E6)
    Vs_fast = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_slow = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha, quick=False)
    Vs_expected = [(0.00015189341729751865+0j), (0.0011670650314512406+0.0011171160630875456j), (0.0011670650314512406-0.0011171160630875456j)]
    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = [3.279647547742308, -0.005484360447729613, 2.75135139518208e-05]
    
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # PR back calculation for T
    eos = RK(Tc=507.6, Pc=3025000,  V=0.00015189341729751865, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00015189341729751865, quick=False)
    assert_allclose(T_slow, 299)
    
    
    diffs_1 = [400451.9103658808, -1773163557098.2456, 2.258403680601321e-07, -5.63963767469079e-13, 4427906.350797926, 2.49717874759626e-06]
    diffs_2 = [-664.0592454189432, 1.5385265309755005e+17, 1.5035170900333218e-09, 2.759679192734741e-20, -130527989946.59952, 1.0340837610012813e-14]
    diffs_mixed = [-7.870472890849004e-15, -10000515150.46239, 0.08069822580205277]
    departures = [-26160.833620674082, -63.01311649400543, 39.8439858825612]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]
    
    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_allclose(i, j)
    
    
    
    
    
    # Test Cp_Dep, Cv_dep
    assert_allclose(eos.Cv_dep_l, 39.8439858825612)
    assert_allclose(eos.Cp_dep_l, 58.57054992395785)
        
    # Integration tests
    eos = RK(Tc=507.6, Pc=3025000, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_allclose(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    # One gas phase property
    assert 'g' == RK(Tc=507.6, Pc=3025000, T=499.,P=1E5).phase



    # Compare against some known  in Walas [2] functions
    eos = RK(Tc=507.6, Pc=3025000, T=299., P=1E6)
    V = eos.V_l
    Z = eos.P*V/(R*eos.T)

    phi_walas = exp(Z - 1 - log(Z*(1 - eos.b/V)) - eos.a/(eos.b*R*eos.T**1.5)*log(1 + eos.b/V))
    phi_l_expect = 0.052632270169019224
    assert_allclose(phi_l_expect, eos.phi_l)
    assert_allclose(phi_walas, eos.phi_l)
    
    S_dep_walas = -R*(log(Z*(1 - eos.b/V)) - eos.a/(2*eos.b*R*eos.T**1.5)*log(1 + eos.b/V))
    S_dep_expect = -63.01311649400542
    assert_allclose(-S_dep_walas, S_dep_expect)
    assert_allclose(S_dep_expect, eos.S_dep_l)
    
    H_dep_walas = R*eos.T*(1 - Z + 1.5*eos.a/(eos.b*R*eos.T**1.5)*log(1 + eos.b/V))
    H_dep_expect = -26160.833620674082
    assert_allclose(-H_dep_walas, H_dep_expect)
    assert_allclose(H_dep_expect, eos.H_dep_l)
    

def test_RK_Psat():
    eos = RK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    
    Ts = np.linspace(507.6*0.32, 504, 100)
    Psats_eos = [eos.Psat(T) for T in Ts]
    fugacity_ls, fugacity_gs = [], []
    for T, P in zip(Ts, Psats_eos):
        eos = RK(Tc=507.6, Pc=3025000, omega=0.2975, T=T, P=P)
        fugacity_ls.append(eos.fugacity_l)
        fugacity_gs.append(eos.fugacity_g)
    
    # Fit is very good
    assert_allclose(fugacity_ls, fugacity_gs, rtol=3E-4)
        

def test_SRK_quick():
    # Test solution for molar volumes
    eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    Vs_fast = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_slow = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha, quick=False)
    Vs_expected = [(0.00014682102759032+0j), (0.00116960122630484+0.0013040890734249049j), (0.00116960122630484-0.0013040890734249049j)]
    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = [3.7271789178606376, -0.007332989159328508, 1.947612023379061e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    
    # PR back calculation for T
    eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00014682102759032, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00014682102759032, quick=False)
    assert_allclose(T_slow, 299)
    
    # Derivatives
    diffs_1 = [507071.37815795804, -2693849768980.0884, 1.8823298314439377e-07, -3.7121594957338967e-13, 5312565.222604471, 1.9721089437796854e-06]
    diffs_2 = [-495.525429968177, 2.685153659083702e+17, 1.3462639881888625e-09, 1.3735644012106488e-20, -201856646370.53476, 3.800656805086382e-15]
    diffs_mixed = [-4.991347301209541e-15, -14322106590.423191, 0.06594013142212454]
    departures = [-31754.65309653571, -74.3732468359525, 28.936520816725874]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]
    
    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_allclose(i, j)
    
        
    # Integration tests
    eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_allclose(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    
    # Compare against some known  in Walas [2] functions
    eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    V = eos.V_l
    Z = eos.P*V/(R*eos.T)
    D = -eos.T*eos.da_alpha_dT
    
    S_dep_walas = R*(-log(Z*(1-eos.b/V)) + D/(eos.b*R*eos.T)*log(1 + eos.b/V))
    S_dep_expect = -74.3732468359525
    assert_allclose(-S_dep_walas, S_dep_expect)
    assert_allclose(S_dep_expect, eos.S_dep_l)
    
    H_dep_walas = eos.T*R*(1 - Z + 1/(eos.b*R*eos.T)*(eos.a_alpha+D)*log(1 + eos.b/V))
    H_dep_expect = -31754.65309653571
    assert_allclose(-H_dep_walas, H_dep_expect)
    assert_allclose(H_dep_expect, eos.H_dep_l)

    phi_walas = exp(Z - 1 - log(Z*(1 - eos.b/V)) - eos.a_alpha/(eos.b*R*eos.T)*log(1 + eos.b/V))
    phi_l_expect = 0.02174822767621331
    assert_allclose(phi_l_expect, eos.phi_l)
    assert_allclose(phi_walas, eos.phi_l)

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
    
    Ts = np.linspace(160, 504, 100)
    Psats_lit = [Psat(T, Tc=507.6, Pc=3025000, omega=0.2975) for T in Ts]
    Psats_eos = [eos.Psat(T) for T in Ts]
    assert_allclose(Psats_lit, Psats_eos, rtol=5E-2)
    # Not sure why the fit was so poor for the original author

    fugacity_ls, fugacity_gs = [], []
    for T, P in zip(Ts, Psats_eos):
        eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=T, P=P)
        fugacity_ls.append(eos.fugacity_l)
        fugacity_gs.append(eos.fugacity_g)
        
    assert allclose_variable(fugacity_ls, fugacity_gs, limits=[0, .1, .5], rtols=[3E-2, 1E-3, 3E-4])


def test_APISRK_quick():
    # Test solution for molar volumes
    eos = APISRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    Vs_fast = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_slow = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha, quick=False)
    Vs_expected = [(0.00014681823858766455+0j), (0.0011696026208061676+0.001304203394096485j), (0.0011696026208061676-0.001304203394096485j)]
    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = [3.727474247064678, -0.0073349099227097685, 1.9482539852821945e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # SRK back calculation for T
    eos = APISRK(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00014681823858766455, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00014681823858766455, quick=False)
    assert_allclose(T_slow, 299)
    # with a S1 set
    eos = APISRK(Tc=514.0, Pc=6137000.0, S1=1.678665, S2=-0.216396, P=1E6, V=7.045692682173252e-05)
    assert_allclose(eos.T, 299)
    eos = APISRK(Tc=514.0, Pc=6137000.0, omega=0.635, S2=-0.216396, P=1E6, V=7.184691383223729e-05)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=7.184691383223729e-05, quick=False)
    assert_allclose(T_slow, 299)

    
    eos = APISRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    # Derivatives
    diffs_1 = [507160.19725861016, -2694519535687.8096, 1.8821915764257067e-07, -3.7112367780430196e-13, 5312955.453232907, 1.9717635678142185e-06]
    diffs_2 = [-495.7033432051597, 2.686049371238787e+17, 1.3462136329121424e-09, 1.3729982416974442e-20, -201893579486.30624, 3.80002419401769e-15]
    diffs_mixed = [-4.990227751881803e-15, -14325368140.50364, 0.06593414440492529]
    departures = [-31759.397282361704, -74.38420560550391, 28.946472091343608]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]
    
    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_allclose(i, j)
    
    # Test Cp_Dep, Cv_dep
    assert_allclose(eos.Cv_dep_l, 28.946472091343608)
    assert_allclose(eos.Cp_dep_l, 49.17373456158243)
        
    # Integration tests
    eos = APISRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_allclose(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    # Error checking
    with pytest.raises(Exception):
        APISRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299.) 
    with pytest.raises(Exception):
        APISRK(Tc=507.6, Pc=3025000, P=1E6,  T=299.)
    

def test_TWUPR_quick():
    # Test solution for molar volumes
    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    Vs_fast = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_slow = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha, quick=False)
    Vs_expected = [(0.0001301754975832377+0j), (0.0011236542243270918+0.0012949257976571766j), (0.0011236542243270918-0.0012949257976571766j)]

    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = [3.806982284033079, -0.006971709974815854, 2.3667018824561144e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T
    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, V=0.0001301754975832377, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.0001301754975832377, quick=False)
    assert_allclose(T_slow, 299)

    
    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    # Derivatives
    diffs_1 = [592877.7698667891, -3683686154532.3066, 1.6094687359218388e-07, -2.7146720921640605e-13, 6213230.351611896, 1.6866883037707508e-06]
    diffs_2 = [-708.101408196832, 4.512488462413035e+17, 1.168546207434993e-09, 9.027515426758444e-21, -280283966933.572, 3.397816790678971e-15]
    diffs_mixed = [-3.82370615408822e-15, -20741143317.758797, 0.07152333089484428]
    departures = [-31652.726391608117, -74.1128253091799, 35.189125483239366]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]
    
    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_allclose(i, j)
    
    # Test Cp_Dep, Cv_dep
    assert_allclose(eos.Cv_dep_l, 35.189125483239366)
    assert_allclose(eos.Cp_dep_l, 55.40579090446679)
        
    # Integration tests
    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_allclose(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    # Error checking
    with pytest.raises(Exception):
        TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=299.) 
        
    # Superctitical test
    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=900., P=1E6)
    eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, V=0.0073716980824289815, P=1E6)
    assert_allclose(eos.T, 900)
    
    
    
def test_TWUSRK_quick():
    # Test solution for molar volumes
    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    Vs_fast = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha)
    Vs_slow = eos.volume_solutions(299, 1E6, eos.b, eos.delta, eos.epsilon, eos.a_alpha, quick=False)
    Vs_expected = [(0.00014689217317770398+0j), (0.001169565653511148+0.0013011778220658073j), (0.001169565653511148-0.0013011778220658073j)]

    assert_allclose(Vs_fast, Vs_expected)
    assert_allclose(Vs_slow, Vs_expected)
    
    # Test of a_alphas
    a_alphas = [3.71966709357206, -0.007269721309490377, 2.305588658885629e-05]
    a_alphas_fast = eos.a_alpha_and_derivatives(299)
    assert_allclose(a_alphas, a_alphas_fast)
    a_alphas_slow = eos.a_alpha_and_derivatives(299, quick=False)
    assert_allclose(a_alphas, a_alphas_slow)

    # back calculation for T
    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00014689217317770398, P=1E6)
    assert_allclose(eos.T, 299)
    T_slow = eos.solve_T(P=1E6, V=0.00014689217317770398, quick=False)
    assert_allclose(T_slow, 299)

    
    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    # Derivatives
    diffs_1 = [504446.40946384566, -2676841551251.3125, 1.8844836341846153e-07, -3.735745956022468e-13, 5306493.417400694, 1.982371132471449e-06]
    diffs_2 = [-586.164516927993, 2.6624358487625542e+17, 1.308861795972448e-09, 1.3880693263695398e-20, -195576504984.95178, 4.566404923205853e-15]
    diffs_mixed = [-5.015403880635795e-15, -14235388178.812284, 0.06816570409464781]
    departures = [-31612.591872087483, -74.02294100343829, 34.24266185576879]
    known_derivs_deps = [diffs_1, diffs_2, diffs_mixed, departures]
    
    for f in [True, False]:
        main_calcs = eos.derivatives_and_departures(eos.T, eos.P, eos.V_l, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=f)
        
        for i, j in zip(known_derivs_deps, main_calcs):
            assert_allclose(i, j)
    
    # Test Cp_Dep, Cv_dep
    assert_allclose(eos.Cv_dep_l, 34.24266185576879)
    assert_allclose(eos.Cp_dep_l, 54.35177004420726)
        
    # Integration tests
    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299.,V=0.00013)
    fast_vars = vars(eos)
    eos.set_properties_from_solution(eos.T, eos.P, eos.V, eos.b, eos.delta, eos.epsilon, eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2, quick=False)
    slow_vars = vars(eos)
    [assert_allclose(slow_vars[i], j) for (i, j) in fast_vars.items() if isinstance(j, float)]

    # Error checking
    with pytest.raises(Exception):
        TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299.) 
    from thermo.eos import TWU_a_alpha_common
    with pytest.raises(Exception):
        TWU_a_alpha_common(299.0, 507.6, 0.2975, 2.5171086468571824, method='FAIL')
        
    # Superctitical test
    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=900., P=1E6)
    eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, V=0.007422210444471012, P=1E6)
    assert_allclose(eos.T, 900)


@pytest.mark.slow
def test_fuzz_dV_dT_and_d2V_dT2_derivatives():
    from thermo import eos
    eos_list = list(eos.__all__); eos_list.remove('GCEOS')
    eos_list.remove('ALPHA_FUNCTIONS'); eos_list.remove('VDW')
    
    phase_extensions = {True: '_l', False: '_g'}
    derivative_bases_dV_dT = {0:'V', 1:'dV_dT', 2:'d2V_dT2'}
    
    def dV_dT(T, P, eos, order=0, phase=True, Tc=507.6, Pc=3025000., omega=0.2975):
        eos = globals()[eos_list[eos]](Tc=Tc, Pc=Pc, omega=omega, T=T, P=P)
        phase_base = phase_extensions[phase]
        attr = derivative_bases_dV_dT[order]+phase_base
        return getattr(eos, attr)
    
    x, y = [], []
    for eos in range(len(eos_list)):
        for T in np.linspace(.1, 1000, 50):
            for P in np.logspace(np.log10(3E4), np.log10(1E6), 50):
                T, P = float(T), float(P)
                for phase in [True, False]:
                    for order in [1, 2]:
                        try:
                            # If dV_dx_phase doesn't exist, will simply abort and continue the loop
                            numer = derivative(dV_dT, T, dx=1E-4, args=(P, eos, order-1, phase))
                            ana = dV_dT(T=T, P=P, eos=eos, order=order, phase=phase)
                        except:
                            continue
                        x.append(numer)
                        y.append(ana)
    assert allclose_variable(x, y, limits=[.009, .05, .65, .93],rtols=[1E-5, 1E-6, 1E-9, 1E-10])


@pytest.mark.slow
def test_fuzz_dV_dP_and_d2V_dP2_derivatives():
    from thermo import eos
    eos_list = list(eos.__all__); eos_list.remove('GCEOS')
    eos_list.remove('ALPHA_FUNCTIONS'); eos_list.remove('VDW')
    
    phase_extensions = {True: '_l', False: '_g'}
    derivative_bases_dV_dP = {0:'V', 1:'dV_dP', 2:'d2V_dP2'}
    
    def dV_dP(P, T, eos, order=0, phase=True, Tc=507.6, Pc=3025000., omega=0.2975):
        eos = globals()[eos_list[eos]](Tc=Tc, Pc=Pc, omega=omega, T=T, P=P)
        phase_base = phase_extensions[phase]
        attr = derivative_bases_dV_dP[order]+phase_base
        return getattr(eos, attr)
    
    
    x, y = [], []
    for eos in range(len(eos_list)):
        for T in np.linspace(.1, 1000, 50):
            for P in np.logspace(np.log10(3E4), np.log10(1E6), 50):
                T, P = float(T), float(P)
                for phase in [True, False]:
                    for order in [1, 2]:
                        try:
                            # If dV_dx_phase doesn't exist, will simply abort and continue the loop
                            numer = derivative(dV_dP, P, dx=15., args=(T, eos, order-1, phase))
                            ana = dV_dP(T=T, P=P, eos=eos, order=order, phase=phase)
                        except:
                            continue
                        x.append(numer)
                        y.append(ana)
    assert allclose_variable(x, y, limits=[.02, .04, .04, .05, .15, .45, .95],
                            rtols=[1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7, 1E-9])
    
@pytest.mark.slow
def test_fuzz_Psat():
    from thermo import eos
    eos_list = list(eos.__all__); eos_list.remove('GCEOS')
    eos_list.remove('ALPHA_FUNCTIONS'); eos_list.remove('eos_list')
    eos_list.remove('GCEOS_DUMMY')
    
    Tc = 507.6
    Pc = 3025000
    omega = 0.2975
    # Basic test
    e = PR(T=400, P=1E5, Tc=507.6, Pc=3025000, omega=0.2975)
    Psats_expect = [22284.314987503185, 466204.89703879296, 2717294.407158156]
    assert_allclose([e.Psat(300), e.Psat(400), e.Psat(500)], Psats_expect)
    
    
    # Test the relative fugacity errors at the correlated Psat are small
    x = []
    for eos in range(len(eos_list)):
        for T in np.linspace(0.318*Tc, Tc*.99, 100):
            e = globals()[eos_list[eos]](Tc=Tc, Pc=Pc, omega=omega, T=T, P=1E5)
            Psat = e.Psat(T)
            e = e.to_TP(T, Psat)
            rerr = (e.fugacity_l - e.fugacity_g)/e.fugacity_g
            x.append(rerr)

    # Assert the average error is under 0.04%
    assert sum(abs(np.array(x)))/len(x) < 1E-4
    
    # Test Polish is working, and that its values are close to the polynomials
    Psats_solved = []
    Psats_poly = []
    for eos in range(len(eos_list)):
        for T in np.linspace(0.4*Tc, Tc*.99, 50):
            e = globals()[eos_list[eos]](Tc=Tc, Pc=Pc, omega=omega, T=T, P=1E5)
            Psats_poly.append(e.Psat(T))
            Psats_solved.append(e.Psat(T, polish=True))
    assert_allclose(Psats_solved, Psats_poly, rtol=1E-4)

    
@pytest.mark.slow
def test_fuzz_dPsat_dT():
    from thermo import eos
    eos_list = list(eos.__all__); eos_list.remove('GCEOS')
    eos_list.remove('ALPHA_FUNCTIONS'); eos_list.remove('eos_list')
    eos_list.remove('GCEOS_DUMMY')
    
    Tc = 507.6
    Pc = 3025000
    omega = 0.2975
    
    e = PR(T=400, P=1E5, Tc=507.6, Pc=3025000, omega=0.2975)
    dPsats_dT_expect = [938.7777925283981, 10287.225576267781, 38814.74676693623]
    assert_allclose([e.dPsat_dT(300), e.dPsat_dT(400), e.dPsat_dT(500)], dPsats_dT_expect)
    
    # Hammer the derivatives for each EOS in a wide range; most are really 
    # accurate. There's an error around the transition between polynomials 
    # though - to be expected; the derivatives are discontinuous there.
    dPsats_derivative = []
    dPsats_analytical = []
    for eos in range(len(eos_list)):
        for T in np.linspace(0.2*Tc, Tc*.999, 50):
            e = globals()[eos_list[eos]](Tc=Tc, Pc=Pc, omega=omega, T=T, P=1E5)
            anal = e.dPsat_dT(T)
            numer = derivative(e.Psat, T, order=9)
            dPsats_analytical.append(anal)
            dPsats_derivative.append(numer)
    assert allclose_variable(dPsats_derivative, dPsats_analytical, limits=[.02, .06], rtols=[1E-5, 1E-7])


def test_Hvaps():
    from thermo import eos
    eos_list = list(eos.__all__); eos_list.remove('GCEOS')
    eos_list.remove('ALPHA_FUNCTIONS'); eos_list.remove('eos_list')
    eos_list.remove('GCEOS_DUMMY')
    
    Tc = 507.6
    Pc = 3025000
    omega = 0.2975

    Hvaps = []
    Hvaps_expect = [31084.972954722154, 31710.347354033467, 31084.972954722154, 31034.19789071903, 31034.19789071903, 13004.11417270758, 26011.811415078664, 31715.119808143718, 31591.421468940156, 31562.23507865849]
    
    for eos in range(len(eos_list)):
        e = globals()[eos_list[eos]](Tc=Tc, Pc=Pc, omega=omega, T=300, P=1E5)
        Hvaps.append(e.Hvap(300))
    
    assert_allclose(Hvaps, Hvaps_expect)



def test_V_l_sats():
    from thermo import eos
    eos_list = list(eos.__all__); eos_list.remove('GCEOS')
    eos_list.remove('ALPHA_FUNCTIONS'); eos_list.remove('eos_list')
    eos_list.remove('GCEOS_DUMMY')
    
    Tc = 507.6
    Pc = 3025000
    omega = 0.2975

    V_l_sats = []
    V_l_sats_expect = [0.00013065653528657878, 0.00014738488907872077, 0.00013065653528657878, 0.00013068333871375792, 0.00013068333871375792, 0.000224969070438342, 0.00015267475707721884, 0.0001473819969852047, 0.00013061078627614464, 0.00014745850642321895]
    
    for eos in range(len(eos_list)):
        e = globals()[eos_list[eos]](Tc=Tc, Pc=Pc, omega=omega, T=300, P=1E5)
        V_l_sats.append(e.V_l_sat(300))
    
    assert_allclose(V_l_sats, V_l_sats_expect)


def test_V_g_sats():
    from thermo import eos
    eos_list = list(eos.__all__); eos_list.remove('GCEOS')
    eos_list.remove('ALPHA_FUNCTIONS'); eos_list.remove('eos_list')
    eos_list.remove('GCEOS_DUMMY')
    
    Tc = 507.6
    Pc = 3025000
    omega = 0.2975
    V_g_sats = []
    V_g_sats_expect = [0.11050456752935825, 0.11367512256304214, 0.11050456752935825, 0.10979754369520009, 0.10979754369520009, 0.009465794716181445, 0.046045503417247724, 0.11374287552483693, 0.11172601823064587, 0.1119690776024331]
    
    for eos in range(len(eos_list)):
        e = globals()[eos_list[eos]](Tc=Tc, Pc=Pc, omega=omega, T=300, P=1E5)
        V_g_sats.append(e.V_g_sat(300))
    
    assert_allclose(V_g_sats, V_g_sats_expect)
