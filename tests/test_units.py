# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from __future__ import division
import pint
import types
import numpy as np
import pytest
import fluids
import thermo
from thermo.units import *
import thermo.units
from fluids.numerics import assert_close, assert_close1d, assert_close2d



def assert_pint_allclose(value, magnitude, units, rtol=1e-7, atol=0):
    assert_close(value.to_base_units().magnitude, magnitude, rtol=rtol, atol=atol)
    if type(units) != dict:
        units = dict(units.dimensionality)
    assert dict(value.dimensionality) == units

def assert_pint_allclose1d(value, magnitude, units, rtol=1e-7, atol=0):
    assert_close1d(value.to_base_units().magnitude, magnitude, rtol=rtol, atol=atol)
    if type(units) != dict:
        units = dict(units.dimensionality)
    assert dict(value.dimensionality) == units

def assert_pint_allclose2d(value, magnitude, units, rtol=1e-7, atol=0):
    assert_close2d(value.to_base_units().magnitude, magnitude, rtol=rtol, atol=atol)
    if type(units) != dict:
        units = dict(units.dimensionality)
    assert dict(value.dimensionality) == units

def test_VDW_units():
    eos = VDW(Tc=507.6*u.K, Pc=3.025000*u.MPa, T=300.*u.K, P=1*u.MPa)
    assert_pint_allclose(eos.U_dep_l, -11108.719659078508, u.J/(u.mol))
    assert_pint_allclose(eos.Zc, 0.375, {})

def test_PR_units():
    eos = PR(Tc=507.6*u.K, Pc=3.025000*u.MPa, T=300.*u.K, P=1*u.MPa, omega=0.015*u.dimensionless)
    assert_pint_allclose(eos.c1, 0.4572355289213822, {})
    assert_pint_allclose(eos.c2, 0.07779607390388846, {})

def test_SRKTranslated_units():
    trans = SRKTranslated(T=305*u.K, P=1.1*u.bar, Tc=512.5*u.K, Pc=8084000.0*u.Pa, omega=0.559, c=-1e-6*u.m**3/u.mol)
    assert_pint_allclose(trans.c, -1e-6, u.m**3/u.mol)

def test_IG_units():
    base = IG(T=300.0*u.K, P=1e6*u.Pa)
    assert_pint_allclose(base.U_dep_g, 0, u.J/(u.mol))
    assert_pint_allclose(base.Cp_dep_g, 0, u.J/(u.mol*u.K))
    assert_pint_allclose(base.S_dep_g, 0, u.J/(u.mol*u.K))
    assert_pint_allclose(base.H_dep_g, 0, u.J/(u.mol))
    assert_pint_allclose(base.dH_dep_dT_g, 0, u.J/(u.mol*u.K))

    ans = base.a_alpha_and_derivatives_pure(300*u.K)
    assert_pint_allclose(ans[0], 0, u("J^2/mol^2/Pa"))
    assert_pint_allclose(ans[1], 0, u("J^2/mol^2/Pa/K"))
    assert_pint_allclose(ans[2], 0, u("J^2/mol^2/Pa/K^2"))

    T = base.solve_T(P=1e8*u.Pa, V=1e-4*u.m**3/u.mol)
    assert_pint_allclose(T, 1202.7235504272605, u.K)

    assert_pint_allclose(base.V_g, 0.002494338785445972, u.m**3/u.mol)
    assert_pint_allclose(base.T, 300, u.K)
    assert_pint_allclose(base.P, 1e6, u.Pa)

def test_IGMIX_units():
    eos = PRMIX(T=115*u.K, P=1*u.MPa, Tcs=[126.1, 190.6]*u.K, Pcs=[33.94E5, 46.04E5]*u.Pa, omegas=[0.04, .008]*u.dimensionless, zs=[0.5, 0.5]*u.dimensionless)

def test_IGMIX_units():
    eos = IGMIX(T=115*u.K, P=1*u.MPa, Tcs=[126.1, 190.6]*u.K, Pcs=[33.94E5, 46.04E5]*u.Pa, omegas=[0.04, .008]*u.dimensionless, zs=[0.5, 0.5]*u.dimensionless)
    assert_pint_allclose(eos.V_g, 0.0009561632010876225, u.m**3/u.mol)
    assert_pint_allclose1d(eos.Tcs, [126.1, 190.6], u.K)
    assert_pint_allclose1d(eos.Pcs, [33.94E5, 46.04E5], u.Pa)
    assert_pint_allclose1d(eos.zs, [0.5, 0.5], {})
    assert_pint_allclose(eos.PIP_g, 1, {})
    assert_pint_allclose(eos.pseudo_Pc, 3999000, u.Pa)
    assert_pint_allclose2d(eos.a_alpha_ijs, [[0, 0],[0,0]], u("J**2/(mol**2*Pa)"))
    assert_pint_allclose(eos.d2P_dT2_g, 0, u.Pa/u.K**2)


@pytest.mark.deprecated
def test_custom_wraps():
    C = Stream(['ethane'], T=200*u.K, zs=[1], n=1*u.mol/u.s)
    D = Stream(['water', 'ethanol'], ns=[1, 2,]*u.mol/u.s, T=300*u.K, P=1E5*u.Pa)
    E = C + D

    assert_pint_allclose(E.zs, [ 0.5,   0.25,  0.25], {})

    assert_pint_allclose(E.T, 200, {'[temperature]': 1.0})


def test_no_bad_units():
    assert not thermo.units.failed_wrapping


def test_wrap_UNIFAC_classmethod():
    from thermo.unifac import DOUFIP2006, DOUFSG
    T = 373.15*u.K
    xs = [0.2, 0.3, 0.1, 0.4]
    chemgroups = [{9: 6}, {78: 6}, {1: 1, 18: 1}, {1: 1, 2: 1, 14: 1}]
    GE = UNIFAC.from_subgroups(T=T, xs=xs, chemgroups=chemgroups, version=1, interaction_data=DOUFIP2006, subgroups=DOUFSG)
    assert_pint_allclose(GE.GE(), 1292.0910446403327, u.J/u.mol)
    

def test_ChemicalConstantsPackage_units():
    obj = ChemicalConstantsPackage(MWs=[18.01528, 106.165]*u.g/u.mol, names=['water', 'm-xylene'],
                             CASs=['7732-18-5', '108-38-3'],
                             smiless=['O', 'CC1=CC(=CC=C1)C'], PubChems=[962, 7929],)
    assert_pint_allclose(obj.MWs[1], .106165, u.kg/u.mol)


def test_VaporPressure_calculate_units():
    EtOH = VaporPressure(Tb=351.39*u.K, Tc=514.0*u.K, Pc=6137000.0*u.Pa, omega=0.635, CASRN='64-17-5')
    ans = EtOH.calculate(300*u.K, 'LEE_KESLER_PSAT')
    assert_pint_allclose(ans, 8491.523803275244, u.Pa)
    # EtOH.method = 'LEE_KESLER_PSAT' # No setter support
    # ans = EtOH.T_dependent_property(300*u.K)
    # assert_pint_allclose(ans, 8491.523803275244, u.Pa)

def test_RegularSolution_units():
    GE = RegularSolution(T=80*u.degC, xs=np.array([.5, .5])*u.dimensionless, Vs=np.array([89, 109])*u.cm**3/u.mol,
                         SPs=np.array([18818.442018403115, 16772.95919031582])*u.Pa**0.5)
    gammas_infinite = GE.gammas_infinite_dilution()
    gammas_infinite_expect = [1.1352128394577634, 1.1680305837879223]
    for i in range(2):
        assert_pint_allclose(gammas_infinite[i], gammas_infinite_expect[i], u.dimensionless)

    get_properties = ['CpE', 'GE', 'HE', 'SE', 'd2GE_dT2', 'd2GE_dTdns', 'd2GE_dTdxs', 'd2GE_dxixjs', 
                      'd2nGE_dTdns', 'd2nGE_dninjs', 'd3GE_dT3', 'd3GE_dxixjxks',
                      'dGE_dT', 'dGE_dns', 'dGE_dxs', 'dHE_dT', 'dHE_dns', 'dHE_dxs',
                      'dSE_dT', 'dSE_dns', 'dSE_dxs', 'dgammas_dT', 'dgammas_dns', 
                      'dnGE_dns', 'dnHE_dns', 'dnSE_dns', 'gammas', 'gammas_infinite_dilution']
    for prop in get_properties:
        res = getattr(GE, prop)()
        assert isinstance(res, pint.Quantity)
    
    
def test_Wilson_units_1():
    '''All the time taken by this function is in pint. Yes, all of it.
    '''
    T = 331.42
    N = 3
    
    A = [[0.0, 3.870101271243586, 0.07939943395502425],
                 [-6.491263271243587, 0.0, -3.276991837288562],
                 [0.8542855660449756, 6.906801837288562, 0.0]]
    
    B = [[0.0, -375.2835, -31.1208],
                 [1722.58, 0.0, 1140.79],
                 [-747.217, -3596.17, -0.0]]
    D = [[-0.0, -0.00791073, -0.000868371],
                 [0.00747788, -0.0, -3.1e-05],
                 [0.00124796, -3e-05, -0.0]]
    
    C = E = F = [[0.0]*N for _ in range(N)]
    
    xs = [0.229, 0.175, 0.596]
    
    lambda_coeffs = [[[A[i][j], B[i][j], C[i][j], D[i][j], E[i][j], F[i][j]] for j in range(N)] for i in range(N)]
    
    model_ABD = Wilson(T=T*u.K, xs=np.array(xs)*u.dimensionless, lambda_as=np.array(A)*u.dimensionless,
                      lambda_bs=np.array(B)*u.K, lambda_ds=np.array(D)/u.K)
    GE_expect = 480.2639266306882
    CpE_expect = 9.654392039281216
    dGE_dxs_expect = [-2199.9758989394595, -2490.5759162306467, -2241.0570605371795]
    gammas_expect = [1.223393433488855, 1.1009459024701462, 1.2052899281172034]

    # From DDBST
    T = (331.42 -273.15)*u.degC
    Vs_ddbst = np.array([74.04, 80.67, 40.73])*u.cm**3/u.mol
    as_ddbst = np.array([[0, 375.2835, 31.1208], [-1722.58, 0, -1140.79], [747.217, 3596.17, 0.0]])*u.K
    bs_ddbst = np.array([[0, -3.78434, -0.67704], [6.405502, 0, 2.59359], [-0.256645, -6.2234, 0]])*u.dimensionless
    cs_ddbst = np.array([[0.0, 7.91073e-3, 8.68371e-4], [-7.47788e-3, 0.0, 3.1e-5], [-1.24796e-3, 3e-5, 0.0]])/u.K
    params = Wilson.from_DDBST_as_matrix(Vs=Vs_ddbst, ais=as_ddbst, bis=bs_ddbst, cis=cs_ddbst, unit_conversion=False)
    xs = np.array([0.229, 0.175, 0.596])*u.dimensionless
    model_from_DDBST = Wilson(T=T, xs=xs, lambda_as=params[0], lambda_bs=params[1], lambda_cs=params[2], lambda_ds=params[3], lambda_es=params[4], lambda_fs=params[5])
    

    model_lambda_coeffs = Wilson(T=T, xs=np.array(xs)*u.dimensionless, lambda_coeffs=np.array(lambda_coeffs)*u.dimensionless)

    for model in (model_ABD, model_from_DDBST, model_lambda_coeffs):
        assert_pint_allclose(model.GE(), GE_expect, u.J/u.mol)
        assert_pint_allclose(model.CpE(), CpE_expect, u.J/u.mol/u.K)
        dGE_dxs = model.dGE_dxs()
        gammas = model.gammas()
        for i in range(3):
            assert_pint_allclose(dGE_dxs[i], dGE_dxs_expect[i], u.J/u.mol)
            assert_pint_allclose(gammas[i], gammas_expect[i], u.dimensionless)
            
    get_properties = ['CpE', 'GE', 'HE', 'SE', 'd2GE_dT2', 'd2GE_dTdns', 'd2GE_dTdxs', 'd2GE_dxixjs', 
                      'd2lambdas_dT2', 'd2nGE_dTdns', 'd2nGE_dninjs', 'd3GE_dT3', 'd3GE_dxixjxks',
                      'd3lambdas_dT3', 'dGE_dT', 'dGE_dns', 'dGE_dxs', 'dHE_dT', 'dHE_dns', 'dHE_dxs',
                      'dSE_dT', 'dSE_dns', 'dSE_dxs', 'dgammas_dT', 'dgammas_dns', 'dlambdas_dT', 
                      'dnGE_dns', 'dnHE_dns', 'dnSE_dns', 'gammas', 'gammas_infinite_dilution', 'lambdas']
    for model in (model_ABD, model_from_DDBST, model_lambda_coeffs):
        for prop in get_properties:
            res = getattr(model, prop)()
            assert isinstance(res, pint.Quantity)