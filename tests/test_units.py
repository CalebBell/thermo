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
    eos = IGMIX(T=115*u.K, P=1*u.MPa, Tcs=[126.1, 190.6]*u.K, Pcs=[33.94E5, 46.04E5]*u.Pa, omegas=[0.04, .008], zs=[0.5, 0.5]*u.dimensionless)
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

