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

from random import uniform
from numpy.testing import assert_allclose
import pytest
import pandas as pd
from thermo.property_package import *
from thermo.chemical import Mixture

def test_Ideal_PP():
    m = Mixture(['ethanol', 'water'], zs=[0.5, 0.5], P=5000, T=298.15)

    vodka = Ideal_PP(m.VaporPressures, m.Tms, m.Tcs, m.Pcs)
    # Low pressure ethanol-water ideal TP flash
    phase, xs, ys, V_over_F = vodka.flash_TP_zs(m.T, m.P, m.zs)
    V_over_F_expect = 0.49376976949268025
    xs_expect = [0.38951827297213176, 0.6104817270278682]
    ys_expect = [0.6132697738819218, 0.3867302261180783]
    assert phase == 'l/g'
    assert_allclose(xs, xs_expect)
    assert_allclose(ys, ys_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    # Same flash with T-VF spec
    phase, xs, ys, V_over_F, P = vodka.flash_TVF_zs(m.T, V_over_F_expect, m.zs)
    assert phase == 'l/g'
    assert_allclose(xs, xs_expect)
    assert_allclose(ys, ys_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    # Same flash with P-VF spec
    phase, xs, ys, V_over_F, T = vodka.flash_PVF_zs(m.P, V_over_F_expect, m.zs)
    assert phase == 'l/g'
    assert_allclose(xs, xs_expect)
    assert_allclose(ys, ys_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    
    # Test the flash interface directly
    T_known = m.T
    V_over_F_known = V_over_F_expect
    zs = m.zs
    
    vodka.flash(T=T_known, VF=V_over_F_known, zs=zs)
    
    P_known = vodka.P
    xs_known = vodka.xs
    ys_known = vodka.ys
    phase_known = vodka.phase
    
    # test TP flash gives the same as TVF
    vodka.flash(T=T_known, P=P_known, zs=zs)    
    assert_allclose(V_over_F_known, vodka.V_over_F)
    assert_allclose(xs_known, vodka.xs)
    assert_allclose(ys_known, vodka.ys)
    assert vodka.phase == phase_known
    # Test PVF flash gives same as well
    vodka.flash(VF=V_over_F_known, P=P_known, zs=zs)
    assert_allclose(xs_known, vodka.xs)
    assert_allclose(ys_known, vodka.ys)
    assert_allclose(xs_known, vodka.xs)
    assert_allclose(T_known, vodka.T)
    assert vodka.phase == phase_known
    
    
@pytest.mark.slow
def test_IdealPP_fuzz_TP_VF():
    m = Mixture(['ethanol', 'water'], zs=[0.5, 0.5], P=5000, T=298.15)
    vodka = Ideal_PP(m.VaporPressures, m.Tms, m.Tcs, m.Pcs)
    
    for i in range(500):
        # May fail right now on the transition between vapor pressure 
        # function boundaries; there are multiple solutions for that case
        # Especially near T = 513.9263246740085 or T = 273.15728497179936
        # Failure is only for PVF flashes
        # There may also be failures for extrapolated vapor pressures, but
        # those are not tested for here.
        zs = [uniform(0, 1) for i in range(2)]
        zs = [i/sum(zs) for i in zs]
        T_known = uniform(200, 513)
        V_over_F_known = uniform(0, 1)
        
        if 273.14 < T_known < 274.15 or 513.85 < T_known < 514.:
            continue

        vodka.flash(T=T_known, VF=V_over_F_known, zs=zs)
        
        P_known = vodka.P
        xs_known = vodka.xs
        ys_known = vodka.ys
        phase_known = vodka.phase
        
        # test TP flash gives the same as TVF
        vodka.flash(T=T_known, P=P_known, zs=zs)    
        assert_allclose(V_over_F_known, vodka.V_over_F)
        assert_allclose(xs_known, vodka.xs)
        assert_allclose(ys_known, vodka.ys)
        assert vodka.phase == phase_known
        # Test PVF flash gives same as well
        vodka.flash(VF=V_over_F_known, P=P_known, zs=zs)
        assert_allclose(xs_known, vodka.xs)
        assert_allclose(ys_known, vodka.ys)
        assert_allclose(xs_known, vodka.xs)
        assert_allclose(T_known, vodka.T)
        assert vodka.phase == phase_known


    names = ['hexane', '2-methylpentane', '3-methylpentane', '2,3-dimethylbutane', '2,2-dimethylbutane']
    m = Mixture(names, zs=[.2, .2, .2, .2, .2], P=1E5, T=300)
    test_pkg = IdealPP(m.VaporPressures, m.Tms, m.Tcs, m.Pcs)
    for i in range(500):
        zs = [uniform(0, 1) for i in range(5)]
        zs = [i/sum(zs) for i in zs]
        T_known = uniform(200, 488.0)
        V_over_F_known = uniform(0, 1)
        
        test_pkg.flash(T=T_known, VF=V_over_F_known, zs=zs)
        
        P_known = test_pkg.P
        xs_known = test_pkg.xs
        ys_known = test_pkg.ys
        phase_known = test_pkg.phase
        
        # test TP flash gives the same as TVF
        test_pkg.flash(T=T_known, P=P_known, zs=zs)    
        assert_allclose(V_over_F_known, test_pkg.V_over_F)
        assert_allclose(xs_known, test_pkg.xs)
        assert_allclose(ys_known, test_pkg.ys)
        assert test_pkg.phase == phase_known
        # Test PVF flash gives same as well
        test_pkg.flash(VF=V_over_F_known, P=P_known, zs=zs)
        assert_allclose(xs_known, test_pkg.xs)
        assert_allclose(ys_known, test_pkg.ys)
        assert_allclose(xs_known, test_pkg.xs)
        assert_allclose(T_known, test_pkg.T)
        assert test_pkg.phase == phase_known

    

def test_UNIFAC_PP():
    m = Mixture(['ethanol', 'water'], zs=[0.5, 0.5], P=6500, T=298.15)
    vodka = UNIFAC_PP(m.UNIFAC_groups, m.VaporPressures, m.Tms, m.Tcs, m.Pcs)

    # Low pressure ethanol-water ideal TP flash
    phase, xs, ys, V_over_F = vodka.flash_TP_zs(m.T, m.P, m.zs)
    V_over_F_expect = 0.7522885045317019    
    xs_expect = [0.2761473052710751, 0.7238526947289249]
    ys_expect = [0.5737096013588943, 0.42629039864110585]
    assert phase == 'l/g'
    assert_allclose(xs, xs_expect)
    assert_allclose(ys, ys_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    # Same flash with T-VF spec
    phase, xs, ys, V_over_F, P = vodka.flash_TVF_zs(m.T, V_over_F_expect, m.zs)
    assert phase == 'l/g'
    assert_allclose(xs, xs_expect, rtol=1E-5)
    assert_allclose(ys, ys_expect, rtol=1E-5)
    assert_allclose(V_over_F, V_over_F_expect, rtol=1E-5)
    # Same flash with P-VF spec
    phase, xs, ys, V_over_F, T = vodka.flash_PVF_zs(m.P, V_over_F_expect, m.zs)
    assert phase == 'l/g'
    assert_allclose(xs, xs_expect, rtol=1E-5)
    assert_allclose(ys, ys_expect, rtol=1E-5)
    assert_allclose(V_over_F, V_over_F_expect, rtol=1E-5)
    
    # Test the flash interface directly
    T_known = m.T
    V_over_F_known = V_over_F_expect
    zs = m.zs
    
    vodka.flash(T=T_known, VF=V_over_F_known, zs=zs)
    
    P_known = vodka.P
    xs_known = vodka.xs
    ys_known = vodka.ys
    phase_known = vodka.phase
    
    # test TP flash gives the same as TVF
    vodka.flash(T=T_known, P=P_known, zs=zs)    
    assert_allclose(V_over_F_known, vodka.V_over_F)
    assert_allclose(xs_known, vodka.xs)
    assert_allclose(ys_known, vodka.ys)
    assert vodka.phase == phase_known
    # Test PVF flash gives same as well
    vodka.flash(VF=V_over_F_known, P=P_known, zs=zs)
    assert_allclose(xs_known, vodka.xs)
    assert_allclose(ys_known, vodka.ys)
    assert_allclose(xs_known, vodka.xs)
    assert_allclose(T_known, vodka.T)
    assert vodka.phase == phase_known


@pytest.mark.slow
def test_UNIFAC_PP_fuzz():
    m = Mixture(['ethanol', 'water'], zs=[0.5, 0.5], P=5000, T=298.15)
    vodka = UNIFAC_PP(m.UNIFAC_groups, m.VaporPressures, m.Tms, m.Tcs, m.Pcs)
    
    for i in range(500):
        zs = [uniform(0, 1) for i in range(2)]
        zs = [i/sum(zs) for i in zs]
        T_known = uniform(274, 513)
        V_over_F_known = uniform(0, 1)
        vodka.flash(T=T_known, VF=V_over_F_known, zs=zs)

        P_known = vodka.P
        xs_known = vodka.xs
        ys_known = vodka.ys
        phase_known = vodka.phase

        # test TP flash gives the same as TVF
        vodka.flash(T=T_known, P=P_known, zs=zs)    
        assert_allclose(V_over_F_known, vodka.V_over_F, rtol=1E-5)
        assert_allclose(xs_known, vodka.xs, rtol=1E-5)
        assert_allclose(ys_known, vodka.ys, rtol=1E-5)
        assert vodka.phase == phase_known
        # Test PVF flash gives same as well
        vodka.flash(VF=V_over_F_known, P=P_known, zs=zs)
        assert_allclose(xs_known, vodka.xs)
        assert_allclose(ys_known, vodka.ys)
        assert_allclose(xs_known, vodka.xs)
        assert_allclose(T_known, vodka.T)
        assert vodka.phase == phase_known


def test_UNIFAC_Dortmund_PP():
    m = Mixture(['ethanol', 'water'], zs=[0.5, 0.5], P=6500, T=298.15)
    vodka = UNIFAC_Dortmund_PP(UNIFAC_groups=m.UNIFAC_Dortmund_groups, VaporPressures=m.VaporPressures, 
                               Tms=m.Tms, Tcs=m.Tcs, Pcs=m.Pcs)
    # Low pressure ethanol-water ideal TP flash
    phase, xs, ys, V_over_F = vodka.flash_TP_zs(m.T, m.P, m.zs)
    V_over_F_expect = 0.721802969194136
    xs_expect = [0.26331608196660095, 0.736683918033399]
    ys_expect = [0.5912226272910779, 0.408777372708922]
    assert phase == 'l/g'
    assert_allclose(xs, xs_expect)
    assert_allclose(ys, ys_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    # Same flash with T-VF spec
    phase, xs, ys, V_over_F, P = vodka.flash_TVF_zs(m.T, V_over_F_expect, m.zs)
    assert phase == 'l/g'
    assert_allclose(xs, xs_expect, rtol=1E-5)
    assert_allclose(ys, ys_expect, rtol=1E-5)
    assert_allclose(V_over_F, V_over_F_expect, rtol=1E-5)
    # Same flash with P-VF spec
    phase, xs, ys, V_over_F, T = vodka.flash_PVF_zs(m.P, V_over_F_expect, m.zs)
    assert phase == 'l/g'
    assert_allclose(xs, xs_expect, rtol=1E-5)
    assert_allclose(ys, ys_expect, rtol=1E-5)
    assert_allclose(V_over_F, V_over_F_expect, rtol=1E-5)
    
    # Test the flash interface directly
    T_known = m.T
    V_over_F_known = V_over_F_expect
    zs = m.zs
    
    vodka.flash(T=T_known, VF=V_over_F_known, zs=zs)
    
    P_known = vodka.P
    xs_known = vodka.xs
    ys_known = vodka.ys
    phase_known = vodka.phase
    
    # test TP flash gives the same as TVF
    vodka.flash(T=T_known, P=P_known, zs=zs)    
    assert_allclose(V_over_F_known, vodka.V_over_F)
    assert_allclose(xs_known, vodka.xs)
    assert_allclose(ys_known, vodka.ys)
    assert vodka.phase == phase_known
    # Test PVF flash gives same as well
    vodka.flash(VF=V_over_F_known, P=P_known, zs=zs)
    assert_allclose(xs_known, vodka.xs)
    assert_allclose(ys_known, vodka.ys)
    assert_allclose(xs_known, vodka.xs)
    assert_allclose(T_known, vodka.T)
    assert vodka.phase == phase_known

