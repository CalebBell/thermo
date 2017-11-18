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
import numpy as np
from thermo.property_package import *
from thermo.chemical import Chemical
from thermo.mixture import Mixture

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
    
    with pytest.raises(Exception):
        vodka.plot_ternary(T=300)
        
    # Test Tdew, Tbubble, Pbubble, Pdew
    T = 298.15
    Pdew = vodka.Pdew(298.15, [0.5, 0.5])
    T_recalc = vodka.Tdew(Pdew, [0.5, 0.5])
    assert_allclose(T_recalc, T)
    assert_allclose(Pdew, 4517, rtol=2E-3)

    T2 = 294.7556209619327
    Pbubble = vodka.Pbubble(T2, [0.5, 0.5])
    assert_allclose(Pbubble, 4517, rtol=2E-3)
    T2_recalc = vodka.Tbubble(4517.277960030594, [0.5, 0.5])
    assert_allclose(T2_recalc, T2)    
    
    
def test_Ideal_PP_single_component():
    m = Mixture(['water'], zs=[1], T=298.15)
    test_pkg = Ideal_PP(m.VaporPressures, m.Tms, m.Tcs, m.Pcs)
    
    # T and P with TP flash
    phase, xs, ys, V_over_F = test_pkg.flash_TP_zs(m.T, m.VaporPressures[0](298.15), m.zs)
    V_over_F_expect = 1
    xs_expect = None
    ys_expect = [1]
    assert phase == 'g'
    assert xs == None
    assert_allclose(ys, ys_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    
    phase, xs, ys, V_over_F = test_pkg.flash_TP_zs(m.T, m.VaporPressures[0](298.15)+1E-10, m.zs)
    V_over_F_expect = 0
    xs_expect = [1]
    ys_expect = None
    assert phase == 'l'
    assert ys == None
    assert_allclose(xs, xs_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    
    # TVF
    phase, xs, ys, V_over_F, P = test_pkg.flash_TVF_zs(m.T, 1, m.zs)
    
    V_over_F_expect = 1
    xs_expect = [1]
    ys_expect = [1]
    assert phase == 'l/g'
    assert xs == xs_expect
    assert_allclose(ys, ys_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    assert_allclose(P, 3167.418523735963, rtol=1E-3)
    
    phase, xs, ys, V_over_F, P = test_pkg.flash_TVF_zs(m.T, 0, m.zs)
    
    V_over_F_expect = 0
    xs_expect = [1]
    ys_expect = [1]
    assert phase == 'l/g'
    assert xs == xs_expect
    assert_allclose(ys, ys_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    
    
    # PVF
    phase, xs, ys, V_over_F, T = test_pkg.flash_PVF_zs(3167, 1, m.zs)
    
    V_over_F_expect = 1
    xs_expect = [1]
    ys_expect = [1]
    assert phase == 'l/g'
    assert xs == xs_expect
    assert_allclose(ys, ys_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    assert_allclose(T, 298.1477829296143, rtol=1E-3)
    
    phase, xs, ys, V_over_F, T = test_pkg.flash_PVF_zs(3167, 0, m.zs)
    
    V_over_F_expect = 0
    xs_expect = [1]
    ys_expect = [1]
    assert phase == 'l/g'
    assert xs == xs_expect
    assert_allclose(ys, ys_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    assert_allclose(V_over_F, V_over_F_expect)
    assert_allclose(T, 298.1477829296143, rtol=1E-3)

    
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
    test_pkg = Ideal_PP(m.VaporPressures, m.Tms, m.Tcs, m.Pcs)
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

    
@pytest.mark.slow
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


@pytest.mark.slow
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


def test_plotting_failures():
    m = Mixture(['ethanol', 'methanol', 'water'], zs=[0.3, 0.3, 0.4], P=5000, T=298.15)
    ternary = Ideal_PP(m.VaporPressures, m.Tms, m.Tcs, m.Pcs)

    with pytest.raises(Exception):
        ternary.plot_Pxy(300)
    with pytest.raises(Exception):
        ternary.plot_Txy(300)
    with pytest.raises(Exception):
        ternary.plot_xy(300)


def test_IdealPPThermodynamic_single_component_H():
    w = Chemical('water')
    EnthalpyVaporization = w.EnthalpyVaporization
    HeatCapacityGas = w.HeatCapacityGas
    VaporPressure = w.VaporPressure
    
    m = Mixture(['water'], zs=[1], T=298.15)
    pkg = IdealPPThermodynamic(VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs, Tcs=m.Tcs, Pcs=m.Pcs, 
                  HeatCapacityLiquids=m.HeatCapacityLiquids, HeatCapacityGases=m.HeatCapacityGases,
                  EnthalpyVaporizations=m.EnthalpyVaporizations)
    
    # Check the enthalpy of vaporization matches at the reference temperature
    pkg.flash(T=298.15, P=1E5, zs=m.zs)
    H_pp = pkg.enthalpy_Cpg_Hvap()
    assert_allclose(H_pp, -EnthalpyVaporization(298.15))
    
    # Check it's pressure independent for the gas (at ref T)
    kw_options = [{'P': w.Psat}, {'P': 100}, {'P': 1E-10}, {'VF': 1}]
    for kw in kw_options:
        pkg.flash(T=298.15, zs=m.zs, **kw)
        H_pp = pkg.enthalpy_Cpg_Hvap()
        assert_allclose(H_pp, 0)
    
        
    # Check it's pressure is independent (so long as it stays liquid)
    kw_options = [{'P': w.Psat+1E-4}, {'P': 1E4}, {'P': 1E10}, {'VF': 0}]
    for kw in kw_options:
        pkg.flash(T=298.15, zs=m.zs, **kw)
        H_pp = pkg.enthalpy_Cpg_Hvap()
        assert_allclose(H_pp, -EnthalpyVaporization(298.15))
    
    # Gas heat capacity along the vapor curve (and above it)
    for T in np.linspace(w.Tm, w.Tc-1):
        for kw in [{'VF': 1}, {'P': VaporPressure(T)*0.5}]:
            pkg.flash(T=T, zs=m.zs, **kw)
            H_pp = pkg.enthalpy_Cpg_Hvap()
            assert_allclose(H_pp, HeatCapacityGas.T_dependent_property_integral(298.15, T))
    
    # Gas heat capacity plus enthalpy of vaporization along the liquid
    for T in np.linspace(w.Tm, w.Tc-1):
        for kw in [{'VF': 0}, {'P': VaporPressure(T)*1.1}]:
            pkg.flash(T=T, zs=m.zs, **kw)
            H_pp = pkg.enthalpy_Cpg_Hvap()
            H_recalc = (HeatCapacityGas.T_dependent_property_integral(298.15, T)
                        -EnthalpyVaporization(T))
            assert_allclose(H_pp, H_recalc)
    
    # Just one basic case at VF = 0.5
    T = 298.15
    pkg.flash(T=T, zs=m.zs, VF=0.5)
    assert_allclose(pkg.enthalpy_Cpg_Hvap(), -0.5*EnthalpyVaporization(T))
    
    # For a variety of vapor fractions and temperatures, check the enthapy is correctly described
    for VF in np.linspace(0., 1, 20):
        for T in np.linspace(w.Tm, w.Tc, 5):
            pkg.flash(T=T, zs=m.zs, VF=VF)
            pkg_calc = pkg.enthalpy_Cpg_Hvap()
            hand_calc = -(1 - VF)*EnthalpyVaporization(T) + HeatCapacityGas.T_dependent_property_integral(298.15, T)
            assert_allclose(pkg_calc, hand_calc)
            
    # Check the liquid and vapor enthalpies are equal at the critical point
    T = w.Tc
    pkg.flash(T=w.Tc, zs=m.zs, VF=1)
    Hvap_Tc_1 = pkg.enthalpy_Cpg_Hvap()
    pkg.flash(T=w.Tc, zs=m.zs, VF=0)
    Hvap_Tc_0 = pkg.enthalpy_Cpg_Hvap()
    assert_allclose(Hvap_Tc_0, Hvap_Tc_1)
    pkg.flash(T=w.Tc, zs=m.zs, VF=0.5)
    Hvap_Tc_half = pkg.enthalpy_Cpg_Hvap()
    assert_allclose(Hvap_Tc_0, Hvap_Tc_half)


def test_IdealPPThermodynamic_binary_H():

    m = Mixture(['water', 'ethanol'], zs=[0.3, 0.7], T=298.15)
    pkg = IdealPPThermodynamic(VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs, Tcs=m.Tcs, Pcs=m.Pcs, 
                  HeatCapacityLiquids=m.HeatCapacityLiquids, HeatCapacityGases=m.HeatCapacityGases,
                  EnthalpyVaporizations=m.EnthalpyVaporizations) 
    
    # Check the enthalpy of vaporization matches at the reference temperature (as a liquid)
    pkg.flash(T=298.15, P=1E5, zs=m.zs)
    H_pp = pkg.enthalpy_Cpg_Hvap()
    assert_allclose(H_pp, (-0.3*m.EnthalpyVaporizations[0](298.15) -0.7*m.EnthalpyVaporizations[1](298.15)))
    # Check the enthalpy of 0 matches at the reference temperature (as a gas)
    pkg.flash(T=298.15, VF=1, zs=m.zs)
    assert_allclose(0, pkg.enthalpy_Cpg_Hvap(), atol=1E-9)
    
    # Check the gas, at various pressure but still Tref, has enthalpy of 0
    pkg.flash(T=298.15, zs=m.zs, VF=1)
    P_dew = pkg.P
    kw_options = [{'P': P_dew}, {'P': 100}, {'P': 1E-10}, {'VF': 1}]
    for kw in kw_options:
        pkg.flash(T=298.15, zs=m.zs, **kw)
        H_pp = pkg.enthalpy_Cpg_Hvap()
        assert_allclose(H_pp, 0, atol=1E-7)
    
    # Check it's pressure is independent (so long as it stays liquid), has enthalpy of 0
    pkg.flash(T=298.15, zs=m.zs, VF=0)
    P_bubble = pkg.P
    
    kw_options = [{'P': P_bubble+1E-4}, {'P': 1E4}, {'P': 1E10}, {'VF': 0}]
    for kw in kw_options:
        pkg.flash(T=298.15, zs=m.zs, **kw)
        H_pp = pkg.enthalpy_Cpg_Hvap()
        H_handcalc = -0.3*m.EnthalpyVaporizations[0](298.15) -0.7*m.EnthalpyVaporizations[1](298.15)
        assert_allclose(H_pp, H_handcalc)
    

    # For a variety of vapor fractions and temperatures, check the enthapy is correctly described
    for VF in np.linspace(0., 1, 6):
        for T in np.linspace(280, 400, 8):
            z1 = uniform(0, 1)
            z2 = 1-z1
            zs = [z1, z2]
            pkg.flash(T=T, zs=zs, VF=VF)
            pkg_calc = pkg.enthalpy_Cpg_Hvap()
            
            # bad hack as the behavior changed after
            if pkg.xs == None:
                pkg.xs = pkg.zs
            
            hand_calc =(-(1 - VF)*(pkg.xs[0]*m.EnthalpyVaporizations[0](T) + pkg.xs[1]*m.EnthalpyVaporizations[1](T)) 
                        + (z1*m.HeatCapacityGases[0].T_dependent_property_integral(298.15, T) + z2*m.HeatCapacityGases[1].T_dependent_property_integral(298.15, T)))
            assert_allclose(pkg_calc, hand_calc)
            
def test_IdealPPThermodynamic_nitrogen_S():
    
    m = Mixture(['nitrogen'], zs=[1], T=298.15)
    pkg = IdealPPThermodynamic(VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs, Tcs=m.Tcs, Pcs=m.Pcs, 
                  HeatCapacityLiquids=m.HeatCapacityLiquids, HeatCapacityGases=m.HeatCapacityGases,
                  EnthalpyVaporizations=m.EnthalpyVaporizations)
    
    # Check the enthalpy of vaporization matches at the reference temperature for a gas
    pkg.flash(T=298.15, P=101325, zs=m.zs)
    S_pp = pkg.entropy_Cpg_Hvap()
    assert_allclose(S_pp, 0, atol=1E-9)
    
    # Check a entropy difference vs coolprop (N2)- 1.5% error
    pkg.flash(T=298.15, P=101325, zs=m.zs)
    S1 = pkg.entropy_Cpg_Hvap()
    pkg.flash(T=298.15, P=2000325, zs=m.zs)
    S2 = pkg.entropy_Cpg_Hvap()
    assert_allclose(S2-S1, -25.16418, rtol=0.015) # 
    
    # Check a entropy difference vs coolprop (N2)- 0.3% error
    pkg.flash(T=298.15, P=101325, zs=m.zs)
    S1 = pkg.entropy_Cpg_Hvap()
    pkg.flash(T=298.15, P=102325, zs=m.zs)
    S2 = pkg.entropy_Cpg_Hvap()
     # 0.3% error with 1 kPa difference
    assert_allclose(S2-S1, -0.08184949145277187, rtol=0.003) # PropsSI('SMOLAR', 'T', 298.15, 'P', 102325, 'N2') - PropsSI('SMOLAR', 'T', 298.15, 'P', 101325, 'N2')
    # S2-S1
    
    # <2.5% error on a 10 MPa/500K N2 vs 298.15 and 1 atm vs coolprop
    pkg.flash(T=298.15, P=101325, zs=m.zs)
    S1 = pkg.entropy_Cpg_Hvap()
    pkg.flash(T=500, P=1E7, zs=m.zs)
    S2 = pkg.entropy_Cpg_Hvap()
    assert_allclose(S2-S1, -23.549468174122012, rtol=0.026) # PropsSI('SMOLAR', 'T', 500, 'P', 1E7, 'N2') - PropsSI('SMOLAR', 'T', 298.15, 'P', 101325, 'N2')
    
    # Entropy change of condensation at the saturation point of 1 bar - very low error
    pkg.flash(VF=1, P=1E5, zs=m.zs)
    S1 = pkg.entropy_Cpg_Hvap()
    pkg.flash(VF=0, P=1E5, zs=m.zs)
    S2 = pkg.entropy_Cpg_Hvap()
    # T_change = PropsSI('T', 'Q', 0, 'P', 1E5, 'N2') # 77.24349973069587
    # dS = PropsSI('SMOLAR', 'Q', 0, 'T', T_change, 'N2') - PropsSI('SMOLAR', 'Q', 1, 'T', T_change, 'N2')
    assert_allclose(S2 - S1, -72.28618677058911, rtol=5E-4)
    
    # Same test as before, 50% condensed
    pkg.flash(VF=1, P=1E5, zs=m.zs)
    S1 = pkg.entropy_Cpg_Hvap()
    pkg.flash(VF=0.5, P=1E5, zs=m.zs)
    S2 = pkg.entropy_Cpg_Hvap()
    assert_allclose(S2 - S1, -72.28618677058911/2, rtol=5E-4)
    
    # Test compressing a liquid doesn't add any entropy
    pkg.flash(VF=0, P=1E5, zs=m.zs)
    S1 = pkg.entropy_Cpg_Hvap()
    T = pkg.T
    pkg.flash(T=T, P=2E5, zs=m.zs)
    S2 = pkg.entropy_Cpg_Hvap()
    assert_allclose(S1-S2, 0)
    
    
def test_IdealPPThermodynamic_enthalpy_Cpl_Cpg_Hvap_binary_Tc_ref():
    w = Chemical('water')
    MeOH = Chemical('methanol')
    
    m = Mixture(['water', 'methanol'], zs=[0.3, 0.7], T=298.15)
    pkg = IdealPPThermodynamic(VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs, Tcs=m.Tcs, Pcs=m.Pcs, 
                  HeatCapacityLiquids=m.HeatCapacityLiquids, HeatCapacityGases=m.HeatCapacityGases,
                  EnthalpyVaporizations=m.EnthalpyVaporizations) 
    pkg.set_T_transitions('Tc')
    
    # Liquid change only, but to the phase change barrier
    pkg.flash(T=298.15+200, VF=0, zs=m.zs)
    dH = pkg.enthalpy_Cpl_Cpg_Hvap()
    dH_hand = (0.3*w.HeatCapacityLiquid.T_dependent_property_integral(298.15, 298.15+200) 
                 +0.7*MeOH.HeatCapacityLiquid.T_dependent_property_integral(298.15, 298.15+200))
    assert_allclose(dH, dH_hand)
    # Flash a minute amount - check the calc still works and the value is the same
    pkg.flash(T=298.15+200, VF=1E-7, zs=m.zs)
    dH = pkg.enthalpy_Cpl_Cpg_Hvap()
    assert_allclose(dH, dH_hand, rtol=1E-6)
    # Flash to vapor at methanol's critical point
    pkg.flash(T=MeOH.Tc, VF=1, zs=m.zs)
    dH = pkg.enthalpy_Cpl_Cpg_Hvap()
    dH_hand = (0.7*MeOH.HeatCapacityLiquid.T_dependent_property_integral(298.15, MeOH.Tc) 
               +0.3*w.HeatCapacityLiquid.T_dependent_property_integral(298.15, w.Tc)
              + 0.3*w.HeatCapacityGas.T_dependent_property_integral(w.Tc, MeOH.Tc))
    assert_allclose(dH, dH_hand)
    # Flash a minute amount more - check the calc still works and the value is the same
    pkg.flash(T=MeOH.Tc, P=pkg.P*.9999999, zs=m.zs)
    dH_minute_diff = pkg.enthalpy_Cpl_Cpg_Hvap()
    assert_allclose(dH, dH_minute_diff)
    # Again
    pkg.flash(T=MeOH.Tc, VF=0.99999999, zs=m.zs)
    dH_minute_diff = pkg.enthalpy_Cpl_Cpg_Hvap()
    assert_allclose(dH, dH_minute_diff)

    # Do a test with 65% liquid
    T = MeOH.Tc
    pkg.flash(T=T, VF=0.35, zs=m.zs)
    dH = pkg.enthalpy_Cpl_Cpg_Hvap()
    
    liq_w_dH = pkg.xs[0]*0.65*w.HeatCapacityLiquid.T_dependent_property_integral(298.15, T)
    liq_MeOH_dH = pkg.xs[1]*0.65*MeOH.HeatCapacityLiquid.T_dependent_property_integral(298.15, T)
    
    dH_w_vapor = 0.35*pkg.ys[0]*(w.HeatCapacityLiquid.T_dependent_property_integral(298.15, w.Tc) 
                                 + w.HeatCapacityGas.T_dependent_property_integral(w.Tc, T))
    dH_MeOH_vapor = 0.35*pkg.ys[1]*(MeOH.HeatCapacityLiquid.T_dependent_property_integral(298.15,T) 
                                 + MeOH.HeatCapacityGas.T_dependent_property_integral(T, T))
    
    dH_hand = dH_MeOH_vapor + dH_w_vapor + liq_MeOH_dH + liq_w_dH
    assert_allclose(dH, dH_hand)
    
    # Full vapor flash, high T
    pkg.flash(T=1200, P=1E7, zs=m.zs)
    dH = pkg.enthalpy_Cpl_Cpg_Hvap()
    
    liq_w_dH = 0.3*w.HeatCapacityLiquid.T_dependent_property_integral(298.15, w.Tc)
    liq_MeOH_dH = 0.7*MeOH.HeatCapacityLiquid.T_dependent_property_integral(298.15, MeOH.Tc)
    dH_w_vapor = 0.3*w.HeatCapacityGas.T_dependent_property_integral(w.Tc, 1200) 
    dH_MeOH_vapor = 0.7*MeOH.HeatCapacityGas.T_dependent_property_integral(MeOH.Tc, 1200) 
    dH_hand = liq_w_dH + liq_MeOH_dH + dH_w_vapor  + dH_MeOH_vapor
    
    assert_allclose(dH_hand, dH)
    
    
def test_IdealPPThermodynamic_enthalpy_Cpl_Cpg_Hvap_binary_Tb_ref():
    w = Chemical('water')
    MeOH = Chemical('methanol')
    
    m = Mixture(['water', 'methanol'], zs=[0.3, 0.7], T=298.15)
    pkg = IdealPPThermodynamic(VaporPressures=m.VaporPressures, Tms=m.Tms, Tbs=m.Tbs, Tcs=m.Tcs, Pcs=m.Pcs, 
                  HeatCapacityLiquids=m.HeatCapacityLiquids, HeatCapacityGases=m.HeatCapacityGases,
                  EnthalpyVaporizations=m.EnthalpyVaporizations) 
    pkg.set_T_transitions('Tb')

    # Full vapor flash, high T
    pkg.flash(T=1200, P=1E7, zs=m.zs)
    dH = pkg.enthalpy_Cpl_Cpg_Hvap()
    
    liq_w_dH = 0.3*w.HeatCapacityLiquid.T_dependent_property_integral(298.15, w.Tb)
    liq_MeOH_dH = 0.7*MeOH.HeatCapacityLiquid.T_dependent_property_integral(298.15, MeOH.Tb)
    dH_w_vapor = 0.3*w.HeatCapacityGas.T_dependent_property_integral(w.Tb, 1200) 
    dH_MeOH_vapor = 0.7*MeOH.HeatCapacityGas.T_dependent_property_integral(MeOH.Tb, 1200) 
    
    liq_w_vap = 0.3*w.EnthalpyVaporization(w.Tb)
    liq_MeOH_vap = 0.7*MeOH.EnthalpyVaporization(MeOH.Tb)
    
    dH_hand = liq_w_dH + liq_MeOH_dH + liq_w_vap + liq_MeOH_vap + dH_w_vapor  + dH_MeOH_vapor
    
    assert_allclose(dH_hand, dH)
    
    # Liquid change only, but to the phase change barrier
    pkg.flash(T=298.15+200, VF=0, zs=m.zs)
    dH = pkg.enthalpy_Cpl_Cpg_Hvap()
    dH_hand = (0.3*w.HeatCapacityLiquid.T_dependent_property_integral(298.15, 298.15+200) 
                 +0.7*MeOH.HeatCapacityLiquid.T_dependent_property_integral(298.15, 298.15+200))
    assert_allclose(dH, dH_hand)
    # Flash a minute amount - check the calc still works and the value is the same
    pkg.flash(T=298.15+200, VF=1E-7, zs=m.zs)
    dH = pkg.enthalpy_Cpl_Cpg_Hvap()
    assert_allclose(dH, dH_hand, rtol=1E-6)
    
    # Flash to vapor at methanol's boiling point
    pkg.flash(T=MeOH.Tb, VF=1, zs=m.zs)
    dH = pkg.enthalpy_Cpl_Cpg_Hvap()
    dH_hand = (0.7*MeOH.HeatCapacityLiquid.T_dependent_property_integral(298.15, MeOH.Tb) 
               +0.3*w.HeatCapacityLiquid.T_dependent_property_integral(298.15, w.Tb)
              + 0.3*w.HeatCapacityGas.T_dependent_property_integral(w.Tb, MeOH.Tb)
              + 0.3*w.EnthalpyVaporization(w.Tb)
              + 0.7*MeOH.EnthalpyVaporization(MeOH.Tb))
    assert_allclose(dH, dH_hand)
    
    # Flash a minute amount more - check the calc still works and the value is the same
    pkg.flash(T=MeOH.Tb, P=pkg.P*.9999999, zs=m.zs)
    dH_minute_diff = pkg.enthalpy_Cpl_Cpg_Hvap()
    assert_allclose(dH, dH_minute_diff)
    # Again
    pkg.flash(T=MeOH.Tb, VF=0.99999999, zs=m.zs)
    dH_minute_diff = pkg.enthalpy_Cpl_Cpg_Hvap()
    assert_allclose(dH, dH_minute_diff)
    
    # Do a test with 65% liquid
    T = 320
    pkg.flash(T=T, VF=0.35, zs=m.zs)
    dH = pkg.enthalpy_Cpl_Cpg_Hvap()
    
    liq_w_dH = pkg.xs[0]*0.65*w.HeatCapacityLiquid.T_dependent_property_integral(298.15, T)
    liq_MeOH_dH = pkg.xs[1]*0.65*MeOH.HeatCapacityLiquid.T_dependent_property_integral(298.15, T)
    
    dH_w_vapor = 0.35*pkg.ys[0]*(w.HeatCapacityLiquid.T_dependent_property_integral(298.15, w.Tb) 
                                 + w.HeatCapacityGas.T_dependent_property_integral(w.Tb, T))
    dH_MeOH_vapor = 0.35*pkg.ys[1]*(MeOH.HeatCapacityLiquid.T_dependent_property_integral(298.15, MeOH.Tb) 
                                 + MeOH.HeatCapacityGas.T_dependent_property_integral(MeOH.Tb, T))
    
    liq_w_vap = pkg.ys[0]*0.35*w.EnthalpyVaporization(w.Tb)
    liq_MeOH_vap = pkg.ys[1]*0.35*MeOH.EnthalpyVaporization(MeOH.Tb)
    
    dH_hand = dH_MeOH_vapor + dH_w_vapor + liq_MeOH_dH + liq_w_dH + liq_MeOH_vap +liq_w_vap
    assert_allclose(dH, dH_hand)
