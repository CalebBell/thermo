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
from thermo.heat_capacity import *
from thermo.heat_capacity import TRCIG, POLING, CRCSTD, COOLPROP, POLING_CONST, VDI_TABULAR
from random import uniform
from math import *
from fluids.numerics import linspace, logspace, NotBoundedError, assert_close, assert_close1d
from thermo.chemical import lock_properties, Chemical
from scipy.integrate import quad


@pytest.mark.meta_T_dept
def test_HeatCapacityGas():
    EtOH = HeatCapacityGas(CASRN='64-17-5', similarity_variable=0.1953615, MW=46.06844)
    methods = list(EtOH.all_methods)
    methods.remove(VDI_TABULAR)
    Cps_calc = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(305))[1] for i in methods]
    assert_allclose(sorted(Cps_calc), 
                    sorted([66.35085001015844, 66.40063819791762, 66.25918325111196, 71.07236200126606, 65.6, 65.21]),
                    rtol=1e-5)

    # VDI interpolation, treat separately due to change in behavior of scipy in 0.19
    assert_allclose(EtOH.calculate(305, VDI_TABULAR), 74.6763493522965, rtol=1E-4)



    EtOH.tabular_extrapolation_permitted = False
    assert [None]*6 == [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(5000))[1] for i in [TRCIG, POLING, CRCSTD, COOLPROP, POLING_CONST, VDI_TABULAR]]

    with pytest.raises(Exception):
        EtOH.test_method_validity('BADMETHOD', 300)

    assert False == EtOH.test_property_validity(-1)
    assert False == EtOH.test_property_validity(1.01E4)


    Ts = [200, 250, 300, 400, 450]
    props = [1.2, 1.3, 1.4, 1.5, 1.6]
    EtOH.set_tabular_data(Ts=Ts, properties=props, name='test_set')
    EtOH.forced = True
    assert_allclose(1.35441088517, EtOH.T_dependent_property(275), rtol=2E-4)

    assert None == EtOH.T_dependent_property(5000)

    # Case where the limits were nans
    obj = HeatCapacityGas(CASRN='7440-37-1', MW=39.948, similarity_variable=0.025032542304996495)
    assert not isnan(obj.Tmax)
    assert not isnan(obj.Tmin)
    assert not isnan(obj.POLING_Tmin)
    assert not isnan(obj.POLING_Tmax)
@pytest.mark.meta_T_dept
def test_HeatCapacityGas_integrals():
    # Enthalpy integrals
    EtOH = HeatCapacityGas(CASRN='64-17-5', similarity_variable=0.1953615, MW=46.06844)
    dH1 = EtOH.calculate_integral(200, 300, 'TRC Thermodynamics of Organic Compounds in the Gas State (1994)')
    assert_close(dH1, 5828.905647337944)

    dH2 = EtOH.calculate_integral(200, 300, 'Poling et al. (2001)')
    assert_close(dH2, 5851.1980281476)
    
    dH3 = EtOH.calculate_integral(200, 300, 'Poling et al. (2001) constant')
    assert_close(dH3, 6520.999999999999)
    
    dH4 = EtOH.calculate_integral(200, 300, 'CRC Standard Thermodynamic Properties of Chemical Substances')
    assert_close(dH4, 6559.999999999999)
    
    dH4 = EtOH.calculate_integral(200, 300,'Lastovka and Shaw (2013)')
    assert_close(dH4, 6183.016942750752, rtol=1e-5)

    dH5 = EtOH.calculate_integral(200, 300,'CoolProp')
    assert_close(dH5, 5838.118293585357, rtol=5e-5)
    
    dH = EtOH.calculate_integral(200, 300, 'VDI Heat Atlas')
    assert_close(dH, 6610.821140000002)
    
    # Entropy integrals
    dS = EtOH.calculate_integral_over_T(200, 300, 'Poling et al. (2001)')
    assert_close(dS, 23.5341074921551)
        
    dS = EtOH.calculate_integral_over_T(200, 300, 'Poling et al. (2001) constant')
    assert_close(dS, 26.4403796997334)

    dS = EtOH.calculate_integral_over_T(200, 300, 'TRC Thermodynamics of Organic Compounds in the Gas State (1994)')
    assert_close(dS, 23.4427894111345)
    
    dS = EtOH.calculate_integral_over_T(200, 300, 'CRC Standard Thermodynamic Properties of Chemical Substances')
    assert_close(dS, 26.59851109189558)
    
    dS =  EtOH.calculate_integral_over_T(200, 300, 'CoolProp')
    assert_close(dS, 23.487556909586853, rtol=1e-5)
    
    dS = EtOH.calculate_integral_over_T(200, 300, 'Lastovka and Shaw (2013)')
    assert_close(dS, 24.86700348570956, rtol=1e-5)
    
    dS = EtOH.calculate_integral_over_T(200, 300, 'VDI Heat Atlas')
    assert_close(dS, 26.590569427910076)


@pytest.mark.meta_T_dept
def test_HeatCapacitySolid():
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    Cps_calc =  [(NaCl.set_user_methods(i, forced=True), NaCl.T_dependent_property(298.15))[1] for i in NaCl.all_methods]
    Cps_exp = [50.38469032, 50.5, 20.065072074682337]
    assert_allclose(sorted(Cps_calc), sorted(Cps_exp))

    assert [None]*3 == [(NaCl.set_user_methods(i, forced=True), NaCl.T_dependent_property(20000))[1] for i in NaCl.all_methods]

    with pytest.raises(Exception):
        NaCl.test_method_validity('BADMETHOD', 300)

    assert False == NaCl.test_property_validity(-1)
    assert False == NaCl.test_property_validity(1.01E5)

    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    Ts = [200, 300, 400, 500, 600]
    Cps = [12.965044960703908, 20.206353934945987, 28.261467986645872, 37.14292010552292, 46.85389719453655]
    NaCl.set_tabular_data(Ts=Ts, properties=Cps, name='stuff')
    NaCl.forced = True
    assert_close(NaCl.T_dependent_property(275), 18.320355898506502, rtol=1E-5)

    NaCl.tabular_extrapolation_permitted = False
    assert None == NaCl.T_dependent_property(601)


@pytest.mark.meta_T_dept
def test_HeatCapacitySolid_integrals():
    from thermo.heat_capacity import LASTOVKA_S, PERRY151, CRCSTD
    # Enthalpy integrals
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    dH1 = NaCl.calculate_integral(100, 150, LASTOVKA_S)
    assert_close(dH1, 401.58058175282446)
    
    dH2 = NaCl.calculate_integral(100, 150, CRCSTD)
    assert_close(dH2, 2525.0) # 50*50.5
    
    dH3 = NaCl.calculate_integral(100, 150,  PERRY151)
    assert_close(dH3, 2367.097999999999)

    # Tabular integration - not great
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    Ts = [200, 300, 400, 500, 600]
    Cps = [12.965044960703908, 20.206353934945987, 28.261467986645872, 37.14292010552292, 46.85389719453655]
    NaCl.set_tabular_data(Ts=Ts, properties=Cps, name='stuff')
    dH4 = NaCl.calculate_integral(200, 300, 'stuff')
    assert_close(dH4, 1651.8556007162392, rtol=1E-5)
    
    # Entropy integrals
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    dS1 = NaCl.calculate_integral_over_T(100, 150, LASTOVKA_S)
    assert_close(dS1, 3.213071341895563)
    
    dS2 = NaCl.calculate_integral_over_T(100, 150,  PERRY151)
    assert_close(dS2, 19.183508272982)
    
    dS3 = NaCl.calculate_integral_over_T(100, 150, CRCSTD)
    assert_close(dS3, 20.4759879594623)
    
    NaCl = HeatCapacitySolid(CASRN='7647-14-5', similarity_variable=0.0342215, MW=58.442769)
    Ts = [200, 300, 400, 500, 600]
    Cps = [12.965044960703908, 20.206353934945987, 28.261467986645872, 37.14292010552292, 46.85389719453655]
    NaCl.set_tabular_data(Ts=Ts, properties=Cps, name='stuff')
    dS4 = NaCl.calculate_integral_over_T(100, 150, 'stuff')
    assert_close(dS4, 3.00533159156869)


@pytest.mark.meta_T_dept
def test_HeatCapacityLiquid():
    tol = HeatCapacityLiquid(CASRN='108-88-3', MW=92.13842, Tc=591.75, omega=0.257, Cpgm=115.30398669098454, similarity_variable=0.16279853724428964)
    Cpl_calc = [(tol.set_user_methods(i, forced=True), tol.T_dependent_property(330))[1] for i in tol.all_methods]
    Cpls = [165.4728226923247, 166.5239869108539, 166.52164399712314, 175.3439256239127, 166.71561127721478, 157.3, 165.4554033804999, 166.69807427725885, 157.29, 167.3380448453572]
    assert_allclose(sorted(Cpl_calc), sorted(Cpls), rtol=5e-6)

    assert [None]*10 == [(tol.set_user_methods(i, forced=True), tol.T_dependent_property(2000))[1] for i in tol.all_methods]

    with pytest.raises(Exception):
        tol.test_method_validity('BADMETHOD', 300)

    assert False == tol.test_property_validity(-1)
    assert False == tol.test_property_validity(1.01E5)
    assert True == tol.test_property_validity(100)



    propylbenzene = HeatCapacityLiquid(MW=120.19158, CASRN='103-65-1', Tc=638.35)
    Cpl_calc = [(propylbenzene.set_user_methods(i, forced=True), propylbenzene.T_dependent_property(298.15))[1] for i in propylbenzene.all_methods]
    Cpls = [214.6499551694668, 214.69679325320664, 214.7, 214.71]
    assert_allclose(sorted(Cpl_calc), sorted(Cpls))

    ctp = HeatCapacityLiquid(MW=118.58462, CASRN='96-43-5')
    Cpl_calc = [(ctp.set_user_methods(i, forced=True), ctp.T_dependent_property(250))[1] for i in ctp.all_methods]
    Cpls = [134.1186737739494, 134.1496585096233]
    assert_allclose(sorted(Cpl_calc), sorted(Cpls))


@pytest.mark.meta_T_dept
def test_HeatCapacityLiquid_integrals():
    from thermo.heat_capacity import (CRCSTD, COOLPROP, DADGOSTAR_SHAW, 
                                      ROWLINSON_POLING, ROWLINSON_BONDI, 
                                      ZABRANSKY_SPLINE, 
                                      ZABRANSKY_QUASIPOLYNOMIAL,
                                      ZABRANSKY_SPLINE_SAT,
                                      ZABRANSKY_QUASIPOLYNOMIAL_SAT,
                                      ZABRANSKY_QUASIPOLYNOMIAL_C,
                                      ZABRANSKY_SPLINE_C)
    tol = HeatCapacityLiquid(CASRN='108-88-3', MW=92.13842, Tc=591.75, 
          omega=0.257, Cpgm=115.30398669098454, similarity_variable=0.16279853724428964)

    propylbenzene = HeatCapacityLiquid(MW=120.19158, CASRN='103-65-1', Tc=638.35)

    ctp = HeatCapacityLiquid(MW=118.58462, CASRN='96-43-5')

    dH = tol.calculate_integral(200, 300, CRCSTD)
    assert_close(dH, 15730)
        
    dH = tol.calculate_integral(200, 300, COOLPROP)
    assert_close(dH, 14501.714588188637)
    
    dH = tol.calculate_integral(200, 300, DADGOSTAR_SHAW)
    assert_close(dH, 14395.231307169146)
    
    dH = tol.calculate_integral(200, 300, ROWLINSON_POLING)
    assert_close(dH, 17332.447330329327)
    
    dH = tol.calculate_integral(200, 300, ROWLINSON_BONDI)
    assert_close(dH, 17161.367460370562)
    
    dH = tol.calculate_integral(200, 300, ZABRANSKY_SPLINE_C)
    assert_close(dH, 14588.050659771678)
    
    # Test over different coefficient sets
    dH = tol.calculate_integral(200, 500, ZABRANSKY_SPLINE_SAT)
    assert_close(dH, 52806.422778119224)
    
    dH = tol.calculate_integral(200, 300, ZABRANSKY_SPLINE_SAT)
    assert_close(dH, 14588.10920744596)
    
    dH = tol.calculate_integral(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_C)
    assert_close(dH, 14662.031376528757)
    
    dH = propylbenzene.calculate_integral(200, 300, ZABRANSKY_QUASIPOLYNOMIAL)
    assert_close(dH, 19863.944414041936)
    
    dH = propylbenzene.calculate_integral(200, 300, ZABRANSKY_SPLINE)
    assert_close(dH, 19865.186385942456)
    
    dH = ctp.calculate_integral(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_SAT)
    assert_close(dH, 13437.28621249451)
        
    # Entropy integrals
    dS = tol.calculate_integral_over_T(200, 300, CRCSTD)
    assert_close(dS, 63.779661505414275)
    
    dS = tol.calculate_integral_over_T(200, 300, COOLPROP)
    assert_close(dS, 58.50970500781979)
    
    dS = tol.calculate_integral_over_T(200, 300, DADGOSTAR_SHAW)
    assert_close(dS, 57.78686119989654)
    
    dS = tol.calculate_integral_over_T(200, 300, ROWLINSON_POLING)
    assert_close(dS, 70.42885653432398)
    
    dS = tol.calculate_integral_over_T(200, 300, ROWLINSON_BONDI)
    assert_close(dS, 69.73750128980184)
    
    dS = tol.calculate_integral_over_T(200, 300, ZABRANSKY_SPLINE_C)
    assert_close(dS, 58.866392640147374)
    
    dS = tol.calculate_integral_over_T(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_C)
    assert_close(dS, 59.16999297436473)
    
    dS = tol.calculate_integral_over_T(200, 300, ZABRANSKY_SPLINE_SAT)
    assert_close(dS, 58.86648035527116)
         
    dS = tol.calculate_integral_over_T(200, 500, ZABRANSKY_SPLINE_SAT)
    assert_close(dS, 154.94766581118256)
    
    dS = propylbenzene.calculate_integral_over_T(200, 300, ZABRANSKY_QUASIPOLYNOMIAL)
    assert_close(dS, 80.13493128839104)
    
    dS = propylbenzene.calculate_integral_over_T(200, 300, ZABRANSKY_SPLINE)
    assert_close(dS, 80.13636874689294)

    dS = ctp.calculate_integral_over_T(200, 300, ZABRANSKY_QUASIPOLYNOMIAL_SAT)
    assert_close(dS, 54.34708465297109)

    
  
def test_HeatCapacitySolidMixture():
    from thermo import Mixture
    from thermo.heat_capacity import HeatCapacitySolidMixture
    
    m = Mixture(['silver', 'platinum'], ws=[0.95, 0.05])
    obj = HeatCapacitySolidMixture(CASs=m.CASs, HeatCapacitySolids=m.HeatCapacitySolids, MWs=m.MWs)
    
    Cp = obj(m.T, m.P, m.zs, m.ws)
    assert_allclose(Cp, 25.32745719036059)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
        
    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')


def test_HeatCapacityGasMixture():
    from thermo import Mixture
    from thermo.heat_capacity import HeatCapacityGasMixture
    
    m = Mixture(['oxygen', 'nitrogen'], ws=[.4, .6], T=350, P=1E6)
    obj = HeatCapacityGasMixture(CASs=m.CASs, HeatCapacityGases=m.HeatCapacityGases, MWs=m.MWs)
    
    Cp = obj(m.T, m.P, m.zs, m.ws)
    assert_allclose(Cp, 29.361054534307893, rtol=1e-5)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
        
    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')


def test_HeatCapacityLiquidMixture():
    from thermo import Mixture
    from thermo.heat_capacity import HeatCapacityLiquidMixture, SIMPLE
    
    m = Mixture(['water', 'sodium chloride'], ws=[.9, .1], T=301.5)
    obj = HeatCapacityLiquidMixture(MWs=m.MWs, CASs=m.CASs, HeatCapacityLiquids=m.HeatCapacityLiquids)
    
    Cp = obj(m.T, m.P, m.zs, m.ws)
    assert_allclose(Cp, 72.29643435124115)
    
    Cp = obj.calculate(m.T, m.P, m.zs, m.ws, SIMPLE)
    assert_allclose(Cp, 73.715439, rtol=.01)
    
    m = Mixture(['toluene', 'decane'], ws=[.9, .1], T=300)
    obj = HeatCapacityLiquidMixture(CASs=m.CASs, HeatCapacityLiquids=m.HeatCapacityLiquids)
    assert_allclose(obj(m.T, m.P, m.zs, m.ws), 168.29157865567112, rtol=1E-4)

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
        
    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')


@pytest.mark.slow
@pytest.mark.fuzz
def test_locked_integral():
    lock_properties(True)
    obj = Chemical('water').HeatCapacityGas
    
    def to_int(T):
        return obj.calculate(T, 'Best fit')
    for i in range(10):
        T1 = uniform(0, 4000)
        T2 = uniform(0, 4000)
        quad_ans = quad(to_int, T1, T2)[0]
        analytical_ans = obj.calculate_integral(T1, T2, "Best fit")
        assert_close(quad_ans, analytical_ans, rtol=1e-6)
    lock_properties(False)


@pytest.mark.slow
@pytest.mark.fuzz
def test_locked_integral_over_T():
    lock_properties(True)
    obj = Chemical('water').HeatCapacityGas
    
    def to_int(T):
        return obj.calculate(T, 'Best fit')/T
    for i in range(10):
        T1 = uniform(0, 4000)
        T2 = uniform(0, 4000)
        quad_ans = quad(to_int, T1, T2)[0]
        analytical_ans = obj.calculate_integral_over_T(T1, T2, "Best fit")
        assert_close(quad_ans, analytical_ans, rtol=1e-5)
    lock_properties(False)