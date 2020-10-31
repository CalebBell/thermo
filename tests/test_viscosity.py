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
from random import uniform
import pytest
from math import log, log10
import numpy as np
import pandas as pd
from fluids.numerics import assert_close
from fluids.constants import psi, atm, foot, lb
from fluids.core import R2K, F2K
from chemicals.utils import normalize, mixing_simple
from chemicals.viscosity import *
from thermo.viscosity import *
from chemicals.identifiers import check_CAS
from thermo.viscosity import COOLPROP, LUCAS
from thermo.mixture import Mixture
from thermo.viscosity import LALIBERTE_MU, MIXING_LOG_MOLAR, MIXING_LOG_MASS, BROKAW, HERNING_ZIPPERER, WILKE, SIMPLE


@pytest.mark.meta_T_dept
def test_ViscosityLiquid():
    EtOH = ViscosityLiquid(MW=46.06844, Tm=159.05, Tc=514.0, Pc=6137000.0, Vc=0.000168, omega=0.635, Psat=7872.16, Vml=5.8676e-5, CASRN='64-17-5')

    mul_calcs = [(EtOH.set_user_methods(i), EtOH.T_dependent_property(298.15))[1] for i in EtOH.all_methods]
    mul_exp = [0.0010623746999654108, 0.0004191198228004424, 0.0010823506202025659, 0.0010720812586059742, 0.0010713697500000004, 0.0031157679801337825, 0.0010774308462863267, 0.0010823506202025659]
    assert_allclose(sorted(mul_calcs), sorted(mul_exp))

    # Test that methods return None
    EtOH.tabular_extrapolation_permitted = False
    Vml_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(600))[1] for i in EtOH.all_methods]
    assert [None]*8 == Vml_calcs

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')

    # Acetic acid to test Viswanath_Natarajan_2_exponential
    acetic_acid = ViscosityLiquid(CASRN='64-19-7', Tc=590.7)
    mul_calcs = [(acetic_acid.set_user_methods(i), acetic_acid.T_dependent_property(350.0))[1] for i in acetic_acid.all_methods]
    mul_exp = [0.0005744169247310638, 0.0005089289428076254, 0.0005799665143154318, 0.0005727888422607339, 0.000587027903931889]
    assert_allclose(sorted(mul_calcs), sorted(mul_exp))
    assert [None]*5 == [(acetic_acid.set_user_methods(i), acetic_acid.T_dependent_property(650.0))[1] for i in acetic_acid.all_methods]

    # Test Viswanath_Natarajan_2 with boron trichloride
    mu = ViscosityLiquid(CASRN='10294-34-5').T_dependent_property(250)
    assert_allclose(mu, 0.0003389255178814321)
    assert None == ViscosityLiquid(CASRN='10294-34-5').T_dependent_property(350)


    # Ethanol compressed
    EtOH = ViscosityLiquid(MW=46.06844, Tm=159.05, Tc=514.0, Pc=6137000.0, Vc=0.000168, omega=0.635, Psat=7872.16, Vml=5.8676e-5, CASRN='64-17-5')

    assert [False, True] == [EtOH.test_method_validity_P(300, P, COOLPROP) for P in (1E3, 1E5)]
    assert [True, True] == [EtOH.test_method_validity_P(300, P, LUCAS) for P in (1E3, 1E5)]

    assert_allclose(EtOH.calculate_P(298.15, 1E6, LUCAS), 0.0010880229239312313)
    assert_allclose(EtOH.calculate_P(298.15, 1E6, COOLPROP), 0.0010885493279015608)

    EtOH = ViscosityLiquid(MW=46.06844, Tm=159.05, Tc=514.0, Pc=6137000.0, Vc=0.000168, omega=0.635, Psat=7872.16, Vml=5.8676e-5, CASRN='64-17-5')
    # Ethanol data, calculated from CoolProp
    Ts = [275, 300, 350]
    Ps = [1E5, 5E5, 1E6]
    TP_data = [[0.0017455993713216815, 0.0010445175985089377, 0.00045053170256051774], [0.0017495149679815605, 0.0010472128172002075, 0.000452108003076486], [0.0017543973013034444, 0.0010505716944451827, 0.00045406921275411145]]
    EtOH.set_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_allclose(TP_data, recalc_pts)

    EtOH.tabular_extrapolation_permitted = False
    EtOH.forced_P = True
    assert None == EtOH.TP_dependent_property(300, 9E4)
    EtOH.tabular_extrapolation_permitted = True
    assert_allclose(EtOH.TP_dependent_property(300, 9E4), 0.0010445175985089377)

    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')

@pytest.mark.meta_T_dept
def test_ViscosityGas():
    EtOH = ViscosityGas(MW=46.06844, Tc=514.0, Pc=6137000.0, Zc=0.2412, dipole=1.44, Vmg=0.02357, CASRN='64-17-5')

    mug_calcs = [(EtOH.set_user_methods(i), EtOH.T_dependent_property(298.15))[1] for i in EtOH.all_methods]
    mug_exp = [8.934627758386856e-06, 8.933684639927153e-06, 7.414017252400231e-06, 8.772549629893446e-06, 8.5445e-06, 7.902892297231681e-06, 8.805532218477024e-06, 7.536618820670175e-06]
    assert_allclose(sorted(mug_calcs), sorted(mug_exp))

    # Test that methods return None
    EtOH.tabular_extrapolation_permitted = False
    mug_calcs = [(EtOH.set_user_methods(i, forced=True), EtOH.T_dependent_property(6000))[1] for i in EtOH.all_methods]
    assert [None]*8 == mug_calcs

    with pytest.raises(Exception):
        EtOH.test_method_validity(300, 'BADMETHOD')


    # Ethanol compressed
    EtOH = ViscosityGas(MW=46.06844, Tc=514.0, Pc=6137000.0, Zc=0.2412, dipole=1.44, Vmg=0.02357, CASRN='64-17-5')

    assert [True, False] == [EtOH.test_method_validity_P(300, P, COOLPROP) for P in (1E3, 1E5)]

    assert_allclose(EtOH.calculate_P(298.15, 1E3, COOLPROP), 8.77706377246337e-06)


    # Ethanol data, calculated from CoolProp
    Ts = [400, 500, 550]
    Ps = [1E3, 1E4, 1E5]
    TP_data = [[1.18634700291489e-05, 1.4762189560203758e-05, 1.6162732753470533e-05], [1.1862505513959454e-05, 1.4762728590964208e-05, 1.6163602669178767e-05], [1.1853229260926176e-05, 1.4768417536555742e-05, 1.617257402798515e-05]]
    EtOH.set_tabular_data_P(Ts, Ps, TP_data, name='CPdata')
    recalc_pts = [[EtOH.TP_dependent_property(T, P) for T in Ts] for P in Ps]
    assert_allclose(TP_data, recalc_pts)

    EtOH.tabular_extrapolation_permitted = False
    EtOH.forced_P = True
    assert None == EtOH.TP_dependent_property(300, 9E4)
    EtOH.tabular_extrapolation_permitted = True
    assert_allclose(EtOH.TP_dependent_property(300, 9E4), 1.1854259955707653e-05)

    with pytest.raises(Exception):
        EtOH.test_method_validity_P(300, 1E5, 'BADMETHOD')



def test_ViscosityLiquidMixture():
    # DIPPR  1983 manual example
    m = Mixture(['carbon tetrachloride', 'isopropanol'], zs=[0.5, 0.5], T=313.2)

    ViscosityLiquids = [i.ViscosityLiquid for i in m.Chemicals]

    obj = ViscosityLiquidMixture(ViscosityLiquids=ViscosityLiquids, CASs=m.CASs, MWs=m.MWs)
    mu = obj.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(mu, 0.0009956952502281852)

    mu = obj.calculate(m.T, m.P, m.zs, m.ws, MIXING_LOG_MOLAR)
    assert_allclose(mu, 0.0009956952502281852)
    mu = obj.calculate(m.T, m.P, m.zs, m.ws, MIXING_LOG_MASS)
    assert_allclose(mu, 0.0008741268796817256)

    mu = obj.calculate(m.T, m.P, m.zs, m.ws, SIMPLE)
    assert_allclose(mu, 0.0010399923381840628)

    # Test Laliberte
    m = Mixture(['water', 'sulfuric acid'], zs=[0.5, 0.5], T=298.15)
    ViscosityLiquids = [i.ViscosityLiquid for i in m.Chemicals]
    obj = ViscosityLiquidMixture(ViscosityLiquids=ViscosityLiquids, CASs=m.CASs, MWs=m.MWs)
    mu = obj.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(mu, 0.024955325569420893)
    assert obj.sorted_valid_methods == [LALIBERTE_MU]

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')


def test_ViscosityGasMixture():
    # DIPPR  1983 manual example
    m = Mixture(['dimethyl ether', 'sulfur dioxide'], zs=[.95, .05], T=308.2)
    ViscosityGases = [i.ViscosityGas for i in m.Chemicals]
    obj = ViscosityGasMixture(MWs=m.MWs, molecular_diameters=m.molecular_diameters, Stockmayers=m.Stockmayers, CASs=m.CASs, ViscosityGases=ViscosityGases)

    mu =  obj.mixture_property(m.T, m.P, m.zs, m.ws)
    assert_allclose(mu, 9.637173494726528e-06)

    viscosity_gas_mixture_methods = [BROKAW, HERNING_ZIPPERER, WILKE, SIMPLE]
    mus = [obj.calculate(m.T, m.P, m.zs, m.ws, method) for method in viscosity_gas_mixture_methods]
    assert_allclose(mus, [9.637173494726528e-06, 9.672122280295219e-06, 9.642294904686337e-06, 9.638962759382555e-06])

    # Unhappy paths
    with pytest.raises(Exception):
        obj.calculate(m.T, m.P, m.zs, m.ws, 'BADMETHOD')

    with pytest.raises(Exception):
        obj.test_method_validity(m.T, m.P, m.zs, m.ws, 'BADMETHOD')
