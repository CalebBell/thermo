# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from thermo import *
from fluids.constants import R
from math import log, exp, sqrt, log10
from fluids.numerics import linspace, derivative, logspace, assert_close, assert_close1d, assert_close2d, assert_close3d

# TODO: add unit tests for solid phase identification and sorting

def test_vapor_score_Tpc():
    score = vapor_score_Tpc(T=300.0, Tcs=[304.2, 507.6], zs=[0.21834418746784942, 0.7816558125321506])
    assert_close(score, -163.18879226903942)

def test_vapor_score_Vpc():
    score = vapor_score_Vpc(V=0.00011316308855449715, Vcs=[9.4e-05, 0.000368], zs=[0.21834418746784942, 0.7816558125321506])
    assert_close(score, -0.000195010604079)

def test_vapor_score_Tpc_weighted():
    score = vapor_score_Tpc_weighted(T=300.0, Tcs=[304.2, 507.6], Vcs=[9.4e-05, 0.000368], zs=[0.9752234962374878, 0.024776503762512052])
    assert_close(score, -22.60037521107)

def test_vapor_score_Tpc_Vpc():
    score = vapor_score_Tpc_Vpc(T=300.0, V=0.00011316308855449715, Tcs=[304.2, 507.6], Vcs=[9.4e-05, 0.000368], zs=[0.21834418746784942, 0.7816558125321506])
    assert_close(score, -55.932094761)

def test_vapor_score_Wilson():
    # 1 component
    score = vapor_score_Wilson(T=300.0, P=1e6, zs=[1], Tcs=[304.2], Pcs=[7376460.0], omegas=[0.2252])
    assert_close(score, -5727363.494462478)


    score = vapor_score_Wilson(T=206.40935716944634, P=100.0, zs=[0.5, 0.5], Tcs=[304.2, 507.6], Pcs=[7376460.0, 3025000.0], omegas=[0.2252, 0.2975])
    assert_close(score, 1.074361930956633)

def test_vapor_score_Poling():
    assert_close(vapor_score_Poling(1.0054239121594122e-05), 1.0137457789955244)

def test_vapor_score_PIP():
    score = vapor_score_PIP(0.024809176851423774, 337.0119286073647, -4009021.959558917, 321440573.3615088, -13659.63987996052)
    assert_close(score, 0.016373735005)

def test_vapor_score_Bennett_Schmidt():
    assert_close(vapor_score_Bennett_Schmidt(7.558572848883679e-06), -7.558572848883679e-06)

def test_vapor_score_traces():
    score = vapor_score_traces(zs=[.218, .782], Tcs=[304.2, 507.6], CASs=['124-38-9', '110-54-3'])
    assert_close(score, 0.218)

    score = vapor_score_traces(zs=[.975, .025], Tcs=[304.2, 507.6], CASs=['124-38-9', '110-54-3'])
    assert_close(score, 0.975)

def test_identity_phase_states_basic():
    constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Vcs=[0.000274, 5.6e-05, 0.000168], Pcs=[4414000.0, 22048320.0, 6137000.0], omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844], CASs=['71-36-3', '7732-18-5', '64-17-5'])
    properties = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
                                             HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
                                             HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                             HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),], )

    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    # flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    T, P = 361, 1e5
    # flashN.flash(T=361, P=1e5, zs=[.25, 0.7, .05]).phase_count
    gas = gas.to(T=T, P=P, zs=[0.2384009970908655, 0.5786839935180925, 0.1829150093910419])
    liq0 = liq.to(T=T, P=P, zs=[7.619975052238032e-05, 0.9989622883894993, 0.0009615118599781474])
    liq1 = liq.to(T=T, P=P, zs=[0.6793120076703771, 0.19699746328631124, 0.12369052904331178])

    VLL_methods_here = [VL_ID_VPC, VL_ID_TPC_VPC,
                     VL_ID_POLING, VL_ID_PIP, VL_ID_BS, VL_ID_TRACES]
    LLL_methods_here = [VL_ID_TPC, VL_ID_TPC_VC_WEIGHTED, VL_ID_WILSON]

    for skip_solids in (True, False):
        for m in VLL_methods_here:
            sln = identity_phase_states(phases=[gas, liq0, liq1], constants=constants, correlations=properties,VL_method=m, skip_solids=skip_solids)
            assert sln[0] is gas
            assert sln[1][0] is liq0
            assert sln[1][1] is liq1
            assert not sln[2]

        for m in LLL_methods_here:
            sln = identity_phase_states(phases=[gas, liq0, liq1], constants=constants, correlations=properties,VL_method=m, skip_solids=skip_solids)
            assert sln[0] is None
            assert sln[1][0] is gas
            assert sln[1][1] is liq0
            assert sln[1][2] is liq1
            assert not sln[2]



        betas = [0.027939322463018015, 0.6139152961492583, 0.35814538138772367]
        phases=[gas, liq0, liq1]

    for skip_solids in (True, False):
        for m in VLL_methods_here:
            settings = BulkSettings(VL_ID=m)
            sln = identify_sort_phases(phases=phases, betas=betas, constants=constants, correlations=properties,
                                     settings=settings, skip_solids=skip_solids)
            assert sln[0] is gas
            assert sln[1][0] is liq0
            assert sln[1][1] is liq1
            assert not sln[2]
            assert_close1d(sln[3], betas)

        for m in LLL_methods_here:
            settings = BulkSettings(VL_ID=m)
            sln = identify_sort_phases(phases=phases, betas=betas, constants=constants, correlations=properties,
                                     settings=settings, skip_solids=skip_solids)
            assert sln[0] is None
            assert sln[1][2] is gas
            assert sln[1][0] is liq0
            assert sln[1][1] is liq1
            assert not sln[2]
            assert_close1d(sln[3], [0.6139152961492583, 0.35814538138772367, 0.027939322463018015])

def test_sort_phases_liquids():
    from thermo.phase_identification import VL_ID_METHODS, PROP_SORT, DENSITY_MASS, DENSITY, ISOTHERMAL_COMPRESSIBILITY, HEAT_CAPACITY
    constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Vcs=[0.000274, 5.6e-05, 0.000168], Pcs=[4414000.0, 22048320.0, 6137000.0], omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844], CASs=['71-36-3', '7732-18-5', '64-17-5'])
    properties = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
                                             HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
                                             HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                             HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),], )

    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    # flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    T, P = 361, 1e5
    # flashN.flash(T=361, P=1e5, zs=[.25, 0.7, .05]).phase_count
    gas = gas.to(T=T, P=P, zs=[0.2384009970908655, 0.5786839935180925, 0.1829150093910419])
    liq0 = liq.to(T=T, P=P, zs=[7.619975052238032e-05, 0.9989622883894993, 0.0009615118599781474])
    liq1 = liq.to(T=T, P=P, zs=[0.6793120076703771, 0.19699746328631124, 0.12369052904331178])
    liq0.constants = liq1.constants = constants

    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=DENSITY_MASS, phase_sort_higher_first=False)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close1d([i.rho_mass() for i in liquids], [650.9479399573546, 717.583719673216])

    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=DENSITY_MASS, phase_sort_higher_first=True)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close1d([i.rho_mass() for i in liquids], [717.583719673216, 650.9479399573546])

    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=DENSITY, phase_sort_higher_first=True)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close1d([i.rho_mass() for i in liquids], [717.583719673216, 650.9479399573546])

    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=DENSITY, phase_sort_higher_first=False)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close1d([i.rho_mass() for i in liquids], [650.9479399573546, 717.583719673216])

    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=ISOTHERMAL_COMPRESSIBILITY, phase_sort_higher_first=False)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close1d([i.rho_mass() for i in liquids], [717.583719673216, 650.9479399573546])

    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=ISOTHERMAL_COMPRESSIBILITY, phase_sort_higher_first=True)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close1d([i.rho_mass() for i in liquids], [650.9479399573546, 717.583719673216])

    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=HEAT_CAPACITY, phase_sort_higher_first=False)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close1d([i.rho_mass() for i in liquids], [717.583719673216, 650.9479399573546])

    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=HEAT_CAPACITY, phase_sort_higher_first=True)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close1d([i.rho_mass() for i in liquids], [650.9479399573546, 717.583719673216])

    # Water settings
    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=DENSITY_MASS, phase_sort_higher_first=False,
                           water_sort=WATER_FIRST)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close(liquids[0].zs[1], 0.9989622883894993)

    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=DENSITY_MASS, phase_sort_higher_first=True,
                           water_sort=WATER_LAST)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close(liquids[1].zs[1], 0.9989622883894993)

    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=DENSITY_MASS, phase_sort_higher_first=True,
                           water_sort=WATER_NOT_SPECIAL)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close(liquids[0].zs[1], 0.9989622883894993)

    settings = BulkSettings(liquid_sort_method=PROP_SORT, liquid_sort_prop=DENSITY_MASS, phase_sort_higher_first=False,
                           water_sort=WATER_NOT_SPECIAL)
    liquids, _ = sort_phases(liquids=[liq0, liq1], solids=[], constants=constants, settings=settings)
    assert_close(liquids[1].zs[1], 0.9989622883894993)
