# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import pytest
import thermo
from thermo import *
from thermo.coolprop import *
from thermo.phases import CoolPropGas, CoolPropLiquid
from fluids.numerics import assert_close
from fluids.numerics import *
from math import *
import json
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass

pure_surfaces_dir = os.path.join(thermo.thermo_dir, '..', 'surfaces', 'pure_coolprop')
pure_fluids = ['water', 'methane', 'ethane', 'decane', 'ammonia', 'nitrogen', 'oxygen', 'methanol', 'eicosane', 'hydrogen']



constants = ChemicalConstantsPackage(Tcs=[647.14, 190.56400000000002, 305.32, 611.7, 405.6, 126.2, 154.58, 512.5,
                                          768.0,
                                          33.2
                                          ],
            Pcs=[22048320.0, 4599000.0, 4872000.0, 2110000.0, 11277472.5, 3394387.5, 5042945.25, 8084000.0,
                 1070000.0,
                 1296960.0
                 ],
            omegas=[0.344, 0.008, 0.098, 0.49, 0.25, 0.04, 0.021, 0.559,
                    0.8805,
                    -0.22,
                    ],
            MWs=[18.01528, 16.04246, 30.06904, 142.28168, 17.03052, 28.0134, 31.9988, 32.04186,
                 282.54748,
                 2.01588
                 ],
            CASs=['7732-18-5', '74-82-8', '74-84-0', '124-18-5', '7664-41-7', '7727-37-9', '7782-44-7', '67-56-1',
                  '112-95-8',
                  '1333-74-0'
                  ])

correlations = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                                                                                   HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                                                                                   HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                                                                                   HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735])),
                                                                                   HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.444966286051841e-23, 9.444106746563928e-20, -1.2490299714587002e-15, 2.6693560979905865e-12, -2.5695131746723413e-09, 1.2022442523089315e-06, -0.00021492132731007108, 0.016616385291696574, 32.84274656062226])),
                                                                                   HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                                                                                   HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
                                                                                   HeatCapacityGas(poly_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924])),
                                                                                   HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.075118433508619e-20, 1.0383055980949049e-16, -2.1577805903757125e-13, 2.373511052680461e-10, -1.4332562489496906e-07, 4.181755403465859e-05, -0.0022544761674344544, -0.15965342941876415, 303.71771182550816])),
                                                                                   HeatCapacityGas(poly_fit=(50.0, 1000.0, [1.1878323802695824e-20, -5.701277266842367e-17, 1.1513022068830274e-13, -1.270076105261405e-10, 8.309937583537026e-08, -3.2694889968431594e-05, 0.007443050245274358, -0.8722920255910297, 66.82863369121873])),
                                                                                   ])


backends = ['HEOS']
@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("backend", backends)
def test_PV_plot(fluid, backend):
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)

    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    gas = CoolPropGas(backend, fluid, T=T, P=P, zs=zs)
    liquid = CoolPropLiquid(backend, fluid, T=T, P=P, zs=zs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

#    CoolPropInfo = coolprop_fluids[pure_const.CASs[0]]
    T_min, T_max, P_max = gas.AS.Tmin(), gas.AS.Tmax(), gas.AS.pmax()
    P_min = PropsSI(fluid, 'PMIN')


    res = flasher.TPV_inputs(zs=zs, pts=50, spec0='T', spec1='P', check0='P', check1='V', prop0='T',
                           trunc_err_low=1e-15,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           Tmax=T_max*(1.0 - 1e-6), Tmin=T_min*(1.0+4e-2),
                           Pmin=P_min*(1.0+1e-2), Pmax=P_max*(1.0 - 1e-5),
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, "PV")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s' %('PV', backend, fluid)

    plot_fig.savefig(os.path.join(path, key + '.png'))
    # TODO log the max error to a file

    plt.close()
#    max_err = np.max(np.abs(errs))
#    limit = 5e-8
#    assert max_err < limit
#for f in pure_fluids:
#    try:
#        test_PV_plot(f, 'HEOS')
#    except:
#        print(f)
#test_PV_plot('water', 'HEOS')

# CoolProp does not pass
del test_PV_plot


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("backend", backends)
def test_TV_plot_CoolProp(fluid, backend):
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)

    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    gas = CoolPropGas(backend, fluid, T=T, P=P, zs=zs)
    liquid = CoolPropLiquid(backend, fluid, T=T, P=P, zs=zs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

#    CoolPropInfo = coolprop_fluids[pure_const.CASs[0]]
    T_min, T_max, P_max = gas.AS.Tmin(), gas.AS.Tmax(), gas.AS.pmax()
    P_min = PropsSI(fluid, 'PMIN')
    V_max = flasher.flash(T=T_max, P=P_min).V()
    V_min = flasher.flash(T=T_min, P=P_max).V()

    res = flasher.TPV_inputs(zs=zs, pts=200, spec0='T', spec1='P', check0='T', check1='V', prop0='P',
                           trunc_err_low=1e-15,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           Tmax=T_max*(1.0 - 1e-6), Tmin=T_min*(1.0+4e-2),
                           Vmin=V_min*(1.0), Vmax=V_max*(1.0 - 1e-5),
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, "TV")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s' %('TV', backend, fluid)

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()
#test_TV_plot_CoolProp('water', 'HEOS')

@pytest.mark.fuzz
@pytest.mark.slow
@pytest.mark.plot
def test_water_95():
    # TVF, PVF, TP, TV, PV are covered and optimized
    T, P = 298.15, 1e5
    fluid = 'water'
    zs = [1.0]
    # fluid = 'toluene'
    m = Mixture([fluid], T=T, P=P, zs=zs)
    CPP_gas = CoolPropGas('HEOS', fluid, T=T, P=P, zs=zs)
    CPP_liq = CoolPropLiquid('HEOS', fluid, T=T, P=P, zs=zs)
    constants = m.constants
    correlations = m.properties()
    flasher = FlashPureVLS(constants, correlations, CPP_gas, [CPP_liq], solids=[])

    constants = ChemicalConstantsPackage(Tcs=[647.14], Pcs=[22048320.0], omegas=[0.344], MWs=[18.01528], CASs=['7732-18-5'])
    correlations = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))], )

    # TP states
    pts = 15
    T_min = PropsSI('TMIN', fluid)
    T_max = PropsSI('TMAX', fluid)
    P_min = PropsSI('PMIN', fluid)
    P_max = PropsSI('PMAX', fluid)
    Ts = linspace(T_min*(1+1e-7), T_max*(1.0 - 1e-7), pts)
    Ps = logspace(log10(PropsSI('PMIN', fluid)*(1.0 + 1e-5)), log10(P_max*(1.0 - 1e-5)), pts)

    Tmin_PV = T_min + 1.0
    for T in Ts:
        for P in Ps:
            res = flasher.flash(T=T, P=P)
            H_base = PropsSI('HMOLAR', 'T', T, 'P', P, fluid)
            assert_close(res.H(), H_base, rtol=1e-12)

            # TV flash
            res_TV = flasher.flash(T=T, V=res.V())
            assert_close(res_TV.H(), H_base, rtol=1e-6) # Some error in here 2e-7 worked

            # PV flash - some failures near triple point
            if T > Tmin_PV:
                res_PV = flasher.flash(P=P, V=res.V())
                assert_close(res_PV.T, T, rtol=1e-7)

    Ps_sat = logspace(log10(PropsSI('PMIN', fluid)*(1.0 + 1e-5)), log10(PropsSI('PCRIT', fluid)*(1.0 - 1e-5)), pts)
    for P in Ps_sat:
        res = flasher.flash(P=P, VF=0.5)
        T_base = PropsSI('T', 'Q', 0.5, 'P', P, fluid)
        assert_close(T_base, res.T, rtol=1e-12)

    Ts_sat = linspace(PropsSI('TMIN', fluid)*(1+1e-7), PropsSI('TCRIT', fluid)*(1.0 - 1e-7), pts)
    for T in Ts_sat:
        res = flasher.flash(T=T, VF=0.5)
        P_base = PropsSI('P', 'Q', 0.5, 'T', T, fluid)
        assert_close(P_base, res.P, rtol=1e-12)

    # Test some hard points
    # Critical point
    Tc, Pc = PropsSI('TCRIT', fluid), PropsSI('PCRIT', fluid)
    H_base = PropsSI('HMOLAR', 'T', Tc, 'P', Pc, fluid)
    H_flashed = flasher.flash(T=Tc, P=Pc).H()
    assert_close(H_base, H_flashed)

    # Minimum temperature and maximum temperature
    # Only had an issue with P min
    Ps = [P_min*(1.0 + 1e-5), P_min*2.0, P_min*10.0, P_min*50.0, P_max*(1.0 - 1e-9)]
    for P in Ps:
        H_base = PropsSI('HMOLAR', 'T', T_min, 'P', P, fluid)
        H_flashed = flasher.flash(T=T_min, P=P).H()
        assert_close(H_base, H_flashed)

        H_base = PropsSI('HMOLAR', 'T', T_max, 'P', P, fluid)
        H_flashed = flasher.flash(T=T_max, P=P).H()
        assert_close(H_base, H_flashed)
