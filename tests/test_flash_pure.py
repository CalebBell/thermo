# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.f

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
from fluids.numerics import assert_close, assert_close1d, assert_close2d
from fluids.numerics import *
from math import *
import json
import os
import numpy as np
from thermo.test_utils import *
from chemicals.exceptions import PhaseExistenceImpossible
import sys
from thermo.test_utils import plot_unsupported
from thermo import eos_volume
try:
    import matplotlib.pyplot as plt
except:
    pass

PY2 = sys.version[0] == '2'
pure_surfaces_dir = os.path.join(thermo.thermo_dir, '..', 'surfaces', 'pure')

pure_fluids = ['water', 'methane', 'ethane', 'decane', 'ammonia', 'nitrogen', 'oxygen', 'methanol', 'eicosane', 'hydrogen']

'''# Recreate the below with the following:
N = len(pure_fluids)
m = Mixture(pure_fluids, zs=[1/N]*N, T=298.15, P=1e5)
print(m.constants.make_str(delim=', \n', properties=('Tcs', 'Pcs', 'omegas', 'MWs', "CASs")))
correlations = m.properties()
print(correlations.as_poly_fit(['HeatCapacityGases']))
'''
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

HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.444966286051841e-23, 9.444106746563928e-20, -1.2490299714587002e-15, 2.6693560979905865e-12, -2.5695131746723413e-09, 1.2022442523089315e-06, -0.00021492132731007108, 0.016616385291696574, 32.84274656062226])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924])),
                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.075118433508619e-20, 1.0383055980949049e-16, -2.1577805903757125e-13, 2.373511052680461e-10, -1.4332562489496906e-07, 4.181755403465859e-05, -0.0022544761674344544, -0.15965342941876415, 303.71771182550816])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [1.1878323802695824e-20, -5.701277266842367e-17, 1.1513022068830274e-13, -1.270076105261405e-10, 8.309937583537026e-08, -3.2694889968431594e-05, 0.007443050245274358, -0.8722920255910297, 66.82863369121873]))]

VaporPressures = [VaporPressure(poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317])),
                  VaporPressure(poly_fit=(90.8, 190.554, [1.2367137894255505e-16, -1.1665115522755316e-13, 4.4703690477414014e-11, -8.405199647262538e-09, 5.966277509881474e-07, 5.895879890001534e-05, -0.016577129223752325, 1.502408290283573, -42.86926854012409])),
                  VaporPressure(poly_fit=(90.4, 305.312, [-1.1908381885079786e-17, 2.1355746620587145e-14, -1.66363909858873e-11, 7.380706042464946e-09, -2.052789573477409e-06, 0.00037073086909253047, -0.04336716238170919, 3.1418840094903784, -102.75040650505277])),
                  VaporPressure(poly_fit=(243.51, 617.69, [-1.9653193622863184e-20, 8.32071200890499e-17, -1.5159284607404818e-13, 1.5658305222329732e-10, -1.0129531274368712e-07, 4.2609908802380584e-05, -0.01163326014833186, 1.962044867057741, -153.15601192906817])),
                  VaporPressure(poly_fit=(195.505, 405.39, [1.8775319752114198e-19, -3.2834459725160406e-16, 1.9723813042226462e-13, -1.3646182471796847e-11, -4.348131713052942e-08, 2.592796525478491e-05, -0.007322263143033041, 1.1431876410642319, -69.06950797691312])),
                  VaporPressure(poly_fit=(63.2, 126.18199999999999, [5.490876411024536e-15, -3.709517805130509e-12, 1.0593254238679989e-09, -1.6344291780087318e-07, 1.4129990091975526e-05, -0.0005776268289835264, -0.004489180523814208, 1.511854256824242, -36.95425216567675])),
                  VaporPressure(poly_fit=(54.370999999999995, 154.57100000000003, [-9.865296960381724e-16, 9.716055729011619e-13, -4.163287834047883e-10, 1.0193358930366495e-07, -1.57202974507404e-05, 0.0015832482627752501, -0.10389607830776562, 4.24779829961549, -74.89465804494587])),
                  VaporPressure(poly_fit=(175.7, 512.49, [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10, -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708])),
                  VaporPressure(poly_fit=(309.58, 768.0, [-7.623520807550015e-21, 3.692055956856486e-17, -7.843475114100242e-14, 9.586109955308152e-11, -7.420461759412686e-08, 3.766437656499104e-05, -0.012473026309766598, 2.5508091747355173, -247.8254681597346])),
                  VaporPressure(poly_fit=(13.967, 33.135000000000005, [2.447649824484286e-11, -1.9092317903068513e-09, -9.871694369510465e-08, 1.8738775057921993e-05, -0.0010528586872742638, 0.032004931729483245, -0.5840683278086365, 6.461815322744102, -23.604507906951046])),]

VolumeLiquids = [VolumeLiquid(poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652])),
                 VolumeLiquid(poly_fit=(90.8, 180.564, [7.730541828225242e-20, -7.911042356530585e-17, 3.51935763791471e-14, -8.885734012624568e-12, 1.3922694980104743e-09, -1.3860056394382538e-07, 8.560110533953199e-06, -0.00029978743425740123, 0.004589555868318768])),
                 VolumeLiquid(poly_fit=(90.4, 295.322, [8.94814934875547e-22, -1.2714020587822148e-18, 7.744406810864275e-16, -2.6371441222674234e-13, 5.484329710203708e-11, -7.125679072517415e-09, 5.646423685845822e-07, -2.487300046509999e-05, 0.0005100376035218567])),
                 VolumeLiquid(poly_fit=(243.51, 607.7, [1.0056823442253386e-22, -3.2166293088353376e-19, 4.442027873447809e-16, -3.4574825216883073e-13, 1.6583965814129937e-10, -5.018203505211133e-08, 9.353680499788552e-06, -0.0009817356348626736, 0.04459313654596568])),
                 VolumeLiquid(poly_fit=(195.505, 395.4, [5.103835649289192e-22, -1.1451165900302792e-18, 1.1154748069727923e-15, -6.160223429581022e-13, 2.1091176437184486e-10, -4.583765753427699e-08, 6.175086699295002e-06, -0.00047144103245914243, 0.015638233047208582])),
                 VolumeLiquid(poly_fit=(63.2, 116.192, [9.50261462694019e-19, -6.351064785670885e-16, 1.8491415360234833e-13, -3.061531642102745e-11, 3.151588109585604e-09, -2.0650965261816766e-07, 8.411110954342014e-06, -0.00019458305886755787, 0.0019857193167955463])),
                 VolumeLiquid(poly_fit=(54.370999999999995, 144.58100000000002, [6.457909929992152e-20, -4.7825644162085234e-17, 1.5319533644419177e-14, -2.7692511820542383e-12, 3.088256295705142e-10, -2.1749171236451626e-08, 9.448300475893009e-07, -2.3081894336450133e-05, 0.00026558114294435354])),
                 VolumeLiquid(poly_fit=(175.7, 502.5, [3.5725079384600736e-23, -9.031033742820083e-20, 9.819637959370411e-17, -5.993173551565636e-14, 2.2442465416964825e-11, -5.27776114586072e-09, 7.610461006178106e-07, -6.148574498547711e-05, 0.00216398089328537])),
                 VolumeLiquid(poly_fit=(309.58, 729.5999999999999, [7.45473064887452e-24, -2.89457102830013e-20, 4.867041417017972e-17, -4.6252122183399004e-14, 2.7157887108452537e-11, -1.0085443480134824e-08, 2.3130153268044497e-06, -0.0002992756488164552, 0.01705648133237398])),
                 VolumeLiquid(poly_fit=(13.967, 29.3074, [1.338998655322118e-14, -2.2300738749278554e-12, 1.6125645123435388e-10, -6.603182508985557e-09, 1.6732222054898376e-07, -2.6846339878160216e-06, 2.6629092007736217e-05, -0.0001490816582989168, 0.0003852732680036591]))]

correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True, VolumeLiquids=VolumeLiquids, VaporPressures=VaporPressures, HeatCapacityGases=HeatCapacityGases)

from thermo.eos_mix import eos_mix_list






@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("auto_range", ['realistic', 'physical'])
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_PV_plot(fluid, eos, auto_range):
    '''
    Normally about 16% of the realistic plot overlaps with the physical. However,
    the realistic is the important one, so do not use fewer points for it.

    The realistic should be clean/clear!
    '''
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)

    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    '''
    m = Mixture([fluid], zs=zs, T=T, P=P)
    pure_const = m.constants
    HeatCapacityGases = m.HeatCapacityGases
    pure_props = PropertyCorrelationsPackage(pure_const, HeatCapacityGases=HeatCapacityGases)
    '''
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])


    res = flasher.TPV_inputs(zs=zs, pts=100, spec0='T', spec1='P', check0='P', check1='V', prop0='T',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range=auto_range,
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, "PV")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s - %s' %('PV', eos.__name__, auto_range, fluid)

    plot_fig.savefig(os.path.join(path, key + '.png'))
    # TODO log the max error to a file

    plt.close()
    max_err = np.max(np.abs(errs))
    limit = 5e-8
    if eos is RKMIX:
        limit = 1e-6 # Need to udpate with a few numerical polish iterations
    try:
        assert max_err < limit
    except:
        print(fluid, eos, auto_range)
        assert max_err < limit
#for e in eos_mix_list:
#    e = TWUSRKMIX
#    print(e)
#    test_PV_plot('hydrogen', e, 'physical')
#test_PV_plot('decane', PRMIXTranslatedConsistent, 'physical')
#test_PV_plot('methanol', SRKMIXTranslatedConsistent, 'physical')

@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("auto_range", ['physical', 'realistic'])
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_TV_plot(fluid, eos, auto_range):
    '''
    A pretty wide region here uses mpmath to polish the volume root,
    and calculate the pressure from the TV specs. This is very important! For
    the liquid region, there is not enough pressure dependence of volume for
    good calculations to work.
    '''
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])


    res = flasher.TPV_inputs(zs=zs, pts=50, spec0='T', spec1='P', check0='T', check1='V', prop0='P',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range=auto_range,
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, "TV")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s - %s' %('TV', eos.__name__, auto_range, fluid)
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(np.abs(errs))

    try:
        assert max_err < 5e-9
    except:
        assert max_err < 5e-9

#test_TV_plot('decane', PRMIXTranslatedConsistent, 'physical')
#test_TV_plot('eicosane', PRMIXTranslatedConsistent, 'physical')

#for e in eos_mix_list:
#    e = TWUPRMIX
#    print(e)
#    test_TV_plot('hydrogen', e, 'realistic')

@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("auto_range", ['physical', 'realistic'])
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_PS_plot(fluid, eos, auto_range):
    '''
    '''
    path = os.path.join(pure_surfaces_dir, fluid, "PS")
    if not os.path.exists(path):
        os.makedirs(path)
    key = '%s - %s - %s - %s' %('PS', eos.__name__, auto_range, fluid)


    if eos in (TWUPRMIX, TWUSRKMIX):
        msg = None
        if fluid in ('hydrogen', 'eicosane', 'decane', 'water'):
            msg = 'Garbage alpha function multiple solutions'
        elif auto_range == 'physical':
            msg = 'Garbage alpha function low T'
        if msg is not None:
            plot_fig = plot_unsupported(msg, color='g')
            plot_fig.savefig(os.path.join(path, key + '.png'), bbox_inches='tight')
            plt.close()
            return

    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    res = flasher.TPV_inputs(zs=zs, pts=60, spec0='T', spec1='P', check0='P', check1='S', prop0='T',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range=auto_range,
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(errs)
    assert max_err < 1e-8
#test_PS_plot("hydrogen", TWUPRMIX, "physical")
#test_PS_plot("hydrogen", TWUSRKMIX, "physical")
#test_PS_plot("hydrogen", TWUPRMIX, "realistic")
#test_PS_plot("hydrogen", TWUSRKMIX, "realistic")

@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("auto_range", ['physical', 'realistic'])
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_PH_plot(fluid, eos, auto_range):
    '''
    '''
    path = os.path.join(pure_surfaces_dir, fluid, "PH")
    if not os.path.exists(path):
        os.makedirs(path)
    key = '%s - %s - %s - %s' %('PH', eos.__name__, auto_range, fluid)

    if eos in (TWUPRMIX, TWUSRKMIX):
        msg = None
        if fluid in ('hydrogen', 'eicosane', 'decane', 'water'):
            msg = 'Garbage alpha function multiple solutions'
        elif auto_range == 'physical':
            msg = 'Garbage alpha function low T'
        if msg is not None:
            plot_fig = plot_unsupported(msg, color='g')
            plot_fig.savefig(os.path.join(path, key + '.png'), bbox_inches='tight')
            plt.close()
            return
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    res = flasher.TPV_inputs(zs=zs, pts=50, spec0='T', spec1='P', check0='P', check1='H', prop0='T',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range=auto_range,
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(errs)
    assert max_err < 1e-8
#test_PH_plot('eicosane', PRMIX, 'physical')
#test_PH_plot("hydrogen", TWUPRMIX, "physical")
#test_PH_plot("hydrogen", TWUSRKMIX, "physical")
#test_PH_plot("hydrogen", TWUPRMIX, "realistic")
#test_PH_plot("hydrogen", TWUSRKMIX, "realistic")


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("auto_range", ['physical', 'realistic'])
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_PU_plot(fluid, eos, auto_range):
    '''Does not seem unique, again :(
    Going to have to add new test functionality that does there tests against a
    reflash at PT.
    '''
    path = os.path.join(pure_surfaces_dir, fluid, "PU")
    if not os.path.exists(path):
        os.makedirs(path)
    key = '%s - %s - %s - %s' %('PU', eos.__name__, auto_range, fluid)

    if eos in (TWUPRMIX, TWUSRKMIX):
        msg = None
        if fluid in ('hydrogen', 'eicosane', 'decane', 'water'):
            msg = 'Garbage alpha function multiple solutions'
        elif auto_range == 'physical':
            msg = 'Garbage alpha function low T'
        if msg is not None:
            plot_fig = plot_unsupported(msg, color='g')
            plot_fig.savefig(os.path.join(path, key + '.png'), bbox_inches='tight')
            plt.close()
            return
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    res = flasher.TPV_inputs(zs=zs, pts=50, spec0='T', spec1='P', check0='P', check1='U', prop0='T',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range=auto_range,
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(errs)
    if eos != VDWMIX:
        # Do not know what is going on there
        # test case vdw decane failing only
#        base =  flasher.flash(T=372.75937203149226, P=255954.79226995228)
#        flasher.flash(P=base.P, U=base.U()).T
        assert max_err < 1e-8

#test_PU_plot("hydrogen", TWUPRMIX, "physical")
#test_PU_plot("hydrogen", TWUSRKMIX, "physical")
#test_PU_plot("hydrogen", TWUPRMIX, "realistic")
#test_PU_plot("hydrogen", TWUSRKMIX, "realistic")

@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("auto_range", ['physical', 'realistic'])
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_VU_plot(fluid, eos, auto_range):
    if eos in (TWUPRMIX, TWUSRKMIX, RKMIX) and auto_range == 'physical':
#         Garbage alpha function for very low T
        return
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    res = flasher.TPV_inputs(zs=zs, pts=50, spec0='T', spec1='P', check0='V', check1='U', prop0='T',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range=auto_range,
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, "VU")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s - %s' %('VU', eos.__name__, auto_range, fluid)
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(errs)
    assert max_err < 1e-7

#for fluid in pure_fluids:
#    print(fluid)
#    test_VU_plot(fluid, PRMIXTranslatedConsistent, 'realistic')
#for e in eos_mix_list:
#    print(e)
#    test_VU_plot('hydrogen', e, 'realistic')
#test_VU_plot('methanol', PRMIXTranslatedConsistent, 'physical')

@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("auto_range", ['physical', 'realistic'])
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_VS_plot(fluid, eos, auto_range):
    '''Some SRK tests are failing because of out-of-bounds issues.
    Hard to know how to fix these.

    RKMIX fails because a_alpha gets to be ~10000 and all the entropy is excess.
    '''

    if eos in (TWUPRMIX, TWUSRKMIX, RKMIX) and auto_range == 'physical':
        return
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    res = flasher.TPV_inputs(zs=zs, pts=50, spec0='T', spec1='P', check0='V', check1='S', prop0='T',
                           trunc_err_low=1e-10, retry=True,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range=auto_range,
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, "VS")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s - %s' %('VS', eos.__name__, auto_range, fluid)
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(errs)
    assert max_err < 1e-8

#test_VS_plot('methanol', PRMIXTranslatedConsistent, 'physical')
#for fluid in pure_fluids:
#    test_VS_plot(fluid, APISRKMIX, 'physical')
#for e in eos_mix_list:
#    print(e)
#    test_VS_plot('eicosane', e, 'physical')
#test_VS_plot('eicosane', RKMIX, 'physical')


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("auto_range", ['physical', 'realistic'])
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_VH_plot(fluid, eos, auto_range):
    if eos in (TWUPRMIX, TWUSRKMIX, RKMIX) and auto_range == 'physical':
#         Garbage alpha function for very low T in all three
        return
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    res = flasher.TPV_inputs(zs=zs, pts=50, spec0='T', spec1='P', check0='V', check1='H', prop0='T',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range=auto_range,
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, "VH")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s - %s' %('VH', eos.__name__, auto_range, fluid)
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(errs)
    assert max_err < 1e-7

#for e in eos_mix_list:
#    print(e)
#    test_VH_plot('eicosane', e, 'physical')


@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.skip
@pytest.mark.parametric
@pytest.mark.parametrize("auto_range", ['physical', 'realistic'])
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_TS_plot(fluid, eos, auto_range):
    '''
    '''
    #if eos in (TWUPRMIX, TWUSRKMIX) and auto_range == 'physical':
        # Garbage alpha function for very low T
    #    return
    path = os.path.join(pure_surfaces_dir, fluid, "TS")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s - %s' %('TS', eos.__name__, auto_range, fluid)

    if eos in (IGMIX,):
        plot_fig = plot_unsupported('Ideal gas has no pressure dependence of entropy', color='g')
        plot_fig.savefig(os.path.join(path, key + '.png'), bbox_inches='tight')
        plt.close()
        return

    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    res = flasher.TPV_inputs(zs=zs, pts=50, spec0='T', spec1='P', check0='T', check1='S', prop0='P',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range=auto_range,
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(errs)
    assert max_err < 1e-8

#for e in eos_mix_list:
#    # Real frustrating - just cannot get enough precision; need mpmath on S_dep as well as S.
#    print(e)
#    test_TS_plot('eicosane', e, 'realistic')



@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.skip
@pytest.mark.parametric
@pytest.mark.parametrize("auto_range", ['physical', 'realistic'])
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_TH_plot(fluid, eos, auto_range):
    '''Solutions are NOT UNIQUE
    '''
    #if eos in (TWUPRMIX, TWUSRKMIX) and auto_range == 'physical':
        # Garbage alpha function for very low T
    #    return
    path = os.path.join(pure_surfaces_dir, fluid, "TH")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s - %s' %('TH', eos.__name__, auto_range, fluid)
    if eos in (IGMIX,):
        plot_fig = plot_unsupported('Ideal gas has no pressure dependence of enthalpy', color='g')
        plot_fig.savefig(os.path.join(path, key + '.png'), bbox_inches='tight')
        plt.close()
        return

    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    res = flasher.TPV_inputs(zs=zs, pts=50, spec0='T', spec1='P', check0='T', check1='H', prop0='P',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range=auto_range,
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(errs)
    assert max_err < 1e-8

#for e in eos_mix_list:
#    print(e)
#    try:
#        test_TH_plot('eicosane', e, 'realistic')
#    except:
#        pass


@pytest.mark.fuzz
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_TP_VF_points_with_HSU_VF0(fluid, eos):
    '''
    '''
    if eos in (IGMIX,):
        return
    # print(eos)
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    pts = 50
    # Might be good to make the solvers go to lower pressure
    Ts = linspace(pure_const.Tcs[0]*0.5, pure_const.Tcs[0], pts)
    Ps = logspace(log10(10), log10(pure_const.Pcs[0]*(1.0-1e-14)), pts)
    VF = 0
    rtol = 1e-8
    for s in ('H', 'S', 'U'): # G, A not unique
        for T in Ts:
            p = flasher.flash(T=T, VF=VF)
            val = getattr(p, s)()
            resolve = flasher.flash(VF=VF, **{s: val})
            assert_close(p.T, resolve.T, rtol=rtol)
            assert_close(p.P, resolve.P, rtol=rtol)
        
        for P in Ps:
            p = flasher.flash(P=P, VF=VF)
            val = getattr(p, s)()
            resolve = flasher.flash(VF=VF, **{s: val})
            assert_close(p.T, resolve.T, rtol=rtol)
            assert_close(p.P, resolve.P, rtol=rtol)

# test_TP_VF_points_with_HSU_VF0('methane', PRMIX)


### Pure EOS only tests
@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametric
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_list)
def test_V_G_min_plot(fluid, eos):
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

#    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
#                  HeatCapacityGases=pure_props.HeatCapacityGases)
#    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    kwargs = dict(Tc=pure_const.Tcs[0], Pc=pure_const.Pcs[0], omega=pure_const.omegas[0])

    gas = eos(T=T, P=P, **kwargs)
    plot_fig = gas.PT_surface_special(show=False, pts=150,
                                       Tmin=1e-4, Tmax=1e4, Pmin=1e-2, Pmax=1e9,
                                       mechanical=False, pseudo_critical=False, Psat=False,
                                       determinant_zeros=False)


    path = os.path.join(pure_surfaces_dir, fluid, "V_G_min")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s' %('V_G_min', eos.__name__, fluid)

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    # Not sure how to add error to this one
#test_V_G_min_plot('ammonia', MSRKTranslated)
#test_V_G_min_plot('hydrogen', TWUSRK)


@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametric
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_list)
def test_a_alpha_plot(fluid, eos):
    path = os.path.join(pure_surfaces_dir, fluid, "a_alpha")
    if not os.path.exists(path):
        os.makedirs(path)
    key = '%s - %s - %s' %('a_alpha', eos.__name__, fluid)

    if eos in (IG,):
        plot_fig = plot_unsupported('Ideal gas has a_alpha of zero', color='g')
        plot_fig.savefig(os.path.join(path, key + '.png'), bbox_inches='tight')
        plt.close()
        return
    T, P = 298.15, 101325.0
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(Tc=pure_const.Tcs[0], Pc=pure_const.Pcs[0], omega=pure_const.omegas[0])

    obj = eos(T=T, P=P, **kwargs)
    _, _, _, _, plot_fig = obj.a_alpha_plot(Tmin=1e-4, Tmax=pure_const.Tcs[0]*30, pts=500,
                                          plot=True, show=False)

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()
    # Not sure how to add error to this one



@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_list)
@pytest.mark.parametric
def test_Psat_plot(fluid, eos):
    path = os.path.join(pure_surfaces_dir, fluid, "Psat")
    if not os.path.exists(path):
        os.makedirs(path)

    key = '%s - %s - %s' %('Psat', eos.__name__, fluid)

    if eos in (IG,):
        plot_fig = plot_unsupported('Ideal gas cannot have a liquid phase', color='g')
        plot_fig.savefig(os.path.join(path, key + '.png'), bbox_inches='tight')
        plt.close()
        return
    if eos in (TWUPR, TWUSRK):
        if fluid in ('hydrogen', ):
            msg = 'Garbage alpha function low T'
            plot_fig = plot_unsupported(msg, color='g')
            plot_fig.savefig(os.path.join(path, key + '.png'), bbox_inches='tight')
            plt.close()
            return

    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(Tc=pure_const.Tcs[0], Pc=pure_const.Pcs[0], omega=pure_const.omegas[0])


    obj = eos(T=T, P=P, **kwargs)

    Tmin = kwargs['Tc']*.03
    if eos is RK:
        Tmin = kwargs['Tc']*.2
    errs, Psats_num, Psats_fit, plot_fig = obj.Psat_errors(plot=True, show=False, pts=100,
                                     Tmin=Tmin, Tmax=kwargs['Tc'], Pmin=1e-100)



    plot_fig.savefig(os.path.join(path, key + '.png'), bbox_inches='tight')
    plt.close()

    # TODO reenable
    max_err = np.max(errs)
    assert max_err < 1e-10
    # HYDROGEN twu broken
#test_Psat_plot('hydrogen', IG)
#test_Psat_plot('eicosane', IG)
#test_Psat_plot('eicosane', PRTranslatedPPJP)





@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametric
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_list)
@pytest.mark.parametrize("P_range", ['high', 'low'])
@pytest.mark.parametrize("solver", [GCEOS.volume_solutions])
def test_V_error_plot(fluid, eos, P_range, solver):
    path = os.path.join(pure_surfaces_dir, fluid, "V_error")
    if not os.path.exists(path):
        os.makedirs(path)
    key = '%s - %s - %s - %s' %('V_error', eos.__name__, fluid, P_range)

    if eos in (IG,):
        plot_fig = plot_unsupported('Ideal gas has only one volume solution', color='g')
        plot_fig.savefig(os.path.join(path, key + '.png'), bbox_inches='tight')
        plt.close()
        return
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

#    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
#                  HeatCapacityGases=pure_props.HeatCapacityGases)
    kwargs = dict(Tc=pure_const.Tcs[0], Pc=pure_const.Pcs[0], omega=pure_const.omegas[0])


    # Caching layer
#    orig_func = GCEOS.volume_solutions_mpmath

    mem_cache = {}

    did_new_dat = [False]

#    @staticmethod
#    def cache_persistent(*args):
#        if args in mem_cache:
#            return mem_cache[args]
#        ans = orig_func(*args)
#        mem_cache[args] = ans
#        did_new_dat[0] = True
#        return ans
#
#    if not PY2:
#        GCEOS.volume_solutions_mpmath = cache_persistent
#    else:
#        setattr(GCEOS, 'volume_solutions_mpmath', cache_persistent)
    # Cannot use json - key is a tuple which json does not support.
    import pickle
    try:
        open_file = open(os.path.join(path, key + '.dat'), 'rb')
        mem_cache = pickle.load(open_file)
        open_file.close()
    except:
        pass

    if P_range == 'high':
        Pmin = 1e-2
        Pmax = 1e9
    elif P_range == 'low':
        Pmax = 1e-2
        Pmin = 1e-60

    class VolumeWrapper(eos):
        @staticmethod
        def volume_solutions_mp(*args):
            if args in mem_cache:
                return mem_cache[args]
            Vs = eos_volume.volume_solutions_mpmath(*args)
            mem_cache[args] = Vs
            did_new_dat[0] = True
            return [float(Vi.real) + float(Vi.imag)*1.0j for Vi in Vs]


    obj = VolumeWrapper(T=T, P=P, **kwargs)
    VolumeWrapper.volume_solutions = solver
    errs, plot_fig = obj.volume_errors(plot=True, show=False, pts=100,
                                       Tmin=1e-4, Tmax=1e4, Pmin=Pmin, Pmax=Pmax,
                                       trunc_err_low=1e-15, color_map=cm_flash_tol())

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    if did_new_dat[0]:
        open_file = open(os.path.join(path, key + '.dat'), 'wb')
        pickle.dump(mem_cache, open_file, protocol=2)
        open_file.close()


    max_err = np.max(errs)
    assert max_err < 1e-13



#test_V_error_plot('ethane', SRK, 'low')
#test_V_error_plot('hydrogen', PR, 'low')
#test_V_error_plot('decane', PR, 'low')
# test_V_error_plot('ethane', SRK, 'high')
#test_V_error_plot('methanol', PR, 'high')
#test_V_error_plot('methanol', PR, 'low')
#test_V_error_plot('hydrogen', SRK, 'high')
#test_V_error_plot('hydrogen', TWUSRKMIX, 'high')
#test_V_error_plot('hydrogen', IGMIX, 'low')
#test_V_error_plot('methane', SRKTranslatedConsistent, 'high')
#test_V_error_plot('methane', APISRK, 'low')
#test_V_error_plot('ethane', VDW, 'low')

#test_V_error_plot('nitrogen', PRTranslatedConsistent, 'high')




#### Ideal property package - dead out of the gate, second T derivative of V is zero - breaks requirements
@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("fluid", pure_fluids)
def test_P_H_plot_ideal_Poy(fluid):
    '''
    '''
    if fluid == 'water':
        return # Causes multiple solutions
    path = os.path.join(pure_surfaces_dir, fluid, "PH")
    if not os.path.exists(path):
        os.makedirs(path)
    key = '%s - %s - %s - %s' %('PH', "idealPoynting", "physical", fluid)

    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    ig_kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)
#
    liquid = GibbsExcessLiquid(VaporPressures=pure_props.VaporPressures,
                           HeatCapacityGases=pure_props.HeatCapacityGases,
                           VolumeLiquids=pure_props.VolumeLiquids,
                           use_phis_sat=False, use_Poynting=True).to_TP_zs(T, P, zs)

    gas = CEOSGas(IGMIX, T=T, P=P, zs=zs, **ig_kwargs)
#
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
    flasher.VL_IG_activity = True
#
    res = flasher.TPV_inputs(zs=zs, pts=50, spec0='T', spec1='P', check0='P', check1='H', prop0='T',
                           trunc_err_low=1e-10,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range='physical',
                           show=False)
#
    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res
#
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()
#
    max_err = np.max(errs)
    assert max_err < 1e-8
#test_PH_plot_ideal_Poy('methanol')


### Non-generic tests

@pytest.mark.parametrize("hacks", [True, False])
def test_some_flashes_bad(hacks):
    '''Basic test with ammonia showing how PG, PA, TG, TA, VA, VG flashes
    have multiple solutions quite close.
    '''
    constants = ChemicalConstantsPackage(Tcs=[405.6], Pcs=[11277472.5], omegas=[0.25], MWs=[17.03052], CASs=['7664-41-7'])
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.444966286051841e-23, 9.444106746563928e-20,
                            -1.2490299714587002e-15, 2.6693560979905865e-12, -2.5695131746723413e-09, 1.2022442523089315e-06,
                            -0.00021492132731007108, 0.016616385291696574, 32.84274656062226]))]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)

    liquid = CEOSLiquid(SRKMIX, T=330, P=1e5, zs=[1], **kwargs)
    gas = CEOSGas(SRKMIX, T=330, P=1e5, zs=[1], **kwargs)

    flasher = FlashPureVLS(constants, correlations, gas, [liquid], [])
    flasher.VL_only_CEOSs_same = hacks

    assert_close(flasher.flash(T=800, P=1e7).G(), flasher.flash(T=725.87092453, P=1e7).G(), rtol=1e-10)

def test_EOS_dew_bubble_same_eos_id():
    constants = ChemicalConstantsPackage(Tcs=[405.6], Pcs=[11277472.5], omegas=[0.25], MWs=[17.03052], CASs=['7664-41-7'])
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.444966286051841e-23, 9.444106746563928e-20,
                            -1.2490299714587002e-15, 2.6693560979905865e-12, -2.5695131746723413e-09, 1.2022442523089315e-06,
                            -0.00021492132731007108, 0.016616385291696574, 32.84274656062226]))]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)
    liquid = CEOSLiquid(SRKMIX, T=330, P=1e5, zs=[1], **kwargs)
    gas = CEOSGas(SRKMIX, T=330, P=1e5, zs=[1], **kwargs)
    flasher = FlashPureVLS(constants, correlations, gas, [liquid], [])

    # Test which covers the ID of returned dew/bubble flash phases are the same
    # for the EOS case
    res = flasher.flash(P=1e5, VF=.5)
    assert res.gas.eos_mix is res.liquid0.eos_mix

    res = flasher.flash(T=300, VF=.5)
    assert res.gas.eos_mix is res.liquid0.eos_mix

@pytest.mark.parametrize("hacks", [True, False])
def test_VS_issue_PRSV(hacks):
    constants = ChemicalConstantsPackage(Tcs=[768.0], Pcs=[1070000.0], omegas=[0.8805], MWs=[282.54748], CASs=['112-95-8'])
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.075118433508619e-20, 1.0383055980949049e-16, -2.1577805903757125e-13, 2.373511052680461e-10, -1.4332562489496906e-07, 4.181755403465859e-05, -0.0022544761674344544, -0.15965342941876415, 303.71771182550816]))]

    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                    HeatCapacityGases=HeatCapacityGases)

    liquid = CEOSLiquid(PRSVMIX, T=330, P=1e5, zs=[1], **kwargs)
    gas = CEOSGas(PRSVMIX, T=330, P=1e5, zs=[1], **kwargs)

    flasher = FlashPureVLS(constants, correlations, gas, [liquid], [])
    flasher.VL_only_CEOSs_same = hacks
    obj = flasher.flash(T=7196.856730011477, P=212095088.7920158)
    assert_close(obj.T, flasher.flash(V=obj.V(), S=obj.S()).T)

@pytest.mark.parametrize("hacks", [True, False])
def test_PS_1P_vs_VL_issue0(hacks):
    '''Made me think there was something wrong with enthalpy maximization.
    However, it was just a root issue.
    '''

    constants = ChemicalConstantsPackage(Tcs=[647.14], Pcs=[22048320.0], omegas=[0.344], MWs=[18.01528],  CASs=['7732-18-5'],)
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)

    liquid = CEOSLiquid(PR78MIX, T=200, P=1e5, zs=[1], **kwargs)
    gas = CEOSGas(PR78MIX, T=200, P=1e5, zs=[1], **kwargs)
    flasher = FlashPureVLS(constants, correlations, gas, [liquid], []) #
    flasher.VL_only_CEOSs_same = hacks

    for P in (0.01, 0.015361749466718281):
        obj = flasher.flash(T=166.0882782627715, P=P)
        hit = flasher.flash(P=obj.P, S=obj.S())
        assert_close(hit.T, obj.T)


@pytest.mark.parametrize("hacks", [True, False])
def test_HSGUA_early_return_eos(hacks):
    '''Need to check metastable
    '''
    T, P, zs = 517.9474679231187, 91029.8177991519, [1.0]
    fluid_idx, eos = 8, PRMIX # eicosane
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
    flasher.VL_only_CEOSs_same = hacks

    res0 = flasher.flash(T=517.9474679231187, P=P)
    res1 = flasher.flash(P=P, H=res0.H())
    assert_close(res1.G(), res0.G(), rtol=1e-7)


@pytest.mark.parametrize("hacks", [True, False])
def test_SRK_high_P_PV_failure(hacks):
    '''Numerical solve_T converging to a T in the range 200,000 K when there
    was a lower T solution
    '''
    T, P, zs = 2000, 1e8, [1.0]
    fluid_idx, eos = 7, SRKMIX # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
    flasher.VL_only_CEOSs_same = hacks

    base = flasher.flash(T=T, P=P)

    PV = flasher.flash(P=P, V=base.V(), solution='low')
    assert_close(T, PV.T, rtol=1e-7)

    PV = flasher.flash(P=P, V=base.V(), solution='high')
    assert_close(242348.637577, PV.T, rtol=1e-7)

@pytest.mark.parametrize("hacks", [True, False])
def test_ethane_PH_failure_high_P(hacks):
    '''Two phase flash for bounding was failing because the VL solution did not
    exist at that pressure.
    '''
    T, P, zs = 402.3703, 101000000.0000, [1.0]
    fluid_idx, eos = 2, SRKMIX # ethane
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
    flasher.VL_only_CEOSs_same = hacks

    base = flasher.flash(T=T, P=P)

    PH = flasher.flash(P=P, H=base.H())
    assert_close(T, PH.T, rtol=1e-7)


@pytest.mark.parametrize("hacks", [True, False])
def test_SRK_high_PT_on_VS_failure(hacks):
    T, P, zs = 7609.496685459907, 423758716.06041414, [1.0]
    fluid_idx, eos = 7, SRKMIX # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
    flasher.VL_only_CEOSs_same = hacks

    base = flasher.flash(T=T, P=P)

    VS = flasher.flash(S=base.S(), V=base.V())
    assert_close(T, VS.T, rtol=1e-7)

    # Point where max P becomes negative - check it is not used
    obj = flasher.flash(T=24.53751106639818, P=33529.24149249553)
    flasher.flash(V=obj.V(), S=obj.S())

@pytest.mark.parametrize("hacks", [True, False])
def test_APISRK_VS_at_Pmax_error_failure(hacks):
    T, P, zs = 7196.856730011477, 355648030.6223078, [1.0]
    fluid_idx, eos = 7, APISRKMIX # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
    flasher.VL_only_CEOSs_same = hacks

    base = flasher.flash(T=T, P=P)

    VS = flasher.flash(S=base.S(), V=base.V())
    assert_close(T, VS.T, rtol=1e-7)

@pytest.mark.parametrize("hacks", [True, False])
def test_Twu_missing_Pmax_on_VS_failure(hacks):
    T, P, zs = 855.4672535565693, 6.260516572014815, [1.0]
    for eos in (TWUSRKMIX, TWUPRMIX):
        fluid_idx = 7 # methanol
        pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

        kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                      HeatCapacityGases=pure_props.HeatCapacityGases)

        liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
        gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
        flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
        flasher.VL_only_CEOSs_same = hacks
        base = flasher.flash(T=T, P=P)
        VS = flasher.flash(S=base.S(), V=base.V())
        assert_close(T, VS.T, rtol=1e-7)

@pytest.mark.parametrize("hacks", [True, False])
def test_TWU_SRK_PR_T_alpha_interp_failure(hacks):
    '''a_alpha becomes 100-500; the EOS makes no sense. Limit it to a Tr no
    around 1E-4 Tr to make it reasonable.
    '''
    T, P, zs = 0.02595024211399719, 6135.90727341312, [1.0]
    fluid_idx = 2 # ethane
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    for eos in [TWUSRKMIX, TWUPRMIX]:
        liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
        gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
        flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
        flasher.VL_only_CEOSs_same = hacks
        base = flasher.flash(T=T, P=P)

        PV = flasher.flash(P=P, V=base.V())
        assert_close(T, PV.T, rtol=1e-8)


@pytest.mark.parametrize("hacks", [True, False])
def test_TS_EOS_fast_path(hacks):
    constants = ChemicalConstantsPackage(Tcs=[647.14], Pcs=[22048320.0], omegas=[0.344], MWs=[18.01528],  CASs=['7732-18-5'],)
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))]
    VaporPressures = [VaporPressure(poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))]
    correlations = PropertyCorrelationsPackage(constants, VaporPressures=VaporPressures, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)

    T, P = 300.0, 1e5
    liquid = CEOSLiquid(PRMIX, T=T, P=P, zs=[1], **kwargs)
    gas = CEOSGas(PRMIX, T=T, P=P, zs=[1], **kwargs)

    # TS one phase fast case
    flasher = FlashPureVLS(constants, correlations, gas, [liquid], []) #
    flasher.VL_only_CEOSs_same = hacks
    # Make a liquid
    S_base = flasher.flash(T=T, P=P).S()
    res_liq = flasher.flash(T=T, S=S_base)
    assert_close(res_liq.P, P)
    # make a gas
    S_base = flasher.flash(T=400.0, P=P).S()
    res = flasher.flash(T=400.0, S=S_base)
    assert_close(res.P, P)

    # TS two phase case
    res_base = flasher.flash(T=T, VF=.3)
    S_base = res_base.S()
    res = flasher.flash(T=T, S=S_base)
    assert_close(res.P, res_base.P)

@pytest.mark.parametrize("hacks", [True, False])
def test_EOS_TP_HSGUA_sln_in_VF(hacks):
    constants = ChemicalConstantsPackage(Tcs=[647.14], Pcs=[22048320.0], omegas=[0.344], MWs=[18.01528],  CASs=['7732-18-5'],)
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))]
    VaporPressures = [VaporPressure(poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))]
    correlations = PropertyCorrelationsPackage(constants, VaporPressures=VaporPressures, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)

    T, P = 300.0, 1e5
    liquid = CEOSLiquid(PRMIX, T=T, P=P, zs=[1], **kwargs)
    gas = CEOSGas(PRMIX, T=T, P=P, zs=[1], **kwargs)

    flasher = FlashPureVLS(constants, correlations, gas, [liquid], []) #
    flasher.VL_only_CEOSs_same = hacks
    base = flasher.flash(T=300, VF=.4)

    new = flasher.flash(H=base.H(), P=base.P)
    assert new.phase_count == 2
    assert_close(new.P, base.P)

    new = flasher.flash(S=base.S(), T=base.T)
    assert new.phase_count == 2
    assert_close(new.P, base.P)

def test_EOS_TP_HSGUA_missing_return_value():
    '''This is only in the hacks for eos. If two volume roots were not present,
    no return value was given.
    '''
    constants = ChemicalConstantsPackage(CASs=['124-38-9'], MWs=[44.0095], omegas=[0.2252], Pcs=[7376460.0], Tcs=[304.2])
    correlations = PropertyCorrelationsPackage(skip_missing=True, constants=constants, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644]))])
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                    HeatCapacityGases=correlations.HeatCapacityGases)
    liquid = CEOSLiquid(PRMIX, T=300, P=1e6, zs=[1], **kwargs)
    gas = CEOSGas(PRMIX, T=300, P=1e6, zs=[1], **kwargs)
    flasher = FlashPureVLS(constants, correlations, gas, [liquid], [])
    H_base = -354.6559586412054
    assert_close(flasher.flash(P=1e6, H=H_base, zs=[1]).T, 300)


def test_EOS_water_hot_start():
    constants = ChemicalConstantsPackage(Tcs=[647.14], Pcs=[22048320.0], omegas=[0.344], MWs=[18.01528],  CASs=['7732-18-5'],)
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)

    T, P = 300.0, 1e5
    liquid = CEOSLiquid(PRMIX, T=T, P=P, zs=[1], **kwargs)
    gas = CEOSGas(PRMIX, T=T, P=P, zs=[1], **kwargs)

    # TS one phase fast case
    flasher = FlashPureVLS(constants, correlations, gas, [liquid], [])

    # Base point
    S_base = flasher.flash(T=T, P=P).S()
    res_liq = flasher.flash(T=T, S=S_base)
    assert_close(res_liq.P, P)

    # Hot start - iterate on P
    resolved = flasher.flash(T=T, S=S_base, hot_start=res_liq)
    assert resolved.flash_convergence['iterations'] < res_liq.flash_convergence['iterations']

    # PH case - iterate on T
    res = flasher.flash(P=P, H=10000)
    resolved = flasher.flash(P=P, H=10000, hot_start=res)
    assert resolved.flash_convergence['iterations'] < res.flash_convergence['iterations']




@pytest.mark.mpmath
@pytest.mark.slow
def test_PRMIXTranslatedConsistent_VS_low_prec_failure():
    '''Numerical issue with volume precision T in the range .001 K at P ~< 100 Pa
    volume_solution works fine, just `V` does not
    '''
    T, P, zs = 0.0013894954943731374, 1e5, [1.0]
    fluid_idx, eos = 7, PRMIXTranslatedConsistent # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    base = flasher.flash(T=T, P=P)

    for P in [109.85411419875584, 1.7575106248547927]:
        base = flasher.flash(T=T, P=P, zs=zs)
        recalc = flasher.flash(S=base.S(), V=base.V_iter(force=True), zs=zs)
        assert_close(base.T, recalc.T, rtol=1e-9)

    # VU failure
    base = flasher.flash(T=2682.6958, P=1e-2, zs=zs)
    recalc = flasher.flash(U=base.U(), V=base.V(), zs=[1])
    assert_close(base.T, recalc.T, rtol=1e-7)

    # VH failure
    base = flasher.flash(T=10000, P=596362331.6594564, zs=zs)
    recalc = flasher.flash(H=base.H(), V=base.V(), zs=zs)
    assert_close(base.T, recalc.T, rtol=1e-7)

    # TV failure - check the higher precision iteraitons are happening
    flashes_base, flashes_new, errs = flasher.TPV_inputs(spec0='T', spec1='P', check0='T', check1='V', prop0='P',
                  Ts=[1e-2, 1e-1, 1, 10], Ps=[1e-1, 1, 100, 1e4], zs=[1], trunc_err_low=1e-20, plot=False)
    assert np.max(errs) < 1e-9


def test_PRMIXTranslatedConsistent_TV_epsilon_consistency_with_fast():
    '''Really interesting bug, where the absolute last place decimal of epsilon
    was different by the slightest amount. The TP and TV initializations used

    -b0*b0 + c*(c + b0 + b0) vs. -b0*b0 + c*c + 2.0*b0

    And this caused all the issues!
    '''
    T, P, zs = 1e-2, 1e-2, [1.0]
    fluid_idx, eos = 8, PRMIXTranslatedConsistent # eicosane
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    base = flasher.flash(T=T, P=P, zs=zs)
    recalc = flasher.flash(T=base.T, V=base.phases[0].V_iter(), zs=zs)
    assert_close(base.P, recalc.P, rtol=1e-7)

def test_SRKMIXTranslatedConsistent_PV_consistency_issue():

    T, P, zs = 1.0001e-3, 1.1e-2, [1.0]
    fluid_idx, eos = 7, SRKMIXTranslatedConsistent # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    flashes_base, flashes_new, errs = flasher.TPV_inputs(spec0='T', spec1='P', check0='P', check1='V', prop0='T',
                  Ts=[1e-3, 1e-2], Ps=[1e-1, 1e-2], zs=[1], trunc_err_low=1e-20, plot=False)
    assert np.max(errs) < 1e-9



def test_TWU_SRK_PR_T_alpha_interp_failure_2():
    T, P, zs = .001, .001, [1.0]
    fluid_idx = 0 # water
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    for eos in [TWUSRKMIX, TWUPRMIX]:
        liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
        gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
        flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
        base = flasher.flash(T=T, P=P)

        PV = flasher.flash(P=P, V=base.V())
        assert_close(T, PV.T, rtol=1e-6)

def test_flash_identical_two_liquids():
    '''Just checks that two flashes, one with two liquids and one with one
    liquid, still both return the same answer.
    '''
    constants = ChemicalConstantsPackage(Tcs=[647.14], Pcs=[22048320.0], omegas=[0.344], MWs=[18.01528],  CASs=['7732-18-5'],)
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)

    liquid = CEOSLiquid(PRMIX, T=300, P=1e5, zs=[1], **kwargs)
    gas = CEOSGas(PRMIX, T=300, P=1e5, zs=[1], **kwargs)
    flasher_2L = FlashPureVLS(constants, correlations, gas, [liquid, liquid], []) #
    Z_2L = flasher_2L.flash(T=300, P=1e5).Z()

    flasher_1L = FlashPureVLS(constants, correlations, gas, [liquid], []) #
    Z_1L = flasher_1L.flash(T=300, P=1e5).Z()
    assert_close(Z_1L, Z_2L)

def test_flash_liquid_only():
    # Test with only one liquid and no gas
    constants = ChemicalConstantsPackage(Tcs=[647.14], Pcs=[22048320.0], omegas=[0.344], MWs=[18.01528],  CASs=['7732-18-5'],)
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)

    liquid = CEOSLiquid(PRMIX, T=400, P=1e5, zs=[1], **kwargs)
    gas = CEOSGas(PRMIX, T=400, P=1e5, zs=[1], **kwargs)
    flasher = FlashPureVLS(constants, correlations, None, [liquid], []) #
    res = flasher.flash(T=400, P=1e5)
    assert_close(res.Z(), 0.0006952919695535529)
    assert gas.G() < res.G()

def test_flash_gas_only():
    # Test with one gas and no liquids
    constants = ChemicalConstantsPackage(Tcs=[647.14], Pcs=[22048320.0], omegas=[0.344], MWs=[18.01528],  CASs=['7732-18-5'],)
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)

    liquid = CEOSLiquid(PRMIX, T=300, P=1e5, zs=[1], **kwargs)
    gas = CEOSGas(PRMIX, T=300, P=1e5, zs=[1], **kwargs)
    flasher = FlashPureVLS(constants, correlations, gas, [], []) #
    res = flasher.flash(T=300, P=1e5)
    assert_close(res.Z(), 0.9847766773458833)
    assert liquid.G() < res.G()


@pytest.mark.parametrize("hacks", [True, False])
def test_VL_EOSMIX_fast_return(hacks):
    T, P, zs = 298.15, 101325.0, [1.0]
    constants = ChemicalConstantsPackage(atom_fractions={'H': 0.6666666666666666, 'O': 0.3333333333333333}, atomss=[{'H': 2, 'O': 1}],CASs=['7732-18-5'], charges=[0], conductivities=[4e-06], dipoles=[1.85], formulas=['H2O'], Gfgs=[-228554.325], Gfgs_mass=[-12686692.9073542], Hcs=[0.0], Hcs_lower=[0.0], Hcs_lower_mass=[0.0], Hcs_mass=[0.0], Hfgs=[-241822.0], Hfgs_mass=[-13423160.783512661], Hfus_Tms=[6010.0], Hfus_Tms_mass=[333605.69472136983], Hvap_298s=[43991.076027756884], Hvap_298s_mass=[2441875.7869850975], Hvap_Tbs=[40643.402624176735], Hvap_Tbs_mass=[2256051.6752543803], InChI_Keys=['XLYOFNOQVPJJNP-UHFFFAOYSA-N'], InChIs=['H2O/h1H2'], logPs=[-1.38], molecular_diameters=[3.24681], MWs=[18.01528], names=['water'], omegas=[0.344], Parachors=[9.368511392279435e-06], Pcs=[22048320.0], phase_STPs=['l'], Psat_298s=[3170.146712628533], PSRK_groups=[{16: 1}], Pts=[610.8773135731733], PubChems=[962], rhocs=[17857.142857142855], rhocs_mass=[321.7014285714285], rhol_STPs=[55287.70167376968], rhol_STPs_mass=[996.0234262094295], S0gs=[188.8], S0gs_mass=[10479.992539666328], Sfgs=[-44.499999999999964], Sfgs_mass=[-2470.1253602497413], similarity_variables=[0.16652530518537598], smiless=['O'], StielPolars=[0.023222134391615246], Stockmayers=[501.01], Tbs=[373.124], Tcs=[647.14], Tms=[273.15], Tts=[273.15], UNIFAC_Dortmund_groups=[{16: 1}], UNIFAC_groups=[{16: 1}], Van_der_Waals_areas=[350000.0], Van_der_Waals_volumes=[1.39564e-05], Vcs=[5.6000000000000006e-05], Vml_STPs=[1.808720510576827e-05], Vml_Tms=[1.801816212354171e-05], Zcs=[0.22947273972184645], UNIFAC_Rs=[0.92], UNIFAC_Qs=[1.4], rhos_Tms=[1126.700421021929], Vms_Tms=[1.5989414456471007e-05], solubility_parameters=[47931.929488349415], Vml_60Fs=[1.8036021352672155e-05], rhol_60Fs=[55287.70167376968], rhol_60Fs_mass=[998.8500039855475])
    properties = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
                                             HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))],
                                             VolumeLiquids=[VolumeLiquid(poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652]))],
                                             VaporPressures=[VaporPressure(poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))],
                                             EnthalpyVaporizations=[EnthalpyVaporization(poly_fit=(273.17, 642.095, 647.14, [3.897048781581251, 94.88210726884502, 975.3860042050983, 5488.360052656942, 18273.86104025691, 36258.751893749475, 42379.63786686855, 39492.82327945519, 58540.84902227406]))],
                                             ViscosityLiquids=[ViscosityLiquid(poly_fit=(273.17, 647.086, [-3.2967840446295976e-19, 1.083422738340624e-15, -1.5170905583877102e-12, 1.1751285808764222e-09, -5.453683174592268e-07, 0.00015251508129341616, -0.024118558027652552, 1.7440690494170135, -24.96090630337129]))],
                                             ViscosityGases=[ViscosityGas(poly_fit=(273.16, 1073.15, [-1.1818252575481647e-27, 6.659356591849417e-24, -1.5958127917299133e-20, 2.1139343137119052e-17, -1.6813187290802144e-14, 8.127448028541097e-12, -2.283481528583874e-09, 3.674008403495927e-07, -1.9313694390100466e-05]))])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas,
    alpha_coeffs=[[0.3872, 0.87587208, 1.9668]], cs=[5.2711E-6])
    gas = CEOSGas(PRMIXTranslatedConsistent, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIXTranslatedConsistent, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases, T=T, P=P, zs=zs)
    flasher = FlashPureVLS(constants, properties, liquids=[liq], gas=gas, solids=[])
    flasher.VL_only_CEOSs_same = hacks

    res = flasher.flash(T=300, P=1e5)
    # Point missing the right phase return
    assert_close(res.Z(), 0.0006437621418058624, rtol=1e-5)

    # Do a vapor check for consistency
    point = flasher.flash(T=300, P=3200)
    assert_close(point.Z(), 0.9995223890086967, rtol=1e-5)

    # Do a check exception is raised for T
    with pytest.raises(PhaseExistenceImpossible):
        res = flasher.flash(T=900, VF=1)
    with pytest.raises(PhaseExistenceImpossible):
        res = flasher.flash(T=900, VF=.5)
    with pytest.raises(PhaseExistenceImpossible):
        res = flasher.flash(T=900, VF=0)
    with pytest.raises(PhaseExistenceImpossible):
        res = flasher.flash(P=constants.Pcs[0]*1.1, VF=1)
    with pytest.raises(PhaseExistenceImpossible):
        res = flasher.flash(P=constants.Pcs[0]*1.1, VF=.5)
    with pytest.raises(PhaseExistenceImpossible):
        res = flasher.flash(P=constants.Pcs[0]*1.1, VF=0)


def test_APISRK_multiple_T_slns():
    constants = ChemicalConstantsPackage(Tcs=[768.0], Pcs=[1070000.0], omegas=[0.8805], MWs=[282.54748], CASs=['112-95-8'])
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.075118433508619e-20, 1.0383055980949049e-16, -2.1577805903757125e-13, 2.373511052680461e-10, -1.4332562489496906e-07, 4.181755403465859e-05, -0.0022544761674344544, -0.15965342941876415, 303.71771182550816]))]

    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                    HeatCapacityGases=HeatCapacityGases)

    liquid = CEOSLiquid(APISRKMIX, T=330, P=1e5, zs=[1], **kwargs)
    gas = CEOSGas(APISRKMIX, T=330, P=1e5, zs=[1], **kwargs)

    flasher = FlashPureVLS(constants, correlations, gas, [liquid], [])

    for T, sln in zip([10000, 10000, 10000, 6741.680441295266, 6741.680441295266],
                      [None, 'high', lambda obj: obj.G(), 'low', lambda obj: obj.T]):
        obj = flasher.flash(V=0.0026896181445057303, P=14954954.954954954, zs=[1], solution=sln)
        assert_close(obj.T, T)

    for T, sln in zip([140184.08901758507, 140184.08901758507, 140184.08901758507, 7220.8089999999975, 7220.8089999999975],
                      [None, 'high', lambda obj: obj.G(), 'low', lambda obj: obj.T]):
        obj = flasher.flash(V=0.0006354909990692889, P=359381366.3805, zs=[1], solution=sln)
        assert_close(obj.T, T)


@pytest.mark.parametrize("hacks", [True, False])
def test_IG_liq_poy_flashes(hacks):
    T, P, zs = 300.0, 1e5, [1.0]
    constants = ChemicalConstantsPackage(atom_fractions={'H': 0.6666666666666666, 'O': 0.3333333333333333},
                                         atomss=[{'H': 2, 'O': 1}],CASs=['7732-18-5'], charges=[0],
                                         conductivities=[4e-06], dipoles=[1.85], formulas=['H2O'], Gfgs=[-228554.325],
                                         Gfgs_mass=[-12686692.9073542], Hcs=[0.0], Hcs_lower=[0.0], Hcs_lower_mass=[0.0],
                                         Hcs_mass=[0.0], Hfgs=[-241822.0], Hfgs_mass=[-13423160.783512661], Hfus_Tms=[6010.0],
                                         Hfus_Tms_mass=[333605.69472136983], Hvap_298s=[43991.076027756884], Hvap_298s_mass=[2441875.7869850975],
                                         Hvap_Tbs=[40643.402624176735], Hvap_Tbs_mass=[2256051.6752543803],
                                         molecular_diameters=[3.24681], MWs=[18.01528], names=['water'], omegas=[0.344],
                                         Parachors=[9.368511392279435e-06], Pcs=[22048320.0], phase_STPs=['l'], Psat_298s=[3170.146712628533],
                                         Pts=[610.8773135731733], PubChems=[962], rhocs=[17857.142857142855], rhocs_mass=[321.7014285714285],
                                         rhol_STPs=[55287.70167376968], rhol_STPs_mass=[996.0234262094295], S0gs=[188.8],
                                         S0gs_mass=[10479.992539666328], Sfgs=[-44.499999999999964], Sfgs_mass=[-2470.1253602497413],
                                         similarity_variables=[0.16652530518537598], smiless=['O'], StielPolars=[0.023222134391615246],
                                         Stockmayers=[501.01], Tbs=[373.124], Tcs=[647.14], Tms=[273.15], Tts=[273.15],
                                         Van_der_Waals_areas=[350000.0], Van_der_Waals_volumes=[1.39564e-05], Vcs=[5.6000000000000006e-05],
                                         Vml_STPs=[1.808720510576827e-05], Vml_Tms=[1.801816212354171e-05], Zcs=[0.22947273972184645],
                                         rhos_Tms=[1126.700421021929], Vms_Tms=[1.5989414456471007e-05], solubility_parameters=[47931.929488349415],
                                         Vml_60Fs=[1.8036021352672155e-05], rhol_60Fs=[55287.70167376968], rhol_60Fs_mass=[998.8500039855475])
    VaporPressures = [VaporPressure(poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317]))]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))]
    VolumeLiquids = [VolumeLiquid(poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652]))]
    correlations = PropertyCorrelationsPackage(constants, skip_missing=True, VolumeLiquids=VolumeLiquids, VaporPressures=VaporPressures, HeatCapacityGases=HeatCapacityGases)
    liquid = GibbsExcessLiquid(VaporPressures=VaporPressures,
                               HeatCapacityGases=HeatCapacityGases,
                               VolumeLiquids=VolumeLiquids,
                               use_phis_sat=False, use_Poynting=True, Psat_extrpolation='ABC').to_TP_zs(T, P, zs)
    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(IGMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    flasher = FlashPureVLS(constants, correlations, liquids=[liquid], gas=gas, solids=[])
    flasher.VL_IG_activity = hacks

    # Neat critical transition
    res = flasher.flash(T=646, P=21777293)
    assert res.gas is not None
    assert 1 == res.phase_count
    assert_close(res.Z(), 1, rtol=1e-12)

    res = flasher.flash(T=646, P=21777294)
    assert res.liquid0 is not None
    assert 1 == res.phase_count
    assert_close(res.rho_mass(), 455.930726761562)


    # Low temperature and very low pressure transition
    res = flasher.flash(T=50, P=1e-4)
    assert res.liquid0 is not None
    assert 1 == res.phase_count
    assert_close(res.H(), -62196.88248213482)
     # If the low pressure vapor pressure extrapolation method changes, this one will fail
    res = flasher.flash(T=50, P=1e-45)
    assert res.gas is not None
    assert 1 == res.phase_count
    assert_close(res.H(), -8268.890506895972)

    # Neat STP transition
    res = flasher.flash(T=298.15, P=3200)
    assert res.liquid0 is not None
    assert 1 == res.phase_count
    assert_close(res.rho_mass(), 996.0234262094295)

    res = flasher.flash(T=298.15, P=3100)
    assert res.gas is not None
    assert 1 == res.phase_count
    assert_close(res.Z(), 1, rtol=1e-12)


    # STP is a liquid
    res = flasher.flash(T=298.15, P=101325.0)
    assert res.liquid0 is not None
    assert 1 == res.phase_count
    assert_close(res.rho_mass(), 996.0234262094295)

    if hacks:
        # Very high T, P transition
        res =  flasher.flash(T=1000, P=1e7)
        assert res.gas is not None
        assert 1 == res.phase_count
        assert_close(res.Z(), 1, rtol=1e-12)

        res =  flasher.flash(T=1000, P=1e10)
        assert res.liquid0 is not None
        assert 1 == res.phase_count
        assert_close1d(res.liquid0.lnphis(), [199.610302])

    # Vapor fraction flashes
    res = flasher.flash(T=646, VF=0)
    assert_close(res.P, 21777293.25835532)
    res = flasher.flash(T=646, VF=.5)
    assert_close(res.P, 21777293.25835532)
    res = flasher.flash(T=646, VF=1)
    assert_close(res.P, 21777293.25835532)

    res = flasher.flash(P=21777293.25835532, VF=0)
    assert_close(res.T, 646)
    res = flasher.flash(P=21777293.25835532, VF=.5)
    assert_close(res.T, 646)
    res = flasher.flash(P=21777293.25835532, VF=1)
    assert_close(res.T, 646)

    # PH
    res = flasher.flash(H=-43925.16879798105, P=1e5)
    assert_close(res.T, 300.0)

def test_VF_H_cases():
    T, P, zs = 350.0, 1e5, [1.0]
    fluid_idx = 7 # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    eos = PRMIX
    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    # Two solutions
    VFs_two = [0.9285628571428575, 0.7857085714285716, 0.5714271428571429,
               0.5714271428571429,
               0.9
               ]
    Ts_two_low = [312.9795918367355, 481.71428571428584, 507.6734693877551,
                  506.84085844724433, 320.0050872321796,
                  ]
    Ts_two_high = [512.214210227972, 484.30741617882694, 511.70674621843665,
                   512.0,
                   512.49999,
                   ]

    for VF, T_low, T_high in zip(VFs_two, Ts_two_low, Ts_two_high):
        base = flasher.flash(T=T_low, VF=VF)
        low = flasher.flash(H=base.H(), VF=VF, solution='low')
        assert_close(low.T, base.T, rtol=1e-8)
        assert_close(low.H(), base.H(), rtol=1e-7)

        high = flasher.flash(H=base.H(), VF=VF, solution='high')
        assert_close(high.T, T_high, rtol=1e-8)
        assert_close(high.H(), base.H(), rtol=1e-7)


    solutions = ['mid', 'low', 'high']

    one_sln_Ts = [512.0,
                  486.0408163265307, # Covers 2 solutions trying to find middle point - make sure is handled
                  512.0,
                  200.0, # Point where room for improvement exists - should not need to find root as sln is far
                  512.5 # boundary
                  ]
    one_sln_VFs = [0.5, 0.5714271428571429, 1e-5, 0.4827586206896553, 0.3]

    for T, VF in zip(one_sln_Ts, one_sln_VFs):
        base = flasher.flash(T=T, VF=VF)
        for s in solutions:
            new = flasher.flash(H=base.H(), VF=VF, solution=s)
            assert_close(new.T, base.T, rtol=1e-8)


def test_VF_U_cases():
    T, P, zs = 350.0, 1e5, [1.0]
    fluid_idx = 7 # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    eos = PRMIX
    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    # Two solutions
    VFs_two =[0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9285628571428575, 0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9]

    Ts_two_low = [481.71428571428584, 507.6734693877551, 506.84085844724433, 337.15037336596805, 484.30741617882694, 509.784104700645, 509.1465329286005, 344.57339768096267]

    Ts_two_high = [496.74470283423096, 512.3840598446428, 512.4750615095567, 512.214210227972, 494.79103350924, 511.70674621843665, 512.0, 512.49999]



    for VF, T_low, T_high in zip(VFs_two, Ts_two_low, Ts_two_high):
        base = flasher.flash(T=T_low, VF=VF)
        low = flasher.flash(U=base.U(), VF=VF, solution='low')
        assert_close(low.T, base.T, rtol=1e-8)
        assert_close(low.U(), base.U(), rtol=1e-7)

        high = flasher.flash(U=base.U(), VF=VF, solution='high')
        assert_close(high.T, T_high, rtol=1e-8)
        assert_close(high.U(), base.U(), rtol=1e-7)


    solutions = ['mid', 'low', 'high']

    one_sln_Ts = [312.9795918367355, 320.0050872321796, 512.0, 486.0408163265307, 512.0, 200.0, 512.5
                  ]
    one_sln_VFs = [0.9285628571428575, 0.9, 0.5, 0.5714271428571429, 1e-05, 0.4827586206896553, 0.3]

    for T, VF in zip(one_sln_Ts, one_sln_VFs):
        base = flasher.flash(T=T, VF=VF)
        for s in solutions:
            new = flasher.flash(U=base.U(), VF=VF, solution=s)
            assert_close(new.T, base.T, rtol=1e-8)


def test_VF_A_cases():
    T, P, zs = 350.0, 1e5, [1.0]
    fluid_idx = 7 # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    eos = PRMIX
    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    # Two solutions
    VFs_two = [1e-05, 0.3]
    Ts_two_low = [505.1428075489603, 510.509170507811]
    Ts_two_high = [512.0, 512.49999]
    for VF, T_low, T_high in zip(VFs_two, Ts_two_low, Ts_two_high):
        base = flasher.flash(T=T_low, VF=VF)
        low = flasher.flash(A=base.A(), VF=VF, solution='low')
        assert_close(low.T, base.T, rtol=1e-8)
        assert_close(low.A(), base.A(), rtol=1e-7)

        high = flasher.flash(A=base.A(), VF=VF, solution='high')
        assert_close(high.T, T_high, rtol=1e-8)
        assert_close(high.A(), base.A(), rtol=1e-7)


    solutions = ['mid', 'low', 'high']
    one_sln_Ts =  [312.9795918367355, 481.71428571428584, 507.6734693877551, 506.84085844724433, 320.0050872321796, 512.214210227972, 484.30741617882694, 511.70674621843665, 512.0, 512.49999, 512.0, 486.0408163265307, 200.0]
    one_sln_VFs = [0.9285628571428575, 0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9, 0.9285628571428575, 0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9, 0.5, 0.5714271428571429, 0.4827586206896553]

    for T, VF in zip(one_sln_Ts, one_sln_VFs):
        base = flasher.flash(T=T, VF=VF)
        for s in solutions:
            new = flasher.flash(A=base.A(), VF=VF, solution=s)
            assert_close(new.T, base.T, rtol=1e-8)

def test_VF_G_cases():
    # No double solutions for G - almost no need to iterate
    T, P, zs = 350.0, 1e5, [1.0]
    fluid_idx = 7 # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    eos = PRMIX
    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])


    solutions = ['mid', 'low', 'high']
    one_sln_Ts =  [312.9795918367355, 481.71428571428584, 507.6734693877551, 506.84085844724433, 320.0050872321796, 512.214210227972, 484.30741617882694, 511.70674621843665, 512.0, 512.49999, 512.0, 486.0408163265307, 512.0, 200.0, 512.49999]
    one_sln_VFs = [0.9285628571428575, 0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9, 0.9285628571428575, 0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9, 0.5, 0.5714271428571429, 1e-05, 0.4827586206896553, 0.3]

    for T, VF in zip(one_sln_Ts, one_sln_VFs):
        base = flasher.flash(T=T, VF=VF)
        for s in solutions:
            new = flasher.flash(G=base.G(), VF=VF, solution=s)
            assert_close(new.T, base.T, rtol=1e-8)


def test_VF_S_cases():
    '''
    S has cases with three solutions. Lots of work remains here! The plot does
    not look very healthy, and I am not convinced the two solution point is not
    actually three point.
    '''
    T, P, zs = 350.0, 1e5, [1.0]
    fluid_idx = 7 # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])

    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    eos = PRMIX
    liquid = CEOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = CEOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    # Two solutions
    VFs_two = [0.5714271428571429, 0.4827586206896553]
    Ts_two_low = [212.16975817104446, 199.99999999981264]
    Ts_two_high = [486.04081632653146, 359.28026319594096]

    for VF, T_low, T_high in zip(VFs_two, Ts_two_low, Ts_two_high):
        base = flasher.flash(T=T_low, VF=VF)
        low = flasher.flash(S=base.S(), VF=VF, solution='low')
        assert_close(low.T, base.T, rtol=1e-5)
        assert_close(low.S(), base.S(), rtol=1e-7)

        high = flasher.flash(S=base.S(), VF=VF, solution='high')
        assert_close(high.T, T_high, rtol=1e-5)
        assert_close(high.S(), base.S(), rtol=1e-7)


    solutions = ['mid', 'low', 'high']
    one_sln_Ts =  [312.9795918367355, 481.71428571428584, 320.0050872321796, 512.214210227972, 484.30741617882694, 512.49999, 512.0, 512.0, 512.49999]
    one_sln_VFs = [0.9285628571428575, 0.7857085714285716, 0.9, 0.9285628571428575, 0.7857085714285716, 0.9, 0.5, 1e-05, 0.3]

    for T, VF in zip(one_sln_Ts, one_sln_VFs):
        base = flasher.flash(T=T, VF=VF)
        for s in solutions:
            new = flasher.flash(S=base.S(), VF=VF, solution=s)
            assert_close(new.T, base.T, rtol=1e-8)


    VFs_three = [0.5714271428571429, 0.5714271428571429, 0.5714271428571429, 0.5714271428571429]
    Ts_three_low = [208.4063161102236, 208.42865241364098, 209.04825869752273, 209.253671056617]
    Ts_three_mid = [507.6734693878061, 506.8408584472489, 500.99800548138063, 499.7044524953537]
    Ts_three_high = [508.178797993246, 508.9013433039185, 511.7067462184334, 512.0000000000291]

    for VF, T_low, T_mid, T_high in zip(VFs_three, Ts_three_low, Ts_three_mid, Ts_three_high):
        base = flasher.flash(T=T_low, VF=VF)
        low = flasher.flash(S=base.S(), VF=VF, solution='low')
        assert_close(low.T, base.T, rtol=1e-5)
        assert_close(low.S(), base.S(), rtol=1e-7)

        mid = flasher.flash(S=base.S(), VF=VF, solution='mid')
        assert_close(mid.T, T_mid, rtol=1e-5)
        assert_close(mid.S(), base.S(), rtol=1e-7)

        high = flasher.flash(S=base.S(), VF=VF, solution='high')
        assert_close(high.T, T_high, rtol=1e-5)
        assert_close(high.S(), base.S(), rtol=1e-7)


def test_methanol_VF_HSGUA_issues():
    pass

def test_methanol_inconsistent_full_example():
    from thermo.heat_capacity import POLING_POLY

    CpObj = HeatCapacityGas(CASRN='67-56-1')
    CpObj.method = POLING_POLY
    constants = ChemicalConstantsPackage(Tcs=[512.5], Pcs=[8084000.0], omegas=[0.559], MWs=[32.04186], CASs=['67-56-1'])
    HeatCapacityGases = [CpObj]

    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    liquid = CEOSLiquid(PRMIX, HeatCapacityGases=HeatCapacityGases, eos_kwargs=eos_kwargs)
    gas = CEOSGas(PRMIX, HeatCapacityGases=HeatCapacityGases, eos_kwargs=eos_kwargs)
    SRK_gas = CEOSGas(SRKMIX, HeatCapacityGases=HeatCapacityGases, eos_kwargs=eos_kwargs)

    # Known to not converge at higher T
    flasher_inconsistent = FlashPureVLS(constants, correlations, gas=SRK_gas, liquids=[liquid], solids=[])
    res = flasher_inconsistent.flash(T=400.0, VF=1)

    assert_close(res.P, 797342.226264512)
    assert_close(res.gas.rho_mass(), 8.416881310028323)
    assert_close(res.liquid0.rho_mass(), 568.7838890605196)


def test_VF_SF_spec_bound_0_1_and_negative_TPV():
    from thermo.heat_capacity import POLING_POLY
    
    CpObj = HeatCapacityGas(CASRN='67-56-1')
    CpObj.method = POLING_POLY
    constants = ChemicalConstantsPackage(Tcs=[512.5], Pcs=[8084000.0], omegas=[0.559], MWs=[32.04186], CASs=['67-56-1'])
    HeatCapacityGases = [CpObj]
    
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    liquid = CEOSLiquid(PRMIX, HeatCapacityGases=HeatCapacityGases, eos_kwargs=eos_kwargs)
    gas = CEOSGas(PRMIX, HeatCapacityGases=HeatCapacityGases, eos_kwargs=eos_kwargs)
    
    flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])
    for VF in (-1, 2, 1.1, 1.0000000000001, -1e-200, 100, -1e-100, 1e200):
        with pytest.raises(ValueError):
            res = flasher.flash(T=400.0, VF=VF)
        with pytest.raises(ValueError):
            res = flasher.flash(P=1e5, VF=VF)
        with pytest.raises(ValueError):
            res = flasher.flash(T=400.0, SF=VF)
        with pytest.raises(ValueError):
            res = flasher.flash(P=1e5, SF=VF)
        for s in ('H', 'S', 'G', 'U', 'A'):
            with pytest.raises(ValueError):
                kwargs = {s: 10.0, 'VF': VF}
                res = flasher.flash(**kwargs)
    
            with pytest.raises(ValueError):
                kwargs = {s: 10.0, 'SF': VF}
                res = flasher.flash(**kwargs)
                
    with pytest.raises(ValueError):              
        flasher.flash(T=300.0, V=-2)
        
    with pytest.raises(ValueError):              
        flasher.flash(T=300.0, P=-2)
        
    with pytest.raises(ValueError):              
        flasher.flash(T=300.0, V=-2)

    with pytest.raises(ValueError):              
        flasher.flash(T=300.0, V=0)
        
    with pytest.raises(ValueError):              
        flasher.flash(T=300.0, P=0)
        
    with pytest.raises(ValueError):              
        flasher.flash(T=300.0, V=0)