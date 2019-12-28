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
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

from numpy.testing import assert_allclose
import pytest
import thermo
from thermo import *
from fluids.numerics import *
from math import *
import json
import os
import numpy as np

from thermo.test_utils import *
import matplotlib.pyplot as plt

pure_surfaces_dir = os.path.join(thermo.thermo_dir, '..', 'surfaces', 'pure')

pure_fluids = ['water', 'methane', 'ethane', 'decane', 'ammonia', 'nitrogen', 'oxygen', 'methanol', 'eicosane', 'hydrogen']

'''# Recreate the below with the following:
N = len(pure_fluids)
m = Mixture(pure_fluids, zs=[1/N]*N, T=298.15, P=1e5)
print(m.constants.make_str(delim=', \n', properties=('Tcs', 'Pcs', 'omegas', 'MWs', "CASs")))
correlations = m.properties()
print(correlations.as_best_fit(['HeatCapacityGases']))
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

correlations = PropertyCorrelationPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
        HeatCapacityGas(best_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [7.444966286051841e-23, 9.444106746563928e-20, -1.2490299714587002e-15, 2.6693560979905865e-12, -2.5695131746723413e-09, 1.2022442523089315e-06, -0.00021492132731007108, 0.016616385291696574, 32.84274656062226])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924])),
        HeatCapacityGas(best_fit=(200.0, 1000.0, [-2.075118433508619e-20, 1.0383055980949049e-16, -2.1577805903757125e-13, 2.373511052680461e-10, -1.4332562489496906e-07, 4.181755403465859e-05, -0.0022544761674344544, -0.15965342941876415, 303.71771182550816])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [1.1878323802695824e-20, -5.701277266842367e-17, 1.1513022068830274e-13, -1.270076105261405e-10, 8.309937583537026e-08, -3.2694889968431594e-05, 0.007443050245274358, -0.8722920255910297, 66.82863369121873])),
        ])

from thermo.eos_mix import eos_mix_list



def plot_unsupported(reason, color='r'):
    '''Helper function - draw a plot with an `x` over it displaying a message
    why that plot is not supported.
    '''
    fig, ax = plt.subplots()

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.plot([0, 1], [0, 1], lw=5, c=color)
    ax.plot([0, 1], [1, 0], lw=5, c=color)

    ax.text(.5, .5, reason, ha='center', va='center', bbox=dict(fc='white'))
    return fig






#eos_mix_list = [PRMIX, PR78MIX, SRKMIX, VDWMIX, PRSVMIX, PRSV2MIX, APISRKMIX, TWUPRMIX, TWUSRKMIX, IGMIX]
#eos_mix_list = [TWUPRMIX, TWUSRKMIX] # issues
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
    pure_props = PropertyCorrelationPackage(pure_const, HeatCapacityGases=HeatCapacityGases)
    '''
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

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

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

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

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

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

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

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
#test_PH_plot('eicosane', TWUPRMIX, 'physical')
#test_PH_plot("hydrogen", TWUPRMIX, "physical")
#test_PH_plot("hydrogen", TWUSRKMIX, "physical")
#test_PH_plot("hydrogen", TWUPRMIX, "realistic")
#test_PH_plot("hydrogen", TWUSRKMIX, "realistic")


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

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

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

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

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

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

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

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

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

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

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



@pytest.mark.slow
@pytest.mark.skip
@pytest.mark.parametric
@pytest.mark.parametrize("auto_range", ['physical', 'realistic'])
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_TH_plot(fluid, eos, auto_range):
    '''
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

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

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

### Pure EOS only tests
@pytest.mark.slow
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
#    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    
    kwargs = dict(Tc=pure_const.Tcs[0], Pc=pure_const.Pcs[0], omega=pure_const.omegas[0])
    
    gas = eos(T=T, P=P, **kwargs)
    errs, plot_fig = gas.volumes_G_min(plot=True, show=False, pts=150,
                                       Tmin=1e-4, Tmax=1e4, Pmin=1e-2, Pmax=1e9)

    
    path = os.path.join(pure_surfaces_dir, fluid, "V_G_min")
    if not os.path.exists(path):
        os.makedirs(path)
    
    key = '%s - %s - %s' %('V_G_min', eos.__name__, fluid)
        
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()
    
    # Not sure how to add error to this one
#test_V_G_min_plot('hydrogen', TWUPR)
#test_V_G_min_plot('hydrogen', TWUSRK)


@pytest.mark.slow
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
@pytest.mark.parametric
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_list)
@pytest.mark.parametrize("P_range", ['high', 'low'])
def test_V_error_plot(fluid, eos, P_range):
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
    
    
    
    if P_range == 'high':
        Pmin = 1e-2
        Pmax = 1e9
    elif P_range == 'low':
        Pmax = 1e-2
        Pmin = 1e-60
#    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    obj = eos(T=T, P=P, **kwargs)
    errs, plot_fig = obj.volume_errors(plot=True, show=False, pts=50,
                                       Tmin=1e-4, Tmax=1e4, Pmin=Pmin, Pmax=Pmax,
                                       trunc_err_low=1e-15, color_map=cm_flash_tol())

    
        
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()
    
    
    max_err = np.max(errs)
    assert max_err < 1e-10

#test_V_error_plot('ethane', PR, 'high')
#test_V_error_plot('decane', PR, 'high')
#test_V_error_plot('hydrogen', TWUSRKMIX, 'high')
#test_V_error_plot('hydrogen', IGMIX, 'low')



@pytest.mark.slow
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
    a_alphas, plot_fig = obj.a_alpha_plot(Tmin=1e-4, Tmax=pure_const.Tcs[0]*30, pts=500,
                                          plot=True, show=False)

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

### Non-generic tests
    
def test_some_flashes_bad():
    '''Basic test with ammonia showing how PG, PA, TG, TA, VA, VG flashes
    have multiple solutions quite close.
    '''
    constants = ChemicalConstantsPackage(Tcs=[405.6], Pcs=[11277472.5], omegas=[0.25], MWs=[17.03052], CASs=['7664-41-7'])
    HeatCapacityGases = [HeatCapacityGas(best_fit=(50.0, 1000.0, [7.444966286051841e-23, 9.444106746563928e-20,
                            -1.2490299714587002e-15, 2.6693560979905865e-12, -2.5695131746723413e-09, 1.2022442523089315e-06, 
                            -0.00021492132731007108, 0.016616385291696574, 32.84274656062226]))]
    correlations = PropertyCorrelationPackage(constants, HeatCapacityGases=HeatCapacityGases)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)
    
    liquid = EOSLiquid(SRKMIX, T=330, P=1e5, zs=[1], **kwargs)
    gas = EOSGas(SRKMIX, T=330, P=1e5, zs=[1], **kwargs)
    
    flasher = FlashPureVLS(constants, correlations, gas, [liquid], [])
        
    assert_allclose(flasher.flash(T=800, P=1e7).G(), flasher.flash(T=725.87092453, P=1e7).G(), rtol=1e-10)

def test_VS_issue_PRSV():
    constants = ChemicalConstantsPackage(Tcs=[768.0], Pcs=[1070000.0], omegas=[0.8805], MWs=[282.54748], CASs=['112-95-8'])
    HeatCapacityGases = [HeatCapacityGas(best_fit=(200.0, 1000.0, [-2.075118433508619e-20, 1.0383055980949049e-16, -2.1577805903757125e-13, 2.373511052680461e-10, -1.4332562489496906e-07, 4.181755403465859e-05, -0.0022544761674344544, -0.15965342941876415, 303.71771182550816]))]
    
    correlations = PropertyCorrelationPackage(constants, HeatCapacityGases=HeatCapacityGases)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                    HeatCapacityGases=HeatCapacityGases)
    
    liquid = EOSLiquid(PRSVMIX, T=330, P=1e5, zs=[1], **kwargs)
    gas = EOSGas(PRSVMIX, T=330, P=1e5, zs=[1], **kwargs)
    
    flasher = FlashPureVLS(constants, correlations, gas, [liquid], [])
    obj = flasher.flash(T=7196.856730011477, P=212095088.7920158)
    assert_allclose(obj.T, flasher.flash(V=obj.V(), S=obj.S()).T)

def test_PS_1P_vs_VL_issue0():
    '''Made me think there was something wrong with enthalpy maximization.
    However, it was just a root issue.
    '''

    constants = ChemicalConstantsPackage(Tcs=[647.14], Pcs=[22048320.0], omegas=[0.344], MWs=[18.01528],  CASs=['7732-18-5'],)
    HeatCapacityGases = [HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))]
    correlations = PropertyCorrelationPackage(constants, HeatCapacityGases=HeatCapacityGases)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                 HeatCapacityGases=HeatCapacityGases)
    
    liquid = EOSLiquid(PR78MIX, T=200, P=1e5, zs=[1], **kwargs)
    gas = EOSGas(PR78MIX, T=200, P=1e5, zs=[1], **kwargs)
    flasher = FlashPureVLS(constants, correlations, gas, [liquid], []) # 
    
    for P in (0.01, 0.015361749466718281):
        obj = flasher.flash(T=166.0882782627715, P=P)
        hit = flasher.flash(P=obj.P, S=obj.S())
        assert_allclose(hit.T, obj.T)


def test_SRK_high_P_PV_failure():
    '''Numerical solve_T converging to a T in the range 200,000 K when there
    was a lower T solution
    '''
    T, P, zs = 2000, 1e8, [1.0]
    fluid_idx, eos = 7, SRKMIX # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    base = flasher.flash(T=T, P=P)
    
    PV = flasher.flash(P=P, V=base.V(), solution='low')
    assert_allclose(T, PV.T, rtol=1e-7)

    PV = flasher.flash(P=P, V=base.V(), solution='high')
    assert_allclose(242348.637577, PV.T, rtol=1e-7)


def test_SRK_high_PT_on_VS_failure():
    T, P, zs = 7609.496685459907, 423758716.06041414, [1.0]
    fluid_idx, eos = 7, SRKMIX # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    base = flasher.flash(T=T, P=P)
    
    VS = flasher.flash(S=base.S(), V=base.V())
    assert_allclose(T, VS.T, rtol=1e-7)

    # Point where max P becomes negative - check it is not used
    obj = flasher.flash(T=24.53751106639818, P=33529.24149249553)
    flasher.flash(V=obj.V(), S=obj.S())
    
def test_APISRK_VS_at_Pmax_error_failure():
    T, P, zs = 7196.856730011477, 355648030.6223078, [1.0]
    fluid_idx, eos = 7, APISRKMIX # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    base = flasher.flash(T=T, P=P)
    
    VS = flasher.flash(S=base.S(), V=base.V())
    assert_allclose(T, VS.T, rtol=1e-7)

def test_Twu_missing_Pmax_on_VS_failure():
    T, P, zs = 855.4672535565693, 6.260516572014815, [1.0]
    for eos in (TWUSRKMIX, TWUPRMIX):
        fluid_idx = 7 # methanol
        pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
        
        kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                      HeatCapacityGases=pure_props.HeatCapacityGases)
    
        liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
        gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
        flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
        base = flasher.flash(T=T, P=P)
        VS = flasher.flash(S=base.S(), V=base.V())
        assert_allclose(T, VS.T, rtol=1e-7)

def test_TWU_SRK_PR_T_alpha_interp_failure():
    '''a_alpha becomes 100-500; the EOS makes no sense. Limit it to a Tr no
    around 1E-4 Tr to make it reasonable. 
    '''
    T, P, zs = 0.02595024211399719, 6135.90727341312, [1.0]
    fluid_idx = 2 # ethane
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)
    
    for eos in [TWUSRKMIX, TWUPRMIX]:
        liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
        gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
        flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
        base = flasher.flash(T=T, P=P)
        
        PV = flasher.flash(P=P, V=base.V())
        assert_allclose(T, PV.T, rtol=1e-8)

def test_PRMIXTranslatedConsistent_VS_low_prec_failure():
    '''Numerical issue with volume precision T in the range .001 K at P ~< 100 Pa
    volume_solution works fine, just `V` does not
    '''
    T, P, zs = 0.0013894954943731374, 1e5, [1.0]
    fluid_idx, eos = 7, PRMIXTranslatedConsistent # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    base = flasher.flash(T=T, P=P)
    
    for P in [109.85411419875584, 1.7575106248547927]:
        base = flasher.flash(T=T, P=P, zs=zs)
        recalc = flasher.flash(S=base.S(), V=base.V_iter(True), zs=zs)
        assert_allclose(base.T, recalc.T, rtol=1e-9)
    
    # VU failure
    base = flasher.flash(T=2682.6958, P=1e-2, zs=zs)
    recalc = flasher.flash(U=base.U(), V=base.V(), zs=[1])
    assert_allclose(base.T, recalc.T, rtol=1e-7)

    # VH failure
    base = flasher.flash(T=10000, P=596362331.6594564, zs=zs)
    recalc = flasher.flash(H=base.H(), V=base.V(), zs=zs)
    assert_allclose(base.T, recalc.T, rtol=1e-7)
    
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

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    base = flasher.flash(T=T, P=P, zs=zs)
    recalc = flasher.flash(T=base.T, V=base.phases[0].V_iter(), zs=zs)
    assert_allclose(base.P, recalc.P, rtol=1e-7)

def test_SRKMIXTranslatedConsistent_PV_consistency_issue():

    T, P, zs = 1.0001e-3, 1.1e-2, [1.0]
    fluid_idx, eos = 7, SRKMIXTranslatedConsistent # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)
    
    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
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
        liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
        gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
        flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])
        base = flasher.flash(T=T, P=P)
        
        PV = flasher.flash(P=P, V=base.V())
        assert_allclose(T, PV.T, rtol=1e-6)

def test_APISRK_multiple_T_slns():
    constants = ChemicalConstantsPackage(Tcs=[768.0], Pcs=[1070000.0], omegas=[0.8805], MWs=[282.54748], CASs=['112-95-8'])
    HeatCapacityGases = [HeatCapacityGas(best_fit=(200.0, 1000.0, [-2.075118433508619e-20, 1.0383055980949049e-16, -2.1577805903757125e-13, 2.373511052680461e-10, -1.4332562489496906e-07, 4.181755403465859e-05, -0.0022544761674344544, -0.15965342941876415, 303.71771182550816]))]
    
    correlations = PropertyCorrelationPackage(constants, HeatCapacityGases=HeatCapacityGases)
    kwargs = dict(eos_kwargs=dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas),
                    HeatCapacityGases=HeatCapacityGases)
    
    liquid = EOSLiquid(APISRKMIX, T=330, P=1e5, zs=[1], **kwargs)
    gas = EOSGas(APISRKMIX, T=330, P=1e5, zs=[1], **kwargs)
    
    flasher = FlashPureVLS(constants, correlations, gas, [liquid], [])
    
    for T, sln in zip([10000, 10000, 10000, 6741.680441295266, 6741.680441295266],
                      [None, 'high', lambda obj: obj.G(), 'low', lambda obj: obj.T]):
        obj = flasher.flash(V=0.0026896181445057303, P=14954954.954954954, zs=[1], solution=sln)
        assert_allclose(obj.T, T)

    for T, sln in zip([140184.08901758507, 140184.08901758507, 140184.08901758507, 7220.8089999999975, 7220.8089999999975],
                      [None, 'high', lambda obj: obj.G(), 'low', lambda obj: obj.T]):
        obj = flasher.flash(V=0.0006354909990692889, P=359381366.3805, zs=[1], solution=sln)
        assert_allclose(obj.T, T)
    

    
    
def test_VF_H_cases():
    T, P, zs = 350.0, 1e5, [1.0]
    fluid_idx = 7 # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)
    
    eos = PRMIX
    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
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
        assert_allclose(low.T, base.T, rtol=1e-8)
        assert_allclose(low.H(), base.H(), rtol=1e-7)

        high = flasher.flash(H=base.H(), VF=VF, solution='high')
        assert_allclose(high.T, T_high, rtol=1e-8)
        assert_allclose(high.H(), base.H(), rtol=1e-7)

    
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
            assert_allclose(new.T, base.T, rtol=1e-8)
        

def test_VF_U_cases():
    T, P, zs = 350.0, 1e5, [1.0]
    fluid_idx = 7 # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)
    
    eos = PRMIX
    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    # Two solutions
    VFs_two =[0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9285628571428575, 0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9]

    Ts_two_low = [481.71428571428584, 507.6734693877551, 506.84085844724433, 337.15037336596805, 484.30741617882694, 509.784104700645, 509.1465329286005, 344.57339768096267]

    Ts_two_high = [496.74470283423096, 512.3840598446428, 512.4750615095567, 512.214210227972, 494.79103350924, 511.70674621843665, 512.0, 512.49999]
    
    

    for VF, T_low, T_high in zip(VFs_two, Ts_two_low, Ts_two_high):
        base = flasher.flash(T=T_low, VF=VF)
        low = flasher.flash(U=base.U(), VF=VF, solution='low')
        assert_allclose(low.T, base.T, rtol=1e-8)
        assert_allclose(low.U(), base.U(), rtol=1e-7)

        high = flasher.flash(U=base.U(), VF=VF, solution='high')
        assert_allclose(high.T, T_high, rtol=1e-8)
        assert_allclose(high.U(), base.U(), rtol=1e-7)

    
    solutions = ['mid', 'low', 'high']
    
    one_sln_Ts = [312.9795918367355, 320.0050872321796, 512.0, 486.0408163265307, 512.0, 200.0, 512.5
                  ]
    one_sln_VFs = [0.9285628571428575, 0.9, 0.5, 0.5714271428571429, 1e-05, 0.4827586206896553, 0.3]
    
    for T, VF in zip(one_sln_Ts, one_sln_VFs):
        base = flasher.flash(T=T, VF=VF)
        for s in solutions:
            new = flasher.flash(U=base.U(), VF=VF, solution=s)
            assert_allclose(new.T, base.T, rtol=1e-8)


def test_VF_A_cases():
    T, P, zs = 350.0, 1e5, [1.0]
    fluid_idx = 7 # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)
    
    eos = PRMIX
    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    # Two solutions
    VFs_two = [1e-05, 0.3]
    Ts_two_low = [505.1428075489603, 510.509170507811]
    Ts_two_high = [512.0, 512.49999]
    for VF, T_low, T_high in zip(VFs_two, Ts_two_low, Ts_two_high):
        base = flasher.flash(T=T_low, VF=VF)
        low = flasher.flash(A=base.A(), VF=VF, solution='low')
        assert_allclose(low.T, base.T, rtol=1e-8)
        assert_allclose(low.A(), base.A(), rtol=1e-7)

        high = flasher.flash(A=base.A(), VF=VF, solution='high')
        assert_allclose(high.T, T_high, rtol=1e-8)
        assert_allclose(high.A(), base.A(), rtol=1e-7)

    
    solutions = ['mid', 'low', 'high']
    one_sln_Ts =  [312.9795918367355, 481.71428571428584, 507.6734693877551, 506.84085844724433, 320.0050872321796, 512.214210227972, 484.30741617882694, 511.70674621843665, 512.0, 512.49999, 512.0, 486.0408163265307, 200.0]
    one_sln_VFs = [0.9285628571428575, 0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9, 0.9285628571428575, 0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9, 0.5, 0.5714271428571429, 0.4827586206896553]
    
    for T, VF in zip(one_sln_Ts, one_sln_VFs):
        base = flasher.flash(T=T, VF=VF)
        for s in solutions:
            new = flasher.flash(A=base.A(), VF=VF, solution=s)
            assert_allclose(new.T, base.T, rtol=1e-8)

def test_VF_G_cases():
    # No double solutions for G - almost no need to iterate
    T, P, zs = 350.0, 1e5, [1.0]
    fluid_idx = 7 # methanol
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)
    
    eos = PRMIX
    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    
    solutions = ['mid', 'low', 'high']
    one_sln_Ts =  [312.9795918367355, 481.71428571428584, 507.6734693877551, 506.84085844724433, 320.0050872321796, 512.214210227972, 484.30741617882694, 511.70674621843665, 512.0, 512.49999, 512.0, 486.0408163265307, 512.0, 200.0, 512.49999]
    one_sln_VFs = [0.9285628571428575, 0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9, 0.9285628571428575, 0.7857085714285716, 0.5714271428571429, 0.5714271428571429, 0.9, 0.5, 0.5714271428571429, 1e-05, 0.4827586206896553, 0.3]

    for T, VF in zip(one_sln_Ts, one_sln_VFs):
        base = flasher.flash(T=T, VF=VF)
        for s in solutions:
            new = flasher.flash(G=base.G(), VF=VF, solution=s)
            assert_allclose(new.T, base.T, rtol=1e-8)
                
            
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
    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    # Two solutions
    VFs_two = [0.5714271428571429, 0.4827586206896553]
    Ts_two_low = [212.16975817104446, 199.99999999981264]
    Ts_two_high = [486.04081632653146, 359.28026319594096]

    for VF, T_low, T_high in zip(VFs_two, Ts_two_low, Ts_two_high):
        base = flasher.flash(T=T_low, VF=VF)
        low = flasher.flash(S=base.S(), VF=VF, solution='low')
        assert_allclose(low.T, base.T, rtol=1e-5)
        assert_allclose(low.S(), base.S(), rtol=1e-7)

        high = flasher.flash(S=base.S(), VF=VF, solution='high')
        assert_allclose(high.T, T_high, rtol=1e-5)
        assert_allclose(high.S(), base.S(), rtol=1e-7)

    
    solutions = ['mid', 'low', 'high']
    one_sln_Ts =  [312.9795918367355, 481.71428571428584, 320.0050872321796, 512.214210227972, 484.30741617882694, 512.49999, 512.0, 512.0, 512.49999]
    one_sln_VFs = [0.9285628571428575, 0.7857085714285716, 0.9, 0.9285628571428575, 0.7857085714285716, 0.9, 0.5, 1e-05, 0.3]

    for T, VF in zip(one_sln_Ts, one_sln_VFs):
        base = flasher.flash(T=T, VF=VF)
        for s in solutions:
            new = flasher.flash(S=base.S(), VF=VF, solution=s)
            assert_allclose(new.T, base.T, rtol=1e-8)


    VFs_three = [0.5714271428571429, 0.5714271428571429, 0.5714271428571429, 0.5714271428571429]
    Ts_three_low = [208.4063161102236, 208.42865241364098, 209.04825869752273, 209.253671056617]
    Ts_three_mid = [507.6734693878061, 506.8408584472489, 500.99800548138063, 499.7044524953537]
    Ts_three_high = [508.178797993246, 508.9013433039185, 511.7067462184334, 512.0000000000291]

    for VF, T_low, T_mid, T_high in zip(VFs_three, Ts_three_low, Ts_three_mid, Ts_three_high):
        base = flasher.flash(T=T_low, VF=VF)
        low = flasher.flash(S=base.S(), VF=VF, solution='low')
        assert_allclose(low.T, base.T, rtol=1e-5)
        assert_allclose(low.S(), base.S(), rtol=1e-7)

        mid = flasher.flash(S=base.S(), VF=VF, solution='mid')
        assert_allclose(mid.T, T_mid, rtol=1e-5)
        assert_allclose(mid.S(), base.S(), rtol=1e-7)

        high = flasher.flash(S=base.S(), VF=VF, solution='high')
        assert_allclose(high.T, T_high, rtol=1e-5)
        assert_allclose(high.S(), base.S(), rtol=1e-7)
