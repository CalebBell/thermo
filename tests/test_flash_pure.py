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

from thermo.test_utils import *
import matplotlib.pyplot as plt

pure_surfaces_dir = os.path.join(thermo.thermo_dir, '..', 'surfaces', 'pure')

pure_fluids = ['water', 'methane', 'ethane', 'decane', 'ammonia', 'nitrogen', 'oxygen', 'methanol']

'''# Recreate the below with the following:
N = len(pure_fluids)
m = Mixture(pure_fluids, zs=[1/N]*N, T=298.15, P=1e5)
print(m.constants.make_str(delim=', \n', properties=('Tcs', 'Pcs', 'omegas', 'MWs', "CASs")))
correlations = m.properties()
print(correlations.as_best_fit(['HeatCapacityGases']))
'''
constants = ChemicalConstantsPackage(Tcs=[647.14, 190.56400000000002, 305.32, 611.7, 405.6, 126.2, 154.58, 512.5], 
            Pcs=[22048320.0, 4599000.0, 4872000.0, 2110000.0, 11277472.5, 3394387.5, 5042945.25, 8084000.0], 
            omegas=[0.344, 0.008, 0.098, 0.49, 0.25, 0.04, 0.021, 0.5589999999999999], 
            MWs=[18.01528, 16.04246, 30.06904, 142.28168, 17.03052, 28.0134, 31.9988, 32.04186], 
            CASs=['7732-18-5', '74-82-8', '74-84-0', '124-18-5', '7664-41-7', '7727-37-9', '7782-44-7', '67-56-1'])

correlations = PropertyCorrelationPackage(constants=constants, HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
        HeatCapacityGas(best_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [7.444966286051841e-23, 9.444106746563928e-20, -1.2490299714587002e-15, 2.6693560979905865e-12, -2.5695131746723413e-09, 1.2022442523089315e-06, -0.00021492132731007108, 0.016616385291696574, 32.84274656062226])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
        HeatCapacityGas(best_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924]))])


from thermo.eos_mix import eos_mix_list
#eos_mix_list = [PRMIX, PR78MIX, SRKMIX, VDWMIX, PRSVMIX, PRSV2MIX, APISRKMIX, TWUPRMIX, TWUSRKMIX, IGMIX]
#eos_mix_list = [TWUPRMIX, TWUSRKMIX] # issues
@pytest.mark.parametrize("auto_range", ['realistic', 'physical'])
@pytest.mark.parametrize("fluid_idx", constants.cmps)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_PV_plot(fluid_idx, eos, auto_range):
    '''
    Normally about 16% of the realistic plot overlaps with the physical. However,
    the realistic is the important one, so do not use fewer points for it.
    
    The realistic should be clean/clear!
    '''
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid = pure_fluids[fluid_idx]
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
#    try:
#        assert np.max(errs) < 1e-5
#    except:
#        print(np.max(errs))
        
    plot_fig.savefig(os.path.join(path, key + '.png'))
    # TODO log the max error to a file
    
    plt.close()


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
    
    PV = flasher.flash(P=P, V=base.V())
    assert_allclose(T, PV.T, rtol=1e-7)


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


@pytest.mark.parametrize("auto_range", ['realistic'])
@pytest.mark.parametrize("fluid_idx", constants.cmps)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_TV_plot(fluid_idx, eos, auto_range):
    '''
    Even with mpmath cubics just do not like to behave; skeptical it can be done
    '''
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid = pure_fluids[fluid_idx]
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    
    res = flasher.TPV_inputs(zs=zs, pts=100, spec0='T', spec1='P', check0='T', check1='V', prop0='P',
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


@pytest.mark.parametrize("auto_range", ['realistic'])
@pytest.mark.parametrize("fluid_idx", constants.cmps)
@pytest.mark.parametrize("eos", eos_mix_list)
def x(fluid_idx, eos, auto_range):
    '''
    The non-smooth region at ~.01 P causes lots of issues
    '''
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid = pure_fluids[fluid_idx]
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)

    liquid = EOSLiquid(eos, T=T, P=P, zs=zs, **kwargs)
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)

    flasher = FlashPureVLS(pure_const, pure_props, gas, [liquid], [])

    
    res = flasher.TPV_inputs(zs=zs, pts=100, spec0='T', spec1='P', check0='P', check1='S', prop0='T',
                           trunc_err_low=1e-10, 
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           auto_range=auto_range, 
                           show=False)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res
    
    path = os.path.join(pure_surfaces_dir, fluid, "PS")
    if not os.path.exists(path):
        os.makedirs(path)
    
    key = '%s - %s - %s - %s' %('PS', eos.__name__, auto_range, fluid)
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()



@pytest.mark.parametrize("fluid_idx", constants.cmps)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_V_G_min_plot(fluid_idx, eos):
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid = pure_fluids[fluid_idx]
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)
    
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    errs, plot_fig = gas.eos_mix.volumes_G_min(plot=True, show=False, pts=150,
                                               Tmin=1e-4, Tmax=1e4, Pmin=1e-2, Pmax=1e9)

    
    path = os.path.join(pure_surfaces_dir, fluid, "V_G_min")
    if not os.path.exists(path):
        os.makedirs(path)
    
    key = '%s - %s - %s' %('V_G_min', eos.__name__, fluid)
        
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()


@pytest.mark.parametrize("fluid_idx", constants.cmps)
@pytest.mark.parametrize("eos", eos_mix_list)
def test_V_error_plot(fluid_idx, eos):
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid = pure_fluids[fluid_idx]
    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])
    
    kwargs = dict(eos_kwargs=dict(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, omegas=pure_const.omegas),
                  HeatCapacityGases=pure_props.HeatCapacityGases)
    
    gas = EOSGas(eos, T=T, P=P, zs=zs, **kwargs)
    errs, plot_fig = gas.eos_mix.volume_errors(plot=True, show=False, pts=50,
                                               Tmin=1e-4, Tmax=1e4, Pmin=1e-2, Pmax=1e9,
                                               trunc_err_low=1e-15, color_map=cm_flash_tol())

    
    path = os.path.join(pure_surfaces_dir, fluid, "V_error")
    if not os.path.exists(path):
        os.makedirs(path)
    
    key = '%s - %s - %s' %('V_error', eos.__name__, fluid)
        
    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()