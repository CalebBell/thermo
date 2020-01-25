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
from fluids.numerics import *
from thermo.flash import *
from thermo.phases import *
from thermo.eos_mix import *
from thermo.eos import *
from thermo.vapor_pressure import VaporPressure
from thermo.heat_capacity import *
from thermo.phase_change import *

def test_minimize_gibbs_NP_transformed():
    T=298.15
    P = 101325.0
    omegas = [0.344, 0.008, 0.394]
    Tcs = [647.14, 190.564, 568.7]
    Pcs = [22048320.0, 4599000.0, 2490000.0]
    kijs=[[0,0, 0],[0,0, 0.0496], [0,0.0496,0]]
    zs = [1.0/3.0]*3
    N = len(zs)
    
    # Gas heat capacities are not strictly needed for EOS - but other models they are. 
    # Should not alter results here.
    HeatCapacityGases = [HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                          HeatCapacityGas(best_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                          HeatCapacityGas(best_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303]))]
    
    eos_kwargs = {'Pcs': Pcs, 'Tcs': Tcs, 'omegas': omegas, 'kijs': kijs}
    gas = EOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq0 = EOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq1 = EOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    
    betas = [ 1-0.5274853623254059, 0.5274853623254059-1e-3, 1e-3]
    
    compositions_guesses =[[0.3092021726552898, 0.0026640919467415093, 0.6881337353979681],
     [0.35494971827752636, 0.6295426084023353, 0.015507673320138696],
     [.998, .001, .001]]
    
    betas, compositions, phases, _, G = minimize_gibbs_NP_transformed(T, P, zs, compositions_guesses, phases=[liq0, gas, liq1],
                                betas=betas, tol=1E-13, method='BFGS')
    assert_allclose(G, -6288.484102530695, rtol=1e-3)
    assert_allclose(betas, [0.33353301800763124, 0.348168600825151, 0.31829838116721776], rtol=1e-3)
    assert_allclose(compositions,  [[0.01710942117103125, 0.004664963388767063, 0.9782256154402017],
      [0.026792761914958576, 0.9529216972153103, 0.0202855408697312],
      [0.9999999901300325, 0.0, 9.869967458145574e-09]], rtol=1e-3, atol=1e-7)
    
    
def test_UNIFAC_LLE_SS():
    from thermo.unifac import LLEUFIP, LLEUFSG, UNIFAC
    from thermo import VaporPressure, HeatCapacityGas, VolumeLiquid
    from thermo.phases import GibbsExcessLiquid
    from thermo.activity import Rachford_Rice_solution
    P = 1e5
    T = 298.15
    xs = [0.9, 0.1]
    chemgroups = [{17: 1}, {1: 1, 2: 3, 14: 1}]
    
    VaporPressures = [VaporPressure(best_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317])),
                      VaporPressure(best_fit=(183.85, 563.1, [-1.369963570009104e-19, 4.601627231730426e-16, -6.744785620228449e-13, 5.655784279629317e-10, -2.986123576859473e-07, 0.00010278182137225028, -0.022995143239892296, 3.186560947413634, -210.12716900412732]))]
    HeatCapacityGases = [HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                         HeatCapacityGas(best_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724]))]
    VolumeLiquids = [VolumeLiquid(best_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652])),
                     VolumeLiquid(best_fit=(183.85, 534.945, [8.166268567991483e-24, -2.165718271472294e-20, 2.4731798748970672e-17, -1.5862095449169107e-14, 6.243674899388041e-12, -1.5433275010768489e-09, 2.3391927454003685e-07, -1.9817325459693386e-05, 0.0007969650387898196]))]
    
    GE = UNIFAC.from_subgroups(T=T, xs=xs, chemgroups=chemgroups,
                               interaction_data=LLEUFIP, subgroups=LLEUFSG, version=0)
    
    liq = GibbsExcessLiquid(VaporPressures=VaporPressures, HeatCapacityGases=HeatCapacityGases,
                            VolumeLiquids=VolumeLiquids, GibbsExcessModel=GE, 
                            T=T, P=P, zs=xs)
    
    Ks = [.5, 10]
    VF, xs_guess, ys_guess = Rachford_Rice_solution(xs, Ks)
    
    VF, xs0, xs1, _, _, _, err = sequential_substitution_2P(T, P, None, xs, ys_guess, xs_guess, liq, liq, maxiter=200, V_over_F_guess=VF)
    assert abs(err) < 1e-10
    assert_allclose(VF, 0.8180880014378398)
    assert_allclose(xs0, [0.5336869025395473, 0.46631309746045285])
    assert_allclose(xs1,[0.9814542537494846, 0.018545746250515603])

def test_dew_bubble_newton_zs():
    T, P = 370.0, 6e5
    zs = [.3, .5, .2]
    eos_kwargs = {'Pcs': [22048320.0, 3025000.0, 4108000.0], 'Tcs': [647.14, 507.6, 591.75], 'omegas': [0.344, 0.2975, 0.257]}
    HeatCapacityGases = [HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                         HeatCapacityGas(best_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
                         HeatCapacityGas(best_fit=(50.0, 1000.0, [-9.48396765770823e-21, 4.444060985512694e-17, -8.628480671647472e-14, 8.883982004570444e-11, -5.0893293251198045e-08, 1.4947108372371731e-05, -0.0015271248410402886, 0.19186172941013854, 30.797883940134057]))]
    
    gas = EOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = EOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    
    # TVF-0
    TVF0 = dew_bubble_newton_zs(P, T, zs, liq, gas, 
                               iter_var='P', fixed_var='T', V_over_F=0, 
                               maxiter=200, xtol=1E-11, comp_guess=None, debug=True)
    (iter_val, comp, iter_phase, const_phase, niter, err), cb = TVF0
    
    TVF0_jac_expect = [[-1.7094866222319638, -0.060889186638232645, -0.06326258796693127, -2.623671448053193e-06], [0.04594194610061586, -2.6779244704304754, 0.09199738718637793, -2.3450702269756142e-06], [0.052225681359212524, 0.10065452377367312, -23.400784985388814, -2.3292014632919617e-06], [-1.0, -1.0, -1.0, 0.0]]
    jac_num = jacobian(lambda x: list(cb(x, jac=False)), comp + [iter_val], scalar=False, perturbation=1e-7)
    jac_analytical = [list(i) for i in cb(comp + [iter_val], jac=True)[1]]
    
    assert_allclose(TVF0_jac_expect, jac_analytical, rtol=1e-7)
    assert_allclose(jac_num, jac_analytical, rtol=1e-6)
    assert_allclose(comp, [0.5959851041217594, 0.3614714142727822, 0.04254348160545845], rtol=1e-6)
    assert_allclose(iter_val, 369706.09616182366, rtol=1e-8)
    
    # TVF-1
    TVF1 = dew_bubble_newton_zs(P, T, zs, liq, gas, 
                               iter_var='P', fixed_var='T', V_over_F=1, 
                               maxiter=200, xtol=1E-11, comp_guess=None, debug=True)
    (iter_val, comp, iter_phase, const_phase, niter, err), cb = TVF1
    jac_num = jacobian(lambda x: list(cb(x, jac=False)), comp + [iter_val], scalar=False, perturbation=1e-7)
    jac_analytical = [list(i) for i in cb(comp + [iter_val], jac=True)[1]]
    TVF1_jac_expect = [[-11.607060987245507, 0.17093888890495346, 0.7136499432808722, 5.053514880988398e-06], [-0.7226038994767894, -2.653184191819002, -0.48362418106386595, 4.725356979277343e-06], [0.38123614577345877, 0.07750480981046248, -1.648559971422293, 4.705268781400021e-06], [-1.0, -1.0, -1.0, 0.0]]
    assert_allclose(TVF1_jac_expect, jac_analytical, rtol=1e-7)
    assert_allclose(jac_num, jac_analytical, rtol=1e-6)
    assert_allclose(comp, [0.07863510496551862, 0.39728142156798496, 0.5240834734664964], rtol=1e-6)
    assert_allclose(iter_val, 196037.49251710708, rtol=1e-8)
    
    # PVF-1
    PVF1 = dew_bubble_newton_zs(T, P, zs, liq, gas, 
                               iter_var='T', fixed_var='P', V_over_F=1, 
                               maxiter=200, xtol=1E-11, debug=True)
    (iter_val, comp, iter_phase, const_phase, niter, err), cb = PVF1
    jac_num = jacobian(lambda x: list(cb(x, jac=False)), comp + [iter_val], scalar=False, perturbation=1e-7)
    jac_analytical = [list(i) for i in cb(comp + [iter_val], jac=True)[1]]
    PVF1_jac_expect = [[-8.981778361176932, 0.2533697774315433, 0.6428032458633708, -0.01567016574390102], [-0.6340304549551559, -2.4698089607476525, -0.5159121776039166, -0.016284877533440625], [0.4377008898727188, 0.16638569879213017, -1.8473188730053174, -0.021260114330616014], [-1.0, -1.0, -1.0, 0.0]]
    assert_allclose(PVF1_jac_expect, jac_analytical, rtol=1e-7)
    assert_allclose(jac_num, jac_analytical, rtol=1e-6)
    assert_allclose(comp, [0.10190680927242819, 0.44581304512199615, 0.45228014560557583])
    assert_allclose(iter_val, 414.5860479637154)

    # PVF-0
    PVF0 = dew_bubble_newton_zs(T, P, zs, liq, gas, 
                               iter_var='T', fixed_var='P', V_over_F=0, 
                               maxiter=200, xtol=1E-11, debug=True)
    (iter_val, comp, iter_phase, const_phase, niter, err), cb = PVF0
    PVF0_jac_expect = [[-1.7799148067485484, -0.09576578045148737, -0.1000043126222332, 0.019325866913386947], [0.06747154401695143, -2.5608953087110042, 0.13445334461342753, 0.018884399921460383], [0.0784219794200535, 0.14964231218727547, -19.81193477319855, 0.024952816338405084], [-1.0, -1.0, -1.0, 0.0]]
    jac_num = jacobian(lambda x: list(cb(x, jac=False)), comp + [iter_val], scalar=False, perturbation=1e-7)
    jac_analytical = [list(i) for i in cb(comp + [iter_val], jac=True)[1]]
    assert_allclose(jac_num, jac_analytical, rtol=1e-6)
    assert_allclose(PVF0_jac_expect, jac_analytical, rtol=1e-7)
    assert_allclose(comp, [0.5781248395738718, 0.3717955398333062, 0.05007962059282194])
    assert_allclose(iter_val, 390.91409227801205)
