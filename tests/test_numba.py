# -*- coding: utf-8 -*-
"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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
SOFTWARE.
"""

from __future__ import division
from thermo import *
import thermo
from math import *
from random import random
from fluids.constants import *
from fluids.numerics import assert_close, assert_close1d, assert_close2d, assert_close3d
from numpy.testing import assert_allclose
from chemicals import normalize, Rackett_fit, Antoine
from thermo.test_utils import check_np_output_activity

import pytest
try:
    import numba
    import numba.core
except:
    numba = None
import numpy as np

if numba is not None:
    import thermo.numba
    import chemicals.numba



def swap_funcs_and_test(names, substitutions, test):
    '''
    names : list[str]
        object names to switch out
    substitutions : list[obj]
        Objects to put in
    test : function
        Unit test to run in the file
    '''
    originals = {}
    glob = test.__globals__
    for name, sub in zip(names, substitutions):
        originals[name] = glob[name]
        glob[name] = sub
    try:
        test()
    except Exception as e:
        glob.update(originals)
        raise e
    glob.update(originals)

def mark_as_numba(func):
    func = pytest.mark.numba(func)
    func = pytest.mark.skipif(numba is None, reason="Numba is missing")(func)
    return func

@mark_as_numba
def test_EOSMIX_outputs_inputs_np():
    kwargs = dict(Tcs=[190.56400000000002, 305.32, 369.83, 126.2],
                  Pcs=[4599000.0, 4872000.0, 4248000.0, 3394387.5],
                  omegas=[0.008, 0.098, 0.152, 0.04],
                  zs=[.1, .2, .3, .4],
                  kijs=[[0.0, -0.0059, 0.0119, 0.0289], [-0.0059, 0.0, 0.0011, 0.0533], [0.0119, 0.0011, 0.0, 0.0878], [0.0289, 0.0533, 0.0878, 0.0]])
    kwargs_np = {k:np.array(v) for k, v in kwargs.items()}

    from thermo.numba import PRMIX as PRMIXNP, SRKMIX as SRKMIXNP
    plain = [PRMIX, SRKMIX]
    transformed = [PRMIXNP, SRKMIXNP]
    
    for p, t in zip(plain, transformed):

        eos = p(T=200, P=1e5, **kwargs)
        eos_np = t(T=200, P=1e5, **kwargs_np)
    
        base_vec_attrs = ['a_alphas', 'da_alpha_dTs', 'd2a_alpha_dT2s', 'a_alpha_roots', 'a_alpha_j_rows', 'da_alpha_dT_j_rows', 'lnphis_l', 'phis_l', 'fugacities_l', 'lnphis_g', 'phis_g', 'fugacities_g']
        extra_vec_attrs = ['db_dzs', 'db_dns', 'dnb_dns', 'd2b_dzizjs', 'd2b_dninjs', 'd3b_dzizjzks', 'd3b_dninjnks', 'd3epsilon_dzizjzks', 'da_alpha_dzs', 'da_alpha_dns', 'dna_alpha_dns', 'd2a_alpha_dzizjs']
        alpha_vec_attrs = ['_a_alpha_j_rows', '_da_alpha_dT_j_rows', 'a_alpha_ijs', 'da_alpha_dT_ijs', 'd2a_alpha_dT2_ijs']
        # TODO: _d2a_alpha_dT2_j_rows, and _a_alpha_j_rows', '_da_alpha_dT_j_rows with .to methods
    
        for attr in base_vec_attrs + extra_vec_attrs + alpha_vec_attrs:
            assert_close1d(getattr(eos, attr), getattr(eos_np, attr), rtol=1e-14)
            assert type(getattr(eos, attr)) is list
            assert type(getattr(eos_np, attr)) is np.ndarray

@mark_as_numba
def test_EOSMIX_numpy_input_cases():
    # Needs its own volume solution
    res = thermo.numba.eos_mix.IGMIX(T=300.0, P=1e5, zs=np.array([.7, .2, .1]), Tcs=np.array([126.2, 304.2, 373.2]), Pcs=np.array([3394387.5, 7376460. , 8936865. ]), omegas=np.array([0.04  , 0.2252, 0.1   ]))
    assert_close(res.V_g, 0.02494338785445972)


@mark_as_numba
def test_IdealSolution_np_out():
    from thermo import IdealSolution
    from thermo.numba import IdealSolution as IdealSolutionnp
    model = IdealSolution(T=300.0, xs=[.1, .2, .3, .4])
    modelnp = IdealSolutionnp(T=300.0, xs=np.array([.1, .2, .3, .4]))
    modelnp2 = modelnp.to_T_xs(T=310.0, xs=np.array([.2, .2, .2, .4]))

    check_np_output_activity(model, modelnp, modelnp2)


@mark_as_numba
def test_Wilson_numpy_output():
    T = 331.42
    N = 3

    from thermo.numba import Wilson as Wilsonnp
    A = [[0.0, 3.870101271243586, 0.07939943395502425],
                 [-6.491263271243587, 0.0, -3.276991837288562],
                 [0.8542855660449756, 6.906801837288562, 0.0]]
    B = [[0.0, -375.2835, -31.1208],
                 [1722.58, 0.0, 1140.79],
                 [-747.217, -3596.17, -0.0]]
    D = [[-0.0, -0.00791073, -0.000868371],
                 [0.00747788, -0.0, -3.1e-05],
                 [0.00124796, -3e-05, -0.0]]

    C = E = F = [[0.0]*N for _ in range(N)]

    xs = [0.229, 0.175, 0.596]

    model = thermo.wilson.Wilson(T=T, xs=xs, ABCDEF=(A, B, C, D, E, F))
    modelnp = Wilsonnp(T=T, xs=np.array(xs), ABCDEF=(np.array(A), np.array(B), np.array(C), np.array(D), np.array(E), np.array(F)))
    modelnp2 = modelnp.to_T_xs(T=T, xs=np.array(xs))

    check_np_output_activity(model, modelnp, modelnp2)

@mark_as_numba
def test_NRTL_numpy_output():
    NRTLnp = thermo.numba.nrtl.NRTL
    alphas = [[[0.0, 2e-05], [0.2937, 7e-05], [0.2999, 0.0001]],
     [[0.2937, 1e-05], [0.0, 4e-05], [0.3009, 8e-05]],
     [[0.2999, 1e-05], [0.3009, 3e-05], [0.0, 5e-05]]]

    taus = [[[6e-05, 0.0, 7e-05, 7e-05, 0.00788, 3.6e-07],
      [3e-05, 624.868, 9e-05, 7e-05, 0.00472, 8.5e-07],
      [3e-05, 398.953, 4e-05, 1e-05, 0.00279, 5.6e-07]],
     [[1e-05, -29.167, 8e-05, 9e-05, 0.00256, 1e-07],
      [2e-05, 0.0, 7e-05, 6e-05, 0.00587, 4.2e-07],
      [0.0, -35.482, 8e-05, 4e-05, 0.00889, 8.2e-07]],
     [[9e-05, -95.132, 6e-05, 1e-05, 0.00905, 5.2e-07],
      [9e-05, 33.862, 2e-05, 6e-05, 0.00517, 1.4e-07],
      [0.0001, 0.0, 6e-05, 2e-05, 0.00095, 7.4e-07]]]

    N = 3
    T = 273.15+70
    dT = T*1e-8
    xs = [.2, .3, .5]
    model = NRTL(T, xs, taus, alphas)
    modelnp = NRTLnp(T=T, xs=np.array(xs), tau_coeffs=np.array(taus), alpha_coeffs=np.array(alphas))
    modelnp2 = modelnp.to_T_xs(T=T, xs=np.array(xs))

    check_np_output_activity(model, modelnp, modelnp2)

@mark_as_numba
def test_UNIQUAC_numpy_output():
    UNIQUACnp = thermo.numba.uniquac.UNIQUAC

    N = 3
    T = 331.42
    xs = [0.229, 0.175, 0.596]
    rs = [2.5735, 2.87, 1.4311]
    qs = [2.336, 2.41, 1.432]
    # madeup numbers to match Wilson example roughly
    tausA = [[0.0, -1.05e-4, -2.5e-4], [3.9e-4, 0.0, 1.6e-4], [-1.123e-4, 6.5e-4, 0]]
    tausB = [[0.0, 235.0, -169.0], [-160, 0.0, -715.0], [11.2, 144.0, 0.0]]
    tausC = [[0.0, -4.23e-4, 2.9e-4], [6.1e-4, 0.0, 8.2e-5], [-7.8e-4, 1.11e-4, 0]]
    tausD = [[0.0, -3.94e-5, 2.22e-5], [8.5e-5, 0.0, 4.4e-5], [-7.9e-5, 3.22e-5, 0]]
    tausE = [[0.0, -4.2e2, 8.32e2], [2.7e2, 0.0, 6.8e2], [3.7e2, 7.43e2, 0]]
    tausF = [[0.0, 9.64e-8, 8.94e-8], [1.53e-7, 0.0, 1.11e-7], [7.9e-8, 2.276e-8, 0]]
    ABCDEF = (tausA, tausB, tausC, tausD, tausE, tausF)
    ABCDEFnp = tuple(np.array(v) for v in ABCDEF)

    model = UNIQUAC(T=T, xs=xs, rs=rs, qs=qs, ABCDEF=ABCDEF)
    modelnp = UNIQUACnp(T=T, xs=np.array(xs), rs=np.array(rs), qs=np.array(qs), ABCDEF=ABCDEFnp)
    modelnp2 = modelnp.to_T_xs(T=T, xs=np.array(xs))

    check_np_output_activity(model, modelnp, modelnp2)

@mark_as_numba
def test_UNIFAC_numpy_output():
    from thermo.unifac import DOUFIP2006, DOUFSG

    UNIFACnp = thermo.numba.unifac.UNIFAC
    N = 4
    T = 373.15
    xs = [0.2, 0.3, 0.1, 0.4]
    chemgroups = [{9:6}, {78:6}, {1:1, 18:1}, {1:1, 2:1, 14:1}]
    model = thermo.unifac.UNIFAC.from_subgroups(T=T, xs=xs, chemgroups=chemgroups, version=1,
                               interaction_data=DOUFIP2006, subgroups=DOUFSG)


    modelnp = UNIFACnp.from_subgroups(T=T, xs=np.array(xs), chemgroups=chemgroups, version=1,
                           interaction_data=DOUFIP2006, subgroups=DOUFSG)
    modelnp2 = modelnp.to_T_xs(T=T, xs=np.array(xs))

    check_np_output_activity(model, modelnp, modelnp2)

    json_string = modelnp.as_json()
    new = UNIFACnp.from_json(json_string)
    assert new == modelnp



@mark_as_numba
def test_a_alpha_aijs_composition_independent_in_all():
    assert 'a_alpha_aijs_composition_independent' in thermo.numba.__all__


@mark_as_numba
def test_a_alpha_aijs_composition_independent():
    # TODO: a_alpha_aijs_composition_independent is being overwritten in thermo.numba somehow!

    kijs = np.array([[0,.083],[0.083,0]])
    a_alphas = np.array([0.2491099357671155, 0.6486495863528039])
    a0, a1, a2 = thermo.numba.eos_mix_methods.a_alpha_aijs_composition_independent(a_alphas, kijs)
    assert type(a0) is np.ndarray
    assert type(a1) is np.ndarray
    assert type(a2) is np.ndarray

    b0, b1, b2 = thermo.eos_mix_methods.a_alpha_aijs_composition_independent(a_alphas, kijs)
    assert_close1d(a1, b1, rtol=1e-13)
    assert_close2d(a0, b0, rtol=1e-13)
    assert_close2d(a2, b2, rtol=1e-13)

    assert thermo.numba.eos_mix_methods.a_alpha_aijs_composition_independent is not thermo.eos_mix_methods.a_alpha_aijs_composition_independent


@mark_as_numba
def test_a_alpha_quadratic_terms_numba():
    T = 299.0
    kijs = np.array([[0,.083],[0.083,0]])
    zs = np.array([0.1164203, 0.8835797])
    a_alphas = np.array([0.2491099357671155, 0.6486495863528039])
    a_alpha_roots = np.sqrt(a_alphas)
    da_alpha_dTs = np.array([-0.0005102028006086241, -0.0011131153520304886])
    d2a_alpha_dT2s = np.array([1.8651128859234162e-06, 3.884331923127011e-06])
    a_alpha, a_alpha_j_rows = thermo.numba.a_alpha_quadratic_terms(a_alphas, a_alpha_roots, T, zs, kijs)
    a_alpha_expect, a_alpha_j_rows_expect = thermo.eos_mix_methods.a_alpha_quadratic_terms(a_alphas, a_alpha_roots, T, zs, kijs)
    assert_close(a_alpha, a_alpha_expect, rtol=1e-13)
    assert_close1d(a_alpha_j_rows, a_alpha_j_rows_expect, rtol=1e-13)
    
    a_alpha_j_rows = np.zeros(len(zs))
    vec0 = np.zeros(len(zs))
    a_alpha, a_alpha_j_rows = thermo.numba.a_alpha_quadratic_terms(a_alphas, a_alpha_roots, T, zs, kijs, a_alpha_j_rows, vec0)
    assert_close(a_alpha, a_alpha_expect, rtol=1e-13)
    assert_close1d(a_alpha_j_rows, a_alpha_j_rows_expect, rtol=1e-13)
    
    a_alpha, a_alpha_j_rows = thermo.numba.a_alpha_quadratic_terms(a_alphas, a_alpha_roots, T, zs, kijs, a_alpha_j_rows)
    assert_close(a_alpha, a_alpha_expect, rtol=1e-13)
    assert_close1d(a_alpha_j_rows, a_alpha_j_rows_expect, rtol=1e-13)

@mark_as_numba
def test_a_alpha_and_derivatives_full():
    kijs = np.array([[0,.083],[0.083,0]])
    zs = np.array([0.1164203, 0.8835797])
    a_alphas = np.array([0.2491099357671155, 0.6486495863528039])
    da_alpha_dTs = np.array([-0.0005102028006086241, -0.0011131153520304886])
    d2a_alpha_dT2s = np.array([1.8651128859234162e-06, 3.884331923127011e-06])
    a_alpha, da_alpha_dT, d2a_alpha_dT2, a_alpha_ijs, da_alpha_dT_ijs, d2a_alpha_dT2_ijs = thermo.numba.a_alpha_and_derivatives_full(a_alphas=a_alphas, da_alpha_dTs=da_alpha_dTs, d2a_alpha_dT2s=d2a_alpha_dT2s, T=299.0, zs=zs, kijs=kijs)

    a_alpha0, da_alpha_dT0, d2a_alpha_dT20, a_alpha_ijs0, da_alpha_dT_ijs0, d2a_alpha_dT2_ijs0 = thermo.eos_mix_methods.a_alpha_and_derivatives_full(a_alphas=a_alphas, da_alpha_dTs=da_alpha_dTs, d2a_alpha_dT2s=d2a_alpha_dT2s, T=299.0, zs=zs, kijs=kijs)


    assert_close(a_alpha, a_alpha0, rtol=1e-13)
    assert_close(da_alpha_dT, da_alpha_dT0, rtol=1e-13)
    assert_close(d2a_alpha_dT2, d2a_alpha_dT20, rtol=1e-13)

    assert_close1d(a_alpha_ijs, a_alpha_ijs0, rtol=1e-13)
    assert_close1d(da_alpha_dT_ijs, da_alpha_dT_ijs0, rtol=1e-13)
    assert_close1d(d2a_alpha_dT2_ijs0, d2a_alpha_dT2_ijs, rtol=1e-13)


@mark_as_numba
def test_IAPWS95_numba():
    assert isinstance(thermo.numba.flash.iapws95_Psat, numba.core.registry.CPUDispatcher)
    assert isinstance(thermo.numba.phases.IAPWS95._d3Ar_ddeltadtau2_func, numba.core.registry.CPUDispatcher)

    from thermo.numba import IAPWS95, IAPWS95Liquid, IAPWS95Gas, FlashPureVLS

    liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])

    assert_close(flasher.flash(T=1000,P=1e4).H(), 71901.67235666412, rtol=1e-8) # TP
    assert_close(flasher.flash(P=1e5, V=.1).T, 1202.8504507662728, rtol=1e-8) # PV
    assert_close(flasher.flash(T=1000, V=.1).P, 83126.1092778793, rtol=1e-8) # TV

    assert_close(flasher.flash(P=1e5,VF=1).T, 372.7559288971221, rtol=1e-8) # PVF
    assert_close(flasher.flash(T=300, VF=.5).P, 3536.806752274638, rtol=1e-8) # TVF
# assert_close(flasher.flash(P=1e4, H=71901.67235666412).T, 1000, rtol=1e-8) # PH - not working yet



@mark_as_numba
def test_RegularSolution_numba():
    N = 20
    xs = normalize([random() for _ in range(N)])
    xs2 = normalize([random() for _ in range(N)])
    SPs = [50000.0*random() for _ in range(N)]
    Vs = [1e-5*random() for _ in range(N)]


    T = 300.0
    lambda_coeffs = [[random()*1e-4 for _ in range(N)] for _ in range(N)]

    GE = RegularSolution(T, xs, Vs, SPs, lambda_coeffs)
    xsnp = np.array(xs)
    xs2np = np.array(xs2)
    Vsnp = np.array(Vs)
    SPsnp = np.array(SPs)
    lambda_coeffsnp = np.array(lambda_coeffs)

    GE = RegularSolution(T, xsnp, Vsnp, SPsnp, lambda_coeffsnp)
    GE.gammas()
    GE.to_T_xs(T=T+1.0, xs=xs2np).gammas()

    assert_close1d(GE.d2GE_dTdxs(), [0.0]*N, atol=0)

@mark_as_numba
def test_RegularSolution_gammas_numba():

    xs = [.4, .3, .2, .1]
    SPs = [19570.2, 18864.7, 29261.4, 47863.5]
    Vs = [7.421e-05, 8.068e-05, 4.083e-05, 1.808e-05]
    N = 4
    T = 300.0
    P = 1e5
    # Made up asymmetric parameters
    lambda_coeffs = [[0.0, 0.01811, 0.01736, 0.02111],
     [0.00662, 0.0, 0.00774, 0.01966],
     [0.01601, 0.01022, 0.0, 0.00698],
     [0.0152, 0.00544, 0.02579, 0.0]]
    
    GE = thermo.regular_solution.RegularSolution(T, xs, Vs, SPs, lambda_coeffs)
    assert_close1d(GE.gammas(), 
               thermo.numba.regular_solution.regular_solution_gammas(T=T, xs=np.array(xs), Vs=np.array(Vs), SPs=np.array(SPs), lambda_coeffs=np.array(lambda_coeffs), N=N), rtol=1e-12)




@mark_as_numba
def test_volume_numba_solvers():

    args = (0.0001, 0.01, 2.590839755349289e-05, 2.590839755349289e-05, 0.0, 348530.6151663297)
    slns = thermo.numba.eos_volume.volume_solutions_halley(*args)
    assert slns[1] == 0
    assert slns[2] == 0
    assert_close(slns[0], 2.5908397553496088e-05, rtol=1e-15)

    args = (0.0001, 154141458.17537114, 2.590839755349289e-05, 2.590839755349289e-05, 0.0, 348530.6151663297)
    slns = thermo.numba.eos_volume.volume_solutions_halley(*args)
    assert slns[1] == 0
    assert slns[2] == 0
    assert_close(slns[0], 2.5908397553496098e-05, rtol=1e-15)

from .test_eos_volume import hard_parameters, validate_volume
@pytest.mark.parametrize("params", hard_parameters)
@mark_as_numba
def test_hard_default_solver_volumes_numba(params):
    # Particularly tough cases
    validate_volume(params, thermo.numba.eos_volume.volume_solutions_halley, rtol=1e-14)    


from .test_flash_pure import test_V_error_plot, pure_fluids, eos_list
@pytest.mark.slow
@pytest.mark.plot
@pytest.mark.parametric
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("eos", eos_list)
@pytest.mark.parametrize("P_range", ['high', 'low'])
@mark_as_numba
def test_V_error_plot_numba(fluid, eos, P_range):
    return test_V_error_plot(fluid=fluid, eos=eos, P_range=P_range, solver=staticmethod(thermo.numba.eos_volume.volume_solutions_halley))

    

@mark_as_numba
def test_numba_dri_air():
    gas = thermo.numba.DryAirLemmon(T=300.0, P=1e5)
    flasher = thermo.numba.flash.FlashPureVLS(constants=thermo.lemmon2000_constants, correlations=thermo.lemmon2000_correlations,
                           gas=gas, liquids=[], solids=[])
    res = flasher.flash(H=4000, P=1e6)
    assert_close(res.T, 146.94641220863562)
    
@mark_as_numba
def test_lnphis_direct_works_at_all():
    zs = np.array([.5, .5])
    eos_kwargs = {'Pcs': np.array([4872000.0, 3370000.0]), 'Tcs': np.array([305.32, 469.7]), 
                  'omegas': np.array([0.098, 0.251])}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=None, T=300.0, P=1e5, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=None, T=300.0, P=1e5, zs=zs)
    assert_close1d(thermo.numba.lnphis_direct(zs, *gas.lnphis_args()), gas.lnphis())
    assert_close1d(thermo.numba.lnphis_direct(zs, *liq.lnphis_args()), liq.lnphis())
    
@mark_as_numba
def test_lnphis_direct_and_sequential_substitution_2P_functional():
    T, P = 300.0, 1.6e6
    constants = ChemicalConstantsPackage(Tcs=[305.32, 469.7], Pcs=[4872000.0, 3370000.0],
                                         omegas=[0.098, 0.251], Tms=[90.3, 143.15],
                                         Tbs=[184.55, 309.21], CASs=['74-84-0', '109-66-0'],
                                         names=['ethane', 'pentane'], MWs=[30.06904, 72.14878])
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                         HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866]))]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases)
    zs = np.array([.5, .5])
    
    eos_kwargs = {'Pcs': np.array(constants.Pcs), 'Tcs': np.array(constants.Tcs), 'omegas': np.array(constants.omegas)}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    
    flasher = FlashVL(constants, correlations, liquid=liq, gas=gas)
    res_expect = flasher.flash(T=T, P=P, zs=zs.tolist())
    VF_expect, xs_expect, ys_expect = res_expect.VF, res_expect.liquid0.zs, res_expect.gas.zs
    
    _, _, VF, xs, ys = chemicals.numba.flash_wilson(zs=zs, Tcs=eos_kwargs['Tcs'], Pcs=eos_kwargs['Pcs'], omegas=eos_kwargs['omegas'], T=T, P=P)
    VF_calc, xs_calc, ys_calc, niter, err = thermo.numba.sequential_substitution_2P_functional(zs=zs, xs_guess=xs, ys_guess=ys,
                                   liquid_args=liq.lnphis_args(), gas_args=gas.lnphis_args(),
                                          maxiter=1000, tol=1E-20,
                                       trivial_solution_tol=1e-5, V_over_F_guess=0.5)
    assert_close(VF_calc, VF_expect, rtol=1e-6)
    assert_close1d(xs_calc, xs_expect)
    assert_close1d(ys_calc, ys_expect)
    
    
@mark_as_numba
def test_fit_T_dep_numba():
    Tc, rhoc, b, n, MW = 545.03, 739.99, 0.3, 0.28571, 105.921
    Ts = np.linspace(331.15, 332.9, 10)
    props_calc = [Rackett_fit(T, Tc, rhoc, b, n, MW) for T in Ts]
    res, stats = TDependentProperty.fit_data_to_model(Ts=Ts, data=props_calc, model='Rackett_fit',
                          do_statistics=True, use_numba=True, model_kwargs={'MW':MW, 'Tc': Tc,},
                          fit_method='lm')
    assert stats['MAE'] < 1e-5

    # Numba was evaluating a 0 division when it shouldn't have, try excepted it
    Ts = np.linspace(845.15, 1043.1999999999998, 20)
    props_calc = [Antoine(T, A=1.9823770201329391, B=651.916, C=-1248.487) for T in Ts]
    res, stats = TDependentProperty.fit_data_to_model(Ts=Ts, data=props_calc, model='Antoine',
                          do_statistics=True, use_numba=True, model_kwargs={'base':10.0},
                          fit_method='lm')
    assert stats['MAE'] < 1e-5
