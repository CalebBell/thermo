# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from math import exp, log
from numpy.testing import assert_allclose
import pytest
import numpy as np
from thermo import normalize
from fluids.constants import calorie, R
from thermo.activity import *
from thermo.mixture import Mixture
from thermo.wilson import Wilson
import random
from thermo import *
import numpy as np
from fluids.numerics import jacobian, hessian, derivative



def test_DDBST_example():
    # One good numerical example 
    T = 331.42
    N = 3
    Vs_ddbst = [74.04, 80.67, 40.73]
    as_ddbst = [[0, 375.2835, 31.1208], [-1722.58, 0, -1140.79], [747.217, 3596.17, 0.0]]
    bs_ddbst = [[0, -3.78434, -0.67704], [6.405502, 0, 2.59359], [-0.256645, -6.2234, 0]]
    cs_ddbst = [[0.0, 7.91073e-3, 8.68371e-4], [-7.47788e-3, 0.0, 3.1e-5], [-1.24796e-3, 3e-5, 0.0]]
    
    dis = eis = fis = [[0.0]*N for _ in range(N)]
    
    params = Wilson.from_DDBST_as_matrix(Vs=Vs_ddbst, ais=as_ddbst, bis=bs_ddbst, 
                                cis=cs_ddbst, dis=dis, eis=eis, fis=fis, unit_conversion=False)
    
    A_expect = [[0.0, 3.870101271243586, 0.07939943395502425],
     [-6.491263271243587, 0.0, -3.276991837288562],
     [0.8542855660449756, 6.906801837288562, 0.0]]
    B_expect = [[0.0, -375.2835, -31.1208],
     [1722.58, 0.0, 1140.79],
     [-747.217, -3596.17, -0.0]]
    D_expect = [[-0.0, -0.00791073, -0.000868371],
     [0.00747788, -0.0, -3.1e-05],
     [0.00124796, -3e-05, -0.0]]
    
    C_expect = E_expect = F_expect = [[0.0]*N for _ in range(N)]
    
    assert_allclose(params[0], A_expect, rtol=1e-12, atol=0)
    assert_allclose(params[1], B_expect, rtol=1e-12, atol=0)
    assert_allclose(params[2], C_expect, rtol=1e-12, atol=0)
    assert_allclose(params[3], D_expect, rtol=1e-12, atol=0)
    assert_allclose(params[4], E_expect, rtol=1e-12, atol=0)
    assert_allclose(params[5], F_expect, rtol=1e-12, atol=0)
    
    xs = [0.229, 0.175, 0.596]
    
    GE = Wilson(T=T, xs=xs, ABCDEF=params)
    
    assert_allclose(GE.gammas(), [1.223393433488855, 1.1009459024701462, 1.2052899281172034], rtol=1e-12)
    
    
    lambdas = GE.lambdas()
    lambdas_expect = [[1.0, 1.1229699812593041, 0.7391181616283594],
     [3.2694762162029805, 1.0, 1.1674967844769508],
     [0.37280197780931773, 0.019179096486191153, 1.0]]
    assert_allclose(lambdas, lambdas_expect, rtol=1e-12)
    
    dlambdas_dT = GE.dlambdas_dT()
    dlambdas_dT_expect = [[0.0, -0.005046703220379676, -0.0004324140595259853],
     [-0.026825598419319092, 0.0, -0.012161812924715213],
     [0.003001348681882189, 0.0006273541924400231, 0.0]]
    assert_allclose(dlambdas_dT, dlambdas_dT_expect)
    
    dT = T*1e-8
    dlambdas_dT_numerical = (np.array(GE.to_T_xs(T+dT, xs).lambdas()) - GE.to_T_xs(T, xs).lambdas())/dT
    assert_allclose(dlambdas_dT, dlambdas_dT_numerical, rtol=1e-7)
    
    
    d2lambdas_dT2 = GE.d2lambdas_dT2()
    d2lambdas_dT2_expect = [[0.0, -4.73530781420922e-07, -1.0107624477842068e-06],
     [0.000529522489227112, 0.0, 0.0001998633344112975],
     [8.85872572550323e-06, 1.6731622007033546e-05, 0.0]]
    assert_allclose(d2lambdas_dT2, d2lambdas_dT2_expect, rtol=1e-12)
    
    d2lambdas_dT2_numerical = (np.array(GE.to_T_xs(T+dT, xs).dlambdas_dT()) - GE.to_T_xs(T, xs).dlambdas_dT())/dT
    assert_allclose(d2lambdas_dT2, d2lambdas_dT2_numerical, rtol=2e-5)

    d3lambdas_dT3 = GE.d3lambdas_dT3()
    d3lambdas_dT3_expect = [[0.0, 4.1982403087995867e-07, 1.3509359183777608e-08],
     [-1.2223067176509094e-05, 0.0, -4.268843384910971e-06],
     [-3.6571009680721684e-08, 3.3369718709496133e-07, 0.0]]
    assert_allclose(d3lambdas_dT3, d3lambdas_dT3_expect, rtol=1e-12)
    
    d3lambdas_dT3_numerical = (np.array(GE.to_T_xs(T+dT, xs).d2lambdas_dT2()) - GE.to_T_xs(T, xs).d2lambdas_dT2())/dT
    assert_allclose(d3lambdas_dT3, d3lambdas_dT3_numerical, rtol=1e-7)

