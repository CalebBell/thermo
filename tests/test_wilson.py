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
    
    gammas_expect = [1.223393433488855, 1.1009459024701462, 1.2052899281172034]
    assert_allclose(GE.gammas(), gammas_expect, rtol=1e-12)
    assert_allclose(GibbsExcess.gammas(GE), gammas_expect)
    
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

    # Gammas
    assert_allclose(GE.GE(), 480.2639266306882, rtol=1e-12)
    gammas = GE.gammas()
    GE_from_gammas = R*T*sum(xi*log(gamma) for xi, gamma in zip(xs, gammas))
    assert_allclose(GE_from_gammas, GE.GE(), rtol=1e-12)
    
    # dGE dT
    dGE_dT_numerical = ((np.array(GE.to_T_xs(T+dT, xs).GE()) - np.array(GE.GE()))/dT) 
    dGE_dT_analytical = GE.dGE_dT()
    assert_allclose(dGE_dT_analytical, 4.355962766232997, rtol=1e-12)
    assert_allclose(dGE_dT_numerical, dGE_dT_analytical)
    
    # d2GE dT2
    d2GE_dT2_numerical = ((np.array(GE.to_T_xs(T+dT, xs).dGE_dT()) - np.array(GE.dGE_dT()))/dT) 
    d2GE_dT2_analytical = GE.d2GE_dT2()
    assert_allclose(d2GE_dT2_analytical, -0.02913038452501723, rtol=1e-12)
    assert_allclose(d2GE_dT2_analytical, d2GE_dT2_numerical, rtol=1e-8)
    
    # d3GE dT3
    d3GE_dT3_numerical = ((np.array(GE.to_T_xs(T+dT, xs).d2GE_dT2()) - np.array(GE.d2GE_dT2()))/dT) 
    d3GE_dT3_analytical = GE.d3GE_dT3()
    assert_allclose(d3GE_dT3_analytical, -0.00019988744724590656, rtol=1e-12)
    assert_allclose(d3GE_dT3_numerical, d3GE_dT3_analytical, rtol=1e-7)
    
    # d2GE_dTdxs
    def dGE_dT_diff(xs):
        return GE.to_T_xs(T, xs).dGE_dT()
    
    d2GE_dTdxs_numerical = jacobian(dGE_dT_diff, xs, perturbation=1e-7)
    d2GE_dTdxs_analytical = GE.d2GE_dTdxs()
    d2GE_dTdxs_expect = [-10.187806161151178, 13.956324059647034, -6.825249918548414]
    assert_allclose(d2GE_dTdxs_analytical, d2GE_dTdxs_expect, rtol=1e-12)
    assert_allclose(d2GE_dTdxs_numerical, d2GE_dTdxs_analytical, rtol=1e-7)
    
    # dGE_dxs
    def dGE_dx_diff(xs):
        return GE.to_T_xs(T, xs).GE()
    
    dGE_dxs_numerical = jacobian(dGE_dx_diff, xs, perturbation=1e-7)
    dGE_dxs_analytical = GE.dGE_dxs()
    dGE_dxs_expect = [-2199.97589893946, -2490.5759162306463, -2241.05706053718]
    assert_allclose(dGE_dxs_analytical, dGE_dxs_expect, rtol=1e-12)
    assert_allclose(dGE_dxs_analytical, dGE_dxs_numerical, rtol=1e-7)
    
    # d2GE_dxixjs
    d2GE_dxixjs_numerical = hessian(dGE_dx_diff, xs, perturbation=1e-5)
    d2GE_dxixjs_analytical = GE.d2GE_dxixjs()
    d2GE_dxixjs_expect = [[-3070.205333938506, -7565.029777297412, -1222.5200812237945],
     [-7565.029777297412, -2156.7810946064815, -1083.4743126696396],
     [-1222.5200812237945, -1083.4743126696396, -3835.5941234746824]]
    assert_allclose(d2GE_dxixjs_analytical, d2GE_dxixjs_expect, rtol=1e-12)
    assert_allclose(d2GE_dxixjs_analytical, d2GE_dxixjs_numerical, rtol=1e-4)

    # d3GE_dxixjxks - very limited accuracy.
    def d2GE_dxixj_diff(xs):
        return GE.to_T_xs(T, xs).dGE_dxs()
    
    d3GE_dxixjxks_numerical = hessian(d2GE_dxixj_diff, xs, perturbation=2e-5, scalar=False)
    
    d3GE_dxixjxks_analytical = GE.d3GE_dxixjxks()
    d3GE_dxixjxks_expect = [[[614.0681113650099, 14845.66663517824, 556.3625424156468],
      [14845.66663517824, 8308.935636377626, 4549.175136703878],
      [556.3625424156469, 4549.175136703878, 501.6902853815983]],
     [[14845.66663517824, 8308.935636377626, 4549.175136703878],
      [8308.935636377626, 173.08338078843053, 375.4114802651511],
      [4549.175136703877, 375.411480265151, -40.24127966770044]],
     [[556.3625424156469, 4549.175136703877, 501.6902853815977],
      [4549.175136703877, 375.411480265151, -40.241279667700574],
      [501.6902853815964, -40.24127966770071, 6254.612872590844]]]
    assert_allclose(d3GE_dxixjxks_analytical, d3GE_dxixjxks_expect, rtol=1e-12)
    assert_allclose(d3GE_dxixjxks_numerical, d3GE_dxixjxks_analytical, rtol=1e-3)
    
    
    ### TEST WHICH ARE COMMON TO ALL GibbsExcess classes
    HE_expected = -963.3892533542517
    HE_analytical = GE.HE()
    assert_allclose(HE_expected, HE_analytical, rtol=1e-12)
    def diff_for_HE(T):
        return GE.to_T_xs(T, xs).GE()/T
    
    HE_numerical = -derivative(diff_for_HE, T, order=13)*T**2
    assert_allclose(HE_analytical, HE_numerical, rtol=1e-12)
    
    
    SE_expected = -4.355962766232997
    SE_analytical = GE.SE()
    assert_allclose(SE_expected, SE_analytical, rtol=1e-12)
    SE_check = (GE.HE() - GE.GE())/T
    assert_allclose(SE_analytical, SE_check, rtol=1e-12)
    
    
    def diff_for_Cp(T):
        return GE.to_T_xs(T, xs).HE()
    Cp_expected = 9.65439203928121
    Cp_analytical = GE.CpE()
    assert_allclose(Cp_expected, Cp_analytical, rtol=1e-12)
    Cp_numerical = derivative(diff_for_Cp, T, order=13)
    assert_allclose(Cp_numerical, Cp_analytical, rtol=1e-12)

    
    def diff_for_dS_dT(T):
        return GE.to_T_xs(T, xs).SE()
    dS_dT_expected = 0.02913038452501723
    dS_dT_analytical = GE.dSE_dT()
    assert_allclose(dS_dT_expected, dS_dT_analytical, rtol=1e-12)
    dS_dT_numerical = derivative(diff_for_dS_dT, T, order=9)
    assert_allclose(dS_dT_analytical, dS_dT_numerical, rtol=1e-12)


    def diff_for_dHE_dx(xs):
        return GE.to_T_xs(T, xs).HE()
    
    dHE_dx_expected = [1176.4668189892636, -7115.980836078867, 20.96726746813556]
    dHE_dx_analytical = GE.dHE_dxs()
    assert_allclose(dHE_dx_expected, dHE_dx_analytical, rtol=1e-12)
    dHE_dx_numerical = jacobian(diff_for_dHE_dx, xs, perturbation=5e-7)
    assert_allclose(dHE_dx_expected, dHE_dx_numerical, rtol=4e-6)


    def diff_for_dHE_dn(xs):
        xs = normalize(xs)
        return GE.to_T_xs(T, xs).HE()
    
    dHE_dn_expected = [2139.856072343515, -6152.591582724615, 984.3565208223869]
    dHE_dn_analytical = GE.dHE_dns()
    assert_allclose(dHE_dn_expected, dHE_dn_analytical, rtol=1e-12)
    
    dHE_dn_numerical = jacobian(diff_for_dHE_dn, xs, perturbation=5e-7)
    assert_allclose(dHE_dn_expected, dHE_dn_numerical, rtol=1e-6)

    
    def diff_for_dnHE_dn(xs):
        nt = sum(xs)
        xs = normalize(xs)
        return nt*GE.to_T_xs(T, xs).HE()
    
    dnHE_dn_expected = [1176.4668189892634, -7115.980836078867, 20.967267468135258]
    dnHE_dn_analytical = GE.dnHE_dns()
    assert_allclose(dnHE_dn_expected, dnHE_dn_analytical, rtol=1e-12)
    
    dnHE_dn_numerical = jacobian(diff_for_dnHE_dn, xs, perturbation=5e-7)
    assert_allclose(dnHE_dn_analytical, dnHE_dn_numerical, rtol=2e-6)

    
    def diff_for_dSE_dx(xs):
        return GE.to_T_xs(T, xs).SE()
    
    dSE_dx_expected = [10.187806161151178, -13.956324059647036, 6.825249918548415]
    dSE_dx_analytical = GE.dSE_dxs()
    assert_allclose(dSE_dx_expected, dSE_dx_analytical, rtol=1e-12)
    dSE_dx_numerical = jacobian(diff_for_dSE_dx, xs, perturbation=5e-7)
    assert_allclose(dSE_dx_expected, dSE_dx_numerical, rtol=4e-6)
    
    
    def diff_for_dSE_dns(xs):
        xs = normalize(xs)
        return GE.to_T_xs(T, xs).SE()
    
    dSE_dns_expected = [6.2293063092309335, -17.91482391156728, 2.8667500666281707]
    dSE_dns_analytical = GE.dSE_dns()
    assert_allclose(dSE_dns_expected, dSE_dns_analytical, rtol=1e-12)
    
    dSE_dns_numerical = jacobian(diff_for_dSE_dns, xs, perturbation=5e-7)
    assert_allclose(dSE_dns_expected, dSE_dns_numerical, rtol=1e-6)
    
    
    def diff_for_dnSE_dn(xs):
        nt = sum(xs)
        xs = normalize(xs)
        return nt*GE.to_T_xs(T, xs).SE()
    
    dnSE_dn_expected = [1.8733435429979384, -22.270786677800274, -1.489212699604825]
    dnSE_dn_analytical = GE.dnSE_dns()
    assert_allclose(dnSE_dn_expected, dnSE_dn_analytical, rtol=1e-12)
    
    dnSE_dn_numerical = jacobian(diff_for_dnSE_dn, xs, perturbation=5e-7)
    assert_allclose(dnSE_dn_analytical, dnSE_dn_numerical, rtol=2e-6)
    
    
    def diff_for_dGE_dn(xs):
        xs = normalize(xs)
        return GE.to_T_xs(T, xs).GE()
    
    dGE_dn_expected = [75.3393753381988, -215.2606419529875, 34.25821374047882]
    dGE_dn_analytical = GE.dGE_dns()
    assert_allclose(dGE_dn_expected, dGE_dn_analytical, rtol=1e-12)
    
    dGE_dn_numerical = jacobian(diff_for_dGE_dn, xs, perturbation=5e-7)
    assert_allclose(dGE_dn_expected, dGE_dn_numerical, rtol=1e-5)
    
    def diff_for_dnGE_dn(xs):
        nt = sum(xs)
        xs = normalize(xs)
        return nt*GE.to_T_xs(T, xs).GE()
    
    dnGE_dn_expected = [555.6033019688871, 265.0032846777008, 514.5221403711671]
    dnGE_dn_analytical = GE.dnGE_dns()
    assert_allclose(dnGE_dn_expected, dnGE_dn_analytical, rtol=1e-12)
    
    dnGE_dn_numerical = jacobian(diff_for_dnGE_dn, xs, perturbation=5e-7)
    assert_allclose(dnGE_dn_analytical, dnGE_dn_numerical, rtol=2e-6)    
    


        
    lambdas = GE.lambdas()
    def gammas_to_diff(xs):
        xs = normalize(xs)
        return np.array(Wilson_gammas(xs, lambdas))
    
    dgammas_dns_analytical = GE.dgammas_dns()
    dgammas_dn_numerical = jacobian(gammas_to_diff, xs, scalar=False)
    dgammas_dn_expect =  [[-0.13968444275751782, -2.135249914756224, 0.6806316652245148],
      [-1.9215360979146614, 0.23923983797040177, 0.668061736204089],
      [0.6705598284218852, 0.7313784266789759, -0.47239836472723573]]
    
    assert_allclose(dgammas_dns_analytical, dgammas_dn_numerical, rtol=1e-5)
    assert_allclose(dgammas_dns_analytical, dgammas_dn_expect, rtol=1e-11)
    
    '''# Using numdifftools, the result was confirmed to the four last decimal places (rtol=12-13).
    from numdifftools import Jacobian
    (Jacobian(gammas_to_diff, step=1e-6, order=37)(xs)/dgammas_dns_analytical).tolist()
    '''
        
    dgammas_dT_numerical = ((np.array(GE.to_T_xs(T+dT, xs).gammas()) - np.array(GE.gammas()))/dT) 
    dgammas_dT_analytical = GE.dgammas_dT()
    dgammas_dT_expect = [-0.001575992756074107, 0.008578456201039092, -2.7672076632932624e-05]
    assert_allclose(dgammas_dT_analytical, dgammas_dT_expect, rtol=1e-12)
    assert_allclose(dgammas_dT_numerical, dgammas_dT_analytical, rtol=2e-6)
    
    
    d2GE_dTdns_expect = [-6.229306309230934, 17.91482391156728, -2.8667500666281702]
    d2GE_dTdns_analytical = GE.d2GE_dTdns()
    d2GE_dTdns_numerical = ((np.array(GE.to_T_xs(T+dT, xs).dGE_dns()) - np.array(GE.dGE_dns()))/dT) 
    assert_allclose(d2GE_dTdns_expect, d2GE_dTdns_analytical, rtol=1e-12)
    assert_allclose(d2GE_dTdns_analytical, d2GE_dTdns_numerical, rtol=1e-7)

    
    d2nGE_dTdns_expect = [-1.8733435429979375, 22.270786677800274, 1.4892126996048267]
    d2nGE_dTdns_analytical = GE.d2nGE_dTdns()
    d2nGE_dTdns_numerical = ((np.array(GE.to_T_xs(T+dT, xs).dnGE_dns()) - np.array(GE.dnGE_dns()))/dT) 
    assert_allclose(d2nGE_dTdns_expect, d2nGE_dTdns_analytical, rtol=1e-12)
    assert_allclose(d2nGE_dTdns_analytical, d2nGE_dTdns_numerical, rtol=1e-6)
    
    
    def to_diff_dnGE2_dninj(ns):
        nt = sum(ns)
        xs = normalize(ns)
        return nt*GE.to_T_xs(T, xs).GE()
    d2nGE_dninjs_numerical = hessian(to_diff_dnGE2_dninj, xs, perturbation=4e-5)
    d2nGE_dninjs_analytical = GE.d2nGE_dninjs()
    d2nGE_dninjs_expect = [[-314.62613303015996, -4809.450576389065, 1533.0591196845521],
     [-4809.450576389066, 598.7981063018656, 1672.104888238707],
     [1533.0591196845517, 1672.1048882387074, -1080.0149225663358]]
    
    assert_allclose(d2nGE_dninjs_analytical, d2nGE_dninjs_expect, rtol=1e-12)
    assert_allclose(d2nGE_dninjs_numerical, d2nGE_dninjs_analytical, rtol=1e-4)