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

from math import log, exp
import pytest
from fluids.constants import R


from thermo.activity import GibbsExcess
from thermo import *
import numpy as np
from fluids.numerics import jacobian, hessian, derivative, normalize, assert_close, assert_close1d, assert_close2d, assert_close3d
from thermo.test_utils import check_np_output_activity
import pickle

def test_Wilson():
    # P05.01a VLE Behavior of Ethanol - Water Using Wilson
    # http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/P05.01a%20VLE%20Behavior%20of%20Ethanol%20-%20Water%20Using%20Wilson.xps
    gammas = Wilson_gammas([0.252, 0.748], [[1, 0.154], [0.888, 1]])
    assert_close1d(gammas, [1.8814926087178843, 1.1655774931125487])

    # Test the general form against the simpler binary form
    def Wilson2(molefracs, lambdas):
        x1 = molefracs[0]
        x2 = molefracs[1]
        l12 = lambdas[0]
        l21 = lambdas[1]
        gamma1 = exp(-log(x1+x2*l12) + x2*(l12/(x1+x2*l12) - l21/(x2+x1*l21)))
        gamma2 = exp(-log(x2+x1*l21) - x1*(l12/(x1+x2*l12) - l21/(x2+x1*l21)))
        return [gamma1, gamma2]
    gammas = Wilson2([0.252, 0.748], [0.154, 0.888])

    assert_close1d(gammas, [1.8814926087178843, 1.1655774931125487])

    # Test 3 parameter version:
    # 05.09 Compare Experimental VLE to Wilson Equation Results
    # http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/05.09%20Compare%20Experimental%20VLE%20to%20Wilson%20Equation%20Results.xps
    # Extra decimals obtained via the actual MathCad worksheet
    xs = [0.229, 0.175, 0.596]
    params = [[1, 1.1229699812593, 0.73911816162836],
              [3.26947621620298, 1, 1.16749678447695],
              [0.37280197780932, 0.01917909648619, 1]]
    gammas = Wilson_gammas(xs, params)
    assert_close1d(gammas, [1.22339343348885, 1.10094590247015, 1.2052899281172])

    # Test the values which produce gamma = 1
    gammas = Wilson_gammas([0.252, 0.748], [[1, 1], [1, 1]])
    assert_close1d(gammas, [1, 1])

def test_DDBST_example():
    # One good numerical example - acetone, chloroform, methanol
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

    assert_close2d(params[0], A_expect, rtol=1e-12, atol=0)
    assert_close2d(params[1], B_expect, rtol=1e-12, atol=0)
    assert_close2d(params[2], C_expect, rtol=1e-12, atol=0)
    assert_close2d(params[3], D_expect, rtol=1e-12, atol=0)
    assert_close2d(params[4], E_expect, rtol=1e-12, atol=0)
    assert_close2d(params[5], F_expect, rtol=1e-12, atol=0)

    xs = [0.229, 0.175, 0.596]

    GE = Wilson(T=T, xs=xs, ABCDEF=params)

    # Test __repr__ contains the needed information
    assert eval(str(GE)).GE() == GE.GE()

    GE2 = Wilson.from_json(GE.as_json())
    assert GE2.__dict__ == GE.__dict__

    gammas_expect = [1.223393433488855, 1.1009459024701462, 1.2052899281172034]
    assert_close1d(GE.gammas(), gammas_expect, rtol=1e-12)
    assert_close1d(GibbsExcess.gammas(GE), gammas_expect)

    lambdas = GE.lambdas()
    lambdas_expect = [[1.0, 1.1229699812593041, 0.7391181616283594],
                     [3.2694762162029805, 1.0, 1.1674967844769508],
                     [0.37280197780931773, 0.019179096486191153, 1.0]]
    assert_close2d(lambdas, lambdas_expect, rtol=1e-12)

    dlambdas_dT = GE.dlambdas_dT()
    dlambdas_dT_expect = [[0.0, -0.005046703220379676, -0.0004324140595259853],
                         [-0.026825598419319092, 0.0, -0.012161812924715213],
                         [0.003001348681882189, 0.0006273541924400231, 0.0]]
    assert_close2d(dlambdas_dT, dlambdas_dT_expect)

    dT = T*1e-8
    dlambdas_dT_numerical = (np.array(GE.to_T_xs(T+dT, xs).lambdas()) - GE.to_T_xs(T, xs).lambdas())/dT
    assert_close2d(dlambdas_dT, dlambdas_dT_numerical, rtol=1e-7)


    d2lambdas_dT2 = GE.d2lambdas_dT2()
    d2lambdas_dT2_expect = [[0.0, -4.73530781420922e-07, -1.0107624477842068e-06],
                             [0.000529522489227112, 0.0, 0.0001998633344112975],
                             [8.85872572550323e-06, 1.6731622007033546e-05, 0.0]]
    assert_close2d(d2lambdas_dT2, d2lambdas_dT2_expect, rtol=1e-12)

    d2lambdas_dT2_numerical = (np.array(GE.to_T_xs(T+dT, xs).dlambdas_dT()) - GE.to_T_xs(T, xs).dlambdas_dT())/dT
    assert_close2d(d2lambdas_dT2, d2lambdas_dT2_numerical, rtol=2e-5)

    d3lambdas_dT3 = GE.d3lambdas_dT3()
    d3lambdas_dT3_expect = [[0.0, 4.1982403087995867e-07, 1.3509359183777608e-08],
                             [-1.2223067176509094e-05, 0.0, -4.268843384910971e-06],
                             [-3.6571009680721684e-08, 3.3369718709496133e-07, 0.0]]
    assert_close2d(d3lambdas_dT3, d3lambdas_dT3_expect, rtol=1e-12)

    d3lambdas_dT3_numerical = (np.array(GE.to_T_xs(T+dT, xs).d2lambdas_dT2()) - GE.to_T_xs(T, xs).d2lambdas_dT2())/dT
    assert_close2d(d3lambdas_dT3, d3lambdas_dT3_numerical, rtol=1e-7)

    # Gammas
    assert_close(GE.GE(), 480.2639266306882, rtol=1e-12)
    gammas = GE.gammas()
    GE_from_gammas = R*T*sum(xi*log(gamma) for xi, gamma in zip(xs, gammas))
    assert_close(GE_from_gammas, GE.GE(), rtol=1e-12)

    # dGE dT
    dGE_dT_numerical = ((np.array(GE.to_T_xs(T+dT, xs).GE()) - np.array(GE.GE()))/dT)
    dGE_dT_analytical = GE.dGE_dT()
    assert_close(dGE_dT_analytical, 4.355962766232997, rtol=1e-12)
    assert_close(dGE_dT_numerical, dGE_dT_analytical)

    # d2GE dT2
    d2GE_dT2_numerical = ((np.array(GE.to_T_xs(T+dT, xs).dGE_dT()) - np.array(GE.dGE_dT()))/dT)
    d2GE_dT2_analytical = GE.d2GE_dT2()
    assert_close(d2GE_dT2_analytical, -0.02913038452501723, rtol=1e-12)
    assert_close(d2GE_dT2_analytical, d2GE_dT2_numerical, rtol=1e-8)

    # d3GE dT3
    d3GE_dT3_numerical = ((np.array(GE.to_T_xs(T+dT, xs).d2GE_dT2()) - np.array(GE.d2GE_dT2()))/dT)
    d3GE_dT3_analytical = GE.d3GE_dT3()
    assert_close(d3GE_dT3_analytical, -0.00019988744724590656, rtol=1e-12)
    assert_close(d3GE_dT3_numerical, d3GE_dT3_analytical, rtol=1e-7)

    # d2GE_dTdxs
    def dGE_dT_diff(xs):
        return GE.to_T_xs(T, xs).dGE_dT()

    d2GE_dTdxs_numerical = jacobian(dGE_dT_diff, xs, perturbation=1e-7)
    d2GE_dTdxs_analytical = GE.d2GE_dTdxs()
    d2GE_dTdxs_expect = [-10.187806161151178, 13.956324059647034, -6.825249918548414]
    assert_close1d(d2GE_dTdxs_analytical, d2GE_dTdxs_expect, rtol=1e-12)
    assert_close1d(d2GE_dTdxs_numerical, d2GE_dTdxs_analytical, rtol=1e-7)

    # dGE_dxs
    def dGE_dx_diff(xs):
        return GE.to_T_xs(T, xs).GE()

    dGE_dxs_numerical = jacobian(dGE_dx_diff, xs, perturbation=1e-7)
    dGE_dxs_analytical = GE.dGE_dxs()
    dGE_dxs_expect = [-2199.97589893946, -2490.5759162306463, -2241.05706053718]
    assert_close1d(dGE_dxs_analytical, dGE_dxs_expect, rtol=1e-12)
    assert_close1d(dGE_dxs_analytical, dGE_dxs_numerical, rtol=1e-7)

    # d2GE_dxixjs
    d2GE_dxixjs_numerical = hessian(dGE_dx_diff, xs, perturbation=1e-5)
    d2GE_dxixjs_analytical = GE.d2GE_dxixjs()
    d2GE_dxixjs_expect = [[-3070.205333938506, -7565.029777297412, -1222.5200812237945],
     [-7565.029777297412, -2156.7810946064815, -1083.4743126696396],
     [-1222.5200812237945, -1083.4743126696396, -3835.5941234746824]]
    assert_close2d(d2GE_dxixjs_analytical, d2GE_dxixjs_expect, rtol=1e-12)
    assert_close2d(d2GE_dxixjs_analytical, d2GE_dxixjs_numerical, rtol=1e-4)

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
    assert_close3d(d3GE_dxixjxks_analytical, d3GE_dxixjxks_expect, rtol=1e-12)
    assert_close3d(d3GE_dxixjxks_numerical, d3GE_dxixjxks_analytical, rtol=1e-3)


    ### TEST WHICH ARE COMMON TO ALL GibbsExcess classes
    HE_expected = -963.3892533542517
    HE_analytical = GE.HE()
    assert_close(HE_expected, HE_analytical, rtol=1e-12)
    def diff_for_HE(T):
        return GE.to_T_xs(T, xs).GE()/T

    HE_numerical = -derivative(diff_for_HE, T, order=13)*T**2
    assert_close(HE_analytical, HE_numerical, rtol=1e-12)


    SE_expected = -4.355962766232997
    SE_analytical = GE.SE()
    assert_close(SE_expected, SE_analytical, rtol=1e-12)
    SE_check = (GE.HE() - GE.GE())/T
    assert_close(SE_analytical, SE_check, rtol=1e-12)


    def diff_for_Cp(T):
        return GE.to_T_xs(T, xs).HE()
    Cp_expected = 9.65439203928121
    Cp_analytical = GE.CpE()
    assert_close(Cp_expected, Cp_analytical, rtol=1e-12)
    Cp_numerical = derivative(diff_for_Cp, T, order=13)
    assert_close(Cp_numerical, Cp_analytical, rtol=1e-12)


    def diff_for_dS_dT(T):
        return GE.to_T_xs(T, xs).SE()
    dS_dT_expected = 0.02913038452501723
    dS_dT_analytical = GE.dSE_dT()
    assert_close(dS_dT_expected, dS_dT_analytical, rtol=1e-12)
    dS_dT_numerical = derivative(diff_for_dS_dT, T, order=9)
    assert_close(dS_dT_analytical, dS_dT_numerical, rtol=1e-12)


    def diff_for_dHE_dx(xs):
        return GE.to_T_xs(T, xs).HE()

    dHE_dx_expected = [1176.4668189892636, -7115.980836078867, 20.96726746813556]
    dHE_dx_analytical = GE.dHE_dxs()
    assert_close1d(dHE_dx_expected, dHE_dx_analytical, rtol=1e-12)
    dHE_dx_numerical = jacobian(diff_for_dHE_dx, xs, perturbation=5e-7)
    assert_close1d(dHE_dx_expected, dHE_dx_numerical, rtol=4e-6)


    def diff_for_dHE_dn(xs):
        xs = normalize(xs)
        return GE.to_T_xs(T, xs).HE()

    dHE_dn_expected = [2139.856072343515, -6152.591582724615, 984.3565208223869]
    dHE_dn_analytical = GE.dHE_dns()
    assert_close1d(dHE_dn_expected, dHE_dn_analytical, rtol=1e-12)

    dHE_dn_numerical = jacobian(diff_for_dHE_dn, xs, perturbation=5e-7)
    assert_close1d(dHE_dn_expected, dHE_dn_numerical, rtol=1e-6)


    def diff_for_dnHE_dn(xs):
        nt = sum(xs)
        xs = normalize(xs)
        return nt*GE.to_T_xs(T, xs).HE()

    dnHE_dn_expected = [1176.4668189892634, -7115.980836078867, 20.967267468135258]
    dnHE_dn_analytical = GE.dnHE_dns()
    assert_close1d(dnHE_dn_expected, dnHE_dn_analytical, rtol=1e-12)

    dnHE_dn_numerical = jacobian(diff_for_dnHE_dn, xs, perturbation=5e-7)
    assert_close1d(dnHE_dn_analytical, dnHE_dn_numerical, rtol=2e-6)


    def diff_for_dSE_dx(xs):
        return GE.to_T_xs(T, xs).SE()

    dSE_dx_expected = [10.187806161151178, -13.956324059647036, 6.825249918548415]
    dSE_dx_analytical = GE.dSE_dxs()
    assert_close1d(dSE_dx_expected, dSE_dx_analytical, rtol=1e-12)
    dSE_dx_numerical = jacobian(diff_for_dSE_dx, xs, perturbation=5e-7)
    assert_close1d(dSE_dx_expected, dSE_dx_numerical, rtol=4e-6)


    def diff_for_dSE_dns(xs):
        xs = normalize(xs)
        return GE.to_T_xs(T, xs).SE()

    dSE_dns_expected = [6.2293063092309335, -17.91482391156728, 2.8667500666281707]
    dSE_dns_analytical = GE.dSE_dns()
    assert_close1d(dSE_dns_expected, dSE_dns_analytical, rtol=1e-12)

    dSE_dns_numerical = jacobian(diff_for_dSE_dns, xs, perturbation=5e-7)
    assert_close1d(dSE_dns_expected, dSE_dns_numerical, rtol=1e-6)


    def diff_for_dnSE_dn(xs):
        nt = sum(xs)
        xs = normalize(xs)
        return nt*GE.to_T_xs(T, xs).SE()

    dnSE_dn_expected = [1.8733435429979384, -22.270786677800274, -1.489212699604825]
    dnSE_dn_analytical = GE.dnSE_dns()
    assert_close1d(dnSE_dn_expected, dnSE_dn_analytical, rtol=1e-12)

    dnSE_dn_numerical = jacobian(diff_for_dnSE_dn, xs, perturbation=5e-7)
    assert_close1d(dnSE_dn_analytical, dnSE_dn_numerical, rtol=2e-6)


    def diff_for_dGE_dn(xs):
        xs = normalize(xs)
        return GE.to_T_xs(T, xs).GE()

    dGE_dn_expected = [75.3393753381988, -215.2606419529875, 34.25821374047882]
    dGE_dn_analytical = GE.dGE_dns()
    assert_close1d(dGE_dn_expected, dGE_dn_analytical, rtol=1e-12)

    dGE_dn_numerical = jacobian(diff_for_dGE_dn, xs, perturbation=5e-7)
    assert_close1d(dGE_dn_expected, dGE_dn_numerical, rtol=1e-5)

    def diff_for_dnGE_dn(xs):
        nt = sum(xs)
        xs = normalize(xs)
        return nt*GE.to_T_xs(T, xs).GE()

    dnGE_dn_expected = [555.6033019688871, 265.0032846777008, 514.5221403711671]
    dnGE_dn_analytical = GE.dnGE_dns()
    assert_close1d(dnGE_dn_expected, dnGE_dn_analytical, rtol=1e-12)

    dnGE_dn_numerical = jacobian(diff_for_dnGE_dn, xs, perturbation=5e-7)
    assert_close1d(dnGE_dn_analytical, dnGE_dn_numerical, rtol=2e-6)




    lambdas = GE.lambdas()
    def gammas_to_diff(xs):
        xs = normalize(xs)
        return np.array(Wilson_gammas(xs, lambdas))

    dgammas_dns_analytical = GE.dgammas_dns()
    dgammas_dn_numerical = jacobian(gammas_to_diff, xs, scalar=False)
    dgammas_dn_expect =  [[-0.13968444275751782, -2.135249914756224, 0.6806316652245148],
      [-1.9215360979146614, 0.23923983797040177, 0.668061736204089],
      [0.6705598284218852, 0.7313784266789759, -0.47239836472723573]]

    assert_close2d(dgammas_dns_analytical, dgammas_dn_numerical, rtol=1e-5)
    assert_close2d(dgammas_dns_analytical, dgammas_dn_expect, rtol=1e-11)

    '''# Using numdifftools, the result was confirmed to the four last decimal places (rtol=12-13).
    from numdifftools import Jacobian
    (Jacobian(gammas_to_diff, step=1e-6, order=37)(xs)/dgammas_dns_analytical).tolist()
    '''

    dgammas_dT_numerical = ((np.array(GE.to_T_xs(T+dT, xs).gammas()) - np.array(GE.gammas()))/dT)
    dgammas_dT_analytical = GE.dgammas_dT()
    dgammas_dT_expect = [-0.001575992756074107, 0.008578456201039092, -2.7672076632932624e-05]
    assert_close1d(dgammas_dT_analytical, dgammas_dT_expect, rtol=1e-12)
    assert_close1d(dgammas_dT_numerical, dgammas_dT_analytical, rtol=2e-6)


    d2GE_dTdns_expect = [-6.229306309230934, 17.91482391156728, -2.8667500666281702]
    d2GE_dTdns_analytical = GE.d2GE_dTdns()
    d2GE_dTdns_numerical = ((np.array(GE.to_T_xs(T+dT, xs).dGE_dns()) - np.array(GE.dGE_dns()))/dT)
    assert_close1d(d2GE_dTdns_expect, d2GE_dTdns_analytical, rtol=1e-12)
    assert_close1d(d2GE_dTdns_analytical, d2GE_dTdns_numerical, rtol=1e-7)


    d2nGE_dTdns_expect = [-1.8733435429979375, 22.270786677800274, 1.4892126996048267]
    d2nGE_dTdns_analytical = GE.d2nGE_dTdns()
    d2nGE_dTdns_numerical = ((np.array(GE.to_T_xs(T+dT, xs).dnGE_dns()) - np.array(GE.dnGE_dns()))/dT)
    assert_close1d(d2nGE_dTdns_expect, d2nGE_dTdns_analytical, rtol=1e-12)
    assert_close1d(d2nGE_dTdns_analytical, d2nGE_dTdns_numerical, rtol=1e-6)


    def to_diff_dnGE2_dninj(ns):
        nt = sum(ns)
        xs = normalize(ns)
        return nt*GE.to_T_xs(T, xs).GE()
    d2nGE_dninjs_numerical = hessian(to_diff_dnGE2_dninj, xs, perturbation=4e-5)
    d2nGE_dninjs_analytical = GE.d2nGE_dninjs()
    d2nGE_dninjs_expect = [[-314.62613303015996, -4809.450576389065, 1533.0591196845521],
     [-4809.450576389066, 598.7981063018656, 1672.104888238707],
     [1533.0591196845517, 1672.1048882387074, -1080.0149225663358]]

    assert_close2d(d2nGE_dninjs_analytical, d2nGE_dninjs_expect, rtol=1e-12)
    assert_close2d(d2nGE_dninjs_numerical, d2nGE_dninjs_analytical, rtol=1e-4)

    # Test with some results stored
    GE2 = Wilson.from_json(GE.as_json())
    assert GE2.__dict__ == GE.__dict__

def test_multicomnent_madeup():
    T=273.15+70
    xs = [1/7.0]*7
#    m = Mixture(['water', 'ethanol', 'methanol', '1-pentanol', '2-pentanol', '3-pentanol',
#                 '1-decanol'],
#                 P=1e5, zs=xs, T=T)

    # Main coefficients with temperature inverse dependency
    lambdasB = [[0.0, -35.3, 40.0, -139.0, -129.0, -128.0, -242.0],
     [-557.0, 0.0, -200.0, 83.2, 84.6, 80.2, 140.0],
     [-280.0, 95.5, 0.0, 88.2, 85.3, 89.1, 119.0],
     [-1260.0, -128.0, -220.0, 0.0, -94.4, -85.5, 59.7],
     [-1280.0, -121.0, -236.0, 80.3, 0.0, -88.8, 61.4],
     [-1370.0, -121.0, -238.0, 75.7, 78.2, 0.0, 63.1],
     [-2670.0, -304.0, -403.0, -93.4, -91.1, -86.6, 0.0]]

    # Add in some random noise for numerical stuff
    lambdasA = [[0.0092, 0.00976, 0.00915, 0.00918, 0.00974, 0.00925, 0.00908],
     [0.00954, 0.00927, 0.00902, 0.00948, 0.00934, 0.009, 0.00995],
     [0.00955, 0.00921, 0.0098, 0.00926, 0.00952, 0.00912, 0.00995],
     [0.00924, 0.00938, 0.00941, 0.00992, 0.00935, 0.00996, 0.0092],
     [0.00992, 0.00946, 0.00935, 0.00917, 0.00998, 0.00903, 0.00924],
     [0.00937, 0.00973, 0.00924, 0.00991, 0.00997, 0.00968, 0.00975],
     [0.00983, 0.00934, 0.00921, 0.00977, 0.00944, 0.00902, 0.00916]]

    lambdasC = [[0.000956, 0.000958, 0.000993, 0.000949, 0.000913, 0.000947, 0.000949],
     [0.000945, 0.000928, 0.000935, 0.000999, 0.000986, 0.000959, 0.000924],
     [0.000957, 0.000935, 0.00097, 0.000906, 0.00098, 0.000952, 0.000939],
     [0.000956, 0.000948, 0.0009, 0.000903, 0.000967, 0.000972, 0.000969],
     [0.000917, 0.000949, 0.000973, 0.000922, 0.000978, 0.000944, 0.000905],
     [0.000947, 0.000996, 0.000961, 0.00091, 0.00096, 0.000982, 0.000998],
     [0.000934, 0.000929, 0.000955, 0.000975, 0.000924, 0.000979, 0.001]]

    lambdasD = [[3.78e-05, 3.86e-05, 3.62e-05, 3.83e-05, 3.95e-05, 3.94e-05, 3.92e-05],
     [3.88e-05, 3.88e-05, 3.75e-05, 3.82e-05, 3.8e-05, 3.76e-05, 3.71e-05],
     [3.93e-05, 3.67e-05, 4e-05, 4e-05, 3.67e-05, 3.72e-05, 3.82e-05],
     [3.95e-05, 3.67e-05, 3.64e-05, 3.62e-05, 3.62e-05, 3.63e-05, 3.97e-05],
     [3.83e-05, 3.68e-05, 3.73e-05, 3.78e-05, 3.9e-05, 3.79e-05, 3.94e-05],
     [3.67e-05, 3.82e-05, 3.76e-05, 3.61e-05, 3.67e-05, 3.88e-05, 3.64e-05],
     [3.7e-05, 3.7e-05, 3.82e-05, 3.91e-05, 3.73e-05, 3.93e-05, 3.89e-05]]

    lambdasE = [[493.0, 474.0, 481.0, 468.0, 467.0, 474.0, 460.0],
     [478.0, 454.0, 460.0, 488.0, 469.0, 479.0, 483.0],
     [469.0, 493.0, 470.0, 476.0, 466.0, 451.0, 478.0],
     [481.0, 470.0, 467.0, 455.0, 473.0, 465.0, 465.0],
     [470.0, 487.0, 472.0, 460.0, 467.0, 468.0, 500.0],
     [480.0, 464.0, 475.0, 469.0, 462.0, 476.0, 469.0],
     [492.0, 460.0, 458.0, 494.0, 465.0, 461.0, 496.0]]

    lambdasF = [[8.25e-08, 8.27e-08, 8.78e-08, 8.41e-08, 8.4e-08, 8.93e-08, 8.98e-08],
     [8.28e-08, 8.35e-08, 8.7e-08, 8.96e-08, 8.15e-08, 8.46e-08, 8.53e-08],
     [8.51e-08, 8.65e-08, 8.24e-08, 8.89e-08, 8.86e-08, 8.71e-08, 8.21e-08],
     [8.75e-08, 8.89e-08, 8.6e-08, 8.42e-08, 8.83e-08, 8.52e-08, 8.53e-08],
     [8.24e-08, 8.27e-08, 8.43e-08, 8.19e-08, 8.74e-08, 8.3e-08, 8.35e-08],
     [8.79e-08, 8.84e-08, 8.31e-08, 8.15e-08, 8.68e-08, 8.55e-08, 8.2e-08],
     [8.63e-08, 8.76e-08, 8.52e-08, 8.46e-08, 8.67e-08, 8.9e-08, 8.38e-08]]


    GE = Wilson(T=T, xs=xs, ABCDEF=(lambdasA, lambdasB, lambdasC, lambdasD, lambdasE, lambdasF))


    dT = T*4e-8
    dlambdas_dT_numerical = ((np.array(GE.to_T_xs(T=T+dT, xs=xs).lambdas()) - np.array(GE.lambdas()))/dT)
    dlambdas_dT_analytical = GE.dlambdas_dT()
    dlambdas_dT_expect = [[7.590031904561817e-05, 0.00035248336206849656, -0.0003094799547918734, 0.0008734144675894445, 0.0008398355729456691, 0.0008388484248830437, 0.0011010388136697634],
                          [0.0009887057582300238, 7.958864330150294e-05, 0.0010333738209314847, -0.0008357318798642386, -0.0008603154287867674, -0.000798471407040265, -0.0017482651322616297],
                          [0.0011329768794643754, -0.0010143841833093821, 7.944935494096504e-05, -0.0009028095210965592, -0.0008655087807908184, -0.0009179733398931945, -0.0013825890320730938],
                          [0.00028607514753594, 0.0008361255545973125, 0.001066623152184149, 7.722830239720729e-05, 0.0006952398032090327, 0.0006509200113968528, -0.0005326090692151773],
                          [0.00027384494917044615, 0.0008057486827319439, 0.001089620949611226, -0.000800537779412656, 8.220344880607542e-05, 0.0006666235375135453, -0.0005573837108687875],
                          [0.00022537157737850808, 0.0008117920594539006, 0.0010915605157597842, -0.0007424034499493812, -0.0007702925324798186, 8.014036862597122e-05, -0.0005807961723267018],
                          [9.912129903248681e-06, 0.001142970658771038, 0.00112743688620555, 0.0006908346953600135, 0.0006797401776554298, 0.000661487700579681, 7.80163901524144e-05]]
    assert_close2d(dlambdas_dT_analytical, dlambdas_dT_expect, rtol=1e-13)
    assert_close2d(GE.dlambdas_dT(), dlambdas_dT_numerical, rtol=4e-7)


    d2lambdas_dT2_expect = [[3.9148862097941074e-07, -1.1715575536585707e-06, 2.841056708836699e-06, -3.4348594220311783e-06, -3.3305413029034426e-06, -3.30503660953987e-06, -3.6315706582334585e-06],
                            [-8.471983417828227e-07, 3.7679068485264176e-07, -3.7137380864269715e-06, 6.512351301871582e-06, 6.618281668596983e-06, 6.196655250267148e-06, 1.3401585927323754e-05],
                            [-3.445457199491784e-06, 7.777520320436478e-06, 3.8176318059273734e-07, 7.003390983275993e-06, 6.702713977066079e-06, 7.0719354763776695e-06, 1.0522581265135932e-05],
                            [1.4383544948858643e-06, -3.3086241806901887e-06, -3.7030893425862076e-06, 3.784586533299158e-07, -2.794563476850137e-06, -2.623796179953121e-06, 4.348980938675828e-06],
                            [1.4208967586173933e-06, -3.2319878656258e-06, -3.6664691274893067e-06, 6.187296499210411e-06, 3.913709343027832e-07, -2.6938291442926234e-06, 4.508630881673789e-06],
                            [1.341457154534956e-06, -3.2254094007592747e-06, -3.6611570052888056e-06, 5.757845600966917e-06, 6.003019989380841e-06, 3.909269296593175e-07, 4.632022286150103e-06],
                            [1.6807740689825766e-07, -3.272186668663736e-06, -2.3655959455488764e-06, -2.778154773804601e-06, -2.7342339993722627e-06, -2.6363752589934457e-06, 3.957722151854727e-07]]

    d2lambdas_dT2_analytical = GE.d2lambdas_dT2()
    d2lambdas_dT2_numerical = ((np.array(GE.to_T_xs(T=T+dT, xs=xs).dlambdas_dT()) - np.array(GE.dlambdas_dT()))/dT)
    assert_close2d(d2lambdas_dT2_analytical, d2lambdas_dT2_expect, rtol=1e-13)
    assert_close2d(d2lambdas_dT2_numerical, d2lambdas_dT2_analytical, rtol=1e-7)

    d3lambdas_dT3_expect = [[-2.458523153557734e-09, 1.075750588231061e-08, -2.5272393065722066e-08, 2.4517104581274395e-08, 2.431790858429151e-08, 2.427274202635229e-08, 1.9490204463227542e-08],
                            [-8.757802083086752e-09, -2.2542994182376097e-09, 2.274425631180586e-08, -6.272451551020672e-08, -6.401893152558163e-08, -5.9505339021928453e-08, -1.3924656024261e-07],
                            [1.580721350387629e-08, -7.635453830620122e-08, -2.338277642369151e-09, -6.793661773544476e-08, -6.473586412797368e-08, -6.868136383836727e-08, -1.0662459524200453e-07],
                            [-5.556692877523925e-09, 2.4287244013271755e-08, 2.1341219536976808e-08, -2.262235043655526e-09, 2.2067296896708128e-08, 2.1072746826311852e-08, -4.033020732041565e-08],
                            [-5.1501733514971684e-09, 2.3982486346807567e-08, 2.0021555356784153e-08, -5.943967388445255e-08, -2.3179928186947205e-09, 2.144234022725535e-08, -4.2054010358506494e-08],
                            [-3.323181667271306e-09, 2.404402632258137e-08, 1.9839516484420424e-08, -5.4961048857581214e-08, -5.7383559610513795e-08, -2.366287903394162e-09, -4.330885396751575e-08],
                            [1.7135848473209763e-09, 1.331477284270524e-08, 3.0148603800370024e-09, 2.1913464581918025e-08, 2.1725189171286703e-08, 2.120416180853968e-08, -2.4707966753243437e-09]]

    d3lambdas_dT3_analytical = GE.d3lambdas_dT3()
    d3lambdas_dT3_numerical = ((np.array(GE.to_T_xs(T=T+dT, xs=xs).d2lambdas_dT2()) - np.array(GE.d2lambdas_dT2()))/dT)
    assert_close2d(d3lambdas_dT3_analytical, d3lambdas_dT3_expect, rtol=1e-13)
    assert_close2d(d3lambdas_dT3_numerical, d3lambdas_dT3_analytical, rtol=2e-7)


@pytest.mark.slow
@pytest.mark.sympy
def test_multicomponent_madeup_sympy():
    from sympy import log, exp, diff, symbols
    A, B, C, D, E, F, T = symbols('A, B, C, D, E, F, T')

    N = 7
    T_num = 273.15+70


    lambdasB = [[0.0, -35.3, 40.0, -139.0, -129.0, -128.0, -242.0],
     [-557.0, 0.0, -200.0, 83.2, 84.6, 80.2, 140.0],
     [-280.0, 95.5, 0.0, 88.2, 85.3, 89.1, 119.0],
     [-1260.0, -128.0, -220.0, 0.0, -94.4, -85.5, 59.7],
     [-1280.0, -121.0, -236.0, 80.3, 0.0, -88.8, 61.4],
     [-1370.0, -121.0, -238.0, 75.7, 78.2, 0.0, 63.1],
     [-2670.0, -304.0, -403.0, -93.4, -91.1, -86.6, 0.0]]

    # Add in some random noise for numerical stuff
    lambdasA = [[0.0092, 0.00976, 0.00915, 0.00918, 0.00974, 0.00925, 0.00908],
     [0.00954, 0.00927, 0.00902, 0.00948, 0.00934, 0.009, 0.00995],
     [0.00955, 0.00921, 0.0098, 0.00926, 0.00952, 0.00912, 0.00995],
     [0.00924, 0.00938, 0.00941, 0.00992, 0.00935, 0.00996, 0.0092],
     [0.00992, 0.00946, 0.00935, 0.00917, 0.00998, 0.00903, 0.00924],
     [0.00937, 0.00973, 0.00924, 0.00991, 0.00997, 0.00968, 0.00975],
     [0.00983, 0.00934, 0.00921, 0.00977, 0.00944, 0.00902, 0.00916]]

    lambdasC = [[0.000956, 0.000958, 0.000993, 0.000949, 0.000913, 0.000947, 0.000949],
     [0.000945, 0.000928, 0.000935, 0.000999, 0.000986, 0.000959, 0.000924],
     [0.000957, 0.000935, 0.00097, 0.000906, 0.00098, 0.000952, 0.000939],
     [0.000956, 0.000948, 0.0009, 0.000903, 0.000967, 0.000972, 0.000969],
     [0.000917, 0.000949, 0.000973, 0.000922, 0.000978, 0.000944, 0.000905],
     [0.000947, 0.000996, 0.000961, 0.00091, 0.00096, 0.000982, 0.000998],
     [0.000934, 0.000929, 0.000955, 0.000975, 0.000924, 0.000979, 0.001]]

    lambdasD = [[3.78e-05, 3.86e-05, 3.62e-05, 3.83e-05, 3.95e-05, 3.94e-05, 3.92e-05],
     [3.88e-05, 3.88e-05, 3.75e-05, 3.82e-05, 3.8e-05, 3.76e-05, 3.71e-05],
     [3.93e-05, 3.67e-05, 4e-05, 4e-05, 3.67e-05, 3.72e-05, 3.82e-05],
     [3.95e-05, 3.67e-05, 3.64e-05, 3.62e-05, 3.62e-05, 3.63e-05, 3.97e-05],
     [3.83e-05, 3.68e-05, 3.73e-05, 3.78e-05, 3.9e-05, 3.79e-05, 3.94e-05],
     [3.67e-05, 3.82e-05, 3.76e-05, 3.61e-05, 3.67e-05, 3.88e-05, 3.64e-05],
     [3.7e-05, 3.7e-05, 3.82e-05, 3.91e-05, 3.73e-05, 3.93e-05, 3.89e-05]]

    lambdasE = [[493.0, 474.0, 481.0, 468.0, 467.0, 474.0, 460.0],
     [478.0, 454.0, 460.0, 488.0, 469.0, 479.0, 483.0],
     [469.0, 493.0, 470.0, 476.0, 466.0, 451.0, 478.0],
     [481.0, 470.0, 467.0, 455.0, 473.0, 465.0, 465.0],
     [470.0, 487.0, 472.0, 460.0, 467.0, 468.0, 500.0],
     [480.0, 464.0, 475.0, 469.0, 462.0, 476.0, 469.0],
     [492.0, 460.0, 458.0, 494.0, 465.0, 461.0, 496.0]]

    lambdasF = [[8.25e-08, 8.27e-08, 8.78e-08, 8.41e-08, 8.4e-08, 8.93e-08, 8.98e-08],
     [8.28e-08, 8.35e-08, 8.7e-08, 8.96e-08, 8.15e-08, 8.46e-08, 8.53e-08],
     [8.51e-08, 8.65e-08, 8.24e-08, 8.89e-08, 8.86e-08, 8.71e-08, 8.21e-08],
     [8.75e-08, 8.89e-08, 8.6e-08, 8.42e-08, 8.83e-08, 8.52e-08, 8.53e-08],
     [8.24e-08, 8.27e-08, 8.43e-08, 8.19e-08, 8.74e-08, 8.3e-08, 8.35e-08],
     [8.79e-08, 8.84e-08, 8.31e-08, 8.15e-08, 8.68e-08, 8.55e-08, 8.2e-08],
     [8.63e-08, 8.76e-08, 8.52e-08, 8.46e-08, 8.67e-08, 8.9e-08, 8.38e-08]]



    T2 = T*T
    Tinv = 1/T
    T2inv = Tinv*Tinv
    logT = log(T)

    lambdas = exp(A + B*Tinv + C*logT + D*T + E*T2inv + F*T2)

    dlambdas_dT = diff(lambdas, T)
    d2lambdas_dT2 = diff(lambdas, T, 2)
    d3lambdas_dT3 = diff(lambdas, T, 3)


    lambdas_sym = [[float(lambdas.subs({T: T_num, A: lambdasA[i][j], B: lambdasB[i][j], C:lambdasC[i][j], D:lambdasD[i][j], E:lambdasE[i][j], F:lambdasF[i][j]}))
                  for j in range(N)] for i in range(N)]
    dlambdas_dT_sym = [[float(dlambdas_dT.subs({T: T_num, A: lambdasA[i][j], B: lambdasB[i][j], C:lambdasC[i][j], D:lambdasD[i][j], E:lambdasE[i][j], F:lambdasF[i][j]}))
                  for j in range(N)] for i in range(N)]
    d2lambdas_dT2_sym = [[float(d2lambdas_dT2.subs({T: T_num, A: lambdasA[i][j], B: lambdasB[i][j], C:lambdasC[i][j], D:lambdasD[i][j], E:lambdasE[i][j], F:lambdasF[i][j]}))
                  for j in range(N)] for i in range(N)]
    d3lambdas_dT3_sym = [[float(d3lambdas_dT3.subs({T: T_num, A: lambdasA[i][j], B: lambdasB[i][j], C:lambdasC[i][j], D:lambdasD[i][j], E:lambdasE[i][j], F:lambdasF[i][j]}))
                  for j in range(N)] for i in range(N)]

    GE = Wilson(T=T_num, xs=[1.0/N]*N, ABCDEF=(lambdasA, lambdasB, lambdasC, lambdasD, lambdasE, lambdasF))
    assert_close2d(GE.lambdas(), lambdas_sym, rtol=1e-15)
    assert_close2d(GE.dlambdas_dT(), dlambdas_dT_sym, rtol=2e-15)
    assert_close2d(GE.d2lambdas_dT2(), d2lambdas_dT2_sym, rtol=4e-15)
    assert_close2d(GE.d3lambdas_dT3(), d3lambdas_dT3_sym, rtol=1e-14)


def test_lambdas_performance_np():
    return # not used yet
    N = 3
    A = np.random.random((N, N))
    B = np.random.random((N, N))*-3000
    D = np.random.random((N, N))*-1e-3

    C = np.random.random((N, N))*1e-6
    E = np.random.random((N, N))*1e-7
    F = np.random.random((N, N))*1e-8
    T = 300.0
    xs = np.abs(np.random.random(N))
    xs = xs/xs.sum()

    GE = Wilson(T=T, xs=xs, ABCDEF=(A, B, C, D, E, F))


def test_lambdas_performance_py():
    return # not used yet
    from random import random
    import thermo
    N = 3
    A = [[random() for _ in range(N)] for _ in range(N)]
    B = [[random()*-3000.0 for _ in range(N)] for _ in range(N)]
    D = [[random()*-1e-3 for _ in range(N)] for _ in range(N)]
    C = [[random()*1e-6 for _ in range(N)] for _ in range(N)]
    E = [[random()*1e-7 for _ in range(N)] for _ in range(N)]
    F = [[random()*1e-8 for _ in range(N)] for _ in range(N)]
    out = [[0.0]*N for _ in range(N)]
    thermo.wilson.interaction_exp(400.0, N, A, B, C, D, E, F, out)

def test_wilson_np_output_and_hash():
    T = 331.42
    N = 3

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

    model = Wilson(T=T, xs=xs, ABCDEF=(A, B, C, D, E, F))
    modelnp = Wilson(T=T, xs=np.array(xs), ABCDEF=(np.array(A), np.array(B), np.array(C), np.array(D), np.array(E), np.array(F)))
    modelnp2 = modelnp.to_T_xs(T=T, xs=np.array(xs))

    check_np_output_activity(model, modelnp, modelnp2)

    json_string = modelnp.as_json()
    new = Wilson.from_json(json_string)
    assert new == modelnp

    assert model.model_hash() == modelnp.model_hash()
    assert new.model_hash() == modelnp.model_hash()
    assert new.model_hash() == modelnp2.model_hash()

    # Pickle checks
    modelnp_pickle = pickle.loads(pickle.dumps(modelnp))
    assert modelnp_pickle == modelnp
    model_pickle = pickle.loads(pickle.dumps(model))
    assert model_pickle == model

def test_wilson_np_hash_different_input_forms():
    T = 331.42
    N = 3

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

    lambda_coeffs = [[[A[i][j], B[i][j], C[i][j], D[i][j], E[i][j], F[i][j]] for j in range(N)] for i in range(N)]

    model_ABCDEF = Wilson(T=T, xs=xs, ABCDEF=(A, B, C, D, E, F))
    model_lambda_coeffs = Wilson(T=T, xs=xs, lambda_coeffs=lambda_coeffs)
    model_lambda_coeffs2 = Wilson.from_json(model_lambda_coeffs.as_json())

    assert model_ABCDEF.GE() == model_lambda_coeffs.GE()
    assert hash(model_ABCDEF) == hash(model_lambda_coeffs)
    assert model_ABCDEF.GE() == model_lambda_coeffs2.GE()
    assert hash(model_ABCDEF) == hash(model_lambda_coeffs2)


    modelnp = Wilson(T=T, xs=np.array(xs), ABCDEF=(np.array(A), np.array(B), np.array(C), np.array(D), np.array(E), np.array(F)))
    model_lambda_coeffsnp = Wilson(T=T, xs=np.array(xs), lambda_coeffs=np.array(lambda_coeffs))
    assert modelnp.GE() == model_lambda_coeffsnp.GE()
    assert hash(modelnp) == hash(model_lambda_coeffsnp)
    model_lambda_coeffsnp2 = Wilson.from_json(model_lambda_coeffsnp.as_json())
    assert hash(modelnp) == hash(model_lambda_coeffsnp2)

def test_Wilson_numpy_output_correct_array_internal_ownership():
    T = 331.42
    N = 3
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
    lambda_coeffs = [[[A[i][j], B[i][j], C[i][j], D[i][j], E[i][j], F[i][j]] for j in range(N)] for i in range(N)]
    xs = [0.229, 0.175, 0.596]
    modelnp = Wilson(T=T, xs=np.array(xs), lambda_coeffs=np.array(lambda_coeffs))
    for name in ('lambda_coeffs_A', 'lambda_coeffs_B', 'lambda_coeffs_C',
                 'lambda_coeffs_D', 'lambda_coeffs_E', 'lambda_coeffs_F'):
        obj = getattr(modelnp, name)
        assert obj.flags.c_contiguous
        assert obj.flags.owndata
