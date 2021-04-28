# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020, 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains a class :obj:`NRTL` for performing activity coefficient
calculations with the NRTL model. An older, functional calculation for
activity coefficients only is also present, :obj:`NRTL_gammas`.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

NRTL Class
==========

.. autoclass:: NRTL
    :members: to_T_xs, GE, dGE_dT, d2GE_dT2, d2GE_dTdxs, dGE_dxs, d2GE_dxixjs, taus, dtaus_dT, d2taus_dT2, d3taus_dT3, alphas, Gs, dGs_dT, d2Gs_dT2, d3Gs_dT3
    :undoc-members:
    :show-inheritance:
    :exclude-members: gammas
    :member-order: bysource

NRTL Functional Calculations
============================
.. autofunction:: NRTL_gammas


'''

from __future__ import division
from math import log, exp
from fluids.constants import R
from fluids.numerics import numpy as np
from thermo.activity import GibbsExcess

__all__ = ['NRTL', 'NRTL_gammas']

try:
    array, zeros, npsum, nplog = np.array, np.zeros, np.sum, np.log
except (ImportError, AttributeError):
    pass

def nrtl_gammas(xs, N, Gs, taus, xj_Gs_jis_inv, xj_Gs_taus_jis, gammas=None, vec0=None, vec1=None):
    if gammas is None:
        gammas = [0.0]*N
    if vec0 is None:
        vec0 = [0.0]*N
    if vec1 is None:
        vec1 = [0.0]*N

    for j in range(N):
        vec0[j] = xs[j]*xj_Gs_jis_inv[j]
    for j in range(N):
        vec1[j] = xj_Gs_taus_jis[j]*xj_Gs_jis_inv[j]

    vec0 = [xs[j]*xj_Gs_jis_inv[j] for j in range(N)]
    vec1 = [xj_Gs_taus_jis[j]*xj_Gs_jis_inv[j] for j in range(N)]

    for i in range(N):
        tot = xj_Gs_taus_jis[i]*xj_Gs_jis_inv[i]
        Gsi = Gs[i]
        tausi = taus[i]
        for j in range(N):
            tot += vec0[j]*Gsi[j]*(tausi[j] - vec1[j])

        gammas[i] = exp(tot)
    return gammas

def nrtl_taus(T, N, A, B, E, F, G, H, taus=None):

    if taus is None:
        taus = [[0.0]*N for _ in range(N)] # numba: delete
#        taus = zeros((N, N)) # numba: uncomment

    T2 = T*T
    Tinv = 1.0/T
    T2inv = Tinv*Tinv
    logT = log(T)
    for i in range(N):
        Ai = A[i]
        Bi = B[i]
        Ei = E[i]
        Fi = F[i]
        Gi = G[i]
        Hi = H[i]
        tausi = taus[i]
        for j in range(N):
            tausi[j] = (Ai[j] + Bi[j]*Tinv + Ei[j]*logT
                            + Fi[j]*T + Gi[j]*T2inv
                            + Hi[j]*T2)
    return taus

def nrtl_dtaus_dT(T, N, B, E, F, G, H, dtaus_dT=None):
    if dtaus_dT is None:
        dtaus_dT = [[0.0]*N for _ in range(N)] # numba: delete
#        dtaus_dT = zeros((N, N)) # numba: uncomment

    Tinv = 1.0/T
    nT2inv = -Tinv*Tinv
    n2T3inv = 2.0*nT2inv*Tinv
    T2 = T + T
    for i in range(N):
        Bi = B[i]
        Ei = E[i]
        Fi = F[i]
        Gi = G[i]
        Hi = H[i]
        dtaus_dTi = dtaus_dT[i]
        for j in range(N):
            dtaus_dTi[j] = (Fi[j] + nT2inv*Bi[j] + Tinv*Ei[j]
            + n2T3inv*Gi[j] + T2*Hi[j])
    return dtaus_dT

def nrtl_d2taus_dT2(T, N, B, E, G, H, d2taus_dT2=None):
    if d2taus_dT2 is None:
        d2taus_dT2 = [[0.0]*N for _ in range(N)] # numba: delete
#        d2taus_dT2 = zeros((N, N)) # numba: uncomment

    Tinv = 1.0/T
    Tinv2 = Tinv*Tinv

    T3inv2 = 2.0*(Tinv2*Tinv)
    nT2inv = -Tinv*Tinv
    T4inv6 = 6.0*(Tinv2*Tinv2)
    for i in range(N):
        Bi = B[i]
        Ei = E[i]
        Gi = G[i]
        Hi = H[i]
        d2taus_dT2i = d2taus_dT2[i]
        for j in range(N):
            d2taus_dT2i[j] = (2.0*Hi[j] + T3inv2*Bi[j]
                                + nT2inv*Ei[j]
                                + T4inv6*Gi[j])
    return d2taus_dT2

def nrtl_d3taus_dT3(T, N, B, E, G, d3taus_dT3=None):
    if d3taus_dT3 is None:
        d3taus_dT3 = [[0.0]*N for _ in range(N)] # numba: delete
#        d3taus_dT3 = zeros((N, N)) # numba: uncomment

    Tinv = 1.0/T
    T2inv = Tinv*Tinv
    nT4inv6 = -6.0*T2inv*T2inv
    T3inv2 = 2.0*T2inv*Tinv
    T5inv24 = -24.0*(T2inv*T2inv*Tinv)

    for i in range(N):
        Bi = B[i]
        Ei = E[i]
        Gi = G[i]
        d3taus_dT3i = d3taus_dT3[i]
        for j in range(N):
            d3taus_dT3i[j] = (nT4inv6*Bi[j]
                                  + T3inv2*Ei[j]
                                  + T5inv24*Gi[j])
    return d3taus_dT3

def nrtl_alphas(T, N, c, d, alphas=None):
    if alphas is None:
        alphas = [[0.0]*N for _ in range(N)] # numba: delete
#        alphas = zeros((N, N)) # numba: uncomment

    for i in range(N):
        ci = c[i]
        di = d[i]
        alphasi = alphas[i]
        for j in range(N):
            alphasi[j] = ci[j] + di[j]*T
    return alphas

def nrtl_Gs(N, alphas, taus, Gs=None):
    if Gs is None:
        Gs = [[0.0]*N for _ in range(N)] # numba: delete
#        Gs = zeros((N, N)) # numba: uncomment

    for i in range(N):
        alphasi = alphas[i]
        tausi = taus[i]
        Gsi = Gs[i]
        for j in range(N):
            Gsi[j] = exp(-alphasi[j]*tausi[j])
    return Gs

def nrtl_dGs_dT(N, alphas, dalphas_dT, taus, dtaus_dT, Gs, dGs_dT=None):
    if dGs_dT is None:
        dGs_dT = [[0.0]*N for _ in range(N)] # numba: delete
#        dGs_dT = zeros((N, N)) # numba: uncomment

    for i in range(N):
        alphasi = alphas[i]
        tausi = taus[i]
        dalphasi = dalphas_dT[i]
        dtausi = dtaus_dT[i]
        Gsi = Gs[i]
        dGs_dTi = dGs_dT[i]
        for j in range(N):
            dGs_dTi[j] = (-alphasi[j]*dtausi[j] - tausi[j]*dalphasi[j])*Gsi[j]
    return dGs_dT

def nrtl_d2Gs_dT2(N, alphas, dalphas_dT, taus, dtaus_dT, d2taus_dT2, Gs, d2Gs_dT2=None):
    if d2Gs_dT2 is None:
        d2Gs_dT2 = [[0.0]*N for _ in range(N)] # numba: delete
#        d2Gs_dT2 = zeros((N, N)) # numba: uncomment

    for i in range(N):
        alphasi = alphas[i]
        tausi = taus[i]
        dalphasi = dalphas_dT[i]
        dtausi = dtaus_dT[i]
        d2taus_dT2i = d2taus_dT2[i]
        Gsi = Gs[i]
        d2Gs_dT2i = d2Gs_dT2[i]
        for j in range(N):
            t1 = alphasi[j]*dtausi[j] + tausi[j]*dalphasi[j]
            d2Gs_dT2i[j] = (t1*t1 - alphasi[j]*d2taus_dT2i[j]
                                    - 2.0*dalphasi[j]*dtausi[j])*Gsi[j]
    return d2Gs_dT2

def nrtl_d3Gs_dT3(N, alphas, dalphas_dT, taus, dtaus_dT, d2taus_dT2, d3taus_dT3, Gs, d3Gs_dT3=None):
    if d3Gs_dT3 is None:
        d3Gs_dT3 = [[0.0]*N for _ in range(N)] # numba: delete
#        d3Gs_dT3 = zeros((N, N)) # numba: uncomment

    for i in range(N):
        alphasi = alphas[i]
        tausi = taus[i]
        dalphasi = dalphas_dT[i]
        dtaus_dTi = dtaus_dT[i]
        d2taus_dT2i = d2taus_dT2[i]
        d3taus_dT3i = d3taus_dT3[i]
        Gsi = Gs[i]
        d3Gs_dT3i = d3Gs_dT3[i]
        for j in range(N):
            x0 = alphasi[j]
            x1 = tausi[j]
            x2 = dalphasi[j]

            x3 = d2taus_dT2i[j]
            x4 = dtaus_dTi[j]
            x5 = x0*x4 + x1*x2
            d3Gs_dT3i[j] = Gsi[j]*(-x0*d3taus_dT3i[j] - 3.0*x2*x3 - x5*x5*x5 + 3.0*x5*(x0*x3 + 2.0*x2*x4))
    return d3Gs_dT3

def nrtl_xj_Gs_jis_and_Gs_taus_jis(N, xs, Gs, taus, xj_Gs_jis=None, xj_Gs_taus_jis=None):
    if xj_Gs_jis is None:
        xj_Gs_jis = [0.0]*N
    if xj_Gs_taus_jis is None:
        xj_Gs_taus_jis = [0.0]*N

    for i in range(N):
        tot1 = 0.0
        tot2 = 0.0
        for j in range(N):
            xjGji = xs[j]*Gs[j][i]
            tot1 += xjGji
            tot2 += xjGji*taus[j][i]
        xj_Gs_jis[i] = tot1
        xj_Gs_taus_jis[i] = tot2
    return xj_Gs_jis, xj_Gs_taus_jis

def nrtl_xj_Gs_jis(N, xs, Gs, xj_Gs_jis=None):
    if xj_Gs_jis is None:
        xj_Gs_jis = [0.0]*N

    for i in range(N):
        tot1 = 0.0
        for j in range(N):
            tot1 += xs[j]*Gs[j][i]
        xj_Gs_jis[i] = tot1
    return xj_Gs_jis

def nrtl_xj_Gs_taus_jis(N, xs, Gs, taus, xj_Gs_taus_jis=None):
    if xj_Gs_taus_jis is None:
        xj_Gs_taus_jis = [0.0]*N

    for i in range(N):
        tot2 = 0.0
        for j in range(N):
            tot2 += xs[j]*Gs[j][i]*taus[j][i]
        xj_Gs_taus_jis[i] = tot2
    return xj_Gs_taus_jis

def nrtl_GE(N, T, xs, xj_Gs_taus_jis, xj_Gs_jis_inv):
    GE = 0.0
    for i in range(N):
        GE += xs[i]*xj_Gs_taus_jis[i]*xj_Gs_jis_inv[i]
    GE *= T*R
    return GE

def nrtl_dGE_dT(N, T, xs, xj_Gs_taus_jis, xj_Gs_jis_inv, xj_dGs_dT_jis, xj_taus_dGs_dT_jis, xj_Gs_dtaus_dT_jis):
    dGE_dT = 0.0
    for i in range(N):
        dGE_dT += (xs[i]*(xj_Gs_taus_jis[i] + T*((xj_taus_dGs_dT_jis[i] + xj_Gs_dtaus_dT_jis[i])
                - (xj_Gs_taus_jis[i]*xj_dGs_dT_jis[i])*xj_Gs_jis_inv[i]))*xj_Gs_jis_inv[i])
    dGE_dT *= R
    return dGE_dT

def nrtl_d2GE_dT2(N, T, xs, taus, dtaus_dT, d2taus_dT2, alphas, dalphas_dT, Gs, dGs_dT, d2Gs_dT2):
    tot = 0.0
    for i in range(N):
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        sum4 = 0.0
        sum5 = 0.0

        sum6 = 0.0
        sum7 = 0.0
        sum8 = 0.0
        sum9 = 0.0
        for j in range(N):
            tauji = taus[j][i]
            dtaus_dTji = dtaus_dT[j][i]

            Gjixj = Gs[j][i]*xs[j]
            dGjidTxj = dGs_dT[j][i]*xs[j]
            d2GjidT2xj = xs[j]*d2Gs_dT2[j][i]

            sum1 += Gjixj
            sum2 += tauji*Gjixj
            sum3 += dGjidTxj

            sum4 += tauji*dGjidTxj
            sum5 += dtaus_dTji*Gjixj

            sum6 += d2GjidT2xj

            sum7 += tauji*d2GjidT2xj

            sum8 += Gjixj*d2taus_dT2[j][i]

            sum9 += dGjidTxj*dtaus_dTji

        term1 = -T*sum2*(sum6 - 2.0*sum3*sum3/sum1)/sum1
        term2 = T*(sum7 + sum8 + 2.0*sum9)
        term3 = -2.0*T*(sum3*(sum4 + sum5))/sum1
        term4 = -2.0*(sum2*sum3)/sum1
        term5 = 2*(sum4 + sum5)

        tot += xs[i]*(term1 + term2 + term3 + term4 + term5)/sum1
    d2GE_dT2 = R*tot
    return d2GE_dT2

def nrtl_dGE_dxs(N, T, xs, taus, Gs, xj_Gs_taus_jis, xj_Gs_jis_inv, dGE_dxs=None):
    if dGE_dxs is None:
        dGE_dxs = [0.0]*N

    RT = R*T
    for k in range(N):
        # k is what is being differentiated
        tot = xj_Gs_taus_jis[k]*xj_Gs_jis_inv[k]
        for i in range(N):
            tot += xs[i]*xj_Gs_jis_inv[i]*Gs[k][i]*(taus[k][i] - xj_Gs_jis_inv[i]*xj_Gs_taus_jis[i])
        dGE_dxs[k] = tot*RT
    return dGE_dxs

def nrtl_d2GE_dxixjs(N, T, xs, taus, alphas, Gs, xj_Gs_taus_jis, xj_Gs_jis_inv, d2GE_dxixjs=None):
    if d2GE_dxixjs is None:
        d2GE_dxixjs = [[0.0]*N for _ in range(N)] # numba: delete
#        d2GE_dxixjs = zeros((N, N)) # numba: uncomment

    RT = R*T
    for i in range(N):
        row = d2GE_dxixjs[i]
        for j in range(N):
            tot = 0.0
            # two small terms
            tot += Gs[i][j]*taus[i][j]*xj_Gs_jis_inv[j]
            tot += Gs[j][i]*taus[j][i]*xj_Gs_jis_inv[i]

            # Two large terms
            tot -= xj_Gs_taus_jis[j]*Gs[i][j]*(xj_Gs_jis_inv[j]*xj_Gs_jis_inv[j])
            tot -= xj_Gs_taus_jis[i]*Gs[j][i]*(xj_Gs_jis_inv[i]*xj_Gs_jis_inv[i])

            # Three terms
            for k in range(N):
                tot += 2.0*xs[k]*xj_Gs_taus_jis[k]*Gs[i][k]*Gs[j][k]*(xj_Gs_jis_inv[k]*xj_Gs_jis_inv[k]*xj_Gs_jis_inv[k])

            # 6 terms
            for k in range(N):
                tot -= xs[k]*Gs[i][k]*Gs[j][k]*(taus[j][k] + taus[i][k])*xj_Gs_jis_inv[k]*xj_Gs_jis_inv[k]

            tot *= RT
            row[j] = tot
    return d2GE_dxixjs

def nrtl_d2GE_dTdxs(N, T, xs, taus, dtaus_dT, Gs, dGs_dT, xj_Gs_taus_jis,
                    xj_Gs_jis_inv, xj_dGs_dT_jis, xj_taus_dGs_dT_jis,
                    xj_Gs_dtaus_dT_jis, d2GE_dTdxs=None, vec0=None, vec1=None):
    if d2GE_dTdxs is None:
        d2GE_dTdxs = [0.0]*N
    if vec0 is None:
        vec0 = [0.0]*N
    if vec1 is None:
        vec1 = [0.0]*N

    sum1 = xj_Gs_jis_inv
    sum2 = xj_Gs_taus_jis
    sum3 = xj_dGs_dT_jis
    sum4 = xj_taus_dGs_dT_jis
    sum5 = xj_Gs_dtaus_dT_jis

    for i in range(N):
        vec0[i] = xs[i]*sum1[i]
    for i in range(N):
        vec1[i] = sum1[i]*sum2[i]

    for i in range(N):
        others = vec1[i]
        tot1 = sum1[i]*(sum3[i]*vec1[i] - (sum5[i] + sum4[i])) # Last singleton and second last singleton

        Gsi = Gs[i]
        tausi = taus[i]
        for j in range(N):
            t0 = vec1[j]*dGs_dT[i][j]
            t0 += sum1[j]*Gsi[j]*(sum3[j]*(tausi[j] - vec1[j] - vec1[j]) + sum5[j] + sum4[j]) # Could store and factor this stuff out but just makes it slower
            t0 -= (Gsi[j]*dtaus_dT[i][j] + tausi[j]*dGs_dT[i][j])

            tot1 += t0*vec0[j]

            others += vec0[j]*Gsi[j]*(tausi[j] - vec1[j])

        d2GE_dTdxs[i] = -R*(T*tot1 - others)
    return d2GE_dTdxs

class NRTL(GibbsExcess):
    r'''Class for representing an a liquid with excess gibbs energy represented
    by the NRTL equation. This model is capable of representing VL and LL
    behavior. [1]_ and [2]_ are good references on this model.

    Parameters
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]
    tau_coeffs : list[list[list[float]]], optional
        NRTL parameters, indexed by [i][j] and then each value is a 6
        element list with parameters [`a`, `b`, `e`, `f`, `g`, `h`];
        either (`tau_coeffs` and `alpha_coeffs`) or `ABEFGHCD` are required, [-]
    alpha_coeffs : list[list[float]], optional
        NRTL alpha parameters, []
    ABEFGHCD : tuple[list[list[float]], 8], optional
        Contains the following. One of (`tau_coeffs` and `alpha_coeffs`) or
        `ABEFGHCD` are required, [-]

        a : list[list[float]]
            `a` parameters used in calculating :obj:`NRTL.taus`, [-]
        b : list[list[float]]
            `b` parameters used in calculating :obj:`NRTL.taus`, [K]
        e : list[list[float]]
            `e` parameters used in calculating :obj:`NRTL.taus`, [-]
        f : list[list[float]]
            `f` paraemeters used in calculating :obj:`NRTL.taus`, [1/K]
        e : list[list[float]]
            `e` parameters used in calculating :obj:`NRTL.taus`, [K^2]
        f : list[list[float]]
            `f` parameters used in calculating :obj:`NRTL.taus`, [1/K^2]
        c : list[list[float]]
            `c` parameters used in calculating :obj:`NRTL.alphas`, [-]
        d : list[list[float]]
            `d` paraemeters used in calculating :obj:`NRTL.alphas`, [1/K]

    Attributes
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]

    Notes
    -----
    In addition to the methods presented here, the methods of its base class
    :obj:`thermo.activity.GibbsExcess` are available as well.

    Examples
    --------
    The DDBST has published numerous problems showing this model a simple
    binary system, Example P05.01b in [2]_, shows how to use parameters from
    the DDBST which are in units of calorie and need the gas constant as a
    multiplier:

    >>> from scipy.constants import calorie, R
    >>> N = 2
    >>> T = 70.0 + 273.15
    >>> xs = [0.252, 0.748]
    >>> tausA = tausE = tausF = tausG = tausH = alphaD = [[0.0]*N for i in range(N)]
    >>> tausB = [[0, -121.2691/R*calorie], [1337.8574/R*calorie, 0]]
    >>> alphaC =  [[0, 0.2974],[.2974, 0]]
    >>> ABEFGHCD = (tausA, tausB, tausE, tausF, tausG, tausH, alphaC, alphaD)
    >>> GE = NRTL(T=T, xs=xs, ABEFGHCD=ABEFGHCD)
    >>> GE.gammas()
    [1.93605165145, 1.15366304520]
    >>> GE
    NRTL(T=343.15, xs=[0.252, 0.748], ABEFGHCD=([[0.0, 0.0], [0.0, 0.0]], [[0, -61.0249799309399], [673.2359767282798, 0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0, 0.2974], [0.2974, 0]], [[0.0, 0.0], [0.0, 0.0]]))
    >>> GE.GE(), GE.dGE_dT(), GE.d2GE_dT2()
    (780.053057219, 0.5743500022, -0.003584843605528)
    >>> GE.HE(), GE.SE(), GE.dHE_dT(), GE.dSE_dT()
    (582.964853938, -0.57435000227, 1.230139083237, 0.0035848436055)

    The solution given by the DDBST has the same values [1.936, 1.154],
    and can be found here:
    http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/P05.01b%20VLE%20Behavior%20of%20Ethanol%20-%20Water%20Using%20NRTL.xps

    References
    ----------
    .. [1] Poling, Bruce E., John M. Prausnitz, and John P. O’Connell. The
       Properties of Gases and Liquids. 5th edition. New York: McGraw-Hill
       Professional, 2000.
    .. [2] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen Rarey.
       Chemical Thermodynamics for Process Simulation. John Wiley & Sons, 2019.
    '''
    
    model_attriubtes = ('tau_coeffs_A', 'tau_coeffs_B', 'tau_coeffs_E', 'tau_coeffs_F',
                         'tau_coeffs_G', 'tau_coeffs_H', 'alpha_coeffs_c', 'alpha_coeffs_d')
    model_id = 100
    
    def __init__(self, T, xs, tau_coeffs=None, alpha_coeffs=None,
                 ABEFGHCD=None):
        self.T = T
        self.xs = xs
        self.scalar = scalar = type(xs) is list

        if ABEFGHCD is not None:
            (self.tau_coeffs_A, self.tau_coeffs_B, self.tau_coeffs_E,
            self.tau_coeffs_F, self.tau_coeffs_G, self.tau_coeffs_H,
            self.alpha_coeffs_c, self.alpha_coeffs_d) = ABEFGHCD
            self.N = N = len(self.tau_coeffs_A)
        else:
            if tau_coeffs is not None:
                if scalar:
                    self.tau_coeffs_A = [[i[0] for i in l] for l in tau_coeffs]
                    self.tau_coeffs_B = [[i[1] for i in l] for l in tau_coeffs]
                    self.tau_coeffs_E = [[i[2] for i in l] for l in tau_coeffs]
                    self.tau_coeffs_F = [[i[3] for i in l] for l in tau_coeffs]
                    self.tau_coeffs_G = [[i[4] for i in l] for l in tau_coeffs]
                    self.tau_coeffs_H = [[i[5] for i in l] for l in tau_coeffs]
                else:
                    self.tau_coeffs_A = array(tau_coeffs[:,:,0], order='C', copy=True)
                    self.tau_coeffs_B = array(tau_coeffs[:,:,1], order='C', copy=True)
                    self.tau_coeffs_E = array(tau_coeffs[:,:,2], order='C', copy=True)
                    self.tau_coeffs_F = array(tau_coeffs[:,:,3], order='C', copy=True)
                    self.tau_coeffs_G = array(tau_coeffs[:,:,4], order='C', copy=True)
                    self.tau_coeffs_H = array(tau_coeffs[:,:,5], order='C', copy=True)
            else:
                raise ValueError("`tau_coeffs` is required")

            if alpha_coeffs is not None:
                if scalar:
                    self.alpha_coeffs_c = [[i[0] for i in l] for l in alpha_coeffs]
                    self.alpha_coeffs_d = [[i[1] for i in l] for l in alpha_coeffs]
                else:
                    self.alpha_coeffs_c = array(alpha_coeffs[:,:,0], order='C', copy=True)
                    self.alpha_coeffs_d = array(alpha_coeffs[:,:,1], order='C', copy=True)
            else:
                raise ValueError("`alpha_coeffs` is required")

            self.N = N = len(self.tau_coeffs_A)

    @property
    def zero_coeffs(self):
        '''Method to return a 2D list-of-lists of zeros.
        '''
        try:
            return self._zero_coeffs
        except AttributeError:
            pass
        N = self.N
        if self.scalar:
            self._zero_coeffs = [[0.0]*N for _ in range(N)]
        else:
            self._zero_coeffs = zeros((N, N))
        return self._zero_coeffs

    def __repr__(self):
        s = '%s(T=%s, xs=%s, ABEFGHCD=%s)' %(self.__class__.__name__, repr(self.T), repr(self.xs),
                (self.tau_coeffs_A,  self.tau_coeffs_B, self.tau_coeffs_E,
                 self.tau_coeffs_F, self.tau_coeffs_G, self.tau_coeffs_H,
                 self.alpha_coeffs_c, self.alpha_coeffs_d))
        return s


    def to_T_xs(self, T, xs):
        r'''Method to construct a new :obj:`NRTL` instance at
        temperature `T`, and mole fractions `xs`
        with the same parameters as the existing object.

        Parameters
        ----------
        T : float
            Temperature, [K]
        xs : list[float]
            Mole fractions of each component, [-]

        Returns
        -------
        obj : NRTL
            New :obj:`NRTL` object at the specified conditions [-]

        Notes
        -----
        If the new temperature is the same temperature as the existing
        temperature, if the `tau`, `Gs`, or `alphas` terms or their derivatives
        have been calculated, they will be set to the new object as well.
        '''
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.xs = xs
        new.N = self.N
        new.scalar = self.scalar
        (new.tau_coeffs_A, new.tau_coeffs_B, new.tau_coeffs_E,
         new.tau_coeffs_F, new.tau_coeffs_G, new.tau_coeffs_H,
         new.alpha_coeffs_c, new.alpha_coeffs_d) = (self.tau_coeffs_A, self.tau_coeffs_B, self.tau_coeffs_E,
                         self.tau_coeffs_F, self.tau_coeffs_G, self.tau_coeffs_H,
                         self.alpha_coeffs_c, self.alpha_coeffs_d)

        if T == self.T:
            try:
                new._taus = self._taus
            except AttributeError:
                pass
            try:
                new._dtaus_dT = self._dtaus_dT
            except AttributeError:
                pass
            try:
                new._d2taus_dT2 = self._d2taus_dT2
            except AttributeError:
                pass
            try:
                new._d3taus_dT3 = self._d3taus_dT3
            except AttributeError:
                pass
            try:
                new._alphas = self._alphas
            except AttributeError:
                pass
            try:
                new._Gs = self._Gs
            except AttributeError:
                pass
            try:
                new._dGs_dT = self._dGs_dT
            except AttributeError:
                pass
            try:
                new._d2Gs_dT2 = self._d2Gs_dT2
            except AttributeError:
                pass
            try:
                new._d3Gs_dT3 = self._d3Gs_dT3
            except AttributeError:
                pass

        try:
            new._zero_coeffs = self.zero_coeffs
        except AttributeError:
            pass

        return new

    def gammas(self):
        try:
            return self._gammas
        except AttributeError:
            pass
        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()
        try:
            Gs = self._Gs
        except AttributeError:
            Gs = self.Gs()

        xs, N = self.xs, self.N

        xj_Gs_jis_inv, xj_Gs_taus_jis = self.xj_Gs_jis_inv(), self.xj_Gs_taus_jis()

        if self.scalar:
            gammas = [0.0]*self.N
        else:
            gammas = zeros(self.N)

        self._gammas = nrtl_gammas(xs, N, Gs, taus, xj_Gs_jis_inv, xj_Gs_taus_jis, gammas)
        return gammas


    def taus(self):
        r'''Calculate and return the `tau` terms for the NRTL model for a
        specified temperature.

        .. math::
            \tau_{ij}=A_{ij}+\frac{B_{ij}}{T}+E_{ij}\ln T + F_{ij}T
            + \frac{G_{ij}}{T^2} + H_{ij}{T^2}

        Returns
        -------
        taus : list[list[float]]
            tau terms, asymmetric matrix [-]

        Notes
        -----
        These `tau ij` values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._taus
        except AttributeError:
            pass
        A = self.tau_coeffs_A
        B = self.tau_coeffs_B
        E = self.tau_coeffs_E
        F = self.tau_coeffs_F
        G = self.tau_coeffs_G
        H = self.tau_coeffs_H
        T, N = self.T, self.N
        if self.scalar:
            taus = [[0.0]*N for _ in range(N)]
        else:
            taus = zeros((N, N))

        self._taus = nrtl_taus(T, N, A, B, E, F, G, H, taus)
        return taus

    def dtaus_dT(self):
        r'''Calculate and return the temperature derivative of the `tau` terms
        for the NRTL model for a specified temperature.

        .. math::
            \frac{\partial \tau_{ij}} {\partial T}_{P, x_i} =
            - \frac{B_{ij}}{T^{2}} + \frac{E_{ij}}{T} + F_{ij}
            - \frac{2 G_{ij}}{T^{3}} + 2 H_{ij} T

        Returns
        -------
        dtaus_dT : list[list[float]]
            First temperature derivative of tau terms, asymmetric matrix [1/K]

        Notes
        -----
        '''
        try:
            return self._dtaus_dT
        except AttributeError:
            pass
        # Believed all correct but not tested
        B = self.tau_coeffs_B
        E = self.tau_coeffs_E
        F = self.tau_coeffs_F
        G = self.tau_coeffs_G
        H = self.tau_coeffs_H
        T, N = self.T, self.N
        if self.scalar:
            dtaus_dT = [[0.0]*N for _ in range(N)]
        else:
            dtaus_dT = zeros((N, N))
        self._dtaus_dT = nrtl_dtaus_dT(T, N, B, E, F, G, H, dtaus_dT)

        return dtaus_dT

    def d2taus_dT2(self):
        r'''Calculate and return the second temperature derivative of the `tau` terms for
        the NRTL model for a specified temperature.

        .. math::
            \frac{\partial^2 \tau_{ij}} {\partial T^2}_{P, x_i} =
            \frac{2 B_{ij}}{T^{3}} - \frac{E_{ij}}{T^{2}} + \frac{6 G_{ij}}
            {T^{4}} + 2 H_{ij}

        Returns
        -------
        d2taus_dT2 : list[list[float]]
            Second temperature derivative of tau terms, asymmetric matrix [1/K^2]

        Notes
        -----
        '''
        try:
            return self._d2taus_dT2
        except AttributeError:
            pass
        B = self.tau_coeffs_B
        E = self.tau_coeffs_E
        G = self.tau_coeffs_G
        H = self.tau_coeffs_H
        T, N = self.T, self.N

        if self.scalar:
            d2taus_dT2 = [[0.0]*N for _ in range(N)]
        else:
            d2taus_dT2 = zeros((N, N))

        self._d2taus_dT2 = nrtl_d2taus_dT2(T, N, B, E, G, H, d2taus_dT2)
        return d2taus_dT2

    def d3taus_dT3(self):
        r'''Calculate and return the third temperature derivative of the `tau`
        terms for the NRTL model for a specified temperature.

        .. math::
            \frac{\partial^3 \tau_{ij}} {\partial T^3}_{P, x_i} =
            - \frac{6 B_{ij}}{T^{4}} + \frac{2 E_{ij}}{T^{3}}
            - \frac{24 G_{ij}}{T^{5}}

        Returns
        -------
        d3taus_dT3 : list[list[float]]
            Third temperature derivative of tau terms, asymmetric matrix [1/K^3]

        Notes
        -----
        '''
        try:
            return self._d3taus_dT3
        except AttributeError:
            pass
        B = self.tau_coeffs_B
        E = self.tau_coeffs_E
        G = self.tau_coeffs_G
        T, N = self.T, self.N
        if self.scalar:
            d3taus_dT3 = [[0.0]*N for _ in range(N)]
        else:
            d3taus_dT3 = zeros((N, N))
        self._d3taus_dT3 = nrtl_d3taus_dT3(T, N, B, E, G, d3taus_dT3)
        return d3taus_dT3

    def alphas(self):
        r'''Calculates and return the `alpha` terms in the NRTL model for a
        specified temperature.

        .. math::
            \alpha_{ij}=c_{ij} + d_{ij}T

        Returns
        -------
        alphas : list[list[float]]
            alpha terms, possibly asymmetric matrix [-]

        Notes
        -----
        `alpha` values (and therefore `cij` and `dij` are normally symmetrical;
        but this is not strictly required.

        Some sources suggest the c term should be fit to a given system; but
        the `d` term should be fit for an entire chemical family to avoid
        overfitting.

        Recommended values for `cij` according to one source are:

        0.30 Nonpolar substances with nonpolar substances; low deviation from ideality.
        0.20 Hydrocarbons that are saturated interacting with polar liquids that do not associate, or systems that for multiple liquid phases which are immiscible
        0.47 Strongly self associative systems, interacting with non-polar substances

        `alpha_coeffs` should be a list[list[cij, dij]] so a 3d array
        '''
        try:
            return self._alphas
        except AttributeError:
            pass
        N = self.N

        if self.scalar:
            alphas = [[0.0]*N for _ in range(N)]
        else:
            alphas = zeros((N, N))

        self._alphas = nrtl_alphas(self.T, N, self.alpha_coeffs_c, self.alpha_coeffs_d, alphas)
        return alphas

    def dalphas_dT(self):
        '''Keep it as a function in case this needs to become more complicated.'''
        return self.alpha_coeffs_d

    def d2alphas_dT2(self):
        '''Keep it as a function in case this needs to become more complicated.'''
        return self.zero_coeffs

    def d3alphas_dT3(self):
        '''Keep it as a function in case this needs to become more complicated.'''
        return self.zero_coeffs

    def Gs(self):
        r'''Calculates and return the `G` terms in the NRTL model for a
        specified temperature.

        .. math::
            G_{ij}=\exp(-\alpha_{ij}\tau_{ij})

        Returns
        -------
        Gs : list[list[float]]
            G terms, asymmetric matrix [-]

        Notes
        -----
        '''
        try:
            return self._Gs
        except AttributeError:
            pass
        alphas = self.alphas()
        taus = self.taus()
        N = self.N

        if self.scalar:
            Gs = [[0.0]*N for _ in range(N)]
        else:
            Gs = zeros((N, N))

        self._Gs = nrtl_Gs(N, alphas, taus, Gs)
        return Gs

    def dGs_dT(self):
        r'''Calculates and return the first temperature derivative of `G` terms
        in the NRTL model for a specified temperature.

        .. math::
            \frac{\partial G_{ij}}{\partial T} = \left(- \alpha{\left(T \right)} \frac{d}{d T} \tau{\left(T \right)}
            - \tau{\left(T \right)} \frac{d}{d T} \alpha{\left(T \right)}\right)
            e^{- \alpha{\left(T \right)} \tau{\left(T \right)}}

        Returns
        -------
        dGs_dT : list[list[float]]
            Temperature derivative of G terms, asymmetric matrix [1/K]

        Notes
        -----
        Derived with SymPy:

        >>> from sympy import * # doctest:+SKIP
        >>> T = symbols('T') # doctest:+SKIP
        >>> alpha, tau = symbols('alpha, tau', cls=Function) # doctest:+SKIP
        >>> diff(exp(-alpha(T)*tau(T)), T) # doctest:+SKIP
        '''
        try:
            return self._dGs_dT
        except AttributeError:
            pass
        alphas = self.alphas()
        dalphas_dT = self.dalphas_dT()
        taus = self.taus()
        dtaus_dT = self.dtaus_dT()
        Gs = self.Gs()
        N = self.N

        if self.scalar:
            dGs_dT = [[0.0]*N for _ in range(N)]
        else:
            dGs_dT = zeros((N, N))

        self._dGs_dT = nrtl_dGs_dT(N, alphas, dalphas_dT, taus, dtaus_dT, Gs, dGs_dT)
        return dGs_dT

    def d2Gs_dT2(self):
        r'''Calculates and return the second temperature derivative of `G` terms
        in the NRTL model for a specified temperature.

        .. math::
            \frac{\partial^2 G_{ij}}{\partial T^2} = \left(\left(\alpha{\left(T
            \right)} \frac{d}{d T} \tau{\left(T \right)} + \tau{\left(T
            \right)} \frac{d}{d T} \alpha{\left(T \right)}\right)^{2}
            - \alpha{\left(T \right)} \frac{d^{2}}{d T^{2}} \tau{\left(T
            \right)} - 2 \frac{d}{d T} \alpha{\left(T \right)} \frac{d}{d T}
            \tau{\left(T \right)}\right) e^{- \alpha{\left(T \right)}
            \tau{\left(T \right)}}


        Returns
        -------
        d2Gs_dT2 : list[list[float]]
            Second temperature derivative of G terms, asymmetric matrix [1/K^2]

        Notes
        -----
        Derived with SymPy:

        >>> from sympy import * # doctest:+SKIP
        >>> T = symbols('T') # doctest:+SKIP
        >>> alpha, tau = symbols('alpha, tau', cls=Function) # doctest:+SKIP
        >>> diff(exp(-alpha(T)*tau(T)), T, 2) # doctest:+SKIP
        '''
        try:
            return self._d2Gs_dT2
        except AttributeError:
            pass
        alphas = self.alphas()
        dalphas_dT = self.dalphas_dT()
        taus = self.taus()
        dtaus_dT = self.dtaus_dT()
        d2taus_dT2 = self.d2taus_dT2()
        Gs = self.Gs()
        N = self.N

        if self.scalar:
            d2Gs_dT2 = [[0.0]*N for _ in range(N)]
        else:
            d2Gs_dT2 = zeros((N, N))

        self._d2Gs_dT2 = nrtl_d2Gs_dT2(N, alphas, dalphas_dT, taus, dtaus_dT, d2taus_dT2, Gs, d2Gs_dT2)
        return d2Gs_dT2

    def d3Gs_dT3(self):
        r'''Calculates and return the third temperature derivative of `G` terms
        in the NRTL model for a specified temperature.

        .. math::
            \frac{\partial^3 G_{ij}}{\partial T^3} = \left(\alpha{\left(T
            \right)} \frac{d}{d T} \tau{\left(T \right)} + \tau{\left(T
            \right)} \frac{d}{d T} \alpha{\left(T \right)}\right)^{3} + \left(3
            \alpha{\left(T \right)} \frac{d}{d T} \tau{\left(T \right)}
            + 3 \tau{\left(T \right)} \frac{d}{d T} \alpha{\left(T \right)}
            \right) \left(\alpha{\left(T \right)} \frac{d^{2}}{d T^{2}}
            \tau{\left(T \right)} + 2 \frac{d}{d T} \alpha{\left(T \right)}
            \frac{d}{d T} \tau{\left(T \right)}\right) - \alpha{\left(T
            \right)} \frac{d^{3}}{d T^{3}} \tau{\left(T \right)}
            - 3 \frac{d}{d T} \alpha{\left(T \right)} \frac{d^{2}}{d T^{2}}
            \tau{\left(T \right)}

        Returns
        -------
        d3Gs_dT3 : list[list[float]]
            Third temperature derivative of G terms, asymmetric matrix [1/K^3]

        Notes
        -----
        Derived with SymPy:

        >>> from sympy import * # doctest:+SKIP
        >>> T = symbols('T') # doctest:+SKIP
        >>> alpha, tau = symbols('alpha, tau', cls=Function) # doctest:+SKIP
        >>> diff(exp(-alpha(T)*tau(T)), T, 3) # doctest:+SKIP
        '''
        try:
            return self._d3Gs_dT3
        except AttributeError:
            pass
        N = self.N
        alphas = self.alphas()
        dalphas_dT = self.dalphas_dT()
        taus = self.taus()
        dtaus_dT = self.dtaus_dT()
        d2taus_dT2 = self.d2taus_dT2()
        d3taus_dT3 = self.d3taus_dT3()
        Gs = self.Gs()

        if self.scalar:
            d3Gs_dT3 = [[0.0]*N for _ in range(N)]
        else:
            d3Gs_dT3 = zeros((N, N))

        self._d3Gs_dT3 = nrtl_d3Gs_dT3(N, alphas, dalphas_dT, taus, dtaus_dT, d2taus_dT2, d3taus_dT3, Gs, d3Gs_dT3)
        return d3Gs_dT3



    def xj_Gs_jis(self):
        # sum1
        try:
            return self._xj_Gs_jis
        except AttributeError:
            pass
        try:
            Gs = self._Gs
        except AttributeError:
            Gs = self.Gs()
        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()

        xs, N = self.xs, self.N
        if self.scalar:
            _xj_Gs_jis = [0.0]*N
            _xj_Gs_taus_jis = [0.0]*N
        else:
            _xj_Gs_jis = zeros(N)
            _xj_Gs_taus_jis = zeros(N)

        nrtl_xj_Gs_jis_and_Gs_taus_jis(N, xs, Gs, taus, _xj_Gs_jis, _xj_Gs_taus_jis)

        self._xj_Gs_jis, self._xj_Gs_taus_jis = _xj_Gs_jis, _xj_Gs_taus_jis
        return _xj_Gs_jis

    def xj_Gs_jis_inv(self):
        try:
            return self._xj_Gs_jis_inv
        except AttributeError:
            pass

        try:
            xj_Gs_jis = self._xj_Gs_jis
        except AttributeError:
            xj_Gs_jis = self.xj_Gs_jis()

        if self.scalar:
            self._xj_Gs_jis_inv = [1.0/i for i in xj_Gs_jis]
        else:
            self._xj_Gs_jis_inv = 1.0/xj_Gs_jis
        return self._xj_Gs_jis_inv

    def xj_Gs_taus_jis(self):
        # sum2
        try:
            return self._xj_Gs_taus_jis
        except AttributeError:
            self.xj_Gs_jis()
            return self._xj_Gs_taus_jis

        try:
            Gs = self._Gs
        except AttributeError:
            Gs = self.Gs()

        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()

        xs, N = self.xs, self.N
        if self.scalar:
            xj_Gs_taus_jis = [0.0]*N
        else:
            xj_Gs_taus_jis = zeros(N)
        self._xj_Gs_taus_jis = nrtl_xj_Gs_taus_jis(N, xs, Gs, taus, xj_Gs_taus_jis)
        return xj_Gs_taus_jis


    def xj_dGs_dT_jis(self):
        # sum3
        try:
            return self._xj_dGs_dT_jis
        except AttributeError:
            pass
        try:
            dGs_dT = self._dGs_dT
        except AttributeError:
            dGs_dT = self.dGs_dT()

        xs, N = self.xs, self.N
        if self.scalar:
            xj_dGs_dT_jis = [0.0]*N
        else:
            xj_dGs_dT_jis = zeros(N)
        self._xj_dGs_dT_jis = nrtl_xj_Gs_jis(N, xs, dGs_dT, xj_dGs_dT_jis)
        return xj_dGs_dT_jis

    def xj_taus_dGs_dT_jis(self):
        # sum4
        try:
            return self._xj_taus_dGs_dT_jis
        except AttributeError:
            pass
        xs, N = self.xs, self.N
        try:
            dGs_dT = self._dGs_dT
        except AttributeError:
            dGs_dT = self.dGs_dT()
        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()

        if self.scalar:
            xj_taus_dGs_dT_jis = [0.0]*N
        else:
            xj_taus_dGs_dT_jis = zeros(N)
        self._xj_taus_dGs_dT_jis = nrtl_xj_Gs_taus_jis(N, xs, dGs_dT, taus, xj_taus_dGs_dT_jis)

        return xj_taus_dGs_dT_jis

    def xj_Gs_dtaus_dT_jis(self):
        # sum5
        try:
            return self._xj_Gs_dtaus_dT_jis
        except AttributeError:
            pass
        xs, N = self.xs, self.N
        try:
            dtaus_dT = self._dtaus_dT
        except AttributeError:
            dtaus_dT = self.dtaus_dT()
        try:
            Gs = self._Gs
        except AttributeError:
            Gs = self.Gs()

        if self.scalar:
            xj_Gs_dtaus_dT_jis = [0.0]*N
        else:
            xj_Gs_dtaus_dT_jis = zeros(N)
        self._xj_Gs_dtaus_dT_jis = nrtl_xj_Gs_taus_jis(N, xs, Gs, dtaus_dT, xj_Gs_dtaus_dT_jis)
        return xj_Gs_dtaus_dT_jis

    def GE(self):
        r'''Calculate and return the excess Gibbs energy of a liquid phase
        represented by the NRTL model.

        .. math::
            g^E = RT\sum_i x_i \frac{\sum_j \tau_{ji} G_{ji} x_j}
            {\sum_j G_{ji}x_j}

        Returns
        -------
        GE : float
            Excess Gibbs energy, [J/mol]

        Notes
        -----
        '''
        try:
            return self._GE
        except AttributeError:
            pass
        N = self.N
        xj_Gs_jis_inv, xj_Gs_taus_jis = self.xj_Gs_jis_inv(), self.xj_Gs_taus_jis()
        T, xs = self.T, self.xs

        self._GE = GE = nrtl_GE(N, T, xs, xj_Gs_taus_jis, xj_Gs_jis_inv)
        return GE

    def dGE_dT(self):
        r'''Calculate and return the first tempreature derivative of excess
        Gibbs energy of a liquid phase represented by the NRTL model.

        Returns
        -------
        dGE_dT : float
            First temperature derivative of excess Gibbs energy, [J/(mol*K)]

        Notes
        -----
        '''
        '''from sympy import *
        R, T, x = symbols('R, T, x')
        g, tau = symbols('g, tau', cls=Function)
        m, n, o = symbols('m, n, o', cls=Function)
        r, s, t = symbols('r, s, t', cls=Function)
        u, v, w = symbols('u, v, w', cls=Function)
        diff(T* (m(T)*n(T) + r(T)*s(T) + u(T)*v(T))/(o(T) + t(T) + w(T)), T)
        '''
        try:
            return self._dGE_dT
        except AttributeError:
            pass
        T, xs, N = self.T, self.xs, self.N

        xj_Gs_jis_inv = self.xj_Gs_jis_inv() # sum1 inv
        xj_Gs_taus_jis = self.xj_Gs_taus_jis() # sum2
        xj_dGs_dT_jis = self.xj_dGs_dT_jis() # sum3
        xj_taus_dGs_dT_jis = self.xj_taus_dGs_dT_jis() # sum4
        xj_Gs_dtaus_dT_jis = self.xj_Gs_dtaus_dT_jis() # sum5

        self._dGE_dT = nrtl_dGE_dT(N, T, xs, xj_Gs_taus_jis, xj_Gs_jis_inv, xj_dGs_dT_jis, xj_taus_dGs_dT_jis, xj_Gs_dtaus_dT_jis)
        return self._dGE_dT


    def d2GE_dT2(self):
        r'''Calculate and return the second tempreature derivative of excess
        Gibbs energy of a liquid phase represented by the NRTL model.

        Returns
        -------
        d2GE_dT2 : float
            Second temperature derivative of excess Gibbs energy, [J/(mol*K^2)]

        Notes
        -----
        '''
        '''from sympy import *
        R, T, x = symbols('R, T, x')
        g, tau = symbols('g, tau', cls=Function)
        m, n, o = symbols('m, n, o', cls=Function)
        r, s, t = symbols('r, s, t', cls=Function)
        u, v, w = symbols('u, v, w', cls=Function)

        (diff(T*(m(T)*n(T) + r(T)*s(T))/(o(T) + t(T)), T, 2))
        '''
        try:
            return self._d2GE_dT2
        except AttributeError:
            pass
        T, xs, N = self.T, self.xs, self.N
        taus = self.taus()
        dtaus_dT = self.dtaus_dT()
        d2taus_dT2 = self.d2taus_dT2()

        alphas = self.alphas()
        dalphas_dT = self.dalphas_dT()

        Gs = self.Gs()
        dGs_dT = self.dGs_dT()
        d2Gs_dT2 = self.d2Gs_dT2()

        self._d2GE_dT2 = d2GE_dT2 = nrtl_d2GE_dT2(N, T, xs, taus, dtaus_dT, d2taus_dT2, alphas, dalphas_dT, Gs, dGs_dT, d2Gs_dT2)
        return d2GE_dT2

    def dGE_dxs(self):
        r'''Calculate and return the mole fraction derivatives of excess Gibbs
        energy of a liquid represented by the NRTL model.

        .. math::
            \frac{\partial g^E}{\partial x_i}

        Returns
        -------
        dGE_dxs : list[float]
            Mole fraction derivatives of excess Gibbs energy, [J/mol]

        Notes
        -----
        '''
        '''
        from sympy import *
        N = 3
        R, T = symbols('R, T')
        x0, x1, x2 = symbols('x0, x1, x2')
        xs = [x0, x1, x2]

        tau00, tau01, tau02, tau10, tau11, tau12, tau20, tau21, tau22 = symbols(
            'tau00, tau01, tau02, tau10, tau11, tau12, tau20, tau21, tau22', cls=Function)
        tau_ijs = [[tau00(T), tau01(T), tau02(T)],
                   [tau10(T), tau11(T), tau12(T)],
                   [tau20(T), tau21(T), tau22(T)]]


        G00, G01, G02, G10, G11, G12, G20, G21, G22 = symbols(
            'G00, G01, G02, G10, G11, G12, G20, G21, G22', cls=Function)
        G_ijs = [[G00(T), G01(T), G02(T)],
                   [G10(T), G11(T), G12(T)],
                   [G20(T), G21(T), G22(T)]]
        ge = 0
        for i in [2]:#range(0):
            num = 0
            den = 0
            for j in range(N):
                num += tau_ijs[j][i]*G_ijs[j][i]*xs[j]
                den += G_ijs[j][i]*xs[j]
            ge += xs[i]*num/den
        ge = ge#*R*T
        diff(ge, x1), diff(ge, x2)
        '''
        try:
            return self._dGE_dxs
        except AttributeError:
            pass
        T, xs, N = self.T, self.xs, self.N
        taus = self.taus()
        Gs = self.Gs()
        xj_Gs_jis_inv = self.xj_Gs_jis_inv()
        xj_Gs_taus_jis = self.xj_Gs_taus_jis()
        if self.scalar:
            dGE_dxs = [0.0]*N
        else:
            dGE_dxs = zeros(N)

        self._dGE_dxs = nrtl_dGE_dxs(N, T, xs, taus, Gs, xj_Gs_taus_jis, xj_Gs_jis_inv, dGE_dxs)
        return dGE_dxs

    def d2GE_dxixjs(self):
        r'''Calculate and return the second mole fraction derivatives of excess
        Gibbs energy of a liquid represented by the NRTL model.

        .. math::
            \frac{\partial^2 g^E}{\partial x_i \partial x_j} = RT\left[
            + \frac{G_{ij}\tau_{ij}}{\sum_m x_m G_{mj}}
            + \frac{G_{ji}\tau_{jiij}}{\sum_m x_m G_{mi}}
            -\frac{(\sum_m x_m G_{mj}\tau_{mj})G_{ij}}{(\sum_m x_m G_{mj})^2}
            -\frac{(\sum_m x_m G_{mi}\tau_{mi})G_{ji}}{(\sum_m x_m G_{mi})^2}
            \sum_k \left(\frac{2x_k(\sum_m x_m \tau_{mk}G_{mk})G_{ik}G_{jk}}{(\sum_m x_m G_{mk})^3}
            - \frac{x_k G_{ik}G_{jk}(\tau_{jk} + \tau_{ik})}{(\sum_m x_m G_{mk})^2}
            \right)
            \right]

        Returns
        -------
        d2GE_dxixjs : list[list[float]]
            Second mole fraction derivatives of excess Gibbs energy, [J/mol]

        Notes
        -----
        '''
        '''
        from sympy import *
        N = 3
        R, T = symbols('R, T')
        x0, x1, x2 = symbols('x0, x1, x2')
        xs = [x0, x1, x2]

        tau00, tau01, tau02, tau10, tau11, tau12, tau20, tau21, tau22 = symbols(
            'tau00, tau01, tau02, tau10, tau11, tau12, tau20, tau21, tau22', cls=Function)
        tau_ijs = [[tau00(T), tau01(T), tau02(T)],
                   [tau10(T), tau11(T), tau12(T)],
                   [tau20(T), tau21(T), tau22(T)]]


        G00, G01, G02, G10, G11, G12, G20, G21, G22 = symbols(
            'G00, G01, G02, G10, G11, G12, G20, G21, G22', cls=Function)
        G_ijs = [[G00(T), G01(T), G02(T)],
                   [G10(T), G11(T), G12(T)],
                   [G20(T), G21(T), G22(T)]]

        tauG00, tauG01, tauG02, tauG10, tauG11, tauG12, tauG20, tauG21, tauG22 = symbols(
            'tauG00, tauG01, tauG02, tauG10, tauG11, tauG12, tauG20, tauG21, tauG22', cls=Function)
        tauG_ijs = [[tauG00(T), tauG01(T), tauG02(T)],
                   [tauG10(T), tauG11(T), tauG12(T)],
                   [tauG20(T), tauG21(T), tauG22(T)]]


        ge = 0
        for i in range(N):#range(0):
            num = 0
            den = 0
            for j in range(N):
        #         num += G_ijs[j][i]*tau_ijs[j][i]*xs[j]
                num += tauG_ijs[j][i]*xs[j]
                den += G_ijs[j][i]*xs[j]

            ge += xs[i]*num/den
        ge = ge#R*T

        diff(ge, x0, x1)
        '''
        try:
            return self._d2GE_dxixjs
        except AttributeError:
            pass
        T, xs, N = self.T, self.xs, self.N
        taus = self.taus()
        alphas = self.alphas()
        Gs = self.Gs()
        xj_Gs_jis_inv = self.xj_Gs_jis_inv()
        xj_Gs_taus_jis = self.xj_Gs_taus_jis()
        if self.scalar:
            d2GE_dxixjs = [[0.0]*N for _ in range(N)]
        else:
            d2GE_dxixjs = zeros((N, N))

        self._d2GE_dxixjs = nrtl_d2GE_dxixjs(N, T, xs, taus, alphas, Gs, xj_Gs_taus_jis, xj_Gs_jis_inv, d2GE_dxixjs)
        return d2GE_dxixjs


    def d2GE_dTdxs(self):
        r'''Calculate and return the temperature derivative of mole fraction
        derivatives of excess Gibbs energy of a liquid represented by the NRTL
        model.

        .. math::
            \frac{\partial^2 g^E}{\partial x_i \partial T} = R\left[-T\left(
            \sum_j \left(
            -\frac{x_j(G_{ij}\frac{\partial \tau_{ij}}{\partial T} + \tau_{ij}\frac{\partial G_{ij}}{\partial T})}{\sum_k x_k G_{kj}}
            + \frac{x_j G_{ij}\tau_{ij}(\sum_k x_k \frac{\partial G_{kj}}{\partial T})}{(\sum_k x_k G_{kj})^2}
            +\frac{x_j \frac{\partial G_{ij}}{\partial T}(\sum_k x_k G_{kj}\tau_{kj})}{(\sum_k x_k G_{kj})^2}
            + \frac{x_jG_{ij}(\sum_k x_k (G_{kj} \frac{\partial \tau_{kj}}{\partial T}  + \tau_{kj} \frac{\partial G_{kj}}{\partial T} ))}{(\sum_k x_k G_{kj})^2}
            -2\frac{x_j G_{ij} (\sum_k x_k \frac{\partial G_{kj}}{\partial T})(\sum_k x_k G_{kj}\tau_{kj})}{(\sum_k x_k G_{kj})^3}
            \right)
            - \frac{\sum_k (x_k G_{ki} \frac{\partial \tau_{ki}}{\partial T}  + x_k \tau_{ki} \frac{\partial G_{ki}}{\partial T}} {\sum_k x_k G_{ki}}
            + \frac{(\sum_k x_k \frac{\partial G_{ki}}{\partial T})(\sum_k x_k G_{ki} \tau_{ki})}{(\sum_k x_k G_{ki})^2}
            \right)
            + \frac{\sum_j x_j G_{ji}\tau_{ji}}{\sum_j x_j G_{ji}} + \sum_j \left(
            \frac{x_j G_{ij}(\sum_k x_k G_{kj}\tau_{kj})}{(\sum_k x_k G_{kj})^2} + \frac{x_j G_{ij}\tau_{ij}}{\sum_k x_k G_{kj}}
            \right)
            \right]

        Returns
        -------
        d2GE_dTdxs : list[float]
            Temperature derivative of mole fraction derivatives of excess Gibbs
            energy, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._d2GE_dTdxs
        except AttributeError:
            pass

        xj_Gs_jis_inv = self.xj_Gs_jis_inv()
        xj_Gs_taus_jis = self.xj_Gs_taus_jis()
        xj_dGs_dT_jis = self.xj_dGs_dT_jis()
        xj_taus_dGs_dT_jis = self.xj_taus_dGs_dT_jis()
        xj_Gs_dtaus_dT_jis = self.xj_Gs_dtaus_dT_jis()


        T, xs, N = self.T, self.xs, self.N
        taus = self.taus()
        dtaus_dT = self.dtaus_dT()

        Gs = self.Gs()
        dGs_dT = self.dGs_dT()
        if self.scalar:
            d2GE_dTdxs = [0.0]*N
        else:
            d2GE_dTdxs = zeros(N)

        self._d2GE_dTdxs = nrtl_d2GE_dTdxs(N, T, xs, taus, dtaus_dT, Gs, dGs_dT, xj_Gs_taus_jis,
                    xj_Gs_jis_inv, xj_dGs_dT_jis, xj_taus_dGs_dT_jis,
                    xj_Gs_dtaus_dT_jis, d2GE_dTdxs)
        return d2GE_dTdxs




def NRTL_gammas(xs, taus, alphas):
    r'''Calculates the activity coefficients of each species in a mixture
    using the Non-Random Two-Liquid (NRTL) method, given their mole fractions,
    dimensionless interaction parameters, and nonrandomness constants. Those
    are normally correlated with temperature in some form, and need to be
    calculated separately.

    .. math::
        \ln(\gamma_i)=\frac{\displaystyle\sum_{j=1}^{n}{x_{j}\tau_{ji}G_{ji}}}
        {\displaystyle\sum_{k=1}^{n}{x_{k}G_{ki}}}+\sum_{j=1}^{n}
        {\frac{x_{j}G_{ij}}{\displaystyle\sum_{k=1}^{n}{x_{k}G_{kj}}}}
        {\left ({\tau_{ij}-\frac{\displaystyle\sum_{m=1}^{n}{x_{m}\tau_{mj}
        G_{mj}}}{\displaystyle\sum_{k=1}^{n}{x_{k}G_{kj}}}}\right )}

    .. math::
        G_{ij}=\text{exp}\left ({-\alpha_{ij}\tau_{ij}}\right )

    Parameters
    ----------
    xs : list[float]
        Liquid mole fractions of each species, [-]
    taus : list[list[float]]
        Dimensionless interaction parameters of each compound with each other,
        [-]
    alphas : list[list[float]]
        Nonrandomness constants of each compound interacting with each other, [-]

    Returns
    -------
    gammas : list[float]
        Activity coefficient for each species in the liquid mixture, [-]

    Notes
    -----
    This model needs N^2 parameters.

    One common temperature dependence of the nonrandomness constants is:

    .. math::
        \alpha_{ij}=c_{ij}+d_{ij}T

    Most correlations for the interaction parameters include some of the terms
    shown in the following form:

    .. math::
        \tau_{ij}=A_{ij}+\frac{B_{ij}}{T}+\frac{C_{ij}}{T^{2}}+D_{ij}
        \ln{\left ({T}\right )}+E_{ij}T^{F_{ij}}

    The original form of this model used the temperature dependence of taus in
    the form (values can be found in the literature, often with units of
    calories/mol):

    .. math::
        \tau_{ij}=\frac{b_{ij}}{RT}

    For this model to produce ideal acitivty coefficients (gammas = 1),
    all interaction parameters should be 0; the value of alpha does not impact
    the calculation when that is the case.

    Examples
    --------
    Ethanol-water example, at 343.15 K and 1 MPa:

    >>> NRTL_gammas(xs=[0.252, 0.748], taus=[[0, -0.178], [1.963, 0]],
    ... alphas=[[0, 0.2974],[.2974, 0]])
    [1.9363183763514304, 1.1537609663170014]

    References
    ----------
    .. [1] Renon, Henri, and J. M. Prausnitz. "Local Compositions in
       Thermodynamic Excess Functions for Liquid Mixtures." AIChE Journal 14,
       no. 1 (1968): 135-144. doi:10.1002/aic.690140124.
    .. [2] Gmehling, Jurgen, Barbel Kolbe, Michael Kleiber, and Jurgen Rarey.
       Chemical Thermodynamics for Process Simulation. 1st edition. Weinheim:
       Wiley-VCH, 2012.
    '''
    gammas = []
    cmps = range(len(xs))
    # Gs does not depend on composition
    Gs = []
    for i in cmps:
        alphasi = alphas[i]
        tausi = taus[i]
        Gs.append([exp(-alphasi[j]*tausi[j]) for j in cmps])


    td2s = []
    tn3s = []
    for j in cmps:
        td2 = 0.0
        tn3 = 0.0
        for k in cmps:
            xkGkj = xs[k]*Gs[k][j]
            td2 += xkGkj
            tn3 += xkGkj*taus[k][j]
        td2 = 1.0/td2
        td2xj = td2*xs[j]
        td2s.append(td2xj)
        tn3s.append(tn3*td2*td2xj)

    for i in cmps:
        tn1, td1, total2 = 0., 0., 0.
        Gsi = Gs[i]
        tausi = taus[i]
        for j in cmps:
            xjGji = xs[j]*Gs[j][i]

            td1 += xjGji
            tn1 += xjGji*taus[j][i]
            total2 += Gsi[j]*(tausi[j]*td2s[j] - tn3s[j])

        gamma = exp(tn1/td1 + total2)
        gammas.append(gamma)
    return gammas