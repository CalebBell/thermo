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

NRTL Regression Calculations
============================
.. autofunction:: NRTL_gammas_binaries

'''

from math import exp, log

from fluids.constants import R
from fluids.numerics import numpy as np
from fluids.numerics import transpose, trunc_exp

from thermo.activity import GibbsExcess

__all__ = ['NRTL', 'NRTL_gammas', 'NRTL_gammas_binaries', 'NRTL_gammas_binaries_jac']

try:
    array, zeros, ones, delete, npsum, nplog, nptranspose, ascontiguousarray = np.array, np.zeros, np.ones, np.delete, np.sum, np.log, np.transpose, np.ascontiguousarray
except (ImportError, AttributeError):
    pass

def nrtl_gammas_from_args(xs, N, Gs, taus, Gs_transposed, Gs_taus_transposed, Gs_taus, xj_Gs_jis=None, xj_Gs_taus_jis=None, vec0=None, vec1=None, gammas=None):
    if xj_Gs_jis is None:
        xj_Gs_jis = [0.0]*N
    if xj_Gs_taus_jis is None:
        xj_Gs_taus_jis = [0.0]*N
    nrtl_xj_Gs_jis_and_Gs_taus_jis(N, xs, Gs, taus, Gs_transposed, Gs_taus_transposed, xj_Gs_jis, xj_Gs_taus_jis)
    for i in range(N):
        # We can reuse the same list instead of making a new one here for xj_Gs_jis_inv
        xj_Gs_jis[i] = 1.0/xj_Gs_jis[i]
    return nrtl_gammas(xs, N, Gs, taus, xj_Gs_jis, xj_Gs_taus_jis, gammas, vec0=vec0, vec1=vec1)

def nrtl_gammas(xs, N, Gs, taus, xj_Gs_jis_inv, xj_Gs_taus_jis, gammas, vec0=None, vec1=None):
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

    for i in range(N):
        tot = vec1[i]
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

def nrtl_xj_Gs_jis_and_Gs_taus_jis(N, xs, Gs, taus, Gs_transposed, Gs_taus_transposed, xj_Gs_jis=None, xj_Gs_taus_jis=None):
    if xj_Gs_jis is None:
        xj_Gs_jis = [0.0]*N
    if xj_Gs_taus_jis is None:
        xj_Gs_taus_jis = [0.0]*N

    for i in range(N):
        tot1 = 0.0
        Gs_row = Gs_transposed[i]
        for j in range(N):
            xjGji = xs[j]*Gs_row[j]
            tot1 += xjGji
        xj_Gs_jis[i] = tot1

    for i in range(N):
        tot2 = 0.0
        Gs_taus_transposed_row = Gs_taus_transposed[i]
        for j in range(N):
            tot2 += xs[j]*Gs_taus_transposed_row[j]
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

def nrtl_d2GE_dT2(N, T, xs, taus, dtaus_dT, d2taus_dT2, Gs, dGs_dT, d2Gs_dT2):
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

def nrtl_d2GE_dxixjs(N, T, xs, taus, Gs, xj_Gs_taus_jis, xj_Gs_jis_inv, d2GE_dxixjs=None):
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

    .. math::
        g^E = RT\sum_i x_i \frac{\sum_j \tau_{ji} G_{ji} x_j}
        {\sum_j G_{ji}x_j}

    .. math::
        G_{ij}=\exp(-\alpha_{ij}\tau_{ij})

    .. math::
        \alpha_{ij}=c_{ij} + d_{ij}T

    .. math::
        \tau_{ij}=A_{ij}+\frac{B_{ij}}{T}+E_{ij}\ln T + F_{ij}T
        + \frac{G_{ij}}{T^2} + H_{ij}{T^2}

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
        `ABEFGHCD` or some of the `tau` or `alpha` parameters are required, [-]
    tau_as : list[list[float]], optional
        `a` parameters used in calculating :obj:`NRTL.taus`, [-]
    tau_bs : list[list[float]], optional
        `b` parameters used in calculating :obj:`NRTL.taus`, [K]
    tau_es : list[list[float]], optional
        `e` parameters used in calculating :obj:`NRTL.taus`, [-]
    tau_fs : list[list[float]], optional
        `f` paraemeters used in calculating :obj:`NRTL.taus`, [1/K]
    tau_gs : list[list[float]], optional
        `e` parameters used in calculating :obj:`NRTL.taus`, [K^2]
    tau_hs : list[list[float]], optional
        `f` parameters used in calculating :obj:`NRTL.taus`, [1/K^2]
    alpha_cs : list[list[float]], optional
        `c` parameters used in calculating :obj:`NRTL.alphas`, [-]
    alpha_ds : list[list[float]], optional
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
    binary ethanol-water system, Example P05.01b in [2]_, shows how to use parameters from
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
    NRTL(T=343.15, xs=[0.252, 0.748], tau_bs=[[0, -61.0249799309399], [673.2359767282798, 0]], alpha_cs=[[0, 0.2974], [0.2974, 0]])
    >>> GE.GE(), GE.dGE_dT(), GE.d2GE_dT2()
    (780.053057219, 0.5743500022, -0.003584843605528)
    >>> GE.HE(), GE.SE(), GE.dHE_dT(), GE.dSE_dT()
    (582.964853938, -0.57435000227, 1.230139083237, 0.0035848436055)

    The solution given by the DDBST has the same values [1.936, 1.154],
    and can be found here:
    http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/P05.01b%20VLE%20Behavior%20of%20Ethanol%20-%20Water%20Using%20NRTL.xps

    References
    ----------
    .. [1] Poling, Bruce E., John M. Prausnitz, and John P. O`Connell. The
       Properties of Gases and Liquids. 5th edition. New York: McGraw-Hill
       Professional, 2000.
    .. [2] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen Rarey.
       Chemical Thermodynamics for Process Simulation. John Wiley & Sons, 2019.
    '''

    _model_attributes = ('tau_as', 'tau_bs', 'tau_es', 'tau_fs',
                         'tau_gs', 'tau_hs', 'alpha_cs', 'alpha_ds')
    model_id = 100

    stored_attributes = _model_attributes + ('alpha_temperature_independent', 'tau_coeffs_nonzero','zero_coeffs')
    _cached_calculated_attributes = ('_alphas', '_dGs_dT', '_xj_dGs_dT_jis', '_xj_Gs_dtaus_dT_jis', '_d2Gs_dT2', '_xj_taus_dGs_dT_jis', '_dtaus_dT', '_Gs', '_d2taus_dT2',
                '_taus',  '_xj_Gs_jis_inv', '_xj_Gs_taus_jis', '_xj_Gs_jis',
                '_d3taus_dT3', '_d3Gs_dT3', '_Gs_transposed', '_Gs_taus_transposed', '_Gs_taus')
    __slots__ = GibbsExcess.__slots__ + stored_attributes + _cached_calculated_attributes
    recalculable_attributes = _cached_calculated_attributes + GibbsExcess.recalculable_attributes



    def gammas_args(self, T=None):
        if T is not None:
            obj = self.to_T_xs(T=T, xs=self.xs)
        else:
            obj = self
        try:
            taus = obj._taus
        except AttributeError:
            taus = obj.taus()
        try:
            Gs = obj._Gs
        except AttributeError:
            Gs = obj.Gs()
        Gs_taus_transposed = obj.Gs_taus_transposed()
        Gs_transposed = obj.Gs_transposed()
        Gs_taus = obj.Gs_taus()

        N = obj.N
        if not self.vectorized:
            xj_Gs_jis, xj_Gs_taus_jis, vec0, vec1 = [0.0]*N, [0.0]*N, [0.0]*N, [0.0]*N
        else:
            xj_Gs_jis, xj_Gs_taus_jis, vec0, vec1 = zeros(N), zeros(N), zeros(N),  zeros(N)

        return (N, Gs, taus, Gs_transposed, Gs_taus_transposed, Gs_taus, xj_Gs_jis, xj_Gs_taus_jis, vec0, vec1)

    gammas_from_args = staticmethod(nrtl_gammas_from_args)

    def __init__(self, *, xs, T=GibbsExcess.T_DEFAULT, tau_coeffs=None, alpha_coeffs=None,
                 ABEFGHCD=None, tau_as=None, tau_bs=None, tau_es=None,
                 tau_fs=None, tau_gs=None, tau_hs=None, alpha_cs=None,
                 alpha_ds=None):
        self.T = T
        self.xs = xs
        self.vectorized = vectorized = type(xs) is not list
        self.N = N = len(xs)

        if not self.vectorized:
            self.zero_coeffs = [[0.0]*N for _ in range(N)]
        else:
            self.zero_coeffs = zeros((N, N))

        multiple_inputs = (tau_as, tau_bs, tau_es, tau_fs, tau_gs, tau_hs,
                        alpha_cs, alpha_ds)

        input_count = ((tau_coeffs is not None or alpha_coeffs is not None)
                       + (ABEFGHCD is not None) + (any(i is not None for i in multiple_inputs)))
        if input_count > 1:
            raise ValueError("Input only one of (tau_coeffs, alpha_coeffs), ABEFGHCD, or (tau_as...alpha_ds)")

        if ABEFGHCD is None:
            ABEFGHCD = multiple_inputs

        if tau_coeffs is not None and alpha_coeffs is not None:
            pass
        else:
            try:
                all_lengths = tuple(len(coeffs) for coeffs in ABEFGHCD if coeffs is not None)
                if len(set(all_lengths)) > 1:
                    raise ValueError(f"Coefficient arrays of different size found: {all_lengths}")
                all_lengths_inner = tuple(len(coeffs[0]) for coeffs in ABEFGHCD if coeffs is not None)
                if len(set(all_lengths_inner)) > 1:
                    raise ValueError(f"Coefficient arrays of different size found: {all_lengths_inner}")
            except:
                raise ValueError("Coefficients not input correctly")

        if tau_coeffs is not None and alpha_coeffs is not None:
            if not vectorized:
                self.tau_as = [[i[0] for i in l] for l in tau_coeffs]
                self.tau_bs = [[i[1] for i in l] for l in tau_coeffs]
                self.tau_es = [[i[2] for i in l] for l in tau_coeffs]
                self.tau_fs = [[i[3] for i in l] for l in tau_coeffs]
                self.tau_gs = [[i[4] for i in l] for l in tau_coeffs]
                self.tau_hs = [[i[5] for i in l] for l in tau_coeffs]
            else:
                self.tau_as = array(tau_coeffs[:,:,0], order='C', copy=True)
                self.tau_bs = array(tau_coeffs[:,:,1], order='C', copy=True)
                self.tau_es = array(tau_coeffs[:,:,2], order='C', copy=True)
                self.tau_fs = array(tau_coeffs[:,:,3], order='C', copy=True)
                self.tau_gs = array(tau_coeffs[:,:,4], order='C', copy=True)
                self.tau_hs = array(tau_coeffs[:,:,5], order='C', copy=True)
            if not vectorized:
                self.alpha_cs = [[i[0] for i in l] for l in alpha_coeffs]
                self.alpha_ds = [[i[1] for i in l] for l in alpha_coeffs]
            else:
                self.alpha_cs = array(alpha_coeffs[:,:,0], order='C', copy=True)
                self.alpha_ds = array(alpha_coeffs[:,:,1], order='C', copy=True)

        else:
            if ABEFGHCD[0] is None:
                self.tau_as = self.zero_coeffs
            else:
                self.tau_as = ABEFGHCD[0]

            if ABEFGHCD[1] is None:
                self.tau_bs = self.zero_coeffs
            else:
                self.tau_bs = ABEFGHCD[1]

            if ABEFGHCD[2] is None:
                self.tau_es = self.zero_coeffs
            else:
                self.tau_es = ABEFGHCD[2]

            if ABEFGHCD[3] is None:
                self.tau_fs = self.zero_coeffs
            else:
                self.tau_fs = ABEFGHCD[3]

            if ABEFGHCD[4] is None:
                self.tau_gs = self.zero_coeffs
            else:
                self.tau_gs = ABEFGHCD[4]

            if ABEFGHCD[5] is None:
                self.tau_hs = self.zero_coeffs
            else:
                self.tau_hs = ABEFGHCD[5]

            if ABEFGHCD[6] is None:
                self.alpha_cs = self.zero_coeffs
            else:
                self.alpha_cs = ABEFGHCD[6]

            if ABEFGHCD[7] is None:
                self.alpha_ds = self.zero_coeffs
            else:
                self.alpha_ds = ABEFGHCD[7]

        # Make an array of values identifying what coefficients are zero.
        # This may be useful for performance optimization in the future but is
        # especially important for reducing the size of the __repr__ string.
        self.tau_coeffs_nonzero = tau_coeffs_nonzero = [True]*6 if not vectorized else ones(6, bool)
        for k, coeffs in enumerate([self.tau_as, self.tau_bs, self.tau_es,
                                    self.tau_fs, self.tau_gs, self.tau_hs]):
            nonzero = False
            for i in range(N):
                r = coeffs[i]
                for j in range(N):
                    if r[j] != 0.0:
                        nonzero = True
                        break
                if nonzero:
                    break
            tau_coeffs_nonzero[k] = nonzero


        alpha_ds = self.alpha_ds
        alpha_temperature_independent = True
        for i in range(N):
            r = alpha_ds[i]
            for j in range(N):
                if r[j] != 0.0:
                    alpha_temperature_independent = False
        self.alpha_temperature_independent = alpha_temperature_independent



    def __repr__(self):
        s = f'{self.__class__.__name__}(T={self.T!r}, xs={self.xs!r}'
        for i, attr in enumerate(self._model_attributes[:6]):
            if self.tau_coeffs_nonzero[i]:
                s += f', {attr}={getattr(self, attr)}'
        s += f', alpha_cs={self.alpha_cs}'
        if not self.alpha_temperature_independent:
            s += f', alpha_ds={self.alpha_ds}'
        s += ')'
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
        (new.T, new.xs, new.N, new.vectorized) = T, xs, self.N, self.vectorized
        (new.tau_as, new.tau_bs, new.tau_es,
         new.tau_fs, new.tau_gs, new.tau_hs,
         new.alpha_cs, new.alpha_ds, new.tau_coeffs_nonzero,
         new.alpha_temperature_independent) = (self.tau_as, self.tau_bs, self.tau_es,
                         self.tau_fs, self.tau_gs, self.tau_hs,
                         self.alpha_cs, self.alpha_ds, self.tau_coeffs_nonzero, self.alpha_temperature_independent)

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

        new.zero_coeffs = self.zero_coeffs
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

        if not self.vectorized:
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
        A = self.tau_as
        B = self.tau_bs
        E = self.tau_es
        F = self.tau_fs
        G = self.tau_gs
        H = self.tau_hs
        T, N = self.T, self.N
        if not self.vectorized:
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
        B = self.tau_bs
        E = self.tau_es
        F = self.tau_fs
        G = self.tau_gs
        H = self.tau_hs
        T, N = self.T, self.N
        if not self.vectorized:
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
        B = self.tau_bs
        E = self.tau_es
        G = self.tau_gs
        H = self.tau_hs
        T, N = self.T, self.N

        if not self.vectorized:
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
        B = self.tau_bs
        E = self.tau_es
        G = self.tau_gs
        T, N = self.T, self.N
        if not self.vectorized:
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

        if self.alpha_temperature_independent:
            self._alphas = alphas = self.alpha_cs
        else:
            if not self.vectorized:
                alphas = [[0.0]*N for _ in range(N)]
            else:
                alphas = zeros((N, N))

            self._alphas = nrtl_alphas(self.T, N, self.alpha_cs, self.alpha_ds, alphas)
        return alphas

    def dalphas_dT(self):
        '''Keep it as a function in case this needs to become more complicated.'''
        return self.alpha_ds

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

        if not self.vectorized:
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

        if not self.vectorized:
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

        if not self.vectorized:
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

        if not self.vectorized:
            d3Gs_dT3 = [[0.0]*N for _ in range(N)]
        else:
            d3Gs_dT3 = zeros((N, N))

        self._d3Gs_dT3 = nrtl_d3Gs_dT3(N, alphas, dalphas_dT, taus, dtaus_dT, d2taus_dT2, d3taus_dT3, Gs, d3Gs_dT3)
        return d3Gs_dT3

    def Gs_transposed(self):
        # For performance
        try:
            return self._Gs_transposed
        except AttributeError:
            pass
        if not self.vectorized:
            self._Gs_transposed = transpose(self.Gs())
        else:
            self._Gs_transposed = ascontiguousarray(nptranspose(self.Gs()))

        return self._Gs_transposed

    def Gs_taus_transposed(self):
        # For performance
        try:
            return self._Gs_taus_transposed
        except AttributeError:
            pass
        if not self.vectorized:
            mat = transpose(self.taus())
            Gs_transposed = self.Gs_transposed()
            N = self.N
            for i in range(N):
                for j in range(N):
                    mat[i][j] *= Gs_transposed[i][j]
            self._Gs_taus_transposed = mat
        else:
            self._Gs_taus_transposed = ascontiguousarray(nptranspose(self.taus())*self.Gs_transposed())

        return self._Gs_taus_transposed

    def Gs_taus(self):
        # For performance
        try:
            return self._Gs_taus
        except AttributeError:
            pass
        if not self.vectorized:
            N = self.N
            Gs_taus = [[0.0]*N for _ in range(N)]
            taus = self.taus()
            Gs = self.Gs()
            for i in range(N):
                for j in range(N):
                    Gs_taus[i][j] = taus[i][j]*Gs[i][j]
            self._Gs_taus = Gs_taus
        else:
            self._Gs_taus = self.taus()*self.Gs()
        return self._Gs_taus

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
        try:
            Gs_transposed = self._Gs_transposed
        except AttributeError:
            Gs_transposed = self.Gs_transposed()
        try:
            Gs_taus_transposed = self._Gs_taus_transposed
        except AttributeError:
            Gs_taus_transposed = self.Gs_taus_transposed()

        xs, N = self.xs, self.N
        if not self.vectorized:
            _xj_Gs_jis = [0.0]*N
            _xj_Gs_taus_jis = [0.0]*N
        else:
            _xj_Gs_jis = zeros(N)
            _xj_Gs_taus_jis = zeros(N)

        nrtl_xj_Gs_jis_and_Gs_taus_jis(N, xs, Gs, taus, Gs_transposed, Gs_taus_transposed, _xj_Gs_jis, _xj_Gs_taus_jis)

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

        if not self.vectorized:
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
        if not self.vectorized:
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
        if not self.vectorized:
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

        if not self.vectorized:
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

        if not self.vectorized:
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
        try:
            xj_Gs_jis_inv = self._xj_Gs_jis_inv
        except:
            xj_Gs_jis_inv = self.xj_Gs_jis_inv()
        try:
            xj_Gs_taus_jis = self._xj_Gs_taus_jis
        except:
            xj_Gs_taus_jis = self.xj_Gs_taus_jis()
        self._GE = GE = nrtl_GE(self.N, self.T, self.xs, xj_Gs_taus_jis, xj_Gs_jis_inv)
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
        """from sympy import *
        R, T, x = symbols('R, T, x')
        g, tau = symbols('g, tau', cls=Function)
        m, n, o = symbols('m, n, o', cls=Function)
        r, s, t = symbols('r, s, t', cls=Function)
        u, v, w = symbols('u, v, w', cls=Function)
        diff(T* (m(T)*n(T) + r(T)*s(T) + u(T)*v(T))/(o(T) + t(T) + w(T)), T)
        """
        try:
            return self._dGE_dT
        except AttributeError:
            pass
        T, xs, N = self.T, self.xs, self.N
        try:
            xj_Gs_jis_inv = self._xj_Gs_jis_inv
        except:
            xj_Gs_jis_inv = self.xj_Gs_jis_inv() # sum1 inv
        try:
            xj_Gs_taus_jis = self._xj_Gs_taus_jis
        except:
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
        """from sympy import *
        R, T, x = symbols('R, T, x')
        g, tau = symbols('g, tau', cls=Function)
        m, n, o = symbols('m, n, o', cls=Function)
        r, s, t = symbols('r, s, t', cls=Function)
        u, v, w = symbols('u, v, w', cls=Function)

        (diff(T*(m(T)*n(T) + r(T)*s(T))/(o(T) + t(T)), T, 2))
        """
        try:
            return self._d2GE_dT2
        except AttributeError:
            pass
        T, xs, N = self.T, self.xs, self.N
        taus = self.taus()
        dtaus_dT = self.dtaus_dT()
        d2taus_dT2 = self.d2taus_dT2()

        Gs = self.Gs()
        dGs_dT = self.dGs_dT()
        d2Gs_dT2 = self.d2Gs_dT2()

        self._d2GE_dT2 = d2GE_dT2 = nrtl_d2GE_dT2(N, T, xs, taus, dtaus_dT, d2taus_dT2, Gs, dGs_dT, d2Gs_dT2)
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
        """
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
        """
        try:
            return self._dGE_dxs
        except AttributeError:
            pass
        T, xs, N = self.T, self.xs, self.N
        taus = self.taus()
        Gs = self.Gs()
        xj_Gs_jis_inv = self.xj_Gs_jis_inv()
        xj_Gs_taus_jis = self.xj_Gs_taus_jis()
        if not self.vectorized:
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
        """
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
        """
        try:
            return self._d2GE_dxixjs
        except AttributeError:
            pass
        T, xs, N = self.T, self.xs, self.N
        taus = self.taus()
        Gs = self.Gs()
        xj_Gs_jis_inv = self.xj_Gs_jis_inv()
        xj_Gs_taus_jis = self.xj_Gs_taus_jis()
        if not self.vectorized:
            d2GE_dxixjs = [[0.0]*N for _ in range(N)]
        else:
            d2GE_dxixjs = zeros((N, N))

        self._d2GE_dxixjs = nrtl_d2GE_dxixjs(N, T, xs, taus, Gs, xj_Gs_taus_jis, xj_Gs_jis_inv, d2GE_dxixjs)
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
        if not self.vectorized:
            d2GE_dTdxs = [0.0]*N
        else:
            d2GE_dTdxs = zeros(N)

        self._d2GE_dTdxs = nrtl_d2GE_dTdxs(N, T, xs, taus, dtaus_dT, Gs, dGs_dT, xj_Gs_taus_jis,
                    xj_Gs_jis_inv, xj_dGs_dT_jis, xj_taus_dGs_dT_jis,
                    xj_Gs_dtaus_dT_jis, d2GE_dTdxs)
        return d2GE_dTdxs


    @classmethod
    def regress_binary_parameters(cls, gammas, xs, symmetric_alphas=False,
                                  use_numba=False, force_alpha=None,
                                  do_statistics=True, **kwargs):
        # Load the functions either locally or with numba
        if use_numba:
            from thermo.numba import NRTL_gammas_binaries as work_func
            from thermo.numba import NRTL_gammas_binaries_jac as jac_func
        else:
            work_func = NRTL_gammas_binaries
            jac_func = NRTL_gammas_binaries_jac

        if force_alpha is not None:
            symmetric_alphas = True
        # Allocate all working memory
        pts = len(xs)
        gammas_iter = zeros(pts*2)
        jac_iter = zeros((pts*2, 4))

        # Plain objective functions
        if symmetric_alphas:
            def fitting_func(xs, tau12, tau21, alpha=force_alpha):
                return work_func(xs, tau12, tau21, alpha, alpha, gammas_iter)
            def analytical_jac(xs, tau12, tau21, alpha=force_alpha):
                out = jac_func(xs, tau12, tau21, alpha, alpha, jac_iter)
                if force_alpha is not None:
                    return delete(out, [2,3], axis=1)
                return delete(out, 3, axis=1)

        else:
            def fitting_func(xs, tau12, tau21, alpha12, alpha21):
                return work_func(xs, tau12, tau21, alpha12, alpha21, gammas_iter)

            def analytical_jac(xs, tau12, tau21, alpha12, alpha21):
                return jac_func(xs, tau12, tau21, alpha12, alpha21, jac_iter)

        # The extend calls has been tested to be the fastest compared to numpy and list comprehension
        xs_working = []
        for xsi in xs:
            xs_working.extend(xsi)
        gammas_working = []
        for gammasi in gammas:
            gammas_working.extend(gammasi)

        xs_working = array(xs_working)
        gammas_working = array(gammas_working)

        # Objective functions for leastsq maximum speed
        if symmetric_alphas:
            def func_wrapped_for_leastsq(params):
                alpha = params[2] if force_alpha is None else force_alpha
                return work_func(xs_working, params[0], params[1], alpha, alpha, gammas_iter) - gammas_working

            def jac_wrapped_for_leastsq(params):
                alpha = params[2] if force_alpha is None else force_alpha
                out = jac_func(xs_working, params[0], params[1], alpha, alpha, jac_iter)
                if force_alpha is None:
                    new_out = delete(out, 3, axis=1)
                    new_out[:, 2] = out[:, 2] + out[:, 3]
                else:
                    new_out = delete(out, [2,3], axis=1)
                return new_out

        else:
            def func_wrapped_for_leastsq(params):
                return work_func(xs_working, params[0], params[1], params[2], params[3], gammas_iter) - gammas_working

            def jac_wrapped_for_leastsq(params):
                return jac_func(xs_working, params[0], params[1], params[2], params[3], jac_iter)

        if symmetric_alphas:
            if force_alpha is not None:
                use_fit_parameters = ['tau12', 'tau21']
            else:
                use_fit_parameters = ['tau12', 'tau21', 'alpha12']
        else:
            use_fit_parameters = ['tau12', 'tau21', 'alpha12', 'alpha21']
        return GibbsExcess._regress_binary_parameters(gammas_working, xs_working, fitting_func=fitting_func,
                                                      fit_parameters=use_fit_parameters,
                                                      use_fit_parameters=use_fit_parameters,
                                                      initial_guesses=cls._gamma_parameter_guesses,
                                                        analytical_jac=jac_func,
                                                       # analytical_jac=None,
                                                      use_numba=use_numba,
                                                      do_statistics=do_statistics,
                                                      func_wrapped_for_leastsq=func_wrapped_for_leastsq,
                                                        jac_wrapped_for_leastsq=jac_wrapped_for_leastsq,
                                                       # jac_wrapped_for_leastsq=None,
                                                      **kwargs)

    # Larger value on the right always (tau)
    # Alpha will almost exclusively be fit as one parameter
    _gamma_parameter_guesses = [{'tau12': 1, 'tau21': 1, 'alpha12': 0.2, 'alpha21': 0.2},
                                {'tau12': 1, 'tau21': 1, 'alpha12': 0.3, 'alpha21': 0.3},
                                {'tau12': 1, 'tau21': 1, 'alpha12': 0.47, 'alpha21': 0.47},
                               ]

    for i in range(len(_gamma_parameter_guesses)):
        r = _gamma_parameter_guesses[i]
        # Swap the taus
        _gamma_parameter_guesses.append({'tau12': r['tau21'], 'tau21': r['tau12'], 'alpha12': r['alpha12'], 'alpha21': r['alpha21']})
        # Swap the alphas
        _gamma_parameter_guesses.append({'tau12': r['tau12'], 'tau21': r['tau21'], 'alpha12': r['alpha21'], 'alpha21': r['alpha12']})
        # Swap both
        _gamma_parameter_guesses.append({'tau12': r['tau21'], 'tau21': r['tau12'], 'alpha12': r['alpha21'], 'alpha21': r['alpha12']})
    del i, r

MIN_TAU_NRTL = -1e100
MIN_ALPHA_NRTL = 1e-10

def NRTL_gammas_binaries(xs, tau12, tau21, alpha12, alpha21, gammas=None):
    r'''Calculates activity coefficients at fixed `tau` and `alpha` values for
    a binary system at a series of mole fractions. This is used for
    regression of `tau` and `alpha` parameters. This function is highly optimized,
    and operates on multiple points at a time.

    .. math::
        \ln \gamma_1 = x_2^2\left[\tau_{21}\left(\frac{G_{21}}{x_1 + x_2 G_{21}}
            \right)^2 + \frac{\tau_{12}G_{12}}{(x_2 + x_1G_{12})^2}
            \right]

    .. math::
        \ln \gamma_2 =  x_1^2\left[\tau_{12}\left(\frac{G_{12}}{x_2 + x_1 G_{12}}
            \right)^2 + \frac{\tau_{21}G_{21}}{(x_1 + x_2 G_{21})^2}
            \right]

    .. math::
        G_{ij}=\exp(-\alpha_{ij}\tau_{ij})


    Parameters
    ----------
    xs : list[float]
        Liquid mole fractions of each species in the format
        x0_0, x1_0, (component 1 point1, component 2 point 1),
        x0_1, x1_1, (component 1 point2, component 2 point 2), ...
        [-]
    tau12 : float
        `tau` parameter for 12, [-]
    tau21 : float
        `tau` parameter for 21, [-]
    alpha12 : float
        `alpha` parameter for 12, [-]
    alpha21 : float
        `alpha` parameter for 21, [-]
    gammas : list[float], optional
        Array to store the activity coefficient for each species in the liquid
        mixture, indexed the same as `xs`; can be omitted or provided
        for slightly better performance [-]

    Returns
    -------
    gammas : list[float]
        Activity coefficient for each species in the liquid mixture,
        indexed the same as `xs`, [-]

    Notes
    -----

    Examples
    --------
    >>> NRTL_gammas_binaries([.1, .9, 0.3, 0.7, .85, .15], 0.1759, 0.7991, .2, .3)
    [2.121421, 1.011342, 1.52177, 1.09773, 1.016062, 1.841391]
    '''
    if tau12 < MIN_TAU_NRTL:
        tau12 = MIN_TAU_NRTL
    if tau21 < MIN_TAU_NRTL:
        tau21 = MIN_TAU_NRTL
    if alpha12 < MIN_ALPHA_NRTL:
        alpha12 = MIN_ALPHA_NRTL
    if alpha21 < MIN_ALPHA_NRTL:
        alpha21 = MIN_ALPHA_NRTL

    pts = len(xs)//2 # Always even

    if gammas is None:
        allocate_size = (pts*2)
        gammas = [0.0]*allocate_size

    tau01, tau10, alpha01, alpha10 = tau12, tau21, alpha12, alpha21

    G01 = exp(-alpha01*tau01)
    G10 = exp(-alpha10*tau10)

    G10_2_tau10 = G10*G10*tau10
    G10_tau10 = G10*tau10

    G01_2_tau01 = G01*G01*tau01
    G01_tau01 = G01*tau01

    for i in range(pts):
        i2 = i*2
        x0 = xs[i2]
        x1 = 1.0 - x0

        c0 = 1.0/(x0 + x1*G10)
        c0 *= c0

        c1 = 1.0/(x1 + x0*G01)
        c1 *= c1

        gamma0 = trunc_exp(x1*x1*(G10_2_tau10*c0 + G01_tau01*c1))
        gamma1 = trunc_exp(x0*x0*(G01_2_tau01*c1 + G10_tau10*c0))


        gammas[i2] = gamma0
        gammas[i2 + 1] = gamma1
    return gammas


def NRTL_gammas_binaries_jac(xs, tau12, tau21, alpha12, alpha21, calc=None):
    if tau12 < MIN_TAU_NRTL:
        tau12 = MIN_TAU_NRTL
    if tau21 < MIN_TAU_NRTL:
        tau21 = MIN_TAU_NRTL
    if alpha12 < MIN_ALPHA_NRTL:
        alpha12 = MIN_ALPHA_NRTL
    if alpha21 < MIN_ALPHA_NRTL:
        alpha21 = MIN_ALPHA_NRTL
    pts = len(xs)//2 # Always even

    tau01, tau10, alpha01, alpha10 = tau12, tau21, alpha12, alpha21

    if calc is None:
        allocate_size = pts*2
        calc = np.zeros((allocate_size, 4))

    x2 = alpha01*tau01
    x3 = trunc_exp(x2)
    x11 = alpha10*tau10
    x12 = trunc_exp(x11)
    x26 = tau01*tau01
    x27 = tau10*tau10

    for i in range(pts):
        i2 = i*2
        x0 = xs[i2]
        x1 = 1.0 - x0


        x4 = x1*x3
        x5 = x0 + x4
        x5_inv = 1.0/x5
        x6 = 2.0*x4
        x7 = x6*x5_inv
        x8 = x2*x7
        x9 = x5_inv*x5_inv
        x10 = x1*x1
        x13 = x0*x12
        x14 = x1 + x13
        x14_inv = 1.0/x14
        x15 = x14_inv*x14_inv
        x16 = tau10*x15
        x17 = tau01*x9
        x18 = x10*trunc_exp(x10*(x16 + x17*x3))
        x19 = x18*x3*x9
        x20 = x0*x0
        x21 = x20*trunc_exp(x20*(x12*x16 + x17))
        x22 = 2.0*x13
        x23 = x22*x14_inv
        x24 = x11*x23
        x25 = x12*x15*x21


        gamma0_row = calc[i2]
        gamma0_row[0] = x19*(x2 - x8 + 1.0)# Right
        gamma0_row[1] = -x15*x18*(x24 - 1.0) # Right
        gamma0_row[2] = -x19*x26*(x7 - 1.0) # Right
        gamma0_row[3] =-x18*x22*x27*x15*x14_inv

        gamma1_row = calc[i2+1]
        gamma1_row[0] = -x21*x9*(x8 - 1.0)
        gamma1_row[1] = x25*(x11 - x24 + 1.0)
        gamma1_row[2] = -x21*x26*x6*x5_inv*x9
        gamma1_row[3] = -x25*x27*(x23 - 1.0)
    return calc

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
