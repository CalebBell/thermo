'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains a class :obj:`UNIQUAC` for performing activity coefficient
calculations with the UNIQUAC model. An older, functional calculation for
activity coefficients only is also present, :obj:`UNIQUAC_gammas`.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

UNIQUAC Class
=============

.. autoclass:: UNIQUAC
    :members: to_T_xs, GE, dGE_dT, d2GE_dT2, d3GE_dT3, d2GE_dTdxs, dGE_dxs,
              d2GE_dxixjs, taus, dtaus_dT, d2taus_dT2, d3taus_dT3, phis,
              thetas, regress_binary_parameters
    :undoc-members:
    :show-inheritance:
    :exclude-members:

UNIQUAC Functional Calculations
===============================
.. autofunction:: UNIQUAC_gammas
'''

from math import exp, log

from fluids.constants import R
from fluids.numerics import numpy as np
from fluids.numerics import trunc_exp

from thermo.activity import GibbsExcess, d2interaction_exp_dT2, d3interaction_exp_dT3, dinteraction_exp_dT, gibbs_excess_gammas, interaction_exp

__all__ = ['UNIQUAC', 'UNIQUAC_gammas', 'UNIQUAC_gammas_binary', 'UNIQUAC_gammas_binaries']

try:
    array, zeros, npsum, nplog = np.array, np.zeros, np.sum, np.log
except (ImportError, AttributeError):
    pass

def uniquac_phis(N, xs, rs, phis=None):
    if phis is None:
        phis = [0.0]*N

    rsxs_sum_inv = 0.0
    for i in range(N):
        phis[i] = rs[i]*xs[i]
        rsxs_sum_inv += phis[i]
    rsxs_sum_inv = 1.0/rsxs_sum_inv
    for i in range(N):
        phis[i] *= rsxs_sum_inv
    return phis, rsxs_sum_inv

def uniquac_dphis_dxs(N, rs, phis, rsxs_sum_inv, dphis_dxs=None, vec0=None):
    if dphis_dxs is None:
        dphis_dxs = [[0.0]*N for i in range(N)] # numba: delete
#        dphis_dxs = zeros((N, N)) # numba: uncomment
    if vec0 is None:
        vec0 = [0.0]*N

    rsxs_sum_inv_m = -rsxs_sum_inv
    for i in range(N):
        vec0[i] = phis[i]*rsxs_sum_inv_m

    for j in range(N):
        for i in range(N):
            dphis_dxs[i][j] = vec0[i]*rs[j]
        # There is no symmetry to exploit here
        dphis_dxs[j][j] += rs[j]*rsxs_sum_inv

    return dphis_dxs

def uniquac_thetaj_taus_jis(N, taus, thetas, thetaj_taus_jis=None):
    if thetaj_taus_jis is None:
        thetaj_taus_jis = [0.0]*N

    for i in range(N):
        tot = 0.0
        for j in range(N):
            tot += thetas[j]*taus[j][i]
        thetaj_taus_jis[i] = tot
    return thetaj_taus_jis

def uniquac_thetaj_taus_ijs(N, taus, thetas, thetaj_taus_ijs=None):
    if thetaj_taus_ijs is None:
        thetaj_taus_ijs = [0.0]*N

    for i in range(N):
        tot = 0.0
        for j in range(N):
            tot += thetas[j]*taus[i][j]
        thetaj_taus_ijs[i] = tot
    return thetaj_taus_ijs

def uniquac_d2phis_dxixjs(N, xs, rs, rsxs_sum_inv, d2phis_dxixjs=None, vec0=None, vec1=None):
    if d2phis_dxixjs is None:
        d2phis_dxixjs = [[[0.0]*N for _ in range(N)] for _ in range(N)] # numba: delete
#        d2phis_dxixjs = zeros((N, N, N)) # numba: uncomment
    if vec0 is None:
        vec0 = [0.0]*N
    if vec1 is None:
        vec1 = [0.0]*N

    rsxs_sum_inv2 = rsxs_sum_inv*rsxs_sum_inv
    rsxs_sum_inv3 = rsxs_sum_inv2*rsxs_sum_inv

    rsxs_sum_inv_2 = rsxs_sum_inv + rsxs_sum_inv
    rsxs_sum_inv2_2 = rsxs_sum_inv2 + rsxs_sum_inv2
    rsxs_sum_inv3_2 = rsxs_sum_inv3 + rsxs_sum_inv3

    for i in range(N):
        vec0[i] = rsxs_sum_inv2*(rs[i]*xs[i]*rsxs_sum_inv_2  - 1.0)
    for i in range(N):
        vec1[i] = rs[i]*xs[i]*rsxs_sum_inv3_2

    for k in range(N):
        # There is symmetry here, but it is complex. 4200 of 8000 (N=20) values are unique.
        # Due to the very large matrices, no gains to be had by exploiting it in this function
        # Removing the branches is the best that can be done (and it is quite good).
        d2phis_dxixjsk = d2phis_dxixjs[k]

        for j in range(N):
            rskrsj = rs[k]*rs[j]
            d2phis_dxixjskj = d2phis_dxixjsk[j]
            for i in range(N):
                d2phis_dxixjskj[i] = rskrsj*vec1[i]
            if j != k:
                d2phis_dxixjskj[k] = rskrsj*vec0[k]
                d2phis_dxixjskj[j] = rskrsj*vec0[j]

        d2phis_dxixjs[k][k][k] -= rs[k]*rs[k]*rsxs_sum_inv2_2
    return d2phis_dxixjs

def uniquac_GE(T, N, z, xs, qs, phis, thetas, thetaj_taus_jis):

    gE = 0.0
    z_2 = 0.5*z
    for i in range(N):
        gE += xs[i]*log(phis[i]/xs[i])
        gE += z_2*qs[i]*xs[i]*log(thetas[i]/phis[i])
        gE -= qs[i]*xs[i]*log(thetaj_taus_jis[i])

    gE *= R*T
    return gE

def uniquac_dGE_dT(T, N, GE, xs, qs, thetaj_taus_jis, thetaj_dtaus_dT_jis):
    dGE = GE/T
    tot = 0.0
    for i in range(N):
        tot -= qs[i]*xs[i]*thetaj_dtaus_dT_jis[i]/thetaj_taus_jis[i]
    dGE += R*T*tot
    return dGE


def uniquac_d2GE_dT2(T, N, GE, dGE_dT, xs, qs, thetaj_taus_jis, thetaj_dtaus_dT_jis, thetaj_d2taus_dT2_jis):
    tot = 0.0
    for i in range(N):
        tot += qs[i]*xs[i]*thetaj_d2taus_dT2_jis[i]/thetaj_taus_jis[i]
        tot -= qs[i]*xs[i]*thetaj_dtaus_dT_jis[i]**2/thetaj_taus_jis[i]**2
    d2GE_dT2 = T*tot - 2.0/(R*T)*(dGE_dT - GE/T)
    d2GE_dT2 *= -R
    return d2GE_dT2

def uniquac_d3GE_dT3(T, N, xs, qs, thetaj_taus_jis,
                     thetaj_dtaus_dT_jis, thetaj_d2taus_dT2_jis,
                     thetaj_d3taus_dT3_jis):
    Ttot, tot = 0.0, 0.0

    for i in range(N):
        Ttot += qs[i]*xs[i]*thetaj_d3taus_dT3_jis[i]/thetaj_taus_jis[i]
        Ttot -= 3.0*qs[i]*xs[i]*thetaj_dtaus_dT_jis[i]*thetaj_d2taus_dT2_jis[i]/thetaj_taus_jis[i]**2
        Ttot += 2.0*qs[i]*xs[i]*thetaj_dtaus_dT_jis[i]**3/thetaj_taus_jis[i]**3

        tot += 3.0*qs[i]*xs[i]*thetaj_d2taus_dT2_jis[i]/thetaj_taus_jis[i]
        tot -= 3.0*qs[i]*xs[i]*thetaj_dtaus_dT_jis[i]**2/thetaj_taus_jis[i]**2

    d3GE_dT3 = -R*(T*Ttot + tot)
    return d3GE_dT3



def uniquac_dGE_dxs(N, T, xs, qs, taus, phis, phis_inv, dphis_dxs, thetas, dthetas_dxs,
                    thetaj_taus_jis, thetaj_taus_jis_inv, dGE_dxs=None):
    z = 10.0
    if dGE_dxs is None:
        dGE_dxs = [0.0]*N
    RT = R*T

    for i in range(N):
        # i is what is being differentiated
        tot = 0.0
        for j in range(N):
            # dthetas_dxs and dphis_dxs indexes could be an issue
            tot += 0.5*qs[j]*xs[j]*phis[j]*z/thetas[j]*(
                    phis_inv[j]*dthetas_dxs[j][i]
                    - thetas[j]*phis_inv[j]*phis_inv[j]*dphis_dxs[j][i]
                    )

            tot3 = 0.0
            for k in range(N):
                tot3 += taus[k][j]*dthetas_dxs[k][i]

            tot -= qs[j]*xs[j]*tot3*thetaj_taus_jis_inv[j]
            if i != j:
                # Double index issue
                tot += xs[j]*phis_inv[j]*dphis_dxs[j][i]

        tot += 0.5*z*qs[i]*log(thetas[i]*phis_inv[i])
        tot -= qs[i]*log(thetaj_taus_jis[i])
        tot += xs[i]*xs[i]*phis_inv[i]*(dphis_dxs[i][i]/xs[i] - phis[i]/(xs[i]*xs[i]))
        tot += log(phis[i]/xs[i])
        # Last terms
        dGE_dxs[i] = RT*tot
    return dGE_dxs

def uniquac_d2GE_dTdxs(N, T, xs, qs, taus, phis, phis_inv, dphis_dxs, thetas, dthetas_dxs, dtaus_dT, thetaj_taus_jis, thetaj_taus_jis_inv, thetaj_dtaus_dT_jis, qsxs, qsxsthetaj_taus_jis_inv, d2GE_dTdxs=None, vec1=None, vec2=None, vec3=None, vec4=None):
    if d2GE_dTdxs is None:
        d2GE_dTdxs = [0.0]*N
    if vec1 is None:
        vec1 = [0.0]*N
    if vec2 is None:
        vec2 = [0.0]*N
    if vec3 is None:
        vec3 = [0.0]*N
    if vec4 is None:
        vec4 = [0.0]*N

    z = 10.0

    for i in range(N):
        vec1[i] = qsxsthetaj_taus_jis_inv[i]*thetaj_dtaus_dT_jis[i]*thetaj_taus_jis_inv[i]
    for i in range(N):
        vec2[i] = qsxs[i]/thetas[i]
    for i in range(N):
        vec3[i] = -thetas[i]*phis_inv[i]*vec2[i]
    for i in range(N):
        vec4[i] = xs[i]*phis_inv[i]

    # index style - [THE THETA FOR WHICH THE DERIVATIVE IS BEING CALCULATED][THE VARIABLE BEING CHANGED CAUsING THE DIFFERENCE]
    for i in range(N):
        # i is what is being differentiated
        tot, Ttot = 0.0, 0.0
        Ttot += qs[i]*thetaj_dtaus_dT_jis[i]*thetaj_taus_jis_inv[i]
        t49_sum = 0.0
        t50_sum = 0.0
        t51_sum = 0.0
        t52_sum = 0.0
        for j in range(N):
            t100 = 0.0
            for k in range(N):
                t100 += dtaus_dT[k][j]*dthetas_dxs[k][i]
            t102 = 0.0
            for k in range(N):
                t102 += taus[k][j]*dthetas_dxs[k][i]

            ## Temperature multiplied terms
            t49_sum += t100*qsxsthetaj_taus_jis_inv[j]

            t50_sum += t102*vec1[j]
            t52_sum += t102*qsxsthetaj_taus_jis_inv[j]

            ## Non temperature multiplied terms
            t51 = vec2[j]*dthetas_dxs[j][i] + vec3[j]*dphis_dxs[j][i]
            t51_sum += t51

#                # Terms reused from dGE_dxs
#                if i != j:
                # Double index issue
            tot += vec4[j]*dphis_dxs[j][i]

        Ttot -= t50_sum
        Ttot += t49_sum

        tot += t51_sum*z*0.5
        tot -= t52_sum

        tot -= xs[i]*phis_inv[i]*dphis_dxs[i][i] # Remove the branches by subreacting i after.

        # First term which is almost like it
        tot += xs[i]*phis_inv[i]*(dphis_dxs[i][i] - phis[i]/xs[i])
        tot += log(phis[i]/xs[i])
        tot -= qs[i]*log(thetaj_taus_jis[i])
        tot += 0.5*z*qs[i]*log(thetas[i]*phis_inv[i])

        d2GE_dTdxs[i] = R*(-T*Ttot + tot)
    return d2GE_dTdxs

def uniquac_d2GE_dxixjs(N, T, xs, qs, taus, phis, thetas, dphis_dxs, d2phis_dxixjs,
                        dthetas_dxs, d2thetas_dxixjs,
                        thetaj_taus_jis, d2GE_dxixjs=None):
    if d2GE_dxixjs is None:
        d2GE_dxixjs = [[0.0]*N for _ in range(N)] # numba: delete
#        d2GE_dxixjs = zeros((N, N)) # numba: uncomment
    RT = R*T
    z = 10

    # index style - [THE THETA FOR WHICH THE DERIVATIVE IS BEING CALCULATED][THE VARIABLE BEING CHANGED CAUsING THE DIFFERENCE]

    # You want to index by [i][k]
    tau_mk_dthetam_xi = [[0.0]*N for _ in range(N)] # numba: delete
#    tau_mk_dthetam_xi = zeros((N, N)) # numba: uncomment

    for i in range(N):
        for j in range(N):
            sum_value = 0.0
            for m in range(N):
                sum_value += taus[m][j] * dthetas_dxs[m][i]
            tau_mk_dthetam_xi[i][j] = sum_value

    for i in range(N):
        for j in range(N):
            ij_min = min(i, j)
            ij_max = max(i, j)
            tot = 0.0
            for (m, n) in [(i, j), (j, i)]:
                # 10-12
                # checked numerically already good!
                tot += qs[m]*z/(2.0*thetas[m])*(dthetas_dxs[m][n] - thetas[m]/phis[m]*dphis_dxs[m][n])

                # 0 -1
                # checked numerically
                if i != j:
                    tot +=  dphis_dxs[m][n]/phis[m]

                # 6-8
                # checked numerically already good!
                tot -= qs[m]*tau_mk_dthetam_xi[n][m]/thetaj_taus_jis[m]

            if i == j:
                # equivalent of 0-1 term
                tot += 3.0*dphis_dxs[i][i]/phis[i] - 3.0/xs[i]

            # 2-4 - two calculations
            # checked numerically Don't ask questions...
            for m in range(N):
                # This one looks like m != j should not be part
                if i < j:
                    if m != i:
                        tot += xs[m]/phis[m]*d2phis_dxixjs[i][j][m]
                else:
                    if m != j:
                        tot += xs[m]/phis[m]*d2phis_dxixjs[i][j][m]
#                v += xs[i]/phis[i]*d2phis_dxixjs[i][j][i]

            # 5-6
            # Now good, checked numerically
            tot -= xs[ij_min]*xs[ij_min]/(phis[ij_min]*phis[ij_min])*(dphis_dxs[ij_min][ij_min]/xs[ij_min] - phis[ij_min]/(xs[ij_min]*xs[ij_min]))*dphis_dxs[ij_min][ij_max]

            # 4-5
            # Now good, checked numerically
            if i != j:
                tot += xs[ij_min]*xs[ij_min]/phis[ij_min]*(d2phis_dxixjs[i][j][ij_min]/xs[ij_min] - dphis_dxs[ij_min][ij_max]/(xs[ij_min]*xs[ij_min]))
            else:
                tot += xs[i]*d2phis_dxixjs[i][i][i]/phis[i] - 2.0*dphis_dxs[i][i]/phis[i] + 2.0/xs[i]


            # Now good, checked numerically
            # This one looks like m != j should not be part
            # 8
            for m in range(N):
#                    if m != i and m != j:
                if i < j:
                    if m != i:
                        tot -= xs[m]/(phis[m]*phis[m])*(dphis_dxs[m][i]*dphis_dxs[m][j])
                else:
                    if m != j:
                        tot -= xs[m]/(phis[m]*phis[m])*(dphis_dxs[m][i]*dphis_dxs[m][j])
            # 9
#                v -= xs[i]*dphis_dxs[i][i]*dphis_dxs[i][j]/phis[i]**2

            for k in range(N):
                # 12-15
                # Good, checked with sympy/numerically
                thing = 0.0
                for m in range(N):
                    thing  += taus[m][k]*d2thetas_dxixjs[i][j][m]
                tot -= qs[k]*xs[k]*thing/thetaj_taus_jis[k]

                # 15-18
                # Good, checked with sympy/numerically
                tot -= qs[k]*xs[k]*tau_mk_dthetam_xi[i][k]*-1*tau_mk_dthetam_xi[j][k]/(thetaj_taus_jis[k]*thetaj_taus_jis[k])

                # 18-21
                # Good, checked with sympy/numerically
                tot += qs[k]*xs[k]*z/(2.0*thetas[k])*(dthetas_dxs[k][i]/phis[k]
                                - thetas[k]*dphis_dxs[k][i]/(phis[k]*phis[k])   )*dphis_dxs[k][j]

                # 21-24
                tot += qs[k]*xs[k]*z*phis[k]/(2.0*thetas[k])*(
                        d2thetas_dxixjs[i][j][k]/phis[k]
                        - 1.0/(phis[k]*phis[k])*(
                                thetas[k]*d2phis_dxixjs[i][j][k]
                                + dphis_dxs[k][i]*dthetas_dxs[k][j] + dphis_dxs[k][j]*dthetas_dxs[k][i])
                        + 2.0*thetas[k]*dphis_dxs[k][i]*dphis_dxs[k][j]/(phis[k]*phis[k]*phis[k])
                        )

                # 24-27
                # 10-13 in latest checking - but very suspiscious that the values are so low
                tot -= qs[k]*xs[k]*z*phis[k]*dthetas_dxs[k][j]/(2.0*thetas[k]*thetas[k])*(
                        dthetas_dxs[k][i]/phis[k] - thetas[k]*dphis_dxs[k][i]/(phis[k]*phis[k])
                        )

            d2GE_dxixjs[i][j] = RT*tot
    return d2GE_dxixjs

def uniquac_gammas_from_args(xs, N, T, z, rs, qs, taus, gammas=None):
    phis, rsxs_sum_inv = uniquac_phis(N, xs, rs, phis=None)
    phis_inv = [0.0]*N
    for i in range(N):
        phis_inv[i] = 1.0/phis[i]

    thetas, qsxs_sum_inv = uniquac_phis(N, xs, qs)
    thetaj_taus_jis = uniquac_thetaj_taus_jis(N, taus, thetas, thetaj_taus_jis=None)

    GE = uniquac_GE(T, N, z, xs, qs, phis, thetas, thetaj_taus_jis)
    dphis_dxs = uniquac_dphis_dxs(N, rs, phis, rsxs_sum_inv, dphis_dxs=None)
    dthetas_dxs = uniquac_dphis_dxs(N, qs, thetas, qsxs_sum_inv, dphis_dxs=None)
    thetaj_taus_jis_inv = [0.0]*N
    for i in range(N):
        thetaj_taus_jis_inv[i] = 1.0/thetaj_taus_jis[i]

    dGE_dxs = uniquac_dGE_dxs(N, T, xs, qs, taus, phis, phis_inv, dphis_dxs, thetas, dthetas_dxs,
                    thetaj_taus_jis, thetaj_taus_jis_inv, dGE_dxs=None)
    gammas = gibbs_excess_gammas(xs, dGE_dxs, GE, T, gammas=gammas)
    return gammas

class UNIQUAC(GibbsExcess):
    r'''Class for representing an a liquid with excess gibbs energy represented
    by the UNIQUAC equation. This model is capable of representing VL and LL
    behavior.

    .. math::
        \frac{G^E}{RT} = \sum_i x_i \ln\frac{\phi_i}{x_i}
        + \frac{z}{2}\sum_i q_i x_i \ln\frac{\theta_i}{\phi_i}
        - \sum_i q_i x_i \ln\left(\sum_j \theta_j \tau_{ji}   \right)

    .. math::
        \phi_i = \frac{r_i x_i}{\sum_j r_j x_j}

    .. math::
        \theta_i = \frac{q_i x_i}{\sum_j q_j x_j}

    .. math::
        \tau_{ij} = \exp\left[a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T
                + d_{ij}T + \frac{e_{ij}}{T^2} + f_{ij}{T^2}\right]

    Parameters
    ----------
    T : float
        Temperature, [K]
    xs : list[float]
        Mole fractions, [-]
    rs : list[float]
        `r` parameters :math:`r_i = \sum_{k=1}^{n} \nu_k R_k` if from UNIFAC,
        otherwise regressed, [-]
    qs : list[float]
        `q` parameters :math:`q_i = \sum_{k=1}^{n}\nu_k Q_k` if from UNIFAC,
        otherwise regressed, [-]
    tau_coeffs : list[list[list[float]]], optional
        UNIQUAC parameters, indexed by [i][j] and then each value is a 6
        element list with parameters [`a`, `b`, `c`, `d`, `e`, `f`];
        either `tau_coeffs` or `ABCDEF` are required, [-]
    ABCDEF : tuple[list[list[float]], 6], optional
        Contains the following. One of `tau_coeffs` or `ABCDEF` or some of the
        `tau_as`, etc parameters are required, [-]
    tau_as : list[list[float]] or None, optional
        `a` parameters used in calculating :obj:`UNIQUAC.taus`, [-]
    tau_bs : list[list[float]] or None, optional
        `b` parameters used in calculating :obj:`UNIQUAC.taus`, [K]
    tau_cs : list[list[float]] or None, optional
        `c` parameters used in calculating :obj:`UNIQUAC.taus`, [-]
    tau_ds : list[list[float]] or None, optional
        `d` paraemeters used in calculating :obj:`UNIQUAC.taus`, [1/K]
    tau_es : list[list[float]] or None, optional
        `e` parameters used in calculating :obj:`UNIQUAC.taus`, [K^2]
    tau_fs : list[list[float]] or None, optional
        `f` parameters used in calculating :obj:`UNIQUAC.taus`, [1/K^2]

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

    .. warning::
        There is no such thing as a missing parameter in the UNIQUAC model.
        It is possible to find :math:`\tau_{ij}` and :math:`\tau_{ji}` which
        make :math:`\gamma_i = 1` and :math:`\gamma_j = 1`, but those tau values
        depend on `rs`, `qs`, and `xs` - the composition, which obviously will
        change. It is therefore
        impossible to make an interaction parameter "missing"; whatever value
        it has will always impact the phase equilibria problem. At best, the
        tau values can produce close to ideal behavior.

    Examples
    --------
    **Example 1**

    Example 5.19 in [2]_ includes the calculation of liquid-liquid activity
    coefficients for the water-ethanol-benzene system. Two calculations are
    reproduced accurately here. Note that the DDBST-style coefficients assume
    a negative sign; for compatibility, their coefficients need to have their
    sign flipped.

    >>> N = 3
    >>> T = 25.0 + 273.15
    >>> xs = [0.7273, 0.0909, 0.1818]
    >>> rs = [.92, 2.1055, 3.1878]
    >>> qs = [1.4, 1.972, 2.4]
    >>> tausA = tausC = tausD = tausE = tausF = [[0.0]*N for i in range(N)]
    >>> tausB = [[0, 526.02, 309.64], [-318.06, 0, -91.532], [1325.1, 302.57, 0]]
    >>> tausB = [[-v for v in r] for r in tausB] # Flip the sign to come into UNIQUAC convention
    >>> ABCDEF = (tausA, tausB, tausC, tausD, tausE, tausF)
    >>> GE = UNIQUAC(T=T, xs=xs, rs=rs, qs=qs, ABCDEF=ABCDEF)
    >>> GE.gammas()
    [1.570393328, 0.2948241614, 18.114329048]

    The given values in [2]_ are [1.570, 0.2948, 18.11], matching exactly. The
    second phase has a different composition; the expected values are
    [8.856, 0.860, 1.425]. Once the :obj:`UNIQUAC` object has been constructed,
    it is very easy to obtain properties at different conditions:

    >>> GE.to_T_xs(T=T, xs=[1/6., 1/6., 2/3.]).gammas()
    [8.8559908058, 0.8595242462, 1.42546014081]

    The string representation of the object presents enough information to
    reconstruct it as well.

    >>> GE
    UNIQUAC(T=298.15, xs=[0.7273, 0.0909, 0.1818], rs=[0.92, 2.1055, 3.1878], qs=[1.4, 1.972, 2.4], ABCDEF=([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0, -526.02, -309.64], [318.06, 0, 91.532], [-1325.1, -302.57, 0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    The phase exposes many properties and derivatives as well.

    >>> GE.GE(), GE.dGE_dT(), GE.d2GE_dT2()
    (1843.96486834, 6.69851118521, -0.015896025970)
    >>> GE.HE(), GE.SE(), GE.dHE_dT(), GE.dSE_dT()
    (-153.19624152, -6.69851118521, 4.7394001431, 0.0158960259705)

    **Example 2**

    Another problem is 8.32 in [1]_ - acetonitrile, benzene, n-heptane at
    45 °C. The sign flip is needed here as well to convert their single
    temperature-dependent values into the correct form, but it has already been
    done to the coefficients:

    >>> N = 3
    >>> T = 45 + 273.15
    >>> xs = [.1311, .0330, .8359]
    >>> rs = [1.87, 3.19, 5.17]
    >>> qs = [1.72, 2.4, 4.4]
    >>> tausA = tausC = tausD = tausE = tausF = [[0.0]*N for i in range(N)]
    >>> tausB = [[0.0, -60.28, -23.71], [-89.57, 0.0, 135.9], [-545.8, -245.4, 0.0]]
    >>> ABCDEF = (tausA, tausB, tausC, tausD, tausE, tausF)
    >>> GE = UNIQUAC(T=T, xs=xs, rs=rs, qs=qs, ABCDEF=ABCDEF)
    >>> GE.gammas()
    [7.1533533992, 1.25052436922, 1.060392792605]

    The given values in [1]_ are [7.15, 1.25, 1.06].

    **Example 3**

    ChemSep is a program for modeling distillation. Chemsep ships with a
    permissive license several
    sets of binary interaction parameters. The UNIQUAC parameters in it can
    be accessed from Thermo as follows. In the following case, we compute
    activity coefficients of the ethanol-water system at mole fractions of
    [.252, 0.748].

    >>> from thermo.interaction_parameters import IPDB
    >>> CAS1, CAS2 = '64-17-5', '7732-18-5'
    >>> xs = [0.252, 0.748]
    >>> rs = [2.11, 0.92]
    >>> qs = [1.97, 1.400]
    >>> N = 2
    >>> T = 343.15
    >>> tau_bs = IPDB.get_ip_asymmetric_matrix(name='ChemSep UNIQUAC', CASs=['64-17-5', '7732-18-5'], ip='bij')
    >>> GE = UNIQUAC(T=T, xs=xs, rs=rs, qs=qs, tau_bs=tau_bs)
    >>> GE.gammas()
    [1.977454, 1.1397696]

    In ChemSep, the form of the UNIQUAC `tau` equation is

    .. math::
        \tau_{ij} = \exp\left( \frac{-A_{ij}}{RT}\right)

    The parameters were converted to the form used by Thermo as follows:

    .. math::
        b_{ij} = \frac{-A_{ij}}{R}= \frac{-A_{ij}}{1.9872042586408316}


    This system was chosen because there is also a sample problem for the same
    components from the DDBST which can be found here:
    http://chemthermo.ddbst.com/Problems_Solutions/Mathcad_Files/P05.01c%20VLE%20Behavior%20of%20Ethanol%20-%20Water%20Using%20UNIQUAC.xps

    In that example, with different data sets and parameters, they obtain at
    the same conditions activity coefficients of [2.359, 1.244].

    References
    ----------
    .. [1] Poling, Bruce E., John M. Prausnitz, and John P. O`Connell. The
       Properties of Gases and Liquids. 5th edition. New York: McGraw-Hill
       Professional, 2000.
    .. [2] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen Rarey.
       Chemical Thermodynamics for Process Simulation. John Wiley & Sons, 2019.
    '''

    z = 10.0
    _x_infinite_dilution = 1e-12
    model_id = 300

    _model_attributes = ('tau_as', 'tau_bs', 'tau_cs',
                        'tau_ds', 'tau_es', 'tau_fs',
                        'rs', 'qs')
    _cached_calculated_attributes = ('_qsxs_sum_inv', '_thetaj_d3taus_dT3_jis',
                 '_thetas', '_d2taus_dT2', '_thetaj_dtaus_dT_jis', '_thetaj_taus_jis', '_thetaj_d2taus_dT2_jis',
                 '_dthetas_dxs', '_rsxs_sum_inv', '_phis_inv', '_dtaus_dT',
                 '_d3taus_dT3', '_d2phis_dxixjs',
                 '_phis', '_dphis_dxs', '_taus', '_thetaj_taus_jis_inv', '_d2thetas_dxixjs', '_d3GE_dT3')

    __slots__ = GibbsExcess.__slots__ + _model_attributes + _cached_calculated_attributes + ('zero_coeffs',)

    def gammas_args(self, T=None):
        if T is not None:
            obj = self.to_T_xs(T=T, xs=self.xs)
        else:
            obj = self
        try:
            taus = obj._taus
        except AttributeError:
            taus = obj.taus()
        N = obj.N
        return (N, obj.T, obj.z, obj.rs, obj.qs, taus)

    gammas_from_args = staticmethod(uniquac_gammas_from_args)

    def __repr__(self):
        s = '{}(T={}, xs={}, rs={}, qs={}, ABCDEF={})'.format(self.__class__.__name__, repr(self.T), repr(self.xs), repr(self.rs), repr(self.qs),
                (self.tau_as,  self.tau_bs, self.tau_cs,
                 self.tau_ds, self.tau_es, self.tau_fs))
        return s

    def __init__(self, *, xs, rs, qs, T=GibbsExcess.T_DEFAULT, tau_coeffs=None, ABCDEF=None,
                 tau_as=None, tau_bs=None, tau_cs=None, tau_ds=None,
                 tau_es=None, tau_fs=None):
        self.T = T
        self.xs = xs
        self.vectorized = vectorized = type(rs) is not list
        self.rs = rs
        self.qs = qs

        self.N = N = len(rs)

        multiple_inputs = (tau_as, tau_bs, tau_cs, tau_ds, tau_es, tau_fs)

        input_count = ((tau_coeffs is not None) + (ABCDEF is not None)
                       + (any(i is not None for i in multiple_inputs)))
        if input_count > 1:
            raise ValueError("Input only one of tau_coeffs, ABCDEF, or (tau_as..)")

        if ABCDEF is None:
            ABCDEF = multiple_inputs


        if tau_coeffs is not None:
            pass
        elif ABCDEF is not None:
            try:
                all_lengths = tuple(len(coeffs) for coeffs in ABCDEF if coeffs is not None)
                if len(set(all_lengths)) > 1:
                    raise ValueError(f"Coefficient arrays of different size found: {all_lengths}")
                all_lengths_inner = tuple(len(coeffs[0]) for coeffs in ABCDEF if coeffs is not None)
                if len(set(all_lengths_inner)) > 1:
                    raise ValueError(f"Coefficient arrays of different size found: {all_lengths_inner}")
            except:
                raise ValueError("Coefficients not input correctly")
        else:
            raise ValueError("`tau_coeffs` or `ABCDEF` is required")

        if not vectorized:
            self.zero_coeffs = zero_coeffs = [[0.0]*N for _ in range(N)]
        else:
            self.zero_coeffs = zero_coeffs = zeros((N, N))

        if tau_coeffs is not None:
            if not vectorized:
                self.tau_as = [[i[0] for i in l] for l in tau_coeffs]
                self.tau_bs = [[i[1] for i in l] for l in tau_coeffs]
                self.tau_cs = [[i[2] for i in l] for l in tau_coeffs]
                self.tau_ds = [[i[3] for i in l] for l in tau_coeffs]
                self.tau_es = [[i[4] for i in l] for l in tau_coeffs]
                self.tau_fs = [[i[5] for i in l] for l in tau_coeffs]
            else:
                self.tau_as = array(tau_coeffs[:,:,0], order='C', copy=True)
                self.tau_bs = array(tau_coeffs[:,:,1], order='C', copy=True)
                self.tau_cs = array(tau_coeffs[:,:,2], order='C', copy=True)
                self.tau_ds = array(tau_coeffs[:,:,3], order='C', copy=True)
                self.tau_es = array(tau_coeffs[:,:,4], order='C', copy=True)
                self.tau_fs = array(tau_coeffs[:,:,5], order='C', copy=True)
        else:
            len_ABCDEF = len(ABCDEF)
            if len_ABCDEF == 0 or ABCDEF[0] is None:
                self.tau_as = zero_coeffs
            else:
                self.tau_as = ABCDEF[0]
            if len_ABCDEF < 2 or ABCDEF[1] is None:
                self.tau_bs = zero_coeffs
            else:
                self.tau_bs = ABCDEF[1]
            if len_ABCDEF < 3 or ABCDEF[2] is None:
                self.tau_cs = zero_coeffs
            else:
                self.tau_cs = ABCDEF[2]
            if len_ABCDEF < 4 or ABCDEF[3] is None:
                self.tau_ds = zero_coeffs
            else:
                self.tau_ds = ABCDEF[3]
            if len_ABCDEF < 5 or ABCDEF[4] is None:
                self.tau_es = zero_coeffs
            else:
                self.tau_es = ABCDEF[4]
            if len_ABCDEF < 6 or ABCDEF[5] is None:
                self.tau_fs = zero_coeffs
            else:
                self.tau_fs = ABCDEF[5]

    def to_T_xs(self, T, xs):
        r'''Method to construct a new :obj:`UNIQUAC` instance at
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
        obj : UNIQUAC
            New :obj:`UNIQUAC` object at the specified conditions [-]

        Notes
        -----
        If the new temperature is the same temperature as the existing
        temperature, if the `tau` terms or their derivatives have been
        calculated, they will be set to the new object as well.
        '''
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.xs = xs
        new.vectorized = self.vectorized
        new.rs = self.rs
        new.qs = self.qs
        new.N = self.N
        new.zero_coeffs = self.zero_coeffs

        (new.tau_as, new.tau_bs, new.tau_cs,
         new.tau_ds, new.tau_es, new.tau_fs) = (self.tau_as, self.tau_bs, self.tau_cs,
                         self.tau_ds, self.tau_es, self.tau_fs)

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
        return new


    def taus(self):
        r'''Calculate and return the `tau` terms for the UNIQUAC model for the
        system temperature.

        .. math::
            \tau_{ij} = \exp\left[a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T
                    + d_{ij}T + \frac{e_{ij}}{T^2} + f_{ij}{T^2}\right]

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
        # 87% of the time of this routine is the exponential.
        A = self.tau_as
        B = self.tau_bs
        C = self.tau_cs
        D = self.tau_ds
        E = self.tau_es
        F = self.tau_fs
        T = self.T
        N = self.N
        if not self.vectorized:
            taus = [[0.0]*N for _ in range(N)]
        else:
            taus = zeros((N, N))
        self._taus = interaction_exp(T, N, A, B, C, D, E, F, taus)
        return taus

    def dtaus_dT(self):
        r'''Calculate and return the temperature derivative of the `tau` terms
        for the UNIQUAC model for a specified temperature.

        .. math::
            \frac{\partial \tau_{ij}}{\partial T} =
            \left(2 T h_{ij} + d_{ij} + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}}
            - \frac{2 e_{ij}}{T^{3}}\right) e^{T^{2} h_{ij} + T d_{ij} + a_{ij}
            + c_{ij} \ln{\left(T \right)} + \frac{b_{ij}}{T}
            + \frac{e_{ij}}{T^{2}}}

        Returns
        -------
        dtaus_dT : list[list[float]]
            First temperature derivatives of tau terms, asymmetric matrix [1/K]

        Notes
        -----
        These values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._dtaus_dT
        except AttributeError:
            pass

        B = self.tau_bs
        C = self.tau_cs
        D = self.tau_ds
        E = self.tau_es
        F = self.tau_fs

        T, N = self.T, self.N
        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()
        if not self.vectorized:
            dtaus_dT = [[0.0]*N for _ in range(N)]
        else:
            dtaus_dT = zeros((N, N))
        self._dtaus_dT = dinteraction_exp_dT(T, N, B, C, D, E, F, taus, dtaus_dT)
        return dtaus_dT

    def d2taus_dT2(self):
        r'''Calculate and return the second temperature derivative of the `tau`
         terms for the UNIQUAC model for a specified temperature.

        .. math::
            \frac{\partial^2 \tau_{ij}}{\partial^2 T} =
            \left(2 f_{ij} + \left(2 T f_{ij} + d_{ij} + \frac{c_{ij}}{T}
            - \frac{b_{ij}}{T^{2}} - \frac{2 e_{ij}}{T^{3}}\right)^{2}
                - \frac{c_{ij}}{T^{2}} + \frac{2 b_{ij}}{T^{3}}
                + \frac{6 e_{ij}}{T^{4}}\right) e^{T^{2} f_{ij} + T d_{ij}
                + a_{ij} + c_{ij} \ln{\left(T \right)} + \frac{b_{ij}}{T}
                + \frac{e_{ij}}{T^{2}}}

        Returns
        -------
        d2taus_dT2 : list[list[float]]
            Second temperature derivatives of tau terms, asymmetric matrix
            [1/K^2]

        Notes
        -----
        These values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._d2taus_dT2
        except AttributeError:
            pass
        B = self.tau_bs
        C = self.tau_cs
        E = self.tau_es
        F = self.tau_fs
        T, N = self.T, self.N

        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()
        try:
            dtaus_dT = self._dtaus_dT
        except AttributeError:
            dtaus_dT = self.dtaus_dT()
        if not self.vectorized:
            d2taus_dT2s = [[0.0]*N for _ in range(N)]
        else:
            d2taus_dT2s = zeros((N, N))
        self._d2taus_dT2 = d2interaction_exp_dT2(T, N, B, C, E, F, taus, dtaus_dT, d2taus_dT2s)
        return d2taus_dT2s

    def d3taus_dT3(self):
        r'''Calculate and return the third temperature derivative of the `tau`
        terms for the UNIQUAC model for a specified temperature.

        .. math::
            \frac{\partial^3 \tau_{ij}}{\partial^3 T} =
            \left(3 \left(2 f_{ij} - \frac{c_{ij}}{T^{2}} + \frac{2 b_{ij}}{T^{3}}
            + \frac{6 e_{ij}}{T^{4}}\right) \left(2 T f_{ij} + d_{ij}
            + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}} - \frac{2 e_{ij}}{T^{3}}\right)
            + \left(2 T f_{ij} + d_{ij} + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}}
            - \frac{2 e_{ij}}{T^{3}}\right)^{3} - \frac{2 \left(- c_{ij}
            + \frac{3 b_{ij}}{T} + \frac{12 e_{ij}}{T^{2}}\right)}{T^{3}}\right)
            e^{T^{2} f_{ij} + T d_{ij} + a_{ij} + c_{ij} \ln{\left(T \right)}
            + \frac{b_{ij}}{T} + \frac{e_{ij}}{T^{2}}}

        Returns
        -------
        d3taus_dT3 : list[list[float]]
            Third temperature derivatives of tau terms, asymmetric matrix
            [1/K^3]

        Notes
        -----
        These values (and the coefficients) are NOT symmetric.
        '''
        try:
            return self._d3taus_dT3
        except AttributeError:
            pass

        T, N = self.T, self.N
        B = self.tau_bs
        C = self.tau_cs
        E = self.tau_es
        F = self.tau_fs

        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()
        try:
            dtaus_dT = self._dtaus_dT
        except AttributeError:
            dtaus_dT = self.dtaus_dT()
        if not self.vectorized:
            d3taus_dT3s = [[0.0]*N for _ in range(N)]
        else:
            d3taus_dT3s = zeros((N, N))
        self._d3taus_dT3 = d3interaction_exp_dT3(T, N, B, C, E, F, taus, dtaus_dT, d3taus_dT3s)
        return d3taus_dT3s

    def phis(self):
        r'''Calculate and return the `phi` parameters at the system
        composition and temperature.

        .. math::
            \phi_i = \frac{r_i x_i}{\sum_j r_j x_j}

        Returns
        -------
        phis : list[float]
            phi parameters, [-]

        Notes
        -----
        '''
        try:
            return self._phis
        except AttributeError:
            pass
        N, xs, rs = self.N, self.xs, self.rs
        if not self.vectorized:
            phis = [0.0]*N
        else:
            phis = zeros(N)

        self._phis, self._rsxs_sum_inv = uniquac_phis(N, xs, rs, phis)
        return self._phis

    def phis_inv(self):
        try:
            return self._phis_inv
        except:
            pass

        phis = self.phis()
        if not self.vectorized:
            phis_inv = [1.0/v for v in phis]
        else:
            phis_inv = 1.0/phis
        self._phis_inv = phis_inv
        return phis_inv

    def dphis_dxs(self):
        r'''

        if i != j:

        .. math::
            \frac{\partial \phi_i}{x_j} = \frac{-r_i r_j x_i}{(\sum_k r_k x_k)^2}

        else:

        .. math::
            \frac{\partial \phi_i}{x_j} = \frac{-r_i r_j x_i}{(\sum_k r_k x_k)^2} + \frac{r_i}{\sum_k r_k x_k}

        '''
        try:
            return self._dphis_dxs
        except AttributeError:
            pass
        N, rs = self.N, self.rs
        if not self.vectorized:
            dphis_dxs = [[0.0]*N for i in range(N)]
        else:
            dphis_dxs = zeros((N, N))
        self._dphis_dxs = uniquac_dphis_dxs(N, rs, self.phis(), self._rsxs_sum_inv, dphis_dxs)
        return dphis_dxs

    def d2phis_dxixjs(self):
        r'''

        if i != j:

        .. math::

        else:

        .. math::

        '''
        try:
            return self._d2phis_dxixjs
        except AttributeError:
            pass
        N, xs, rs = self.N, self.xs, self.rs

        self.phis() # Ensure the sum is there
        rsxs_sum_inv = self._rsxs_sum_inv
        if not self.vectorized:
            d2phis_dxixjs = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            d2phis_dxixjs = zeros((N, N, N))


        uniquac_d2phis_dxixjs(N, xs, rs, rsxs_sum_inv, d2phis_dxixjs)

        self._d2phis_dxixjs = d2phis_dxixjs
        return d2phis_dxixjs

    def thetas(self):
        r'''Calculate and return the `theta` parameters at the system
        composition and temperature.

        .. math::
            \theta_i = \frac{q_i x_i}{\sum_j q_j x_j}

        Returns
        -------
        thetas : list[float]
            theta parameters, [-]

        Notes
        -----
        '''
        try:
            return self._thetas
        except AttributeError:
            pass
        N, xs = self.N, self.xs
        qs = self.qs
        if not self.vectorized:
            thetas = [0.0]*N
        else:
            thetas = zeros(N)
        self._thetas, self._qsxs_sum_inv = uniquac_phis(N, xs, qs, thetas)
        return thetas

    def dthetas_dxs(self):
        r'''

        if i != j:

        .. math::
            \frac{\partial \theta_i}{x_j} = \frac{-r_i r_j x_i}{(\sum_k r_k x_k)^2}

        else:

        .. math::
            \frac{\partial \theta_i}{x_j} = \frac{-r_i r_j x_i}{(\sum_k r_k x_k)^2} + \frac{r_i}{\sum_k r_k x_k}

        '''
        try:
            return self._dthetas_dxs
        except AttributeError:
            pass
        N, qs = self.N, self.qs
        if not self.vectorized:
            dthetas_dxs =  [[0.0]*N for i in range(N)]
        else:
            dthetas_dxs = zeros((N, N))

        self._dthetas_dxs = dthetas_dxs = uniquac_dphis_dxs(N, qs, self.thetas(), self._qsxs_sum_inv, dthetas_dxs)
        return dthetas_dxs

    def d2thetas_dxixjs(self):
        r'''

        if i != j:

        .. math::

        else:

        .. math::

        '''
        try:
            return self._d2thetas_dxixjs
        except AttributeError:
            pass
        N, xs, qs = self.N, self.xs, self.qs

        self.thetas() # Ensure the sum is there
        qsxs_sum_inv = self._qsxs_sum_inv
        if not self.vectorized:
            d2thetas_dxixjs = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            d2thetas_dxixjs = zeros((N, N, N))

        uniquac_d2phis_dxixjs(N, xs, qs, qsxs_sum_inv, d2thetas_dxixjs)

        self._d2thetas_dxixjs = d2thetas_dxixjs
        return d2thetas_dxixjs

    def thetaj_taus_jis(self):
        # sum1
        try:
            return self._thetaj_taus_jis
        except AttributeError:
            pass
        try:
            thetas = self._thetas
        except AttributeError:
            thetas = self.thetas()

        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()

        N = self.N
        if not self.vectorized:
            thetaj_taus_jis = [0.0]*N
        else:
            thetaj_taus_jis = zeros(N)
        self._thetaj_taus_jis = thetaj_taus_jis = uniquac_thetaj_taus_jis(N, taus, thetas, thetaj_taus_jis)
        return thetaj_taus_jis

    def thetaj_taus_jis_inv(self):
        try:
            return self._thetaj_taus_jis_inv
        except:
            pass

        thetaj_taus_jis = self.thetaj_taus_jis()
        if not self.vectorized:
            thetaj_taus_jis_inv = [1.0/v for v in thetaj_taus_jis]
        else:
            thetaj_taus_jis_inv = 1.0/thetaj_taus_jis
        self._thetaj_taus_jis_inv = thetaj_taus_jis_inv
        return thetaj_taus_jis_inv

    def thetaj_taus_ijs(self):
        # no name yet
        try:
            return self._thetaj_taus_ijs
        except AttributeError:
            pass
        try:
            thetas = self._thetas
        except AttributeError:
            thetas = self.thetas()

        try:
            taus = self._taus
        except AttributeError:
            taus = self.taus()

        N = self.N
        self._thetaj_taus_ijs = thetaj_taus_ijs = uniquac_thetaj_taus_ijs(N, taus, thetas)
        return thetaj_taus_ijs

    def thetaj_dtaus_dT_jis(self):
        # sum3 maybe?
        try:
            return self._thetaj_dtaus_dT_jis
        except AttributeError:
            pass
        try:
            thetas = self._thetas
        except AttributeError:
            thetas = self.thetas()
        try:
            dtaus_dT = self._dtaus_dT
        except AttributeError:
            dtaus_dT = self.dtaus_dT()

        N = self.N
        if not self.vectorized:
            thetaj_dtaus_dT_jis = [0.0]*N
        else:
            thetaj_dtaus_dT_jis = zeros(N)
        self._thetaj_dtaus_dT_jis = uniquac_thetaj_taus_jis(N, dtaus_dT, thetas, thetaj_dtaus_dT_jis)
        return thetaj_dtaus_dT_jis



    def thetaj_d2taus_dT2_jis(self):
        # sum3 maybe?
        try:
            return self._thetaj_d2taus_dT2_jis
        except AttributeError:
            pass
        try:
            thetas = self._thetas
        except AttributeError:
            thetas = self.thetas()
        try:
            d2taus_dT2 = self._d2taus_dT2
        except AttributeError:
            d2taus_dT2 = self.d2taus_dT2()

        N = self.N
        if not self.vectorized:
            thetaj_d2taus_dT2_jis = [0.0]*N
        else:
            thetaj_d2taus_dT2_jis = zeros(N)
        self._thetaj_d2taus_dT2_jis = uniquac_thetaj_taus_jis(N, d2taus_dT2, thetas, thetaj_d2taus_dT2_jis)
        return thetaj_d2taus_dT2_jis

    def thetaj_d3taus_dT3_jis(self):
        try:
            return self._thetaj_d3taus_dT3_jis
        except AttributeError:
            pass
        try:
            thetas = self._thetas
        except AttributeError:
            thetas = self.thetas()
        try:
            d3taus_dT3 = self._d3taus_dT3
        except AttributeError:
            d3taus_dT3 = self.d3taus_dT3()

        N = self.N
        if not self.vectorized:
            thetaj_d3taus_dT3_jis = [0.0]*N
        else:
            thetaj_d3taus_dT3_jis = zeros(N)
        self._thetaj_d3taus_dT3_jis = uniquac_thetaj_taus_jis(N, d3taus_dT3, thetas, thetaj_d3taus_dT3_jis)
        return thetaj_d3taus_dT3_jis

    def GE(self):
        r'''Calculate and return the excess Gibbs energy of a liquid phase
        using the UNIQUAC model.

        .. math::
            \frac{G^E}{RT} = \sum_i x_i \ln\frac{\phi_i}{x_i}
            + \frac{z}{2}\sum_i q_i x_i \ln\frac{\theta_i}{\phi_i}
            - \sum_i q_i x_i \ln\left(\sum_j \theta_j \tau_{ji}   \right)

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
        T, N, xs = self.T, self.N, self.xs
        qs = self.qs
        phis = self.phis()
        thetas = self.thetas()
        thetaj_taus_jis = self.thetaj_taus_jis()
        self._GE = gE = uniquac_GE(T, N, self.z, xs, qs, phis, thetas, thetaj_taus_jis)
        return gE

    def dGE_dT(self):
        r'''Calculate and return the temperature derivative of excess Gibbs
        energy of a liquid phase using the UNIQUAC model.

        .. math::
            \frac{\partial G^E}{\partial T} = \frac{G^E}{T} - RT\left(\sum_i
            \frac{q_i x_i(\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T}
            )}{\sum_j \theta_j \tau_{ji}}\right)

        Returns
        -------
        dGE_dT : float
            First temperature derivative of excess Gibbs energy, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._dGE_dT
        except AttributeError:
            pass

        T, N, xs = self.T, self.N, self.xs
        qs = self.qs
        thetaj_taus_jis = self.thetaj_taus_jis()
        thetaj_dtaus_dT_jis = self.thetaj_dtaus_dT_jis()

        self._dGE_dT = dGE = uniquac_dGE_dT(T, N, self.GE(), xs, qs, thetaj_taus_jis, thetaj_dtaus_dT_jis)
        return dGE

    def d2GE_dT2(self):
        r'''Calculate and return the second temperature derivative of excess
        Gibbs energy of a liquid phase using the UNIQUAC model.

        .. math::
            \frac{\partial G^E}{\partial T^2} = -R\left[T\sum_i\left(
            \frac{q_i x_i(\sum_j \theta_j \frac{\partial^2 \tau_{ji}}{\partial T^2})}{\sum_j \theta_j \tau_{ji}}
            - \frac{q_i x_i(\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T})^2}{(\sum_j \theta_j \tau_{ji})^2}
            \right) + 2\left(\sum_i \frac{q_i x_i(\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T} )}{\sum_j \theta_j \tau_{ji}}\right)
            \right]

        Returns
        -------
        d2GE_dT2 : float
            Second temperature derivative of excess Gibbs energy, [J/(mol*K^2)]

        Notes
        -----
        '''
        try:
            return self._d2GE_dT2
        except AttributeError:
            pass

        T, N, xs = self.T, self.N, self.xs
        qs = self.qs
        thetaj_taus_jis = self.thetaj_taus_jis()
        thetaj_dtaus_dT_jis = self.thetaj_dtaus_dT_jis()
        thetaj_d2taus_dT2_jis = self.thetaj_d2taus_dT2_jis()

        GE = self.GE()
        dGE_dT = self.dGE_dT()

        self._d2GE_dT2 = d2GE_dT2 = uniquac_d2GE_dT2(T, N, GE, dGE_dT, xs, qs, thetaj_taus_jis,
                                                     thetaj_dtaus_dT_jis, thetaj_d2taus_dT2_jis)
        return d2GE_dT2

    def d3GE_dT3(self):
        r'''Calculate and return the third temperature derivative of excess
        Gibbs energy of a liquid phase using the UNIQUAC model.

        .. math::
            \frac{\partial^3 G^E}{\partial T^3} = -R\left[T\sum_i\left(
            \frac{q_i x_i(\sum_j \theta_j \frac{\partial^3 \tau_{ji}}{\partial T^3})}{(\sum_j \theta_j \tau_{ji})}
            - \frac{3q_i x_i(\sum_j \theta_j \frac{\partial^2 \tau_{ji}}{\partial T^2}) (\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T})}{(\sum_j \theta_j \tau_{ji})^2}
            + \frac{2q_i x_i(\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T})^3}{(\sum_j \theta_j \tau_{ji})^3}
            \right) + \sum_i \left(\frac{3q_i x_i(\sum_j x_j \frac{\partial^2 \tau_{ji}}{\partial T^2} ) }{\sum_j \theta_j \tau_{ji}}
            - \frac{3q_ix_i (\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T})^2}{(\sum_j \theta_j \tau_{ji})^2}
            \right)\right]

        Returns
        -------
        d3GE_dT3 : float
            Third temperature derivative of excess Gibbs energy, [J/(mol*K^3)]

        Notes
        -----
        '''
        try:
            return self._d3GE_dT3
        except AttributeError:
            pass

        T, N, xs = self.T, self.N, self.xs
        qs = self.qs
        thetaj_taus_jis = self.thetaj_taus_jis()
        thetaj_dtaus_dT_jis = self.thetaj_dtaus_dT_jis()
        thetaj_d2taus_dT2_jis = self.thetaj_d2taus_dT2_jis()
        thetaj_d3taus_dT3_jis = self.thetaj_d3taus_dT3_jis()

        self._d3GE_dT3 = d3GE_dT3 = uniquac_d3GE_dT3(T, N, xs, qs, thetaj_taus_jis,
                     thetaj_dtaus_dT_jis, thetaj_d2taus_dT2_jis,
                     thetaj_d3taus_dT3_jis)
        return d3GE_dT3

    def dGE_dxs(self):
        r'''Calculate and return the mole fraction derivatives of excess Gibbs
        energy using the UNIQUAC model.

        .. math::
            \frac{\partial G^E}{\partial x_i} = RT\left[
            \sum_j \frac{q_j x_j \phi_j z}{2\theta_j}\left(\frac{1}{\phi_j} \cdot \frac{\partial \theta_j}{\partial x_i}
            - \frac{\theta_j}{\phi_j^2}\cdot \frac{\partial \phi_j}{\partial x_i}
            \right)
            - \sum_j \left(\frac{q_j x_j(\sum_k \tau_{kj} \frac{\partial \theta_k}{\partial x_i} )}{\sum_k \tau_{kj} \theta_{k}}\right)
            + 0.5 z q_i\ln\left(\frac{\theta_i}{\phi_i}\right)
            - q_i\ln\left(\sum_j \tau_{ji}\theta_j \right)
            + \frac{x_i^2}{\phi_i}\left(\frac{\partial \phi_i}{\partial x_i}/x_i - \phi_i/x_i^2\right)
            + \sum_{j!= i} \frac{x_j}{\phi_j}\frac{\partial \phi_j}{\partial x_i}
            + \ln\left(\frac{\phi_i}{x_i} \right)
            \right]

        Returns
        -------
        dGE_dxs : list[float]
            Mole fraction derivatives of excess Gibbs energy, [J/mol]

        Notes
        -----
        '''
        try:
            return self._dGE_dxs
        except:
            pass
        z, T, xs, N = self.z, self.T, self.xs, self.N
        qs = self.qs
        taus = self.taus()
        phis = self.phis()
        phis_inv = self.phis_inv()
        dphis_dxs = self.dphis_dxs()
        thetas = self.thetas()
        dthetas_dxs = self.dthetas_dxs()
        thetaj_taus_jis = self.thetaj_taus_jis()
        thetaj_taus_jis_inv = self.thetaj_taus_jis_inv()

        if not self.vectorized:
            dGE_dxs = [0.0]*N
        else:
            dGE_dxs = zeros(N)

        uniquac_dGE_dxs(N, T, xs, qs, taus, phis, phis_inv, dphis_dxs, thetas, dthetas_dxs,
                        thetaj_taus_jis, thetaj_taus_jis_inv, dGE_dxs)
        self._dGE_dxs = dGE_dxs
        return dGE_dxs

    def d2GE_dTdxs(self):
        r'''Calculate and return the temperature derivative of mole fraction
        derivatives of excess Gibbs energy using the UNIQUAC model.

        .. math::
            \frac{\partial G^E}{\partial x_i \partial T} = R\left[-T\left\{
            \frac{q_i(\sum_j \theta_j \frac{\partial \tau_{ji}}{\partial T} )}{\sum_j \tau_{ki} \theta_k}
            + \sum_j \frac{q_j x_j(\sum_k \frac{\partial \tau_{kj}}{\partial T} \frac{\partial \theta_k}{\partial x_i} )        }{\sum_k \tau_{kj} \theta_k}
            - \sum_j \frac{q_j x_j (\sum_k \tau_{kj}  \frac{\partial \theta_k}{\partial x_i})
            (\sum_k \theta_k \frac{\partial \tau_{kj}}{\partial T})}{(\sum_k \tau_{kj} \theta_k)^2}
            \right\}
            + \sum_j \frac{q_j x_j z\left(\frac{\partial \theta_j}{\partial x_i} - \frac{\theta_j}{\phi_j}\frac{\partial \phi_j}{\partial x_i}  \right)}{2 \theta_j}
            - \sum_j \frac{q_j x_j \sum_k \tau_{kj}\frac{\partial \theta_k}{\partial x_i}}{\sum_k \tau_{kj}\theta_k}
            + 0.5zq_i \ln\left(\frac{\theta_i}{\phi_i}  \right) - q_i \ln\left(\sum_j \tau_{ji} \theta_j\right)
            + \ln\left(\frac{\phi_i}{x_i}    \right)
            + \frac{x_i}{\phi_i}\left(\frac{\partial \phi_i}{\partial x_i} -\frac{\phi_i}{x_i}  \right)
            + \sum_{j\ne i} \frac{x_j}{\phi_j}\frac{\partial \phi_j}{\partial x_i}
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
        z, T, xs, N = self.z, self.T, self.xs, self.N
        qs = self.qs
        taus = self.taus()
        phis = self.phis()
        phis_inv = self.phis_inv()
        dphis_dxs = self.dphis_dxs()
        thetas = self.thetas()
        dthetas_dxs = self.dthetas_dxs()
        dtaus_dT = self.dtaus_dT()

        thetaj_taus_jis = self.thetaj_taus_jis()
        thetaj_taus_jis_inv = self.thetaj_taus_jis_inv()
        thetaj_dtaus_dT_jis = self.thetaj_dtaus_dT_jis()

        if not self.vectorized:
            d2GE_dTdxs = [0.0]*N
            qsxs = [qs[i]*xs[i] for i in range(N)]
            qsxsthetaj_taus_jis_inv = [thetaj_taus_jis_inv[i]*qsxs[i] for i in range(N)]
#            vec1 = [qsxsthetaj_taus_jis_inv[i]*thetaj_dtaus_dT_jis[i]*thetaj_taus_jis_inv[i] for i in range(N)]
#            vec2 = [qsxs[i]/thetas[i] for i in range(N)]
#            vec3 = [-thetas[i]*phis_inv[i]*vec2[i] for i in range(N)]
#            vec4 = [xs[i]*phis_inv[i] for i in range(N)]
        else:
            d2GE_dTdxs = zeros(N)
            qsxs = qs*xs
            qsxsthetaj_taus_jis_inv = qsxs*thetaj_taus_jis_inv


        uniquac_d2GE_dTdxs(N, T, xs, qs, taus, phis, phis_inv, dphis_dxs, thetas, dthetas_dxs, dtaus_dT, thetaj_taus_jis, thetaj_taus_jis_inv, thetaj_dtaus_dT_jis, qsxs, qsxsthetaj_taus_jis_inv, d2GE_dTdxs)
#            vec1 = qsxsthetaj_taus_jis_inv*thetaj_dtaus_dT_jis*thetaj_taus_jis_inv
#            vec2 = qsxs/thetas
#            vec3 = -thetas*phis_inv*vec2
#            vec4 = xs*phis_inv

#        # index style - [THE THETA FOR WHICH THE DERIVATIVE IS BEING CALCULATED][THE VARIABLE BEING CHANGED CAUsING THE DIFFERENCE]
#        for i in range(N):
#            # i is what is being differentiated
#            tot, Ttot = 0.0, 0.0
#            Ttot += qs[i]*thetaj_dtaus_dT_jis[i]*thetaj_taus_jis_inv[i]
#            t49_sum = 0.0
#            t50_sum = 0.0
#            t51_sum = 0.0
#            t52_sum = 0.0
#            for j in range(N):
#                t100 = 0.0
#                for k in range(N):
#                    t100 += dtaus_dT[k][j]*dthetas_dxs[k][i]
#                t102 = 0.0
#                for k in range(N):
#                    t102 += taus[k][j]*dthetas_dxs[k][i]
#
#                ## Temperature multiplied terms
#                t49_sum += t100*qsxsthetaj_taus_jis_inv[j]
#
#                t50_sum += t102*vec1[j]
#                t52_sum += t102*qsxsthetaj_taus_jis_inv[j]
#
#                ## Non temperature multiplied terms
#                t51 = vec2[j]*dthetas_dxs[j][i] + vec3[j]*dphis_dxs[j][i]
#                t51_sum += t51
#
##                # Terms reused from dGE_dxs
##                if i != j:
#                    # Double index issue
#                tot += vec4[j]*dphis_dxs[j][i]
#
#            Ttot -= t50_sum
#            Ttot += t49_sum
#
#            tot += t51_sum*z*0.5
#            tot -= t52_sum
#
#            tot -= xs[i]*phis_inv[i]*dphis_dxs[i][i] # Remove the branches by subreacting i after.
#
#            # First term which is almost like it
#            tot += xs[i]*phis_inv[i]*(dphis_dxs[i][i] - phis[i]/xs[i])
#            tot += log(phis[i]/xs[i])
#            tot -= qs[i]*log(thetaj_taus_jis[i])
#            tot += 0.5*z*qs[i]*log(thetas[i]*phis_inv[i])
#
#            d2GE_dTdxs[i] = R*(-T*Ttot + tot)
        self._d2GE_dTdxs = d2GE_dTdxs
        return d2GE_dTdxs

    def d2GE_dxixjs(self):
        r'''Calculate and return the second mole fraction derivatives of excess
        Gibbs energy using the UNIQUAC model.

        .. math::
            \frac{\partial^2 g^E}{\partial x_i \partial x_j}

        Returns
        -------
        d2GE_dxixjs : list[list[float]]
            Second mole fraction derivatives of excess Gibbs energy, [J/mol]

        Notes
        -----
        The formula is extremely long and painful; see the source code for
        details.
        '''
        try:
            return self._d2GE_dxixjs
        except AttributeError:
            pass

        T, N = self.T, self.N
        xs = self.xs
        qs = self.qs
        taus = self.taus()
        phis = self.phis()
        thetas = self.thetas()
        dphis_dxs = self.dphis_dxs()
        d2phis_dxixjs = self.d2phis_dxixjs()
        dthetas_dxs = self.dthetas_dxs()
        d2thetas_dxixjs = self.d2thetas_dxixjs()
        thetaj_taus_jis = self.thetaj_taus_jis()

        if not self.vectorized:
            d2GE_dxixjs = [[0.0]*N for _ in range(N)]
        else:
            d2GE_dxixjs = zeros((N, N))

        self._d2GE_dxixjs = uniquac_d2GE_dxixjs(N=N, T=T, xs=xs, qs=qs, taus=taus, phis=phis, thetas=thetas, 
                                                dphis_dxs=dphis_dxs, d2phis_dxixjs=d2phis_dxixjs, dthetas_dxs=dthetas_dxs, 
                                                d2thetas_dxixjs=d2thetas_dxixjs, thetaj_taus_jis=thetaj_taus_jis, 
                                                d2GE_dxixjs=d2GE_dxixjs)
        return self._d2GE_dxixjs
                
#            debug_mat.append(debug_row)
        if self.vectorized:
            d2GE_dxixjs = array(d2GE_dxixjs)
        self._d2GE_dxixjs = d2GE_dxixjs
        return d2GE_dxixjs

    @classmethod
    def regress_binary_parameters(cls, gammas, xs, rs, qs, use_numba=False,
                                  do_statistics=True, **kwargs):
        r'''Perform a basic regression to determine the values of the `tau`
        terms in the UNIQUAC model, given a series of known or predicted
        activity coefficients and mole fractions.

        Parameters
        ----------
        gammas : list[list[float, 2]]
            List of activity coefficient pairs, [-]
        xs : list[list[float, 2]]
            List of binary mole fraction pairs, [-]
        rs : list[float]
            Van der Waals volume parameters for each species, [-]
        qs : list[float]
            Surface area parameters for each species, [-]
        use_numba : bool, optional
            Whether or not to try to use numba to speed up the computation, [-]
        do_statistics : bool, optional
            Whether or not to compute statistical measures on the outputs, [-]
        kwargs : dict
            Extra parameters to be passed to the fitting function (not yet
            documented), [-]

        Returns
        -------
        parameters : dict[str, float]
            Dimentionless interaction parameters of each compound with each
            other; these are the actual `tau` values. [-]
        statistics : dict[str: float]
            Statistics, calculated and returned only if `do_statistics` is True, [-]

        Notes
        -----
        Notes on getting fitting coefficients that yield gammas of 1:

            * This is possible some of the time to a pretty high accuracy
            * This is not possible whatsoever in some cases
            * The values of `rs`, and `qs` determine how close the fitting can be
            * If `rs` and `qs` are close to each other, it may well fit nicely
            * If they are distant (1.2-1.5x) they usually will not fit

        Examples
        --------
        In the following example, the `tau` values required to zero-out the
        coefficients for the n-pentane and n-hexane system are calculated. The
        parameters are converted back into `aij` parameters as used by this
        activity coefficient object, and then the calculated values are
        verified to be fairly nearly one.

        >>> from thermo import UNIQUAC
        >>> import numpy as np
        >>> pts = 30
        >>> rs = [3.8254, 4.4998]
        >>> qs = [3.316, 3.856]
        >>> xs = [[xi, 1.0 - xi] for xi in np.linspace(1e-7, 1-1e-7, pts)]
        >>> gammas = [[1, 1] for i in range(pts)]
        >>> coeffs, stats = UNIQUAC.regress_binary_parameters(gammas, xs, rs, qs)
        >>> coeffs
        {'tau12': 1.04220685, 'tau21': 0.95538082}
        >>> assert stats['MAE'] < 1e-6
        >>> tausB = tausC = tausD = tausE = tausF = [[0.0]*2 for i in range(2)]
        >>> tausA = [[0, np.log(coeffs['tau12'])], [np.log(coeffs['tau21']), 0]]
        >>> ABCDEF = (tausA, tausB, tausC, tausD, tausE, tausF)
        >>> GE = UNIQUAC(T=300, xs=[.5, .5], rs=rs, qs=qs, ABCDEF=ABCDEF)
        >>> GE.gammas()
        [1.000000466, 1.000000180]

        Note how the `tau` coefficients need to be converted into the `a`
        parameters of the `tau` equation. They could also have been converted
        into any of the other parameters, but then the activity coefficients
        predicted would no longer be close to 1 at other temperatures.

        .. math::
            \tau_{ij} = \exp\left[a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T
                    + d_{ij}T + \frac{e_{ij}}{T^2} + f_{ij}{T^2}\right]


        The UNIQUAC model's `r` and `q` parameters create their own biases in
        the model, based on the structure of each of the pure species. Water
        and n-pentane are not miscible liquids; they will form two liquid
        phases except when one component is present in trace amounts. No
        matter the values of `tau`, it is not possible to make the UNIQUAC
        equation predict activity coefficients very close to one for this
        system, as shown in the following sample.

        >>> rs = [3.8254, 0.92]
        >>> qs = [3.316, 1.4]
        >>> pts = 6
        >>> xs = [[xi, 1.0 - xi] for xi in np.linspace(1e-7, 1-1e-7, pts)]
        >>> gammas = [[1, 1] for i in range(pts)]
        >>> coeffs, stats = UNIQUAC.regress_binary_parameters(gammas, xs, rs, qs)
        >>> stats['MAE']
        0.0254

        '''
        if use_numba:
            from thermo.numba import UNIQUAC_gammas_binaries as work_func
            rs = array(rs)
            qs = array(qs)
        else:
            work_func = UNIQUAC_gammas_binaries

        def fitting_func(xs, tau12, tau21):
            # Capture rs, qs unfortunately is necessary. Works nicely with numba though.
            # try:
            return work_func(xs, rs, qs, tau12, tau21)
            # except:
            #     print(xs.tolist(), tau12, tau21)


        pts = len(xs)
        xs_working = []
        for i in range(pts):
            xs_working.append(xs[i][0])
            xs_working.append(xs[i][1])
        gammas_working = []
        for i in range(pts):
            gammas_working.append(gammas[i][0])
            gammas_working.append(gammas[i][1])

        xs_working = np.array(xs_working)
        gammas_working = np.array(gammas_working)

        return GibbsExcess._regress_binary_parameters(gammas_working, xs_working, fitting_func=fitting_func,
                                                      fit_parameters=['tau12', 'tau21'],
                                                      use_fit_parameters=['tau12', 'tau21'],
                                                      initial_guesses=cls._gamma_parameter_guesses,
                                                      analytical_jac=None,
                                                      use_numba=use_numba,
                                                      do_statistics=do_statistics,
                                                      **kwargs)


    _gamma_parameter_guesses = [#{'tau12': 1, 'tau21': 1}, # 1 is always tried
                            {'tau12': 1.0529981904211922, 'tau21': 1.1976772649513237},
                            {'tau12': 1.8748910210873349, 'tau21': 998.612171671497}, # Found seeking gamma = 1 for rs, qs = [[1.4, 7.219], [39.95, 47.2727]]
                            {'tau12': 0.6080855151163854, 'tau21': 1.5266917396579502},  # Found seeking gamma = 1 for rs, qs = [[30.49447368421054, 38.253], [30.195389473684212, 42.39346842105263]]
                            {'tau12': 0.75, 'tau21': 0.053}, # triethylamine and water from ChemSep main
                            {'tau12': 0.51, 'tau21': 0.00115}, # triethylamine and water from ChemSep main low T
                            {'tau12': 0.0003578, 'tau21': 32.6}, # acetic acid and 1-propanol from ChemSep main
                            {'tau12': 0.00056, 'tau21': 41.9}, # acetone and chloroform from ChemSep main
                            {'tau12': 1.45, 'tau21': 2.2e-7}, # methanol and ethene, chloro- from ChemSep main
                            ]

    for i in range(len(_gamma_parameter_guesses)):
        r = _gamma_parameter_guesses[i]
        _gamma_parameter_guesses.append({'tau12': r['tau21'], 'tau21': r['tau12']})



MIN_TAU_UNIQUAC = 1e-20

def UNIQUAC_gammas_binaries(xs, rs, qs, tau12, tau21, calc=None):
    # xs: array [x0_0, x1_0, x0_1, x1_1]
    if tau12 < MIN_TAU_UNIQUAC:
        tau12 = MIN_TAU_UNIQUAC
    if tau21 < MIN_TAU_UNIQUAC:
        tau21 = MIN_TAU_UNIQUAC
    pts = len(xs)//2 # Always even
    r1, r2 = rs
    q1, q2 = qs
    allocate_size = (pts*2)
    if calc is None:
        calc = [0.0]*allocate_size
    for i in range(pts):
        g0, g1 = UNIQUAC_gammas_binary(xs[i*2], r1, r2, q1, q2, tau12, tau21)
        calc[i*2] = g0
        calc[i*2+1] = g1
    return calc

def UNIQUAC_gammas_binary(x1, r1, r2, q1, q2, tau12, tau21):
    x0 = q1*x1
    x2 = x1 - 1
    x3 = q2*x2
    x4 = -tau21*x3 + x0
    x5 = -x3
    x6 = x0 + x5
    x7 = 1.0/x6
    x8 = log(x4*x7)
    x9 = r1*x1
    x10 = r2*x2
    x11 = -x10 + x9
    x12 = x11*x7
    x13 = 5.0*log(q1*x12/r1)
    x14 = 1/x11
    x15 = r1*x14
    x16 = 1 - x1
    x17 = r2*x16
    x18 = x17 + x9
    x19 = x11/x18**2
    x20 = 1/x4
    x21 = q2*x16
    x22 = 1/(x0 + x21)
    x23 = x0*x22
    x24 = x21*x22
    x25 = x22*x6
    x26 = 1/x18
    x27 = x18*x22
    x28 = q1*x27
    x29 = 5*x14
    x30 = x29*x6
    x31 = tau12*x0
    x32 = x31 + x5
    x33 = x25/x32
    x34 = q1**2*x1*x20*x25*(tau21*x24 + x23 - 1) + q1*x13 - q1*x3*x33*(-tau12*(1 - x23) + x24) - q1*x8 + x15*x2 - x19*x9 + x25*x29*x3*(-r1 + x28) - x28*x30*(x23 - x26*x9)
    x35 = log(x15)
    x36 = r2*x14
    x37 = log(x36)
    x38 = log(x32*x7)
    x39 = 5*log(q2*x12/r2)
    x40 = x23*x6
    x41 = -q2**2*x2*x33*(x22*x31 + x24 - 1) + q2*x20*x40*(-tau21*(1 - x24) + x23) - q2*x38 + q2*x39 - x1*x36 + x10*x19 - x29*x40*(q2*x27 - r2) + x27*x3*x30*(-x17*x26 + x24)/x16
    x42 = x0*x13 - x0*x8 + x1*x35 - x1*(x34 + x35) - x2*x37 + x2*(x37 + x41) + x3*x38 - x3*x39
    return (x15*trunc_exp(x34 + x42), x36*trunc_exp(x41 + x42))


def UNIQUAC_gammas(xs, rs, qs, taus):
    r'''Calculates the activity coefficients of each species in a mixture
    using the Universal quasi-chemical (UNIQUAC) equation, given their mole
    fractions, `rs`, `qs`, and dimensionless interaction parameters. The
    interaction parameters are normally correlated with temperature, and need
    to be calculated separately.

    .. math::
        \ln \gamma_i = \ln \frac{\Phi_i}{x_i} + \frac{z}{2} q_i \ln
        \frac{\theta_i}{\Phi_i}+ l_i - \frac{\Phi_i}{x_i}\sum_j^N x_j l_j
        - q_i \ln\left( \sum_j^N \theta_j \tau_{ji}\right)+ q_i - q_i\sum_j^N
        \frac{\theta_j \tau_{ij}}{\sum_k^N \theta_k \tau_{kj}}

    .. math::
        \theta_i = \frac{x_i q_i}{\displaystyle\sum_{j=1}^{n} x_j q_j}

    .. math::
        \Phi_i = \frac{x_i r_i}{\displaystyle\sum_{j=1}^{n} x_j r_j}

    .. math::
        l_i = \frac{z}{2}(r_i - q_i) - (r_i - 1)

    Parameters
    ----------
    xs : list[float]
        Liquid mole fractions of each species, [-]
    rs : list[float]
        Van der Waals volume parameters for each species, [-]
    qs : list[float]
        Surface area parameters for each species, [-]
    taus : list[list[float]]
        Dimensionless interaction parameters of each compound with each other,
        [-]

    Returns
    -------
    gammas : list[float]
        Activity coefficient for each species in the liquid mixture, [-]

    Notes
    -----
    This model needs N^2 parameters.

    The original expression for the interaction parameters is as follows:

    .. math::
        \tau_{ji} = \exp\left(\frac{-\Delta u_{ij}}{RT}\right)

    However, it is seldom used. Most correlations for the interaction
    parameters include some of the terms shown in the following form:

    .. math::
        \ln \tau_{ij} =a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T + d_{ij}T
        + \frac{e_{ij}}{T^2}

    This model is recast in a slightly more computationally efficient way in
    [2]_, as shown below:

    .. math::
        \ln \gamma_i = \ln \gamma_i^{res} + \ln \gamma_i^{comb}

    .. math::
        \ln \gamma_i^{res} = q_i \left(1 - \ln\frac{\sum_j^N q_j x_j \tau_{ji}}
        {\sum_j^N q_j x_j}- \sum_j \frac{q_k x_j \tau_{ij}}{\sum_k q_k x_k
        \tau_{kj}}\right)

    .. math::
        \ln \gamma_i^{comb} = (1 - V_i + \ln V_i) - \frac{z}{2}q_i\left(1 -
        \frac{V_i}{F_i} + \ln \frac{V_i}{F_i}\right)

    .. math::
        V_i = \frac{r_i}{\sum_j^N r_j x_j}

    .. math::
        F_i = \frac{q_i}{\sum_j q_j x_j}


    There is no global set of parameters which will make this model yield
    ideal acitivty coefficients (gammas = 1) for this model.

    Examples
    --------
    Ethanol-water example, at 343.15 K and 1 MPa:

    >>> UNIQUAC_gammas(xs=[0.252, 0.748], rs=[2.1055, 0.9200], qs=[1.972, 1.400],
    ... taus=[[1.0, 1.0919744384510301], [0.37452902779205477, 1.0]])
    [2.35875137797083, 1.2442093415968987]

    References
    ----------
    .. [1] Abrams, Denis S., and John M. Prausnitz. "Statistical Thermodynamics
       of Liquid Mixtures: A New Expression for the Excess Gibbs Energy of
       Partly or Completely Miscible Systems." AIChE Journal 21, no. 1 (January
       1, 1975): 116-28. doi:10.1002/aic.690210115.
    .. [2] Gmehling, Jurgen, Barbel Kolbe, Michael Kleiber, and Jurgen Rarey.
       Chemical Thermodynamics for Process Simulation. 1st edition. Weinheim:
       Wiley-VCH, 2012.
    .. [3] Maurer, G., and J. M. Prausnitz. "On the Derivation and Extension of
       the Uniquac Equation." Fluid Phase Equilibria 2, no. 2 (January 1,
       1978): 91-99. doi:10.1016/0378-3812(78)85002-X.
    '''
    cmps = range(len(xs))

    rsxs = [rs[i]*xs[i] for i in cmps]
    rsxs_sum_inv = 1.0/sum(rsxs)
    phis = [rsxs[i]*rsxs_sum_inv for i in cmps]


    qsxs = [qs[i]*xs[i] for i in cmps]
    qsxs_sum_inv = 1.0/sum(qsxs)
    vs = [qsxs[i]*qsxs_sum_inv for i in cmps]

    Ss = [sum([vs[j]*taus[j][i] for j in cmps]) for i in cmps]
    VsSs = [vs[j]/Ss[j] for j in cmps]

    ans = []
    for i in cmps:
        x1 = phis[i]/xs[i]
        x2 = phis[i]/vs[i]
        loggammac = log(x1) + 1.0 - x1 - 5.0*qs[i]*(log(x2) + 1.0 - x2)
        loggammar = qs[i]*(1.0 - log(Ss[i]) - sum([taus[i][j]*VsSs[j] for j in cmps]))
        ans.append(exp(loggammac + loggammar))
    return ans
