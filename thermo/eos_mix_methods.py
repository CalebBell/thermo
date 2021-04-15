# -*- coding: utf-8 -*-
r'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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

This file contains a number of overflow methods for EOSs which for various
reasons are better implemented as functions.
Documentation is not provided for this file and no methods are intended to be
used outside this library.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Alpha Function Mixing Rules
---------------------------
These are where the bulk of the time is spent in solving the equation of state.
For that reason, these functional forms often duplicate functionality but have
different performance characteristics.

Implementations which store N^2 matrices for other calculations:

.. autofunction:: a_alpha_aijs_composition_independent
.. autofunction:: a_alpha_aijs_composition_independent_support_zeros
.. autofunction:: a_alpha_and_derivatives_full

Compute only the alpha term itself:

.. autofunction:: a_alpha_and_derivatives

Faster implementations which do not store N^2 matrices:

.. autofunction:: a_alpha_quadratic_terms
.. autofunction:: a_alpha_and_derivatives_quadratic_terms
'''
'''
Direct fugacity calls
---------------------
The object-oriented interface is quite convenient. However, sometimes it is
desireable to perform a calculation at maximum speed, with no garbage collection
and the only temperature-dependent parts re-used each calculation.
For that reason, select equations of state have these functional forms
implemented

.. autofunction:: PR_lnphis
.. autofunction:: PR_lnphis_fastest


'''
# TODO: put methods like "_fast_init_specific" in here so numba can accelerate them.
from fluids.constants import R
from fluids.numerics import numpy as np, catanh
from math import sqrt, log
from thermo.eos_volume import volume_solutions_halley

__all__ = ['a_alpha_aijs_composition_independent',
           'a_alpha_and_derivatives', 'a_alpha_and_derivatives_full',
           'a_alpha_quadratic_terms', 'a_alpha_and_derivatives_quadratic_terms',
           'PR_lnphis', 'VDW_lnphis',
           'PR_lnphis_fastest', 'lnphis_direct',
           'G_dep_lnphi_d_helper', 'PR_translated_ddelta_dzs',
           'PR_translated_depsilon_dzs',
           'eos_mix_dV_dzs']


R2 = R*R
R_inv = 1.0/R
R2_inv = R_inv*R_inv
root_two = sqrt(2.)
root_two_m1 = root_two - 1.0
root_two_p1 = root_two + 1.0

def a_alpha_aijs_composition_independent(a_alphas, kijs):
    r'''Calculates the matrix :math:`(a\alpha)_{ij}` as well as the array
    :math:`\sqrt{(a\alpha)_{i}}` and the matrix
    :math:`\frac{1}{\sqrt{(a\alpha)_{i}}\sqrt{(a\alpha)_{j}}}`.

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    This routine is efficient in both numba and PyPy, but it is generally
    better to avoid calculating and storing **any** N^2 matrices. However,
    this particular calculation only depends on `T` so in some circumstances
    this can be feasible.

    Parameters
    ----------
    a_alphas : list[float]
        EOS attractive terms, [J^2/mol^2/Pa]
    kijs : list[list[float]]
        Constant kijs, [-]

    Returns
    -------
    a_alpha_ijs : list[list[float]]
        Matrix of :math:`(1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}`, [J^2/mol^2/Pa]
    a_alpha_roots : list[float]
        Array of :math:`\sqrt{(a\alpha)_{i}}` values, [J/mol/Pa^0.5]
    a_alpha_ij_roots_inv : list[list[float]]
        Matrix of :math:`\frac{1}{\sqrt{(a\alpha)_{i}}\sqrt{(a\alpha)_{j}}}`,
        [mol^2*Pa/J^2]

    Notes
    -----

    Examples
    --------
    >>> kijs = [[0,.083],[0.083,0]]
    >>> a_alphas = [0.2491099357671155, 0.6486495863528039]
    >>> a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, kijs)
    >>> a_alpha_ijs
    [[0.249109935767, 0.36861239374], [0.36861239374, 0.64864958635]]
    >>> a_alpha_roots
    [0.49910914213, 0.80538784840]
    >>> a_alpha_ij_roots_inv
    [[4.0142919105, 2.487707997796], [2.487707997796, 1.54166443799]]
    '''
    N = len(a_alphas)
    _sqrt = sqrt

    a_alpha_ijs = [[0.0]*N for _ in range(N)] # numba: comment
#    a_alpha_ijs = np.zeros((N, N)) # numba: uncomment
    a_alpha_roots = [0.0]*N
    for i in range(N):
        a_alpha_roots[i] = _sqrt(a_alphas[i])

    a_alpha_ij_roots_inv = [[0.0]*N for _ in range(N)] # numba: comment
#    a_alpha_ij_roots_inv = np.zeros((N, N)) # numba: uncomment

    for i in range(N):
        kijs_i = kijs[i]
        a_alpha_ijs_is = a_alpha_ijs[i]
        a_alpha_ij_roots_i_inv = a_alpha_ij_roots_inv[i]
        # Using range like this saves 20% of the comp time for 44 components!
        a_alpha_i_root_i = a_alpha_roots[i]
        for j in range(i, N):
            term = a_alpha_i_root_i*a_alpha_roots[j]
            a_alpha_ij_roots_i_inv[j] = a_alpha_ij_roots_inv[j][i] = 1.0/term
            a_alpha_ijs_is[j] = a_alpha_ijs[j][i] = (1. - kijs_i[j])*term
    return a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv


def a_alpha_aijs_composition_independent_support_zeros(a_alphas, kijs):
    # Same as the above but works when there are zeros
    N = len(a_alphas)
    cmps = range(N)

    a_alpha_ijs = [[0.0] * N for _ in cmps]
    a_alpha_roots = [a_alpha_i ** 0.5 for a_alpha_i in a_alphas]
    a_alpha_ij_roots_inv = [[0.0] * N for _ in cmps]

    for i in cmps:
        kijs_i = kijs[i]
        a_alpha_i = a_alphas[i]
        a_alpha_ijs_is = a_alpha_ijs[i]
        a_alpha_ij_roots_i_inv = a_alpha_ij_roots_inv[i]
        a_alpha_i_root_i = a_alpha_roots[i]
        for j in range(i, N):
            term = a_alpha_i_root_i * a_alpha_roots[j]
            try:
                a_alpha_ij_roots_i_inv[j] = 1.0/term
            except ZeroDivisionError:
                a_alpha_ij_roots_i_inv[j] = 1e100
            a_alpha_ijs_is[j] = a_alpha_ijs[j][i] = (1. - kijs_i[j]) * term
    return a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv


def a_alpha_and_derivatives(a_alphas, T, zs, kijs, a_alpha_ijs=None,
                            a_alpha_roots=None, a_alpha_ij_roots_inv=None):
    N = len(a_alphas)
    da_alpha_dT, d2a_alpha_dT2 = 0.0, 0.0

    if a_alpha_ijs is None or a_alpha_roots is None or a_alpha_ij_roots_inv is None:
        a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, kijs)

    a_alpha = 0.0
    for i in range(N):
        a_alpha_ijs_i = a_alpha_ijs[i]
        zi = zs[i]
        for j in range(i+1, N):
            term = a_alpha_ijs_i[j]*zi*zs[j]
            a_alpha += term + term

        a_alpha += a_alpha_ijs_i[i]*zi*zi

    return a_alpha, None, a_alpha_ijs


def a_alpha_and_derivatives_full(a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, zs,
                                 kijs, a_alpha_ijs=None, a_alpha_roots=None,
                                 a_alpha_ij_roots_inv=None):
    r'''Calculates the `a_alpha` term, and its first two temperature
    derivatives, for an equation of state along with the
    matrix quantities calculated in the process.

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        \frac{\partial (a\alpha)}{\partial T} = \sum_i \sum_j z_i z_j
        \frac{\partial (a\alpha)_{ij}}{\partial T}

    .. math::
        \frac{\partial^2 (a\alpha)}{\partial T^2} = \sum_i \sum_j z_i z_j
        \frac{\partial^2 (a\alpha)_{ij}}{\partial T^2}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        \frac{\partial (a\alpha)_{ij}}{\partial T} =
        \frac{\sqrt{\operatorname{a\alpha_{i}}{\left(T \right)} \operatorname{a\alpha_{j}}
        {\left(T \right)}} \left(1 - k_{ij}\right) \left(\frac{\operatorname{a\alpha_{i}}
        {\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}}{2}
        + \frac{\operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T} \operatorname{
        a\alpha_{i}}{\left(T \right)}}{2}\right)}{\operatorname{a\alpha_{i}}{\left(T \right)}
        \operatorname{a\alpha_{j}}{\left(T \right)}}

    .. math::
        \frac{\partial^2 (a\alpha)_{ij}}{\partial T^2} =
        - \frac{\sqrt{\operatorname{a\alpha_{i}}{\left(T \right)} \operatorname{a\alpha_{j}}
        {\left(T \right)}} \left(k_{ij} - 1\right) \left(\frac{\left(\operatorname{
        a\alpha_{i}}{\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}
        + \operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{i}}
        {\left(T \right)}\right)^{2}}{4 \operatorname{a\alpha_{i}}{\left(T \right)}
        \operatorname{a\alpha_{j}}{\left(T \right)}} - \frac{\left(\operatorname{a\alpha_{i}}
        {\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}
        + \operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T}
        \operatorname{a\alpha_{i}}{\left(T \right)}\right) \frac{d}{d T}
        \operatorname{a\alpha_{j}}{\left(T \right)}}{2 \operatorname{a\alpha_{j}}
        {\left(T \right)}} - \frac{\left(\operatorname{a\alpha_{i}}{\left(T \right)}
        \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}
        + \operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T}
        \operatorname{a\alpha_{i}}{\left(T \right)}\right) \frac{d}{d T}
        \operatorname{a\alpha_{i}}{\left(T \right)}}{2 \operatorname{a\alpha_{i}}
        {\left(T \right)}} + \frac{\operatorname{a\alpha_{i}}{\left(T \right)}
        \frac{d^{2}}{d T^{2}} \operatorname{a\alpha_{j}}{\left(T \right)}}{2}
        + \frac{\operatorname{a\alpha_{j}}{\left(T \right)} \frac{d^{2}}{d T^{2}}
        \operatorname{a\alpha_{i}}{\left(T \right)}}{2} + \frac{d}{d T}
        \operatorname{a\alpha_{i}}{\left(T \right)} \frac{d}{d T}
        \operatorname{a\alpha_{j}}{\left(T \right)}\right)}
        {\operatorname{a\alpha_{i}}{\left(T \right)} \operatorname{a\alpha_{j}}
        {\left(T \right)}}


    Parameters
    ----------
    a_alphas : list[float]
        EOS attractive terms, [J^2/mol^2/Pa]
    da_alpha_dTs : list[float]
        Temperature derivative of coefficient calculated by EOS-specific
        method, [J^2/mol^2/Pa/K]
    d2a_alpha_dT2s : list[float]
        Second temperature derivative of coefficient calculated by
        EOS-specific method, [J^2/mol^2/Pa/K**2]
    T : float
        Temperature, not used, [K]
    zs : list[float]
        Mole fractions of each species
    kijs : list[list[float]]
        Constant kijs, [-]
    a_alpha_ijs : list[list[float]], optional
        Matrix of :math:`(1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}`, [J^2/mol^2/Pa]
    a_alpha_roots : list[float], optional
        Array of :math:`\sqrt{(a\alpha)_{i}}` values, [J/mol/Pa^0.5]
    a_alpha_ij_roots_inv : list[list[float]], optional
        Matrix of :math:`\frac{1}{\sqrt{(a\alpha)_{i}}\sqrt{(a\alpha)_{j}}}`,
        [mol^2*Pa/J^2]

    Returns
    -------
    a_alpha : float
        EOS attractive term, [J^2/mol^2/Pa]
    da_alpha_dT : float
        Temperature derivative of coefficient calculated by EOS-specific
        method, [J^2/mol^2/Pa/K]
    d2a_alpha_dT2 : float
        Second temperature derivative of coefficient calculated by
        EOS-specific method, [J^2/mol^2/Pa/K**2]
    a_alpha_ijs : list[list[float]], optional
        Matrix of :math:`(1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}`,
        [J^2/mol^2/Pa]
    da_alpha_dT_ijs : list[list[float]], optional
        Matrix of :math:`\frac{\partial (a\alpha)_{ij}}{\partial T}`,
        [J^2/mol^2/Pa/K]
    d2a_alpha_dT2_ijs : list[list[float]], optional
        Matrix of :math:`\frac{\partial^2 (a\alpha)_{ij}}{\partial T^2}`,
        [J^2/mol^2/Pa/K^2]

    Notes
    -----

    Examples
    --------
    >>> kijs = [[0,.083],[0.083,0]]
    >>> zs = [0.1164203, 0.8835797]
    >>> a_alphas = [0.2491099357671155, 0.6486495863528039]
    >>> da_alpha_dTs = [-0.0005102028006086241, -0.0011131153520304886]
    >>> d2a_alpha_dT2s = [1.8651128859234162e-06, 3.884331923127011e-06]
    >>> a_alpha, da_alpha_dT, d2a_alpha_dT2, a_alpha_ijs, da_alpha_dT_ijs, d2a_alpha_dT2_ijs = a_alpha_and_derivatives_full(a_alphas=a_alphas, da_alpha_dTs=da_alpha_dTs, d2a_alpha_dT2s=d2a_alpha_dT2s, T=299.0, zs=zs, kijs=kijs)
    >>> a_alpha, da_alpha_dT, d2a_alpha_dT2
    (0.58562139582, -0.001018667672, 3.56669817856e-06)
    >>> a_alpha_ijs
    [[0.2491099357, 0.3686123937], [0.36861239374, 0.64864958635]]
    >>> da_alpha_dT_ijs
    [[-0.000510202800, -0.0006937567844], [-0.000693756784, -0.00111311535]]
    >>> d2a_alpha_dT2_ijs
    [[1.865112885e-06, 2.4734471244e-06], [2.4734471244e-06, 3.8843319e-06]]
    '''
    # For 44 components, takes 150 us in PyPy.

    N = len(a_alphas)
    da_alpha_dT, d2a_alpha_dT2 = 0.0, 0.0

    if a_alpha_ijs is None or a_alpha_roots is None or a_alpha_ij_roots_inv is None:
        a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, kijs)

    z_products = [[zs[i]*zs[j] for j in range(N)] for i in range(N)] # numba : delete
#    z_products = np.zeros((N, N)) # numba: uncomment
#    for i in range(N): # numba: uncomment
#        for j in range(N): # numba: uncomment
#            z_products[i][j] = zs[i]*zs[j] # numba: uncomment


    a_alpha = 0.0
    for i in range(N):
        a_alpha_ijs_i = a_alpha_ijs[i]
        z_products_i = z_products[i]
        for j in range(i):
            term = a_alpha_ijs_i[j]*z_products_i[j]
            a_alpha += term + term
        a_alpha += a_alpha_ijs_i[i]*z_products_i[i]

    da_alpha_dT_ijs = [[0.0]*N for _ in range(N)] # numba : delete
#    da_alpha_dT_ijs = np.zeros((N, N)) # numba: uncomment
    d2a_alpha_dT2_ijs = [[0.0]*N for _ in range(N)] # numba : delete
#    d2a_alpha_dT2_ijs = np.zeros((N, N)) # numba: uncomment

    d2a_alpha_dT2_ij = 0.0

    for i in range(N):
        kijs_i = kijs[i]
        a_alphai = a_alphas[i]
        z_products_i = z_products[i]
        da_alpha_dT_i = da_alpha_dTs[i]
        d2a_alpha_dT2_i = d2a_alpha_dT2s[i]
        a_alpha_ij_roots_inv_i = a_alpha_ij_roots_inv[i]
        da_alpha_dT_ijs_i = da_alpha_dT_ijs[i]

        for j in range(N):
#        for j in range(0, i+1):
            if j < i:
#                # skip the duplicates
                continue
            a_alphaj = a_alphas[j]
            x0_05_inv = a_alpha_ij_roots_inv_i[j]
            zi_zj = z_products_i[j]
            da_alpha_dT_j = da_alpha_dTs[j]

            x1 = a_alphai*da_alpha_dT_j
            x2 = a_alphaj*da_alpha_dT_i
            x1_x2 = x1 + x2
            x3 = x1_x2 + x1_x2

            kij_m1 = kijs_i[j] - 1.0

            da_alpha_dT_ij = -0.5*kij_m1*x1_x2*x0_05_inv
            # For temperature derivatives of fugacities
            da_alpha_dT_ijs_i[j] = da_alpha_dT_ijs[j][i] = da_alpha_dT_ij

            da_alpha_dT_ij *= zi_zj


            x0 = a_alphai*a_alphaj

            d2a_alpha_dT2_ij = kij_m1*(  (x0*(
            -0.5*(a_alphai*d2a_alpha_dT2s[j] + a_alphaj*d2a_alpha_dT2_i)
            - da_alpha_dT_i*da_alpha_dT_j) +.25*x1_x2*x1_x2)/(x0_05_inv*x0*x0))
            d2a_alpha_dT2_ijs[i][j] = d2a_alpha_dT2_ijs[j][i] = d2a_alpha_dT2_ij

            d2a_alpha_dT2_ij *= zi_zj

            if i != j:
                da_alpha_dT += da_alpha_dT_ij + da_alpha_dT_ij
                d2a_alpha_dT2 += d2a_alpha_dT2_ij + d2a_alpha_dT2_ij
            else:
                da_alpha_dT += da_alpha_dT_ij
                d2a_alpha_dT2 += d2a_alpha_dT2_ij

    return a_alpha, da_alpha_dT, d2a_alpha_dT2, a_alpha_ijs, da_alpha_dT_ijs, d2a_alpha_dT2_ijs


def a_alpha_quadratic_terms(a_alphas, a_alpha_roots, T, zs, kijs):
    r'''Calculates the `a_alpha` term for an equation of state along with the
    vector quantities needed to compute the fugacities of the mixture. This
    routine is efficient in both numba and PyPy.

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    The secondary values are as follows:

    .. math::
        \sum_i y_i(a\alpha)_{ij}

    Parameters
    ----------
    a_alphas : list[float]
        EOS attractive terms, [J^2/mol^2/Pa]
    a_alpha_roots : list[float]
        Square roots of `a_alphas`; provided for speed [J/mol/Pa^0.5]
    T : float
        Temperature, not used, [K]
    zs : list[float]
        Mole fractions of each species
    kijs : list[list[float]]
        Constant kijs, [-]

    Returns
    -------
    a_alpha : float
        EOS attractive term, [J^2/mol^2/Pa]
    a_alpha_j_rows : list[float]
        EOS attractive term row sums, [J^2/mol^2/Pa]

    Notes
    -----
    Tried moving the i=j loop out, no difference in speed, maybe got a bit slower
    in PyPy.

    Examples
    --------
    >>> kijs = [[0,.083],[0.083,0]]
    >>> zs = [0.1164203, 0.8835797]
    >>> a_alphas = [0.2491099357671155, 0.6486495863528039]
    >>> a_alpha_roots = [i**0.5 for i in a_alphas]
    >>> a_alpha, a_alpha_j_rows = a_alpha_quadratic_terms(a_alphas, a_alpha_roots, 299.0, zs, kijs)
    >>> a_alpha, a_alpha_j_rows
    (0.58562139582, [0.35469988173, 0.61604757237])
    '''
    # This is faster in PyPy and can be made even faster optimizing a_alpha!
#    N = len(a_alphas)
#    a_alpha_j_rows = [0.0]*N
#    a_alpha = 0.0
#    for i in range(N):
#        kijs_i = kijs[i]
#        a_alpha_i_root_i = a_alpha_roots[i]
#        for j in range(i):
#            a_alpha_ijs_ij = (1. - kijs_i[j])*a_alpha_i_root_i*a_alpha_roots[j]
#            t200 = a_alpha_ijs_ij*zs[i]
#            a_alpha_j_rows[j] += t200
#            a_alpha_j_rows[i] += zs[j]*a_alpha_ijs_ij
#            t200 *= zs[j]
#            a_alpha += t200 + t200
#
#        t200 = (1. - kijs_i[i])*a_alphas[i]*zs[i]
#        a_alpha += t200*zs[i]
#        a_alpha_j_rows[i] += t200
#
#    return a_alpha, a_alpha_j_rows

    N = len(a_alphas)
    a_alpha_j_rows = [0.0]*N
    things0 = [0.0]*N
    for i in range(N):
        things0[i] = a_alpha_roots[i]*zs[i]

    a_alpha = 0.0
    i = 0
    while i < N:
        kijs_i = kijs[i]
        j = 0
        while j < i:
            # Numba appears to be better with this split into two loops.
            # PyPy has 1.5x speed reduction when so.
            a_alpha_j_rows[j] += (1. - kijs_i[j])*things0[i]
            a_alpha_j_rows[i] += (1. - kijs_i[j])*things0[j]
            j += 1
        i += 1

    for i in range(N):
        a_alpha_j_rows[i] *= a_alpha_roots[i]
        a_alpha_j_rows[i] += (1. -  kijs[i][i])*a_alphas[i]*zs[i]
        a_alpha += a_alpha_j_rows[i]*zs[i]

    return a_alpha, a_alpha_j_rows


def a_alpha_and_derivatives_quadratic_terms(a_alphas, a_alpha_roots,
                                            da_alpha_dTs, d2a_alpha_dT2s, T,
                                            zs, kijs):
    r'''Calculates the `a_alpha` term, and its first two temperature
    derivatives, for an equation of state along with the
    vector quantities needed to compute the fugacitie and temperature
    derivatives of fugacities of the mixture. This
    routine is efficient in both numba and PyPy.

    .. math::
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}

    .. math::
        \frac{\partial (a\alpha)}{\partial T} = \sum_i \sum_j z_i z_j
        \frac{\partial (a\alpha)_{ij}}{\partial T}

    .. math::
        \frac{\partial^2 (a\alpha)}{\partial T^2} = \sum_i \sum_j z_i z_j
        \frac{\partial^2 (a\alpha)_{ij}}{\partial T^2}

    .. math::
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

    .. math::
        \frac{\partial (a\alpha)_{ij}}{\partial T} =
        \frac{\sqrt{\operatorname{a\alpha_{i}}{\left(T \right)} \operatorname{a\alpha_{j}}
        {\left(T \right)}} \left(1 - k_{ij}\right) \left(\frac{\operatorname{a\alpha_{i}}
        {\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}}{2}
        + \frac{\operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T} \operatorname{
        a\alpha_{i}}{\left(T \right)}}{2}\right)}{\operatorname{a\alpha_{i}}{\left(T \right)}
        \operatorname{a\alpha_{j}}{\left(T \right)}}

    .. math::
        \frac{\partial^2 (a\alpha)_{ij}}{\partial T^2} =
        - \frac{\sqrt{\operatorname{a\alpha_{i}}{\left(T \right)} \operatorname{a\alpha_{j}}
        {\left(T \right)}} \left(k_{ij} - 1\right) \left(\frac{\left(\operatorname{
        a\alpha_{i}}{\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}
        + \operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{i}}
        {\left(T \right)}\right)^{2}}{4 \operatorname{a\alpha_{i}}{\left(T \right)}
        \operatorname{a\alpha_{j}}{\left(T \right)}} - \frac{\left(\operatorname{a\alpha_{i}}
        {\left(T \right)} \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}
        + \operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T}
        \operatorname{a\alpha_{i}}{\left(T \right)}\right) \frac{d}{d T}
        \operatorname{a\alpha_{j}}{\left(T \right)}}{2 \operatorname{a\alpha_{j}}
        {\left(T \right)}} - \frac{\left(\operatorname{a\alpha_{i}}{\left(T \right)}
        \frac{d}{d T} \operatorname{a\alpha_{j}}{\left(T \right)}
        + \operatorname{a\alpha_{j}}{\left(T \right)} \frac{d}{d T}
        \operatorname{a\alpha_{i}}{\left(T \right)}\right) \frac{d}{d T}
        \operatorname{a\alpha_{i}}{\left(T \right)}}{2 \operatorname{a\alpha_{i}}
        {\left(T \right)}} + \frac{\operatorname{a\alpha_{i}}{\left(T \right)}
        \frac{d^{2}}{d T^{2}} \operatorname{a\alpha_{j}}{\left(T \right)}}{2}
        + \frac{\operatorname{a\alpha_{j}}{\left(T \right)} \frac{d^{2}}{d T^{2}}
        \operatorname{a\alpha_{i}}{\left(T \right)}}{2} + \frac{d}{d T}
        \operatorname{a\alpha_{i}}{\left(T \right)} \frac{d}{d T}
        \operatorname{a\alpha_{j}}{\left(T \right)}\right)}
        {\operatorname{a\alpha_{i}}{\left(T \right)} \operatorname{a\alpha_{j}}
        {\left(T \right)}}

    The secondary values are as follows:

    .. math::
        \sum_i y_i(a\alpha)_{ij}

    .. math::
        \sum_i y_i \frac{\partial (a\alpha)_{ij}}{\partial T}

    Parameters
    ----------
    a_alphas : list[float]
        EOS attractive terms, [J^2/mol^2/Pa]
    a_alpha_roots : list[float]
        Square roots of `a_alphas`; provided for speed [J/mol/Pa^0.5]
    da_alpha_dTs : list[float]
        Temperature derivative of coefficient calculated by EOS-specific
        method, [J^2/mol^2/Pa/K]
    d2a_alpha_dT2s : list[float]
        Second temperature derivative of coefficient calculated by
        EOS-specific method, [J^2/mol^2/Pa/K**2]
    T : float
        Temperature, not used, [K]
    zs : list[float]
        Mole fractions of each species
    kijs : list[list[float]]
        Constant kijs, [-]

    Returns
    -------
    a_alpha : float
        EOS attractive term, [J^2/mol^2/Pa]
    da_alpha_dT : float
        Temperature derivative of coefficient calculated by EOS-specific
        method, [J^2/mol^2/Pa/K]
    d2a_alpha_dT2 : float
        Second temperature derivative of coefficient calculated by
        EOS-specific method, [J^2/mol^2/Pa/K**2]
    a_alpha_j_rows : list[float]
        EOS attractive term row sums, [J^2/mol^2/Pa]
    da_alpha_dT_j_rows : list[float]
        Temperature derivative of EOS attractive term row sums, [J^2/mol^2/Pa/K]

    Notes
    -----

    Examples
    --------
    >>> kijs = [[0,.083],[0.083,0]]
    >>> zs = [0.1164203, 0.8835797]
    >>> a_alphas = [0.2491099357671155, 0.6486495863528039]
    >>> a_alpha_roots = [i**0.5 for i in a_alphas]
    >>> da_alpha_dTs = [-0.0005102028006086241, -0.0011131153520304886]
    >>> d2a_alpha_dT2s = [1.8651128859234162e-06, 3.884331923127011e-06]
    >>> a_alpha_and_derivatives_quadratic_terms(a_alphas, a_alpha_roots, da_alpha_dTs, d2a_alpha_dT2s, 299.0, zs, kijs)
    (0.58562139582, -0.001018667672, 3.56669817856e-06, [0.35469988173, 0.61604757237], [-0.000672387374, -0.001064293501])
    '''
    N = len(a_alphas)
    a_alpha = da_alpha_dT = d2a_alpha_dT2 = 0.0

#     da_alpha_dT_off = d2a_alpha_dT2_off = 0.0
#     a_alpha_j_rows = np.zeros(N)
    a_alpha_j_rows = [0.0]*N
#     da_alpha_dT_j_rows = np.zeros(N)
    da_alpha_dT_j_rows = [0.0]*N

    # If d2a_alpha_dT2s were all halved, could save one more multiply
    for i in range(N):
        kijs_i = kijs[i]
        a_alpha_i_root_i = a_alpha_roots[i]

        # delete these references?
        a_alphai = a_alphas[i]
        da_alpha_dT_i = da_alpha_dTs[i]
        d2a_alpha_dT2_i = d2a_alpha_dT2s[i]
        workingd1 = workings2 = 0.0

        for j in range(i):
            # TODO: optimize this, compute a_alpha after
            v0 = a_alpha_i_root_i*a_alpha_roots[j]
            a_alpha_ijs_ij = (1. - kijs_i[j])*v0
            t200 = a_alpha_ijs_ij*zs[i]
            a_alpha_j_rows[j] += t200
            a_alpha_j_rows[i] += zs[j]*a_alpha_ijs_ij
            t200 *= zs[j]
            a_alpha += t200 + t200

            a_alphaj = a_alphas[j]
            da_alpha_dT_j = da_alpha_dTs[j]
            zi_zj = zs[i]*zs[j]

            x1 = a_alphai*da_alpha_dT_j
            x2 = a_alphaj*da_alpha_dT_i
            x1_x2 = x1 + x2

            kij_m1 = kijs_i[j] - 1.0

            v0_inv = 1.0/v0
            v1 = kij_m1*v0_inv
            da_alpha_dT_ij = x1_x2*v1
#             da_alpha_dT_ij = -0.5*x1_x2*v1 # Factor the -0.5 out, apply at end
            da_alpha_dT_j_rows[j] += zs[i]*da_alpha_dT_ij
            da_alpha_dT_j_rows[i] += zs[j]*da_alpha_dT_ij

            da_alpha_dT_ij *= zi_zj

            x0 = a_alphai*a_alphaj

            # Technically could use a second list of double a_alphas, probably not used
            d2a_alpha_dT2_ij =  v0_inv*v0_inv*v1*(  (x0*(
                              -0.5*(a_alphai*d2a_alpha_dT2s[j] + a_alphaj*d2a_alpha_dT2_i)
                              - da_alpha_dT_i*da_alpha_dT_j) +.25*x1_x2*x1_x2))

            d2a_alpha_dT2_ij *= zi_zj
            workingd1 += da_alpha_dT_ij
            workings2 += d2a_alpha_dT2_ij
            # 23 multiplies, 1 divide in this loop


        # Simplifications for j=i, kij is always 0 by definition.
        t200 = a_alphas[i]*zs[i]
        a_alpha_j_rows[i] += t200
        a_alpha += t200*zs[i]
        zi_zj = zs[i]*zs[i]
        da_alpha_dT_ij = -da_alpha_dT_i - da_alpha_dT_i#da_alpha_dT_i*-2.0
        da_alpha_dT_j_rows[i] += zs[i]*da_alpha_dT_ij
        da_alpha_dT_ij *= zi_zj
        da_alpha_dT -= 0.5*(da_alpha_dT_ij + (workingd1 + workingd1))
        d2a_alpha_dT2 += d2a_alpha_dT2_i*zi_zj + (workings2 + workings2)
    for i in range(N):
        da_alpha_dT_j_rows[i] *= -0.5

    return a_alpha, da_alpha_dT, d2a_alpha_dT2, a_alpha_j_rows, da_alpha_dT_j_rows




def eos_mix_dV_dzs(T, P, Z, b, delta, epsilon, a_alpha, db_dzs, ddelta_dzs,
                   depsilon_dzs, da_alpha_dzs, N, out=None):
    if out is None:
        out = [0.0]*N
    T = T
    RT = R*T
    V = Z*RT/P

    x0 = delta
    x1 = a_alpha = a_alpha
    x2 = epsilon = epsilon

    x0V = x0*V
    Vmb = V - b
    x5 = Vmb*Vmb
    x1x5 = x1*x5
    x0x1x5 = x0*x1x5
    t0 = V*x1x5
    x6 = x2*x1x5
    x9 = V*V
    x7 = x9*t0
    x8 = x2*t0
    x10 = x0V + x2 + x9
    x10x10 = x10*x10
    x11 = R*T*x10*x10x10
    x13 = x0x1x5*x9
    x7x8 = x7 + x8

    t2 = -1.0/(x0V*x0x1x5 + x0*x6 - x11 + 3.0*x13 + x7x8 + x7x8)
    t1 = t2*x10x10*x5
    t3 = x0V*x1x5
    t4 = x1x5*x9
    t5 = t2*(t3 + t4 + x6)
    t6 = t2*(x13 + x7x8)
    x11t2 = x11*t2
    for i in range(N):
        out[i] = t5*depsilon_dzs[i] - t1*da_alpha_dzs[i] + x11t2*db_dzs[i] + t6*ddelta_dzs[i]
    return out


def G_dep_lnphi_d_helper(T, P, b, delta, epsilon, a_alpha, N,
    Z, dbs, depsilons, ddelta, dVs, da_alphas, G, out=None):
    if out is None:
        out = [0.0]*N

    x3 = b
    x4 = delta
    x5 = epsilon
    RT = R*T
    x0 = V = Z*RT/P

    x2 = 1.0/(RT)
    x6 = x4*x4 - 4.0*x5
    if x6 == 0.0:
        # VDW has x5 as zero as delta, epsilon = 0
        x6 = 1e-100
    x7 = x6**-0.5
    x8 = a_alpha
    x9 = x0 + x0
    x10 = x4 + x9
    x11 = x2 + x2
    x12 = x11*catanh(x10*x7).real
    x15 = 1.0/x6

    db_dns = dbs
    depsilon_dns = depsilons
    ddelta_dns = ddelta
    dV_dns = dVs
    da_alpha_dns = da_alphas

    t1 = P*x2
    t2 = x11*x15*x8/(x10*x10*x15 - 1.0)
    t3 = x12*x8*x6**(-1.5)
    t4 = x12*x7
    t5 = 1.0/(x0 - x3)
    t6 = x4 + x9

    if G:
        t1 *= RT
        t2 *= RT
        t3 *= RT
        t4 *= RT
        t5 *= RT

    for i in range(N):
        x13 = ddelta_dns[i]
        x14 = x13*x4 - 2.0*depsilon_dns[i]
        x16 = x14*x15
        x1 = dV_dns[i]
        diff = (x1*t1 + t2*(x1 + x1 + x13 - x16*t6) + x14*t3 - t4*da_alpha_dns[i] - t5*(x1 - db_dns[i]))
        out[i] = diff
    return out


def PR_translated_ddelta_dzs(b0s, cs, N, out=None):
    if out is None:
        out = [0.0]*N
    for i in range(N):
        out[i] = 2.0*(cs[i] + b0s[i])
    return out

def PR_translated_depsilon_dzs(epsilon, c, b, b0s, cs, N, out=None):
    if out is None:
        out = [0.0]*N
        b0 = b + c
    for i in range(N):
        out[i] = cs[i]*(2.0*b0 + c) + c*(2.0*b0s[i] + cs[i]) - 2.0*b0*b0s[i]
    return out


def PR_lnphis(T, P, Z, b, a_alpha, bs, a_alpha_j_rows, N, out=None):
    if out is None:
        out = [0.0]*N
    T_inv = 1.0/T
    P_T = P*T_inv

    A = a_alpha*P_T*R2_inv*T_inv
    B = b*P_T*R_inv
    x0 = log(Z - B)
    root_two_B = B*root_two
    two_root_two_B = root_two_B + root_two_B
    ZB = Z + B
    x4 = A*log((ZB + root_two_B)/(ZB - root_two_B))
    t50 = (x4 + x4)/(a_alpha*two_root_two_B)
    t51 = (x4 + (Z - 1.0)*two_root_two_B)/(b*two_root_two_B)
    for i in range(N):
        out[i] = bs[i]*t51 - x0 - t50*a_alpha_j_rows[i]
    return out

def VDW_lnphis(T, P, Z, b, a_alpha, bs, a_alpha_roots, out=None):
    N = len(bs)
    if out is None:
        out = [0.0]*N
    V = Z*R*T/P

    sqrt_a_alpha = sqrt(a_alpha)
    t1 = log(Z*(1. - b/V))
    t2 = 2.0*sqrt_a_alpha/(R*T*V)
    t3 = 1.0/(V - b)
    for i in range(N):
        out[i] = (bs[i]*t3 - t1 - t2*a_alpha_roots[i])
    return out

def lnphis_direct(zs, model, T, P, *args):
    if model == 10200:
        return PR_lnphis_fastest(zs, T, P, *args)
    return PR_lnphis_fastest(zs, T, P, *args)

def PR_lnphis_fastest(zs, T, P, kijs, l, g, bs, a_alphas, a_alpha_roots):
    # Uses precomputed values
    # Only creates its own arrays for a_alpha_j_rows and PR_lnphis
    N = len(bs)
    b = 0.0
    for i in range(N):
        b += bs[i]*zs[i]
    delta = 2.0*b
    epsilon = -b*b

    a_alpha, a_alpha_j_rows = a_alpha_quadratic_terms(a_alphas, a_alpha_roots, T, zs, kijs)
    V0, V1, V2 = volume_solutions_halley(T, P, b, delta, epsilon, a_alpha)
    if l:
        # Prefer liquid, ensure V0 is the smalest root
        if V1 != 0.0:
            if V0 > V1 and V1 > b:
                V0 = V1
            if V0 > V2 and V2 > b:
                V0 = V2
    elif g:
        if V1 != 0.0:
            if V0 < V1 and V1 > b:
                V0 = V1
            if V0 < V2 and V2 > b:
                V0 = V2
    else:
        raise ValueError("Root must be specified")
    Z = Z = P*V0/(R*T)
    return PR_lnphis(T, P, Z, b, a_alpha, bs, a_alpha_j_rows, N)
