r'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020, 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
.. autofunction:: a_alpha_and_derivatives_full

Compute only the alpha term itself:

.. autofunction:: a_alpha_and_derivatives

Faster implementations which do not store N^2 matrices:

.. autofunction:: a_alpha_quadratic_terms
.. autofunction:: a_alpha_and_derivatives_quadratic_terms
'''

# Direct fugacity calls
# ---------------------
# The object-oriented interface is quite convenient. However, sometimes it is
# desireable to perform a calculation at maximum speed, with no garbage collection
# and the only temperature-dependent parts re-used each calculation.
# For that reason, select equations of state have these functional forms
# implemented

# .. autofunction:: PR_lnphis
# .. autofunction:: PR_lnphis_fastest
__all__ = ['a_alpha_aijs_composition_independent',
           'a_alpha_and_derivatives', 'a_alpha_and_derivatives_full',
           'a_alpha_quadratic_terms', 'a_alpha_and_derivatives_quadratic_terms',
           'PR_lnphis', 'VDW_lnphis', 'SRK_lnphis', 'eos_mix_lnphis_general',

           'VDW_lnphis_fastest', 'PR_lnphis_fastest',
           'SRK_lnphis_fastest', 'RK_lnphis_fastest',
           'PR_translated_lnphis_fastest',

           'G_dep_lnphi_d_helper',

           'RK_d3delta_dninjnks',
           'PR_ddelta_dzs', 'PR_ddelta_dns',
           'PR_d2delta_dninjs', 'PR_d3delta_dninjnks',

           'PR_depsilon_dns', 'PR_d2epsilon_dninjs', 'PR_d3epsilon_dninjnks',
           'PR_d2epsilon_dzizjs', 'PR_depsilon_dzs',

           'PR_translated_d2delta_dninjs', 'PR_translated_d3delta_dninjnks',
           'PR_translated_d3epsilon_dninjnks',

           'PR_translated_ddelta_dzs', 'PR_translated_ddelta_dns',
           'PR_translated_depsilon_dzs', 'PR_translated_depsilon_dns',
           'PR_translated_d2epsilon_dzizjs', 'PR_translated_d2epsilon_dninjs',

           'SRK_translated_ddelta_dns', 'SRK_translated_depsilon_dns',
           'SRK_translated_d2epsilon_dzizjs', 'SRK_translated_depsilon_dzs',
           'SRK_translated_d2delta_dninjs',
           'SRK_translated_d3delta_dninjnks',
           'SRK_translated_d2epsilon_dninjs', 'SRK_translated_d3epsilon_dninjnks',


           'SRK_translated_lnphis_fastest',


           'eos_mix_db_dns', 'eos_mix_da_alpha_dns',

           'eos_mix_dV_dzs', 'eos_mix_a_alpha_volume']
from math import log, sqrt

from fluids.constants import R
from fluids.numerics import catanh

from thermo.eos import eos_G_dep, eos_lnphi
from thermo.eos_volume import volume_solutions_halley

R2 = R*R
R_inv = 1.0/R
R2_inv = R_inv*R_inv
root_two = sqrt(2.)
root_two_m1 = root_two - 1.0
root_two_p1 = root_two + 1.0

def a_alpha_aijs_composition_independent(a_alphas, one_minus_kijs, a_alpha_ijs=None,
a_alpha_roots=None, a_alpha_ij_roots_inv=None):
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
    one_minus_kijs : list[list[float]]
        One minus the constant kijs, [-]
    a_alpha_ijs : list[list[float]]
        Optional output array, [J^2/mol^2/Pa]
    a_alpha_roots : list[float]
        Optional output array, [J/mol/Pa^0.5]
    a_alpha_ij_roots_inv : list[list[float]]
        Optional output array, [mol^2*Pa/J^2]

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
    >>> one_minus_kijs = [[1.0 - kij for kij in row] for row in kijs]
    >>> a_alphas = [0.2491099357671155, 0.6486495863528039]
    >>> a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, one_minus_kijs)
    >>> a_alpha_ijs
    [[0.249109935767, 0.36861239374], [0.36861239374, 0.64864958635]]
    >>> a_alpha_roots
    [0.49910914213, 0.80538784840]
    >>> a_alpha_ij_roots_inv
    [[4.0142919105, 2.487707997796], [2.487707997796, 1.54166443799]]
    '''
    N = len(a_alphas)
    _sqrt = sqrt

    if a_alpha_ijs is None:
        a_alpha_ijs = [[0.0]*N for _ in range(N)] # numba: comment
#        a_alpha_ijs = np.zeros((N, N)) # numba: uncomment
    if a_alpha_roots is None:
        a_alpha_roots = [0.0]*N
    for i in range(N):
        a_alpha_roots[i] = _sqrt(a_alphas[i])

    if a_alpha_ij_roots_inv is None:
        a_alpha_ij_roots_inv = [[0.0]*N for _ in range(N)] # numba: comment
#        a_alpha_ij_roots_inv = np.zeros((N, N)) # numba: uncomment

    for i in range(N):
        one_minus_kijs_i = one_minus_kijs[i]
        a_alpha_ijs_is = a_alpha_ijs[i]
        a_alpha_ij_roots_i_inv = a_alpha_ij_roots_inv[i]
        # Using range like this saves 20% of the comp time for 44 components!
        a_alpha_i_root_i = a_alpha_roots[i]
        for j in range(i, N):
            term = a_alpha_i_root_i*a_alpha_roots[j]
            try:
                a_alpha_ij_roots_i_inv[j] = a_alpha_ij_roots_inv[j][i] = 1.0/term
            except:
                a_alpha_ij_roots_i_inv[j] = 1e100
            a_alpha_ijs_is[j] = a_alpha_ijs[j][i] = one_minus_kijs_i[j]*term
    return a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv




def a_alpha_and_derivatives(a_alphas, T, zs, one_minus_kijs, a_alpha_ijs=None,
                            a_alpha_roots=None, a_alpha_ij_roots_inv=None):
    N = len(a_alphas)
    da_alpha_dT, d2a_alpha_dT2 = 0.0, 0.0

    if a_alpha_ijs is None or a_alpha_roots is None or a_alpha_ij_roots_inv is None:
        a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, one_minus_kijs)

    a_alpha = 0.0
    for i in range(N):
        a_alpha_ijs_i = a_alpha_ijs[i]
        zi = zs[i]
        if zi > 0.0:
            for j in range(i+1, N):
                term = a_alpha_ijs_i[j]*zi*zs[j]
                a_alpha += term + term

            a_alpha += a_alpha_ijs_i[i]*zi*zi

    return a_alpha, None, a_alpha_ijs


def a_alpha_and_derivatives_full(a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, zs,
                                 one_minus_kijs, a_alpha_ijs=None, a_alpha_roots=None,
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
    one_minus_kijs : list[list[float]]
        One minus the constant kijs, [-]
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
    >>> one_minus_kijs = [[1.0 - kij for kij in row] for row in kijs]
    >>> zs = [0.1164203, 0.8835797]
    >>> a_alphas = [0.2491099357671155, 0.6486495863528039]
    >>> da_alpha_dTs = [-0.0005102028006086241, -0.0011131153520304886]
    >>> d2a_alpha_dT2s = [1.8651128859234162e-06, 3.884331923127011e-06]
    >>> a_alpha, da_alpha_dT, d2a_alpha_dT2, a_alpha_ijs, da_alpha_dT_ijs, d2a_alpha_dT2_ijs = a_alpha_and_derivatives_full(a_alphas=a_alphas, da_alpha_dTs=da_alpha_dTs, d2a_alpha_dT2s=d2a_alpha_dT2s, T=299.0, zs=zs, one_minus_kijs=one_minus_kijs)
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
        a_alpha_ijs, a_alpha_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, one_minus_kijs)

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
        one_minus_kijs_i = one_minus_kijs[i]
        a_alphai = a_alphas[i]
        z_products_i = z_products[i]
        da_alpha_dT_i = da_alpha_dTs[i]
        d2a_alpha_dT2_i = d2a_alpha_dT2s[i]
        a_alpha_ij_roots_inv_i = a_alpha_ij_roots_inv[i]
        da_alpha_dT_ijs_i = da_alpha_dT_ijs[i]

        if zs[i] > 0.0:
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

                kij_m1 = -one_minus_kijs_i[j]

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


def a_alpha_quadratic_terms(a_alphas, a_alpha_roots, T, zs, one_minus_kijs,
                            a_alpha_j_rows=None, vec0=None):
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
    one_minus_kijs : list[list[float]]
        One minus the constant kijs, [-]
    a_alpha_j_rows : list[float], optional
        EOS attractive term row destimation vector (does not need
        to be zeroed, should be provided to prevent allocations),
        [J^2/mol^2/Pa]
    vec0 : list[float], optional
        Empty vector, used in internal calculations, provide to avoid
        the allocations; does not need to be zeroed, [-]

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
    >>> one_minus_kijs = [[1.0 - kij for kij in row] for row in kijs]
    >>> zs = [0.1164203, 0.8835797]
    >>> a_alphas = [0.2491099357671155, 0.6486495863528039]
    >>> a_alpha_roots = [i**0.5 for i in a_alphas]
    >>> a_alpha, a_alpha_j_rows = a_alpha_quadratic_terms(a_alphas, a_alpha_roots, 299.0, zs, one_minus_kijs)
    >>> a_alpha, a_alpha_j_rows
    (0.58562139582, [0.35469988173, 0.61604757237])
    '''
    N = len(a_alphas)
    if a_alpha_j_rows is None:
        a_alpha_j_rows = [0.0]*N

    for i in range(N):
        a_alpha_j_rows[i] = 0.0

    if vec0 is None:
        vec0 = [0.0]*N
    for i in range(N):
        vec0[i] = a_alpha_roots[i]*zs[i]

    a_alpha = 0.0
    i = 0
    while i < N:
        one_minus_kijs_i = one_minus_kijs[i]
        j = 0
        while j < i:
            # Numba appears to be better with this split into two loops.
            # PyPy has 1.5x speed reduction when so.
            a_alpha_j_rows[j] += (one_minus_kijs_i[j])*vec0[i]
            a_alpha_j_rows[i] += (one_minus_kijs_i[j])*vec0[j]
            j += 1
        i += 1

    for i in range(N):
        a_alpha_j_rows[i] *= a_alpha_roots[i]
        a_alpha_j_rows[i] += (one_minus_kijs[i][i])*a_alphas[i]*zs[i]
        a_alpha += a_alpha_j_rows[i]*zs[i]

    return float(a_alpha), a_alpha_j_rows
    # This is faster in PyPy and can be made even faster optimizing a_alpha!
    """
    N = len(a_alphas)
    a_alpha_j_rows = [0.0]*N
    a_alpha = 0.0
    for i in range(N):
        kijs_i = kijs[i]
        a_alpha_i_root_i = a_alpha_roots[i]
        for j in range(i):
            a_alpha_ijs_ij = (1. - kijs_i[j])*a_alpha_i_root_i*a_alpha_roots[j]
            t200 = a_alpha_ijs_ij*zs[i]
            a_alpha_j_rows[j] += t200
            a_alpha_j_rows[i] += zs[j]*a_alpha_ijs_ij
            t200 *= zs[j]
            a_alpha += t200 + t200

        t200 = (1. - kijs_i[i])*a_alphas[i]*zs[i]
        a_alpha += t200*zs[i]
        a_alpha_j_rows[i] += t200

    return a_alpha, a_alpha_j_rows
    """


def a_alpha_and_derivatives_quadratic_terms(a_alphas, a_alpha_roots,
                                            da_alpha_dTs, d2a_alpha_dT2s, T,
                                            zs, one_minus_kijs, a_alpha_j_rows=None,
                                            da_alpha_dT_j_rows=None):
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
    one_minus_kijs : list[list[float]]
        One minus the constant kijs, [-]

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
    >>> one_minus_kijs = [[1.0 - kij for kij in row] for row in kijs]
    >>> zs = [0.1164203, 0.8835797]
    >>> a_alphas = [0.2491099357671155, 0.6486495863528039]
    >>> a_alpha_roots = [i**0.5 for i in a_alphas]
    >>> da_alpha_dTs = [-0.0005102028006086241, -0.0011131153520304886]
    >>> d2a_alpha_dT2s = [1.8651128859234162e-06, 3.884331923127011e-06]
    >>> a_alpha_and_derivatives_quadratic_terms(a_alphas, a_alpha_roots, da_alpha_dTs, d2a_alpha_dT2s, 299.0, zs, one_minus_kijs)
    (0.58562139582, -0.001018667672, 3.56669817856e-06, [0.35469988173, 0.61604757237], [-0.000672387374, -0.001064293501])
    '''
    N = len(a_alphas)
    a_alpha = da_alpha_dT = d2a_alpha_dT2 = 0.0

    if a_alpha_j_rows is None:
        a_alpha_j_rows = [0.0]*N
    if da_alpha_dT_j_rows is None:
        da_alpha_dT_j_rows = [0.0]*N

    # If d2a_alpha_dT2s were all halved, could save one more multiply
    for i in range(N):
        one_minus_kijs_i = one_minus_kijs[i]
        a_alpha_i_root_i = a_alpha_roots[i]

        # delete these references?
        a_alphai = a_alphas[i]
        da_alpha_dT_i = da_alpha_dTs[i]
        d2a_alpha_dT2_i = d2a_alpha_dT2s[i]
        workingd1 = workings2 = 0.0
        if a_alphai == 0.0 or zs[i] == 0.0:
            continue

        for j in range(i):
            # TODO: optimize this, compute a_alpha after
            v0 = a_alpha_i_root_i*a_alpha_roots[j]
            a_alpha_ijs_ij = (one_minus_kijs_i[j])*v0
            t200 = a_alpha_ijs_ij*zs[i]
            a_alpha_j_rows[j] += t200
            a_alpha_j_rows[i] += zs[j]*a_alpha_ijs_ij
            t200 *= zs[j]
            a_alpha += t200 + t200

            a_alphaj = a_alphas[j]
            if a_alphaj == 0.0:
                continue
            da_alpha_dT_j = da_alpha_dTs[j]
            zi_zj = zs[i]*zs[j]

            x1 = a_alphai*da_alpha_dT_j
            x2 = a_alphaj*da_alpha_dT_i
            x1_x2 = x1 + x2

            kij_m1 = -one_minus_kijs_i[j]

            v0_inv = 1.0/v0
            v1 = kij_m1*v0_inv
            da_alpha_dT_ij = x1_x2*v1
#             da_alpha_dT_ij = -0.5*x1_x2*v1 # Factor the -0.5 out, apply at end
            da_alpha_dT_j_rows[j] += zs[i]*da_alpha_dT_ij
            da_alpha_dT_j_rows[i] += zs[j]*da_alpha_dT_ij

            da_alpha_dT_ij *= zi_zj

            x0 = a_alphai*a_alphaj

            # Technically could use a second list of double a_alphas, probably not used
            d2a_alpha_dT2_ij =  v0_inv*v0_inv*v1*(  x0*(
                              -0.5*(a_alphai*d2a_alpha_dT2s[j] + a_alphaj*d2a_alpha_dT2_i)
                              - da_alpha_dT_i*da_alpha_dT_j) +.25*x1_x2*x1_x2)

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

    return float(a_alpha), float(da_alpha_dT), float(d2a_alpha_dT2), a_alpha_j_rows, da_alpha_dT_j_rows




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
    x11 = RT*x10*x10x10
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
    RT = R*T
    x0 = V = Z*RT/P

    x2 = 1.0/(RT)
    x6 = delta*delta - 4.0*epsilon
    if x6 == 0.0:
        # VDW has x5 as zero as delta, epsilon = 0
        x6 = 1e-100
    x7 = 1.0/sqrt(x6)
    x8 = a_alpha
    x9 = x0 + x0
    x10 = delta + x9
    x11 = x2 + x2
    x12 = x11*catanh(x10*x7).real
    x15 = x7*x7

    db_dns = dbs
    depsilon_dns = depsilons
    ddelta_dns = ddelta
    dV_dns = dVs
    da_alpha_dns = da_alphas

    t1 = P*x2
    t2 = x11*x15*x8/(x10*x10*x15 - 1.0)
    t3 = x12*x8*x15*x7
    t4 = x12*x7
    t5 = 1.0/(x0 - b)
    t6 = delta + x9

    if G:
        t1 *= RT
        t2 *= RT
        t3 *= RT
        t4 *= RT
        t5 *= RT

    c0 = t1 + t2*2.0 - t5
    for i in range(N):
        x13 = ddelta_dns[i]
        x14 = x13*delta - 2.0*depsilon_dns[i]
        x16 = x14*x15
        diff = (dV_dns[i]*c0 - t4*da_alpha_dns[i] + t5*db_dns[i]
                + t2*(x13 - x16*t6) + x14*t3 )
        # diff = (x1*t1 + t2*(x1 + x1 + x13 - x16*t6) + x14*t3 - t4*da_alpha_dns[i] - t5*(x1 - db_dns[i]))
        out[i] = diff
    return out

def eos_mix_a_alpha_volume(g, l, T, P, zs, one_minus_kijs, b, delta, epsilon, a_alphas, a_alpha_roots, a_alpha_j_rows=None, vec0=None):
    a_alpha, a_alpha_j_rows = a_alpha_quadratic_terms(a_alphas, a_alpha_roots, T, zs, one_minus_kijs, a_alpha_j_rows, vec0)

    V0, V1, V2 = volume_solutions_halley(T, P, b, delta, epsilon, a_alpha)
    if l and g:
        # Use the lowest Gibbs energy root
        V_low = V0
        if V1 != 0.0:
            if V_low > V1 and V1 > b:
                V_low = V1
            if V_low > V2 and V2 > b:
                V_low = V2
        V_high = V0
        if V1 != 0.0:
            if V_high < V1 and V1 > b:
                V_high = V1
            if V_high < V2 and V2 > b:
                V_high = V2
        if V_low != V_high:
            # two roots found, see which has the lowest
            G_V_low = eos_G_dep(T, P, V_low, b, delta, epsilon, a_alpha)
            G_V_high = eos_G_dep(T, P, V_high, b, delta, epsilon, a_alpha)
            V0 = V_low if G_V_low < G_V_high else V_high


    elif not g:
        # Prefer liquid, ensure V0 is the smalest root
        if V1 != 0.0:
            if V0 > V1 and V1 > b:
                V0 = V1
            if V0 > V2 and V2 > b:
                V0 = V2
    else:
        if V1 != 0.0:
            if V0 < V1 and V1 > b:
                V0 = V1
            if V0 < V2 and V2 > b:
                V0 = V2
    Z = P*V0/(R*T)
    return Z, a_alpha, a_alpha_j_rows

def eos_mix_db_dns(b, bs, N, out=None):
    if out is None:
        out = [0.0]*N
    for i in range(N):
        out[i] = bs[i] - b
    return out

def eos_mix_da_alpha_dns(a_alpha, a_alpha_j_rows, N, out=None):
    if out is None:
        out = [0.0]*N
    a_alpha_n_2 = -2.0*a_alpha
    for i in range(N):
        out[i] = 2.0*a_alpha_j_rows[i] + a_alpha_n_2
    return out

def RK_d3delta_dninjnks(b, bs, N, out=None):
    if out is None:
        out = [[[0.0]*N for _ in range(N)] for _ in range(N)]# numba: delete
        # out = np.zeros((N, N, N)) # numba: uncomment
    m6b = -6.0*b
    for i in range(N):
        bi = bs[i]
        d3b_dnjnks = out[i]
        for j in range(N):
            bj = bs[j]
            r = d3b_dnjnks[j]
            x0 = m6b + 2.0*(bi + bj)
            for k in range(N):
                r[k] = x0 + 2.0*bs[k]
    return out

def PR_ddelta_dzs(bs, N, out=None):
    if out is None:
        out = [0.0]*N
    for i in range(N):
        out[i] = 2.0*bs[i]
    return out

def PR_ddelta_dns(bs, b, N, out=None):
    if out is None:
        out = [0.0]*N
    nb2 = -2.0*b
    for i in range(N):
        out[i] = 2.0*bs[i] + nb2
    return out

def PR_depsilon_dns(b, bs, N, out=None):
    if out is None:
        out = [0.0]*N
    b2 = b + b
    b2b = b2*b
    for i in range(N):
        out[i] = b2b - b2*bs[i]
    return out

def PR_d2delta_dninjs(b, bs, N, out=None):
    if out is None:
        out = [[0.0]*N for _ in range(N)]# numba: delete
        # out = np.zeros((N, N)) # numba: uncomment
    bb = 2.0*b
    for i in range(N):
        bi = bs[i]
        r = out[i]
        x0 = 2.0*(bb - bi)
        for j in range(N):
            r[j] = x0 - 2.0*bs[j]
    return out

def PR_d3delta_dninjnks(b, bs, N, out=None):
    if out is None:
        out = [[[0.0]*N for _ in range(N)] for _ in range(N)]# numba: delete
        # out = np.zeros((N, N, N)) # numba: uncomment
    m3b = -3.0*b
    for i in range(N):
        bi = bs[i]
        d3b_dnjnks = out[i]
        for j in range(N):
            bj = bs[j]
            r = d3b_dnjnks[j]
            x0 = 4.0*(m3b + bi + bj)
            for k in range(N):
                r[k] = x0 + 4.0*bs[k]
    return out


def PR_d2epsilon_dzizjs(b, bs, N, out=None):
    if out is None:
        out = [[0.0]*N for _ in range(N)]# numba: delete
        # out = np.zeros((N, N)) # numba: uncomment

    for i in range(N):
        l = out[i]
        x0 = -2.0*bs[i]
        for j in range(N):
            l[j] = x0*bs[j]
    return out

def PR_depsilon_dzs(b, bs, N, out=None):
    if out is None:
        out = [0.0]*N
    b2n = -2.0*b

    for i in range(N):
        out[i] = b2n*bs[i]
    return out



def PR_d2epsilon_dninjs(b, bs, N, out=None):
    if out is None:
        out = [[0.0]*N for _ in range(N)]# numba: delete
        # out = np.zeros((N, N)) # numba: uncomment

    bb = b + b
    b2 = b*b
    c0 = -bb*bb - 2.0*b2
    c1 = 2.0*(b + 0.5*bb)
    c2 = 2.0*b + bb
    for i in range(N):
        l = out[i]
        bi = bs[i]
        x0 = c0 + c1*bi
        x1 = c2 - 2.0*bi
        for j in range(N):
            l[j] = x0 + bs[j]*x1
    return out


def PR_d3epsilon_dninjnks(b, bs, N, out=None):
    if out is None:
        out = [[[0.0]*N for _ in range(N)] for _ in range(N)]# numba: delete
        # out = np.zeros((N, N, N)) # numba: uncomment

    c0 = 24.0*b*b
    for i in range(N):
        bi = bs[i]
        d3b_dnjnks = out[i]
        c10 = -12.0*b + 4.0*bi
        c11 = c0 -12.0*b*bi
        c12 = (-12.0*b + 4.0*bi)

        for j in range(N):
            bj = bs[j]
            x0 = c11 + bj*c12
            x1 = c10 + 4.0*bj
            row = d3b_dnjnks[j]
            for k in range(N):
                bk = bs[k]
                term = x0 + bk*x1
                row[k] = term
    return out



def PR_translated_ddelta_dzs(b0s, cs, N, out=None):
    if out is None:
        out = [0.0]*N
    for i in range(N):
        out[i] = 2.0*(cs[i] + b0s[i])
    return out

def PR_translated_d2epsilon_dzizjs(b0s, cs, N, out=None):
    if out is None:
        out = [[0.0]*N for _ in range(N)] # numba: delete
        # out = np.zeros((N, N)) # numba: uncomment
    for j in range(N):
        r = out[j]
        x1 = 2.0*b0s[j]
        x2 = 2.0*cs[j]
        for i in range(N):
            # Optimized
            r[i] = x1*(cs[i] - b0s[i]) + x2*(b0s[i] + cs[i])
    return out

def PR_translated_d2epsilon_dninjs(b0s, cs, b, c, N, out=None):
    if out is None:
        out = [[0.0]*N for _ in range(N)] # numba: delete
        # out = np.zeros((N, N)) # numba: uncomment
    b0 = b + c

    v0 = -6.0*b0*b0 + 12.0*b0*c + 6.0*c*c
    v1 = 4.0*b0 - 4.0*c
    v2 = (-4.0*b0 - 4.0*c)


    for i in range(N):
        l = out[i]
        b0i = b0s[i]
        ci = cs[i]
        x0 = v0 + b0i*v1 + ci*v2
        x1 = v1 - 2.0*b0i + 2.0*ci
        x2 = v2 + 2.0*b0i + 2.0*ci
        for j in range(N):
            l[j] = x0 + b0s[j]*x1 +  cs[j]*x2
    return out

def PR_translated_ddelta_dns(b0s, cs, delta, N, out=None):
    if out is None:
        out = [0.0]*N
    for i in range(N):
        out[i] = 2.0*(cs[i] + b0s[i]) - delta
    return out

def SRK_translated_ddelta_dns(b0s, cs, delta, N, out=None):
    if out is None:
        out = [0.0]*N
    for i in range(N):
        out[i] = 2.0*cs[i] + b0s[i] - delta
    return out


def SRK_translated_depsilon_dns(b, c, b0s, cs, N, out=None):
    if out is None:
        out = [0.0]*N
    b0 = b + c
    x0 = -2.0*b0*c - 2.0*c*c
    x1 = (b0 + 2.0*c)
    for i in range(N):
        out[i] = x0 + b0s[i]*c + cs[i]*x1
    return out

def SRK_translated_depsilon_dzs(b0s, cs, b, c, N, out=None):
    if out is None:
        out = [0.0]*N
    b0 = b + c
    x0 = b0 + 2.0*c
    for i in range(N):
        out[i] = b0s[i]*c + cs[i]*x0
    return out

def SRK_translated_d2epsilon_dzizjs(b0s, cs, b, c, N, out=None):
    if out is None:
        out = [[0.0]*N for _ in range(N)] # numba: delete
        # out = np.zeros((N, N)) # numba: uncomment
    b0 = b + c
    for i in range(N):
        r = out[i]
        x0 = 2.0*cs[i]
        b0i = b0s[i]
        c0i = cs[i]
        for j in range(N):
            r[j] = cs[j]*(x0 + b0i) + b0s[j]*c0i
    return out


def SRK_translated_d2delta_dninjs(b0s, cs, b, c, delta, N, out=None):
    if out is None:
        out = [[0.0]*N for _ in range(N)] # numba: delete
        # out = np.zeros((N, N)) # numba: uncomment

    b0 = b + c
    c_4 = 4.0*c
    for i in range(N):
        t = delta - b0s[i] - cs[i]
        r = out[i]
        x0 = 2.0*(b0 - cs[i]) + c_4 - b0s[i]
        for j in range(N):
            r[j] =  x0 - b0s[j] - 2.0*cs[j]
    return out

def SRK_translated_d3delta_dninjnks(b0s, cs, b, c, delta, N, out=None):
    if out is None:
        out = [[[0.0]*N for _ in range(N)] for _ in range(N)] # numba: delete
        # out = np.zeros((N, N, N)) # numba: uncomment

    b0 = b + c
    for i in range(N):
        mat = out[i]
        for j in range(N):
            r = mat[j]
            for k in range(N):
                r[k] = (-6.0*b0 + 2.0*(b0s[i] + b0s[j] + b0s[k])
                - 12.0*c + 4.0*(cs[i] + cs[j] + cs[k]))
    return out

def SRK_translated_d2epsilon_dninjs(b0s, cs, b, c, N, out=None):
    if out is None:
        out = [[0.0]*N for _ in range(N)] # numba: delete
        # out = np.zeros((N, N)) # numba: uncomment
    b0 = b + c
    for i in range(N):
        l = out[i]
        for j in range(N):
            v = (b0*(2.0*c - cs[i] - cs[j]) + c*(2.0*b0 - b0s[i] - b0s[j])
            +2.0*c*(2.0*c - cs[i] - cs[j])
            + (b0 - b0s[i])*(c - cs[j])
            + (b0 - b0s[j])*(c - cs[i])
            + 2.0*(c - cs[i])*(c - cs[j])
            )
            l[j] = v
    return out

def SRK_translated_d3epsilon_dninjnks(b0s, cs, b, c, epsilon, N, out=None):
    if out is None:
        out = [[[0.0]*N for _ in range(N)] for _ in range(N)]# numba: delete
        # out = np.zeros((N, N, N)) # numba: uncomment

    b0 = b + c
    for i in range(N):
        d3b_dnjnks = out[i]
        for j in range(N):
            row = d3b_dnjnks[j]
            for k in range(N):
                term = (-2.0*b0*(3.0*c - cs[i] - cs[j] - cs[k])
                    - 2.0*c*(3.0*b0 - b0s[i] - b0s[j] - b0s[k])
                    - 4.0*c*(3.0*c - cs[i] - cs[j] - cs[k])
                    - (b0 - b0s[i])*(2.0*c - cs[j] - cs[k])
                    - (b0 - b0s[j])*(2.0*c - cs[i] - cs[k])
                    - (b0 - b0s[k])*(2.0*c - cs[i] - cs[j])
                    - (c - cs[i])*(2.0*b0 - b0s[j] - b0s[k])
                    - (c - cs[j])*(2.0*b0 - b0s[i] - b0s[k])
                    - (c - cs[k])*(2.0*b0 - b0s[i] - b0s[j])
                    - 2.0*(c - cs[i])*(2.0*c - cs[j] - cs[k])
                    - 2.0*(c - cs[j])*(2.0*c - cs[i] - cs[k])
                    - 2.0*(c - cs[k])*(2.0*c - cs[i] - cs[j])
                    )
                row[k] = term
    return out

def PR_translated_depsilon_dzs(c, b, b0s, cs, N, out=None):
    if out is None:
        out = [0.0]*N
    b0 = b + c
    b0_2 = b0*2.0
    x0 = (b0_2 + 2.0*c)
    x1 = c*2.0 - b0_2
    for i in range(N):
        out[i] = cs[i]*x0 + x1*b0s[i]
    return out

def PR_translated_depsilon_dns(b, c, b0s, cs, N, out=None):
    if out is None:
        out = [0.0]*N
    b0 = b + c
    x0 = 2.0*b0*b0 - 4.0*b0*c - 2.0*c*c
    x1 = -2.0*b0 + 2.0*c
    x2 = (2.0*b0 + 2.0*c)
    for i in range(N):
        out[i] = x0 + b0s[i]*x1 + cs[i]*x2
    return out

def PR_translated_d2delta_dninjs(b0s, cs, b, c, delta, N, out=None):
    if out is None:
        out = [[0.0]*N for _ in range(N)] # numba: delete
        # out = np.zeros((N, N)) # numba: uncomment

    b0 = b + c
    for i in range(N):
        t = delta - b0s[i] - cs[i]
        r = out[i]
        for j in range(N):
            r[j] = 2.0*(t - b0s[j] - cs[j])
    return out

def PR_translated_d3delta_dninjnks(b0s, cs, delta, N, out=None):
    if out is None:
        out = [[[0.0]*N for _ in range(N)] for _ in range(N)]# numba: delete
        # out = np.zeros((N, N, N)) # numba: uncomment

    delta_six = 6.0*delta
    for i in range(N):
        b0ici = b0s[i] + cs[i]
        d3b_dnjnks = out[i]
        for j in range(N):
            b0jcj = b0s[j] + cs[j]
            r = d3b_dnjnks[j]
            v0 = 4.0*(b0ici + b0jcj) - delta_six
            for k in range(N):
                r[k] = v0 + 4.0*(b0s[k] + cs[k])
    return out


def PR_translated_d3epsilon_dninjnks(b0s, cs, b, c, epsilon, N, out=None):
    if out is None:
        out = [[[0.0]*N for _ in range(N)] for _ in range(N)]# numba: delete
        # out = np.zeros((N, N, N)) # numba: uncomment

    b0 = b + c
    for i in range(N):
        d3b_dnjnks = out[i]
        for j in range(N):
            row = d3b_dnjnks[j]
            for k in range(N):
                term = (4.0*b0*(3.0*b0 - b0s[i] - b0s[j] - b0s[k])
                -2.0*c*(6.0*b0 + 3.0*c - 2.0*(b0s[i] + b0s[j] + b0s[k]) -(cs[i] + cs[j] + cs[k]))

                + 2.0*(b0 - b0s[i])*(2.0*b0 - b0s[j] - b0s[k])
                + 2.0*(b0 - b0s[j])*(2.0*b0 - b0s[i] - b0s[k])
                + 2.0*(b0 - b0s[k])*(2.0*b0 - b0s[i] - b0s[j])

                - (c - cs[i])*(4.0*b0 - 2.0*b0s[j] - 2.0*b0s[k] + 2.0*c - cs[j] - cs[k])
                - (c - cs[j])*(4.0*b0 - 2.0*b0s[i] - 2.0*b0s[k] + 2.0*c - cs[i] - cs[k])
                - (c - cs[k])*(4.0*b0 - 2.0*b0s[i] - 2.0*b0s[j] + 2.0*c - cs[i] - cs[j])

                - 2.0*(c + 2.0*b0)*(3.0*c - cs[i] - cs[j] - cs[k])

                - (2.0*c - cs[i] - cs[j])*(2.0*b0 + c - 2.0*b0s[k] - cs[k])
                - (2.0*c - cs[i] - cs[k])*(2.0*b0 + c - 2.0*b0s[j] - cs[j])
                - (2.0*c - cs[j] - cs[k])*(2.0*b0 + c - 2.0*b0s[i] - cs[i])

                )
                row[k] = term
    return out





def PR_lnphis(T, P, Z, b, a_alpha, bs, a_alpha_j_rows, N, lnphis=None):
    if lnphis is None:
        lnphis = [0.0]*N
    T_inv = 1.0/T
    P_T = P*T_inv

    A = a_alpha*P_T*R2_inv*T_inv
    B = b*P_T*R_inv
    to_log = Z - B
    if to_log <= 0.0:
        # Should never happen
        x0 = 0.0
    else:
        x0 = log(to_log)
    root_two_B = B*root_two
    two_root_two_B = root_two_B + root_two_B
    ZB = Z + B
    to_log = (ZB + root_two_B)/(ZB - root_two_B)
    if to_log <= 0.0:
        x4 = 0.0
    else:
        x4 = A*log(to_log)
    to_div = a_alpha*two_root_two_B
    if to_div != 0.0:
        t50 = (x4 + x4)/(to_div)
    else:
        for i in range(N):
            lnphis[i] = 0.0
            return lnphis
    t51 = (x4 + (Z - 1.0)*two_root_two_B)/(b*two_root_two_B)
    for i in range(N):
        lnphis[i] = bs[i]*t51 - x0 - t50*a_alpha_j_rows[i]
    return lnphis

def SRK_lnphis(T, P, Z, b, a_alpha, bs, a_alpha_j_rows, N, lnphis=None):
    if lnphis is None:
        lnphis = [0.0]*N
    RT = T*R
    P_RT = P/RT
    A = a_alpha*P/(RT*RT)
    B = b*P/RT
    B_inv = 1.0/B
    A_B = A*B_inv
    t0 = log(Z - B)
    t3 = log(1. + B/Z)
    Z_minus_one_over_B = (Z - 1.0)*B_inv
    two_over_a_alpha = 2./a_alpha
    x0 = A_B*B_inv*t3
    x1 = A_B*two_over_a_alpha*t3
    x2 = (Z_minus_one_over_B + x0)*P_RT
    for i in range(N):
        lnphis[i] = bs[i]*x2 - t0 - x1*a_alpha_j_rows[i]
    return lnphis

def VDW_lnphis(T, P, Z, b, a_alpha, bs, a_alpha_roots, N, lnphis=None):
    if lnphis is None:
        lnphis = [0.0]*N
    V = Z*R*T/P

    sqrt_a_alpha = sqrt(a_alpha)
    t1 = log(Z*(1. - b/V))
    t2 = 2.0*sqrt_a_alpha/(R*T*V)
    t3 = 1.0/(V - b)
    for i in range(N):
        lnphis[i] = (bs[i]*t3 - t1 - t2*a_alpha_roots[i])
    return lnphis

def eos_mix_lnphis_general(T, P, Z, b, delta, epsilon, a_alpha, bs,
                           a_alpha_roots, N, db_dns, da_alpha_dns, ddelta_dns,
                           depsilon_dns, lnphis=None):
    if lnphis is None:
        lnphis = [0.0]*N
    V = Z*R*T/P
    dV_dns = eos_mix_dV_dzs(T, P, Z, b, delta, epsilon,
                            a_alpha, db_dns, ddelta_dns,
                            depsilon_dns, da_alpha_dns, N)

    dlnphi_dns = G_dep_lnphi_d_helper(T, P, b, delta, epsilon, a_alpha, N,
                                      Z, db_dns, depsilon_dns, ddelta_dns, dV_dns,
                                      da_alpha_dns, G=False)

    lnphi = eos_lnphi(T, P, V, b, delta, epsilon, a_alpha)
    for i in range(N):
        lnphis[i] = lnphi + dlnphi_dns[i]
    return lnphis





def PR_lnphis_fastest(zs, T, P, N, one_minus_kijs, l, g, bs, a_alphas, a_alpha_roots, a_alpha_j_rows=None, vec0=None,
                      lnphis=None):
    b = 0.0
    for i in range(N):
        b += bs[i]*zs[i]
    delta = 2.0*b
    epsilon = -b*b

    Z, a_alpha, a_alpha_j_rows = eos_mix_a_alpha_volume(g, l, T, P, zs, one_minus_kijs, b, delta, epsilon, a_alphas, a_alpha_roots,
                                                        a_alpha_j_rows=a_alpha_j_rows, vec0=vec0)
    return PR_lnphis(T, P, Z, b, a_alpha, bs, a_alpha_j_rows, N, lnphis=lnphis)


def SRK_lnphis_fastest(zs, T, P, N, one_minus_kijs, l, g, bs, a_alphas, a_alpha_roots, a_alpha_j_rows=None, vec0=None,
                      lnphis=None):
    b = 0.0
    for i in range(N):
        b += bs[i]*zs[i]
    delta = b
    epsilon = 0.0

    Z, a_alpha, a_alpha_j_rows = eos_mix_a_alpha_volume(g, l, T, P, zs, one_minus_kijs, b, delta, epsilon, a_alphas, a_alpha_roots,
                                                        a_alpha_j_rows=a_alpha_j_rows, vec0=vec0)
    return SRK_lnphis(T, P, Z, b, a_alpha, bs, a_alpha_j_rows, N, lnphis=lnphis)


def VDW_lnphis_fastest(zs, T, P, N, one_minus_kijs, l, g, bs, a_alphas, a_alpha_roots, a_alpha_j_rows=None, vec0=None,
                      lnphis=None):
    b = 0.0
    for i in range(N):
        b += bs[i]*zs[i]
    delta = 0.0
    epsilon = 0.0

    Z, a_alpha, a_alpha_j_rows = eos_mix_a_alpha_volume(g, l, T, P, zs, one_minus_kijs, b, delta, epsilon, a_alphas, a_alpha_roots,
                                                        a_alpha_j_rows=a_alpha_j_rows, vec0=vec0)
    return VDW_lnphis(T, P, Z, b, a_alpha, bs, a_alpha_roots, N, lnphis=lnphis)


def RK_lnphis_fastest(zs, T, P, N, one_minus_kijs, l, g, bs, a_alphas, a_alpha_roots, a_alpha_j_rows=None, vec0=None,
                      lnphis=None):
    b = 0.0
    for i in range(N):
        b += bs[i]*zs[i]
    delta = b
    epsilon = 0.0

    Z, a_alpha, a_alpha_j_rows = eos_mix_a_alpha_volume(g, l, T, P, zs, one_minus_kijs, b, delta, epsilon, a_alphas, a_alpha_roots,
                                                        a_alpha_j_rows=a_alpha_j_rows, vec0=vec0)

    ddelta_dns = db_dns = eos_mix_db_dns(b, bs, N, out=None)
    da_alpha_dns = eos_mix_da_alpha_dns(a_alpha, a_alpha_j_rows, N, out=None)
    depsilon_dns = [0.0]*N


    return eos_mix_lnphis_general(T, P, Z, b, delta, epsilon, a_alpha, bs,
                           a_alpha_roots, N, db_dns, da_alpha_dns, ddelta_dns,
                           depsilon_dns, lnphis=lnphis)


def PR_translated_lnphis_fastest(zs, T, P, N, one_minus_kijs, l, g, b0s, bs, cs, a_alphas, a_alpha_roots, a_alpha_j_rows=None, vec0=None,
                      lnphis=None):
    b0, c = 0.0, 0.0
    for i in range(N):
        b0 += b0s[i]*zs[i]
        c += cs[i]*zs[i]

    b = b0 - c
    delta = 2.0*(c + b0)
    epsilon = -b0*b0 + c*(c + b0 + b0)
    Z, a_alpha, a_alpha_j_rows = eos_mix_a_alpha_volume(g, l, T, P, zs, one_minus_kijs, b, delta, epsilon, a_alphas, a_alpha_roots,
                                                        a_alpha_j_rows=a_alpha_j_rows, vec0=vec0)

    db_dns = eos_mix_db_dns(b, bs, N, out=None)
    da_alpha_dns = eos_mix_da_alpha_dns(a_alpha, a_alpha_j_rows, N, out=None)
    depsilon_dns = PR_translated_depsilon_dns(b, c, b0s, cs, N, out=None)
    ddelta_dns = PR_translated_ddelta_dns(b0s, cs, delta, N, out=None)


    return eos_mix_lnphis_general(T, P, Z, b, delta, epsilon, a_alpha, bs,
                           a_alpha_roots, N, db_dns, da_alpha_dns, ddelta_dns,
                           depsilon_dns, lnphis=lnphis)


def SRK_translated_lnphis_fastest(zs, T, P, N, one_minus_kijs, l, g, b0s, bs, cs, a_alphas, a_alpha_roots, a_alpha_j_rows=None, vec0=None,
                      lnphis=None):
    b0, c = 0.0, 0.0
    for i in range(N):
        b0 += b0s[i]*zs[i]
        c += cs[i]*zs[i]

    b = b0 - c
    delta =  c + c + b0
    epsilon = c*(b0 + c)
    Z, a_alpha, a_alpha_j_rows = eos_mix_a_alpha_volume(g, l, T, P, zs, one_minus_kijs, b, delta, epsilon, a_alphas, a_alpha_roots,
                                                        a_alpha_j_rows=a_alpha_j_rows, vec0=vec0)

    db_dns = eos_mix_db_dns(b, bs, N, out=None)
    da_alpha_dns = eos_mix_da_alpha_dns(a_alpha, a_alpha_j_rows, N, out=None)
    depsilon_dns = SRK_translated_depsilon_dns(b, c, b0s, cs, N, out=None)
    ddelta_dns = SRK_translated_ddelta_dns(b0s, cs, delta, N, out=None)


    return eos_mix_lnphis_general(T, P, Z, b, delta, epsilon, a_alpha, bs,
                           a_alpha_roots, N, db_dns, da_alpha_dns, ddelta_dns,
                           depsilon_dns, lnphis=lnphis)

