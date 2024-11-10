'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2023 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
'''

__all__ = ['redlich_kister_reverse','redlich_kister_reverse_2d', 'redlich_kister_excess_inner',
'redlich_kister_build_structure', 'redlich_kister_T_dependence', 'redlich_kister_excess_inner_binary',
'redlich_kister_excess_binary', 'redlich_kister_fitting_to_use']
from math import log


def redlich_kister_reverse_2d(data_2d):
    data_2d = [d.copy() for d in data_2d]
    params = len(data_2d)
    T_dep_params = len(data_2d[0])
    for i, k in enumerate(range(params)):
        if k%2 == 1:
            for j in range(T_dep_params):
                data_2d[i][j] = -data_2d[i][j]
    return data_2d


def redlich_kister_reverse(data_1d):
    data_1d = data_1d.copy()
    params = len(data_1d)
    for i, k in enumerate(range(params)):
        if k%2 == 1:
            data_1d[i] = -data_1d[i]
    return data_1d


def redlich_kister_excess_inner(N, N_terms, a_tensor, xs):
    r'''Calculate the excess property for a given set of composition
    and temperature-independent constants.

    .. math::
        E = 0.5 \sum_i^N \sum_j^N \sum_k^{N_terms} \A_{ijk} x_i x_j (x_i - x_j)^k

    Parameters
    ----------
    N : int
        The number of components in the mixture, [-]
    N_terms : int
        The number of terms in the redlich-kister expansion. This must
        be the same for each component interaction; pad with zeros, [-]
    a_tensor : list[list[list[float]]]
        RK parameters, indexed by [i][j][k], [-]
    xs : list[float]
        Mole fractions, [-]

    Returns
    -------
    excess : float
        The calculated excess value; in some cases this is
        dimensionless and multiplied by a constant like `R*T`
        or `1000 kg/m^3` to obtain its dimensionality and
        in other cases it has the dimensions of the property
        being calculated like `N/m` or `Pa*s`, [-]

    Notes
    -----
    Temperature dependent constants should be calculated
    in an outer function.

    '''
    excess = 0.0
    for i in range(N):
        outer = 0.0
        for j in range(N):
#             if i ==j:
#                 continue
            inner = 0.0
            factor = 1.0
            diff = (xs[i] - xs[j])
            for k in range(N_terms):
                inner += a_tensor[i][j][k]*factor
                factor *= diff
            outer += inner*xs[j]
        excess += outer*xs[i]
    return excess*0.5

def redlich_kister_build_structure(N, shape, data, indexes):
    r'''Builds a redlich-kister compatible
    tensor (data structure) from pairs of indexes, and
    the coefficients associated with those indexes.
    This is especially important because of the asymmetry
    of the model.

    Parameters
    ----------
    N : int
        The number of components in the mixture, [-]
    shape : tuple(int,) of dimensionality desired
        The parameter to specify the shape of each of the
        [i][j] component data structures, [-]
    data : list[float] or list[list[float]]
        The coefficients regressed or from literature, [various]
    indexes : list[tuple(int, int)]
        A list of index pairs. This is used to determine
        which components are associated with what data.
        Order matters! (i,j) != (j, i)

    Returns
    -------
    structure : list[list[data]] of shape [N, N, *shape]
        Output structure

    Notes
    -----
    This is a fairly computationally intensive calculation
    and should be cached
    '''
    out = []

    if len(shape) == 1:
        one_d = True
        two_d = False
    elif len(shape) == 2:
        one_d = False
        two_d = True
    else:
        raise ValueError("Shape must be provided")

    if len(indexes) != len(data):
        raise ValueError("Index and data length must be the same")

    data_dict = {idx: d for idx, d in zip(indexes, data)}

    for i in range(N):
        l = []
        for j in range(N):
            if one_d:
                l.append([0.0]*shape[0])
            else:
                l.append([[0.0]*shape[1] for _ in range(shape[0])])
        out.append(l)

    for i in range(N):
        for j in range(N):
            if (i, j) in data_dict:
                thing = data_dict[(i, j)]
                out[i][j] = thing
            elif (j, i) in data_dict:
                thing = data_dict[(j, i)]
                if one_d:
                    thing = redlich_kister_reverse(thing)
                else:
                    thing = redlich_kister_reverse_2d(thing)
                out[i][j] = thing
    return out


def redlich_kister_T_dependence(structure, T, N, N_terms, N_T):
    r'''Compute a redlich-kister set of A coefficients from
    a temperature-dependent data struture with number
    of temperature-dependent terms `N_T`. Essentially
    this takes a 4d array and returns a 3d one.

    Parameters
    ----------
    structure : list[list[list[float]]] of shape [N, N, N_terms, N_T]
        Input structure with temperature dependence coefficients, [various]
    T : float
        Temperature, [K]
    N : int
        The number of components in the mixture, [-]
    N_terms : int
        The number of terms in the expansion, [-]
    N_T : int
        The number of terms in the temperature expansion, [-]

    Returns
    -------
    structure : list[list[data]] of shape [N, N, N_terms]
        Output structure

    Notes
    -----
    '''
    T2 = T*T
    Tinv = 1.0/T
    T2inv = Tinv*Tinv
    logT = log(T)
    out = [[[0.0]*N_terms for _ in range(N)] for _ in range(N)]

    if N_T == 2:
        for i in range(N):
            out_2d = out[i]
            in_3d = structure[i]
            for j in range(N):
                out_1d = out_2d[j]
                in_2d = in_3d[j]
                for k in range(N_terms):
                    in_1d = in_2d[k]
                    out_1d[k] = in_1d[0] + in_1d[1]*Tinv
    elif N_T == 3:
        for i in range(N):
            out_2d = out[i]
            in_3d = structure[i]
            for j in range(N):
                out_1d = out_2d[j]
                in_2d = in_3d[j]
                for k in range(N_terms):
                    in_1d = in_2d[k]
                    out_1d[k] = in_1d[0] + in_1d[1]*Tinv + in_1d[2]*logT
    elif N_T == 6:
        for i in range(N):
            out_2d = out[i]
            in_3d = structure[i]
            for j in range(N):
                out_1d = out_2d[j]
                in_2d = in_3d[j]
                for k in range(N_terms):
                    in_1d = in_2d[k]
                    out_1d[k] = in_1d[0] + in_1d[1]*Tinv + in_1d[2]*logT + in_1d[3]*T + in_1d[4]*T2inv + in_1d[5]*T2
    else:
        raise ValueError("Unsupported number of terms")
    return out


def redlich_kister_excess_inner_binary(ais, xs):
    r'''Compute the redlich-kister excess for a binary
    system. This calculation is optimized. This works
    with the same values of coefficients as `redlich_kister_excess_inner`
    but without the excess dimensionality of the input
    data.

    Parameters
    ----------
    ais : list[float] of size N_terms
        `A` coefficients`, [-]
    xs : list[float]
        Binary mole fractions, [-]

    Returns
    -------
    excess : float
        The calculated excess value [-]

    Notes
    -----
    '''
    x0, x1 = xs
    x_diff = (x0 - x1)
    x_product = x0*x1
    terms = len(ais)
    GE = 0.0
    factor = 1.0
    for i in range(terms):
        GE += ais[i]*x_product*factor
        factor *= x_diff
    return GE

def redlich_kister_excess_binary(coefficients, x0, T, N_T, N_terms):
    T2 = T*T
    Tinv = 1.0/T
    T2inv = Tinv*Tinv
    logT = log(T)
    x1 = 1.0 - x0
    x_diff = (x0 - x1)
    x_product = x0*x1
    GE = 0.0
    factor = 1.0
    for i in range(N_terms):
        offset = i*N_T
        ai = coefficients[offset]
        if N_T >= 2:
            ai +=  coefficients[offset+1]*Tinv
        if N_T >= 3:
            ai += coefficients[offset+2]*logT
        if N_T >= 4:
            ai +=  coefficients[offset+3]*T
        if N_T >= 5:
            ai += coefficients[offset+4]*T2inv
        if N_T >= 6:
            ai += coefficients[offset+5]*T2
        GE += ai*x_product*factor
        factor *= x_diff
    return GE

def redlich_kister_fitting_to_use(coeffs, N_terms, N_T):
    out = []
    for i in range(N_terms):
        l = []
        for j in range(N_T):
            l.append(coeffs[i*N_T + j])
        out.append(l)
    return out
