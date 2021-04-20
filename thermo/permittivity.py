# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains implementations of :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
representing liquid permittivity. A variety of estimation
and data methods are available as included in the `chemicals` library.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Pure Liquid Permittivity
========================
.. autoclass:: PermittivityLiquid
    :members: calculate, test_method_validity,
              name, property_max, property_min,
              units, Tmin, Tmax, ranked_methods
    :undoc-members:
    :show-inheritance:
    :exclude-members:

.. autodata:: permittivity_methods
'''

from __future__ import division

__all__ = ['PermittivityLiquid']

import os
from fluids.numerics import numpy as np
from fluids.constants import N_A, epsilon_0, k

from chemicals.utils import isnan
from chemicals.permittivity import permittivity_IAPWS
from chemicals import permittivity

from thermo.utils import TDependentProperty


CRC = 'CRC'
CRC_CONSTANT = 'CRC_CONSTANT'
permittivity_methods = [CRC, CRC_CONSTANT]
'''Holds all methods available for the :obj:`PermittivityLiquid` class, for use in
iterating over them.'''


class PermittivityLiquid(TDependentProperty):
    r'''Class for dealing with liquid permittivity as a function of temperature.
    Consists of one temperature-dependent simple expression and one constant
    value source.

    Parameters
    ----------
    CASRN : str, optional
        The CAS number of the chemical
    load_data : bool, optional
        If False, do not load property coefficients from data sources in files
        [-]
    extrapolation : str or None
        None to not extrapolate; see
        :obj:`TDependentProperty <thermo.utils.TDependentProperty>`
        for a full list of all options, [-]
    poly_fit : tuple(float, float, list[float]), optional
        Tuple of (Tmin, Tmax, coeffs) representing a prefered fit to the
        permittivity of the compound; the coefficients are evaluated with
        horner's method, and the input variable and output are transformed by
        the default transformations of this object; used instead of any other
        default method if provided. [-]
    method : str or None, optional
        If specified, use this method by default and do not use the ranked
        sorting; an exception is raised if this is not a valid method for the
        provided inputs, [-]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`permittivity_methods`.

    **CRC**:
        Simple polynomials for calculating permittivity over a specified
        temperature range only. The full expression is:

        .. math::
            \epsilon_r = A + BT + CT^2 + DT^3

        Not all chemicals use all terms; in fact, few do. Data is available
        for 759 liquids, from [1]_.
    **CRC_CONSTANT**:
        Constant permittivity values at specified temperatures only.
        Data is from [1]_, and is available for 1303 liquids.

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    '''
    name = 'liquid relative permittivity'
    units = '-'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default.'''
    property_min = 1.0
    '''Relative permittivity must always be larger than 1; nothing is better
    than a vacuum.'''
    property_max = 1000.0
    '''Maximum valid of permittivity; highest in the data available is ~240.'''

    ranked_methods = [CRC, CRC_CONSTANT]
    '''Default rankings of the available methods.'''

    _fit_force_n = {}
    '''Dictionary containing method: fit_n, for use in methods which should
    only ever be fit to a specific `n` value'''
    _fit_force_n[CRC_CONSTANT] = 1

    _fit_max_n = {CRC: 4}

    @staticmethod
    def _method_indexes():
        '''Returns a dictionary of method: index for all methods
        that use data files to retrieve constants. The use of this function
        ensures the data files are not loaded until they are needed.
        '''
        A = permittivity.permittivity_data_CRC['A'].values
        return {CRC_CONSTANT: permittivity.permittivity_data_CRC.index,
                CRC: [CAS for i, CAS in enumerate(permittivity.permittivity_data_CRC.index)
                      if not isnan(A[i])],
                }

    custom_args = ()

    def __init__(self, CASRN='', extrapolation='linear', **kwargs):
        self.CASRN = CASRN
        super(PermittivityLiquid, self).__init__(extrapolation, **kwargs)

    def load_all_methods(self, load_data):
        r'''Method which picks out coefficients for the specified chemical
        from the various dictionaries and DataFrames storing it. All data is
        stored as attributes. This method also sets :obj:`Tmin`, :obj:`Tmax`,
        and :obj:`all_methods` as a set of methods for which the data exists for.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods = []
        Tmins, Tmaxs = [], []
        self.T_limits = T_limits = {}
        if load_data:
            if self.CASRN in permittivity.permittivity_data_CRC.index:
                methods.append(CRC_CONSTANT)
                self.CRC_CONSTANT_T, self.CRC_permittivity, A, B, C, D, Tmin, Tmax = permittivity.permittivity_values_CRC[permittivity.permittivity_data_CRC.index.get_loc(self.CASRN)].tolist()
                if isnan(Tmin) and isnan(Tmax):
                    Tmin, Tmax = self.CRC_CONSTANT_T, self.CRC_CONSTANT_T
                T_limits[CRC_CONSTANT] = (self.CRC_CONSTANT_T, self.CRC_CONSTANT_T)
                self.CRC_coeffs = [0 if isnan(x) else x for x in [A, B, C, D] ]
                self.CRC_Tmin = Tmin
                self.CRC_Tmax = Tmax
                if self.CRC_coeffs[0] and not isnan(Tmin):
                    Tmins.append(Tmin); Tmaxs.append(Tmax)
                    methods.append(CRC)
                    T_limits[CRC] = (Tmin, Tmax)

        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin = min(Tmins)
            self.Tmax = max(Tmaxs)

    def calculate(self, T, method):
        r'''Method to calculate permittivity of a liquid at temperature `T`
        with a given method.

        This method has no exception handling; see :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate relative permittivity, [K]
        method : str
            Name of the method to use

        Returns
        -------
        epsilon : float
            Relative permittivity of the liquid at T, [-]
        '''
        if method == CRC:
            A, B, C, D = self.CRC_coeffs
            epsilon = A + T*(B + T*(C + D*T))
        elif method == CRC_CONSTANT:
            epsilon = self.CRC_permittivity
        elif method in self.tabular_data:
            epsilon = self.interpolate(T, method)
        return epsilon

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For tabular data,
        extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the
        extrapolation is considered valid for all temperatures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.

        Parameters
        ----------
        T : float
            Temperature at which to test the method, [K]
        method : str
            Name of the method to test

        Returns
        -------
        validity : bool
            Whether or not a method is valid
        '''
        validity = True
        if method == CRC:
            if T < self.CRC_Tmin or T > self.CRC_Tmax:
                validity = False
        elif method == CRC_CONSTANT:
            # Arbitraty choice of temperature limits
            if T < self.CRC_CONSTANT_T - 20 or T > self.CRC_CONSTANT_T + 20:
                validity = False
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        else:
            raise ValueError('Method not valid')
        return validity

