# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from __future__ import division

__all__ = ['standard_entropy', 'S0_basis_converter']

import os
from fluids.numerics import quad, numpy as np
from chemicals.utils import isnan


def standard_entropy(c=None, dS_trans_s=None, dH_trans_s=None, T_trans_s=None,
                     Cp_s_fun=None,
                     Sfusm=None, Hfusm=None, Tm=None, Cp_l_fun=None,
                     Svapm=None, Hvapm=None, Tb=None, Cp_g_fun=None,
                     T_ref=298.15, T_low=0.5, force_gas=True):
    if Tm is None:
        Tm = c.Tm
    if Tb is None:
        Tb = c.Tb
    if Hfusm is None:
        Hfusm = c.Hfusm
    if Hvapm is None:
        Hvapm = c.EnthalpyVaporization(Tb)

    # Misc crystalline transitions
    tot = 0.0
    if dS_trans_s is not None:
        tot += sum(dS_trans_s)
    if dH_trans_s is not None and T_trans_s is not None:
        for dH, T in zip(dH_trans_s, T_trans_s):
            if T < T_ref or force_gas:
                tot += dH/T

    # Solid heat capacity integral
    if Cp_s_fun is not None:
        tot += float(quad(lambda T: Cp_s_fun(T)/T, T_low, Tm)[0])
    else:
        tot += c.HeatCapacitySolid.T_dependent_property_integral_over_T(T_low, Tm)

    # Heat of fusion
    if force_gas or Tm < T_ref:
        if Sfusm is not None:
            tot += Sfusm
        else:
            tot += Hfusm/Tm

    # Liquid heat capacity
    if not force_gas and Tb > T_ref:
        T_liquid_int = T_ref
    else:
        T_liquid_int = Tb
    if force_gas or Tm < T_ref:
        if Cp_l_fun is not None:
            tot += float(quad(lambda T: Cp_l_fun(T)/T, Tm, T_liquid_int)[0])
        else:
            tot += c.HeatCapacityLiquid.T_dependent_property_integral_over_T(Tm, T_liquid_int)

    # Heat of vaporization
    if force_gas or Tb < T_ref:
        if Svapm is not None:
            tot += Svapm
        else:
            tot += Hvapm/Tb

    if force_gas or Tb < T_ref:
        # gas heat capacity
        if Cp_g_fun is not None:
            tot += float(quad(lambda T: Cp_g_fun(T)/T, Tb, T_ref)[0])
        else:
            tot += c.HeatCapacityGas.T_dependent_property_integral_over_T(Tb, T_ref)

    return tot


def S0_basis_converter(c, S0_liq=None, S0_gas=None, T_ref=298.15):
    r'''This function converts a liquid or gas standard entropy to the
    other. This is useful, as thermodynamic packages often work with ideal-
    gas as the reference state and require ideal-gas Gibbs energies of
    formation.

    Parameters
    ----------
    c : Chemical
        Chemical object, [-]
    S0_liq : float, optional
        Liquid absolute entropy of the compound at the reference temperature
        [J/mol/K]
    S0_gas : float, optional
        Gas absolute entropy of the compound at the reference temperature
        [J/mol/K]
    T_ref : float, optional
        The standard state temperature, default 298.15 K; few values are
        tabulated at other temperatures, [-]

    Returns
    -------
    S0_calc : float
        Standard absolute entropy of the compound at the reference temperature
        in the other state to the one provided, [J/mol]

    Notes
    -----
    This function relies in accurate heat capacity curves for both the liquid
    and gas state.

    Examples
    --------
    >>> from thermo.chemical import Chemical
    >>> S0_basis_converter(Chemical('decane'), S0_liq=425.89)
    544.6792
    >>> S0_basis_converter(Chemical('decane'), S0_gas=545.7)
    426.9107
    '''
    if S0_liq is None and S0_gas is None:
        raise ValueError("Provide either a liquid or a gas standard absolute entropy")
    if S0_liq is None:
        dS = c.HeatCapacityGas.T_dependent_property_integral_over_T(T_ref, c.Tb)
        dS -= c.EnthalpyVaporization(c.Tb)/c.Tb
        dS += c.HeatCapacityLiquid.T_dependent_property_integral_over_T(c.Tb, T_ref)
        return S0_gas + dS
    else:
        dS = c.HeatCapacityLiquid.T_dependent_property_integral_over_T(T_ref, c.Tb)
        dS += c.EnthalpyVaporization(c.Tb)/c.Tb
        dS += c.HeatCapacityGas.T_dependent_property_integral_over_T(c.Tb, T_ref)
        return S0_liq + dS

