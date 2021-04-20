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

This module contains base classes for temperature `T`, pressure `P`, and
composition `zs` dependent properties. These power the various interfaces for
each property.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/chemicals/>`_.

.. contents:: :local:

Temperature Dependent
---------------------
.. autoclass:: TDependentProperty
   :members: name, units, extrapolation, property_min, property_max,
             critical_zero, ranked_methods, __call__, fit_polynomial,
             method, valid_methods, test_property_validity,
             T_dependent_property, plot_T_dependent_property, interpolate,
             add_method, add_tabular_data, solve_property,
             calculate_derivative, T_dependent_property_derivative,
             calculate_integral, T_dependent_property_integral,
             calculate_integral_over_T, T_dependent_property_integral_over_T,
             extrapolate, test_method_validity, calculate, from_json, as_json,
             interpolation_T, interpolation_T_inv, interpolation_property,
             interpolation_property_inv, T_limits, all_methods, __repr__,
             add_correlation
   :undoc-members:

Temperature and Pressure Dependent
----------------------------------
.. autoclass:: TPDependentProperty
   :members: name, units, extrapolation, property_min, property_max,
             ranked_methods, __call__,
             method, valid_methods, test_property_validity,
             add_method, add_tabular_data, solve_property,
             extrapolate, test_method_validity, calculate,
             interpolation_T, interpolation_T_inv, interpolation_property,
             interpolation_property_inv, T_limits, all_methods, all_methods_P,
             method_P, valid_methods_P, TP_dependent_property,
             TP_or_T_dependent_property, add_tabular_data_P, plot_isotherm,
             plot_isobar, plot_TP_dependent_property, calculate_derivative_T,
             calculate_derivative_P, TP_dependent_property_derivative_T,
             TP_dependent_property_derivative_P
   :undoc-members:
   :show-inheritance:

Temperature, Pressure, and Composition Dependent
------------------------------------------------
.. autoclass:: MixtureProperty
    :members:
    :undoc-members:
    :show-inheritance:

'''

from __future__ import division

__all__ = ['has_matplotlib', 'Stateva_Tsvetkov_TPDF', 'TPD',
'assert_component_balance', 'assert_energy_balance', 'allclose_variable',
'TDependentProperty','TPDependentProperty', 'MixtureProperty', 'identify_phase',
'phase_select_property', 'NEGLIGIBLE', 'DIPPR_PERRY_8E', 'POLY_FIT', 'VDI_TABULAR',
'VDI_PPDS', 'COOLPROP', 'LINEAR']

import os
from cmath import sqrt as csqrt
from fluids.numerics import quad, brenth, newton, secant, linspace, polyint, polyint_over_x, derivative, polyder, horner, horner_and_der2, quadratic_from_f_ders, assert_close, numpy as np
from fluids.constants import R
from chemicals.utils import PY37, isnan, isinf, log, exp, ws_to_zs, zs_to_ws, e
from chemicals.utils import mix_multiple_component_flows, hash_any_primitive
from chemicals.vapor_pressure import Antoine, Antoine_coeffs_from_point, Antoine_AB_coeffs_from_point, DIPPR101_ABC_coeffs_from_point
from chemicals.vapor_pressure import Wagner, Wagner_original, TRC_Antoine_extended, dAntoine_dT, d2Antoine_dT2, dWagner_original_dT, d2Wagner_original_dT2, dWagner_dT, d2Wagner_dT2, dTRC_Antoine_extended_dT, d2TRC_Antoine_extended_dT2
from chemicals.dippr import EQ100, EQ101, EQ102, EQ104, EQ105, EQ106, EQ107, EQ114, EQ115, EQ116, EQ127
from chemicals.phase_change import Watson, Watson_n, Alibakhshi, PPDS12
from chemicals.viscosity import (Viswanath_Natarajan_2, Viswanath_Natarajan_2_exponential,
                                 Viswanath_Natarajan_3, PPDS9, dPPDS9_dT)
from chemicals.heat_capacity import (Poling, Poling_integral, Poling_integral_over_T,
                                     TRCCp, TRCCp_integral, TRCCp_integral_over_T,
                                     Zabransky_quasi_polynomial, Zabransky_quasi_polynomial_integral, Zabransky_quasi_polynomial_integral_over_T,
                                     Zabransky_cubic, Zabransky_cubic_integral, Zabransky_cubic_integral_over_T)
from chemicals.interface import REFPROP_sigma, Somayajulu, Jasper
from chemicals.volume import volume_VDI_PPDS
from thermo import serialize
from thermo.eos import GCEOS
from thermo.eos_mix import GCEOSMIX
from thermo.coolprop import coolprop_fluids

NEGLIGIBLE = 'NEGLIGIBLE'
DIPPR_PERRY_8E = 'DIPPR_PERRY_8E'
POLY_FIT = 'POLY_FIT'
VDI_TABULAR = 'VDI_TABULAR'
VDI_PPDS = 'VDI_PPDS'
COOLPROP = 'COOLPROP'
'''CoolProp method'''
EOS = 'EOS'
LINEAR = 'LINEAR'

global _has_matplotlib
_has_matplotlib = None
def has_matplotlib():
    global _has_matplotlib
    if _has_matplotlib is None:
        try:
            import matplotlib.pyplot as plt
            _has_matplotlib = True
        except:
            _has_matplotlib = False
    return _has_matplotlib


def allclose_variable(a, b, limits, rtols=None, atols=None):
    """Returns True if two arrays are element-wise equal within several
    different tolerances. Tolerance values are always positive, usually very
    small. Based on numpy's allclose function.

    Only atols or rtols needs to be specified; both are used if given.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    limits : array_like
        Fractions of elements allowed to not match to within each tolerance.
    rtols : array_like
        The relative tolerance parameters.
    atols : float
        The absolute tolerance parameters.

    Returns
    -------
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerances; False otherwise.

    Examples
    --------
    10 random similar variables, all of them matching to within 1E-5, allowing
    up to half to match up to 1E-6.

    >>> x = [2.7244322249597719e-08, 3.0105683900110473e-10, 2.7244124924802327e-08, 3.0105259397637556e-10, 2.7243929226310193e-08, 3.0104990272770901e-10, 2.7243666849384451e-08, 3.0104101821236015e-10, 2.7243433745917367e-08, 3.0103707421519949e-10]
    >>> y = [2.7244328304561904e-08, 3.0105753470546008e-10, 2.724412872417824e-08,  3.0105303055834564e-10, 2.7243914341030203e-08, 3.0104819238021998e-10, 2.7243684057561379e-08, 3.0104299541023674e-10, 2.7243436694839306e-08, 3.010374130526363e-10]
    >>> allclose_variable(x, y, limits=[.0, .5], rtols=[1E-5, 1E-6])
    True
    """
    l = float(len(a))
    if rtols is None and atols is None:
        raise Exception('Either absolute errors or relative errors must be supplied.')
    elif rtols is None:
        rtols = [0 for i in atols]
    elif atols is None:
        atols = [0 for i in rtols]

    for atol, rtol, lim in zip(atols, rtols, limits):
        matches = np.count_nonzero(np.isclose(a, b, rtol=rtol, atol=atol))
        if 1-matches/l > lim:
            return False
    return True

def phase_select_property(phase=None, s=None, l=None, g=None, V_over_F=None,
                          self=None):
    r'''Determines which phase's property should be set as a default, given
    the phase a chemical is, and the property values of various phases. For the
    case of liquid-gas phase, returns None. If the property is not available
    for the current phase, or if the current phase is not known, returns None.

    Parameters
    ----------
    phase : str
        One of {'s', 'l', 'g', 'two-phase'}
    s : float
        Solid-phase property, [`prop`]
    l : float
        Liquid-phase property, [`prop`]
    g : float
        Gas-phase property, [`prop`]
    V_over_F : float
        Vapor phase fraction, [-]
    self : Object, optional
        If self is not None, the properties are assumed to be python properties
        with a fget method available, [-]

    Returns
    -------
    prop : float
        The selected/calculated property for the relevant phase, [`prop`]

    Notes
    -----
    Could calculate mole-fraction weighted properties for the two phase regime.
    Could also implement equilibria with solid phases.

    The use of self and fget ensures the properties not needed are not
    calculated.

    Examples
    --------
    >>> phase_select_property(phase='g', l=1560.14, g=3312.)
    3312.0
    '''
    if phase == 's':
        if self is not None and s is not None:
            return s.fget(self)
        return s
    elif phase == 'l':
        if self is not None and l is not None:
            return l.fget(self)
        return l
    elif phase == 'g':
        if self is not None and g is not None:
            return g.fget(self)
        return g
    elif phase is None or phase == 'two-phase':
        return None
    else:
        raise Exception('Property not recognized')

def identify_phase(T, P=101325.0, Tm=None, Tb=None, Tc=None, Psat=None):
    r'''Determines the phase of a one-species chemical system according to
    basic rules, using whatever information is available. Considers only the
    phases liquid, solid, and gas; does not consider two-phase
    scenarios, as should occurs between phase boundaries.

    * If the melting temperature is known and the temperature is under or equal
      to it, consider it a solid.
    * If the critical temperature is known and the temperature is greater or
      equal to it, consider it a gas.
    * If the vapor pressure at `T` is known and the pressure is under or equal
      to it, consider it a gas. If the pressure is greater than the vapor
      pressure, consider it a liquid.
    * If the melting temperature, critical temperature, and vapor pressure are
      not known, attempt to use the boiling point to provide phase information.
      If the pressure is between 90 kPa and 110 kPa (approximately normal),
      consider it a liquid if it is under the boiling temperature and a gas if
      above the boiling temperature.
    * If the pressure is above 110 kPa and the boiling temperature is known,
      consider it a liquid if the temperature is under the boiling temperature.
    * Return None otherwise.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    Tm : float, optional
        Normal melting temperature, [K]
    Tb : float, optional
        Normal boiling point, [K]
    Tc : float, optional
        Critical temperature, [K]
    Psat : float, optional
        Vapor pressure of the fluid at `T`, [Pa]

    Returns
    -------
    phase : str
        Either 's', 'l', 'g', or None if the phase cannot be determined

    Notes
    -----
    No special attential is paid to any phase transition. For the case where
    the melting point is not provided, the possibility of the fluid being solid
    is simply ignored.

    Examples
    --------
    >>> identify_phase(T=280, P=101325, Tm=273.15, Psat=991)
    'l'
    '''
    if Tm and T <= Tm:
        return 's'
    elif Tc and T >= Tc:
        # No special return value for the critical point
        return 'g'
    elif Psat:
        # Do not allow co-existence of phases; transition to 'l' directly under
        if P <= Psat:
            return 'g'
        elif P > Psat:
            return 'l'
    elif Tb:
        # Crude attempt to model phases without Psat
        # Treat Tb as holding from 90 kPa to 110 kPa
        if P is not None and (9E4 < P < 1.1E5):
            if T < Tb:
                return  'l'
            else:
                return 'g'
        elif P is not None and (P > 101325.0 and T <= Tb):
            # For the higher-pressure case, it is definitely liquid if under Tb
            # Above the normal boiling point, impossible to say - return None
            return 'l'
        else:
            return None
    else:
        return None




def d2ns_to_dn2_partials(d2ns, dns):
    '''from sympy import *
n1, n2 = symbols('n1, n2')
f, g, h = symbols('f, g, h', cls=Function)

diff(h(n1, n2)*f(n1,  n2), n1, n2)
    '''
    cmps = range(len(dns))
    hess = []
    for i in cmps:
        row = []
        for j in cmps:
            v = d2ns[i][j] + dns[i] + dns[j]
            row.append(v)
        hess.append(row)
    return hess


#def d2xs_to_d2ns(d2xs, dxs, dns):
#    # Could use some simplifying. Derived with trial and error via lots of inner loops
#    # Not working; must have just worked for the one thing for which the derivative was
#    # calculated in the test
#    # Should implement different dns and dxs for parts of equations
#    N = len(d2xs)
#    cmps = range(N)
#    hess = []
#    for i in cmps:
#        row = []
#        for j in cmps:
#            v = d2xs[i][j] -2*dns[i] - dns[j] - dxs[j]
#            row.append(v)
#        hess.append(row)
#    return hess

def TPD(T, zs, lnphis, ys, lnphis_test):
    r'''Function for calculating the Tangent Plane Distance function
    according to the original Michelsen definition. More advanced
    transformations of the TPD function are available in the literature for
    performing calculations.

    For a mixture to be stable, it is necessary and sufficient for this to
    be positive for all trial phase compositions.

    .. math::
        \text{TPD}(y) =  \sum_{j=1}^n y_j(\mu_j (y) - \mu_j(z))
        = RT \sum_i y_i\left(\ln(y_i) + \ln(\phi_i(y)) - d_i(z)\right)

        d_i(z) = \ln z_i + \ln \phi_i(z)

    Parameters
    ----------
    T : float
        Temperature of the system, [K]
    zs : list[float]
        Mole fractions of the phase undergoing stability
        testing (`test` phase), [-]
    lnphis : list[float]
        Log fugacity coefficients of the phase undergoing stability
        testing (if two roots are available, always use the lower Gibbs
        energy root), [-]
    ys : list[float]
        Mole fraction trial phase composition, [-]
    lnphis_test : list[float]
        Log fugacity coefficients of the trial phase (if two roots are
        available, always use the lower Gibbs energy root), [-]

    Returns
    -------
    TPD : float
        Original Tangent Plane Distance function, [J/mol]

    Notes
    -----
    A dimensionless version of this is often used as well, divided by
    RT.

    At the dew point (with test phase as the liquid and vapor incipient
    phase as the trial phase), TPD is zero [3]_.
    At the bubble point (with test phase as the vapor and liquid incipient
    phase as the trial phase), TPD is zero [3]_.

    Examples
    --------
    Solved bubble point for ethane/n-pentane 50-50 wt% at 1 MPa

    >>> from thermo.eos_mix import PRMIX
    >>> gas = PRMIX(Tcs=[305.32, 469.7], Pcs=[4872000.0, 3370000.0], omegas=[0.098, 0.251], kijs=[[0, 0.0078], [0.0078, 0]], zs=[0.9946656798618667, 0.005334320138133337], T=254.43857191839297, P=1000000.0)
    >>> liq = PRMIX(Tcs=[305.32, 469.7], Pcs=[4872000.0, 3370000.0], omegas=[0.098, 0.251], kijs=[[0, 0.0078], [0.0078, 0]], zs=[0.7058334393128614, 0.2941665606871387], T=254.43857191839297, P=1000000.0)
    >>> TPD(liq.T, liq.zs, liq.lnphis_l, gas.zs, gas.lnphis_g)
    -4.0339949303e-09

    References
    ----------
    .. [1] Michelsen, Michael L. "The Isothermal Flash Problem. Part I.
       Stability." Fluid Phase Equilibria 9, no. 1 (December 1982): 1-19.
    .. [2] Hoteit, Hussein, and Abbas Firoozabadi. "Simple Phase Stability
       -Testing Algorithm in the Reduction Method." AIChE Journal 52, no.
       8 (August 1, 2006): 2909-20.
    .. [3] Qiu, Lu, Yue Wang, Qi Jiao, Hu Wang, and Rolf D. Reitz.
       "Development of a Thermodynamically Consistent, Robust and Efficient
       Phase Equilibrium Solver and Its Validations." Fuel 115 (January 1,
       2014): 1-16. https://doi.org/10.1016/j.fuel.2013.06.039.
    '''
    tot = 0.0
    for yi, phi_yi, zi, phi_zi in zip(ys, lnphis_test, zs, lnphis):
        di = log(zi) + phi_zi
        tot += yi*(log(yi) + phi_yi - di)
    return tot*R*T

def Stateva_Tsvetkov_TPDF(lnphis, zs, lnphis_trial, ys):
    r'''Modified Tangent Plane Distance function according to [1]_ and
    [2]_. The stationary points of a system are all zeros of this function;
    so once all zeroes have been located, the stability can be evaluated
    at the stationary points only. It may be required to use multiple
    guesses to find all stationary points, and there is no method of
    confirming all points have been found.

    .. math::
        \phi(y) = \sum_i^{N} (k_{i+1}(y) - k_i(y))^2

        k_i(y) = \ln \phi_i(y) + \ln(y_i) - d_i

        k_{N+1}(y) = k_1(y)

        d_i(z) = \ln z_i + \ln \phi_i(z)

    Parameters
    ----------
    zs : list[float]
        Mole fractions of the phase undergoing stability
        testing (`test` phase), [-]
    lnphis : list[float]
        Log fugacity coefficients of the phase undergoing stability
        testing (if two roots are available, always use the lower Gibbs
        energy root), [-]
    ys : list[float]
        Mole fraction trial phase composition, [-]
    lnphis_test : list[float]
        Log fugacity coefficients of the trial phase (if two roots are
        available, always use the lower Gibbs energy root), [-]

    Returns
    -------
    TPDF_Stateva_Tsvetkov : float
        Modified Tangent Plane Distance function according to [1]_, [-]

    Notes
    -----
    In [1]_, a typo omitted the squaring of the expression. This method
    produces plots matching the shapes given in literature.

    References
    ----------
    .. [1] Ivanov, Boyan B., Anatolii A. Galushko, and Roumiana P. Stateva.
       "Phase Stability Analysis with Equations of State-A Fresh Look from
       a Different Perspective." Industrial & Engineering Chemistry
       Research 52, no. 32 (August 14, 2013): 11208-23.
    .. [2] Stateva, Roumiana P., and Stefan G. Tsvetkov. "A Diverse
       Approach for the Solution of the Isothermal Multiphase Flash
       Problem. Application to Vapor-Liquid-Liquid Systems." The Canadian
       Journal of Chemical Engineering 72, no. 4 (August 1, 1994): 722-34.
    '''
    kis = []
    for yi, log_phi_yi, zi, log_phi_zi in zip(ys, lnphis_trial, zs, lnphis):
        di = log_phi_zi + (log(zi) if zi > 0.0 else -690.0)
        try:
            ki = log_phi_yi + log(yi) - di
        except ValueError:
            # log - yi is negative; convenient to handle it to make the optimization take negative comps
            ki = log_phi_yi + -690.0 - di
        kis.append(ki)
    kis.append(kis[0])

    tot = 0.0
    for i in range(len(zs)):
        t = kis[i+1] - kis[i]
        tot += t*t
    return tot






def assert_component_balance(inlets, outlets, rtol=1E-9, atol=0, reactive=False):
    r'''Checks a mole balance for a group of inlet streams against outlet
    streams. Inlets and outlets must be Stream objects. The check is performed
    on a mole-basis; an exception is raised if the balance is not satisfied.

    Parameters
    ----------
    inlets : list[Stream] or Stream
        Inlet streams to be checked, [-]
    outlets : list[Stream] or Stream
        Outlet streams to be checked, [-]
    rtol : float, optional
        Relative tolerance, [-]
    atol : float, optional
        Absolute tolerance, [mol/s]
    reactive : bool, optional
        Whether or not to perform the check on a reactive basis (check mass,
        not moles, and element flows as well), [-]

    Notes
    -----
    No checks for zero flow are performed.

    Examples
    --------
    >>> from thermo.stream import Stream
    >>> f1 = Stream(['water', 'ethanol', 'pentane'], zs=[.5, .4, .1], T=300, P=1E6, n=50)
    >>> f2 = Stream(['water', 'methanol'], zs=[.5, .5], T=300, P=9E5, n=25)
    >>> f3 = Stream(IDs=['109-66-0', '64-17-5', '67-56-1', '7732-18-5'], ns=[5.0, 20.0, 12.5, 37.5], T=300, P=850000)
    >>> assert_component_balance([f1, f2], f3)
    >>> assert_component_balance([f1, f2], f3, reactive=True)
    '''
    try:
        [_ for _ in inlets]
    except TypeError:
        inlets = [inlets]
    try:
        [_ for _ in outlets]
    except TypeError:
        outlets = [outlets]

    feed_CASs = [i.CASs for i in inlets]
    product_CASs = [i.CASs for i in outlets]

    if reactive:
        # mass balance
        assert_close(sum([i.m for i in inlets]), sum([i.m for i in outlets]))

        try:
            ws = [i.ws for i in inlets]
        except:
            ws = [i.ws() for i in inlets]

        feed_cmps, feed_masses = mix_multiple_component_flows(IDs=feed_CASs,
                                                              flows=[i.m for i in inlets],
                                                              fractions=ws)
        feed_mass_flows = {i:j for i, j in zip(feed_cmps, feed_masses)}

        product_cmps, product_mols = mix_multiple_component_flows(IDs=product_CASs,
                                                                  flows=[i.n for i in outlets],
                                                                  fractions=[i.ns for i in outlets])
        product_mass_flows = {i:j for i, j in zip(product_cmps, product_mols)}

        # Mass flow of each component does not balance.
#        for CAS, flow in feed_mass_flows.items():
#            assert_allclose(flow, product_mass_flows[CAS], rtol=rtol, atol=atol)

        # Check the component set is right
        if set(feed_cmps) != set(product_cmps):
            raise Exception('Product and feeds have different components in them')

        # element balance
        feed_cmps, feed_element_flows = mix_multiple_component_flows(IDs=[list(i.atoms.keys()) for i in inlets],
                                                              flows=[i.n for i in inlets],
                                                              fractions=[list(i.atoms.values()) for i in inlets])
        feed_element_flows = {i:j for i, j in zip(feed_cmps, feed_element_flows)}


        product_cmps, product_element_flows = mix_multiple_component_flows(IDs=[list(i.atoms.keys()) for i in outlets],
                                                              flows=[i.n for i in outlets],
                                                              fractions=[list(i.atoms.values()) for i in outlets])
        product_element_flows = {i:j for i, j in zip(product_cmps, product_element_flows)}

        for ele, flow in feed_element_flows.items():
            assert_close(flow, product_element_flows[ele], rtol=rtol, atol=atol)

        if set(feed_cmps) != set(product_cmps):
            raise Exception('Product and feeds have different elements in them')
        return

    feed_ns = [i.n for i in inlets]
    feed_zs = [i.zs for i in inlets]

    product_ns = [i.n for i in outlets]
    product_zs = [i.zs for i in outlets]

    feed_cmps, feed_mols = mix_multiple_component_flows(IDs=feed_CASs, flows=feed_ns, fractions=feed_zs)
    feed_flows = {i:j for i, j in zip(feed_cmps, feed_mols)}

    product_cmps, product_mols = mix_multiple_component_flows(IDs=product_CASs, flows=product_ns, fractions=product_zs)
    product_flows = {i:j for i, j in zip(product_cmps, product_mols)}

    # Fail on unmatching
    if set(feed_cmps) != set(product_cmps):
        raise ValueError('Product and feeds have different components in them')
    for CAS, flow in feed_flows.items():
        assert_close(flow, product_flows[CAS], rtol=rtol, atol=atol)


def assert_energy_balance(inlets, outlets, energy_inlets, energy_outlets,
                          rtol=1E-9, atol=0.0, reactive=False):
    try:
        [_ for _ in inlets]
    except TypeError:
        inlets = [inlets]
    try:
        [_ for _ in outlets]
    except TypeError:
        outlets = [outlets]
    try:
        [_ for _ in energy_inlets]
    except TypeError:
        energy_inlets = [energy_inlets]

    try:
        [_ for _ in energy_outlets]
    except TypeError:
        energy_outlets = [energy_outlets]

    # Energy streams need to handle direction, not just magnitude
    energy_in = 0.0
    for feed in inlets:
        if not reactive:
            energy_in += feed.energy
        else:
            energy_in += feed.energy_reactive
    for feed in energy_inlets:
        energy_in += feed.Q

    energy_out = 0.0
    for product in outlets:
        if not reactive:
            energy_out += product.energy
        else:
            energy_out += product.energy_reactive
    for product in energy_outlets:
        energy_out += product.Q

    assert_close(energy_in, energy_out, rtol=rtol, atol=atol)

class TDependentProperty(object):
    '''Class for calculating temperature-dependent chemical properties.

    On creation, a :obj:`TDependentProperty` examines all the possible methods
    implemented for calculating the property, loads whichever coefficients it
    needs (unless `load_data` is set to False), examines its input parameters,
    and selects the method it prefers. This method will continue to be used for
    all calculations until the method is changed by setting a new method
    to the to :obj:`method` attribute.

    The default list of preferred method orderings is at :obj:`ranked_methods`
    for all properties; the order can be modified there in-place, and this
    will take effect on all new :obj:`TDependentProperty` instances created
    but NOT on existing instances.

    All methods have defined criteria for determining if they are valid before
    calculation, i.e. a minimum and maximum temperature for coefficients to be
    valid. For constant property values used due to lack of
    temperature-dependent data, a short range is normally specified as valid.

    It is not assumed that a specified method will succeed; for example many
    expressions are not mathematically valid past the critical point, and in
    some cases there is no easy way to determine the temperature where a
    property stops being reasonable.

    Accordingly, all properties calculated are checked
    by a sanity function :obj:`test_property_validity <TDependentProperty.test_property_validity>`,
    which has basic sanity checks. If the property is not reasonable, None is
    returned.

    This framework also supports tabular data, which is interpolated from if
    specified. Interpolation is cubic-spline based if 5 or more points are
    given, and linearly interpolated with if few points are given. A transform
    may be applied so that a property such as
    vapor pressure can be interpolated non-linearly. These are functions or
    lambda expressions which are set for the variables :obj:`interpolation_T`,
    :obj:`interpolation_property`, and :obj:`interpolation_property_inv`.

    In order to calculate properties outside of the range of their correlations,
    a number of extrapolation method are available. Extrapolation is used by
    default on some properties but not all.
    The extrapolation methods available are as follows:

        * 'constant' - returns the model values as calculated at the temperature limits
        * 'linear' - fits the model at its temperature limits to a linear model
        * 'interp1d' - SciPy's :obj:`interp1d <scipy.interpolate.interp1d>` is used to extrapolate
        * 'AntoineAB' - fits the model to :obj:`Antoine <chemicals.vapor_pressure.Antoine>`'s
          equation at the temperature limits using only the A and B coefficient
        * 'DIPPR101_ABC' - fits the model at its temperature limits to the
          :obj:`EQ101 <chemicals.dippr.EQ101>` equation
        * 'Watson' - fits the model to the Heat of Vaporization model
          :obj:`Watson <chemicals.phase_change.Watson>`

    It is possible to use different extrapolation methods for the
    low-temperature and the high-temperature region. Specify the extrapolation
    parameter with the '|' symbols between the two methods; the first method
    is used for low-temperature, and the second for the high-temperature.

    Attributes
    ----------
    name : str
        The name of the property being calculated, [-]
    units : str
        The units of the property, [-]
    method : str
        The method to be used for property calculations, [-]
    interpolation_T : callable or None
        A function or lambda expression to transform the temperatures of
        tabular data for interpolation; e.g. 'lambda self, T: 1./T'
    interpolation_T_inv : callable or None
        A function or lambda expression to invert the transform of temperatures
        of tabular data for interpolation; e.g. 'lambda self, x: self.Tc*(1 - x)'
    interpolation_property : callable or None
        A function or lambda expression to transform tabular property values
        prior to interpolation; e.g. 'lambda self, P: log(P)'
    interpolation_property_inv : callable or None
        A function or property expression to transform interpolated property
        values from the transform performed by :obj:`interpolation_property` back
        to their actual form, e.g.  'lambda self, P: exp(P)'
    Tmin : float
        Maximum temperature at which no method can calculate the property above;
        set based on rough rules for some methods. Used to solve for a
        particular property value, and as a default minimum for plotting. Often
        higher than where the property is theoretically higher, i.e. liquid
        density above the triple point, but this information may still be
        needed for liquid mixtures with elevated critical points.
    Tmax : float
        Minimum temperature at which no method can calculate the property under;
        set based on rough rules for some methods. Used to solve for a
        particular property value, and as a default minimum for plotting. Often
        lower than where the property is theoretically higher, i.e. liquid
        density beneath the triple point, but this information may still be
        needed for subcooled liquids or mixtures with depressed freezing points.
    property_min : float
        Lowest value expected for a property while still being valid;
        this is a criteria used by :obj:`test_method_validity`.
    property_max : float
        Highest value expected for a property while still being valid;
        this is a criteria used by :obj:`test_method_validity`.
    ranked_methods : list
        Constant list of ranked methods by default
    tabular_data : dict
        Stores all user-supplied property data for interpolation in format
        {name: (Ts, properties)}, [-]
    tabular_data_interpolators : dict
        Stores all interpolation objects, idexed by name and property
        transform methods with the format {(name, interpolation_T,
        interpolation_property, interpolation_property_inv):
        (extrapolator, spline)}, [-]
    all_methods : set
        Set of all methods available for a given CASRN and set of properties,
        [-]
    '''
    def __init_subclass__(cls):
        cls.__full_path__ = "%s.%s" %(cls.__module__, cls.__qualname__)
    
    # Dummy properties
    name = 'Property name'
    units = 'Property units'

    interpolation_T = None
    interpolation_T_inv = None
    interpolation_property = None
    interpolation_property_inv = None

    extrapolation = 'linear'
    extrapolations = []

    tabular_extrapolation_pts = 20
    '''The number of points to calculate at and use when doing a tabular
    extrapolation calculation.'''

    interp1d_extrapolate_kind = 'linear'
    '''The `kind` parameter for scipy's interp1d function,
    when it is used for extrapolation.'''

    P_dependent = False

    _method = None
    forced = False

    property_min = 0
    property_max = 1E4  # Arbitrary max
    T_cached = None
    use_poly_fit = False


    T_limits = {}
    '''Dictionary containing method: (Tmin, Tmax) pairs for all methods applicable
    to the chemical'''

    all_methods = set()
    '''Set of all methods loaded and ready to use for the chemical
    property.'''

    critical_zero = False
    '''Whether or not the property is declining and reaching zero at the
    critical point. This is used by numerical solvers.'''

#    Tmin = None
#    Tmax = None
    ranked_methods = []

    # For methods specified by a user
    local_methods = {}

    _fit_force_n = {}
    '''Dictionary containing method: fit_n, for use in methods which should
    only ever be fit to a specific `n` value'''

    _fit_max_n = {}
    '''Dictionary containing method: max_n, for use in methods which should
    only ever be fit to a `n` value equal to or less than `n`'''

    pure_references = ()
    pure_reference_types = ()

    obj_references = ()
    obj_references_types = ()

    _json_obj_by_CAS = ('CP_f',)

    correlation_models = {
        'Antoine': (['A', 'B', 'C'], ['base'], {'f': Antoine, 'f_der': dAntoine_dT, 'f_der2': d2Antoine_dT2}),
        'TRC_Antoine_extended': (['Tc', 'to', 'A', 'B', 'C', 'n', 'E', 'F'], [], {'f': TRC_Antoine_extended, 'f_der': dTRC_Antoine_extended_dT, 'f_der2': d2TRC_Antoine_extended_dT2}),
        'Wagner_original': (['Tc', 'Pc', 'a', 'b', 'c', 'd'], [], {'f': Wagner_original, 'f_der': dWagner_original_dT, 'f_der2': d2Wagner_original_dT2}),
        'Wagner': (['Tc', 'Pc', 'a', 'b', 'c', 'd'], [], {'f': Wagner, 'f_der': dWagner_dT, 'f_der2': d2Wagner_dT2}),

    'Alibakhshi': (['Tc', 'C'], [], {'f': Alibakhshi}),
    'PPDS12': (['Tc', 'A', 'B', 'C', 'D', 'E'], [], {'f': PPDS12}),
    'Watson': (['Hvap_ref', 'T_ref', 'Tc'], ['exponent'], {'f': Watson}),

    'Viswanath_Natarajan_2': (['A', 'B',], [], {'f': Viswanath_Natarajan_2}),
    'Viswanath_Natarajan_2_exponential': (['C', 'D',], [], {'f': Viswanath_Natarajan_2_exponential}),
    'Viswanath_Natarajan_3': (['A', 'B', 'C'], [], {'f': Viswanath_Natarajan_3}),
    'PPDS9': (['A', 'B', 'C', 'D', 'E'], [], {'f': PPDS9, 'f_der': dPPDS9_dT}),

    'Poling': (['a', 'b', 'c', 'd', 'e'], [], {'f': Poling, 'f_int': Poling_integral, 'f_int_over_T': Poling_integral_over_T}),
    'TRCCp': (['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'], [], {'f': TRCCp, 'f_int': TRCCp_integral, 'f_int_over_T': TRCCp_integral_over_T}),
    'Zabransky_quasi_polynomial': (['Tc', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'], [], {'f': Zabransky_quasi_polynomial, 'f_int': Zabransky_quasi_polynomial_integral, 'f_int_over_T': Zabransky_quasi_polynomial_integral_over_T}),
    'Zabransky_cubic': (['a1', 'a2', 'a3', 'a4'], [], {'f': Zabransky_cubic, 'f_int': Zabransky_cubic_integral, 'f_int_over_T': Zabransky_cubic_integral_over_T}),

    'REFPROP_sigma': (['Tc', 'sigma0', 'n0'], ['sigma1', 'n1', 'sigma2', 'n2'], {'f': REFPROP_sigma}),
    'Somayajulu': (['Tc', 'A', 'B', 'C'], [], {'f': Somayajulu}),
    'Jasper': (['a', 'b',], [], {'f': Jasper}),

    'volume_VDI_PPDS': (['Tc', 'rhoc', 'a', 'b', 'c', 'd', 'MW',], [], {'f': volume_VDI_PPDS}),

    'DIPPR100': ([],
      ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
      {'f': EQ100,
       'f_der': lambda T, **kwargs: EQ100(T, order=1, **kwargs),
       'f_int': lambda T, **kwargs: EQ100(T, order=-1, **kwargs),
       'f_int_over_T': lambda T, **kwargs: EQ100(T, order=-1j, **kwargs)}),
     'DIPPR101': (['A', 'B'],
      ['C', 'D', 'E'],
      {'f': EQ101,
       'f_der': lambda T, **kwargs: EQ101(T, order=1, **kwargs),
       'f_der2': lambda T, **kwargs: EQ101(T, order=2, **kwargs),
       'f_der3': lambda T, **kwargs: EQ101(T, order=3, **kwargs)}),
     'DIPPR102': (['A', 'B', 'C', 'D'],
      [],
      {'f': EQ102,
       'f_der': lambda T, **kwargs: EQ102(T, order=1, **kwargs),
       'f_int': lambda T, **kwargs: EQ102(T, order=-1, **kwargs),
       'f_int_over_T': lambda T, **kwargs: EQ102(T, order=-1j, **kwargs)}),
     'DIPPR104': (['A', 'B'],
      ['C', 'D', 'E'],
      {'f': EQ104,
       'f_der': lambda T, **kwargs: EQ104(T, order=1, **kwargs),
       'f_int': lambda T, **kwargs: EQ104(T, order=-1, **kwargs),
       'f_int_over_T': lambda T, **kwargs: EQ104(T, order=-1j, **kwargs)}),
     'DIPPR105': (['A', 'B', 'C', 'D'],
      [],
      {'f': EQ105,
       'f_der': lambda T, **kwargs: EQ105(T, order=1, **kwargs),
       'f_der2': lambda T, **kwargs: EQ105(T, order=2, **kwargs),
       'f_der3': lambda T, **kwargs: EQ105(T, order=3, **kwargs)}),
     'DIPPR106': (['Tc', 'A', 'B'],
      ['C', 'D', 'E'],
      {'f': EQ106,
       'f_der': lambda T, **kwargs: EQ106(T, order=1, **kwargs),
       'f_der2': lambda T, **kwargs: EQ106(T, order=2, **kwargs),
       'f_der3': lambda T, **kwargs: EQ106(T, order=3, **kwargs)}),
     'YawsSigma': (['Tc', 'A', 'B'],
      [],
      {'f': EQ106,
       'f_der': lambda T, **kwargs: EQ106(T, order=1, **kwargs),
       'f_der2': lambda T, **kwargs: EQ106(T, order=2, **kwargs),
       'f_der3': lambda T, **kwargs: EQ106(T, order=3, **kwargs)}),

     'DIPPR107': ([],
      ['A', 'B', 'C', 'D', 'E'],
      {'f': EQ107,
       'f_der': lambda T, **kwargs: EQ107(T, order=1, **kwargs),
       'f_int': lambda T, **kwargs: EQ107(T, order=-1, **kwargs),
       'f_int_over_T': lambda T, **kwargs: EQ107(T, order=-1j, **kwargs)}),
     'DIPPR114': (['Tc', 'A', 'B', 'C', 'D'],
      [],
      {'f': EQ114,
       'f_der': lambda T, **kwargs: EQ114(T, order=1, **kwargs),
       'f_int': lambda T, **kwargs: EQ114(T, order=-1, **kwargs),
       'f_int_over_T': lambda T, **kwargs: EQ114(T, order=-1j, **kwargs)}),
     'DIPPR115': (['A', 'B'],
      ['C', 'D', 'E'],
      {'f': EQ115,
       'f_der': lambda T, **kwargs: EQ115(T, order=1, **kwargs),
       'f_der2': lambda T, **kwargs: EQ115(T, order=2, **kwargs),
       'f_der3': lambda T, **kwargs: EQ115(T, order=3, **kwargs)}),
     'DIPPR116': (['A', 'B'],
      ['C', 'D', 'E'],
      {'f': EQ116,
       'f_der': lambda T, **kwargs: EQ116(T, order=1, **kwargs),
       'f_int': lambda T, **kwargs: EQ116(T, order=-1, **kwargs),
       'f_int_over_T': lambda T, **kwargs: EQ116(T, order=-1j, **kwargs)}),
     'DIPPR127': (['A', 'B', 'C', 'D', 'E', 'F', 'G'],
      [],
      {'f': EQ127,
       'f_der': lambda T, **kwargs: EQ127(T, order=1, **kwargs),
       'f_int': lambda T, **kwargs: EQ127(T, order=-1, **kwargs),
       'f_int_over_T': lambda T, **kwargs: EQ127(T, order=-1j, **kwargs)}),



    }

    available_correlations = frozenset(correlation_models.keys())

    correlation_parameters = {k: k + '_parameters' for k in correlation_models.keys()}



    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # By default, share state among subsequent objects
        return self

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    hash_ignore_props = ('linear_extrapolation_coeffs', 'Antoine_AB_coeffs',
                         'DIPPR101_ABC_coeffs', 'Watson_coeffs',
                         'interp1d_extrapolators', 'prop_cached',
                         'TP_cached', 'tabular_data_interpolators',
                         'tabular_data_interpolators_P', 'T_cached')
    def __hash__(self):
        d = self.__dict__
        # extrapolation values and interpolation objects should be ignored
        temp_store = {}
        for k in self.hash_ignore_props:
            try:
                temp_store[k] = d[k]
                del d[k]
            except:
                pass
        ans = hash_any_primitive((self.__class__, d))
        d.update(temp_store)

        return ans

    def __repr__(self):
        r'''Create and return a string representation of the object. The design
        of the return string is such that it can be :obj:`eval`'d into itself.
        This is very convinient for creating tests. Note that several methods
        are not compatible with the :obj:`eval`'ing principle.

        Returns
        -------
        repr : str
            String representation, [-]
        '''
        clsname = self.__class__.__name__
        base = '%s(' % (clsname)
        if self.CASRN:
            base += 'CASRN="%s", ' %(self.CASRN)
        for k in self.custom_args:
            v = getattr(self, k)
            if v is not None:
                base += '%s=%s, ' %(k, v)

        extrap_str = '"%s"' %(self.extrapolation) if self.extrapolation is not None else 'None'
        base += 'extrapolation=%s, ' %(extrap_str)

        method_str = '"%s"' %(self.method) if self.method is not None else 'None'
        base += 'method=%s, ' %(method_str)
        if self.tabular_data:
            if not (len(self.tabular_data) == 1 and VDI_TABULAR in self.tabular_data):
                base += 'tabular_data=%s, ' %(self.tabular_data)

        if self.P_dependent:
            method_P_str = '"%s"' %(self.method_P) if self.method_P is not None else 'None'
            base += 'method_P=%s, ' %(method_P_str)
            if self.tabular_data_P:
                base += 'tabular_data_P=%s, ' %(self.tabular_data_P)
            if 'tabular_extrapolation_permitted' in self.__dict__:
                base += 'tabular_extrapolation_permitted=%s, ' %(self.tabular_extrapolation_permitted)


        if hasattr(self, 'poly_fit_Tmin') and self.poly_fit_Tmin is not None:
            base += 'poly_fit=(%s, %s, %s), ' %(self.poly_fit_Tmin, self.poly_fit_Tmax, self.poly_fit_coeffs)
        for k in self.correlation_parameters.values():
            extra_model = getattr(self, k, None)
            if extra_model:
                base += '%s=%s, ' %(k, extra_model)


        if base[-2:] == ', ':
            base = base[:-2]
        return base + ')'

    def __call__(self, T):
        r'''Convenience method to calculate the property; calls
        :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`. Caches previously calculated value,
        which is an overhead when calculating many different values of
        a property. See :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>` for more details as to the
        calculation procedure.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        if T == self.T_cached:
            return self.prop_cached
        else:
            self.prop_cached = self.T_dependent_property(T)
            self.T_cached = T
            return self.prop_cached

    def as_json(self, references=1):
        r'''Method to create a JSON serialization of the property model
        which can be stored, and reloaded later.

        Parameters
        ----------
        references : int
            How to handle references to other objects; internal parameter, [-]

        Returns
        -------
        json_repr : dict
            JSON-friendly representation, [-]

        Notes
        -----

        Examples
        --------
        '''
        # vaguely jsonpickle compatible
        d = self.__dict__.copy()
        d["py/object"] = self.__full_path__
#        print(self.__full_path__)
        d["json_version"] = 1

        d['all_methods'] = list(d['all_methods'])
        d['tabular_data_interpolators'] = {}


        try:
            del d['interp1d_extrapolators']
        except:
            pass

        try:
            del d['correlations']
        except:
            pass

        if hasattr(self, 'all_methods_P'):
            d['all_methods_P'] = list(d['all_methods_P'])
            d['tabular_data_interpolators_P'] = {}

        del_objs = references == 0
        for name in self.pure_references:
            prop_obj = getattr(self, name)
            if prop_obj is not None and type(prop_obj) not in (float, int):
                if del_objs:
                    del d[name]
                else:
                    d[name] = prop_obj.as_json()

        for name in self._json_obj_by_CAS:
            CASRN = self.CASRN
            if hasattr(self, name):
                d[name] = CASRN
        try:
            eos = getattr(self, 'eos')
            if eos:
                d['eos'] = eos[0].as_json()
        except:
            pass
        return d

    @classmethod
    def _load_json_CAS_references(cls, d):
        try:
            d['CP_f'] = coolprop_fluids[d['CP_f']]
        except:
            pass

    @classmethod
    def from_json(cls, json_repr):
        r'''Method to create a property model from a JSON
        serialization of another property model.

        Parameters
        ----------
        json_repr : dict
            JSON-friendly representation, [-]

        Returns
        -------
        model : :obj:`TDependentProperty` or :obj:`TPDependentProperty`
            Newly created object from the json serialization, [-]

        Notes
        -----
        It is important that the input string be in the same format as that
        created by :obj:`TDependentProperty.as_json`.

        Examples
        --------
        '''
        d = json_repr#serialize.json.loads(json_repr)
        cls._load_json_CAS_references(d)
        try:
            eos = d['eos']
            if eos is not None:
                d['eos'] = [GCEOS.from_json(eos)]
        except:
            pass

        d['all_methods'] = set(d['all_methods'])
        try:
            d['all_methods_P'] = set(d['all_methods_P'])
        except:
            pass
        d['T_limits'] = {k: tuple(v) for k, v in d['T_limits'].items()}
        d['tabular_data'] = {k: tuple(v) for k, v in d['tabular_data'].items()}

        for k, sub_cls in zip(cls.pure_references, cls.pure_reference_types):
            if k in d:
                if type(d[k]) is dict:
                    sub_json = d[k]
                    d[k] = sub_cls.from_json(sub_json)

        d['correlations'] = correlations = {}
        for correlation_name in cls.correlation_models.keys():
            # Should be lazy created?
            correlation_key = cls.correlation_parameters[correlation_name]
            if correlation_key in d:
                call = cls.correlation_models[correlation_name][2]['f']
                for model_name, kwargs in d[correlation_key].items():
                    model_kwargs = kwargs.copy()
                    model_kwargs.pop('Tmin')
                    model_kwargs.pop('Tmax')
                    correlations[model_name] = (call, model_kwargs, correlation_name)

        del d['py/object']
        del d["json_version"]
        new = cls.__new__(cls)
        new.__dict__ = d
        return new

    @classmethod
    def _fit_export_polynomials(cls, method=None, start_n=3, max_n=30,
                                eval_pts=100, save=False):
        import json
        dat = {}
        folder = os.path.join(source_path, cls.name)

        sources = cls._method_indexes()

        fit_max_n = cls._fit_max_n

        if method is None:
            methods = list(sources.keys())
            indexes = list(sources.values())
        else:
            methods = [method]
            indexes = [sources[method]]
        for method, index in zip(methods, indexes):
            method_dat = {}
            n = cls._fit_force_n.get(method, None)
            max_n_method = fit_max_n[method] if method in fit_max_n else max_n
            for CAS in index:
                print(CAS)
                obj = cls(CASRN=CAS)
                coeffs, (low, high), stats = obj.fit_polynomial(method, n=n, start_n=start_n, max_n=max_n_method, eval_pts=eval_pts)
                max_error = max(abs(1.0 - stats[2]), abs(1.0 - stats[3]))
                method_dat[CAS] = {'Tmax': high, 'Tmin': low, 'error_average': stats[0],
                   'error_std': stats[1], 'max_error': max_error , 'method': method,
                   'coefficients': coeffs}

            if save:
                f = open(os.path.join(folder, method + '_polyfits.json'), 'w')
                out_str = json.dumps(method_dat, sort_keys=True, indent=4, separators=(', ', ': '))
                f.write(out_str)
                f.close()
                dat[method] = method_dat

        return dat

    def fit_polynomial(self, method, n=None, start_n=3, max_n=30, eval_pts=100):
        r'''Method to fit a T-dependent property to a polynomial. The degree
        of the polynomial can be specified with the `n` parameter, or it will
        be automatically selected for maximum accuracy.

        Parameters
        ----------
        method : str
            Method name to fit, [-]
        n : int, optional
            The degree of the polynomial, if specified
        start_n : int
            If `n` is not specified, all polynomials of degree `start_n` to
            `max_n` will be tried and the highest-accuracy will be selected;
            [-]
        max_n : int
            If `n` is not specified, all polynomials of degree `start_n` to
            `max_n` will be tried and the highest-accuracy will be selected;
            [-]
        eval_pts : int
            The number of points to evaluate the fitted functions at to check
            for accuracy; more is better but slower, [-]

        Returns
        -------
        coeffs : list[float]
            Fit coefficients, [-]
        Tmin : float
            The minimum temperature used for the fitting, [K]
        Tmax : float
            The maximum temperature used for the fitting, [K]
        err_avg : float
            Mean error in the evaluated points, [-]
        err_std : float
            Standard deviation of errors in the evaluated points, [-]
        min_ratio : float
            Lowest ratio of calc/actual in any found points, [-]
        max_ratio : float
            Highest ratio of calc/actual in any found points, [-]
        '''
        # Ready to be documented
        from thermo.fitting import fit_cheb_poly, poly_fit_statistics, fit_cheb_poly_auto
        interpolation_property = self.interpolation_property
        interpolation_property_inv = self.interpolation_property_inv

        try:
            low, high = self.T_limits[method]
        except KeyError:
            raise ValueError("Unknown method")

        func = lambda T: self.calculate(T, method)

        if n is None:
            n, coeffs, stats = fit_cheb_poly_auto(func, low=low, high=high,
                      interpolation_property=interpolation_property,
                      interpolation_property_inv=interpolation_property_inv,
                      start_n=start_n, max_n=max_n, eval_pts=eval_pts)
        else:

            coeffs = fit_cheb_poly(func, low=low, high=high, n=n,
                          interpolation_property=interpolation_property,
                          interpolation_property_inv=interpolation_property_inv)

            stats = poly_fit_statistics(func, coeffs=coeffs, low=low, high=high, pts=eval_pts,
                          interpolation_property_inv=interpolation_property_inv)

        return coeffs, (low, high), stats

    @property
    def method(self):
        r'''Method used to set a specific property method or to obtain the name
        of the method in use.

        When setting a method, an exception is raised if the method specified
        isnt't available for the chemical with the provided information.

        If `method` is None, no calculations can be performed.

        Parameters
        ----------
        method : str
            Method to use, [-]
        '''
        return self._method

    @method.setter
    def method(self, method):
        if method not in self.all_methods and method != POLY_FIT and method is not None:
            raise ValueError("The given methods is not available for this chemical")
        self.T_cached = None
        self._method = method
        self.extrapolation = self.extrapolation

    def valid_methods(self, T=None):
        r'''Method to obtain a sorted list of methods that have data
        available to be used. The methods are ranked in the following order:

        * The currently selected method is first (if one is selected)
        * Other available methods are ranked by the attribute :obj:`ranked_methods`

        If `T` is provided, the methods will be checked against the temperature
        limits of the correlations as well.

        Parameters
        ----------
        T : float or None
            Temperature at which to test methods, [K]

        Returns
        -------
        sorted_valid_methods : list
            Sorted lists of methods valid at T according to
            :obj:`test_method_validity`, [-]
        '''
        considered_methods = list(self.all_methods)

        if self.method is not None:
            considered_methods.remove(self.method)

        # Index the rest of the methods by ranked_methods, and add them to a list, sorted_methods
        preferences = sorted([self.ranked_methods.index(i) for i in considered_methods])
        sorted_methods = [self.ranked_methods[i] for i in preferences]

        # Add back the user's methods to the top, in order.
        if self.method is not None:
            sorted_methods.insert(0, self.method)

        if T is not None:
            sorted_valid_methods = []
            for method in sorted_methods:
                if self.test_method_validity(T, method):
                    sorted_valid_methods.append(method)

            return sorted_valid_methods
        else:
            return sorted_methods

    @classmethod
    def test_property_validity(self, prop):
        r'''Method to test the validity of a calculated property. Normally,
        this method is used by a given property class, and has maximum and
        minimum limits controlled by the variables :obj:`property_min` and
        :obj:`property_max`.

        Parameters
        ----------
        prop : float
            property to be tested, [`units`]

        Returns
        -------
        validity : bool
            Whether or not a specifid method is valid
        '''
        if isinstance(prop, complex):
            return False
        elif prop < self.property_min:
            return False
        elif prop > self.property_max:
            return False
        return True

    def _custom_set_poly_fit(self):
        pass

    def _set_poly_fit(self, poly_fit, set_limits=False):
        if (poly_fit is not None and len(poly_fit) and (poly_fit[0] is not None
           and poly_fit[1] is not None and  poly_fit[2] is not None)
            and not isnan(poly_fit[0]) and not isnan(poly_fit[1])):
            self.use_poly_fit = True
            self.poly_fit_Tmin = Tmin = poly_fit[0]
            self.poly_fit_Tmax = Tmax = poly_fit[1]
            self.poly_fit_coeffs = poly_fit_coeffs = poly_fit[2]
            self.T_limits[POLY_FIT] = (Tmin, Tmax)
            self.method = POLY_FIT

            self.poly_fit_int_coeffs = polyint(poly_fit_coeffs)
            self.poly_fit_T_int_T_coeffs, self.poly_fit_log_coeff = polyint_over_x(poly_fit_coeffs)

            poly_fit_d_coeffs = polyder(poly_fit_coeffs[::-1])
            self.poly_fit_d2_coeffs = polyder(poly_fit_d_coeffs)
            self.poly_fit_d2_coeffs.reverse()
            self.poly_fit_d_coeffs = poly_fit_d_coeffs
            poly_fit_d_coeffs.reverse()

            # Extrapolation slope on high and low
            slope_delta_T = (self.poly_fit_Tmax - self.poly_fit_Tmin)*.05

            self.poly_fit_Tmax_value = self.calculate(self.poly_fit_Tmax, POLY_FIT)
            if self.interpolation_property is not None:
                self.poly_fit_Tmax_value = self.interpolation_property(self.poly_fit_Tmax_value)


            # Calculate the average derivative for the last 5% of the curve
#            fit_value_high = self.calculate(self.poly_fit_Tmax - slope_delta_T, POLY_FIT)
#            if self.interpolation_property is not None:
#                fit_value_high = self.interpolation_property(fit_value_high)

#            self.poly_fit_Tmax_slope = (self.poly_fit_Tmax_value
#                                        - fit_value_high)/slope_delta_T
            self.poly_fit_Tmax_slope = horner(self.poly_fit_d_coeffs, self.poly_fit_Tmax)
            self.poly_fit_Tmax_dT2 = horner(self.poly_fit_d2_coeffs, self.poly_fit_Tmax)


            # Extrapolation to lower T
            self.poly_fit_Tmin_value = self.calculate(self.poly_fit_Tmin, POLY_FIT)
            if self.interpolation_property is not None:
                self.poly_fit_Tmin_value = self.interpolation_property(self.poly_fit_Tmin_value)

#            fit_value_low = self.calculate(self.poly_fit_Tmin + slope_delta_T, POLY_FIT)
#            if self.interpolation_property is not None:
#                fit_value_low = self.interpolation_property(fit_value_low)
#            self.poly_fit_Tmin_slope = (fit_value_low
#                                        - self.poly_fit_Tmin_value)/slope_delta_T

            self.poly_fit_Tmin_slope = horner(self.poly_fit_d_coeffs, self.poly_fit_Tmin)
            self.poly_fit_Tmin_dT2 = horner(self.poly_fit_d2_coeffs, self.poly_fit_Tmin)

            self._custom_set_poly_fit()

            if set_limits:
                if self.Tmin is None:
                    self.Tmin = self.poly_fit_Tmin
                if self.Tmax is None:
                    self.Tmax = self.poly_fit_Tmax


    def as_poly_fit(self):
        return '%s(load_data=False, poly_fit=(%s, %s, %s))' %(self.__class__.__name__,
                  repr(self.poly_fit_Tmin), repr(self.poly_fit_Tmax),
                  repr(self.poly_fit_coeffs))


    def _base_calculate(self, T, method):
        if method in self.tabular_data:
            return self.interpolate(T, method)
        elif method in self.local_methods:
            return self.local_methods[method][0](T)
        elif method in self.correlations:
            call, kwargs, _ = self.correlations[method]
            return call(T, **kwargs)
        else:
            raise ValueError("Unknown method")

    def _base_calculate_P(self, T, P, method):
        if method in self.tabular_data_P:
            return self.interpolate_P(T, P, method)
#        elif method in self.local_methods_P:
#            return self.local_methods_P[method][0](T, P)
        else:
            raise ValueError("Unknown method")

    def _calculate_extrapolate(self, T, method):
        if self.use_poly_fit:
            try:
                return self.calculate(T, POLY_FIT)
            except Exception as e:
                pass

        if method is None:
            return None
        try:
            T_low, T_high = self.T_limits[method]
            in_range = T_low <= T <= T_high
        except (KeyError, AttributeError):
            in_range = self.test_method_validity(T, method)

        if in_range:
            try:
                prop = self.calculate(T, method)
            except:
                return None
            if self.test_property_validity(prop):
                return prop
        elif self._extrapolation is not None:
            try:
                return self.extrapolate(T, method)
            except:
                return None
        # Function returns None if it does not work.
        return None

    def _T_dependent_property(self, T, method):
        # Same as T_dependent_property but accepts method as an argument
        if method is None:
            return None
        try:
            T_low, T_high = self.T_limits[method]
            in_range = T_low <= T <= T_high
        except (KeyError, AttributeError):
            in_range = self.test_method_validity(T, method)

        if in_range:
            try:
                prop = self.calculate(T, method)
            except:
                return None
            if self.test_property_validity(prop):
                return prop
        elif self._extrapolation is not None:
            try:
                return self.extrapolate(T, method)
            except:
                return None
        # Function returns None if it does not work.
        return None

    def T_dependent_property(self, T):
        r'''Method to calculate the property with sanity checking and using
        the selected :obj:`method <thermo.utils.TDependentProperty.method>`.

        In the unlikely event the calculation of the property fails, None
        is returned.

        The calculated result is checked with
        :obj:`test_property_validity <thermo.utils.TDependentProperty.test_property_validity>`
        and None is returned if the calculated value is nonsensical.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        if self.use_poly_fit:
            try:
                return self.calculate(T, POLY_FIT)
            except Exception as e:
                pass

        method = self.method
        if method is None:
            return None
        try:
            T_low, T_high = self.T_limits[method]
            in_range = T_low <= T <= T_high
        except (KeyError, AttributeError):
            in_range = self.test_method_validity(T, method)

        if in_range:
            try:
                prop = self.calculate(T, method)
            except:
                return None
            if self.test_property_validity(prop):
                return prop
        elif self._extrapolation is not None:
            try:
                return self.extrapolate(T, method)
            except:
                return None
        # Function returns None if it does not work.
        return None

    def plot_T_dependent_property(self, Tmin=None, Tmax=None, methods=[],
                                  pts=250, only_valid=True, order=0, show=True,
                                  axes='semilogy'):  # pragma: no cover
        r'''Method to create a plot of the property vs temperature according to
        either a specified list of methods, or user methods (if set), or all
        methods. User-selectable number of points, and temperature range. If
        only_valid is set,:obj:`test_method_validity` will be used to check if each
        temperature in the specified range is valid, and
        :obj:`test_property_validity` will be used to test the answer, and the
        method is allowed to fail; only the valid points will be plotted.
        Otherwise, the result will be calculated and displayed as-is. This will
        not suceed if the method fails.

        Parameters
        ----------
        Tmin : float
            Minimum temperature, to begin calculating the property, [K]
        Tmax : float
            Maximum temperature, to stop calculating the property, [K]
        methods : list, optional
            List of methods to consider
        pts : int, optional
            A list of points to calculate the property at; if Tmin to Tmax
            covers a wide range of method validities, only a few points may end
            up calculated for a given method so this may need to be large
        only_valid : bool
            If True, only plot successful methods and calculated properties,
            and handle errors; if False, attempt calculation without any
            checking and use methods outside their bounds
        show : bool
            If True, displays the plot; otherwise, returns it
        '''
        # This function cannot be tested
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        if Tmin is None:
#            if methods:
#                try:
#                    T_limits = self.T_limits
#                    Tmin = min(T_limits[m][0] for m in methods)
#                except:
#                    Tmin = self.Tmin
            if self.Tmin is not None:
                Tmin = self.Tmin
            else:
                raise Exception('Minimum temperature could not be auto-detected; please provide it')
        if Tmax is None:
            if self.Tmax is not None:
                Tmax = self.Tmax
            else:
                raise Exception('Maximum temperature could not be auto-detected; please provide it')
        import matplotlib.pyplot as plt

        if not methods:
            methods = self.all_methods

#        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        NUM_COLORS = len(methods)
#        ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

        plot_fun = {'semilogy': plt.semilogy, 'semilogx': plt.semilogx, 'plot': plt.plot}[axes]
        Ts = linspace(Tmin, Tmax, pts)
        if order == 0:
            for method in methods:
                if only_valid:
                    properties, Ts2 = [], []
                    for T in Ts:
                        if self.test_method_validity(T, method):
                            try:
                                p = self._calculate_extrapolate(T=T, method=method)
                                if self.test_property_validity(p):
                                    properties.append(p)
                                    Ts2.append(T)
                            except:
                                pass
                    plot_fun(Ts2, properties, label=method)
                else:
                    properties = [self._calculate_extrapolate(T=T, method=method) for T in Ts]
                    plot_fun(Ts, properties, label=method)
            plt.ylabel(self.name + ', ' + self.units)
            title = self.name
            if self.CASRN:
                title += ' of ' + self.CASRN
            plt.title(title)
        elif order > 0:
            for method in methods:
                if only_valid:
                    properties, Ts2 = [], []
                    for T in Ts:
                        if self.test_method_validity(T, method):
                            try:
                                p = self.calculate_derivative(T=T, method=method, order=order)
                                properties.append(p)
                                Ts2.append(T)
                            except:
                                pass
                    plot_fun(Ts2, properties, label=method)
                else:
                    properties = [self.calculate_derivative(T=T, method=method, order=order) for T in Ts]
                    plot_fun(Ts, properties, label=method)
            plt.ylabel(self.name + ', ' + self.units + '/K^%d derivative of order %d' % (order, order))

            title = self.name + ' derivative of order %d' % order
            if self.CASRN:
                title += ' of ' + self.CASRN
            plt.title(title)
        plt.legend(loc='best', fancybox=True, framealpha=0.5)
        plt.xlabel('Temperature, K')
        if show:
            plt.show()
        else:
            return plt

    def interpolate(self, T, name):
        r'''Method to perform interpolation on a given tabular data set
        previously added via :obj:`add_tabular_data`. This method will create the
        interpolators the first time it is used on a property set, and store
        them for quick future use.

        Interpolation is cubic-spline based if 5 or more points are available,
        and linearly interpolated if not. Extrapolation is always performed
        linearly. This function uses the transforms :obj:`interpolation_T`,
        :obj:`interpolation_property`, and :obj:`interpolation_property_inv` if set. If
        any of these are changed after the interpolators were first created,
        new interpolators are created with the new transforms.
        All interpolation is performed via the `interp1d` function.

        Parameters
        ----------
        T : float
            Temperature at which to interpolate the property, [K]
        name : str
            The name assigned to the tabular data set

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        # Cannot use method as key - need its id; faster also
        key = (name, id(self.interpolation_T), id(self.interpolation_property), id(self.interpolation_property_inv))

        # If the interpolator and extrapolator has already been created, load it
#        if isinstance(self.tabular_data_interpolators, dict) and key in self.tabular_data_interpolators:
#            extrapolator, spline = self.tabular_data_interpolators[key]

        if key in self.tabular_data_interpolators:
            extrapolator, spline = self.tabular_data_interpolators[key]
        else:
            from scipy.interpolate import interp1d
            Ts, properties = self.tabular_data[name]

            if self.interpolation_T is not None:  # Transform ths Ts with interpolation_T if set
                Ts_interp = [self.interpolation_T(T) for T in Ts]
            else:
                Ts_interp = Ts
            if self.interpolation_property is not None:  # Transform ths props with interpolation_property if set
                properties_interp = [self.interpolation_property(p) for p in properties]
            else:
                properties_interp = properties
            # Only allow linear extrapolation, but with whatever transforms are specified
            extrapolator = interp1d(Ts_interp, properties_interp, fill_value='extrapolate')
            # If more than 5 property points, create a spline interpolation
            if len(properties) >= 5:
                spline = interp1d(Ts_interp, properties_interp, kind='cubic')
            else:
                spline = None
#            if isinstance(self.tabular_data_interpolators, dict):
#                self.tabular_data_interpolators[key] = (extrapolator, spline)
#            else:
#                self.tabular_data_interpolators = {key: (extrapolator, spline)}
            self.tabular_data_interpolators[key] = (extrapolator, spline)

        # Load the stores values, tor checking which interpolation strategy to
        # use.
        Ts, properties = self.tabular_data[name]

        if T < Ts[0] or T > Ts[-1] or not spline:
            tool = extrapolator
        else:
            tool = spline

        if self.interpolation_T:
            T = self.interpolation_T(T)
        prop = tool(T)  # either spline, or linear interpolation

        if self.interpolation_property:
            prop = self.interpolation_property_inv(prop)

        return float(prop)

    def add_correlation(self, name, model, Tmin, Tmax, **kwargs):
        r'''Method to add a new set of emperical fit equation coefficients to
        the object and select it for future property calculations.

        A number of hardcoded `model` names are implemented; other models
        are not supported.

        Parameters
        ----------
        name : str
            The name of the coefficient set; user specified, [-]
        model : str
            A string representing the supported models, [-]
        Tmin : float
            Minimum temperature to use the method at, [K]
        Tmax : float
            Maximum temperature to use the method at, [K]
        kwargs : dict
            Various keyword arguments accepted by the model, [-]

        Notes
        -----
        The correlation models and links to their functions, describing
        their parameters, are as follows:

        '''
        if model not in self.available_correlations:
            raise ValueError("Model is not available; available models are %s" %(self.available_correlations,))
        model_data = self.correlation_models[model]
        if not all(k in kwargs and kwargs[k] is not None for k in model_data[0]):
            raise ValueError("Required arguments for this model are %s" %(model_data[0],))

        model_kwargs = {k: kwargs[k] for k in model_data[0]}
        for param in model_data[1]:
            if param in kwargs:
                model_kwargs[param] = kwargs[param]

        d = getattr(self, model + '_parameters', None)
        if d is None:
            d = {}
            setattr(self, model + '_parameters', d)

        full_kwargs = model_kwargs.copy()
        full_kwargs['Tmax'] = Tmax
        full_kwargs['Tmin'] = Tmin
        d[name] = full_kwargs

        self.T_limits[name] = (Tmin, Tmax)
        self.all_methods.add(name)

        call = self.correlation_models[model][2]['f']
        self.correlations[name] = (call, model_kwargs, model)
        self.method = name

    _text = '\n'
    for correlation_name, _correlation_parameters in correlation_models.items():
        f = _correlation_parameters[2]['f']
        correlation_func_name = f.__name__
        correlation_func_mod = f.__module__
        s = '        * "%s": :obj:`%s <%s.%s>`, required parameters %s' %(correlation_name, correlation_func_name, correlation_func_mod, correlation_func_name, tuple(_correlation_parameters[0]))
        if _correlation_parameters[1]:
            s += ', optional parameters %s.\n' %(tuple(_correlation_parameters[1]),)
        else:
            s += '.\n'
        _text += s
    try:
        add_correlation.__doc__ += _text
    except:
        pass

    def add_method(self, f, name, Tmin, Tmax, f_der_general=None,
                   f_der=None, f_der2=None, f_der3=None, f_int=None,
                   f_int_over_T=None):
        r'''Method to add a new method and select it for future property
        calculations.

        Parameters
        ----------
        f : callable
            Object which calculates the property given the temperature in K,
            [-]
        name : str, optional
            Name assigned to the data
        Tmin : float
            Minimum temperature to use the method at, [K]
        Tmax : float
            Maximum temperature to use the method at, [K]
        f_der_general : callable or None
            If specified, should take as arguments (T, order) and return the
            derivative of the property at the specified temperature and
            order, [-]
        f_der : callable or None
            If specified, should take as an argument the temperature and
            return the first derivative of the property, [-]
        f_der2 : callable or None
            If specified, should take as an argument the temperature and
            return the second derivative of the property, [-]
        f_der3 : callable or None
            If specified, should take as an argument the temperature and
            return the third derivative of the property, [-]
        f_int : callable or None
            If specified, should take `T1` and `T2` and return the integral of
            the property from `T1` to `T2`, [-]
        f_int_over_T : callable or None
            If specified, should take `T1` and `T2` and return the integral of
            the property over T from `T1` to `T2`, [-]

        Notes
        -----
        Once a custom method has been added to an object, that object can no
        longer be serialized to json and the :obj:`TDependentProperty.__repr__`
        method can no longer be used to reconstruct the object completely.

        '''
        if not self.local_methods:
            self.local_methods = local_methods = {}
        local_methods[name] = (f, Tmin, Tmax, f_der_general, f_der, f_der2,
                      f_der3, f_int, f_int_over_T)
        self.T_limits[name] = (Tmin, Tmax)
        self.all_methods.add(name)
        self.method = name

    def add_tabular_data(self, Ts, properties, name=None, check_properties=True):
        r'''Method to set tabular data to be used for interpolation.
        Ts must be in increasing order. If no name is given, data will be
        assigned the name 'Tabular data series #x', where x is the number of
        previously added tabular data series.

        After adding the data, this method becomes the selected method.

        Parameters
        ----------
        Ts : array-like
            Increasing array of temperatures at which properties are specified, [K]
        properties : array-like
            List of properties at Ts, [`units`]
        name : str, optional
            Name assigned to the data
        check_properties : bool
            If True, the properties will be checked for validity with
            :obj:`test_property_validity` and raise an exception if any are not
            valid
        '''
        # Ts must be in increasing order.
        if check_properties:
            for p in properties:
                if not self.test_property_validity(p):
                    raise ValueError('One of the properties specified are not feasible')
        if not all(b > a for a, b in zip(Ts, Ts[1:])):
            raise ValueError('Temperatures are not sorted in increasing order')

        if name is None:
            name = 'Tabular data series #' + str(len(self.tabular_data))  # Will overwrite a poorly named series
        self.tabular_data[name] = (Ts, properties)
        self.T_limits[name] = (min(Ts), max(Ts))

        self.all_methods.add(name)
        self.method = name

    def solve_property(self, goal):
        r'''Method to solve for the temperature at which a property is at a
        specified value. :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>` is used to calculate the value
        of the property as a function of temperature.

        Checks the given property value with :obj:`test_property_validity` first
        and raises an exception if it is not valid.

        Parameters
        ----------
        goal : float
            Propoerty value desired, [`units`]

        Returns
        -------
        T : float
            Temperature at which the property is the specified value [K]
        '''
#        if self.Tmin is None or self.Tmax is None:
#            raise Exception('Both a minimum and a maximum value are not present indicating there is not enough data for temperature dependency.')
        if not self.test_property_validity(goal):
            raise ValueError('Input property is not considered plausible; no method would calculate it.')

        def error(T):
            err = self.T_dependent_property(T) - goal
            return err
        T_limits = self.T_limits[self.method]
        if self.extrapolation is None:
            try:
                return brenth(error, T_limits[0], T_limits[1])
            except ValueError:
                raise Exception('To within the implemented temperature range, it is not possible to calculate the desired value.')
        else:
            high = self.Tc if self.critical_zero and self.Tc is not None else None
            x0 = T_limits[0]
            x1 = T_limits[1]
            f0 = error(x0)
            f1 = error(x1)
            if f0*f1 > 0.0: # same sign, pick which side to start the search at based on which error is lower
                if abs(f0) < abs(f1):
                    # under Tmin
                    x1 = T_limits[0] - (T_limits[1] - T_limits[0])*1e-3
                    f1 = error(x1)
                else:
                    # above Tmax
                    x0 = T_limits[1] + (T_limits[1] - T_limits[0])*1e-3
                    if high is not None and x0 > high:
                        x0 = high*(1-1e-6)
                    f0 = error(x0)
            #try:
            return secant(error, x0=x0, x1=x1, f0=f0, f1=f1, low=1e-4, xtol=1e-12, bisection=True, high=high)
            #except:
            #    return secant(error, x0=x0, x1=x1, f0=f0, f1=f1, low=1e-4, xtol=1e-12, bisection=True, high=high, damping=.01)

    def _calculate_derivative_transformed(self, T, method, order=1):
        r'''Basic funtion which wraps calculate_derivative such that the output
        of the derivative is in the transformed basis.'''
        if self.interpolation_property is None and self.interpolation_T is None:
            return self.calculate_derivative(T, method, order=1)

        interpolation_T = self.interpolation_T
        if interpolation_T is None:
            interpolation_T = lambda T: T
        interpolation_property = self.interpolation_property
        if interpolation_property is None:
            interpolation_property = lambda x: x

        try:
            return derivative(lambda T_trans: (interpolation_property(self.calculate(interpolation_T(T_trans), method=method))),
                              interpolation_T(T), dx=interpolation_T(T)*1e-6, n=order, order=1+order*2)
        except:
            Tmin, Tmax = self.T_limits[method]
            Tmin_trans, Tmax_trans = interpolation_T(Tmin), interpolation_T(Tmax)
            lower_limit = min(Tmin_trans, Tmax_trans)
            upper_limit = max(Tmin_trans, Tmax_trans)

            return derivative(lambda T_trans: interpolation_property(self.calculate(interpolation_T(T_trans), method=method)),
                              interpolation_T(T),
                              dx=interpolation_T(T)*1e-6, n=order, order=1+order*2,
                              lower_limit=lower_limit, upper_limit=upper_limit)

    def calculate_derivative(self, T, method, order=1):
        r'''Method to calculate a derivative of a property with respect to
        temperature, of a given order  using a specified method. Uses SciPy's
        derivative function, with a delta of 1E-6 K and a number of points
        equal to 2*order + 1.

        This method can be overwritten by subclasses who may perfer to add
        analytical methods for some or all methods as this is much faster.

        If the calculation does not succeed, returns the actual error
        encountered.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        method : str
            Method for which to find the derivative
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        derivative : float
            Calculated derivative property, [`units/K^order`]
        '''
        if method in self.correlations:
            _, model_kwargs, model = self.correlations[method]
            calls = self.correlation_models[model][2]
            if order == 1 and 'f_der' in calls:
                return calls['f_der'](T, **model_kwargs)
            elif order == 2 and 'f_der2' in calls:
                return calls['f_der2'](T, **model_kwargs)
            elif order == 3 and 'f_der3' in calls:
                return calls['f_der3'](T, **model_kwargs)

        pts = 1 + order*2
        dx = T*1e-6
        args = [method]
        Tmin, Tmax = self.T_limits[method]
        if Tmin <= T <= Tmax:
            # Adjust to be just inside bounds
            return derivative(self.calculate, T, dx=dx, args=args, n=order, order=pts,
                              lower_limit=Tmin, upper_limit=Tmax)
        else:
            # Allow extrapolation
            return derivative(self._T_dependent_property, T, dx=dx, args=args, n=order, order=pts)
#

    def T_dependent_property_derivative(self, T, order=1):
        r'''Method to obtain a derivative of a property with respect to
        temperature, of a given order.

        Calls :obj:`calculate_derivative` internally to perform the actual
        calculation.

        .. math::
            \text{derivative} = \frac{d (\text{property})}{d T}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        derivative : float
            Calculated derivative property, [`units/K^order`]
        '''
        try:
            return self.calculate_derivative(T, self._method, order)
        except:
            pass
        return None

    def calculate_integral(self, T1, T2, method):
        r'''Method to calculate the integral of a property with respect to
        temperature, using a specified method. Uses SciPy's `quad` function
        to perform the integral, with no options.

        This method can be overwritten by subclasses who may perfer to add
        analytical methods for some or all methods as this is much faster.

        If the calculation does not succeed, returns the actual error
        encountered.

        Parameters
        ----------
        T1 : float
            Lower limit of integration, [K]
        T2 : float
            Upper limit of integration, [K]
        method : str
            Method for which to find the integral

        Returns
        -------
        integral : float
            Calculated integral of the property over the given range,
            [`units*K`]
        '''
        if method in self.correlations:
            _, model_kwargs, model = self.correlations[method]
            calls = self.correlation_models[model][2]
            if 'f_int' in calls:
                return calls['f_int'](T2, **model_kwargs) - calls['f_int'](T1, **model_kwargs)

        return float(quad(self.calculate, T1, T2, args=(method,))[0])

    def T_dependent_property_integral(self, T1, T2):
        r'''Method to calculate the integral of a property with respect to
        temperature, using the selected method.

        Calls :obj:`calculate_integral` internally to perform the actual
        calculation.

        .. math::
            \text{integral} = \int_{T_1}^{T_2} \text{property} \; dT

        Parameters
        ----------
        T1 : float
            Lower limit of integration, [K]
        T2 : float
            Upper limit of integration, [K]

        Returns
        -------
        integral : float
            Calculated integral of the property over the given range,
            [`units*K`]
        '''
        try:
            return self.calculate_integral(T1, T2, self._method)
        except:
            pass
        return None

    def calculate_integral_over_T(self, T1, T2, method):
        r'''Method to calculate the integral of a property over temperature
        with respect to temperature, using a specified method. Uses SciPy's
        `quad` function to perform the integral, with no options.

        This method can be overwritten by subclasses who may perfer to add
        analytical methods for some or all methods as this is much faster.

        If the calculation does not succeed, returns the actual error
        encountered.

        Parameters
        ----------
        T1 : float
            Lower limit of integration, [K]
        T2 : float
            Upper limit of integration, [K]
        method : str
            Method for which to find the integral

        Returns
        -------
        integral : float
            Calculated integral of the property over the given range,
            [`units`]
        '''
        if method in self.correlations:
            _, model_kwargs, model = self.correlations[method]
            calls = self.correlation_models[model][2]
            if 'f_int_over_T' in calls:
                return calls['f_int_over_T'](T2, **model_kwargs) - calls['f_int_over_T'](T1, **model_kwargs)


        return float(quad(lambda T: self.calculate(T, method)/T, T1, T2)[0])

    def T_dependent_property_integral_over_T(self, T1, T2):
        r'''Method to calculate the integral of a property over temperature
        with respect to temperature, using the selected method.

        Calls :obj:`calculate_integral_over_T` internally to perform the actual
        calculation.

        .. math::
            \text{integral} = \int_{T_1}^{T_2} \frac{\text{property}}{T} \; dT

        Parameters
        ----------
        T1 : float
            Lower limit of integration, [K]
        T2 : float
            Upper limit of integration, [K]

        Returns
        -------
        integral : float
            Calculated integral of the property over the given range,
            [`units`]
        '''
        try:
            return self.calculate_integral_over_T(T1, T2, self._method)
        except:
            pass
        return None

    def _load_extapolation_coeffs(self, method):
        T_limits = self.T_limits

        for extrapolation in self.extrapolations:
            if extrapolation == 'linear':
                try:
                    linear_extrapolation_coeffs = self.linear_extrapolation_coeffs
                except:
                    self.linear_extrapolation_coeffs = linear_extrapolation_coeffs = {}
                interpolation_T = self.interpolation_T
                interpolation_property = self.interpolation_property
                interpolation_property_inv = self.interpolation_property_inv

                if method in linear_extrapolation_coeffs:
                    continue
                Tmin, Tmax = T_limits[method]
                if interpolation_T is not None:
                    Tmin_trans, Tmax_trans = interpolation_T(Tmin), interpolation_T(Tmax)
                try:
                    v_low = self.calculate(T=Tmin, method=method)
                    if interpolation_property is not None:
                        v_low = interpolation_property(v_low)
                    d_low = self._calculate_derivative_transformed(T=Tmin, method=method, order=1)
                except:
                    v_low, d_low = None, None
                try:
                    v_high = self.calculate(T=Tmax, method=method)
                    if interpolation_property is not None:
                        v_high = interpolation_property(v_high)
                    d_high = self._calculate_derivative_transformed(T=Tmax, method=method, order=1)
                except:
                    v_high, d_high = None, None
                linear_extrapolation_coeffs[method] = [v_low, d_low, v_high, d_high]
            elif extrapolation == 'constant':
                try:
                    constant_extrapolation_coeffs = self.constant_extrapolation_coeffs
                except:
                    self.constant_extrapolation_coeffs = constant_extrapolation_coeffs = {}

                if method in constant_extrapolation_coeffs:
                    continue
                Tmin, Tmax = T_limits[method]
                try:
                    v_low = self.calculate(T=Tmin, method=method)
                except:
                    v_low = None
                try:
                    v_high = self.calculate(T=Tmax, method=method)
                except:
                    v_high = None
                constant_extrapolation_coeffs[method] = [v_low, v_high]
            elif extrapolation == 'AntoineAB':
                try:
                    Antoine_AB_coeffs = self.Antoine_AB_coeffs
                except:
                    self.Antoine_AB_coeffs = Antoine_AB_coeffs = {}
                if method in Antoine_AB_coeffs:
                    continue
                Tmin, Tmax = T_limits[method]
                try:
                    v_low = self.calculate(T=Tmin, method=method)
                    d_low = self.calculate_derivative(T=Tmin, method=method, order=1)
                    AB_low = Antoine_AB_coeffs_from_point(T=Tmin, Psat=v_low, dPsat_dT=d_low, base=e)
                except:
                    AB_low = None
                try:
                    v_high = self.calculate(T=Tmax, method=method)
                    d_high = self.calculate_derivative(T=Tmax, method=method, order=1)
                    AB_high = Antoine_AB_coeffs_from_point(T=Tmax, Psat=v_high, dPsat_dT=d_high, base=e)
                except:
                    AB_high = None
                Antoine_AB_coeffs[method] = [AB_low, AB_high]
            elif extrapolation == 'DIPPR101_ABC':
                try:
                    DIPPR101_ABC_coeffs = self.DIPPR101_ABC_coeffs
                except:
                    self.DIPPR101_ABC_coeffs = DIPPR101_ABC_coeffs = {}
                if method in DIPPR101_ABC_coeffs:
                    continue
                Tmin, Tmax = T_limits[method]
                try:
                    v_low = self.calculate(T=Tmin, method=method)
                    d0_low = self.calculate_derivative(T=Tmin, method=method, order=1)
                    d1_low = self.calculate_derivative(T=Tmin, method=method, order=2)
                    DIPPR101_ABC_low = DIPPR101_ABC_coeffs_from_point(Tmin, v_low, d0_low, d1_low)
                except:
                    DIPPR101_ABC_low = None
                try:
                    v_high = self.calculate(T=Tmax, method=method)
                    d0_high = self.calculate_derivative(T=Tmax, method=method, order=1)
                    d1_high = self.calculate_derivative(T=Tmax, method=method, order=2)
                    DIPPR101_ABC_high = DIPPR101_ABC_coeffs_from_point(Tmax, v_high, d0_high, d1_high)
                except:
                    DIPPR101_ABC_high = None
                DIPPR101_ABC_coeffs[method] = [DIPPR101_ABC_low, DIPPR101_ABC_high]
            elif extrapolation == 'Watson':
                try:
                    Watson_coeffs = self.Watson_coeffs
                except:
                    self.Watson_coeffs = Watson_coeffs = {}
                if method in Watson_coeffs:
                    continue
                Tmin, Tmax = T_limits[method]
                delta = (Tmax-Tmin)*1e-4
                try:
                    v0_low = self.calculate(T=Tmin, method=method)
                    v1_low = self.calculate(T=Tmin+delta, method=method)
                    n_low = Watson_n(Tmin, Tmin+delta, v0_low, v1_low, self.Tc)
                except:
                    v0_low, v1_low, n_low = None, None, None
                try:
                    v0_high = self.calculate(T=Tmax, method=method)
                    v1_high = self.calculate(T=Tmax-delta, method=method)
                    n_high = Watson_n(Tmax, Tmax-delta, v0_high, v1_high, self.Tc)
                except:
                    v0_high, v1_high, n_high = None, None, None
                Watson_coeffs[method] = [v0_low, n_low, v0_high, n_high]
            elif extrapolation == 'interp1d':
                from scipy.interpolate import interp1d
                try:
                    interp1d_extrapolators = self.interp1d_extrapolators
                except:
                    self.interp1d_extrapolators = interp1d_extrapolators = {}
                interpolation_T = self.interpolation_T
                interpolation_property = self.interpolation_property
                interpolation_property_inv = self.interpolation_property_inv

                if method in interp1d_extrapolators:
                    continue
                Tmin, Tmax = T_limits[method]
                if method in self.tabular_data:
                    Ts, properties = self.tabular_data[method]
                else:
                    Ts = linspace(Tmin, Tmax, self.tabular_extrapolation_pts)
                    properties = [self.calculate(T, method=method) for T in Ts]

                if interpolation_T is not None:  # Transform ths Ts with interpolation_T if set
                    Ts_interp = [interpolation_T(T) for T in Ts]
                else:
                    Ts_interp = Ts
                if interpolation_property is not None:  # Transform ths props with interpolation_property if set
                    properties_interp = [interpolation_property(p) for p in properties]
                else:
                    properties_interp = properties
                # Only allow linear extrapolation, but with whatever transforms are specified
                extrapolator = interp1d(Ts_interp, properties_interp, fill_value='extrapolate', kind=self.interp1d_extrapolate_kind)
                interp1d_extrapolators[method] = extrapolator

            elif extrapolation is None or extrapolation == 'None':
                pass
            else:
                raise ValueError("Could not recognize extrapolation setting")


    @property
    def extrapolation(self):
        '''The string setting of the current extrapolation settings.
        This can be set to a new value to change which extrapolation setting
        is used.
        '''
        return self._extrapolation

    @extrapolation.setter
    def extrapolation(self, extrapolation):
        self._extrapolation = extrapolation
        if extrapolation is None:
            self.extrapolation_split = False
            self._extrapolation_low = self._extrapolation_high = self.extrapolations = None
            return
        self.extrapolation_split = '|' in extrapolation

        if not self.extrapolation_split:
            extrapolations = [extrapolation]
            self._extrapolation_low = self._extrapolation_high = extrapolation
        else:
            extrapolations = extrapolation.split('|')
            if len(extrapolations) != 2:
                raise ValueError("Must have only two extrapolation methods")
            self._extrapolation_low, self._extrapolation_high = extrapolations
            if extrapolations[0] == extrapolations[1]:
                extrapolations.pop()
        self.extrapolations = extrapolations

        if self._method is None:
            return # not set yet
        self._load_extapolation_coeffs(self._method)



    def extrapolate(self, T, method, in_range='error'):
        r'''Method to perform extrapolation on a given method according to the
        :obj:`extrapolation` setting.

        Parameters
        ----------
        T : float
            Temperature at which to extrapolate the property, [K]
        method : str
            The method to use, [-]
        in_range : str
            How to handle inputs which are not outside the temperature limits;
            set to 'low' to use the low T extrapolation, 'high' to use the
            high T extrapolation, and 'error' or anything else to raise an
            error in those cases, [-]

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        T_limits = self.T_limits
        if T < 0.0:
            raise ValueError("Negative temperature")
        T_low, T_high = T_limits[method]
        if T <= T_low or in_range == 'low':
            low = True
            extrapolation = self._extrapolation_low
        elif T >= T_high or in_range == 'high':
            low = False
            extrapolation = self._extrapolation_high
        else:
            raise ValueError("Not outside normal range")

        if extrapolation == 'linear':
            try:
                linear_extrapolation_coeffs = self.linear_extrapolation_coeffs
            except:
                self._load_extapolation_coeffs(method)
                linear_extrapolation_coeffs = self.linear_extrapolation_coeffs
            v_low, d_low, v_high, d_high = linear_extrapolation_coeffs[method]

            interpolation_T = self.interpolation_T
            interpolation_property_inv = self.interpolation_property_inv
            if interpolation_T is not None:
                T_low, T_high = interpolation_T(T_low), interpolation_T(T_high)
                T = interpolation_T(T)

            if low:
                if v_low is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at minimum temperature")
                val = v_low + d_low*(T - T_low)
                if interpolation_property_inv is not None:
                    val = interpolation_property_inv(val)
                return val
            else:
                if v_high is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at maximum temperature")
                val = v_high + d_high*(T - T_high)
                if interpolation_property_inv is not None:
                    val = interpolation_property_inv(val)
                return val
        elif extrapolation == 'constant':
            try:
                constant_extrapolation_coeffs = self.constant_extrapolation_coeffs
            except:
                self._load_extapolation_coeffs(method)
                constant_extrapolation_coeffs = self.constant_extrapolation_coeffs
            v_low, v_high = constant_extrapolation_coeffs[method]
            if low:
                if v_low is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at minimum temperature")
                return v_low
            else:
                if v_high is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at maximum temperature")
                return v_high

        elif extrapolation == 'AntoineAB':
            T_low, T_high = T_limits[method]
            try:
                AB_low, AB_high = self.Antoine_AB_coeffs[method]
            except:
                self._load_extapolation_coeffs(method)
                AB_low, AB_high = self.Antoine_AB_coeffs[method]
            if low:
                if AB_low is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at minimum temperature")
                return Antoine(T, A=AB_low[0], B=AB_low[1], C=0.0, base=e)
            else:
                if AB_high is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at maximum temperature")
                return Antoine(T, A=AB_high[0], B=AB_high[1], C=0.0, base=e)
        elif extrapolation == 'DIPPR101_ABC':
            T_low, T_high = T_limits[method]
            DIPPR101_ABC_low, DIPPR101_ABC_high = self.DIPPR101_ABC_coeffs[method]
            if low:
                if DIPPR101_ABC_low is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at minimum temperature")
                return EQ101(T, DIPPR101_ABC_low[0], DIPPR101_ABC_low[1], DIPPR101_ABC_low[2], 0.0, 0.0)
            else:
                if DIPPR101_ABC_high is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at maximum temperature")
                return EQ101(T, DIPPR101_ABC_high[0], DIPPR101_ABC_high[1], DIPPR101_ABC_high[2], 0.0, 0.0)
        elif extrapolation == 'Watson':
            T_low, T_high = T_limits[method]
            v0_low, n_low, v0_high, n_high = self.Watson_coeffs[method]
            if low:
                if v0_low is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at minimum temperature")
                return Watson(T, Hvap_ref=v0_low, T_ref=T_low, Tc=self.Tc, exponent=n_low)
            else:
                if v0_high is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at maximum temperature")
                return Watson(T, Hvap_ref=v0_high, T_ref=T_high, Tc=self.Tc, exponent=n_high)
        elif extrapolation == 'interp1d':
            T_low, T_high = T_limits[method]
            try:
                extrapolator = self.interp1d_extrapolators[method]
            except:
                self._load_extapolation_coeffs(method)
                extrapolator = self.interp1d_extrapolators[method]

            interpolation_T = self.interpolation_T
            if interpolation_T is not None:
                T = interpolation_T(T)
            prop = extrapolator(T)
            if self.interpolation_property is not None:
                prop = self.interpolation_property_inv(prop)
        return float(prop)



    def __init__(self, extrapolation, **kwargs):
        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        property under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        property above.'''

        self.tabular_data = {}
        '''tabular_data, dict: Stored (Ts, properties) for any
        tabular data; indexed by provided or autogenerated name.'''
        self.tabular_data_interpolators = {}
        '''tabular_data_interpolators, dict: Stored (extrapolator,
        spline) tuples which are interp1d instances for each set of tabular
        data; indexed by tuple of (name, interpolation_T,
        interpolation_property, interpolation_property_inv) to ensure that
        if an interpolation transform is altered, the old interpolator which
        had been created is no longer used.'''

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`.'''
        self.all_methods = set()
        '''Set of all methods available for a given CASRN and properties;
        filled by :obj:`load_all_methods`.'''
        self.load_all_methods(kwargs.get('load_data', True))
        self.extrapolation = extrapolation

        if kwargs.get('tabular_data', None):
            for name, (Ts, properties) in kwargs['tabular_data'].items():
                self.add_tabular_data(Ts, properties, name=name, check_properties=False)


        self.correlations = {}
        for correlation_name in self.correlation_models.keys():
            # Should be lazy created?
            correlation_key = self.correlation_parameters[correlation_name]
#            setattr(self, correlation_key, {})

            if correlation_key in kwargs:
                for corr_i, corr_kwargs in kwargs[correlation_key].items():
                    self.add_correlation(name=corr_i, model=correlation_name,
                                         **corr_kwargs)


        poly_fit = kwargs.get('poly_fit', None)
        method =  kwargs.get('method', None)
        if poly_fit is not None:
            if self.__class__.__name__ == 'EnthalpyVaporization':
                self.poly_fit_Tc = poly_fit[2]
                self._set_poly_fit((poly_fit[0], poly_fit[1], poly_fit[3]))
            elif self.__class__.__name__ == 'VaporPressure':
                self._set_poly_fit(poly_fit)
                if self.Tmin is None and hasattr(self, 'poly_fit_Tmin'):
                    self.Tmin = self.poly_fit_Tmin*.01
                if self.Tmax is None and hasattr(self, 'poly_fit_Tmax'):
                    self.Tmax = self.poly_fit_Tmax*10.0
            elif self.__class__.__name__ == 'SublimationPressure':
                self._set_poly_fit(poly_fit)
                if self.Tmin is None and hasattr(self, 'poly_fit_Tmin'):
                    self.Tmin = self.poly_fit_Tmin*.001
                if self.Tmax is None and hasattr(self, 'poly_fit_Tmax'):
                    self.Tmax = self.poly_fit_Tmax*10.0
            else:
                self._set_poly_fit(poly_fit)
        elif method is not None:
            self.method = method
        else:
            methods = self.valid_methods(T=None)
            if methods:
                self.method = methods[0]

    def load_all_methods(self):
        pass

    def calculate(self, T, method):
        r'''Method to calculate a property with a specified method, with no
        validity checking or error handling. Demo function for testing only;
        must be implemented according to the methods available for each
        individual method. Include the interpolation call here.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        method : str
            Method name to use

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        if method in self.tabular_data:
            prop = self.interpolate(T, method)
        return prop

    def test_method_validity(self, T, method):
        r'''Method to test the validity of a specified method for a given
        temperature. Demo function for testing only;
        must be implemented according to the methods available for each
        individual method. Include the interpolation check here.

        Parameters
        ----------
        T : float
            Temperature at which to determine the validity of the method, [K]
        method : str
            Method name to use

        Returns
        -------
        validity : bool
            Whether or not a specifid method is valid
        '''
        validity = True
        if method in self.tabular_data:
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        else:
            raise ValueError('Method not valid')
        return validity


class TPDependentProperty(TDependentProperty):
    '''Class for calculating temperature and pressure dependent chemical
    properties.

    On creation, a :obj:`TPDependentProperty` examines all the possible methods
    implemented for calculating the property, loads whichever coefficients it
    needs (unless `load_data` is set to False), examines its input parameters,
    and selects the method it prefers. This method will continue to be used for
    all calculations until the method is changed by setting a new method
    to the to :obj:`method` attribute.

    Because many pressure dependent property methods are implemented as a
    low-pressure correlation and a high-pressure correlation, this class
    works essentially the same as :obj:`TDependentProperty` but with extra
    methods that accept pressure as a parameter.

    The object also selects the pressure-dependent method it prefers.
    This method will continue to be used for all pressure-dependent
    calculations until the pressure-dependent method is changed by setting a
    new method_P to the to :obj:`method_P` attribute.

    The default list of preferred pressure-dependent method orderings is at
    :obj:`ranked_methods_P`
    for all properties; the order can be modified there in-place, and this
    will take effect on all new :obj:`TPDependentProperty` instances created
    but NOT on existing instances.

    Tabular data can be provided as either temperature-dependent or
    pressure-dependent data. The same `extrapolation` settings as in
    :obj:`TDependentProperty` are implemented here for the low-pressure
    correlations.

    In addition to the methods and attributes shown here, all those from
    :obj:`TPDependentProperty` are also available.

    Attributes
    ----------
    method_P : str
        The method was which was last used successfully to calculate a property;
        set only after the first property calculation.
    method : str
        The method to be used for property calculations, [-]
    '''
    P_dependent = True
    interpolation_P = None
    TP_cached = (None, None)
    _method_P = None
    '''Previously specified `T` and `P` in the calculation.'''

    all_methods_P = set()
    '''Set of all pressure-dependent methods loaded and ready to use for the
    chemical property.'''

    def __init__(self, extrapolation, **kwargs):
        self.sorted_valid_methods_P = []
        '''sorted_valid_methods_P, list: Stored methods which were found valid
        at a specific temperature; set by :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`.'''
        self.user_methods_P = []
        '''user_methods_P, list: Stored methods which were specified by the user
        in a ranked order of preference; set by :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`.'''


        self.tabular_data_P = {}
        '''tabular_data_P, dict: Stored (Ts, Ps, properties) for any
        tabular data; indexed by provided or autogenerated name.'''
        self.tabular_data_interpolators_P = {}
        '''tabular_data_interpolators_P, dict: Stored (extrapolator,
        spline) tuples which are interp2d instances for each set of tabular
        data; indexed by tuple of (name, interpolation_T, interpolation_P,
        interpolation_property, interpolation_property_inv) to ensure that
        if an interpolation transform is altered, the old interpolator which
        had been created is no longer used.'''

        self.all_methods_P = set()
        '''Set of all high-pressure methods available for a given CASRN and
        properties; filled by :obj:`load_all_methods`.'''

        super(TPDependentProperty, self).__init__(extrapolation, **kwargs)


        self.tabular_extrapolation_permitted = kwargs.get('tabular_extrapolation_permitted', True)

        if kwargs.get('tabular_data_P', None):
            for name, (Ts, Ps, properties) in kwargs['tabular_data_P'].items():
                self.add_tabular_data_P(Ts, Ps, properties, name=name, check_properties=False)

        method_P = kwargs.get('method_P', None)
        if method_P is not None:
            self.method_P = method_P
        else:
            methods = self.valid_methods_P(T=None, P=None)
            if methods:
                self.method_P = methods[0]
            else:
                self.method_P = None


    @property
    def method_P(self):
        r'''Method used to set or get a specific property method.

        An exception is raised if the method specified isnt't available
        for the chemical with the provided information.

        Parameters
        ----------
        method : str or list
            Methods by name to be considered or preferred
        '''
        return self._method_P

    @method_P.setter
    def method_P(self, method_P):
        if method_P not in self.all_methods_P and method_P is not None:
            raise ValueError("The given methods is not available for this chemical")
        self.TP_cached = None
        self._method_P = method_P

    def __call__(self, T, P):
        r'''Convenience method to calculate the property; calls
        :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`. Caches previously calculated value,
        which is an overhead when calculating many different values of
        a property. See :obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>` for more details as to the
        calculation procedure.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        P : float
            Pressure at which to calculate the property, [Pa]

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        if (T, P) == self.TP_cached:
            return self.prop_cached
        else:
            if P is not None:
                self.prop_cached = self.TP_dependent_property(T, P)
            else:
                self.prop_cached = self.T_dependent_property(T)
            self.TP_cached = (T, P)
            return self.prop_cached

    def valid_methods_P(self, T=None, P=None):
        r'''Method to obtain a sorted list of high-pressure methods that have
        data available to be used. The methods are ranked in the following
        order:

        * The currently selected :obj:`method_P` is first (if one is selected)
        * Other available pressure-depenent methods are ranked by the attribute
          :obj:`ranked_methods_P`

        If `T` and `P` are provided, the methods will be checked against the
        temperature and pressure limits of the correlations as well.

        Parameters
        ----------
        T : float
            Temperature at which to test methods, [K]
        P : float
            Pressure at which to test methods, [Pa]

        Returns
        -------
        sorted_valid_methods_P : list
            Sorted lists of methods valid at T and P according to
            :obj:`test_method_validity_P`
        '''
        considered_methods = list(self.all_methods_P)

        if self._method_P is not None:
            considered_methods.remove(self._method_P)

        preferences = sorted([self.ranked_methods_P.index(i) for i in considered_methods])
        sorted_methods = [self.ranked_methods_P[i] for i in preferences]

        if self._method_P is not None:
            sorted_methods.insert(0, self._method_P)

        if T is not None and P is not None:
            sorted_valid_methods_P = []
            for method in sorted_methods:
                if self.test_method_validity_P(T, P, method):
                    sorted_valid_methods_P.append(method)
            return sorted_valid_methods_P
        else:
            return sorted_methods


    def TP_dependent_property(self, T, P):
        r'''Method to calculate the property given a temperature and pressure
        according to the selected :obj:`method_P` and :obj:`method`.
        The pressure-dependent method is always used and required to succeed.
        The result is checked with :obj:`test_property_validity`.

        If the method does not succeed, returns None.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        P : float
            Pressure at which to calculate the property, [Pa]

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        try:
            prop = self.calculate_P(T, P, self._method_P)
            if self.test_property_validity(prop):
                return prop
        except:  # pragma: no cover
            pass
        # Function returns None if it does not work.
        return None

    def TP_or_T_dependent_property(self, T, P):
        r'''Method to calculate the property given a temperature and pressure
        according to the selected :obj:`method_P` and :obj:`method`.
        The pressure-dependent method is always tried.
        The result is checked with :obj:`test_property_validity`.

        If the pressure-dependent method does not succeed, the low-pressure
        method is tried and its result is returned.

        .. warning::
            It can seem like a good idea to switch between a low-pressure and
            a high-pressure method if the high pressure method is not working,
            however it can cause discontinuities and prevent numerical methods
            from converging

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        P : float
            Pressure at which to calculate the property, [Pa]

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        if P is not None:
            prop = self.TP_dependent_property(T, P)
        if P is None or prop is None:
            prop = self.T_dependent_property(T)
        return prop


    def add_tabular_data_P(self, Ts, Ps, properties, name=None, check_properties=True):
        r'''Method to set tabular data to be used for interpolation.
        Ts and Psmust be in increasing order. If no name is given, data will be
        assigned the name 'Tabular data series #x', where x is the number of
        previously added tabular data series.

        After adding the data, this method becomes the selected high-pressure
        method.


        Parameters
        ----------
        Ts : array-like
            Increasing array of temperatures at which properties are specified, [K]
        Ps : array-like
            Increasing array of pressures at which properties are specified, [Pa]
        properties : array-like
            List of properties at `Ts` and `Ps`; the data should be indexed
            [P][T], [`units`]
        name : str, optional
            Name assigned to the data
        check_properties : bool
            If True, the properties will be checked for validity with
            :obj:`test_property_validity` and raise an exception if any are not
            valid
        '''
        # Ts must be in increasing order.
        if check_properties:
            for p in np.array(properties).ravel():
                if not self.test_property_validity(p):
                    raise ValueError('One of the properties specified are not feasible')
        if not all(b > a for a, b in zip(Ts, Ts[1:])):
            raise ValueError('Temperatures are not sorted in increasing order')
        if not all(b > a for a, b in zip(Ps, Ps[1:])):
            raise ValueError('Pressures are not sorted in increasing order')

        if name is None:
            name = 'Tabular data series #' + str(len(self.tabular_data))  # Will overwrite a poorly named series
        self.tabular_data_P[name] = [Ts, Ps, properties]
        self.all_methods_P.add(name)

        self.method_P = name

    def interpolate_P(self, T, P, name):
        r'''Method to perform interpolation on a given tabular data set
        previously added via :obj:`add_tabular_data_P`. This method will create the
        interpolators the first time it is used on a property set, and store
        them for quick future use.

        Interpolation is cubic-spline based if 5 or more points are available,
        and linearly interpolated if not. Extrapolation is always performed
        linearly. This function uses the transforms :obj:`interpolation_T`,
        :obj:`interpolation_P`,
        :obj:`interpolation_property`, and :obj:`interpolation_property_inv` if set. If
        any of these are changed after the interpolators were first created,
        new interpolators are created with the new transforms.
        All interpolation is performed via the `interp2d` function.

        Parameters
        ----------
        T : float
            Temperature at which to interpolate the property, [K]
        T : float
            Pressure at which to interpolate the property, [Pa]
        name : str
            The name assigned to the tabular data set

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        key = (name, self.interpolation_T, id(self.interpolation_P), id(self.interpolation_property), id(self.interpolation_property_inv))
        Ts, Ps, properties = self.tabular_data_P[name]
        if not self.tabular_extrapolation_permitted:
            if T < Ts[0] or T > Ts[-1] or P < Ps[0] or P > Ps[-1]:
                raise ValueError("Extrapolation not permitted and conditions outside of range")

        # If the interpolator and extrapolator has already been created, load it
        if key in self.tabular_data_interpolators_P:
            extrapolator, spline = self.tabular_data_interpolators_P[key]
        else:
            from scipy.interpolate import interp2d


            if self.interpolation_T:  # Transform ths Ts with interpolation_T if set
                Ts2 = [self.interpolation_T(T2) for T2 in Ts]
            else:
                Ts2 = Ts
            if self.interpolation_P:  # Transform ths Ts with interpolation_T if set
                Ps2 = [self.interpolation_P(P2) for P2 in Ps]
            else:
                Ps2 = Ps
            if self.interpolation_property:  # Transform ths props with interpolation_property if set
                properties2 = [[self.interpolation_property(p) for p in r] for r in properties]
            else:
                properties2 = properties
            # Only allow linear extrapolation, but with whatever transforms are specified
            extrapolator = interp2d(Ts2, Ps2, properties2)  # interpolation if fill value is missing
            # If more than 5 property points, create a spline interpolation
            if len(properties) >= 5:
                spline = interp2d(Ts2, Ps2, properties2, kind='cubic')
            else:
                spline = None
            self.tabular_data_interpolators_P[key] = (extrapolator, spline)

        # Load the stores values, tor checking which interpolation strategy to
        # use.
        Ts, Ps, properties = self.tabular_data_P[name]

        if T < Ts[0] or T > Ts[-1] or not spline or P < Ps[0] or P > Ps[-1]:
            tool = extrapolator
        else:
            tool = spline

        if self.interpolation_T:
            T = self.interpolation_T(T)
        if self.interpolation_P:
            P = self.interpolation_P(P)
        prop = tool(T, P)  # either spline, or linear interpolation

        if self.interpolation_property:
            prop = self.interpolation_property_inv(prop)

        return float(prop)

    def plot_isotherm(self, T, Pmin=None, Pmax=None, methods_P=[], pts=50,
                      only_valid=True, show=True):  # pragma: no cover
        r'''Method to create a plot of the property vs pressure at a specified
        temperature according to either a specified list of methods, or the
        user methods (if set), or all methods. User-selectable number of
        points, and pressure range. If only_valid is set,
        :obj:`test_method_validity_P` will be used to check if each condition in
        the specified range is valid, and :obj:`test_property_validity` will be used
        to test the answer, and the method is allowed to fail; only the valid
        points will be plotted. Otherwise, the result will be calculated and
        displayed as-is. This will not suceed if the method fails.

        Parameters
        ----------
        T : float
            Temperature at which to create the plot, [K]
        Pmin : float
            Minimum pressure, to begin calculating the property, [Pa]
        Pmax : float
            Maximum pressure, to stop calculating the property, [Pa]
        methods_P : list, optional
            List of methods to consider
        pts : int, optional
            A list of points to calculate the property at; if Pmin to Pmax
            covers a wide range of method validities, only a few points may end
            up calculated for a given method so this may need to be large
        only_valid : bool
            If True, only plot successful methods and calculated properties,
            and handle errors; if False, attempt calculation without any
            checking and use methods outside their bounds
        show : bool
            If True, displays the plot; otherwise, returns it
        '''
        # This function cannot be tested
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        import matplotlib.pyplot as plt
        if Pmin is None:
            if self.Pmin is not None:
                Pmin = self.Pmin
            else:
                raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Pmax is None:
            if self.Pmax is not None:
                Pmax = self.Pmax
            else:
                raise Exception('Maximum pressure could not be auto-detected; please provide it')
        fig = plt.figure()

        if not methods_P:
            methods_P = self.all_methods_P
        Ps = linspace(Pmin, Pmax, pts)
        for method_P in methods_P:
            if only_valid:
                properties, Ps2 = [], []
                for P in Ps:
                    if self.test_method_validity_P(T, P, method_P):
                        try:
                            p = self.calculate_P(T, P, method_P)
                            if self.test_property_validity(p):
                                properties.append(p)
                                Ps2.append(P)
                        except:
                            pass
                plt.plot(Ps2, properties, label=method_P)
            else:
                properties = [self.calculate_P(T, P, method_P) for P in Ps]
                plt.plot(Ps, properties, label=method_P)
        plt.legend(loc='best')
        plt.ylabel(self.name + ', ' + self.units)
        plt.xlabel('Pressure, Pa')
        plt.title(self.name + ' of ' + self.CASRN)
        if show:
            plt.show()
        else:
            return plt

    def plot_isobar(self, P, Tmin=None, Tmax=None, methods_P=[], pts=50,
                    only_valid=True, show=True):  # pragma: no cover
        r'''Method to create a plot of the property vs temperature at a
        specific pressure according to
        either a specified list of methods, or user methods (if set), or all
        methods. User-selectable number of points, and temperature range. If
        only_valid is set,:obj:`test_method_validity_P` will be used to check if
        each condition in the specified range is valid, and
        :obj:`test_property_validity` will be used to test the answer, and the
        method is allowed to fail; only the valid points will be plotted.
        Otherwise, the result will be calculated and displayed as-is. This will
        not suceed if the method fails.

        Parameters
        ----------
        P : float
            Pressure for the isobar, [Pa]
        Tmin : float
            Minimum temperature, to begin calculating the property, [K]
        Tmax : float
            Maximum temperature, to stop calculating the property, [K]
        methods_P : list, optional
            List of methods to consider
        pts : int, optional
            A list of points to calculate the property at; if Tmin to Tmax
            covers a wide range of method validities, only a few points may end
            up calculated for a given method so this may need to be large
        only_valid : bool
            If True, only plot successful methods and calculated properties,
            and handle errors; if False, attempt calculation without any
            checking and use methods outside their bounds
        '''
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        import matplotlib.pyplot as plt
        if Tmin is None:
            if self.Tmin is not None:
                Tmin = self.Tmin
            else:
                raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Tmax is None:
            if self.Tmax is not None:
                Tmax = self.Tmax
            else:
                raise Exception('Maximum pressure could not be auto-detected; please provide it')
        if hasattr(P, '__call__'):
            P_changes = True
            P_func = P
        else:
            P_changes = False
        if not methods_P:
            methods_P = self.all_methods_P
        Ts = linspace(Tmin, Tmax, pts)
        fig = plt.figure()
        for method_P in methods_P:
            if only_valid:
                properties, Ts2 = [], []
                for T in Ts:
                    if P_changes:
                        P = P_func(T)
                    if self.test_method_validity_P(T, P, method_P):
                        try:
                            p = self.calculate_P(T, P, method_P)
                            if self.test_property_validity(p):
                                properties.append(p)
                                Ts2.append(T)
                        except:
                            pass
                plt.plot(Ts2, properties, label=method_P)
            else:
                properties = []
                for T in Ts:
                    if P_changes:
                        P = P_func(T)
                properties.append(self.calculate_P(T, P, method_P))

                plt.plot(Ts, properties, label=method_P)
        plt.legend(loc='best')
        plt.ylabel(self.name + ', ' + self.units)
        plt.xlabel('Temperature, K')
        plt.title(self.name + ' of ' + self.CASRN)
        if show:
            plt.show()
        else:
            return plt


    def plot_TP_dependent_property(self, Tmin=None, Tmax=None, Pmin=None,
                                   Pmax=None,  methods_P=[], pts=15,
                                   only_valid=True):  # pragma: no cover
        r'''Method to create a plot of the property vs temperature and pressure
        according to either a specified list of methods, or user methods (if
        set), or all methods. User-selectable number of points for each
        variable. If only_valid is set,:obj:`test_method_validity_P` will be used to
        check if each condition in the specified range is valid, and
        :obj:`test_property_validity` will be used to test the answer, and the
        method is allowed to fail; only the valid points will be plotted.
        Otherwise, the result will be calculated and displayed as-is. This will
        not suceed if the any method fails for any point.

        Parameters
        ----------
        Tmin : float
            Minimum temperature, to begin calculating the property, [K]
        Tmax : float
            Maximum temperature, to stop calculating the property, [K]
        Pmin : float
            Minimum pressure, to begin calculating the property, [Pa]
        Pmax : float
            Maximum pressure, to stop calculating the property, [Pa]
        methods_P : list, optional
            List of methods to plot
        pts : int, optional
            A list of points to calculate the property at for both temperature
            and pressure; pts^2 points will be calculated.
        only_valid : bool
            If True, only plot successful methods and calculated properties,
            and handle errors; if False, attempt calculation without any
            checking and use methods outside their bounds
        '''
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib.ticker import FormatStrFormatter
        import numpy.ma as ma
        import matplotlib.pyplot as plt

        if Pmin is None:
            if self.Pmin is not None:
                Pmin = self.Pmin
            else:
                raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Pmax is None:
            if self.Pmax is not None:
                Pmax = self.Pmax
            else:
                raise Exception('Maximum pressure could not be auto-detected; please provide it')
        if Tmin is None:
            if self.Tmin is not None:
                Tmin = self.Tmin
            else:
                raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Tmax is None:
            if self.Tmax is not None:
                Tmax = self.Tmax
            else:
                raise Exception('Maximum pressure could not be auto-detected; please provide it')

        if not methods_P:
            methods_P = self.user_methods_P if self.user_methods_P else self.all_methods_P
        Ps = np.linspace(Pmin, Pmax, pts)
        Ts = np.linspace(Tmin, Tmax, pts)
        Ts_mesh, Ps_mesh = np.meshgrid(Ts, Ps)
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        handles = []
        for method_P in methods_P:
            if only_valid:
                properties = []
                for T in Ts:
                    T_props = []
                    for P in Ps:
                        if self.test_method_validity_P(T, P, method_P):
                            try:
                                p = self.calculate_P(T, P, method_P)
                                if self.test_property_validity(p):
                                    T_props.append(p)
                                else:
                                    T_props.append(None)
                            except:
                                T_props.append(None)
                        else:
                            T_props.append(None)
                    properties.append(T_props)
                properties = ma.masked_invalid(np.array(properties, dtype=np.float).T)
                handles.append(ax.plot_surface(Ts_mesh, Ps_mesh, properties, cstride=1, rstride=1, alpha=0.5))
            else:
                properties = np.array([[self.calculate_P(T, P, method_P) for T in Ts] for P in Ps])
                handles.append(ax.plot_surface(Ts_mesh, Ps_mesh, properties, cstride=1, rstride=1, alpha=0.5))

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.set_xlabel('Temperature, K')
        ax.set_ylabel('Pressure, Pa')
        ax.set_zlabel(self.name + ', ' + self.units)
        plt.title(self.name + ' of ' + self.CASRN)
        plt.show(block=False)
        # The below is a workaround for a matplotlib bug
        ax.legend(handles, methods_P)
        plt.show(block=False)


    def calculate_derivative_T(self, T, P, method, order=1):
        r'''Method to calculate a derivative of a temperature and pressure
        dependent property with respect to  temperature at constant pressure,
        of a given order using a specified  method. Uses SciPy's  derivative
        function, with a delta of 1E-6 K and a number of points equal to
        2*order + 1.

        This method can be overwritten by subclasses who may perfer to add
        analytical methods for some or all methods as this is much faster.

        If the calculation does not succeed, returns the actual error
        encountered.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        P : float
            Pressure at which to calculate the derivative, [Pa]
        method : str
            Method for which to find the derivative
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        dprop_dT_P : float
            Calculated derivative property at constant pressure,
            [`units/K^order`]
        '''
        return derivative(self.calculate_P, T, dx=1e-6, args=[P, method], n=order, order=1+order*2)

    def calculate_derivative_P(self, P, T, method, order=1):
        r'''Method to calculate a derivative of a temperature and pressure
        dependent property with respect to pressure at constant temperature,
        of a given order using a specified method. Uses SciPy's derivative
        function, with a delta of 0.01 Pa and a number of points equal to
        2*order + 1.

        This method can be overwritten by subclasses who may perfer to add
        analytical methods for some or all methods as this is much faster.

        If the calculation does not succeed, returns the actual error
        encountered.

        Parameters
        ----------
        P : float
            Pressure at which to calculate the derivative, [Pa]
        T : float
            Temperature at which to calculate the derivative, [K]
        method : str
            Method for which to find the derivative
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        dprop_dP_T : float
            Calculated derivative property at constant temperature,
            [`units/Pa^order`]
        '''
        f = lambda P: self.calculate_P(T, P, method)
        return derivative(f, P, dx=1e-2, n=order, order=1+order*2)

    def TP_dependent_property_derivative_T(self, T, P, order=1):
        r'''Method to calculate a derivative of a temperature and pressure
        dependent property with respect to temperature at constant pressure,
        of a given order, according to the selected :obj:`method_P`.

        Calls :obj:`calculate_derivative_T` internally to perform the actual
        calculation.

        .. math::
            \text{derivative} = \frac{d (\text{property})}{d T}|_{P}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        P : float
            Pressure at which to calculate the derivative, [Pa]
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        dprop_dT_P : float
            Calculated derivative property, [`units/K^order`]
        '''
        try:
            return self.calculate_derivative_T(T, P, self._method_P, order)
        except:
            pass
        return None

    def TP_dependent_property_derivative_P(self, T, P, order=1):
        r'''Method to calculate a derivative of a temperature and pressure
        dependent property with respect to pressure at constant temperature,
        of a given order, according to the selected :obj:`method_P`.

        Calls :obj:`calculate_derivative_P` internally to perform the actual
        calculation.

        .. math::
            \text{derivative} = \frac{d (\text{property})}{d P}|_{T}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        P : float
            Pressure at which to calculate the derivative, [Pa]
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        dprop_dP_T : float
            Calculated derivative property, [`units/Pa^order`]
        '''
        try:
            return self.calculate_derivative_P(P, T, self._method_P, order)
        except:
            pass
        return None

class MixtureProperty(object):
    name = 'Test'
    units = 'test units'
    property_min = 0.0
    property_max = 10.0
    ranked_methods = []
    TP_zs_ws_cached = (None, None, None, None)
    prop_cached = None
    all_poly_fit = False
    _correct_pressure_pure = True
    _method = None

    pure_references = ()
    pure_reference_types = ()

    skip_prop_validity_check = False
    '''Flag to disable checking the output of the value. Saves a little time.
    '''
    skip_method_validity_check = False
    '''Flag to disable checking the validity of the method at the
    specified conditions. Saves a little time.
    '''

    def __init_subclass__(cls):
        cls.__full_path__ = "%s.%s" %(cls.__module__, cls.__qualname__)

    def set_poly_fit_coeffs(self):
        pure_objs = self.pure_objs()
        if all(i.method == POLY_FIT for i in pure_objs):
            self.all_poly_fit = True
            self.poly_fit_data = [[i.poly_fit_Tmin for i in pure_objs],
                               [i.poly_fit_Tmin_slope for i in pure_objs],
                               [i.poly_fit_Tmin_value for i in pure_objs],
                               [i.poly_fit_Tmax for i in pure_objs],
                               [i.poly_fit_Tmax_slope for i in pure_objs],
                               [i.poly_fit_Tmax_value for i in pure_objs],
                               [i.poly_fit_coeffs for i in pure_objs]]


    def __repr__(self):
        clsname = self.__class__.__name__
        base = '%s(' % (clsname)
        for k in self.custom_args:
            v = getattr(self, k)
            if v is not None:
                base += '%s=%s, ' %(k, v)
        base += 'CASs=%s, ' %(self.CASs)
        base += 'correct_pressure_pure=%s, ' %(self._correct_pressure_pure)
        base += 'method="%s", ' %(self.method)
        for attr in self.pure_references:
            base += '%s=%s, ' %(attr, getattr(self, attr))

        if base[-2:] == ', ':
            base = base[:-2]
        return base + ')'


    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        d = self.__dict__
        ans = hash_any_primitive((self.__class__, d))
        return ans

    def pure_objs(self):
        return getattr(self, self.pure_references[0])

    def __init__(self, **kwargs):
        self._correct_pressure_pure = kwargs.get('correct_pressure_pure', self._correct_pressure_pure)

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        property under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        property above.'''

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`.'''
        self.all_methods = set()
        '''Set of all methods available for a given set of information;
        filled by :obj:`load_all_methods`.'''
        self.load_all_methods()

        self.set_poly_fit_coeffs()

        if 'method' in kwargs:
            self.method = kwargs['method']

    def as_json(self, references=1):
        r'''Method to create a JSON serialization of the mixture property
        which can be stored, and reloaded later.

        Parameters
        ----------
        references : int
            How to handle references to other objects; internal parameter, [-]

        Returns
        -------
        json_repr : dict
            JSON-friendly representation, [-]

        Notes
        -----

        Examples
        --------
        '''

        d = self.__dict__.copy() # Not a the real object dictionary
        if references == 1:
            for k in self.pure_references:
                d[k] = [v.as_json() for v in d[k]]

        try:
            eos = getattr(self, 'eos')
            if eos:
                d['eos'] = eos[0].as_json()
        except:
            pass

        d['json_version'] = 1
        d["py/object"] = self.__full_path__
        d['all_methods'] = list(d['all_methods'])
        return d

    @classmethod
    def from_json(cls, string):
        r'''Method to create a MixtureProperty from a JSON
        serialization of another MixtureProperty.

        Parameters
        ----------
        json_repr : dict
            JSON-friendly representation, [-]

        Returns
        -------
        constants : :obj:`MixtureProperty`
            Newly created object from the json serialization, [-]

        Notes
        -----
        It is important that the input string be in the same format as that
        created by :obj:`MixtureProperty.as_json`.

        Examples
        --------
        '''
        d = string
        d['all_methods'] = set(d['all_methods'])
        for k, sub_cls in zip(cls.pure_references, cls.pure_reference_types):
            sub_jsons = d[k]
            d[k] = [sub_cls.from_json(j) if j is not None else None
                    for j in sub_jsons]

        try:
            eos = d['eos']
            if eos is not None:
                d['eos'] = [GCEOSMIX.from_json(eos)]
        except:
            pass
        del d['py/object']
        del d["json_version"]
        new = cls.__new__(cls)
        new.__dict__ = d
        return new

    @property
    def method(self):
        r'''Method to set the T, P, and composition dependent property method
        desired. See the :obj:`all_methods` attribute for a list of methods valid
        for the specified chemicals and inputs.
        '''
        return self._method

    @method.setter
    def method(self, method):
        self._method = method
        self.TP_zs_ws_cached = [None, None, None, None]

    @property
    def correct_pressure_pure(self):
        r'''Method to set the pressure-dependence of the model;
        if set to False, only temperature dependence is used, and if
        True, temperature and pressure dependence are used.
        '''
        return self._correct_pressure_pure

    @correct_pressure_pure.setter
    def correct_pressure_pure(self, v):
        if v != self._correct_pressure_pure:
            self._correct_pressure_pure = v
            self.TP_zs_ws_cached = [None, None, None, None]

    def _complete_zs_ws(self, zs, ws):
        if zs is None and ws is None:
            raise Exception('No Composition Specified')
        elif zs is None:
            return ws_to_zs(ws, self.MWs), ws
        elif ws is None:
            return zs, zs_to_ws(zs, self.MWs)


    def __call__(self, T, P, zs=None, ws=None):
        r'''Convenience method to calculate the property; calls
        :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`. Caches previously calculated value,
        which is an overhead when calculating many different values of
        a property. See :obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>` for more details as to the
        calculation procedure. One or both of `zs` and `ws` are required.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        P : float
            Pressure at which to calculate the property, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        if (T, P, zs, ws) == self.TP_zs_ws_cached:
            return self.prop_cached
        else:
            self.prop_cached = self.mixture_property(T, P, zs, ws)
            self.TP_zs_ws_cached = [T, P, zs, ws]
            return self.prop_cached

    @classmethod
    def test_property_validity(self, prop):
        r'''Method to test the validity of a calculated property. Normally,
        this method is used by a given property class, and has maximum and
        minimum limits controlled by the variables :obj:`property_min` and
        :obj:`property_max`.

        Parameters
        ----------
        prop : float
            property to be tested, [`units`]

        Returns
        -------
        validity : bool
            Whether or not a specifid method is valid
        '''
        if isinstance(prop, complex):
            return False
        elif prop < self.property_min:
            return False
        elif prop > self.property_max:
            return False
        return True


    def mixture_property(self, T, P, zs=None, ws=None):
        r'''Method to calculate the property with sanity checking and without
        specifying a specific method. :obj:`valid_methods` is used to obtain
        a sorted list of methods to try. Methods are then tried in order until
        one succeeds. The methods are allowed to fail, and their results are
        checked with :obj:`test_property_validity`. On success, the used method
        is stored in the variable :obj:`method`.

        If :obj:`method` is set, this method is first checked for validity with
        :obj:`test_method_validity` for the specified temperature, and if it is
        valid, it is then used to calculate the property. The result is checked
        for validity, and returned if it is valid. If either of the checks fail,
        the function retrieves a full list of valid methods with
        :obj:`valid_methods` and attempts them as described above.

        If no methods are found which succeed, returns None.
        One or both of `zs` and `ws` are required.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        P : float
            Pressure at which to calculate the property, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        try:
            method = self._method
            if not self.skip_method_validity_check:
                if not self.test_method_validity(T, P, zs, ws, method):
                    return None

            prop = self.calculate(T, P, zs, ws, self._method)
            if self.skip_prop_validity_check:
                return prop
            else:
                if self.test_property_validity(prop):
                    return prop
        except:  # pragma: no cover
            pass

        # Function returns None if it does not work.
        return None

    def excess_property(self, T, P, zs=None, ws=None):
        r'''Method to calculate the excess property with sanity checking and
        without specifying a specific method. This requires the calculation of
        the property as a function of composition at the limiting concentration
        of each component. One or both of `zs` and `ws` are required.

        .. math::
            m^E = m_{mixing} = m - \sum_i m_{i, pure}\cdot z_i

        Parameters
        ----------
        T : float
            Temperature at which to calculate the excess property, [K]
        P : float
            Pressure at which to calculate the excess property, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]

        Returns
        -------
        excess_prop : float
            Calculated excess property, [`units`]
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        N = len(zs)
        prop = self.mixture_property(T, P, zs, ws)
        tot = 0.0
        for i in range(N):
            zs2, ws2 = [0.0]*N, [0.0]*N
            zs2[i], ws2[i] = 1.0, 1.0
            tot += zs[i]*self.mixture_property(T, P, zs2, ws2)
        return prop - tot

    def partial_property(self, T, P, i, zs=None, ws=None):
        r'''Method to calculate the partial molar property with sanity checking
        and without specifying a specific method for the specified compound
        index and composition.

        .. math::
            \bar m_i = \left( \frac{\partial (n_T m)} {\partial n_i}
            \right)_{T, P, n_{j\ne i}}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the partial property, [K]
        P : float
            Pressure at which to calculate the partial property, [Pa]
        i : int
            Compound index, [-]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]

        Returns
        -------
        partial_prop : float
            Calculated partial property, [`units`]
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        def prop_extensive(ni, ns, i):
            ns[i] = ni
            n_tot = sum(ns)
            zs = normalize(ns)
            prop = self.mixture_property(T, P, zs)
            return prop*n_tot
        return derivative(prop_extensive, zs[i], dx=1E-6, args=[list(zs), i])


    def calculate_derivative_T(self, T, P, zs, ws, method, order=1):
        r'''Method to calculate a derivative of a mixture property with respect
        to temperature at constant pressure and composition
        of a given order using a specified  method. Uses SciPy's derivative
        function, with a delta of 1E-6 K and a number of points equal to
        2*order + 1.

        This method can be overwritten by subclasses who may perfer to add
        analytical methods for some or all methods as this is much faster.

        If the calculation does not succeed, returns the actual error
        encountered.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        P : float
            Pressure at which to calculate the derivative, [Pa]
        zs : list[float]
            Mole fractions of all species in the mixture, [-]
        ws : list[float]
            Weight fractions of all species in the mixture, [-]
        method : str
            Method for which to find the derivative
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        d_prop_d_T_at_P : float
            Calculated derivative property at constant pressure,
            [`units/K^order`]
        '''
        return derivative(self.calculate, T, dx=1e-6, args=[P, zs, ws, method], n=order, order=1+order*2)

    def calculate_derivative_P(self, P, T, zs, ws, method, order=1):
        r'''Method to calculate a derivative of a mixture property with respect
        to pressure at constant temperature and composition
        of a given order using a specified method. Uses SciPy's derivative
        function, with a delta of 0.01 Pa and a number of points equal to
        2*order + 1.

        This method can be overwritten by subclasses who may perfer to add
        analytical methods for some or all methods as this is much faster.

        If the calculation does not succeed, returns the actual error
        encountered.

        Parameters
        ----------
        P : float
            Pressure at which to calculate the derivative, [Pa]
        T : float
            Temperature at which to calculate the derivative, [K]
        zs : list[float]
            Mole fractions of all species in the mixture, [-]
        ws : list[float]
            Weight fractions of all species in the mixture, [-]
        method : str
            Method for which to find the derivative
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        d_prop_d_P_at_T : float
            Calculated derivative property at constant temperature,
            [`units/Pa^order`]
        '''
        f = lambda P: self.calculate(T, P, zs, ws, method)
        return derivative(f, P, dx=1e-2, n=order, order=1+order*2)


    def property_derivative_T(self, T, P, zs=None, ws=None, order=1):
        r'''Method to calculate a derivative of a mixture property with respect
        to temperature at constant pressure and composition,
        of a given order. Methods found valid by :obj:`valid_methods` are
        attempted until a method succeeds. If no methods are valid and succeed,
        None is returned.

        Calls :obj:`calculate_derivative_T` internally to perform the actual
        calculation.

        .. math::
            \text{derivative} = \frac{d (\text{property})}{d T}|_{P, z}

        One or both of `zs` and `ws` are required.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        P : float
            Pressure at which to calculate the derivative, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        d_prop_d_T_at_P : float
            Calculated derivative property, [`units/K^order`]
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        try:
            return self.calculate_derivative_T(T, P, zs, ws, self._method, order)
        except:
            pass
        return None


    def property_derivative_P(self, T, P, zs=None, ws=None, order=1):
        r'''Method to calculate a derivative of a mixture property with respect
        to pressure at constant temperature and composition,
        of a given order. Methods found valid by :obj:`valid_methods` are
        attempted until a method succeeds. If no methods are valid and succeed,
        None is returned.

        Calls :obj:`calculate_derivative_P` internally to perform the actual
        calculation.

        .. math::
            \text{derivative} = \frac{d (\text{property})}{d P}|_{T, z}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [K]
        P : float
            Pressure at which to calculate the derivative, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]
        order : int
            Order of the derivative, >= 1

        Returns
        -------
        d_prop_d_P_at_T : float
            Calculated derivative property, [`units/Pa^order`]
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        try:
            return self.calculate_derivative_P(P, T, zs, ws, self._method, order)
        except:
            pass
        return None

    def plot_isotherm(self, T, zs=None, ws=None, Pmin=None, Pmax=None,
                      methods=[], pts=50, only_valid=True):  # pragma: no cover
        r'''Method to create a plot of the property vs pressure at a specified
        temperature and composition according to either a specified list of
        methods, or the set method. User-selectable
        number of  points, and pressure range. If only_valid is set,
        :obj:`test_method_validity` will be used to check if each condition in
        the specified range is valid, and :obj:`test_property_validity` will be used
        to test the answer, and the method is allowed to fail; only the valid
        points will be plotted. Otherwise, the result will be calculated and
        displayed as-is. This will not suceed if the method fails.
        One or both of `zs` and `ws` are required.

        Parameters
        ----------
        T : float
            Temperature at which to create the plot, [K]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]
        Pmin : float
            Minimum pressure, to begin calculating the property, [Pa]
        Pmax : float
            Maximum pressure, to stop calculating the property, [Pa]
        methods : list, optional
            List of methods to consider
        pts : int, optional
            A list of points to calculate the property at; if Pmin to Pmax
            covers a wide range of method validities, only a few points may end
            up calculated for a given method so this may need to be large
        only_valid : bool
            If True, only plot successful methods and calculated properties,
            and handle errors; if False, attempt calculation without any
            checking and use methods outside their bounds
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        # This function cannot be tested
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        if Pmin is None:
            if self.Pmin is not None:
                Pmin = self.Pmin
            else:
                raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Pmax is None:
            if self.Pmax is not None:
                Pmax = self.Pmax
            else:
                raise Exception('Maximum pressure could not be auto-detected; please provide it')

        if not methods:
            methods = [self._method]
        Ps = linspace(Pmin, Pmax, pts)
        for method in methods:
            if only_valid:
                properties, Ps2 = [], []
                for P in Ps:
                    if self.test_method_validity(T, P, zs, ws, method):
                        try:
                            p = self.calculate(T, P, zs, ws, method)
                            if self.test_property_validity(p):
                                properties.append(p)
                                Ps2.append(P)
                        except:
                            pass
                plt.plot(Ps2, properties, label=method)
            else:
                properties = [self.calculate(T, P, zs, ws, method) for P in Ps]
                plt.plot(Ps, properties, label=method)
        plt.legend(loc='best')
        plt.ylabel(self.name + ', ' + self.units)
        plt.xlabel('Pressure, Pa')
        plt.title(self.name + ' of a mixture of ' + ', '.join(self.CASs)
                  + ' at mole fractions of ' + ', '.join(str(round(i, 4)) for i in zs) + '.')
        plt.show()


    def plot_isobar(self, P, zs=None, ws=None, Tmin=None, Tmax=None,
                    methods=[], pts=50, only_valid=True):  # pragma: no cover
        r'''Method to create a plot of the property vs temperature at a
        specific pressure and composition according to
        either a specified list of methods, or the selected method. User-selectable number of points, and temperature range. If
        only_valid is set,:obj:`test_method_validity` will be used to check if
        each condition in the specified range is valid, and
        :obj:`test_property_validity` will be used to test the answer, and the
        method is allowed to fail; only the valid points will be plotted.
        Otherwise, the result will be calculated and displayed as-is. This will
        not suceed if the method fails. One or both of `zs` and `ws` are
        required.

        Parameters
        ----------
        P : float
            Pressure for the isobar, [Pa]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]
        Tmin : float
            Minimum temperature, to begin calculating the property, [K]
        Tmax : float
            Maximum temperature, to stop calculating the property, [K]
        methods : list, optional
            List of methods to consider
        pts : int, optional
            A list of points to calculate the property at; if Tmin to Tmax
            covers a wide range of method validities, only a few points may end
            up calculated for a given method so this may need to be large
        only_valid : bool
            If True, only plot successful methods and calculated properties,
            and handle errors; if False, attempt calculation without any
            checking and use methods outside their bounds
        '''
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        if Tmin is None:
            if self.Tmin is not None:
                Tmin = self.Tmin
            else:
                raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Tmax is None:
            if self.Tmax is not None:
                Tmax = self.Tmax
            else:
                raise Exception('Maximum pressure could not be auto-detected; please provide it')

        if not methods:
            methods = [self._method]
        Ts = linspace(Tmin, Tmax, pts)
        for method in methods:
            if only_valid:
                properties, Ts2 = [], []
                for T in Ts:
                    if self.test_method_validity(T, P, zs, ws, method):
                        try:
                            p = self.calculate(T, P, zs, ws, method)
                            if self.test_property_validity(p):
                                properties.append(p)
                                Ts2.append(T)
                        except:
                            pass
                plt.plot(Ts2, properties, label=method)
            else:
                properties = [self.calculate(T, P, zs, ws, method) for T in Ts]
                plt.plot(Ts, properties, label=method)
        plt.legend(loc='best')
        plt.ylabel(self.name + ', ' + self.units)
        plt.xlabel('Temperature, K')
        plt.title(self.name + ' of a mixture of ' + ', '.join(self.CASs)
                  + ' at mole fractions of ' + ', '.join(str(round(i, 4)) for i in zs) + '.')
        plt.show()


    def plot_property(self, zs=None, ws=None, Tmin=None, Tmax=None, Pmin=1E5,
                      Pmax=1E6, methods=[], pts=15, only_valid=True):  # pragma: no cover
        r'''Method to create a plot of the property vs temperature and pressure
        according to either a specified list of methods, or the selected method.
        User-selectable number of points for each
        variable. If only_valid is set,:obj:`test_method_validity` will be used to
        check if each condition in the specified range is valid, and
        :obj:`test_property_validity` will be used to test the answer, and the
        method is allowed to fail; only the valid points will be plotted.
        Otherwise, the result will be calculated and displayed as-is. This will
        not suceed if the any method fails for any point. One or both of `zs`
        and `ws` are required.

        Parameters
        ----------
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]
        ws : list[float], optional
            Weight fractions of all species in the mixture, [-]
        Tmin : float
            Minimum temperature, to begin calculating the property, [K]
        Tmax : float
            Maximum temperature, to stop calculating the property, [K]
        Pmin : float
            Minimum pressure, to begin calculating the property, [Pa]
        Pmax : float
            Maximum pressure, to stop calculating the property, [Pa]
        methods : list, optional
            List of methods to consider
        pts : int, optional
            A list of points to calculate the property at for both temperature
            and pressure; pts^2 points will be calculated.
        only_valid : bool
            If True, only plot successful methods and calculated properties,
            and handle errors; if False, attempt calculation without any
            checking and use methods outside their bounds
        '''
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib.ticker import FormatStrFormatter
        import numpy.ma as ma
        if zs is None or ws is None:
            zs, ws = self._complete_zs_ws(zs, ws)
        if Pmin is None:
            if self.Pmin is not None:
                Pmin = self.Pmin
            else:
                raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Pmax is None:
            if self.Pmax is not None:
                Pmax = self.Pmax
            else:
                raise Exception('Maximum pressure could not be auto-detected; please provide it')
        if Tmin is None:
            if self.Tmin is not None:
                Tmin = self.Tmin
            else:
                raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Tmax is None:
            if self.Tmax is not None:
                Tmax = self.Tmax
            else:
                raise Exception('Maximum pressure could not be auto-detected; please provide it')

        if not methods:
            methods = [self._method]
        Ps = np.linspace(Pmin, Pmax, pts)
        Ts = np.linspace(Tmin, Tmax, pts)
        Ts_mesh, Ps_mesh = np.meshgrid(Ts, Ps)
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        handles = []
        for method in methods:
            if only_valid:
                properties = []
                for T in Ts:
                    T_props = []
                    for P in Ps:
                        if self.test_method_validity(T, P, zs, ws, method):
                            try:
                                p = self.calculate(T, P, zs, ws, method)
                                if self.test_property_validity(p):
                                    T_props.append(p)
                                else:
                                    T_props.append(None)
                            except:
                                T_props.append(None)
                        else:
                            T_props.append(None)
                    properties.append(T_props)
                properties = ma.masked_invalid(np.array(properties, dtype=np.float).T)
                handles.append(ax.plot_surface(Ts_mesh, Ps_mesh, properties, cstride=1, rstride=1, alpha=0.5))
            else:
                properties = [[self.calculate(T, P, zs, ws, method) for P in Ps] for T in Ts]
                handles.append(ax.plot_surface(Ts_mesh, Ps_mesh, properties, cstride=1, rstride=1, alpha=0.5))

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.set_xlabel('Temperature, K')
        ax.set_ylabel('Pressure, Pa')
        ax.set_zlabel(self.name + ', ' + self.units)
        plt.title(self.name + ' of a mixture of ' + ', '.join(self.CASs)
                  + ' at mole fractions of ' + ', '.join(str(round(i, 4)) for i in zs) + '.')
        plt.show(block=False)
        # The below is a workaround for a matplotlib bug
        ax.legend(handles, methods)
        plt.show(block=False)


class MultiCheb1D(object):
    '''Simple class to store set of coefficients for multiple chebyshev
    approximations and perform calculations from them.
    '''
    def __init__(self, points, coeffs):
        self.points = points
        self.coeffs = coeffs
        self.N = len(points)-1

    def __call__(self, x):
        from bisect import bisect_left
        coeffs = self.coeffs[bisect_left(self.points, x)]
        return coeffs(x)
#        return self.chebval(x, coeffs)

    @staticmethod
    def chebval(x, c):
        # copied from numpy's source, slightly optimized
        # https://github.com/numpy/numpy/blob/v1.13.0/numpy/polynomial/chebyshev.py#L1093-L1177
        x2 = 2.*x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1*x2
        return c0 + c1*x
