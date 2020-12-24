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
             set_method, select_valid_methods, test_property_validity,
             T_dependent_property, plot_T_dependent_property, interpolate,
             add_new_method, set_tabular_data, solve_prop,
             calculate_derivative, T_dependent_property_derivative,
             calculate_integral, T_dependent_property_integral,
             calculate_integral_over_T, T_dependent_property_integral_over_T,
             extrapolate, test_method_validity, calculate, tabular_extrapolation_permitted,
             interpolation_T, interpolation_T_inv, interpolation_property,  interpolation_property_inv
   :undoc-members:

Temperature and Pressure Dependent
----------------------------------
.. autoclass:: TPDependentProperty
    :members:
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
'phase_select_property']

import os
from cmath import sqrt as csqrt
from fluids.numerics import quad, brenth, newton, secant, linspace, polyint, polyint_over_x, derivative, polyder, horner, horner_and_der2, quadratic_from_f_ders, assert_close, numpy as np
from fluids.constants import R
from chemicals.utils import isnan, isinf, log, exp, ws_to_zs, zs_to_ws, e
from chemicals.utils import mix_multiple_component_flows, hash_any_primitive
from chemicals.vapor_pressure import Antoine, Antoine_coeffs_from_point, Antoine_AB_coeffs_from_point, DIPPR101_ABC_coeffs_from_point
from chemicals.dippr import EQ101
from chemicals.phase_change import Watson, Watson_n



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

try:  # pragma: no cover
    from appdirs import user_data_dir, user_config_dir
    data_dir = user_config_dir('thermo')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
except ImportError:  # pragma: no cover
    data_dir = ''

try:
    source_path = os.path.dirname(__file__) # micropython
except:
    source_path = ''

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
        = RT \sum_i y_i\left(\log(y_i) + \log(\phi_i(y)) - d_i(z)\right)

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


TEST_METHOD_1 = 'Test method 1'
TEST_METHOD_2 = 'Test method 2'
BESTFIT = 'Best fit'


class TDependentProperty(object):
    '''Class for calculating temperature-dependent chemical properties.

    On creation, a :obj:`TDependentProperty` examines all the possible methods
    implemented for calculating the property, loads whichever coefficients it
    needs (unless `load_data` is set to False), examines its input parameters,
    and selects the method it prefers. This method will continue to be used for
    all calculations until the method is changed by a call to :obj:`set_method`.

    The default list of preferred method orderings is at :obj:`ranked_methods`
    for all properties; the order can be modified there in-place, and this
    will take effect on all new :obj:`TDependentProperty` instances created.

    All methods have defined criteria for determining if they are valid before
    calculation, i.e. a minimum and maximum temperature for coefficients to be
    valid. For constant property values used due to lack of
    temperature-dependent data, a short range is normally specified as valid.

    It is not assumed that a specified method will succeed; for example many
    expressions are not mathematically valid past the critical point, and in
    some cases there is no easy way to etermine the temperature where a
    property stops being reasonable.

    Accordingly, all properties calculated are checked
    by a sanity function :obj:`test_property_validity <TDependentProperty.test_property_validity>`,
    which has basic sanity checks. If the property is not reasonable, None is
    returned.

    This framework also supports tabular data, which is interpolated from if
    specified. Interpolation is cubic-spline based if 5 or more points are
    given, and linearly interpolated with if few points are given. Tabular
    extrapolation is permitted if :obj:`tabular_extrapolation_permitted` is set to True.
    For both interpolation and extrapolation, a transform may be applied so that a property such as
    vapor pressure can be interpolated non-linearly. These are functions or
    lambda expressions which must be set for the variables :obj:`interpolation_T`,
    :obj:`interpolation_property`, and :obj:`interpolation_property_inv`.

    The extrapolation methods available are as follows:

        * 'linear' - fits the model at its temperature limits to a linear model
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
        The name of the property being calculated
    units : str
        The units of the property
    method : str
        The method was which was last used successfully to calculate a property;
        set only after the first property calculation.
    forced : bool
        If True, only user specified methods will be considered; otherwise all
        methods will be considered if none of the user specified methods succeed
    interpolation_T : function
        A function or lambda expression to transform the temperatures of
        tabular data for interpolation; e.g. 'lambda self, T: 1./T'
    interpolation_T_inv : function
        A function or lambda expression to invert the transform of temperatures
        of tabular data for interpolation; e.g. 'lambda self, x: self.Tc*(1 - x)'
    interpolation_property : function
        A function or lambda expression to transform tabular property values
        prior to interpolation; e.g. 'lambda self, P: log(P)'
    interpolation_property_inv : function
        A function or property expression to transform interpolated property
        values from the transform performed by `interpolation_property` back
        to their actual form, e.g.  'lambda self, P: exp(P)'
    tabular_extrapolation_permitted : bool
        Whether or not to allow extrapolation from tabulated data for a
        property
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
        this is a criteria used by `test_method_validity`.
    property_max : float
        Highest value expected for a property while still being valid;
        this is a criteria used by `test_method_validity`.
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
    # Dummy properties
    name = 'Property name'
    units = 'Property units'
    tabular_extrapolation_permitted = True

    interpolation_T = None
    interpolation_T_inv = None
    interpolation_property = None
    interpolation_property_inv = None

    extrapolation = 'linear'

    method = None
    forced = False

    property_min = 0
    property_max = 1E4  # Arbitrary max
    T_cached = None
    locked = False

    critical_zero = False
    '''Whether or not the property is declining and reaching zero at the
    critical point.'''

#    Tmin = None
#    Tmax = None
    ranked_methods = []

    # For methods specified by a user
    local_methods = {}

    _fit_force_n = {}
    '''Dictionary containing method: fit_n, for use in methods which should
    only ever be fit to a specific `n` value'''

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # By default, share state among subsequent objects
        return self

    def __hash__(self):
        return hash_any_primitive([self.__class__, self.__dict__])

    def __call__(self, T):
        r'''Convenience method to calculate the property; calls
        :obj::obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>`. Caches previously calculated value,
        which is an overhead when calculating many different values of
        a property. See :obj::obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>` for more details as to the
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

    def _set_common_attributes(self):

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

        self.all_methods = set()

    @classmethod
    def _fit_export_polynomials(cls, method=None, start_n=3, max_n=30,
                                eval_pts=100, save=False):
        import json
        dat = {}
        folder = os.path.join(source_path, cls.name)

        sources = cls._method_indexes()

        if method is None:
            methods = list(sources.keys())
            indexes = list(sources.values())
        else:
            methods = [method]
            indexes = [sources[method]]
        for method, index in zip(methods, indexes):
            method_dat = {}
            n = cls._fit_force_n.get(method, None)
            for CAS in index:
                print(CAS)
                obj = cls(CASRN=CAS)
                coeffs, (low, high), stats = obj.fit_polynomial(method, n=n, start_n=start_n, max_n=max_n, eval_pts=eval_pts)
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

    def set_method(self, method):
        r'''Method used to select certain property methods as having a higher
        priority than were set by default. If `forced` is true, then methods
        which were not specified are excluded from consideration.

        As a side effect, `method` is removed to ensure than the new methods
        will be used in calculations afterwards.

        An exception is raised if any of the methods specified aren't available
        for the chemical. An exception is raised if no methods are provided.

        Parameters
        ----------
        method : str or list
            Methods by name to be considered or preferred
        '''
        if method not in self.all_methods:
            raise ValueError("The given methods is not available for this chemical")
        self.method = method
        self.T_cached = None

    def select_valid_methods(self, T, check_validity=True):
        r'''Method to obtain a sorted list of methods which are valid at `T`
        according to `test_method_validity`. Considers either only user methods
        if forced is True, or all methods. User methods are first tested
        according to their listed order, and unless forced is True, then all
        methods are tested and sorted by their order in `ranked_methods`.

        Parameters
        ----------
        T : float
            Temperature at which to test methods, [K]
        check_validity : bool
            Whether or not to use `test_method_validity` to check the
            method for validity or not, [-]

        Returns
        -------
        sorted_valid_methods : list
            Sorted lists of methods valid at T according to
            `test_method_validity`
        '''
        # Consider either only the user's methods or all methods
        # Tabular data will be in both when inserted
#        if self.forced:
#            considered_methods = [self.method]
#        else:
        considered_methods = list(self.all_methods)

        # User methods (incl. tabular data); add back later, after ranking the rest
#        if self.user_methods:
#            [considered_methods.remove(i) for i in self.user_methods]
        if self.method is not None:
            considered_methods.remove(self.method)

        # Index the rest of the methods by ranked_methods, and add them to a list, sorted_methods
        preferences = sorted([self.ranked_methods.index(i) for i in considered_methods])
        sorted_methods = [self.ranked_methods[i] for i in preferences]

        # Add back the user's methods to the top, in order.
        if self.method is not None:
            sorted_methods.insert(0, self.method)
#            [sorted_methods.insert(0, i) for i in reversed(self.user_methods)]

        if check_validity:
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
        minimum limits controlled by the variables `property_min` and
        `property_max`.

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
            self.locked = True
            self.method = BESTFIT
            self.poly_fit_Tmin = Tmin = poly_fit[0]
            self.poly_fit_Tmax = Tmax = poly_fit[1]
            self.T_limits[BESTFIT] = (Tmin, Tmax)
            self.poly_fit_coeffs = poly_fit_coeffs = poly_fit[2]

            self.poly_fit_int_coeffs = polyint(poly_fit_coeffs)
            self.poly_fit_T_int_T_coeffs, self.poly_fit_log_coeff = polyint_over_x(poly_fit_coeffs)

            poly_fit_d_coeffs = polyder(poly_fit_coeffs[::-1])
            self.poly_fit_d2_coeffs = polyder(poly_fit_d_coeffs)
            self.poly_fit_d2_coeffs.reverse()
            self.poly_fit_d_coeffs = poly_fit_d_coeffs
            poly_fit_d_coeffs.reverse()

            # Extrapolation slope on high and low
            slope_delta_T = (self.poly_fit_Tmax - self.poly_fit_Tmin)*.05

            self.poly_fit_Tmax_value = self.calculate(self.poly_fit_Tmax, BESTFIT)
            if self.interpolation_property is not None:
                self.poly_fit_Tmax_value = self.interpolation_property(self.poly_fit_Tmax_value)


            # Calculate the average derivative for the last 5% of the curve
#            fit_value_high = self.calculate(self.poly_fit_Tmax - slope_delta_T, BESTFIT)
#            if self.interpolation_property is not None:
#                fit_value_high = self.interpolation_property(fit_value_high)

#            self.poly_fit_Tmax_slope = (self.poly_fit_Tmax_value
#                                        - fit_value_high)/slope_delta_T
            self.poly_fit_Tmax_slope = horner(self.poly_fit_d_coeffs, self.poly_fit_Tmax)
            self.poly_fit_Tmax_dT2 = horner(self.poly_fit_d2_coeffs, self.poly_fit_Tmax)


            # Extrapolation to lower T
            self.poly_fit_Tmin_value = self.calculate(self.poly_fit_Tmin, BESTFIT)
            if self.interpolation_property is not None:
                self.poly_fit_Tmin_value = self.interpolation_property(self.poly_fit_Tmin_value)

#            fit_value_low = self.calculate(self.poly_fit_Tmin + slope_delta_T, BESTFIT)
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
        else:
            raise ValueError("Unknown method")


    def T_dependent_property(self, T):
        r'''Method to calculate the property with sanity checking and without
        specifying a specific method. `select_valid_methods` is used to obtain
        a sorted list of methods to try. Methods are then tried in order until
        one succeeds. The methods are allowed to fail, and their results are
        checked with `test_property_validity`. On success, the used method
        is stored in the variable `method`.

        If `method` is set, this method is first checked for validity with
        `test_method_validity` for the specified temperature, and if it is
        valid, it is then used to calculate the property. The result is checked
        for validity, and returned if it is valid. If either of the checks fail,
        the function retrieves a full list of valid methods with
        `select_valid_methods` and attempts them as described above.

        If no methods are found which succeed, returns None.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]

        Returns
        -------
        prop : float
            Calculated property, [`units`]
        '''
        if self.locked:
            try:
                return self.calculate(T, BESTFIT)
            except Exception as e:
#                print(e)
                pass
        # Optimistic track, with the already set method
#        if self.method:
#            # retest within range
#            if self.test_method_validity(T, self.method):
#                try:
#                    prop = self.calculate(T, self.method)
#                    if self.test_property_validity(prop):
#                        return prop
#                except:  # pragma: no cover
#                    pass

        method = self.method
        if method is None:
            return None
        try:
            T_low, T_high = self.T_limits[method]
            in_range = T_low <= T <= T_high
        except (KeyError, AttributeError):
            in_range = self.test_method_validity(T, method)

        # Temporary - hope to make unified extrapolation behavior
        if not in_range and method in self.tabular_data and self.tabular_extrapolation_permitted:
            in_range = True


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
            #if self.test_property_validity(prop):
            #    return prop

        # Function returns None if it does not work.
        return None
        # get valid methods at T, and try them until one yields a valid
        # property; store the method and return the answer
        # self.sorted_valid_methods = self.select_valid_methods(T)
        # for method in self.sorted_valid_methods:
        #     try:
        #         prop = self.calculate(T, method)
        #         if self.test_property_validity(prop):
        #             self.method = method
        #             return prop
        #     except:  # pragma: no cover
        #         pass
        #
        # # Function returns None if it does not work.
        # return None

#    def plot(self, Tmin=None, Tmax=None, methods=[], pts=50, only_valid=True, order=0): # pragma: no cover
#            return self.plot_T_dependent_property(Tmin=Tmin, Tmax=Tmax, methods=methods, pts=pts, only_valid=only_valid, order=order)

    def plot_T_dependent_property(self, Tmin=None, Tmax=None, methods=[],
                                  pts=250, only_valid=True, order=0, show=True,
                                  axes='semilogy'):  # pragma: no cover
        r'''Method to create a plot of the property vs temperature according to
        either a specified list of methods, or user methods (if set), or all
        methods. User-selectable number of points, and temperature range. If
        only_valid is set,`test_method_validity` will be used to check if each
        temperature in the specified range is valid, and
        `test_property_validity` will be used to test the answer, and the
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
            if self.locked:
                methods.add('Best fit')

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
                                p = self.calculate(T=T, method=method)
                                if self.test_property_validity(p):
                                    properties.append(p)
                                    Ts2.append(T)
                            except:
                                pass
                    plot_fun(Ts2, properties, label=method)
                else:
                    properties = [self.calculate(T=T, method=method) for T in Ts]
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

#    def extrapolate_tabular(self, T):
#        if 'EXTRAPOLATE_TABULAR' not in self.tabular_data:
#            if self.Tmin is None or self.Tmax is None:
#                raise Exception('Could not automatically generate interpolation'
#                                ' data for property %s of %s because temperature '
#                                'limits could not be determined.' %(self.name, self.CASRN))
#
#            Tmin = max(20, self.Tmin)
#            if hasattr(self, 'Tb') and self.Tb is not None:
#                Tmin = min(Tmin, self.Tb)
#
#            Ts = linspace(Tmin, self.Tmax, 200)
#            properties = [self.T_dependent_property(T) for T in Ts]
#            Ts_cleaned = []
#            properties_cleaned = []
#            for T, p in zip(Ts, properties):
#                if p is not None:
#                    Ts_cleaned.append(T)
#                    properties_cleaned.append(p)
#            self.tabular_data['EXTRAPOLATE_TABULAR'] = (Ts_cleaned, properties_cleaned)
#        return self.interpolate(T, 'EXTRAPOLATE_TABULAR')
#

    def interpolate(self, T, name):
        r'''Method to perform interpolation on a given tabular data set
        previously added via :obj:`set_tabular_data`. This method will create the
        interpolators the first time it is used on a property set, and store
        them for quick future use.

        Interpolation is cubic-spline based if 5 or more points are available,
        and linearly interpolated if not. Extrapolation is always performed
        linearly. This function uses the transforms `interpolation_T`,
        `interpolation_property`, and `interpolation_property_inv` if set. If
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

            if self.interpolation_T:  # Transform ths Ts with interpolation_T if set
                Ts2 = [self.interpolation_T(T2) for T2 in Ts]
            else:
                Ts2 = Ts
            if self.interpolation_property:  # Transform ths props with interpolation_property if set
                properties2 = [self.interpolation_property(p) for p in properties]
            else:
                properties2 = properties
            # Only allow linear extrapolation, but with whatever transforms are specified
            extrapolator = interp1d(Ts2, properties2, fill_value='extrapolate')
            # If more than 5 property points, create a spline interpolation
            if len(properties) >= 5:
                spline = interp1d(Ts2, properties2, kind='cubic')
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

    def add_new_method(self, f, name, Tmin, Tmax, f_der_general=None,
                       f_der=None, f_der2=None, f_der3=None, f_int=None,
                       f_int_over_T=None):
        if not self.local_methods:
            self.local_methods = local_methods = {}
        local_methods[name] = (f, Tmin, Tmax, f_der_general, f_der, f_der2,
                      f_der3, f_int, f_int_over_T)
        self.all_methods.add(name)

    def set_tabular_data(self, Ts, properties, name=None, check_properties=True):
        r'''Method to set tabular data to be used for interpolation.
        Ts must be in increasing order. If no name is given, data will be
        assigned the name 'Tabular data series #x', where x is the number of
        previously added tabular data series. The name is added to all
        methods and iserted at the start of user methods,

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
            `test_property_validity` and raise an exception if any are not
            valid
        '''
        # Ts must be in increasing order.
        if check_properties:
            for p in properties:
                if not self.test_property_validity(p):
                    raise Exception('One of the properties specified are not feasible')
        if not all(b > a for a, b in zip(Ts, Ts[1:])):
            raise Exception('Temperatures are not sorted in increasing order')

        if name is None:
            name = 'Tabular data series #' + str(len(self.tabular_data))  # Will overwrite a poorly named series
        self.tabular_data[name] = (Ts, properties)

        self.method = None
        self.all_methods.add(name)

        self.set_method(method=name)

    def solve_prop(self, goal):
        r'''Method to solve for the temperature at which a property is at a
        specified value. :obj:`T_dependent_property <thermo.utils.TDependentProperty.T_dependent_property>` is used to calculate the value
        of the property as a function of temperature.

        Checks the given property value with `test_property_validity` first
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
            raise Exception('Input property is not considered plausible; no method would calculate it.')

        def error(T):
            return self.T_dependent_property(T) - goal
        T_limits = self.T_limits[self.method]
        if self.extrapolation is None:
            try:
                return brenth(error, T_limits[0], T_limits[1])
            except ValueError:
                raise Exception('To within the implemented temperature range, it is not possible to calculate the desired value.')
        else:
            high = self.Tc if self.critical_zero and self.Tc is not None else None
            return secant(error, x0=T_limits[0], x1=T_limits[1], low=1e-4, xtol=1e-12, bisection=True, high=high)

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
        try:
            return derivative(self.calculate, T, dx=T*1e-6, args=[method], n=order, order=1+order*2)
        except:
            Tmin, Tmax = self.T_limits[method]
            return derivative(self.calculate, T, dx=T*1e-6, args=[method], n=order, order=1+order*2,
                              lower_limit=Tmin, upper_limit=Tmax)

    def T_dependent_property_derivative(self, T, order=1):
        r'''Method to obtain a derivative of a property with respect to
        temperature, of a given order. Methods found valid by
        `select_valid_methods` are attempted until a method succeeds. If no
        methods are valid and succeed, None is returned.

        Calls `calculate_derivative` internally to perform the actual
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
        if self.locked:
            try:
                return self.calculate_derivative(T, BESTFIT, order)
            except Exception as e:
                pass

        sorted_valid_methods = self.select_valid_methods(T)
        for method in sorted_valid_methods:
            try:
                return self.calculate_derivative(T, method, order)
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
        return float(quad(self.calculate, T1, T2, args=(method))[0])

    def T_dependent_property_integral(self, T1, T2):
        r'''Method to calculate the integral of a property with respect to
        temperature, using a specified method. Methods found valid by
        `select_valid_methods` are attempted until a method succeeds. If no
        methods are valid and succeed, None is returned.

        Calls `calculate_integral` internally to perform the actual
        calculation.

        .. math::
            \text{integral} = \int_{T_1}^{T_2} \text{property} \; dT

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
        if self.locked:
            return self.calculate_integral(T1, T2, BESTFIT)

        Tavg = 0.5*(T1+T2)
#        if self.method:
#            # retest within range
#            if self.test_method_validity(Tavg, self.method):
#                try:
#                    return self.calculate_integral(T1, T2, self.method)
#                except:  # pragma: no cover
#                    pass

        sorted_valid_methods = self.select_valid_methods(Tavg)
        for method in sorted_valid_methods:
            try:
                return self.calculate_integral(T1, T2, method)
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
        return float(quad(lambda T: self.calculate(T, method)/T, T1, T2)[0])

    def T_dependent_property_integral_over_T(self, T1, T2):
        r'''Method to calculate the integral of a property over temperature
        with respect to temperature, using a specified method. Methods found
        valid by `select_valid_methods` are attempted until a method succeeds.
        If no methods are valid and succeed, None is returned.

        Calls `calculate_integral_over_T` internally to perform the actual
        calculation.

        .. math::
            \text{integral} = \int_{T_1}^{T_2} \frac{\text{property}}{T} \; dT

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
        if self.locked:
            return self.calculate_integral_over_T(T1, T2, BESTFIT)


        Tavg = 0.5*(T1+T2)
#        if self.method:
#            # retest within range
#            if self.test_method_validity(Tavg, self.method):
#                try:
#                    return self.calculate_integral_over_T(T1, T2, self.method)
#                except:  # pragma: no cover
#                    pass

        sorted_valid_methods = self.select_valid_methods(Tavg)
        for method in sorted_valid_methods:
            try:
                return self.calculate_integral_over_T(T1, T2, method)
            except:
                pass
        return None

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

        T_limits = self.T_limits
        for extrapolation in extrapolations:

            if extrapolation == 'linear':
                self.linear_extrapolation_coeffs = linear_extrapolation_coeffs = {}
                for m in self.all_methods:
                    Tmin, Tmax = T_limits[m]
                    try:
                        v_low = self.calculate(T=Tmin, method=m)
                        d_low = self.calculate_derivative(T=Tmin, method=m, order=1)
                    except:
                        v_low, d_low = None, None
                    try:
                        v_high = self.calculate(T=Tmax, method=m)
                        d_high = self.calculate_derivative(T=Tmax, method=m, order=1)
                    except:
                        v_high, d_high = None, None
                    linear_extrapolation_coeffs[m] = (v_low, d_low, v_high, d_high)
            elif extrapolation == 'AntoineAB':
                self.Antoine_AB_coeffs = Antoine_AB_coeffs = {}
                for m in self.all_methods:
                    Tmin, Tmax = T_limits[m]
                    try:
                        v_low = self.calculate(T=Tmin, method=m)
                        d_low = self.calculate_derivative(T=Tmin, method=m, order=1)
                        AB_low = Antoine_AB_coeffs_from_point(T=Tmin, Psat=v_low, dPsat_dT=d_low, base=e)
                    except:
                        AB_low = None
                    try:
                        v_high = self.calculate(T=Tmax, method=m)
                        d_high = self.calculate_derivative(T=Tmax, method=m, order=1)
                        AB_high = Antoine_AB_coeffs_from_point(T=Tmax, Psat=v_high, dPsat_dT=d_high, base=e)
                    except:
                        AB_high = None
                    Antoine_AB_coeffs[m] = (AB_low, AB_high)
            elif extrapolation == 'DIPPR101_ABC':
                self.DIPPR101_ABC_coeffs = DIPPR101_ABC_coeffs = {}
                for m in self.all_methods:
                    Tmin, Tmax = T_limits[m]
                    try:
                        v_low = self.calculate(T=Tmin, method=m)
                        d0_low = self.calculate_derivative(T=Tmin, method=m, order=1)
                        d1_low = self.calculate_derivative(T=Tmin, method=m, order=2)
                        DIPPR101_ABC_low = DIPPR101_ABC_coeffs_from_point(Tmin, v_low, d0_low, d1_low)
                    except:
                        DIPPR101_ABC_low = None
                    try:
                        v_high = self.calculate(T=Tmax, method=m)
                        d0_high = self.calculate_derivative(T=Tmax, method=m, order=1)
                        d1_high = self.calculate_derivative(T=Tmax, method=m, order=2)
                        DIPPR101_ABC_high = DIPPR101_ABC_coeffs_from_point(Tmax, v_high, d0_high, d1_high)
                    except:
                        DIPPR101_ABC_high = None
                    DIPPR101_ABC_coeffs[m] = (DIPPR101_ABC_low, DIPPR101_ABC_high)
            elif extrapolation == 'Watson':
                self.Watson_coeffs = Watson_coeffs = {}
                for m in self.all_methods:
                    Tmin, Tmax = T_limits[m]
                    delta = (Tmax-Tmin)*1e-4
                    try:
                        v0_low = self.calculate(T=Tmin, method=m)
                        v1_low = self.calculate(T=Tmin+delta, method=m)
                        n_low = Watson_n(Tmin, Tmin+delta, v0_low, v1_low, self.Tc)
                    except:
                        v0_low, v1_low, n_low = None, None, None
                    try:
                        v0_high = self.calculate(T=Tmax, method=m)
                        v1_high = self.calculate(T=Tmax-delta, method=m)
                        n_high = Watson_n(Tmax, Tmax-delta, v0_high, v1_high, self.Tc)
                    except:
                        v0_high, v1_high, n_high = None, None, None
                    Watson_coeffs[m] = (v0_low, n_low, v0_high, n_high)
            else:
                raise ValueError("Could not recognize extrapolation setting")


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
            v_low, d_low, v_high, d_high = self.linear_extrapolation_coeffs[method]
            if low:
                if v_low is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at minimum temperature")
                return v_low + d_low*(T - T_low)
            else:
                if v_high is None:
                    raise ValueError("Could not extrapolate - model failed to calculate at maximum temperature")
                return v_high + d_high*(T - T_high)
        elif extrapolation == 'AntoineAB':
            T_low, T_high = T_limits[method]
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





    # Dummy functions, always to be overwritten, only for testing

    def __init__(self, CASRN=''):
        '''Create an instance of TDependentProperty. Should be overwritten by
        a method created specific to a property. Should take all constant
        properties on creation.

        Attributes
        ----------
        '''
        self.CASRN = CASRN
        self.load_all_methods()

        self.ranked_methods = [TEST_METHOD_2, TEST_METHOD_1]  # Never changes
        self.tabular_data = {}
        self.tabular_data_interpolators = {}

    def load_all_methods(self):
        r'''Method to load all data, and set all_methods based on the available
        data and properties. Demo function for testing only; must be
        implemented according to the methods available for each individual
        method.
        '''
        methods = []
        Tmins, Tmaxs = [], []
        if self.CASRN in ['7732-18-5', '67-56-1', '64-17-5']:
            methods.append(TEST_METHOD_1)
            self.TEST_METHOD_1_Tmin = 200.
            self.TEST_METHOD_1_Tmax = 350
            self.TEST_METHOD_1_coeffs = [1, .002]
            Tmins.append(self.TEST_METHOD_1_Tmin); Tmaxs.append(self.TEST_METHOD_1_Tmax)
        if self.CASRN in ['67-56-1']:
            methods.append(TEST_METHOD_2)
            self.TEST_METHOD_2_Tmin = 300.
            self.TEST_METHOD_2_Tmax = 400
            self.TEST_METHOD_2_coeffs = [1, .003]
            Tmins.append(self.TEST_METHOD_2_Tmin); Tmaxs.append(self.TEST_METHOD_2_Tmax)
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            self.Tmin = min(Tmins)
            self.Tmax = max(Tmaxs)

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
        if method == TEST_METHOD_1:
            prop = self.TEST_METHOD_1_coeffs[0] + self.TEST_METHOD_1_coeffs[1]*T
        elif method == TEST_METHOD_2:
            prop = self.TEST_METHOD_2_coeffs[0] + self.TEST_METHOD_2_coeffs[1]*T
        elif method in self.tabular_data:
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
        if method == TEST_METHOD_1:
            if T < self.TEST_METHOD_1_Tmin or T > self.TEST_METHOD_1_Tmax:
                validity = False
        elif method == TEST_METHOD_2:
            if T < self.TEST_METHOD_2_Tmin or T > self.TEST_METHOD_2_Tmax:
                validity = False
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity


class TPDependentProperty(TDependentProperty):
    '''Class for calculating temperature and pressure dependent chemical
    properties.'''
    interpolation_P = None
    method_P = None
    forced_P = False
    TP_cached = None

    def __call__(self, T, P):
        r'''Convenience method to calculate the property; calls
        :obj::obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>`. Caches previously calculated value,
        which is an overhead when calculating many different values of
        a property. See :obj::obj:`TP_dependent_property <thermo.utils.TPDependentProperty.TP_dependent_property>` for more details as to the
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
            self.prop_cached = self.TP_or_T_dependent_property(T, P)
            self.TP_cached = (T, P)
            return self.prop_cached

    def set_user_methods_P(self, user_methods_P, forced_P=False):
        r'''Method to set the pressure-dependent property methods desired for
        consideration by the user. Can be used to exclude certain methods which
        might have unacceptable accuracy.

        As a side effect, the previously selected method is removed when
        this method is called to ensure user methods are tried in the desired
        order.

        Parameters
        ----------
        user_methods_P : str or list
            Methods by name to be considered or preferred for pressure effect.
        forced_P : bool, optional
            If True, only the user specified methods will ever be considered;
            if False other methods will be considered if no user methods
            suceed.
        '''
        # Accept either a string or a list of methods, and whether
        # or not to only consider the false methods
        if isinstance(user_methods_P, str):
            user_methods_P = [user_methods_P]

        # The user's order matters and is retained for use by select_valid_methods
        self.user_methods_P = user_methods_P
        self.forced_P = forced_P

        # Validate that the user's specified methods are actual methods
        if set(self.user_methods_P).difference(self.all_methods_P):
            raise Exception("One of the given methods is not available for this chemical")
        if not self.user_methods_P and self.forced:
            raise Exception('Only user specified methods are considered when forced is True, but no methods were provided')

        # Remove previously selected methods
        self.method_P = None
        self.sorted_valid_methods_P = []
        self.TP_cached = None

    def select_valid_methods_P(self, T, P, check_validity=True):
        r'''Method to obtain a sorted list methods which are valid at `T`
        according to `test_method_validity`. Considers either only user methods
        if forced is True, or all methods. User methods are first tested
        according to their listed order, and unless forced is True, then all
        methods are tested and sorted by their order in `ranked_methods`.

        Parameters
        ----------
        T : float
            Temperature at which to test methods, [K]
        P : float
            Pressure at which to test methods, [Pa]
        check_validity : bool
            Whether or not to use `test_method_validity` to check the
            method for validity or not, [-]

        Returns
        -------
        sorted_valid_methods_P : list
            Sorted lists of methods valid at T and P according to
            `test_method_validity`
        '''
        # Same as select_valid_methods but with _P added to variables
        if self.forced_P:
            considered_methods = list(self.user_methods_P)
        else:
            considered_methods = list(self.all_methods_P)

        if self.user_methods_P:
            [considered_methods.remove(i) for i in self.user_methods_P]

        preferences = sorted([self.ranked_methods_P.index(i) for i in considered_methods])
        sorted_methods = [self.ranked_methods_P[i] for i in preferences]

        if self.user_methods_P:
            [sorted_methods.insert(0, i) for i in reversed(self.user_methods_P)]

        if not check_validity:
            return sorted_methods

        sorted_valid_methods_P = []
        for method in sorted_methods:
            if self.test_method_validity_P(T, P, method):
                sorted_valid_methods_P.append(method)

        return sorted_valid_methods_P

    def TP_dependent_property(self, T, P):
        r'''Method to calculate the property with sanity checking and without
        specifying a specific method. `select_valid_methods_P` is used to obtain
        a sorted list of methods to try. Methods are then tried in order until
        one succeeds. The methods are allowed to fail, and their results are
        checked with `test_property_validity`. On success, the used method
        is stored in the variable `method_P`.

        If `method_P` is set, this method is first checked for validity with
        `test_method_validity_P` for the specified temperature, and if it is
        valid, it is then used to calculate the property. The result is checked
        for validity, and returned if it is valid. If either of the checks fail,
        the function retrieves a full list of valid methods with
        `select_valid_methods_P` and attempts them as described above.

        If no methods are found which succeed, returns None.

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
        # Optimistic track, with the already set method
#        if self.method_P:
#            # retest within range
#            if self.test_method_validity_P(T, P, self.method_P):
#                try:
#                    prop = self.calculate_P(T, P, self.method_P)
#                    if self.test_property_validity(prop):
#                        return prop
#                except:  # pragma: no cover
#                    pass
#
        # get valid methods at T, and try them until one yields a valid
        # property; store the method_P and return the answer
#        self.sorted_valid_methods_P = self.select_valid_methods_P(T, P)
#        for method_P in self.sorted_valid_methods_P:
        try:
            prop = self.calculate_P(T, P, self.method_P)
            if self.test_property_validity(prop):
                return prop
        except:  # pragma: no cover
            pass
        # Function returns None if it does not work.
        return None

    def TP_or_T_dependent_property(self, T, P):
#        self.method = None
#        self.method_P = None
        if P is not None:
            prop = self.TP_dependent_property(T, P)
        if P is None or prop is None:
            prop = self.T_dependent_property(T)
        return prop


    def set_tabular_data_P(self, Ts, Ps, properties, name=None, check_properties=True):
        r'''Method to set tabular data to be used for interpolation.
        Ts and Psmust be in increasing order. If no name is given, data will be
        assigned the name 'Tabular data series #x', where x is the number of
        previously added tabular data series. The name is added to all
        methods and is inserted at the start of user methods,

        Parameters
        ----------
        Ts : array-like
            Increasing array of temperatures at which properties are specified, [K]
        Ps : array-like
            Increasing array of pressures at which properties are specified, [Pa]
        properties : array-like
            List of properties at Ts, [`units`]
        name : str, optional
            Name assigned to the data
        check_properties : bool
            If True, the properties will be checked for validity with
            `test_property_validity` and raise an exception if any are not
            valid
        '''
        # Ts must be in increasing order.
        if check_properties:
            for p in np.array(properties).ravel():
                if not self.test_property_validity(p):
                    raise Exception('One of the properties specified are not feasible')
        if not all(b > a for a, b in zip(Ts, Ts[1:])):
            raise Exception('Temperatures are not sorted in increasing order')
        if not all(b > a for a, b in zip(Ps, Ps[1:])):
            raise Exception('Pressures are not sorted in increasing order')

        if name is None:
            name = 'Tabular data series #' + str(len(self.tabular_data))  # Will overwrite a poorly named series
        self.tabular_data[name] = (Ts, Ps, properties)

        self.method_P = name
#        self.user_methods_P.insert(0, name)
        self.all_methods_P.add(name)


#        self.set_user_methods_P(user_methods_P=self.user_methods_P, forced_P=self.forced_P)

    def interpolate_P(self, T, P, name):
        r'''Method to perform interpolation on a given tabular data set
        previously added via `set_tabular_data_P`. This method will create the
        interpolators the first time it is used on a property set, and store
        them for quick future use.

        Interpolation is cubic-spline based if 5 or more points are available,
        and linearly interpolated if not. Extrapolation is always performed
        linearly. This function uses the transforms `interpolation_T`,
        `interpolation_P`,
        `interpolation_property`, and `interpolation_property_inv` if set. If
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
        Ts, Ps, properties = self.tabular_data[name]
        if not self.tabular_extrapolation_permitted:
            if T < Ts[0] or T > Ts[-1] or P < Ps[0] or P > Ps[-1]:
                raise ValueError("Extrapolation not permitted and conditions outside of range")

        # If the interpolator and extrapolator has already been created, load it
        if key in self.tabular_data_interpolators:
            extrapolator, spline = self.tabular_data_interpolators[key]
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
                properties2 = [self.interpolation_property(p) for p in properties]
            else:
                properties2 = properties
            # Only allow linear extrapolation, but with whatever transforms are specified
            extrapolator = interp2d(Ts2, Ps2, properties2)  # interpolation if fill value is missing
            # If more than 5 property points, create a spline interpolation
            if len(properties) >= 5:
                spline = interp2d(Ts2, Ps2, properties2, kind='cubic')
            else:
                spline = None
            self.tabular_data_interpolators[key] = (extrapolator, spline)

        # Load the stores values, tor checking which interpolation strategy to
        # use.
        Ts, Ps, properties = self.tabular_data[name]

        if T < Ts[0] or T > Ts[-1] or not spline or P < Ps[0] or P > Ps[-1]:
            tool = extrapolator
        else:
            tool = spline

        if self.interpolation_T:
            T = self.interpolation_T(T)
        if self.interpolation_P:
            P = self.interpolation_T(P)
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
        `test_method_validity_P` will be used to check if each condition in
        the specified range is valid, and `test_property_validity` will be used
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
            if self.user_methods_P:
                methods_P = self.user_methods_P
            else:
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
        only_valid is set,`test_method_validity_P` will be used to check if
        each condition in the specified range is valid, and
        `test_property_validity` will be used to test the answer, and the
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
        if not methods_P:
            if self.user_methods_P:
                methods_P = self.user_methods_P
            else:
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
        variable. If only_valid is set,`test_method_validity_P` will be used to
        check if each condition in the specified range is valid, and
        `test_property_validity` will be used to test the answer, and the
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
                properties = [[self.calculate_P(T, P, method_P) for P in Ps] for T in Ts]
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
        d_prop_d_T_at_P : float
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
        d_prop_d_P_at_T : float
            Calculated derivative property at constant temperature,
            [`units/Pa^order`]
        '''
        f = lambda P: self.calculate_P(T, P, method)
        return derivative(f, P, dx=1e-2, n=order, order=1+order*2)

    def TP_dependent_property_derivative_T(self, T, P, order=1):
        r'''Method to calculate a derivative of a temperature and pressure
        dependent property with respect to temperature at constant pressure,
        of a given order. Methods found valid by `select_valid_methods_P` are
        attempted until a method succeeds. If no methods are valid and succeed,
        None is returned.

        Calls `calculate_derivative_T` internally to perform the actual
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
        d_prop_d_T_at_P : float
            Calculated derivative property, [`units/K^order`]
        '''
        sorted_valid_methods_P = self.select_valid_methods_P(T, P)
        for method in sorted_valid_methods_P:
            try:
                return self.calculate_derivative_T(T, P, method, order)
            except:
                pass
        return None

    def TP_dependent_property_derivative_P(self, T, P, order=1):
        r'''Method to calculate a derivative of a temperature and pressure
        dependent property with respect to pressure at constant temperature,
        of a given order. Methods found valid by `select_valid_methods_P` are
        attempted until a method succeeds. If no methods are valid and succeed,
        None is returned.

        Calls `calculate_derivative_P` internally to perform the actual
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
        d_prop_d_P_at_T : float
            Calculated derivative property, [`units/Pa^order`]
        '''
        sorted_valid_methods_P = self.select_valid_methods_P(T, P)
        for method in sorted_valid_methods_P:
            try:
                return self.calculate_derivative_P(P, T, method, order)
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
    _correct_pressure_pure = True
    _method = None

    skip_prop_validity_check = False
    '''Flag to disable checking the output of the value. Saves a little time.
    '''
    skip_method_validity_check = False
    '''Flag to disable checking the validity of the method at the
    specified conditions. Saves a little time.
    '''

    def set_poly_fit_coeffs(self):
        if all(i.locked for i in self.pure_objs):
            self.locked = True
            pure_objs = self.pure_objs
            self.poly_fit_data = [[i.poly_fit_Tmin for i in pure_objs],
                               [i.poly_fit_Tmin_slope for i in pure_objs],
                               [i.poly_fit_Tmin_value for i in pure_objs],
                               [i.poly_fit_Tmax for i in pure_objs],
                               [i.poly_fit_Tmax_slope for i in pure_objs],
                               [i.poly_fit_Tmax_value for i in pure_objs],
                               [i.poly_fit_coeffs for i in pure_objs]]

    @property
    def method(self):
        r'''Method to set the T, P, and composition dependent property method
        desired. See the `all_methods` attribute for a list of methods valid
        for the specified chemicals and inputs.
        '''
        return self._method

    @method.setter
    def method(self, method):
        self._method = method
        self.TP_zs_ws_cached = (None, None, None, None)

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
            self.TP_zs_ws_cached = (None, None, None, None)

    def _complete_zs_ws(self, zs, ws):
        if zs is None and ws is None:
            raise Exception('No Composition Specified')
        elif zs is None:
            return ws_to_zs(ws, self.MWs), ws
        elif ws is None:
            return zs, zs_to_ws(zs, self.MWs)


    def __call__(self, T, P, zs=None, ws=None):
        r'''Convenience method to calculate the property; calls
        :obj::obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>`. Caches previously calculated value,
        which is an overhead when calculating many different values of
        a property. See :obj::obj:`mixture_property <thermo.utils.MixtureProperty.mixture_property>` for more details as to the
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
            self.TP_zs_ws_cached = (T, P, zs, ws)
            return self.prop_cached

    @classmethod
    def test_property_validity(self, prop):
        r'''Method to test the validity of a calculated property. Normally,
        this method is used by a given property class, and has maximum and
        minimum limits controlled by the variables `property_min` and
        `property_max`.

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
        specifying a specific method. `select_valid_methods` is used to obtain
        a sorted list of methods to try. Methods are then tried in order until
        one succeeds. The methods are allowed to fail, and their results are
        checked with `test_property_validity`. On success, the used method
        is stored in the variable `method`.

        If `method` is set, this method is first checked for validity with
        `test_method_validity` for the specified temperature, and if it is
        valid, it is then used to calculate the property. The result is checked
        for validity, and returned if it is valid. If either of the checks fail,
        the function retrieves a full list of valid methods with
        `select_valid_methods` and attempts them as described above.

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
        of a given order. Methods found valid by `select_valid_methods` are
        attempted until a method succeeds. If no methods are valid and succeed,
        None is returned.

        Calls `calculate_derivative_T` internally to perform the actual
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
        of a given order. Methods found valid by `select_valid_methods` are
        attempted until a method succeeds. If no methods are valid and succeed,
        None is returned.

        Calls `calculate_derivative_P` internally to perform the actual
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
        `test_method_validity` will be used to check if each condition in
        the specified range is valid, and `test_property_validity` will be used
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
        only_valid is set,`test_method_validity` will be used to check if
        each condition in the specified range is valid, and
        `test_property_validity` will be used to test the answer, and the
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
        variable. If only_valid is set,`test_method_validity` will be used to
        check if each condition in the specified range is valid, and
        `test_property_validity` will be used to test the answer, and the
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
