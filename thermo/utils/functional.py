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
'''

__all__ = ['has_matplotlib', 'Stateva_Tsvetkov_TPDF', 'TPD',
'assert_component_balance', 'assert_energy_balance', 'allclose_variable',
'identify_phase', 'phase_select_property']

from chemicals.utils import mix_multiple_component_flows, velocity_to_molar_velocity
from fluids.constants import R
from fluids.numerics import assert_close, log, trunc_log
from fluids.numerics import numpy as np

_has_matplotlib = None
def has_matplotlib():
    global _has_matplotlib
    if _has_matplotlib is None:
        try:
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
    -4.0396017390e-09

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
        if yi == 0 and zi == 0:
            continue
        di = trunc_log(zi) + phi_zi
        tot += yi*(trunc_log(yi) + phi_yi - di)
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
    '''
    try:
        [_ for _ in inlets]
    except TypeError:
        inlets = [inlets]
    try:
        [_ for _ in outlets]
    except TypeError:
        outlets = [outlets]

    if not inlets and not outlets:
        return True

    feed_CASs = [i.CASs for i in inlets]
    product_CASs = [i.CASs for i in outlets]

    if reactive:
        # mass balance
        assert_close(sum([i.m for i in inlets]), sum([i.m for i in outlets]))

        try:
            ws = [i.ws() for i in inlets]
        except:
            ws = [i.ws for i in inlets]

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
        feed_cmps, feed_element_flows = mix_multiple_component_flows(
            IDs=[list(i.atom_flows().keys()) for i in inlets],
            flows=[1 for i in inlets],
            fractions=[list(i.atom_flows().values()) for i in inlets])

        # feed_cmps, feed_element_flows = mix_multiple_component_flows(IDs=[list(i.atoms.keys()) for i in inlets],
        #                                                       flows=[i.n for i in inlets],
        #                                                       fractions=[list(i.atoms.values()) for i in inlets])
        feed_element_flows = {i:j for i, j in zip(feed_cmps, feed_element_flows)}


        product_cmps, product_element_flows = mix_multiple_component_flows(IDs=[list(i.atom_flows().keys()) for i in outlets],
                                                              flows=[1 for i in outlets],
                                                              fractions=[list(i.atom_flows().values()) for i in outlets])
        # product_cmps, product_element_flows = mix_multiple_component_flows(IDs=[list(i.atoms.keys()) for i in outlets],
        #                                                       flows=[i.n for i in outlets],
        #                                                       fractions=[list(i.atoms.values()) for i in outlets])
        product_element_flows = {i:j for i, j in zip(product_cmps, product_element_flows)}

        for ele, flow in feed_element_flows.items():
            assert_close(flow, product_element_flows[ele], rtol=rtol, atol=atol)

        if set(feed_cmps) != set(product_cmps):
            raise Exception('Product and feeds have different elements in them')
        return True

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

    return True

def assert_energy_balance(inlets, outlets, energy_inlets, energy_outlets,
                          rtol=1E-9, atol=0.0, reactive=False,
                          inlet_areas=None, outlet_areas=None,
                          inlet_elevations=None, outlet_elevations=None):
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

    if inlet_areas is not None:
        inlet_molar_velocity = [velocity_to_molar_velocity(inlet.Q/A, inlet.MW()) for A, inlet in zip(inlet_areas, inlets)]
    if outlet_areas is not None:
        outlet_molar_velocity = [velocity_to_molar_velocity(outlet.Q/A, outlet.MW()) for A, outlet in zip(outlet_areas, outlets)]

    # Energy streams need to handle direction, not just magnitude
    energy_in = 0.0
    for feed in inlets:
        if not reactive:
            energy_in += feed.energy
        else:
            energy_in += feed.energy_reactive
    for feed in energy_inlets:
        energy_in += feed.Q
    if inlet_areas is not None:
        for v, inlet in zip(inlet_molar_velocity, inlets):
            energy_in += 0.5*v*v*inlet.n

    energy_out = 0.0
    for product in outlets:
        if not reactive:
            energy_out += product.energy
        else:
            energy_out += product.energy_reactive
    for product in energy_outlets:
        energy_out += product.Q
    if outlet_areas is not None:
        for v, outlet in zip(outlet_molar_velocity, outlets):
            energy_out += 0.5*v*v*outlet.n

    assert_close(energy_in, energy_out, rtol=rtol, atol=atol)
