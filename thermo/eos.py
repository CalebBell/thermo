r'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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

This module contains implementations of most cubic equations of state for
pure components. This includes Peng-Robinson, SRK, Van der Waals, PRSV, TWU and
many other variants.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Base Class
==========
.. autoclass:: GCEOS
    :members:
    :undoc-members:
    :special-members: __repr__
    :show-inheritance:
    :exclude-members: _P_zero_g_cheb_coeffs, _P_zero_l_cheb_coeffs,
                      main_derivatives_and_departures, derivatives_and_departures

Standard Peng-Robinson Family EOSs
==================================

Standard Peng Robinson
----------------------
.. autoclass:: PR
   :show-inheritance:
   :members: a_alpha_pure, a_alpha_and_derivatives_pure, d3a_alpha_dT3_pure,
             solve_T, P_max_at_V, c1, c2, Zc

Peng Robinson (1978)
--------------------
.. autoclass:: PR78
   :show-inheritance:
   :members: low_omega_constants, high_omega_constants

Peng Robinson Stryjek-Vera
--------------------------
.. autoclass:: PRSV
   :show-inheritance:
   :members: solve_T, a_alpha_and_derivatives_pure, a_alpha_pure

Peng Robinson Stryjek-Vera 2
----------------------------
.. autoclass:: PRSV2
   :show-inheritance:
   :members: solve_T, a_alpha_and_derivatives_pure, a_alpha_pure

Peng Robinson Twu (1995)
------------------------
.. autoclass:: TWUPR
   :show-inheritance:
   :members: a_alpha_and_derivatives_pure, a_alpha_pure

Peng Robinson Polynomial alpha Function
---------------------------------------
.. autoclass:: PRTranslatedPoly
   :show-inheritance:
   :members: a_alpha_and_derivatives_pure, a_alpha_pure



Volume Translated Peng-Robinson Family EOSs
===========================================

Peng Robinson Translated
------------------------
.. autoclass:: PRTranslated
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Peng Robinson Translated Twu (1991)
-----------------------------------
.. autoclass:: PRTranslatedTwu
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Peng Robinson Translated-Consistent
-----------------------------------
.. autoclass:: PRTranslatedConsistent
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Peng Robinson Translated (Pina-Martinez, Privat, and Jaubert Variant)
---------------------------------------------------------------------
.. autoclass:: PRTranslatedPPJP
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Soave-Redlich-Kwong Family EOSs
===============================

Standard SRK
------------
.. autoclass:: SRK
   :show-inheritance:
   :members: c1, c2, epsilon, Zc, a_alpha_and_derivatives_pure, a_alpha_pure, P_max_at_V, solve_T

Twu SRK (1995)
--------------
.. autoclass:: TWUSRK
   :show-inheritance:
   :members: a_alpha_and_derivatives_pure, a_alpha_pure

API SRK
-------
.. autoclass:: APISRK
   :show-inheritance:
   :members: a_alpha_and_derivatives_pure, a_alpha_pure, solve_T

SRK Translated
--------------
.. autoclass:: SRKTranslated
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

SRK Translated-Consistent
-------------------------
.. autoclass:: SRKTranslatedConsistent
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

SRK Translated (Pina-Martinez, Privat, and Jaubert Variant)
-----------------------------------------------------------
.. autoclass:: SRKTranslatedPPJP
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

MSRK Translated
---------------
.. autoclass:: MSRKTranslated
   :show-inheritance:
   :members: estimate_MN

Van der Waals Equations of State
================================
.. autoclass:: VDW
   :show-inheritance:
   :members: a_alpha_and_derivatives_pure, a_alpha_pure, solve_T, T_discriminant_zeros_analytical, P_discriminant_zeros_analytical, delta, epsilon, omega, Zc

Redlich-Kwong Equations of State
================================
.. autoclass:: RK
   :show-inheritance:
   :members: a_alpha_and_derivatives_pure, a_alpha_pure, solve_T, T_discriminant_zeros_analytical, epsilon, omega, Zc, c1, c2

Ideal Gas Equation of State
===========================
.. autoclass:: IG
   :show-inheritance:
   :members: volume_solutions, Zc, a, b, delta, epsilon, a_alpha_pure, a_alpha_and_derivatives_pure, solve_T


Lists of Equations of State
===========================
.. autodata:: eos_list
.. autodata:: eos_2P_list

Demonstrations of Concepts
==========================


Maximum Pressure at Constant Volume
-----------------------------------

Some equations of state show this behavior. At a liquid volume, if the
temperature is increased, the pressure should increase as well to create that
same volume. However in some cases this is not the case as can be demonstrated
for this hypothetical dodecane-like fluid:

.. plot:: plots/PR_maximum_pressure.py

Through experience, it is observed that this behavior is only shown for some
sets of critical constants. It was found that if the expression for
:math:`\frac{\partial P}{\partial T}_{V}` is set to zero, an analytical
expression can be determined for exactly what that maximum pressure is.
Some EOSs implement this function as `P_max_at_V`; those that don't, and fluids
where there is no maximum pressure, will have that method but it will return None.

Debug Plots to Understand EOSs
------------------------------
The :obj:`GCEOS.volume_errors` method shows the relative error in the volume
solution. `mpmath` is requried for this functionality. It is not likely there
is an error here but many problems have been found in the past.


.. plot:: plots/PRTC_volume_error.py

The :obj:`GCEOS.PT_surface_special` method shows some of the special curves of
the EOS.

.. plot:: plots/PRTC_PT_surface_special.py

The :obj:`GCEOS.a_alpha_plot` method shows the alpha function curve. The
following sample shows the SRK's default alpha function for methane.

.. plot:: plots/SRK_a_alpha.py

If this doesn't look healthy, that is because it is not. There are strict
thermodynamic consistency requirements that we know of today:

* The alpha function must be positive and continuous
* The first derivative must be negative and continuous
* The second derivative must be positive and continuous
* The third derivative must be negative

The first criterial and second criteria fail here.

There are two methods to review the saturation properties solution.
The more general way is to review saturation properties as a plot:

.. plot:: plots/SRK_H_dep.py

.. plot:: plots/SRK_fugacity.py

The second plot is more detailed, and is focused on the direct calculation of
vapor pressure without using an iterative solution. It shows the relative
error of the fit, which normally way below where it would present any issue -
only 10-100x more error than it is possible to get with floating point numbers
at all.

.. plot:: plots/SRK_Psat_error.py

'''


__all__ = ['GCEOS', 'PR', 'SRK', 'PR78', 'PRSV', 'PRSV2', 'VDW', 'RK',
'APISRK', 'TWUPR', 'TWUSRK', 'eos_list', 'eos_2P_list',
'IG', 'PRTranslatedPPJP', 'SRKTranslatedPPJP',
'PRTranslatedConsistent', 'SRKTranslatedConsistent', 'MSRKTranslated',
'SRKTranslated', 'PRTranslated', 'PRTranslatedCoqueletChapoyRichon',
'PRTranslatedTwu', 'PRTranslatedPoly',
'main_derivatives_and_departures',
                'main_derivatives_and_departures_VDW',
                'eos_lnphi'
]

from cmath import log as clog
from math import isinf, isnan, log1p, log10

from chemicals.flash_basic import Wilson_K_value
from chemicals.utils import hash_any_primitive, object_data
from fluids.constants import R, mmHg
from fluids.numerics import (
    NoSolutionError,
    bisect,
    brenth,
    catanh,
    chebder,
    chebval,
    derivative,
    exp,
    horner,
    inf,
    is_micropython,
    isclose,
    linspace,
    log,
    logspace,
    newton,
    polyder,
    roots_cubic,
    roots_quartic,
    secant,
    sqrt,
)
from fluids.numerics import numpy as np

from thermo.eos_alpha_functions import Mathias_Copeman_poly_a_alpha, Poly_a_alpha, Soave_1979_a_alpha, Twu91_a_alpha, TwuPR95_a_alpha, TwuSRK95_a_alpha
from thermo.eos_volume import volume_solutions_halley, volume_solutions_ideal, volume_solutions_mpmath, volume_solutions_mpmath_float, volume_solutions_NR
from thermo.serialize import JsonOptEncodable

R2 = R*R
R_2 = 0.5*R
R_inv = 1.0/R
R_inv2 = R_inv*R_inv


def main_derivatives_and_departures(T, P, V, b, delta, epsilon, a_alpha,
                                    da_alpha_dT, d2a_alpha_dT2):
    epsilon2 = epsilon + epsilon
    x0 = 1.0/(V - b)
    x1 = 1.0/(V*(V + delta) + epsilon)
    x3 = R*T
    x4 = x0*x0
    x5 = V + V + delta
    x6 = x1*x1
    x7 = a_alpha*x6
    x8 = P*V
    x9 = delta*delta
    x10 = x9 - epsilon2 - epsilon2
    try:
        x11 = 1.0/sqrt(x10)
    except:
        # Needed for ideal gas model
        x11 = 0.0
    x11_half = 0.5*x11

#    hard_input = x11*x5
#    arg2 = (hard_input + 1.0)/(hard_input - 1.0)
#    fancy = 0.25*log(arg2*arg2)
#    x12 = 2.*x11*fancy # Possible to use a catan, but then a complex division and sq root is needed too
    hard_input = x11*x5
    hard_term = catanh(hard_input).real # numba: delete
#    hard_term = 0.25*log1p(4.*hard_input/((1.0-hard_input)*(1.0-hard_input))) # numba: uncomment
    x12 = 2.*x11*hard_term # Possible to use a catan, but then a complex division and sq root is needed too
    x14 = 0.5*x5
    x15 = epsilon2*x11
    x16 = x11_half*x9
    x17 = x5*x6
    dP_dT = R*x0 - da_alpha_dT*x1
    dP_dV = x5*x7 - x3*x4
    d2P_dT2 = -d2a_alpha_dT2*x1

    d2P_dV2 = (x7 + x3*x4*x0 - a_alpha*x5*x17*x1)
    d2P_dV2 += d2P_dV2

    d2P_dTdV = da_alpha_dT*x17 - R*x4
    H_dep = x12*(T*da_alpha_dT - a_alpha) - x3 + x8

    t1 = (x3*x0/P)
    S_dep = -R*log(t1) + da_alpha_dT*x12  # Consider Real part of the log only via log(x**2)/2 = Re(log(x))
#        S_dep = -R_2*log(t1*t1) + da_alpha_dT*x12  # Consider Real part of the log only via log(x**2)/2 = Re(log(x))
    x18 = x16 - x15
    x19 = (x14 + x18)/(x14 - x18)
    Cv_dep = T*d2a_alpha_dT2*x11*(log(x19)) # Consider Real part of the log only via log(x**2)/2 = Re(log(x))
    return dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep


def main_derivatives_and_departures_VDW(T, P, V, b, delta, epsilon, a_alpha,
                                    da_alpha_dT, d2a_alpha_dT2):
    '''Re-implementation of derivatives and excess property calculations,
    as ZeroDivisionError errors occur with the general solution. The
    following derivation is the source of these formulas.

    >>> from sympy import *
    >>> P, T, V, R, b, a = symbols('P, T, V, R, b, a')
    >>> P_vdw = R*T/(V-b) - a/(V*V)
    >>> vdw = P_vdw - P
    >>>
    >>> dP_dT = diff(vdw, T)
    >>> dP_dV = diff(vdw, V)
    >>> d2P_dT2 = diff(vdw, T, 2)
    >>> d2P_dV2 = diff(vdw, V, 2)
    >>> d2P_dTdV = diff(vdw, T, V)
    >>> H_dep = integrate(T*dP_dT - P_vdw, (V, oo, V))
    >>> H_dep += P*V - R*T
    >>> S_dep = integrate(dP_dT - R/V, (V,oo,V))
    >>> S_dep += R*log(P*V/(R*T))
    >>> Cv_dep = T*integrate(d2P_dT2, (V,oo,V))
    >>>
    >>> dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep
    (R/(V - b), -R*T/(V - b)**2 + 2*a/V**3, 0, 2*(R*T/(V - b)**3 - 3*a/V**4), -R/(V - b)**2, P*V - R*T - a/V, R*(-log(V) + log(V - b)) + R*log(P*V/(R*T)), 0)
    '''
    V_inv = 1.0/V
    V_inv2 = V_inv*V_inv
    Vmb = V - b
    Vmb_inv = 1.0/Vmb
    dP_dT = R*Vmb_inv
    dP_dV = -R*T*Vmb_inv*Vmb_inv + 2.0*a_alpha*V_inv*V_inv2
    d2P_dT2 = 0.0
    d2P_dV2 = 2.0*(R*T*Vmb_inv*Vmb_inv*Vmb_inv - 3.0*a_alpha*V_inv2*V_inv2) # Causes issues at low T when V fourth power fails
    d2P_dTdV = -R*Vmb_inv*Vmb_inv
    H_dep = P*V - R*T - a_alpha*V_inv
    S_dep = R*(-log(V) + log(Vmb)) + R*log(P*V/(R*T))
    Cv_dep = 0.0
    return (dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep)


def eos_lnphi(T, P, V, b, delta, epsilon, a_alpha):
    r'''Calculate the log fugacity coefficient of the general cubic equation
    of state form.

    .. math::
        \ln \phi = \frac{P V}{R T} + \log{\left(V \right)} - \log{\left(\frac{P
        V}{R T} \right)} - \log{\left(V - b \right)} - 1 - \frac{2 a {\alpha}
        \operatorname{atanh}{\left(\frac{2 V}{\sqrt{\delta^{2} - 4 \epsilon}}
        + \frac{\delta}{\sqrt{\delta^{2} - 4 \epsilon}} \right)}}
        {R T \sqrt{\delta^{2} - 4 \epsilon}}

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    V : float
        Molar volume, [m^3/mol]
    b : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

    Returns
    -------
    lnphi : float
        Log fugacity coefficient, [-]

    Examples
    --------
    >>> eos_lnphi(299.0, 100000.0, 0.00013128, 0.000109389, 0.00021537, -1.1964711e-08, 3.8056296)
    -1.560560970726

    '''
    RT = R*T
    RT_inv = 1.0/RT
    x0 = 1.0/sqrt(delta*delta - 4.0*epsilon)

    arg = 2.0*V*x0 + delta*x0
    # fancy = catanh(arg).real

    fancy = 0.25*log1p(4.*arg/((1.0-arg)*(1.0-arg)))

    return (P*V*RT_inv + log(RT/(P*(V-b))) - 1.0
            - 2.0*a_alpha*fancy*RT_inv*x0)

def eos_G_dep(T, P, V, b, delta, epsilon, a_alpha):
    lnphi = eos_lnphi(T, P, V, b, delta, epsilon, a_alpha)
    return lnphi*R*T

class GCEOS:
    r'''Class for solving a generic Pressure-explicit three-parameter cubic
    equation of state. Does not implement any parameters itself; must be
    subclassed by an equation of state class which uses it. Works for mixtures
    or pure species for all properties except fugacity. All properties are
    derived with the CAS SymPy, not relying on any derivations previously
    published.

    .. math::
        P=\frac{RT}{V-b}-\frac{a\alpha(T)}{V^2 + \delta V + \epsilon}

    The main methods (in order they are called) are :obj:`GCEOS.solve`, :obj:`GCEOS.set_from_PT`,
    :obj:`GCEOS.volume_solutions`, and :obj:`GCEOS.set_properties_from_solution`.

    :obj:`GCEOS.solve` calls :obj:`GCEOS.check_sufficient_inputs`, which checks if two of `T`, `P`,
    and `V` were set. It then solves for the
    remaining variable. If `T` is missing, method :obj:`GCEOS.solve_T` is used; it is
    parameter specific, and so must be implemented in each specific EOS.
    If `P` is missing, it is directly calculated. If `V` is missing, it
    is calculated with the method :obj:`GCEOS.volume_solutions`. At this point, either
    three possible volumes or one user specified volume are known. The
    value of `a_alpha`, and its first and second temperature derivative are
    calculated with the EOS-specific method :obj:`GCEOS.a_alpha_and_derivatives`.

    If `V` is not provided, :obj:`GCEOS.volume_solutions` calculates the three
    possible molar volumes which are solutions to the EOS; in the single-phase
    region, only one solution is real and correct. In the two-phase region, all
    volumes are real, but only the largest and smallest solution are physically
    meaningful, with the largest being that of the gas and the smallest that of
    the liquid.

    :obj:`GCEOS.set_from_PT` is called to sort out the possible molar volumes. For the
    case of a user-specified `V`, the possibility of there existing another
    solution is ignored for speed. If there is only one real volume, the
    method :obj:`GCEOS.set_properties_from_solution` is called with it. If there are
    two real volumes, :obj:`GCEOS.set_properties_from_solution` is called once with each
    volume. The phase is returned by :obj:`GCEOS.set_properties_from_solution`, and the
    volumes is set to either :obj:`GCEOS.V_l` or :obj:`GCEOS.V_g` as appropriate.

    :obj:`GCEOS.set_properties_from_solution` is a large function which calculates all relevant
    partial derivatives and properties of the EOS. 17 derivatives and excess
    enthalpy and entropy are calculated first.
    Finally, it sets all these properties as attibutes for either
    the liquid or gas phase with the convention of adding on `_l` or `_g` to
    the variable names, respectively.


    Attributes
    ----------
    T : float
        Temperature of cubic EOS state, [K]
    P : float
        Pressure of cubic EOS state, [Pa]
    a : float
        `a` parameter of cubic EOS; formulas vary with the EOS, [Pa*m^6/mol^2]
    b : float
        `b` parameter of cubic EOS; formulas vary with the EOS, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
    da_alpha_dT : float
        Temperature derivative of :math:`a \alpha` calculated by EOS-specific
        method, [J^2/mol^2/Pa/K]
    d2a_alpha_dT2 : float
        Second temperature derivative of :math:`a \alpha` calculated by
        EOS-specific method, [J^2/mol^2/Pa/K**2]
    Zc : float
        Critical compressibility of cubic EOS state, [-]
    phase : str
        One of 'l', 'g', or 'l/g' to represent whether or not there is a
        liquid-like solution, vapor-like solution, or both available, [-]
    raw_volumes : list[(float, complex), 3]
        Calculated molar volumes from the volume solver; depending on the state
        and selected volume solver, imaginary volumes may be represented by 0
        or -1j to save the time of actually calculating them, [m^3/mol]
    V_l : float
        Liquid phase molar volume, [m^3/mol]
    V_g : float
        Vapor phase molar volume, [m^3/mol]
    V : float or None
        Molar volume specified as input; otherwise None, [m^3/mol]
    Z_l : float
        Liquid phase compressibility, [-]
    Z_g : float
        Vapor phase compressibility, [-]
    PIP_l : float
        Liquid phase phase identification parameter, [-]
    PIP_g : float
        Vapor phase phase identification parameter, [-]
    dP_dT_l : float
        Liquid phase temperature derivative of pressure at constant volume,
        [Pa/K].

        .. math::
            \left(\frac{\partial P}{\partial T}\right)_V = \frac{R}{V - b}
            - \frac{a \frac{d \alpha{\left (T \right )}}{d T}}{V^{2} + V \delta
            + \epsilon}
    dP_dT_g : float
        Vapor phase temperature derivative of pressure at constant volume,
        [Pa/K].

        .. math::
            \left(\frac{\partial P}{\partial T}\right)_V = \frac{R}{V - b}
            - \frac{a \frac{d \alpha{\left (T \right )}}{d T}}{V^{2} + V \delta
            + \epsilon}
    dP_dV_l : float
        Liquid phase volume derivative of pressure at constant temperature,
        [Pa*mol/m^3].

        .. math::
            \left(\frac{\partial P}{\partial V}\right)_T = - \frac{R T}{\left(
            V - b\right)^{2}} - \frac{a \left(- 2 V - \delta\right) \alpha{
            \left (T \right )}}{\left(V^{2} + V \delta + \epsilon\right)^{2}}
    dP_dV_g : float
        Gas phase volume derivative of pressure at constant temperature,
        [Pa*mol/m^3].

        .. math::
            \left(\frac{\partial P}{\partial V}\right)_T = - \frac{R T}{\left(
            V - b\right)^{2}} - \frac{a \left(- 2 V - \delta\right) \alpha{
            \left (T \right )}}{\left(V^{2} + V \delta + \epsilon\right)^{2}}
    dV_dT_l : float
        Liquid phase temperature derivative of volume at constant pressure,
        [m^3/(mol*K)].

        .. math::
            \left(\frac{\partial V}{\partial T}\right)_P =-\frac{
            \left(\frac{\partial P}{\partial T}\right)_V}{
            \left(\frac{\partial P}{\partial V}\right)_T}
    dV_dT_g : float
        Gas phase temperature derivative of volume at constant pressure,
        [m^3/(mol*K)].

        .. math::
            \left(\frac{\partial V}{\partial T}\right)_P =-\frac{
            \left(\frac{\partial P}{\partial T}\right)_V}{
            \left(\frac{\partial P}{\partial V}\right)_T}
    dV_dP_l : float
        Liquid phase pressure derivative of volume at constant temperature,
        [m^3/(mol*Pa)].

        .. math::
            \left(\frac{\partial V}{\partial P}\right)_T =-\frac{
            \left(\frac{\partial V}{\partial T}\right)_P}{
            \left(\frac{\partial P}{\partial T}\right)_V}
    dV_dP_g : float
        Gas phase pressure derivative of volume at constant temperature,
        [m^3/(mol*Pa)].

        .. math::
            \left(\frac{\partial V}{\partial P}\right)_T =-\frac{
            \left(\frac{\partial V}{\partial T}\right)_P}{
            \left(\frac{\partial P}{\partial T}\right)_V}
    dT_dV_l : float
        Liquid phase volume derivative of temperature at constant pressure,
        [K*mol/m^3].

        .. math::
            \left(\frac{\partial T}{\partial V}\right)_P = \frac{1}
            {\left(\frac{\partial V}{\partial T}\right)_P}
    dT_dV_g : float
        Gas phase volume derivative of temperature at constant pressure,
        [K*mol/m^3]. See :obj:`GCEOS.set_properties_from_solution` for
        the formula.
    dT_dP_l : float
        Liquid phase pressure derivative of temperature at constant volume,
        [K/Pa].

        .. math::
            \left(\frac{\partial T}{\partial P}\right)_V = \frac{1}
            {\left(\frac{\partial P}{\partial T}\right)_V}
    dT_dP_g : float
        Gas phase pressure derivative of temperature at constant volume,
        [K/Pa].

        .. math::
            \left(\frac{\partial T}{\partial P}\right)_V = \frac{1}
            {\left(\frac{\partial P}{\partial T}\right)_V}
    d2P_dT2_l : float
        Liquid phase second derivative of pressure with respect to temperature
        at constant volume, [Pa/K^2].

        .. math::
            \left(\frac{\partial^2  P}{\partial T^2}\right)_V =  - \frac{a
            \frac{d^{2} \alpha{\left (T \right )}}{d T^{2}}}{V^{2} + V \delta
            + \epsilon}
    d2P_dT2_g : float
        Gas phase second derivative of pressure with respect to temperature
        at constant volume, [Pa/K^2].

        .. math::
            \left(\frac{\partial^2  P}{\partial T^2}\right)_V =  - \frac{a
            \frac{d^{2} \alpha{\left (T \right )}}{d T^{2}}}{V^{2} + V \delta
            + \epsilon}
    d2P_dV2_l : float
        Liquid phase second derivative of pressure with respect to volume
        at constant temperature, [Pa*mol^2/m^6].

        .. math::
            \left(\frac{\partial^2  P}{\partial V^2}\right)_T = 2 \left(\frac{
            R T}{\left(V - b\right)^{3}} - \frac{a \left(2 V + \delta\right)^{
            2} \alpha{\left (T \right )}}{\left(V^{2} + V \delta + \epsilon
            \right)^{3}} + \frac{a \alpha{\left (T \right )}}{\left(V^{2} + V
            \delta + \epsilon\right)^{2}}\right)
    d2P_dTdV_l : float
        Liquid phase second derivative of pressure with respect to volume
        and then temperature, [Pa*mol/(K*m^3)].

        .. math::
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) = - \frac{
            R}{\left(V - b\right)^{2}} + \frac{a \left(2 V + \delta\right)
            \frac{d \alpha{\left (T \right )}}{d T}}{\left(V^{2} + V \delta
            + \epsilon\right)^{2}}
    d2P_dTdV_g : float
        Gas phase second derivative of pressure with respect to volume
        and then temperature, [Pa*mol/(K*m^3)].

        .. math::
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) = - \frac{
            R}{\left(V - b\right)^{2}} + \frac{a \left(2 V + \delta\right)
            \frac{d \alpha{\left (T \right )}}{d T}}{\left(V^{2} + V \delta
            + \epsilon\right)^{2}}
    H_dep_l : float
        Liquid phase departure enthalpy, [J/mol]. See
        :obj:`GCEOS.set_properties_from_solution` for the formula.
    H_dep_g : float
        Gas phase departure enthalpy, [J/mol]. See
        :obj:`GCEOS.set_properties_from_solution` for the formula.
    S_dep_l : float
        Liquid phase departure entropy, [J/(mol*K)]. See
        :obj:`GCEOS.set_properties_from_solution` for the formula.
    S_dep_g : float
        Gas phase departure entropy, [J/(mol*K)]. See
        :obj:`GCEOS.set_properties_from_solution` for the formula.
    G_dep_l : float
        Liquid phase departure Gibbs energy, [J/mol].

        .. math::
            G_{dep} = H_{dep} - T S_{dep}
    G_dep_g : float
        Gas phase departure Gibbs energy, [J/mol].

        .. math::
            G_{dep} = H_{dep} - T S_{dep}
    Cp_dep_l : float
        Liquid phase departure heat capacity, [J/(mol*K)]

        .. math::
            C_{p, dep} = (C_p-C_v)_{\text{from EOS}} + C_{v, dep} - R
    Cp_dep_g : float
        Gas phase departure heat capacity, [J/(mol*K)]

        .. math::
            C_{p, dep} = (C_p-C_v)_{\text{from EOS}} + C_{v, dep} - R
    Cv_dep_l : float
        Liquid phase departure constant volume heat capacity, [J/(mol*K)].
        See :obj:`GCEOS.set_properties_from_solution` for
        the formula.

    Cv_dep_g : float
        Gas phase departure constant volume heat capacity, [J/(mol*K)].
        See :obj:`GCEOS.set_properties_from_solution` for
        the formula.
    c1 : float
        Full value of the constant in the `a` parameter, set in some EOSs, [-]
    c2 : float
        Full value of the constant in the `b` parameter, set in some EOSs, [-]

    A_dep_g
    A_dep_l
    beta_g
    beta_l
    Cp_minus_Cv_g
    Cp_minus_Cv_l
    d2a_alpha_dTdP_g_V
    d2a_alpha_dTdP_l_V
    d2H_dep_dT2_g
    d2H_dep_dT2_g_P
    d2H_dep_dT2_g_V
    d2H_dep_dT2_l
    d2H_dep_dT2_l_P
    d2H_dep_dT2_l_V
    d2H_dep_dTdP_g
    d2H_dep_dTdP_l
    d2P_drho2_g
    d2P_drho2_l
    d2P_dT2_PV_g
    d2P_dT2_PV_l
    d2P_dTdP_g
    d2P_dTdP_l
    d2P_dTdrho_g
    d2P_dTdrho_l
    d2P_dVdP_g
    d2P_dVdP_l
    d2P_dVdT_g
    d2P_dVdT_l
    d2P_dVdT_TP_g
    d2P_dVdT_TP_l
    d2rho_dP2_g
    d2rho_dP2_l
    d2rho_dPdT_g
    d2rho_dPdT_l
    d2rho_dT2_g
    d2rho_dT2_l
    d2S_dep_dT2_g
    d2S_dep_dT2_g_V
    d2S_dep_dT2_l
    d2S_dep_dT2_l_V
    d2S_dep_dTdP_g
    d2S_dep_dTdP_l
    d2T_dP2_g
    d2T_dP2_l
    d2T_dPdrho_g
    d2T_dPdrho_l
    d2T_dPdV_g
    d2T_dPdV_l
    d2T_drho2_g
    d2T_drho2_l
    d2T_dV2_g
    d2T_dV2_l
    d2T_dVdP_g
    d2T_dVdP_l
    d2V_dP2_g
    d2V_dP2_l
    d2V_dPdT_g
    d2V_dPdT_l
    d2V_dT2_g
    d2V_dT2_l
    d2V_dTdP_g
    d2V_dTdP_l
    d3a_alpha_dT3
    da_alpha_dP_g_V
    da_alpha_dP_l_V
    dbeta_dP_g
    dbeta_dP_l
    dbeta_dT_g
    dbeta_dT_l
    dfugacity_dP_g
    dfugacity_dP_l
    dfugacity_dT_g
    dfugacity_dT_l
    dH_dep_dP_g
    dH_dep_dP_g_V
    dH_dep_dP_l
    dH_dep_dP_l_V
    dH_dep_dT_g
    dH_dep_dT_g_V
    dH_dep_dT_l
    dH_dep_dT_l_V
    dH_dep_dV_g_P
    dH_dep_dV_g_T
    dH_dep_dV_l_P
    dH_dep_dV_l_T
    dP_drho_g
    dP_drho_l
    dphi_dP_g
    dphi_dP_l
    dphi_dT_g
    dphi_dT_l
    drho_dP_g
    drho_dP_l
    drho_dT_g
    drho_dT_l
    dS_dep_dP_g
    dS_dep_dP_g_V
    dS_dep_dP_l
    dS_dep_dP_l_V
    dS_dep_dT_g
    dS_dep_dT_g_V
    dS_dep_dT_l
    dS_dep_dT_l_V
    dS_dep_dV_g_P
    dS_dep_dV_g_T
    dS_dep_dV_l_P
    dS_dep_dV_l_T
    dT_drho_g
    dT_drho_l
    dZ_dP_g
    dZ_dP_l
    dZ_dT_g
    dZ_dT_l
    fugacity_g
    fugacity_l
    kappa_g
    kappa_l
    lnphi_g
    lnphi_l
    more_stable_phase
    mpmath_volume_ratios
    mpmath_volumes
    mpmath_volumes_float
    phi_g
    phi_l
    rho_g
    rho_l
    sorted_volumes
    state_specs
    U_dep_g
    U_dep_l
    Vc
    V_dep_g
    V_dep_l
    V_g_mpmath
    V_l_mpmath
    '''

    # Slots does not help performance in either implementation
    kwargs = {}
    """Dictionary which holds input parameters to an EOS which are non-standard;
    this excludes `T`, `P`, `V`, `omega`, `Tc`, `Pc`, `Vc` but includes EOS
    specific parameters like `S1` and `alpha_coeffs`.
    """

    N = 1
    """The number of components in the EOS"""
    vectorized = False

    multicomponent = False
    """Whether or not the EOS is multicomponent or not"""
    _P_zero_l_cheb_coeffs = None
    P_zero_l_cheb_limits = (0.0, 0.0)
    _P_zero_g_cheb_coeffs = None
    P_zero_g_cheb_limits = (0.0, 0.0)
    Psat_cheb_range = (0.0, 0.0)

    main_derivatives_and_departures = staticmethod(main_derivatives_and_departures)

    c1 = None
    """Parameter used by some equations of state in the `a` calculation"""
    c2 = None
    """Parameter used by some equations of state in the `b` calculation"""

    nonstate_constants = ('Tc', 'Pc', 'omega', 'kwargs', 'a', 'b', 'delta', 'epsilon')
    kwargs_keys = tuple()

    if not is_micropython:
        def __init_subclass__(cls):
            cls.__full_path__ = f"{cls.__module__}.{cls.__qualname__}"
    else:
        __full_path__ = None

    def state_hash(self):
        r'''Basic method to calculate a hash of the state of the model and its
        model parameters.

        Note that the hashes should only be compared on the same system running
        in the same process!

        Returns
        -------
        state_hash : int
            Hash of the object's model parameters and state, [-]
        '''
        if self.multicomponent:
            comp = self.zs
        else:
            comp = 0
        return hash_any_primitive((self.model_hash(), self.T, self.P, self.V, comp))

    __hash__ = state_hash

    def model_hash(self):
        r'''Basic method to calculate a hash of the non-state parts of the model
        This is useful for comparing to models to
        determine if they are the same, i.e. in a VLL flash it is important to
        know if both liquids have the same model.

        Note that the hashes should only be compared on the same system running
        in the same process!

        Returns
        -------
        model_hash : int
            Hash of the object's model parameters, [-]
        '''
        try:
            return self._model_hash
        except AttributeError:
            pass
        h = hash(self.__class__.__name__)
        for s in self.nonstate_constants:
            try:
                h = hash((h, s, hash_any_primitive(getattr(self, s))))
            except AttributeError:
                pass
        self._model_hash = h
        return h

    def exact_hash(self):
        r'''Method to calculate and return a hash representing the exact state
        of the object.

        Returns
        -------
        hash : int
            Hash of the object, [-]
        '''
        d = object_data(self)
        ans = hash_any_primitive((self.__class__.__name__, d))
        return ans

    def __eq__(self, other):
        return self.__hash__() == hash(other)


    @property
    def state_specs(self):
        '''Convenience method to return the two specified state specs (`T`,
        `P`, or `V`) as a dictionary.

        Examples
        --------
        >>> PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=500.0, V=1.0).state_specs
        {'T': 500.0, 'V': 1.0}
        '''
        d = {}
        if hasattr(self, 'no_T_spec') and self.no_T_spec:
            d['P'] = self.P
            d['V'] = self.V
        elif self.V is not None:
            d['T'] = self.T
            d['V'] = self.V
        else:
            d['T'] = self.T
            d['P'] = self.P
        return d

    def __repr__(self):
        '''Create a string representation of the EOS - by default, include
        all parameters so as to make it easy to construct new instances from
        states. Includes the two specified state variables, `Tc`, `Pc`, `omega`
        and any `kwargs`.

        Returns
        -------
        recreation : str
            String which is valid Python and recreates the current state of
            the object if ran, [-]

        Examples
        --------
        >>> eos = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400.0, P=1e6)
        >>> eos
        PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400.0, P=1000000.0)
        '''
        s = f'{self.__class__.__name__}(Tc={self.Tc!r}, Pc={self.Pc!r}, omega={self.omega!r}, '
        for k, v in self.kwargs.items():
            s += f'{k}={v}, '

        if hasattr(self, 'no_T_spec') and self.no_T_spec:
            s += f'P={self.P!r}, V={self.V!r}'
        elif self.V is not None:
            s += f'T={self.T!r}, V={self.V!r}'
        else:
            s += f'T={self.T!r}, P={self.P!r}'
        s += ')'
        return s

    def as_json(self, cache=None, option=0):
        r'''Method to create a JSON-friendly serialization of the eos
        which can be stored, and reloaded later.

        Returns
        -------
        json_repr : dict
            JSON-friendly representation, [-]

        Notes
        -----

        Examples
        --------
        >>> import json
        >>> eos = MSRKTranslated(Tc=507.6, Pc=3025000, omega=0.2975, c=22.0561E-6, M=0.7446, N=0.2476, T=250., P=1E6)
        >>> assert eos == MSRKTranslated.from_json(json.loads(json.dumps(eos.as_json())))
        '''
        return JsonOptEncodable.as_json(self, cache, option)

    json_version = 1
    obj_references = []
    non_json_attributes = ['kwargs', 'one_minus_kijs', '_model_hash']

    @classmethod
    def from_json(cls, json_repr, cache=None):
        r'''Method to create a eos from a JSON
        serialization of another eos.

        Parameters
        ----------
        json_repr : dict
            JSON-friendly representation, [-]

        Returns
        -------
        eos : :obj:`GCEOS`
            Newly created object from the json serialization, [-]

        Notes
        -----
        It is important that the input string be in the same format as that
        created by :obj:`GCEOS.as_json`.

        Examples
        --------
        >>> eos = MSRKTranslated(Tc=507.6, Pc=3025000, omega=0.2975, c=22.0561E-6, M=0.7446, N=0.2476, T=250., P=1E6)
        >>> string = eos.as_json()
        >>> new_eos = GCEOS.from_json(string)
        >>> assert eos.__dict__ == new_eos.__dict__
        '''
        return JsonOptEncodable.from_json(json_repr, cache)

    def _custom_from_json(self, *args):
        eos_name = self.__full_path__
        eos = eos_full_path_dict[eos_name]
        try:
            self.raw_volumes = tuple(self.raw_volumes)
        except:
            pass
        try:
            self.alpha_coeffs = tuple(self.alpha_coeffs)
        except:
            pass
        if eos.kwargs_keys:
            self.kwargs = {k: getattr(self, k) for k in eos.kwargs_keys}
            try:
                self.kwargs['alpha_coeffs'] = tuple(self.kwargs['alpha_coeffs'])
            except:
                pass

    def check_sufficient_inputs(self):
        '''Method to an exception if none of the pairs (T, P), (T, V), or
        (P, V) are given.
        '''
        if not ((self.T is not None and self.P is not None) or
                (self.T is not None and self.V is not None) or
                (self.P is not None and self.V is not None)):
            raise ValueError('Either T and P, or T and V, or P and V are required')


    def solve(self, pure_a_alphas=True, only_l=False, only_g=False, full_alphas=True):
        '''First EOS-generic method; should be called by all specific EOSs.
        For solving for `T`, the EOS must provide the method `solve_T`.
        For all cases, the EOS must provide `a_alpha_and_derivatives`.
        Calls `set_from_PT` once done.
        '''
#        self.check_sufficient_inputs()

        if self.V is not None:
            V = self.V
            if self.P is not None:
                solution = 'g' if (only_g and not only_l) else ('l' if only_l else None)
                self.T = self.solve_T(self.P, V, solution=solution)
                self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2 = self.a_alpha_and_derivatives(self.T, pure_a_alphas=pure_a_alphas)
            elif self.T is not None:
                self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2 = self.a_alpha_and_derivatives(self.T, pure_a_alphas=pure_a_alphas)

                # Tested to change the result at the 7th decimal once
#                V_r3 = V**(1.0/3.0)
#                T, b, a_alpha, delta, epsilon = self.T, self.b, self.a_alpha, self.delta, self.epsilon
#                P = R*T/(V-b) - a_alpha/((V_r3*V_r3)*(V_r3*(V+delta)) + epsilon)
#
#                for _ in range(10):
#                    err = -T + (P*V**3 - P*V**2*b + P*V**2*delta - P*V*b*delta + P*V*epsilon - P*b*epsilon + V*a_alpha - a_alpha*b)/(R*(V**2 + V*delta + epsilon))
#                    derr = (V**3 - V**2*b + V**2*delta - V*b*delta + V*epsilon - b*epsilon)/(R*(V**2 + V*delta + epsilon))
#                    P = P - err/derr
#                self.P = P
                # Equation re-aranged to hopefully solve better

                # Allow mpf multiple precision volume for flash initialization
                # DO NOT TAKE OUT FLOAT CONVERSION!
                T = self.T
                if not isinstance(V, (float, int)):
                    import mpmath as mp
                    # mp.mp.dps = 50 # Do not need more decimal places than needed
                    # Need to complete the calculation with the RT term having higher precision as well
                    T = mp.mpf(T)
                self.P = float(R*T/(V-self.b) - self.a_alpha/(V*V + self.delta*V + self.epsilon))
                if self.P <= 0.0:
                    raise ValueError(f"TV inputs result in negative pressure of {self.P:f} Pa")
#                self.P = R*self.T/(V-self.b) - self.a_alpha/(V*(V + self.delta) + self.epsilon)
            else:
                raise ValueError("Two specs are required")
            Vs = [V, 1.0j, 1.0j]
        elif self.T is None or self.P is None:
            raise ValueError("Two specs are required")
        else:
            if full_alphas:
                self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2 = self.a_alpha_and_derivatives(self.T, pure_a_alphas=pure_a_alphas)
            else:
                self.a_alpha = self.a_alpha_and_derivatives(self.T, full=False, pure_a_alphas=pure_a_alphas)
                self.da_alpha_dT, self.d2a_alpha_dT2 = -5e-3, 1.5e-5
            self.raw_volumes = Vs = self.volume_solutions(self.T, self.P, self.b, self.delta, self.epsilon, self.a_alpha)
        self.set_from_PT(Vs, only_l=only_l, only_g=only_g)

    def resolve_full_alphas(self):
        '''Generic method to resolve the eos with fully calculated alpha
        derviatives. Re-calculates properties with the new alpha derivatives
        for any previously solved roots.
        '''
        self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2 = self.a_alpha_and_derivatives(self.T, full=True, pure_a_alphas=False)
        self.set_from_PT(self.raw_volumes, only_l=hasattr(self, 'V_l'), only_g=hasattr(self, 'V_g'))

    def solve_missing_volumes(self):
        r'''Generic method to ensure both volumes, if solutions are physical,
        have calculated properties. This effectively un-does the optimization
        of the `only_l` and `only_g` keywords.
        '''
        if self.phase == 'l/g':
            try:
                self.V_l
            except:
                self.set_from_PT(self.raw_volumes, only_l=True, only_g=False)
            try:
                self.V_g
            except:
                self.set_from_PT(self.raw_volumes, only_l=False, only_g=True)


    def set_from_PT(self, Vs, only_l=False, only_g=False):
        r'''Counts the number of real volumes in `Vs`, and determines what to do.
        If there is only one real volume, the method
        `set_properties_from_solution` is called with it. If there are
        two real volumes, `set_properties_from_solution` is called once with
        each volume. The phase is returned by `set_properties_from_solution`,
        and the volumes is set to either `V_l` or `V_g` as appropriate.

        Parameters
        ----------
        Vs : list[float]
            Three possible molar volumes, [m^3/mol]
        only_l : bool
            When true, if there is a liquid and a vapor root, only the liquid
            root (and properties) will be set.
        only_g : bool
            When true, if there is a liquid and a vapor root, only the vapor
            root (and properties) will be set.

        Notes
        -----
        An optimization attempt was made to remove min() and max() from this
        function; that is indeed possible, but the check for handling if there
        are two or three roots makes it not worth it.
        '''
#        good_roots = [i.real for i in Vs if i.imag == 0.0 and i.real > 0.0]
#        good_root_count = len(good_roots)
            # All roots will have some imaginary component; ignore them if > 1E-9 (when using a solver that does not strip them)
        b = self.b
#        good_roots = [i.real for i in Vs if (i.real ==0 or abs(i.imag/i.real) < 1E-12) and i.real > 0.0]
        # good_roots = [i.real for i in Vs if (i.real > b and
        #             (i.imag == 0.0 or abs(i.imag/i.real) < 1E-12))]

        Vmin = 1e100
        Vmax = -1e100
        good_root_count = 0
        for V in Vs:
            if V.real == 0.0 or V.imag == 0.0 or abs(V.imag/V.real) < 1E-12:
                V = V.real
                if V <= b:
                    continue
                good_root_count += 1
                if V < Vmin:
                    Vmin = V
                if V > Vmax:
                    Vmax = V

        # Counter for the case of testing volume solutions that don't work
#        good_roots = [i.real for i in Vs if (i.real > 0.0 and (i.real == 0.0 or abs(i.imag) < 1E-9))]
        # good_root_count = len(good_roots)

        if good_root_count == 1 or Vmin == Vmax:
            V = Vmin if Vmin != 1e100 else Vmax
            self.phase = self.set_properties_from_solution(self.T, self.P,
                                                           V, b,
                                                           self.delta, self.epsilon,
                                                           self.a_alpha, self.da_alpha_dT,
                                                           self.d2a_alpha_dT2)

            if self.N == 1 and (
                    (self.multicomponent and (self.Tcs[0] == self.T and self.Pcs[0] == self.P))
                    or (not self.multicomponent and self.Tc == self.T and self.Pc == self.P)):
                # Do not have any tests for this - not good!

                force_l = not self.phase == 'l'
                force_g = not self.phase == 'g'
                V = Vmin if Vmin != 1e100 else Vmax
                self.set_properties_from_solution(self.T, self.P,
                                                  V, b,
                                                  self.delta, self.epsilon,
                                                  self.a_alpha, self.da_alpha_dT,
                                                  self.d2a_alpha_dT2,
                                                  force_l=force_l,
                                                  force_g=force_g)
                self.phase = 'l/g'
        elif good_root_count > 1:
            if not only_g:
                self.set_properties_from_solution(self.T, self.P, Vmin, b,
                                                   self.delta, self.epsilon,
                                                   self.a_alpha, self.da_alpha_dT,
                                                   self.d2a_alpha_dT2,
                                                   force_l=True)
            if not only_l:
                self.set_properties_from_solution(self.T, self.P, Vmax, b,
                                                   self.delta, self.epsilon,
                                                   self.a_alpha, self.da_alpha_dT,
                                                   self.d2a_alpha_dT2, force_g=True)
            self.phase = 'l/g'
        else:
            # Even in the case of three real roots, it is still the min/max that make sense
            print([self.T, self.P, b, self.delta, self.epsilon, self.a_alpha, 'coordinates of failure'])
            if self.multicomponent:
                extra = f', zs is {self.zs}'
            else:
                extra = ''
            raise ValueError(f'No acceptable roots were found; the roots are {Vs!s}, T is {self.T!s} K, P is {self.P!s} Pa, a_alpha is {[self.a_alpha]!s}, b is {[self.b]!s}{extra}')


    def set_properties_from_solution(self, T, P, V, b, delta, epsilon, a_alpha,
                                     da_alpha_dT, d2a_alpha_dT2,
                                     force_l=False, force_g=False):
        r'''Sets all interesting properties which can be calculated from an
        EOS alone. Determines which phase the fluid is on its own; for details,
        see `phase_identification_parameter`.

        The list of properties set is as follows, with all properties suffixed
        with '_l' or '_g'.

        dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP, d2P_dT2, d2P_dV2, d2V_dT2,
        d2V_dP2, d2T_dV2, d2T_dP2, d2V_dPdT, d2P_dTdV, d2T_dPdV, H_dep, S_dep,
        G_dep and PIP.

        Parameters
        ----------
        T : float
            Temperature, [K]
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        b : float
            Coefficient calculated by EOS-specific method, [m^3/mol]
        delta : float
            Coefficient calculated by EOS-specific method, [m^3/mol]
        epsilon : float
            Coefficient calculated by EOS-specific method, [m^6/mol^2]
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K**2]

        Returns
        -------
        phase : str
            Either 'l' or 'g'

        Notes
        -----
        The individual formulas for the derivatives and excess properties are
        as follows. For definitions of `beta`, see `isobaric_expansion`;
        for `kappa`, see isothermal_compressibility; for `Cp_minus_Cv`, see
        `Cp_minus_Cv`; for `phase_identification_parameter`, see
        `phase_identification_parameter`.

        First derivatives; in part using the Triple Product Rule [2]_, [3]_:

        .. math::
            \left(\frac{\partial P}{\partial T}\right)_V = \frac{R}{V - b}
            - \frac{a \frac{d \alpha{\left (T \right )}}{d T}}{V^{2} + V \delta
            + \epsilon}

        .. math::
            \left(\frac{\partial P}{\partial V}\right)_T = - \frac{R T}{\left(
            V - b\right)^{2}} - \frac{a \left(- 2 V - \delta\right) \alpha{
            \left (T \right )}}{\left(V^{2} + V \delta + \epsilon\right)^{2}}

        .. math::
            \left(\frac{\partial V}{\partial T}\right)_P =-\frac{
            \left(\frac{\partial P}{\partial T}\right)_V}{
            \left(\frac{\partial P}{\partial V}\right)_T}

        .. math::
            \left(\frac{\partial V}{\partial P}\right)_T =-\frac{
            \left(\frac{\partial V}{\partial T}\right)_P}{
            \left(\frac{\partial P}{\partial T}\right)_V}

        .. math::
            \left(\frac{\partial T}{\partial V}\right)_P = \frac{1}
            {\left(\frac{\partial V}{\partial T}\right)_P}

        .. math::
            \left(\frac{\partial T}{\partial P}\right)_V = \frac{1}
            {\left(\frac{\partial P}{\partial T}\right)_V}

        Second derivatives with respect to one variable; those of `T` and `V`
        use identities shown in [1]_ and verified numerically:

        .. math::
            \left(\frac{\partial^2  P}{\partial T^2}\right)_V =  - \frac{a
            \frac{d^{2} \alpha{\left (T \right )}}{d T^{2}}}{V^{2} + V \delta
            + \epsilon}

        .. math::
            \left(\frac{\partial^2  P}{\partial V^2}\right)_T = 2 \left(\frac{
            R T}{\left(V - b\right)^{3}} - \frac{a \left(2 V + \delta\right)^{
            2} \alpha{\left (T \right )}}{\left(V^{2} + V \delta + \epsilon
            \right)^{3}} + \frac{a \alpha{\left (T \right )}}{\left(V^{2} + V
            \delta + \epsilon\right)^{2}}\right)

        Second derivatives with respect to the other two variables; those of
        `T` and `V` use identities shown in [1]_ and verified numerically:

        .. math::
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) = - \frac{
            R}{\left(V - b\right)^{2}} + \frac{a \left(2 V + \delta\right)
            \frac{d \alpha{\left (T \right )}}{d T}}{\left(V^{2} + V \delta
            + \epsilon\right)^{2}}

        Excess properties

        .. math::
            H_{dep} = \int_{\infty}^V \left[T\frac{\partial P}{\partial T}_V
            - P\right]dV + PV - RT= P V - R T + \frac{2}{\sqrt{
            \delta^{2} - 4 \epsilon}} \left(T a \frac{d \alpha{\left (T \right
            )}}{d T}  - a \alpha{\left (T \right )}\right) \operatorname{atanh}
            {\left (\frac{2 V + \delta}{\sqrt{\delta^{2} - 4 \epsilon}}
            \right)}

        .. math::
            S_{dep} = \int_{\infty}^V\left[\frac{\partial P}{\partial T}
            - \frac{R}{V}\right] dV + R\ln\frac{PV}{RT} = - R \ln{\left (V
            \right )} + R \ln{\left (\frac{P V}{R T} \right )} + R \ln{\left
            (V - b \right )} + \frac{2 a \frac{d\alpha{\left (T \right )}}{d T}
            }{\sqrt{\delta^{2} - 4 \epsilon}} \operatorname{atanh}{\left (\frac
            {2 V + \delta}{\sqrt{\delta^{2} - 4 \epsilon}} \right )}

        .. math::
            G_{dep} = H_{dep} - T S_{dep}

        .. math::
            C_{v, dep} = T\int_\infty^V \left(\frac{\partial^2 P}{\partial
            T^2}\right) dV = - T a \left(\sqrt{\frac{1}{\delta^{2} - 4
            \epsilon}} \ln{\left (V - \frac{\delta^{2}}{2} \sqrt{\frac{1}{
            \delta^{2} - 4 \epsilon}} + \frac{\delta}{2} + 2 \epsilon \sqrt{
            \frac{1}{\delta^{2} - 4 \epsilon}} \right )} - \sqrt{\frac{1}{
            \delta^{2} - 4 \epsilon}} \ln{\left (V + \frac{\delta^{2}}{2}
            \sqrt{\frac{1}{\delta^{2} - 4 \epsilon}} + \frac{\delta}{2}
            - 2 \epsilon \sqrt{\frac{1}{\delta^{2} - 4 \epsilon}} \right )}
            \right) \frac{d^{2} \alpha{\left (T \right )} }{d T^{2}}

        .. math::
            C_{p, dep} = (C_p-C_v)_{\text{from EOS}} + C_{v, dep} - R


        References
        ----------
        .. [1] Thorade, Matthis, and Ali Saadat. "Partial Derivatives of
           Thermodynamic State Properties for Dynamic Simulation."
           Environmental Earth Sciences 70, no. 8 (April 10, 2013): 3497-3503.
           doi:10.1007/s12665-013-2394-z.
        .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th
           edition. New York: McGraw-Hill Professional, 2000.
        .. [3] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
           Butterworth-Heinemann, 1985.
        '''
        dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep = (
        self.main_derivatives_and_departures(T, P, V, b, delta, epsilon,
                                             a_alpha, da_alpha_dT,
                                             d2a_alpha_dT2))
        try:
            dV_dP = 1.0/dP_dV
        except:
            dV_dP = inf
        dT_dP = 1./dP_dT

        dV_dT = -dP_dT*dV_dP
        dT_dV = 1./dV_dT

        Z = P*V*R_inv/T
        Cp_dep = T*dP_dT*dV_dT + Cv_dep - R
        G_dep = H_dep - T*S_dep
        PIP = V*(d2P_dTdV*dT_dP - d2P_dV2*dV_dP) # phase_identification_parameter(V, dP_dT, dP_dV, d2P_dV2, d2P_dTdV)
         # 1 + 1e-14 - allow a few dozen unums of toleranve to keep ideal gas model a gas
        if force_l or (not force_g and PIP > 1.00000000000001):
            (self.V_l, self.Z_l, self.PIP_l, self.dP_dT_l, self.dP_dV_l,
             self.dV_dT_l, self.dV_dP_l, self.dT_dV_l, self.dT_dP_l,
             self.d2P_dT2_l, self.d2P_dV2_l, self.d2P_dTdV_l, self.H_dep_l,
             self.S_dep_l, self.G_dep_l, self.Cp_dep_l, self.Cv_dep_l) = (
                     V, Z, PIP, dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP,
                     d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, G_dep, Cp_dep,
                     Cv_dep)
            return 'l'
        else:
            (self.V_g, self.Z_g, self.PIP_g, self.dP_dT_g, self.dP_dV_g,
             self.dV_dT_g, self.dV_dP_g, self.dT_dV_g, self.dT_dP_g,
             self.d2P_dT2_g, self.d2P_dV2_g, self.d2P_dTdV_g, self.H_dep_g,
             self.S_dep_g, self.G_dep_g, self.Cp_dep_g, self.Cv_dep_g) = (
                     V, Z, PIP, dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP,
                     d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, G_dep, Cp_dep,
                     Cv_dep)
            return 'g'



    def a_alpha_and_derivatives(self, T, full=True, quick=True,
                                pure_a_alphas=True):
        r'''Method to calculate :math:`a \alpha` and its first and second
        derivatives.

        Parameters
        ----------
        T : float
            Temperature, [K]
        full : bool, optional
            If False, calculates and returns only `a_alpha`, [-]
        quick : bool, optional
           Legary parameter being phased out [-]
        pure_a_alphas : bool, optional
            Whether or not to recalculate the a_alpha terms of pure components
            (for the case of mixtures only) which stay the same as the
            composition changes (i.e in a PT flash); does nothing in the case
            of pure EOSs [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]
        '''
        if full:
            return self.a_alpha_and_derivatives_pure(T=T)
        return self.a_alpha_pure(T)

    def a_alpha_and_derivatives_pure(self, T):
        r'''Dummy method to calculate :math:`a \alpha` and its first and second
        derivatives. Should be implemented with the same function signature in
        each EOS variant; this only raises a NotImplemented Exception.
        Should return 'a_alpha', 'da_alpha_dT', and 'd2a_alpha_dT2'.

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]
        '''
        raise NotImplementedError('a_alpha and its first and second derivatives '
                                  'should be calculated by this method, in a user subclass.')

    @property
    def d3a_alpha_dT3(self):
        r'''Method to calculate the third temperature derivative of
        :math:`a \alpha`, [J^2/mol^2/Pa/K^3]. This parameter is needed for
        some higher derivatives that are needed in some flash calculations.

        Returns
        -------
        d3a_alpha_dT3 : float
            Third temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^3]
        '''
        try:
            return self._d3a_alpha_dT3
        except AttributeError:
            pass
        self._d3a_alpha_dT3 = self.d3a_alpha_dT3_pure(self.T)
        return self._d3a_alpha_dT3


    def a_alpha_plot(self, Tmin=1e-4, Tmax=None, pts=1000, plot=True,
                     show=True):
        r'''Method to create a plot of the :math:`a \alpha` parameter and its
        first two derivatives. This easily allows identification of EOSs which
        are displaying inconsistent behavior.

        Parameters
        ----------
        Tmin : float
            Minimum temperature of calculation, [K]
        Tmax : float
            Maximum temperature of calculation, [K]
        pts : int, optional
            The number of temperature points to include [-]
        plot : bool
            If False, the calculated values and temperatures are returned
            without plotting the data, [-]
        show : bool
            Whether or not the plot should be rendered and shown; a handle to
            it is returned if `plot` is True for other purposes such as saving
            the plot to a file, [-]

        Returns
        -------
        Ts : list[float]
            Logarithmically spaced temperatures in specified range, [K]
        a_alpha : list[float]
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : list[float]
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : list[float]
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]
        fig : matplotlib.figure.Figure
            Plotted figure, only returned if `plot` is True, [-]
        '''
        if Tmax is None:
            if self.multicomponent:
                Tc = self.pseudo_Tc
            else:
                Tc = self.Tc
            Tmax = Tc*10

        Ts = logspace(log10(Tmin), log10(Tmax), pts)

        a_alphas = []
        da_alphas = []
        d2a_alphas = []
        for T in Ts:
            v, d1, d2 = self.a_alpha_and_derivatives(T, full=True)
            a_alphas.append(v)
            da_alphas.append(d1)
            d2a_alphas.append(d2)

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            ax1.set_xlabel('Temperature [K]')
            ln0 = ax1.plot(Ts, a_alphas, 'r', label=r'$a \alpha$ [J^2/mol^2/Pa]')
            ln2 = ax1.plot(Ts, d2a_alphas, 'g', label='Second derivative [J^2/mol^2/Pa/K^2]')
            ax1.set_yscale('log')
            ax1.set_ylabel(r'$a \alpha$ and $\frac{\partial (a \alpha)^2}{\partial T^2}$')
            ax2 = ax1.twinx()
            ax2.set_yscale('symlog')
            ln1 = ax2.plot(Ts, da_alphas, 'b', label='First derivative [J^2/mol^2/Pa/K]')
            ax2.set_ylabel(r'$\frac{\partial a \alpha}{\partial T}$')
            ax1.set_title(fr'$a \alpha$ vs temperature; range {max(a_alphas):.4g} to {min(a_alphas):.4g}')

            lines = ln0 + ln1 + ln2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc=9, bbox_to_anchor=(0.5,-0.18))
            if show:
                plt.show()
            return Ts, a_alphas, da_alphas, d2a_alphas, fig
        return Ts, a_alphas, da_alphas, d2a_alphas


    def solve_T(self, P, V, solution=None):
        '''Generic method to calculate `T` from a specified `P` and `V`.
        Provides SciPy's `newton` solver, and iterates to solve the general
        equation for `P`, recalculating `a_alpha` as a function of temperature
        using `a_alpha_and_derivatives` each iteration.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        solution : str or None, optional
            'l' or 'g' to specify a liquid of vapor solution (if one exists);
            if None, will select a solution more likely to be real (closer to
            STP, attempting to avoid temperatures like 60000 K or 0.0001 K).

        Returns
        -------
        T : float
            Temperature, [K]
        '''
        high_prec = type(V) is not float
        denominator_inv = 1.0/(V*V + self.delta*V + self.epsilon)
        V_minus_b_inv = 1.0/(V-self.b)
        self.no_T_spec = True

        # dP_dT could be added to use a derivative-based method, however it is
        # quite costly in comparison to the extra evaluations because it
        # requires the temperature derivative of da_alpha_dT
        def to_solve(T):
            a_alpha = self.a_alpha_and_derivatives(T, full=False)
            P_calc = R*T*V_minus_b_inv - a_alpha*denominator_inv
            err = P_calc - P
            return err

        def to_solve_newton(T):
            a_alpha, da_alpha_dT, _ = self.a_alpha_and_derivatives(T, full=True)
            P_calc = R*T*V_minus_b_inv - a_alpha*denominator_inv
            err = P_calc - P
            derr_dT = R*V_minus_b_inv - denominator_inv*da_alpha_dT
            return err, derr_dT

        # import matplotlib.pyplot as plt
        # xs = np.logspace(np.log10(1), np.log10(1e12), 15000)
        # ys = np.abs([to_solve(T) for T in xs])
        # plt.loglog(xs, ys)
        # plt.show()
        # max(ys), min(ys)

        T_guess_ig = P*V*R_inv
        T_guess_liq = P*V*R_inv*1000.0 # Compressibility factor of 0.001 for liquids
        err_ig = to_solve(T_guess_ig)
        err_liq = to_solve(T_guess_liq)

        base_tol = 1e-12
        if high_prec:
            base_tol = 1e-18

        T_brenth, T_secant = None, None
        if err_ig*err_liq < 0.0 and T_guess_liq < 3e4:
            try:
                T_brenth = brenth(to_solve, T_guess_ig, T_guess_liq, xtol=base_tol,
                              fa=err_ig, fb=err_liq)
                # Check the error
                err = to_solve(T_brenth)
            except:
                pass
            # if abs(err/P) < 1e-7:
            #     return T_brenth


        if abs(err_ig) < abs(err_liq) or T_guess_liq > 20000 or solution == 'g':
            T_guess = T_guess_ig
            f0 = err_ig
        else:
            T_guess = T_guess_liq
            f0 = err_liq
        # T_guess = self.Tc*0.5
        # ytol=T_guess*1e-9,
        try:
            T_secant = secant(to_solve, T_guess, low=1e-12, xtol=base_tol, same_tol=1e4, f0=f0)
        except:
            T_guess = T_guess_ig if T_guess != T_guess_ig else T_guess_liq
            try:
                T_secant = secant(to_solve, T_guess, low=1e-12, xtol=base_tol, same_tol=1e4, f0=f0)
            except:
                if T_brenth is None:
                    # Hardcoded limits, all the cleverness sometimes does not work
                    T_brenth = brenth(to_solve, 1e-3, 1e4, xtol=base_tol)
        if solution is not None:
            if T_brenth is None or (T_secant is not None and isclose(T_brenth, T_secant, rel_tol=1e-7)):
                if T_secant is not None:
                    attempt_bounds = [(1e-3, T_secant-1e-5), (T_secant+1e-3, 1e4), (T_secant+1e-3, 1e5)]
                else:
                    attempt_bounds = [(1e-3, 1e4), (1e4, 1e5)]
                if T_guess_liq > 1e5:
                    attempt_bounds.append((1e4, T_guess_liq))
                    attempt_bounds.append((T_guess_liq, T_guess_liq*10))

                for low, high in attempt_bounds:
                    try:
                        T_brenth = brenth(to_solve, low, high, xtol=base_tol)
                        break
                    except:
                        pass
            if T_secant is None:
                if T_secant is not None:
                    attempt_bounds = [(1e-3, T_brenth-1e-5), (T_brenth+1e-3, 1e4), (T_brenth+1e-3, 1e5)]
                else:
                    attempt_bounds = [(1e4, 1e5), (1e-3, 1e4)]
                if T_guess_liq > 1e5:
                    attempt_bounds.append((1e4, T_guess_liq))
                    attempt_bounds.append((T_guess_liq, T_guess_liq*10))

                for low, high in attempt_bounds:
                    try:
                        T_secant = brenth(to_solve, low, high, xtol=base_tol)
                        break
                    except:
                        pass
        try:
            del self.a_alpha_ijs
            del self.a_alpha_roots
            del self.a_alpha_ij_roots_inv
        except AttributeError:
            pass

        if T_secant is not None:
            T_secant = float(T_secant)
        if T_brenth is not None:
            T_brenth = float(T_brenth)

        if solution is not None:
            if (T_secant is not None and T_brenth is not None):
                if solution == 'g':
                    return max(T_brenth, T_secant)
                else:
                    return min(T_brenth, T_secant)

        if T_brenth is None:
            return T_secant
        elif T_brenth is not None and T_secant is not None and (abs(T_brenth - 298.15) < abs(T_secant - 298.15)):
            return T_brenth
        elif T_secant is not None:
            return T_secant
        return T_brenth

        # return min(T_brenth, T_secant)



    # Default method
#    volume_solutions = volume_solutions_NR#volume_solutions_numpy#volume_solutions_NR
#    volume_solutions = staticmethod(volume_solutions_numpy)
#    volume_solutions = volume_solutions_fast
#    volume_solutions = staticmethod(volume_solutions_Cardano)
    volume_solutions = staticmethod(volume_solutions_halley)
#    volume_solutions = staticmethod(volume_solutions_doubledouble_float)

    volume_solutions_mp = staticmethod(volume_solutions_mpmath)

    # Solver which actually has the roots
    volume_solutions_full = staticmethod(volume_solutions_NR)

#    volume_solutions = volume_solutions_mpmath_float

    @property
    def mpmath_volumes(self):
        r'''Method to calculate to a high precision the exact roots to the
        cubic equation, using `mpmath`.

        Returns
        -------
        Vs : tuple[mpf]
            3 Real or not real volumes as calculated by `mpmath`, [m^3/mol]

        Notes
        -----

        Examples
        --------
        >>> eos = PRTranslatedTwu(T=300, P=1e5, Tc=512.5, Pc=8084000.0, omega=0.559, alpha_coeffs=(0.694911, 0.9199, 1.7), c=-1e-6)
        >>> eos.mpmath_volumes
        (mpf('0.0000489261705320261435106226558966745'), mpf('0.000541508154451321441068958547812526'), mpf('0.0243149463942697410611501615357228'))
        '''
        return volume_solutions_mpmath(self.T, self.P, self.b, self.delta, self.epsilon, self.a_alpha)

    @property
    def mpmath_volumes_float(self):
        r'''Method to calculate real roots of a cubic equation, using `mpmath`,
        but returned as floats.

        Returns
        -------
        Vs : list[float]
            All volumes calculated by `mpmath`, [m^3/mol]

        Notes
        -----

        Examples
        --------
        >>> eos = PRTranslatedTwu(T=300, P=1e5, Tc=512.5, Pc=8084000.0, omega=0.559, alpha_coeffs=(0.694911, 0.9199, 1.7), c=-1e-6)
        >>> eos.mpmath_volumes_float
        ((4.892617053202614e-05+0j), (0.0005415081544513214+0j), (0.024314946394269742+0j))
        '''
        return volume_solutions_mpmath_float(self.T, self.P, self.b, self.delta, self.epsilon, self.a_alpha)

    @property
    def mpmath_volume_ratios(self):
        r'''Method to compare, as ratios, the volumes of the implemented
        cubic solver versus those calculated using `mpmath`.

        Returns
        -------
        ratios : list[mpc]
            Either 1 or 3 volume ratios as calculated by `mpmath`, [-]

        Notes
        -----

        Examples
        --------
        >>> eos = PRTranslatedTwu(T=300, P=1e5, Tc=512.5, Pc=8084000.0, omega=0.559, alpha_coeffs=(0.694911, 0.9199, 1.7), c=-1e-6)
        >>> eos.mpmath_volume_ratios
        (mpc(real='0.99999999999999995', imag='0.0'), mpc(real='0.999999999999999965', imag='0.0'), mpc(real='1.00000000000000005', imag='0.0'))
        '''
        return tuple(i/j for i, j in zip(self.sorted_volumes, self.mpmath_volumes))

    def Vs_mpmath(self):
        r'''Method to calculate real roots of a cubic equation, using `mpmath`.

        Returns
        -------
        Vs : list[mpf]
            Either 1 or 3 real volumes as calculated by `mpmath`, [m^3/mol]

        Notes
        -----

        Examples
        --------
        >>> eos = PRTranslatedTwu(T=300, P=1e5, Tc=512.5, Pc=8084000.0, omega=0.559, alpha_coeffs=(0.694911, 0.9199, 1.7), c=-1e-6)
        >>> eos.Vs_mpmath()
        [mpf('0.0000489261705320261435106226558966745'), mpf('0.000541508154451321441068958547812526'), mpf('0.0243149463942697410611501615357228')]
        '''
        Vs = self.mpmath_volumes
        good_roots = [i.real for i in Vs if (i.real > 0.0 and abs(i.imag/i.real) < 1E-12)]
        good_roots.sort()
        return good_roots


    def volume_error(self):
        r'''Method to calculate the relative absolute error in the calculated
        molar volumes. This is computed with `mpmath`. If the number of real
        roots is different between mpmath and the implemented solver, an
        error of 1 is returned.

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        error : float
            relative absolute error in molar volumes , [-]

        Notes
        -----

        Examples
        --------
        >>> eos = PRTranslatedTwu(T=300, P=1e5, Tc=512.5, Pc=8084000.0, omega=0.559, alpha_coeffs=(0.694911, 0.9199, 1.7), c=-1e-6)
        >>> eos.volume_error()
        5.2192e-17
        '''
#        Vs_good, Vs = self.mpmath_volumes, self.sorted_volumes
        # Compare the reals only if mpmath has the imaginary roots
        Vs_good = self.volume_solutions_mp(self.T, self.P, self.b, self.delta, self.epsilon, self.a_alpha)
        Vs_filtered = [i.real for i in Vs_good if (i.real ==0 or abs(i.imag/i.real) < 1E-20) and i.real > self.b]
        if len(Vs_filtered) in (2, 3):
            two_roots_mpmath = True
            Vl_mpmath, Vg_mpmath = min(Vs_filtered), max(Vs_filtered)
        else:
            if hasattr(self, 'V_l') and hasattr(self, 'V_g'):
                # Wrong number of roots!
                return 1
            elif hasattr(self, 'V_l'):
                Vl_mpmath = Vs_filtered[0]
            elif hasattr(self, 'V_g'):
                Vg_mpmath = Vs_filtered[0]
            two_roots_mpmath = False
        err = 0

        if two_roots_mpmath:
            if (not hasattr(self, 'V_l') or not hasattr(self, 'V_g')):
                return 1.0

        # Important not to confuse the roots and also to not consider the third root
        try:
            Vl = self.V_l
            err_i = abs((Vl - Vl_mpmath)/Vl_mpmath)
            if err_i > err:
                err = err_i
        except:
            pass
        try:
            Vg = self.V_g
            err_i = abs((Vg - Vg_mpmath)/Vg_mpmath)
            if err_i > err:
                err = err_i
        except:
            pass
        return float(err)

    def _mpmath_volume_matching(self, V):
        '''Helper method which, given one of the three molar volume solutions
        of the EOS, returns the mpmath molar volume which is nearest it.
        '''
        Vs = self.mpmath_volumes
        rel_diffs = []

        for Vi in Vs:
            err = abs(Vi.real - V.real) + abs(Vi.imag - V.imag)
            rel_diffs.append(err)
        return Vs[rel_diffs.index(min(rel_diffs))]

    @property
    def V_l_mpmath(self):
        r'''The molar volume of the liquid phase calculated with `mpmath` to
        a higher precision, [m^3/mol]. This is useful for validating the
        cubic root solver(s). It is not quite a true arbitrary solution to the
        EOS, because the constants `b`,`epsilon`, `delta` and `a_alpha` as well
        as the input arguments `T` and `P` are not calculated with arbitrary
        precision. This is a feature when comparing the volume solution
        algorithms however as they work with the same finite-precision
        variables.
        '''
        if not hasattr(self, 'V_l'):
            raise ValueError("Not solved for that volume")
        return self._mpmath_volume_matching(self.V_l)

    @property
    def V_g_mpmath(self):
        r'''The molar volume of the gas phase calculated with `mpmath` to
        a higher precision, [m^3/mol]. This is useful for validating the
        cubic root solver(s). It is not quite a true arbitrary solution to the
        EOS, because the constants `b`,`epsilon`, `delta` and `a_alpha` as well
        as the input arguments `T` and `P` are not calculated with arbitrary
        precision. This is a feature when comparing the volume solution
        algorithms however as they work with the same finite-precision
        variables.
        '''
        if not hasattr(self, 'V_g'):
            raise ValueError("Not solved for that volume")
        return self._mpmath_volume_matching(self.V_g)

#    def fugacities_mpmath(self, dps=30):
#        # At one point thought maybe the fugacity equation was the source of error.
#        # No. always the volume equation.
#        import mpmath as mp
#        mp.mp.dps = dps
#        R_mp = mp.mpf(R)
#        b, T, P, epsilon, delta, a_alpha = self.b, self.T, self.P, self.epsilon, self.delta, self.a_alpha
#        b, T, P, epsilon, delta, a_alpha = [mp.mpf(i) for i in [b, T, P, epsilon, delta, a_alpha]]
#
#        Vs_good = volume_solutions_mpmath(self.T, self.P, self.b, self.delta, self.epsilon, self.a_alpha)
#        Vs_filtered = [i.real for i in Vs_good if (i.real == 0 or abs(i.imag/i.real) < 1E-20) and i.real > self.b]
#
#        if len(Vs_filtered) in (2, 3):
#            Vs = min(Vs_filtered), max(Vs_filtered)
#        else:
#            if hasattr(self, 'V_l') and hasattr(self, 'V_g'):
#                # Wrong number of roots!
#                raise ValueError("Error")
#            Vs = Vs_filtered
##            elif hasattr(self, 'V_l'):
##                Vs = Vs_filtered[0]
##            elif hasattr(self, 'V_g'):
##                Vg_mpmath = Vs_filtered[0]
#
#        log, exp, atanh, sqrt = mp.log, mp.exp, mp.atanh, mp.sqrt
#
#        return [P*exp((P*V + R_mp*T*log(V) - R_mp*T*log(P*V/(R_mp*T)) - R_mp*T*log(V - b)
#                       - R_mp*T - 2*a_alpha*atanh(2*V/sqrt(delta**2 - 4*epsilon)
#                       + delta/sqrt(delta**2 - 4*epsilon)).real/sqrt(delta**2 - 4*epsilon))/(R_mp*T))
#                for V in Vs]



    def volume_errors(self, Tmin=1e-4, Tmax=1e4, Pmin=1e-2, Pmax=1e9,
                      pts=50, plot=False, show=False, trunc_err_low=1e-18,
                      trunc_err_high=1.0, color_map=None, timing=False):
        r'''Method to create a plot of the relative absolute error in the
        cubic volume solution as compared to a higher-precision calculation.
        This method is incredible valuable for the development of more reliable
        floating-point based cubic solutions.

        Parameters
        ----------
        Tmin : float
            Minimum temperature of calculation, [K]
        Tmax : float
            Maximum temperature of calculation, [K]
        Pmin : float
            Minimum pressure of calculation, [Pa]
        Pmax : float
            Maximum pressure of calculation, [Pa]
        pts : int, optional
            The number of points to include in both the `x` and `y` axis;
            the validation calculation is slow, so increasing this too much
            is not advisable, [-]
        plot : bool
            If False, the calculated errors are returned without plotting
            the data, [-]
        show : bool
            Whether or not the plot should be rendered and shown; a handle to
            it is returned if `plot` is True for other purposes such as saving
            the plot to a file, [-]
        trunc_err_low : float
            Minimum plotted error; values under this are rounded to 0, [-]
        trunc_err_high : float
            Maximum plotted error; values above this are rounded to 1, [-]
        color_map : matplotlib.cm.ListedColormap
            Matplotlib colormap object, [-]
        timing : bool
            If True, plots the time taken by the volume root calculations
            themselves; this can reveal whether the solvers are taking fast or
            slow paths quickly, [-]

        Returns
        -------
        errors : list[list[float]]
            Relative absolute errors in the volume calculation (or timings in
            seconds if `timing` is True), [-]
        fig : matplotlib.figure.Figure
            Plotted figure, only returned if `plot` is True, [-]
        '''
        if timing:
            try:
                from time import perf_counter
            except:
                from time import clock as perf_counter
        Ts = logspace(log10(Tmin), log10(Tmax), pts)
        Ps = logspace(log10(Pmin), log10(Pmax), pts)
        kwargs = {}
        if hasattr(self, 'zs'):
            kwargs['zs'] = self.zs
            kwargs['fugacities'] = False

        errs = []
        for T in Ts:
            err_row = []
            for P in Ps:
                kwargs['T'] = T
                kwargs['P'] = P
                try:
                    obj = self.to(**kwargs)
                except Exception as e:
                    print(f'Failed to go to point, kwargs={kwargs} with exception {e}')
                    # So bad we failed to calculate a real point
                    val = 1.0
                if timing:
                    t0 = perf_counter()
                    obj.volume_solutions(obj.T, obj.P, obj.b, obj.delta, obj.epsilon, obj.a_alpha)
                    val = perf_counter() - t0
                else:
                    val = float(obj.volume_error())
                    if val > 1e-7:
                        print([obj.T, obj.P, obj.b, obj.delta, obj.epsilon, obj.a_alpha, 'coordinates of failure', obj])
                err_row.append(val)
            errs.append(err_row)

        if plot:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.colors import LogNorm
            X, Y = np.meshgrid(Ts, Ps)
            z = np.array(errs).T
            fig, ax = plt.subplots()
            if not timing:
                if trunc_err_low is not None:
                    z[np.where(abs(z) < trunc_err_low)] = trunc_err_low
                if trunc_err_high is not None:
                    z[np.where(abs(z) > trunc_err_high)] = trunc_err_high

            if color_map is None:
                color_map = cm.viridis

            if not timing:
                norm = LogNorm(vmin=trunc_err_low, vmax=trunc_err_high)
            else:
                z *= 1e-6
                norm = None
            im = ax.pcolormesh(X, Y, z, cmap=color_map, norm=norm)
            cbar = fig.colorbar(im, ax=ax)
            if timing:
                cbar.set_label('Time [us]')
            else:
                cbar.set_label('Relative error')

            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel('T [K]')
            ax.set_ylabel('P [Pa]')

            max_err = np.max(errs)
            if trunc_err_low is not None and max_err < trunc_err_low:
                max_err = 0
            if trunc_err_high is not None and max_err > trunc_err_high:
                max_err = trunc_err_high

            if timing:
                ax.set_title('Volume timings; max %.2e us' %(max_err*1e6))
            else:
                ax.set_title(f'Volume solution validation; max err {max_err:.4e}')
            if show:
                plt.show()

            return errs, fig
        else:
            return errs

    def PT_surface_special(self, Tmin=1e-4, Tmax=1e4, Pmin=1e-2, Pmax=1e9,
                      pts=50, show=False, color_map=None,
                      mechanical=True, pseudo_critical=True, Psat=True,
                      determinant_zeros=True, phase_ID_transition=True,
                      base_property='V', base_min=None, base_max=None,
                      base_selection='Gmin'):
        r'''Method to create a plot of the special curves of a pure fluid -
        vapor pressure, determinant zeros, pseudo critical point,
        and mechanical critical point.

        The color background is a plot of the molar volume (by default) which
        has the minimum Gibbs energy (by default). If shown with a sufficient
        number of points, the curve between vapor and liquid should be shown
        smoothly.

        When called on a mixture, this method does not have physical
        significance for the `Psat` term.

        Parameters
        ----------
        Tmin : float, optional
            Minimum temperature of calculation, [K]
        Tmax : float, optional
            Maximum temperature of calculation, [K]
        Pmin : float, optional
            Minimum pressure of calculation, [Pa]
        Pmax : float, optional
            Maximum pressure of calculation, [Pa]
        pts : int, optional
            The number of points to include in both the `x` and `y` axis [-]
        show : bool, optional
            Whether or not the plot should be rendered and shown; a handle to
            it is returned if `plot` is True for other purposes such as saving
            the plot to a file, [-]
        color_map : matplotlib.cm.ListedColormap, optional
            Matplotlib colormap object, [-]
        mechanical : bool, optional
            Whether or not to include the mechanical critical point; this is
            the same as the critical point for a pure compound but not for a
            mixture, [-]
        pseudo_critical : bool, optional
            Whether or not to include the pseudo critical point; this is
            the same as the critical point for a pure compound but not for a
            mixture, [-]
        Psat : bool, optional
            Whether or not to include the vapor pressure curve; for mixtures
            this is neither the bubble nor dew curve, but rather a hypothetical
            one which uses the same equation as the pure components, [-]
        determinant_zeros : bool, optional
            Whether or not to include a curve showing when the EOS's
            determinant hits zero, [-]
        phase_ID_transition : bool, optional
            Whether or not to show a curve of where the PIP hits 1 exactly, [-]
        base_property : str, optional
            The property which should be plotted; '_l' and '_g' are added
            automatically according to the selected phase, [-]
        base_min : float, optional
            If specified, the `base` property will values will be limited to
            this value at the minimum, [-]
        base_max : float, optional
            If specified, the `base` property will values will be limited to
            this value at the maximum, [-]
        base_selection : str, optional
            For the base property, there are often two possible phases and but
            only one value can be plotted; use 'l' to pefer liquid-like values,
            'g' to prefer gas-like values, and 'Gmin' to prefer values of the
            phase with the lowest Gibbs energy, [-]


        Returns
        -------
        fig : matplotlib.figure.Figure
            Plotted figure, only returned if `plot` is True, [-]
        '''
        Ts = logspace(log10(Tmin), log10(Tmax), pts)
        Ps = logspace(log10(Pmin), log10(Pmax), pts)
        kwargs = {}
        if hasattr(self, 'zs'):
            kwargs['zs'] = self.zs

        l_prop = base_property + '_l'
        g_prop = base_property + '_g'
        base_positive = True

        # Are we an ideal-gas or multicomponent?
        if self.Zc == 1.:
            phase_ID_transition = False
            Psat = False

        Vs = []
        for T in Ts:
            V_row = []
            for P in Ps:
                kwargs['T'] = T
                kwargs['P'] = P
                obj = self.to(**kwargs)
                if obj.phase == 'l/g':
                    if base_selection == 'Gmin':
                        V = getattr(obj, l_prop) if obj.G_dep_l < obj.G_dep_g else getattr(obj, g_prop)
                    elif base_selection == 'l':
                        V = getattr(obj, l_prop)
                    elif base_selection == 'g':
                        V = getattr(obj, g_prop)
                    else:
                        raise ValueError("Unknown value for base_selection")
                elif obj.phase == 'l':
                    V = getattr(obj, l_prop)
                else:
                    V = getattr(obj, g_prop)
                if base_max is not None and V > base_max: V = base_max
                if base_min is not None and V < base_min: V = base_min
                V_row.append(V)
                base_positive = base_positive and V > 0.0
            Vs.append(V_row)

        if self.multicomponent:
            Tc, Pc = self.pseudo_Tc, self.pseudo_Pc
        else:
            Tc, Pc = self.Tc, self.Pc

        if Psat:
            Pmax_Psat = min(Pc, Pmax)
            Pmin_Psat = max(1e-20, Pmin)
            Tmin_Psat, Tmax_Psat = self.Tsat(Pmin_Psat), self.Tsat(Pmax_Psat)
            if Tmin_Psat < Tmin or Tmin_Psat > Tmax: Tmin_Psat = Tmin
            if Tmax_Psat > Tmax or Tmax_Psat < Tmin: Tmax_Psat = Tmax

            Ts_Psats = []
            Psats = []
            for T in linspace(Tmin_Psat, Tmax_Psat, pts):
                P = self.Psat(T)
                Ts_Psats.append(T)
                Psats.append(P)
        if phase_ID_transition:
            Pmin_Psat = max(1e-20, Pmin)
            Tmin_ID = self.Tsat(Pmin_Psat)
            Tmax_ID = Tmax
            phase_ID_Ts = linspace(Tmin_ID, Tmax_ID, pts)
            low_P_limit = min(1e-4, Pmin)
            phase_ID_Ps = [self.P_PIP_transition(T, low_P_limit=low_P_limit)
            for T in phase_ID_Ts]


        if mechanical:
            if self.multicomponent:
                TP_mechanical = self.mechanical_critical_point()
            else:
                TP_mechanical = (Tc, Pc)

        if determinant_zeros:
            lows_det_Ps, high_det_Ps, Ts_dets_low, Ts_dets_high = [], [], [], []
            for T in Ts:
                a_alpha = self.a_alpha_and_derivatives(T, full=False)
                P_dets = self.P_discriminant_zeros_analytical(T=T, b=self.b, delta=self.delta,
                                                              epsilon=self.epsilon, a_alpha=a_alpha, valid=True)
                if P_dets:
                    P_det_min = min(P_dets)
                    P_det_max = max(P_dets)
                    if Pmin <= P_det_min <= Pmax:
                        lows_det_Ps.append(P_det_min)
                        Ts_dets_low.append(T)

                    if Pmin <= P_det_max <= Pmax:
                        high_det_Ps.append(P_det_max)
                        Ts_dets_high.append(T)


#        if plot:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import LogNorm
        X, Y = np.meshgrid(Ts, Ps)
        z = np.array(Vs).T
        fig, ax = plt.subplots()
        if color_map is None:
            color_map = cm.viridis

        norm = LogNorm() if base_positive else None
        im = ax.pcolormesh(X, Y, z, cmap=color_map, norm=norm)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f'{base_property}')

        if Psat:
            plt.plot(Ts_Psats, Psats, label='Psat')

        if determinant_zeros:
            plt.plot(Ts_dets_low, lows_det_Ps, label='Low trans')
            plt.plot(Ts_dets_high, high_det_Ps, label='High trans')

        if pseudo_critical:
            plt.plot([Tc], [Pc], 'x', label='Pseudo crit')
        if mechanical:
            plt.plot([TP_mechanical[0]], [TP_mechanical[1]], 'o', label='Mechanical')
        if phase_ID_transition:
            plt.plot(phase_ID_Ts, phase_ID_Ps, label='PIP=1')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('T [K]')
        ax.set_ylabel('P [Pa]')

        if (Psat or determinant_zeros or pseudo_critical or mechanical
            or phase_ID_transition):
            plt.legend()


        ax.set_title(f'{base_property} vs minimum Gibbs validation')
        if show:
            plt.show()

        return fig

    def saturation_prop_plot(self, prop, Tmin=None, Tmax=None, pts=100,
                             plot=False, show=False, both=False):
        r'''Method to create a plot of a specified property of the EOS along
        the (pure component) saturation line.

        Parameters
        ----------
        prop : str
            Property to be used; such as 'H_dep_l' ( when `both` is False)
            or 'H_dep' (when `both` is True), [-]
        Tmin : float
            Minimum temperature of calculation; if this is too low the
            saturation routines will stop converging, [K]
        Tmax : float
            Maximum temperature of calculation; cannot be above the critical
            temperature, [K]
        pts : int, optional
            The number of temperature points to include [-]
        plot : bool
            If False, the calculated values and temperatures are returned
            without plotting the data, [-]
        show : bool
            Whether or not the plot should be rendered and shown; a handle to
            it is returned if `plot` is True for other purposes such as saving
            the plot to a file, [-]
        both : bool
            When true, append '_l' and '_g' and draw both the liquid and vapor
            property specified and return two different sets of values.

        Returns
        -------
        Ts : list[float]
            Logarithmically spaced temperatures in specified range, [K]
        props : list[float]
            The property specified if `both` is False; otherwise, the liquid
            properties, [various]
        props_g : list[float]
            The gas properties, only returned if `both` is True, [various]
        fig : matplotlib.figure.Figure
            Plotted figure, only returned if `plot` is True, [-]
        '''
        if Tmax is None:
            if self.multicomponent:
                Tmax = self.pseudo_Tc
            else:
                Tmax = self.Tc
        if Tmin is None:
            Tmin = self.Tsat(1e-5)


        Ts = logspace(log10(Tmin), log10(Tmax), pts)
        kwargs = {}
        if hasattr(self, 'zs'):
            kwargs['zs'] = self.zs
        props = []
        if both:
            props2 = []
            prop_l = prop + '_l'
            prop_g = prop + '_g'

        for T in Ts:
            kwargs['T'] = T
            kwargs['P'] = self.Psat(T)
            obj = self.to(**kwargs)
            if both:
                v = getattr(obj, prop_l)
                try:
                    v = v()
                except:
                    pass
                props.append(v)

                v = getattr(obj, prop_g)
                try:
                    v = v()
                except:
                    pass
                props2.append(v)
            else:
                v = getattr(obj, prop)
                try:
                    v = v()
                except:
                    pass
                props.append(v)

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

            if both:
                plt.plot(Ts, props, label='Liquid')
                plt.plot(Ts, props2, label='Gas')
                plt.legend()
            else:
                plt.plot(Ts, props)

            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel(rf'{prop}')


            ax.set_title(rf'Saturation {prop} curve')
            if show:
                plt.show()

            if both:
                return Ts, props, props2, fig
            return Ts, props, fig
        if both:
            return Ts, props, props2
        return Ts, props

    def Psat_errors(self, Tmin=None, Tmax=None, pts=50, plot=False, show=False,
                    trunc_err_low=1e-18, trunc_err_high=1.0, Pmin=1e-100):
        r'''Method to create a plot of vapor pressure and the relative error
        of its calculation vs. the iterative `polish` approach.

        Parameters
        ----------
        Tmin : float
            Minimum temperature of calculation; if this is too low the
            saturation routines will stop converging, [K]
        Tmax : float
            Maximum temperature of calculation; cannot be above the critical
            temperature, [K]
        pts : int, optional
            The number of temperature points to include [-]
        plot : bool
            If False, the solution is returned without plotting the data, [-]
        show : bool
            Whether or not the plot should be rendered and shown; a handle to
            it is returned if `plot` is True for other purposes such as saving
            the plot to a file, [-]
        trunc_err_low : float
            Minimum plotted error; values under this are rounded to 0, [-]
        trunc_err_high : float
            Maximum plotted error; values above this are rounded to 1, [-]
        Pmin : float
            Minimum pressure for the solution to work on, [Pa]

        Returns
        -------
        errors : list[float]
            Absolute relative errors, [-]
        Psats_num : list[float]
            Vapor pressures calculated to full precision, [Pa]
        Psats_fit : list[float]
            Vapor pressures calculated with the fast solution, [Pa]
        fig : matplotlib.figure.Figure
            Plotted figure, only returned if `plot` is True, [-]
        '''
        try:
            Tc = self.Tc
        except:
            Tc = self.pseudo_Tc


        if Tmax is None:
            Tmax = Tc
        if Tmin is None:
            Tmin = .1*Tc

        try:
            # Can we get the direct temperature for Pmin
            if Pmin is not None:
                Tmin_Pmin = self.Tsat(P=Pmin, polish=True)
        except:
            Tmin_Pmin = None

        if Tmin_Pmin is not None:
            Tmin = max(Tmin, Tmin_Pmin)

        Ts = logspace(log10(Tmin), log10(Tmax), int(pts/3))
        Ts[-1] = Tmax

        Ts_mid = linspace(Tmin, Tmax, int(pts/3))

        Ts_high = linspace(Tmax*.99, Tmax, int(pts/3))
        Ts = list(sorted(Ts_high + Ts + Ts_mid))



        Ts_worked, Psats_num, Psats_fit = [], [], []
        for T in Ts:
            failed = False
            try:
                Psats_fit.append(self.Psat(T, polish=False))
            except NoSolutionError:
                # Trust the fit - do not continue if no good
                continue
            except Exception as e:
                raise ValueError(f"Failed to converge at {T:.16f} K with unexpected error", e, self)

            try:
                Psat_polished = self.Psat(T, polish=True)
                Psats_num.append(Psat_polished)
            except Exception as e:
                failed = True
                raise ValueError(f"Failed to converge at {T:.16f} K with unexpected error", e, self)

            Ts_worked.append(T)
        Ts = Ts_worked

        errs = np.array([abs(i-j)/i for i, j in zip(Psats_num, Psats_fit)])
        if plot:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            if trunc_err_low is not None:
                errs[np.where(abs(errs) < trunc_err_low)] = trunc_err_low
            if trunc_err_high is not None:
                errs[np.where(abs(errs) > trunc_err_high)] = trunc_err_high

            Trs = np.array(Ts)/Tc
            ax1.plot(Trs, errs)

            ax2.plot(Trs, Psats_num)
            ax2.plot(Trs, Psats_fit)
            ax1.set_yscale('log')
            ax1.set_xscale('log')

            ax2.set_yscale('log')
            ax2.set_xscale('log')

            ax1.set_xlabel('Tr [-]')
            ax1.set_ylabel('AARD [-]')

            ax2.set_ylabel('Psat [Pa]')

            max_err = np.max(errs)
            if trunc_err_low is not None and max_err < trunc_err_low:
                max_err = 0
            if trunc_err_high is not None and max_err > trunc_err_high:
                max_err = trunc_err_high

            ax1.set_title(f'Vapor pressure validation; max rel err {max_err:.4e}')
            if show:
                plt.show()

            return errs, Psats_num, Psats_fit, fig
        else:
            return errs, Psats_num, Psats_fit

#    def PIP_map(self, Tmin=1e-4, Tmax=1e4, Pmin=1e-2, Pmax=1e9,
#                pts=50, plot=False, show=False, color_map=None):
#        # TODO rename PIP_ID_map or add flag to change if it plots PIP or bools.
#        # TODO add doc
#        Ts = logspace(log10(Tmin), log10(Tmax), pts)
#        Ps = logspace(log10(Pmin), log10(Pmax), pts)
#        kwargs = {}
#        if hasattr(self, 'zs'):
#            kwargs['zs'] = self.zs
#
#        PIPs = []
#        for T in Ts:
#            PIP_row = []
#            for P in Ps:
#                kwargs['T'] = T
#                kwargs['P'] = P
#                obj = self.to(**kwargs)
##                v = obj.discriminant
##                # need make negatives under 1, positive above 1
##                if v > 0.0:
##                    v = (1.0 + (1e10 - 1.0)/(1.0 + trunc_exp(-v)))
##                else:
##                    v = (1e-10 + (1.0 - 1e-10)/(1.0 + trunc_exp(-v)))
#
#                if obj.phase == 'l/g':
#                    v = 1
#                elif obj.phase == 'g':
#                    v = 0
#                elif obj.phase == 'l':
#                    v = 2
#                PIP_row.append(v)
#            PIPs.append(PIP_row)
#
#        if plot:
#            import matplotlib.pyplot as plt
#            from matplotlib import ticker, cm
#            from matplotlib.colors import LogNorm
#            X, Y = np.meshgrid(Ts, Ps)
#            z = np.array(PIPs).T
#            fig, ax = plt.subplots()
#            if color_map is None:
#                color_map = cm.viridis
#
#            im = ax.pcolormesh(X, Y, z, cmap=color_map,
##                               norm=LogNorm(vmin=1e-10, vmax=1e10)
#                               )
#            cbar = fig.colorbar(im, ax=ax)
#            cbar.set_label('PIP')
#
#            ax.set_yscale('log')
#            ax.set_xscale('log')
#            ax.set_xlabel('T [K]')
#            ax.set_ylabel('P [Pa]')
#
#
#            ax.set_title('Volume root/phase ID validation')
#            if show:
#                plt.show()
#
#            return PIPs, fig

    def derivatives_and_departures(self, T, P, V, b, delta, epsilon, a_alpha, da_alpha_dT, d2a_alpha_dT2, quick=True):

        dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep = (
        self.main_derivatives_and_departures(T, P, V, b, delta, epsilon,
                                             a_alpha, da_alpha_dT,
                                             d2a_alpha_dT2))
        try:
            dV_dP = 1.0/dP_dV
        except:
            dV_dP = inf
        dT_dP = 1./dP_dT

        dV_dT = -dP_dT*dV_dP
        dT_dV = 1./dV_dT


        dV_dP2 = dV_dP*dV_dP
        dV_dP3 = dV_dP*dV_dP2

        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP

        d2V_dP2 = -d2P_dV2*dV_dP3 # unused
        d2T_dP2 = -d2P_dT2*inverse_dP_dT3 # unused

        d2T_dV2 = (-(d2P_dV2*dP_dT - dP_dV*d2P_dTdV)*inverse_dP_dT2
                   +(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3*dP_dV) # unused
        d2V_dT2 = (-(d2P_dT2*dP_dV - dP_dT*d2P_dTdV)*dV_dP2 # unused
                   +(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*dV_dP3*dP_dT)

        d2V_dPdT = -(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*dV_dP3 # unused
        d2T_dPdV = -(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3 # unused

        return (dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP,
                d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2,
                d2V_dPdT, d2P_dTdV, d2T_dPdV, # d2P_dTdV is used
                H_dep, S_dep, Cv_dep)



    @property
    def sorted_volumes(self):
        r'''List of lexicographically-sorted molar volumes available from the
        root finding algorithm used to solve the PT point. The convention of
        sorting lexicographically comes from numpy's handling of complex
        numbers, which python does not define. This method was added to
        facilitate testing, as the volume solution method changes over time
        and the ordering does as well.

        Examples
        --------
        >>> PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6).sorted_volumes
        ((0.000130222125139+0j), (0.00112363131346-0.00129269672343j), (0.00112363131346+0.00129269672343j))
        '''
        sort_fun = lambda x: (x.real, x.imag)
        full_volumes = self.volume_solutions_full(self.T, self.P, self.b, self.delta, self.epsilon, self.a_alpha)
        full_volumes = [i + 0.0j for i in full_volumes]
        return tuple(sorted(full_volumes, key=sort_fun))


    def Tsat(self, P, polish=False):
        r'''Generic method to calculate the temperature for a specified
        vapor pressure of the pure fluid.
        This is simply a bounded solver running between `0.2Tc` and `Tc` on the
        `Psat` method.

        Parameters
        ----------
        P : float
            Vapor pressure, [Pa]
        polish : bool, optional
            Whether to attempt to use a numerical solver to make the solution
            more precise or not

        Returns
        -------
        Tsat : float
            Temperature of saturation, [K]

        Notes
        -----
        It is recommended not to run with `polish=True`, as that will make the
        calculation much slower.
        '''
        fprime = False
        global curr_err

        def to_solve_newton(T):
            global curr_err
            assert T > 0.0
            e = self.to_TP(T, P)
            try:
                fugacity_l = e.fugacity_l
            except AttributeError as err:
                raise err
            try:
                fugacity_g = e.fugacity_g
            except AttributeError as err:
                raise err

            curr_err = fugacity_l - fugacity_g
            if fprime:
                d_err_d_T = e.dfugacity_dT_l - e.dfugacity_dT_g
                return curr_err, d_err_d_T

            # print('err', err, 'rel err', err/T, 'd_err_d_T', d_err_d_T, 'T', T)

            return curr_err

        logP = log(P)
        def to_solve(T):
            global curr_err
            if fprime:
                dPsat_dT, Psat = self.dPsat_dT(T, polish=polish, also_Psat=True)
                curr_err = Psat - P

                # Log translation - tends to save a few iterations
                err_trans = log(Psat) - logP
                return err_trans, dPsat_dT/Psat
#                return curr_err, derr_dT
            curr_err = self.Psat(T, polish=polish) - P
            return curr_err#, derr_dT
#            return copysign(log(abs(err)), err)
        # Outstanding improvements to do: Better guess; get NR working;
        # see if there is a general curve

        try:
            Tc, Pc = self.Tc, self.Pc
        except:
            Tc, Pc = self.pseudo_Tc, self.pseudo_Pc

        guess = -5.4*Tc/(1.0*log(P/Pc) - 5.4)
        high = guess*2.0
        low = guess*0.5
#        return newton(to_solve, guess, fprime=True, ytol=1e-6, high=self.Pc)
#        return newton(to_solve, guess, ytol=1e-6, high=self.Pc)

        # Methanol is a good example of why 1.5 is needed
        low_hope, high_hope = max(guess*.5, 0.2*Tc), min(Tc, guess*1.5)


        try:
            err_low, err_high = to_solve(low_hope), to_solve(high_hope)
            if err_low*err_high < 0.0:
                if guess < low_hope or guess > high_hope:
                    guess = 0.5*(low_hope + high_hope)
                fprime = True
                Tsat = newton(to_solve, guess, xtol=1.48e-10,fprime=True, low=low_hope, high=high_hope, bisection=True)
    #            fprime = False
    #            Tsat = brenth(to_solve, low_hope, high_hope)
                abs_rel_err = abs(curr_err)/P
                if abs_rel_err < 1e-9:
                    return Tsat
                elif abs_rel_err < 1e-2:
                    guess = Tsat
            else:
                try:
                    return brenth(to_solve, 0.2*Tc, Tc)
                except:
                    try:
                        return brenth(to_solve, 0.2*Tc, Tc*1.5)
                    except:
                        pass
        except:
            pass
        fprime = True

        try:
            try:
                Tsat = newton(to_solve_newton, guess, fprime=True, maxiter=100,
                              xtol=4e-13, require_eval=False, damping=1.0, low=Tc*1e-5)
            except:
                try:
                    Tsat = newton(to_solve_newton, guess, fprime=True, maxiter=100,
                                  xtol=4e-13, require_eval=False, damping=1.0, low=low, high=high)
                    assert Tsat not in (low, high)
                except:
                    Tsat = newton(to_solve_newton, guess, fprime=True, maxiter=250, # the wider range can take more iterations
                                  xtol=4e-13, require_eval=False, damping=1.0, low=low, high=high*2)
                    assert Tsat != low and Tsat != high*2
        except:
            # high = self.Tc
            # try:
            #     high = min(high, self.T_discriminant_zero_l()*(1-1e-8))
            # except:
            #     pass
            # Does not seem to be working
            try:
                Tsat = None
                Tsat = newton(to_solve_newton, guess, fprime=True, maxiter=200, high=high, low=low,
                              xtol=4e-13, require_eval=False, damping=1.0)
            except:
                pass
            fprime = False
            if Tsat is None or abs(to_solve_newton(Tsat)) == P:
                Tsat = brenth(to_solve_newton, low, high)

        return Tsat

    def Psat(self, T, polish=False, guess=None):
        r'''Generic method to calculate vapor pressure for a specified `T`.

        From Tc to 0.32Tc, uses a 10th order polynomial of the following form:

        .. math::
            \ln\frac{P_r}{T_r} = \sum_{k=0}^{10} C_k\left(\frac{\alpha}{T_r}
            -1\right)^{k}

        If `polish` is True, SciPy's `newton` solver is launched with the
        calculated vapor pressure as an initial guess in an attempt to get more
        accuracy. This may not converge however.

        Results above the critical temperature are meaningless. A first-order
        polynomial is used to extrapolate under 0.32 Tc; however, there is
        normally not a volume solution to the EOS which can produce that
        low of a pressure.

        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to attempt to use a numerical solver to make the solution
            more precise or not

        Returns
        -------
        Psat : float
            Vapor pressure, [Pa]

        Notes
        -----
        EOSs sharing the same `b`, `delta`, and `epsilon` have the same
        coefficient sets.

        Form for the regression is inspired from [1]_.

        No volume solution is needed when `polish=False`; the only external
        call is for the value of `a_alpha`.

        References
        ----------
        .. [1] Soave, G. "Direct Calculation of Pure-Compound Vapour Pressures
           through Cubic Equations of State." Fluid Phase Equilibria 31, no. 2
           (January 1, 1986): 203-7. doi:10.1016/0378-3812(86)90013-0.
        '''
        Tc, Pc = self.Tc, self.Pc
        if T == Tc:
            return Pc
        a_alpha = self.a_alpha_and_derivatives(T, full=False)
        alpha = a_alpha/self.a
        Tr = T/self.Tc
        x = alpha/Tr - 1.


        if Tr > 0.999 and not isinstance(self, RK):
            y = horner(self.Psat_coeffs_critical, x)
            Psat = y*Tr*Pc
            if Psat > Pc and T < Tc:
                Psat = Pc*(1.0 - 1e-14)
        else:
            # TWUPR/SRK TODO need to be prepared for x being way outside the range (in the weird direction - at the start)
            Psat_ranges_low = self.Psat_ranges_low
            if x > Psat_ranges_low[-1]:
                if not polish:
                    raise NoSolutionError(f"T {T:.8f} K is too low for equations to converge")
                else:
                    # Needs to still be here for generating better data
                    x = Psat_ranges_low[-1]
                    polish = True

            for i in range(len(Psat_ranges_low)):
                if x < Psat_ranges_low[i]:
                    break
            y = 0.0
            for c in self.Psat_coeffs_low[i]:
                y = y*x + c

            try:
                Psat = exp(y)*Tr*Pc
                if Psat == 0.0:
                    if polish:
                        Psat = 1e-100
                    else:
                        raise NoSolutionError(f"T {T:.8f} K is too low for equations to converge")
            except OverflowError:
                # coefficients sometimes overflow before T is lowered to 0.32Tr
                # For
                polish = True # There is no solution available to polish
                Psat = 1

        if polish:
            if T > Tc:
                raise ValueError("Cannot solve for equifugacity condition "
                                 "beyond critical temperature")
            if guess is not None:
                Psat = guess
            converged = False
            def to_solve_newton(P):
                # For use by newton. Only supports initialization with Tc, Pc and omega
                # ~200x slower and not guaranteed to converge (primary issue is one phase)
                # not existing
                assert P > 0.0
                e = self.to_TP(T, P)
                # print(e.volume_error(), e)
                try:
                    fugacity_l = e.fugacity_l
                except AttributeError as err:
                    # return 1000, 1000
                    raise err

                try:
                    fugacity_g = e.fugacity_g
                except AttributeError as err:
                    # return 1000, 1000
                    raise err

                err = fugacity_l - fugacity_g

                d_err_d_P = e.dfugacity_dP_l - e.dfugacity_dP_g # -1 for low pressure
                if isnan(d_err_d_P):
                    d_err_d_P = -1.0
                # print('err', err, 'rel err', err/P, 'd_err_d_P', d_err_d_P, 'P', P)
                # Clamp the derivative - if it will step to zero or negative, dampen to half the distance which gets to zero
                if (P - err/d_err_d_P) <= 0.0: # This is the one matching newton
                # if (P - err*d_err_d_P) <= 0.0:
                    d_err_d_P = -1.0001

                return err, d_err_d_P
            try:
                try:
                    boundaries = GCEOS.P_discriminant_zeros_analytical(T, self.b, self.delta, self.epsilon, a_alpha, valid=True)
                    low, high = min(boundaries), max(boundaries)
                except:
                    pass
                try:
                    high = self.P_discriminant_zero()
                except:
                    high = Pc


                # def damping_func(p0, step, damping):
                #     if step == 1:
                #         damping = damping*0.5
                #     p = p0 + step * damping
                #     return p

                Psat = newton(to_solve_newton, Psat, high=high, fprime=True, maxiter=100,
                              xtol=4e-13, require_eval=False, damping=1.0) #  ,ytol=1e-6*Psat # damping_func=damping_func
#                print(to_solve_newton(Psat), 'newton error')
                converged = True
            except:
                pass

            if not converged:
                def to_solve_bisect(P):
                    e = self.to_TP(T, P)
                    # print(e.volume_error(), e)
                    try:
                        fugacity_l = e.fugacity_l
                    except AttributeError as err:
                        return 1e20

                    try:
                        fugacity_g = e.fugacity_g
                    except AttributeError as err:
                        return -1e20
                    err = fugacity_l - fugacity_g
#                    print(err, 'err', 'P', P)
                    return err
                for low, high in zip([.98*Psat, 1, 1e-40, Pc*.9, Psat*.9999], [1.02*Psat, Pc, 1, Pc*1.000000001, Pc]):
                    try:
                        Psat = bisect(to_solve_bisect, low, high, ytol=1e-6*Psat, maxiter=128)
#                        print(to_solve_bisect(Psat), 'bisect error')
                        converged = True
                        break
                    except:
                        pass

            # Last ditch attempt
            if not converged:
                # raise ValueError("Could not converge")
                if Tr > 0.5:
                    # Near critical temperature issues
                    points = [Pc*f for f in linspace(1e-3, 1-1e-8, 50) + linspace(.9, 1-1e-8, 50)]
                    ytol = 1e-6*Psat
                else:
                    # Low temperature issues
                    points = [Psat*f for f in logspace(-5.5, 5.5, 16)]
                    # points = [Psat*f for f in logspace(-2.5, 2.5, 100)]
                    ytol = None # Cryogenic point unlikely to work to desired tolerance
                    # Work on point closer to Psat first
                    points.sort(key=lambda x: abs(log10(x)))
                low, high = None, None
                for point in points:
                    try:
                        err = to_solve_newton(point)[0] # Do not use bisect function as it does not raise errors
                        if err > 0.0:
                            high = point
                        elif err < 0.0:
                            low = point
                    except:
                        pass
                    if low is not None and high is not None:
                        # print('reached bisection')
                        Psat = brenth(to_solve_bisect, low, high, ytol=ytol, maxiter=128)
#                        print(to_solve_bisect(Psat), 'bisect error')
                        converged = True
                        break
                # print('tried all points')
                # Check that the fugacity error vs. Psat is OK
                if abs(to_solve_bisect(Psat)/Psat) > .0001:
                    converged = False

            if not converged:
                raise ValueError(f"Could not converge at T={T:.6f} K")

        return Psat


    def dPsat_dT(self, T, polish=False, also_Psat=False):
        r'''Generic method to calculate the temperature derivative of vapor
        pressure for a specified `T`. Implements the analytical derivative
        of the three polynomials described in `Psat`.

        As with `Psat`, results above the critical temperature are meaningless.
        The first-order polynomial which is used to calculate it under 0.32 Tc
        may not be physicall meaningful, due to there normally not being a
        volume solution to the EOS which can produce that low of a pressure.

        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to attempt to use a numerical solver to make the solution
            more precise or not
        also_Psat : bool, optional
            Calculating `dPsat_dT` necessarily involves calculating `Psat`;
            when this is set to True, a second return value is added, whic is
            the actual `Psat` value.

        Returns
        -------
        dPsat_dT : float
            Derivative of vapor pressure with respect to temperature, [Pa/K]
        Psat : float, returned if `also_Psat` is `True`
            Vapor pressure, [Pa]

        Notes
        -----
        There is a small step change at 0.32 Tc for all EOS due to the two
        switch between polynomials at that point.

        Useful for calculating enthalpy of vaporization with the Clausius
        Clapeyron Equation. Derived with SymPy's diff and cse.
        '''
        if polish:
            # Calculate the derivative of saturation pressure analytically
            Psat = self.Psat(T, polish=polish)
            sat_eos = self.to(T=T, P=Psat)
            dfg_T, dfl_T = sat_eos.dfugacity_dT_g, sat_eos.dfugacity_dT_l
            dfg_P, dfl_P = sat_eos.dfugacity_dP_g, sat_eos.dfugacity_dP_l
            dPsat_dT = (dfg_T - dfl_T)/(dfl_P - dfg_P)
            if also_Psat:
                return dPsat_dT, Psat
            return dPsat_dT

        a_alphas = self.a_alpha_and_derivatives(T)
        a_inv = 1.0/self.a
        try:
            Tc, Pc = self.Tc, self.Pc
        except:
            Tc, Pc = self.pseudo_Tc, self.pseudo_Pc

        alpha, d_alpha_dT = a_alphas[0]*a_inv, a_alphas[1]*a_inv
        Tc_inv = 1.0/Tc
        T_inv = 1.0/T
        Tr = T*Tc_inv
#        if Tr < 0.32 and not isinstance(self, PR):
#            # Delete
#            c = self.Psat_coeffs_limiting
#            return self.Pc*T*c[0]*(self.Tc*d_alpha_dT/T - self.Tc*alpha/(T*T)
#                              )*exp(c[0]*(-1. + self.Tc*alpha/T) + c[1]
#                              )/self.Tc + self.Pc*exp(c[0]*(-1.
#                              + self.Tc*alpha/T) + c[1])/self.Tc
        if Tr > 0.999 and not isinstance(self, RK):
            # OK
            x = alpha/Tr - 1.
            y = horner(self.Psat_coeffs_critical, x)
            dy_dT = T_inv*(Tc*d_alpha_dT - Tc*alpha*T_inv)*horner(self.Psat_coeffs_critical_der, x)
            dPsat_dT = Pc*(T*dy_dT*Tc_inv + y*Tc_inv)
            if also_Psat:
                Psat = y*Tr*Pc
                return dPsat_dT, Psat
            return dPsat_dT
        else:
            Psat_coeffs_low = self.Psat_coeffs_low
            Psat_ranges_low = self.Psat_ranges_low
            x = alpha/Tr - 1.
            if x > Psat_ranges_low[-1]:
                raise NoSolutionError(f"T {T:.8f} K is too low for equations to converge")

            for i in range(len(Psat_ranges_low)):
                if x < Psat_ranges_low[i]:
                    break
            y, dy = 0.0, 0.0
            for c in Psat_coeffs_low[i]:
                dy = x*dy + y
                y = x*y + c

            exp_y = exp(y)
            dy_dT = Tc*T_inv*(d_alpha_dT - alpha*T_inv)*dy#horner_and_der(Psat_coeffs_low[i], x)[1]
            Psat = exp_y*Tr*Pc

            dPsat_dT = (T*dy_dT + 1.0)*Pc*exp_y*Tc_inv
            if also_Psat:
                return dPsat_dT, Psat
            return dPsat_dT


#            # change chebval to horner, and get new derivative
#            x = alpha/Tr - 1.
#            arg = (self.Psat_cheb_constant_factor[1]*(x + self.Psat_cheb_constant_factor[0]))
#            y = chebval(arg, self.Psat_cheb_coeffs)
#
#            exp_y = exp(y)
#            dy_dT = T_inv*(Tc*d_alpha_dT - Tc*alpha*T_inv)*chebval(arg,
#                     self.Psat_cheb_coeffs_der)*self.Psat_cheb_constant_factor[1]
#            Psat = Pc*T*exp_y*dy_dT*Tc_inv + Pc*exp_y*Tc_inv
#            return Psat

    def phi_sat(self, T, polish=True):
        r'''Method to calculate the saturation fugacity coefficient of the
        compound. This does not require solving the EOS itself.

        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to perform a rigorous calculation or to use a polynomial
            fit, [-]

        Returns
        -------
        phi_sat : float
            Fugacity coefficient along the liquid-vapor saturation line, [-]

        Notes
        -----
        Accuracy is generally around 1e-7. If Tr is under 0.32, the rigorous
        method is always used, but a solution may not exist if both phases
        cannot coexist. If Tr is above 1, likewise a solution does not exist.
        '''
        Tr = T/self.Tc
        if polish or not 0.32 <= Tr <= 1.0:
            e = self.to_TP(T=T, P=self.Psat(T, polish=True)) # True
            try:
                return e.phi_l
            except:
                return e.phi_g

        alpha = self.a_alpha_and_derivatives(T, full=False)/self.a
        x = alpha/Tr - 1.
        return horner(self.phi_sat_coeffs, x)

    def dphi_sat_dT(self, T, polish=True):
        r'''Method to calculate the temperature derivative of saturation
        fugacity coefficient of the
        compound. This does require solving the EOS itself.

        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to perform a rigorous calculation or to use a polynomial
            fit, [-]

        Returns
        -------
        dphi_sat_dT : float
            First temperature derivative of fugacity coefficient along the
            liquid-vapor saturation line, [1/K]

        Notes
        -----
        '''
        if T == self.Tc:
            T = (self.Tc*(1.0 - 1e-15))
        Psat = self.Psat(T, polish=polish)
        sat_eos = self.to(T=T, P=Psat)
        dfg_T, dfl_T = sat_eos.dfugacity_dT_g, sat_eos.dfugacity_dT_l
        dfg_P, dfl_P = sat_eos.dfugacity_dP_g, sat_eos.dfugacity_dP_l
        dPsat_dT = (dfg_T - dfl_T)/(dfl_P - dfg_P)

        fugacity = sat_eos.fugacity_l
        dfugacity_sat_dT = dPsat_dT*sat_eos.dfugacity_dP_l + sat_eos.dfugacity_dT_l

        Psat_inv = 1.0/Psat

        return (dfugacity_sat_dT - fugacity*dPsat_dT*Psat_inv)*Psat_inv

    def d2phi_sat_dT2(self, T, polish=True):
        r'''Method to calculate the second temperature derivative of saturation
        fugacity coefficient of the
        compound. This does require solving the EOS itself.

        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to perform a rigorous calculation or to use a polynomial
            fit, [-]

        Returns
        -------
        d2phi_sat_dT2 : float
            Second temperature derivative of fugacity coefficient along the
            liquid-vapor saturation line, [1/K^2]

        Notes
        -----
        This is presently a numerical calculation.
        '''
        return derivative(lambda T: self.dphi_sat_dT(T, polish=polish), T,
                          dx=T*1e-7, upper_limit=self.Tc)

    def V_l_sat(self, T):
        r'''Method to calculate molar volume of the liquid phase along the
        saturation line.

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        V_l_sat : float
            Liquid molar volume along the saturation line, [m^3/mol]

        Notes
        -----
        Computes `Psat`, and then uses `volume_solutions` to obtain the three
        possible molar volumes. The lowest value is returned.
        '''
        Psat = self.Psat(T)
        a_alpha = self.a_alpha_and_derivatives(T, full=False)
        Vs = self.volume_solutions(T, Psat, self.b, self.delta, self.epsilon, a_alpha)
        # Assume we can safely take the Vmax as gas, Vmin as l on the saturation line
        return min([i.real for i in Vs if i.real > self.b])

    def V_g_sat(self, T):
        r'''Method to calculate molar volume of the vapor phase along the
        saturation line.

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        V_g_sat : float
            Gas molar volume along the saturation line, [m^3/mol]

        Notes
        -----
        Computes `Psat`, and then uses `volume_solutions` to obtain the three
        possible molar volumes. The highest value is returned.
        '''
        Psat = self.Psat(T)
        a_alpha = self.a_alpha_and_derivatives(T, full=False)
        Vs = self.volume_solutions(T, Psat, self.b, self.delta, self.epsilon, a_alpha)
        # Assume we can safely take the Vmax as gas, Vmin as l on the saturation line
        return max([i.real for i in Vs])

    def Hvap(self, T):
        r'''Method to calculate enthalpy of vaporization for a pure fluid from
        an equation of state, without iteration.

        .. math::
            \frac{dP^{sat}}{dT}=\frac{\Delta H_{vap}}{T(V_g - V_l)}

        Results above the critical temperature are meaningless. A first-order
        polynomial is used to extrapolate under 0.32 Tc; however, there is
        normally not a volume solution to the EOS which can produce that
        low of a pressure.

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        Hvap : float
            Increase in enthalpy needed for vaporization of liquid phase along
            the saturation line, [J/mol]

        Notes
        -----
        Calculates vapor pressure and its derivative with `Psat` and `dPsat_dT`
        as well as molar volumes of the saturation liquid and vapor phase in
        the process.

        Very near the critical point this provides unrealistic results due to
        `Psat`'s polynomials being insufficiently accurate.

        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
           Butterworth-Heinemann, 1985.
        '''
        Psat = self.Psat(T)
        dPsat_dT = self.dPsat_dT(T)
        a_alpha = self.a_alpha_and_derivatives(T, full=False)
        Vs = self.volume_solutions(T, Psat, self.b, self.delta, self.epsilon, a_alpha)
        # Assume we can safely take the Vmax as gas, Vmin as l on the saturation line
        Vs = [i.real for i in Vs]
        V_l, V_g = min(Vs), max(Vs)
        return dPsat_dT*T*(V_g - V_l)

    def dH_dep_dT_sat_l(self, T, polish=False):
        r'''Method to calculate and return the temperature derivative of
        saturation liquid excess enthalpy.

        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to perform a rigorous calculation or to use a polynomial
            fit, [-]

        Returns
        -------
        dH_dep_dT_sat_l : float
            Liquid phase temperature derivative of excess enthalpy along the
            liquid-vapor saturation line, [J/mol/K]

        Notes
        -----
        '''
        sat_eos = self.to(T=T, P=self.Psat(T, polish=polish))
        dfg_T, dfl_T = sat_eos.dfugacity_dT_g, sat_eos.dfugacity_dT_l
        dfg_P, dfl_P = sat_eos.dfugacity_dP_g, sat_eos.dfugacity_dP_l
        dPsat_dT = (dfg_T - dfl_T)/(dfl_P - dfg_P)
        return dPsat_dT*sat_eos.dH_dep_dP_l + sat_eos.dH_dep_dT_l

    def dH_dep_dT_sat_g(self, T, polish=False):
        r'''Method to calculate and return the temperature derivative of
        saturation vapor excess enthalpy.

        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to perform a rigorous calculation or to use a polynomial
            fit, [-]

        Returns
        -------
        dH_dep_dT_sat_g : float
            Vapor phase temperature derivative of excess enthalpy along the
            liquid-vapor saturation line, [J/mol/K]

        Notes
        -----
        '''
        sat_eos = self.to(T=T, P=self.Psat(T, polish=polish))
        dfg_T, dfl_T = sat_eos.dfugacity_dT_g, sat_eos.dfugacity_dT_l
        dfg_P, dfl_P = sat_eos.dfugacity_dP_g, sat_eos.dfugacity_dP_l
        dPsat_dT = (dfg_T - dfl_T)/(dfl_P - dfg_P)
        return dPsat_dT*sat_eos.dH_dep_dP_g + sat_eos.dH_dep_dT_g

    def dS_dep_dT_sat_g(self, T, polish=False):
        r'''Method to calculate and return the temperature derivative of
        saturation vapor excess entropy.

        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to perform a rigorous calculation or to use a polynomial
            fit, [-]

        Returns
        -------
        dS_dep_dT_sat_g : float
            Vapor phase temperature derivative of excess entropy along the
            liquid-vapor saturation line, [J/mol/K^2]

        Notes
        -----
        '''
        sat_eos = self.to(T=T, P=self.Psat(T, polish=polish))
        dfg_T, dfl_T = sat_eos.dfugacity_dT_g, sat_eos.dfugacity_dT_l
        dfg_P, dfl_P = sat_eos.dfugacity_dP_g, sat_eos.dfugacity_dP_l
        dPsat_dT = (dfg_T - dfl_T)/(dfl_P - dfg_P)
        return dPsat_dT*sat_eos.dS_dep_dP_g + sat_eos.dS_dep_dT_g

    def dS_dep_dT_sat_l(self, T, polish=False):
        r'''Method to calculate and return the temperature derivative of
        saturation liquid excess entropy.

        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to perform a rigorous calculation or to use a polynomial
            fit, [-]

        Returns
        -------
        dS_dep_dT_sat_l : float
            Liquid phase temperature derivative of excess entropy along the
            liquid-vapor saturation line, [J/mol/K^2]

        Notes
        -----
        '''
        sat_eos = self.to(T=T, P=self.Psat(T, polish=polish))
        dfg_T, dfl_T = sat_eos.dfugacity_dT_g, sat_eos.dfugacity_dT_l
        dfg_P, dfl_P = sat_eos.dfugacity_dP_g, sat_eos.dfugacity_dP_l
        dPsat_dT = (dfg_T - dfl_T)/(dfl_P - dfg_P)
        return dPsat_dT*sat_eos.dS_dep_dP_l + sat_eos.dS_dep_dT_l


    def a_alpha_for_V(self, T, P, V):
        r'''Method to calculate which value of :math:`a \alpha` is required for
        a given `T`, `P` pair to match a specified `V`. This is a
        straightforward analytical equation.

        Parameters
        ----------
        T : float
            Temperature, [K]
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]

        Returns
        -------
        a_alpha : float
            Value calculated to match specified volume for the current EOS,
            [J^2/mol^2/Pa]

        Notes
        -----
        The derivation of the solution is as follows:


        >>> from sympy import * # doctest:+SKIP
        >>> P, T, V, R, b, a, delta, epsilon = symbols('P, T, V, R, b, a, delta, epsilon') # doctest:+SKIP
        >>> a_alpha = symbols('a_alpha') # doctest:+SKIP
        >>> CUBIC = R*T/(V-b) - a_alpha/(V*V + delta*V + epsilon) # doctest:+SKIP
        >>> solve(Eq(CUBIC, P), a_alpha)# doctest:+SKIP
        [(-P*V**3 + P*V**2*b - P*V**2*delta + P*V*b*delta - P*V*epsilon + P*b*epsilon + R*T*V**2 + R*T*V*delta + R*T*epsilon)/(V - b)]
        '''
        b, delta, epsilon = self.b, self.delta, self.epsilon
        x0 = P*b
        x1 = R*T
        x2 = V*delta
        x3 = V*V
        x4 = x3*V
        return ((-P*x4 - P*V*epsilon - P*delta*x3 + epsilon*x0 + epsilon*x1
                 + x0*x2 + x0*x3 + x1*x2 + x1*x3)/(V - b))


    def a_alpha_for_Psat(self, T, Psat, a_alpha_guess=None):
        r'''Method to calculate which value of :math:`a \alpha` is required for
        a given `T`, `Psat` pair. This is a numerical solution, but not a very
        complicated one.

        Parameters
        ----------
        T : float
            Temperature, [K]
        Psat : float
            Vapor pressure specified, [Pa]
        a_alpha_guess : float
            Optionally, an initial guess for the solver [J^2/mol^2/Pa]

        Returns
        -------
        a_alpha : float
            Value calculated to match specified volume for the current EOS,
            [J^2/mol^2/Pa]

        Notes
        -----
        The implementation of this function is a direct calculation of
        departure gibbs energy, which is equal in both phases at saturation.

        Examples
        --------
        >>> eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
        >>> eos.a_alpha_for_Psat(T=400, Psat=5e5)
        3.1565798926
        '''
        P = Psat
        b, delta, epsilon = self.b, self.delta, self.epsilon
        RT = R*T
        RT_inv = 1.0/RT
        x0 = 1.0/sqrt(delta*delta - 4.0*epsilon)
        x1 = delta*x0
        x2 = 2.0*x0

        def fug(V, a_alpha):
            # Can simplify this to not use a function, avoid 1 log anywayS
            G_dep = (P*V - RT - RT*log(P*RT_inv*(V-b))
                      - x2*a_alpha*catanh(2.0*V*x0 + x1).real)
            return G_dep # No point going all the way to fugacity

        def err(a_alpha):
            # Needs some work right up to critical point
            Vs = self.volume_solutions(T, P, b, delta, epsilon, a_alpha)
            good_roots = [i.real for i in Vs if i.imag == 0.0 and i.real > 0.0]
            good_root_count = len(good_roots)
            if good_root_count == 1:
                raise ValueError("Guess did not have two roots")
            V_l, V_g = min(good_roots), max(good_roots)
#            print(V_l, V_g, a_alpha)
            return fug(V_l, a_alpha) - fug(V_g, a_alpha)

        if a_alpha_guess is None:
            try:
                a_alpha_guess = self.a_alpha
            except AttributeError:
                a_alpha_guess = 0.002

        try:
            return secant(err, a_alpha_guess, xtol=1e-13)
        except:
            return secant(err, self.to(T=T, P=Psat).a_alpha, xtol=1e-13)

    def to_TP(self, T, P):
        r'''Method to construct a new EOS object at the spcified `T` and `P`.
        In the event the `T` and `P` match the current object's `T` and `P`,
        it will be returned unchanged.

        Parameters
        ----------
        T : float
            Temperature, [K]
        P : float
            Pressure, [Pa]

        Returns
        -------
        obj : EOS
            Pure component EOS at specified `T` and `P`, [-]

        Notes
        -----
        Constructs the object with parameters `Tc`, `Pc`, `omega`, and
        `kwargs`.

        Examples
        --------
        >>> base = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=500.0, P=1E6)
        >>> new = base.to_TP(T=1.0, P=2.0)
        >>> base.state_specs, new.state_specs
        ({'T': 500.0, 'P': 1000000.0}, {'T': 1.0, 'P': 2.0})
        '''
        if T != self.T or P != self.P:
            return self.__class__(T=T, P=P, Tc=self.Tc, Pc=self.Pc, omega=self.omega, **self.kwargs)
        else:
            return self

    def to_TV(self, T, V):
        r'''Method to construct a new EOS object at the spcified `T` and `V`.
        In the event the `T` and `V` match the current object's `T` and `V`,
        it will be returned unchanged.

        Parameters
        ----------
        T : float
            Temperature, [K]
        V : float
            Molar volume, [m^3/mol]

        Returns
        -------
        obj : EOS
            Pure component EOS at specified `T` and `V`, [-]

        Notes
        -----
        Constructs the object with parameters `Tc`, `Pc`, `omega`, and
        `kwargs`.

        Examples
        --------
        >>> base = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=500.0, P=1E6)
        >>> new = base.to_TV(T=1000000.0, V=1.0)
        >>> base.state_specs, new.state_specs
        ({'T': 500.0, 'P': 1000000.0}, {'T': 1000000.0, 'V': 1.0})
        '''
        if T != self.T or V != self.V:
            # Only allow creation of new class if volume actually specified
            # Ignores the posibility that V is V_l or V_g
            return self.__class__(T=T, V=V, Tc=self.Tc, Pc=self.Pc, omega=self.omega, **self.kwargs)
        else:
            return self

    def to_PV(self, P, V):
        r'''Method to construct a new EOS object at the spcified `P` and `V`.
        In the event the `P` and `V` match the current object's `P` and `V`,
        it will be returned unchanged.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]

        Returns
        -------
        obj : EOS
            Pure component EOS at specified `P` and `V`, [-]

        Notes
        -----
        Constructs the object with parameters `Tc`, `Pc`, `omega`, and
        `kwargs`.

        Examples
        --------
        >>> base = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=500.0, P=1E6)
        >>> new = base.to_PV(P=1000.0, V=1.0)
        >>> base.state_specs, new.state_specs
        ({'T': 500.0, 'P': 1000000.0}, {'P': 1000.0, 'V': 1.0})
        '''
        if P != self.P or V != self.V:
            return self.__class__(V=V, P=P, Tc=self.Tc, Pc=self.Pc, omega=self.omega, **self.kwargs)
        else:
            return self

    def to(self, T=None, P=None, V=None):
        r'''Method to construct a new EOS object at two of `T`, `P` or `V`.
        In the event the specs match those of the current object, it will be
        returned unchanged.

        Parameters
        ----------
        T : float or None, optional
            Temperature, [K]
        P : float or None, optional
            Pressure, [Pa]
        V : float or None, optional
            Molar volume, [m^3/mol]

        Returns
        -------
        obj : EOS
            Pure component EOS at the two specified specs, [-]

        Notes
        -----
        Constructs the object with parameters `Tc`, `Pc`, `omega`, and
        `kwargs`.

        Examples
        --------
        >>> base = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=500.0, P=1E6)
        >>> base.to(T=300.0, P=1e9).state_specs
        {'T': 300.0, 'P': 1000000000.0}
        >>> base.to(T=300.0, V=1.0).state_specs
        {'T': 300.0, 'V': 1.0}
        >>> base.to(P=1e5, V=1.0).state_specs
        {'P': 100000.0, 'V': 1.0}
        '''
        if T is not None and P is not None:
            return self.to_TP(T, P)
        elif T is not None and V is not None:
            return self.to_TV(T, V)
        elif P is not None and V is not None:
            return self.to_PV(P, V)
        else:
            # Error message
            return self.__class__(T=T, V=V, P=P, Tc=self.Tc, Pc=self.Pc, omega=self.omega, **self.kwargs)

    def T_min_at_V(self, V, Pmin=1e-15):
        '''Returns the minimum temperature for the EOS to have the
        volume as specified. Under this temperature, the pressure will go
        negative (and the EOS will not solve).
        '''
        return self.solve_T(P=Pmin, V=V)

    def T_max_at_V(self, V, Pmax=None):
        r'''Method to calculate the maximum temperature the EOS can create at a
        constant volume, if one exists; returns None otherwise.

        Parameters
        ----------
        V : float
            Constant molar volume, [m^3/mol]
        Pmax : float
            Maximum possible isochoric pressure, if already known [Pa]

        Returns
        -------
        T : float
            Maximum possible temperature, [K]

        Notes
        -----


        Examples
        --------
        >>> e = PR(P=1e5, V=0.0001437, Tc=512.5, Pc=8084000.0, omega=0.559)
        >>> e.T_max_at_V(e.V)
        431155.5
        '''
        if Pmax is None:
            Pmax = self.P_max_at_V(V)
            if Pmax is None:
                return None
        return self.solve_T(P=Pmax, V=V)

    def P_max_at_V(self, V):
        r'''Dummy method. The idea behind this method, which is implemented by some
        subclasses, is to calculate the maximum pressure the EOS can create at a
        constant volume, if one exists; returns None otherwise. This method,
        as a dummy method, always returns None.

        Parameters
        ----------
        V : float
            Constant molar volume, [m^3/mol]

        Returns
        -------
        P : float
            Maximum possible isochoric pressure, [Pa]
        '''
        return None

    @property
    def more_stable_phase(self):
        r'''Checks the Gibbs energy of each possible phase, and returns
        'l' if the liquid-like phase is more stable, and 'g' if the vapor-like
        phase is more stable.

        Examples
        --------
        >>> PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6).more_stable_phase
        'l'
        '''
        try:
            if self.G_dep_l < self.G_dep_g:
                return 'l'
            else:
                return 'g'
        except:
            try:
                self.Z_g
                return 'g'
            except:
                return 'l'

    def discriminant(self, T=None, P=None):
        r'''Method to compute the discriminant of the cubic volume solution
        with the current EOS parameters, optionally at the same (assumed) `T`,
        and `P` or at different ones, if values are specified.

        Parameters
        ----------
        T : float, optional
            Temperature, [K]
        P : float, optional
            Pressure, [Pa]

        Returns
        -------
        discriminant : float
            Discriminant, [-]

        Notes
        -----
        This call is quite quick; only :math:`a \alpha` is needed and if `T` is
        the same as the current object than it has already been computed.

        The formula is as follows:

        .. math::
            \text{discriminant} = - \left(- \frac{27 P^{2} \epsilon \left(
            \frac{P b}{R T} + 1\right)}{R^{2} T^{2}} - \frac{27 P^{2} b
            \operatorname{a \alpha}{\left(T \right)}}{R^{3} T^{3}}\right)
            \left(- \frac{P^{2} \epsilon \left(\frac{P b}{R T} + 1\right)}
            {R^{2} T^{2}} - \frac{P^{2} b \operatorname{a \alpha}{\left(T
            \right)}}{R^{3} T^{3}}\right) + \left(- \frac{P^{2} \epsilon \left(
            \frac{P b}{R T} + 1\right)}{R^{2} T^{2}} - \frac{P^{2} b
            \operatorname{a \alpha}{\left(T \right)}}{R^{3} T^{3}}\right)
            \left(- \frac{18 P b}{R T} + \frac{18 P \delta}{R T} - 18\right)
            \left(\frac{P^{2} \epsilon}{R^{2} T^{2}} - \frac{P \delta \left(
            \frac{P b}{R T} + 1\right)}{R T} + \frac{P \operatorname{a \alpha}
            {\left(T \right)}}{R^{2} T^{2}}\right) - \left(- \frac{P^{2}
            \epsilon \left(\frac{P b}{R T} + 1\right)}{R^{2} T^{2}} - \frac{
            P^{2} b \operatorname{a \alpha}{\left(T \right)}}{R^{3} T^{3}}
            \right) \left(- \frac{4 P b}{R T} + \frac{4 P \delta}{R T}
            - 4\right) \left(- \frac{P b}{R T} + \frac{P \delta}{R T}
            - 1\right)^{2} + \left(- \frac{P b}{R T} + \frac{P \delta}{R T}
            - 1\right)^{2} \left(\frac{P^{2} \epsilon}{R^{2} T^{2}} - \frac{P
            \delta \left(\frac{P b}{R T} + 1\right)}{R T} + \frac{P
            \operatorname{a \alpha}{\left(T \right)}}{R^{2} T^{2}}\right)^{2}
            - \left(\frac{P^{2} \epsilon}{R^{2} T^{2}} - \frac{P \delta \left(
            \frac{P b}{R T} + 1\right)}{R T} + \frac{P \operatorname{a \alpha}{
            \left(T \right)}}{R^{2} T^{2}}\right)^{2} \left(\frac{4 P^{2}
            \epsilon}{R^{2} T^{2}} - \frac{4 P \delta \left(\frac{P b}{R T}
            + 1\right)}{R T} + \frac{4 P \operatorname{a \alpha}{\left(T
            \right)}}{R^{2} T^{2}}\right)

        The formula is derived as follows:

        >>> from sympy import *
        >>> P, T, R, b = symbols('P, T, R, b')
        >>> a_alpha = symbols(r'a\ \alpha', cls=Function)
        >>> delta, epsilon = symbols('delta, epsilon')
        >>> eta = b
        >>> B = b*P/(R*T)
        >>> deltas = delta*P/(R*T)
        >>> thetas = a_alpha(T)*P/(R*T)**2
        >>> epsilons = epsilon*(P/(R*T))**2
        >>> etas = eta*P/(R*T)
        >>> a = 1
        >>> b = (deltas - B - 1)
        >>> c = (thetas + epsilons - deltas*(B+1))
        >>> d = -(epsilons*(B+1) + thetas*etas)
        >>> disc = b*b*c*c - 4*a*c*c*c - 4*b*b*b*d - 27*a*a*d*d + 18*a*b*c*d

        Examples
        --------
        >>> base = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=500.0, P=1E6)
        >>> base.discriminant()
        -0.001026390999
        >>> base.discriminant(T=400)
         0.0010458828
        >>> base.discriminant(T=400, P=1e9)
        12584660355.4
        '''
        if P is None:
            P = self.P
        if T is None:
            T = self.T
            a_alpha = self.a_alpha
        else:
            a_alpha = self.a_alpha_and_derivatives(T, full=False)
        RT = R*self.T
        RT6 = RT*RT
        RT6 *= RT6*RT6
        x0 = P*P
        x1 = P*self.b + RT
        x2 = a_alpha*self.b + self.epsilon*x1
        x3 = P*self.epsilon
        x4 = self.delta*x1
        x5 = -P*self.delta + x1
        x6 = a_alpha + x3 - x4
        x2_2 = x2*x2
        x5_2 = x5*x5
        x6_2 = x6*x6
        x7 = (-a_alpha - x3 + x4)
        return x0*(18.0*P*x2*x5*x6 + 4.0*P*x7*x7*x7
                   - 27.0*x0*x2_2 - 4.0*x2*x5_2*x5 + x5_2*x6_2)/RT6


    def _discriminant_at_T_mp(self, P):
        # Hopefully numerical difficulties can eventually be figured out to as to
        # not need mpmath
        import mpmath as mp
        mp.mp.dps = 70
        P, T, b, a_alpha, delta, epsilon, R_mp = (mp.mpf(i) for i in [P, self.T, self.b, self.a_alpha, self.delta, self.epsilon, R])
        RT = R_mp*T
        RT6 = RT**6
        x0 = P*P
        x1 = P*b + RT
        x2 = a_alpha*b + epsilon*x1
        x3 = P*epsilon
        x4 = delta*x1
        x5 = -P*delta + x1
        x6 = a_alpha + x3 - x4
        x2_2 = x2*x2
        x5_2 = x5*x5
        x6_2 = x6*x6
        disc = (x0*(18.0*P*x2*x5*x6 + 4.0*P*(-a_alpha - x3 + x4)**3
                   - 27.0*x0*x2_2 - 4.0*x2*x5_2*x5 + x5_2*x6_2)/RT6)
        return disc

    def P_discriminant_zero_l(self):
        r'''Method to calculate the pressure which zero the discriminant
        function of the general cubic eos, and is likely to sit on a boundary
        between not having a liquid-like volume; and having a liquid-like volume.

        Returns
        -------
        P_discriminant_zero_l : float
            Pressure which make the discriminants zero at the right condition,
            [Pa]

        Notes
        -----

        Examples
        --------
        >>> eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
        >>> P_trans = eos.P_discriminant_zero_l()
        >>> P_trans
        478346.37289

        In this case, the discriminant transition shows the change in roots:

        >>> eos.to(T=eos.T, P=P_trans*.99999999).mpmath_volumes_float
        ((0.00013117994140177062+0j), (0.002479717165903531+0j), (0.002480236178570793+0j))
        >>> eos.to(T=eos.T, P=P_trans*1.0000001).mpmath_volumes_float
        ((0.0001311799413872173+0j), (0.002479976386402769-8.206310112063695e-07j), (0.002479976386402769+8.206310112063695e-07j))
        '''
        return self._P_discriminant_zero(low=True)

    def P_discriminant_zero_g(self):
        r'''Method to calculate the pressure which zero the discriminant
        function of the general cubic eos, and is likely to sit on a boundary
        between not having a vapor-like volume; and having a vapor-like volume.

        Returns
        -------
        P_discriminant_zero_g : float
            Pressure which make the discriminants zero at the right condition,
            [Pa]

        Notes
        -----

        Examples
        --------
        >>> eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
        >>> P_trans = eos.P_discriminant_zero_g()
        >>> P_trans
        149960391.7

        In this case, the discriminant transition does not reveal a transition
        to two roots being available, only negative roots becoming negative
        and imaginary.

        >>> eos.to(T=eos.T, P=P_trans*.99999999).mpmath_volumes_float
        ((-0.0001037013146195082-1.5043987866732543e-08j), (-0.0001037013146195082+1.5043987866732543e-08j), (0.00011799201928619508+0j))
        >>> eos.to(T=eos.T, P=P_trans*1.0000001).mpmath_volumes_float
        ((-0.00010374888853182635+0j), (-0.00010365374200380354+0j), (0.00011799201875924273+0j))
        '''
        return self._P_discriminant_zero(low=False)

    def P_discriminant_zeros(self):
        r'''Method to calculate the pressures which zero the discriminant
        function of the general cubic eos, at the current temperature.

        Returns
        -------
        P_discriminant_zeros : list[float]
            Pressures which make the discriminants zero, [Pa]

        Notes
        -----

        Examples
        --------
        >>> eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
        >>> eos.P_discriminant_zeros()
        [478346.3, 149960391.7]
        '''
        return GCEOS.P_discriminant_zeros_analytical(self.T, self.b, self.delta, self.epsilon, self.a_alpha, valid=True)

    @staticmethod
    def P_discriminant_zeros_analytical(T, b, delta, epsilon, a_alpha, valid=False):
        r'''Method to calculate the pressures which zero the discriminant
        function of the general cubic eos. This is a quartic function
        solved analytically.


        Parameters
        ----------
        T : float
            Temperature, [K]
        b : float
            Coefficient calculated by EOS-specific method, [m^3/mol]
        delta : float
            Coefficient calculated by EOS-specific method, [m^3/mol]
        epsilon : float
            Coefficient calculated by EOS-specific method, [m^6/mol^2]
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        valid : bool
            Whether to filter the calculated pressures so that they are all
            real, and positive only, [-]

        Returns
        -------
        P_discriminant_zeros : float
            Pressures which make the discriminants zero, [Pa]

        Notes
        -----
        Calculated analytically. Derived as follows.

        >>> from sympy import *
        >>> P, T, V, R, b, a, delta, epsilon = symbols('P, T, V, R, b, a, delta, epsilon')
        >>> eta = b
        >>> B = b*P/(R*T)
        >>> deltas = delta*P/(R*T)
        >>> thetas = a*P/(R*T)**2
        >>> epsilons = epsilon*(P/(R*T))**2
        >>> etas = eta*P/(R*T)
        >>> a_coeff = 1
        >>> b_coeff = (deltas - B - 1)
        >>> c = (thetas + epsilons - deltas*(B+1))
        >>> d = -(epsilons*(B+1) + thetas*etas)
        >>> disc = b_coeff*b_coeff*c*c - 4*a_coeff*c*c*c - 4*b_coeff*b_coeff*b_coeff*d - 27*a_coeff*a_coeff*d*d + 18*a_coeff*b_coeff*c*d
        >>> base = -(expand(disc/P**2*R**3*T**3))
        >>> sln = collect(base, P)
        '''
        # Can also have one at g
#        T, a_alpha = self.T, self.a_alpha
        a = a_alpha
#        b, epsilon, delta = self.b, self.epsilon, self.delta

        T_inv = 1.0/T
        # TODO cse
        x0 = 4.0*a
        x1 = b*x0
        x2 = a+a
        x3 = delta*x2
        x4 = R*T
        x5 = 4.0*epsilon
        x6 = delta*delta
        x7 = a*a
        x8 = T_inv*R_inv
        x9 = 8.0*epsilon
        x10 = b*x9
        x11 = 4.0*delta
        x12 = delta*x6
        x13 = 2.0*x6
        x14 = b*x13
        x15 = a*x8
        x16 = epsilon*x15
        x20 = x8*x8
        x17 = x20*x8
        x18 = b*delta
        x19 = 6.0*x15
        x21 = x20*x7
        x22 = 10.0*b
        x23 = b*b
        x24 = 6.0*x23
        x25 = x0*x8
        x26 = x6*x6
        x27 = epsilon*epsilon
        x28 = 8.0*x27
        x29 = 24.0*epsilon
        x30 = b*x12
        x31 = epsilon*x13
        x32 = epsilon*x8
        x33 = 12.0*epsilon
        x34 = b*x23
        x35 = x2*x8
        x36 = 8.0*x21
        x37 = x15*x6
        x38 = delta*x23
        x39 = b*x28
        x40 = x34*x9
        x41 = epsilon*x12
        x42 = x23*x23

        e = x1 + x3 + x4*x5 - x4*x6 - x7*x8
        d = (4.0*x7*a*x17 - 10.0*delta*x21 + 2.0*(epsilon*x11 + x10 - x12
             - x14 + x15*x24 + x18*x19 - x21*x22 + x25*x6) - 20.0*x16)
        c = x8*(-x1*x32 + x12*x35 + x15*(12.0*x34 + 18.0*x38) + x18*(x29 + x36)
                + x21*(x33 - x6) + x22*x37 + x23*(x29 + x36) - x24*x6 - x26
                + x28 - x3*x32 - 6.0*x30 + x31)
        b_coeff = (2.0*x20*(-b*x26 + delta*(x10*x15 + x25*x34) + epsilon*x14
                            + x23*(x15*x9 - 3.0*x12 + x37) - x13*x34 - x15*x30
                            -x16*x6 + x27*(x19 + x11) + x33*x38 + x35*x42
                            + x39 + x40 - x41))
        a_coeff = x17*(-2.0*b*x41 + delta*(x39 + x40)
                       + x27*(4.0*epsilon - x6)
                       - 2.0*x12*x34 + x23*(x28 + x31 - x26)
                       + x42*(x5 - x6))

#        e = (2*a*delta + 4*a*b -R*T*delta**2 - a**2/(R*T) + 4*R*T*epsilon)
#        d = (-4*b*delta**2 + 16*b*epsilon - 2*delta**3 + 8*delta*epsilon + 12*a*b**2/(R*T) + 12*a*b*delta/(R*T) + 8*a*delta**2/(R*T) - 20*a*epsilon/(R*T) - 20*a**2*b/(R**2*T**2) - 10*a**2*delta/(R**2*T**2) + 4*a**3/(R**3*T**3))
#        c = (-6*b**2*delta**2/(R*T) + 24*b**2*epsilon/(R*T) - 6*b*delta**3/(R*T) + 24*b*delta*epsilon/(R*T) - delta**4/(R*T) + 2*delta**2*epsilon/(R*T) + 8*epsilon**2/(R*T) + 12*a*b**3/(R**2*T**2) + 18*a*b**2*delta/(R**2*T**2) + 10*a*b*delta**2/(R**2*T**2) - 4*a*b*epsilon/(R**2*T**2) + 2*a*delta**3/(R**2*T**2) - 2*a*delta*epsilon/(R**2*T**2) + 8*a**2*b**2/(R**3*T**3) + 8*a**2*b*delta/(R**3*T**3) - a**2*delta**2/(R**3*T**3) + 12*a**2*epsilon/(R**3*T**3))
#        b_coeff = (-4*b**3*delta**2/(R**2*T**2) + 16*b**3*epsilon/(R**2*T**2) - 6*b**2*delta**3/(R**2*T**2) + 24*b**2*delta*epsilon/(R**2*T**2) - 2*b*delta**4/(R**2*T**2) + 4*b*delta**2*epsilon/(R**2*T**2) + 16*b*epsilon**2/(R**2*T**2) - 2*delta**3*epsilon/(R**2*T**2) + 8*delta*epsilon**2/(R**2*T**2) + 4*a*b**4/(R**3*T**3) + 8*a*b**3*delta/(R**3*T**3) + 2*a*b**2*delta**2/(R**3*T**3) + 16*a*b**2*epsilon/(R**3*T**3) - 2*a*b*delta**3/(R**3*T**3) + 16*a*b*delta*epsilon/(R**3*T**3) - 2*a*delta**2*epsilon/(R**3*T**3) + 12*a*epsilon**2/(R**3*T**3))
#        a_coeff = (-b**4*delta**2/(R**3*T**3) + 4*b**4*epsilon/(R**3*T**3) - 2*b**3*delta**3/(R**3*T**3) + 8*b**3*delta*epsilon/(R**3*T**3) - b**2*delta**4/(R**3*T**3) + 2*b**2*delta**2*epsilon/(R**3*T**3) + 8*b**2*epsilon**2/(R**3*T**3) - 2*b*delta**3*epsilon/(R**3*T**3) + 8*b*delta*epsilon**2/(R**3*T**3) - delta**2*epsilon**2/(R**3*T**3) + 4*epsilon**3/(R**3*T**3))
        roots = roots_quartic(a_coeff, b_coeff, c, d, e)
#        roots = np.roots([a_coeff, b_coeff, c, d, e]).tolist()
        if valid:
            # TODO - only include ones when switching phases from l/g to either g/l
            # Do not know how to handle
            roots = [r.real for r in roots if (r.real >= 0.0)]
            roots.sort()
        return roots


    def _P_discriminant_zero(self, low):
        # Can also have one at g
        T, a_alpha = self.T, self.a_alpha
        b, epsilon, delta = self.b, self.epsilon, self.delta
        global niter
        niter = 0
        RT = R*T
        x13 = RT**-6.0
        x14 = b*epsilon
        x15 = -b*delta + epsilon
        x18 = b - delta
        def discriminant_fun(P):
            if P < 0:
                raise ValueError("Will not converge")
            global niter
            niter += 1
            x0 = P*P
            x1 = P*epsilon
            x2 = P*b + RT
            x3 = a_alpha - delta*x2 + x1
            x3_x3 = x3*x3
            x4 = x3*x3_x3
            x5 = a_alpha*b + epsilon*x2
            x6 = 27.0*x5*x5
            x7 = -P*delta + x2
            x9 = x7*x7
            x8 = x7*x9
            x11 = x3*x5*x7
            x12 = -18.0*P*x11 + 4.0*(P*x4 +x5*x8) + x0*x6 - x3_x3*x9
            x16 = P*x15
            x17 = 9.0*x3
            x19 = x18*x5
            # 26 mult so far
            err = -x0*x12*x13
            fprime = (-2.0*P*x13*(P*(-P*x17*x19 + P*x6 - b*x1*x17*x7
                                     + 27.0*x0*x14*x5 + 6.0*x3_x3*x16 - x3_x3*x18*x7
                                     - 9.0*x11 + 2.0*x14*x8 - x15*x3*x9 - 9.0*x16*x5*x7 + 6.0*x19*x9 + 2.0*x4) + x12))

            if niter > 3 and (.40 < (err/(P*fprime)) < 0.55):
                raise ValueError("Not going to work")
                # a = (err/fprime)/P
                # print('low probably kill point')
            return err, fprime

        # New answer: Above critical T only high P result
        # Ps = logspace(log10(1), log10(1e10), 40000)
        # errs = []
        # for P in Ps:
        #     erri = self.discriminant(P=P)
        #     if erri < 0:
        #         erri = -log10(abs(erri))
        #     else:
        #         erri = log10(erri)
        #     errs.append(erri)
        # import matplotlib.pyplot as plt
        # plt.semilogx(Ps, errs, 'x')
        # # plt.ylim((-1e-3, 1e-3))
        # plt.show()

        # Checked once
        # def damping_func(p0, step, damping):
        #     if p0 + step < 0.0:
        #         return 0.9*p0
        #     # while p0 + step < 1e3:
        #     # if p0 + step < 1e3:
        #     #     step = 0.5*step
        #     return p0 + step
        #low=1,damping_func=damping_func
        # 5e7

        try:
            Tc = self.Tc
        except:
            Tc = self.pseudo_Tc


        guesses = [1e5, 1e6, 1e7, 1e8, 1e9, .5, 1e-4, 1e-8, 1e-12, 1e-16, 1e-20]
        if not low:
            guesses = [1e9, 1e10, 1e10, 5e10, 2e10, 5e9, 5e8, 1e8]
        if self.N == 1 and low:
            try:
                try:
                    Tc, Pc, omega = self.Tc, self.Pc, self.omega
                except:
                    Tc, Pc, omega = self.Tcs[0], self.Pcs[0], self.omegas[0]
                guesses.append(Pc*.99999999)
                assert T/Tc > .3
                P_wilson = Wilson_K_value(self.T, self.P, Tc, Pc, omega)*self.P
                guesses.insert(0, P_wilson*3)
            except:
                pass

        if low:
            coeffs = self._P_zero_l_cheb_coeffs
            coeffs_low, coeffs_high = self.P_zero_l_cheb_limits
        else:
            coeffs = self._P_zero_g_cheb_coeffs
            coeffs_low, coeffs_high = self.P_zero_g_cheb_limits


        if coeffs is not None:
            try:
                a = self.a
            except:
                a = self.pseudo_a
            alpha = self.a_alpha/a

            try:
                Pc = self.Pc
            except:
                Pc = self.pseudo_Pc

            Tr = self.T/Tc
            alpha_Tr = alpha/(Tr)
            x = alpha_Tr - 1.0
            if coeffs_low < x <  coeffs_high:
                constant = 0.5*(-coeffs_low - coeffs_high)
                factor = 2.0/(coeffs_high - coeffs_low)

                y = chebval(factor*(x + constant), coeffs)
                P_trans = y*Tr*Pc

                guesses.insert(0, P_trans)


        global_iter = 0
        for P in guesses:
            try:
                global_iter += niter
                niter = 0
                # try:
                #     P_disc = newton(discriminant_fun, P, fprime=True, xtol=1e-16, low=1, maxiter=200, bisection=False, damping=1)
                # except:
#                high = None
#                if self.N == 1:
#                    try:
#                        high = self.Pc
#                    except:
#                        high = self.Pcs[0]
#                    high *= (1+1e-11)
                if not low and T < Tc:
                    low_bound = 1e8
                else:
                    if Tr > .3:
                        low_bound = 1.0
                    else:
                        low_bound = None
                P_disc = newton(discriminant_fun, P, fprime=True, xtol=4e-12, low=low_bound,
                                maxiter=80, bisection=False, damping=1)
                assert P_disc > 0 and not P_disc == 1
                if not low:
                    assert P_disc > low_bound
                break
            except:
                pass

        if not low:
            assert P_disc > low_bound


        global_iter += niter
        # for i in range(1000):
        #     a = 1

        if 0:
            try:
                P_disc = bisect(self._discriminant_at_T_mp, P_disc*(1-1e-8), P_disc*(1+1e-8), xtol=1e-18)
            except:
                try:
                    P_disc = bisect(self._discriminant_at_T_mp, P_disc*(1-1e-5), P_disc*(1+1e-5), xtol=1e-18)
                except:
                    try:
                        P_disc = bisect(self._discriminant_at_T_mp, P_disc*(1-1e-2), P_disc*(1+1e-2))
                    except:
                        pass

#        if not low:
#            P_disc_base = None
#            try:
#                if T < Tc:
#                    P_disc_base = self._P_discriminant_zero(True)
#            except:
#                pass
#            if P_disc_base is not None:
#                # pass
#               if isclose(P_disc_base, P_disc, rel_tol=1e-4):
#                   raise ValueError("Converged to wrong solution")


        return float(P_disc)


        # Can take a while to converge
        P_disc = secant(lambda P: self.discriminant(P=P), self.P, xtol=1e-7, low=1e-12, maxiter=200, bisection=True)
        if P_disc <= 0.0:
            P_disc = secant(lambda P: self.discriminant(P=P), self.P*100, xtol=1e-7, maxiter=200)
#            P_max = self.P*1000
#            P_disc = brenth(lambda P: self.discriminant(P=P), self.P*1e-3, P_max, rtol=1e-7, maxiter=200)
        return P_disc

    def _plot_T_discriminant_zero(self):
         Ts = logspace(log10(1), log10(1e4), 10000)
         errs = []
         for T in Ts:
             erri = self.discriminant(T=T)
#             if erri < 0:
#                 erri = -log10(abs(erri))
#             else:
#                 erri = log10(erri)
             errs.append(erri)
         import matplotlib.pyplot as plt
         plt.semilogx(Ts, errs, 'x')
#         plt.ylim((-1e-3, 1e-3))
         plt.show()

    def T_discriminant_zero_l(self, T_guess=None):
        r'''Method to calculate the temperature which zeros the discriminant
        function of the general cubic eos, and is likely to sit on a boundary
        between not having a liquid-like volume; and having a liquid-like volume.

        Parameters
        ----------
        T_guess : float, optional
            Temperature guess, [K]

        Returns
        -------
        T_discriminant_zero_l : float
            Temperature which make the discriminants zero at the right condition,
            [K]

        Notes
        -----
        Significant numerical issues remain in improving this method.

        Examples
        --------
        >>> eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
        >>> T_trans = eos.T_discriminant_zero_l()
        >>> T_trans
        644.3023307

        In this case, the discriminant transition does not reveal a transition
        to two roots being available, only to there being a double (imaginary)
        root.

        >>> eos.to(P=eos.P, T=T_trans).mpmath_volumes_float
        ((9.309597822372529e-05-0.00015876248805149625j), (9.309597822372529e-05+0.00015876248805149625j), (0.005064847204219234+0j))
        '''
        # Can also have one at g
        global niter
        niter = 0
        guesses = [100, 150, 200, 250, 300, 350, 400, 450]
        if T_guess is not None:
            guesses.append(T_guess)
        if self.N == 1:
            pass

        global_iter = 0
        for T in guesses:
            try:
                global_iter += niter
                niter = 0
                T_disc = secant(lambda T: self.discriminant(T=T), T, xtol=1e-10, low=1, maxiter=60, bisection=False, damping=1)
                assert T_disc > 0 and not T_disc == 1
                break
            except:
                pass
        global_iter += niter
        return T_disc

    def T_discriminant_zero_g(self, T_guess=None):
        r'''Method to calculate the temperature which zeros the discriminant
        function of the general cubic eos, and is likely to sit on a boundary
        between not having a vapor-like volume; and having a vapor-like volume.

        Parameters
        ----------
        T_guess : float, optional
            Temperature guess, [K]

        Returns
        -------
        T_discriminant_zero_g : float
            Temperature which make the discriminants zero at the right condition,
            [K]

        Notes
        -----
        Significant numerical issues remain in improving this method.

        Examples
        --------
        >>> eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
        >>> T_trans = eos.T_discriminant_zero_g()
        >>> T_trans
        644.3023307

        In this case, the discriminant transition does not reveal a transition
        to two roots being available, only to there being a double (imaginary)
        root.

        >>> eos.to(P=eos.P, T=T_trans).mpmath_volumes_float
        ((9.309597822372529e-05-0.00015876248805149625j), (9.309597822372529e-05+0.00015876248805149625j), (0.005064847204219234+0j))
        '''
        global niter
        niter = 0
        guesses = [700, 600, 500, 400, 300, 200]
        if T_guess is not None:
            guesses.append(T_guess)
        if self.N == 1:
            pass

        global_iter = 0
        for T in guesses:
            try:
                global_iter += niter
                niter = 0
                T_disc = secant(lambda T: self.discriminant(T=T), T, xtol=1e-10, low=1, maxiter=60, bisection=False, damping=1)
                assert T_disc > 0 and not T_disc == 1
                break
            except:
                pass
        global_iter += niter
        return T_disc

    def P_PIP_transition(self, T, low_P_limit=0.0):
        r'''Method to calculate the pressure which makes the phase
        identification parameter exactly 1. There are three regions for this
        calculation:

            * subcritical - PIP = 1 for the gas-like phase at P = 0
            * initially supercritical - PIP = 1 on a curve starting at the
              critical point, increasing for a while, decreasing for a while,
              and then curving sharply back to a zero pressure.
            * later supercritical - PIP = 1 for the liquid-like phase at P = 0

        Parameters
        ----------
        T : float
            Temperature for the calculation, [K]
        low_P_limit : float
            What value to return for the subcritical and later region, [Pa]

        Returns
        -------
        P : float
            Pressure which makes the PIP = 1, [Pa]

        Notes
        -----
        The transition between the region where this function returns values
        and the high temperature region that doesn't is the Joule-Thomson
        inversion point at a pressure of zero and can be directly solved for.

        Examples
        --------
        >>> eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
        >>> eos.P_PIP_transition(100)
        0.0
        >>> low_T = eos.to(T=100.0, P=eos.P_PIP_transition(100, low_P_limit=1e-5))
        >>> low_T.PIP_l, low_T.PIP_g
        (45.778088191, 0.9999999997903)
        >>> initial_super = eos.to(T=600.0, P=eos.P_PIP_transition(600))
        >>> initial_super.P, initial_super.PIP_g
        (6456282.17132, 0.999999999999)
        >>> high_T = eos.to(T=900.0, P=eos.P_PIP_transition(900, low_P_limit=1e-5))
        >>> high_T.P, high_T.PIP_g
        (12536704.763, 0.9999999999)
        '''
        try:
            subcritical = T < self.Tc
        except:
            subcritical = T < self.pseudo_Tc
        if subcritical:
            return low_P_limit
        else:
            def to_solve(P):
                e = self.to(T=T, P=P)
                # TODO: as all a_alpha is the same for all conditions, should be
                # able to derive a direct expression for this from the EOS which
                # only uses a volume solution
                # TODO: should be able to get the derivative of PIP w.r.t. pressure
                if hasattr(e, 'V_l'):
                    return e.PIP_l-1.0
                else:
                    return e.PIP_g-1.0
        try:
            # Near the critical point these equations turn extremely nasty!
            # bisection is the most reliable solver
            if subcritical:
                Psat = self.Psat(T)
                low, high = 10.0*Psat, Psat
            else:
                low, high = 1e-3, 1e11
            P = bisect(to_solve, low, high)
            return P
        except:
            err_low = to_solve(low_P_limit)
            if abs(err_low) < 1e-9:
                # Well above the critical point all solutions except the
                # zero-pressure limit have PIP values above 1
                # This corresponds to the JT inversion temperature at a
                # pressure of zero.
                return low_P_limit
            raise ValueError("Could not converge")



    def _V_g_extrapolated(self):
        P_pseudo_mc = sum([self.Pcs[i]*self.zs[i] for i in self.cmps])
        T_pseudo_mc = sum([(self.Tcs[i]*self.Tcs[j])**0.5*self.zs[j]*self.zs[i]
                           for i in self.cmps for j in self.cmps])
        V_pseudo_mc = (self.Zc*R*T_pseudo_mc)/P_pseudo_mc
        rho_pseudo_mc = 1.0/V_pseudo_mc

        P_disc = self.P_discriminant_zero_l()

        try:
            P_low = max(P_disc - 10.0, 1e-3)
            eos_low = self.to_TP_zs(T=self.T, P=P_low, zs=self.zs)
            rho_low = 1.0/eos_low.V_g
        except:
            P_low = max(P_disc + 10.0, 1e-3)
            eos_low = self.to_TP_zs(T=self.T, P=P_low, zs=self.zs)
            rho_low = 1.0/eos_low.V_g

        rho0 = (rho_low + 1.4*rho_pseudo_mc)*0.5

        dP_drho = eos_low.dP_drho_g
        rho1 = P_low*((rho_low - 1.4*rho_pseudo_mc) + P_low/dP_drho)

        rho2 = -P_low*P_low*((rho_low - 1.4*rho_pseudo_mc)*0.5 + P_low/dP_drho)
        rho_ans = rho0 + rho1/eos_low.P + rho2/(eos_low.P*eos_low.P)
        return 1.0/rho_ans

    @property
    def fugacity_l(self):
        r'''Fugacity for the liquid phase, [Pa].

        .. math::
            \text{fugacity} = P\exp\left(\frac{G_{dep}}{RT}\right)
        '''
        arg = self.G_dep_l*R_inv/self.T
        try:
            return self.P*exp(arg)
        except:
            return self.P*1e308

    @property
    def fugacity_g(self):
        r'''Fugacity for the gas phase, [Pa].

        .. math::
            \text{fugacity} = P\exp\left(\frac{G_{dep}}{RT}\right)
        '''
        arg = self.G_dep_g*R_inv/self.T
        try:
            return self.P*exp(arg)
        except:
            return self.P*1e308

    @property
    def phi_l(self):
        r'''Fugacity coefficient for the liquid phase, [Pa].

        .. math::
            \phi = \frac{\text{fugacity}}{P}
        '''
        arg = self.G_dep_l*R_inv/self.T
        try:
            return exp(arg)
        except:
            return 1e308

    @property
    def phi_g(self):
        r'''Fugacity coefficient for the gas phase, [Pa].

        .. math::
            \phi = \frac{\text{fugacity}}{P}
        '''
        arg = self.G_dep_g*R_inv/self.T
        try:
            return exp(arg)
        except:
            return 1e308

    @property
    def Cp_minus_Cv_l(self):
        r'''Cp - Cv for the liquid phase, [J/mol/K].

        .. math::
            C_p - C_v = -T\left(\frac{\partial P}{\partial T}\right)_V^2/
            \left(\frac{\partial P}{\partial V}\right)_T
        '''
        return -self.T*self.dP_dT_l*self.dP_dT_l*self.dV_dP_l

    @property
    def Cp_minus_Cv_g(self):
        r'''Cp - Cv for the gas phase, [J/mol/K].

        .. math::
            C_p - C_v = -T\left(\frac{\partial P}{\partial T}\right)_V^2/
            \left(\frac{\partial P}{\partial V}\right)_T
        '''
        return -self.T*self.dP_dT_g*self.dP_dT_g*self.dV_dP_g

    @property
    def beta_l(self):
        r'''Isobaric (constant-pressure) expansion coefficient for the liquid
        phase, [1/K].

        .. math::
            \beta = \frac{1}{V}\frac{\partial V}{\partial T}
        '''
        return self.dV_dT_l/self.V_l

    @property
    def beta_g(self):
        r'''Isobaric (constant-pressure) expansion coefficient for the gas
        phase, [1/K].

        .. math::
            \beta = \frac{1}{V}\frac{\partial V}{\partial T}
        '''
        return self.dV_dT_g/self.V_g

    @property
    def kappa_l(self):
        r'''Isothermal (constant-temperature) expansion coefficient for the liquid
        phase, [1/Pa].

        .. math::
            \kappa = \frac{-1}{V}\frac{\partial V}{\partial P}
        '''
        return -self.dV_dP_l/self.V_l

    @property
    def kappa_g(self):
        r'''Isothermal (constant-temperature) expansion coefficient for the gas
        phase, [1/Pa].

        .. math::
            \kappa = \frac{-1}{V}\frac{\partial V}{\partial P}
        '''
        return -self.dV_dP_g/self.V_g

    @property
    def V_dep_l(self):
        r'''Departure molar volume from ideal gas behavior for the liquid phase,
        [m^3/mol].

        .. math::
            V_{dep} = V - \frac{RT}{P}
        '''
        return self.V_l - self.T*R/self.P

    @property
    def V_dep_g(self):
        r'''Departure molar volume from ideal gas behavior for the gas phase,
        [m^3/mol].

        .. math::
            V_{dep} = V - \frac{RT}{P}
        '''
        return self.V_g - self.T*R/self.P

    @property
    def U_dep_l(self):
        r'''Departure molar internal energy from ideal gas behavior for the
        liquid phase, [J/mol].

        .. math::
            U_{dep} = H_{dep} - P V_{dep}
        '''
        return self.H_dep_l - self.P*(self.V_l - self.T*R/self.P)

    @property
    def U_dep_g(self):
        r'''Departure molar internal energy from ideal gas behavior for the
        gas phase, [J/mol].

        .. math::
            U_{dep} = H_{dep} - P V_{dep}
        '''
        return self.H_dep_g - self.P*(self.V_g - self.T*R/self.P)

    @property
    def A_dep_l(self):
        r'''Departure molar Helmholtz energy from ideal gas behavior for the
        liquid phase, [J/mol].

        .. math::
            A_{dep} = U_{dep} - T S_{dep}
        '''
        return self.H_dep_l - self.P*(self.V_l - self.T*R/self.P) - self.T*self.S_dep_l

    @property
    def A_dep_g(self):
        r'''Departure molar Helmholtz energy from ideal gas behavior for the
        gas phase, [J/mol].

        .. math::
            A_{dep} = U_{dep} - T S_{dep}
        '''
        return self.H_dep_g - self.P*(self.V_g - self.T*R/self.P) - self.T*self.S_dep_g

    @property
    def d2T_dPdV_l(self):
        r'''Second partial derivative of temperature with respect to
        pressure (constant volume) and then volume (constant pressure)
        for the liquid phase, [K*mol/(Pa*m^3)].

        .. math::
           \left(\frac{\partial^2 T}{\partial P\partial V}\right) =
            - \left[\left(\frac{\partial^2 P}{\partial T \partial V}\right)
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \right]\left(\frac{\partial P}{\partial T}\right)_V^{-3}

        '''
        inverse_dP_dT2 = self.dT_dP_l*self.dT_dP_l
        inverse_dP_dT3 = inverse_dP_dT2*self.dT_dP_l
        d2T_dPdV = -(self.d2P_dTdV_l*self.dP_dT_l - self.dP_dV_l*self.d2P_dT2_l)*inverse_dP_dT3
        return d2T_dPdV

    @property
    def d2T_dPdV_g(self):
        r'''Second partial derivative of temperature with respect to
        pressure (constant volume) and then volume (constant pressure)
        for the gas phase, [K*mol/(Pa*m^3)].

        .. math::
           \left(\frac{\partial^2 T}{\partial P\partial V}\right) =
            - \left[\left(\frac{\partial^2 P}{\partial T \partial V}\right)
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \right]\left(\frac{\partial P}{\partial T}\right)_V^{-3}

        '''
        inverse_dP_dT2 = self.dT_dP_g*self.dT_dP_g
        inverse_dP_dT3 = inverse_dP_dT2*self.dT_dP_g
        d2T_dPdV = -(self.d2P_dTdV_g*self.dP_dT_g - self.dP_dV_g*self.d2P_dT2_g)*inverse_dP_dT3
        return d2T_dPdV

    @property
    def d2V_dPdT_l(self):
        r'''Second partial derivative of volume with respect to
        pressure (constant temperature) and then presssure (constant temperature)
        for the liquid phase, [m^3/(K*Pa*mol)].

        .. math::
            \left(\frac{\partial^2 V}{\partial T\partial P}\right) =
            - \left[\left(\frac{\partial^2 P}{\partial T \partial V}\right)
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            \right]\left(\frac{\partial P}{\partial V}\right)_T^{-3}
        '''
        dV_dP = self.dV_dP_l
        return -(self.d2P_dTdV_l*self.dP_dV_l - self.dP_dT_l*self.d2P_dV2_l)*dV_dP*dV_dP*dV_dP

    @property
    def d2V_dPdT_g(self):
        r'''Second partial derivative of volume with respect to
        pressure (constant temperature) and then presssure (constant temperature)
        for the gas phase, [m^3/(K*Pa*mol)].

        .. math::
            \left(\frac{\partial^2 V}{\partial T\partial P}\right) =
            - \left[\left(\frac{\partial^2 P}{\partial T \partial V}\right)
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            \right]\left(\frac{\partial P}{\partial V}\right)_T^{-3}
        '''
        dV_dP = self.dV_dP_g
        return -(self.d2P_dTdV_g*self.dP_dV_g - self.dP_dT_g*self.d2P_dV2_g)*dV_dP*dV_dP*dV_dP

    @property
    def d2T_dP2_l(self):
        r'''Second partial derivative of temperature with respect to
        pressure (constant temperature) for the liquid phase, [K/Pa^2].

        .. math::
            \left(\frac{\partial^2 T}{\partial P^2}\right)_V = -\left(\frac{
            \partial^2 P}{\partial T^2}\right)_V \left(\frac{\partial P}{
            \partial T}\right)^{-3}_V

        '''
        dT_dP = self.dT_dP_l
        return -self.d2P_dT2_l*dT_dP*dT_dP*dT_dP # unused

    @property
    def d2T_dP2_g(self):
        r'''Second partial derivative of temperature with respect to
        pressure (constant volume) for the gas phase, [K/Pa^2].

        .. math::
            \left(\frac{\partial^2 T}{\partial P^2}\right)_V = -\left(\frac{
            \partial^2 P}{\partial T^2}\right)_V \left(\frac{\partial P}{
            \partial T}\right)^{-3}_V

        '''
        dT_dP = self.dT_dP_g
        return -self.d2P_dT2_g*dT_dP*dT_dP*dT_dP # unused

    @property
    def d2V_dP2_l(self):
        r'''Second partial derivative of volume with respect to
        pressure (constant temperature) for the liquid phase, [m^3/(Pa^2*mol)].

        .. math::
            \left(\frac{\partial^2 V}{\partial P^2}\right)_T = -\left(\frac{
            \partial^2 P}{\partial V^2}\right)_T \left(\frac{\partial P}{
            \partial V}\right)^{-3}_T

        '''
        dV_dP = self.dV_dP_l
        return -self.d2P_dV2_l*dV_dP*dV_dP*dV_dP

    @property
    def d2V_dP2_g(self):
        r'''Second partial derivative of volume with respect to
        pressure (constant temperature) for the gas phase, [m^3/(Pa^2*mol)].

        .. math::
            \left(\frac{\partial^2 V}{\partial P^2}\right)_T = -\left(\frac{
            \partial^2 P}{\partial V^2}\right)_T \left(\frac{\partial P}{
            \partial V}\right)^{-3}_T

        '''
        dV_dP = self.dV_dP_g
        return -self.d2P_dV2_g*dV_dP*dV_dP*dV_dP

    @property
    def d2T_dV2_l(self):
        r'''Second partial derivative of temperature with respect to
        volume (constant pressure) for the liquid phase, [K*mol^2/m^6].

        .. math::
            \left(\frac{\partial^2 T}{\partial V^2}\right)_P = -\left[
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) \right]
            \left(\frac{\partial P}{\partial T}\right)^{-2}_V
            + \left[\left(\frac{\partial^2 P}{\partial T\partial V}\right)
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V\right]
            \left(\frac{\partial P}{\partial T}\right)_V^{-3}
            \left(\frac{\partial P}{\partial V}\right)_T
        '''
        dT_dP = self.dT_dP_l
        dT_dP2 = dT_dP*dT_dP

        d2T_dV2 = dT_dP2*(-(self.d2P_dV2_l*self.dP_dT_l - self.dP_dV_l*self.d2P_dTdV_l)
                   +(self.d2P_dTdV_l*self.dP_dT_l - self.dP_dV_l*self.d2P_dT2_l)*dT_dP*self.dP_dV_l)
        return d2T_dV2

    @property
    def d2T_dV2_g(self):
        r'''Second partial derivative of temperature with respect to
        volume (constant pressure) for the gas phase, [K*mol^2/m^6].

        .. math::
            \left(\frac{\partial^2 T}{\partial V^2}\right)_P = -\left[
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) \right]
            \left(\frac{\partial P}{\partial T}\right)^{-2}_V
            + \left[\left(\frac{\partial^2 P}{\partial T\partial V}\right)
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V\right]
            \left(\frac{\partial P}{\partial T}\right)_V^{-3}
            \left(\frac{\partial P}{\partial V}\right)_T
        '''
        dT_dP = self.dT_dP_g
        dT_dP2 = dT_dP*dT_dP

        d2T_dV2 = dT_dP2*(-(self.d2P_dV2_g*self.dP_dT_g - self.dP_dV_g*self.d2P_dTdV_g)
                   +(self.d2P_dTdV_g*self.dP_dT_g - self.dP_dV_g*self.d2P_dT2_g)*dT_dP*self.dP_dV_g)
        return d2T_dV2


    @property
    def d2V_dT2_l(self):
        r'''Second partial derivative of volume with respect to
        temperature (constant pressure) for the liquid phase, [m^3/(mol*K^2)].

        .. math::
            \left(\frac{\partial^2 V}{\partial T^2}\right)_P = -\left[
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) \right]
            \left(\frac{\partial P}{\partial V}\right)^{-2}_T
            + \left[\left(\frac{\partial^2 P}{\partial T\partial V}\right)
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T\right]
            \left(\frac{\partial P}{\partial V}\right)_T^{-3}
            \left(\frac{\partial P}{\partial T}\right)_V

        '''
        dV_dP = self.dV_dP_l
        dP_dV = self.dP_dV_l
        d2P_dTdV = self.d2P_dTdV_l
        dP_dT = self.dP_dT_l
        d2V_dT2 = dV_dP*dV_dP*(-(self.d2P_dT2_l*dP_dV - dP_dT*d2P_dTdV) # unused
                   +(d2P_dTdV*dP_dV - dP_dT*self.d2P_dV2_l)*dV_dP*dP_dT)
        return d2V_dT2


    @property
    def d2V_dT2_g(self):
        r'''Second partial derivative of volume with respect to
        temperature (constant pressure) for the gas phase, [m^3/(mol*K^2)].

        .. math::
            \left(\frac{\partial^2 V}{\partial T^2}\right)_P = -\left[
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) \right]
            \left(\frac{\partial P}{\partial V}\right)^{-2}_T
            + \left[\left(\frac{\partial^2 P}{\partial T\partial V}\right)
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T\right]
            \left(\frac{\partial P}{\partial V}\right)_T^{-3}
            \left(\frac{\partial P}{\partial T}\right)_V

        '''
        dV_dP = self.dV_dP_g
        dP_dV = self.dP_dV_g
        d2P_dTdV = self.d2P_dTdV_g
        dP_dT = self.dP_dT_g
        d2V_dT2 = dV_dP*dV_dP*(-(self.d2P_dT2_g*dP_dV - dP_dT*d2P_dTdV) # unused
                   +(d2P_dTdV*dP_dV - dP_dT*self.d2P_dV2_g)*dV_dP*dP_dT)
        return d2V_dT2

    @property
    def Vc(self):
        r'''Critical volume, [m^3/mol].

        .. math::
            V_c = \frac{Z_c R T_c}{P_c}

        '''
        return self.Zc*R*self.Tc/self.Pc

    @property
    def rho_l(self):
        r'''Liquid molar density, [mol/m^3].

        .. math::
            \rho_l = \frac{1}{V_l}

        '''
        return 1.0/self.V_l

    @property
    def rho_g(self):
        r'''Gas molar density, [mol/m^3].

        .. math::
            \rho_g = \frac{1}{V_g}

        '''
        return 1.0/self.V_g


    @property
    def dZ_dT_l(self):
        r'''Derivative of compressibility factor with respect to temperature
        for the liquid phase, [1/K].

        .. math::
            \frac{\partial Z}{\partial T} = \frac{P}{RT}\left(
            \frac{\partial V}{\partial T} - \frac{V}{T}
            \right)

        '''
        T_inv = 1.0/self.T
        return self.P*R_inv*T_inv*(self.dV_dT_l - self.V_l*T_inv)

    @property
    def dZ_dT_g(self):
        r'''Derivative of compressibility factor with respect to temperature
        for the gas phase, [1/K].

        .. math::
            \frac{\partial Z}{\partial T} = \frac{P}{RT}\left(
            \frac{\partial V}{\partial T} - \frac{V}{T}
            \right)

        '''
        T_inv = 1.0/self.T
        return self.P*R_inv*T_inv*(self.dV_dT_g - self.V_g*T_inv)

    @property
    def dZ_dP_l(self):
        r'''Derivative of compressibility factor with respect to pressure
        for the liquid phase, [1/Pa].

        .. math::
            \frac{\partial Z}{\partial P} = \frac{1}{RT}\left(
            V - \frac{\partial V}{\partial P}
            \right)

        '''
        return (self.V_l + self.P*self.dV_dP_l)/(self.T*R)

    @property
    def dZ_dP_g(self):
        r'''Derivative of compressibility factor with respect to pressure
        for the gas phase, [1/Pa].

        .. math::
            \frac{\partial Z}{\partial P} = \frac{1}{RT}\left(
            V - \frac{\partial V}{\partial P}
            \right)

        '''
        return (self.V_g + self.P*self.dV_dP_g)/(self.T*R)

    d2V_dTdP_l = d2V_dPdT_l
    d2V_dTdP_g = d2V_dPdT_g
    d2T_dVdP_l = d2T_dPdV_l
    d2T_dVdP_g = d2T_dPdV_g

    @property
    def d2P_dVdT_l(self):
        '''Alias of :obj:`GCEOS.d2P_dTdV_l`'''
        return self.d2P_dTdV_l

    @property
    def d2P_dVdT_g(self):
        '''Alias of :obj:`GCEOS.d2P_dTdV_g`'''
        return self.d2P_dTdV_g

    @property
    def dP_drho_l(self):
        r'''Derivative of pressure with respect to molar density for the liquid
        phase, [Pa/(mol/m^3)].

        .. math::
            \frac{\partial P}{\partial \rho} = -V^2 \frac{\partial P}{\partial V}
        '''
        return -self.V_l*self.V_l*self.dP_dV_l

    @property
    def dP_drho_g(self):
        r'''Derivative of pressure with respect to molar density for the gas
        phase, [Pa/(mol/m^3)].

        .. math::
            \frac{\partial P}{\partial \rho} = -V^2 \frac{\partial P}{\partial V}
        '''
        return -self.V_g*self.V_g*self.dP_dV_g

    @property
    def drho_dP_l(self):
        r'''Derivative of molar density with respect to pressure for the liquid
        phase, [(mol/m^3)/Pa].

        .. math::
            \frac{\partial \rho}{\partial P} = \frac{-1}{V^2} \frac{\partial V}{\partial P}
        '''
        return -self.dV_dP_l/(self.V_l*self.V_l)

    @property
    def drho_dP_g(self):
        r'''Derivative of molar density with respect to pressure for the gas
        phase, [(mol/m^3)/Pa].

        .. math::
            \frac{\partial \rho}{\partial P} = \frac{-1}{V^2} \frac{\partial V}{\partial P}
        '''
        return -self.dV_dP_g/(self.V_g*self.V_g)

    @property
    def d2P_drho2_l(self):
        r'''Second derivative of pressure with respect to molar density for the
        liquid phase, [Pa/(mol/m^3)^2].

        .. math::
            \frac{\partial^2 P}{\partial \rho^2} = -V^2\left(
            -V^2\frac{\partial^2 P}{\partial V^2} - 2V \frac{\partial P}{\partial V}
            \right)
        '''
        return -self.V_l**2*(-self.V_l**2*self.d2P_dV2_l - 2*self.V_l*self.dP_dV_l)

    @property
    def d2P_drho2_g(self):
        r'''Second derivative of pressure with respect to molar density for the
        gas phase, [Pa/(mol/m^3)^2].

        .. math::
            \frac{\partial^2 P}{\partial \rho^2} = -V^2\left(
            -V^2\frac{\partial^2 P}{\partial V^2} - 2V \frac{\partial P}{\partial V}
            \right)
        '''
        return -self.V_g**2*(-self.V_g**2*self.d2P_dV2_g - 2*self.V_g*self.dP_dV_g)

    @property
    def d2rho_dP2_l(self):
        r'''Second derivative of molar density with respect to pressure for the
        liquid phase, [(mol/m^3)/Pa^2].

        .. math::
            \frac{\partial^2 \rho}{\partial P^2} =
            -\frac{\partial^2 V}{\partial P^2}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial P}\right)^2\frac{1}{V^3}
        '''
        return -self.d2V_dP2_l/self.V_l**2 + 2*self.dV_dP_l**2/self.V_l**3

    @property
    def d2rho_dP2_g(self):
        r'''Second derivative of molar density with respect to pressure for the
        gas phase, [(mol/m^3)/Pa^2].

        .. math::
            \frac{\partial^2 \rho}{\partial P^2} =
            -\frac{\partial^2 V}{\partial P^2}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial P}\right)^2\frac{1}{V^3}
        '''
        return -self.d2V_dP2_g/self.V_g**2 + 2*self.dV_dP_g**2/self.V_g**3


    @property
    def dT_drho_l(self):
        r'''Derivative of temperature with respect to molar density for the
        liquid phase, [K/(mol/m^3)].

        .. math::
            \frac{\partial T}{\partial \rho} = V^2 \frac{\partial T}{\partial V}
        '''
        return -self.V_l*self.V_l*self.dT_dV_l

    @property
    def dT_drho_g(self):
        r'''Derivative of temperature with respect to molar density for the
        gas phase, [K/(mol/m^3)].

        .. math::
            \frac{\partial T}{\partial \rho} = V^2 \frac{\partial T}{\partial V}
        '''
        return -self.V_g*self.V_g*self.dT_dV_g

    @property
    def d2T_drho2_l(self):
        r'''Second derivative of temperature with respect to molar density for
        the liquid phase, [K/(mol/m^3)^2].

        .. math::
            \frac{\partial^2 T}{\partial \rho^2} =
            -V^2(-V^2 \frac{\partial^2 T}{\partial V^2} -2V \frac{\partial T}{\partial V}  )
        '''
        return -self.V_l**2*(-self.V_l**2*self.d2T_dV2_l - 2*self.V_l*self.dT_dV_l)

    @property
    def d2T_drho2_g(self):
        r'''Second derivative of temperature with respect to molar density for
        the gas phase, [K/(mol/m^3)^2].

        .. math::
            \frac{\partial^2 T}{\partial \rho^2} =
            -V^2(-V^2 \frac{\partial^2 T}{\partial V^2} -2V \frac{\partial T}{\partial V}  )
        '''
        return -self.V_g**2*(-self.V_g**2*self.d2T_dV2_g - 2*self.V_g*self.dT_dV_g)


    @property
    def drho_dT_l(self):
        r'''Derivative of molar density with respect to temperature for the
        liquid phase, [(mol/m^3)/K].

        .. math::
            \frac{\partial \rho}{\partial T} = - \frac{1}{V^2}
            \frac{\partial V}{\partial T}
        '''
        return -self.dV_dT_l/(self.V_l*self.V_l)

    @property
    def drho_dT_g(self):
        r'''Derivative of molar density with respect to temperature for the
        gas phase, [(mol/m^3)/K].

        .. math::
            \frac{\partial \rho}{\partial T} = - \frac{1}{V^2}
            \frac{\partial V}{\partial T}
        '''
        return -self.dV_dT_g/(self.V_g*self.V_g)

    @property
    def d2rho_dT2_l(self):
        r'''Second derivative of molar density with respect to temperature for
        the liquid phase, [(mol/m^3)/K^2].

        .. math::
            \frac{\partial^2 \rho}{\partial T^2} =
            -\frac{\partial^2 V}{\partial T^2}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial T}\right)^2\frac{1}{V^3}
        '''
        return -self.d2V_dT2_l/self.V_l**2 + 2*self.dV_dT_l**2/self.V_l**3

    @property
    def d2rho_dT2_g(self):
        r'''Second derivative of molar density with respect to temperature for
        the gas phase, [(mol/m^3)/K^2].

        .. math::
            \frac{\partial^2 \rho}{\partial T^2} =
            -\frac{\partial^2 V}{\partial T^2}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial T}\right)^2\frac{1}{V^3}
        '''
        return -self.d2V_dT2_g/self.V_g**2 + 2*self.dV_dT_g**2/self.V_g**3

    @property
    def d2P_dTdrho_l(self):
        r'''Derivative of pressure with respect to molar density, and
        temperature for the liquid phase, [Pa/(K*mol/m^3)].

        .. math::
            \frac{\partial^2 P}{\partial \rho\partial T}
            = -V^2 \frac{\partial^2 P}{\partial T \partial V}
        '''
        return -(self.V_l*self.V_l)*self.d2P_dTdV_l

    @property
    def d2P_dTdrho_g(self):
        r'''Derivative of pressure with respect to molar density, and
        temperature for the gas phase, [Pa/(K*mol/m^3)].

        .. math::
            \frac{\partial^2 P}{\partial \rho\partial T}
            = -V^2 \frac{\partial^2 P}{\partial T \partial V}
        '''
        return -(self.V_g*self.V_g)*self.d2P_dTdV_g

    @property
    def d2T_dPdrho_l(self):
        r'''Derivative of temperature with respect to molar density, and
        pressure for the liquid phase, [K/(Pa*mol/m^3)].

        .. math::
            \frac{\partial^2 T}{\partial \rho\partial P}
            = -V^2 \frac{\partial^2 T}{\partial P \partial V}
        '''
        return -(self.V_l*self.V_l)*self.d2T_dPdV_l

    @property
    def d2T_dPdrho_g(self):
        r'''Derivative of temperature with respect to molar density, and
        pressure for the gas phase, [K/(Pa*mol/m^3)].

        .. math::
            \frac{\partial^2 T}{\partial \rho\partial P}
            = -V^2 \frac{\partial^2 T}{\partial P \partial V}
        '''
        return -(self.V_g*self.V_g)*self.d2T_dPdV_g

    @property
    def d2rho_dPdT_l(self):
        r'''Second derivative of molar density with respect to pressure
        and temperature for the liquid phase, [(mol/m^3)/(K*Pa)].

        .. math::
            \frac{\partial^2 \rho}{\partial T \partial P} =
            -\frac{\partial^2 V}{\partial T \partial P}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial T}\right)
            \left(\frac{\partial V}{\partial P}\right)
            \frac{1}{V^3}
        '''
        return -self.d2V_dPdT_l/self.V_l**2 + 2*self.dV_dT_l*self.dV_dP_l/self.V_l**3

    @property
    def d2rho_dPdT_g(self):
        r'''Second derivative of molar density with respect to pressure
        and temperature for the gas phase, [(mol/m^3)/(K*Pa)].

        .. math::
            \frac{\partial^2 \rho}{\partial T \partial P} =
            -\frac{\partial^2 V}{\partial T \partial P}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial T}\right)
            \left(\frac{\partial V}{\partial P}\right)
            \frac{1}{V^3}
        '''
        return -self.d2V_dPdT_g/self.V_g**2 + 2*self.dV_dT_g*self.dV_dP_g/self.V_g**3

    @property
    def dH_dep_dT_l(self):
        r'''Derivative of departure enthalpy with respect to
        temperature for the liquid phase, [(J/mol)/K].

        .. math::
            \frac{\partial H_{dep, l}}{\partial T} = P \frac{d}{d T} V{\left (T
            \right )} - R + \frac{2 T}{\sqrt{\delta^{2} - 4 \epsilon}}
                \operatorname{atanh}{\left (\frac{\delta + 2 V{\left (T \right
                )}}{\sqrt{\delta^{2} - 4 \epsilon}} \right )} \frac{d^{2}}{d
                T^{2}}  \operatorname{a \alpha}{\left (T \right )} + \frac{4
                \left(T \frac{d}{d T} \operatorname{a \alpha}{\left (T \right
                )} - \operatorname{a \alpha}{\left (T \right )}\right) \frac{d}
                {d T} V{\left (T \right )}}{\left(\delta^{2} - 4 \epsilon
                \right) \left(- \frac{\left(\delta + 2 V{\left (T \right )}
                \right)^{2}}{\delta^{2} - 4 \epsilon} + 1\right)}
        '''
        x0 = self.V_l
        x1 = self.dV_dT_l
        x2 = self.a_alpha
        x3 = self.delta*self.delta - 4.0*self.epsilon
        if x3 == 0.0:
            x3 = 1e-100

        x4 = x3**-0.5
        x5 = self.delta + x0 + x0
        x6 = 1.0/x3
        return (self.P*x1 - R + 2.0*self.T*x4*catanh(x4*x5).real*self.d2a_alpha_dT2
                - 4.0*x1*x6*(self.T*self.da_alpha_dT - x2)/(x5*x5*x6 - 1.0))

    @property
    def dH_dep_dT_g(self):
        r'''Derivative of departure enthalpy with respect to
        temperature for the gas phase, [(J/mol)/K].

        .. math::
            \frac{\partial H_{dep, g}}{\partial T} = P \frac{d}{d T} V{\left (T
            \right )} - R + \frac{2 T}{\sqrt{\delta^{2} - 4 \epsilon}}
                \operatorname{atanh}{\left (\frac{\delta + 2 V{\left (T \right
                )}}{\sqrt{\delta^{2} - 4 \epsilon}} \right )} \frac{d^{2}}{d
                T^{2}}  \operatorname{a \alpha}{\left (T \right )} + \frac{4
                \left(T \frac{d}{d T} \operatorname{a \alpha}{\left (T \right
                )} - \operatorname{a \alpha}{\left (T \right )}\right) \frac{d}
                {d T} V{\left (T \right )}}{\left(\delta^{2} - 4 \epsilon
                \right) \left(- \frac{\left(\delta + 2 V{\left (T \right )}
                \right)^{2}}{\delta^{2} - 4 \epsilon} + 1\right)}
        '''
        x0 = self.V_g
        x1 = self.dV_dT_g
        if x0 > 1e50:
            if isinf(self.dV_dT_g) or self.H_dep_g == 0.0:
                return 0.0
        x2 = self.a_alpha
        x3 = self.delta*self.delta - 4.0*self.epsilon
        if x3 == 0.0:
            x3 = 1e-100
        x4 = x3**-0.5
        x5 = self.delta + x0 + x0
        x6 = 1.0/x3
        return (self.P*x1 - R + 2.0*self.T*x4*catanh(x4*x5).real*self.d2a_alpha_dT2
                - 4.0*x1*x6*(self.T*self.da_alpha_dT - x2)/(x5*x5*x6 - 1.0))

    @property
    def dH_dep_dT_l_V(self):
        r'''Derivative of departure enthalpy with respect to
        temperature at constant volume for the liquid phase, [(J/mol)/K].

        .. math::
            \left(\frac{\partial H_{dep, l}}{\partial T}\right)_{V} =
            - R + \frac{2 T
            \operatorname{atanh}{\left(\frac{2 V_l + \delta}{\sqrt{\delta^{2}
            - 4 \epsilon}} \right)} \frac{d^{2}}{d T^{2}} \operatorname{
            a_{\alpha}}{\left(T \right)}}{\sqrt{\delta^{2} - 4 \epsilon}}
            + V_l \frac{\partial}{\partial T} P{\left(T,V \right)}
        '''
        T = self.T
        delta, epsilon = self.delta, self.epsilon
        V = self.V_l
        dP_dT = self.dP_dT_l
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100
        return -R + 2.0*T*x0*catanh(x0*(V + V + delta)).real*self.d2a_alpha_dT2 + V*dP_dT

    @property
    def dH_dep_dT_g_V(self):
        r'''Derivative of departure enthalpy with respect to
        temperature at constant volume for the gas phase, [(J/mol)/K].

        .. math::
            \left(\frac{\partial H_{dep, g}}{\partial T}\right)_{V} =
            - R + \frac{2 T
            \operatorname{atanh}{\left(\frac{2 V_g + \delta}{\sqrt{\delta^{2}
            - 4 \epsilon}} \right)} \frac{d^{2}}{d T^{2}} \operatorname{
                a_{\alpha}}{\left(T \right)}}{\sqrt{\delta^{2} - 4 \epsilon}}
                + V_g \frac{\partial}{\partial T} P{\left(T,V \right)}
        '''
        T = self.T
        delta, epsilon = self.delta, self.epsilon
        V = self.V_g
        dP_dT = self.dP_dT_g
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100
        return -R + 2.0*T*x0*catanh(x0*(V + V + delta)).real*self.d2a_alpha_dT2 + V*dP_dT

    @property
    def dH_dep_dP_l(self):
        r'''Derivative of departure enthalpy with respect to
        pressure for the liquid phase, [(J/mol)/Pa].

        .. math::
            \frac{\partial H_{dep, l}}{\partial P} = P \frac{d}{d P} V{\left (P
            \right )} + V{\left (P \right )} + \frac{4 \left(T \frac{d}{d T}
            \operatorname{a \alpha}{\left (T \right )} - \operatorname{a
            \alpha}{\left (T \right )}\right) \frac{d}{d P} V{\left (P \right
            )}}{\left(\delta^{2} - 4 \epsilon\right) \left(- \frac{\left(\delta
            + 2 V{\left (P \right )}\right)^{2}}{\delta^{2} - 4 \epsilon}
            + 1\right)}
        '''
        delta = self.delta
        x0 = self.V_l
        x2 = delta*delta - 4.0*self.epsilon
        x4 = (delta + x0 + x0)
        return (x0 + self.dV_dP_l*(self.P - 4.0*(self.T*self.da_alpha_dT
                - self.a_alpha)/(x4*x4 - x2)))

    @property
    def dH_dep_dP_g(self):
        r'''Derivative of departure enthalpy with respect to
        pressure for the gas phase, [(J/mol)/Pa].

        .. math::
            \frac{\partial H_{dep, g}}{\partial P} = P \frac{d}{d P} V{\left (P
            \right )} + V{\left (P \right )} + \frac{4 \left(T \frac{d}{d T}
            \operatorname{a \alpha}{\left (T \right )} - \operatorname{a
            \alpha}{\left (T \right )}\right) \frac{d}{d P} V{\left (P \right
            )}}{\left(\delta^{2} - 4 \epsilon\right) \left(- \frac{\left(\delta
            + 2 V{\left (P \right )}\right)^{2}}{\delta^{2} - 4 \epsilon}
            + 1\right)}
        '''
        delta = self.delta
        x0 = self.V_g
        x2 = delta*delta - 4.0*self.epsilon
        x4 = (delta + x0 + x0)
#        if isinf(self.dV_dP_g):
            # This does not appear to be correct
#            return 0.0
        return (x0 + self.dV_dP_g*(self.P - 4.0*(self.T*self.da_alpha_dT
                - self.a_alpha)/(x4*x4 - x2)))

    @property
    def dH_dep_dP_l_V(self):
        r'''Derivative of departure enthalpy with respect to
        pressure at constant volume for the gas phase, [(J/mol)/Pa].

        .. math::
            \left(\frac{\partial H_{dep, g}}{\partial P}\right)_{V} =
            - R \left(\frac{\partial T}{\partial P}\right)_V + V + \frac{2 \left(T
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}
            + \left(\frac{\partial a \alpha}{\partial T}\right)_P
            \left(\frac{\partial T}{\partial P}\right)_V - \left(\frac{
            \partial a \alpha}{\partial P}\right)_{V} \right)
            \operatorname{atanh}{\left(\frac{2 V + \delta}
            {\sqrt{\delta^{2} - 4 \epsilon}} \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}}
        '''
        T, V, delta, epsilon = self.T, self.V_l, self.delta, self.epsilon
        da_alpha_dT, d2a_alpha_dT2 = self.da_alpha_dT, self.d2a_alpha_dT2
        dT_dP = self.dT_dP_l

        d2a_alpha_dTdP_V = d2a_alpha_dT2*dT_dP
        da_alpha_dP_V = da_alpha_dT*dT_dP
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100

        return (-R*dT_dP + V + 2.0*x0*(
                T*d2a_alpha_dTdP_V + dT_dP*da_alpha_dT - da_alpha_dP_V)
                *catanh(x0*(V + V + delta)).real)

    @property
    def dH_dep_dP_g_V(self):
        r'''Derivative of departure enthalpy with respect to
        pressure at constant volume for the liquid phase, [(J/mol)/Pa].

        .. math::
            \left(\frac{\partial H_{dep, g}}{\partial P}\right)_{V} =
            - R \left(\frac{\partial T}{\partial P}\right)_V + V + \frac{2 \left(T
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}
            + \left(\frac{\partial a \alpha}{\partial T}\right)_P
            \left(\frac{\partial T}{\partial P}\right)_V - \left(\frac{
            \partial a \alpha}{\partial P}\right)_{V} \right)
            \operatorname{atanh}{\left(\frac{2 V + \delta}
            {\sqrt{\delta^{2} - 4 \epsilon}} \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}}
        '''
        T, V, delta, epsilon = self.T, self.V_g, self.delta, self.epsilon
        da_alpha_dT, d2a_alpha_dT2 = self.da_alpha_dT, self.d2a_alpha_dT2
        dT_dP = self.dT_dP_g

        d2a_alpha_dTdP_V = d2a_alpha_dT2*dT_dP
        da_alpha_dP_V = da_alpha_dT*dT_dP
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100

        return (-R*dT_dP + V + 2.0*x0*(
                T*d2a_alpha_dTdP_V + dT_dP*da_alpha_dT - da_alpha_dP_V)
                *catanh(x0*(V + V + delta)).real)

    @property
    def dH_dep_dV_g_T(self):
        r'''Derivative of departure enthalpy with respect to
        volume at constant temperature for the gas phase, [J/m^3].

        .. math::
            \left(\frac{\partial H_{dep, g}}{\partial V}\right)_{T} =
            \left(\frac{\partial H_{dep, g}}{\partial P}\right)_{T} \cdot
            \left(\frac{\partial P}{\partial V}\right)_{T}
        '''
        return self.dH_dep_dP_g*self.dP_dV_g

    @property
    def dH_dep_dV_l_T(self):
        r'''Derivative of departure enthalpy with respect to
        volume at constant temperature for the gas phase, [J/m^3].

        .. math::
            \left(\frac{\partial H_{dep, l}}{\partial V}\right)_{T} =
            \left(\frac{\partial H_{dep, l}}{\partial P}\right)_{T} \cdot
            \left(\frac{\partial P}{\partial V}\right)_{T}
        '''
        return self.dH_dep_dP_l*self.dP_dV_l

    @property
    def dH_dep_dV_g_P(self):
        r'''Derivative of departure enthalpy with respect to
        volume at constant pressure for the gas phase, [J/m^3].

        .. math::
            \left(\frac{\partial H_{dep, g}}{\partial V}\right)_{P} =
            \left(\frac{\partial H_{dep, g}}{\partial T}\right)_{P} \cdot
            \left(\frac{\partial T}{\partial V}\right)_{P}
        '''
        return self.dH_dep_dT_g*self.dT_dV_g

    @property
    def dH_dep_dV_l_P(self):
        r'''Derivative of departure enthalpy with respect to
        volume at constant pressure for the liquid phase, [J/m^3].

        .. math::
            \left(\frac{\partial H_{dep, l}}{\partial V}\right)_{P} =
            \left(\frac{\partial H_{dep, l}}{\partial T}\right)_{P} \cdot
            \left(\frac{\partial T}{\partial V}\right)_{P}
        '''
        return self.dH_dep_dT_l*self.dT_dV_l

    @property
    def dS_dep_dT_l(self):
        r'''Derivative of departure entropy with respect to
        temperature for the liquid phase, [(J/mol)/K^2].

        .. math::
            \frac{\partial S_{dep, l}}{\partial T} = - \frac{R \frac{d}{d T}
            V{\left (T \right )}}{V{\left (T \right )}} + \frac{R \frac{d}{d T}
            V{\left (T \right )}}{- b + V{\left (T \right )}} + \frac{4
            \frac{d}{d T} V{\left (T \right )} \frac{d}{d T} \operatorname{a
            \alpha}{\left (T \right )}}{\left(\delta^{2} - 4 \epsilon\right)
            \left(- \frac{\left(\delta + 2 V{\left (T \right )}\right)^{2}}
            {\delta^{2} - 4 \epsilon} + 1\right)} + \frac{2 \frac{d^{2}}{d
            T^{2}}  \operatorname{a \alpha}{\left (T \right )}}
            {\sqrt{\delta^{2} - 4 \epsilon}} \operatorname{atanh}{\left (\frac{
            \delta + 2 V{\left (T \right )}}{\sqrt{\delta^{2} - 4 \epsilon}}
            \right )} + \frac{R^{2} T}{P V{\left (T \right )}} \left(\frac{P}
            {R T} \frac{d}{d T} V{\left (T \right )} - \frac{P}{R T^{2}}
            V{\left (T \right )}\right)
        '''
        x0 = self.V_l
        x1 = 1./x0
        x2 = self.dV_dT_l
        x3 = R*x2
        x4 = self.a_alpha
        x5 = self.delta*self.delta - 4.0*self.epsilon
        if x5 == 0.0:
            x5 = 1e-100
        x6 = x5**-0.5
        x7 = self.delta + 2.0*x0
        x8 = 1.0/x5
        return (R*x1*(x2 - x0/self.T) - x1*x3 - 4.0*x2*x8*self.da_alpha_dT
                /(x7*x7*x8 - 1.0) - x3/(self.b - x0)
                + 2.0*x6*catanh(x6*x7).real*self.d2a_alpha_dT2)

    @property
    def dS_dep_dT_g(self):
        r'''Derivative of departure entropy with respect to
        temperature for the gas phase, [(J/mol)/K^2].

        .. math::
            \frac{\partial S_{dep, g}}{\partial T} = - \frac{R \frac{d}{d T}
            V{\left (T \right )}}{V{\left (T \right )}} + \frac{R \frac{d}{d T}
            V{\left (T \right )}}{- b + V{\left (T \right )}} + \frac{4
            \frac{d}{d T} V{\left (T \right )} \frac{d}{d T} \operatorname{a
            \alpha}{\left (T \right )}}{\left(\delta^{2} - 4 \epsilon\right)
            \left(- \frac{\left(\delta + 2 V{\left (T \right )}\right)^{2}}
            {\delta^{2} - 4 \epsilon} + 1\right)} + \frac{2 \frac{d^{2}}{d
            T^{2}}  \operatorname{a \alpha}{\left (T \right )}}
            {\sqrt{\delta^{2} - 4 \epsilon}} \operatorname{atanh}{\left (\frac{
            \delta + 2 V{\left (T \right )}}{\sqrt{\delta^{2} - 4 \epsilon}}
            \right )} + \frac{R^{2} T}{P V{\left (T \right )}} \left(\frac{P}
            {R T} \frac{d}{d T} V{\left (T \right )} - \frac{P}{R T^{2}}
            V{\left (T \right )}\right)
        '''
        x0 = self.V_g
        if x0 > 1e50:
            if self.S_dep_g == 0.0:
                return 0.0
        x1 = 1./x0
        x2 = self.dV_dT_g
        if isinf(x2):
            return 0.0
        x3 = R*x2
        x4 = self.a_alpha

        x5 = self.delta*self.delta - 4.0*self.epsilon
        if x5 == 0.0:
            x5 = 1e-100
        x6 = x5**-0.5
        x7 = self.delta + 2.0*x0
        x8 = 1.0/x5
        return (R*x1*(x2 - x0/self.T) - x1*x3 - 4.0*x2*x8*self.da_alpha_dT
                /(x7*x7*x8 - 1.0) - x3/(self.b - x0)
                + 2.0*x6*catanh(x6*x7).real*self.d2a_alpha_dT2)

    @property
    def dS_dep_dT_l_V(self):
        r'''Derivative of departure entropy with respect to
        temperature at constant volume for the liquid phase, [(J/mol)/K^2].

        .. math::
            \left(\frac{\partial S_{dep, l}}{\partial T}\right)_{V} =
            \frac{R^{2} T \left(\frac{V \frac{\partial}{\partial T} P{\left(T,V
            \right)}}{R T} - \frac{V P{\left(T,V \right)}}{R T^{2}}\right)}{
            V P{\left(T,V \right)}} + \frac{2 \operatorname{atanh}{\left(
            \frac{2 V + \delta}{\sqrt{\delta^{2} - 4 \epsilon}} \right)}
            \frac{d^{2}}{d T^{2}} \operatorname{a \alpha}{\left(T \right)}}
            {\sqrt{\delta^{2} - 4 \epsilon}}
        '''
        T, P = self.T, self.P
        delta, epsilon = self.delta, self.epsilon
        V = self.V_l
        dP_dT = self.dP_dT_l
        try:
            x1 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x1 = 1e100
        return (R*(dP_dT/P - 1.0/T) + 2.0*x1*catanh(x1*(V + V + delta)).real*self.d2a_alpha_dT2)

    @property
    def dS_dep_dT_g_V(self):
        r'''Derivative of departure entropy with respect to
        temperature at constant volume for the gas phase, [(J/mol)/K^2].

        .. math::
            \left(\frac{\partial S_{dep, g}}{\partial T}\right)_{V} =
            \frac{R^{2} T \left(\frac{V \frac{\partial}{\partial T} P{\left(T,V
            \right)}}{R T} - \frac{V P{\left(T,V \right)}}{R T^{2}}\right)}{
            V P{\left(T,V \right)}} + \frac{2 \operatorname{atanh}{\left(
            \frac{2 V + \delta}{\sqrt{\delta^{2} - 4 \epsilon}} \right)}
            \frac{d^{2}}{d T^{2}} \operatorname{a \alpha}{\left(T \right)}}
            {\sqrt{\delta^{2} - 4 \epsilon}}
        '''
        T, P = self.T, self.P
        delta, epsilon = self.delta, self.epsilon
        V = self.V_g
        dP_dT = self.dP_dT_g
        try:
            x1 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x1 = 1e100
        return (R*(dP_dT/P - 1.0/T) + 2.0*x1*catanh(x1*(V + V + delta)).real*self.d2a_alpha_dT2)

    @property
    def dS_dep_dP_l(self):
        r'''Derivative of departure entropy with respect to
        pressure for the liquid phase, [(J/mol)/K/Pa].

        .. math::
            \frac{\partial S_{dep, l}}{\partial P} = - \frac{R \frac{d}{d P}
            V{\left (P \right )}}{V{\left (P \right )}} + \frac{R \frac{d}{d P}
            V{\left (P \right )}}{- b + V{\left (P \right )}} + \frac{4 \frac{
            d}{d P} V{\left (P \right )} \frac{d}{d T} \operatorname{a \alpha}
            {\left (T \right )}}{\left(\delta^{2} - 4 \epsilon\right) \left(
            - \frac{\left(\delta + 2 V{\left (P \right )}\right)^{2}}{
            \delta^{2} - 4 \epsilon} + 1\right)} + \frac{R^{2} T}{P V{\left (P
            \right )}} \left(\frac{P}{R T} \frac{d}{d P} V{\left (P \right )}
            + \frac{V{\left (P \right )}}{R T}\right)
        '''
        x0 = self.V_l
        x1 = 1.0/x0
        x2 = self.dV_dP_l
        x3 = R*x2
        try:
            x4 = 1.0/(self.delta*self.delta - 4.0*self.epsilon)
        except ZeroDivisionError:
            x4 = 1e50
        return (-x1*x3 - 4.0*x2*x4*self.da_alpha_dT/(x4*(self.delta + 2*x0)**2
                - 1) - x3/(self.b - x0) + R*x1*(self.P*x2 + x0)/self.P)

    @property
    def dS_dep_dP_g(self):
        r'''Derivative of departure entropy with respect to
        pressure for the gas phase, [(J/mol)/K/Pa].

        .. math::
            \frac{\partial S_{dep, g}}{\partial P} = - \frac{R \frac{d}{d P}
            V{\left (P \right )}}{V{\left (P \right )}} + \frac{R \frac{d}{d P}
            V{\left (P \right )}}{- b + V{\left (P \right )}} + \frac{4 \frac{
            d}{d P} V{\left (P \right )} \frac{d}{d T} \operatorname{a \alpha}
            {\left (T \right )}}{\left(\delta^{2} - 4 \epsilon\right) \left(
            - \frac{\left(\delta + 2 V{\left (P \right )}\right)^{2}}{
            \delta^{2} - 4 \epsilon} + 1\right)} + \frac{R^{2} T}{P V{\left (P
            \right )}} \left(\frac{P}{R T} \frac{d}{d P} V{\left (P \right )}
            + \frac{V{\left (P \right )}}{R T}\right)
        '''
        x0 = self.V_g
        x1 = 1.0/x0
        x2 = self.dV_dP_g
        x3 = R*x2
        try:
            x4 = 1.0/(self.delta*self.delta - 4.0*self.epsilon)
        except ZeroDivisionError:
            x4 = 1e200
        ans = (-x1*x3 - 4.0*x2*x4*self.da_alpha_dT/(x4*(self.delta + 2*x0)**2
                - 1) - x3/(self.b - x0) + R*x1*(self.P*x2 + x0)/self.P)
        return ans

    @property
    def dS_dep_dP_g_V(self):
        r'''Derivative of departure entropy with respect to
        pressure at constant volume for the gas phase, [(J/mol)/K/Pa].

        .. math::
            \left(\frac{\partial S_{dep, g}}{\partial P}\right)_{V} =
            \frac{2 \operatorname{atanh}{\left(\frac{2 V + \delta}{
            \sqrt{\delta^{2} - 4 \epsilon}} \right)}
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}}{\sqrt{\delta^{2} - 4 \epsilon}}
            + \frac{R^{2} \left(- \frac{P V \frac{d}{d P} T{\left(P \right)}}
            {R T^{2}{\left(P \right)}}
             + \frac{V}{R T{\left(P \right)}}\right) T{\left(P \right)}}{P V}
        '''
        T, P, delta, epsilon = self.T, self.P, self.delta, self.epsilon
        d2a_alpha_dT2 = self.d2a_alpha_dT2
        V, dT_dP = self.V_g, self.dT_dP_g
        d2a_alpha_dTdP_V = d2a_alpha_dT2*dT_dP
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100
        return (2.0*x0*catanh(x0*(V + V + delta)).real*d2a_alpha_dTdP_V
                - R*(P*dT_dP/T - 1.0)/P)

    @property
    def dS_dep_dP_l_V(self):
        r'''Derivative of departure entropy with respect to
        pressure at constant volume for the liquid phase, [(J/mol)/K/Pa].

        .. math::
            \left(\frac{\partial S_{dep, l}}{\partial P}\right)_{V} =
            \frac{2 \operatorname{atanh}{\left(\frac{2 V + \delta}{
            \sqrt{\delta^{2} - 4 \epsilon}} \right)}
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}}{\sqrt{\delta^{2} - 4 \epsilon}}
            + \frac{R^{2} \left(- \frac{P V \frac{d}{d P} T{\left(P \right)}}
            {R T^{2}{\left(P \right)}}
             + \frac{V}{R T{\left(P \right)}}\right) T{\left(P \right)}}{P V}
        '''
        T, P, delta, epsilon = self.T, self.P, self.delta, self.epsilon
        d2a_alpha_dT2 = self.d2a_alpha_dT2
        V, dT_dP = self.V_l, self.dT_dP_l
        d2a_alpha_dTdP_V = d2a_alpha_dT2*dT_dP
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100
        return (2.0*x0*catanh(x0*(V + V + delta)).real*d2a_alpha_dTdP_V
                - R*(P*dT_dP/T - 1.0)/P)

    @property
    def dS_dep_dV_g_T(self):
        r'''Derivative of departure entropy with respect to
        volume at constant temperature for the gas phase, [J/K/m^3].

        .. math::
            \left(\frac{\partial S_{dep, g}}{\partial V}\right)_{T} =
            \left(\frac{\partial S_{dep, g}}{\partial P}\right)_{T} \cdot
            \left(\frac{\partial P}{\partial V}\right)_{T}
        '''
        return self.dS_dep_dP_g*self.dP_dV_g

    @property
    def dS_dep_dV_l_T(self):
        r'''Derivative of departure entropy with respect to
        volume at constant temperature for the gas phase, [J/K/m^3].

        .. math::
            \left(\frac{\partial S_{dep, l}}{\partial V}\right)_{T} =
            \left(\frac{\partial S_{dep, l}}{\partial P}\right)_{T} \cdot
            \left(\frac{\partial P}{\partial V}\right)_{T}
        '''
        return self.dS_dep_dP_l*self.dP_dV_l

    @property
    def dS_dep_dV_g_P(self):
        r'''Derivative of departure entropy with respect to
        volume at constant pressure for the gas phase, [J/K/m^3].

        .. math::
            \left(\frac{\partial S_{dep, g}}{\partial V}\right)_{P} =
            \left(\frac{\partial S_{dep, g}}{\partial T}\right)_{P} \cdot
            \left(\frac{\partial T}{\partial V}\right)_{P}
        '''
        return self.dS_dep_dT_g*self.dT_dV_g

    @property
    def dS_dep_dV_l_P(self):
        r'''Derivative of departure entropy with respect to
        volume at constant pressure for the liquid phase, [J/K/m^3].

        .. math::
            \left(\frac{\partial S_{dep, l}}{\partial V}\right)_{P} =
            \left(\frac{\partial S_{dep, l}}{\partial T}\right)_{P} \cdot
            \left(\frac{\partial T}{\partial V}\right)_{P}
        '''
        return self.dS_dep_dT_l*self.dT_dV_l

    @property
    def d2H_dep_dT2_g(self):
        r'''Second temperature derivative of departure enthalpy with respect to
        temperature for the gas phase, [(J/mol)/K^2].

        .. math::
            \frac{\partial^2 H_{dep, g}}{\partial T^2} =
            P \frac{d^{2}}{d T^{2}} V{\left(T \right)} - \frac{8 T \frac{d}{d T}
            V{\left(T \right)} \frac{d^{2}}{d T^{2}} \operatorname{a\alpha}
            {\left(T \right)}}{\left(\delta^{2} - 4 \epsilon\right) \left(\frac{
            \left(\delta + 2 V{\left(T \right)}\right)^{2}}{\delta^{2}
            - 4 \epsilon} - 1\right)} + \frac{2 T \operatorname{atanh}{\left(
            \frac{\delta + 2 V{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}} \right)} \frac{d^{3}}{d T^{3}}
            \operatorname{a\alpha}{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}} + \frac{16 \left(\delta + 2 V{\left(T \right)}
            \right) \left(T \frac{d}{d T} \operatorname{a\alpha}{\left(T
            \right)} - \operatorname{a\alpha}{\left(T \right)}\right) \left(
            \frac{d}{d T} V{\left(T \right)}\right)^{2}}{\left(\delta^{2}
            - 4 \epsilon\right)^{2} \left(\frac{\left(\delta + 2 V{\left(T
            \right)}\right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)^{2}}
            - \frac{4 \left(T \frac{d}{d T} \operatorname{a\alpha}{\left(T
            \right)} - \operatorname{a\alpha}{\left(T \right)}\right)
            \frac{d^{2}}{d T^{2}} V{\left(T \right)}}{\left(\delta^{2}
            - 4 \epsilon\right) \left(\frac{\left(\delta + 2 V{\left(T \right)}
            \right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)} + \frac{2
            \operatorname{atanh}{\left(\frac{\delta + 2 V{\left(T \right)}}
            {\sqrt{\delta^{2} - 4 \epsilon}} \right)} \frac{d^{2}}{d T^{2}}
            \operatorname{a\alpha}{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}}
        '''
        T, P, delta, epsilon = self.T, self.P, self.delta, self.epsilon
        x0 = self.V_g
        x1 = self.d2V_dT2_g
        x2 = self.a_alpha
        x3 = self.d2a_alpha_dT2
        x4 = delta*delta - 4.0*epsilon
        try:
            x5 = x4**-0.5
        except:
            x5 = 1e100
        x6 = delta + x0 + x0
        x7 = 2.0*x5*catanh(x5*x6).real
        x8 = self.dV_dT_g
        x9 = x5*x5
        x10 = x6*x6*x9 - 1.0
        x11 = x9/x10
        x12 = T*self.da_alpha_dT - x2
        x50 = self.d3a_alpha_dT3
        return (P*x1  + x3*x7  + T*x7*x50- 4.0*x1*x11*x12  - 8.0*T*x11*x3*x8 + 16.0*x12*x6*x8*x8*x11*x11)

    d2H_dep_dT2_g_P = d2H_dep_dT2_g

    @property
    def d2H_dep_dT2_l(self):
        r'''Second temperature derivative of departure enthalpy with respect to
        temperature for the liquid phase, [(J/mol)/K^2].

        .. math::
            \frac{\partial^2 H_{dep, l}}{\partial T^2} =
            P \frac{d^{2}}{d T^{2}} V{\left(T \right)} - \frac{8 T \frac{d}{d T}
            V{\left(T \right)} \frac{d^{2}}{d T^{2}} \operatorname{a\alpha}
            {\left(T \right)}}{\left(\delta^{2} - 4 \epsilon\right) \left(\frac{
            \left(\delta + 2 V{\left(T \right)}\right)^{2}}{\delta^{2}
            - 4 \epsilon} - 1\right)} + \frac{2 T \operatorname{atanh}{\left(
            \frac{\delta + 2 V{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}} \right)} \frac{d^{3}}{d T^{3}}
            \operatorname{a\alpha}{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}} + \frac{16 \left(\delta + 2 V{\left(T \right)}
            \right) \left(T \frac{d}{d T} \operatorname{a\alpha}{\left(T
            \right)} - \operatorname{a\alpha}{\left(T \right)}\right) \left(
            \frac{d}{d T} V{\left(T \right)}\right)^{2}}{\left(\delta^{2}
            - 4 \epsilon\right)^{2} \left(\frac{\left(\delta + 2 V{\left(T
            \right)}\right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)^{2}}
            - \frac{4 \left(T \frac{d}{d T} \operatorname{a\alpha}{\left(T
            \right)} - \operatorname{a\alpha}{\left(T \right)}\right)
            \frac{d^{2}}{d T^{2}} V{\left(T \right)}}{\left(\delta^{2}
            - 4 \epsilon\right) \left(\frac{\left(\delta + 2 V{\left(T \right)}
            \right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)} + \frac{2
            \operatorname{atanh}{\left(\frac{\delta + 2 V{\left(T \right)}}
            {\sqrt{\delta^{2} - 4 \epsilon}} \right)} \frac{d^{2}}{d T^{2}}
            \operatorname{a\alpha}{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}}
        '''
        T, P, delta, epsilon = self.T, self.P, self.delta, self.epsilon
        x0 = self.V_l
        x1 = self.d2V_dT2_l
        x2 = self.a_alpha
        x3 = self.d2a_alpha_dT2
        x4 = delta*delta - 4.0*epsilon
        try:
            x5 = x4**-0.5
        except:
            x5 = 1e100
        x6 = delta + x0 + x0
        x7 = 2.0*x5*catanh(x5*x6).real
        x8 = self.dV_dT_l
        x9 = x5*x5
        x10 = x6*x6*x9 - 1.0
        x11 = x9/x10
        x12 = T*self.da_alpha_dT - x2
        x50 = self.d3a_alpha_dT3
        return (P*x1  + x3*x7  + T*x7*x50- 4.0*x1*x11*x12  - 8.0*T*x11*x3*x8 + 16.0*x12*x6*x8*x8*x11*x11)

    d2H_dep_dT2_l_P = d2H_dep_dT2_l

    @property
    def d2S_dep_dT2_g(self):
        r'''Second temperature derivative of departure entropy with respect to
        temperature for the gas phase, [(J/mol)/K^3].

        .. math::
            \frac{\partial^2 S_{dep, g}}{\partial T^2} = - \frac{R \left(
            \frac{d}{d T} V{\left(T \right)} - \frac{V{\left(T \right)}}{T}
            \right) \frac{d}{d T} V{\left(T \right)}}{V^{2}{\left(T \right)}}
            + \frac{R \left(\frac{d^{2}}{d T^{2}} V{\left(T \right)}
            - \frac{2 \frac{d}{d T} V{\left(T \right)}}{T} + \frac{2
            V{\left(T \right)}}{T^{2}}\right)}{V{\left(T \right)}}
            - \frac{R \frac{d^{2}}{d T^{2}} V{\left(T \right)}}{V{\left(T
            \right)}} + \frac{R \left(\frac{d}{d T} V{\left(T \right)}
            \right)^{2}}{V^{2}{\left(T \right)}} - \frac{R \frac{d^{2}}{dT^{2}}
            V{\left(T \right)}}{b - V{\left(T \right)}} - \frac{R \left(
            \frac{d}{d T} V{\left(T \right)}\right)^{2}}{\left(b - V{\left(T
            \right)}\right)^{2}} + \frac{R \left(\frac{d}{d T} V{\left(T
            \right)} - \frac{V{\left(T \right)}}{T}\right)}{T V{\left(T
            \right)}} + \frac{16 \left(\delta + 2 V{\left(T \right)}\right)
            \left(\frac{d}{d T} V{\left(T \right)}\right)^{2} \frac{d}{d T}
            \operatorname{a\alpha}{\left(T \right)}}{\left(\delta^{2}
            - 4 \epsilon\right)^{2} \left(\frac{\left(\delta + 2 V{\left(T
            \right)}\right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)^{2}}
            - \frac{8 \frac{d}{d T} V{\left(T \right)} \frac{d^{2}}{d T^{2}}
            \operatorname{a\alpha}{\left(T \right)}}{\left(\delta^{2}
            - 4 \epsilon\right) \left(\frac{\left(\delta + 2 V{\left(T \right)}
            \right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)} - \frac{4
            \frac{d^{2}}{d T^{2}} V{\left(T \right)} \frac{d}{d T}
            \operatorname{a\alpha}{\left(T \right)}}{\left(\delta^{2}
            - 4 \epsilon\right) \left(\frac{\left(\delta + 2 V{\left(T \right)}
            \right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)} + \frac{2
            \operatorname{atanh}{\left(\frac{\delta + 2 V{\left(T
            \right)}}{\sqrt{\delta^{2} - 4 \epsilon}} \right)} \frac{d^{3}}
            {d T^{3}} \operatorname{a\alpha}{\left(T \right)}}
            {\sqrt{\delta^{2} - 4 \epsilon}}
        '''
        T, P, b, delta, epsilon = self.T, self.P, self.b, self.delta, self.epsilon
        V = x0 = self.V_g
        V_inv = 1.0/V
        x1 = self.d2V_dT2_g
        x2 = R*V_inv
        x3 = V_inv*V_inv
        x4 = self.dV_dT_g
        x5 = x4*x4
        x6 = R*x5
        x7 = b - x0
        x8 = 1.0/T
        x9 = -x0*x8 + x4
        x10 = x0 + x0
        x11 = self.a_alpha
        x12 = delta*delta - 4.0*epsilon
        try:
            x13 = x12**-0.5
        except ZeroDivisionError:
            x13 = 1e100
        x14 = delta + x10
        x15 = x13*x13
        x16 = x14*x14*x15 - 1.0
        x51 = 1.0/x16
        x17 = x15*x51
        x18 = self.da_alpha_dT
        x50 = 1.0/x7
        d2a_alpha_dT2 = self.d2a_alpha_dT2
        d3a_alpha_dT3 = self.d3a_alpha_dT3
        return (-R*x1*x50 - R*x3*x4*x9 - 4.0*x1*x17*x18 - x1*x2
                + 2.0*x13*catanh(x13*x14).real*d3a_alpha_dT3
                - 8.0*x17*x4*d2a_alpha_dT2 + x2*x8*x9
                + x2*(x1 - 2.0*x4*x8 + x10*x8*x8) + x3*x6 - x6*x50*x50
                + 16.0*x14*x18*x5*x51*x51*x15*x15)

    @property
    def d2S_dep_dT2_l(self):
        r'''Second temperature derivative of departure entropy with respect to
        temperature for the liquid phase, [(J/mol)/K^3].

        .. math::
            \frac{\partial^2 S_{dep, l}}{\partial T^2} = - \frac{R \left(
            \frac{d}{d T} V{\left(T \right)} - \frac{V{\left(T \right)}}{T}
            \right) \frac{d}{d T} V{\left(T \right)}}{V^{2}{\left(T \right)}}
            + \frac{R \left(\frac{d^{2}}{d T^{2}} V{\left(T \right)}
            - \frac{2 \frac{d}{d T} V{\left(T \right)}}{T} + \frac{2
            V{\left(T \right)}}{T^{2}}\right)}{V{\left(T \right)}}
            - \frac{R \frac{d^{2}}{d T^{2}} V{\left(T \right)}}{V{\left(T
            \right)}} + \frac{R \left(\frac{d}{d T} V{\left(T \right)}
            \right)^{2}}{V^{2}{\left(T \right)}} - \frac{R \frac{d^{2}}{dT^{2}}
            V{\left(T \right)}}{b - V{\left(T \right)}} - \frac{R \left(
            \frac{d}{d T} V{\left(T \right)}\right)^{2}}{\left(b - V{\left(T
            \right)}\right)^{2}} + \frac{R \left(\frac{d}{d T} V{\left(T
            \right)} - \frac{V{\left(T \right)}}{T}\right)}{T V{\left(T
            \right)}} + \frac{16 \left(\delta + 2 V{\left(T \right)}\right)
            \left(\frac{d}{d T} V{\left(T \right)}\right)^{2} \frac{d}{d T}
            \operatorname{a\alpha}{\left(T \right)}}{\left(\delta^{2}
            - 4 \epsilon\right)^{2} \left(\frac{\left(\delta + 2 V{\left(T
            \right)}\right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)^{2}}
            - \frac{8 \frac{d}{d T} V{\left(T \right)} \frac{d^{2}}{d T^{2}}
            \operatorname{a\alpha}{\left(T \right)}}{\left(\delta^{2}
            - 4 \epsilon\right) \left(\frac{\left(\delta + 2 V{\left(T \right)}
            \right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)} - \frac{4
            \frac{d^{2}}{d T^{2}} V{\left(T \right)} \frac{d}{d T}
            \operatorname{a\alpha}{\left(T \right)}}{\left(\delta^{2}
            - 4 \epsilon\right) \left(\frac{\left(\delta + 2 V{\left(T \right)}
            \right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)} + \frac{2
            \operatorname{atanh}{\left(\frac{\delta + 2 V{\left(T
            \right)}}{\sqrt{\delta^{2} - 4 \epsilon}} \right)} \frac{d^{3}}
            {d T^{3}} \operatorname{a\alpha}{\left(T \right)}}
            {\sqrt{\delta^{2} - 4 \epsilon}}
        '''
        T, P, b, delta, epsilon = self.T, self.P, self.b, self.delta, self.epsilon
        V = x0 = self.V_l
        V_inv = 1.0/V
        x1 = self.d2V_dT2_l
        x2 = R*V_inv
        x3 = V_inv*V_inv
        x4 = self.dV_dT_l
        x5 = x4*x4
        x6 = R*x5
        x7 = b - x0
        x8 = 1.0/T
        x9 = -x0*x8 + x4
        x10 = x0 + x0
        x11 = self.a_alpha
        x12 = delta*delta - 4.0*epsilon
        try:
            x13 = x12**-0.5
        except ZeroDivisionError:
            x13 = 1e100
        x14 = delta + x10
        x15 = x13*x13
        x16 = x14*x14*x15 - 1.0
        x51 = 1.0/x16
        x17 = x15*x51
        x18 = self.da_alpha_dT
        x50 = 1.0/x7
        d2a_alpha_dT2 = self.d2a_alpha_dT2
        d3a_alpha_dT3 = self.d3a_alpha_dT3
        return (-R*x1*x50 - R*x3*x4*x9 - 4.0*x1*x17*x18 - x1*x2
                + 2.0*x13*catanh(x13*x14).real*d3a_alpha_dT3
                - 8.0*x17*x4*d2a_alpha_dT2 + x2*x8*x9
                + x2*(x1 - 2.0*x4*x8 + x10*x8*x8) + x3*x6 - x6*x50*x50
                + 16.0*x14*x18*x5*x51*x51*x15*x15)

    @property
    def d2H_dep_dT2_g_V(self):
        r'''Second temperature derivative of departure enthalpy with respect to
        temperature at constant volume for the gas phase, [(J/mol)/K^2].

        .. math::
            \left(\frac{\partial^2 H_{dep, g}}{\partial T^2}\right)_V =
            \frac{2 T \operatorname{atanh}{\left(\frac{2 V + \delta}{\sqrt{
            \delta^{2} - 4 \epsilon}} \right)} \frac{d^{3}}{d T^{3}}
            \operatorname{a\alpha}{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}} + V \frac{\partial^{2}}{\partial T^{2}}
            P{\left(V,T \right)} + \frac{2 \operatorname{atanh}{\left(\frac{
            2 V + \delta}{\sqrt{\delta^{2} - 4 \epsilon}} \right)} \frac{d^{2}}
            {d T^{2}} \operatorname{a\alpha}{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}}
        '''
        V, T, delta, epsilon = self.V_g, self.T, self.delta, self.epsilon
        x51 = delta*delta - 4.0*epsilon
        d2a_alpha_dT2 = self.d2a_alpha_dT2
        d3a_alpha_dT3 = self.d3a_alpha_dT3
        d2P_dT2 = self.d2P_dT2_g
        try:
            x1 = x51**-0.5
        except ZeroDivisionError:
            x1 = 1e100
        x2 = 2.0*x1*catanh(x1*(V + V + delta)).real
        return T*x2*d3a_alpha_dT3 + V*d2P_dT2 + x2*d2a_alpha_dT2

    @property
    def d2H_dep_dT2_l_V(self):
        r'''Second temperature derivative of departure enthalpy with respect to
        temperature at constant volume for the liquid phase, [(J/mol)/K^2].

        .. math::
            \left(\frac{\partial^2 H_{dep, l}}{\partial T^2}\right)_V =
            \frac{2 T \operatorname{atanh}{\left(\frac{2 V + \delta}{\sqrt{
            \delta^{2} - 4 \epsilon}} \right)} \frac{d^{3}}{d T^{3}}
            \operatorname{a\alpha}{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}} + V \frac{\partial^{2}}{\partial T^{2}}
            P{\left(V,T \right)} + \frac{2 \operatorname{atanh}{\left(\frac{
            2 V + \delta}{\sqrt{\delta^{2} - 4 \epsilon}} \right)} \frac{d^{2}}
            {d T^{2}} \operatorname{a\alpha}{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}}
        '''
        V, T, delta, epsilon = self.V_l, self.T, self.delta, self.epsilon
        x51 = delta*delta - 4.0*epsilon
        d2a_alpha_dT2 = self.d2a_alpha_dT2
        d3a_alpha_dT3 = self.d3a_alpha_dT3
        d2P_dT2 = self.d2P_dT2_l
        try:
            x1 = x51**-0.5
        except ZeroDivisionError:
            x1 = 1e100
        x2 = 2.0*x1*catanh(x1*(V + V + delta)).real
        return T*x2*d3a_alpha_dT3 + V*d2P_dT2 + x2*d2a_alpha_dT2

    @property
    def d2S_dep_dT2_g_V(self):
        r'''Second temperature derivative of departure entropy with respect to
        temperature at constant volume for the gas phase, [(J/mol)/K^3].

        .. math::
            \left(\frac{\partial^2 S_{dep, g}}{\partial T^2}\right)_V =
            - \frac{R \left(\frac{\partial}{\partial T} P{\left(V,T \right)}
            - \frac{P{\left(V,T \right)}}{T}\right) \frac{\partial}{\partial T}
            P{\left(V,T \right)}}{P^{2}{\left(V,T \right)}} + \frac{R \left(
            \frac{\partial^{2}}{\partial T^{2}} P{\left(V,T \right)} - \frac{2
            \frac{\partial}{\partial T} P{\left(V,T \right)}}{T} + \frac{2
            P{\left(V,T \right)}}{T^{2}}\right)}{P{\left(V,T \right)}}
            + \frac{R \left(\frac{\partial}{\partial T} P{\left(V,T \right)}
            - \frac{P{\left(V,T \right)}}{T}\right)}{T P{\left(V,T \right)}}
            + \frac{2 \operatorname{atanh}{\left(\frac{2 V + \delta}{\sqrt{
            \delta^{2} - 4 \epsilon}} \right)} \frac{d^{3}}{d T^{3}}
            \operatorname{a\alpha}{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}}
        '''
        V, T, delta, epsilon = self.V_g, self.T, self.delta, self.epsilon
        d2a_alpha_dT2 = self.d2a_alpha_dT2
        d3a_alpha_dT3 = self.d3a_alpha_dT3
        d2P_dT2 = self.d2P_dT2_g

        x0 = 1.0/T
        x1 = self.P
        P_inv = 1.0/x1
        x2 = self.dP_dT_g
        x3 = -x0*x1 + x2
        x4 = R*P_inv
        try:
            x5 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x5 = 1e100
        return (-R*x2*x3*P_inv*P_inv + x0*x3*x4 + x4*(d2P_dT2 - 2.0*x0*x2
                + 2.0*x1*x0*x0) + 2.0*x5*catanh(x5*(V + V + delta)
                ).real*d3a_alpha_dT3)

    @property
    def d2S_dep_dT2_l_V(self):
        r'''Second temperature derivative of departure entropy with respect to
        temperature at constant volume for the liquid phase, [(J/mol)/K^3].

        .. math::
            \left(\frac{\partial^2 S_{dep, l}}{\partial T^2}\right)_V =
            - \frac{R \left(\frac{\partial}{\partial T} P{\left(V,T \right)}
            - \frac{P{\left(V,T \right)}}{T}\right) \frac{\partial}{\partial T}
            P{\left(V,T \right)}}{P^{2}{\left(V,T \right)}} + \frac{R \left(
            \frac{\partial^{2}}{\partial T^{2}} P{\left(V,T \right)} - \frac{2
            \frac{\partial}{\partial T} P{\left(V,T \right)}}{T} + \frac{2
            P{\left(V,T \right)}}{T^{2}}\right)}{P{\left(V,T \right)}}
            + \frac{R \left(\frac{\partial}{\partial T} P{\left(V,T \right)}
            - \frac{P{\left(V,T \right)}}{T}\right)}{T P{\left(V,T \right)}}
            + \frac{2 \operatorname{atanh}{\left(\frac{2 V + \delta}{\sqrt{
            \delta^{2} - 4 \epsilon}} \right)} \frac{d^{3}}{d T^{3}}
            \operatorname{a\alpha}{\left(T \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}}
        '''
        V, T, delta, epsilon = self.V_l, self.T, self.delta, self.epsilon
        d2a_alpha_dT2 = self.d2a_alpha_dT2
        d3a_alpha_dT3 = self.d3a_alpha_dT3
        d2P_dT2 = self.d2P_dT2_l
        x0 = 1.0/T
        x1 = self.P
        P_inv = 1.0/x1
        x2 = self.dP_dT_l
        x3 = -x0*x1 + x2
        x4 = R*P_inv
        try:
            x5 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x5 = 1e100
        return (-R*x2*x3*P_inv*P_inv + x0*x3*x4 + x4*(d2P_dT2 - 2.0*x0*x2
                + 2.0*x1*x0*x0) + 2.0*x5*catanh(x5*(V + V + delta)
                ).real*d3a_alpha_dT3)

    @property
    def d2H_dep_dTdP_g(self):
        r'''Temperature and pressure derivative of departure enthalpy
        at constant pressure then temperature for the gas phase,
        [(J/mol)/K/Pa].

        .. math::
            \left(\frac{\partial^2 H_{dep, g}}{\partial T \partial P}\right)_{T, P}
            = P \frac{\partial^{2}}{\partial T\partial P} V{\left(T,P \right)}
            - \frac{4 T \frac{\partial}{\partial P} V{\left(T,P \right)}
            \frac{d^{2}}{d T^{2}} \operatorname{a\alpha}{\left(T \right)}}
            {\left(\delta^{2} - 4 \epsilon\right) \left(\frac{\left(\delta
            + 2 V{\left(T,P \right)}\right)^{2}}{\delta^{2} - 4 \epsilon}
            - 1\right)} + \frac{16 \left(\delta + 2 V{\left(T,P \right)}\right)
            \left(T \frac{d}{d T} \operatorname{a\alpha}{\left(T \right)}
            - \operatorname{a\alpha}{\left(T \right)}\right) \frac{\partial}
            {\partial P} V{\left(T,P \right)} \frac{\partial}{\partial T}
            V{\left(T,P \right)}}{\left(\delta^{2} - 4 \epsilon\right)^{2}
            \left(\frac{\left(\delta + 2 V{\left(T,P \right)}\right)^{2}}
            {\delta^{2} - 4 \epsilon} - 1\right)^{2}} + \frac{\partial}
            {\partial T} V{\left(T,P \right)} - \frac{4 \left(T \frac{d}{d T}
            \operatorname{a\alpha}{\left(T \right)} - \operatorname{a\alpha}
            {\left(T \right)}\right) \frac{\partial^{2}}{\partial T\partial P}
            V{\left(T,P \right)}}{\left(\delta^{2} - 4 \epsilon\right)
            \left(\frac{\left(\delta + 2 V{\left(T,P \right)}\right)^{2}}
            {\delta^{2} - 4 \epsilon} - 1\right)}
        '''
        V, T, P, delta, epsilon = self.V_g, self.T, self.P, self.delta, self.epsilon
        dV_dT = self.dV_dT_g
        d2V_dTdP = self.d2V_dTdP_g
        dV_dP = self.dV_dP_g
        a_alpha = self.a_alpha
        d2a_alpha_dT2 = self.d2a_alpha_dT2
        x5 = delta*delta - 4.0*epsilon
        try:
            x6 = 1.0/x5
        except ZeroDivisionError:
            x6 = 1e100
        x7 = delta + V  + V
        x8 = x6*x7*x7 - 1.0
        x8_inv = 1.0/x8
        x9 = 4.0*x6*x8_inv
        x10 = T*self.da_alpha_dT - a_alpha
        return (P*d2V_dTdP - T*dV_dP*x9*d2a_alpha_dT2
                + 16.0*dV_dT*x10*dV_dP*x7*x6*x6*x8_inv*x8_inv
                + dV_dT - x10*d2V_dTdP*x9)

    @property
    def d2H_dep_dTdP_l(self):
        r'''Temperature and pressure derivative of departure enthalpy
        at constant pressure then temperature for the liquid phase,
        [(J/mol)/K/Pa].

        .. math::
            \left(\frac{\partial^2 H_{dep, l}}{\partial T \partial P}\right)_V
            = P \frac{\partial^{2}}{\partial T\partial P} V{\left(T,P \right)}
            - \frac{4 T \frac{\partial}{\partial P} V{\left(T,P \right)}
            \frac{d^{2}}{d T^{2}} \operatorname{a\alpha}{\left(T \right)}}
            {\left(\delta^{2} - 4 \epsilon\right) \left(\frac{\left(\delta
            + 2 V{\left(T,P \right)}\right)^{2}}{\delta^{2} - 4 \epsilon}
            - 1\right)} + \frac{16 \left(\delta + 2 V{\left(T,P \right)}\right)
            \left(T \frac{d}{d T} \operatorname{a\alpha}{\left(T \right)}
            - \operatorname{a\alpha}{\left(T \right)}\right) \frac{\partial}
            {\partial P} V{\left(T,P \right)} \frac{\partial}{\partial T}
            V{\left(T,P \right)}}{\left(\delta^{2} - 4 \epsilon\right)^{2}
            \left(\frac{\left(\delta + 2 V{\left(T,P \right)}\right)^{2}}
            {\delta^{2} - 4 \epsilon} - 1\right)^{2}} + \frac{\partial}
            {\partial T} V{\left(T,P \right)} - \frac{4 \left(T \frac{d}{d T}
            \operatorname{a\alpha}{\left(T \right)} - \operatorname{a\alpha}
            {\left(T \right)}\right) \frac{\partial^{2}}{\partial T\partial P}
            V{\left(T,P \right)}}{\left(\delta^{2} - 4 \epsilon\right)
            \left(\frac{\left(\delta + 2 V{\left(T,P \right)}\right)^{2}}
            {\delta^{2} - 4 \epsilon} - 1\right)}
        '''
        V, T, P, delta, epsilon = self.V_l, self.T, self.P, self.delta, self.epsilon
        dV_dT = self.dV_dT_l
        d2V_dTdP = self.d2V_dTdP_l
        dV_dP = self.dV_dP_l
        a_alpha = self.a_alpha
        d2a_alpha_dT2 = self.d2a_alpha_dT2
        x5 = delta*delta - 4.0*epsilon
        try:
            x6 = 1.0/x5
        except ZeroDivisionError:
            x6 = 1e100
        x7 = delta + V  + V
        x8 = x6*x7*x7 - 1.0
        x8_inv = 1.0/x8
        x9 = 4.0*x6*x8_inv
        x10 = T*self.da_alpha_dT - a_alpha
        return (P*d2V_dTdP - T*dV_dP*x9*d2a_alpha_dT2
                + 16.0*dV_dT*x10*dV_dP*x7*x6*x6*x8_inv*x8_inv
                + dV_dT - x10*d2V_dTdP*x9)

    @property
    def d2S_dep_dTdP_g(self):
        r'''Temperature and pressure derivative of departure entropy
        at constant pressure then temperature for the gas phase,
        [(J/mol)/K^2/Pa].

        .. math::
            \left(\frac{\partial^2 S_{dep, g}}{\partial T \partial P}\right)_{T, P}
            = - \frac{R \frac{\partial^{2}}{\partial T\partial P} V{\left(T,P
            \right)}}{V{\left(T,P \right)}} + \frac{R \frac{\partial}{\partial
            P} V{\left(T,P \right)} \frac{\partial}{\partial T} V{\left(T,P
            \right)}}{V^{2}{\left(T,P \right)}} - \frac{R \frac{\partial^{2}}
            {\partial T\partial P} V{\left(T,P \right)}}{b - V{\left(T,P
            \right)}} - \frac{R \frac{\partial}{\partial P} V{\left(T,P
            \right)} \frac{\partial}{\partial T} V{\left(T,P \right)}}{\left(b
            - V{\left(T,P \right)}\right)^{2}} + \frac{16 \left(\delta
            + 2 V{\left(T,P \right)}\right) \frac{\partial}{\partial P}
            V{\left(T,P \right)} \frac{\partial}{\partial T} V{\left(T,P
            \right)} \frac{d}{d T} \operatorname{a\alpha}{\left(T \right)}}
            {\left(\delta^{2} - 4 \epsilon\right)^{2} \left(\frac{\left(\delta
            + 2 V{\left(T,P \right)}\right)^{2}}{\delta^{2} - 4 \epsilon}
            - 1\right)^{2}} - \frac{4 \frac{\partial}{\partial P} V{\left(T,P
            \right)} \frac{d^{2}}{d T^{2}} \operatorname{a\alpha}{\left(T
            \right)}}{\left(\delta^{2} - 4 \epsilon\right) \left(\frac{\left(
            \delta + 2 V{\left(T,P \right)}\right)^{2}}{\delta^{2}
            - 4 \epsilon} - 1\right)} - \frac{4 \frac{d}{d T}
            \operatorname{a\alpha}{\left(T \right)} \frac{\partial^{2}}
            {\partial T\partial P} V{\left(T,P \right)}}{\left(\delta^{2}
            - 4 \epsilon\right) \left(\frac{\left(\delta + 2 V{\left(T,P
            \right)}\right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)}
            - \frac{R \left(P \frac{\partial}{\partial P} V{\left(T,P \right)}
            + V{\left(T,P \right)}\right) \frac{\partial}{\partial T}
            V{\left(T,P \right)}}{P V^{2}{\left(T,P \right)}} + \frac{R
            \left(P \frac{\partial^{2}}{\partial T\partial P} V{\left(T,P
            \right)} - \frac{P \frac{\partial}{\partial P} V{\left(T,P
            \right)}}{T} + \frac{\partial}{\partial T} V{\left(T,P \right)}
            - \frac{V{\left(T,P \right)}}{T}\right)}{P V{\left(T,P \right)}}
            + \frac{R \left(P \frac{\partial}{\partial P} V{\left(T,P \right)}
            + V{\left(T,P \right)}\right)}{P T V{\left(T,P \right)}}
        '''
        V, T, P, b, delta, epsilon = self.V_g, self.T, self.P, self.b, self.delta, self.epsilon
        dV_dT = self.dV_dT_g
        d2V_dTdP = self.d2V_dTdP_g
        dV_dP = self.dV_dP_g

        x0 = V
        V_inv = 1.0/V
        x2 = d2V_dTdP
        x3 = R*x2
        x4 = dV_dT
        x5 = x4*V_inv*V_inv
        x6 = dV_dP
        x7 = R*x6
        x8 = b - V
        x8_inv = 1.0/x8
        x9 = 1.0/T
        x10 = P*x6
        x11 = V + x10
        x12 = R/P
        x13 = V_inv*x12
        x14 = self.a_alpha
        x15 = delta*delta - 4.0*epsilon
        try:
            x16 = 1.0/x15
        except ZeroDivisionError:
            x16 = 1e100
        x17 = delta + V + V
        x18 = x16*x17*x17 - 1.0
        x50 = 1.0/x18
        x19 = 4.0*x16*x50
        x20 = self.da_alpha_dT
        return (-V_inv*x3 - x11*x12*x5 + x11*x13*x9 + x13*(P*x2 - V*x9 - x10*x9
                + x4) - x19*x2*x20 - x19*x6*self.d2a_alpha_dT2 - x3*x8_inv
                - x4*x7*x8_inv*x8_inv + x5*x7
                + 16.0*x17*x20*x4*x6*x16*x16*x50*x50)

    @property
    def d2S_dep_dTdP_l(self):
        r'''Temperature and pressure derivative of departure entropy
        at constant pressure then temperature for the liquid phase,
        [(J/mol)/K^2/Pa].

        .. math::
            \left(\frac{\partial^2 S_{dep, l}}{\partial T \partial P}\right)_{T, P}
            = - \frac{R \frac{\partial^{2}}{\partial T\partial P} V{\left(T,P
            \right)}}{V{\left(T,P \right)}} + \frac{R \frac{\partial}{\partial
            P} V{\left(T,P \right)} \frac{\partial}{\partial T} V{\left(T,P
            \right)}}{V^{2}{\left(T,P \right)}} - \frac{R \frac{\partial^{2}}
            {\partial T\partial P} V{\left(T,P \right)}}{b - V{\left(T,P
            \right)}} - \frac{R \frac{\partial}{\partial P} V{\left(T,P
            \right)} \frac{\partial}{\partial T} V{\left(T,P \right)}}{\left(b
            - V{\left(T,P \right)}\right)^{2}} + \frac{16 \left(\delta
            + 2 V{\left(T,P \right)}\right) \frac{\partial}{\partial P}
            V{\left(T,P \right)} \frac{\partial}{\partial T} V{\left(T,P
            \right)} \frac{d}{d T} \operatorname{a\alpha}{\left(T \right)}}
            {\left(\delta^{2} - 4 \epsilon\right)^{2} \left(\frac{\left(\delta
            + 2 V{\left(T,P \right)}\right)^{2}}{\delta^{2} - 4 \epsilon}
            - 1\right)^{2}} - \frac{4 \frac{\partial}{\partial P} V{\left(T,P
            \right)} \frac{d^{2}}{d T^{2}} \operatorname{a\alpha}{\left(T
            \right)}}{\left(\delta^{2} - 4 \epsilon\right) \left(\frac{\left(
            \delta + 2 V{\left(T,P \right)}\right)^{2}}{\delta^{2}
            - 4 \epsilon} - 1\right)} - \frac{4 \frac{d}{d T}
            \operatorname{a\alpha}{\left(T \right)} \frac{\partial^{2}}
            {\partial T\partial P} V{\left(T,P \right)}}{\left(\delta^{2}
            - 4 \epsilon\right) \left(\frac{\left(\delta + 2 V{\left(T,P
            \right)}\right)^{2}}{\delta^{2} - 4 \epsilon} - 1\right)}
            - \frac{R \left(P \frac{\partial}{\partial P} V{\left(T,P \right)}
            + V{\left(T,P \right)}\right) \frac{\partial}{\partial T}
            V{\left(T,P \right)}}{P V^{2}{\left(T,P \right)}} + \frac{R
            \left(P \frac{\partial^{2}}{\partial T\partial P} V{\left(T,P
            \right)} - \frac{P \frac{\partial}{\partial P} V{\left(T,P
            \right)}}{T} + \frac{\partial}{\partial T} V{\left(T,P \right)}
            - \frac{V{\left(T,P \right)}}{T}\right)}{P V{\left(T,P \right)}}
            + \frac{R \left(P \frac{\partial}{\partial P} V{\left(T,P \right)}
            + V{\left(T,P \right)}\right)}{P T V{\left(T,P \right)}}
        '''
        V, T, P, b, delta, epsilon = self.V_l, self.T, self.P, self.b, self.delta, self.epsilon
        dV_dT = self.dV_dT_l
        d2V_dTdP = self.d2V_dTdP_l
        dV_dP = self.dV_dP_l

        x0 = V
        V_inv = 1.0/V
        x2 = d2V_dTdP
        x3 = R*x2
        x4 = dV_dT
        x5 = x4*V_inv*V_inv
        x6 = dV_dP
        x7 = R*x6
        x8 = b - V
        x8_inv = 1.0/x8
        x9 = 1.0/T
        x10 = P*x6
        x11 = V + x10
        x12 = R/P
        x13 = V_inv*x12
        x14 = self.a_alpha
        x15 = delta*delta - 4.0*epsilon
        try:
            x16 = 1.0/x15
        except ZeroDivisionError:
            x16 = 1e100
        x17 = delta + V + V
        x18 = x16*x17*x17 - 1.0
        x50 = 1.0/x18
        x19 = 4.0*x16*x50
        x20 = self.da_alpha_dT
        return (-V_inv*x3 - x11*x12*x5 + x11*x13*x9 + x13*(P*x2 - V*x9 - x10*x9
                + x4) - x19*x2*x20 - x19*x6*self.d2a_alpha_dT2 - x3*x8_inv
                - x4*x7*x8_inv*x8_inv + x5*x7
                + 16.0*x17*x20*x4*x6*x16*x16*x50*x50)

    @property
    def dfugacity_dT_l(self):
        r'''Derivative of fugacity with respect to temperature for the liquid
        phase, [Pa/K].

        .. math::
            \frac{\partial (\text{fugacity})_{l}}{\partial T} = P \left(\frac{1}
            {R T} \left(- T \frac{\partial}{\partial T} \operatorname{S_{dep}}
            {\left (T,P \right )} - \operatorname{S_{dep}}{\left (T,P \right )}
            + \frac{\partial}{\partial T} \operatorname{H_{dep}}{\left (T,P
            \right )}\right) - \frac{1}{R T^{2}} \left(- T \operatorname{
                S_{dep}}{\left (T,P \right )} + \operatorname{H_{dep}}{\left
                (T,P \right )}\right)\right) e^{\frac{1}{R T} \left(- T
                \operatorname{S_{dep}}{\left (T,P \right )} + \operatorname
                {H_{dep}}{\left (T,P \right )}\right)}
        '''
        T, P = self.T, self.P
        T_inv = 1.0/T
        S_dep_l = self.S_dep_l
        x4 = R_inv*(self.H_dep_l - T*S_dep_l)
        return P*(T_inv*R_inv*(self.dH_dep_dT_l - T*self.dS_dep_dT_l - S_dep_l)
                  - x4*T_inv*T_inv)*exp(T_inv*x4)

    @property
    def dfugacity_dT_g(self):
        r'''Derivative of fugacity with respect to temperature for the gas
        phase, [Pa/K].

        .. math::
            \frac{\partial (\text{fugacity})_{g}}{\partial T} = P \left(\frac{1}
            {R T} \left(- T \frac{\partial}{\partial T} \operatorname{S_{dep}}
            {\left (T,P \right )} - \operatorname{S_{dep}}{\left (T,P \right )}
            + \frac{\partial}{\partial T} \operatorname{H_{dep}}{\left (T,P
            \right )}\right) - \frac{1}{R T^{2}} \left(- T \operatorname{
            S_{dep}}{\left (T,P \right )} + \operatorname{H_{dep}}{\left
            (T,P \right )}\right)\right) e^{\frac{1}{R T} \left(- T
            \operatorname{S_{dep}}{\left (T,P \right )} + \operatorname
            {H_{dep}}{\left (T,P \right )}\right)}
        '''
        T, P = self.T, self.P
        T_inv = 1.0/T
        S_dep_g = self.S_dep_g
        x4 = R_inv*(self.H_dep_g - T*S_dep_g)
        return P*(T_inv*R_inv*(self.dH_dep_dT_g - T*self.dS_dep_dT_g - S_dep_g)
                  - x4*T_inv*T_inv)*exp(T_inv*x4)

    @property
    def dfugacity_dP_l(self):
        r'''Derivative of fugacity with respect to pressure for the liquid
        phase, [-].

        .. math::
            \frac{\partial (\text{fugacity})_{l}}{\partial P} = \frac{P}{R T}
            \left(- T \frac{\partial}{\partial P} \operatorname{S_{dep}}{\left
            (T,P \right )} + \frac{\partial}{\partial P} \operatorname{H_{dep}}
            {\left (T,P \right )}\right) e^{\frac{1}{R T} \left(- T
            \operatorname{S_{dep}}{\left (T,P \right )} + \operatorname{
            H_{dep}}{\left (T,P \right )}\right)} + e^{\frac{1}{R T}
            \left(- T \operatorname{S_{dep}}{\left (T,P \right )}
            + \operatorname{H_{dep}}{\left (T,P \right )}\right)}
        '''
        T, P = self.T, self.P
        x0 = 1.0/(R*T)
        return (1.0 - P*x0*(T*self.dS_dep_dP_l - self.dH_dep_dP_l))*exp(
                -x0*(T*self.S_dep_l - self.H_dep_l))

    @property
    def dfugacity_dP_g(self):
        r'''Derivative of fugacity with respect to pressure for the gas
        phase, [-].

        .. math::
            \frac{\partial (\text{fugacity})_{g}}{\partial P} = \frac{P}{R T}
            \left(- T \frac{\partial}{\partial P} \operatorname{S_{dep}}{\left
            (T,P \right )} + \frac{\partial}{\partial P} \operatorname{H_{dep}}
            {\left (T,P \right )}\right) e^{\frac{1}{R T} \left(- T
            \operatorname{S_{dep}}{\left (T,P \right )} + \operatorname{
            H_{dep}}{\left (T,P \right )}\right)} + e^{\frac{1}{R T}
            \left(- T \operatorname{S_{dep}}{\left (T,P \right )}
            + \operatorname{H_{dep}}{\left (T,P \right )}\right)}
        '''
        T, P = self.T, self.P
        x0 = 1.0/(R*T)
        try:
            ans =  (1.0 - P*x0*(T*self.dS_dep_dP_g - self.dH_dep_dP_g))*exp(
                    -x0*(T*self.S_dep_g - self.H_dep_g))
            if isinf(ans) or isnan(ans):
                return 1.0
            return ans
        except Exception as e:
            if P < 1e-50:
                # Applies to gas phase only!
                return 1.0
            else:
                raise e

    @property
    def dphi_dT_l(self):
        r'''Derivative of fugacity coefficient with respect to temperature for
        the liquid phase, [1/K].

        .. math::
            \frac{\partial \phi}{\partial T} = \left(\frac{- T \frac{\partial}
            {\partial T} \operatorname{S_{dep}}{\left(T,P \right)}
            - \operatorname{S_{dep}}{\left(T,P \right)} + \frac{\partial}
            {\partial T} \operatorname{H_{dep}}{\left(T,P \right)}}{R T}
            - \frac{- T \operatorname{S_{dep}}{\left(T,P \right)}
            + \operatorname{H_{dep}}{\left(T,P \right)}}{R T^{2}}\right)
            e^{\frac{- T \operatorname{S_{dep}}{\left(T,P \right)}
            + \operatorname{H_{dep}}{\left(T,P \right)}}{R T}}
        '''
        T, P = self.T, self.P
        T_inv = 1.0/T
        x4 = T_inv*(T*self.S_dep_l - self.H_dep_l)
        return (-R_inv*T_inv*(T*self.dS_dep_dT_l + self.S_dep_l - x4
                             - self.dH_dep_dT_l)*exp(-R_inv*x4))

    @property
    def dphi_dT_g(self):
        r'''Derivative of fugacity coefficient with respect to temperature for
        the gas phase, [1/K].

        .. math::
            \frac{\partial \phi}{\partial T} = \left(\frac{- T \frac{\partial}
            {\partial T} \operatorname{S_{dep}}{\left(T,P \right)}
            - \operatorname{S_{dep}}{\left(T,P \right)} + \frac{\partial}
            {\partial T} \operatorname{H_{dep}}{\left(T,P \right)}}{R T}
            - \frac{- T \operatorname{S_{dep}}{\left(T,P \right)}
            + \operatorname{H_{dep}}{\left(T,P \right)}}{R T^{2}}\right)
            e^{\frac{- T \operatorname{S_{dep}}{\left(T,P \right)}
            + \operatorname{H_{dep}}{\left(T,P \right)}}{R T}}
        '''
        T, P = self.T, self.P
        T_inv = 1.0/T
        x4 = T_inv*(T*self.S_dep_g - self.H_dep_g)
        return (-R_inv*T_inv*(T*self.dS_dep_dT_g + self.S_dep_g - x4
                             - self.dH_dep_dT_g)*exp(-R_inv*x4))

    @property
    def dphi_dP_l(self):
        r'''Derivative of fugacity coefficient with respect to pressure for
        the liquid phase, [1/Pa].

        .. math::
            \frac{\partial \phi}{\partial P} = \frac{\left(- T \frac{\partial}
            {\partial P} \operatorname{S_{dep}}{\left(T,P \right)}
            + \frac{\partial}{\partial P} \operatorname{H_{dep}}{\left(T,P
            \right)}\right) e^{\frac{- T \operatorname{S_{dep}}{\left(T,P
            \right)} + \operatorname{H_{dep}}{\left(T,P \right)}}{R T}}}{R T}
        '''
        T = self.T
        x0 = self.S_dep_l
        x1 = self.H_dep_l
        x2 = 1.0/(R*T)
        return -x2*(T*self.dS_dep_dP_l - self.dH_dep_dP_l)*exp(-x2*(T*x0 - x1))

    @property
    def dphi_dP_g(self):
        r'''Derivative of fugacity coefficient with respect to pressure for
        the gas phase, [1/Pa].

        .. math::
            \frac{\partial \phi}{\partial P} = \frac{\left(- T \frac{\partial}
            {\partial P} \operatorname{S_{dep}}{\left(T,P \right)}
            + \frac{\partial}{\partial P} \operatorname{H_{dep}}{\left(T,P
            \right)}\right) e^{\frac{- T \operatorname{S_{dep}}{\left(T,P
            \right)} + \operatorname{H_{dep}}{\left(T,P \right)}}{R T}}}{R T}
        '''
        T = self.T
        x0 = self.S_dep_g
        x1 = self.H_dep_g
        x2 = 1.0/(R*T)
        return -x2*(T*self.dS_dep_dP_g - self.dH_dep_dP_g)*exp(-x2*(T*x0 - x1))

    @property
    def dbeta_dT_g(self):
        r'''Derivative of isobaric expansion coefficient with respect to
        temperature for the gas phase, [1/K^2].

        .. math::
            \frac{\partial \beta_g}{\partial T} = \frac{\frac{\partial^{2}}
            {\partial T^{2}} V{\left (T,P \right )_g}}{V{\left (T,P \right )_g}} -
            \frac{\left(\frac{\partial}{\partial T} V{\left (T,P \right )_g}
            \right)^{2}}{V^{2}{\left (T,P \right )_g}}
        '''
        V_inv = 1.0/self.V_g
        dV_dT = self.dV_dT_g
        return V_inv*(self.d2V_dT2_g - dV_dT*dV_dT*V_inv)

    @property
    def dbeta_dT_l(self):
        r'''Derivative of isobaric expansion coefficient with respect to
        temperature for the liquid phase, [1/K^2].

        .. math::
            \frac{\partial \beta_l}{\partial T} = \frac{\frac{\partial^{2}}
            {\partial T^{2}} V{\left (T,P \right )_l}}{V{\left (T,P \right )_l}} -
            \frac{\left(\frac{\partial}{\partial T} V{\left (T,P \right )_l}
            \right)^{2}}{V^{2}{\left (T,P \right )_l}}
        '''
        V_inv = 1.0/self.V_l
        dV_dT = self.dV_dT_l
        return V_inv*(self.d2V_dT2_l - dV_dT*dV_dT*V_inv)

    @property
    def dbeta_dP_g(self):
        r'''Derivative of isobaric expansion coefficient with respect to
        pressure for the gas phase, [1/(Pa*K)].

        .. math::
            \frac{\partial \beta_g}{\partial P} = \frac{\frac{\partial^{2}}
            {\partial T\partial P} V{\left (T,P \right )_g}}{V{\left (T,
            P \right )_g}} - \frac{\frac{\partial}{\partial P} V{\left (T,P
            \right )_g} \frac{\partial}{\partial T} V{\left (T,P \right )_g}}
            {V^{2}{\left (T,P \right )_g}}
        '''
        V_inv = 1.0/self.V_g
        dV_dT = self.dV_dT_g
        dV_dP = self.dV_dP_g
        return V_inv*(self.d2V_dTdP_g - dV_dT*dV_dP*V_inv)

    @property
    def dbeta_dP_l(self):
        r'''Derivative of isobaric expansion coefficient with respect to
        pressure for the liquid phase, [1/(Pa*K)].

        .. math::
            \frac{\partial \beta_g}{\partial P} = \frac{\frac{\partial^{2}}
            {\partial T\partial P} V{\left (T,P \right )_l}}{V{\left (T,
            P \right )_l}} - \frac{\frac{\partial}{\partial P} V{\left (T,P
            \right )_l} \frac{\partial}{\partial T} V{\left (T,P \right )_l}}
            {V^{2}{\left (T,P \right )_l}}
        '''
        V_inv = 1.0/self.V_l
        dV_dT = self.dV_dT_l
        dV_dP = self.dV_dP_l
        return V_inv*(self.d2V_dTdP_l - dV_dT*dV_dP*V_inv)

    @property
    def da_alpha_dP_g_V(self):
        r'''Derivative of the `a_alpha` with respect to
        pressure at constant volume (varying T) for the gas phase,
        [J^2/mol^2/Pa^2].

        .. math::
            \left(\frac{\partial a \alpha}{\partial P}\right)_{V}
            = \left(\frac{\partial a \alpha}{\partial T}\right)_{P}
            \cdot\left( \frac{\partial T}{\partial P}\right)_V
        '''
        return self.da_alpha_dT*self.dT_dP_g

    @property
    def da_alpha_dP_l_V(self):
        r'''Derivative of the `a_alpha` with respect to
        pressure at constant volume (varying T) for the liquid phase,
        [J^2/mol^2/Pa^2].

        .. math::
            \left(\frac{\partial a \alpha}{\partial P}\right)_{V}
            = \left(\frac{\partial a \alpha}{\partial T}\right)_{P}
            \cdot\left( \frac{\partial T}{\partial P}\right)_V
        '''
        return self.da_alpha_dT*self.dT_dP_l

    @property
    def d2a_alpha_dTdP_g_V(self):
        r'''Derivative of the temperature derivative of `a_alpha` with respect
        to pressure at constant volume (varying T) for the gas phase,
        [J^2/mol^2/Pa^2/K].

        .. math::
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}
            = \left(\frac{\partial^2 a \alpha}{\partial T^2}\right)_{P}
            \cdot\left( \frac{\partial T}{\partial P}\right)_V
        '''
        return self.d2a_alpha_dT2*self.dT_dP_g

    @property
    def d2a_alpha_dTdP_l_V(self):
        r'''Derivative of the temperature derivative of `a_alpha` with respect
        to pressure at constant volume (varying T) for the liquid phase,
        [J^2/mol^2/Pa^2/K].

        .. math::
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}
            = \left(\frac{\partial^2 a \alpha}{\partial T^2}\right)_{P}
            \cdot\left( \frac{\partial T}{\partial P}\right)_V
        '''
        return self.d2a_alpha_dT2*self.dT_dP_l

    @property
    def d2P_dVdP_g(self):
        r'''Second derivative of pressure with respect to molar volume and
        then pressure for the gas phase, [mol/m^3].

        .. math::
            \frac{\partial^2 P}{\partial V \partial P} =
            \frac{2 R T \frac{d}{d P} V{\left(P \right)}}{\left(- b + V{\left(P
            \right)}\right)^{3}} - \frac{\left(- \delta - 2 V{\left(P \right)}
            \right) \left(- 2 \delta \frac{d}{d P} V{\left(P \right)}
            - 4 V{\left(P \right)} \frac{d}{d P} V{\left(P \right)}\right)
            \operatorname{a\alpha}{\left(T \right)}}{\left(\delta V{\left(P
            \right)} + \epsilon + V^{2}{\left(P \right)}\right)^{3}} + \frac{2
            \operatorname{a\alpha}{\left(T \right)} \frac{d}{d P} V{\left(P
            \right)}}{\left(\delta V{\left(P \right)} + \epsilon + V^{2}
            {\left(P \right)}\right)^{2}}

        '''
        r"""Feels like a really strange derivative. Have not been able to construct
        it from others yet. Value is Symmetric - can calculate it both ways.
        Still feels like there should be a general method for obtaining these derivatives.

        from sympy import *
        P, T, R, b, delta, epsilon = symbols('P, T, R, b, delta, epsilon')
        a_alpha, V = symbols(r'a\alpha, V', cls=Function)

        dP_dV = 1/(1/(-R*T/(V(P) - b)**2 - a_alpha(T)*(-2*V(P) - delta)/(V(P)**2 + V(P)*delta + epsilon)**2))
        cse(diff(dP_dV, P), optimizations='basic')
        """
        T, P, b, delta, epsilon = self.T, self.P, self.b, self.delta, self.epsilon
        x0 = self.V_g
        x1 = self.a_alpha
        x2 = delta*x0 + epsilon + x0*x0
        x50 = self.dV_dP_g
        x51 = x0 + x0 + delta
        x52 = 1.0/(b - x0)
        x2_inv = 1.0/x2
        return 2.0*(-R*T*x52*x52*x52 + x1*x2_inv*x2_inv*(1.0 - x51*x51*x2_inv))*x50

    @property
    def d2P_dVdP_l(self):
        r'''Second derivative of pressure with respect to molar volume and
        then pressure for the liquid phase, [mol/m^3].

        .. math::
            \frac{\partial^2 P}{\partial V \partial P} =
            \frac{2 R T \frac{d}{d P} V{\left(P \right)}}{\left(- b + V{\left(P
            \right)}\right)^{3}} - \frac{\left(- \delta - 2 V{\left(P \right)}
            \right) \left(- 2 \delta \frac{d}{d P} V{\left(P \right)}
            - 4 V{\left(P \right)} \frac{d}{d P} V{\left(P \right)}\right)
            \operatorname{a\alpha}{\left(T \right)}}{\left(\delta V{\left(P
            \right)} + \epsilon + V^{2}{\left(P \right)}\right)^{3}} + \frac{2
            \operatorname{a\alpha}{\left(T \right)} \frac{d}{d P} V{\left(P
            \right)}}{\left(\delta V{\left(P \right)} + \epsilon + V^{2}
            {\left(P \right)}\right)^{2}}

        '''
        T, b, delta, epsilon = self.T, self.b, self.delta, self.epsilon
        x0 = self.V_l
        x1 = self.a_alpha
        x2 = delta*x0 + epsilon + x0*x0
        x50 = self.dV_dP_l
        x51 = x0 + x0 + delta
        x52 = 1.0/(b - x0)
        x2_inv = 1.0/x2
        return 2.0*(-R*T*x52*x52*x52 + x1*x2_inv*x2_inv*(1.0 - x51*x51*x2_inv))*x50

    @property
    def d2P_dVdT_TP_g(self):
        r'''Second derivative of pressure with respect to molar volume and
        then temperature at constant temperature then pressure for the gas
        phase, [Pa*mol/m^3/K].

        .. math::
            \left(\frac{\partial^2 P}{\partial V \partial T}\right)_{T,P} =
            \frac{2 R T \frac{d}{d T} V{\left(T \right)}}{\left(- b + V{\left(T
            \right)}\right)^{3}} - \frac{R}{\left(- b + V{\left(T \right)}
            \right)^{2}} - \frac{\left(- \delta - 2 V{\left(T \right)}\right)
            \left(- 2 \delta \frac{d}{d T} V{\left(T \right)} - 4 V{\left(T
            \right)} \frac{d}{d T} V{\left(T \right)}\right) \operatorname{
            a\alpha}{\left(T \right)}}{\left(\delta V{\left(T \right)}
            + \epsilon + V^{2}{\left(T \right)}\right)^{3}} - \frac{\left(
            - \delta - 2 V{\left(T \right)}\right) \frac{d}{d T} \operatorname{
            a\alpha}{\left(T \right)}}{\left(\delta V{\left(T \right)}
            + \epsilon + V^{2}{\left(T \right)}\right)^{2}} + \frac{2
            \operatorname{a\alpha}{\left(T \right)} \frac{d}{d T} V{\left(T
            \right)}}{\left(\delta V{\left(T \right)} + \epsilon + V^{2}{\left(
            T \right)}\right)^{2}}
        '''
        T, b, delta, epsilon = self.T, self.b, self.delta, self.epsilon
        x0 = self.V_g
        x2 = 2.0*self.dV_dT_g
        x1 = self.b - x0
        x1_inv = 1.0/x1
        x3 = delta*x0 + epsilon + x0*x0
        x3_inv = 1.0/x3
        x4 = x3_inv*x3_inv
        x5 = self.a_alpha
        x6 = x2*x5
        x7 = delta + x0 + x0
        return (-x1_inv*x1_inv*R*(T*x2*x1_inv + 1.0) + x4*x6
                + x4*x7*(self.da_alpha_dT - x6*x7*x3_inv))

    @property
    def d2P_dVdT_TP_l(self):
        r'''Second derivative of pressure with respect to molar volume and
        then temperature at constant temperature then pressure for the liquid
        phase, [Pa*mol/m^3/K].

        .. math::
            \left(\frac{\partial^2 P}{\partial V \partial T}\right)_{T,P} =
            \frac{2 R T \frac{d}{d T} V{\left(T \right)}}{\left(- b + V{\left(T
            \right)}\right)^{3}} - \frac{R}{\left(- b + V{\left(T \right)}
            \right)^{2}} - \frac{\left(- \delta - 2 V{\left(T \right)}\right)
            \left(- 2 \delta \frac{d}{d T} V{\left(T \right)} - 4 V{\left(T
            \right)} \frac{d}{d T} V{\left(T \right)}\right) \operatorname{
            a\alpha}{\left(T \right)}}{\left(\delta V{\left(T \right)}
            + \epsilon + V^{2}{\left(T \right)}\right)^{3}} - \frac{\left(
            - \delta - 2 V{\left(T \right)}\right) \frac{d}{d T} \operatorname{
            a\alpha}{\left(T \right)}}{\left(\delta V{\left(T \right)}
            + \epsilon + V^{2}{\left(T \right)}\right)^{2}} + \frac{2
            \operatorname{a\alpha}{\left(T \right)} \frac{d}{d T} V{\left(T
            \right)}}{\left(\delta V{\left(T \right)} + \epsilon + V^{2}{\left(
            T \right)}\right)^{2}}
        '''
        T, b, delta, epsilon = self.T, self.b, self.delta, self.epsilon
        x0 = self.V_l
        x2 = 2.0*self.dV_dT_l
        x1 = self.b - x0
        x1_inv = 1.0/x1
        x3 = delta*x0 + epsilon + x0*x0
        x3_inv = 1.0/x3
        x4 = x3_inv*x3_inv
        x5 = self.a_alpha
        x6 = x2*x5
        x7 = delta + x0 + x0
        return (-x1_inv*x1_inv*R*(T*x2*x1_inv + 1.0) + x4*x6
                + x4*x7*(self.da_alpha_dT - x6*x7*x3_inv))

    @property
    def d2P_dT2_PV_g(self):
        r'''Second derivative of pressure with respect to temperature twice,
        but with pressure held constant the first time and volume held
        constant the second time for the gas phase, [Pa/K^2].

        .. math::
            \left(\frac{\partial^2 P}{\partial T \partial T}\right)_{P,V} =
            - \frac{R \frac{d}{d T} V{\left(T \right)}}{\left(- b + V{\left(T
            \right)}\right)^{2}} - \frac{\left(- \delta \frac{d}{d T} V{\left(T
            \right)} - 2 V{\left(T \right)} \frac{d}{d T} V{\left(T \right)}
            \right) \frac{d}{d T} \operatorname{a\alpha}{\left(T \right)}}
            {\left(\delta V{\left(T \right)} + \epsilon + V^{2}{\left(T
            \right)}\right)^{2}} - \frac{\frac{d^{2}}{d T^{2}}
            \operatorname{a\alpha}{\left(T \right)}}{\delta V{\left(T \right)}
            + \epsilon + V^{2}{\left(T \right)}}
        '''
        T, b, delta, epsilon = self.T, self.b, self.delta, self.epsilon
        V = self.V_g
        dV_dT = self.dV_dT_g

        x2 = self.a_alpha
        x0 = V
        x1 = dV_dT
        x3 = delta*x0 + epsilon + x0*x0
        x3_inv = 1.0/x3
        x50 = 1.0/(b - x0)
        return (-R*x1*x50*x50 + x1*(delta + x0 + x0)*self.da_alpha_dT*x3_inv*x3_inv - self.d2a_alpha_dT2*x3_inv)

    @property
    def d2P_dT2_PV_l(self):
        r'''Second derivative of pressure with respect to temperature twice,
        but with pressure held constant the first time and volume held
        constant the second time for the liquid phase, [Pa/K^2].

        .. math::
            \left(\frac{\partial^2 P}{\partial T \partial T}\right)_{P,V} =
            - \frac{R \frac{d}{d T} V{\left(T \right)}}{\left(- b + V{\left(T
            \right)}\right)^{2}} - \frac{\left(- \delta \frac{d}{d T} V{\left(T
            \right)} - 2 V{\left(T \right)} \frac{d}{d T} V{\left(T \right)}
            \right) \frac{d}{d T} \operatorname{a\alpha}{\left(T \right)}}
            {\left(\delta V{\left(T \right)} + \epsilon + V^{2}{\left(T
            \right)}\right)^{2}} - \frac{\frac{d^{2}}{d T^{2}}
            \operatorname{a\alpha}{\left(T \right)}}{\delta V{\left(T \right)}
            + \epsilon + V^{2}{\left(T \right)}}
        '''
        T, b, delta, epsilon = self.T, self.b, self.delta, self.epsilon
        V = self.V_l
        dV_dT = self.dV_dT_l

        x0 = V
        x1 = dV_dT
        x2 = self.a_alpha
        x3 = delta*x0 + epsilon + x0*x0
        x3_inv = 1.0/x3
        x50 = 1.0/(b - x0)
        return (-R*x1*x50*x50 + x1*(delta + x0 + x0)*self.da_alpha_dT*x3_inv*x3_inv - self.d2a_alpha_dT2*x3_inv)

    @property
    def d2P_dTdP_g(self):
        r'''Second derivative of pressure with respect to temperature and,
        then pressure; and with volume held constant at first, then temperature,
        for the gas phase, [1/K].

        .. math::
            \left(\frac{\partial^2 P}{\partial T \partial P}\right)_{V, T} =
            - \frac{R \frac{d}{d P} V{\left(P \right)}}{\left(- b + V{\left(P
            \right)}\right)^{2}} - \frac{\left(- \delta \frac{d}{d P} V{\left(P
            \right)} - 2 V{\left(P \right)} \frac{d}{d P} V{\left(P \right)}
            \right) \frac{d}{d T} \operatorname{a\alpha}{\left(T \right)}}
            {\left(\delta V{\left(P \right)} + \epsilon + V^{2}{\left(P
            \right)}\right)^{2}}
        '''
        V = self.V_g
        dV_dP = self.dV_dP_g
        T, b, delta, epsilon = self.T, self.b, self.delta, self.epsilon
        da_alpha_dT = self.da_alpha_dT
        x0 = V - b
        x1 = delta*V + epsilon + V*V
        return (-R*dV_dP/(x0*x0) - (-delta*dV_dP - 2.0*V*dV_dP)*da_alpha_dT/(x1*x1))


    @property
    def d2P_dTdP_l(self):
        r'''Second derivative of pressure with respect to temperature and,
        then pressure; and with volume held constant at first, then temperature,
        for the liquid phase, [1/K].

        .. math::
            \left(\frac{\partial^2 P}{\partial T \partial P}\right)_{V, T} =
            - \frac{R \frac{d}{d P} V{\left(P \right)}}{\left(- b + V{\left(P
            \right)}\right)^{2}} - \frac{\left(- \delta \frac{d}{d P} V{\left(P
            \right)} - 2 V{\left(P \right)} \frac{d}{d P} V{\left(P \right)}
            \right) \frac{d}{d T} \operatorname{a\alpha}{\left(T \right)}}
            {\left(\delta V{\left(P \right)} + \epsilon + V^{2}{\left(P
            \right)}\right)^{2}}
        '''
        V = self.V_l
        dV_dP = self.dV_dP_l
        T, b, delta, epsilon = self.T, self.b, self.delta, self.epsilon
        da_alpha_dT = self.da_alpha_dT
        x0 = V - b
        x1 = delta*V + epsilon + V*V
        return (-R*dV_dP/(x0*x0) - (-delta*dV_dP - 2.0*V*dV_dP)*da_alpha_dT/(x1*x1))

    @property
    def lnphi_l(self):
        r'''The natural logarithm of the fugacity coefficient for
        the liquid phase, [-].
        '''
        return self.G_dep_l*R_inv/self.T

    @property
    def lnphi_g(self):
        r'''The natural logarithm of the fugacity coefficient for
        the gas phase, [-].
        '''
        return log(self.phi_g)


class IG(GCEOS):
    r'''Class for solving the ideal gas equation in the `GCEOS` framework.
    This provides access to a number of derivatives and properties easily.
    It also keeps a common interface for all gas models. However, it is
    somewhat slow.

    Subclasses :obj:`GCEOS`, which
    provides the methods for solving the EOS and calculating its assorted
    relevant thermodynamic properties. Solves the EOS on initialization.

    Two of `T`, `P`, and `V` are needed to solve the EOS; values for `Tc` and
    `Pc` and `omega`, which are not used in the calculates, are set to those of
    methane by default to allow use without specifying them.

    .. math::
        P = \frac{RT}{V}

    Parameters
    ----------
    Tc : float, optional
        Critical temperature, [K]
    Pc : float, optional
        Critical pressure, [Pa]
    omega : float, optional
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    T-P initialization, and exploring each phase's properties:

    >>> eos = IG(T=400., P=1E6)
    >>> eos.V_g, eos.phase
    (0.003325785047261296, 'g')
    >>> eos.H_dep_g, eos.S_dep_g, eos.U_dep_g, eos.G_dep_g, eos.A_dep_g
    (0.0, 0.0, 0.0, 0.0, 0.0)
    >>> eos.beta_g, eos.kappa_g, eos.Cp_dep_g, eos.Cv_dep_g
    (0.0025, 1e-06, 0.0, 0.0)
    >>> eos.fugacity_g, eos.PIP_g, eos.Z_g, eos.dP_dT_g
    (1000000.0, 0.9999999999999999, 1.0, 2500.0)

    Notes
    -----

    References
    ----------
    .. [1] Smith, J. M, H. C Van Ness, and Michael M Abbott. Introduction to
       Chemical Engineering Thermodynamics. Boston: McGraw-Hill, 2005.
    '''

    Zc = 1.0
    """float: Critical compressibility for an ideal gas is 1"""
    a = 0.0
    """float: `a` parameter for an ideal gas is 0"""
    b = 0.0
    """float: `b` parameter for an ideal gas is 0"""
    delta = 0.0
    """float: `delta` parameter for an ideal gas is 0"""
    epsilon = 0.0
    """float: `epsilon` parameter for an ideal gas is 0"""
    volume_solutions = staticmethod(volume_solutions_ideal)

    # Handle the properties where numerical error puts values - but they should
    # be zero. Not all of them are non-zero all the time - but some times
    # they are
    def _zero(self): return 0.0
    def _set_nothing(self, thing): return

    try:
        try:
            doc = GCEOS.d2T_dV2_g.__doc__
        except:
            doc = ''
        d2T_dV2_g = property(_zero, fset=_set_nothing, doc=doc)
        try:
            doc = GCEOS.d2V_dT2_g.__doc__
        except:
            doc = ''
        d2V_dT2_g = property(_zero, fset=_set_nothing, doc=doc)
        try:
            doc = GCEOS.U_dep_g.__doc__
        except:
            doc = ''
        U_dep_g = property(_zero, fset=_set_nothing, doc=doc)
        try:
            doc = GCEOS.A_dep_g.__doc__
        except:
            doc = ''
        A_dep_g = property(_zero, fset=_set_nothing, doc=doc)
        try:
            doc = GCEOS.V_dep_g.__doc__
        except:
            doc = ''
            V_dep_g = property(_zero, fset=_set_nothing, doc=doc)
        G_dep_g = property(_zero, fset=_set_nothing, doc='Departure Gibbs free energy of an ideal gas is zero, [J/(mol)]')
        H_dep_g = property(_zero, fset=_set_nothing, doc='Departure enthalpy of an ideal gas is zero, [J/(mol)]')
        S_dep_g = property(_zero, fset=_set_nothing, doc='Departure entropy of an ideal gas is zero, [J/(mol*K)]')
        Cp_dep_g = property(_zero, fset=_set_nothing, doc='Departure heat capacity of an ideal gas is zero, [J/(mol*K)]')
        # Replace methods
        dH_dep_dP_g = property(_zero, doc=GCEOS.dH_dep_dP_g.__doc__)
        dH_dep_dT_g = property(_zero, doc=GCEOS.dH_dep_dT_g.__doc__)
        dS_dep_dP_g = property(_zero, doc=GCEOS.dS_dep_dP_g.__doc__)
        dS_dep_dT_g = property(_zero, doc=GCEOS.dS_dep_dT_g.__doc__)
        dfugacity_dT_g = property(_zero, doc=GCEOS.dfugacity_dT_g.__doc__)
        dphi_dP_g = property(_zero, doc=GCEOS.dphi_dP_g.__doc__)
        dphi_dT_g = property(_zero, doc=GCEOS.dphi_dT_g.__doc__)

    except:
        pass


    def __init__(self, Tc=None, Pc=None, omega=None, T=None, P=None,
                 V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.solve()

    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate :math:`a \alpha` and its first and second
        derivatives for this EOS. All values are zero.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]

        '''
        return (0.0, 0.0, 0.0)

    def a_alpha_pure(self, T):
        r'''Method to calculate :math:`a \alpha` for the ideal gas law, which
        is zero.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        '''
        return 0.0

    def solve_T(self, P, V, solution=None):
        r'''Method to calculate `T` from a specified `P` and `V` for the
        ideal gas equation of state.

        .. math::
            T = \frac{PV}{R}

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        solution : str or None, optional
            Not used, [-]

        Returns
        -------
        T : float
            Temperature, [K]

        Notes
        -----
        '''
        self.no_T_spec = True
        return P*V*R_inv

    def phi_sat(self, T, polish=True):
        return 1.0
    def dphi_sat_dT(self, T, polish=True):
        return 0.0
    def d2phi_sat_dT2(self, T, polish=True):
        return 0.0


class PR(GCEOS):
    r'''Class for solving the Peng-Robinson [1]_ [2]_ cubic
    equation of state for a pure compound. Subclasses :obj:`GCEOS`, which
    provides the methods for solving the EOS and calculating its assorted
    relevant thermodynamic properties. Solves the EOS on initialization.

    The main methods here are :obj:`PR.a_alpha_and_derivatives_pure`, which calculates
    :math:`a \alpha` and its first and second derivatives, and :obj:`PR.solve_T`, which from a
    specified `P` and `V` obtains `T`.

    Two of (`T`, `P`, `V`) are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

    .. math::
        a=0.45724\frac{R^2T_c^2}{P_c}

    .. math::
         b=0.07780\frac{RT_c}{P_c}

    .. math::
        \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2

    .. math::
        \kappa=0.37464+1.54226\omega-0.26992\omega^2

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    T-P initialization, and exploring each phase's properties:

    >>> eos = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400., P=1E6)
    >>> eos.V_l, eos.V_g
    (0.000156073184785, 0.0021418768167)
    >>> eos.phase
    'l/g'
    >>> eos.H_dep_l, eos.H_dep_g
    (-26111.8775716, -3549.30057795)
    >>> eos.S_dep_l, eos.S_dep_g
    (-58.098447843, -6.4394518931)
    >>> eos.U_dep_l, eos.U_dep_g
    (-22942.1657091, -2365.3923474)
    >>> eos.G_dep_l, eos.G_dep_g
    (-2872.49843435, -973.51982071)
    >>> eos.A_dep_l, eos.A_dep_g
    (297.21342811, 210.38840980)
    >>> eos.beta_l, eos.beta_g
    (0.00269337091778, 0.0101232239111)
    >>> eos.kappa_l, eos.kappa_g
    (9.3357215438e-09, 1.97106698097e-06)
    >>> eos.Cp_minus_Cv_l, eos.Cp_minus_Cv_g
    (48.510162249, 44.544161128)
    >>> eos.Cv_dep_l, eos.Cp_dep_l
    (18.8921126734, 59.0878123050)

    P-T initialization, liquid phase, and round robin trip:

    >>> eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.000130222125139, -31134.75084, -72.47561931)

    T-V initialization, liquid phase:

    >>> eos2 = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., V=eos.V_l)
    >>> eos2.P, eos2.phase
    (1000000.00, 'l')

    P-V initialization at same state:

    >>> eos3 = PR(Tc=507.6, Pc=3025000, omega=0.2975, V=eos.V_l, P=1E6)
    >>> eos3.T, eos3.phase
    (299.0000000000, 'l')

    Notes
    -----
    The constants in the expresions for `a` and `b` are given to full precision
    in the actual code, as derived in [3]_.

    The full expression for critical compressibility is:

    .. math::
        Z_c = \frac{1}{32} \left(\sqrt[3]{16 \sqrt{2}-13}-\frac{7}{\sqrt[3]
        {16 \sqrt{2}-13}}+11\right)

    References
    ----------
    .. [1] Peng, Ding-Yu, and Donald B. Robinson. "A New Two-Constant Equation
       of State." Industrial & Engineering Chemistry Fundamentals 15, no. 1
       (February 1, 1976): 59-64. doi:10.1021/i160057a011.
    .. [2] Robinson, Donald B., Ding-Yu Peng, and Samuel Y-K Chung. "The
       Development of the Peng - Robinson Equation and Its Application to Phase
       Equilibrium in a System Containing Methanol." Fluid Phase Equilibria 24,
       no. 1 (January 1, 1985): 25-41. doi:10.1016/0378-3812(85)87035-7.
    .. [3] Privat, R., and J.-N. Jaubert. "PPR78, a Thermodynamic Model for the
       Prediction of Petroleum Fluid-Phase Behaviour," 11. EDP Sciences, 2011.
       doi:10.1051/jeep/201100011.
    '''

    # constant part of `a`,
    # X = (-1 + (6*sqrt(2)+8)**Rational(1,3) - (6*sqrt(2)-8)**Rational(1,3))/3
    # (8*(5*X+1)/(49-37*X)).evalf(40)
    c1 = 0.4572355289213821893834601962251837888504
    """Full value of the constant in the `a` parameter"""
    c1R2 = c1*R2

    # Constant part of `b`, (X/(X+3)).evalf(40)
    c2 = 0.0777960739038884559718447100373331839711
    """Full value of the constant in the `b` parameter"""
    c2R = c2*R
    c1R2_c2R = c1R2/c2R
    #    c1, c2 = 0.45724, 0.07780

    # Zc is the mechanical compressibility for mixtures as well.
    Zc = 0.3074013086987038480093850966542222720096
    """Mechanical compressibility of Peng-Robinson EOS"""

    Psat_coeffs_limiting = [-3.4758880164801873, 0.7675486448347723]

    Psat_coeffs_critical = [13.906174756604267, -8.978515559640332,
                            6.191494729386664, -3.3553014047359286,
                            1.0000000000011509]

    Psat_cheb_coeffs = [-7.693430141477579, -7.792157693145173, -0.12584439451814622, 0.0045868660863990305,
                        0.011902728116315585, -0.00809984848593371, 0.0035807374586641324, -0.001285457896498948,
                        0.0004379441379448949, -0.0001701325511665626, 7.889450459420399e-05, -3.842330780886875e-05,
                        1.7884847876342805e-05, -7.9432179091441e-06, 3.51726370898656e-06, -1.6108797741557683e-06,
                        7.625638345550717e-07, -3.6453554523813245e-07, 1.732454904858089e-07, -8.195124459058523e-08,
                        3.8929380082904216e-08, -1.8668536344161905e-08, 9.021955971552252e-09, -4.374277331168795e-09,
                        2.122697092724708e-09, -1.0315557015083254e-09, 5.027805333255708e-10, -2.4590905784642285e-10,
                        1.206301486380689e-10, -5.932583414867791e-11, 2.9274476912683964e-11, -1.4591650777202522e-11,
                        7.533835507484918e-12, -4.377200831613345e-12, 1.7413208326438542e-12]
    # below  - down to .14 Tr
#    Psat_cheb_coeffs = [-69.78144560030312, -70.82020621910401, -0.5505993362058134, 0.262763240774557, -0.13586962327984622, 0.07091484524874882, -0.03531507189835045, 0.015348266653126313, -0.004290800414097142, -0.0015192254949775404, 0.004230003950690049, -0.005148646330256051, 0.005067979846360524, -0.004463618393006094, 0.0036338412594165456, -0.002781745442601943, 0.0020410583004693912, -0.0014675469823800154, 0.001041797382518202, -0.0007085008245359792, 0.0004341450533632967, -0.00023059133991796472, 0.00012404966848973944, -0.00010575986390189084, 0.00011927874294723816, -0.00010216011382070127, 4.142986825089964e-05, 1.6994654942134455e-05, -2.0393896226146606e-05, -3.05495184394464e-05, 7.840494892004187e-05, -6.715144915784917e-05, 1.9360256298218764e-06, 5.342823303794287e-05, -4.2445268102696054e-05, -2.258059184830652e-05, 7.156133295478447e-05, -5.0419963297068014e-05, -2.1185333936025785e-05, 6.945722167248469e-05, -4.3468774802286496e-05, -3.0211658906858938e-05, 7.396450066832002e-05, -4.0987041756199036e-05, -3.4507186813052766e-05, 3.6619358939125855e-05]
    # down to .05 Tr
#    Psat_cheb_coeffs = [-71.62442148475718, -72.67946752713178, -0.5550432977559888, 0.2662527679044299, -0.13858385912471755, 0.07300013042829502, -0.03688566755461173, 0.01648745160444604, -0.005061858504315144, -0.0010519595693067093, 0.0039868988560367085, -0.005045456840770146, 0.00504419254495023, -0.0044982000664379905, 0.003727506855649437, -0.002922838794275898, 0.0021888012528213734, -0.0015735578492615076, 0.0010897606359061226, -0.0007293553555925913, 0.0004738606767778966, -0.00030120118607927907, 0.00018992197213856394, -0.00012147385378832608, 8.113736696036817e-05, -5.806550163389163e-05, 4.4822397778703055e-05, -3.669084579413651e-05, 3.0945466319478186e-05, -2.62003968013127e-05, 2.1885122184587654e-05, -1.786717828032663e-05, 1.420082721312861e-05, -1.0981475209780111e-05, 8.276527284992199e-06, -6.100440122314813e-06, 4.420342273408809e-06, -3.171239452318529e-06, 2.2718591475182304e-06, -1.641149583754854e-06, 1.2061284404980935e-06, -9.067266070702959e-07, 6.985214276328142e-07, -5.490755862981909e-07, 4.372991567070929e-07, -3.504743494298746e-07, 2.8019662848682576e-07, -2.2266768846404626e-07, 1.7533403880408145e-07, -1.3630227589226426e-07, 1.0510214144142285e-07, -8.02098792008235e-08, 6.073935683412093e-08, -4.6105511380996746e-08, 3.478599121821662e-08, -2.648029023793574e-08, 2.041302301328165e-08, -1.5671212844805128e-08, 1.2440282394539782e-08, -9.871977759603047e-09, 7.912503992331811e-09, -6.6888910721434e-09, 5.534654087073205e-09, -4.92019981055108e-09, 4.589363968756223e-09, -2.151778718334702e-09]
    # down to .05 Tr polishing
#    Psat_cheb_coeffs = [-73.9119088855554, -74.98674794418481, -0.5603678572345178, 0.2704608002227193, -0.1418754021264281, 0.07553218818095526, -0.03878657980070652, 0.017866520164384912, -0.0060152224341743525, -0.0004382750653244775, 0.003635841462596336, -0.004888955750612924, 0.005023631814771542, -0.004564880757514128, 0.003842769402817585, -0.0030577040987875793, 0.0023231191552369407, -0.001694755295849508, 0.0011913577693282759, -0.0008093955530850967, 0.0005334402485338361, -0.0003431831424850387, 0.00021792836239828482, -0.00013916167527852, 9.174638441139245e-05, -6.419699908390207e-05, 4.838277855408256e-05, -3.895686370452493e-05, 3.267491660000825e-05, -2.7780478658642705e-05, 2.3455257030895833e-05, -1.943068869205973e-05, 1.5702249378726904e-05, -1.2352834841441616e-05, 9.468188716352547e-06, -7.086815965689662e-06, 5.202794456673999e-06, -3.7660662091643354e-06, 2.710802447723022e-06, -1.9547001517481854e-06, 1.4269579917305496e-06, -1.0627333211922062e-06, 8.086972219940435e-07, -6.313736088052035e-07, 5.002098614800398e-07, -4.014517222719182e-07, 3.222357369727768e-07, -2.591706410738203e-07, 2.0546606649125658e-07, -1.6215902481453263e-07, 1.2645321295092458e-07, -9.678506993483597e-08, 7.52490799383037e-08, -5.60685972986457e-08, 4.3358661542007224e-08, -3.2329350971261814e-08, 2.5091238603112617e-08, -1.8903964302567286e-08, 1.4892047699817043e-08, -1.1705624527623068e-08, 8.603302527636011e-09, -7.628847828412486e-09, 5.0543164590698825e-09, -5.102159698856454e-09, 3.0709992836479988e-09, -2.972533529000884e-09, 2.0494601230946347e-09, -1.626141536313283e-09, 1.6617716853181003e-09, -6.470653307871083e-10, 1.1333690091031717e-09, -1.2451614782651999e-10, 1.098942683163892e-09, 9.673645066411718e-11, 6.206934530152836e-10, -1.1913910201270805e-10, 3.559906774745769e-11, -5.419942764994107e-10, -2.372580701782284e-10, -5.785415972247437e-10, -1.789757696430208e-10]
    # down to .05 with lots of failures C40 only
#    Psat_cheb_coeffs =  [-186.30264784196294, -188.01235085131194, -0.6975588305160902, 0.38422679790906106, -0.2358303051434559, 0.15258449381119304, -0.101338177792044, 0.0679573457611134, -0.045425247476661136, 0.029879338234709937, -0.019024330378443737, 0.011418999154577504, -0.006113230472632388, 0.00246054797767154, -4.3960533109688155e-06, -0.0015825897164979809, 0.002540504992834563, -0.003046881596822211, 0.0032353807402903272, -0.0032061955400497044, 0.0030337264005811464, -0.0027744314554593126, 0.002469806934918433, -0.002149376765619085, 0.001833408492489406, -0.00153552022142691, 0.0012645817528752557, -0.0010249792000921317, 0.0008181632585418055, -0.0006436998283177283, 0.0004995903113614604, -0.0003828408287994695, 0.0002896812774307662, -0.00021674416012176133, 0.00016131784370737042, -0.00012009195488808489, 8.966908457382076e-05, -6.764450681363164e-05, 5.209192773849304e-05, -4.1139971086693995e-05, 3.3476318185800505e-05, -2.8412997762476805e-05, 2.513421113263226e-05, -2.2567508719078435e-05, 2.0188809493379843e-05, -1.810962700274516e-05, 1.643508229137845e-05, -1.503569055933669e-05, 1.3622272823701577e-05, -1.2076671646564277e-05, 1.054271875585668e-05, -9.007273271254411e-06, 7.523720857264602e-06, -6.424404525130439e-06, 5.652203861001342e-06, -4.7755499168431625e-06, 3.7604252783225858e-06, -2.92395389072605e-06, 2.3520802660480336e-06, -1.9209673206999083e-06, 1.6125790706312328e-06, -1.4083468032508143e-06, 1.1777450938630518e-06, -8.636616122606049e-07, 5.749905340593687e-07, -4.644992178826096e-07, 5.109912172256424e-07, -5.285927442208997e-07, 4.4610491153173465e-07, -3.3435155715273366e-07, 2.2022096388817243e-07, -1.3138808837994352e-07, 1.5788807254228123e-07, -2.6570415873228444e-07, 2.820563887584985e-07, -1.6783703722562406e-07, 4.477559158897425e-08, -2.4698813388799755e-09, 5.082691394016857e-08, -1.364026020206371e-07, 1.6850593650100272e-07, -1.0443374638586546e-07, -6.029473813268628e-10, 5.105380858617091e-08, -1.5066843023282578e-08, -5.630921379297198e-08, 9.561766786891034e-08, -8.044216329068123e-08, 3.359993333902796e-08, 1.692366968619578e-08, -2.021364343358841e-08]
     # down to .03, plenty of failures
#    Psat_cheb_coeffs = [-188.50329975567104, -190.22994960376462, -0.6992886012204886, 0.3856961269737735, -0.23707446208582353, 0.15363415372584763, -0.10221883018831106, 0.06869084576669, -0.046030774233320346, 0.03037297246598552, -0.019421744608583133, 0.011732910491046633, -0.006355800820106353, 0.0026413894471214202, -0.0001333621829559692, -0.0014967435287118152, 0.002489721202961943, -0.00302447283347462, 0.0032350727289014642, -0.0032223921492743357, 0.0030622558268892, -0.0028113049747675455, 0.002511348612059362, -0.002192644454555338, 0.0018764599744331163, -0.0015770771123065552, 0.0013034116032509804, -0.0010603100672178776, 0.00084960767850329, -0.0006709816561447436, 0.0005226330473731801, -0.0004018349441941878, 0.0003053468509191052, -0.00022974201509485604, 0.00017163053097478257, -0.0001278303586505278, 9.545950876002835e-05, -7.200007894259846e-05, 5.5312909934416405e-05, -4.3632781581719854e-05, 3.554641644507928e-05, -2.99488097950353e-05, 2.6011962388807256e-05, -2.3127603908643427e-05, 2.0875472981740965e-05, -1.8975408339047864e-05, 1.7255291079923385e-05, -1.562250114123633e-05, 1.4033483268247027e-05, -1.2483202707948607e-05, 1.0981181475278024e-05, -9.547990214685254e-06, 8.20534723265339e-06, -6.970215811404035e-06, 5.857096216944197e-06, -4.8714713996210945e-06, 4.015088107327757e-06, -3.2837642912761844e-06, 2.6688332761922373e-06, -2.1605704853781956e-06, 1.745415965345872e-06, -1.4112782858614675e-06, 1.1450344603347899e-06, -9.34468189749192e-07, 7.693687927218034e-07, -6.395653830685742e-07, 5.378418354520407e-07, -4.570688107726579e-07, 3.922470141699613e-07, -3.396066879296283e-07, 2.9547505651179775e-07, -2.5824629138078686e-07, 2.259435099158857e-07, -1.9759059073588738e-07, 1.7245665023281603e-07, -1.499107122703144e-07, 1.2993920706246258e-07, -1.1188458371271578e-07, 9.59786582193289e-08, -8.193904465038978e-08, 6.951736088200208e-08, -5.883242593822998e-08, 4.953479013200448e-08, -4.159778119910192e-08, 3.4903544554923914e-08, -2.9199660726126307e-08, 2.4491065764276586e-08, -2.0543807377807442e-08, 1.716620639244989e-08, -1.4598093803545008e-08, 1.2247184453541803e-08, -1.0378062685590349e-08, 8.941636289359033e-09, -7.547512972569913e-09, 6.5406029883590885e-09, -5.55017639345453e-09, 4.857924129262302e-09, -4.170327848134446e-09, 3.5473818590708514e-09, -3.1820101162273115e-09, 2.634813506155291e-09, -2.3186710334946806e-09, 1.9854991410760484e-09, -1.698026932061246e-09, 1.4939355398374196e-09, -1.2257013267845049e-09, 1.1034926144506615e-09, -8.867213325365261e-10, 7.759313594207437e-10, -6.85530513757325e-10, 5.315937675947832e-10, -5.001264119638624e-10, 4.2230130059116994e-10, -3.259379961024697e-10, 2.8696408042785254e-10, -2.654348289559891e-10, 2.240260857681517e-10, -1.5881755448515084e-10, 1.7089871651079086e-10, -1.743032336304004e-10, 5.736029218880029e-11, -9.974594793790009e-11, 1.2854164813721342e-10, -5.569999528883679e-11, 5.432760350528726e-11, -5.900487596351839e-11, 7.348655484042815e-11, 1.9834070367000245e-12, 3.887800704201888e-11, -6.528210426664377e-11, 6.144420801150463e-12, -2.0697350409069892e-11, 9.512216860539657e-12, -4.439607915237426e-11, -1.6185927706642567e-11, -2.8071628138323645e-12, 6.158579755107668e-11, 2.148407244207534e-11, 5.277970985609337e-13, -9.859059640730805e-12, 4.1564767036192385e-12, -1.5577673049063656e-11, -1.2654069415571345e-12, -1.9761710714008562e-12, 9.40276686806768e-12, 4.583732482119074e-13, -1.8523582732792032e-11, -1.7428972653131536e-11, 2.334371921024897e-11, 1.2661569384099514e-11, -2.4431492094169338e-11, -2.720598171659233e-11, 1.579179961710281e-11, 4.682966091729829e-11, 2.026395923889618e-11, -4.163510324266956e-11, -2.7091399111035808e-11, 3.978859743850732e-11, 3.993365393136633e-11, -2.4706365750991333e-11, -2.8201589338545247e-11]
#    Psat_cheb_coeffs =  [-188.81248459710693, -190.53226813843213, -0.6992718797266877, 0.3857083557782601, -0.23710917890714395, 0.15368561772753983, -0.10228211161653594, 0.06876166878498034, -0.046105558737181966, 0.030448740221432544, -0.019496099441454324, 0.01180400058944964, -0.006422229275450882, 0.002702227307086234, -0.00018800410519084597, -0.0014485238631714243, 0.0024479474900583895, -0.002988894024752606, 0.0032053382330997785, -0.003197984048551589, 0.0030426262430619812, -0.0027958384579597137, 0.0024994432437511482, -0.00218371114178375, 0.0018699437151919942, -0.0015724843629802854, 0.0013002928376298992, -0.0010582955457831876, 0.0008483768179051751, -0.0006702845742590901, 0.0005222702922150421, -0.0004016564112164708, 0.0003052504825598366, -0.00022965330503168022, 0.00017151209256412164, -0.00012765639237664444, 9.522751362437718e-05, -7.17145087909031e-05, 5.498576051758942e-05, -4.328024825801364e-05, 3.518008638334846e-05, -2.9585552080573432e-05, 2.5660899927246663e-05, -2.2801213593209296e-05, 2.0579135430209277e-05, -1.871227629774825e-05, 1.702697381072197e-05, -1.5427107330232484e-05, 1.3871955438611369e-05, -1.235063269577285e-05, 1.087503047126396e-05, -9.463372111120008e-06, 8.138409928400627e-06, -6.918751587310431e-06, 5.817036690746729e-06, -4.841268302762132e-06, 3.990762592248579e-06, -3.264055878954419e-06, 2.6526744772618845e-06, -2.146826614278467e-06, 1.7339220505229884e-06, -1.4002686597492801e-06, 1.1352817872143799e-06, -9.252727697582733e-07, 7.610055457905131e-07, -6.319237506120556e-07, 5.30160897737689e-07, -4.5034836164150563e-07, 3.8588236023116243e-07, -3.345288398991865e-07, 2.910099599025734e-07, -2.538502269447694e-07, 2.2221275929649412e-07, -1.9404386102611735e-07, 1.7012903413041972e-07, -1.4791267614537682e-07, 1.281131161442957e-07, -1.1035351009983888e-07, 9.412216917920838e-08, -8.103521480312085e-08, 6.889862034626618e-08, -5.823229805384481e-08, 4.888865274151847e-08, -4.0647361572055817e-08, 3.461181492625629e-08, -2.890818104595808e-08, 2.4189127295759093e-08, -2.036506388954876e-08, 1.6621054692260028e-08, -1.4376599744841544e-08, 1.2262293144383739e-08, -1.0166543599991339e-08, 8.776172074614484e-09, -7.244748882363349e-09, 6.552057774765062e-09, -5.655401910624057e-09, 4.4124427509814644e-09, -4.138406545361605e-09, 3.4155934985322144e-09, -3.1467981765942498e-09, 3.138041596064127e-09, -2.097881746535653e-09, 1.6538597491971884e-09, -1.4302796654967797e-09, 1.3958696624380472e-09, -1.6941697510614072e-09, 1.1559050790778446e-09, -8.424336557798272e-10, 7.445069759938515e-10, -3.8008350586066653e-10, 6.681447868524303e-10, -5.609484209193093e-10, 1.1709177677205352e-10, -5.781259004102078e-10, 5.45265361901197e-10, -1.3987335287680026e-10, 1.7128157135074418e-10, 1.0377866018526204e-10, 1.449451573983006e-10, -4.977625195297418e-10, 1.7368603686632612e-10, -3.571321706516851e-11, -1.6249813391308165e-10, 4.6148221569532015e-11, 3.9554757121876716e-10, -1.0268016727946628e-10, -7.436027752479989e-11, -1.6876374859490107e-10, -4.24547853876368e-11, 9.538626006134858e-12, 1.5150070863903953e-10, 2.7005277922459003e-10, -1.6342760518896042e-11, -4.572503911555491e-10, 4.922727672815753e-11, 9.160300994028991e-11, -7.120976338703244e-11, 2.164872706420613e-10, 1.1646536920908047e-10, -2.7132159904485077e-10, -9.18445653054099e-11, 1.1410414945528784e-10, 1.1967624164073171e-10, -5.5743966043066313e-11, 3.9042323803713426e-11, 4.316392256370049e-11, -1.8428367625021157e-10, -9.040283123061977e-11, 1.857434297108983e-10, 1.592233467198178e-11, -1.173771592481677e-10, 1.1665496090537252e-10, 1.2886364193873557e-10, -2.1093389704449506e-10, -2.4675247129314452e-11, 1.515767676711589e-10, -1.2689980450730342e-10, -4.2776899169681866e-11, 1.6317818359826586e-10, -1.4821901477978135e-11, -5.8141610036405774e-11]

    Psat_cheb_coeffs_der = chebder(Psat_cheb_coeffs)
    Psat_coeffs_critical_der = polyder(Psat_coeffs_critical[::-1])[::-1]
    Psat_cheb_constant_factor = (-2.355355160853182, 0.42489124941587103)
#    Psat_cheb_constant_factor = (-19.744219083323905, 0.050649991923423815) # down to .14 Tr
#    Psat_cheb_constant_factor = (-20.25334447874608, 0.049376705093756613) # down to .05
#    Psat_cheb_constant_factor = (-20.88507690836272, 0.0478830941599295) # down to .05 repolishing
#    Psat_cheb_constant_factor = (-51.789209241068214, 0.019310239068163836) # down to .05 with lots of failures C40 only
#    Psat_cheb_constant_factor = (-52.392851049631986, 0.01908689378961204) # down to .03, plenty of failures
#    Psat_cheb_constant_factor = (-52.47770345042524, 0.01905687810661655)

    Psat_cheb_range = (0.003211332390446207, 104.95219556846003)

    phi_sat_coeffs = [4.040440857039882e-09, -1.512382901024055e-07, 2.5363900091436416e-06,
                      -2.4959001060510725e-05, 0.00015714708105355206, -0.0006312347348814933,
                      0.0013488647482434379, 0.0008510254890166079, -0.017614759099592196,
                      0.06640627813169839, -0.13427456425899886, 0.1172205279608668,
                      0.13594473870160448, -0.5560225934266592, 0.7087599054079694,
                      0.6426353018023558]

    _P_zero_l_cheb_coeffs = [0.13358936990391557, -0.20047353906149878, 0.15101308518135467, -0.11422662323168498, 0.08677799907222833, -0.06622719396774103, 0.05078577177767531, -0.03913992025038471, 0.030322206247168845, -0.023618484941949063, 0.018500212460075605, -0.014575143278285305, 0.011551352410948363, -0.00921093058565245, 0.007390713292456164, -0.005968132800177682, 0.00485080886172241, -0.003968872414987763, 0.003269291360484698, -0.002711665819666899, 0.0022651044970457743, -0.0019058978265104418, 0.0016157801830935644, -0.0013806283122768208, 0.0011894838915417153, -0.0010338173333182162, 0.0009069721482541163, -0.0008037443041438563, 0.0007200633946601682, -0.0006527508698173454, 0.0005993365082194993, -0.0005579199462298259, 0.0005270668422661141, -0.0005057321913053223, 0.0004932057251527365, -0.00024453764761005106]
    P_zero_l_cheb_limits = (0.002068158270122966, 27.87515959722943)

    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.b = b = self.c2R*Tc/Pc
        self.a = b*Tc*self.c1R2_c2R
        self.kappa = omega*(-0.26992*omega + 1.54226) + 0.37464
        self.delta, self.epsilon = 2.0*b, -b*b
        self.solve()

    def a_alpha_pure(self, T):
        r'''Method to calculate :math:`a \alpha` for this EOS. Uses the set values of
        `Tc`, `kappa`, and `a`.

        .. math::
            a\alpha = a \left(\kappa \left(- \frac{T^{0.5}}{Tc^{0.5}}
            + 1\right) + 1\right)^{2}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the value, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

        Notes
        -----
        This method does not alter the object's state and the temperature
        provided can be a different than that of the object.

        Examples
        --------
        Dodecane at 250 K:

        >>> eos = PR(Tc=658.0, Pc=1820000.0, omega=0.562, T=500., P=1e5)
        >>> eos.a_alpha_pure(250.0)
        15.66839156301
        '''
        x0 = (1.0 + self.kappa*(1.0 - sqrt(T/self.Tc)))
        return self.a*x0*x0

    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate :math:`a \alpha` and its first and second
        derivatives for this EOS. Uses the set values of `Tc`, `kappa`, and `a`.

        .. math::
            a\alpha = a \left(\kappa \left(- \frac{T^{0.5}}{Tc^{0.5}}
            + 1\right) + 1\right)^{2}

        .. math::
            \frac{d a\alpha}{dT} = - \frac{1.0 a \kappa}{T^{0.5} Tc^{0.5}}
            \left(\kappa \left(- \frac{T^{0.5}}{Tc^{0.5}} + 1\right) + 1\right)

        .. math::
            \frac{d^2 a\alpha}{dT^2} = 0.5 a \kappa \left(- \frac{1}{T^{1.5}
            Tc^{0.5}} \left(\kappa \left(\frac{T^{0.5}}{Tc^{0.5}} - 1\right)
            - 1\right) + \frac{\kappa}{T^{1.0} Tc^{1.0}}\right)

        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]

        Notes
        -----
        This method does not alter the object's state and the temperature
        provided can be a different than that of the object.

        Examples
        --------
        Dodecane at 250 K:

        >>> eos = PR(Tc=658.0, Pc=1820000.0, omega=0.562, T=500., P=1e5)
        >>> eos.a_alpha_and_derivatives_pure(250.0)
        (15.66839156301, -0.03094091246957, 9.243186769880e-05)
        '''
        # TODO custom water a_alpha?
        # Peng, DY, and DB Robinson. "Two-and Three-Phase Equilibrium Calculations
        # for Coal Gasification and Related Processes,", 1980
        # Thermodynamics of aqueous systems with industrial applications 133 (1980): 393-414.
        # Applies up to Tr .85.
        # Suggested in Equations of State And PVT Analysis.
        Tc, kappa, a = self.Tc, self.kappa, self.a
        x0 = sqrt(T)
        x1 = 1.0/sqrt(Tc)
        x2 = kappa*(x0*x1 - 1.) - 1.
        x3 = a*kappa
        x4 = x1*x2/x0

        a_alpha = a*x2*x2
        da_alpha_dT = x4*x3
        d2a_alpha_dT2 = 0.5*x3*(kappa*x1*x1 - x4)/T

        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def d3a_alpha_dT3_pure(self, T):
        r'''Method to calculate the third temperature derivative of `a_alpha`.
        Uses the set values of `Tc`, `kappa`, and `a`. This property is not
        normally needed.

        .. math::
            \frac{d^3 a\alpha}{dT^3} = \frac{3 a\kappa \left(- \frac{\kappa}
            {T_{c}} + \frac{\sqrt{\frac{T}{T_{c}}} \left(\kappa \left(\sqrt{\frac{T}
            {T_{c}}} - 1\right) - 1\right)}{T}\right)}{4 T^{2}}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the derivative, [-]

        Returns
        -------
        d3a_alpha_dT3 : float
            Third temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^3]

        Notes
        -----
        This method does not alter the object's state and the temperature
        provided can be a different than that of the object.

        Examples
        --------
        Dodecane at 500 K:

        >>> eos = PR(Tc=658.0, Pc=1820000.0, omega=0.562, T=500., P=1e5)
        >>> eos.d3a_alpha_dT3_pure(500.0)
        -9.8038800671e-08
        '''
        kappa = self.kappa
        x0 = 1.0/self.Tc
        T_inv = 1.0/T
        x1 = sqrt(T*x0)
        return -self.a*0.75*kappa*(kappa*x0 - x1*(kappa*(x1 - 1.0) - 1.0)*T_inv)*T_inv*T_inv

    def P_max_at_V(self, V):
        r'''Method to calculate the maximum pressure the EOS can create at a
        constant volume, if one exists; returns None otherwise.

        Parameters
        ----------
        V : float
            Constant molar volume, [m^3/mol]

        Returns
        -------
        P : float
            Maximum possible isochoric pressure, [Pa]

        Notes
        -----
        The analytical determination of this formula involved some part of the
        discriminant, and much black magic.

        Examples
        --------
        >>> e = PR(P=1e5, V=0.0001437, Tc=512.5, Pc=8084000.0, omega=0.559)
        >>> e.P_max_at_V(e.V)
        2247886208.7
        '''
        """# Partial notes on how this was determined.
        from sympy import *
        P, T, V = symbols('P, T, V', positive=True)
        Tc, Pc, omega = symbols('Tc, Pc, omega', positive=True)
        R, a, b, kappa = symbols('R, a, b, kappa')

        main = P*R*Tc*V**2 + 2*P*R*Tc*V*b - P*R*Tc*b**2 - P*V*a*kappa**2 + P*a*b*kappa**2 + R*Tc*a*kappa**2 + 2*R*Tc*a*kappa + R*Tc*a
        to_subs = {b: thing.b,
                   kappa: thing.kappa,
                   a: thing.a, R: thermo.eos.R, Tc: thing.Tc, V: thing.V, Tc: thing.Tc, omega: thing.omega}
        solve(Eq(main, 0), P)[0].subs(to_subs)
        """
        try:
            Tc, a, b, kappa = self.Tc, self.a, self.b, self.kappa
        except:
            Tc, a, b, kappa = self.Tcs[0], self.ais[0], self.bs[0], self.kappas[0]
        P_max = (-R*Tc*a*(kappa**2 + 2*kappa + 1)/(R*Tc*V**2 + 2*R*Tc*V*b - R*Tc*b**2 - V*a*kappa**2 + a*b*kappa**2))
        if P_max < 0.0:
            # No positive pressure - it's negative
            return None
        return P_max


    # (V - b)**3*(V**2 + 2*V*b - b**2)*(P*R*Tc*V**2 + 2*P*R*Tc*V*b - P*R*Tc*b**2 - P*V*a*kappa**2 + P*a*b*kappa**2 + R*Tc*a*kappa**2 + 2*R*Tc*a*kappa + R*Tc*a)


    def solve_T(self, P, V, solution=None):
        r'''Method to calculate `T` from a specified `P` and `V` for the PR
        EOS. Uses `Tc`, `a`, `b`, and `kappa` as well, obtained from the
        class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        solution : str or None, optional
            'l' or 'g' to specify a liquid of vapor solution (if one exists);
            if None, will select a solution more likely to be real (closer to
            STP, attempting to avoid temperatures like 60000 K or 0.0001 K).

        Returns
        -------
        T : float
            Temperature, [K]

        Notes
        -----
        The exact solution can be derived as follows, and is excluded for
        breviety.

        >>> from sympy import *
        >>> P, T, V = symbols('P, T, V')
        >>> Tc, Pc, omega = symbols('Tc, Pc, omega')
        >>> R, a, b, kappa = symbols('R, a, b, kappa')
        >>> a_alpha = a*(1 + kappa*(1-sqrt(T/Tc)))**2
        >>> PR_formula = R*T/(V-b) - a_alpha/(V*(V+b)+b*(V-b)) - P
        >>> #solve(PR_formula, T)

        After careful evaluation of the results of the analytical formula,
        it was discovered, that numerical precision issues required several
        NR refinement iterations; at at times, when the analytical value is
        extremely erroneous, a call to a full numerical solver not using the
        analytical solution at all is required.

        Examples
        --------
        >>> eos = PR(Tc=658.0, Pc=1820000.0, omega=0.562, T=500., P=1e5)
        >>> eos.solve_T(P=eos.P, V=eos.V_g)
        500.0000000
        '''
        self.no_T_spec = True
        Tc, a, b, kappa = self.Tc, self.a, self.b, self.kappa
        # Needs to be improved to do a NR or two at the end!
        x0 = V*V
        x1 = R*Tc
        x2 = x0*x1
        x3 = kappa*kappa
        x4 = a*x3
        x5 = b*x4
        x6 = 2.*V*b
        x7 = x1*x6
        x8 = b*b
        x9 = x1*x8
        x10 = V*x4
        thing = (x2 - x10 + x5 + x7 - x9)
        x11 = thing*thing
        x12 = x0*x0
        x13 = R*R
        x14 = Tc*Tc
        x15 = x13*x14
        x16 = x8*x8
        x17 = a*a
        x18 = x3*x3
        x19 = x17*x18
        x20 = x0*V
        x21 = 2.*R*Tc*a*x3
        x22 = x8*b
        x23 = 4.*V*x22
        x24 = 4.*b*x20
        x25 = a*x1
        x26 = x25*x8
        x27 = x26*x3
        x28 = x0*x25
        x29 = x28*x3
        x30 = 2.*x8
        x31 = (6.*V*x27 - 2.*b*x29 + x0*x13*x14*x30 + x0*x19 + x12*x15
               + x15*x16 - x15*x23 + x15*x24 - x19*x6 + x19*x8 - x20*x21
               - x21*x22)
        V_m_b = V - b
        x33 = 2.*(R*Tc*a*kappa)
        x34 = P*x2
        x35 = P*x5
        x36 = x25*x3
        x37 = P*x10
        x38 = P*R*Tc
        x39 = V*x17
        x40 = 2.*kappa*x3
        x41 = b*x17
        x42 = P*a*x3

        # 2.*a*kappa - add a negative sign to get the high temperature solution
        # sometimes it is complex!
#            try:
        root_term = sqrt(V_m_b**3*(x0 + x6 - x8)*(P*x7 -
                                          P*x9 + x25 + x33 + x34 + x35
                                          + x36 - x37))
#            except ValueError:
#                # negative number in sqrt
#                return super(PR, self).solve_T(P, V)

        x100 = 2.*a*kappa*x11*(root_term*(kappa + 1.))
        x101 = (x31*V_m_b*((4.*V)*(R*Tc*a*b*kappa) + x0*x33 - x0*x35 + x12*x38
                     + x16*x38 + x18*x39 - x18*x41 - x20*x42 - x22*x42
                     - x23*x38 + x24*x38 + x25*x6 - x26 - x27 + x28 + x29
                     + x3*x39 - x3*x41 + x30*x34 - x33*x8 + x36*x6
                     + 3*x37*x8 + x39*x40 - x40*x41))
        x102 = -Tc/(x11*x31)

        T_calc = (x102*(x100 - x101)) # Normally the correct root
        if T_calc < 0.0:
            # Ruined, call the numerical method; sometimes it happens
            return super().solve_T(P, V, solution=solution)

        Tc_inv = 1.0/Tc

        T_calc_high = (x102*(-x100 - x101))
        if solution is not None and solution == 'g':
            T_calc = T_calc_high
        if True:
            c1, c2 = R/(V_m_b), a/(V*(V+b) + b*V_m_b)

            rt = (T_calc*Tc_inv)**0.5
            alpha_root = (1.0 + kappa*(1.0-rt))
            err = c1*T_calc - alpha_root*alpha_root*c2 - P
            if abs(err/P) > 1e-2:
                # Numerical issue - such a bad solution we cannot converge
                return super().solve_T(P, V, solution=solution)

            # Newton step - might as well compute it
            derr = c1 + c2*kappa*rt*(kappa*(1.0 -rt) + 1.0)/T_calc
            if derr == 0.0:
                return T_calc
            T_calc = T_calc - err/derr

            # Step 2 - cannot find occasion to need more steps, most of the time
            # this does nothing!
            rt = (T_calc*Tc_inv)**0.5
            alpha_root = (1.0 + kappa*(1.0-rt))
            err = c1*T_calc - alpha_root*alpha_root*c2 - P
            derr = c1 + c2*kappa*rt*(kappa*(1.0 -rt) + 1.0)/T_calc
            T_calc = T_calc - err/derr

            return T_calc


            c1, c2 = R/(V_m_b), a/(V*(V+b) + b*V_m_b)

            rt = (T_calc_high*Tc_inv)**0.5
            alpha_root = (1.0 + kappa*(1.0-rt))
            err = c1*T_calc_high - alpha_root*alpha_root*c2 - P

            # Newton step - might as well compute it
            derr = c1 + c2*kappa*rt*(kappa*(1.0 -rt) + 1.0)/T_calc_high
            T_calc_high = T_calc_high - err/derr

            # Step 2 - cannot find occasion to need more steps, most of the time
            # this does nothing!
            rt = (T_calc_high*Tc_inv)**0.5
            alpha_root = (1.0 + kappa*(1.0-rt))
            err = c1*T_calc_high - alpha_root*alpha_root*c2 - P
            derr = c1 + c2*kappa*rt*(kappa*(1.0 -rt) + 1.0)/T_calc_high
            T_calc_high = T_calc_high - err/derr





            delta, epsilon = self.delta, self.epsilon
            w0 = 1.0*(delta*delta - 4.0*epsilon)**-0.5
            w1 = delta*w0
            w2 = 2.0*w0

#            print(T_calc, T_calc_high)

            a_alpha_low = a*(1.0 + kappa*(1.0-(T_calc/Tc)**0.5))**2.0
            a_alpha_high = a*(1.0 + kappa*(1.0-(T_calc_high/Tc)**0.5))**2.0

            err_low = abs(R*T_calc/(V-b) - a_alpha_low/(V*V + delta*V + epsilon) - P)
            err_high = abs(R*T_calc_high/(V-b) - a_alpha_high/(V*V + delta*V + epsilon) - P)
#                print(err_low, err_high, T_calc, T_calc_high, a_alpha_low, a_alpha_high)

            RT_low = R*T_calc
            G_dep_low = (P*V - RT_low - RT_low*clog(P/RT_low*(V-b)).real
                        - w2*a_alpha_low*catanh(2.0*V*w0 + w1).real)

            RT_high = R*T_calc_high
            G_dep_high = (P*V - RT_high - RT_high*clog(P/RT_high*(V-b)).real
                        - w2*a_alpha_high*catanh(2.0*V*w0 + w1).real)

#                print(G_dep_low, G_dep_high)
            # ((err_low > err_high*2)) and
            if  (T_calc.imag != 0.0 and T_calc_high.imag == 0.0) or (G_dep_high < G_dep_low and (err_high < err_low)):
                T_calc = T_calc_high

            return T_calc


#            if err_high < err_low:
#                T_calc = T_calc_high

#            for Ti in (T_calc, T_calc_high):
#                a_alpha = a*(1.0 + kappa*(1.0-(Ti/Tc)**0.5))**2.0
#
#
#                # Compute P, and the difference?
#                self.P = float(R*self.T/(V-self.b) - self.a_alpha/(V*V + self.delta*V + self.epsilon)
#
#
#
#                RT = R*Ti
#                print(RT, V-b, P/RT*(V-b))
#                G_dep = (P*V - RT - RT*log(P/RT*(V-b))
#                            - w2*a_alpha*catanh(2.0*V*w0 + w1).real)
#                print(G_dep)
#                if G_dep < G_dep_base:
#                    T = Ti
#                    G_dep_base = G_dep
#            T_calc = T

#            print(T_calc, T_calc_high)


#            T_calc = (-Tc*(2.*a*kappa*x11*sqrt(V_m_b**3*(x0 + x6 - x8)*(P*x7 -
#                                              P*x9 + x25 + x33 + x34 + x35
#                                              + x36 - x37))*(kappa + 1.) -
#                x31*V_m_b*((4.*V)*(R*Tc*a*b*kappa) + x0*x33 - x0*x35 + x12*x38
#                         + x16*x38 + x18*x39 - x18*x41 - x20*x42 - x22*x42
#                         - x23*x38 + x24*x38 + x25*x6 - x26 - x27 + x28 + x29
#                         + x3*x39 - x3*x41 + x30*x34 - x33*x8 + x36*x6
#                         + 3*x37*x8 + x39*x40 - x40*x41))/(x11*x31))
#            print(T_calc2/T_calc)

            # Validation code - although the solution is analytical some issues
            # with floating points can still occur
            # Although 99.9 % of points anyone would likely want are plenty good,
            # there are some edge cases as P approaches T or goes under it.

#            c1, c2 = R/(V_m_b), a/(V*(V+b) + b*V_m_b)
#
#            rt = (T_calc*Tc_inv)**0.5
#            alpha_root = (1.0 + kappa*(1.0-rt))
#            err = c1*T_calc - alpha_root*alpha_root*c2 - P
#
#            # Newton step - might as well compute it
#            derr = c1 + c2*kappa*rt*(kappa*(1.0 -rt) + 1.0)/T_calc
#            T_calc = T_calc - err/derr
#
#            # Step 2 - cannot find occasion to need more steps, most of the time
#            # this does nothing!
#            rt = (T_calc*Tc_inv)**0.5
#            alpha_root = (1.0 + kappa*(1.0-rt))
#            err = c1*T_calc - alpha_root*alpha_root*c2 - P
#            derr = c1 + c2*kappa*rt*(kappa*(1.0 -rt) + 1.0)/T_calc
#            T_calc = T_calc - err/derr
##            print(T_calc)
#            return T_calc

#            P_inv = 1.0/P
#            if abs(err/P) < 1e-6:
#                return T_calc
##            print(abs(err/P))
##            return GCEOS.solve_T(self, P, V)
#            for i in range(7):
#                rt = (T_calc*Tc_inv)**0.5
#                alpha_root = (1.0 + kappa*(1.0-rt))
#                err = c1*T_calc - alpha_root*alpha_root*c2 - P
#                derr = c1 + c2*kappa*rt*(kappa*(1.0 -rt) + 1.0)/T_calc
#
#                T_calc = T_calc - err/derr
#                print(err/P, T_calc, derr)
#                if abs(err/P) < 1e-12:
#                    return T_calc
#            return T_calc


    # starts at 0.0008793111898930736
#    Psat_ranges_low = (0.011527649224138653,
#                       0.15177700441811506, 0.7883172905889053, 2.035659276638337,
#                       4.53501754500169, 10.745446771738406, 22.67639480888016,
#                       50.03388490796283, 104.02786866285064)
    # 2019 Nov
#    Psat_ranges_low = (0.15674244743681393, 0.8119861320343748, 2.094720219302703, 4.960845727141835, 11.067460617890934, 25.621853405705796, 43.198888850643804, 104.02786866285064)
#    Psat_coeffs_low = [[-227953.8193412378, 222859.8202525231, -94946.0644714779, 22988.662866916213, -3436.218010266234, 314.10561626462993, -12.536721169650086, -2.392026378146748, 1.7425442228873158, -1.2062891595039678, 0.9256591091303878, -0.7876053099939332, 0.5624587154041579, -3.3553013976814365, 5.4012350148013866e-14], [0.017979999443171253, -0.1407329351142875, 0.5157655870958351, -1.1824391743389553, 1.9175463304080598, -2.370060249233812, 2.3671981077067543, -2.0211919069051754, 1.5662532616167582, -1.1752554496422438, 0.9211423805826566, -0.7870983088912286, 0.5624192663836626, -3.3552995268181935, -4.056076807756881e-08], [2.3465238783212443e-06, -5.1803023754491137e-05, 0.0005331498955415226, -0.0034021195248914006, 0.015107808977575897, -0.04968952806811015, 0.12578046832772882, -0.25143473221174495, 0.40552536074726614, -0.5443994966086247, 0.6434269285808626, -0.6923484892423339, 0.5390886452491613, -3.3516377955152628, -0.0002734868035272342], [-4.149916661961022e-10, 2.1845922714910234e-08, -5.293093383029167e-07, 7.799519138713084e-06, -7.769053551547911e-05, 0.0005486109959120195, -0.0027872878510967723, 0.010013711509364028, -0.023484350891214936, 0.024784713187904924, 0.04189568427991252, -0.2040017547275196, 0.25395831370937016, -3.2456178797446413, -0.01903130694686439], [5.244405747881219e-16, -1.5454390343008565e-14, -2.0604241377631507e-12, 1.8208689279561933e-10, -7.250743412052849e-09, 1.8247981842001254e-07, -3.226779942705286e-06, 4.21332816427672e-05, -0.00041707954900317614, 0.003173654759907457, -0.01868692125208627, 0.0855653889368932, -0.31035507126284995, -2.6634237299183328, -0.2800897855694018], [-2.1214680302656463e-19, 5.783021422459962e-17, -7.315923275334905e-15, 5.698692571821259e-13, -3.0576045765082714e-11, 1.1975824393534794e-09, -3.540115921441331e-08, 8.052781011110919e-07, -1.424237637885889e-05, 0.00019659116938228988, -0.0021156267397923314, 0.017700252965885416, -0.11593142002481696, -3.013661988282298, 0.01996154251720128], [-2.8970166603270677e-23, 1.694610551839978e-20, -4.467776279776866e-18, 7.096773522723984e-16, -7.632413053542317e-14, 5.906374821509563e-12, -3.4056397726361876e-10, 1.4928364875485495e-08, -5.025465019680778e-07, 1.3027126331371714e-05, -0.00025915855275578494, 0.003928557567224198, -0.04532442889219183, -3.235941699431832, 0.33934709098936366], [-1.0487638177712636e-27, 1.1588074100262264e-24, -5.933272229330526e-22, 1.8676144445612704e-19, -4.0425091708892395e-17, 6.37584823835825e-15, -7.573969719222655e-13, 6.907076002118451e-11, -4.883344880881757e-09, 2.6844313931168583e-07, -1.1443544240867529e-05, 0.0003760349651708502, -0.009520080664949915, -3.464433298845877, 1.0399494170785033]]
    # 2019 Dec 08 #1
#    Psat_ranges_low = ([0.1566663623710075, 0.8122712349481437, 2.0945197784666294, 4.961535043425216, 11.064718660459363, 25.62532893636351, 43.17405809523583, 85.5638421625653, 169.8222874125952)
#    Psat_coeffs_low = [[-6.364470992262544e-23, 1.5661396802352383e-19, -1.788719435685493e-16, 1.2567790299823932e-13, -6.068855158259506e-11, 2.130642024043302e-08, -5.608337854780211e-06, 0.0011243910475529856, -0.17253439771817053, 20.164796917496496, -1766.983966143576, 112571.42973915562, -4928969.89775339, 132767165.35442507, -1659856970.7084315], [-6.755028337063007e-31, 1.2373135465776702e-27, -1.0534911582623026e-24, 5.532082037130418e-22, -2.0042818462405888e-19, 5.3092667094437664e-17, -1.0629813459498251e-14, 1.6396189295145161e-12, -1.9677160870915945e-10, 1.8425759971191095e-08, -1.3425348946576017e-06, 7.562661739651473e-05, -0.0032885862389808195, -3.5452990752336735, 1.5360178058346605], [-5.909795950371768e-27, 5.645060782013921e-24, -2.5062698828832408e-21, 6.861883492029141e-19, -1.2960098086863643e-16, 1.7893963536931406e-14, -1.8669999568680822e-12, 1.5005071785133313e-10, -9.381783948347974e-09, 4.576967837674971e-07, -1.7378660968493725e-05, 0.0005105597560223805, -0.011603105202254462, -3.4447117223858394, 0.9538198797898474], [-2.8780483706946006e-23, 1.4693097909367858e-20, -3.492711723365092e-18, 5.129438453755985e-16, -5.2066819983096923e-14, 3.87131295903126e-12, -2.1797843188384387e-10, 9.475510493050094e-09, -3.212229879279181e-07, 8.520129885652724e-06, -0.00017645941977890718, 0.0028397690069188186, -0.035584878748907235, -3.2889972189483, 0.47227047696507896], [-2.133647784270567e-19, 5.813855761166538e-17, -7.351939324704256e-15, 5.724415520048679e-13, -3.0701524683808055e-11, 1.2020043191332715e-09, -3.5517231986184477e-08, 8.075833591581873e-07, -1.4277180602174389e-05, 0.0001969886336996064, -0.0021190060629508248, 0.017720993486168023, -0.11601827744842373, -3.0134398433062954, 0.019699769017179847], [5.217055552725474e-16, -1.561972494582649e-14, -2.027739589933126e-12, 1.8030004183143271e-10, -7.1961213928967356e-09, 1.8138160781745565e-07, -3.2112101506231723e-06, 4.197218861582643e-05, -0.00041584453068251905, 0.0031666287443832307, -0.018657602063128432, 0.08547811393673718, -0.31017952035114504, -2.6636376461277504, -0.27997050354186115], [-4.1558987320232216e-10, 2.1874838982254277e-08, -5.299524926441045e-07, 7.808241563359814e-06, -7.777110034030892e-05, 0.0005491470176474339, -0.002789936581283384, 0.010023585334231266, -0.023512249664927133, 0.02484416646533969, 0.04180162903589153, -0.20389464760201653, 0.25387532037317434, -3.245578712101638, -0.01903980099778657], [2.3320945490434305e-06, -5.15194336734163e-05, 0.0005305911686609431, -0.003388078003236081, 0.015055473744080193, -0.049549442201717114, 0.12550289037335455, -0.251021291476035, 0.40506041321992375, -0.5440068047537978, 0.6431818377117259, -0.6922389245218481, 0.5390554975784367, -3.3516317236219626, -0.00027399457467680577], [0.017760683349597454, -0.1392342452029993, 0.5111179189769633, -1.1737814955588932, 1.9067391494716879, -2.3605113086814407, 2.361048334775187, -2.0182633656154794, 1.5652184041682835, -1.1749857171593956, 0.92109138142958, -0.7870915307971148, 0.5624186680171368, -3.3552994954150326, -4.130013597780646e-08], [1842638.012244339, -2064103.5077599594, 1029111.4284441478, -300839.92590603326, 57174.96949130112, -7405.305505076668, 668.4504791023379, -43.94219790319933, 3.4634979070792977, -1.2528527563309222, 0.9264289045482768, -0.787612207652486, 0.5624587411994793, -3.3553013976928456, 4.846123502488808e-14]]

    # 2019 Dec 08 #2
#    Psat_ranges_low = (0.15674244743681393, 0.8119861320343748, 2.094720219302703, 4.961535043425216, 11.064718660459363, 25.62532893636351, 43.17405809523583, 85.5638421625653, 169.8222874125952, 192.707581659434)
#    Psat_coeffs_low = [[-393279.9328001248, 414920.88015712175, -194956.1186003408, 53799.692378381624, -9679.442200674115, 1189.1133946984114, -99.38789237175924, 3.7558250389696366, 1.4341105372610397, -1.195532646019414, 0.9254075742030472, -0.7876016031722438, 0.5624586846061402, -3.355301397567417, -2.475797344914099e-14], [0.018200741617324958, -0.14216111513088853, 0.5199706046777292, -1.1898993034816217, 1.9264460624802726, -2.377604380463091, 2.3718790446551283, -2.0233492715449346, 1.5669946704278936, -1.175444344921655, 0.9211774746760774, -0.787102916441927, 0.5624196703434721, -3.3552995479850125, -4.006059328709455e-08], [2.362594082154845e-06, -5.213477214805086e-05, 0.0005363047209564668, -0.0034204334370065157, 0.015180294585886198, -0.04989640532490752, 0.1262194343941631, -0.252138050376706, 0.4063802322466773, -0.5451837881722801, 0.643961448026334, -0.6926108644042617, 0.5391763183580807, -3.3516556444811516, -0.00027181665396192045], [-4.1566510211197074e-10, 2.1878563345656593e-08, -5.30037387599558e-07, 7.809422248533072e-06, -7.77822904769859e-05, 0.0005492234565335112, -0.002790324592151159, 0.010025071882175543, -0.023516568419967406, 0.024853633218471893, 0.04178621870041742, -0.20387658476895476, 0.2538609101701838, -3.2455717084245443, -0.019041365569938407], [5.952860605957254e-16, -2.3560872386568428e-14, -1.6328974906691505e-12, 1.6831386671561567e-10, -6.947967158882692e-09, 1.77675502929117e-07, -3.170039732850266e-06, 4.162662881336586e-05, -0.0004136425496617131, 0.0031560285189308705, -0.018619655683130842, 0.085380163769752, -0.3100071777702119, -2.6638226631426187, -0.279879068340815], [-2.1336825570293267e-19, 5.813946215182557e-17, -7.352047876443287e-15, 5.724495165386215e-13, -3.0701923762367554e-11, 1.2020187632285275e-09, -3.5517621350872006e-08, 8.075912994222895e-07, -1.4277303680626562e-05, 0.00019699007656794466, -0.00211901865445771, 0.01772107279538477, -0.1160186182468458, -3.0134389491023668, 0.019698688209032866], [-2.8780483706946006e-23, 1.4693097909367858e-20, -3.492711723365092e-18, 5.129438453755985e-16, -5.2066819983096923e-14, 3.87131295903126e-12, -2.1797843188384387e-10, 9.475510493050094e-09, -3.212229879279181e-07, 8.520129885652724e-06, -0.00017645941977890718, 0.0028397690069188186, -0.035584878748907235, -3.2889972189483, 0.47227047696507896], [-5.909795950371768e-27, 5.645060782013921e-24, -2.5062698828832408e-21, 6.861883492029141e-19, -1.2960098086863643e-16, 1.7893963536931406e-14, -1.8669999568680822e-12, 1.5005071785133313e-10, -9.381783948347974e-09, 4.576967837674971e-07, -1.7378660968493725e-05, 0.0005105597560223805, -0.011603105202254462, -3.4447117223858394, 0.9538198797898474], [-6.755028337063007e-31, 1.2373135465776702e-27, -1.0534911582623026e-24, 5.532082037130418e-22, -2.0042818462405888e-19, 5.3092667094437664e-17, -1.0629813459498251e-14, 1.6396189295145161e-12, -1.9677160870915945e-10, 1.8425759971191095e-08, -1.3425348946576017e-06, 7.562661739651473e-05, -0.0032885862389808195, -3.5452990752336735, 1.5360178058346605], [-6.364470992262544e-23, 1.5661396802352383e-19, -1.788719435685493e-16, 1.2567790299823932e-13, -6.068855158259506e-11, 2.130642024043302e-08, -5.608337854780211e-06, 0.0011243910475529856, -0.17253439771817053, 20.164796917496496, -1766.983966143576, 112571.42973915562, -4928969.89775339, 132767165.35442507, -1659856970.7084315]]
    # 2019 Dec 08 #3
    Psat_ranges_low = (0.038515189998761204, 0.6472853332269844, 2.0945197784666294, 4.961232873814024, 11.067553885784903, 25.624838497870584, 43.20169529076582, 85.5588271726612, 192.72834691988226)
    Psat_coeffs_low = [[2338676895826482.5, -736415034973095.6, 105113277697825.1, -8995168780410.754, 514360029044.81494, -20734723655.83978, 605871516.8891307, -12994014.122638363, 204831.11357912835, -2351.9913154464143, 18.149657683324232, 0.8151930684866298, -0.7871881357728392, 0.5624577476810062, -3.35530139647672, -4.836964162535651e-13], [-0.13805715433070773, 0.8489231609102119, -2.450329797856018, 4.447856574793218, -5.767299107094559, 5.794674157897756, -4.825296555657044, 3.5520183799445926, -2.4600869594916634, 1.6909163275418595, -1.2021498414235525, 0.9254639369127162, -0.7875982246546266, 0.5624585116206676, -3.3553013938160787, -3.331224185387782e-11], [-2.3814071133383825e-06, 5.318261908739265e-05, -0.0005538990617858645, 0.0035761255785055936, -0.016054997425247523, 0.05333504500541739, -0.13636391080337568, 0.27593424749870343, -0.4517901507372948, 0.6114112167354924, -0.7059858408782421, 0.7385376731146207, -0.7329884294338728, 0.5509890744823249, -3.353773232516225, -9.646546737407391e-05], [2.6058661808460023e-11, -1.75914103924121e-09, 5.396299167286894e-08, -1.0007922530068192e-06, 1.2554484077194732e-05, -0.0001125821062183067, 0.0007410322067253991, -0.0035992993229111833, 0.012657105041028169, -0.030121969848977304, 0.03753504314148813, 0.02349666014556937, -0.18469580367455368, 0.24005237728233714, -3.239469690554324, -0.020289142467969867], [-1.082394018559102e-15, 1.2914854481231322e-13, -7.104839518580019e-12, 2.3832489222439473e-10, -5.425087002560749e-09, 8.804418548276272e-08, -1.0364065054630989e-06, 8.719985338278278e-06, -4.8325538208084174e-05, 0.00011200959608941485, 0.0008028675551716892, -0.010695106054891056, 0.06594801536296582, -0.27725262867260253, -2.6977571369079514, -0.2635895959694814], [1.1488824622125947e-20, -3.331154652317046e-18, 4.503372697637035e-16, -3.7684497582121125e-14, 2.1852058912840643e-12, -9.313780852814459e-11, 3.019939074381905e-09, -7.605074783395472e-08, 1.5052679183948458e-06, -2.354701523431422e-05, 0.00029127690705745875, -0.0028399757838276493, 0.02173245057169364, -0.13135011490812692, -2.9774476427885146, -0.01942256817236654], [1.0436558787976772e-24, -5.473723131383567e-22, 1.3452696879486453e-19, -2.0573736968717295e-17, 2.1924486360657888e-15, -1.7272619586846295e-13, 1.0413985148866247e-11, -4.906312890258065e-10, 1.8279149292524938e-08, -5.414588408693672e-07, 1.275367009914141e-05, -0.00023786604002741, 0.0034903075344121025, -0.04033658323380905, -3.2676007023496245, 0.42749816097639837], [9.060766533667912e-29, -9.196819760777788e-26, 4.3601925662975664e-23, -1.2818245897574232e-20, 2.615903295904718e-18, -3.930631843509798e-16, 4.500311702777485e-14, -4.007582103109645e-12, 2.808196479352211e-10, -1.5562164421777763e-08, 6.818206236433737e-07, -2.350273523243411e-05, 0.0006326097721162514, -0.013277937187152783, -3.4305615375066876, 0.8983326523220114], [1.1247677438654667e-33, -2.4697583969349065e-30, 2.5286510080356973e-27, -1.6024926981128421e-24, 7.03655740810716e-22, -2.2705238015446456e-19, 5.57121222696514e-17, -1.0609879702627998e-14, 1.5863699537553053e-12, -1.8713657213281574e-10, 1.7407548458856668e-08, -1.2702047168798462e-06, 7.210856106809965e-05, -0.0031754110755806966, -3.5474790036315795, 1.555110704923493]]



class PR78(PR):
    r'''Class for solving the Peng-Robinson cubic
    equation of state for a pure compound according to the 1978 variant [1]_ [2]_.
    Subclasses :obj:`PR`, which provides everything except the variable `kappa`.
    Solves the EOS on initialization. See :obj:`PR` for further documentation.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

    .. math::
        a=0.45724\frac{R^2T_c^2}{P_c}

    .. math::
        b=0.07780\frac{RT_c}{P_c}

    .. math::
        \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2

    .. math::
        \kappa_i = 0.37464+1.54226\omega_i-0.26992\omega_i^2 \text{ if } \omega_i
        \le 0.491

    .. math::
        \kappa_i = 0.379642 + 1.48503 \omega_i - 0.164423\omega_i^2 + 0.016666
        \omega_i^3 \text{ if } \omega_i > 0.491

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    P-T initialization (furfuryl alcohol), liquid phase:

    >>> eos = PR78(Tc=632, Pc=5350000, omega=0.734, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 8.3519628969e-05, -63764.671093, -130.737153225)

    Notes
    -----
    This variant is recommended over the original.

    References
    ----------
    .. [1] Robinson, Donald B, and Ding-Yu Peng. The Characterization of the
       Heptanes and Heavier Fractions for the GPA Peng-Robinson Programs.
       Tulsa, Okla.: Gas Processors Association, 1978.
    .. [2] Robinson, Donald B., Ding-Yu Peng, and Samuel Y-K Chung. "The
       Development of the Peng - Robinson Equation and Its Application to Phase
       Equilibrium in a System Containing Methanol." Fluid Phase Equilibria 24,
       no. 1 (January 1, 1985): 25-41. doi:10.1016/0378-3812(85)87035-7.
    '''

    low_omega_constants = (0.37464, 1.54226, -0.26992)
    """Constants for the `kappa` formula for the low-omega region."""

    high_omega_constants = (0.379642, 1.48503, -0.164423, 0.016666)
    """Constants for the `kappa` formula for the high-omega region."""

    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1R2*Tc*Tc/Pc
        self.b = b = self.c2R*Tc/Pc
        self.delta, self.epsilon = 2.0*b, -b*b

        if omega <= 0.491:
            self.kappa = 0.37464 + omega*(1.54226 - 0.26992*omega)
        else:
            self.kappa = omega*(omega*(0.016666*omega - 0.164423) + 1.48503) + 0.379642

        self.solve()

class PRTranslated(PR):
    r'''Class for solving the volume translated Peng-Robinson equation of state.
    Subclasses :obj:`PR`. Solves the EOS on initialization.
    This is intended as a base class for all translated variants of the
    Peng-Robinson EOS.

    .. math::
        P = \frac{RT}{v + c - b} - \frac{a\alpha(T)}{(v+c)(v + c + b)+b(v
        + c - b)}

    .. math::
        a=0.45724\frac{R^2T_c^2}{P_c}

    .. math::
        b=0.07780\frac{RT_c}{P_c}

    .. math::
        \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2

    .. math::
        \kappa=0.37464+1.54226\omega-0.26992\omega^2

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    alpha_coeffs : tuple or None
        Coefficients which may be specified by subclasses; set to None to use
        the original Peng-Robinson alpha function, [-]
    c : float, optional
        Volume translation parameter, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    P-T initialization:

    >>> eos = PRTranslated(T=305, P=1.1e5, Tc=512.5, Pc=8084000.0, omega=0.559, c=-1e-6)
    >>> eos.phase, eos.V_l, eos.V_g
    ('l/g', 4.90798083711e-05, 0.0224350982488)

    Notes
    -----

    References
    ----------
    .. [1] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen Rarey.
       Chemical Thermodynamics for Process Simulation. John Wiley & Sons, 2019.
    '''

    solve_T = GCEOS.solve_T
    P_max_at_V = GCEOS.P_max_at_V
    kwargs_keys = ('c', 'alpha_coeffs')
    def __init__(self, Tc, Pc, omega, alpha_coeffs=None, c=0.0, T=None, P=None,
                 V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        Pc_inv = 1.0/Pc

        self.a = self.c1R2*Tc*Tc*Pc_inv

        self.c = c
        if alpha_coeffs is None:
            self.kappa = omega*(-0.26992*omega + 1.54226) + 0.37464

        # Does not have an impact on phase equilibria
        self.alpha_coeffs = alpha_coeffs
        self.kwargs = {'c': c, 'alpha_coeffs': alpha_coeffs}
#        self.C0, self.C1, self.C2 = Twu_coeffs

        b0 = self.c2*R*Tc*Pc_inv
        self.b = b = b0 - c

        # Cannot reference b directly
        self.delta = 2.0*(c + b0)
        self.epsilon = -b0*b0 + c*c + 2.0*c*b0

        # C**2 + 2*C*b + V**2 + V*(2*C + 2*b) - b**2

        self.solve()


class PRTranslatedPPJP(PRTranslated):
    r'''Class for solving the volume translated Pina-Martinez, Privat, Jaubert,
    and Peng revision of the Peng-Robinson equation of state
    for a pure compound according to [1]_.
    Subclasses :obj:`PRTranslated`, which provides everything except the variable `kappa`.
    Solves the EOS on initialization. See :obj:`PRTranslated` for further documentation.

    .. math::
        P = \frac{RT}{v + c - b} - \frac{a\alpha(T)}{(v+c)(v + c + b)+b(v
        + c - b)}

    .. math::
        a=0.45724\frac{R^2T_c^2}{P_c}

    .. math::
        b=0.07780\frac{RT_c}{P_c}

    .. math::
        \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2

    .. math::
        \kappa = 0.3919 + 1.4996 \omega - 0.2721\omega^2 + 0.1063\omega^3

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    c : float, optional
        Volume translation parameter, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    P-T initialization (methanol), liquid phase:

    >>> eos = PRTranslatedPPJP(Tc=507.6, Pc=3025000, omega=0.2975, c=0.6390E-6, T=250., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.0001229231238092, -33466.2428296, -80.75610242427)

    Notes
    -----
    This variant offers incremental improvements in accuracy only, but those
    can be fairly substantial for some substances.

    References
    ----------
    .. [1] Pina-Martinez, Andrés, Romain Privat, Jean-Noël Jaubert, and
       Ding-Yu Peng. "Updated Versions of the Generalized Soave a-Function
       Suitable for the Redlich-Kwong and Peng-Robinson Equations of State."
       Fluid Phase Equilibria, December 7, 2018.
       https://doi.org/10.1016/j.fluid.2018.12.007.
    '''

    # Direct solver for T could be implemented but cannot use the PR one
    kwargs_keys = ('c',)
    def __init__(self, Tc, Pc, omega, c=0.0, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        Pc_inv = 1.0/Pc
        self.a = self.c1*R2*Tc*Tc*Pc_inv
        self.c = c
        # 0.3919 + 1.4996*omega - 0.2721*omega**2+0.1063*omega**3
        self.kappa = omega*(omega*(0.1063*omega - 0.2721) + 1.4996) + 0.3919
        self.kwargs = {'c': c}
        b0 = self.c2*R*Tc*Pc_inv
        self.b = b = b0 - c

        self.delta = 2.0*(c + b0)
        self.epsilon = -b0*b0 + c*c + 2.0*c*b0
        self.solve()

    def P_max_at_V(self, V):
        if self.c == 0.0:
            return PR.P_max_at_V(self, V)
        return None

class PRTranslatedPoly(Poly_a_alpha, PRTranslated):
    r'''Class for solving the volume translated Peng-Robinson equation of state
    with a polynomial alpha function. With the right coefficients, this model
    can reproduce any property incredibly well.
    Subclasses :obj:`PRTranslated`. Solves the EOS on initialization.
    This is intended as a base class for all translated variants of the
    Peng-Robinson EOS.

    .. math::
        P = \frac{RT}{v + c - b} - \frac{a\alpha(T)}{(v+c)(v + c + b)+b(v
        + c - b)}

    .. math::
        a=0.45724\frac{R^2T_c^2}{P_c}

    .. math::
        b=0.07780\frac{RT_c}{P_c}

    .. math::
        \alpha(T)=f(T)

    .. math::
        \kappa=0.37464+1.54226\omega-0.26992\omega^2

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    alpha_coeffs : tuple or None
        Coefficients which may be specified by subclasses; set to None to use
        the original Peng-Robinson alpha function, [-]
    c : float, optional
        Volume translation parameter, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    Methanol, with alpha functions reproducing CoolProp's implementation of
    its vapor pressure (up to 13 coefficients)

    >>> alpha_coeffs_exact = [9.645280470011588e-32, -4.362226651748652e-28, 9.034194757823037e-25, -1.1343330204981244e-21, 9.632898335494218e-19, -5.841502902171077e-16, 2.601801729901228e-13, -8.615431349241052e-11, 2.1202999753932622e-08, -3.829144045293198e-06, 0.0004930777289075716, -0.04285337965522619, 2.2473964123842705, -51.13852710672087]
    >>> kwargs = dict(Tc=512.5, Pc=8084000.0, omega=0.559, alpha_coeffs=alpha_coeffs_exact, c=1.557458e-05)
    >>> eos = PRTranslatedPoly(T=300, P=1e5, **kwargs)
    >>> eos.Psat(500)/PropsSI("P", 'T', 500.0, 'Q', 0, 'methanol') # doctest:+SKIP
    1.0000112765

    Notes
    -----

    '''

class PRTranslatedMathiasCopeman(Mathias_Copeman_poly_a_alpha, PRTranslated):
    pass

class PRTranslatedCoqueletChapoyRichon(PRTranslatedMathiasCopeman):
    kwargs_keys = ('c', 'alpha_coeffs')
    def __init__(self, Tc, Pc, omega, c=0.0, alpha_coeffs=None, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        Pc_inv = 1.0/Pc
        self.a = self.c1*R2*Tc*Tc*Pc_inv
        self.c = c
        if alpha_coeffs is None:
            c1 = omega*(0.1316*omega + 1.4031) + 0.3906
            c2 = omega*(-1.3127*omega + 0.3015) - 0.1213
            c3 = 0.7661*omega + 0.3041
            alpha_coeffs = [c3, c2, c1, 1.0]
        elif alpha_coeffs[-1] != 1.0:
            alpha_coeffs = list(alpha_coeffs)
            alpha_coeffs.append(1.0)

        self.kwargs = {'c': c, 'alpha_coeffs': alpha_coeffs}
        self.alpha_coeffs = alpha_coeffs
        b0 = self.c2*R*Tc*Pc_inv
        self.b = b = b0 - c

        self.delta = 2.0*(c + b0)
        self.epsilon = -b0*b0 + c*c + 2.0*c*b0
        self.solve()


class PRTranslatedTwu(Twu91_a_alpha, PRTranslated):
    r'''Class for solving the volume translated Peng-Robinson equation of state
    with the Twu (1991) [1]_ alpha function.
    Subclasses :obj:`thermo.eos_alpha_functions.Twu91_a_alpha` and :obj:`PRTranslated`.
    Solves the EOS on initialization.

    .. math::
        P = \frac{RT}{v + c - b} - \frac{a\alpha(T)}{(v+c)(v + c + b)+b(v
        + c - b)}

    .. math::
        a=0.45724\frac{R^2T_c^2}{P_c}

    .. math::
        b=0.07780\frac{RT_c}{P_c}

    .. math::
        \alpha = \left(\frac{T}{T_{c}}\right)^{c_{3} \left(c_{2}
        - 1\right)} e^{c_{1} \left(- \left(\frac{T}{T_{c}}
        \right)^{c_{2} c_{3}} + 1\right)}

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    alpha_coeffs : tuple(float[3])
        Coefficients L, M, N (also called C1, C2, C3) of TWU 1991 form, [-]
    c : float, optional
        Volume translation parameter, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    P-T initialization:

    >>> alpha_coeffs = (0.694911381318495, 0.919907783415812, 1.70412689631515)
    >>> kwargs = dict(Tc=512.5, Pc=8084000.0, omega=0.559, alpha_coeffs=alpha_coeffs, c=-1e-6)
    >>> eos = PRTranslatedTwu(T=300, P=1e5, **kwargs)
    >>> eos.phase, eos.V_l, eos.V_g
    ('l/g', 4.8918748906e-05, 0.024314406330)

    Notes
    -----
    This variant offers substantial improvements to the PR-type EOSs - likely
    getting about as accurate as this form of cubic equation can get.

    References
    ----------
    .. [1] Twu, Chorng H., David Bluck, John R. Cunningham, and John E.
       Coon. "A Cubic Equation of State with a New Alpha Function and a
       New Mixing Rule." Fluid Phase Equilibria 69 (December 10, 1991):
       33-50. doi:10.1016/0378-3812(91)90024-2.
    '''

class PRTranslatedConsistent(PRTranslatedTwu):
    r'''Class for solving the volume translated Le Guennec, Privat, and Jaubert
    revision of the Peng-Robinson equation of state
    for a pure compound according to [1]_.
    Subclasses :obj:`PRTranslatedTwu`, which provides everything except the
    estimation of `c` and the alpha coefficients. This model's `alpha` is based
    on the TWU 1991 model; when estimating, `N` is set to 2.
    Solves the EOS on initialization. See :obj:`PRTranslated` for further documentation.

    .. math::
        P = \frac{RT}{v + c - b} - \frac{a\alpha(T)}{(v+c)(v + c + b)+b(v
        + c - b)}

    .. math::
        a=0.45724\frac{R^2T_c^2}{P_c}

    .. math::
        b=0.07780\frac{RT_c}{P_c}

    .. math::
        \alpha = \left(\frac{T}{T_{c}}\right)^{c_{3} \left(c_{2}
        - 1\right)} e^{c_{1} \left(- \left(\frac{T}{T_{c}}
        \right)^{c_{2} c_{3}} + 1\right)}

    If `c` is not provided, it is estimated as:

    .. math::
        c =\frac{R T_c}{P_c}(0.0198\omega - 0.0065)

    If `alpha_coeffs` is not provided, the parameters `L` and `M` are estimated
    from the acentric factor as follows:

    .. math::
        L = 0.1290\omega^2 + 0.6039\omega + 0.0877

    .. math::
        M = 0.1760\omega^2 - 0.2600\omega + 0.8884

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    alpha_coeffs : tuple(float[3]), optional
        Coefficients L, M, N (also called C1, C2, C3) of TWU 1991 form, [-]
    c : float, optional
        Volume translation parameter, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    P-T initialization (methanol), liquid phase:

    >>> eos = PRTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=250., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.000124374813374486, -34155.16119794619, -83.34913258614345)

    Notes
    -----
    This variant offers substantial improvements to the PR-type EOSs - likely
    getting about as accurate as this form of cubic equation can get.

    References
    ----------
    .. [1] Le Guennec, Yohann, Romain Privat, and Jean-Noël Jaubert.
       "Development of the Translated-Consistent Tc-PR and Tc-RK Cubic
       Equations of State for a Safe and Accurate Prediction of Volumetric,
       Energetic and Saturation Properties of Pure Compounds in the Sub- and
       Super-Critical Domains." Fluid Phase Equilibria 429 (December 15, 2016):
       301-12. https://doi.org/10.1016/j.fluid.2016.09.003.
    '''

    kwargs_keys = ('c', 'alpha_coeffs')
    def __init__(self, Tc, Pc, omega, alpha_coeffs=None, c=None, T=None,
                 P=None, V=None):
        # estimates volume translation and alpha function parameters
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V
        Pc_inv = 1.0/Pc

        # Limit the fitting omega to a little under the range reported
        o = min(max(omega, -0.01), 1.48)
        if c is None:
            c = R*Tc*Pc_inv*(0.0198*o - 0.0065)

        if alpha_coeffs is None:
            L = o*(0.1290*o + 0.6039) + 0.0877
            M = o*(0.1760*o - 0.2600) + 0.8884
            N = 2.0
            alpha_coeffs = (L, M, N)

        self.c = c
        self.alpha_coeffs = alpha_coeffs
        self.kwargs = {'c': c, 'alpha_coeffs': alpha_coeffs}

        self.a = self.c1*R2*Tc*Tc*Pc_inv
        b0 = self.c2*R*Tc*Pc_inv
        self.b = b = b0 - c

        self.delta = 2.0*(c + b0)
        self.epsilon = -b0*b0 + c*c + 2.0*c*b0

        self.solve()

class PRSV(PR):
    r'''Class for solving the Peng-Robinson-Stryjek-Vera equations of state for
    a pure compound as given in [1]_. The same as the Peng-Robinson EOS,
    except with a different `kappa` formula and with an optional fit parameter.
    Subclasses :obj:`PR`, which provides only several constants. See :obj:`PR` for
    further documentation and examples.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

    .. math::
        a=0.45724\frac{R^2T_c^2}{P_c}

    .. math::
        b=0.07780\frac{RT_c}{P_c}

    .. math::
        \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2

    .. math::
        \kappa = \kappa_0 + \kappa_1(1 + T_r^{0.5})(0.7 - T_r)

    .. math::
        \kappa_0 = 0.378893 + 1.4897153\omega - 0.17131848\omega^2
        + 0.0196554\omega^3

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    kappa1 : float, optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]

    Examples
    --------
    P-T initialization (hexane, with fit parameter in [1]_), liquid phase:

    >>> eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.000130126913554, -31698.926746, -74.16751538)

    Notes
    -----
    [1]_ recommends that `kappa1` be set to 0 for Tr > 0.7. This is not done by
    default; the class boolean `kappa1_Tr_limit` may be set to True and the
    problem re-solved with that specified if desired. `kappa1_Tr_limit` is not
    supported for P-V inputs.

    Solutions for P-V solve for `T` with SciPy's `newton` solver, as there is no
    analytical solution for `T`

    [2]_ and [3]_ are two more resources documenting the PRSV EOS. [4]_ lists
    `kappa` values for 69 additional compounds. See also `PRSV2`. Note that
    tabulated `kappa` values should be used with the critical parameters used
    in their fits. Both [1]_ and [4]_ only considered vapor pressure in fitting
    the parameter.

    References
    ----------
    .. [1] Stryjek, R., and J. H. Vera. "PRSV: An Improved Peng-Robinson
       Equation of State for Pure Compounds and Mixtures." The Canadian Journal
       of Chemical Engineering 64, no. 2 (April 1, 1986): 323-33.
       doi:10.1002/cjce.5450640224.
    .. [2] Stryjek, R., and J. H. Vera. "PRSV - An Improved Peng-Robinson
       Equation of State with New Mixing Rules for Strongly Nonideal Mixtures."
       The Canadian Journal of Chemical Engineering 64, no. 2 (April 1, 1986):
       334-40. doi:10.1002/cjce.5450640225.
    .. [3] Stryjek, R., and J. H. Vera. "Vapor-liquid Equilibrium of
       Hydrochloric Acid Solutions with the PRSV Equation of State." Fluid
       Phase Equilibria 25, no. 3 (January 1, 1986): 279-90.
       doi:10.1016/0378-3812(86)80004-8.
    .. [4] Proust, P., and J. H. Vera. "PRSV: The Stryjek-Vera Modification of
       the Peng-Robinson Equation of State. Parameters for Other Pure Compounds
       of Industrial Interest." The Canadian Journal of Chemical Engineering
       67, no. 1 (February 1, 1989): 170-73. doi:10.1002/cjce.5450670125.
    '''

    kappa1_Tr_limit = False
    kwargs_keys = ('kappa1',)
    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None, kappa1=None):
        self.T, self.P, self.V, self.omega, self.Tc, self.Pc = T, P, V, omega, Tc, Pc

        if kappa1 is None:
            kappa1 = 0.0
        self.kwargs = {'kappa1': kappa1}

        self.b = b = self.c2R*Tc/Pc
        self.a = self.c1R2_c2R*Tc*b
        self.delta = 2.0*b
        self.epsilon = -b*b
        self.kappa0 = omega*(omega*(0.0196554*omega - 0.17131848) + 1.4897153) + 0.378893

        self.check_sufficient_inputs()
        if V and P:
            # Deal with T-solution here; does NOT support kappa1_Tr_limit.
            self.kappa1 = kappa1
            self.T = T = self.solve_T(P, V)
            Tr = T/Tc
        else:
            Tr = T/Tc
            if self.kappa1_Tr_limit and Tr > 0.7:
                self.kappa1 = 0.0
            else:
                self.kappa1 = kappa1

        self.kappa = self.kappa0 + self.kappa1*(1.0 + sqrt(Tr))*(0.7 - Tr)
        self.solve()

    def solve_T(self, P, V, solution=None):
        r'''Method to calculate `T` from a specified `P` and `V` for the PRSV
        EOS. Uses `Tc`, `a`, `b`, `kappa0`  and `kappa` as well, obtained from
        the class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        solution : str or None, optional
            'l' or 'g' to specify a liquid of vapor solution (if one exists);
            if None, will select a solution more likely to be real (closer to
            STP, attempting to avoid temperatures like 60000 K or 0.0001 K).

        Returns
        -------
        T : float
            Temperature, [K]

        Notes
        -----
        Not guaranteed to produce a solution. There are actually two solution,
        one much higher than normally desired; it is possible the solver could
        converge on this.
        '''
        Tc, a, b, kappa0, kappa1 = self.Tc, self.a, self.b, self.kappa0, self.kappa1
        self.no_T_spec = True
        x0 = V - b
        R_x0 = R/x0
        x3_inv = 1.0/(100.*(V*(V + b) + b*x0))
        x4 = 10.*kappa0
        kappa110 = kappa1*10.
        kappa17 = kappa1*7.
        Tc_inv = 1.0/Tc
        x51 = x3_inv*a
        def to_solve(T):
            x1 = T*Tc_inv
            x2 = x1**0.5
            x10 = ((x4 - (kappa110*x1 - kappa17)*(x2 + 1.))*(x2 - 1.) - 10.)
            x11 = T*R_x0 - P
            return x11 - x10*x10*x51
        if solution is None:
            try:
                return secant(to_solve, Tc*0.5)
            except:
                pass
            # The above method handles fewer cases, but the below is less optimized
        return GCEOS.solve_T(self, P, V, solution=solution)

    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate :math:`a \alpha` and its first and second
        derivatives for this EOS. Uses the set values of `Tc`, `kappa0`,
        `kappa1`, and `a`.

        The `a_alpha` function is shown below; the first and second derivatives
        are not shown for brevity.

        .. math::
            a\alpha = a \left(\left(\kappa_{0} + \kappa_{1} \left(\sqrt{\frac{
            T}{T_{c}}} + 1\right) \left(- \frac{T}{T_{c}} + \frac{7}{10}\right)
            \right) \left(- \sqrt{\frac{T}{T_{c}}} + 1\right) + 1\right)^{2}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]

        Notes
        -----
        This method does not alter the object's state and the temperature
        provided can be a different than that of the object.

        The expressions can be derived as follows:

        >>> from sympy import *
        >>> P, T, V = symbols('P, T, V')
        >>> Tc, Pc, omega = symbols('Tc, Pc, omega')
        >>> R, a, b, kappa0, kappa1 = symbols('R, a, b, kappa0, kappa1')
        >>> kappa = kappa0 + kappa1*(1 + sqrt(T/Tc))*(Rational(7, 10)-T/Tc)
        >>> a_alpha = a*(1 + kappa*(1-sqrt(T/Tc)))**2
        >>> # diff(a_alpha, T)
        >>> # diff(a_alpha, T, 2)

        Examples
        --------
        >>> eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=406.08, P=1E6, kappa1=0.05104)
        >>> eos.a_alpha_and_derivatives_pure(185.0)
        (4.76865472591, -0.0101408587212, 3.9138298092e-05)
        '''
        Tc, a, kappa0, kappa1 = self.Tc, self.a, self.kappa0, self.kappa1
        x1 = T/Tc
        T_inv = 1.0/T
        x2 = sqrt(x1)
        x3 = x2 - 1.
        x4 = 10.*x1 - 7.
        x5 = x2 + 1.
        x6 = 10.*kappa0 - kappa1*x4*x5
        x7 = x3*x6
        x8 = x7*0.1 - 1.
        x10 = x6*T_inv
        x11 = kappa1*x3
        x12 = x4*T_inv
        x13 = 20./Tc*x5 + x12*x2
        x14 = -x10*x2 + x11*x13
        a_alpha = a*x8*x8
        da_alpha_dT = -a*x14*x8*0.1
        d2a_alpha_dT2 = a*(x14*x14 - x2*T_inv*(x7 - 10.)*(2.*kappa1*x13 + x10 + x11*(40.*x1*T_inv - x12)))*(1.0/200.)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        r'''Method to calculate :math:`a \alpha` for this EOS. Uses the set values of `Tc`, `kappa0`,
        `kappa1`, and `a`.

        .. math::
            a\alpha = a \left(\left(\kappa_{0} + \kappa_{1} \left(\sqrt{\frac{
            T}{T_{c}}} + 1\right) \left(- \frac{T}{T_{c}} + \frac{7}{10}\right)
            \right) \left(- \sqrt{\frac{T}{T_{c}}} + 1\right) + 1\right)^{2}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the value, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

        Notes
        -----
        This method does not alter the object's state and the temperature
        provided can be a different than that of the object.

        Examples
        --------
        >>> eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=406.08, P=1E6, kappa1=0.05104)
        >>> eos.a_alpha_pure(185.0)
        4.7686547259
        '''
        Tc, a, kappa0, kappa1 = self.Tc, self.a, self.kappa0, self.kappa1
        Tr = T/Tc
        sqrtTr = sqrt(Tr)
        v = ((kappa0 + kappa1*(sqrtTr + 1.0)*(-Tr + 0.7))*(-sqrtTr + 1.0) + 1.0)
        return a*v*v


class PRSV2(PR):
    r'''Class for solving the Peng-Robinson-Stryjek-Vera 2 equations of state
    for a pure compound as given in [1]_. The same as the Peng-Robinson EOS,
    except with a different `kappa` formula and with three fit parameters.
    Subclasses :obj:`PR`, which provides only several constants. See :obj:`PR` for
    further documentation and examples.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

    .. math::
        a=0.45724\frac{R^2T_c^2}{P_c}

    .. math::
        b=0.07780\frac{RT_c}{P_c}

    .. math::
        \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2

    .. math::
        \kappa = \kappa_0 + [\kappa_1 + \kappa_2(\kappa_3 - T_r)(1-T_r^{0.5})]
        (1 + T_r^{0.5})(0.7 - T_r)

    .. math::
        \kappa_0 = 0.378893 + 1.4897153\omega - 0.17131848\omega^2
        + 0.0196554\omega^3

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    kappa1 : float, optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]
    kappa2 : float, optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]
    kappa : float, optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]

    Examples
    --------
    P-T initialization (hexane, with fit parameter in [1]_), liquid phase:

    >>> eos = PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104, kappa2=0.8634, kappa3=0.460)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.000130188257591, -31496.1841687, -73.615282963)

    Notes
    -----
    Note that tabulated `kappa` values should be used with the critical
    parameters used in their fits. [1]_ considered only vapor
    pressure in fitting the parameter.

    References
    ----------
    .. [1] Stryjek, R., and J. H. Vera. "PRSV2: A Cubic Equation of State for
       Accurate Vapor-liquid Equilibria Calculations." The Canadian Journal of
       Chemical Engineering 64, no. 5 (October 1, 1986): 820-26.
       doi:10.1002/cjce.5450640516.
    '''

    kwargs_keys = ('kappa1', 'kappa2', 'kappa3')
    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None, kappa1=0, kappa2=0, kappa3=0):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V
        self.check_sufficient_inputs()
        self.kwargs = {'kappa1': kappa1, 'kappa2': kappa2, 'kappa3': kappa3}

        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = 2*self.b
        self.epsilon = -self.b*self.b
        self.kappa0 = kappa0 = omega*(omega*(0.0196554*omega - 0.17131848) + 1.4897153) + 0.378893
        self.kappa1, self.kappa2, self.kappa3 = kappa1, kappa2, kappa3

        if self.V and self.P:
            # Deal with T-solution here
            self.T = T = self.solve_T(self.P, self.V)

        Tr = T/Tc
        sqrtTr = sqrt(Tr)
        self.kappa = kappa0 + ((kappa1 + kappa2*(kappa3 - Tr)*(1.0 -sqrtTr))*(1.0 + sqrtTr)*(0.7 - Tr))

        self.solve()

    def solve_T(self, P, V, solution=None):
        r'''Method to calculate `T` from a specified `P` and `V` for the PRSV2
        EOS. Uses `Tc`, `a`, `b`, `kappa0`, `kappa1`, `kappa2`, and `kappa3`
        as well, obtained from the class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        solution : str or None, optional
            'l' or 'g' to specify a liquid of vapor solution (if one exists);
            if None, will select a solution more likely to be real (closer to
            STP, attempting to avoid temperatures like 60000 K or 0.0001 K).

        Returns
        -------
        T : float
            Temperature, [K]

        Notes
        -----
        Not guaranteed to produce a solution. There are actually 8 solutions,
        six with an imaginary component at a tested point. The two temperature
        solutions are quite far apart, with one much higher than the other;
        it is possible the solver could converge on the higher solution, so use
        `T` inputs with care. This extra solution is a perfectly valid one
        however. The secant method is implemented at present.

        Examples
        --------
        >>> eos = PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, T=400., P=1E6, kappa1=0.05104, kappa2=0.8634, kappa3=0.460)
        >>> eos.solve_T(P=eos.P, V=eos.V_g)
        400.0
        '''
        self.no_T_spec = True
        if solution is None:
            Tc, a, b, kappa0, kappa1, kappa2, kappa3 = self.Tc, self.a, self.b, self.kappa0, self.kappa1, self.kappa2, self.kappa3
            x0 = V - b
            R_x0 = R/x0
            x5_inv = 1.0/(100.*(V*(V + b) + b*x0))
            x4 = 10.*kappa0
            def to_solve(T):
                x1 = T/Tc
                x2 = sqrt(x1)
                x3 = x2 - 1.
                x100 = (x3*(x4 - (kappa1 + kappa2*x3*(-kappa3 + x1))*(10.*x1 - 7.)*(x2 + 1.)) - 10.)
                return (R_x0*T - a*x100*x100*x5_inv) - P
            try:
                return secant(to_solve, Tc*0.5)
            except:
                pass
            # The above method handles fewer cases, but the below is less optimized
        return GCEOS.solve_T(self, P, V, solution=solution)


    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate :math:`a \alpha` and its first and second
        derivatives for this EOS. Uses the set values of `Tc`, `kappa0`, `kappa1`,
        `kappa2`, `kappa3`, and `a`.

        .. math::
            \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2

        .. math::
            \kappa = \kappa_0 + [\kappa_1 + \kappa_2(\kappa_3 - T_r)(1-T_r^{0.5})]
            (1 + T_r^{0.5})(0.7 - T_r)


        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]

        Notes
        -----
        The first and second derivatives of `a_alpha` are available through the
        following SymPy expression.

        >>> from sympy import *  # doctest:+SKIP
        >>> P, T, V = symbols('P, T, V')  # doctest:+SKIP
        >>> Tc, Pc, omega = symbols('Tc, Pc, omega')  # doctest:+SKIP
        >>> R, a, b, kappa0, kappa1, kappa2, kappa3 = symbols('R, a, b, kappa0, kappa1, kappa2, kappa3')  # doctest:+SKIP
        >>> Tr = T/Tc  # doctest:+SKIP
        >>> kappa = kappa0 + (kappa1 + kappa2*(kappa3-Tr)*(1-sqrt(Tr)))*(1+sqrt(Tr))*(Rational('0.7')-Tr)  # doctest:+SKIP
        >>> a_alpha = a*(1 + kappa*(1-sqrt(T/Tc)))**2  # doctest:+SKIP
        >>> diff(a_alpha, T)  # doctest:+SKIP
        >>> diff(a_alpha, T, 2)  # doctest:+SKIP

        Examples
        --------
        >>> eos = PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, T=400., P=1E6, kappa1=0.05104, kappa2=0.8634, kappa3=0.460)
        >>> eos.a_alpha_and_derivatives_pure(311.0)
        (3.7245418495, -0.0066115440470, 2.05871011677e-05)
        '''
        Tc, a, kappa0, kappa1, kappa2, kappa3 = self.Tc, self.a, self.kappa0, self.kappa1, self.kappa2, self.kappa3
        Tc_inv = 1.0/Tc
        T_inv = 1.0/T
        x1 = T*Tc_inv
        x2 = sqrt(x1)
        x3 = x2 - 1.
        x4 = x2 + 1.
        x5 = 10.*x1 - 7.
        x6 = -kappa3 + x1
        x7 = kappa1 + kappa2*x3*x6
        x8 = x5*x7
        x9 = 10.*kappa0 - x4*x8
        x10 = x3*x9
        x11 = x10*0.1 - 1.
        x13 = x2*T_inv
        x14 = x7*Tc_inv
        x15 = kappa2*x4*x5
        x16 = 2.*(-x2 + 1.)*Tc_inv + x13*(kappa3 - x1)
        x17 = -x13*x8 - x14*(20.*x2 + 20.) + x15*x16
        x18 = x13*x9 + x17*x3
        x19 = x2*T_inv*T_inv
        x20 = 2.*x2*T_inv

        a_alpha = a*x11*x11
        da_alpha_dT = a*x11*x18*0.1
        d2a_alpha_dT2 = a*(x18*x18 + (x10 - 10.)*(x17*x20 - x19*x9
                        + x3*(40.*kappa2*Tc_inv*x16*x4 + kappa2*x16*x20*x5
                        - 40.*T_inv*x14*x2 - x15*T_inv*x2*(4.*Tc_inv - x6*T_inv) + x19*x8)))*(1.0/200.)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        r'''Method to calculate :math:`a \alpha` for this EOS. Uses the set values of `Tc`, `kappa0`, `kappa1`,
        `kappa2`, `kappa3`, and `a`.

        .. math::
            \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2

        .. math::
            \kappa = \kappa_0 + [\kappa_1 + \kappa_2(\kappa_3 - T_r)(1-T_r^{0.5})]
            (1 + T_r^{0.5})(0.7 - T_r)

        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

        Examples
        --------
        >>> eos = PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, T=400., P=1E6, kappa1=0.05104, kappa2=0.8634, kappa3=0.460)
        >>> eos.a_alpha_pure(1276.0)
        33.321674050
        '''
        Tc, a, kappa0, kappa1, kappa2, kappa3 = self.Tc, self.a, self.kappa0, self.kappa1, self.kappa2, self.kappa3
        Tr = T/Tc
        sqrtTr = sqrt(Tr)
        kappa = kappa0 + ((kappa1 + kappa2*(kappa3 - Tr)*(1.0 -sqrtTr))*(1.0 + sqrtTr)*(0.7 - Tr))
        x0 = (1.0 + kappa*(1.0 - sqrtTr))
        return a*x0*x0

class VDW(GCEOS):
    r'''Class for solving the Van der Waals [1]_ [2]_ cubic
    equation of state for a pure compound. Subclasses :obj:`GCEOS`, which
    provides the methods for solving the EOS and calculating its assorted
    relevant thermodynamic properties. Solves the EOS on initialization.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P=\frac{RT}{V-b}-\frac{a}{V^2}

    .. math::
        a=\frac{27}{64}\frac{(RT_c)^2}{P_c}

    .. math::
        b=\frac{RT_c}{8P_c}

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    omega : float, optional
        Acentric factor - not used in equation of state!, [-]

    Examples
    --------
    >>> eos = VDW(Tc=507.6, Pc=3025000, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.000223329856081, -13385.7273746, -32.65923125)

    Notes
    -----
    `omega` is allowed as an input for compatibility with the other EOS forms,
    but is not used.

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th
       edition. New York: McGraw-Hill Professional, 2000.
    .. [2] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
       Butterworth-Heinemann, 1985.
    '''

    delta = 0.0
    """`delta` is always zero for the :obj:`VDW` EOS"""
    epsilon = 0.0
    """`epsilon` is always zero for the :obj:`VDW` EOS"""
    omega = None
    """`omega` has no impact on the :obj:`VDW` EOS"""
    Zc = 3.0/8.
    """Mechanical compressibility of :obj:`VDW` EOS"""

    c1 = 27.0/64.0
    c2 = 1.0/8.0

    c1R2 = c1*R2
    c2R = c2*R
    c1R2_c2R = c1R2/c2R

    Psat_coeffs_limiting = [-3.0232164484175756, 0.20980668241160666]

    Psat_coeffs_critical = [9.575399398167086, -5.742004486758378,
                            4.8000085098196745, -3.000000002903554,
                            1.0000000000002651]

    Psat_cheb_coeffs = [-3.0938407448693392, -3.095844800654779, -0.01852425171597184, -0.009132810281704463,
                        0.0034478548769173167, -0.0007513250489879469, 0.0001425235859202672, -3.18455900032599e-05,
                        8.318773833859442e-06, -2.125810773856036e-06, 5.171012493290658e-07, -1.2777009201877978e-07,
                        3.285945705657834e-08, -8.532047244427343e-09, 2.196978792832582e-09, -5.667409821199761e-10,
                        1.4779624173003134e-10, -3.878590467732996e-11, 1.0181633097391951e-11, -2.67662653922595e-12,
                        7.053635426397184e-13, -1.872821965868618e-13, 4.9443291800198297e-14, -1.2936198878592264e-14,
                        2.9072203628840998e-15, -4.935694864968698e-16, 2.4160767787481663e-15, 8.615748088927622e-16,
                        -5.198342841253312e-16, -2.19739320055784e-15, -1.0876309618559898e-15, 7.727786509661994e-16,
                        7.958450521858285e-16, 2.088444434750203e-17, -1.3864912907016191e-16]
    Psat_cheb_constant_factor = (-1.0630005005005003, 0.9416200294550813)
    Psat_cheb_coeffs_der = chebder(Psat_cheb_coeffs)
    Psat_coeffs_critical_der = polyder(Psat_coeffs_critical[::-1])[::-1]

    # old
#    Psat_ranges_low = (0.01718036869384043, 0.17410832232527684, 0.8905407354904298, 2.2829574334301284, 4.426814725758547, 8.56993985840095, 17.61166533755772, 34.56469477688733, 77.37121667459365, 153.79268928425782, 211.1818863599383)
#    Psat_coeffs_low = [[7.337186734125958e+20, -1.0310276898768364e+20, 6.571811243740342e+18, -2.5142563688011328e+17, 6438369447367660.0, -116506416837877.06, 1533164370692.7314, -14873856417.118628, 106700642.42147608, -562656.3974485798, 2148.713955632803, -5.548751336002541, -0.32195156938958336, 0.29998758814658794, -2.9999999917461664, -2.349558048120315e-12], [-47716.736350665735, 77971.24589571664, -57734.17539993985, 25680.2304910714, -7665.471399138683, 1624.1156139380123, -251.90520898330854, 29.14383349387213, -2.6381384873389155, 0.32059956364530534, -0.19926642267720887, 0.24817720022049916, -0.33257749010194143, 0.3000000932130581, -3.000000000833664, 3.2607250233240848e-12], [-0.00016633444687699065, 0.0015485952911635057, -0.006863235318198338, 0.019451003485722398, -0.04015910528306605, 0.06568824742274948, -0.09110903840180812, 0.11396786842400031, -0.13566008631414134, 0.15955575257376992, -0.19156365559810004, 0.2479007648557237, -0.33256946631897144, 0.29999987151721086, -2.9999999954189773, -5.750377951585506e-11], [-7.428009121019926e-08, 1.9162063229590812e-06, -2.309905985537932e-05, 0.00017322131856139868, -0.0009091957309876683, 0.003571436963791831, -0.010987705972491674, 0.027369774602560334, -0.05652165868271822, 0.0989022663430445, -0.15341525362487263, 0.22884492561559433, -0.325345424979866, 0.2980576936663475, -2.999671413421381, -2.6222517302443293e-05], [-1.304829383752042e-10, 6.461311349903354e-09, -1.4697523398500654e-07, 2.0279063155086473e-06, -1.8858104539128372e-05, 0.0001240696514816671, -0.0005894769787540043, 0.0020347845952011574, -0.0051863170297969646, 0.010970032738000915, -0.02673039729181951, 0.07989719021025658, -0.18981116875502171, 0.20975049244314053, -2.9633925585336254, -0.007040461060364933], [-8.483418232487059e-14, 8.63974862373075e-12, -4.1017440987789216e-10, 1.2043221191702449e-08, -2.4458795830937423e-07, 3.6395727498191334e-06, -4.098703864306506e-05, 0.00035554065503799216, -0.0023925268788656615, 0.012458721418882563, -0.04951152909450557, 0.14533920657540916, -0.29136837208180943, 0.2957408216961629, -2.992226481317782, -0.010497873866540886], [1.2057846439922303e-18, -2.498037369038981e-16, 2.4169298089982375e-14, -1.4499679111714204e-12, 6.038719260446175e-11, -1.8521190625043091e-09, 4.3303351110208387e-08, -7.880798028324914e-07, 1.130024120583253e-05, -0.00012841965962337936, 0.0011579326786125476, -0.008265573217317893, 0.04660472417726327, -0.20986415552448967, -2.53897638918461, -0.1886467797596083], [5.456456120655508e-23, -2.2443095704330367e-20, 4.312575509674247e-18, -5.139829986632021e-16, 4.2535953008708246e-14, -2.592784941046291e-12, 1.2047971820045452e-10, -4.356922932676532e-09, 1.2408018207471443e-07, -2.797822690789934e-06, 4.99619390186859e-05, -0.0007038763395268029, 0.007780538926245783, -0.06771345498616513, -2.871932127187338, 0.18812475659373717], [7.867249251522955e-28, -6.9880843771805455e-25, 2.8943351684355176e-22, -7.420582747372289e-20, 1.3183352580324197e-17, -1.7213805273492628e-15, 1.7095155009610707e-13, -1.3180402544717915e-11, 7.98160114045682e-10, -3.815575213074619e-08, 1.4395507736418072e-06, -4.266216879490852e-05, 0.0009859688747049194, -0.01775904876058947, -3.107864564311215, 0.729035082405801], [1.8311839997067204e-31, -3.0736833050041296e-28, 2.3992665338781007e-25, -1.1556016072304783e-22, 3.842081607645237e-20, -9.344614680779128e-18, 1.7187598144123416e-15, -2.436914034605369e-13, 2.6895257030219836e-11, -2.3163949881188256e-09, 1.5507887324406105e-07, -7.988762893413175e-06, 0.0003116704944009503, -0.00906717310261831, -3.1702256862598945, 0.8792501521348868], [3.527506950185549e-29, -9.295645359865239e-26, 1.1416182235481363e-22, -8.667770574363085e-20, 4.550011954816112e-17, -1.7491901974552787e-14, 5.087496940001709e-12, -1.1399408706170077e-09, 1.9839477686722418e-07, -2.6819268483650753e-05, 0.0027930054114231415, -0.2200566323239571, 12.697254920421862, -506.5185701848038, 12488.177914134796, -143563.720494474]]

    Psat_ranges_low = (0.01718036869384043, 0.17410832232527684, 0.8905407354904298, 2.2829574334301284, 4.588735762374165, 9.198213969343113, 19.164801360905702, 39.162202265367675, 89.80614296441635, 211.1818863599383)
    Psat_coeffs_low = [[3.709170427241228e+20, -5.369876399773602e+19, 3.55437646625649e+18, -1.4237055014239443e+17, 3848808938963197.0, -74140204939074.31, 1047156764817.9473, -10990806592.937004, 85953150.78472495, -497625.69096407224, 2099.583641752129, -6.04136142777935, -0.31974206497045565, 0.29998332413453277, -2.9999999877560994, -3.795894154556834e-12], [202181.2230940579, -296048.9678516585, 197754.0893494042, -79779.91915062582, 21690.65516277411, -4199.021355925494, 596.0913050938211, -62.88287531914928, 4.838615877346316, -0.13231119871015717, -0.17907064524060426, 0.24752941358391634, -0.33256309490225905, 0.29999988493075114, -2.9999999990847974, -3.155253835984695e-12], [-0.00014029558609732686, 0.0013389844121899472, -0.0060888948581993155, 0.017710934821923482, -0.0375010580271008, 0.06276682660210708, -0.08872419862950512, 0.11249655170661062, -0.1349689265044028, 0.15930871464919516, -0.19149706848521642, 0.24788748254751106, -0.3325675697571886, 0.2999996886491002, -2.999999984780528, -3.387914393471192e-10], [-7.63500937520021e-08, 1.963108844693943e-06, -2.3590460584275485e-05, 0.0001763784708554673, -0.0009231029234653729, 0.0036159164438775227, -0.011094383271741722, 0.027565091035355725, -0.05679683707503139, 0.09920052620961826, -0.1536618676986315, 0.22899765112464673, -0.32541398139525823, 0.29807874689899555, -2.9996753674171845, -2.5880253545551568e-05], [-8.042041394684212e-11, 3.985029597822566e-09, -9.007790594461407e-08, 1.222381124984873e-06, -1.1000101355126748e-05, 6.812347222100813e-05, -0.00028917548838158866, 0.0007973445988826675, -0.0012398033018025893, 0.0012288503460726383, -0.008274005514143725, 0.05353750365067098, -0.16234102442586626, 0.19003067203938392, -2.9546730846617515, -0.008830685622417178], [-3.611616220031973e-14, 3.8767347748860204e-12, -1.9376800248066728e-10, 5.982204949605256e-09, -1.2756854237545315e-07, 1.9899440138492874e-06, -2.344690467724149e-05, 0.00021230626049088423, -0.0014868552467794268, 0.008024812789608007, -0.03284208875861816, 0.0980794205896591, -0.19356262181013717, 0.15625529853158554, -2.8696503839999488, -0.06053293499127932], [3.9101454054983654e-19, -8.77263958995087e-17, 9.190024742411764e-15, -5.968146499787873e-13, 2.6900134213207918e-11, -8.926852543915742e-10, 2.2576199533792054e-08, -4.44287432728095e-07, 6.886340930232061e-06, -8.455663245991894e-05, 0.0008233162588686339, -0.006341223591956234, 0.038529175147467405, -0.1865188282734164, -2.580546988555211, -0.15427433578447847], [1.0947478032468857e-23, -5.042777995894199e-21, 1.0846467451595606e-18, -1.4462299284667368e-16, 1.3382745001929549e-14, -9.116107354161303e-13, 4.730996512803351e-11, -1.9095977742126322e-09, 6.06591823816572e-08, -1.524502829215733e-06, 3.0318007813133075e-05, -0.00047520009323191216, 0.005836174792523882, -0.056313989030004126, -2.913137841802409, 0.2573512083085632], [1.0417003605891542e-28, -1.0617427986062235e-25, 5.045404400073696e-23, -1.4839108957248214e-20, 3.023754129891556e-18, -4.527590970541259e-16, 5.155153568793495e-14, -4.555877665139844e-12, 3.161476174754809e-10, -1.731323722560287e-08, 7.479908191797176e-07, -2.537158469008294e-05, 0.0006706500253454564, -0.013799582450476145, -3.1384761956985416, 0.8388793704120587], [3.619666517093681e-34, -8.603334584486169e-31, 9.530335406766774e-28, -6.531547415539122e-25, 3.100047792152191e-22, -1.0806960083892972e-19, 2.863332612760165e-17, -5.885004890375537e-15, 9.491165245604907e-13, -1.207013136689613e-10, 1.2097185039766346e-08, -9.505275874736017e-07, 5.807225072909577e-05, -0.002750509744231165, -3.2675197703421506, 1.578136241005211]]


    phi_sat_coeffs = [-4.703247660146169e-06, 7.276853488756492e-05, -0.0005008397610615123,
                      0.0019560274384829595, -0.004249875101260566, 0.001839985687730564,
                      0.02021191780955066, -0.07056928933569773, 0.09941120467466309,
                      0.021295687530901747, -0.32582447905247514, 0.521321793740683,
                      0.6950957738017804]

    _P_zero_l_cheb_coeffs = [0.23949680596158576, -0.28552048884377407, 0.17223773827357045, -0.10535895068953466, 0.06539081523178862, -0.04127943642449526, 0.02647106353835149, -0.017260750015435533, 0.011558172064668568, -0.007830624115831804, 0.005422844032253547, -0.00383463423135285, 0.0027718803475398936, -0.0020570084561681613, 0.0015155074622906842, -0.0011495238177958583, 0.000904782154904249, -0.000683347677699564, 0.0005800187592994201, -0.0004529246894177611, 0.00032901743817593566, -0.0002990561659229427, 0.00023524411148843384, -0.00019464055011993858, 0.0001441665975916752, -0.00013106835607900116, 9.72812311007959e-05, -7.611327134024459e-05, 5.240433315348986e-05, -3.6415012576658176e-05, 3.89310794418167e-05, -2.2160354688301534e-05, 2.7908599229672926e-05, 1.6405692108915904e-05, -1.3931165551671343e-06, -4.80770003354232e-06]
    P_zero_l_cheb_limits = (0.002354706203222534, 9.0)
    main_derivatives_and_departures = staticmethod(main_derivatives_and_departures_VDW)
    def __init__(self, Tc, Pc, T=None, P=None, V=None, omega=None):
        self.Tc = Tc
        self.Pc = Pc
        self.T = T
        self.P = P
        self.V = V

        Tc_Pc = Tc/Pc
        self.a = (27.0/64.0)*R2*Tc*Tc_Pc
        self.b = 0.125*R*Tc_Pc
        self.solve()

    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate :math:`a \alpha` and its first and second
        derivatives for this EOS. Uses the set values of `a`.

        .. math::
            a\alpha = a

        .. math::
            \frac{d a\alpha}{dT} = 0

        .. math::
            \frac{d^2 a\alpha}{dT^2} = 0

        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]
        '''
        return self.a, 0.0, 0.0

    def a_alpha_pure(self, T):
        r'''Method to calculate :math:`a \alpha`. Uses the set values of `a`.

        .. math::
            a\alpha = a

        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        '''
        return self.a

    def solve_T(self, P, V, solution=None):
        r'''Method to calculate `T` from a specified `P` and `V` for the :obj:`VDW`
        EOS. Uses `a`, and `b`, obtained from the class's namespace.

        .. math::
            T =  \frac{1}{R V^{2}} \left(P V^{2} \left(V - b\right)
            + V a - a b\right)

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        solution : str or None, optional
            'l' or 'g' to specify a liquid of vapor solution (if one exists);
            if None, will select a solution more likely to be real (closer to
            STP, attempting to avoid temperatures like 60000 K or 0.0001 K).

        Returns
        -------
        T : float
            Temperature, [K]
        '''
        self.no_T_spec = True
        return (P*V**2*(V - self.b) + V*self.a - self.a*self.b)/(R*V**2)

    def T_discriminant_zeros_analytical(self, valid=False):
        r'''Method to calculate the temperatures which zero the discriminant
        function of the :obj:`VDW` eos. This is an analytical cubic function solved
        analytically.

        Parameters
        ----------
        valid : bool
            Whether to filter the calculated temperatures so that they are all
            real, and positive only, [-]

        Returns
        -------
        T_discriminant_zeros : list[float]
            Temperatures which make the discriminant zero, [K]

        Notes
        -----
        Calculated analytically. Derived as follows. Has multiple solutions.

        >>> from sympy import * # doctest:+SKIP
        >>> P, T, V, R, b, a = symbols('P, T, V, R, b, a') # doctest:+SKIP
        >>> delta, epsilon = 0, 0 # doctest:+SKIP
        >>> eta = b # doctest:+SKIP
        >>> B = b*P/(R*T) # doctest:+SKIP
        >>> deltas = delta*P/(R*T) # doctest:+SKIP
        >>> thetas = a*P/(R*T)**2 # doctest:+SKIP
        >>> epsilons = epsilon*(P/(R*T))**2 # doctest:+SKIP
        >>> etas = eta*P/(R*T) # doctest:+SKIP
        >>> a_coeff = 1 # doctest:+SKIP
        >>> b_coeff = (deltas - B - 1) # doctest:+SKIP
        >>> c = (thetas + epsilons - deltas*(B+1)) # doctest:+SKIP
        >>> d = -(epsilons*(B+1) + thetas*etas) # doctest:+SKIP
        >>> disc = b_coeff*b_coeff*c*c - 4*a_coeff*c*c*c - 4*b_coeff*b_coeff*b_coeff*d - 27*a_coeff*a_coeff*d*d + 18*a_coeff*b_coeff*c*d # doctest:+SKIP
        >>> base = -(expand(disc/P**2*R**3*T**3/a)) # doctest:+SKIP
        >>> base_T = simplify(base*T**3) # doctest:+SKIP
        >>> sln = collect(expand(base_T), T).args # doctest:+SKIP
        '''
        P, a, b = self.P, self.a_alpha, self.b

        b2 = b*b
        x1 = P*b2
        x0 = 12.0*x1

        d = 4.0*P*R_inv2*R_inv*(a*a + x1*(x1 +  2.0*a))
        c = (x0 - 20.0*a)*R_inv2*b*P
        b_coeff = (x0 - a)*R_inv
        a_coeff = 4.0*b

        roots = roots_cubic(a_coeff, b_coeff, c, d)
#        roots = np.roots([a_coeff, b_coeff, c, d]).tolist()
        if valid:
            # TODO - only include ones when switching phases from l/g to either g/l
            # Do not know how to handle
            roots = [r.real for r in roots if (r.real >= 0.0 and (abs(r.imag) <= 1e-12))]
            roots.sort()
        return roots

    @staticmethod
    def P_discriminant_zeros_analytical(T, b, delta, epsilon, a_alpha, valid=False):
        r'''Method to calculate the pressures which zero the discriminant
        function of the :obj:`VDW` eos. This is an cubic function solved
        analytically.

        Parameters
        ----------
        T : float
            Temperature, [K]
        b : float
            Coefficient calculated by EOS-specific method, [m^3/mol]
        delta : float
            Coefficient calculated by EOS-specific method, [m^3/mol]
        epsilon : float
            Coefficient calculated by EOS-specific method, [m^6/mol^2]
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        valid : bool
            Whether to filter the calculated pressures so that they are all
            real, and positive only, [-]

        Returns
        -------
        P_discriminant_zeros : tuple[float]
            Pressures which make the discriminant zero, [Pa]

        Notes
        -----
        Calculated analytically. Derived as follows. Has multiple solutions.

        >>> from sympy import * # doctest:+SKIP
        >>> P, T, V, R, b, a = symbols('P, T, V, R, b, a') # doctest:+SKIP
        >>> P_vdw = R*T/(V-b) - a/(V*V) # doctest:+SKIP
        >>> delta, epsilon = 0, 0 # doctest:+SKIP
        >>> eta = b # doctest:+SKIP
        >>> B = b*P/(R*T) # doctest:+SKIP
        >>> deltas = delta*P/(R*T) # doctest:+SKIP
        >>> thetas = a*P/(R*T)**2 # doctest:+SKIP
        >>> epsilons = epsilon*(P/(R*T))**2 # doctest:+SKIP
        >>> etas = eta*P/(R*T) # doctest:+SKIP
        >>> a_coeff = 1 # doctest:+SKIP
        >>> b_coeff = (deltas - B - 1) # doctest:+SKIP
        >>> c = (thetas + epsilons - deltas*(B+1)) # doctest:+SKIP
        >>> d = -(epsilons*(B+1) + thetas*etas) # doctest:+SKIP
        >>> disc = b_coeff*b_coeff*c*c - 4*a_coeff*c*c*c - 4*b_coeff*b_coeff*b_coeff*d - 27*a_coeff*a_coeff*d*d + 18*a_coeff*b_coeff*c*d # doctest:+SKIP
        >>> base = -(expand(disc/P**2*R**3*T**3/a)) # doctest:+SKIP
        >>> collect(base, P).args # doctest:+SKIP
        '''
#        T, a_alpha = self.T, self.a_alpha
#        a = a_alpha
#        b, epsilon, delta = self.b, self.epsilon, self.delta
        a = a_alpha
        d = 4*b - a/(R*T)
        c = (12*b**2/(R*T) - 20*a*b/(R**2*T**2) + 4*a**2/(R**3*T**3))
        b_coeff = (12*b**3/(R**2*T**2) + 8*a*b**2/(R**3*T**3))
        a_coeff = 4*b**4/(R**3*T**3)

        roots = roots_cubic(a_coeff, b_coeff, c, d)
#        roots = np.roots([a_coeff, b_coeff, c, d]).tolist()
        return roots

class RK(GCEOS):
    r'''Class for solving the Redlich-Kwong [1]_ [2]_ [3]_ cubic
    equation of state for a pure compound. Subclasses :obj:`GCEOS`, which
    provides the methods for solving the EOS and calculating its assorted
    relevant thermodynamic properties. Solves the EOS on initialization.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P =\frac{RT}{V-b}-\frac{a}{V\sqrt{\frac{T}{T_{c}}}(V+b)}

    .. math::
        a=\left(\frac{R^2(T_c)^{2}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2.5}}{P_c}

    .. math::
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    >>> eos = RK(Tc=507.6, Pc=3025000, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.000151893468781, -26160.8424877, -63.013137852)

    Notes
    -----
    `omega` is allowed as an input for compatibility with the other EOS forms,
    but is not used.

    References
    ----------
    .. [1] Redlich, Otto., and J. N. S. Kwong. "On the Thermodynamics of
       Solutions. V. An Equation of State. Fugacities of Gaseous Solutions."
       Chemical Reviews 44, no. 1 (February 1, 1949): 233-44.
       doi:10.1021/cr60137a013.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th
       edition. New York: McGraw-Hill Professional, 2000.
    .. [3] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
       Butterworth-Heinemann, 1985.
    '''

    c1 = 0.4274802335403414043909906940611707345513 # 1/(9*(2**(1/3.)-1))
    """Full value of the constant in the `a` parameter"""
    c2 = 0.08664034996495772158907020242607611685675 # (2**(1/3.)-1)/3
    """Full value of the constant in the `b` parameter"""

    epsilon = 0.0
    """`epsilon` is always zero for the :obj:`RK` EOS"""
    omega = None
    """`omega` has no impact on the :obj:`RK` EOS"""
    Zc = 1.0/3.
    """Mechanical compressibility of :obj:`RK` EOS"""

    c1R2 = c1*R2
    c2R = c2*R
    c1R2_c2R = c1R2/c2R

    Psat_coeffs_limiting = [-72.700288369511583, -68.76714163049]
    Psat_coeffs_critical = [1129250.3276866912, 4246321.053155941,
                            5988691.4873851035, 3754317.4112657467,
                            882716.2189281426]

    Psat_cheb_coeffs = [-6.8488798834192215, -6.93992806360099, -0.11216113842675507, 0.0022494496508455135,
                        0.00995148012561513, -0.005789786392208277, 0.0021454644555051177, -0.0006192510387981658,
                        0.00016870584348326536, -5.828094356536212e-05, 2.5829410448955883e-05, -1.1312372380559225e-05,
                        4.374040785359406e-06, -1.5546789700246184e-06, 5.666723613325655e-07, -2.2701147218271074e-07,
                        9.561199996134724e-08, -3.934646467524511e-08, 1.55272396700466e-08, -6.061097474369418e-09,
                        2.4289648176102022e-09, -1.0031987621530753e-09, 4.168016003137324e-10, -1.7100917451312765e-10,
                        6.949731049432813e-11, -2.8377758503521713e-11, 1.1741734564892428e-11, -4.891469634936765e-12,
                        2.0373765879672795e-12, -8.507821454718095e-13, 3.4975627537410514e-13, -1.4468659018281038e-13,
                        6.536766028637786e-14, -2.7636123641275323e-14, 1.105377996166862e-14]
    Psat_cheb_constant_factor = (0.8551757791729341, 9.962912449541513)

    Psat_cheb_coeffs_der = chebder(Psat_cheb_coeffs)
    Psat_coeffs_critical_der = polyder(Psat_coeffs_critical[::-1])[::-1]

    phi_sat_coeffs = [156707085.9178746, 1313005585.0874271, 4947242291.244957,
                      11038959845.808495, 16153986262.1129, 16199294577.496677,
                      11273931409.81048, 5376831929.990161, 1681814895.2875218,
                      311544335.80653775, 25954329.68176187]

    Psat_ranges_low = (0.033797068457719265, 0.06786604965443845, 0.1712297613585108, 0.34622987689428786, 0.7712336381743264, 1.745621379817678, 3.7256294306343207, 6.581228646986647, 12.781884795234102, 24.412307224840184, 48.39951213433041, 99.16043966361465, 206.52538850089107)
    Psat_coeffs_low = [[-2.0473027805583304e+16, 5590407450630671.0, -691651500345804.2, 51280971870837.32, -2539543204646.707, 88630534161.11136, -2241691916.726171, 41616814.12629884, -568152.2176538995, 5661.177783316078, -40.73060128671707, 0.5116910477178825, -0.3837083115168163, 0.33323887045969375, -3.0536215774324824, 7.04644675941779e-13], [308999097192961.8, -225585388170583.75, 76474322668841.14, -15968075011367.047, 2296463426010.7324, -240935052543.90527, 19048380996.66752, -1155476440.462194, 54215636.4938359, -1967442.1357566533, 54762.443278119186, -1147.7177667787553, 17.16085684519316, 0.14875852741749432, -3.052428192332575, -3.5796888761263634e-06], [-19650136.181735344, 31781193.928728923, -23482166.19636667, 10469551.38856498, -3130203.8712914013, 658159.0021417511, -98797.59681465346, 10408.710407311624, -708.5507506253232, 20.511506773041855, 1.1894690300458857, 0.09491188992340026, -0.3699746434334233, 0.3327794679246402, -3.053612527869441, -7.854416195218761e-08], [27999.20768241787, -105403.37695226824, 184231.72791343703, -198316.18594155577, 147020.31646897402, -79504.54402254449, 32396.282402624805, -10128.009032305941, 2449.0150598681134, -457.86581824588563, 65.46517369332473, -6.79444639388651, 0.17688177142370798, 0.30286151881162965, -3.052607289294409, -1.571427065993891e-05], [0.03260876969558781, -0.2687907725537127, 1.0267017754905683, -2.4094567772527298, 3.8817887528347166, -4.539672024261187, 3.9650465848385767, -2.6039495096751812, 1.2459350220734762, -0.3527513106804435, -0.07989160047420704, 0.2668158546701413, -0.37571189865522364, 0.33238476075275014, -3.053560614808337, -2.0178433899342707e-06], [-7.375386804046186e-07, 1.5495867993995408e-05, -0.00015284972568903337, 0.0009416918743616606, -0.0040706288679345, 0.013170886849436726, -0.033323844522302436, 0.0682465756200053, -0.11659032663823216, 0.171245314502395, -0.22682939669727753, 0.29437853298508776, -0.3782483397093579, 0.33220274751099305, -3.053481229409229, -8.962708854642898e-06], [-9.64878085834743e-10, 4.3676512898669364e-08, -9.195173654927243e-07, 1.192726889996796e-05, -0.00010641091957706412, 0.0006901346742808183, -0.0033536294555478845, 0.012421045099127406, -0.03550565516993044, 0.07992816727224494, -0.14853460918439382, 0.24510479474467387, -0.35702531869477633, 0.32684341660780225, -3.053040405250035, 6.241135226403571e-05], [-3.5002452124187664e-13, 3.2833906636665076e-11, -1.430637752387875e-09, 3.84703168582193e-08, -7.149328354678022e-07, 9.737245185617094e-06, -0.0001004961143222046, 0.0008008011389293154, -0.004967550562579805, 0.023964447789498418, -0.08887464138262652, 0.24644247973322814, -0.4795516986170165, 0.5340043615968529, -3.218003440328873, 0.0551398944675352], [5.034255484339691e-17, -7.864397963490581e-15, 5.752225082446006e-13, -2.615885881066589e-11, 8.282570031686524e-10, -1.9374016535411608e-08, 3.4664702879779335e-07, -4.845857364906252e-06, 5.359278262994369e-05, -0.00047191509173119853, 0.003314508017194061, -0.018546171084714562, 0.08264133469800018, -0.2976511343203724, -2.45051660976546, -0.2780886842259136], [6.989268516115097e-21, -2.0569549793133175e-18, 2.829181210013469e-16, -2.4145085922204994e-14, 1.4314693471489465e-12, -6.253869824774858e-11, 2.0839676060467857e-09, -5.407898264555903e-08, 1.10600368057607e-06, -1.7926789523973723e-05, 0.00023042013161443632, -0.0023410716986771423, 0.018721566225139197, -0.11858986125950052, -2.7694948645429545, -0.005601354706335826], [4.123334769968695e-25, -2.3690486736737077e-22, 6.357338194540137e-20, -1.0578385702299804e-17, 1.2218888207640753e-15, -1.0392124936479462e-13, 6.735204427884249e-12, -3.395649622455689e-10, 1.3474573409568848e-08, -4.230554509480506e-07, 1.0508993131776807e-05, -0.000205653116778391, 0.0031500689366463458, -0.037812816408928154, -3.0367716832005502, 0.42028748728880316], [1.2637873346500257e-29, -1.4691821304898006e-26, 7.97371063405237e-24, -2.6821456640173466e-21, 6.259659753789673e-19, -1.0750710034717391e-16, 1.4061365634529063e-14, -1.4296683760114274e-12, 1.1431380830397977e-10, -7.224377488638715e-09, 3.607342375851496e-07, -1.4162123664055414e-05, 0.0004338140901526731, -0.010352095429488306, -3.214333239715912, 0.9750918877217316], [2.4467625180409936e-34, -5.900060775579966e-31, 6.640727340409195e-28, -4.631434540342247e-25, 2.240558311318526e-22, -7.974393923385324e-20, 2.1607792871336745e-17, -4.549744513625842e-15, 7.53070203647074e-13, -9.846740932533425e-11, 1.0165520378636594e-08, -8.242924637302648e-07, 5.2066757251344576e-05, -0.002554238879420431, -3.3164175313132174, 1.6226038152294109]]

    # Thought the below was better - is failing. Need better ranges
#    Psat_ranges_low = (0.002864867449609054, 0.00841672469500749, 0.016876032463065772, 0.0338307436429719, 0.06789926110832244, 0.17126106018604287, 0.346311983890616, 0.7714158394657995, 1.7457236753368228, 3.726128003826199, 6.581228646986647, 12.781884795234102, 24.412307224840184, 48.39951213433041, 99.16043966361465, 206.52538850089107)
#    Psat_coeffs_low = [[7.4050168086686e+36, -2.3656234050215595e+35, 3.5140849350729625e+33, -3.219960263668302e+31, 2.0353831678446376e+29, -9.401903615685883e+26, 3.27870193930672e+24, -8.790141973297096e+21, 1.8267653379529515e+19, -2.9430079135183156e+16, 36458127001690.625, -34108081817.37184, 23328876.278645795, -11013.595381204012, 0.15611893268569021, -0.0004353146166747694], [-9.948446571395775e+25, 5.19510926774992e+24, -7.189006307350884e+22, -1.5680848799655576e+21, 8.249811116671015e+19, -1.6950901342526528e+18, 2.1845526238020284e+16, -197095048940187.34, 1299650648245.581, -6369933293.417471, 23232428.90696315, -62270.805040856954, 118.78599182230242, 0.17914187929689024, -3.053500975368179, -4.3120679728281264e-08], [9.397084120968658e+23, -1.7568386783383563e+23, 1.5270752841173062e+22, -8.18635952561484e+20, 3.0268596563705795e+19, -8.176367091459369e+17, 1.6669079239736548e+16, -261159410245086.28, 3170226264128.2866, -29816091359.74791, 215481434.94837964, -1175107.961691127, 4680.2781732486055, -12.521703234052518, -3.031856323244789, -1.7125428611007576e-05], [-5.837633410530431e+19, 2.2354665725528674e+19, -3.9748648336736783e+18, 4.353010078685106e+17, -3.2833626753213436e+16, 1806702598986387.0, -74919223763900.97, 2383883719613.177, -58681349911.8085, 1117424654.4682977, -16325312.825120728, 179698.66647387162, -1442.9221906440623, 8.305809571517658, -3.0807467744908252, 4.282801743092646e-05], [371247743335500.9, -296343269154610.4, 109587709190787.98, -24907918727301.76, 3891709522603.097, -442798937995.7918, 37904104192.54725, -2485814525.910429, 125929932.89424846, -4928093.141384071, 147763.74249085123, -3333.473113107637, 54.40288611988696, -0.28587642030919863, -3.04931950649037, -1.3857488372237547e-05], [16208032.322457436, -28812339.698805887, 23769768.592785712, -12069159.698844183, 4216830.274106061, -1073538.4841311441, 205658.8661178185, -30177.62398799642, 3418.24878855674, -298.57690432008536, 19.71936439114255, -0.6937801749294776, -0.34639354927988253, 0.3323199955834364, -3.053607492200311, -9.988880111944098e-08], [-27604.735354758857, 105025.68737778495, -185570.40811520678, 201981.46688618566, -151443.86118322334, 82852.55333788031, -34164.546515476206, 10812.149811177835, -2647.7278495315263, 501.83786547752425, -73.19000295074032, 8.299579502519556, -1.0215340085992137, 0.3683751980772054, -3.0548122319685604, 1.8727318915723323e-05], [0.005501339854516574, -0.030871868660992247, 0.06262306950800559, -0.016054714694926787, -0.19049678051599286, 0.4916342706149181, -0.6988226755673006, 0.6995056451067202, -0.5569157797560214, 0.40541813786380926, -0.32359936610250595, 0.32562413248075583, -0.38602496065455755, 0.33362571128345786, -3.0536522381393123, 1.1116248239684268e-06], [-7.625288226006888e-07, 1.592958164613657e-05, -0.00015633640254263914, 0.0009589163584223709, -0.004129113991499528, 0.01331550052681755, -0.033592935849933975, 0.06863043023079776, -0.11701377319077316, 0.17160678539192847, -0.22706640260572322, 0.2944958500921784, -0.3782908182523437, 0.3322133797180262, -3.0534828760507775, -8.843634685451462e-06], [-9.602485478694793e-10, 4.349693466931302e-08, -9.162924468728032e-07, 1.1891710154307035e-05, -0.00010614176529656917, 0.0006886536353876484, -0.003347511076288892, 0.012401728251598975, -0.03545868060099854, 0.07984021871367283, -0.1484089324448234, 0.24497026761102456, -0.35692097886460605, 0.32678811005407216, -3.0530225111683147, 5.975140457969985e-05], [-3.271879015155258e-13, 3.106593030234596e-11, -1.3669954450963802e-09, 3.7057309629853086e-08, -6.932939676452128e-07, 9.49513835382941e-06, -9.845166709551053e-05, 0.000787533270198052, -0.0049008339165666475, 0.023704510936127153, -0.08809636022958753, 0.24468388333303692, -0.4766488164995153, 0.5306997327949121, -3.2156835292074972, 0.05438278235662608], [5.027674758216368e-17, -7.856316772563329e-15, 5.74771486417108e-13, -2.6143808092714456e-11, 8.279258549554745e-10, -1.9369063289334646e-08, 3.4659817641020265e-07, -4.8455984691411336e-06, 5.3593276238540865e-05, -0.0004719384663395697, 0.003314740905347161, -0.018547549367299895, 0.08264669293617401, -0.29766465209956755, -2.450496401425025, -0.2781023348227336], [7.050221439220754e-21, -2.0731943309577094e-18, 2.84928020400626e-16, -2.429837955412963e-14, 1.4395267045494148e-12, -6.284786016386046e-11, 2.092913781550848e-09, -5.427778893624791e-08, 1.1094245992741828e-06, -1.7972372175390485e-05, 0.0002308866520395049, -0.002344673517182507, 0.018741874333835694, -0.11866881228864186, -2.7693056026459097, -0.00581227798597439], [4.1192237682040195e-25, -2.366064116476803e-22, 6.347935162866163e-20, -1.0560893898667048e-17, 1.2197114527874372e-15, -1.037276215071595e-13, 6.722432950732127e-12, -3.389265643942945e-10, 1.3450132487487853e-08, -4.2233752380466274e-07, 1.0492923968957467e-05, -0.00020538369710127645, 0.003146790904310898, -0.0377854748223825, -3.036911550333188, 0.4206184380126672], [1.2604142367817428e-29, -1.4652181650108445e-26, 7.952241970757233e-24, -2.6750298800006056e-21, 6.243504642033308e-19, -1.0724081450429274e-16, 1.4028428417591862e-14, -1.4265539327144317e-12, 1.1408674781257439e-10, -7.211610181866389e-09, 3.6018494473345496e-07, -1.4144362093631258e-05, 0.0004333961723540267, -0.010345339188037806, -3.2144003543293014, 0.9754007550013739], [2.4330594298876834e-34, -5.870171131499116e-31, 6.610469013522427e-28, -4.6125772590980585e-25, 2.232467903153395e-22, -7.949083788375843e-20, 2.1548151171435654e-17, -4.538965432777591e-15, 7.515638568810398e-13, -9.83046463909207e-11, 1.0152034186437456e-08, -8.234510061656029e-07, 5.202848938898853e-05, -0.002553041403736697, -3.316440584285066, 1.6228096260313123]]



    def __init__(self, Tc, Pc, T=None, P=None, V=None, omega=None):
        self.Tc = Tc
        self.Pc = Pc
        self.T = T
        self.P = P
        self.V = V
        self.omega = omega

        self.a = self.c1R2*Tc*Tc/Pc
        self.b = self.delta = self.c2R*Tc/Pc
        self.solve()

    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate :math:`a \alpha` and its first and second
        derivatives for this EOS. Uses the set values of `a`.

        .. math::
            a\alpha = \frac{a}{\sqrt{\frac{T}{T_{c}}}}

        .. math::
            \frac{d a\alpha}{dT} = - \frac{a}{2 T\sqrt{\frac{T}{T_{c}}}}

        .. math::
            \frac{d^2 a\alpha}{dT^2} = \frac{3 a}{4 T^{2}\sqrt{\frac{T}{T_{c}}}}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]
        '''
        Tc = self.Tc
        sqrt_Tr_inv = (T/Tc)**-0.5
        a_alpha = self.a*sqrt_Tr_inv
        T_inv = 1.0/T
        da_alpha_dT = -0.5*self.a*T_inv*sqrt_Tr_inv
        d2a_alpha_dT2 = 0.75*self.a*T_inv*T_inv*sqrt_Tr_inv
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        r'''Method to calculate :math:`a \alpha` for this EOS. Uses the set
        values of `a`.

        .. math::
            a\alpha = \frac{a}{\sqrt{\frac{T}{T_{c}}}}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        '''
        Tc = self.Tc
        sqrt_Tr_inv = sqrt(Tc/T)
        return self.a*sqrt_Tr_inv

    def solve_T(self, P, V, solution=None):
        r'''Method to calculate `T` from a specified `P` and `V` for the RK
        EOS. Uses `a`, and `b`, obtained from the class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        solution : str or None, optional
            'l' or 'g' to specify a liquid of vapor solution (if one exists);
            if None, will select a solution more likely to be real (closer to
            STP, attempting to avoid temperatures like 60000 K or 0.0001 K).

        Returns
        -------
        T : float
            Temperature, [K]

        Notes
        -----
        The exact solution can be derived as follows; it is excluded for
        breviety.

        >>> from sympy import *  # doctest:+SKIP
        >>> P, T, V, R = symbols('P, T, V, R')  # doctest:+SKIP
        >>> Tc, Pc = symbols('Tc, Pc')  # doctest:+SKIP
        >>> a, b = symbols('a, b')  # doctest:+SKIP
        >>> RK = Eq(P, R*T/(V-b) - a/sqrt(T)/(V*V + b*V))  # doctest:+SKIP
        >>> solve(RK, T)  # doctest:+SKIP
        '''
        a, b = self.a, self.b
        a = a*self.Tc**0.5
#        print([R, V, b, P, a])
        if solution is None:
            # COnfirmed with mpmath - has numerical issues
            x0 = 3**0.5
            x1 = 1j*x0
            x2 = x1 + 1.0
            x3 = V + b
            x4 = V - b
            x5 = x4/R
            x6 = (x0*(x4**2*(-4*P**3*x5 + 27*a**2/(V**2*x3**2))/R**2+0.0j)**0.5 - 9*a*x5/(V*x3))**0.333333333333333
            x7 = 0.190785707092222*x6
            x8 = P*x5/x6
            x9 =1.7471609294726*x8
            x10 = 1.0 - x1

            slns = [(x2*x7 + x9/x2)**2,
                    (x10*x7 + x9/x10)**2,
                    (0.381571414184444*x6 + 0.873580464736299*x8)**2]
            try:
                self.no_T_spec = True
                x1 = -1.j*1.7320508075688772 + 1.
                x2 = V - b
                x3 = x2/R
                x4 = V + b
                x5 = (1.7320508075688772*(x2*x2*(-4.*P*P*P*x3 + 27.*a*a/(V*V*x4*x4))/(R*R))**0.5 - 9.*a*x3/(V*x4) +0j)**(1./3.)
                T_sln = (3.3019272488946263*(11.537996562459266*P*x3/(x1*x5) + 1.2599210498948732*x1*x5)**2/144.0).real
                if T_sln > 1e-3:
                    return T_sln
            except:
                pass
            # Turns out the above solution does not cover all cases
        return super().solve_T(P, V, solution=solution)


    def T_discriminant_zeros_analytical(self, valid=False):
        r'''Method to calculate the temperatures which zero the discriminant
        function of the `RK` eos. This is an analytical function with an
        11-coefficient polynomial which is solved with `numpy`.

        Parameters
        ----------
        valid : bool
            Whether to filter the calculated temperatures so that they are all
            real, and positive only, [-]

        Returns
        -------
        T_discriminant_zeros : float
            Temperatures which make the discriminant zero, [K]

        Notes
        -----
        Calculated analytically. Derived as follows. Has multiple solutions.

        >>> from sympy import *  # doctest:+SKIP
        >>> P, T, V, R, b, a, Troot = symbols('P, T, V, R, b, a, Troot') # doctest:+SKIP
        >>> a_alpha = a/sqrt(T) # doctest:+SKIP
        >>> delta, epsilon = b, 0 # doctest:+SKIP
        >>> eta = b # doctest:+SKIP
        >>> B = b*P/(R*T) # doctest:+SKIP
        >>> deltas = delta*P/(R*T) # doctest:+SKIP
        >>> thetas = a_alpha*P/(R*T)**2 # doctest:+SKIP
        >>> epsilons = epsilon*(P/(R*T))**2 # doctest:+SKIP
        >>> etas = eta*P/(R*T) # doctest:+SKIP
        >>> a_coeff = 1 # doctest:+SKIP
        >>> b_coeff = (deltas - B - 1) # doctest:+SKIP
        >>> c = (thetas + epsilons - deltas*(B+1)) # doctest:+SKIP
        >>> d = -(epsilons*(B+1) + thetas*etas) # doctest:+SKIP
        >>> disc = b_coeff*b_coeff*c*c - 4*a_coeff*c*c*c - 4*b_coeff*b_coeff*b_coeff*d - 27*a_coeff*a_coeff*d*d + 18*a_coeff*b_coeff*c*d # doctest:+SKIP
        >>> new_disc = disc.subs(sqrt(T), Troot) # doctest:+SKIP
        >>> new_T_base = expand(expand(new_disc)*Troot**15) # doctest:+SKIP
        >>> ans = collect(new_T_base, Troot).args # doctest:+SKIP
        '''
        P, a, b, epsilon, delta = self.P, self.a, self.b, self.epsilon, self.delta

        a *= self.Tc**0.5 # pre-dates change in alpha definition

        P2 = P*P
        P3 = P2*P
        P4 = P2*P2
        R_inv4 = R_inv2*R_inv2
        R_inv5 = R_inv4*R_inv
        R_inv6 = R_inv4*R_inv2
        b2 = b*b
        b3 = b2*b
        b4 = b2*b2
        a2 = a*a
        x5 = 15.0*a2
        x8 = 2.0*P3
        x9 = b4*b*R_inv
        x13 = 6.0*R_inv*R_inv2

        coeffs = [P2*b2*R_inv2,
                  0.0,
                  P3*b3*x13,
                  -a*b*P2*x13,
                  13.0*R_inv4*P4*b4,
                  -32.0*a*P3*R_inv4*b2,
                  P2*R_inv4*(12.0*P3*x9 + a2),
                  -42.0*a*b3*P4*R_inv5,
                  b*R_inv5*x8*(x5 + x8*x9),
                  -12.0*P2*P3*a*R_inv6*b4,
                  -R_inv6*b2*P4*x5,
                  -4.0*a2*a*P3*R_inv6]

        roots = np.roots(coeffs).tolist()
        roots = [i*i for i in roots]
        if valid:
            # TODO - only include ones when switching phases from l/g to either g/l
            # Do not know how to handle
            roots = [r.real for r in roots if (r.real >= 0.0 and (abs(r.imag) <= 1e-12))]
            roots.sort()
        return roots

class SRK(GCEOS):
    r'''Class for solving the Soave-Redlich-Kwong [1]_ [2]_ [3]_ cubic
    equation of state for a pure compound. Subclasses :obj:`GCEOS`, which
    provides the methods for solving the EOS and calculating its assorted
    relevant thermodynamic properties. Solves the EOS on initialization.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}

    .. math::
        a=\left(\frac{R^2(T_c)^{2}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2}}{P_c}

    .. math::
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}

    .. math::
        \alpha(T) = \left[1 + m\left(1 - \sqrt{\frac{T}{T_c}}\right)\right]^2

    .. math::
        m = 0.480 + 1.574\omega - 0.176\omega^2

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    >>> eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.000146821077354, -31754.663859, -74.373272044)

    References
    ----------
    .. [1] Soave, Giorgio. "Equilibrium Constants from a Modified Redlich-Kwong
       Equation of State." Chemical Engineering Science 27, no. 6 (June 1972):
       1197-1203. doi:10.1016/0009-2509(72)80096-4.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th
       edition. New York: McGraw-Hill Professional, 2000.
    .. [3] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
       Butterworth-Heinemann, 1985.
    '''

    c1 = 0.4274802335403414043909906940611707345513 # 1/(9*(2**(1/3.)-1))
    """Full value of the constant in the `a` parameter"""


    c2 = 0.08664034996495772158907020242607611685675 # (2**(1/3.)-1)/3
    """Full value of the constant in the `b` parameter"""

    c1R2 = c1*R2
    c2R = c2*R
    c1R2_c2R = c1R2/c2R

    epsilon = 0.0
    """`epsilon` is always zero for the :obj:`SRK` EOS"""
    Zc = 1/3.
    """Mechanical compressibility of :obj:`SRK` EOS"""

    Psat_coeffs_limiting = [-3.2308843103522107, 0.7210534170705403]

    Psat_coeffs_critical = [9.374273428735918, -6.15924292062784,
                            4.995561268009732, -3.0536215892966374,
                            1.0000000000023588]

    Psat_cheb_coeffs = [-7.871741490227961, -7.989748461289071, -0.1356344797770207, 0.009506579247579184,
                        0.009624489219138763, -0.007066708482598217, 0.003075503887853841, -0.001012177935988426,
                        0.00028619693856193646, -8.960150789432905e-05, 3.8678642545223406e-05, -1.903594210476056e-05,
                        8.531492278109217e-06, -3.345456890803595e-06, 1.2311165149343946e-06, -4.784033464026011e-07,
                        2.0716513992539553e-07, -9.365210448247373e-08, 4.088078067054522e-08, -1.6950725229317957e-08,
                        6.9147476960875615e-09, -2.9036036947212296e-09, 1.2683728020787197e-09, -5.610046772833513e-10,
                        2.444858416194781e-10, -1.0465240317131946e-10, 4.472305869824417e-11, -1.9380782026977295e-11,
                        8.525075935982007e-12, -3.770209730351304e-12, 1.6512636527230007e-12, -7.22057288092548e-13,
                        3.2921267708457824e-13, -1.616661448808343e-13, 6.227456701354828e-14]
    Psat_cheb_constant_factor = (-2.5857326352412238, 0.38702722494279784)
    Psat_cheb_coeffs_der = chebder(Psat_cheb_coeffs)
    Psat_coeffs_critical_der = polyder(Psat_coeffs_critical[::-1])[::-1]

    phi_sat_coeffs = [4.883976406433718e-10, -2.00532968010467e-08, 3.647765457046907e-07,
                      -3.794073186960753e-06, 2.358762477641146e-05, -7.18419726211543e-05,
                      -0.00013493130050539593, 0.002716443506003684, -0.015404883730347763,
                      0.05251643616017714, -0.11346125895127993, 0.12885073074459652,
                      0.0403144920149403, -0.39801902918654086, 0.5962308106352003,
                      0.6656153310272716]

    _P_zero_l_cheb_coeffs = [0.08380676900731782, -0.14019219743961803, 0.11742103327156811, -0.09849160801348428, 0.08273868596563422, -0.0696144897386927, 0.05866765693877264, -0.04952599518184439, 0.04188237509387957, -0.03548315864697149, 0.03011872010893725, -0.02561566850666065, 0.021830462208254395, -0.018644172238802145, 0.015958169671823057, -0.013690592984703707, 0.011773427351986342, -0.01015011684404267, 0.00877358083868034, -0.007604596012758029, 0.006610446573984768, -0.005763823407070205, 0.005041905306198975, -0.004425605918781876, 0.003898948480582476, -0.003448548272342346, 0.0030631866753461218, -0.002733454718851159, 0.0024514621141303247, -0.0022105921907339815, 0.002005302198095145, -0.0018309561248985515, 0.0016836870172771135, -0.0015602844635190134, 0.0014581002673540663, -0.0013749738886825284, 0.0013091699610779176, -0.001259330218826276, 0.001224435336044407, -0.0012037764696538264, 0.0005984681105455358]
    P_zero_l_cheb_limits = (0.0009838646849082977, 77.36362033836788)

    _P_zero_g_cheb_coeffs = [4074.379698522392, 4074.0787931079158, -0.011974050537509407, 0.011278738948946121, -0.010623695898806596, 0.010006612855718989, -0.00942531345107397, 0.008877745971729046, -0.008361976307962505, 0.007876181274528127, -0.007418642356788098, 0.006987739799033855, -0.006581946966943887, 0.006199825106351055, -0.00584001837117817, 0.005501249138059018, -0.0051823135959307666, 0.004882077534036101, -0.004599472449233056, 0.004333491845900562, -0.004083187738391304, 0.00384766734038441, -0.0036260899632846967, 0.00341766412351482, -0.003221644783037071, 0.003037330724301647, -0.0028640621256170408, 0.0027012182634600047, -0.0025482153614670667, 0.00240450452795168, -0.0022695698397005816, 0.002142926545674241, -0.0020241193744505405, 0.0019127209575542047, -0.0018083302956923518, 0.0017105713491642703, -0.0016190917803071369, 0.0015335616642137794, -0.0014536723452853405, 0.001379135339081262, -0.0013096813358352787, 0.001245059275896766, -0.0011850354079059, 0.001129392510023498, -0.0010779290997626433, 0.0010304587604847658, -0.0009868094600730913, 0.0009468229117978965, -0.0009103540735282088, 0.0008772706097128445, -0.0008474524304184726, 0.0008207912556403528, -0.0007971902270068286, 0.000776563594266667, -0.0007588363976502542, 0.0007439441894165576, -0.0007318328255327643, 0.0007224582401159317, -0.0007157863644244543, 0.0007117929301416425, -0.0003552316997513632]
    P_zero_g_cheb_limits = (-0.9648141211597231, 34.80547339996925)

    # Nov 2019
#    Psat_ranges_low = (0.016623281310365744, 0.1712825877822172, 0.8775228637034642, 2.4185778704993384, 4.999300695376596, 10.621733701210832, 21.924686089216046, 46.23658652939059, 111.97303237634476)
#    Psat_coeffs_low = [[3.3680989254730784e+17, -4.074818435768317e+16, 2209815483748018.2, -70915386325767.22, 1497265032843.7883, -21873765985.033226, 226390057.35154417, -1671116.2651395416, 8737.327630885395, -31.38783252903762, -0.3062576872041857, 0.33312130842499577, -3.053621478895965, -3.31001545617049e-11], [-1277.9939354449687, 1593.2536430725866, -891.7360022419283, 295.8342513857935, -64.78832619622327, 9.999684056700664, -1.2867875988497843, 0.32779561506053606, -0.2700702867816281, 0.3102313474312917, -0.38304293907646136, 0.3332375577689779, -3.0536215764869326, 2.360556194958008e-12], [-0.000830505972635258, 0.006865390327553869, -0.026817234829898506, 0.0665672622542815, -0.119964606281312, 0.17169598695361063, -0.2106764625423519, 0.23752105248153738, -0.2630589319705226, 0.3095504696125893, -0.3829670684832488, 0.3332307853291866, -3.053621198743048, -9.605050754757372e-09], [-3.9351433749518387e-07, 9.598894454703375e-06, -0.00010967371180484353, 0.0007796748366203162, -0.0038566383026144148, 0.014042499802344673, -0.03890674173205208, 0.08460429046640522, -0.15233989442440943, 0.24596202389042182, -0.3552486542779753, 0.32467374082763434, -3.0519642694194045, -0.00015063394168279842], [9.630665339915374e-10, -4.7371315246695036e-08, 1.058979499331494e-06, -1.4148499852908107e-05, 0.00012453404851616215, -0.000744874809594022, 0.00295128240256063, -0.006592268033281193, -0.00018461593083082123, 0.055334589912369295, -0.18248894355952128, 0.21859711935534465, -3.0129799097945456, -0.006486114455004355], [-2.710468940879406e-14, 2.361990546087112e-12, -8.567303244166706e-11, 1.4893663407003366e-09, -3.858548795875803e-09, -4.2380960424066427e-07, 1.1242926127857602e-05, -0.0001632710637823122, 0.001612992494042694, -0.011564861884906304, 0.06197160418123528, -0.25590435995049254, -2.502008242669764, -0.2488201810754127], [1.6083147737513614e-17, -3.625948333600919e-15, 3.779796571460543e-13, -2.4156008540207418e-11, 1.0579233657428334e-09, -3.361617809293566e-08, 8.003083689291334e-07, -1.4534087164009375e-05, 0.0002031152121569259, -0.002189767259610491, 0.018210271943770506, -0.11793369058835379, -2.7679151499088324, -0.010786625844602327], [1.4110844119182223e-21, -6.673466228406512e-19, 1.458414381897635e-16, -1.9526180721541405e-14, 1.790075947985849e-12, -1.1894965358505194e-10, 5.91471378248656e-09, -2.2398516390225003e-07, 6.512345505221323e-06, -0.0001455617494641103, 0.002494987494838556, -0.03292429235639192, -3.0591018122950038, 0.4673525314587721], [2.1123051961710074e-25, -2.1083091388936946e-22, 9.662063972407386e-20, -2.693978918168614e-17, 5.1041762501040065e-15, -6.950692277142413e-13, 7.0161397235176e-11, -5.335818543025505e-09, 3.076755677498343e-07, -1.3436008106355354e-05, 0.00044163897730386165, -0.010903751691783911, -3.2044489982966082, 0.9087101274749898]]
    # Dec 2019
    Psat_ranges_low = (0.016623281310365744, 0.1712825877822172, 0.8775228637034642, 2.4185778704993384, 5.001795116965993, 9.695206781787403, 21.057832476172877, 46.27931918489475, 106.60008206151481, 206.46094535380982)
    Psat_coeffs_low = [[-3.021742864809473e+22, 4.214239383836392e+21, -2.6741136978422124e+20, 1.0217841941834105e+19, -2.622411357075726e+17, 4774407979621216.0, -63484876960438.69, 625375442527.9032, -4581025644.8943615, 24826982.029340215, -98159.10795313062, 276.50340512054163, -0.9143654631342611, 0.3338916360475657, -3.0536220324119654, 1.3475048507571863e-10], [1851732.9501797194, -2605860.7399044684, 1659613.6490526376, -632650.6574176748, 160893.98711246205, -28808.208934565508, 3736.231492867314, -355.6971459537775, 24.760714782747538, -1.0423063013028406, -0.21733901552867047, 0.3087713311546577, -0.3830146794101142, 0.3332371947330795, -3.05362157370627, -7.227607401461e-12], [-0.0004521121892877062, 0.004090442697504045, -0.0175578643282661, 0.04794251207356016, -0.09461038038864959, 0.14623339766010854, -0.18881543977182574, 0.216237584862916, -0.23240996967465383, 0.2455135904651031, -0.2652544858737314, 0.30999258403096164, -0.383030205401439, 0.33323681959713436, -3.053621543830392, -7.017832981404126e-10], [-7.808701919427604e-08, 2.077661560061249e-06, -2.5817479151146433e-05, 0.00019946831868046186, -0.0010777009706245766, 0.004349178573005843, -0.01369170969849733, 0.03466227625915358, -0.07207265601431376, 0.12553031872708703, -0.19076746372442283, 0.2729239080553058, -0.36893226041645655, 0.32941620336187777, -3.0529679799065943, -5.2835266906470224e-05], [1.4671478312986887e-11, -1.0110442264467632e-09, 3.1974970384785367e-08, -6.163209773850742e-07, 8.094505145684214e-06, -7.656745582686593e-05, 0.0005363374762358416, -0.002806916403123579, 0.010866043155195763, -0.029908987582571975, 0.052158093336682546, -0.032663621693629505, -0.0751716186255902, 0.12892284191276268, -2.967051324985992, -0.017359415309005755], [-1.091394193997453e-14, 1.2357554812857725e-12, -6.508815052182767e-11, 2.1148805902814486e-09, -4.738775276800668e-08, 7.750511363006775e-07, -9.547217015303917e-06, 9.00036272090859e-05, -0.0006521700537524791, 0.0036045130640996606, -0.014814913109019256, 0.04242862345426275, -0.06750190169757553, -0.04208608128453949, -2.7194330511309253, -0.14620471607422303], [1.2884100931052691e-19, -3.1352234465014476e-17, 3.5604267936540494e-15, -2.505139585733175e-13, 1.222651372599914e-11, -4.3907452596568276e-10, 1.200890247630985e-08, -2.5540606353043675e-07, 4.275108728223701e-06, -5.664187536164573e-05, 0.0005945229499256919, -0.004930179620565587, 0.03219833012293544, -0.16707167951374932, -2.661698607705941, -0.11728474036871717], [1.4718778168288712e-24, -7.827527144097033e-22, 1.9419598473231654e-19, -2.9838110847734e-17, 3.17855768668334e-15, -2.4899868244476545e-13, 1.4844855897901513e-11, -6.8756271353964e-10, 2.503217441471856e-08, -7.201291894218883e-07, 1.637034511468395e-05, -0.0002928284260408074, 0.004096142697714258, -0.0448856246444143, -3.0042011777749145, 0.3506385318223977], [8.493893312557766e-30, -1.0255827364680145e-26, 5.772959502607649e-24, -2.0110473459289428e-21, 4.853221287871829e-19, -8.60543209118676e-17, 1.1601546784661254e-14, -1.2138195783117383e-12, 9.970282232925484e-11, -6.461623477876083e-09, 3.3028403185783974e-07, -1.3249473054302956e-05, 0.00041394079374352697, -0.010055381746092074, -3.217048228957832, 0.986564340774919], [5.909724328094521e-34, -1.373533672302407e-30, 1.4873473750776103e-27, -9.960045776404399e-25, 4.616398303867023e-22, -1.5703667768168258e-19, 4.056122702342784e-17, -8.116898226364067e-15, 1.2725759075121173e-12, -1.5701193776679807e-10, 1.5228962066971988e-08, -1.1543563076442494e-06, 6.776387262920499e-05, -0.0030684174426771718, -3.30604432857107, 1.5254388255202684]]

    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.m = 0.480 + 1.574*omega - 0.176*omega*omega
        self.delta = self.b
        self.solve()

    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate :math:`a \alpha` and its first and second
        derivatives for this EOS. Uses the set values of `Tc`, `m`, and `a`.

        .. math::
            a\alpha = a \left(m \left(- \sqrt{\frac{T}{T_{c}}} + 1\right)
            + 1\right)^{2}

        .. math::
            \frac{d a\alpha}{dT} = \frac{a m}{T} \sqrt{\frac{T}{T_{c}}} \left(m
            \left(\sqrt{\frac{T}{T_{c}}} - 1\right) - 1\right)

        .. math::
            \frac{d^2 a\alpha}{dT^2} = \frac{a m \sqrt{\frac{T}{T_{c}}}}{2 T^{2}}
            \left(m + 1\right)

        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by
            EOS-specific method, [J^2/mol^2/Pa/K^2]
        '''
        a, Tc, m = self.a, self.Tc, self.m
        sqTr = (T/Tc)**0.5
        a_alpha = a*(m*(1. - sqTr) + 1.)**2
        da_alpha_dT = -a*m*sqTr*(m*(-sqTr + 1.) + 1.)/T
        d2a_alpha_dT2 =  a*m*sqTr*(m + 1.)/(2.*T*T)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        r'''Method to calculate :math:`a \alpha` for this EOS. Uses the set
        values of `Tc`, `m`, and `a`.

        .. math::
            a\alpha = a \left(m \left(- \sqrt{\frac{T}{T_{c}}} + 1\right)
            + 1\right)^{2}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the values, [-]

        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        '''
        a, Tc, m = self.a, self.Tc, self.m
        sqTr = sqrt(T/Tc)
        x0 = (m*(1. - sqTr) + 1.)
        return a*x0*x0

    def P_max_at_V(self, V):
        r'''Method to calculate the maximum pressure the EOS can create at a
        constant volume, if one exists; returns None otherwise.

        Parameters
        ----------
        V : float
            Constant molar volume, [m^3/mol]

        Returns
        -------
        P : float
            Maximum possible isochoric pressure, [Pa]

        Notes
        -----
        The analytical determination of this formula involved some part of the
        discriminant, and much black magic.

        Examples
        --------
        >>> e = SRK(P=1e5, V=0.0001437, Tc=512.5, Pc=8084000.0, omega=0.559)
        >>> e.P_max_at_V(e.V)
        490523786.2
        '''
        """
        from sympy import *
        # Solve for when T equal
        P, T, V, R, a, b, m = symbols('P, T, V, R, a, b, m')
        Tc, Pc, omega = symbols('Tc, Pc, omega')

        # from the T solution, get the square root part, find when it hits zero
        # to_zero = sqrt(Tc**2*V*a**2*m**2*(V - b)**3*(V + b)*(m + 1)**2*(P*R*Tc*V**2 + P*R*Tc*V*b - P*V*a*m**2 + P*a*b*m**2 + R*Tc*a*m**2 + 2*R*Tc*a*m + R*Tc*a))

        lhs = P*R*Tc*V**2 + P*R*Tc*V*b - P*V*a*m**2 + P*a*b*m**2
        rhs = R*Tc*a*m**2 + 2*R*Tc*a*m + R*Tc*a
        hit = solve(Eq(lhs, rhs), P)
        """
        # grows unbounded for all mixture EOS?
        try:
            Tc, a, m, b = self.Tc, self.a, self.m, self.b
        except:
            Tc, a, m, b = self.Tcs[0], self.ais[0], self.ms[0], self.bs[0]

        P_max = -R*Tc*a*(m**2 + 2*m + 1)/(R*Tc*V**2 + R*Tc*V*b - V*a*m**2 + a*b*m**2)
        if P_max < 0.0:
            return None
        return P_max


    def solve_T(self, P, V, solution=None):
        r'''Method to calculate `T` from a specified `P` and `V` for the SRK
        EOS. Uses `a`, `b`, and `Tc` obtained from the class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        solution : str or None, optional
            'l' or 'g' to specify a liquid of vapor solution (if one exists);
            if None, will select a solution more likely to be real (closer to
            STP, attempting to avoid temperatures like 60000 K or 0.0001 K).

        Returns
        -------
        T : float
            Temperature, [K]

        Notes
        -----
        The exact solution can be derived as follows; it is excluded for
        breviety.

        >>> from sympy import * # doctest:+SKIP
        >>> P, T, V, R, a, b, m = symbols('P, T, V, R, a, b, m') # doctest:+SKIP
        >>> Tc, Pc, omega = symbols('Tc, Pc, omega') # doctest:+SKIP
        >>> a_alpha = a*(1 + m*(1-sqrt(T/Tc)))**2 # doctest:+SKIP
        >>> SRK = R*T/(V-b) - a_alpha/(V*(V+b)) - P # doctest:+SKIP
        >>> solve(SRK, T) # doctest:+SKIP
        '''
        # Takes like half an hour to be derived, saved here for convenience
#         ([(Tc*(V - b)*(R**2*Tc**2*V**4 + 2*R**2*Tc**2*V**3*b + R**2*Tc**2*V**2*b**2
#        - 2*R*Tc*V**3*a*m**2 + 2*R*Tc*V*a*b**2*m**2 + V**2*a**2*m**4 - 2*V*a**2*b*m**4
#        + a**2*b**2*m**4)*(P*R*Tc*V**4 + 2*P*R*Tc*V**3*b + P*R*Tc*V**2*b**2
#                        - P*V**3*a*m**2 + P*V*a*b**2*m**2 + R*Tc*V**2*a*m**2
#                        + 2*R*Tc*V**2*a*m + R*Tc*V**2*a + R*Tc*V*a*b*m**2
#                        + 2*R*Tc*V*a*b*m + R*Tc*V*a*b + V*a**2*m**4 + 2*V*a**2*m**3
#                        + V*a**2*m**2 - a**2*b*m**4 - 2*a**2*b*m**3 - a**2*b*m**2)
#                - 2*sqrt(Tc**2*V*a**2*m**2*(V - b)**3*(V + b)*(m + 1)**2*(P*R*Tc*V**2
#                         + P*R*Tc*V*b - P*V*a*m**2 + P*a*b*m**2 + R*Tc*a*m**2 + 2*R*Tc*a*m + R*Tc*a))*(R*Tc*V**2 + R*Tc*V*b - V*a*m**2 + a*b*m**2)**2)/((R*Tc*V**2 + R*Tc*V*b - V*a*m**2 + a*b*m**2)**2*(R**2*Tc**2*V**4 + 2*R**2*Tc**2*V**3*b + R**2*Tc**2*V**2*b**2 - 2*R*Tc*V**3*a*m**2 + 2*R*Tc*V*a*b**2*m**2 + V**2*a**2*m**4 - 2*V*a**2*b*m**4 + a**2*b**2*m**4)),
#        (Tc*(V - b)*(R**2*Tc**2*V**4 + 2*R**2*Tc**2*V**3*b + R**2*Tc**2*V**2*b**2
#        - 2*R*Tc*V**3*a*m**2 + 2*R*Tc*V*a*b**2*m**2 + V**2*a**2*m**4 - 2*V*a**2*b*m**4
#        + a**2*b**2*m**4)*(P*R*Tc*V**4 + 2*P*R*Tc*V**3*b + P*R*Tc*V**2*b**2
#                        - P*V**3*a*m**2 + P*V*a*b**2*m**2 + R*Tc*V**2*a*m**2
#                        + 2*R*Tc*V**2*a*m + R*Tc*V**2*a + R*Tc*V*a*b*m**2
#                        + 2*R*Tc*V*a*b*m + R*Tc*V*a*b + V*a**2*m**4 + 2*V*a**2*m**3
#                        + V*a**2*m**2 - a**2*b*m**4 - 2*a**2*b*m**3 - a**2*b*m**2)
#                + 2*sqrt(Tc**2*V*a**2*m**2*(V - b)**3*(V + b)*(m + 1)**2*(P*R*Tc*V**2
#                         + P*R*Tc*V*b - P*V*a*m**2 + P*a*b*m**2 + R*Tc*a*m**2
#                         + 2*R*Tc*a*m + R*Tc*a))*(R*Tc*V**2 + R*Tc*V*b - V*a*m**2
#                + a*b*m**2)**2)/((R*Tc*V**2 + R*Tc*V*b - V*a*m**2 + a*b*m**2
#                              )**2*(R**2*Tc**2*V**4 + 2*R**2*Tc**2*V**3*b
#                              + R**2*Tc**2*V**2*b**2 - 2*R*Tc*V**3*a*m**2
#                              + 2*R*Tc*V*a*b**2*m**2 + V**2*a**2*m**4
#                              - 2*V*a**2*b*m**4 + a**2*b**2*m**4))])
        self.no_T_spec = True
        a, b, Tc, m = self.a, self.b, self.Tc, self.m
        if True:
            x0 = R*Tc
            x1 = V*b
            x2 = x0*x1
            x3 = V*V
            x4 = x0*x3
            x5 = m*m
            x6 = a*x5
            x7 = b*x6
            x8 = V*x6
            x9 = (x2 + x4 + x7 - x8)**2
            x10 = x3*x3
            x11 = R*R*Tc*Tc
            x12 = a*a
            x13 = x5*x5
            x14 = x12*x13
            x15 = b*b
            x16 = x3*V
            x17 = a*x0
            x18 = x17*x5
            x19 = 2.*b*x16
            x20 = -2.*V*b*x14 + 2.*V*x15*x18 + x10*x11 + x11*x15*x3 + x11*x19 + x14*x15 + x14*x3 - 2*x16*x18
            x21 = V - b
            x22 = 2*m*x17
            x23 = P*x4
            x24 = P*x8
            x25 = x1*x17
            x26 = P*R*Tc
            x27 = x17*x3
            x28 = V*x12
            x29 = 2.*m*m*m
            x30 = b*x12
            T_calc = -Tc*(2.*a*m*x9*(V*x21*x21*x21*(V + b)*(P*x2 + P*x7 + x17 + x18 + x22 + x23 - x24))**0.5*(m + 1.) - x20*x21*(-P*x16*x6 + x1*x22 + x10*x26 + x13*x28 - x13*x30 + x15*x23 + x15*x24 + x19*x26 + x22*x3 + x25*x5 + x25 + x27*x5 + x27 + x28*x29 + x28*x5 - x29*x30 - x30*x5))/(x20*x9)
            if abs(T_calc.imag) > 1e-12:
                raise ValueError(f"Calculated imaginary temperature {T_calc}")
            return T_calc
        else:
            return Tc*(-2*a*m*sqrt(V*(V - b)**3*(V + b)*(P*R*Tc*V**2 + P*R*Tc*V*b - P*V*a*m**2 + P*a*b*m**2 + R*Tc*a*m**2 + 2*R*Tc*a*m + R*Tc*a))*(m + 1)*(R*Tc*V**2 + R*Tc*V*b - V*a*m**2 + a*b*m**2)**2 + (V - b)*(R**2*Tc**2*V**4 + 2*R**2*Tc**2*V**3*b + R**2*Tc**2*V**2*b**2 - 2*R*Tc*V**3*a*m**2 + 2*R*Tc*V*a*b**2*m**2 + V**2*a**2*m**4 - 2*V*a**2*b*m**4 + a**2*b**2*m**4)*(P*R*Tc*V**4 + 2*P*R*Tc*V**3*b + P*R*Tc*V**2*b**2 - P*V**3*a*m**2 + P*V*a*b**2*m**2 + R*Tc*V**2*a*m**2 + 2*R*Tc*V**2*a*m + R*Tc*V**2*a + R*Tc*V*a*b*m**2 + 2*R*Tc*V*a*b*m + R*Tc*V*a*b + V*a**2*m**4 + 2*V*a**2*m**3 + V*a**2*m**2 - a**2*b*m**4 - 2*a**2*b*m**3 - a**2*b*m**2))/((R*Tc*V**2 + R*Tc*V*b - V*a*m**2 + a*b*m**2)**2*(R**2*Tc**2*V**4 + 2*R**2*Tc**2*V**3*b + R**2*Tc**2*V**2*b**2 - 2*R*Tc*V**3*a*m**2 + 2*R*Tc*V*a*b**2*m**2 + V**2*a**2*m**4 - 2*V*a**2*b*m**4 + a**2*b**2*m**4))

class SRKTranslated(SRK):
    r'''Class for solving the volume translated Peng-Robinson equation of state.
    Subclasses :obj:`SRK`. Solves the EOS on initialization.
    This is intended as a base class for all translated variants of the
    SRK EOS.

    .. math::
        P = \frac{RT}{V + c - b} - \frac{a\alpha(T)}{(V + c)(V + c + b)}

    .. math::
        a=\left(\frac{R^2(T_c)^{2}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2}}{P_c}

    .. math::
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}

    .. math::
        \alpha(T) = \left[1 + m\left(1 - \sqrt{\frac{T}{T_c}}\right)\right]^2

    .. math::
        m = 0.480 + 1.574\omega - 0.176\omega^2

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    alpha_coeffs : tuple or None
        Coefficients which may be specified by subclasses; set to None to use
        the original Peng-Robinson alpha function, [-]
    c : float, optional
        Volume translation parameter, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    P-T initialization:

    >>> eos = SRKTranslated(T=305, P=1.1e5, Tc=512.5, Pc=8084000.0, omega=0.559, c=-1e-6)
    >>> eos.phase, eos.V_l, eos.V_g
    ('l/g', 5.5131657318e-05, 0.022447661363)

    Notes
    -----

    References
    ----------
    .. [1] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen Rarey.
       Chemical Thermodynamics for Process Simulation. John Wiley & Sons, 2019.
    '''

    solve_T = GCEOS.solve_T
    P_max_at_V = GCEOS.P_max_at_V
    kwargs_keys = ('c', 'alpha_coeffs')
    def __init__(self, Tc, Pc, omega, alpha_coeffs=None, c=0.0, T=None, P=None,
                 V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        Pc_inv = 1.0/Pc

        self.a = self.c1*R*R*Tc*Tc*Pc_inv

        self.c = c
        if alpha_coeffs is None:
            self.m = 0.480 + 1.574*omega - 0.176*omega*omega

        self.alpha_coeffs = alpha_coeffs
        self.kwargs = {'c': c, 'alpha_coeffs': alpha_coeffs}

        b0 = self.c2*R*Tc*Pc_inv
        self.b = b0 - c

        ### from sympy.abc import V, c, b, epsilon, delta
        ### expand((V+c)*((V+c)+b))
        # delta = (b + 2*c)
        self.delta = c + c + b0
        # epsilon = b*c + c*c
        self.epsilon = c*(b0 + c)


        self.solve()


class MSRKTranslated(Soave_1979_a_alpha, SRKTranslated):
    r'''Class for solving the volume translated Soave (1980) alpha function,
    revision of the Soave-Redlich-Kwong equation of state
    for a pure compound according to [1]_. Uses two fitting parameters `N` and
    `M` to more accurately fit the vapor pressure of pure species.
    Subclasses `SRKTranslated`.
    Solves the EOS on initialization. See `SRKTranslated` for further
    documentation.

    .. math::
        P = \frac{RT}{V + c - b} - \frac{a\alpha(T)}{(V + c)(V + c + b)}

    .. math::
        a=\left(\frac{R^2(T_c)^{2}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2}}{P_c}

    .. math::
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}

    .. math::
        \alpha(T) = 1 + (1 - T_r)(M + \frac{N}{T_r})

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    c : float, optional
        Volume translation parameter, [m^3/mol]
    alpha_coeffs : tuple(float[3]), optional
        Coefficients M, N of this EOS's alpha function, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    P-T initialization (hexane), liquid phase:

    >>> eos = MSRKTranslated(Tc=507.6, Pc=3025000, omega=0.2975, c=22.0561E-6, M=0.7446, N=0.2476, T=250., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.0001169276461322, -34571.6862673, -84.757900348)

    Notes
    -----
    This is an older correlation that offers lower accuracy on many properties
    which were sacrificed to obtain the vapor pressure accuracy. The alpha
    function of this EOS does not meet any of the consistency requriements for
    alpha functions.

    Coefficients can be found in [2]_, or estimated with the method in [3]_.
    The estimation method in [3]_ works as follows, using the acentric factor
    and true critical compressibility:

    .. math::
        M = 0.4745 + 2.7349(\omega Z_c) + 6.0984(\omega Z_c)^2

    .. math::
        N = 0.0674 + 2.1031(\omega Z_c) + 3.9512(\omega Z_c)^2

    An alternate estimation scheme is provided in [1]_, which provides
    analytical solutions to calculate the parameters `M` and `N` from two
    points on the vapor pressure curve, suggested as 10 mmHg and 1 atm.
    This is used as an estimation method here if the parameters are not
    provided, and the two vapor pressure points are obtained from the original
    SRK equation of state.

    References
    ----------
    .. [1] Soave, G. "Rigorous and Simplified Procedures for Determining
       the Pure-Component Parameters in the Redlich—Kwong—Soave Equation of
       State." Chemical Engineering Science 35, no. 8 (January 1, 1980):
       1725-30. https://doi.org/10.1016/0009-2509(80)85007-X.
    .. [2] Sandarusi, Jamal A., Arthur J. Kidnay, and Victor F. Yesavage.
       "Compilation of Parameters for a Polar Fluid Soave-Redlich-Kwong
       Equation of State." Industrial & Engineering Chemistry Process Design
       and Development 25, no. 4 (October 1, 1986): 957-63.
       https://doi.org/10.1021/i200035a020.
    .. [3] Valderrama, Jose O., Héctor De la Puente, and Ahmed A. Ibrahim.
       "Generalization of a Polar-Fluid Soave-Redlich-Kwong Equation of State."
       Fluid Phase Equilibria 93 (February 11, 1994): 377-83.
       https://doi.org/10.1016/0378-3812(94)87021-7.
    '''

    kwargs_keys = ('c', 'alpha_coeffs')
    def __init__(self, Tc, Pc, omega, M=None, N=None, alpha_coeffs=None, c=0.0,
                 T=None, P=None, V=None):
        # Ready for mixture class implemenentation
        # M, N may be specified instead of alpha_coeffs currently only

        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        Pc_inv = 1.0/Pc

        self.a = self.c1*R*R*Tc*Tc*Pc_inv

        self.c = c

        b0 = self.c2*R*Tc*Pc_inv
        self.b = b0 - c

        self.delta = c + c + b0
        self.epsilon = c*(b0 + c)
        if alpha_coeffs is None and (M is None or N is None):
            alpha_coeffs = MSRKTranslated.estimate_MN(Tc, Pc, omega, c)
        if M is not None and N is not None:
            alpha_coeffs = (M, N)
        self.alpha_coeffs = alpha_coeffs

        self.kwargs = {'c': c, 'alpha_coeffs': alpha_coeffs}

        self.solve()

    @staticmethod
    def estimate_MN(Tc, Pc, omega, c=0.0):
        r'''Calculate the alpha values for the MSRK equation to match two pressure
        points, and solve analytically for the M, N required to match exactly that.
        Since no experimental data is available, make it up with the original
        SRK EOS.

        Parameters
        ----------
        Tc : float
            Critical temperature, [K]
        Pc : float
            Critical pressure, [Pa]
        omega : float
            Acentric factor, [-]
        c : float, optional
            Volume translation parameter, [m^3/mol]

        Returns
        -------
        M : float
            M parameter, [-]
        N : float
            N parameter, [-]

        Examples
        --------
        >>> from sympy import *  # doctest:+SKIP
        >>> Tc, m, n = symbols('Tc, m, n')  # doctest:+SKIP
        >>> T0, T1 = symbols('T_10, T_760')  # doctest:+SKIP
        >>> alpha0, alpha1 = symbols('alpha_10, alpha_760')  # doctest:+SKIP
        >>> Eqs = [Eq(alpha0, 1 + (1 - T0/Tc)*(m + n/(T0/Tc))), Eq(alpha1, 1 + (1 - T1/Tc)*(m + n/(T1/Tc)))]  # doctest:+SKIP
        >>> solve(Eqs, [n, m])  # doctest:+SKIP
        '''
        SRK_base = SRKTranslated(T=Tc*0.5, P=Pc*0.5, c=c, Tc=Tc, Pc=Pc, omega=omega)
        # Temperatures at 10 mmHg, 760 mmHg
        P_10, P_760 = 10.0*mmHg, 760.0*mmHg
        T_10 = SRK_base.Tsat(P_10)
        T_760 = SRK_base.Tsat(P_760)



        alpha_10 = SRK_base.a_alpha_and_derivatives(T=T_10, full=False)/SRK_base.a
        alpha_760 = SRK_base.a_alpha_and_derivatives(T=T_760, full=False)/SRK_base.a

        N = T_10*T_760*(-(T_10 - Tc)*(alpha_760 - 1) + (T_760 - Tc)*(alpha_10 - 1))/((T_10 - T_760)*(T_10 - Tc)*(T_760 - Tc))
        M = Tc*(-T_10*(T_760 - Tc)*(alpha_10 - 1) + T_760*(T_10 - Tc)*(alpha_760 - 1))/((T_10 - T_760)*(T_10 - Tc)*(T_760 - Tc))
        return (M, N)



class SRKTranslatedPPJP(SRK):
    r'''Class for solving the volume translated Pina-Martinez, Privat, Jaubert,
    and Peng revision of the Soave-Redlich-Kwong equation of state
    for a pure compound according to [1]_.
    Subclasses `SRK`, which provides everything except the variable `kappa`.
    Solves the EOS on initialization. See `SRK` for further documentation.

    .. math::
        P = \frac{RT}{V + c - b} - \frac{a\alpha(T)}{(V + c)(V + c + b)}

    .. math::
        a=\left(\frac{R^2(T_c)^{2}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2}}{P_c}

    .. math::
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}

    .. math::
        \alpha(T) = \left[1 + m\left(1 - \sqrt{\frac{T}{T_c}}\right)\right]^2

    .. math::
        m = 0.4810 + 1.5963 \omega - 0.2963\omega^2 + 0.1223\omega^3

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    c : float, optional
        Volume translation parameter, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    P-T initialization (hexane), liquid phase:

    >>> eos = SRKTranslatedPPJP(Tc=507.6, Pc=3025000, omega=0.2975, c=22.3098E-6, T=250., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.00011666322408111662, -34158.934132722185, -83.06507748137201)

    Notes
    -----
    This variant offers incremental improvements in accuracy only, but those
    can be fairly substantial for some substances.

    References
    ----------
    .. [1] Pina-Martinez, Andrés, Romain Privat, Jean-Noël Jaubert, and
       Ding-Yu Peng. "Updated Versions of the Generalized Soave a-Function
       Suitable for the Redlich-Kwong and Peng-Robinson Equations of State."
       Fluid Phase Equilibria, December 7, 2018.
       https://doi.org/10.1016/j.fluid.2018.12.007.
    '''

    kwargs_keys = ('c',)
    # No point in subclassing SRKTranslated - just disables direct solver for T
    def __init__(self, Tc, Pc, omega, c=0.0, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        Pc_inv = 1.0/Pc
        self.a = self.c1*R2*Tc*Tc*Pc_inv
        self.c = c
        self.m = omega*(omega*(0.1223*omega - 0.2963) + 1.5963) + 0.4810
        self.kwargs = {'c': c}

        b0 = self.c2*R*Tc*Pc_inv
        self.b = b0 - c
        self.delta = c + c + b0
        self.epsilon = c*(b0 + c)
        self.solve()


class SRKTranslatedTwu(Twu91_a_alpha, SRKTranslated): pass

class SRKTranslatedConsistent(Twu91_a_alpha, SRKTranslated):
    r'''Class for solving the volume translated Le Guennec, Privat, and Jaubert
    revision of the SRK equation of state
    for a pure compound according to [1]_.


    This model's `alpha` is based
    on the TWU 1991 model; when estimating, `N` is set to 2.
    Solves the EOS on initialization. See `SRK` for further documentation.

    .. math::
        P = \frac{RT}{V + c - b} - \frac{a\alpha(T)}{(V + c)(V + c + b)}

    .. math::
        a=\left(\frac{R^2(T_c)^{2}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2}}{P_c}

    .. math::
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}

    .. math::
        \alpha = \left(\frac{T}{T_{c}}\right)^{c_{3} \left(c_{2}
        - 1\right)} e^{c_{1} \left(- \left(\frac{T}{T_{c}}
        \right)^{c_{2} c_{3}} + 1\right)}

    If `c` is not provided, it is estimated as:

    .. math::
        c =\frac{R T_c}{P_c}(0.0172\omega - 0.0096)

    If `alpha_coeffs` is not provided, the parameters `L` and `M` are estimated
    from the acentric factor as follows:

    .. math::
        L = 0.0947\omega^2 + 0.6871\omega + 0.1508

    .. math::
        M = 0.1615\omega^2 - 0.2349\omega + 0.8876

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    alpha_coeffs : tuple(float[3]), optional
        Coefficients L, M, N (also called C1, C2, C3) of TWU 1991 form, [-]
    c : float, optional
        Volume translation parameter, [m^3/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    P-T initialization (methanol), liquid phase:

    >>> eos = SRKTranslatedConsistent(Tc=507.6, Pc=3025000, omega=0.2975, T=250., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.00011846802568940222, -34324.05211005662, -83.83861726864234)

    Notes
    -----
    This variant offers substantial improvements to the SRK-type EOSs - likely
    getting about as accurate as this form of cubic equation can get.

    References
    ----------
    .. [1] Le Guennec, Yohann, Romain Privat, and Jean-Noël Jaubert.
       "Development of the Translated-Consistent Tc-PR and Tc-RK Cubic
       Equations of State for a Safe and Accurate Prediction of Volumetric,
       Energetic and Saturation Properties of Pure Compounds in the Sub- and
       Super-Critical Domains." Fluid Phase Equilibria 429 (December 15, 2016):
       301-12. https://doi.org/10.1016/j.fluid.2016.09.003.
    '''

    kwargs_keys = ('c', 'alpha_coeffs')
    def __init__(self, Tc, Pc, omega, alpha_coeffs=None, c=None, T=None,
                 P=None, V=None):
        # estimates volume translation and alpha function parameters
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V
        Pc_inv = 1.0/Pc

        # limit oemga to 0.01 under the eos limit 1.47 for the estimation
        o = min(max(omega, -0.01), 1.46)
        if c is None:
            c = R*Tc*Pc_inv*(0.0172*o + 0.0096)

        if alpha_coeffs is None:
            L = o*(0.0947*o + 0.6871) + 0.1508
            M = o*(0.1615*o - 0.2349) + 0.8876
            N = 2.0
            alpha_coeffs = (L, M, N)

        self.c = c
        self.alpha_coeffs = alpha_coeffs
        self.kwargs = {'c': c, 'alpha_coeffs': alpha_coeffs}

        self.a = self.c1*R2*Tc*Tc*Pc_inv
        b0 = self.c2*R*Tc*Pc_inv
        self.b = b = b0 - c

        self.delta = c + c + b0
        self.epsilon = c*(b0 + c)

        self.solve()

class APISRK(SRK):
    r'''Class for solving the Refinery Soave-Redlich-Kwong cubic
    equation of state for a pure compound shown in the API Databook [1]_.
    Subclasses :obj:`GCEOS`, which
    provides the methods for solving the EOS and calculating its assorted
    relevant thermodynamic properties. Solves the EOS on initialization.

    Implemented methods here are `a_alpha_and_derivatives`, which sets
    :math:`a \alpha` and its first and second derivatives, and `solve_T`, which from a
    specified `P` and `V` obtains `T`. Two fit constants are used in this
    expresion, with an estimation scheme for the first if unavailable and the
    second may be set to zero.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}

    .. math::
        a=\left(\frac{R^2(T_c)^{2}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2}}{P_c}

    .. math::
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}

    .. math::
        \alpha(T) = \left[1 + S_1\left(1-\sqrt{T_r}\right) + S_2\frac{1
        - \sqrt{T_r}}{\sqrt{T_r}}\right]^2

    .. math::
        S_1 = 0.48508 + 1.55171\omega - 0.15613\omega^2 \text{ if S1 is not tabulated }

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float, optional
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    S1 : float, optional
        Fit constant or estimated from acentric factor if not provided [-]
    S2 : float, optional
        Fit constant or 0 if not provided [-]

    Examples
    --------
    >>> eos = APISRK(Tc=514.0, Pc=6137000.0, S1=1.678665, S2=-0.216396, P=1E6, T=299)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 7.0456950702e-05, -42826.286146, -103.626979037)

    References
    ----------
    .. [1] API Technical Data Book: General Properties & Characterization.
       American Petroleum Institute, 7E, 2005.
    '''

    kwargs_keys = ('S1', 'S2')

    def __init__(self, Tc, Pc, omega=None, T=None, P=None, V=None, S1=None,
                 S2=0):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V
        self.check_sufficient_inputs()

        if S1 is None and omega is None:
            raise Exception('Either acentric factor of S1 is required')

        if S1 is None:
            self.S1 = S1 = 0.48508 + 1.55171*omega - 0.15613*omega*omega
        else:
            self.S1 = S1
        self.S2 = S2
        self.kwargs = {'S1': S1, 'S2': S2}
        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = self.b

        self.solve()

    def a_alpha_and_derivatives_pure(self, T):
        r'''Method to calculate :math:`a \alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more
        documentation. Uses the set values of `Tc`, `a`, `S1`, and `S2`.

        .. math::
            a\alpha(T) = a\left[1 + S_1\left(1-\sqrt{T_r}\right) + S_2\frac{1
            - \sqrt{T_r}}{\sqrt{T_r}}\right]^2

        .. math::
            \frac{d a\alpha}{dT} = a\frac{T_{c}}{T^{2}} \left(- S_{2} \left(\sqrt{
            \frac{T}{T_{c}}} - 1\right) + \sqrt{\frac{T}{T_{c}}} \left(S_{1} \sqrt{
            \frac{T}{T_{c}}} + S_{2}\right)\right) \left(S_{2} \left(\sqrt{\frac{
            T}{T_{c}}} - 1\right) + \sqrt{\frac{T}{T_{c}}} \left(S_{1} \left(\sqrt{
            \frac{T}{T_{c}}} - 1\right) - 1\right)\right)

        .. math::
            \frac{d^2 a\alpha}{dT^2} = a\frac{1}{2 T^{3}} \left(S_{1}^{2} T
            \sqrt{\frac{T}{T_{c}}} - S_{1} S_{2} T \sqrt{\frac{T}{T_{c}}} + 3 S_{1}
            S_{2} Tc \sqrt{\frac{T}{T_{c}}} + S_{1} T \sqrt{\frac{T}{T_{c}}}
            - 3 S_{2}^{2} Tc \sqrt{\frac{T}{T_{c}}} + 4 S_{2}^{2} Tc + 3 S_{2}
            Tc \sqrt{\frac{T}{T_{c}}}\right)
        '''
        # possible TODO: custom hydrogen a_alpha from
        # Graboski, Michael S., and Thomas E. Daubert. "A Modified Soave Equation
        # of State for Phase Equilibrium Calculations. 3. Systems Containing
        # Hydrogen." Industrial & Engineering Chemistry Process Design and
        # Development 18, no. 2 (April 1, 1979): 300-306. https://doi.org/10.1021/i260070a022.
        # 1.202*exp(-.30228Tr)
        # Will require CAss in kwargs, is_hydrogen array (or skip vectorized approach)
        a, Tc, S1, S2 = self.a, self.Tc, self.S1, self.S2
        x0 = (T/Tc)**0.5
        x1 = x0 - 1.
        x2 = x1/x0
        x3 = S2*x2
        x4 = S1*x1 + x3 - 1.
        x5 = S1*x0
        x6 = S2 - x3 + x5
        x7 = 3.*S2
        a_alpha = a*x4*x4
        da_alpha_dT = a*x4*x6/T
        d2a_alpha_dT2 = a*(-x4*(-x2*x7 + x5 + x7) + x6*x6)/(2.*T*T)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def a_alpha_pure(self, T):
        a, Tc, S1, S2 = self.a, self.Tc, self.S1, self.S2
        return a*(S1*(-(T/Tc)**0.5 + 1.) + S2*(-(T/Tc)**0.5 + 1)*(T/Tc)**-0.5 + 1)**2

    def solve_T(self, P, V, solution=None):
        r'''Method to calculate `T` from a specified `P` and `V` for the API
        SRK EOS. Uses `a`, `b`, and `Tc` obtained from the class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        solution : str or None, optional
            'l' or 'g' to specify a liquid of vapor solution (if one exists);
            if None, will select a solution more likely to be real (closer to
            STP, attempting to avoid temperatures like 60000 K or 0.0001 K).

        Returns
        -------
        T : float
            Temperature, [K]

        Notes
        -----
        If S2 is set to 0, the solution is the same as in the SRK EOS, and that
        is used. Otherwise, newton's method must be used to solve for `T`.
        There are 8 roots of T in that case, six of them real. No guarantee can
        be made regarding which root will be obtained.
        '''
        self.no_T_spec = True
        if self.S2 == 0:
            self.m = self.S1
            return SRK.solve_T(self, P, V, solution=solution)

        else:
            # Previously coded method is  63 microseconds vs 47 here
#            return super(SRK, self).solve_T(P, V)
            Tc, a, b, S1, S2 = self.Tc, self.a, self.b, self.S1, self.S2
            x2 = R/(V-b)
            x3 = (V*(V + b))
            def to_solve(T):
                x0 = (T/Tc)**0.5
                x1 = x0 - 1.
                return (x2*T - a*(S1*x1 + S2*x1/x0 - 1.)**2/x3) - P

        if solution is None:
            try:
                return secant(to_solve, Tc*0.5)
            except:
                pass
        return GCEOS.solve_T(self, P, V, solution=solution)


    def P_max_at_V(self, V):
        if self.S2 == 0:
            self.m = self.S1
            return SRK.P_max_at_V(self, V)
        return GCEOS.P_max_at_V(self, V)


class TWUPR(TwuPR95_a_alpha, PR):
    r'''Class for solving the Twu (1995) [1]_ variant of the Peng-Robinson cubic
    equation of state for a pure compound. Subclasses :obj:`PR`, which
    provides the methods for solving the EOS and calculating its assorted
    relevant thermodynamic properties. Solves the EOS on initialization.

    The main implemented method here is :obj:`a_alpha_and_derivatives_pure`,
    which sets :math:`a \alpha` and its first and second derivatives.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

    .. math::
        a=0.45724\frac{R^2T_c^2}{P_c}

    .. math::
        b=0.07780\frac{RT_c}{P_c}

    .. math::
       \alpha = \alpha^{(0)} + \omega(\alpha^{(1)}-\alpha^{(0)})

    .. math::
       \alpha^{(i)} = T_r^{N(M-1)}\exp[L(1-T_r^{NM})]

    For sub-critical conditions:

    L0, M0, N0 =  0.125283, 0.911807,  1.948150;

    L1, M1, N1 = 0.511614, 0.784054, 2.812520

    For supercritical conditions:

    L0, M0, N0 = 0.401219, 4.963070, -0.2;

    L1, M1, N1 = 0.024955, 1.248089, -8.

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    >>> eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    >>> eos.V_l, eos.H_dep_l, eos.S_dep_l
    (0.00013017554170, -31652.73712, -74.112850429)

    Notes
    -----
    Claimed to be more accurate than the PR, PR78 and PRSV equations.

    There is no analytical solution for `T`. There are multiple possible
    solutions for `T` under certain conditions; no guaranteed are provided
    regarding which solution is obtained.

    References
    ----------
    .. [1] Twu, Chorng H., John E. Coon, and John R. Cunningham. "A New
       Generalized Alpha Function for a Cubic Equation of State Part 1.
       Peng-Robinson Equation." Fluid Phase Equilibria 105, no. 1 (March 15,
       1995): 49-59. doi:10.1016/0378-3812(94)02601-V.
    '''

    P_max_at_V = GCEOS.P_max_at_V
    solve_T = GCEOS.solve_T

    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V
        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = 2.*self.b
        self.epsilon = -self.b*self.b
        self.check_sufficient_inputs()

        self.solve()


class TWUSRK(TwuSRK95_a_alpha, SRK):
    r'''Class for solving the Soave-Redlich-Kwong cubic
    equation of state for a pure compound. Subclasses :obj:`GCEOS`, which
    provides the methods for solving the EOS and calculating its assorted
    relevant thermodynamic properties. Solves the EOS on initialization.

    The main implemented method here is :obj:`a_alpha_and_derivatives_pure`,
    which sets :math:`a \alpha` and its first and second derivatives.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}

    .. math::
        a=\left(\frac{R^2(T_c)^{2}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2}}{P_c}

    .. math::
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}

    .. math::
        \alpha = \alpha^{(0)} + \omega(\alpha^{(1)}-\alpha^{(0)})

    .. math::
        \alpha^{(i)} = T_r^{N(M-1)}\exp[L(1-T_r^{NM})]

    For sub-critical conditions:

    L0, M0, N0 =  0.141599, 0.919422, 2.496441

    L1, M1, N1 = 0.500315, 0.799457, 3.291790

    For supercritical conditions:

    L0, M0, N0 = 0.441411, 6.500018, -0.20

    L1, M1, N1 = 0.032580,  1.289098, -8.0

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    >>> eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.000146892222966, -31612.6025870, -74.022966093)

    Notes
    -----
    There is no analytical solution for `T`. There are multiple possible
    solutions for `T` under certain conditions; no guaranteed are provided
    regarding which solution is obtained.

    References
    ----------
    .. [1] Twu, Chorng H., John E. Coon, and John R. Cunningham. "A New
       Generalized Alpha Function for a Cubic Equation of State Part 2.
       Redlich-Kwong Equation." Fluid Phase Equilibria 105, no. 1 (March 15,
       1995): 61-69. doi:10.1016/0378-3812(94)02602-W.
    '''

    P_max_at_V = GCEOS.P_max_at_V
    solve_T = GCEOS.solve_T

    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = self.b
        self.check_sufficient_inputs()
        self.solve()



eos_list = [IG, PR, PR78, PRSV, PRSV2, VDW, RK, SRK, APISRK, TWUPR, TWUSRK,
            PRTranslatedPPJP, SRKTranslatedPPJP, MSRKTranslated,
            PRTranslatedConsistent, SRKTranslatedConsistent]
"""list : List of all cubic equation of state classes.
"""
eos_2P_list = list(eos_list)
"""list : List of all cubic equation of state classes that can represent
multiple phases.
"""

eos_2P_list.remove(IG)

eos_dict = {c.__name__: c for c in eos_list}
"""dict : Dict of all cubic equation of state classes, indexed by their class name.
"""

eos_full_path_dict = {c.__full_path__: c for c in eos_list}
"""dict : Dict of all cubic equation of state classes, indexed by their module path and class name.
"""

