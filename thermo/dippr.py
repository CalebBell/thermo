# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['EQ100', 'EQ101', 'EQ102', 'EQ104', 'EQ105', 'EQ106', 'EQ107', 
           'EQ114', 'EQ115', 'EQ116', 'EQ127']

from thermo.utils import log, exp, sinh, cosh


def EQ100(T, A=0, B=0, C=0, D=0, E=0, F=0, G=0):
    r'''DIPPR Equation # 100. Used in calculating the molar heat capacities
    of liquids and solids, liquid thermal conductivity, and solid density.
    All parameters default to zero. As this is a straightforward polynomial,
    no restrictions on parameters apply. Note that high-order polynomials like
    this may need large numbers of decimal places to avoid unnecessary error.

    .. math::
        Y = A + BT + CT^2 + DT^3 + ET^4 + FT^5 + GT^6

    Parameters
    ----------
    T : float
        Temperature, [K]
    A-G : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    Y : float
        Property [constant-specific]

    Examples
    --------
    Water liquid heat capacity; DIPPR coefficients normally listed in J/kmol/K.

    >>> EQ100(300, 276370., -2090.1, 8.125, -0.014116, 0.0000093701)
    75355.81

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    '''
    return A + B*T + C*T**2 + D*T**3 + E*T**4 + F*T**5 + G*T**6


def EQ101(T, A, B, C, D, E):
    r'''DIPPR Equation # 101. Used in calculating vapor pressure, sublimation
    pressure, and liquid viscosity.
    All 5 parameters are required. E is often an integer. As the model is
    exponential, a sufficiently high temperature will cause an OverflowError.
    A negative temperature (or just low, if fit poorly) may cause a math domain
    error.

    .. math::
        Y = \exp\left(A + \frac{B}{T} + C\cdot \ln T + D \cdot T^E\right)

    Parameters
    ----------
    T : float
        Temperature, [K]
    A-E : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    Y : float
        Property [constant-specific]

    Examples
    --------
    Water vapor pressure; DIPPR coefficients normally listed in Pa.

    >>> EQ101(300, 73.649, -7258.2, -7.3037, 4.1653E-6, 2)
    3537.44834545549

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    '''
    return exp(A + B/T + C*log(T) + D*T**E)


def EQ102(T, A, B, C, D):
    r'''DIPPR Equation # 102. Used in calculating vapor viscosity, vapor
    thermal conductivity, and sometimes solid heat capacity. High values of B
    raise an OverflowError.
    All 4 parameters are required. C and D are often 0.

    .. math::
        Y = \frac{A\cdot T^B}{1 + \frac{C}{T} + \frac{D}{T^2}}

    Parameters
    ----------
    T : float
        Temperature, [K]
    A-D : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    Y : float
        Property [constant-specific]

    Examples
    --------
    Water vapor viscosity; DIPPR coefficients normally listed in Pa*S.

    >>> EQ102(300, 1.7096E-8, 1.1146, 0, 0)
    9.860384711890639e-06

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    '''
    return A*T**B/(1. + C/T + D/(T*T))


def EQ104(T, A, B, C, D, E):
    r'''DIPPR Equation #104. Often used in calculating second virial
    coefficients of gases. All 5 parameters are required.
    C, D, and E are normally large values.

    .. math::
        Y = A + \frac{B}{T} + \frac{C}{T^3} + \frac{D}{T^8} + \frac{E}{T^9}

    Parameters
    ----------
    T : float
        Temperature, [K]
    A-E : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    Y : float
        Property [constant-specific]

    Examples
    --------
    Water second virial coefficient; DIPPR coefficients normally dimensionless.

    >>> EQ104(300, 0.02222, -26.38, -16750000, -3.894E19, 3.133E21)
    -1.1204179007265151

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    '''
    return A + B/T + C/T**3. + D/T**8. + E/T**9.


def EQ105(T, A, B, C, D):
    r'''DIPPR Equation #105. Often used in calculating liquid molar density.
    All 4 parameters are required. C is sometimes the fluid's critical
    temperature.

    .. math::
        Y = \frac{A}{B^{1 + (1-\frac{T}{C})^D}}

    Parameters
    ----------
    T : float
        Temperature, [K]
    A-D : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    Y : float
        Property [constant-specific]

    Examples
    --------
    Hexane molar density; DIPPR coefficients normally in kmol/m^3.

    >>> EQ105(300., 0.70824, 0.26411, 507.6, 0.27537)
    7.593170096339236

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    '''
    return A/B**(1 + (1-T/C)**D)


def EQ106(T, Tc, A, B, C=0, D=0, E=0):
    r'''DIPPR Equation #106. Often used in calculating liquid surface tension,
    and heat of vaporization.
    Only parameters A and B parameters are required; many fits include no
    further parameters. Critical temperature is also required.

    .. math::
        Y = A(1-T_r)^{B + C T_r + D T_r^2 + E T_r^3}

        Tr = \frac{T}{Tc}

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tc : float
        Critical temperature, [K]
    A-D : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    Y : float
        Property [constant-specific]

    Examples
    --------
    Water surface tension; DIPPR coefficients normally in Pa*S.

    >>> EQ106(300, 647.096, 0.17766, 2.567, -3.3377, 1.9699)
    0.07231499373541

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    '''
    Tr = T/Tc
    return A*(1.-Tr)**(B + C*Tr + D*Tr**2. + E*Tr**3.)


def EQ107(T, A=0, B=0, C=0, D=0, E=0):
    r'''DIPPR Equation #107. Often used in calculating ideal-gas heat capacity.
    All 5 parameters are required.
    Also called the Aly-Lee equation.

    .. math::
        Y = A + B\left[\frac{C/T}{\sinh(C/T)}\right]^2 + D\left[\frac{E/T}{
        \cosh(E/T)}\right]^2

    Parameters
    ----------
    T : float
        Temperature, [K]
    A-E : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    Y : float
        Property [constant-specific]

    Examples
    --------
    Water ideal gas molar heat capacity; DIPPR coefficients normally in
    J/kmol/K

    >>> EQ107(300., 33363., 26790., 2610.5, 8896., 1169.)
    33585.90452768923

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    .. [2] Aly, Fouad A., and Lloyd L. Lee. "Self-Consistent Equations for
       Calculating the Ideal Gas Heat Capacity, Enthalpy, and Entropy." Fluid
       Phase Equilibria 6, no. 3 (January 1, 1981): 169-79.
       doi:10.1016/0378-3812(81)85002-9.
    '''
    return A + B*((C/T)/sinh(C/T))**2 + D*((E/T)/cosh(E/T))**2


def EQ114(T, Tc, A, B, C, D):
    r'''DIPPR Equation #114. Rarely used, normally as an alternate liquid
    heat capacity expression. All 4 parameters are required, as well as
    critical temperature.

    .. math::
        Y = \frac{A^2}{\tau} + B - 2AC\tau - AD\tau^2 - \frac{1}{3}C^2\tau^3
        - \frac{1}{2}CD\tau^4 - \frac{1}{5}D^2\tau^5

        \tau = 1 - \frac{T}{Tc}

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tc : float
        Critical temperature, [K]
    A-D : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    Y : float
        Property [constant-specific]

    Examples
    --------
    Hydrogen liquid heat capacity; DIPPR coefficients normally in J/kmol/K.

    >>> EQ114(20, 33.19, 66.653, 6765.9, -123.63, 478.27)
    19423.948911676463

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    '''
    t = 1.-T/Tc
    return A**2./t + B - 2.*A*C*t - A*D*t**2. - C**2.*t**3./3. - C*D*t**4./2. - D**2*t**5./5.


def EQ115(T, A, B, C=0, D=0, E=0):
    r'''DIPPR Equation #115. No major uses; has been used as an alternate
    liquid viscosity expression, and as a model for vapor pressure.
    Only parameters A and B are required.

    .. math::
        Y = \exp\left(A + \frac{B}{T} + C\log T + D T^2 + \frac{E}{T^2}\right)

    Parameters
    ----------
    T : float
        Temperature, [K]
    A-E : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    Y : float
        Property [constant-specific]

    Examples
    --------
    No coefficients found for this expression.

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    '''
    return exp(A+B/T+C*log(T)+D*T**2 + E/T**2)


def EQ116(T, Tc, A, B, C, D, E):
    r'''DIPPR Equation #116. Used to describe the molar density of water fairly
    precisely; no other uses listed. All 5 parameters are needed, as well as
    the critical temperature.

    .. math::
        Y = A + B\tau^{0.35} + C\tau^{2/3} + D\tau + E\tau^{4/3}

        \tau = 1 - \frac{T}{Tc}

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tc : float
        Critical temperature, [K]
    A-E : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    Y : float
        Property [constant-specific]

    Examples
    --------
    Water liquid molar density; DIPPR coefficients normally in kmol/m^3.

    >>> EQ116(300., 647.096, 17.863, 58.606, -95.396, 213.89, -141.26)
    55.17615446406527

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    '''
    tau = 1-T/Tc
    return A + B*tau**0.35 + C*tau**(2/3.) + D*tau + E*tau**(4/3.)


def EQ127(T, A, B, C, D, E, F, G):
    r'''DIPPR Equation #127. Rarely used, and then only in calculating
    ideal-gas heat capacity. All 7 parameters are required.

    .. math::
        Y = A+B\left[\frac{\left(\frac{C}{T}\right)^2\exp\left(\frac{C}{T}
        \right)}{\left(\exp\frac{C}{T}-1 \right)^2}\right]
        +D\left[\frac{\left(\frac{E}{T}\right)^2\exp\left(\frac{E}{T}\right)}
        {\left(\exp\frac{E}{T}-1 \right)^2}\right]
        +F\left[\frac{\left(\frac{G}{T}\right)^2\exp\left(\frac{G}{T}\right)}
        {\left(\exp\frac{G}{T}-1 \right)^2}\right]

    Parameters
    ----------
    T : float
        Temperature, [K]
    A-G : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    Y : float
        Property [constant-specific]

    Examples
    --------
    Ideal gas heat capacity of methanol; DIPPR coefficients normally in
    J/kmol/K

    >>> EQ127(20., 3.3258E4, 3.6199E4, 1.2057E3, 1.5373E7, 3.2122E3, -1.5318E7, 3.2122E3)
    33258.0

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    '''
    return A+B*((C/T)**2*exp(C/T)/(exp(C/T) - 1)**2) + \
        D*((E/T)**2*exp(E/T)/(exp(E/T)-1)**2) + \
        F*((G/T)**2*exp(G/T)/(exp(G/T)-1)**2)
