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
from scipy.constants import R

__all__ = ['BVirial_Pitzer_Curl', 'BVirial_Abbott', 'BVirial_Tsonopoulos',
           'BVirial_Tsonopoulos_Extended']

### Second Virial Coefficients


def BVirial_Pitzer_Curl(T, Tc, Pc, omega):
    r'''Calculates the second virial coefficient using the model in [1]_.
    Designed for simple calculations.

    .. math::
        B_r=B^{(0)}+\omega B^{(1)}

        B^{(0)}=0.1445-0.33/T_r-0.1385/T_r^2-0.0121/T_r^3

        B^{(1)} = 0.073+0.46/T_r-0.5/T_r^2 -0.097/T_r^3 - 0.0073/T_r^8

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    omega : float
        Acentric factor for fluid, [-]

    Returns
    -------
    BVirial : float
        Second virial coefficient, [m^3/mol]

    Notes
    -----

    Examples
    --------
    Example matching that in BVirial_Abbott, for isobutane.

    >>> BVirial_Pitzer_Curl(510., 425.2, 38E5, 0.193)
    -0.0002084535541385102

    References
    ----------
    .. [1] Pitzer, Kenneth S., and R. F. Curl. "The Volumetric and
       Thermodynamic Properties of Fluids. III. Empirical Equation for the
       Second Virial Coefficient1." Journal of the American Chemical Society
       79, no. 10 (May 1, 1957): 2369-70. doi:10.1021/ja01567a007.
    '''
    Tr = T/Tc
    B0 = 0.1445 - 0.33/Tr - 0.1385/Tr**2 - 0.0121/Tr**3
    B1 = 0.073 + 0.46/Tr - 0.5/Tr**2 - 0.097/Tr**3 - 0.0073/Tr**8
    Br = B0 + omega*B1
    BVirial = Br*R*Tc/Pc
    return BVirial


def BVirial_Abbott(T, Tc, Pc, omega):
    r'''Calculates the second virial coefficient using the model in [1]_.
    Simple fit to the Lee-Kesler equation.

    .. math::
        B_r=B^{(0)}+\omega B^{(1)}

        B^{(0)}=0.083+\frac{0.422}{T_r^{1.6}}

        B^{(1)}=0.139-\frac{0.172}{T_r^{4.2}}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    omega : float
        Acentric factor for fluid, [-]

    Returns
    -------
    BVirial : float
        Second virial coefficient, [m^3/mol]

    Notes
    -----

    Examples
    --------
    Ecample is from [1]_, p. 93, and matches the result exactly, for Isobutane.

    >>> BVirial_Abbott(510., 425.2, 38E5, 0.193)
    -0.00020570178037383633

    References
    ----------
    .. [1] Smith, H. C. Van Ness Joseph M. Introduction to Chemical Engineering
       Thermodynamics 4E 1987.
    '''
    Tr = T/Tc
    B0 = 0.083 - 0.422/Tr**1.6
    B1 = 0.139 - 0.172/Tr**4.2
    Br = B0 + omega*B1
    BVirial = Br*R*Tc/Pc
    return BVirial


def BVirial_Tsonopoulos(T, Tc, Pc, omega):
    r'''Calculates the second virial coefficient using the model in [1]_.

    .. math::
        B_r=B^{(0)}+\omega B^{(1)}

        B^{(0)}=0.1445-0.33/T_r-0.1385/T_r^2-0.0121/T_r^3

        B^{(1)} = 0.0637+0.331/T_r^2-0.423/T_r^3 -0.423/T_r^3 - 0.008/T_r^8

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    omega : float
        Acentric factor for fluid, [-]

    Returns
    -------
    BVirial : float
        Second virial coefficient, [m^3/mol]

    Notes
    -----
    A more complete expression is also available, in
    BVirial_Tsonopoulos_Extended.

    Examples
    --------
    Example matching that in BVirial_Abbott, for isobutane.

    >>> BVirial_Tsonopoulos(510., 425.2, 38E5, 0.193)
    -0.00020935288308483694

    References
    ----------
    .. [1] Tsonopoulos, Constantine. "An Empirical Correlation of Second Virial
       Coefficients." AIChE Journal 20, no. 2 (March 1, 1974): 263-72.
       doi:10.1002/aic.690200209.
    '''
    Tr = T/Tc
    B0 = 0.1445 - 0.33/Tr - 0.1385/Tr**2 - 0.0121/Tr**3 - 0.000607/Tr**8
    B1 = 0.0637 + 0.331/Tr**2 - 0.423/Tr**3 - 0.008/Tr**8
    Br = (B0+omega*B1)
    BVirial = Br*R*Tc/Pc
    return BVirial


def BVirial_Tsonopoulos_Extended(T, Tc, Pc, omega, a=0, b=0, speciestype='', dipole=0):
    r'''Calculates the second virial coefficient using the
    comprehensive model in [1]_.

    .. math::
        TODO

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    omega : float
        Acentric factor for fluid, [-]
    a : float
        Fit parameter, optional
    b : float
        Fit parameter, optional
    Dipole : float
        dipole moment, optional, [Debye]

    Returns
    -------
    BVirial : float
        Second virial coefficient, [m^3/mol]

    Notes
    -----

    Examples
    --------
    Example from Perry's Handbook, 8E, p2-499. Matches to a decimal place.

    >>> BVirial_Tsonopoulos_Extended(430., 405.65, 11.28E6, 0.252608, a=0, b=0, speciestype='ketone', dipole=1.469)
    -9.679715056695323e-05

    References
    ----------
    .. [1] Tsonopoulos, C., and J. L. Heidman. "From the Virial to the Cubic
       Equation of State." Fluid Phase Equilibria 57, no. 3 (1990): 261-76.
       doi:10.1016/0378-3812(90)85126-U
    .. [2] Tsonopoulos, Constantine, and John H. Dymond. "Second Virial
       Coefficients of Normal Alkanes, Linear 1-Alkanols (and Water), Alkyl
       Ethers, and Their Mixtures." Fluid Phase Equilibria, International
       Workshop on Vapour-Liquid Equilibria and Related Properties in Binary
       and Ternary Mixtures of Ethers, Alkanes and Alkanols, 133, no. 1-2
       (June 1997): 11-34. doi:10.1016/S0378-3812(97)00058-7.
    '''
    Tr = T/Tc
    B0 = 0.1445 - 0.33/Tr - 0.1385/Tr**2 - 0.0121/Tr**3 - 0.000607/Tr**8
    B1 = 0.0637+0.331/Tr**2-0.423/Tr**3-0.008/Tr**8
    B2 = 1./Tr**6
    B3 = -1./Tr**8

    if a == 0 and b == 0 and speciestype != '':
        if speciestype == 'simple' or speciestype == 'normal':
            a, b = 0, 0
        elif speciestype == 'methyl alcohol':
            a, b = 0.0878, 0.0525
        elif speciestype == 'water':
            a, b = -0.0109, 0
        elif dipole != 0 and Tc != 0 and Pc != 0:
            dipole_r = 1E5*dipole**2*(Pc/101325.0)/Tc**2

            if (speciestype == 'ketone' or speciestype == 'aldehyde'
            or speciestype == 'alkyl nitrile' or speciestype == 'ether'
            or speciestype == 'carboxylic acid' or speciestype == 'ester'):
                a, b = -2.14E-4*dipole_r-4.308E-21*dipole_r**8, 0
            elif (speciestype == 'alkyl halide' or speciestype == 'mercaptan'
            or speciestype == 'sulfide' or speciestype == 'disulfide'):
                a, b = -2.188E-4*dipole_r**4-7.831E-21*dipole_r**8, 0

            elif speciestype == 'alkanol':
                a, b = 0.0878, 0.00908+0.0006957*dipole_r
    Br = B0 + omega*B1 + a*B2 + b*B3
    BVirial = Br*R*Tc/Pc
    return BVirial
