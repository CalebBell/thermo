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

__all__ = ['BVirial_Pitzer_Curl', 'BVirial_Abbott', 'BVirial_Tsonopoulos',
           'BVirial_Tsonopoulos_extended']

from scipy.misc import derivative
from thermo.utils import log
from thermo.utils import R


### Second Virial Coefficients


def BVirial_Pitzer_Curl(T, Tc, Pc, omega, order=0):
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
    order : int, optional
        Order of the calculation. 0 for the calculation of B itself; for 1/2/3, 
        the first/second/third derivative of B with respect to temperature; and  
        for -1/-2, the first/second indefinite integral of B with respect to 
        temperature. No other integrals or derivatives are implemented, and an 
        exception will be raised if any other order is given.

    Returns
    -------
    B : float
        Second virial coefficient in density form or its integral/derivative if
        specified, [m^3/mol or m^3/mol/K^order]

    Notes
    -----
    Analytical models for derivatives and integrals are available for orders
    -2, -1, 1, 2, and 3, all obtained with SymPy.

    For first temperature derivative of B:
    
    .. math::
        \frac{d B^{(0)}}{dT} = \frac{33 Tc}{100 T^{2}} + \frac{277 Tc^{2}}{1000 T^{3}} + \frac{363 Tc^{3}}{10000 T^{4}}

        \frac{d B^{(1)}}{dT} = - \frac{23 Tc}{50 T^{2}} + \frac{Tc^{2}}{T^{3}} + \frac{291 Tc^{3}}{1000 T^{4}} + \frac{73 Tc^{8}}{1250 T^{9}}

    For the second temperature derivative of B:
    
    .. math::
        \frac{d^2 B^{(0)}}{dT^2} = - \frac{3 Tc}{5000 T^{3}} \left(1100 + \frac{1385 Tc}{T} + \frac{242 Tc^{2}}{T^{2}}\right)

        \frac{d^2 B^{(1)}}{dT^2} = \frac{Tc}{T^{3}} \left(\frac{23}{25} - \frac{3 Tc}{T} - \frac{291 Tc^{2}}{250 T^{2}} - \frac{657 Tc^{7}}{1250 T^{7}}\right)

    For the third temperature derivative of B:
    
    .. math::
        \frac{d^3 B^{(0)}}{dT^3} = \frac{3 Tc}{500 T^{4}} \left(330 + \frac{554 Tc}{T} + \frac{121 Tc^{2}}{T^{2}}\right)

        \frac{d^3 B^{(1)}}{dT^3} = \frac{3 Tc}{T^{4}} \left(- \frac{23}{25} + \frac{4 Tc}{T} + \frac{97 Tc^{2}}{50 T^{2}} + \frac{219 Tc^{7}}{125 T^{7}}\right)
    
    For the first indefinite integral of B:
    
    .. math::
        \int{B^{(0)}} dT = \frac{289 T}{2000} - \frac{33 Tc}{100} \log{\left (T \right )} + \frac{1}{20000 T^{2}} \left(2770 T Tc^{2} + 121 Tc^{3}\right)
        
        \int{B^{(1)}} dT = \frac{73 T}{1000} + \frac{23 Tc}{50} \log{\left (T \right )} + \frac{1}{70000 T^{7}} \left(35000 T^{6} Tc^{2} + 3395 T^{5} Tc^{3} + 73 Tc^{8}\right)
    
    For the second indefinite integral of B:

    .. math::
        \int\int B^{(0)} dT dT = \frac{289 T^{2}}{4000} - \frac{33 T}{100} Tc \log{\left (T \right )} + \frac{33 T}{100} Tc + \frac{277 Tc^{2}}{2000} \log{\left (T \right )} - \frac{121 Tc^{3}}{20000 T}
    
        \int\int B^{(1)} dT dT = \frac{73 T^{2}}{2000} + \frac{23 T}{50} Tc \log{\left (T \right )} - \frac{23 T}{50} Tc + \frac{Tc^{2}}{2} \log{\left (T \right )} - \frac{1}{420000 T^{6}} \left(20370 T^{5} Tc^{3} + 73 Tc^{8}\right)

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
    if order == 0:
        B0 = 0.1445 - 0.33/Tr - 0.1385/Tr**2 - 0.0121/Tr**3
        B1 = 0.073 + 0.46/Tr - 0.5/Tr**2 - 0.097/Tr**3 - 0.0073/Tr**8
    elif order == 1:
        B0 = Tc*(3300*T**2 + 2770*T*Tc + 363*Tc**2)/(10000*T**4)
        B1 = Tc*(-2300*T**7 + 5000*T**6*Tc + 1455*T**5*Tc**2 + 292*Tc**7)/(5000*T**9)
    elif order == 2:
        B0 = -3*Tc*(1100*T**2 + 1385*T*Tc + 242*Tc**2)/(5000*T**5)
        B1 = Tc*(1150*T**7 - 3750*T**6*Tc - 1455*T**5*Tc**2 - 657*Tc**7)/(1250*T**10)
    elif order == 3:
        B0 = 3*Tc*(330*T**2 + 554*T*Tc + 121*Tc**2)/(500*T**6)
        B1 = 3*Tc*(-230*T**7 + 1000*T**6*Tc + 485*T**5*Tc**2 + 438*Tc**7)/(250*T**11)
    elif order == -1:
        B0 = 289*T/2000 - 33*Tc*log(T)/100 + (2770*T*Tc**2 + 121*Tc**3)/(20000*T**2)
        B1 = 73*T/1000 + 23*Tc*log(T)/50 + (35000*T**6*Tc**2 + 3395*T**5*Tc**3 + 73*Tc**8)/(70000*T**7)
    elif order == -2:
        B0 = 289*T**2/4000 - 33*T*Tc*log(T)/100 + 33*T*Tc/100 + 277*Tc**2*log(T)/2000 - 121*Tc**3/(20000*T)
        B1 = 73*T**2/2000 + 23*T*Tc*log(T)/50 - 23*T*Tc/50 + Tc**2*log(T)/2 - (20370*T**5*Tc**3 + 73*Tc**8)/(420000*T**6)
    else: 
        raise Exception('Only orders -2, -1, 0, 1, 2 and 3 are supported.')
    Br = B0 + omega*B1
    return Br*R*Tc/Pc


def BVirial_Abbott(T, Tc, Pc, omega, order=0):
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
    order : int, optional
        Order of the calculation. 0 for the calculation of B itself; for 1/2/3, 
        the first/second/third derivative of B with respect to temperature; and  
        for -1/-2, the first/second indefinite integral of B with respect to 
        temperature. No other integrals or derivatives are implemented, and an 
        exception will be raised if any other order is given.

    Returns
    -------
    B : float
        Second virial coefficient in density form or its integral/derivative if
        specified, [m^3/mol or m^3/mol/K^order]

    Notes
    -----
    Analytical models for derivatives and integrals are available for orders
    -2, -1, 1, 2, and 3, all obtained with SymPy.

    For first temperature derivative of B:
    
    .. math::
        \frac{d B^{(0)}}{dT} = \frac{0.6752}{T \left(\frac{T}{Tc}\right)^{1.6}}

        \frac{d B^{(1)}}{dT} = \frac{0.7224}{T \left(\frac{T}{Tc}\right)^{4.2}}

    For the second temperature derivative of B:
    
    .. math::
        \frac{d^2 B^{(0)}}{dT^2} = - \frac{1.75552}{T^{2} \left(\frac{T}{Tc}\right)^{1.6}}

        \frac{d^2 B^{(1)}}{dT^2} = - \frac{3.75648}{T^{2} \left(\frac{T}{Tc}\right)^{4.2}}

    For the third temperature derivative of B:
    
    .. math::
        \frac{d^3 B^{(0)}}{dT^3} = \frac{6.319872}{T^{3} \left(\frac{T}{Tc}\right)^{1.6}}

        \frac{d^3 B^{(1)}}{dT^3} = \frac{23.290176}{T^{3} \left(\frac{T}{Tc}\right)^{4.2}}
    
    For the first indefinite integral of B:
    
    .. math::
        \int{B^{(0)}} dT = 0.083 T + \frac{\frac{211}{300} Tc}{\left(\frac{T}{Tc}\right)^{0.6}}
        
        \int{B^{(1)}} dT = 0.139 T + \frac{0.05375 Tc}{\left(\frac{T}{Tc}\right)^{3.2}}
    
    For the second indefinite integral of B:

    .. math::
        \int\int B^{(0)} dT dT = 0.0415 T^{2} + \frac{211}{120} Tc^{2} \left(\frac{T}{Tc}\right)^{0.4}
    
        \int\int B^{(1)} dT dT = 0.0695 T^{2} - \frac{\frac{43}{1760} Tc^{2}}{\left(\frac{T}{Tc}\right)^{2.2}}
    
    Examples
    --------
    Example is from [1]_, p. 93, and matches the result exactly, for isobutane.

    >>> BVirial_Abbott(510., 425.2, 38E5, 0.193)
    -0.00020570178037383633

    References
    ----------
    .. [1] Smith, H. C. Van Ness Joseph M. Introduction to Chemical Engineering
       Thermodynamics 4E 1987.
    '''
    Tr = T/Tc
    if order == 0:
        B0 = 0.083 - 0.422/Tr**1.6
        B1 = 0.139 - 0.172/Tr**4.2
    elif order == 1:
        B0 = 0.6752*Tr**(-1.6)/T
        B1 = 0.7224*Tr**(-4.2)/T
    elif order == 2:
        B0 = -1.75552*Tr**(-1.6)/T**2
        B1 = -3.75648*Tr**(-4.2)/T**2
    elif order == 3:
        B0 = 6.319872*Tr**(-1.6)/T**3
        B1 = 23.290176*Tr**(-4.2)/T**3
    elif order == -1:
        B0 = 0.083*T + 211/300.*Tc*(Tr)**(-0.6)
        B1 = 0.139*T + 0.05375*Tc*Tr**(-3.2)
    elif order == -2:
        B0 = 0.0415*T**2 + 211/120.*Tc**2*Tr**0.4
        B1 = 0.0695*T**2 - 43/1760.*Tc**2*Tr**(-2.2)
    else: 
        raise Exception('Only orders -2, -1, 0, 1, 2 and 3 are supported.')
    Br = B0 + omega*B1
    return Br*R*Tc/Pc


def BVirial_Tsonopoulos(T, Tc, Pc, omega, order=0):
    r'''Calculates the second virial coefficient using the model in [1]_.

    .. math::
        B_r=B^{(0)}+\omega B^{(1)}

        B^{(0)}= 0.1445-0.330/T_r - 0.1385/T_r^2 - 0.0121/T_r^3 - 0.000607/T_r^8

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
    order : int, optional
        Order of the calculation. 0 for the calculation of B itself; for 1/2/3, 
        the first/second/third derivative of B with respect to temperature; and  
        for -1/-2, the first/second indefinite integral of B with respect to 
        temperature. No other integrals or derivatives are implemented, and an 
        exception will be raised if any other order is given.

    Returns
    -------
    B : float
        Second virial coefficient in density form or its integral/derivative if
        specified, [m^3/mol or m^3/mol/K^order]

    Notes
    -----
    A more complete expression is also available, in
    BVirial_Tsonopoulos_extended.

    Analytical models for derivatives and integrals are available for orders
    -2, -1, 1, 2, and 3, all obtained with SymPy.

    For first temperature derivative of B:
    
    .. math::
        \frac{d B^{(0)}}{dT} = \frac{33 Tc}{100 T^{2}} + \frac{277 Tc^{2}}{1000 T^{3}} + \frac{363 Tc^{3}}{10000 T^{4}} + \frac{607 Tc^{8}}{125000 T^{9}}

        \frac{d B^{(1)}}{dT} = - \frac{331 Tc^{2}}{500 T^{3}} + \frac{1269 Tc^{3}}{1000 T^{4}} + \frac{8 Tc^{8}}{125 T^{9}}

    For the second temperature derivative of B:
    
    .. math::
        \frac{d^2 B^{(0)}}{dT^2} = - \frac{3 Tc}{125000 T^{3}} \left(27500 + \frac{34625 Tc}{T} + \frac{6050 Tc^{2}}{T^{2}} + \frac{1821 Tc^{7}}{T^{7}}\right)

        \frac{d^2 B^{(1)}}{dT^2} = \frac{3 Tc^{2}}{500 T^{4}} \left(331 - \frac{846 Tc}{T} - \frac{96 Tc^{6}}{T^{6}}\right)

    For the third temperature derivative of B:
    
    .. math::
        \frac{d^3 B^{(0)}}{dT^3} = \frac{3 Tc}{12500 T^{4}} \left(8250 + \frac{13850 Tc}{T} + \frac{3025 Tc^{2}}{T^{2}} + \frac{1821 Tc^{7}}{T^{7}}\right)

        \frac{d^3 B^{(1)}}{dT^3} = \frac{3 Tc^{2}}{250 T^{5}} \left(-662 + \frac{2115 Tc}{T} + \frac{480 Tc^{6}}{T^{6}}\right)
    
    For the first indefinite integral of B:
    
    .. math::
        \int{B^{(0)}} dT = \frac{289 T}{2000} - \frac{33 Tc}{100} \log{\left (T \right )} + \frac{1}{7000000 T^{7}} \left(969500 T^{6} Tc^{2} + 42350 T^{5} Tc^{3} + 607 Tc^{8}\right)
        
        \int{B^{(1)}} dT = \frac{637 T}{10000} - \frac{1}{70000 T^{7}} \left(23170 T^{6} Tc^{2} - 14805 T^{5} Tc^{3} - 80 Tc^{8}\right)
        
    For the second indefinite integral of B:

    .. math::
        \int\int B^{(0)} dT dT = \frac{289 T^{2}}{4000} - \frac{33 T}{100} Tc \log{\left (T \right )} + \frac{33 T}{100} Tc + \frac{277 Tc^{2}}{2000} \log{\left (T \right )} - \frac{1}{42000000 T^{6}} \left(254100 T^{5} Tc^{3} + 607 Tc^{8}\right)
    
        \int\int B^{(1)} dT dT = \frac{637 T^{2}}{20000} - \frac{331 Tc^{2}}{1000} \log{\left (T \right )} - \frac{1}{210000 T^{6}} \left(44415 T^{5} Tc^{3} + 40 Tc^{8}\right)

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
    if order == 0:
        B0 = 0.1445 - 0.33/Tr - 0.1385/Tr**2 - 0.0121/Tr**3 - 0.000607/Tr**8
        B1 = 0.0637 + 0.331/Tr**2 - 0.423/Tr**3 - 0.008/Tr**8
    elif order == 1:
        B0 = 33*Tc/(100*T**2) + 277*Tc**2/(1000*T**3) + 363*Tc**3/(10000*T**4) + 607*Tc**8/(125000*T**9)
        B1 = -331*Tc**2/(500*T**3) + 1269*Tc**3/(1000*T**4) + 8*Tc**8/(125*T**9)
    elif order == 2:
        B0 = -3*Tc*(27500 + 34625*Tc/T + 6050*Tc**2/T**2 + 1821*Tc**7/T**7)/(125000*T**3)
        B1 = 3*Tc**2*(331 - 846*Tc/T - 96*Tc**6/T**6)/(500*T**4)
    elif order == 3:
        B0 = 3*Tc*(8250 + 13850*Tc/T + 3025*Tc**2/T**2 + 1821*Tc**7/T**7)/(12500*T**4)
        B1 = 3*Tc**2*(-662 + 2115*Tc/T + 480*Tc**6/T**6)/(250*T**5)
    elif order == -1:
        B0 = 289*T/2000. - 33*Tc*log(T)/100. + (969500*T**6*Tc**2 + 42350*T**5*Tc**3 + 607*Tc**8)/(7000000.*T**7)
        B1 = 637*T/10000. - (23170*T**6*Tc**2 - 14805*T**5*Tc**3 - 80*Tc**8)/(70000.*T**7)
    elif order == -2:
        B0 = 289*T**2/4000. - 33*T*Tc*log(T)/100. + 33*T*Tc/100. + 277*Tc**2*log(T)/2000. - (254100*T**5*Tc**3 + 607*Tc**8)/(42000000.*T**6)
        B1 = 637*T**2/20000. - 331*Tc**2*log(T)/1000. - (44415*T**5*Tc**3 + 40*Tc**8)/(210000.*T**6)
    else: 
        raise Exception('Only orders -2, -1, 0, 1, 2 and 3 are supported.')
    Br = (B0+omega*B1)
    return Br*R*Tc/Pc


def BVirial_Tsonopoulos_extended(T, Tc, Pc, omega, a=0, b=0, species_type='', 
                                 dipole=0, order=0):
    r'''Calculates the second virial coefficient using the
    comprehensive model in [1]_. See the notes for the calculation of `a` and
    `b`.

    .. math::
        \frac{BP_c}{RT_c} = B^{(0)} + \omega B^{(1)} + a B^{(2)} + b B^{(3)}
    
        B^{(0)}=0.1445-0.33/T_r-0.1385/T_r^2-0.0121/T_r^3

        B^{(1)} = 0.0637+0.331/T_r^2-0.423/T_r^3 -0.423/T_r^3 - 0.008/T_r^8
        
        B^{(2)} = 1/T_r^6

        B^{(3)} = -1/T_r^8

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
    a : float, optional
        Fit parameter, calculated based on species_type if a is not given and
        species_type matches on of the supported chemical classes.
    b : float, optional
        Fit parameter, calculated based on species_type if a is not given and
        species_type matches on of the supported chemical classes.
    species_type : str, optional
        One of .
    dipole : float
        dipole moment, optional, [Debye]
    order : int, optional
        Order of the calculation. 0 for the calculation of B itself; for 1/2/3, 
        the first/second/third derivative of B with respect to temperature; and  
        for -1/-2, the first/second indefinite integral of B with respect to 
        temperature. No other integrals or derivatives are implemented, and an 
        exception will be raised if any other order is given.

    Returns
    -------
    B : float
        Second virial coefficient in density form or its integral/derivative if
        specified, [m^3/mol or m^3/mol/K^order]

    Notes
    -----
    Analytical models for derivatives and integrals are available for orders
    -2, -1, 1, 2, and 3, all obtained with SymPy.


    To calculate `a` or `b`, the following rules are used:

    For 'simple' or 'normal' fluids:
    
    .. math::
        a = 0
        
        b = 0

    For 'ketone', 'aldehyde', 'alkyl nitrile', 'ether', 'carboxylic acid',
    or 'ester' types of chemicals:
    
    .. math::
        a = -2.14\times 10^{-4} \mu_r - 4.308 \times 10^{-21} (\mu_r)^8
        
        b = 0
    
    For 'alkyl halide', 'mercaptan', 'sulfide', or 'disulfide' types of 
    chemicals:

    .. math::
        a = -2.188\times 10^{-4} (\mu_r)^4 - 7.831 \times 10^{-21} (\mu_r)^8
        
        b = 0

    For 'alkanol' types of chemicals (except methanol):
    
    .. math::
        a = 0.0878
    
        b = 0.00908 + 0.0006957 \mu_r
        
    For methanol:
    
    .. math::
        a = 0.0878
        
        b = 0.0525
    
    For water:
    
    .. math::
        a = -0.0109
        
        b = 0
    
    If required, the form of dipole moment used in the calculation of some
    types of `a` and `b` values is as follows:
    
    .. math::
        \mu_r = 100000\frac{\mu^2(Pc/101325.0)}{Tc^2}


    For first temperature derivative of B:
    
    .. math::
        \frac{d B^{(0)}}{dT} = \frac{33 Tc}{100 T^{2}} + \frac{277 Tc^{2}}{1000 T^{3}} + \frac{363 Tc^{3}}{10000 T^{4}} + \frac{607 Tc^{8}}{125000 T^{9}}

        \frac{d B^{(1)}}{dT} = - \frac{331 Tc^{2}}{500 T^{3}} + \frac{1269 Tc^{3}}{1000 T^{4}} + \frac{8 Tc^{8}}{125 T^{9}}

        \frac{d B^{(2)}}{dT} = - \frac{6 Tc^{6}}{T^{7}}

        \frac{d B^{(3)}}{dT} = \frac{8 Tc^{8}}{T^{9}}

    For the second temperature derivative of B:
    
    .. math::
        \frac{d^2 B^{(0)}}{dT^2} = - \frac{3 Tc}{125000 T^{3}} \left(27500 + \frac{34625 Tc}{T} + \frac{6050 Tc^{2}}{T^{2}} + \frac{1821 Tc^{7}}{T^{7}}\right)

        \frac{d^2 B^{(1)}}{dT^2} = \frac{3 Tc^{2}}{500 T^{4}} \left(331 - \frac{846 Tc}{T} - \frac{96 Tc^{6}}{T^{6}}\right)

        \frac{d^2 B^{(2)}}{dT^2} = \frac{42 Tc^{6}}{T^{8}}

        \frac{d^2 B^{(3)}}{dT^2} = - \frac{72 Tc^{8}}{T^{10}}

    For the third temperature derivative of B:
    
    .. math::
        \frac{d^3 B^{(0)}}{dT^3} = \frac{3 Tc}{12500 T^{4}} \left(8250 + \frac{13850 Tc}{T} + \frac{3025 Tc^{2}}{T^{2}} + \frac{1821 Tc^{7}}{T^{7}}\right)

        \frac{d^3 B^{(1)}}{dT^3} = \frac{3 Tc^{2}}{250 T^{5}} \left(-662 + \frac{2115 Tc}{T} + \frac{480 Tc^{6}}{T^{6}}\right)

        \frac{d^3 B^{(2)}}{dT^3} = - \frac{336 Tc^{6}}{T^{9}}

        \frac{d^3 B^{(3)}}{dT^3} = \frac{720 Tc^{8}}{T^{11}}

    For the first indefinite integral of B:
    
    .. math::
        \int{B^{(0)}} dT = \frac{289 T}{2000} - \frac{33 Tc}{100} \log{\left (T \right )} + \frac{1}{7000000 T^{7}} \left(969500 T^{6} Tc^{2} + 42350 T^{5} Tc^{3} + 607 Tc^{8}\right)
        
        \int{B^{(1)}} dT = \frac{637 T}{10000} - \frac{1}{70000 T^{7}} \left(23170 T^{6} Tc^{2} - 14805 T^{5} Tc^{3} - 80 Tc^{8}\right)

        \int{B^{(2)}} dT = - \frac{Tc^{6}}{5 T^{5}}

        \int{B^{(3)}} dT = \frac{Tc^{8}}{7 T^{7}}

    For the second indefinite integral of B:

    .. math::
        \int\int B^{(0)} dT dT = \frac{289 T^{2}}{4000} - \frac{33 T}{100} Tc \log{\left (T \right )} + \frac{33 T}{100} Tc + \frac{277 Tc^{2}}{2000} \log{\left (T \right )} - \frac{1}{42000000 T^{6}} \left(254100 T^{5} Tc^{3} + 607 Tc^{8}\right)
    
        \int\int B^{(1)} dT dT = \frac{637 T^{2}}{20000} - \frac{331 Tc^{2}}{1000} \log{\left (T \right )} - \frac{1}{210000 T^{6}} \left(44415 T^{5} Tc^{3} + 40 Tc^{8}\right)

        \int\int B^{(2)} dT dT = \frac{Tc^{6}}{20 T^{4}}
        
        \int\int B^{(3)} dT dT = - \frac{Tc^{8}}{42 T^{6}}
        
    Examples
    --------
    Example from Perry's Handbook, 8E, p2-499. Matches to a decimal place.

    >>> BVirial_Tsonopoulos_extended(430., 405.65, 11.28E6, 0.252608, a=0, b=0, species_type='ketone', dipole=1.469)
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
    if order == 0:
        B0 = 0.1445 - 0.33/Tr - 0.1385/Tr**2 - 0.0121/Tr**3 - 0.000607/Tr**8
        B1 = 0.0637 + 0.331/Tr**2 - 0.423/Tr**3 - 0.008/Tr**8
        B2 = 1./Tr**6
        B3 = -1./Tr**8
    elif order == 1:
        B0 = 33*Tc/(100*T**2) + 277*Tc**2/(1000*T**3) + 363*Tc**3/(10000*T**4) + 607*Tc**8/(125000*T**9)
        B1 = -331*Tc**2/(500*T**3) + 1269*Tc**3/(1000*T**4) + 8*Tc**8/(125*T**9)
        B2 = -6.0*Tc**6/T**7
        B3 = 8.0*Tc**8/T**9
    elif order == 2:
        B0 = -3*Tc*(27500 + 34625*Tc/T + 6050*Tc**2/T**2 + 1821*Tc**7/T**7)/(125000*T**3)
        B1 = 3*Tc**2*(331 - 846*Tc/T - 96*Tc**6/T**6)/(500*T**4)
        B2 = 42.0*Tc**6/T**8
        B3 = -72.0*Tc**8/T**10
    elif order == 3:
        B0 = 3*Tc*(8250 + 13850*Tc/T + 3025*Tc**2/T**2 + 1821*Tc**7/T**7)/(12500*T**4)
        B1 = 3*Tc**2*(-662 + 2115*Tc/T + 480*Tc**6/T**6)/(250*T**5)
        B2 = -336.0*Tc**6/T**9
        B3 = 720.0*Tc**8/T**11
    elif order == -1:
        B0 = 289*T/2000. - 33*Tc*log(T)/100. + (969500*T**6*Tc**2 + 42350*T**5*Tc**3 + 607*Tc**8)/(7000000.*T**7)
        B1 = 637*T/10000. - (23170*T**6*Tc**2 - 14805*T**5*Tc**3 - 80*Tc**8)/(70000.*T**7)
        B2 = -Tc**6/(5*T**5)
        B3 = Tc**8/(7*T**7)
    elif order == -2:
        B0 = 289*T**2/4000. - 33*T*Tc*log(T)/100. + 33*T*Tc/100. + 277*Tc**2*log(T)/2000. - (254100*T**5*Tc**3 + 607*Tc**8)/(42000000.*T**6)
        B1 = 637*T**2/20000. - 331*Tc**2*log(T)/1000. - (44415*T**5*Tc**3 + 40*Tc**8)/(210000.*T**6)
        B2 = Tc**6/(20*T**4)
        B3 = -Tc**8/(42*T**6)
    else: 
        raise Exception('Only orders -2, -1, 0, 1, 2 and 3 are supported.')
    if a == 0 and b == 0 and species_type != '':
        if species_type == 'simple' or species_type == 'normal':
            a, b = 0, 0
        elif species_type == 'methyl alcohol':
            a, b = 0.0878, 0.0525
        elif species_type == 'water':
            a, b = -0.0109, 0
        elif dipole != 0 and Tc != 0 and Pc != 0:
            dipole_r = 1E5*dipole**2*(Pc/101325.0)/Tc**2

            if (species_type == 'ketone' or species_type == 'aldehyde'
            or species_type == 'alkyl nitrile' or species_type == 'ether'
            or species_type == 'carboxylic acid' or species_type == 'ester'):
                a, b = -2.14E-4*dipole_r-4.308E-21*dipole_r**8, 0
            elif (species_type == 'alkyl halide' or species_type == 'mercaptan'
            or species_type == 'sulfide' or species_type == 'disulfide'):
                a, b = -2.188E-4*dipole_r**4-7.831E-21*dipole_r**8, 0

            elif species_type == 'alkanol':
                a, b = 0.0878, 0.00908+0.0006957*dipole_r
    Br = B0 + omega*B1 + a*B2 + b*B3
    return Br*R*Tc/Pc
