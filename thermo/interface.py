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

__all__ = ['Mulero_Cachadina_data', 'Jasper_Lange_data', 'Somayajulu_data', 
           'VDI_PPDS_11', 'Somayajulu_data_2', 'REFPROP', 'Somayajulu', 'Jasper', 
           'Brock_Bird', 'Pitzer', 'Sastri_Rao', 'Zuo_Stenby', 
           'Hakim_Steinberg_Stiel', 'Miqueu', 'Aleem', 'surface_tension_methods', 
           'SurfaceTension', 'Winterfeld_Scriven_Davis', 'Diguilio_Teja', 
           'surface_tension_mixture_methods', 'SurfaceTensionMixture']

import os
import pandas as pd
from thermo.utils import log, exp
from thermo.utils import mixing_simple, none_and_length_check, Vm_to_rho
from thermo.utils import N_A, k
from thermo.utils import TDependentProperty, MixtureProperty
from thermo.miscdata import _VDISaturationDict, VDI_tabular_data

folder = os.path.join(os.path.dirname(__file__), 'Interface')


Mulero_Cachadina_data = pd.read_csv(os.path.join(folder,
                        'MuleroCachadinaParameters.tsv'), sep='\t', index_col=0)
_Mulero_Cachadina_data_values = Mulero_Cachadina_data.values

Jasper_Lange_data = pd.read_csv(os.path.join(folder, 'Jasper-Lange.tsv'),
                      sep='\t', index_col=0)
_Jasper_Lange_data_values = Jasper_Lange_data.values

Somayajulu_data = pd.read_csv(os.path.join(folder, 'Somayajulu.tsv'),
                      sep='\t', index_col=0)
_Somayajulu_data_values = Somayajulu_data.values

Somayajulu_data_2 = pd.read_csv(os.path.join(folder, 'SomayajuluRevised.tsv'),
                      sep='\t', index_col=0)
_Somayajulu_data_2_values = Somayajulu_data_2.values

VDI_PPDS_11 = pd.read_csv(os.path.join(folder, 'VDI PPDS surface tensions.tsv'),
                          sep='\t', index_col=0)
_VDI_PPDS_11_values = VDI_PPDS_11.values


### Regressed coefficient-based functions

def REFPROP(T, Tc, sigma0, n0, sigma1=0, n1=0, sigma2=0, n2=0):
    r'''Calculates air-liquid surface tension  using the REFPROP [1]_
    regression-based method. Relatively recent, and most accurate.

    .. math::
        \sigma(T)=\sigma_0\left(1-\frac{T}{T_c}\right)^{n_0}+
        \sigma_1\left(1-\frac{T}{T_c}\right)^{n_1}+
        \sigma_2\left(1-\frac{T}{T_c}\right)^{n_2}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    sigma0 : float
        First emperical coefficient of a fluid
    n0 : float
        First emperical exponent of a fluid
    sigma1 : float, optional
        Second emperical coefficient of a fluid.
    n1 : float, optional
        Second emperical exponent of a fluid.
    sigma1 : float, optional
        Third emperical coefficient of a fluid.
    n2 : float, optional
        Third emperical exponent of a fluid.

    Returns
    -------
    sigma : float
        Liquid surface tension, N/m

    Notes
    -----
    Function as implemented in [1]_. No example necessary; results match
    literature values perfectly.
    Form of function returns imaginary results when T > Tc; None is returned
    if this is the case.


    Examples
    --------
    Parameters for water at 298.15 K

    >>> REFPROP(298.15, 647.096, -0.1306, 2.471, 0.2151, 1.233)
    0.07205503890847453

    References
    ----------
    .. [1] Diky, Vladimir, Robert D. Chirico, Chris D. Muzny, Andrei F.
       Kazakov, Kenneth Kroenlein, Joseph W. Magee, Ilmutdin Abdulagatov, and
       Michael Frenkel. "ThermoData Engine (TDE): Software Implementation of
       the Dynamic Data Evaluation Concept." Journal of Chemical Information
       and Modeling 53, no. 12 (2013): 3418-30. doi:10.1021/ci4005699.
    '''
    Tr = T/Tc
    sigma = sigma0*(1.-Tr)**n0 + sigma1*(1.-Tr)**n1 + sigma2*(1.-Tr)**n2
    return sigma


def Somayajulu(T, Tc, A, B, C):
    r'''Calculates air-water surface tension  using the [1]_
    emperical (parameter-regressed) method. Well regressed, no recent data.

    .. math::
        \sigma=aX^{5/4}+bX^{9/4}+cX^{13/4}
        X=(T_c-T)/T_c

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    A : float
        Regression parameter
    B : float
        Regression parameter
    C : float
        Regression parameter

    Returns
    -------
    sigma : float
        Liquid surface tension, N/m

    Notes
    -----
    Presently untested, but matches expected values. Internal units are mN/m.
    Form of function returns imaginary results when T > Tc; None is returned
    if this is the case. Function is claimed valid from the triple to the
    critical point. Results can be evaluated beneath the triple point.

    Examples
    --------
    Water at 300 K

    >>> Somayajulu(300, 647.126, 232.713514, -140.18645, -4.890098)
    0.07166386387996757

    References
    ----------
    .. [1] Somayajulu, G. R. "A Generalized Equation for Surface Tension from
       the Triple Point to the Critical Point." International Journal of
       Thermophysics 9, no. 4 (July 1988): 559-66. doi:10.1007/BF00503154.
    '''
    X = (Tc-T)/Tc
    sigma = (A*X**1.25 + B*X**2.25 + C*X**3.25)/1000.
    return sigma


def Jasper(T, a, b):
    r'''Calculates surface tension of a fluid given two parameters, a linear
    fit in Celcius from [1]_ with data reprinted in [2]_.

    .. math::
        \sigma = a - bT

    Parameters
    ----------
    T : float
        Temperature of fluid, [K]
    a : float
        Parameter for equation. Chemical specific.
    b : float
        Parameter for equation. Chemical specific.

    Returns
    -------
    sigma: float
        Surface tension [N/m]

    Notes
    -----
    Internal units are mN/m, and degrees Celcius.
    This function has been checked against several references.

    Examples
    --------
    >>> Jasper(298.15, 24, 0.0773)
    0.0220675

    References
    ----------
    .. [1] Jasper, Joseph J. "The Surface Tension of Pure Liquid Compounds."
       Journal of Physical and Chemical Reference Data 1, no. 4
       (October 1, 1972): 841-1010. doi:10.1063/1.3253106.
    .. [2] Speight, James. Lange's Handbook of Chemistry. 16 edition.
       McGraw-Hill Professional, 2005.
    '''
    sigma = (a - b*(T-273.15))/1000
    return sigma


### CSP methods


def Brock_Bird(T, Tb, Tc, Pc):
    r'''Calculates air-water surface tension  using the [1]_
    emperical method. Old and tested.

    .. math::
        \sigma = P_c^{2/3}T_c^{1/3}Q(1-T_r)^{11/9}

        Q = 0.1196 \left[ 1 + \frac{T_{br}\ln (P_c/1.01325)}{1-T_{br}}\right]-0.279

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tb : float
        Boiling temperature of the fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]

    Returns
    -------
    sigma : float
        Liquid surface tension, N/m

    Notes
    -----
    Numerous arrangements of this equation are available.
    This is DIPPR Procedure 7A: Method for the Surface Tension of Pure,
    Nonpolar, Nonhydrocarbon Liquids
    The exact equation is not in the original paper.
    If the equation yields a negative result, return None.

    Examples
    --------
    p-dichloribenzene at 412.15 K, from DIPPR; value differs due to a slight
    difference in method.

    >>> Brock_Bird(412.15, 447.3, 685, 3.952E6)
    0.02208448325192495

    Chlorobenzene from Poling, as compared with a % error value at 293 K.

    >>> Brock_Bird(293.15, 404.75, 633.0, 4530000.0)
    0.032985686413713036

    References
    ----------
    .. [1] Brock, James R., and R. Byron Bird. "Surface Tension and the
       Principle of Corresponding States." AIChE Journal 1, no. 2
       (June 1, 1955): 174-77. doi:10.1002/aic.690010208
    '''
    Tbr = Tb/Tc
    Tr = T/Tc
    Pc = Pc/1E5  # Convert to bar
    Q = 0.1196*(1 + Tbr*log(Pc/1.01325)/(1-Tbr))-0.279
    sigma = (Pc)**(2/3.)*Tc**(1/3.)*Q*(1-Tr)**(11/9.)
    sigma = sigma/1000  # convert to N/m
    return sigma


def Pitzer(T, Tc, Pc, omega):
    r'''Calculates air-water surface tension using the correlation derived
    by [1]_ from the works of [2]_ and [3]_. Based on critical property CSP
    methods.

    .. math::
        \sigma = P_c^{2/3}T_c^{1/3}\frac{1.86 + 1.18\omega}{19.05}
        \left[ \frac{3.75 + 0.91 \omega}{0.291 - 0.08 \omega}\right]^{2/3} (1-T_r)^{11/9}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]
    omega : float
        Acentric factor for fluid, [-]

    Returns
    -------
    sigma : float
        Liquid surface tension, N/m

    Notes
    -----
    The source of this equation has not been reviewed.
    Internal units of presure are bar, surface tension of mN/m.

    Examples
    --------
    Chlorobenzene from Poling, as compared with a % error value at 293 K.

    >>> Pitzer(293., 633.0, 4530000.0, 0.249)
    0.03458453513446387

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    .. [2] Curl, R. F., and Kenneth Pitzer. "Volumetric and Thermodynamic
       Properties of Fluids-Enthalpy, Free Energy, and Entropy." Industrial &
       Engineering Chemistry 50, no. 2 (February 1, 1958): 265-74.
       doi:10.1021/ie50578a047
    .. [3] Pitzer, K. S.: Thermodynamics, 3d ed., New York, McGraw-Hill,
       1995, p. 521.
    '''
    Tr = T/Tc
    Pc = Pc/1E5  # Convert to bar
    sigma = Pc**(2/3.0)*Tc**(1/3.0)*(1.86+1.18*omega)/19.05 * (
        (3.75+0.91*omega)/(0.291-0.08*omega))**(2/3.0)*(1-Tr)**(11/9.0)
    sigma = sigma/1000  # N/m, please
    return sigma


def Sastri_Rao(T, Tb, Tc, Pc, chemicaltype=None):
    r'''Calculates air-water surface tension using the correlation derived by
    [1]_ based on critical property CSP methods and chemical classes.

    .. math::
        \sigma = K P_c^xT_b^y T_c^z\left[\frac{1-T_r}{1-T_{br}}\right]^m

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tb : float
        Boiling temperature of the fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]

    Returns
    -------
    sigma : float
        Liquid surface tension, N/m

    Notes
    -----
    The source of this equation has not been reviewed.
    Internal units of presure are bar, surface tension of mN/m.

    Examples
    --------
    Chlorobenzene from Poling, as compared with a % error value at 293 K.

    >>> Sastri_Rao(293.15, 404.75, 633.0, 4530000.0)
    0.03234567739694441

    References
    ----------
    .. [1] Sastri, S. R. S., and K. K. Rao. "A Simple Method to Predict
       Surface Tension of Organic Liquids." The Chemical Engineering Journal
       and the Biochemical Engineering Journal 59, no. 2 (October 1995): 181-86.
       doi:10.1016/0923-0467(94)02946-6.
    '''
    if chemicaltype == 'alcohol':
        k, x, y, z, m = 2.28, 0.25, 0.175, 0, 0.8
    elif chemicaltype == 'acid':
        k, x, y, z, m = 0.125, 0.50, -1.5, 1.85, 11/9.0
    else:
        k, x, y, z, m = 0.158, 0.50, -1.5, 1.85, 11/9.0
    Tr = T/Tc
    Tbr = Tb/Tc
    Pc = Pc/1E5  # Convert to bar
    sigma = k*Pc**x*Tb**y*Tc**z*((1 - Tr)/(1 - Tbr))**m
    sigma = sigma/1000  # N/m
    return sigma


def Zuo_Stenby(T, Tc, Pc, omega):
    r'''Calculates air-water surface tension using the reference fluids
    methods of [1]_.

    .. math::
        \sigma^{(1)} = 40.520(1-T_r)^{1.287}
        \sigma^{(2)} = 52.095(1-T_r)^{1.21548}
        \sigma_r = \sigma_r^{(1)}+ \frac{\omega - \omega^{(1)}}
        {\omega^{(2)}-\omega^{(1)}} (\sigma_r^{(2)}-\sigma_r^{(1)})
        \sigma = T_c^{1/3}P_c^{2/3}[\exp{(\sigma_r)} -1]

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]
    omega : float
        Acentric factor for fluid, [-]

    Returns
    -------
    sigma : float
        Liquid surface tension, N/m

    Notes
    -----
    Presently untested. Have not personally checked the sources.
    I strongly believe it is broken.
    The reference values for methane and n-octane are from the DIPPR database.

    Examples
    --------
    Chlorobenzene

    >>> Zuo_Stenby(293., 633.0, 4530000.0, 0.249)
    0.03345569011871088

    References
    ----------
    .. [1] Zuo, You-Xiang, and Erling H. Stenby. "Corresponding-States and
       Parachor Models for the Calculation of Interfacial Tensions." The
       Canadian Journal of Chemical Engineering 75, no. 6 (December 1, 1997):
       1130-37. doi:10.1002/cjce.5450750617
    '''
    Tc_1, Pc_1, omega_1 = 190.56, 4599000.0/1E5, 0.012
    Tc_2, Pc_2, omega_2 = 568.7, 2490000.0/1E5, 0.4
    Pc = Pc/1E5

    def ST_r(ST, Tc, Pc):
        return log(1 + ST/(Tc**(1/3.0)*Pc**(2/3.0)))

    ST_1 = 40.520*(1 - T/Tc)**1.287  # Methane
    ST_2 = 52.095*(1 - T/Tc)**1.21548  # n-octane

    ST_r_1, ST_r_2 = ST_r(ST_1, Tc_1, Pc_1), ST_r(ST_2, Tc_2, Pc_2)

    sigma_r = ST_r_1 + (omega-omega_1)/(omega_2 - omega_1)*(ST_r_2-ST_r_1)
    sigma = Tc**(1/3.0)*Pc**(2/3.0)*(exp(sigma_r)-1)
    sigma = sigma/1000  # N/m, please
    return sigma


def Hakim_Steinberg_Stiel(T, Tc, Pc, omega, StielPolar=0):
    r'''Calculates air-water surface tension using the reference fluids methods
    of [1]_.

    .. math::
        \sigma = 4.60104\times 10^{-7} P_c^{2/3}T_c^{1/3}Q_p \left(\frac{1-T_r}{0.4}\right)^m

        Q_p = 0.1574+0.359\omega-1.769\chi-13.69\chi^2-0.51\omega^2+1.298\omega\chi

        m = 1.21+0.5385\omega-14.61\chi-32.07\chi^2-1.65\omega^2+22.03\omega\chi

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]
    omega : float
        Acentric factor for fluid, [-]
    StielPolar : float, optional
        Stiel Polar Factor, [-]

    Returns
    -------
    sigma : float
        Liquid surface tension, N/m

    Notes
    -----
    Original equation for m and Q are used. Internal units are atm and mN/m.

    Examples
    --------
    1-butanol, as compared to value in CRC Handbook of 0.02493.

    >>> Hakim_Steinberg_Stiel(298.15, 563.0, 4414000.0, 0.59, StielPolar=-0.07872)
    0.021907902575190447

    References
    ----------
    .. [1] Hakim, D. I., David Steinberg, and L. I. Stiel. "Generalized
       Relationship for the Surface Tension of Polar Fluids." Industrial &
       Engineering Chemistry Fundamentals 10, no. 1 (February 1, 1971): 174-75.
       doi:10.1021/i160037a032.
    '''
    Q = (0.1574 + 0.359*omega - 1.769*StielPolar - 13.69*StielPolar**2
        - 0.510*omega**2 + 1.298*StielPolar*omega)
    m = (1.210 + 0.5385*omega - 14.61*StielPolar - 32.07*StielPolar**2
        - 1.656*omega**2 + 22.03*StielPolar*omega)
    Tr = T/Tc
    Pc = Pc/101325.
    sigma = Pc**(2/3.)*Tc**(1/3.)*Q*((1 - Tr)/0.4)**m
    sigma = sigma/1000.  # convert to N/m
    return sigma


def Miqueu(T, Tc, Vc, omega):
    r'''Calculates air-water surface tension using the methods of [1]_.

    .. math::
        \sigma = k T_c \left( \frac{N_a}{V_c}\right)^{2/3}
        (4.35 + 4.14 \omega)t^{1.26}(1+0.19t^{0.5} - 0.487t)

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Vc : float
        Critical volume of fluid [m^3/mol]
    omega : float
        Acentric factor for fluid, [-]

    Returns
    -------
    sigma : float
        Liquid surface tension, N/m

    Notes
    -----
    Uses Avogadro's constant and the Boltsman constant.
    Internal units of volume are mL/mol and mN/m. However, either a typo
    is in the article or author's work, or my value of k is off by 10; this is
    corrected nonetheless.
    Created with 31 normal fluids, none polar or hydrogen bonded. Has an
    AARD of 3.5%.

    Examples
    --------
    Bromotrifluoromethane, 2.45 mN/m

    >>> Miqueu(300., 340.1, 0.000199, 0.1687)
    0.003474099603581931

    References
    ----------
    .. [1] Miqueu, C, D Broseta, J Satherley, B Mendiboure, J Lachaise, and
       A Graciaa. "An Extended Scaled Equation for the Temperature Dependence
       of the Surface Tension of Pure Compounds Inferred from an Analysis of
       Experimental Data." Fluid Phase Equilibria 172, no. 2 (July 5, 2000):
       169-82. doi:10.1016/S0378-3812(00)00384-8.
    '''
    Vc = Vc*1E6
    t = 1.-T/Tc
    sigma = k*Tc*(N_A/Vc)**(2/3.)*(4.35 + 4.14*omega)*t**1.26*(1+0.19*t**0.5 - 0.25*t)*10000
    return sigma

    
def Aleem(T, MW, Tb, rhol, Hvap_Tb, Cpl):
    r'''Calculates vapor-liquid surface tension using the correlation derived by
    [1]_ based on critical property CSP methods.

    .. math::
        \sigma = \phi \frac{MW^{1/3}} {6N_A^{1/3}}\rho_l^{2/3}\left[H_{vap}
        + C_{p,l}(T_b-T)\right]

        \phi = 1 - 0.0047MW + 6.8\times 10^{-6} MW^2
            
    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    MW : float
        Molecular weight [g/mol]
    Tb : float
        Boiling temperature of the fluid [K]
    rhol : float
        Liquid density at T and P [kg/m^3]
    Hvap_Tb : float
        Mass enthalpy of vaporization at the normal boiling point [kg/m^3]
    Cpl : float
        Liquid heat capacity of the chemical at T [J/kg/K]

    Returns
    -------
    sigma : float
        Liquid-vapor surface tension [N/m]

    Notes
    -----
    Internal units of molecuar weight are kg/mol. This model is dimensionally
    consistent.
    
    This model does not use the critical temperature. After it predicts a 
    surface tension of 0 at a sufficiently high temperature, it returns 
    negative results. The temperature at which this occurs (the "predicted"
    critical temperature) can be calculated as follows:
        
    .. math::
        \sigma = 0 \to T_{c,predicted} \text{ at } T_b + \frac{H_{vap}}{Cp_l}
    
    Because of its dependence on density, it has the potential to model the 
    effect of pressure on surface tension.
    
    Claims AAD of 4.3%. Developed for normal alkanes. Total of 472 data points. 
    Behaves worse for higher alkanes. Behaves very poorly overall.
    
    Examples
    --------
    Methane at 90 K
    
    >>> Aleem(T=90, MW=16.04246, Tb=111.6, rhol=458.7, Hvap_Tb=510870.,
    ... Cpl=2465.)
    0.01669970221165325

    References
    ----------
    .. [1] Aleem, W., N. Mellon, S. Sufian, M. I. A. Mutalib, and D. Subbarao.
       "A Model for the Estimation of Surface Tension of Pure Hydrocarbon 
       Liquids." Petroleum Science and Technology 33, no. 23-24 (December 17, 
       2015): 1908-15. doi:10.1080/10916466.2015.1110593.
    '''
    MW = MW/1000. # Use kg/mol for consistency with the other units
    sphericity = 1. - 0.0047*MW + 6.8E-6*MW*MW
    return sphericity*MW**(1/3.)/(6.*N_A**(1/3.))*rhol**(2/3.)*(Hvap_Tb + Cpl*(Tb-T))
    
    
STREFPROP = 'REFPROP'
SUPERCRITICAL = 'SUPERCRITICAL'
SOMAYAJULU2 = 'SOMAYAJULU2'
SOMAYAJULU = 'SOMAYAJULU'
VDI_TABULAR = 'VDI_TABULAR'
JASPER = 'JASPER'
MIQUEU = 'MIQUEU'
BROCK_BIRD = 'BROCK_BIRD'
SASTRI_RAO = 'SASTRI_RAO'
PITZER = 'PITZER'
ZUO_STENBY = 'ZUO_STENBY'
HAKIM_STEINBERG_STIEL = 'HAKIM_STEINBERG_STIEL'
ALEEM = 'Aleem'
VDI_PPDS = 'VDI_PPDS'
NONE = 'NONE'


surface_tension_methods = [STREFPROP, SOMAYAJULU2, SOMAYAJULU, VDI_PPDS, VDI_TABULAR,
                           JASPER, MIQUEU, BROCK_BIRD, SASTRI_RAO, PITZER,
                           ZUO_STENBY, ALEEM]
'''Holds all methods available for the SurfaceTension class, for use in
iterating over them.'''


class SurfaceTension(TDependentProperty):
    '''Class for dealing with surface tension as a function of temperature.
    Consists of three coefficient-based methods and four data sources, one
    source of tabular information, and five corresponding-states estimators.

    Parameters
    ----------
    Tb : float, optional
        Boiling point, [K]
    MW : float, optional
        Molecular weight, [g/mol]
    Tc : float, optional
        Critical temperature, [K]
    Pc : float, optional
        Critical pressure, [Pa]
    Vc : float, optional
        Critical volume, [m^3/mol]
    Zc : float, optional
        Critical compressibility
    omega : float, optional
        Acentric factor, [-]
    StielPolar : float, optional
        Stiel polar factor
    Hvap_Tb : float
        Mass enthalpy of vaporization at the normal boiling point [kg/m^3]
    CASRN : str, optional
        The CAS number of the chemical
    Vml : float or callable, optional
        Liquid molar volume at a given temperature and pressure or callable
        for the same, [m^3/mol]
    Cpl : float or callable, optional
        Mass heat capacity of the fluid at a pressure and temperature or 
        or callable for the same, [J/kg/K]

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`surface_tension_methods`.

    **STREFPROP**:
        The REFPROP coefficient-based method, documented in the function
        :obj:`REFPROP` for 115 fluids from [5]_.
    **SOMAYAJULU and SOMAYAJULU2**:
        The Somayajulu coefficient-based method,
        documented in the function :obj:`Somayajulu`. Both methods have data
        for 64 fluids. The first data set if from [1]_, and the second
        from [2]_. The later, revised coefficients should be used prefered.
    **JASPER**:
        Fit with a single temperature coefficient from Jaspen (1972)
        as documented in the function :obj:`Jasper`. Data for 522 fluids is
        available, as shown in [4]_ but originally in [3]_.
    **BROCK_BIRD**:
        CSP method documented in :obj:`Brock_Bird`.
        Most popular estimation method; from 1955.
    **SASTRI_RAO**:
        CSP method documented in :obj:`Sastri_Rao`.
        Second most popular estimation method; from 1995.
    **PITZER**:
        CSP method documented in :obj:`Pitzer`; from 1958.
    **ZUO_STENBY**:
        CSP method documented in :obj:`Zuo_Stenby`; from 1997.
    **MIQUEU**:
        CSP method documented in :obj:`Miqueu`.
    **ALEEM**:
        CSP method documented in :obj:`Aleem`.
    **VDI_TABULAR**:
        Tabular data in [6]_ along the saturation curve; interpolation is as
        set by the user or the default.

    See Also
    --------
    REFPROP
    Somayajulu
    Jasper
    Brock_Bird
    Sastri_Rao
    Pitzer
    Zuo_Stenby
    Miqueu
    Aleem

    References
    ----------
    .. [1] Somayajulu, G. R. "A Generalized Equation for Surface Tension from
       the Triple Point to the Critical Point." International Journal of
       Thermophysics 9, no. 4 (July 1988): 559-66. doi:10.1007/BF00503154.
    .. [2] Mulero, A., M. I. Parra, and I. Cachadina. "The Somayajulu
       Correlation for the Surface Tension Revisited." Fluid Phase
       Equilibria 339 (February 15, 2013): 81-88.
       doi:10.1016/j.fluid.2012.11.038.
    .. [3] Jasper, Joseph J. "The Surface Tension of Pure Liquid Compounds."
       Journal of Physical and Chemical Reference Data 1, no. 4
       (October 1, 1972): 841-1010. doi:10.1063/1.3253106.
    .. [4] Speight, James. Lange's Handbook of Chemistry. 16 edition.
       McGraw-Hill Professional, 2005.
    .. [5] Mulero, A., I. Cachadiña, and M. I. Parra. “Recommended
       Correlations for the Surface Tension of Common Fluids.” Journal of
       Physical and Chemical Reference Data 41, no. 4 (December 1, 2012):
       043105. doi:10.1063/1.4768782.
    .. [6] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    name = 'Surface tension'
    units = 'N/m'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default; values below 0 will be obtained
    at high temperatures.'''
    property_min = 0
    '''Mimimum valid value of surface tension. This occurs at the critical
    point exactly.'''
    property_max = 4.0
    '''Maximum valid value of surface tension. Set to roughly twice that of 
    cobalt at its melting point.'''

    ranked_methods = [STREFPROP, SOMAYAJULU2, SOMAYAJULU, VDI_PPDS, VDI_TABULAR,
                      JASPER, MIQUEU, BROCK_BIRD, SASTRI_RAO, PITZER,
                      ZUO_STENBY, ALEEM]
    '''Default rankings of the available methods.'''

    def __init__(self, MW=None, Tb=None, Tc=None, Pc=None, Vc=None, Zc=None, omega=None,
                 StielPolar=None, Hvap_Tb=None, CASRN='', Vml=None, Cpl=None):
        self.MW = MW
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.Vc = Vc
        self.Zc = Zc
        self.omega = omega
        self.StielPolar = StielPolar
        self.Hvap_Tb = Hvap_Tb
        self.CASRN = CASRN
        self.Vml = Vml
        self.Cpl = Cpl

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        surface tension under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        surface tension above.'''

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
        at a specific temperature; set by `T_dependent_property`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `T_dependent_property`.'''

        self.all_methods = set()
        '''Set of all methods available for a given CASRN and properties;
        filled by :obj:`load_all_methods`.'''

        self.load_all_methods()

    def load_all_methods(self):
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
        if self.CASRN in Mulero_Cachadina_data.index:
            methods.append(STREFPROP)
            _, sigma0, n0, sigma1, n1, sigma2, n2, Tc, self.STREFPROP_Tmin, self.STREFPROP_Tmax = _Mulero_Cachadina_data_values[Mulero_Cachadina_data.index.get_loc(self.CASRN)].tolist()
            self.STREFPROP_coeffs = [sigma0, n0, sigma1, n1, sigma2, n2, Tc]
            Tmins.append(self.STREFPROP_Tmin); Tmaxs.append(self.STREFPROP_Tmax)
        if self.CASRN in Somayajulu_data_2.index:
            methods.append(SOMAYAJULU2)
            _, self.SOMAYAJULU2_Tt, self.SOMAYAJULU2_Tc, A, B, C = _Somayajulu_data_2_values[Somayajulu_data_2.index.get_loc(self.CASRN)].tolist()
            self.SOMAYAJULU2_coeffs = [A, B, C]
            Tmins.append(self.SOMAYAJULU2_Tt); Tmaxs.append(self.SOMAYAJULU2_Tc)
        if self.CASRN in Somayajulu_data.index:
            methods.append(SOMAYAJULU)
            _, self.SOMAYAJULU_Tt, self.SOMAYAJULU_Tc, A, B, C = _Somayajulu_data_values[Somayajulu_data.index.get_loc(self.CASRN)].tolist()
            self.SOMAYAJULU_coeffs = [A, B, C]
            Tmins.append(self.SOMAYAJULU_Tt); Tmaxs.append(self.SOMAYAJULU_Tc)
        if self.CASRN in _VDISaturationDict:
            methods.append(VDI_TABULAR)
            Ts, props = VDI_tabular_data(self.CASRN, 'sigma')
            self.VDI_Tmin = Ts[0]
            self.VDI_Tmax = Ts[-1]
            self.tabular_data[VDI_TABULAR] = (Ts, props)
            Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
        if self.CASRN in Jasper_Lange_data.index:
            methods.append(JASPER)
            _, a, b, self.JASPER_Tmin, self.JASPER_Tmax= _Jasper_Lange_data_values[Jasper_Lange_data.index.get_loc(self.CASRN)].tolist()
            self.JASPER_coeffs = [a, b]
            Tmins.append(self.JASPER_Tmin); Tmaxs.append(self.JASPER_Tmax)
        if all((self.Tc, self.Vc, self.omega)):
            methods.append(MIQUEU)
            Tmins.append(0.0); Tmaxs.append(self.Tc)
        if all((self.Tb, self.Tc, self.Pc)):
            methods.append(BROCK_BIRD)
            methods.append(SASTRI_RAO)
            Tmins.append(0.0); Tmaxs.append(self.Tc)
        if all((self.Tc, self.Pc, self.omega)):
            methods.append(PITZER)
            methods.append(ZUO_STENBY)
            Tmins.append(0.0); Tmaxs.append(self.Tc)
        if self.CASRN in VDI_PPDS_11.index:
            _,  Tm, Tc, A, B, C, D, E = _VDI_PPDS_11_values[VDI_PPDS_11.index.get_loc(self.CASRN)].tolist()
            self.VDI_PPDS_coeffs = [A, B, C, D, E]
            self.VDI_PPDS_Tc = Tc
            self.VDI_PPDS_Tm = Tm
            methods.append(VDI_PPDS)
            Tmins.append(self.VDI_PPDS_Tm) ; Tmaxs.append(self.VDI_PPDS_Tc); 
        if all((self.Tb, self.Hvap_Tb, self.MW)):
            # Cache Cpl at Tb for ease of calculation of Tmax
            self.Cpl_Tb = self.Cpl(self.Tb) if hasattr(self.Cpl, '__call__') else self.Cpl
            if self.Cpl_Tb:
                methods.append(ALEEM)
                # Tmin and Tmax for this method is known
                Tmax_possible = self.Tb + self.Hvap_Tb/self.Cpl_Tb
                # This method will ruin solve_prop as it is typically valid
                # well above Tc. If Tc is available, limit it to that.
                if self.Tc:
                    Tmax_possible = min(self.Tc, Tmax_possible)
                Tmins.append(0.0); Tmaxs.append(Tmax_possible)
        self.all_methods = set(methods)
        if Tmins and Tmaxs:
            # Note: All methods work right down to 0 K.
            self.Tmin = min(Tmins)
            self.Tmax = max(Tmaxs)

    def calculate(self, T, method):
        r'''Method to calculate surface tension of a liquid at temperature `T`
        with a given method.

        This method has no exception handling; see `T_dependent_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate surface tension, [K]
        method : str
            Name of the method to use

        Returns
        -------
        sigma : float
            Surface tension of the liquid at T, [N/m]
        '''
        if method == STREFPROP:
            sigma0, n0, sigma1, n1, sigma2, n2, Tc = self.STREFPROP_coeffs
            sigma = REFPROP(T, Tc=Tc, sigma0=sigma0, n0=n0, sigma1=sigma1, n1=n1,
                            sigma2=sigma2, n2=n2)
        elif method == VDI_PPDS:
            A, B, C, D, E = self.VDI_PPDS_coeffs
            Tr = T/self.VDI_PPDS_Tc
            sigma = A*(1. - Tr)**(B + C*Tr + D*Tr**2 + E*Tr**3)
        elif method == SOMAYAJULU2:
            A, B, C = self.SOMAYAJULU2_coeffs
            sigma = Somayajulu(T, Tc=self.SOMAYAJULU2_Tc, A=A, B=B, C=C)
        elif method == SOMAYAJULU:
            A, B, C = self.SOMAYAJULU_coeffs
            sigma = Somayajulu(T, Tc=self.SOMAYAJULU_Tc, A=A, B=B, C=C)
        elif method == JASPER:
            sigma = Jasper(T, a=self.JASPER_coeffs[0], b=self.JASPER_coeffs[1])
        elif method == BROCK_BIRD:
            sigma = Brock_Bird(T, self.Tb, self.Tc, self.Pc)
        elif method == SASTRI_RAO:
            sigma = Sastri_Rao(T, self.Tb, self.Tc, self.Pc)
        elif method == PITZER:
            sigma = Pitzer(T, self.Tc, self.Pc, self.omega)
        elif method == ZUO_STENBY:
            sigma = Zuo_Stenby(T, self.Tc, self.Pc, self.omega)
        elif method == MIQUEU:
            sigma = Miqueu(T, self.Tc, self.Vc, self.omega)
        elif method == ALEEM:
            Cpl = self.Cpl(T) if hasattr(self.Cpl, '__call__') else self.Cpl
            Vml = self.Vml(T) if hasattr(self.Vml, '__call__') else self.Vml
            rhol = Vm_to_rho(Vml, self.MW)
            sigma = Aleem(T=T, MW=self.MW, Tb=self.Tb, rhol=rhol, Hvap_Tb=self.Hvap_Tb, Cpl=Cpl)
        elif method in self.tabular_data:
            sigma = self.interpolate(T, method)
        return sigma

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For CSP methods, the models
        are considered valid from 0 K to the critical point. For tabular data,
        extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the extrapolation
        is considered valid for all temperatures.

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
        if method == STREFPROP:
            if T < self.STREFPROP_Tmin or T > self.STREFPROP_Tmax:
                validity = False
        elif method == VDI_PPDS:
            if T > self.VDI_PPDS_Tc:
                # Could also check for low temp, but not necessary as it extrapolates
                validity = False
        elif method == SOMAYAJULU2:
            if T < self.SOMAYAJULU2_Tt or T > self.SOMAYAJULU2_Tc:
                validity = False
        elif method == SOMAYAJULU:
            if T < self.SOMAYAJULU_Tt or T > self.SOMAYAJULU_Tc:
                validity = False
        elif method == JASPER:
            if T < self.JASPER_Tmin or T > self.JASPER_Tmax:
                validity = False
        elif method in [BROCK_BIRD, SASTRI_RAO, PITZER, ZUO_STENBY, MIQUEU]:
            if T > self.Tc:
                validity = False
        elif method == ALEEM:
            if T > self.Tb + self.Hvap_Tb/self.Cpl_Tb:
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


### Surface Tension Mixtures

def Winterfeld_Scriven_Davis(xs, sigmas, rhoms):
    r'''Calculates surface tension of a liquid mixture according to
    mixing rules in [1]_ and also in [2]_.

    .. math::
        \sigma_M = \sum_i \sum_j \frac{1}{V_L^{L2}}\left(x_i V_i \right)
        \left( x_jV_j\right)\sqrt{\sigma_i\cdot \sigma_j}

    Parameters
    ----------
    xs : array-like
        Mole fractions of all components, [-]
    sigmas : array-like
        Surface tensions of all components, [N/m]
    rhoms : array-like
        Molar densities of all components, [mol/m^3]

    Returns
    -------
    sigma : float
        Air-liquid surface tension of mixture, [N/m]

    Notes
    -----
    DIPPR Procedure 7C: Method for the Surface Tension of Nonaqueous Liquid
    Mixtures

    Becomes less accurate as liquid-liquid critical solution temperature is
    approached. DIPPR Evaluation:  3-4% AARD, from 107 nonaqueous binary
    systems, 1284 points. Internally, densities are converted to kmol/m^3. The
    Amgat function is used to obtain liquid mixture density in this equation.

    Raises a ZeroDivisionError if either molar volume are zero, and a
    ValueError if a surface tensions of a pure component is negative.

    Examples
    --------
    >>> Winterfeld_Scriven_Davis([0.1606, 0.8394], [0.01547, 0.02877],
    ... [8610., 15530.])
    0.024967388450439817

    References
    ----------
    .. [1] Winterfeld, P. H., L. E. Scriven, and H. T. Davis. "An Approximate
       Theory of Interfacial Tensions of Multicomponent Systems: Applications
       to Binary Liquid-Vapor Tensions." AIChE Journal 24, no. 6
       (November 1, 1978): 1010-14. doi:10.1002/aic.690240610.
    .. [2] Danner, Ronald P, and Design Institute for Physical Property Data.
       Manual for Predicting Chemical Process Design Data. New York, N.Y, 1982.
    '''
    if not none_and_length_check([xs, sigmas, rhoms]):
        raise Exception('Function inputs are incorrect format')
    rhoms = [i/1E3 for i in rhoms]
    Vms = [(i)**-1 for i in rhoms]
    rho = 1./mixing_simple(xs, Vms)
    cmps = range(len(xs))
    return sum([rho*rho*xs[i]/rhoms[i]*xs[j]/rhoms[j]*(sigmas[j]*sigmas[i])**0.5
                for i in cmps for j in cmps])


def Diguilio_Teja(T, xs, sigmas_Tb, Tbs, Tcs):
    r'''Calculates surface tension of a liquid mixture according to
    mixing rules in [1]_.

    .. math::
        \sigma = 1.002855(T^*)^{1.118091} \frac{T}{T_b} \sigma_r

        T^*  = \frac{(T_c/T)-1}{(T_c/T_b)-1}

        \sigma_r = \sum x_i \sigma_i

        T_b = \sum x_i T_{b,i}

        T_c = \sum x_i T_{c,i}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    xs : array-like
        Mole fractions of all components
    sigmas_Tb : array-like
        Surface tensions of all components at the boiling point, [N/m]
    Tbs : array-like
        Boiling temperatures of all components, [K]
    Tcs : array-like
        Critical temperatures of all components, [K]

    Returns
    -------
    sigma : float
        Air-liquid surface tension of mixture, [N/m]

    Notes
    -----
    Simple model, however it has 0 citations. Gives similar results to the
    `Winterfeld_Scriven_Davis` model.

    Raises a ValueError if temperature is greater than the mixture's critical
    temperature or if the given temperature is negative, or if the mixture's
    boiling temperature is higher than its critical temperature.
    
    [1]_ claims a 4.63 percent average absolute error on 21 binary and 4 
    ternary non-aqueous systems. [1]_ also considered Van der Waals mixing 
    rules for `Tc`, but found it provided a higher error of 5.58%

    Examples
    --------
    >>> Diguilio_Teja(T=298.15, xs=[0.1606, 0.8394],
    ... sigmas_Tb=[0.01424, 0.02530], Tbs=[309.21, 312.95], Tcs=[469.7, 508.0])
    0.025716823875045505

    References
    ----------
    .. [1] Diguilio, Ralph, and Amyn S. Teja. "Correlation and Prediction of
       the Surface Tensions of Mixtures." The Chemical Engineering Journal 38,
       no. 3 (July 1988): 205-8. doi:10.1016/0300-9467(88)80079-0.
    '''
    if not none_and_length_check([xs, sigmas_Tb, Tbs, Tcs]):
        raise Exception('Function inputs are incorrect format')

    Tc = mixing_simple(xs, Tcs)
    if T > Tc:
        raise ValueError('T > Tc according to Kays rule - model is not valid in this range.')
    Tb = mixing_simple(xs, Tbs)
    sigmar = mixing_simple(xs, sigmas_Tb)
    Tst = (Tc/T - 1.)/(Tc/Tb - 1)
    return 1.002855*Tst**1.118091*(T/Tb)*sigmar


WINTERFELDSCRIVENDAVIS = 'Winterfeld, Scriven, and Davis (1978)'
DIGUILIOTEJA = 'Diguilio and Teja (1988)'
SIMPLE = 'Simple'
NONE = 'None'

surface_tension_mixture_methods = [WINTERFELDSCRIVENDAVIS, DIGUILIOTEJA, SIMPLE]


class SurfaceTensionMixture(MixtureProperty):
    '''Class for dealing with surface tension of a mixture as a function of 
    temperature, pressure, and composition.
    Consists of two mixing rules specific to surface tension, and mole
    weighted averaging. 
         
    Prefered method is :obj:`Winterfeld_Scriven_Davis` which requires mole
    fractions, pure component surface tensions, and the molar density of each
    pure component. :obj:`Diguilio_Teja` is of similar accuracy, but requires
    the surface tensions of pure components at their boiling points, as well
    as boiling points and critical points and mole fractions. An ideal mixing
    rule based on mole fractions, **SIMPLE**, is also available and is still
    relatively accurate.
        
    Parameters
    ----------
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]
    Tbs : list[float], optional
        Boiling points of all species in the mixture, [K]
    Tcs : list[float], optional
        Critical temperatures of all species in the mixture, [K]
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
    SurfaceTensions : list[SurfaceTension], optional
        SurfaceTension objects created for all species in the mixture, normally 
        created by :obj:`thermo.chemical.Chemical`.
    VolumeLiquids : list[VolumeLiquid], optional
        VolumeLiquid objects created for all species in the mixture, normally 
        created by :obj:`thermo.chemical.Chemical`.

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`surface_tension_mixture_methods`.

    **WINTERFELDSCRIVENDAVIS**:
        Mixing rule described in :obj:`Winterfeld_Scriven_Davis`.
    **DIGUILIOTEJA**:
        Mixing rule described in :obj:`Diguilio_Teja`.
    **SIMPLE**:
        Mixing rule described in :obj:`thermo.utils.mixing_simple`.

    See Also
    --------
    Winterfeld_Scriven_Davis
    Diguilio_Teja

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''

    name = 'Surface tension'
    units = 'N/m'
    property_min = 0
    '''Mimimum valid value of surface tension. This occurs at the critical
    point exactly.'''
    property_max = 4.0
    '''Maximum valid value of surface tension. Set to roughly twice that of 
    cobalt at its melting point.'''
                            
    ranked_methods = [WINTERFELDSCRIVENDAVIS, DIGUILIOTEJA, SIMPLE]

    def __init__(self, MWs=[], Tbs=[], Tcs=[], CASs=[], SurfaceTensions=[], 
                 VolumeLiquids=[]):
        self.MWs = MWs
        self.Tbs = Tbs
        self.Tcs = Tcs
        self.CASs = CASs
        self.SurfaceTensions = SurfaceTensions
        self.VolumeLiquids = VolumeLiquids

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        surface tension under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        surface tension above.'''

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by `mixture_property`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `mixture_property`.'''
        self.all_methods = set()
        '''Set of all methods available for a given set of information;
        filled by :obj:`load_all_methods`.'''
        self.load_all_methods()

    def load_all_methods(self):
        r'''Method to initialize the object by precomputing any values which
        may be used repeatedly and by retrieving mixture-specific variables.
        All data are stored as attributes. This method also sets :obj:`Tmin`, 
        :obj:`Tmax`, and :obj:`all_methods` as a set of methods which should 
        work to calculate the property.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods = []        
        methods.append(SIMPLE) # Needs sigma
        methods.append(WINTERFELDSCRIVENDAVIS) # Nothing to load, needs rhoms, sigma
        if none_and_length_check((self.Tbs, self.Tcs)):
            self.sigmas_Tb = [i(Tb) for i, Tb in zip(self.SurfaceTensions, self.Tbs)]
            if none_and_length_check([self.sigmas_Tb]):
                methods.append(DIGUILIOTEJA)
        self.all_methods = set(methods)
            
    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate surface tension of a liquid mixture at 
        temperature `T`, pressure `P`, mole fractions `zs` and weight fractions
        `ws` with a given method.

        This method has no exception handling; see `mixture_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate the property, [K]
        P : float
            Pressure at which to calculate the property, [Pa]
        zs : list[float]
            Mole fractions of all species in the mixture, [-]
        ws : list[float]
            Weight fractions of all species in the mixture, [-]
        method : str
            Name of the method to use

        Returns
        -------
        sigma : float
            Surface tension of the liquid at given conditions, [N/m]
        '''
        if method == SIMPLE:
            sigmas = [i(T) for i in self.SurfaceTensions]
            return mixing_simple(zs, sigmas)
        elif method == DIGUILIOTEJA:
            return Diguilio_Teja(T=T, xs=zs, sigmas_Tb=self.sigmas_Tb, 
                                 Tbs=self.Tbs, Tcs=self.Tcs)
        elif method == WINTERFELDSCRIVENDAVIS:
            sigmas = [i(T) for i in self.SurfaceTensions]
            rhoms = [1./i(T, P) for i in self.VolumeLiquids]
            return Winterfeld_Scriven_Davis(zs, sigmas, rhoms)
        else:
            raise Exception('Method not valid')

    def test_method_validity(self, T, P, zs, ws, method):
        r'''Method to test the validity of a specified method for the given
        conditions. No methods have implemented checks or strict ranges of 
        validity.

        Parameters
        ----------
        T : float
            Temperature at which to check method validity, [K]
        P : float
            Pressure at which to check method validity, [Pa]
        zs : list[float]
            Mole fractions of all species in the mixture, [-]
        ws : list[float]
            Weight fractions of all species in the mixture, [-]
        method : str
            Method name to use

        Returns
        -------
        validity : bool
            Whether or not a specifid method is valid
        '''
        # SIMPLE and WINTERFELDSCRIVENDAVIS need to calculate sigma for pure
        # species - doesn't work above Tc for any compound.
        # DIGUILIOTEJA needs Tcs, not sure.
        if method in [SIMPLE, DIGUILIOTEJA, WINTERFELDSCRIVENDAVIS]:
            return True
        else:
            raise Exception('Method not valid')

