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

__all__ = ['COSTALD_data', 'SNM0_data', 'Perry_l_data', 'CRC_inorg_l_data', 
'VDI_PPDS_2',
'CRC_inorg_l_const_data', 'CRC_inorg_s_const_data', 'CRC_virial_data', 
'Yen_Woods_saturation', 'Rackett', 'Yamada_Gunn', 'Townsend_Hales', 
'Bhirud_normal', 'COSTALD', 'Campbell_Thodos', 'SNM0', 'CRC_inorganic', 
'volume_liquid_methods', 'volume_liquid_methods_P', 'VolumeLiquid', 
'COSTALD_compressed', 'Amgat', 'Rackett_mixture', 'COSTALD_mixture', 
'ideal_gas', 'volume_gas_methods', 'VolumeGas', 
'volume_gas_mixture_methods', 'volume_solid_mixture_methods', 'Goodman',
 'volume_solid_methods', 'VolumeSolid',
'VolumeLiquidMixture', 'VolumeGasMixture', 'VolumeSolidMixture']

import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

from thermo.utils import R
from thermo.utils import log, exp
from thermo.utils import Vm_to_rho, rho_to_Vm, mixing_simple, none_and_length_check
from thermo.virial import BVirial_Pitzer_Curl, BVirial_Abbott, BVirial_Tsonopoulos, BVirial_Tsonopoulos_extended
from thermo.miscdata import _VDISaturationDict, VDI_tabular_data
from thermo.dippr import EQ105
from thermo.electrochem import _Laliberte_Density_ParametersDict, Laliberte_density
from thermo.coolprop import has_CoolProp, PropsSI, PhaseSI, coolprop_fluids, coolprop_dict, CoolProp_T_dependent_property
from thermo.utils import TDependentProperty, TPDependentProperty, MixtureProperty
from thermo.eos import PR78

folder = os.path.join(os.path.dirname(__file__), 'Density')

COSTALD_data = pd.read_csv(os.path.join(folder, 'COSTALD Parameters.tsv'),
                           sep='\t', index_col=0)

SNM0_data = pd.read_csv(os.path.join(folder, 'Mchaweh SN0 deltas.tsv'),
                        sep='\t', index_col=0)

Perry_l_data = pd.read_csv(os.path.join(folder, 'Perry Parameters 105.tsv'),
                           sep='\t', index_col=0)
_Perry_l_data_values = Perry_l_data.values

VDI_PPDS_2 = pd.read_csv(os.path.join(folder, 'VDI PPDS Density of Saturated Liquids.tsv'),
                          sep='\t', index_col=0)
_VDI_PPDS_2_values = VDI_PPDS_2.values


CRC_inorg_l_data = pd.read_csv(os.path.join(folder, 'CRC Inorganics densties of molten compounds and salts.tsv'),
                               sep='\t', index_col=0)
_CRC_inorg_l_data_values = CRC_inorg_l_data.values

CRC_inorg_l_const_data = pd.read_csv(os.path.join(folder, 'CRC Liquid Inorganic Constant Densities.tsv'),
                                     sep='\t', index_col=0)

CRC_inorg_s_const_data = pd.read_csv(os.path.join(folder, 'CRC Solid Inorganic Constant Densities.tsv'),
                                     sep='\t', index_col=0)

CRC_virial_data = pd.read_csv(os.path.join(folder, 'CRC Virial polynomials.tsv'),
                              sep='\t', index_col=0)
_CRC_virial_data_values = CRC_virial_data.values

### Critical-properties based


def Yen_Woods_saturation(T, Tc, Vc, Zc):
    r'''Calculates saturation liquid volume, using the Yen and Woods [1]_ CSP
    method and a chemical's critical properties.

    The molar volume of a liquid is given by:

    .. math::
        Vc/Vs = 1 + A(1-T_r)^{1/3} + B(1-T_r)^{2/3} + D(1-T_r)^{4/3}

        D = 0.93-B

        A = 17.4425 - 214.578Z_c + 989.625Z_c^2 - 1522.06Z_c^3

        B = -3.28257 + 13.6377Z_c + 107.4844Z_c^2-384.211Z_c^3
        \text{ if } Zc \le 0.26

        B = 60.2091 - 402.063Z_c + 501.0 Z_c^2 + 641.0 Z_c^3
        \text{ if } Zc \ge 0.26


    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Vc : float
        Critical volume of fluid [m^3/mol]
    Zc : float
        Critical compressibility of fluid, [-]

    Returns
    -------
    Vs : float
        Saturation liquid volume, [m^3/mol]

    Notes
    -----
    Original equation was in terms of density, but it is converted here.

    No example has been found, nor are there points in the article. However,
    it is believed correct. For compressed liquids with the Yen-Woods method,
    see the `YenWoods_compressed` function.

    Examples
    --------
    >>> Yen_Woods_saturation(300, 647.14, 55.45E-6, 0.245)
    1.7695330765295693e-05

    References
    ----------
    .. [1] Yen, Lewis C., and S. S. Woods. "A Generalized Equation for Computer
       Calculation of Liquid Densities." AIChE Journal 12, no. 1 (1966):
       95-99. doi:10.1002/aic.690120119
    '''
    Tr = T/Tc
    A = 17.4425 - 214.578*Zc + 989.625*Zc**2 - 1522.06*Zc**3
    if Zc <= 0.26:
        B = -3.28257 + 13.6377*Zc + 107.4844*Zc**2 - 384.211*Zc**3
    else:
        B = 60.2091 - 402.063*Zc + 501.0*Zc**2 + 641.0*Zc**3
    D = 0.93 - B
    Vm = Vc/(1 + A*(1-Tr)**(1/3.) + B*(1-Tr)**(2/3.) + D*(1-Tr)**(4/3.))
    return Vm


def Rackett(T, Tc, Pc, Zc):
    r'''Calculates saturation liquid volume, using Rackett CSP method and
    critical properties.

    The molar volume of a liquid is given by:

    .. math::
        V_s = \frac{RT_c}{P_c}{Z_c}^{[1+(1-{T/T_c})^{2/7} ]}

    Units are all currently in m^3/mol - this can be changed to kg/m^3

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]
    Zc : float
        Critical compressibility of fluid, [-]

    Returns
    -------
    Vs : float
        Saturation liquid volume, [m^3/mol]

    Notes
    -----
    Units are dependent on gas constant R, imported from scipy
    According to Reid et. al, underpredicts volume for compounds with Zc < 0.22

    Examples
    --------
    Propane, example from the API Handbook

    >>> Vm_to_rho(Rackett(272.03889, 369.83, 4248000.0, 0.2763), 44.09562)
    531.3223212651092

    References
    ----------
    .. [1] Rackett, Harold G. "Equation of State for Saturated Liquids."
       Journal of Chemical & Engineering Data 15, no. 4 (1970): 514-517.
       doi:10.1021/je60047a012
    '''
    return R*Tc/Pc*Zc**(1 + (1 - T/Tc)**(2/7.))


def Yamada_Gunn(T, Tc, Pc, omega):
    r'''Calculates saturation liquid volume, using Yamada and Gunn CSP method
    and a chemical's critical properties and acentric factor.

    The molar volume of a liquid is given by:

    .. math::
        V_s = \frac{RT_c}{P_c}{(0.29056-0.08775\omega)}^{[1+(1-{T/T_c})^{2/7}]}

    Units are in m^3/mol.

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
    Vs : float
        saturation liquid volume, [m^3/mol]

    Notes
    -----
    This equation is an improvement on the Rackett equation.
    This is often presented as the Rackett equation.
    The acentric factor is used here, instead of the critical compressibility
    A variant using a reference fluid also exists

    Examples
    --------
    >>> Yamada_Gunn(300, 647.14, 22048320.0, 0.245)
    2.1882836429895796e-05

    References
    ----------
    .. [1] Gunn, R. D., and Tomoyoshi Yamada. "A Corresponding States
        Correlation of Saturated Liquid Volumes." AIChE Journal 17, no. 6
        (1971): 1341-45. doi:10.1002/aic.690170613
    .. [2] Yamada, Tomoyoshi, and Robert D. Gunn. "Saturated Liquid Molar
        Volumes. Rackett Equation." Journal of Chemical & Engineering Data 18,
        no. 2 (1973): 234-36. doi:10.1021/je60057a006
    '''
    return R*Tc/Pc*(0.29056 - 0.08775*omega)**(1 + (1 - T/Tc)**(2/7.))


def Townsend_Hales(T, Tc, Vc, omega):
    r'''Calculates saturation liquid density, using the Townsend and Hales
    CSP method as modified from the original Riedel equation. Uses
    chemical critical volume and temperature, as well as acentric factor

    The density of a liquid is given by:

    .. math::
        Vs = V_c/\left(1+0.85(1-T_r)+(1.692+0.986\omega)(1-T_r)^{1/3}\right)

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
    Vs : float
        Saturation liquid volume, [m^3/mol]

    Notes
    -----
    The requirement for critical volume and acentric factor requires all data.

    Examples
    --------
    >>> Townsend_Hales(300, 647.14, 55.95E-6, 0.3449)
    1.8007361992619923e-05

    References
    ----------
    .. [1] Hales, J. L, and R Townsend. "Liquid Densities from 293 to 490 K of
       Nine Aromatic Hydrocarbons." The Journal of Chemical Thermodynamics
       4, no. 5 (1972): 763-72. doi:10.1016/0021-9614(72)90050-X
    '''
    Tr = T/Tc
    return Vc/(1 + 0.85*(1-Tr) + (1.692 + 0.986*omega)*(1-Tr)**(1/3.))


Bhirud_normal_Trs = [0.98, 0.982, 0.984, 0.986, 0.988, 0.99, 0.992, 0.994,
            0.996, 0.998, 0.999, 1]
Bhirud_normal_lnU0s = [-1.6198, -1.604, -1.59, -1.578, -1.564, -1.548, -1.533,
              -1.515, -1.489, -1.454, -1.425, -1.243]
Bhirud_normal_lnU1 = [-0.4626, -0.459, -0.451, -0.441, -0.428, -0.412, -0.392,
              -0.367, -0.337, -0.302, -0.283, -0.2629]
Bhirud_normal_lnU0_interp = interp1d(Bhirud_normal_Trs, Bhirud_normal_lnU0s, kind='cubic')
Bhirud_normal_lnU1_interp = interp1d(Bhirud_normal_Trs, Bhirud_normal_lnU1, kind='cubic')


def Bhirud_normal(T, Tc, Pc, omega):
    r'''Calculates saturation liquid density using the Bhirud [1]_ CSP method.
    Uses Critical temperature and pressure and acentric factor.

    The density of a liquid is given by:

    .. math::
        &\ln \frac{P_c}{\rho RT} = \ln U^{(0)} + \omega\ln U^{(1)}

        &\ln U^{(0)} = 1.396 44 - 24.076T_r+ 102.615T_r^2
        -255.719T_r^3+355.805T_r^4-256.671T_r^5 + 75.1088T_r^6

        &\ln U^{(1)} = 13.4412 - 135.7437 T_r + 533.380T_r^2-
        1091.453T_r^3+1231.43T_r^4 - 728.227T_r^5 + 176.737T_r^6

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
    Vm : float
        Saturated liquid molar volume, [mol/m^3]

    Notes
    -----
    Claimed inadequate by others.

    An interpolation table for ln U values are used from Tr = 0.98 - 1.000.
    Has terrible behavior at low reduced temperatures.

    Examples
    --------
    Pentane

    >>> Bhirud_normal(280.0, 469.7, 33.7E5, 0.252)
    0.00011249654029488583

    References
    ----------
    .. [1] Bhirud, Vasant L. "Saturated Liquid Densities of Normal Fluids."
       AIChE Journal 24, no. 6 (November 1, 1978): 1127-31.
       doi:10.1002/aic.690240630
    '''
    Tr = T/Tc
    if Tr <= 0.98:
        lnU0 = 1.39644 - 24.076*Tr + 102.615*Tr**2 - 255.719*Tr**3 \
            + 355.805*Tr**4 - 256.671*Tr**5 + 75.1088*Tr**6
        lnU1 = 13.4412 - 135.7437*Tr + 533.380*Tr**2-1091.453*Tr**3 \
            + 1231.43*Tr**4 - 728.227*Tr**5 + 176.737*Tr**6
    elif Tr > 1:
        raise Exception('Critical phase, correlation does not apply')
    else:
        lnU0 = Bhirud_normal_lnU0_interp(Tr)
        lnU1 = Bhirud_normal_lnU1_interp(Tr)

    Unonpolar = exp(lnU0 + omega*lnU1)
    Vm = Unonpolar*R*T/Pc
    return Vm


def COSTALD(T, Tc, Vc, omega):
    r'''Calculate saturation liquid density using the COSTALD CSP method.

    A popular and accurate estimation method. If possible, fit parameters are
    used; alternatively critical properties work well.

    The density of a liquid is given by:

    .. math::
        V_s=V^*V^{(0)}[1-\omega_{SRK}V^{(\delta)}]

        V^{(0)}=1-1.52816(1-T_r)^{1/3}+1.43907(1-T_r)^{2/3}
        - 0.81446(1-T_r)+0.190454(1-T_r)^{4/3}

        V^{(\delta)}=\frac{-0.296123+0.386914T_r-0.0427258T_r^2-0.0480645T_r^3}
        {T_r-1.00001}

    Units are that of critical or fit constant volume.

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tc : float
        Critical temperature of fluid [K]
    Vc : float
        Critical volume of fluid [m^3/mol].
        This parameter is alternatively a fit parameter
    omega : float
        (ideally SRK) Acentric factor for fluid, [-]
        This parameter is alternatively a fit parameter.

    Returns
    -------
    Vs : float
        Saturation liquid volume

    Notes
    -----
    196 constants are fit to this function in [1]_.
    Range: 0.25 < Tr < 0.95, often said to be to 1.0

    This function has been checked with the API handbook example problem.

    Examples
    --------
    Propane, from an example in the API Handbook

    >>> Vm_to_rho(COSTALD(272.03889, 369.83333, 0.20008161E-3, 0.1532), 44.097)
    530.3009967969841


    References
    ----------
    .. [1] Hankinson, Risdon W., and George H. Thomson. "A New Correlation for
       Saturated Densities of Liquids and Their Mixtures." AIChE Journal
       25, no. 4 (1979): 653-663. doi:10.1002/aic.690250412
    '''
    Tr = T/Tc
    V_delta = (-0.296123 + 0.386914*Tr - 0.0427258*Tr**2
        - 0.0480645*Tr**3)/(Tr - 1.00001)
    V_0 = 1 - 1.52816*(1-Tr)**(1/3.) + 1.43907*(1-Tr)**(2/3.) \
        - 0.81446*(1-Tr) + 0.190454*(1-Tr)**(4/3.)
    return Vc*V_0*(1-omega*V_delta)


def Campbell_Thodos(T, Tb, Tc, Pc, M, dipole=None, hydroxyl=False):
    r'''Calculate saturation liquid density using the Campbell-Thodos [1]_
    CSP method.

    An old and uncommon estimation method.

    .. math::
        V_s = \frac{RT_c}{P_c}{Z_{RA}}^{[1+(1-T_r)^{2/7}]}

        Z_{RA} = \alpha + \beta(1-T_r)

        \alpha = 0.3883-0.0179s

        s = T_{br} \frac{\ln P_c}{(1-T_{br})}

        \beta = 0.00318s-0.0211+0.625\Lambda^{1.35}

        \Lambda = \frac{P_c^{1/3}} { M^{1/2} T_c^{5/6}}

    For polar compounds:

    .. math::
        \theta = P_c \mu^2/T_c^2

        \alpha = 0.3883 - 0.0179s - 130540\theta^{2.41}

        \beta = 0.00318s - 0.0211 + 0.625\Lambda^{1.35} + 9.74\times
        10^6 \theta^{3.38}

    Polar Combounds with hydroxyl groups (water, alcohols)

    .. math::
        \alpha = \left[0.690T_{br} -0.3342 + \frac{5.79\times 10^{-10}}
        {T_{br}^{32.75}}\right] P_c^{0.145}

        \beta = 0.00318s - 0.0211 + 0.625 \Lambda^{1.35} + 5.90\Theta^{0.835}

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
    M : float
        Molecular weight of the fluid [g/mol]
    dipole : float, optional
        Dipole moment of the fluid [debye]
    hydroxyl : bool, optional
        Swith to use the hydroxyl variant for polar fluids

    Returns
    -------
    Vs : float
        Saturation liquid volume

    Notes
    -----
    If a dipole is provided, the polar chemical method is used.
    The paper is an excellent read.
    Pc is internally converted to atm.

    Examples
    --------
    Ammonia, from [1]_.

    >>> Campbell_Thodos(T=405.45, Tb=239.82, Tc=405.45, Pc=111.7*101325, M=17.03, dipole=1.47)
    7.347363635885525e-05

    References
    ----------
    .. [1] Campbell, Scott W., and George Thodos. "Prediction of Saturated
       Liquid Densities and Critical Volumes for Polar and Nonpolar
       Substances." Journal of Chemical & Engineering Data 30, no. 1
       (January 1, 1985): 102-11. doi:10.1021/je00039a032.
    '''
    Tr = T/Tc
    Tbr = Tb/Tc
    Pc = Pc/101325.
    s = Tbr * log(Pc)/(1-Tbr)
    Lambda = Pc**(1/3.)/(M**0.5*Tc**(5/6.))
    alpha = 0.3883 - 0.0179*s
    beta = 0.00318*s - 0.0211 + 0.625*Lambda**(1.35)
    if dipole:
        theta = Pc*dipole**2/Tc**2
        alpha -= 130540 * theta**2.41
        beta += 9.74E6 * theta**3.38
    if hydroxyl:
        beta = 0.00318*s - 0.0211 + 0.625*Lambda**(1.35) + 5.90*theta**0.835
        alpha = (0.69*Tbr - 0.3342 + 5.79E-10/Tbr**32.75)*Pc**0.145
    Zra = alpha + beta*(1-Tr)
    Vs = R*Tc/(Pc*101325)*Zra**(1+(1-Tr)**(2/7.))
    return Vs


def SNM0(T, Tc, Vc, omega, delta_SRK=None):
    r'''Calculates saturated liquid density using the Mchaweh, Moshfeghian
    model [1]_. Designed for simple calculations.

    .. math::
        V_s = V_c/(1+1.169\tau^{1/3}+1.818\tau^{2/3}-2.658\tau+2.161\tau^{4/3}

        \tau = 1-\frac{(T/T_c)}{\alpha_{SRK}}

        \alpha_{SRK} = [1 + m(1-\sqrt{T/T_C}]^2

        m = 0.480+1.574\omega-0.176\omega^2

    If the fit parameter `delta_SRK` is provided, the following is used:

    .. math::
        V_s = V_C/(1+1.169\tau^{1/3}+1.818\tau^{2/3}-2.658\tau+2.161\tau^{4/3})
        /\left[1+\delta_{SRK}(\alpha_{SRK}-1)^{1/3}\right]

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
    delta_SRK : float, optional
        Fitting parameter [-]

    Returns
    -------
    Vs : float
        Saturation liquid volume, [m^3/mol]

    Notes
    -----
    73 fit parameters have been gathered from the article.

    Examples
    --------
    Argon, without the fit parameter and with it. Tabulated result in Perry's
    is 3.4613e-05. The fit increases the error on this occasion.

    >>> SNM0(121, 150.8, 7.49e-05, -0.004)
    3.4402256402733416e-05
    >>> SNM0(121, 150.8, 7.49e-05, -0.004, -0.03259620)
    3.493288100008123e-05

    References
    ----------
    .. [1] Mchaweh, A., A. Alsaygh, Kh. Nasrifar, and M. Moshfeghian.
       "A Simplified Method for Calculating Saturated Liquid Densities."
       Fluid Phase Equilibria 224, no. 2 (October 1, 2004): 157-67.
       doi:10.1016/j.fluid.2004.06.054
    '''
    Tr = T/Tc
    m = 0.480 + 1.574*omega - 0.176*omega*omega
    alpha_SRK = (1. + m*(1. - Tr**0.5))**2
    tau = 1. - Tr/alpha_SRK

    rho0 = 1. + 1.169*tau**(1/3.) + 1.818*tau**(2/3.) - 2.658*tau + 2.161*tau**(4/3.)
    V0 = 1./rho0

    if not delta_SRK:
        return Vc*V0
    else:
        return Vc*V0/(1. + delta_SRK*(alpha_SRK - 1.)**(1/3.))


def CRC_inorganic(T, rho0, k, Tm):
    r'''Calculates liquid density of a molten element or salt at temperature
    above the melting point. Some coefficients are given nearly up to the
    boiling point.

    The mass density of the inorganic liquid is given by:

    .. math::
        \rho = \rho_{0} - k(T-T_m)

    Parameters
    ----------
    T : float
        Temperature of the liquid, [K]
    rho0 : float
        Mass density of the liquid at Tm, [kg/m^3]
    k : float
        Linear temperature dependence of the mass density, [kg/m^3/K]
    Tm : float
        The normal melting point, used in the correlation [K]

    Returns
    -------
    rho : float
        Mass density of molten metal or salt, [kg/m^3]

    Notes
    -----
    [1]_ has units of g/mL. While the individual densities could have been
    converted to molar units, the temperature coefficient could only be
    converted by refitting to calculated data. To maintain compatibility with
    the form of the equations, this was not performed.

    This linear form is useful only in small temperature ranges.
    Coefficients for one compound could be used to predict the temperature
    dependence of density of a similar compound.


    Examples
    --------
    >>> CRC_inorganic(300, 2370.0, 2.687, 239.08)
    2206.30796

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
        Chemistry and Physics, 95E. [Boca Raton, FL]: CRC press, 2014.
    '''
    return rho0 - k*(T-Tm)


COOLPROP = 'COOLPROP'
PERRYDIPPR = 'PERRYDIPPR'
VDI_PPDS = 'VDI_PPDS'
MMSNM0 = 'MMSNM0'
MMSNM0FIT = 'MMSNM0FIT'
VDI_TABULAR = 'VDI_TABULAR'
HTCOSTALD = 'HTCOSTALD'
HTCOSTALDFIT = 'HTCOSTALDFIT'
COSTALD_COMPRESSED = 'COSTALD_COMPRESSED'
RACKETT = 'RACKETT'
RACKETTFIT = 'RACKETTFIT'
YEN_WOODS_SAT = 'YEN_WOODS_SAT'
YAMADA_GUNN = 'YAMADA_GUNN'
BHIRUD_NORMAL = 'BHIRUD_NORMAL'
TOWNSEND_HALES = 'TOWNSEND_HALES'
CAMPBELL_THODOS = 'CAMPBELL_THODOS'
EOS = 'EOS'


CRC_INORG_L = 'CRC_INORG_L'
CRC_INORG_L_CONST = 'CRC_INORG_L_CONST'

volume_liquid_methods = [PERRYDIPPR, VDI_PPDS, COOLPROP, MMSNM0FIT, VDI_TABULAR,
                         HTCOSTALDFIT, RACKETTFIT, CRC_INORG_L,
                         CRC_INORG_L_CONST, MMSNM0, HTCOSTALD,
                         YEN_WOODS_SAT, RACKETT, YAMADA_GUNN,
                         BHIRUD_NORMAL, TOWNSEND_HALES, CAMPBELL_THODOS]
'''Holds all low-pressure methods available for the VolumeLiquid class, for use
in iterating over them.'''

volume_liquid_methods_P = [COOLPROP, COSTALD_COMPRESSED, EOS]
'''Holds all high-pressure methods available for the VolumeLiquid class, for
use in iterating over them.'''


class VolumeLiquid(TPDependentProperty):
    r'''Class for dealing with liquid molar volume as a function of
    temperature and pressure.

    For low-pressure (at 1 atm while under the vapor pressure; along the
    saturation line otherwise) liquids, there are six coefficient-based methods
    from five data sources, one source of tabular information, one source of
    constant values, eight corresponding-states estimators, and the external
    library CoolProp.

    For high-pressure liquids (also, <1 atm liquids), there is one
    corresponding-states estimator, and the external library CoolProp.

    Parameters
    ----------
    CASRN : str, optional
        The CAS number of the chemical
    MW : float, optional
        Molecular weight, [g/mol]
    Tb : float, optional
        Boiling point, [K]
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
    dipole : float, optional
        Dipole, [debye]
    Psat : float or callable, optional
        Vapor pressure at a given temperature, or callable for the same [Pa]
    eos : object, optional
        Equation of State object after :obj:`thermo.eos.GCEOS`

    Notes
    -----
    A string holding each method's name is assigned to the following variables
    in this module, intended as the most convenient way to refer to a method.
    To iterate over all methods, use the lists stored in
    :obj:`surface_tension_methods` and :obj:`volume_liquid_methods_P` for low
    and high pressure methods respectively.

    Low pressure methods:

    **PERRYDIPPR**:
        A simple polynomial as expressed in [1]_, with data available for
        344 fluids. Temperature limits are available for all fluids. Believed
        very accurate.
    **VDI_PPDS**:
        Coefficients for a equation form developed by the PPDS, published 
        openly in [3]_. Valid up to the critical temperature, and extrapolates
        to very low temperatures well.
    **MMSNM0FIT**:
        Uses a fit coefficient for better accuracy in the :obj:`SNM0` method,
        Coefficients available for 73 fluids from [2]_. Valid to the critical
        point.
    **HTCOSTALDFIT**:
        A method with two fit coefficients to the :obj:`COSTALD` method.
        Coefficients available for 192 fluids, from [3]_. Valid to the critical
        point.
    **RACKETTFIT**:
        The :obj:`Racket` method, with a fit coefficient Z_RA. Data is
        available for 186 fluids, from [3]_. Valid to the critical point.
    **CRC_INORG_L**:
        Single-temperature coefficient linear model in terms of mass density
        for the density of inorganic liquids; converted to molar units
        internally. Data is available for 177 fluids normally valid over a
        narrow range above the melting point, from [4]_; described in
        :obj:`CRC_inorganic`.
    **MMSNM0**:
        CSP method, described in :obj:`SNM0`.
    **HTCOSTALD**:
        CSP method, described in :obj:`COSTALD`.
    **YEN_WOODS_SAT**:
        CSP method, described in :obj:`Yen_Woods_saturation`.
    **RACKETT**:
        CSP method, described in :obj:`Rackett`.
    **YAMADA_GUNN**:
        CSP method, described in :obj:`Yamada_Gunn`.
    **BHIRUD_NORMAL**:
        CSP method, described in :obj:`Bhirud_normal`.
    **TOWNSEND_HALES**:
        CSP method, described in :obj:`Townsend_Hales`.
    **CAMPBELL_THODOS**:
        CSP method, described in :obj:`Campbell_Thodos`.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [5]_. Very slow.
    **CRC_INORG_L_CONST**:
        Constant inorganic liquid densities, in [4]_.
    **VDI_TABULAR**:
        Tabular data in [6]_ along the saturation curve; interpolation is as
        set by the user or the default.

    High pressure methods:

    **COSTALD_COMPRESSED**:
        CSP method, described in :obj:`COSTALD_compressed`. Calculates a
        low-pressure molar volume first, using `T_dependent_property`.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [5]_. Very slow, but unparalled in accuracy for pressure
        dependence.
    **EOS**:
        Equation of state provided by user.

    See Also
    --------
    Yen_Woods_saturation
    Rackett
    Yamada_Gunn
    Townsend_Hales
    Bhirud_normal
    COSTALD
    Campbell_Thodos
    SNM0
    CRC_inorganic
    COSTALD_compressed

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       8E. McGraw-Hill Professional, 2007.
    .. [2] Mchaweh, A., A. Alsaygh, Kh. Nasrifar, and M. Moshfeghian.
       "A Simplified Method for Calculating Saturated Liquid Densities."
       Fluid Phase Equilibria 224, no. 2 (October 1, 2004): 157-67.
       doi:10.1016/j.fluid.2004.06.054
    .. [3] Hankinson, Risdon W., and George H. Thomson. "A New Correlation for
       Saturated Densities of Liquids and Their Mixtures." AIChE Journal
       25, no. 4 (1979): 653-663. doi:10.1002/aic.690250412
    .. [4] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    .. [5] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    .. [6] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    name = 'Liquid molar volume'
    units = 'mol/m^3'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_P = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default.'''
    property_min = 0
    '''Mimimum valid value of liquid molar volume. It should normally occur at the
    triple point, and be well above this.'''
    property_max = 2e-3
    '''Maximum valid value of liquid molar volume. Generous limit.'''

    ranked_methods = [PERRYDIPPR, VDI_PPDS, COOLPROP, MMSNM0FIT, VDI_TABULAR,
                      HTCOSTALDFIT, RACKETTFIT, CRC_INORG_L,
                      CRC_INORG_L_CONST, MMSNM0, HTCOSTALD,
                      YEN_WOODS_SAT, RACKETT, YAMADA_GUNN,
                      BHIRUD_NORMAL, TOWNSEND_HALES, CAMPBELL_THODOS, EOS]
    '''Default rankings of the low-pressure methods.'''

    ranked_methods_P = [COOLPROP, COSTALD_COMPRESSED, EOS]
    '''Default rankings of the high-pressure methods.'''


    def __init__(self, MW=None, Tb=None, Tc=None, Pc=None, Vc=None, Zc=None,
                 omega=None, dipole=None, Psat=None, CASRN='', eos=None):
        self.CASRN = CASRN
        self.MW = MW
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.Vc = Vc
        self.Zc = Zc
        self.omega = omega
        self.dipole = dipole
        self.Psat = Psat
        self.eos = eos

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        liquid molar volume under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        liquid molar volume above.'''

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

        self.sorted_valid_methods = []
        '''sorted_valid_methods, list: Stored methods which were found valid
        at a specific temperature; set by `T_dependent_property`.'''
        self.sorted_valid_methods_P = []
        '''sorted_valid_methods_P, list: Stored methods which were found valid
        at a specific temperature; set by `TP_dependent_property`.'''
        self.user_methods = []
        '''user_methods, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `T_dependent_property`.'''
        self.user_methods_P = []
        '''user_methods_P, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `TP_dependent_property`.'''

        self.all_methods = set()
        '''Set of all low-pressure methods available for a given CASRN and
        properties; filled by :obj:`load_all_methods`.'''
        self.all_methods_P = set()
        '''Set of all high-pressure methods available for a given CASRN and
        properties; filled by :obj:`load_all_methods`.'''

        self.load_all_methods()

    def load_all_methods(self):
        r'''Method which picks out coefficients for the specified chemical
        from the various dictionaries and DataFrames storing it. All data is
        stored as attributes. This method also sets :obj:`Tmin`, :obj:`Tmax`,
        :obj:`all_methods` and obj:`all_methods_P` as a set of methods for
        which the data exists for.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods = []
        methods_P = []
        Tmins, Tmaxs = [], []
        if has_CoolProp and self.CASRN in coolprop_dict:
            methods.append(COOLPROP); methods_P.append(COOLPROP)
            self.CP_f = coolprop_fluids[self.CASRN]
            Tmins.append(self.CP_f.Tt); Tmaxs.append(self.CP_f.Tc)
        if self.CASRN in CRC_inorg_l_data.index:
            methods.append(CRC_INORG_L)
            _, self.CRC_INORG_L_MW, self.CRC_INORG_L_rho, self.CRC_INORG_L_k, self.CRC_INORG_L_Tm, self.CRC_INORG_L_Tmax = _CRC_inorg_l_data_values[CRC_inorg_l_data.index.get_loc(self.CASRN)].tolist()
            Tmins.append(self.CRC_INORG_L_Tm); Tmaxs.append(self.CRC_INORG_L_Tmax)
        if self.CASRN in Perry_l_data.index:
            methods.append(PERRYDIPPR)
            _, C1, C2, C3, C4, self.DIPPR_Tmin, self.DIPPR_Tmax = _Perry_l_data_values[Perry_l_data.index.get_loc(self.CASRN)].tolist()
            self.DIPPR_coeffs = [C1, C2, C3, C4]
            Tmins.append(self.DIPPR_Tmin); Tmaxs.append(self.DIPPR_Tmax)
        if self.CASRN in VDI_PPDS_2.index:
            methods.append(VDI_PPDS)
            _, MW, Tc, rhoc, A, B, C, D = _VDI_PPDS_2_values[VDI_PPDS_2.index.get_loc(self.CASRN)].tolist()
            self.VDI_PPDS_coeffs = [A, B, C, D]
            self.VDI_PPDS_MW = MW
            self.VDI_PPDS_Tc = Tc
            self.VDI_PPDS_rhoc = rhoc
            Tmaxs.append(self.VDI_PPDS_Tc)
        if self.CASRN in _VDISaturationDict:
            methods.append(VDI_TABULAR)
            Ts, props = VDI_tabular_data(self.CASRN, 'Volume (l)')
            self.VDI_Tmin = Ts[0]
            self.VDI_Tmax = Ts[-1]
            self.tabular_data[VDI_TABULAR] = (Ts, props)
            Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
        if self.Tc and self.CASRN in COSTALD_data.index:
            methods.append(HTCOSTALDFIT)
            self.COSTALD_Vchar = float(COSTALD_data.at[self.CASRN, 'Vchar'])
            self.COSTALD_omega_SRK = float(COSTALD_data.at[self.CASRN, 'omega_SRK'])
            Tmins.append(0); Tmaxs.append(self.Tc)
        if self.Tc and self.Pc and self.CASRN in COSTALD_data.index and not np.isnan(COSTALD_data.at[self.CASRN, 'Z_RA']):
            methods.append(RACKETTFIT)
            self.RACKETT_Z_RA = float(COSTALD_data.at[self.CASRN, 'Z_RA'])
            Tmins.append(0); Tmaxs.append(self.Tc)
        if self.CASRN in CRC_inorg_l_const_data.index:
            methods.append(CRC_INORG_L_CONST)
            self.CRC_INORG_L_CONST_Vm = float(CRC_inorg_l_const_data.at[self.CASRN, 'Vm'])
            # Roughly data at STP; not guaranteed however; not used for Trange
        if all((self.Tc, self.Vc, self.Zc)):
            methods.append(YEN_WOODS_SAT)
            Tmins.append(0); Tmaxs.append(self.Tc)
        if all((self.Tc, self.Pc, self.Zc)):
            methods.append(RACKETT)
            Tmins.append(0); Tmaxs.append(self.Tc)
        if all((self.Tc, self.Pc, self.omega)):
            methods.append(YAMADA_GUNN)
            methods.append(BHIRUD_NORMAL)
            Tmins.append(0); Tmaxs.append(self.Tc)
        if all((self.Tc, self.Vc, self.omega)):
            methods.append(TOWNSEND_HALES)
            methods.append(HTCOSTALD)
            methods.append(MMSNM0)
            if self.CASRN in SNM0_data.index:
                methods.append(MMSNM0FIT)
                self.SNM0_delta_SRK = float(SNM0_data.at[self.CASRN, 'delta_SRK'])
            Tmins.append(0); Tmaxs.append(self.Tc)
        if all((self.Tc, self.Vc, self.omega, self.Tb, self.MW)):
            methods.append(CAMPBELL_THODOS)
            Tmins.append(0); Tmaxs.append(self.Tc)
        if all((self.Tc, self.Pc, self.omega)):
            methods_P.append(COSTALD_COMPRESSED)
            if self.eos:
                methods_P.append(EOS)

        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)

        self.all_methods = set(methods)
        self.all_methods_P = set(methods_P)

    def calculate(self, T, method):
        r'''Method to calculate low-pressure liquid molar volume at tempearture
        `T` with a given method.

        This method has no exception handling; see `T_dependent_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate molar volume, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Vm : float
            Molar volume of the liquid at T and a low pressure, [m^3/mol]
        '''
        if method == RACKETT:
            Vm = Rackett(T, self.Tc, self.Pc, self.Zc)
        elif method == YAMADA_GUNN:
            Vm = Yamada_Gunn(T, self.Tc, self.Pc, self.omega)
        elif method == BHIRUD_NORMAL:
            Vm = Bhirud_normal(T, self.Tc, self.Pc, self.omega)
        elif method == TOWNSEND_HALES:
            Vm = Townsend_Hales(T, self.Tc, self.Vc, self.omega)
        elif method == HTCOSTALD:
            Vm = COSTALD(T, self.Tc, self.Vc, self.omega)
        elif method == YEN_WOODS_SAT:
            Vm = Yen_Woods_saturation(T, self.Tc, self.Vc, self.Zc)
        elif method == MMSNM0:
            Vm = SNM0(T, self.Tc, self.Vc, self.omega)
        elif method == MMSNM0FIT:
            Vm = SNM0(T, self.Tc, self.Vc, self.omega, self.SNM0_delta_SRK)
        elif method == CAMPBELL_THODOS:
            Vm = Campbell_Thodos(T, self.Tb, self.Tc, self.Pc, self.MW, self.dipole)
        elif method == HTCOSTALDFIT:
            Vm = COSTALD(T, self.Tc, self.COSTALD_Vchar, self.COSTALD_omega_SRK)
        elif method == RACKETTFIT:
            Vm = Rackett(T, self.Tc, self.Pc, self.RACKETT_Z_RA)
        elif method == PERRYDIPPR:
            A, B, C, D = self.DIPPR_coeffs
            Vm = 1./EQ105(T, A, B, C, D)
        elif method == CRC_INORG_L:
            rho = CRC_inorganic(T, self.CRC_INORG_L_rho, self.CRC_INORG_L_k, self.CRC_INORG_L_Tm)
            Vm = rho_to_Vm(rho, self.CRC_INORG_L_MW)
        elif method == VDI_PPDS:
            A, B, C, D = self.VDI_PPDS_coeffs
            tau = 1. - T/self.VDI_PPDS_Tc
            rho = self.VDI_PPDS_rhoc + A*tau**0.35 + B*tau**(2/3.) + C*tau + D*tau**(4/3.)
            Vm = rho_to_Vm(rho, self.VDI_PPDS_MW)
        elif method == CRC_INORG_L_CONST:
            Vm = self.CRC_INORG_L_CONST_Vm
        elif method == COOLPROP:
            Vm = 1./CoolProp_T_dependent_property(T, self.CASRN, 'DMOLAR', 'l')
        elif method in self.tabular_data:
            Vm = self.interpolate(T, method)
        return Vm

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent liquid molar volume at
        temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see `TP_dependent_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate molar volume, [K]
        P : float
            Pressure at which to calculate molar volume, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Vm : float
            Molar volume of the liquid at T and P, [m^3/mol]
        '''
        if method == COSTALD_COMPRESSED:
            Vm = self.T_dependent_property(T)
            Psat = self.Psat(T) if hasattr(self.Psat, '__call__') else self.Psat
            Vm = COSTALD_compressed(T, P, Psat, self.Tc, self.Pc, self.omega, Vm)
        elif method == COOLPROP:
            Vm = 1./PropsSI('DMOLAR', 'T', T, 'P', P, self.CASRN)
        elif method == EOS:
            self.eos[0] = self.eos[0].to_TP(T=T, P=P)
            Vm = self.eos[0].V_l
        elif method in self.tabular_data:
            Vm = self.interpolate_P(T, P, method)
        return Vm

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a method. Follows the given
        ranges for all coefficient-based methods. For CSP methods, the models
        are considered valid from 0 K to the critical point. For tabular data,
        extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the extrapolation
        is considered valid for all temperatures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.

        **BHIRUD_NORMAL** behaves poorly at low temperatures and is not used
        under 0.35Tc. The constant value available for inorganic chemicals,
        from method **CRC_INORG_L_CONST**, is considered valid for all
        temperatures.

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
        if method == PERRYDIPPR:
            if T < self.DIPPR_Tmin or T > self.DIPPR_Tmax:
                validity = False
        elif method == VDI_PPDS:
            validity = T <= self.VDI_PPDS_Tc 
        elif method == CRC_INORG_L:
            if T < self.CRC_INORG_L_Tm or T > self.CRC_INORG_L_Tmax:
                validity = False
        elif method == COOLPROP:
            if T < self.CP_f.Tmin or T < self.CP_f.Tt or T > self.CP_f.Tc:
                return False
        elif method in [RACKETT, YAMADA_GUNN, TOWNSEND_HALES,
                        HTCOSTALD, YEN_WOODS_SAT, MMSNM0, MMSNM0FIT,
                        CAMPBELL_THODOS, HTCOSTALDFIT, RACKETTFIT]:
            if T >= self.Tc:
                validity = False
        elif method == BHIRUD_NORMAL:
            if T/self.Tc < 0.35:
                validity = False
            # Has bad interpolation behavior lower than roughly this
        elif method == CRC_INORG_L_CONST:
            pass  # Weird range, consider valid for all conditions
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity

    def test_method_validity_P(self, T, P, method):
        r'''Method to check the validity of a high-pressure method. For
        **COOLPROP**, the fluid must be both a liquid and under the maximum
        pressure of the fluid's EOS. **COSTALD_COMPRESSED** is considered
        valid for all values of temperature and pressure. However, it very
        often will not actually work, due to the form of the polynomial in
        terms of Tr, the result of which is raised to a negative power.
        For tabular data, extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the
        extrapolation is considered valid for all temperatures and pressures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.

        Parameters
        ----------
        T : float
            Temperature at which to test the method, [K]
        P : float
            Pressure at which to test the method, [Pa]
        method : str
            Name of the method to test

        Returns
        -------
        validity : bool
            Whether or not a method is valid
        '''
        validity = True
        if method == COSTALD_COMPRESSED:
            pass
        elif method == COOLPROP:
            validity = PhaseSI('T', T, 'P', P, self.CASRN) == 'liquid'
        elif method == EOS:
            self.eos[0] = self.eos[0].to_TP(T=T, P=P)
            validity = hasattr(self.eos[0], 'V_l')
        elif method in self.tabular_data:
            if not self.tabular_extrapolation_permitted:
                Ts, Ps, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1] or P < Ps[0] or P > Ps[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity


def COSTALD_compressed(T, P, Psat, Tc, Pc, omega, Vs):
    r'''Calculates compressed-liquid volume, using the COSTALD [1]_ CSP
    method and a chemical's critical properties.

    The molar volume of a liquid is given by:

    .. math::
        V = V_s\left( 1 - C \ln \frac{B + P}{B + P^{sat}}\right)

        \frac{B}{P_c} = -1 + a\tau^{1/3} + b\tau^{2/3} + d\tau + e\tau^{4/3}

        e = \exp(f + g\omega_{SRK} + h \omega_{SRK}^2)

        C = j + k \omega_{SRK}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    P : float
        Pressure of fluid [Pa]
    Psat : float
        Saturation pressure of the fluid [Pa]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]
    omega : float
        (ideally SRK) Acentric factor for fluid, [-]
        This parameter is alternatively a fit parameter.
    Vs : float
        Saturation liquid volume, [m^3/mol]

    Returns
    -------
    V_dense : float
        High-pressure liquid volume, [m^3/mol]

    Notes
    -----
    Original equation was in terms of density, but it is converted here.

    The example is from DIPPR, and exactly correct.
    This is DIPPR Procedure 4C: Method for Estimating the Density of Pure
    Organic Liquids under Pressure.

    Examples
    --------
    >>> COSTALD_compressed(303., 9.8E7, 85857.9, 466.7, 3640000.0, 0.281, 0.000105047)
    9.287482879788506e-05

    References
    ----------
    .. [1]  Thomson, G. H., K. R. Brobst, and R. W. Hankinson. "An Improved
       Correlation for Densities of Compressed Liquids and Liquid Mixtures."
       AIChE Journal 28, no. 4 (July 1, 1982): 671-76. doi:10.1002/aic.690280420
    '''
    a = -9.070217
    b = 62.45326
    d = -135.1102
    f = 4.79594
    g = 0.250047
    h = 1.14188
    j = 0.0861488
    k = 0.0344483
    tau = 1 - T/Tc
    e = exp(f + g*omega + h*omega**2)
    C = j + k*omega
    B = Pc*(-1 + a*tau**(1/3.) + b*tau**(2/3.) + d*tau + e*tau**(4/3.))
    return Vs*(1 - C*log((B + P)/(B + Psat)))


### Liquid Mixtures

def Amgat(xs, Vms):
    r'''Calculate mixture liquid density using the Amgat mixing rule.
    Highly inacurate, but easy to use. Assumes idea liquids with
    no excess volume. Average molecular weight should be used with it to obtain
    density.

    .. math::
        V_{mix} = \sum_i x_i V_i

    or in terms of density:

    .. math::

        \rho_{mix} = \sum\frac{x_i}{\rho_i}

    Parameters
    ----------
    xs: array
        Mole fractions of each component, []
    Vms : array
        Molar volumes of each fluids at conditions [m^3/mol]

    Returns
    -------
    Vm : float
        Mixture liquid volume [m^3/mol]

    Notes
    -----
    Units are that of the given volumes.
    It has been suggested to use this equation with weight fractions,
    but the results have been less accurate.

    Examples
    --------
    >>> Amgat([0.5, 0.5], [4.057e-05, 5.861e-05])
    4.9590000000000005e-05
    '''
    if not none_and_length_check([xs, Vms]):
        raise Exception('Function inputs are incorrect format')
    return mixing_simple(xs, Vms)


def Rackett_mixture(T, xs, MWs, Tcs, Pcs, Zrs):
    r'''Calculate mixture liquid density using the Rackett-derived mixing rule
    as shown in [2]_.

    .. math::
        V_m = \sum_i\frac{x_i T_{ci}}{MW_i P_{ci}} Z_{R,m}^{(1 + (1 - T_r)^{2/7})} R \sum_i x_i MW_i

    Parameters
    ----------
    T : float
        Temperature of liquid [K]
    xs: list
        Mole fractions of each component, []
    MWs : list
        Molecular weights of each component [g/mol]
    Tcs : list
        Critical temperatures of each component [K]
    Pcs : list
        Critical pressures of each component [Pa]
    Zrs : list
        Rackett parameters of each component []

    Returns
    -------
    Vm : float
        Mixture liquid volume [m^3/mol]

    Notes
    -----
    Model for pure compounds in [1]_ forms the basis for this model, shown in
    [2]_. Molecular weights are used as weighing by such has been found to
    provide higher accuracy in [2]_. The model can also be used without
    molecular weights, but results are somewhat different.

    As with the Rackett model, critical compressibilities may be used if
    Rackett parameters have not been regressed.

    Critical mixture temperature, and compressibility are all obtained with
    simple mixing rules.

    Examples
    --------
    Calculation in [2]_ for methanol and water mixture. Result matches example.

    >>> Rackett_mixture(T=298., xs=[0.4576, 0.5424], MWs=[32.04, 18.01], Tcs=[512.58, 647.29], Pcs=[8.096E6, 2.209E7], Zrs=[0.2332, 0.2374])
    2.625288603174508e-05

    References
    ----------
    .. [1] Rackett, Harold G. "Equation of State for Saturated Liquids."
       Journal of Chemical & Engineering Data 15, no. 4 (1970): 514-517.
       doi:10.1021/je60047a012
    .. [2] Danner, Ronald P, and Design Institute for Physical Property Data.
       Manual for Predicting Chemical Process Design Data. New York, N.Y, 1982.
    '''
    if not none_and_length_check([xs, MWs, Tcs, Pcs, Zrs]):
        raise Exception('Function inputs are incorrect format')
    Tc = mixing_simple(xs, Tcs)
    Zr = mixing_simple(xs, Zrs)
    MW = mixing_simple(xs, MWs)
    Tr = T/Tc
    bigsum = sum(xs[i]*Tcs[i]/Pcs[i]/MWs[i] for i in range(len(xs)))
    return (R*bigsum*Zr**(1. + (1. - Tr)**(2/7.)))*MW


def COSTALD_mixture(xs, T, Tcs, Vcs, omegas):
    r'''Calculate mixture liquid density using the COSTALD CSP method.

    A popular and accurate estimation method. If possible, fit parameters are
    used; alternatively critical properties work well.

    The mixing rules giving parameters for the pure component COSTALD
    equation are:

    .. math::
        T_{cm} = \frac{\sum_i\sum_j x_i x_j (V_{ij}T_{cij})}{V_m}

        V_m = 0.25\left[ \sum x_i V_i + 3(\sum x_i V_i^{2/3})(\sum_i x_i V_i^{1/3})\right]

        V_{ij}T_{cij} = (V_iT_{ci}V_{j}T_{cj})^{0.5}

        \omega = \sum_i z_i \omega_i

    Parameters
    ----------
    xs: list
        Mole fractions of each component
    T : float
        Temperature of fluid [K]
    Tcs : list
        Critical temperature of fluids [K]
    Vcs : list
        Critical volumes of fluids [m^3/mol].
        This parameter is alternatively a fit parameter
    omegas : list
        (ideally SRK) Acentric factor of all fluids, [-]
        This parameter is alternatively a fit parameter.

    Returns
    -------
    Vs : float
        Saturation liquid mixture volume

    Notes
    -----
    Range: 0.25 < Tr < 0.95, often said to be to 1.0
    No example has been found.
    Units are that of critical or fit constant volume.

    Examples
    --------
    >>> COSTALD_mixture([0.4576, 0.5424], 298.,  [512.58, 647.29],[0.000117, 5.6e-05], [0.559,0.344] )
    2.706588773271354e-05

    References
    ----------
    .. [1] Hankinson, Risdon W., and George H. Thomson. "A New Correlation for
       Saturated Densities of Liquids and Their Mixtures." AIChE Journal
       25, no. 4 (1979): 653-663. doi:10.1002/aic.690250412
    '''
    cmps = range(len(xs))
    if not none_and_length_check([xs, Tcs, Vcs, omegas]):
        raise Exception('Function inputs are incorrect format')
    sum1 = sum([xi*Vci for xi, Vci in zip(xs, Vcs)])
    sum2 = sum([xi*Vci**(2/3.) for xi, Vci in zip(xs, Vcs)])
    sum3 = sum([xi*Vci**(1/3.) for xi, Vci in zip(xs, Vcs)])
    Vm = 0.25*(sum1 + 3.*sum2*sum3)
    VijTcij = [[(Tcs[i]*Tcs[j]*Vcs[i]*Vcs[j])**0.5 for j in cmps] for i in cmps]
    omega = mixing_simple(xs, omegas)
    Tcm = sum([xs[i]*xs[j]*VijTcij[i][j]/Vm for j in cmps for i in cmps])
    return COSTALD(T, Tcm, Vm, omega)


NONE = 'None'
LALIBERTE = 'Laliberte'
COSTALD_MIXTURE = 'COSTALD mixture'
COSTALD_MIXTURE_FIT = 'COSTALD mixture parameters'
SIMPLE = 'SIMPLE'
RACKETT = 'RACKETT'
RACKETT_PARAMETERS = 'RACKETT Parameters'
volume_liquid_mixture_methods = [LALIBERTE, SIMPLE, COSTALD_MIXTURE_FIT, RACKETT_PARAMETERS, COSTALD, RACKETT]


class VolumeLiquidMixture(MixtureProperty):
    '''Class for dealing with the molar volume of a liquid mixture as a   
    function of temperature, pressure, and composition.
    Consists of one electrolyte-specific method, four corresponding states
    methods which do not use pure-component volumes, and one mole-weighted
    averaging method.
    
    Prefered method is **SIMPLE**, or **Laliberte** if the mixture is aqueous
    and has electrolytes.  
        
    Parameters
    ----------
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]
    Tcs : list[float], optional
        Critical temperatures of all species in the mixture, [K]
    Pcs : list[float], optional
        Critical pressures of all species in the mixture, [Pa]
    Vcs : list[float], optional
        Critical molar volumes of all species in the mixture, [m^3/mol]
    Zcs : list[float], optional
        Critical compressibility factors of all species in the mixture, [Pa]
    omegas : list[float], optional
        Accentric factors of all species in the mixture, [-]                 
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
    VolumeLiquids : list[VolumeLiquid], optional
        VolumeLiquid objects created for all species in the mixture,  
        normally created by :obj:`thermo.chemical.Chemical`.
                 
    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`volume_liquid_mixture_methods`.

    **Laliberte**:
        Aqueous electrolyte model equation with coefficients; see
        :obj:`thermo.electrochem.Laliberte_density` for more details.
    **COSTALD mixture**:
        CSP method described in :obj:`COSTALD_mixture`.
    **COSTALD mixture parameters**:
        CSP method described in :obj:`COSTALD_mixture`, with two mixture 
        composition independent fit coefficients, `Vc` and `omega`.
    **RACKETT**:
        CSP method described in :obj:`Rackett_mixture`.
    **RACKETT Parameters**:
        CSP method described in :obj:`Rackett_mixture`, but with a mixture
        independent fit coefficient for compressibility factor for each species.
    **SIMPLE**:
        Linear mole fraction mixing rule described in 
        :obj:`thermo.utils.mixing_simple`; also known as Amgat's law.

    See Also
    --------
    

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    name = 'Liquid volume'
    units = 'm^3/mol'
    property_min = 0
    '''Mimimum valid value of liquid molar volume. It should normally occur at the
    triple point, and be well above this.'''
    property_max = 2e-3
    '''Maximum valid value of liquid molar volume. Generous limit.'''
                            
    ranked_methods = [LALIBERTE, SIMPLE, COSTALD_MIXTURE_FIT, 
                      RACKETT_PARAMETERS, COSTALD_MIXTURE, RACKETT]

    def __init__(self, MWs=[], Tcs=[], Pcs=[], Vcs=[], Zcs=[], omegas=[], 
                 CASs=[], VolumeLiquids=[]):
        self.MWs = MWs
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.Vcs = Vcs
        self.Zcs = Zcs
        self.omegas = omegas
        self.CASs = CASs
        self.VolumeLiquids = VolumeLiquids

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        liquid molar volume under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        liquid molar volume above.'''

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
        methods = [SIMPLE]        
        
        if none_and_length_check([self.Tcs, self.Vcs, self.omegas, self.CASs]):
            methods.append(COSTALD_MIXTURE)
            if all([i in COSTALD_data.index for i in self.CASs]):
                self.COSTALD_Vchars = [COSTALD_data.at[CAS, 'Vchar'] for CAS in self.CASs]
                self.COSTALD_omegas = [COSTALD_data.at[CAS, 'omega_SRK'] for CAS in self.CASs]
                methods.append(COSTALD_MIXTURE_FIT)
            
        if none_and_length_check([self.MWs, self.Tcs, self.Pcs, self.Zcs, self.CASs]):
            methods.append(RACKETT)
            if all([CAS in COSTALD_data.index for CAS in self.CASs]):
                Z_RAs = [COSTALD_data.at[CAS, 'Z_RA'] for CAS in self.CASs]
                if not any(np.isnan(Z_RAs)):
                    self.Z_RAs = Z_RAs
                    methods.append(RACKETT_PARAMETERS)
        
        if len(self.CASs) > 1 and '7732-18-5' in self.CASs:
            wCASs = [i for i in self.CASs if i != '7732-18-5'] 
            if all([i in _Laliberte_Density_ParametersDict for i in wCASs]):
                methods.append(LALIBERTE)
                self.wCASs = wCASs
                self.index_w = self.CASs.index('7732-18-5')
        self.all_methods = set(methods)
            
    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate molar volume of a liquid mixture at 
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
        Vm : float
            Molar volume of the liquid mixture at the given conditions, 
            [m^3/mol]
        '''
        if method == SIMPLE:
            Vms = [i(T, P) for i in self.VolumeLiquids]
            return Amgat(zs, Vms)
        elif method == COSTALD_MIXTURE:
            return COSTALD_mixture(zs, T, self.Tcs, self.Vcs, self.omegas)
        elif method == COSTALD_MIXTURE_FIT:
            return COSTALD_mixture(zs, T, self.Tcs, self.COSTALD_Vchars, self.COSTALD_omegas)
        elif method == RACKETT:
            return Rackett_mixture(T, zs, self.MWs, self.Tcs, self.Pcs, self.Zcs)
        elif method == RACKETT_PARAMETERS:
            return Rackett_mixture(T, zs, self.MWs, self.Tcs, self.Pcs, self.Z_RAs)
        elif method == LALIBERTE:
            ws = list(ws) ; ws.pop(self.index_w)
            rho = Laliberte_density(T, ws, self.wCASs)
            MW = mixing_simple(zs, self.MWs)
            return rho_to_Vm(rho, MW)
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
        if LALIBERTE in self.all_methods:
            # If everything is an electrolyte, accept only it as a method
            if method in self.all_methods:
                return method == LALIBERTE
        if method in self.all_methods:
            return True
        else:
            raise Exception('Method not valid')


### Gases


def ideal_gas(T, P):
    r'''Calculates ideal gas molar volume.
    The molar volume of an ideal gas is given by:

    .. math::
        V = \frac{RT}{P}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    P : float
        Pressure of fluid [Pa]

    Returns
    -------
    V : float
        Gas volume, [m^3/mol]

    Examples
    --------
    >>> ideal_gas(298.15, 101325.)
    0.02446539540458919
    '''
    return R*T/P


#PR = 'PR'
CRC_VIRIAL = 'CRC_VIRIAL'
TSONOPOULOS_EXTENDED = 'TSONOPOULOS_EXTENDED'
TSONOPOULOS = 'TSONOPOULOS'
ABBOTT = 'ABBOTT'
PITZER_CURL = 'PITZER_CURL'
IDEAL = 'IDEAL'
NONE = 'NONE'
volume_gas_methods = [COOLPROP, EOS, CRC_VIRIAL, TSONOPOULOS_EXTENDED, TSONOPOULOS,
                      ABBOTT, PITZER_CURL, IDEAL]
'''Holds all methods available for the VolumeGas class, for use in
iterating over them.'''


class VolumeGas(TPDependentProperty):
    r'''Class for dealing with gas molar volume as a function of
    temperature and pressure.

    All considered methods are both temperature and pressure dependent. Included
    are four CSP methods for calculating second virial coefficients, one
    source of polynomials for calculating second virial coefficients, one
    equation of state (Peng-Robinson), and the ideal gas law.

    Parameters
    ----------
    CASRN : str, optional
        The CAS number of the chemical
    MW : float, optional
        Molecular weight, [g/mol]
    Tc : float, optional
        Critical temperature, [K]
    Pc : float, optional
        Critical pressure, [Pa]
    omega : float, optional
        Acentric factor, [-]
    dipole : float, optional
        Dipole, [debye]

    Notes
    -----
    A string holding each method's name is assigned to the following variables
    in this module, intended as the most convenient way to refer to a method.
    To iterate over all methods, use the list stored in
    :obj:`volume_gas_methods`.

    **PR**:
        Peng-Robinson Equation of State. See the appropriate module for more
        information.
    **CRC_VIRIAL**:
        Short polynomials, for 105 fluids from [1]_.  The full expression is:

        .. math::
            B = \sum_1^4 a_i\left[T_0/298.15-1\right]^{i-1}

    **TSONOPOULOS_EXTENDED**:
        CSP method for second virial coefficients, described in
        :obj:`thermo.virial.BVirial_Tsonopoulos_extended`
    **TSONOPOULOS**:
        CSP method for second virial coefficients, described in
        :obj:`thermo.virial.BVirial_Tsonopoulos`
    **ABBOTT**:
        CSP method for second virial coefficients, described in
        :obj:`thermo.virial.BVirial_Abbott`. This method is the simplest CSP
        method implemented.
    **PITZER_CURL**:
        CSP method for second virial coefficients, described in
        :obj:`thermo.virial.BVirial_Pitzer_Curl`.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [2]_. Very slow, but unparalled in accuracy for pressure
        dependence.

    See Also
    --------
    :obj:`thermo.virial.BVirial_Pitzer_Curl`
    :obj:`thermo.virial.BVirial_Abbott`
    :obj:`thermo.virial.BVirial_Tsonopoulos`
    :obj:`thermo.virial.BVirial_Tsonopoulos_extended`

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    .. [2] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    '''
    name = 'Gas molar volume'
    units = 'mol/m^3'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_P = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default.'''
    property_min = 0
    '''Mimimum valid value of gas molar volume. It should normally be well
    above this.'''
    property_max = 1E10
    '''Maximum valid value of gas molar volume. Set roughly at an ideal gas
    at 1 Pa and 2 billion K.'''

    Pmax = 1E9  # 1 GPa
    '''Maximum pressure at which no method can calculate gas molar volume
    above.'''
    Pmin = 0
    '''Minimum pressure at which no method can calculate gas molar volume
    under.'''
    ranked_methods = []
    '''Default rankings of the low-pressure methods.'''
    ranked_methods_P = [COOLPROP, EOS, TSONOPOULOS_EXTENDED, TSONOPOULOS, ABBOTT,
                        PITZER_CURL, CRC_VIRIAL, IDEAL]
    '''Default rankings of the pressure-dependent methods.'''


    def __init__(self, CASRN='', MW=None, Tc=None, Pc=None, omega=None,
                 dipole=None, eos=None):
        # Only use TPDependentPropoerty functions here
        self.CASRN = CASRN
        self.MW = MW
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.dipole = dipole
        self.eos = eos

        self.Tmin = 0
        '''Minimum temperature at which no method can calculate the
        gas molar volume under.'''
        self.Tmax = 2E9
        '''Maximum temperature at which no method can calculate the
        gas molar volume above.'''

        self.tabular_data = {}
        '''tabular_data, dict: Stored (Ts, properties) for any
        tabular data; indexed by provided or autogenerated name.'''
        self.tabular_data_interpolators = {}
        self.tabular_data_interpolators = {}
        '''tabular_data_interpolators, dict: Stored (extrapolator,
        spline) tuples which are interp1d instances for each set of tabular
        data; indexed by tuple of (name, interpolation_T,
        interpolation_property, interpolation_property_inv) to ensure that
        if an interpolation transform is altered, the old interpolator which
        had been created is no longer used.'''

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

        self.sorted_valid_methods_P = []
        '''sorted_valid_methods_P, list: Stored methods which were found valid
        at a specific temperature; set by `TP_dependent_property`.'''
        self.user_methods_P = []
        '''user_methods_P, list: Stored methods which were specified by the user
        in a ranked order of preference; set by `TP_dependent_property`.'''

        self.all_methods_P = set()
        '''Set of all high-pressure methods available for a given CASRN and
        properties; filled by :obj:`load_all_methods`.'''

        self.load_all_methods()

    def load_all_methods(self):
        r'''Method which picks out coefficients for the specified chemical
        from the various dictionaries and DataFrames storing it. All data is
        stored as attributes. This method also sets obj:`all_methods_P` as a
        set of methods for which the data exists for.

        Called on initialization only. See the source code for the variables at
        which the coefficients are stored. The coefficients can safely be
        altered once the class is initialized. This method can be called again
        to reset the parameters.
        '''
        methods_P = [IDEAL]
        # no point in getting Tmin, Tmax
        if all((self.Tc, self.Pc, self.omega)):
            methods_P.extend([TSONOPOULOS_EXTENDED, TSONOPOULOS, ABBOTT,
                            PITZER_CURL])
            if self.eos:
                methods_P.append(EOS)
        if self.CASRN in CRC_virial_data.index:
            methods_P.append(CRC_VIRIAL)
            self.CRC_VIRIAL_coeffs = _CRC_virial_data_values[CRC_virial_data.index.get_loc(self.CASRN)].tolist()[1:]
        if has_CoolProp and self.CASRN in coolprop_dict:
            methods_P.append(COOLPROP)
            self.CP_f = coolprop_fluids[self.CASRN]
        self.all_methods_P = set(methods_P)

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent gas molar volume at
        temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see `TP_dependent_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate molar volume, [K]
        P : float
            Pressure at which to calculate molar volume, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Vm : float
            Molar volume of the gas at T and P, [m^3/mol]
        '''
        if method == EOS:
            self.eos[0] = self.eos[0].to_TP(T=T, P=P)
            Vm = self.eos[0].V_g
        elif method == TSONOPOULOS_EXTENDED:
            B = BVirial_Tsonopoulos_extended(T, self.Tc, self.Pc, self.omega, dipole=self.dipole)
            Vm = ideal_gas(T, P) + B
        elif method == TSONOPOULOS:
            B = BVirial_Tsonopoulos(T, self.Tc, self.Pc, self.omega)
            Vm = ideal_gas(T, P) + B
        elif method == ABBOTT:
            B = BVirial_Abbott(T, self.Tc, self.Pc, self.omega)
            Vm = ideal_gas(T, P) + B
        elif method == PITZER_CURL:
            B = BVirial_Pitzer_Curl(T, self.Tc, self.Pc, self.omega)
            Vm = ideal_gas(T, P) + B
        elif method == CRC_VIRIAL:
            a1, a2, a3, a4, a5 = self.CRC_VIRIAL_coeffs
            t = 298.15/T - 1.
            B = (a1 + a2*t + a3*t**2 + a4*t**3 + a5*t**4)/1E6
            Vm = ideal_gas(T, P) + B
        elif method == IDEAL:
            Vm = ideal_gas(T, P)
        elif method == COOLPROP:
            Vm = 1./PropsSI('DMOLAR', 'T', T, 'P', P, self.CASRN)
        elif method in self.tabular_data:
            Vm = self.interpolate_P(T, P, method)
        return Vm

    def test_method_validity_P(self, T, P, method):
        r'''Method to check the validity of a pressure and temperature
        dependent gas molar volume method. For the four CSP methods that
        calculate second virial coefficient, the method is considered valid for
        all temperatures and pressures, with validity checking based on the
        result only. For **CRC_VIRIAL**, there is no limit but there should
        be one; at some conditions, a negative volume will result!
        For **COOLPROP**, the fluid must be both a gas at the given conditions
        and under the maximum pressure of the fluid's EOS.

        For the equation of state **PR**, the determined phase must be a gas.
        For **IDEAL**, there are no limits.

        For tabular data, extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the
        extrapolation is considered valid for all temperatures and pressures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.

        Parameters
        ----------
        T : float
            Temperature at which to test the method, [K]
        P : float
            Pressure at which to test the method, [Pa]
        method : str
            Name of the method to test

        Returns
        -------
        validity : bool
            Whether or not a method is valid
        '''
        validity = True
        if T < 0 or P < 0:
            validity = False
        elif method in [TSONOPOULOS_EXTENDED, TSONOPOULOS, ABBOTT,
                        PITZER_CURL, CRC_VIRIAL, IDEAL]:
            pass
            # Would be nice to have a limit on CRC_VIRIAL
        elif method == EOS:
            eos = self.eos[0]
            # Some EOSs do not implement Psat, and so we must assume Vmg is
            # unavailable
            try:
                if T < eos.Tc and P > eos.Psat(T):
                    validity = False
            except:
                validity = False
        elif method == COOLPROP:
            validity = PhaseSI('T', T, 'P', P, self.CASRN) in ['gas', 'supercritical_gas', 'supercritical', 'supercritical_liquid']
        elif method in self.tabular_data:
            if not self.tabular_extrapolation_permitted:
                Ts, Ps, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1] or P < Ps[0] or P > Ps[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity


volume_gas_mixture_methods = [EOS, SIMPLE, IDEAL]



class VolumeGasMixture(MixtureProperty):
    '''Class for dealing with the molar volume of a gas mixture as a   
    function of temperature, pressure, and composition.
    Consists of an equation of state, the ideal gas law, and one mole-weighted
    averaging method.
    
    Prefered method is **EOS**, or **IDEAL** if critical properties of
    components are unavailable.
        
    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
    VolumeGases : list[VolumeGas], optional
        VolumeGas objects created for all species in the mixture,  
        normally created by :obj:`thermo.chemical.Chemical`.
    eos : container[EOS Object], optional
        Equation of state object, normally created by 
        :obj:`thermo.chemical.Mixture`.
                 
    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`volume_gas_mixture_methods`.

    **EOS**:
        Equation of State object, normally provided by 
        :obj:`thermo.chemical.Mixture`. See :obj:`thermo.eos_mix` for more 
        details.
    **SIMPLE**:
        Linear mole fraction mixing rule described in 
        :obj:`thermo.utils.mixing_simple`; more correct than the ideal gas
        law.
    **IDEAL**:
        The ideal gas law.

    See Also
    --------
    ideal_gas
    :obj:`thermo.eos_mix`

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    name = 'Gas volume'
    units = 'm^3/mol'
    property_min = 0
    '''Mimimum valid value of gas molar volume. It should normally be well
    above this.'''
    property_max = 1E10
    '''Maximum valid value of gas molar volume. Set roughly at an ideal gas
    at 1 Pa and 2 billion K.'''
                            
    ranked_methods = [EOS, SIMPLE, IDEAL]

    def __init__(self, eos=None, CASs=[], VolumeGases=[]):
        self.CASs = CASs
        self.VolumeGases = VolumeGases
        self.eos = eos

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        gas molar volume under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        gas molar volume above.'''

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
        methods = [SIMPLE, IDEAL]     
        if self.eos:
            methods.append(EOS)
        self.all_methods = set(methods)
        
    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate molar volume of a gas mixture at 
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
        Vm : float
            Molar volume of the gas mixture at the given conditions, [m^3/mol]
        '''
        if method == SIMPLE:
            Vms = [i(T, P) for i in self.VolumeGases]
            return mixing_simple(zs, Vms)
        elif method == IDEAL:
            return ideal_gas(T, P)
        elif method == EOS:
            self.eos[0] = self.eos[0].to_TP_zs(T=T, P=P, zs=zs)
            return self.eos[0].V_g
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
        if method in self.all_methods:
            return True
        else:
            raise Exception('Method not valid')

### Solids

def Goodman(T, Tt, rhol):
    r'''Calculates solid density at T using the simple relationship
    by a member of the DIPPR.

    The molar volume of a solid is given by:

    .. math::
        \frac{1}{V_m} = \left( 1.28 - 0.16 \frac{T}{T_t}\right)
        \frac{1}{{Vm}_L(T_t)}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    Tt : float
        Triple temperature of fluid [K]
    rhol : float
        Liquid density, [m^3/mol]

    Returns
    -------
    rhos : float
        Solid volume, [m^3/mol]

    Notes
    -----
    Works to the next solid transition temperature or to approximately 0.3Tt.

    Examples
    --------
    >>> Goodman(281.46, 353.43, 7.6326)
    8.797191839062899

    References
    ----------
    .. [1] Goodman, Benjamin T., W. Vincent Wilding, John L. Oscarson, and
       Richard L. Rowley. "A Note on the Relationship between Organic Solid
       Density and Liquid Density at the Triple Point." Journal of Chemical &
       Engineering Data 49, no. 6 (2004): 1512-14. doi:10.1021/je034220e.
    '''
    rhos = (1.28 - 0.16*(T/Tt))*(rhol)
    return rhos


GOODMAN = 'GOODMAN'
CRC_INORG_S = 'CRC_INORG_S'
volume_solid_methods = [GOODMAN, CRC_INORG_S]
'''Holds all methods available for the VolumeSolid class, for use in
iterating over them.'''


class VolumeSolid(TDependentProperty):
    r'''Class for dealing with solid molar volume as a function of temperature.
    Consists of one constant value source, and one simple estimator based on
    liquid molar volume.

    Parameters
    ----------
    CASRN : str, optional
        CAS number
    MW : float, optional
        Molecular weight, [g/mol]
    Tt : float, optional
        Triple temperature
    Vml_Tt : float, optional
        Liquid molar volume at the triple point

    Notes
    -----
    A string holding each method's name is assigned to the following variables
    in this module, intended as the most convenient way to refer to a method.
    To iterate over all methods, use the list stored in
    :obj:`volume_solid_methods`.

    **CRC_INORG_S**:
        Constant values in [1]_, for 1872 chemicals.
    **GOODMAN**:
        Simple method using the liquid molar volume. Good up to 0.3*Tt.
        See :obj:`Goodman` for details.

    See Also
    --------
    Goodman

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics. [Boca Raton, FL]: CRC press, 2014.
    '''
    name = 'Solid molar volume'
    units = 'mol/m^3'
    interpolation_T = None
    '''No interpolation transformation by default.'''
    interpolation_property = None
    '''No interpolation transformation by default.'''
    interpolation_property_inv = None
    '''No interpolation transformation by default.'''
    tabular_extrapolation_permitted = True
    '''Allow tabular extrapolation by default.'''
    property_min = 0
    '''Molar volume cannot be under 0.'''
    property_max = 2e-3
    '''Maximum value of Heat capacity; arbitrarily set to 0.002, as the largest
    in the data is 0.00136.'''

    ranked_methods = [CRC_INORG_S]  # GOODMAN
    '''Default rankings of the available methods.'''

    def __init__(self, CASRN='', MW=None, Tt=None, Vml_Tt=None):
        self.CASRN = CASRN
        self.MW = MW
        self.Tt = Tt
        self.Vml_Tt = Vml_Tt

        self.Tmin = 0
        '''Minimum temperature at which no method can calculate the
        solid molar volume under.'''
        self.Tmax = 1E4
        '''Maximum temperature at which no method can calculate the
        solid molar volume above; assumed 10 000 K even under ultra-high pressure.'''

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
        if self.CASRN in CRC_inorg_s_const_data.index:
            methods.append(CRC_INORG_S)
            self.CRC_INORG_S_Vm = float(CRC_inorg_s_const_data.at[self.CASRN, 'Vm'])
#        if all((self.Tt, self.Vml_Tt, self.MW)):
#            self.rhol_Tt = Vm_to_rho(self.Vml_Tt, self.MW)
#            methods.append(GOODMAN)
        self.all_methods = set(methods)

    def calculate(self, T, method):
        r'''Method to calculate the molar volume of a solid at tempearture `T`
        with a given method.

        This method has no exception handling; see `T_dependent_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate molar volume, [K]
        method : str
            Name of the method to use

        Returns
        -------
        Vms : float
            Molar volume of the solid at T, [m^3/mol]
        '''
        if method == CRC_INORG_S:
            Vms = self.CRC_INORG_S_Vm
#        elif method == GOODMAN:
#            Vms = Goodman(T, self.Tt, self.rhol_Tt)
        elif method in self.tabular_data:
            Vms = self.interpolate(T, method)
        return Vms

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
        if T < 0:
            validity = False
        elif method == CRC_INORG_S:
            pass
            # Assume the solid density value is good at any possible T
#        elif method == GOODMAN:
#            if T < self.Tt*0.3:
#                validity = False
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity


volume_solid_mixture_methods = [SIMPLE]

class VolumeSolidMixture(MixtureProperty):
    '''Class for dealing with the molar volume of a solid mixture as a   
    function of temperature, pressure, and composition.
    Consists of only mole-weighted averaging.
            
    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
    VolumeSolids : list[VolumeSolid], optional
        VolumeSolid objects created for all species in the mixture,  
        normally created by :obj:`thermo.chemical.Chemical`.
                 
    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`volume_solid_mixture_methods`.

    **SIMPLE**:
        Linear mole fraction mixing rule described in 
        :obj:`thermo.utils.mixing_simple`.
    '''
    name = 'Solid molar volume'
    units = 'm^3/mol'
    property_min = 0
    '''Molar volume cannot be under 0.'''
    property_max = 2e-3
    '''Maximum value of Heat capacity; arbitrarily set to 0.002, as the largest
    in the data is 0.00136.'''
                            
    ranked_methods = [SIMPLE]

    def __init__(self, CASs=[], VolumeSolids=[]):
        self.CASs = CASs
        self.VolumeSolids = VolumeSolids

        self.Tmin = 0
        '''Minimum temperature at which no method can calculate the
        solid molar volume under.'''
        self.Tmax = 1E4
        '''Maximum temperature at which no method can calculate the
        solid molar volume above; assumed 10 000 K even under ultra-high 
        pressure.'''

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
        methods = [SIMPLE]     
        self.all_methods = set(methods)
        
    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate molar volume of a solid mixture at 
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
        Vm : float
            Molar volume of the solid mixture at the given conditions,
            [m^3/mol]
        '''
        if method == SIMPLE:
            Vms = [i(T, P) for i in self.VolumeSolids]
            return mixing_simple(zs, Vms)
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
        if method in self.all_methods:
            return True
        else:
            raise Exception('Method not valid')
