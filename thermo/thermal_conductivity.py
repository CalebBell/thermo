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

__all__ = ['Sheffy_Johnson', 'Sato_Riedel', 'Lakshmi_Prasad', 
'Gharagheizi_liquid', 'Nicola_original', 'Nicola', 'Bahadori_liquid', 
'thermal_conductivity_liquid_methods', 'ThermalConductivityLiquid', 'DIPPR9G',
 'Missenard', 'DIPPR9H', 'Filippov', 
 'Eucken', 'Eucken_modified', 'DIPPR9B', 'Chung', 'eli_hanley', 
 'Gharagheizi_gas', 'Bahadori_gas', 'thermal_conductivity_gas_methods', 
 'thermal_conductivity_gas_methods_P', 'ThermalConductivityGas', 
 'stiel_thodos_dense', 'eli_hanley_dense', 'chung_dense', 'Lindsay_Bromley',
 'Perrys2_314', 'Perrys2_315', 'VDI_PPDS_9',
 'VDI_PPDS_10', 'ThermalConductivityGasMixture']

import os
import numpy as np
from scipy.interpolate import interp2d
import pandas as pd

from thermo.utils import R
from thermo.utils import log, exp
from thermo.utils import mixing_simple, none_and_length_check, TPDependentProperty, MixtureProperty, horner
from thermo.miscdata import _VDISaturationDict, VDI_tabular_data
from thermo.coolprop import has_CoolProp, coolprop_dict, coolprop_fluids, CoolProp_T_dependent_property, PropsSI, PhaseSI
from thermo.electrochem import thermal_conductivity_Magomedov, Magomedovk_thermal_cond
from thermo.dippr import EQ100, EQ102


folder = os.path.join(os.path.dirname(__file__), 'Thermal Conductivity')

Perrys2_314 = pd.read_csv(os.path.join(folder, 'Table 2-314 Vapor Thermal Conductivity of Inorganic and Organic Substances.tsv'),
                          sep='\t', index_col=0)
_Perrys2_314_values = Perrys2_314.values

Perrys2_315 = pd.read_csv(os.path.join(folder, 'Table 2-315 Thermal Conductivity of Inorganic and Organic Liquids.tsv'),
                          sep='\t', index_col=0)
_Perrys2_315_values = Perrys2_315.values

VDI_PPDS_9 = pd.read_csv(os.path.join(folder, 'VDI PPDS Thermal conductivity of saturated liquids.tsv'),
                          sep='\t', index_col=0)
_VDI_PPDS_9_values = VDI_PPDS_9.values

VDI_PPDS_10 = pd.read_csv(os.path.join(folder, 'VDI PPDS Thermal conductivity of gases.tsv'),
                          sep='\t', index_col=0)
_VDI_PPDS_10_values = VDI_PPDS_10.values

### Purely CSP Methods - Liquids


def Sheffy_Johnson(T, M, Tm):
    r'''Calculate the thermal conductivity of a liquid as a function of
    temperature using the Sheffy-Johnson (1961) method. Requires
    Temperature, molecular weight, and melting point.

    .. math::
        k = 1.951 \frac{1-0.00126(T-T_m)}{T_m^{0.216}MW^{0.3}}

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    M : float
        Molecular weight of the fluid [g/mol]
    Tm : float
        Melting point of the fluid [K]

    Returns
    -------
    kl : float
        Thermal conductivity of the fluid, W/m/k

    Notes
    -----
    The origin of this equation has been challenging to trace. It is
    presently unknown, and untested.

    Examples
    --------
    >>> Sheffy_Johnson(300, 47, 280)
    0.17740150413112196

    References
    ----------
    .. [1] Scheffy, W. J., and E. F. Johnson. "Thermal Conductivities of
       Liquids at High Temperatures." Journal of Chemical & Engineering Data
       6, no. 2 (April 1, 1961): 245-49. doi:10.1021/je60010a019
    '''
    return 1.951*(1 - 0.00126*(T - Tm))/(Tm**0.216*M**0.3)


def Sato_Riedel(T, M, Tb, Tc):
    r'''Calculate the thermal conductivity of a liquid as a function of
    temperature using the CSP method of Sato-Riedel [1]_, [2]_, published in
    Reid [3]_. Requires temperature, molecular weight, and boiling and critical
    temperatures.

    .. math::
        k = \frac{1.1053}{\sqrt{MW}}\frac{3+20(1-T_r)^{2/3}}
        {3+20(1-T_{br})^{2/3}}

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    M : float
        Molecular weight of the fluid [g/mol]
    Tb : float
        Boiling temperature of the fluid [K]
    Tc : float
        Critical temperature of the fluid [K]

    Returns
    -------
    kl : float
        Estimated liquid thermal conductivity [W/m/k]

    Notes
    -----
    This equation has a complicated history. It is proposed by Reid [3]_.
    Limited accuracy should be expected. Uncheecked.

    Examples
    --------
    >>> Sato_Riedel(300, 47, 390, 520)
    0.21037692461337687

    References
    ----------
    .. [1] Riedel, L.: Chem. Ing. Tech., 21, 349 (1949); 23: 59, 321, 465 (1951)
    .. [2] Maejima, T., private communication, 1973
    .. [3] Properties of Gases and Liquids", 3rd Ed., McGraw-Hill, 1977
    '''
    Tr = T/Tc
    Tbr = Tb/Tc
    return 1.1053*(3. + 20.*(1 - Tr)**(2/3.))*M**-0.5/(3. + 20.*(1 - Tbr)**(2/3.))


def Lakshmi_Prasad(T, M):
    r'''Estimates thermal conductivity of pure liquids as a function of
    temperature using a reference fluid approach. Low accuracy but quick.
    Developed using several organic fluids.

    .. math::
        \lambda = 0.0655-0.0005T + \frac{1.3855-0.00197T}{M^{0.5}}

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    M : float
        Molecular weight of the fluid [g/mol]

    Returns
    -------
    kl : float
        Estimated liquid thermal conductivity [W/m/k]

    Notes
    -----
    This equation returns negative numbers at high T sometimes.
    This equation is one of those implemented by DDBST.
    If this results in a negative thermal conductivity, no value is returned.

    Examples
    --------
    >>> Lakshmi_Prasad(273.15, 100)
    0.013664450000000009

    References
    ----------
    .. [1] Lakshmi, D. S., and D. H. L. Prasad. "A Rapid Estimation Method for
       Thermal Conductivity of Pure Liquids." The Chemical Engineering Journal
       48, no. 3 (April 1992): 211-14. doi:10.1016/0300-9467(92)80037-B
    '''
    return 0.0655 - 0.0005*T + (1.3855 - 0.00197*T)*M**-0.5


def Gharagheizi_liquid(T, M, Tb, Pc, omega):
    r'''Estimates the thermal conductivity of a liquid as a function of
    temperature using the CSP method of Gharagheizi [1]_. A  convoluted
    method claiming high-accuracy and using only statistically significant
    variable following analalysis.

    Requires temperature, molecular weight, boiling temperature and critical
    pressure and acentric factor.

    .. math::
        &k = 10^{-4}\left[10\omega + 2P_c-2T+4+1.908(T_b+\frac{1.009B^2}{MW^2})
        +\frac{3.9287MW^4}{B^4}+\frac{A}{B^8}\right]

        &A = 3.8588MW^8(1.0045B+6.5152MW-8.9756)

        &B = 16.0407MW+2T_b-27.9074

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    M : float
        Molecular weight of the fluid [g/mol]
    Tb : float
        Boiling temperature of the fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    omega : float
        Acentric factor of the fluid [-]

    Returns
    -------
    kl : float
        Estimated liquid thermal conductivity [W/m/k]

    Notes
    -----
    Pressure is internally converted into bar, as used in the original equation.

    This equation was derived with 19000 points representing 1640 unique compounds.

    Examples
    --------
    >>> Gharagheizi_liquid(300, 40, 350, 1E6, 0.27)
    0.2171113029534838

    References
    ----------
    .. [1] Gharagheizi, Farhad, Poorandokht Ilani-Kashkouli, Mehdi Sattari,
        Amir H. Mohammadi, Deresh Ramjugernath, and Dominique Richon.
        "Development of a General Model for Determination of Thermal
        Conductivity of Liquid Chemical Compounds at Atmospheric Pressure."
        AIChE Journal 59, no. 5 (May 1, 2013): 1702-8. doi:10.1002/aic.13938
    '''
    Pc = Pc/1E5
    B = 16.0407*M + 2.*Tb - 27.9074
    A = 3.8588*M**8*(1.0045*B + 6.5152*M - 8.9756)
    kl = 1E-4*(10.*omega + 2.*Pc - 2.*T + 4. + 1.908*(Tb + 1.009*B*B/(M*M))
        + 3.9287*M**4*B**-4 + A*B**-8)
    return kl


def Nicola_original(T, M, Tc, omega, Hfus):
    r'''Estimates the thermal conductivity of a liquid as a function of
    temperature using the CSP method of Nicola [1]_. A  simpler but long
    method claiming high-accuracy and using only statistically significant
    variable following analalysis.

    Requires temperature, molecular weight, critical temperature, acentric
    factor and the heat of vaporization.

    .. math::
        \frac{\lambda}{1 \text{Wm/K}}=-0.5694-0.1436T_r+5.4893\times10^{-10}
        \frac{\Delta_{fus}H}{\text{kmol/J}}+0.0508\omega +
        \left(\frac{1 \text{kg/kmol}}{MW}\right)^{0.0622}

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    M : float
        Molecular weight of the fluid [g/mol]
    Tc : float
        Critical temperature of the fluid [K]
    omega : float
        Acentric factor of the fluid [-]
    Hfus : float
        Heat of fusion of the fluid [J/mol]

    Returns
    -------
    kl : float
        Estimated liquid thermal conductivity [W/m/k]

    Notes
    -----
    A weird statistical correlation. Recent and yet to be reviewed.
    This correlation has been superceded by the author's later work.
    Hfus is internally converted to be in J/kmol.

    Examples
    --------
    >>> Nicola_original(300, 142.3, 611.7, 0.49, 201853)
    0.2305018632230984

    References
    ----------
    .. [1] Nicola, Giovanni Di, Eleonora Ciarrocchi, Mariano Pierantozzi, and
        Roman Stryjek. "A New Equation for the Thermal Conductivity of Organic
        Compounds." Journal of Thermal Analysis and Calorimetry 116, no. 1
        (April 1, 2014): 135-40. doi:10.1007/s10973-013-3422-7
    '''
    Tr = T/Tc
    Hfus = Hfus*1000
    return -0.5694 - 0.1436*Tr + 5.4893E-10*Hfus + 0.0508*omega + (1./M)**0.0622


def Nicola(T, M, Tc, Pc, omega):
    r'''Estimates the thermal conductivity of a liquid as a function of
    temperature using the CSP method of [1]_. A statistically derived
    equation using any correlated terms.

    Requires temperature, molecular weight, critical temperature and pressure,
    and acentric factor.

    .. math::
        \frac{\lambda}{0.5147 W/m/K} = -0.2537T_r+\frac{0.0017Pc}{\text{bar}}
        +0.1501 \omega + \left(\frac{1}{MW}\right)^{-0.2999}

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    M : float
        Molecular weight of the fluid [g/mol]
    Tc : float
        Critical temperature of the fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    omega : float
        Acentric factor of the fluid [-]

    Returns
    -------
    kl : float
        Estimated liquid thermal conductivity [W/m/k]

    Notes
    -----
    A statistical correlation. A revision of an original correlation.

    Examples
    --------
    >>> Nicola(300, 142.3, 611.7, 2110000.0, 0.49)
    0.10863821554584034

    References
    ----------
    .. [1] Di Nicola, Giovanni, Eleonora Ciarrocchi, Gianluca Coccia, and
       Mariano Pierantozzi. "Correlations of Thermal Conductivity for
       Liquid Refrigerants at Atmospheric Pressure or near Saturation."
       International Journal of Refrigeration. 2014.
       doi:10.1016/j.ijrefrig.2014.06.003
    '''
    Tr = T/Tc
    Pc = Pc/1E5
    return 0.5147*(-0.2537*Tr + 0.0017*Pc + 0.1501*omega + (1./M)**0.2999)


def Bahadori_liquid(T, M):
    r'''Estimates the thermal conductivity of parafin liquid hydrocarbons.
    Fits their data well, and is useful as only MW is required.
    X is the Molecular weight, and Y the temperature.

    .. math::
        K = a + bY + CY^2 + dY^3

        a = A_1 + B_1 X + C_1 X^2 + D_1 X^3

        b = A_2 + B_2 X + C_2 X^2 + D_2 X^3

        c = A_3 + B_3 X + C_3 X^2 + D_3 X^3

        d = A_4 + B_4 X + C_4 X^2 + D_4 X^3

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    M : float
        Molecular weight of the fluid [g/mol]

    Returns
    -------
    kl : float
        Estimated liquid thermal conductivity [W/m/k]

    Notes
    -----
    The accuracy of this equation has not been reviewed.

    Examples
    --------
    Data point from [1]_.

    >>> Bahadori_liquid(273.15, 170)
    0.14274278108272603

    References
    ----------
    .. [1] Bahadori, Alireza, and Saeid Mokhatab. "Estimating Thermal
       Conductivity of Hydrocarbons." Chemical Engineering 115, no. 13
       (December 2008): 52-54
    '''
    A = [-6.48326E-2, 2.715015E-3, -1.08580E-5, 9.853917E-9]
    B = [1.565612E-2, -1.55833E-4, 5.051114E-7, -4.68030E-10]
    C = [-1.80304E-4, 1.758693E-6, -5.55224E-9, 5.201365E-12]
    D = [5.880443E-7, -5.65898E-9, 1.764384E-11, -1.65944E-14]
    X, Y = M, T
    a = A[0] + B[0]*X + C[0]*X**2 + D[0]*X**3
    b = A[1] + B[1]*X + C[1]*X**2 + D[1]*X**3
    c = A[2] + B[2]*X + C[2]*X**2 + D[2]*X**3
    d = A[3] + B[3]*X + C[3]*X**2 + D[3]*X**3
    return a + b*Y + c*Y**2 + d*Y**3


VDI_TABULAR = 'VDI_TABULAR'
VDI_PPDS = 'VDI_PPDS'
COOLPROP = 'COOLPROP'
GHARAGHEIZI_L = 'GHARAGHEIZI_L'
NICOLA = 'NICOLA'
NICOLA_ORIGINAL = 'NICOLA_ORIGINAL'
SATO_RIEDEL = 'SATO_RIEDEL'
SHEFFY_JOHNSON = 'SHEFFY_JOHNSON'
BAHADORI_L = 'BAHADORI_L'
LAKSHMI_PRASAD = 'LAKSHMI_PRASAD'
MISSENARD = 'MISSENARD'
NONE = 'NONE'
DIPPR_PERRY_8E = 'DIPPR_PERRY_8E'
NEGLIGIBLE = 'NEGLIGIBLE'
DIPPR_9G = 'DIPPR_9G'

thermal_conductivity_liquid_methods = [COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, 
                                       VDI_TABULAR, GHARAGHEIZI_L, 
                                       SHEFFY_JOHNSON, SATO_RIEDEL,
                                       LAKSHMI_PRASAD, BAHADORI_L,
                                       NICOLA, NICOLA_ORIGINAL]
'''Holds all low-pressure methods available for the ThermalConductivityLiquid
class, for use in iterating over them.'''

thermal_conductivity_liquid_methods_P = [COOLPROP, DIPPR_9G, MISSENARD]
'''Holds all high-pressure methods available for the ThermalConductivityLiquid
class, for use in iterating over them.'''

class ThermalConductivityLiquid(TPDependentProperty):
    r'''Class for dealing with liquid thermal conductivity as a function of
    temperature and pressure.

    For low-pressure (at 1 atm while under the vapor pressure; along the
    saturation line otherwise) liquids, there is one source of tabular
    information, one polynomial-based method, 7 corresponding-states estimators, 
    and the external library CoolProp.

    For high-pressure liquids (also, <1 atm liquids), there are two
    corresponding-states estimator, and the external library CoolProp.

    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
    MW : float, optional
        Molecular weight, [g/mol]
    Tm : float, optional
        Melting point, [K]
    Tb : float, optional
        Boiling point, [K]
    Tc : float, optional
        Critical temperature, [K]
    Pc : float, optional
        Critical pressure, [Pa]
    omega : float, optional
        Acentric factor, [-]
    Hfus : float, optional
        Heat of fusion, [J/mol]

    Notes
    -----
    To iterate over all methods, use the lists stored in
    :obj:`thermal_conductivity_liquid_methods` and
    :obj:`thermal_conductivity_liquid_methods_P` for low and high pressure
    methods respectively.

    Low pressure methods:

    **GHARAGHEIZI_L**:
        CSP method, described in :obj:`Gharagheizi_liquid`.
    **SATO_RIEDEL**:
        CSP method, described in :obj:`Sato_Riedel`.
    **NICOLA**:
        CSP method, described in :obj:`Nicola`.
    **NICOLA_ORIGINAL**:
        CSP method, described in :obj:`Nicola_original`.
    **SHEFFY_JOHNSON**:
        CSP method, described in :obj:`Sheffy_Johnson`.
    **BAHADORI_L**:
        CSP method, described in :obj:`Bahadori_liquid`.
    **LAKSHMI_PRASAD**:
        CSP method, described in :obj:`Lakshmi_Prasad`.
    **DIPPR_PERRY_8E**:
        A collection of 340 coefficient sets from the DIPPR database published
        openly in [3]_. Provides temperature limits for all its fluids. 
        :obj:`thermo.dippr.EQ100` is used for its fluids.
    **VDI_PPDS**:
        Coefficients for a equation form developed by the PPDS, published 
        openly in [2]_. Covers a large temperature range, but does not 
        extrapolate well at very high or very low temperatures. 271 compounds.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow.
    **VDI_TABULAR**:
        Tabular data in [2]_ along the saturation curve; interpolation is as
        set by the user or the default.

    High pressure methods:

    **DIPPR_9G**:
        CSP method, described in :obj:`DIPPR9G`. Calculates a
        low-pressure thermal conductivity first, using `T_dependent_property`.
    **MISSENARD**:
        CSP method, described in :obj:`Missenard`. Calculates a
        low-pressure thermal conductivity first, using `T_dependent_property`.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow, but unparalled in accuracy for pressure
        dependence.

    See Also
    --------
    Sheffy_Johnson
    Sato_Riedel
    Lakshmi_Prasad
    Gharagheizi_liquid
    Nicola_original
    Nicola
    Bahadori_liquid
    DIPPR9G
    Missenard

    References
    ----------
    .. [1] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    .. [3] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    name = 'liquid thermal conductivity'
    units = 'W/m/K'
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
    '''Mimimum valid value of liquid thermal conductivity.'''
    property_max = 10
    '''Maximum valid value of liquid thermal conductivity. Generous limit.'''

    ranked_methods = [COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, VDI_TABULAR,
                      GHARAGHEIZI_L, SHEFFY_JOHNSON, SATO_RIEDEL, 
                      LAKSHMI_PRASAD, BAHADORI_L, NICOLA, NICOLA_ORIGINAL]
    '''Default rankings of the low-pressure methods.'''
    ranked_methods_P = [COOLPROP, DIPPR_9G, MISSENARD]
    '''Default rankings of the high-pressure methods.'''


    def __init__(self, CASRN='', MW=None, Tm=None, Tb=None, Tc=None, Pc=None,
                 omega=None, Hfus=None):
        self.CASRN = CASRN
        self.MW = MW
        self.Tm = Tm
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.Hfus = Hfus

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        liquid thermal conductivity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        liquid thermal conductivity above.'''

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
        methods, methods_P = [], []
        Tmins, Tmaxs = [], []
        if self.CASRN in _VDISaturationDict:
            methods.append(VDI_TABULAR)
            Ts, props = VDI_tabular_data(self.CASRN, 'K (l)')
            self.VDI_Tmin = Ts[0]
            self.VDI_Tmax = Ts[-1]
            self.tabular_data[VDI_TABULAR] = (Ts, props)
            Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
        if has_CoolProp and self.CASRN in coolprop_dict:
            methods.append(COOLPROP); methods_P.append(COOLPROP)
            self.CP_f = coolprop_fluids[self.CASRN]
            Tmins.append(self.CP_f.Tmin); Tmaxs.append(self.CP_f.Tc)
        if self.MW:
            methods.extend([BAHADORI_L, LAKSHMI_PRASAD])
            # Tmin and Tmax are not extended by these simple models, who often
            # give values of 0; BAHADORI_L even has 3 roots.
            # LAKSHMI_PRASAD works down to 0 K, and has an upper limit of
            # 50.0*(131.0*sqrt(M) + 2771.0)/(50.0*M**0.5 + 197.0)
            # where it becomes 0.
        if self.CASRN in Perrys2_315.index:
            methods.append(DIPPR_PERRY_8E)
            _, C1, C2, C3, C4, C5, self.Perrys2_315_Tmin, self.Perrys2_315_Tmax = _Perrys2_315_values[Perrys2_315.index.get_loc(self.CASRN)].tolist()
            self.Perrys2_315_coeffs = [C1, C2, C3, C4, C5]
            Tmins.append(self.Perrys2_315_Tmin); Tmaxs.append(self.Perrys2_315_Tmax)
        if self.CASRN in VDI_PPDS_9.index:
            _,  A, B, C, D, E = _VDI_PPDS_9_values[VDI_PPDS_9.index.get_loc(self.CASRN)].tolist()
            self.VDI_PPDS_coeffs = [A, B, C, D, E]
            self.VDI_PPDS_coeffs.reverse()
            methods.append(VDI_PPDS)
        if all([self.MW, self.Tm]):
            methods.append(SHEFFY_JOHNSON)
            Tmins.append(0); Tmaxs.append(self.Tm + 793.65)
            # Works down to 0, has a nice limit at T = Tm+793.65 from Sympy
        if all([self.Tb, self.Pc, self.omega]):
            methods.append(GHARAGHEIZI_L)
            Tmins.append(self.Tb); Tmaxs.append(self.Tc)
            # Chosen as the model is weird
        if all([self.Tc, self.Pc, self.omega]):
            methods.append(NICOLA)
        if all([self.Tb, self.Tc]):
            methods.append(SATO_RIEDEL)
        if all([self.Hfus, self.Tc, self.omega]):
            methods.append(NICOLA_ORIGINAL)
        if all([self.Tc, self.Pc]):
            methods_P.extend([DIPPR_9G, MISSENARD])
        self.all_methods = set(methods)
        self.all_methods_P = set(methods_P)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)

    def calculate(self, T, method):
        r'''Method to calculate low-pressure liquid thermal conductivity at
        tempearture `T` with a given method.

        This method has no exception handling; see `T_dependent_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature of the liquid, [K]
        method : str
            Name of the method to use

        Returns
        -------
        kl : float
            Thermal conductivity of the liquid at T and a low pressure, [W/m/K]
        '''
        if method == SHEFFY_JOHNSON:
            kl = Sheffy_Johnson(T, self.MW, self.Tm)
        elif method == SATO_RIEDEL:
            kl = Sato_Riedel(T, self.MW, self.Tb, self.Tc)
        elif method == GHARAGHEIZI_L:
            kl = Gharagheizi_liquid(T, self.MW, self.Tb, self.Pc, self.omega)
        elif method == NICOLA:
            kl = Nicola(T, self.MW, self.Tc, self.Pc, self.omega)
        elif method == NICOLA_ORIGINAL:
            kl = Nicola_original(T, self.MW, self.Tc, self.omega, self.Hfus)
        elif method == LAKSHMI_PRASAD:
            kl = Lakshmi_Prasad(T, self.MW)
        elif method == BAHADORI_L:
            kl = Bahadori_liquid(T, self.MW)
        elif method == DIPPR_PERRY_8E:
            kl = EQ100(T, *self.Perrys2_315_coeffs)
        elif method == VDI_PPDS:
            kl = horner(self.VDI_PPDS_coeffs, T)
        elif method == COOLPROP:
            kl = CoolProp_T_dependent_property(T, self.CASRN, 'L', 'l')
        elif method in self.tabular_data:
            kl = self.interpolate(T, method)
        return kl

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent liquid thermal conductivity
        at temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see `TP_dependent_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate liquid thermal conductivity, [K]
        P : float
            Pressure at which to calculate liquid thermal conductivity, [K]
        method : str
            Name of the method to use

        Returns
        -------
        kl : float
            Thermal conductivity of the liquid at T and P, [W/m/K]
        '''
        if method == DIPPR_9G:
            kl = self.T_dependent_property(T)
            kl = DIPPR9G(T, P, self.Tc, self.Pc, kl)
        elif method == MISSENARD:
            kl = self.T_dependent_property(T)
            kl = Missenard(T, P, self.Tc, self.Pc, kl)
        elif method == COOLPROP:
            kl = PropsSI('L', 'T', T, 'P', P, self.CASRN)
        elif method in self.tabular_data:
            kl = self.interpolate_P(T, P, method)
        return kl

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a temperature-dependent
        low-pressure method. For CSP methods, the models **BAHADORI_L**,
        **LAKSHMI_PRASAD**, and **SHEFFY_JOHNSON** are considered valid for all
        temperatures. For methods **GHARAGHEIZI_L**, **NICOLA**,
        and **NICOLA_ORIGINAL**, the methods are considered valid up to 1.5Tc
        and down to 0 K. Method **SATO_RIEDEL** does not work above the
        critical point, so it is valid from 0 K to the critical point.

        For tabular data, extrapolation outside of the range is used if
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
        if method == SATO_RIEDEL:
            if T > self.Tc:
                return False
                # Doesn't run, no lower limit though
        elif method in [GHARAGHEIZI_L, NICOLA, NICOLA_ORIGINAL]:
            if T > self.Tc*1.5:
                return False
            # No lower limit, give a wide margin of acceptability here
        elif method == DIPPR_PERRY_8E:
            if T < self.Perrys2_315_Tmin or T > self.Perrys2_315_Tmax:
                return False
        elif method in [BAHADORI_L, LAKSHMI_PRASAD, SHEFFY_JOHNSON]:
            pass
            # no limits at all
        elif method == VDI_PPDS:
            if self.Tc and T > self.Tc:
                return False
        elif method == COOLPROP:
            if T < self.CP_f.Tt or T > self.CP_f.Tc:
                return False
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    return False
        else:
            raise Exception('Method not valid')
        return True

    def test_method_validity_P(self, T, P, method):
        r'''Method to check the validity of a high-pressure method. For
        **COOLPROP**, the fluid must be both a liquid and under the maximum
        pressure of the fluid's EOS. **MISSENARD** has defined limits;
        between 0.5Tc and 0.8Tc, and below 200Pc. The CSP method **DIPPR_9G**
        is considered valid for all temperatures and pressures.

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
        if method == MISSENARD:
            if T/self.Tc < 0.5 or T/self.Tc > 0.8 or P/self.Pc > 200:
                validity = False
        elif method == DIPPR_9G:
            if T < 0 or P < 0:
                validity = False
        elif method == COOLPROP:
            validity = PhaseSI('T', T, 'P', P, self.CASRN) in ['liquid', 'supercritical_liquid']
        elif method in self.tabular_data:
            if not self.tabular_extrapolation_permitted:
                Ts, Ps, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1] or P < Ps[0] or P > Ps[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity


### Thermal Conductivity of Dense Liquids

def DIPPR9G(T, P, Tc, Pc, kl):
    r'''Adjustes for pressure the thermal conductivity of a liquid using an
    emperical formula based on [1]_, but as given in [2]_.

    .. math::
        k = k^* \left[ 0.98 + 0.0079 P_r T_r^{1.4} + 0.63 T_r^{1.2}
        \left( \frac{P_r}{30 + P_r}\right)\right]

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    P : float
        Pressure of fluid [Pa]
    Tc: float
        Critical point of fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    kl : float
        Thermal conductivity of liquid at 1 atm or saturation, [W/m/K]

    Returns
    -------
    kl_dense : float
        Thermal conductivity of liquid at P, [W/m/K]

    Notes
    -----
    This equation is entrely dimensionless; all dimensions cancel.
    The original source has not been reviewed.

    This is DIPPR Procedure 9G: Method for the Thermal Conductivity of Pure
    Nonhydrocarbon Liquids at High Pressures

    Examples
    --------
    From [2]_, for butyl acetate.

    >>> DIPPR9G(515.05, 3.92E7, 579.15, 3.212E6, 7.085E-2)
    0.0864419738671184

    References
    ----------
    .. [1] Missenard, F. A., Thermal Conductivity of Organic Liquids of a
       Series or a Group of Liquids , Rev. Gen.Thermodyn., 101 649 (1970).
    .. [2] Danner, Ronald P, and Design Institute for Physical Property Data.
       Manual for Predicting Chemical Process Design Data. New York, N.Y, 1982.
    '''
    Tr = T/Tc
    Pr = P/Pc
    return kl*(0.98 + 0.0079*Pr*Tr**1.4 + 0.63*Tr**1.2*(Pr/(30. + Pr)))


Trs_Missenard = [0.8, 0.7, 0.6, 0.5]
Prs_Missenard = [1, 5, 10, 50, 100, 200]
Qs_Missenard = np.array([[0.036, 0.038, 0.038, 0.038, 0.038, 0.038],
                         [0.018, 0.025, 0.027, 0.031, 0.032, 0.032],
                         [0.015, 0.020, 0.022, 0.024, 0.025, 0.025],
                         [0.012, 0.0165, 0.017, 0.019, 0.020, 0.020]])
Qfunc_Missenard = interp2d(Prs_Missenard, Trs_Missenard, Qs_Missenard)


def Missenard(T, P, Tc, Pc, kl):
    r'''Adjustes for pressure the thermal conductivity of a liquid using an
    emperical formula based on [1]_, but as given in [2]_.

    .. math::
        \frac{k}{k^*} = 1 + Q P_r^{0.7}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    P : float
        Pressure of fluid [Pa]
    Tc: float
        Critical point of fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    kl : float
        Thermal conductivity of liquid at 1 atm or saturation, [W/m/K]

    Returns
    -------
    kl_dense : float
        Thermal conductivity of liquid at P, [W/m/K]

    Notes
    -----
    This equation is entirely dimensionless; all dimensions cancel.
    An interpolation routine is used here from tabulated values of Q.
    The original source has not been reviewed.

    Examples
    --------
    Example from [2]_, toluene; matches.

    >>> Missenard(304., 6330E5, 591.8, 41E5, 0.129)
    0.2198375777069657

    References
    ----------
    .. [1] Missenard, F. A., Thermal Conductivity of Organic Liquids of a
       Series or a Group of Liquids , Rev. Gen.Thermodyn., 101 649 (1970).
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    Tr = T/Tc
    Pr = P/Pc
    Q = float(Qfunc_Missenard(Pr, Tr))
    return kl*(1. + Q*Pr**0.7)

### Thermal conductivity of liquid mixtures


def DIPPR9H(ws, ks):
    r'''Calculates thermal conductivity of a liquid mixture according to
    mixing rules in [1]_ and also in [2]_.

    .. math::
        \lambda_m = \left( \sum_i w_i \lambda_i^{-2}\right)^{-1/2}

    Parameters
    ----------
    ws : float
        Mass fractions of components
    ks : float
        Liquid thermal conductivites of all components, [W/m/K]

    Returns
    -------
    kl : float
        Thermal conductivity of liquid mixture, [W/m/K]

    Notes
    -----
    This equation is entirely dimensionless; all dimensions cancel.
    The example is from [2]_; all results agree.
    The original source has not been reviewed.

    DIPPR Procedure 9H: Method for the Thermal Conductivity of Nonaqueous Liquid Mixtures

    Average deviations of 3%. for 118 nonaqueous systems with 817 data points.
    Max deviation 20%. According to DIPPR.

    Examples
    --------
    >>> DIPPR9H([0.258, 0.742], [0.1692, 0.1528])
    0.15657104706719646

    References
    ----------
    .. [1] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E. The
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    .. [2] Danner, Ronald P, and Design Institute for Physical Property Data.
       Manual for Predicting Chemical Process Design Data. New York, N.Y, 1982.
    '''
    if not none_and_length_check([ks, ws]):  # check same-length inputs
        raise Exception('Function inputs are incorrect format')
    return sum(ws[i]/ks[i]**2 for i in range(len(ws)))**(-0.5)


def Filippov(ws, ks):
    r'''Calculates thermal conductivity of a binary liquid mixture according to
    mixing rules in [2]_ as found in [1]_.

    .. math::
        \lambda_m = w_1 \lambda_1 + w_2\lambda_2
        - 0.72 w_1 w_2(\lambda_2-\lambda_1)

    Parameters
    ----------
    ws : float
        Mass fractions of components
    ks : float
        Liquid thermal conductivites of all components, [W/m/K]

    Returns
    -------
    kl : float
        Thermal conductivity of liquid mixture, [W/m/K]

    Notes
    -----
    This equation is entirely dimensionless; all dimensions cancel.
    The original source has not been reviewed.
    Only useful for binary mixtures.

    Examples
    --------
    >>> Filippov([0.258, 0.742], [0.1692, 0.1528])
    0.15929167628799998

    References
    ----------
    .. [1] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E. The
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    .. [2] Filippov, L. P.: Vest. Mosk. Univ., Ser. Fiz. Mat. Estestv. Nauk,
       (8I0E): 67-69A955); Chem. Abstr., 50: 8276 A956).
       Filippov, L. P., and N. S. Novoselova: Vestn. Mosk. Univ., Ser. F
       iz. Mat. Estestv.Nauk, CI0B): 37-40A955); Chem. Abstr., 49: 11366 A955).
    '''
    if not none_and_length_check([ks, ws], 2):  # check same-length inputs
        raise Exception('Function inputs are incorrect format')
    return ws[0]*ks[0] + ws[1]*ks[1] - 0.72*ws[0]*ws[1]*(ks[1] - ks[0])


MAGOMEDOV = 'Magomedov'
DIPPR_9H = 'DIPPR9H'
FILIPPOV = 'Filippov'
SIMPLE = 'SIMPLE'

thermal_conductivity_liquid_mixture_methods = [MAGOMEDOV, DIPPR_9H, FILIPPOV, SIMPLE]


class ThermalConductivityLiquidMixture(MixtureProperty):
    '''Class for dealing with thermal conductivity of a liquid mixture as a   
    function of temperature, pressure, and composition.
    Consists of two mixing rule specific to liquid thremal conductivity, one
    coefficient-based method for aqueous electrolytes, and mole weighted 
    averaging. 
         
    Prefered method is :obj:`DIPPR9H` which requires mass
    fractions, and pure component liquid thermal conductivities. This is 
    substantially better than the ideal mixing rule based on mole fractions, 
    **SIMPLE**. **Filippov** is of similar accuracy but applicable to binary
    systems only.
        
    Parameters
    ----------
    CASs : str, optional
        The CAS numbers of all species in the mixture
    ThermalConductivityLiquids : list[ThermalConductivityLiquid], optional
        ThermalConductivityLiquid objects created for all species in the
        mixture, normally created by :obj:`thermo.chemical.Chemical`.

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`thermal_conductivity_liquid_mixture_methods`.

    **DIPPR9H**:
        Mixing rule described in :obj:`DIPPR9H`.
    **Filippov**:
        Mixing rule described in :obj:`Filippov`; for two binary systems only.
    **Magomedov**:
        Coefficient-based method for aqueous electrolytes only, described in
        :obj:`thermo.electrochem.thermal_conductivity_Magomedov`.
    **SIMPLE**:
        Mixing rule described in :obj:`thermo.utils.mixing_simple`.

    See Also
    --------
    DIPPR9H
    Filippov
    thermal_conductivity_Magomedov

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    name = 'liquid thermal conductivity'
    units = 'W/m/K'
    property_min = 0
    '''Mimimum valid value of liquid thermal conductivity.'''
    property_max = 10
    '''Maximum valid value of liquid thermal conductivity. Generous limit.'''
                            
    ranked_methods = [DIPPR_9H, SIMPLE, MAGOMEDOV, FILIPPOV]

    def __init__(self, CASs=[], ThermalConductivityLiquids=[]):
        self.CASs = CASs
        self.ThermalConductivityLiquids = ThermalConductivityLiquids

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        liquid thermal conductivity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        liquid thermal conductivity above.'''

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
        methods = [DIPPR_9H, SIMPLE]        
        if len(self.CASs) == 2:
            methods.append(FILIPPOV)
        if '7732-18-5' in self.CASs and len(self.CASs)>1:
            wCASs = [i for i in self.CASs if i != '7732-18-5']
            if all([i in Magomedovk_thermal_cond.index for i in wCASs]):
                methods.append(MAGOMEDOV)
                self.wCASs = wCASs
                self.index_w = self.CASs.index('7732-18-5')
            
        self.all_methods = set(methods)
        Tmins = [i.Tmin for i in self.ThermalConductivityLiquids if i.Tmin]
        Tmaxs = [i.Tmax for i in self.ThermalConductivityLiquids if i.Tmax]
        if Tmins:
            self.Tmin = max(Tmins)
        if Tmaxs:
            self.Tmax = max(Tmaxs)
        
    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate thermal conductivity of a liquid mixture at 
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
        k : float
            Thermal conductivity of the liquid mixture, [W/m/K]
        '''
        if method == SIMPLE:
            ks = [i(T, P) for i in self.ThermalConductivityLiquids]
            return mixing_simple(zs, ks)
        elif method == DIPPR_9H:
            ks = [i(T, P) for i in self.ThermalConductivityLiquids]
            return DIPPR9H(ws, ks)
        elif method == FILIPPOV:
            ks = [i(T, P) for i in self.ThermalConductivityLiquids]
            return Filippov(ws, ks)
        elif method == MAGOMEDOV:
            k_w = self.ThermalConductivityLiquids[self.index_w](T, P)
            ws = list(ws) ; ws.pop(self.index_w)
            return thermal_conductivity_Magomedov(T, P, ws, self.wCASs, k_w)
        else:
            raise Exception('Method not valid')

    def test_method_validity(self, T, P, zs, ws, method):
        r'''Method to test the validity of a specified method for the given
        conditions. If **Magomedov** is applicable (electrolyte system), no
        other methods are considered viable. Otherwise, there are no easy
        checks that can be performed here.

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
        if MAGOMEDOV in self.all_methods:
            if method in self.all_methods:
                return method == MAGOMEDOV
        if method in [SIMPLE, DIPPR_9H, FILIPPOV]:
            return True
        else:
            raise Exception('Method not valid')


### Thermal Conductivity of Gases

def Eucken(MW, Cvm, mu):
    r'''Estimates the thermal conductivity of a gas as a function of
    temperature using the CSP method of Eucken [1]_.

    .. math::
        \frac{\lambda M}{\eta C_v} = 1 + \frac{9/4}{C_v/R}

    Parameters
    ----------
    MW : float
        Molecular weight of the gas [g/mol]
    Cvm : float
        Molar contant volume heat capacity of the gas [J/mol/K]
    mu : float
        Gas viscosity [Pa*S]

    Returns
    -------
    kg : float
        Estimated gas thermal conductivity [W/m/k]

    Notes
    -----
    Temperature dependence is introduced via heat capacity and viscosity.
    A theoretical equation. No original author located.
    MW internally converted to kg/g-mol.

    Examples
    --------
    2-methylbutane at low pressure, 373.15 K. Mathes calculation in [1]_.

    >>> Eucken(MW=72.151, Cvm=135.9, mu=8.77E-6)
    0.018792644287722975

    References
    ----------
    .. [1] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    MW = MW/1000.
    return (1. + 9/4./(Cvm/R))*mu*Cvm/MW


def Eucken_modified(MW, Cvm, mu):
    r'''Estimates the thermal conductivity of a gas as a function of
    temperature using the Modified CSP method of Eucken [1]_.

    .. math::
        \frac{\lambda M}{\eta C_v} = 1.32 + \frac{1.77}{C_v/R}

    Parameters
    ----------
    MW : float
        Molecular weight of the gas [g/mol]
    Cvm : float
        Molar contant volume heat capacity of the gas [J/mol/K]
    mu : float
        Gas viscosity [Pa*S]

    Returns
    -------
    kg : float
        Estimated gas thermal conductivity [W/m/k]

    Notes
    -----
    Temperature dependence is introduced via heat capacity and viscosity.
    A theoretical equation. No original author located.
    MW internally converted to kg/g-mol.

    Examples
    --------
    2-methylbutane at low pressure, 373.15 K. Mathes calculation in [1]_.

    >>> Eucken_modified(MW=72.151, Cvm=135.9, mu=8.77E-6)
    0.023593536999201956

    References
    ----------
    .. [1] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    MW = MW/1000.
    return (1.32 + 1.77/(Cvm/R))*mu*Cvm/MW


def DIPPR9B(T, MW, Cvm, mu, Tc=None, chemtype=None):
    r'''Calculates the thermal conductivity of a gas using one of several
    emperical equations developed in [1]_, [2]_, and presented in [3]_.

    For monoatomic gases:

    .. math::
        k = 2.5 \frac{\eta C_v}{MW}

    For linear molecules:

    .. math::
        k = \frac{\eta}{MW} \left( 1.30 C_v + 14644.00 - \frac{2928.80}{T_r}\right)

    For nonlinear molecules:

    .. math::
        k = \frac{\eta}{MW}(1.15C_v + 16903.36)

    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    Tc : float
        Critical temperature of the fluid [K]
    MW : float
        Molwcular weight of fluid [g/mol]
    Cvm : float
        Molar heat capacity at constant volume of fluid, [J/mol/K]
    mu : float
        Viscosity of gas, [Pa*S]

    Returns
    -------
    k_g : float
        Thermal conductivity of gas, [W/m/k]

    Notes
    -----
    Tested with DIPPR values.
    Cvm is internally converted to J/kmol/K.

    Examples
    --------
    CO:

    >>> DIPPR9B(200., 28.01, 20.826, 1.277E-5, 132.92, chemtype='linear')
    0.01813208676438415

    References
    ----------
    .. [1] Bromley, LeRoy A., Berkeley. University of California, and U.S.
       Atomic Energy Commission. Thermal Conductivity of Gases at Moderate
       Pressures. UCRL;1852. Berkeley, CA: University of California Radiation
       Laboratory, 1952.
    .. [2] Stiel, Leonard I., and George Thodos. "The Thermal Conductivity of
       Nonpolar Substances in the Dense Gaseous and Liquid Regions." AIChE
       Journal 10, no. 1 (January 1, 1964): 26-30. doi:10.1002/aic.690100114
    .. [3] Danner, Ronald P, and Design Institute for Physical Property Data.
       Manual for Predicting Chemical Process Design Data. New York, N.Y, 1982.
    '''
    Cvm = Cvm*1000.  # J/g/K to J/kmol/K
    if not chemtype:
        chemtype = 'linear'
    if chemtype == 'monoatomic':
        return 2.5*mu*Cvm/MW
    elif chemtype == 'linear':
        Tr = T/Tc
        return mu/MW*(1.30*Cvm + 14644 - 2928.80/Tr)
    elif chemtype == 'nonlinear':
        return mu/MW*(1.15*Cvm + 16903.36)
    else:
        raise Exception('Specified chemical type is not an option')


def Chung(T, MW, Tc, omega, Cvm, mu):
    r'''Estimates the thermal conductivity of a gas as a function of
    temperature using the CSP method of Chung [1]_.

    .. math::
        \frac{\lambda M}{\eta C_v} = \frac{3.75 \Psi}{C_v/R}

        \Psi = 1 + \alpha \left\{[0.215+0.28288\alpha-1.061\beta+0.26665Z]/
        [0.6366+\beta Z + 1.061 \alpha \beta]\right\}

        \alpha = \frac{C_v}{R}-1.5

        \beta = 0.7862-0.7109\omega + 1.3168\omega^2

        Z=2+10.5T_r^2

    Parameters
    ----------
    T : float
        Temperature of the gas [K]
    MW : float
        Molecular weight of the gas [g/mol]
    Tc : float
        Critical temperature of the gas [K]
    omega : float
        Acentric factor of the gas [-]
    Cvm : float
        Molar contant volume heat capacity of the gas [J/mol/K]
    mu : float
        Gas viscosity [Pa*S]

    Returns
    -------
    kg : float
        Estimated gas thermal conductivity [W/m/k]

    Notes
    -----
    MW internally converted to kg/g-mol.

    Examples
    --------
    2-methylbutane at low pressure, 373.15 K. Mathes calculation in [2]_.

    >>> Chung(T=373.15, MW=72.151, Tc=460.4, omega=0.227, Cvm=135.9, mu=8.77E-6)
    0.023015653729496946

    References
    ----------
    .. [1] Chung, Ting Horng, Lloyd L. Lee, and Kenneth E. Starling.
       "Applications of Kinetic Gas Theories and Multiparameter Correlation for
       Prediction of Dilute Gas Viscosity and Thermal Conductivity."
       Industrial & Engineering Chemistry Fundamentals 23, no. 1
       (February 1, 1984): 8-13. doi:10.1021/i100013a002
    .. [2] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    MW = MW/1000.
    alpha = Cvm/R - 1.5
    beta = 0.7862 - 0.7109*omega + 1.3168*omega**2
    Z = 2 + 10.5*(T/Tc)**2
    psi = 1 + alpha*((0.215 + 0.28288*alpha - 1.061*beta + 0.26665*Z)
                      /(0.6366 + beta*Z + 1.061*alpha*beta))
    return 3.75*psi/(Cvm/R)/MW*mu*Cvm


def eli_hanley(T, MW, Tc, Vc, Zc, omega, Cvm):
    r'''Estimates the thermal conductivity of a gas as a function of
    temperature using the reference fluid method of Eli and Hanley [1]_ as
    shown in [2]_.

    .. math::
        \lambda = \lambda^* + \frac{\eta^*}{MW}(1.32)\left(C_v - \frac{3R}{2}\right)

        Tr = \text{min}(Tr, 2)

        \theta = 1 + (\omega-0.011)\left(0.56553 - 0.86276\ln Tr - \frac{0.69852}{Tr}\right)

        \psi = [1 + (\omega - 0.011)(0.38560 - 1.1617\ln Tr)]\frac{0.288}{Z_c}

        f = \frac{T_c}{190.4}\theta

        h = \frac{V_c}{9.92E-5}\psi

        T_0 = T/f

        \eta_0^*(T_0)= \sum_{n=1}^9 C_n T_0^{(n-4)/3}

        \theta_0 = 1944 \eta_0

        \lambda^* = \lambda_0 H

        \eta^* = \eta^*_0 H \frac{MW}{16.04}

        H = \left(\frac{16.04}{MW}\right)^{0.5}f^{0.5}/h^{2/3}

    Parameters
    ----------
    T : float
        Temperature of the gas [K]
    MW : float
        Molecular weight of the gas [g/mol]
    Tc : float
        Critical temperature of the gas [K]
    Vc : float
        Critical volume of the gas [m^3/mol]
    Zc : float
        Critical compressibility of the gas []
    omega : float
        Acentric factor of the gas [-]
    Cvm : float
        Molar contant volume heat capacity of the gas [J/mol/K]

    Returns
    -------
    kg : float
        Estimated gas thermal conductivity [W/m/k]

    Notes
    -----
    Reference fluid is Methane.
    MW internally converted to kg/g-mol.

    Examples
    --------
    2-methylbutane at low pressure, 373.15 K. Mathes calculation in [2]_.

    >>> eli_hanley(T=373.15, MW=72.151, Tc=460.4, Vc=3.06E-4, Zc=0.267,
    ... omega=0.227, Cvm=135.9)
    0.02247951789135337

    References
    ----------
    .. [1] Ely, James F., and H. J. M. Hanley. "Prediction of Transport
       Properties. 2. Thermal Conductivity of Pure Fluids and Mixtures."
       Industrial & Engineering Chemistry Fundamentals 22, no. 1 (February 1,
       1983): 90-97. doi:10.1021/i100009a016.
    .. [2] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    Cs = [2.907741307E6, -3.312874033E6, 1.608101838E6, -4.331904871E5, 
          7.062481330E4, -7.116620750E3, 4.325174400E2, -1.445911210E1, 2.037119479E-1]

    Tr = T/Tc
    if Tr > 2: Tr = 2
    theta = 1 + (omega - 0.011)*(0.56553 - 0.86276*log(Tr) - 0.69852/Tr)
    psi = (1 + (omega-0.011)*(0.38560 - 1.1617*log(Tr)))*0.288/Zc
    f = Tc/190.4*theta
    h = Vc/9.92E-5*psi
    T0 = T/f
    eta0 = 1E-7*sum([Ci*T0**((i+1. - 4.)/3.) for i, Ci in enumerate(Cs)])
    k0 = 1944*eta0

    H = (16.04/MW)**0.5*f**0.5*h**(-2/3.)
    etas = eta0*H*MW/16.04
    ks = k0*H
    return ks + etas/(MW/1000.)*1.32*(Cvm - 1.5*R)


def Gharagheizi_gas(T, MW, Tb, Pc, omega):
    r'''Estimates the thermal conductivity of a gas as a function of
    temperature using the CSP method of Gharagheizi [1]_. A  convoluted
    method claiming high-accuracy and using only statistically significant
    variable following analalysis.

    Requires temperature, molecular weight, boiling temperature and critical
    pressure and acentric factor.

    .. math::
        k = 7.9505\times 10^{-4} + 3.989\times 10^{-5} T
        -5.419\times 10^-5 M + 3.989\times 10^{-5} A

       A = \frac{\left(2\omega + T - \frac{(2\omega + 3.2825)T}{T_b} + 3.2825\right)}{0.1MP_cT}
        \times (3.9752\omega + 0.1 P_c + 1.9876B + 6.5243)^2


    Parameters
    ----------
    T : float
        Temperature of the fluid [K]
    MW: float
        Molecular weight of the fluid [g/mol]
    Tb : float
        Boiling temperature of the fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    omega : float
        Acentric factor of the fluid [-]

    Returns
    -------
    kg : float
        Estimated gas thermal conductivity [W/m/k]

    Notes
    -----
    Pressure is internally converted into 10*kPa but author used correlation with
    kPa; overall, errors have been corrected in the presentation of the formula.

    This equation was derived with 15927 points and 1574 compounds.
    Example value from [1]_ is the first point in the supportinf info, for CH4.

    Examples
    --------
    >>> Gharagheizi_gas(580., 16.04246, 111.66, 4599000.0, 0.0115478000)
    0.09594861261873211

    References
    ----------
    .. [1] Gharagheizi, Farhad, Poorandokht Ilani-Kashkouli, Mehdi Sattari,
       Amir H. Mohammadi, Deresh Ramjugernath, and Dominique Richon.
       "Development of a General Model for Determination of Thermal
       Conductivity of Liquid Chemical Compounds at Atmospheric Pressure."
       AIChE Journal 59, no. 5 (May 1, 2013): 1702-8. doi:10.1002/aic.13938
    '''
    Pc = Pc/1E4
    B = T + (2.*omega + 2.*T - 2.*T*(2.*omega + 3.2825)/Tb + 3.2825)/(2*omega + T - T*(2*omega+3.2825)/Tb + 3.2825) - T*(2*omega+3.2825)/Tb
    A = (2*omega + T - T*(2*omega + 3.2825)/Tb + 3.2825)/(0.1*MW*Pc*T) * (3.9752*omega + 0.1*Pc + 1.9876*B + 6.5243)**2
    return 7.9505E-4 + 3.989E-5*T - 5.419E-5*MW + 3.989E-5*A


def Bahadori_gas(T, MW):
    r'''Estimates the thermal conductivity of hydrocarbons gases at low P.
    Fits their data well, and is useful as only MW is required.
    Y is the Molecular weight, and X the temperature.

    .. math::
        K = a + bY + CY^2 + dY^3

        a = A_1 + B_1 X + C_1 X^2 + D_1 X^3

        b = A_2 + B_2 X + C_2 X^2 + D_2 X^3

        c = A_3 + B_3 X + C_3 X^2 + D_3 X^3

        d = A_4 + B_4 X + C_4 X^2 + D_4 X^3

    Parameters
    ----------
    T : float
        Temperature of the gas [K]
    MW : float
        Molecular weight of the gas [g/mol]

    Returns
    -------
    kg : float
        Estimated gas thermal conductivity [W/m/k]

    Notes
    -----
    The accuracy of this equation has not been reviewed.

    Examples
    --------
    >>> Bahadori_gas(40+273.15, 20) # Point from article
    0.031968165337873326

    References
    ----------
    .. [1] Bahadori, Alireza, and Saeid Mokhatab. "Estimating Thermal
       Conductivity of Hydrocarbons." Chemical Engineering 115, no. 13
       (December 2008): 52-54
    '''
    A = [4.3931323468E-1, -3.88001122207E-2, 9.28616040136E-4, -6.57828995724E-6]
    B = [-2.9624238519E-3, 2.67956145820E-4, -6.40171884139E-6, 4.48579040207E-8]
    C = [7.54249790107E-6, -6.46636219509E-7, 1.5124510261E-8, -1.0376480449E-10]
    D = [-6.0988433456E-9, 5.20752132076E-10, -1.19425545729E-11, 8.0136464085E-14]
    X, Y = T, MW
    a = A[0] + B[0]*X + C[0]*X**2 + D[0]*X**3
    b = A[1] + B[1]*X + C[1]*X**2 + D[1]*X**3
    c = A[2] + B[2]*X + C[2]*X**2 + D[2]*X**3
    d = A[3] + B[3]*X + C[3]*X**2 + D[3]*X**3
    return a + b*Y + c*Y**2 + d*Y**3


GHARAGHEIZI_G = 'GHARAGHEIZI_G'
CHUNG = 'CHUNG'
ELI_HANLEY = 'ELI_HANLEY'
ELI_HANLEY_DENSE = 'ELI_HANLEY_DENSE'
CHUNG_DENSE = 'CHUNG_DENSE'
EUCKEN_MOD = 'EUCKEN_MOD'
EUCKEN = 'EUCKEN'
BAHADORI_G = 'BAHADORI_G'
STIEL_THODOS_DENSE = 'STIEL_THODOS_DENSE'
DIPPR_9B = 'DIPPR_9B'



thermal_conductivity_gas_methods = [COOLPROP, DIPPR_PERRY_8E, VDI_PPDS, VDI_TABULAR, GHARAGHEIZI_G,
                                    DIPPR_9B, CHUNG, ELI_HANLEY, EUCKEN_MOD,
                                    EUCKEN, BAHADORI_G]
'''Holds all low-pressure methods available for the ThermalConductivityGas
class, for use in iterating over them.'''
thermal_conductivity_gas_methods_P = [COOLPROP, ELI_HANLEY_DENSE, CHUNG_DENSE,
                                      STIEL_THODOS_DENSE]
'''Holds all high-pressure methods available for the ThermalConductivityGas
class, for use in iterating over them.'''

class ThermalConductivityGas(TPDependentProperty):
    r'''Class for dealing with gas thermal conductivity as a function of
    temperature and pressure.

    For gases at atmospheric pressure, there are 7 corresponding-states
    estimators, one source of tabular information, and the external library
    CoolProp.

    For gases under the fluid's boiling point (at sub-atmospheric pressures),
    and high-pressure gases above the boiling point, there are three
    corresponding-states estimators, and the external library CoolProp.

    Parameters
    ----------
    CASs : list[str], optional
        The CAS numbers of all species in the mixture
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
        Critical compressibility, [-]
    omega : float, optional
        Acentric factor, [-]
    dipole : float, optional
        Dipole moment of the fluid, [debye]
    Vmg : float or callable, optional
        Molar volume of the fluid at a pressure and temperature or callable for
        the same, [m^3/mol]
    Cvgm : float or callable, optional
        Molar heat capacity of the fluid at a pressure and temperature or 
        or callable for the same, [J/mol/K]
    mug : float or callable, optional
        Gas viscosity of the fluid at a pressure and temperature or callable
        for the same, [Pa*S]

    Notes
    -----
    To iterate over all methods, use the lists stored in
    :obj:`thermal_conductivity_gas_methods` and
    :obj:`thermal_conductivity_gas_methods_P` for low and high pressure
    methods respectively.

    Low pressure methods:

    **GHARAGHEIZI_G**:
        CSP method, described in :obj:`Gharagheizi_gas`.
    **DIPPR_9B**:
        CSP method, described in :obj:`DIPPR9B`.
    **CHUNG**:
        CSP method, described in :obj:`Chung`.
    **ELI_HANLEY**:
        CSP method, described in :obj:`eli_hanley`.
    **EUCKEN_MOD**:
        CSP method, described in :obj:`Eucken_modified`.
    **EUCKEN**:
        CSP method, described in :obj:`Eucken`.
    **BAHADORI_G**:
        CSP method, described in :obj:`Bahadori_gas`.
    **DIPPR_PERRY_8E**:
        A collection of 345 coefficient sets from the DIPPR database published
        openly in [3]_. Provides temperature limits for all its fluids. 
        :obj:`thermo.dippr.EQ102` is used for its fluids.
    **VDI_PPDS**:
        Coefficients for a equation form developed by the PPDS, published 
        openly in [2]_. Covers a large temperature range, but does not 
        extrapolate well at very high or very low temperatures. 275 compounds.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow.
    **VDI_TABULAR**:
        Tabular data in [2]_ along the saturation curve; interpolation is as
        set by the user or the default.

    High pressure methods:

    **STIEL_THODOS_DENSE**:
        CSP method, described in :obj:`stiel_thodos_dense`. Calculates a
        low-pressure thermal conductivity first, using `T_dependent_property`.
    **ELI_HANLEY_DENSE**:
        CSP method, described in :obj:`eli_hanley_dense`. Calculates a
        low-pressure thermal conductivity first, using `T_dependent_property`.
    **CHUNG_DENSE**:
        CSP method, described in :obj:`chung_dense`. Calculates a
        low-pressure thermal conductivity first, using `T_dependent_property`.
    **COOLPROP**:
        CoolProp external library; with select fluids from its library.
        Range is limited to that of the equations of state it uses, as
        described in [1]_. Very slow, but unparalled in accuracy for pressure
        dependence.

    See Also
    --------
    Bahadori_gas
    Gharagheizi_gas
    eli_hanley
    Chung
    DIPPR9B
    Eucken_modified
    Eucken
    stiel_thodos_dense
    eli_hanley_dense
    chung_dense

    References
    ----------
    .. [1] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    .. [3] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       Eighth Edition. McGraw-Hill Professional, 2007.
    '''
    name = 'gas thermal conductivity'
    units = 'W/m/K'
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
    '''Mimimum valid value of gas thermal conductivity.'''
    property_max = 10
    '''Maximum valid value of gas thermal conductivity. Generous limit.'''

    ranked_methods = [COOLPROP, VDI_PPDS, DIPPR_PERRY_8E, VDI_TABULAR, GHARAGHEIZI_G, DIPPR_9B,
                      CHUNG, ELI_HANLEY, EUCKEN_MOD, EUCKEN,
                      BAHADORI_G]
    '''Default rankings of the low-pressure methods.'''
    ranked_methods_P = [COOLPROP, ELI_HANLEY_DENSE, CHUNG_DENSE,
                        STIEL_THODOS_DENSE]
    '''Default rankings of the high-pressure methods.'''

    def __init__(self, CASRN='', MW=None, Tb=None, Tc=None, Pc=None, Vc=None,
                 Zc=None, omega=None, dipole=None, Vmg=None, Cvgm=None, mug=None):
        self.CASRN = CASRN
        self.MW = MW
        self.Tb = Tb
        self.Tc = Tc
        self.Pc = Pc
        self.Vc = Vc
        self.Zc = Zc
        self.omega = omega
        self.dipole = dipole
        self.Vmg = Vmg
        self.Cvgm = Cvgm
        self.mug = mug

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        gas thermal conductivity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        gas thermal conductivity above.'''

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
        methods, methods_P = [], []
        Tmins, Tmaxs = [], []
        if self.CASRN in _VDISaturationDict:
            methods.append(VDI_TABULAR)
            Ts, props = VDI_tabular_data(self.CASRN, 'K (g)')
            self.VDI_Tmin = Ts[0]
            self.VDI_Tmax = Ts[-1]
            self.tabular_data[VDI_TABULAR] = (Ts, props)
            Tmins.append(self.VDI_Tmin); Tmaxs.append(self.VDI_Tmax)
        if has_CoolProp and self.CASRN in coolprop_dict:
            methods.append(COOLPROP); methods_P.append(COOLPROP)
            self.CP_f = coolprop_fluids[self.CASRN]
            Tmins.append(self.CP_f.Tmin); Tmaxs.append(self.CP_f.Tc)
        if self.CASRN in Perrys2_314.index:
            methods.append(DIPPR_PERRY_8E)
            _, C1, C2, C3, C4, self.Perrys2_314_Tmin, self.Perrys2_314_Tmax = _Perrys2_314_values[Perrys2_314.index.get_loc(self.CASRN)].tolist()
            self.Perrys2_314_coeffs = [C1, C2, C3, C4]
            Tmins.append(self.Perrys2_314_Tmin); Tmaxs.append(self.Perrys2_314_Tmax)
        if self.CASRN in VDI_PPDS_10.index:
            _,  A, B, C, D, E = _VDI_PPDS_10_values[VDI_PPDS_10.index.get_loc(self.CASRN)].tolist()
            self.VDI_PPDS_coeffs = [A, B, C, D, E]
            self.VDI_PPDS_coeffs.reverse()
            methods.append(VDI_PPDS)
        if all((self.MW, self.Tb, self.Pc, self.omega)):
            methods.append(GHARAGHEIZI_G)
            # Turns negative at low T; do not set Tmin
            Tmaxs.append(3000)
        if all((self.Cvgm, self.mug, self.MW, self.Tc)):
            methods.append(DIPPR_9B)
            Tmins.append(0.01); Tmaxs.append(1E4)  # No limit here
        if all((self.Cvgm, self.mug, self.MW, self.Tc, self.omega)):
            methods.append(CHUNG)
            Tmins.append(0.01); Tmaxs.append(1E4)  # No limit
        if all((self.Cvgm, self.MW, self.Tc, self.Vc, self.Zc, self.omega)):
            methods.append(ELI_HANLEY)
            Tmaxs.append(1E4)  # Numeric error at low T
        if all((self.Cvgm, self.mug, self.MW)):
            methods.append(EUCKEN_MOD)
            methods.append(EUCKEN)
            Tmins.append(0.01); Tmaxs.append(1E4)  # No limits
        if self.MW:
            methods.append(BAHADORI_G)
            # Terrible method, so don't set methods
        if all([self.MW, self.Tc, self.Vc, self.Zc, self.omega]):
            methods_P.append(ELI_HANLEY_DENSE)
        if all([self.MW, self.Tc, self.Vc, self.omega, self.dipole]):
            methods_P.append(CHUNG_DENSE)
        if all([self.MW, self.Tc, self.Pc, self.Vc, self.Zc]):
            methods_P.append(STIEL_THODOS_DENSE)
        self.all_methods = set(methods)
        self.all_methods_P = set(methods_P)
        if Tmins and Tmaxs:
            self.Tmin, self.Tmax = min(Tmins), max(Tmaxs)

    def calculate(self, T, method):
        r'''Method to calculate low-pressure gas thermal conductivity at
        tempearture `T` with a given method.

        This method has no exception handling; see `T_dependent_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature of the gas, [K]
        method : str
            Name of the method to use

        Returns
        -------
        kg : float
            Thermal conductivity of the gas at T and a low pressure, [W/m/K]
        '''
        if method == GHARAGHEIZI_G:
            kg = Gharagheizi_gas(T, self.MW, self.Tb, self.Pc, self.omega)
        elif method == DIPPR_9B:
            Cvgm = self.Cvgm(T) if hasattr(self.Cvgm, '__call__') else self.Cvgm
            mug = self.mug(T) if hasattr(self.mug, '__call__') else self.mug
            kg = DIPPR9B(T, self.MW, Cvgm, mug, self.Tc)
        elif method == CHUNG:
            Cvgm = self.Cvgm(T) if hasattr(self.Cvgm, '__call__') else self.Cvgm
            mug = self.mug(T) if hasattr(self.mug, '__call__') else self.mug
            kg = Chung(T, self.MW, self.Tc, self.omega, Cvgm, mug)
        elif method == ELI_HANLEY:
            Cvgm = self.Cvgm(T) if hasattr(self.Cvgm, '__call__') else self.Cvgm
            kg = eli_hanley(T, self.MW, self.Tc, self.Vc, self.Zc, self.omega, Cvgm)
        elif method == EUCKEN_MOD:
            Cvgm = self.Cvgm(T) if hasattr(self.Cvgm, '__call__') else self.Cvgm
            mug = self.mug(T) if hasattr(self.mug, '__call__') else self.mug
            kg = Eucken_modified(self.MW, Cvgm, mug)
        elif method == EUCKEN:
            Cvgm = self.Cvgm(T) if hasattr(self.Cvgm, '__call__') else self.Cvgm
            mug = self.mug(T) if hasattr(self.mug, '__call__') else self.mug
            kg = Eucken(self.MW, Cvgm, mug)
        elif method == DIPPR_PERRY_8E:
            kg = EQ102(T, *self.Perrys2_314_coeffs)
        elif method == VDI_PPDS:
            kg = horner(self.VDI_PPDS_coeffs, T)
        elif method == BAHADORI_G:
            kg = Bahadori_gas(T, self.MW)
        elif method == COOLPROP:
            kg = CoolProp_T_dependent_property(T, self.CASRN, 'L', 'g')
        elif method in self.tabular_data:
            kg = self.interpolate(T, method)
        return kg

    def calculate_P(self, T, P, method):
        r'''Method to calculate pressure-dependent gas thermal conductivity
        at temperature `T` and pressure `P` with a given method.

        This method has no exception handling; see `TP_dependent_property`
        for that.

        Parameters
        ----------
        T : float
            Temperature at which to calculate gas thermal conductivity, [K]
        P : float
            Pressure at which to calculate gas thermal conductivity, [K]
        method : str
            Name of the method to use

        Returns
        -------
        kg : float
            Thermal conductivity of the gas at T and P, [W/m/K]
        '''
        if method == ELI_HANLEY_DENSE:
            Vmg = self.Vmg(T, P) if hasattr(self.Vmg, '__call__') else self.Vmg
            Cvgm = self.Cvgm(T) if hasattr(self.Cvgm, '__call__') else self.Cvgm
            kg = eli_hanley_dense(T, self.MW, self.Tc, self.Vc, self.Zc, self.omega, Cvgm, Vmg)
        elif method == CHUNG_DENSE:
            Vmg = self.Vmg(T, P) if hasattr(self.Vmg, '__call__') else self.Vmg
            Cvgm = self.Cvgm(T) if hasattr(self.Cvgm, '__call__') else self.Cvgm
            mug = self.mug(T, P) if hasattr(self.mug, '__call__') else self.mug
            kg = chung_dense(T, self.MW, self.Tc, self.Vc, self.omega, Cvgm, Vmg, mug, self.dipole)
        elif method == STIEL_THODOS_DENSE:
            kg = self.T_dependent_property(T)
            Vmg = self.Vmg(T, P) if hasattr(self.Vmg, '__call__') else self.Vmg
            kg = stiel_thodos_dense(T, self.MW, self.Tc, self.Pc, self.Vc, self.Zc, Vmg, kg)
        elif method == COOLPROP:
            kg = PropsSI('L', 'T', T, 'P', P, self.CASRN)
        elif method in self.tabular_data:
            kg = self.interpolate_P(T, P, method)
        return kg

    def test_method_validity(self, T, method):
        r'''Method to check the validity of a temperature-dependent
        low-pressure method. For CSP methods, the all methods are considered
        valid from 0 K and up.

        For tabular data, extrapolation outside of the range is used if
        :obj:`tabular_extrapolation_permitted` is set; if it is, the extrapolation
        is considered valid for all temperatures.

        It is not guaranteed that a method will work or give an accurate
        prediction simply because this method considers the method valid.
        **GHARAGHEIZI_G** and **BAHADORI_G** are known to sometimes produce
        negative results.

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
        if method in [GHARAGHEIZI_G, DIPPR_9B, CHUNG, ELI_HANLEY, EUCKEN_MOD,
                      EUCKEN, BAHADORI_G, VDI_PPDS]:
            pass
        elif method == DIPPR_PERRY_8E:
            if T < self.Perrys2_314_Tmin or T > self.Perrys2_314_Tmax:
                return False
        elif method == COOLPROP:
            if T < self.CP_f.Tmin or T > self.CP_f.Tmax:
                return False
        elif method in self.tabular_data:
            # if tabular_extrapolation_permitted, good to go without checking
            if not self.tabular_extrapolation_permitted:
                Ts, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1]:
                    return False
        else:
            raise Exception('Method not valid')
        return True

    def test_method_validity_P(self, T, P, method):
        r'''Method to check the validity of a high-pressure method. For
        **COOLPROP**, the fluid must be both a gas and under the maximum
        pressure of the fluid's EOS. The CSP method **ELI_HANLEY_DENSE**,
        **CHUNG_DENSE**, and **STIEL_THODOS_DENSE** are considered valid for
        all temperatures and pressures.

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
        if method in [ELI_HANLEY_DENSE, CHUNG_DENSE, STIEL_THODOS_DENSE]:
            if T < 0 or P < 0:
                validity = False
            # no better checks known
        elif method == COOLPROP:
            if T < self.CP_f.Tmin or T > self.CP_f.Tmax or P > self.CP_f.Pmax:
                return False
            else:
                return PhaseSI('T', T, 'P', P, self.CASRN) in ['gas', 'supercritical_gas', 'supercritical', 'supercritical_liquid']
        elif method in self.tabular_data:
            if not self.tabular_extrapolation_permitted:
                Ts, Ps, properties = self.tabular_data[method]
                if T < Ts[0] or T > Ts[-1] or P < Ps[0] or P > Ps[-1]:
                    validity = False
        else:
            raise Exception('Method not valid')
        return validity


### Thermal Conductivity of dense gases

def stiel_thodos_dense(T, MW, Tc, Pc, Vc, Zc, Vm, kg):
    r'''Estimates the thermal conductivity of a gas at high pressure as a
    function of temperature using difference method of Stiel and Thodos [1]_
    as shown in [2]_.

    if \rho_r < 0.5:

    .. math::
        (\lambda-\lambda^\circ)\Gamma Z_c^5=1.22\times 10^{-2} [\exp(0.535 \rho_r)-1]

    if 0.5 < \rho_r < 2.0:

    .. math::
        (\lambda-\lambda^\circ)\Gamma Z_c^5=1.22\times 10^{-2} [\exp(0.535 \rho_r)-1]

    if 2 < \rho_r < 2.8:

    .. math::
        (\lambda-\lambda^\circ)\Gamma Z_c^5=1.22\times 10^{-2} [\exp(0.535 \rho_r)-1]

        \Gamma = 210 \left(\frac{T_cMW^3}{P_c^4}\right)^{1/6}

    Parameters
    ----------
    T : float
        Temperature of the gas [K]
    MW : float
        Molecular weight of the gas [g/mol]
    Tc : float
        Critical temperature of the gas [K]
    Pc : float
        Critical pressure of the gas [Pa]
    Vc : float
        Critical volume of the gas [m^3/mol]
    Zc : float
        Critical compressibility of the gas [-]
    Vm : float
        Molar volume of the gas at T and P [m^3/mol]
    kg : float
        Low-pressure gas thermal conductivity [W/m/k]

    Returns
    -------
    kg : float
        Estimated dense gas thermal conductivity [W/m/k]

    Notes
    -----
    Pc is internally converted to bar.

    Examples
    --------
    >>> stiel_thodos_dense(T=378.15, MW=44.013, Tc=309.6, Pc=72.4E5,
    ... Vc=97.4E-6, Zc=0.274, Vm=144E-6, kg=2.34E-2)
    0.041245574404863684

    References
    ----------
    .. [1] Stiel, Leonard I., and George Thodos. "The Thermal Conductivity of
       Nonpolar Substances in the Dense Gaseous and Liquid Regions." AIChE
       Journal 10, no. 1 (January 1, 1964): 26-30. doi:10.1002/aic.690100114.
    .. [2] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    gamma = 210*(Tc*MW**3./(Pc/1E5)**4)**(1/6.)
    rhor = Vc/Vm
    if rhor < 0.5:
        term = 1.22E-2*(exp(0.535*rhor) - 1.)
    elif rhor < 2:
        term = 1.14E-2*(exp(0.67*rhor) - 1.069)
    else:
        # Technically only up to 2.8
        term = 2.60E-3*(exp(1.155*rhor) + 2.016)
    diff = term/Zc**5/gamma
    kg = kg + diff
    return kg


def eli_hanley_dense(T, MW, Tc, Vc, Zc, omega, Cvm, Vm):
    r'''Estimates the thermal conductivity of a gas at high pressure as a
    function of temperature using the reference fluid method of Eli and
    Hanley [1]_ as shown in [2]_.

    .. math::
        Tr = min(Tr, 2)

        Vr = min(Vr, 2)

        f = \frac{T_c}{190.4}\theta

        h = \frac{V_c}{9.92E-5}\psi

        T_0 = T/f

        \rho_0 = \frac{16.04}{V}h

        \theta = 1 + (\omega-0.011)\left(0.09057 - 0.86276\ln Tr + \left(
        0.31664 - \frac{0.46568}{Tr}\right) (V_r - 0.5)\right)

        \psi = [1 + (\omega - 0.011)(0.39490(V_r - 1.02355) - 0.93281(V_r -
        0.75464)\ln T_r]\frac{0.288}{Z_c}

        \lambda_1 = 1944 \eta_0

        \lambda_2 = \left\{b_1 + b_2\left[b_3 - \ln \left(\frac{T_0}{b_4}
        \right)\right]^2\right\}\rho_0

        \lambda_3 = \exp\left(a_1 + \frac{a_2}{T_0}\right)\left\{\exp[(a_3 +
        \frac{a_4}{T_0^{1.5}})\rho_0^{0.1} + (\frac{\rho_0}{0.1617} - 1)
        \rho_0^{0.5}(a_5 + \frac{a_6}{T_0} + \frac{a_7}{T_0^2})] - 1\right\}

        \lambda^{**} = [\lambda_1 + \lambda_2 + \lambda_3]H

        H = \left(\frac{16.04}{MW}\right)^{0.5}f^{0.5}/h^{2/3}

        X = \left\{\left[1 - \frac{T}{f}\left(\frac{df}{dT}\right)_v \right]
        \frac{0.288}{Z_c}\right\}^{1.5}

        \left(\frac{df}{dT}\right)_v = \frac{T_c}{190.4}\left(\frac{d\theta}
        {d T}\right)_v

        \left(\frac{d\theta}{d T}\right)_v = (\omega-0.011)\left[
        \frac{-0.86276}{T} + (V_r-0.5)\frac{0.46568T_c}{T^2}\right]

    Parameters
    ----------
    T : float
        Temperature of the gas [K]
    MW : float
        Molecular weight of the gas [g/mol]
    Tc : float
        Critical temperature of the gas [K]
    Vc : float
        Critical volume of the gas [m^3/mol]
    Zc : float
        Critical compressibility of the gas []
    omega : float
        Acentric factor of the gas [-]
    Cvm : float
        Molar contant volume heat capacity of the gas [J/mol/K]
    Vm : float
        Volume of the gas at T and P [m^3/mol]

    Returns
    -------
    kg : float
        Estimated dense gas thermal conductivity [W/m/k]

    Notes
    -----
    Reference fluid is Methane.
    MW internally converted to kg/g-mol.

    Examples
    --------
    >>> eli_hanley_dense(T=473., MW=42.081, Tc=364.9, Vc=1.81E-4, Zc=0.274,
    ... omega=0.144, Cvm=82.70, Vm=1.721E-4)
    0.06038475936515042

    References
    ----------
    .. [1] Ely, James F., and H. J. M. Hanley. "Prediction of Transport
       Properties. 2. Thermal Conductivity of Pure Fluids and Mixtures."
       Industrial & Engineering Chemistry Fundamentals 22, no. 1 (February 1,
       1983): 90-97. doi:10.1021/i100009a016.
    .. [2] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    Cs = [2.907741307E6, -3.312874033E6, 1.608101838E6, -4.331904871E5,
          7.062481330E4, -7.116620750E3, 4.325174400E2, -1.445911210E1,
          2.037119479E-1]

    Tr = T/Tc
    if Tr > 2:
        Tr = 2
    Vr = Vm/Vc
    if Vr > 2:
        Vr = 2
    theta = 1 + (omega - 0.011)*(0.09057 - 0.86276*log(Tr) + (0.31664 - 0.46568/Tr)*(Vr-0.5))
    psi = (1 + (omega-0.011)*(0.39490*(Vr-1.02355) - 0.93281*(Vr-0.75464)*log(Tr)))*0.288/Zc
    f = Tc/190.4*theta
    h = Vc/9.92E-5*psi
    T0 = T/f
    rho0 = 16.04/(Vm*1E6)*h  # Vm must be in cm^3/mol here.
    eta0 = 1E-7*sum([Cs[i]*T0**((i+1-4)/3.) for i in range(len(Cs))])
    k1 = 1944*eta0
    b1 = -0.25276920E0
    b2 = 0.334328590E0
    b3 = 1.12
    b4 = 0.1680E3
    k2 = (b1 + b2*(b3 - log(T0/b4))**2)/1000.*rho0

    a1 = -7.19771
    a2 = 85.67822
    a3 = 12.47183
    a4 = -984.6252
    a5 = 0.3594685
    a6 = 69.79841
    a7 = -872.8833

    k3 = exp(a1 + a2/T0)*(exp((a3 + a4/T0**1.5)*rho0**0.1 + (rho0/0.1617 - 1)*rho0**0.5*(a5 + a6/T0 + a7/T0**2)) - 1)/1000.

    if T/Tc > 2:
        dtheta = 0
    else:
        dtheta = (omega - 0.011)*(-0.86276/T + (Vr-0.5)*0.46568*Tc/T**2)
    dfdT = Tc/190.4*dtheta
    X = ((1 - T/f*dfdT)*0.288/Zc)**1.5

    H = (16.04/MW)**0.5*f**0.5/h**(2/3.)
    ks = (k1*X + k2 + k3)*H

    ### Uses calculations similar to those for pure species here
    theta = 1 + (omega - 0.011)*(0.56553 - 0.86276*log(Tr) - 0.69852/Tr)
    psi = (1 + (omega-0.011)*(0.38560 - 1.1617*log(Tr)))*0.288/Zc
    f = Tc/190.4*theta
    h = Vc/9.92E-5*psi
    T0 = T/f
    eta0 = 1E-7*sum([Cs[i]*T0**((i+1-4)/3.) for i in range(len(Cs))])
    H = (16.04/MW)**0.5*f**0.5/h**(2/3.)
    etas = eta0*H*MW/16.04
    k = ks + etas/(MW/1000.)*1.32*(Cvm-3*R/2.)
    return k


def chung_dense(T, MW, Tc, Vc, omega, Cvm, Vm, mu, dipole, association=0):
    r'''Estimates the thermal conductivity of a gas at high pressure as a
    function of temperature using the reference fluid method of
    Chung [1]_ as shown in [2]_.

    .. math::
        \lambda = \frac{31.2 \eta^\circ \Psi}{M'}(G_2^{-1} + B_6 y)+qB_7y^2T_r^{1/2}G_2

        \Psi = 1 + \alpha \left\{[0.215+0.28288\alpha-1.061\beta+0.26665Z]/
        [0.6366+\beta Z + 1.061 \alpha \beta]\right\}

        \alpha = \frac{C_v}{R}-1.5

        \beta = 0.7862-0.7109\omega + 1.3168\omega^2

        Z=2+10.5T_r^2

        q = 3.586\times 10^{-3} (T_c/M')^{1/2}/V_c^{2/3}

        y = \frac{V_c}{6V}

        G_1 = \frac{1-0.5y}{(1-y)^3}

        G_2 = \frac{(B_1/y)[1-\exp(-B_4y)]+ B_2G_1\exp(B_5y) + B_3G_1}
        {B_1B_4 + B_2 + B_3}

        B_i = a_i + b_i \omega + c_i \mu_r^4 + d_i \kappa


    Parameters
    ----------
    T : float
        Temperature of the gas [K]
    MW : float
        Molecular weight of the gas [g/mol]
    Tc : float
        Critical temperature of the gas [K]
    Vc : float
        Critical volume of the gas [m^3/mol]
    omega : float
        Acentric factor of the gas [-]
    Cvm : float
        Molar contant volume heat capacity of the gas [J/mol/K]
    Vm : float
        Molar volume of the gas at T and P [m^3/mol]
    mu : float
        Low-pressure gas viscosity [Pa*S]
    dipole : float
        Dipole moment [debye]
    association : float, optional
        Association factor [-]

    Returns
    -------
    kg : float
        Estimated dense gas thermal conductivity [W/m/k]

    Notes
    -----
    MW internally converted to kg/g-mol.
    Vm internally converted to mL/mol.
    [1]_ is not the latest form as presented in [1]_.
    Association factor is assumed 0. Relates to the polarity of the gas.

    Coefficients as follows:
    ais = [2.4166E+0, -5.0924E-1, 6.6107E+0, 1.4543E+1, 7.9274E-1, -5.8634E+0, 9.1089E+1]

    bis = [7.4824E-1, -1.5094E+0, 5.6207E+0, -8.9139E+0, 8.2019E-1, 1.2801E+1, 1.2811E+2]

    cis = [-9.1858E-1, -4.9991E+1, 6.4760E+1, -5.6379E+0, -6.9369E-1, 9.5893E+0, -5.4217E+1]

    dis = [1.2172E+2, 6.9983E+1, 2.7039E+1, 7.4344E+1, 6.3173E+0, 6.5529E+1, 5.2381E+2]


    Examples
    --------
    >>> chung_dense(T=473., MW=42.081, Tc=364.9, Vc=184.6E-6, omega=0.142,
    ... Cvm=82.67, Vm=172.1E-6, mu=134E-7, dipole=0.4)
    0.06160570379787278

    References
    ----------
    .. [1] Chung, Ting Horng, Mohammad Ajlan, Lloyd L. Lee, and Kenneth E.
       Starling. "Generalized Multiparameter Correlation for Nonpolar and Polar
       Fluid Transport Properties." Industrial & Engineering Chemistry Research
       27, no. 4 (April 1, 1988): 671-79. doi:10.1021/ie00076a024.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    ais = [2.4166E+0, -5.0924E-1, 6.6107E+0, 1.4543E+1, 7.9274E-1, -5.8634E+0, 9.1089E+1]
    bis = [7.4824E-1, -1.5094E+0, 5.6207E+0, -8.9139E+0, 8.2019E-1, 1.2801E+1, 1.2811E+2]
    cis = [-9.1858E-1, -4.9991E+1, 6.4760E+1, -5.6379E+0, -6.9369E-1, 9.5893E+0, -5.4217E+1]
    dis = [1.2172E+2, 6.9983E+1, 2.7039E+1, 7.4344E+1, 6.3173E+0, 6.5529E+1, 5.2381E+2]
    Tr = T/Tc
    mur = 131.3*dipole/(Vc*1E6*Tc)**0.5

    # From Chung Method
    alpha = Cvm/R - 1.5
    beta = 0.7862 - 0.7109*omega + 1.3168*omega**2
    Z = 2 + 10.5*(T/Tc)**2
    psi = 1 + alpha*((0.215 + 0.28288*alpha - 1.061*beta + 0.26665*Z)/(0.6366 + beta*Z + 1.061*alpha*beta))

    y = Vc/(6*Vm)
    B1, B2, B3, B4, B5, B6, B7 = [ais[i] + bis[i]*omega + cis[i]*mur**4 + dis[i]*association for i in range(7)]
    G1 = (1 - 0.5*y)/(1. - y)**3
    G2 = (B1/y*(1 - exp(-B4*y)) + B2*G1*exp(B5*y) + B3*G1)/(B1*B4 + B2 + B3)
    q = 3.586E-3*(Tc/(MW/1000.))**0.5/(Vc*1E6)**(2/3.)
    return 31.2*mu*psi/(MW/1000.)*(G2**-1 + B6*y) + q*B7*y**2*Tr**0.5*G2


### Thermal conductivity of gas mixtures

def Lindsay_Bromley(T, ys, ks, mus, Tbs, MWs):
    r'''Calculates thermal conductivity of a gas mixture according to
    mixing rules in [1]_ and also in [2]_.

    .. math::
        k = \sum \frac{y_i k_i}{\sum y_i A_{ij}}

        A_{ij} = \frac{1}{4} \left\{ 1 + \left[\frac{\eta_i}{\eta_j}
        \left(\frac{MW_j}{MW_i}\right)^{0.75} \left( \frac{T+S_i}{T+S_j}\right)
        \right]^{0.5} \right\}^2 \left( \frac{T+S_{ij}}{T+S_i}\right)

        S_{ij} = S_{ji} = (S_i S_j)^{0.5}

    Parameters
    ----------
    T : float
        Temperature of gas [K]
    ys : float
        Mole fractions of gas components
    ks : float
        Liquid thermal conductivites of all components, [W/m/K]
    mus : float
        Gas viscosities of all components, [Pa*S]
    Tbs : float
        Boiling points of all components, [K]
    MWs : float
        Molecular weights of all components, [g/mol]

    Returns
    -------
    kg : float
        Thermal conductivity of gas mixture, [W/m/K]

    Notes
    -----
    This equation is entirely dimensionless; all dimensions cancel.
    The example is from [2]_; all results agree.
    The original source has not been reviewed.

    DIPPR Procedure 9D: Method for the Thermal Conductivity of Gas Mixtures

    Average deviations of 4-5% for 77 binary mixtures reviewed in [2]_, from
    1342 points; also six ternary mixtures (70  points); max deviation observed
    was 40%. (DIPPR)

    TODO: Finish documenting this.

    Examples
    --------
    >>> Lindsay_Bromley(323.15, [0.23, 0.77], [1.939E-2, 1.231E-2], [1.002E-5, 1.015E-5], [248.31, 248.93], [46.07, 50.49])
    0.01390264417969313

    References
    ----------
    .. [1] Lindsay, Alexander L., and LeRoy A. Bromley. "Thermal Conductivity
       of Gas Mixtures." Industrial & Engineering Chemistry 42, no. 8
       (August 1, 1950): 1508-11. doi:10.1021/ie50488a017.
    .. [2] Danner, Ronald P, and Design Institute for Physical Property Data.
       Manual for Predicting Chemical Process Design Data. New York, N.Y, 1982.
    '''
    if not none_and_length_check([ys, ks, mus, Tbs, MWs]):
        raise Exception('Function inputs are incorrect format')

    cmps = range(len(ys))
    Ss = [1.5*Tb for Tb in Tbs]
    Sij = [[(Si*Sj)**0.5 for Sj in Ss] for Si in Ss]

    Aij = [[0.25*(1. + (mus[i]/mus[j]*(MWs[j]/MWs[i])**0.75
            *(T+Ss[i])/(T+Ss[j]))**0.5 )**2 *(T+Sij[i][j])/(T+Ss[i])
            for j in cmps] for i in cmps]
            
    return sum([ys[i]*ks[i]/sum(ys[j]*Aij[i][j] for j in cmps) for i in cmps])



LINDSAY_BROMLEY = 'LINDSAY_BROMLEY'
thermal_conductivity_gas_methods = [LINDSAY_BROMLEY, SIMPLE]


class ThermalConductivityGasMixture(MixtureProperty):
    '''Class for dealing with thermal conductivity of a gas mixture as a   
    function of temperature, pressure, and composition.
    Consists of one mixing rule specific to gas thremal conductivity, and mole
    weighted averaging. 
         
    Prefered method is :obj:`Lindsay_Bromley` which requires mole
    fractions, pure component viscosities and thermal conductivities, and the 
    boiling point and molecular weight of each pure component. This is 
    substantially better than the ideal mixing rule based on mole fractions, 
    **SIMPLE** which is also available.
        
    Parameters
    ----------
    MWs : list[float], optional
        Molecular weights of all species in the mixture, [g/mol]
    Tbs : list[float], optional
        Boiling points of all species in the mixture, [K]
    CASs : str, optional
        The CAS numbers of all species in the mixture
    ThermalConductivityGases : list[ThermalConductivityGas], optional
        ThermalConductivityGas objects created for all species in the mixture, 
        normally created by :obj:`thermo.chemical.Chemical`.
    ViscosityGases : list[ViscosityGas], optional
        ViscosityGas objects created for all species in the mixture, normally 
        created by :obj:`thermo.chemical.Chemical`.

    Notes
    -----
    To iterate over all methods, use the list stored in
    :obj:`thermal_conductivity_gas_methods`.

    **LINDSAY_BROMLEY**:
        Mixing rule described in :obj:`Lindsay_Bromley`.
    **SIMPLE**:
        Mixing rule described in :obj:`thermo.utils.mixing_simple`.

    See Also
    --------
    Lindsay_Bromley

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    name = 'gas thermal conductivity'
    units = 'W/m/K'
    property_min = 0.
    '''Mimimum valid value of gas thermal conductivity.'''
    property_max = 10.
    '''Maximum valid value of gas thermal conductivity. Generous limit.'''
                            
    ranked_methods = [LINDSAY_BROMLEY, SIMPLE]

    def __init__(self, MWs=[], Tbs=[], CASs=[], ThermalConductivityGases=[], 
                 ViscosityGases=[]):
        self.MWs = MWs
        self.Tbs = Tbs
        self.CASs = CASs
        self.ThermalConductivityGases = ThermalConductivityGases
        self.ViscosityGases = ViscosityGases                     

        self.Tmin = None
        '''Minimum temperature at which no method can calculate the
        gas thermal conductivity under.'''
        self.Tmax = None
        '''Maximum temperature at which no method can calculate the
        gas thermal conductivity above.'''

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
        methods.append(SIMPLE)
        if none_and_length_check((self.Tbs, self.MWs)):
            methods.append(LINDSAY_BROMLEY)
        self.all_methods = set(methods)
        Tmins = [i.Tmin for i in self.ThermalConductivityGases if i.Tmin]
        Tmaxs = [i.Tmax for i in self.ThermalConductivityGases if i.Tmax]
        if Tmins:
            self.Tmin = max(Tmins)
        if Tmaxs:
            self.Tmax = max(Tmaxs)

    def calculate(self, T, P, zs, ws, method):
        r'''Method to calculate thermal conductivity of a gas mixture at 
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
        kg : float
            Thermal conductivity of gas mixture, [W/m/K]
        '''
        if method == SIMPLE:
            ks = [i(T, P) for i in self.ThermalConductivityGases]
            return mixing_simple(zs, ks)
        elif method == LINDSAY_BROMLEY:
            ks = [i(T, P) for i in self.ThermalConductivityGases]
            mus = [i(T, P) for i in self.ViscosityGases]
            return Lindsay_Bromley(T=T, ys=zs, ks=ks, mus=mus, Tbs=self.Tbs, MWs=self.MWs)
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
        if method in [SIMPLE, LINDSAY_BROMLEY]:
            return True
        else:
            raise Exception('Method not valid')

