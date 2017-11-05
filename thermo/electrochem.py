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

__all__ = ['conductivity', 'Laliberte_density', 'Laliberte_heat_capacity', 
           'Laliberte_viscosity', 'Laliberte_data', 'Laliberte_viscosity_w', 
           'Laliberte_viscosity_i', 'Laliberte_density_w', 
           'Laliberte_density_i', 'Laliberte_heat_capacity_w', 
           'Laliberte_heat_capacity_i', 'dilute_ionic_conductivity', 
           'conductivity_McCleskey', 'Lange_cond_pure', 
           'conductivity_methods', 'Magomedovk_thermal_cond',
           'thermal_conductivity_Magomedov', 'ionic_strength', 'Kweq_1981', 
           'Kweq_IAPWS_gas', 'Kweq_IAPWS', 'Marcus_ion_conductivities',
           'balance_ions', 'McCleskey_conductivities', 'CRC_ion_conductivities',
           'CRC_aqueous_thermodynamics', 'electrolyte_dissociation_reactions']

import os
from collections import namedtuple
from thermo.utils import exp, log10
from thermo.utils import e, N_A
from thermo.utils import to_num, ws_to_zs, horner
from thermo.identifiers import pubchem_db
from scipy.interpolate import interp1d
from scipy.optimize import newton
import pandas as pd


F = e*N_A


folder = os.path.join(os.path.dirname(__file__), 'Electrolytes')



Lange_cond_pure = pd.read_csv(os.path.join(folder, 'Lange Pure Species Conductivity.tsv'),
                          sep='\t', index_col=0)

Marcus_ion_conductivities = pd.read_csv(os.path.join(folder, 'Marcus Ion Conductivities.tsv'),
                          sep='\t', index_col=0)

CRC_ion_conductivities = pd.read_csv(os.path.join(folder, 'CRC conductivity infinite dilution.tsv'),
                          sep='\t', index_col=0)

Magomedovk_thermal_cond = pd.read_csv(os.path.join(folder, 'Magomedov Thermal Conductivity.tsv'),
                          sep='\t', index_col=0)

CRC_aqueous_thermodynamics = pd.read_csv(os.path.join(folder, 'CRC Thermodynamic Properties of Aqueous Ions.csv'),
                          sep='\t', index_col=0) 

electrolyte_dissociation_reactions = pd.read_csv(os.path.join(folder, 'Electrolyte dissociations.csv'), sep='\t')


_Laliberte_Density_ParametersDict = {}
_Laliberte_Viscosity_ParametersDict = {}
_Laliberte_Heat_Capacity_ParametersDict = {}


# Do not re-implement with Pandas, as current methodology uses these dicts in each function
with open(os.path.join(folder, 'Laliberte2009.tsv')) as f:
    next(f)
    for line in f:
        values = to_num(line.split('\t'))

        _name, CASRN, _formula, _MW, c0, c1, c2, c3, c4, Tmin, Tmax, wMax, pts = values[0:13]
        if c0:
            _Laliberte_Density_ParametersDict[CASRN] = {"Name":_name, "Formula":_formula,
            "MW":_MW, "C0":c0, "C1":c1, "C2":c2, "C3":c3, "C4":c4, "Tmin":Tmin, "Tmax":Tmax, "wMax":wMax}

        v1, v2, v3, v4, v5, v6, Tmin, Tmax, wMax, pts = values[13:23]
        if v1:
            _Laliberte_Viscosity_ParametersDict[CASRN] = {"Name":_name, "Formula":_formula,
            "MW":_MW, "V1":v1, "V2":v2, "V3":v3, "V4":v4, "V5":v5, "V6":v6, "Tmin":Tmin, "Tmax":Tmax, "wMax":wMax}

        a1, a2, a3, a4, a5, a6, Tmin, Tmax, wMax, pts = values[23:34]
        if a1:
            _Laliberte_Heat_Capacity_ParametersDict[CASRN] = {"Name":_name, "Formula":_formula,
            "MW":_MW, "A1":a1, "A2":a2, "A3":a3, "A4":a4, "A5":a5, "A6":a6, "Tmin":Tmin, "Tmax":Tmax, "wMax":wMax}
Laliberte_data = pd.read_csv(os.path.join(folder, 'Laliberte2009.tsv'),
                          sep='\t', index_col=0)


### Laliberty Viscosity Functions


def Laliberte_viscosity_w(T):
    r'''Calculate the viscosity of a water using the form proposed by [1]_.
    No parameters are needed, just a temperature. Units are Kelvin and Pa*s.
    t is temperature in degrees Celcius.

    .. math::
        \mu_w = \frac{t + 246}{(0.05594t+5.2842)t + 137.37}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]

    Returns
    -------
    mu_w : float
        Water viscosity, Pa*s

    Notes
    -----
    Original source or pure water viscosity is not cited.
    No temperature range is given for this equation.

    Examples
    --------
    >>> Laliberte_viscosity_w(298)
    0.0008932264487033279

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    t = T-273.15
    mu_w = (t + 246)/((0.05594*t+5.2842)*t + 137.37)
    return mu_w/1000.


def Laliberte_viscosity_i(T, w_w, v1, v2, v3, v4, v5, v6):
    r'''Calculate the viscosity of a solute using the form proposed by [1]_
    Parameters are needed, and a temperature. Units are Kelvin and Pa*s.

    .. math::
        \mu_i = \frac{\exp\left( \frac{v_1(1-w_w)^{v_2}+v_3}{v_4 t +1}\right)}
            {v_5(1-w_w)^{v_6}+1}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    w_w : float
        Weight fraction of water in the solution
    v1-v6 : floats
        Function fit parameters

    Returns
    -------
    mu_i : float
        Solute partial viscosity, Pa*s

    Notes
    -----
    Temperature range check is outside of this function.
    Check is performed using NaCl at 5 degC from the first value in [1]_'s spreadsheet.

    Examples
    --------
    >>> d =  _Laliberte_Viscosity_ParametersDict['7647-14-5']
    >>> Laliberte_viscosity_i(273.15+5, 1-0.005810, d["V1"], d["V2"], d["V3"], d["V4"], d["V5"], d["V6"] )
    0.004254025533308794

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    t = T-273.15
    mu_i = exp((v1*(1-w_w)**v2 + v3)/(v4*t+1))/(v5*(1-w_w)**v6 + 1)
    return mu_i/1000.


def Laliberte_viscosity(T, ws, CASRNs):
    r'''Calculate the viscosity of an aqueous mixture using the form proposed by [1]_.
    Parameters are loaded by the function as needed. Units are Kelvin and Pa*s.

    .. math::
        \mu_m = \mu_w^{w_w} \Pi\mu_i^{w_i}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    ws : array
        Weight fractions of fluid components other than water
    CASRNs : array
        CAS numbers of the fluid components other than water

    Returns
    -------
    mu_i : float
        Solute partial viscosity, Pa*s

    Notes
    -----
    Temperature range check is not used here.
    Check is performed using NaCl at 5 degC from the first value in [1]_'s spreadsheet.

    Examples
    --------
    >>> Laliberte_viscosity(273.15+5, [0.005810], ['7647-14-5'])
    0.0015285828581961414

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    mu_w = Laliberte_viscosity_w(T)*1000.
    w_w = 1 - sum(ws)
    mu = mu_w**(w_w)
    for i in range(len(CASRNs)):
        d = _Laliberte_Viscosity_ParametersDict[CASRNs[i]]
        mu_i = Laliberte_viscosity_i(T, w_w, d["V1"], d["V2"], d["V3"], d["V4"], d["V5"], d["V6"])*1000.
        mu = mu_i**(ws[i])*mu
    return mu/1000.


### Laliberty Density Functions

def Laliberte_density_w(T):
    r'''Calculate the density of water using the form proposed by [1]_.
    No parameters are needed, just a temperature. Units are Kelvin and kg/m^3h.

    .. math::
        \rho_w = \frac{\left\{\left([(-2.8054253\times 10^{-10}\cdot t +
        1.0556302\times 10^{-7})t - 4.6170461\times 10^{-5}]t
        -0.0079870401\right)t + 16.945176   \right\}t + 999.83952}
        {1 + 0.01687985\cdot t}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]

    Returns
    -------
    rho_w : float
        Water density, [kg/m^3]

    Notes
    -----
    Original source not cited
    No temperature range is used.

    Examples
    --------
    >>> Laliberte_density_w(298.15)
    997.0448954179155
    >>> Laliberte_density_w(273.15 + 50)
    988.0362916114763

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    t = T-273.15
    rho_w = (((((-2.8054253E-10*t + 1.0556302E-7)*t - 4.6170461E-5)*t - 0.0079870401)*t + 16.945176)*t + 999.83952) \
        / (1 + 0.01687985*t)
    return rho_w


def Laliberte_density_i(T, w_w, c0, c1, c2, c3, c4):
    r'''Calculate the density of a solute using the form proposed by Laliberte [1]_.
    Parameters are needed, and a temperature, and water fraction. Units are Kelvin and Pa*s.

    .. math::
        \rho_{app,i} = \frac{(c_0[1-w_w]+c_1)\exp(10^{-6}[t+c_4]^2)}
        {(1-w_w) + c_2 + c_3 t}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    w_w : float
        Weight fraction of water in the solution
    c0-c4 : floats
        Function fit parameters

    Returns
    -------
    rho_i : float
        Solute partial density, [kg/m^3]

    Notes
    -----
    Temperature range check is TODO


    Examples
    --------
    >>> d = _Laliberte_Density_ParametersDict['7647-14-5']
    >>> Laliberte_density_i(273.15+0, 1-0.0037838838, d["C0"], d["C1"], d["C2"], d["C3"], d["C4"])
    3761.8917585699983

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    t = T - 273.15
    return ((c0*(1 - w_w)+c1)*exp(1E-6*(t + c4)**2))/((1 - w_w) + c2 + c3*t)


def Laliberte_density(T, ws, CASRNs):
    r'''Calculate the density of an aqueous electrolyte mixture using the form proposed by [1]_.
    Parameters are loaded by the function as needed. Units are Kelvin and Pa*s.

    .. math::
        \rho_m = \left(\frac{w_w}{\rho_w} + \sum_i \frac{w_i}{\rho_{app_i}}\right)^{-1}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    ws : array
        Weight fractions of fluid components other than water
    CASRNs : array
        CAS numbers of the fluid components other than water

    Returns
    -------
    rho_i : float
        Solution density, [kg/m^3]

    Notes
    -----
    Temperature range check is not used here.

    Examples
    --------
    >>> Laliberte_density(273.15, [0.0037838838], ['7647-14-5'])
    1002.6250120185854

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    rho_w = Laliberte_density_w(T)
    w_w = 1 - sum(ws)
    rho = w_w/rho_w
    for i in range(len(CASRNs)):
        d = _Laliberte_Density_ParametersDict[CASRNs[i]]
        rho_i = Laliberte_density_i(T, w_w, d["C0"], d["C1"], d["C2"], d["C3"], d["C4"])
        rho = rho + ws[i]/rho_i
    return 1./rho


### Laliberty Heat Capacity Functions

_T_array = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]
_Cp_array = [4294.03, 4256.88, 4233.58, 4219.44, 4204.95, 4195.45, 4189.1, 4184.8, 4181.9, 4180.02, 4178.95, 4178.86, 4178.77, 4179.56, 4180.89, 4182.77, 4185.17, 4188.1, 4191.55, 4195.52, 4200.01, 4205.02, 4210.57, 4216.64, 4223.23, 4230.36, 4238.07, 4246.37, 4255.28, 4264.84, 4275.08, 4286.04]
Laliberte_heat_capacity_w_interp = interp1d(_T_array, _Cp_array, kind='cubic')

def Laliberte_heat_capacity_w(T):
    r'''Calculate the heat capacity of water using the interpolation proposed by [1]_.
    No parameters are needed, just a temperature.

    .. math::
        Cp_w = Cp_1 + (Cp_2-Cp_1) \left( \frac{t-t_1}{t_2-t_1}\right)
        + \frac{(Cp_3 - 2Cp_2 + Cp_1)}{2}\left( \frac{t-t_1}{t_2-t_1}\right)
        \left( \frac{t-t_1}{t_2-t_1}-1\right)

    Parameters
    ----------
    T : float
        Temperature of fluid [K]

    Returns
    -------
    Cp_w : float
        Water heat capacity, [J/kg/K]

    Notes
    -----
    Units are Kelvin and J/kg/K.
    Original source not cited
    No temperature range is used.
    The original equation is not used, but rather a cubic scipy interpolation routine.

    Examples
    --------
    >>> Laliberte_heat_capacity_w(273.15+3.56)
    4208.878020261102

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    return float(Laliberte_heat_capacity_w_interp(T - 273.15))


def Laliberte_heat_capacity_i(T, w_w, a1, a2, a3, a4, a5, a6):
    r'''Calculate the heat capacity of a solute using the form proposed by [1]_
    Parameters are needed, and a temperature, and water fraction.

    .. math::
        Cp_i = a_1 e^\alpha + a_5(1-w_w)^{a_6}
        \alpha = a_2 t + a_3 \exp(0.01t) + a_4(1-w_w)

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    w_w : float
        Weight fraction of water in the solution
    a1-a6 : floats
        Function fit parameters

    Returns
    -------
    Cp_i : float
        Solute partial heat capacity, [J/kg/K]

    Notes
    -----
    Units are Kelvin and J/kg/K.
    Temperature range check is TODO

    Examples
    --------
    >>> d = _Laliberte_Heat_Capacity_ParametersDict['7647-14-5']
    >>> Laliberte_heat_capacity_i(1.5+273.15, 1-0.00398447, d["A1"], d["A2"], d["A3"], d["A4"], d["A5"], d["A6"])
    -2930.7353945880477

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    t = T - 273.15
    alpha = a2*t + a3*exp(0.01*t) + a4*(1. - w_w)
    Cp_i = a1*exp(alpha) + a5*(1. - w_w)**a6
    return Cp_i*1000.


def Laliberte_heat_capacity(T, ws, CASRNs):
    r'''Calculate the heat capacity of an aqueous electrolyte mixture using the
    form proposed by [1]_.
    Parameters are loaded by the function as needed.

    .. math::
        TODO

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    ws : array
        Weight fractions of fluid components other than water
    CASRNs : array
        CAS numbers of the fluid components other than water

    Returns
    -------
    Cp : float
        Solution heat capacity, [J/kg/K]

    Notes
    -----
    Temperature range check is not implemented.
    Units are Kelvin and J/kg/K.

    Examples
    --------
    >>> Laliberte_heat_capacity(273.15+1.5, [0.00398447], ['7647-14-5']) 
    4186.569908672113

    References
    ----------
    .. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
       Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
       Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
       doi:10.1021/je8008123
    '''
    Cp_w = Laliberte_heat_capacity_w(T)
    w_w = 1 - sum(ws)
    Cp = w_w*Cp_w

    for i in range(len(CASRNs)):
        d = _Laliberte_Heat_Capacity_ParametersDict[CASRNs[i]]
        Cp_i = Laliberte_heat_capacity_i(T, w_w, d["A1"], d["A2"], d["A3"], d["A4"], d["A5"], d["A6"])
        Cp = Cp + ws[i]*Cp_i
    return Cp

#print Laliberte_heat_capacity(298.15, [0.1], ['7664-41-7']) #4186.0988

## Aqueous HCl, trying to find heat capacity of Cl- as H+ is zero.
#zero = Laliberte_heat_capacity(298.15, [0.0000000000000001], ['7647-01-0'])
#small = Laliberte_heat_capacity(298.15, [0.1], ['7647-01-0'])  # 1 molal
#print zero, small
#print (zero-small)*36.46094/100
## cRC gives -136.4 J/mol
## I cannot reproduce this at all.


### Electrical Conductivity


def dilute_ionic_conductivity(ionic_conductivities, zs, rhom):
    r'''This function handles the calculation of the electrical conductivity of 
    a dilute electrolytic aqueous solution. Requires the mole fractions of 
    each ion, the molar density of the whole mixture, and ionic conductivity 
    coefficients for each ion.
    
    .. math::
        \lambda = \sum_i \lambda_i^\circ z_i \rho_m
    
    Parameters
    ----------
    ionic_conductivities : list[float]
        Ionic conductivity coefficients of each ion in the mixture [m^2*S/mol]
    zs : list[float]
        Mole fractions of each ion in the mixture, [-]
    rhom : float
        Overall molar density of the solution, [mol/m^3]

    Returns
    -------
    kappa : float
        Electrical conductivity of the fluid, [S/m]

    Notes
    -----
    The ionic conductivity coefficients should not be `equivalent` coefficients; 
    for example, 0.0053 m^2*S/mol is the equivalent conductivity coefficient of
    Mg+2, but this method expects twice its value - 0.0106. Both are reported
    commonly in literature.
    
    Water can be included in this caclulation by specifying a coefficient of
    0. The conductivity of any electrolyte eclipses its own conductivity by 
    many orders of magnitude. Any other solvents present will affect the
    conductivity extensively and there are few good methods to predict this 
    effect.

    Examples
    --------
    Complex mixture of electrolytes ['Cl-', 'HCO3-', 'SO4-2', 'Na+', 'K+', 
    'Ca+2', 'Mg+2']:
    
    >>> ionic_conductivities = [0.00764, 0.00445, 0.016, 0.00501, 0.00735, 0.0119, 0.01061]
    >>> zs = [0.03104, 0.00039, 0.00022, 0.02413, 0.0009, 0.0024, 0.00103]
    >>> dilute_ionic_conductivity(ionic_conductivities=ionic_conductivities, zs=zs, rhom=53865.9)
    22.05246783663

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics, 95E. Boca Raton, FL: CRC press, 2014.
    '''
    return sum([ci*(zi*rhom) for zi, ci in zip(zs, ionic_conductivities)])


McCleskey_parameters = namedtuple("McCleskey_parameters",
                                  ["Formula", 'lambda_coeffs', 'A_coeffs', 'B', 'multiplier'])

McCleskey_conductivities = {}
with open(os.path.join(folder, 'McCleskey Electrical Conductivity.csv')) as f:
    next(f)
    for line in f:
        values = line.strip().split('\t')
        formula, CASRN, lbt2, lbt, lbc, At2, At, Ac, B, multiplier = to_num(values)
        McCleskey_conductivities[CASRN] = McCleskey_parameters(formula, 
            [lbt2, lbt, lbc], [At2, At, Ac], B, multiplier)


def conductivity_McCleskey(T, M, lambda_coeffs, A_coeffs, B, multiplier, rho=1000.):
    r'''This function handles the calculation of the electrical conductivity of 
    an electrolytic aqueous solution with one electrolyte in solution. It
    handles temperature dependency and concentrated solutions. Requires the 
    temperature of the solution; its molality, and four sets of coefficients
    `lambda_coeffs`, `A_coeffs`, `B`, and `multiplier`.
    
    .. math::
        \Lambda = \frac{\kappa}{C}
        
        \Lambda = \Lambda^0(t) - A(t) \frac{m^{1/2}}{1+Bm^{1/2}}
        
        \Lambda^\circ(t) = c_1 t^2 + c_2 t + c_3
        
        A(t) = d_1 t^2 + d_2 t + d_3
        
    In the above equations, `t` is temperature in degrees Celcius;
    `m` is molality in mol/kg, and C is the concentration of the elctrolytes
    in mol/m^3, calculated as the product of density and molality.
    
    Parameters
    ----------
    T : float
        Temperature of the solution, [K]
    M : float
        Molality of the solution with respect to one electrolyte
        (mol solute / kg solvent), [mol/kg]
    lambda_coeffs : list[float]
        List of coefficients for the polynomial used to calculate `lambda`;
        length-3 coefficients provided in [1]_,  [-]
    A_coeffs : list[float]
        List of coefficients for the polynomial used to calculate `A`;
        length-3 coefficients provided in [1]_, [-]
    B : float
        Empirical constant for an electrolyte, [-]
    multiplier : float
        The multiplier to obtain the absolute conductivity from the equivalent
        conductivity; ex 2 for CaCl2, [-]
    rho : float, optional
        The mass density of the aqueous mixture, [kg/m^3]

    Returns
    -------
    kappa : float
        Electrical conductivity of the solution at the specified molality and 
        temperature [S/m]

    Notes
    -----
    Coefficients provided in [1]_ result in conductivity being calculated in
    units of mS/cm; they are converted to S/m before returned.

    Examples
    --------
    A 0.5 wt% solution of CaCl2, conductivity calculated in mS/cm
    
    >>> conductivity_McCleskey(T=293.15, M=0.045053, A_coeffs=[.03918, 3.905, 
    ... 137.7], lambda_coeffs=[0.01124, 2.224, 72.36], B=3.8, multiplier=2)
    0.8482584585108555

    References
    ----------
    .. [1] McCleskey, R. Blaine. "Electrical Conductivity of Electrolytes Found
       In Natural Waters from (5 to 90) °C." Journal of Chemical & Engineering 
       Data 56, no. 2 (February 10, 2011): 317-27. doi:10.1021/je101012n.
    '''
    t = T - 273.15
    lambda_coeff = horner(lambda_coeffs, t)
    A = horner(A_coeffs, t)
    M_root = M**0.5
    param = lambda_coeff - A*M_root/(1. + B*M_root)
    C = M*rho/1000. # convert to mol/L to get concentration
    return param*C*multiplier*0.1 # convert from mS/cm to S/m




Lange_cond_pure = pd.read_csv(os.path.join(folder, 'Lange Pure Species Conductivity.tsv'),
                          sep='\t', index_col=0)


LANGE_COND = "LANGE_COND"
NONE = 'None'

conductivity_methods = [LANGE_COND]


def conductivity(CASRN=None, AvailableMethods=False, Method=None, full_info=True):
    r'''This function handles the retrieval of a chemical's conductivity.
    Lookup is based on CASRNs. Will automatically select a data source to use
    if no Method is provided; returns None if the data is not available.

    Function has data for approximately 100 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    kappa : float
        Electrical conductivity of the fluid, [S/m]
    T : float, only returned if full_info == True
        Temperature at which conductivity measurement was made
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain RI with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        conductivity_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        conductivity for the desired chemical, and will return methods instead
        of conductivity
    full_info : bool, optional
        If True, function will return the temperature at which the conductivity
        reading was made

    Notes
    -----
    Only one source is available in this function. It is:

        * 'LANGE_COND' which is from Lange's Handbook, Table 8.34 Electrical 
        Conductivity of Various Pure Liquids', a compillation of data in [1]_.

    Examples
    --------
    >>> conductivity('7732-18-5')
    (4e-06, 291.15)

    References
    ----------
    .. [1] Speight, James. Lange's Handbook of Chemistry. 16 edition.
       McGraw-Hill Professional, 2005.
    '''
    def list_methods():
        methods = []
        if CASRN in Lange_cond_pure.index:
            methods.append(LANGE_COND)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    if Method == LANGE_COND:
        kappa = float(Lange_cond_pure.at[CASRN, 'Conductivity'])
        if full_info:
            T = float(Lange_cond_pure.at[CASRN, 'T'])

    elif Method == NONE:
        kappa, T = None, None
    else:
        raise Exception('Failure in in function')

    if full_info:
        return kappa, T
    else:
        return kappa





def thermal_conductivity_Magomedov(T, P, ws, CASRNs, k_w=None):
    r'''Calculate the thermal conductivity of an aqueous mixture of
    electrolytes using the form proposed by Magomedov [1]_.
    Parameters are loaded by the function as needed. Function will fail if an
    electrolyte is not in the database.

    .. math::
        \lambda = \lambda_w\left[ 1 - \sum_{i=1}^n A_i (w_i + 2\times10^{-4}
        w_i^3)\right] - 2\times10^{-8} PT\sum_{i=1}^n w_i

    Parameters
    ----------
    T : float
        Temperature of liquid [K]
    P : float
        Pressure of the liquid [Pa]
    ws : array
        Weight fractions of liquid components other than water
    CASRNs : array
        CAS numbers of the liquid components other than water
    k_w : float
        Liquid thermal condiuctivity or pure water at T and P, [W/m/K]

    Returns
    -------
    kl : float
        Liquid thermal condiuctivity, [W/m/K]

    Notes
    -----
    Range from 273 K to 473 K, P from 0.1 MPa to 100 MPa. C from 0 to 25 mass%.
    Internal untis are MPa for pressure and weight percent.

    An example is sought for this function. It is not possible to reproduce
    the author's values consistently.

    Examples
    --------
    >>> thermal_conductivity_Magomedov(293., 1E6, [.25], ['7758-94-3'], k_w=0.59827)
    0.548654049375

    References
    ----------
    .. [1] Magomedov, U. B. "The Thermal Conductivity of Binary and
       Multicomponent Aqueous Solutions of Inorganic Substances at High
       Parameters of State." High Temperature 39, no. 2 (March 1, 2001):
       221-26. doi:10.1023/A:1017518731726.
    '''
    P = P/1E6
    ws = [i*100 for i in ws]
    if not k_w:
        raise Exception('k_w correlation must be provided')

    sum1 = 0
    for i, CASRN in enumerate(CASRNs):
        Ai = float(Magomedovk_thermal_cond.at[CASRN, 'Ai'])
        sum1 += Ai*(ws[i] + 2E-4*ws[i]**3)
    return k_w*(1 - sum1) - 2E-8*P*T*sum(ws)


def ionic_strength(mis, zis):
    r'''Calculate the ionic strength of a solution in one of two ways,
    depending on the inputs only. For Pitzer and Bromley models,
    `mis` should be molalities of each component. For eNRTL models,
    `mis` should be mole fractions of each electrolyte in the solution.
    This will sum to be much less than 1.

    .. math::
        I = \frac{1}{2} \sum M_i z_i^2

        I = \frac{1}{2} \sum x_i z_i^2

    Parameters
    ----------
    mis : list
        Molalities of each ion, or mole fractions of each ion [mol/kg or -]
    zis : list
        Charges of each ion [-]

    Returns
    -------
    I : float
        ionic strength, [?]

    Examples
    --------
    >>> ionic_strength([0.1393, 0.1393], [1, -1])
    0.1393

    References
    ----------
    .. [1] Chen, Chau-Chyun, H. I. Britt, J. F. Boston, and L. B. Evans. "Local
       Composition Model for Excess Gibbs Energy of Electrolyte Systems.
       Part I: Single Solvent, Single Completely Dissociated Electrolyte
       Systems." AIChE Journal 28, no. 4 (July 1, 1982): 588-96.
       doi:10.1002/aic.690280410
    .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process Simulation.
       Weinheim, Germany: Wiley-VCH, 2012.
    '''
    return 0.5*sum([mi*zi*zi for mi, zi in zip(mis, zis)])


def Kweq_1981(T, rho_w):
    r'''Calculates equilibrium constant for OH- and H+ in water, according to
    [1]_. Second most recent formulation.

    .. math::
        \log_{10} K_w= A + B/T + C/T^2 + D/T^3 + (E+F/T+G/T^2)\log_{10} \rho_w

    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    rho_w : float
        Density of water, [kg/m^3]

    Returns
    -------
    Kweq : float
        Ionization constant of water, [-]

    Notes
    -----
    Density is internally converted to units of g/cm^3.

    A = -4.098;
    B = -3245.2;
    C = 2.2362E5;
    D = -3.984E7;
    E = 13.957;
    F = -1262.3;
    G = 8.5641E5

    Examples
    --------
    >>> -1*log10(Kweq_1981(600, 700))
    11.274522047458206
    
    References
    ----------
    .. [1] Marshall, William L., and E. U. Franck. "Ion Product of Water
       Substance, 0-1000  degree C, 1010,000 Bars New International Formulation
       and Its Background." Journal of Physical and Chemical Reference Data 10,
       no. 2 (April 1, 1981): 295-304. doi:10.1063/1.555643.
    '''
    rho_w = rho_w/1000.
    A = -4.098
    B = -3245.2
    C = 2.2362E5
    D = -3.984E7
    E = 13.957
    F = -1262.3
    G = 8.5641E5
    return 10**(A + B/T + C/T**2 + D/T**3 + (E + F/T + G/T**2)*log10(rho_w))


def Kweq_IAPWS_gas(T):
    r'''Calculates equilibrium constant for OH- and H+ in water vapor,
    according to [1]_.
    This is the most recent formulation available.

    .. math::
        -log_{10}  K_w^G = \gamma_0 + \gamma_1 T^{-1} + \gamma_2 T^{-2} + \gamma_3 T^{-3}

    Parameters
    ----------
    T : float
        Temperature of H2O [K]

    Returns
    -------
    K_w_G : float

    Notes
    -----
    gamma0 = 6.141500E-1; 
    gamma1 = 4.825133E4; 
    gamma2 = -6.770793E4; 
    gamma3 = 1.010210E7

    Examples
    --------
    >>> Kweq_IAPWS_gas(800)
    1.4379721554798815e-61
    
    References
    ----------
    .. [1] Bandura, Andrei V., and Serguei N. Lvov. "The Ionization Constant
       of Water over Wide Ranges of Temperature and Density." Journal of Physical
       and Chemical Reference Data 35, no. 1 (March 1, 2006): 15-30.
       doi:10.1063/1.1928231
    '''
    gamma0 = 6.141500E-1
    gamma1 = 4.825133E4
    gamma2 = -6.770793E4
    gamma3 = 1.010210E7
    K_w_G = 10**(-1*(gamma0 + gamma1/T + gamma2/T**2 + gamma3/T**3))
    return K_w_G


def Kweq_IAPWS(T, rho_w):
    r'''Calculates equilibrium constant for OH- and H+ in water, according to
    [1]_.
    This is the most recent formulation available.

    .. math::
        Q = \rho \exp(\alpha_0 + \alpha_1 T^{-1} + \alpha_2 T^{-2} \rho^{2/3})
        
        - \log_{10} K_w = -2n \left[ \log_{10}(1+Q) - \frac{Q}{Q+1} \rho 
        (\beta_0 + \beta_1 T^{-1} + \beta_2 \rho) \right]
        -\log_{10} K_w^G + 2 \log_{10} \frac{18.015268}{1000}

    Parameters
    ----------
    T : float
        Temperature of water [K]
    rho_w : float
        Density of water at temperature and pressure [kg/m^3]

    Returns
    -------
    Kweq : float
        Ionization constant of water, [-]

    Notes
    -----
    Formulation is in terms of density in g/cm^3; density
    is converted internally.
    
    n = 6;
    alpha0 = -0.864671;
    alpha1 = 8659.19;
    alpha2 = -22786.2;
    beta0 = 0.642044;
    beta1 = -56.8534;
    beta2 = -0.375754

    Examples
    --------
    Example from IAPWS check:
    
    >>> -1*log10(Kweq_IAPWS(600, 700))
    11.203153057603775

    References
    ----------
    .. [1] Bandura, Andrei V., and Serguei N. Lvov. "The Ionization Constant
       of Water over Wide Ranges of Temperature and Density." Journal of Physical
       and Chemical Reference Data 35, no. 1 (March 1, 2006): 15-30.
       doi:10.1063/1.1928231
    '''
    K_w_G = Kweq_IAPWS_gas(T)
    rho_w = rho_w/1000.
    n = 6
    alpha0 = -0.864671
    alpha1 = 8659.19
    alpha2 = -22786.2
    beta0 = 0.642044
    beta1 = -56.8534
    beta2 = -0.375754

    Q = rho_w*exp(alpha0 + alpha1/T + alpha2/T**2*rho_w**(2/3.))
    K_w = 10**(-1*(-2*n*(log10(1+Q)-Q/(Q+1) * rho_w *(beta0 + beta1/T + beta2*rho_w)) -
    log10(K_w_G) + 2*log10(18.015268/1000) ))
    return K_w



charge_balance_methods = ['dominant', 'decrease dominant', 'increase dominant',
                          'proportional insufficient ions increase', 
                          'proportional excess ions decrease', 
                          'proportional cation adjustment', 
                          'proportional anion adjustment', 'Na or Cl increase',
                          'Na or Cl decrease', 'adjust', 'increase', 
                          'decrease', 'makeup']

def ion_balance_adjust_wrapper(charges, zs, n_anions, n_cations,
                               anions, cations, selected_ion, increase=None):
    charge = selected_ion.charge
    positive = charge > 0
    if charge == 0:  # pragma: no cover
        raise Exception('Cannot adjust selected compound as it has no charge!')
    assert charge != 0.0
    
    if selected_ion not in anions and selected_ion not in cations:
        if charge < 0.:
            anions.append(selected_ion)
            charges.insert(n_anions, charge)
            zs.insert(n_anions, 0.)
            n_anions += 1
            adjust = n_anions - 1
            anion_index = n_anions - 1
        else:
            cations.append(selected_ion)
            charges.insert(-1, charge)
            zs.insert(-1, 0.)
            n_cations += 1
            cation_index = n_cations - 1
            adjust = n_anions + n_cations - 1
        old_zi = 0
    else:
        if selected_ion in anions:
            anion_index = anions.index(selected_ion)
            old_zi = zs[anion_index]
            adjust = anion_index
        else:
            cation_index = cations.index(selected_ion)
            old_zi = zs[n_anions + cation_index]
            adjust = n_anions + cation_index
    anion_zs, cation_zs, z_water = ion_balance_adjust_one(charges, zs, n_anions, n_cations, adjust=adjust)
    new_zi = cation_zs[cation_index] if positive else anion_zs[anion_index]
    if increase == True and new_zi < old_zi:
        raise Exception('Adjusting specified ion %s resulted in a decrease of its quantity but an increase was specified' % selected_ion.formula)
    elif increase == False and new_zi > old_zi:
        raise Exception('Adjusting specified ion %s resulted in a increase of its quantity but an decrease was specified' % selected_ion.formula)
    return anion_zs, cation_zs, z_water


def ion_balance_adjust_one(charges, zs, n_anions, n_cations, adjust):
    main_tot = sum([zs[i]*charges[i] for i in range(len(charges)) if i != adjust])
    zs[adjust] = -main_tot/charges[adjust]
    if zs[adjust] < 0:
        raise Exception('A negative value of %f ion mole fraction was required to balance the charge' %zs[adjust])
    
    z_water = 1. - sum(zs[0:-1])
    anion_zs = zs[0:n_anions]
    cation_zs = zs[n_anions:n_cations+n_anions]
    return anion_zs, cation_zs, z_water


def ion_balance_dominant(impacts, balance_error, charges, zs, n_anions, 
                         n_cations, method):
    if method == 'dominant':
        # Highest concentration species in the inferior type always gets adjusted, up or down regardless
        low = min(impacts)
        high = max(impacts)
        if abs(low) > high:
            adjust = impacts.index(low)
        else:
            adjust = impacts.index(high)
    elif method == 'decrease dominant':        
        if balance_error < 0:
            # Decrease the dominant anion
            adjust = impacts.index(min(impacts))
        else:
             # Decrease the dominant cation
            adjust = impacts.index(max(impacts))
    elif method == 'increase dominant':
        if balance_error < 0:
            adjust = impacts.index(max(impacts))
        else:
            adjust = impacts.index(min(impacts))
    else:
        raise Exception('Allowable methods are %s' %charge_balance_methods)
    return ion_balance_adjust_one(charges, zs, n_anions, n_cations, adjust)


def ion_balance_proportional(anion_charges, cation_charges, zs, n_anions, 
                             n_cations, balance_error, method):
    '''Helper method for balance_ions for the proportional family of methods. 
    See balance_ions for a description of the methods; parameters are fairly
    obvious.
    '''
    anion_zs = zs[0:n_anions]
    cation_zs = zs[n_anions:n_cations+n_anions]
    anion_balance_error = sum([zi*ci for zi, ci in zip(anion_zs, anion_charges)])
    cation_balance_error = sum([zi*ci for zi, ci in zip(cation_zs, cation_charges)])
    if method == 'proportional insufficient ions increase':
        if balance_error < 0:
            multiplier = -anion_balance_error/cation_balance_error
            cation_zs = [i*multiplier for i in cation_zs]
        else:
            multiplier = -cation_balance_error/anion_balance_error
            anion_zs = [i*multiplier for i in anion_zs]
    elif method == 'proportional excess ions decrease':
        if balance_error < 0:
            multiplier = -cation_balance_error/anion_balance_error
            anion_zs = [i*multiplier for i in anion_zs]
        else:
            multiplier = -anion_balance_error/cation_balance_error
            cation_zs = [i*multiplier for i in cation_zs]
    elif method == 'proportional cation adjustment':
        multiplier = -anion_balance_error/cation_balance_error
        cation_zs = [i*multiplier for i in cation_zs]
    elif method == 'proportional anion adjustment':
        multiplier = -cation_balance_error/anion_balance_error
        anion_zs = [i*multiplier for i in anion_zs]
    else:
        raise Exception('Allowable methods are %s' %charge_balance_methods)
    z_water = 1. - sum(anion_zs) - sum(cation_zs)
    return anion_zs, cation_zs, z_water


def balance_ions(anions, cations, anion_zs=None, cation_zs=None, 
                 anion_concs=None, cation_concs=None, rho_w=997.1, 
                 method='increase dominant', selected_ion=None):
    r'''Performs an ion balance to adjust measured experimental ion 
    compositions to electroneutrality. Can accept either the actual mole 
    fractions of the ions, or their concentrations in units of [mg/L] as well
    for convinience.
    
    The default method will locate the most prevalent ion in the type of 
    ion not in excess - and increase it until the two ion types balance.

    Parameters
    ----------
    anions : list(ChemicalMetadata)
        List of all negatively charged ions measured as being in the solution;
        ChemicalMetadata instances or simply objects with the attributes `MW` 
        and `charge`, [-]
    cations : list(ChemicalMetadata)
        List of all positively charged ions measured as being in the solution;
        ChemicalMetadata instances or simply objects with the attributes `MW` 
        and `charge`, [-]
    anion_zs : list, optional
        Mole fractions of each anion as measured in the aqueous solution, [-]
    cation_zs : list, optional
        Mole fractions of each cation as measured in the aqueous solution, [-]
    anion_concs : list, optional
        Concentrations of each anion in the aqueous solution in the units often
        reported (for convinience only) [mg/L]
    cation_concs : list, optional
        Concentrations of each cation in the aqueous solution in the units 
        often reported (for convinience only) [mg/L]
    rho_w : float, optional
        Density of the aqueous solutionr at the temperature and pressure the
        anion and cation concentrations were measured (if specified), [kg/m^3]
    method : str, optional
        The method to use to balance the ionimbalance; one of 'dominant', 
        'decrease dominant', 'increase dominant',
        'proportional insufficient ions increase', 
        'proportional excess ions decrease', 
        'proportional cation adjustment', 'proportional anion adjustment', 
        'Na or Cl increase', 'Na or Cl decrease', 'adjust', 'increase', 
        'decrease', 'makeup'].
    selected_ion : ChemicalMetadata, optional
        Some methods adjust only one user-specified ion; this is that input.
        For the case of the 'makeup' method, this is a tuple of (anion, cation)
        ChemicalMetadata instances and only the ion type not in excess will be
        used.

    Returns
    -------
    anions : list(ChemicalMetadata)
        List of all negatively charged ions measured as being in the solution;
        ChemicalMetadata instances after potentially adding in an ion which
        was not present but specified by the user, [-]
    cations : list(ChemicalMetadata)
        List of all positively charged ions measured as being in the solution;
        ChemicalMetadata instances after potentially adding in an ion which
        was not present but specified by the user, [-]
    anion_zs : list,
        Mole fractions of each anion in the aqueous solution after the charge
        balance, [-]
    cation_zs : list
        Mole fractions of each cation in the aqueous solution after the charge
        balance, [-]
    z_water : float
        Mole fraction of the water in the solution, [-]

    Notes
    -----
    The methods perform the charge balance as follows:
        
    * 'dominant' : The ion with the largest mole fraction in solution has its
      concentration adjusted up or down as necessary to balance the solution.
    * 'decrease dominant' : The ion with the largest mole fraction in the type
      of ion with *excess* charge has its own mole fraction decreased to balance
      the solution.
    * 'increase dominant' : The ion with the largest mole fraction in the type
      of ion with *insufficient* charge has its own mole fraction decreased to 
      balance the solution.
    * 'proportional insufficient ions increase' : The ion charge type which is
      present insufficiently has each of the ions mole fractions *increased*
      proportionally until the solution is balanced.
    * 'proportional excess ions decrease' :  The ion charge type which is
      present in excess has each of the ions mole fractions *decreased*
      proportionally until the solution is balanced.
    * 'proportional cation adjustment' : All *cations* have their mole fractions
      increased or decreased proportionally as necessary to balance the 
      solution.
    * 'proportional anion adjustment' : All *anions* have their mole fractions
      increased or decreased proportionally as necessary to balance the 
      solution.
    * 'Na or Cl increase' : Either Na+ or Cl- is *added* to the solution until
      the solution is balanced; the species will be added if they were not
      present initially as well.
    * 'Na or Cl decrease' : Either Na+ or Cl- is *removed* from the solution 
      until the solution is balanced; the species will be added if they were 
      not present initially as well.
    * 'adjust' : An ion specified with the parameter `selected_ion` has its
      mole fraction *increased or decreased* as necessary to balance the 
      solution. An exception is raised if the specified ion alone cannot 
      balance the solution.
    * 'increase' : An ion specified with the parameter `selected_ion` has its
      mole fraction *increased* as necessary to balance the 
      solution. An exception is raised if the specified ion alone cannot 
      balance the solution.
    * 'decrease' : An ion specified with the parameter `selected_ion` has its
      mole fraction *decreased* as necessary to balance the 
      solution. An exception is raised if the specified ion alone cannot 
      balance the solution.
    * 'makeup' : Two ions ase specified as a tuple with the parameter 
      `selected_ion`. Whichever ion type is present in the solution 
      insufficiently is added; i.e. if the ions were Mg+2 and Cl-, and there
      was too much negative charge in the solution, Mg+2 would be added until
      the solution was balanced.
        
    Examples
    --------
    >>> anions_n = ['Cl-', 'HCO3-', 'SO4-2']
    >>> cations_n = ['Na+', 'K+', 'Ca+2', 'Mg+2']
    >>> cations = [pubchem_db.search_name(i) for i in cations_n]
    >>> anions = [pubchem_db.search_name(i) for i in anions_n]
    >>> an_res, cat_res, an_zs, cat_zs, z_water = balance_ions(anions, cations,
    ... anion_zs=[0.02557, 0.00039, 0.00026], cation_zs=[0.0233, 0.00075,
    ... 0.00262, 0.00119], method='proportional excess ions decrease')
    >>> an_zs
    [0.02557, 0.00039, 0.00026]
    >>> cat_zs
    [0.01948165456267761, 0.0006270918850647299, 0.0021906409851594564, 0.0009949857909693717]
    >>> z_water
    0.9504856267761288
    
    References
    ----------
    '''
    anions = list(anions)
    cations = list(cations)
    n_anions = len(anions)
    n_cations = len(cations)
    ions = anions + cations
    anion_charges = [i.charge for i in anions]
    cation_charges = [i.charge for i in cations]
    charges = anion_charges + cation_charges + [0]

    MW_water = [18.01528]
    rho_w = rho_w/1000 # Convert to kg/liter
    
    if anion_concs is not None and cation_concs is not None:
        anion_ws = [i*1E-6/rho_w for i in anion_concs]
        cation_ws = [i*1E-6/rho_w for i in cation_concs]
        w_water = 1 - sum(anion_ws) - sum(cation_ws)
    
        anion_MWs = [i.MW for i in anions]
        cation_MWs = [i.MW for i in cations]
        MWs = anion_MWs + cation_MWs + MW_water
        zs = ws_to_zs(anion_ws + cation_ws + [w_water], MWs)
    else:
        if anion_zs is None or cation_zs is None:
            raise Exception('Either both of anion_concs and cation_concs or '
                            'anion_zs and cation_zs must be specified.')
        else:
            zs = anion_zs + cation_zs
            zs = zs + [1 - sum(zs)]
    
    impacts = [zi*ci for zi, ci in zip(zs, charges)]
    balance_error = sum(impacts)
    
    
    if abs(balance_error) < 1E-7:
        anion_zs = zs[0:n_anions]
        cation_zs = zs[n_anions:n_cations+n_anions]
        z_water = zs[-1]
        return anions, cations, anion_zs, cation_zs, z_water
    if 'dominant' in method:
        anion_zs, cation_zs, z_water = ion_balance_dominant(impacts,
            balance_error, charges, zs, n_anions, n_cations, method)
        return anions, cations, anion_zs, cation_zs, z_water
    elif 'proportional' in method:
        anion_zs, cation_zs, z_water = ion_balance_proportional(
            anion_charges, cation_charges, zs, n_anions, n_cations,
            balance_error, method)
        return anions, cations, anion_zs, cation_zs, z_water
    elif method == 'Na or Cl increase':
        increase = True
        if balance_error < 0:
            selected_ion = pubchem_db.search_name('Na+')
        else:
            selected_ion = pubchem_db.search_name('Cl-')
    elif method == 'Na or Cl decrease':
        increase = False
        if balance_error > 0:
            selected_ion = pubchem_db.search_name('Na+')
        else:
            selected_ion = pubchem_db.search_name('Cl-')
    # All of the below work with the variable selected_ion
    elif method == 'adjust':
        # A single ion will be increase or decreased to fix the balance automatically
        increase = None
    elif method == 'increase':
        increase = True
        # Raise exception if approach doesn't work
    elif method == 'decrease':
        increase = False
        # Raise exception if approach doesn't work
    elif method == 'makeup':
        # selected ion starts out as a tuple in this case; always adding the compound
        increase = True
        if balance_error < 0:
            selected_ion = selected_ion[1]
        else:
            selected_ion = selected_ion[0]
    else:
        raise Exception('Method not recognized')
    if selected_ion is None:
        raise Exception("For methods 'adjust', 'increase', 'decrease', and "
                        "'makeup', an ion must be specified with the "
                        "`selected_ion` parameter")
        
    anion_zs, cation_zs, z_water = ion_balance_adjust_wrapper(charges, zs, n_anions, n_cations,
                                                              anions, cations, selected_ion, increase=increase)
    return anions, cations, anion_zs, cation_zs, z_water


