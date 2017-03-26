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
           'Laliberte_heat_capacity_i', 'Lange_cond_pure', 
           'conductivity_methods', 'Magomedovk_thermal_cond',
           'thermal_conductivity_Magomedov', 'ionic_strength', 'Kweq_1981', 
           'Kweq_IAPWS_gas', 'Kweq_IAPWS']

import os
from thermo.utils import exp, log10
from thermo.utils import e, N_A
from thermo.utils import to_num
from scipy.interpolate import interp1d
import pandas as pd


F = e*N_A


folder = os.path.join(os.path.dirname(__file__), 'Electrolytes')


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


Magomedovk_thermal_cond = pd.read_csv(os.path.join(folder, 'Magomedov Thermal Conductivity.tsv'),
                          sep='\t', index_col=0)


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
