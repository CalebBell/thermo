# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['solubility_parameter_methods', 'solubility_parameter', 
           'solubility_eutectic', 'Tm_depression_eutectic',
           'Henry_converter', 'Henry_pressure', 'Henry_pressure_mixture']
           
import os
from fluids.constants import R, atm
from thermo.utils import log, exp

folder = os.path.join(os.path.dirname(__file__), 'Solubility')


#def MolarSolubility(CASRN=None, Pi=101325):
#    '''Assume molarity'''
#    data = _HenrysLawSiteDict[CASRN]
#    Pi = Pi/101325. # Adjust for atmosphere unit
#    molarity = Pi*data["HA"]/rho_w_298
#    return molarity

#print MolarSolubility(CASRN='7446-09-5', Pi=101325*.025)
#print MolarSolubility(CASRN='10049-04-4', Pi=101325*.16*.03)*67.4518

#def fit_m(CASRN=None, MW=None, Pmin=10000, Pmax=1E5, PT=None):
#    '''m is in mole fraction gas'''
#    if not MW or not CASRN:
#        raise Exception('CAS number and MW are required for this function')
#    if not PT:
#        PT = Pmax
#    Ps = np.linspace(Pmin, Pmax, 100)
#    Xs = []
#    for Pi in Ps:
#        Ci = MolarSolubility(CASRN, Pi=Pi)
#        Xs.append(Ci/((1000.-Ci*MW)/MW_w))
#    Ys = [Pi/PT for Pi in Ps]
#    ms = [(Ys[i]/Xs[i])**2 for i in range(len(Xs))]
#    m = np.average(ms)**0.5
##    plt.plot(Xs, Ys)
##    plt.show()
#    return m

# Reference 2521 of acp-15-4399-2015 might be the best source

#MSO2 = 64.0638
#print fit_m('1310-73-2', 40)

DEFINITION = 'DEFINITION'
NONE = 'NONE'
solubility_parameter_methods = [DEFINITION]


def solubility_parameter(T=298.15, Hvapm=None, Vml=None,
                         CASRN='', AvailableMethods=False, Method=None):
    r'''This function handles the calculation of a chemical's solubility
    parameter. Calculation is a function of temperature, but is not always
    presented as such. No lookup values are available; either `Hvapm`, `Vml`,
    and `T` are provided or the calculation cannot be performed.

    .. math::
        \delta = \sqrt{\frac{\Delta H_{vap} - RT}{V_m}}

    Parameters
    ----------
    T : float
        Temperature of the fluid [k]
    Hvapm : float
        Heat of vaporization [J/mol/K]
    Vml : float
        Specific volume of the liquid [m^3/mol]
    CASRN : str, optional
        CASRN of the fluid, not currently used [-]

    Returns
    -------
    delta : float
        Solubility parameter, [Pa^0.5]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain the solubility parameter
        with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        solubility_parameter_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        the solubility parameter for the desired chemical, and will return
        methods instead of the solubility parameter

    Notes
    -----
    Undefined past the critical point. For convenience, if Hvap is not defined,
    an error is not raised; None is returned instead. Also for convenience,
    if Hvapm is less than RT, None is returned to avoid taking the root of a
    negative number.

    This parameter is often given in units of cal/ml, which is 2045.48 times
    smaller than the value returned here.

    Examples
    --------
    Pentane at STP

    >>> solubility_parameter(T=298.2, Hvapm=26403.3, Vml=0.000116055)
    14357.681538173534

    References
    ----------
    .. [1] Barton, Allan F. M. CRC Handbook of Solubility Parameters and Other
       Cohesion Parameters, Second Edition. CRC Press, 1991.
    '''
    def list_methods():
        methods = []
        if T and Hvapm and Vml:
            methods.append(DEFINITION)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == DEFINITION:
        if (not Hvapm) or (not T) or (not Vml):
            delta = None
        else:
            if Hvapm < R*T or Vml < 0:  # Prevent taking the root of a negative number
                delta = None
            else:
                delta = ((Hvapm - R*T)/Vml)**0.5
    elif Method == NONE:
        delta = None
    else:
        raise Exception('Failure in in function')
    return delta


def solubility_eutectic(T, Tm, Hm, Cpl=0, Cps=0, gamma=1):
    r'''Returns the maximum solubility of a solute in a solvent.

    .. math::
        \ln x_i^L \gamma_i^L = \frac{\Delta H_{m,i}}{RT}\left(
        1 - \frac{T}{T_{m,i}}\right) - \frac{\Delta C_{p,i}(T_{m,i}-T)}{RT}
        + \frac{\Delta C_{p,i}}{R}\ln\frac{T_m}{T}

        \Delta C_{p,i} = C_{p,i}^L - C_{p,i}^S

    Parameters
    ----------
    T : float
        Temperature of the system [K]
    Tm : float
        Melting temperature of the solute [K]
    Hm : float
        Heat of melting at the melting temperature of the solute [J/mol]
    Cpl : float, optional
        Molar heat capacity of the solute as a liquid [J/mol/K]
    Cpls: float, optional
        Molar heat capacity of the solute as a solid [J/mol/K]
    gamma : float, optional
        Activity coefficient of the solute as a liquid [-]

    Returns
    -------
    x : float
        Mole fraction of solute at maximum solubility [-]

    Notes
    -----
    gamma is of the solute in liquid phase

    Examples
    --------
    From [1]_, matching example

    >>> solubility_eutectic(T=260., Tm=278.68, Hm=9952., Cpl=0, Cps=0, gamma=3.0176)
    0.24340068761677464

    References
    ----------
    .. [1] Gmehling, Jurgen. Chemical Thermodynamics: For Process Simulation.
       Weinheim, Germany: Wiley-VCH, 2012.
    '''
    dCp = Cpl-Cps
    x = exp(- Hm/R/T*(1-T/Tm) + dCp*(Tm-T)/R/T - dCp/R*log(Tm/T))/gamma
    return x

#print solubility_eutectic(T=293.15, Tm=489.6, Hm=28860., Cpl=0, Cps=0, gamma=1)
#print [solubility_eutectic(T=293.15, Tm=369.4, Hm=18640., Cpl=0, Cps=0, gamma=1)]

#print [solubility_eutectic(T=260., Tm=278.68, Hm=9952., Cpl=0, Cps=0, gamma=3.0176)] # 0.243400708394

#print UNIQUAC(T=260., xs=[.7566, .2434], rs=[2.1055, 3.1878], qs=[1.972, 2.4],
#              umat=[[0, -43.],[384.09, 0]])

#def err(x):
#    from thermo.activity import UNIQUAC
#    gamma = UNIQUAC(T=260., xs=[1-x, x], rs=[2.1055, 3.1878], qs=[1.972, 2.4],
#              umat=[[0, -43.],[384.09, 0]])[1]
#    x2 = solubility_eutectic(T=260., Tm=278.68, Hm=9952., Cpl=0, Cps=0, gamma=gamma)
#    return (x-x2)**2
#
#from scipy.optimize import fsolve
#print fsolve(err, .9)
#[ 0.24340135]


def Tm_depression_eutectic(Tm, Hm, x=None, M=None, MW=None):
    r'''Returns the freezing point depression caused by a solute in a solvent.
    Can use either the mole fraction of the solute or its molality and the
    molecular weight of the solvent. Assumes ideal system behavior.

    .. math::
        \Delta T_m = \frac{R T_m^2 x}{\Delta H_m}

        \Delta T_m = \frac{R T_m^2 (MW) M}{1000 \Delta H_m}

    Parameters
    ----------
    Tm : float
        Melting temperature of the solute [K]
    Hm : float
        Heat of melting at the melting temperature of the solute [J/mol]
    x : float, optional
        Mole fraction of the solute [-]
    M : float, optional
        Molality [mol/kg]
    MW: float, optional
        Molecular weight of the solvent [g/mol]

    Returns
    -------
    dTm : float
        Freezing point depression [K]

    Notes
    -----
    MW is the molecular weight of the solvent. M is the molality of the solute.

    Examples
    --------
    From [1]_, matching example.

    >>> Tm_depression_eutectic(353.35, 19110, .02)
    1.0864594900639515

    References
    ----------
    .. [1] Gmehling, Jurgen. Chemical Thermodynamics: For Process Simulation.
       Weinheim, Germany: Wiley-VCH, 2012.
    '''
    if x:
        dTm = R*Tm**2*x/Hm
    elif M and MW:
        MW = MW/1000. #g/mol to kg/mol
        dTm = R*Tm**2*MW*M/Hm
    else:
        raise Exception('Either molality or mole fraction of the solute must be specified; MW of the solvent is required also if molality is provided')
    return dTm


HENRY_SCALES_HCP = ('Hcp', 'mol/(m^3*Pa)')
HENRY_SCALES_HCP_MOLALITY = ('M/atm',)
HENRY_SCALES_HCC = ('Hcc',)
HENRY_SCALES_HBP_SI = ('mol/(kg*Pa)',)
HENRY_SCALES_HBP = ('Hbp', 'mol/(kg*atm)')
HENRY_SCALES_HXP = ('Hxp', '1/atm')
HENRY_SCALES_BUNSEN = ('alpha', 'bunsen coefficient')

HENRY_SCALES_KHPX = ('KHpx', 'atm')
HENRY_SCALES_KHPC_SI =  ('m^3*Pa/mol',)
HENRY_SCALES_KHPC = ('KHpc', 'm^3*atm/mol')
HENRY_SCALES_KHCC = ('KHcc',)
HENRY_SCALES_SI = ('SI',)

HENRY_SCALES_LIST = (HENRY_SCALES_HCP + HENRY_SCALES_HCP_MOLALITY 
                     + HENRY_SCALES_HCC + HENRY_SCALES_HBP_SI + HENRY_SCALES_HBP
                     + HENRY_SCALES_HXP + HENRY_SCALES_BUNSEN + HENRY_SCALES_KHPX
                     + HENRY_SCALES_KHPC_SI + HENRY_SCALES_KHPC + HENRY_SCALES_KHCC
                     + HENRY_SCALES_SI)


def Henry_converter(val, old_scale, new_scale, rhom=None, MW=None):
    r'''Converts Henry's law constant for a gas with respect to a solvent from 
    one scale to another. 
    
    There are many scales, but it is recommemed to operate in the scale of
    `SI` - which returns a value with units `Pa`, and directly gets used in
    place of vapor pressure inside a flash calculation. This removes the
    complexity of Henry's law, avoiding possible simplication in favor of use
    with other thermodynamic models.
    
    Only some scales require the molecular weight and the molar density of the
    solvent. Values for water, the most common solute, are 55344.59 mol/m^3 at
    STP and 18.01528 g/mol.
        
    Parameters
    ----------
    val : float
        Henry's law constant, various units
    old_scale : str
        String representing the scale that `val` is in originally.
    new_scale : str
        String representing the scale that `val` should be converted to.
        
    Returns
    -------
    result : float
        Input `val` converted from `old_scale` to `new_scale`, various units

    Notes
    -----
    The valid scales for this function are any of the following:
        
    ('Hcp', 'mol/(m^3*Pa)', 'M/atm', 'Hcc', 'mol/(kg*Pa)', 'Hbp', 
    'mol/(kg*atm)', 'Hxp', '1/atm', 'alpha', 'bunsen coefficient', 'KHpx',
    'atm', 'm^3*Pa/mol', 'KHpc', 'm^3*atm/mol', 'KHcc', 'SI')
    
    Examples
    --------
    >>> Henry_converter(1.2e-5, old_scale='Hcp', new_scale='SI', rhom=55344.59, 
    ... MW=18.01528)
    4612049166.666666
    
    >>> Henry_converter(0.0297475, old_scale='Hcc', new_scale='KHcc', 
    ... rhom=55344.59, MW=18.01528)
    33.61627027481301

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
       8E. McGraw-Hill Professional, 2007.
    '''
    # Convert val to units of 1/atm
    if old_scale in HENRY_SCALES_HCP:
        factor = atm/rhom
        Hxp = val*factor # multiplication of 1.83089 
    elif old_scale in HENRY_SCALES_HCP_MOLALITY: # Hcp in Molarity
        factor = 1000.0/rhom
        Hxp = val*factor # multiplication of 0.0180695 
    elif old_scale in HENRY_SCALES_HCC:
        # Aqueous concentration divided by gas concentration
        factor = (atm/(R*298.15*rhom))
        Hxp = val*factor # multiplication of 7.38573E-4
    elif old_scale in HENRY_SCALES_HBP_SI: # Hbp in SI
        rho = 1e-3*rhom*MW  # Vm_to_rho(1.0/rhom, MW)
        factor = rho*atm/rhom
        Hxp = val*factor# Multiplication of 1825.40 
    elif old_scale in HENRY_SCALES_HBP:
        rho = 1e-3*rhom*MW  # Vm_to_rho(1.0/rhom, MW)
        factor = rho/rhom
        Hxp = val*factor # multiplication of 0.0180153 
    elif old_scale in HENRY_SCALES_HXP:
        Hxp = val
    elif old_scale in HENRY_SCALES_BUNSEN:
        factor = atm/(R*273.15*rhom)
        Hxp = val*factor # multiplication of 8.06171E-4
    # Volatility constants
    elif old_scale in HENRY_SCALES_KHPX:
        Hxp = 1.0/val
    elif old_scale in HENRY_SCALES_KHPC_SI:
        factor = atm/rhom
        Hxp = factor/val # multiplication of 0.546182 
    elif old_scale in HENRY_SCALES_KHPC:
        factor = 1.0/rhom
        Hxp = factor/val # multiplication of 55341.9 
    elif old_scale in HENRY_SCALES_KHCC:
        factor = atm/(R*298.15*rhom) # gas concentration divided by Aqueous concentration 
        Hxp = factor/val # Multiplication of 1353.96 
    elif old_scale in HENRY_SCALES_SI:
        Hxp = atm/val
    else:
        raise ValueError("Not recognized input scale: %s" %old_scale)
    
    # Convert from the constant `Hxp` to the desired unit
    if new_scale in HENRY_SCALES_HCP:
        factor = atm/rhom
        conv_val = Hxp/factor
    elif new_scale in HENRY_SCALES_HCP_MOLALITY:
        factor = 1000.0/rhom
        conv_val = Hxp/factor
    elif new_scale in HENRY_SCALES_HCC:
        factor = (atm/(R*298.15*rhom))
        conv_val = Hxp/factor
    elif new_scale in HENRY_SCALES_HBP_SI:
        rho = 1e-3*rhom*MW  # Vm_to_rho(1.0/rhom, MW)
        factor = rho*atm/rhom
        conv_val = Hxp/factor
    elif new_scale in HENRY_SCALES_HBP:
        rho = 1e-3*rhom*MW  # Vm_to_rho(1.0/rhom, MW)
        factor = rho/rhom
        conv_val = Hxp/factor
    elif new_scale in HENRY_SCALES_HXP:
        conv_val = Hxp
    elif new_scale in HENRY_SCALES_BUNSEN:
        factor = atm/(R*273.15*rhom)
        conv_val = Hxp/factor
    elif new_scale in HENRY_SCALES_KHPX:
        conv_val = 1.0/Hxp
    elif new_scale in HENRY_SCALES_KHPC_SI:
        factor = atm/rhom
        conv_val = factor/Hxp
    elif new_scale in HENRY_SCALES_KHPC:
        factor = 1.0/rhom
        conv_val = factor/Hxp
    elif new_scale in HENRY_SCALES_KHCC:
        factor = atm/(R*298.15*rhom)
        conv_val = factor/(Hxp)
    elif new_scale in HENRY_SCALES_SI:
        conv_val = atm/Hxp
    else:
        raise ValueError("Not recognized input scale: %s" %new_scale)

    return conv_val
        

def Henry_pressure(T, A, B=0.0, C=0.0, D=0.0, E=0.0, F=0.0):
    r'''Calculates Henry's law constant as a function of temperature according
    to the SI units of `Pa` and using a common temperature dependence as used
    in many process simulation applications.
    
    Only the `A` parameter is required - which has no temperature dependence
    when used by itself.
    As the model is exponential, a sufficiently high temperature may cause an
    OverflowError.
    A negative temperature (or just low, if fit poorly) may cause a math domain
    error.

    .. math::
        H_{12} = \exp\left(A_{12} + \frac{B_{12}}{T} + C_{12}\log(T) + D_{12}T
         + \frac{E_{12}}{T^2} \right)

    Parameters
    ----------
    T : float
        Temperature, [K]
    A-F : float
        Parameter for the equation; chemical and property specific [-]

    Returns
    -------
    H12 : float
        Henry's constant [Pa]

    Notes
    -----
    Add 11.51292 to the `A` constant if it is said to provide units of `bar`,
    so that it provides units of `Pa` instead.
    
    The `F` parameter is not often included in models. It is rare to fit
    all parameters.

    Examples
    --------
    Random test example.

    >>> Henry_pressure(300.0, A=15.0, B=300.0, C=.04, D=1e-3, E=1e2, F=1e-5)
    37105004.47898146

    References
    ----------
    .. [1] Gmehling, Jurgen. Chemical Thermodynamics: For Process Simulation.
       Weinheim, Germany: Wiley-VCH, 2012.
    '''
    return exp(A + B/T + C*log(T) + D*T + E/T**2 + F*T**2)


def Henry_pressure_mixture(Hs, weights=None, zs=None):
    r'''Mixing rule for Henry's law components. Applies a logarithmic average
    to all solvent components and mole fractions. Optionally, weight factors
    can be provided instead of using mole fractions - only specify one of them.
    
    A common weight factor is using volume fractions of powers of them, or 
    using critical volumes.

    Parameters
    ----------
    Hs : list[float or None]
        Henry's law constant between each gas and the solvent (None for other
        solvents of gases without parameters available), [Pa]
    weights : list[float], optional
        Weight factors, [-]
    zs : list[float]
        Mole fractions of all species in phase, [-]

    Returns
    -------
    H : value
        Henry's law constant for the gas in the liquid phase, [-]

    Notes
    -----
    The default weight factor formulation is from [1]_.

    Examples
    --------
    >>> Henry_pressure_mixture([1072330.36341, 744479.751106, None], zs=[.48, .48, .04])
    893492.1611602883

    References
    ----------
    .. [1] Gmehling, Jurgen. Chemical Thermodynamics: For Process Simulation.
       Weinheim, Germany: Wiley-VCH, 2012.
    '''
    cmps = range(len(Hs))
    if weights is None:
        # Default parameters - when weight specified only weight by that
        z_solvent = sum(zs[i] for i in cmps if Hs[i] is not None)
        weights = [zs[i]/z_solvent for i in cmps]
    num = 0.0
    for i in cmps:
        if Hs[i] is not None:
            num += weights[i]*log(Hs[i])
    H = exp(num)
    return H
