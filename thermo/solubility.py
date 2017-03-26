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

__all__ = ['solubility_parameter_methods', 'solubility_parameter', 
           'solubility_eutectic', 'Tm_depression_eutectic']
           
import os
from thermo.utils import log, exp
from thermo.utils import R

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
