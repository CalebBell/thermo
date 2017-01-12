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

__all__ = ['omega', 'LK_omega', 'omega_mixture','StielPolar']
__all__.extend(['omega_methods', 'omega_mixture_methods', 'Stiel_polar_methods'])

import numpy as np
import pandas as pd
from thermo.utils import log, log10
from thermo.utils import mixing_simple, none_and_length_check
from thermo.critical import Tc, Pc
from thermo.critical import _crit_PSRKR4, _crit_PassutDanner, _crit_Yaws
from thermo.phase_change import Tb
from thermo.vapor_pressure import VaporPressure


omega_methods = ['PSRK', 'PD', 'YAWS', 'LK', 'DEFINITION']


def omega(CASRN, AvailableMethods=False, Method=None, IgnoreMethods=['LK', 'DEFINITION']):
    r'''This function handles the retrieval of a chemical's acentric factor,
    `omega`, or its calculation from correlations or directly through the
    definition of acentric factor if possible. Requires a known boiling point,
    critical temperature and pressure for use of the correlations. Requires
    accurate vapor pressure data for direct calculation.

    Will automatically select a method to use if no Method is provided;
    returns None if the data is not available and cannot be calculated.

    .. math::
        \omega \equiv -\log_{10}\left[\lim_{T/T_c=0.7}(P^{sat}/P_c)\right]-1.0

    Examples
    --------
    >>> omega(CASRN='64-17-5')
    0.635

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    omega : float
        Acentric factor of compound
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain omega with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'PSRK', 'PD', 'YAWS', 
        'LK', and 'DEFINITION'. All valid values are also held in the list
        omega_methods.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        omega for the desired chemical, and will return methods instead of
        omega
    IgnoreMethods : list, optional
        A list of methods to ignore in obtaining the full list of methods,
        useful for for performance reasons and ignoring inaccurate methods

    Notes
    -----
    A total of five sources are available for this function. They are:

        * 'PSRK', a compillation of experimental and estimated data published 
          in the Appendix of [15]_, the fourth revision of the PSRK model.
        * 'PD', an older compillation of
          data published in (Passut & Danner, 1973) [16]_.
        * 'YAWS', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [17]_.
        * 'LK', a estimation method for hydrocarbons.
        * 'DEFINITION', based on the definition of omega as
          presented in [1]_, using vapor pressure data.

    References
    ----------
    .. [1] Pitzer, K. S., D. Z. Lippmann, R. F. Curl, C. M. Huggins, and
       D. E. Petersen: The Volumetric and Thermodynamic Properties of Fluids.
       II. Compressibility Factor, Vapor Pressure and Entropy of Vaporization.
       J. Am. Chem. Soc., 77: 3433 (1955).
    .. [2] Horstmann, Sven, Anna Jabłoniec, Jörg Krafczyk, Kai Fischer, and
       Jürgen Gmehling. "PSRK Group Contribution Equation of State:
       Comprehensive Revision and Extension IV, Including Critical Constants
       and Α-Function Parameters for 1000 Components." Fluid Phase Equilibria
       227, no. 2 (January 25, 2005): 157-64. doi:10.1016/j.fluid.2004.11.002.
    .. [3] Passut, Charles A., and Ronald P. Danner. "Acentric Factor. A
       Valuable Correlating Parameter for the Properties of Hydrocarbons."
       Industrial & Engineering Chemistry Process Design and Development 12,
       no. 3 (July 1, 1973): 365-68. doi:10.1021/i260047a026.
    .. [4] Yaws, Carl L. Thermophysical Properties of Chemicals and
       Hydrocarbons, Second Edition. Amsterdam Boston: Gulf Professional
       Publishing, 2014.
    '''
    def list_methods():
        methods = []
        if CASRN in _crit_PSRKR4.index and not np.isnan(_crit_PSRKR4.at[CASRN, 'omega']):
            methods.append('PSRK')
        if CASRN in _crit_PassutDanner.index and not np.isnan(_crit_PassutDanner.at[CASRN, 'omega']):
            methods.append('PD')
        if CASRN in _crit_Yaws.index and not np.isnan(_crit_Yaws.at[CASRN, 'omega']):
            methods.append('YAWS')
        Tcrit, Pcrit = Tc(CASRN), Pc(CASRN)
        if Tcrit and Pcrit:
            if Tb(CASRN):
                methods.append('LK')
            if VaporPressure(CASRN=CASRN).T_dependent_property(Tcrit*0.7):
                methods.append('DEFINITION')  # TODO: better integration
        if IgnoreMethods:
            for Method in IgnoreMethods:
                if Method in methods:
                    methods.remove(Method)
        methods.append('NONE')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == 'PSRK':
        _omega = float(_crit_PSRKR4.at[CASRN, 'omega'])
    elif Method == 'PD':
        _omega = float(_crit_PassutDanner.at[CASRN, 'omega'])
    elif Method == 'YAWS':
        _omega = float(_crit_Yaws.at[CASRN, 'omega'])
    elif Method == 'LK':
        _omega = LK_omega(Tb(CASRN), Tc(CASRN), Pc(CASRN))
    elif Method == 'DEFINITION':
        P = VaporPressure(CASRN=CASRN).T_dependent_property(Tc(CASRN)*0.7)
        _omega = -log10(P/Pc(CASRN)) - 1.0
    elif Method == 'NONE':
        _omega = None
    else:
        raise Exception('Failure in in function')
    return _omega


def LK_omega(Tb, Tc, Pc):
    r'''Estimates the acentric factor of a fluid using a correlation in [1]_.

    .. math::
        \omega = \frac{\ln P_{br}^{sat} - 5.92714 + 6.09648/T_{br} + 1.28862
        \ln T_{br} -0.169347T_{br}^6}
        {15.2518 - 15.6875/T_{br} - 13.4721 \ln T_{br} + 0.43577 T_{br}^6}

    Parameters
    ----------
    Tb : float
        Boiling temperature of the fluid [K]
    Tc : float
        Critical temperature of the fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]

    Returns
    -------
    omega : float
        Acentric factor of the fluid [-]

    Notes
    -----
    Internal units are atmosphere and Kelvin.
    Example value from Reid (1987). Using ASPEN V8.4, LK method gives 0.325595.

    Examples
    --------
    Isopropylbenzene

    >>> LK_omega(425.6, 631.1, 32.1E5)
    0.32544249926397856

    References
    ----------
    .. [1] Lee, Byung Ik, and Michael G. Kesler. "A Generalized Thermodynamic
       Correlation Based on Three-Parameter Corresponding States." AIChE Journal
       21, no. 3 (1975): 510-527. doi:10.1002/aic.690210313.
    '''
    T_br = Tb/Tc
    omega = (log(101325.0/Pc) - 5.92714 + 6.09648/T_br + 1.28862*log(T_br) -
             0.169347*T_br**6)/(15.2518 - 15.6875/T_br - 13.4721*log(T_br) +
             0.43577*T_br**6)
    return omega


omega_mixture_methods = ['SIMPLE', 'NONE']


def omega_mixture(omegas, zs, CASRNs=None, Method=None,
                  AvailableMethods=False):
    r'''This function handles the calculation of a mixture's acentric factor.
    Calculation is based on the omegas provided for each pure component. Will
    automatically select a method to use if no Method is provided;
    returns None if insufficient data is available.

    Examples
    --------
    >>> omega_mixture([0.025, 0.12], [0.3, 0.7])
    0.0915

    Parameters
    ----------
    omegas : array-like
        acentric factors of each component, [-]
    zs : array-like
        mole fractions of each component, [-]
    CASRNs: list of strings
        CASRNs, not currently used [-]

    Returns
    -------
    omega : float
        acentric factor of the mixture, [-]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain omega with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Only 'SIMPLE' is accepted so far.
        All valid values are also held in the list omega_mixture_methods.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        omega for the desired chemical, and will return methods instead of
        omega

    Notes
    -----
    The only data used in the methods implemented to date are mole fractions
    and pure-component omegas. An alternate definition could be based on
    the dew point or bubble point of a multicomponent mixture, but this has
    not been done to date.

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    def list_methods():
        methods = []
        if none_and_length_check([zs, omegas]):
            methods.append('SIMPLE')
        methods.append('NONE')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == 'SIMPLE':
        _omega = mixing_simple(zs, omegas)
    elif Method == 'NONE':
        _omega = None
    else:
        raise Exception('Failure in in function')
    return _omega


Stiel_polar_methods = ['DEFINITION']


def StielPolar(Tc=None, Pc=None, omega=None, CASRN='', Method=None,
               AvailableMethods=False):
    r'''This function handles the calculation of a chemical's Stiel Polar
    factor, directly through the definition of Stiel-polar factor if possible.
    Requires Tc, Pc, acentric factor, and a vapor pressure datum at Tr=0.6.

    Will automatically select a method to use if no Method is provided;
    returns None if the data is not available and cannot be calculated.

    .. math::
        x = \log P_r|_{T_r=0.6} + 1.70 \omega + 1.552

    Parameters
    ----------
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]
    omega : float
        Acentric factor of the fluid [-]
    CASRN : string
        CASRN [-]

    Returns
    -------
    factor : float
        Stiel polar factor of compound
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Stiel polar factor with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Only 'DEFINITION' is accepted so far.
        All valid values are also held in the list Stiel_polar_methods.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Stiel-polar factor for the desired chemical, and will return methods
        instead of stiel-polar factor

    Notes
    -----
    Only one source is available for this function. It is:

        * 'DEFINITION', based on the definition of
          Stiel Polar Factor presented in [1]_, using vapor pressure data.

    A few points have also been published in [2]_, which may be used for
    comparison. Currently this is only used for a surface tension correlation.

    Examples
    --------
    >>> StielPolar(647.3, 22048321.0, 0.344, CASRN='7732-18-5')
    0.024581140348734376

    References
    ----------
    .. [1] Halm, Roland L., and Leonard I. Stiel. "A Fourth Parameter for the
       Vapor Pressure and Entropy of Vaporization of Polar Fluids." AIChE
       Journal 13, no. 2 (1967): 351-355. doi:10.1002/aic.690130228.
    .. [2] D, Kukoljac Miloš, and Grozdanić Dušan K. "New Values of the
       Polarity Factor." Journal of the Serbian Chemical Society 65, no. 12
       (January 1, 2000). http://www.shd.org.rs/JSCS/Vol65/No12-Pdf/JSCS12-07.pdf
    '''
    def list_methods():
        methods = []
        if Tc and Pc and omega:
            methods.append('DEFINITION')
        methods.append('NONE')
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    if Method == 'DEFINITION':
        P = VaporPressure(CASRN=CASRN).T_dependent_property(Tc*0.6)
        if not P:
            factor = None
        else:
            Pr = P/Pc
            factor = log10(Pr) + 1.70*omega + 1.552
    elif Method == 'NONE':
        factor = None
    else:
        raise Exception('Failure in in function')
    return factor


