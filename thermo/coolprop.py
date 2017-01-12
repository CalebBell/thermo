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

__all__ = ['has_CoolProp', 'coolprop_dict', 'CP_fluid', 'coolprop_fluids', 
'CoolProp_T_dependent_property', 'PropsSI', 'PhaseSI', 'CP', 'AbstractState']

try:
    from CoolProp.CoolProp import PropsSI, PhaseSI
    import CoolProp.CoolProp as CP
    from CoolProp import AbstractState
    has_CoolProp = True
except ImportError:  # pragma: no cover
    has_CoolProp = False
    PropsSI, PhaseSI, CP, AbstractState = [None, None, None, None]
#has_CoolProp = False # For testing

# All of these can be inputs to the PropsSI function!
coolprop_dict = ['100-41-4', '10024-97-2', '102687-65-0', '106-42-3',
'106-97-8', '106-98-9', '107-46-0', '107-51-7', '107-52-8', '107-83-5',
'108-38-3', '108-88-3', '109-66-0', '110-54-3', '110-82-7', '111-65-9',
'111-84-2', '112-39-0', '112-40-3', '112-61-8', '112-62-9', '112-63-0',
'1120-21-4', '115-07-1', '115-10-6', '115-11-7', '115-25-3', '124-18-5',
'124-38-9', '1333-74-0', '141-62-8', '141-63-9', '142-82-5', '1717-00-6',
'2551-62-4', '2837-89-0', '287-92-3', '29118-24-9', '29118-25-0', '301-00-8',
'306-83-2', '353-36-6', '354-33-6', '406-58-6', '420-46-2', '421-14-7',
'431-63-0', '431-89-0', '460-73-1', '463-58-1', '463-82-1', '540-97-6',
'556-67-2', '590-18-1', '593-53-3', '616-38-6', '624-64-6', '630-08-0',
'64-17-5', '67-56-1', '67-64-1', '690-39-1', '71-43-2', '74-82-8', '74-84-0',
'74-85-1', '74-87-3', '74-98-6', '74-99-7', '7439-90-9', '7440-01-9',
'7440-37-1', '7440-59-7', '7440-63-3', '7446-09-5', '75-10-5', '75-19-4',
'75-28-5', '75-37-6', '75-43-4', '75-45-6', '75-46-7', '75-68-3', '75-69-4',
'75-71-8', '75-72-9', '75-73-0', '754-12-1', '756-13-8', '76-13-1', '76-14-2',
'76-15-3', '76-16-4', '76-19-7', '7664-41-7', '7727-37-9', '7732-18-5',
'7782-39-0', '7782-41-4', '7782-44-7', '7783-06-4', '7789-20-0', '78-78-4',
'811-97-2', '95-47-6']


class CP_fluid(object):
    # Basic object to store constants for a coolprop fluid, much faster than
    # calling coolprop to retrieve the data when needed
    __slots__ = ['Tmin', 'Tmax', 'Pmax', 'has_melting_line', 'Tc', 'Pc', 'Tt',
                 'omega', 'HEOS']
    
    def __deepcopy__(self, memo):
        # AbstractState("HEOS", CAS) has no deep copy;
        # fortunately, none is needed, so we can just return the existing copy
        return self

    def __init__(self, Tmin, Tmax, Pmax, has_melting_line, Tc, Pc, Tt, omega,
                 HEOS):
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.Pmax = Pmax
        self.has_melting_line = has_melting_line
        self.Tc = Tc
        self.Pc = Pc
        self.Tt = Tt
        self.omega = omega
        self.HEOS = HEOS


# Store the propoerties in a dict of CP_fluid instances
coolprop_fluids = {}
if has_CoolProp:
    for CASRN in coolprop_dict:
        HEOS = AbstractState("HEOS", CASRN)
        coolprop_fluids[CASRN] = CP_fluid(Tmin=HEOS.Tmin(), Tmax=HEOS.Tmax(), Pmax=HEOS.pmax(),
                       has_melting_line=HEOS.has_melting_line(), Tc=HEOS.T_critical(), Pc=HEOS.p_critical(),
                       Tt=HEOS.Ttriple(), omega=HEOS.acentric_factor(), HEOS=HEOS)


def CoolProp_T_dependent_property(T, CASRN, prop, phase):
    r'''Calculates a property of a chemical in either the liquid or gas phase
    as a function of temperature only. This means that the property is
    either at 1 atm or along the saturation curve.

    Parameters
    ----------
        T : float
            Temperature of the fluid [K]
        CASRN : str
            CAS number of the fluid
        prop : str
            CoolProp string shortcut for desired property
        phase : str
            Either 'l' or 'g' for liquid or gas properties respectively

    Returns
    -------
        prop : float
            Desired chemical property, [units]

    Notes
    -----
    For liquids above their boiling point, the liquid property is found on the
    saturation line (at higher pressures). Under their boiling point, the
    property is calculated at 1 atm.

    No liquid calculations are permitted above the critical temperature.

    For gases under the chemical's boiling point, the gas property is found
    on the saturation line (at sub-atmospheric pressures). Above the boiling
    point, the property is calculated at 1 atm.

    An exception is raised if the desired CAS is not supported, or if CoolProp
    is not available.

    The list of strings acceptable as an input for property types is:
    http://www.coolprop.org/coolprop/HighLevelAPI.html#table-of-string-inputs-to-propssi-function

    Examples
    --------
    Water at STP according to IAPWS-95

    >>> CoolProp_T_dependent_property(298.15, '7732-18-5', 'D', 'l')
    997.047636760347

    References
    ----------
    .. [1] Bell, Ian H., Jorrit Wronski, Sylvain Quoilin, and Vincent Lemort.
       "Pure and Pseudo-Pure Fluid Thermophysical Property Evaluation and the
       Open-Source Thermophysical Property Library CoolProp." Industrial &
       Engineering Chemistry Research 53, no. 6 (February 12, 2014):
       2498-2508. doi:10.1021/ie4033999. http://www.coolprop.org/
    '''
    if not has_CoolProp:  # pragma: no cover
        raise Exception('CoolProp library is not installed')
    if CASRN not in coolprop_dict:
        raise Exception('CASRN not in list of supported fluids')
    Tc = coolprop_fluids[CASRN].Tc
    T = float(T) # Do not allow custom objects here
    if phase == 'l':
        if T > Tc:
            raise Exception('For liquid properties, must be under the critical temperature.')
        if PhaseSI('T', T, 'P', 101325, CASRN) in [u'liquid', u'supercritical_liquid']:
            return PropsSI(prop, 'T', T, 'P', 101325, CASRN)
        else:
            return PropsSI(prop, 'T', T, 'Q', 0, CASRN)
    elif phase == 'g':
        if PhaseSI('T', T, 'P', 101325, CASRN) == 'gas':
            return PropsSI(prop, 'T', T, 'P', 101325, CASRN)
        else:
            if T < Tc:
                return PropsSI(prop, 'T', T, 'Q', 1, CASRN)
            else:
                # catch supercritical_gas and friends
                return PropsSI(prop, 'T', T, 'P', 101325, CASRN)
    else:
        raise Exception('Error in CoolProp property function')
