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
SOFTWARE.
'''


__all__ = ['has_CoolProp', 'coolprop_dict', 'CP_fluid', 'coolprop_fluids',
'CoolProp_T_dependent_property', 'CoolProp_failing_PT_flashes',
'PropsSI', 'PhaseSI','HAPropsSI', 'AbstractState',
'Helmholtz_A0', 'Helmholtz_dA0_dtau', 'Helmholtz_d2A0_dtau2',
'Helmholtz_d3A0_dtau3',
'CoolProp_json_alpha0_to_kwargs',
'Helmholtz_A0_data',
'Cp_ideal_gas_Helmholtz', 'H_ideal_gas_Helmholtz',
'S_ideal_gas_Helmholtz']
import os
from math import exp, log

from chemicals.utils import mark_numba_incompatible
from fluids.numerics import numpy as np

from thermo.base import data_dir

#try:
#    import CoolProp
#    has_CoolProp = True
##    CPiP_min = CP.iP_min
#except:  # pragma: no cover
#    # Don't just except
#    has_CoolProp = False
#has_CoolProp = False # For testing

CPiP_min = 17
_has_CoolProp = None
@mark_numba_incompatible
def has_CoolProp():
    global _has_CoolProp
    if _has_CoolProp is None:
        try:
            load_coolprop_fluids()
            _has_CoolProp = True
        except:
            _has_CoolProp = False
    return _has_CoolProp

_PropsSI = None
@mark_numba_incompatible
def PropsSI(*args, **kwargs):
    global _PropsSI
    if _PropsSI is None:
        from CoolProp.CoolProp import PropsSI as _PropsSI
    return _PropsSI(*args, **kwargs)

_HAPropsSI = None
@mark_numba_incompatible
def HAPropsSI(*args, **kwargs):
    global _HAPropsSI
    if _HAPropsSI is None:
        from CoolProp.CoolProp import HAPropsSI as _HAPropsSI
    return _HAPropsSI(*args, **kwargs)

_PhaseSI = None
@mark_numba_incompatible
def PhaseSI(*args, **kwargs):
    global _PhaseSI
    if _PhaseSI is None:
        from CoolProp.CoolProp import PhaseSI as _PhaseSI
    return _PhaseSI(*args, **kwargs)

_AbstractState = None
@mark_numba_incompatible
def AbstractState(*args, **kwargs):
    global _AbstractState
    if _AbstractState is None:
        from CoolProp.CoolProp import AbstractState as _AbstractState
    return _AbstractState(*args, **kwargs)



# Load the constants, store




# CoolProp.FluidsList() indicates some new fluids have been added
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


CoolProp_has_mu_CASs = {'74-82-8', '109-66-0', '67-56-1', '115-07-1', '76-16-4', '75-72-9', '811-97-2', '75-73-0', '1717-00-6', '75-68-3', '76-19-7', '431-89-0', '431-63-0', '690-39-1', '115-25-3', '75-69-4', '75-71-8', '420-46-2', '306-83-2', '102687-65-0', '754-12-1', '29118-24-9', '2837-89-0', '75-37-6', '75-45-6', '460-73-1', '75-10-5', '354-33-6', '75-46-7', 'R404A.PPF', 'R407C.PPF', 'R410A.PPF', 'R507A.PPF', '2551-62-4', '108-88-3', '7732-18-5', '108-38-3', '106-97-8', '124-18-5', '111-84-2', '111-65-9', '112-40-3', '142-82-5', '110-54-3', '74-98-6', '95-47-6', '106-42-3', 'AIR.PPF', '7440-37-1', '7727-37-9', '7782-44-7', '7664-41-7', '71-43-2', '124-38-9', '110-82-7', '287-92-3', '78-78-4', '115-10-6', '74-84-0', '64-17-5', '7789-20-0', '7440-59-7', '1333-74-0', '7783-06-4', '75-28-5'}
CoolProp_has_k_CASs = {'74-82-8', '67-56-1', '115-07-1', '76-16-4', '75-72-9', '75-73-0', '1717-00-6', '75-68-3', '76-19-7', '431-89-0', '431-63-0', '690-39-1', '115-25-3', '2837-89-0', '460-73-1', '75-10-5', '811-97-2', '75-69-4', '75-71-8', '420-46-2', '75-45-6', '306-83-2', '754-12-1', '29118-24-9', '354-33-6', '75-37-6', '75-46-7', 'R404A.PPF', 'R407C.PPF', 'R410A.PPF', 'R507A.PPF', '2551-62-4', '108-88-3', '7732-18-5', '106-97-8', '124-18-5', '111-84-2', '111-65-9', '112-40-3', '142-82-5', '110-54-3', '74-98-6', 'AIR.PPF', '7440-37-1', '7727-37-9', '7782-44-7', '7664-41-7', '71-43-2', '124-38-9', '109-66-0', '287-92-3', '78-78-4', '74-84-0', '64-17-5', '100-41-4', '108-38-3', '95-47-6', '106-42-3', '7789-20-0', '7440-59-7', '1333-74-0p', '1333-74-0', '75-28-5'}
CoolProp_has_sigma_CASs = {'107-46-0', '74-87-3', '75-69-4', '616-38-6', '75-19-4', '29118-24-9', '754-12-1', '74-98-6', '463-82-1', '75-71-8', '353-36-6', '107-51-7', '7732-18-5', '1333-74-0', '115-10-6', '7446-09-5', '112-61-8', '124-38-9', '630-08-0', '7782-39-0', '76-13-1', '406-58-6', '67-56-1', '76-19-7', '67-64-1', '287-92-3', '64-17-5', '141-63-9', '102687-65-0', '107-52-8', '7440-59-7', '2837-89-0', '556-67-2', '100-41-4', '7727-37-9', '460-73-1', '7789-20-0', '110-82-7', '7439-90-9', '354-33-6', '7440-01-9', '75-68-3', '7782-41-4', '590-18-1', '106-97-8', '141-62-8', '7440-63-3', '74-99-7', '431-89-0', '7664-41-7', '108-88-3', '78-78-4', '115-25-3', '115-07-1', '75-10-5', '624-64-6', '111-84-2', '690-39-1', '75-28-5', '2551-62-4', '112-62-9', '74-82-8', '107-83-5', '420-46-2', '7440-37-1', '110-54-3', '112-40-3', '95-47-6', '74-84-0', '115-11-7', '75-73-0', '75-46-7', '76-16-4', '593-53-3', '76-14-2', '1120-21-4', '106-42-3', '75-72-9', '109-66-0', '306-83-2', '111-65-9', '108-38-3', '75-37-6', '71-43-2', '112-39-0', '7783-06-4', '106-98-9', '431-63-0', '29118-25-0', '74-85-1', '142-82-5', '7782-44-7', '811-97-2', '1717-00-6', '463-58-1', '75-43-4', '124-18-5', '75-45-6', '540-97-6', '112-63-0', '10024-97-2'}
CoolProp_k_failing_CASs = {'100-41-4', '2837-89-0', '460-73-1', '75-10-5', '75-45-6'}

CoolProp_failing_PT_flashes = {'115-07-1', '115-25-3', '1717-00-6', '420-46-2',
                                '431-63-0', '431-89-0', '690-39-1', '75-68-3', '75-69-4', '75-71-8', '75-72-9', '75-73-0', '76-19-7',
                                '110-82-7', '7782-44-7'}

CoolProp_Tmin_overrides = {
    '106-97-8': 135,
    '106-98-9': 87.9,
    '109-66-0': 144,
    '110-82-7': 279.52,
    '67-56-1': 175.7,
    '74-82-8': 90.8,
    '74-84-0': 90.4,
    '74-85-1': 104.1,
    '75-28-5': 114,
    '7727-37-9': 63.2,
    '100-41-4': 263.5,
}

CoolProp_Tmax_overrides = {
    '107-51-7': 563,
}

class CP_fluid:
    # Basic object to store constants for a coolprop fluid, much faster than
    # calling coolprop to retrieve the data when needed
    __slots__ = ['Tmin', 'Tmax', 'Pmax', 'has_melting_line', 'Tc', 'Pc', 'Tt',
                 'omega', 'HEOS', 'CAS', 'Vc', 'Pt', 'Tb']
    @property
    def has_k(self):
        return self.CAS in CoolProp_has_k_CASs and self.CAS not in CoolProp_k_failing_CASs

    @property
    def has_mu(self):
        return self.CAS in CoolProp_has_mu_CASs

    @property
    def has_sigma(self):
        return self.CAS in CoolProp_has_sigma_CASs


    def as_json(self):
        return {k: getattr(self, k) for k in self.__slots__}
    def __deepcopy__(self, memo):
        # AbstractState("HEOS", CAS) has no deep copy;
        # fortunately, none is needed, so we can just return the existing copy
        return self

    def __init__(self, Tmin, Tmax, Pmax, has_melting_line, Tc, Pc, Tt, omega,
                 HEOS, CAS, Vc, Pt, Tb):
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.Pmax = Pmax
        self.has_melting_line = has_melting_line
        self.Tc = Tc
        self.Pc = Pc
        self.Tt = Tt
        self.omega = omega
        self.HEOS = HEOS
        self.CAS = CAS
        self.Vc = Vc
        self.Pt = Pt
        self.Tb = Tb


# Store the propoerties in a dict of CP_fluid instances
coolprop_fluids = {}
# if has_CoolProp:
#     #for fluid in CP.FluidsList():
#     #    CASRN = CP.get_fluid_param_string(fluid, 'CAS')
#     for CASRN in coolprop_dict:
#         # TODO find other way of getting the data faster - there is no way
#         # TODO use appdirs, store this data as a cache
#         HEOS = AbstractState("HEOS", CASRN)
#         coolprop_fluids[CASRN] = CP_fluid(Tmin=HEOS.Tmin(), Tmax=HEOS.Tmax(), Pmax=HEOS.pmax(),
#                        has_melting_line=HEOS.has_melting_line(), Tc=HEOS.T_critical(), Pc=HEOS.p_critical(),
#                        Tt=HEOS.Ttriple(), omega=HEOS.acentric_factor(), HEOS=None)

@mark_numba_incompatible
def store_coolprop_fluids():
    import json

    import CoolProp
    for CASRN in coolprop_dict:
        HEOS = AbstractState("HEOS", CASRN)
        try:
          Tb = PropsSI("T", "Q", 0, "P", 101325, CASRN)
        except:
          Tb = None
        coolprop_fluids[CASRN] = CP_fluid(Tmin=HEOS.Tmin(), Tmax=HEOS.Tmax(), Pmax=HEOS.pmax(),
                       has_melting_line=HEOS.has_melting_line(), Tc=HEOS.T_critical(), Pc=HEOS.p_critical(),
                       Tt=HEOS.Ttriple(), omega=HEOS.acentric_factor(), HEOS=None, CAS=CASRN,
                       Vc=1/HEOS.rhomolar_critical(), Pt=PropsSI('PTRIPLE', CASRN),
                       Tb=Tb)


    data = {CASRN: coolprop_fluids[CASRN].as_json() for CASRN in coolprop_dict}
    ver = CoolProp.__version__
    file = open(os.path.join(data_dir, f'CoolPropFluids{ver}.json'), 'w')
    json.dump(data, file)
    file.close()

@mark_numba_incompatible
def load_coolprop_fluids(depth=0):
    import json

    import CoolProp
    ver = CoolProp.__version__
    pth = os.path.join(data_dir, f'CoolPropFluids{ver}.json')
    try:
        file = open(pth)
    except:
        store_coolprop_fluids()
        file = open(pth)
    try:
      data = json.load(file)
      for CASRN in coolprop_dict:
          d = data[CASRN]
          coolprop_fluids[CASRN] = CP_fluid(Tmin=d['Tmin'], Tmax=d['Tmax'], Pmax=d['Pmax'],
                        has_melting_line=d['has_melting_line'], Tc=d['Tc'], Pc=d['Pc'],
                        Tt=d['Tt'], omega=d['omega'], HEOS=None, CAS=CASRN, Vc=d['Vc'], Pt=d['Pt'], Tb=d['Tb'])
    except Exception as e:
      if depth == 0:
        store_coolprop_fluids()
        load_coolprop_fluids(depth=1)
      else:
        raise e

@mark_numba_incompatible
def CoolProp_T_dependent_property(T, CASRN, prop, phase, Tc=None):
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
    Tc : float or None
      Optionally, the critical temperature can be provided [K]

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

    >>> CoolProp_T_dependent_property(298.15, '7732-18-5', 'D', 'l') # doctest:+SKIP
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
    if CASRN not in coolprop_dict and 'REFPROP' not in CASRN:
        raise Exception('CASRN not in list of supported fluids')
    if prop in ('CP0MOLAR', 'Cp0molar'):
      try:
        # Some cases due to melting point need a high pressure
        return PropsSI('Cp0molar', 'T', T,'P', 10132500.0, CASRN)
      except:
          # And some cases don't converge at high P
          return PropsSI('Cp0molar', 'T', T,'P', 101325.0, CASRN)

    if Tc is None:
        Tc = coolprop_fluids[CASRN].Tc
    T = float(T) # Do not allow custom objects here
    if phase == 'l':
        if T > Tc:
            raise Exception('For liquid properties, must be under the critical temperature.')
        if PhaseSI('T', T, 'P', 101325, CASRN) in ['liquid', 'supercritical_liquid']:
          # If under the vapor pressure curve e.g. water at STP, use 1 atm
          # at the normal boiling point this switches to the saturation curve
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


def CoolProp_json_alpha0_to_kwargs(json_data, as_np=False):
    '''

    Parameters
    ----------
    json_data : TYPE
        loaded CoolProp .json file
    as_np : TYPE, optional
        Whether or not to have the vector data as numpy arrays.

    Returns
    -------
    kwargs : dict
        The parameters accepted by the residual functions.

    '''
    alpha0 = json_data['EOS'][0]['alpha0']
    Tc_A0 = json_data['EOS'][0]['STATES']['reducing']['T']
    kwargs = {}
    for d in alpha0:
        if d['type'] == 'IdealGasHelmholtzLead':
            kwargs['IdealGasHelmholtzLead_a1'] = d['a1']
            kwargs['IdealGasHelmholtzLead_a2'] = d['a2']
        elif d['type'] == 'IdealGasHelmholtzLogTau':
            kwargs['IdealGasHelmholtzLogTau_a'] = d['a']
        elif d['type'] == 'IdealGasHelmholtzPlanckEinstein':
            kwargs['IdealGasHelmholtzPlanckEinstein_ns'] = d['n']
            kwargs['IdealGasHelmholtzPlanckEinstein_ts'] = d['t']
        elif d['type'] == 'IdealGasHelmholtzPlanckEinsteinGeneralized':
            n, t, c, d = d['n'], d['t'], d['c'], d['d']
            for arg, name in zip([n, t, c, d], ['IdealGasHelmholtzPlanckEinsteinGeneralized_ns',
                                                    'IdealGasHelmholtzPlanckEinsteinGeneralized_ts',
                                                    'IdealGasHelmholtzPlanckEinsteinGeneralized_cs',
                                                    'IdealGasHelmholtzPlanckEinsteinGeneralized_ds']):
                if name in kwargs:
                    kwargs[name].extend(arg)
                else:
                    kwargs[name] = arg
        elif d['type'] == 'IdealGasHelmholtzPower':
            kwargs['IdealGasHelmholtzPower_ns'] = d['n']
            kwargs['IdealGasHelmholtzPower_ts'] = d['t']
        elif d['type'] == 'IdealGasHelmholtzPlanckEinsteinFunctionT':
            n = d['n']
            t = d['v']
            theta = [-v/Tc_A0 for v in t]
            c = [1]*len(t)
            d = [-1]*len(t)
            kwargs['IdealGasHelmholtzPlanckEinsteinGeneralized_ts'] = theta
            kwargs['IdealGasHelmholtzPlanckEinsteinGeneralized_cs'] = c
            kwargs['IdealGasHelmholtzPlanckEinsteinGeneralized_ds'] = d

            for arg, name in zip([n, theta, c, d], ['IdealGasHelmholtzPlanckEinsteinGeneralized_ns',
                                                    'IdealGasHelmholtzPlanckEinsteinGeneralized_ts',
                                                    'IdealGasHelmholtzPlanckEinsteinGeneralized_cs',
                                                    'IdealGasHelmholtzPlanckEinsteinGeneralized_ds']):
                if name in kwargs:
                    kwargs[name].extend(arg)
                else:
                    kwargs[name] = arg

        elif d['type'] == 'IdealGasHelmholtzEnthalpyEntropyOffset':
            # Not relevant
            continue
        else:
            raise ValueError("Unrecognized alpha0 type {}".format(d['type']))

    if as_np:
        for k, v in kwargs.items():
            if type(v) is list:
                kwargs[k] = np.array(v)

    return kwargs


def Helmholtz_A0(tau, delta,
            IdealGasHelmholtzLead_a1=0.0, IdealGasHelmholtzLead_a2=0.0,
            IdealGasHelmholtzLogTau_a=0.0,
           IdealGasHelmholtzPlanckEinstein_ns=None,
           IdealGasHelmholtzPlanckEinstein_ts=None,
           IdealGasHelmholtzPower_ns=None,
           IdealGasHelmholtzPower_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ns=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_cs=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ds=None):
    '''Compute the dimensionless ideal gas Helmholtz energy for a  Helmholtz
    equation of state.

    These parameters can be found in the literature, or extracted from a
    CoolProp .json file.
    '''
    lntau = log(tau)
    A0 = log(delta)

    A0 += IdealGasHelmholtzLead_a1 + IdealGasHelmholtzLead_a2*tau

    A0 += IdealGasHelmholtzLogTau_a*lntau

    if IdealGasHelmholtzPlanckEinstein_ns is not None:
        for i in range(len(IdealGasHelmholtzPlanckEinstein_ns)):
            ni, ti = IdealGasHelmholtzPlanckEinstein_ns[i], IdealGasHelmholtzPlanckEinstein_ts[i]
            A0 += ni*log(1.0 - exp(-ti*tau))

    if IdealGasHelmholtzPower_ns is not None:
        for i in range(len(IdealGasHelmholtzPower_ns)):
            ni, ti = IdealGasHelmholtzPower_ns[i], IdealGasHelmholtzPower_ts[i]
            # A0 += ni*tau**ti
            A0 += ni*exp(ti*lntau)

    if IdealGasHelmholtzPlanckEinsteinGeneralized_ns is not None:
        for i in range(len(IdealGasHelmholtzPlanckEinsteinGeneralized_ns)):
            ni, ti = IdealGasHelmholtzPlanckEinsteinGeneralized_ns[i], IdealGasHelmholtzPlanckEinsteinGeneralized_ts[i]
            ci, di = IdealGasHelmholtzPlanckEinsteinGeneralized_cs[i], IdealGasHelmholtzPlanckEinsteinGeneralized_ds[i]
            A0 += ni*log(ci + di*exp(ti*tau))
    return A0

def Helmholtz_dA0_dtau(tau, delta=0.0,
            IdealGasHelmholtzLead_a1=0.0, IdealGasHelmholtzLead_a2=0.0,
            IdealGasHelmholtzLogTau_a=0.0,
           IdealGasHelmholtzPlanckEinstein_ns=None,
           IdealGasHelmholtzPlanckEinstein_ts=None,
           IdealGasHelmholtzPower_ns=None,
           IdealGasHelmholtzPower_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ns=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_cs=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ds=None):
    dA0 = 0.0
    dA0 += IdealGasHelmholtzLead_a2

    dA0 += IdealGasHelmholtzLogTau_a/tau

    if IdealGasHelmholtzPlanckEinstein_ns is not None:
        for i in range(len(IdealGasHelmholtzPlanckEinstein_ns)):
            ni, ti = IdealGasHelmholtzPlanckEinstein_ns[i], IdealGasHelmholtzPlanckEinstein_ts[i]
            x0 = exp(-tau*ti)
            dA0 += ni*ti*x0/(1.0 - x0) # done

    if IdealGasHelmholtzPower_ns is not None:
        lntau = log(tau)
        for i in range(len(IdealGasHelmholtzPower_ns)):
            ni, ti = IdealGasHelmholtzPower_ns[i], IdealGasHelmholtzPower_ts[i]
            # dA0 += ni*ti*tau**(ti-1) # done

            # This term could be calculated at the same time as A0
            dA0 += ni*ti*exp((ti-1.0)*lntau)

    if IdealGasHelmholtzPlanckEinsteinGeneralized_ns is not None:
        for i in range(len(IdealGasHelmholtzPlanckEinsteinGeneralized_ns)):
            ni, ti = IdealGasHelmholtzPlanckEinsteinGeneralized_ns[i], IdealGasHelmholtzPlanckEinsteinGeneralized_ts[i]
            ci, di = IdealGasHelmholtzPlanckEinsteinGeneralized_cs[i], IdealGasHelmholtzPlanckEinsteinGeneralized_ds[i]
            x0 = di*exp(tau*ti)
            dA0 += ni*ti*x0/(ci + x0) # done
    return dA0

def Helmholtz_d2A0_dtau2(tau, delta=0.0,
            IdealGasHelmholtzLead_a1=0.0, IdealGasHelmholtzLead_a2=0.0,
            IdealGasHelmholtzLogTau_a=0.0,
           IdealGasHelmholtzPlanckEinstein_ns=None,
           IdealGasHelmholtzPlanckEinstein_ts=None,
           IdealGasHelmholtzPower_ns=None,
           IdealGasHelmholtzPower_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ns=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_cs=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ds=None):
    d2A0 = 0.0

    d2A0 -= IdealGasHelmholtzLogTau_a/(tau*tau)

    if IdealGasHelmholtzPlanckEinstein_ns is not None:
        for i in range(len(IdealGasHelmholtzPlanckEinstein_ns)):
            ni, ti = IdealGasHelmholtzPlanckEinstein_ns[i], IdealGasHelmholtzPlanckEinstein_ts[i]
            x1 = exp(-tau*ti)
            x2 = x1*ti/(1.0 - x1)
            d2A0 -= ni*(ti + x2)*x2 # done

    if IdealGasHelmholtzPower_ns is not None:
        lntau = log(tau)
        for i in range(len(IdealGasHelmholtzPower_ns)):
            ni, ti = IdealGasHelmholtzPower_ns[i], IdealGasHelmholtzPower_ts[i]
            d2A0 += ni*ti*(ti - 1.0)*exp((ti-2.0)*lntau)

    if IdealGasHelmholtzPlanckEinsteinGeneralized_ns is not None:
        for i in range(len(IdealGasHelmholtzPlanckEinsteinGeneralized_ns)):
            ni, ti = IdealGasHelmholtzPlanckEinsteinGeneralized_ns[i], IdealGasHelmholtzPlanckEinsteinGeneralized_ts[i]
            ci, di = IdealGasHelmholtzPlanckEinsteinGeneralized_cs[i], IdealGasHelmholtzPlanckEinsteinGeneralized_ds[i]
            x0 = exp(tau*ti)
            x1 = di*x0/(ci + di*x0)
            d2A0 += -ni*ti*ti*(x1 - 1.0)*x1 # done
    return d2A0

def Helmholtz_d3A0_dtau3(tau, delta=0.0,
            IdealGasHelmholtzLead_a1=0.0, IdealGasHelmholtzLead_a2=0.0,
            IdealGasHelmholtzLogTau_a=0.0,
           IdealGasHelmholtzPlanckEinstein_ns=None,
           IdealGasHelmholtzPlanckEinstein_ts=None,
           IdealGasHelmholtzPower_ns=None,
           IdealGasHelmholtzPower_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ns=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_cs=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ds=None):
    d3A0 = 0.0

    d3A0 += 2.0*IdealGasHelmholtzLogTau_a/(tau*tau*tau)

    if IdealGasHelmholtzPlanckEinstein_ns is not None:
        for i in range(len(IdealGasHelmholtzPlanckEinstein_ns)):
            ni, ti = IdealGasHelmholtzPlanckEinstein_ns[i], IdealGasHelmholtzPlanckEinstein_ts[i]
            x1 = exp(-tau*ti)
            x3 = x1/(1.0 - x1)
            d3A0 += ni*ti*ti*ti*x3*(1.0 + x3*(3.0 + 2.0*x3))

    if IdealGasHelmholtzPower_ns is not None:
        lntau = log(tau)
        for i in range(len(IdealGasHelmholtzPower_ns)):
            ni, ti = IdealGasHelmholtzPower_ns[i], IdealGasHelmholtzPower_ts[i]
            d3A0 += ni*ti*(ti*ti - 3.0*ti + 2.0)*exp((ti-3.0)*lntau)

    if IdealGasHelmholtzPlanckEinsteinGeneralized_ns is not None:
        for i in range(len(IdealGasHelmholtzPlanckEinsteinGeneralized_ns)):
            ni, ti = IdealGasHelmholtzPlanckEinsteinGeneralized_ns[i], IdealGasHelmholtzPlanckEinsteinGeneralized_ts[i]
            ci, di = IdealGasHelmholtzPlanckEinsteinGeneralized_cs[i], IdealGasHelmholtzPlanckEinsteinGeneralized_ds[i]
            x1 = di*exp(tau*ti)
            x3 = x1/(ci + x1)
            d3A0 += ni*ti*ti*ti*x3*(1.0 + x3*(-3.0 + 2.0*x3))
    return d3A0

def Cp_ideal_gas_Helmholtz(T, Tc, R, IdealGasHelmholtzLead_a1=0.0, IdealGasHelmholtzLead_a2=0.0,
            IdealGasHelmholtzLogTau_a=0.0,
           IdealGasHelmholtzPlanckEinstein_ns=None,
           IdealGasHelmholtzPlanckEinstein_ts=None,
           IdealGasHelmholtzPower_ns=None,
           IdealGasHelmholtzPower_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ns=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_cs=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ds=None):
    tau = Tc/T

    d2A0 = Helmholtz_d2A0_dtau2(tau, delta=0,
                                IdealGasHelmholtzLead_a1=IdealGasHelmholtzLead_a1,
                                IdealGasHelmholtzLead_a2=IdealGasHelmholtzLead_a2,
            IdealGasHelmholtzLogTau_a=IdealGasHelmholtzLogTau_a,
           IdealGasHelmholtzPlanckEinstein_ns=IdealGasHelmholtzPlanckEinstein_ns,
           IdealGasHelmholtzPlanckEinstein_ts=IdealGasHelmholtzPlanckEinstein_ts,
           IdealGasHelmholtzPower_ns=IdealGasHelmholtzPower_ns,
           IdealGasHelmholtzPower_ts=IdealGasHelmholtzPower_ts,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ns=IdealGasHelmholtzPlanckEinsteinGeneralized_ns,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ts=IdealGasHelmholtzPlanckEinsteinGeneralized_ts,
           IdealGasHelmholtzPlanckEinsteinGeneralized_cs=IdealGasHelmholtzPlanckEinsteinGeneralized_cs,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ds=IdealGasHelmholtzPlanckEinsteinGeneralized_ds)
    Cp = R*(-tau*tau*(d2A0)+1.0)
    return Cp


def H_ideal_gas_Helmholtz(T, Tc, R, IdealGasHelmholtzLead_a1=0.0, IdealGasHelmholtzLead_a2=0.0,
            IdealGasHelmholtzLogTau_a=0.0,
           IdealGasHelmholtzPlanckEinstein_ns=None,
           IdealGasHelmholtzPlanckEinstein_ts=None,
           IdealGasHelmholtzPower_ns=None,
           IdealGasHelmholtzPower_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ns=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_cs=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ds=None):
    tau = Tc/T

    dA0 = Helmholtz_dA0_dtau(tau, delta=0,
                                IdealGasHelmholtzLead_a1=IdealGasHelmholtzLead_a1,
                                IdealGasHelmholtzLead_a2=IdealGasHelmholtzLead_a2,
            IdealGasHelmholtzLogTau_a=IdealGasHelmholtzLogTau_a,
           IdealGasHelmholtzPlanckEinstein_ns=IdealGasHelmholtzPlanckEinstein_ns,
           IdealGasHelmholtzPlanckEinstein_ts=IdealGasHelmholtzPlanckEinstein_ts,
           IdealGasHelmholtzPower_ns=IdealGasHelmholtzPower_ns,
           IdealGasHelmholtzPower_ts=IdealGasHelmholtzPower_ts,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ns=IdealGasHelmholtzPlanckEinsteinGeneralized_ns,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ts=IdealGasHelmholtzPlanckEinsteinGeneralized_ts,
           IdealGasHelmholtzPlanckEinsteinGeneralized_cs=IdealGasHelmholtzPlanckEinsteinGeneralized_cs,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ds=IdealGasHelmholtzPlanckEinsteinGeneralized_ds)
    H = R*T*(tau*dA0 + 1.0)
    return H

def S_ideal_gas_Helmholtz(T, Tc, R, IdealGasHelmholtzLead_a1=0.0, IdealGasHelmholtzLead_a2=0.0,
            IdealGasHelmholtzLogTau_a=0.0,
           IdealGasHelmholtzPlanckEinstein_ns=None,
           IdealGasHelmholtzPlanckEinstein_ts=None,
           IdealGasHelmholtzPower_ns=None,
           IdealGasHelmholtzPower_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ns=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ts=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_cs=None,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ds=None):
    '''Calculate the ideal-gas helmholtz energy, in J/mol.

    Note that this function CANNOT be used to implement calculate_integral_over_T
    and the heat capacity function CANNOT be integrated over T analytically,
    so unfortunately these fancy models must be done numerically.
    '''
    tau = Tc/T
    delta = 1.0

    A0 = Helmholtz_A0(tau, delta=delta,
                                IdealGasHelmholtzLead_a1=IdealGasHelmholtzLead_a1,
                                IdealGasHelmholtzLead_a2=IdealGasHelmholtzLead_a2,
            IdealGasHelmholtzLogTau_a=IdealGasHelmholtzLogTau_a,
           IdealGasHelmholtzPlanckEinstein_ns=IdealGasHelmholtzPlanckEinstein_ns,
           IdealGasHelmholtzPlanckEinstein_ts=IdealGasHelmholtzPlanckEinstein_ts,
           IdealGasHelmholtzPower_ns=IdealGasHelmholtzPower_ns,
           IdealGasHelmholtzPower_ts=IdealGasHelmholtzPower_ts,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ns=IdealGasHelmholtzPlanckEinsteinGeneralized_ns,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ts=IdealGasHelmholtzPlanckEinsteinGeneralized_ts,
           IdealGasHelmholtzPlanckEinsteinGeneralized_cs=IdealGasHelmholtzPlanckEinsteinGeneralized_cs,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ds=IdealGasHelmholtzPlanckEinsteinGeneralized_ds)
    dA0 = Helmholtz_dA0_dtau(tau, delta=delta,
                                IdealGasHelmholtzLead_a1=IdealGasHelmholtzLead_a1,
                                IdealGasHelmholtzLead_a2=IdealGasHelmholtzLead_a2,
            IdealGasHelmholtzLogTau_a=IdealGasHelmholtzLogTau_a,
           IdealGasHelmholtzPlanckEinstein_ns=IdealGasHelmholtzPlanckEinstein_ns,
           IdealGasHelmholtzPlanckEinstein_ts=IdealGasHelmholtzPlanckEinstein_ts,
           IdealGasHelmholtzPower_ns=IdealGasHelmholtzPower_ns,
           IdealGasHelmholtzPower_ts=IdealGasHelmholtzPower_ts,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ns=IdealGasHelmholtzPlanckEinsteinGeneralized_ns,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ts=IdealGasHelmholtzPlanckEinsteinGeneralized_ts,
           IdealGasHelmholtzPlanckEinsteinGeneralized_cs=IdealGasHelmholtzPlanckEinsteinGeneralized_cs,
           IdealGasHelmholtzPlanckEinsteinGeneralized_ds=IdealGasHelmholtzPlanckEinsteinGeneralized_ds)
    S = R*(tau*dA0 - A0)
    return S

"""Data from CoolProp, for the fast calculation of ideal-gas heat capacity.
"""

Helmholtz_A0_data = {'107-51-7': {'alpha0': {'IdealGasHelmholtzLead_a1': 117.9946064218,
   'IdealGasHelmholtzLead_a2': -19.6600754238,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [28.817, 46.951, 31.054],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.0353756335,
    2.7769872306,
    8.3132738751]},
  'R': 8.3144621,
  'Tc': 565.3609},
 '7446-09-5': {'alpha0': {'IdealGasHelmholtzLead_a1': -4.5414235721,
   'IdealGasHelmholtzLead_a2': 4.4732289572,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPower_ns': [-0.0159272204],
   'IdealGasHelmholtzPower_ts': [-1],
   'IdealGasHelmholtzPlanckEinstein_ns': [1.0875, 1.916],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.8182240386401636,
    4.328441389559726]},
  'R': 8.3144621,
  'Tc': 430.64},
 '590-18-1': {'alpha0': {'IdealGasHelmholtzLead_a1': 0.2591542,
   'IdealGasHelmholtzLead_a2': 2.4189888,
   'IdealGasHelmholtzLogTau_a': 2.9687,
   'IdealGasHelmholtzPlanckEinstein_ns': [3.2375, 7.0437, 11.414, 7.3722],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.5691336775674125,
    2.714859437751004,
    4.800917957544463,
    10.09064830751578]},
  'R': 8.314472,
  'Tc': 435.75},
 '108-38-3': {'alpha0': {'IdealGasHelmholtzLead_a1': 12.652887,
   'IdealGasHelmholtzLead_a2': 0.45975624,
   'IdealGasHelmholtzLogTau_a': 1.169909,
   'IdealGasHelmholtzPlanckEinstein_ns': [4.44312,
    2.862794,
    24.83298,
    16.26077],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.259365527079382,
    0.3079965634067662,
    2.160839047480102,
    5.667136766684498]},
  'R': 8.314472,
  'Tc': 616.89},
 '7440-59-7': {'alpha0': {'IdealGasHelmholtzLead_a1': 0.1871304489697973,
   'IdealGasHelmholtzLead_a2': 0.4848903984696551,
   'IdealGasHelmholtzLogTau_a': 1.5},
  'R': 8.3144598,
  'Tc': 5.1953},
 '1333-74-0p': {'alpha0': {'IdealGasHelmholtzLead_a1': -1.4485891134,
   'IdealGasHelmholtzLead_a2': 1.884521239,
   'IdealGasHelmholtzLogTau_a': 1.5,
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ts': [-15.14967514724634,
    -25.092598214827856,
    -29.473556378650795,
    -35.40591414172081,
    -40.72499848199648,
    -163.79257999878558,
    -309.2173173841763,
    -15.14967514724634,
    -25.092598214827856,
    -29.473556378650795,
    -35.40591414172081,
    -40.72499848199648,
    -163.79257999878558,
    -309.2173173841763],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_cs': [1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ds': [-1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ns': [4.30256,
    13.0289,
    -47.7365,
    50.0013,
    -18.6261,
    0.993973,
    0.536078]},
  'R': 8.314472,
  'Tc': 32.938},
 'R404A.PPF': {'alpha0': {'IdealGasHelmholtzLead_a1': 7.00407,
   'IdealGasHelmholtzLead_a2': 7.98695,
   'IdealGasHelmholtzLogTau_a': -1,
   'IdealGasHelmholtzPower_ns': [-18.8664],
   'IdealGasHelmholtzPower_ts': [-0.3],
   'IdealGasHelmholtzPlanckEinstein_ns': [0.63078, 3.5979, 5.0335],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.19617, 2.32861, 5.00188]},
  'R': 8.314472,
  'Tc': 345.27},
 '67-64-1': {'alpha0': {'IdealGasHelmholtzLead_a1': -9.4883659997,
   'IdealGasHelmholtzLead_a2': 7.1422719708,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [3.7072, 7.0675, 11.012],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.6101161188742373,
    6.849045463491438,
    3.101751623696123]},
  'R': 8.314472,
  'Tc': 508.1},
 '2314-97-8': {'alpha0': {'IdealGasHelmholtzLead_a1': 0,
   'IdealGasHelmholtzLead_a2': 0,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [6.2641],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.7505801634547473]},
  'R': 8.3144621,
  'Tc': 396.44},
 '76-15-3': {'alpha0': {'IdealGasHelmholtzLead_a1': 0,
   'IdealGasHelmholtzLead_a2': 0,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [7.142, 10.61],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.8184650240725007,
    3.6845086377796656]},
  'R': 8.3144621,
  'Tc': 353.1},
 '406-58-6': {'alpha0': {'IdealGasHelmholtzLead_a1': -16.3423704513,
   'IdealGasHelmholtzLead_a2': 10.2889710846,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [17.47, 16.29],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.236956521739131,
    4.852173913043479]},
  'R': 8.3144621,
  'Tc': 460},
 '95-47-6': {'alpha0': {'IdealGasHelmholtzLead_a1': 10.137376,
   'IdealGasHelmholtzLead_a2': -0.91282993,
   'IdealGasHelmholtzLogTau_a': 2.748798,
   'IdealGasHelmholtzPlanckEinstein_ns': [4.754892,
    6.915052,
    25.84813,
    10.93886],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.3569960920827787,
    0.9948291099373432,
    2.738556688599449,
    7.83963418213782]},
  'R': 8.314472,
  'Tc': 630.259},
 '110-54-3': {'alpha0': {'IdealGasHelmholtzLead_a1': 12.313791358169851,
   'IdealGasHelmholtzLead_a2': -1.3163412546284243,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [9.21, 6.04, 25.3, 10.96],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.3741483202709622,
    5.9076050569099285,
    2.9538025284549643,
    8.861407585364892]},
  'R': 8.3144598,
  'Tc': 507.82},
 '7440-37-1': {'alpha0': {'IdealGasHelmholtzLead_a1': 8.31666243,
   'IdealGasHelmholtzLead_a2': -4.94651164,
   'IdealGasHelmholtzLogTau_a': 1.5},
  'R': 8.31451,
  'Tc': 150.687},
 '624-64-6': {'alpha0': {'IdealGasHelmholtzLead_a1': 0.5917816,
   'IdealGasHelmholtzLead_a2': 2.1427758,
   'IdealGasHelmholtzLogTau_a': 2.9988,
   'IdealGasHelmholtzPlanckEinstein_ns': [5.3276, 13.29, 9.6745, 0.40087],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.8445906535078509,
    3.739996733627307,
    8.700216980471756,
    10.5620494155526]},
  'R': 8.314472,
  'Tc': 428.61},
 '112-40-3': {'alpha0': {'IdealGasHelmholtzLead_a1': 0,
   'IdealGasHelmholtzLead_a2': 0,
   'IdealGasHelmholtzLogTau_a': 22.085,
   'IdealGasHelmholtzPlanckEinstein_ns': [37.776, 29.369, 12.461, 7.7733],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.944993162133414,
    3.645342653092235,
    8.661297675125361,
    21.07430481689713]},
  'R': 8.314472,
  'Tc': 658.1},
 '7782-39-0p': {'alpha0': {'IdealGasHelmholtzLead_a1': -2.0683998716,
   'IdealGasHelmholtzLead_a2': 2.4241000701,
   'IdealGasHelmholtzLogTau_a': 1.5,
   'IdealGasHelmholtzPlanckEinstein_ns': [1.28527,
    1.11376,
    -2.491,
    6.38763,
    6.17406,
    -3.13698,
    -3.14254,
    -2.29511,
    -3.37,
    1.13634,
    0.72512],
   'IdealGasHelmholtzPlanckEinstein_ts': [132.1857068335941,
    26.10328638497652,
    6.820552947313511,
    11.40323422013563,
    8.145539906103286,
    9.984350547730829,
    9.30620761606677,
    7.686489306207615,
    17.79864371413667,
    6.416275430359937,
    7.227438706311946]},
  'R': 8.3144621,
  'Tc': 38.34},
 '100-41-4': {'alpha0': {'IdealGasHelmholtzLead_a1': 5.70409,
   'IdealGasHelmholtzLead_a2': -0.52414353,
   'IdealGasHelmholtzLogTau_a': 4.2557889,
   'IdealGasHelmholtzPlanckEinstein_ns': [9.7329909, 11.201832, 25.440749],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.9479517759917034,
    7.16230230749287,
    2.710980036297641]},
  'R': 8.314472,
  'Tc': 617.12},
 '115-07-1': {'alpha0': {'IdealGasHelmholtzLead_a1': 4.9916462,
   'IdealGasHelmholtzLead_a2': -0.1709449,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [1.544, 4.013, 8.923, 6.02],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.8895942187358427,
    2.671528317376466,
    5.304617378387802,
    11.85301926630442]},
  'R': 8.314472,
  'Tc': 364.211},
 '115-10-6': {'alpha0': {'IdealGasHelmholtzLead_a1': -1.980976,
   'IdealGasHelmholtzLead_a2': 3.171218,
   'IdealGasHelmholtzLogTau_a': 3.039,
   'IdealGasHelmholtzPlanckEinstein_ns': [2.641, 2.123, 8.992, 6.191],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.901647942694154,
    2.432701097462898,
    4.785477723551244,
    10.36520488138709]},
  'R': 8.314472,
  'Tc': 400.378},
 '29118-25-0': {'alpha0': {'IdealGasHelmholtzLead_a1': -2.422442259,
   'IdealGasHelmholtzLead_a2': 8.190539844,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [4.2365, 13.063],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.047251163559902665,
    3.154015167623503]},
  'R': 8.3144598,
  'Tc': 423.27},
 '7440-01-9': {'alpha0': {'IdealGasHelmholtzLead_a1': -3.0384719151147275,
   'IdealGasHelmholtzLead_a2': 3.253690479855404,
   'IdealGasHelmholtzLogTau_a': 1.5},
  'R': 8.3144598,
  'Tc': 44.4},
 '690-39-1': {'alpha0': {'IdealGasHelmholtzLead_a1': -17.5983849,
   'IdealGasHelmholtzLead_a2': 8.87150449,
   'IdealGasHelmholtzLogTau_a': 9.175,
   'IdealGasHelmholtzPlanckEinstein_ns': [9.8782, 18.236, 49.934],
   'IdealGasHelmholtzPlanckEinstein_ts': [2.416660386364208,
    6.014017635089306,
    13.03288366367724]},
  'R': 8.314472,
  'Tc': 398.07},
 '7782-41-4': {'alpha0': {'IdealGasHelmholtzPower_ns': [3.0717001e-06,
    -5.2985762e-05,
    -16.372517,
    3.6884682e-05,
    4.3887271],
   'IdealGasHelmholtzPower_ts': [-4, -3, 1, 2, 0],
   'IdealGasHelmholtzLogTau_a': 2.5011231,
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ns': [1.012767],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ts': [-8.9057501],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_cs': [1],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ds': [-1],
   'IdealGasHelmholtzLead_a1': 0,
   'IdealGasHelmholtzLead_a2': 0},
  'R': 8.31448,
  'Tc': 144.414},
 '1333-74-0': {'alpha0': {'IdealGasHelmholtzLead_a1': -1.4579856475,
   'IdealGasHelmholtzLead_a2': 1.888076782,
   'IdealGasHelmholtzLogTau_a': 1.5,
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ts': [-16.020515914919294,
    -22.658017800573237,
    -60.00905113893498,
    -74.94343038165636,
    -206.93920651682,
    -16.020515914919294,
    -22.658017800573237,
    -60.00905113893498,
    -74.94343038165636,
    -206.93920651682],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_cs': [1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ds': [-1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ns': [1.616,
    -0.4117,
    -0.792,
    0.758,
    1.217]},
  'R': 8.314472,
  'Tc': 33.145},
 '1333-74-0o': {'alpha0': {'IdealGasHelmholtzLead_a1': -1.4675442336,
   'IdealGasHelmholtzLead_a2': 1.8845068862,
   'IdealGasHelmholtzLogTau_a': 1.5,
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ts': [-25.76760987357014,
    -43.467790487658036,
    -66.04455147501506,
    -209.75316074653824,
    -25.76760987357014,
    -43.467790487658036,
    -66.04455147501506,
    -209.75316074653824],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_cs': [1, 1, 1, 1, 1, 1, 1, 1],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ds': [-1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ns': [2.54151,
    -2.3661,
    1.00365,
    1.22447]},
  'R': 8.314472,
  'Tc': 33.22},
 '75-71-8': {'alpha0': {'IdealGasHelmholtzLead_a1': 10.0100905,
   'IdealGasHelmholtzLead_a2': -4.66434985,
   'IdealGasHelmholtzLogTau_a': 3.00361975,
   'IdealGasHelmholtzPlanckEinstein_ns': [3.16062357,
    0.371258136,
    3.56226039,
    2.12152336],
   'IdealGasHelmholtzPlanckEinstein_ts': [3.72204562,
    6.30985083,
    1.78037889,
    1.07087607]},
  'R': 8.314471,
  'Tc': 385.12},
 '463-58-1': {'alpha0': {'IdealGasHelmholtzLead_a1': -3.6587449805,
   'IdealGasHelmholtzLead_a2': 3.7349245016,
   'IdealGasHelmholtzLogTau_a': 2.5,
   'IdealGasHelmholtzPlanckEinstein_ns': [2.1651, 0.93456, 1.0623, 0.34269],
   'IdealGasHelmholtzPlanckEinstein_ts': [2.027615703461204,
    3.59848984872086,
    8.382395649074637,
    33.87015867148929]},
  'R': 8.314472,
  'Tc': 378.77},
 '287-92-3': {'alpha0': {'IdealGasHelmholtzLead_a1': 3.2489131288,
   'IdealGasHelmholtzLead_a2': 2.6444166315,
   'IdealGasHelmholtzLogTau_a': 0.96,
   'IdealGasHelmholtzPlanckEinstein_ns': [3.34, 18.6, 13.9, 4.86],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.2345032439615415,
    2.540451809583366,
    5.276322989134683,
    10.35722660830141]},
  'R': 8.314472,
  'Tc': 511.72},
 '7439-90-9': {'alpha0': {'IdealGasHelmholtzLead_a1': -3.7506412806,
   'IdealGasHelmholtzLead_a2': 3.7798018435,
   'IdealGasHelmholtzLogTau_a': 1.5},
  'R': 8.314472,
  'Tc': 209.48},
 '10024-97-2': {'alpha0': {'IdealGasHelmholtzLead_a1': -4.4262736272,
   'IdealGasHelmholtzLead_a2': 4.3120475243,
   'IdealGasHelmholtzLogTau_a': 2.5,
   'IdealGasHelmholtzPlanckEinstein_ns': [2.1769, 1.6145, 0.48393],
   'IdealGasHelmholtzPlanckEinstein_ts': [2.839881106229,
    7.663478935125356,
    17.598216593435]},
  'R': 8.314472,
  'Tc': 309.52},
 '7664-41-7': {'alpha0': {'IdealGasHelmholtzLead_a1': -6.59406093943886,
   'IdealGasHelmholtzLead_a2': 5.60101151987913,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [2.224, 3.148, 0.9579],
   'IdealGasHelmholtzPlanckEinstein_ts': [4.0585856593352405,
    9.776605187888352,
    17.829667620080876]},
  'R': 8.3144598,
  'Tc': 405.56},
 '7782-39-0': {'alpha0': {'IdealGasHelmholtzLead_a1': -2.0677351753,
   'IdealGasHelmholtzLead_a2': 2.4237151502,
   'IdealGasHelmholtzLogTau_a': 1.5,
   'IdealGasHelmholtzPlanckEinstein_ns': [-3.54145,
    3.0326,
    -3.52422,
    -1.73421,
    -3.57135,
    2.14858,
    6.23107,
    -3.30425,
    6.23098,
    -3.57137,
    3.32901,
    0.97782],
   'IdealGasHelmholtzPlanckEinstein_ts': [187.1178925404277,
    225.2217005738132,
    23.54460093896714,
    4.723526343244653,
    11.43714136671883,
    131.3041210224309,
    7.039645279081897,
    5.996348461137194,
    17.38132498695879,
    11.81011997913406,
    5.007824726134585,
    30.97548252477829]},
  'R': 8.3144621,
  'Tc': 38.34},
 '7789-20-0': {'alpha0': {'IdealGasHelmholtzLead_a1': -8.670994022646008,
   'IdealGasHelmholtzLead_a2': 6.960335784587801,
   'IdealGasHelmholtzLogTau_a': 3.0,
   'IdealGasHelmholtzPlanckEinstein_ns': [0.010633, 0.99787, 2.1483, 0.3549],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.47837452065475183,
    2.632613027629235,
    6.1334447469662825,
    16.023993277906087]},
  'R': 8.3144598,
  'Tc': 643.847},
 '106-97-8': {'alpha0': {'IdealGasHelmholtzLead_a1': 12.54882924,
   'IdealGasHelmholtzLead_a2': -5.46976878,
   'IdealGasHelmholtzLogTau_a': 3.24680487,
   'IdealGasHelmholtzPlanckEinstein_ns': [5.54913289,
    11.4648996,
    7.59987584,
    9.66033239],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.774840445,
    3.340602552,
    4.970513096,
    9.975553778]},
  'R': 8.314472,
  'Tc': 425.125},
 '2551-62-4': {'alpha0': {'IdealGasHelmholtzLead_a1': 11.638611086,
   'IdealGasHelmholtzLead_a2': -6.392241811,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [3.66118232, 7.87885103, 3.45981679],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.617282065,
    2.747115139,
    4.232907175]},
  'R': 8.314472,
  'Tc': 318.7232},
 '75-21-8': {'alpha0': {'IdealGasHelmholtzLead_a1': -3.90644775,
   'IdealGasHelmholtzLead_a2': 4.0000956,
   'IdealGasHelmholtzLogTau_a': 3.0,
   'IdealGasHelmholtzPlanckEinstein_ns': [6.79, 4.53, 3.68],
   'IdealGasHelmholtzPlanckEinstein_ts': [2.8363047001620743,
    4.627655037106543,
    9.532542864454491]},
  'R': 8.3144621,
  'Tc': 468.92},
 'SES36.ppf': {'alpha0': {'IdealGasHelmholtzLead_a1': 0,
   'IdealGasHelmholtzLead_a2': 0,
   'IdealGasHelmholtzLogTau_a': 12.09,
   'IdealGasHelmholtzPlanckEinstein_ns': [85.26],
   'IdealGasHelmholtzPlanckEinstein_ts': [4.874639449744842]},
  'R': 8.314472,
  'Tc': 450.7},
 '74-87-3': {'alpha0': {'IdealGasHelmholtzLead_a1': 7.499423,
   'IdealGasHelmholtzLead_a2': -2.997533,
   'IdealGasHelmholtzLogTau_a': 2.92518,
   'IdealGasHelmholtzPower_ns': [-0.060842, -0.11525, 0.010843],
   'IdealGasHelmholtzPower_ts': [-1, -2, -3],
   'IdealGasHelmholtzPlanckEinstein_ns': [3.764997],
   'IdealGasHelmholtzPlanckEinstein_ts': [3.7101]},
  'R': 8.314472,
  'Tc': 416.3},
 'R507A.PPF': {'alpha0': {'IdealGasHelmholtzLead_a1': 9.93541,
   'IdealGasHelmholtzLead_a2': 7.9985,
   'IdealGasHelmholtzLogTau_a': -1,
   'IdealGasHelmholtzPower_ns': [-21.6054],
   'IdealGasHelmholtzPower_ts': [-0.25],
   'IdealGasHelmholtzPlanckEinstein_ns': [0.95006, 4.1887, 5.5184],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.05886, 2.37081, 5.14305]},
  'R': 8.314472,
  'Tc': 343.765},
 'AIR.PPF': {'alpha0': {'IdealGasHelmholtzLead_a1': 0,
   'IdealGasHelmholtzLead_a2': 0,
   'IdealGasHelmholtzPower_ns': [6.057194e-08,
    -2.10274769e-05,
    -0.000158860716,
    -13.841928076,
    17.275266575,
    -0.00019536342],
   'IdealGasHelmholtzPower_ts': [-3, -2, -1, 0, 1, 1.5],
   'IdealGasHelmholtzLogTau_a': 2.490888032,
   'IdealGasHelmholtzPlanckEinstein_ns': [0.791309509, 0.212236768],
   'IdealGasHelmholtzPlanckEinstein_ts': [25.36365, 16.90741],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ns': [-0.197938904],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ts': [87.31279],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_cs': [0.6666666666666666],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ds': [1]},
  'R': 8.31451,
  'Tc': 132.6312},
 '74-85-1': {'alpha0': {'IdealGasHelmholtzLead_a1': 8.68815523,
   'IdealGasHelmholtzLead_a2': -4.47960564,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [2.49395851,
    3.0027152,
    2.5126584,
    3.99064217],
   'IdealGasHelmholtzPlanckEinstein_ts': [4.43266896,
    5.74840149,
    7.8027825,
    15.5851154]},
  'R': 8.31451,
  'Tc': 282.35},
 '75-46-7': {'alpha0': {'IdealGasHelmholtzLead_a1': -8.31386064,
   'IdealGasHelmholtzLead_a2': 6.55087253,
   'IdealGasHelmholtzLogTau_a': 2.999,
   'IdealGasHelmholtzPlanckEinstein_ns': [2.371, 3.237, 2.61, 0.8274],
   'IdealGasHelmholtzPlanckEinstein_ts': [2.485858339486724,
    4.874821663052594,
    7.133477896242144,
    16.4086697650797]},
  'R': 8.314472,
  'Tc': 299.293},
 '75-28-5': {'alpha0': {'IdealGasHelmholtzLead_a1': 11.60865546,
   'IdealGasHelmholtzLead_a2': -5.29450411,
   'IdealGasHelmholtzLogTau_a': 3.05956619,
   'IdealGasHelmholtzPlanckEinstein_ns': [4.94641014,
    4.09475197,
    15.6632824,
    9.73918122],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.951277902,
    2.387895885,
    4.346904269,
    10.36885864]},
  'R': 8.314472,
  'Tc': 407.81},
 '7727-37-9': {'alpha0': {'IdealGasHelmholtzLead_a1': -12.76952708,
   'IdealGasHelmholtzLead_a2': -0.00784163,
   'IdealGasHelmholtzLogTau_a': 2.5,
   'IdealGasHelmholtzPower_ns': [-0.0001934819, -1.247742e-05, 6.678326e-08],
   'IdealGasHelmholtzPower_ts': [-1, -2, -3],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ts': [-26.657878470901483,
    -26.657878470901483],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_cs': [1, 1],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ds': [-1, -1],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ns': [1.012941]},
  'R': 8.31451,
  'Tc': 126.192},
 '29118-24-9': {'alpha0': {'IdealGasHelmholtzLead_a1': 0,
   'IdealGasHelmholtzLead_a2': 0,
   'IdealGasHelmholtzLogTau_a': 3.0,
   'IdealGasHelmholtzPlanckEinstein_ns': [9.3575, 10.717],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.34113088966911, 5.15538034001459]},
  'R': 8.314472,
  'Tc': 382.513},
 '679-86-7': {'alpha0': {'IdealGasHelmholtzLead_a1': -18.09410031,
   'IdealGasHelmholtzLead_a2': 8.996084665,
   'IdealGasHelmholtzLogTau_a': 7.888,
   'IdealGasHelmholtzPlanckEinstein_ns': [0.8843, 14.46, 5.331],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.9326585785463726,
    2.5068704336751795,
    6.323033268539]},
  'R': 8.3144621,
  'Tc': 447.57},
 '7440-63-3': {'alpha0': {'IdealGasHelmholtzLead_a1': -3.8227178129,
   'IdealGasHelmholtzLead_a2': 3.8416395351,
   'IdealGasHelmholtzLogTau_a': 1.5},
  'R': 8.314472,
  'Tc': 289.733},
 '7783-06-4': {'alpha0': {'IdealGasHelmholtzLead_a1': -4.0740770957,
   'IdealGasHelmholtzLead_a2': 3.7632137341,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPower_ns': [-0.002753352822675789],
   'IdealGasHelmholtzPower_ts': [-1.5],
   'IdealGasHelmholtzPlanckEinstein_ns': [1.1364, 1.9721],
   'IdealGasHelmholtzPlanckEinstein_ts': [4.886089520235862,
    10.62717770034843]},
  'R': 8.314472,
  'Tc': 373.1},
 '2837-89-0': {'alpha0': {'IdealGasHelmholtzLead_a1': -11.669406,
   'IdealGasHelmholtzLead_a2': 9.8760443,
   'IdealGasHelmholtzLogTau_a': 2.175638,
   'IdealGasHelmholtzPower_ns': [-7.389735, 0.8736831, -0.1115133],
   'IdealGasHelmholtzPower_ts': [-1, -2, -3]},
  'R': 8.314471,
  'Tc': 395.425},
 '107-83-5': {'alpha0': {'IdealGasHelmholtzLead_a1': 6.9259123919,
   'IdealGasHelmholtzLead_a2': -0.3128629679,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [7.9127, 16.871, 19.257, 14.075],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.6530038175607796,
    2.310628892907374,
    4.816154309825196,
    11.84046614426361]},
  'R': 8.314472,
  'Tc': 497.7},
 '78-78-4': {'alpha0': {'IdealGasHelmholtzLead_a1': 2.5822330405,
   'IdealGasHelmholtzLead_a2': 1.1609103419,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [7.4056, 9.5772, 15.765, 12.119],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.9601390246551537,
    2.409036602584989,
    4.494406429890301,
    9.108287172803301]},
  'R': 8.314472,
  'Tc': 460.35},
 '756-13-8': {'alpha0': {'IdealGasHelmholtzLead_a1': -30.6610503233,
   'IdealGasHelmholtzLead_a2': 6.8305296372,
   'IdealGasHelmholtzLogTau_a': 29.8,
   'IdealGasHelmholtzPlanckEinstein_ns': [29.8],
   'IdealGasHelmholtzPlanckEinstein_ts': [4.391027817387565]},
  'R': 8.3144621,
  'Tc': 441.81},
 'R407C.PPF': {'alpha0': {'IdealGasHelmholtzLead_a1': 2.13194,
   'IdealGasHelmholtzLead_a2': 8.05008,
   'IdealGasHelmholtzLogTau_a': -1,
   'IdealGasHelmholtzPower_ns': [-14.3914],
   'IdealGasHelmholtzPower_ts': [-0.4],
   'IdealGasHelmholtzPlanckEinstein_ns': [1.4245, 3.9419, 3.1209],
   'IdealGasHelmholtzPlanckEinstein_ts': [2.40437, 5.25122, 13.3632]},
  'R': 8.314472,
  'Tc': 359.345},
 '76-16-4': {'alpha0': {'IdealGasHelmholtzLead_a1': -10.7088650331,
   'IdealGasHelmholtzLead_a2': 8.9148979056,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [2.4818, 7.0622, 7.9951],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.6483977749718459,
    2.122649558065727,
    5.016551206361124]},
  'R': 8.314472,
  'Tc': 293.03},
 'R410A.PPF': {'alpha0': {'IdealGasHelmholtzLead_a1': 36.8871,
   'IdealGasHelmholtzLead_a2': 7.15807,
   'IdealGasHelmholtzLogTau_a': -1,
   'IdealGasHelmholtzPower_ns': [-46.87575],
   'IdealGasHelmholtzPower_ts': [-0.1],
   'IdealGasHelmholtzPlanckEinstein_ns': [2.0623, 5.9751, 1.5612],
   'IdealGasHelmholtzPlanckEinstein_ts': [2.02326, 5.00154, 11.2484]},
  'R': 8.314472,
  'Tc': 344.494},
 '76-19-7': {'alpha0': {'IdealGasHelmholtzLead_a1': -15.6587335175,
   'IdealGasHelmholtzLead_a2': 11.4531412796,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [7.2198, 7.2692, 11.599],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.9448727609993625,
    1.724537707958959,
    4.315691843951075]},
  'R': 8.314472,
  'Tc': 345.02},
 '7782-44-7': {'alpha0': {'IdealGasHelmholtzLead_a1': 0,
   'IdealGasHelmholtzLead_a2': 0,
   'IdealGasHelmholtzLogTau_a': 2.51808732,
   'IdealGasHelmholtzPlanckEinstein_ns': [1.02323928,
    0.784357918,
    0.00337183363,
    -0.0170864084,
    0.0463751562],
   'IdealGasHelmholtzPlanckEinstein_ts': [14.5316979447668,
    72.8419165356674,
    7.7710849975094,
    0.446425786480874,
    34.4677188658373]},
  'R': 8.31434,
  'Tc': 154.581},
 '106-98-9': {'alpha0': {'IdealGasHelmholtzLead_a1': -0.00101126,
   'IdealGasHelmholtzLead_a2': 2.3869174,
   'IdealGasHelmholtzLogTau_a': 2.9197,
   'IdealGasHelmholtzPlanckEinstein_ns': [2.9406, 6.5395, 14.535, 5.8971],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.6534856543203987,
    2.268119917002552,
    5.072861265472584,
    13.71842877244866]},
  'R': 8.314472,
  'Tc': 419.29},
 '74-98-6': {'alpha0': {'IdealGasHelmholtzLead_a1': -4.970583,
   'IdealGasHelmholtzLead_a2': 4.29352,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [3.043, 5.874, 9.337, 7.922],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.062478,
    3.344237,
    5.363757,
    11.762957]},
  'R': 8.314472,
  'Tc': 369.89},
 '64-17-5': {'alpha0': {'IdealGasHelmholtzLead_a1': -12.7531,
   'IdealGasHelmholtzLead_a2': 9.39094,
   'IdealGasHelmholtzLogTau_a': 3.43069,
   'IdealGasHelmholtzPlanckEinstein_ns': [2.14326, 5.09206, 6.60138, 5.70777],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.816771,
    2.59175,
    3.80408,
    8.58736]},
  'R': 8.314472,
  'Tc': 514.71},
 '107-52-8': {'alpha0': {'IdealGasHelmholtzLead_a1': 88.10187245449355,
   'IdealGasHelmholtzLead_a2': -39.55376118918867,
   'IdealGasHelmholtzLogTau_a': 3.0,
   'IdealGasHelmholtzPlanckEinstein_ns': [97.16, 69.73, 38.43],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.9338640538885487,
    3.7966932026944273,
    9.797917942437232]},
  'R': 8.3144598,
  'Tc': 653.2},
 '124-38-9': {'alpha0': {'IdealGasHelmholtzLead_a1': 8.37304456,
   'IdealGasHelmholtzLead_a2': -3.70454304,
   'IdealGasHelmholtzLogTau_a': 2.5,
   'IdealGasHelmholtzPlanckEinstein_ns': [1.99427042,
    0.62105248,
    0.41195293,
    1.04028922,
    0.08327678],
   'IdealGasHelmholtzPlanckEinstein_ts': [3.15163,
    6.1119,
    6.77708,
    11.32384,
    27.08792]},
  'R': 8.31451,
  'Tc': 304.1282},
 '74-84-0': {'alpha0': {'IdealGasHelmholtzLead_a1': 9.212802589,
   'IdealGasHelmholtzLead_a2': -4.68224855,
   'IdealGasHelmholtzLogTau_a': 3.003039265,
   'IdealGasHelmholtzPlanckEinstein_ns': [1.117433359,
    3.467773215,
    6.94194464,
    5.970850948],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.409105233,
    4.009917071,
    6.596709834,
    13.97981027]},
  'R': 8.314472,
  'Tc': 305.322},
 '71-43-2': {'alpha0': {'IdealGasHelmholtzLead_a1': -0.6740687105,
   'IdealGasHelmholtzLead_a2': 2.5560188958,
   'IdealGasHelmholtzLogTau_a': 2.94645,
   'IdealGasHelmholtzPlanckEinstein_ns': [7.36374, 18.649, 4.01834],
   'IdealGasHelmholtzPlanckEinstein_ts': [7.32358279064802,
    2.6885164229031,
    1.120956549588983]},
  'R': 8.314472,
  'Tc': 562.02},
 '556-67-2': {'alpha0': {'IdealGasHelmholtzLead_a1': 71.1636049792958,
   'IdealGasHelmholtzLead_a2': -21.6743650975623,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [0.292757, 38.2456, 58.975],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.06820119352088662,
    0.3410059676044331,
    3.0690537084398977]},
  'R': 8.3144621,
  'Tc': 586.5},
 '75-10-5': {'alpha0': {'IdealGasHelmholtzLead_a1': -8.258096,
   'IdealGasHelmholtzLead_a2': 6.353098,
   'IdealGasHelmholtzLogTau_a': 3.004486,
   'IdealGasHelmholtzPlanckEinstein_ns': [1.160761,
    2.645151,
    5.794987,
    1.129475],
   'IdealGasHelmholtzPlanckEinstein_ts': [2.2718538,
    11.914421,
    5.1415638,
    32.768217]},
  'R': 8.314471,
  'Tc': 351.255},
 '107-06-2': {'alpha0': {'IdealGasHelmholtzLead_a1': 25.029988,
   'IdealGasHelmholtzLead_a2': -4.8999527,
   'IdealGasHelmholtzLogTau_a': 3.0,
   'IdealGasHelmholtzPlanckEinstein_ns': [5.35, 10.05],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.04006410256410256,
    3.587962962962963]},
  'R': 8.3144621,
  'Tc': 561.6},
 '107-46-0': {'alpha0': {'IdealGasHelmholtzLead_a1': 0,
   'IdealGasHelmholtzLead_a2': 0,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [19.74, 29.58, 18.59, 4.87],
   'IdealGasHelmholtzPlanckEinstein_ts': [6.940427993,
    2.699055331,
    0.038557933,
    12.14574899]},
  'R': 8.3144621,
  'Tc': 518.7},
 '60-29-7': {'alpha0': {'IdealGasHelmholtzLead_a1': 17.099494,
   'IdealGasHelmholtzLead_a2': -6.160844,
   'IdealGasHelmholtzLogTau_a': 3.36281,
   'IdealGasHelmholtzPower_ns': [-8.943822, 0.54621, -0.016604],
   'IdealGasHelmholtzPower_ts': [-1, -2, -3]},
  'R': 8.314472,
  'Tc': 466.7},
 '76-13-1': {'alpha0': {'IdealGasHelmholtzLead_a1': 13.1479282,
   'IdealGasHelmholtzLead_a2': -5.4053715,
   'IdealGasHelmholtzLogTau_a': 2.9999966,
   'IdealGasHelmholtzPlanckEinstein_ns': [12.4464495,
    2.72181845,
    0.692712415,
    3.32248298],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.04971737,
    3.29788641,
    8.62650812,
    3.29670446]},
  'R': 8.314471,
  'Tc': 487.21},
 '463-82-1': {'alpha0': {'IdealGasHelmholtzLead_a1': 0.8702452614,
   'IdealGasHelmholtzLead_a2': 1.6071746358,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [14.422, 12.868, 17.247, 12.663],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.636925346982063,
    3.977036934569097,
    7.562133997325587,
    17.9531516576751]},
  'R': 8.314472,
  'Tc': 433.74},
 '111-84-2': {'alpha0': {'IdealGasHelmholtzLead_a1': 10.7927224829,
   'IdealGasHelmholtzLead_a2': -8.2418318753,
   'IdealGasHelmholtzLogTau_a': 16.349,
   'IdealGasHelmholtzPlanckEinstein_ns': [24.926, 24.842, 11.188, 17.483],
   'IdealGasHelmholtzPlanckEinstein_ts': [2.053654024051804,
    3.774283071230343,
    8.423177192834919,
    19.71911529728366]},
  'R': 8.314472,
  'Tc': 594.55},
 '431-89-0': {'alpha0': {'IdealGasHelmholtzLead_a1': -15.8291124137,
   'IdealGasHelmholtzLead_a2': 11.0879509962,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [11.43, 12.83],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.07495332088557,
    3.809015737530008]},
  'R': 8.3144621,
  'Tc': 374.9},
 '1717-00-6': {'alpha0': {'IdealGasHelmholtzLead_a1': -15.5074814985,
   'IdealGasHelmholtzLead_a2': 9.1871858933,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [6.8978, 7.8157, 3.2039],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.05130890052356,
    3.290052356020942,
    9.63979057591623]},
  'R': 8.314472,
  'Tc': 477.5},
 '108-88-3': {'alpha0': {'IdealGasHelmholtzLead_a1': 3.5241174832,
   'IdealGasHelmholtzLead_a2': 1.1360823464,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [1.6994,
    8.0577,
    17.059,
    8.4567,
    8.6423],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.3210815378115758,
    1.346852555978031,
    2.735952682720744,
    5.191381495564005,
    13.37558090409801]},
  'R': 8.314472,
  'Tc': 591.75},
 '111-65-9': {'alpha0': {'IdealGasHelmholtzLead_a1': 16.93282558002394,
   'IdealGasHelmholtzLead_a2': -4.06060393716559,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [17.47, 33.25, 15.63],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.6681436157119246,
    3.031262088124626,
    6.8238562436262615]},
  'R': 8.3144598,
  'Tc': 568.74},
 '7782-39-0o': {'alpha0': {'IdealGasHelmholtzLead_a1': -2.0672670563,
   'IdealGasHelmholtzLead_a2': 2.4234599781,
   'IdealGasHelmholtzLogTau_a': 1.5,
   'IdealGasHelmholtzPlanckEinstein_ns': [4.04482,
    -4.65391,
    -4.65342,
    3.46313,
    -4.58637,
    -4.6503,
    -4.65124,
    2.67024,
    15.20455,
    0.87164,
    -4.7608,
    4.32447],
   'IdealGasHelmholtzPlanckEinstein_ts': [41.49713093375065,
    12.56129368805425,
    12.32133541992697,
    9.447052686489306,
    53.15597287428273,
    12.08137715179969,
    12.81429316640584,
    70.77203964527908,
    16.13458528951487,
    225.4042775169536,
    25.08346374543558,
    6.604068857589984]},
  'R': 8.3144621,
  'Tc': 38.34},
 '106-42-3': {'alpha0': {'IdealGasHelmholtzLead_a1': 5.9815241,
   'IdealGasHelmholtzLead_a2': -0.52477835,
   'IdealGasHelmholtzLogTau_a': 4.2430504,
   'IdealGasHelmholtzPlanckEinstein_ns': [5.2291378,
    19.549862,
    16.656178,
    5.9390291],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.6718946780748107,
    2.038405110294595,
    4.299152179275782,
    10.84282208748263]},
  'R': 8.314472,
  'Tc': 616.168},
 '141-63-9': {'alpha0': {'IdealGasHelmholtzLead_a1': 68.11672041661723,
   'IdealGasHelmholtzLead_a2': -29.80919654255515,
   'IdealGasHelmholtzLogTau_a': 3.0,
   'IdealGasHelmholtzPlanckEinstein_ns': [81.2386, 61.191, 51.1798],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.9713375796178344,
    3.9808917197452227,
    11.94267515923567]},
  'R': 8.3144598,
  'Tc': 628.0},
 '616-38-6': {'alpha0': {'IdealGasHelmholtzLead_a1': 4.9916462,
   'IdealGasHelmholtzLead_a2': -0.1709449,
   'IdealGasHelmholtzLogTau_a': 8.28421,
   'IdealGasHelmholtzPlanckEinstein_ns': [1.48525, 0.822585, 16.2453, 1.15925],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.03770197486535009,
    2.405745062836625,
    3.001795332136445,
    13.27648114901257]},
  'R': 8.314472,
  'Tc': 557},
 '141-62-8': {'alpha0': {'IdealGasHelmholtzLead_a1': 131.089725,
   'IdealGasHelmholtzLead_a2': -26.3839138,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [28.59, 56.42, 50.12],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.033366700033366704,
    1.9686353019686353,
    7.073740407073741]},
  'R': 8.3144598,
  'Tc': 599.4},
 '115-11-7': {'alpha0': {'IdealGasHelmholtzLead_a1': -0.12737888,
   'IdealGasHelmholtzLead_a2': 2.3125128,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [4.8924, 7.832, 7.2867, 8.7293],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.9543399746466072,
    3.037623478198474,
    4.795618168336961,
    9.60797914324667]},
  'R': 8.314472,
  'Tc': 418.09},
 '811-97-2': {'alpha0': {'IdealGasHelmholtzLead_a1': -1.019535,
   'IdealGasHelmholtzLead_a2': 9.047135,
   'IdealGasHelmholtzLogTau_a': -1.629789,
   'IdealGasHelmholtzPower_ns': [-9.723916, -3.92717],
   'IdealGasHelmholtzPower_ts': [-0.5, -0.75]},
  'R': 8.314471,
  'Tc': 374.18},
 '124-18-5': {'alpha0': {'IdealGasHelmholtzLead_a1': 13.9361966549,
   'IdealGasHelmholtzLead_a2': -10.5265128286,
   'IdealGasHelmholtzLogTau_a': 18.109,
   'IdealGasHelmholtzPlanckEinstein_ns': [25.685, 28.233, 12.417, 10.035],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.931358264529707,
    3.46446495062328,
    7.710862878419944,
    17.58458798769629]},
  'R': 8.314472,
  'Tc': 617.7},
 '67-56-1': {'alpha0': {'IdealGasHelmholtzLead_a1': 13.9864114647,
   'IdealGasHelmholtzLead_a2': 3200.6369296,
   'IdealGasHelmholtzLogTau_a': 3.1950423807804,
   'IdealGasHelmholtzPower_ns': [-0.585735321498174,
    -0.06899642310301084,
    0.008650264506162275],
   'IdealGasHelmholtzPower_ts': [-1, -2, -3],
   'IdealGasHelmholtzPlanckEinstein_ns': [4.70118076896145],
   'IdealGasHelmholtzPlanckEinstein_ts': [3.7664265756]},
  'R': 8.314472,
  'Tc': 512.5},
 '7647-01-0': {'alpha0': {'IdealGasHelmholtzLead_a1': -4.069044527,
   'IdealGasHelmholtzLead_a2': 4.0257768311,
   'IdealGasHelmholtzLogTau_a': 2.5,
   'IdealGasHelmholtzPlanckEinstein_ns': [0.0033327, 0.935243, 0.209996],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.9239866945915979,
    12.319822594554639,
    19.403720586423557]},
  'R': 8.3144598,
  'Tc': 324.68},
 '353-36-6': {'alpha0': {'IdealGasHelmholtzLead_a1': -6.9187,
   'IdealGasHelmholtzLead_a2': 5.4788,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [2.059, 9.253, 6.088],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.11925383077948,
    4.125249833444371,
    10.34510326449034]},
  'R': 8.314472,
  'Tc': 375.25},
 '431-63-0': {'alpha0': {'IdealGasHelmholtzLead_a1': -14.121424135,
   'IdealGasHelmholtzLead_a2': 10.2355589225,
   'IdealGasHelmholtzLogTau_a': 2.762,
   'IdealGasHelmholtzPlanckEinstein_ns': [0.7762, 10.41, 12.18, 3.332],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.3491416933372127,
    0.93346911065852,
    3.724178062263602,
    17.26554165454369]},
  'R': 8.314472,
  'Tc': 412.44},
 '7732-18-5': {'alpha0': {'IdealGasHelmholtzLead_a1': -8.3204464837497,
   'IdealGasHelmholtzLead_a2': 6.6832105275932,
   'IdealGasHelmholtzLogTau_a': 3.00632,
   'IdealGasHelmholtzPlanckEinstein_ns': [0.012436,
    0.97315,
    1.2795,
    0.96956,
    0.24873],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.28728967,
    3.53734222,
    7.74073708,
    9.24437796,
    27.5075105]},
  'R': 8.314371357587,
  'Tc': 647.096},
 '677-21-4': {'alpha0': {'IdealGasHelmholtzLead_a1': -12.06546081,
   'IdealGasHelmholtzLead_a2': 8.207216743,
   'IdealGasHelmholtzLogTau_a': 3.0,
   'IdealGasHelmholtzPlanckEinstein_ns': [11.247, 8.1391],
   'IdealGasHelmholtzPlanckEinstein_ts': [2.093226859098506,
    5.826015440532725]},
  'R': 8.3144598,
  'Tc': 376.93},
 '110-82-7': {'alpha0': {'IdealGasHelmholtzLead_a1': 0.9891140602,
   'IdealGasHelmholtzLead_a2': 1.6359660572,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [0.83775, 16.036, 24.636, 7.1715],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.396315028901734,
    1.69978323699422,
    3.946893063583815,
    8.11958092485549]},
  'R': 8.3144621,
  'Tc': 553.6},
 '630-08-0': {'alpha0': {'IdealGasHelmholtzLead_a1': -3.3728318564,
   'IdealGasHelmholtzLead_a2': 3.3683460039,
   'IdealGasHelmholtzLogTau_a': 2.5,
   'IdealGasHelmholtzPower_ns': [-9.111274701235156e-05],
   'IdealGasHelmholtzPower_ts': [-1.5],
   'IdealGasHelmholtzPlanckEinstein_ns': [1.0128],
   'IdealGasHelmholtzPlanckEinstein_ts': [23.25003763359927]},
  'R': 8.314472,
  'Tc': 132.86},
 '460-73-1': {'alpha0': {'IdealGasHelmholtzLead_a1': -13.4283638514,
   'IdealGasHelmholtzLead_a2': 9.87236538,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [5.5728, 10.385, 12.554],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.5197115834815994,
    2.364453600524394,
    5.735555763648281]},
  'R': 8.3144621,
  'Tc': 427.01},
 '109-66-0': {'alpha0': {'IdealGasHelmholtzLead_a1': 8.509252832282648,
   'IdealGasHelmholtzLead_a2': 0.06430584062033375,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [6.618, 15.97, 15.29],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.3278688524590164,
    2.8188205237385566,
    5.607834788162657]},
  'R': 8.3144598,
  'Tc': 469.7},
 '593-53-3': {'alpha0': {'IdealGasHelmholtzLead_a1': -4.867644116,
   'IdealGasHelmholtzLead_a2': 4.2527951258,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPower_ns': [-0.0268688568],
   'IdealGasHelmholtzPower_ts': [-1],
   'IdealGasHelmholtzPlanckEinstein_ns': [5.6936, 2.9351],
   'IdealGasHelmholtzPlanckEinstein_ts': [5.802445789208271,
    13.33837619768028]},
  'R': 8.314472,
  'Tc': 317.28},
 '541-02-6': {'alpha0': {'IdealGasHelmholtzLead_a1': 94.38924286309562,
   'IdealGasHelmholtzLead_a2': -31.110222240232282,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [51.0, 57.9, 35.0],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.3574316674753356,
    2.802846514636908,
    7.349183244379751]},
  'R': 8.3144598,
  'Tc': 618.3},
 '74-82-8': {'alpha0': {'IdealGasHelmholtzLead_a1': 9.91243972,
   'IdealGasHelmholtzLead_a2': -6.33270087,
   'IdealGasHelmholtzLogTau_a': 3.0016,
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ts': [-3.4004324006632944,
    -10.269515753237759,
    -20.439327470036314,
    -29.937448836086567,
    -79.1335194475347,
    -3.4004324006632944,
    -10.269515753237759,
    -20.439327470036314,
    -29.937448836086567,
    -79.1335194475347],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_cs': [1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ds': [-1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1],
   'IdealGasHelmholtzPlanckEinsteinGeneralized_ns': [0.008449,
    4.6942,
    3.4865,
    1.6572,
    1.4115]},
  'R': 8.31451,
  'Tc': 190.564},
 '754-12-1': {'alpha0': {'IdealGasHelmholtzLead_a1': -12.837928,
   'IdealGasHelmholtzLead_a2': 8.042605,
   'IdealGasHelmholtzLogTau_a': 4.944,
   'IdealGasHelmholtzPlanckEinstein_ns': [7.549, 1.537, 2.03, 7.455],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.951882560826424,
    2.384123963572108,
    12.13809976892755,
    4.770966426532554]},
  'R': 8.314472,
  'Tc': 367.85},
 '354-33-6': {'alpha0': {'IdealGasHelmholtzLead_a1': 37.2674,
   'IdealGasHelmholtzLead_a2': 8.88404,
   'IdealGasHelmholtzLogTau_a': -1,
   'IdealGasHelmholtzPower_ns': [-49.8651],
   'IdealGasHelmholtzPower_ts': [-0.1],
   'IdealGasHelmholtzPlanckEinstein_ns': [2.303, 5.086, 7.3],
   'IdealGasHelmholtzPlanckEinstein_ts': [0.92578, 2.22895, 5.03283]},
  'R': 8.314472,
  'Tc': 339.173},
 '75-68-3': {'alpha0': {'IdealGasHelmholtzLead_a1': -12.6016527149,
   'IdealGasHelmholtzLead_a2': 8.3160183265,
   'IdealGasHelmholtzLogTau_a': 3,
   'IdealGasHelmholtzPlanckEinstein_ns': [5.0385, 6.8356, 4.0591, 2.8136],
   'IdealGasHelmholtzPlanckEinstein_ts': [1.152927411885146,
    3.061473212109394,
    6.086384244137864,
    16.67235411690148]},
  'R': 8.314472,
  'Tc': 410.26}}
