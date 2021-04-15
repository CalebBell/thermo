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

__all__ = ['has_CoolProp', 'coolprop_dict', 'CP_fluid', 'coolprop_fluids',
'CoolProp_T_dependent_property', 'CoolProp_failing_PT_flashes',
'PropsSI', 'PhaseSI','HAPropsSI', 'AbstractState']
import os
from fluids.numerics import assert_close1d, numpy as np
from thermo.base import data_dir, source_path

#try:
#    import CoolProp
#    has_CoolProp = True
##    CPiP_min = CP.iP_min
#except:  # pragma: no cover
#    # Don't just except
#    has_CoolProp = False
#has_CoolProp = False # For testing

CPiP_min = 17
global _PropsSI

global _has_CoolProp
_has_CoolProp = None
def has_CoolProp():
    global _has_CoolProp
    if _has_CoolProp is None:
        try:
            import CoolProp
            load_coolprop_fluids()
            _has_CoolProp = True
        except:
            _has_CoolProp = False
    return _has_CoolProp

_PropsSI = None
def PropsSI(*args, **kwargs):
    global _PropsSI
    if _PropsSI is None:
        from CoolProp.CoolProp import PropsSI as _PropsSI
    return _PropsSI(*args, **kwargs)

global _HAPropsSI
_HAPropsSI = None
def HAPropsSI(*args, **kwargs):
    global _HAPropsSI
    if _HAPropsSI is None:
        from CoolProp.CoolProp import HAPropsSI as _HAPropsSI
    return _HAPropsSI(*args, **kwargs)

global _PhaseSI
_PhaseSI = None
def PhaseSI(*args, **kwargs):
    global _PhaseSI
    if _PhaseSI is None:
        from CoolProp.CoolProp import PhaseSI as _PhaseSI
    return _PhaseSI(*args, **kwargs)

global _AbstractState
_AbstractState = None
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


CoolProp_has_mu_CASs = set(['74-82-8', '109-66-0', '67-56-1', '115-07-1', '76-16-4', '75-72-9', '811-97-2', '75-73-0', '1717-00-6', '75-68-3', '76-19-7', '431-89-0', '431-63-0', '690-39-1', '115-25-3', '75-69-4', '75-71-8', '420-46-2', '306-83-2', '102687-65-0', '754-12-1', '29118-24-9', '2837-89-0', '75-37-6', '75-45-6', '460-73-1', '75-10-5', '354-33-6', '75-46-7', 'R404A.PPF', 'R407C.PPF', 'R410A.PPF', 'R507A.PPF', '2551-62-4', '108-88-3', '7732-18-5', '108-38-3', '106-97-8', '124-18-5', '111-84-2', '111-65-9', '112-40-3', '142-82-5', '110-54-3', '74-98-6', '95-47-6', '106-42-3', 'AIR.PPF', '7440-37-1', '7727-37-9', '7782-44-7', '7664-41-7', '71-43-2', '124-38-9', '110-82-7', '287-92-3', '78-78-4', '115-10-6', '74-84-0', '64-17-5', '7789-20-0', '7440-59-7', '1333-74-0', '7783-06-4', '75-28-5'])
CoolProp_has_k_CASs = set(['74-82-8', '67-56-1', '115-07-1', '76-16-4', '75-72-9', '75-73-0', '1717-00-6', '75-68-3', '76-19-7', '431-89-0', '431-63-0', '690-39-1', '115-25-3', '2837-89-0', '460-73-1', '75-10-5', '811-97-2', '75-69-4', '75-71-8', '420-46-2', '75-45-6', '306-83-2', '754-12-1', '29118-24-9', '354-33-6', '75-37-6', '75-46-7', 'R404A.PPF', 'R407C.PPF', 'R410A.PPF', 'R507A.PPF', '2551-62-4', '108-88-3', '7732-18-5', '106-97-8', '124-18-5', '111-84-2', '111-65-9', '112-40-3', '142-82-5', '110-54-3', '74-98-6', 'AIR.PPF', '7440-37-1', '7727-37-9', '7782-44-7', '7664-41-7', '71-43-2', '124-38-9', '109-66-0', '287-92-3', '78-78-4', '74-84-0', '64-17-5', '100-41-4', '108-38-3', '95-47-6', '106-42-3', '7789-20-0', '7440-59-7', '1333-74-0p', '1333-74-0', '75-28-5'])

CoolProp_k_failing_CASs = set(['100-41-4', '2837-89-0', '460-73-1', '75-10-5', '75-45-6'])

CoolProp_failing_PT_flashes = set(['115-07-1', '115-25-3', '1717-00-6', '420-46-2',
                                '431-63-0', '431-89-0', '690-39-1', '75-68-3', '75-69-4', '75-71-8', '75-72-9', '75-73-0', '76-19-7',
                                '110-82-7', '7782-44-7'])

class CP_fluid(object):
    # Basic object to store constants for a coolprop fluid, much faster than
    # calling coolprop to retrieve the data when needed
    __slots__ = ['Tmin', 'Tmax', 'Pmax', 'has_melting_line', 'Tc', 'Pc', 'Tt',
                 'omega', 'HEOS', 'CAS']
    @property
    def has_k(self):
        return self.CAS in CoolProp_has_k_CASs and self.CAS not in CoolProp_k_failing_CASs

    @property
    def has_mu(self):
        return self.CAS in CoolProp_has_mu_CASs

    def as_json(self):
        return {k: getattr(self, k) for k in self.__slots__}
    def __deepcopy__(self, memo):
        # AbstractState("HEOS", CAS) has no deep copy;
        # fortunately, none is needed, so we can just return the existing copy
        return self

    def __init__(self, Tmin, Tmax, Pmax, has_melting_line, Tc, Pc, Tt, omega,
                 HEOS, CAS):
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

def store_coolprop_fluids():
    import CoolProp
    import json
    for CASRN in coolprop_dict:
        HEOS = AbstractState("HEOS", CASRN)
        coolprop_fluids[CASRN] = CP_fluid(Tmin=HEOS.Tmin(), Tmax=HEOS.Tmax(), Pmax=HEOS.pmax(),
                       has_melting_line=HEOS.has_melting_line(), Tc=HEOS.T_critical(), Pc=HEOS.p_critical(),
                       Tt=HEOS.Ttriple(), omega=HEOS.acentric_factor(), HEOS=None, CAS=CASRN)

    data = {CASRN: coolprop_fluids[CASRN].as_json() for CASRN in coolprop_dict}
    ver = CoolProp.__version__
    file = open(os.path.join(data_dir, 'CoolPropFluids%s.json' %ver), 'w')
    json.dump(data, file)
    file.close()

def load_coolprop_fluids():
    import json
    import CoolProp
    ver = CoolProp.__version__
    pth = os.path.join(data_dir, 'CoolPropFluids%s.json' %ver)
    try:
        file = open(pth, 'r')
    except:
        store_coolprop_fluids()
        file = open(pth, 'r')
    data = json.load(file)
    for CASRN in coolprop_dict:
        d = data[CASRN]
        coolprop_fluids[CASRN] = CP_fluid(Tmin=d['Tmin'], Tmax=d['Tmax'], Pmax=d['Pmax'],
                       has_melting_line=d['has_melting_line'], Tc=d['Tc'], Pc=d['Pc'],
                       Tt=d['Tt'], omega=d['omega'], HEOS=None, CAS=CASRN)



class MultiCheb1D(object):
    '''Simple class to store set of coefficients for multiple chebshev
    approximations and perform calculations from them.
    '''
    def __init__(self, points, coeffs):
        self.points = points
        self.coeffs = coeffs
        self.N = len(points)-1

    def __call__(self, x):
        from bisect import bisect_left
        i = bisect_left(self.points, x)
        if i == 0:
            if x == self.points[0]:
                # catch the case of being exactly on the lower limit
                i = 1
            else:
                raise Exception('Requested value is under the limits')
        if i > self.N:
            raise Exception('Requested value is above the limits')

        coeffs = self.coeffs[i-1]
        a, b = self.points[i-1], self.points[i]
        x = (2.0*x-a-b)/(b-a)
        return self.chebval(x, coeffs)

    @staticmethod
    def chebval(x, c):
        # copied from numpy's source, slightly optimized
        # https://github.com/numpy/numpy/blob/v1.13.0/numpy/polynomial/chebyshev.py#L1093-L1177
        # Will not support length-1 coefficient sets, must be 2 or more
        x2 = 2.*x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1*x2
        return c0 + c1*x



class CP_fluid_approximator(object):
    '''A class to hold (and calculate) approximations for certain aspects of
    CoolProp chemical's properties. This could apply equally well to REFPROP.
    '''
    __slots__ = ['CAS', 'Tmin', 'Tmax', 'Pmax', 'has_melting_line', 'Tc', 'Pc', 'Tt',
                 'omega', 'HEOS', 'DMOLAR_g', 'HMOLAR_g', 'SMOLAR_g',
                 'SPEED_OF_SOUND_g', 'CONDUCTIVITY_g', 'VISCOSITY_g',
                 'CPMOLAR_g', 'CVMOLAR_g', 'DMOLAR_l', 'HMOLAR_l', 'SMOLAR_l',
                 'SPEED_OF_SOUND_l', 'CONDUCTIVITY_l', 'VISCOSITY_l',
                 'CPMOLAR_l', 'CVMOLAR_l', 'CP0MOLAR']
    def calculate(self, T, prop, phase):
        assert phase in ['l', 'g']
        phase_key = '_g' if phase == 'g' else '_l'
        name = prop + phase_key
        try:
            return getattr(self, name)(T)
        except AttributeError:
            raise Exception('Given chemical does not have a fit available for '
                            'that property and phase')

    def validate_prop(self, prop, phase, evaluated_points=30):
        phase_key = '_g' if phase == 'g' else '_l'
        name = prop + phase_key
        if prop in ['CP0MOLAR']:
            name = prop
        pts = getattr(self, name).points
        predictor = getattr(self, name)
        for i in range(len(pts)-1):
            Ts = np.linspace(pts[i], pts[i+1], evaluated_points)
#            print(Ts[0], Ts[-1])
            prop_approx  = [predictor(T) for T in Ts]
            prop_calc = [CoolProp_T_dependent_property(T, self.CAS, prop, phase) for T in Ts]
#            print(prop_approx)
#            print(prop_calc)
            # The approximators do give differences at the very low value side
            # so we need atol=1E-9
#            print(prop, self.CAS, prop_approx[0], prop_calc[0])

            try:
                assert_close1d(prop_approx, prop_calc, rtol=1E-7, atol=1E-9)
            except:
                '''There are several cases that assert_allclose doesn't deal
                with well for some reason. We could increase rtol, but instead
                the relative errors are used here to check everything is as desidred.

                Some example errors this won't trip on but assert_allclose does
                are:

                1.00014278827e-08
                1.62767956613e-06
                -0.0
                -1.63895899641e-16
                -4.93284549625e-15
                '''
                prop_calc = np.array(prop_calc)
                prop_approx = np.array(prop_approx)
                errs = abs((prop_calc-prop_approx)/prop_calc)
                try:
                    assert max(errs) < 2E-6
                except:
                    print('%s %s failed with mean relative error %s and maximum relative error %s' %(self.CAS, prop, str(np.mean(errs)), str(max(errs))))



#



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

if has_CoolProp and 0:
    folder = os.path.join(os.path.dirname(__file__), 'Misc')

    f = open(os.path.join(folder, 'CoolProp vapor properties fits.json'), 'r')
    vapor_properties = json.load(f)
    f.close()

    f = open(os.path.join(folder, 'CoolProp CP0MOLAR fits.json'), 'r')
    idea_gas_heat_capacity = json.load(f)
    f.close()

    CP_approximators = {}

    for CAS in coolprop_dict:
        obj = CP_fluid_approximator()
        CP_approximators[CAS] = obj
        obj.CAS = CAS
        HEOS = AbstractState("HEOS", CAS)

        obj.Tmin = HEOS.Tmin()
        obj.Tmax = HEOS.Tmax()
        obj.Pmax = HEOS.pmax()
        obj.has_melting_line = HEOS.has_melting_line()
        obj.Tc = HEOS.T_critical()
        obj.Pc = HEOS.p_critical(),
        obj.Tt = HEOS.Ttriple()
        obj.omega = HEOS.acentric_factor()


        if CAS in vapor_properties:
            for key, value in vapor_properties[CAS].items():
                chebcoeffs, limits = value
                limits = [limits[0][0]] + [i[1] for i in limits]
                approximator = MultiCheb1D(limits, chebcoeffs)
                setattr(obj, key+'_g', approximator)

        if CAS in idea_gas_heat_capacity:
            chebcoeffs, Tmin, Tmax = idea_gas_heat_capacity[CAS]['CP0MOLAR']
            approximator = MultiCheb1D([Tmin, Tmax], chebcoeffs)
            setattr(obj, 'CP0MOLAR', approximator)

#            obj.validate_prop('CP0MOLAR', 'g')


def CoolProp_T_dependent_property_approximation(T, CASRN, prop, phase):
    pass


