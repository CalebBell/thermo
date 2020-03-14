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

import os
import numpy as np
import pandas as pd
from enum import Enum
import csv
from math import pow, log, sqrt
from scipy.optimize import newton
from numpy import isclose

class RegionPoint(object):
    def __init__(self, i=None, j=None, n=None):    
        if i is not None:
            self.i = i
        if j is not None:
            self.j = j
        if n is not None:
            self.n = n

    i = float('nan')
    j = float('nan')
    n = float('nan')


def __GetRegionPoints(fileName): 
    folder = os.path.join(os.path.dirname(__file__), 'Steam Properties')
    iterator = csv.reader(open(os.path.join(folder, fileName)))
    headers = next(iterator)
    i_getter = lambda row : float('NaN')
    j_getter = lambda row : float('NaN')
    n_getter = lambda row : float('NaN')
    for i, header in enumerate(headers):
        header = header.lower()
        getter = lambda row, idx = i : float(row[idx])
        if 'i' in header:
            i_getter = getter
        elif 'j' in header:
            j_getter = getter
        elif 'n' in header:
            n_getter = getter

    return [RegionPoint(i_getter(row), j_getter(row), n_getter(row)) for row in iterator]

__region1and4Points = __GetRegionPoints('Region1and4.csv')
__region2IdealPoints = __GetRegionPoints('Region2Ideal.csv')
__region2ResiduaPoints = __GetRegionPoints('Region2Residual.csv')
__region3Points = __GetRegionPoints('Region3.csv')
__nRegion4Points = __GetRegionPoints('nRegion4.csv')
__region5IdealPoints = __GetRegionPoints('Region5Ideal.csv')
__region5ResiduaPoints = __GetRegionPoints('Region5Residual.csv')
__nBoundary34Points = __GetRegionPoints('nBoundary34.csv')
# In K
__critical_temperature = 647.096
# In Pa
__critical_pressure = 22.06e6
# In J/(kg * K)
__water_gas_constant = 461.526

class __SteamEquationRegion(Enum):
    OutOfRange = 0
    Region1 = 1
    Region2 = 2
    Region3 = 3
    Region4 = 4
    Region5 = 5

class PhaseRegion(Enum):
    OutOfRange = 0

    SupercriticalFluid = 1
    '''
    Temperature and pressure is above the critical point
    '''

    Gas = 2
    '''
    Pressure is less than both the sublimation and vaporization curve and is below the critical temperature
    '''

    Vapor = 3
    '''
    Pressure is above the vaporization curve and the temperature is greater than the fusion curve and less than the critical temperature
    '''

    Liquid = 4
    '''
    Pressure is above the sublimation curve and temperature is less than the fusion curve
    '''

    Solid = 5
    SolidLiquid = 6           
    LiquidVapor = 7           
    SolidVapor = 8         
    SolidLiquidVapor = 9


class PhaseInfo(object):
    def __init__(self, _phase_region = PhaseRegion.OutOfRange, _vapor = 0, _liquid = 0, _solid = 0):    
        self._phase_region = _phase_region
        self._vapor = _vapor
        self._liquid = _liquid
        self._solid = _solid

    def get_phase_region(self):
        ''' 
        '''
        return self._phase_region

    def get_vapor(self):
        ''' 
        Vapor fraction
        '''
        return self._vapor

    def get_liquid(self):
        ''' 
        Liquid fraction
        '''
        return self._liquid

    def get_solid(self):
        ''' 
        Solid fraction
        '''
        return self._solid


class PTVEntry(object):
    def __init__(self, 
            _temperature = None, 
            _pressure = None, 
            _phase_info = None,
            _internal_energy = None,
            _enthalpy = None,
            _entropy = None,
            _isochoric_heat_capacity = None,
            _isobaric_heat_capacity = None,
            _speed_of_sound = None,
            _density = None):    
        self._temperature = _temperature
        self._pressure = _pressure
        self._phase_info = _phase_info
        self._internal_energy = _internal_energy
        self._enthalpy = _enthalpy
        self._entropy = _entropy
        self._isochoric_heat_capacity = _isochoric_heat_capacity
        self._isobaric_heat_capacity = _isobaric_heat_capacity
        self._speed_of_sound = _speed_of_sound
        self._density = _density

    def get_temperature(self):
        ''' 
        Temperature, K
        '''
        return self._temperature

    def get_pressure(self):
        ''' 
        Pressure, Pa
        '''
        return self._pressure

    def get_phase_info(self):
        ''' 
        '''
        return self._phase_info

    def get_specific_volume(self):
        '''
        Specific Volume, m3/kg 
        '''
        return 1.0 / self._density 

    def get_internal_energy(self):
        '''
        Internal Energy, J/kg 
        '''
        return self._internal_energy

    def get_enthalpy(self):
        '''
        Enthalpy, J/kg 
        '''
        return self._enthalpy

    def get_entropy(self):
        '''
        Entropy, J/kg 
        '''
        return self._entropy

    def get_isochoric_heat_capacity(self):
        '''
        Cv, Heat Capacity at constant volume (J/(kg*K))
        '''
        return self._isochoric_heat_capacity

    def get_isobaric_heat_capacity(self):
        '''
        Cp, Heat Capacity at constant pressure (J/(kg*K))
        '''
        return self._isobaric_heat_capacity

    def get_speed_of_sound(self):
        '''
        Speed of Sound, m/s
        '''
        return self._speed_of_sound

    def get_density(self):
        '''
        Density, kg/m3
        '''
        return self._density

class __SpecificRegionPoint(object):
    
    def __init__(self, _temperature = 0, _pressure = 0, _tau = 0, _pi = 0, _gamma = 0,
     _gamma_pi = 0, _gamma_pi_pi = 0, _gamma_tau = 0, _gamma_tau_tau = 0, _gamma_pi_tau = 0):    
        self._temperature = _temperature
        self._pressure = _pressure
        self._tau = _tau
        self._pi = _pi
        self._gamma = _gamma
        self._gamma_pi = _gamma_pi
        self._gamma_pi_pi = _gamma_pi_pi
        self._gamma_tau = _gamma_tau
        self._gamma_tau_tau = _gamma_tau_tau
        self._gamma_pi_tau = _gamma_pi_tau

    def get_temperature(self):
        ''' 
        Temperature, In Kelvin
        '''
        return self._temperature
    
    def get_pressure(self):
        ''' 
        Pressure, In Pascals
        '''
        return self._pressure

    def get_tau(self):
        ''' 
        Inverse reduced temperature, T*/T
        '''
        return self._tau

    def get_pi(self):
        ''' 
        Reduced pressure, p/p*
        '''
        return self._pi
    
    def get_gamma(self):
        ''' 
        Dimensionless Gibbs free energy, g/(RT)
        '''
        return self._gamma

    def get_gamma_pi(self):
        ''' 
        Derivative of gamma with respect to pi
        '''
        return self._gamma_pi

    def get_gamma_pi_pi(self):
        ''' 
        Derivative of gamma with respect to pi with respect to pi
        '''
        return self._gamma_pi_pi

    def get_gamma_tau(self):
        ''' 
        Derivative of gamma with respect to tau
        '''
        return self._gamma_tau
    
    def get_gamma_tau_tau(self):
        ''' 
        Derivative of gamma with respect to tau with respect to tau
        '''
        return self._gamma_tau_tau

    def get_gamma_pi_tau(self):
        ''' 
        Derivative of gamma with respect to pi with respect to tau
        '''
        return self._gamma_pi_tau

class SteamTableQuery(object):
    def __init__(self, 
     _temperature = None, 
     _pressure  =  None, 
     _enthalpy  =  None, 
     _entropy  =  None, 
     _phase_region = None,
     _is_saturated = None
     ):
        self._pressure = _pressure
        self._temperature = _temperature
        self._enthalpy = _enthalpy
        self._entropy = _entropy
        self._phase_region = _phase_region
        self._is_saturated = _is_saturated

    def get_temperature(self):
        ''' 
        Temperature, In Kelvin
        '''
        return self._temperature

    def has_temperature(self): 
        return self.get_temperature() is not None
    
    def get_pressure(self):
        ''' 
        Pressure, In Pascals
        '''
        return self._pressure

    def has_pressure(self): 
        return self.get_pressure() is not None

    def get_enthalpy(self):
        ''' 
        enthalpy, In kJ / kg
        '''
        return self._enthalpy

    def has_enthalpy(self): 
        return self.get_enthalpy() is not None

    def get_entropy(self):
        ''' 
        Entropy, In kJ / (kg * K)
        '''
        return self._entropy

    def has_entropy(self): 
        return self.get_entropy() is not None

    def get_phase_region(self):
        ''' 
        '''
        return self._phase_region

    def has_phase_region(self): 
        return self.get_phase_region() is not None
    
    def get_is_saturated(self):
        ''' 
        '''
        return self._is_saturated

    def has_is_saturated(self): 
        return self.get_is_saturated() is not None


class __ValueRange(object):
    def __init__(self, min=None, max=None):    
        if min is not None:
            self.min = min
        if max is not None:
            self.max = max
    min = float('nan')
    max = float('nan')
    def isWithinRange(self, value):
        return self.min <= value and value <= self.max

class __PT_Point(object):
    def __init__(self, pressure=None, temperature=None):    
        if pressure is not None:
            self.pressure = pressure
        if temperature is not None:
            self.temperature = temperature
    pressure = float('nan')
    temperature = float('nan')

def __determine_temperature_range(pressure):
    '''
    Temperature is in Kelvin
    Pressure is in Pascals
    '''
    if pressure > 50e6:
        return __ValueRange(min=273.15, max=800 + 273.15)
    return __ValueRange(min=273.15, max=2000 + 273.15)

def __determine_pressure_range(temperature):
    '''
    Temperature is in Kelvin
    Pressure is in Pascals
    '''
    if temperature > 800 + 273.15:
        return __ValueRange(min=0, max=50e6)
    return __ValueRange(min=0, max=100e6)

def __get_sat_pressure(temperature):
    '''
    Temperature is in Kelvin
    Pressure is in Pascals
    '''
    if (temperature >= __critical_temperature):
        return (False, None, 'Temperature is greater than the critical temperature')
    
    sat_temp_ratio = temperature / 1
    theta = sat_temp_ratio + (__nRegion4Points[8].n / (sat_temp_ratio - __nRegion4Points[9].n))
    A = pow(theta, 2) + __nRegion4Points[0].n * theta + __nRegion4Points[1].n
    B = __nRegion4Points[2].n * pow(theta, 2) + __nRegion4Points[3].n * theta + __nRegion4Points[4].n
    C = __nRegion4Points[5].n * pow(theta, 2) + __nRegion4Points[6].n * theta + __nRegion4Points[7].n
    pressure = pow((2 * C) / (-B + pow(pow(B, 2) - 4 * A * C, 0.5)), 4) * 1e6
    return (True, pressure, None)

def __get_sat_temperature(pressure):
    '''
    Temperature is in Kelvin
    Pressure is in Pascals
    '''
    if (pressure >= __critical_pressure):
        return (False, None, 'Pressure is greater than the critical pressure')
    beta = pow(pressure / 1e6, 0.25)
    E = pow(beta, 2) + __nRegion4Points[2].n * beta + __nRegion4Points[5].n
    F = __nRegion4Points[0].n * pow(beta, 2) + __nRegion4Points[3].n * beta + __nRegion4Points[6].n
    G = __nRegion4Points[1].n * pow(beta, 2) + __nRegion4Points[4].n * beta + __nRegion4Points[7].n
    D = (2 * G) / (-F - pow(pow(F, 2) - 4 * E * G, 0.5))
    pressure = (__nRegion4Points[9].n + D - pow(pow(__nRegion4Points[9].n + D, 2) - 4 * (__nRegion4Points[8].n + __nRegion4Points[9].n * D),0.5)) / 2
    return (True, pressure, None)

def __get_boundary_34_pressure(temperature):
    '''
    Temperature is in Kelvin
    Pressure is in Pascals
    '''
    if (temperature < __critical_temperature):
        return (False, None, 'Temperature is less than the critical temperature')
        
    theta = temperature / 1
    pressure = (__nBoundary34Points[0].n + __nBoundary34Points[1].n * theta + __nBoundary34Points[2].n * pow(theta , 2)) * 1e6
    return (True, pressure, None)

def __temperature_is_within_range(steam_table_query):
    if (not steam_table_query.has_pressure() or not steam_table_query.has_temperature()):
        return True
    temperature_range = __determine_temperature_range(steam_table_query.get_pressure())
    return temperature_range.isWithinRange(steam_table_query.get_temperature())

def __pressure_is_within_range(steam_table_query):
    if (not steam_table_query.has_pressure() or not steam_table_query.has_temperature()):
        return True
    pressure_range = __determine_pressure_range(steam_table_query.get_temperature())
    return pressure_range.isWithinRange(steam_table_query.get_pressure())

def __is_saturation_point_query(steam_table_query):
    has_temperature = steam_table_query.has_temperature()
    has_pressure = steam_table_query.has_pressure()
    has_phase_region = steam_table_query.get_phase_region() is not None
    has_saturated_flag = steam_table_query.get_is_saturated() is not None
    return (has_temperature or has_pressure) and has_phase_region and has_saturated_flag and steam_table_query.get_is_saturated

def __perform_saturation_point_query(steam_table_query):
    temperature = steam_table_query.get_temperature()
    pressure = steam_table_query.get_pressure()

    if (steam_table_query.has_pressure()):
        (_, temperature, error_msg) = __get_sat_temperature(pressure)
    elif (steam_table_query.has_temperature()):
        (_, pressure, error_msg) = __get_sat_pressure(temperature)
    else :
        error_msg = 'Must have pressure or temperature'
    
    equation_region = __SteamEquationRegion.OutOfRange
    phase_region = steam_table_query.get_phase_region()

    if (error_msg is None and phase_region == PhaseRegion.Liquid):
        equation_region = __SteamEquationRegion.Region1
    elif (error_msg is None and phase_region == PhaseRegion.Vapor):
        equation_region = __SteamEquationRegion.Region2
    else :
        error_msg = 'No saturated points can be found for following region : ' + str(phase_region)
    
    return (equation_region, __PT_Point(pressure=pressure, temperature=temperature), error_msg)

def __perform_non_saturation_point_query(steam_table_query):
    temperature = steam_table_query.get_temperature()
    pressure = steam_table_query.get_pressure()
    equation_region = __SteamEquationRegion.OutOfRange
    (found_sat_pressure, satPressure, _) = __get_sat_pressure(temperature)
    (found_boundary_pressure, boundaryPressure, _) = __get_boundary_34_pressure(temperature)

    error_msg = None
    if (temperature > (273.15 + 800)):
        equation_region = __SteamEquationRegion.Region5
    elif (temperature > (600 + 273.15)):
        equation_region = __SteamEquationRegion.Region2
    elif (found_sat_pressure):
        if (satPressure == pressure):
            equation_region = __SteamEquationRegion.Region4
        elif (satPressure > pressure):
            equation_region = __SteamEquationRegion.Region2
        else :
            equation_region = __SteamEquationRegion.Region1
    elif (found_boundary_pressure):
        if (boundaryPressure > pressure):
            equation_region = __SteamEquationRegion.Region2
        else :
            equation_region = __SteamEquationRegion.Region3
    else :
        error_msg = 'Out of range'

    return (equation_region, __PT_Point(pressure=pressure, temperature=temperature), error_msg)

def __resolve_region(steam_table_query):
    if (not __temperature_is_within_range(steam_table_query)):
        return (__SteamEquationRegion.OutOfRange, None, 'Temperature is out of range')

    if (not __pressure_is_within_range(steam_table_query)):
        return (__SteamEquationRegion.OutOfRange, None, 'Pressure is out of range')

    if (__is_saturation_point_query(steam_table_query)):
        return __perform_saturation_point_query(steam_table_query)

    if (steam_table_query.has_temperature() and steam_table_query.has_pressure()):
        return __perform_non_saturation_point_query(steam_table_query)

    return (__SteamEquationRegion.OutOfRange, None, 'Not enough constraints')

def __create_region_1245_entry(specificRegionPoint, phase_info):
    temperature = specificRegionPoint.get_temperature()
    pressure = specificRegionPoint.get_pressure()
    pi = specificRegionPoint.get_pi()
    tau = specificRegionPoint.get_tau()
    gamma = specificRegionPoint.get_gamma()
    gamma_pi = specificRegionPoint.get_gamma_pi()
    gamma_pi_pi = specificRegionPoint.get_gamma_pi_pi()
    gamma_tau = specificRegionPoint.get_gamma_tau()
    gamma_tau_tau = specificRegionPoint.get_gamma_tau_tau()
    gamma_pi_tau = specificRegionPoint.get_gamma_pi_tau()
    
    return (True, PTVEntry(
        _temperature=temperature,
        _pressure=pressure,
        _phase_info=phase_info,
        _internal_energy=__water_gas_constant * temperature * (tau * gamma_tau - pi * gamma_pi),
        _enthalpy=__water_gas_constant * temperature * tau * gamma_tau,
        _entropy=__water_gas_constant * (tau * gamma_tau - gamma),
        _isochoric_heat_capacity=__water_gas_constant * (-pow(-tau, 2) * gamma_tau_tau + pow(gamma_pi - tau * gamma_pi_tau, 2) / gamma_pi_pi),
        _isobaric_heat_capacity=__water_gas_constant * -pow(-tau, 2) * gamma_tau_tau,
        _speed_of_sound=sqrt(__water_gas_constant * temperature *
            ((pow(gamma_pi, 2)) / ((pow(gamma_pi - tau * gamma_pi_tau, 2) / (pow(tau, 2) * gamma_tau_tau)) - gamma_pi_pi))),
        _density=1.0 / (pi * (gamma_pi * __water_gas_constant * temperature) / pressure)
    ), None)     

def __gibbs_method(pt_point):
    pressure = pt_point.pressure 
    temperature = pt_point.temperature 
    pi = pressure / 16.53e6
    tau = 1386.0 / temperature
    gamma = 0
    gamma_pi = 0
    gamma_pi_pi = 0
    gamma_tau = 0
    gamma_tau_tau = 0
    gamma_pi_tau = 0
    phaseInfo = PhaseInfo(PhaseRegion.Liquid, 0, 1, 0)
    
    for regionPoint in __region1and4Points:
        gamma += regionPoint.n * pow(7.1 - pi, regionPoint.i) * pow(tau - 1.222, regionPoint.j)
        gamma_pi += -regionPoint.n * regionPoint.i * pow(7.1 - pi, regionPoint.i - 1) * pow(tau - 1.222, regionPoint.j)
        gamma_pi_pi += regionPoint.n * regionPoint.i * (regionPoint.i - 1) * pow(7.1 - pi, regionPoint.i - 2) * pow(tau - 1.222, regionPoint.j)
        gamma_tau += regionPoint.n * regionPoint.j * pow(7.1 - pi, regionPoint.i) * pow(tau - 1.222, regionPoint.j - 1)
        gamma_tau_tau += regionPoint.n * regionPoint.j * (regionPoint.j - 1) * pow(7.1 - pi, regionPoint.i) * pow(tau - 1.222, regionPoint.j - 2)
        gamma_pi_tau += -regionPoint.n * regionPoint.i * regionPoint.j * pow(7.1 - pi, regionPoint.i - 1) * pow(tau - 1.222, regionPoint.j - 1)
    
    specificRegionPoint = __SpecificRegionPoint(
        _temperature=temperature,
        _pressure=pressure,
        _tau=tau,
        _pi=pi,
        _gamma=gamma,
        _gamma_pi=gamma_pi,
        _gamma_pi_pi=gamma_pi_pi,
        _gamma_tau=gamma_tau,
        _gamma_tau_tau=gamma_tau_tau,
        _gamma_pi_tau=gamma_pi_tau
    )

    return __create_region_1245_entry(specificRegionPoint, phaseInfo)

def __create_phase_info(pt_point):
    region = PhaseRegion.Vapor
    vaporFrac = 1
    if (pt_point.temperature > __critical_temperature):
        if (pt_point.pressure > __critical_pressure):
            region = PhaseRegion.SupercriticalFluid
        else :
            region = PhaseRegion.Gas
        vaporFrac = 0
    return PhaseInfo(region, vaporFrac, 0, 0)

def __vapor_method(tau, 
            tau_shift, 
            pt_point, 
            idealPoints, 
            residualPoints):
    pressure = pt_point.pressure
    temperature = pt_point.temperature
    pi = pressure / 1.0e6
    gamma = log(pi)
    gamma_pi = 1.0 / pi
    gamma_pi_pi = -1.0 / pow(pi, 2)
    gamma_tau = 0
    gamma_tau_tau = 0
    gamma_pi_tau = 0
    phaseInfo = __create_phase_info(pt_point)
    
    for regionPoint in idealPoints:
        gamma += regionPoint.n * pow(tau, regionPoint.j)
        gamma_tau += regionPoint.n * regionPoint.j * pow(tau, regionPoint.j - 1)
        gamma_tau_tau += regionPoint.n * regionPoint.j * (regionPoint.j - 1) * pow(tau, regionPoint.j - 2)

    for regionPoint in residualPoints:
        gamma += regionPoint.n * pow(pi, regionPoint.i) * pow(tau - tau_shift, regionPoint.j)
        gamma_pi += regionPoint.n * regionPoint.i * pow(pi, regionPoint.i - 1) * pow(tau - tau_shift, regionPoint.j)
        gamma_pi_pi += regionPoint.n * regionPoint.i * (regionPoint.i - 1) * pow(pi, regionPoint.i - 2) * pow(tau - tau_shift, regionPoint.j)
        gamma_tau += regionPoint.n * pow(pi, regionPoint.i) * regionPoint.j * pow(tau - tau_shift, regionPoint.j - 1)
        gamma_tau_tau += regionPoint.n * pow(pi, regionPoint.i) * regionPoint.j * (regionPoint.j - 1) * pow(tau - tau_shift, regionPoint.j - 2)
        gamma_pi_tau += regionPoint.n * regionPoint.i * pow(pi, regionPoint.i - 1) * regionPoint.j * pow(tau - tau_shift, regionPoint.j - 1)

    specificRegionPoint = __SpecificRegionPoint(
        _temperature=temperature,
        _pressure=pressure,
        _tau=tau,
        _pi=pi,
        _gamma=gamma,
        _gamma_pi=gamma_pi,
        _gamma_pi_pi=gamma_pi_pi,
        _gamma_tau=gamma_tau,
        _gamma_tau_tau=gamma_tau_tau,
        _gamma_pi_tau=gamma_pi_tau
    )

    return __create_region_1245_entry(specificRegionPoint, phaseInfo)

def __region3_density(pt_point, density):
    region3_iter = iter(__region3Points)
    n1 = next(region3_iter).n
    delta = density / 322.0
    temperature = pt_point.temperature
    tau = 647.096 / temperature
    phi = n1 * log(delta)
    phi_delta = n1 / delta
    phi_delta_delta = -n1 / pow(delta, 2)
    phi_tau = 0
    phi_tau_tau = 0
    phi_delta_tau = 0

    for regionPoint in region3_iter:
        phi += regionPoint.n * pow(delta, regionPoint.i) * pow(tau, regionPoint.j)
        phi_delta += regionPoint.n * regionPoint.i * pow(delta, regionPoint.i - 1) * pow(tau, regionPoint.j)
        phi_delta_delta += regionPoint.n * regionPoint.i * (regionPoint.i - 1) * pow(delta, regionPoint.i - 2) * pow(tau, regionPoint.j)
        phi_tau += regionPoint.n * pow(delta, regionPoint.i) * regionPoint.j * pow(tau, regionPoint.j - 1)
        phi_tau_tau += regionPoint.n * pow(delta, regionPoint.i) * regionPoint.j * (regionPoint.j - 1) * pow(tau, regionPoint.j - 2)
        phi_delta_tau += regionPoint.n * regionPoint.i * pow(delta, regionPoint.i - 1) * regionPoint.j * pow(tau, regionPoint.j - 1)

    pressure = phi_delta * delta * density * __water_gas_constant * temperature

    phase_info = PhaseInfo(PhaseRegion.SupercriticalFluid, 0, 0, 0)
    internal_energy = tau * phi_tau * __water_gas_constant * temperature
    enthalpy = (tau * phi_tau + delta * phi_delta) * __water_gas_constant * temperature
    entropy = (tau * phi_tau - phi) * __water_gas_constant
    isochoric_heat_capacity = -pow(tau, 2) * phi_tau_tau * __water_gas_constant
    isobaric_heat_capacity = (-pow(tau, 2) * phi_tau_tau
        + pow(delta * phi_delta - delta * tau * phi_delta_tau, 2) 
        / (2 * delta * phi_delta + pow(delta, 2) * phi_delta_delta)) * __water_gas_constant
    
    try:
        speed_of_sound = sqrt((2 * delta * phi_delta + pow(delta, 2) * phi_delta_delta -
            pow(delta * phi_delta - delta * tau * phi_delta_tau, 2) 
            / (pow(tau, 2) * phi_tau_tau)) * __water_gas_constant * temperature)
    except:
        speed_of_sound = float('NaN')

    
    return PTVEntry(
        _temperature=temperature,
        _pressure=pressure,
        _phase_info=phase_info,
        _internal_energy=internal_energy,
        _enthalpy=enthalpy,
        _entropy=entropy,
        _isochoric_heat_capacity=isochoric_heat_capacity,
        _isobaric_heat_capacity=isobaric_heat_capacity,
        _speed_of_sound=speed_of_sound,
        _density=density
    )          

def __region3_method(pt_point):
    result = newton(func=lambda x:__region3_density(pt_point, x).get_pressure() - pt_point.pressure, x0=1)
    return (True, __region3_density(pt_point, result), None)

def __is_within_tolerance(actualValue, queryParameter):
    if (queryParameter is None):
        return True
    return isclose(actualValue, queryParameter)

def __entry_is_within_query_tol(entry, steam_table_query):
    if (steam_table_query.has_phase_region() and entry.get_phase_info().get_phase_region() != steam_table_query.get_phase_region()):
        return False

    pressure_is_okay = __is_within_tolerance(entry.get_pressure(), steam_table_query.get_pressure())
    temperature_is_okay = __is_within_tolerance(entry.get_temperature(), steam_table_query.get_temperature())
    enthalpy_is_okay = __is_within_tolerance(entry.get_enthalpy(), steam_table_query.get_enthalpy())
    entropy_is_okay = __is_within_tolerance(entry.get_entropy(), steam_table_query.get_entropy())

    return pressure_is_okay and temperature_is_okay and entropy_is_okay and enthalpy_is_okay

def __interpolate_entry_property(liquid_entry, vapor_entry, liqFrac, get_property):
    vaporFrac = 1 - liqFrac
    return (get_property(liquid_entry) * liqFrac) + (get_property(vapor_entry) * vaporFrac)

def __interpolate_entry(liquid_entry, vapor_entry, liqFrac):
    phase_info = PhaseInfo(PhaseRegion.LiquidVapor, _vapor=(1 - liqFrac), _liquid=liqFrac, _solid=0)
    temperature = __interpolate_entry_property(liquid_entry, vapor_entry, liqFrac, lambda entry: entry.get_temperature())
    pressure = __interpolate_entry_property(liquid_entry, vapor_entry, liqFrac, lambda entry: entry.get_pressure())
    internal_energy = __interpolate_entry_property(liquid_entry, vapor_entry, liqFrac, lambda entry: entry.get_internal_energy())
    enthalpy = __interpolate_entry_property(liquid_entry, vapor_entry, liqFrac, lambda entry: entry.get_enthalpy())
    entropy = __interpolate_entry_property(liquid_entry, vapor_entry, liqFrac, lambda entry: entry.get_entropy())
    isochoric_heat_capacity = __interpolate_entry_property(liquid_entry, vapor_entry, liqFrac, lambda entry: entry.get_isochoric_heat_capacity())
    isobaric_heat_capacity = __interpolate_entry_property(liquid_entry, vapor_entry, liqFrac, lambda entry: entry.get_isobaric_heat_capacity())
    speed_of_sound = __interpolate_entry_property(liquid_entry, vapor_entry, liqFrac, lambda entry: entry.get_speed_of_sound())
    density = __interpolate_entry_property(liquid_entry, vapor_entry, liqFrac, lambda entry: entry.get_density())
    return PTVEntry(
        _temperature=temperature,
        _pressure=pressure,
        _phase_info=phase_info,
        _internal_energy=internal_energy,
        _enthalpy=enthalpy,
        _entropy=entropy,
        _isochoric_heat_capacity=isochoric_heat_capacity,
        _isobaric_heat_capacity=isobaric_heat_capacity,
        _speed_of_sound=speed_of_sound,
        _density=density
    ) 

def __get_prop_at_t_and_p(pressure, temperature, get_prop_value):
    query = SteamTableQuery(_pressure=pressure, _temperature=temperature)
    (passed, entry, _) = __perform_steam_table_query(query)
    if (not passed):
        return float('NaN')
    return get_prop_value(entry)

def __resolve_non_pressure_temperature_property(pressure, target_value, get_prop_value):
    liquid_query = SteamTableQuery(_pressure=pressure, _phase_region=PhaseRegion.Liquid, _is_saturated=True)
    vapor_query = SteamTableQuery(_pressure=pressure, _phase_region=PhaseRegion.Vapor, _is_saturated=True)
    (liquid_success, liquid_entry, _) = __perform_steam_table_query(liquid_query)
    (vapor_success, vapor_entry, _) = __perform_steam_table_query(vapor_query)
    is_success = False
    entry = None
    err_msg = None
    if (liquid_success and vapor_success and get_prop_value(vapor_entry) >= target_value and get_prop_value(liquid_entry) <= target_value):
        is_success = True
        liqFrac = (get_prop_value(vapor_entry) - target_value) / (get_prop_value(vapor_entry) - get_prop_value(liquid_entry))
        entry = __interpolate_entry(liquid_entry, vapor_entry, liqFrac)
    else :
        result = newton(func=lambda x:__get_prop_at_t_and_p(pressure, x, get_prop_value) - target_value, x0=300)
        query = SteamTableQuery(_pressure=pressure, _temperature=result)
        (is_success, entry, err_msg) = __perform_steam_table_query(query)
    
    return (is_success, entry, err_msg)

def __handle_out_of_range(steam_table_query, default_err_msg):
    is_success = False
    entry = None
    err_msg = default_err_msg
    pressure = steam_table_query.get_pressure()
    enthalpy = steam_table_query.get_enthalpy()
    entropy = steam_table_query.get_entropy()

    if (enthalpy is not None and pressure is not None):
        (is_success, entry, err_msg) = __resolve_non_pressure_temperature_property(pressure, enthalpy, lambda entry: entry.get_enthalpy())
    elif (entropy is not None and pressure is not None):
        (is_success, entry, err_msg) = __resolve_non_pressure_temperature_property(pressure, entropy, lambda entry: entry.get_entropy())
    return (is_success, entry, err_msg)

def __perform_steam_table_query(steam_table_query):
    (region, pt_point, err_msg) = __resolve_region(steam_table_query)
    
    if (region ==  __SteamEquationRegion.OutOfRange):
        (is_success, entry, err_msg) = __handle_out_of_range(steam_table_query, err_msg)
    elif (region ==  __SteamEquationRegion.Region1):
        (is_success, entry, err_msg) = __gibbs_method(pt_point)
    elif (region ==  __SteamEquationRegion.Region2):
        (is_success, entry, err_msg) =  __vapor_method(540.0 / pt_point.temperature, 0.5, pt_point, __region2IdealPoints, __region2ResiduaPoints)
    elif (region ==  __SteamEquationRegion.Region3):
        (is_success, entry, err_msg) =  __region3_method(pt_point)
    elif (region ==  __SteamEquationRegion.Region4):
        (is_success, entry, err_msg) =  __gibbs_method(pt_point)
    elif (region ==  __SteamEquationRegion.Region5):
        (is_success, entry, err_msg) =  __vapor_method(1000.0 / pt_point.temperature, 0, pt_point, __region5IdealPoints, __region5ResiduaPoints)
    else :
        is_success = False
        entry = None
        err_msg = 'Unknown Steam Region'
    
    if (is_success and not __entry_is_within_query_tol(entry, steam_table_query)):
        is_success = False
        entry = None
        err_msg = 'Could not find an entry which was within the given tolerance'

    return (is_success, entry, err_msg)

def get_steam_table_entry(
    temperature = None, 
    pressure  =  None, 
    enthalpy  =  None, 
    entropy  =  None, 
    phase_region = None,
    is_saturated = None):
    r'''Calculates steam properties given passed values using the IAPWS-97 standard for Industrial Formulation [1].

    Parameters
    ----------
    temperature : float
        Temperature [K]
    pressure : float
        Pressure [Pa]
    enthalpy : float
        Enthalpy [j / kg]
    entropy : float
        Entropy [j / (kg * K)]
    phase_region : PhaseRegion
        The phase of the entry
    is_saturated : bool
        Entry must be saturated

    Returns
    -------
    is_success : bool
        did the query execute without error?
    entry : PTVEntry
        the result of the query - Returns None if is_success = false
    error_message : string
        error message of query - Returns None if is_success = true

    Notes
    -----
    Original model is in terms of J/g/K. Note that the model is for predicting
    mass heat capacity, not molar heat capacity like most other methods!
    Integral was computed with SymPy.

    Examples
    --------
    >>> get_steam_table_entry(pressure=40e6, temperature=473.15)
    (True, <__main__.PTVEntry object at 0x7fb0b86add90>, None)

    References
    ----------
    [1] Wagner, W., et al. “The IAPWS Industrial Formulation 1997 for 
    the Thermodynamic Properties of Water and Steam.” 
    Journal of Engineering for Gas Turbines and Power, 
    vol. 122, no. 1, 2000, pp. 150–184.,
    doi:10.1115/1.483186.
    '''
    steam_table_query = SteamTableQuery(
        _temperature= temperature,
        _pressure= pressure,
        _enthalpy= enthalpy,
        _entropy= entropy,
        _phase_region= phase_region,
        _is_saturated= is_saturated
    )
    return __perform_steam_table_query(steam_table_query)


(actual_is_success, actual_entry, actual_error_message) = get_steam_table_entry(
    pressure=10e3, 
    temperature=None, 
    enthalpy=None, 
    entropy=6.6858e3, 
    phase_region=None, 
    is_saturated=None)
_ = 1
