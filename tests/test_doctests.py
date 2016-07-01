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


from thermo import elements
from thermo import combustion
from thermo import critical
from thermo import triple
from thermo import pr
from thermo import virial
from thermo import activity
from thermo import vapor_pressure
from thermo import viscosity
from thermo import reaction
from thermo import safety
from thermo import dipole
from thermo import thermal_conductivity
from thermo import phase_change
from thermo import environment
from thermo import identifiers
from thermo import permittivity
from thermo import volume
from thermo import heat_capacity
from thermo import solubility
from thermo import utils
from thermo import lennard_jones
from thermo import law
from thermo import refractivity
from thermo import electrochem
from thermo import interface

from math import exp, log
import os

import warnings

from scipy.optimize import fsolve
import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt

from thermo.identifiers import CASfromAny
#warnings.simplefilter("always") #error for error
warnings.simplefilter("ignore") #error for error


if __name__ == '__main__':
    print('Testing Begins here:')
    import doctest
    doctest.testmod(critical)
    doctest.testmod(triple)
    doctest.testmod(pr)
    doctest.testmod(virial)
#    doctest.testmod(adsorption)
#    doctest.testmod(activity)
    doctest.testmod(interface)
    doctest.testmod(vapor_pressure)
    doctest.testmod(viscosity)
    doctest.testmod(reaction)
    doctest.testmod(safety)
    doctest.testmod(dipole)
    doctest.testmod(thermal_conductivity)
    doctest.testmod(phase_change)
    doctest.testmod(environment)
    doctest.testmod(identifiers)
    doctest.testmod(permittivity)
    doctest.testmod(volume)
    doctest.testmod(heat_capacity)
    doctest.testmod(solubility)
    doctest.testmod(utils)
    doctest.testmod(lennard_jones)
#    doctest.testmod(law)
    doctest.testmod(refractivity)
    doctest.testmod(elements)
    doctest.testmod(combustion)
#    doctest.testmod(electrochem)



































#'''Problem from Hwang, In-Chan, Kyu-Jin Han, and So-Jin Park. "Isothermal Binary
#and Ternary VLE for the Mixtures of Propyl Vinyl Ether + Ethanol + Isooctane
#at 323.15 K and VE at 293.15 K."
#Journal of Chemical & Engineering Data 52, no. 3 (May 1, 2007): 1118-22. doi:10.1021/je700094y.
##
##Chemicals:  Propyl Vinyl Ether, Ethanol
##'''
#PVE_ABC=[(11.8597),1640.5297,-111.4676] #kPa, Kelvin
#T =  322.72 #323.15 #Kelvin
#Ps = [34.81, 38.64, 40.33, 43.85, 49.03, 53.51, 57.23, 60.53, 61.86, 63.41, 64.30, 65.39, 66.46, 67.19, 67.75, 68.03, 67.93, 67.43, 66.70, 65.86, 64.61, 63.34]
#X1_PVE = [0.0201,  0.0398,  0.0592,  0.0797,  0.1193,  0.1688,  0.2290,  0.2898,  0.3490,  0.4086,  0.4686,  0.5294,  0.5894,  0.6512,  0.7088,  0.7700,  0.8287,  0.8778,  0.9195,  0.9416,  0.9601,  0.9784]
#Y1_PVE = [0.1681, 0.2625, 0.2997, 0.3702, 0.4554, 0.5189, 0.5670, 0.6081, 0.6253, 0.6476, 0.6621, 0.6837, 0.7087, 0.7311, 0.7562, 0.7799, 0.8009, 0.8297, 0.8549, 0.8772, 0.9087, 0.9397]
##print T, Ps, X1_PVE, Y1_PVE
#
#activity.fit_NRTL2(Ps, Ts, X1_PVE, Y1_PVE)



#print EINCES('Radon')

#from scipy import integrate
#print PR.PR_density(300, 101325, ID='helium-4', phase='gas')*.025/60.0



#print PR.PR_density(298, 101E5, ID='nitrogen')/ PR.PR_density(298, 1E5, ID='nitrogen')


#    if k == None:
#        print i
#        count += 1

#print count