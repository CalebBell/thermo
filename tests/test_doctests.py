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


from thermo import eos
from thermo import eos_mix
from thermo import elements
from thermo import combustion
from thermo import critical
from thermo import triple
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
    import doctest
    doctest.testmod(critical)
    doctest.testmod(triple)
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
    doctest.testmod(eos)
    doctest.testmod(eos_mix)

