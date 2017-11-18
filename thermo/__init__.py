# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from . import acentric
from . import activity
from . import chemical
from . import combustion
from . import critical
from . import coolprop
from . import dipole
from . import dippr
from . import datasheet
from . import electrochem
from . import elements
from . import environment
from . import eos
from . import eos_mix
from . import heat_capacity
from . import identifiers
from . import joback
from . import law
from . import lennard_jones
from . import miscdata
from . import mixture
from . import permittivity
from . import phase_change
from . import property_package
from . import reaction
from . import refractivity
from . import safety
from . import solubility
from . import stream
from . import interface
from . import thermal_conductivity
from . import triple
from . import unifac
from . import utils
from . import vapor_pressure
from . import virial
from . import viscosity
from . import volume


from .acentric import *
from .activity import *
from .chemical import *
from .combustion import *
from .critical import *
from .coolprop import *
from .dipole import *
from .dippr import *
from .datasheet import *
from .electrochem import *
from .elements import *
from .environment import *
from .eos import *
from .eos_mix import *
from .heat_capacity import *
from .joback import *
from .identifiers import *
from .law import *
from .lennard_jones import *
from .miscdata import *
from .mixture import *
from .permittivity import *
from .phase_change import *
from .property_package import *
from .reaction import *
from .refractivity import *
from .safety import *
from .solubility import *
from .stream import *
from .interface import *
from .thermal_conductivity import *
from .triple import *
from .unifac import *
from .utils import *
from .vapor_pressure import *
from .virial import *
from .viscosity import *
from .volume import *


__all__ = ['activity', 'chemical', 'combustion', 'critical',
 'dipole', 'electrochem', 'elements', 'environment', 'eos', 'eos_mix',
 'heat_capacity',  'identifiers', 'joback', 'law', 'lennard_jones',
 'miscdata',
 'permittivity', 'phase_change', 'property_package', 'reaction',
 'refractivity', 'safety', 'solubility', 'interface',
 'thermal_conductivity', 'triple', 'utils',
 'vapor_pressure', 'virial', 'viscosity', 'volume', 'acentric', 'coolprop', 
 'datasheet', 'dippr', 'unifac', 'stream', 'mixture']


__all__.extend(acentric.__all__)
__all__.extend(activity.__all__)
__all__.extend(chemical.__all__)
__all__.extend(combustion.__all__)
__all__.extend(critical.__all__)
__all__.extend(coolprop.__all__)
__all__.extend(dipole.__all__)
__all__.extend(dippr.__all__)
__all__.extend(datasheet.__all__)
__all__.extend(electrochem.__all__)
__all__.extend(elements.__all__)
__all__.extend(environment.__all__)
__all__.extend(eos.__all__)
__all__.extend(eos_mix.__all__)
__all__.extend(heat_capacity.__all__)
__all__.extend(identifiers.__all__)
__all__.extend(joback.__all__)
__all__.extend(law.__all__)
__all__.extend(lennard_jones.__all__)
__all__.extend(miscdata.__all__)
__all__.extend(mixture.__all__)
__all__.extend(permittivity.__all__)
__all__.extend(phase_change.__all__)
__all__.extend(property_package.__all__)
__all__.extend(reaction.__all__)
__all__.extend(refractivity.__all__)
__all__.extend(safety.__all__)
__all__.extend(solubility.__all__)
__all__.extend(stream.__all__)
__all__.extend(interface.__all__)
__all__.extend(thermal_conductivity.__all__)
__all__.extend(triple.__all__)
__all__.extend(utils.__all__)
__all__.extend(unifac.__all__)
__all__.extend(vapor_pressure.__all__)
__all__.extend(virial.__all__)
__all__.extend(viscosity.__all__)
__all__.extend(volume.__all__)

# backwards compatibility hack to allow thermo.chemical.Mixture to still be importable
chemical.__dict__['Mixture'] = mixture.Mixture
chemical.__dict__['Stream'] = stream.Stream
# However, they cannot go in thermo.chemical's __all__ or they will appear in the
# documentation and Sphinx currently has no wat to exclude them

__version__ = '0.1.39'
