# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020, 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids import numerics

if not numerics.is_micropython:

    from . import eos_alpha_functions
    from . import eos_volume
    from chemicals import acentric
    from chemicals import rachford_rice
    from chemicals import flash_basic
    from . import chemical
    from . import chemical_package
    from chemicals import combustion
    from chemicals import critical
    from . import coolprop
    from chemicals import dipole
    from chemicals import dippr
    from . import datasheet
    from . import electrochem
    from chemicals import elements
    from chemicals import environment
    from . import eos
    from . import eos_mix
    from . import equilibrium
    from . import flash
    from . import heat_capacity
    from chemicals import identifiers
    from . import interaction_parameters
    from . import joback
    from . import law
    from chemicals import lennard_jones
    from chemicals import miscdata
    from . import mixture
    from . import permittivity
    from . import phase_change
    from . import phases
    from . import phase_identification
    from . import property_package
    from . import property_package_constants
    from chemicals import reaction
    from chemicals import refractivity
    from . import regular_solution
    from chemicals import safety
    from chemicals import solubility
    from . import stream
    from . import interface
    from . import thermal_conductivity
    from chemicals import triple
    from . import unifac
    from . import utils
    from . import vapor_pressure
    from chemicals import virial
    from . import viscosity
    from . import volume
    from . import fitting
    from . import chemical_utils
    from . import wilson
    from . import nrtl
    from . import uniquac
    from . import bulk
    from chemicals import temperature
    from . import eos_mix_methods
    from . import activity
    
    from .eos_alpha_functions import *
    from .eos_mix_methods import *
    from .eos_volume import *
    from chemicals.acentric import *
    from chemicals.rachford_rice import *
    from chemicals.flash_basic import *
    from .chemical import *
    from .chemical_package import *
    from chemicals.combustion import *
    from chemicals.critical import *
    from .coolprop import *
    from chemicals.dipole import *
    from chemicals.dippr import *
    from .datasheet import *
    from .electrochem import *
    from chemicals.elements import *
    from chemicals.environment import *
    from .eos import *
    from .eos_mix import *
    from .flash import *
    from .heat_capacity import *
    from .joback import *
    from chemicals.identifiers import *
    from .interaction_parameters import *
    from .law import *
    from .bulk import *
    from chemicals.lennard_jones import *
    from chemicals.miscdata import *
    from .mixture import *
    from .permittivity import *
    from .phase_change import *
    from .phases import *
    from .phase_identification import *
    from .property_package import *
    from .property_package_constants import *
    from chemicals.reaction import *
    from chemicals.refractivity import *
    from .regular_solution import *
    from chemicals.safety import *
    from chemicals.solubility import *
    from .stream import *
    from .interface import *
    from .thermal_conductivity import *
    from chemicals.triple import *
    from .unifac import *
    from .utils import *
    from .vapor_pressure import *
    from chemicals.virial import *
    from .viscosity import *
    from .volume import *
    from .chemical_utils import *
    from .wilson import *
    from .nrtl import *
    from .uniquac import *
    from .equilibrium import *
    from chemicals.temperature import *
    from .activity import *
    from .fitting import *
    
    #from chemicals import *
    
    
    __all__ = ['rachford_rice', 'flash_basic', 'chemical', 'chemical_package', 'combustion', 'critical', 'flash',
     'dipole', 'electrochem', 'elements', 'environment', 'eos', 'eos_mix',
     'heat_capacity',  'identifiers', 'joback', 'law', 'lennard_jones',
     'miscdata',
     'permittivity', 'phase_change', 'phases', 'property_package', 'reaction',
     'refractivity', 'safety', 'solubility', 'interface', 'interaction_parameters',
     'thermal_conductivity', 'triple', 'utils',
     'vapor_pressure', 'virial', 'viscosity', 'volume', 'acentric', 'coolprop',
     'datasheet', 'dippr', 'unifac', 'stream', 'mixture', 'property_package_constants',
     'chemical_utils', 'wilson', 'nrtl', 'uniquac', 'regular_solution',
     'equilibrium', 'phase_identification', 'temperature', 'fitting',
     'eos_alpha_functions', 'eos_volume', 'bulk', 'eos_mix_methods', 'activity']
    
    __all__.extend(eos_volume.__all__)
    __all__.extend(eos_alpha_functions.__all__)
    __all__.extend(acentric.__all__)
    __all__.extend(rachford_rice.__all__)
    __all__.extend(flash_basic.__all__)
    __all__.extend(chemical_package.__all__)
    __all__.extend(chemical.__all__)
    __all__.extend(combustion.__all__)
    __all__.extend(critical.__all__)
    __all__.extend(coolprop.__all__)
    #__all__.extend(dipole.__all__)
    __all__.extend(dippr.__all__)
    __all__.extend(datasheet.__all__)
    __all__.extend(electrochem.__all__)
    __all__.extend(elements.__all__)
    __all__.extend(environment.__all__)
    __all__.extend(eos.__all__)
    __all__.extend(eos_mix.__all__)
    __all__.extend(flash.__all__)
    __all__.extend(heat_capacity.__all__)
    __all__.extend(identifiers.__all__)
    __all__.extend(interaction_parameters.__all__)
    __all__.extend(joback.__all__)
    __all__.extend(law.__all__)
    __all__.extend(lennard_jones.__all__)
    __all__.extend(miscdata.__all__)
    __all__.extend(mixture.__all__)
    __all__.extend(permittivity.__all__)
    __all__.extend(phase_change.__all__)
    __all__.extend(phases.__all__)
    __all__.extend(phase_identification.__all__)
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
    __all__.extend(property_package_constants.__all__)
    __all__.extend(chemical_utils.__all__)
    __all__.extend(wilson.__all__)
    __all__.extend(nrtl.__all__)
    __all__.extend(uniquac.__all__)
    __all__.extend(regular_solution.__all__)
    __all__.extend(equilibrium.__all__)
    __all__.extend(temperature.__all__)
    __all__.extend(bulk.__all__)
    __all__.extend(eos_mix_methods.__all__)
    __all__.extend(activity.__all__)
    __all__.extend(fitting.__all__)
    
    
    # backwards compatibility hack to allow thermo.chemical.Mixture to still be importable
    try:
        chemical.__dict__['Mixture'] = mixture.Mixture
        chemical.__dict__['Stream'] = stream.Stream
    except:
        pass
    # However, they cannot go in thermo.chemical's __all__ or they will appear in the
    # documentation and Sphinx currently has no wat to exclude them
    submodules = [activity, chemical, chemical_package, chemical_utils, coolprop, datasheet,
                  electrochem, eos, eos_mix, equilibrium, heat_capacity,
                  identifiers, interaction_parameters, interface, joback, law,
                  mixture, nrtl, permittivity, phase_change, phase_identification,
                  property_package, property_package_constants, regular_solution, 
                  stream, thermal_conductivity, unifac, uniquac, safety,
                  fitting,
                  utils, vapor_pressure, viscosity, volume, wilson, eos_alpha_functions,
                  eos_volume, eos_mix_methods,              
                  flash, flash.flash_base, flash.flash_pure_vls,
                  flash.flash_utils, flash.flash_vl, flash.flash_vln,              
                  phases, phases.air_phase, phases.ceos, phases.combined, 
                  phases.coolprop_phase, phases.gibbs_excess, phases.helmholtz_eos,
                  phases.iapws_phase, phases.ideal_gas, phases.petroleum, 
                  phases.phase, phases.phase_utils, phases.virial_phase]
    
    def complete_lazy_loading():
        import chemicals
        chemicals.complete_lazy_loading()
        electrochem._load_electrochem_data()
        interaction_parameters.IPDB
        law.load_law_data()
        law.load_economic_data()
        unifac.load_unifac_ip()
        unifac.load_group_assignments_DDBST()
        try:
            import CoolProp
        except:
            pass
    if hasattr(os, '_called_from_test'):
        # pytest timings are hard to measure with lazy loading
        complete_lazy_loading()
    
    global vectorized, numba, units, numba_vectorized
    if numerics.PY37:
        def __getattr__(name):
            global vectorized, numba, units, numba_vectorized
            if name == 'vectorized':
                import thermo.vectorized as vectorized
                return vectorized
            if name == 'numba':
                import thermo.numba as numba
                return numba
            if name == 'units':
                import thermo.units as units
                return units
            if name == 'numba_vectorized':
                import thermo.numba as numba
                import thermo.numba_vectorized as numba_vectorized
                return numba_vectorized
            raise AttributeError("module %s has no attribute %s" %(__name__, name))
    else:
        from . import vectorized
    
try:
    thermo_dir = os.path.dirname(__file__)
except:
    thermo_dir = ''

__version__ = '0.2.7'


