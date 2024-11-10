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
SOFTWARE.
'''

import os

from fluids import numerics

if not numerics.is_micropython:

    from chemicals import (
        acentric,
        combustion,
        critical,
        dipole,
        dippr,
        elements,
        environment,
        flash_basic,
        identifiers,
        lennard_jones,
        miscdata,
        rachford_rice,
        reaction,
        refractivity,
        safety,
        solubility,
        temperature,
        triple,
        virial,
    )
    from chemicals.acentric import *  # noqa: F403
    from chemicals.combustion import *  # noqa: F403
    from chemicals.critical import *  # noqa: F403
    from chemicals.dipole import *  # noqa: F403
    from chemicals.dippr import *  # noqa: F403
    from chemicals.elements import *  # noqa: F403
    from chemicals.environment import *  # noqa: F403
    from chemicals.flash_basic import *  # noqa: F403
    from chemicals.identifiers import *  # noqa: F403
    from chemicals.lennard_jones import *  # noqa: F403
    from chemicals.miscdata import *  # noqa: F403
    from chemicals.rachford_rice import *  # noqa: F403
    from chemicals.reaction import *  # noqa: F403
    from chemicals.refractivity import *  # noqa: F403
    from chemicals.safety import *  # noqa: F403
    from chemicals.solubility import *  # noqa: F403
    from chemicals.temperature import *  # noqa: F403
    from chemicals.triple import *  # noqa: F403
    from chemicals.virial import *  # noqa: F403

    from . import (
        activity,
        bulk,
        chemical,
        chemical_package,
        chemical_utils,
        coolprop,
        datasheet,
        electrochem,
        eos,
        eos_alpha_functions,
        eos_mix,
        eos_mix_methods,
        eos_volume,
        equilibrium,
        fitting,
        flash,
        functional_groups,
        group_contribution,
        heat_capacity,
        interaction_parameters,
        interface,
        law,
        mixture,
        nrtl,
        permittivity,
        phase_change,
        phase_identification,
        phases,
        property_package,
        redlich_kister,
        regular_solution,
        stream,
        thermal_conductivity,
        unifac,
        uniquac,
        utils,
        vapor_pressure,
        viscosity,
        volume,
        wilson,
    )
    from .activity import *  # noqa: F403
    from .bulk import *  # noqa: F403
    from .chemical import *  # noqa: F403
    from .chemical_package import *  # noqa: F403
    from .chemical_utils import *  # noqa: F403
    from .coolprop import *  # noqa: F403
    from .datasheet import *  # noqa: F403
    from .electrochem import *  # noqa: F403
    from .eos import *  # noqa: F403
    from .eos_alpha_functions import *  # noqa: F403
    from .eos_mix import *  # noqa: F403
    from .eos_mix_methods import *  # noqa: F403
    from .eos_volume import *  # noqa: F403
    from .equilibrium import *  # noqa: F403
    from .fitting import *  # noqa: F403
    from .flash import *  # noqa: F403
    from .functional_groups import *  # noqa: F403
    from .group_contribution import *  # noqa: F403
    from .heat_capacity import *  # noqa: F403
    from .interaction_parameters import *  # noqa: F403
    from .interface import *  # noqa: F403
    from .law import *  # noqa: F403
    from .mixture import *  # noqa: F403
    from .nrtl import *  # noqa: F403
    from .permittivity import *  # noqa: F403
    from .phase_change import *  # noqa: F403
    from .phase_identification import *  # noqa: F403
    from .phases import *  # noqa: F403
    from .property_package import *  # noqa: F403
    from .redlich_kister import *  # noqa: F403
    from .regular_solution import *  # noqa: F403
    from .stream import *  # noqa: F403
    from .thermal_conductivity import *  # noqa: F403
    from .unifac import *  # noqa: F403
    from .uniquac import *  # noqa: F403
    from .utils import *  # noqa: F403
    from .vapor_pressure import *  # noqa: F403
    from .viscosity import *  # noqa: F403
    from .volume import *  # noqa: F403
    from .wilson import *  # noqa: F403

    #from chemicals import *


    __all__ = ['rachford_rice', 'flash_basic', 'chemical', 'chemical_package', 'combustion', 'critical', 'flash',
     'dipole', 'electrochem', 'elements', 'environment', 'eos', 'eos_mix',
     'heat_capacity',  'identifiers', 'group_contribution', 'law', 'lennard_jones',
     'miscdata',
     'permittivity', 'phase_change', 'phases', 'property_package', 'reaction',
     'refractivity', 'safety', 'solubility', 'interface', 'interaction_parameters',
     'thermal_conductivity', 'triple', 'utils',
     'vapor_pressure', 'virial', 'viscosity', 'volume', 'acentric', 'coolprop',
     'datasheet', 'dippr', 'unifac', 'stream', 'mixture',
     'chemical_utils', 'wilson', 'nrtl', 'uniquac', 'regular_solution',
     'equilibrium', 'phase_identification', 'temperature', 'fitting',
     'eos_alpha_functions', 'eos_volume', 'bulk', 'eos_mix_methods', 'activity',
     'functional_groups', 'redlich_kister']

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
    __all__.extend(group_contribution.__all__)
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
    __all__.extend(functional_groups.__all__)
    __all__.extend(redlich_kister.__all__)

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
                  identifiers, interaction_parameters, interface, group_contribution.joback, law,
                  mixture, nrtl, permittivity, phase_change, phase_identification,
                  property_package, regular_solution,
                  stream, thermal_conductivity, unifac, uniquac, safety,
                  fitting,functional_groups,
                  utils, vapor_pressure, viscosity, volume, wilson, eos_alpha_functions,
                  eos_volume, eos_mix_methods,
                  flash, flash.flash_base, flash.flash_pure_vls,
                  flash.flash_utils, flash.flash_vl, flash.flash_vln,
                  phases, phases.air_phase, phases.ceos, phases.combined,
                  phases.coolprop_phase, phases.gibbs_excess, phases.helmholtz_eos,
                  phases.iapws_phase, phases.ideal_gas, phases.petroleum,
                  phases.phase, phases.phase_utils, phases.virial_phase,
                  utils.functional, utils.mixture_property,
                  utils.t_dependent_property, utils.tp_dependent_property,
                  redlich_kister]

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
            pass
        except:
            pass
    if hasattr(os, '_called_from_test'):
        # pytest timings are hard to measure with lazy loading
        complete_lazy_loading()

    if numerics.PY37:
        def __getattr__(name):
            global vectorized, numba, units, numba_vectorized
            if name == 'vectorized':
                import thermo.vectorized
                return thermo.vectorized
            if name == 'numba':
                import thermo.numba
                return thermo.numba
            if name == 'units':
                import thermo.units
                return thermo.units
            if name == 'numba_vectorized':
                import thermo.numba_vectorized
                return thermo.numba_vectorized
            raise AttributeError(f"module {__name__} has no attribute {name}")
    else:
        pass

try:
    thermo_dir = os.path.dirname(__file__)
except:
    thermo_dir = ''

__version__ = '0.4.0'

