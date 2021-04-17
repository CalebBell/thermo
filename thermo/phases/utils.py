# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['Phase']

import sys, os
from math import isinf, isnan, sqrt
from fluids.constants import R, R_inv
import fluids.constants
from fluids.numerics import (horner, horner_and_der, horner_and_der2, horner_log, jacobian, derivative,
                             poly_fit_integral_value, poly_fit_integral_over_T_value,
                             evaluate_linear_fits, evaluate_linear_fits_d,
                             evaluate_linear_fits_d2, quadratic_from_f_ders,
                             newton_system, trunc_log, trunc_exp, newton)
from chemicals.utils import (log, log10, exp, Cp_minus_Cv, phase_identification_parameter,
                          isothermal_compressibility, isobaric_expansion, property_mass_to_molar,
                          Joule_Thomson, speed_of_sound, dxs_to_dns, dns_to_dn_partials,
                          normalize, hash_any_primitive, rho_to_Vm, Vm_to_rho)
from random import randint
from collections import OrderedDict
from chemicals.iapws import *
from chemicals.air import *
from chemicals.viscosity import mu_IAPWS, mu_air_lemmon
from chemicals.thermal_conductivity import k_IAPWS
import chemicals.iapws
from chemicals.iapws import iapws95_d3Ar_ddelta2dtau, iapws95_d3Ar_ddeltadtau2

from thermo.serialize import arrays_to_lists
from thermo.coolprop import has_CoolProp
from thermo.eos import GCEOS, eos_full_path_dict
from thermo.eos_mix import IGMIX, GCEOSMIX, eos_mix_full_path_dict, eos_mix_full_path_reverse_dict
from thermo.eos_mix_methods import PR_lnphis_fastest

from thermo.activity import GibbsExcess, IdealSolution
from thermo.wilson import Wilson
from thermo.unifac import UNIFAC
from thermo.regular_solution import RegularSolution
from thermo.uniquac import UNIQUAC

from thermo.chemical_package import iapws_correlations

from thermo.utils import POLY_FIT
from thermo.heat_capacity import HeatCapacityGas, HeatCapacityLiquid
from thermo.volume import VolumeLiquid, VolumeSolid
from thermo.vapor_pressure import VaporPressure, SublimationPressure
from thermo.phase_change import EnthalpyVaporization, EnthalpySublimation


R2 = R*R


SORTED_DICT = sys.version_info >= (3, 6)
INCOMPRESSIBLE_CONST = 1e30

activity_pointer_reference_dicts = {'thermo.activity.IdealSolution': IdealSolution,
                                    'thermo.wilson.Wilson': Wilson,
                                    'thermo.unifac.UNIFAC': UNIFAC,
                                    'thermo.regular_solution.RegularSolution': RegularSolution,
                                    'thermo.uniquac.UNIQUAC': UNIQUAC,
                                    }
activity_reference_pointer_dicts = {v: k for k, v in activity_pointer_reference_dicts.items()}

object_lookups = activity_pointer_reference_dicts.copy()
object_lookups.update(eos_mix_full_path_dict)
object_lookups.update(eos_full_path_dict)

