  # -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

.. contents:: :local:
'''

from __future__ import division

__all__ = ['PropertyPackageConstants', 'IDEAL_PKG', 'NRTL_PKG', 'UNIFAC_PKG', 
           'UNIFAC_DORTMUND_PKG', 'PR_PKG', 'SRK_PKG']

from copy import copy
from random import uniform, shuffle, seed
import numpy as np
from scipy.optimize import golden, brent, minimize, fmin_slsqp, fsolve
from fluids.numerics import brenth, ridder, derivative

from thermo.utils import log, exp
from thermo.utils import has_matplotlib, R, pi, N_A
from thermo.utils import remove_zeros, normalize, Cp_minus_Cv
from thermo.identifiers import IDs_to_CASs
from thermo.activity import K_value, Wilson_K_value, flash_inner_loop, dew_at_T, bubble_at_T, NRTL
from thermo.activity import get_T_bub_est, get_T_dew_est, get_P_dew_est, get_P_bub_est
from thermo.unifac import UNIFAC, UFSG, DOUFSG, DOUFIP2006
from thermo.eos_mix import *
from thermo.eos import *
from thermo.chemical import Chemical
from thermo.mixture import Mixture

from thermo.property_package import *




IDEAL_PKG = 'Ideal'
NRTL_PKG = 'NRTL'
UNIFAC_PKG = 'Unifac'
UNIFAC_DORTMUND_PKG = 'Unifac Dortmund'

PR_PKG = 'PR'
SRK_PKG = 'SRK'

property_packages = [IDEAL_PKG, NRTL_PKG, UNIFAC_PKG, UNIFAC_DORTMUND_PKG, 
                     PR_PKG, SRK_PKG]
property_packages_cubic = [ PR_PKG, SRK_PKG]

property_package_to_eos = {PR_PKG: PRMIX, SRK_PKG: SRKMIX}
property_package_to_eos_pures = {PR_PKG: PR, SRK_PKG: SRK}

property_package_names_to_objs = {IDEAL_PKG: IdealCaloric, 
                                  NRTL_PKG: Nrtl, # Not complete  - enthalpy missing
                                  UNIFAC_PKG: UnifacCaloric,
                                  UNIFAC_DORTMUND_PKG: UnifacDortmundCaloric,
                                  PR_PKG: GceosBase,
                                  SRK_PKG: GceosBase,
                                 }


class PropertyPackageConstants(object):
    '''Class to store kijs, as well as allow properties to be edited; load 
    them from the database and then be ready to store them.
    '''
    def __init__(self, mixture, name=IDEAL_PKG, **kwargs):
        if isinstance(mixture, list):
            self.CASs = IDs_to_CASs(mixture)
            self.Chemicals = [Chemical(CAS) for CAS in self.CASs]
        elif isinstance(mixture, Mixture):
            self.Chemicals = mixture.Chemicals
        self.name = name
        
        if name not in property_packages_cubic:
            eos_mix = PRMIX
            eos = PR
        else:
            eos_mix = property_package_to_eos[name]
            eos = property_package_to_eos_pures[name]
            
        self.eos_in_a_box = [eos_mix]
        
        self.pkg_obj = property_package_names_to_objs[self.name]
                
        self.set_chemical_constants()
        self.set_Chemical_property_objects()
        self.set_TP_sources()
        
        pkg_args = {'VaporPressures': self.VaporPressures,
                   'Tms': self.Tms, 'Tbs': self.Tbs, 'Tcs': self.Tcs,
                   'Pcs': self.Pcs, 'omegas': self.omegas, 'VolumeLiquids': self.VolumeLiquids,
                   'HeatCapacityLiquids': self.HeatCapacityLiquids,
                   'HeatCapacityGases': self.HeatCapacityGases,
                   'EnthalpyVaporizations': self.EnthalpyVaporizations,
                   'VolumeLiquids': self.VolumeLiquids,
                   'VolumeGases': self.VolumeGases,
                    'eos': eos, 'eos_mix': eos_mix,
                    'MWs': self.MWs,
                    'atomss': self.atomss
                   }
        pkg_args.update(kwargs) 
        
        if self.name == UNIFAC_PKG:
            pkg_args['UNIFAC_groups'] = self.UNIFAC_groups
        elif self.name == UNIFAC_DORTMUND_PKG:
            pkg_args['UNIFAC_groups'] = self.UNIFAC_Dortmund_groups
        
        
        
        
        
#        print(pkg_args, self.pkg_obj)
        self.pkg = self.pkg_obj(**pkg_args)
        
        
    def from_json(self, json):
        self.__dict__.update(json)
        self.set_Chemical_property_objects()
        self.set_TP_sources()


transfer_methods = ['set_chemical_constants', 'set_Chemical_property_objects',
                   'set_TP_sources', 'UNIFAC_Dortmund_groups', 'UNIFAC_groups',
                   'atomss']

try:
    for method in transfer_methods:
        attr = Mixture.__dict__[method]
        setattr(PropertyPackageConstants, method, attr)
except:
    for method in transfer_methods:
        setattr(PropertyPackageConstants, method, getattr(Mixture, method))

