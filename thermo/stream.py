# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['Stream', 'EnergyTypes', 'EnergyStream', 'StreamArgs', 'EquilibriumStream']
 
import enum
from copy import copy, deepcopy
from collections import OrderedDict
from numbers import Number

from thermo.utils import R
from thermo.utils import property_molar_to_mass, property_mass_to_molar, solve_flow_composition_mix
from thermo.utils import mixing_simple, normalize, Vfs_to_zs, ws_to_zs
from thermo.mixture import Mixture, preprocess_mixture_composition
from thermo.equilibrium import EquilibriumState
from thermo.flash import FlashBase
from fluids.pump import voltages_1_phase_residential, voltages_3_phase, frequencies


# Could just assume IDs is always specified and constant.
# This might be useful for regular streams too, just to keep track of what values were specified by the user!
# If one composition value gets set to, remove those from every other value

base_specifications = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None, 
                       'ns': None, 'ms': None, 'Qls': None, 'Qgs': None,
                       'n': None, 'm': None, 'Q': None,
                       'T': None, 'P': None, 'VF': None, 'H': None,
                       'Hm': None, 'S': None, 'Sm': None, 'energy': None}

class StreamArgs(object):
    flashed = False
    def __add__(self, b):
        if not isinstance(b, StreamArgs):
            raise Exception('Adding to a StreamArgs requires that the other object '
                            'also be a StreamArgs.')
    
        a_flow_spec, b_flow_spec = self.flow_spec, b.flow_spec
        a_composition_spec, b_composition_spec = self.composition_spec, b.composition_spec
        a_state_specs, b_state_specs = self.state_specs, b.state_specs
        
        args = {}
        if b.IDs:
            args['IDs'] = b.IDs
        elif self.IDs:
            args['IDs'] = self.IDs
    
        flow_spec = b_flow_spec if b_flow_spec else a_flow_spec
        if flow_spec:
            args[flow_spec[0]] = flow_spec[1]
            
        composition_spec = b_composition_spec if b_composition_spec else a_composition_spec
        if composition_spec:
            args[composition_spec[0]] = composition_spec[1]
            
        if b_state_specs:
            for i, j in b_state_specs:
                args[i] = j
            
        c = StreamArgs(**args)
        if b_state_specs and len(b_state_specs) < 2 and a_state_specs:
            for i, j in a_state_specs:
                try:
                    setattr(c, i, j)
                except:
                    pass
        
        return c
    
    def copy(self):
        # single_composition_basis may mean multiple sets of specs for comp/flow
        kwargs = self.specifications.copy()
        if kwargs['zs'] is not None:
            kwargs['zs'] = [i for i in kwargs['zs']]
        if kwargs['ws'] is not None:
            kwargs['ws'] = [i for i in kwargs['ws']]
        if kwargs['ns'] is not None:
            kwargs['ns'] = [i for i in kwargs['ns']]
        if kwargs['ms'] is not None:
            kwargs['ms'] = [i for i in kwargs['ms']]
        if kwargs['Qls'] is not None:
            kwargs['Qls'] = [i for i in kwargs['Qls']]
        if kwargs['Qgs'] is not None:
            kwargs['Qgs'] = [i for i in kwargs['Qgs']]
        if kwargs['Vfgs'] is not None:
            kwargs['Vfgs'] = [i for i in kwargs['Vfgs']]
        if kwargs['Vfls'] is not None:
            kwargs['Vfls'] = [i for i in kwargs['Vfls']]
        return StreamArgs(Vf_TP=self.Vf_TP, Q_TP=self.Q_TP, pkg=self.pkg,
                 single_composition_basis=self.single_composition_basis, **kwargs)
    
    __copy__ = copy
#        return self.copy()
#        return deepcopy(self)
    
    @property
    def IDs(self):
        return self.specifications['IDs']
    @IDs.setter
    def IDs(self, IDs):
        self.specifications['IDs'] = IDs

    @property
    def energy(self):
        return self.specifications['energy']
    @energy.setter
    def energy(self, energy):
        if energy is None:
            self.specifications['energy'] = energy
            return None
        if self.specified_state_vars > 1 and self.flow_specified and self.energy is None:
            raise Exception('Two state vars and a flow var already specified; unset another first')
        self.specifications['energy'] = energy

    @property
    def T(self):
        return self.specifications['T']
    @T.setter
    def T(self, T):
        if T is None:
            self.specifications['T'] = T
            return None
        if self.specifications['T'] is None and self.specified_state_vars > 1:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['T'] = T

    @property
    def P(self):
        return self.specifications['P']
    @P.setter
    def P(self, P):
        if P is None:
            self.specifications['P'] = P
            return None
        if self.P is None and self.specified_state_vars > 1:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['P'] = P

    @property
    def VF(self):
        return self.specifications['VF']
    @VF.setter
    def VF(self, VF):
        if VF is None:
            self.specifications['VF'] = VF
            return None
        if self.specified_state_vars > 1 and self.VF is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['VF'] = VF

    @property
    def H(self):
        return self.specifications['H']
    @H.setter
    def H(self, H):
        if H is None:
            self.specifications['H'] = H
            return None
        if self.specified_state_vars > 1 and self.H is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['H'] = H

    @property
    def Hm(self):
        return self.specifications['Hm']
    @Hm.setter
    def Hm(self, Hm):
        if Hm is None:
            self.specifications['Hm'] = Hm
            return None
        if self.specified_state_vars > 1 and self.Hm is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['Hm'] = Hm

    @property
    def S(self):
        return self.specifications['S']
    @S.setter
    def S(self, S):
        if S is None:
            self.specifications['S'] = S
            return None
        if self.specified_state_vars > 1 and self.S is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['S'] = S

    @property
    def Sm(self):
        return self.specifications['Sm']
    @Sm.setter
    def Sm(self, Sm):
        if Sm is None:
            self.specifications['Sm'] = Sm
            return None
        if self.specified_state_vars > 1 and self.Sm is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['Sm'] = Sm

    @property
    def zs(self):
        return self.specifications['zs']
    @zs.setter
    def zs(self, arg):
        if arg is None:
            self.specifications['zs'] = arg
        else:
            if self.single_composition_basis:
                args = {'zs': arg, 'ws': None, 'Vfls': None, 'Vfgs': None, 'ns': None,
                        'ms': None, 'Qls': None, 'Qgs': None}
                self.specifications.update(args)
            else:
                self.specifications['zs'] = arg

    @property
    def ws(self):
        return self.specifications['ws']
    @ws.setter
    def ws(self, arg):
        if arg is None:
            self.specifications['ws'] = arg
        else:
            if self.single_composition_basis:
                args = {'zs': None, 'ws': arg, 'Vfls': None, 'Vfgs': None, 'ns': None,
                        'ms': None, 'Qls': None, 'Qgs': None}
                self.specifications.update(args)
            else:
                self.specifications['ws'] = arg

    @property
    def Vfls(self):
        return self.specifications['Vfls']
    @Vfls.setter
    def Vfls(self, arg):
        if arg is None:
            self.specifications['Vfls'] = arg
        else:
            if self.single_composition_basis:
                args = {'zs': None, 'ws': None, 'Vfls': arg, 'Vfgs': None, 'ns': None,
                        'ms': None, 'Qls': None, 'Qgs': None}
                self.specifications.update(args)
            else:
                self.specifications['Vfls'] = arg
                
    @property
    def Vfgs(self):
        return self.specifications['Vfgs']
    @Vfgs.setter
    def Vfgs(self, arg):
        if arg is None:
            self.specifications['Vfgs'] = arg
        else:
            if self.single_composition_basis:
                args = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': arg, 'ns': None,
                        'ms': None, 'Qls': None, 'Qgs': None}
                self.specifications.update(args)
            else:
                self.specifications['Vfgs'] = arg
    @property
    def ns(self):
        return self.specifications['ns']
    @ns.setter
    def ns(self, arg):
        if arg is None:
            self.specifications['ns'] = arg
        else:
            if self.single_composition_basis:
                args = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None, 
                        'ns': arg, 'ms': None, 'Qls': None, 'Qgs': None,
                        'n': None, 'm': None, 'Q': None}
                self.specifications.update(args)
            else:
                self.specifications['ns'] = arg
    @property
    def ms(self):
        return self.specifications['ms']
    @ms.setter
    def ms(self, arg):
        if arg is None:
            self.specifications['ms'] = arg
        else:
            if self.single_composition_basis:
                args = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None, 
                        'ns': None, 'ms': arg, 'Qls': None, 'Qgs': None,
                        'n': None, 'm': None, 'Q': None}
                self.specifications.update(args)
            else:
                self.specifications['ms'] = arg
    @property
    def Qls(self):
        return self.specifications['Qls']
    @Qls.setter
    def Qls(self, arg):
        if arg is None:
            self.specifications['Qls'] = arg
        else:
            if self.single_composition_basis:
                args = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None, 
                        'ns': None, 'ms': None, 'Qls': arg, 'Qgs': None,
                        'n': None, 'm': None, 'Q': None}
                self.specifications.update(args)
            else:
                self.specifications['Qls'] = arg
    
    @property
    def Qgs(self):
        return self.specifications['Qgs']
    @Qgs.setter
    def Qgs(self, arg):
        if arg is None:
            self.specifications['Qgs'] = arg
        else:
            if self.single_composition_basis:
                args = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None, 
                        'ns': None, 'ms': None, 'Qls': None, 'Qgs': arg,
                        'n': None, 'm': None, 'Q': None}
                self.specifications.update(args)
            else:
                self.specifications['Qgs'] = arg
                
    @property
    def m(self):
        return self.specifications['m']
    @m.setter
    def m(self, arg):
        if arg is None:
            self.specifications['m'] = arg
        else:
            args = {'ns': None, 'ms': None, 'Qls': None, 'Qgs': None,
                    'n': None, 'm': arg, 'Q': None}
            self.specifications.update(args)

    @property
    def n(self):
        return self.specifications['n']
    @n.setter
    def n(self, arg):
        if arg is None:
            self.specifications['n'] = arg
        else:
            args = {'ns': None, 'ms': None, 'Qls': None, 'Qgs': None,
                    'n': arg, 'm': None, 'Q': None}
            self.specifications.update(args)
    
    @property
    def Q(self):
        return self.specifications['Q']
    @Q.setter
    def Q(self, arg):
        if arg is None:
            self.specifications['Q'] = arg
        else:
            args = {'ns': None, 'ms': None, 'Qls': None, 'Qgs': None,
                    'n': None, 'm': None, 'Q': arg}
            self.specifications.update(args)

    def __repr__(self):
        return '<StreamArgs, specs %s>' % self.specifications
    def __init__(self, IDs=None, zs=None, ws=None, Vfls=None, Vfgs=None,
                 T=None, P=None, 
                 VF=None, H=None, Hm=None, S=None, Sm=None,
                 ns=None, ms=None, Qls=None, Qgs=None, m=None, n=None, Q=None,
                 energy=None,
                 Vf_TP=(None, None), Q_TP=(None, None, ''), pkg=None,
                 single_composition_basis=True):
#        self.specifications = base_specifications.copy()
        self.specifications = s = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None, 
                       'ns': None, 'ms': None, 'Qls': None, 'Qgs': None,
                       'n': None, 'm': None, 'Q': None,
                       'T': None, 'P': None, 'VF': None, 'H': None,
                       'Hm': None, 'S': None, 'Sm': None, 'energy': None}
        
        # If this is False, DO NOT CLEAR OTHER COMPOSITION / FLOW VARIABLES WHEN SETTING ONE!
        # This makes sense for certain cases but not all.
        self.single_composition_basis = single_composition_basis
        self.IDs = IDs
        self.Vf_TP = Vf_TP
        self.Q_TP = Q_TP
        # pkg should be either a property package or property package constants
        self.pkg = self.property_package = self.property_package_constants = pkg
        self.equilibrium_pkg = isinstance(pkg, FlashBase)
        
        composition_specs = state_specs = flow_specs = 0
        if zs is not None:
            s['zs'] = zs
            composition_specs += 1
        if ws is not None:
            s['ws'] = ws
            composition_specs += 1
        if Vfls is not None:
            s['Vfls'] = Vfls
            composition_specs += 1
        if Vfgs is not None:
            s['Vfgs'] = Vfls
            composition_specs += 1

        if ns is not None:
            s['ns'] = ns
            composition_specs += 1
            flow_specs += 1
        if ms is not None:
            s['ms'] = ms
            composition_specs += 1
            flow_specs += 1
        if Qls is not None:
            s['Qls'] = Qls
            composition_specs += 1
            flow_specs += 1
        if Qgs is not None:
            s['Qgs'] = Qgs
            composition_specs += 1
            flow_specs += 1
        
        if n is not None:
            s['n'] = n
            flow_specs += 1
        if m is not None:
            s['m'] = m
            flow_specs += 1
        if Q is not None:
            s['Q'] = Q
            flow_specs += 1

        if T is not None:
            s['T'] = T
            state_specs += 1
        if P is not None:
            s['P'] = P
            state_specs += 1
        if VF is not None:
            s['VF'] = VF
            state_specs += 1
        if Hm is not None:
            s['Hm'] = Hm
            state_specs += 1
        if H is not None:
            s['H'] = H
            state_specs += 1
        if Sm is not None:
            s['Sm'] = Sm
            state_specs += 1
        if S is not None:
            s['S'] = S
            state_specs += 1
        if energy is not None:
            s['energy'] = energy
            state_specs += 1
            
        if flow_specs > 1:
            raise ValueError("Flow specification is overspecified")
        if composition_specs > 1 and single_composition_basis:
            raise ValueError("Composition specification is overspecified")
        if state_specs > 2:
            raise ValueError("State specification is overspecified")
#        self.zs = zs
#        self.ws = ws
#        self.Vfls = Vfls
#        self.Vfgs = Vfgs
#        self.T = T
#        self.P = P
#        self.VF = VF
#        self.H = H
#        self.Hm = Hm
#        self.S = S
#        self.Sm = Sm
#        
#        self.ns = ns
#        self.ms = ms
#        self.Qls = Qls
#        self.Qgs = Qgs
#        self.m = m
#        self.n = n
#        self.Q = Q
#        self.energy = energy
            
    
    @property
    def composition_spec(self):       
        if self.equilibrium_pkg:
            s = self.specifications
            if s['zs'] is not None:
                return 'zs', s['zs']
            if s['ws'] is not None:
                return 'ws', s['ws']
            if s['Vfls'] is not None:
                return 'Vfls', s['Vfls']
            if s['Vfgs'] is not None:
                return 'Vfgs', s['Vfgs']
            if s['ns'] is not None:
                return 'ns', s['ns']
            if s['ms'] is not None:
                return 'ms', s['ms']
            if s['Qls'] is not None:
                return 'Qls', s['Qls']
            if s['Qgs'] is not None:
                return 'Qgs', s['Qgs']
        else:
            IDs, zs, ws, Vfls, Vfgs = preprocess_mixture_composition(IDs=self.IDs,
                                    zs=self.zs, ws=self.ws, Vfls=self.Vfls, 
                                    Vfgs=self.Vfgs, ignore_exceptions=True)

            if zs is not None:
                return 'zs', zs
            elif ws is not None:
                return 'ws', ws
            elif Vfls is not None:
                return 'Vfls', Vfls
            elif Vfgs is not None:
                return 'Vfgs', Vfgs
            elif self.ns is not None:
                return 'ns', self.ns
            elif self.ms is not None:
                return 'ms', self.ms
            elif self.Qls is not None:
                return 'Qls', self.Qls
            elif self.Qgs is not None:
                return 'Qgs', self.Qgs
    
    @property
    def clean(self):
        '''If no variables (other than IDs) have been specified, return True, 
        otherwise return False.
        '''
        if self.composition_specified or self.state_specs or self.flow_specified:
            return False
        return True
        
    
    @property
    def specified_composition_vars(self):
        IDs, zs, ws, Vfls, Vfgs = preprocess_mixture_composition(IDs=self.IDs,
                                zs=self.zs, ws=self.ws, Vfls=self.Vfls, 
                                Vfgs=self.Vfgs, ignore_exceptions=True)
        
        return sum(i is not None for i in (zs, ws, Vfls, Vfgs, self.ns, self.ms, self.Qls, self.Qgs))

    @property
    def composition_specified(self):
        if self.equilibrium_pkg:
            s = self.specifications
            if s['zs'] is not None:
                return True
            if s['ws'] is not None:
                return True
            if s['Vfls'] is not None:
                return True
            if s['Vfgs'] is not None:
                return True
            if s['ns'] is not None:
                return True
            if s['ms'] is not None:
                return True
            if s['Qls'] is not None:
                return True
            if s['Qgs'] is not None:
                return True
            return False
        IDs, zs, ws, Vfls, Vfgs = preprocess_mixture_composition(IDs=self.IDs,
                                zs=self.zs, ws=self.ws, Vfls=self.Vfls, 
                                Vfgs=self.Vfgs, ignore_exceptions=True)
        
        specified_vals = (i is not None for i in (zs, ws, Vfls, Vfgs, self.ns, self.ms, self.Qls, self.Qgs))
        if any(specified_vals) and IDs:
            return True
        return False
    
    @property
    def state_specs(self):
        s = self.specifications
        specs = []
        if s['T'] is not None:
            specs.append(('T', s['T']))
        if s['P'] is not None:
            specs.append(('P', s['P']))
        if s['VF'] is not None:
            specs.append(('VF', s['VF']))
        if s['Hm'] is not None:
            specs.append(('Hm', s['Hm']))
        if s['H'] is not None:
            specs.append(('H', s['H']))
        if s['Sm'] is not None:
            specs.append(('Sm', s['Sm']))
        if s['S'] is not None:
            specs.append(('S', s['S']))
        if s['energy'] is not None:
            specs.append(('energy', s['energy']))
#        for var in ('T', 'P', 'VF', 'Hm', 'H', 'Sm', 'S', 'energy'):
#            v = getattr(self, var)
#            if v is not None:
#                specs.append((var, v))
        return specs
    
    @property
    def specified_state_vars(self):
        # Slightly faster
        return sum(self.specifications[i] is not None for i in ('T', 'P', 'VF', 'Hm', 'H', 'Sm', 'S', 'energy'))
#        return sum(i is not None for i in (self.T, self.P, self.VF, self.Hm, self.H, self.Sm, self.S, self.energy))
    
    @property
    def state_specified(self):
        s = self.specifications

        state_vars = 0
        if s['T'] is not None:
            state_vars += 1
        if s['P'] is not None:
            state_vars += 1
        if s['VF'] is not None:
            state_vars += 1
        if s['Hm'] is not None:
            state_vars += 1
        if s['H'] is not None:
            state_vars += 1
        if s['S'] is not None:
            state_vars += 1
        if s['Sm'] is not None:
            state_vars += 1
        if s['energy'] is not None:
            state_vars += 1
#        state_vars = (i is not None for i in (self.T, self.P, self.VF, self.Hm, self.H, self.Sm, self.S, self.energy))
        if state_vars == 2:
            return True
        return False
    
    @property
    def non_pressure_spec_specified(self):
        state_vars = (i is not None for i in (self.T, self.VF, self.Hm, self.H, self.Sm, self.S, self.energy))
        if sum(state_vars) >= 1:
            return True
        return False
    
    @property
    def flow_spec(self):
        s = self.specifications
        if s['ns'] is not None:
            return ('ns', s['ns'])
        if s['ms'] is not None:
            return ('ms', s['ms'])
        if s['Qls'] is not None:
            return ('Qls', s['Qls'])
        if s['Qgs'] is not None:
            return ('Qgs', s['Qgs'])
        if s['m'] is not None:
            return ('m', s['m'])
        if s['n'] is not None:
            return ('n', s['n'])
        if s['Q'] is not None:
            return ('Q', s['Q'])
#        
#        # TODO consider energy?
#        specs = []
#        for var in ('ns', 'ms', 'Qls', 'Qgs', 'm', 'n', 'Q'):
#            v = getattr(self, var)
#            if v is not None:
#                return var, v

    @property
    def specified_flow_vars(self):
        return sum(i is not None for i in (self.ns, self.ms, self.Qls, self.Qgs, self.m, self.n, self.Q))

    @property
    def flow_specified(self):
        s = self.specifications
        if s['ns'] is not None:
            return True
        if s['ms'] is not None:
            return True
        if s['Qls'] is not None:
            return True
        if s['Qgs'] is not None:
            return True
        if s['m'] is not None:
            return True
        if s['n'] is not None:
            return True
        if s['Q'] is not None:
            return True
        return False
#        flow_vars = (i is not None for i in (self.ns, self.ms, self.Qls, self.Qgs, self.m, self.n, self.Q))
#        if sum(flow_vars) == 1:
#            return True
#        return False
        
    def update(self, **kwargs):
        for key, value in kwargs:
            setattr(self, key, value)
    
    def flash(self, hot_start=None, existing_flash=None):
#        if self.flow_specified and self.composition_specified and self.state_specified:
        s = self.specifications.copy()
        del s['IDs']
        if 'S' in s:
            if s['S'] is not None:
                s['S_mass'] = s['S']
            del s['S']
        if 'Sm' in s:
            if s['Sm'] is not None:
                s['S'] = s['Sm']
            del s['Sm']
        if 'H' in s:
            if s['H'] is not None:
                s['H_mass'] = s['H']
            del s['H']
        if 'Hm' in s:
            if s['Hm'] is not None:
                s['H'] = s['Hm']
            del s['Hm']
        return EquilibriumStream(self.property_package, hot_start=hot_start,
                                 existing_flash=existing_flash, **s)
    
    @property
    def stream(self):
        if self.equilibrium_pkg:
            if self.flow_specified and self.composition_specified and self.state_specified:
                s = self.specifications.copy()
                del s['IDs']
                if 'S' in s:
                    if s['S'] is not None:
                        s['S_mass'] = s['S']
                    del s['S']
                if 'Sm' in s:
                    if s['Sm'] is not None:
                        s['S'] = s['Sm']
                    del s['Sm']
                if 'H' in s:
                    if s['H'] is not None:
                        s['H_mass'] = s['H']
                    del s['H']
                if 'Hm' in s:
                    if s['Hm'] is not None:
                        s['H'] = s['Hm']
                    del s['Hm']
                return EquilibriumStream(self.property_package, **s)
        else:
            if self.IDs and self.composition_specified and self.state_specified and self.flow_specified:
                return Stream(IDs=self.IDs, zs=self.zs, ws=self.ws, Vfls=self.Vfls, Vfgs=self.Vfgs,
                     ns=self.ns, ms=self.ms, Qls=self.Qls, Qgs=self.Qgs, 
                     n=self.n, m=self.m, Q=self.Q, 
                     T=self.T, P=self.P, VF=self.VF, H=self.H, S=self.S, 
                     Hm=self.Hm, Sm=self.Sm, 
                     Vf_TP=self.Vf_TP, Q_TP=self.Q_TP, energy=self.energy,
                     pkg=self.property_package)
        
    @property
    def mixture(self):
        if self.IDs and self.composition_specified and self.state_specified:
            return Mixture(IDs=self.IDs, zs=self.zs, ws=self.ws, Vfls=self.Vfls, Vfgs=self.Vfgs,
                 T=self.T, P=self.P, VF=self.VF, H=self.H, S=self.S, Hm=self.Hm, Sm=self.Sm, 
                 Vf_TP=self.Vf_TP, pkg=self.property_package)

class Stream(Mixture):
    '''Creates a Stream object which is useful for modeling mass and energy 
    balances.
    
    Streams have five variables. The flow rate, composition, and components are
    mandatory; and two of the variables temperature, pressure, vapor fraction, 
    enthalpy, or entropy are required. Entropy and enthalpy may also be
    provided in a molar basis; energy can also be provided, which when
    combined with either a flow rate or enthalpy will calculate the other
    variable.
    
    The composition and flow rate may be specified together or separately. The
    options for specifying them are:
    
    * Mole fractions `zs`
    * Mass fractions `ws`
    * Liquid standard volume fractions `Vfls`
    * Gas standard volume fractions `Vfgs`
    * Mole flow rates `ns`
    * Mass flow rates `ms`
    * Liquid flow rates `Qls` (based on pure component volumes at the T and P 
      specified by `Q_TP`)
    * Gas flow rates `Qgs` (based on pure component volumes at the T and P 
      specified by `Q_TP`)

    If only the composition is specified by providing any of `zs`, `ws`, `Vfls`
    or `Vfgs`, the flow rate must be specified by providing one of these:
        
    * Mole flow rate `n`
    * Mass flow rate `m`
    * Volumetric flow rate `Q` at the provided `T` and `P` or if specified,
      `Q_TP`
    * Energy `energy`
    
    The state variables must be two of the following. Not all combinations 
    result in a supported flash.
        
    * Tempetarure `T`
    * Pressure `P`
    * Vapor fraction `VF`
    * Enthalpy `H`
    * Molar enthalpy `Hm`
    * Entropy `S`
    * Molar entropy `Sm`
    * Energy `energy`
    
    Parameters
    ----------
    IDs : list, optional
        List of chemical identifiers - names, CAS numbers, SMILES or InChi 
        strings can all be recognized and may be mixed [-]
    zs : list or dict, optional
        Mole fractions of all components in the stream [-]
    ws : list or dict, optional
        Mass fractions of all components in the stream [-]
    Vfls : list or dict, optional
        Volume fractions of all components as a hypothetical liquid phase based 
        on pure component densities [-]
    Vfgs : list or dict, optional
        Volume fractions of all components as a hypothetical gas phase based 
        on pure component densities [-]
    ns : list or dict, optional
        Mole flow rates of each component [mol/s]
    ms : list or dict, optional
        Mass flow rates of each component [kg/s]
    Qls : list or dict, optional
        Liquid flow rates of all components as a hypothetical liquid phase  
        based on pure component densities [m^3/s]
    Qgs : list or dict, optional
        Gas flow rates of all components as a hypothetical gas phase  
        based on pure component densities [m^3/s]
    n : float, optional
        Total mole flow rate of all components in the stream [mol/s]
    m : float, optional
        Total mass flow rate of all components in the stream [kg/s]
    Q : float, optional
        Total volumetric flow rate of all components in the stream based on the
        temperature and pressure specified by `T` and `P` [m^3/s]
    T : float, optional
        Temperature of the stream (default 298.15 K), [K]
    P : float, optional
        Pressure of the stream (default 101325 Pa) [Pa]
    VF : float, optional
        Vapor fraction (mole basis) of the stream, [-]
    H : float, optional
        Mass enthalpy of the stream, [J]
    Hm : float, optional
        Molar enthalpy of the stream, [J/mol]
    S : float, optional
        Mass entropy of the stream, [J/kg/K]
    Sm : float, optional
        Molar entropy of the stream, [J/mol/K]
    energy : float, optional
        Flowing energy of the stream (`H`*`m`), [W]
    pkg : object 
        The thermodynamic property package to use for flash calculations;
        one of the caloric packages in :obj:`thermo.property_package`;
        defaults to the ideal model [-]
    Vf_TP : tuple(2, float), optional
        The (T, P) at which the volume fractions are specified to be at, [K] 
        and [Pa] 
    Q_TP : tuple(3, float, float, str), optional
        The (T, P, phase) at which the volumetric flow rate is specified to be 
        at, [K] and [Pa]
 
    Examples
    --------
    Creating Stream objects:
        
        
    A stream of vodka with volume fractions 60% liquid, 40% ethanol, 1 kg/s:

    >>> from thermo import Stream
    >>> Stream(['water', 'ethanol'], Vfls=[.6, .4], T=300, P=1E5, m=1)
    <Stream, components=['water', 'ethanol'], mole fractions=[0.8299, 0.1701], mole flow=43.8839741023 mol/s, T=300.00 K, P=100000 Pa>
    
    A stream of air at 400 K and 1 bar, flow rate of 1 mol/s:
    
    >>> Stream('air', T=400, P=1e5, n=1)
    <Stream, components=['nitrogen', 'argon', 'oxygen'], mole fractions=[0.7812, 0.0092, 0.2096], mole flow=1 mol/s, T=400.00 K, P=100000 Pa>

    A flow of 1 L/s of 10 wt% phosphoric acid at 320 K:
    
    >>> Stream(['water', 'phosphoric acid'], ws=[.9, .1], T=320, P=1E5, Q=0.001)
    <Stream, components=['water', 'phosphoric acid'], mole fractions=[0.98, 0.02], mole flow=53.2136286991 mol/s, T=320.00 K, P=100000 Pa>
    
    Instead of specifying the composition and flow rate separately, they can
    be specified as a list of flow rates in the appropriate units.
    
    80 kg/s of furfuryl alcohol/water solution:
    
    >>> Stream(['furfuryl alcohol', 'water'], ms=[50, 30])
    <Stream, components=['furfuryl alcohol', 'water'], mole fractions=[0.2343, 0.7657], mole flow=2174.93735951 mol/s, T=298.15 K, P=101325 Pa>
    
    A stream of 100 mol/s of 400 K, 1 MPa argon:

    >>> Stream(['argon'], ns=[100], T=400, P=1E6)
    <Stream, components=['argon'], mole fractions=[1.0], mole flow=100 mol/s, T=400.00 K, P=1000000 Pa>

    A large stream of vinegar, 8 volume %:
        
    >>> Stream(['Acetic acid', 'water'], Qls=[1, 1/.088])
    <Stream, components=['acetic acid', 'water'], mole fractions=[0.0269, 0.9731], mole flow=646268.518749 mol/s, T=298.15 K, P=101325 Pa>

    A very large stream of 100 m^3/s of steam at 500 K and 2 MPa:

    >>> Stream(['water'], Qls=[100], T=500, P=2E6)
    <Stream, components=['water'], mole fractions=[1.0], mole flow=4617174.33613 mol/s, T=500.00 K, P=2000000 Pa>

    A real example of a stream from a pulp mill:
        
    >>> Stream(['Methanol', 'Sulphuric acid', 'sodium chlorate', 'Water', 'Chlorine dioxide', 'Sodium chloride', 'Carbon dioxide', 'Formic Acid', 'sodium sulfate', 'Chlorine'], T=365.2, P=70900, ns=[0.3361749, 11.5068909, 16.8895876, 7135.9902928, 1.8538332, 0.0480655, 0.0000000, 2.9135162, 205.7106922, 0.0012694])
    <Stream, components=['methanol', 'sulfuric acid', 'sodium chlorate', 'water', 'chlorine dioxide', 'sodium chloride', 'carbon dioxide', 'formic acid', 'sodium sulfate', 'chlorine'], mole fractions=[0.0, 0.0016, 0.0023, 0.9676, 0.0003, 0.0, 0.0, 0.0004, 0.0279, 0.0], mole flow=7375.2503227 mol/s, T=365.20 K, P=70900 Pa>

    For streams with large numbers of components, it may be confusing to enter
    the composition separate from the names of the chemicals. For that case,
    the syntax using dictionaries as follows is supported with any composition
    specification:
    
    >>> comp = OrderedDict([('methane', 0.96522),
    ...                     ('nitrogen', 0.00259),
    ...                     ('carbon dioxide', 0.00596),
    ...                     ('ethane', 0.01819),
    ...                     ('propane', 0.0046),
    ...                     ('isobutane', 0.00098),
    ...                     ('butane', 0.00101),
    ...                     ('2-methylbutane', 0.00047),
    ...                     ('pentane', 0.00032),
    ...                     ('hexane', 0.00066)])
    >>> m = Stream(ws=comp, m=33)
    '''    
    def __repr__(self): # pragma: no cover
        txt = '<Stream, components=%s, mole fractions=%s, mass flow=%s kg/s, mole flow=%s mol/s' % (self.names, [round(i,4) for i in self.zs], self.m, self.n)
        # T and P may not be available if a flash has failed
        try:
            txt += ', T=%.2f K, P=%.0f Pa>' %(self.T, self.P)
        except:
            txt += ', thermodynamic conditions unknown>'
        return txt


    def __init__(self, IDs=None, zs=None, ws=None, Vfls=None, Vfgs=None,
                 ns=None, ms=None, Qls=None, Qgs=None, 
                 n=None, m=None, Q=None, 
                 T=None, P=None, VF=None, H=None, Hm=None, S=None, Sm=None,
                 energy=None, pkg=None, Vf_TP=(None, None), Q_TP=(None, None, '')):
        
        composition_options = ('zs', 'ws', 'Vfls', 'Vfgs', 'ns', 'ms', 'Qls', 'Qgs')
        composition_option_count = 0
        for i in composition_options:
            if locals()[i] is not None:
                composition_option_count += 1
                self.composition_spec = (i, locals()[i])
                
        if hasattr(IDs, 'strip') or (type(IDs) == list and len(IDs) == 1):
            pass # one component only - do not raise an exception
        elif composition_option_count < 1:
            raise Exception("No composition information is provided; one of "
                            "'ws', 'zs', 'Vfls', 'Vfgs', 'ns', 'ms', 'Qls' or "
                            "'Qgs' must be specified")
        elif composition_option_count > 1:
            raise Exception("More than one source of composition information "
                            "is provided; only one of "
                            "'ws', 'zs', 'Vfls', 'Vfgs', 'ns', 'ms', 'Qls' or "
                            "'Qgs' can be specified")
        
        
            
        # if more than 1 of composition_options is given, raise an exception
        flow_options = ('ns', 'ms', 'Qls', 'Qgs', 'm', 'n', 'Q') # energy
        flow_option_count = 0
        for i in flow_options:
            if locals()[i] is not None:
                flow_option_count += 1
                self.flow_spec = (i, locals()[i])
        
        
#        flow_option_count = sum(i is not None for i in flow_options)
        # Energy can be used as an enthalpy spec or a flow rate spec
        if flow_option_count > 1 and energy is not None:
            if Hm is not None or H is not None:
                flow_option_count -= 1
        
        if flow_option_count < 1:
            raise Exception("No flow rate information is provided; one of "
                            "'m', 'n', 'Q', 'ms', 'ns', 'Qls', 'Qgs' or "
                            "'energy' must be specified")
        elif flow_option_count > 1:
            raise Exception("More than one source of flow rate information is "
                            "provided; only one of "
                            "'m', 'n', 'Q', 'ms', 'ns', 'Qls', 'Qgs' or "
                            "'energy' can be specified")
        
        if ns is not None:
            zs = ns
        elif ms is not None:
            ws = ms
        elif Qls is not None:
            Vfls = Qls
        elif Qgs is not None:
            Vfgs = Qgs
        
        # If T and P are known, only need to flash once
        if T is not None and P is not None:
            super(Stream, self).__init__(IDs, zs=zs, ws=ws, Vfls=Vfls, Vfgs=Vfgs,
                 T=T, P=P, Vf_TP=Vf_TP, pkg=pkg)
        else:
            # Initialize without a flash
            Mixture.autoflash = False
            super(Stream, self).__init__(IDs, zs=zs, ws=ws, Vfls=Vfls, Vfgs=Vfgs,
                 Vf_TP=Vf_TP, pkg=pkg)
            Mixture.autoflash = True
                        
        

        if n is not None:
            self.n = n
        elif m is not None:
            self.n = property_molar_to_mass(m, self.MW) # m*10000/MW
        elif Q is not None:
            try:
                if Q_TP != (None, None, ''):
                    if len(Q_TP) == 2 or (len(Q_TP) == 3 and not Q_TP[-1]):
                        # Calculate the phase via the property package
                        self.property_package.flash(self.zs, T=Q_TP[0], P=Q_TP[1])
                        phase = self.property_package.phase if self.property_package.phase in ('l', 'g') else 'g'
                    else:
                        phase = Q_TP[-1]
                    if phase == 'l':
                        Vm = self.VolumeLiquidMixture(T=Q_TP[0], P=Q_TP[1], zs=self.zs, ws=self.ws)
                    else:
                        Vm = self.VolumeGasMixture(T=Q_TP[0], P=Q_TP[1], zs=self.zs, ws=self.ws)
                    
                else:
                    Vm = self.Vm
                self.n = Q/Vm
            except:
                raise Exception('Molar volume could not be calculated to determine the flow rate of the stream.')
        elif ns is not None:
            if isinstance(ns, (OrderedDict, dict)):
                ns = ns.values()
            self.n = sum(ns)
        elif ms is not None:
            if isinstance(ms, (OrderedDict, dict)):
                ms = ms.values()
            self.n = property_molar_to_mass(sum(ms), self.MW)
        elif Qls is not None:
            # volume flows and total enthalpy/entropy should be disabled
            try:
                if isinstance(Qls, (OrderedDict, dict)):
                    Qls = Qls.values()
                self.n = sum([Q/Vml for Q, Vml in zip(Qls, self.Vmls)])
            except:
                raise Exception('Liquid molar volume could not be calculated to determine the flow rate of the stream.')
        elif Qgs is not None:
            try:
                if isinstance(Qgs, (OrderedDict, dict)):
                    Qgs = Qgs.values()
                self.n = sum([Q/Vmg for Q, Vmg in zip(Qgs, self.Vmgs)])
            except:
                raise Exception('Gas molar volume could not be calculated to determine the flow rate of the stream.')
        elif energy is not None:
            if H is not None:
                self.m = energy/H # Watt/(J/kg) = kg/s
            elif Hm is not None:
                self.n = energy/Hm # Watt/(J/kg) = mol/s
            else:
                raise NotImplemented
        
        # Energy specified - calculate H or Hm
        if energy is not None:
            if hasattr(self, 'm'):
                H = energy/self.m
            if hasattr(self, 'n'):
                Hm = energy/self.n
        
        if T is None or P is None: 
            non_TP_state_vars = sum(i is not None for i in [VF, H, Hm, S, Sm, energy])
            if non_TP_state_vars == 0:
                if T is None:
                    T = self.T_default
                if P is None:
                    P = self.P_default
                
        self.flash(T=T, P=P, VF=VF, H=H, Hm=Hm, S=S, Sm=Sm)
        
        self.set_extensive_flow(self.n)
        self.set_extensive_properties()

    def set_extensive_flow(self, n=None):
        if n is None:
            n = self.n
        self.n = n
        self.m = property_mass_to_molar(self.n, self.MW)
        self.ns = [self.n*zi for zi in self.zs]
        self.ms =  [self.m*wi for wi in self.ws]
        try:
            self.Q = self.m/self.rho
        except:
            pass
        try:
            self.Qgs = [m/rho for m, rho in zip(self.ms, self.rhogs)]
        except:
            pass
        try:
            self.Qls = [m/rho for m, rho in zip(self.ms, self.rhols)]
        except:
            pass
        
        if self.phase == 'l/g' or self.phase == 'l':
            self.nl = self.n*(1. - self.V_over_F)
            self.nls = [xi*self.nl for xi in self.xs]
            self.mls = [ni*MWi*1E-3 for ni, MWi in zip(self.nls, self.MWs)]
            self.ml = sum(self.mls)
            if self.rhol:
                self.Ql = self.ml/self.rhol 
            else:
                self.Ql = None

        if self.phase == 'l/g' or self.phase == 'g':
            self.ng = self.n*self.V_over_F
            self.ngs = [yi*self.ng for yi in self.ys]
            self.mgs = [ni*MWi*1E-3 for ni, MWi in zip(self.ngs, self.MWs)]
            self.mg = sum(self.mgs)
            if self.rhog:
                self.Qg = self.mg/self.rhog
            else:
                self.Qg = None
        
    def StreamArgs(self):
        '''Goal to create a StreamArgs instance, with the user specified
        variables always being here.
        
        The state variables are currently correctly tracked. The flow rate and
        composition variable needs to be tracked as a function of what was
        specified as the input variables.
        
        The flow rate needs to be changed wen the stream flow rate is changed.
        Note this stores unnormalized specs, but that this is OK.
        '''
        kwargs = {i:j for i, j in zip(('T', 'P', 'VF', 'H', 'Hm', 'S', 'Sm'), self.specs)}
        kwargs['IDs'] = self.IDs
        kwargs[self.composition_spec[0]] = self.composition_spec[1]
        kwargs[self.flow_spec[0]] = self.flow_spec[1]
        return StreamArgs(**kwargs)
    
    
    # flow_spec, composition_spec are attributes already
    @property
    def specified_composition_vars(self):
        '''number of composition variables'''
        return 1

    @property
    def composition_specified(self):
        '''Always needs a composition'''
        return True
    
    @property
    def specified_state_vars(self):
        '''Always needs two states'''
        return 2
    
    @property
    def non_pressure_spec_specified(self):
        '''Cannot have a stream without an energy-type spec.
        '''
        return True
    
    @property
    def state_specified(self):
        '''Always needs a state'''
        return True
    
    @property
    def state_specs(self):
        '''Returns a list of tuples of (state_variable, state_value) representing
        the thermodynamic state of the system.
        '''
        specs = []
        for i, var in enumerate(('T', 'P', 'VF', 'Hm', 'H', 'Sm', 'S', 'energy')):
            v = self.specs[i]
            if v is not None:
                specs.append((var, v))
        return specs

    @property
    def specified_flow_vars(self):
        '''Always needs only one flow specified'''
        return 1

    @property
    def flow_specified(self):
        '''Always needs a flow specified'''
        return True

        
    def flash(self, T=None, P=None, VF=None, H=None, Hm=None, S=None, Sm=None, 
              energy=None):
        self.specs = (T, P, VF, H, Hm, S, Sm, energy)
        if energy is not None:
            H = energy/self.m
        
        if H is not None:
            Hm = property_mass_to_molar(H, self.MW)

        if S is not None:
            Sm = property_mass_to_molar(S, self.MW)
        super(Stream, self).flash_caloric(T=T, P=P, VF=VF, Hm=Hm, Sm=Sm)
        self.set_extensive_properties()


    def set_extensive_properties(self):
        if not hasattr(self, 'm'):
            self.energy = None
            return None
        if self.H is not None and self.m is not None:
            self.energy = self.H*self.m
            self.energy_reactive = self.H_reactive*self.m
        else:
            self.energy = None

    def calculate(self, T=None, P=None):
        self.set_TP(T=T, P=P)
        self.set_phase()
        if hasattr(self, 'rho') and self.rho:
            self.Q = self.m/self.rho
        else:
            self.Q = None
        self.set_extensive_flow()
        self.set_extensive_properties()

    def __add__(self, other):
        if not isinstance(other, Stream):
            raise Exception('Adding to a stream requires that the other object '
                            'also be a stream.')
        
        if (set(self.CASs) == set(other.CASs)) and (len(self.CASs) == len(other.CASs)):
            cmps = self.CASs
        else:
            cmps = sorted(list(set((self.CASs + other.CASs))))
        mole = self.n + other.n
        moles = []
        for cmp in cmps:
            moles.append(0)
            if cmp in self.CASs:
                ind = self.CASs.index(cmp)
                moles[-1] += self.zs[ind]*self.n
            if cmp in other.CASs:
                ind = other.CASs.index(cmp)
                moles[-1] += other.zs[ind]*other.n

        T = min(self.T, other.T)
        P = min(self.P, other.P)
        return Stream(IDs=cmps, ns=moles, T=T, P=P, pkg=self.property_package)

    def __sub__(self, other):
        # Subtracts the mass flow rates in other from self and returns a new
        # Stream instance

        # Check if all components are present in the original stream,
        # while ignoring 0-flow streams in other
        components_in_self = [i in self.CASs for i in other.CASs]
        if not all(components_in_self):
            for i, in_self in enumerate(components_in_self):
                if not in_self and other.zs[i] > 0:
                    raise Exception('Not all components to be removed are \
present in the first stream; %s is not present.' %other.IDs[i])

        # Calculate the mole flows of each species
        ns_self = list(self.ns)
        ns_other = list(other.ns)
        n_product = sum(ns_self) - sum(ns_other)

        for i, CAS in enumerate(self.CASs):
            if CAS in other.CASs:
                nj = ns_other[other.CASs.index(CAS)]
                # Merely normalizing the mole flow difference is enough to make 
                # ~1E-16 relative differences; allow for a little tolerance here 
                relative_difference_product = abs(ns_self[i] - nj)/n_product
                relative_difference_self = abs(ns_self[i] - nj)/ns_self[i]
                if ns_self[i] - nj < 0 and (relative_difference_product > 1E-12 or relative_difference_self > 1E-9):
                    raise Exception('Attempting to remove more %s than is in the \
first stream.' %self.IDs[i])
                if ns_self[i] - nj < 0.:
                    ns_self[i] = 0.
                elif relative_difference_product < 1E-12:
                    ns_self[i] = 0.
                else:
                    ns_self[i] = ns_self[i] - nj


        # Remove now-empty streams:
        ns_product = []
        CASs_product = []
        for n, CAS in zip(ns_self, self.CASs):
            if n != 0:
                ns_product.append(n)
                CASs_product.append(CAS)
        # Create the resulting stream
        return Stream(IDs=CASs_product, ns=ns_product, T=self.T, P=self.P)



class EquilibriumStream(EquilibriumState):
    flashed = True
    
    def __init__(self, flasher, zs=None, ws=None, Vfls=None, Vfgs=None,
                 ns=None, ms=None, Qls=None, Qgs=None, 
                 n=None, m=None, Q=None, 
                 T=None, P=None, VF=None, H=None, H_mass=None, S=None, S_mass=None,
                 energy=None, Vf_TP=None, Q_TP=None, hot_start=None, 
                 existing_flash=None):
        
        constants = flasher.constants
        
        # Composition information
        composition_option_count = 0
        if zs is not None:
            composition_option_count += 1
            self.composition_spec = ('zs', zs)
        if ws is not None:
            composition_option_count += 1
            self.composition_spec = ('ws', ws)
        if Vfls is not None:
            composition_option_count += 1
            self.composition_spec = ('Vfls', Vfls)
        if Vfgs is not None:
            composition_option_count += 1
            self.composition_spec = ('Vfgs', Vfgs)
        if ns is not None:
            composition_option_count += 1
            self.composition_spec = ('ns', ns)
        if ms is not None:
            composition_option_count += 1
            self.composition_spec = ('ms', ms)
        if Qls is not None:
            composition_option_count += 1
            self.composition_spec = ('Qls', Qls)
        if Qgs is not None:
            composition_option_count += 1
            self.composition_spec = ('Qgs', Qgs)
        if composition_option_count < 1:
            raise Exception("No composition information is provided; one of "
                            "'ws', 'zs', 'Vfls', 'Vfgs', 'ns', 'ms', 'Qls' or "
                            "'Qgs' must be specified")
        elif composition_option_count > 1:
            raise Exception("More than one source of composition information "
                            "is provided; only one of "
                            "'ws', 'zs', 'Vfls', 'Vfgs', 'ns', 'ms', 'Qls' or "
                            "'Qgs' can be specified")
        
        flow_option_count = 0
        if ns is not None:
            flow_option_count += 1
            self.flow_spec = ('ns', ns)
        if ms is not None:
            flow_option_count += 1
            self.flow_spec = ('ms', ms)
        if Qls is not None:
            flow_option_count += 1
            self.flow_spec = ('Qls', Qls)
        if Qgs is not None:
            flow_option_count += 1
            self.flow_spec = ('Qgs', Qgs)
        if m is not None:
            flow_option_count += 1
            self.flow_spec = ('m', m)
        if n is not None:
            flow_option_count += 1
            self.flow_spec = ('n', n)
        if Q is not None:
            flow_option_count += 1
            self.flow_spec = ('Q', Q)


        if flow_option_count > 1 and energy is not None:
            if H is not None or H_mass is not None:
                flow_option_count -= 1
        
        if flow_option_count < 1:
            raise Exception("No flow rate information is provided; one of "
                            "'m', 'n', 'Q', 'ms', 'ns', 'Qls', 'Qgs' or "
                            "'energy' must be specified")
        elif flow_option_count > 1:
            raise Exception("More than one source of flow rate information is "
                            "provided; only one of "
                            "'m', 'n', 'Q', 'ms', 'ns', 'Qls', 'Qgs' or "
                            "'energy' can be specified")
        
        # Make sure mole fractions are available
        if ns is not None:
            zs = normalize(ns)
        elif ms is not None:
            zs = ws_to_zs(normalize(ms), constants.MWs)
        elif ws is not None:
            zs = ws_to_zs(ws, constants.MWs)
        elif Qls is not None or Vfls is not None:
            if Vfls is None:
                Vfls = normalize(Qls)
            if Vf_TP is not None and Vf_TP != (None, None):
                VolumeObjects = flasher.properties.VolumeLiquids
                Vms_TP = [i(T_vf, P_vf) for i in VolumeObjects]
            else:
                Vms_TP = constants.Vml_STPs
            zs = Vfs_to_zs(Vfls, Vms_TP)
        elif Qgs is not None: 
            zs = normalize(Qgs)
        elif Vfgs is not None:
            zs = Vfgs
        
        self._MW = MW = mixing_simple(zs, constants.MWs)
        MW_inv = 1.0/MW

        if H_mass is not None:
            H = property_mass_to_molar(H_mass, MW)
        if S_mass is not None:
            S = property_mass_to_molar(S_mass, MW)
        if energy is not None:
            # Handle the various mole flows - converting to get energy; subset for now
            if m is not None:
                n = property_molar_to_mass(m, MW)  # m*10000/MW
            elif ns is not None:
                n = sum(ns)
            elif ms is not None:
                n = property_molar_to_mass(sum(ms), MW)
            H = energy/n

        if existing_flash is not None:
            self.__dict__.update(existing_flash)
        else:
            dest = super(EquilibriumStream, self).__init__
            flasher.flash(T=T, P=P, VF=VF, H=H, S=S, dest=dest, zs=zs, hot_start=hot_start)
        
        # Convert the flow rate into total molar flow
        if m is not None:
            n = property_molar_to_mass(m, MW) # m*10000/MW
        elif ns is not None:
            n = sum(ns)
        elif ms is not None:
            n = property_molar_to_mass(sum(ms), MW)
        elif Q is not None:
            try:
                if Q_TP is not None:
                    if len(Q_TP) == 2 or (len(Q_TP) == 3 and not Q_TP[-1]):
                        # Calculate the volume via the property package
                        expensive_flash = flasher.flash(zs=zs, T=Q_TP[0], P=Q_TP[1])
                        V = expensive_flash.V()
                    if Q_TP[-1] == 'l':
                        V = flasher.liquids[0].to(T=Q_TP[0], P=Q_TP[1], zs=zs).V()
                    elif Q_TP[-1] == 'g':
                        V = flasher.gas.to(T=Q_TP[0], P=Q_TP[1], zs=zs).V()
                else:
                    V = self.V()
                n = Q/V
            except:
                raise Exception('Molar volume could not be calculated to determine the flow rate of the stream.')
        elif Qls is not None:
            n = 0.0
            try:
                for i in self.cmps:
                    n += Qls[i]/Vms_TP[i]
            except:
                raise Exception('Liquid molar volume could not be calculated to determine the flow rate of the stream.')
        elif Qgs is not None:
            # Use only ideal gas law; allow user T, P but default to flasher settings when not speced
            if Q_TP is not None and Q_TP[0] is not None and Q_TP[1] is not None:
                V = R*Q_TP[0]/Q_TP[1]
            else:
                V = R*flasher.settings.T_gas_ref/flasher.settings.P_gas_ref
            n = sum(Qgs)/V
        elif energy is not None:
            n = energy/H # Watt/(J/mol) = mol/s # probably wrong
        self.n = n

        self.m = m = property_mass_to_molar(n, MW)
        self.ns = [n*zi for zi in zs]
        self._ws = ws = [zi*MWi*MW_inv for zi, MWi in zip(zs, constants.MWs)]
        self.ms = [m*wi for wi in ws]
    
    @property
    def energy(self):
        return self.H()*self.n
        
    @property
    def energy_reactive(self):
        return self.H_reactive()*self.n
    
    @property    
    def Q(self):
        return self.m/self.rho_mass()

    @property
    def Qgs(self):
        # Always use flash settings - do not store weird input
        settings = self.flasher.settings
        V = R*settings.T_gas_ref/settings.P_gas_ref
        n = self.n
        Vn = V*n
        return [zi*Vn for zi in self.zs]

    @property
    def Qls(self):
        T_liquid_volume_ref = self.flasher.settings.T_liquid_volume_ref
        ns = self.ns
        Vms_TP = self.constants.Vml_STPs
        return [ns[i]*Vms_TP[i] for i in self.cmps]
        
    def StreamArgs(self):
        '''Goal to create a StreamArgs instance, with the user specified
        variables always being here.
        
        The state variables are currently correctly tracked. The flow rate and
        composition variable needs to be tracked as a function of what was
        specified as the input variables.
        
        The flow rate needs to be changed wen the stream flow rate is changed.
        Note this stores unnormalized specs, but that this is OK.
        '''
        kwargs = self.flash_specs.copy()
        del kwargs['zs']
        kwargs['pkg'] = self.flasher
        kwargs[self.composition_spec[0]] = self.composition_spec[1]
        kwargs[self.flow_spec[0]] = self.flow_spec[1]
        return StreamArgs(**kwargs)
    
    
    # flow_spec, composition_spec are attributes already
    @property
    def specified_composition_vars(self):
        '''number of composition variables'''
        return 1

    @property
    def composition_specified(self):
        '''Always needs a composition'''
        return True
    
    @property
    def specified_state_vars(self):
        '''Always needs two states'''
        return 2
    
    @property
    def non_pressure_spec_specified(self):
        '''Cannot have a stream without an energy-type spec.
        '''
        return True
    
    @property
    def state_specified(self):
        '''Always needs a state'''
        return True
    
    @property
    def state_specs(self):
        '''Returns a list of tuples of (state_variable, state_value) representing
        the thermodynamic state of the system.
        '''
        specs = []
        flash_specs = self.flash_specs
        for i, var in enumerate(('T', 'P', 'VF', 'H', 'S', 'energy')):
            if var in flash_specs:
                v = flash_specs[var]
                if v is not None:
                    specs.append((var, v))
        return specs

    @property
    def specified_flow_vars(self):
        '''Always needs only one flow specified'''
        return 1

    @property
    def flow_specified(self):
        '''Always needs a flow specified'''
        return True



energy_types = {'LP_STEAM': 'Steam 50 psi',
                'MP_STEAM': 'Steam 150 psi',
                'HP_STEAM': 'Steam 300 psi',
                'ELECTRICITY': 'Electricity',
                'AC_ELECTRICITY': 'AC Electricity',
                'DC_ELECTRICITY': 'DC Electricity'}
for freq in frequencies:
    for voltage in voltages_1_phase_residential:
        energy_types['AC_ELECTRICITY_1_PHASE_%s_V_%s_Hz'% (str(voltage), str(freq))] = 'AC_ELECTRICITY 1 PHASE %s V %s Hz'% (str(voltage), str(freq))
    for voltage in voltages_3_phase:
        energy_types['AC_ELECTRICITY_3_PHASE_%s_V_%s_Hz'% (str(voltage), str(freq))] = 'AC_ELECTRICITY 3 PHASE %s V %s Hz'%  (str(voltage), str(freq))

EnergyTypes = enum.Enum('EnergyTypes', energy_types)


class EnergyStream(object):
    '''
    '''
    Q = None
    medium = None
    def __repr__(self):
        return '<Energy stream, Q=%s W, medium=%s>' %(self.Q, self.medium.value)
    
    def __init__(self, Q, medium=EnergyTypes.ELECTRICITY):
        self.medium = medium
        # isinstance test is slow, especially with Number - faster to check float and int first
        if not (Q is None or isinstance(Q, (float, int, Number))):
            raise Exception('Energy stream flow rate is not a flow rate')
        self.Q = Q

