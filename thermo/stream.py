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

__all__ = ['Stream', 'EnergyTypes', 'EnergyStream', 'StreamArgs', 'EquilibriumStream', 'mole_balance', 'energy_balance']

#import enum
try:
    from collections import OrderedDict
except:
    pass

from fluids.constants import R
from chemicals.utils import property_molar_to_mass, property_mass_to_molar, solve_flow_composition_mix
from chemicals.exceptions import OverspeficiedError
from chemicals.volume import ideal_gas
from chemicals.utils import mixing_simple, normalize, Vfs_to_zs, ws_to_zs, zs_to_ws, Vm_to_rho, zs_to_Vfs
from thermo.mixture import Mixture, preprocess_mixture_composition
from thermo.equilibrium import EquilibriumState
from thermo.flash import Flash
from fluids.pump import voltages_1_phase_residential, voltages_3_phase, residential_power_frequencies



class StreamArgs(object):
    flashed = False
    _state_cache = None
    

    def __init__(self, zs=None, ws=None, Vfls=None, Vfgs=None,
                 T=None, P=None,
                 VF=None, H=None, H_mass=None, S=None, S_mass=None,
                 U=None, U_mass=None, G=None, G_mass=None, A=None, A_mass=None,
                 V=None, rho=None, rho_mass=None,
                 
                 ns=None, ms=None, Qls=None, Qgs=None, m=None, n=None, Q=None,
                 energy=None, energy_reactive=None, H_reactive=None,
                 Vf_TP=(None, None), Q_TP=(None, None, ''), flasher=None,
                 single_composition_basis=True):
        self.specifications = s = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None,
                       'ns': None, 'ms': None, 'Qls': None, 'Qgs': None,
                       'n': None, 'm': None, 'Q': None,
                       
                       'T': None, 'P': None, 
                       'V': None, 'rho': None, 'rho_mass': None,
                       
                       'VF': None, 
                       'H': None,'H_mass': None,
                       'S': None, 'S_mass': None,
                       'U': None, 'U_mass': None,
                       'A': None, 'A_mass': None,
                       'G': None, 'G_mass': None,
                       'energy': None, 'energy_reactive': None, 'H_reactive': None}

        # If this is False, DO NOT CLEAR OTHER COMPOSITION / FLOW VARIABLES WHEN SETTING ONE!
        # This makes sense for certain cases but not all.
        self.single_composition_basis = single_composition_basis
        self.Vf_TP = Vf_TP
        self.Q_TP = Q_TP
        self.flasher = flasher

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
        if V is not None:
            s['V'] = V
            state_specs += 1
        if rho is not None:
            s['rho'] = rho
            state_specs += 1
        if rho_mass is not None:
            s['rho_mass'] = rho_mass
            state_specs += 1
        if VF is not None:
            s['VF'] = VF
            state_specs += 1
        if H_mass is not None:
            s['H_mass'] = H_mass
            state_specs += 1
        if H is not None:
            s['H'] = H
            state_specs += 1
        if S_mass is not None:
            s['S_mass'] = S_mass
            state_specs += 1
        if S is not None:
            s['S'] = S
            state_specs += 1
        if U_mass is not None:
            s['U_mass'] = U_mass
            state_specs += 1
        if U is not None:
            s['U'] = U
            state_specs += 1
        if G_mass is not None:
            s['G_mass'] = G_mass
            state_specs += 1
        if G is not None:
            s['G'] = G
            state_specs += 1
        if A_mass is not None:
            s['A_mass'] = A_mass
            state_specs += 1
        if A is not None:
            s['A'] = A
            state_specs += 1
        if energy is not None:
            s['energy'] = energy
            state_specs += 1
        if energy_reactive is not None:
            s['energy_reactive'] = energy_reactive
            state_specs += 1
        if H_reactive is not None:
            s['H_reactive'] = H_reactive
            state_specs += 1
        if flow_specs > 1 or composition_specs > 1:
            self.reconcile_flows()
#            raise ValueError("Flow specification is overspecified")
        if composition_specs > 1 and single_composition_basis:
            raise ValueError("Composition specification is overspecified")
        if state_specs > 2:
            raise ValueError("State specification is overspecified")

    def __add__(self, b):
        if not isinstance(b, StreamArgs):
            raise TypeError('Adding to a StreamArgs requires that the other object '
                            'also be a StreamArgs.')

        a_flow_spec, b_flow_spec = self.flow_spec, b.flow_spec
        a_composition_spec, b_composition_spec = self.composition_spec, b.composition_spec
        a_state_specs, b_state_specs = self.state_specs, b.state_specs

        args = {}

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
        return StreamArgs(Vf_TP=self.Vf_TP, Q_TP=self.Q_TP, flasher=self.flasher,
                 single_composition_basis=self.single_composition_basis, **kwargs)

    __copy__ = copy

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
    def energy_reactive(self):
        return self.specifications['energy_reactive']
    @energy_reactive.setter
    def energy_reactive(self, energy_reactive):
        if energy_reactive is None:
            self.specifications['energy_reactive'] = energy_reactive
            return None
        if self.specified_state_vars > 1 and self.flow_specified and self.energy_reactive is None:
            raise Exception('Two state vars and a flow var already specified; unset another first')
        self.specifications['energy_reactive'] = energy_reactive

    @property
    def T(self):
        return self.specifications['T']
    @T.setter
    def T(self, T):
        s = self.specifications
        if T is None:
            s['T'] = T
            return None
        if s['T'] is None and self.state_specified:
            raise Exception('Two state vars already specified; unset another first')
        s['T'] = T

    @property
    def T_calc(self):
        T = self.specifications['T']
        if T is not None:
            return T
        try:
            return self.flash_state().T
        except:
            return None

    @property
    def P_calc(self):
        P = self.specifications['P']
        if P is not None:
            return P
        try:
            return self.flash_state().P
        except:
            return None

    @property
    def VF_calc(self):
        VF = self.specifications['VF']
        if VF is not None:
            return VF
        try:
            return self.flash_state().VF
        except:
            return None

    @property
    def P(self):
        return self.specifications['P']
    @P.setter
    def P(self, P):
        s = self.specifications
        if P is None:
            s['P'] = None
            return None
        if s['P'] is None and self.state_specified:
            raise Exception('Two state vars already specified; unset another first')
        s['P'] = P

    @property
    def V(self):
        return self.specifications['V']
    @V.setter
    def V(self, V):
        if V is None:
            self.specifications['V'] = V
            return None
        if self.state_specified and self.V is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['V'] = V

    @property
    def rho(self):
        return self.specifications['rho']
    @rho.setter
    def rho(self, rho):
        if rho is None:
            self.specifications['rho'] = rho
            return None
        if self.state_specified and self.rho is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['rho'] = rho

    @property
    def rho_mass(self):
        return self.specifications['rho_mass']
    @rho_mass.setter
    def rho_mass(self, rho_mass):
        if rho_mass is None:
            self.specifications['rho_mass'] = rho_mass
            return None
        if self.state_specified and self.rho_mass is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['rho_mass'] = rho_mass

    @property
    def VF(self):
        return self.specifications['VF']
    @VF.setter
    def VF(self, VF):
        if VF is None:
            self.specifications['VF'] = VF
            return None
        if self.state_specified and self.VF is None:
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
        if self.state_specified and self.H is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['H'] = H

    @property
    def H_calc(self):
        H = self.specifications['H']
        if H is not None:
            return H
        try:
            return self.flash_state().H()
        except:
            return None

    @property
    def H_reactive_calc(self):
        H_reactive = self.specifications['H_reactive']
        if H_reactive is not None:
            return H_reactive
        try:
            return self.flash_state().H_reactive()
        except:
            return None

    @property
    def H_mass(self):
        return self.specifications['H_mass']
    @H_mass.setter
    def H_mass(self, H_mass):
        if H_mass is None:
            self.specifications['H_mass'] = H_mass
            return None
        if self.specified_state_vars > 1 and self.H_mass is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['H_mass'] = H_mass

    @property
    def H_mass_calc(self):
        H_mass = self.specifications['H_mass']
        if H_mass is not None:
            return H_mass
        try:
            return self.flash_state().H_mass()
        except:
            return None

    @property
    def U(self):
        return self.specifications['U']
    @U.setter
    def U(self, U):
        if U is None:
            self.specifications['U'] = U
            return None
        if self.state_specified and self.U is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['U'] = U

    @property
    def U_mass(self):
        return self.specifications['U_mass']
    @U_mass.setter
    def U_mass(self, U_mass):
        if U_mass is None:
            self.specifications['U_mass'] = U_mass
            return None
        if self.specified_state_vars > 1 and self.U_mass is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['U_mass'] = U_mass

    @property
    def G(self):
        return self.specifications['G']
    @G.setter
    def G(self, G):
        if G is None:
            self.specifications['G'] = G
            return None
        if self.state_specified and self.G is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['G'] = G

    @property
    def G_mass(self):
        return self.specifications['G_mass']
    @G_mass.setter
    def G_mass(self, G_mass):
        if G_mass is None:
            self.specifications['G_mass'] = G_mass
            return None
        if self.specified_state_vars > 1 and self.G_mass is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['G_mass'] = G_mass

    @property
    def A(self):
        return self.specifications['A']
    @A.setter
    def A(self, A):
        if A is None:
            self.specifications['A'] = A
            return None
        if self.state_specified and self.A is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['A'] = A

    @property
    def A_mass(self):
        return self.specifications['A_mass']
    @A_mass.setter
    def A_mass(self, A_mass):
        if A_mass is None:
            self.specifications['A_mass'] = A_mass
            return None
        if self.specified_state_vars > 1 and self.A_mass is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['A_mass'] = A_mass

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
    def S_mass(self):
        return self.specifications['S_mass']
    @S_mass.setter
    def S_mass(self, S_mass):
        if S_mass is None:
            self.specifications['S_mass'] = S_mass
            return None
        if self.specified_state_vars > 1 and self.S_mass is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['S_mass'] = S_mass

    @property
    def H_reactive(self):
        return self.specifications['H_reactive']
    @H_reactive.setter
    def H_reactive(self, H_reactive):
        if H_reactive is None:
            self.specifications['H_reactive'] = H_reactive
            return None
        if self.state_specified and self.H_reactive is None:
            raise Exception('Two state vars already specified; unset another first')
        self.specifications['H_reactive'] = H_reactive
    @property
    def zs(self):
        return self.specifications['zs']
    @zs.setter
    def zs(self, arg):
        s = self.specifications
        if arg is None:
            s['zs'] = arg
        else:
            # enforce a length
            if not arg:
                arg = None
            if self.single_composition_basis:
                s['zs'] = arg
                s['ws'] = s['Vfls'] = s['Vfgs'] = s['ns'] = s['ms'] = s['Qls'] = s['Qgs'] = None
            else:
                s['zs'] = arg

    @property
    def zs_calc(self):
        '''
        '''
        # This forms the basis for the calculations
        s = self.specifications
        zs = s['zs']
        if zs is not None:
            if self.single_composition_basis:
                return zs
            else:
                if None not in zs:
                    return zs
                return None
        ns = s['ns']
        if ns is not None:
            if self.single_composition_basis:
                return normalize(ns)

        ws = s['ws']
        if ws is not None and None not in ws:
            MWs = self.flasher.constants.MWs
            try:
                return ws_to_zs(ws, MWs)
            except ZeroDivisionError:
                pass
        Vfls = s['Vfls']
        if Vfls is not None and None not in Vfls:
            Vms = self.flasher.V_liquids_ref()
            try:
                return Vfs_to_zs(Vfls, Vms)
            except ZeroDivisionError:
                pass


        ms = s['ms']
        if ms is not None and None not in ms:
            MWs = self.flasher.constants.MWs
            return ws_to_zs(normalize(ms), MWs)

        Qls = s['Qls']
        if Qls is not None and None not in Qls:
            Vms = self.flasher.V_liquids_ref()
            return Vfs_to_zs(normalize(Qls), Vms)
        
        Qgs = s['Qgs']
        if Qgs is not None and None not in Qgs:
            return normalize(Qgs)
            

        return None

    @property
    def ws(self):
        return self.specifications['ws']
    @ws.setter
    def ws(self, arg):
        if arg is None:
            self.specifications['ws'] = arg
        else:
            # enforce a length
            if not arg:
                arg = None
            if self.single_composition_basis:
                args = {'zs': None, 'ws': arg, 'Vfls': None, 'Vfgs': None, 'ns': None,
                        'ms': None, 'Qls': None, 'Qgs': None}
                self.specifications.update(args)
            else:
                self.specifications['ws'] = arg
                
    @property
    def ws_calc(self):
        ws = self.specifications['ws']
        if ws is not None:
            return ws
        zs = self.zs_calc
        if zs is not None:
            MWs = self.flasher.constants.MWs
            return zs_to_ws(zs, MWs)

    @property
    def Vfls(self):
        return self.specifications['Vfls']
    @Vfls.setter
    def Vfls(self, arg):
        if arg is None:
            self.specifications['Vfls'] = arg
        else:
            # enforce a length
            if not arg:
                arg = None
            if self.single_composition_basis:
                args = {'zs': None, 'ws': None, 'Vfls': arg, 'Vfgs': None, 'ns': None,
                        'ms': None, 'Qls': None, 'Qgs': None}
                self.specifications.update(args)
            else:
                self.specifications['Vfls'] = arg

                
    @property
    def Vfls_calc(self):
        Vfls = self.specifications['Vfls']
        if Vfls is not None:
            return Vfls
        zs = self.zs_calc
        if zs is not None:
            Vms = self.flasher.V_liquids_ref()
            return zs_to_Vfs(zs, Vms)
    

    @property
    def Vfgs(self):
        return self.specifications['Vfgs']
    @Vfgs.setter
    def Vfgs(self, arg):
        if arg is None:
            self.specifications['Vfgs'] = arg
        else:
            # enforce a length
            if not arg:
                arg = None
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
        s = self.specifications
        if arg is None:
            s['ns'] = arg
        else:
            if self.single_composition_basis:
                s['zs'] = s['ws'] = s['Vfls'] = s['Vfgs'] = s['ms'] = s['Qls'] = s['Qgs'] = s['n'] = s['m'] = s['Q'] = None
                s['ns'] = arg
            else:
                s['ns'] = arg

    @property
    def ns_calc(self):
        s = self.specifications
        ns = s['ns']
        if ns is not None:
            if self.single_composition_basis:
                return ns
            else:
                if None not in ns:
                    return ns
                return None
        n = s['n']
        if n is not None:
            zs = self.zs_calc
            if zs is not None:
                return [n*zi for zi in zs]
        m = s['m']
        if m is not None:
            zs = self.zs_calc
            try:
                MWs = self.flasher.constants.MWs
                MW = mixing_simple(MWs, zs)
                n = property_molar_to_mass(m, MW)
                return [n*zi for zi in zs]
            except:
                pass
        ms = s['ms']
        if ms is not None and None not in ms:
            zs = self.zs_calc
            m = sum(ms)
            MWs = self.flasher.constants.MWs
            MW = mixing_simple(MWs, zs)
            n = m*1000.0/MW
            return [n*zi for zi in zs]
        Qls = s['Qls']
        if Qls is not None and None not in Qls:
            Vms = self.flasher.V_liquids_ref()
            return [Ql/Vm for Vm, Ql in zip(Vms, Qls)]
        Qgs = s['Qgs']
        if Qgs is not None and None not in Qgs:
            flasher = self.flasher
            V = R*flasher.settings.T_gas_ref/flasher.settings.P_gas_ref
            return [Qgi/V for Qgi in Qgs]
        Q = s['Q']
        if Q is not None:
            zs = self.zs_calc
            if zs is not None:
                Q_TP = self.Q_TP
                if Q_TP is not None:
                    if len(Q_TP) == 2 or (len(Q_TP) == 3 and not Q_TP[-1]):
                        # Calculate the volume via the property package
                        expensive_flash = self.flasher.flash(zs=zs, T=Q_TP[0], P=Q_TP[1])
                        V = expensive_flash.V()
                    if Q_TP[-1] == 'l':
                        V = self.flasher.liquids[0].to(T=Q_TP[0], P=Q_TP[1], zs=zs).V()
                    elif Q_TP[-1] == 'g':
                        V = self.flasher.gas.to(T=Q_TP[0], P=Q_TP[1], zs=zs).V()
                else:
                    mixture = self.flash_state()
                    if mixture is not None:
                        V = mixture.V()
                if V is not None:
                    n = Q/V
                    return [n*zi for zi in zs]
        return None

    @property
    def ms(self):
        return self.specifications['ms']
    @ms.setter
    def ms(self, arg):
        s = self.specifications
        if arg is None:
            s['ms'] = arg
        else:
            if self.single_composition_basis:
                s['zs'] = s['ws'] = s['Vfls'] = s['Vfgs'] = s['ns'] = s['Qls'] = s['Qgs'] = s['n'] = s['m'] = s['Q'] = None
                s['ms'] = arg
            else:
                s['ms'] = arg

    @property
    def ms_calc(self):
        ns = self.ns_calc
        if ns is not None:
            zs = self.zs_calc
            n = sum(ns)
            MW = self.MW
            m = property_mass_to_molar(n, MW)
            MW_inv = 1.0/MW
            MWs = self.flasher.constants.MWs
            ws = [zi*MWi*MW_inv for zi, MWi in zip(zs, MWs)]
            return [m*wi for wi in ws]

    @property
    def Qls(self):
        return self.specifications['Qls']
    @Qls.setter
    def Qls(self, arg):
        s = self.specifications
        if arg is None:
            s['Qls'] = arg
        else:
            if self.single_composition_basis:
                s['zs'] = s['ws'] = s['Vfls'] = s['Vfgs'] = s['ms'] = s['ns'] = s['Qgs'] = s['n'] = s['m'] = s['Q'] = None
                s['Qls'] = arg
            else:
                s['Qls'] = arg
    
    @property
    def Qls_calc(self):
        ns_calc = self.ns_calc
        if ns_calc is not None:
            Vms = self.flasher.V_liquids_ref()
            Qls = [ni*Vm for ni, Vm in zip(ns_calc, Vms)]
            return Qls

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
    def Qgs_calc(self):
        ns_calc = self.ns_calc
        if ns_calc is not None:
            flasher = self.flasher
            V = R*flasher.settings.T_gas_ref/flasher.settings.P_gas_ref
            Qgs = [ni*V for ni in ns_calc]
            return Qgs

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
        s = self.specifications
        if arg is None:
            s['n'] = None
        else:
            s['n'] = arg
            s['ns'] = s['ms'] = s['Qls'] = s['Qgs'] = s['m'] = s['Q'] = None

    @property
    def n_calc(self):
        s = self.specifications
        n = s['n']
        if n is not None:
            return n
        ns = s['ns']
        if ns is not None and None not in ns:
            return sum(ns)
        # Everything funnels into ns_calc to avoid conflicts
        ns_calc = self.ns_calc
        if ns_calc is not None and None not in ns_calc:
            return sum(ns_calc)
    
        return None

    @property
    def MW(self):
        try:
            MWs = self.flasher.constants.MWs
            zs = self.zs_calc
            MW = mixing_simple(MWs, zs)
            return MW
        except:
            return None

    @property
    def m_calc(self):
        m = self.specifications['m']
        if m is not None:
            return m
        ms = self.specifications['ms']
        if ms is not None and None not in ms:
            return sum(ms)
        ms_calc = self.ms_calc
        if ms_calc is not None:
            return sum(ms_calc)
        return None

    @property
    def energy_calc(self):
        s = self.specifications
        # Try to get H from energy, or a molar specification
        Q = s['energy']
        m, n = None, None
        if Q is None:
            H = s['H']
            if H is not None:
                n = s['n']
                if n is None:
                    n = self.n_calc
                if n is not None:
                    Q = n*H
        # Try to get H from a mass specification
        if Q is None:
            H_mass = s['H_mass']
            if H_mass is not None:
                m = s['m']
                if m is None:
                    m = self.m_calc
                if m is not None:
                    Q = m*H_mass
        # Try to flash and get enthalpy

        if Q is None:
            n = self.n_calc
            if n is None:
                m = self.m_calc
            if m is not None or n is not None:
                mixture = self.flash_state()
                if mixture is not None:
                    if n is not None:
                        Q = mixture.H()*n
                    elif m is not None:
                        Q = mixture.H()*property_molar_to_mass(m, mixture.MW())
        return Q
    
    @property
    def energy_reactive_calc(self):
        return self.specifications['energy_reactive']



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
        s = '%s(flasher=%s, ' %(self.__class__.__name__, self.flasher is not None)
        for k, v in self.specifications.items():
            if v is not None:
                s += '%s=%s, ' %(k, repr(v))
        s = s[:-2]
        s += ')'
        return s

    def reconcile_flows(self, n_tol=2e-15, m_tol=2e-15):
        s = self.specifications
        n, m, Q = s['n'], s['m'], s['Q']
        if n is not None:
            if m is not None:
                raise OverspeficiedError("Flow specification is overspecified: n=%g, m=%g" %(n, m))
            elif Q is not None:
                raise OverspeficiedError("Flow specification is overspecified: n=%g, Q=%g" %(n, Q))
        elif m is not None and Q is not None:
            raise OverspeficiedError("Flow specification is overspecified: m=%g, Q=%g" %(m, Q))

        ns, zs, ms, ws = s['ns'], s['zs'], s['ms'], s['ws']
        if n is not None and ns is not None:
            calc = 0.0
            missing = 0
            missing_idx = None
            for i in range(len(ns)):
                if ns[i] is None:
                    missing += 1
                    missing_idx = i
                else:
                    calc += ns[i]
            if missing == 0:
                if abs((calc - n)/n) > n_tol:
                    raise ValueError("Flow specification is overspecified and inconsistent")
            elif missing == 1:
                ns[missing_idx] = n - calc

        if m is not None and ms is not None:
            calc = 0.0
            missing = 0
            missing_idx = None
            for i in range(len(ms)):
                if ms[i] is None:
                    missing += 1
                    missing_idx = i
                else:
                    calc += ms[i]
            if missing == 0:
                if abs((calc - m)/m) > m_tol:
                    raise ValueError("Flow specification is overspecified and inconsistent")
            elif missing == 1:
                ms[missing_idx] = m - calc
        if ns is not None and ms is not None:
            try:
            # Convert any ms to ns
                MWs = self.flasher.constants.MWs
            except:
                return False
            for i in range(len(ms)):
                if ms[i] is not None:
                    ni = property_molar_to_mass(ms[i], MWs[i])
                    if ns[i] is not None and abs((ns[i]  - ni)/ni) > n_tol:
                        raise ValueError("Flow specification is overspecified and inconsistent on component %d" %i)
                    else:
                        ns[i] = ni

        if (zs is not None or ns is not None) and (ws is not None or ms is not None) and (m is not None or n is not None or ns is not None or ms is not None):
            # We need the MWs
            try:
                MWs = self.flasher.constants.MWs
                if zs is None:
                    zs = [None]*len(MWs)
                if ws is None:
                    ws = [None]*len(MWs)
                ns, zs, ws = solve_flow_composition_mix(ns, zs, ws, MWs)
                s['ns'] = ns
            except:
                return False



    @property
    def composition_spec(self):
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

    @property
    def clean(self):
        '''If no variables have been specified, return True,
        otherwise return False.
        '''
        if self.composition_specified or self.state_specs or self.flow_specified:
            return False
        return True


    @property
    def composition_specified(self):
        s = self.specifications
        if s['zs'] is not None and None not in s['zs'] and sum(s['zs']) != 0.0:
            return True
        if s['ws'] is not None and None not in s['ws'] and sum(s['ws']) != 0.0:
            return True
        if s['Vfls'] is not None and None not in s['Vfls'] and sum(s['Vfls']) != 0.0:
            return True
        if s['Vfgs'] is not None and None not in s['Vfgs'] and sum(s['Vfgs']) != 0.0:
            return True
        if s['ns'] is not None and None not in s['ns']:
            return True
        if s['ms'] is not None and None not in s['ms']:
            return True
        if s['Qls'] is not None and None not in s['Qls']:
            return True
        if s['Qgs'] is not None and None not in s['Qgs']:
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
        if s['V'] is not None:
            specs.append(('V', s['V']))
        if s['rho'] is not None:
            specs.append(('rho', s['rho']))
        if s['rho_mass'] is not None:
            specs.append(('rho_mass', s['rho_mass']))
        if s['VF'] is not None:
            specs.append(('VF', s['VF']))
        if s['H_mass'] is not None:
            specs.append(('H_mass', s['H_mass']))
        if s['H'] is not None:
            specs.append(('H', s['H']))
        if s['S_mass'] is not None:
            specs.append(('S_mass', s['S_mass']))
        if s['S'] is not None:
            specs.append(('S', s['S']))
        if s['U_mass'] is not None:
            specs.append(('U_mass', s['U_mass']))
        if s['U'] is not None:
            specs.append(('U', s['U']))
        if s['G_mass'] is not None:
            specs.append(('G_mass', s['G_mass']))
        if s['G'] is not None:
            specs.append(('G', s['G']))
        if s['A_mass'] is not None:
            specs.append(('A_mass', s['A_mass']))
        if s['A'] is not None:
            specs.append(('A', s['A']))
        if s['energy'] is not None:
            specs.append(('energy', s['energy']))
        if s['energy_reactive'] is not None:
            specs.append(('energy_reactive', s['energy_reactive']))
#        for var in ('T', 'P', 'VF', 'Hm', 'H', 'Sm', 'S', 'energy'):
#            v = getattr(self, var)
#            if v is not None:
#                specs.append((var, v))
        return specs

    @property
    def specified_state_vars(self):
        # Slightly faster
        s = self.specifications
        return sum(s[i] is not None for i in ('T', 'P', 'V', 'rho', 'rho_mass', 
                                                                'VF', 'H_mass', 'H', 'S_mass', 'S',
                                                                'U_mass', 'U', 'A_mass', 'A', 'G_mass', 'G',
                                                                'energy', 'energy_reactive'))
#        return sum(i is not None for i in (self.T, self.P, self.VF, self.Hm, self.H, self.Sm, self.S, self.energy))

    @property
    def state_specified(self):
        s = self.specifications

        state_vars = 0
        if s['T'] is not None:
            state_vars += 1
        if s['P'] is not None:
            state_vars += 1
        if s['V'] is not None:
            state_vars += 1
        if s['rho'] is not None:
            state_vars += 1
        if s['rho_mass'] is not None:
            state_vars += 1
        if s['VF'] is not None:
            state_vars += 1
        if s['H_mass'] is not None:
            state_vars += 1
        if s['H'] is not None:
            state_vars += 1
        if s['S'] is not None:
            state_vars += 1
        if s['S_mass'] is not None:
            state_vars += 1
        if s['U'] is not None:
            state_vars += 1
        if s['U_mass'] is not None:
            state_vars += 1
        if s['G'] is not None:
            state_vars += 1
        if s['G_mass'] is not None:
            state_vars += 1
        if s['A'] is not None:
            state_vars += 1
        if s['A_mass'] is not None:
            state_vars += 1
        if s['energy'] is not None:
            state_vars += 1
        if s['energy_reactive'] is not None:
            state_vars += 1
        if s['H_reactive'] is not None:
            state_vars += 1
        if state_vars == 2:
            return True
        return False

    @property
    def non_pressure_spec_specified(self):
        state_vars = (i is not None for i in (self.T, self.VF, self.V, self.rho, self.rho_mass, 
                                              self.H_mass, self.H, self.S_mass, self.S, 
                                              self.U_mass, self.U, self.A_mass, self.A, self.G_mass, self.G,
                                              self.energy, self.energy_reactive, self.H_reactive))
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
        s = self.specifications
        return EquilibriumStream(self.flasher, hot_start=hot_start,
                                 existing_flash=existing_flash, **s)

    @property
    def stream(self):
        if self.flow_specified and self.composition_specified and self.state_specified:
            s = self.specifications.copy()
            return EquilibriumStream(self.flasher, **s)
    def flash_state(self, hot_start=None):
        if self.composition_specified and self.state_specified:
            s = self.specifications
            # Flash call only takes `zs`
            zs = self.zs_calc
            T, P, VF = s['T'], s['P'], s['VF']
            H = H_reactive = None
            # Do we need
            spec_count = 0
            if T is not None:
                spec_count += 1
            if P is not None:
                spec_count += 1
            if s['V'] is not None:
                spec_count += 1
            if s['rho'] is not None:
                spec_count += 1
            if s['rho_mass'] is not None:
                spec_count += 1
            if s['H_mass'] is not None:
                spec_count += 1
            if s['H'] is not None:
                spec_count += 1
            if s['S_mass'] is not None:
                spec_count += 1
            if s['S'] is not None:
                spec_count += 1
            if s['U_mass'] is not None:
                spec_count += 1
            if s['U'] is not None:
                spec_count += 1
            if s['G_mass'] is not None:
                spec_count += 1
            if s['G'] is not None:
                spec_count += 1
            if s['A_mass'] is not None:
                spec_count += 1
            if s['A'] is not None:
                spec_count += 1
            if s['H_reactive'] is not None:
                spec_count += 1
            if VF is not None:
                spec_count += 1
                
            if spec_count < 2:
                energy = s['energy']
                if energy is not None:
                    n = self.n_calc
                    if n is not None:
                        H = energy/n
                        spec_count += 1
                energy_reactive = s['energy_reactive']
                if energy_reactive is not None:
                    n = self.n_calc
                    if n is not None:
                        H_reactive = energy_reactive/n
                        spec_count += 1
            
            H_flash = s['H'] if s['H'] is not None else H
            H_reactive = s['H_reactive'] if s['H_reactive'] is not None else H_reactive
            state_cache = (T, P, VF, s['S_mass'], s['S'], s['H'], H_flash, s['H_reactive'], H_reactive,
                           s['U'], s['U_mass'], s['G'], s['G_mass'], s['A'], s['A_mass'], 
                           s['V'], s['rho'], s['rho_mass'], tuple(zs))
            if state_cache == self._state_cache:
                try:
                    return self._mixture
                except:
                    pass

            m = self.flasher.flash(T=T, P=P, zs=zs, H=H_flash, H_mass=s['H_mass'],
                                            S=s['S'], S_mass=s['S_mass'],
                                            U=s['U'], U_mass=s['U_mass'],
                                            G=s['G'], G_mass=s['G_mass'],
                                            A=s['A'], A_mass=s['A_mass'],
                                            V=s['V'], rho=s['rho'], rho_mass=s['rho_mass'],
                                            H_reactive=H_reactive,
                                            VF=VF, hot_start=hot_start)
            self._mixture = m
            self._state_cache = state_cache
            return m

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


    A stream of vodka with volume fractions 60% water, 40% ethanol, 1 kg/s:

    >>> from thermo import Stream
    >>> Stream(['water', 'ethanol'], Vfls=[.6, .4], T=300, P=1E5, m=1)
    <Stream, components=['water', 'ethanol'], mole fractions=[0.8299, 0.1701], mass flow=1.0 kg/s, mole flow=43.883974 mol/s, T=300.00 K, P=100000 Pa>

    A stream of air at 400 K and 1 bar, flow rate of 1 mol/s:

    >>> Stream('air', T=400, P=1e5, n=1)
    <Stream, components=['nitrogen', 'argon', 'oxygen'], mole fractions=[0.7812, 0.0092, 0.2096], mass flow=0.028958 kg/s, mole flow=1 mol/s, T=400.00 K, P=100000 Pa>

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

    Notes
    -----

    .. warning::
        The Stream class is not designed for high-performance or the ability
        to use different thermodynamic models. It is especially limited in its
        multiphase support and the ability to solve with specifications other
        than temperature and pressure. It is impossible to change constant
        properties such as a compound's critical temperature in this interface.

        It is recommended to switch over to the :obj:`thermo.flash` and
        :obj:`EquilibriumStream` interfaces
        which solves those problems and are better positioned to grow. That
        interface also requires users to be responsible for their chemical
        constants and pure component correlations; while default values can
        easily be loaded for most compounds, the user is ultimately responsible
        for them.

    '''
    flashed = True
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
                self.n = sum([Q/Vmg for Q, Vmg in zip(Qgs, [ideal_gas(T, P)]*self.N)])
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
            self.Qgs = [m/Vm_to_rho(ideal_gas(self.T, self.P), MW=MW) for m, MW in zip(self.ms, self.MWs)]
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

    # def __repr__(self):
    #     s = '%s(T=%s, P=%s, zs=%s, betas=%s, n=%s' %(self.__class__.__name__, self.T, self.P, self.zs, self.betas, self.n)
    #     s += ', gas=%s' %(self.gas.__repr__())
    #     s += ', liquids=%s' %(self.liquids)
    #     s += ', solids=%s' %(self.solids)
    #     s += ')'
    #     return s

    def __repr__(self):
        s = '<EquilibriumStream, T=%.4f, P=%.4f, zs=%s, betas=%s, mass flow=%s kg/s, mole flow=%s mol/s, phases=%s>'
        s = s %(self.T, self.P, self.zs, self.betas, self.m, self.n, self.phases)
        return s

    def __copy__(self):
        # immutable
        return self

    def __init__(self, flasher, zs=None, ws=None, Vfls=None, Vfgs=None,
                 ns=None, ms=None, Qls=None, Qgs=None,
                 n=None, m=None, Q=None,
                 T=None, P=None, 
                 
                 V=None, rho=None, rho_mass=None,
                 
                 VF=None,
                 H=None, H_mass=None,
                 S=None, S_mass=None,
                 U=None, U_mass=None,
                 G=None, G_mass=None,
                 A=None, A_mass=None,
                 energy=None, energy_reactive=None, H_reactive=None,
                 Vf_TP=None, Q_TP=None, hot_start=None,
                 existing_flash=None, spec_fun=None):

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
                            "'m', 'n', 'Q', 'ms', 'ns', 'Qls', 'Qgs' "
                            "'energy' or 'energy_reactive' must be specified")
        elif flow_option_count > 1:
            raise Exception("More than one source of flow rate information is "
                            "provided; only one of "
                            "'m', 'n', 'Q', 'ms', 'ns', 'Qls', 'Qgs' "
                            "'energy' or 'energy_reactive' can be specified")

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
                T_vf, P_vf = Vf_TP
                Vms_TP = [i(T_vf, P_vf) for i in VolumeObjects]
            else:
                Vms_TP = constants.Vml_STPs
            zs = Vfs_to_zs(Vfls, Vms_TP)
        elif Qgs is not None:
            zs = normalize(Qgs)
        elif Vfgs is not None:
            zs = Vfgs

        MW = 0.0
        N = constants.N
        MWs = constants.MWs
        for i in range(N):
            MW  += zs[i]*MWs[i]
        self._MW = MW
        MW_inv = 1.0/MW

        if energy is not None:
            # Handle the various mole flows - converting to get energy; subset for now
            if m is not None:
                n = property_molar_to_mass(m, MW)  # m*10000/MW
            elif ns is not None:
                n = sum(ns)
            elif ms is not None:
                n = property_molar_to_mass(sum(ms), MW)
            H = energy/n
        elif energy_reactive is not None:
            if m is not None:
                n = property_molar_to_mass(m, MW)  # m*10000/MW
            elif ns is not None:
                n = sum(ns)
            elif ms is not None:
                n = property_molar_to_mass(sum(ms), MW)
            H_reactive = energy/n

        if existing_flash is not None:
            # All variable which have been set
            if type(existing_flash) is EquilibriumStream:
                composition_spec, flow_spec = self.composition_spec, self.flow_spec
            self.__dict__.update(existing_flash.__dict__)
            if type(existing_flash) is EquilibriumStream:
                self.composition_spec, self.flow_spec = composition_spec, flow_spec
                # TODO: are any variables caried over from an existing equilibrium stream?
                # Delete if so

        else:
            dest = super(EquilibriumStream, self).__init__
            flasher.flash(T=T, P=P, V=V, rho=rho, rho_mass=rho_mass, VF=VF,
                          H=H, H_mass=H_mass, S=S, S_mass=S_mass,
                          G=G, G_mass=G_mass, U=U, U_mass=U_mass,
                          A=A, A_mass=A_mass, H_reactive=H_reactive,
                          
                          dest=dest, zs=zs, hot_start=hot_start,
                          spec_fun=spec_fun)

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
            Vms = flasher.V_liquids_ref()
            try:
                for i in range(N):
                    n += Qls[i]/Vms[i]
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
    def T_calc(self):
        return self.T

    @property
    def P_calc(self):
        return self.P

    @property
    def VF_calc(self):
        return self.VF

    @property
    def zs_calc(self):
        return self.zs

    @property
    def ns_calc(self):
        return self.ns

    @property
    def ms_calc(self):
        return self.ms

    @property
    def m_calc(self):
        return self.m

    @property
    def n_calc(self):
        return self.n

    @property
    def energy(self):
        return self.H()*self.n

    energy_calc = energy

    @property
    def energy_reactive(self):
        return self.H_reactive()*self.n

    energy_reactive_calc = energy_reactive

    @property
    def pkg(self):
        return self.flasher

    @property
    def Q(self):
        return self.n*self.V()

    Q_calc = Q

    @property
    def Qgs(self):
        # Always use flash settings - do not store weird input
        settings = self.flasher.settings
        V = R*settings.T_gas_ref/settings.P_gas_ref
        n = self.n
        Vn = V*n
        return [zi*Vn for zi in self.zs]

    Qgs_calc = Qgs

    @property
    def Qls(self):
        T_liquid_volume_ref = self.flasher.settings.T_liquid_volume_ref
        ns = self.ns
        Vms_TP = self.constants.Vml_STPs
        return [ns[i]*Vms_TP[i] for i in range(self.N)]

    Qls_calc = Qls

    @property
    def Q_liquid_ref(self):
        return sum(self.Qls)

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
for freq in residential_power_frequencies:
    for voltage in voltages_1_phase_residential:
        energy_types['AC_ELECTRICITY_1_PHASE_%s_V_%s_Hz'% (str(voltage), str(freq))] = 'AC_ELECTRICITY 1 PHASE %s V %s Hz'% (str(voltage), str(freq))
    for voltage in voltages_3_phase:
        energy_types['AC_ELECTRICITY_3_PHASE_%s_V_%s_Hz'% (str(voltage), str(freq))] = 'AC_ELECTRICITY 3 PHASE %s V %s Hz'%  (str(voltage), str(freq))

try:
    EnergyTypes = enum.Enum('EnergyTypes', energy_types)
except:
    EnergyTypes = ''


class EnergyStream(object):
    '''
    '''
    Q = None
    medium = None
    Hm = None

    def copy(self):
        return EnergyStream(Q=self.Q, medium=self.medium)
    
    __copy__ = copy

    def __repr__(self):
        try:
            medium = self.medium.value
        except:
            medium = self.medium
        return '<Energy stream, Q=%s W, medium=%s>' %(self.Q, medium)

    def __init__(self, Q, medium=None):
        self.medium = medium
        # isinstance test is slow, especially with Number - faster to check float and int first
        if not (Q is None or isinstance(Q, (float, int))):
            raise Exception('Energy stream flow rate is not a flow rate')
        self.Q = Q

    @property
    def energy(self):
        return self.Q

    @energy.setter
    def energy(self, energy):
        self.Q = energy

    energy_calc = energy


def _mole_balance_process_ns(f, ns, compounds, use_mass=True, use_volume=True):
    if use_mass:
        ms = f.specifications['ms']
        if ms is not None and any(v is not None for v in ms):
            MWs = f.flasher.constants.MWs
            if ns is None:
                ns = [None]*compounds
            else:
                ns = list(ns)
            for i in range(compounds):
                if ns[i] is None and ms[i] is not None:
                    ns[i] = property_molar_to_mass(ms[i], MWs[i])
    if use_volume:
        Qls = f.specifications['Qls']
        if Qls is not None and any(v is not None for v in Qls):
            Vms = f.flasher.V_liquids_ref()
            if ns is None:
                ns = [None]*compounds
            else:
                ns = list(ns)
            for i in range(compounds):
                if ns[i] is None and Qls[i] is not None:
                    ns[i] = Qls[i]/Vms[i]
        Qgs = f.specifications['Qgs']
        if Qgs is not None and any(v is not None for v in Qgs):
            Vm = R*f.flasher.settings.T_gas_ref/f.flasher.settings.P_gas_ref
            if ns is None:
                ns = [None]*compounds
            else:
                ns = list(ns)
            for i in range(compounds):
                if ns[i] is None and Qgs[i] is not None:
                    ns[i] = Qgs[i]/Vm
    return ns

def mole_balance(inlets, outlets, compounds, use_mass=True, use_volume=True):
    inlet_count = len(inlets)
    outlet_count = len(outlets)

    in_unknown_count = out_unknown_count = 0
    in_unknown_idx = out_unknown_idx = None
    all_ns_in, all_ns_out = [], []
    all_in_known = all_out_known = True

    for i in range(inlet_count):
        f = inlets[i]
        try:
            ns = f.specifications['ns']
        except:
            ns = f.ns
        if ns is None and use_mass:
            ns = f.ns_calc
        if ns is None or None in ns:
            if use_mass:
                ns = _mole_balance_process_ns(f, ns, compounds, use_mass, use_volume)
            all_in_known = False
            in_unknown_count += 1
            in_unknown_idx = i
        all_ns_in.append(ns)

    for i in range(outlet_count):
        f = outlets[i]
        try:
            ns = f.specifications['ns']
        except:
            ns = f.ns
        if ns is None and use_mass:
            ns = f.ns_calc
        if ns is None or None in ns:
            if use_mass:
                ns = _mole_balance_process_ns(f, ns, compounds, use_mass, use_volume)
            all_out_known = False
            out_unknown_count += 1
            out_unknown_idx = i
        all_ns_out.append(ns)

    if all_out_known and all_in_known:
        # Fast path - all known
        return False

    if all_in_known:
        inlet_ns = [] # List of all molar flows in; set only when everything in is known
        for j in range(compounds):
            v = 0.0
            for i in range(inlet_count):
                v += all_ns_in[i][j]
            inlet_ns.append(v)

    if all_out_known:
        outlet_ns = []
        for j in range(compounds):
            v = 0.0
            for i in range(outlet_count):
                v += all_ns_out[i][j]
            outlet_ns.append(v)

    if out_unknown_count == 1 and in_unknown_count == 0:
        if outlet_count == 1:
            out_ns_calc = [i for i in inlet_ns]
        else:
            out_ns_calc = [inlet_ns[i] - sum(all_ns_out[j][i] for j in range(outlet_count) if (all_ns_out[j] and all_ns_out[j][i] is not None))
                           for i in range(compounds)]

        outlets[out_unknown_idx].ns = out_ns_calc
        return True
    if in_unknown_count == 1 and out_unknown_count == 0:
        if inlet_count == 1:
            in_ns_calc = [i for i in outlet_ns]
        else:
            in_ns_calc = [outlet_ns[i] - sum(all_ns_in[j][i] for j in range(inlet_count) if (all_ns_in[j] and all_ns_in[j][i] is not None))
                           for i in range(compounds)]

        inlets[in_unknown_idx].ns = in_ns_calc
        return True
    elif in_unknown_count == 0 and out_unknown_count == 0:
        return False # Nothing to do - everything is known

    progress = False
    # For each component, see if only one stream is missing it
    for j in range(compounds):
        in_missing, idx_missing = None, None
        missing_count = 0
        v = 0
        for i in range(inlet_count):
            ns = all_ns_in[i]
            if ns is None or ns[j] is None:
                missing_count += 1
                in_missing, idx_missing = True, i
            else:
                v += ns[j]
        for i in range(outlet_count):
            ns = all_ns_out[i]
            if ns is None or ns[j] is None:
                missing_count += 1
                in_missing, idx_missing = False, i
            else:
                v -= ns[j]
        if missing_count == 1:
            progress = True
            if in_missing:
                set_to_ns = inlets[idx_missing].specifications['ns']
                if set_to_ns is not None:
                    set_to_ns[j] = -v
                else:
                    set_to_ms = inlets[idx_missing].specifications['ms']
                    if set_to_ms is not None:
                        set_to_ms[j] = property_mass_to_molar(-v, inlets[idx_missing].flasher.constants.MWs[j])
                    else:
                        set_to_Qls = inlets[idx_missing].specifications['Qls']
                        if set_to_Qls is not None:
                            Vms = inlets[idx_missing].flasher.V_liquids_ref()
                            set_to_Qls[j] = -v*Vms[j]
                        else:
                            set_to_Qgs = inlets[idx_missing].specifications['Qgs']
                            if set_to_Qgs is not None:
                                Vm = R*inlets[idx_missing].flasher.settings.T_gas_ref/inlets[idx_missing].flasher.settings.P_gas_ref
                                set_to_Qgs[j] = -v*Vm
            else:
                set_to_ns = outlets[idx_missing].specifications['ns']
                if set_to_ns is not None:
                    set_to_ns[j] = v
                else:
                    set_to_ms = outlets[idx_missing].specifications['ms']
                    if set_to_ms is not None:
                        set_to_ms[j] = property_mass_to_molar(v, outlets[idx_missing].flasher.constants.MWs[j])
                    else:
                        set_to_Qls = outlets[idx_missing].specifications['Qls']
                        if set_to_Qls is not None:
                            Vms = outlets[idx_missing].flasher.V_liquids_ref()
                            set_to_Qls[j] = v*Vms[j]
                        else:
                            set_to_Qgs = outlets[idx_missing].specifications['Qgs']
                            if set_to_Qgs is not None:
                                Vm = R*outlets[idx_missing].flasher.settings.T_gas_ref/outlets[idx_missing].flasher.settings.P_gas_ref
                                set_to_Qgs[j] = v*Vm
    if progress:
        return progress

    # Try a total mole balance
    n_in_missing_count = 0
    if all_in_known:
        n_in = sum(inlet_ns)
    else:
        n_in_missing_idx = None
        n_in = 0.0
        for i in range(inlet_count):
            f = inlets[i]
            n = f.specifications['n']
            if n is None:
                n = f.n_calc
            if n is None:
                n_in_missing_count += 1
                n_in_missing_idx = i
            else:
                n_in += n

    n_out_missing_count = 0
    if all_out_known:
        n_out = sum(outlet_ns)
    else:
        n_out_missing_idx = None
        n_out = 0.0
        for i in range(outlet_count):
            f = outlets[i]
            n = f.specifications['n']
            if n is None:
                n = f.n_calc
            if n is None:
                n_out_missing_count += 1
                n_out_missing_idx = i
            else:
                n_out += n

    if n_out_missing_count == 0 and n_in_missing_count == 1:
        inlets[n_in_missing_idx].specifications['n'] = n_out - n_in
        progress = True
    if n_in_missing_count == 0 and n_out_missing_count == 1:
        outlets[n_out_missing_idx].specifications['n'] = n_in - n_out
        progress = True
    return progress


def energy_balance(inlets, outlets, reactive=False):
    inlet_count = len(inlets)
    outlet_count = len(outlets)

    in_unknown_count = out_unknown_count = 0
    in_unknown_idx = out_unknown_idx = None
    all_energy_in, all_energy_out = [], []
    all_in_known = all_out_known = True

    if inlet_count == 1 and outlet_count == 1:
        # Don't need flow rates for one in one out
        fin = inlets[0]
        fout = outlets[0]
        if not isinstance(fin, EnergyStream) and not isinstance(fout, EnergyStream):
            if reactive:
                try:
                    H_reactive_in = fin.H_reactive()
                except:
                    H_reactive_in = fin.H_reactive_calc
                try:
                    H_reactive_out = fout.H_reactive()
                except:
                    H_reactive_out = fout.H_reactive_calc

                if H_reactive_in is not None and H_reactive_out is None:
                    fout.H_reactive = H_reactive_in
                    return True
                elif H_reactive_in is None and H_reactive_out is not None:
                    fin.H_reactive = H_reactive_out
                    return True
            else:
                try:
                    Hin = fin.H()
                except:
                    Hin = fin.H_calc
                try:
                    Hout = fout.H()
                except:
                    Hout = fout.H_calc

                if Hin is not None and Hout is None:
                    fout.H = Hin
                    return True
                elif Hin is None and Hout is not None:
                    fin.H = Hout
                    return True

    for i in range(inlet_count):
        f = inlets[i]
        if reactive and not isinstance(f, EnergyStream):
            Q = f.energy_reactive
            if Q is None:
                Q = f.energy_reactive_calc
        else:
            Q = f.energy
            if Q is None:
                Q = f.energy_calc
            
        if Q is None:
            all_in_known = False
            in_unknown_count += 1
            in_unknown_idx = i
        all_energy_in.append(Q)

    for i in range(outlet_count):
        f = outlets[i]
        if reactive and not isinstance(f, EnergyStream):
            Q = f.energy_reactive
            if Q is None:
                Q = f.energy_reactive_calc
        else:
            Q = f.energy
            if Q is None:
                Q = f.energy_calc
        if Q is None:
            all_out_known = False
            out_unknown_count += 1
            out_unknown_idx = i
        all_energy_out.append(Q)

    if all_out_known and all_in_known:
        # Fast path - all known
        return False

    if all_in_known:
        inlet_energy = sum(all_energy_in)
    if all_out_known:
        outlet_energy = sum(all_energy_out)

    if out_unknown_count == 1 and in_unknown_count == 0:
        set_energy = inlet_energy
        for v in all_energy_out:
            if v is not None:
                set_energy -= v
        if reactive and not isinstance(outlets[out_unknown_idx], EnergyStream):
            outlets[out_unknown_idx].energy_reactive = set_energy
        else:
            outlets[out_unknown_idx].energy = set_energy
        return True
    if in_unknown_count == 1 and out_unknown_count == 0:
        set_energy = outlet_energy
        for v in all_energy_in:
            if v is not None:
                set_energy -= v
        if reactive and not isinstance(inlets[in_unknown_idx], EnergyStream):
            inlets[in_unknown_idx].energy_reactive = set_energy
        else:
            inlets[in_unknown_idx].energy = set_energy

        return True
    return False