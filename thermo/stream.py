# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['Stream', 'EnergyTypes', 'EnergyStream', 'StreamArgs']
 
import enum
from collections import OrderedDict
from numbers import Number

from thermo.utils import property_molar_to_mass, property_mass_to_molar
from thermo.mixture import Mixture, preprocess_mixture_composition
from fluids.pump import voltages_1_phase_residential, voltages_3_phase, frequencies


# Could just assume IDs is always specified and constant.
# This might be useful for regular streams too, just to keep track of what values were specified by the user!
# If one composition value gets set to, remove those from every other value

class StreamArgs(object):
    
    @property
    def IDs(self):
        return self.specifications['IDs']
    @IDs.setter
    def IDs(self, IDs):
        self.specifications['IDs'] = IDs

    @property
    def T(self):
        return self.specifications['T']
    @T.setter
    def T(self, T):
        if T is None:
            self.specifications['T'] = T
            return None
        if self.specified_state_vars > 1:
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
        if self.specified_state_vars > 1:
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
        if self.specified_state_vars > 1:
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
        if self.specified_state_vars > 1:
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
        if self.specified_state_vars > 1:
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
        if self.specified_state_vars > 1:
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
        if self.specified_state_vars > 1:
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
            args = {'zs': arg, 'ws': None, 'Vfls': None, 'Vfgs': None, 'ns': None,
                    'ms': None, 'Qls': None, 'Qgs': None}
            self.specifications.update(args)

    @property
    def ws(self):
        return self.specifications['ws']
    @ws.setter
    def ws(self, arg):
        if arg is None:
            self.specifications['ws'] = arg
        else:
            args = {'zs': None, 'ws': arg, 'Vfls': None, 'Vfgs': None, 'ns': None,
                    'ms': None, 'Qls': None, 'Qgs': None}
            self.specifications.update(args)

    @property
    def Vfls(self):
        return self.specifications['Vfls']
    @Vfls.setter
    def Vfls(self, arg):
        if arg is None:
            self.specifications['Vfls'] = arg
        else:
            args = {'zs': None, 'ws': None, 'Vfls': arg, 'Vfgs': None, 'ns': None,
                    'ms': None, 'Qls': None, 'Qgs': None}
            self.specifications.update(args)

    @property
    def Vfgs(self):
        return self.specifications['Vfgs']
    @Vfgs.setter
    def Vfgs(self, arg):
        if arg is None:
            self.specifications['Vfgs'] = arg
        else:
            args = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': arg, 'ns': None,
                    'ms': None, 'Qls': None, 'Qgs': None}
            self.specifications.update(args)

    @property
    def ns(self):
        return self.specifications['ns']
    @ns.setter
    def ns(self, arg):
        if arg is None:
            self.specifications['ns'] = arg
        else:
            args = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None, 
                    'ns': arg, 'ms': None, 'Qls': None, 'Qgs': None,
                    'n': None, 'm': None, 'Q': None}
            self.specifications.update(args)

    @property
    def ms(self):
        return self.specifications['ms']
    @ms.setter
    def ms(self, arg):
        if arg is None:
            self.specifications['ms'] = arg
        else:
            args = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None, 
                    'ns': None, 'ms': arg, 'Qls': None, 'Qgs': None,
                    'n': None, 'm': None, 'Q': None}
            self.specifications.update(args)

    @property
    def Qls(self):
        return self.specifications['Qls']
    @Qls.setter
    def Qls(self, arg):
        if arg is None:
            self.specifications['Qls'] = arg
        else:
            args = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None, 
                    'ns': None, 'ms': None, 'Qls': arg, 'Qgs': None,
                    'n': None, 'm': None, 'Q': None}
            self.specifications.update(args)

    @property
    def Qgs(self):
        return self.specifications['Qgs']
    @Qgs.setter
    def Qgs(self, arg):
        if arg is None:
            self.specifications['Qgs'] = arg
        else:
            args = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None, 
                    'ns': None, 'ms': None, 'Qls': None, 'Qgs': arg,
                    'n': None, 'm': None, 'Q': None}
            self.specifications.update(args)

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

    def __init__(self, IDs=None, zs=None, ws=None, Vfls=None, Vfgs=None,
                 T=None, P=None, 
                 VF=None, H=None, Hm=None, S=None, Sm=None,
                 ns=None, ms=None, Qls=None, Qgs=None, m=None, n=None, Q=None):
        self.specifications = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None, 
                               'ns': None, 'ms': None, 'Qls': None, 'Qgs': None,
                               'n': None, 'm': None, 'Q': None,
                               'T': None, 'P': None, 'VF': None, 'H': None,
                               'Hm': None, 'S': None, 'Sm': None}
        
        self.IDs = IDs
        self.zs = zs
        self.ws = ws
        self.Vfls = Vfls
        self.Vfgs = Vfgs
        self.T = T
        self.P = P
        self.VF = VF
        self.H = H
        self.Hm = Hm
        self.S = S
        self.Sm = Sm
        
        self.ns = ns
        self.ms = ms
        self.Qls = Qls
        self.Qgs = Qgs
        self.m = m
        self.n = n
        self.Q = Q
        self.V_TP = (None, None)
    
    
    @property
    def composition_spec(self):
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
    def specified_composition_vars(self):
        IDs, zs, ws, Vfls, Vfgs = preprocess_mixture_composition(IDs=self.IDs,
                                zs=self.zs, ws=self.ws, Vfls=self.Vfls, 
                                Vfgs=self.Vfgs, ignore_exceptions=True)
        
        return sum(i is not None for i in (zs, ws, Vfls, Vfgs, self.ns, self.ms, self.Qls, self.Qgs))

    @property
    def composition_specified(self):
        IDs, zs, ws, Vfls, Vfgs = preprocess_mixture_composition(IDs=self.IDs,
                                zs=self.zs, ws=self.ws, Vfls=self.Vfls, 
                                Vfgs=self.Vfgs, ignore_exceptions=True)
        
        specified_vals = (i is not None for i in (zs, ws, Vfls, Vfgs, self.ns, self.ms, self.Qls, self.Qgs))
        if any(specified_vals) and IDs:
            return True
        return False
    
    @property
    def state_specs(self):
        specs = []
        for var in ('T', 'P', 'VF', 'Hm', 'H', 'Sm', 'S'):
            v = getattr(self, var)
            if v is not None:
                specs.append((var, v))
        return specs
    
    @property
    def specified_state_vars(self):
        return sum(i is not None for i in (self.T, self.P, self.VF, self.Hm, self.H, self.Sm, self.S))
    
    @property
    def state_specified(self):
        state_vars = (i is not None for i in (self.T, self.P, self.VF, self.Hm, self.H, self.Sm, self.S))
        if sum(state_vars) == 2:
            return True
        return False
    
    @property
    def flow_spec(self):
        specs = []
        for var in ('ns', 'ms', 'Qls', 'Qgs', 'm', 'n', 'Q'):
            v = getattr(self, var)
            if v is not None:
                return var, v

    @property
    def specified_flow_vars(self):
        return sum(i is not None for i in (self.ns, self.ms, self.Qls, self.Qgs, self.m, self.n, self.Q))

    @property
    def flow_specified(self):
        flow_vars = (i is not None for i in (self.ns, self.ms, self.Qls, self.Qgs, self.m, self.n, self.Q))
        if sum(flow_vars) == 1:
            return True
        return False
        
    def update(self, **kwargs):
        for key, value in kwargs:
            setattr(self, key, value)
    
    @property
    def stream(self):
        if self.IDs and self.composition_specified and self.state_specified and self.flow_specified:
            return Stream(IDs=self.IDs, zs=self.zs, ws=self.ws, Vfls=self.Vfls, Vfgs=self.Vfgs,
                 ns=self.ns, ms=self.ms, Qls=self.Qls, Qgs=self.Qgs, 
                 n=self.n, m=self.m, Q=self.Q, 
                 T=self.T, P=self.P, VF=self.VF, H=self.H, S=self.S, 
                 V_TP=self.V_TP)
        

class Stream(Mixture):
    '''Creates a Stream object which is useful for modeling mass and energy 
    balances.
    
    Streams have five variables. The flow rate, composition, and components are
    mandatory; and two of the variables temperature, pressure, vapor fraction, 
    enthalpy, or entropy are required.
    
    The composition and flow rate may be specified together or separately. The
    options for specifying them are:
    
    * Mole fractions `zs`
    * Mass fractions `ws`
    * Liquid standard volume fractions `Vfls`
    * Gas standard volume fractions `Vfgs`
    * Mole flow rates `ns`
    * Mass flow rates `ms`
    * Liquid flow rates `Qls` (based on pure component volumes at the T and P 
      specified by `V_TP`)
    * Gas flow rates `Qgs` (based on pure component volumes at the T and P 
      specified by `V_TP`)

    If only the composition is specified by providing any of `zs`, `ws`, `Vfls`
    or `Vfgs`, the flow rate must be specified by providing one of these:
        
    * Mole flow rate `n`
    * Mass flow rate `m`
    * Volumetric flow rate `Q` at the provided `T` and `P`
    
    The enthalpy or entropy, if specified, are not specific values, but 
    extensive ones for the whole stream (units J and J/K respectively).
    
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
    S : float, optional
        Mass entropy of the stream, [J/K]
    pkg : object 
        The thermodynamic property package to use for flash calculations;
        one of the caloric packages in :obj:`thermo.property_package`;
        defaults to the ideal model [-]
    Vf_TP : tuple(2, float), optional
        The (T, P) at which the volume fractions are specified to be at, [K] 
        and [Pa] 
 
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
                 T=None, P=None, VF=None, H=None, S=None, 
                 V_TP=(None, None)):
        composition_options = (zs, ws, Vfls, Vfgs, ns, ms, Qls, Qgs)
        composition_option_count = sum(i is not None for i in composition_options)
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
        flow_options = (ns, ms, Qls, Qgs, m, n, Q)
        flow_option_count = sum(i is not None for i in flow_options)
        if flow_option_count < 1:
            raise Exception("No flow rate information is provided; one of "
                            "'m', 'n', 'Q', 'ms', 'ns', 'Qls', or 'Qgs' must "
                            "be specified")
        elif flow_option_count > 1:
            raise Exception("More than one source of flow rate information is "
                            "provided; only one of "
                            "'m', 'n', 'Q', 'ms', 'ns', 'Qls', or 'Qgs' can "
                            "be specified")
        
        if ns is not None:
            zs = ns
        elif ms is not None:
            ws = ms
        elif Qls is not None:
            Vfls = Qls
        elif Qgs is not None:
            Vfgs = Qgs
        
        if T is not None and P is not None:
            super(Stream, self).__init__(IDs, zs=zs, ws=ws, Vfls=Vfls, Vfgs=Vfgs,
                 T=T, P=P, Vf_TP=V_TP)
        else:
            Mixture.autoflash = False
            super(Stream, self).__init__(IDs, zs=zs, ws=ws, Vfls=Vfls, Vfgs=Vfgs,
                 Vf_TP=V_TP)
            Mixture.autoflash = True
                        
        

        if n is not None:
            self.n = n
        elif m is not None:
            self.n = property_molar_to_mass(m, self.MW) # m*10000/MW
        elif Q is not None:
            try:
                self.n = Q/self.Vm
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
        
        
        if T is None or P is None: 
            non_TP_state_vars = sum(i is not None for i in [VF, H, S])
            if non_TP_state_vars == 0:
                if T is None:
                    T = self.T_default
                if P is None:
                    P = self.P_default
                
        self.flash(T=T, P=P, VF=VF, H=H, S=S)
        
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
        
            
    def flash(self, T=None, P=None, VF=None, H=None, S=None):
        if H is not None:
            Hm = H/self.n
        else:
            Hm = None
        if S is not None:
            Sm = S/self.n
        else:
            Sm = None
        super(Stream, self).flash_caloric(T=T, P=P, VF=VF, Hm=Hm, Sm=Sm)
        self.set_extensive_properties()


    def set_extensive_properties(self):
        # TODO: make sure this is called in post-flash routine
        if not hasattr(self, 'm'):
            return None
        if hasattr(self, 'Hm') and self.Hm is not None:
            self.H *= self.m
            self.Hm *= self.n
        if hasattr(self, 'Sm') and self.Sm is not None:
            self.S *= self.m
            self.Sm *= self.n
        if hasattr(self, 'Gm') and self.Sm is not None:
            self.G *= self.m
            self.Gm *= self.n

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
        return Stream(IDs=cmps, ns=moles, T=T, P=P)

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
        return '<Energy stream, Q=%g W, medium=%s>' %(self.Q, self.medium.value)
    
    def __init__(self, Q, medium=EnergyTypes.ELECTRICITY):
        self.medium = medium
        # isinstance test is slow, especially with Number - faster to check float and int first
        if not isinstance(Q, (float, int, Number)):
            raise Exception('Energy stream flow rate is not a flow rate')
        self.Q = Q
