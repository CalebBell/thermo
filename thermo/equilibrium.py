# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
__all__ = ['EquilibriumState']

from fluids.constants import R, R_inv
from thermo.utils import log, exp, normalize, zs_to_ws, vapor_mass_quality
from thermo.phases import gas_phases, liquid_phases, solid_phases
from thermo.elements import atom_fractions, mass_fractions, simple_formula_parser, molecular_weight, mixture_atomic_composition
from thermo.chemical_package import ChemicalConstantsPackage
from thermo.bulk import Bulk

all_phases = gas_phases + liquid_phases + solid_phases


class EquilibriumState(object):
    '''Goal is to retrieve literally every thing about the flashed phases here.
    Keep props molar, but add mass options too.
    Viscosity, thermal conductivity, atoms stuff, all there.
    Get viscosity working before figure out MW - should make more sense then!
    
    Units should not have a "property package" that has any state - they should
    have a reference to a package, and call on that package when needed to generate
    one of these objects. Anything ever needed should come from this state.
    This will be a tons of functions, but that's OK.
    
    NOTE: This means a stream is no longer the basis of simulation.
    
    '''
    max_liquid_phases = 1
    reacted = False
    
    def __init__(self, T, P, zs, 
                 gas, liquids, solids, betas,
                 flash_specs, flash_convergence,
                 constants, properties,
                 ):
        # T, P are the only properties constant across phase
        self.T = T
        self.P = P
        self.zs = zs
        
        self.N = N = len(zs)
        self.cmps = range(N)

        self.gas_count = 1 if gas is not None else 0
        self.liquid_count = len(liquids)
        self.solid_count = len(solids)
        
        self.phase_count = self.gas_count + self.liquid_count + self.solid_count
        
        self.gas = gas
        self.liquids = liquids
        self.solids = solids
        self.phases = [gas] + liquids + solids
        
        for i, l in enumerate(self.liquids):
            setattr(self, 'liquid' + str(i), l)
        for i, s in enumerate(self.solids):
            setattr(self, 'solid' + str(i), s)

        self.betas = betas
        self.beta_gas = betas[0] if self.gas_count else 0.0
        self.betas_liquids = betas_liquids = betas[self.gas_count: self.gas_count+self.liquid_count]
        self.betas_solids = betas[self.gas_count+self.liquid_count: ]
        
        if liquids:
            self.liquid_zs = normalize([sum([betas_liquids[j]*liquids[j].zs[i] for j in range(self.liquid_count)])
                               for i in self.cmps])
            self.liquid_bulk = Bulk(self.liquid_zs, self.phases, self.betas_liquids)

        if solids:
            self.solid_zs = normalize([sum([betas_solids[j]*solids[j].zs[i] for j in range(self.solid_count)])
                               for i in self.cmps])
            self.solid_bulk = Bulk(self.solid_zs, solids, self.betas_solids)
        
        self.bulk = Bulk(zs, all_phases, betas)
        
        self.flash_specs = flash_specs
        self.flash_convergence = flash_convergence
        
        self.constants = constants
        self.properties = properties
        for phase in self.phases:
            phase.result = self
            phase.constants = constants
        
    
    
    def atom_fractions(self, phase=None):
        r'''Dictionary of atomic fractions for each atom in the phase.
        '''
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
        things = dict()
        for zi, atoms in zip(zs, self.constants.atomss):
            for atom, count in atoms.items():
                if atom in things:
                    things[atom] += zi*count
                else:
                    things[atom] = zi*count

        tot_inv = 1.0/sum(things.values())
        return {atom : value*tot_inv for atom, value in things.items()}

    def atom_mass_fractions(self, phase=None):
        r'''Dictionary of mass fractions for each atom in the phase.

        '''
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
        things = dict()
        for zi, atoms in zip(zs, self.constants.atomss):
            for atom, count in atoms.items():
                if atom in things:
                    things[atom] += zi*count
                else:
                    things[atom] = zi*count
        return mass_fractions(things)

    def ws(self, phase=None):
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
        return zs_to_ws(zs, self.constants.MWs)
    
    def MW(self, phase=None):
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
        mixing_simple(zs, self.constants.MWs)
        
    def quality(self):
        return vapor_mass_quality(self.beta_gas, MWl=self.liquid_bulk.MW, MWg=self.gas.MW)

    
    def rho_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        
        Vm = phase.Vm()
        MW = phase.MW()
        if Vm is not None and MW is not None:
            return Vm_to_rho(Vm, MW)
        return None

    def H_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.H()*1e3*phase.MW_inv()
    
    def S_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.S()*1e3*phase.MW_inv()
    
    def U_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.U()*1e3*phase.MW_inv()

    def A_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.A()*1e3*phase.MW_inv()

    def G_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.G()*1e3*phase.MW_inv()
    
    def Cp_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.Cp()*1e3*phase.MW_inv()

    def Cv_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.Cv()*1e3*phase.MW_inv()
    
    def alpha(self, phase=None):
        rho = self.rho_mass(phase)
        k = self.k(phase)
        Cp = self.Cp_mass(phase)
        return thermal_diffusivity(k=k, rho=rho, Cp=Cp)

    
    def mu(self, phase=None):
        if isinstance(phase, gas_phases):
            return self.properties.ViscosityGasMixture.mixture_property(phase.T, phase.P, phase.zs, phase.ws)
        elif isinstance(phase, liquid_phases):
            return self.properties.ViscosityLiquidMixture.mixture_property(phase.T, phase.P, phase.zs, phase.ws)
        else:
            return None
    
    def k(self, phase=None):
        if phase is None:
            phase = self.bulk
        if isinstance(phase, gas_phases):
            return self.properties.ThermalConductivityGasMixture.mixture_property(
                    phase.T, phase.P, phase.zs, phase.ws)
        elif isinstance(phase, liquid_phases):
            return self.properties.ThermalConductivityLiquidMixture.mixture_property(
                    phase.T, phase.P, phase.zs, phase.ws)
        elif isinstance(phase, solid_phases):
            solid_index = phase.zs.index(1)
            return self.properties.ThermalConductivitySolids[solid_index].mixture_property(
                    phase.T, phase.P, phase.zs, phase.ws)
    
    def SG(self, phase):
        if phase is None:
            phase = self.bulk
        rho_mass = self.rho_mass(phase)
        return SG(rho_mass)

    def API(self, phase):
        if phase is None:
            phase = self.bulk
        # Going to get a list of liquid phasee models, determine which model 
        # to use. Construct a new phase object, get the density
        
    def Bvirial(self, phase):
        if phase is None:
            phase = self.bulk
        return B_from_Z(phase.Z(), self.T, self.P)


    @property
    def sigma(self, phase=None):
        if phase is None:
            phase = self.bulk

        if isinstance(phase, liquid_phases):
            return self.properties.SurfaceTensionMixture.mixture_property(
                    phase.T, phase.P, phase.zs, phase.ws)
        if isinstance(phase, gas_phases):
            return 0
    
    @property
    def Ks(self, phase):
        ref_phase = self.flash_convergence['ref_phase']
        ref_lnphis = self.phases[ref_phase].lnphis()
        lnphis = phase.lnphis()
        Ks = [exp(l - g) for l, g in zip(ref_lnphis, lnphis)]
        return Ks
        
# Add some fancy things for easier access to properties

def _make_getter_constants(name):
    def get(self):
        return getattr(self.constants, name)
    return get

constant_blacklist = set(['atom_fractions'])

for name in ChemicalConstantsPackage.properties:
    if name not in constant_blacklist:
        getter = property(_make_getter_constants(name))
        setattr(EquilibriumState, name, getter)
        for phase in all_phases:
            setattr(phase, name, getter)
        
        
def _make_getter_bulk_props(name):
    def get(self):
        return getattr(self.bulk, name)
    return get


bulk_props = ['V', 'rho', 'Cp']
for name in bulk_props:
    getter = property(_make_getter_bulk_props(name))
    setattr(EquilibriumState, name, getter)


