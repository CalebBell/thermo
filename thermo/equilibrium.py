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
from thermo.utils import log, exp, mass_fractions
from thermo.phases import gas_phases, liquid_phases

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
    
    def __init__(gas, liquids, solids):
        self.gas = gas
        self.liquids = liquids
        self.solids = solids
        
        
        
        self.liquid_bulk = Bulk(liquid_zs_avg, liquids, phase_fractions_liquid)
        self.solid_bulk = Bulk(solid_zs_avg, solids, phase_fractions_solids)
        self.bulk = bulk(all_zs, all_phases, phase_fractions_all)
    
    @property
    def chemicals(self):
        return 'thing that lets you do EquilibriumState.chemicals.Methane.MW'
    
    @property
    def Tms(self):
        return self._Tms
    
    @property
    def Tcs(self):
        return self._Tcs
        
    def atom_fractions(self, phase):
        r'''Dictionary of atomic fractions for each atom in the phase.
        '''
        things = dict()
        for zi, atoms in zip(phase.zs, self.atomss):
            for atom, count in atoms.items():
                if atom in things:
                    things[atom] += zi*count
                else:
                    things[atom] = zi*count

        tot_inv = sum(things.values())
        return {atom : value*tot_inv for atom, value in things.items()}

    def mass_fractions(self, phase):
        r'''Dictionary of mass fractions for each atom in the phase.

        '''
        things = dict()
        for zi, atoms in zip(phase.zs, self.atomss):
            for atom, count in atoms.items():
                if atom in things:
                    things[atom] += zi*count
                else:
                    things[atom] = zi*count
        return mass_fractions(things)
    
    def mu(self, phase):
        if isinstance(phase, gas_phases):
            return self.ViscosityGasMixture.TP_dependent_property(self.T, self.P, self.gas.zs, self.gas.ws)
        elif isinstance(phase, liquid_phases):
            return self.ViscosityLiquidMixture.TP_dependent_property(self.T, self.P, self.gas.zs, self.gas.ws)
        else:
            return None
