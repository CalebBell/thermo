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
from fluids.core import thermal_diffusivity
from thermo.utils import log, exp, normalize, zs_to_ws, vapor_mass_quality, mixing_simple, Vm_to_rho, SG
from thermo.phases import gas_phases, liquid_phases, solid_phases, Phase
from thermo.elements import atom_fractions, mass_fractions, simple_formula_parser, molecular_weight, mixture_atomic_composition
from thermo.chemical_package import ChemicalConstantsPackage, PropertyCorrelationPackage
from thermo.bulk import Bulk, BulkSettings, default_settings

all_phases = gas_phases + liquid_phases + solid_phases

CAS_H2O = '7732-18-5'


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

    T_REF_IG = Phase.T_REF_IG
    T_REF_IG_INV = Phase.T_REF_IG_INV
    P_REF_IG = Phase.P_REF_IG
    P_REF_IG_INV = Phase.P_REF_IG_INV
    
    def __repr__(self):
        s = '<EquilibriumState, T=%.4f, P=%.4f, zs=%s, betas=%s, phases=%s>'
        s = s %(self.T, self.P, self.zs, self.betas, self.phases)
        return s
    
    def __init__(self, T, P, zs, 
                 gas, liquids, solids, betas,
                 flash_specs=None, flash_convergence=None,
                 constants=None, correlations=None, flasher=None,
                 settings=default_settings):
        # T, P are the only properties constant across phase
        self.T = T
        self.P = P
        self.zs = zs
        
        self.N = N = len(zs)
        self.cmps = range(N)

        self.gas_count = gas_count = 1 if gas is not None else 0
        self.liquid_count = liquid_count = len(liquids)
        self.solid_count = solid_count = len(solids)
        
        self.phase_count = gas_count + liquid_count + solid_count
        
        self.gas = gas
        self.liquids = liquids
        self.solids = solids
        if gas is not None:
            self.phases = [gas] + liquids + solids
        else:
            self.phases = liquids + solids
        
        self.betas = betas
        self.beta_gas = betas[0] if gas_count else 0.0
        self.betas_liquids = betas_liquids = betas[gas_count:gas_count + liquid_count]
        self.betas_solids = betas_solids = betas[gas_count + liquid_count:]
        
        if liquid_count > 1:
#                tot_inv = 1.0/sum(values)
#                return [i*tot_inv for i in values]
            self.liquid_zs = normalize([sum([betas_liquids[j]*liquids[j].zs[i] for j in range(liquid_count)])
                               for i in self.cmps])
            self.liquid_bulk = liquid_bulk = Bulk(T, P, self.liquid_zs, self.liquids, self.betas_liquids)
            liquid_bulk.result = self
            liquid_bulk.constants = constants
            liquid_bulk.correlations = correlations
            for i, l in enumerate(liquids):
                setattr(self, 'liquid%d'%(i), l)
        elif liquid_count:
            self.liquid_zs = liquids[0].zs
            self.liquid_bulk = liquids[0]
            self.liquid0 = liquids[0]
        if solids:
            self.solid_zs = normalize([sum([betas_solids[j]*solids[j].zs[i] for j in range(self.solid_count)])
                               for i in self.cmps])
            self.solid_bulk = solid_bulk = Bulk(T, P, self.solid_zs, solids, self.betas_solids)
            solid_bulk.result = self
            solid_bulk.constants = constants
            solid_bulk.correlations = correlations
            for i, s in enumerate(solids):
                setattr(self, 'solid%d' %(i), s)

        self.bulk = bulk = Bulk(T, P, zs, self.phases, betas)
        bulk.result = self
        bulk.constants = constants
        bulk.correlations = correlations
        bulk.flasher = flasher
        bulk.settings = settings

        self.flash_specs = flash_specs
        self.flash_convergence = flash_convergence
        self.flasher = flasher
        self.settings = settings
        self.constants = constants
        self.correlations = correlations
        for phase in self.phases:
            phase.result = self
            phase.constants = constants
            phase.correlations = correlations
    @property
    def phases_str(self):
        s = ''
        if self.gas:
            s += 'V'
        s += 'L'*len(self.liquids)
        s += 'S'*len(self.solids)
        return s

    
    @property
    def betas_mass(self):
        phase_iter = range(self.phase_count)
        betas = self.betas
        MWs_phases = [i.MW() for i in self.phases]
        
        tot = 0.0
        for i in phase_iter:
            tot += MWs_phases[i]*betas[i]
        tot_inv = 1.0/tot
        return [betas[i]*MWs_phases[i]*tot_inv for i in phase_iter]
    
    @property
    def betas_volume(self):
        phase_iter = range(self.phase_count)
        betas = self.betas
        Vs_phases = [i.V() for i in self.phases]
        
        tot = 0.0
        for i in phase_iter:
            tot += Vs_phases[i]*betas[i]
        tot_inv = 1.0/tot
        return [betas[i]*Vs_phases[i]*tot_inv for i in phase_iter]
    
    def V_liquids_ref(self):
        T_liquid_volume_ref = self.settings.T_liquid_volume_ref
        if T_liquid_volume_ref == 298.15:
            Vls = self.Vml_STPs
        elif T_liquid_volume_ref == 288.7055555555555:
            Vls = self.Vml_60Fs
        else:
            Vls = [i(T_liquid_volume_ref) for i in self.VolumeLiquids]
        return Vls
    
    def V_liquid_ref(self, phase=None):
        if phase is None:
            phase = self.bulk
            zs = self.zs
        else:
            zs = phase.zs
            
        Vls = self.V_liquids_ref()
        V = 0.0
        for i in self.cmps:
            V += zs[i]*Vls[i]
        return V
    
    def rho_liquid_ref(self, phase=None):
        if phase is None:
            phase = self.bulk

        V = self.V_liquid_ref(phase)
        MW = phase.MW()
        return Vm_to_rho(V, MW)
    
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
            phase = self.bulk
        zs = phase.zs
        things = dict()
        for zi, atoms in zip(zs, self.constants.atomss):
            for atom, count in atoms.items():
                if atom in things:
                    things[atom] += zi*count
                else:
                    things[atom] = zi*count
        return mass_fractions(things, phase.MW())

    def ws(self, phase=None):
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
        return zs_to_ws(zs, self.constants.MWs)
    
    def Vfls(self, phase=None):
        if phase is None:
            phase = self.bulk
            zs = self.zs
        else:
            zs = phase.zs
            
        Vls = self.V_liquids_ref()
        V = 0.0
        for i in self.cmps:
            V += zs[i]*Vls[i]
        V_inv = 1.0/V
        return [V_inv*Vls[i]*zs[i] for i in self.cmps]

    def Vfgs(self, phase=None):
        # Ideal volume fractions - do not attempt to compensate for non-ideality
        # Make an _Actual property to try that.
        if phase is None:
            phase = self.bulk
            zs = self.zs
        else:
            zs = phase.zs
        return zs
    
    def MW(self, phase=None):
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
            
        MWs = self.constants.MWs
        MW = 0.0
        for i in self.cmps:
            MW += zs[i]*MWs[i]
        return MW

    def pseudo_Tc(self, phase=None):
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
            
        Tcs = self.constants.Tcs
        Tc = 0.0
        for i in self.cmps:
            Tc += zs[i]*Tcs[i]
        return Tc

    def pseudo_Pc(self, phase=None):
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
            
        Pcs = self.constants.Pcs
        Pc = 0.0
        for i in self.cmps:
            Pc += zs[i]*Pcs[i]
        return Pc

    def pseudo_Vc(self, phase=None):
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
            
        Vcs = self.constants.Vcs
        Vc = 0.0
        for i in self.cmps:
            Vc += zs[i]*Vcs[i]
        return Vc
        
    def pseudo_Zc(self, phase=None):
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
            
        Zcs = self.constants.Zcs
        Zc = 0.0
        for i in self.cmps:
            Zc += zs[i]*Zcs[i]
        return Zc

    def quality(self):
        return vapor_mass_quality(self.beta_gas, MWl=self.liquid_bulk.MW(), MWg=self.gas.MW())
    
    def Tmc(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.Tmc()

    def Pmc(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.Pmc()

    def Vmc(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.Vmc()

    def Zmc(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.Zmc()
    
    def rho_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        
        V = phase.V()
        MW = phase.MW()
        return Vm_to_rho(V, MW)

    def V_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        
        V = phase.V()
        MW = phase.MW()
#        return 1.0/((Vm)**-1*MW/1000.)
        return 1.0/Vm_to_rho(V, MW)

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


    def Cp_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.Cp()*1e3*phase.MW_inv()

    def Cp_ideal_gas(self, phase=None):
        if phase is None:
            phase = self.bulk
        try:
            return phase.Cp_ideal_gas()
        except:
            pass

        HeatCapacityGases = self.correlations.HeatCapacityGases
        T = self.T
        Cpigs_pure = [i.T_dependent_property(T) for i in HeatCapacityGases]

        Cp, zs = 0.0, phase.zs
        for i in self.cmps:
            Cp += zs[i]*Cpigs_pure[i]
        return Cp
    
    def H_dep(self, phase=None):
        if phase is None:
            phase = self.bulk
        elif phase is not self.bulk:
            try:
                return phase.H_dep()
            except:
                pass
        return phase.H() - self.H_ideal_gas(phase)

    def S_dep(self, phase=None):
        if phase is None:
            phase = self.bulk
        elif phase is not self.bulk:
            try:
                return phase.S_dep()
            except:
                pass
        return phase.S() - self.S_ideal_gas(phase)

    def Cp_dep(self, phase=None):
        if phase is None:
            phase = self.bulk
        elif phase is not self.bulk:
            try:
                return phase.Cp_dep()
            except:
                pass
        return phase.Cp() - self.Cp_ideal_gas(phase)

    def Cv_dep(self, phase=None):
        if phase is None:
            phase = self.bulk
        elif phase is not self.bulk:
            try:
                return phase.Cv_dep()
            except:
                pass
        return phase.Cv() - self.Cv_ideal_gas(phase)

    def H_ideal_gas(self, phase=None):
        if phase is None:
            phase = self.bulk
        try:
            return phase.H_ideal_gas()
        except:
            pass

        HeatCapacityGases = self.correlations.HeatCapacityGases
        T, T_REF_IG = self.T, self.T_REF_IG
        Cpig_integrals_pure = [obj.T_dependent_property_integral(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        H = 0.0
        for zi, Cp_int in zip(phase.zs, Cpig_integrals_pure):
            H += zi*Cp_int
        return H

    def S_ideal_gas(self, phase=None):
        if phase is None:
            phase = self.bulk
        try:
            return phase.S_ideal_gas()
        except:
            pass

        HeatCapacityGases = self.correlations.HeatCapacityGases
        T, T_REF_IG = self.T, self.T_REF_IG

        Cpig_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                      for obj in HeatCapacityGases]

        log_zs = self.log_zs()
        T, P, zs, cmps = self.T, self.P, phase.zs, self.cmps
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)
        
        for i in cmps:
            S += zs[i]*Cpig_integrals_over_T_pure[i]

        return S
    
    def Cv_ideal_gas(self, phase=None):
        if phase is None:
            phase = self.bulk
        return self.Cp_ideal_gas(phase) - R

    def Cp_Cv_ratio_ideal_gas(self, phase=None):
        return self.Cp_ideal_gas(phase)/self.Cv_ideal_gas(phase)

    def G_ideal_gas(self, phase=None):
        G_ideal_gas = self.H_ideal_gas(phase) - self.T*self.S_ideal_gas(phase)
        return G_ideal_gas

    def U_ideal_gas(self, phase=None):
        U_ideal_gas = self.H_ideal_gas(phase) - self.P*self.V_ideal_gas(phase)
        return U_ideal_gas

    def A_ideal_gas(self, phase=None):
        A_ideal_gas = self.U_ideal_gas(phase) - self.T*self.S_ideal_gas(phase)
        return A_ideal_gas

    def V_ideal_gas(self, phase=None):
        return R*self.T/self.P


    def H_formation_ideal_gas(self, phase=None):
        if phase is None:
            phase = self.bulk
        elif phase is not self.bulk:
            try:
                return phase.H_formation_ideal_gas()
            except:
                pass
        
        Hf = 0.0
        zs = phase.zs
        Hfgs = self.constants.Hfgs
        for i in self.cmps:
            Hf += zs[i]*Hfgs[i]
        return Hf

    def S_formation_ideal_gas(self, phase=None):
        if phase is None:
            phase = self.bulk
        elif phase is not self.bulk:
            try:
                return phase.S_formation_ideal_gas()
            except:
                pass
        
        Sf = 0.0
        zs = phase.zs
        Sfgs = self.constants.Sfgs
        for i in self.cmps:
            Sf += zs[i]*Sfgs[i]
        return Sf

    def G_formation_ideal_gas(self, phase=None):
        Gf = self.H_formation_ideal_gas(phase) - self.T_REF_IG*self.S_formation_ideal_gas(phase)
        return Gf
    
    def U_formation_ideal_gas(self, phase=None):
        Uf = self.H_formation_ideal_gas(phase) - self.P_REF_IG*self.V_ideal_gas(phase)
        return Uf
    
    def A_formation_ideal_gas(self, phase=None):
        Af = self.U_formation_ideal_gas(phase) - self.T_REF_IG*self.S_formation_ideal_gas(phase)
        return Af
    
    def alpha(self, phase=None):
        rho = self.rho_mass(phase)
        k = self.k(phase)
        Cp = self.Cp_mass(phase)
        return thermal_diffusivity(k=k, rho=rho, Cp=Cp)

    
    def mu(self, phase=None):
        if phase is None:
            if self.phase_count == 1:
                phase = self.phases[0]
            else:
                phase = None
                for beta, p in zip(self.betas, self.phases):
                    if beta == 1.0:
                        phase = p
                        break
                if phase is None:
                    phase = self.bulk
        if phase is not self.bulk:
            return phase.mu()
        if isinstance(phase, gas_phases):
            return self.correlations.ViscosityGasMixture.mixture_property(phase.T, phase.P, phase.zs, phase.ws())
        elif isinstance(phase, liquid_phases):
            return self.correlations.ViscosityLiquidMixture.mixture_property(phase.T, phase.P, phase.zs, phase.ws())
        else:
            raise NotImplementedError("no bulk methods")
            
    def nu(self, phase=None):
        return self.mu(phase)/self.rho_mass(phase)
    
    def k(self, phase=None):
        if phase is None:
            if self.phase_count == 1:
                phase = self.phases[0]
            else:
                phase = self.bulk
#        try:
#            k = phase.k()
#            if k is not None:
#                return k
#        except:
#            pass
        if isinstance(phase, gas_phases):
            return self.correlations.ThermalConductivityGasMixture.mixture_property(
                    phase.T, phase.P, phase.zs, phase.ws())
            
        elif isinstance(phase, liquid_phases):
            return self.correlations.ThermalConductivityLiquidMixture.mixture_property(
                    phase.T, phase.P, phase.zs, phase.ws())
            
        elif isinstance(phase, solid_phases):
            solid_index = phase.zs.index(1)
            return self.correlations.ThermalConductivitySolids[solid_index].mixture_property(
                    phase.T, phase.P, phase.zs, phase.ws())
        else:
            raise NotImplementedError("no bulk methods")
    
    def SG(self, phase=None):
        if phase is None:
            phase = self.bulk
        ws = phase.ws()
        rhol_60Fs_mass = self.constants.rhol_60Fs_mass
        # Better results than using the phase's density model anyway
        rho_mass_60F = 0.0
        for i in self.cmps:
            rho_mass_60F += ws[i]*rhol_60Fs_mass[i]
        return SG(rho_mass_60F, rho_ref=999.0170824078306)
    
    def SG_gas(self, phase=None):
        if phase is None:
            phase = self.bulk
        MW = phase.MW()
        # Standard MW of air as in dry air standard
        # It would be excessive to do a true density call
        '''Lemmon, Eric W., Richard T. Jacobsen, Steven G. Penoncello, and
        Daniel G. Friend. "Thermodynamic Properties of Air and Mixtures of 
        Nitrogen, Argon, and Oxygen From 60 to 2000 K at Pressures to 2000 
        MPa." Journal of Physical and Chemical Reference Data 29, no. 3 (May
        1, 2000): 331-85. https://doi.org/10.1063/1.1285884.
        '''
        return MW/28.9586
#        rho_mass = self.rho_mass(phase)
#        return SG(rho_mass, rho_ref=1.2)


    def V_gas_standard(self, phase=None):
        return R*self.settings.T_standard/self.settings.P_standard
    
    def V_gas_normal(self, phase=None):
        return R*self.settings.T_normal/self.settings.P_normal

    def API(self, phase=None):
        if phase is None:
            phase = self.bulk
        # Going to get a list of liquid phasee models, determine which model 
        # to use. Construct a new phase object, get the density
        
    def Bvirial(self, phase=None):
        if phase is None:
            phase = self.bulk
        return B_from_Z(phase.Z(), self.T, self.P)
    
    def H_C_ratio(self, phase=None):
        if phase is None:
            phase = self.bulk
        atom_fractions = self.atom_fractions()
        H = atom_fractions.get('H', 0.0)
        C = atom_fractions.get('C', 0.0)
        try:
            return H/C
        except ZeroDivisionError:
            return None
        
    def H_C_ratio_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        atom_fractions = self.atom_mass_fractions()
        H = atom_fractions.get('H', 0.0)
        C = atom_fractions.get('C', 0.0)
        try:
            return H/C
        except ZeroDivisionError:
            return None
    
    @property
    def water_phase_index(self):
        try:
            return self._water_phase_index
        except AttributeError:
            pass
        try:
            water_index = self._water_index
        except AttributeError:
            water_index = self.water_index
            
        max_zw, max_phase, max_phase_idx = 0.0, None, None
        for i, l in enumerate(self.liquids):
            z_w = l.zs[water_index]
            if z_w > max_zw:
                max_phase, max_zw, max_phase_idx = l, z_w, i
        
        self._water_phase_index = max_phase_idx
        return max_phase_idx

    @property
    def lightest_liquid(self):
        liquids = self.liquids
        if not liquids:
            return None
        elif len(liquids) == 1:
            return liquids[0]
        else:
            rhos = [i.rho_mass() for i in liquids]
            min_rho = min(rhos)
            return liquids[rhos.index(min_rho)]

    @property
    def heaviest_liquid(self):
        liquids = self.liquids
        if not liquids:
            return None
        elif len(liquids) == 1:
            return liquids[0]
        else:
            rhos = [i.rho_mass() for i in liquids]
            max_rho = max(rhos)
            return liquids[rhos.index(max_rho)]


    @property
    def water_phase(self):
        try:
            return self.liquids[self.water_phase_index]
        except:
            return None
    
    @property
    def water_index(self):
        try:
            return self._water_index
        except AttributeError:
            pass
        
        try:
            self._water_index = self.constants.CASs.index(CAS_H2O)
        except ValueError:
            self._water_index = None
        return self._water_index
        
        
    def molar_water_content(self, phase=None):
        if phase is None:
            phase = self.bulk
            
        water_index = self.water_index
        if water_index is None:
            return 0.0

        MW_water = self.constants.MWs[water_index]
        return MW_water*phase.ws()[water_index]

    def zs_no_water(self, phase=None):
        if phase is None:
            phase = self.bulk
            
        water_index = self.water_index
        if water_index is None:
            return phase.zs
        zs = list(phase.zs)
        z_water = zs[water_index]
        m = 1/(1.0 - z_water)
        for i in self.cmps:
            zs[i] *= m
        return m
        
    def ws_no_water(self, phase=None):
        if phase is None:
            phase = self.bulk
            
        water_index = self.water_index
        if water_index is None:
            return phase.ws()
        ws = list(phase.ws())
        z_water = ws[water_index]
        m = 1/(1.0 - z_water)
        for i in self.cmps:
            ws[i] *= m
        return m

    @property
    def sigma(self, phase=None):
        if phase is None:
            phase = self.bulk

        if isinstance(phase, liquid_phases):
            return self.correlations.SurfaceTensionMixture.mixture_property(
                    phase.T, phase.P, phase.zs, phase.ws())
        if isinstance(phase, gas_phases):
            return 0
    
    def Ks(self, phase):
        ref_phase = self.flash_convergence['ref_phase']
        ref_lnphis = self.phases[ref_phase].lnphis()
        lnphis = phase.lnphis()
        Ks = [exp(l - g) for l, g in zip(ref_lnphis, lnphis)]
        return Ks

    def Hc(self, phase=None):
        if phase is None:
            phase = self.bulk
        return mixing_simple(self.constants.Hcs, phase.ws())

    def Hc_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        return mixing_simple(self.constants.Hcs_mass, phase.zs)
    
    def Hc_normal(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.Hc()/phase.V_gas_normal()

    def Hc_standard(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.Hc()/phase.V_gas_standard()


    def Hc_lower(self, phase=None):
        if phase is None:
            phase = self.bulk
        return mixing_simple(self.constants.Hcs_lower, phase.zs)
    
    def Hc_lower_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
        return mixing_simple(self.constants.Hcs_lower_mass, phase.ws())
    
    def Hc_lower_normal(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.Hc_lower()/phase.V_gas_normal()

    def Hc_lower_standard(self, phase=None):
        if phase is None:
            phase = self.bulk
        return phase.Hc_lower()/phase.V_gas_standard()

    
    def Wobbe_index(self, phase=None):
        if phase is None:
            phase = self.bulk
            
        Hc = abs(phase.Hc())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
            
        Hc = abs(phase.Hc_mass())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_lower(self, phase=None):
        if phase is None:
            phase = self.bulk
            
        Hc = abs(phase.Hc_lower())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_lower_mass(self, phase=None):
        if phase is None:
            phase = self.bulk
            
        Hc = abs(phase.Hc_lower_mass())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5
    
    
    def Wobbe_index_standard(self, phase=None):
        if phase is None:
            phase = self.bulk
            
        Hc = abs(phase.Hc_standard())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_normal(self, phase=None):
        if phase is None:
            phase = self.bulk
            
        Hc = abs(phase.Hc_normal())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_lower_standard(self, phase=None):
        if phase is None:
            phase = self.bulk
            
        Hc = abs(phase.Hc_lower_standard())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_lower_normal(self, phase=None):
        if phase is None:
            phase = self.bulk
            
        Hc = abs(phase.Hc_lower_normal())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5
    
    def value(self, name):
        v = getattr(self, name)
        try:
            v = v()
        except:
            pass
        return v
    
    @property
    def IDs(self):
        return self.constants.CASs

        
# Add some fancy things for easier access to properties

def _make_getter_constants(name):
    def get(self):
        return getattr(self.constants, name)
    return get

def _make_getter_correlations(name):
    def get(self):
        return getattr(self.correlations, name)
    return get

def _make_getter_EquilibriumState(name):
    def get(self):
        return getattr(self.result, name)(self)
    return get

def _make_getter_bulk_props(name):
    def get(self):
        return getattr(self.bulk, name)
    return get

### For the pure component fixed properties, allow them to be retrived from the phase
# and bulk object as well as the Equilibrium State Object
constant_blacklist = set(['atom_fractions'])

for name in ChemicalConstantsPackage.properties:
    if name not in constant_blacklist:
        getter = property(_make_getter_constants(name))
        setattr(EquilibriumState, name, getter)
        setattr(Phase, name, getter)

### For the temperature-dependent correlations, allow them to be retrieved by their
# name from the EquilibriumState ONLY
for name in PropertyCorrelationPackage.correlations:
    getter = property(_make_getter_correlations(name))
    setattr(EquilibriumState, name, getter)


### For certain properties not supported by Phases/Bulk, allow them to call up to the 
# EquilibriumState to get the property
phases_properties_to_EquilibriumState = ['atom_fractions', 'atom_mass_fractions',
                                         'Hc', 'Hc_mass', 'Hc_lower', 'Hc_lower_mass', 'SG', 'SG_gas',
                                         'pseudo_Tc', 'pseudo_Pc', 'pseudo_Vc', 'pseudo_Zc',
                                         'V_gas_standard', 'V_gas_normal', 'Hc_normal', 'Hc_standard',
                                         'Hc_lower_normal', 'Hc_lower_standard',
                                         'Wobbe_index_lower_normal', 'Wobbe_index_lower_standard',
                                         'Wobbe_index_normal', 'Wobbe_index_standard',
                                         'Wobbe_index_lower_mass', 'Wobbe_index_lower', 
                                         'Wobbe_index_mass', 'Wobbe_index', 'V_mass',
                                         'rho_liquid_ref', 'V_liquid_ref',
                                         'molar_water_content',
                                         'ws_no_water', 'zs_no_water',
                                         'H_C_ratio', 'H_C_ratio_mass',
                                         'Vfls', 'Vfgs',
                                         ]
for name in phases_properties_to_EquilibriumState:
    method = _make_getter_EquilibriumState(name)
    setattr(Phase, name, method)

### For certain properties not supported by Bulk, allow them to call up to the 
# EquilibriumState to get the property
Bulk_properties_to_EquilibriumState = [#'H_ideal_gas', 'Cp_ideal_gas','S_ideal_gas', 
       'V_ideal_gas', 'G_ideal_gas', 'U_ideal_gas',
        'Cv_ideal_gas', 'Cp_Cv_ratio_ideal_gas',
       'A_ideal_gas', 'H_formation_ideal_gas', 'S_formation_ideal_gas', 
       'G_formation_ideal_gas', 'U_formation_ideal_gas', 'A_formation_ideal_gas',
       'H_dep', 'S_dep', 'Cp_dep', 'Cv_dep']
for name in Bulk_properties_to_EquilibriumState:
    method = _make_getter_EquilibriumState(name)
    setattr(Bulk, name, method)

### For certain properties of the Bulk phase, make EquilibriumState get it from the Bulk
bulk_props = ['V', 'Z', 'rho', 'Cp', 'Cv', 'H', 'S', 'U', 'G', 'A', 'dH_dT', 'dH_dP', 'dS_dT', 'dS_dP',
              'dU_dT', 'dU_dP', 'dG_dT', 'dG_dP', 'dA_dT', 'dA_dP', 
              'H_reactive', 'S_reactive', 'G_reactive', 'U_reactive', 'A_reactive',
              'Cp_Cv_ratio', 'log_zs', 'isothermal_bulk_modulus',
              'dP_dT_frozen', 'dP_dV_frozen', 'd2P_dT2_frozen', 'd2P_dV2_frozen',
              'd2P_dTdV_frozen',
              'd2P_dTdV', 'd2P_dV2', 'd2P_dT2', 'dP_dV', 'dP_dT', 'isentropic_exponent',
              
              'PIP', 'kappa', 'beta', 'Joule_Thomson', 'speed_of_sound',
              'speed_of_sound_mass',
              'U_dep', 'G_dep', 'A_dep', 'V_dep', 'V_iter',
              ]

for name in bulk_props:
    # Maybe take this out and implement it manually for performance?
    getter = property(_make_getter_bulk_props(name))
    setattr(EquilibriumState, name, getter)


