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
SOFTWARE.'''

from __future__ import division
__all__ = ['EquilibriumState']

from fluids.constants import R, R_inv
from fluids.core import thermal_diffusivity
from chemicals.utils import log, exp, normalize, zs_to_ws, vapor_mass_quality, mixing_simple, Vm_to_rho, SG
from chemicals.virial import B_from_Z
from chemicals.elements import atom_fractions, mass_fractions, simple_formula_parser, molecular_weight, mixture_atomic_composition
from thermo.phases import gas_phases, liquid_phases, solid_phases, Phase, derivatives_thermodynamic, derivatives_thermodynamic_mass, derivatives_jacobian
from thermo.chemical_package import ChemicalConstantsPackage, PropertyCorrelationPackage, constants_docstrings
from thermo.bulk import Bulk, BulkSettings, default_settings

all_phases = gas_phases + liquid_phases + solid_phases

CAS_H2O = '7732-18-5'

PHASE_GAS = 'gas'
PHASE_LIQUID0 = 'liquid0'
PHASE_LIQUID1 = 'liquid1'
PHASE_LIQUID2 = 'liquid2'
PHASE_LIQUID3 = 'liquid3'
PHASE_BULK_LIQUID = 'liquid_bulk'
PHASE_WATER_LIQUID = 'water_phase'
PHASE_LIGHTEST_LIQUID = 'lightest_liquid'
PHASE_HEAVIEST_LIQUID = 'heaviest_liquid'
PHASE_SOLID0 = 'solid0'
PHASE_SOLID1 = 'solid1'
PHASE_SOLID2 = 'solid2'
PHASE_SOLID3 = 'solid3'
PHASE_BULK_SOLID = 'solid_bulk'
PHASE_BULK = 'bulk'

PHASE_REFERENCES = [PHASE_GAS, PHASE_LIQUID0, PHASE_LIQUID1, PHASE_LIQUID2,
                    PHASE_LIQUID3, PHASE_BULK_LIQUID, PHASE_WATER_LIQUID,
                    PHASE_LIGHTEST_LIQUID, PHASE_HEAVIEST_LIQUID, PHASE_SOLID0,
                    PHASE_SOLID1, PHASE_SOLID2, PHASE_SOLID3, PHASE_BULK_SOLID,
                    PHASE_BULK]

__all__.extend(['PHASE_GAS', 'PHASE_LIQUID0', 'PHASE_LIQUID1', 'PHASE_LIQUID2',
                'PHASE_LIQUID3', 'PHASE_BULK_LIQUID', 'PHASE_WATER_LIQUID',
                'PHASE_LIGHTEST_LIQUID', 'PHASE_HEAVIEST_LIQUID', 'PHASE_SOLID0',
                'PHASE_SOLID1', 'PHASE_SOLID2', 'PHASE_SOLID3', 'PHASE_BULK_SOLID',
                'PHASE_BULK', 'PHASE_REFERENCES'])

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
    flashed = True

    liquid_bulk = None
    solid_bulk = None

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
        self.gas_beta = betas[0] if gas_count else 0.0
        self.liquids_betas = betas_liquids = betas[gas_count:gas_count + liquid_count]
        self.solids_betas = betas_solids = betas[gas_count + liquid_count:]

        if liquid_count > 1:
#                tot_inv = 1.0/sum(values)
#                return [i*tot_inv for i in values]
            self.liquid_zs = normalize([sum([betas_liquids[j]*liquids[j].zs[i] for j in range(liquid_count)])
                               for i in self.cmps])
            self.liquid_bulk = liquid_bulk = Bulk(T, P, self.liquid_zs, self.liquids, self.liquids_betas, 'l')
            liquid_bulk.result = self
            liquid_bulk.constants = constants
            liquid_bulk.correlations = correlations
            liquid_bulk.settings = settings
            for i, l in enumerate(liquids):
                setattr(self, 'liquid%d'%(i), l)
        elif liquid_count:
            self.liquid_zs = liquids[0].zs
            self.liquid_bulk = liquids[0]
            self.liquid0 = liquids[0]
        if solids:
            self.solid_zs = normalize([sum([betas_solids[j]*solids[j].zs[i] for j in range(self.solid_count)])
                               for i in self.cmps])
            self.solid_bulk = solid_bulk = Bulk(T, P, self.solid_zs, solids, self.solids_betas, 's')
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
    def phase(self):
        r'''Method to calculate and return a string representing the phase of
        the mixture. The return string uses 'V' to represent the gas phase,
        'L' to represent a liquid phase, and 'S' to represent a solid phase
        (always in that order).

        A state with three liquids, two solids, and a gas would return
        'VLLLSS'.

        Returns
        -------
        phase : str
            Phase string, [-]

        Notes
        -----
        '''
        s = ''
        if self.gas:
            s += 'V'
        s += 'L'*len(self.liquids)
        s += 'S'*len(self.solids)
        return s

    @property
    def VF(self):
        r'''Method to return the vapor fraction of the equilibrium state.
        If no vapor/gas is present, 0 is always returned.

        Returns
        -------
        VF : float
            Vapor fraction, [-]

        Notes
        -----
        '''
        if self.gas is not None:
            return self.betas[0]
        return 0.0 # No gas phase

    @property
    def quality(self):
        r'''Method to return the mass vapor fraction of the equilibrium state.
        If no vapor/gas is present, 0 is always returned. This is normally
        called the quality.

        Returns
        -------
        quality : float
            Vapor mass fraction, [-]

        Notes
        -----
        '''
        try:
            return vapor_mass_quality(self.gas_beta, MWl=self.liquid_bulk.MW(), MWg=self.gas.MW())
        except:
            return 0.0

    @property
    def betas_states(self):
        r'''Method to return the molar phase fractions of each of the three
        fundamental `types` of phases.

        Returns
        -------
        betas_states : list[float, 3]
            List containing the molar phase fraction of gas, liquid, and solid,
            [-]

        Notes
        -----
        '''
        return [self.gas_beta, sum(self.liquids_betas), sum(self.solids_betas)]

    @property
    def betas_mass_states(self):
        r'''Method to return the mass phase fractions of each of the three
        fundamental `types` of phases.

        Returns
        -------
        betas_mass_states : list[float, 3]
            List containing the mass phase fraction of gas, liquid, and solid,
            [-]

        Notes
        -----
        '''
        g_tot = l_tot = s_tot = 0.0
        # Compute the mass fraction of the gas phase
        gas, liquids, solids = self.gas, self.liquids, self.solids
        beta_gas, betas_liquids, betas_solids = self.gas_beta, self.liquids_betas, self.solids_betas
        gas_MW = gas.MW()
        liq_MWs = [i.MW() for i in liquids]
        solid_MWs = [i.MW() for i in solids]

        g_tot = gas_MW*beta_gas
        for i in range(self.liquid_count):
            l_tot += liq_MWs[i]*betas_liquids[i]
        for i in range(self.solid_count):
            s_tot += solid_MWs[i]*betas_solids[i]
        tot = g_tot + l_tot + s_tot
        tot = 1.0/tot

        return [g_tot*tot, l_tot*tot, s_tot*tot]

    @property
    def betas_volume_states(self):
        r'''Method to return the volume phase fractions of each of the three
        fundamental `types` of phases.

        Returns
        -------
        betas_volume_states : list[float, 3]
            List containing the volume phase fraction of gas, liquid, and solid,
            [-]

        Notes
        -----
        '''
        g_tot = l_tot = s_tot = 0.0
        # Compute the mass fraction of the gas phase
        gas, liquids, solids = self.gas, self.liquids, self.solids
        beta_gas, betas_liquids, betas_solids = self.gas_beta, self.liquids_betas, self.solids_betas
        gas_V = gas.V()
        liq_Vs = [i.V() for i in liquids]
        solid_Vs = [i.V() for i in solids]

        g_tot = gas_V*beta_gas
        for i in range(self.liquid_count):
            l_tot += liq_Vs[i]*betas_liquids[i]
        for i in range(self.solid_count):
            s_tot += solid_Vs[i]*betas_solids[i]
        tot = g_tot + l_tot + s_tot
        tot = 1.0/tot

        return [g_tot*tot, l_tot*tot, s_tot*tot]


    @property
    def betas_mass(self):
        r'''Method to calculate and return the mass fraction of all of the
        phases in the system.

        Returns
        -------
        betas_mass : list[float]
            Mass phase fractions of all the phases, ordered vapor, liquid, then
            solid , [-]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the volume fraction of all of the
        phases in the system.

        Returns
        -------
        betas_volume : list[float]
            Volume phase fractions of all the phases, ordered vapor, liquid, then
            solid , [-]

        Notes
        -----
        '''
        phase_iter = range(self.phase_count)
        betas = self.betas
        Vs_phases = [i.V() for i in self.phases]

        tot = 0.0
        for i in phase_iter:
            tot += Vs_phases[i]*betas[i]
        tot_inv = 1.0/tot
        return [betas[i]*Vs_phases[i]*tot_inv for i in phase_iter]

    @property
    def betas_liquids(self):
        r'''Method to calculate and return the fraction of the liquid phase
        that each liquid phase is, by molar phase fraction.
        If the system is VLLL with phase fractions of 0.125 vapor, and
        [.25, .125, .5] for the three liquids phases respectively, the return
        value would be [0.28571428, 0.142857142, 0.57142857].

        Returns
        -------
        betas_liquids : list[float]
            Molar phase fractions of the overall liquid phase, [-]

        Notes
        -----
        '''
        liquids_betas = self.liquids_betas
        tot = 0.0
        for vi in liquids_betas:
            tot += vi
        tot = 1.0/tot
        return [vi*tot for vi in liquids_betas]

    @property
    def betas_mass_liquids(self):
        r'''Method to calculate and return the fraction of the liquid phase
        that each liquid phase is, by mass phase fraction.
        If the system is VLLL with mass phase fractions of 0.125 vapor, and
        [.25, .125, .5] for the three liquids phases respectively, the return
        value would be [0.28571428, 0.142857142, 0.57142857].

        Returns
        -------
        betas_mass_liquids : list[float]
            Mass phase fractions of the overall liquid phase, [-]

        Notes
        -----
        '''
        phase_iter = range(self.liquid_count)
        betas = self.liquids_betas
        MWs_phases = [i.MW() for i in self.liquids]
        tot = 0.0
        for i in phase_iter:
            tot += MWs_phases[i]*betas[i]
        tot_inv = 1.0/tot
        return [betas[i]*MWs_phases[i]*tot_inv for i in phase_iter]

    @property
    def betas_volume_liquids(self):
        r'''Method to calculate and return the fraction of the liquid phase
        that each liquid phase is, by volume phase fraction.
        If the system is VLLL with volume phase fractions of 0.125 vapor, and
        [.25, .125, .5] for the three liquids phases respectively, the return
        value would be [0.28571428, 0.142857142, 0.57142857].

        Returns
        -------
        betas_volume_liquids : list[float]
            Volume phase fractions of the overall liquid phase, [-]

        Notes
        -----
        '''
        phase_iter = range(self.liquid_count)
        betas = self.liquids_betas
        Vs_phases = [i.V() for i in self.liquids]

        tot = 0.0
        for i in phase_iter:
            tot += Vs_phases[i]*betas[i]
        tot_inv = 1.0/tot
        return [betas[i]*Vs_phases[i]*tot_inv for i in phase_iter]

    def V_liquids_ref(self):
        r'''Method to calculate and return the liquid reference molar volumes
        according to the temperature variable `T_liquid_volume_ref` of
        :obj:`thermo.bulk.BulkSettings`.

        Returns
        -------
        V_liquids_ref : list[float]
            Liquid molar volumes at the reference condition, [m^3/mol]

        Notes
        -----
        '''
        T_liquid_volume_ref = self.settings.T_liquid_volume_ref
        if T_liquid_volume_ref == 298.15:
            Vls = self.Vml_STPs
        elif T_liquid_volume_ref == 288.7055555555555:
            Vls = self.Vml_60Fs
        else:
            Vls = [i(T_liquid_volume_ref) for i in self.VolumeLiquids]
        return Vls

    def V_liquid_ref(self, phase=None):
        r'''Method to calculate and return the liquid reference molar volume
        according to the temperature variable `T_liquid_volume_ref` of
        :obj:`thermo.bulk.BulkSettings` and the composition of the phase.

        .. math::
            V = \sum_i z_i V_i

        Returns
        -------
        V_liquid_ref : float
            Liquid molar volume at the reference condition, [m^3/mol]

        Notes
        -----
        '''
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

    def rho_mass_liquid_ref(self, phase=None):
        r'''Method to calculate and return the liquid reference mass density
        according to the temperature variable `T_liquid_volume_ref` of
        :obj:`thermo.bulk.BulkSettings` and the composition of the phase.

        Returns
        -------
        rho_mass_liquid_ref : float
            Liquid mass density at the reference condition, [kg/m^3]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk

        V = self.V_liquid_ref(phase)
        MW = phase.MW()
        return Vm_to_rho(V, MW)

    def atom_fractions(self, phase=None):
        r'''Method to calculate and return the atomic composition of the phase;
        returns a dictionary of atom fraction (by count), containing only those
        elements who are present.

        Returns
        -------
        atom_fractions : dict[str: float]
            Atom fractions, [-]

        Notes
        -----
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
        r'''Method to calculate and return the atomic mass fractions of the phase;
        returns a dictionary of atom fraction (by mass), containing only those
        elements who are present.

        Returns
        -------
        atom_mass_fractions : dict[str: float]
            Atom mass fractions, [-]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk
        zs = phase.zs
        things = {}
        for zi, atoms in zip(zs, self.constants.atomss):
            for atom, count in atoms.items():
                if atom in things:
                    things[atom] += zi*count
                else:
                    things[atom] = zi*count
        return mass_fractions(things, phase.MW())

    def ws(self, phase=None):
        r'''Method to calculate and return the mass fractions of the phase, [-]

        Returns
        -------
        ws : list[float]
            Mass fractions, [-]

        Notes
        -----
        '''
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
        return zs_to_ws(zs, self.constants.MWs)

    def Vfls(self, phase=None):
        r'''Method to calculate and return the ideal-liquid volume fractions of
        the components of the phase, using the standard liquid densities at
        the temperature variable `T_liquid_volume_ref` of
        :obj:`thermo.bulk.BulkSettings` and the composition of the phase.

        Returns
        -------
        Vfls : list[float]
            Ideal-liquid volume fractions of the components of the phase, [-]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the ideal-gas volume fractions of
        the components of the phase. This is the same as the mole fractions.

        Returns
        -------
        Vfgs : list[float]
            Ideal-gas volume fractions of the components of the phase, [-]

        Notes
        -----
        '''
        # TODO: use partial molar volumes or something to compute an acutal
        # gas volume fractions?
        if phase is None:
            phase = self.bulk
            zs = self.zs
        else:
            zs = phase.zs
        return zs

    def MW(self, phase=None):
        r'''Method to calculate and return the molecular weight of the phase.

        .. math::
            \text{MW} = \sum_i z_i \text{MW}_{i}

        Returns
        -------
        MW : float
            Molecular weight of the phase, [g/mol]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the pseudocritical temperature
        calculated using Kay's rule (linear mole fractions):

        .. math::
            T_{c, pseudo} = \sum_i z_i T_{c,i}

        Returns
        -------
        pseudo_Tc : float
            Pseudocritical temperature of the phase, [K]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the pseudocritical pressure
        calculated using Kay's rule (linear mole fractions):

        .. math::
            P_{c, pseudo} = \sum_i z_i P_{c,i}

        Returns
        -------
        pseudo_Pc : float
            Pseudocritical pressure of the phase, [Pa]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the pseudocritical volume
        calculated using Kay's rule (linear mole fractions):

        .. math::
            V_{c, pseudo} = \sum_i z_i V_{c,i}

        Returns
        -------
        pseudo_Vc : float
            Pseudocritical volume of the phase, [m^3/mol]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the pseudocritical compressibility
        calculated using Kay's rule (linear mole fractions):

        .. math::
            Z_{c, pseudo} = \sum_i z_i Z_{c,i}

        Returns
        -------
        pseudo_Zc : float
            Pseudocritical compressibility of the phase, [-]

        Notes
        -----
        '''
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs

        Zcs = self.constants.Zcs
        Zc = 0.0
        for i in self.cmps:
            Zc += zs[i]*Zcs[i]
        return Zc


    def Tmc(self, phase=None):
        r'''Method to calculate and return the mechanical critical temperature
        of the phase.

        Returns
        -------
        Tmc : float
            Mechanical critical temperature, [K]
        '''
        if phase is None:
            phase = self.bulk
        return phase.Tmc()

    def Pmc(self, phase=None):
        r'''Method to calculate and return the mechanical critical pressure
        of the phase.

        Returns
        -------
        Pmc : float
            Mechanical critical pressure, [Pa]
        '''
        if phase is None:
            phase = self.bulk
        return phase.Pmc()

    def Vmc(self, phase=None):
        r'''Method to calculate and return the mechanical critical volume
        of the phase.

        Returns
        -------
        Vmc : float
            Mechanical critical volume, [m^3/mol]
        '''
        if phase is None:
            phase = self.bulk
        return phase.Vmc()

    def Zmc(self, phase=None):
        r'''Method to calculate and return the mechanical critical
        compressibility of the phase.

        Returns
        -------
        Zmc : float
            Mechanical critical compressibility, [-]
        '''
        if phase is None:
            phase = self.bulk
        return phase.Zmc()

    def rho_mass(self, phase=None):
        r'''Method to calculate and return mass density of the phase.

        .. math::
            \rho = \frac{MW}{1000\cdot VM}

        Returns
        -------
        rho_mass : float
            Mass density, [kg/m^3]
        '''
        if phase is None:
            phase = self.bulk

        V = phase.V()
        MW = phase.MW()
        return Vm_to_rho(V, MW)

    def V_mass(self, phase=None):
        r'''Method to calculate and return the specific volume of the phase.

        .. math::
            V_{mass} = \frac{1000\cdot VM}{MW}

        Returns
        -------
        V_mass : float
            Specific volume of the phase, [m^3/kg]
        '''
        if phase is None:
            phase = self.bulk

        V = phase.V()
        MW = phase.MW()
        return 1.0/Vm_to_rho(V, MW)

    def H_mass(self, phase=None):
        r'''Method to calculate and return mass enthalpy of the phase.

        .. math::
            H_{mass} = \frac{1000 H_{molar}}{MW}

        Returns
        -------
        H_mass : float
            Mass enthalpy, [J/kg]
        '''
        if phase is None:
            phase = self.bulk
        return phase.H()*1e3*phase.MW_inv()

    def S_mass(self, phase=None):
        r'''Method to calculate and return mass entropy of the phase.

        .. math::
            S_{mass} = \frac{1000 S_{molar}}{MW}

        Returns
        -------
        S_mass : float
            Mass enthalpy, [J/(kg*K)]
        '''
        if phase is None:
            phase = self.bulk
        return phase.S()*1e3*phase.MW_inv()

    def U_mass(self, phase=None):
        r'''Method to calculate and return mass internal energy of the phase.

        .. math::
            U_{mass} = \frac{1000 U_{molar}}{MW}

        Returns
        -------
        U_mass : float
            Mass internal energy, [J/(kg)]
        '''
        if phase is None:
            phase = self.bulk
        return phase.U()*1e3*phase.MW_inv()

    def A_mass(self, phase=None):
        r'''Method to calculate and return mass Helmholtz energy of the phase.

        .. math::
            A_{mass} = \frac{1000 A_{molar}}{MW}

        Returns
        -------
        A_mass : float
            Mass Helmholtz energy, [J/(kg)]
        '''
        if phase is None:
            phase = self.bulk
        return phase.A()*1e3*phase.MW_inv()

    def G_mass(self, phase=None):
        r'''Method to calculate and return mass Gibbs energy of the phase.

        .. math::
            G_{mass} = \frac{1000 G_{molar}}{MW}

        Returns
        -------
        G_mass : float
            Mass Gibbs energy, [J/(kg)]
        '''
        if phase is None:
            phase = self.bulk
        return phase.G()*1e3*phase.MW_inv()

    def Cp_mass(self, phase=None):
        r'''Method to calculate and return mass constant pressure heat capacity
        of the phase.

        .. math::
            Cp_{mass} = \frac{1000 Cp_{molar}}{MW}

        Returns
        -------
        Cp_mass : float
            Mass heat capacity, [J/(kg*K)]
        '''
        if phase is None:
            phase = self.bulk
        return phase.Cp()*1e3*phase.MW_inv()

    def Cv_mass(self, phase=None):
        r'''Method to calculate and return mass constant volume heat capacity
        of the phase.

        .. math::
            Cv_{mass} = \frac{1000 Cv_{molar}}{MW}

        Returns
        -------
        Cv_mass : float
            Mass constant volume heat capacity, [J/(kg*K)]
        '''
        if phase is None:
            phase = self.bulk
        return phase.Cv()*1e3*phase.MW_inv()

    def Cp_ideal_gas(self, phase=None):
        r'''Method to calculate and return the ideal-gas heat capacity of the
        phase.

        .. math::
            C_p^{ig} = \sum_i z_i {C_{p,i}^{ig}}

        Returns
        -------
        Cp : float
            Ideal gas heat capacity, [J/(mol*K)]
        '''
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
        r'''Method to calculate and return the difference between the actual
        `H` and the ideal-gas enthalpy of the phase.

        .. math::
            H^{dep} = H - H^{ig}

        Returns
        -------
        H_dep : float
            Departure enthalpy, [J/(mol)]
        '''
        if phase is None:
            phase = self.bulk
        elif phase is not self.bulk:
            try:
                return phase.H_dep()
            except:
                pass
        return phase.H() - self.H_ideal_gas(phase)

    def S_dep(self, phase=None):
        r'''Method to calculate and return the difference between the actual
        `S` and the ideal-gas entropy of the phase.

        .. math::
            S^{dep} = S - S^{ig}

        Returns
        -------
        S_dep : float
            Departure entropy, [J/(mol*K)]
        '''
        if phase is None:
            phase = self.bulk
        elif phase is not self.bulk:
            try:
                return phase.S_dep()
            except:
                pass
        return phase.S() - self.S_ideal_gas(phase)

    def Cp_dep(self, phase=None):
        r'''Method to calculate and return the difference between the actual
        `Cp` and the ideal-gas heat
        capacity :math:`C_p^{ig}` of the phase.

        .. math::
            C_p^{dep} = C_p - C_p^{ig}

        Returns
        -------
        Cp_dep : float
            Departure ideal gas heat capacity, [J/(mol*K)]
        '''
        if phase is None:
            phase = self.bulk
        elif phase is not self.bulk:
            try:
                return phase.Cp_dep()
            except:
                pass
        return phase.Cp() - self.Cp_ideal_gas(phase)

    def Cv_dep(self, phase=None):
        r'''Method to calculate and return the difference between the actual
        `Cv` and the ideal-gas constant volume heat
        capacity :math:`C_v^{ig}` of the phase.

        .. math::
            C_v^{dep} = C_v - C_v^{ig}

        Returns
        -------
        Cv_dep : float
            Departure ideal gas constant volume heat capacity, [J/(mol*K)]
        '''
        if phase is None:
            phase = self.bulk
        elif phase is not self.bulk:
            try:
                return phase.Cv_dep()
            except:
                pass
        return phase.Cv() - self.Cv_ideal_gas(phase)

    def H_ideal_gas(self, phase=None):
        r'''Method to calculate and return the ideal-gas enthalpy of the phase.

        .. math::
            H^{ig} = \sum_i z_i {H_{i}^{ig}}

        Returns
        -------
        H : float
            Ideal gas enthalpy, [J/(mol)]
        '''
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
        r'''Method to calculate and return the ideal-gas entropy of the phase.

        .. math::
            S^{ig} = \sum_i z_i S_{i}^{ig} - R\log\left(\frac{P}{P_{ref}}\right)
            - R\sum_i z_i \log(z_i)

        Returns
        -------
        S : float
            Ideal gas molar entropy, [J/(mol*K)]
        '''
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
        r'''Method to calculate and return the ideal-gas constant volume heat
        capacity of the phase.

        .. math::
            C_v^{ig} = \sum_i z_i {C_{p,i}^{ig}} - R

        Returns
        -------
        Cv : float
            Ideal gas constant volume heat capacity, [J/(mol*K)]
        '''
        if phase is None:
            phase = self.bulk
        return self.Cp_ideal_gas(phase) - R

    def Cp_Cv_ratio_ideal_gas(self, phase=None):
        r'''Method to calculate and return the ratio of the ideal-gas heat
        capacity to its constant-volume heat capacity.

        .. math::
            \frac{C_p^{ig}}{C_v^{ig}}

        Returns
        -------
        Cp_Cv_ratio_ideal_gas : float
            Cp/Cv for the phase as an ideal gas, [-]
        '''
        return self.Cp_ideal_gas(phase)/self.Cv_ideal_gas(phase)

    def G_ideal_gas(self, phase=None):
        r'''Method to calculate and return the ideal-gas Gibbs free energy of
        the phase.

        .. math::
            G^{ig} = H^{ig} - T S^{ig}

        Returns
        -------
        G_ideal_gas : float
            Ideal gas free energy, [J/(mol)]
        '''
        G_ideal_gas = self.H_ideal_gas(phase) - self.T*self.S_ideal_gas(phase)
        return G_ideal_gas

    def U_ideal_gas(self, phase=None):
        r'''Method to calculate and return the ideal-gas internal energy of
        the phase.

        .. math::
            U^{ig} = H^{ig} - P V^{ig}

        Returns
        -------
        U_ideal_gas : float
            Ideal gas internal energy, [J/(mol)]
        '''
        U_ideal_gas = self.H_ideal_gas(phase) - self.P*self.V_ideal_gas(phase)
        return U_ideal_gas

    def A_ideal_gas(self, phase=None):
        r'''Method to calculate and return the ideal-gas Helmholtz energy of
        the phase.

        .. math::
            A^{ig} = U^{ig} - T S^{ig}

        Returns
        -------
        A_ideal_gas : float
            Ideal gas Helmholtz free energy, [J/(mol)]
        '''
        A_ideal_gas = self.U_ideal_gas(phase) - self.T*self.S_ideal_gas(phase)
        return A_ideal_gas

    def V_ideal_gas(self, phase=None):
        r'''Method to calculate and return the ideal-gas molar volume of the
        phase.

        .. math::
            V^{ig} = \frac{RT}{P}

        Returns
        -------
        V : float
            Ideal gas molar volume, [m^3/mol]
        '''
        return R*self.T/self.P


    def H_formation_ideal_gas(self, phase=None):
        r'''Method to calculate and return the ideal-gas enthalpy of formation
        of the phase (as if the phase was an ideal gas).

        .. math::
            H_{reactive}^{ig} = \sum_i z_i {H_{f,i}}

        Returns
        -------
        H_formation_ideal_gas : float
            Enthalpy of formation of the phase on a reactive basis
            as an ideal gas, [J/mol]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the ideal-gas entropy of formation
        of the phase (as if the phase was an ideal gas).

        .. math::
            S_{reactive}^{ig} = \sum_i z_i {S_{f,i}}

        Returns
        -------
        S_formation_ideal_gas : float
            Entropy of formation of the phase on a reactive basis
            as an ideal gas, [J/(mol*K)]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the ideal-gas Gibbs free energy of
        formation of the phase (as if the phase was an ideal gas).

        .. math::
            G_{reactive}^{ig} = H_{reactive}^{ig} - T_{ref}^{ig}
            S_{reactive}^{ig}

        Returns
        -------
        G_formation_ideal_gas : float
            Gibbs free energy of formation of the phase on a reactive basis
            as an ideal gas, [J/(mol)]

        Notes
        -----
        '''
        Gf = self.H_formation_ideal_gas(phase) - self.T_REF_IG*self.S_formation_ideal_gas(phase)
        return Gf

    def U_formation_ideal_gas(self, phase=None):
        r'''Method to calculate and return the ideal-gas internal energy of
        formation of the phase (as if the phase was an ideal gas).

        .. math::
            U_{reactive}^{ig} = H_{reactive}^{ig} - P_{ref}^{ig}
            V^{ig}

        Returns
        -------
        U_formation_ideal_gas : float
            Internal energy of formation of the phase on a reactive basis
            as an ideal gas, [J/(mol)]

        Notes
        -----
        '''
        Uf = self.H_formation_ideal_gas(phase) - self.P_REF_IG*self.V_ideal_gas(phase)
        return Uf

    def A_formation_ideal_gas(self, phase=None):
        r'''Method to calculate and return the ideal-gas Helmholtz energy of
        formation of the phase (as if the phase was an ideal gas).

        .. math::
            A_{reactive}^{ig} = U_{reactive}^{ig} - T_{ref}^{ig}
            S_{reactive}^{ig}

        Returns
        -------
        A_formation_ideal_gas : float
            Helmholtz energy of formation of the phase on a reactive basis
            as an ideal gas, [J/(mol)]

        Notes
        -----
        '''
        Af = self.U_formation_ideal_gas(phase) - self.T_REF_IG*self.S_formation_ideal_gas(phase)
        return Af

    def alpha(self, phase=None):
        r'''Method to calculate and return the thermal diffusivity of the
        equilibrium state.

        .. math::
            \alpha = \frac{k}{\rho Cp}

        Returns
        -------
        alpha : float
            Thermal diffusivity, [m^2/s]

        Notes
        -----
        '''
        rho = self.rho_mass(phase)
        k = self.k(phase)
        Cp = self.Cp_mass(phase)
        return thermal_diffusivity(k=k, rho=rho, Cp=Cp)


#    def mu(self, phase=None):
#        if phase is None:
#            if self.phase_count == 1:
#                phase = self.phases[0]
#            else:
#                phase = None
#                for beta, p in zip(self.betas, self.phases):
#                    if beta == 1.0:
#                        phase = p
#                        break
#                if phase is None:
#                    phase = self.bulk
#        if phase is not self.bulk:
#            return phase.mu()
#        if isinstance(phase, gas_phases):
#            return self.correlations.ViscosityGasMixture.mixture_property(phase.T, phase.P, phase.zs, phase.ws())
#        elif isinstance(phase, liquid_phases):
#            return self.correlations.ViscosityLiquidMixture.mixture_property(phase.T, phase.P, phase.zs, phase.ws())
#        else:
#            raise NotImplementedError("no bulk methods")

    def nu(self, phase=None):
        r'''Method to calculate and return the kinematic viscosity of the
        equilibrium state.

        .. math::
            \nu = \frac{\mu}{\rho}

        Returns
        -------
        nu : float
            Kinematic viscosity, [m^2/s]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the standard liquid specific gravity
        of the phase, using constant liquid pure component densities not
        calculated by the phase object, at 60 °F.

        Returns
        -------
        SG : float
            Specific gravity of the liquid, [-]

        Notes
        -----
        The reference density of water is from the IAPWS-95 standard -
        999.0170824078306 kg/m^3.
        '''
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
        r'''Method to calculate and return the specific gravity of the phase
        with respect to a gas reference density.

        Returns
        -------
        SG_gas : float
            Specific gravity of the gas, [-]

        Notes
        -----
        The reference molecular weight of air used is 28.9586 g/mol.
        '''
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
        r'''Method to calculate and return the ideal-gas molar volume of the
        phase at the standard temperature and pressure,  according to the
        temperature variable `T_standard` and pressure variable `P_standard`
        of the :obj:`thermo.bulk.BulkSettings`.


        .. math::
            V^{ig} = \frac{RT_{std}}{P_{std}}

        Returns
        -------
        V_gas_standard : float
            Ideal gas molar volume at standard temperature and pressure,
            [m^3/mol]
        '''
        return R*self.settings.T_standard/self.settings.P_standard

    def V_gas_normal(self, phase=None):
        r'''Method to calculate and return the ideal-gas molar volume of the
        phase at the normal temperature and pressure,  according to the
        temperature variable `T_normal` and pressure variable `P_normal`
        of the :obj:`thermo.bulk.BulkSettings`.


        .. math::
            V^{ig} = \frac{RT_{norm}}{P_{norm}}

        Returns
        -------
        V_gas_normal : float
            Ideal gas molar volume at normal temperature and pressure,
            [m^3/mol]
        '''
        return R*self.settings.T_normal/self.settings.P_normal


    def Bvirial(self, phase=None):
        r'''Method to calculate and return the `B` virial coefficient of the
        phase at its current conditions.

        Returns
        -------
        Bvirial : float
            Virial coefficient, [m^3/mol]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk
        return B_from_Z(phase.Z(), self.T, self.P)

    def H_C_ratio(self, phase=None):
        r'''Method to calculate and return the atomic ratio of hydrogen atoms
        to carbon atoms, based on the current composition of the phase.

        Returns
        -------
        H_C_ratio : float
            H/C ratio on a molar basis, [-]

        Notes
        -----
        None is returned if no species are present that have carbon atoms.
        '''
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
        r'''Method to calculate and return the mass ratio of hydrogen atoms
        to carbon atoms, based on the current composition of the phase.

        Returns
        -------
        H_C_ratio_mass : float
            H/C ratio on a mass basis, [-]

        Notes
        -----
        None is returned if no species are present that have carbon atoms.
        '''
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
    def lightest_liquid(self):
        r'''The liquid-like phase with the lowest mass density, [-]

        Returns
        -------
        lightest_liquid : Phase or None
            Phase with the lowest mass density or None if there are no liquid
            like phases, [-]

        Notes
        -----
        '''
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
        r'''The liquid-like phase with the highest mass density, [-]

        Returns
        -------
        heaviest_liquid : Phase or None
            Phase with the highest mass density or None if there are no liquid
            like phases, [-]

        Notes
        -----
        '''
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
    def water_phase_index(self):
        r'''The liquid-like phase with the highest mole fraction of water, [-]

        Returns
        -------
        water_phase_index : int
            Index into the attribute :obj:`EquilibriumState.liquids` which
            refers to the liquid-like phase with the highest water mole
            fraction, [-]

        Notes
        -----
        '''
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
    def water_phase(self):
        r'''The liquid-like phase with the highest water mole fraction, [-]

        Returns
        -------
        water_phase : Phase or None
            Phase with the highest water mole fraction or None if there are no
            liquid like phases with water, [-]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the molar water content; this is
        the g/mol of the fluid which is coming from water, [g/mol].

        .. math::
            \text{water content} = \text{MW}_{H2O} w_{H2O}

        Returns
        -------
        molar_water_content : float
            Molar water content, [g/mol]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk

        water_index = self.water_index
        if water_index is None:
            return 0.0

        MW_water = self.constants.MWs[water_index]
        return MW_water*phase.ws()[water_index]

    def zs_no_water(self, phase=None):
        r'''Method to calculate and return the mole fractions of all species
        in the phase, normalized to a water-free basis (the mole fraction of
        water returned is zero).

        Returns
        -------
        zs_no_water : list[float]
            Mole fractions on a water free basis, [-]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the mass fractions of all species
        in the phase, normalized to a water-free basis (the mass fraction of
        water returned is zero).

        Returns
        -------
        ws_no_water : list[float]
            Mass fractions on a water free basis, [-]

        Notes
        -----
        '''
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
        r'''Method to calculate and return the molar ideal-gas higher heat of
        combustion of the object, [J/mol]

        Returns
        -------
        Hc : float
            Molar higher heat of combustion, [J/(mol)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk
        return mixing_simple(self.constants.Hcs, phase.zs)

    def Hc_mass(self, phase=None):
        r'''Method to calculate and return the mass ideal-gas higher heat of
        combustion of the object, [J/mol]

        Returns
        -------
        Hc_mass : float
            Mass higher heat of combustion, [J/(kg)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk
        return mixing_simple(self.constants.Hcs_mass, phase.ws())

    def Hc_normal(self, phase=None):
        r'''Method to calculate and return the volumetric ideal-gas higher heat
        of combustion of the object using the normal gas volume, [J/m^3]

        Returns
        -------
        Hc_normal : float
            Volumetric (normal) higher heat of combustion, [J/(m^3)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk
        return phase.Hc()/phase.V_gas_normal()

    def Hc_standard(self, phase=None):
        r'''Method to calculate and return the volumetric ideal-gas higher heat
        of combustion of the object using the standard gas volume, [J/m^3]

        Returns
        -------
        Hc_normal : float
            Volumetric (standard) higher heat of combustion, [J/(m^3)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk
        return phase.Hc()/phase.V_gas_standard()

    def Hc_lower(self, phase=None):
        r'''Method to calculate and return the molar ideal-gas lower heat of
        combustion of the object, [J/mol]

        Returns
        -------
        Hc_lower : float
            Molar lower heat of combustion, [J/(mol)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk
        return mixing_simple(self.constants.Hcs_lower, phase.zs)

    def Hc_lower_mass(self, phase=None):
        r'''Method to calculate and return the mass ideal-gas lower heat of
        combustion of the object, [J/mol]

        Returns
        -------
        Hc_lower_mass : float
            Mass lower heat of combustion, [J/(kg)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk
        return mixing_simple(self.constants.Hcs_lower_mass, phase.ws())

    def Hc_lower_normal(self, phase=None):
        r'''Method to calculate and return the volumetric ideal-gas lower heat
        of combustion of the object using the normal gas volume, [J/m^3]

        Returns
        -------
        Hc_lower_normal : float
            Volumetric (normal) lower heat of combustion, [J/(m^3)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk
        return phase.Hc_lower()/phase.V_gas_normal()

    def Hc_lower_standard(self, phase=None):
        r'''Method to calculate and return the volumetric ideal-gas lower heat
        of combustion of the object using the standard gas volume, [J/m^3]

        Returns
        -------
        Hc_lower_standard : float
            Volumetric (standard) lower heat of combustion, [J/(m^3)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk
        return phase.Hc_lower()/phase.V_gas_standard()


    def Wobbe_index(self, phase=None):
        r'''Method to calculate and return the molar Wobbe index of the object,
        [J/mol].

        .. math::
            I_W = \frac{H_{comb}^{higher}}{\sqrt{\text{SG}}}

        Returns
        -------
        Wobbe_index : float
            Molar Wobbe index, [J/(mol)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk
        Hc = abs(phase.Hc())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_mass(self, phase=None):
        r'''Method to calculate and return the mass Wobbe index of the object,
        [J/kg].

        .. math::
            I_W = \frac{H_{comb}^{higher}}{\sqrt{\text{SG}}}

        Returns
        -------
        Wobbe_index_mass : float
            Mass Wobbe index, [J/(kg)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk

        Hc = abs(phase.Hc_mass())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_lower(self, phase=None):
        r'''Method to calculate and return the molar lower Wobbe index of the
         object, [J/mol].

        .. math::
            I_W = \frac{H_{comb}^{lower}}{\sqrt{\text{SG}}}

        Returns
        -------
        Wobbe_index_lower : float
            Molar lower Wobbe index, [J/(mol)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk

        Hc = abs(phase.Hc_lower())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_lower_mass(self, phase=None):
        r'''Method to calculate and return the lower mass Wobbe index of the
        object, [J/kg].

        .. math::
            I_W = \frac{H_{comb}^{lower}}{\sqrt{\text{SG}}}

        Returns
        -------
        Wobbe_index_lower_mass : float
            Mass lower Wobbe index, [J/(kg)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk

        Hc = abs(phase.Hc_lower_mass())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5


    def Wobbe_index_standard(self, phase=None):
        r'''Method to calculate and return the volumetric standard Wobbe index
        of the object, [J/m^3]. The standard gas volume is used in this
        calculation.

        .. math::
            I_W = \frac{H_{comb}^{higher}}{\sqrt{\text{SG}}}

        Returns
        -------
        Wobbe_index_standard : float
            Volumetric standard Wobbe index, [J/(m^3)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk

        Hc = abs(phase.Hc_standard())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_normal(self, phase=None):
        r'''Method to calculate and return the volumetric normal Wobbe index
        of the object, [J/m^3]. The normal gas volume is used in this
        calculation.

        .. math::
            I_W = \frac{H_{comb}^{higher}}{\sqrt{\text{SG}}}

        Returns
        -------
        Wobbe_index : float
            Volumetric normal Wobbe index, [J/(m^3)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk

        Hc = abs(phase.Hc_normal())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_lower_standard(self, phase=None):
        r'''Method to calculate and return the volumetric standard lower Wobbe
        index of the object, [J/m^3]. The standard gas volume is used in this
        calculation.

        .. math::
            I_W = \frac{H_{comb}^{lower}}{\sqrt{\text{SG}}}

        Returns
        -------
        Wobbe_index_lower_standard : float
            Volumetric standard lower Wobbe index, [J/(m^3)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk

        Hc = abs(phase.Hc_lower_standard())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def Wobbe_index_lower_normal(self, phase=None):
        r'''Method to calculate and return the volumetric normal lower Wobbe
        index of the object, [J/m^3]. The normal gas volume is used in this
        calculation.

        .. math::
            I_W = \frac{H_{comb}^{lower}}{\sqrt{\text{SG}}}

        Returns
        -------
        Wobbe_index_lower_normal : float
            Volumetric normal lower Wobbe index, [J/(m^3)]

        Notes
        -----
        '''
        if phase is None:
            phase = self.bulk

        Hc = abs(phase.Hc_lower_normal())
        SG_gas = phase.SG_gas()
        return Hc*SG_gas**-0.5

    def value(self, name, phase=None):
        r'''Method to retrieve a property from a string. This more or less
        wraps `getattr`, but also allows for the property to be returned for a
        specific phase if `phase` is provided.

        `name` could be a python property like 'Tms' or a callable method
        like 'H'; and if the property is on a per-phase basis like 'betas_mass',
        a phase object can be provided as the second argument and only the
        value for that phase will be returned.

        Parameters
        ----------
        name : str
            String representing the property, [-]
        phase : :obj:`thermo.phase.Phase`, optional
            Phase to retrieve the property for only (if specified), [-]

        Returns
        -------
        value : various
            Value specified, [various]

        Notes
        -----
        '''
        if phase is not None:
            phase_idx = self.phases.index(phase)

        v = getattr(self, name)
        try:
            v = v()
        except:
            pass
        if phase is not None:
            return v[phase_idx]
        return v

    @property
    def IDs(self):
        '''Alias of CASs.'''
        return self.constants.CASs

    def API(self, phase=None):
        if phase is None:
            phase = self.bulk
        # Going to get a list of liquid phasee models, determine which model
        # to use. Construct a new phase object, get the density


# Add some fancy things for easier access to properties

def _make_getter_constants(name):
    def get(self):
        return getattr(self.constants, name)
    return get

def _make_getter_correlations(name):
    def get(self):
        return getattr(self.correlations, name)

    text = '''Wrapper to obtain the list of %s objects of the associated
:obj:`thermo.chemical_package.PropertyCorrelationPackage`.''' %(name)
    get.__doc__ = text
    return get

def _make_getter_EquilibriumState(name):
    def get(self):
        return getattr(self.result, name)(self)
    get.__doc__ = getattr(EquilibriumState, name).__doc__
    return get

def _make_getter_bulk_props(name):
    def get(self):
        return getattr(self.bulk, name)
    try:
        get.__doc__ = getattr(Bulk, name).__doc__
    except:
        pass
    return get

### For the pure component fixed properties, allow them to be retrived from the phase
# and bulk object as well as the Equilibrium State Object
constant_blacklist = set(['atom_fractions'])

for name in ChemicalConstantsPackage.properties:
    if name not in constant_blacklist:
        getter = property(_make_getter_constants(name))
        try:
            var_type, desc, units, return_desc = constants_docstrings[name]
            type_name = var_type if type(var_type) is str else var_type.__name__
            if return_desc is None:
                return_desc = desc
            full_desc = '''%s, %s.

Returns
-------
%s : %s
    %s, %s.''' %(desc, units, name, type_name, return_desc, units)
#            print(full_desc)
            getter.__doc__ = full_desc
        except:
            pass
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
                                         'rho_mass_liquid_ref', 'V_liquid_ref',
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
bulk_props = ['V', 'Z', 'rho', 'Cp', 'Cv', 'H', 'S', 'U', 'G', 'A', #'dH_dT', 'dH_dP', 'dS_dT', 'dS_dP',
              #'dU_dT', 'dU_dP', 'dG_dT', 'dG_dP', 'dA_dT', 'dA_dP',
              'H_reactive', 'S_reactive', 'G_reactive', 'U_reactive', 'A_reactive',
              'Cp_Cv_ratio', 'log_zs', 'isothermal_bulk_modulus',
              'dP_dT_frozen', 'dP_dV_frozen', 'd2P_dT2_frozen', 'd2P_dV2_frozen',
              'd2P_dTdV_frozen',
              'd2P_dTdV', 'd2P_dV2', 'd2P_dT2', 'dP_dV', 'dP_dT', 'isentropic_exponent',

              'PIP', 'kappa', 'isobaric_expansion', 'Joule_Thomson', 'speed_of_sound',
              'speed_of_sound_mass',
              'U_dep', 'G_dep', 'A_dep', 'V_dep', 'V_iter',
              'mu',
              ]
bulk_props += derivatives_thermodynamic
bulk_props += derivatives_thermodynamic_mass
bulk_props += derivatives_jacobian

for name in bulk_props:
    # Maybe take this out and implement it manually for performance?
    getter = property(_make_getter_bulk_props(name))
    setattr(EquilibriumState, name, getter)


