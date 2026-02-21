"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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


This module contains an object designed to store the result of a flash
calculation and provide convinient access to all properties of the calculated
phases and bulks.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

EquilibriumState
================
.. autoclass:: EquilibriumState
    :members:
    :undoc-members:
    :exclude-members: dH_dP_V, dH_dT_V, dH_dV_P, dH_dV_T, dS_dP_V, dS_dT, dS_dT_P, dS_dT_V
"""

__all__ = ["EquilibriumState"]

from chemicals.elements import periodic_table
from chemicals.utils import Vm_to_rho, hash_any_primitive, normalize, vapor_mass_quality, zs_to_ws
from fluids.constants import N_A, R
from fluids.numerics import log
from fluids.numerics import numpy as np

from thermo.bulk import Bulk, JsonOptEncodable, default_settings
from thermo.chemical_package import ChemicalConstantsPackage, PropertyCorrelationsPackage, constants_docstrings
from thermo.phases import Phase, derivatives_jacobian, derivatives_thermodynamic, derivatives_thermodynamic_mass, gas_phases, liquid_phases, solid_phases
from thermo.phases.phase import phase_shared_methods
from thermo.serialize import object_lookups

all_phases = gas_phases + liquid_phases + solid_phases

try:
    array = np.array
except:
    pass

CAS_H2O = "7732-18-5"

PHASE_GAS = "gas"
PHASE_LIQUID0 = "liquid0"
PHASE_LIQUID1 = "liquid1"
PHASE_LIQUID2 = "liquid2"
PHASE_LIQUID3 = "liquid3"
PHASE_BULK_LIQUID = "liquid_bulk"
PHASE_WATER_LIQUID = "water_phase"
PHASE_LIGHTEST_LIQUID = "lightest_liquid"
PHASE_HEAVIEST_LIQUID = "heaviest_liquid"
PHASE_SOLID0 = "solid0"
PHASE_SOLID1 = "solid1"
PHASE_SOLID2 = "solid2"
PHASE_SOLID3 = "solid3"
PHASE_BULK_SOLID = "solid_bulk"
PHASE_BULK = "bulk"

PHASE_REFERENCES = [PHASE_GAS, PHASE_LIQUID0, PHASE_LIQUID1, PHASE_LIQUID2,
                    PHASE_LIQUID3, PHASE_BULK_LIQUID, PHASE_WATER_LIQUID,
                    PHASE_LIGHTEST_LIQUID, PHASE_HEAVIEST_LIQUID, PHASE_SOLID0,
                    PHASE_SOLID1, PHASE_SOLID2, PHASE_SOLID3, PHASE_BULK_SOLID,
                    PHASE_BULK]

__all__.extend([
    "PHASE_BULK",
    "PHASE_BULK_LIQUID",
    "PHASE_BULK_SOLID",
    "PHASE_GAS",
    "PHASE_HEAVIEST_LIQUID",
    "PHASE_LIGHTEST_LIQUID",
    "PHASE_LIQUID0",
    "PHASE_LIQUID1",
    "PHASE_LIQUID2",
    "PHASE_LIQUID3",
    "PHASE_REFERENCES",
    "PHASE_SOLID0",
    "PHASE_SOLID1",
    "PHASE_SOLID2",
    "PHASE_SOLID3",
    "PHASE_WATER_LIQUID",
])

class EquilibriumState:
    r"""Class to represent a thermodynamic equilibrium state with one or more
    phases in it. This object is designed to be the output of the
    :obj:`thermo.flash.Flash` interface and to provide easy acess to all
    properties of the mixture.

    Properties like :obj:`Cp <EquilibriumState.Cp>` are calculated using the
    mixing rules configured by the
    :obj:`BulkSettings <thermo.bulk.BulkSettings>` object. For states with a
    single phase, this will always reduce to the properties of that phase.

    This interface allows calculation of thermodynamic properties,
    and transport properties. Both molar and mass outputs are provided, as
    separate calls (ex. :obj:`Cp <EquilibriumState.Cp>` and
    :obj:`Cp_mass <EquilibriumState.Cp_mass>`).

    Parameters
    ----------
    T : float
        Temperature of state, [K]
    P : float
        Pressure of state, [Pa]
    zs : list[float]
        Overall mole fractions of all species in the state, [-]
    gas : :obj:`Phase <thermo.phases.Phase>`
        The calcualted gas phase object, if one was found, [-]
    liquids : list[:obj:`Phase <thermo.phases.Phase>`]
        A list of liquid phase objects, if any were found, [-]
    solids : list[:obj:`Phase <thermo.phases.Phase>`]
        A list of solid phase objects, if any were found, [-]
    betas : list[float]
        Molar phase fractions of every phase, ordered [`gas beta`,
        `liquid beta0`, `liquid beta1`, ..., `solid beta0`, `solid beta1`, ...]
    flash_specs : dict[str : float], optional
        A dictionary containing the specifications for the flash calculations,
        [-]
    flash_convergence : dict[str : float], optional
        A dictionary containing the convergence results for the flash
        calculations; this is to help support development of the library only
        and the contents of this dictionary is subject to change, [-]
    constants : :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>`, optional
        Package of chemical constants; all cases these properties are
        accessible as attributes of this object, [-]
        :obj:`EquilibriumState <thermo.equilibrium.EquilibriumState>` object, [-]
    correlations : :obj:`PropertyCorrelationsPackage <thermo.chemical_package.PropertyCorrelationsPackage>`, optional
        Package of chemical T-dependent properties; these properties are
        accessible as attributes of this object object, [-]
    flasher : :obj:`Flash <thermo.flash.Flash>` object, optional
        This reference can be provided to this object to allow the object to
        return properties which are themselves calculated from results of flash
        calculations, [-]
    settings : :obj:`BulkSettings <thermo.bulk.BulkSettings>`, optional
        Object containing settings for calculating bulk and transport
        properties, [-]

    Examples
    --------
    The following sample shows a flash for the CO2-n-hexane system with all
    constants provided, using no data from thermo.

    >>> from thermo import *
    >>> constants = ChemicalConstantsPackage(names=['carbon dioxide', 'hexane'], CASs=['124-38-9', '110-54-3'], MWs=[44.0095, 86.17536], omegas=[0.2252, 0.2975], Pcs=[7376460.0, 3025000.0], Tbs=[194.67, 341.87], Tcs=[304.2, 507.6], Tms=[216.65, 178.075])
    >>> correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True,
    ...                                            HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
    ...                                                               HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998]))])
    >>> eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    >>> gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases)
    >>> liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases)
    >>> flasher = FlashVL(constants, correlations, liquid=liq, gas=gas)
    >>> state = flasher.flash(P=1e5, T=196.0, zs=[0.5, 0.5])
    >>> type(state) is EquilibriumState
    True
    >>> state.phase_count
    2
    >>> state.bulk.Cp()
    108.3164692
    >>> state.flash_specs
    {'zs': [0.5, 0.5], 'T': 196.0, 'P': 100000.0}
    >>> state.Tms
    [216.65, 178.075]
    >>> state.liquid0.H()
    -34376.4853
    >>> state.gas.H()
    -3608.0551

    Attributes
    ----------
    gas_count : int
        Number of gas phases present (0 or 1), [-]
    liquid_count : int
        Number of liquid phases present, [-]
    solid_count : int
        Number of solid phases present, [-]
    phase_count : int
        Number of phases present, [-]
    gas_beta : float
        Molar phase fraction of the gas phase; 0 if no gas phase is present,
        [-]
    liquids_betas : list[float]
        Liquid molar phase fractions, [-]
    solids_betas : list[float]
        Solid molar phase fractions, [-]
    liquid_zs : list[float]
        Overall mole fractions of each component in the overall liquid phase,
        [-]
    liquid_bulk : :obj:`Bulk<thermo.bulk.Bulk>`
        Liquid phase bulk, [-]
    solid_zs : list[float]
        Overall mole fractions of each component in the overall solid phase,
        [-]
    solid_bulk : :obj:`Bulk<thermo.bulk.Bulk>`
        Solid phase bulk, [-]
    bulk : :obj:`Bulk<thermo.bulk.Bulk>`
        Overall phase bulk, [-]
    """

    max_liquid_phases = 1
    reacted = False
    flashed = True
    vectorized = False # not supported yet

    liquid_bulk = None
    solid_bulk = None

    R = Phase.R
    T_REF_IG = Phase.T_REF_IG
    T_REF_IG_INV = Phase.T_REF_IG_INV
    P_REF_IG = Phase.P_REF_IG
    P_REF_IG_INV = Phase.P_REF_IG_INV

    __full_path__ = f"{__module__}.{__qualname__}"

    __slots__ = (
        "N",
        "P",
        "T",
        "__dict__",
        "betas",
        "bulk",
        "constants",
        "correlations",
        "flash_convergence",
        "flash_specs",
        "flasher",
        "gas",
        "gas_beta",
        "gas_count",
        "liquid0",
        "liquid1",
        "liquid2",
        "liquid_count",
        "liquid_zs", #'liquid_bulk',
        "liquids",
        "liquids_betas",
        "phase_count",
        "phases",
        "settings",
        "solid_count",
        "solids",
        "solids_betas",
        "zs",
    )

    obj_references = ("liquid_bulk", "solid_bulk", "bulk", "gas", "liquids", "phases",
                    "solids",  "settings", "constants", "correlations", "flasher",
                      "liquid0", "liquid1", "liquid2")

    def __eq__(self, other):
        return self.__hash__() == hash(other)

    def __hash__(self):
        r"""Basic method to calculate a hash of the state.

        Note that the hashes should only be compared on the same system running
        in the same process!

        Returns
        -------
        hash : int
            Hash of the state including the phases making it up [-]
        """
        return hash_any_primitive([self.phases, self.betas, self.gas_count, self.liquid_count, self.solid_count, self.settings, self.flasher])

    def __str__(self):
        s = "<EquilibriumState, T=%.4f, P=%.4f, zs=%s, betas=%s, phases=%s>"
        s = s %(self.T, self.P, self.zs, self.betas, str([str(i) for i in self.phases]).replace("'", ""))
        return s

    def __repr__(self):
        s = f"{self.__class__.__name__}(T={self.T}, P={self.P}, zs={self.zs}, betas={self.betas}"
        s += f", gas={self.gas.__repr__()}"
        s += f", liquids={self.liquids.__repr__()}"
        s += f", solids={self.solids.__repr__()}"
        s += ")"
        return s

    # __str__ = __repr__


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

        self.gas_count = gas_count = 1 if gas is not None else 0
        self.liquid_count = liquid_count = len(liquids)
        self.solid_count = solid_count = len(solids)

        self.phase_count = gas_count + liquid_count + solid_count

        self.gas = gas
        self.liquids = liquids
        self.solids = solids
        if gas is not None:
            self.phases = [gas] + liquids + solids
            gas.assigned_phase = "g"
        else:
            self.phases = liquids + solids

        self.betas = betas
        self.gas_beta = betas[0] if gas_count else 0.0
        self.liquids_betas = betas_liquids = betas[gas_count:gas_count + liquid_count]
        self.solids_betas = betas_solids = betas[gas_count + liquid_count:]


        try:
            V_liquids_ref = flasher.V_liquids_ref()
        except:
            V_liquids_ref = None

        if liquid_count > 1:
#                tot_inv = 1.0/sum(values)
#                return [i*tot_inv for i in values]
            self.liquid_zs = normalize([sum([betas_liquids[j]*liquids[j].zs[i] for j in range(liquid_count)])
                               for i in range(self.N)])
            self.liquid_bulk = liquid_bulk = Bulk(T, P, self.liquid_zs, self.liquids, self.liquids_betas, "l")
            liquid_bulk.flasher = flasher
            liquid_bulk.result = self
            liquid_bulk.constants = constants
            liquid_bulk.correlations = correlations
            liquid_bulk.settings = settings
            liquid_bulk._V_liquids_ref = V_liquids_ref
            liquid_bulk._gas_beta = betas[0] if gas_count else 0.0
            liquid_bulk._beta = sum(betas_liquids)
            for i, l in enumerate(liquids):
                setattr(self, f"liquid{i}", l)
                l.assigned_phase = "l"
        elif liquid_count:
            l = liquids[0]
            self.liquid_zs = l.zs
            self.liquid_bulk = l
            self.liquid0 = l
            l.assigned_phase = "l"

        if solids:
            self.solid_zs = normalize([sum([betas_solids[j]*solids[j].zs[i] for j in range(self.solid_count)])
                               for i in range(self.N)])
            self.solid_bulk = solid_bulk = Bulk(T, P, self.solid_zs, solids, self.solids_betas, "s")
            solid_bulk.result = self
            solid_bulk.constants = constants
            solid_bulk.correlations = correlations
            solid_bulk.flasher = flasher
            liquid_bulk.settings = settings
            solid_bulk._V_liquids_ref = V_liquids_ref
            solid_bulk._gas_beta = betas[0] if gas_count else 0.0
            solid_bulk._beta = sum(betas_solids)
            for i, s in enumerate(solids):
                setattr(self, f"solid{i}", s)

        self.bulk = bulk = Bulk(T, P, zs, self.phases, betas)
        bulk.result = self
        bulk.constants = constants
        bulk.correlations = correlations
        bulk.flasher = flasher
        bulk.settings = settings
        bulk._V_liquids_ref = V_liquids_ref
        bulk._gas_beta = betas[0] if gas_count else 0.0
        bulk._beta = 1.0
        bulk._beta_mass = 1.0

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
            phase.settings = settings
        gas_beta = self.gas_beta
        try:
            betas_mass = self.betas_mass
        except:
            betas_mass = [None]*self.phase_count
        try:
            betas_volume = self.betas_volume
        except:
            betas_volume = [None]*self.phase_count
        try:
            betas_volume_liquid_ref = self.betas_volume_liquid_ref
        except:
            betas_volume_liquid_ref = [None]*self.phase_count
        for i, phase in enumerate(self.phases):
            phase._V_liquids_ref = V_liquids_ref
            phase._beta = betas[i]
            phase._beta_mass = betas_mass[i]
            phase._beta_volume = betas_volume[i]
            phase._beta_volume_liquid_ref = betas_volume_liquid_ref[i]
            phase._gas_beta = gas_beta
        if liquid_count > 1:
            try:
                liquid_bulk._beta_mass = sum(betas_mass[gas_count:gas_count + liquid_count])
            except:
                liquid_bulk._beta_mass = None
        if solids:
            try:
                solid_bulk._beta_mass = sum(betas_mass[gas_count + liquid_count:])
            except:
                solid_bulk._beta_mass = None

    def as_json(self, cache=None, option=0):
        return JsonOptEncodable.as_json(self, cache, option)
    @classmethod
    def from_json(cls, json_repr, cache=None):
        return JsonOptEncodable.from_json(json_repr, cache)

    json_version = 1
    non_json_attributes = []

    @property
    def phase(self):
        r"""Method to calculate and return a string representing the phase of
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
        """
        s = ""
        if self.gas:
            s += "V"
        s += "L"*len(self.liquids)
        s += "S"*len(self.solids)
        return s

    @property
    def VF(self):
        r"""Method to return the vapor fraction of the equilibrium state.
        If no vapor/gas is present, 0 is always returned.

        Returns
        -------
        VF : float
            Vapor molar fraction, [-]

        Notes
        -----
        """
        if self.gas is not None:
            return self.betas[0]
        return 0.0 # No gas phase


    @property
    def LF(self):
        r"""Method to return the liquid fraction of the equilibrium state.
        If no liquid is present, 0 is always returned.

        Returns
        -------
        LF : float
            Liquid molar fraction, [-]

        Notes
        -----
        """
        return sum(self.liquids_betas)

    @property
    def quality(self):
        r"""Method to return the mass vapor fraction of the equilibrium state.
        If no vapor/gas is present, 0 is always returned. This is normally
        called the quality.

        Returns
        -------
        quality : float
            Vapor mass fraction, [-]

        Notes
        -----
        """
        try:
            return self._quality
        except:
            pass
        gas = self.gas
        liquid_bulk = self.liquid_bulk
        if gas is not None and liquid_bulk is not None:
            quality = vapor_mass_quality(self.gas_beta, MWl=liquid_bulk.MW(), MWg=gas.MW())
        elif gas is not None:
            quality = 1.0
        else:
            quality = 0.0
        self._quality = quality
        return quality

    @property
    def betas_states(self):
        r"""Method to return the molar phase fractions of each of the three
        fundamental `types` of phases.

        Returns
        -------
        betas_states : list[float, 3]
            List containing the molar phase fraction of gas, liquid, and solid,
            [-]

        Notes
        -----
        """
        try:
            return self._betas_states
        except:
            pass
        self._betas_states = [self.gas_beta, sum(self.liquids_betas), sum(self.solids_betas)]
        return self._betas_states

    @property
    def betas_mass_states(self):
        r"""Method to return the mass phase fractions of each of the three
        fundamental `types` of phases.

        Returns
        -------
        betas_mass_states : list[float, 3]
            List containing the mass phase fraction of gas, liquid, and solid,
            [-]

        Notes
        -----
        """
        try:
            return self._betas_mass_states
        except:
            pass
        g_tot = l_tot = s_tot = 0.0
        # Compute the mass fraction of the gas phase
        gas, liquids, solids = self.gas, self.liquids, self.solids
        beta_gas, betas_liquids, betas_solids = self.gas_beta, self.liquids_betas, self.solids_betas
        gas_MW = gas.MW() if gas is not None else 0.
        liq_MWs = [i.MW() for i in liquids]
        solid_MWs = [i.MW() for i in solids]

        g_tot = gas_MW*beta_gas
        for i in range(self.liquid_count):
            l_tot += liq_MWs[i]*betas_liquids[i]
        for i in range(self.solid_count):
            s_tot += solid_MWs[i]*betas_solids[i]
        tot = g_tot + l_tot + s_tot
        tot = 1.0/tot

        self._betas_mass_states = [g_tot*tot, l_tot*tot, s_tot*tot]
        return self._betas_mass_states

    @property
    def betas_volume_states(self):
        r"""Method to return the volume phase fractions of each of the three
        fundamental `types` of phases.

        Returns
        -------
        betas_volume_states : list[float, 3]
            List containing the volume phase fraction of gas, liquid, and solid,
            [-]

        Notes
        -----
        """
        try:
            return self._betas_volume_states
        except:
            pass
        g_tot = l_tot = s_tot = 0.0
        # Compute the mass fraction of the gas phase
        gas, liquids, solids = self.gas, self.liquids, self.solids
        beta_gas, betas_liquids, betas_solids = self.gas_beta, self.liquids_betas, self.solids_betas
        gas_V = gas.V() if gas is not None else 0.0
        liq_Vs = [i.V() for i in liquids]
        solid_Vs = [i.V() for i in solids]

        g_tot = gas_V*beta_gas
        for i in range(self.liquid_count):
            l_tot += liq_Vs[i]*betas_liquids[i]
        for i in range(self.solid_count):
            s_tot += solid_Vs[i]*betas_solids[i]
        tot = g_tot + l_tot + s_tot
        tot = 1.0/tot

        self._betas_volume_states = [g_tot*tot, l_tot*tot, s_tot*tot]
        return self._betas_volume_states


    @property
    def betas_mass(self):
        r"""Method to calculate and return the mass fraction of all of the
        phases in the system.

        Returns
        -------
        betas_mass : list[float]
            Mass phase fractions of all the phases, ordered vapor, liquid, then
            solid , [-]

        Notes
        -----
        """
        try:
            return self._betas_mass
        except:
            pass
        phase_iter = range(self.phase_count)
        betas = self.betas
        MWs_phases = [i.MW() for i in self.phases]

        tot = 0.0
        for i in phase_iter:
            tot += MWs_phases[i]*betas[i]
        tot_inv = 1.0/tot
        self._betas_mass = [betas[i]*MWs_phases[i]*tot_inv for i in phase_iter]
        return self._betas_mass

    @property
    def betas_volume(self):
        r"""Method to calculate and return the volume fraction of all of the
        phases in the system.

        Returns
        -------
        betas_volume : list[float]
            Volume phase fractions of all the phases, ordered vapor, liquid, then
            solid , [-]

        Notes
        -----
        """
        try:
            return self._betas_volume
        except:
            pass
        phase_iter = range(self.phase_count)
        betas = self.betas
        Vs_phases = [i.V() for i in self.phases]

        tot = 0.0
        for i in phase_iter:
            tot += Vs_phases[i]*betas[i]
        tot_inv = 1.0/tot
        self._betas_volume = [betas[i]*Vs_phases[i]*tot_inv for i in phase_iter]
        return self._betas_volume

    @property
    def betas_volume_liquid_ref(self):
        r"""Method to calculate and return the standard liquid volume fraction of all of the
        phases in the bulk.

        Returns
        -------
        betas_volume_liquid_ref : list[float]
            Standard liquid volume phase fractions of all the phases in the bulk, ordered
            vapor, liquid, then solid , [-]

        Notes
        -----
        """
        try:
            return self._betas_volume_liquid_ref
        except:
            pass
        phase_iter = range(self.phase_count)
        betas = self.betas
        Vs_phases = [i.V_liquid_ref() for i in self.phases]
        tot = 0.0
        for i in phase_iter:
            tot += Vs_phases[i]*betas[i]
        tot_inv = 1.0/tot
        self._betas_volume_liquid_ref = [betas[i]*Vs_phases[i]*tot_inv for i in phase_iter]
        return self._betas_volume_liquid_ref

    @property
    def betas_liquids(self):
        r"""Method to calculate and return the fraction of the liquid phase
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
        """
        try:
            return self._betas_liquids
        except:
            pass
        liquids_betas = self.liquids_betas
        tot = 0.0
        for vi in liquids_betas:
            tot += vi
        if tot == 0.0:
            return []
        tot = 1.0/tot
        self._betas_liquids = [vi*tot for vi in liquids_betas]
        return self._betas_liquids

    @property
    def betas_mass_liquids(self):
        r"""Method to calculate and return the fraction of the liquid phase
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
        """
        if self.liquid_count:
            phase_iter = range(self.liquid_count)
            betas = self.liquids_betas
            MWs_phases = [i.MW() for i in self.liquids]
            tot = 0.0
            for i in phase_iter:
                tot += MWs_phases[i]*betas[i]
            tot_inv = 1.0/tot
            return [betas[i]*MWs_phases[i]*tot_inv for i in phase_iter]
        else:
            return []

    @property
    def betas_volume_liquids(self):
        r"""Method to calculate and return the fraction of the liquid phase
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
        """
        if self.liquid_count:
            phase_iter = range(self.liquid_count)
            betas = self.liquids_betas
            Vs_phases = [i.V() for i in self.liquids]

            tot = 0.0
            for i in phase_iter:
                tot += Vs_phases[i]*betas[i]
            tot_inv = 1.0/tot
            return [betas[i]*Vs_phases[i]*tot_inv for i in phase_iter]
        else:
            return []

    def V_liquids_ref(self):
        r"""Method to calculate and return the liquid reference molar volumes
        according to the temperature variable `T_liquid_volume_ref` of
        :obj:`thermo.bulk.BulkSettings`.

        Returns
        -------
        V_liquids_ref : list[float]
            Liquid molar volumes at the reference condition, [m^3/mol]

        Notes
        -----
        """
        return self.flasher.V_liquids_ref()

    def ws(self, phase=None):
        r"""Method to calculate and return the mass fractions of the phase, [-]

        Returns
        -------
        ws : list[float]
            Mass fractions, [-]

        Notes
        -----
        """
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs
        return zs_to_ws(zs, self.constants.MWs)

    def MW(self, phase=None):
        r"""Method to calculate and return the molecular weight of the phase.

        .. math::
            \text{MW} = \sum_i z_i \text{MW}_{i}

        Returns
        -------
        MW : float
            Molecular weight of the phase, [g/mol]

        Notes
        -----
        """
        if phase is None:
            zs = self.zs
        else:
            zs = phase.zs

        MWs = self.constants.MWs
        MW = 0.0
        for i in range(self.N):
            MW += zs[i]*MWs[i]
        return MW

    def Tmc(self, phase=None):
        r"""Method to calculate and return the mechanical critical temperature
        of the phase.

        Returns
        -------
        Tmc : float
            Mechanical critical temperature, [K]
        """
        if phase is None:
            phase = self.bulk
        return phase.Tmc()

    def Pmc(self, phase=None):
        r"""Method to calculate and return the mechanical critical pressure
        of the phase.

        Returns
        -------
        Pmc : float
            Mechanical critical pressure, [Pa]
        """
        if phase is None:
            phase = self.bulk
        return phase.Pmc()

    def Vmc(self, phase=None):
        r"""Method to calculate and return the mechanical critical volume
        of the phase.

        Returns
        -------
        Vmc : float
            Mechanical critical volume, [m^3/mol]
        """
        if phase is None:
            phase = self.bulk
        return phase.Vmc()

    def Zmc(self, phase=None):
        r"""Method to calculate and return the mechanical critical
        compressibility of the phase.

        Returns
        -------
        Zmc : float
            Mechanical critical compressibility, [-]
        """
        if phase is None:
            phase = self.bulk
        return phase.Zmc()

    def rho_mass(self, phase=None):
        r"""Method to calculate and return mass density of the phase.

        .. math::
            \rho = \frac{MW}{1000\cdot VM}

        Returns
        -------
        rho_mass : float
            Mass density, [kg/m^3]
        """
        if phase is None:
            phase = self.bulk

        V = phase.V()
        MW = phase.MW()
        return Vm_to_rho(V, MW)

    def H_mass(self, phase=None):
        r"""Method to calculate and return mass enthalpy of the phase.

        .. math::
            H_{mass} = \frac{1000 H_{molar}}{MW}

        Returns
        -------
        H_mass : float
            Mass enthalpy, [J/kg]
        """
        if phase is None:
            phase = self.bulk
        return phase.H()*1e3*phase.MW_inv()

    def S_mass(self, phase=None):
        r"""Method to calculate and return mass entropy of the phase.

        .. math::
            S_{mass} = \frac{1000 S_{molar}}{MW}

        Returns
        -------
        S_mass : float
            Mass enthalpy, [J/(kg*K)]
        """
        if phase is None:
            phase = self.bulk
        return phase.S()*1e3*phase.MW_inv()

    def U_mass(self, phase=None):
        r"""Method to calculate and return mass internal energy of the phase.

        .. math::
            U_{mass} = \frac{1000 U_{molar}}{MW}

        Returns
        -------
        U_mass : float
            Mass internal energy, [J/(kg)]
        """
        if phase is None:
            phase = self.bulk
        return phase.U()*1e3*phase.MW_inv()

    def A_mass(self, phase=None):
        r"""Method to calculate and return mass Helmholtz energy of the phase.

        .. math::
            A_{mass} = \frac{1000 A_{molar}}{MW}

        Returns
        -------
        A_mass : float
            Mass Helmholtz energy, [J/(kg)]
        """
        if phase is None:
            phase = self.bulk
        return phase.A()*1e3*phase.MW_inv()

    def G_mass(self, phase=None):
        r"""Method to calculate and return mass Gibbs energy of the phase.

        .. math::
            G_{mass} = \frac{1000 G_{molar}}{MW}

        Returns
        -------
        G_mass : float
            Mass Gibbs energy, [J/(kg)]
        """
        if phase is None:
            phase = self.bulk
        return phase.G()*1e3*phase.MW_inv()

    def Cp_mass(self, phase=None):
        r"""Method to calculate and return mass constant pressure heat capacity
        of the phase.

        .. math::
            Cp_{mass} = \frac{1000 Cp_{molar}}{MW}

        Returns
        -------
        Cp_mass : float
            Mass heat capacity, [J/(kg*K)]
        """
        if phase is None:
            phase = self.bulk
        return phase.Cp()*1e3*phase.MW_inv()

    def Cv_mass(self, phase=None):
        r"""Method to calculate and return mass constant volume heat capacity
        of the phase.

        .. math::
            Cv_{mass} = \frac{1000 Cv_{molar}}{MW}

        Returns
        -------
        Cv_mass : float
            Mass constant volume heat capacity, [J/(kg*K)]
        """
        if phase is None:
            phase = self.bulk
        return phase.Cv()*1e3*phase.MW_inv()

    def Cp_ideal_gas(self, phase=None):
        r"""Method to calculate and return the ideal-gas heat capacity of the
        phase.

        .. math::
            C_p^{ig} = \sum_i z_i {C_{p,i}^{ig}}

        Returns
        -------
        Cp : float
            Ideal gas heat capacity, [J/(mol*K)]
        """
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
        for i in range(self.N):
            Cp += zs[i]*Cpigs_pure[i]
        return Cp

    def H_dep(self, phase=None):
        r"""Method to calculate and return the difference between the actual
        `H` and the ideal-gas enthalpy of the phase.

        .. math::
            H^{dep} = H - H^{ig}

        Returns
        -------
        H_dep : float
            Departure enthalpy, [J/(mol)]
        """
        if phase is None:
            phase = self.bulk
        if not phase.bulk_phase_type:
            return phase.H_dep()
        return phase.H() - self.H_ideal_gas(phase)

    def S_dep(self, phase=None):
        r"""Method to calculate and return the difference between the actual
        `S` and the ideal-gas entropy of the phase.

        .. math::
            S^{dep} = S - S^{ig}

        Returns
        -------
        S_dep : float
            Departure entropy, [J/(mol*K)]
        """
        if phase is None:
            phase = self.bulk
        if not phase.bulk_phase_type:
            return phase.S_dep()
        S_dep = 0.0
        for p, beta in zip(phase.phases, phase.phase_fractions):
            S_dep += p.S_dep()*beta
        return S_dep

    def Cp_dep(self, phase=None):
        r"""Method to calculate and return the difference between the actual
        `Cp` and the ideal-gas heat
        capacity :math:`C_p^{ig}` of the phase.

        .. math::
            C_p^{dep} = C_p - C_p^{ig}

        Returns
        -------
        Cp_dep : float
            Departure ideal gas heat capacity, [J/(mol*K)]
        """
        if phase is None:
            phase = self.bulk
        if not phase.bulk_phase_type:
            return phase.Cp_dep()
        return phase.Cp() - phase.Cp_ideal_gas()

    def Cv_dep(self, phase=None):
        r"""Method to calculate and return the difference between the actual
        `Cv` and the ideal-gas constant volume heat
        capacity :math:`C_v^{ig}` of the phase.

        .. math::
            C_v^{dep} = C_v - C_v^{ig}

        Returns
        -------
        Cv_dep : float
            Departure ideal gas constant volume heat capacity, [J/(mol*K)]
        """
        if phase is None:
            phase = self.bulk
        if not phase.bulk_phase_type:
            return phase.Cv_dep()
        return phase.Cv() - phase.Cv_ideal_gas()


    def H_ideal_gas(self, phase=None):
        r"""Method to calculate and return the ideal-gas enthalpy of the phase.

        .. math::
            H^{ig} = \sum_i z_i {H_{i}^{ig}}

        Returns
        -------
        H : float
            Ideal gas enthalpy, [J/(mol)]
        """
        if phase is None:
            phase = self.bulk

        # Return the phase implementation of ideal gas
        if not phase.bulk_phase_type:
            return phase.H_ideal_gas()

        HeatCapacityGases = self.correlations.HeatCapacityGases
        T, T_REF_IG = self.T, self.T_REF_IG
        Cpig_integrals_pure = [obj.T_dependent_property_integral(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        H = 0.0
        for zi, Cp_int in zip(phase.zs, Cpig_integrals_pure):
            H += zi*Cp_int
        return H

    def S_ideal_gas(self, phase=None):
        r"""Method to calculate and return the ideal-gas entropy of the phase.

        .. math::
            S^{ig} = \sum_i z_i S_{i}^{ig} - R\ln\left(\frac{P}{P_{ref}}\right)
            - R\sum_i z_i \ln(z_i)

        Returns
        -------
        S : float
            Ideal gas molar entropy, [J/(mol*K)]
        """
        if phase is None:
            phase = self.bulk
        if not phase.bulk_phase_type:
            return phase.S_ideal_gas()

        HeatCapacityGases = self.correlations.HeatCapacityGases
        T, T_REF_IG = self.T, self.T_REF_IG

        Cpig_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                      for obj in HeatCapacityGases]

        log_zs = self.log_zs()
        T, P, zs, cmps = self.T, self.P, phase.zs, range(self.N)
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)

        for i in cmps:
            S += zs[i]*Cpig_integrals_over_T_pure[i]

        return S

    def H_formation_ideal_gas(self, phase=None):
        r"""Method to calculate and return the ideal-gas enthalpy of formation
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
        """
        if phase is None:
            phase = self.bulk
        if not phase.bulk_phase_type:
            return phase.H_formation_ideal_gas()

        Hf = 0.0
        zs = phase.zs
        Hfgs = self.constants.Hfgs
        for i in range(self.N):
            Hf += zs[i]*Hfgs[i]
        return Hf

    def S_formation_ideal_gas(self, phase=None):
        r"""Method to calculate and return the ideal-gas entropy of formation
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
        """
        if phase is None:
            phase = self.bulk
        if not phase.bulk_phase_type:
            return phase.S_formation_ideal_gas()

        Sf = 0.0
        zs = phase.zs
        Sfgs = self.constants.Sfgs
        for i in range(self.N):
            Sf += zs[i]*Sfgs[i]
        return Sf

    def G_formation_ideal_gas(self, phase=None):
        r"""Method to calculate and return the ideal-gas Gibbs free energy of
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
        """
        Gf = self.H_formation_ideal_gas(phase) - self.T_REF_IG*self.S_formation_ideal_gas(phase)
        return Gf

    def U_formation_ideal_gas(self, phase=None):
        r"""Method to calculate and return the ideal-gas internal energy of
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
        """
        Uf = self.H_formation_ideal_gas(phase) - self.P_REF_IG*self.V_ideal_gas()
        return Uf

    def A_formation_ideal_gas(self, phase=None):
        r"""Method to calculate and return the ideal-gas Helmholtz energy of
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
        """
        Af = self.U_formation_ideal_gas(phase) - self.T_REF_IG*self.S_formation_ideal_gas(phase)
        return Af




    def nu(self, phase=None):
        r"""Method to calculate and return the kinematic viscosity of the
        equilibrium state.

        .. math::
            \nu = \frac{\mu}{\rho}

        Returns
        -------
        nu : float
            Kinematic viscosity, [m^2/s]

        Notes
        -----
        """
        return self.mu(phase)/self.rho_mass(phase)

    @property
    def lightest_liquid(self):
        r"""The liquid-like phase with the lowest mass density, [-]

        Returns
        -------
        lightest_liquid : Phase or None
            Phase with the lowest mass density or None if there are no liquid
            like phases, [-]

        Notes
        -----
        """
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
        r"""The liquid-like phase with the highest mass density, [-]

        Returns
        -------
        heaviest_liquid : Phase or None
            Phase with the highest mass density or None if there are no liquid
            like phases, [-]

        Notes
        -----
        """
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
        r"""The liquid-like phase with the highest mole fraction of water, [-]

        Returns
        -------
        water_phase_index : int
            Index into the attribute :obj:`EquilibriumState.liquids` which
            refers to the liquid-like phase with the highest water mole
            fraction, [-]

        Notes
        -----
        """
        try:
            return self._water_phase_index
        except AttributeError:
            pass
        water_index = self.constants.water_index

        max_zw, max_phase, max_phase_idx = 0.0, None, None
        for i, l in enumerate(self.liquids):
            z_w = l.zs[water_index]
            if z_w > max_zw:
                max_phase, max_zw, max_phase_idx = l, z_w, i

        self._water_phase_index = max_phase_idx
        return max_phase_idx

    @property
    def water_phase(self):
        r"""The liquid-like phase with the highest water mole fraction, [-]

        Returns
        -------
        water_phase : Phase or None
            Phase with the highest water mole fraction or None if there are no
            liquid like phases with water, [-]

        Notes
        -----
        """
        try:
            return self.liquids[self.water_phase_index]
        except:
            return None

    def phis(self, phase=None):
        if phase is not None:
            return phase.phis()
        if self.phase_count == 1:
            return self.phases[0].phis()
        raise ValueError("This property is not defined for EquilibriumStates with more than one phase")

    def Ks(self, phase, ref_phase=None):
        r"""Method to calculate and return the K-values of each phase.
        These are NOT just liquid-vapor K values; these are thermodynamic K
        values. The reference phase can be specified with `ref_phase`, and then
        the K-values will be with respect to that phase.

        .. math::
            K_i = \frac{z_{i, \text{phase}}}{z_{i, \text{ref phase}}}

        If no reference phase is provided, the following criteria is used to
        select one:

            * If the flash algorithm provided a reference phase, use that
            * Otherwise use the liquid0 phase if one is present
            * Otherwise use the solid0 phase if one is present
            * Otherwise use the gas phase if one is present

        Returns
        -------
        Ks : list[float]
            Equilibrium K values, [-]

        Notes
        -----
        """
        if ref_phase is None:
            try:
                ref_phase = self.flash_convergence["ref_phase"]
            except:
                if self.liquid_count:
                    ref_phase = self.liquid0
                elif self.solid_count:
                    ref_phase = self.solid0
                else:
                    ref_phase = self.gas
        ref_zs = ref_phase.zs
        zs = phase.zs
        if self.flasher.vectorized:
            Ks = zs/ref_zs
        else:
            Ks = [g/l for l, g in zip(ref_zs, zs)]
        return Ks

    def value(self, name, phase=None):
        r"""Method to retrieve a property from a string. This more or less
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
        """
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
        """Alias of CASs."""
        return self.constants.CASs

    def V_iter(self, phase=None, force=False):
        if phase is None:
            phase = self.bulk
        return phase.V_iter(force=force)

    try:
        V_iter.__doc__ = Phase.V_iter.__doc__
    except:
        pass


_add_attrs_doc = []
for s in dir(EquilibriumState):
    obj = getattr(EquilibriumState, s)
    if type(obj) is property:
        _add_attrs_doc.append(s)

# Add some fancy things for easier access to properties

def _make_getter_constants(name):
    def get_constant(self):
        return getattr(self.constants, name)
    return get_constant

def _make_getter_correlations(name):
    def get_correlation(self):
        return getattr(self.correlations, name)

    text = f"""Wrapper to obtain the list of {name} objects of the associated
:obj:`PropertyCorrelationsPackage <thermo.chemical_package.PropertyCorrelationsPackage>`."""
    try:
        get_correlation.__doc__ = text
    except:
        pass
    return get_correlation

def _make_getter_EquilibriumState(name):
    def get_EquilibriumState(self):
        return getattr(self.result, name)(self)
    try:
        get_EquilibriumState.__doc__ = getattr(EquilibriumState, name).__doc__
    except:
        pass
    return get_EquilibriumState

def _make_getter_argumentless_EquilibriumState(name):
    def get_EquilibriumState_argumentless(self):
        return getattr(self.result, name)()
    try:
        get_EquilibriumState_argumentless.__doc__ = getattr(EquilibriumState, name).__doc__
    except:
        pass
    return get_EquilibriumState_argumentless


def _make_getter_bulk_props(name):
    def get_bulk_prop(self):
        return getattr(self.bulk, name)()
    try:
        doc = getattr(Bulk, name).__doc__
        if doc is None:
            doc = getattr(Phase, name).__doc__
        get_bulk_prop.__doc__ = doc
    except:
        pass
    return get_bulk_prop

def _make_getter_bulk_property(name):
    @property
    def get_bulk_property(self):
        return getattr(self.bulk, name)
    try:
        doc = getattr(Bulk, name).__doc__
        if doc is None:
            doc = getattr(Phase, name).__doc__
        get_bulk_property.__doc__ = doc
    except:
        pass
    return get_bulk_property

### For the pure component fixed properties, allow them to be retrived from the phase
# and bulk object as well as the Equilibrium State Object
constant_blacklist = {"atom_fractions"}

for name in ChemicalConstantsPackage.properties:
    if name not in constant_blacklist:
        _add_attrs_doc.append(name)
        getter = property(_make_getter_constants(name))
        try:
            var_type, desc, units, return_desc = constants_docstrings[name]
            type_name = var_type if type(var_type) is str else var_type.__name__
            if return_desc is None:
                return_desc = desc
            full_desc = f"""{desc}, {units}.

Returns
-------
{name} : {type_name}
    {return_desc}, {units}."""
#            print(full_desc)
            getter.__doc__ = full_desc
        except:
            pass
        setattr(EquilibriumState, name, getter)
        setattr(Phase, name, getter)

### For the temperature-dependent correlations, allow them to be retrieved by their
# name from the EquilibriumState ONLY
for name in PropertyCorrelationsPackage.correlations:
    getter = property(_make_getter_correlations(name))
    setattr(EquilibriumState, name, getter)
    _add_attrs_doc.append(name)


for method in phase_shared_methods:
    setattr(EquilibriumState, method.__name__, method)

### For certain properties not supported by Bulk, allow them to call up to the
# EquilibriumState to get the property
Bulk_properties_to_EquilibriumState = [#'H_ideal_gas', 'Cp_ideal_gas','S_ideal_gas',
       "H_formation_ideal_gas", "S_formation_ideal_gas",
       "G_formation_ideal_gas", "U_formation_ideal_gas", "A_formation_ideal_gas",
       "H_dep", "S_dep", "Cp_dep", "Cv_dep"]
for name in Bulk_properties_to_EquilibriumState:
    method = _make_getter_EquilibriumState(name)
    setattr(Bulk, name, method)

### For certain properties of the Bulk phase, make EquilibriumState get it from the Bulk
bulk_props = ["V", "Z", "rho", "Cp", "Cv", "H", "S", "U", "G", "A", #'dH_dT', 'dH_dP', 'dS_dT', 'dS_dP',
              #'dU_dT', 'dU_dP', 'dG_dT', 'dG_dP', 'dA_dT', 'dA_dP',
              "H_reactive", "S_reactive", "G_reactive", "U_reactive", "A_reactive",
              "H_reactive_mass", "S_reactive_mass", "G_reactive_mass", "U_reactive_mass", "A_reactive_mass",
              "H_ideal_gas_mass", "S_ideal_gas_mass", "G_ideal_gas_mass", "U_ideal_gas_mass", "A_ideal_gas_mass",
              "H_formation_ideal_gas_mass", "S_formation_ideal_gas_mass", "G_formation_ideal_gas_mass",
              "U_formation_ideal_gas_mass", "A_formation_ideal_gas_mass",
              "H_dep_mass", "S_dep_mass", "G_dep_mass", "U_dep_mass", "A_dep_mass",
              "Cp_Cv_ratio", "log_zs", "isothermal_bulk_modulus",
              "dP_dT_frozen", "dP_dV_frozen", "d2P_dT2_frozen", "d2P_dV2_frozen",
              "d2P_dTdV_frozen",
              "d2P_dTdV", "d2P_dV2", "d2P_dT2", "dP_dV", "dP_dT", "isentropic_exponent",
              "alpha", "thermal_diffusivity",
              "PIP", "kappa", "isobaric_expansion", "Joule_Thomson", "speed_of_sound",
              "speed_of_sound_mass", "speed_of_sound_ideal_gas", "speed_of_sound_ideal_gas_mass",
              "U_dep", "G_dep", "A_dep", "V_dep", "B_from_Z",
              "Cp_dep_mass", "Cp_ideal_gas_mass", "Cv_dep_mass", "G_min_criteria",
              "mu", "k", "sigma", "Prandtl",
              "isentropic_exponent", "isentropic_exponent_PV", "isentropic_exponent_TV",
              "isentropic_exponent_PT",

              "concentrations_mass", "concentrations", "Qls", "ms", "ns", "Q", "m", "n",
              "nu", "kinematic_viscosity", "partial_pressures",
              "H_ideal_gas_standard_state", "Hs_ideal_gas_standard_state", "G_ideal_gas_standard_state",
               "Gs_ideal_gas_standard_state", "S_ideal_gas_standard_state", "Ss_ideal_gas_standard_state",

                "concentrations_mass_gas", "concentrations_mass_gas_normal", "concentrations_mass_gas_standard",
                "concentrations_gas_standard", "concentrations_gas_normal", "concentrations_gas"
 ]



bulk_props += derivatives_thermodynamic
bulk_props += derivatives_thermodynamic_mass
bulk_props += derivatives_jacobian

for name in bulk_props:
    # Maybe take this out and implement it manually for performance?
    getter = _make_getter_bulk_props(name)
    setattr(EquilibriumState, name, getter)

# properties
bulk_properties = ["Ql", "Ql_calc", "Qls_calc", "Qls", "Qg_calc", "Qg", "Qgs_calc", "Qgs", "ms_calc",
                     "ns_calc",  "Q_calc", "Q", "m_calc",  "n_calc",
                     "H_calc",
                    #'n','m','ns','ms',
                    "T_calc", "P_calc", "VF_calc", "zs_calc", "ws_calc",  "Vfls_calc", "Vfgs_calc",
                    "energy_reactive_calc", "energy_reactive", "energy_calc", "energy"]
for name in bulk_properties:
    # Maybe take this out and implement it manually for performance?
    getter = _make_getter_bulk_property(name)
    setattr(EquilibriumState, name, getter)

try:
    EquilibriumState.__doc__ = EquilibriumState.__doc__ +"\n    " + "\n    ".join(_add_attrs_doc)
except:
    pass

def make_getter_one_phase_property(prop_name):
    def property_one_phase_only(self, phase=None):
        if phase is not None:
            return getattr(phase, prop_name)()
        if self.phase_count == 1:
            return getattr(self.phases[0], prop_name)()
        raise ValueError("This property is not defined for EquilibriumStates with more than one phase")
    return property_one_phase_only

one_phase_properties = ["phis", "lnphis", "fugacities", "fugacities", "dlnphis_dT", "dphis_dT", "dfugacities_dT",
                         "dlnphis_dP", "dphis_dP", "dfugacities_dP", "dphis_dzs", "dlnphis_dns", "activities"]
for prop in one_phase_properties:
    getter = make_getter_one_phase_property(prop)
    setattr(EquilibriumState, prop, getter)


def _make_getter_atom_fraction(element_symbol):
    def get_atom_fraction(self):
        try:
            try:
                return self._atom_fractions[element_symbol]
            except KeyError:
                return 0.0
            except AttributeError:
                return self.atom_fractions()[element_symbol]
        except KeyError:
            return 0.0
    return get_atom_fraction

for ele in periodic_table:
    getter = _make_getter_atom_fraction(ele.symbol)
    name = f"{ele.name}_atom_fraction"

    _add_attrs_doc =  rf"""Method to calculate and return the mole fraction that
            is {ele.name} element, [-]
            """
    getter.__doc__ = _add_attrs_doc
    setattr(EquilibriumState, name, getter)
    setattr(Phase, name, getter)

def _make_getter_atom_mass_fraction(element_symbol):
    def get_atom_mass_fraction(self):
        try:
            try:
                return self._atom_mass_fractions[element_symbol]
            except KeyError:
                return 0.0
            except AttributeError:
                return self.atom_mass_fractions()[element_symbol]
        except KeyError:
            return 0.0

    return get_atom_mass_fraction
for ele in periodic_table:
    getter = _make_getter_atom_mass_fraction(ele.symbol)
    name = f"{ele.name}_atom_mass_fraction"

    _add_attrs_doc =  rf"""Method to calculate and return the mass fraction of the phase
            that is {ele.name} element, [-]
            """
    getter.__doc__ = _add_attrs_doc
    setattr(EquilibriumState, name, getter)
    setattr(Phase, name, getter)

def _make_getter_atom_mass_flow(element_symbol):
    def get_atom_mass_flow(self):
        try:
            try:
                return self._atom_mass_fractions[element_symbol]*self.m
            except KeyError:
                return 0.0
            except AttributeError:
                return self.atom_mass_fractions()[element_symbol]*self.m
        except KeyError:
            return 0.0
    return get_atom_mass_flow

for ele in periodic_table:
    getter = _make_getter_atom_mass_flow(ele.symbol)
    name = f"{ele.name}_atom_mass_flow"

    _add_attrs_doc =  rf"""Method to calculate and return the mass flow of atoms
            that are {ele.name} element, [kg/s]
            """
    getter.__doc__ = _add_attrs_doc
    setattr(EquilibriumState, name, getter)
    setattr(Phase, name, getter)

def _make_getter_atom_flow(element_symbol):
    def get_atom_flow(self):
        try:
            try:
                return self._atom_content[element_symbol]*self.n
            except KeyError:
                return 0.0
            except AttributeError:
                return self.atom_content()[element_symbol]*self.n
        except KeyError:
            return 0.0
    return get_atom_flow

for ele in periodic_table:
    getter = _make_getter_atom_flow(ele.symbol)
    name = f"{ele.name}_atom_flow"

    _add_attrs_doc =  rf"""Method to calculate and return the mole flow that is
            {ele.name}, [mol/s]
            """
    getter.__doc__ = _add_attrs_doc
    setattr(EquilibriumState, name, getter)
    setattr(Phase, name, getter)

def _make_getter_atom_count_flow(element_symbol):
    def get_atom_count_flow(self):
        try:
            try:
                return self._atom_content[element_symbol]*self.n*N_A
            except KeyError:
                return 0.0
            except AttributeError:
                return self.atom_content()[element_symbol]*self.n*N_A
        except KeyError:
            return 0.0
    return get_atom_count_flow

for ele in periodic_table:
    getter = _make_getter_atom_count_flow(ele.symbol)
    name = f"{ele.name}_atom_count_flow"

    _add_attrs_doc =  rf"""Method to calculate and return the number of atoms in the
            flow which are {ele.name}, [atoms/s]
            """
    getter.__doc__ = _add_attrs_doc
    setattr(EquilibriumState, name, getter)
    setattr(Phase, name, getter)


_comonent_specific_properties = {"water": CAS_H2O,
                                 "carbon_dioxide": "124-38-9",
                                 "hydrogen_sulfide": "7783-06-4",
                                 "hydrogen": "1333-74-0",
                                 "helium": "7440-59-7",
                                 "nitrogen": "7727-37-9",
                                 "oxygen": "7782-44-7",
                                 "argon": "7440-37-1",
                                 "methane": "74-82-8",
                                 "ammonia": "7664-41-7",
                                 }
def _make_getter_partial_pressure(CAS):
    def get(self):
        try:
            idx = self.CASs.index(CAS)
        except ValueError:
            # Not present
            return 0.0
        return self.P*self.zs[idx]
    return get
for _name, _CAS in _comonent_specific_properties.items():
    getter = _make_getter_partial_pressure(_CAS)
    name = f"{_name}_partial_pressure"

    _add_attrs_doc =  rf"""Method to calculate and return the ideal partial pressure of {_name}, [Pa]
            """
    getter.__doc__ = _add_attrs_doc
    setattr(EquilibriumState, name, getter)
    setattr(Phase, name, getter)


def _make_getter_component_molar_weight(CAS):
    def get(self):
        try:
            idx = self.CASs.index(CAS)
        except ValueError:
            # Not present
            return 0.0
        return self.MW()*self.ws()[idx]
    return get


for _name, _CAS in _comonent_specific_properties.items():
    getter = _make_getter_component_molar_weight(_CAS)
    name = f"{_name}_molar_weight"

    _add_attrs_doc =  rf"""Method to calculate and return the effective quantiy
    of {_name} in the phase as a molar weight, [g/mol].

    This is the molecular weight of the phase times the mass fraction of the
    {_name} component.
            """
    getter.__doc__ = _add_attrs_doc
    setattr(EquilibriumState, name, getter)
    setattr(Phase, name, getter)
del _add_attrs_doc


object_lookups[EquilibriumState.__full_path__] = EquilibriumState
object_lookups[ChemicalConstantsPackage.__full_path__] = ChemicalConstantsPackage
object_lookups[PropertyCorrelationsPackage.__full_path__] = PropertyCorrelationsPackage

from thermo.chemical_package import mix_properties_to_classes, properties_to_classes  # noqa: E402

for o in mix_properties_to_classes.values():
    object_lookups[o.__full_path__] = o
for o in properties_to_classes.values():
    object_lookups[o.__full_path__] = o
