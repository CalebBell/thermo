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

This module contains classes for storing data and objects which are necessary
for doing thermodynamic calculations. The intention for these classes is to
serve as an in-memory storage layer between the disk and methods which do
full thermodynamic calculations.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Chemical Constants Class
========================
.. autoclass:: ChemicalConstantsPackage
    :members: subset, properties
    :undoc-members:
    :show-inheritance:
    :exclude-members:

Sample Class: Water
-------------------
.. autodata:: iapws_constants
.. autofunction:: iapws_correlations

Property Correlations Class
===========================

.. autoclass:: PropertyCorrelationPackage
    :members: subset, as_best_fit
    :undoc-members:
    :show-inheritance:
    :exclude-members:

'''

from __future__ import division

__all__ = ['ChemicalConstantsPackage', 'PropertyCorrelationPackage',
           'iapws_constants', 'iapws_correlations']

from thermo.chemical import Chemical, get_chemical_constants
from chemicals.identifiers import *
from thermo.thermal_conductivity import ThermalConductivityLiquid, ThermalConductivityGas, ThermalConductivityLiquidMixture, ThermalConductivityGasMixture
from thermo.volume import VolumeLiquidMixture, VolumeGasMixture, VolumeSolidMixture, VolumeLiquid, VolumeGas, VolumeSolid
from thermo.permittivity import *
from thermo.heat_capacity import HeatCapacitySolid, HeatCapacityGas, HeatCapacityLiquid, HeatCapacitySolidMixture, HeatCapacityGasMixture, HeatCapacityLiquidMixture
from thermo.interface import SurfaceTension, SurfaceTensionMixture
from thermo.viscosity import ViscosityLiquid, ViscosityGas, ViscosityLiquidMixture, ViscosityGasMixture
from chemicals.utils import property_molar_to_mass
from thermo.utils import *
from thermo.vapor_pressure import VaporPressure, SublimationPressure
from thermo.phase_change import EnthalpyVaporization, EnthalpySublimation


CAS_H2O = '7732-18-5'


class ChemicalConstantsPackage(object):
    non_vector_properties = ('atomss', 'Carcinogens', 'CASs', 'Ceilings', 'charges',
                 'conductivities', 'dipoles', 'economic_statuses', 'formulas', 'Gfgs',
                 'Gfgs_mass', 'GWPs', 'Hcs', 'Hcs_lower', 'Hcs_lower_mass', 'Hcs_mass',
                 'Hfgs', 'Hfgs_mass', 'Hfus_Tms', 'Hfus_Tms_mass', 'Hsub_Tms',
                 'Hsub_Tms_mass', 'Hvap_298s', 'Hvap_298s_mass', 'Hvap_Tbs', 'Hvap_Tbs_mass',
                 'InChI_Keys', 'InChIs', 'legal_statuses', 'LFLs', 'logPs',
                 'molecular_diameters', 'MWs', 'names', 'ODPs', 'omegas',
                 'Parachors', 'Pcs', 'phase_STPs', 'Psat_298s', 'PSRK_groups',
                 'Pts', 'PubChems', 'rhocs', 'rhocs_mass', 'rhol_STPs',
                 'rhol_STPs_mass', 'RIs', 'S0gs', 'S0gs_mass', 'Sfgs',
                 'Sfgs_mass', 'similarity_variables', 'Skins', 'smiless',
                 'STELs', 'StielPolars', 'Stockmayers', 'Tautoignitions',
                 'Tbs', 'Tcs', 'Tflashs', 'Tms', 'Tts', 'TWAs', 'UFLs',
                 'UNIFAC_Dortmund_groups', 'UNIFAC_groups',
                 'Van_der_Waals_areas', 'Van_der_Waals_volumes', 'Vcs',
                 'Vml_STPs', 'Vml_Tms', 'Zcs', 'UNIFAC_Rs', 'UNIFAC_Qs',
                 'rhos_Tms', 'Vms_Tms', 'solubility_parameters',
                 'Vml_60Fs', 'rhol_60Fs', 'rhol_60Fs_mass',
                 'conductivity_Ts', 'RI_Ts',
                 )
    properties = ('atom_fractions',) + non_vector_properties
    '''Tuple of all properties that can be held by this object.'''

    __slots__ = properties + ('N', 'cmps', 'water_index', 'n_atoms')
    non_vectors = ('atom_fractions',)
    non_vectors_set = set(non_vectors)
    def subset(self, idxs=None, properties=None):
        r'''Method to construct a new ChemicalConstantsPackage that removes
        all components not specified in the `idxs` argument. Although this
        class has a great many attributes, it is often sufficient to work with
        a subset of those properties; and if a list of properties is provided,
        only those properties will be added to the new object as well.

        Parameters
        ----------
        idxs : list[int] or Slice or None
            Indexes of components that should be included; if None, all
            components will be included , [-]
        properties : tuple[str] or None
            List of properties to be included; all properties will be included
            if this is not specified

        Returns
        -------
        subset_consts : ChemicalConstantsPackage
            Object with reduced properties and or components, [-]

        Notes
        -----
        It is not intended for properties to be edited in this object!
        One optimization is that all entirely empty properties use the same
        list-of-Nones.

        All properties should have been specified before constructing the first
        ChemicalConstantsPackage.

        Examples
        --------
        >>> base = ChemicalConstantsPackage(MWs=[18.01528, 106.165, 106.165, 106.165], names=['water', 'o-xylene', 'p-xylene', 'm-xylene'], omegas=[0.344, 0.3118, 0.324, 0.331], Pcs=[22048320.0, 3732000.0, 3511000.0, 3541000.0], Tcs=[647.14, 630.3, 616.2, 617.0])
        >>> base.subset([0])
        ChemicalConstantsPackage(MWs=[18.01528], names=['water'], omegas=[0.344], Pcs=[22048320.0], Tcs=[647.14])
        >>> base.subset(slice(1,4))
        ChemicalConstantsPackage(MWs=[106.16499999999999, 106.16499999999999, 106.16499999999999], names=['o-xylene', 'p-xylene', 'm-xylene'], omegas=[0.3118, 0.324, 0.331], Pcs=[3732000.0, 3511000.0, 3541000.0], Tcs=[630.3, 616.2, 617.0])
        >>> base.subset(idxs=[0, 3], properties=('names', 'MWs'))
        ChemicalConstantsPackage(MWs=[18.01528, 106.16499999999999], names=['water', 'm-xylene'])
        '''
        if idxs is None:
            idxs = self.cmps
        if properties is None:
            properties = self.non_vector_properties
        is_slice = isinstance(idxs, slice)
        if not is_slice:
            is_one = len(idxs) == 1
            idx = idxs[0]

        def atindexes(values):
            if is_slice:
                return values[idxs]
            if is_one:
                return [values[idx]]
            return [values[i] for i in idxs]

        new = {}
        for p in properties:
            v = getattr(self, p)
            if v is not None:
                new[p] = atindexes(v)
        return ChemicalConstantsPackage(**new)

    def __repr__(self):
        return self._make_str()

    def _make_str(self, delim=', ', properties=None):
        '''Method to create a new string representing the
        object.
        '''
        if properties is None:
            properties = self.properties


        s = 'ChemicalConstantsPackage('
        for k in properties:
            if any(i is not None for i in getattr(self, k)):
                s += '%s=%s%s'%(k, getattr(self, k), delim)
        s = s[:-2] + ')'
        return s

    def __init__(self, CASs=None, names=None, MWs=None, Tms=None, Tbs=None,
                 # Critical state points
                 Tcs=None, Pcs=None, Vcs=None, omegas=None,
                 Zcs=None, rhocs=None, rhocs_mass=None,
                 # Phase change enthalpy
                 Hfus_Tms=None, Hfus_Tms_mass=None, Hvap_Tbs=None,
                 Hvap_Tbs_mass=None,
                 # Standard values
                 Vml_STPs=None, rhol_STPs=None, rhol_STPs_mass=None,
                 Vml_60Fs=None, rhol_60Fs=None, rhol_60Fs_mass=None,
                 # Reaction (ideal gas)
                 Hfgs=None, Hfgs_mass=None, Gfgs=None, Gfgs_mass=None,
                 Sfgs=None, Sfgs_mass=None, S0gs=None, S0gs_mass=None,

                 # Triple point
                 Tts=None, Pts=None, Hsub_Tms=None, Hsub_Tms_mass=None,
                 # Combustion
                 Hcs=None, Hcs_mass=None, Hcs_lower=None, Hcs_lower_mass=None,
                 # Fire safety
                 Tflashs=None, Tautoignitions=None, LFLs=None, UFLs=None,
                 # Other safety
                 TWAs=None, STELs=None, Ceilings=None, Skins=None,
                 Carcinogens=None, legal_statuses=None, economic_statuses=None,
                 # Environmental
                 GWPs=None, ODPs=None, logPs=None,
                 Psat_298s=None, Hvap_298s=None, Hvap_298s_mass=None,
                 Vml_Tms=None, rhos_Tms=None, Vms_Tms=None,

                 # Analytical
                 RIs=None, RI_Ts=None, conductivities=None,
                 conductivity_Ts=None,
                 # Odd constants
                 charges=None, dipoles=None, Stockmayers=None,
                 molecular_diameters=None, Van_der_Waals_volumes=None,
                 Van_der_Waals_areas=None, Parachors=None, StielPolars=None,
                 atomss=None, atom_fractions=None,
                 similarity_variables=None, phase_STPs=None,
                 solubility_parameters=None,
                 # Other identifiers
                 PubChems=None, formulas=None, smiless=None, InChIs=None,
                 InChI_Keys=None,
                 # Groups
                 UNIFAC_groups=None, UNIFAC_Dortmund_groups=None,
                 PSRK_groups=None, UNIFAC_Rs=None, UNIFAC_Qs=None,
                 ):
        self.N = N = len(MWs)
        self.cmps = range(N)

        empty_list = [None]*N

        if atom_fractions is None: atom_fractions = empty_list
        if atomss is None: atomss = empty_list
        if Carcinogens is None: Carcinogens = empty_list
        if CASs is None: CASs = empty_list
        if Ceilings is None: Ceilings = empty_list
        if charges is None: charges = empty_list
        if conductivities is None: conductivities = empty_list
        if dipoles is None: dipoles = empty_list
        if economic_statuses is None: economic_statuses = empty_list
        if formulas is None: formulas = empty_list
        if Gfgs is None: Gfgs = empty_list
        if Gfgs_mass is None: Gfgs_mass = empty_list
        if GWPs is None: GWPs = empty_list
        if Hcs is None: Hcs = empty_list
        if Hcs_lower is None: Hcs_lower = empty_list
        if Hcs_lower_mass is None: Hcs_lower_mass = empty_list
        if Hcs_mass is None: Hcs_mass = empty_list
        if Hfgs is None: Hfgs = empty_list
        if Hfgs_mass is None: Hfgs_mass = empty_list
        if Hfus_Tms is None: Hfus_Tms = empty_list
        if Hfus_Tms_mass is None: Hfus_Tms_mass = empty_list
        if Hsub_Tms is None: Hsub_Tms = empty_list
        if Hsub_Tms_mass is None: Hsub_Tms_mass = empty_list
        if Hvap_298s is None: Hvap_298s = empty_list
        if Hvap_298s_mass is None: Hvap_298s_mass = empty_list
        if Hvap_Tbs is None: Hvap_Tbs = empty_list
        if Hvap_Tbs_mass is None: Hvap_Tbs_mass = empty_list
        if InChI_Keys is None: InChI_Keys = empty_list
        if InChIs is None: InChIs = empty_list
        if legal_statuses is None: legal_statuses = empty_list
        if LFLs is None: LFLs = empty_list
        if logPs is None: logPs = empty_list
        if molecular_diameters is None: molecular_diameters = empty_list
        if names is None: names = empty_list
        if ODPs is None: ODPs = empty_list
        if omegas is None: omegas = empty_list
        if Parachors is None: Parachors = empty_list
        if Pcs is None: Pcs = empty_list
        if phase_STPs is None: phase_STPs = empty_list
        if Psat_298s is None: Psat_298s = empty_list
        if PSRK_groups is None: PSRK_groups = empty_list
        if Pts is None: Pts = empty_list
        if PubChems is None: PubChems = empty_list
        if rhocs is None: rhocs = empty_list
        if rhocs_mass is None: rhocs_mass = empty_list
        if rhol_STPs is None: rhol_STPs = empty_list
        if rhol_STPs_mass is None: rhol_STPs_mass = empty_list
        if RIs is None: RIs = empty_list
        if S0gs is None: S0gs = empty_list
        if S0gs_mass is None: S0gs_mass = empty_list
        if Sfgs is None: Sfgs = empty_list
        if Sfgs_mass is None: Sfgs_mass = empty_list
        if similarity_variables is None: similarity_variables = empty_list
        if Skins is None: Skins = empty_list
        if smiless is None: smiless = empty_list
        if STELs is None: STELs = empty_list
        if StielPolars is None: StielPolars = empty_list
        if Stockmayers is None: Stockmayers = empty_list
        if solubility_parameters is None: solubility_parameters = empty_list
        if Tautoignitions is None: Tautoignitions = empty_list
        if Tbs is None: Tbs = empty_list
        if Tcs is None: Tcs = empty_list
        if Tflashs is None: Tflashs = empty_list
        if Tms is None: Tms = empty_list
        if Tts is None: Tts = empty_list
        if TWAs is None: TWAs = empty_list
        if UFLs is None: UFLs = empty_list
        if UNIFAC_Dortmund_groups is None: UNIFAC_Dortmund_groups = empty_list
        if UNIFAC_groups is None: UNIFAC_groups = empty_list
        if UNIFAC_Rs is None: UNIFAC_Rs = empty_list
        if UNIFAC_Qs is None: UNIFAC_Qs = empty_list
        if Van_der_Waals_areas is None: Van_der_Waals_areas = empty_list
        if Van_der_Waals_volumes is None: Van_der_Waals_volumes = empty_list
        if Vcs is None: Vcs = empty_list
        if Vml_STPs is None: Vml_STPs = empty_list
        if Vml_Tms is None: Vml_Tms = empty_list
        if rhos_Tms is None: rhos_Tms = empty_list
        if Vms_Tms is None: Vms_Tms = empty_list
        if Zcs is None: Zcs = empty_list
        if Vml_60Fs is None: Vml_60Fs = empty_list
        if rhol_60Fs is None: rhol_60Fs = empty_list
        if rhol_60Fs_mass is None: rhol_60Fs_mass = empty_list
        if RI_Ts is None: RI_Ts = empty_list
        if conductivity_Ts is None: conductivity_Ts = empty_list

        self.atom_fractions = atom_fractions
        self.atomss = atomss
        self.Carcinogens = Carcinogens
        self.CASs = CASs
        self.Ceilings = Ceilings
        self.charges = charges
        self.conductivities = conductivities
        self.dipoles = dipoles
        self.economic_statuses = economic_statuses
        self.formulas = formulas
        self.Gfgs = Gfgs
        self.Gfgs_mass = Gfgs_mass
        self.GWPs = GWPs
        self.Hcs = Hcs
        self.Hcs_lower = Hcs_lower
        self.Hcs_lower_mass = Hcs_lower_mass
        self.Hcs_mass = Hcs_mass
        self.Hfgs = Hfgs
        self.Hfgs_mass = Hfgs_mass
        self.Hfus_Tms = Hfus_Tms
        self.Hfus_Tms_mass = Hfus_Tms_mass
        self.Hsub_Tms = Hsub_Tms
        self.Hsub_Tms_mass = Hsub_Tms_mass
        self.Hvap_298s = Hvap_298s
        self.Hvap_298s_mass = Hvap_298s_mass
        self.Hvap_Tbs = Hvap_Tbs
        self.Hvap_Tbs_mass = Hvap_Tbs_mass
        self.InChI_Keys = InChI_Keys
        self.InChIs = InChIs
        self.legal_statuses = legal_statuses
        self.LFLs = LFLs
        self.logPs = logPs
        self.molecular_diameters = molecular_diameters
        self.MWs = MWs
        self.names = names
        self.ODPs = ODPs
        self.omegas = omegas
        self.Parachors = Parachors
        self.Pcs = Pcs
        self.phase_STPs = phase_STPs
        self.Psat_298s = Psat_298s
        self.PSRK_groups = PSRK_groups
        self.Pts = Pts
        self.PubChems = PubChems
        self.rhocs = rhocs
        self.rhocs_mass = rhocs_mass
        self.rhol_STPs = rhol_STPs
        self.rhol_STPs_mass = rhol_STPs_mass
        self.RIs = RIs
        self.S0gs = S0gs
        self.S0gs_mass = S0gs_mass
        self.Sfgs = Sfgs
        self.Sfgs_mass = Sfgs_mass
        self.similarity_variables = similarity_variables
        self.solubility_parameters = solubility_parameters
        self.Skins = Skins
        self.smiless = smiless
        self.STELs = STELs
        self.StielPolars = StielPolars
        self.Stockmayers = Stockmayers
        self.Tautoignitions = Tautoignitions
        self.Tbs = Tbs
        self.Tcs = Tcs
        self.Tflashs = Tflashs
        self.Tms = Tms
        self.Tts = Tts
        self.TWAs = TWAs
        self.UFLs = UFLs
        self.UNIFAC_Dortmund_groups = UNIFAC_Dortmund_groups
        self.UNIFAC_groups = UNIFAC_groups
        self.UNIFAC_Rs = UNIFAC_Rs
        self.UNIFAC_Qs = UNIFAC_Qs
        self.Van_der_Waals_areas = Van_der_Waals_areas
        self.Van_der_Waals_volumes = Van_der_Waals_volumes
        self.Vcs = Vcs
        self.Vml_STPs = Vml_STPs
        self.Vml_Tms = Vml_Tms
        self.rhos_Tms = rhos_Tms
        self.Vms_Tms = Vms_Tms
        self.Zcs = Zcs
        self.Vml_60Fs = Vml_60Fs
        self.rhol_60Fs = rhol_60Fs
        self.rhol_60Fs_mass = rhol_60Fs_mass
        self.conductivity_Ts = conductivity_Ts
        self.RI_Ts = RI_Ts

        try:
            self.water_index = CASs.index(CAS_H2O)
        except ValueError:
            self.water_index = None

        try:
            self.n_atoms = [sum(i.values()) for i in atomss]
        except:
            self.n_atoms = None

constants_docstrings = {'N': (int, "Number of components in the package", "[-]", None),
'cmps': (range, "Iterator over all components", "[-]", None),
'rhol_60Fs': ("list[float]", "Liquid standard molar densities at 60 °F", "[mol/m^3]", None),
'atom_fractions': ("list[dict]", "Breakdown of each component into its elemental fractions, as a dict", "[-]", None),
'atomss': ("list[dict]", "Breakdown of each component into its elements and their counts, as a dict", "[-]", None),
'Carcinogens': ("list[dict]", "Status of each component in cancer causing registries", "[-]", None),
'CASs': ("list[str]", "CAS registration numbers for each component", "[-]", None),
'Ceilings': ("list[tuple[(float, str)]]", "Ceiling exposure limits to chemicals (and their units)", "[ppm or mg/m^3]", None),
'charges': ("list[float]", "Charge number (valence) for each component", "[-]", None),
'conductivities': ("list[float]", "Electrical conductivities for each component", "[S/m]", None),
'conductivity_Ts': ("list[float]", "Temperatures at which the electrical conductivities for each component were measured", "[K]", None),
'dipoles': ("list[float]", "Dipole moments for each component", "[debye]", None),
'economic_statuses': ("list[dict]", "Status of each component in in relation to import and export from various regions", "[-]", None),
'formulas': ("list[str]", "Formulas of each component", "[-]", None),
'Gfgs': ("list[float]", "Ideal gas standard molar Gibbs free energy of formation for each component", "[J/mol]", None),
'Gfgs_mass': ("list[float]", "Ideal gas standard Gibbs free energy of formation for each component", "[J/kg]", None),
'GWPs': ("list[float]", "Global Warming Potentials for each component", " [(impact/mass chemical)/(impact/mass CO2)]", None),
'Hcs': ("list[float]", "Higher standard molar heats of combustion for each component", "[J/mol]", None),
'Hcs_mass': ("list[float]", "Higher standard heats of combustion for each component", "[J/kg]", None),
'Hcs_lower': ("list[float]", "Lower standard molar heats of combustion for each component", "[J/mol]", None),
'Hcs_lower_mass': ("list[float]", "Lower standard heats of combustion for each component", "[J/kg]", None),
'Hfgs': ("list[float]", "Ideal gas standard molar enthalpies of formation for each component", "[J/mol]", None),
'Hfgs_mass': ("list[float]", "Ideal gas standard enthalpies of formation for each component", "[J/kg]", None),
'Hfus_Tms': ("list[float]", "Molar heats of fusion for each component at their respective melting points", "[J/mol]", None),
'Hfus_Tms_mass': ("list[float]", "Heats of fusion for each component at their respective melting points", "[J/kg]", None),
'Hsub_Tms': ("list[float]", "Heats of sublimation for each component at their respective melting points", "[J/mol]", None),
'Hsub_Tms_mass': ("list[float]", "Heats of sublimation for each component at their respective melting points", "[J/kg]", None),
'Hvap_298s': ("list[float]", "Molar heats of vaporization for each component at 298.15 K", "[J/mol]", None),
'Hvap_298s_mass': ("list[float]", "Heats of vaporization for each component at 298.15 K", "[J/kg]", None),
'Hvap_Tbs': ("list[float]", "Molar heats of vaporization for each component at their respective normal boiling points", "[J/mol]", None),
'Hvap_Tbs_mass': ("list[float]", "Heats of vaporization for each component at their respective normal boiling points", "[J/kg]", None),
'InChI_Keys': ("list[str]", "InChI Keys for each component", "[-]", None),
'InChIs': ("list[str]", "InChI strings for each component", "[-]", None),
'legal_statuses': ("list[dict]", "Status of each component in in relation to import and export rules from various regions", "[-]", None),
'LFLs': ("list[float]", "Lower flammability limits for each component", "[-]", None),
'logPs': ("list[float]", "Octanol-water partition coefficients for each component", "[-]", None),
'molecular_diameters': ("list[float]", "Lennard-Jones molecular diameters for each component", "[Angstrom]", None),
'MWs': ("list[float]", "Molecular weights for each component", "[g/mol]", None),
'names': ("list[str]", "Names for each component", "[-]", None),
'ODPs': ("list[float]", "Ozone Depletion Potentials for each component", "[(impact/mass chemical)/(impact/mass CFC-11)]", None),
'omegas': ("list[float]", "Acentric factors for each component", "[-]", None),
'Parachors': ("list[float]", "Parachors for each component", "[N^0.25*m^2.75/mol]", None),
'Pcs': ("list[float]", "Critical pressures for each component", "[Pa]", None),
'phase_STPs': ("list[str]", "Standard states ('g', 'l', or 's') for each component", "[-]", None),
'Psat_298s': ("list[float]", "Vapor pressures for each component at 298.15 K", "[Pa]", None),
'PSRK_groups': ("list[dict]", "PSRK subgroup: count groups for each component", "[-]", None),
'Pts': ("list[float]", "Triple point pressures for each component", "[Pa]", None),
'PubChems': ("list[int]", "Pubchem IDs for each component", "[-]", None),
'rhocs': ("list[float]", "Molar densities at the critical point for each component", "[mol/m^3]", None),
'rhocs_mass': ("list[float]", "Densities at the critical point for each component", "[kg/m^3]", None),
'rhol_STPs': ("list[float]", "Molar liquid densities at STP for each component", "[mol/m^3]", None),
'rhol_STPs_mass': ("list[float]", "Liquid densities at STP for each component", "[kg/m^3]", None),
'RIs': ("list[float]", "Refractive indexes for each component", "[-]", None),
'RI_Ts': ("list[float]", "Temperatures at which the refractive indexes were reported for each component", "[K]", None),
'S0gs': ("list[float]", "Ideal gas absolute molar entropies at 298.15 K at 1 atm for each component", "[J/(mol*K)]", None),
'S0gs_mass': ("list[float]", "Ideal gas absolute entropies at 298.15 K at 1 atm for each component", "[J/(kg*K)]", None),
'Sfgs': ("list[float]", "Ideal gas standard molar entropies of formation for each component", "[J/(mol*K)]", None),
'Sfgs_mass': ("list[float]", "Ideal gas standard entropies of formation for each component", "[J/(kg*K)]", None),
'MWs': ("list[float]", "Similatiry variables for each component", "[mol/g]", None),
'solubility_parameters': ("list[float]", "Solubility parameters for each component", "[Pa^0.5]", None),
'Skins': ("list[bool]", "Whether each compound can be absorbed through the skin or not", "[-]", None),
'smiless': ("list[str]", "SMILES identifiers for each component", "[-]", None),
'STELs': ("list[tuple[(float, str)]]", "Short term exposure limits to chemicals (and their units)", "[ppm or mg/m^3]", None),
'StielPolars': ("list[float]", "Stiel polar factors for each component", "[-]", None),
'Stockmayers': ("list[float]", "Lennard-Jones Stockmayer parameters (depth of potential-energy minimum over k) for each component", "[K]", None),
'Tautoignitions': ("list[float]", "Autoignition temperatures for each component", "[K]", None),
'Tbs': ("list[float]", "Boiling temperatures for each component", "[K]", None),
'Tcs': ("list[float]", "Critical temperatures for each component", "[K]", None),
'Tms': ("list[float]", "Melting temperatures for each component", "[K]", None),
'Tflashs': ("list[float]", "Flash point temperatures for each component", "[K]", None),
'Tts': ("list[float]", "Triple point temperatures for each component", "[K]", None),
'TWAs': ("list[tuple[(float, str)]]", "Time-weighted average exposure limits to chemicals (and their units)", "[ppm or mg/m^3]", None),
'UFLs': ("list[float]", "Upper flammability limits for each component", "[-]", None),
'UNIFAC_Dortmund_groups': ("list[dict]", "UNIFAC_Dortmund_group: count groups for each component", "[-]", None),
'UNIFAC_groups': ("list[dict]", "UNIFAC_group: count groups for each component", "[-]", None),
'UNIFAC_Rs': ("list[float]", "UNIFAC `R` parameters for each component", "[-]", None),
'UNIFAC_Qs': ("list[float]", "UNIFAC `Q` parameters for each component", "[-]", None),
'Van_der_Waals_areas': ("list[float]", "Unnormalized Van der Waals areas for each component", "[m^2/mol]", None),
'Van_der_Waals_volumes': ("list[float]", "Unnormalized Van der Waals volumes for each component", "[m^3/mol]", None),
'Vcs': ("list[float]", "Critical molar volumes for each component", "[m^3/mol]", None),
'Vml_STPs': ("list[float]", "Liquid molar volumes for each component at STP", "[m^3/mol]", None),
'Vms_Tms': ("list[float]", "Solid molar volumes for each component at their respective melting points", "[m^3/mol]", None),
'Vml_60Fs': ("list[float]", "Liquid molar volumes for each component at 60 °F", "[m^3/mol]", None),
'rhos_Tms': ("list[float]", "Solid molar densities for each component at their respective melting points", "[mol/m^3]", None),
'rhol_60Fs': ("list[float]", "Liquid molar densities for each component at 60 °F", "[mol/m^3]", None),
'rhol_60Fs_mass': ("list[float]", "Liquid mass densities for each component at 60 °F", "[kg/m^3]", None),
'Zcs': ("list[float]", "Critical compressibilities for each component", "[-]", None),
'n_atoms': ("int", "Number of total atoms in a collection of 1 molecule of each species", "[-]", None),
'water_index': ("int", "Index of water in the package", "[-]", None),
}

constants_doc = r'''Class for storing efficiently chemical constants for a
group of components. All arguments are attributes. This is intended as a base
object from which a set of thermodynamic methods can access miscellaneous for
purposes such as phase identification or initialization.

Examples
--------
Create a package with water and the xylenes, suitable for use with equations of
state:

>>> ChemicalConstantsPackage(MWs=[18.01528, 106.165, 106.165, 106.165], names=['water', 'o-xylene', 'p-xylene', 'm-xylene'], omegas=[0.344, 0.3118, 0.324, 0.331], Pcs=[22048320.0, 3732000.0, 3511000.0, 3541000.0], Tcs=[647.14, 630.3, 616.2, 617.0])
ChemicalConstantsPackage(MWs=[18.01528, 106.165, 106.165, 106.165], names=['water', 'o-xylene', 'p-xylene', 'm-xylene'], omegas=[0.344, 0.3118, 0.324, 0.331], Pcs=[22048320.0, 3732000.0, 3511000.0, 3541000.0], Tcs=[647.14, 630.3, 616.2, 617.0])


Attributes
----------
'''
for name, (var_type, desc, units, return_desc) in constants_docstrings.items():
    type_name = var_type if type(var_type) is str else var_type.__name__
    new = '''%s : %s
    %s, %s.
''' %(name, type_name, desc, units)
    constants_doc += new

try:
    ChemicalConstantsPackage.__doc__ = constants_doc
except:
    pass # py2
#print(constants_doc)

class PropertyCorrelationPackage(object):
    r'''Class for creating and storing `T` and `P` and `zs` dependent chemical
    property objects. All parameters are also attributes.

    This object can be used either to hold already-created property objects;
    or to create new ones and hold them.

    Parameters
    ----------
    constants : :obj:`ChemicalConstantsPackage`
        Object holding all constant properties, [-]
    VaporPressures : list[:obj:`thermo.vapor_pressure.VaporPressure`], optional
        Objects holding vapor pressure data and methods, [-]
    SublimationPressures : list[:obj:`thermo.vapor_pressure.SublimationPressure`], optional
        Objects holding sublimation pressure data and methods, [-]
    VolumeGases : list[:obj:`thermo.volume.VolumeGas`], optional
        Objects holding volume data and methods, [-]
    VolumeGases : list[:obj:`thermo.volume.VolumeGas`], optional
        Objects holding gas volume data and methods, [-]
    VolumeLiquids : list[:obj:`thermo.volume.VolumeLiquid`], optional
        Objects holding liquid volume data and methods, [-]
    VolumeSolids : list[:obj:`thermo.volume.VolumeSolid`], optional
        Objects holding solid volume data and methods, [-]
    HeatCapacityGases : list[:obj:`thermo.heat_capacity.HeatCapacityGas`], optional
        Objects holding gas heat capacity data and methods, [-]
    HeatCapacityLiquids : list[:obj:`thermo.heat_capacity.HeatCapacityLiquid`], optional
        Objects holding liquid heat capacity data and methods, [-]
    HeatCapacitySolids : list[:obj:`thermo.heat_capacity.HeatCapacitySolid`], optional
        Objects holding solid heat capacity data and methods, [-]
    ViscosityGases : list[:obj:`thermo.viscosity.ViscosityGas`], optional
        Objects holding gas viscosity data and methods, [-]
    ViscosityLiquids : list[:obj:`thermo.viscosity.ViscosityLiquid`], optional
        Objects holding liquid viscosity data and methods, [-]
    ThermalConductivityGases : list[:obj:`thermo.thermal_conductivity.ThermalConductivityGas`], optional
        Objects holding gas thermal conductivity data and methods, [-]
    ThermalConductivityLiquids : list[:obj:`thermo.thermal_conductivity.ThermalConductivityLiquid`], optional
        Objects holding liquid thermal conductivity data and methods, [-]
    EnthalpyVaporizations : list[:obj:`thermo.phase_change.EnthalpyVaporization`], optional
        Objects holding enthalpy of vaporization data and methods, [-]
    EnthalpySublimations : list[:obj:`thermo.phase_change.EnthalpySublimation`], optional
        Objects holding enthalpy of sublimation data and methods, [-]
    SurfaceTensions : list[:obj:`thermo.interface.SurfaceTension`], optional
        Objects holding surface tension data and methods, [-]
    Permittivities : list[:obj:`thermo.permittivity.Permittivity`], optional
        Objects holding permittivity data and methods, [-]
    skip_missing : bool, optional
        If False, any properties not provided will have objects created; if
        True, no extra objects will be created.
    VolumeSolidMixture : :obj:`thermo.volume.VolumeSolidMixture`, optional
        Predictor object for the volume of solid mixtures, [-]
    VolumeLiquidMixture : :obj:`thermo.volume.VolumeLiquidMixture`, optional
        Predictor object for the volume of liquid mixtures, [-]
    VolumeGasMixture : :obj:`thermo.volume.VolumeGasMixture`, optional
        Predictor object for the volume of gas mixtures, [-]
    HeatCapacityLiquidMixture : :obj:`thermo.heat_capacity.HeatCapacityLiquidMixture`, optional
        Predictor object for the heat capacity of liquid mixtures, [-]
    HeatCapacityGasMixture : :obj:`thermo.heat_capacity.HeatCapacityGasMixture`, optional
        Predictor object for the heat capacity of gas mixtures, [-]
    HeatCapacitySolidMixture : :obj:`thermo.heat_capacity.HeatCapacitySolidMixture`, optional
        Predictor object for the heat capacity of solid mixtures, [-]
    ViscosityLiquidMixture : :obj:`thermo.viscosity.ViscosityLiquidMixture`, optional
        Predictor object for the viscosity of liquid mixtures, [-]
    ViscosityGasMixture : :obj:`thermo.viscosity.ViscosityGasMixture`, optional
        Predictor object for the viscosity of gas mixtures, [-]
    ThermalConductivityLiquidMixture : :obj:`thermo.thermal_conductivity.ThermalConductivityLiquidMixture`, optional
        Predictor object for the thermal conductivity of liquid mixtures, [-]
    ThermalConductivityGasMixture : :obj:`thermo.thermal_conductivity.ThermalConductivityGasMixture`, optional
        Predictor object for the thermal conductivity of gas mixtures, [-]
    SurfaceTensionMixture : :obj:`thermo.interface.SurfaceTensionMixture`, optional
        Predictor object for the surface tension of liquid mixtures, [-]

    Attributes
    ----------
    pure_correlations : tuple(str)
        List of all pure component property objects, [-]

    Examples
    --------

    Create a package from CO2 and n-hexane, with ideal-gas heat capacities
    provided while excluding all other properties:

    >>> constants = ChemicalConstantsPackage(CASs=['124-38-9', '110-54-3'], MWs=[44.0095, 86.17536], names=['carbon dioxide', 'hexane'], omegas=[0.2252, 0.2975], Pcs=[7376460.0, 3025000.0], Tbs=[194.67, 341.87], Tcs=[304.2, 507.6], Tms=[216.65, 178.075])
    >>> correlations = PropertyCorrelationPackage(constants=constants, skip_missing=True, HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])), HeatCapacityGas(best_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998]))])

    Create a package from various data files, creating all property objects:

    >>> correlations = PropertyCorrelationPackage(constants=constants, skip_missing=True)

    '''
    correlations = ('VaporPressures', 'SublimationPressures', 'VolumeGases',
               'VolumeLiquids', 'VolumeSolids', 'HeatCapacityGases',
               'HeatCapacityLiquids', 'HeatCapacitySolids', 'ViscosityGases',
               'ViscosityLiquids', 'ThermalConductivityGases', 'ThermalConductivityLiquids',
               'EnthalpyVaporizations', 'EnthalpySublimations', 'SurfaceTensions',
               'Permittivities',

               'VolumeGasMixture', 'VolumeLiquidMixture', 'VolumeSolidMixture',
               'HeatCapacityGasMixture', 'HeatCapacityLiquidMixture',
               'HeatCapacitySolidMixture', 'ViscosityGasMixture',
               'ViscosityLiquidMixture', 'ThermalConductivityGasMixture',
               'ThermalConductivityLiquidMixture', 'SurfaceTensionMixture',
               )

    __slots__ = correlations + ('constants',)

    pure_correlations = ('VaporPressures', 'VolumeLiquids', 'VolumeGases',
                         'VolumeSolids', 'HeatCapacityGases', 'HeatCapacitySolids',
                         'HeatCapacityLiquids', 'EnthalpyVaporizations',
                         'EnthalpySublimations', 'SublimationPressures',
                         'Permittivities', 'ViscosityLiquids', 'ViscosityGases',
                         'ThermalConductivityLiquids', 'ThermalConductivityGases',
                         'SurfaceTensions')

    def subset(self, idxs, skip_missing=False):
        is_slice = isinstance(idxs, slice)

        def atindexes(values):
            if is_slice:
                return values[idxs]
            return [values[i] for i in idxs]

        new = {'constants': self.constants.subset(idxs)}
        for p in self.pure_correlations:
            if hasattr(self, p):
                v = getattr(self, p)
                if v is not None:
                    new[p] = atindexes(v)
        return PropertyCorrelationPackage(skip_missing=skip_missing, **new)


    def __init__(self, constants, VaporPressures=None, SublimationPressures=None,
                 VolumeGases=None, VolumeLiquids=None, VolumeSolids=None,
                 HeatCapacityGases=None, HeatCapacityLiquids=None, HeatCapacitySolids=None,
                 ViscosityGases=None, ViscosityLiquids=None,
                 ThermalConductivityGases=None, ThermalConductivityLiquids=None,
                 EnthalpyVaporizations=None, EnthalpySublimations=None,
                 SurfaceTensions=None, Permittivities=None,

                 VolumeGasMixtureObj=None, VolumeLiquidMixtureObj=None, VolumeSolidMixtureObj=None,
                 HeatCapacityGasMixtureObj=None, HeatCapacityLiquidMixtureObj=None, HeatCapacitySolidMixtureObj=None,
                 ViscosityGasMixtureObj=None, ViscosityLiquidMixtureObj=None,
                 ThermalConductivityGasMixtureObj=None, ThermalConductivityLiquidMixtureObj=None,
                 SurfaceTensionMixtureObj=None, skip_missing=False,
                 ):
        self.constants = constants
        cmps = constants.cmps

        if VaporPressures is None and not skip_missing:
            VaporPressures = [VaporPressure(Tb=constants.Tbs[i], Tc=constants.Tcs[i], Pc=constants.Pcs[i],
                                            omega=constants.omegas[i], CASRN=constants.CASs[i],
                                            best_fit=get_chemical_constants(constants.CASs[i], 'VaporPressure'))
                              for i in cmps]

        if VolumeLiquids is None and not skip_missing:
            VolumeLiquids = [VolumeLiquid(MW=constants.MWs[i], Tb=constants.Tbs[i], Tc=constants.Tcs[i],
                              Pc=constants.Pcs[i], Vc=constants.Vcs[i], Zc=constants.Zcs[i], omega=constants.omegas[i],
                              dipole=constants.dipoles[i],
                              Psat=VaporPressures[i],
                              best_fit=get_chemical_constants(constants.CASs[i], 'VolumeLiquid'),
                              eos=None, CASRN=constants.CASs[i])
                              for i in cmps]

        if VolumeGases is None and not skip_missing:
            VolumeGases = [VolumeGas(MW=constants.MWs[i], Tc=constants.Tcs[i], Pc=constants.Pcs[i],
                                   omega=constants.omegas[i], dipole=constants.dipoles[i],
                                   eos=None, CASRN=constants.CASs[i])
                              for i in cmps]

        if VolumeSolids is None and not skip_missing:
            VolumeSolids = [VolumeSolid(CASRN=constants.CASs[i], MW=constants.MWs[i],
                                        Tt=constants.Tts[i], Vml_Tt=constants.Vml_Tms[i])
                              for i in cmps]

        if HeatCapacityGases is None and not skip_missing:
            HeatCapacityGases = [HeatCapacityGas(CASRN=constants.CASs[i], MW=constants.MWs[i],
                                                 similarity_variable=constants.similarity_variables[i],
                                                 best_fit=get_chemical_constants(constants.CASs[i], 'HeatCapacityGas'))
                              for i in cmps]

        if HeatCapacitySolids is None and not skip_missing:
            HeatCapacitySolids = [HeatCapacitySolid(MW=constants.MWs[i], similarity_variable=constants.similarity_variables[i],
                                                    CASRN=constants.CASs[i], best_fit=get_chemical_constants(constants.CASs[i], 'HeatCapacitySolid'))
                              for i in cmps]

        if HeatCapacityLiquids is None and not skip_missing:
            HeatCapacityLiquids = [HeatCapacityLiquid(CASRN=constants.CASs[i], MW=constants.MWs[i],
                                                      similarity_variable=constants.similarity_variables[i],
                                                      Tc=constants.Tcs[i], omega=constants.omegas[i],
                                                      Cpgm=HeatCapacityGases[i], best_fit=get_chemical_constants(constants.CASs[i], 'HeatCapacityLiquid'))
                              for i in cmps]

        if EnthalpyVaporizations is None and not skip_missing:
            EnthalpyVaporizations = [EnthalpyVaporization(CASRN=constants.CASs[i], Tb=constants.Tbs[i],
                                                          Tc=constants.Tcs[i], Pc=constants.Pcs[i], omega=constants.omegas[i],
                                                          similarity_variable=constants.similarity_variables[i],
                                                          best_fit=get_chemical_constants(constants.CASs[i], 'EnthalpyVaporization'))
                              for i in cmps]

        if EnthalpySublimations is None and not skip_missing:
            EnthalpySublimations = [EnthalpySublimation(CASRN=constants.CASs[i], Tm=constants.Tms[i], Tt=constants.Tts[i],
                                                       Cpg=HeatCapacityGases[i], Cps=HeatCapacitySolids[i],
                                                       Hvap=EnthalpyVaporizations[i])
                                    for i in cmps]

        if SublimationPressures is None and not skip_missing:
            SublimationPressures = [SublimationPressure(CASRN=constants.CASs[i], Tt=constants.Tts[i], Pt=constants.Pts[i],
                                                        Hsub_t=constants.Hsub_Tms[i])
                                    for i in cmps]

        if Permittivities is None and not skip_missing:
            Permittivities = [Permittivity(CASRN=constants.CASs[i]) for i in cmps]

        # missing -  ThermalConductivityGas, SurfaceTension
        if ViscosityLiquids is None and not skip_missing:
            ViscosityLiquids = [ViscosityLiquid(CASRN=constants.CASs[i], MW=constants.MWs[i], Tm=constants.Tms[i],
                                                Tc=constants.Tcs[i], Pc=constants.Pcs[i], Vc=constants.Vcs[i],
                                                omega=constants.omegas[i], Psat=VaporPressures[i].T_dependent_property,
                                                Vml=VolumeLiquids[i])
                                for i in cmps]


        if ViscosityGases is None and not skip_missing:
            ViscosityGases = [ViscosityGas(CASRN=constants.CASs[i], MW=constants.MWs[i], Tc=constants.Tcs[i],
                                           Pc=constants.Pcs[i], Zc=constants.Zcs[i], dipole=constants.dipoles[i],
                                           Vmg=lambda T: VolumeGases[i](T, 101325.0)) # Might be an issue with what i refers too
                                for i in cmps]
        if ThermalConductivityLiquids is None and not skip_missing:
            ThermalConductivityLiquids = [ThermalConductivityLiquid(CASRN=constants.CASs[i], MW=constants.MWs[i],
                                                                    Tm=constants.Tms[i], Tb=constants.Tbs[i],
                                                                    Tc=constants.Tcs[i], Pc=constants.Pcs[i],
                                                                    omega=constants.omegas[i], Hfus=constants.Hfus_Tms[i])
                                                for i in cmps]

        if ThermalConductivityGases is None and not skip_missing:
            ThermalConductivityGases = [ThermalConductivityGas(CASRN=constants.CASs[i], MW=constants.MWs[i], Tb=constants.Tbs[i],
                                                               Tc=constants.Tcs[i], Pc=constants.Pcs[i], Vc=constants.Vcs[i],
                                                               Zc=constants.Zcs[i], omega=constants.omegas[i], dipole=constants.dipoles[i],
                                                               Vmg=VolumeGases[i], mug=ViscosityLiquids[i].T_dependent_property,
                                                               Cvgm=lambda T : HeatCapacityGases[i].T_dependent_property(T) - R)
                                                for i in cmps]

        if SurfaceTensions is None and not skip_missing:
            SurfaceTensions = [SurfaceTension(CASRN=constants.CASs[i], MW=constants.MWs[i], Tb=constants.Tbs[i],
                                              Tc=constants.Tcs[i], Pc=constants.Pcs[i], Vc=constants.Vcs[i],
                                              Zc=constants.Zcs[i], omega=constants.omegas[i], StielPolar=constants.StielPolars[i],
                                              Hvap_Tb=constants.Hvap_Tbs[i], Vml=VolumeLiquids[i].T_dependent_property,
                                              Cpl=lambda T : property_molar_to_mass(HeatCapacityLiquids[i].T_dependent_property(T), constants.MWs[i]))
                                    for i in cmps]

        self.VaporPressures = VaporPressures
        self.VolumeLiquids = VolumeLiquids
        self.VolumeGases = VolumeGases
        self.VolumeSolids = VolumeSolids
        self.HeatCapacityGases = HeatCapacityGases
        self.HeatCapacitySolids = HeatCapacitySolids
        self.HeatCapacityLiquids = HeatCapacityLiquids
        self.EnthalpyVaporizations = EnthalpyVaporizations
        self.EnthalpySublimations = EnthalpySublimations
        self.SublimationPressures = SublimationPressures
        self.Permittivities = Permittivities
        self.ViscosityLiquids = ViscosityLiquids
        self.ViscosityGases = ViscosityGases
        self.ThermalConductivityLiquids = ThermalConductivityLiquids
        self.ThermalConductivityGases = ThermalConductivityGases
        self.SurfaceTensions = SurfaceTensions

        # Mixture objects

        if VolumeSolidMixtureObj is None and not skip_missing:
            VolumeSolidMixtureObj = VolumeSolidMixture(CASs=constants.CASs, MWs=constants.MWs, VolumeSolids=VolumeSolids)
        if VolumeLiquidMixtureObj is None and not skip_missing:
            VolumeLiquidMixtureObj = VolumeLiquidMixture(MWs=constants.MWs, Tcs=constants.Tcs, Pcs=constants.Pcs, Vcs=constants.Vcs, Zcs=constants.Zcs, omegas=constants.omegas, CASs=constants.CASs, VolumeLiquids=VolumeLiquids)
        if VolumeGasMixtureObj is None and not skip_missing:
            VolumeGasMixtureObj = VolumeGasMixture(eos=None, MWs=constants.MWs, CASs=constants.CASs, VolumeGases=VolumeGases)

        if HeatCapacityLiquidMixtureObj is None and not skip_missing:
            HeatCapacityLiquidMixtureObj = HeatCapacityLiquidMixture(MWs=constants.MWs, CASs=constants.CASs, HeatCapacityLiquids=HeatCapacityLiquids)
        if HeatCapacityGasMixtureObj is None and not skip_missing:
            HeatCapacityGasMixtureObj = HeatCapacityGasMixture(MWs=constants.MWs, CASs=constants.CASs, HeatCapacityGases=HeatCapacityGases)
        if HeatCapacitySolidMixtureObj is None and not skip_missing:
            HeatCapacitySolidMixtureObj = HeatCapacitySolidMixture(MWs=constants.MWs, CASs=constants.CASs, HeatCapacitySolids=HeatCapacitySolids)

        if ViscosityLiquidMixtureObj is None and not skip_missing:
            ViscosityLiquidMixtureObj = ViscosityLiquidMixture(MWs=constants.MWs, CASs=constants.CASs, ViscosityLiquids=ViscosityLiquids)
        if ViscosityGasMixtureObj is None and not skip_missing:
            ViscosityGasMixtureObj = ViscosityGasMixture(MWs=constants.MWs, molecular_diameters=constants.molecular_diameters, Stockmayers=constants.Stockmayers, CASs=constants.CASs, ViscosityGases=ViscosityGases)

        if ThermalConductivityLiquidMixtureObj is None and not skip_missing:
            ThermalConductivityLiquidMixtureObj = ThermalConductivityLiquidMixture(CASs=constants.CASs, MWs=constants.MWs, ThermalConductivityLiquids=ThermalConductivityLiquids)
        if ThermalConductivityGasMixtureObj is None and not skip_missing:
            ThermalConductivityGasMixtureObj = ThermalConductivityGasMixture(MWs=constants.MWs, Tbs=constants.Tbs, CASs=constants.CASs, ThermalConductivityGases=ThermalConductivityGases, ViscosityGases=ViscosityGases)

        if SurfaceTensionMixtureObj is None and not skip_missing:
            SurfaceTensionMixtureObj = SurfaceTensionMixture(MWs=constants.MWs, Tbs=constants.Tbs, Tcs=constants.Tcs, CASs=constants.CASs, SurfaceTensions=SurfaceTensions, VolumeLiquids=VolumeLiquids)

        self.VolumeSolidMixture = VolumeSolidMixtureObj
        self.VolumeLiquidMixture = VolumeLiquidMixtureObj
        self.VolumeGasMixture = VolumeGasMixtureObj

        self.HeatCapacityLiquidMixture = HeatCapacityLiquidMixtureObj
        self.HeatCapacityGasMixture = HeatCapacityGasMixtureObj
        self.HeatCapacitySolidMixture = HeatCapacitySolidMixtureObj

        self.ViscosityLiquidMixture = ViscosityLiquidMixtureObj
        self.ViscosityGasMixture = ViscosityGasMixtureObj

        self.ThermalConductivityLiquidMixture = ThermalConductivityLiquidMixtureObj
        self.ThermalConductivityGasMixture = ThermalConductivityGasMixtureObj

        self.SurfaceTensionMixture = SurfaceTensionMixtureObj

    def as_best_fit(self, props=None):
        multiple_props = isinstance(props, (tuple, list)) and isinstance(props[0], str)
        if props is None or multiple_props:
            if multiple_props:
                iter_props = props
            else:
                iter_props = self.pure_correlations

            s = '%s(' %(self.__class__.__name__)
            s += 'constants=constants, skip_missing=True,\n'
            for prop in iter_props:
                prop_attr = getattr(self, prop)
                if prop_attr is not None:
                    try:
                        s += '%s=%s,\n' %(prop, self.as_best_fit(prop_attr))
                    except Exception as e:
                        print(e, prop)

            s += ')'
            return s

        s = '['
        for obj in props:
            s += (obj.as_best_fit() + ',\n')
        s += ']'
        return s


# Values except for omega from IAPWS; heat capacity isn't official.
iapws_constants = ChemicalConstantsPackage(CASs=['7732-18-5'], MWs=[18.015268], omegas=[0.344],
                                           Pcs=[22064000.0], Tcs=[647.096])
'''ChemicalConstantsPackage : Object intended to hold the IAPWS-95 water constants
for use with the :obj:`thermo.phases.IAPWS95` phase object.
'''

global _iapws_correlations
_iapws_correlations = None
def iapws_correlations():
    '''Function to construct a global IAPWS T/P dependent property
    :obj:`PropertyCorrelationPackage` object.

    Returns
    -------
    iapws_correlations : :obj:`PropertyCorrelationPackage`
        IAPWS correlations and properties, [-]
    '''
    global _iapws_correlations
    if _iapws_correlations is None:
        _iapws_correlations = PropertyCorrelationPackage(constants=iapws_constants,
                        HeatCapacityGases=[HeatCapacityGas(best_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18,
                                                                                    4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))])

    return _iapws_correlations