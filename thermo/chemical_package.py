# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020, 2021 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
    :members: subset, properties, correlations_from_IDs, constants_from_IDs,
              from_IDs, with_new_constants, as_json, from_json
    :undoc-members:
    :exclude-members:

Chemical Correlations Class
===========================

.. autoclass:: PropertyCorrelationsPackage
    :members: subset
    :undoc-members:
    :exclude-members:

Sample Constants and Correlations
=================================
.. autodata:: iapws_constants
.. autodata:: iapws_correlations
.. autodata:: lemmon2000_constants
.. autodata:: lemmon2000_correlations

'''

from __future__ import division

__all__ = ['ChemicalConstantsPackage', 'PropertyCorrelationsPackage',
           'iapws_constants', 'iapws_correlations', 'lemmon2000_constants',
           'lemmon2000_correlations']

from fluids.constants import R

from thermo.chemical import Chemical, get_chemical_constants
from chemicals.identifiers import *
from chemicals import identifiers
from chemicals.utils import hash_any_primitive

from thermo.thermal_conductivity import ThermalConductivityLiquid, ThermalConductivityGas, ThermalConductivityLiquidMixture, ThermalConductivityGasMixture
from thermo.volume import VolumeLiquidMixture, VolumeGasMixture, VolumeSolidMixture, VolumeLiquid, VolumeGas, VolumeSolid
from thermo.permittivity import *
from thermo.heat_capacity import HeatCapacitySolid, HeatCapacityGas, HeatCapacityLiquid, HeatCapacitySolidMixture, HeatCapacityGasMixture, HeatCapacityLiquidMixture
from thermo.interface import SurfaceTension, SurfaceTensionMixture
from thermo.viscosity import ViscosityLiquid, ViscosityGas, ViscosityLiquidMixture, ViscosityGasMixture
from chemicals.utils import property_molar_to_mass, Parachor
from thermo.utils import *
from thermo.vapor_pressure import VaporPressure, SublimationPressure
from thermo.phase_change import EnthalpyVaporization, EnthalpySublimation

# For Constants
from chemicals.critical import Tc, Pc, Vc
from chemicals.phase_change import Tb, Tm, Hfus
from chemicals.acentric import omega, Stiel_polar_factor
from chemicals.triple import Tt, Pt
from chemicals.reaction import Hfs, Hfl, Hfg, S0g, S0l, S0s, Gibbs_formation, Hf_basis_converter, entropy_formation
from chemicals.safety import T_flash, T_autoignition, LFL, UFL, TWA, STEL, Ceiling, Skin, Carcinogen
from chemicals.solubility import solubility_parameter
from chemicals.dipole import dipole_moment as dipole
from chemicals.lennard_jones import Stockmayer, molecular_diameter
from chemicals.environment import GWP, ODP, logP
from chemicals.refractivity import RI
from chemicals.elements import atom_fractions, mass_fractions, similarity_variable, atoms_to_Hill, simple_formula_parser, molecular_weight, charge_from_formula, periodic_table, homonuclear_elements
from chemicals.combustion import combustion_stoichiometry, HHV_stoichiometry, LHV_from_HHV

from thermo.unifac import DDBST_UNIFAC_assignments, DDBST_MODIFIED_UNIFAC_assignments, DDBST_PSRK_assignments, load_group_assignments_DDBST, UNIFAC_RQ, Van_der_Waals_volume, Van_der_Waals_area
from thermo.electrochem import conductivity
from thermo.law import legal_status, economic_status
from thermo.eos import PR
from thermo import serialize

CAS_H2O = '7732-18-5'



warn_chemicals_msg ='''`chemicals <https://github.com/CalebBell/chemicals>`_ is a
            project with a focus on collecting data and
            correlations from various sources. In no way is it a project to
            critically evaluate these and provide recommendations. You are
            strongly encouraged to check values from it and modify them
            if you want different values. If you believe there is a value
            which has a typographical error please report it to the
            `chemicals <https://github.com/CalebBell/chemicals>`_
            project. If data is missing or not as accuracte
            as you would like, and you know of a better method or source,
            new methods and sources can be added to
            `chemicals <https://github.com/CalebBell/chemicals>`_
            fairly easily once the data entry is complete.
            It is not feasible to add individual components,
            so please submit a complete table of data from the source.'''


class ChemicalConstantsPackage(object):
    non_vector_properties = ('atomss', 'Carcinogens', 'CASs', 'Ceilings', 'charges',
                 'conductivities', 'dipoles', 'economic_statuses', 'formulas', 'Gfgs',
                 'Gfgs_mass', 'GWPs', 'Hcs', 'Hcs_lower', 'Hcs_lower_mass', 'Hcs_mass',
                 'Hfgs', 'Hfgs_mass', 'Hfus_Tms', 'Hfus_Tms_mass', 'Hsub_Tts',
                 'Hsub_Tts_mass', 'Hvap_298s', 'Hvap_298s_mass', 'Hvap_Tbs', 'Hvap_Tbs_mass',
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
                 'rhos_Tms', 'Vms_Tms', 'rhos_Tms_mass',  'solubility_parameters',
                 'Vml_60Fs', 'rhol_60Fs', 'rhol_60Fs_mass',
                 'conductivity_Ts', 'RI_Ts',
                 'Vmg_STPs', 'rhog_STPs', 'rhog_STPs_mass', 'sigma_STPs',
                 'sigma_Tms', 'sigma_Tbs', 'Hf_STPs', 'Hf_STPs_mass',
                 )
    __full_path__ = "%s.%s" %(__module__, __qualname__)
    properties = ('atom_fractions',) + non_vector_properties
    '''Tuple of all properties that can be held by this object.'''

    __slots__ = properties + ('N', 'cmps', 'water_index', 'n_atoms') + ('json_version', '_hash')
    non_vectors = ('atom_fractions',)
    non_vectors_set = set(non_vectors)

    def _missing_properties(self):
        missing = []
        empty_list = [None]*self.N
        for prop in self.non_vector_properties:
            if getattr(self, prop) == empty_list:
                missing.append(prop)
        return tuple(missing)

    @property
    def __dict__(self):
        new = {}
        properties = self.properties
        for p in properties:
            v = getattr(self, p)
            if v is not None:
                new[p] = v
        return new

    def as_json(self):
        r'''Method to create a JSON friendly serialization of the chemical constants
        package which can be stored, and reloaded later.

        Returns
        -------
        json_repr : dict
            Json friendly representation, [-]

        Notes
        -----

        Examples
        --------
        >>> import json
        >>> constants = ChemicalConstantsPackage(MWs=[18.01528, 106.165], names=['water', 'm-xylene'])
        >>> string = json.dumps(constants.as_json())
        '''
        d = self.__dict__.copy()
        for k in ('PSRK_groups', 'UNIFAC_Dortmund_groups', 'UNIFAC_groups'):
            # keys are stored as strings and not ints
            d[k] = [{str(k): v for k, v in r.items()} if r is not None else r for r in d[k]]

        d['json_version'] = 1
        d['py/object'] = self.__full_path__
        return d

    @classmethod
    def from_json(cls, json_repr):
        r'''Method to create a ChemicalConstantsPackage from a JSON
        serialization of another ChemicalConstantsPackage.

        Parameters
        ----------
        json_repr : dict
            Json representation, [-]

        Returns
        -------
        constants : ChemicalConstantsPackage
            Newly created object from the json serialization, [-]

        Notes
        -----
        It is important that the input be in the same format as that
        created by :obj:`ChemicalConstantsPackage.as_json`.

        Examples
        --------
        >>> import json
        >>> constants = ChemicalConstantsPackage(MWs=[18.01528, 106.165], names=['water', 'm-xylene'])
        >>> string = json.dumps(constants.as_json())
        >>> new_constants  = ChemicalConstantsPackage.from_json(json.loads(string))
        >>> assert hash(new_constants) == hash(constants)
        '''
        d = json_repr

        for k in ('TWAs', 'STELs'):
            # tuple gets converted to a json list
            l = d[k]
            d[k] = [tuple(v) if v is not None else v for v in l]

        for k in ('PSRK_groups', 'UNIFAC_Dortmund_groups', 'UNIFAC_groups'):
            # keys are stored as strings and not ints
            d[k] = [{int(k): v for k, v in r.items()} if r is not None else r for r in d[k]]

        del d['json_version']
        del d['py/object']
        return cls(**d)

    def __hash__(self):
        try:
            return self._hash
        except:
            pass
        hashes = []
        for k in self.properties:
            hashes.append(hash_any_primitive(getattr(self, k)))
        self._hash = hash_any_primitive(hashes)
        return self._hash

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def with_new_constants(self, **kwargs):
        r'''Method to construct a new ChemicalConstantsPackage that replaces or
        adds one or more properties for all components.

        Parameters
        ----------
        kwargs : dict[str: list[float]]
            Properties specified by name [various]

        Returns
        -------
        new_constants : ChemicalConstantsPackage
            Object with new and/or replaced properties, [-]

        Notes
        -----

        Examples
        --------
        >>> base = ChemicalConstantsPackage(MWs=[18.01528, 106.165, 106.165, 106.165], names=['water', 'o-xylene', 'p-xylene', 'm-xylene'], omegas=[0.344, 0.3118, 0.324, 0.331], Pcs=[22048320.0, 3732000.0, 3511000.0, 3541000.0], Tcs=[647.14, 630.3, 616.2, 617.0])
        >>> base.with_new_constants(Tms=[40.0, 20.0, 10.0, 30.0], omegas=[0.0, 0.1, 0.2, 0.3])
        ChemicalConstantsPackage(MWs=[18.01528, 106.165, 106.165, 106.165], names=['water', 'o-xylene', 'p-xylene', 'm-xylene'], omegas=[0.0, 0.1, 0.2, 0.3], Pcs=[22048320.0, 3732000.0, 3511000.0, 3541000.0], Tcs=[647.14, 630.3, 616.2, 617.0], Tms=[40.0, 20.0, 10.0, 30.0])
        '''
        new = {}
        properties = self.non_vector_properties
        for p in properties:
            v = getattr(self, p)
            if v is not None:
                new[p] = v
        new.update(kwargs)
        return ChemicalConstantsPackage(**new)


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
        ChemicalConstantsPackage(MWs=[106.165, 106.165, 106.165], names=['o-xylene', 'p-xylene', 'm-xylene'], omegas=[0.3118, 0.324, 0.331], Pcs=[3732000.0, 3511000.0, 3541000.0], Tcs=[630.3, 616.2, 617.0])
        >>> base.subset(idxs=[0, 3], properties=('names', 'MWs'))
        ChemicalConstantsPackage(MWs=[18.01528, 106.165], names=['water', 'm-xylene'])
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

    def compound_index(self, CAS=None, name=None, smiles=None, InChI=None, 
                       InChI_Key=None, PubChem=None):
        r'''Method to retrieve the index of a compound given one of the
        optional identifiers for the compound.
        
        Parameters
        ----------
        CAS : str, optional
            The CAS number of the compound, [-]
        name : str, optional
            The (provided) name of the compound, [-]
        smiles : str, optional
            Smiles identifier, [-]
        InChI : str, optional
            InChI identifier, [-]
        InChI_Key : str, optional
            InChI key identifier, [-]
        PubChem : int, optional
            PubChem identifier, [-]
            
        Returns
        -------
        index : int
            The index of the component (if found), [-]

        Raises
        ------
        ValueError
            If no identifier is provided for any argument, this is raised
        IndexError
            If no match is found for the provided identifier, this is raised
        '''
        if CAS is not None:
            return self.CASs.index(CAS)
        elif name is not None:
            return self.names.index(name)
        elif smiles is not None:
            return self.smiless.index(smiles)
        elif InChI is not None:
            return self.InChIs.index(InChI)
        elif InChI_Key is not None:
            return self.InChI_Keys.index(InChI_Key)
        elif PubChem is not None:
            return self.PubChems.index(PubChem)
        else:
            raise ValueError("No identifier provided")


    @staticmethod
    def constants_from_IDs(IDs):
        r'''Method to construct a new `ChemicalConstantsPackage` with loaded
        parameters from the `chemicals library <https://github.com/CalebBell/chemicals>`_,
        using whatever default methods and values happen to be in that library.
        Expect values to change over time.

        Parameters
        ----------
        IDs : list[str]
            Identifying strings for each compound;
            most identifiers are accepted and all inputs are documented in
            :obj:`chemicals.identifiers.search_chemical`, [-]

        Returns
        -------
        constants : ChemicalConstantsPackage
            New `ChemicalConstantsPackage` with loaded values, [-]

        Notes
        -----

        .. warning::
            %s

        Examples
        --------
        >>> constants = ChemicalConstantsPackage.constants_from_IDs(IDs=['water', 'hexane'])
        '''
        return ChemicalConstantsPackage._from_IDs(IDs, correlations=False)

    try:
        constants_from_IDs.__func__.__doc__ = constants_from_IDs.__func__.__doc__ %(warn_chemicals_msg)
    except:
        pass

    @staticmethod
    def correlations_from_IDs(IDs):
        r'''Method to construct a new `PropertyCorrelationsPackage` with loaded
        parameters from the `chemicals library <https://github.com/CalebBell/chemicals>`_,
        using whatever default methods and values happen to be in that library.
        Expect values to change over time.

        Parameters
        ----------
        IDs : list[str]
            Identifying strings for each compound;
            most identifiers are accepted and all inputs are documented in
            :obj:`chemicals.identifiers.search_chemical`, [-]

        Returns
        -------
        correlations : PropertyCorrelationsPackage
            New `PropertyCorrelationsPackage` with loaded values, [-]

        Notes
        -----

        .. warning::
            %s

        Examples
        --------
        >>> correlations = ChemicalConstantsPackage.constants_from_IDs(IDs=['ethanol', 'methanol'])
        '''
        return ChemicalConstantsPackage._from_IDs(IDs, correlations=True)[1]
    try:
        correlations_from_IDs.__func__.__doc__ = correlations_from_IDs.__func__.__doc__ %(warn_chemicals_msg)
    except:
        pass

    @staticmethod
    def from_IDs(IDs):
        r'''Method to construct a new `ChemicalConstantsPackage` and
        `PropertyCorrelationsPackage` with loaded
        parameters from the `chemicals library <https://github.com/CalebBell/chemicals>`_,
        using whatever default methods and values happen to be in that library.
        Expect values to change over time.

        Parameters
        ----------
        IDs : list[str]
            Identifying strings for each compound;
            most identifiers are accepted and all inputs are documented in
            :obj:`chemicals.identifiers.search_chemical`, [-]

        Returns
        -------
        constants : PropertyCorrelationsPackage
            New `PropertyCorrelationsPackage` with loaded values, [-]
        correlations : PropertyCorrelationsPackage
            New `PropertyCorrelationsPackage` with loaded values, [-]

        Notes
        -----

        .. warning::
            %s

        Examples
        --------
        >>> constants, correlations = ChemicalConstantsPackage.from_IDs(IDs=['water', 'decane'])
        '''
        return ChemicalConstantsPackage._from_IDs(IDs, correlations=True)

    try:
        from_IDs.__func__.__doc__ = from_IDs.__func__.__doc__ %(warn_chemicals_msg)
    except:
        pass

    @staticmethod
    def _from_IDs(IDs, correlations=False):

        # Properties which were wrong from Mixture, Chemical: Parachor, solubility_parameter
        N = len(IDs)
        CASs = [CAS_from_any(ID) for ID in IDs]
        pubchem_db = identifiers.pubchem_db
        metadatas = [pubchem_db.search_CAS(CAS) for CAS in CASs]
        names = [i.common_name.lower() for i in metadatas]
        PubChems = [i.pubchemid for i in metadatas]
        formulas = [i.formula for i in metadatas]
        smiless = [i.smiles for i in metadatas]
        InChIs = [i.InChI for i in metadatas]
        InChI_Keys = [i.InChI_key for i in metadatas]
        atomss = [simple_formula_parser(f) for f in formulas]
#        MWs = [i.MW for i in metadatas] # Should be the same but there are still some inconsistencies
        MWs = [molecular_weight(atomss[i]) for i in range(N)]

        similarity_variables = [similarity_variable(atoms, MW) for atoms, MW in zip(atomss, MWs)]
        charges = [charge_from_formula(formula) for formula in formulas]

        Tms = [Tm(CAS) for CAS in CASs]
        Tbs = [Tb(CAS) for CAS in CASs]
        Tcs = [Tc(CAS) for CAS in CASs]
        Pcs = [Pc(CAS) for CAS in CASs]
        Vcs = [Vc(CAS) for CAS in CASs]
        omegas = [omega(CAS) for CAS in CASs]
        dipoles = [dipole(CAS) for CAS in CASs]
        Tts = [Tt(CAS) for CAS in CASs]
        Pts = [Pt(CAS) for CAS in CASs]


        Zcs = [None]*N
        for i in range(N):
            try:
                Zcs[i] = Vcs[i]*Pcs[i]/(R*Tcs[i])
            except:
                pass
        VaporPressures = [VaporPressure(Tb=Tbs[i], Tc=Tcs[i], Pc=Pcs[i], omega=omegas[i], CASRN=CASs[i],
                                        poly_fit=get_chemical_constants(CASs[i], 'VaporPressure'))
                            for i in range(N)]
        Psat_298s = [VaporPressures[i].T_dependent_property(298.15) for i in range(N)]

        phase_STPs = [identify_phase(T=298.15, P=101325., Tm=Tms[i], Tb=Tbs[i], Tc=Tcs[i], Psat=Psat_298s[i]) for i in range(N)]
        for i in range(N):
            if Pts[i] is None:
                try:
                    Pts[i] = VaporPressures[i].T_dependent_property(Tts[i])
                except:
                    pass

        rhocs = [1.0/Vc if Vc else None for Vc in Vcs]
        rhocs_mass = [1e-3*MW/Vc if Vc else None for Vc, MW in zip(Vcs, MWs)]

        Hfus_Tms = [Hfus(CAS) for CAS in CASs]
        Hfus_Tms_mass = [Hfus*1000.0/MW if Hfus is not None else None for Hfus, MW in zip(Hfus_Tms, MWs)]

        EnthalpyVaporizations = [EnthalpyVaporization(CASRN=CAS, Tb=Tb, Tc=Tc, Pc=Pc, omega=omega, similarity_variable=sv,
                                                      poly_fit=get_chemical_constants(CAS, 'EnthalpyVaporization'))
                                 for CAS, Tb, Tc, Pc, sv, omega in zip(CASs, Tbs, Tcs, Pcs, similarity_variables, omegas)]


        Hvap_Tbs = [o.T_dependent_property(Tb) if Tb else None for o, Tb, in zip(EnthalpyVaporizations, Tbs)]
        Hvap_Tbs_mass =  [Hvap*1000.0/MW if Hvap is not None else None for Hvap, MW in zip(Hvap_Tbs, MWs)]

        Hvap_298s = [o.T_dependent_property(298.15) for o in EnthalpyVaporizations]
        Hvap_298s_mass =  [Hvap*1000.0/MW if Hvap is not None else None for Hvap, MW in zip(Hvap_298s, MWs)]

        StielPolars = [None]*N
        for i in range(N):
            try:
                StielPolars[i] = Stiel_polar_factor(Psat=VaporPressures[i].T_dependent_property(T=Tcs[i]*0.6), Pc=Pcs[i], omega=omegas[i])
            except:
                pass





        enclosed_eoss = []
        for i in range(N):
            try:
                enclosed_eoss.append([PR(T=298.15, P=101325.0, Tc=Tcs[i], Pc=Pcs[i], omega=omegas[i])])
            except:
                enclosed_eoss.append(None)
        VolumeGases = [VolumeGas(MW=MWs[i], Tc=Tcs[i], Pc=Pcs[i],
                                   omega=omegas[i], dipole=dipoles[i],
                                   eos=enclosed_eoss[i], CASRN=CASs[i]) for i in range(N)]
        Vmg_STPs = [VolumeGases[i].TP_dependent_property(298.15, 101325.0)
                   for i in range(N)]
        rhog_STPs = [1.0/V if V is not None else None for V in Vmg_STPs]
        rhog_STPs_mass = [1e-3*MW/V if V is not None else None for V, MW in zip(Vmg_STPs, MWs)]


        VolumeLiquids = [VolumeLiquid(MW=MWs[i], Tb=Tbs[i], Tc=Tcs[i],
                          Pc=Pcs[i], Vc=Vcs[i], Zc=Zcs[i], omega=omegas[i], dipole=dipoles[i],
                          Psat=VaporPressures[i], CASRN=CASs[i],
                          eos=enclosed_eoss[i], poly_fit=get_chemical_constants(CASs[i], 'VolumeLiquid'))
                         for i in range(N)]

        Vml_Tbs = [VolumeLiquids[i].T_dependent_property(Tbs[i]) if Tbs[i] is not None else None
                   for i in range(N)]
        Vml_Tms = [VolumeLiquids[i].T_dependent_property(Tms[i]) if Tms[i] is not None else None
                   for i in range(N)]
        Vml_STPs = [VolumeLiquids[i].T_dependent_property(298.15)
                   for i in range(N)]
        Vml_60Fs = [VolumeLiquids[i].T_dependent_property(288.7055555555555)
                   for i in range(N)]
        rhol_STPs = [1.0/V if V is not None else None for V in Vml_STPs]
        rhol_60Fs = [1.0/V if V is not None else None for V in Vml_60Fs]
        rhol_STPs_mass = [1e-3*MW/V if V is not None else None for V, MW in zip(Vml_STPs, MWs)]
        rhol_60Fs_mass = [1e-3*MW/V if V is not None else None for V, MW in zip(Vml_60Fs, MWs)]

        VolumeSolids = [VolumeSolid(CASRN=CASs[i], MW=MWs[i], Tt=Tts[i], Vml_Tt=Vml_Tms[i], poly_fit=get_chemical_constants(CASs[i], 'VolumeSolid')) for i in range(N)]
        Vms_Tms = [VolumeSolids[i].T_dependent_property(Tms[i]) if Tms[i] is not None else None for i in range(N)]
        rhos_Tms = [1.0/V if V is not None else None for V in Vms_Tms]
        rhos_Tms_mass = [1e-3*MW/V if V is not None else None for V, MW in zip(Vms_Tms, MWs)]

        Hfgs = [Hfg(CAS) for CAS in CASs]
        Hfgs_mass = [Hf*1000.0/MW if Hf is not None else None for Hf, MW in zip(Hfgs, MWs)]

        Hfls = [Hfl(CAS) for CAS in CASs]
        Hfls_mass = [Hf*1000.0/MW if Hf is not None else None for Hf, MW in zip(Hfls, MWs)]

        Hfss = [Hfs(CAS) for CAS in CASs]
        Hfss_mass = [Hf*1000.0/MW if Hf is not None else None for Hf, MW in zip(Hfss, MWs)]

        S0gs = [S0g(CAS) for CAS in CASs]
        S0gs_mass = [S0*1000.0/MW if S0 is not None else None for S0, MW in zip(S0gs, MWs)]

        Hf_STPs, Hf_STPs_mass = [None]*N, [None]*N
        for i in range(N):
            if phase_STPs[i] == 'g':
                Hf_STPs[i] = Hfgs[i]
                Hf_STPs_mass[i] = Hfgs_mass[i]
            elif phase_STPs[i] == 'l':
                Hf_STPs[i] = Hfls[i]
                Hf_STPs_mass[i] = Hfls_mass[i]
            elif phase_STPs[i] == 's':
                Hf_STPs[i] = Hfss[i]
                Hf_STPs_mass[i] = Hfss_mass[i]

        # Compute Gfgs
        Gfgs = [None]*N
        for i in range(N):
            # Compute Gf and Gf(ig)
            dHfs_std = []
            S0_abs_elements = []
            coeffs_elements = []
            for atom, count in atomss[i].items():
                try:
                    ele = periodic_table[atom]
                    H0, S0 = ele.Hf, ele.S0
                    if ele.number in homonuclear_elements:
                        H0, S0 = 0.5*H0, 0.5*S0
                except KeyError:
                    H0, S0 = None, None # D, T
                dHfs_std.append(H0)
                S0_abs_elements.append(S0)
                coeffs_elements.append(count)

            elemental_reaction_data = (dHfs_std, S0_abs_elements, coeffs_elements)
            try:
                Gfgs[i] = Gibbs_formation(Hfgs[i], S0gs[i], dHfs_std, S0_abs_elements, coeffs_elements)
            except:
                pass
        Gfgs_mass = [Gf*1000.0/MW if Gf is not None else None for Gf, MW in zip(Gfgs, MWs)]

        Sfgs = [(Hfgs[i] - Gfgs[i])*(1.0/298.15) if (Hfgs[i] is not None and Gfgs[i] is not None) else None
                for i in range(N)]
        Sfgs_mass = [Sf*1000.0/MW if Sf is not None else None for Sf, MW in zip(Sfgs, MWs)]



        HeatCapacityGases = [HeatCapacityGas(CASRN=CASs[i], MW=MWs[i], similarity_variable=similarity_variables[i],
                                             poly_fit=get_chemical_constants(CASs[i], 'HeatCapacityGas')) for i in range(N)]

        HeatCapacitySolids = [HeatCapacitySolid(CASRN=CASs[i], MW=MWs[i], similarity_variable=similarity_variables[i],
                                                poly_fit=get_chemical_constants(CASs[i], 'HeatCapacitySolid')) for i in range(N)]
        HeatCapacityLiquids = [HeatCapacityLiquid(CASRN=CASs[i], MW=MWs[i], similarity_variable=similarity_variables[i], Tc=Tcs[i], omega=omegas[i],
                                                  Cpgm=HeatCapacityGases[i],
                                                  poly_fit=get_chemical_constants(CASs[i], 'HeatCapacityLiquid')) for i in range(N)]


        EnthalpySublimations = [EnthalpySublimation(CASRN=CASs[i], Tm=Tms[i], Tt=Tts[i],
                                                       Cpg=HeatCapacityGases[i], Cps=HeatCapacitySolids[i],
                                                       Hvap=EnthalpyVaporizations[i], poly_fit=get_chemical_constants(CASs[i], 'EnthalpySublimation'))
                                for i in range(N)]



        Hsub_Tts = [EnthalpySublimations[i](Tts[i]) if Tts[i] is not None else None
                           for i in range(N)]
        Hsub_Tts_mass = [Hsub*1000.0/MW if Hsub is not None else None for Hsub, MW in zip(Hsub_Tts, MWs)]


        combustion_stoichiometries = [combustion_stoichiometry(atoms) for atoms in atomss]
        Hcs = [None]*N
        for i in range(N):
            try:
                Hcs[i] = HHV_stoichiometry(combustion_stoichiometries[i], Hf=Hf_STPs[i]) if Hf_STPs[i] is not None else None
            except:
                pass
        Hcs_mass = [Hc*1000.0/MW if Hc is not None else None for Hc, MW in zip(Hcs, MWs)]
        Hcs_lower = [LHV_from_HHV(Hcs[i], combustion_stoichiometries[i].get('H2O', 0.0)) if Hcs[i] is not None else None
                     for i in range(N)]
        Hcs_lower_mass = [Hc*1000.0/MW if Hc is not None else None for Hc, MW in zip(Hcs_lower, MWs)]

        Tflashs = [T_flash(CAS) for CAS in CASs]
        Tautoignitions = [T_autoignition(CAS) for CAS in CASs]
        LFLs = [LFL(Hc=Hcs[i], atoms=atomss[i], CASRN=CASs[i]) for i in range(N)]
        UFLs = [UFL(Hc=Hcs[i], atoms=atomss[i], CASRN=CASs[i]) for i in range(N)]

        TWAs = [TWA(CAS) for CAS in CASs]
        STELs = [STEL(CAS) for CAS in CASs]
        Skins = [Skin(CAS) for CAS in CASs]
        Ceilings = [Ceiling(CAS) for CAS in CASs]
        Carcinogens = [Carcinogen(CAS) for CAS in CASs]


        # Environmental
        GWPs = [GWP(CAS) for CAS in CASs]
        ODPs = [ODP(CAS) for CAS in CASs]
        logPs = [logP(CAS) for CAS in CASs]

        # Analytical
        RIs, RI_Ts = [None]*N, [None]*N
        for i in range(N):
            RIs[i], RI_Ts[i] = RI(CASs[i])

        conductivities, conductivity_Ts = [None]*N, [None]*N
        for i in range(N):
            conductivities[i], conductivity_Ts[i] = conductivity(CASs[i])


        Stockmayers = [Stockmayer(Tm=Tms[i], Tb=Tbs[i], Tc=Tcs[i], Zc=Zcs[i], omega=omegas[i], CASRN=CASs[i]) for i in range(N)]
        molecular_diameters = [molecular_diameter(Tc=Tcs[i], Pc=Pcs[i], Vc=Vcs[i], Zc=Zcs[i], omega=omegas[i],
                                                  Vm=Vml_Tms[i], Vb=Vml_Tbs[i], CASRN=CASs[i]) for i in range(N)]

        load_group_assignments_DDBST()

        UNIFAC_groups = [DDBST_UNIFAC_assignments.get(InChI_Keys[i], None) for i in range(N)]
        UNIFAC_Dortmund_groups = [DDBST_MODIFIED_UNIFAC_assignments.get(InChI_Keys[i], None) for i in range(N)]
        PSRK_groups = [DDBST_PSRK_assignments.get(InChI_Keys[i], None) for i in range(N)]

        UNIFAC_Rs, UNIFAC_Qs = [None]*N, [None]*N
        for i in range(N):
            groups = UNIFAC_groups[i]
            if groups is not None:
                UNIFAC_Rs[i], UNIFAC_Qs[i] = UNIFAC_RQ(groups)

        solubility_parameters = [solubility_parameter(T=298.15, Hvapm=Hvap_298s[i], Vml=Vml_STPs[i]) if (Hvap_298s[i] is not None and  Vml_STPs[i] is not None) else None
                                 for i in range(N)]

        Van_der_Waals_volumes = [Van_der_Waals_volume(UNIFAC_Rs[i]) if UNIFAC_Rs[i] is not None else None for i in range(N)]
        Van_der_Waals_areas = [Van_der_Waals_area(UNIFAC_Qs[i]) if UNIFAC_Qs[i] is not None else None for i in range(N)]

        SurfaceTensions = [SurfaceTension(CASRN=CASs[i], MW=MWs[i], Tb=Tbs[i], Tc=Tcs[i], Pc=Pcs[i], Vc=Vcs[i], Zc=Zcs[i],
                          omega=omegas[i], StielPolar=StielPolars[i], Hvap_Tb=Hvap_Tbs[i], Vml=VolumeLiquids[i],
                          Cpl=HeatCapacityLiquids[i], poly_fit=get_chemical_constants(CASs[i], 'SurfaceTension'))
                                             for i in range(N)]

        sigma_STPs = [SurfaceTensions[i].T_dependent_property(298.15) for i in range(N)]
        sigma_Tbs = [SurfaceTensions[i].T_dependent_property(Tbs[i]) if Tbs[i] is not None else None for i in range(N)]
        sigma_Tms = [SurfaceTensions[i].T_dependent_property(Tms[i]) if Tms[i] is not None else None for i in range(N)]

        Parachors = [None]*N
        for i in range(N):
            try:
                Parachors[i] = Parachor(sigma=sigma_STPs[i], MW=MWs[i], rhol=rhol_STPs_mass[i], rhog=rhog_STPs_mass[i])
            except:
                pass

        atom_fractionss = [atom_fractions(atomss[i]) for i in range(N)]


        economic_statuses = [economic_status(CASs[i], method='Combined') for i in range(N)]
        legal_statuses = [legal_status(CASs[i], method='COMBINED') for i in range(N)]

        GWPs = [GWP(CASRN=CASs[i]) for i in range(N)]
        ODPs = [ODP(CASRN=CASs[i]) for i in range(N)]
        logPs = [logP(CASRN=CASs[i]) for i in range(N)]


        constants = ChemicalConstantsPackage(CASs=CASs, names=names, MWs=MWs, Tms=Tms,
                Tbs=Tbs, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas, Zcs=Zcs,
                rhocs=rhocs, rhocs_mass=rhocs_mass, Hfus_Tms=Hfus_Tms,
                Hfus_Tms_mass=Hfus_Tms_mass, Hvap_Tbs=Hvap_Tbs,
                Hvap_Tbs_mass=Hvap_Tbs_mass,

                 Vml_STPs=Vml_STPs, rhol_STPs=rhol_STPs, rhol_STPs_mass=rhol_STPs_mass,
                 Vml_60Fs=Vml_60Fs, rhol_60Fs=rhol_60Fs, rhol_60Fs_mass=rhol_60Fs_mass,
                 Vmg_STPs=Vmg_STPs, rhog_STPs=rhog_STPs, rhog_STPs_mass=rhog_STPs_mass,

                 Hfgs=Hfgs, Hfgs_mass=Hfgs_mass, Gfgs=Gfgs, Gfgs_mass=Gfgs_mass,
                 Sfgs=Sfgs, Sfgs_mass=Sfgs_mass, S0gs=S0gs, S0gs_mass=S0gs_mass,
                 Hf_STPs=Hf_STPs, Hf_STPs_mass=Hf_STPs_mass,

                 Tts=Tts, Pts=Pts, Hsub_Tts=Hsub_Tts, Hsub_Tts_mass=Hsub_Tts_mass,
                 Hcs=Hcs, Hcs_mass=Hcs_mass, Hcs_lower=Hcs_lower, Hcs_lower_mass=Hcs_lower_mass,
                 Tflashs=Tflashs, Tautoignitions=Tautoignitions, LFLs=LFLs, UFLs=UFLs,
                 TWAs=TWAs, STELs=STELs, Ceilings=Ceilings, Skins=Skins,
                 Carcinogens=Carcinogens,
                 Psat_298s=Psat_298s, Hvap_298s=Hvap_298s, Hvap_298s_mass=Hvap_298s_mass,
                 Vml_Tms=Vml_Tms, Vms_Tms=Vms_Tms, rhos_Tms=rhos_Tms, rhos_Tms_mass=rhos_Tms_mass,

                 sigma_STPs=sigma_STPs, sigma_Tbs=sigma_Tbs, sigma_Tms=sigma_Tms,
                 RIs=RIs, RI_Ts=RI_Ts, conductivities=conductivities,
                 conductivity_Ts=conductivity_Ts,
                 charges=charges, dipoles=dipoles, Stockmayers=Stockmayers,
                 molecular_diameters=molecular_diameters, Van_der_Waals_volumes=Van_der_Waals_volumes,
                 Van_der_Waals_areas=Van_der_Waals_areas, Parachors=Parachors, StielPolars=StielPolars,
                 atomss=atomss, atom_fractions=atom_fractionss,
                 similarity_variables=similarity_variables, phase_STPs=phase_STPs,
                 UNIFAC_Rs=UNIFAC_Rs, UNIFAC_Qs=UNIFAC_Qs, solubility_parameters=solubility_parameters,
               UNIFAC_groups=UNIFAC_groups, UNIFAC_Dortmund_groups=UNIFAC_Dortmund_groups,
               PSRK_groups=PSRK_groups,
                 # Other identifiers
                 PubChems=PubChems, formulas=formulas, smiless=smiless, InChIs=InChIs,
                 InChI_Keys=InChI_Keys,

                 economic_statuses=economic_statuses, legal_statuses=legal_statuses,
                 GWPs=GWPs, ODPs=ODPs, logPs=logPs,
                )

        if not correlations:
            return constants

        SublimationPressures = [SublimationPressure(CASRN=CASs[i], Tt=Tts[i], Pt=Pts[i], Hsub_t=Hsub_Tts[i],
                                                    poly_fit=get_chemical_constants(CASs[i], 'SublimationPressure'))
                                                    for i in range(N)]

        PermittivityLiquids = [PermittivityLiquid(CASRN=CASs[i], poly_fit=get_chemical_constants(CASs[i], 'PermittivityLiquid')) for i in range(N)]

        ViscosityLiquids = [ViscosityLiquid(CASRN=CASs[i], MW=MWs[i], Tm=Tms[i], Tc=Tcs[i], Pc=Pcs[i], Vc=Vcs[i], omega=omegas[i], Psat=VaporPressures[i],
                                            Vml=VolumeLiquids[i], poly_fit=get_chemical_constants(CASs[i], 'ViscosityLiquid')) for i in range(N)]

        ViscosityGases = [ViscosityGas(CASRN=CASs[i], MW=MWs[i], Tc=Tcs[i], Pc=Pcs[i], Zc=Zcs[i], dipole=dipoles[i],
                                       Vmg=VolumeGases[i], poly_fit=get_chemical_constants(CASs[i], 'ViscosityGas')) for i in range(N)]

        ThermalConductivityLiquids = [ThermalConductivityLiquid(CASRN=CASs[i], MW=MWs[i], Tm=Tms[i], Tb=Tbs[i], Tc=Tcs[i], Pc=Pcs[i],
                                                                omega=omegas[i], Hfus=Hfus_Tms[i], poly_fit=get_chemical_constants(CASs[i], 'ThermalConductivityLiquid'))
                                    for i in range(N)]

        ThermalConductivityGases =[ThermalConductivityGas(CASRN=CASs[i], MW=MWs[i], Tb=Tbs[i], Tc=Tcs[i], Pc=Pcs[i], Vc=Vcs[i],
                                                          Zc=Zcs[i], omega=omegas[i], dipole=dipoles[i], Vmg=VolumeGases[i],
                                                          Cpgm=HeatCapacityGases[i], mug=ViscosityGases[i],
                                                          poly_fit=get_chemical_constants(CASs[i], 'ThermalConductivityGas'))
                                                          for i in range(N)]
        properties = PropertyCorrelationsPackage(constants, VaporPressures=VaporPressures, SublimationPressures=SublimationPressures,
                                                 VolumeGases=VolumeGases, VolumeLiquids=VolumeLiquids, VolumeSolids=VolumeSolids,
                                                 HeatCapacityGases=HeatCapacityGases, HeatCapacityLiquids=HeatCapacityLiquids, HeatCapacitySolids=HeatCapacitySolids,
                                                 ViscosityGases=ViscosityGases, ViscosityLiquids=ViscosityLiquids,
                                                 ThermalConductivityGases=ThermalConductivityGases, ThermalConductivityLiquids=ThermalConductivityLiquids,
                                                 EnthalpyVaporizations=EnthalpyVaporizations, EnthalpySublimations=EnthalpySublimations,
                                                 SurfaceTensions=SurfaceTensions, PermittivityLiquids=PermittivityLiquids)
        return constants, properties


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
                 Vmg_STPs=None, rhog_STPs=None, rhog_STPs_mass=None,
                 # Reaction (ideal gas)
                 Hfgs=None, Hfgs_mass=None, Gfgs=None, Gfgs_mass=None,
                 Sfgs=None, Sfgs_mass=None, S0gs=None, S0gs_mass=None,
                 Hf_STPs=None, Hf_STPs_mass=None,

                 # Triple point
                 Tts=None, Pts=None, Hsub_Tts=None, Hsub_Tts_mass=None,
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
                 Vml_Tms=None, rhos_Tms=None, Vms_Tms=None, rhos_Tms_mass=None,

                 # Analytical
                 sigma_STPs=None, sigma_Tbs=None, sigma_Tms=None,
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
        if Hsub_Tts is None: Hsub_Tts = empty_list
        if Hsub_Tts_mass is None: Hsub_Tts_mass = empty_list
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
        if rhos_Tms_mass is None: rhos_Tms_mass = empty_list
        if Vmg_STPs is None: Vmg_STPs = empty_list
        if rhog_STPs is None: rhog_STPs = empty_list
        if rhog_STPs_mass is None: rhog_STPs_mass = empty_list
        if sigma_STPs is None: sigma_STPs = empty_list
        if sigma_Tbs is None: sigma_Tbs = empty_list
        if sigma_Tms is None: sigma_Tms = empty_list
        if Hf_STPs is None: Hf_STPs = empty_list
        if Hf_STPs_mass is None: Hf_STPs_mass = empty_list

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
        self.Hsub_Tts = Hsub_Tts
        self.Hsub_Tts_mass = Hsub_Tts_mass
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
        self.rhos_Tms_mass = rhos_Tms_mass
        self.Vmg_STPs = Vmg_STPs
        self.rhog_STPs = rhog_STPs
        self.rhog_STPs_mass = rhog_STPs_mass
        self.sigma_STPs = sigma_STPs
        self.sigma_Tbs = sigma_Tbs
        self.sigma_Tms = sigma_Tms
        self.Hf_STPs = Hf_STPs
        self.Hf_STPs_mass = Hf_STPs_mass


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
'Hsub_Tts': ("list[float]", "Heats of sublimation for each component at their respective triple points", "[J/mol]", None),
'Hsub_Tts_mass': ("list[float]", "Heats of sublimation for each component at their respective triple points", "[J/kg]", None),
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
'solubility_parameters': ("list[float]", "Solubility parameters for each component at 298.15 K", "[Pa^0.5]", None),
'similarity_variables': ("list[float]", "Similarity variables for each component", "[mol/g]", None),
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
'Vml_Tms': ("list[float]", "Liquid molar volumes for each component at their respective melting points", "[m^3/mol]", None),
'Vms_Tms': ("list[float]", "Solid molar volumes for each component at their respective melting points", "[m^3/mol]", None),
'Vml_60Fs': ("list[float]", "Liquid molar volumes for each component at 60 °F", "[m^3/mol]", None),
'rhos_Tms': ("list[float]", "Solid molar densities for each component at their respective melting points", "[mol/m^3]", None),
'rhol_60Fs': ("list[float]", "Liquid molar densities for each component at 60 °F", "[mol/m^3]", None),
'rhol_60Fs_mass': ("list[float]", "Liquid mass densities for each component at 60 °F", "[kg/m^3]", None),
'rhos_Tms_mass': ("list[float]", "Solid mass densities for each component at their melting point", "[kg/m^3]", None),
'Zcs': ("list[float]", "Critical compressibilities for each component", "[-]", None),
'n_atoms': ("int", "Number of total atoms in a collection of 1 molecule of each species", "[-]", None),
'water_index': ("int", "Index of water in the package", "[-]", None),
'Vmg_STPs': ("list[float]", "Gas molar volumes for each component at STP; metastable if normally another state", "[m^3/mol]", None),
'rhog_STPs': ("list[float]", "Molar gas densities at STP for each component; metastable if normally another state", "[mol/m^3]", None),
'rhog_STPs_mass': ("list[float]", "Gas densities at STP for each component; metastable if normally another state", "[kg/m^3]", None),
'sigma_STPs': ("list[float]", "Liquid-air surface tensions at 298.15 K and the higher of 101325 Pa or the saturation pressure", "[N/m]", None),
'sigma_Tms': ("list[float]", "Liquid-air surface tensions at the melting point and 101325 Pa", "[N/m]", None),
'sigma_Tbs': ("list[float]", "Liquid-air surface tensions at the normal boiling point and 101325 Pa", "[N/m]", None),
'Hf_STPs': ("list[float]", "Standard state molar enthalpies of formation for each component", "[J/mol]", None),
'Hf_STPs_mass': ("list[float]", "Standard state mass enthalpies of formation for each component", "[J/kg]", None),
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

properties_to_classes = {'VaporPressures': VaporPressure,
'VolumeLiquids': VolumeLiquid,
'VolumeGases': VolumeGas,
'VolumeSolids': VolumeSolid,
'HeatCapacityGases': HeatCapacityGas,
'HeatCapacitySolids': HeatCapacitySolid,
'HeatCapacityLiquids': HeatCapacityLiquid,
'EnthalpyVaporizations': EnthalpyVaporization,
'EnthalpySublimations': EnthalpySublimation,
'SublimationPressures': SublimationPressure,
'PermittivityLiquids': PermittivityLiquid,
'ViscosityLiquids': ViscosityLiquid,
'ViscosityGases': ViscosityGas,
'ThermalConductivityLiquids': ThermalConductivityLiquid,
'ThermalConductivityGases': ThermalConductivityGas,
'SurfaceTensions': SurfaceTension}

classes_to_properties = {v:k for k, v in properties_to_classes.items()}

mix_properties_to_classes = {'VolumeGasMixture': VolumeGasMixture,
                            'VolumeLiquidMixture': VolumeLiquidMixture,
                            'VolumeSolidMixture': VolumeSolidMixture,
                            'HeatCapacityGasMixture': HeatCapacityGasMixture,
                            'HeatCapacityLiquidMixture': HeatCapacityLiquidMixture,
                            'HeatCapacitySolidMixture': HeatCapacitySolidMixture,
                            'ViscosityGasMixture': ViscosityGasMixture,
                            'ViscosityLiquidMixture': ViscosityLiquidMixture,
                            'ThermalConductivityGasMixture': ThermalConductivityGasMixture,
                            'ThermalConductivityLiquidMixture': ThermalConductivityLiquidMixture,
                            'SurfaceTensionMixture': SurfaceTensionMixture}

class PropertyCorrelationsPackage(object):
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
    PermittivityLiquids : list[:obj:`thermo.permittivity.PermittivityLiquid`], optional
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
    >>> correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True, HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])), HeatCapacityGas(poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998]))])

    Create a package from various data files, creating all property objects:

    >>> correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=False)

    '''
    pure_correlations = ('VaporPressures', 'VolumeLiquids', 'VolumeGases',
                         'VolumeSolids', 'HeatCapacityGases', 'HeatCapacitySolids',
                         'HeatCapacityLiquids', 'EnthalpyVaporizations',
                         'EnthalpySublimations', 'SublimationPressures',
                         'PermittivityLiquids', 'ViscosityLiquids', 'ViscosityGases',
                         'ThermalConductivityLiquids', 'ThermalConductivityGases',
                         'SurfaceTensions')
    mixture_correlations = ('VolumeGasMixture', 'VolumeLiquidMixture', 'VolumeSolidMixture',
               'HeatCapacityGasMixture', 'HeatCapacityLiquidMixture',
               'HeatCapacitySolidMixture', 'ViscosityGasMixture',
               'ViscosityLiquidMixture', 'ThermalConductivityGasMixture',
               'ThermalConductivityLiquidMixture', 'SurfaceTensionMixture')

    correlations = pure_correlations + mixture_correlations
#    __slots__ = correlations + ('constants', 'skip_missing')

    __full_path__ = "%s.%s" %(__module__, __qualname__)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        hashes = []
        for k in self.correlations:
            hashes.append(hash_any_primitive(getattr(self, k)))
        return hash_any_primitive(hashes)


    def as_json(self):
        r'''Method to create a JSON friendly serialization of the chemical
        properties package which can be stored, and reloaded later.

        Returns
        -------
        json_repr : dict
            JSON-friendly representation, [-]

        Notes
        -----

        Examples
        --------
        '''
        d = self.__dict__.copy()
        d["py/object"] = self.__full_path__
        N = self.constants.N

        props_to_store = []
        for prop_name in self.pure_correlations:
            l = getattr(self, prop_name)
            if l is None:
                props_to_store.append((prop_name, l))
            else:
                props = []
                for o in l:
                    s = o.as_json(references=0)
                    # Remove references to other properties
                    for ref in o.pure_references:
                        s[ref] = None
                    props.append(s)

                props_to_store.append((prop_name, props))
            del d[prop_name]

        mix_props_to_store = []
        for prop_name in self.mixture_correlations:
            l = getattr(self, prop_name)
            if l is None:
                mix_props_to_store.append((prop_name, l))
            else:
                s = l.as_json(references=0)
                for ref in l.pure_references:
                    s[ref] = [None]*N
                mix_props_to_store.append((prop_name, s))
            del d[prop_name]



        d['constants'] = self.constants.as_json()
        d['mixture_properties'] = mix_props_to_store
        d['pure_properties'] = props_to_store
        d['json_version'] = 1
        d['skip_missing'] = self.skip_missing
        return d

    @classmethod
    def from_json(cls, json_repr):
        r'''Method to create a :obj:`PropertyCorrelationsPackage` from a JSON
        serialization of another :obj:`PropertyCorrelationsPackage`.

        Parameters
        ----------
        json_repr : dict
            JSON-friendly representation, [-]

        Returns
        -------
        correlations : :obj:`PropertyCorrelationsPackage`
            Newly created object from the json serialization, [-]

        Notes
        -----
        It is important that the input string be in the same format as that
        created by :obj:`PropertyCorrelationsPackage.as_json`.

        Examples
        --------
        '''
        d = json_repr
        new = cls.__new__(cls)
        d2 = new.__dict__
        new.constants = ChemicalConstantsPackage.from_json(d['constants'])
        N = new.constants.N
        new.skip_missing = d['skip_missing']

        for prop, value in d['pure_properties']:
            if value is None:
                d2[prop] = value
            else:
                callable = properties_to_classes[prop].from_json
                d2[prop] = [callable(v) for v in value]


        # Set the links back to other objects
        for prop_name, prop_cls in properties_to_classes.items():
            l = d2[prop_name]
            if l:
                for ref_name, ref_cls in zip(prop_cls.pure_references, prop_cls.pure_reference_types):
                    real_ref_list = getattr(new, classes_to_properties[ref_cls])
                    if real_ref_list:
                        for i in range(N):
                            setattr(l[i], ref_name, real_ref_list[i])


        for prop, value in d['mixture_properties']:
            if value is None:
                d2[prop] = value
            else:
                mix_prop = mix_properties_to_classes[prop].from_json(value)
                for k, sub_cls in zip(mix_prop.pure_references, mix_prop.pure_reference_types):
                    real_ref_list = getattr(new, classes_to_properties[sub_cls])
                    setattr(mix_prop, k, real_ref_list)
                d2[prop] = mix_prop


        return new

    def subset(self, idxs):
        r'''Method to construct a new PropertyCorrelationsPackage that removes
        all components not specified in the `idxs` argument.

        Parameters
        ----------
        idxs : list[int] or Slice or None
            Indexes of components that should be included; if None, all
            components will be included , [-]

        Returns
        -------
        subset_correlations : PropertyCorrelationsPackage
            Object with components, [-]

        Notes
        -----
        '''
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
        return PropertyCorrelationsPackage(skip_missing=self.skip_missing, **new)


    def __init__(self, constants, VaporPressures=None, SublimationPressures=None,
                 VolumeGases=None, VolumeLiquids=None, VolumeSolids=None,
                 HeatCapacityGases=None, HeatCapacityLiquids=None, HeatCapacitySolids=None,
                 ViscosityGases=None, ViscosityLiquids=None,
                 ThermalConductivityGases=None, ThermalConductivityLiquids=None,
                 EnthalpyVaporizations=None, EnthalpySublimations=None,
                 SurfaceTensions=None, PermittivityLiquids=None,

                 VolumeGasMixtureObj=None, VolumeLiquidMixtureObj=None, VolumeSolidMixtureObj=None,
                 HeatCapacityGasMixtureObj=None, HeatCapacityLiquidMixtureObj=None, HeatCapacitySolidMixtureObj=None,
                 ViscosityGasMixtureObj=None, ViscosityLiquidMixtureObj=None,
                 ThermalConductivityGasMixtureObj=None, ThermalConductivityLiquidMixtureObj=None,
                 SurfaceTensionMixtureObj=None, skip_missing=False,
                 ):
        self.constants = constants
        self.skip_missing = skip_missing
        cmps = constants.cmps

        if VaporPressures is None and not skip_missing:
            VaporPressures = [VaporPressure(Tb=constants.Tbs[i], Tc=constants.Tcs[i], Pc=constants.Pcs[i],
                                            omega=constants.omegas[i], CASRN=constants.CASs[i],
                                            poly_fit=get_chemical_constants(constants.CASs[i], 'VaporPressure'))
                              for i in cmps]

        if VolumeLiquids is None and not skip_missing:
            VolumeLiquids = [VolumeLiquid(MW=constants.MWs[i], Tb=constants.Tbs[i], Tc=constants.Tcs[i],
                              Pc=constants.Pcs[i], Vc=constants.Vcs[i], Zc=constants.Zcs[i], omega=constants.omegas[i],
                              dipole=constants.dipoles[i],
                              Psat=VaporPressures[i],
                              poly_fit=get_chemical_constants(constants.CASs[i], 'VolumeLiquid'),
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
                                                 poly_fit=get_chemical_constants(constants.CASs[i], 'HeatCapacityGas'))
                              for i in cmps]

        if HeatCapacitySolids is None and not skip_missing:
            HeatCapacitySolids = [HeatCapacitySolid(MW=constants.MWs[i], similarity_variable=constants.similarity_variables[i],
                                                    CASRN=constants.CASs[i], poly_fit=get_chemical_constants(constants.CASs[i], 'HeatCapacitySolid'))
                              for i in cmps]

        if HeatCapacityLiquids is None and not skip_missing:
            HeatCapacityLiquids = [HeatCapacityLiquid(CASRN=constants.CASs[i], MW=constants.MWs[i],
                                                      similarity_variable=constants.similarity_variables[i],
                                                      Tc=constants.Tcs[i], omega=constants.omegas[i],
                                                      Cpgm=HeatCapacityGases[i], poly_fit=get_chemical_constants(constants.CASs[i], 'HeatCapacityLiquid'))
                              for i in cmps]

        if EnthalpyVaporizations is None and not skip_missing:
            EnthalpyVaporizations = [EnthalpyVaporization(CASRN=constants.CASs[i], Tb=constants.Tbs[i],
                                                          Tc=constants.Tcs[i], Pc=constants.Pcs[i], omega=constants.omegas[i],
                                                          similarity_variable=constants.similarity_variables[i],
                                                          poly_fit=get_chemical_constants(constants.CASs[i], 'EnthalpyVaporization'))
                              for i in cmps]

        if EnthalpySublimations is None and not skip_missing:
            EnthalpySublimations = [EnthalpySublimation(CASRN=constants.CASs[i], Tm=constants.Tms[i], Tt=constants.Tts[i],
                                                       Cpg=HeatCapacityGases[i], Cps=HeatCapacitySolids[i],
                                                       Hvap=EnthalpyVaporizations[i])
                                    for i in cmps]

        if SublimationPressures is None and not skip_missing:
            SublimationPressures = [SublimationPressure(CASRN=constants.CASs[i], Tt=constants.Tts[i], Pt=constants.Pts[i],
                                                        Hsub_t=constants.Hsub_Tts[i])
                                    for i in cmps]

        if PermittivityLiquids is None and not skip_missing:
            PermittivityLiquids = [PermittivityLiquid(CASRN=constants.CASs[i]) for i in cmps]

        # missing -  ThermalConductivityGas, SurfaceTension
        if ViscosityLiquids is None and not skip_missing:
            ViscosityLiquids = [ViscosityLiquid(CASRN=constants.CASs[i], MW=constants.MWs[i], Tm=constants.Tms[i],
                                                Tc=constants.Tcs[i], Pc=constants.Pcs[i], Vc=constants.Vcs[i],
                                                omega=constants.omegas[i], Psat=VaporPressures[i],
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
                                                               Cpgm=HeatCapacityGases[i])
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
        self.PermittivityLiquids = PermittivityLiquids
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

    def as_poly_fit(self, props=None):
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
                        s += '%s=%s,\n' %(prop, self.as_poly_fit(prop_attr))
                    except Exception as e:
                        print(e, prop)

            s += ')'
            return s

        s = '['
        for obj in props:
            s += (obj.as_poly_fit() + ',\n')
        s += ']'
        return s


# Values except for omega from IAPWS; heat capacity isn't official.
iapws_constants = ChemicalConstantsPackage(CASs=['7732-18-5'], MWs=[18.015268], omegas=[0.344],
                                           Pcs=[22064000.0], Tcs=[647.096])
''':obj:`ChemicalConstantsPackage` : Object intended to hold the IAPWS-95 water constants
for use with the :obj:`thermo.phases.IAPWS95` phase object.
'''

iapws_correlations = PropertyCorrelationsPackage(constants=iapws_constants, skip_missing=True,
                                                 HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18,
                                                                            4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759]))])
''':obj:`PropertyCorrelationsPackage`: IAPWS correlations and properties, [-]'''

lemmon2000_constants = ChemicalConstantsPackage(CASs=['132259-10-0'], MWs=[28.9586], omegas=[0.0335],
                                           Pcs=[3.78502E6], Tcs=[132.6312])
''':obj:`ChemicalConstantsPackage` : Object intended to hold the Lemmon (2000) air constants
for use with the :obj:`thermo.phases.DryAirLemmon` phase object.
'''

# 20 coefficients gets a very good fit 1e-8 rtol
lemmon2000_correlations = PropertyCorrelationsPackage(constants=lemmon2000_constants, skip_missing=True,
                                                 HeatCapacityGases=[HeatCapacityGas(load_data=False, poly_fit=(132.6313, 2000.0, [-6.626125905505976e-57, 1.3834500819751098e-52, -1.33739947283832e-48, 7.939514061796618e-45, -3.2364291450043207e-41, 9.594346367764268e-38, -2.13653752371752e-34, 3.6389955418840433e-31, -4.779579487030328e-28, 4.842352128842408e-25, -3.7575972075674665e-22, 2.2015407920080106e-19, -9.545492841183412e-17, 3.0147537523176223e-14, -7.116946884523906e-12, 1.4112957512038422e-09, -2.416177742609162e-07, 3.041947869442721e-05, -0.0022420811935852042, 29.099089803167224]))])
''':obj:`PropertyCorrelationsPackage`: Lemmon (2000) air correlations and properties, [-]'''

