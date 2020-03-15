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

__all__ = ['ChemicalConstantsPackage', 'PropertyCorrelationPackage']

from thermo.chemical import Chemical, get_chemical_constants
from thermo.identifiers import *
from thermo.activity import identify_phase_mixture, Pbubble_mixture, Pdew_mixture
from thermo.critical import Tc_mixture, Pc_mixture, Vc_mixture
from thermo.thermal_conductivity import ThermalConductivityLiquid, ThermalConductivityGas, ThermalConductivityLiquidMixture, ThermalConductivityGasMixture
from thermo.volume import VolumeLiquidMixture, VolumeGasMixture, VolumeSolidMixture, VolumeLiquid, VolumeGas, VolumeSolid
from thermo.permittivity import *
from thermo.heat_capacity import HeatCapacitySolid, HeatCapacityGas, HeatCapacityLiquid, HeatCapacitySolidMixture, HeatCapacityGasMixture, HeatCapacityLiquidMixture
from thermo.interface import SurfaceTension, SurfaceTensionMixture
from thermo.viscosity import ViscosityLiquid, ViscosityGas, ViscosityLiquidMixture, ViscosityGasMixture
from thermo.utils import *
from thermo.vapor_pressure import VaporPressure, SublimationPressure
from thermo.phase_change import EnthalpyVaporization, EnthalpySublimation


CAS_H2O = '7732-18-5'


class ChemicalConstantsPackage(object):
    properties = ('atom_fractions', 'atomss', 'Carcinogens', 'CASs', 'Ceilings', 'charges',
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
                 )
    
    __slots__ = properties + ('N', 'cmps', 'water_index', 'n_atoms')
    non_vectors = ('atom_fractions',)
    
    def subset(self, idxs):
        is_slice = isinstance(idxs, slice)
        
        def atindexes(values):
            if is_slice:
                return values[idxs]
            return [values[i] for i in idxs]
        
        new = {}
        for p in self.properties:
            if hasattr(self, p) and getattr(self, p) is not None and p not in self.non_vectors:
                new[p] = atindexes(getattr(self, p))
        return ChemicalConstantsPackage(**new)
    
    def __repr__(self):
        return self.make_str()
    
    def make_str(self, delim=', ', properties=None):
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
                 RIs=None, conductivities=None,
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
    
        if atom_fractions is None: atom_fractions = [None]*N
        if atomss is None: atomss = [None]*N
        if Carcinogens is None: Carcinogens = [None]*N
        if CASs is None: CASs = [None]*N
        if Ceilings is None: Ceilings = [None]*N
        if charges is None: charges = [None]*N
        if conductivities is None: conductivities = [None]*N
        if dipoles is None: dipoles = [None]*N
        if economic_statuses is None: economic_statuses = [None]*N
        if formulas is None: formulas = [None]*N
        if Gfgs is None: Gfgs = [None]*N
        if Gfgs_mass is None: Gfgs_mass = [None]*N
        if GWPs is None: GWPs = [None]*N
        if Hcs is None: Hcs = [None]*N
        if Hcs_lower is None: Hcs_lower = [None]*N
        if Hcs_lower_mass is None: Hcs_lower_mass = [None]*N
        if Hcs_mass is None: Hcs_mass = [None]*N
        if Hfgs is None: Hfgs = [None]*N
        if Hfgs_mass is None: Hfgs_mass = [None]*N
        if Hfus_Tms is None: Hfus_Tms = [None]*N
        if Hfus_Tms_mass is None: Hfus_Tms_mass = [None]*N
        if Hsub_Tms is None: Hsub_Tms = [None]*N
        if Hsub_Tms_mass is None: Hsub_Tms_mass = [None]*N
        if Hvap_298s is None: Hvap_298s = [None]*N
        if Hvap_298s_mass is None: Hvap_298s_mass = [None]*N
        if Hvap_Tbs is None: Hvap_Tbs = [None]*N
        if Hvap_Tbs_mass is None: Hvap_Tbs_mass = [None]*N
        if InChI_Keys is None: InChI_Keys = [None]*N
        if InChIs is None: InChIs = [None]*N
        if legal_statuses is None: legal_statuses = [None]*N
        if LFLs is None: LFLs = [None]*N
        if logPs is None: logPs = [None]*N
        if molecular_diameters is None: molecular_diameters = [None]*N
        if names is None: names = [None]*N
        if ODPs is None: ODPs = [None]*N
        if omegas is None: omegas = [None]*N
        if Parachors is None: Parachors = [None]*N
        if Pcs is None: Pcs = [None]*N
        if phase_STPs is None: phase_STPs = [None]*N
        if Psat_298s is None: Psat_298s = [None]*N
        if PSRK_groups is None: PSRK_groups = [None]*N
        if Pts is None: Pts = [None]*N
        if PubChems is None: PubChems = [None]*N
        if rhocs is None: rhocs = [None]*N
        if rhocs_mass is None: rhocs_mass = [None]*N
        if rhol_STPs is None: rhol_STPs = [None]*N
        if rhol_STPs_mass is None: rhol_STPs_mass = [None]*N
        if RIs is None: RIs = [None]*N
        if S0gs is None: S0gs = [None]*N
        if S0gs_mass is None: S0gs_mass = [None]*N
        if Sfgs is None: Sfgs = [None]*N
        if Sfgs_mass is None: Sfgs_mass = [None]*N
        if similarity_variables is None: similarity_variables = [None]*N
        if Skins is None: Skins = [None]*N
        if smiless is None: smiless = [None]*N
        if STELs is None: STELs = [None]*N
        if StielPolars is None: StielPolars = [None]*N
        if Stockmayers is None: Stockmayers = [None]*N
        if solubility_parameters is None: solubility_parameters = [None]*N
        if Tautoignitions is None: Tautoignitions = [None]*N
        if Tbs is None: Tbs = [None]*N
        if Tcs is None: Tcs = [None]*N
        if Tflashs is None: Tflashs = [None]*N
        if Tms is None: Tms = [None]*N
        if Tts is None: Tts = [None]*N
        if TWAs is None: TWAs = [None]*N
        if UFLs is None: UFLs = [None]*N
        if UNIFAC_Dortmund_groups is None: UNIFAC_Dortmund_groups = [None]*N
        if UNIFAC_groups is None: UNIFAC_groups = [None]*N
        if UNIFAC_Rs is None: UNIFAC_Rs = [None]*N
        if UNIFAC_Qs is None: UNIFAC_Qs = [None]*N
        if Van_der_Waals_areas is None: Van_der_Waals_areas = [None]*N
        if Van_der_Waals_volumes is None: Van_der_Waals_volumes = [None]*N
        if Vcs is None: Vcs = [None]*N
        if Vml_STPs is None: Vml_STPs = [None]*N
        if Vml_Tms is None: Vml_Tms = [None]*N
        if rhos_Tms is None: rhos_Tms = [None]*N
        if Vms_Tms is None: Vms_Tms = [None]*N
        if Zcs is None: Zcs = [None]*N
        if Vml_60Fs is None: Vml_60Fs = [None]*N
        if rhol_60Fs is None: rhol_60Fs = [None]*N
        if rhol_60Fs_mass is None: rhol_60Fs_mass = [None]*N

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

        try:
            self.water_index = CASs.index(CAS_H2O)
        except ValueError:
            self.water_index = None
            
        try:
            self.n_atoms = [sum(i.values()) for i in atomss]
        except:
            self.n_atoms = None
        

class PropertyCorrelationPackage(object):
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

    def subset(self, idxs):
        is_slice = isinstance(idxs, slice)
        
        def atindexes(values):
            if is_slice:
                return values[idxs]
            return [values[i] for i in idxs]
        
        new = {'constants': self.constants.subset(idxs)}
        for p in self.pure_correlations:
            if hasattr(self, p) and getattr(self, p) is not None:
                new[p] = atindexes(getattr(self, p))
        return PropertyCorrelationPackage(**new)


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
                 SurfaceTensionMixtureObj=None,
                 ):
        self.constants = constants
        cmps = constants.cmps
        
        if VaporPressures is None:
            VaporPressures = [VaporPressure(Tb=constants.Tbs[i], Tc=constants.Tcs[i], Pc=constants.Pcs[i],
                                            omega=constants.omegas[i], CASRN=constants.CASs[i],
                                            best_fit=get_chemical_constants(constants.CASs[i], 'VaporPressure'))
                              for i in cmps]
            
        if VolumeLiquids is None:
            VolumeLiquids = [VolumeLiquid(MW=constants.MWs[i], Tb=constants.Tbs[i], Tc=constants.Tcs[i],
                              Pc=constants.Pcs[i], Vc=constants.Vcs[i], Zc=constants.Zcs[i], omega=constants.omegas[i],
                              dipole=constants.dipoles[i],
                              Psat=VaporPressures[i],
                              best_fit=get_chemical_constants(constants.CASs[i], 'VolumeLiquid'),
                              eos=None, CASRN=constants.CASs[i])
                              for i in cmps]
            
        if VolumeGases is None:
            VolumeGases = [VolumeGas(MW=constants.MWs[i], Tc=constants.Tcs[i], Pc=constants.Pcs[i],
                                   omega=constants.omegas[i], dipole=constants.dipoles[i],
                                   eos=None, CASRN=constants.CASs[i])
                              for i in cmps]
            
        if VolumeSolids is None:
            VolumeSolids = [VolumeSolid(CASRN=constants.CASs[i], MW=constants.MWs[i],
                                        Tt=constants.Tts[i], Vml_Tt=constants.Vml_Tms[i])
                              for i in cmps]
        
        if HeatCapacityGases is None:
            HeatCapacityGases = [HeatCapacityGas(CASRN=constants.CASs[i], MW=constants.MWs[i],
                                                 similarity_variable=constants.similarity_variables[i],
                                                 best_fit=get_chemical_constants(constants.CASs[i], 'HeatCapacityGas'))
                              for i in cmps]
            
        if HeatCapacitySolids is None:
            HeatCapacitySolids = [HeatCapacitySolid(MW=constants.MWs[i], similarity_variable=constants.similarity_variables[i],
                                                    CASRN=constants.CASs[i], best_fit=get_chemical_constants(constants.CASs[i], 'HeatCapacitySolid'))
                              for i in cmps]

        if HeatCapacityLiquids is None:
            HeatCapacityLiquids = [HeatCapacityLiquid(CASRN=constants.CASs[i], MW=constants.MWs[i], 
                                                      similarity_variable=constants.similarity_variables[i],
                                                      Tc=constants.Tcs[i], omega=constants.omegas[i],
                                                      Cpgm=HeatCapacityGases[i], best_fit=get_chemical_constants(constants.CASs[i], 'HeatCapacityLiquid'))
                              for i in cmps]

        if EnthalpyVaporizations is None:
            EnthalpyVaporizations = [EnthalpyVaporization(CASRN=constants.CASs[i], Tb=constants.Tbs[i],
                                                          Tc=constants.Tcs[i], Pc=constants.Pcs[i], omega=constants.omegas[i],
                                                          similarity_variable=constants.similarity_variables[i],
                                                          best_fit=get_chemical_constants(constants.CASs[i], 'EnthalpyVaporization'))
                              for i in cmps]

        if EnthalpySublimations is None:
            EnthalpySublimations = [EnthalpySublimation(CASRN=constants.CASs[i], Tm=constants.Tms[i], Tt=constants.Tts[i], 
                                                       Cpg=HeatCapacityGases[i], Cps=HeatCapacitySolids[i],
                                                       Hvap=EnthalpyVaporizations[i])
                                    for i in cmps]
            
        if SublimationPressures is None:
            SublimationPressures = [SublimationPressure(CASRN=constants.CASs[i], Tt=constants.Tts[i], Pt=constants.Pts[i],
                                                        Hsub_t=constants.Hsub_Tms[i])
                                    for i in cmps]
        
        if Permittivities is None:
            Permittivities = [Permittivity(CASRN=constants.CASs[i]) for i in cmps]
            
        # missing -  ThermalConductivityGas, SurfaceTension
        if ViscosityLiquids is None:
            ViscosityLiquids = [ViscosityLiquid(CASRN=constants.CASs[i], MW=constants.MWs[i], Tm=constants.Tms[i],
                                                Tc=constants.Tcs[i], Pc=constants.Pcs[i], Vc=constants.Vcs[i],
                                                omega=constants.omegas[i], Psat=VaporPressures[i].T_dependent_property,
                                                Vml=VolumeLiquids[i])
                                for i in cmps]
        

        if ViscosityGases is None:
            ViscosityGases = [ViscosityGas(CASRN=constants.CASs[i], MW=constants.MWs[i], Tc=constants.Tcs[i],
                                           Pc=constants.Pcs[i], Zc=constants.Zcs[i], dipole=constants.dipoles[i],
                                           Vmg=lambda T: VolumeGases[i](T, 101325.0)) # Might be an issue with what i refers too
                                for i in cmps]
        if ThermalConductivityLiquids is None:
            ThermalConductivityLiquids = [ThermalConductivityLiquid(CASRN=constants.CASs[i], MW=constants.MWs[i], 
                                                                    Tm=constants.Tms[i], Tb=constants.Tbs[i],
                                                                    Tc=constants.Tcs[i], Pc=constants.Pcs[i], 
                                                                    omega=constants.omegas[i], Hfus=constants.Hfus_Tms[i])
                                                for i in cmps]

        if ThermalConductivityGases is None:
            ThermalConductivityGases = [ThermalConductivityGas(CASRN=constants.CASs[i], MW=constants.MWs[i], Tb=constants.Tbs[i],
                                                               Tc=constants.Tcs[i], Pc=constants.Pcs[i], Vc=constants.Vcs[i],
                                                               Zc=constants.Zcs[i], omega=constants.omegas[i], dipole=constants.dipoles[i],
                                                               Vmg=VolumeGases[i], mug=ViscosityLiquids[i].T_dependent_property,
                                                               Cvgm=lambda T : HeatCapacityGases[i].T_dependent_property(T) - R)
                                                for i in cmps]

        if SurfaceTensions is None:
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

        if VolumeSolidMixtureObj is None:
            VolumeSolidMixtureObj = VolumeSolidMixture(CASs=constants.CASs, MWs=constants.MWs, VolumeSolids=VolumeSolids)
        if VolumeLiquidMixtureObj is None:
            VolumeLiquidMixtureObj = VolumeLiquidMixture(MWs=constants.MWs, Tcs=constants.Tcs, Pcs=constants.Pcs, Vcs=constants.Vcs, Zcs=constants.Zcs, omegas=constants.omegas, CASs=constants.CASs, VolumeLiquids=VolumeLiquids)
        if VolumeGasMixtureObj is None:
            VolumeGasMixtureObj = VolumeGasMixture(eos=None, MWs=constants.MWs, CASs=constants.CASs, VolumeGases=VolumeGases)

        if HeatCapacityLiquidMixtureObj is None:
            HeatCapacityLiquidMixtureObj = HeatCapacityLiquidMixture(MWs=constants.MWs, CASs=constants.CASs, HeatCapacityLiquids=HeatCapacityLiquids)
        if HeatCapacityGasMixtureObj is None:
            HeatCapacityGasMixtureObj = HeatCapacityGasMixture(MWs=constants.MWs, CASs=constants.CASs, HeatCapacityGases=HeatCapacityGases)
        if HeatCapacitySolidMixtureObj is None:
            HeatCapacitySolidMixtureObj = HeatCapacitySolidMixture(MWs=constants.MWs, CASs=constants.CASs, HeatCapacitySolids=HeatCapacitySolids)

        if ViscosityLiquidMixtureObj is None:
            ViscosityLiquidMixtureObj = ViscosityLiquidMixture(MWs=constants.MWs, CASs=constants.CASs, ViscosityLiquids=ViscosityLiquids)
        if ViscosityGasMixtureObj is None:
            ViscosityGasMixtureObj = ViscosityGasMixture(MWs=constants.MWs, molecular_diameters=constants.molecular_diameters, Stockmayers=constants.Stockmayers, CASs=constants.CASs, ViscosityGases=ViscosityGases)

        if ThermalConductivityLiquidMixtureObj is None:
            ThermalConductivityLiquidMixtureObj = ThermalConductivityLiquidMixture(CASs=constants.CASs, MWs=constants.MWs, ThermalConductivityLiquids=ThermalConductivityLiquids)
        if ThermalConductivityGasMixtureObj is None:
            ThermalConductivityGasMixtureObj = ThermalConductivityGasMixture(MWs=constants.MWs, Tbs=constants.Tbs, CASs=constants.CASs, ThermalConductivityGases=ThermalConductivityGases, ViscosityGases=ViscosityGases)

        if SurfaceTensionMixtureObj is None:
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
            for prop in iter_props:
                try:
                    s += '%s=%s,\n' %(prop, self.as_best_fit(getattr(self, prop)))
                except Exception as e:
                    print(e, prop)
                    
            s += ')'
            return s
        
        s = '['
        for obj in props:
            s += (obj.as_best_fit() + ',\n')
        s += ']'
        return s
