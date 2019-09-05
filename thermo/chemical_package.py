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
from thermo.thermal_conductivity import ThermalConductivityLiquidMixture, ThermalConductivityGasMixture
from thermo.volume import VolumeLiquidMixture, VolumeGasMixture, VolumeSolidMixture
from thermo.permittivity import *
from thermo.heat_capacity import HeatCapacitySolidMixture, HeatCapacityGasMixture, HeatCapacityLiquidMixture
from thermo.interface import SurfaceTensionMixture
from thermo.viscosity import ViscosityLiquidMixture, ViscosityGasMixture
from thermo.utils import *


class ChemicalConstantsPackage(object):
    __slots__ = ('atom_fractions', 'atomss', 'Carcinogens', 'CASs', 'Ceilings', 'charges',
                 'conductivities', 'dipoles', 'economic_statuses', 'formulas', 'Gfgs', 
                 'Gfgs_mass', 'GWPs', 'Hcs', 'Hcs_lower', 'Hcs_lower_mass', 'Hcs_mass', 
                 'Hfgs', 'Hfgs_mass', 'Hfus_Tms', 'Hfus_Tms_mass', 'Hsub_Tms', 
                 'Hsub_Tms_mass', 'Hvap_298s', 'Hvap_Tbs', 'Hvap_Tbs_mass', 
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
                 'Vml_STPs', 'Zcs')
    
    def __repr__(self):
        s = 'ChemicalPackage('
        for k in self.__slots__:
            if any(i is not None for i in getattr(self, k)):
                s += '%s=%s, '%(k, getattr(self, k))
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
                 Psat_298s=None, Hvap_298s=None,
                 # Analytical
                 RIs=None, conductivities=None,
                 # Odd constants
                 charges=None, dipoles=None, Stockmayers=None, 
                 molecular_diameters=None, Van_der_Waals_volumes=None,
                 Van_der_Waals_areas=None, Parachors=None, StielPolars=None,
                 atomss=None, atom_fractions=None,
                 similarity_variables=None, phase_STPs=None,
                 # Other identifiers
                 PubChems=None, formulas=None, smiless=None, InChIs=None,
                 InChI_Keys=None,
                 # Groups
                 UNIFAC_groups=None, UNIFAC_Dortmund_groups=None, 
                 PSRK_groups=None
                 ):
        self.N = N = len(MWs)
    
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
        if Van_der_Waals_areas is None: Van_der_Waals_areas = [None]*N
        if Van_der_Waals_volumes is None: Van_der_Waals_volumes = [None]*N
        if Vcs is None: Vcs = [None]*N
        if Vml_STPs is None: Vml_STPs = [None]*N
        if Zcs is None: Zcs = [None]*N
        
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
        self.Van_der_Waals_areas = Van_der_Waals_areas
        self.Van_der_Waals_volumes = Van_der_Waals_volumes
        self.Vcs = Vcs
        self.Vml_STPs = Vml_STPs
        self.Zcs = Zcs


class PropertyCorrelationPackage(object):
    __slots__ = ()
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
        cmps = constants.cmps
        if VaporPressures is None:
            VaporPressures = [VaporPressure(Tb=constants.Tbs[i], Tc=constants.Tcs[i], Pc=constants.Pcs[i],
                                            omega=constants.omegas[i], CASRN=constants.CASs[i],
                                            best_fit=get_chemical_constants(constants.CASs[i], 'VaporPressure'))
                              for i in cmps]
        # TODO remaining properties
        

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
        
        self.VolumeSolidMixtureObj = VolumeSolidMixtureObj
        self.VolumeLiquidMixtureObj = VolumeLiquidMixtureObj
        self.VolumeGasMixtureObj = VolumeGasMixtureObj
        
        self.HeatCapacityLiquidMixtureObj = HeatCapacityLiquidMixtureObj
        self.HeatCapacityGasMixtureObj = HeatCapacityGasMixtureObj
        self.HeatCapacitySolidMixtureObj = HeatCapacitySolidMixtureObj
        
        self.ViscosityLiquidMixtureObj = ViscosityLiquidMixtureObj
        self.ViscosityGasMixtureObj = ViscosityGasMixtureObj
        
        self.ThermalConductivityLiquidMixtureObj = ThermalConductivityLiquidMixtureObj
        self.ThermalConductivityGasMixtureObj = ThermalConductivityGasMixtureObj
        
        self.SurfaceTensionMixtureObj = SurfaceTensionMixtureObj


