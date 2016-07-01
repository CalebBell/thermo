# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from thermo.identifiers import *
from thermo.identifiers import _MixtureDict
from thermo.vapor_pressure import VaporPressure
from thermo.phase_change import Tb, Tm, Hfus, Hsub, Tliquidus, EnthalpyVaporization
from thermo.activity import identify_phase, identify_phase_mixture, Pbubble_mixture, Pdew_mixture

from thermo.critical import Tc, Pc, Vc, Zc, Tc_mixture, Pc_mixture, Vc_mixture
from thermo.acentric import omega, omega_mixture, StielPolar
from thermo.triple import Tt, Pt
from thermo.thermal_conductivity import thermal_conductivity_liquid_mixture, thermal_conductivity_gas_mixture, ThermalConductivityLiquid, ThermalConductivityGas
from thermo.volume import VolumeGas, VolumeLiquid, VolumeSolid, volume_liquid_mixture, volume_gas_mixture
from thermo.permittivity import *
from thermo.heat_capacity import HeatCapacitySolid, HeatCapacityGas, HeatCapacityLiquid, Cp_gas_mixture, Cv_gas_mixture, Cp_liq_mixture
from thermo.interface import SurfaceTension, surface_tension_mixture
from thermo.viscosity import viscosity_liquid_mixture, viscosity_gas_mixture, ViscosityLiquid, ViscosityGas
from thermo.reaction import Hf
from thermo.combustion import Hcombustion
from thermo.safety import Tflash, Tautoignition, LFL, UFL, TWA, STEL, Ceiling, Skin, Carcinogen, LFL_mixture, UFL_mixture
from thermo.solubility import solubility_parameter
from thermo.dipole import dipole
from thermo.utils import *
from fluids.core import Reynolds, Capillary, Weber, Bond, Grashof, Peclet_heat
from thermo.lennard_jones import Stockmayer, molecular_diameter
from thermo.environment import GWP, ODP, logP
from thermo.law import legal_status, economic_status
from thermo.refractivity import refractive_index
from thermo.electrochem import conductivity
from thermo.elements import atom_fractions, mass_fractions, similarity_variable, atoms_to_Hill, simple_formula_parser
from thermo.coolprop import has_CoolProp

from fluids.core import *

# RDKIT
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import IPythonConsole
except:  # pragma: no cover
    pass


from collections import Counter

import warnings
warnings.filterwarnings("ignore")


class Chemical(object): # pragma: no cover
    '''Class for obtaining properties of chemicals.
    Considered somewhat stable, but changes to some mthods are expected.


    Default initialization is for 298.15 K, 1 atm.
    Goal is for, when a method fails, a warning is printed.
    '''

    def __init__(self, ID, T=298.15, P=101325):
        self.ID = ID
        self.P = P
        self.T = T

        # Identification
        self.CAS = CASfromAny(ID)
        self.PubChem = PubChem(self.CAS)
        self.MW = MW(self.CAS)
        self.formula = formula(self.CAS)
        self.smiles = smiles(self.CAS)
        self.InChI = InChI(self.CAS)
        self.InChI_Key = InChI_Key(self.CAS)
        self.IUPAC_name = IUPAC_name(self.CAS).lower()
        self.name = name(self.CAS).lower()
        self.synonyms = [i.lower() for i in synonyms(self.CAS)]

        self.set_none()
        self.set_structure()

        self.set_constant_sources()
        self.set_constants()
        self.set_T_sources()
        self.set_T(self.T)
        # Run it twice to deal with dependencies
        self.set_constant_sources()
        self.set_constants()
        self.set_T_sources()
        self.set_T(self.T)
        self.set_phase()


    def set_structure(self):
        try:
            self.rdkitmol = Chem.MolFromSmiles(self.smiles)
            self.rdkitmol_Hs = Chem.AddHs(self.rdkitmol)
            self.atoms = dict(Counter(atom.GetSymbol() for atom in self.rdkitmol_Hs.GetAtoms()))
            self.charge = Chem.GetFormalCharge(self.rdkitmol)
            self.rings = Chem.Descriptors.RingCount(self.rdkitmol)
            self.atom_fractions = atom_fractions(self.atoms)
            self.mass_fractions = mass_fractions(self.atoms, self.MW)
            self.similarity_variable = similarity_variable(self.atoms, self.MW)
            self.Hill = atoms_to_Hill(self.atoms)
        except:
            self.atoms = simple_formula_parser(self.formula)
            self.atom_fractions = atom_fractions(self.atoms)
            self.mass_fractions = mass_fractions(self.atoms, self.MW)
            self.similarity_variable = similarity_variable(self.atoms, self.MW)
            self.Hill = atoms_to_Hill(self.atoms)

    def draw_2d(self):
        try:
            return Draw.MolToImage(self.rdkitmol)
        except:
            return 'Rdkit required'

    def draw_3d(self):
        try:
            import py3Dmol
            AllChem.EmbedMultipleConfs(self.rdkitmol_Hs)
            mb = Chem.MolToMolBlock(self.rdkitmol_Hs)
            p = py3Dmol.view(width=300,height=300)
            p.addModel(mb,'sdf')
            p.setStyle({'stick':{}})
            # Styles: stick, line, cross, sphere
            p.zoomTo()
            p.show()
            return p
        except:
            return 'py3Dmol and rdkit required'

    def set_none(self):
        # Null values as necessary
        self.rdkitmol = None
        self.atoms = None
        self.charge = None
        self.rings = None
        self.atom_fractions = None
        self.mass_fractions = None
        self.similarity_variable = None

        self.Tm = None
        self.Tb = None

        self.Tc = None
        self.Pc = None
        self.Vc = None
        self.omega = None
        self.StielPolar = None

        self.Zc = None
        self.rhoC = None
        self.rhoCm = None

        self.Pt = None
        self.Tt = None

        self.Hf = None
        self.Hf298 = None
        self.Hfl = None
        self.Hfg = None
        self.Hc = None

        self.Tflash = None
        self.Tautoignition = None
        self.LFL = None
        self.UFL = None

        self.TWA = None
        self.STEL = None
        self.Ceiling = None
        self.Skin = None
        self.Carcinogen = None

        self.dipole = None
        self.molecular_diameter = None
        self.Stockmayer = None

        # Temperature dependent values
        self.Psat = None
        self.Psat_298 = None
        self.rhos = None
        self.Vms = None
        self.Zs = None
        self.rhosm = None
        self.rhol = None
        self.Vml = None
        self.Zl = None
        self.rholm = None
        self.rhog = None
        self.Vmg = None
        self.Zg = None
        self.rhogm = None
        self.Bvirial = None

        self.Cps = None
        self.Cpl = None
        self.Cpg = None
        self.Cvg = None

        self.Cpsm = None
        self.Cplm = None
        self.Cpgm = None
        self.Cvgm = None
        self.isentropic_exponent = None

        self.permittivity = None
        self.conductivity = None
        self.sigma = None
        self.mul = None
        self.mug = None

        self.Hvap = None
        self.HvapTb = None
        self.Hfus = None
        self.Hsub = None

        self.Hvapm = None
        self.HvapTbm = None
        self.Hfusm = None
        self.Hsubm = None

        self.H = None

        self.ks = None
        self.kl = None
        self.kg = None

        self.Prl = None
        self.Prg = None
        self.alphal = None
        self.alphag = None

        self.solubility_parameter = None
        self.Parachor = None

        self.GWP = None
        self.ODP = None
        self.logP = None

        self.RI = None
        self.RIT = None
        self.conductivity = None
        self.conductivityT = None



    def set_constant_sources(self):
        self.Tm_sources = Tm(CASRN=self.CAS, AvailableMethods=True)
        self.Tm_source = self.Tm_sources[0]
        self.Tb_sources = Tb(CASRN=self.CAS, AvailableMethods=True)
        self.Tb_source = self.Tb_sources[0]

        # Critical Point
        self.Tc_methods = Tc(self.CAS, AvailableMethods=True)
        self.Tc_method = self.Tc_methods[0]
        self.Pc_methods = Pc(self.CAS, AvailableMethods=True)
        self.Pc_method = self.Pc_methods[0]
        self.Vc_methods = Vc(self.CAS, AvailableMethods=True)
        self.Vc_method = self.Vc_methods[0]
        self.omega_methods = omega(CASRN=self.CAS, AvailableMethods=True)
        self.omega_method = self.omega_methods[0]
        self.StielPolar_methods = StielPolar(Tc=self.Tc, Pc=self.Pc, omega=self.omega, CASRN=self.CAS, AvailableMethods=True)
        self.StielPolar_method = self.StielPolar_methods[0]

        # Triple point
        self.Tt_sources = Tt(self.CAS, AvailableMethods=True)
        self.Tt_source = self.Tt_sources[0]
        self.Pt_sources = Pt(self.CAS, AvailableMethods=True)
        self.Pt_source = self.Pt_sources[0]

        # Fire Safety Limits
        self.Tflash_sources = Tflash(self.CAS, AvailableMethods=True)
        self.Tflash_source = self.Tflash_sources[0]
        self.Tautoignition_sources = Tautoignition(self.CAS, AvailableMethods=True)
        self.Tautoignition_source = self.Tautoignition_sources[0]
        self.LFL_sources = LFL(atoms=self.atoms, Hc=self.Hc, CASRN=self.CAS, AvailableMethods=True)
        self.LFL_source = self.LFL_sources[0]
        self.UFL_sources = UFL(atoms=self.atoms, Hc=self.Hc, CASRN=self.CAS, AvailableMethods=True)
        self.UFL_source = self.UFL_sources[0]

        # Chemical Exposure Limits
        self.TWA_sources = TWA(self.CAS, AvailableMethods=True)
        self.TWA_source = self.TWA_sources[0]
        self.STEL_sources = STEL(self.CAS, AvailableMethods=True)
        self.STEL_source = self.STEL_sources[0]
        self.Ceiling_sources = Ceiling(self.CAS, AvailableMethods=True)
        self.Ceiling_source = self.Ceiling_sources[0]
        self.Skin_sources = Skin(self.CAS, AvailableMethods=True)
        self.Skin_source = self.Skin_sources[0]
        self.Carcinogen_sources = Carcinogen(self.CAS, AvailableMethods=True)
        self.Carcinogen_source = self.Carcinogen_sources[0]

        # Chemistry - currently molar
        self.Hf_sources = Hf(CASRN=self.CAS, AvailableMethods=True)
        self.Hf_source = self.Hf_sources[0]

        # Misc
        self.dipole_sources = dipole(CASRN=self.CAS, AvailableMethods=True)
        self.dipole_source = self.dipole_sources[0]
        self.molecular_diameter_sources = molecular_diameter(Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega, Vm=None, Vb=None, AvailableMethods=True, CASRN=self.CAS)
        self.molecular_diameter_source = self.molecular_diameter_sources[0]
        self.Stockmayer_sources = Stockmayer(Tc=self.Tc, Zc=self.Zc, omega=self.omega, AvailableMethods=True, CASRN=self.CAS)
        self.Stockmayer_source = self.Stockmayer_sources[0]

        # Environmental
        self.GWP_sources = GWP(CASRN=self.CAS, AvailableMethods=True)
        self.GWP_source = self.GWP_sources[0]
        self.ODP_sources = ODP(CASRN=self.CAS, AvailableMethods=True)
        self.ODP_source = self.ODP_sources[0]
        self.logP_sources = logP(CASRN=self.CAS, AvailableMethods=True)
        self.logP_source = self.logP_sources[0]

        # Legal
        self.legal_status_sources = legal_status(CASRN=self.CAS, AvailableMethods=True)
        self.legal_status_source = self.legal_status_sources[0]
        self.economic_status_sources = economic_status(CASRN=self.CAS, AvailableMethods=True)
        self.economic_status_source = self.economic_status_sources[0]

        # Analytical
        self.RI_sources = refractive_index(CASRN=self.CAS, AvailableMethods=True)
        self.RI_source = self.RI_sources[0]

        self.conductivity_sources = conductivity(CASRN=self.CAS, AvailableMethods=True)
        self.conductivity_source = self.conductivity_sources[0]


    def set_constants(self):
        self.Tm = Tm(self.CAS, Method=self.Tm_source)
        self.Tb = Tb(self.CAS, Method=self.Tb_source)

        # Critical Point
        self.Tc = Tc(self.CAS, Method=self.Tc_method)
        self.Pc = Pc(self.CAS, Method=self.Pc_method)
        self.Vc = Vc(self.CAS, Method=self.Vc_method)
        self.omega = omega(self.CAS, Method=self.omega_method)
        self.StielPolar = StielPolar(Tc=self.Tc, Pc=self.Pc, omega=self.omega, CASRN=self.CAS, Method=self.StielPolar_method)

        if all((self.Tc, self.Pc, self.Vc)):
            self.Zc = Z(self.Tc, self.Pc, self.Vc)

        if self.Vc:
            self.rhoC = Vm_to_rho(self.Vc, self.MW)
            self.rhoCm = 1./self.Vc

        # Triple point
        self.Pt = Pt(self.CAS, Method=self.Pt_source)
        self.Tt = Tt(self.CAS, Method=self.Tt_source)

        # Chemistry
        self.Hf298 = Hf(CASRN=self.CAS, Method=self.Hf_source)
        self.Hc = Hcombustion(atoms=self.atoms, Hf=self.Hf298)

        # Fire Safety Limits
        self.Tflash = Tflash(self.CAS, Method=self.Tflash_source)
        self.Tautoignition = Tautoignition(self.CAS, Method=self.Tautoignition_source)
        self.LFL = LFL(atoms=self.atoms, Hc=self.Hc, CASRN=self.CAS, Method=self.LFL_source)
        self.UFL = UFL(atoms=self.atoms, Hc=self.Hc, CASRN=self.CAS, Method=self.UFL_source)

        # Chemical Exposure Limits
        self.TWA = TWA(self.CAS, Method=self.TWA_source)
        self.STEL = STEL(self.CAS, Method=self.STEL_source)
        self.Ceiling = Ceiling(self.CAS, Method=self.Ceiling_source)
        self.Skin = Skin(self.CAS, Method=self.Skin_source)
        self.Carcinogen = Carcinogen(self.CAS, Method=self.Carcinogen_source)

        # Misc
        self.dipole = dipole(self.CAS, Method=self.dipole_source) # Units of Debye
        self.molecular_diameter = molecular_diameter(Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega, Vm=None, Vb=None, Method=self.molecular_diameter_source, CASRN=self.CAS)
        self.Stockmayer = Stockmayer(Tm=self.Tm, Tb=self.Tb, Tc=self.Tc, Zc=self.Zc, omega=self.omega, Method=self.Stockmayer_source, CASRN=self.CAS)

        # Environmental
        self.GWP = GWP(CASRN=self.CAS, Method=self.GWP_source)
        self.ODP = ODP(CASRN=self.CAS, Method=self.ODP_source)
        self.logP = logP(CASRN=self.CAS, Method=self.logP_source)

        # Legal
        self.legal_status = legal_status(self.CAS, Method=self.legal_status_source)
        self.economic_status = economic_status(self.CAS, Method=self.economic_status_source)

        # Analytical
        self.RI, self.RIT =  refractive_index(CASRN=self.CAS, Method=self.RI_source)
        self.conductivity, self.conductivityT = conductivity(CASRN=self.CAS, Method=self.conductivity_source)


    def set_T_sources(self):
        # Tempearture and Pressure Denepdence
        # Get and choose initial methods
        self.VaporPressure = VaporPressure(Tb=self.Tb, Tc=self.Tc, Pc=self.Pc, omega=self.omega, CASRN=self.CAS)
        self.Psat_298 = self.VaporPressure.T_dependent_property(298.15)

        self.VolumeLiquid = VolumeLiquid(MW=self.MW, Tb=self.Tb, Tc=self.Tc,
                          Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega,
                          dipole=self.dipole, Psat=self.Psat, CASRN=self.CAS)
        if self.Tb:
            self.Vml_Tb = self.VolumeLiquid.T_dependent_property(self.Tb)
        else:
            self.Vml_Tb = None
        if self.Tm:
            self.Vml_Tm = self.VolumeLiquid.T_dependent_property(self.Tm)
        else:
            self.Vml_Tm = None
        self.Vml_STP = self.VolumeLiquid.T_dependent_property(298.15)

        self.VolumeGas = VolumeGas(MW=self.MW, Tc=self.Tc, Pc=self.Pc, omega=self.omega, dipole=self.dipole, CASRN=self.CAS)

        self.VolumeSolid = VolumeSolid(CASRN=self.CAS, MW=self.MW, Tt=self.Tt)

        self.HeatCapacitySolid = HeatCapacitySolid(MW=self.MW, similarity_variable=self.similarity_variable, CASRN=self.CAS)

        self.HeatCapacityLiquid = HeatCapacityLiquid(CASRN=self.CAS, MW=self.MW, similarity_variable=self.similarity_variable, Tc=self.Tc, omega=self.omega, Cpgm=self.Cpgm)

        self.HeatCapacityGas = HeatCapacityGas(CASRN=self.CAS, MW=self.MW, similarity_variable=self.similarity_variable)

        self.ViscosityLiquid = ViscosityLiquid(CASRN=self.CAS, MW=self.MW, Tm=self.Tm, Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, omega=self.omega, Psat=self.Psat, Vml=self.Vml)

        self.ViscosityGas = ViscosityGas(CASRN=self.CAS, MW=self.MW, Tc=self.Tc, Pc=self.Pc, Zc=self.Zc, dipole=self.dipole, Vmg=self.Vmg)

        self.EnthalpyVaporization = EnthalpyVaporization(CASRN=self.CAS, Tb=self.Tb, Tc=self.Tc, Pc=self.Pc, omega=self.omega, similarity_variable=self.similarity_variable)
        self.HvapTbm = self.EnthalpyVaporization.T_dependent_property(self.Tb)
        self.HvapTb = property_molar_to_mass(self.HvapTbm, self.MW)

        self.Hfus_methods = Hfus(T=self.T, P=self.P, MW=self.MW, AvailableMethods=True, CASRN=self.CAS)
        self.Hfus_method = self.Hfus_methods[0]

        self.Hsub_methods = Hsub(T=self.T, P=self.P, MW=self.MW, AvailableMethods=True, CASRN=self.CAS)
        self.Hsub_method = self.Hsub_methods[0]

        self.ThermalConductivityLiquid = ThermalConductivityLiquid(CASRN=self.CAS, MW=self.MW, Tm=self.Tm, Tb=self.Tb, Tc=self.Tc, Pc=self.Pc, omega=self.omega, Hfus=self.Hfusm)

        self.ThermalConductivityGas = ThermalConductivityGas(CASRN=self.CAS, MW=self.MW, Tb=self.Tb, Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega, dipole=self.dipole, Vmg=self.Vmg, Cvgm=self.Cvgm, mug=self.mug)

        self.SurfaceTension = SurfaceTension(CASRN=self.CAS, Tb=self.Tb, Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega, StielPolar=self.StielPolar)

        self.Permittivity = Permittivity(CASRN=self.CAS)

        self.solubility_parameter_methods = solubility_parameter(T=self.T, Hvapm=self.Hvapm, Vml=self.Vml, AvailableMethods=True, CASRN=self.CAS)
        self.solubility_parameter_method = self.solubility_parameter_methods[0]

    # Preferences for Chemical
    # Note: First we list all of the methods that we have the information for.

    def set_T(self, T=None):
        if T:
            self.T = T
        self.Psat = self.VaporPressure.T_dependent_property(T=self.T)
        # Update Psat for various properties

        self.Vms = self.VolumeSolid.T_dependent_property(T=self.T)
        if self.Vms:
            self.rhos = Vm_to_rho(self.Vms, self.MW)
            self.rhosm = 1/self.Vms
            self.Zs = Z(self.T, self.P, self.Vms)
        else:
            self.Vms = None
            self.rhosm = None
            self.Zs = None


        self.VolumeLiquid.Psat = self.Psat
        self.Vml = self.VolumeLiquid.TP_dependent_property(self.T, self.P)
        # TODO: derivative
        self.isobaric_expansion_l = isobaric_expansion(V1=self.Vml, dT=0.01, V2=self.VolumeLiquid.TP_dependent_property(self.T+0.01, self.P))
        if not self.Vml:
            self.Vml = self.VolumeLiquid.T_dependent_property(self.T)
            self.isobaric_expansion_l = isobaric_expansion(V1=self.Vml, dT=0.01, V2=self.VolumeLiquid.T_dependent_property(self.T+0.01))

        if self.Vml:
            self.rhol = Vm_to_rho(self.Vml, self.MW)
            self.Zl = Z(self.T, self.P, self.Vml)
            self.rholm = 1./self.Vml
        else:
            self.rhol = None
            self.Zl = None
            self.rholm = None


        self.Vmg = self.VolumeGas.TP_dependent_property(T=self.T, P=self.P)
        if self.Vmg:
            self.rhog = Vm_to_rho(self.Vmg, self.MW)
            self.Zg = Z(self.T, self.P, self.Vmg)
            self.rhogm = 1./self.Vmg
            self.Bvirial = B_from_Z(self.Zg, self.T, self.P)
        else:
            self.rhog = None
            self.Zg = None
            self.rhogm = None
            self.Bvirial = None



        self.isobaric_expansion_g = isobaric_expansion(V1=self.Vmg, dT=0.01, V2=self.VolumeGas.TP_dependent_property(T=self.T+0.01, P=self.P))

        self.Cpsm = self.HeatCapacitySolid.T_dependent_property(self.T)
        self.Cpgm = self.HeatCapacityGas.T_dependent_property(self.T)

        self.HeatCapacityLiquid.Cpgm = self.Cpgm
        self.Cplm = self.HeatCapacityLiquid.T_dependent_property(self.T)
        if self.Cplm:
            self.Cpl = property_molar_to_mass(self.Cplm, self.MW)

        self.Cps = property_molar_to_mass(self.Cpsm, self.MW)
        self.Cpg = property_molar_to_mass(self.Cpgm, self.MW)

        if self.Cpgm:
            self.Cvgm = self.Cpgm - R
            self.Cvg = property_molar_to_mass(self.Cvgm, self.MW)
        else:
            self.Cvgm = None
            self.Cvg = None


        if all((self.Cpg, self.Cvg)):
            self.isentropic_exponent = isentropic_exponent(self.Cpg, self.Cvg)
        else:
            self.isentropic_exponent = None

        self.EnthalpyVaporization.Psat = self.Psat
        self.EnthalpyVaporization.Zl = self.Zl
        self.EnthalpyVaporization.Zg = self.Zg
        self.Hvapm = self.EnthalpyVaporization.T_dependent_property(self.T)
        self.Hvap = property_molar_to_mass(self.Hvapm, self.MW)

        self.Hfus = Hfus(T=self.T, P=self.P, MW=self.MW, Method=self.Hfus_method, CASRN=self.CAS)
        self.Hsub = Hsub(T=self.T, P=self.P, MW=self.MW, Method=self.Hsub_method, CASRN=self.CAS)
        self.Hfusm = property_mass_to_molar(self.Hfus, self.MW)
        self.Hsubm = property_mass_to_molar(self.Hsub, self.MW)

        self.ViscosityLiquid.Psat = self.Psat
        self.ViscosityLiquid.Vml = self.Vml
        self.mul = self.ViscosityLiquid.TP_dependent_property(self.T, self.P)
        if not self.mul:
            self.mul = self.ViscosityLiquid.T_dependent_property(self.T)

        self.ViscosityGas.Vmg = self.Vmg
        self.mug = self.ViscosityGas.TP_dependent_property(self.T, self.P)
        if not self.mug:
            self.mug = self.ViscosityGas.T_dependent_property(self.T)

        self.kl = self.ThermalConductivityLiquid.TP_dependent_property(self.T, self.P)
        if not self.kl:
            self.kl = self.ThermalConductivityLiquid.T_dependent_property(self.T)

        self.ThermalConductivityGas.Vmg = self.Vmg
        self.ThermalConductivityGas.Cvgm = self.Cvgm
        self.ThermalConductivityGas.mug = self.mug
        self.kg = self.ThermalConductivityGas.TP_dependent_property(self.T, self.P)
        if not self.kg:
            self.kg = self.ThermalConductivityGas.T_dependent_property(self.T)


        self.sigma = self.SurfaceTension.T_dependent_property(self.T)
        self.permittivity = self.Permittivity.T_dependent_property(self.T)

        self.solubility_parameter = solubility_parameter(T=self.T, Hvapm=self.Hvapm, Vml=self.Vml, Method=self.solubility_parameter_method, CASRN=self.CAS)

        if all((self.sigma, self.MW, self.rhol, self.rhog)):
            self.Parachor = Parachor(sigma=self.sigma, MW=self.MW, rhol=self.rhol, rhog=self.rhog)
        else:
            self.Parachor = None

        self.JTl = JT(T=self.T, V=self.Vml, Cp=self.Cplm, isobaric_expansion=self.isobaric_expansion_l)
        self.JTg = JT(T=self.T, V=self.Vmg, Cp=self.Cpgm, isobaric_expansion=self.isobaric_expansion_g)

        try:
            self.nul = nu_mu_converter(mu=self.mul, rho=self.rhol)
        except Exception:
            self.nul = None
        try:
            self.nug = nu_mu_converter(mu=self.mug, rho=self.rhog)
        except Exception:
            self.nug = None

        try:
            self.Prl = Prandtl(Cp=self.Cpl, mu=self.mul, k=self.kl)
        except Exception:
            self.Prl = None
        try:
            self.Prg = Prandtl(Cp=self.Cpg, mu=self.mug, k=self.kg)
        except Exception:
            self.Prg = None

        try:
            self.alphal = thermal_diffusivity(k=self.kl, rho=self.rhol, Cp=self.Cpl)
        except Exception:
            self.alphal = None
        try:
            self.alphag = thermal_diffusivity(k=self.kg, rho=self.rhog, Cp=self.Cpg)
        except Exception:
            self.alphag = None

        self.set_phase()

        return True

    def set_phase(self):
        self.phase_STP = identify_phase(T=298.15, P=101325., Tm=self.Tm, Tb=self.Tb, Tc=self.Tc, Psat=self.Psat_298)
        self.phase = identify_phase(T=self.T, P=self.P, Tm=self.Tm, Tb=self.Tb, Tc=self.Tc, Psat=self.Psat)
        self.k = phase_set_property(phase=self.phase, s=self.ks, l=self.kl, g=self.kg)
        self.rho = phase_set_property(phase=self.phase, s=self.rhos, l=self.rhol, g=self.rhog)
        self.Vm = phase_set_property(phase=self.phase, s=self.Vms, l=self.Vml, g=self.Vmg)
        if self.Vm:
            self.Z = Z(self.T, self.P, self.Vm)
        else:
            self.Z = None
        self.Cp = phase_set_property(phase=self.phase, s=self.Cps, l=self.Cpl, g=self.Cpg)
        self.Cpm = phase_set_property(phase=self.phase, s=self.Cpsm, l=self.Cplm, g=self.Cpgm)
        self.mu = phase_set_property(phase=self.phase, l=self.mul, g=self.mug)
        self.nu = phase_set_property(phase=self.phase, l=self.nul, g=self.nug)
        self.Pr = phase_set_property(phase=self.phase, l=self.Prl, g=self.Prg)
        self.alpha = phase_set_property(phase=self.phase, l=self.alphal, g=self.alphag)
        self.isobaric_expansion = phase_set_property(phase=self.phase, l=self.isobaric_expansion_l, g=self.isobaric_expansion_g)
        self.JT = phase_set_property(phase=self.phase, l=self.JTl, g=self.JTg)
        # TODO
        self.H = 0
        self.Hm = 0


    def Tsat(self, P):
        return self.VaporPressure.solve_prop(P)


    def Reynolds(self, V=None, D=None):
        return Reynolds(V=V, D=D, rho=self.rho, mu=self.mu)

    def Capillary(self, V=None):
        return Capillary(V=V, mu=self.mu, sigma=self.sigma)

    def Weber(self, V=None, D=None):
        return Weber(V=V, L=D, rho=self.rho, sigma=self.sigma)

    def Bond(self, L=None):
        return Bond(rhol=self.rhol, rhog=self.rhog, sigma=self.sigma, L=L)

    def Jakob(self, Tw=None):
        return Jakob(Cp=self.Cp, Hvap=self.Hvap, Te=Tw-self.T)

    def Grashof(self, Tw=None, L=None):
        return Grashof(L=L, beta=self.isobaric_expansion, T1=Tw, T2=self.T,
                       rho=self.rho, mu=self.mu)

    def Peclet_heat(self, V=None, D=None):
        return Peclet_heat(V=V, L=D, rho=self.rho, Cp=self.Cp, k=self.k)


class Mixture(object):  # pragma: no cover
    '''Class for obtaining properties of mixtures of chemicals.
    Must be considered unstable due to the goal of changing each of the
    property methods into object-oriented interfaces.

    Most methods are relatively accurate.

    Default initialization is for 298.15 K, 1 atm.
    '''
    def __init__(self, IDs, zs=None, ws=None, Vfls=None, Vfgs=None,
                 T=298.15, P=101325):
        self.P = P
        self.T = T

        if type(IDs) == type('') or (type(IDs) == type([]) and len(IDs) == 1):
            mixname = mixture_from_any(IDs)
            if mixname:
                _d = _MixtureDict[mixname]
                IDs = _d["CASs"]
                ws = _d["ws"]
                self.mixname = mixname
                self.mixsource = _d["Source"]

        self.components = tuple(IDs)
        self.Chemicals = [Chemical(component, P=P, T=T) for component in self.components]
        self.names = [i.name for i in self.Chemicals]
        self.MWs = [i.MW for i in self.Chemicals]
        self.CASs = [i.CAS for i in self.Chemicals]
        self.PubChems = [i.PubChem for i in self.Chemicals]
        self.formulas = [i.formula for i in self.Chemicals]
        self.smiless = [i.smiles for i in self.Chemicals]
        self.InChIs = [i.InChI for i in self.Chemicals]
        self.InChI_Keys = [i.InChI_Key for i in self.Chemicals]
        self.IUPAC_names = [i.IUPAC_name for i in self.Chemicals]
        self.synonymss = [i.synonyms for i in self.Chemicals]

        self.charges = [i.charge for i in self.Chemicals]
        self.atomss = [i.atoms for i in self.Chemicals]
        self.ringss = [i.rings for i in self.Chemicals]
        self.atom_fractionss = [i.atom_fractions for i in self.Chemicals]
        self.mass_fractionss = [i.mass_fractions for i in self.Chemicals]

        # Required for densities for volume fractions before setting fractions
        self.set_chemical_constants()
        self.set_chemical_T()
        if zs:
            self.zs = zs
            self.ws = zs_to_ws(zs, self.MWs)
            try:
                self.Vfls = zs_to_Vfs(self.zs, self.Vmls)
            except:
                pass
            try:
                self.Vfgs = zs_to_Vfs(self.zs, self.Vmgs)
            except:
                pass

        elif ws:
            self.ws = ws
            self.zs = ws_to_zs(ws, self.MWs)
            try:
                self.Vfls = zs_to_Vfs(self.zs, self.Vmls)
            except:
                pass
            try:
                self.Vfgs = zs_to_Vfs(self.zs, self.Vmgs)
            except:
                pass
        elif Vfls:
            self.zs = Vfs_to_zs(Vfls, self.Vmls)
            self.ws = zs_to_ws(self.zs, self.MWs)
            try:
                self.Vfls = zs_to_Vfs(self.zs, self.Vmls)
            except:
                pass
            try:
                self.Vfgs = zs_to_Vfs(self.zs, self.Vmgs)
            except:
                pass
        elif Vfgs:
            self.zs = Vfs_to_zs(Vfgs, self.Vmgs)
            self.ws = zs_to_ws(self.zs, self.MWs)
            try:
                self.Vfls = zs_to_Vfs(self.zs, self.Vmls)
            except:
                pass
            try:
                self.Vfgs = zs_to_Vfs(self.zs, self.Vmgs)
            except:
                pass
        else:
            raise Exception('No composition provided')

        self.MW = mixing_simple(self.zs, self.MWs)
        self.set_none()
        self.set_constant_sources()
        self.set_constants()

        self.set_T_sources()
        self.set_T()
        self.set_phase()

    def set_none(self):
        # Null values as necessary
        self.ks = None
        self.Vms = None
        self.rhos = None
        self.rhol = None
        self.Cps = None
        self.Cpsm = None
        self.xs = None
        self.ys = None
        self.phase = None
        self.V_over_F = None
        self.isentropic_exponent = None
        self.conductivity = None
        self.Hm = None
        self.H = None

    def set_chemical_constants(self):
        self.Tms = [i.Tm for i in self.Chemicals]
        self.Tbs = [i.Tb for i in self.Chemicals]

        # Critical Point
        self.Tcs = [i.Tc for i in self.Chemicals]
        self.Pcs = [i.Pc for i in self.Chemicals]
        self.Vcs = [i.Vc for i in self.Chemicals]
        self.omegas = [i.omega for i in self.Chemicals]
        self.StielPolars = [i.StielPolar for i in self.Chemicals]

        self.Zcs = [i.Zc for i in self.Chemicals]
        self.rhoCs = [i.rhoC for i in self.Chemicals]
        self.rhoCms = [i.rhoCm for i in self.Chemicals]

        # Triple point
        self.Pts = [i.Pt for i in self.Chemicals]
        self.Tts = [i.Tt for i in self.Chemicals]

        # Chemistry
        self.Hfs = [i.Hf for i in self.Chemicals]
        self.Hcs = [i.Hc for i in self.Chemicals]

        # Fire Safety Limits
        self.Tflashs = [i.Tflash for i in self.Chemicals]
        self.Tautoignitions = [i.Tautoignition for i in self.Chemicals]
        self.LFLs = [i.LFL for i in self.Chemicals]
        self.UFLs = [i.UFL for i in self.Chemicals]

        # Chemical Exposure Limits
        self.TWAs = [i.TWA for i in self.Chemicals]
        self.STELs = [i.STEL for i in self.Chemicals]
        self.Ceilings = [i.Ceiling for i in self.Chemicals]
        self.Skins = [i.Skin for i in self.Chemicals]
        self.Carcinogens = [i.Carcinogen for i in self.Chemicals]

        # Misc
        self.dipoles = [i.dipole for i in self.Chemicals]
        self.molecular_diameters = [i.molecular_diameter for i in self.Chemicals]
        self.Stockmayers = [i.Stockmayer for i in self.Chemicals]

        # Environmental
        self.GWPs = [i.GWP for i in self.Chemicals]
        self.ODPs = [i.ODP for i in self.Chemicals]
        self.logPs = [i.logP for i in self.Chemicals]

        # Legal
        self.legal_statuses = [i.legal_status for i in self.Chemicals]
        self.economic_statuses = [i.economic_status for i in self.Chemicals]

    def set_chemical_T(self):
        # Tempearture and Pressure Denepdence
        # Get and choose initial methods
        # TODO: Solids?
        for i in self.Chemicals:
            i.set_T(self.T)
        self.Psats = [i.Psat for i in self.Chemicals]

        self.Vmls = [i.Vml for i in self.Chemicals]
        self.rhols = [i.rhol for i in self.Chemicals]
        self.rholms = [i.rholm for i in self.Chemicals]
        self.Zls = [i.Zl for i in self.Chemicals]
        self.Vmgs = [i.Vmg for i in self.Chemicals]
        self.rhogs = [i.rhog for i in self.Chemicals]
        self.rhogms = [i.rhogm for i in self.Chemicals]
        self.Zgs = [i.Zg for i in self.Chemicals]
        self.isobaric_expansion_ls = [i.isobaric_expansion_l for i in self.Chemicals]
        self.isobaric_expansion_gs = [i.isobaric_expansion_g for i in self.Chemicals]

        self.Cpls = [i.Cpl for i in self.Chemicals]
        self.Cpgs = [i.Cpg for i in self.Chemicals]
        self.Cvgs = [i.Cvg for i in self.Chemicals]
        self.Cplms = [i.Cplm for i in self.Chemicals]
        self.Cpgms = [i.Cpgm for i in self.Chemicals]
        self.Cvgms = [i.Cvgm for i in self.Chemicals]
        self.isentropic_exponents = [i.isentropic_exponent for i in self.Chemicals]

        self.Hvaps = [i.Hvap for i in self.Chemicals]
        self.Hfuss = [i.Hfus for i in self.Chemicals]
        self.Hsubs = [i.Hsub for i in self.Chemicals]
        self.Hvapms = [i.Hvapm for i in self.Chemicals]
        self.Hfusms = [i.Hfusm for i in self.Chemicals]
        self.Hsubms = [i.Hsubm for i in self.Chemicals]

        self.muls = [i.mul for i in self.Chemicals]
        self.mugs = [i.mug for i in self.Chemicals]
        self.kls = [i.kl for i in self.Chemicals]
        self.kgs = [i.kg for i in self.Chemicals]
        self.sigmas = [i.sigma for i in self.Chemicals]
        self.solubility_parameters = [i.solubility_parameter for i in self.Chemicals]
        self.permittivites = [i.permittivity for i in self.Chemicals]

        self.Prls = [i.Prl for i in self.Chemicals]
        self.Prgs = [i.Prg for i in self.Chemicals]
        self.alphals = [i.alphal for i in self.Chemicals]
        self.alphags = [i.alphag for i in self.Chemicals]

        self.Hs = [i.H for i in self.Chemicals]
        self.Hms = [i.Hm for i in self.Chemicals]

    def set_constant_sources(self):
        # Tliquidus assumes worst-case for now
        self.Tm_methods = Tliquidus(Tms=self.Tms, ws=self.ws, xs=self.zs, CASRNs=self.CASs, AvailableMethods=True)
        self.Tm_method = self.Tm_methods[0]

        # Critical Point, Methods only for Tc, Pc, Vc
        self.Tc_methods = Tc_mixture(Tcs=self.Tcs, zs=self.zs, CASRNs=self.CASs, AvailableMethods=True)
        self.Tc_method = self.Tc_methods[0]
        self.Pc_methods = Pc_mixture(Pcs=self.Pcs, zs=self.zs, CASRNs=self.CASs, AvailableMethods=True)
        self.Pc_method = self.Pc_methods[0]
        self.Vc_methods = Vc_mixture(Vcs=self.Vcs, zs=self.zs, CASRNs=self.CASs, AvailableMethods=True)
        self.Vc_method = self.Vc_methods[0]
        self.omega_methods = omega_mixture(omegas=self.omegas, zs=self.zs, CASRNs=self.CASs, AvailableMethods=True)
        self.omega_method = self.omega_methods[0]

        # No Flammability limits
        self.LFL_methods = LFL_mixture(ys=self.zs, LFLs=self.LFLs, AvailableMethods=True)
        self.LFL_method = self.LFL_methods[0]
        self.UFL_methods = UFL_mixture(ys=self.zs, UFLs=self.UFLs, AvailableMethods=True)
        self.UFL_method = self.UFL_methods[0]
        # No triple point
        # Mixed Hf linear
        # Exposure limits are minimum of any of them or lower

    def set_constants(self):
        # Melting point
        self.Tm = Tliquidus(Tms=self.Tms, ws=self.ws, xs=self.zs, CASRNs=self.CASs, Method=self.Tm_method)
        # Critical Point
        self.Tc = Tc_mixture(Tcs=self.Tcs, zs=self.zs, CASRNs=self.CASs, Method=self.Tc_method)
        self.Pc = Pc_mixture(Pcs=self.Pcs, zs=self.zs, CASRNs=self.CASs, Method=self.Pc_method)
        self.Vc = Vc_mixture(Vcs=self.Vcs, zs=self.zs, CASRNs=self.CASs, Method=self.Vc_method)
        self.omega = omega_mixture(omegas=self.omegas, zs=self.zs, CASRNs=self.CASs, Method=self.omega_method)

        if all((self.Tc, self.Pc, self.Vc)):
            self.Zc = Z(self.Tc, self.Pc, self.Vc)
        else:
            self.Zc = None

        if self.Vc:
            self.rhoC = Vm_to_rho(self.Vc, self.MW)
        else:
            self.rhoC = None
        if self.Vc: self.rhoCm = 1./self.Vc

        self.LFL = LFL_mixture(ys=self.zs, LFLs=self.LFLs, Method=self.LFL_method)
        self.UFL = UFL_mixture(ys=self.zs, UFLs=self.UFLs, Method=self.UFL_method)


    def set_T_sources(self):
        # Tempearture and Pressure Denepdence
        # No vapor pressure (bubble-dew points)

        self.Vl_methods = volume_liquid_mixture(xs=self.zs, ws=self.ws, Vms=self.Vmls, T=self.T, MWs=self.MWs, MW=self.MW, Tcs=self.Tcs, Pcs=self.Pcs, Vcs=self.Vcs, Zcs=self.Zcs, omegas=self.omegas, Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega, CASRNs=self.CASs, AvailableMethods=True)
        self.Vl_method = self.Vl_methods[0]
#
        self.Vg_methods = volume_gas_mixture(ys=self.zs, Vms=self.Vmgs, T=self.T, P=self.P, Tc=self.Tc, Pc=self.Pc, omega=self.omega, MW=self.MW, CASRNs=self.CASs, AvailableMethods=True)
        self.Vg_method = self.Vg_methods[0]

        # No solid density, or heat capacity
        # No Hvap, no Hsub, no Hfus

        self.Cpl_methods = Cp_liq_mixture(zs=self.zs, ws=self.ws, Cps=self.Cpls, T=self.T, CASRNs=self.CASs, AvailableMethods=True)
        self.Cpl_method = self.Cpl_methods[0]

        self.Cpg_methods = Cp_gas_mixture(zs=self.zs, ws=self.ws, Cps=self.Cpgs, CASRNs=self.CASs, AvailableMethods=True)
        self.Cpg_method = self.Cpg_methods[0]

        self.Cvg_methods = Cv_gas_mixture(zs=self.zs, ws=self.ws, Cps=self.Cvgs, CASRNs=self.CASs, AvailableMethods=True)
        self.Cvg_method = self.Cvg_methods[0]

        self.mul_methods = viscosity_liquid_mixture(zs=self.zs, ws=self.ws, mus=self.muls, T=self.T, MW=self.MW, CASRNs=self.CASs, AvailableMethods=True)
        self.mul_method = self.mul_methods[0]

        self.mug_methods = viscosity_gas_mixture(T=self.T, ys=self.zs, ws=self.ws, mus=self.mugs, MWs=self.MWs, molecular_diameters=self.molecular_diameters, Stockmayers=self.Stockmayers, CASRNs=self.CASs, AvailableMethods=True)
        self.mug_method = self.mug_methods[0]

        self.kl_methods = thermal_conductivity_liquid_mixture(T=self.T, P=self.P, zs=self.zs, ws=self.ws, ks=self.kls, CASRNs=self.CASs, AvailableMethods=True)
        self.kl_method = self.kl_methods[0]

        self.kg_methods = thermal_conductivity_gas_mixture(T=self.T, ys=self.zs, ws=self.ws, ks=self.kgs, mus=self.mugs, Tbs=self.Tbs, MWs=self.MWs, CASRNs=self.CASs, AvailableMethods=True)
        self.kg_method = self.kg_methods[0]

        self.sigma_methods = surface_tension_mixture(xs=self.zs, sigmas=self.sigmas, rhoms=self.rholms, CASRNs=self.CASs, AvailableMethods=True)
        self.sigma_method = self.sigma_methods[0]


    def set_T(self, T=None):
        if T:
            self.T = T
        self.set_chemical_T()

        self.Vml = volume_liquid_mixture(xs=self.zs, ws=self.ws, Vms=self.Vmls, T=self.T, MWs=self.MWs, MW=self.MW, Tcs=self.Tcs, Pcs=self.Pcs, Vcs=self.Vcs, Zcs=self.Zcs, omegas=self.omegas, Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega, CASRNs=self.CASs, Molar=True, Method=self.Vl_method)
        if self.Vml:
            self.rhol = Vm_to_rho(self.Vml, self.MW)
        else:
            self.rhol = None

        if self.Vml:
            self.Zl = Z(self.T, self.P, self.Vml)
        else:
            self.Zl = None

        if self.Vml: self.rholm = 1./self.Vml

        self.Vmg = volume_gas_mixture(ys=self.zs, Vms=self.Vmgs, T=self.T, P=self.P, Tc=self.Tc, Pc=self.Pc, omega=self.omega, MW=self.MW, CASRNs=self.CASs, Method=self.Vg_method)
        if self.Vmg:
            self.rhog = Vm_to_rho(self.Vmg, self.MW)
        else:
            self.rhog = None

        if self.Vmg:
            self.Zg = Z(self.T, self.P, self.Vmg)
        else:
            self.Zg = None
        if self.Vmg: self.rhogm = 1./self.Vmg

        # Coefficient of isobaric_expansion_coefficient
        for i in self.Chemicals:
            i.set_T(self.T+0.01)
        _Vmls_2 = [i.Vml for i in self.Chemicals]
        _Vmgs_2 = [i.Vmg for i in self.Chemicals]

        _Vml_2 = volume_liquid_mixture(xs=self.zs, ws=self.ws, Vms=_Vmls_2, T=self.T+0.01, MWs=self.MWs, MW=self.MW, Tcs=self.Tcs, Pcs=self.Pcs, Vcs=self.Vcs, Zcs=self.Zcs,  Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega, omegas=self.omegas,  CASRNs=self.CASs, Molar=True, Method=self.Vl_method)
        _Vmg_2 = volume_gas_mixture(ys=self.zs, Vms=_Vmgs_2, T=self.T+0.01, P=self.P, Tc=self.Tc, Pc=self.Pc, omega=self.omega, MW=self.MW, CASRNs=self.CASs, Method=self.Vg_method)
        self.isobaric_expansion_l = isobaric_expansion(V1=self.Vml, dT=0.01, V2=_Vml_2)
        self.isobaric_expansion_g = isobaric_expansion(V1=self.Vmg, dT=0.01, V2=_Vmg_2)
        for i in self.Chemicals:
            i.set_T(self.T)


        self.Cpl = Cp_liq_mixture(zs=self.zs, ws=self.ws, Cps=self.Cpls, T=self.T, CASRNs=self.CASs, Method=self.Cpl_method)
        self.Cpg = Cp_gas_mixture(zs=self.zs, ws=self.ws, Cps=self.Cpgs, CASRNs=self.CASs, Method=self.Cpg_method)
        self.Cvg = Cv_gas_mixture(zs=self.zs, ws=self.ws, Cps=self.Cvgs, CASRNs=self.CASs, Method=self.Cvg_method)
        self.Cpgm = property_mass_to_molar(self.Cpg, self.MW)
        self.Cplm = property_mass_to_molar(self.Cpl, self.MW)
        self.Cvgm = property_mass_to_molar(self.Cvg, self.MW)

        if self.Cpg and self.Cvg:
            self.isentropic_exponent = isentropic_exponent(self.Cpg, self.Cvg)
        else:
            self.isentropic_exponent = None

        self.mul = viscosity_liquid_mixture(zs=self.zs, ws=self.ws, mus=self.muls, T=self.T, MW=self.MW, CASRNs=self.CASs, Method=self.mul_method)
        self.mug = viscosity_gas_mixture(T=self.T, ys=self.zs, ws=self.ws, mus=self.mugs, MWs=self.MWs, molecular_diameters=self.molecular_diameters, Stockmayers=self.Stockmayers, CASRNs=self.CASs, Method=self.mug_method)
        self.kl = thermal_conductivity_liquid_mixture(T=self.T, P=self.P, zs=self.zs, ws=self.ws, ks=self.kls, CASRNs=self.CASs, Method=self.kl_method)
        self.kg = thermal_conductivity_gas_mixture(T=self.T, ys=self.zs, ws=self.ws, ks=self.kgs, mus=self.mugs, Tbs=self.Tbs, MWs=self.MWs, CASRNs=self.CASs, Method=self.kg_method)

        self.sigma = surface_tension_mixture(xs=self.zs, sigmas=self.sigmas, rhoms=self.rholms, CASRNs=self.CASs, Method=self.sigma_method)

        self.JTl = JT(T=self.T, V=self.Vml, Cp=self.Cplm, isobaric_expansion=self.isobaric_expansion_l)
        self.JTg = JT(T=self.T, V=self.Vmg, Cp=self.Cpgm, isobaric_expansion=self.isobaric_expansion_g)

        try:
            self.nul = nu_mu_converter(mu=self.mul, rho=self.rhol)
        except Exception:
            self.nul = None
        try:
            self.nug = nu_mu_converter(mu=self.mug, rho=self.rhog)
        except Exception:
            self.nug = None

        try:
            self.Prl = Prandtl(Cp=self.Cpl, mu=self.mul, k=self.kl)
        except Exception:
            self.Prl = None
        try:
            self.Prg = Prandtl(Cp=self.Cpg, mu=self.mug, k=self.kg)
        except Exception:
            self.Prg = None

        try:
            self.alphal = thermal_diffusivity(k=self.kl, rho=self.rhol, Cp=self.Cpl)
        except Exception:
            self.alphal = None
        try:
            self.alphag = thermal_diffusivity(k=self.kg, rho=self.rhog, Cp=self.Cpg)
        except Exception:
            self.alphag = None


    def set_phase(self):
        self.phase_methods = identify_phase_mixture(T=self.T, P=self.P, zs=self.zs, Tcs=self.Tcs, Pcs=self.Pcs, Psats=self.Psats, CASRNs=self.CASs, AvailableMethods=True)
        self.phase_method = self.phase_methods[0]
        self.phase, self.xs, self.ys, self.V_over_F = identify_phase_mixture(T=self.T, P=self.P, zs=self.zs, Tcs=self.Tcs, Pcs=self.Pcs, Psats=self.Psats, CASRNs=self.CASs, Method=self.phase_method)
        self.Pbubble_methods = Pbubble_mixture(T=self.T, zs=self.zs, Psats=self.Psats, CASRNs=self.CASs, AvailableMethods=True)
        self.Pbubble_method = self.Pbubble_methods[0]
        self.Pbubble = Pbubble_mixture(T=self.T, zs=self.zs, Psats=self.Psats, CASRNs=self.CASs, Method=self.Pbubble_method)

        self.Pdew_methods = Pdew_mixture(T=self.T, zs=self.zs, Psats=self.Psats, CASRNs=self.CASs, AvailableMethods=True)
        self.Pdew_method = self.Pdew_methods[0]
        self.Pdew = Pdew_mixture(T=self.T, zs=self.zs, Psats=self.Psats, CASRNs=self.CASs, Method=self.Pdew_method)

        self.rho = phase_set_property(phase=self.phase, s=self.rhos, l=self.rhol, g=self.rhog, V_over_F=self.V_over_F)
        self.Vm = phase_set_property(phase=self.phase, s=self.Vms, l=self.Vml, g=self.Vmg, V_over_F=self.V_over_F)
        if self.Vm:
            self.Z = Z(self.T, self.P, self.Vm)
        else:
            self.Z = None

        self.Cp = phase_set_property(phase=self.phase, s=self.Cps, l=self.Cpl, g=self.Cpg, V_over_F=self.V_over_F)
        self.Cpm = phase_set_property(phase=self.phase, s=self.Cpsm, l=self.Cplm, g=self.Cpgm, V_over_F=self.V_over_F)

        self.k = phase_set_property(phase=self.phase, s=self.ks, l=self.kl, g=self.kg)
        self.mu = phase_set_property(phase=self.phase, l=self.mul, g=self.mug)
        self.nu = phase_set_property(phase=self.phase, l=self.nul, g=self.nug)
        self.Pr = phase_set_property(phase=self.phase, l=self.Prl, g=self.Prg)
        self.alpha = phase_set_property(phase=self.phase, l=self.alphal, g=self.alphag)
        self.isobaric_expansion = phase_set_property(phase=self.phase, l=self.isobaric_expansion_l, g=self.isobaric_expansion_g)
        self.JT = phase_set_property(phase=self.phase, l=self.JTl, g=self.JTg)

        if all(self.Hs):
            self.H = mixing_simple(self.Hs, self.ws)
        if all(self.Hms):
            self.Hm = mixing_simple(self.Hms, self.ws)

    def Reynolds(self, V=None, D=None):
        return Reynolds(V=V, D=D, rho=self.rho, mu=self.mu)

    def Capillary(self, V=None):
        return Capillary(V=V, mu=self.mu, sigma=self.sigma)

    def Weber(self, V=None, D=None):
        return Weber(V=V, L=D, rho=self.rho, sigma=self.sigma)

    def Bond(self, L=None):
        return Bond(rhol=self.rhol, rhog=self.rhog, sigma=self.sigma, L=L)

    def Jakob(self, Tw=None):
        return Jakob(Cp=self.Cp, Hvap=self.Hvap, Te=Tw-self.T)

    def Grashof(self, Tw=None, L=None):
        return Grashof(L=L, beta=self.isobaric_expansion, T1=Tw, T2=self.T,
                       rho=self.rho, mu=self.mu)

    def Peclet_heat(self, V=None, D=None):
        return Peclet_heat(V=V, L=D, rho=self.rho, Cp=self.Cp, k=self.k)



class Stream(Mixture): # pragma: no cover

    def __init__(self, IDs, zs=None, ws=None, Vfls=None, Vfgs=None,
                 m=None, Q=None, T=298.15, P=101325):
        Mixture.__init__(self, IDs, zs=zs, ws=ws, Vfls=Vfls, Vfgs=Vfgs,
                 T=T, P=P)
        # TODO: Molar total input.
        if self.phase:
            if Q:
                self.Q = Q
                self.m = self.rho*Q
            else:
                self.m = m
                self.Q = self.m/self.rho
        else:
            raise Exception('phase algorithm failed')


#fluid_51 = Stream(IDs='natural gas', m=2E5/3600, T=273.15+93)
#print fluid_51.H
#
#fluid_51 = Stream(IDs='natural gas', m=2E5/3600, T=273.15+65)
#print fluid_51.H
#
#fluid_51.set_T(273.15+93)
#print fluid_51.H
#
#print Chemical('Ethylene glycol', T-273.15+40)