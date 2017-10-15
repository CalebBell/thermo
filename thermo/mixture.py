# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['Mixture']

from thermo.chemical import Chemical
from thermo.identifiers import *
from thermo.identifiers import _MixtureDict
from thermo.phase_change import Tliquidus
from thermo.activity import identify_phase_mixture, Pbubble_mixture, Pdew_mixture

from thermo.critical import Tc_mixture, Pc_mixture, Vc_mixture
from thermo.acentric import omega_mixture
from thermo.thermal_conductivity import ThermalConductivityLiquidMixture, ThermalConductivityGasMixture
from thermo.volume import VolumeLiquidMixture, VolumeGasMixture, VolumeSolidMixture
from thermo.permittivity import *
from thermo.heat_capacity import HeatCapacitySolidMixture, HeatCapacityGasMixture, HeatCapacityLiquidMixture
from thermo.interface import SurfaceTensionMixture
from thermo.viscosity import ViscosityLiquidMixture, ViscosityGasMixture
from thermo.safety import LFL_mixture, UFL_mixture
from thermo.utils import *
from fluids.core import Reynolds, Capillary, Weber, Bond, Grashof, Peclet_heat
from thermo.elements import atom_fractions, mass_fractions, simple_formula_parser, molecular_weight
from thermo.eos import *
from thermo.eos_mix import *


from fluids.core import *
from scipy.optimize import newton
import numpy as np

# RDKIT
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
except:
    # pragma: no cover
    pass


from collections import Counter
from pprint import pprint


class Mixture(object): 
    '''Creates a Mixture object which contains basic information such as 
    molecular weight and the structure of the species, as well as thermodynamic
    and transport properties as a function of temperature and pressure.
    
    The components of the mixture must be specified by specifying the names of
    the chemicals; the composition can be specified by providing any one of the
    following parameters:
        
    * Mass fractions `ws`
    * Mole fractions `zs`
    * Liquid volume fractions (based on pure component densities) `Vfls`
    * Gas volume fractions (based on pure component densities) `Vfgs`
    
    If volume fractions are provided, by default the pure component volumes
    are calculated at the specified `T` and `P`. To use another reference 
    temperature and pressure specify it as a tuple for the argument `Vf_TP`. 

    Parameters
    ----------
    IDs : list
        List of chemical identifiers - names, CAS numbers, SMILES or InChi 
        strings can all be recognized and may be mixed [-]
    zs : list, optional
        Mole fractions of all components in the mixture [-]
    ws : list, optional
        Mass fractions of all components in the mixture [-]
    Vfls : list, optional
        Volume fractions of all components as a hypothetical liquid phase based 
        on pure component densities [-]
    Vfgs : list, optional
        Volume fractions of all components as a hypothetical gas phase based 
        on pure component densities [-]
    T : float, optional
        Temperature of the chemical (default 298.15 K), [K]
    P : float, optional
        Pressure of the chemical (default 101325 Pa) [Pa]
    Vf_TP : tuple(2, float), optional
        The (T, P) at which the volume fractions are specified to be at, [K] 
        and [Pa]
    
    
    Attributes
    ----------
    MW : float
        Mole-weighted average molecular weight all chemicals in the mixture, 
        [g/mol]
    MWs : list of float
        Molecular weights of all chemicals in the mixture, [g/mol]
    Tms : list of float
        Melting temperatures of all chemicals in the mixture, [K]
    Tbs : list of float
        Boiling temperatures of all chemicals in the mixture, [K]
    Tcs : list of float
        Critical temperatures of all chemicals in the mixture, [K]
    Pcs : list of float
        Critical pressures of all chemicals in the mixture, [Pa]
    Vcs : list of float
        Critical volumes of all chemicals in the mixture, [m^3/mol]
    Zcs : list of float
        Critical compressibilities of all chemicals in the mixture, [-]
    rhocs : list of float
        Critical densities of all chemicals in the mixture, [kg/m^3]
    rhocms : list of float
        Critical molar densities of all chemicals in the mixture, [mol/m^3]
    omegas : list of float
        Acentric factors of all chemicals in the mixture, [-]
    StielPolars : list of float
        Stiel Polar factors of all chemicals in the mixture, 
        see :obj:`thermo.acentric.StielPolar` for the definition, [-]
    Tts : list of float
        Triple temperatures of all chemicals in the mixture, [K]
    Pts : list of float
        Triple pressures of all chemicals in the mixture, [Pa]
    Hfuss : list of float
        Enthalpy of fusions of all chemicals in the mixture, [J/kg]
    Hfusms : list of float
        Molar enthalpy of fusions of all chemicals in the mixture, [J/mol]
    Hsubs : list of float
        Enthalpy of sublimations of all chemicals in the mixture, [J/kg]
    Hsubms : list of float
        Molar enthalpy of sublimations of all chemicals in the mixture, [J/mol]
    Hfs : list of float
        Enthalpy of formations of all chemicals in the mixture, [J/mol]
    Hcs : list of float
        Molar enthalpy of combustions of all chemicals in the mixture, [J/mol]
    Tflashs : list of float
        Flash points of all chemicals in the mixture, [K]
    Tautoignitions : list of float
        Autoignition points of all chemicals in the mixture, [K]
    LFLs : list of float
        Lower flammability limits of the gases in an atmosphere at STP, mole 
        fractions, [-]
    UFLs : list of float
        Upper flammability limit of the gases in an atmosphere at STP, mole 
        fractions, [-]
    TWAs : list of list of tuple(quantity, unit)
        Time-Weighted Average limits on worker exposure to dangerous chemicals.
    STELs : list of tuple(quantity, unit)
        Short-term Exposure limits on worker exposure to dangerous chemicals.
    Ceilings : list of tuple(quantity, unit)
        Ceiling limits on worker exposure to dangerous chemicals.
    Skins : list of bool
        Whether or not each of the chemicals can be absorbed through the skin.
    Carcinogens : list of str or dict
        Carcinogen status information for each chemical in the mixture.
    dipoles : list of float
        Dipole moments of all chemicals in the mixture in debye, 
        [3.33564095198e-30 ampere*second^2]
    Stockmayers : list of float
        Lennard-Jones depth of potential-energy minimum over k for all 
        chemicals in the mixture, [K]
    molecular_diameters : list of float
        Lennard-Jones molecular diameters of all chemicals in the mixture,
        [angstrom]
    GWPs : list of float
        Global warming potentials (default 100-year outlook) (impact/mass 
        chemical)/(impact/mass CO2) of all chemicals in the mixture, [-]
    ODPs : list of float
        Ozone Depletion potentials (impact/mass chemical)/(impact/mass CFC-11),
        of all chemicals in the mixture, [-]
    logPs : list of float
        Octanol-water partition coefficients of all chemicals in the mixture,
        [-]
    Psat_298s : list of float
        Vapor pressure of the chemicals in the mixture at 298.15 K, [Pa]
    phase_STPs : list of str
        Phase of the chemicals in the mixture at 298.15 K and 101325 Pa; one of
        's', 'l', 'g', or 'l/g'.
    Vml_Tbs : list of float
        Molar volumes of the chemicals in the mixture as liquids at their 
        normal boiling points, [m^3/mol]
    Vml_Tms : list of float
        Molar volumes of the chemicals in the mixture as liquids at their 
        melting points, [m^3/mol]
    Vml_STPs : list of float
        Molar volume of the chemicals in the mixture as liquids at 298.15 K and
        101325 Pa, [m^3/mol]
    Vmg_STPs : list of float
        Molar volume of the chemicals in the mixture as gases at 298.15 K and 
        101325 Pa, [m^3/mol]
    Hvap_Tbms : list of float
        Molar enthalpies of vaporization of the chemicals in the mixture at 
        their normal boiling points, [J/mol]
    Hvap_Tbs : list of float
        Mass enthalpies of vaporization of the chemicals in the mixture at
        their normal boiling points, [J/kg]
    alpha
    alphag
    alphags
    alphal
    alphals
    atom_fractions
    atom_fractionss
    atomss
    Bvirial
    charges
    Cp
    Cpg
    Cpgm
    Cpgms
    Cpgs
    Cpl
    Cplm
    Cplms
    Cpls
    Cpm
    Cps
    Cpsm
    Cpsms
    Cpss
    Cvg
    Cvgm
    Cvgms
    Cvgs
    economic_statuses
    eos
    formulas
    Hvapms
    Hvaps
    InChI_Keys
    InChIs
    isentropic_exponent
    isentropic_exponents
    isobaric_expansion
    isobaric_expansion_gs
    isobaric_expansion_ls
    IUPAC_names
    JT
    JTg
    JTgs
    JTl
    JTls
    k
    kg
    kgs
    kl
    kls
    legal_statuses
    mass_fractions
    mass_fractionss
    mu
    mug
    mugs
    mul
    muls
    nu
    nug
    nugs
    nul
    nuls
    permittivites
    Pr
    Prg
    Prgs
    Prl
    Prls
    Psats
    PSRK_groups
    PubChems
    rho
    rhog
    rhogm
    rhogms
    rhogm_STP
    rhogs
    rhog_STP
    rhol
    rholm
    rholms
    rholm_STP
    rhols
    rhol_STP
    rhom
    rhosms
    rhoss
    ringss
    sigma
    sigmas
    smiless
    solubility_parameters
    synonymss
    UNIFAC_Dortmund_groups
    UNIFAC_groups
    Vm
    Vmg
    Vmgs
    Vmg_STP
    Vml
    Vmls
    Vml_STP
    Vmss
    Z
    Zg
    Zgs
    Zg_STP
    Zl
    Zls
    Zl_STP
    Zss

    Examples
    --------
    Creating Mixture objects:
        
    >>> Mixture(['water', 'ethanol'], Vfls=[.6, .4], T=300, P=1E5)
    <Mixture, components=['water', 'ethanol'], mole fractions=[0.8299, 0.1701], T=300.00 K, P=100000 Pa>
    '''
    eos_in_a_box = []
    ks = None
    Vms = None
    rhos = None
    xs = None
    ys = None
    phase = None
    V_over_F = None
    conductivity = None
    Hm = None
    H = None
    isobaric_expansion_g = None
    isobaric_expansion_l = None

    def __repr__(self):
        return '<Mixture, components=%s, mole fractions=%s, T=%.2f K, P=%.0f \
Pa>' % (self.names, [round(i,4) for i in self.zs], self.T, self.P)

    def __init__(self, IDs, zs=None, ws=None, Vfls=None, Vfgs=None,
                 T=298.15, P=101325, Vf_TP=(None, None)):
        self.P = P
        self.T = T

        if hasattr(IDs, 'strip') or (isinstance(IDs, list) and len(IDs) == 1):
            try:
                mixname = mixture_from_any(IDs)
                _d = _MixtureDict[mixname]
                IDs = _d["CASs"]
                ws = _d["ws"]
                self.mixname = mixname
                self.mixsource = _d["Source"]
            except:
                if hasattr(IDs, 'strip'):
                    IDs = [IDs]
                    zs = [1]
                elif isinstance(IDs, list) and len(IDs) == 1:
                    pass
                else:
                    raise Exception('Could not recognize the mixture IDs')

        # Handle numpy array inputs; also turn mutable inputs into copies
        if zs is not None:
            zs = list(zs)
            length_matching = len(zs) == len(IDs)
        elif ws is not None:
            ws = list(ws)
            length_matching = len(ws) == len(IDs)
        elif Vfls is not None:
            Vfls = list(Vfls)
            length_matching = len(Vfls) == len(IDs)
        elif Vfgs is not None:
            Vfgs = list(Vfgs)
            length_matching = len(Vfgs) == len(IDs)
        else:
            raise Exception("One of 'zs', 'ws', 'Vfls', or 'Vfgs' is required to define the mixture")
        if not length_matching:
            raise Exception('Composition is not the same length as the component identifiers')


        self.components = tuple(IDs)
        self.Chemicals = [Chemical(component, P=P, T=T) for component in self.components]
        self.names = [i.name for i in self.Chemicals]
        self.MWs = [i.MW for i in self.Chemicals]
        self.CASs = [i.CAS for i in self.Chemicals]

        # Required for densities for volume fractions before setting fractions
        self.set_chemical_constants()
        self.set_chemical_TP()
        self.set_Chemical_property_objects()

        if zs:
            self.zs = zs if sum(zs) == 1 else [zi/sum(zs) for zi in zs]
            self.ws = zs_to_ws(zs, self.MWs)
        elif ws:
            self.ws = ws if sum(ws) == 1 else [wi/sum(ws) for wi in ws]
            self.zs = ws_to_zs(ws, self.MWs)
        elif Vfls or Vfgs:
            T_vf, P_vf = Vf_TP
            if T_vf is None: 
                T_vf = T
            if P_vf is None: 
                P_vf = P

            if Vfls:
                Vfs = Vfls if sum(Vfls) == 1 else [Vfli/sum(Vfls) for Vfli in Vfls]
                VolumeObjects = self.VolumeLiquids
                Vms_TP = self.Vmls
            else:
                Vfs = Vfgs if sum(Vfgs) == 1 else [Vfgi/sum(Vfgs) for Vfgi in Vfgs]
                VolumeObjects = self.VolumeGases
                Vms_TP = self.Vmgs

            if T_vf != T or P_vf != P:
                Vms_TP = [i(T_vf, P_vf) for i in VolumeObjects]
            self.zs = Vfs_to_zs(Vfs, Vms_TP)
            self.ws = zs_to_ws(self.zs, self.MWs)
        else:
            raise Exception('One of mole fractions `zs`, weight fractions `ws`,'
                            ' pure component liquid volume fractions `Vfls`, or'
                            ' pure component gas volume fractions `Vfgs` must '
                            'be provided.')

        self.MW = mixing_simple(self.zs, self.MWs)
        self.set_constant_sources()
        self.set_constants()
        self.set_TP_sources()

        self.set_TP()
        self.set_phase()


    def set_chemical_constants(self):
        # Set lists of everything set by Chemical.set_constants
        self.Tms = [i.Tm for i in self.Chemicals]
        self.Tbs = [i.Tb for i in self.Chemicals]

        # Critical Point
        self.Tcs = [i.Tc for i in self.Chemicals]
        self.Pcs = [i.Pc for i in self.Chemicals]
        self.Vcs = [i.Vc for i in self.Chemicals]
        self.omegas = [i.omega for i in self.Chemicals]
        self.StielPolars = [i.StielPolar for i in self.Chemicals]

        self.Zcs = [i.Zc for i in self.Chemicals]
        self.rhocs = [i.rhoc for i in self.Chemicals]
        self.rhocms = [i.rhocm for i in self.Chemicals]

        # Triple point
        self.Pts = [i.Pt for i in self.Chemicals]
        self.Tts = [i.Tt for i in self.Chemicals]

        # Enthalpy
        self.Hfuss = [i.Hfus for i in self.Chemicals]
        self.Hsubs = [i.Hsub for i in self.Chemicals]
        self.Hfusms = [i.Hfusm for i in self.Chemicals]
        self.Hsubms = [i.Hsubm for i in self.Chemicals]

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


        # Analytical
        self.RIs = [i.RI for i in self.Chemicals]
        self.conductivities = [i.conductivity for i in self.Chemicals]

        # Constant properties obtained from TP
        self.Vml_STPs = [i.Vml_STP for i in self.Chemicals]
        self.Vmg_STPs = [i.Vmg_STP for i in self.Chemicals]

        self.Psat_298s = [i.Psat_298 for i in self.Chemicals]
        self.phase_STPs = [i.phase_STP for i in self.Chemicals]
        self.Vml_Tbs = [i.Vml_Tb for i in self.Chemicals]
        self.Vml_Tms = [i.Vml_Tm for i in self.Chemicals]
        self.Hvap_Tbms = [i.Hvap_Tbm for i in self.Chemicals]
        self.Hvap_Tbs = [i.Hvap_Tb for i in self.Chemicals]

    ### More stuff here

    def set_chemical_TP(self):
        # Tempearture and Pressure Denepdence
        # Get and choose initial methods
        [i.calculate(self.T, self.P) for i in self.Chemicals]

        try:
            self.Hs = [i.H for i in self.Chemicals]
            self.Hms = [i.Hm for i in self.Chemicals]

            self.Ss = [i.S for i in self.Chemicals]
            self.Sms = [i.Sm for i in self.Chemicals]
            # Ignore G, A, U - which depend on molar volume
        except:
            self.Hs = None
            self.Hsm = None
            self.Ss = None
            self.Sms = None

    def set_constant_sources(self):
        # None of this takes much time or is important
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
        # None of this takes much time or is important
        # Melting point
        self.Tm = Tliquidus(Tms=self.Tms, ws=self.ws, xs=self.zs, CASRNs=self.CASs, Method=self.Tm_method)
        # Critical Point
        self.Tc = Tc_mixture(Tcs=self.Tcs, zs=self.zs, CASRNs=self.CASs, Method=self.Tc_method)
        self.Pc = Pc_mixture(Pcs=self.Pcs, zs=self.zs, CASRNs=self.CASs, Method=self.Pc_method)
        self.Vc = Vc_mixture(Vcs=self.Vcs, zs=self.zs, CASRNs=self.CASs, Method=self.Vc_method)
        self.omega = omega_mixture(omegas=self.omegas, zs=self.zs, CASRNs=self.CASs, Method=self.omega_method)

        self.Zc = Z(self.Tc, self.Pc, self.Vc) if all((self.Tc, self.Pc, self.Vc)) else None
        self.rhoc = Vm_to_rho(self.Vc, self.MW) if self.Vc else None
        self.rhocm = 1./self.Vc if self.Vc else None

        self.LFL = LFL_mixture(ys=self.zs, LFLs=self.LFLs, Method=self.LFL_method)
        self.UFL = UFL_mixture(ys=self.zs, UFLs=self.UFLs, Method=self.UFL_method)

    def set_eos(self, T, P, eos=PRMIX):
        try:
            self.eos = eos(T=T, P=P, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, zs=self.zs)
        except:
            # Handle overflow errors and so on
            self.eos = GCEOS_DUMMY(T=T, P=P)

    @property
    def eos(self):
        r'''Equation of state object held by the mixture. See :
        obj:`thermo.eos_mix` for a full listing.

        Examples
        --------
        '''
        return self.eos_in_a_box[0]

    @eos.setter
    def eos(self, eos):
        if self.eos_in_a_box:
            self.eos_in_a_box.pop()
        self.eos_in_a_box.append(eos)

    def set_Chemical_property_objects(self):
        self.VolumeSolids = [i.VolumeSolid for i in self.Chemicals]
        self.VolumeLiquids = [i.VolumeLiquid for i in self.Chemicals]
        self.VolumeGases = [i.VolumeGas for i in self.Chemicals]
        self.HeatCapacitySolids = [i.HeatCapacitySolid for i in self.Chemicals]
        self.HeatCapacityLiquids = [i.HeatCapacityLiquid for i in self.Chemicals]
        self.HeatCapacityGases = [i.HeatCapacityGas for i in self.Chemicals]
        self.ViscosityLiquids = [i.ViscosityLiquid for i in self.Chemicals]
        self.ViscosityGases = [i.ViscosityGas for i in self.Chemicals]
        self.ThermalConductivityLiquids = [i.ThermalConductivityLiquid for i in self.Chemicals]
        self.ThermalConductivityGases = [i.ThermalConductivityGas for i in self.Chemicals]
        self.SurfaceTensions = [i.SurfaceTension for i in self.Chemicals]
        self.Permittivities = [i.Permittivity for i in self.Chemicals]

        self.VaporPressures = [i.VaporPressure for i in self.Chemicals]
        self.EnthalpyVaporizations = [i.EnthalpyVaporization for i in self.Chemicals]

    def set_TP_sources(self):

        self.VolumeSolidMixture = VolumeSolidMixture(CASs=self.CASs, VolumeSolids=self.VolumeSolids)
        self.VolumeLiquidMixture = VolumeLiquidMixture(MWs=self.MWs, Tcs=self.Tcs, Pcs=self.Pcs, Vcs=self.Vcs, Zcs=self.Zcs, omegas=self.omegas, CASs=self.CASs, VolumeLiquids=self.VolumeLiquids)
        self.VolumeGasMixture = VolumeGasMixture(eos=self.eos_in_a_box, CASs=self.CASs, VolumeGases=self.VolumeGases)

        self.HeatCapacityLiquidMixture = HeatCapacityLiquidMixture(MWs=self.MWs, CASs=self.CASs, HeatCapacityLiquids=self.HeatCapacityLiquids)
        self.HeatCapacityGasMixture = HeatCapacityGasMixture(CASs=self.CASs, HeatCapacityGases=self.HeatCapacityGases)
        self.HeatCapacitySolidMixture = HeatCapacitySolidMixture(CASs=self.CASs, HeatCapacitySolids=self.HeatCapacitySolids)

        self.ViscosityLiquidMixture = ViscosityLiquidMixture(CASs=self.CASs, ViscosityLiquids=self.ViscosityLiquids)
        self.ViscosityGasMixture = ViscosityGasMixture(MWs=self.MWs, molecular_diameters=self.molecular_diameters, Stockmayers=self.Stockmayers, CASs=self.CASs, ViscosityGases=self.ViscosityGases)

        self.ThermalConductivityLiquidMixture = ThermalConductivityLiquidMixture(CASs=self.CASs, ThermalConductivityLiquids=self.ThermalConductivityLiquids)
        self.ThermalConductivityGasMixture = ThermalConductivityGasMixture(MWs=self.MWs, Tbs=self.Tbs, CASs=self.CASs, ThermalConductivityGases=self.ThermalConductivityGases, ViscosityGases=self.ViscosityGases)

        self.SurfaceTensionMixture = SurfaceTensionMixture(MWs=self.MWs, Tbs=self.Tbs, Tcs=self.Tcs, CASs=self.CASs, SurfaceTensions=self.SurfaceTensions, VolumeLiquids=self.VolumeLiquids)

    def set_TP(self, T=None, P=None):
        if T:
            self.T = T
        if P:
            self.P = P
        self.set_chemical_TP()
        self.set_eos(T=self.T, P=self.P)

    def set_phase(self):
        try:
            self.phase_methods = identify_phase_mixture(T=self.T, P=self.P, zs=self.zs, Tcs=self.Tcs, Pcs=self.Pcs, Psats=self.Psats, CASRNs=self.CASs, AvailableMethods=True)
            self.phase_method = self.phase_methods[0]
            self.phase, self.xs, self.ys, self.V_over_F = identify_phase_mixture(T=self.T, P=self.P, zs=self.zs, Tcs=self.Tcs, Pcs=self.Pcs, Psats=self.Psats, CASRNs=self.CASs, Method=self.phase_method)

            if self.phase == 'two-phase':
                self.wsl = zs_to_ws(self.xs, self.MWs)
                self.wsg = zs_to_ws(self.ys, self.MWs)
                
                ng = self.V_over_F
                nl = (1. - self.V_over_F)
                self.MWl = mixing_simple(self.xs, self.MWs)
                self.MWg = mixing_simple(self.ys, self.MWs)
                self.x = self.quality = ng*self.MWg/(nl*self.MWl + ng*self.MWg)

            self.Pbubble_methods = Pbubble_mixture(T=self.T, zs=self.zs, Psats=self.Psats, CASRNs=self.CASs, AvailableMethods=True)
            self.Pbubble_method = self.Pbubble_methods[0]
            self.Pbubble = Pbubble_mixture(T=self.T, zs=self.zs, Psats=self.Psats, CASRNs=self.CASs, Method=self.Pbubble_method)

            self.Pdew_methods = Pdew_mixture(T=self.T, zs=self.zs, Psats=self.Psats, CASRNs=self.CASs, AvailableMethods=True)
            self.Pdew_method = self.Pdew_methods[0]
            self.Pdew = Pdew_mixture(T=self.T, zs=self.zs, Psats=self.Psats, CASRNs=self.CASs, Method=self.Pdew_method)
            if not None in self.Hs:
                self.H = mixing_simple(self.Hs, self.ws)
                self.Hm = property_mass_to_molar(self.H, self.MW)

            if not None in self.Ss:
                # Ideal gas contribution
                self.Sm = mixing_simple(self.Sms, self.zs) - R*sum([zi*log(zi) for zi in self.zs if zi > 0])
                self.S = property_molar_to_mass(self.Sm, self.MW)
        except:
            pass

    def calculate(self, T=None, P=None):
        if T:
            if T < 0:
                raise Exception('Negative value specified for Mixture temperature - aborting!')
            self.T = T
        else:
            T = self.T
        if P:
            if P < 0:
                raise Exception('Negative value specified for Mixture pressure - aborting!')
        else:
            P = self.P
        self.set_TP(T=T, P=P)
        self.set_phase()

    def calculate_TH(self, T, H):
        def to_solve(P):
            self.calculate(T, P)
            return self.H - H
        return newton(to_solve, self.P)

    def calculate_PH(self, P, H):
        def to_solve(T):
            self.calculate(T, P)
            return self.H - H
        return newton(to_solve, self.T)

    def calculate_TS(self, T, S):
        def to_solve(P):
            self.calculate(T, P)
            return self.S - S
        return newton(to_solve, self.P)

    def calculate_PS(self, P, S):
        def to_solve(T):
            self.calculate(T, P)
            return self.S - S
        return newton(to_solve, self.T)


    def Vfls(self, T=None, P=None):
        r'''Volume fractions of all species in a hypothetical pure-liquid phase 
        at the current or specified temperature and pressure. If temperature 
        or pressure are specified, the non-specified property is assumed to be 
        that of the mixture. Note this is a method, not a property. Volume 
        fractions are calculated based on **pure species volumes only**.

        Examples
        --------
        >>> Mixture(['hexane', 'pentane'], zs=[.5, .5], T=315).Vfls()
        [0.5299671144566751, 0.47003288554332484]
        
        >>> S = Mixture(['hexane', 'decane'], zs=[0.25, 0.75])
        >>> S.Vfls(298.16, 101326)
        [0.18301434895886864, 0.8169856510411313]
        '''
        if (T is None or T == self.T) and (P is None or P == self.P):
            Vmls = self.Vmls
        else:
            if T is None: T = self.T
            if P is None: P = self.P
            Vmls = [i(T, P) for i in self.VolumeLiquids]
        if none_and_length_check([Vmls]):
            return zs_to_Vfs(self.zs, Vmls)
        return None


    def Vfgs(self, T=None, P=None):
        r'''Volume fractions of all species in a hypothetical pure-gas phase 
        at the current or specified temperature and pressure. If temperature 
        or pressure are specified, the non-specified property is assumed to be 
        that of the mixture. Note this is a method, not a property. Volume 
        fractions are calculated based on **pure species volumes only**.

        Examples
        --------
        >>> Mixture(['sulfur hexafluoride', 'methane'], zs=[.2, .9], T=315).Vfgs()
        [0.18062059238682632, 0.8193794076131737]
        
        >>> S = Mixture(['sulfur hexafluoride', 'methane'], zs=[.1, .9])
        >>> S.Vfgs(P=1E2)
        [0.0999987466608421, 0.9000012533391578]
        '''
        if (T is None or T == self.T) and (P is None or P == self.P):
            Vmgs = self.Vmgs
        else:
            if T is None: T = self.T
            if P is None: P = self.P
            Vmgs = [i(T, P) for i in self.VolumeGases]
        if none_and_length_check([Vmgs]):
            return zs_to_Vfs(self.zs, Vmgs)
        return None


    # Unimportant constants
    @property
    def PubChems(self):
        r'''PubChem Component ID numbers for all chemicals in the mixture.

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5]).PubChems
        [241, 1140]
        '''
        return [i.PubChem for i in self.Chemicals]

    @property
    def formulas(self):
        r'''Chemical formulas for all chemicals in the mixture.

        Examples
        --------
        >>> Mixture(['ethanol', 'trichloroethylene', 'furfuryl alcohol'],
        ... ws=[0.5, 0.2, 0.3]).formulas
        ['C2H6O', 'C2HCl3', 'C5H6O2']
        '''
        return [i.formula for i in self.Chemicals]

    @property
    def smiless(self):
        r'''SMILES strings for all chemicals in the mixture.

        Examples
        --------
        >>> Mixture(['methane', 'ethane', 'propane', 'butane'],
        ... zs=[0.25, 0.25, 0.25, 0.25]).smiless
        ['C', 'CC', 'CCC', 'CCCC']
        '''
        return [i.smiles for i in self.Chemicals]

    @property
    def InChIs(self):
        r'''InChI strings for all chemicals in the mixture.

        Examples
        --------
        >>> Mixture(['methane', 'ethane', 'propane', 'butane'],
        ... zs=[0.25, 0.25, 0.25, 0.25]).InChIs
        ['CH4/h1H4', 'C2H6/c1-2/h1-2H3', 'C3H8/c1-3-2/h3H2,1-2H3', 'C4H10/c1-3-4-2/h3-4H2,1-2H3']
        '''
        return [i.InChI for i in self.Chemicals]

    @property
    def InChI_Keys(self):
        r'''InChI keys for all chemicals in the mixture.

        Examples
        --------
        >>> Mixture(['1-nonene'], zs=[1]).InChI_Keys
        ['JRZJOMJEPLMPRA-UHFFFAOYSA-N']
        '''
        return [i.InChI_Key for i in self.Chemicals]

    @property
    def IUPAC_names(self):
        r'''IUPAC names for all chemicals in the mixture.

        Examples
        --------
        >>> Mixture(['1-hexene', '1-nonene'], zs=[.7, .3]).IUPAC_names
        ['hex-1-ene', 'non-1-ene']
        '''
        return [i.IUPAC_name for i in self.Chemicals]

    @property
    def synonymss(self):
        r'''Lists of synonyms for all chemicals in the mixture.

        Examples
        --------
        >>> Mixture(['Tetradecene', 'Pentadecene'], zs=[.1, .9]).synonymss
        [['tetradec-2-ene', 'tetradecene', '2-tetradecene', 'tetradec-2-ene', '26952-13-6', '35953-53-8', '1652-97-7'], ['pentadec-1-ene', '1-pentadecene', 'pentadecene,1-', 'pentadec-1-ene', '13360-61-7', 'pentadecene']]
        '''
        return [i.synonyms for i in self.Chemicals]

    @property
    def charges(self):
        r'''Charges for all chemicals in the mixture, [faraday].

        Examples
        --------
        >>> Mixture(['water', 'sodium ion', 'chloride ion'], zs=[.9, .05, .05]).charges
        [0, 1, -1]
        '''
        return [i.charge for i in self.Chemicals]

    @property
    def similarity_variables(self):
        r'''Similarity variables for all chemicals in the mixture, see 
        :obj:`thermo.elements.similarity_variable` for the definition, [mol/g]

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5]).similarity_variables
        [0.15362587797189262, 0.16279853724428964]
        '''
        return [i.similarity_variable for i in self.Chemicals]

    @property
    def atomss(self):
        r'''List of dictionaries of atom counts for all chemicals in the mixture.

        Examples
        --------
        >>> Mixture(['nitrogen', 'oxygen'], zs=[.01, .99]).atomss
        [{'N': 2}, {'O': 2}]
        '''
        return [i.atoms for i in self.Chemicals]

    @property
    def ringss(self):
        r'''List of ring counts for all chemicals in the mixture.

        Examples
        --------
        >>> Mixture(['Docetaxel', 'Paclitaxel'], zs=[.5, .5]).ringss
        [6, 7]
        '''
        return [i.rings for i in self.Chemicals]

    @property
    def atom_fractionss(self):
        r'''List of dictionaries of atomic fractions for all chemicals in the
        mixture.

        Examples
        --------
        >>> Mixture(['oxygen', 'nitrogen'], zs=[.5, .5]).atom_fractionss
        [{'O': 1.0}, {'N': 1.0}]
        '''
        return [i.atom_fractions for i in self.Chemicals]

    @property
    def atom_fractions(self):
        r'''Dictionary of atomic fractions for each atom in the mixture.

        Examples
        --------
        >>> Mixture(['CO2', 'O2'], zs=[0.5, 0.5]).atom_fractions
        {'C': 0.2, 'O': 0.8}
        '''
        things = dict()
        for zi, atoms in zip(self.zs, self.atomss):
            for atom, count in atoms.iteritems():
                if atom in things:
                    things[atom] += zi*count
                else:
                    things[atom] = zi*count

        tot = sum(things.values())
        return {atom : value/tot for atom, value in things.iteritems()}

    @property
    def mass_fractionss(self):
        r'''List of dictionaries of mass fractions for all chemicals in the mixture.

        Examples
        --------
        >>> Mixture(['oxygen', 'nitrogen'], zs=[.5, .5]).mass_fractionss
        [{'O': 1.0}, {'N': 1.0}]
        '''
        return [i.mass_fractions for i in self.Chemicals]

    @property
    def mass_fractions(self):
        r'''Dictionary of mass fractions for each atom in the mixture.

        Examples
        --------
        >>> Mixture(['CO2', 'O2'], zs=[0.5, 0.5]).mass_fractions
        {'C': 0.15801826905745822, 'O': 0.8419817309425419}
        '''
        things = dict()
        for zi, atoms in zip(self.zs, self.atomss):
            for atom, count in atoms.iteritems():
                if atom in things:
                    things[atom] += zi*count
                else:
                    things[atom] = zi*count
        return mass_fractions(things)


    @property
    def legal_statuses(self):
        r'''List of dictionaries of the legal status for all chemicals in the
        mixture.

        Examples
        --------
        >>> pprint(Mixture(['oxygen', 'nitrogen'], zs=[.5, .5]).legal_statuses)
        [{'DSL': 'LISTED',
          'EINECS': 'LISTED',
          'NLP': 'UNLISTED',
          'SPIN': 'LISTED',
          'TSCA': 'LISTED'},
         {'DSL': 'LISTED',
          'EINECS': 'LISTED',
          'NLP': 'UNLISTED',
          'SPIN': 'LISTED',
          'TSCA': 'LISTED'}]
        '''
        return [i.legal_status for i in self.Chemicals]

    @property
    def economic_statuses(self):
        r'''List of dictionaries of the economic status for all chemicals in
        the mixture.

        Examples
        --------
        >>> pprint(Mixture(['o-xylene', 'm-xylene'], zs=[.5, .5]).economic_statuses)
        [["US public: {'Manufactured': 0.0, 'Imported': 0.0, 'Exported': 0.0}",
          u'100,000 - 1,000,000 tonnes per annum',
          'OECD HPV Chemicals'],
         ["US public: {'Manufactured': 39.805, 'Imported': 0.0, 'Exported': 0.0}",
          u'100,000 - 1,000,000 tonnes per annum',
          'OECD HPV Chemicals']]
        '''
        return [i.economic_status for i in self.Chemicals]

    @property
    def UNIFAC_groups(self):
        r'''List of dictionaries of UNIFAC subgroup: count groups for each chemical in the mixture. Uses the original
        UNIFAC subgroups, as determined by `DDBST's online service <http://www.ddbst.com/unifacga.html>`_.

        Examples
        --------
        >>> pprint(Mixture(['1-pentanol', 'decane'], ws=[0.5, 0.5]).UNIFAC_groups)
        [{1: 1, 2: 4, 14: 1}, {1: 2, 2: 8}]
        '''
        return [i.UNIFAC_groups for i in self.Chemicals]

    @property
    def UNIFAC_Dortmund_groups(self):
        r'''List of dictionaries of Dortmund UNIFAC subgroup: count groups for each chemcial in the mixture. Uses the
        Dortmund UNIFAC subgroups, as determined by `DDBST's online service <http://www.ddbst.com/unifacga.html>`_.

        Examples
        --------
        >>> pprint(Mixture(['1-pentanol', 'decane'], ws=[0.5, 0.5]).UNIFAC_Dortmund_groups)
        [{1: 1, 2: 4, 14: 1}, {1: 2, 2: 8}]
        '''
        return [i.UNIFAC_Dortmund_groups for i in self.Chemicals]

    @property
    def PSRK_groups(self):
        r'''List of dictionaries of PSRK subgroup: count groups for each chemical in the mixture. Uses the PSRK subgroups,
        as determined by `DDBST's online service <http://www.ddbst.com/unifacga.html>`_.

        Examples
        --------
        >>> pprint(Mixture(['1-pentanol', 'decane'], ws=[0.5, 0.5]).PSRK_groups)
        [{1: 1, 2: 4, 14: 1}, {1: 2, 2: 8}]
        '''
        return [i.PSRK_groups for i in self.Chemicals]

    @property
    def R_specific(self):
        r'''Specific gas constant of the mixture, in units of [J/kg/K].

        Examples
        --------
        >>> Mixture(['N2', 'O2'], zs=[0.79, .21]).R_specific
        288.1928437986195
        '''
        return property_molar_to_mass(R, self.MW)

    @property
    def charge_balance(self):
        r'''Charge imbalance of the mixture, in units of [faraday].
        Mixtures meeting the electroneutrality condition will have an imbalance
        of 0.

        Examples
        --------
        >>> Mixture(['Na+', 'Cl-', 'water'], zs=[.01, .01, .98]).charge_balance
        0.0
        '''
        return sum([zi*ci for zi, ci in zip(self.zs, self.charges)])
        
    ### One phase properties - calculate lazily
    @property
    def Psats(self):
        r'''Pure component vapor pressures of the chemicals in the mixture at
        its current temperature, in units of [Pa].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Psats
        [32029.25774454549, 10724.419010511821]
        '''
        return [i.Psat for i in self.Chemicals]

    @property
    def Hvapms(self):
        r'''Pure component enthalpies of vaporization of the chemicals in the
        mixture at its current temperature, in units of [J/mol].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Hvapms
        [32639.806783391632, 36851.7902195611]
        '''
        return [i.Hvapm for i in self.Chemicals]

    @property
    def Hvaps(self):
        r'''Enthalpy of vaporization of the chemicals in the mixture at its
        current temperature, in units of [J/kg].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Hvaps
        [417859.9144942896, 399961.16950519773]
        '''
        return [i.Hvap for i in self.Chemicals]

    @property
    def Cpsms(self):
        r'''Solid-phase pure component heat capacity of the chemicals in the
        mixture at its current temperature, in units of [J/mol/K].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Cpsms
        [109.77384365511931, 135.22614707678474]
        '''
        return [i.Cpsm for i in self.Chemicals]

    @property
    def Cplms(self):
        r'''Liquid-phase pure component heat capacity of the chemicals in the
        mixture at its current temperature, in units of [J/mol/K].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Cplms
        [140.9113971170526, 163.62584810669068]
        '''
        return [i.Cplm for i in self.Chemicals]

    @property
    def Cpgms(self):
        r'''Gas-phase ideal gas heat capacity of the chemicals at its current
        temperature, in units of [J/mol/K].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Cpgms
        [89.55804092586159, 111.70390334788907]
        '''
        return [i.Cpgm for i in self.Chemicals]

    @property
    def Cpss(self):
        r'''Solid-phase pure component heat capacity of the chemicals in the
        mixture at its current temperature, in units of [J/kg/K].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Cpss
        [1405.341925822248, 1467.6412627521154]
        '''
        return [i.Cps for i in self.Chemicals]

    @property
    def Cpls(self):
        r'''Liquid-phase pure component heat capacity of the chemicals in the
        mixture at its  current temperature, in units of [J/kg/K].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Cpls
        [1803.9697581961016, 1775.869915141704]
        '''
        return [i.Cpl for i in self.Chemicals]

    @property
    def Cpgs(self):
        r'''Gas-phase pure component heat capacity of the chemicals in the
        mixture at its current temperature, in units of [J/kg/K].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Cpgs
        [1146.5360555565146, 1212.3488046342566]
        '''
        return [i.Cpg for i in self.Chemicals]

    @property
    def Cvgms(self):
        r'''Gas-phase pure component ideal-gas contant-volume heat capacities
        of the chemicals in the mixture at its current temperature, in units
        of [J/mol/K].  Subtracts R from the ideal-gas heat capacities; does not
        include pressure-compensation from an equation of state.

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Cvgms
        [81.2435811258616, 103.38944354788907]
        '''
        return [i.Cvgm for i in self.Chemicals]

    @property
    def Cvgs(self):
        r'''Gas-phase pure component ideal-gas contant-volume heat capacities
        of the chemicals in the mixture at its current temperature, in units of
        [J/kg/K]. Subtracts R from the ideal-gas heat capacity; does not include
        pressure-compensation from an equation of state.

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Cvgs
        [1040.093040003431, 1122.1100117398266]
        '''
        return [i.Cvg for i in self.Chemicals]

    @property
    def isentropic_exponents(self):
        r'''Gas-phase pure component ideal-gas isentropic exponent of the
        chemicals in the  mixture at its current temperature, [dimensionless].
         Does not include pressure-compensation from an equation of state.

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).isentropic_exponents
        [1.1023398979313739, 1.080418846592871]
        '''
        return [i.isentropic_exponent for i in self.Chemicals]

    @property
    def Vmss(self):
        r'''Pure component solid-phase molar volumes of the chemicals in the
        mixture at its current temperature, in units of [mol/m^3].

        Examples
        --------
        >>> Mixture(['iron'], ws=[1], T=320).Vmss
        [7.09593392630242e-06]
        '''
        return [i.Vms for i in self.Chemicals]

    @property
    def Vmls(self):
        r'''Pure component liquid-phase molar volumes of the chemicals in the
        mixture at its current temperature and pressure, in units of [mol/m^3].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Vmls
        [9.188896727673715e-05, 0.00010946199496993461]
        '''
        return [i.Vml for i in self.Chemicals]

    @property
    def Vmgs(self):
        r'''Pure component gas-phase molar volumes of the chemicals in the
        mixture at its current temperature and pressure, in units of [mol/m^3].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Vmgs
        [0.024929001982294974, 0.024150186467130488]
        '''
        return [i.Vmg for i in self.Chemicals]

    @property
    def rhoss(self):
        r'''Pure component solid-phase mass density of the chemicals in the
        mixture at its  current temperature, in units of [kg/m^3].

        Examples
        --------
        >>> Mixture(['iron'], ws=[1], T=320).rhoss
        [7869.999999999994]
        '''
        return [i.rhos for i in self.Chemicals]

    @property
    def rhols(self):
        r'''Pure-component liquid-phase mass density of the chemicals in the
        mixture at its current temperature and pressure, in units of [kg/m^3].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).rhols
        [850.0676666084917, 841.7389069631628]
        '''
        return [i.rhol for i in self.Chemicals]

    @property
    def rhogs(self):
        r'''Pure-component gas-phase mass densities of the chemicals in the
        mixture at its current temperature and pressure, in units of [kg/m^3].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).rhogs
        [3.1333721283939258, 3.8152260283954584]
        '''
        return [i.rhog for i in self.Chemicals]

    @property
    def rhosms(self):
        r'''Pure component molar densities of the chemicals in the solid phase
        at the current temperature and pressure, in units of [mol/m^3].

        Examples
        --------
        >>> Mixture(['iron'], ws=[1], T=320).rhosms
        [140925.7767033753]
        '''
        return [i.rhosm for i in self.Chemicals]

    @property
    def rholms(self):
        r'''Pure component molar densities of the chemicals in the mixture in
        the liquid phase at the current temperature and pressure, in units of
        [mol/m^3].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).rholms
        [10882.699301520635, 9135.590853014008]
        '''
        return [i.rholm for i in self.Chemicals]

    @property
    def rhogms(self):
        r'''Pure component molar densities of the chemicals in the gas phase at
        the current temperature and pressure, in units of [mol/m^3].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).rhogms
        [40.11392035309789, 41.407547778608084]
        '''
        return [i.rhogm for i in self.Chemicals]


    @property
    def Zss(self):
        r'''Pure component compressibility factors of the chemicals in the
        mixture in the solid phase at the current temperature and pressure,
        [dimensionless].

        Examples
        --------
        >>> Mixture(['palladium'], ws=[1]).Zss
        [0.00036248477437931853]
        '''
        return [i.Zs for i in self.Chemicals]

    @property
    def Zls(self):
        r'''Pure component compressibility factors of the chemicals in the
        liquid phase at the current temperature and pressure, [dimensionless].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Zls
        [0.0034994191720201235, 0.004168655010037687]
        '''
        return [i.Zl for i in self.Chemicals]

    @property
    def Zgs(self):
        r'''Pure component compressibility factors of the chemicals in the
        mixture in the gas phase at the current temperature and pressure,
        [dimensionless].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Zgs
        [0.9493743379816593, 0.9197146081359057]
        '''
        return [i.Zg for i in self.Chemicals]

    @property
    def SGs(self):
        r'''Specific gravity of a hypothetical solid phase of the mixture at the 
        specified temperature and pressure, [dimensionless].
        The reference condition is water at 4 °C and 1 atm 
        (rho=999.017 kg/m^3). The SG varries with temperature and pressure
        but only very slightly.
        '''
        rhos = self.rhos
        if rhos is not None:
            return SG(rhos)
        return None

    @property
    def SGl(self):
        r'''Specific gravity of a hypothetical liquid phase of the mixture at  
        the specified temperature and pressure, [dimensionless].
        The reference condition is water at 4 °C and 1 atm 
        (rho=999.017 kg/m^3). For liquids, SG is defined that the reference
        chemical's T and P are fixed, but the chemical itself varies with
        the specified T and P.
        
        Examples
        --------
        >>> Mixture('water', ws=[1], T=365).SGl
        0.9650065522428539
        '''
        rhol = self.rhol
        if rhol is not None:
            return SG(rhol)
        return None

    @property
    def isobaric_expansion_ls(self):
        r'''Pure component isobaric (constant-pressure) expansions of the
        chemicals in the mixture in the liquid phase at its current temperature
        and pressure, in units of [1/K].

        .. math::
            \beta = \frac{1}{V}\left(\frac{\partial V}{\partial T} \right)_P

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).isobaric_expansion_ls
        [0.0012736035771253886, 0.0011234157437069571]
        '''
        return [i.isobaric_expansion_l for i in self.Chemicals]

    @property
    def isobaric_expansion_gs(self):
        r'''Pure component isobaric (constant-pressure) expansions of the
        chemicals in the mixture in the gas phase at its current temperature
        and pressure, in units of [1/K].

        .. math::
            \beta = \frac{1}{V}\left(\frac{\partial V}{\partial T} \right)_P

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).isobaric_expansion_gs
        [0.0038091518363900499, 0.0043556759306508453]
        '''
        return [i.isobaric_expansion_g for i in self.Chemicals]

    @property
    def muls(self):
        r'''Pure component viscosities of the chemicals in the mixture in the
        liquid phase at its current temperature and pressure, in units of 
        [Pa*s].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).muls
        [0.00045545522798131764, 0.00043274394349114754]
        '''
        return [i.mul for i in self.Chemicals]

    @property
    def mugs(self):
        r'''Pure component viscosities of the chemicals in the mixture in the
        gas phase at its current temperature and pressure, in units of [Pa*s].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).mugs
        [8.082880451060605e-06, 7.442602145854158e-06]
        '''
        return [i.mug for i in self.Chemicals]

    @property
    def kls(self):
        r'''Pure component thermal conductivities of the chemicals in the
        mixture in the liquid phase at its current temperature and pressure, in
        units of [W/m/K].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).kls
        [0.13391538485205587, 0.12429339088930591]
        '''
        return [i.kl for i in self.Chemicals]


    @property
    def kgs(self):
        r'''Pure component thermal conductivies of the chemicals in the mixture
        in the gas phase at its current temperature and pressure, in units of
        [W/m/K].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).kgs
        [0.011865404482987936, 0.010981336502491088]
        '''
        return [i.kg for i in self.Chemicals]

    @property
    def sigmas(self):
        r'''Pure component surface tensions of the chemicals in the mixture at
        its current temperature, in units of [N/m].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).sigmas
        [0.02533469712937521, 0.025254723406585546]
        '''
        return [i.sigma for i in self.Chemicals]

    @property
    def permittivites(self):
        r'''Pure component relative permittivities of the chemicals in the
        mixture at its current temperature, [dimensionless].

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).permittivites
        [2.23133472, 1.8508128]
        '''
        return [i.permittivity for i in self.Chemicals]

    @property
    def JTls(self):
        r'''Pure component Joule Thomson coefficients of the chemicals in the
        mixture in the liquid phase at its current temperature and pressure, in
        units of [K/Pa].

        .. math::
            \mu_{JT} = \left(\frac{\partial T}{\partial P}\right)_H = \frac{1}{C_p}
            \left[T \left(\frac{\partial V}{\partial T}\right)_P - V\right]
            = \frac{V}{C_p}\left(\beta T-1\right)

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).JTls
        [-3.8633730709853161e-07, -3.464395792560331e-07]
        '''
        return [i.JTl for i in self.Chemicals]

    @property
    def JTgs(self):
        r'''Pure component Joule Thomson coefficients of the chemicals in the
        mixture in the gas phase at its current temperature and pressure, in
        units of [K/Pa].

        .. math::
            \mu_{JT} = \left(\frac{\partial T}{\partial P}\right)_H = \frac{1}{C_p}
            \left[T \left(\frac{\partial V}{\partial T}\right)_P - V\right]
            = \frac{V}{C_p}\left(\beta T-1\right)

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).JTgs
        [6.0940046688790938e-05, 4.1290005523287549e-05]
        '''
        return [i.JTg for i in self.Chemicals]

    @property
    def nuls(self):
        r'''Pure component kinematic viscosities of the liquid phase of the
        chemicals in the mixture at its current temperature and pressure, in
        units of [m^2/s].

        .. math::
            \nu = \frac{\mu}{\rho}

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).nuls
        [5.357870271650772e-07, 3.8129130341250897e-07]
        '''
        return [i.nul for i in self.Chemicals]

    @property
    def nugs(self):
        r'''Pure component kinematic viscosities of the gas phase of the
        chemicals in the mixture at its current temperature and pressure, in
        units of [m^2/s].

        .. math::
            \nu = \frac{\mu}{\rho}

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).nugs
        [2.579610757948387e-06, 1.9149095260590705e-06]
        '''
        return [i.nul for i in self.Chemicals]

    @property
    def alphals(self):
        r'''Pure component thermal diffusivities of the chemicals in the
        mixture in the liquid phase at the current temperature and pressure, in
        units of [m^2/s].

        .. math::
            \alpha = \frac{k}{\rho Cp}

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).alphals
        [8.732683564481583e-08, 7.57355434073289e-08]
        '''
        return [i.alphal for i in self.Chemicals]

    @property
    def alphags(self):
        r'''Pure component thermal diffusivities of the chemicals in the
        mixture in the gas phase at the current temperature and pressure, in
        units of [m^2/s].

        .. math::
            \alpha = \frac{k}{\rho Cp}

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).alphags
        [3.3028044028118324e-06, 2.4413332489215457e-06]
        '''
        return [i.alphag for i in self.Chemicals]

    @property
    def Prls(self):
        r'''Pure component Prandtl numbers of the liquid phase of the chemicals
        in the mixture at its current temperature and pressure, [dimensionless].

        .. math::
            Pr = \frac{C_p \mu}{k}

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).Prls
        [6.13542244155373, 5.034509376420631]
        '''
        return [i.Prl for i in self.Chemicals]

    @property
    def Prgs(self):
        r'''Pure component Prandtl numbers of the gas phase of the chemicals
        in the mixture at its current temperature and pressure, [dimensionless].

        .. math::
            Pr = \frac{C_p \mu}{k}

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).Prgs
        [0.7810364900059606, 0.7843703955226016]
        '''
        return [i.Prg for i in self.Chemicals]

    @property
    def solubility_parameters(self):
        r'''Pure component solubility parameters of the chemicals in the
        mixture at its current temperature and pressure, in units of [Pa^0.5].

        .. math::
            \delta = \sqrt{\frac{\Delta H_{vap} - RT}{V_m}}

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).solubility_parameters
        [18062.51359608708, 14244.12852702228]
        '''
        return [i.solubility_parameter for i in self.Chemicals]

    ### Overall mixture properties
    @property
    def rhol(self):
        r'''Liquid-phase mass density of the mixture at its current
        temperature, pressure, and composition in units of [kg/m^3]. For
        calculation of this property at other temperatures, pressures,
        compositions or specifying manually the method used to calculate it,
        and more - see the object oriented interface
        :obj:`thermo.volume.VolumeLiquidMixture`; each Mixture instance
        creates one to actually perform the calculations. Note that that
        interface provides output in molar units.

        Examples
        --------
        >>> Mixture(['o-xylene'], ws=[1], T=297).rhol
        876.9946785618097
        '''
        Vml = self.Vml
        if Vml:
            return Vm_to_rho(Vml, self.MW)
        return None

    @property
    def rhog(self):
        r'''Gas-phase mass density of the mixture at its current temperature,
        pressure, and composition in units of [kg/m^3]. For calculation of this
        property at other temperatures, pressures, or compositions or
        specifying manually the method used to calculate it, and more - see the
        object oriented interface :obj:`thermo.volume.VolumeGasMixture`; each
        Mixture instance creates one to actually perform the calculations. Note
        that that interface provides output in molar units.

        Examples
        --------
        >>> Mixture(['hexane'], ws=[1], T=300, P=2E5).rhog
        7.914205150685313
        '''
        Vmg = self.Vmg
        if Vmg:
            return Vm_to_rho(Vmg, self.MW)
        return None

    @property
    def rholm(self):
        r'''Molar density of the mixture in the liquid phase at the
        current temperature, pressure, and composition in units of [mol/m^3].

        Utilizes the object oriented interface and
        :obj:`thermo.volume.VolumeLiquidMixture` to perform the actual
        calculation of molar volume.

        Examples
        --------
        >>> Mixture(['water'], ws=[1], T=300).rholm
        55317.352773503124
        '''
        Vml = self.Vml
        if Vml:
            return 1./Vml
        return None

    @property
    def rhogm(self):
        r'''Molar density of the mixture in the gas phase at the
        current temperature, pressure, and composition in units of [mol/m^3].

        Utilizes the object oriented interface and
        :obj:`thermo.volume.VolumeGasMixture` to perform the actual
        calculation of molar volume.

        Examples
        --------
        >>> Mixture(['water'], ws=[1], T=500).rhogm
        24.467426039789093
        '''
        Vmg = self.Vmg
        if Vmg:
            return 1./Vmg
        return None


    @property
    def Zl(self):
        r'''Compressibility factor of the mixture in the liquid phase at the
        current temperature, pressure, and composition, [dimensionless].

        Utilizes the object oriented interface and
        :obj:`thermo.volume.VolumeLiquidMixture` to perform the actual
        calculation of molar volume.

        Examples
        --------
        >>> Mixture(['water'], ws=[1]).Zl
        0.0007385375470263454
        '''
        Vml = self.Vml
        if Vml:
            return Z(self.T, self.P, Vml)
        return None

    @property
    def Zg(self):
        r'''Compressibility factor of the mixture in the gas phase at the
        current temperature, pressure, and composition, [dimensionless].

        Utilizes the object oriented interface and
        :obj:`thermo.volume.VolumeGasMixture` to perform the actual calculation
        of molar volume.

        Examples
        --------
        >>> Mixture(['hexane'], ws=[1], T=300, P=1E5).Zg
        0.9403859376888882
        '''
        Vmg = self.Vmg
        if Vmg:
            return Z(self.T, self.P, Vmg)
        return None

    @property
    def Cpsm(self):
        r'''Solid-phase heat capacity of the mixture at its current temperature
        and composition, in units of [J/mol/K]. For calculation of this property
        at other temperatures or compositions, or specifying manually the
        method used to calculate it, and more - see the object oriented
        interface :obj:`thermo.heat_capacity.HeatCapacitySolidMixture`; each
        Mixture instance creates one to actually perform the calculations.

        Examples
        --------
        >>> Mixture(['silver', 'platinum'], ws=[0.95, 0.05]).Cpsm
        25.32745719036059
        '''
        return self.HeatCapacitySolidMixture(self.T, self.P, self.zs, self.ws)

    @property
    def Cplm(self):
        r'''Liquid-phase heat capacity of the mixture at its current
        temperature and composition, in units of [J/mol/K]. For calculation of
        this property at other temperatures or compositions, or specifying
        manually the method used to calculate it, and more - see the object
        oriented interface :obj:`thermo.heat_capacity.HeatCapacityLiquidMixture`;
        each Mixture instance creates one to actually perform the calculations.

        Examples
        --------
        >>> Mixture(['toluene', 'decane'], ws=[.9, .1], T=300).Cplm
        168.29157865567112
        '''
        return self.HeatCapacityLiquidMixture(self.T, self.P, self.zs, self.ws)

    @property
    def Cpgm(self):
        r'''Gas-phase heat capacity of the mixture at its current temperature
        and composition, in units of [J/mol/K]. For calculation of this property
        at other temperatures or compositions, or specifying manually the
        method used to calculate it, and more - see the object oriented
        interface :obj:`thermo.heat_capacity.HeatCapacityGasMixture`; each
        Mixture instance creates one to actually perform the calculations.

        Examples
        --------
        >>> Mixture(['oxygen', 'nitrogen'], ws=[.4, .6], T=350, P=1E6).Cpgm
        29.361044582498046
        '''
        return self.HeatCapacityGasMixture(self.T, self.P, self.zs, self.ws)

    @property
    def Cps(self):
        r'''Solid-phase heat capacity of the mixture at its current temperature
        and composition, in units of [J/kg/K]. For calculation of this property
        at other temperatures or compositions, or specifying manually the
        method used to calculate it,  and more - see the object oriented
        interface :obj:`thermo.heat_capacity.HeatCapacitySolidMixture`; each
        Mixture instance creates one to actually perform the calculations. Note
        that that interface provides output in molar units.

        Examples
        --------
        >>> Mixture(['silver', 'platinum'], ws=[0.95, 0.05]).Cps
        229.55145722105294
        '''
        Cpsm = self.HeatCapacitySolidMixture(self.T, self.P, self.zs, self.ws)
        if Cpsm:
            return property_molar_to_mass(Cpsm, self.MW)
        return None

    @property
    def Cpl(self):
        r'''Liquid-phase heat capacity of the mixture at its current
        temperature and composition, in units of [J/kg/K]. For calculation of
        this property at other temperatures or compositions, or specifying
        manually the method used to calculate it, and more - see the object
        oriented interface :obj:`thermo.heat_capacity.HeatCapacityLiquidMixture`;
        each Mixture instance creates one to actually perform the calculations.
        Note that that interface provides output in molar units.

        Examples
        --------
        >>> Mixture(['water', 'sodium chloride'], ws=[.9, .1], T=301.5).Cpl
        3735.4604049449786
        '''
        Cplm = self.HeatCapacityLiquidMixture(self.T, self.P, self.zs, self.ws)
        if Cplm:
            return property_molar_to_mass(Cplm, self.MW)
        return None

    @property
    def Cpg(self):
        r'''Gas-phase heat capacity of the mixture at its current temperature ,
        and composition in units of [J/kg/K]. For calculation of this property at
        other temperatures or compositions, or specifying manually the method
        used to calculate it, and more - see the object oriented interface
        :obj:`thermo.heat_capacity.HeatCapacityGasMixture`; each Mixture
        instance creates one to actually perform the calculations. Note that
        that interface provides output in molar units.

        Examples
        --------
        >>> Mixture(['oxygen', 'nitrogen'], ws=[.4, .6], T=350, P=1E6).Cpg
        995.8911053614883
        '''
        Cpgm = self.HeatCapacityGasMixture(self.T, self.P, self.zs, self.ws)
        if Cpgm:
            return property_molar_to_mass(Cpgm, self.MW)
        return None

    @property
    def Cvgm(self):
        r'''Gas-phase ideal-gas contant-volume heat capacity of the mixture at
        its current temperature and composition, in units of [J/mol/K]. Subtracts R from
        the ideal-gas heat capacity; does not include pressure-compensation
        from an equation of state.

        Examples
        --------
        >>> Mixture(['water'], ws=[1], T=520).Cvgm
        27.13366316134193
        '''
        Cpgm = self.HeatCapacityGasMixture(self.T, self.P, self.zs, self.ws)
        if Cpgm:
            return Cpgm - R
        return None

    @property
    def Cvg(self):
        r'''Gas-phase ideal-gas contant-volume heat capacity of the mixture at
        its current temperature, in units of [J/kg/K]. Subtracts R from
        the ideal-gas heat capacity; does not include pressure-compensation
        from an equation of state.

        Examples
        --------
        >>> Mixture(['water'], ws=[1], T=520).Cvg
        1506.1471795798861
        '''
        Cvgm = self.Cvgm
        if Cvgm:
            return property_molar_to_mass(Cvgm, self.MW)
        return None

    @property
    def isentropic_exponent(self):
        r'''Gas-phase ideal-gas isentropic exponent of the mixture at its
        current temperature, [dimensionless]. Does not include
        pressure-compensation from an equation of state.

        Examples
        --------
        >>> Mixture(['hydrogen'], ws=[1]).isentropic_exponent
        1.405237786321222
        '''
        Cp, Cv = self.Cpg, self.Cvg
        if Cp and Cv:
            return isentropic_exponent(Cp, Cv)
        return None

    @property
    def Bvirial(self):
        r'''Second virial coefficient of the gas phase of the mixture at its
        current temperature, pressure, and composition in units of [mol/m^3].

        This property uses the object-oriented interface
        :obj:`thermo.volume.VolumeGasMixture`, converting its result with
        :obj:`thermo.utils.B_from_Z`.

        Examples
        --------
        >>> Mixture(['hexane'], ws=[1], T=300, P=1E5).Bvirial
        -0.0014869761738013018
        '''
        if self.Vmg:
            return B_from_Z(self.Zg, self.T, self.P)
        return None

    @property
    def JTl(self):
        r'''Joule Thomson coefficient of the liquid phase of the mixture if one
        exists at its current temperature and pressure, in units of [K/Pa].

        .. math::
            \mu_{JT} = \left(\frac{\partial T}{\partial P}\right)_H = \frac{1}{C_p}
            \left[T \left(\frac{\partial V}{\partial T}\right)_P - V\right]
            = \frac{V}{C_p}\left(\beta T-1\right)

        Examples
        --------
        >>> Mixture(['dodecane'], ws=[1], T=400).JTl
        -3.193910574559279e-07
        '''
        Vml, Cplm, isobaric_expansion_l = self.Vml, self.Cplm, self.isobaric_expansion_l
        if all((Vml, Cplm, isobaric_expansion_l)):
            return Joule_Thomson(T=self.T, V=Vml, Cp=Cplm, beta=isobaric_expansion_l)
        return None

    @property
    def JTg(self):
        r'''Joule Thomson coefficient of the gas phase of the mixture if one
        exists at its current temperature and pressure, in units of [K/Pa].

        .. math::
            \mu_{JT} = \left(\frac{\partial T}{\partial P}\right)_H = \frac{1}{C_p}
            \left[T \left(\frac{\partial V}{\partial T}\right)_P - V\right]
            = \frac{V}{C_p}\left(\beta T-1\right)

        Examples
        --------
        >>> Mixture(['dodecane'], ws=[1], T=400, P=1000).JTg
        5.4089897835384913e-05
        '''
        Vmg, Cpgm, isobaric_expansion_g = self.Vmg, self.Cpgm, self.isobaric_expansion_g
        if all((Vmg, Cpgm, isobaric_expansion_g)):
            return Joule_Thomson(T=self.T, V=Vmg, Cp=Cpgm, beta=isobaric_expansion_g)
        return None

    @property
    def nul(self):
        r'''Kinematic viscosity of the liquid phase of the mixture if one
        exists at its current temperature and pressure, in units of [m^2/s].

        .. math::
            \nu = \frac{\mu}{\rho}

        Examples
        --------
        >>> Mixture(['methane'], ws=[1], T=110).nul
        2.85818467411866e-07
        '''
        mul, rhol = self.mul, self.rhol
        if all([mul, rhol]):
            return nu_mu_converter(mu=mul, rho=rhol)
        return None

    @property
    def nug(self):
        r'''Kinematic viscosity of the gas phase of the mixture if one exists
        at its current temperature and pressure, in units of [m^2/s].

        .. math::
            \nu = \frac{\mu}{\rho}

        Examples
        --------
        >>> Mixture(['methane'], ws=[1], T=115).nug
        2.5057767760931785e-06
        '''
        mug, rhog = self.mug, self.rhog
        if all([mug, rhog]):
            return nu_mu_converter(mu=mug, rho=rhog)
        return None

    @property
    def alphal(self):
        r'''Thermal diffusivity of the liquid phase of the mixture if one
        exists at its current temperature and pressure, in units of [m^2/s].

        .. math::
            \alpha = \frac{k}{\rho Cp}

        Examples
        --------
        >>> Mixture(['nitrogen'], ws=[1], T=70).alphal
        9.504101801042264e-08
        '''
        kl, rhol, Cpl = self.kl, self.rhol, self.Cpl
        if all([kl, rhol, Cpl]):
            return thermal_diffusivity(k=kl, rho=rhol, Cp=Cpl)
        return None

    @property
    def alphag(self):
        r'''Thermal diffusivity of the gas phase of the mixture if one exists
        at its current temperature and pressure, in units of [m^2/s].

        .. math::
            \alpha = \frac{k}{\rho Cp}

        Examples
        --------
        >>> Mixture(['ammonia'], ws=[1]).alphag
        1.6931865425158556e-05
        '''
        kg, rhog, Cpg = self.kg, self.rhog, self.Cpg
        if all([kg, rhog, Cpg]):
            return thermal_diffusivity(k=kg, rho=rhog, Cp=Cpg)
        return None

    @property
    def Prl(self):
        r'''Prandtl number of the liquid phase of the mixture if one exists at
        its current temperature and pressure, [dimensionless].

        .. math::
            Pr = \frac{C_p \mu}{k}

        Examples
        --------
        >>> Mixture(['nitrogen'], ws=[1], T=70).Prl
        2.7655015690791696
        '''
        Cpl, mul, kl = self.Cpl, self.mul, self.kl
        if all([Cpl, mul, kl]):
            return Prandtl(Cp=Cpl, mu=mul, k=kl)
        return None

    @property
    def Prg(self):
        r'''Prandtl number of the gas phase of the mixture if one exists at its
        current temperature and pressure, [dimensionless].

        .. math::
            Pr = \frac{C_p \mu}{k}

        Examples
        --------
        >>> Mixture(['NH3'], ws=[1]).Prg
        0.8472637319330079
        '''
        Cpg, mug, kg = self.Cpg, self.mug, self.kg
        if all([Cpg, mug, kg]):
            return Prandtl(Cp=Cpg, mu=mug, k=kg)
        return None

    ### Properties from Mixture objects
    @property
    def Vml(self):
        r'''Liquid-phase molar volume of the mixture at its current
        temperature, pressure, and composition in units of [mol/m^3]. For
        calculation of this property at other temperatures or pressures or
        compositions, or specifying manually the method used to calculate it,
        and more - see the object oriented interface
        :obj:`thermo.volume.VolumeLiquidMixture`; each Mixture instance
        creates one to actually perform the calculations.

        Examples
        --------
        >>> Mixture(['cyclobutane'], ws=[1], T=225).Vml
        7.42395423425395e-05
        '''
        return self.VolumeLiquidMixture(T=self.T, P=self.P, zs=self.zs, ws=self.ws)

    @property
    def Vmg(self):
        r'''Gas-phase molar volume of the mixture at its current
        temperature, pressure, and composition in units of [mol/m^3]. For
        calculation of this property at other temperatures or pressures or
        compositions, or specifying manually the method used to calculate it,
        and more - see the object oriented interface
        :obj:`thermo.volume.VolumeGasMixture`; each Mixture instance
        creates one to actually perform the calculations.

        Examples
        --------
        >>> Mixture(['hexane'], ws=[1], T=300, P=2E5).Vmg
        0.010888694235142216
        '''
        return self.VolumeGasMixture(T=self.T, P=self.P, zs=self.zs, ws=self.ws)

    @property
    def SGg(self):
        r'''Specific gravity of a hypothetical gas phase of the mixture, .
        [dimensionless]. The reference condition is air at 15.6 °C (60 °F) and 
        1 atm (rho=1.223 kg/m^3). The definition for gases uses the 
        compressibility factor of the reference gas and the mixture both at the 
        reference conditions, not the conditions of the mixture.
            
        Examples
        --------
        >>> Mixture('argon').SGg
        1.3800407778218216
        '''
        Vmg = self.VolumeGasMixture(T=288.70555555555552, P=101325, zs=self.zs, ws=self.ws)
        if Vmg:
            rho = Vm_to_rho(Vmg, self.MW)
            return SG(rho, rho_ref=1.2231876628642968) # calculated with Mixture
        return None

    @property
    def mul(self):
        r'''Viscosity of the mixture in the liquid phase at its current
        temperature, pressure, and composition in units of [Pa*s].

        For calculation of this property at other temperatures and pressures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface
        :obj:`thermo.viscosity.ViscosityLiquidMixture`; each Mixture instance
        creates one to actually perform the calculations.

        Examples
        --------
        >>> Mixture(['water'], ws=[1], T=320).mul
        0.0005767262693751547
        '''
        return self.ViscosityLiquidMixture(self.T, self.P, self.zs, self.ws)

    @property
    def mug(self):
        r'''Viscosity of the mixture in the gas phase at its current
        temperature, pressure, and composition in units of [Pa*s].

        For calculation of this property at other temperatures and pressures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface
        :obj:`thermo.viscosity.ViscosityGasMixture`; each Mixture instance
        creates one to actually perform the calculations.

        Examples
        --------
        >>> Mixture(['water'], ws=[1], T=500).mug
        1.7298722343367148e-05
        '''
        return self.ViscosityGasMixture(self.T, self.P, self.zs, self.ws)

    @property
    def sigma(self):
        r'''Surface tension of the mixture at its current temperature and
        composition, in units of [N/m].

        For calculation of this property at other temperatures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface :obj:`thermo.interface.SurfaceTensionMixture`;
        each Mixture instance creates one to actually perform the calculations.

        Examples
        --------
        >>> Mixture(['water'], ws=[1], T=300, P=1E5).sigma
        0.07176932405246211
        '''
        return self.SurfaceTensionMixture(self.T, self.P, self.zs, self.ws)

    @property
    def kl(self):
        r'''Thermal conductivity of the mixture in the liquid phase at its current
        temperature, pressure, and composition in units of [Pa*s].

        For calculation of this property at other temperatures and pressures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface
        :obj:`thermo.thermal_conductivity.ThermalConductivityLiquidMixture`;
        each Mixture instance creates one to actually perform the calculations.

        Examples
        --------
        >>> Mixture(['water'], ws=[1], T=320).kl
        0.6369957248212118
        '''
        return self.ThermalConductivityLiquidMixture(self.T, self.P, self.zs, self.ws)

    @property
    def kg(self):
        r'''Thermal conductivity of the mixture in the gas phase at its current
        temperature, pressure, and composition in units of [Pa*s].

        For calculation of this property at other temperatures and pressures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface
        :obj:`thermo.thermal_conductivity.ThermalConductivityGasMixture`;
        each Mixture instance creates one to actually perform the calculations.

        Examples
        --------
        >>> Mixture(['water'], ws=[1], T=500).kg
        0.036035173297862676
        '''
        return self.ThermalConductivityGasMixture(self.T, self.P, self.zs, self.ws)

    ### Single-phase properties

    @property
    def Cp(self):
        r'''Mass heat capacity of the mixture at its current phase and
        temperature, in units of [J/kg/K].

        Examples
        --------
        >>> w = Mixture(['water'], ws=[1])
        >>> w.Cp, w.phase
        (4180.597021827336, 'l')
        >>> Pd = Mixture(['palladium'], ws=[1])
        >>> Pd.Cp, Pd.phase
        (234.26767209171211, 's')
        '''
        return phase_select_property(phase=self.phase, s=self.Cps, l=self.Cpl, g=self.Cpg)

    @property
    def Cpm(self):
        r'''Molar heat capacity of the mixture at its current phase and
        temperature, in units of [J/mol/K]. Available only if single phase.

        Examples
        --------
        >>> Mixture(['ethylbenzene'], ws=[1], T=550, P=3E6).Cpm
        294.18449553310046
        '''
        return phase_select_property(phase=self.phase, s=self.Cpsm, l=self.Cplm, g=self.Cpgm)

    @property
    def Vm(self):
        r'''Molar volume of the mixture at its current phase and
        temperature and pressure, in units of [m^3/mol].
        Available only if single phase.

        Examples
        --------
        >>> Mixture(['ethylbenzene'], ws=[1], T=550, P=3E6).Vm
        0.00017758024401627633
        '''
        return phase_select_property(phase=self.phase, s=self.Vms, l=self.Vml, g=self.Vmg)

    @property
    def rho(self):
        r'''Mass density of the mixture at its current phase and
        temperature and pressure, in units of [kg/m^3].
        Available only if single phase.

        Examples
        --------
        >>> Mixture(['decane'], ws=[1], T=550, P=2E6).rho
        498.6549441720744
        '''
        return phase_select_property(phase=self.phase, s=self.rhos, l=self.rhol, g=self.rhog)

    @property
    def rhom(self):
        r'''Molar density of the mixture at its current phase and
        temperature and pressure, in units of [mol/m^3].
        Available only if single phase.

        Examples
        --------
        >>> Mixture(['1-hexanol'], ws=[1]).rhom
        7853.086232143972
        '''
        return phase_select_property(phase=self.phase, s=self.rhosm, l=self.rholm, g=self.rhogm)

    @property
    def Z(self):
        r'''Compressibility factor of the mixture at its current phase and
        temperature and pressure, [dimensionless].
        Available only if single phase.

        Examples
        --------
        >>> Mixture(['MTBE'], ws=[1], T=900, P=1E-2).Z
        0.9999999999056376
        '''
        Vm = self.Vm
        if Vm:
            return Z(self.T, self.P, Vm)
        return None

    @property
    def SG(self):
        r'''Specific gravity of the mixture, [dimensionless]. 
        
        For gas-phase conditions, this is calculated at 15.6 °C (60 °F) and 1 
        atm for the mixture and the reference fluid, air. 
        For liquid and solid phase conditions, this is calculated based on a 
        reference fluid of water at 4°C at 1 atm, but the with the liquid or 
        solid mixture's density at the currently specified conditions.

        Examples
        --------
        >>> Mixture('MTBE').SG
        0.7428160596603596
        '''
        return phase_select_property(phase=self.phase, s=self.SGs, l=self.SGl, g=self.SGg)


    ### Single-phase properties

    @property
    def isobaric_expansion(self):
        r'''Isobaric (constant-pressure) expansion of the mixture at its
        current phase, temperature, and pressure in units of [1/K].
        Available only if single phase.

        .. math::
            \beta = \frac{1}{V}\left(\frac{\partial V}{\partial T} \right)_P

        Examples
        --------
        >>> Mixture(['water'], ws=[1], T=647.1, P=22048320.0).isobaric_expansion
        0.34074205839222449
        '''
        return phase_select_property(phase=self.phase, l=self.isobaric_expansion_l, g=self.isobaric_expansion_g)

    @property
    def JT(self):
        r'''Joule Thomson coefficient of the mixture at its
        current phase, temperature, and pressure in units of [K/Pa].
        Available only if single phase.

        .. math::
            \mu_{JT} = \left(\frac{\partial T}{\partial P}\right)_H = \frac{1}{C_p}
            \left[T \left(\frac{\partial V}{\partial T}\right)_P - V\right]
            = \frac{V}{C_p}\left(\beta T-1\right)

        Examples
        --------
        >>> Mixture(['water'], ws=[1]).JT
        -2.2150394958666412e-07
        '''
        return phase_select_property(phase=self.phase, l=self.JTl, g=self.JTg)

    @property
    def mu(self):
        r'''Viscosity of the mixture at its current phase, temperature, and
        pressure in units of [Pa*s].
        Available only if single phase.

        Examples
        --------
        >>> Mixture(['ethanol'], ws=[1], T=400).mu
        1.1853097849748213e-05
        '''
        return phase_select_property(phase=self.phase, l=self.mul, g=self.mug)

    @property
    def k(self):
        r'''Thermal conductivity of the mixture at its current phase,
        temperature, and pressure in units of [W/m/K].
        Available only if single phase.

        Examples
        --------
        >>> Mixture(['ethanol'], ws=[1], T=300).kl
        0.16313594741877802
        '''
        return phase_select_property(phase=self.phase, s=None, l=self.kl, g=self.kg)

    @property
    def nu(self):
        r'''Kinematic viscosity of the the mixture at its current temperature,
        pressure, and phase in units of [m^2/s].
        Available only if single phase.

        .. math::
            \nu = \frac{\mu}{\rho}

        Examples
        --------
        >>> Mixture(['argon'], ws=[1]).nu
        1.3846930410865003e-05
        '''
        return phase_select_property(phase=self.phase, l=self.nul, g=self.nug)

    @property
    def alpha(self):
        r'''Thermal diffusivity of the mixture at its current temperature,
        pressure, and phase in units of [m^2/s].
        Available only if single phase.

        .. math::
            \alpha = \frac{k}{\rho Cp}

        Examples
        --------
        >>> Mixture(['furfural'], ws=[1]).alpha
        7.672866198927953e-08
        '''
        return phase_select_property(phase=self.phase, l=self.alphal, g=self.alphag)

    @property
    def Pr(self):
        r'''Prandtl number of the mixture at its current temperature,
        pressure, and phase; [dimensionless].
        Available only if single phase.

        .. math::
            Pr = \frac{C_p \mu}{k}

        Examples
        --------
        >>> Mixture(['acetone'], ws=[1]).Pr
        4.450368847076066
        '''
        return phase_select_property(phase=self.phase, l=self.Prl, g=self.Prg)

    ### Standard state properties

    @property
    def Vml_STP(self):
        r'''Liquid-phase molar volume of the mixture at 298.15 K and 101.325 kPa,
        and the current composition in units of [mol/m^3].

        Examples
        --------
        >>> Mixture(['cyclobutane'], ws=[1]).Vml_STP
        8.143327329133706e-05
        '''
        return self.VolumeLiquidMixture(T=298.15, P=101325, zs=self.zs, ws=self.ws)

    @property
    def Vmg_STP(self):
        r'''Gas-phase molar volume of the mixture at 298.15 K and 101.325 kPa,
        and the current composition in units of [mol/m^3].

        Examples
        --------
        >>> Mixture(['nitrogen'], ws=[1]).Vmg_STP
        0.023832508854853822
        '''
        return self.VolumeGasMixture(T=298.15, P=101325, zs=self.zs, ws=self.ws)

    @property
    def rhol_STP(self):
        r'''Liquid-phase mass density of the mixture at 298.15 K and 101.325 kPa,
        and the current composition in units of [kg/m^3].

        Examples
        --------
        >>> Mixture(['cyclobutane'], ws=[1]).rhol_STP
        688.9851989526821
        '''
        Vml = self.Vml_STP
        if Vml:
            return Vm_to_rho(Vml, self.MW)
        return None

    @property
    def rhog_STP(self):
        r'''Gas-phase mass density of the mixture at 298.15 K and 101.325 kPa,
        and the current composition in units of [kg/m^3].

        Examples
        --------
        >>> Mixture(['nitrogen'], ws=[1]).rhog_STP
        1.145534453639403
        '''
        Vmg = self.Vmg_STP
        if Vmg:
            return Vm_to_rho(Vmg, self.MW)
        return None

    @property
    def Zl_STP(self):
        r'''Liquid-phase compressibility factor of the mixture at 298.15 K and 101.325 kPa,
        and the current composition, [dimensionless].

        Examples
        --------
        >>> Mixture(['cyclobutane'], ws=[1]).Zl_STP
        0.0033285083663950068
        '''
        Vml = self.Vml
        if Vml:
            return Z(self.T, self.P, Vml)
        return None

    @property
    def Zg_STP(self):
        r'''Gas-phase compressibility factor of the mixture at 298.15 K and 101.325 kPa,
        and the current composition, [dimensionless].

        Examples
        --------
        >>> Mixture(['nitrogen'], ws=[1]).Zg_STP
        0.9995520809691023
        '''
        Vmg = self.Vmg
        if Vmg:
            return Z(self.T, self.P, Vmg)
        return None

    @property
    def rholm_STP(self):
        r'''Molar density of the mixture in the liquid phase at 298.15 K and 101.325 kPa,
        and the current composition, in units of [mol/m^3].

        Examples
        --------
        >>> Mixture(['water'], ws=[1]).rholm_STP
        55344.59086372442
        '''
        Vml = self.Vml_STP
        if Vml:
            return 1./Vml
        return None


    @property
    def rhogm_STP(self):
        r'''Molar density of the mixture in the gas phase at 298.15 K and 101.325 kPa,
        and the current composition, in units of [mol/m^3].

        Examples
        --------
        >>> Mixture(['nitrogen'], ws=[1]).rhogm_STP
        40.892374850585895
        '''
        Vmg = self.Vmg_STP
        if Vmg:
            return 1./Vmg
        return None

    @property
    def API(self):
        r'''API gravity of the hypothetical liquid phase of the mixture, 
        [degrees]. The reference condition is water at 15.6 °C (60 °F) and 1 atm 
        (rho=999.016 kg/m^3, standardized).
            
        Examples
        --------
        >>> Mixture(['hexane', 'decane'], ws=[0.5, 0.5]).API
        71.35326639656284
        '''
        Vml = self.VolumeLiquidMixture(T=288.70555555555552, P=101325, zs=self.zs, ws=self.ws)
        if Vml:
            rho = Vm_to_rho(Vml, self.MW)
        sg = SG(rho, rho_ref=999.016)
        return SG_to_API(sg)

    def draw_2d(self,  Hs=False): # pragma: no cover
        r'''Interface for drawing a 2D image of all the molecules in the
        mixture. Requires an HTML5 browser, and the libraries RDKit and
        IPython. An exception is raised if either of these libraries is
        absent.

        Parameters
        ----------
        Hs : bool
            Whether or not to show hydrogen

        Examples
        --------
        Mixture(['natural gas']).draw_2d()
        '''
        try:
            from rdkit.Chem import Draw
            from rdkit.Chem.Draw import IPythonConsole
            if Hs:
                mols = [i.rdkitmol_Hs for i in self.Chemicals]
            else:
                mols = [i.rdkitmol for i in self.Chemicals]
            return Draw.MolsToImage(mols)
        except:
            return 'Rdkit is required for this feature.'


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


