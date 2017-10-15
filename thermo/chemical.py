# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['Chemical', 'reference_states']


from thermo.identifiers import *
from thermo.identifiers import _MixtureDict
from thermo.vapor_pressure import VaporPressure
from thermo.phase_change import Tb, Tm, Hfus, Hsub, Tliquidus, EnthalpyVaporization
from thermo.activity import identify_phase, identify_phase_mixture, Pbubble_mixture, Pdew_mixture

from thermo.critical import Tc, Pc, Vc, Tc_mixture, Pc_mixture, Vc_mixture
from thermo.acentric import omega, omega_mixture, StielPolar
from thermo.triple import Tt, Pt
from thermo.thermal_conductivity import ThermalConductivityLiquid, ThermalConductivityGas, ThermalConductivityLiquidMixture, ThermalConductivityGasMixture
from thermo.volume import VolumeGas, VolumeLiquid, VolumeSolid, VolumeLiquidMixture, VolumeGasMixture, VolumeSolidMixture
from thermo.permittivity import *
from thermo.heat_capacity import HeatCapacitySolid, HeatCapacityGas, HeatCapacityLiquid, HeatCapacitySolidMixture, HeatCapacityGasMixture, HeatCapacityLiquidMixture
from thermo.interface import SurfaceTension, SurfaceTensionMixture
from thermo.viscosity import ViscosityLiquid, ViscosityGas, ViscosityLiquidMixture, ViscosityGasMixture, viscosity_index
from thermo.reaction import Hf
from thermo.combustion import Hcombustion
from thermo.safety import Tflash, Tautoignition, LFL, UFL, TWA, STEL, Ceiling, Skin, Carcinogen, LFL_mixture, UFL_mixture
from thermo.solubility import solubility_parameter
from thermo.dipole import dipole_moment as dipole
from thermo.utils import *
from fluids.core import Reynolds, Capillary, Weber, Bond, Grashof, Peclet_heat
from thermo.lennard_jones import Stockmayer, molecular_diameter
from thermo.environment import GWP, ODP, logP
from thermo.law import legal_status, economic_status
from thermo.refractivity import refractive_index
from thermo.electrochem import conductivity
from thermo.elements import atom_fractions, mass_fractions, similarity_variable, atoms_to_Hill, simple_formula_parser, molecular_weight, charge_from_formula
from thermo.coolprop import has_CoolProp
from thermo.eos import *
from thermo.eos_mix import *
from thermo.unifac import DDBST_UNIFAC_assignments, DDBST_MODIFIED_UNIFAC_assignments, DDBST_PSRK_assignments, load_group_assignments_DDBST, UNIFAC_RQ, Van_der_Waals_volume, Van_der_Waals_area


from fluids.core import *
from scipy.optimize import newton
from scipy.constants import physical_constants
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

#import warnings
#warnings.filterwarnings("ignore")

caching = True

# Format: (T, P, phase, H, S, molar=True)
IAPWS = (273.16, 611.655, 'l', 0.00922, 0, True) # Water; had to convert Href from mass to molar
ASHRAE = (233.15, 'Psat', 'l', 0, 0, True) # As described in REFPROP
IIR = (273.15, 'Psat', 'l', 200E3, 1000, False) # 200 kj/kg reference, as described in REFPROP
REFPROP = ('Tb', 101325, 'l', 0, 0, True)
CHEMSEP = (298., 101325, 'g', 0, 0, True) # It has an option to add Hf to the reference
PRO_II = (298.15, 101325, 'gas', 0, 0, True)
HYSYS = (298.15, 101325, 'calc', 'Hf', 0, True)
UNISIM = HYSYS #
SUPERPRO = (298.15, 101325, 'calc', 0, 0, True) # No support for entropy found, 0 assumed
# note soecifying a phase works for chemicals but not mixtures.

reference_states = [IAPWS, ASHRAE, IIR, REFPROP, CHEMSEP, PRO_II, HYSYS,
                    UNISIM, SUPERPRO]
_chemical_cache = {}


class Chemical(object): # pragma: no cover
    '''Creates a Chemical object which contains basic information such as 
    molecular weight and the structure of the species, as well as thermodynamic
    and transport properties as a function of temperature and pressure.
    
    Parameters
    ----------
    ID : str
        One of the following [-]:
            * Name, in IUPAC form or common form or a synonym registered in PubChem
            * InChI name, prefixed by 'InChI=1S/' or 'InChI=1/'
            * InChI key, prefixed by 'InChIKey='
            * PubChem CID, prefixed by 'PubChem='
            * SMILES (prefix with 'SMILES=' to ensure smiles parsing)
            * CAS number
    T : float, optional
        Temperature of the chemical (default 298.15 K), [K]
    P : float, optional
        Pressure of the chemical (default 101325 Pa) [Pa]
        
    Examples
    --------
    Creating chemical objects:
    
    >>> Chemical('hexane')
    <Chemical [hexane], T=298.15 K, P=101325 Pa>

    >>> Chemical('CCCCCCCC', T=500, P=1E7)
    <Chemical [octane], T=500.00 K, P=10000000 Pa>
    
    >>> Chemical('7440-36-0', P=1000)
    <Chemical [antimony], T=298.15 K, P=1000 Pa>
    
    Getting basic properties:
        
    >>> N2 = Chemical('Nitrogen')
    >>> N2.Tm, N2.Tb, N2.Tc # melting, boiling, and critical points [K]
    (63.15, 77.355, 126.2)
    >>> N2.Pt, N2.Pc # sublimation and critical pressure [Pa]
    (12526.9697368421, 3394387.5)
    >>> N2.CAS, N2.formula, N2.InChI, N2.smiles, N2.atoms # CAS number, formula, InChI string, smiles string, dictionary of atomic elements and their count
    ('7727-37-9', 'N2', 'N2/c1-2', 'N#N', {'N': 2})
    
    Changing the T/P of the chemical, and gettign temperature-dependent
    properties:
        
    >>> N2.Cp, N2.rho, N2.mu # Heat capacity [J/kg/K], density [kg/m^3], viscosity [Pa*s]
    (1039.4978324480921, 1.1452416223829405, 1.7804740647270688e-05)
    >>> N2.calculate(T=65, P=1E6) # set it to a liquid at 65 K and 1 MPa
    >>> N2.phase
    'l'
    >>> N2.Cp, N2.rho, N2.mu # properties are now of the liquid phase
    (1984.9709554303004, 861.3539919443364, 0.0002857739143670701)
    
    Molar units are also available for properties:
        
    >>> N2.Cpm, N2.Vm, N2.Hvapm # heat capacity [J/mol/K], molar volume [m^3/mol], enthalpy of vaporization [J/mol]
    (55.60578536285118, 3.252251717875631e-05, 5982.710998291719)
    
    A great deal of properties are available; for a complete list look at the
    attributes list. 
    
    >>> N2.alpha, N2.JT # thermal diffusivity [m^2/s], Joule-Thompson coefficient [K/Pa]
    (9.963988241081473e-08, -4.0370955160282056e-07)
    
    >>> N2.isentropic_exponent, N2.isobaric_expansion
    (1.4000000000000001, 0.0047654228408661571)
    
    For pure species, the phase is easily identified, allowing for properties 
    to be obtained without needing to specify the phase. However, the 
    properties are also available in the hypothetical gas phase (when under the
    boiling point) and in the hypothetical liquid phase (when above the boiling
    point) as these properties are needed to evaluate mixture properties. 
    Specify the phase of a property to be retrieved by appending 'l' or 'g' or 
    's' to the property.
    
    >>> tol = Chemical('toluene')

    >>> tol.rhog, tol.Cpg, tol.kg, tol.mug
    (4.032009635018902, 1126.5533755283168, 0.010736843919054837, 6.973325939594919e-06)

    Temperature dependent properties are calculated by objects which provide 
    many useful features related to the properties. To determine the 
    temperature at which nitrogen has a saturation pressure of 1 MPa:
    
    >>> N2.VaporPressure.solve_prop(1E6)
    103.73528598652341
    
    To compute an integral of the ideal-gas heat capacity of nitrogen
    to determine the enthalpy required for a given change in temperature.
    Note the thermodynamic objects calculate values in molar units always.
        
    >>> N2.HeatCapacityGas.T_dependent_property_integral(100, 120) # J/mol/K
    582.0121860897898
    
    Derivatives of properties can be calculated as well, as may be needed by
    for example heat transfer calculations:
        
    >>> N2.SurfaceTension.T_dependent_property_derivative(77)
    -0.00022695346296730534
    
    If a property is needed at multiple temperatures or pressures, it is faster
    to use the object directly to perform the calculation rather than setting
    the conditions for the chemical.
    
    >>> [N2.VaporPressure(T) for T in range(80, 120, 10)]
    [135987.2098445901, 357853.1185240415, 773372.8975626591, 1458809.6559246336]
    
    These objects are also how the methods by which the properties are 
    calculated can be changed. To see the available methods for a property:
        
    >>> N2.VaporPressure.all_methods
    set(['VDI_PPDS', 'BOILING_CRITICAL', 'WAGNER_MCGARRY', 'AMBROSE_WALTON', 'COOLPROP', 'LEE_KESLER_PSAT', 'EOS', 'ANTOINE_POLING', 'SANJARI', 'DIPPR_PERRY_8E', 'Edalat', 'WAGNER_POLING'])
            
    To specify the method which should be used for calculations of a property.
    In the example below, the Lee-kesler correlation for vapor pressure is
    specified.
        
    >>> N2.calculate(80)
    >>> N2.Psat
    136979.4840843189
    >>> N2.VaporPressure.set_user_methods('LEE_KESLER_PSAT')
    >>> N2.Psat
    134987.76815364443
    
    For memory reduction, these objects are shared by all chemicals which are
    the same; new instances will use the same specified methods. 
    
    >>> N2_2 = Chemical('nitrogen')
    >>> N2_2.VaporPressure.user_methods
    ['LEE_KESLER_PSAT']
    
    To disable this behavior, set thermo.chemical.caching to False.
    
    >>> thermo.chemical.caching = False
    >>> N2_3 = Chemical('nitrogen')
    >>> N2_3.VaporPressure.user_methods
    []
    
    Properties may also be plotted via these objects:
        
    >>> N2.VaporPressure.plot_T_dependent_property()
    >>> N2.VolumeLiquid.plot_isotherm(T=77, Pmin=1E5, Pmax=1E7)
    >>> N2.VolumeLiquid.plot_isobar(P=1E6,  Tmin=66, Tmax=120)
    >>> N2.VolumeLiquid.plot_TP_dependent_property(Tmin=60, Tmax=100,  Pmin=1E5, Pmax=1E7)
    
    Attributes
    ----------
    T : float
        Temperature of the chemical, [K]
    P : float
        Pressure of the chemical, [Pa]
    phase : str
        Phase of the chemical; one of 's', 'l', 'g', or 'l/g'.
    ID : str
        User specified string by which the chemical's CAS was looked up.
    CAS : str
        The CAS number of the chemical.
    PubChem : int
        PubChem Compound identifier (CID) of the chemical; all chemicals are
        sourced from their database. Chemicals can be looked at online at
        `<https://pubchem.ncbi.nlm.nih.gov>`_.
    MW : float
        Molecular weight of the compound, [g/mol]
    formula : str
        Molecular formula of the compound.
    atoms : dict
        dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]
    similarity_variable : float
        Similarity variable, see :obj:`thermo.elements.similarity_variable`
        for the definition, [mol/g]
    smiles : str
        Simplified molecular-input line-entry system representation of the
        compound.
    InChI : str
        IUPAC International Chemical Identifier of the compound.
    InChI_Key : str
        25-character hash of the compound's InChI.
    IUPAC_name : str
        Preferred IUPAC name for a compound.
    synonyms : list of strings
        All synonyms for the compound found in PubChem, sorted by popularity.
    Tm : float
        Melting temperature [K]
    Tb : float
        Boiling temperature [K]
    Tc : float
        Critical temperature [K]
    Pc : float
        Critical pressure [Pa]
    Vc : float
        Critical volume [m^3/mol]
    Zc : float
        Critical compressibility [-]
    rhoc : float
        Critical density [kg/m^3]
    rhocm : float
        Critical molar density [mol/m^3]
    omega : float
        Acentric factor [-]
    StielPolar : float
        Stiel Polar factor, see :obj:`thermo.acentric.StielPolar` for
        the definition [-]
    Tt : float
        Triple temperature, [K]
    Pt : float
        Triple pressure, [Pa]
    Hfus : float
        Enthalpy of fusion [J/kg]
    Hfusm : float
        Molar enthalpy of fusion [J/mol]
    Hsub : float
        Enthalpy of sublimation [J/kg]
    Hsubm : float
        Molar enthalpy of sublimation [J/mol]
    Hf : float
        Enthalpy of formation [J/mol]
    Hc : float
        Molar enthalpy of combustion [J/mol]
    Tflash : float
        Flash point of the chemical, [K]
    Tautoignition : float
        Autoignition point of the chemical, [K]
    LFL : float
        Lower flammability limit of the gas in an atmosphere at STP, mole 
        fraction [-]
    UFL : float
        Upper flammability limit of the gas in an atmosphere at STP, mole 
        fraction [-]
    TWA : tuple[quantity, unit]
        Time-Weighted Average limit on worker exposure to dangerous chemicals.
    STEL : tuple[quantity, unit]
        Short-term Exposure limit on worker exposure to dangerous chemicals.
    Ceiling : tuple[quantity, unit]
        Ceiling limits on worker exposure to dangerous chemicals.
    Skin : bool
        Whether or not a chemical can be absorbed through the skin.
    Carcinogen : str or dict
        Carcinogen status information.
    dipole : float
        Dipole moment in debye, [3.33564095198e-30 ampere*second^2]
    Stockmayer : float
        Lennard-Jones depth of potential-energy minimum over k, [K]
    molecular_diameter : float
        Lennard-Jones molecular diameter, [angstrom]
    GWP : float
        Global warming potential (default 100-year outlook) (impact/mass 
        chemical)/(impact/mass CO2), [-]
    ODP : float
        Ozone Depletion potential (impact/mass chemical)/(impact/mass CFC-11),
        [-]
    logP : float
        Octanol-water partition coefficient, [-]
    legal_status : str or dict
        Legal status information [-]
    economic_status : list
        Economic status information [-]
    RI : float
        Refractive Index on the Na D line, [-]
    RIT : float
        Temperature at which refractive index reading was made
    conductivity : float
        Electrical conductivity of the fluid, [S/m]
    conductivityT : float
        Temperature at which conductivity measurement was made
    VaporPressure : object
        Instance of :obj:`thermo.vapor_pressure.VaporPressure`, with data and
        methods loaded for the chemical; performs the actual calculations of
        vapor pressure of the chemical.
    EnthalpyVaporization : object
        Instance of :obj:`thermo.phase_change.EnthalpyVaporization`, with data
        and methods loaded for the chemical; performs the actual calculations
        of molar enthalpy of vaporization of the chemical.
    VolumeSolid : object
        Instance of :obj:`thermo.volume.VolumeSolid`, with data and methods
        loaded for the chemical; performs the actual calculations of molar
        volume of the solid phase of the chemical.
    VolumeLiquid : object
        Instance of :obj:`thermo.volume.VolumeLiquid`, with data and methods
        loaded for the chemical; performs the actual calculations of molar
        volume of the liquid phase of the chemical.
    VolumeGas : object
        Instance of :obj:`thermo.volume.VolumeGas`, with data and methods
        loaded for the chemical; performs the actual calculations of molar
        volume of the gas phase of the chemical.
    HeatCapacitySolid : object
        Instance of :obj:`thermo.heat_capacity.HeatCapacitySolid`, with data and
        methods loaded for the chemical; performs the actual calculations of
        molar heat capacity of the solid phase of the chemical.
    HeatCapacityLiquid : object
        Instance of :obj:`thermo.heat_capacity.HeatCapacityLiquid`, with data and
        methods loaded for the chemical; performs the actual calculations of
        molar heat capacity of the liquid phase of the chemical.
    HeatCapacityGas : object
        Instance of :obj:`thermo.heat_capacity.HeatCapacityGas`, with data and
        methods loaded for the chemical; performs the actual calculations of
        molar heat capacity of the gas phase of the chemical.
    ViscosityLiquid : object
        Instance of :obj:`thermo.viscosity.ViscosityLiquid`, with data and
        methods loaded for the chemical; performs the actual calculations of
        viscosity of the liquid phase of the chemical.
    ViscosityGas : object
        Instance of :obj:`thermo.viscosity.ViscosityGas`, with data and
        methods loaded for the chemical; performs the actual calculations of
        viscosity of the gas phase of the chemical.
    ThermalConductivityLiquid : object
        Instance of :obj:`thermo.thermal_conductivity.ThermalConductivityLiquid`,
        with data and methods loaded for the chemical; performs the actual
        calculations of thermal conductivity of the liquid phase of the
        chemical.
    ThermalConductivityGas : object
        Instance of :obj:`thermo.thermal_conductivity.ThermalConductivityGas`,
        with data and methods loaded for the chemical; performs the actual
        calculations of thermal conductivity of the gas phase of the chemical.
    SurfaceTension : object
        Instance of :obj:`thermo.interface.SurfaceTension`, with data and
        methods loaded for the chemical; performs the actual calculations of
        surface tension of the chemical.
    Permittivity : object
        Instance of :obj:`thermo.permittivity.Permittivity`, with data and
        methods loaded for the chemical; performs the actual calculations of
        permittivity of the chemical.
    Psat_298 : float
        Vapor pressure of the chemical at 298.15 K, [Pa]
    phase_STP : str
        Phase of the chemical at 298.15 K and 101325 Pa; one of 's', 'l', 'g',
        or 'l/g'.
    Vml_Tb : float
        Molar volume of liquid phase at the normal boiling point [m^3/mol]
    Vml_Tm : float
        Molar volume of liquid phase at the melting point [m^3/mol]
    Vml_STP : float
        Molar volume of liquid phase at 298.15 K and 101325 Pa [m^3/mol]
    Vmg_STP : float
        Molar volume of gas phase at 298.15 K and 101325 Pa [m^3/mol]
    Hvap_Tbm : float
        Molar enthalpy of vaporization at the normal boiling point [J/mol]
    Hvap_Tb : float
        Mass enthalpy of vaporization at the normal boiling point [J/kg]
    alpha
    alphag
    alphal
    API
    aromatic_rings
    atom_fractions
    Bvirial
    charge
    Cp
    Cpg
    Cpgm
    Cpl
    Cplm
    Cpm
    Cps
    Cpsm
    Cvg
    Cvgm
    eos
    Hill
    Hvap
    Hvapm
    isentropic_exponent
    isobaric_expansion
    isobaric_expansion_g
    isobaric_expansion_l
    JT
    JTg
    JTl
    k
    kg
    kl
    mass_fractions
    mu
    mug
    mul
    nu
    nug
    nul
    Parachor
    permittivity
    Poynting
    Pr
    Prg
    Prl
    Psat
    PSRK_groups
    rdkitmol
    rdkitmol_Hs
    rho
    rhog
    rhogm
    rhol
    rholm
    rhom
    rhos
    rhosm
    rings
    SG
    SGg
    SGl
    SGs
    sigma
    solubility_parameter
    UNIFAC_Dortmund_groups
    UNIFAC_groups
    UNIFAC_R
    UNIFAC_Q
    Van_der_Waals_area
    Van_der_Waals_volume
    Vm
    Vmg
    Vml
    Vms
    Z
    Zg
    Zl
    Zs
    '''

    __atom_fractions = None
    __mass_fractions = None
    __UNIFAC_groups = None
    __UNIFAC_Dortmund_groups = None
    __PSRK_groups = None
    __rdkitmol = None
    __rdkitmol_Hs = None
    __Hill = None
    __legal_status = None
    __economic_status = None
    def __repr__(self):
        return '<Chemical [%s], T=%.2f K, P=%.0f Pa>' %(self.name, self.T, self.P)

    def __init__(self, ID, T=298.15, P=101325):
        if isinstance(ID, dict):
            self.CAS = ID['CASRN']
            self.ID = self.name = ID['name']
            self.formula = ID['formula']
            self.MW = ID['MW'] if 'MW' in ID else molecular_weight(simple_formula_parser(self.formula))
            self.PubChem = ID['PubChem'] if 'PubChem' in ID else None
            self.smiles = ID['smiles'] if 'smiles' in ID else None
            self.InChI = ID['InChI'] if 'InChI' in ID else None
            self.InChI_Key = ID['InChI_Key'] if 'InChI_Key' in ID else None
            self.synonyms = ID['synonyms'] if 'synonyms' in ID else None
        else:
            self.ID = ID
            # Identification
            self.CAS = CAS_from_any(ID)
            self.ChemicalMetadata = pubchem_db.search_CAS(self.CAS)


        if self.CAS in _chemical_cache and caching:
            self.__dict__.update(_chemical_cache[self.CAS].__dict__)
            self.calculate(T, P)
        else:
            if not isinstance(ID, dict):
                self.PubChem = self.ChemicalMetadata.pubchemid
                self.MW = self.ChemicalMetadata.MW
                self.formula = self.ChemicalMetadata.formula
                self.smiles = self.ChemicalMetadata.smiles
                self.InChI = self.ChemicalMetadata.InChI
                self.InChI_Key = self.ChemicalMetadata.InChI_key
                self.IUPAC_name = self.ChemicalMetadata.iupac_name.lower()
                self.name = self.ChemicalMetadata.common_name.lower()
                self.synonyms = self.ChemicalMetadata.all_names

            self.atoms = simple_formula_parser(self.formula)
            self.similarity_variable = similarity_variable(self.atoms, self.MW)

            self.eos_in_a_box = []
            self.set_constant_sources()
            self.set_constants()
            self.set_eos(T=T, P=P)
            self.set_TP_sources()
            self.set_ref()
            self.calculate(T, P)
            if len(_chemical_cache) < 1000:
                _chemical_cache[self.CAS] = self



    def calculate(self, T=None, P=None):
        if (hasattr(self, 'T') and T == self.T and hasattr(self, 'P') and P == self.P):
            return None
        if T:
            if T < 0:
                raise Exception('Negative value specified for Chemical temperature - aborting!')
            self.T = T
        if P:
            if P < 0:
                raise Exception('Negative value specified for Chemical pressure - aborting!')
            self.P = P


        self.phase = identify_phase(T=self.T, P=self.P, Tm=self.Tm, Tb=self.Tb, Tc=self.Tc, Psat=self.Psat)
        self.eos = self.eos.to_TP(T=self.T, P=self.P)
        self.eos_in_a_box[0] = self.eos
        self.set_thermo()


    def draw_2d(self, width=300, height=300, Hs=False): # pragma: no cover
        r'''Interface for drawing a 2D image of the molecule.
        Requires an HTML5 browser, and the libraries RDKit and
        IPython. An exception is raised if either of these libraries is
        absent.

        Parameters
        ----------
        width : int
            Number of pixels wide for the view
        height : int
            Number of pixels tall for the view
        Hs : bool
            Whether or not to show hydrogen

        Examples
        --------
        >>> Chemical('decane').draw_2d()
        '''
        try:
            from rdkit.Chem import Draw
            from rdkit.Chem.Draw import IPythonConsole
            if Hs:
                mol = self.rdkitmol_Hs
            else:
                mol = self.rdkitmol
            return Draw.MolToImage(mol, size=(width, height))
        except:
            return 'Rdkit is required for this feature.'

    def draw_3d(self, width=300, height=500, style='stick', Hs=True): # pragma: no cover
        r'''Interface for drawing an interactive 3D view of the molecule.
        Requires an HTML5 browser, and the libraries RDKit, pymol3D, and
        IPython. An exception is raised if all three of these libraries are
        absent.

        Parameters
        ----------
        width : int
            Number of pixels wide for the view
        height : int
            Number of pixels tall for the view
        style : str
            One of 'stick', 'line', 'cross', or 'sphere'
        Hs : bool
            Whether or not to show hydrogen

        Examples
        --------
        >>> Chemical('cubane').draw_3d()
        '''
        try:
            import py3Dmol
            from IPython.display import display
            if Hs:
                mol = self.rdkitmol_Hs
            else:
                mol = self.rdkitmol
            AllChem.EmbedMultipleConfs(mol)
            mb = Chem.MolToMolBlock(mol)
            p = py3Dmol.view(width=width,height=height)
            p.addModel(mb,'sdf')
            p.setStyle({style:{}})
            p.zoomTo()
            display(p.show())
        except:
            return 'py3Dmol, RDKit, and IPython are required for this feature.'

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

        # Triple point
        self.Tt_sources = Tt(self.CAS, AvailableMethods=True)
        self.Tt_source = self.Tt_sources[0]
        self.Pt_sources = Pt(self.CAS, AvailableMethods=True)
        self.Pt_source = self.Pt_sources[0]

        # Enthalpy
        self.Hfus_methods = Hfus(MW=self.MW, AvailableMethods=True, CASRN=self.CAS)
        self.Hfus_method = self.Hfus_methods[0]

        self.Hsub_methods = Hsub(MW=self.MW, AvailableMethods=True, CASRN=self.CAS)
        self.Hsub_method = self.Hsub_methods[0]

        # Fire Safety Limits
        self.Tflash_sources = Tflash(self.CAS, AvailableMethods=True)
        self.Tflash_source = self.Tflash_sources[0]
        self.Tautoignition_sources = Tautoignition(self.CAS, AvailableMethods=True)
        self.Tautoignition_source = self.Tautoignition_sources[0]

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

        # Environmental
        self.GWP_sources = GWP(CASRN=self.CAS, AvailableMethods=True)
        self.GWP_source = self.GWP_sources[0]
        self.ODP_sources = ODP(CASRN=self.CAS, AvailableMethods=True)
        self.ODP_source = self.ODP_sources[0]
        self.logP_sources = logP(CASRN=self.CAS, AvailableMethods=True)
        self.logP_source = self.logP_sources[0]

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

        self.StielPolar_methods = StielPolar(Tc=self.Tc, Pc=self.Pc, omega=self.omega, CASRN=self.CAS, AvailableMethods=True)
        self.StielPolar_method = self.StielPolar_methods[0]
        self.StielPolar = StielPolar(Tc=self.Tc, Pc=self.Pc, omega=self.omega, CASRN=self.CAS, Method=self.StielPolar_method)

        self.Zc = Z(self.Tc, self.Pc, self.Vc) if all((self.Tc, self.Pc, self.Vc)) else None
        self.rhoc = Vm_to_rho(self.Vc, self.MW) if self.Vc else None
        self.rhocm = 1./self.Vc if self.Vc else None

        # Triple point
        self.Pt = Pt(self.CAS, Method=self.Pt_source)
        self.Tt = Tt(self.CAS, Method=self.Tt_source)

        # Enthalpy
        self.Hfus = Hfus(MW=self.MW, Method=self.Hfus_method, CASRN=self.CAS)
        self.Hfusm = property_mass_to_molar(self.Hfus, self.MW) if self.Hfus else None

        self.Hsub = Hsub(MW=self.MW, Method=self.Hsub_method, CASRN=self.CAS)
        self.Hsubm = property_mass_to_molar(self.Hsub, self.MW)

        # Chemistry
        self.Hf = Hf(CASRN=self.CAS, Method=self.Hf_source)
        self.Hc = Hcombustion(atoms=self.atoms, Hf=self.Hf)

        # Fire Safety Limits
        self.Tflash = Tflash(self.CAS, Method=self.Tflash_source)
        self.Tautoignition = Tautoignition(self.CAS, Method=self.Tautoignition_source)
        self.LFL_sources = LFL(atoms=self.atoms, Hc=self.Hc, CASRN=self.CAS, AvailableMethods=True)
        self.LFL_source = self.LFL_sources[0]
        self.UFL_sources = UFL(atoms=self.atoms, Hc=self.Hc, CASRN=self.CAS, AvailableMethods=True)
        self.UFL_source = self.UFL_sources[0]
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
        self.Stockmayer_sources = Stockmayer(Tc=self.Tc, Zc=self.Zc, omega=self.omega, AvailableMethods=True, CASRN=self.CAS)
        self.Stockmayer_source = self.Stockmayer_sources[0]
        self.Stockmayer = Stockmayer(Tm=self.Tm, Tb=self.Tb, Tc=self.Tc, Zc=self.Zc, omega=self.omega, Method=self.Stockmayer_source, CASRN=self.CAS)

        # Environmental
        self.GWP = GWP(CASRN=self.CAS, Method=self.GWP_source)
        self.ODP = ODP(CASRN=self.CAS, Method=self.ODP_source)
        self.logP = logP(CASRN=self.CAS, Method=self.logP_source)

        # Analytical
        self.RI, self.RIT = refractive_index(CASRN=self.CAS, Method=self.RI_source)
        self.conductivity, self.conductivityT = conductivity(CASRN=self.CAS, Method=self.conductivity_source)

    def set_eos(self, T, P, eos=PR):
        try:
            self.eos = eos(T=T, P=P, Tc=self.Tc, Pc=self.Pc, omega=self.omega)
        except:
            # Handle overflow errors and so on
            self.eos = GCEOS_DUMMY(T=T, P=P)

    @property
    def eos(self):
        r'''Equation of state object held by the chemical; used to calculate
        excess thermodynamic quantities, and also provides a vapor pressure
        curve, enthalpy of vaporization curve, fugacity, thermodynamic partial
        derivatives, and more; see :obj:`thermo.eos` for a full listing.

        Examples
        --------
        >>> Chemical('methane').eos.V_g
        0.024410195021818258
        '''
        return self.eos_in_a_box[0]

    @eos.setter
    def eos(self, eos):
        if self.eos_in_a_box:
            self.eos_in_a_box.pop()
        # Pass this mutable list to objects so if it is changed, it gets
        # changed in the property method too
        self.eos_in_a_box.append(eos)


    def set_TP_sources(self):
        # Tempearture and Pressure Denepdence
        # Get and choose initial methods
        self.VaporPressure = VaporPressure(Tb=self.Tb, Tc=self.Tc, Pc=self.Pc,
                                           omega=self.omega, CASRN=self.CAS,
                                           eos=self.eos_in_a_box)
        self.Psat_298 = self.VaporPressure.T_dependent_property(298.15)
        self.phase_STP = identify_phase(T=298.15, P=101325., Tm=self.Tm, Tb=self.Tb, Tc=self.Tc, Psat=self.Psat_298)

        self.VolumeLiquid = VolumeLiquid(MW=self.MW, Tb=self.Tb, Tc=self.Tc,
                          Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega,
                          dipole=self.dipole,
                          Psat=self.VaporPressure.T_dependent_property,
                          eos=self.eos_in_a_box, CASRN=self.CAS)

        self.Vml_Tb = self.VolumeLiquid.T_dependent_property(self.Tb) if self.Tb else None
        self.Vml_Tm = self.VolumeLiquid.T_dependent_property(self.Tm) if self.Tm else None
        self.Vml_STP = self.VolumeLiquid.T_dependent_property(298.15)

        self.VolumeGas = VolumeGas(MW=self.MW, Tc=self.Tc, Pc=self.Pc,
                                   omega=self.omega, dipole=self.dipole,
                                   eos=self.eos_in_a_box, CASRN=self.CAS)

        self.Vmg_STP = self.VolumeGas.TP_dependent_property(298.15, 101325)

        self.VolumeSolid = VolumeSolid(CASRN=self.CAS, MW=self.MW, Tt=self.Tt)

        self.HeatCapacityGas = HeatCapacityGas(CASRN=self.CAS, MW=self.MW, similarity_variable=self.similarity_variable)

        self.HeatCapacitySolid = HeatCapacitySolid(MW=self.MW, similarity_variable=self.similarity_variable, CASRN=self.CAS)

        self.HeatCapacityLiquid = HeatCapacityLiquid(CASRN=self.CAS, MW=self.MW, similarity_variable=self.similarity_variable, Tc=self.Tc, omega=self.omega, Cpgm=self.HeatCapacityGas.T_dependent_property)

        self.EnthalpyVaporization = EnthalpyVaporization(CASRN=self.CAS, Tb=self.Tb, Tc=self.Tc, Pc=self.Pc, omega=self.omega, similarity_variable=self.similarity_variable)
        self.Hvap_Tbm = self.EnthalpyVaporization.T_dependent_property(self.Tb) if self.Tb else None
        self.Hvap_Tb = property_molar_to_mass(self.Hvap_Tbm, self.MW)

        self.ViscosityLiquid = ViscosityLiquid(CASRN=self.CAS, MW=self.MW, Tm=self.Tm, Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, omega=self.omega, Psat=self.VaporPressure.T_dependent_property, Vml=self.VolumeLiquid.T_dependent_property)

        Vmg_atm_T_dependent = lambda T : self.VolumeGas.TP_dependent_property(T, 101325)
        self.ViscosityGas = ViscosityGas(CASRN=self.CAS, MW=self.MW, Tc=self.Tc, Pc=self.Pc, Zc=self.Zc, dipole=self.dipole, Vmg=Vmg_atm_T_dependent)

        self.ThermalConductivityLiquid = ThermalConductivityLiquid(CASRN=self.CAS, MW=self.MW, Tm=self.Tm, Tb=self.Tb, Tc=self.Tc, Pc=self.Pc, omega=self.omega, Hfus=self.Hfusm)

        Cvgm_calc = lambda T : self.HeatCapacityGas.T_dependent_property(T) - R
        self.ThermalConductivityGas = ThermalConductivityGas(CASRN=self.CAS, MW=self.MW, Tb=self.Tb, Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega, dipole=self.dipole, Vmg=self.VolumeGas, Cvgm=Cvgm_calc, mug=self.ViscosityGas.T_dependent_property)

        Cpl_calc = lambda T : property_molar_to_mass(self.HeatCapacityLiquid.T_dependent_property(T), self.MW)
        self.SurfaceTension = SurfaceTension(CASRN=self.CAS, MW=self.MW, Tb=self.Tb, Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega, StielPolar=self.StielPolar, Hvap_Tb=self.Hvap_Tb, Vml=self.VolumeLiquid.T_dependent_property, Cpl=Cpl_calc)

        self.Permittivity = Permittivity(CASRN=self.CAS)

        self.solubility_parameter_methods = solubility_parameter(Hvapm=self.Hvap_Tbm, Vml=self.Vml_STP, AvailableMethods=True, CASRN=self.CAS)
        self.solubility_parameter_method = self.solubility_parameter_methods[0]

        # set molecular_diameter; depends on Vml_Tb, Vml_Tm
        self.molecular_diameter_sources = molecular_diameter(Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega, Vm=self.Vml_Tm, Vb=self.Vml_Tb, AvailableMethods=True, CASRN=self.CAS)
        self.molecular_diameter_source = self.molecular_diameter_sources[0]
        self.molecular_diameter = molecular_diameter(Tc=self.Tc, Pc=self.Pc, Vc=self.Vc, Zc=self.Zc, omega=self.omega, Vm=self.Vml_Tm, Vb=self.Vml_Tb, Method=self.molecular_diameter_source, CASRN=self.CAS)


    def set_ref(self, T_ref=298.15, P_ref=101325, phase_ref='calc', H_ref=0, S_ref=0):
        # Muse run after set_TP_sources, set_phase due to HeatCapacity*, phase_STP
        self.T_ref = getattr(self, T_ref) if isinstance(T_ref, str) else T_ref
        self.P_ref = getattr(self, P_ref) if isinstance(P_ref, str) else P_ref
        self.H_ref = getattr(self, H_ref) if isinstance(H_ref, str) else H_ref
        self.S_ref = getattr(self, S_ref) if isinstance(S_ref, str) else S_ref
        self.phase_ref = self.phase_STP if phase_ref == 'calc' else phase_ref

        integrators = {'s': self.HeatCapacitySolid.T_dependent_property_integral,
           'l': self.HeatCapacityLiquid.T_dependent_property_integral,
           'g': self.HeatCapacityGas.T_dependent_property_integral}

        integrators_T = {'s': self.HeatCapacitySolid.T_dependent_property_integral_over_T,
           'l': self.HeatCapacityLiquid.T_dependent_property_integral_over_T,
           'g': self.HeatCapacityGas.T_dependent_property_integral_over_T}

        # Integrals stored to avoid recalculation, all from T_low to T_high
        try:
            # Enthalpy integrals
            if self.phase_ref != 'l' and self.Tm and self.Tb:
                self.H_int_l_Tm_to_Tb = integrators['l'](self.Tm, self.Tb)
            if self.phase_ref == 's' and self.Tm:
                self.H_int_T_ref_s_to_Tm = integrators['s'](self.T_ref, self.Tm)
            if self.phase_ref == 'g' and self.Tb:
                self.H_int_Tb_to_T_ref_g = integrators['g'](self.Tb, self.T_ref)
            if self.phase_ref == 'l' and self.Tm and self.Tb:
                self.H_int_l_T_ref_l_to_Tb = integrators['l'](self.T_ref, self.Tb)
                self.H_int_l_Tm_to_T_ref_l = integrators['l'](self.Tm, self.T_ref)

            # Entropy integrals
            if self.phase_ref != 'l' and self.Tm and self.Tb:
                self.S_int_l_Tm_to_Tb = integrators_T['l'](self.Tm, self.Tb)
            if self.phase_ref == 's' and self.Tm:
                self.S_int_T_ref_s_to_Tm = integrators_T['s'](self.T_ref, self.Tm)
            if self.phase_ref == 'g' and self.Tb:
                self.S_int_Tb_to_T_ref_g = integrators_T['g'](self.Tb, self.T_ref)
            if self.phase_ref == 'l' and self.Tm and self.Tb:
                self.S_int_l_T_ref_l_to_Tb = integrators_T['l'](self.T_ref, self.Tb)
                self.S_int_l_Tm_to_T_ref_l = integrators_T['l'](self.Tm, self.T_ref)
        except:
            pass

        # Excess properties stored
        try:
            if self.phase_ref == 'g':
                self.eos_phase_ref = self.eos.to_TP(self.T_ref, self.P_ref)
                self.H_dep_ref_g = self.eos_phase_ref.H_dep_g
                self.S_dep_ref_g = self.eos_phase_ref.S_dep_g


            elif self.phase_ref == 'l':
                self.eos_phase_ref = self.eos.to_TP(self.T_ref, self.P_ref)
                self.H_dep_ref_l = self.eos_phase_ref.H_dep_l
                self.S_dep_ref_l = self.eos_phase_ref.S_dep_l

                self.H_dep_T_ref_Pb = self.eos.to_TP(self.T_ref, 101325).H_dep_l
                self.S_dep_T_ref_Pb = self.eos.to_TP(self.T_ref, 101325).S_dep_l

            if self.Tb:
                self.eos_Tb = self.eos.to_TP(self.Tb, 101325)
                self.H_dep_Tb_Pb_g = self.eos_Tb.H_dep_g
                self.H_dep_Tb_Pb_l = self.eos_Tb.H_dep_l

                self.H_dep_Tb_P_ref_g = self.eos.to_TP(self.Tb, self.P_ref).H_dep_g
                self.S_dep_Tb_P_ref_g = self.eos.to_TP(self.Tb, self.P_ref).S_dep_g


                self.S_dep_Tb_Pb_g = self.eos_Tb.S_dep_g
                self.S_dep_Tb_Pb_l = self.eos_Tb.S_dep_l

#            if self.Tt and self.Pt:
#                self.eos_Tt = self.eos.to_TP(self.Tt, self.Pt)
#                self.H_dep_Tt_g = self.eos_Tt.H_dep_g
##                self.H_dep_Tt_l = self.eos_Tt.H_dep_l
#
#                self.S_dep_Tt_g = self.eos_Tt.S_dep_g
##                self.S_dep_Tt_l = self.eos_Tt.S_dep_l
        except:
            pass

    def calc_H(self, T, P):

        integrators = {'s': self.HeatCapacitySolid.T_dependent_property_integral,
           'l': self.HeatCapacityLiquid.T_dependent_property_integral,
           'g': self.HeatCapacityGas.T_dependent_property_integral}
        try:
            H = self.H_ref
            if self.phase == self.phase_ref:
                H += integrators[self.phase](self.T_ref, T)

            elif self.phase_ref == 's' and self.phase == 'l':
                H += self.H_int_T_ref_s_to_Tm + self.Hfusm + integrators['l'](self.Tm, T)

            elif self.phase_ref == 'l' and self.phase == 's':
                H += -self.H_int_l_Tm_to_T_ref_l - self.Hfusm + integrators['s'](self.Tm, T)

            elif self.phase_ref == 'l' and self.phase == 'g':
                H += self.H_int_l_T_ref_l_to_Tb + self.Hvap_Tbm + integrators['g'](self.Tb, T)

            elif self.phase_ref == 'g' and self.phase == 'l':
                H += -self.H_int_Tb_to_T_ref_g - self.Hvap_Tbm + integrators['l'](self.Tb, T)

            elif self.phase_ref == 's' and self.phase == 'g':
                H += self.H_int_T_ref_s_to_Tm + self.Hfusm + self.H_int_l_Tm_to_Tb + self.Hvap_Tbm + integrators['g'](self.Tb, T)

            elif self.phase_ref == 'g' and self.phase == 's':
                H += -self.H_int_Tb_to_T_ref_g - self.Hvap_Tbm - self.H_int_l_Tm_to_Tb - self.Hfusm + integrators['s'](self.Tm, T)
            else:
                raise Exception('Unknown error')
        except:
            return None
        return H

    def calc_H_excess(self, T, P):
        H_dep = 0
        if self.phase_ref == 'g' and self.phase == 'g':
            H_dep += self.eos.to_TP(T, P).H_dep_g - self.H_dep_ref_g

        elif self.phase_ref == 'l' and self.phase == 'l':
            try:
                H_dep += self.eos.to_TP(T, P).H_dep_l - self._eos_T_101325.H_dep_l
            except:
                H_dep += 0

        elif self.phase_ref == 'g' and self.phase == 'l':
            H_dep += self.H_dep_Tb_Pb_g - self.H_dep_Tb_P_ref_g
            H_dep += (self.eos.to_TP(T, P).H_dep_l - self._eos_T_101325.H_dep_l)

        elif self.phase_ref == 'l' and self.phase == 'g':
            H_dep += self.H_dep_T_ref_Pb - self.H_dep_ref_l
            H_dep += (self.eos.to_TP(T, P).H_dep_g - self.H_dep_Tb_Pb_g)
        return H_dep

    def calc_S_excess(self, T, P):
        S_dep = 0
        if self.phase_ref == 'g' and self.phase == 'g':
            S_dep += self.eos.to_TP(T, P).S_dep_g - self.S_dep_ref_g

        elif self.phase_ref == 'l' and self.phase == 'l':
            try:
                S_dep += self.eos.to_TP(T, P).S_dep_l - self._eos_T_101325.S_dep_l
            except:
                S_dep += 0

        elif self.phase_ref == 'g' and self.phase == 'l':
            S_dep += self.S_dep_Tb_Pb_g - self.S_dep_Tb_P_ref_g
            S_dep += (self.eos.to_TP(T, P).S_dep_l - self._eos_T_101325.S_dep_l)

        elif self.phase_ref == 'l' and self.phase == 'g':
            S_dep += self.S_dep_T_ref_Pb - self.S_dep_ref_l
            S_dep += (self.eos.to_TP(T, P).S_dep_g - self.S_dep_Tb_Pb_g)
        return S_dep

    def calc_S(self, T, P):

        integrators_T = {'s': self.HeatCapacitySolid.T_dependent_property_integral_over_T,
           'l': self.HeatCapacityLiquid.T_dependent_property_integral_over_T,
           'g': self.HeatCapacityGas.T_dependent_property_integral_over_T}

        try:
            S = self.S_ref
            if self.phase == self.phase_ref:
                S += integrators_T[self.phase](self.T_ref, T)
                if self.phase in ['l', 'g']:
                    S += -R*log(P/self.P_ref)

            elif self.phase_ref == 's' and self.phase == 'l':
                S += self.S_int_T_ref_s_to_Tm + self.Hfusm/self.Tm + integrators_T['l'](self.Tm, T)

            elif self.phase_ref == 'l' and self.phase == 's':
                S += - self.S_int_l_Tm_to_T_ref_l - self.Hfusm/self.Tm + integrators_T['s'](self.Tm, T)

            elif self.phase_ref == 'l' and self.phase == 'g':
                S += self.S_int_l_T_ref_l_to_Tb + self.Hvap_Tbm/self.Tb + integrators_T['g'](self.Tb, T) -R*log(P/self.P_ref) # TODO add to other states

            elif self.phase_ref == 'g' and self.phase == 'l':
                S += - self.S_int_Tb_to_T_ref_g - self.Hvapm/self.Tb + integrators_T['l'](self.Tb, T)

            elif self.phase_ref == 's' and self.phase == 'g':
                S += self.S_int_T_ref_s_to_Tm + self.Hfusm/self.Tm + self.S_int_l_Tm_to_Tb + self.Hvap_Tbm/self.Tb + integrators_T['g'](self.Tb, T)

            elif self.phase_ref == 'g' and self.phase == 's':
                S += - self.S_int_Tb_to_T_ref_g - self.Hvap_Tbm/self.Tb - self.S_int_l_Tm_to_Tb - self.Hfusm/self.Tm + integrators_T['s'](self.Tm, T)
            else:
                raise Exception('Unknown error')
        except:
            return None
        return S

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

    def set_thermo(self):
        try:
            self._eos_T_101325 = self.eos.to_TP(self.T, 101325)


            self.Hm = self.calc_H(self.T, self.P)
            self.Hm += self.calc_H_excess(self.T, self.P)
            self.H = property_molar_to_mass(self.Hm, self.MW) if (self.Hm is not None) else None

            self.Sm = self.calc_S(self.T, self.P)
            self.Sm += self.calc_S_excess(self.T, self.P)
            self.S = property_molar_to_mass(self.Sm, self.MW) if (self.Sm is not None) else None

            self.G = self.H - self.T*self.S if (self.H is not None and self.S is not None) else None
            self.Gm = self.Hm - self.T*self.Sm if (self.Hm is not None and self.Sm is not None) else None
        except:
            pass

    @property
    def Um(self):
        r'''Internal energy of the chemical at its current temperature and
        pressure, in units of [J/mol].

        This property requires that :obj:`thermo.chemical.set_thermo` ran
        successfully to be accurate.
        It also depends on the molar volume of the chemical at its current
        conditions.
        '''
        return self.Hm - self.P*self.Vm if (self.Vm and self.Hm is not None) else None

    @property
    def U(self):
        r'''Internal energy of the chemical at its current temperature and
        pressure, in units of [J/kg].

        This property requires that :obj:`thermo.chemical.set_thermo` ran
        successfully to be accurate.
        It also depends on the molar volume of the chemical at its current
        conditions.
        '''
        return property_molar_to_mass(self.Um, self.MW) if (self.Um is not None) else None

    @property
    def Am(self):
        r'''Helmholtz energy of the chemical at its current temperature and
        pressure, in units of [J/mol].

        This property requires that :obj:`thermo.chemical.set_thermo` ran
        successfully to be accurate.
        It also depends on the molar volume of the chemical at its current
        conditions.
        '''
        return self.Um - self.T*self.Sm if (self.Um is not None and self.Sm is not None) else None

    @property
    def A(self):
        r'''Helmholtz energy of the chemical at its current temperature and
        pressure, in units of [J/kg].

        This property requires that :obj:`thermo.chemical.set_thermo` ran
        successfully to be accurate.
        It also depends on the molar volume of the chemical at its current
        conditions.
        '''
        return self.U - self.T*self.S if (self.U is not None and self.S is not None) else None


    ### Temperature independent properties - calculate lazily
    @property
    def charge(self):
        r'''Charge of a chemical, computed with RDKit from a chemical's SMILES.
        If RDKit is not available, holds None.

        Examples
        --------
        >>> Chemical('sodium ion').charge
        1
        '''
        try:
            if not self.rdkitmol:
                return charge_from_formula(self.formula)
            else:
                return Chem.GetFormalCharge(self.rdkitmol)
        except:
            return charge_from_formula(self.formula)

    @property
    def rings(self):
        r'''Number of rings in a chemical, computed with RDKit from a
        chemical's SMILES. If RDKit is not available, holds None.

        Examples
        --------
        >>> Chemical('Paclitaxel').rings
        7
        '''
        try:
            return Chem.Descriptors.RingCount(self.rdkitmol)
        except:
            return None

    @property
    def aromatic_rings(self):
        r'''Number of aromatic rings in a chemical, computed with RDKit from a
        chemical's SMILES. If RDKit is not available, holds None.

        Examples
        --------
        >>> Chemical('Paclitaxel').aromatic_rings
        3
        '''
        try:
            return Chem.Descriptors.NumAromaticRings(self.rdkitmol)
        except:
            return None

    @property
    def rdkitmol(self):
        r'''RDKit object of the chemical, without hydrogen. If RDKit is not
        available, holds None.

        For examples of what can be done with RDKit, see
        `their website <http://www.rdkit.org/docs/GettingStartedInPython.html>`_.
        '''
        if self.__rdkitmol:
            return self.__rdkitmol
        else:
            try:
                self.__rdkitmol = Chem.MolFromSmiles(self.smiles)
                return self.__rdkitmol
            except:
                return None

    @property
    def rdkitmol_Hs(self):
        r'''RDKit object of the chemical, with hydrogen. If RDKit is not
        available, holds None.

        For examples of what can be done with RDKit, see
        `their website <http://www.rdkit.org/docs/GettingStartedInPython.html>`_.
        '''
        if self.__rdkitmol_Hs:
            return self.__rdkitmol_Hs
        else:
            try:
                self.__rdkitmol_Hs = Chem.AddHs(self.rdkitmol)
                return self.__rdkitmol_Hs
            except:
                return None

    @property
    def Hill(self):
        r'''Hill formula of a compound. For a description of the Hill system,
        see :obj:`thermo.elements.atoms_to_Hill`.

        Examples
        --------
        >>> Chemical('furfuryl alcohol').Hill
        'C5H6O2'
        '''
        if self.__Hill:
            return self.__Hill
        else:
            self.__Hill = atoms_to_Hill(self.atoms)
            return self.__Hill

    @property
    def atom_fractions(self):
        r'''Dictionary of atom:fractional occurence of the elements in a
        chemical. Useful when performing element balances. For mass-fraction
        occurences, see :obj:`mass_fractions`.

        Examples
        --------
        >>> Chemical('Ammonium aluminium sulfate').atom_fractions
        {'H': 0.25, 'S': 0.125, 'Al': 0.0625, 'O': 0.5, 'N': 0.0625}
        '''
        if self.__atom_fractions:
            return self.__atom_fractions
        else:
            self.__atom_fractions = atom_fractions(self.atoms)
            return self.__atom_fractions

    @property
    def mass_fractions(self):
        r'''Dictionary of atom:mass-weighted fractional occurence of elements.
        Useful when performing mass balances. For atom-fraction occurences, see
        :obj:`atom_fractions`.

        Examples
        --------
        >>> Chemical('water').mass_fractions
        {'H': 0.11189834407236524, 'O': 0.8881016559276347}
        '''
        if self.__mass_fractions:
            return self.__mass_fractions
        else:
            self.__mass_fractions =  mass_fractions(self.atoms, self.MW)
            return self.__mass_fractions

    @property
    def legal_status(self):
        r'''Dictionary of legal status indicators for the chemical.

        Examples
        --------
        >>> pprint(Chemical('benzene').legal_status)
        {'DSL': 'LISTED',
         'EINECS': 'LISTED',
         'NLP': 'UNLISTED',
         'SPIN': 'LISTED',
         'TSCA': 'LISTED'}
        '''
        if self.__legal_status:
            return self.__legal_status
        else:
            self.__legal_status = legal_status(self.CAS, Method='COMBINED')
            return self.__legal_status

    @property
    def economic_status(self):
        r'''Dictionary of economic status indicators for the chemical.

        Examples
        --------
        >>> pprint(Chemical('benzene').economic_status)
        ["US public: {'Manufactured': 6165232.1, 'Imported': 463146.474, 'Exported': 271908.252}",
         u'1,000,000 - 10,000,000 tonnes per annum',
         u'Intermediate Use Only',
         'OECD HPV Chemicals']
        '''
        if self.__economic_status:
            return self.__economic_status
        else:
            self.__economic_status = economic_status(self.CAS, Method='Combined')
            return self.__economic_status


    @property
    def UNIFAC_groups(self):
        r'''Dictionary of UNIFAC subgroup: count groups for the original
        UNIFAC subgroups, as determined by `DDBST's online service <http://www.ddbst.com/unifacga.html>`_.

        Examples
        --------
        >>> pprint(Chemical('Cumene').UNIFAC_groups)
        {1: 2, 9: 5, 13: 1}
        '''
        if self.__UNIFAC_groups:
            return self.__UNIFAC_groups
        else:
            load_group_assignments_DDBST()
            if self.InChI_Key in DDBST_UNIFAC_assignments:
                self.__UNIFAC_groups = DDBST_UNIFAC_assignments[self.InChI_Key]
                return self.__UNIFAC_groups
            else:
                return None

    @property
    def UNIFAC_Dortmund_groups(self):
        r'''Dictionary of Dortmund UNIFAC subgroup: count groups for the
        Dortmund UNIFAC subgroups, as determined by `DDBST's online service <http://www.ddbst.com/unifacga.html>`_.

        Examples
        --------
        >>> pprint(Chemical('Cumene').UNIFAC_Dortmund_groups)
        {1: 2, 9: 5, 13: 1}
        '''
        if self.__UNIFAC_Dortmund_groups:
            return self.__UNIFAC_Dortmund_groups
        else:
            load_group_assignments_DDBST()
            if self.InChI_Key in DDBST_MODIFIED_UNIFAC_assignments:
                self.__UNIFAC_Dortmund_groups = DDBST_MODIFIED_UNIFAC_assignments[self.InChI_Key]
                return self.__UNIFAC_Dortmund_groups
            else:
                return None

    @property
    def PSRK_groups(self):
        r'''Dictionary of PSRK subgroup: count groups for the PSRK subgroups,
        as determined by `DDBST's online service <http://www.ddbst.com/unifacga.html>`_.

        Examples
        --------
        >>> pprint(Chemical('Cumene').PSRK_groups)
        {1: 2, 9: 5, 13: 1}
        '''
        if self.__PSRK_groups:
            return self.__PSRK_groups
        else:
            load_group_assignments_DDBST()
            if self.InChI_Key in DDBST_PSRK_assignments:
                self.__PSRK_groups = DDBST_PSRK_assignments[self.InChI_Key]
                return self.__PSRK_groups
            else:
                return None

    @property
    def UNIFAC_R(self):
        r'''UNIFAC `R` (normalized Van der Waals volume), dimensionless.
        Used in the UNIFAC model.

        Examples
        --------
        >>> Chemical('benzene').UNIFAC_R
        3.1878
        '''
        if self.UNIFAC_groups:
            return UNIFAC_RQ(self.UNIFAC_groups)[0]
        return None

    @property
    def UNIFAC_Q(self):
        r'''UNIFAC `Q` (normalized Van der Waals area), dimensionless.
        Used in the UNIFAC model.

        Examples
        --------
        >>> Chemical('decane').UNIFAC_Q
        6.016
        '''
        if self.UNIFAC_groups:
            return UNIFAC_RQ(self.UNIFAC_groups)[1]
        return None

    @property
    def Van_der_Waals_volume(self):
        r'''Unnormalized Van der Waals volume, in units of [m^3/mol].

        Examples
        --------
        >>> Chemical('hexane').Van_der_Waals_volume
        6.8261966e-05
        '''
        if self.UNIFAC_R:
            return Van_der_Waals_volume(self.UNIFAC_R)
        return None

    @property
    def Van_der_Waals_area(self):
        r'''Unnormalized Van der Waals area, in units of [m^2/mol].

        Examples
        --------
        >>> Chemical('hexane').Van_der_Waals_area
        964000.0
        '''
        if self.UNIFAC_Q:
            return Van_der_Waals_area(self.UNIFAC_Q)
        return None

    @property
    def R_specific(self):
        r'''Specific gas constant, in units of [J/kg/K].

        Examples
        --------
        >>> Chemical('water').R_specific
        461.52265188218
        '''
        return property_molar_to_mass(R, self.MW)

    ### One phase properties - calculate lazily
    @property
    def Psat(self):
        r'''Vapor pressure of the chemical at its current temperature, in units
        of [Pa]. For calculation of this property at other temperatures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface :obj:`thermo.vapor_pressure.VaporPressure`;
        each Chemical instance creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('water', T=320).Psat
        10533.614271198725
        >>> Chemical('water').VaporPressure.T_dependent_property(320)
        10533.614271198725
        >>> Chemical('water').VaporPressure.all_methods
        set(['VDI_PPDS', 'BOILING_CRITICAL', 'WAGNER_MCGARRY', 'AMBROSE_WALTON', 'COOLPROP', 'LEE_KESLER_PSAT', 'EOS', 'ANTOINE_POLING', 'SANJARI', 'DIPPR_PERRY_8E', 'Edalat'])
        '''
        return self.VaporPressure(self.T)

    @property
    def Hvapm(self):
        r'''Enthalpy of vaporization of the chemical at its current temperature,
        in units of [J/mol]. For calculation of this property at other
        temperatures, or specifying manually the method used to calculate it,
        and more - see the object oriented interface
        :obj:`thermo.phase_change.EnthalpyVaporization`; each Chemical instance
        creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('water', T=320).Hvapm
        43048.23612280223
        >>> Chemical('water').EnthalpyVaporization.T_dependent_property(320)
        43048.23612280223
        >>> Chemical('water').EnthalpyVaporization.all_methods
        set(['VDI_PPDS', 'MORGAN_KOBAYASHI', 'VETERE', 'VELASCO', 'LIU', 'COOLPROP', 'CRC_HVAP_298', 'CLAPEYRON', 'SIVARAMAN_MAGEE_KOBAYASHI', 'ALIBAKHSHI', 'DIPPR_PERRY_8E', 'RIEDEL', 'CHEN', 'PITZER', 'CRC_HVAP_TB'])
        '''
        return self.EnthalpyVaporization(self.T)

    @property
    def Hvap(self):
        r'''Enthalpy of vaporization of the chemical at its current temperature,
        in units of [J/kg].

        This property uses the object-oriented interface
        :obj:`thermo.phase_change.EnthalpyVaporization`, but converts its
        results from molar to mass units.

        Examples
        --------
        >>> Chemical('water', T=320).Hvap
        2389540.219347256
        '''
        Hvamp = self.Hvapm
        if Hvamp:
            return property_molar_to_mass(Hvamp, self.MW)
        return None

    @property
    def Cpsm(self):
        r'''Solid-phase heat capacity of the chemical at its current temperature,
        in units of [J/mol/K]. For calculation of this property at other
        temperatures, or specifying manually the method used to calculate it,
        and more - see the object oriented interface
        :obj:`thermo.heat_capacity.HeatCapacitySolid`; each Chemical instance
        creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('palladium').Cpsm
        24.930765664000003
        >>> Chemical('palladium').HeatCapacitySolid.T_dependent_property(320)
        25.098979200000002
        >>> Chemical('palladium').HeatCapacitySolid.all_methods
        set(["Perry's Table 2-151", 'CRC Standard Thermodynamic Properties of Chemical Substances', 'Lastovka, Fulem, Becerra and Shaw (2008)'])
        '''
        return self.HeatCapacitySolid(self.T)

    @property
    def Cplm(self):
        r'''Liquid-phase heat capacity of the chemical at its current temperature,
        in units of [J/mol/K]. For calculation of this property at other
        temperatures, or specifying manually the method used to calculate it,
        and more - see the object oriented interface
        :obj:`thermo.heat_capacity.HeatCapacityLiquid`; each Chemical instance
        creates one to actually perform the calculations.

        Notes
        -----
        Some methods give heat capacity along the saturation line, some at
        1 atm but only up to the normal boiling point, and some give heat
        capacity at 1 atm up to the normal boiling point and then along the
        saturation line. Real-liquid heat capacity is pressure dependent, but
        this interface is not.

        Examples
        --------
        >>> Chemical('water').Cplm
        75.31462591538556
        >>> Chemical('water').HeatCapacityLiquid.T_dependent_property(320)
        75.2591744360631
        >>> Chemical('water').HeatCapacityLiquid.T_dependent_property_integral(300, 320)
        1505.0619005000553
        '''
        return self.HeatCapacityLiquid(self.T)

    @property
    def Cpgm(self):
        r'''Gas-phase ideal gas heat capacity of the chemical at its current
        temperature, in units of [J/mol/K]. For calculation of this property at
        other temperatures, or specifying manually the method used to calculate
        it, and more - see the object oriented interface
        :obj:`thermo.heat_capacity.HeatCapacityGas`; each Chemical instance
        creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('water').Cpgm
        33.583577868850675
        >>> Chemical('water').HeatCapacityGas.T_dependent_property(320)
        33.67865044005934
        >>> Chemical('water').HeatCapacityGas.T_dependent_property_integral(300, 320)
        672.6480417835064
        '''
        return self.HeatCapacityGas(self.T)

    @property
    def Cps(self):
        r'''Solid-phase heat capacity of the chemical at its current temperature,
        in units of [J/kg/K]. For calculation of this property at other
        temperatures, or specifying manually the method used to calculate it,
        and more - see the object oriented interface
        :obj:`thermo.heat_capacity.HeatCapacitySolid`; each Chemical instance
        creates one to actually perform the calculations. Note that that
        interface provides output in molar units.

        Examples
        --------
        >>> Chemical('palladium', T=400).Cps
        241.63563239992484
        >>> Pd = Chemical('palladium', T=400)
        >>> Cpsms = [Pd.HeatCapacitySolid.T_dependent_property(T) for T in np.linspace(300,500, 5)]
        >>> [property_molar_to_mass(Cps, Pd.MW) for Cps in Cpsms]
        [234.40150347679008, 238.01856793835751, 241.63563239992484, 245.25269686149224, 248.86976132305958]
        '''
        Cpsm = self.HeatCapacitySolid(self.T)
        if Cpsm:
            return property_molar_to_mass(Cpsm, self.MW)
        return None

    @property
    def Cpl(self):
        r'''Liquid-phase heat capacity of the chemical at its current temperature,
        in units of [J/kg/K]. For calculation of this property at other
        temperatures, or specifying manually the method used to calculate it,
        and more - see the object oriented interface
        :obj:`thermo.heat_capacity.HeatCapacityLiquid`; each Chemical instance
        creates one to actually perform the calculations. Note that that
        interface provides output in molar units.

        Examples
        --------
        >>> Chemical('water', T=320).Cpl
        4177.518996988284

        Ideal entropy change of water from 280 K to 340 K, output converted
        back to mass-based units of J/kg/K.

        >>> dSm = Chemical('water').HeatCapacityLiquid.T_dependent_property_integral_over_T(280, 340)
        >>> property_molar_to_mass(dSm, Chemical('water').MW)
        812.1024585274956
        '''
        Cplm = self.HeatCapacityLiquid(self.T)
        if Cplm:
            return property_molar_to_mass(Cplm, self.MW)
        return None

    @property
    def Cpg(self):
        r'''Gas-phase heat capacity of the chemical at its current temperature,
        in units of [J/kg/K]. For calculation of this property at other
        temperatures, or specifying manually the method used to calculate it,
        and more - see the object oriented interface
        :obj:`thermo.heat_capacity.HeatCapacityGas`; each Chemical instance
        creates one to actually perform the calculations. Note that that
        interface provides output in molar units.

        Examples
        --------
        >>> w = Chemical('water', T=520)
        >>> w.Cpg
        1967.6698314620658
        '''
        Cpgm = self.HeatCapacityGas(self.T)
        if Cpgm:
            return property_molar_to_mass(Cpgm, self.MW)
        return None

    @property
    def Cvgm(self):
        r'''Gas-phase ideal-gas contant-volume heat capacity of the chemical at
        its current temperature, in units of [J/mol/K]. Subtracts R from
        the ideal-gas heat capacity; does not include pressure-compensation
        from an equation of state.

        Examples
        --------
        >>> w = Chemical('water', T=520)
        >>> w.Cvgm
        27.13366316134193
        '''
        Cpgm = self.HeatCapacityGas(self.T)
        if Cpgm:
            return Cpgm - R
        return None

    @property
    def Cvg(self):
        r'''Gas-phase ideal-gas contant-volume heat capacity of the chemical at
        its current temperature, in units of [J/kg/K]. Subtracts R from
        the ideal-gas heat capacity; does not include pressure-compensation
        from an equation of state.

        Examples
        --------
        >>> w = Chemical('water', T=520)
        >>> w.Cvg
        1506.1471795798861
        '''
        Cvgm = self.Cvgm
        if Cvgm:
            return property_molar_to_mass(Cvgm, self.MW)
        return None

    @property
    def isentropic_exponent(self):
        r'''Gas-phase ideal-gas isentropic exponent of the chemical at its
        current temperature, [dimensionless]. Does not include
        pressure-compensation from an equation of state.

        Examples
        --------
        >>> Chemical('hydrogen').isentropic_exponent
        1.405237786321222
        '''
        Cp, Cv = self.Cpg, self.Cvg
        if all((Cp, Cv)):
            return isentropic_exponent(Cp, Cv)
        return None

    @property
    def Vms(self):
        r'''Solid-phase molar volume of the chemical at its current
        temperature, in units of [mol/m^3]. For calculation of this property at
        other temperatures, or specifying manually the method used to calculate
        it, and more - see the object oriented interface
        :obj:`thermo.volume.VolumeSolid`; each Chemical instance
        creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('iron').Vms
        7.09593392630242e-06
        '''
        return self.VolumeSolid(self.T)

    @property
    def Vml(self):
        r'''Liquid-phase molar volume of the chemical at its current
        temperature and pressure, in units of [mol/m^3]. For calculation of this
        property at other temperatures or pressures, or specifying manually the
        method used to calculate it, and more - see the object oriented interface
        :obj:`thermo.volume.VolumeLiquid`; each Chemical instance
        creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('cyclobutane', T=225).Vml
        7.42395423425395e-05
        '''
        return self.VolumeLiquid(self.T, self.P)

    @property
    def Vmg(self):
        r'''Gas-phase molar volume of the chemical at its current
        temperature and pressure, in units of [mol/m^3]. For calculation of this
        property at other temperatures or pressures, or specifying manually the
        method used to calculate it, and more - see the object oriented interface
        :obj:`thermo.volume.VolumeGas`; each Chemical instance
        creates one to actually perform the calculations.

        Examples
        --------
        Estimate the molar volume of the core of the sun, at 15 million K and
        26.5 PetaPascals, assuming pure helium (actually 68% helium):

        >>> Chemical('helium', T=15E6, P=26.5E15).Vmg
        4.805464238181197e-07
        '''
        return self.VolumeGas(self.T, self.P)

    @property
    def rhos(self):
        r'''Solid-phase mass density of the chemical at its current temperature,
        in units of [kg/m^3]. For calculation of this property at
        other temperatures, or specifying manually the method used
        to calculate it, and more - see the object oriented interface
        :obj:`thermo.volume.VolumeSolid`; each Chemical instance
        creates one to actually perform the calculations. Note that that
        interface provides output in molar units.

        Examples
        --------
        >>> Chemical('iron').rhos
        7869.999999999994
        '''
        Vms = self.Vms
        if Vms:
            return Vm_to_rho(Vms, self.MW)
        return None

    @property
    def rhol(self):
        r'''Liquid-phase mass density of the chemical at its current
        temperature and pressure, in units of [kg/m^3]. For calculation of this
        property at other temperatures and pressures, or specifying manually
        the method used to calculate it, and more - see the object oriented
        interface :obj:`thermo.volume.VolumeLiquid`; each Chemical instance
        creates one to actually perform the calculations. Note that that
        interface provides output in molar units.

        Examples
        --------
        >>> Chemical('o-xylene', T=297).rhol
        876.9946785618097
        '''
        Vml = self.Vml
        if Vml:
            return Vm_to_rho(Vml, self.MW)
        return None

    @property
    def rhog(self):
        r'''Gas-phase mass density of the chemical at its current temperature
        and pressure, in units of [kg/m^3]. For calculation of this property at
        other temperatures or pressures, or specifying manually the method used
        to calculate it, and more - see the object oriented interface
        :obj:`thermo.volume.VolumeGas`; each Chemical instance
        creates one to actually perform the calculations. Note that that
        interface provides output in molar units.

        Examples
        --------
        Estimate the density of the core of the sun, at 15 million K and
        26.5 PetaPascals, assuming pure helium (actually 68% helium):

        >>> Chemical('helium', T=15E6, P=26.5E15).rhog
        8329.27226509739

        Compared to a result on
        `Wikipedia <https://en.wikipedia.org/wiki/Solar_core>`_ of 150000
        kg/m^3, the fundamental equation of state performs poorly.

        >>> He = Chemical('helium', T=15E6, P=26.5E15)
        >>> He.VolumeGas.set_user_methods_P(['IDEAL']); He.rhog
        850477.8065477367

        The ideal-gas law performs somewhat better, but vastly overshoots
        the density prediction.
        '''
        Vmg = self.Vmg
        if Vmg:
            return Vm_to_rho(Vmg, self.MW)
        return None

    @property
    def rhosm(self):
        r'''Molar density of the chemical in the solid phase at the
        current temperature and pressure, in units of [mol/m^3].

        Utilizes the object oriented interface and
        :obj:`thermo.volume.VolumeSolid` to perform the actual calculation of
        molar volume.

        Examples
        --------
        >>> Chemical('palladium').rhosm
        112760.75925577903
        '''
        Vms = self.Vms
        if Vms:
            return 1./Vms
        return None

    @property
    def rholm(self):
        r'''Molar density of the chemical in the liquid phase at the
        current temperature and pressure, in units of [mol/m^3].

        Utilizes the object oriented interface and
        :obj:`thermo.volume.VolumeLiquid` to perform the actual calculation of
        molar volume.

        Examples
        --------
        >>> Chemical('nitrogen', T=70).rholm
        29937.20179186975
        '''
        Vml = self.Vml
        if Vml:
            return 1./Vml
        return None

    @property
    def rhogm(self):
        r'''Molar density of the chemical in the gas phase at the
        current temperature and pressure, in units of [mol/m^3].

        Utilizes the object oriented interface and
        :obj:`thermo.volume.VolumeGas` to perform the actual calculation of
        molar volume.

        Examples
        --------
        >>> Chemical('tungsten hexafluoride').rhogm
        42.01349946063116
        '''
        Vmg = self.Vmg
        if Vmg:
            return 1./Vmg
        return None

    @property
    def Zs(self):
        r'''Compressibility factor of the chemical in the solid phase at the
        current temperature and pressure, [dimensionless].

        Utilizes the object oriented interface and
        :obj:`thermo.volume.VolumeSolid` to perform the actual calculation of
        molar volume.

        Examples
        --------
        >>> Chemical('palladium').Z
        0.00036248477437931853
        '''
        Vms = self.Vms
        if Vms:
            return Z(self.T, self.P, Vms)
        return None

    @property
    def Zl(self):
        r'''Compressibility factor of the chemical in the liquid phase at the
        current temperature and pressure, [dimensionless].

        Utilizes the object oriented interface and
        :obj:`thermo.volume.VolumeLiquid` to perform the actual calculation of
        molar volume.

        Examples
        --------
        >>> Chemical('water').Zl
        0.0007385375470263454
        '''
        Vml = self.Vml
        if Vml:
            return Z(self.T, self.P, Vml)
        return None

    @property
    def Zg(self):
        r'''Compressibility factor of the chemical in the gas phase at the
        current temperature and pressure, [dimensionless].

        Utilizes the object oriented interface and
        :obj:`thermo.volume.VolumeGas` to perform the actual calculation of
        molar volume.

        Examples
        --------
        >>> Chemical('sulfur hexafluoride', T=700, P=1E9).Zg
        11.140084184207813
        '''
        Vmg = self.Vmg
        if Vmg:
            return Z(self.T, self.P, Vmg)
        return None

    @property
    def SGs(self):
        r'''Specific gravity of the solid phase of the chemical at the 
        specified temperature and pressure, [dimensionless].
        The reference condition is water at 4 °C and 1 atm 
        (rho=999.017 kg/m^3). The SG varries with temperature and pressure
        but only very slightly.
        
        Examples
        --------
        >>> Chemical('iron').SGs
        7.87774317235069
        '''
        rhos = self.rhos
        if rhos is not None:
            return SG(rhos)
        return None

    @property
    def SGl(self):
        r'''Specific gravity of the liquid phase of the chemical at the 
        specified temperature and pressure, [dimensionless].
        The reference condition is water at 4 °C and 1 atm 
        (rho=999.017 kg/m^3). For liquids, SG is defined that the reference
        chemical's T and P are fixed, but the chemical itself varies with
        the specified T and P.
        
        Examples
        --------
        >>> Chemical('water', T=365).SGl
        0.9650065522428539
        '''
        rhol = self.rhol
        if rhol is not None:
            return SG(rhol)
        return None
    
    @property
    def SGg(self):
        r'''Specific gravity of the gas phase of the chemical, [dimensionless].
        The reference condition is air at 15.6 °C (60 °F) and 1 atm 
        (rho=1.223 kg/m^3). The definition for gases uses the compressibility
        factor of the reference gas and the chemical both at the reference
        conditions, not the conditions of the chemical.
            
        Examples
        --------
        >>> Chemical('argon').SGg
        1.3795835970877504
        '''
        Vmg = self.VolumeGas(T=288.70555555555552, P=101325)
        if Vmg:
            rho = Vm_to_rho(Vmg, self.MW)
            return SG(rho, rho_ref=1.2231876628642968) # calculated with Mixture
        return None
    
    @property
    def API(self):
        r'''API gravity of the liquid phase of the chemical, [degrees].
        The reference condition is water at 15.6 °C (60 °F) and 1 atm 
        (rho=999.016 kg/m^3, standardized).
            
        Examples
        --------
        >>> Chemical('water').API
        9.999752435378895
        '''
        Vml = self.VolumeLiquid(T=288.70555555555552, P=101325)
        if Vml:
            rho = Vm_to_rho(Vml, self.MW)
        sg = SG(rho, rho_ref=999.016)
        return SG_to_API(sg)
        
    @property
    def Bvirial(self):
        r'''Second virial coefficient of the gas phase of the chemical at its
        current temperature and pressure, in units of [mol/m^3].

        This property uses the object-oriented interface
        :obj:`thermo.volume.VolumeGas`, converting its result with
        :obj:`thermo.utils.B_from_Z`.

        Examples
        --------
        >>> Chemical('water').Bvirial
        -0.0009596286322838357
        '''
        if self.Vmg:
            return B_from_Z(self.Zg, self.T, self.P)
        return None

    @property
    def isobaric_expansion_l(self):
        r'''Isobaric (constant-pressure) expansion of the liquid phase of the
        chemical at its current temperature and pressure, in units of [1/K].

        .. math::
            \beta = \frac{1}{V}\left(\frac{\partial V}{\partial T} \right)_P

        Utilizes the temperature-derivative method of
        :obj:`thermo.volume.VolumeLiquid` to perform the actual calculation.
        The derivatives are all numerical.

        Examples
        --------
        >>> Chemical('dodecane', T=400).isobaric_expansion_l
        0.0011617555762469477
        '''
        dV_dT = self.VolumeLiquid.TP_dependent_property_derivative_T(self.T, self.P)
        Vm = self.Vml
        if dV_dT and Vm:
            return isobaric_expansion(V=Vm, dV_dT=dV_dT)

    @property
    def isobaric_expansion_g(self):
        r'''Isobaric (constant-pressure) expansion of the gas phase of the
        chemical at its current temperature and pressure, in units of [1/K].

        .. math::
            \beta = \frac{1}{V}\left(\frac{\partial V}{\partial T} \right)_P

        Utilizes the temperature-derivative method of
        :obj:`thermo.VolumeGas` to perform the actual calculation.
        The derivatives are all numerical.

        Examples
        --------
        >>> Chemical('Hexachlorobenzene', T=900).isobaric_expansion_g
        0.001151869741981048
        '''
        dV_dT = self.VolumeGas.TP_dependent_property_derivative_T(self.T, self.P)
        Vm = self.Vmg
        if dV_dT and Vm:
            return isobaric_expansion(V=Vm, dV_dT=dV_dT)

    @property
    def mul(self):
        r'''Viscosity of the chemical in the liquid phase at its current
        temperature and pressure, in units of [Pa*s].

        For calculation of this property at other temperatures and pressures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface
        :obj:`thermo.viscosity.ViscosityLiquid`; each Chemical instance
        creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('water', T=320).mul
        0.0005767262693751547
        '''
        return self.ViscosityLiquid(self.T, self.P)

    @property
    def mug(self):
        r'''Viscosity of the chemical in the gas phase at its current
        temperature and pressure, in units of [Pa*s].

        For calculation of this property at other temperatures and pressures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface
        :obj:`thermo.viscosity.ViscosityGas`; each Chemical instance
        creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('water', T=320, P=100).mug
        1.0431450856297212e-05
        '''
        return self.ViscosityGas(self.T, self.P)

    @property
    def kl(self):
        r'''Thermal conductivity of the chemical in the liquid phase at its
        current temperature and pressure, in units of [W/m/K].

        For calculation of this property at other temperatures and pressures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface
        :obj:`thermo.thermal_conductivity.ThermalConductivityLiquid`; each
        Chemical instance creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('water', T=320).kl
        0.6369957248212118
        '''
        return self.ThermalConductivityLiquid(self.T, self.P)

    @property
    def kg(self):
        r'''Thermal conductivity of the chemical in the gas phase at its
        current temperature and pressure, in units of [W/m/K].

        For calculation of this property at other temperatures and pressures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface
        :obj:`thermo.thermal_conductivity.ThermalConductivityGas`; each
        Chemical instance creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('water', T=320).kg
        0.021273128263091207
        '''
        return self.ThermalConductivityGas(self.T, self.P)

    @property
    def sigma(self):
        r'''Surface tension of the chemical at its current temperature, in
        units of [N/m].

        For calculation of this property at other temperatures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface :obj:`thermo.interface.SurfaceTension`;
        each Chemical instance creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('water', T=320).sigma
        0.06855002575793023
        >>> Chemical('water', T=320).SurfaceTension.solve_prop(0.05)
        416.8307110842183
        '''
        return self.SurfaceTension(self.T)

    @property
    def permittivity(self):
        r'''Relative permittivity (dielectric constant) of the chemical at its 
        current temperature, [dimensionless].

        For calculation of this property at other temperatures,
        or specifying manually the method used to calculate it, and more - see
        the object oriented interface :obj:`thermo.permittivity.Permittivity`;
        each Chemical instance creates one to actually perform the calculations.

        Examples
        --------
        >>> Chemical('toluene', T=250).permittivity
        2.49775625
        '''
        return self.Permittivity(self.T)
    
    @property
    def absolute_permittivity(self):
        r'''Absolute permittivity of the chemical at its current temperature,
        in units of [farad/meter]. Those units are equivalent to 
        ampere^2*second^4/kg/m^3.

        Examples
        --------
        >>> Chemical('water', T=293.15).absolute_permittivity
        7.096684821859018e-10
        '''
        permittivity = self.permittivity
        if permittivity is not None:
            return permittivity*physical_constants['electric constant'][0]
        return None

    @property
    def JTl(self):
        r'''Joule Thomson coefficient of the chemical in the liquid phase at
        its current temperature and pressure, in units of [K/Pa].

        .. math::
            \mu_{JT} = \left(\frac{\partial T}{\partial P}\right)_H = \frac{1}{C_p}
            \left[T \left(\frac{\partial V}{\partial T}\right)_P - V\right]
            = \frac{V}{C_p}\left(\beta T-1\right)

        Utilizes the temperature-derivative method of
        :obj:`thermo.volume.VolumeLiquid` and the temperature-dependent heat
        capacity method :obj:`thermo.heat_capacity.HeatCapacityLiquid` to
        obtain the properties required for the actual calculation.

        Examples
        --------
        >>> Chemical('dodecane', T=400).JTl
        -3.1037120844444807e-07
        '''
        Vml, Cplm, isobaric_expansion_l = self.Vml, self.Cplm, self.isobaric_expansion_l
        if all((Vml, Cplm, isobaric_expansion_l)):
            return Joule_Thomson(T=self.T, V=Vml, Cp=Cplm, beta=isobaric_expansion_l)
        return None

    @property
    def JTg(self):
        r'''Joule Thomson coefficient of the chemical in the gas phase at
        its current temperature and pressure, in units of [K/Pa].

        .. math::
            \mu_{JT} = \left(\frac{\partial T}{\partial P}\right)_H = \frac{1}{C_p}
            \left[T \left(\frac{\partial V}{\partial T}\right)_P - V\right]
            = \frac{V}{C_p}\left(\beta T-1\right)

        Utilizes the temperature-derivative method of
        :obj:`thermo.volume.VolumeGas` and the temperature-dependent heat
        capacity method :obj:`thermo.heat_capacity.HeatCapacityGas` to
        obtain the properties required for the actual calculation.

        Examples
        --------
        >>> Chemical('dodecane', T=400, P=1000).JTg
        5.4089897835384913e-05
        '''
        Vmg, Cpgm, isobaric_expansion_g = self.Vmg, self.Cpgm, self.isobaric_expansion_g
        if all((Vmg, Cpgm, isobaric_expansion_g)):
            return Joule_Thomson(T=self.T, V=Vmg, Cp=Cpgm, beta=isobaric_expansion_g)
        return None

    @property
    def nul(self):
        r'''Kinematic viscosity of the liquid phase of the chemical at its
        current temperature and pressure, in units of [m^2/s].

        .. math::
            \nu = \frac{\mu}{\rho}

        Utilizes the temperature and pressure dependent object oriented
        interfaces :obj:`thermo.volume.VolumeLiquid`,
        :obj:`thermo.viscosity.ViscosityLiquid`  to calculate the
        actual properties.

        Examples
        --------
        >>> Chemical('methane', T=110).nul
        2.858184674118658e-07
        '''
        mul, rhol = self.mul, self.rhol
        if all([mul, rhol]):
            return nu_mu_converter(mu=mul, rho=rhol)
        return None

    @property
    def nug(self):
        r'''Kinematic viscosity of the gas phase of the chemical at its
        current temperature and pressure, in units of [m^2/s].

        .. math::
            \nu = \frac{\mu}{\rho}

        Utilizes the temperature and pressure dependent object oriented
        interfaces :obj:`thermo.volume.VolumeGas`,
        :obj:`thermo.viscosity.ViscosityGas`  to calculate the
        actual properties.

        Examples
        --------
        >>> Chemical('methane', T=115).nug
        2.5119305527611988e-06
        '''
        mug, rhog = self.mug, self.rhog
        if all([mug, rhog]):
            return nu_mu_converter(mu=mug, rho=rhog)
        return None

    @property
    def alphal(self):
        r'''Thermal diffusivity of the liquid phase of the chemical at its
        current temperature and pressure, in units of [m^2/s].

        .. math::
            \alpha = \frac{k}{\rho Cp}

        Utilizes the temperature and pressure dependent object oriented
        interfaces :obj:`thermo.volume.VolumeLiquid`,
        :obj:`thermo.thermal_conductivity.ThermalConductivityLiquid`,
        and :obj:`thermo.heat_capacity.HeatCapacityLiquid` to calculate the
        actual properties.

        Examples
        --------
        >>> Chemical('nitrogen', T=70).alphal
        9.504101801042264e-08
        '''
        kl, rhol, Cpl = self.kl, self.rhol, self.Cpl
        if all([kl, rhol, Cpl]):
            return thermal_diffusivity(k=kl, rho=rhol, Cp=Cpl)
        return None

    @property
    def alphag(self):
        r'''Thermal diffusivity of the gas phase of the chemical at its
        current temperature and pressure, in units of [m^2/s].

        .. math::
            \alpha = \frac{k}{\rho Cp}

        Utilizes the temperature and pressure dependent object oriented
        interfaces :obj:`thermo.volume.VolumeGas`,
        :obj:`thermo.thermal_conductivity.ThermalConductivityGas`,
        and :obj:`thermo.heat_capacity.HeatCapacityGas` to calculate the
        actual properties.

        Examples
        --------
        >>> Chemical('ammonia').alphag
        1.6931865425158556e-05
        '''
        kg, rhog, Cpg = self.kg, self.rhog, self.Cpg
        if all([kg, rhog, Cpg]):
            return thermal_diffusivity(k=kg, rho=rhog, Cp=Cpg)
        return None

    @property
    def Prl(self):
        r'''Prandtl number of the liquid phase of the chemical at its
        current temperature and pressure, [dimensionless].

        .. math::
            Pr = \frac{C_p \mu}{k}

        Utilizes the temperature and pressure dependent object oriented
        interfaces :obj:`thermo.viscosity.ViscosityLiquid`,
        :obj:`thermo.thermal_conductivity.ThermalConductivityLiquid`,
        and :obj:`thermo.heat_capacity.HeatCapacityLiquid` to calculate the
        actual properties.

        Examples
        --------
        >>> Chemical('nitrogen', T=70).Prl
        2.7655015690791696
        '''
        Cpl, mul, kl = self.Cpl, self.mul, self.kl
        if all([Cpl, mul, kl]):
            return Prandtl(Cp=Cpl, mu=mul, k=kl)
        return None

    @property
    def Prg(self):
        r'''Prandtl number of the gas phase of the chemical at its
        current temperature and pressure, [dimensionless].

        .. math::
            Pr = \frac{C_p \mu}{k}

        Utilizes the temperature and pressure dependent object oriented
        interfaces :obj:`thermo.viscosity.ViscosityGas`,
        :obj:`thermo.thermal_conductivity.ThermalConductivityGas`,
        and :obj:`thermo.heat_capacity.HeatCapacityGas` to calculate the
        actual properties.

        Examples
        --------
        >>> Chemical('NH3').Prg
        0.847263731933008
        '''
        Cpg, mug, kg = self.Cpg, self.mug, self.kg
        if all([Cpg, mug, kg]):
            return Prandtl(Cp=Cpg, mu=mug, k=kg)
        return None

    @property
    def solubility_parameter(self):
        r'''Solubility parameter of the chemical at its
        current temperature and pressure, in units of [Pa^0.5].

        .. math::
            \delta = \sqrt{\frac{\Delta H_{vap} - RT}{V_m}}

        Calculated based on enthalpy of vaporization and molar volume.
        Normally calculated at STP. For uses of this property, see
        :obj:`thermo.solubility.solubility_parameter`.

        Examples
        --------
        >>> Chemical('NH3').solubility_parameter
        24766.329043856073
        '''
        return solubility_parameter(T=self.T, Hvapm=self.Hvapm, Vml=self.Vml,
                                    Method=self.solubility_parameter_method,
                                    CASRN=self.CAS)

    @property
    def Parachor(self):
        r'''Parachor of the chemical at its
        current temperature and pressure, in units of [N^0.25*m^2.75/mol].

        .. math::
            P = \frac{\sigma^{0.25} MW}{\rho_L - \rho_V}

        Calculated based on surface tension, density of the liquid and gas
        phase, and molecular weight. For uses of this property, see
        :obj:`thermo.utils.Parachor`.

        Examples
        --------
        >>> Chemical('octane').Parachor
        6.291693072841486e-05
        '''
        sigma, rhol, rhog = self.sigma, self.rhol, self.rhog
        if all((sigma, rhol, rhog, self.MW)):
            return Parachor(sigma=sigma, MW=self.MW, rhol=rhol, rhog=rhog)
        return None


    ### Single-phase properties
    @property
    def Cp(self):
        r'''Mass heat capacity of the chemical at its current phase and
        temperature, in units of [J/kg/K].

        Utilizes the object oriented interfaces
        :obj:`thermo.heat_capacity.HeatCapacitySolid`,
        :obj:`thermo.heat_capacity.HeatCapacityLiquid`,
        and :obj:`thermo.heat_capacity.HeatCapacityGas` to perform the
        actual calculation of each property. Note that those interfaces provide
        output in molar units (J/mol/K).

        Examples
        --------
        >>> w = Chemical('water')
        >>> w.Cp, w.phase
        (4180.597021827336, 'l')
        >>> Chemical('palladium').Cp
        234.26767209171211
        '''
        return phase_select_property(phase=self.phase, s=self.Cps, l=self.Cpl, g=self.Cpg)

    @property
    def Cpm(self):
        r'''Molar heat capacity of the chemical at its current phase and
        temperature, in units of [J/mol/K].

        Utilizes the object oriented interfaces
        :obj:`thermo.heat_capacity.HeatCapacitySolid`,
        :obj:`thermo.heat_capacity.HeatCapacityLiquid`,
        and :obj:`thermo.heat_capacity.HeatCapacityGas` to perform the
        actual calculation of each property.

        Examples
        --------
        >>> Chemical('cubane').Cpm
        137.05489206785944
        >>> Chemical('ethylbenzene', T=550, P=3E6).Cpm
        294.18449553310046
        '''
        return phase_select_property(phase=self.phase, s=self.Cpsm, l=self.Cplm, g=self.Cpgm)

    @property
    def Vm(self):
        r'''Molar volume of the chemical at its current phase and
        temperature and pressure, in units of [m^3/mol].

        Utilizes the object oriented interfaces
        :obj:`thermo.volume.VolumeSolid`,
        :obj:`thermo.volume.VolumeLiquid`,
        and :obj:`thermo.volume.VolumeGas` to perform the
        actual calculation of each property.

        Examples
        --------
        >>> Chemical('ethylbenzene', T=550, P=3E6).Vm
        0.00017758024401627633
        '''
        return phase_select_property(phase=self.phase, s=self.Vms, l=self.Vml, g=self.Vmg)

    @property
    def rho(self):
        r'''Mass density of the chemical at its current phase and
        temperature and pressure, in units of [kg/m^3].

        Utilizes the object oriented interfaces
        :obj:`thermo.volume.VolumeSolid`,
        :obj:`thermo.volume.VolumeLiquid`,
        and :obj:`thermo.volume.VolumeGas` to perform the
        actual calculation of each property. Note that those interfaces provide
        output in units of m^3/mol.

        Examples
        --------
        >>> Chemical('decane', T=550, P=2E6).rho
        498.6549441720744
        '''
        return phase_select_property(phase=self.phase, s=self.rhos, l=self.rhol, g=self.rhog)

    @property
    def rhom(self):
        r'''Molar density of the chemical at its current phase and
        temperature and pressure, in units of [mol/m^3].

        Utilizes the object oriented interfaces
        :obj:`thermo.volume.VolumeSolid`,
        :obj:`thermo.volume.VolumeLiquid`,
        and :obj:`thermo.volume.VolumeGas` to perform the
        actual calculation of each property. Note that those interfaces provide
        output in units of m^3/mol.

        Examples
        --------
        >>> Chemical('1-hexanol').rhom
        7983.414573003429
        '''
        return phase_select_property(phase=self.phase, s=self.rhosm, l=self.rholm, g=self.rhogm)

    @property
    def Z(self):
        r'''Compressibility factor of the chemical at its current phase and
        temperature and pressure, [dimensionless].

        Examples
        --------
        >>> Chemical('MTBE', T=900, P=1E-2).Z
        0.9999999999079768
        '''
        Vm = self.Vm
        if Vm:
            return Z(self.T, self.P, Vm)
        return None

    @property
    def SG(self):
        r'''Specific gravity of the chemical, [dimensionless]. 
        
        For gas-phase conditions, this is calculated at 15.6 °C (60 °F) and 1 
        atm for the chemical and the reference fluid, air. 
        For liquid and solid phase conditions, this is calculated based on a 
        reference fluid of water at 4°C at 1 atm, but the with the liquid or 
        solid chemical's density at the currently specified conditions.

        Examples
        --------
        >>> Chemical('MTBE').SG
        0.7428160596603596
        '''
        phase = self.phase
        if phase == 'l':
            return self.SGl
        elif phase == 's':
            return self.SGs
        elif phase == 'g':
            return self.SGg
        rho = self.rho
        if rho is not None:
            return SG(rho)
        return None
    
    @property
    def isobaric_expansion(self):
        r'''Isobaric (constant-pressure) expansion of the chemical at its
        current phase and temperature, in units of [1/K].

        .. math::
            \beta = \frac{1}{V}\left(\frac{\partial V}{\partial T} \right)_P

        Examples
        --------
        Radical change  in value just above and below the critical temperature
        of water:

        >>> Chemical('water', T=647.1, P=22048320.0).isobaric_expansion
        0.34074205839222449

        >>> Chemical('water', T=647.2, P=22048320.0).isobaric_expansion
        0.18143324022215077
        '''
        return phase_select_property(phase=self.phase, l=self.isobaric_expansion_l, g=self.isobaric_expansion_g)

    @property
    def JT(self):
        r'''Joule Thomson coefficient of the chemical at its
        current phase and temperature, in units of [K/Pa].

        .. math::
            \mu_{JT} = \left(\frac{\partial T}{\partial P}\right)_H = \frac{1}{C_p}
            \left[T \left(\frac{\partial V}{\partial T}\right)_P - V\right]
            = \frac{V}{C_p}\left(\beta T-1\right)

        Examples
        --------
        >>> Chemical('water').JT
        -2.2150394958666407e-07
        '''
        return phase_select_property(phase=self.phase, l=self.JTl, g=self.JTg)

    @property
    def mu(self):
        r'''Viscosity of the chemical at its current phase, temperature, and
        pressure in units of [Pa*s].

        Utilizes the object oriented interfaces
        :obj:`thermo.viscosity.ViscosityLiquid` and
        :obj:`thermo.viscosity.ViscosityGas` to perform the
        actual calculation of each property.

        Examples
        --------
        >>> Chemical('ethanol', T=300).mu
        0.001044526538460911
        >>> Chemical('ethanol', T=400).mu
        1.1853097849748217e-05
        '''
        return phase_select_property(phase=self.phase, l=self.mul, g=self.mug)

    @property
    def k(self):
        r'''Thermal conductivity of the chemical at its current phase,
        temperature, and pressure in units of [W/m/K].

        Utilizes the object oriented interfaces
        :obj:`thermo.thermal_conductivity.ThermalConductivityLiquid` and
        :obj:`thermo.thermal_conductivity.ThermalConductivityGas` to perform
        the actual calculation of each property.

        Examples
        --------
        >>> Chemical('ethanol', T=300).kl
        0.16313594741877802
        >>> Chemical('ethanol', T=400).kg
        0.026019924109310026
        '''
        return phase_select_property(phase=self.phase, s=None, l=self.kl, g=self.kg)

    @property
    def nu(self):
        r'''Kinematic viscosity of the the chemical at its current temperature,
        pressure, and phase in units of [m^2/s].

        .. math::
            \nu = \frac{\mu}{\rho}

        Examples
        --------
        >>> Chemical('argon').nu
        1.3846930410865003e-05
        '''
        return phase_select_property(phase=self.phase, l=self.nul, g=self.nug)

    @property
    def alpha(self):
        r'''Thermal diffusivity of the chemical at its current temperature,
        pressure, and phase in units of [m^2/s].

        .. math::
            \alpha = \frac{k}{\rho Cp}

        Examples
        --------
        >>> Chemical('furfural').alpha
        7.672866198927953e-08
        '''
        return phase_select_property(phase=self.phase, l=self.alphal, g=self.alphag)

    @property
    def Pr(self):
        r'''Prandtl number of the chemical at its current temperature,
        pressure, and phase; [dimensionless].

        .. math::
            Pr = \frac{C_p \mu}{k}

        Examples
        --------
        >>> Chemical('acetone').Pr
        4.450368847076066
        '''
        return phase_select_property(phase=self.phase, l=self.Prl, g=self.Prg)

    @property
    def Poynting(self):
        r'''Poynting correction factor [dimensionless] for use in phase 
        equilibria methods based on activity coefficients or other reference 
        states. Performs the shortcut calculation assuming molar volume is 
        independent of pressure.

        .. math::
            \text{Poy} =  \exp\left[\frac{V_l (P-P^{sat})}{RT}\right]

        The full calculation normally returns values very close to the
        approximate ones. This property is defined in terms of
        pure components only.

        Examples
        --------
        >>> Chemical('pentane', T=300, P=1E7).Poynting
        1.5743051250679803

        Notes
        -----
        The full equation shown below can be used as follows:

        .. math::
            \text{Poy} = \exp\left[\frac{\int_{P_i^{sat}}^P V_i^l dP}{RT}\right]

        >>> from scipy.integrate import quad
        >>> c = Chemical('pentane', T=300, P=1E7)
        >>> exp(quad(lambda P : c.VolumeLiquid(c.T, P), c.Psat, c.P)[0]/R/c.T)
        1.5821826990975127
        '''
        Vml, Psat = self.Vml, self.Psat
        if Vml and Psat:
            return exp(Vml*(self.P-Psat)/R/self.T)
        return None

    def Tsat(self, P):
        return self.VaporPressure.solve_prop(P)

    ### Convenience Dimensionless numbers
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

