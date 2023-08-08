'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
'''


__all__ = ['Mixture']

from collections import OrderedDict

from chemicals.elements import mass_fractions, mixture_atomic_composition
from chemicals.identifiers import CAS_from_any, mixture_from_any
from chemicals.utils import (
    SG,
    Joule_Thomson,
    Parachor,
    R,
    SG_to_API,
    Vfs_to_zs,
    Vm_to_rho,
    Z,
    isentropic_exponent,
    isobaric_expansion,
    mixing_simple,
    none_and_length_check,
    property_mass_to_molar,
    property_molar_to_mass,
    speed_of_sound,
    vapor_mass_quality,
    ws_to_zs,
    zs_to_Vfs,
    zs_to_ws,
)
from chemicals.virial import B_from_Z
from chemicals.volume import ideal_gas
from fluids.core import Bond, Capillary, Grashof, Jakob, Peclet_heat, Prandtl, Reynolds, Weber, nu_mu_converter, thermal_diffusivity
from fluids.numerics import numpy as np

from thermo.chemical import Chemical
from thermo.eos import IG, PR
from thermo.eos_mix import PRMIX
from thermo.heat_capacity import HeatCapacityGasMixture, HeatCapacityLiquidMixture, HeatCapacitySolidMixture
from thermo.interface import SurfaceTensionMixture
from thermo.thermal_conductivity import ThermalConductivityGasMixture, ThermalConductivityLiquidMixture
from thermo.utils import phase_select_property
from thermo.viscosity import ViscosityGasMixture, ViscosityLiquidMixture
from thermo.volume import LINEAR_MISSING_IDEAL, VolumeGasMixture, VolumeLiquidMixture, VolumeSolidMixture


def preprocess_mixture_composition(IDs=None, zs=None, ws=None, Vfls=None,
                                   Vfgs=None, ignore_exceptions=False):
    r'''Composition preprocessing function for the :obj:`thermo.mixture.Mixture`
    class, as it had grown to the size it required its own function.

    This function accepts the possible ways of specifying composition, parses
    and checks them to an extent, and returns the same arguments it receives.

    The tasks it performs are as follows:

        * Check if the input ID was a string, or a 1-length list, which is one
          of the main keys or synonyms retrievable from
          :obj:`thermo.identifiers.mixture_from_any`; if it is, take the
          composition from that method (weight fractions will be returned).
        * If the ID is a string or a 1-length list, set the composition to
          be pure (if no other composition was specified).
        * If the composition (zs, ws, Vfls, Vfgs) is a list, turn it into a
          copy of the list to not change other instances of it.
        * If the composition is a numpy array, convert it to a list for greater
          speed.
        * If the composition is a dict or OrderedDict, take the keys of it
          as the identifiers from its keys and the composition as its values.

    If no composition has been specified after the above parsing, an exception
    is raised.

    If multiple ways of specifying composition were used, raise an exception.

    If the length of the specified composition is not the same as the number
    of identifiers given, an exception is raised.

    Note this method does not normalize composition to one; or check the
    identifiers are valid.
    '''
    # Test if the input ID a string or a list
    if hasattr(IDs, 'strip') or (isinstance(IDs, list) and len(IDs) == 1):
        try:
            # Assume the name was a pre-defined mixture
            mix = mixture_from_any(IDs)
            IDs = mix.CASs#d["CASs"]
            ws = mix.ws#_d["ws"]
        except:
            if hasattr(IDs, 'strip'):
                IDs = [IDs]

                zs = [1.0]
            elif isinstance(IDs, list) and len(IDs) == 1:
                if zs is None and ws is None and Vfls is None and Vfgs is None:
                    zs = [1.0]
            else:
                if not ignore_exceptions:
                    raise Exception('Could not recognize the mixture IDs')
                else:
                    return IDs, zs, ws, Vfls, Vfgs

    # Handle numpy array inputs; also turn mutable inputs into copies
    if zs is not None:
        t = type(zs)
        if t == list:
            zs = list(zs)
        elif t == np.ndarray:
            zs = zs.tolist()
        elif isinstance(zs, (OrderedDict, dict)):
            IDs = list(zs.keys())
            zs = list(zs.values())
        length_matching = len(zs) == len(IDs)
    elif ws is not None:
        t = type(ws)
        if t == list:
            ws = list(ws)
        elif t == np.ndarray:
            ws = ws.tolist()
        elif isinstance(ws, (OrderedDict, dict)):
            IDs = list(ws.keys())
            ws = list(ws.values())
        length_matching = len(ws) == len(IDs)
    elif Vfls is not None:
        t = type(Vfls)
        if t == list:
            Vfls = list(Vfls)
        elif t == np.ndarray:
            Vfls = Vfls.tolist()
        elif isinstance(Vfls, (OrderedDict, dict)):
            IDs = list(Vfls.keys())
            Vfls = list(Vfls.values())
        length_matching = len(Vfls) == len(IDs)
    elif Vfgs is not None:
        t = type(Vfgs)
        if t == list:
            Vfgs = list(Vfgs)
        elif t == np.ndarray:
            Vfgs = Vfgs.tolist()
        elif isinstance(Vfgs, (OrderedDict, dict)):
            IDs = list(Vfgs.keys())
            Vfgs = list(Vfgs.values())
        length_matching = len(Vfgs) == len(IDs)
    else:
        if not ignore_exceptions:
            raise Exception("One of 'zs', 'ws', 'Vfls', or 'Vfgs' is required to define the mixture")
    # Do not to a test on multiple composition inputs in case the user specified
    # a composition, plus one was set (it will be zero anyway)
    if not ignore_exceptions:
        if len(IDs) > 1 and ((zs is not None) + (ws is not None) + (Vfgs is not None) + (Vfls is not None)) > 1:
            raise Exception('Multiple different composition arguments were '
                            "specified; specify only one of the arguments "
                            "'zs', 'ws', 'Vfls', or 'Vfgs'.")
        if not length_matching:
            raise Exception('Composition is not the same length as the component identifiers')
    return IDs, zs, ws, Vfls, Vfgs


class Mixture:
    '''Creates a Mixture object which contains basic information such as
    molecular weight and the structure of the species, as well as thermodynamic
    and transport properties as a function of two of the variables temperature,
    pressure, vapor fraction, enthalpy, or entropy.

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

    If no thermodynamic conditions are specified, or if only one of T and P
    are specifed without another thermodynamic variable as well, the T and P
    298.15 K and/or 101325 Pa will be set instead of the missing variables.

    Parameters
    ----------
    IDs : list, optional
        List of chemical identifiers - names, CAS numbers, SMILES or InChi
        strings can all be recognized and may be mixed [-]
    zs : list or dict, optional
        Mole fractions of all components in the mixture [-]
    ws : list or dict, optional
        Mass fractions of all components in the mixture [-]
    Vfls : list or dict, optional
        Volume fractions of all components as a hypothetical liquid phase based
        on pure component densities [-]
    Vfgs : list, or dict optional
        Volume fractions of all components as a hypothetical gas phase based
        on pure component densities [-]
    T : float, optional
        Temperature of the mixture (default 298.15 K), [K]
    P : float, optional
        Pressure of the mixture (default 101325 Pa) [Pa]
    VF : float, optional
        Vapor fraction (mole basis) of the mixture, [-]
    Hm : float, optional
        Molar enthalpy of the mixture, [J/mol]
    H : float, optional
        Mass enthalpy of the mixture, [J/kg]
    Sm : float, optional
        Molar entropy of the mixture, [J/mol/K]
    S : float, optional
        Mass entropy of the mixture, [J/kg/K]
    pkg : object
        The thermodynamic property package to use for flash calculations;
        one of the caloric packages in :obj:`thermo.property_package`;
        defaults to the ideal model [-]
    Vf_TP : tuple(2, float), optional
        The (T, P) at which the volume fractions are specified to be at, [K]
        and [Pa]

    Attributes
    ----------
    MW : float
        Mole-weighted average molecular weight all chemicals in the mixture,
        [g/mol]
    IDs : list of str
        Names of all the species in the mixture as given in the input, [-]
    names : list of str
        Names of all the species in the mixture, [-]
    CASs : list of str
        CAS numbers of all species in the mixture, [-]
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
        see :obj:`chemicals.acentric.Stiel_polar_factor` for the definition, [-]
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
    Hfms : list of float
        Molar enthalpy of formations of all chemicals in the mixture, [J/mol]
    Hfs : list of float
        Enthalpy of formations of all chemicals in the mixture, [J/kg]
    Gfms : list of float
        Molar Gibbs free energies of formation of all chemicals in the mixture,
        [J/mol]
    Gfs : list of float
        Gibbs free energies of formation of all chemicals in the mixture,
        [J/kg]
    Sfms : list of float
        Molar entropy of formation of all chemicals in the mixture,
        [J/mol/K]
    Sfs : list of float
        Entropy of formation of all chemicals in the mixture,
        [J/kg/K]
    S0ms : list of float
        Standard absolute entropies of all chemicals in the mixture,
        [J/mol/K]
    S0s : list of float
        Standard absolute entropies of all chemicals in the mixture,
        [J/kg/K]
    Hcms : list of float
        Molar higher heats of combustions of all chemicals in the mixture,
        [J/mol]
    Hcs : list of float
        Higher heats of combustions of all chemicals in the mixture,
        [J/kg]
    Hcms_lower : list of float
        Molar lower heats of combustions of all chemicals in the mixture,
        [J/mol]
    Hcs_lower : list of float
        Higher lower of combustions of all chemicals in the mixture,
        [J/kg]
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
    Chemicals : list of Chemical instances
        Chemical instances used in calculating mixture properties, [-]
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
    rhoml_STPs : list of float
        Molar densities of the chemicals in the mixture as liquids at 298.15 K
        and 101325 Pa, [mol/m^3]
    Vmg_STPs : list of float
        Molar volume of the chemicals in the mixture as gases at 298.15 K and
        101325 Pa, [m^3/mol]
    Vms_Tms : list of float
        Molar volumes of solid phase at the melting point [m^3/mol]
    rhos_Tms : list of float
        Mass densities of solid phase at the melting point [kg/m^3]
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
    A
    Am
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
    isobaric_expansion_g
    isobaric_expansion_gs
    isobaric_expansion_l
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
    U
    Um
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

    Notes
    -----
    .. warning::
        The Mixture class is not designed for high-performance or the ability
        to use different thermodynamic models. It is especially limited in its
        multiphase support and the ability to solve with specifications other
        than temperature and pressure. It is impossible to change constant
        properties such as a compound's critical temperature in this interface.

        It is recommended to switch over to the :obj:`thermo.flash` interface
        which solves those problems and is better positioned to grow. That
        interface also requires users to be responsible for their chemical
        constants and pure component correlations; while default values can
        easily be loaded for most compounds, the user is ultimately responsible
        for them.

    Examples
    --------
    Creating Mixture objects:

    >>> Mixture(['water', 'ethanol'], Vfls=[.6, .4], T=300, P=1E5)
    <Mixture, components=['water', 'ethanol'], mole fractions=[0.8299, 0.1701], T=300.00 K, P=100000 Pa>

    For mixtures with large numbers of components, it may be confusing to enter
    the composition separate from the names of the chemicals. For that case,
    the syntax using dictionaries as follows is supported with any composition
    specification:

    >>> comp = OrderedDict([('methane', 0.96522),
    ...                     ('nitrogen', 0.00259),
    ...                     ('carbon dioxide', 0.00596),
    ...                     ('ethane', 0.01819),
    ...                     ('propane', 0.0046),
    ...                     ('isobutane', 0.00098),
    ...                     ('butane', 0.00101),
    ...                     ('2-methylbutane', 0.00047),
    ...                     ('pentane', 0.00032),
    ...                     ('hexane', 0.00066)])
    >>> m = Mixture(zs=comp)
    '''

    flashed = True
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
    T_default = 298.15
    P_default = 101325.
    autoflash = True # Whether or not to flash on init

    def __repr__(self):
        txt = f'<Mixture, components={self.names}, mole fractions={[round(i,4) for i in self.zs]}'
        # T and P may not be available if a flash has failed
        try:
            txt += f', T={self.T:.2f} K, P={self.P:.0f} Pa>'
        except:
            txt += ', thermodynamic conditions unknown>'
        return txt

    def __init__(self, IDs=None, zs=None, ws=None, Vfls=None, Vfgs=None,
                 T=None, P=None,
                 VF=None, H=None, Hm=None, S=None, Sm=None, pkg=None, Vf_TP=(None, None)):
        # Perofrm preprocessing of the mixture composition separately so it
        # can be tested on its own
        IDs, zs, ws, Vfls, Vfgs = preprocess_mixture_composition(IDs=IDs,
                                                                 zs=zs, ws=ws,
                                                                 Vfls=Vfls,
                                                                 Vfgs=Vfgs)
        self.IDs = IDs
        self.N = len(IDs)
        self.cmps = range(self.N)

        T_unsolved = T if T is not None else self.T_default
        P_unsolved = P if P is not None else self.P_default
        self.Chemicals = [Chemical(ID, P=P_unsolved, T=T_unsolved, autocalc=False) for ID in self.IDs]

        # Required for densities for volume fractions before setting fractions
        self.set_chemical_constants()
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
                T_vf = T_unsolved
            if P_vf is None:
                P_vf = P_unsolved

            if Vfls:
                Vfs = Vfls if sum(Vfls) == 1 else [Vfli/sum(Vfls) for Vfli in Vfls]
                VolumeObjects = self.VolumeLiquids
                Vms_TP = self.Vmls
            else:
                Vfs = Vfgs if sum(Vfgs) == 1 else [Vfgi/sum(Vfgs) for Vfgi in Vfgs]
                VolumeObjects = self.VolumeGases
                #Vms_TP = self.Vmgs
                Vms_TP = [ideal_gas(T_vf, P_vf)]*self.N

            if (T_vf != T or P_vf != P) and Vfls:
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


        # To preserve backwards compatibility, mixures with no other state vars
        # specified will have their T and P initialized to the values of
        # T_default and P_default (but only if the values VF, Hm, H, Sm, S are
        # None)
        non_TP_state_vars = sum(i is not None for i in [VF, Hm, H, Sm, S])
        if non_TP_state_vars == 0:
            if T is None:
                T = self.T_default
            if P is None:
                P = self.P_default

        self.set_property_package(pkg=pkg)
        if self.autoflash:
            self.flash_caloric(T=T, P=P, VF=VF, Hm=Hm, Sm=Sm, H=H, S=S)


    def set_chemical_constants(self):
        r'''Basic method which retrieves and sets constants of chemicals to be
        accessible as lists from a Mixture object. This gets called
        automatically on the instantiation of a new Mixture instance.
        '''
        self.names = [i.name for i in self.Chemicals]
        self.MWs = MWs = [i.MW for i in self.Chemicals]
        self.CASs = [i.CAS for i in self.Chemicals]

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

        # Chemistry - standard state
        self.Hfms = [i.Hfm for i in self.Chemicals]
        self.Hfs = [i.Hf for i in self.Chemicals]

        self.S0ms = [i.S0m for i in self.Chemicals]
        self.S0s = [i.S0 for i in self.Chemicals]

        self.Gfms = [i.Gfm for i in self.Chemicals]
        self.Gfs = [i.Gf for i in self.Chemicals]

        self.Sfms = [i.Sfm for i in self.Chemicals]
        self.Sfs = [i.Sf for i in self.Chemicals]

        # Ideal gas state
        self.Hfgms = [i.Hfgm for i in self.Chemicals]
        self.Hfgs = [i.Hfg for i in self.Chemicals]

        self.S0gms = [i.S0gm for i in self.Chemicals]
        self.S0gs = [i.S0g for i in self.Chemicals]

        self.Gfgms = [i.Gfgm for i in self.Chemicals]
        self.Gfgs = [i.Gfg for i in self.Chemicals]

        self.Sfgms = [i.Sfgm for i in self.Chemicals]
        self.Sfgs = [i.Sfg for i in self.Chemicals]

        # Combustion
        self.Hcms = [i.Hcm for i in self.Chemicals]
        self.Hcs = [i.Hc for i in self.Chemicals]

        self.Hcms_lower = [i.Hcm_lower for i in self.Chemicals]
        self.Hcs_lower = [i.Hc_lower for i in self.Chemicals]

        self.Hcgms = [i.Hcgm for i in self.Chemicals]
        self.Hcgs = [i.Hcg for i in self.Chemicals]

        self.Hcgms_lower = [i.Hcgm_lower for i in self.Chemicals]
        self.Hcgs_lower = [i.Hcg_lower for i in self.Chemicals]

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
        self.RI_Ts = [i.RIT for i in self.Chemicals]
        self.RIs = [i.RI for i in self.Chemicals]
        self.conductivities = [i.conductivity for i in self.Chemicals]
        self.conductivity_Ts = [i.conductivityT for i in self.Chemicals]

        # Constant properties obtained from TP
        self.Vml_STPs = Vml_STPs = [i.Vml_STP for i in self.Chemicals]
        self.rholm_STPs = [i.rhoml_STP for i in self.Chemicals]
        self.rhol_STPs = [i.rhol_STP for i in self.Chemicals]

        self.Vml_60Fs = Vml_STPs = [i.Vml_60F for i in self.Chemicals]
        self.rhoml_60Fs = [i.rhoml_60F for i in self.Chemicals]
        self.rhol_60Fs = [i.rhol_60F for i in self.Chemicals]

        self.Vmg_STPs = [i.Vmg_STP for i in self.Chemicals]

        self.Vms_Tms = [i.Vms_Tm for i in self.Chemicals]
        self.rhoms_Tm = [i.rhoms_Tm for i in self.Chemicals]
        self.rhos_Tms = [i.rhos_Tm for i in self.Chemicals]

        self.Psat_298s = [i.Psat_298 for i in self.Chemicals]
        self.phase_STPs = [i.phase_STP for i in self.Chemicals]
        self.Vml_Tbs = [i.Vml_Tb for i in self.Chemicals]
        self.Vml_Tms = [i.Vml_Tm for i in self.Chemicals]
        self.Hvap_Tbms = [i.Hvap_Tbm for i in self.Chemicals]
        self.Hvap_Tbs = [i.Hvap_Tb for i in self.Chemicals]
        self.Hvapm_298s = [i.Hvapm_298 for i in self.Chemicals]
        self.Hvap_298s = [i.Hvap_298 for i in self.Chemicals]
        self.solubility_parameters_STP = [i.solubility_parameter_STP for i in self.Chemicals]

    ### More stuff here

    def set_chemical_TP(self, T=None, P=None):
        '''Basic method to change all chemical instances to be at the T and P
        specified. If they are not specified, the the values of the mixture
        will be used. This is not necessary for using the Mixture instance
        unless values specified to chemicals are required.
        '''
        # Tempearture and Pressure Denepdence
        # Get and choose initial methods
        if T is None:
            T = self.T
        if P is None:
            P = self.P
        [i.calculate(T=T, P=P) for i in self.Chemicals]

    def set_constant_sources(self):
        # None of this takes much time or is important
        # Critical Point, Methods only for Tc, Pc, Vc
        self.Tc_methods = []#Tc_mixture(Tcs=self.Tcs, zs=self.zs, CASRNs=self.CASs, get_methods=True)
        self.Tc_method = None#self.Tc_methods[0]
        self.Pc_methods = []#Pc_mixture(Pcs=self.Pcs, zs=self.zs, CASRNs=self.CASs, get_methods=True)
        self.Pc_method = None#self.Pc_methods[0]
        self.Vc_methods = []#Vc_mixture(Vcs=self.Vcs, zs=self.zs, CASRNs=self.CASs, get_methods=True)
        self.Vc_method = None#self.Vc_methods[0]
        self.omega_methods = []#omega_mixture(omegas=self.omegas, zs=self.zs, CASRNs=self.CASs, get_methods=True)
        self.omega_method = None#self.omega_methods[0]

        # No Flammability limits
#        self.LFL_methods = LFL_mixture(ys=self.zs, LFLs=self.LFLs, get_methods=True)
#        self.LFL_method = self.LFL_methods[0]
#        self.UFL_methods = UFL_mixture(ys=self.zs, UFLs=self.UFLs, get_methods=True)
#        self.UFL_method = self.UFL_methods[0]
        # No triple point
        # Mixed Hf linear
        # Exposure limits are minimum of any of them or lower

    def set_constants(self):
        # None of this takes much time or is important
        # Melting point
        zs = self.zs
        self.Tm = mixing_simple(self.Tms, zs)
        # Critical Point
        try:
            self.Tc = mixing_simple(zs, self.Tcs)
        except:
            self.Tc = None
        try:
            self.Pc = mixing_simple(zs, self.Pcs)
        except:
            self.Pc = None
        try:
            self.Vc = mixing_simple(zs, self.Vcs)
        except:
            self.Vc = None
        try:
            self.omega = mixing_simple(zs, self.omegas)
        except:
            self.omega = None

        self.Zc = Z(self.Tc, self.Pc, self.Vc) if all((self.Tc, self.Pc, self.Vc)) else None
        self.rhoc = Vm_to_rho(self.Vc, self.MW) if self.Vc else None
        self.rhocm = 1./self.Vc if self.Vc else None

#        self.LFL = LFL_mixture(ys=self.zs, LFLs=self.LFLs, method=self.LFL_method)
#        self.UFL = UFL_mixture(ys=self.zs, UFLs=self.UFLs, method=self.UFL_method)

    def set_eos(self, T, P, eos=PRMIX):
        try:
            self.eos = eos(T=T, P=P, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, zs=self.zs)
        except:
            # Handle overflow errors and so on
            self.eos = IG(T=T, P=P)

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

    def eos_pures(self, eos=PR, T=None, P=None):
        if T is None:
            T = self.T
        if P is None:
            P = self.P
        Tcs, Pcs, omegas = self.Tcs, self.Pcs, self.omegas
        eos_list = []
        for i in range(len(self.zs)):
            try:
                e = eos(T=T, P=P, Tc=Tcs[i], Pc=Pcs[i], omega=omegas[i])
            except:
                e = None
            eos_list.append(e)
        return eos_list

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
        self.ThermalConductivitySolids = [i.ThermalConductivitySolid for i in self.Chemicals]
        self.SurfaceTensions = [i.SurfaceTension for i in self.Chemicals]
        self.Permittivities = [i.Permittivity for i in self.Chemicals]

        self.VaporPressures = [i.VaporPressure for i in self.Chemicals]
        self.SublimationPressures = [i.SublimationPressure for i in self.Chemicals]
        self.EnthalpyVaporizations = [i.EnthalpyVaporization for i in self.Chemicals]
        self.EnthalpySublimations = [i.EnthalpySublimation for i in self.Chemicals]

    def set_TP_sources(self):

        self.VolumeSolidMixture = VolumeSolidMixture(CASs=self.CASs, MWs=self.MWs, VolumeSolids=self.VolumeSolids)
        self.VolumeLiquidMixture = VolumeLiquidMixture(MWs=self.MWs, Tcs=self.Tcs, Pcs=self.Pcs, Vcs=self.Vcs, Zcs=self.Zcs, omegas=self.omegas, CASs=self.CASs, VolumeLiquids=self.VolumeLiquids)
        self.VolumeGasMixture = VolumeGasMixture(eos=self.eos_in_a_box, MWs=self.MWs, CASs=self.CASs, VolumeGases=self.VolumeGases)

        # Temporary
        self.VolumeGasMixture.method = LINEAR_MISSING_IDEAL

        self.HeatCapacityLiquidMixture = HeatCapacityLiquidMixture(MWs=self.MWs, CASs=self.CASs, HeatCapacityLiquids=self.HeatCapacityLiquids)
        self.HeatCapacityGasMixture = HeatCapacityGasMixture(MWs=self.MWs, CASs=self.CASs, HeatCapacityGases=self.HeatCapacityGases)
        self.HeatCapacitySolidMixture = HeatCapacitySolidMixture(MWs=self.MWs, CASs=self.CASs, HeatCapacitySolids=self.HeatCapacitySolids)

        self.ViscosityLiquidMixture = ViscosityLiquidMixture(MWs=self.MWs, CASs=self.CASs, ViscosityLiquids=self.ViscosityLiquids, correct_pressure_pure=False)
        self.ViscosityGasMixture = ViscosityGasMixture(MWs=self.MWs, molecular_diameters=self.molecular_diameters, Stockmayers=self.Stockmayers, CASs=self.CASs, ViscosityGases=self.ViscosityGases, correct_pressure_pure=False)

        self.ThermalConductivityLiquidMixture = ThermalConductivityLiquidMixture(CASs=self.CASs, MWs=self.MWs, ThermalConductivityLiquids=self.ThermalConductivityLiquids, correct_pressure_pure=False)
        self.ThermalConductivityGasMixture = ThermalConductivityGasMixture(MWs=self.MWs, Tbs=self.Tbs, CASs=self.CASs, ThermalConductivityGases=self.ThermalConductivityGases, ViscosityGases=self.ViscosityGases, correct_pressure_pure=False)

        self.SurfaceTensionMixture = SurfaceTensionMixture(MWs=self.MWs, Tbs=self.Tbs, Tcs=self.Tcs, CASs=self.CASs, SurfaceTensions=self.SurfaceTensions, VolumeLiquids=self.VolumeLiquids)


    def set_property_package(self, pkg=None):
        if pkg is None:
            from thermo.property_package import IdealCaloric as pkg

        eos_mix = type(self.eos_in_a_box[0]) if self.eos_in_a_box else PRMIX

        if type(pkg) == type:
            self.property_package = pkg(VaporPressures=self.VaporPressures,
                                         Tms=self.Tms, Tbs=self.Tbs,
                                         Tcs=self.Tcs, Pcs=self.Pcs,
                                         HeatCapacityLiquids=self.HeatCapacityLiquids,
                                         HeatCapacityGases=self.HeatCapacityGases,
                                         EnthalpyVaporizations=self.EnthalpyVaporizations,
                                         UNIFAC_groups=self.UNIFAC_groups, omegas=self.omegas,
                                         Hfs=self.Hfgms, Gfs=self.Gfgms,
                                         VolumeLiquids=self.VolumeLiquids, eos=type(self.Chemicals[0].eos),
                                         eos_mix=eos_mix)
        else:
            # no need to initialize, already exists
            self.property_package = pkg


    def flash_caloric(self, T=None, P=None, VF=None, Hm=None, Sm=None,
                      H=None, S=None):
        # TODO check if the input values are the same as the current ones
        # The property package works only on a mole-basis, so convert
        # H or S if specified to a mole basis
        if H is not None:
            Hm = property_mass_to_molar(H, self.MW)
        if S is not None:
            Sm = property_mass_to_molar(S, self.MW)
        self.property_package.flash_caloric(zs=self.zs, T=T, P=P, VF=VF, Hm=Hm, Sm=Sm)
        self.status = self.property_package.status
        if self.status is True:
            self.T = self.property_package.T
            self.P = self.property_package.P
            self.V_over_F = self.VF = self.property_package.V_over_F
            self.xs = self.property_package.xs
            self.ys = self.property_package.ys
            self.phase = self.property_package.phase

            self.Hm = self.property_package.Hm
            self.Sm = self.property_package.Sm
            self.Gm = self.property_package.Gm

            try:
                self.Hm_reactive = self.property_package.Hm_reactive
                self.H_reactive = property_molar_to_mass(self.Hm_reactive, self.MW)
            except:
                self.Hm_reactive = self.H_reactive = None
            try:
                self.Sm_reactive = self.property_package.Sm_reactive
                self.S_reactive = property_molar_to_mass(self.Sm_reactive, self.MW)
            except:
                self.Sm_reactive = self.S_reactive = None
            try:
                self.Gm_reactive = self.property_package.Gm_reactive
                self.G_reactive = property_molar_to_mass(self.Gm_reactive, self.MW)
            except:
                self.Gm_reactive = self.G_reactive = None

            self.H = property_molar_to_mass(self.Hm, self.MW)
            self.S = property_molar_to_mass(self.Sm, self.MW)
            self.G = property_molar_to_mass(self.Gm, self.MW)


            # values are None when not in the appropriate phase
            self.MWl = mixing_simple(self.xs, self.MWs) if self.xs is not None else None
            self.MWg = mixing_simple(self.ys, self.MWs) if self.ys is not None else None
            self.wsl = zs_to_ws(self.xs, self.MWs) if self.xs is not None else None
            self.wsg = zs_to_ws(self.ys, self.MWs) if self.ys is not None else None

            if (self.MWl is not None and self.MWg is not None):
                self.quality = self.x = vapor_mass_quality(self.V_over_F, MWl=self.MWl, MWg=self.MWg)
            else:
                self.quality = self.x = 1 if self.phase == 'g' else 0


            if self.xs is None:
                self.wsl = zs_to_ws(self.ys, self.MWs)
                self.MWl = mixing_simple(self.ys, self.MWs)

            if self.ys is None:
                self.MWg = mixing_simple(self.xs, self.MWs)
                self.wsg = zs_to_ws(self.xs, self.MWs)

            # TODO: volume fractions - attempt
#            if (self.rhol is not None and self.rhog is not None):
#                self.Vfg = vapor_mass_quality(self.quality, MWl=self.Vml, MWg=self.Vmg)
#            else:
#                self.Vfg = None

        else:
            # flash failed. still want to set what variables that can be set though.
            for var in ['T', 'P', 'VF', 'Hm', 'Sm', 'H', 'S']:
                if var is not None:
                    setattr(self, var, locals()[var])

        # Not strictly necessary
        [i.calculate(self.T, self.P) for i in self.Chemicals]
#        self.set_eos(T=self.T, P=self.P)

    @property
    def Um(self):
        r'''Internal energy of the mixture at its current state, in units of
        [J/mol].

        This property requires that the property package of the mixture
        found a solution to the given state variables.
        It also depends on the molar volume of the mixture at its current
        conditions.
        '''
        return self.Hm - self.P*self.Vm if (self.Vm and self.Hm is not None) else None

    @property
    def U(self):
        r'''Internal energy of the mixture at its current state,
        in units of [J/kg].

        This property requires that the property package of the mixture
        found a solution to the given state variables.
        It also depends on the molar volume of the mixture at its current
        conditions.
        '''
        return property_molar_to_mass(self.Um, self.MW) if (self.Um is not None) else None

    @property
    def Am(self):
        r'''Helmholtz energy of the mixture at its current state,
        in units of [J/mol].

        This property requires that the property package of the mixture
        found a solution to the given state variables.
        It also depends on the molar volume of the mixture at its current
        conditions.
        '''
        return self.Um - self.T*self.Sm if (self.Um is not None and self.Sm is not None) else None

    @property
    def A(self):
        r'''Helmholtz energy of the mixture at its current state,
        in units of [J/kg].

        This property requires that the property package of the mixture
        found a solution to the given state variables.
        It also depends on the molar volume of the mixture at its current
        conditions.
        '''
        return self.U - self.T*self.S if (self.U is not None and self.S is not None) else None

    @property
    def Tdew(self):
        r'''Dew point temperature of the mixture at its current pressure and
        composition, in units of [K].

        This property requires that the property package of the mixture
        found a solution to the given state variables.
        '''
        return self.property_package.Tdew(P=self.P, zs=self.zs)

    @property
    def Pdew(self):
        r'''Dew point pressure of the mixture at its current temperature and
        composition, in units of [Pa].

        This property requires that the property package of the mixture
        found a solution to the given state variables.
        '''
        return self.property_package.Pdew(T=self.T, zs=self.zs)

    @property
    def Tbubble(self):
        r'''Bubble point temperature of the mixture at its current pressure and
        composition, in units of [K].

        This property requires that the property package of the mixture
        found a solution to the given state variables.
        '''
        return self.property_package.Tbubble(P=self.P, zs=self.zs)

    @property
    def Pbubble(self):
        r'''Bubble point pressure of the mixture at its current temperature and
        composition, in units of [Pa].

        This property requires that the property package of the mixture
        found a solution to the given state variables.
        '''
        return self.property_package.Pbubble(T=self.T, zs=self.zs)


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
        return self.zs
#        if (T is None or T == self.T) and (P is None or P == self.P):
#            Vmgs = self.Vmgs
#        else:
#            if T is None: T = self.T
#            if P is None: P = self.P
#            Vmgs = [i(T, P) for i in self.VolumeGases]
#        if none_and_length_check([Vmgs]):
#            return zs_to_Vfs(self.zs, Vmgs)
#        return None
#
    def compound_index(self, CAS):
        try:
            return self.CASs.index(CAS)
        except ValueError:
            return self.CASs.index(CAS_from_any(CAS))

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
        :obj:`chemicals.elements.similarity_variable` for the definition, [mol/g]

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5]).similarity_variables
        [0.15362587797189262, 0.16279853724428964]
        '''
        return [i.similarity_variable for i in self.Chemicals]

    @property
    def atoms(self):
        r'''Mole-averaged dictionary of atom counts for all atoms of the
        chemicals in the mixture.

        Examples
        --------
        >>> Mixture(['nitrogen', 'oxygen'], zs=[.01, .99]).atoms
        {'O': 1.98, 'N': 0.02}
        '''
        return mixture_atomic_composition(self.atomss, self.zs)

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
            for atom, count in atoms.items():
                if atom in things:
                    things[atom] += zi*count
                else:
                    things[atom] = zi*count

        tot = sum(things.values())
        return {atom : value/tot for atom, value in things.items()}

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
            for atom, count in atoms.items():
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
        >>> Mixture(['oxygen', 'nitrogen'], zs=[.5, .5]).legal_statuses
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
        >>> Mixture(['o-xylene', 'm-xylene'], zs=[.5, .5]).economic_statuses
        [["US public: {'Manufactured': 0.0, 'Imported': 0.0, 'Exported': 0.0}",
          u'100,000 - 1,000,000 tonnes per annum',
          'OECD HPV Chemicals'],
         ["US public: {'Manufactured': 39.805, 'Imported': 0.0, 'Exported': 0.0}",
          u'100,000 - 1,000,000 tonnes per annum',
          'OECD HPV Chemicals']]
        '''
        return [i.economic_status for i in self.Chemicals]

    @property
    def UNIFAC_Rs(self):
        r'''UNIFAC `R` (normalized Van der Waals volume) values, dimensionless.
        Used in the UNIFAC model.

        Examples
        --------
        >>> Mixture(['o-xylene', 'm-xylene'], zs=[.5, .5]).UNIFAC_Rs
        [4.6578, 4.6578]
        '''
        return [i.UNIFAC_R for i in self.Chemicals]

    @property
    def UNIFAC_Qs(self):
        r'''UNIFAC `Q` (normalized Van der Waals area) values, dimensionless.
        Used in the UNIFAC model.

        Examples
        --------
        >>> Mixture(['o-xylene', 'decane'], zs=[.5, .5]).UNIFAC_Qs
        [3.536, 6.016]
        '''
        return [i.UNIFAC_Q for i in self.Chemicals]

    @property
    def UNIFAC_groups(self):
        r'''List of dictionaries of UNIFAC subgroup: count groups for each chemical in the mixture. Uses the original
        UNIFAC subgroups, as determined by `DDBST's online service <http://www.ddbst.com/unifacga.html>`_.

        Examples
        --------
        >>> Mixture(['1-pentanol', 'decane'], ws=[0.5, 0.5]).UNIFAC_groups
        [{1: 1, 2: 4, 14: 1}, {1: 2, 2: 8}]
        '''
        return [i.UNIFAC_groups for i in self.Chemicals]

    @property
    def UNIFAC_Dortmund_groups(self):
        r'''List of dictionaries of Dortmund UNIFAC subgroup: count groups for each chemcial in the mixture. Uses the
        Dortmund UNIFAC subgroups, as determined by `DDBST's online service <http://www.ddbst.com/unifacga.html>`_.

        Examples
        --------
        >>> Mixture(['1-pentanol', 'decane'], ws=[0.5, 0.5]).UNIFAC_Dortmund_groups
        [{1: 1, 2: 4, 14: 1}, {1: 2, 2: 8}]
        '''
        return [i.UNIFAC_Dortmund_groups for i in self.Chemicals]

    @property
    def PSRK_groups(self):
        r'''List of dictionaries of PSRK subgroup: count groups for each chemical in the mixture. Uses the PSRK subgroups,
        as determined by `DDBST's online service <http://www.ddbst.com/unifacga.html>`_.

        Examples
        --------
        >>> Mixture(['1-pentanol', 'decane'], ws=[0.5, 0.5]).PSRK_groups
        [{1: 1, 2: 4, 14: 1}, {1: 2, 2: 8}]
        '''
        return [i.PSRK_groups for i in self.Chemicals]

    @property
    def Van_der_Waals_volumes(self):
        r'''List of unnormalized Van der Waals volumes of all the chemicals in
        the mixture, in units of [m^3/mol].

        Examples
        --------
        >>> Mixture(['1-pentanol', 'decane'], ws=[0.5, 0.5]).Van_der_Waals_volumes
        [6.9762279e-05, 0.00010918455800000001]
        '''
        return [i.Van_der_Waals_volume for i in self.Chemicals]

    @property
    def Van_der_Waals_areas(self):
        r'''List of unnormalized Van der Waals areas of all the chemicals
        in the mixture, in units of [m^2/mol].

        Examples
        --------
        >>> Mixture(['1-pentanol', 'decane'], ws=[0.5, 0.5]).Van_der_Waals_areas
        [1052000.0, 1504000.0]
        '''
        return [i.Van_der_Waals_area for i in self.Chemicals]

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
    def Hc(self):
        r'''Standard higher heat of combustion of the mixture,
        in units of [J/kg].

        This property depends on the bulk composition only.
        '''
        return mixing_simple(self.Hcs, self.ws)

    @property
    def Hcm(self):
        r'''Standard higher molar heat of combustion of the mixture,
        in units of [J/mol].

        This property depends on the bulk composition only.
        '''
        return mixing_simple(self.Hcms, self.zs)

    @property
    def Hcm_lower(self):
        r'''Standard lower molar heat of combustion of the mixture,
        in units of [J/mol].

        This property depends on the bulk composition only.
        '''
        return mixing_simple(self.Hcms_lower, self.zs)

    @property
    def Hc_lower(self):
        r'''Standard lower heat of combustion of the mixture,
        in units of [J/kg].

        This property depends on the bulk composition only.
        '''
        return mixing_simple(self.Hcs_lower, self.ws)

    def Hc_volumetric_g(self, T=288.7055555555555, P=101325.0):
        r'''Standard higher molar heat of combustion of the mixture,
        in units of [J/m^3] at the specified `T` and `P` in the gas phase.

        This property depends on the bulk composition only.

        Parameters
        ----------
        T : float, optional
            Reference temperature, [K]
        P : float, optional
            Reference pressure, [Pa]

        Returns
        -------
        Hc_volumetric_g : float, optional
            Higher heat of combustion on a volumetric basis, [J/m^3]
        '''
        Vm = self.VolumeGasMixture(T=T, P=P, zs=self.zs, ws=self.ws)
        Hcm = self.Hcm
        return Hcm/Vm

    def Hc_volumetric_g_lower(self, T=288.7055555555555, P=101325.0):
        r'''Standard lower molar heat of combustion of the mixture,
        in units of [J/m^3] at the specified `T` and `P` in the gas phase.

        This property depends on the bulk composition only.

        Parameters
        ----------
        T : float, optional
            Reference temperature, [K]
        P : float, optional
            Reference pressure, [Pa]

        Returns
        -------
        Hc_volumetric_g : float, optional
            Lower heat of combustion on a volumetric basis, [J/m^3]
        '''
        Vm = self.VolumeGasMixture(T=T, P=P, zs=self.zs, ws=self.ws)
        Hcm_lower = self.Hcm_lower
        return Hcm_lower/Vm

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
        mixture at its current temperature, in units of [m^3/mol].

        Examples
        --------
        >>> Mixture(['iron'], ws=[1], T=320).Vmss
        [7.09593392630242e-06]
        '''
        return [i.Vms for i in self.Chemicals]

    @property
    def Vmls(self):
        r'''Pure component liquid-phase molar volumes of the chemicals in the
        mixture at its current temperature and pressure, in units of [m^3/mol].

        Examples
        --------
        >>> Mixture(['benzene', 'toluene'], ws=[0.5, 0.5], T=320).Vmls
        [9.188896727673715e-05, 0.00010946199496993461]
        '''
        return [i.Vml for i in self.Chemicals]

    @property
    def Vmgs(self):
        r'''Pure component gas-phase molar volumes of the chemicals in the
        mixture at its current temperature and pressure, in units of [m^3/mol].

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
        [5.357870271650772e-07, 3.8127962283230277e-07]
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
        [5.357870271650772e-07, 3.8127962283230277e-07]
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
        [3.3028044028118324e-06, 2.4412958544059014e-06]
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
        [6.13542244155373, 5.034355147908088]
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
        [0.7810364900059606, 0.784358381123896]
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

    @property
    def Parachors(self):
        r'''Pure component Parachor parameters of the chemicals in the
        mixture at its current temperature and pressure, in units
        of [N^0.25*m^2.75/mol].

        .. math::
            P = \frac{\sigma^{0.25} MW}{\rho_L - \rho_V}

        Calculated based on surface tension, density of the liquid and gas
        phase, and molecular weight. For uses of this property, see
        :obj:`thermo.utils.Parachor`.

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).Parachors
        [3.6795616000855504e-05, 4.82947303150274e-05]
        '''
        return [i.Parachor for i in self.Chemicals]

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
            return Vm_to_rho(Vml, self.MWl)
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
        7.914447603999089
        '''
        Vmg = self.Vmg
        if Vmg:
            return Vm_to_rho(Vmg, self.MWg)
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
        0.9403859376888885
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
        25.32745796347474
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
        168.29127923518843
        '''
        return self.HeatCapacityLiquidMixture(self.T, self.P, self.xs, self.wsl)

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
        return self.HeatCapacityGasMixture(self.T, self.P, self.ys, self.wsg)

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
        229.55166388430328
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
        Cplm = self.HeatCapacityLiquidMixture(self.T, self.P, self.xs, self.wsl)
        if Cplm:
            return property_molar_to_mass(Cplm, self.MWl)
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
        Cpgm = self.HeatCapacityGasMixture(self.T, self.P, self.ys, self.wsg)
        if Cpgm:
            return property_molar_to_mass(Cpgm, self.MWg)
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
        Cpgm = self.HeatCapacityGasMixture(self.T, self.P, self.ys, self.wsg)
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
            return property_molar_to_mass(Cvgm, self.MWg)
        return None

    @property
    def speed_of_sound_g(self):
        r'''Gas-phase speed of sound of the mixture at its
        current temperature, [m/s].

        Examples
        --------
        >>> Mixture(['nitrogen'], ws=[1]).speed_of_sound_g
        351.77445481641661
        '''
        dP_dV = 1.0/self.VolumeGasMixture.property_derivative_P(T=self.T, P=self.P,
                                                                zs=self.ys, ws=self.wsg, order=1)

        return speed_of_sound(V=self.Vmg, dP_dV=dP_dV, Cp=self.property_package.Cpgm,
                              Cv=self.property_package.Cvgm, MW=self.MWg)

    @property
    def speed_of_sound_l(self):
        r'''Liquid-phase speed of sound of the mixture at its
        current temperature, [m/s].

        Examples
        --------
        >>> Mixture(['toluene'], P=1E5, T=300, ws=[1]).speed_of_sound_l
        1116.0852487852942
        '''
        dP_dV = 1.0/self.VolumeLiquidMixture.property_derivative_P(T=self.T, P=self.P,
                                                                zs=self.xs, ws=self.wsl, order=1)

        return speed_of_sound(V=self.Vml, dP_dV=dP_dV, Cp=self.property_package.Cplm,
                              Cv=self.property_package.Cvlm, MW=self.MWl)
    @property
    def speed_of_sound(self):
        r'''Bulk speed of sound of the mixture at its
        current temperature, [m/s].

        Examples
        --------
        >>> Mixture(['toluene'], P=1E5, VF=0.5, ws=[1]).speed_of_sound
        478.99527258140211
        '''
        if self.phase == 'l':
            return self.speed_of_sound_l
        elif self.phase == 'g':
            return self.speed_of_sound_g
        elif self.phase == 'l/g':
            return self.speed_of_sound_g*self.x + (1.0 - self.x)*self.speed_of_sound_l

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
        -0.001486976173801296
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
        2.858088468937333e-07
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
        2.5118460023343146e-06
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
        9.444949636299626e-08
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
        1.6968517002221566e-05
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
        2.782821450148889
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

    @property
    def Parachor(self):
        r'''Parachor of the mixture at its
        current temperature and pressure, in units of [N^0.25*m^2.75/mol].

        .. math::
            P = \frac{\sigma^{0.25} MW}{\rho_L - \rho_V}

        Calculated based on surface tension, density of the liquid and gas
        phase, and molecular weight. For uses of this property, see
        :obj:`thermo.utils.Parachor`.

        Examples
        --------
        >>> Mixture(['benzene', 'hexane'], ws=[0.5, 0.5], T=320).Parachor
        4.233407085050756e-05
        '''
        sigma, rhol, rhog = self.sigma, self.rhol, self.rhog
        if all((sigma, rhol, rhog, self.MW)):
            return Parachor(sigma=sigma, MW=self.MW, rhol=rhol, rhog=rhog)
        return None

    ### Properties from Mixture objects
    @property
    def Vml(self):
        r'''Liquid-phase molar volume of the mixture at its current
        temperature, pressure, and composition in units of [m^3/mol]. For
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
        return self.VolumeLiquidMixture(T=self.T, P=self.P, zs=self.xs, ws=self.wsl)

    @property
    def Vmg(self):
        r'''Gas-phase molar volume of the mixture at its current
        temperature, pressure, and composition in units of [m^3/mol]. For
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
        return self.VolumeGasMixture(T=self.T, P=self.P, zs=self.ys, ws=self.wsg)

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
        Vmg = self.VolumeGasMixture(T=288.70555555555552, P=101325, zs=self.ys, ws=self.wsg)
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
        return self.ViscosityLiquidMixture(self.T, self.P, self.xs, self.wsl)

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
        return self.ViscosityGasMixture(self.T, self.P, self.ys, self.wsg)

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
        return self.SurfaceTensionMixture(self.T, self.P, self.xs, self.wsl)

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
        return self.ThermalConductivityLiquidMixture(self.T, self.P, self.xs, self.wsl)

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
        return self.ThermalConductivityGasMixture(self.T, self.P, self.ys, self.wsg)

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
        return phase_select_property(phase=self.phase, s=Mixture.Cps,
                                     l=Mixture.Cpl, g=Mixture.Cpg, self=self)

    @property
    def Cpm(self):
        r'''Molar heat capacity of the mixture at its current phase and
        temperature, in units of [J/mol/K]. Available only if single phase.

        Examples
        --------
        >>> Mixture(['ethylbenzene'], ws=[1], T=550, P=3E6).Cpm
        294.18449553310046
        '''
        return phase_select_property(phase=self.phase, s=Mixture.Cpsm,
                                     l=Mixture.Cplm, g=Mixture.Cpgm, self=self)

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
        return phase_select_property(phase=self.phase, s=Mixture.Vms,
                                     l=Mixture.Vml, g=Mixture.Vmg, self=self)

    @property
    def rho(self):
        r'''Mass density of the mixture at its current phase and
        temperature and pressure, in units of [kg/m^3].
        Available only if single phase.

        Examples
        --------
        >>> Mixture(['decane'], ws=[1], T=550, P=2E6).rho
        498.67008448640604
        '''
        if self.phase == 'l/g':
            # Volume fraction mixing rule for density
            rhol, rhog = self.rhol, self.rhog
            a, b = (1.0 - self.x)/rhol, self.x/rhog
            return rhol*a/(a+b) + b/(a+b)*rhog
        return phase_select_property(phase=self.phase, s=Mixture.rhos,
                                     l=Mixture.rhol, g=Mixture.rhog, self=self)

    @property
    def rhom(self):
        r'''Molar density of the mixture at its current phase and
        temperature and pressure, in units of [mol/m^3].
        Available only if single phase.

        Examples
        --------
        >>> Mixture(['1-hexanol'], ws=[1]).rhom
        7983.414573003429
        '''
        if self.phase == 'l/g':
            # Volume fraction mixing rule for density
            rholm, rhogm = self.rholm, self.rhogm
            a, b = (1.0 - self.x)/rholm, self.x/rhogm
            return rholm*a/(a+b) + b/(a+b)*rhogm
        return phase_select_property(phase=self.phase, s=None, l=Mixture.rholm,
                                     g=Mixture.rhogm, self=self)

    @property
    def Z(self):
        r'''Compressibility factor of the mixture at its current phase and
        temperature and pressure, [dimensionless].
        Available only if single phase.

        Examples
        --------
        >>> Mixture(['MTBE'], ws=[1], T=900, P=1E-2).Z
        0.9999999999056374
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
        return phase_select_property(phase=self.phase, s=Mixture.SGs,
                                     l=Mixture.SGl, g=Mixture.SGg, self=self)


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
        return phase_select_property(phase=self.phase, l=Mixture.isobaric_expansion_l,
                                     g=Mixture.isobaric_expansion_g, self=self)

    @property
    def isobaric_expansion_g(self):
        r'''Isobaric (constant-pressure) expansion of the gas phase of the
        mixture at its current temperature and pressure, in units of [1/K].
        Available only if single phase.

        .. math::
            \beta = \frac{1}{V}\left(\frac{\partial V}{\partial T} \right)_P

        Examples
        --------
        >>> Mixture(['argon'], ws=[1], T=647.1, P=22048320.0).isobaric_expansion_g
        0.0015661100323025273
        '''
        dV_dT = self.VolumeGasMixture.property_derivative_T(self.T, self.P, self.zs, self.ws)
        Vm = self.Vmg
        if dV_dT and Vm:
            return isobaric_expansion(V=Vm, dV_dT=dV_dT)

    @property
    def isobaric_expansion_l(self):
        r'''Isobaric (constant-pressure) expansion of the liquid phase of the
        mixture at its current temperature and pressure, in units of [1/K].
        Available only if single phase.

        .. math::
            \beta = \frac{1}{V}\left(\frac{\partial V}{\partial T} \right)_P

        Examples
        --------
        >>> Mixture(['argon'], ws=[1], T=647.1, P=22048320.0).isobaric_expansion_l
        0.001859152875154442
        '''
        dV_dT = self.VolumeLiquidMixture.property_derivative_T(self.T, self.P, self.zs, self.ws)
        Vm = self.Vml
        if dV_dT and Vm:
            return isobaric_expansion(V=Vm, dV_dT=dV_dT)

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
        return phase_select_property(phase=self.phase, l=Mixture.JTl,
                                     g=Mixture.JTg, self=self)

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
        return phase_select_property(phase=self.phase, l=Mixture.mul,
                                     g=Mixture.mug, self=self)

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
        return phase_select_property(phase=self.phase, s=None, l=Mixture.kl,
                                     g=Mixture.kg, self=self)

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
        1.3842643382482236e-05
        '''
        return phase_select_property(phase=self.phase, l=Mixture.nul,
                                     g=Mixture.nug, self=self)

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
        8.696537158635412e-08
        '''
        return phase_select_property(phase=self.phase, l=Mixture.alphal,
                                     g=Mixture.alphag, self=self)

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
        4.183039103542711
        '''
        return phase_select_property(phase=self.phase, l=Mixture.Prl,
                                     g=Mixture.Prg, self=self)

    ### Standard state properties

    @property
    def Vml_STP(self):
        r'''Liquid-phase molar volume of the mixture at 298.15 K and 101.325 kPa,
        and the current composition in units of [m^3/mol].

        Examples
        --------
        >>> Mixture(['cyclobutane'], ws=[1]).Vml_STP
        8.143327329133706e-05
        '''
        return self.VolumeLiquidMixture(T=298.15, P=101325, zs=self.zs, ws=self.ws)

    @property
    def Vmg_STP(self):
        r'''Gas-phase molar volume of the mixture at 298.15 K and 101.325 kPa,
        and the current composition in units of [m^3/mol].

        Examples
        --------
        >>> Mixture(['nitrogen'], ws=[1]).Vmg_STP
        0.02445443688838904
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
        71.34707841728181
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

    @property
    def constants(self):
        r'''Returns a :obj:`thermo.chemical_package.ChemicalConstantsPackage
        instance with constants from the mixture, [-].

        '''
        try:
            return self._constants
        except AttributeError:
            pass
        from thermo.chemical_package import ChemicalConstantsPackage

        self._constants = ChemicalConstantsPackage(CASs=self.CASs, names=self.names, MWs=self.MWs,
                                                   Tms=self.Tms, Tbs=self.Tbs,
                 # Critical state points
                 Tcs=self.Tcs, Pcs=self.Pcs, Vcs=self.Vcs, omegas=self.omegas,
                 Zcs=self.Zcs, rhocs=self.rhocms, rhocs_mass=self.rhocs,
                 # Phase change enthalpy
                 Hfus_Tms=self.Hfusms, Hfus_Tms_mass=self.Hfuss, Hvap_Tbs=self.Hvap_Tbms,
                 Hvap_Tbs_mass=self.Hvap_Tbs,
                 # Standard values
                 Vml_STPs=self.Vml_STPs, rhol_STPs=self.rholm_STPs, rhol_STPs_mass=self.rhol_STPs,
                 Vml_60Fs=self.Vml_60Fs, rhol_60Fs=self.rhoml_60Fs, rhol_60Fs_mass=self.rhol_60Fs,
                 # Reaction (ideal gas)
                 Hfgs=self.Hfgms, Hfgs_mass=self.Hfgs, Gfgs=self.Gfgms, Gfgs_mass=self.Gfgs,
                 Sfgs=self.Sfgms, Sfgs_mass=self.Sfgs, S0gs=self.S0gms, S0gs_mass=self.S0gs,

                 # Triple point
                 Tts=self.Tts, Pts=self.Pts, Hsub_Tts=self.Hsubms, Hsub_Tts_mass=self.Hsubs,
                 # Combustion
                 Hcs=self.Hcms, Hcs_mass=self.Hcs, Hcs_lower=self.Hcms_lower, Hcs_lower_mass=self.Hcs_lower,
                 # Fire safety
                 Tflashs=self.Tflashs, Tautoignitions=self.Tautoignitions, LFLs=self.LFLs, UFLs=self.UFLs,
                 # Other safety
                 TWAs=self.TWAs, STELs=self.STELs, Ceilings=self.Ceilings, Skins=self.Skins,
                 Carcinogens=self.Carcinogens, legal_statuses=self.legal_statuses, economic_statuses=self.economic_statuses,
                 # Environmental
                 GWPs=self.GWPs, ODPs=self.ODPs, logPs=self.logPs,
                 Psat_298s=self.Psat_298s, Hvap_298s=self.Hvapm_298s,
                 Hvap_298s_mass=self.Hvap_298s, Vml_Tms=self.Vml_Tms,
                 rhos_Tms=self.rhoms_Tm, rhos_Tms_mass=self.rhos_Tms, Vms_Tms=self.Vms_Tms,
                 # Analytical
                 RIs=self.RIs, RI_Ts=self.RI_Ts, conductivities=self.conductivities,
                 conductivity_Ts=self.conductivity_Ts,
                 # Odd constants
                 charges=self.charges, dipoles=self.dipoles, Stockmayers=self.Stockmayers,
                 molecular_diameters=self.molecular_diameters, Van_der_Waals_volumes=self.Van_der_Waals_volumes,
                 Van_der_Waals_areas=self.Van_der_Waals_areas, Parachors=self.Parachors, StielPolars=self.StielPolars,
                 atomss=self.atomss, atom_fractions=self.atom_fractionss,
                 similarity_variables=self.similarity_variables, phase_STPs=self.phase_STPs,
                 UNIFAC_Rs=self.UNIFAC_Rs, UNIFAC_Qs=self.UNIFAC_Qs, solubility_parameters=self.solubility_parameters_STP,
                 # Other identifiers
                 PubChems=self.PubChems, formulas=self.formulas, smiless=self.smiless, InChIs=self.InChIs,
                 InChI_Keys=self.InChI_Keys,
                 # Groups
                 UNIFAC_groups=self.UNIFAC_groups, UNIFAC_Dortmund_groups=self.UNIFAC_Dortmund_groups,
                 PSRK_groups=self.PSRK_groups)
        return self._constants


    def properties(self, copy_pures=True, copy_mixtures=True):
        try:
            return self._properties
        except AttributeError:
            pass

        from thermo.chemical_package import PropertyCorrelationsPackage
        constants = self.constants
        kwargs = dict(constants=constants)
        if copy_pures:
            kwargs.update(VaporPressures=self.VaporPressures, SublimationPressures=self.SublimationPressures,
                 VolumeGases=self.VolumeGases, VolumeLiquids=self.VolumeLiquids, VolumeSolids=self.VolumeSolids,
                 HeatCapacityGases=self.HeatCapacityGases, HeatCapacityLiquids=self.HeatCapacityLiquids,
                 HeatCapacitySolids=self.HeatCapacitySolids,
                 ViscosityGases=self.ViscosityGases, ViscosityLiquids=self.ViscosityLiquids,
                 ThermalConductivityGases=self.ThermalConductivityGases, ThermalConductivityLiquids=self.ThermalConductivityLiquids,
                 ThermalConductivitySolids=self.ThermalConductivitySolids,
                 EnthalpyVaporizations=self.EnthalpyVaporizations, EnthalpySublimations=self.EnthalpySublimations,
                 SurfaceTensions=self.SurfaceTensions, PermittivityLiquids=self.Permittivities)
        if copy_mixtures:
            kwargs.update(VolumeGasMixtureObj=self.VolumeGasMixture, VolumeLiquidMixtureObj=self.VolumeLiquidMixture,
                          VolumeSolidMixtureObj=self.VolumeSolidMixture,
                          HeatCapacityGasMixtureObj=self.HeatCapacityGasMixture,
                          HeatCapacityLiquidMixtureObj=self.HeatCapacityLiquidMixture,
                          HeatCapacitySolidMixtureObj=self.HeatCapacitySolidMixture,
                          ViscosityGasMixtureObj=self.ViscosityGasMixture,
                          ViscosityLiquidMixtureObj=self.ViscosityLiquidMixture,
                          ThermalConductivityGasMixtureObj=self.ThermalConductivityGasMixture,
                          ThermalConductivityLiquidMixtureObj=self.ThermalConductivityLiquidMixture,
                          SurfaceTensionMixtureObj=self.SurfaceTensionMixture)

        self._properties = PropertyCorrelationsPackage(**kwargs)
        return self._properties
