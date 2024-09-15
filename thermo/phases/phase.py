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
'''

__all__ = [
    'Phase',
    'IdealGasDeparturePhase',
    'derivatives_thermodynamic',
    'derivatives_thermodynamic_mass',
    'derivatives_jacobian',
]

from math import sqrt

from chemicals.utils import (
    Cp_minus_Cv,
    Joule_Thomson,
    dns_to_dn_partials,
    dxs_to_dns,
    hash_any_primitive,
    isentropic_exponent_PT,
    isentropic_exponent_PV,
    isentropic_exponent_TV,
    mixing_simple,
    normalize,
    object_data,
    phase_identification_parameter,
    property_molar_to_mass,
    speed_of_sound,
)
from chemicals.virial import B_from_Z
from fluids.constants import R, R_inv
from fluids.core import c_ideal_gas, thermal_diffusivity
from fluids.numerics import (
    horner,
    horner_log,
    is_micropython,
    jacobian,
    log,
    newton_system,
    poly_fit_integral_over_T_value,
    poly_fit_integral_value,
    trunc_exp,
    trunc_exp_numpy,
    trunc_log,
    trunc_log_numpy,
)
from fluids.numerics import numpy as np

from thermo import phases
from thermo.serialize import JsonOptEncodable
from thermo.utils import POLY_FIT

try:
    dot, zeros, array = np.dot, np.zeros, np.array
except:
    pass

class Phase:
    '''`Phase` is the base class for all phase objects in `thermo`. Each
    sub-class implements a number of core properties; many other properties
    can be calculated from them.

    Among those properties are `H`, `S`, `Cp`, `dP_dT`, `dP_dV`,
    `d2P_dT2`, `d2P_dV2`, and `d2P_dTdV`.

    An additional set of properties that can be implemented and that enable
    more functionality are `dH_dP`, `dS_dT`, `dS_dP`, `d2H_dT2`, `d2H_dP2`,
    `d2S_dP2`, `dH_dT_V`, `dH_dP_V`, `dH_dV_T`, `dH_dV_P`, `dS_dT_V`,
    `dS_dP_V`, `d2H_dTdP`, `d2H_dT2_V`, `d2P_dTdP`, `d2P_dVdP`, `d2P_dVdT_TP`,
    `d2P_dT2_PV`.

    Some models may re-implement properties which would normally be
    calculated by this `Phase` base class because they have more explicit,
    faster ways of calculating the property.

    When a phase object is the result of a Flash calculation, the resulting
    phase objects have a reference to a
    :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>`
    object and all of its properties can be accessed from from the resulting
    phase objects as well.

    A :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>`
    object can also be manually set to the attribute `constants` to enable
    access to those properties. This includes mass-based properties, which are
    not accessible from Phase objects without a reference to the constants.

    '''

    INCOMPRESSIBLE_CONST = 1e30
    R = R
    R2 = R*R
    R_inv = R_inv

    is_solid = False

    ideal_gas_basis = False # Parameter fot has the same ideal gas Cp
    T_REF_IG = 298.15
    T_REF_IG_INV = 1.0/T_REF_IG
    """The numerical inverse of :obj:`T_REF_IG`, stored to save a division.
    """
    P_REF_IG = 101325.
    P_REF_IG_INV = 1.0/P_REF_IG
    LOG_P_REF_IG = log(P_REF_IG)

    T_MAX_FLASH = T_MAX_FIXED = 10000.0
    T_MIN_FLASH = T_MIN_FIXED = 1e-3

    P_MAX_FIXED = 1e9
    P_MIN_FIXED = 1e-2 # 1e-3 was so low issues happened in the root stuff, could not be fixed

    V_MIN_FIXED = 1e-9 # m^3/mol
    V_MAX_FIXED = 1e9 # m^#/mol

    T_MIN_FLASH = 1e-300

    T_DEFAULT = T_REF_IG
    P_DEFAULT = P_REF_IG

    force_phase = None
    """Attribute which can be set to a global Phase object to force the phases
    identification routines to label it a certain phase. Accepts values of ('g', 'l', 's')."""

    _Psats_data = None
    _Cpgs_data = None
    Psats_poly_fit = False
    Cpgs_poly_fit = False
    composition_independent = False
    vectorized = False

    supports_lnphis_args = False

    __slots__ = ('__dict__', 'result', 'constants', 'correlations')

    pure_references = ()
    """Tuple of attribute names which hold lists of :obj:`thermo.utils.TDependentProperty`
    or :obj:`thermo.utils.TPDependentProperty` instances."""

    pure_reference_types = ()
    """Tuple of types of :obj:`thermo.utils.TDependentProperty`
    or :obj:`thermo.utils.TPDependentProperty` corresponding to `pure_references`."""

    obj_references = ('result', 'constants', 'correlations')
    """Tuple of object instances which should be stored as json using their own
    as_json method.
    """
    json_version = 1
    non_json_attributes =  ['_model_hash', '_model_hash_ignore_phase']

    pointer_references = ()
    """Tuple of attributes which should be stored by converting them to
    a string, and then they will be looked up in their corresponding
    `pointer_reference_dicts` entry.
    """
    pointer_reference_dicts = ()
    """Tuple of dictionaries for string -> object
    """
    reference_pointer_dicts = ()
    """Tuple of dictionaries for object -> string
    """

    bulk_phase_type = False

    if not is_micropython:
        def __init_subclass__(cls):
            cls.__full_path__ = f"{cls.__module__}.{cls.__qualname__}"
    else:
        __full_path__ = None

    def __str__(self):
        s =  f'<{self.__class__.__name__}, '
        try:
            s += f'T={self.T:g} K, P={self.P:g} Pa'
        except:
            pass
        s += '>'
        return s

    def as_json(self, cache=None, option=0):
        r'''Method to create a JSON-friendly serialization of the phase
        which can be stored, and reloaded later.

        Returns
        -------
        json_repr : dict
            JSON-friendly representation, [-]

        Notes
        -----

        Examples
        --------
        >>> import json
        >>> from thermo import IAPWS95Liquid
        >>> phase = IAPWS95Liquid(T=300, P=1e5, zs=[1])
        >>> new_phase = Phase.from_json(json.loads(json.dumps(phase.as_json())))
        >>> assert phase == new_phase
        '''
        return JsonOptEncodable.as_json(self, cache, option)

    @classmethod
    def from_json(cls, json_repr, cache=None):
        r'''Method to create a phase from a JSON
        serialization of another phase.

        Parameters
        ----------
        json_repr : dict
            JSON-friendly representation, [-]

        Returns
        -------
        phase : :obj:`Phase`
            Newly created phase object from the json serialization, [-]

        Notes
        -----
        It is important that the input string be in the same format as that
        created by :obj:`Phase.as_json`.

        Examples
        --------
        '''
        return JsonOptEncodable.from_json(json_repr, cache)

    def __eq__(self, other):
        return self.__hash__() == hash(other)

    def state_hash(self):
        r'''Basic method to calculate a hash of the state of the phase and its
        model parameters.

        Note that the hashes should only be compared on the same system running
        in the same process!

        Returns
        -------
        state_hash : int
            Hash of the object's model parameters and state, [-]
        '''
        return hash_any_primitive((self.model_hash(), self.T, self.P, self.V(), self.zs))

    __hash__ = state_hash

    def exact_hash(self):
        r'''Method to calculate and return a hash representing the exact state
        of the object.

        Returns
        -------
        hash : int
            Hash of the object, [-]
        '''
        # Ensure the hash is set so it is always part of the object hash
        self.model_hash(False)
        self.model_hash(True)
        self.state_hash()
        d = object_data(self)
        ans = hash_any_primitive((self.__class__.__name__, d))
        return ans


    def is_same_model(self, other_phase, ignore_phase=False):
        r'''Method to check whether or not a model is the exact same
        as another. In the case `ignore_phase` is True, whether the
        model is liquid or gas is omitted as in the case of CEOSGas
        and CEOSLiquid.

        Parameters
        ----------
        other_phase : Phase
            The phase to compare against, [-]
        ignore_phase : bool
            Whether or not to include the specifc class of the model in the
            hash

        Returns
        -------
        same : bool
            Whether they are the same or not

        Notes
        -----
        This may be quicker to calculate than the model hash.
        '''
        # Are we the same object? If so, obviously the same
        if self is other_phase:
            return True

        # Are the attributes the same? If not, we can't be the same
        if self.model_attributes != other_phase.model_attributes:
            return False

        identical_model_attribute_ids = True
        for attr in self.model_attributes:
            if getattr(self, attr) is not getattr(other_phase, attr):
                identical_model_attribute_ids = False
                break
        if identical_model_attribute_ids:
            if ignore_phase:
                return True
            return self.__class__.__name__ == other_phase.__class__.__name__

        # Using identities only we could not confirm if the phase was the same or not.
        # The values may still be the same if the identities are not.
        return self.model_hash(ignore_phase) == other_phase.model_hash(ignore_phase)

    def model_hash(self, ignore_phase=False):
        r'''Method to compute a hash of a phase.

        Parameters
        ----------
        ignore_phase : bool
            Whether or not to include the specifc class of the model in the
            hash

        Returns
        -------
        hash : int
            Hash representing the settings of the phase; phases with all
            identical model parameters should have the same hash.
        '''
        if ignore_phase:
            try:
                return self._model_hash_ignore_phase
            except AttributeError:
                pass
        else:
            try:
                return self._model_hash
            except AttributeError:
                pass
        # Note: not all attributes are in __dict__, must use getattr
        to_hash = [getattr(self, v) for v in self.model_attributes]
        to_hash = [v.model_hash() if( hasattr(v, 'model_hash') and not isinstance(v, type)) else v for v in to_hash]
        # print(to_hash)
        self._model_hash_ignore_phase = h = hash_any_primitive(to_hash)
        self._model_hash = hash((self.__class__.__name__, h))
        if ignore_phase:
            return self._model_hash_ignore_phase
        else:
            return self._model_hash

    def value(self, name):
        r'''Method to retrieve a property from a string. This more or less
        wraps `getattr`.

        `name` could be a python property like 'Tms' or a callable method
        like 'H'.

        Parameters
        ----------
        name : str
            String representing the property, [-]

        Returns
        -------
        value : various
            Value specified, [various]

        Notes
        -----
        '''
        if name in ('beta_mass',):
            return self.result.value(name, self)

        v = getattr(self, name)
        try:
            v = v()
        except:
            pass
        return v

    ### Methods that should be implemented by subclasses

    def to_TP_zs(self, T, P, zs):
        r'''Method to create a new Phase object with the same constants as the
        existing Phase but at a different `T` and `P`.

        Parameters
        ----------
        zs : list[float]
            Molar composition of the new phase, [-]
        T : float
            Temperature of the new phase, [K]
        P : float
            Pressure of the new phase, [Pa]

        Returns
        -------
        new_phase : Phase
            New phase at the specified conditions, [-]

        Notes
        -----
        This method is marginally faster than :obj:`Phase.to` as it does not
        need to check what the inputs are.

        Examples
        --------
        >>> from thermo import IdealGas
        >>> phase = IdealGas(T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=[])
        >>> phase.to_TP_zs(T=1e5, P=1e3, zs=[.5, .5])
        IdealGas(HeatCapacityGases=[], T=100000.0, P=1000.0, zs=[0.5, 0.5])
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def to(self, zs, T=None, P=None, V=None):
        r'''Method to create a new Phase object with the same constants as the
        existing Phase but at different conditions. Mole fractions `zs` are
        always required and any two of `T`, `P`, and `V` are required.

        Parameters
        ----------
        zs : list[float]
            Molar composition of the new phase, [-]
        T : float, optional
            Temperature of the new phase, [K]
        P : float, optional
            Pressure of the new phase, [Pa]
        V : float, optional
            Molar volume of the new phase, [m^3/mol]

        Returns
        -------
        new_phase : Phase
            New phase at the specified conditions, [-]

        Notes
        -----

        Examples
        --------
        These sample cases illustrate the three combinations of inputs.
        Note that some thermodynamic models may have multiple solutions for
        some inputs!

        >>> from thermo import IdealGas
        >>> phase = IdealGas(T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=[])
        >>> phase.to(T=1e5, P=1e3, zs=[.5, .5])
        IdealGas(HeatCapacityGases=[], T=100000.0, P=1000.0, zs=[0.5, 0.5])
        >>> phase.to(V=1e-4, P=1e3, zs=[.1, .9])
        IdealGas(HeatCapacityGases=[], T=0.012027235504, P=1000.0, zs=[0.1, 0.9])
        >>> phase.to(T=1e5, V=1e12, zs=[.2, .8])
        IdealGas(HeatCapacityGases=[], T=100000.0, P=8.31446261e-07, zs=[0.2, 0.8])

        '''
        raise NotImplementedError("Must be implemented by subphases")

    def V(self):
        r'''Method to return the molar volume of the phase.

        Returns
        -------
        V : float
            Molar volume, [m^3/mol]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def dP_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        pressure of the phase.

        Returns
        -------
        dP_dT : float
            First temperature derivative of pressure, [Pa/K]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def dP_dV(self):
        r'''Method to calculate and return the first volume derivative of
        pressure of the phase.

        Returns
        -------
        dP_dV : float
            First volume derivative of pressure, [Pa*mol/m^3]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def d2P_dT2(self):
        r'''Method to calculate and return the second temperature derivative of
        pressure of the phase.

        Returns
        -------
        d2P_dT2 : float
            Second temperature derivative of pressure, [Pa/K^2]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def d2P_dV2(self):
        r'''Method to calculate and return the second volume derivative of
        pressure of the phase.

        Returns
        -------
        d2P_dV2 : float
            Second volume derivative of pressure, [Pa*mol^2/m^6]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def d2P_dTdV(self):
        r'''Method to calculate and return the second derivative of
        pressure with respect to temperature and volume of the phase.

        Returns
        -------
        d2P_dTdV : float
            Second volume derivative of pressure, [mol*Pa^2/(J*K)]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def lnphis(self):
        r'''Method to calculate and return the log of fugacity coefficients of
        each component in the phase.

        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def dlnphis_dT(self):
        r'''Method to calculate and return the temperature derivative of the
        log of fugacity coefficients of each component in the phase.

        Returns
        -------
        dlnphis_dT : list[float]
            First temperature derivative of log fugacity coefficients, [1/K]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def dlnphis_dP(self):
        r'''Method to calculate and return the pressure derivative of the
        log of fugacity coefficients of each component in the phase.

        Returns
        -------
        dlnphis_dP : list[float]
            First pressure derivative of log fugacity coefficients, [1/Pa]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def H(self):
        r'''Method to calculate and return the enthalpy of the phase.
        The reference state for most subclasses is an ideal-gas enthalpy of
        zero at 298.15 K and 101325 Pa.

        Returns
        -------
        H : float
            Molar enthalpy, [J/(mol)]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def S(self):
        r'''Method to calculate and return the entropy of the phase.
        The reference state for most subclasses is an ideal-gas entropy of
        zero at 298.15 K and 101325 Pa.

        Returns
        -------
        S : float
            Molar entropy, [J/(mol*K)]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def Cp(self):
        r'''Method to calculate and return the constant-pressure heat capacity
        of the phase.

        Returns
        -------
        Cp : float
            Molar heat capacity, [J/(mol*K)]
        '''
        raise NotImplementedError("Must be implemented by subphases")

    ### Benchmarking methods
    def _compute_main_properties(self):
        '''Method which computes some basic properties. For benchmarking;
        accepts no arguments and returns nothing. A timer should be used
        outside of this method.
        '''
        self.H()
        self.S()
        self.Cp()
        self.Cv()
        self.dP_dT()
        self.dP_dV()
        self.d2P_dT2()
        self.d2P_dV2()
        self.d2P_dTdV()
        self.PIP()

    ### Consistency Checks
    def S_phi_consistency(self):
        r'''Method to calculate and return a consistency check between ideal
        gas entropy behavior, and the fugacity coefficients and their
        temperature derivatives.

        .. math::
             S = S^{ig} - \sum_{i} z_i R\left(\ln \phi_i +  T \frac{\partial \ln
             \phi_i}{\partial T}\right)

        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - S^{\text{from phi}}/S^\text{implemented}|`, [-]
        '''
        # From coco
        try:
            return self._S_phi_consistency
        except:
            pass
        S0 = self.S_ideal_gas()
        lnphis = self.lnphis()
        dlnphis_dT = self.dlnphis_dT()
        T, zs = self.T, self.zs
        if self.vectorized:
            S0 -= float((zs*(R*lnphis + R*T*dlnphis_dT)).sum())
        else:
            for i in range(self.N):
                S0 -= zs[i]*(R*lnphis[i] + R*T*dlnphis_dT[i])
        self._S_phi_consistency = abs(1.0 - S0/self.S())
        return self._S_phi_consistency


    def H_phi_consistency(self):
        r'''Method to calculate and return a consistency check between ideal
        gas enthalpy behavior, and the fugacity coefficients and their
        temperature derivatives.

        .. math::
             H^{\text{from phi}} = H^{ig} - RT^2\sum_i z_i \frac{\partial \ln
             \phi_i}{\partial T}

        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - H^{\text{from phi}}/H^\text{implemented}|`, [-]
        '''
        try:
            return self._H_phi_consistency
        except:
            pass
        self._H_phi_consistency = abs(1.0 - self.H_from_phi()/self.H())
        return self._H_phi_consistency

    def G_dep_phi_consistency(self):
        r'''Method to calculate and return a consistency check between
        departure Gibbs free energy, and the fugacity coefficients.

        .. math::
             G^{\text{from phi}}_{dep} = RT\sum_i z_i \phi_i

        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - G^{\text{from phi}}_{dep}/G^\text{implemented}_{dep}|`, [-]
        '''
        # Chapter 2 equation 31 Michaelson
        try:
            return self._G_dep_phi_consistency
        except:
            pass
        zs, T = self.zs, self.T
        G_dep_RT = 0.0
        lnphis = self.lnphis()
        if self.vectorized:
            G_dep_RT = float(dot(zs, lnphis))
        else:
            G_dep_RT = sum(zs[i]*lnphis[i] for i in range(self.N))
        G_dep = G_dep_RT*R*T
        self._G_dep_phi_consistency = abs(1.0 - G_dep/self.G_dep())
        return self._G_dep_phi_consistency

    def H_dep_phi_consistency(self):
        r'''Method to calculate and return a consistency check between
        departure enthalpy, and the fugacity coefficients' temperature
        derivatives.

        .. math::
             H^{\text{from phi}}_{dep} = -RT^2\sum_i z_i \frac{\partial \ln
             \phi_i}{\partial T}

        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - H^{\text{from phi}}_{dep}/H^\text{implemented}_{dep}|`, [-]
        '''
        try:
            return self._H_dep_phi_consistency
        except:
            pass
        H_dep_RT2 = 0.0
        dlnphis_dTs = self.dlnphis_dT()
        zs, T = self.zs, self.T
        if self.vectorized:
            H_dep_RT2 = float(dot(zs, dlnphis_dTs))
        else:
            H_dep_RT2 = sum([zs[i]*dlnphis_dTs[i] for i in range(self.N)])
        H_dep_recalc = -H_dep_RT2*R*T*T
        H_dep = self.H_dep()
        self._H_dep_phi_consistency = abs(1.0 - H_dep/H_dep_recalc)
        return self._H_dep_phi_consistency

    def S_dep_phi_consistency(self):
        r'''Method to calculate and return a consistency check between ideal
        gas entropy behavior, and the fugacity coefficients and their
        temperature derivatives.

        .. math::
             S_{dep}^{\text{from phi}} = - \sum_{i} z_i R\left(\ln \phi_i
             +  T \frac{\partial \ln \phi_i}{\partial T}\right)

        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - S^{\text{from phi}}_{dep}/S^\text{implemented}_{dep}|`, [-]
        '''
        try:
            return self._S_dep_phi_consistency
        except:
            pass
        # From coco
        lnphis = self.lnphis()
        dlnphis_dT = self.dlnphis_dT()
        T, zs = self.T, self.zs
        S_dep = 0.0
        if self.vectorized:
            S_dep -= float((zs*(R*lnphis + R*T*dlnphis_dT)).sum())
        else:
            for i in range(self.N):
                S_dep -= zs[i]*(R*lnphis[i] + R*T*dlnphis_dT[i])
        self._S_dep_phi_consistency = abs(1.0 - S_dep/self.S_dep())
        return self._S_dep_phi_consistency

    def V_phi_consistency(self):
        r'''Method to calculate and return a consistency check between
        molar volume, and the fugacity coefficients' pressures
        derivatives.

        .. math::
            V^{\text{from phi P der}} = \left(\left(\sum_i z_i \frac{\partial \ln
             \phi_i}{\partial P}\right)P + 1\right)RT/P


        Returns
        -------
        error : float
            Relative consistency error
            :math:`|1 - V^{\text{from phi P der}}/V^\text{implemented}|`, [-]
        '''
        try:
            return self._V_phi_consistency
        except:
            pass
        zs, P = self.zs, self.P
        dlnphis_dP = self.dlnphis_dP()
        if self.vectorized:
            lhs = float(dot(zs, dlnphis_dP))
        else:
            lhs = sum(zs[i]*dlnphis_dP[i] for i in range(self.N))
        Z_calc = lhs*P + 1.0
        V_calc = Z_calc*self.R*self.T/P
        V = self.V()
        self._V_phi_consistency = abs(1.0 - V_calc/V)
        return self._V_phi_consistency

    def H_from_phi(self):
        r'''Method to calculate and return the enthalpy of the fluid as
        calculated from the ideal-gas enthalpy and the the fugacity
        coefficients' temperature derivatives.

        .. math::
             H^{\text{from phi}} = H^{ig} - RT^2\sum_i z_i \frac{\partial \ln
             \phi_i}{\partial T}

        Returns
        -------
        H : float
            Enthalpy as calculated from fugacity coefficient temperature
            derivatives [J/mol]
        '''
        H0 = self.H_ideal_gas()
        dlnphis_dT = self.dlnphis_dT()
        T, zs = self.T, self.zs
        if self.vectorized:
            H0 -= R*T*T*float(dot(zs, dlnphis_dT))
        else:
            for i in range(self.N):
                H0 -= R*T*T*zs[i]*dlnphis_dT[i]
        return H0

    def S_from_phi(self):
        r'''Method to calculate and return the entropy of the fluid as
        calculated from the ideal-gas entropy and the the fugacity
        coefficients' temperature derivatives.

        .. math::
             S = S^{ig} - \sum_{i} z_i R\left(\ln \phi_i +  T \frac{\partial \ln
             \phi_i}{\partial T}\right)

        Returns
        -------
        S : float
            Entropy as calculated from fugacity coefficient temperature
            derivatives [J/(mol*K)]
        '''
        S0 = self.S_ideal_gas()
        lnphis = self.lnphis()
        dlnphis_dT = self.dlnphis_dT()
        T, zs = self.T, self.zs
        if self.vectorized:
            S0 -= float((zs*(R*lnphis + R*T*dlnphis_dT)).sum())
        else:
            for i in range(self.N):
                S0 -= zs[i]*(R*lnphis[i] + R*T*dlnphis_dT[i])
        return S0

    def V_from_phi(self):
        r'''Method to calculate and return the molar volume of the fluid as
        calculated from the pressure derivatives of fugacity coefficients.

        .. math::
            V^{\text{from phi P der}} = \left(\left(\sum_i z_i \frac{\partial \ln
             \phi_i}{\partial P}\right)P + 1\right)RT/P


        Returns
        -------
        V : float
            Molar volume, [m^3/mol]
        '''
        zs, P = self.zs, self.P
        dlnphis_dP = self.dlnphis_dP()
        if self.vectorized:
            obj = float(dot(zs, dlnphis_dP))
        else:
            obj = sum(zs[i]*dlnphis_dP[i] for i in range(self.N))
        Z = P*obj + 1.0
        return Z*self.R*self.T/P

    def G_min_criteria(self):
        r'''Method to calculate and return the Gibbs energy criteria required
        for comparing phase stability. This calculation can be faster
        than calculating the full Gibbs energy. For this comparison to work,
        all phases must use the ideal gas basis.

        .. math::
             G^{\text{criteria}} = G^{dep} + RT\sum_i z_i \ln z_i

        Returns
        -------
        G_crit : float
            Gibbs free energy like criteria [J/mol]
        '''
        # Definition implemented that does not use the H, or S ideal gas contribution
        # Allows for faster checking of which phase is at lowest G, but can only
        # be used when all models use an ideal gas basis
        zs = self.zs
        log_zs = self.log_zs()
        G_crit = 0.0
        if self.vectorized:
            G_crit += float(dot(zs, log_zs))
        else:
            for i in range(self.N):
                G_crit += zs[i]*log_zs[i]

        G_crit = G_crit*R*self.T + self.G_dep()
        return G_crit

    def lnphis_at_zs(self, zs, most_stable=False):
        r'''Method to directly calculate the log fugacity coefficients at a
        different composition than the current phase.
        This is implemented to allow for the possibility of more direct
        calls to obtain fugacities than is possible with the phase interface.
        This base method simply creates a new phase, gets its log fugacity
        coefficients, and returns them.

        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
        '''
        obj = self.to_TP_zs(self.T, self.P, zs)
        if most_stable:
            return obj.lnphis_lowest_Gibbs()
        return obj.lnphis()

    def fugacities_at_zs(self, zs, most_stable=False):
        r'''Method to directly calculate the figacities at a
        different composition than the current phase.
        This is implemented to allow for the possibility of more direct
        calls to obtain fugacities than is possible with the phase interface.
        This base method simply creates a new phase, gets its log fugacity
        coefficients, exponentiates them, and multiplies them by `P` and
        compositions.

        Returns
        -------
        fugacities : list[float]
            Fugacities, [Pa]
        '''
        obj = self.to_TP_zs(self.T, self.P, zs)
        if most_stable:
            return obj.fugacities_lowest_Gibbs()
        return obj.fugacities()
        # P = self.P
        # fugacities_lowest_Gibbs
        # lnphis = self.lnphis_at_zs(zs, most_stable)
        # if self.vectorized:
        #     return P*zs*trunc_exp_numpy(lnphis)
        # else:
        #     return [P*zs[i]*trunc_exp(lnphis[i]) for i in range(self.N)]

    def lnphi(self):
        r'''Method to calculate and return the log of fugacity coefficient of
        the phase; provided the phase is 1 component.

        Returns
        -------
        lnphi : list[float]
            Log fugacity coefficient, [-]
        '''
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.lnphis()[0]

    def phi(self):
        r'''Method to calculate and return the fugacity coefficient of
        the phase; provided the phase is 1 component.

        Returns
        -------
        phi : list[float]
            Fugacity coefficient, [-]
        '''
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.phis()[0]

    def fugacity(self):
        r'''Method to calculate and return the fugacity of
        the phase; provided the phase is 1 component.

        Returns
        -------
        fugacity : list[float]
            Fugacity, [Pa]
        '''
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.fugacities()[0]

    def dfugacity_dT(self):
        r'''Method to calculate and return the temperature derivative of
        fugacity of the phase; provided the phase is 1 component.

        Returns
        -------
        dfugacity_dT : list[float]
            Fugacity first temperature derivative, [Pa/K]
        '''
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.dfugacities_dT()[0]

    def dfugacity_dP(self):
        r'''Method to calculate and return the pressure derivative of
        fugacity of the phase; provided the phase is 1 component.

        Returns
        -------
        dfugacity_dP : list[float]
            Fugacity first pressure derivative, [-]
        '''
        if self.N != 1:
            raise ValueError("Property not supported for multicomponent phases")
        return self.dfugacities_dP()[0]


    def fugacities(self):
        r'''Method to calculate and return the fugacities of the phase.

        .. math::
            f_i = P z_i \exp(\ln \phi_i)

        Returns
        -------
        fugacities : list[float]
            Fugacities, [Pa]
        '''
        try:
            return self._fugacities
        except:
            pass
        P = self.P
        zs = self.zs
        lnphis = self.lnphis()
        if self.vectorized:
            self._fugacities = P*zs*trunc_exp_numpy(lnphis)
        else:
            self._fugacities = [P*zs[i]*trunc_exp(lnphis[i]) for i in range(self.N)]
        return self._fugacities

    def lnfugacities(self):
        r'''Method to calculate and return the log of fugacities of the phase.

        .. math::
            \ln f_i = \ln\left( P z_i \exp(\ln \phi_i)\right)
            = \ln(P) + \ln(z_i) + \ln \phi_i

        Returns
        -------
        lnfugacities : list[float]
            Log fugacities, [log(Pa)]
        '''
        try:
            return self._lnfugacities
        except:
            pass
        P = self.P
        lnphis = self.lnphis()
        logP = log(P)
        log_zs = self.log_zs()
        if self.vectorized:
            lnfugacities = logP + log_zs + lnphis
        else:
            lnfugacities = [logP + log_zs[i] + lnphis[i] for i in range(self.N)]
        self._lnfugacities = lnfugacities
        return lnfugacities

    fugacities_lowest_Gibbs = fugacities

    def lnphis_lowest_Gibbs(self):
        try:
            return self._lnphis_lowest_Gibbs
        except:
            pass
        P = self.P
        zs = self.zs
        fugacities_lowest_Gibbs = self.fugacities_lowest_Gibbs()
        if self.vectorized:
            lnphis_lowest_Gibbs = trunc_log_numpy(fugacities_lowest_Gibbs/(zs*P))
        else:
            lnphis_lowest_Gibbs = [trunc_log(fi/(zi*P)) for fi, zi in zip(fugacities_lowest_Gibbs, zs)]
        self._lnphis_lowest_Gibbs = lnphis_lowest_Gibbs
        return lnphis_lowest_Gibbs

    def dfugacities_dT(self):
        r'''Method to calculate and return the temperature derivative of fugacities
        of the phase.

        .. math::
            \frac{\partial f_i}{\partial T} = P z_i \frac{\partial
            \ln \phi_i}{\partial T}

        Returns
        -------
        dfugacities_dT : list[float]
            Temperature derivative of fugacities of all components
            in the phase, [Pa/K]

        Notes
        -----
        '''
        try:
            return self._dfugacities_dT
        except:
            pass
        dphis_dT = self.dphis_dT()
        P, zs = self.P, self.zs
        if self.vectorized:
            self._dfugacities_dT = P*zs*dphis_dT
        else:
            self._dfugacities_dT = [P*zs[i]*dphis_dT[i] for i in range(self.N)]
        return self._dfugacities_dT

    def lnphis_G_min(self):
        r'''Method to calculate and return the log fugacity coefficients of the
        phase. If the phase can have multiple solutions at its `T` and `P`,
        this method should return those with the lowest Gibbs energy. This
        needs to be implemented on phases with that criteria like cubic EOSs.

        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
        '''
        return self.lnphis()

    def phis(self):
        r'''Method to calculate and return the fugacity coefficients of the
        phase.

        .. math::
            \phi_i = \exp (\ln \phi_i)

        Returns
        -------
        phis : list[float]
            Fugacity coefficients, [-]
        '''
        try:
            return self._phis
        except:
            pass
        if self.vectorized:
            self._phis = trunc_exp_numpy(self.lnphis())
        else:
            self._phis = [trunc_exp(i) for i in self.lnphis()]
        return self._phis

    def dphis_dT(self):
        r'''Method to calculate and return the temperature derivative of fugacity
        coefficients of the phase.

        .. math::
            \frac{\partial \phi_i}{\partial T} = \phi_i \frac{\partial
            \ln \phi_i}{\partial T}

        Returns
        -------
        dphis_dT : list[float]
            Temperature derivative of fugacity coefficients of all components
            in the phase, [1/K]

        Notes
        -----
        '''
        try:
            return self._dphis_dT
        except AttributeError:
            pass
        try:
            dlnphis_dT = self._dlnphis_dT
        except AttributeError:
            dlnphis_dT = self.dlnphis_dT()

        try:
            phis = self._phis
        except AttributeError:
            phis = self.phis()
        if self.vectorized:
            self._dphis_dT = dlnphis_dT*phis
        else:
            self._dphis_dT = [dlnphis_dT[i]*phis[i] for i in range(self.N)]
        return self._dphis_dT

    def dphis_dP(self):
        r'''Method to calculate and return the pressure derivative of fugacity
        coefficients of the phase.

        .. math::
            \frac{\partial \phi_i}{\partial P} = \phi_i \frac{\partial
            \ln \phi_i}{\partial P}

        Returns
        -------
        dphis_dP : list[float]
            Pressure derivative of fugacity coefficients of all components
            in the phase, [1/Pa]

        Notes
        -----
        '''
        try:
            return self._dphis_dP
        except AttributeError:
            pass
        try:
            dlnphis_dP = self._dlnphis_dP
        except AttributeError:
            dlnphis_dP = self.dlnphis_dP()

        try:
            phis = self._phis
        except AttributeError:
            phis = self.phis()
        if self.vectorized:
            self._dphis_dP = dlnphis_dP*phis
        else:
            self._dphis_dP = [dlnphis_dP[i]*phis[i] for i in range(self.N)]

        return self._dphis_dP

    def dphis_dzs(self):
        r'''Method to calculate and return the molar composition derivative of
        fugacity coefficients of the phase.

        .. math::
            \frac{\partial \phi_i}{\partial z_j} = \phi_i \frac{\partial
            \ln \phi_i}{\partial z_j}

        Returns
        -------
        dphis_dzs : list[list[float]]
            Molar derivative of fugacity coefficients of all components
            in the phase, [-]

        Notes
        -----
        '''
        try:
            return self._dphis_dzs
        except AttributeError:
            pass
        try:
            dlnphis_dzs = self._dlnphis_dzs
        except AttributeError:
            dlnphis_dzs = self.dlnphis_dzs()

        try:
            phis = self._phis
        except AttributeError:
            phis = self.phis()

        N = self.N
        self._dphis_dzs = [[dlnphis_dzs[i][j]*phis[i] for j in range(N)]
                           for i in range(N)]
        if self.vectorized:
            self._dphis_dzs = array(self._dphis_dzs)
        return self._dphis_dzs


    def dfugacities_dP(self):
        r'''Method to calculate and return the pressure derivative of the
        fugacities of the components in the phase.

        .. math::
            \frac{\partial f_i}{\partial P} = z_i \left(P \frac{\partial
            \phi_i}{\partial P}  + \phi_i \right)

        Returns
        -------
        dfugacities_dP : list[float]
            Pressure derivative of fugacities of all components
            in the phase, [-]

        Notes
        -----
        For models without pressure dependence of fugacity, the returned result
        may not be exactly zero due to inaccuracy in floating point results;
        results are likely on the order of 1e-14 or lower in that case.
        '''
        try:
            dphis_dP = self._dphis_dP
        except AttributeError:
            dphis_dP = self.dphis_dP()

        try:
            phis = self._phis
        except AttributeError:
            phis = self.phis()

        P, zs = self.P, self.zs
        if self.vectorized:
            return zs*(P*dphis_dP + phis)
        else:
            return [zs[i]*(P*dphis_dP[i] + phis[i]) for i in range(self.N)]

    def dfugacities_dns(self):
        r'''Method to calculate and return the mole number derivative of the
        fugacities of the components in the phase.

        if i != j:

        .. math::
            \frac{\partial f_i}{\partial n_j} = P\phi_i z_i \left(
            \frac{\partial \ln \phi_i}{\partial n_j} - 1
            \right)

        if i == j:

        .. math::
            \frac{\partial f_i}{\partial n_j} = P\phi_i z_i \left(
            \frac{\partial \ln \phi_i}{\partial n_j} - 1
            \right) + P\phi_i

        Returns
        -------
        dfugacities_dns : list[list[float]]
            Mole number derivatives of the fugacities of all components
            in the phase, [Pa/mol]

        Notes
        -----
        '''
        phis = self.phis()
        dlnphis_dns = self.dlnphis_dns()
        P, zs, N = self.P, self.zs, self.N
        matrix = zeros((N, N)) if self.vectorized else [[0.0]*N for _ in range(N)]
        for i in range(N):
            phi_P = P*phis[i]
            ziPphi = phi_P*zs[i]
            r = dlnphis_dns[i]
            for j in range(N):
                matrix[i][j] = ziPphi*(r[j] - 1.0)
            matrix[i][i] += phi_P
        return matrix

    def dlnfugacities_dns(self):
        r'''Method to calculate and return the mole number derivative of the
        log of fugacities of the components in the phase.

        .. math::
            \frac{\partial \ln f_i}{\partial n_j} = \frac{1}{f_i}
            \frac{\partial  f_i}{\partial n_j}

        Returns
        -------
        dlnfugacities_dns : list[list[float]]
            Mole number derivatives of the log of fugacities of all components
            in the phase, [log(Pa)/mol]

        Notes
        -----
        '''
        fugacities = self.fugacities()
        dlnfugacities_dns = [list(i) for i in self.dfugacities_dns()]
        if self.vectorized:
            dlnfugacities_dns = array(dlnfugacities_dns)
        fugacities_inv = [1.0/fi for fi in fugacities]
        cmps = range(self.N)
        for i in cmps:
            r = dlnfugacities_dns[i]
            for j in cmps:
                r[j]*= fugacities_inv[i]
        return dlnfugacities_dns

    def dlnfugacities_dzs(self):
        r'''Method to calculate and return the mole fraction derivative of the
        log of fugacities of the components in the phase.

        .. math::
            \frac{\partial \ln f_i}{\partial z_j} = \frac{1}{f_i}
            \frac{\partial  f_i}{\partial z_j}

        Returns
        -------
        dlnfugacities_dzs : list[list[float]]
            Mole fraction derivatives of the log of fugacities of all components
            in the phase, [log(Pa)]

        Notes
        -----
        '''
        fugacities = self.fugacities()
        dlnfugacities_dzs = [list(i) for i in self.dfugacities_dzs()]
        fugacities_inv = [1.0/fi for fi in fugacities]
        cmps = range(self.N)
        for i in cmps:
            r = dlnfugacities_dzs[i]
            for j in cmps:
                r[j]*= fugacities_inv[i]
        return dlnfugacities_dzs

    def log_zs(self):
        r'''Method to calculate and return the log of mole fractions specified.
        These are used in calculating entropy and in many other formulas.

        .. math::
            \ln z_i

        Returns
        -------
        log_zs : list[float]
            Log of mole fractions, [-]

        Notes
        -----
        '''
        try:
            return self._log_zs
        except AttributeError:
            pass
        if self.vectorized:
            self._log_zs = trunc_log_numpy(self.zs)
        else:
            self._log_zs = [trunc_log(zi) for zi in self.zs]
        # except ValueError:
        #     self._log_zs = _log_zs = []
        #     for zi in self.zs:
        #         try:
        #             _log_zs.append(log(zi))
        #         except ValueError:
        #             _log_zs.append(-690.7755278982137) # log(1e-300)
        return self._log_zs

    def V_iter(self, force=False):
        r'''Method to calculate and return the volume of the phase in a way
        suitable for a TV resolution to converge on the same pressure. This
        often means the return value of this method is an mpmath `mpf`.
        This dummy method simply returns the implemented V method.

        Returns
        -------
        V : float or mpf
            Molar volume, [m^3/mol]

        Notes
        -----
        '''
        return self.V()

    def G(self):
        r'''Method to calculate and return the Gibbs free energy of the phase.

        .. math::
            G = H - TS

        Returns
        -------
        G : float
            Gibbs free energy, [J/mol]

        Notes
        -----
        '''
        try:
            return self._G
        except AttributeError:
            pass
        G = self.H() - self.T*self.S()
        self._G = G
        return G

    G_min = G

    def U(self):
        r'''Method to calculate and return the internal energy of the phase.

        .. math::
            U = H - PV

        Returns
        -------
        U : float
            Internal energy, [J/mol]

        Notes
        -----
        '''
        U = self.H() - self.P*self.V()
        return U

    def A(self):
        r'''Method to calculate and return the Helmholtz energy of the phase.

        .. math::
            A = U - TS

        Returns
        -------
        A : float
            Helmholtz energy, [J/mol]

        Notes
        -----
        '''
        A = self.U() - self.T*self.S()
        return A

    def dH_dns(self):
        r'''Method to calculate and return the mole number derivative of the
        enthalpy of the phase.

        .. math::
            \frac{\partial H}{\partial n_i}

        Returns
        -------
        dH_dns : list[float]
            Mole number derivatives of the enthalpy of the phase,
            [J/mol^2]

        Notes
        -----
        '''
        out = zeros(self.N) if self.vectorized else [0.0]*self.N
        return dxs_to_dns(self.dH_dzs(), self.zs, out)

    def dS_dns(self):
        r'''Method to calculate and return the mole number derivative of the
        entropy of the phase.

        .. math::
            \frac{\partial S}{\partial n_i}

        Returns
        -------
        dS_dns : list[float]
            Mole number derivatives of the entropy of the phase,
            [J/(mol^2*K)]

        Notes
        -----
        '''
        out = zeros(self.N) if self.vectorized else [0.0]*self.N
        return dxs_to_dns(self.dS_dzs(), self.zs, out)

    def dG_dT(self):
        r'''Method to calculate and return the constant-pressure
        temperature derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial T}\right)_{P}
            = -T\left(\frac{\partial S}{\partial T}\right)_{P}
            - S + \left(\frac{\partial H}{\partial T}\right)_{P}

        Returns
        -------
        dG_dT : float
            Constant-pressure temperature derivative of Gibbs free energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return -self.T*self.dS_dT() - self.S() + self.dH_dT()

    dG_dT_P = dG_dT

    def dG_dT_V(self):
        r'''Method to calculate and return the constant-volume
        temperature derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial T}\right)_{V}
            = -T\left(\frac{\partial S}{\partial T}\right)_{V}
            - S + \left(\frac{\partial H}{\partial T}\right)_{V}

        Returns
        -------
        dG_dT_V : float
            Constant-volume temperature derivative of Gibbs free energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return -self.T*self.dS_dT_V() - self.S() + self.dH_dT_V()

    def dG_dP(self):
        r'''Method to calculate and return the constant-temperature
        pressure derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial P}\right)_{T}
            = -T\left(\frac{\partial S}{\partial P}\right)_{T}
            + \left(\frac{\partial H}{\partial P}\right)_{T}

        Returns
        -------
        dG_dP : float
            Constant-temperature pressure derivative of Gibbs free energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return -self.T*self.dS_dP() + self.dH_dP()

    dG_dP_T = dG_dP

    def dG_dP_V(self):
        r'''Method to calculate and return the constant-volume
        pressure derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial P}\right)_{V}
            = -T\left(\frac{\partial S}{\partial P}\right)_{V}
            - S \left(\frac{\partial T}{\partial P}\right)_{V}
            + \left(\frac{\partial H}{\partial P}\right)_{V}

        Returns
        -------
        dG_dP_V : float
            Constant-volume pressure derivative of Gibbs free energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return -self.T*self.dS_dP_V() - self.dT_dP()*self.S() + self.dH_dP_V()

    def dG_dV_T(self):
        r'''Method to calculate and return the constant-temperature
        volume derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial V}\right)_{T}
            = \left(\frac{\partial G}{\partial P}\right)_{T}
            \left(\frac{\partial P}{\partial V}\right)_{T}

        Returns
        -------
        dG_dV_T : float
            Constant-temperature volume derivative of Gibbs free energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dG_dP_T()*self.dP_dV()

    def dG_dV_P(self):
        r'''Method to calculate and return the constant-pressure
        volume derivative of Gibbs free energy.

        .. math::
            \left(\frac{\partial G}{\partial V}\right)_{P}
            = \left(\frac{\partial G}{\partial T}\right)_{P}
            \left(\frac{\partial T}{\partial V}\right)_{P}

        Returns
        -------
        dG_dV_P : float
            Constant-pressure volume derivative of Gibbs free energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dG_dT_P()*self.dT_dV()


    def dU_dT(self):
        r'''Method to calculate and return the constant-pressure
        temperature derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial T}\right)_{P}
            = -P \left(\frac{\partial V}{\partial T}\right)_{P}
             + \left(\frac{\partial H}{\partial T}\right)_{P}

        Returns
        -------
        dU_dT : float
            Constant-pressure temperature derivative of internal energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return -self.P*self.dV_dT() + self.dH_dT()

    dU_dT_P = dU_dT

    def dU_dT_V(self):
        r'''Method to calculate and return the constant-volume
        temperature derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial T}\right)_{V}
            = \left(\frac{\partial H}{\partial T}\right)_{V}
             - V \left(\frac{\partial P}{\partial T}\right)_{V}

        Returns
        -------
        dU_dT_V : float
            Constant-volume temperature derivative of internal energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return self.dH_dT_V() - self.V()*self.dP_dT()

    def dU_dP(self):
        r'''Method to calculate and return the constant-temperature
        pressure derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial P}\right)_{T}
            = -P \left(\frac{\partial V}{\partial P}\right)_{T}
             - V + \left(\frac{\partial H}{\partial P}\right)_{T}

        Returns
        -------
        dU_dP : float
            Constant-temperature pressure derivative of internal energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return -self.P*self.dV_dP() - self.V() + self.dH_dP()

    dU_dP_T = dU_dP

    def dU_dP_V(self):
        r'''Method to calculate and return the constant-volume
        pressure derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial P}\right)_{V}
            = \left(\frac{\partial H}{\partial P}\right)_{V}
             - V

        Returns
        -------
        dU_dP_V : float
            Constant-volume pressure derivative of internal energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return self.dH_dP_V() - self.V()

    def dU_dV_T(self):
        r'''Method to calculate and return the constant-temperature
        volume derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial V}\right)_{T}
            = \left(\frac{\partial U}{\partial P}\right)_{T}
             \left(\frac{\partial P}{\partial V}\right)_{T}

        Returns
        -------
        dU_dV_T : float
            Constant-temperature volume derivative of internal energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dU_dP_T()*self.dP_dV()

    def dU_dV_P(self):
        r'''Method to calculate and return the constant-pressure
        volume derivative of internal energy.

        .. math::
            \left(\frac{\partial U}{\partial V}\right)_{P}
            = \left(\frac{\partial U}{\partial T}\right)_{P}
             \left(\frac{\partial T}{\partial V}\right)_{P}

        Returns
        -------
        dU_dV_P : float
            Constant-pressure volume derivative of internal energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dU_dT_P()*self.dT_dV()

    def dA_dT(self):
        r'''Method to calculate and return the constant-pressure
        temperature derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial T}\right)_{P}
            = -T \left(\frac{\partial S}{\partial T}\right)_{P}
             - S + \left(\frac{\partial U}{\partial T}\right)_{P}

        Returns
        -------
        dA_dT : float
            Constant-pressure temperature derivative of Helmholtz energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return -self.T*self.dS_dT() - self.S() + self.dU_dT()

    dA_dT_P = dA_dT

    def dA_dT_V(self):
        r'''Method to calculate and return the constant-volume
        temperature derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial T}\right)_{V}
            =  \left(\frac{\partial H}{\partial T}\right)_{V}
             - V \left(\frac{\partial P}{\partial T}\right)_{V}
             - T \left(\frac{\partial S}{\partial T}\right)_{V}
             - S

        Returns
        -------
        dA_dT_V : float
            Constant-volume temperature derivative of Helmholtz energy,
            [J/(mol*K)]

        Notes
        -----
        '''
        return (self.dH_dT_V() - self.V()*self.dP_dT() - self.T*self.dS_dT_V()
                - self.S())

    def dA_dP(self):
        r'''Method to calculate and return the constant-temperature
        pressure derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial P}\right)_{T}
            = -T \left(\frac{\partial S}{\partial P}\right)_{T}
             + \left(\frac{\partial U}{\partial P}\right)_{T}

        Returns
        -------
        dA_dP : float
            Constant-temperature pressure derivative of Helmholtz energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return -self.T*self.dS_dP() + self.dU_dP()

    dA_dP_T = dA_dP

    def dA_dP_V(self):
        r'''Method to calculate and return the constant-volume
        pressure derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial P}\right)_{V}
            = \left(\frac{\partial H}{\partial P}\right)_{V}
            - V - S\left(\frac{\partial T}{\partial P}\right)_{V}
            -T \left(\frac{\partial S}{\partial P}\right)_{V}

        Returns
        -------
        dA_dP_V : float
            Constant-volume pressure derivative of Helmholtz energy,
            [J/(mol*Pa)]

        Notes
        -----
        '''
        return (self.dH_dP_V() - self.V() - self.dT_dP()*self.S()
                - self.T*self.dS_dP_V())

    def dA_dV_T(self):
        r'''Method to calculate and return the constant-temperature
        volume derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial V}\right)_{T}
            = \left(\frac{\partial A}{\partial P}\right)_{T}
              \left(\frac{\partial P}{\partial V}\right)_{T}

        Returns
        -------
        dA_dV_T : float
            Constant-temperature volume derivative of Helmholtz energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dA_dP_T()*self.dP_dV()

    def dA_dV_P(self):
        r'''Method to calculate and return the constant-pressure
        volume derivative of Helmholtz energy.

        .. math::
            \left(\frac{\partial A}{\partial V}\right)_{P}
            = \left(\frac{\partial A}{\partial T}\right)_{P}
              \left(\frac{\partial T}{\partial V}\right)_{P}

        Returns
        -------
        dA_dV_P : float
            Constant-pressure volume derivative of Helmholtz energy,
            [J/(m^3)]

        Notes
        -----
        '''
        return self.dA_dT_P()*self.dT_dV()

    def G_dep(self):
        r'''Method to calculate and return the departure Gibbs free energy of
        the phase.

        .. math::
            G_{dep} = H_{dep} - TS_{dep}

        Returns
        -------
        G_dep : float
            Departure Gibbs free energy, [J/mol]

        Notes
        -----
        '''
        G_dep = self.H_dep() - self.T*self.S_dep()
        return G_dep

    def V_dep(self):
        r'''Method to calculate and return the departure (from ideal gas
        behavior) molar volume of the phase.

        .. math::
            V_{dep} = V - \frac{RT}{P}

        Returns
        -------
        V_dep : float
            Departure molar volume, [m^3/mol]

        Notes
        -----
        '''
        V_dep = self.V() - self.R*self.T/self.P
        return V_dep

    def U_dep(self):
        r'''Method to calculate and return the departure internal energy of
        the phase.

        .. math::
            U_{dep} = H_{dep} - PV_{dep}

        Returns
        -------
        U_dep : float
            Departure internal energy, [J/mol]

        Notes
        -----
        '''
        return self.H_dep() - self.P*self.V_dep()

    def A_dep(self):
        r'''Method to calculate and return the departure Helmholtz energy of
        the phase.

        .. math::
            A_{dep} = U_{dep} - TS_{dep}

        Returns
        -------
        A_dep : float
            Departure Helmholtz energy, [J/mol]

        Notes
        -----
        '''
        return self.U_dep() - self.T*self.S_dep()

    def H_dep_mass(self):
        r'''Method to calculate and return the mass departure enthalpy of
        the phase.

        Returns
        -------
        H_dep_mass : float
            Departure mass enthalpy free energy, [J/kg]

        Notes
        -----
        '''
        return property_molar_to_mass(self.H_dep(), self.MW())

    def S_dep_mass(self):
        r'''Method to calculate and return the mass departure entropy of
        the phase.

        Returns
        -------
        S_dep_mass : float
            Departure mass entropy free energy, [J/(kg*K)]

        Notes
        -----
        '''
        return property_molar_to_mass(self.S_dep(), self.MW())

    def G_dep_mass(self):
        r'''Method to calculate and return the mass departure Gibbs free energy of
        the phase.

        Returns
        -------
        G_dep_mass : float
            Departure mass Gibbs free energy, [J/kg]

        Notes
        -----
        '''
        return property_molar_to_mass(self.G_dep(), self.MW())

    def U_dep_mass(self):
        r'''Method to calculate and return the departure mass internal energy of
        the phase.

        Returns
        -------
        U_dep_mass : float
            Departure mass internal energy, [J/kg]

        Notes
        -----
        '''
        return property_molar_to_mass(self.U_dep(), self.MW())

    def A_dep_mass(self):
        r'''Method to calculate and return the departure mass Helmholtz energy of
        the phase.

        Returns
        -------
        A_dep_mass : float
            Departure mass Helmholtz energy, [J/kg]

        Notes
        -----
        '''
        return property_molar_to_mass(self.A_dep(), self.MW())

    def H_reactive(self):
        r'''Method to calculate and return the enthalpy of the phase on a
        reactive basis, using the `Hfs` values of the phase.

        .. math::
            H_{reactive} = H + \sum_i z_i {H_{f,i}}

        Returns
        -------
        H_reactive : float
            Enthalpy of the phase on a reactive basis, [J/mol]

        Notes
        -----
        '''
        try:
            return self._H_reactive
        except AttributeError:
            pass
        H = self.H()
        if self.vectorized:
            H += float(dot(self.zs, self.Hfs))
        else:
            for zi, Hf in zip(self.zs, self.Hfs):
                H += zi*Hf
        self._H_reactive = H
        return H

    def S_reactive(self):
        r'''Method to calculate and return the entropy of the phase on a
        reactive basis, using the `Sfs` values of the phase.

        .. math::
            S_{reactive} = S + \sum_i z_i {S_{f,i}}

        Returns
        -------
        S_reactive : float
            Entropy of the phase on a reactive basis, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._S_reactive
        except:
            pass
        S = self.S()
        if self.vectorized:
            S += float(dot(self.zs, self.Sfs))
        else:
            for zi, Sf in zip(self.zs, self.Sfs):
                S += zi*Sf
        self._S_reactive = S
        return S

    def G_reactive(self):
        r'''Method to calculate and return the Gibbs free energy of the phase
        on a reactive basis.

        .. math::
            G_{reactive} = H_{reactive} - TS_{reactive}

        Returns
        -------
        G_reactive : float
            Gibbs free energy of the phase on a reactive basis, [J/(mol)]

        Notes
        -----
        '''
        G = self.H_reactive() - self.T*self.S_reactive()
        return G

    def U_reactive(self):
        r'''Method to calculate and return the internal energy of the phase
        on a reactive basis.

        .. math::
            U_{reactive} = H_{reactive} - PV

        Returns
        -------
        U_reactive : float
            Internal energy of the phase on a reactive basis, [J/(mol)]

        Notes
        -----
        '''
        U = self.H_reactive() - self.P*self.V()
        return U

    def A_reactive(self):
        r'''Method to calculate and return the Helmholtz free energy of the
        phase on a reactive basis.

        .. math::
            A_{reactive} = U_{reactive} - TS_{reactive}

        Returns
        -------
        A_reactive : float
            Helmholtz free energy of the phase on a reactive basis, [J/(mol)]

        Notes
        -----
        '''
        A = self.U_reactive() - self.T*self.S_reactive()
        return A

    def H_formation_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas enthalpy of formation
        of the phase (as if the phase was an ideal gas).

        .. math::
            H_{formation}^{ig} = \sum_i z_i {H_{f,i}}

        Returns
        -------
        H_formation_ideal_gas : float
            Enthalpy of formation of the phase on a formation basis
            as an ideal gas, [J/mol]

        Notes
        -----
        '''
        try:
            return self._H_formation_ideal_gas
        except AttributeError:
            pass
        Hf_ideal_gas = 0.0
        if self.vectorized:
            Hf_ideal_gas = float(self.zs, self.Hfs)
        else:
            for zi, Hf in zip(self.zs, self.Hfs):
                Hf_ideal_gas += zi*Hf
        self._H_formation_ideal_gas = Hf_ideal_gas
        return Hf_ideal_gas

    def S_formation_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas entropy of formation
        of the phase (as if the phase was an ideal gas).

        .. math::
            S_{formation}^{ig} = \sum_i z_i {S_{f,i}}

        Returns
        -------
        S_formation_ideal_gas : float
            Entropy of formation of the phase on a formation basis
            as an ideal gas, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._S_formation_ideal_gas
        except:
            pass
        Sf_ideal_gas = 0.0
        if self.vectorized:
            Sf_ideal_gas = float(dot(self.zs, self.Sfs))
        else:
            for zi, Sf in zip(self.zs, self.Sfs):
                Sf_ideal_gas += zi*Sf
        self._S_formation_ideal_gas = Sf_ideal_gas
        return Sf_ideal_gas

    def G_formation_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas Gibbs free energy of
        formation of the phase (as if the phase was an ideal gas).

        .. math::
            G_{formation}^{ig} = H_{formation}^{ig} - T_{ref}^{ig}
            S_{formation}^{ig}

        Returns
        -------
        G_formation_ideal_gas : float
            Gibbs free energy of formation of the phase on a formation basis
            as an ideal gas, [J/(mol)]

        Notes
        -----
        '''
        Gf = self.H_formation_ideal_gas() - self.T_REF_IG*self.S_formation_ideal_gas()
        return Gf

    def U_formation_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas internal energy of
        formation of the phase (as if the phase was an ideal gas).

        .. math::
            U_{formation}^{ig} = H_{formation}^{ig} - P_{ref}^{ig}
            V^{ig}

        Returns
        -------
        U_formation_ideal_gas : float
            Internal energy of formation of the phase on a formation basis
            as an ideal gas, [J/(mol)]

        Notes
        -----
        '''
        Uf = self.H_formation_ideal_gas() - self.P_REF_IG*self.V_ideal_gas()
        return Uf

    def A_formation_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas Helmholtz energy of
        formation of the phase (as if the phase was an ideal gas).

        .. math::
            A_{formation}^{ig} = U_{formation}^{ig} - T_{ref}^{ig}
            S_{formation}^{ig}

        Returns
        -------
        A_formation_ideal_gas : float
            Helmholtz energy of formation of the phase on a formation basis
            as an ideal gas, [J/(mol)]

        Notes
        -----
        '''
        Af = self.U_formation_ideal_gas() - self.T_REF_IG*self.S_formation_ideal_gas()
        return Af

    def H_formation_ideal_gas_mass(self):
        r'''Method to calculate and return the mass ideal-gas formation enthalpy of
        the phase.

        Returns
        -------
        H_formation_ideal_gas_mass  : float
            Formation mass enthalpy, [J/kg]

        Notes
        -----
        '''
        return property_molar_to_mass(self.H_formation_ideal_gas(), self.MW())

    def S_formation_ideal_gas_mass(self):
        r'''Method to calculate and return the mass ideal-gas formation entropy of
        the phase.

        Returns
        -------
        S_formation_ideal_gas_mass  : float
            Formation mass entropy, [J/(kg*K)]

        Notes
        -----
        '''
        return property_molar_to_mass(self.S_formation_ideal_gas(), self.MW())

    def G_formation_ideal_gas_mass(self):
        r'''Method to calculate and return the mass ideal-gas formation Gibbs free energy of
        the phase.

        Returns
        -------
        G_formation_ideal_gas_mass  : float
            Formation mass Gibbs free energy, [J/kg]

        Notes
        -----
        '''
        return property_molar_to_mass(self.G_formation_ideal_gas(), self.MW())

    def U_formation_ideal_gas_mass(self):
        r'''Method to calculate and return the ideal-gas formation mass internal energy of
        the phase.

        Returns
        -------
        U_formation_ideal_gas_mass  : float
            Formation mass internal energy, [J/kg]

        Notes
        -----
        '''
        return property_molar_to_mass(self.U_formation_ideal_gas(), self.MW())

    def A_formation_ideal_gas_mass(self):
        r'''Method to calculate and return the ideal-gas formation mass Helmholtz energy of
        the phase.

        Returns
        -------
        A_formation_ideal_gas_mass  : float
            Formation mass Helmholtz energy, [J/kg]

        Notes
        -----
        '''
        return property_molar_to_mass(self.A_formation_ideal_gas(), self.MW())

    def Cv(self):
        r'''Method to calculate and return the constant-volume heat
        capacity `Cv` of the phase.

        .. math::
            C_v = T\left(\frac{\partial P}{\partial T}\right)_V^2/
            \left(\frac{\partial P}{\partial V}\right)_T + Cp

        Returns
        -------
        Cv : float
            Constant volume molar heat capacity, [J/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._Cv
        except AttributeError:
            pass
        # checks out

        Cp_m_Cv = Cp_minus_Cv(self.T, self.dP_dT(), self.dP_dV())
        Cp = self.Cp()
        self._Cv = Cv = Cp - Cp_m_Cv
        return Cv

    def dCv_dT_P(self):
        r'''Method to calculate the temperature derivative of Cv, constant
        volume heat capacity, at constant pressure.

        .. math::
            \left(\frac{\partial C_v}{\partial T}\right)_P =
            - \frac{T \operatorname{dPdT_{V}}^{2}{\left(T \right)} \frac{d}{dT}
            \operatorname{dPdV_{T}}{\left(T \right)}}{\operatorname{dPdV_{T}}^{2}
            {\left(T \right)}} + \frac{2 T \operatorname{dPdT_{V}}{\left(T \right)}
            \frac{d}{d T} \operatorname{dPdT_{V}}{\left(T \right)}}
            {\operatorname{dPdV_{T}}{\left(T \right)}} + \frac{\operatorname{
            dPdT_{V}}^{2}{\left(T \right)}}{\operatorname{dPdV_{T}}{\left(T
            \right)}} + \frac{d}{d T} \operatorname{Cp}{\left(T \right)}

        Returns
        -------
        dCv_dT_P : float
           Temperature derivative of constant volume heat capacity at constant
           pressure, [J/mol/K^2]

        Notes
        -----
        Requires `d2P_dT2_PV`, `d2P_dVdT_TP`, and `d2H_dT2`.
        '''
        T = self.T
        x0 = self.dP_dT_V()
        x1 = x0*x0
        x2 = self.dP_dV_T()
        x3 = 1.0/x2

        x50 = self.d2P_dT2_PV()
        x51 = self.d2P_dVdT_TP()
        x52 = self.d2H_dT2()
        return 2.0*T*x0*x3*x50 - T*x1*x51*x3*x3 + x1*x3 + x52

    def dCv_dP_T(self):
        r'''Method to calculate the pressure derivative of Cv, constant
        volume heat capacity, at constant temperature.

        .. math::
            \left(\frac{\partial C_v}{\partial P}\right)_T =
            - T \operatorname{dPdT_{V}}{\left(P \right)} \frac{d}{d P}
            \operatorname{dVdT_{P}}{\left(P \right)} - T \operatorname{
            dVdT_{P}}{\left(P \right)} \frac{d}{d P} \operatorname{dPdT_{V}}
            {\left(P \right)} + \frac{d}{d P} \operatorname{Cp}{\left(P\right)}

        Returns
        -------
        dCv_dP_T : float
           Pressure derivative of constant volume heat capacity at constant
           temperature, [J/mol/K/Pa]

        Notes
        -----
        Requires `d2V_dTdP`, `d2P_dTdP`, and `d2H_dTdP`.
        '''
        T = self.T
        dP_dT_V = self.dP_dT_V()
        d2V_dTdP = self.d2V_dTdP()
        dV_dT_P = self.dV_dT_P()
        d2P_dTdP = self.d2P_dTdP()
        d2H_dep_dTdP = self.d2H_dTdP()
        return -T*dP_dT_V*d2V_dTdP - T*dV_dT_P*d2P_dTdP + d2H_dep_dTdP

    def chemical_potential(self):
        r'''Method to calculate and return the chemical potentials of each
        component in the phase [-]. For a pure substance, this is the
        molar Gibbs energy on a reactive basis.

        .. math::
            \frac{\partial G}{\partial n_i}_{T, P, N_{j \ne i}}

        Returns
        -------
        chemical_potential : list[float]
            Chemical potentials, [J/mol]
        '''
        try:
            return self._chemical_potentials
        except AttributeError:
            pass
        dS_dzs = self.dS_dzs()
        dH_dzs = self.dH_dzs()
        T, Hfs, Sfs = self.T, self.Hfs, self.Sfs
        if self.vectorized:
            dG_reactive_dzs = Hfs - T*(Sfs + dS_dzs) + dH_dzs
        else:
            dG_reactive_dzs = [Hfs[i] - T*(Sfs[i] + dS_dzs[i]) + dH_dzs[i] for i in range(self.N)]
        dG_reactive_dns = zeros(self.N) if self.vectorized else [0.0]*self.N
        dG_reactive_dns = dxs_to_dns(dG_reactive_dzs, self.zs, dG_reactive_dns)
        chemical_potentials = dns_to_dn_partials(dG_reactive_dns, self.G_reactive())
        self._chemical_potentials = chemical_potentials
        return chemical_potentials
#        # CORRECT DO NOT CHANGE
#        # TODO analytical implementation
#        def to_diff(ns):
#            tot = sum(ns)
#            zs = normalize(ns)
#            return tot*self.to_TP_zs(self.T, self.P, zs).G_reactive()
#        return jacobian(to_diff, self.zs)

    def activities(self):
        r'''Method to calculate and return the activities of each component
        in the phase [-].

        .. math::
            a_i(T, P, x; f_i^0) = \frac{f_i(T, P, x)}{f_i^0(T, P_i^0)}

        Returns
        -------
        activities : list[float]
            Activities, [-]
        '''
        # For a good discussion, see
        # Thermodynamics: Fundamentals for Applications, J. P. O'Connell, J. M. Haile
        # 5.4 DEVIATIONS FROM IDEAL SOLUTIONS: RATIO MEASURES page 201
        # CORRECT DO NOT CHANGE
        fugacities = self.fugacities()
        fugacities_std = self.fugacities_std() # TODO implement fugacities_std
        if self.vectorized:
            return fugacities/fugacities_std
        return [fugacities[i]/fugacities_std[i] for i in range(self.N)]

    def gammas(self):
        r'''Method to calculate and return the activity coefficients of the
        phase, [-].

        Activity coefficients are defined as the ratio of
        the actual fugacity coefficients times the pressure to the reference
        pure fugacity coefficients times the reference pressure.
        The reference pressure can be set to the actual pressure (the Lewis
        Randall standard state) which makes the pressures cancel.

        .. math::
            \gamma_i(T, P, x; f_i^0(T, P_i^0)) =
            \frac{\phi_i(T, P, x)P}{\phi_i^0(T, P_i^0) P_i^0}

        Returns
        -------
        gammas : list[float]
            Activity coefficients, [-]
        '''
        try:
            return self._gammas
        except:
            pass
        # For a good discussion, see
        # Thermodynamics: Fundamentals for Applications, J. P. O'Connell, J. M. Haile
        # 5.5 ACTIVITY COEFFICIENTS FROM FUGACITY COEFFICIENTS
        # There is no one single definition for gamma but it is believed this is
        # the most generally used one for EOSs; and activity methods
        # override this
        phis = self.phis()
        T, P, N, vectorized = self.T, self.P, self.N, self.vectorized
        self._gammas = gammas = zeros(N) if vectorized else [0.0]*N
        for i in range(N):
            comp = zeros(N) if vectorized else [0.0]*N
            comp[i] = 1.0
            phi = self.to_TP_zs(T=T, P=P, zs=comp).phis()[i]
            gammas[i] = phis[i]/phi

        self._gammas = gammas
        return gammas

    _x_infinite_dilution = 0.0

    def gammas_infinite_dilution(self):
        r'''Calculate and return the infinite dilution activity coefficients
        of each component.

        Returns
        -------
        gammas_infinite : list[float]
            Infinite dilution activity coefficients, [-]

        Notes
        -----
        The algorithm is as follows. For each component, set its composition to
        zero. Normalize the remaining compositions to 1. Create a new object
        with that composition, and calculate the activity coefficient of the
        component whose concentration was set to zero.
        '''
        T, P, N = self.T, self.P, self.N
        zs_base = self.zs
        x_infinite_dilution = self._x_infinite_dilution
        # x_infinite_dilution = 1e-7
        if self.vectorized:
            gammas_inf = zeros(N)
            copy_fun = array
        else:
            gammas_inf = [0.0]*N
            copy_fun = list
        phis = self.phis()
        for i in range(N):
            zs = copy_fun(zs_base)
            zs[i] = x_infinite_dilution
            zs = normalize(zs)
            # phi = self.to_TP_zs(T=T, P=P, zs=zs).phis()[i]
            # gammas_inf[i] = phis[i]/phi
            # No need to double-count
            gammas_inf[i] = self.to_TP_zs(T=T, P=P, zs=zs).gammas()[i]
        return gammas_inf

    def Cp_Cv_ratio(self):
        r'''Method to calculate and return the Cp/Cv ratio of the phase.

        .. math::
            \frac{C_p}{C_v}

        Returns
        -------
        Cp_Cv_ratio : float
            Cp/Cv ratio, [-]

        Notes
        -----
        '''
        return self.Cp()/self.Cv()


    def isentropic_exponent_PV(self):
        r'''Method to calculate and return the real gas isentropic exponent
        of the phase, which satisfies the relationship
        :math:`PV^k = \text{const}`.

        .. math::
            k = -\frac{V}{P}\frac{C_p}{C_v}\left(\frac{\partial P}{\partial V}\right)_T

        Returns
        -------
        k_PV : float
            Isentropic exponent of a real fluid, [-]

        Notes
        -----
        '''
        return isentropic_exponent_PV(Cp=self.Cp(), Cv=self.Cv(), Vm=self.V(), P=self.P, dP_dV_T=self.dP_dV_T())

    isentropic_exponent = isentropic_exponent_PV

    def isentropic_exponent_PT(self):
        r'''Method to calculate and return the real gas isentropic exponent
        of the phase, which satisfies the relationship
        :math:`P^{(1-k)}T^k = \text{const}`.

        .. math::
            k = \frac{1}{1 - \frac{P}{C_p}\left(\frac{\partial V}{\partial T}\right)_P}

        Returns
        -------
        k_PT : float
            Isentropic exponent of a real fluid, [-]

        Notes
        -----
        '''
        return isentropic_exponent_PT(Cp=self.Cp(), P=self.P, dV_dT_P=self.dV_dT_P())

    def isentropic_exponent_TV(self):
        r'''Method to calculate and return the real gas isentropic exponent
        of the phase, which satisfies the relationship
        :math:`TV^{k-1} = \text{const}`.

        .. math::
            k = 1 + \frac{V}{C_v} \left(\frac{\partial P}{\partial T}\right)_V

        Returns
        -------
        k_TV : float
            Isentropic exponent of a real fluid, [-]

        Notes
        -----
        '''
        return isentropic_exponent_TV(Cv=self.Cv(), Vm=self.V(), dP_dT_V=self.dP_dT_V())


    def Prandtl(self):
        r'''Method to calculate and return the Prandtl number of the phase

        .. math::
            Pr = \frac{C_p \mu}{k} = \frac{\nu}{\alpha} = \frac{C_p \rho \nu}{k}

        Returns
        -------
        Pr : float
            Prandtl number []

        Notes
        -----
        '''
        return self.Cp_mass()*self.mu()/self.k()


    def Z(self):
        r'''Method to calculate and return the compressibility factor of the
        phase.

        .. math::
            Z = \frac{PV}{RT}

        Returns
        -------
        Z : float
            Compressibility factor, [-]

        Notes
        -----
        '''
        return self.P*self.V()/(self.R*self.T)

    def rho(self):
        r'''Method to calculate and return the molar density of the
        phase.

        .. math::
            \rho = frac{1}{V}

        Returns
        -------
        rho : float
            Molar density, [mol/m^3]

        Notes
        -----
        '''
        return 1.0/self.V()

    def dT_dP(self):
        r'''Method to calculate and return the constant-volume pressure
        derivative of temperature of the phase.

        .. math::
            \left(\frac{\partial T}{\partial P}\right)_V = \frac{1}{\left(\frac{
            \partial P}{\partial T}\right)_V}

        Returns
        -------
        dT_dP : float
            Constant-volume pressure derivative of temperature, [K/Pa]

        Notes
        -----
        '''
        return 1.0/self.dP_dT()

    def dV_dT(self):
        r'''Method to calculate and return the constant-pressure temperature
        derivative of volume of the phase.

        .. math::
            \left(\frac{\partial V}{\partial T}\right)_P =
            \frac{-\left(\frac{\partial P}{\partial T}\right)_V}
            {\left(\frac{\partial P}{\partial V}\right)_T}

        Returns
        -------
        dV_dT : float
            Constant-pressure temperature derivative of volume, [m^3/(mol*K)]

        Notes
        -----
        '''
        try:
            return self._dV_dT
        except AttributeError:
            pass
        dV_dT = self._dV_dT = -self.dP_dT()/self.dP_dV()
        return dV_dT

    def dV_dP(self):
        r'''Method to calculate and return the constant-temperature pressure
        derivative of volume of the phase.

        .. math::
            \left(\frac{\partial V}{\partial P}\right)_T =
            {-\left(\frac{\partial V}{\partial T}\right)_P}
            {\left(\frac{\partial T}{\partial P}\right)_V}

        Returns
        -------
        dV_dP : float
            Constant-temperature pressure derivative of volume, [m^3/(mol*Pa)]

        Notes
        -----
        '''
        return -self.dV_dT()*self.dT_dP()

    def dT_dV(self):
        r'''Method to calculate and return the constant-pressure volume
        derivative of temperature of the phase.

        .. math::
            \left(\frac{\partial T}{\partial V}\right)_P =
            \frac{1}
            {\left(\frac{\partial V}{\partial T}\right)_P}

        Returns
        -------
        dT_dV : float
            Constant-pressure volume derivative of temperature, [K*m^3/(m^3)]

        Notes
        -----
        '''
        return 1./self.dV_dT()

    def d2V_dP2(self):
        r'''Method to calculate and return the constant-temperature pressure
        derivative of volume of the phase.

        .. math::
            \left(\frac{\partial^2 V}{\partial P^2}\right)_T =
            -\frac{\left(\frac{\partial^2 P}{\partial V^2}\right)_T}
            {\left(\frac{\partial P}{\partial V}\right)_T^3}

        Returns
        -------
        d2V_dP2 : float
            Constant-temperature pressure derivative of volume, [m^3/(mol*Pa^2)]

        Notes
        -----
        '''
        inverse_dP_dV = 1.0/self.dP_dV()
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV*inverse_dP_dV
        return -self.d2P_dV2()*inverse_dP_dV3

    def d2T_dP2(self):
        r'''Method to calculate and return the constant-volume second pressure
        derivative of temperature of the phase.

        .. math::
            \left(\frac{\partial^2 T}{\partial P^2}\right)_V =
            -\left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \left(\frac{\partial T}{\partial P}\right)_V^3

        Returns
        -------
        d2T_dP2 : float
            Constant-volume second pressure derivative of temperature, [K/Pa^2]

        Notes
        -----
        '''
        dT_dP = self.dT_dP()
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP
        return -self.d2P_dT2()*inverse_dP_dT3

    def d2T_dV2(self):
        r'''Method to calculate and return the constant-pressure second volume
        derivative of temperature of the phase.

        .. math::
            \left(\frac{\partial^2 T}{\partial V^2}\right)_P = -\left[
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) \right]
            \left(\frac{\partial P}{\partial T}\right)^{-2}_V
            + \left[\left(\frac{\partial^2 P}{\partial T\partial V}\right)
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V\right]
            \left(\frac{\partial P}{\partial T}\right)_V^{-3}
            \left(\frac{\partial P}{\partial V}\right)_T

        Returns
        -------
        d2T_dV2 : float
            Constant-pressure second volume derivative of temperature,
            [K*mol^2/m^6]

        Notes
        -----
        '''
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dTdV = self.d2P_dTdV()
        d2P_dT2 = self.d2P_dT2()
        dT_dP = self.dT_dP()
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP

        return (-(self.d2P_dV2()*dP_dT - dP_dV*d2P_dTdV)*inverse_dP_dT2
                   +(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3*dP_dV)

    def d2V_dT2(self):
        r'''Method to calculate and return the constant-pressure second
        temperature derivative of volume of the phase.

        .. math::
            \left(\frac{\partial^2 V}{\partial T^2}\right)_P = -\left[
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) \right]
            \left(\frac{\partial P}{\partial V}\right)^{-2}_T
            + \left[\left(\frac{\partial^2 P}{\partial T\partial V}\right)
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T\right]
            \left(\frac{\partial P}{\partial V}\right)_T^{-3}
            \left(\frac{\partial P}{\partial T}\right)_V

        Returns
        -------
        d2V_dT2 : float
            Constant-pressure second temperature derivative of volume,
            [m^3/(mol*K^2)]

        Notes
        -----
        '''
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dTdV = self.d2P_dTdV()
        d2P_dT2 = self.d2P_dT2()
        d2P_dV2 = self.d2P_dV2()

        inverse_dP_dV = 1.0/dP_dV
        inverse_dP_dV2 = inverse_dP_dV*inverse_dP_dV
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV2

        return  (-(d2P_dT2*dP_dV - dP_dT*d2P_dTdV)*inverse_dP_dV2
                   +(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*inverse_dP_dV3*dP_dT)

    def d2V_dPdT(self):
        r'''Method to calculate and return the derivative of pressure and then
        the derivative of temperature of volume of the phase.

        .. math::
            \left(\frac{\partial^2 V}{\partial T\partial P}\right) =
            - \left[\left(\frac{\partial^2 P}{\partial T \partial V}\right)
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            \right]\left(\frac{\partial P}{\partial V}\right)_T^{-3}

        Returns
        -------
        d2V_dPdT : float
            Derivative of pressure and then the derivative of temperature
            of volume, [m^3/(mol*K*Pa)]

        Notes
        -----
        '''
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dTdV = self.d2P_dTdV()
        d2P_dV2 = self.d2P_dV2()

        inverse_dP_dV = 1.0/dP_dV
        inverse_dP_dV2 = inverse_dP_dV*inverse_dP_dV
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV2

        return -(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*inverse_dP_dV3
    d2V_dTdP = d2V_dPdT

    def d2T_dPdV(self):
        r'''Method to calculate and return the derivative of pressure and then
        the derivative of volume of temperature of the phase.

        .. math::
           \left(\frac{\partial^2 T}{\partial P\partial V}\right) =
            - \left[\left(\frac{\partial^2 P}{\partial T \partial V}\right)
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \right]\left(\frac{\partial P}{\partial T}\right)_V^{-3}

        Returns
        -------
        d2T_dPdV : float
            Derivative of pressure and then the derivative of volume
            of temperature, [K*mol/(Pa*m^3)]

        Notes
        -----
        '''
        dT_dP = self.dT_dP()
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP

        d2P_dTdV = self.d2P_dTdV()
        dP_dT = self.dP_dT()
        dP_dV = self.dP_dV()
        d2P_dT2 = self.d2P_dT2()
        return -(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3

    d2T_dVdP = d2T_dPdV
    def d2P_dVdT(self):
        r'''Method to calculate and return the second derivative of
        pressure with respect to temperature and volume of the phase.
        This is an alias of `d2P_dTdV`.

        .. math::
             \frac{\partial^2 P}{\partial V \partial T}

        Returns
        -------
        d2P_dVdT : float
            Second volume derivative of pressure, [mol*Pa^2/(J*K)]
        '''
        return self.d2P_dTdV()

    def dZ_dzs(self):
        r'''Method to calculate and return the mole fraction derivatives of the
        compressibility factor `Z` of the phase.

        .. math::
            \frac{\partial Z}{\partial z_i}

        Returns
        -------
        dZ_dzs : list[float]
            Mole fraction derivatives of the compressibility factor of the
            phase, [-]

        Notes
        -----
        '''
        factor = self.P/(self.T*self.R)
        if self.vectorized:
            return factor*self.dV_dzs()
        return [dV*factor for dV in self.dV_dzs()]

    def dZ_dns(self):
        r'''Method to calculate and return the mole number derivatives of the
        compressibility factor `Z` of the phase.

        .. math::
            \frac{\partial Z}{\partial n_i}

        Returns
        -------
        dZ_dns : list[float]
            Mole number derivatives of the compressibility factor of the
            phase, [1/mol]

        Notes
        -----
        '''
        out = zeros(self.N) if self.vectorized else [0.0]*self.N
        return dxs_to_dns(self.dZ_dzs(), self.zs, out)

    def dV_dzs(self):
        r'''Method to calculate and return the mole fraction derivatives of the
        molar volume `V` of the phase.

        .. math::
            \frac{\partial V}{\partial z_i}

        Returns
        -------
        dV_dzs : list[float]
            Mole fraction derivatives of the molar volume of the phase, [m^3/mol]

        Notes
        -----
        '''
        raise NotImplementedError("Must be implemented by subphases")

    def dV_dns(self):
        r'''Method to calculate and return the mole number derivatives of the
        molar volume `V` of the phase.

        .. math::
            \frac{\partial V}{\partial n_i}

        Returns
        -------
        dV_dns : list[float]
            Mole number derivatives of the molar volume of the phase, [m^3/mol^2]

        Notes
        -----
        '''
        out = zeros(self.N) if self.vectorized else [0.0]*self.N
        return dxs_to_dns(self.dV_dzs(), self.zs, out)

    def dnV_dns(self):
        r'''Method to calculate and return the partial mole number derivatives
        of the molar volume `V` of the phase.

        .. math::
            \frac{\partial n V}{\partial n_i}

        Returns
        -------
        dnV_dns : list[float]
            Partial mole number derivatives of the molar volume of the phase,
            [m^3/mol]

        Notes
        -----
        '''
        out = zeros(self.N) if self.vectorized else [0.0]*self.N
        return dns_to_dn_partials(self.dV_dns(), self.V(), out)

    # Derived properties
    def PIP(self):
        r'''Method to calculate and return the phase identification parameter
        of the phase.

        .. math::
            \Pi = V \left[\frac{\frac{\partial^2 P}{\partial V \partial T}}
            {\frac{\partial P }{\partial T}}- \frac{\frac{\partial^2 P}{\partial
            V^2}}{\frac{\partial P}{\partial V}} \right]

        Returns
        -------
        PIP : float
            Phase identification parameter, [-]
        '''
        return phase_identification_parameter(self.V(), self.dP_dT(), self.dP_dV(),
                                              self.d2P_dV2(), self.d2P_dTdV())

    def kappa(self):
        r'''Method to calculate and return the isothermal compressibility
        of the phase.

        .. math::
            \kappa = -\frac{1}{V}\left(\frac{\partial V}{\partial P} \right)_T

        Returns
        -------
        kappa : float
            Isothermal coefficient of compressibility, [1/Pa]
        '''
        return -1.0/self.V()*self.dV_dP()
    isothermal_compressibility = kappa

    def dkappa_dT(self):
        r'''Method to calculate and return the temperature derivative of
        isothermal compressibility of the phase.

        .. math::
            \frac{\partial \kappa}{\partial T} = -\frac{ \left(\frac{\partial^2 V}{\partial P\partial T} \right)}{V}
            + \frac{\left(\frac{\partial V}{\partial P} \right)_T\left(\frac{\partial V}{\partial T} \right)_P}{V^2}

        Returns
        -------
        dkappa_dT : float
            First temperature derivative of isothermal coefficient of
            compressibility, [1/(Pa*K)]
        '''
        V, dV_dP, dV_dT, d2V_dTdP = self.V(), self.dV_dP(), self.dV_dT(), self.d2V_dTdP()
        return -d2V_dTdP/V + dV_dP*dV_dT/(V*V)

    disothermal_compressibility_dT = dkappa_dT

    def isothermal_bulk_modulus(self):
        r'''Method to calculate and return the isothermal bulk modulus
        of the phase.

        .. math::
            K_T = -V\left(\frac{\partial P}{\partial V} \right)_T

        Returns
        -------
        isothermal_bulk_modulus : float
            Isothermal bulk modulus, [Pa]
        '''
        return 1.0/self.kappa()

    def isobaric_expansion(self):
        r'''Method to calculate and return the isobatic expansion coefficient
        of the phase.

        .. math::
            \beta = \frac{1}{V}\left(\frac{\partial V}{\partial T} \right)_P

        Returns
        -------
        beta : float
            Isobaric coefficient of a thermal expansion, [1/K]
        '''
        return self.dV_dT()/self.V()

    def disobaric_expansion_dT(self):
        r'''Method to calculate and return the temperature derivative of
        isobatic expansion coefficient of the phase.

        .. math::
            \frac{\partial \beta}{\partial T} = \frac{1}{V}\left(
            \left(\frac{\partial^2 V}{\partial T^2} \right)_P
            - \left(\frac{\partial V}{\partial T} \right)_P^2/V
            \right)

        Returns
        -------
        dbeta_dT : float
            Temperature derivative of isobaric coefficient of a thermal
            expansion, [1/K^2]
        '''
        """
        from sympy import *
        T, P = symbols('T, P')
        V = symbols('V', cls=Function)
        expr = 1/V(T, P)*Derivative(V(T, P), T)
        diff(expr, T)
        Derivative(V(T, P), (T, 2))/V(T, P) - Derivative(V(T, P), T)**2/V(T, P)**2
        # Untested
        """
        V_inv = 1.0/self.V()
        dV_dT = self.dV_dT()
        return V_inv*(self.d2V_dT2() - dV_dT*dV_dT*V_inv)

    def disobaric_expansion_dP(self):
        r'''Method to calculate and return the pressure derivative of
        isobatic expansion coefficient of the phase.

        .. math::
            \frac{\partial \beta}{\partial P} = \frac{1}{V}\left(
            \left(\frac{\partial^2 V}{\partial T\partial P} \right)
            -\frac{ \left(\frac{\partial V}{\partial T} \right)_P
             \left(\frac{\partial V}{\partial P} \right)_T}{V}
            \right)

        Returns
        -------
        dbeta_dP : float
            Pressure derivative of isobaric coefficient of a thermal
            expansion, [1/(K*Pa)]
        '''
        """
        from sympy import *
        T, P = symbols('T, P')
        V = symbols('V', cls=Function)
        expr = 1/V(T, P)*Derivative(V(T, P), T)
        diff(expr, P)
        Derivative(V(T, P), P, T)/V(T, P) - Derivative(V(T, P), P)*Derivative(V(T, P), T)/V(T, P)**2

        """
        V_inv = 1.0/self.V()
        dV_dT = self.dV_dT()
        dV_dP = self.dV_dP()
        return V_inv*(self.d2V_dTdP() - dV_dT*dV_dP*V_inv)

    def Joule_Thomson(self):
        r'''Method to calculate and return the Joule-Thomson coefficient
        of the phase.

        .. math::
            \mu_{JT} = \left(\frac{\partial T}{\partial P}\right)_H = \frac{1}{C_p}
            \left[T \left(\frac{\partial V}{\partial T}\right)_P - V\right]
            = \frac{V}{C_p}\left(\beta T-1\right)

        Returns
        -------
        mu_JT : float
            Joule-Thomson coefficient [K/Pa]
        '''
        return Joule_Thomson(self.T, self.V(), self.Cp(), dV_dT=self.dV_dT(), beta=self.isobaric_expansion())

    def speed_of_sound(self):
        r'''Method to calculate and return the molar speed of sound
        of the phase.

        .. math::
            w = \left[-V^2 \left(\frac{\partial P}{\partial V}\right)_T \frac{C_p}
            {C_v}\right]^{1/2}

        A similar expression based on molar density is:

        .. math::
           w = \left[\left(\frac{\partial P}{\partial \rho}\right)_T \frac{C_p}
           {C_v}\right]^{1/2}

        Returns
        -------
        w : float
            Speed of sound for a real gas, [m*kg^0.5/(s*mol^0.5)]
        '''
        # Intentionally molar
        return speed_of_sound(self.V(), self.dP_dV(), self.Cp(), self.Cv())

    def speed_of_sound_ideal_gas_mass(self):
        r'''Method to calculate and return the mass speed of sound
        of an ideal gas phase at the current conditions.

        .. math::
            c = \sqrt{kR_{specific, ideal gas}T}

        Returns
        -------
        w : float
            Speed of sound for an ideal gas, [m/s]
        '''
        k = self.Cp_ideal_gas()/self.Cv_ideal_gas()
        return c_ideal_gas(self.T, k, self.MW())

    def speed_of_sound_ideal_gas(self):
        r'''Method to calculate and return the molar speed of sound
        of an ideal gas phase at the current conditions.


        .. math::
            w = \left[-V^2 \left(\frac{\partial P}{\partial V}\right)_T \frac{C_p}
            {C_v}\right]^{1/2}

        .. math::
            \left(\frac{\partial P}{\partial V}\right)_T = \frac{-P^2}{RT}

        Returns
        -------
        w : float
            Speed of sound for a real gas, [m*kg^0.5/(s*mol^0.5)]
        '''
        # Intentionally molar
        V = self.R*self.T/self.P
        dP_dV = -self.P*self.P/(self.R*self.T)
        return speed_of_sound(V, dP_dV, self.Cp_ideal_gas(), self.Cv_ideal_gas())

    ### Compressibility factor derivatives
    def dZ_dT(self):
        r'''Method to calculate and return the temperature derivative of
        compressibility of the phase.

        .. math::
            \frac{\partial Z}{\partial P} = P\frac{\left(\frac{\partial
            V}{\partial T}\right)_P - \frac{-V}{T}}{RT}

        Returns
        -------
        dZ_dT : float
            Temperature derivative of compressibility, [1/K]
        '''
        T_inv = 1.0/self.T
        return self.P*self.R_inv*T_inv*(self.dV_dT() - self.V()*T_inv)

    def dZ_dP(self):
        r'''Method to calculate and return the pressure derivative of
        compressibility of the phase.

        .. math::
            \frac{\partial Z}{\partial P} = \frac{V + P\left(\frac{\partial
            V}{\partial P}\right)_T}{RT}

        Returns
        -------
        dZ_dP : float
            Pressure derivative of compressibility, [1/Pa]
        '''
        return 1.0/(self.T*self.R)*(self.V() + self.P*self.dV_dP())

    def dZ_dV(self):
        r'''Method to calculate and return the volume derivative of
        compressibility of the phase.

        .. math::
            \frac{\partial Z}{\partial V} = \frac{P - \rho \left(\frac{\partial
            P}{\partial \rho}\right)_T}{RT}

        Returns
        -------
        dZ_dV : float
            Volume derivative of compressibility, [mol/(m^3)]
        '''
        return (self.P - self.rho()*self.dP_drho())/(self.R*self.T)
    # Could add more

    def Bvirial(self):
        r'''Method to calculate and return the `B` virial coefficient of the
        phase at its current conditions.

        Returns
        -------
        Bvirial : float
            Virial coefficient, [m^3/mol]

        Notes
        -----
        '''
        return B_from_Z(self.Z(), self.T, self.P)

    ### Derivatives in the molar density basis
    def dP_drho(self):
        r'''Method to calculate and return the molar density derivative of
        pressure of the phase.

        .. math::
            \frac{\partial P}{\partial \rho} = -V^2\left(\frac{\partial
            P}{\partial V}\right)_T

        Returns
        -------
        dP_drho : float
            Molar density derivative of pressure, [Pa*m^3/mol]
        '''
        V = self.V()
        return -V*V*self.dP_dV()

    def drho_dP(self):
        r'''Method to calculate and return the pressure derivative of
        molar density of the phase.

        .. math::
            \frac{\partial \rho}{\partial P} = -\frac{1}{V^2}\left(\frac{\partial
            V}{\partial P}\right)_T

        Returns
        -------
        drho_dP : float
            Pressure derivative of Molar density, [mol/(Pa*m^3)]
        '''
        V = self.V()
        return -self.dV_dP()/(V*V)

    def d2P_drho2(self):
        r'''Method to calculate and return the second molar density derivative
        of pressure of the phase.

        .. math::
            \frac{\partial^2 P}{\partial \rho^2} = -V^2\left(
            -V^2 \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            -2 V \left(\frac{\partial P}{\partial V}\right)_T
            \right)

        Returns
        -------
        d2P_drho2 : float
            Second molar density derivative of pressure, [Pa*m^6/mol^2]
        '''
        V = self.V()
        return V*V*V*(V*self.d2P_dV2() + 2.0*self.dP_dV())

    def d2rho_dP2(self):
        r'''Method to calculate and return the second pressure derivative
        of molar density of the phase.

        .. math::
            \frac{\partial^2 \rho}{\partial P^2} = -\frac{1}{V^2}
            \left(\frac{\partial^2 V}{\partial P^2}\right)_T
            + \frac{2}{V^3} \left(\frac{\partial V}{\partial P}\right)_T^2

        Returns
        -------
        d2rho_dP2 : float
            Second pressure derivative of molar density, [mol^2/(Pa*m^6)]
        '''
        V = self.V()
        return -self.d2V_dP2()/V**2 + 2*self.dV_dP()**2/V**3

    def dT_drho(self):
        r'''Method to calculate and return the molar density derivative of
        temperature of the phase.

        .. math::
            \frac{\partial T}{\partial \rho} = -V^2\left(\frac{\partial
            T}{\partial V}\right)_P

        Returns
        -------
        dT_drho : float
            Molar density derivative of temperature, [K*m^3/mol]
        '''
        V = self.V()
        return -V*V*self.dT_dV()

    def d2T_drho2(self):
        r'''Method to calculate and return the second molar density derivative
        of temperature of the phase.

        .. math::
            \frac{\partial^2 T}{\partial \rho^2} = -V^2\left(
            -V^2 \left(\frac{\partial^2 T}{\partial V^2}\right)_P
            -2 V \left(\frac{\partial T}{\partial V}\right)_P
            \right)

        Returns
        -------
        d2T_drho2 : float
            Second molar density derivative of temperature, [K*m^6/mol^2]
        '''
        V = self.V()
        return V*V*V*(V*self.d2T_dV2() + 2.0*self.dT_dV())

    def drho_dT(self):
        r'''Method to calculate and return the temperature derivative of
        molar density of the phase.

        .. math::
            \frac{\partial \rho}{\partial T} = -\frac{1}{V^2}\left(\frac{\partial
            V}{\partial T}\right)_P

        Returns
        -------
        drho_dT : float
            Temperature derivative of molar density, [mol/(K*m^3)]
        '''
        V = self.V()
        return -self.dV_dT()/(V*V)

    def d2rho_dT2(self):
        r'''Method to calculate and return the second temperature derivative
        of molar density of the phase.

        .. math::
            \frac{\partial^2 \rho}{\partial T^2} = -\frac{1}{V^2}
            \left(\frac{\partial^2 V}{\partial T^2}\right)_P
            + \frac{2}{V^3} \left(\frac{\partial V}{\partial T}\right)_T^2

        Returns
        -------
        d2rho_dT2 : float
            Second temperature derivative of molar density, [mol^2/(K*m^6)]
        '''
        d2V_dT2 = self.d2V_dT2()
        V = self.V()
        dV_dT = self.dV_dT()
        return -d2V_dT2/V**2 + 2*dV_dT**2/V**3

    def d2P_dTdrho(self):
        r'''Method to calculate and return the temperature derivative
        and then molar density derivative of the pressure of the phase.

        .. math::
            \frac{\partial^2 P}{\partial T \partial \rho} = -V^2
            \left(\frac{\partial^2 P}{\partial T \partial V}\right)

        Returns
        -------
        d2P_dTdrho : float
            Temperature derivative and then molar density derivative of the
            pressure, [Pa*m^3/(K*mol)]
        '''
        V = self.V()
        d2P_dTdV = self.d2P_dTdV()
        return -(V*V)*d2P_dTdV

    def d2T_dPdrho(self):
        r'''Method to calculate and return the pressure derivative
        and then molar density derivative of the temperature of the phase.

        .. math::
            \frac{\partial^2 T}{\partial P \partial \rho} = -V^2
            \left(\frac{\partial^2 T}{\partial P \partial V}\right)

        Returns
        -------
        d2T_dPdrho : float
            Pressure derivative and then molar density derivative of the
            temperature, [K*m^3/(Pa*mol)]
        '''
        V = self.V()
        d2T_dPdV = self.d2T_dPdV()
        return -(V*V)*d2T_dPdV

    def d2rho_dPdT(self):
        r'''Method to calculate and return the pressure derivative
        and then temperature derivative of the molar density of the phase.

        .. math::
            \frac{\partial^2 \rho}{\partial P \partial T} = -\frac{1}{V^2}
            \left(\frac{\partial^2 V}{\partial P \partial T}\right)
            + \frac{2}{V^3}  \left(\frac{\partial V}{\partial T}\right)_P
             \left(\frac{\partial V}{\partial P}\right)_T

        Returns
        -------
        d2rho_dPdT : float
            Pressure derivative and then temperature derivative of the
            molar density, [mol/(m^3*K*Pa)]
        '''
        d2V_dPdT = self.d2V_dPdT()
        dV_dT = self.dV_dT()
        dV_dP = self.dV_dP()
        V = self.V()
        return -d2V_dPdT/V**2 + 2*dV_dT*dV_dP/V**3

    def drho_dV_T(self):
        r'''Method to calculate and return the volume derivative of
        molar density of the phase.

        .. math::
            \frac{\partial \rho}{\partial V} = -\frac{1}{V^2}

        Returns
        -------
        drho_dV_T : float
            Molar density derivative of volume, [mol^2/m^6]
        '''
        V = self.V()
        return -1.0/(V*V)

    def drho_dT_V(self):
        r'''Method to calculate and return the temperature derivative of
        molar density of the phase at constant volume.

        .. math::
            \left(\frac{\partial \rho}{\partial T}\right)_V = 0

        Returns
        -------
        drho_dT_V : float
            Temperature derivative of molar density of the phase at constant
            volume, [mol/(m^3*K)]
        '''
        return 0.0

    # Idea gas heat capacity

    def _setup_Cpigs(self, HeatCapacityGases):
        Cpgs_data = None
        Cpgs_poly_fit = all(i.method == POLY_FIT for i in HeatCapacityGases) if HeatCapacityGases is not None else False
        if Cpgs_poly_fit:
            T_REF_IG = self.T_REF_IG
            Cpgs_data = [[i.poly_fit_Tmin for i in HeatCapacityGases],
                              [i.poly_fit_Tmin_slope for i in HeatCapacityGases],
                              [i.poly_fit_Tmin_value for i in HeatCapacityGases],
                              [i.poly_fit_Tmax for i in HeatCapacityGases],
                              [i.poly_fit_Tmax_slope for i in HeatCapacityGases],
                              [i.poly_fit_Tmax_value for i in HeatCapacityGases],
                              [i.poly_fit_log_coeff for i in HeatCapacityGases],
#                              [horner(i.poly_fit_int_coeffs, i.poly_fit_Tmin) for i in HeatCapacityGases],
                              [horner(i.poly_fit_int_coeffs, i.poly_fit_Tmin) - i.poly_fit_Tmin*(0.5*i.poly_fit_Tmin_slope*i.poly_fit_Tmin + i.poly_fit_Tmin_value - i.poly_fit_Tmin_slope*i.poly_fit_Tmin) for i in HeatCapacityGases],
#                              [horner(i.poly_fit_int_coeffs, i.poly_fit_Tmax) for i in HeatCapacityGases],
                              [horner(i.poly_fit_int_coeffs, i.poly_fit_Tmax) - horner(i.poly_fit_int_coeffs, i.poly_fit_Tmin) + i.poly_fit_Tmin*(0.5*i.poly_fit_Tmin_slope*i.poly_fit_Tmin + i.poly_fit_Tmin_value - i.poly_fit_Tmin_slope*i.poly_fit_Tmin) for i in HeatCapacityGases],
#                              [horner_log(i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmin) for i in HeatCapacityGases],
                              [horner_log(i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmin) -(i.poly_fit_Tmin_slope*i.poly_fit_Tmin + (i.poly_fit_Tmin_value - i.poly_fit_Tmin_slope*i.poly_fit_Tmin)*log(i.poly_fit_Tmin)) for i in HeatCapacityGases],
#                              [horner_log(i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmax) for i in HeatCapacityGases],
                              [(horner_log(i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmax)
                                - horner_log(i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmin)
                                + (i.poly_fit_Tmin_slope*i.poly_fit_Tmin + (i.poly_fit_Tmin_value - i.poly_fit_Tmin_slope*i.poly_fit_Tmin)*log(i.poly_fit_Tmin))
                                - (i.poly_fit_Tmax_value -i.poly_fit_Tmax*i.poly_fit_Tmax_slope)*log(i.poly_fit_Tmax)) for i in HeatCapacityGases],
                              [poly_fit_integral_value(T_REF_IG, i.poly_fit_int_coeffs, i.poly_fit_Tmin,
                                                       i.poly_fit_Tmax, i.poly_fit_Tmin_value,
                                                       i.poly_fit_Tmax_value, i.poly_fit_Tmin_slope,
                                                       i.poly_fit_Tmax_slope) for i in HeatCapacityGases],
                              [i.poly_fit_coeffs for i in HeatCapacityGases],
                              [i.poly_fit_int_coeffs for i in HeatCapacityGases],
                              [i.poly_fit_T_int_T_coeffs for i in HeatCapacityGases],
                              [poly_fit_integral_over_T_value(T_REF_IG, i.poly_fit_T_int_T_coeffs, i.poly_fit_log_coeff, i.poly_fit_Tmin,
                                                       i.poly_fit_Tmax, i.poly_fit_Tmin_value,
                                                       i.poly_fit_Tmax_value, i.poly_fit_Tmin_slope,
                                                       i.poly_fit_Tmax_slope) for i in HeatCapacityGases],

                              ]
        return (Cpgs_poly_fit, Cpgs_data)


    def _Cp_pure_fast(self, Cps_data):
        T, N = self.T, self.N
        Cps = zeros(N) if self.vectorized else [0.0]*N
        Tmins, Tmaxs, coeffs = Cps_data[0], Cps_data[3], Cps_data[12]
        Tmin_slopes = Cps_data[1]
        Tmin_values = Cps_data[2]
        Tmax_slopes = Cps_data[4]
        Tmax_values = Cps_data[5]

        for i in range(N):
            if T < Tmins[i]:
                Cp = (T -  Tmins[i])*Tmin_slopes[i] + Tmin_values[i]
            elif T > Tmaxs[i]:
                Cp = (T - Tmaxs[i])*Tmax_slopes[i] + Tmax_values[i]
            else:
                Cp = 0.0
                for c in coeffs[i]:
                    Cp = Cp*T + c
            Cps[i] = Cp
        return Cps

    def _dCp_dT_pure_fast(self, Cps_data):
        T, N = self.T, self.N
        dCps = zeros(N) if self.vectorized else [0.0]*N
        Tmins, Tmaxs, coeffs = Cps_data[0], Cps_data[3], Cps_data[12]
        Tmin_slopes = Cps_data[1]
        Tmax_slopes = Cps_data[4]
        for i in range(N):
            if T < Tmins[i]:
                dCp = Tmin_slopes[i]
            elif T > Tmaxs[i]:
                dCp = Tmax_slopes[i]
            else:
                Cp, dCp = 0.0, 0.0
                for c in coeffs[i]:
                    dCp = T*dCp + Cp
                    Cp = T*Cp + c
            dCps[i] = dCp
        return dCps

    def _Cp_integrals_pure_fast(self, Cps_data):
        T, N = self.T, self.N
        Cp_integrals_pure = zeros(N) if self.vectorized else [0.0]*N
        Tmins, Tmaxes, int_coeffs = Cps_data[0], Cps_data[3], Cps_data[13]
        for i in range(N):
            # If indeed everything is working here, need to optimize to decide what to store
            # Try to save lookups to avoid cache misses
            # Instead of storing horner Tmin and Tmax, store -:
            # tot(Tmin) - Cps_data[7][i]
            # and tot1 + tot for the high T
            # Should save quite a bit of lookups! est. .12 go to .09
#                Tmin = Tmins[i]
#                if T < Tmin:
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    H = T*(0.5*Cps_data[1][i]*T + x1)
#                elif (T <= Tmaxes[i]):
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    tot = Tmin*(0.5*Cps_data[1][i]*Tmin + x1)
#
#                    tot1 = 0.0
#                    for c in int_coeffs[i]:
#                        tot1 = tot1*T + c
#                    tot1 -= Cps_data[7][i]
##                    tot1 = horner(int_coeffs[i], T) - horner(int_coeffs[i], Tmin)
#                    H = tot + tot1
#                else:
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    tot = Tmin*(0.5*Cps_data[1][i]*Tmin + x1)
#
#                    tot1 = Cps_data[8][i] - Cps_data[7][i]
#
#                    x1 = Cps_data[5][i] - Cps_data[4][i]*Tmaxes[i]
#                    tot2 = T*(0.5*Cps_data[4][i]*T + x1) - Tmaxes[i]*(0.5*Cps_data[4][i]*Tmaxes[i] + x1)
#                    H = tot + tot1 + tot2



            # ATTEMPT AT FAST HERE (NOW WORKING)
            if T < Tmins[i]:
                x1 = Cps_data[2][i] - Cps_data[1][i]*Tmins[i]
                H = T*(0.5*Cps_data[1][i]*T + x1)
            elif (T <= Tmaxes[i]):
                H = 0.0
                for c in int_coeffs[i]:
                    H = H*T + c
                H -= Cps_data[7][i]
            else:
                Tmax_slope = Cps_data[4][i]
                x1 = Cps_data[5][i] - Tmax_slope*Tmaxes[i]
                H = T*(0.5*Tmax_slope*T + x1) - Tmaxes[i]*(0.5*Tmax_slope*Tmaxes[i] + x1)
                H += Cps_data[8][i]

            Cp_integrals_pure[i] = (H - Cps_data[11][i])
        return Cp_integrals_pure

    def _Cp_integrals_over_T_pure_fast(self, Cps_data):
        T, N = self.T, self.N
        Tmins, Tmaxes, T_int_T_coeffs = Cps_data[0], Cps_data[3], Cps_data[14]
        Cp_integrals_over_T_pure = zeros(N) if self.vectorized else [0.0]*N
        logT = log(T)
        for i in range(N):
            Tmin = Tmins[i]
            if T < Tmin:
                x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
                S = (Cps_data[1][i]*T + x1*logT)
            elif (Tmin <= T <= Tmaxes[i]):
                S = 0.0
                for c in T_int_T_coeffs[i]:
                    S = S*T + c
                S += Cps_data[6][i]*logT
                # The below should be in a constant - taking the place of Cps_data[9]
                S -= Cps_data[9][i]
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    S += (Cps_data[1][i]*Tmin + x1*log(Tmin))
            else:
#                    x1 = Cps_data[2][i] - Cps_data[1][i]*Tmin
#                    S = (Cps_data[1][i]*Tmin + x1*log(Tmin))
#                    S += (Cps_data[10][i] - Cps_data[9][i])
                S = Cps_data[10][i]
                # The above should be in the constant Cps_data[10], - x2*log(Tmaxes[i]) also
                x2 = Cps_data[5][i] - Tmaxes[i]*Cps_data[4][i]
                S += -Cps_data[4][i]*(Tmaxes[i] - T) + x2*logT #- x2*log(Tmaxes[i])

            Cp_integrals_over_T_pure[i] = (S - Cps_data[15][i])
        return Cp_integrals_over_T_pure

    def Cpigs_pure(self):
        r'''Method to calculate and return the ideal-gas heat capacities of
        every component in the phase. This method is powered by the
        `HeatCapacityGases` objects, except when all components have the same
        heat capacity form and a fast implementation has been written for it
        (currently only polynomials).

        Returns
        -------
        Cp_ig : list[float]
            Molar ideal gas heat capacities, [J/(mol*K)]
        '''
        try:
            return self._Cpigs
        except AttributeError:
            pass
        if self.Cpgs_poly_fit:
            self._Cpigs = self._Cp_pure_fast(self._Cpgs_data)
            return self._Cpigs

        T = self.T
        Cpigs = [i.T_dependent_property(T) for i in self.HeatCapacityGases]
        if self.vectorized:
            Cpigs = array(Cpigs)
        self._Cpigs = Cpigs
        return Cpigs

    def Cpig_integrals_pure(self):
        r'''Method to calculate and return the integrals of the ideal-gas heat
        capacities of every component in the phase from a temperature of
        :obj:`Phase.T_REF_IG` to the system temperature. This method is powered by the
        `HeatCapacityGases` objects, except when all components have the same
        heat capacity form and a fast implementation has been written for it
        (currently only polynomials).

        .. math::
            \Delta H^{ig} = \int^T_{T_{ref}} C_p^{ig} dT

        Returns
        -------
        dH_ig : list[float]
            Integrals of ideal gas heat capacity from the reference
            temperature to the system temperature, [J/(mol)]
        '''
        try:
            return self._Cpig_integrals_pure
        except AttributeError:
            pass
        if self.Cpgs_poly_fit:
            self._Cpig_integrals_pure = self._Cp_integrals_pure_fast(self._Cpgs_data)
            return self._Cpig_integrals_pure

        T, T_REF_IG, HeatCapacityGases = self.T, self.T_REF_IG, self.HeatCapacityGases
        Cpig_integrals_pure = [obj.T_dependent_property_integral(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        if self.vectorized:
            Cpig_integrals_pure = array(Cpig_integrals_pure)
        self._Cpig_integrals_pure = Cpig_integrals_pure
        return Cpig_integrals_pure

    def Cpig_integrals_over_T_pure(self):
        r'''Method to calculate and return the integrals of the ideal-gas heat
        capacities divided by temperature of every component in the phase from
        a temperature of :obj:`Phase.T_REF_IG` to the system temperature.
        This method is powered by the
        `HeatCapacityGases` objects, except when all components have the same
        heat capacity form and a fast implementation has been written for it
        (currently only polynomials).

        .. math::
            \Delta S^{ig} = \int^T_{T_{ref}} \frac{C_p^{ig}}{T} dT

        Returns
        -------
        dS_ig : list[float]
            Integrals of ideal gas heat capacity over temperature from the
            reference temperature to the system temperature, [J/(mol)]
        '''
        try:
            return self._Cpig_integrals_over_T_pure
        except AttributeError:
            pass

        if self.Cpgs_poly_fit:
            self._Cpig_integrals_over_T_pure = self._Cp_integrals_over_T_pure_fast(self._Cpgs_data)
            return self._Cpig_integrals_over_T_pure


        T, T_REF_IG, HeatCapacityGases = self.T, self.T_REF_IG, self.HeatCapacityGases
        Cpig_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                   for obj in HeatCapacityGases]
        if self.vectorized:
            Cpig_integrals_over_T_pure = array(Cpig_integrals_over_T_pure)
        self._Cpig_integrals_over_T_pure = Cpig_integrals_over_T_pure
        return Cpig_integrals_over_T_pure

    def dCpigs_dT_pure(self):
        r'''Method to calculate and return the first temperature derivative of
        ideal-gas heat capacities of every component in the phase. This method
        is powered by the `HeatCapacityGases` objects, except when all
        components have the same heat capacity form and a fast implementation
        has been written for it (currently only polynomials).

        .. math::
            \frac{\partial C_p^{ig}}{\partial T}

        Returns
        -------
        dCp_ig_dT : list[float]
            First temperature derivatives of molar ideal gas heat capacities,
            [J/(mol*K^2)]
        '''
        try:
            return self._dCpigs_dT
        except AttributeError:
            pass
        if self.Cpgs_poly_fit:
            self._dCpigs_dT = self._dCp_dT_pure_fast(self._Cpgs_data)
            return self._dCpigs_dT

        T = self.T
        dCpigs_dT = [i.T_dependent_property_derivative(T) for i in self.HeatCapacityGases]
        if self.vectorized:
            dCpigs_dT = array(dCpigs_dT)
        self._dCpigs_dT = dCpigs_dT
        return dCpigs_dT


    def _Cpls_pure(self):
        try:
            return self._Cpls
        except AttributeError:
            pass
        if self.Cpls_poly_fit:
            self._Cpls = self._Cp_pure_fast(self._Cpls_data)
            return self._Cpls

        T = self.T
        Cpls = [i.T_dependent_property(T) for i in self.HeatCapacityLiquids]
        if self.vectorized:
            Cpls = array(Cpls)
        self._Cpls = Cpls
        return Cpls

    def _Cpl_integrals_pure(self):
        try:
            return self._Cpl_integrals_pure
        except AttributeError:
            pass
#        def to_quad(T, i):
#            l2 = self.to_TP_zs(T, self.P, self.zs)
#            return l2._Cpls_pure()[i] + (l2.Vms_sat()[i] - T*l2.dVms_sat_dT()[i])*l2.dPsats_dT()[i]
#        from scipy.integrate import quad
#        vals = [float(quad(to_quad, self.T_REF_IG, self.T, args=i)[0]) for i in range(self.N)]
##        print(vals, self._Cp_integrals_pure_fast(self._Cpls_data))
#        return vals

        if self.Cpls_poly_fit:
            self._Cpl_integrals_pure = self._Cp_integrals_pure_fast(self._Cpls_data)
            return self._Cpl_integrals_pure

        T, T_REF_IG, HeatCapacityLiquids = self.T, self.T_REF_IG, self.HeatCapacityLiquids
        Cpl_integrals_pure = [obj.T_dependent_property_integral(T_REF_IG, T)
                                   for obj in HeatCapacityLiquids]
        if self.vectorized:
            Cpl_integrals_pure = array(Cpl_integrals_pure)
        self._Cpl_integrals_pure = Cpl_integrals_pure
        return Cpl_integrals_pure

    def _Cpl_integrals_over_T_pure(self):
        try:
            return self._Cpl_integrals_over_T_pure
        except AttributeError:
            pass
#        def to_quad(T, i):
#            l2 = self.to_TP_zs(T, self.P, self.zs)
#            return (l2._Cpls_pure()[i] + (l2.Vms_sat()[i] - T*l2.dVms_sat_dT()[i])*l2.dPsats_dT()[i])/T
#        from scipy.integrate import quad
#        vals = [float(quad(to_quad, self.T_REF_IG, self.T, args=i)[0]) for i in range(self.N)]
##        print(vals, self._Cp_integrals_over_T_pure_fast(self._Cpls_data))
#        return vals

        if self.Cpls_poly_fit:
            self._Cpl_integrals_over_T_pure = self._Cp_integrals_over_T_pure_fast(self._Cpls_data)
            return self._Cpl_integrals_over_T_pure


        T, T_REF_IG, HeatCapacityLiquids = self.T, self.T_REF_IG, self.HeatCapacityLiquids
        Cpl_integrals_over_T_pure = [obj.T_dependent_property_integral_over_T(T_REF_IG, T)
                                   for obj in HeatCapacityLiquids]
        if self.vectorized:
            Cpl_integrals_over_T_pure = array(Cpl_integrals_over_T_pure)
        self._Cpl_integrals_over_T_pure = Cpl_integrals_over_T_pure
        return Cpl_integrals_over_T_pure

    def V_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas molar volume of the
        phase.

        .. math::
            V^{ig} = \frac{RT}{P}

        Returns
        -------
        V : float
            Ideal gas molar volume, [m^3/mol]
        '''
        return self.R*self.T/self.P

    def H_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas enthalpy of the phase.

        .. math::
            H^{ig} = \sum_i z_i {H_{i}^{ig}}

        Returns
        -------
        H : float
            Ideal gas enthalpy, [J/(mol)]
        '''
        try:
            return self._H_ideal_gas
        except AttributeError:
            pass
        if self.vectorized:
            H = float(dot(self.zs, self.Cpig_integrals_pure()))
        else:
            H = 0.0
            for zi, Cp_int in zip(self.zs, self.Cpig_integrals_pure()):
                H += zi*Cp_int
        self._H_ideal_gas = H
        return H

    def S_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas entropy of the phase.

        .. math::
            S^{ig} = \sum_i z_i S_{i}^{ig} - R\ln\left(\frac{P}{P_{ref}}\right)
            - R\sum_i z_i \ln(z_i)

        Returns
        -------
        S : float
            Ideal gas molar entropy, [J/(mol*K)]
        '''
        try:
            return self._S_ideal_gas
        except AttributeError:
            pass
        Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()
        log_zs = self.log_zs()
        P, zs, cmps = self.P, self.zs, range(self.N)
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*log(P*P_REF_IG_INV)

        if self.vectorized:
            S -= R*float(dot(zs, log_zs))
            S += float(dot(zs, Cpig_integrals_over_T_pure))
        else:
            S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
            for i in cmps:
                S += zs[i]*Cpig_integrals_over_T_pure[i]
        self._S_ideal_gas = S
        return S

    def Cp_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas heat capacity of the
        phase.

        .. math::
            C_p^{ig} = \sum_i z_i {C_{p,i}^{ig}}

        Returns
        -------
        Cp : float
            Ideal gas heat capacity, [J/(mol*K)]
        '''
        try:
            return self._Cp_ideal_gas
        except AttributeError:
            pass
        Cpigs_pure = self.Cpigs_pure()
        Cp, zs = 0.0, self.zs
        if self.vectorized:
            Cp = float(dot(zs, Cpigs_pure))
        else:
            for i in range(self.N):
                Cp += zs[i]*Cpigs_pure[i]
        self._Cp_ideal_gas = Cp
        return Cp

    def Cp_ideal_gas_mass(self):
        r'''Method to calculate and return mass constant pressure
        departure heat capacity of the phase.

        .. math::
            Cp_{ideal, mass} = \frac{1000 Cp_{ideal, molar}}{MW}

        Returns
        -------
        Cp_ideal_gas_mass : float
            Mass departure heat capacity, [J/(kg*K)]
        '''
        try:
            return self._Cp_ideal_gas_mass
        except AttributeError:
            pass

        self._Cp_ideal_gas_mass = Cp_ideal_gas_mass = self.Cp_ideal_gas()*1e3*self.MW_inv()
        return Cp_ideal_gas_mass


    def Cv_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas constant volume heat
        capacity of the phase.

        .. math::
            C_v^{ig} = \sum_i z_i {C_{p,i}^{ig}} - R

        Returns
        -------
        Cv : float
            Ideal gas constant volume heat capacity, [J/(mol*K)]
        '''
        try:
            Cp = self._Cp_ideal_gas
        except AttributeError:
            Cp = self.Cp_ideal_gas()
        return Cp - self.R

    def Cv_dep(self):
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
        return self.Cv() - self.Cv_ideal_gas()

    def Cp_Cv_ratio_ideal_gas(self):
        r'''Method to calculate and return the ratio of the ideal-gas heat
        capacity to its constant-volume heat capacity.

        .. math::
            \frac{C_p^{ig}}{C_v^{ig}}

        Returns
        -------
        Cp_Cv_ratio_ideal_gas : float
            Cp/Cv for the phase as an ideal gas, [-]
        '''
        return self.Cp_ideal_gas()/self.Cv_ideal_gas()

    def G_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas Gibbs free energy of
        the phase.

        .. math::
            G^{ig} = H^{ig} - T S^{ig}

        Returns
        -------
        G_ideal_gas : float
            Ideal gas free energy, [J/(mol)]
        '''
        G_ideal_gas = self.H_ideal_gas() - self.T*self.S_ideal_gas()
        return G_ideal_gas

    def U_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas internal energy of
        the phase.

        .. math::
            U^{ig} = H^{ig} - P V^{ig}

        Returns
        -------
        U_ideal_gas : float
            Ideal gas internal energy, [J/(mol)]
        '''
        U_ideal_gas = self.H_ideal_gas() - self.P*self.V_ideal_gas()
        return U_ideal_gas

    def A_ideal_gas(self):
        r'''Method to calculate and return the ideal-gas Helmholtz energy of
        the phase.

        .. math::
            A^{ig} = U^{ig} - T S^{ig}

        Returns
        -------
        A_ideal_gas : float
            Ideal gas Helmholtz free energy, [J/(mol)]
        '''
        A_ideal_gas = self.U_ideal_gas() - self.T*self.S_ideal_gas()
        return A_ideal_gas

    def H_ideal_gas_mass(self):
        r'''Method to calculate and return the mass ideal-gas enthalpy of the phase.

        Returns
        -------
        H_ideal_gas_mass : float
            Ideal gas mass enthalpy, [J/(kg)]
        '''
        return property_molar_to_mass(self.H_ideal_gas(), self.MW())

    def S_ideal_gas_mass(self):
        r'''Method to calculate and return the mass ideal-gas entropy of the phase.

        Returns
        -------
        S_ideal_gas_mass : float
            Ideal gas mass entropy, [J/(kg*K)]
        '''
        return property_molar_to_mass(self.S_ideal_gas(), self.MW())

    def G_ideal_gas_mass(self):
        r'''Method to calculate and return the mass ideal-gas Gibbs free energy of
        the phase.

        Returns
        -------
        G_ideal_gas_mass : float
            Ideal gas mass free energy, [J/(kg)]
        '''
        return property_molar_to_mass(self.G_ideal_gas(), self.MW())

    def U_ideal_gas_mass(self):
        r'''Method to calculate and return the mass ideal-gas internal energy of
        the phase.

        Returns
        -------
        U_ideal_gas_mass : float
            Ideal gas mass internal energy, [J/(kg)]
        '''
        return property_molar_to_mass(self.U_ideal_gas(), self.MW())

    def A_ideal_gas_mass(self):
        r'''Method to calculate and return the mass ideal-gas Helmholtz energy of
        the phase.

        Returns
        -------
        A_ideal_gas_mass : float
            Ideal gas mass Helmholtz free energy, [J/(kg)]
        '''
        return property_molar_to_mass(self.A_ideal_gas(), self.MW())












    def _set_ideal_gas_standard_state(self):
        # TODO: Do not depend on Chemical infrastructure in the future
        CASs = self.CASs
        T = self.T
        zs = self.zs
        from thermo.chemical_utils import _standard_state_ideal_gas_formation_direct
        H_chemicals = []
        S_chemicals = []
        G_chemicals = []

        try:
            Hfs = self.Hfs
        except:
            Hfs = self.constants.Hfgs
        try:
            Sfs = self.Sfs
        except:
            Sfs = self.constants.Sfgs
        try:
            HeatCapacityGases = self.HeatCapacityGases
        except:
            HeatCapacityGases = self.correlations.HeatCapacityGases
        atomss = self.constants.atomss

        for i in range(self.N):
            # Hi, Si, Gi = standard_state_ideal_gas_formation(Chemical(CASs[i]), T)
            Hi, Si, Gi = _standard_state_ideal_gas_formation_direct(T, Hfs[i], Sfs[i], atoms=atomss[i], gas_Cp=HeatCapacityGases[i])
            H_chemicals.append(Hi)
            S_chemicals.append(Si)
            G_chemicals.append(Gi)
        G = mixing_simple(G_chemicals, zs)
        H = mixing_simple(H_chemicals, zs)
        S = mixing_simple(S_chemicals, zs)

        self._H_ideal_gas_standard_state = H
        self._Hs_ideal_gas_standard_state = H_chemicals
        self._S_ideal_gas_standard_state = S
        self._Ss_ideal_gas_standard_state = S_chemicals
        self._G_ideal_gas_standard_state = G
        self._Gs_ideal_gas_standard_state = G_chemicals

    def H_ideal_gas_standard_state(self):
        try:
            return self._H_ideal_gas_standard_state
        except:
            self._set_ideal_gas_standard_state()
            return self._H_ideal_gas_standard_state

    def Hs_ideal_gas_standard_state(self):
        try:
            return self._Hs_ideal_gas_standard_state
        except:
            self._set_ideal_gas_standard_state()
            return self._Hs_ideal_gas_standard_state

    def G_ideal_gas_standard_state(self):
        try:
            return self._G_ideal_gas_standard_state
        except:
            self._set_ideal_gas_standard_state()
            return self._G_ideal_gas_standard_state

    def Gs_ideal_gas_standard_state(self):
        try:
            return self._Gs_ideal_gas_standard_state
        except:
            self._set_ideal_gas_standard_state()
            return self._Gs_ideal_gas_standard_state

    def S_ideal_gas_standard_state(self):
        try:
            return self._S_ideal_gas_standard_state
        except:
            self._set_ideal_gas_standard_state()
            return self._S_ideal_gas_standard_state

    def Ss_ideal_gas_standard_state(self):
        try:
            return self._Ss_ideal_gas_standard_state
        except:
            self._set_ideal_gas_standard_state()
            return self._Ss_ideal_gas_standard_state

    def _set_mechanical_critical_point(self):
        zs = self.zs
        # Get initial guess
        try:
            try:
                Tcs, Pcs = self.Tcs, self.Pcs
            except:
                try:
                    Tcs, Pcs = self.eos_mix.Tcs, self.eos_mix.Pcs
                except:
                    Tcs, Pcs = self.constants.Tcs, self.constants.Pcs
            Pmc, Tmc = 0.0, 0.0
            for i in range(self.N):
                Pmc += Pcs[i]*zs[i]

            Tc_rts = [sqrt(Tc) for Tc in Tcs]
            for i in range(self.N):
                tot = 0.0
                for j in range(self.N):
                    tot += zs[j]*Tc_rts[j]
                Tmc += tot*Tc_rts[i]*zs[i]
        except:
            Tmc = 300.0
            Pmc = 1e6

        # Try to solve it
        solution = [None]
        def to_solve(TP):
            global new
            T, P = float(TP[0]), float(TP[1])
            new = self.to_TP_zs(T=T, P=P, zs=zs)
            errs = [new.dP_drho(), new.d2P_drho2()]
            solution[0] = new
            return errs

        jac = lambda TP: jacobian(to_solve, TP, scalar=False)
        TP, iters = newton_system(to_solve, [Tmc, Pmc], jac=jac, ytol=1e-10)
#        TP = fsolve(to_solve, [Tmc, Pmc]) # fsolve handles the discontinuities badly
        T, P = float(TP[0]), float(TP[1])
        new = solution[0]
        V = new.V()
        self._mechanical_critical_T = T
        self._mechanical_critical_P = P
        self._mechanical_critical_V = V
        return T, P, V

    def Tmc(self):
        r'''Method to calculate and return the mechanical critical temperature
        of the phase.

        Returns
        -------
        Tmc : float
            Mechanical critical temperature, [K]
        '''
        try:
            return self._mechanical_critical_T
        except:
            self._set_mechanical_critical_point()
            return self._mechanical_critical_T

    def Pmc(self):
        r'''Method to calculate and return the mechanical critical pressure
        of the phase.

        Returns
        -------
        Pmc : float
            Mechanical critical pressure, [Pa]
        '''
        try:
            return self._mechanical_critical_P
        except:
            self._set_mechanical_critical_point()
            return self._mechanical_critical_P

    def Vmc(self):
        r'''Method to calculate and return the mechanical critical volume
        of the phase.

        Returns
        -------
        Vmc : float
            Mechanical critical volume, [m^3/mol]
        '''
        try:
            return self._mechanical_critical_V
        except:
            self._set_mechanical_critical_point()
            return self._mechanical_critical_V

    def Zmc(self):
        r'''Method to calculate and return the mechanical critical
        compressibility of the phase.

        Returns
        -------
        Zmc : float
            Mechanical critical compressibility, [-]
        '''
        return (self.Pmc()*self.Vmc())/(self.R*self.Tmc())

    def dH_dT_P(self):
        r'''Method to calculate and return the temperature derivative of
        enthalpy of the phase at constant pressure.

        Returns
        -------
        dH_dT_P : float
            Temperature derivative of enthalpy, [J/(mol*K)]
        '''
        return self.dH_dT()

    def dH_dP_T(self):
        r'''Method to calculate and return the pressure derivative of
        enthalpy of the phase at constant pressure.

        Returns
        -------
        dH_dP_T : float
            Pressure derivative of enthalpy, [J/(mol*Pa)]
        '''
        return self.dH_dP()

    def dS_dP_T(self):
        r'''Method to calculate and return the pressure derivative of
        entropy of the phase at constant pressure.

        Returns
        -------
        dS_dP_T : float
            Pressure derivative of entropy, [J/(mol*K*Pa)]
        '''
        return self.dS_dP()

    def dS_dV_T(self):
        r'''Method to calculate and return the volume derivative of
        entropy of the phase at constant temperature.

        Returns
        -------
        dS_dV_T : float
            Volume derivative of entropy, [J/(K*m^3)]
        '''
        return self.dS_dP_T()*self.dP_dV()

    def dS_dV_P(self):
        r'''Method to calculate and return the volume derivative of
        entropy of the phase at constant pressure.

        Returns
        -------
        dS_dV_P : float
            Volume derivative of entropy, [J/(K*m^3)]
        '''
        return self.dS_dT_P()*self.dT_dV()

    def dP_dT_P(self):
        r'''Method to calculate and return the temperature derivative of
        temperature of the phase at constant pressure.

        Returns
        -------
        dP_dT_P : float
            Temperature derivative of temperature, [-]
        '''
        return 0.0

    def dP_dV_P(self):
        r'''Method to calculate and return the volume derivative of
        pressure of the phase at constant pressure.

        Returns
        -------
        dP_dV_P : float
            Volume derivative of pressure of the phase at constant pressure,
            [Pa*mol/m^3]
        '''
        return 0.0

    def dT_dP_T(self):
        r'''Method to calculate and return the pressure derivative of
        temperature of the phase at constant temperature.

        Returns
        -------
        dT_dP_T : float
            Pressure derivative of temperature of the phase at constant
            temperature, [K/Pa]
        '''
        return 0.0

    def dT_dV_T(self):
        r'''Method to calculate and return the volume derivative of
        temperature of the phase at constant temperature.

        Returns
        -------
        dT_dV_T : float
            Pressure derivative of temperature of the phase at constant
            temperature, [K*mol/m^3]
        '''
        return 0.0

    def dV_dT_V(self):
        r'''Method to calculate and return the temperature derivative of
        volume of the phase at constant volume.

        Returns
        -------
        dV_dT_V : float
             Temperature derivative of volume of the phase at constant volume,
             [m^3/(mol*K)]
        '''
        return 0.0

    def dV_dP_V(self):
        r'''Method to calculate and return the volume derivative of
        pressure of the phase at constant volume.

        Returns
        -------
        dV_dP_V : float
             Pressure derivative of volume of the phase at constant pressure,
             [m^3/(mol*Pa)]
        '''
        return 0.0

    def dP_dP_T(self):
        r'''Method to calculate and return the pressure derivative of
        pressure of the phase at constant temperature.

        Returns
        -------
        dP_dP_T : float
             Pressure derivative of pressure of the phase at constant
             temperature, [-]
        '''
        return 1.0

    def dP_dP_V(self):
        r'''Method to calculate and return the pressure derivative of
        pressure of the phase at constant volume.

        Returns
        -------
        dP_dP_V : float
             Pressure derivative of pressure of the phase at constant
             volume, [-]
        '''
        return 1.0

    def dT_dT_P(self):
        r'''Method to calculate and return the temperature derivative of
        temperature of the phase at constant pressure.

        Returns
        -------
        dT_dT_P : float
             Temperature derivative of temperature of the phase at constant
             pressure, [-]
        '''
        return 1.0

    def dT_dT_V(self):
        r'''Method to calculate and return the temperature derivative of
        temperature of the phase at constant volume.

        Returns
        -------
        dT_dT_V : float
             Temperature derivative of temperature of the phase at constant
             volume, [-]
        '''
        return 1.0

    def dV_dV_T(self):
        r'''Method to calculate and return the volume derivative of
        volume of the phase at constant temperature.

        Returns
        -------
        dV_dV_T : float
             Volume derivative of volume of the phase at constant
             temperature, [-]
        '''
        return 1.0

    def dV_dV_P(self):
        r'''Method to calculate and return the volume derivative of
        volume of the phase at constant pressure.

        Returns
        -------
        dV_dV_P : float
             Volume derivative of volume of the phase at constant
             pressure, [-]
        '''
        return 1.0

    d2T_dV2_P = d2T_dV2
    d2V_dT2_P = d2V_dT2
    d2V_dP2_T = d2V_dP2
    d2T_dP2_V = d2T_dP2
    dV_dP_T = dV_dP
    dV_dT_P = dV_dT
    dT_dP_V = dT_dP
    dT_dV_P = dT_dV


    # More derivatives - at const H, S, G, U, A
    _derivs_jacobian_x = 'V'
    _derivs_jacobian_y = 'T'

    def _derivs_jacobian(self, a, b, c, x=_derivs_jacobian_x,
                         y=_derivs_jacobian_y):
        r'''Calculates and returns a first-order derivative of one property
        with respect to another property at constant another property.

        This is particularly useful to obtain derivatives with respect to
        another property which is not an intensive variable in a model,
        allowing for example derivatives at constant enthalpy or Gibbs energy
        to be obtained. This formula is obtained from the first derivative
        principles of reciprocity, the chain rule, and the triple product rule
        as shown in [1]_.

        ... math::
            \left(\frac{\partial a}{\partial b}\right)_{c}=
            \frac{\left(\frac{\partial a}{\partial x}\right)_{y}\left(
            \frac{\partial c}{\partial y}\right)_{x}-\left(\frac{\partial a}{
            \partial y}\right)_{x}\left(\frac{\partial c}{\partial x}
            \right)_{y}}{\left(\frac{\partial b}{\partial x}\right)_{y}\left(
            \frac{\partial c}{\partial y}\right)_{x}-\left(\frac{\partial b}
            {\partial y}\right)_{x}\left(\frac{\partial c}{\partial x}
            \right)_{y}}

        References
        ----------
        .. [1] Thorade, Matthis, and Ali Saadat. "Partial Derivatives of
           Thermodynamic State Properties for Dynamic Simulation."
           Environmental Earth Sciences 70, no. 8 (April 10, 2013): 3497-3503.
           https://doi.org/10.1007/s12665-013-2394-z.
        '''
        n0 = getattr(self, f'd{a}_d{x}_{y}')()
        n1 = getattr(self, f'd{c}_d{y}_{x}')()

        n2 = getattr(self, f'd{a}_d{y}_{x}')()
        n3 = getattr(self, f'd{c}_d{x}_{y}')()

        d0 = getattr(self, f'd{b}_d{x}_{y}')()
        d1 = getattr(self, f'd{c}_d{y}_{x}')()

        d2 = getattr(self, f'd{b}_d{y}_{x}')()
        d3 = getattr(self, f'd{c}_d{x}_{y}')()

        return (n0*n1 - n2*n3)/(d0*d1 - d2*d3)


    ### Transport properties - pass them on!
    # Properties that use `constants` attributes

    def MW(self):
        r'''Method to calculate and return molecular weight of the phase.

        .. math::
            \text{MW} = \sum_i z_i \text{MW}_i

        Returns
        -------
        MW : float
             Molecular weight, [g/mol]
        '''
        try:
            return self._MW
        except AttributeError:
            pass
        zs, MWs = self.zs, self.constants.MWs
        if self.vectorized:
            MW = float(dot(zs, MWs))
        else:
            MW = 0.0
            for i in range(self.N):
                MW += zs[i]*MWs[i]
        self._MW = MW
        return MW

    def MW_inv(self):
        r'''Method to calculate and return inverse of molecular weight of the
        phase.

        .. math::
            \frac{1}{\text{MW}} = \frac{1}{\sum_i z_i \text{MW}_i}

        Returns
        -------
        MW_inv : float
             Inverse of molecular weight, [mol/g]
        '''
        try:
            return self._MW_inv
        except AttributeError:
            pass
        self._MW_inv = MW_inv = 1.0/self.MW()
        return MW_inv

    def speed_of_sound_mass(self):
        r'''Method to calculate and return the speed of sound
        of the phase.

        .. math::
            w = \left[-V^2 \frac{1000}{MW}\left(\frac{\partial P}{\partial V}
            \right)_T \frac{C_p}{C_v}\right]^{1/2}

        Returns
        -------
        w : float
            Speed of sound for a real gas, [m/s]
        '''
        # 1000**0.5 = 31.622776601683793
        return 31.622776601683793/sqrt(self.MW())*self.speed_of_sound()

    def rho_mass(self):
        r'''Method to calculate and return mass density of the phase.

        .. math::
            \rho = \frac{MW}{1000\cdot VM}

        Returns
        -------
        rho_mass : float
            Mass density, [kg/m^3]
        '''
        try:
            return self._rho_mass
        except AttributeError:
            pass
        self._rho_mass = rho_mass = self.MW()/(1000.0*self.V())
        return rho_mass

    def drho_mass_dT(self):
        r'''Method to calculate the mass density derivative with respect to
        temperature, at constant pressure.

        .. math::
            \left(\frac{\partial \rho}{\partial T}\right)_{P} =
            \frac{-\text{MW} \frac{\partial V_m}{\partial T}}{1000 V_m^2}

        Returns
        -------
        drho_mass_dT : float
           Temperature derivative of mass density at constant pressure,
           [kg/m^3/K]

        Notes
        -----
        Requires `dV_dT`, `MW`, and `V`.
        '''
        """

        This expression is readily obtainable with SymPy:

        >>> from sympy import * # doctest: +SKIP
        >>> T, P, MW = symbols('T, P, MW') # doctest: +SKIP
        >>> Vm = symbols('Vm', cls=Function) # doctest: +SKIP
        >>> rho_mass = (Vm(T))**-1*MW/1000 # doctest: +SKIP
        >>> diff(rho_mass, T) # doctest: +SKIP
        -MW*Derivative(Vm(T), T)/(1000*Vm(T)**2)
        """
        try:
            return self._drho_mass_dT
        except AttributeError:
            pass
        MW = self.MW()
        V = self.V()
        dV_dT = self.dV_dT()
        self._drho_mass_dT = drho_mass_dT = -MW*dV_dT/(1000.0*V*V)
        return drho_mass_dT

    def drho_mass_dP(self):
        r'''Method to calculate the mass density derivative with respect to
        pressure, at constant temperature.

        .. math::
            \left(\frac{\partial \rho}{\partial P}\right)_{T} =
            \frac{-\text{MW} \frac{\partial V_m}{\partial P}}{1000 V_m^2}

        Returns
        -------
        drho_mass_dP : float
           Pressure derivative of mass density at constant temperature,
           [kg/m^3/Pa]

        Notes
        -----
        Requires `dV_dP`, `MW`, and `V`.
        '''
        """

        This expression is readily obtainable with SymTy:

        >>> from sympy import * # doctest: +SKIP
        >>> P, T, MW = symbols('P, T, MW') # doctest: +SKIP
        >>> Vm = symbols('Vm', cls=Function) # doctest: +SKIP
        >>> rho_mass = (Vm(P))**-1*MW/1000 # doctest: +SKIP
        >>> diff(rho_mass, P) # doctest: +SKIP
        -MW*Derivative(Vm(P), P)/(1000*Vm(P)**2)
        """
        try:
            return self._drho_mass_dP
        except AttributeError:
            pass
        MW = self.MW()
        V = self.V()
        dV_dP = self.dV_dP()
        self._drho_mass_dP = drho_mass_dP = -MW*dV_dP/(1000.0*V*V)
        return drho_mass_dP

    def H_mass(self):
        r'''Method to calculate and return mass enthalpy of the phase.

        .. math::
            H_{mass} = \frac{1000 H_{molar}}{MW}

        Returns
        -------
        H_mass : float
            Mass enthalpy, [J/kg]
        '''
        try:
            return self._H_mass
        except AttributeError:
            pass

        self._H_mass = H_mass = self.H()*1e3*self.MW_inv()
        return H_mass

    def H_reactive_mass(self):
        r'''Method to calculate and return mass enthalpy on a reactive basis
        of the phase.

        .. math::
            H_{reactive,mass} = \frac{1000 H_{reactive, molar}}{MW}

        Returns
        -------
        H_reactive_mass : float
            Mass enthalpy on a reactive basis, [J/kg]
        '''
        try:
            return self._H_reactive_mass
        except AttributeError:
            pass

        self._H_reactive_mass = H_reactive_mass = self.H_reactive()*1e3*self.MW_inv()
        return H_reactive_mass

    def S_mass(self):
        r'''Method to calculate and return mass entropy of the phase.

        .. math::
            S_{mass} = \frac{1000 S_{molar}}{MW}

        Returns
        -------
        S_mass : float
            Mass enthalpy, [J/(kg*K)]
        '''
        try:
            return self._S_mass
        except AttributeError:
            pass

        self._S_mass = S_mass = self.S()*1e3*self.MW_inv()
        return S_mass

    def S_reactive_mass(self):
        r'''Method to calculate and return mass entropy on a reactive basis
        of the phase.

        .. math::
            S_{reactive,mass} = \frac{1000 S_{reactive, molar}}{MW}

        Returns
        -------
        S_reactive_mass : float
            Mass entropy on a reactive basis, [J/(kg*K)]
        '''
        try:
            return self._S_reactive_mass
        except AttributeError:
            pass

        self._S_reactive_mass = S_reactive_mass = self.S_reactive()*1e3*self.MW_inv()
        return S_reactive_mass

    def U_mass(self):
        r'''Method to calculate and return mass internal energy of the phase.

        .. math::
            U_{mass} = \frac{1000 U_{molar}}{MW}

        Returns
        -------
        U_mass : float
            Mass internal energy, [J/(kg)]
        '''
        try:
            return self._U_mass
        except AttributeError:
            pass

        self._U_mass = U_mass = self.U()*1e3*self.MW_inv()
        return U_mass

    def U_reactive_mass(self):
        r'''Method to calculate and return mass internal energy on a reactive basis
        of the phase.

        .. math::
            U_{reactive,mass} = \frac{1000 U_{reactive, molar}}{MW}

        Returns
        -------
        U_reactive_mass : float
            Mass internal energy on a reactive basis, [J/kg]
        '''
        try:
            return self._U_reactive_mass
        except AttributeError:
            pass

        self._U_reactive_mass = U_reactive_mass = self.U_reactive()*1e3*self.MW_inv()
        return U_reactive_mass

    def A_mass(self):
        r'''Method to calculate and return mass Helmholtz energy of the phase.

        .. math::
            A_{mass} = \frac{1000 A_{molar}}{MW}

        Returns
        -------
        A_mass : float
            Mass Helmholtz energy, [J/(kg)]
        '''
        try:
            return self._A_mass
        except AttributeError:
            pass

        self._A_mass = A_mass = self.A()*1e3*self.MW_inv()
        return A_mass

    def A_reactive_mass(self):
        r'''Method to calculate and return mass Helmholtz energy on a reactive basis
        of the phase.

        .. math::
            A_{reactive,mass} = \frac{1000 A_{reactive, molar}}{MW}

        Returns
        -------
        A_reactive_mass : float
            Mass Helmholtz energy on a reactive basis, [J/kg]
        '''
        try:
            return self._A_reactive_mass
        except AttributeError:
            pass

        self._A_reactive_mass = A_reactive_mass = self.A_reactive()*1e3*self.MW_inv()
        return A_reactive_mass

    def G_mass(self):
        r'''Method to calculate and return mass Gibbs energy of the phase.

        .. math::
            G_{mass} = \frac{1000 G_{molar}}{MW}

        Returns
        -------
        G_mass : float
            Mass Gibbs energy, [J/(kg)]
        '''
        try:
            return self._G_mass
        except AttributeError:
            pass

        self._G_mass = G_mass = self.G()*1e3*self.MW_inv()
        return G_mass

    def G_reactive_mass(self):
        r'''Method to calculate and return mass Gibbs free energy on a reactive basis
        of the phase.

        .. math::
            G_{reactive,mass} = \frac{1000 G_{reactive, molar}}{MW}

        Returns
        -------
        G_reactive_mass : float
           Gibbs free energy on a reactive basis, [J/kg]
        '''
        try:
            return self._G_reactive_mass
        except AttributeError:
            pass

        self._G_reactive_mass = G_reactive_mass = self.G_reactive()*1e3*self.MW_inv()
        return G_reactive_mass

    def Cp_mass(self):
        r'''Method to calculate and return mass constant pressure heat capacity
        of the phase.

        .. math::
            Cp_{mass} = \frac{1000 Cp_{molar}}{MW}

        Returns
        -------
        Cp_mass : float
            Mass heat capacity, [J/(kg*K)]
        '''
        try:
            return self._Cp_mass
        except AttributeError:
            pass

        self._Cp_mass = Cp_mass = self.Cp()*1e3*self.MW_inv()
        return Cp_mass

    def Cp_dep_mass(self):
        r'''Method to calculate and return mass constant pressure
        departure heat capacity of the phase.

        .. math::
            Cp_{dep, mass} = \frac{1000 Cp_{dep, molar}}{MW}

        Returns
        -------
        Cp_dep_mass : float
            Mass departure heat capacity, [J/(kg*K)]
        '''
        try:
            return self._Cp_dep_mass
        except AttributeError:
            pass

        self._Cp_dep_mass = Cp_dep_mass = self.Cp_dep()*1e3*self.MW_inv()
        return Cp_dep_mass


    def Cv_mass(self):
        r'''Method to calculate and return mass constant volume heat capacity
        of the phase.

        .. math::
            Cv_{mass} = \frac{1000 Cv_{molar}}{MW}

        Returns
        -------
        Cv_mass : float
            Mass constant volume heat capacity, [J/(kg*K)]
        '''
        try:
            return self._Cv_mass
        except AttributeError:
            pass

        self._Cv_mass = Cv_mass = self.Cv()*1e3*self.MW_inv()
        return Cv_mass

    def Cv_dep_mass(self):
        r'''Method to calculate and return mass constant pressure
        departure heat capacity of the phase.

        .. math::
            Cv_{dep, mass} = \frac{1000 Cv_{dep, molar}}{MW}

        Returns
        -------
        Cv_dep_mass : float
            Mass departure heat capacity, [J/(kg*K)]
        '''
        try:
            return self._Cv_dep_mass
        except AttributeError:
            pass

        self._Cv_dep_mass = Cv_dep_mass = self.Cv_dep()*1e3*self.MW_inv()
        return Cv_dep_mass

    def P_transitions(self):
        r'''Dummy method. The idea behind this method is to calculate any
        pressures (at constant temperature) which cause the phase properties to
        become discontinuous.

        Returns
        -------
        P_transitions : list[float]
            Transition pressures, [Pa]
        '''
        return []

    def T_max_at_V(self, V):
        r'''Method to calculate the maximum temperature the phase can create at a
        constant volume, if one exists; returns None otherwise.

        Parameters
        ----------
        V : float
            Constant molar volume, [m^3/mol]
        Pmax : float
            Maximum possible isochoric pressure, if already known [Pa]

        Returns
        -------
        T : float
            Maximum possible temperature, [K]

        Notes
        -----
        '''
        return None

    def P_max_at_V(self, V):
        r'''Dummy method. The idea behind this method, which is implemented by some
        subclasses, is to calculate the maximum pressure the phase can create at a
        constant volume, if one exists; returns None otherwise. This method,
        as a dummy method, always returns None.

        Parameters
        ----------
        V : float
            Constant molar volume, [m^3/mol]

        Returns
        -------
        P : float
            Maximum possible isochoric pressure, [Pa]
        '''
        return None

    def dspeed_of_sound_dT_P(self):
        r'''Method to calculate the temperature derivative of speed of sound
        at constant pressure in molar units.

        .. math::
            \left(\frac{\partial c}{\partial T}\right)_P =
            - \frac{\sqrt{- \frac{\operatorname{Cp}{\left(T \right)} V^{2}
            {\left(T \right)} \operatorname{dPdV_{T}}{\left(T \right)}}
            {\operatorname{Cv}{\left(T \right)}}} \left(- \frac{\operatorname{Cp}
            {\left(T \right)} V^{2}{\left(T \right)} \frac{d}{d T}
            \operatorname{dPdV_{T}}{\left(T \right)}}{2 \operatorname{Cv}{\left(T
            \right)}} - \frac{\operatorname{Cp}{\left(T \right)} V{\left(T
            \right)} \operatorname{dPdV_{T}}{\left(T \right)} \frac{d}{d T}
            V{\left(T \right)}}{\operatorname{Cv}{\left(T \right)}}
            + \frac{\operatorname{Cp}{\left(T \right)} V^{2}{\left(T \right)}
            \operatorname{dPdV_{T}}{\left(T \right)} \frac{d}{d T}
            \operatorname{Cv}{\left(T \right)}}{2 \operatorname{Cv}^{2}
            {\left(T \right)}} - \frac{V^{2}{\left(T \right)} \operatorname{
            dPdV_{T}}{\left(T \right)} \frac{d}{d T} \operatorname{Cp}{\left(T
            \right)}}{2 \operatorname{Cv}{\left(T \right)}}\right)
            \operatorname{Cv}{\left(T \right)}}{\operatorname{Cp}{\left(T
            \right)} V^{2}{\left(T \right)} \operatorname{dPdV_{T}}{\left(T
            \right)}}

        Returns
        -------
        dspeed_of_sound_dT_P : float
           Temperature derivative of speed of sound at constant pressure,
           [m*kg^0.5/s/mol^0.5/K]

        Notes
        -----
        Requires the temperature derivative of Cp and Cv both at constant
        pressure, as wel as the volume and temperature derivative of pressure,
        calculated at constant temperature and then pressure respectively.
        These can be tricky to obtain.
        '''
        """Calculation with SymPy:
        from sympy import *
        T = symbols('T')
        V, dPdV_T, Cp, Cv = symbols('V, dPdV_T, Cp, Cv', cls=Function)
        c = sqrt(-V(T)**2*dPdV_T(T)*Cp(T)/Cv(T))
        """
        x0 = self.Cp()
        x1 = self.V()
        x2 = self.dP_dV()
        x3 = self.Cv()
        x4 = x0*x2
        x5 = x4/x3
        x6 = 0.5*x1

        x50 = self.d2P_dVdT_TP()
        x51 = self.d2H_dT2()
        x52 = self.dV_dT()
        x53 = self.dCv_dT_P()

        return (-x1*x1*x5)**0.5*(x0*x6*x50 + x2*x6*x51 + x4*x52- x5*x6*x53)/(x0*x1*x2)

    def dspeed_of_sound_dP_T(self):
        r'''Method to calculate the pressure derivative of speed of sound
        at constant temperature in molar units.

        .. math::
            \left(\frac{\partial c}{\partial P}\right)_T =
            - \frac{\sqrt{- \frac{\operatorname{Cp}{\left(P \right)} V^{2}
            {\left(P \right)} \operatorname{dPdV_{T}}{\left(P \right)}}
            {\operatorname{Cv}{\left(P \right)}}} \left(- \frac{
            \operatorname{Cp}{\left(P \right)} V^{2}{\left(P \right)} \frac{d}
            {d P} \operatorname{dPdV_{T}}{\left(P \right)}}{2 \operatorname{Cv}
            {\left(P \right)}} - \frac{\operatorname{Cp}{\left(P \right)}
            V{\left(P \right)} \operatorname{dPdV_{T}}{\left(P \right)}
            \frac{d}{d P} V{\left(P \right)}}{\operatorname{Cv}{\left(P \right)
            }} + \frac{\operatorname{Cp}{\left(P \right)} V^{2}{\left(P \right)
            } \operatorname{dPdV_{T}}{\left(P \right)} \frac{d}{d P}
            \operatorname{Cv}{\left(P \right)}}{2 \operatorname{Cv}^{2}{\left(P
            \right)}} - \frac{V^{2}{\left(P \right)} \operatorname{dPdV_{T}}
            {\left(P \right)} \frac{d}{d P} \operatorname{Cp}{\left(P \right)}}
            {2 \operatorname{Cv}{\left(P \right)}}\right) \operatorname{Cv}
            {\left(P \right)}}{\operatorname{Cp}{\left(P \right)} V^{2}{\left(P
            \right)} \operatorname{dPdV_{T}}{\left(P \right)}}

        Returns
        -------
        dspeed_of_sound_dP_T : float
           Pressure derivative of speed of sound at constant temperature,
           [m*kg^0.5/s/mol^0.5/Pa]

        Notes
        -----
        '''
        """
        from sympy import *
        P = symbols('P')
        V, dPdV_T, Cp, Cv = symbols('V, dPdV_T, Cp, Cv', cls=Function)
        c = sqrt(-V(P)**2*dPdV_T(P)*Cp(P)/Cv(P))
        print(latex(diff(c, P)))
        """
        x0 = self.Cp()
        x1 = self.V()
        x2 = self.dP_dV()
        x3 = self.Cv()
        x4 = x0*x2
        x5 = x4/x3
        x6 = 0.5*x1

        x50 = self.d2P_dVdP()
        x51 = self.d2H_dTdP()
        x52 = self.dV_dP()
        x53 = self.dCv_dP_T()

        return (-x1*x1*x5)**0.5*(x0*x6*x50 + x2*x6*x51 + x4*x52- x5*x6*x53)/(x0*x1*x2)


    # Transport properties
    def mu(self):
        if isinstance(self, phases.gas_phases):
            return self.correlations.ViscosityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        elif isinstance(self, phases.liquid_phases):
            return self.correlations.ViscosityLiquidMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        else:
            raise NotImplementedError("Did not work")

    def nu(self):
        r'''Method to calculate and return the kinematic viscosity of the
        phase, [m^2/s]

        Returns
        -------
        nu : float
            Kinematic viscosity, [m^2/s]

        Notes
        -----
        '''
        mu = self.mu()
        if mu is None:
            return None
        return mu/self.rho_mass()

    kinematic_viscosity = nu

    def alpha(self):
        r'''Method to calculate and return the thermal diffusivity of the
        phase.

        .. math::
            \alpha = \frac{k}{\rho Cp}

        Returns
        -------
        alpha : float
            Thermal diffusivity, [m^2/s]

        Notes
        -----
        '''
        rho = self.rho_mass()
        k = self.k()
        Cp = self.Cp_mass()
        return thermal_diffusivity(k=k, rho=rho, Cp=Cp)

    thermal_diffusivity = alpha

    def ws(self):
        r'''Method to calculate and return the mass fractions of the phase, [-]

        Returns
        -------
        ws : list[float]
            Mass fractions, [-]

        Notes
        -----
        '''
        try:
            return self._ws
        except AttributeError:
            pass
        MWs = self.constants.MWs
        zs, cmps = self.zs, range(self.N)
        if self.vectorized:
            ws = zs*MWs
            Mavg = 1.0/ws.sum()
            ws *= Mavg
        else:
            ws = [zs[i]*MWs[i] for i in cmps]
            Mavg = 1.0/sum(ws)
            for i in cmps:
                ws[i] *= Mavg
        self._ws = ws
        return ws

    def sigma(self):
        r'''Calculate and return the surface tension of the phase.
        For details of the implementation, see
        :obj:`SurfaceTensionMixture <thermo.interface.SurfaceTensionMixture>`.

        This property is strictly the ideal-gas to liquid surface tension,
        not a true inter-phase property.

        Returns
        -------
        sigma : float
            Surface tension, [N/m]
        '''
        try:
            return self._sigma
        except AttributeError:
            pass
        try:
            phase = self.assigned_phase
        except:
            if self.is_liquid:
                phase = 'l'
            else:
                phase = 'g'
        if phase == 'g':
            return None
        elif phase == 'l':
            sigma = self.correlations.SurfaceTensionMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._sigma = sigma
        return sigma

    @property
    def beta(self):
        r'''Method to return the phase fraction of this phase.
        This method is only
        available when the phase is linked to an EquilibriumState.

        Returns
        -------
        beta : float
            Phase fraction on a molar basis, [-]

        Notes
        -----
        '''
        try:
            result = self.result
        except:
            return None
        for i, p in enumerate(result.phases):
            if p is self:
                return result.betas[i]

    @property
    def beta_mass(self):
        r'''Method to return the mass phase fraction of this phase.
        This method is only
        available when the phase is linked to an EquilibriumState.

        Returns
        -------
        beta_mass : float
            Phase fraction on a mass basis, [-]

        Notes
        -----
        '''
        try:
            result = self.result
        except:
            return None
        for i, p in enumerate(result.phases):
            if p is self:
                return result.betas_mass[i]

    @property
    def beta_volume(self):
        r'''Method to return the volumetric phase fraction of this phase.
        This method is only
        available when the phase is linked to an EquilibriumState.

        Returns
        -------
        beta_volume : float
            Phase fraction on a volumetric basis, [-]

        Notes
        -----
        '''
        try:
            result = self.result
        except:
            return None
        for i, p in enumerate(result.phases):
            if p is self:
                return result.betas_volume[i]

    @property
    def beta_volume_liquid_ref(self):
        r'''Method to return the standard liquid volume fraction of this phase.
        This method is only
        available when the phase is linked to an EquilibriumState.

        Returns
        -------
        beta_volume : float
            Phase fraction on a volumetric basis, [-]

        Notes
        -----
        '''
        try:
            result = self.result
        except:
            return None
        for i, p in enumerate(result.phases):
            if p is self:
                return result.betas_volume_liquid_ref[i]
    @property
    def VF(self):
        r'''Method to return the vapor fraction of the phase.
        If no vapor/gas is present, 0 is always returned. This method is only
        available when the phase is linked to an EquilibriumState.

        Returns
        -------
        VF : float
            Vapor fraction, [-]

        Notes
        -----
        '''
        return self.result.gas_beta

    @property
    def energy(self):
        r'''Method to return the energy (enthalpy times flow rate) of this
        phase.
        This method is only
        available when the phase is linked to an EquilibriumStream.

        Returns
        -------
        energy : float
            Enthalpy flow rate, [W]

        Notes
        -----
        '''
        try:
            try:
                return self._energy
            except:
                pass
            self._energy = self.n*self.H()
            return self._energy
        except:
            return None

    energy_calc = energy

    @property
    def energy_reactive(self):
        r'''Method to return the reactive energy (reactive enthalpy times flow rate) of this
        phase.
        This method is only
        available when the phase is linked to an EquilibriumStream.

        Returns
        -------
        energy_reactive : float
            Reactive enthalpy flow rate, [W]

        Notes
        -----
        '''
        try:
            try:
                return self._energy_reactive
            except:
                pass
            self._energy_reactive = self.n*self.H_reactive()
            return self._energy_reactive
        except:
            return None

    energy_reactive_calc = energy_reactive

    @property
    def n(self):
        r'''Method to return the molar flow rate of this phase.
        This method is only
        available when the phase is linked to an EquilibriumStream.

        Returns
        -------
        n : float
            Molar flow of the phase, [mol/s]

        Notes
        -----
        '''
        try:
            try:
                return self._n
            except:
                pass
            self._n = self.result.n*self.beta
            return self._n
        except:
            return None

    n_calc = n

    @property
    def m(self):
        r'''Method to return the mass flow rate of this phase.
        This method is only
        available when the phase is linked to an EquilibriumStream.

        Returns
        -------
        m : float
            Mass flow of the phase, [kg/s]

        Notes
        -----
        '''
        try:
            try:
                return self._m
            except:
                self._m = self.result.m*self.beta_mass
                return self._m
        except:
            return None

    m_calc = m

    @property
    def Q(self):
        r'''Method to return the actual volumetric flow rate of this phase.
        This method is only
        available when the phase is linked to an EquilibriumStream.

        Returns
        -------
        Q : float
            Volume flow of the phase, [m^3/s]

        Notes
        -----
        '''
        try:
            try:
                return self._Q
            except:
                self._Q = self.n*self.V()
                return self._Q
        except:
            return None

    Q_calc = Q

    @property
    def ns(self):
        r'''Method to return the molar flow rates of each component in
        this phase. This method is only
        available when the phase is linked to an EquilibriumStream.

        Returns
        -------
        ns : float
            Molar flow of the components in the phase, [mol/s]

        Notes
        -----
        '''
        try:
            try:
                return self._ns
            except:
                n = self.result.n*self.beta
                if self.vectorized:
                    self._ns = self.zs*n
                else:
                    self._ns = [n*zi for zi in self.zs]
                return self._ns
        except:
            return None

    ns_calc = ns

    @property
    def ms(self):
        r'''Method to return the mass flow rates of each component in
        this phase. This method is only
        available when the phase is linked to an EquilibriumStream.

        Returns
        -------
        ms : float
            Mass flow of the components in the phase, [kg/s]

        Notes
        -----
        '''
        try:
            try:
                return self._ms
            except:
                pass
            m = self.result.m*self.beta_mass
            if self.vectorized:
                self._ms = m*self.ws()
            else:
                self._ms = [m*wi for wi in self.ws()]
            return self._ms
        except:
            return None

    ms_calc = ms

    @property
    def Qgs(self):
        r'''Method to return the volume flow rate of each component in
        this phase as an ideal gas, using the configured
        temperature `T_gas_ref` and pressure `P_gas_ref`. This method is only
        available when the phase is linked to an EquilibriumStream.
        This method totally ignores phase equilibrium.

        Returns
        -------
        Qgs : float
            Ideal gas flow rates of the components in the phase, [m^3/s]

        Notes
        -----
        '''
        try:
            return self._Qgs
        except:
            pass
        settings = self.result.settings
        V = R*settings.T_gas_ref/settings.P_gas_ref
        n = self.n
        Vn = V*n
        if self.vectorized:
            self._Qgs = self.zs*Vn
        else:
            self._Qgs = [zi*Vn for zi in self.zs]
        return self._Qgs

    Qgs_calc = Qgs

    @property
    def Qg(self):
        r'''Method to return the volume flow rate of
        this phase as an ideal gas, using the configured
        temperature `T_gas_ref` and pressure `P_gas_ref`. This method is only
        available when the phase is linked to an EquilibriumStream.
        This method totally ignores phase equilibrium.

        Returns
        -------
        Qg : float
            Ideal gas flow rate of the phase, [m^3/s]

        Notes
        -----
        '''
        try:
            try:
                return self._Qg
            except:
                pass
            self._Qg = sum(self.Qgs)
            return self._Qg
        except:
            return None

    Qg_calc = Qg

    @property
    def Qls(self):
        r'''Method to return the volume flow rate of each component in
        this phase as an ideal liquid, using the configured
        `V_liquids_ref`. This method is only
        available when the phase is linked to an EquilibriumStream.
        This method totally ignores phase equilibrium.

        Returns
        -------
        Qls : float
            Ideal liquid flow rates of the components in the phase, [m^3/s]

        Notes
        -----
        '''
        try:
            return self._Qls
        except:
            pass
        ns = self.ns
        Vmls = self.result.V_liquids_ref()
        if self.vectorized:
            self._Qls = ns*Vmls
        else:
            self._Qls = [ns[i]*Vmls[i] for i in range(self.N)]
        return self._Qls

    Qls_calc = Qls

    @property
    def Ql(self):
        r'''Method to return the volume flow rate of
        this phase as an ideal liquid, using the configured
        standard molar volumes `Vml_STPs`. This method is only
        available when the phase is linked to an EquilibriumStream.
        This method totally ignores phase equilibrium.

        Returns
        -------
        Ql : float
            Ideal liquid flow rate of the phase, [m^3/s]

        Notes
        -----
        '''
        try:
            try:
                return self._Ql
            except:
                pass
            self._Ql = sum(self.Qls)
            return self._Ql
        except:
            return None

    Ql_calc = Ql


    @property
    def T_calc(self):
        return self.T

    @property
    def P_calc(self):
        return self.P

    @property
    def VF_calc(self):
        return self.VF

    @property
    def zs_calc(self):
        return self.zs

    @property
    def ws_calc(self):
        return self.ws()

    @property
    def Vfls_calc(self):
        return self.Vfls()

    @property
    def Vfgs_calc(self):
        return self.zs

    @property
    def m_calc(self):
        return self.m

    @property
    def n_calc(self):
        return self.n

    @property
    def H_calc(self):
        return self.H()

    def as_EquilibriumState(self, flasher=None):
        has_result = hasattr(self, 'result')
        if not has_result and flasher is None:
            raise ValueError("Phase must have been created into an EquilibriumState already or be provided with a flasher")
        from thermo.equilibrium import EquilibriumState
        if has_result:
            flasher = self.result.flasher
            state = self.assigned_phase
        else:
            if self.is_gas:
                state = 'g'
            elif self.is_liquid:
                state = 'l'
            elif self.is_solid:
                state = 's'

        if self.bulk_phase_type:
            if self.phase_bulk == 'l':
                gas, liquids, solids = None, [v.to_TP_zs(T=v.T, P=v.P, zs=v.zs) for v in self.phases], []
            elif self.phase_bulk == 's':
                gas, liquids, solids = None, [], [v.to_TP_zs(T=v.T, P=v.P, zs=v.zs) for v in self.phases]
            betas = self.phase_fractions
        else:
            betas = [1]
            phase_copy = self.to_TP_zs(T=self.T, P=self.P, zs=self.zs)
            if state == 'l':
                gas, liquids, solids = None, [phase_copy], []
            elif state == 'g':
                gas, liquids, solids = phase_copy, [], []
            elif state == 's':
                gas, liquids, solids = None, [], [phase_copy]
        if not hasattr(self, 'result'):
            flash_specs = {'T': self.T, 'P': self.P}
        else:
            flash_specs = self.result.flash_specs.copy()
        flash_specs['phase_as_state'] = True

        return EquilibriumState(T=self.T, P=self.P, zs=self.zs, gas=gas, liquids=liquids, solids=solids, betas=betas,
            constants=flasher.constants, correlations=flasher.correlations, flasher=flasher, settings=flasher.settings,
            flash_specs=flash_specs)

    def as_EquilibriumStream(self, flasher=None, n=None):
        state = self.as_EquilibriumState(flasher)
        from thermo.stream import EquilibriumStream
        n = self.n if n is None else n
        return EquilibriumStream(flasher=state.flasher, zs=self.zs, n=n,  P=self.P, T=self.T, existing_flash=state)

    def concentrations(self):
        r'''Method to return the molar concentrations of each component in the
        phase in units of mol/m^3. Molarity is a term used in chemistry for a
        similar concept, usually given in units of mol/L.

        Returns
        -------
        concentrations : list[float]
            Molar concentrations of all the components in the phase, [mol/m^3]

        Notes
        -----
        '''
        try:
            return self._concentrations
        except:
            pass
        rho = self.rho()
        zs = self.zs
        if self.vectorized:
            self._concentrations = concentrations = rho*zs
        else:
            self._concentrations = concentrations = [rho*zi for zi in zs]
        return concentrations

    def concentrations_gas(self):
        r'''Method to return the molar concentrations of each component in the
        phase in units of mol/m^3, using the ideal-gas molar volume of the
        phase at the chosen reference temperature and pressure.

        Returns
        -------
        concentrations_gas : list[float]
            Molar concentrations of all the components in the phase, [mol/m^3]

        Notes
        -----
        '''
        rho = self.rho_gas()
        zs = self.zs
        if self.vectorized:
            concentrations = rho*zs
        else:
            concentrations = [rho*zi for zi in zs]
        return concentrations

    def concentrations_gas_normal(self):
        r'''Method to return the molar concentrations of each component in the
        phase in units of mol/m^3, using the ideal-gas molar volume of the
        phase at the normal temperature and pressure.

        Returns
        -------
        concentrations_gas_normal : list[float]
            Molar concentrations of all the components in the phase, [mol/m^3]

        Notes
        -----
        '''
        rho = self.rho_gas_normal()
        zs = self.zs
        if self.vectorized:
            concentrations = rho*zs
        else:
            concentrations = [rho*zi for zi in zs]
        return concentrations

    def concentrations_gas_standard(self):
        r'''Method to return the molar concentrations of each component in the
        phase in units of mol/m^3, using the ideal-gas molar volume of the
        phase at the standard temperature and pressure.

        Returns
        -------
        concentrations_gas_standard : list[float]
            Molar concentrations of all the components in the phase, [mol/m^3]

        Notes
        -----
        '''
        rho = self.rho_gas_standard()
        zs = self.zs
        if self.vectorized:
            concentrations = rho*zs
        else:
            concentrations = [rho*zi for zi in zs]
        return concentrations

    def concentrations_mass(self):
        r'''Method to return the mass concentrations of each component in the
        phase in units of kg/m^3.

        Returns
        -------
        concentrations_mass : list[float]
            Mass concentrations of all the components in the phase, [kg/m^3]

        Notes
        -----
        '''
        try:
            return self._concentrations_mass
        except:
            pass
        rho_mass = self.rho_mass()
        ws = self.ws()
        if self.vectorized:
            self._concentrations_mass = rho_mass*ws
        else:
            self._concentrations_mass = [rho_mass*wi for wi in ws]
        return self._concentrations_mass

    def concentrations_mass_gas(self):
        r'''Method to return the mass concentrations of each component in the
        phase in units of kg/m^3, using the ideal-gas molar volume of the
        phase at the chosen reference temperature and pressure.

        Returns
        -------
        concentrations_mass_gas : list[float]
            Mass concentrations of all the components in the phase, [kg/m^3]

        Notes
        -----
        '''
        rho_mass = self.rho_mass_gas()
        ws = self.ws()
        if self.vectorized:
            concentrations_mass = rho_mass*ws
        else:
            concentrations_mass = [rho_mass*wi for wi in ws]
        return concentrations_mass

    def concentrations_mass_gas_normal(self):
        r'''Method to return the mass concentrations of each component in the
        phase in units of kg/m^3, using the ideal-gas molar volume of the
        phase at the normal temperature and pressure.

        Returns
        -------
        concentrations_mass_gas_normal : list[float]
            Mass concentrations of all the components in the phase, [kg/m^3]

        Notes
        -----
        '''
        rho_mass = self.rho_mass_gas_normal()
        ws = self.ws()
        if self.vectorized:
            concentrations_mass = rho_mass*ws
        else:
            concentrations_mass = [rho_mass*wi for wi in ws]
        return concentrations_mass

    def concentrations_mass_gas_standard(self):
        r'''Method to return the mass concentrations of each component in the
        phase in units of kg/m^3, using the ideal-gas molar volume of the
        phase at the standard temperature and pressure.

        Returns
        -------
        concentrations_mass_gas_standard : list[float]
            Mass concentrations of all the components in the phase, [kg/m^3]

        Notes
        -----
        '''
        rho_mass = self.rho_mass_gas_standard()
        ws = self.ws()
        if self.vectorized:
            concentrations_mass = rho_mass*ws
        else:
            concentrations_mass = [rho_mass*wi for wi in ws]
        return concentrations_mass

    def partial_pressures(self):
        r'''Method to return the partial pressures of each component in the
        phase. Note that this is the conventional definition assumed in almost
        every source; there is also a non-ideal definition.

        .. math::
            P_i = z_i P

        Returns
        -------
        partial_pressures : list[float]
            Partial pressures of all the components in the phase, [Pa]

        Notes
        -----
        '''
        try:
            return self._partial_pressures
        except:
            pass
        P = self.P
        if self.vectorized:
            self._partial_pressures = self.zs*P
        else:
            self._partial_pressures = [zi*P for zi in self.zs]
        return self._partial_pressures


class IdealGasDeparturePhase(Phase):
    # Internal phase base for calculating properties that use the ideal gas
    # reference state with Ideal Gas objects
    def H(self):
        try:
            return self._H
        except AttributeError:
            pass
        H = self.H_dep()
        if self.vectorized:
            H += float(dot(self.zs, self.Cpig_integrals_pure()))
        else:
            for zi, Cp_int in zip(self.zs, self.Cpig_integrals_pure()):
                H += zi*Cp_int

        self._H = H
        return H

    def S(self):
        try:
            return self._S
        except AttributeError:
            pass
        Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()
        log_zs = self.log_zs()
        P, zs, cmps = self.P, self.zs, range(self.N)
        P_REF_IG_INV = self.P_REF_IG_INV
        R = self.R
        S = 0.0

        if self.vectorized:
            S -= R*float(dot(zs, log_zs))
            S += float(dot(zs, Cpig_integrals_over_T_pure))
        else:
            S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
            for i in cmps:
                S += zs[i]*Cpig_integrals_over_T_pure[i]
        S -= R*log(P*P_REF_IG_INV)
        S += self.S_dep()
        self._S = S
        return S

    def Cp(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        Cpigs_pure = self.Cpigs_pure()
        Cp, zs = 0.0, self.zs
        if self.vectorized:
            Cp = float(dot(zs, Cpigs_pure))
        else:
            for i in range(self.N):
                Cp += zs[i]*Cpigs_pure[i]
        Cp += self.Cp_dep()
        self._Cp = Cp
        return Cp

    dH_dT = dH_dT_P = Cp

    def dH_dP(self):
        try:
            return self._dH_dP
        except AttributeError:
            pass
        self._dH_dP = dH_dP = self.dH_dep_dP_T()
        return dH_dP

    dH_dP_T = dH_dP

    def dH_dT_V(self):
        dH_dT_V = self.Cp_ideal_gas()
        dH_dT_V += self.dH_dep_dT_V()
        return dH_dT_V

    def dH_dP_V(self):
        dH_dP_V = self.Cp_ideal_gas()*self.dT_dP()
        dH_dP_V+= self.dH_dep_dP_V()
        return dH_dP_V

    def dH_dV_T(self):
        return self.dH_dep_dV_T()

    def dH_dV_P(self):
        dH_dV_P = self.dT_dV()*self.Cp_ideal_gas()
        dH_dV_P += self.dH_dep_dV_P()
        return dH_dV_P

    def d2H_dT2(self):
        try:
            return self._d2H_dT2
        except AttributeError:
            pass
        dCpigs_pure = self.dCpigs_dT_pure()
        dCp, zs = 0.0, self.zs
        if self.vectorized:
            dCp = float(dot(zs, dCpigs_pure))
        else:
            for i in range(self.N):
                dCp += zs[i]*dCpigs_pure[i]
        dCp += self.d2H_dep_dT2()
        self._d2H_dT2 = dCp
        return dCp

    def d2H_dT2_V(self):
        dCpigs_pure = self.dCpigs_dT_pure()
        dCp, zs = 0.0, self.zs
        if self.vectorized:
            dCp = float(dot(zs, dCpigs_pure))
        else:
            for i in range(self.N):
                dCp += zs[i]*dCpigs_pure[i]
        return dCp + self.d2H_dep_dT2_V()

    def dH_dzs(self):
        try:
            return self._dH_dzs
        except AttributeError:
            pass
        dH_dep_dzs = self.dH_dep_dzs()
        Cpig_integrals_pure = self.Cpig_integrals_pure()
        if self.vectorized:
            self._dH_dzs = dH_dep_dzs + Cpig_integrals_pure
        else:
            self._dH_dzs = [dH_dep_dzs[i] + Cpig_integrals_pure[i] for i in range(self.N)]
        return self._dH_dzs

    def dS_dT(self):
        dS_dT = self.Cp_ideal_gas() / self.T
        dS_dT += self.dS_dep_dT()
        return dS_dT

    def dS_dP(self):
        dS = 0.0
        P = self.P
        dS -= self.R/P
        dS += self.dS_dep_dP_T()
        return dS

    def dS_dT_P(self):
        return self.dS_dT()

    def dS_dT_V(self):
        r'''Method to calculate and return the first temperature derivative of
        molar entropy at constant volume of the phase.

        .. math::
            \left(\frac{\partial S}{\partial T}\right)_V =
            \frac{C_p^{ig}}{T} - \frac{R}{P}\frac{\partial P}{\partial T}
            + \left(\frac{\partial S_{dep}}{\partial T}\right)_V

        Returns
        -------
        dS_dT_V : float
            First temperature derivative of molar entropy at constant volume,
            [J/(mol*K^2)]
        '''
        # Good
        """
        # Second last bit from
        from sympy import *
        T, R = symbols('T, R')
        P = symbols('P', cls=Function)
        expr =-R*log(P(T)/101325)
        diff(expr, T)
        """
        dS_dT_V = self.Cp_ideal_gas()/self.T - self.R/self.P*self.dP_dT()
        dS_dT_V += self.dS_dep_dT_V()
        return dS_dT_V


    def dS_dP_V(self):
        dS_dP_V = -self.R/self.P + self.Cp_ideal_gas()/self.T*self.dT_dP()
        dS_dP_V += self.dS_dep_dP_V()
        return dS_dP_V

    def d2S_dP2(self):
        P = self.P
        d2S = self.R/(P*P)
        return d2S + self.d2S_dep_dP()

    def dS_dzs(self):
        try:
            return self._dS_dzs
        except AttributeError:
            pass
        cmps, eos_mix = range(self.N), self.eos_mix

        log_zs = self.log_zs()
        integrals = self.Cpig_integrals_over_T_pure()
        dS_dep_dzs = self.dS_dep_dzs()
        R = self.R
        if self.vectorized:
            self._dS_dzs = integrals - R*(log_zs + 1.0) + dS_dep_dzs
        else:
            self._dS_dzs = [integrals[i] - R*(log_zs[i] + 1.0) + dS_dep_dzs[i] for i in cmps]
        return self._dS_dzs

    def gammas(self):
        try:
            return self._gammas
        except:
            pass
        phis = self.phis()
        phi_pures = self.phi_pures()
        if self.vectorized:
            self._gammas = phis/phi_pures
        else:
            self._gammas = [phis[i]/phi_pures[i] for i in range(self.N)]
        return self._gammas


derivatives_jacobian = []

prop_iter = (('T', 'P', 'V', 'rho'), ('T', 'P', 'V', r'\rho'), ('K', 'Pa', 'm^3/mol', 'mol/m^3'), ('temperature', 'pressure', 'volume', 'density'))
for a, a_str, a_units, a_name in zip(*prop_iter):
    for b, b_str, b_units, b_name in zip(*prop_iter):
        for c, c_name in zip(('H', 'S', 'G', 'U', 'A'), ('enthalpy', 'entropy', 'Gibbs energy', 'internal energy', 'Helmholtz energy')):
            def _der(self, property=a, differentiate_by=b, at_constant=c):
                return self._derivs_jacobian(a=property, b=differentiate_by, c=at_constant)
            t = f'd{a}_d{b}_{c}'
            doc = rf"""Method to calculate and return the {b_name} derivative of {a_name} of the phase at constant {c_name}.

    .. math::
        \left(\frac{{\partial {a_str}}}{{\partial {b_str}}}\right)_{{{c}}}

Returns
-------
{t} : float
    The {b_name} derivative of {a_name} of the phase at constant {c_name}, [{a_units}/{b_units}]
"""
            setattr(Phase, t, _der)
            try:
                _der.__doc__ = doc
            except:
                pass
            derivatives_jacobian.append(t)

derivatives_thermodynamic = ['dA_dP', 'dA_dP_T', 'dA_dP_V', 'dA_dT', 'dA_dT_P', 'dA_dT_V', 'dA_dV_P', 'dA_dV_T',
             'dCv_dP_T', 'dCv_dT_P', 'dG_dP', 'dG_dP_T', 'dG_dP_V', 'dG_dT', 'dG_dT_P', 'dG_dT_V',
             'dG_dV_P', 'dG_dV_T', 'dH_dP', 'dH_dP_T', 'dH_dP_V', 'dH_dT', 'dH_dT_P', 'dH_dT_V',
             'dH_dV_P', 'dH_dV_T', 'dS_dP', 'dS_dP_T', 'dS_dP_V', 'dS_dT', 'dS_dT_P', 'dS_dT_V',
             'dS_dV_P', 'dS_dV_T', 'dU_dP', 'dU_dP_T', 'dU_dP_V', 'dU_dT', 'dU_dT_P', 'dU_dT_V',
             'dU_dV_P', 'dU_dV_T',
             # These will probably need their doc fixed
             'd2G_dP2', 'd2G_dT2',
             'd2G_dPdT', 'd2G_dTdP',

             ]
derivatives_thermodynamic_mass = []

prop_names = {'A' : 'Helmholtz energy',
              'G': 'Gibbs free energy',
              'U': 'internal energy',
              'H': 'enthalpy',
              'S': 'entropy',
              'T': 'temperature',
              'P': 'pressure',
              'V': 'volume', 'Cv': 'Constant-volume heat capacity'}
prop_units = {'Cv': 'J/(mol*K)', 'A': 'J/mol', 'G': 'J/mol', 'H': 'J/mol',
              'S': 'J/(mol*K)', 'U': 'J/mol', 'T': 'K', 'P': 'Pa', 'V': 'm^3/mol'}
for attr in derivatives_thermodynamic:
    def _der(self, prop=attr):
        return getattr(self, prop)()*1e3*self.MW_inv()
    try:
        base, end = attr.split('_', maxsplit=1)
    except:
        splits = attr.split('_')
        base = splits[0]
        end = '_'.join(splits[1:])

    vals = attr.replace('d', '').split('_')
    try:
        prop, diff_by, at_constant = vals
    except:
        prop, diff_by = vals
        at_constant = 'T' if diff_by == 'P' else 'P'
    s = f'{base}_mass_{end}'

    if '2' not in attr:
        # TODO docs for second mass derivatives

        doc = rf"""Method to calculate and return the {prop_names[diff_by]} derivative of mass {prop_names[prop]} of the phase at constant {prop_names[at_constant]}.

        .. math::
            \left(\frac{{\partial {prop}_{{\text{{mass}}}}}}{{\partial {diff_by}}}\right)_{{{at_constant}}}

    Returns
    -------
    {s} : float
        The {prop_names[diff_by]} derivative of mass {prop_names[prop]} of the phase at constant {prop_names[at_constant]}, [{prop_units[prop]}/{prop_units[diff_by]}]
    """
        try:
            _der.__doc__ = doc#'Automatically generated derivative. %s %s' %(base, end)
        except:
            pass
    setattr(Phase, s, _der)
    derivatives_thermodynamic_mass.append(s)
del prop_names, prop_units
