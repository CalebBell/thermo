'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019, 2020, 2021, 2022, 2023
Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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


__all__ = ['Stream', 'EnergyStream', 'StreamArgs', 'EquilibriumStream',
           'energy_balance', 'mole_balance']

#import enum
try:
    from collections import OrderedDict
except:
    pass

from chemicals.exceptions import OverspeficiedError
from chemicals.utils import (
    Vfs_to_zs,
    Vm_to_rho,
    hash_any_primitive,
    mixing_simple,
    normalize,
    object_data,
    property_mass_to_molar,
    property_molar_to_mass,
    solve_flow_composition_mix,
    ws_to_zs,
    zs_to_Vfs,
    zs_to_ws,
)
from chemicals.volume import ideal_gas
from fluids.constants import R

from thermo.equilibrium import EquilibriumState
from thermo.mixture import Mixture
from thermo.serialize import JsonOptEncodable, object_lookups


class StreamArgs:
    '''Creates an StreamArgs object, which is a holder and specification
    tracking/counting utility object for the purposes of making
    heat and material balances. Unlike :obj:`EquilibriumStream`, this
    object is mutable and doesn't require any specifications. Specifications
    can be set as they become available, and then a :obj:`EquilibriumStream`
    can be generated from the method :obj:`StreamArgs.flash`.

    The specification tracking is the key purpose of this object. Once a
    `T` and `P` have been set, `V` **can't** be set because there are no
    degrees of freedom.
    Another state specification mst be removed by setting it to None
    before a new specification can be set. The specifications supported are the same
    as those of :obj:`EquilibriumStream`.
    Handling flow flow vs. composition is fuzzier because of the common use cases
    where a flow rate may be known, but not composition; or the composition but not
    the flow, however later when the product
    flows are known it is desireable to set it. To allow this,
    the variables `ns`, `ms`, `Qls`, and `Qgs` are allowed to overwrite an existing
    flow and/or composition specification even if if `multiple_composition_basis` is False.

    Eveen if the property wasn't set to this object, but if enough degrees of freedom
    are known, the actual value can usually be queried by adding `_calc` to the attribute,
    e.g. `H_calc` is available if `T`, `P`, and `zs` are known. This will perform the
    flash if it hasn't already been done.

    The state can be specified using any two of:

    * Temperature `T` [K]
    * Pressure `P` [Pa]
    * Vapor fraction `VF`
    * Enthalpy `H` [J/mol] or `H_mass` [J/kg]
    * Entropy `S` [J/mol/K] or `S_mass` [J/kg/K]
    * Internal energy `U` [J/mol] or `U_mass` [J/kg]
    * Gibbs free energy `G` [J/mol] or `G_mass` [J/kg]
    * Helmholtz energy `A` [J/mol] or `A_mass` [J/kg]
    * Energy `energy` [W] and `energy_reactive` [W] which count as a enthalpy spec only when flow rate is given
    * Reactive enthalpy `H_reactive` [J/mol]
    * Molar volume `V` [m^3/mol], molar density `rho` [mol/m^3], or mass density `rho_mass`, [kg/m^3]

    The composition can be specified using any of:

    * Mole fractions `zs`
    * Mass fractions `ws`
    * Liquid standard volume fractions `Vfls`
    * Gas standard volume fractions `Vfgs`
    * Mole flow rates of each component `ns` [mol/s]
    * Mass flow rates of each component `ms` [kg/s]
    * Liquid standard volume flow rates of each component `Qls` [m^3/s]
    * Gas standard flow rates of each component `Qgs` [m^3/s]

    In addition to setting flow rate via component flow rates, total flow rates can be specified using:

    * Mole flow rate `n` [mol/s]
    * Mass flow rate `m` [kg/s]
    * Actual volume flow rate `Q` [m^3/s]
    * Liquid volume standard flow rate `Ql` [m^3/s]
    * Gas standard volume flow rate `Qg` [m^3/s]

    All parameters are also attributes.

    Parameters
    ----------
    flasher : One of :obj:`thermo.flash.FlashPureVLS`, :obj:`thermo.flash.FlashVL`, :obj:`thermo.flash.FlashVLN`,
        The configured flash object which can perform flashes for the
        configured components, [-]
    zs : list, optional
        Mole fractions of all components [-]
    ws : list, optional
        Mass fractions of all components [-]
    Vfls : list, optional
        Volume fractions of all components as a hypothetical liquid phase based
        on pure component densities [-]
    Vfgs : list, optional
        Volume fractions of all components as a hypothetical gas phase based
        on pure component densities [-]
    ns : list, optional
        Mole flow rates of each component [mol/s]
    ms : list, optional
        Mass flow rates of each component [kg/s]
    Qls : list, optional
        Component volumetric flow rate specs for a hypothetical liquid phase based on
        :obj:`EquilibriumState.V_liquids_ref` [m^3/s]
    Qgs : list, optional
        Component volumetric flow rate specs for a hypothetical gas phase based on
        :obj:`EquilibriumState.V_gas` [m^3/s]
    Ql : float, optional
        Total volumetric flow rate spec for a hypothetical liquid phase based on
        :obj:`EquilibriumState.V_liquids_ref` [m^3/s]
    Qg : float, optional
        Total volumetric flow rate spec for a hypothetical gas phase based on
        :obj:`EquilibriumState.V_gas` [m^3/s]
    Q : float, optional
        Total actual volumetric flow rate of the stream based on the
        density of the stream at the specified conditions [m^3/s]
    n : float, optional
        Total mole flow rate of all components in the stream [mol/s]
    m : float, optional
        Total mass flow rate of all components in the stream [kg/s]
    T : float, optional
        Temperature of the stream, [K]
    P : float, optional
        Pressure of the stream [Pa]
    VF : float, optional
        Vapor fraction (mole basis) of the stream, [-]
    V : float, optional
        Molar volume of the overall stream [m^3/mol]
    rho : float, optional
        Molar density of the overall stream [mol/m^3]
    rho_mass : float, optional
        Mass density of the overall stream [kg/m^3]
    H : float, optional
        Molar enthalpy of the stream [J/mol]
    H_mass : float, optional
        Mass enthalpy of the stream [J/kg]
    S : float, optional
        Molar entropy of the stream [J/mol/K]
    S_mass : float, optional
        Mass entropy of the stream [J/kg/K]
    U : float, optional
        Molar internal energy of the stream [J/mol]
    U_mass : float, optional
        Mass internal energy of the stream [J/kg]
    G : float, optional
        Molar Gibbs free energy of the stream [J/mol]
    G_mass : float, optional
        Mass Gibbs free energy of the stream [J/kg]
    A : float, optional
        Molar Helmholtz energy of the stream [J/mol]
    A_mass : float, optional
        Mass Helmholtz energy of the stream [J/kg]
    energy : float, optional
        Flowing energy of the stream [W]
    energy_reactive : float, optional
        Flowing energy of the stream on a reactive basis [W]
    H_reactive : float, optional
        Reactive molar enthalpy of the stream [J/mol]
    Q_TP : tuple(3, float, float, str), optional
        The (T, P, phase) at which the actual volumetric flow rate (if specified) is
        calculated to be at, [K] and [Pa]
    multiple_composition_basis : bool, optional
        Fun toggle to allow the object to recieve multiple different types
        of composition information - e.g. a partial list of mole fractions
        and a partial list of mass fractions. This can be useful for some
        problems
    '''

    flashed = False
    """`flashed` can be checked to quickly determine if an object is already flashed. It is always False for `StreamArgs`,
    and True for `EquilibriumState` and `EquilibriumStream`"""
    __full_path__ = f"{__module__}.{__qualname__}"

    __slots__ = ('specifications', 'multiple_composition_basis', 'Vf_TP', 'Q_TP', 'flasher', '_state_cache', '_flash')
    obj_references = ('flasher', '_flash')

    as_json = JsonOptEncodable.as_json
    from_json = JsonOptEncodable.from_json
    json_version = 1
    non_json_attributes = []
    vectorized = False

    def __eq__(self, other):
        return self.__hash__() == hash(other)

    def __hash__(self):
        return hash_any_primitive([self.specifications, self.multiple_composition_basis, self.Vf_TP, self.Q_TP, self.flasher])

    def _custom_from_json(self, *args):
        if self._state_cache:
            zs = tuple(self._state_cache[-1])
            self._state_cache = tuple(self._state_cache[0:-1])+(zs,)

    def __init__(self, *, zs=None, ws=None, Vfls=None, Vfgs=None,
                 T=None, P=None,
                 VF=None, H=None, H_mass=None, S=None, S_mass=None,
                 U=None, U_mass=None, G=None, G_mass=None, A=None, A_mass=None,
                 V=None, rho=None, rho_mass=None,

                 ns=None, ms=None, Qls=None, Qgs=None, m=None, n=None, Q=None,
                 Ql=None, Qg=None,
                 energy=None, energy_reactive=None, H_reactive=None,
                 Vf_TP=(None, None), Q_TP=(None, None, ''), flasher=None,
                 multiple_composition_basis=False):
        self.specifications = {'zs': zs, 'ws': ws, 'Vfls': Vfls, 'Vfgs': Vfgs,
                       'ns': ns, 'ms': ms, 'Qls': Qls, 'Qgs': Qgs,
                       'n': n, 'm': m, 'Q': Q, 'Ql': Ql, 'Qg': Qg,

                       'T': T, 'P': P,
                       'V': V, 'rho': rho, 'rho_mass': rho_mass,

                       'VF': VF,
                       'H': H,'H_mass': H_mass,
                       'S': S, 'S_mass': S_mass,
                       'U': U, 'U_mass': U_mass,
                       'A': A, 'A_mass': A_mass,
                       'G': G, 'G_mass': G_mass,
                       'energy': energy, 'energy_reactive': energy_reactive, 'H_reactive': energy_reactive}

        # If this is True, DO NOT CLEAR OTHER COMPOSITION / FLOW VARIABLES WHEN SETTING ONE!
        # This makes sense for certain cases but not all.
        self.multiple_composition_basis = multiple_composition_basis
        self.Vf_TP = Vf_TP
        self.Q_TP = Q_TP
        self.flasher = flasher
        self._state_cache = None

        composition_specs = state_specs = flow_specs = 0
        # Note that these int statements are faster than adding the booleans
        if zs is not None:
            composition_specs += 1
        if ws is not None:
            composition_specs += 1
        if Vfls is not None:
            composition_specs += 1
        if Vfgs is not None:
            composition_specs += 1


        if ns is not None:
            composition_specs += 1
            flow_specs += 1
        if ms is not None:
            composition_specs += 1
            flow_specs += 1
        if Qls is not None:
            composition_specs += 1
            flow_specs += 1
        if Qgs is not None:
            composition_specs += 1
            flow_specs += 1

        if n is not None:
            flow_specs += 1
        if m is not None:
            flow_specs += 1
        if Q is not None:
            flow_specs += 1
        if Ql is not None:
            flow_specs += 1
        if Qg is not None:
            flow_specs += 1

        if T is not None:
            state_specs += 1
        if P is not None:
            state_specs += 1
        if V is not None:
            state_specs += 1
        if rho is not None:
            state_specs += 1
        if rho_mass is not None:
            state_specs += 1
        if VF is not None:
            state_specs += 1
        if H_mass is not None:
            state_specs += 1
        if H is not None:
            state_specs += 1
        if S_mass is not None:
            state_specs += 1
        if S is not None:
            state_specs += 1
        if U_mass is not None:
            state_specs += 1
        if U is not None:
            state_specs += 1
        if G_mass is not None:
            state_specs += 1
        if G is not None:
            state_specs += 1
        if A_mass is not None:
            state_specs += 1
        if A is not None:
            state_specs += 1
        if energy is not None:
            state_specs += 1
        if energy_reactive is not None:
            state_specs += 1
        if H_reactive is not None:
            state_specs += 1
        if flow_specs > 1 or composition_specs > 1:
            self.reconcile_flows()
#            raise ValueError("Flow specification is overspecified")
        if composition_specs > 1 and not multiple_composition_basis:
            raise ValueError("Composition specification is overspecified")
        if state_specs > 2:
            raise ValueError("State specification is overspecified")

    def copy(self):
        """Create a deep copy of the StreamArgs instance. Parameters
        `multiple_composition_basis`, `Vf_TP`, `Q_TP` and `flasher`
        are set to the new object without copying.

        Returns
        -------
        StreamArgs
            A new instance of StreamArgs with the same specifications
            and configuration as the original.

        Examples
        --------
        >>> original_stream = StreamArgs(T=300, P=101325, zs=[0.5, 0.5])
        >>> copied_stream = original_stream.copy()
        >>> # Modify the copy without affecting the original
        >>> copied_stream.T = 350
        >>> original_stream.T
        300
        >>> original_stream.zs is copied_stream.zs
        False
        """
        # multiple_composition_basis may mean multiple sets of specs for comp/flow
        kwargs = self.specifications.copy()
        if kwargs['zs'] is not None:
            kwargs['zs'] = [i for i in kwargs['zs']]
        if kwargs['ws'] is not None:
            kwargs['ws'] = [i for i in kwargs['ws']]
        if kwargs['ns'] is not None:
            kwargs['ns'] = [i for i in kwargs['ns']]
        if kwargs['ms'] is not None:
            kwargs['ms'] = [i for i in kwargs['ms']]
        if kwargs['Qls'] is not None:
            kwargs['Qls'] = [i for i in kwargs['Qls']]
        if kwargs['Qgs'] is not None:
            kwargs['Qgs'] = [i for i in kwargs['Qgs']]
        if kwargs['Vfgs'] is not None:
            kwargs['Vfgs'] = [i for i in kwargs['Vfgs']]
        if kwargs['Vfls'] is not None:
            kwargs['Vfls'] = [i for i in kwargs['Vfls']]
        return StreamArgs(Vf_TP=self.Vf_TP, Q_TP=self.Q_TP, flasher=self.flasher,
                 multiple_composition_basis=self.multiple_composition_basis, **kwargs)

    __copy__ = copy

    @property
    def energy(self):
        r'''Flowing energy of the stream specification if specified [W]'''
        return self.specifications['energy']

    @energy.setter
    def energy(self, energy):
        r'''Set the flowing energy of the stream [W]. This variable can set either the
        flow rate, or act as an enthalpy spec if another flow rate specification is specified.
        '''
        if energy is None:
            self.specifications['energy'] = energy
            return None
        if self.specified_state_vars > 1 and self.flow_specified and self.energy is None:
            raise ValueError('Two state vars and a flow var already specified; unset another first')
        self.specifications['energy'] = energy

    @property
    def energy_reactive(self):
        r'''Flowing energy of the stream on a reactive basis specification if specified [W]'''
        return self.specifications['energy_reactive']
    @energy_reactive.setter
    def energy_reactive(self, energy_reactive):
        r'''Set the flowing energy of the stream on a reactive basis [W]. This variable can set either the
        flow rate, or act as an enthalpy spec if another flow rate specification is specified.
        '''
        if energy_reactive is None:
            self.specifications['energy_reactive'] = energy_reactive
            return None
        if self.specified_state_vars > 1 and self.flow_specified and self.energy_reactive is None:
            raise ValueError('Two state vars and a flow var already specified; unset another first')
        self.specifications['energy_reactive'] = energy_reactive

    @property
    def T(self):
        r'''Temperature of the stream specification if specified [K]'''
        return self.specifications['T']
    @T.setter
    def T(self, T):
        r'''Set the temperature of the stream specification [K]'''
        s = self.specifications
        if T is None:
            s['T'] = T
            return None
        if s['T'] is None and self.state_specified:
            raise ValueError('Two state vars already specified; unset another first')
        s['T'] = T

    @property
    def T_calc(self):
        r'''Temperature of the stream; specified or calculated if enough information is available [K]'''
        T = self.specifications['T']
        if T is not None:
            return T
        try:
            return self.flash_state().T
        except:
            return None

    @property
    def P_calc(self):
        r'''Pressure of the stream; specified or calculated if enough information is available [Pa]'''
        P = self.specifications['P']
        if P is not None:
            return P
        try:
            return self.flash_state().P
        except:
            return None

    @property
    def VF_calc(self):
        r'''Vapor fraction of the stream; specified or calculated if enough information is available [-]'''
        VF = self.specifications['VF']
        if VF is not None:
            return VF
        try:
            return self.flash_state().VF
        except:
            return None

    @property
    def P(self):
        r'''Pressure of the stream specification if specified [Pa]'''
        return self.specifications['P']
    @P.setter
    def P(self, P):
        r'''Set the pressure of the stream specification [Pa]'''
        s = self.specifications
        if P is None:
            s['P'] = None
            return None
        if s['P'] is None and self.state_specified:
            raise ValueError('Two state vars already specified; unset another first')
        s['P'] = P

    @property
    def V(self):
        r'''Molar volume of the stream specification if specified [m^3/mol]'''
        return self.specifications['V']
    @V.setter
    def V(self, V):
        r'''Set the molar volume of the stream specification [m^3/mol]'''
        if V is None:
            self.specifications['V'] = V
            return None
        if self.state_specified and self.V is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['V'] = V

    @property
    def rho(self):
        r'''Molar density of the stream specification if specified [mol/m^3]'''
        return self.specifications['rho']
    @rho.setter
    def rho(self, rho):
        r'''Set the molar density of the stream specification [m^3/mol]'''
        if rho is None:
            self.specifications['rho'] = rho
            return None
        if self.state_specified and self.rho is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['rho'] = rho

    @property
    def rho_mass(self):
        r'''Mass density of the stream specification if specified [kg/m^3]'''
        return self.specifications['rho_mass']
    @rho_mass.setter
    def rho_mass(self, rho_mass):
        r'''Set the mass density of the stream specification [kg/m^3]'''
        if rho_mass is None:
            self.specifications['rho_mass'] = rho_mass
            return None
        if self.state_specified and self.rho_mass is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['rho_mass'] = rho_mass

    @property
    def VF(self):
        r'''Vapor fraction of the stream specification if specified [-]'''
        return self.specifications['VF']
    @VF.setter
    def VF(self, VF):
        r'''Set the vapor fraction of the stream specification [-]'''
        if VF is None:
            self.specifications['VF'] = VF
            return None
        if self.state_specified and self.VF is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['VF'] = VF

    @property
    def H(self):
        r'''Molar enthalpy of the stream specification if specified [J/mol]'''
        return self.specifications['H']

    @H.setter
    def H(self, H):
        r'''Set the molar enthalpy of the stream specification [J/mol]'''
        if H is None:
            self.specifications['H'] = H
            return None
        if self.state_specified and self.H is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['H'] = H

    @property
    def H_calc(self):
        r'''Molar enthalpy of the stream; specified or calculated if enough information is available [J/mol]'''
        H = self.specifications['H']
        if H is not None:
            return H
        try:
            return self.flash_state().H()
        except:
            return None

    @property
    def H_reactive(self):
        r'''Molar reactive enthalpy of the stream specification if specified [J/mol]'''
        return self.specifications['H_reactive']
    @H_reactive.setter
    def H_reactive(self, H_reactive):
        r'''Set the molar reactive enthalpy of the stream specification [J/mol]'''
        if H_reactive is None:
            self.specifications['H_reactive'] = H_reactive
            return None
        if self.state_specified and self.H_reactive is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['H_reactive'] = H_reactive

    @property
    def H_reactive_calc(self):
        r'''Molar reactive enthalpy of the stream; specified or calculated if enough information is available [J/mol]'''
        H_reactive = self.specifications['H_reactive']
        if H_reactive is not None:
            return H_reactive
        try:
            return self.flash_state().H_reactive()
        except:
            return None

    @property
    def H_mass(self):
        r'''Mass enthalpy of the stream specification if specified [J/kg]'''
        return self.specifications['H_mass']
    @H_mass.setter
    def H_mass(self, H_mass):
        r'''Set the mass enthalpy of the stream specification [J/kg]'''
        if H_mass is None:
            self.specifications['H_mass'] = H_mass
            return None
        if self.specified_state_vars > 1 and self.specifications['H_mass'] is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['H_mass'] = H_mass

    @property
    def H_mass_calc(self):
        r'''Mass enthalpy of the stream; specified or calculated if enough information is available [J/kg]'''
        H_mass = self.specifications['H_mass']
        if H_mass is not None:
            return H_mass
        try:
            return self.flash_state().H_mass()
        except:
            return None

    @property
    def U(self):
        r'''Molar internal energy of the stream specification if specified [J/mol]'''
        return self.specifications['U']
    @U.setter
    def U(self, U):
        r'''Set the molar internal energy of the stream specification [J/mol]'''
        if U is None:
            self.specifications['U'] = U
            return None
        if self.state_specified and self.U is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['U'] = U

    @property
    def U_mass(self):
        r'''Mass internal energy of the stream specification if specified [J/kg]'''
        return self.specifications['U_mass']
    @U_mass.setter
    def U_mass(self, U_mass):
        r'''Set the mass internal energy of the stream specification [J/kg]'''
        if U_mass is None:
            self.specifications['U_mass'] = U_mass
            return None
        if self.specified_state_vars > 1 and self.U_mass is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['U_mass'] = U_mass

    @property
    def G(self):
        r'''Molar Gibbs free energy of the stream specification if specified [J/mol]'''
        return self.specifications['G']
    @G.setter
    def G(self, G):
        r'''Set the molar Gibbs free energy of the stream specification [J/mol]'''
        if G is None:
            self.specifications['G'] = G
            return None
        if self.state_specified and self.G is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['G'] = G

    @property
    def G_mass(self):
        r'''Mass Gibbs free energy of the stream specification if specified [J/kg]'''
        return self.specifications['G_mass']
    @G_mass.setter
    def G_mass(self, G_mass):
        r'''Set the mass Gibbs free energy of the stream specification [J/kg]'''
        if G_mass is None:
            self.specifications['G_mass'] = G_mass
            return None
        if self.specified_state_vars > 1 and self.G_mass is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['G_mass'] = G_mass

    @property
    def A(self):
        r'''Molar Hemholtz free energy of the stream specification if specified [J/mol]'''
        return self.specifications['A']
    @A.setter
    def A(self, A):
        r'''Set the molar Hemholtz free energy of the stream specification [J/mol]'''
        if A is None:
            self.specifications['A'] = A
            return None
        if self.state_specified and self.A is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['A'] = A

    @property
    def A_mass(self):
        r'''Mass Hemholtz free energy of the stream specification if specified [J/kg]'''
        return self.specifications['A_mass']
    @A_mass.setter
    def A_mass(self, A_mass):
        r'''Set the mass Hemholtz free energy of the stream specification [J/kg]'''
        if A_mass is None:
            self.specifications['A_mass'] = A_mass
            return None
        if self.specified_state_vars > 1 and self.A_mass is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['A_mass'] = A_mass

    @property
    def S(self):
        r'''Molar entropy of the stream specification if specified [J/(mol*K)]'''
        return self.specifications['S']
    @S.setter
    def S(self, S):
        r'''Set the molar entropy of the stream specification [J/(mol*K)]'''
        if S is None:
            self.specifications['S'] = S
            return None
        if self.specified_state_vars > 1 and self.S is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['S'] = S

    @property
    def S_mass(self):
        r'''Mass entropy of the stream specification if specified [J/(kg*K)]'''
        return self.specifications['S_mass']
    @S_mass.setter
    def S_mass(self, S_mass):
        r'''Set the mass entropy of the stream specification [J/(kg*K)]'''
        if S_mass is None:
            self.specifications['S_mass'] = S_mass
            return None
        if self.specified_state_vars > 1 and self.S_mass is None:
            raise ValueError('Two state vars already specified; unset another first')
        self.specifications['S_mass'] = S_mass

    @property
    def zs(self):
        r'''Mole fractions of the stream if specified [-]'''
        return self.specifications['zs']
    @zs.setter
    def zs(self, arg):
        r'''Set the mole fractions of the stream [-]'''
        s = self.specifications
        if arg is None:
            s['zs'] = arg
        else:
            if not arg: # force empty list to None
                arg = None
            if not self.multiple_composition_basis:
                composition_spec = self.composition_spec
                if composition_spec is not None and composition_spec[0] != 'zs':
                    raise ValueError('Another composition spec already specified; unset another first')
                s['zs'] = arg
                s['ws'] = s['Vfls'] = s['Vfgs'] = s['ns'] = s['ms'] = s['Qls'] = s['Qgs'] = None
            else:
                s['zs'] = arg

    @property
    def zs_calc(self):
        r'''Mole fractions of the stream; specified or calculated if enough information is available [-]'''
        # This forms the basis for the calculations
        s = self.specifications
        zs = s['zs']
        if zs is not None:
            if not self.multiple_composition_basis:
                return zs
            else:
                if None not in zs:
                    return zs
                return None
        ns = s['ns']
        if ns is not None:
            if not self.multiple_composition_basis:
                return normalize(ns)

        ws = s['ws']
        if ws is not None and None not in ws:
            MWs = self.flasher.constants.MWs
            try:
                return ws_to_zs(ws, MWs)
            except ZeroDivisionError:
                pass
        Vfls = s['Vfls']
        if Vfls is not None and None not in Vfls:
            Vms = self.flasher.V_liquids_ref()
            try:
                return Vfs_to_zs(Vfls, Vms)
            except ZeroDivisionError:
                pass
        Vfgs = s['Vfgs']
        if Vfgs is not None and None not in Vfgs:
            return Vfgs


        ms = s['ms']
        if ms is not None and None not in ms:
            MWs = self.flasher.constants.MWs
            return ws_to_zs(normalize(ms), MWs)

        Qls = s['Qls']
        if Qls is not None and None not in Qls:
            Vms = self.flasher.V_liquids_ref()
            return Vfs_to_zs(normalize(Qls), Vms)

        Qgs = s['Qgs']
        if Qgs is not None and None not in Qgs:
            return normalize(Qgs)
        return None

    @property
    def ws(self):
        r'''Mass fractions of the stream if specified [-]'''
        return self.specifications['ws']
    @ws.setter
    def ws(self, arg):
        r'''Set the mass fractions of the stream [-]'''
        s = self.specifications
        if arg is None:
            s['ws'] = arg
        else:
            # enforce a length
            if not arg:
                arg = None
            if not self.multiple_composition_basis:
                composition_spec = self.composition_spec
                if composition_spec is not None and composition_spec[0] != 'ws':
                    raise ValueError('Another composition spec already specified; unset another first')
                s['ws'] = arg
                s['zs'] = s['Vfls'] = s['Vfgs'] = s['ns'] = s['ms'] = s['Qls'] = s['Qgs'] = None
            else:
                s['ws'] = arg

    @property
    def ws_calc(self):
        r'''Mass fractions of the stream; specified or calculated if enough information is available [-]'''
        ws = self.specifications['ws']
        if ws is not None:
            return ws
        zs = self.zs_calc
        if zs is not None:
            MWs = self.flasher.constants.MWs
            return zs_to_ws(zs, MWs)

    @property
    def Vfls(self):
        r'''Liquid volume fractions of the stream if specified [-]'''
        return self.specifications['Vfls']
    @Vfls.setter
    def Vfls(self, arg):
        r'''Set the liquid volume fractions of the stream [-]'''
        s = self.specifications
        if arg is None:
            s['Vfls'] = arg
        else:
            # enforce a length
            if not arg:
                arg = None
            if not self.multiple_composition_basis:
                composition_spec = self.composition_spec
                if composition_spec is not None and composition_spec[0] != 'Vfls':
                    raise ValueError('Another composition spec already specified; unset another first')
                s['Vfls'] = arg
                s['zs'] = s['ws'] = s['Vfgs'] = s['ns'] = s['ms'] = s['Qls'] = s['Qgs'] = None
            else:
                s['Vfls'] = arg


    @property
    def Vfls_calc(self):
        r'''Liquid volume fractions of the stream; specified or calculated if enough information is available [-]'''
        Vfls = self.specifications['Vfls']
        if Vfls is not None:
            return Vfls
        zs = self.zs_calc
        if zs is not None:
            Vms = self.flasher.V_liquids_ref()
            return zs_to_Vfs(zs, Vms)


    @property
    def Vfgs(self):
        r'''Gas volume fractions of the stream if specified [-]'''
        return self.specifications['Vfgs']
    @Vfgs.setter
    def Vfgs(self, arg):
        r'''Set the gas volume fractions of the stream [-]'''
        s = self.specifications
        if arg is None:
            s['Vfgs'] = arg
        else:
            # enforce a length
            if not arg:
                arg = None
            if not self.multiple_composition_basis:
                composition_spec = self.composition_spec
                if composition_spec is not None and composition_spec[0] != 'Vfgs':
                    raise ValueError('Another composition spec already specified; unset another first')
                s['Vfgs'] = arg
                s['zs'] = s['ws'] = s['Vfls'] = s['ns'] = s['ms'] = s['Qls'] = s['Qgs'] = None
            else:
                s['Vfgs'] = arg
    @property
    def ns(self):
        r'''Mole flows of the stream if specified [mol/s]'''
        return self.specifications['ns']

    @ns.setter
    def ns(self, arg):
        r'''Set the mole flows of the stream [mol/s]'''
        s = self.specifications
        if arg is None:
            s['ns'] = arg
        else:
            # enforce a length
            if not arg:
                arg = None
            if not self.multiple_composition_basis:
                s['zs'] = s['ws'] = s['Vfls'] = s['Vfgs'] = s['ms'] = s['Qls'] = s['Qgs'] = s['n'] = s['m'] = s['Q'] = s['Ql'] = s['Qg'] = None
                s['ns'] = arg
            else:
                s['ns'] = arg

    @property
    def ns_calc(self):
        r'''Mole flows of the stream; specified or calculated if enough information is available [mol/s]'''
        s = self.specifications
        ns = s['ns']
        if ns is not None:
            if not self.multiple_composition_basis:
                return ns
            else:
                if None not in ns:
                    return ns
                return None
        n = s['n']
        if n is not None:
            zs = self.zs_calc
            if zs is not None:
                return [n*zi for zi in zs]
        m = s['m']
        if m is not None:
            zs = self.zs_calc
            if zs is not None:
                MWs = self.flasher.constants.MWs
                MW = mixing_simple(MWs, zs)
                n = property_molar_to_mass(m, MW)
                return [n*zi for zi in zs]
        ms = s['ms']
        if ms is not None and None not in ms:
            zs = self.zs_calc
            m = sum(ms)
            MWs = self.flasher.constants.MWs
            MW = mixing_simple(MWs, zs)
            n = m*1000.0/MW
            return [n*zi for zi in zs]
        Qls = s['Qls']
        if Qls is not None and None not in Qls:
            Vms = self.flasher.V_liquids_ref()
            return [Ql/Vm for Vm, Ql in zip(Vms, Qls)]
        Ql = s['Ql']
        if Ql is not None:
            Vms = self.flasher.V_liquids_ref()
            zs = self.zs_calc
            Vfs = zs_to_Vfs(zs, Vms)
            return [Ql*Vf/Vm for Vm, Vf in zip(Vms, Vfs)]
        Qgs = s['Qgs']
        if Qgs is not None and None not in Qgs:
            flasher = self.flasher
            V = R*flasher.settings.T_gas_ref/flasher.settings.P_gas_ref
            return [Qgi/V for Qgi in Qgs]
        Qg = s['Qg']
        if Qg is not None:
            flasher = self.flasher
            V = R*flasher.settings.T_gas_ref/flasher.settings.P_gas_ref
            n = Qg/V
            zs = self.zs_calc
            return [zi*n for zi in zs]
        Q = s['Q']
        if Q is not None:
            zs = self.zs_calc
            if zs is not None:
                Q_TP = self.Q_TP
                if Q_TP is not None and (Q_TP[0] is not None and Q_TP[1] is not None):
                    if (len(Q_TP) == 2 or (len(Q_TP) == 3 and not Q_TP[-1])):
                        # Calculate the volume via the property package
                        expensive_flash = self.flasher.flash(zs=zs, T=Q_TP[0], P=Q_TP[1])
                        V = expensive_flash.V()
                    if Q_TP[-1] == 'l':
                        V = self.flasher.liquids[0].to(T=Q_TP[0], P=Q_TP[1], zs=zs).V()
                    elif Q_TP[-1] == 'g':
                        V = self.flasher.gas.to(T=Q_TP[0], P=Q_TP[1], zs=zs).V()
                else:
                    mixture = self.flash_state()
                    if mixture is not None:
                        V = mixture.V()
                if V is not None:
                    n = Q/V
                    return [n*zi for zi in zs]
        return None

    @property
    def ms(self):
        r'''Mass flows of the stream if specified [kg/s]'''
        return self.specifications['ms']
    @ms.setter
    def ms(self, arg):
        r'''Set the mass flows of the stream [kg/s]'''
        s = self.specifications
        if arg is None:
            s['ms'] = arg
        else:
            if not self.multiple_composition_basis:
                s['zs'] = s['ws'] = s['Vfls'] = s['Vfgs'] = s['ns'] = s['Qls'] = s['Qgs'] = s['n'] = s['m'] = s['Q'] = s['Ql'] = s['Qg'] = None
                s['ms'] = arg
            else:
                s['ms'] = arg

    @property
    def ms_calc(self):
        r'''Mass flows of the stream; specified or calculated if enough information is available [kg/s]'''
        ns = self.ns_calc
        if ns is not None:
            zs = self.zs_calc
            n = sum(ns)
            MW = self.MW
            m = property_mass_to_molar(n, MW)
            MW_inv = 1.0/MW
            MWs = self.flasher.constants.MWs
            ws = [zi*MWi*MW_inv for zi, MWi in zip(zs, MWs)]
            return [m*wi for wi in ws]

    @property
    def Qls(self):
        r'''Standard liquid volume flows of the stream if specified [m^3/s]'''
        return self.specifications['Qls']
    @Qls.setter
    def Qls(self, arg):
        r'''Set the standard liquid volume flows of the stream [m^3/s]'''
        s = self.specifications
        if arg is None:
            s['Qls'] = arg
        else:
            if not self.multiple_composition_basis:
                s['zs'] = s['ws'] = s['Vfls'] = s['Vfgs'] = s['ms'] = s['ns'] = s['Qgs'] = s['n'] = s['m'] = s['Q'] = s['Ql'] = s['Qg'] = None
                s['Qls'] = arg
            else:
                s['Qls'] = arg

    @property
    def Qls_calc(self):
        r'''Standard liquid volume flows of the stream; specified or calculated if enough information is available [m^3/s]'''
        ns_calc = self.ns_calc
        if ns_calc is not None:
            Vms = self.flasher.V_liquids_ref()
            Qls = [ni*Vm for ni, Vm in zip(ns_calc, Vms)]
            return Qls

    @property
    def Qgs(self):
        r'''Standard gas volume flows of the stream if specified [m^3/s]'''
        return self.specifications['Qgs']
    @Qgs.setter
    def Qgs(self, arg):
        r'''Set the standard gas volume flows of the stream [m^3/s]'''
        if arg is None:
            self.specifications['Qgs'] = arg
        else:
            if not self.multiple_composition_basis:
                args = {'zs': None, 'ws': None, 'Vfls': None, 'Vfgs': None,
                        'ns': None, 'ms': None, 'Qls': None, 'Qgs': arg,
                        'n': None, 'm': None, 'Q': None, 'Ql': None, 'Qg': None}
                self.specifications.update(args)
            else:
                self.specifications['Qgs'] = arg

    @property
    def Qgs_calc(self):
        r'''Standard gas volume flows of the stream; specified or calculated if enough information is available [m^3/s]'''
        ns_calc = self.ns_calc
        if ns_calc is not None:
            flasher = self.flasher
            V = R*flasher.settings.T_gas_ref/flasher.settings.P_gas_ref
            Qgs = [ni*V for ni in ns_calc]
            return Qgs

    @property
    def m(self):
        r'''Mass flow of the stream if specified [kg/s]'''
        return self.specifications['m']
    @m.setter
    def m(self, arg):
        r'''Set the mass flow of the stream [kg/s]'''
        s = self.specifications
        if arg is None:
            s['m'] = arg
        else:
            flow_spec = self.flow_spec
            if flow_spec is not None and flow_spec[0] != 'm':
                raise ValueError('Another flow spec already specified; unset another first')
            s['Qls'] = s['ms'] = s['ns'] = s['Qgs'] = s['n'] = s['m'] = s['Q'] = s['Ql'] = s['Qg'] = None
            s['m'] = arg

    @property
    def m_calc(self):
        r'''Mass flow of the stream; specified or calculated if enough information is available [kg/s]'''
        m = self.specifications['m']
        if m is not None:
            return m
        ms = self.specifications['ms']
        if ms is not None and None not in ms:
            return sum(ms)
        ms_calc = self.ms_calc
        if ms_calc is not None:
            return sum(ms_calc)
        return None

    @property
    def n(self):
        r'''Mole flow of the stream if specified [mol/s]'''
        return self.specifications['n']
    @n.setter
    def n(self, arg):
        r'''Set the mole flow of the stream [mol/s]'''
        s = self.specifications
        if arg is None:
            s['n'] = arg
        else:
            flow_spec = self.flow_spec
            if flow_spec is not None and flow_spec[0] != 'n':
                raise ValueError('Another flow spec already specified; unset another first')
            s['Qls'] = s['ms'] = s['ns'] = s['Qgs'] = s['n'] = s['m'] = s['Q'] = s['Ql'] = s['Qg'] = None
            s['n'] = arg

    @property
    def n_calc(self):
        r'''Mole flow of the stream; specified or calculated if enough information is available [mol/s]'''
        s = self.specifications
        n = s['n']
        if n is not None:
            return n
        ns = s['ns']
        if ns is not None and None not in ns:
            return sum(ns)
        # Everything funnels into ns_calc to avoid conflicts
        ns_calc = self.ns_calc
        if ns_calc is not None and None not in ns_calc:
            return sum(ns_calc)

        return None

    @property
    def Ql(self):
        r'''Standard liquid volume flow of the stream if specified [m^3/s]'''
        return self.specifications['Ql']
    @Ql.setter
    def Ql(self, arg):
        r'''Set the standard liquid volume flow of the stream [m^3/s]'''
        s = self.specifications
        if arg is None:
            s['Ql'] = arg
        else:
            flow_spec = self.flow_spec
            if flow_spec is not None and flow_spec[0] != 'Ql':
                raise ValueError('Another flow spec already specified; unset another first')
            s['Qls'] = s['ms'] = s['ns'] = s['Qgs'] = s['n'] = s['m'] = s['Q'] = s['Ql'] = s['Qg'] = None
            s['Ql'] = arg

    @property
    def Ql_calc(self):
        r'''Standard liquid volume flow of the stream; specified or calculated if enough information is available [m^3/s]'''
        Ql = self.specifications['Ql']
        if Ql is not None:
            return Ql
        Qls = self.specifications['Qls']
        if Qls is not None and None not in Qls:
            return sum(Qls)
        Qls_calc = self.Qls_calc
        if Qls_calc is not None:
            return sum(Qls_calc)
        return None

    @property
    def Qg(self):
        r'''Standard gas volume flow of the stream if specified [m^3/s]'''
        return self.specifications['Qg']
    @Qg.setter
    def Qg(self, arg):
        r'''Set the standard gas volume flow of the stream [m^3/s]'''
        s = self.specifications
        if arg is None:
            s['Qg'] = arg
        else:
            flow_spec = self.flow_spec
            if flow_spec is not None and flow_spec[0] != 'Qg':
                raise ValueError('Another flow spec already specified; unset another first')
            s['Qls'] = s['ms'] = s['ns'] = s['Qgs'] = s['n'] = s['m'] = s['Q'] = s['Ql'] = s['Qg'] = None
            s['Qg'] = arg

    @property
    def Qg_calc(self):
        r'''Standard gas volume flow of the stream; specified or calculated if enough information is available [m^3/s]'''
        Qg = self.specifications['Qg']
        if Qg is not None:
            return Qg
        Qgs = self.specifications['Qgs']
        if Qgs is not None and None not in Qgs:
            return sum(Qgs)
        Qgs_calc = self.Qgs_calc
        if Qgs_calc is not None:
            return sum(Qgs_calc)
        return None

    @property
    def Q(self):
        r'''Actual volume flow of the stream if specified [m^3/s]'''
        return self.specifications['Q']
    @Q.setter
    def Q(self, arg):
        r'''Set the actual volume flow of the stream [m^3/s]'''
        s = self.specifications
        if arg is None:
            s['Q'] = arg
        else:
            flow_spec = self.flow_spec
            if flow_spec is not None and flow_spec[0] != 'Q':
                raise ValueError('Another flow spec already specified; unset another first')
            s['Qls'] = s['ms'] = s['ns'] = s['Qgs'] = s['n'] = s['m'] = s['Q'] = s['Ql'] = s['Qg'] = None
            s['Q'] = arg

    @property
    def MW(self):
        r'''Molecular weight of the stream; calculated if enough information is available [g/mol]'''
        try:
            MWs = self.flasher.constants.MWs
            zs = self.zs_calc
            MW = mixing_simple(MWs, zs)
            return MW
        except:
            return None

    @property
    def energy_calc(self):
        r'''Energy flow of the stream; specified or calculated if enough information is available [W]'''
        s = self.specifications
        # Try to get H from energy, or a molar specification
        Q = s['energy']
        m, n = None, None
        if Q is None:
            H = s['H']
            if H is not None:
                n = s['n']
                if n is None:
                    n = self.n_calc
                if n is not None:
                    Q = n*H
        # Try to get H from a mass specification
        if Q is None:
            H_mass = s['H_mass']
            if H_mass is not None:
                m = s['m']
                if m is None:
                    m = self.m_calc
                if m is not None:
                    Q = m*H_mass
        # Try to flash and get enthalpy

        if Q is None:
            n = self.n_calc
            if n is None:
                m = self.m_calc
            if m is not None or n is not None:
                mixture = self.flash_state()
                if mixture is not None:
                    if n is not None:
                        Q = mixture.H()*n
                    elif m is not None:
                        Q = mixture.H()*property_molar_to_mass(m, mixture.MW())
        return Q

    @property
    def energy_reactive_calc(self):
        r'''Energy flow of the stream; specified or calculated if enough information is available [W]'''
        return self.specifications['energy_reactive']


    def __repr__(self):
        s = f'{self.__class__.__name__}(flasher={self.flasher is not None}, '
        for k, v in self.specifications.items():
            if v is not None:
                s += f'{k}={v!r}, '
        s = s[:-2]
        s += ')'
        return s

    def reconcile_flows(self, n_tol=2e-15, m_tol=2e-15):
        """Evaluate the flow specifications provided to the StreamArgs instance
        in the event the original inputs are overspecified, in the assumption that
        the inputs are internally consistent. If they are not consistent to the
        tolerances of this function, the :obj:`StreamArgs` instance will raise an
        error.

        Examples this function can check are both `ms` and `m` provided,
        or `ms` with one missing value and `m` provided (`ms` will be modified
        with the calculated value of the previously missing flow), if both
        `ms` and `ns` are set but with only some values in `ms` and some in `ns`
        (the spec will be set as `ns` and calculate as many values as possible),
        and the rather absurd case of specifying various inputs of `zs`, `ns`, and `ws`
        and expecting the software to make sense of it. Obviously if no `flasher`
        is available with molecular weights the number of checks that can be performed
        are limited.

        Parameters
        ----------
        n_tol : float, optional
            The tolerance for checking the consistency of mole flow rates.
            Default is 2e-15.
        m_tol : float, optional
            The tolerance for checking the consistency of mass flow rates.
            Default is 2e-15.


        Examples
        --------
        >>> stream = StreamArgs(ns=[1.0, None], n=6.0)
        >>> stream.ns
        [1.0, 5.0]

        >>> stream = StreamArgs(ms=[4.0, None], m=10.0)
        >>> stream.ms
        [4.0, 6.0]

        Notes
        -----
        """
        s = self.specifications
        n, m, Q = s['n'], s['m'], s['Q']
        if n is not None:
            if m is not None:
                raise OverspeficiedError(f"Flow specification is overspecified: n={n:g}, m={m:g}")
            elif Q is not None:
                raise OverspeficiedError(f"Flow specification is overspecified: n={n:g}, Q={Q:g}")
        elif m is not None and Q is not None:
            raise OverspeficiedError(f"Flow specification is overspecified: m={m:g}, Q={Q:g}")

        ns, zs, ms, ws = s['ns'], s['zs'], s['ms'], s['ws']
        if n is not None and ns is not None:
            calc = 0.0
            missing = 0
            missing_idx = None
            for i in range(len(ns)):
                if ns[i] is None:
                    missing += 1
                    missing_idx = i
                else:
                    calc += ns[i]
            if missing == 0:
                if abs((calc - n)/n) > n_tol:
                    raise ValueError("Flow specification is overspecified and inconsistent")
            elif missing == 1:
                ns[missing_idx] = n - calc

        if m is not None and ms is not None:
            calc = 0.0
            missing = 0
            missing_idx = None
            for i in range(len(ms)):
                if ms[i] is None:
                    missing += 1
                    missing_idx = i
                else:
                    calc += ms[i]
            if missing == 0:
                if abs((calc - m)/m) > m_tol:
                    raise ValueError("Flow specification is overspecified and inconsistent")
            elif missing == 1:
                ms[missing_idx] = m - calc
        if ns is not None and ms is not None:
            try:
            # Convert any ms to ns
                MWs = self.flasher.constants.MWs
            except:
                return False
            for i in range(len(ms)):
                if ms[i] is not None:
                    ni = property_molar_to_mass(ms[i], MWs[i])
                    if ns[i] is not None and abs((ns[i]  - ni)/ni) > n_tol:
                        raise ValueError("Flow specification is overspecified and inconsistent on component %d" %i)
                    else:
                        ns[i] = ni

        if (zs is not None or ns is not None) and (ws is not None or ms is not None) and (m is not None or n is not None or ns is not None or ms is not None):
            # We need the MWs
            try:
                MWs = self.flasher.constants.MWs
                if zs is None:
                    zs = [None]*len(MWs)
                if ws is None:
                    ws = [None]*len(MWs)
                ns, zs, ws = solve_flow_composition_mix(ns, zs, ws, MWs)
                s['ns'] = ns
            except:
                return False


    def clear_composition_spec(self):
        '''Removes any composition specification(s) that are set, otherwise
        does nothing.

        >>> S = StreamArgs(T=400, H=500, ws=[.3, .7])
        >>> S.clear_composition_spec()
        >>> S.ws
        '''
        s = self.specifications
        s['ns'] = s['zs'] = s['ws'] = s['Vfls'] = s['Vfgs'] = s['ms'] = s['Qls'] = s['Qgs'] = None

    @property
    def composition_spec(self):
        '''Composition specification if one has been set, as a tuples
        of ('specification_name', specification_value); otherwise None.

        >>> StreamArgs(T=400, H=500, ws=[.3, .7]).composition_spec
        ('ws', [0.3, 0.7])
        >>> StreamArgs(T=400, H=500).composition_spec
        >>> StreamArgs(T=400, H=500, Qls=[300.0, 1200.0]).composition_spec
        ('Qls', [300.0, 1200.0])
        '''
        s = self.specifications
        if s['zs'] is not None:
            return 'zs', s['zs']
        if s['ws'] is not None:
            return 'ws', s['ws']
        if s['Vfls'] is not None:
            return 'Vfls', s['Vfls']
        if s['Vfgs'] is not None:
            return 'Vfgs', s['Vfgs']
        if s['ns'] is not None:
            return 'ns', s['ns']
        if s['ms'] is not None:
            return 'ms', s['ms']
        if s['Qls'] is not None:
            return 'Qls', s['Qls']
        if s['Qgs'] is not None:
            return 'Qgs', s['Qgs']

    @property
    def clean(self):
        '''If no variables have been specified, True,
        otherwis False.
        '''
        return not (self.composition_specified or self.state_specs or self.flow_specified)


    @property
    def composition_specified(self):
        '''True if a composition for the stream has been defined.
        If component flow rates are the specification, requires all of them to be specified.
        If `zs` or `ws` or Vfls` or `Vfgs` is set, also requires all of them to be specified
        and additionally them not to be an empty list.
        '''
        s = self.specifications
        if s['zs'] is not None and None not in s['zs'] and sum(s['zs']) != 0.0:
            return True
        if s['ws'] is not None and None not in s['ws'] and sum(s['ws']) != 0.0:
            return True
        if s['Vfls'] is not None and None not in s['Vfls'] and sum(s['Vfls']) != 0.0:
            return True
        if s['Vfgs'] is not None and None not in s['Vfgs'] and sum(s['Vfgs']) != 0.0:
            return True
        if s['ns'] is not None and None not in s['ns']:
            return True
        if s['ms'] is not None and None not in s['ms']:
            return True
        if s['Qls'] is not None and None not in s['Qls']:
            return True
        return bool(s['Qgs'] is not None and None not in s['Qgs'])

    def clear_state_specs(self):
        '''Removes any state specification(s) that are set, otherwise
        does nothing. Composition is not considered a state spec in
        this class.

        >>> S = StreamArgs(T=400, H=500, ws=[.3, .7])
        >>> S.clear_state_specs()
        >>> S.T
        >>> S.H
        '''
        s = self.specifications
        for v in ('T', 'P', 'V', 'rho', 'rho_mass',
                'VF', 'H_mass', 'H', 'S_mass', 'S',
                'U_mass', 'U', 'A_mass', 'A', 'G_mass', 'G',
                'energy', 'energy_reactive'):
            s[v] = None

    @property
    def state_specs(self):
        '''List of state specifications which have been set, as tuples
        of ('specification_name', specification_value). Energy is
        included regardless of whether it is being used as a state
        specification.

        >>> StreamArgs(T=400, H=500).state_specs
        [('T', 400), ('H', 500)]
        '''
        s = self.specifications
        specs = []
        if s['T'] is not None:
            specs.append(('T', s['T']))
        if s['P'] is not None:
            specs.append(('P', s['P']))
        if s['V'] is not None:
            specs.append(('V', s['V']))
        if s['rho'] is not None:
            specs.append(('rho', s['rho']))
        if s['rho_mass'] is not None:
            specs.append(('rho_mass', s['rho_mass']))
        if s['VF'] is not None:
            specs.append(('VF', s['VF']))
        if s['H_mass'] is not None:
            specs.append(('H_mass', s['H_mass']))
        if s['H'] is not None:
            specs.append(('H', s['H']))
        if s['S_mass'] is not None:
            specs.append(('S_mass', s['S_mass']))
        if s['S'] is not None:
            specs.append(('S', s['S']))
        if s['U_mass'] is not None:
            specs.append(('U_mass', s['U_mass']))
        if s['U'] is not None:
            specs.append(('U', s['U']))
        if s['G_mass'] is not None:
            specs.append(('G_mass', s['G_mass']))
        if s['G'] is not None:
            specs.append(('G', s['G']))
        if s['A_mass'] is not None:
            specs.append(('A_mass', s['A_mass']))
        if s['A'] is not None:
            specs.append(('A', s['A']))
        if s['energy'] is not None:
            specs.append(('energy', s['energy']))
        if s['energy_reactive'] is not None:
            specs.append(('energy_reactive', s['energy_reactive']))
        return specs

    @property
    def non_pressure_state_specs(self):
        '''List of state specifications which have been set, as tuples
        of ('specification_name', specification_value), excluding pressure.
        Energy is included regardless of whether it is being used as a state
        specification.

        >>> StreamArgs(T=400, P=1e5, n=4).non_pressure_state_specs
        [('T', 400)]
        '''
        specs = self.state_specs
        return [(s, v) for s, v in specs if s != 'P']

    @property
    def specified_state_vars(self):
        '''Number of state specifications which have been set.
        Energy is included regardless of whether it is being used as a state
        specification.

        >>> StreamArgs(T=400, P=1e5, n=4).specified_state_vars
        2
        '''
        # Slightly faster
        s = self.specifications
        return sum(s[i] is not None for i in ('T', 'P', 'V', 'rho', 'rho_mass',
                                                                'VF', 'H_mass', 'H', 'S_mass', 'S',
                                                                'U_mass', 'U', 'A_mass', 'A', 'G_mass', 'G',
                                                                'energy', 'energy_reactive'))
#        return sum(i is not None for i in (self.T, self.P, self.VF, self.Hm, self.H, self.Sm, self.S, self.energy))

    @property
    def state_specified(self):
        '''Whether or not the state has been fully specified.
        Energy is included regardless of whether it is being used as a state
        specification.

        >>> StreamArgs(T=400, P=1e5, n=4).state_specified
        True
        >>> StreamArgs(P=1e5, n=4).state_specified
        False
        '''
        s = self.specifications
        state_vars = 0
        if s['T'] is not None:
            state_vars += 1
        if s['P'] is not None:
            state_vars += 1
        if s['V'] is not None:
            state_vars += 1
        if s['rho'] is not None:
            state_vars += 1
        if s['rho_mass'] is not None:
            state_vars += 1
        if s['VF'] is not None:
            state_vars += 1
        if s['H_mass'] is not None:
            state_vars += 1
        if s['H'] is not None:
            state_vars += 1
        if s['S'] is not None:
            state_vars += 1
        if s['S_mass'] is not None:
            state_vars += 1
        if s['U'] is not None:
            state_vars += 1
        if s['U_mass'] is not None:
            state_vars += 1
        if s['G'] is not None:
            state_vars += 1
        if s['G_mass'] is not None:
            state_vars += 1
        if s['A'] is not None:
            state_vars += 1
        if s['A_mass'] is not None:
            state_vars += 1
        if s['energy'] is not None:
            state_vars += 1
        if s['energy_reactive'] is not None:
            state_vars += 1
        if s['H_reactive'] is not None:
            state_vars += 1
        return state_vars == 2

    @property
    def non_pressure_spec_specified(self):
        '''Whether there is at least one state specification excluding a pressure
        specification.
        Energy is included regardless of whether it is being used as a state
        specification.

        >>> StreamArgs(T=400, P=1e5, n=4).non_pressure_spec_specified
        True
        >>> StreamArgs(S=400.0, VF=.3, n=4).non_pressure_spec_specified
        True
        >>> StreamArgs(P=1e5, n=4).non_pressure_spec_specified
        False
        '''
        state_vars = (i is not None for i in (self.T, self.VF, self.V, self.rho, self.rho_mass,
                                              self.H_mass, self.H, self.S_mass, self.S,
                                              self.U_mass, self.U, self.A_mass, self.A, self.G_mass, self.G,
                                              self.energy, self.energy_reactive, self.H_reactive))
        return sum(state_vars) >= 1


    def clear_flow_spec(self):
        '''Removes any flow specification(s) that are set, otherwise
        does nothing. This will also remove the composition spec
        if it is coming from `ns`, `ms`, `Qls`, or `Qgs`.

        >>> S = StreamArgs(T=400, H=500, ns=[30, 70])
        >>> S.clear_flow_spec()
        >>> S.ns
        '''
        s = self.specifications
        s['ns'] = s['ms'] = s['Qls'] = s['Qgs'] = s['m'] = s['n'] = s['Q'] = s['Ql'] = s['Qg'] = None

    @property
    def flow_spec(self):
        '''Flow specification if one has been set, as a tuples
        of ('specification_name', specification_value); otherwise None.
        Energy is never included in this.

        >>> StreamArgs(T=400, H=500, n=1234).flow_spec
        ('n', 1234)
        >>> StreamArgs(T=400, H=500).flow_spec
        '''
        s = self.specifications
        if s['ns'] is not None:
            return ('ns', s['ns'])
        if s['ms'] is not None:
            return ('ms', s['ms'])
        if s['Qls'] is not None:
            return ('Qls', s['Qls'])
        if s['Qgs'] is not None:
            return ('Qgs', s['Qgs'])
        if s['m'] is not None:
            return ('m', s['m'])
        if s['n'] is not None:
            return ('n', s['n'])
        if s['Q'] is not None:
            return ('Q', s['Q'])
        if s['Ql'] is not None:
            return ('Ql', s['Ql'])
        if s['Qg'] is not None:
            return ('Qg', s['Qg'])
#
#        # TODO consider energy?

    @property
    def specified_flow_vars(self):
        '''Number of flow specifications which have been set.
        Energy is never included regardless of whether it is being used as a flow
        specification.

        >>> StreamArgs(T=400, P=1e5, n=4).specified_flow_vars
        1
        >>> StreamArgs(T=400, P=1e5).specified_flow_vars
        0
        '''
        return sum(i is not None for i in (self.ns, self.ms, self.Qls, self.Qgs, self.m, self.n, self.Q, self.Ql, self.Qg))

    @property
    def flow_specified(self):
        '''Whether or not the flow has been specified.
        Energy is never included regardless of whether it is being used as a flow
        specification.

        >>> StreamArgs(T=400, P=1e5, n=4).flow_specified
        True
        >>> StreamArgs(T=400, P=1e5).flow_specified
        False
        '''
        s = self.specifications
        if s['ns'] is not None:
            return True
        if s['ms'] is not None:
            return True
        if s['Qls'] is not None:
            return True
        if s['Qgs'] is not None:
            return True
        if s['m'] is not None:
            return True
        if s['n'] is not None:
            return True
        if s['Q'] is not None:
            return True
        if s['Ql'] is not None:
            return True
        return s['Qg'] is not None

    def update(self, **kwargs):
        """Update the specifications of the StreamArgs instance with new values provided as keyword arguments.

        Parameters
        ----------
        kwargs : dict
            Key-value pairs of specification names and values, [various]

        Examples
        --------
        >>> stream = StreamArgs(T=300, P=101325)
        >>> stream.update(T=350, P=105000)
        >>> stream.T
        350
        >>> stream.P
        105000

        >>> stream = StreamArgs(S=100, P=101325)
        >>> stream.update(S=None, P=95000, H=5)
        >>> (stream.S, stream.P, stream.H)
        (None, 95000, 5)

        >>> stream = StreamArgs(zs=[0.2, 0.8])
        >>> stream.update(zs=None, ws=[0.15, 0.85])
        >>> (stream.zs, stream.ws)
        (None, [0.15, 0.85])

        Notes
        -----
        This can be convinient when creating a dictionary of specifications and setting them all at once
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def stream(self, existing_flash=None, hot_start=None):
        r'''Create and return an EquilibriumStream object using the set specifications.
        If `existing_flash` is provided, that :obj:`EquilibriumState` object will be used
        as the state specifications without checking that it contains the same specs as set
         to this object. If insufficient information is available, return None.

        Parameters
        ----------
        existing_flash : EquilibriumState, optional
            Existing flash which will be used instead of the composition and
            state specs of this object [-]
        hot_start : EquilibriumState, optional
            Flash at nearby conditions that may be used to initialize the flash [-]

        Returns
        -------
        stream : EquilibriumStream
            Created stream, [-]
        '''
        if self.flow_specified:
            s = self.specifications
            if existing_flash is None:
                existing_flash = self.flash_state(hot_start)
            if existing_flash is not None:
                return EquilibriumStream(self.flasher, hot_start=hot_start,
                                        existing_flash=existing_flash, **s)

    def flash_state(self, hot_start=None):
        r'''Create and return an EquilibriumState object using the set specifications.
        If insufficient information is available, return None.

        Parameters
        ----------
        hot_start : EquilibriumState, optional
            Flash at nearby conditions that may be used to initialize the flash [-]

        Returns
        -------
        flash : EquilibriumState
            Created equilibrium state, [-]
        '''
        if self.composition_specified and self.state_specified:
            s = self.specifications
            # Flash call only takes `zs`
            zs = self.zs_calc
            T, P, VF = s['T'], s['P'], s['VF']
            H = H_reactive = None
            # Do we need
            spec_count = 0
            if T is not None:
                spec_count += 1
            if P is not None:
                spec_count += 1
            if s['V'] is not None:
                spec_count += 1
            if s['rho'] is not None:
                spec_count += 1
            if s['rho_mass'] is not None:
                spec_count += 1
            if s['H_mass'] is not None:
                spec_count += 1
            if s['H'] is not None:
                spec_count += 1
            if s['S_mass'] is not None:
                spec_count += 1
            if s['S'] is not None:
                spec_count += 1
            if s['U_mass'] is not None:
                spec_count += 1
            if s['U'] is not None:
                spec_count += 1
            if s['G_mass'] is not None:
                spec_count += 1
            if s['G'] is not None:
                spec_count += 1
            if s['A_mass'] is not None:
                spec_count += 1
            if s['A'] is not None:
                spec_count += 1
            if s['H_reactive'] is not None:
                spec_count += 1
            if VF is not None:
                spec_count += 1

            if spec_count < 2:
                energy = s['energy']
                if energy is not None:
                    n = self.n_calc
                    if n is not None:
                        H = energy/n
                        spec_count += 1
                energy_reactive = s['energy_reactive']
                if energy_reactive is not None:
                    n = self.n_calc
                    if n is not None:
                        H_reactive = energy_reactive/n
                        spec_count += 1

            H_flash = s['H'] if s['H'] is not None else H
            H_reactive = s['H_reactive'] if s['H_reactive'] is not None else H_reactive
            state_cache = (T, P, VF, s['S_mass'], s['S'], s['H'], H_flash, s['H_reactive'], H_reactive,
                           s['U'], s['U_mass'], s['G'], s['G_mass'], s['A'], s['A_mass'],
                           s['V'], s['rho'], s['rho_mass'], tuple(zs))
            if state_cache == self._state_cache:
                try:
                    return self._flash
                except:
                    pass

            m = self.flasher.flash(T=T, P=P, zs=zs, H=H_flash, H_mass=s['H_mass'],
                                            S=s['S'], S_mass=s['S_mass'],
                                            U=s['U'], U_mass=s['U_mass'],
                                            G=s['G'], G_mass=s['G_mass'],
                                            A=s['A'], A_mass=s['A_mass'],
                                            V=s['V'], rho=s['rho'], rho_mass=s['rho_mass'],
                                            H_reactive=H_reactive,
                                            VF=VF, hot_start=hot_start)
            self._flash = m
            self._state_cache = state_cache
            return m

    def value(self, name):
        r'''Wrapper around getattr that obtains a property specified. This method
        exists to unify e.g. H() on a EquilibriumState with H here which is a property.
        Either object can simply be called obj.value("H"). [various]
        '''
        v = getattr(self, name)
        try:
            v = v()
        except:
            pass
        return v

class Stream(Mixture):
    '''Creates a Stream object which is useful for modeling mass and energy
    balances.

    Streams have five variables. The flow rate, composition, and components are
    mandatory; and two of the variables temperature, pressure, vapor fraction,
    enthalpy, or entropy are required. Entropy and enthalpy may also be
    provided in a molar basis; energy can also be provided, which when
    combined with either a flow rate or enthalpy will calculate the other
    variable.

    The composition and flow rate may be specified together or separately. The
    options for specifying them are:

    * Mole fractions `zs`
    * Mass fractions `ws`
    * Liquid standard volume fractions `Vfls`
    * Gas standard volume fractions `Vfgs`
    * Mole flow rates `ns`
    * Mass flow rates `ms`
    * Liquid flow rates `Qls` (based on pure component volumes at the T and P
      specified by `Q_TP`)
    * Gas flow rates `Qgs` (based on pure component volumes at the T and P
      specified by `Q_TP`)

    If only the composition is specified by providing any of `zs`, `ws`, `Vfls`
    or `Vfgs`, the flow rate must be specified by providing one of these:

    * Mole flow rate `n`
    * Mass flow rate `m`
    * Volumetric flow rate `Q` at the provided `T` and `P` or if specified,
      `Q_TP`
    * Energy `energy`

    The state variables must be two of the following. Not all combinations
    result in a supported flash.

    * Tempetarure `T`
    * Pressure `P`
    * Vapor fraction `VF`
    * Enthalpy `H`
    * Molar enthalpy `Hm`
    * Entropy `S`
    * Molar entropy `Sm`
    * Energy `energy`

    Parameters
    ----------
    IDs : list, optional
        List of chemical identifiers - names, CAS numbers, SMILES or InChi
        strings can all be recognized and may be mixed [-]
    zs : list or dict, optional
        Mole fractions of all components in the stream [-]
    ws : list or dict, optional
        Mass fractions of all components in the stream [-]
    Vfls : list or dict, optional
        Volume fractions of all components as a hypothetical liquid phase based
        on pure component densities [-]
    Vfgs : list or dict, optional
        Volume fractions of all components as a hypothetical gas phase based
        on pure component densities [-]
    ns : list or dict, optional
        Mole flow rates of each component [mol/s]
    ms : list or dict, optional
        Mass flow rates of each component [kg/s]
    Qls : list or dict, optional
        Liquid flow rates of all components as a hypothetical liquid phase
        based on pure component densities [m^3/s]
    Qgs : list or dict, optional
        Gas flow rates of all components as a hypothetical gas phase
        based on pure component densities [m^3/s]
    n : float, optional
        Total mole flow rate of all components in the stream [mol/s]
    m : float, optional
        Total mass flow rate of all components in the stream [kg/s]
    Q : float, optional
        Total volumetric flow rate of all components in the stream based on the
        temperature and pressure specified by `T` and `P` [m^3/s]
    T : float, optional
        Temperature of the stream (default 298.15 K), [K]
    P : float, optional
        Pressure of the stream (default 101325 Pa) [Pa]
    VF : float, optional
        Vapor fraction (mole basis) of the stream, [-]
    H : float, optional
        Mass enthalpy of the stream, [J]
    Hm : float, optional
        Molar enthalpy of the stream, [J/mol]
    S : float, optional
        Mass entropy of the stream, [J/kg/K]
    Sm : float, optional
        Molar entropy of the stream, [J/mol/K]
    energy : float, optional
        Flowing energy of the stream (`H`*`m`), [W]
    pkg : object
        The thermodynamic property package to use for flash calculations;
        one of the caloric packages in :obj:`thermo.property_package`;
        defaults to the ideal model [-]
    Vf_TP : tuple(2, float), optional
        The (T, P) at which the volume fractions are specified to be at, [K]
        and [Pa]
    Q_TP : tuple(3, float, float, str), optional
        The (T, P, phase) at which the volumetric flow rate is specified to be
        at, [K] and [Pa]

    Examples
    --------
    Creating Stream objects:


    A stream of vodka with volume fractions 60% water, 40% ethanol, 1 kg/s:

    >>> from thermo import Stream
    >>> Stream(['water', 'ethanol'], Vfls=[.6, .4], T=300, P=1E5, m=1)
    <Stream, components=['water', 'ethanol'], mole fractions=[0.8299, 0.1701], mass flow=1.0 kg/s, mole flow=43.883974 mol/s, T=300.00 K, P=100000 Pa>

    A stream of air at 400 K and 1 bar, flow rate of 1 mol/s:

    >>> Stream('air', T=400, P=1e5, n=1)
    <Stream, components=['nitrogen', 'argon', 'oxygen'], mole fractions=[0.7812, 0.0092, 0.2096], mass flow=0.028958 kg/s, mole flow=1 mol/s, T=400.00 K, P=100000 Pa>

    Instead of specifying the composition and flow rate separately, they can
    be specified as a list of flow rates in the appropriate units.

    80 kg/s of furfuryl alcohol/water solution:

    >>> Stream(['furfuryl alcohol', 'water'], ms=[50, 30])
    <Stream, components=['furfuryl alcohol', 'water'], mole fractions=[0.2343, 0.7657], mass flow=80.0 kg/s, mole flow=2174.937359509809 mol/s, T=298.15 K, P=101325 Pa>

    A stream of 100 mol/s of 400 K, 1 MPa argon:

    >>> Stream(['argon'], ns=[100], T=400, P=1E6)
    <Stream, components=['argon'], mole fractions=[1.0], mass flow=3.9948 kg/s, mole flow=100 mol/s, T=400.00 K, P=1000000 Pa>

    A large stream of vinegar, 8 volume %:

    >>> Stream(['Acetic acid', 'water'], Qls=[1, 1/.088])
    <Stream, components=['acetic acid', 'water'], mole fractions=[0.0269, 0.9731], mass flow=12372.158780648899 kg/s, mole flow=646268.5186913002 mol/s, T=298.15 K, P=101325 Pa>

    A very large stream of 100 m^3/s of steam at 500 K and 2 MPa:

    >>> Stream(['water'], Qls=[100], T=500, P=2E6)
    <Stream, components=['water'], mole fractions=[1.0], mole flow=4617174.33613 mol/s, T=500.00 K, P=2000000 Pa>

    A real example of a stream from a pulp mill:

    >>> Stream(['Methanol', 'Sulphuric acid', 'sodium chlorate', 'Water', 'Chlorine dioxide', 'Sodium chloride', 'Carbon dioxide', 'Formic Acid', 'sodium sulfate', 'Chlorine'], T=365.2, P=70900, ns=[0.3361749, 11.5068909, 16.8895876, 7135.9902928, 1.8538332, 0.0480655, 0.0000000, 2.9135162, 205.7106922, 0.0012694])
    <Stream, components=['methanol', 'sulfuric acid', 'sodium chlorate', 'water', 'chlorine dioxide', 'sodium chloride', 'carbon dioxide', 'formic acid', 'sodium sulfate', 'chlorine'], mole fractions=[0.0, 0.0016, 0.0023, 0.9676, 0.0003, 0.0, 0.0, 0.0004, 0.0279, 0.0], mole flow=7375.2503227 mol/s, T=365.20 K, P=70900 Pa>

    For streams with large numbers of components, it may be confusing to enter
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
    >>> m = Stream(ws=comp, m=33)

    Notes
    -----
    .. warning::
        The Stream class is not designed for high-performance or the ability
        to use different thermodynamic models. It is especially limited in its
        multiphase support and the ability to solve with specifications other
        than temperature and pressure. It is impossible to change constant
        properties such as a compound's critical temperature in this interface.

        It is recommended to switch over to the :obj:`thermo.flash` and
        :obj:`EquilibriumStream` interfaces
        which solves those problems and are better positioned to grow. That
        interface also requires users to be responsible for their chemical
        constants and pure component correlations; while default values can
        easily be loaded for most compounds, the user is ultimately responsible
        for them.

    '''

    flashed = True
    def __repr__(self): # pragma: no cover
        txt = f'<Stream, components={self.names}, mole fractions={[round(i,4) for i in self.zs]}, mass flow={self.m} kg/s, mole flow={self.n} mol/s'
        # T and P may not be available if a flash has failed
        try:
            txt += f', T={self.T:.2f} K, P={self.P:.0f} Pa>'
        except:
            txt += ', thermodynamic conditions unknown>'
        return txt


    def __init__(self, IDs=None, zs=None, ws=None, Vfls=None, Vfgs=None,
                 ns=None, ms=None, Qls=None, Qgs=None,
                 n=None, m=None, Q=None,
                 T=None, P=None, VF=None, H=None, Hm=None, S=None, Sm=None,
                 energy=None, pkg=None, Vf_TP=(None, None), Q_TP=(None, None, '')):

        composition_options = ('zs', 'ws', 'Vfls', 'Vfgs', 'ns', 'ms', 'Qls', 'Qgs')
        composition_option_count = 0
        for i in composition_options:
            if locals()[i] is not None:
                composition_option_count += 1
                self.composition_spec = (i, locals()[i])

        if hasattr(IDs, 'strip') or (type(IDs) == list and len(IDs) == 1):
            pass # one component only - do not raise an exception
        elif composition_option_count < 1:
            raise Exception("No composition information is provided; one of "
                            "'ws', 'zs', 'Vfls', 'Vfgs', 'ns', 'ms', 'Qls' or "
                            "'Qgs' must be specified")
        elif composition_option_count > 1:
            raise Exception("More than one source of composition information "
                            "is provided; only one of "
                            "'ws', 'zs', 'Vfls', 'Vfgs', 'ns', 'ms', 'Qls' or "
                            "'Qgs' can be specified")



        # if more than 1 of composition_options is given, raise an exception
        flow_options = ('ns', 'ms', 'Qls', 'Qgs', 'm', 'n', 'Q') # energy
        flow_option_count = 0
        for i in flow_options:
            if locals()[i] is not None:
                flow_option_count += 1
                self.flow_spec = (i, locals()[i])


#        flow_option_count = sum(i is not None for i in flow_options)
        # Energy can be used as an enthalpy spec or a flow rate spec
        if flow_option_count > 1 and energy is not None:
            if Hm is not None or H is not None:
                flow_option_count -= 1

        if flow_option_count < 1:
            raise Exception("No flow rate information is provided; one of "
                            "'m', 'n', 'Q', 'ms', 'ns', 'Qls', 'Qgs' or "
                            "'energy' must be specified")
        elif flow_option_count > 1:
            raise Exception("More than one source of flow rate information is "
                            "provided; only one of "
                            "'m', 'n', 'Q', 'ms', 'ns', 'Qls', 'Qgs' or "
                            "'energy' can be specified")

        if ns is not None:
            zs = ns
        elif ms is not None:
            ws = ms
        elif Qls is not None:
            Vfls = Qls
        elif Qgs is not None:
            Vfgs = Qgs

        # If T and P are known, only need to flash once
        if T is not None and P is not None:
            super().__init__(IDs, zs=zs, ws=ws, Vfls=Vfls, Vfgs=Vfgs,
                 T=T, P=P, Vf_TP=Vf_TP, pkg=pkg)
        else:
            # Initialize without a flash
            Mixture.autoflash = False
            super().__init__(IDs, zs=zs, ws=ws, Vfls=Vfls, Vfgs=Vfgs,
                 Vf_TP=Vf_TP, pkg=pkg)
            Mixture.autoflash = True



        if n is not None:
            self.n = n
        elif m is not None:
            self.n = property_molar_to_mass(m, self.MW) # m*10000/MW
        elif Q is not None:
            try:
                if Q_TP != (None, None, ''):
                    if len(Q_TP) == 2 or (len(Q_TP) == 3 and not Q_TP[-1]):
                        # Calculate the phase via the property package
                        self.property_package.flash(self.zs, T=Q_TP[0], P=Q_TP[1])
                        phase = self.property_package.phase if self.property_package.phase in ('l', 'g') else 'g'
                    else:
                        phase = Q_TP[-1]
                    if phase == 'l':
                        Vm = self.VolumeLiquidMixture(T=Q_TP[0], P=Q_TP[1], zs=self.zs, ws=self.ws)
                    else:
                        Vm = self.VolumeGasMixture(T=Q_TP[0], P=Q_TP[1], zs=self.zs, ws=self.ws)

                else:
                    Vm = self.Vm
                self.n = Q/Vm
            except:
                raise Exception('Molar volume could not be calculated to determine the flow rate of the stream.')
        elif ns is not None:
            if isinstance(ns, (OrderedDict, dict)):
                ns = ns.values()
            self.n = sum(ns)
        elif ms is not None:
            if isinstance(ms, (OrderedDict, dict)):
                ms = ms.values()
            self.n = property_molar_to_mass(sum(ms), self.MW)
        elif Qls is not None:
            # volume flows and total enthalpy/entropy should be disabled
            try:
                if isinstance(Qls, (OrderedDict, dict)):
                    Qls = Qls.values()
                self.n = sum([Q/Vml for Q, Vml in zip(Qls, self.Vmls)])
            except:
                raise Exception('Liquid molar volume could not be calculated to determine the flow rate of the stream.')
        elif Qgs is not None:
            try:
                if isinstance(Qgs, (OrderedDict, dict)):
                    Qgs = Qgs.values()
                self.n = sum([Q/Vmg for Q, Vmg in zip(Qgs, [ideal_gas(T, P)]*self.N)])
            except:
                raise Exception('Gas molar volume could not be calculated to determine the flow rate of the stream.')
        elif energy is not None:
            if H is not None:
                self.m = energy/H # Watt/(J/kg) = kg/s
            elif Hm is not None:
                self.n = energy/Hm # Watt/(J/kg) = mol/s
            else:
                raise NotImplementedError

        # Energy specified - calculate H or Hm
        if energy is not None:
            if hasattr(self, 'm'):
                H = energy/self.m
            if hasattr(self, 'n'):
                Hm = energy/self.n

        if T is None or P is None:
            non_TP_state_vars = sum(i is not None for i in [VF, H, Hm, S, Sm, energy])
            if non_TP_state_vars == 0:
                if T is None:
                    T = self.T_default
                if P is None:
                    P = self.P_default

        self.flash(T=T, P=P, VF=VF, H=H, Hm=Hm, S=S, Sm=Sm)

        self.set_extensive_flow(self.n)
        self.set_extensive_properties()

    def set_extensive_flow(self, n=None):
        if n is None:
            n = self.n
        T, P = self.T, self.P
        self.n = n
        self.m = m = property_mass_to_molar(n, self.MW)
        self.ns = [n*zi for zi in self.zs]
        self.ms = [m*wi for wi in self.ws]
        try:
            self.Q = m/self.rho
        except:
            pass
        try:
            V_ig = ideal_gas(T, P)
            self.Qgs = [m/Vm_to_rho(V_ig, MW=MW) for m, MW in zip(self.ms, self.MWs)]
        except:
            pass
        try:
            self.Qls = [m/rho for m, rho in zip(self.ms, self.rhols)]
        except:
            pass

        if self.phase in ('l/g', 'l'):
            self.nl = nl = n*(1. - self.V_over_F)
            self.nls = [xi*nl for xi in self.xs]
            self.mls = [ni*MWi*1E-3 for ni, MWi in zip(self.nls, self.MWs)]
            self.ml = sum(self.mls)
            if self.rhol:
                self.Ql = self.ml/self.rhol
            else:
                self.Ql = None

        if self.phase in ('l/g', 'g'):
            self.ng = ng = n*self.V_over_F
            self.ngs = [yi*ng for yi in self.ys]
            self.mgs = [ni*MWi*1E-3 for ni, MWi in zip(self.ngs, self.MWs)]
            self.mg = sum(self.mgs)
            if self.rhog:
                self.Qg = self.mg/self.rhog
            else:
                self.Qg = None


    # flow_spec, composition_spec are attributes already
    @property
    def specified_composition_vars(self):
        '''number of composition variables'''
        return 1

    @property
    def composition_specified(self):
        '''Always needs a composition'''
        return True

    @property
    def specified_state_vars(self):
        '''Always needs two states'''
        return 2

    @property
    def non_pressure_spec_specified(self):
        '''Cannot have a stream without an energy-type spec.
        '''
        return True

    @property
    def state_specified(self):
        '''Always needs a state'''
        return True

    @property
    def state_specs(self):
        '''Returns a list of tuples of (state_variable, state_value) representing
        the thermodynamic state of the system.
        '''
        specs = []
        for i, var in enumerate(('T', 'P', 'VF', 'Hm', 'H', 'Sm', 'S', 'energy')):
            v = self.specs[i]
            if v is not None:
                specs.append((var, v))
        return specs

    @property
    def specified_flow_vars(self):
        '''Always needs only one flow specified'''
        return 1

    @property
    def flow_specified(self):
        '''Always needs a flow specified'''
        return True


    def flash(self, T=None, P=None, VF=None, H=None, Hm=None, S=None, Sm=None,
              energy=None):
        self.specs = (T, P, VF, H, Hm, S, Sm, energy)
        if energy is not None:
            H = energy/self.m

        if H is not None:
            Hm = property_mass_to_molar(H, self.MW)

        if S is not None:
            Sm = property_mass_to_molar(S, self.MW)
        super().flash_caloric(T=T, P=P, VF=VF, Hm=Hm, Sm=Sm)
        self.set_extensive_properties()


    def set_extensive_properties(self):
        if not hasattr(self, 'm'):
            self.energy = None
            return None
        if self.H is not None and self.m is not None:
            self.energy = self.H*self.m
            self.energy_reactive = self.H_reactive*self.m
        else:
            self.energy = None

    def calculate(self, T=None, P=None):
        self.set_TP(T=T, P=P)
        self.set_phase()
        if hasattr(self, 'rho') and self.rho:
            self.Q = self.m/self.rho
        else:
            self.Q = None
        self.set_extensive_flow()
        self.set_extensive_properties()

    def __add__(self, other):
        if not isinstance(other, Stream):
            raise Exception('Adding to a stream requires that the other object '
                            'also be a stream.')

        if (set(self.CASs) == set(other.CASs)) and (len(self.CASs) == len(other.CASs)):
            cmps = self.CASs
        else:
            cmps = sorted(list(set(self.CASs + other.CASs)))
        mole = self.n + other.n
        moles = []
        for cmp in cmps:
            moles.append(0)
            if cmp in self.CASs:
                ind = self.CASs.index(cmp)
                moles[-1] += self.zs[ind]*self.n
            if cmp in other.CASs:
                ind = other.CASs.index(cmp)
                moles[-1] += other.zs[ind]*other.n

        T = min(self.T, other.T)
        P = min(self.P, other.P)
        return Stream(IDs=cmps, ns=moles, T=T, P=P, pkg=self.property_package)

    def __sub__(self, other):
        # Subtracts the mass flow rates in other from self and returns a new
        # Stream instance

        # Check if all components are present in the original stream,
        # while ignoring 0-flow streams in other
        components_in_self = [i in self.CASs for i in other.CASs]
        if not all(components_in_self):
            for i, in_self in enumerate(components_in_self):
                if not in_self and other.zs[i] > 0:
                    raise Exception(f'Not all components to be removed are \
present in the first stream; {other.IDs[i]} is not present.')

        # Calculate the mole flows of each species
        ns_self = list(self.ns)
        ns_other = list(other.ns)
        n_product = sum(ns_self) - sum(ns_other)

        for i, CAS in enumerate(self.CASs):
            if CAS in other.CASs:
                nj = ns_other[other.CASs.index(CAS)]
                # Merely normalizing the mole flow difference is enough to make
                # ~1E-16 relative differences; allow for a little tolerance here
                relative_difference_product = abs(ns_self[i] - nj)/n_product
                relative_difference_self = abs(ns_self[i] - nj)/ns_self[i]
                if ns_self[i] - nj < 0 and (relative_difference_product > 1E-12 or relative_difference_self > 1E-9):
                    raise Exception(f'Attempting to remove more {self.IDs[i]} than is in the \
first stream.')
                if ns_self[i] - nj < 0.:
                    ns_self[i] = 0.
                elif relative_difference_product < 1E-12:
                    ns_self[i] = 0.
                else:
                    ns_self[i] = ns_self[i] - nj


        # Remove now-empty streams:
        ns_product = []
        CASs_product = []
        for n, CAS in zip(ns_self, self.CASs):
            if n != 0:
                ns_product.append(n)
                CASs_product.append(CAS)
        # Create the resulting stream
        return Stream(IDs=CASs_product, ns=ns_product, T=self.T, P=self.P)



class EquilibriumStream(EquilibriumState):
    '''Creates an EquilibriumStream object, built off :obj:`EquilibriumState`
    to contain flow rate amounts, making mass and energy balances easier.

    EquilibriumStreams can have their flow rate, state, and composition
    defined using any sufficient set of the following. Note that not all
    sets of specs will have a solution, or a unique solution, or an
    algorithm to solve the problem implemented.

    The state can be specified using any two of:

    * Temperature `T` [K]
    * Pressure `P` [Pa]
    * Vapor fraction `VF`
    * Enthalpy `H` [J/mol] or `H_mass` [J/kg]
    * Entropy `S` [J/mol/K] or `S_mass` [J/kg/K]
    * Internal energy `U` [J/mol] or `U_mass` [J/kg]
    * Gibbs free energy `G` [J/mol] or `G_mass` [J/kg]
    * Helmholtz energy `A` [J/mol] or `A_mass` [J/kg]
    * Energy `energy` [W] and `energy_reactive` [W] which count as a enthalpy spec only when flow rate is given
    * Reactive enthalpy `H_reactive` [J/mol]
    * Molar volume `V` [m^3/mol], molar density `rho` [mol/m^3], or mass density `rho_mass`, [kg/m^3]

    The composition can be specified using any of:

    * Mole fractions `zs`
    * Mass fractions `ws`
    * Liquid standard volume fractions `Vfls`
    * Gas standard volume fractions `Vfgs`
    * Mole flow rates of each component `ns` [mol/s]
    * Mass flow rates of each component `ms` [kg/s]
    * Liquid standard volume flow rates of each component `Qls` [m^3/s]
    * Gas standard flow rates of each component `Qgs` [m^3/s]

    Total flow rates can be specified using:

    * Mole flow rate `n` [mol/s]
    * Mass flow rate `m` [kg/s]
    * Actual volume flow rate `Q` [m^3/s]
    * Liquid volume standard flow rate `Ql` [m^3/s]
    * Gas standard volume flow rate `Qg` [m^3/s]

    Note that the liquid flow rates `Ql` and `Qls` will by default use the
    pure component liquid standard molar densities, but the temperature and
    pressure used to calculate the liquid molar densities can be set with
    the two-tuple `Vf_TP`.
    See :obj:`EquilibriumState.V_liquids_ref` for details.

    Parameters
    ----------
    flasher : One of :obj:`thermo.flash.FlashPureVLS`, :obj:`thermo.flash.FlashVL`, :obj:`thermo.flash.FlashVLN`,
        The configured flash object which can perform flashes for the
        configured components, [-]
    zs : list, optional
        Mole fractions of all components [-]
    ws : list, optional
        Mass fractions of all components [-]
    Vfls : list, optional
        Volume fractions of all components as a hypothetical liquid phase based
        on pure component densities [-]
    Vfgs : list, optional
        Volume fractions of all components as a hypothetical gas phase based
        on pure component densities [-]
    ns : list, optional
        Mole flow rates of each component [mol/s]
    ms : list, optional
        Mass flow rates of each component [kg/s]
    Qls : list, optional
        Component volumetric flow rate specs for a hypothetical liquid phase based on
        :obj:`EquilibriumState.V_liquids_ref` [m^3/s]
    Qgs : list, optional
        Component volumetric flow rate specs for a hypothetical gas phase based on
        :obj:`EquilibriumState.V_gas` [m^3/s]
    Ql : float, optional
        Total volumetric flow rate spec for a hypothetical liquid phase based on
        :obj:`EquilibriumState.V_liquids_ref` [m^3/s]
    Qg : float, optional
        Total volumetric flow rate spec for a hypothetical gas phase based on
        :obj:`EquilibriumState.V_gas` [m^3/s]
    Q : float, optional
        Total actual volumetric flow rate of the stream based on the
        density of the stream at the specified conditions [m^3/s]
    n : float, optional
        Total mole flow rate of all components in the stream [mol/s]
    m : float, optional
        Total mass flow rate of all components in the stream [kg/s]
    T : float, optional
        Temperature of the stream, [K]
    P : float, optional
        Pressure of the stream [Pa]
    VF : float, optional
        Vapor fraction (mole basis) of the stream, [-]
    V : float, optional
        Molar volume of the overall stream [m^3/mol]
    rho : float, optional
        Molar density of the overall stream [mol/m^3]
    rho_mass : float, optional
        Mass density of the overall stream [kg/m^3]
    H : float, optional
        Molar enthalpy of the stream [J/mol]
    H_mass : float, optional
        Mass enthalpy of the stream [J/kg]
    S : float, optional
        Molar entropy of the stream [J/mol/K]
    S_mass : float, optional
        Mass entropy of the stream [J/kg/K]
    U : float, optional
        Molar internal energy of the stream [J/mol]
    U_mass : float, optional
        Mass internal energy of the stream [J/kg]
    G : float, optional
        Molar Gibbs free energy of the stream [J/mol]
    G_mass : float, optional
        Mass Gibbs free energy of the stream [J/kg]
    A : float, optional
        Molar Helmholtz energy of the stream [J/mol]
    A_mass : float, optional
        Mass Helmholtz energy of the stream [J/kg]
    energy : float, optional
        Flowing energy of the stream [W]
    energy_reactive : float, optional
        Flowing energy of the stream on a reactive basis [W]
    H_reactive : float, optional
        Reactive molar enthalpy of the stream [J/mol]
    hot_start : :obj:`EquilibriumState`, optional
        See :obj:`EquilibriumState.hot_start`; not recommended as an input, [-]
    existing_flash : :obj:`EquilibriumState`, optional
        Previously calculated :obj:`EquilibriumState` at the exact conditions, will be used
        instead of performing a new flash calculation if provided [-]
    '''

    flashed = True
    __full_path__ = f"{__module__}.{__qualname__}"

    def __repr__(self):
        flow_spec, flow_spec_val = self.flow_spec
        s = f'{self.__class__.__name__}({flow_spec}={flow_spec_val}, '
        s += 'flasher=flasher'
        s += ', existing_flash={}'.format(EquilibriumState.__repr__(self).replace('EquilibriumStream', 'EquilibriumState'))
        s += ')'
        return s

    def __str__(self):
        s = '<EquilibriumStream, T=%.4f, P=%.4f, zs=%s, betas=%s, mass flow=%s kg/s, mole flow=%s mol/s, phases=%s>'
        s = s %(self.T, self.P, self.zs, self.betas, self.m, self.n, self.phases)
        return s

    def __hash__(self):
        # Same as EquilibriumState but with ns
        return hash_any_primitive([self.phases, self.betas, self.gas_count, self.liquid_count, self.solid_count, self.settings, self.flasher, self.ns])

    def __copy__(self):
        # immutable
        return self

    def __init__(self, flasher, *, zs=None, ws=None, Vfls=None, Vfgs=None,
                 ns=None, ms=None, Qls=None, Qgs=None,
                 n=None, m=None, Q=None, Ql=None, Qg=None,
                 T=None, P=None,

                 V=None, rho=None, rho_mass=None,

                 VF=None,
                 H=None, H_mass=None,
                 S=None, S_mass=None,
                 U=None, U_mass=None,
                 G=None, G_mass=None,
                 A=None, A_mass=None,
                 energy=None, energy_reactive=None, H_reactive=None,
                 Vf_TP=None, Q_TP=None, hot_start=None,
                 existing_flash=None, spec_fun=None):

        self.constants = constants = flasher.constants
        self.correlations = flasher.correlations

        # Composition information
        composition_option_count = 0
        if zs is not None:
            composition_option_count += 1
            self.composition_spec = ('zs', zs)
        if ws is not None:
            composition_option_count += 1
            self.composition_spec = ('ws', ws)
        if Vfls is not None:
            composition_option_count += 1
            self.composition_spec = ('Vfls', Vfls)
        if Vfgs is not None:
            composition_option_count += 1
            self.composition_spec = ('Vfgs', Vfgs)
        if ns is not None:
            composition_option_count += 1
            self.composition_spec = ('ns', ns)
        if ms is not None:
            composition_option_count += 1
            self.composition_spec = ('ms', ms)
        if Qls is not None:
            composition_option_count += 1
            self.composition_spec = ('Qls', Qls)
        if Qgs is not None:
            composition_option_count += 1
            self.composition_spec = ('Qgs', Qgs)
        if composition_option_count < 1:
            if existing_flash is not None:
                zs = existing_flash.zs
                composition_option_count += 1
                self.composition_spec = ('zs', zs)
            else:
                raise ValueError("No composition information is provided; one of "
                                "'ws', 'zs', 'Vfls', 'Vfgs', 'ns', 'ms', 'Qls' or "
                                "'Qgs' must be specified")
        elif composition_option_count > 1:
            raise ValueError("More than one source of composition information "
                            "is provided; only one of "
                            "'ws', 'zs', 'Vfls', 'Vfgs', 'ns', 'ms', 'Qls' or "
                            "'Qgs' can be specified")

        flow_option_count = 0
        if ns is not None:
            flow_option_count += 1
            self.flow_spec = ('ns', ns)
        if ms is not None:
            flow_option_count += 1
            self.flow_spec = ('ms', ms)
        if Qls is not None:
            flow_option_count += 1
            self.flow_spec = ('Qls', Qls)
        if Qgs is not None:
            flow_option_count += 1
            self.flow_spec = ('Qgs', Qgs)
        if m is not None:
            flow_option_count += 1
            self.flow_spec = ('m', m)
        if n is not None:
            flow_option_count += 1
            self.flow_spec = ('n', n)
        if Q is not None:
            flow_option_count += 1
            self.flow_spec = ('Q', Q)
        if Ql is not None:
            flow_option_count += 1
            self.flow_spec = ('Ql', Ql)
        if Qg is not None:
            flow_option_count += 1
            self.flow_spec = ('Qg', Qg)

        if flow_option_count > 1 and energy is not None:
            if H is not None or H_mass is not None:
                flow_option_count -= 1

        if flow_option_count < 1:
            raise ValueError("No flow rate information is provided; one of "
                            "'m', 'n', 'Q', 'ms', 'ns', 'Qls', 'Qgs' "
                            "'energy' or 'energy_reactive' must be specified")
        elif flow_option_count > 1:
            raise ValueError("More than one source of flow rate information is "
                            "provided; only one of "
                            "'m', 'n', 'Q', 'Ql', 'Qg', 'ms', 'ns', 'Qls', 'Qgs' "
                            "'energy' or 'energy_reactive' can be specified")

        # Make sure mole fractions are available
        if ns is not None:
            zs = normalize(ns)
        elif ms is not None:
            zs = ws_to_zs(normalize(ms), constants.MWs)
        elif ws is not None:
            zs = ws_to_zs(ws, constants.MWs)
        elif Qls is not None or Vfls is not None:
            if Vfls is None:
                Vfls = normalize(Qls)
            if Vf_TP is not None and Vf_TP != (None, None):
                VolumeObjects = flasher.properties.VolumeLiquids
                T_vf, P_vf = Vf_TP
                Vms_TP = [i(T_vf, P_vf) for i in VolumeObjects]
            else:
                Vms_TP = flasher.V_liquids_ref()
            zs = Vfs_to_zs(Vfls, Vms_TP)
        elif Qgs is not None:
            zs = normalize(Qgs)
        elif Vfgs is not None:
            zs = Vfgs

        MW = 0.0
        N = constants.N
        MWs = constants.MWs
        for i in range(N):
            MW  += zs[i]*MWs[i]
        self._MW = MW
        MW_inv = 1.0/MW

        if energy is not None:
            # Handle the various mole flows - converting to get energy; subset for now
            if m is not None:
                n = property_molar_to_mass(m, MW)  # m*10000/MW
            elif ns is not None:
                n = sum(ns)
            elif ms is not None:
                n = property_molar_to_mass(sum(ms), MW)
            elif Qls is not None:
                n = 0.0
                Vms = flasher.V_liquids_ref()
                for i in range(N):
                    n += Qls[i]/Vms[i]
            elif Qgs is not None:
                V_ig = R*flasher.settings.T_gas_ref/flasher.settings.P_gas_ref
                n = sum(Qgs)/V_ig
            H = energy/n
        elif energy_reactive is not None:
            if m is not None:
                n = property_molar_to_mass(m, MW)  # m*10000/MW
            elif ns is not None:
                n = sum(ns)
            elif ms is not None:
                n = property_molar_to_mass(sum(ms), MW)
            H_reactive = energy_reactive/n

        if existing_flash is not None:
            # All variable which have been set
            if type(existing_flash) is EquilibriumStream:
                composition_spec, flow_spec = self.composition_spec, self.flow_spec

            super().__init__(T=existing_flash.T, P=existing_flash.P, zs=existing_flash.zs, gas=existing_flash.gas, liquids=existing_flash.liquids, solids=existing_flash.solids,
                 betas=existing_flash.betas, flash_specs=existing_flash.flash_specs,
                 flash_convergence=existing_flash.flash_convergence,
                 constants=existing_flash.constants, correlations=existing_flash.correlations,
                 settings=existing_flash.settings, flasher=flasher)
            if type(existing_flash) is EquilibriumStream:
                self.composition_spec, self.flow_spec = composition_spec, flow_spec
                # TODO: are any variables caried over from an existing equilibrium stream?
                # Delete if so

        else:
            dest = super().__init__
            # print(dict(T=T, P=P, V=V, rho=rho, rho_mass=rho_mass, VF=VF,
            #               H=H, H_mass=H_mass, S=S, S_mass=S_mass,
            #               G=G, G_mass=G_mass, U=U, U_mass=U_mass,
            #               A=A, A_mass=A_mass, H_reactive=H_reactive,

            #               zs=zs))
            flasher.flash(T=T, P=P, V=V, rho=rho, rho_mass=rho_mass, VF=VF,
                          H=H, H_mass=H_mass, S=S, S_mass=S_mass,
                          G=G, G_mass=G_mass, U=U, U_mass=U_mass,
                          A=A, A_mass=A_mass, H_reactive=H_reactive,

                          dest=dest, zs=zs, hot_start=hot_start,
                          spec_fun=spec_fun)

        # Convert the flow rate into total molar flow
        if n is not None:
            pass
        elif m is not None:
            n = property_molar_to_mass(m, MW) # m*10000/MW
        elif ns is not None:
            n = sum(ns)
        elif ms is not None:
            n = property_molar_to_mass(sum(ms), MW)
        elif Q is not None:
            try:
                if Q_TP is not None:
                    if len(Q_TP) == 2 or (len(Q_TP) == 3 and not Q_TP[-1]):
                        # Calculate the volume via the property package
                        expensive_flash = flasher.flash(zs=zs, T=Q_TP[0], P=Q_TP[1])
                        V = expensive_flash.V()
                    if Q_TP[-1] == 'l':
                        V = flasher.liquids[0].to(T=Q_TP[0], P=Q_TP[1], zs=zs).V()
                    elif Q_TP[-1] == 'g':
                        V = flasher.gas.to(T=Q_TP[0], P=Q_TP[1], zs=zs).V()
                else:
                    V = self.V()
                n = Q/V
            except:
                raise ValueError('Molar volume could not be calculated to determine the flow rate of the stream.')
        elif Qls is not None:
            n = 0.0
            Vms = flasher.V_liquids_ref()
            try:
                for i in range(N):
                    n += Qls[i]/Vms[i]
            except:
                raise ValueError('Liquid molar volume could not be calculated to determine the flow rate of the stream.')
        elif Ql is not None:
            n = 0.0
            Vms = flasher.V_liquids_ref()
            Vfls = zs_to_Vfs(zs, Vms)
            try:
                for i in range(N):
                    n += Ql*Vfls[i]/(Vms[i])
            except:
                raise ValueError('Liquid molar volume could not be calculated to determine the flow rate of the stream.')
        elif Qg is not None:
            n = 0.0
            settings = self.flasher.settings
            V = R*settings.T_gas_ref/settings.P_gas_ref
            try:
                for i in range(N):
                    n += Qg*zs[i]/V
            except:
                raise ValueError('Liquid molar volume could not be calculated to determine the flow rate of the stream.')
        elif Qgs is not None:
            # Use only ideal gas law; allow user T, P but default to flasher settings when not speced
            if Q_TP is not None and Q_TP[0] is not None and Q_TP[1] is not None:
                V = R*Q_TP[0]/Q_TP[1]
            else:
                V = R*flasher.settings.T_gas_ref/flasher.settings.P_gas_ref
            n = sum(Qgs)/V
        elif energy is not None:
            n = energy/H # Watt/(J/mol) = mol/s # probably wrong
        self.n = n

        self.m = m = property_mass_to_molar(n, MW)
        self.ns = [n*zi for zi in zs]
        self._ws = ws = [zi*MWi*MW_inv for zi, MWi in zip(zs, constants.MWs)]
        self.ms = [m*wi for wi in ws]




    @property
    def Q(self):
        return self.n*self.V()

    Q_calc = Q

    @property
    def Qgs(self):
        # Always use flash settings - do not store weird input
        settings = self.flasher.settings
        V = R*settings.T_gas_ref/settings.P_gas_ref
        n = self.n
        Vn = V*n
        return [zi*Vn for zi in self.zs]


    @property
    def Qls(self):
        ns = self.ns
        Vms_TP = self.V_liquids_ref()
        return [ns[i]*Vms_TP[i] for i in range(self.N)]

    Qls_calc = Qls

    @property
    def Ql(self):
        return sum(self.Qls)

    Ql_calc = Ql

    @property
    def Q_liquid_ref(self):
        return sum(self.Qls)

    def StreamArgs(self):
        '''Goal to create a StreamArgs instance, with the user specified
        variables always being here.

        The state variables are currently correctly tracked. The flow rate and
        composition variable needs to be tracked as a function of what was
        specified as the input variables.

        The flow rate needs to be changed wen the stream flow rate is changed.
        Note this stores unnormalized specs, but that this is OK.
        '''
        kwargs = self.flash_specs.copy()
        del kwargs['zs']
        kwargs['flasher'] = self.flasher
        kwargs[self.composition_spec[0]] = self.composition_spec[1]
        kwargs[self.flow_spec[0]] = self.flow_spec[1]
        return StreamArgs(**kwargs)


    # flow_spec, composition_spec are attributes already
    @property
    def specified_composition_vars(self):
        '''number of composition variables'''
        return 1

    @property
    def composition_specified(self):
        '''Always needs a composition'''
        return True

    @property
    def specified_state_vars(self):
        '''Always needs two states'''
        return 2

    @property
    def non_pressure_spec_specified(self):
        '''Cannot have a stream without an energy-type spec.
        '''
        return True

    @property
    def state_specified(self):
        '''Always needs a state'''
        return True

    @property
    def state_specs(self):
        '''Returns a list of tuples of (state_variable, state_value) representing
        the thermodynamic state of the system.
        '''
        specs = []
        flash_specs = self.flash_specs
        for i, var in enumerate(('T', 'P', 'VF', 'H', 'S', 'energy')):
            if var in flash_specs:
                v = flash_specs[var]
                if v is not None:
                    specs.append((var, v))
        return specs


    @property
    def specified_flow_vars(self):
        '''Always needs only one flow specified'''
        return 1

    @property
    def flow_specified(self):
        '''Always needs a flow specified'''
        return True

EquilibriumStream.non_pressure_state_specs = StreamArgs.non_pressure_state_specs



ok_energy_types = (type(None), float, int)

class EnergyStream:
    '''Creates an EnergyStream object which contains an energy flow rate.
    This object is made available to help make mass and energy balances easier.

    Parameters
    ----------
    Q : float, optional
        Energy flow rate, None if unknown [W]

    Examples
    --------
    >>> EnergyStream(Q=10.0)
    EnergyStream(Q=10.0)
    >>> EnergyStream(Q=None)
    EnergyStream(Q=None)
    '''

    __full_path__ = f"{__module__}.{__qualname__}"

    obj_references = []
    json_version = 1
    non_json_attributes = []
    vectorized = False

    def copy(self):
        r'''Method to copy the EnergyStream.

        Returns
        -------
        copy : EnergyStream
            Copied Energy Stream, [-]
        '''
        return EnergyStream(Q=self.Q)

    __copy__ = copy

    def __str__(self):
        return f'<Energy stream, Q={self.Q} W>'

    def __repr__(self):
        return f'EnergyStream(Q={self.Q})'

    def __init__(self, Q):
        self.Q = Q
        if not isinstance(Q, ok_energy_types):
            raise ValueError('Energy stream flow rate is not a flow rate')

    @property
    def energy(self):
        '''Getter/setter for the energy of the stream. This method is a compatibility
        shim to make this object work the same way as StreamArgs, so energy balances
        can treat the two objects the same way.
        '''
        return self.Q

    @energy.setter
    def energy(self, energy):
        self.Q = energy

    energy_calc = energy

    def as_json(self, cache=None, option=0):
        r'''Method to create a JSON-friendly representation of the EnergyStream
        object which can be stored, and reloaded later.

        Returns
        -------
        json_repr : dict
            JSON-friendly representation, [-]

        Notes
        -----

        Examples
        --------
        >>> import json
        >>> obj = EnergyStream(Q=325.0)
        >>> json_view = obj.as_json()
        >>> json_str = json.dumps(json_view)
        >>> assert type(json_str) is str
        >>> obj_copy = EnergyStream.from_json(json.loads(json_str))
        >>> assert obj_copy == obj
        '''
        return JsonOptEncodable.as_json(self, cache, option)

    @classmethod
    def from_json(cls, json_repr, cache=None):
        r'''Method to create a EnergyStream object from a JSON-friendly
        serialization of another EnergyStream.

        Parameters
        ----------
        json_repr : dict
            JSON-friendly representation, [-]

        Returns
        -------
        model : :obj:`EnergyStream`
            Newly created object from the json serialization, [-]

        Notes
        -----
        It is important that the input string be in the same format as that
        created by :obj:`EnergyStream.as_json`.

        Examples
        --------
        >>> obj = EnergyStream(Q=550)
        >>> json_view = obj.as_json()
        >>> new_obj = EnergyStream.from_json(json_view)
        >>> assert obj == new_obj
        '''
        return JsonOptEncodable.from_json(json_repr, cache)

    def __eq__(self, other):
        return self.__hash__() == hash(other)

    def __hash__(self):
        r'''Method to calculate and return a hash representing the exact state
        of the object.

        Returns
        -------
        hash : int
            Hash of the object, [-]
        '''
        d = object_data(self)
        ans = hash_any_primitive((self.__class__.__name__, d))
        return ans

def _mole_balance_process_ns(f, ns, compounds, use_mass=True, use_volume=True):
    if use_mass:
        ms = f.specifications['ms']
        if ms is not None and any(v is not None for v in ms):
            MWs = f.flasher.constants.MWs
            if ns is None:
                ns = [None]*compounds
            else:
                ns = list(ns)
            for i in range(compounds):
                if ns[i] is None and ms[i] is not None:
                    ns[i] = property_molar_to_mass(ms[i], MWs[i])
    if use_volume:
        Qls = f.specifications['Qls']
        if Qls is not None and any(v is not None for v in Qls):
            Vms = f.flasher.V_liquids_ref()
            if ns is None:
                ns = [None]*compounds
            else:
                ns = list(ns)
            for i in range(compounds):
                if ns[i] is None and Qls[i] is not None:
                    ns[i] = Qls[i]/Vms[i]
        Qgs = f.specifications['Qgs']
        if Qgs is not None and any(v is not None for v in Qgs):
            Vm = R*f.flasher.settings.T_gas_ref/f.flasher.settings.P_gas_ref
            if ns is None:
                ns = [None]*compounds
            else:
                ns = list(ns)
            for i in range(compounds):
                if ns[i] is None and Qgs[i] is not None:
                    ns[i] = Qgs[i]/Vm
    return ns

def mole_balance(inlets, outlets, compounds, use_mass=True, use_volume=True):
    r'''Basic function for performing material balances between a set of
    inputs and outputs. The API of the function is to accept
    two lists of {EquilibriumStream, StreamArgs} objects.
    The objects are subjected to the condition
    :math:`\text{material in} = \text{material out}`

    If an overdetermined set of inputs is provided (flow known for all objects),
    objects are unchanged and the function returns False.

    If an underdetermined set of inputs is provided (insufficient unknowns to calculate
    anything), objects are unchanged and the function returns False.

    More usefully, if there is exactly one unknown flow, that unknown will be calculated.
    This means the :obj:`StreamArgs` object with the unknown flow will have the balanced
    mole flow `n` set to it.
    In certain cases where StreamArgs is used with `multiple_composition_basis`, it can also
    calculate for a single component flow as part of the specs `ns`, `ms`, `Qls`, or `Qgs`
    if that is the unknown.
    EquilibriumStream objects are never unknown by definition and are always unchanged.

    If `use_mass` is False, mass flows will never be used.
    If `use_volume` is False, volume flows will never be used. The volume flows mentioned are
    'Qls' and 'Qgs', the not-actual component flow rates which can be balanced by definition.

    Parameters
    ----------
    inlets : list[EquilibriumStream or StreamArgs]
        Objects containing known or unknown inlet flows [-]
    outlets : list[EquilibriumStream or StreamArgs]
        Objects containing known or unknown outlet flows [-]
    compounds : int
        The number of components, [-]
    use_mass : bool
        Whether or not to utilize the mass flows at the same time as
        the mole flows to calculate mole flows, if mole flows are not known [-]
    use_volume : bool
        Whether or not to utilize the standard volume flows at the same time as
        the mole flows to calculate mole flows, if mole flows are not known [-]

    Returns
    -------
    progress : bool
        Whether or not the function was able to calculate any unknowns, [-]

    Examples
    --------
    The examples here don't use EquilirbiumStreams to make the examples concise.

    >>> f0 = StreamArgs(ns=[1,2,3,4])
    >>> f1 = StreamArgs(n=5, zs=[.5, 0, 0, .5])
    >>> p0 = StreamArgs()
    >>> progress = mole_balance([f0, f1], [p0], compounds=4)
    >>> p0
    StreamArgs(flasher=False, ns=[3.5, 2.0, 3.0, 6.5])

    >>> f0 = StreamArgs(ns=[1, 2, 3, None])
    >>> f1 = StreamArgs(ns=[3, 5, 9, 3])
    >>> p0 = StreamArgs(ns=[None, None, None, 5])
    >>> progress = mole_balance([f0, f1], [p0], compounds=4)
    >>> f0
    StreamArgs(flasher=False, ns=[1, 2, 3, 2])
    >>> f1
    StreamArgs(flasher=False, ns=[3, 5, 9, 3])
    >>> p0
    StreamArgs(flasher=False, ns=[4, 7, 12, 5])

    Notes
    -----
    The algorithm is formulaic, with equations spelled out and no numerical
    methods used. It is hoped all edge cases have been included. It is suggested
    that simple material and energy balances can be solved by repeatedly calling
    :obj:`mole_balance` and :obj:`energy_balance` until both functions return False.

    An exhaustive effort was put into ensuring every possible set of
    specifications is covered. Nevertheless, this is one of the more complex
    pieces of code in the library because of the number of inputs, so please
    reach out if you believe there is a missing case.

    This function will hapilly set negative flows if the math calculates negative
    flows.
    '''
    inlet_count = len(inlets)
    outlet_count = len(outlets)

    in_unknown_count = out_unknown_count = 0
    in_unknown_idx = out_unknown_idx = None
    all_ns_in, all_ns_out = [], []
    all_in_known = all_out_known = True

    for i in range(inlet_count):
        f = inlets[i]
        try:
            ns = f.specifications['ns']
        except:
            ns = f.ns
        if ns is None and use_mass:
            ns = f.ns_calc
        if ns is None or None in ns:
            if use_mass:
                ns = _mole_balance_process_ns(f, ns, compounds, use_mass, use_volume)
            all_in_known = False
            in_unknown_count += 1
            in_unknown_idx = i
        all_ns_in.append(ns)

    for i in range(outlet_count):
        f = outlets[i]
        try:
            ns = f.specifications['ns']
        except:
            ns = f.ns
        if ns is None and use_mass:
            ns = f.ns_calc
        if ns is None or None in ns:
            if use_mass:
                ns = _mole_balance_process_ns(f, ns, compounds, use_mass, use_volume)
            all_out_known = False
            out_unknown_count += 1
            out_unknown_idx = i
        all_ns_out.append(ns)

    if all_out_known and all_in_known:
        # Fast path - all known
        return False

    if all_in_known:
        inlet_ns = [] # List of all molar flows in; set only when everything in is known
        for j in range(compounds):
            v = 0.0
            for i in range(inlet_count):
                v += all_ns_in[i][j]
            inlet_ns.append(v)

    if all_out_known:
        outlet_ns = []
        for j in range(compounds):
            v = 0.0
            for i in range(outlet_count):
                v += all_ns_out[i][j]
            outlet_ns.append(v)

    if out_unknown_count == 1 and in_unknown_count == 0:
        if outlet_count == 1:
            out_ns_calc = [i for i in inlet_ns]
        else:
            out_ns_calc = [inlet_ns[i] - sum(all_ns_out[j][i] for j in range(outlet_count) if (all_ns_out[j] and all_ns_out[j][i] is not None))
                           for i in range(compounds)]

        outlets[out_unknown_idx].ns = out_ns_calc
        return True
    if in_unknown_count == 1 and out_unknown_count == 0:
        if inlet_count == 1:
            in_ns_calc = [i for i in outlet_ns]
        else:
            in_ns_calc = [outlet_ns[i] - sum(all_ns_in[j][i] for j in range(inlet_count) if (all_ns_in[j] and all_ns_in[j][i] is not None))
                           for i in range(compounds)]

        inlets[in_unknown_idx].ns = in_ns_calc
        return True
    elif in_unknown_count == 0 and out_unknown_count == 0:
        return False # Nothing to do - everything is known

    progress = False
    # For each component, see if only one stream is missing it
    for j in range(compounds):
        in_missing, idx_missing = None, None
        missing_count = 0
        v = 0
        for i in range(inlet_count):
            ns = all_ns_in[i]
            if ns is None or ns[j] is None:
                missing_count += 1
                in_missing, idx_missing = True, i
            else:
                v += ns[j]
        for i in range(outlet_count):
            ns = all_ns_out[i]
            if ns is None or ns[j] is None:
                missing_count += 1
                in_missing, idx_missing = False, i
            else:
                v -= ns[j]
        if missing_count == 1:
            progress = True
            if in_missing:
                set_to_ns = inlets[idx_missing].specifications['ns']
                if set_to_ns is not None:
                    set_to_ns[j] = -v
                else:
                    set_to_ms = inlets[idx_missing].specifications['ms']
                    if set_to_ms is not None:
                        set_to_ms[j] = property_mass_to_molar(-v, inlets[idx_missing].flasher.constants.MWs[j])
                    else:
                        set_to_Qls = inlets[idx_missing].specifications['Qls']
                        if set_to_Qls is not None:
                            Vms = inlets[idx_missing].flasher.V_liquids_ref()
                            set_to_Qls[j] = -v*Vms[j]
                        else:
                            set_to_Qgs = inlets[idx_missing].specifications['Qgs']
                            if set_to_Qgs is not None:
                                Vm = R*inlets[idx_missing].flasher.settings.T_gas_ref/inlets[idx_missing].flasher.settings.P_gas_ref
                                set_to_Qgs[j] = -v*Vm
            else:
                set_to_ns = outlets[idx_missing].specifications['ns']
                if set_to_ns is not None:
                    set_to_ns[j] = v
                else:
                    set_to_ms = outlets[idx_missing].specifications['ms']
                    if set_to_ms is not None:
                        set_to_ms[j] = property_mass_to_molar(v, outlets[idx_missing].flasher.constants.MWs[j])
                    else:
                        set_to_Qls = outlets[idx_missing].specifications['Qls']
                        if set_to_Qls is not None:
                            Vms = outlets[idx_missing].flasher.V_liquids_ref()
                            set_to_Qls[j] = v*Vms[j]
                        else:
                            set_to_Qgs = outlets[idx_missing].specifications['Qgs']
                            if set_to_Qgs is not None:
                                Vm = R*outlets[idx_missing].flasher.settings.T_gas_ref/outlets[idx_missing].flasher.settings.P_gas_ref
                                set_to_Qgs[j] = v*Vm
    if progress:
        return progress

    # Try a total mole balance
    n_in_missing_count = 0
    if all_in_known:
        n_in = sum(inlet_ns)
    else:
        n_in_missing_idx = None
        n_in = 0.0
        for i in range(inlet_count):
            f = inlets[i]
            n = f.n
            if n is None:
                n = f.n_calc
            if n is None:
                n_in_missing_count += 1
                n_in_missing_idx = i
            else:
                n_in += n

    n_out_missing_count = 0
    if all_out_known:
        n_out = sum(outlet_ns)
    else:
        n_out_missing_idx = None
        n_out = 0.0
        for i in range(outlet_count):
            f = outlets[i]
            n = f.n
            if n is None:
                n = f.n_calc
            if n is None:
                n_out_missing_count += 1
                n_out_missing_idx = i
            else:
                n_out += n

    if n_out_missing_count == 0 and n_in_missing_count == 1:
        inlets[n_in_missing_idx].specifications['n'] = n_out - n_in
        progress = True
    if n_in_missing_count == 0 and n_out_missing_count == 1:
        outlets[n_out_missing_idx].specifications['n'] = n_in - n_out
        progress = True
    return progress


def energy_balance(inlets, outlets, reactive=False, use_mass=False):
    r'''Basic function for performing energy balances between a set of
    inputs and outputs. The API of the function is to accept
    two lists of {EnergyStream, EquilibriumStream, StreamArgs} objects.
    The objects are subjected to the condition
    :math:`\text{energy in} =  \text{energy out}`

    If an overdetermined set of inputs is provided (energy known for all objects),
    objects are unchanged and the function returns False.

    If an underdetermined set of inputs is provided (insufficient unknowns to calculate
    anything), objects are unchanged and the function returns False.

    More usefully, if there is exactly one unknown, that unknown will be calculated.
    For an EnergyStream this means replacing Q=None with Q=value. For a StreamArgs
    object this means setting the .energy attribute. EquilibriumStream objects
    are never unknown by definition and are always unchanged.

    If `use_mass` is True, it is possible in certain cases to solve
    for two unknowns using the additional equation
    :math:`\text{mass in} =  \text{mass out}`
    if all stream enthalpies are known.

    If `reactive` is True, the stream energies used is the form on a
    reactive basis.

    Parameters
    ----------
    inlets : list[EnergyStream or EquilibriumStream or StreamArgs]
        Objects containing known or unknown inlet energies [-]
    outlets : list[EnergyStream or EquilibriumStream or StreamArgs]
        Objects containing known or unknown outlet energies [-]
    reactive : bool
        Whether to use the `energy_reactive` form of energy of the
        plain `energy` form. Note that this function will not be able
        to balance stream energies if the heats of formation are not
        available and `reactive` is True [-]
    use_mass : bool
        Whether or not to utilize the mass balance at the same time as
        the energy balance in certain situations [-]

    Returns
    -------
    progress : bool
        Whether or not the function was able to calculate any unknowns, [-]

    Examples
    --------
    >>> inlets = [EnergyStream(Q=10.0), EnergyStream(Q=400.0)]
    >>> outlets = [EnergyStream(Q=160.0), EnergyStream(Q=None)]
    >>> energy_balance(inlets, outlets)
    True
    >>> outlets
    [EnergyStream(Q=160.0), EnergyStream(Q=250.0)]


    Notes
    -----
    The algorithm is formulaic, with equations spelled out and no numerical
    methods used. It is hoped all edge cases have been included. It is suggested
    that simple material and energy balances can be solved by repeatedly calling
    `mole_balance` and `energy_balance` until both functions return False.
    '''
    inlet_count = len(inlets)
    outlet_count = len(outlets)

    in_unknown_count = out_unknown_count = 0
    in_unknown_idx = out_unknown_idx = None
    all_energy_in, all_energy_out = [], []
    all_in_known = all_out_known = True

    if inlet_count == 1 and outlet_count == 1:
        # Don't need flow rates for one in one out
        fin = inlets[0]
        fout = outlets[0]
        if not isinstance(fin, EnergyStream) and not isinstance(fout, EnergyStream):
            if reactive:
                try:
                    H_reactive_in = fin.H_reactive()
                except:
                    H_reactive_in = fin.H_reactive_calc
                try:
                    H_reactive_out = fout.H_reactive()
                except:
                    H_reactive_out = fout.H_reactive_calc

                if H_reactive_in is not None and H_reactive_out is None:
                    fout.H_reactive = H_reactive_in
                    return True
                elif H_reactive_in is None and H_reactive_out is not None:
                    fin.H_reactive = H_reactive_out
                    return True
            else:
                try:
                    Hin = fin.H()
                except:
                    Hin = fin.H_calc
                try:
                    Hout = fout.H()
                except:
                    Hout = fout.H_calc

                if Hin is not None and Hout is None:
                    fout.H = Hin
                    return True
                elif Hin is None and Hout is not None:
                    fin.H = Hout
                    return True

    for i in range(inlet_count):
        f = inlets[i]
        if reactive and not isinstance(f, EnergyStream):
            Q = f.energy_reactive
            if Q is None:
                Q = f.energy_reactive_calc
        else:
            Q = f.energy
            if Q is None:
                Q = f.energy_calc

        if Q is None:
            all_in_known = False
            in_unknown_count += 1
            in_unknown_idx = i
        all_energy_in.append(Q)

    for i in range(outlet_count):
        f = outlets[i]
        if reactive and not isinstance(f, EnergyStream):
            Q = f.energy_reactive
            if Q is None:
                Q = f.energy_reactive_calc
        else:
            Q = f.energy
            if Q is None:
                Q = f.energy_calc
        if Q is None:
            all_out_known = False
            out_unknown_count += 1
            out_unknown_idx = i
        all_energy_out.append(Q)

    if all_out_known and all_in_known:
        # Fast path - all known
        return False

    if all_in_known:
        inlet_energy = sum(all_energy_in)
    if all_out_known:
        outlet_energy = sum(all_energy_out)

    if out_unknown_count == 1 and in_unknown_count == 0:
        set_energy = inlet_energy
        for v in all_energy_out:
            if v is not None:
                set_energy -= v
        if reactive and not isinstance(outlets[out_unknown_idx], EnergyStream):
            outlets[out_unknown_idx].energy_reactive = set_energy
        else:
            outlets[out_unknown_idx].energy = set_energy
        return True
    elif in_unknown_count == 1 and out_unknown_count == 0:
        set_energy = outlet_energy
        for v in all_energy_in:
            if v is not None:
                set_energy -= v
        if reactive and not isinstance(inlets[in_unknown_idx], EnergyStream):
            inlets[in_unknown_idx].energy_reactive = set_energy
        else:
            inlets[in_unknown_idx].energy = set_energy

        return True

    elif (in_unknown_count==1 and out_unknown_count == 1 and use_mass
        and isinstance(inlets[in_unknown_idx], StreamArgs) and isinstance(outlets[out_unknown_idx], StreamArgs)
        and inlets[in_unknown_idx].state_specified and outlets[out_unknown_idx].state_specified):
        """
        from sympy import *
        m_in_known, m_in_unknown, m_out_known, m_out_unknown = symbols('m_in_known, m_in_unknown, m_out_known, m_out_unknown')
        e_in_known, e_out_known = symbols('e_in_known, e_out_known')
        H_in, H_out = symbols('H_in, H_out')

        e_in_unknown = m_in_unknown*H_in
        e_out_unknown = m_out_unknown*H_out

        Eq0 = Eq(e_in_known+e_in_unknown, e_out_unknown+e_out_known)
        Eq1 = Eq(m_in_known+ m_in_unknown, m_out_known+m_out_unknown)
        solve([Eq0, Eq1], [m_in_unknown, m_out_unknown])"""
        unknown_in_state = inlets[in_unknown_idx].flash_state()
        unknown_out_state = outlets[out_unknown_idx].flash_state()
        H_mass_in_unknown = unknown_in_state.H_mass() if not reactive else unknown_in_state.H_reactive_mass()
        H_mass_out_unknown = unknown_out_state.H_mass() if not reactive else unknown_out_state.H_reactive_mass()
        energy_in_known = sum(v for v in all_energy_in if v is not None)
        energy_out_known = sum(v for v in all_energy_out if v is not None)
        m_in_known = sum(v.m for i, v in enumerate(inlets) if (i != in_unknown_idx and not isinstance(v, EnergyStream)))
        m_out_known = sum(v.m for i, v in enumerate(outlets) if (i != out_unknown_idx and not isinstance(v, EnergyStream)))
        inlets[in_unknown_idx].m = (H_mass_out_unknown*m_in_known - H_mass_out_unknown*m_out_known - energy_in_known + energy_out_known)/(H_mass_in_unknown - H_mass_out_unknown)
        outlets[out_unknown_idx].m = (H_mass_in_unknown*m_in_known - H_mass_in_unknown*m_out_known - energy_in_known + energy_out_known)/(H_mass_in_unknown - H_mass_out_unknown)
        return True
    elif in_unknown_count == 2 and out_unknown_count == 0 and use_mass:
        unknown_inlet_idxs = []
        for i in range(inlet_count):
            f = inlets[i]
            if isinstance(f, StreamArgs) and f.state_specified:
                unknown_inlet_idxs.append(i)
        if len(unknown_inlet_idxs) == 2:
            """
            from sympy import *
            m_in_known, m_in_unknown0, m_in_unknown1, m_out_known = symbols('m_in_known, m_in_unknown0, m_in_unknown1, m_out_known')
            e_in_known, e_known = symbols('e_in_known, e_known')
            H_unkown0, H_unkown1 = symbols('H_unkown0, H_unkown1')

            e_unkown0 = m_in_unknown0*H_unkown0
            e_unkown1 = m_in_unknown1*H_unkown1

            Eq0 = Eq(e_known, e_in_known + e_unkown1+e_unkown0 )
            Eq1 = Eq(m_in_known+ m_in_unknown0+ m_in_unknown1, m_out_known)
            solve([Eq0, Eq1], [m_in_unknown0, m_in_unknown1])
            """
            in_unknown_idx_0, in_unknown_idx_1 = unknown_inlet_idxs
            unknown_state_0 = inlets[in_unknown_idx_0].flash_state()
            unknown_state_1 = inlets[in_unknown_idx_1].flash_state()
            H_mass_in_unknown_0 = unknown_state_0.H_mass() if not reactive else unknown_state_0.H_reactive_mass()
            H_mass_in_unknown_1 = unknown_state_1.H_mass() if not reactive else unknown_state_1.H_reactive_mass()
            energy_in_known = sum(v for v in all_energy_in if v is not None)
            m_in_known = sum(v.m for i, v in enumerate(inlets) if (i not in (in_unknown_idx_0, in_unknown_idx_1) and not isinstance(v, EnergyStream)))
            m_out_known = sum(v.m for i, v in enumerate(outlets) if not isinstance(v, EnergyStream))

            inlets[in_unknown_idx_0].m = (H_mass_in_unknown_1*m_in_known - H_mass_in_unknown_1*m_out_known - energy_in_known + outlet_energy)/(H_mass_in_unknown_0 - H_mass_in_unknown_1)
            inlets[in_unknown_idx_1].m = (-H_mass_in_unknown_0*m_in_known + H_mass_in_unknown_0*m_out_known + energy_in_known - outlet_energy)/(H_mass_in_unknown_0 - H_mass_in_unknown_1)
            return True
    elif out_unknown_count == 2 and in_unknown_count == 0 and use_mass:
        unknown_outlet_idxs = []
        for i in range(outlet_count):
            f = outlets[i]
            if isinstance(f, StreamArgs) and f.state_specified:
                unknown_outlet_idxs.append(i)
        if len(unknown_outlet_idxs) == 2:
            out_unknown_idx_0, out_unknown_idx_1 = unknown_outlet_idxs
            unknown_state_0 = outlets[out_unknown_idx_0].flash_state()
            unknown_state_1 = outlets[out_unknown_idx_1].flash_state()
            H_mass_out_unknown_0 = unknown_state_0.H_mass() if not reactive else unknown_state_0.H_reactive_mass()
            H_mass_out_unknown_1 = unknown_state_1.H_mass() if not reactive else unknown_state_1.H_reactive_mass()
            energy_out_known = sum(v for v in all_energy_out if v is not None)
            m_out_known = sum(v.m for i, v in enumerate(outlets) if (i not in (out_unknown_idx_0, out_unknown_idx_1) and not isinstance(v, EnergyStream)))
            m_in_known = sum(v.m for i, v in enumerate(inlets) if not isinstance(v, EnergyStream))

            outlets[out_unknown_idx_0].m = (H_mass_out_unknown_1*m_out_known - H_mass_out_unknown_1*m_in_known - energy_out_known + inlet_energy)/(H_mass_out_unknown_0 - H_mass_out_unknown_1)
            outlets[out_unknown_idx_1].m = (-H_mass_out_unknown_0*m_out_known + H_mass_out_unknown_0*m_in_known + energy_out_known - inlet_energy)/(H_mass_out_unknown_0 - H_mass_out_unknown_1)
            return True
    return False


object_lookups[StreamArgs.__full_path__] = StreamArgs
object_lookups[EquilibriumStream.__full_path__] = EquilibriumStream
object_lookups[EnergyStream.__full_path__] = EnergyStream
