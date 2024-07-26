'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020, 2021, 2022 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains classes and functions for performing flash calculations.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.


'''

__all__ = ['Flash']

from math import floor, log10, nan

from chemicals.utils import hash_any_primitive, mixing_simple, property_mass_to_molar, rho_to_Vm
from fluids.constants import R
from fluids.numerics import linspace, logspace
from fluids.numerics import numpy as np

from thermo import phases
from thermo.equilibrium import EquilibriumState
from thermo.flash.flash_utils import (
    LL_boolean_check,
    VL_boolean_check,
    VLL_boolean_check,
    VLL_or_LL_boolean_check,
    VLN_or_LN_boolean_check,
    flash_phase_boundary_one_sided_secant,
    incipient_liquid_bounded_PT_sat,
    incipient_phase_bounded_naive,
    incipient_phase_one_sided_secant,
)
from thermo.phase_identification import identify_sort_phases
from thermo.serialize import JsonOptEncodable
from thermo.utils import has_matplotlib

try:
    zeros, ones, ndarray = np.zeros, np.ones, np.ndarray
except:
    pass


spec_to_iter_vars = {
     (True, False, False, True, False, False) : ('T', 'H', 'P'), # Iterating on P is slow, derivatives look OK
     # (True, False, False, True, False, False) : ('T', 'H', 'V'), # Iterating on P is slow, derivatives look OK
     (True, False, False, False, True, False) : ('T', 'S', 'P'),
     (True, False, False, False, False, True) : ('T', 'U', 'V'),

     (False, True, False, True, False, False) : ('P', 'H', 'T'),
     (False, True, False, False, True, False) : ('P', 'S', 'T'),
     (False, True, False, False, False, True) : ('P', 'U', 'T'),

     (False, False, True, True, False, False) : ('V', 'H', 'P'), # TODO: change these ones to iterate on T?
     (False, False, True, False, True, False) : ('V', 'S', 'P'),
     (False, False, True, False, False, True) : ('V', 'U', 'P'),
}

spec_to_iter_vars_backup =  {
    (True, False, False, True, False, False) : ('T', 'H', 'V'),
    (True, False, False, False, True, False) : ('T', 'S', 'V'),
    (True, False, False, False, False, True) : ('T', 'U', 'P'),

    (False, True, False, True, False, False) : ('P', 'H', 'V'),
    (False, True, False, False, True, False) : ('P', 'S', 'V'),
    (False, True, False, False, False, True) : ('P', 'U', 'V'),

    (False, False, True, True, False, False) : ('V', 'H', 'T'),
    (False, False, True, False, True, False) : ('V', 'S', 'T'),
    (False, False, True, False, False, True) : ('V', 'U', 'T'),
}

empty_flash_conv = {'iterations': 0, 'err': 0.0, 'stab_guess_name': None}
one_in_list = [1.0]
empty_list = []

NAIVE_BISECTION_PHASE_MIXING_BOUNDARY = 'NAIVE_BISECTION_PHASE_MIXING_BOUNDARY'
SATURATION_SECANT_PHASE_MIXING_BOUNDARY = 'SATURATION_SECANT_PHASE_MIXING_BOUNDARY'
PT_SECANT_PHASE_MIXING_BOUNDARY = 'SATURATION_SECANT_PHASE_MIXING_BOUNDARY'

SECANT_PHASE_BOUNDARY = 'SECANT_PHASE_BOUNDARY'

CAS_H2O = '7732-18-5'


class Flash:
    r'''Base class for performing flash calculations. All Flash objects need
    to inherit from this, and common methods can be added to it.
    '''

    def __init_subclass__(cls):
        cls.__full_path__ = f"{cls.__module__}.{cls.__qualname__}"

    def __eq__(self, other):
        return self.__hash__() == hash(other)

    obj_references = ('correlations', 'constants', 'settings', 'gas', 'liquids', 'liquid', 'solids', 'stab', 'unique_liquids', 'liquid0', 'phases', 'unique_phases', 'eos_pure_STP')
    json_version = 1
    non_json_attributes = []
    vectorized = False

    hash_references = ('correlations', 'constants', 'settings', 'gas', 'liquids', 'liquid', 'solids')

    def __hash__(self):
        to_hash = [self.__class__.__name__]
        for attr in self.hash_references:
            try:
                obj = getattr(self, attr)
                hash_value = hash_any_primitive(obj)
                to_hash.append(hash_value)
                # print(attr, hash_value) # for debugging when something isn't the same
            except:
                pass
        # Unfortunately we can't cache this because things can be changed on us
        ans = hash_any_primitive(to_hash)
        return ans

    def as_json(self, cache=None, option=0):
        r'''Method to create a JSON-friendly serialization of the
        Flasher which can be stored, and reloaded later.

        Returns
        -------
        json_repr : dict
            JSON-friendly representation, [-]

        Notes
        -----

        Examples
        --------
        >>> import json
        '''
        return JsonOptEncodable.as_json(self, cache, option)

    @classmethod
    def from_json(cls, json_repr, cache=None):
        return JsonOptEncodable.from_json(json_repr, cache)

    def flash(self, zs=None, T=None, P=None, VF=None, SF=None, V=None, H=None,
              S=None, G=None, U=None, A=None, solution=None, hot_start=None,
              retry=False, dest=None, rho=None, rho_mass=None, H_mass=None,
              S_mass=None, G_mass=None, U_mass=None, A_mass=None,
              spec_fun=None, H_reactive=None):
        r'''Method to perform a flash calculation and return the result as an
        :obj:`EquilibriumState <thermo.equilibrium.EquilibriumState>` object.
        This generic interface allows flashes with any combination of valid
        specifications; if a flash is unimplemented and error will be raised.

        Parameters
        ----------
        zs : list[float], optional
            Mole fractions of each component, required unless there is only
            one component, [-]
        T : float, optional
            Temperature, [K]
        P : float, optional
            Pressure, [Pa]
        VF : float, optional
            Vapor fraction, [-]
        SF : float, optional
            Solid fraction, [-]
        V : float, optional
            Molar volume of the overall bulk, [m^3/mol]
        H : float, optional
            Molar enthalpy of the overall bulk, [J/mol]
        S : float, optional
            Molar entropy of the overall bulk, [J/(mol*K)]
        G : float, optional
            Molar Gibbs free energy of the overall bulk, [J/mol]
        U : float, optional
            Molar internal energy of the overall bulk, [J/mol]
        A : float, optional
            Molar Helmholtz energy of the overall bulk, [J/mol]
        solution : str or int, optional
           When multiple solutions exist, if more than one is found they will
           be sorted by T (and then P) increasingly; this number will index
           into the multiple solution array. Negative indexing is supported.
           'high' is an alias for 0, and 'low' an alias for -1. Setting this
           parameter may make a flash slower because in some cases more checks
           are performed. [-]
        hot_start : :obj:`EquilibriumState <thermo.equilibrium.EquilibriumState>`
            A previously converged flash or initial guessed state from which
            the flash can begin; this parameter can save time in some cases,
            [-]
        retry : bool
            Usually for flashes like UV or PH, there are multiple sets of
            possible iteration variables. For the UV case, the prefered
            iteration variable is P, so each iteration a PV solve is done on
            the phase; but equally the flash can be done iterating on
            `T`, where a TV solve is done on the phase each iteration.
            Depending on the tolerances, the flash type, the thermodynamic
            consistency of the phase, and other factors, it is possible the
            flash can fail. If `retry` is set to True, the alternate variable
            set will be iterated as a backup if the first flash fails. [-]
        dest : None or :obj:`EquilibriumState <thermo.equilibrium.EquilibriumState>` or :obj:`EquilibriumStream <thermo.stream.EquilibriumStream>`
            What type of object the flash result is set into; leave as None to
            obtain the normal `EquilibriumState` results, [-]
        rho : float, optional
            Molar density of the overall bulk; this is trivially converted to
            a `V` spec, [mol/m^3]
        rho_mass : float, optional
            Mass density of the overall bulk; this is trivially converted to
            a `rho` spec, [kg/m^3]
        H_mass : float, optional
            Mass enthalpy of the overall bulk; this is trivially converted to
            a `H` spec, [J/kg]
        S_mass : float, optional
            Mass entropy of the overall bulk; this is trivially converted to
            a `S` spec, [J/(kg*K)]
        G_mass : float, optional
            Mass Gibbs free energy of the overall bulk; this is trivially converted to
            a `G` spec, [J/kg]
        U_mass : float, optional
            Mass internal energy of the overall bulk; this is trivially converted to
            a `U` spec, [J/kg]
        A_mass : float, optional
            Mass Helmholtz energy of the overall bulk; this is trivially converted to
            a `A` spec, [J/kg]
        H_reactive : float, optional
            Molar reactive enthalpy, [J/mol]

        Returns
        -------
        results : :obj:`EquilibriumState <thermo.equilibrium.EquilibriumState>`
            Equilibrium object containing the state of the phases after the
            flash calculation [-]

        Notes
        -----
        .. warning::
            Not all flash specifications have a unique solution. Not all flash
            specifications will converge, whether from a bad model, bad inputs,
            or simply a lack of convergence by the implemented algorithms.
            You are welcome to submit these cases to the author but the library
            is provided AS IS, with NO SUPPORT.

        .. warning::
            Convergence of a flash may be impaired by providing `hot_start`.
            If reliability is desired, do not use this parameter.

        .. warning::
            The most likely thermodynamic methods to converge are
            thermodynamically consistent ones. This means e.g. an ideal liquid
            and an ideal gas; or an equation of state for both
            phases. Mixing thermodynamic models increases the possibility of
            multiple solutions, discontinuities, and other not-fun issues for
            the algorithms.

        Examples
        --------
        '''
        # print('flashing',T, P, H, S, VF)
        if zs is None:
            if self.N == 1:
                zs = ones(1) if self.vectorized else [1.0]
            else:
                raise ValueError("Composition missing for flash")
        constants, correlations = self.constants, self.correlations
        settings = self.settings
        if dest is None:
            dest = EquilibriumState
#        if self.N > 1 and 0:
#            for zi in zs:
#                if zi == 1.0:
#                    # Does not work - phases expect multiple components mole fractions
#                    return self.flash_pure.flash(zs=zs, T=T, P=P, VF=VF, SF=SF,
#                                           V=V, H=H, S=S, U=U, G=G, A=A,
#                                           solution=solution, retry=retry,
#                                           hot_start=hot_start)
        if rho is not None:
            V, rho = 1.0/rho, None
        if rho_mass is not None:
            V, rho_mass = rho_to_Vm(rho_mass, mixing_simple(zs, constants.MWs)), None
        if H_mass is not None:
            H, H_mass = property_mass_to_molar(H_mass, mixing_simple(zs, constants.MWs)), None
        if S_mass is not None:
            S, S_mass = property_mass_to_molar(S_mass, mixing_simple(zs, constants.MWs)), None
        if G_mass is not None:
            G, G_mass = property_mass_to_molar(G_mass, mixing_simple(zs, constants.MWs)), None
        if U_mass is not None:
            U, U_mass = property_mass_to_molar(U_mass, mixing_simple(zs, constants.MWs)), None
        if A_mass is not None:
            A, A_mass = property_mass_to_molar(A_mass, mixing_simple(zs, constants.MWs)), None
        if H_reactive is not None:
            Hfgs = constants.Hfgs
            H = H_reactive - sum(zs[i]*Hfgs[i] for i in range(constants.N))
            H_reactive = None

        T_spec = T is not None
        P_spec = P is not None
        V_spec = V is not None
        H_spec = H is not None
        S_spec = S is not None
        U_spec = U is not None
        # Normally multiple solutions
        A_spec = A is not None
        G_spec = G is not None

        HSGUA_spec_count = H_spec + S_spec + G_spec + U_spec + A_spec


        VF_spec = VF is not None
        SF_spec = SF is not None

        flash_specs = {'zs': zs}
        if T_spec:
            T = float(T)
            flash_specs['T'] = T
            if T < self.T_MIN_FLASH_ANY:
                raise ValueError(f"Specified temperature ({T} K) is below the minimum temeprature ({self.T_MIN_FLASH_ANY} K) "
                                 "supported by any of the provided phases")
            # if T <= 0.0:
            #     raise ValueError("Specified temperature (%s K) is unphysical" %(T,))
        if P_spec:
            P = float(P)
            flash_specs['P'] = P
            if P <= 0.0:
                raise ValueError(f"Specified pressure ({P} Pa) is unphysical")
        if V_spec:
            # high-precision volumes are needed in some cases
            # V = float(V)
            flash_specs['V'] = V
            if V <= 0.0:
                raise ValueError(f"Specified molar volume ({V} m^3/mol) is unphysical")
        if H_spec:
            H = float(H)
            flash_specs['H'] = H
        if S_spec:
            S = float(S)
            flash_specs['S'] = S
        if U_spec:
            U = float(U)
            flash_specs['U'] = U
        if G_spec:
            G = float(G)
            flash_specs['G'] = G
        if A_spec:
            A = float(A)
            flash_specs['A'] = A

        if VF_spec:
            VF_spec = float(VF_spec)
            flash_specs['VF'] = VF
            if VF < 0.0 or VF > 1.0:
                raise ValueError(f"Specified vapor fraction ({VF}) is not between 0 and 1")
            elif not self.supports_VF_flash:
                raise ValueError("Cannot flash with a vapor fraction spec without at least one gas and liquid phase defined")
        if SF_spec:
            SF_spec = float(SF_spec)
            flash_specs['SF'] = SF
            if SF < 0.0 or SF > 1.0:
                raise ValueError(f"Specified solid fraction ({VF}) is not between 0 and 1")
            elif not self.supports_SF_flash:
                raise ValueError("Cannot flash with a solid fraction spec without at least one gas and liquid phase defined, as well as a solid phase")

        if ((T_spec and (P_spec or V_spec)) or (P_spec and V_spec)):
            g, ls, ss, betas, flash_convergence = self.flash_TPV(T=T, P=P, V=V, zs=zs, solution=solution, hot_start=hot_start)
            # TODO can creating a list here be avoided?
            if g is not None:
                id_phases = [g] + ls + ss
            else:
                id_phases = ls + ss

            g, ls, ss, betas = identify_sort_phases(id_phases, betas, constants,
                                                    correlations, settings=settings,
                                                    skip_solids=self.skip_solids)

            a_phase = id_phases[0]
            return dest(a_phase.T, a_phase.P, zs, gas=g, liquids=ls, solids=ss,
                                    betas=betas, flash_specs=flash_specs,
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    settings=settings, flasher=self)

        elif T_spec and VF_spec:
            # All dew/bubble are the same with 1 component
            Psat, ls, g, iterations, err = self.flash_TVF(T, VF=VF, zs=zs, hot_start=hot_start)
            if type(ls) is not list:
                ls = [ls]
            flash_convergence = {'iterations': iterations, 'err': err}

            return dest(T, Psat, zs, gas=g, liquids=ls, solids=[],
                                    betas=[VF, 1.0 - VF], flash_specs=flash_specs,
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    settings=settings, flasher=self)

        elif P_spec and VF_spec:
            # All dew/bubble are the same with 1 component
            Tsat, ls, g, iterations, err = self.flash_PVF(P, VF=VF, zs=zs, hot_start=hot_start)
            if type(ls) is not list:
                ls = [ls]
            flash_convergence = {'iterations': iterations, 'err': err}

            return dest(Tsat, P, zs, gas=g, liquids=ls, solids=[],
                                    betas=[VF, 1.0 - VF], flash_specs=flash_specs,
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    settings=settings, flasher=self)
        elif T_spec and SF_spec:
            Psub, other_phase, s, iterations, err = self.flash_TSF(T, SF=SF, zs=zs, hot_start=hot_start)
            if other_phase.is_gas:
                g, liquids = other_phase, []
            else:
                g, liquids = None, [other_phase]
            flash_convergence = {'iterations': iterations, 'err': err}
            return dest(T, Psub, zs, gas=g, liquids=liquids, solids=[s],
                                    betas=[1-SF, SF], flash_specs=flash_specs,
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    settings=settings, flasher=self)
        elif P_spec and SF_spec:
            Tsub, other_phase, s, iterations, err = self.flash_PSF(P, SF=SF, zs=zs, hot_start=hot_start)
            if other_phase.is_gas:
                g, liquids = other_phase, []
            else:
                g, liquids = None, [other_phase]
            flash_convergence = {'iterations': iterations, 'err': err}
            return dest(Tsub, P, zs, gas=g, liquids=liquids, solids=[s],
                                    betas=[1-SF, SF], flash_specs=flash_specs,
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    settings=settings, flasher=self)
        elif VF_spec and any([H_spec, S_spec, U_spec, G_spec, A_spec]):
            spec_var, spec_val = next((k, v) for k, v in flash_specs.items() if k not in ('VF', 'zs'))
            T, Psat, liquid, gas, iters_inner, err_inner, err, iterations = self.flash_VF_HSGUA(VF, spec_val, fixed_var='VF', spec_var=spec_var, zs=zs, solution=solution, hot_start=hot_start)
            flash_convergence = {'iterations': iterations, 'err': err, 'inner_flash_convergence': {'iterations': iters_inner, 'err': err_inner}}
            return dest(T, Psat, zs, gas=gas, liquids=[liquid], solids=[],
                                    betas=[VF, 1.0 - VF], flash_specs=flash_specs,
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    settings=settings, flasher=self)
        elif HSGUA_spec_count == 2:
            pass

        single_iter_key = (T_spec, P_spec, V_spec, H_spec, S_spec, U_spec)
        if single_iter_key in spec_to_iter_vars:
            fixed_var, spec, iter_var = spec_to_iter_vars[single_iter_key]
            _, _, iter_var_backup = spec_to_iter_vars_backup[single_iter_key]
            if T_spec:
                fixed_var_val = T
            elif P_spec:
                fixed_var_val = P
            else:
                fixed_var_val = V

            if H_spec:
                spec_val = H
            elif S_spec:
                spec_val = S
            else:
                spec_val = U

            # Only allow one
#            g, ls, ss, betas, flash_convergence = self.flash_TPV_HSGUA(fixed_var_val, spec_val, fixed_var, spec, iter_var)
            try:
                g, ls, ss, betas, flash_convergence = self.flash_TPV_HSGUA(fixed_var_val, spec_val, fixed_var, spec, iter_var, zs=zs, solution=solution, hot_start=hot_start, spec_fun=spec_fun)
            except Exception as e:
                if retry:
                    print('retrying HSGUA flash')
                    g, ls, ss, betas, flash_convergence = self.flash_TPV_HSGUA(fixed_var_val, spec_val, fixed_var, spec, iter_var_backup, zs=zs, solution=solution, hot_start=hot_start, spec_fun=spec_fun)
                else:
                    raise e
#            except UnconvergedError as e:
#                 if fixed_var == 'T' and iter_var in ('S', 'H', 'U'):
#                     g, ls, ss, betas, flash_convergence = self.flash_TPV_HSGUA(fixed_var_val, spec_val, fixed_var, spec, iter_var_backup, solution=solution)
#                 else:
                # Not sure if good idea - would prefer to converge without
            phases = ls + ss
            if g:
                phases += [g]
            T, P = phases[0].T, phases[0].P
            if self.N > 1:
                g, ls, ss, betas = identify_sort_phases(phases, betas, constants,
                                                        correlations, settings=settings,
                                                        skip_solids=self.skip_solids)

            return dest(T, P, zs, gas=g, liquids=ls, solids=ss,
                                    betas=betas, flash_specs=flash_specs,
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    settings=settings, flasher=self)

        else:
            raise Exception('Flash inputs unsupported')

    flash_phase_boundary_algos = [flash_phase_boundary_one_sided_secant]
    flash_phase_boundary_methods = [SECANT_PHASE_BOUNDARY]

    FLASH_PHASE_BOUNDARY_MAXITER = 100
    FLASH_PHASE_BOUNDARY_MAXITER_XTOL = None
    FLASH_PHASE_BOUNDARY_MAXITER_YTOL = 1e-6


    def flash_phase_boundary(self, zs, phase_frac_check, T=None, P=None, V=None, H=None, S=None, U=None,
                             hot_start=None):
        T_spec = T is not None
        P_spec = P is not None
        V_spec = V is not None
        H_spec = H is not None
        S_spec = S is not None
        U_spec = U is not None
        spec_count = T_spec + P_spec + V_spec + H_spec + S_spec + U_spec
        if spec_count != 1:
            raise ValueError("One specification must be provided")
        if T_spec:
            iter_var, backup_iter_var = 'P', 'H'
            spec_var, spec_val = 'T', T
        if P_spec:
            iter_var, backup_iter_var = 'T', 'H'
            spec_var, spec_val = 'P', P
        if V_spec:
            iter_var, backup_iter_var = 'P', 'H'
            spec_var, spec_val = 'V', V
        if H_spec:
            iter_var, backup_iter_var = 'P', 'T'
            spec_var, spec_val = 'H', H
        if S_spec:
            iter_var, backup_iter_var = 'P', 'T'
            spec_var, spec_val = 'S', S
        if U_spec:
            iter_var, backup_iter_var = 'P', 'T'
            spec_var, spec_val = 'U', U


        for method in self.flash_phase_boundary_algos:
            res, bounding_attempts, iterations = method(flasher=self, zs=zs, spec_var=spec_var, spec_val=spec_val, iter_var=iter_var,
                                                    check=phase_frac_check, hot_start=hot_start,
                                                    xtol=self.FLASH_PHASE_BOUNDARY_MAXITER_XTOL,
                                                    ytol=self.FLASH_PHASE_BOUNDARY_MAXITER_YTOL,
                                                    maxiter=self.FLASH_PHASE_BOUNDARY_MAXITER)

            res.flash_convergence = {'inner_flash_convergence': res.flash_convergence,
                                     'bounding_attempts': bounding_attempts,
                                     'iterations': iterations}


            return res
        raise ValueError("Could not find a solution")



    flash_mixing_phase_boundary_methods = [PT_SECANT_PHASE_MIXING_BOUNDARY,
                                           NAIVE_BISECTION_PHASE_MIXING_BOUNDARY,
                                           SATURATION_SECANT_PHASE_MIXING_BOUNDARY,
                                           ]
    flash_mixing_phase_boundary_algos = [incipient_phase_one_sided_secant,
                                         incipient_phase_bounded_naive,
                                         incipient_liquid_bounded_PT_sat,
                                         ]

    FLASH_MIXING_PHASE_BOUNDARY_MAXITER = 100
    FLASH_MIXING_PHASE_BOUNDARY_XTOL = 1e-6
    FLASH_MIXING_PHASE_BOUNDARY_YTOL = 1e-6

    def flash_mixing_phase_boundary(self, specs, zs_existing, zs_added, boundary='VL'):
        if boundary == 'VL':
            # if we start from a liquid, will converge to a gas with VF=0
            # if we start from a gas, will converge to a liquid with LF=0
            check = VL_boolean_check
        elif boundary == 'LL':
            check = LL_boolean_check
        elif boundary == 'VLL':
            # Hard to say if this flash will converge to a VF=0 or a second liquid fraction of 0
            check = VLL_boolean_check
        elif boundary == 'VLN/LN':
            check = VLN_or_LN_boolean_check
        elif boundary == 'VLL/LL':
            check = VLL_or_LL_boolean_check
        else:
            raise ValueError("Unrecognized boundary")


        for method in self.flash_mixing_phase_boundary_algos:
            # try:
            if method is incipient_phase_one_sided_secant:
                # can only solve two phase prolems
                if boundary not in ('VL',):
                    continue
                res, bounding_attempts, iters, mixing_factor = incipient_phase_one_sided_secant(
                    flasher=self, specs=specs, zs_existing=zs_existing, zs_added=zs_added, check=check, ytol=self.FLASH_MIXING_PHASE_BOUNDARY_YTOL)
            elif method is incipient_phase_bounded_naive:
                res, bounding_attempts, iters, mixing_factor = incipient_phase_bounded_naive(flasher=self, specs=specs, zs_existing=zs_existing, zs_added=zs_added, check=check,
                                                                                    xtol=self.FLASH_MIXING_PHASE_BOUNDARY_XTOL)
            elif method is incipient_liquid_bounded_PT_sat and boundary == 'VL':
                res, bounding_attempts, iters, mixing_factor = incipient_liquid_bounded_PT_sat(flasher=self, specs=specs, zs_existing=zs_existing, zs_added=zs_added, check=check,
                                                                                            xtol=self.FLASH_MIXING_PHASE_BOUNDARY_XTOL)


            res.flash_convergence = {'inner_flash_convergence': res.flash_convergence,
                                     'bounding_attempts': bounding_attempts,
                                     'iterations': iters,
                                     'mixing_factor': mixing_factor}


            return res
            # except Exception as e:
            #     print(e)
        raise ValueError("Could not find a solution")



    def generate_Ts(self, Ts=None, Tmin=None, Tmax=None, pts=50, zs=None,
                    method=None):
        if method is None:
            method = 'physical'

        constants = self.constants

        N = constants.N
        if zs is None:
            zs = [1.0/N]*N

        physical = method == 'physical'
        realistic = method == 'realistic'

        Tcs = constants.Tcs
        Tc = sum([zs[i]*Tcs[i] for i in range(N)])

        if Ts is None:
            if Tmin is None:
                if physical:
                    Tmin = max(p.T_MIN_FIXED for p in self.phases)
                elif realistic:
                    # Round the temperature widely, ensuring consistent rounding
                    Tmin = 1e-2*round(floor(Tc), -1)
            if Tmax is None:
                if physical:
                    Tmax = min(p.T_MAX_FIXED for p in self.phases)
                elif realistic:
                    # Round the temperature widely, ensuring consistent rounding
                    Tmax = min(10*round(floor(Tc), -1), 2000)

            Ts = logspace(log10(Tmin), log10(Tmax), pts)
            # Ensure limits
            Ts[0] = Tmin
            Ts[-1] = Tmax
#            Ts = linspace(Tmin, Tmax, pts)
        return Ts


    def generate_Ps(self, Ps=None, Pmin=None, Pmax=None, pts=50, zs=None,
                    method=None):
        if method is None:
            method = 'physical'

        constants = self.constants

        N = constants.N
        if zs is None:
            zs = [1.0/N]*N

        physical = method == 'physical'
        realistic = method == 'realistic'

        Pcs = constants.Pcs
        Pc = sum([zs[i]*Pcs[i] for i in range(N)])

        if Ps is None:
            if Pmin is None:
                if physical:
                    Pmin = phases.Phase.P_MIN_FIXED
                elif realistic:
                    # Round the pressure widely, ensuring consistent rounding
                    Pmin = min(1e-5*round(floor(Pc), -1), 100)
            if Pmax is None:
                if physical:
                    Pmax = phases.Phase.P_MAX_FIXED
                elif realistic:
                    # Round the pressure widely, ensuring consistent rounding
                    Pmax = min(10*round(floor(Pc), -1), 1e8)

            Ps = logspace(log10(Pmin), log10(Pmax), pts)
        return Ps

    def generate_Vs(self, Vs=None, Vmin=None, Vmax=None, pts=50, zs=None,
                    method=None):
        if method is None:
            method = 'physical'

        constants = self.constants

        N = constants.N
        if zs is None:
            zs = [1.0/N]*N

        physical = method == 'physical'
        realistic = method == 'realistic'

        Vcs = constants.Vcs
        Vc = sum([zs[i]*Vcs[i] for i in range(N)])

        min_bound = None
        CEOS_phases = (phases.CEOSLiquid, phases.CEOSGas)
        for phase in self.phases:
            if isinstance(phase, CEOS_phases):
                c2R = phase.eos_class.c2*R
                Tcs, Pcs = constants.Tcs, constants.Pcs
                b = sum([c2R*Tcs[i]*zs[i]/Pcs[i] for i in range(constants.N)])
                min_bound = b*(1.0 + 1e-15) if min_bound is None else min(min_bound, b*(1.0 + 1e-15))



        if Vs is None:
            if Vmin is None:
                if physical:
                    Vmin = phases.Phase.V_MIN_FIXED
                elif realistic:
                    # Round the pressure widely, ensuring consistent rounding
                    Vmin = round(Vc, 5)
                if Vmin < min_bound:
                    Vmin = min_bound
            if Vmax is None:
                if physical:
                    Vmax = phases.Phase.V_MAX_FIXED
                elif realistic:
                    # Round the pressure widely, ensuring consistent rounding
                    Vmax = 1e5*round(Vc, 5)

            Vs = logspace(log10(Vmin), log10(Vmax), pts)
        return Vs

    def grid_flash(self, zs, Ts=None, Ps=None, Vs=None,
                   VFs=None, SFs=None, Hs=None, Ss=None, Us=None,
                   props=None, store=True):

        flashes = []

        T_spec = Ts is not None
        P_spec = Ps is not None
        V_spec = Vs is not None
        H_spec = Hs is not None
        S_spec = Ss is not None
        U_spec = Us is not None
        VF_spec = VFs is not None
        SF_spec = SFs is not None

        flash_specs = {'zs': zs}
        spec_keys = []
        spec_iters = []
        if T_spec:
            spec_keys.append('T')
            spec_iters.append(Ts)
        if P_spec:
            spec_keys.append('P')
            spec_iters.append(Ps)
        if V_spec:
            spec_keys.append('V')
            spec_iters.append(Vs)
        if H_spec:
            spec_keys.append('H')
            spec_iters.append(Hs)
        if S_spec:
            spec_keys.append('S')
            spec_iters.append(Ss)
        if U_spec:
            spec_keys.append('U')
            spec_iters.append(Us)
        if VF_spec:
            spec_keys.append('VF')
            spec_iters.append(VFs)
        if SF_spec:
            spec_keys.append('SF')
            spec_iters.append(SFs)

        do_props = props is not None
        vectorized_props = isinstance(props, str)

        calc_props = []
        for n0, spec0 in enumerate(spec_iters[0]):
            if do_props:
                row_props = []
            if store:
                row_flashes = []
            for n1, spec1 in enumerate(spec_iters[1]):
                flash_specs = {'zs': zs, spec_keys[0]: spec0, spec_keys[1]: spec1}
                try:
                    state = self.flash(**flash_specs)
                except Exception as e:
                    state = None
                    print(f'Failed trying to flash {flash_specs}, with exception {e}.')

                if store:
                    row_flashes.append(state)
                if do_props:
                    if vectorized_props:
                        state_props = state.value(props) if state is not None else None
                    else:
                        state_props = [state.value(s) for s in props] if state is not None else [None for s in props]

                    row_props.append(state_props)

            if do_props:
                calc_props.append(row_props)
            if store:
                flashes.append(row_flashes)

        if do_props and store:
            return flashes, calc_props
        elif do_props:
            return calc_props
        elif store:
            return flashes
        return None

    def _finish_initialization_base(self):
        solids = self.solids
        liquids = self.liquids
        gas = self.gas
        if solids is None:
            solids = []

        liquids_to_unique_liquids = []
        unique_liquids = []
        liquid_count = len(liquids)
        if liquid_count == 1 or (liquid_count > 1 and all(liquids[0].is_same_model(l) for l in liquids[1:])):
            # All the liquids are the same
            unique_liquids.append(liquids[0])
            liquids_to_unique_liquids.extend([0]*len(liquids))
        else:
            unique_liquid_hashes = []
            # unique_liquid_hashes is not used except in this code block
            for i, l in enumerate(liquids):
                h = l.model_hash()
                if h not in unique_liquid_hashes:
                    unique_liquid_hashes.append(h)
                    unique_liquids.append(l)
                    liquids_to_unique_liquids.append(i)
                else:
                    liquids_to_unique_liquids.append(unique_liquid_hashes.index(h))
        gas_to_unique_liquid = None
        if gas is not None:
            if len(unique_liquids) == 1:
                for i, l in enumerate(liquids):
                    if l.is_same_model(gas, ignore_phase=True):
                        gas_to_unique_liquid = liquids_to_unique_liquids[i]
                        self.ceos_gas_liquid_compatible = True
                        break

        self.gas_to_unique_liquid = gas_to_unique_liquid
        self.liquids_to_unique_liquids = liquids_to_unique_liquids

        self.unique_liquids = unique_liquids
        self.unique_liquid_count = len(unique_liquids)
        self.unique_phases = [gas] + unique_liquids if gas is not None else unique_liquids
        if solids:
            self.unique_phases += solids
        self.unique_phase_count = (1 if gas is not None else 0) + self.unique_liquid_count + len(solids)

        self.T_MIN_FLASH = self.T_MIN_FLASH_ANY = max(p.T_MIN_FLASH for p in self.phases)
        self.T_MAX_FLASH = self.T_MAX_FLASH_ANY = min(p.T_MAX_FLASH for p in self.phases)
        vectorized = False
        vectorized_statuses = {i.vectorized for i in self.phases}
        if len(vectorized_statuses) > 1:
            raise ValueError("Can only perform flashes with all phases in a numpy basis or all phases in a pure Python basis")
        self.vectorized = vectorized_statuses.pop()

        self.supports_lnphis_args = all(p.supports_lnphis_args for p in self.phases)

        # Make the phases aware of the constants and properties
        constants = self.constants
        correlations = self.correlations
        for p in self.phases:
            if hasattr(p, 'constants') and p.constants is not constants:
                raise ValueError("Provided phase is associated with a different constants object")
            p.constants = constants
            if hasattr(p, 'correlations') and p.correlations is not correlations:
                raise ValueError("Provided phase is associated with a different correlations object")
            p.correlations = correlations


    def debug_grid_flash(self, zs, check0, check1, Ts=None, Ps=None, Vs=None,
                         VFs=None, SFs=None, Hs=None, Ss=None, Us=None,
                         retry=False, verbose=True):

        matrix_spec_flashes = []
        matrix_flashes = []
        nearest_check_prop = 'T' if 'T' not in (check0, check1) else 'P'

        T_spec = Ts is not None
        P_spec = Ps is not None
        V_spec = Vs is not None
        H_spec = Hs is not None
        S_spec = Ss is not None
        U_spec = Us is not None
        VF_spec = VFs is not None
        SF_spec = SFs is not None

        flash_specs = {'zs': zs}
        spec_keys = []
        spec_iters = []
        if T_spec:
            spec_keys.append('T')
            spec_iters.append(Ts)
        if P_spec:
            spec_keys.append('P')
            spec_iters.append(Ps)
        if V_spec:
            spec_keys.append('V')
            spec_iters.append(Vs)
        if H_spec:
            spec_keys.append('H')
            spec_iters.append(Hs)
        if S_spec:
            spec_keys.append('S')
            spec_iters.append(Ss)
        if U_spec:
            spec_keys.append('U')
            spec_iters.append(Us)
        if VF_spec:
            spec_keys.append('VF')
            spec_iters.append(VFs)
        if SF_spec:
            spec_keys.append('SF')
            spec_iters.append(SFs)

        V_set = {check1, check0}
        TV_iter = V_set == {'T', 'V'}
        PV_iter = V_set == {'P', 'V'}
        high_prec_V = TV_iter or PV_iter

        for n0, spec0 in enumerate(spec_iters[0]):
            row = []
            row_flashes = []
            row_spec_flashes = []
            for n1, spec1 in enumerate(spec_iters[1]):

                flash_specs = {'zs': zs, spec_keys[0]: spec0, spec_keys[1]: spec1}
                state = self.flash(**flash_specs)

                check0_spec = getattr(state, check0)
                try:
                    check0_spec = check0_spec()
                except:
                    pass
                check1_spec = getattr(state, check1)
                try:
                    check1_spec = check1_spec()
                except:
                    pass

                kwargs = {}
                kwargs[check0] = check0_spec
                kwargs[check1] = check1_spec

                # TV_iter is important to always do
                if TV_iter:
                    kwargs['V'] = state.V_iter(force=False)
                kwargs['retry'] = retry
                kwargs['solution'] = lambda new: abs(new.value(nearest_check_prop) - state.value(nearest_check_prop))
                try:
                    new = self.flash(**kwargs)
                    if PV_iter:
                        # Do a check here on tolerance
                        err = abs((new.value(nearest_check_prop) - state.value(nearest_check_prop))/state.value(nearest_check_prop))
                        if err > 1e-8:
                            kwargs['V'] = state.V_iter(force=True)
                            new = self.flash(**kwargs)
                except Exception as e:
                    # Was it a precision issue? Some flashes can be brutal
                    if 'V' in kwargs:
                        try:
                             kwargs['V'] = state.V_iter(True)
                             new = self.flash(**kwargs)
                        except Exception as e2:
                            new = None
                            if verbose:
                                print(f'Failed trying to flash {kwargs}, from original point {flash_specs}, with exception {e}.')
                    else:
                        new = None
                        if verbose:
                            print(f'Failed trying to flash {kwargs}, from original point {flash_specs}, with exception {e}.')
                row_spec_flashes.append(state)
                row_flashes.append(new)

            matrix_spec_flashes.append(row_spec_flashes)
            matrix_flashes.append(row_flashes)
        return matrix_spec_flashes, matrix_flashes

    def debug_err_flash_grid(self, matrix_spec_flashes, matrix_flashes,
                             check, method='rtol', verbose=True):
        matrix = []
        N0 = len(matrix_spec_flashes)
        N1 = len(matrix_spec_flashes[0])
        for i in range(N0):
            row = []
            for j in range(N1):
                state = matrix_spec_flashes[i][j]
                new = matrix_flashes[i][j]

                act = getattr(state, check)
                try:
                    act = act()
                except:
                    pass

                if new is None:
                    err = 1.0
                else:
                    calc = getattr(new, check)
                    try:
                        calc = calc()
                    except:
                        pass
                    if method == 'rtol':
                        err = abs((act - calc)/act)

                if err > 1e-6 and verbose:
                    try:
                        print([matrix_flashes[i][j].T, matrix_spec_flashes[i][j].T])
                        print([matrix_flashes[i][j].P, matrix_spec_flashes[i][j].P])
                        print(matrix_flashes[i][j], matrix_spec_flashes[i][j])
                    except:
                        pass
                row.append(err)
            matrix.append(row)
        return matrix


    def TPV_inputs(self, spec0='T', spec1='P', check0='P', check1='V', prop0='T',
                   Ts=None, Tmin=None, Tmax=None,
                   Ps=None, Pmin=None, Pmax=None,
                   Vs=None, Vmin=None, Vmax=None,
                   VFs=None, SFs=None,
                   auto_range=None, zs=None, pts=50,
                   trunc_err_low=1e-15, trunc_err_high=None, plot=True,
                   show=True, color_map=None, retry=False, verbose=True):

        specs = []
        for a_spec in (spec0, spec1):
            if 'T' == a_spec:
                Ts = self.generate_Ts(Ts=Ts, Tmin=Tmin, Tmax=Tmax, pts=pts, zs=zs,
                                      method=auto_range)
                specs.append(Ts)
            elif 'P' == a_spec:
                Ps = self.generate_Ps(Ps=Ps, Pmin=Pmin, Pmax=Pmax, pts=pts, zs=zs,
                                  method=auto_range)
                specs.append(Ps)
            elif 'V' == a_spec:
                Vs = self.generate_Vs(Vs=Vs, Vmin=Vmin, Vmax=Vmax, pts=pts, zs=zs,
                                  method=auto_range)
                specs.append(Vs)
            elif 'VF' == a_spec:
                if VFs is None:
                    VFs = linspace(0, 1, pts)
                specs.append(VFs)
            elif 'SF' == a_spec:
                if SFs is None:
                    SFs = linspace(0, 1, pts)
                specs.append(SFs)

        specs0, specs1 = specs
        matrix_spec_flashes, matrix_flashes = self.debug_grid_flash(zs,
            check0=check0, check1=check1, Ts=Ts, Ps=Ps, Vs=Vs, VFs=VFs,
            retry=retry, verbose=verbose)

        errs = self.debug_err_flash_grid(matrix_spec_flashes,
            matrix_flashes, check=prop0, verbose=verbose)

        if plot:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.colors import LogNorm
#            plt.ioff()
            X, Y = np.meshgrid(specs0, specs1)
            z = np.array(errs).T
            fig, ax = plt.subplots()
            z[np.where(abs(z) < trunc_err_low)] = trunc_err_low
            if trunc_err_high is not None:
                z[np.where(abs(z) > trunc_err_high)] = trunc_err_high

            if color_map is None:
                color_map = cm.viridis

            # im = ax.pcolormesh(X, Y, z, cmap=cm.PuRd, norm=LogNorm())
            im = ax.pcolormesh(X, Y, z, cmap=color_map, norm=LogNorm(vmin=trunc_err_low, vmax=trunc_err_high))
            # im = ax.pcolormesh(X, Y, z, cmap=cm.viridis, norm=LogNorm(vmin=1e-7, vmax=1))
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Relative error')

            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(spec0)
            ax.set_ylabel(spec1)

            max_err = np.max(errs)
            if trunc_err_low is not None and max_err < trunc_err_low:
                max_err = 0
            if trunc_err_high is not None and max_err > trunc_err_high:
                max_err = trunc_err_high

            ax.set_title(f'{check0} {check1} validation of {prop0}; Reference flash {spec0} {spec1}; max err {max_err:.1e}')


            if show:
                plt.show()

            return matrix_spec_flashes, matrix_flashes, errs, fig
        return matrix_spec_flashes, matrix_flashes, errs

    def grid_props(self, spec0='T', spec1='P', prop='H',
                   Ts=None, Tmin=None, Tmax=None,
                   Ps=None, Pmin=None, Pmax=None,
                   Vs=None, Vmin=None, Vmax=None,
                   VFs=None, SFs=None,
                   auto_range=None, zs=None, pts=50, plot=True,
                   show=True, color_map=None):

        specs = []
        for a_spec in (spec0, spec1):
            if 'T' == a_spec:
                Ts = self.generate_Ts(Ts=Ts, Tmin=Tmin, Tmax=Tmax, pts=pts, zs=zs,
                                      method=auto_range)
                specs.append(Ts)
            if 'P' == a_spec:
                Ps = self.generate_Ps(Ps=Ps, Pmin=Pmin, Pmax=Pmax, pts=pts, zs=zs,
                                  method=auto_range)
                specs.append(Ps)
            if 'V' == a_spec:
                Vs = self.generate_Vs(Vs=Vs, Vmin=Vmin, Vmax=Vmax, pts=pts, zs=zs,
                                  method=auto_range)
                specs.append(Vs)
            if 'VF' == a_spec:
                if VFs is None:
                    VFs = linspace(0, 1, pts)
                specs.append(VFs)
            if 'SF' == a_spec:
                if SFs is None:
                    SFs = linspace(0, 1, pts)
                specs.append(SFs)

        specs0, specs1 = specs
        props = self.grid_flash(zs, Ts=Ts, Ps=Ps, Vs=Vs, VFs=VFs, props=prop, store=False)
#        props = []
#        pts_iter = range(pts)
#        for i in pts_iter:
#            row = []
#            for j in pts_iter:
#                flash = matrix_flashes[i][j]
#                try:
#                    v = getattr(flash, prop)
#                    try:
#                        v = v()
#                    except:
#                        pass
#                except:
#                    v = None
#                row.append(v)
#            props.append(row)

        if plot:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.colors import LogNorm
            X, Y = np.meshgrid(specs0, specs1)
            z = np.array(props).T
            fig, ax = plt.subplots()

            if color_map is None:
                color_map = cm.viridis

            if np.any(np.array(props) < 0):
                norm = None
            else:
                norm = LogNorm()
            im = ax.pcolormesh(X, Y, z, cmap=color_map, norm=norm)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(prop)

            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(spec0)
            ax.set_ylabel(spec1)

#            ax.set_title()
            if show:
                plt.show()
            return props, fig

        return props

    def debug_mixing_phase_boundary_PT(self, zs, zs_mixing, Pmin=None, Pmax=None,
                                       Tmin=None, Tmax=None, pts=50,
                ignore_errors=True, values=False, verbose=False, show=False,
                T_pts=None, P_pts=None, Ts=None, Ps=None, boundary='VL'): # pragma: no cover
        if not has_matplotlib() and not values:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if Pmin is None:
            Pmin = 1e2
        if Pmax is None:
            Pmax = min(self.constants.Pcs)
        if Tmin is None:
            Tmin = min(self.constants.Tms)*.9
        if Tmax is None:
            Tmax = max(self.constants.Tcs)*1.5
        if T_pts is None:
            T_pts = pts
        if P_pts is None:
            P_pts = pts

        Ts = self.generate_Ts(Ts=Ts, Tmin=Tmin, Tmax=Tmax, pts=T_pts, zs=zs)
        Ps = self.generate_Ps(Ps=Ps, Pmin=Pmin, Pmax=Pmax, pts=P_pts, zs=zs)

        matrix = []
        for T in Ts:
            row = []
            for P in Ps:
                try:
                    state = self.flash_mixing_phase_boundary(specs={'T': T, 'P': P}, zs_existing=zs, zs_added=zs_mixing, boundary=boundary)
                    row.append(state.flash_convergence['mixing_factor'])
                except Exception as e:
                    if verbose:
                        print([T, P, e])
                    if ignore_errors:
                        row.append(nan)
                    else:
                        raise e
            matrix.append(row)

        if values:
            return Ts, Ps, matrix

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        Ts, Ps = np.meshgrid(Ts, Ps)
        im = ax.pcolormesh(Ts, Ps, matrix, cmap=plt.get_cmap('viridis'))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Factor')

        ax.set_yscale('log')
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Pressure [Pa]')
        plt.title(f'PT flash mixing {boundary} boundary flashes, zs={zs}, zs_mixing={zs_mixing}')
        if show:
            plt.show()
        else:
            return fig

    def debug_PT(self, zs, Pmin=None, Pmax=None, Tmin=None, Tmax=None, pts=50,
                ignore_errors=True, values=False, verbose=False, show=False,
                T_pts=None, P_pts=None, Ts=None, Ps=None): # pragma: no cover
        if not has_matplotlib() and not values:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if Pmin is None:
            Pmin = 1e4
        if Pmax is None:
            Pmax = min(self.constants.Pcs)
        if Tmin is None:
            Tmin = min(self.constants.Tms)*.9
        if Tmax is None:
            Tmax = max(self.constants.Tcs)*1.5
        if T_pts is None:
            T_pts = pts
        if P_pts is None:
            P_pts = pts

        Ts = self.generate_Ts(Ts=Ts, Tmin=Tmin, Tmax=Tmax, pts=T_pts, zs=zs)
        Ps = self.generate_Ps(Ps=Ps, Pmin=Pmin, Pmax=Pmax, pts=P_pts, zs=zs)

#        Ps = logspace(log10(Pmin), log10(Pmax), pts)
#        Ts = linspace(Tmin, Tmax, pts)

        matrix = []
        for T in Ts:
            row = []
            for P in Ps:
                try:
                    state = self.flash(T=T, P=P, zs=zs)
                    row.append(state.phase)
                except Exception as e:
                    if verbose:
                        print([T, P, e])
                    if ignore_errors:
                        row.append('F')
                    else:
                        raise e
            matrix.append(row)

        if values:
            return Ts, Ps, matrix


        regions = {'V': 1, 'L': 2, 'S': 3, 'VL': 4, 'LL': 5, 'VLL': 6,
                       'VLS': 7, 'VLLS': 8, 'VLLSS': 9, 'LLL': 10, 'VLLL': 11,
                       'VLLLL': 12, 'LLLL': 13, 'F': 0}

        used_regions = set()
        for row in matrix:
            for v in row:
                used_regions.add(v)

        region_keys = list(regions.keys())
        used_keys = [i for i in region_keys if i in used_regions]

        regions_keys = [n for _, n in sorted(zip([regions[i] for i in used_keys], used_keys))]
        used_values = [regions[i] for i in regions_keys]

        new_map = list(range(len(used_values)))
        new_map_trans = {i: j for i, j in zip(used_values, new_map)}

        dat = [[new_map_trans[regions[matrix[j][i]]] for j in range(pts)] for i in range(pts)]
#        print(dat)
        import matplotlib.pyplot as plt
        from matplotlib import colors
        fig, ax = plt.subplots()
        Ts, Ps = np.meshgrid(Ts, Ps)

        # need 3 more
        cmap = colors.ListedColormap(['y','b','r', 'g', 'c', 'm', 'k', 'fuchsia', 'gold', 'lime', 'indigo', 'cyan'][0:len(used_values)])

        vmax = len(used_values) - 1

#        ax.scatter(Ts,Ps, s=dat)
        print(np.array(Ts).shape, np.array(Ps).shape, np.array(dat).shape)
        im = ax.pcolormesh(Ts, Ps, dat, cmap=cmap, norm=colors.Normalize(vmin=0, vmax=vmax)) # , cmap=color_map, norm=LogNorm()
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Phase')
        cbar.ax.locator_params(nbins=len(used_values))
#        cbar = plt.colorbar()

#        cs = ax.contourf(Ts, Ps, dat, levels=list(sorted(regions.values())))
#        cbar = fig.colorbar(ax)

        # used_region_keys = regions_keys
        cbar.ax.set_yticklabels(regions_keys)

        # cbar.ax.set_yticklabels([n for _, n in sorted(zip(regions.values(), regions.keys()))])
#        cbar.ax.set_yticklabels(regions_keys)
#        ax.set_yscale('log')
        plt.yscale('log')
        plt.xlabel('System temperature, K')
        plt.ylabel('System pressure, Pa')
#        plt.imshow(dat, interpolation='nearest')
#        plt.legend(loc='best', fancybox=True, framealpha=0.5)
#        return fig, ax

        if len(zs) > 4:
            zs = '...'
        plt.title(f'PT system flashes, zs={zs}')
        if show:
            plt.show()
        else:
            return fig


    def plot_TP(self, zs, Tmin=None, Tmax=None, pts=50, branches=None,
                ignore_errors=True, values=False, show=True, hot=False): # pragma: no cover
        r'''Method to create a plot of the phase envelope as can be calculated
        from a series of temperature & vapor fraction spec flashes. By default
        vapor fractions of 0 and 1 are plotted; additional vapor fraction
        specifications can be specified in the `branches` argument as a list.

        Parameters
        ----------
        zs : list[float]
            Mole fractions of the feed, [-]
        Tmin : float, optional
            Minimum temperature to begin the plot, [K]
        Tmax : float, optional
            Maximum temperature to end the plot, [K]
        pts : int, optional
           The number of points to calculated for each vapor fraction value,
           [-]
        branches : list[float], optional
            Extra vapor fraction values to plot, [-]
        ignore_errors : bool, optional
            Whether to fail on a calculation failure, or to ignore the bad
            point, [-]
        values : bool, optional
            If True, the calculated values will be returned instead of
            plotted, [-]
        show : bool, optional
            If True, the plot will be created and displayed; if False and `values`
            is False, the plot Figure object will be returned but not displayed;
            and if False and `values` is False, no plot will be created or shown
            [-]
        hot : bool, optional
            Whether to restart the next flash from the previous flash or not
            (intended to speed the call when True), [-]

        Returns
        -------
        Ts : list[float]
            Temperatures, [K]
        P_dews : list[float]
            Bubble point pressures at the evaluated points, [Pa]
        P_bubbles : list[float]
            Dew point pressures at the evaluated points, [Pa]
        branch_Ps : None or list[list[float]]
            Pressures which yield the equilibrium vapor fractions specified;
            formatted as [[P1_VFx, P2_VFx, ... Pn_VFx], ...,
            [P1_VFy, P2_VFy, ... Pn_VFy]], [Pa]
        '''
        if not has_matplotlib() and not values:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if not Tmin:
            Tmin = min(self.constants.Tms)
        if not Tmax:
            Tmax = min(self.constants.Tcs)
        Ts = linspace(Tmin, Tmax, pts)
        P_dews = []
        P_bubbles = []
        branch = branches is not None
        if branch:
            branch_Ps = [[] for i in range(len(branches))]
        else:
            branch_Ps = None

        state_TVF0, state_TVF1 = None, None
        for T in Ts:
            if not hot:
                state_TVF0, state_TVF1 = None, None
            try:
                state_TVF0 = self.flash(T=T, VF=0.0, zs=zs, hot_start=state_TVF0)
                assert state_TVF0 is not None
                P_bubbles.append(state_TVF0.P)
            except Exception as e:
                if ignore_errors:
                    P_bubbles.append(None)
                else:
                    raise e
            try:
                state_TVF1 = self.flash(T=T, VF=1.0, zs=zs, hot_start=state_TVF1)
                assert state_TVF1 is not None
                P_dews.append(state_TVF1.P)
            except Exception as e:
                if ignore_errors:
                    P_dews.append(None)
                else:
                    raise e

            if branch:
                for VF, Ps in zip(branches, branch_Ps):
                    try:
                        state = self.flash(T=T, VF=VF, zs=zs)
                        Ps.append(state.P)
                    except Exception as e:
                        if ignore_errors:
                            Ps.append(None)
                        else:
                            raise e
        if values and not show:
            return Ts, P_dews, P_bubbles, branch_Ps

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.semilogy(Ts, P_dews, label='TP dew point curve')
        plt.semilogy(Ts, P_bubbles, label='TP bubble point curve')
        plt.xlabel('System temperature, K')
        plt.ylabel('System pressure, Pa')
        plt.title(f'PT system curve, zs={zs}')
        if branch:
            for VF, Ps in zip(branches, branch_Ps):
                plt.semilogy(Ts, Ps, label=f'TP curve for VF={VF}')
        plt.legend(loc='best')
        if show:
            plt.show()
        if values:
            return Ts, P_dews, P_bubbles, branch_Ps
        else:
            return fig


    def plot_PT(self, zs, Pmin=None, Pmax=None, pts=50, branches=[],
                ignore_errors=True, values=False, show=True, hot=False): # pragma: no cover
        r'''Method to create a plot of the phase envelope as can be calculated
        from a series of pressure & vapor fraction spec flashes. By default
        vapor fractions of 0 and 1 are plotted; additional vapor fraction
        specifications can be specified in the `branches` argument as a list.

        Parameters
        ----------
        zs : list[float]
            Mole fractions of the feed, [-]
        Pmin : float, optional
            Minimum pressure to begin the plot, [Pa]
        Pmax : float, optional
            Maximum pressure to end the plot, [Pa]
        pts : int, optional
           The number of points to calculated for each vapor fraction value,
           [-]
        branches : list[float], optional
            Extra vapor fraction values to plot, [-]
        ignore_errors : bool, optional
            Whether to fail on a calculation failure, or to ignore the bad
            point, [-]
        values : bool, optional
            If True, the calculated values will be returned instead of
            plotted, [-]
        show : bool, optional
            If True, the plot will be created and displayed; if False and `values`
            is False, the plot Figure object will be returned but not displayed;
            and if False and `values` is False, no plot will be created or shown
            [-]
        hot : bool, optional
            Whether to restart the next flash from the previous flash or not
            (intended to speed the call when True), [-]

        Returns
        -------
        Ps : list[float]
            Pressures, [Pa]
        T_dews : list[float]
            Bubble point temperatures at the evaluated points, [K]
        T_bubbles : list[float]
            Dew point temperatures at the evaluated points, [K]
        branch_Ts : None or list[list[float]]
            Temperatures which yield the equilibrium vapor fractions specified;
            formatted as [[T1_VFx, T2_VFx, ... Tn_VFx], ...,
            [T1_VFy, T2_VFy, ... Tn_VFy]], [k]
        '''
        if not has_matplotlib() and not values:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if not Pmin:
            Pmin = 1e4
        if not Pmax:
            Pmax = min(self.constants.Pcs)
        Ps = logspace(log10(Pmin), log10(Pmax), pts)
        T_dews = []
        T_bubbles = []
        branch = bool(len(branches))
        if branch:
            branch_Ts = [[] for i in range(len(branches))]
        else:
            branch_Ts = None
        state_PVF0, state_PVF1 = None, None
        for P in Ps:
            if not hot:
                state_PVF0, state_PVF1 = None, None
            try:
                state_PVF0 = self.flash(P=P, VF=0, zs=zs, hot_start=state_PVF0)
                assert state_PVF0 is not None
                T_bubbles.append(state_PVF0.T)
            except Exception as e:
                if ignore_errors:
                    T_bubbles.append(None)
                else:
                    raise e
            try:
                state_PVF1 = self.flash(P=P, VF=1, zs=zs, hot_start=state_PVF1)
                assert state_PVF1 is not None
                T_dews.append(state_PVF1.T)
            except Exception as e:
                if ignore_errors:
                    T_dews.append(None)
                else:
                    raise e

            if branch:
                for VF, Ts in zip(branches, branch_Ts):
                    try:
                        state = self.flash(P=P, VF=VF, zs=zs)
                        Ts.append(state.T)
                    except Exception as e:
                        if ignore_errors:
                            Ts.append(None)
                        else:
                            raise e
        if values:
            return Ps, T_dews, T_bubbles, branch_Ts
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.plot(Ps, T_dews, label='PT dew point curve')
        plt.plot(Ps, T_bubbles, label='PT bubble point curve')
        plt.xlabel('System pressure, Pa')
        plt.ylabel('System temperature, K')
        plt.title(f'PT system curve, zs={zs}')
        if branch:
            for VF, Ts in zip(branches, branch_Ts):
                plt.plot(Ps, Ts, label=f'PT curve for VF={VF}')
        plt.legend(loc='best')

        if show:
            plt.show()
        else:
            return fig

    def plot_ternary(self, T=None, P=None, scale=10): # pragma: no cover
        r'''Method to create a ternary plot of the system at either a specified
        temperature or pressure.

        Parameters
        ----------
        T : float, optional
            Temperature, [K]
        P : float, optional
            Pressure, [Pa]
        '''
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        try:
            import ternary
        except:
            raise Exception('Optional dependency python-ternary is required for ternary plotting')
        if self.N != 3:
            raise Exception('Ternary plotting requires a mixture of exactly three components')

        is_T_spec = T is not None
        if not is_T_spec and P is None:
            raise ValueError("Either T or P must be specified")
        values = []

        cond = {'T': T} if is_T_spec else {'P': P}

        def dew_at_zs(zs):
            res = self.flash(zs=zs, VF=0, **cond)
            if is_T_spec:
                values.append(res.P)
                return res.P
            else:
                values.append(res.T)
                return res.T

        def bubble_at_zs(zs):
            res = self.flash(zs=zs, VF=1, **cond)
            if is_T_spec:
                return res.P
            else:
                return res.T

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        axes_colors = {'b': 'g', 'l': 'r', 'r':'b'}
        ticks = [round(i / float(10), 1) for i in range(10+1)]

        fig, ax = plt.subplots(1, 3, gridspec_kw = {'width_ratios':[4, 4, 1]})
        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")
        names = self.constants.aliases

        for axis, f, i in zip(ax[0:2], [dew_at_zs, bubble_at_zs], [0, 1]):
            figure, tax = ternary.figure(ax=axis, scale=scale)
            figure.set_size_inches(12, 4)
            if not i:
                tax.heatmapf(f, boundary=True, colorbar=False, vmin=0)
            else:
                tax.heatmapf(f, boundary=True, colorbar=False, vmin=0, vmax=max(values))

            tax.boundary(linewidth=2.0)
            tax.left_axis_label(f"mole fraction {names[1]}", offset=0.16, color=axes_colors['l'])
            tax.right_axis_label(f"mole fraction {names[0]}", offset=0.16, color=axes_colors['r'])
            tax.bottom_axis_label(f"mole fraction {names[2]}", offset=0.16, color=axes_colors['b'])

            tax.ticks(ticks=ticks, axis='rlb', linewidth=1, clockwise=True,
                      axes_colors=axes_colors, offset=0.03, tick_formats="%.1f")

            tax.gridlines(multiple=scale/10., linewidth=2,
                          horizontal_kwargs={'color':axes_colors['b']},
                          left_kwargs={'color':axes_colors['l']},
                          right_kwargs={'color':axes_colors['r']},
                          alpha=0.5)

        norm = plt.Normalize(vmin=0, vmax=max(values))
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
        sm._A = []
        cb = plt.colorbar(sm, ax=ax[2])
        text = 'Pressure, [Pa]' if is_T_spec else 'Temperature, [K]'
        cb.set_label(text, rotation=270, ha='center', va='center')
        cb.locator = mpl.ticker.LinearLocator(numticks=7)
        cb.formatter = mpl.ticker.ScalarFormatter()
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        # plt.tight_layout()
        if is_T_spec:
            text = f"Bubble pressure vs composition (left) and dew pressure vs composition (right) at {T} K"
        else:
            text = f"Bubble temperature vs composition (left) and dew temperature vs composition (right) at {P} Pa"
        fig.suptitle(text, fontsize=14)
        fig.subplots_adjust(top=0.85)
        plt.show()


    def plot_Txy(self, P, pts=30, ignore_errors=True, values=False, show=True): # pragma: no cover
        r'''Method to create a Txy plot for a binary system (holding pressure
        constant); the mole fraction of the first species is varied.

        Parameters
        ----------
        P : float
            Pressure for the plot, [Pa]
        pts : int, optional
           The number of points to calculated for each vapor fraction value,
           [-]
        ignore_errors : bool, optional
            Whether to fail on a calculation failure, or to ignore the bad
            point, [-]
        values : bool, optional
            If True, the calculated values will be returned instead of
            plotted, [-]
        show : bool, optional
            If True, the plot will be created and displayed; if False and `values`
            is False, the plot Figure object will be returned but not displayed;
            and if False and `values` is False, no plot will be created or shown
            [-]

        Returns
        -------
        fig : Figure
            Plot object, [-]
        z1 : list[float]
            Mole fractions of the first component at each point, [-]
        z2 : list[float]
            Mole fractions of the second component at each point, [-]
        T_dews : list[float]
            Bubble point temperatures at the evaluated points, [K]
        T_bubbles : list[float]
            Dew point temperatures at the evaluated points, [K]
        '''
        if not has_matplotlib() and values is not False:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if self.N != 2:
            raise Exception('Txy plotting requires a mixture of exactly two components')
        z1 = linspace(0, 1, pts)
        z2 = [1.0 - zi for zi in z1]
        Ts_dew = []
        Ts_bubble = []

        for i in range(pts):
            try:
                res = self.flash(P=P, VF=0, zs=[z1[i], z2[i]])
                Ts_bubble.append(res.T)
            except Exception as e:
                if ignore_errors:
                    Ts_bubble.append(None)
                else:
                    raise e
            try:
                res = self.flash(P=P, VF=1, zs=[z1[i], z2[i]])
                Ts_dew.append(res.T)
            except Exception as e:
                if ignore_errors:
                    Ts_dew.append(None)
                else:
                    raise e
        if values:
            return z1, z2, Ts_dew, Ts_bubble
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        names = self.constants.aliases
        plt.title(f'Txy diagram at P={P} Pa')
        plt.plot(z1, Ts_dew, label='Dew temperature, K')
        plt.plot(z1, Ts_bubble, label='Bubble temperature, K')
        plt.xlabel(f'Mole fraction {names[0]}')
        plt.ylabel('System temperature, K')
        plt.legend(loc='best')
        if show:
            plt.show()
        else:
            return fig

    def plot_Pxy(self, T, pts=30, ignore_errors=True, values=False, show=True): # pragma: no cover
        r'''Method to create a Pxy plot for a binary system (holding temperature
        constant); the mole fraction of the first species is varied.

        Parameters
        ----------
        T : float
            Temperature for the plot, [K]
        pts : int, optional
           The number of points to calculated for each vapor fraction value,
           [-]
        ignore_errors : bool, optional
            Whether to fail on a calculation failure, or to ignore the bad
            point, [-]
        values : bool, optional
            If True, the calculated values will be returned instead of
            plotted, [-]
        show : bool, optional
            If True, the plot will be created and displayed; if False and `values`
            is False, the plot Figure object will be returned but not displayed;
            and if False and `values` is False, no plot will be created or shown
            [-]

        Returns
        -------
        fig : Figure
            Plot object, [-]
        z1 : list[float]
            Mole fractions of the first component at each point, [-]
        z2 : list[float]
            Mole fractions of the second component at each point, [-]
        P_dews : list[float]
            Bubble point pressures at the evaluated points, [Pa]
        P_bubbles : list[float]
            Dew point pressures at the evaluated points, [Pa]
        '''
        if not has_matplotlib() and values is not False:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if self.N != 2:
            raise Exception('Pxy plotting requires a mixture of exactly two components')
        z1 = linspace(0, 1, pts)
        z2 = [1.0 - zi for zi in z1]
        Ps_dew = []
        Ps_bubble = []
        names = self.constants.aliases

        for i in range(pts):
            try:
                res = self.flash(T=T, VF=0, zs=[z1[i], z2[i]])
                Ps_bubble.append(res.P)
            except Exception as e:
                if ignore_errors:
                    Ps_bubble.append(None)
                else:
                    raise e
            try:
                res = self.flash(T=T, VF=1, zs=[z1[i], z2[i]])
                Ps_dew.append(res.P)
            except Exception as e:
                if ignore_errors:
                    Ps_dew.append(None)
                else:
                    raise e
        if values:
            return z1, z2, Ps_dew, Ps_bubble

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.title(f'Pxy diagram at T={T} K')
        plt.plot(z1, Ps_dew, label='Dew pressure')
        plt.plot(z1, Ps_bubble, label='Bubble pressure')
        plt.xlabel(f'Mole fraction {names[0]}')
        plt.ylabel('System pressure, Pa')
        plt.legend(loc='best')
        if show:
            plt.show()
        else:
            return fig

    def plot_xy(self, P=None, T=None, pts=30, ignore_errors=True, values=False,
                show=True, VF=0.0): # pragma: no cover
        r'''Method to create a xy diagram for a binary system. Either a
        temperature or pressure can be specified. By default, bubble point
        flashes are performed; this can be varied by changing `VF`.

        Parameters
        ----------
        P : float, optional
            The specified pressure, [Pa]
        T : float, optional
            The specified temperature, [K]
        pts : int, optional
           The number of points in the plot [-]
        ignore_errors : bool, optional
            Whether to fail on a calculation failure, or to ignore the bad
            point, [-]
        values : bool, optional
            If True, the calculated values will be returned instead of
            plotted, [-]
        show : bool, optional
            If True, the plot will be created and displayed; if False and `values`
            is False, the plot Figure object will be returned but not displayed;
            and if False and `values` is False, no plot will be created or shown
            [-]

        Returns
        -------
        fig : Figure
            Plot object, [-]
        z1 : list[float]
            Overall mole fractions of the first component at each point, [-]
        z2 : list[float]
            Overall mole fractions of the second component at each point, [-]
        x1 : list[float]
            Liquid mole fractions of component 1 at each point, [-]
        y1 : list[float]
            Vapor mole fractions of component 1 at each point, [-]
        '''
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        if self.N != 2:
            raise Exception('xy plotting requires a mixture of exactly two components')
        z1 = linspace(0.0, 1.0, pts)
        z2 = [1.0 - zi for zi in z1]
        y1_bubble = []
        x1_bubble = []
        for i in range(pts):
            try:
                if T is not None:
                    res = self.flash(T=T, VF=VF, zs=[z1[i], z2[i]])
                elif P is not None:
                    res = self.flash(P=P, VF=VF, zs=[z1[i], z2[i]])
                x1_bubble.append(res.liquid_bulk.zs[0])
                y1_bubble.append(res.gas.zs[0])
            except Exception as e:
                if ignore_errors:
                    print('Failed on pt %d' %(i), e)
                else:
                    raise e
        if values:
            return z1, z2, x1_bubble, y1_bubble
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if T is not None:
            plt.title(f'xy diagram at T={T} K (varying P)')
        else:
            plt.title(f'xy diagram at P={P} Pa (varying T)')
        names = self.constants.aliases
        plt.xlabel(f'Liquid mole fraction {names[0]}')
        plt.ylabel(f'Vapor mole fraction {names[0]}')
        plt.plot(x1_bubble, y1_bubble, '-', label='liquid vs vapor composition')
        plt.legend(loc='best')
        plt.plot([0, 1], [0, 1], '--')
        plt.axis((0,1,0,1))
        if show:
            plt.show()
        else:
            return fig

    def V_liquids_ref(self):
        r'''Method to calculate and return the liquid reference molar volumes
        according to the temperature variable `T_liquid_volume_ref` of
        :obj:`thermo.bulk.BulkSettings`.

        Returns
        -------
        V_liquids_ref : list[float]
            Liquid molar volumes at the reference condition, [m^3/mol]

        Notes
        -----
        '''
        T_liquid_volume_ref = self.settings.T_liquid_volume_ref
        if T_liquid_volume_ref == 298.15:
            Vls = self.constants.Vml_STPs
        elif T_liquid_volume_ref == 288.7055555555555:
            Vls = self.constants.Vml_60Fs
        else:
            Vls = [i(T_liquid_volume_ref, None) for i in self.correlations.VolumeLiquids]
        return Vls


    @property
    def water_index(self):
        r'''The index of the component water in the components. None if water
        is not present. Water is recognized by its CAS number.

        Returns
        -------
        water_index : int
            The index of the component water, [-]

        Notes
        -----
        '''
        try:
            return self._water_index
        except AttributeError:
            pass

        try:
            self._water_index = self.constants.CASs.index(CAS_H2O)
        except ValueError:
            self._water_index = None
        return self._water_index
