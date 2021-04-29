# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

.. contents:: :local:

Main Interfaces
===============

Pure Components
---------------
.. autoclass:: FlashPureVLS
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Vapor-Liquid Systems
--------------------
.. autoclass:: FlashVL
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Vapor and Multiple Liquid Systems
---------------------------------
.. autoclass:: FlashVLN
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Base Flash Class
----------------
.. autoclass:: Flash
   :show-inheritance:
   :members: flash, plot_TP
   :exclude-members:


Specific Flash Algorithms
=========================
It is recommended to use the Flash classes, which are designed to have generic
interfaces. The implemented specific flash algorithms may be changed in the
future, but reading their source code may be helpful for instructive purposes.

'''

__all__ = ['Flash']

from fluids.constants import R
from thermo.equilibrium import EquilibriumState
from thermo.phase_identification import identify_sort_phases
from thermo.utils import has_matplotlib
from fluids.numerics import logspace, linspace, numpy as np
from chemicals.utils import log10, floor
from thermo import phases

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

class Flash(object):
    r'''Base class for performing flash calculations. All Flash objects need
    to inherit from this, and common methods can be added to it.'''

    def __init_subclass__(cls):
        cls.__full_path__ = "%s.%s" %(cls.__module__, cls.__qualname__)

    def flash(self, zs=None, T=None, P=None, VF=None, SF=None, V=None, H=None,
              S=None, G=None, U=None, A=None, solution=None, hot_start=None,
              retry=False, dest=None):
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

        Returns
        -------
        results : :obj:`EquilibriumState <thermo.equilibrium.EquilibriumState>`
            Equilibrium object containing the state of the phases after the
            flash calculation [-]

        Notes
        -----

        Examples
        --------
        '''
        if zs is None:
            if self.N == 1:
                zs = [1.0]
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
            flash_specs['T'] = T
            if T < self.T_MIN_FLASH:
                raise ValueError("Specified temperature (%s K) is below the minimum temeprature (%s K) "
                                 "supported by the provided phases" %(T, self.T_MIN_FLASH))
            # if T <= 0.0:
            #     raise ValueError("Specified temperature (%s K) is unphysical" %(T,))
        if P_spec:
            flash_specs['P'] = P
            if P <= 0.0:
                raise ValueError("Specified pressure (%s Pa) is unphysical"%(P,))
        if V_spec:
            flash_specs['V'] = V
            if V <= 0.0:
                raise ValueError("Specified molar volume (%s m^3/mol) is unphysical"%(V,))
        if H_spec:
            flash_specs['H'] = H
        if S_spec:
            flash_specs['S'] = S
        if U_spec:
            flash_specs['U'] = U
        if G_spec:
            flash_specs['G'] = G
        if A_spec:
            flash_specs['A'] = A

        if VF_spec:
            flash_specs['VF'] = VF
            if VF < 0.0 or VF > 1.0:
                raise ValueError("Specified vapor fraction (%s) is not between 0 and 1"%(VF,))
            elif not self.supports_VF_flash:
                raise ValueError("Cannot flash with a vapor fraction spec without at least one gas and liquid phase defined")
        if SF_spec:
            flash_specs['SF'] = SF
            if SF < 0.0 or SF > 1.0:
                raise ValueError("Specified solid fraction (%s) is not between 0 and 1"%(VF,))
            elif not self.supports_SF_flash:
                raise ValueError("Cannot flash with a solid fraction spec without at least one gas and liquid phase defined, as well as a solid phase")

        if ((T_spec and (P_spec or V_spec)) or (P_spec and V_spec)):
            g, ls, ss, betas, flash_convergence = self.flash_TPV(T=T, P=P, V=V, zs=zs, solution=solution, hot_start=hot_start)
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
#
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
#
            return dest(Tsub, P, zs, gas=g, liquids=liquids, solids=[s],
                                    betas=[1-SF, SF], flash_specs=flash_specs,
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    settings=settings, flasher=self)
        elif VF_spec and any([H_spec, S_spec, U_spec, G_spec, A_spec]):
            spec_var, spec_val = [(k, v) for k, v in flash_specs.items() if k not in ('VF', 'zs')][0]
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
                g, ls, ss, betas, flash_convergence = self.flash_TPV_HSGUA(fixed_var_val, spec_val, fixed_var, spec, iter_var, zs=zs, solution=solution, hot_start=hot_start)
            except Exception as e:
                if retry:
                    print('retrying HSGUA flash')
                    g, ls, ss, betas, flash_convergence = self.flash_TPV_HSGUA(fixed_var_val, spec_val, fixed_var, spec, iter_var_backup, zs=zs, solution=solution, hot_start=hot_start)
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
        scalar_props = isinstance(props, str)

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
                    print('Failed trying to flash %s, with exception %s.'%(flash_specs, e))

                if store:
                    row_flashes.append(state)
                if do_props:
                    if scalar_props:
                        state_props = state.value(props)if state is not None else None
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

        V_set = set([check1, check0])
        TV_iter = V_set == set(['T', 'V'])
        PV_iter = V_set == set(['P', 'V'])
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
                    kwargs['V'] = getattr(state, 'V_iter')(force=False)
                kwargs['retry'] = retry
                kwargs['solution'] = lambda new: abs(new.value(nearest_check_prop) - state.value(nearest_check_prop))
                try:
                    new = self.flash(**kwargs)
                    if PV_iter:
                        # Do a check here on tolerance
                        err = abs((new.value(nearest_check_prop) - state.value(nearest_check_prop))/state.value(nearest_check_prop))
                        if err > 1e-8:
                            kwargs['V'] = getattr(state, 'V_iter')(force=True)
                            new = self.flash(**kwargs)
                except Exception as e:
                    # Was it a precision issue? Some flashes can be brutal
                    if 'V' in kwargs:
                        try:
                             kwargs['V'] = getattr(state, 'V_iter')(True)
                             new = self.flash(**kwargs)
                        except Exception as e2:
                            new = None
                            if verbose:
                                print('Failed trying to flash %s, from original point %s, with exception %s.'%(kwargs, flash_specs, e))
                    else:
                        new = None
                        if verbose:
                            print('Failed trying to flash %s, from original point %s, with exception %s.' % (kwargs, flash_specs, e))
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
            from matplotlib import ticker, cm
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

            ax.set_title('%s %s validation of %s; Reference flash %s %s; max err %.1e' %(check0, check1, prop0, spec0, spec1, max_err))


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
            from matplotlib import ticker, cm
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
            im = ax.pcolormesh(X, Y, z, cmap=color_map, norm=norm) #
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
                       'VLS': 7, 'VLLS': 8, 'VLLSS': 9, 'LLL': 10, 'F': 0}

        used_regions = set([])
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
        cmap = colors.ListedColormap(['y','b','r', 'g', 'c', 'm', 'k', 'w', 'w', 'w', 'w'][0:len(used_values)])

        vmax = len(used_values) - 1

#        ax.scatter(Ts,Ps, s=dat)
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
        plt.title('PT system flashes, zs=%s' %zs)
        if show:
            plt.show()
        else:
            return fig


    def plot_TP(self, zs, Tmin=None, Tmax=None, pts=50, branches=None,
                ignore_errors=True, values=False, show=True, hot=True): # pragma: no cover
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
            Whether to fail on a calculation failure or to ignore the bad
            point, [-]
        values : bool, optional
            If True, the calculated values will be returned instead of
            plotted, [-]
        show : bool, optional
            If False, the plot will be returned instead of shown, [-]
        hot : bool, optional
            Whether to restart the next flash from the previous flash or not
            (intended to speed the call when True), [-]

        Returns
        -------
        Ts : list[float]
            Temperatures, [K]
        P_dews, P_bubbles, branch_Ps

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
        plt.title('PT system curve, zs=%s' %zs)
        if branch:
            for VF, Ps in zip(branches, branch_Ps):
                plt.semilogy(Ts, Ps, label='TP curve for VF=%s'%VF)
        plt.legend(loc='best')
        if show:
            plt.show()
        if values:
            return Ts, P_dews, P_bubbles, branch_Ps
        else:
            return fig


    def plot_PT(self, zs, Pmin=None, Pmax=None, pts=50, branches=[],
                ignore_errors=True, values=False, hot=True): # pragma: no cover
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
        plt.plot(Ps, T_dews, label='PT dew point curve')
        plt.plot(Ps, T_bubbles, label='PT bubble point curve')
        plt.xlabel('System pressure, Pa')
        plt.ylabel('System temperature, K')
        plt.title('PT system curve, zs=%s' %zs)
        if branch:
            for VF, Ts in zip(branches, branch_Ts):
                plt.plot(Ps, Ts, label='PT curve for VF=%s'%VF)
        plt.legend(loc='best')
        plt.show()

    def plot_ternary(self, T, scale=10): # pragma: no cover
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        try:
            import ternary
        except:
            raise Exception('Optional dependency ternary is required for ternary plotting')
        if self.N != 3:
            raise Exception('Ternary plotting requires a mixture of exactly three components')

        P_values = []

        def P_dew_at_T_zs(zs):
            res = self.flash(T=T, zs=zs, VF=0)
            P_values.append(res.P)
            return res.P

        def P_bubble_at_T_zs(zs):
            res = self.flash(T=T, zs=zs, VF=1)
            return res.P

        import matplotlib.pyplot as plt
        import matplotlib
        axes_colors = {'b': 'g', 'l': 'r', 'r':'b'}
        ticks = [round(i / float(10), 1) for i in range(10+1)]

        fig, ax = plt.subplots(1, 3, gridspec_kw = {'width_ratios':[4, 4, 1]})
        ax[0].axis("off") ; ax[1].axis("off")  ; ax[2].axis("off")

        for axis, f, i in zip(ax[0:2], [P_dew_at_T_zs, P_bubble_at_T_zs], [0, 1]):
            figure, tax = ternary.figure(ax=axis, scale=scale)
            figure.set_size_inches(12, 4)
            if not i:
                tax.heatmapf(f, boundary=True, colorbar=False, vmin=0)
            else:
                tax.heatmapf(f, boundary=True, colorbar=False, vmin=0, vmax=max(P_values))

            tax.boundary(linewidth=2.0)
            tax.left_axis_label("mole fraction $x_2$", offset=0.16, color=axes_colors['l'])
            tax.right_axis_label("mole fraction $x_1$", offset=0.16, color=axes_colors['r'])
            tax.bottom_axis_label("mole fraction $x_3$", offset=-0.06, color=axes_colors['b'])

            tax.ticks(ticks=ticks, axis='rlb', linewidth=1, clockwise=True,
                      axes_colors=axes_colors, offset=0.03)

            tax.gridlines(multiple=scale/10., linewidth=2,
                          horizontal_kwargs={'color':axes_colors['b']},
                          left_kwargs={'color':axes_colors['l']},
                          right_kwargs={'color':axes_colors['r']},
                          alpha=0.5)

        norm = plt.Normalize(vmin=0, vmax=max(P_values))
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
        sm._A = []
        cb = plt.colorbar(sm, ax=ax[2])
        cb.locator = matplotlib.ticker.LinearLocator(numticks=7)
        cb.formatter = matplotlib.ticker.ScalarFormatter()
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        plt.tight_layout()
        fig.suptitle("Bubble pressure vs composition (left) and dew pressure vs composition (right) at %s K, in Pa" %T, fontsize=14);
        fig.subplots_adjust(top=0.85)
        plt.show()

