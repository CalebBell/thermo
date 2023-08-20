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
'''

__all__ = ['FlashVLN']

from chemicals.exceptions import PhaseCountReducedError, TrivialSolutionError
from chemicals.rachford_rice import flash_inner_loop
from fluids.numerics import OscillationError, UnconvergedError

from thermo.bulk import default_settings
from thermo.flash.flash_utils import deduplicate_stab_results, empty_flash_conv, one_in_list, sequential_substitution_NP
from thermo.flash.flash_vl import FlashVL
from thermo.phase_identification import identify_sort_phases
from thermo.property_package import StabilityTester

CAS_H2O = '7732-18-5'

class FlashVLN(FlashVL):
    r'''Class for performing flash calculations on multiphase vapor-liquid
    systems. This rigorous class does not make any assumptions and will search
    for up to the maximum amount of liquid phases specified by the user. Vapor
    and each liquid phase do not need to use a consistent thermodynamic model.

    The minimum information that is needed in addition to the :obj:`Phase`
    objects is:

    * MWs
    * Vapor pressure curve
    * Functioning enthalpy models for each phase

    Parameters
    ----------
    constants : :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>` object
        Package of chemical constants; these are used as boundaries at times,
        initial guesses other times, and in all cases these properties are
        accessible as attributes of the resulting
        :obj:`EquilibriumState <thermo.equilibrium.EquilibriumState>` object, [-]
    correlations : :obj:`PropertyCorrelationsPackage <thermo.chemical_package.PropertyCorrelationsPackage>`
        Package of chemical T-dependent properties; these are used as boundaries at times,
        for initial guesses other times, and in all cases these properties are
        accessible as attributes of the resulting
        :obj:`EquilibriumState <thermo.equilibrium.EquilibriumState>` object, [-]
    gas : :obj:`Phase <thermo.phases.Phase>` object
        A single phase which can represent the gas phase, [-]
    liquids : list[:obj:`Phase <thermo.phases.Phase>`]
        A list of phase objects that can represent the liquid phases;
        if working with a VLL system with a consistent model, specify the same
        liquid phase twice; the length of this list is the maximum number of
        liquid phases that will be searched for, [-]
    solids : list[:obj:`Phase <thermo.phases.Phase>`]
        Not used, [-]
    settings : :obj:`BulkSettings <thermo.bulk.BulkSettings>` object
        Object containing settings for calculating bulk and transport
        properties, [-]

    Attributes
    ----------
    SS_NP_MAXITER : int
        Maximum number of sequential substitution iterations to try when
        converging a three or more phase solution, [-]
    SS_NP_TOL : float
        Convergence tolerance in sequential substitution for a three or more
        phase solution [-]
    SS_NP_TRIVIAL_TOL : float
        Tolerance at which to quick a three-phase flash because it is
        converging to the trivial solution, [-]
    SS_STAB_AQUEOUS_CHECK : bool
        If True, the first three-phase stability check will be on water (if
        it is present) as it forms a three-phase solution more than any
        other component, [-]
    DOUBLE_CHECK_2P : bool
        This parameter should be set to True if any issues in the solution are
        noticed. It can slow down two-phase solution. It ensures that all
        potential vapor-liquid and liquid-liquid phase pairs are searched for
        stability, instead of testing first for a vapor-liquid solution and
        then moving on to a three phase flash if an instability is detected,
        [-]

    Notes
    -----
    The algorithms in this object are mostly from [1]_, [2]_ and [3]_.
    Sequential substitution without acceleration is used by default to converge
    multiphase systems.

    Additional information that can be provided in the
    :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>`
    object and :obj:`PropertyCorrelationsPackage <thermo.chemical_package.PropertyCorrelationsPackage>`
    object that may help convergence is:

    * `Tc`, `Pc`, `omega`, `Tb`, and `atoms`
    * Gas heat capacity correlations
    * Liquid molar volume correlations
    * Heat of vaporization correlations

    Examples
    --------
    A three-phase flash of butanol, water, and ethanol with the SRK EOS without
    BIPs:

    >>> from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, SRKMIX, FlashVLN, PropertyCorrelationsPackage, HeatCapacityGas
    >>> constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Pcs=[4414000.0, 22048320.0, 6137000.0], omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844], CASs=['71-36-3', '7732-18-5', '64-17-5'])
    >>> properties = PropertyCorrelationsPackage(constants=constants,
    ...                                     HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
    ...                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
    ...                                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),], )
    >>> eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    >>> gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    >>> liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    >>> flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)
    >>> res = flashN.flash(T=361, P=1e5, zs=[.25, 0.7, .05])
    >>> res.phase_count
    3


    References
    ----------
    .. [1] Michelsen, Michael L., and Jørgen M. Mollerup. Thermodynamic Models:
       Fundamentals & Computational Aspects. Tie-Line Publications, 2007.
    .. [2] Poling, Bruce E., John M. Prausnitz, and John P. O`Connell. The
       Properties of Gases and Liquids. 5th edition. New York: McGraw-Hill
       Professional, 2000.
    .. [3] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen Rarey.
       Chemical Thermodynamics for Process Simulation. John Wiley & Sons, 2019.
    '''

    SS_NP_MAXITER = FlashVL.PT_SS_MAXITER
    SS_NP_TRIVIAL_TOL = 5e-5
    SS_NP_TOL = 1e-15
    SS_STAB_AQUEOUS_CHECK = True

    DOUBLE_CHECK_2P = False

    SS_NP_STAB_HIGHEST_COMP_DIFF = False
    SS_NP_STAB_COMP_DIFF_MIN = None

    K_COMPOSITION_INDEPENDENT_HACK = True
    skip_solids = True

    supports_VF_flash = True
    supports_SF_flash = False

    def __init__(self, constants, correlations, liquids, gas, solids=None, settings=default_settings):
        self.constants = constants
        self.correlations = correlations
        self.liquids = liquids
        self.gas = gas
        self.settings = settings
        if solids:
            raise ValueError("Solids are not supported in this model")
        self._finish_initialization()

    def _finish_initialization(self):
        constants, correlations, settings = self.constants, self.correlations, self.settings
        liquids, gas = self.liquids, self.gas

        if gas is None:
            raise ValueError("Gas phase is required in this model")
        self.liquid0 = liquids[0] if liquids else None
        self.liquid_count = self.max_liquids = len(liquids)
        self.max_phases = 1 + self.max_liquids if gas is not None else self.max_liquids
        self.phases = [gas] + liquids if gas is not None else liquids


        self.N = constants.N

        self.K_composition_independent = all(i.composition_independent for i in self.phases)
        self.ideal_gas_basis = all(i.ideal_gas_basis for i in self.phases)


        self.aqueous_check = (self.SS_STAB_AQUEOUS_CHECK and '7732-18-5' in constants.CASs)
        self.stab = StabilityTester(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas,
                                    aqueous_check=self.aqueous_check, CASs=constants.CASs)
        try:
            self._water_index = constants.CASs.index(CAS_H2O)
        except ValueError:
            self._water_index = None

        self._finish_initialization_base()

#        self.flash_pure = FlashPureVLS(constants=constants, correlations=correlations,
#                                       gas=gas, liquids=unique_liquids, solids=[],
#                                       settings=settings)

    def flash_TVF(self, T, VF, zs, solution=None, hot_start=None, liquid_idx=None):
        if self.unique_liquid_count == 1:
            return self.flash_TVF_2P(T, VF, zs, self.liquids[0], self.gas, solution=solution, hot_start=hot_start)
        elif liquid_idx is not None:
            return self.flash_TVF_2P(T, VF, zs, self.liquids[liquid_idx], self.gas, solution=solution, hot_start=hot_start)
        else:
            sln_G_min, G_min = None, 1e100
            for l in self.unique_liquids:
                try:
                    sln = self.flash_TVF_2P(T, VF, zs, l, self.gas, solution=solution, hot_start=hot_start)
                    sln_G = (sln[1].G()*(1.0 - VF) + sln[2].G()*VF)
                    if sln_G < G_min:
                        sln_G_min, G_min = sln, sln_G
                except:
                    pass
            return sln_G_min


    def flash_PVF(self, P, VF, zs, solution=None, hot_start=None, liquid_idx=None):
        if self.unique_liquid_count == 1:
            sln_2P = self.flash_PVF_2P(P, VF, zs, self.liquids[0], self.gas, solution=solution, hot_start=hot_start)
        elif liquid_idx is not None:
            sln_2P = self.flash_PVF_2P(P, VF, zs, self.liquids[liquid_idx], self.gas, solution=solution, hot_start=hot_start)
        else:
            sln_G_min, G_min = None, 1e100
            for l in self.unique_liquids:
                try:
                    sln = self.flash_PVF_2P(P, VF, zs, l, self.gas, solution=solution, hot_start=hot_start)
                    sln_G = (sln[1].G()*(1.0 - VF) + sln[2].G()*VF)
                    if sln_G < G_min:
                        sln_G_min, G_min = sln, sln_G
                except:
                    pass
            sln_2P = sln_G_min
        return sln_2P


    def phases_at(self, T, P, zs, V=None):
        # Avoid doing excess work here
        # Goal: bring each phase to T, P, zs; using whatever duplicate information
        # possible
        # returns gas, [liquids], phases
        if V is None:
            gas = None
            gas_to_unique_liquid = self.gas_to_unique_liquid
            liquids = [None]*self.max_liquids
            for i, liq in enumerate(self.unique_liquids):
                l = liq.to(T=T, P=P, zs=zs)
                for j, idx in enumerate(self.liquids_to_unique_liquids):
                    if idx == i:
                        liquids[j] = l
                if i == gas_to_unique_liquid:
                    try:
                        gas = self.gas.to_TP_zs(T, P, zs, other_eos=l.eos_mix)
                    except:
                        gas = self.gas.to_TP_zs(T, P, zs)

            if gas is None:
                gas = self.gas.to(T=T, P=P, zs=zs)
            return gas, liquids, [gas] + liquids
        else:
            # TODO: handle unique liquids in this function
            if T is not None:
                gas = self.gas.to(T=T, V=V, zs=zs)
                liquids = [l.to(T=T, V=V, zs=zs) for l in self.liquids]
            elif P is not None:
                gas = self.gas.to(P=P, V=V, zs=zs)
                liquids = [l.to(P=P, V=V, zs=zs) for l in self.liquids]
            else:
                raise ValueError("Two of three specs are required")
            return gas, liquids, [gas] + liquids

    def sequential_substitution_NP(self, T, P, zs, comps, betas, phases, maxiter, tol, trivial_solution_tol):

        return sequential_substitution_NP(T, P, zs, comps, betas, phases,
                    maxiter=maxiter, tol=tol, trivial_solution_tol=trivial_solution_tol)


    def flash_TPV_hot(self, T, P, V, zs, hot_start, solution=None):
        if hot_start.phase_count == 2:
            xs = hot_start.phases[0].zs
            ys = hot_start.phases[1].zs
            double_check_sln = self.flash_2P(T, P, zs, xs, ys, hot_start.phases[0],
                                                         hot_start.phases[1],
                                                         None, None, V_over_F_guess=hot_start.betas[1], LL=True)
            failed = not double_check_sln[0] and not double_check_sln[1]
            if not failed:
                return double_check_sln
        elif hot_start.phase_count > 2:
            phases = hot_start.phases
            comps = [i.zs for i in hot_start.phases]
            betas = hot_start.betas
            slnN = self.sequential_substitution_NP(
                T, P, zs, comps, betas, phases,
                maxiter=self.SS_NP_MAXITER, tol=self.SS_NP_TOL,
                trivial_solution_tol=self.SS_NP_TRIVIAL_TOL
            )
            return None, slnN[2], [], slnN[0], {'iterations': slnN[3], 'err': slnN[4],
                                                'stab_guess_name': None}
        if failed:
            return self.flash_TPV(T=T, P=P, V=V, zs=zs, solution=solution)


    def flash_TP_K_composition_idependent(self, T, P, zs):
        if self.max_phases == 1:
            phase = self.phases[0].to(T=T, P=P, zs=zs)
            return None, [phase], [], [1.0], {'iterations': 0, 'err': 0}


        Ks = liquid_phis = self.liquid0.phis_at(T, P, zs)
        try:
            VF, xs, ys = flash_inner_loop(zs, Ks)
        except PhaseCountReducedError:
            K_low, K_high = False, False
            for zi, Ki in zip(zs, Ks):
                if zi != 0.0:
                    if Ki > 1.0:
                        K_high = True
                    else:
                        K_low = True
            if K_low and not K_high:
                VF = -0.5
            elif K_high and not K_low:
                VF = 1.5
            else:
                raise ValueError("Error")

        if VF > 1.0:
            return None, [self.gas.to(T=T, P=P, zs=zs)], [], one_in_list, empty_flash_conv
        elif VF < 0.0:
            return None, [self.liquid0.to(T=T, P=P, zs=zs)], [], one_in_list, empty_flash_conv
        else:
            gas = self.gas.to(T=T, P=P, zs=ys)
            liquid = self.liquid0.to(T=T, P=P, zs=xs)
            return gas, [liquid], [], [VF, 1.0 - VF], empty_flash_conv

    def flash_TPV(self, T, P, V, zs=None, solution=None, hot_start=None):
        if T is None:
            return self.flash_PV(P, V, zs, solution, hot_start)
        if P is None:
            return self.flash_TV(T, V, zs, solution, hot_start)

        if hot_start is not None and hot_start.phase_count > 1 and solution is None:
            # Only allow hot start when there are multiple phases
            try:
                return self.flash_TPV_hot(T, P, V, zs, hot_start, solution=solution)
            except Exception as e:
                # Let anything fail
                pass
        if self.K_composition_independent and self.K_COMPOSITION_INDEPENDENT_HACK and solution is None:
            return self.flash_TP_K_composition_idependent(T, P, zs)

        gas, liquids, phases = self.phases_at(T, P, zs, V=V)
        gas_at_conditions = gas
#        if self.K_composition_independent and self.K_COMPOSITION_INDEPENDENT_HACK:
#            # TODO move into new function?
#            if self.max_phases == 2:
#                gas_phis = gas.phis()
#                liquid_phis = liquids[0].phis()
#                Ks = [liquid_phis[i]/gas_phis[i] for i in range(self.N)]
#                VF, xs, ys = flash_inner_loop(zs, Ks)
#                if VF > 1.0:
#                    return None, [gas], [], one_in_list, empty_flash_conv
#                elif VF < 0.0:
#                    return None, [liquids[0]], [], one_in_list, empty_flash_conv
#                else:
#                    gas = gas.to(T=T, P=P, zs=ys)
#                    liquid = liquids[0].to(T=T, P=P, zs=xs)
#                    return gas, [liquid], [], [VF, 1.0 - VF], empty_flash_conv

        min_phase_1P, G_min_1P = None, 1e100
        ideal_gas_basis = self.ideal_gas_basis
        if ideal_gas_basis:
            for p in phases:
                G = p.G_min_criteria()
                if G < G_min_1P:
                    min_phase_1P, G_min_1P = p, G
        else:
            for p in phases:
                G = p.G()
                if G < G_min_1P:
                    min_phase_1P, G_min_1P = p, G
        one_phase_sln = None, [min_phase_1P], [], one_in_list, empty_flash_conv

        sln_2P, one_phase_min = None, None
        VL_solved, LL_solved = False, False
        phase_evolved = [False]*self.max_phases


        try:
            sln_2P = self.flash_TP_stability_test(T, P, zs, liquids[0], gas, solution=solution, phases_ready=True)
            if len(sln_2P[3]) == 2: # One phase only
                VL_solved = True
                g, l0 = sln_2P[0], sln_2P[1][0]
                found_phases = [g, l0]
                phase_evolved[0] = phase_evolved[1] = True
                found_betas = sln_2P[3]
        except:
            VL_solved = False

        if not VL_solved and self.max_liquids > 1:
            for n_liq, a_liq in enumerate(liquids[1:]):
                # Come up with algorithm to skip
                try:
                    sln_2P = self.flash_TP_stability_test(T, P, zs, liquids[0], a_liq, solution=solution, LL=True)
                    if len(sln_2P[3]) == 2:
                        LL_solved = True
                        g = None
                        l0, l1 = sln_2P[1]
                        found_phases = [l0, l1]
                        found_betas = sln_2P[3]
                        break
                except:
                    pass
        if not LL_solved and not VL_solved:
            found_phases = [min_phase_1P]
            found_betas = [1]

        existing_comps = [i.zs for i in found_phases]
        if ideal_gas_basis:
            G_2P = sum([found_betas[i]*found_phases[i].G_min_criteria() for i in range(len(found_phases))])
        else:
            G_2P = sum([found_betas[i]*found_phases[i].G() for i in range(len(found_phases))])

        if sln_2P is not None and self.DOUBLE_CHECK_2P:
            g_id, ls_id, _, _ = identify_sort_phases(found_phases, found_betas, self.constants,
                                                    self.correlations, settings=self.settings,
                                                    skip_solids=self.skip_solids)
            if g_id is None:
                another_phase, base_phase = gas, liquids[0]
            else:
                another_phase, base_phase = liquids[0], gas

            all_solutions = self.stability_test_Michelsen(T, P, zs, another_phase, base_phase, all_solutions=True) + self.stability_test_Michelsen(T, P, zs, base_phase, another_phase, all_solutions=True)
            all_solutions = deduplicate_stab_results(all_solutions)
            for stab_sln in all_solutions:
                trial_zs, appearing_zs, V_over_F, stab_guess_name, _, _, _, _ = stab_sln
                if V_over_F < 1.000001 and V_over_F > -.000001:
                    try:
                        double_check_sln = self.flash_2P(T, P, zs, trial_zs, appearing_zs, another_phase,
                                                         base_phase, gas, liquids[0], V_over_F_guess=V_over_F, LL=True)
                    except (UnconvergedError, OscillationError, PhaseCountReducedError):
                        continue
                    double_check_betas = double_check_sln[3]
                    if len(double_check_betas) == 2:
                        double_check_phases = double_check_sln[1]
                        if ideal_gas_basis:
                            G_2P_new = sum([double_check_betas[i]*double_check_phases[i].G_min_criteria() for i in range(2)])
                        else:
                            G_2P_new = sum([double_check_betas[i]*double_check_phases[i].G() for i in range(2)])
                        if G_2P_new < G_2P:
                            sln_2P = double_check_sln
                            G_2P = G_2P_new
                            found_phases = double_check_phases
                            existing_comps = [i.zs for i in found_phases]
                            found_betas = double_check_betas


        # Can still be a VLL solution now that a new phase has been added
        if (LL_solved and (self.max_liquids == 2) or (VL_solved and self.max_liquids == 1) or (self.N < 3 and (VL_solved or LL_solved))):
            # Check the Gibbs
            if G_2P < G_min_1P:
                return sln_2P
            else:
                # May be missing possible 3 phase solutions which have lower G
                return one_phase_sln
        if not LL_solved and not VL_solved:
            return one_phase_sln
        if self.N < 3:
            # Gibbs phase rule 5.9: Multiphase Split and Stability Analysis
            # in Thermodynamics and Applications in Hydrocarbon Energy Production by Firoozabadi (2016)
            # Can only have three phases when either T or P are not specified
            return sln_2P

        # Always want the other phase to be type of one not present.
        min_phase = sln_2P[0] if sln_2P[0] is not None else sln_2P[1][0]
        other_phase_flashed = found_phases[0] if found_phases[0] is not min_phase else found_phases[1]
        other_phase = gas_at_conditions if LL_solved else liquids[1]

        SWITCH_EXPECT_LIQ_Z = 0.25
        expect_liquid = (bool(other_phase_flashed.Z() > SWITCH_EXPECT_LIQ_Z or min_phase.Z() > SWITCH_EXPECT_LIQ_Z))
        expect_aqueous = False
        if self.aqueous_check and self.water_index is not None and zs[self.water_index] > 1e-3:
            # Probably a water phase exists
            expect_aqueous = True

        stable, (trial_zs, appearing_zs, V_over_F, stab_guess_name, stab_guess_number, stab_sum_zs_test, stab_lnK_2_tot, _) = self.stability_test_Michelsen(
                T, P, zs, min_phase, other_phase, existing_comps=existing_comps, expect_liquid=expect_liquid,
                expect_aqueous=expect_aqueous, handle_iffy=False, highest_comp_diff=self.SS_NP_STAB_HIGHEST_COMP_DIFF, min_comp_diff=self.SS_NP_STAB_COMP_DIFF_MIN)
        if stable and self.unique_liquid_count > 2:
            for other_phase in liquids[2:]:
                stable, (trial_zs, appearing_zs, V_over_F, stab_guess_name, stab_guess_number, stab_sum_zs_test, stab_lnK_2_tot, _) = self.stability_test_Michelsen(T, P, zs,
                                                                                                            min_phase,
                                                                                                            other_phase, existing_comps=existing_comps)
                if not stable:
                    break
        if stable:
            # Return the two phase solution
            return sln_2P
        else:
            sln3 = None
            flash_phases = found_phases + [other_phase]
            flash_comps = [i.zs for i in found_phases]
            flash_comps.append(appearing_zs)
            flash_betas = list(found_betas)
            flash_betas.append(0.0)
            try_LL_3P_failed = False
            try:
                failed_3P = False

                sln3 = self.sequential_substitution_NP(
                    T, P, zs, flash_comps, flash_betas, flash_phases,
                    maxiter=self.SS_NP_MAXITER, tol=self.SS_NP_TOL,
                    trivial_solution_tol=self.SS_NP_TRIVIAL_TOL
                )
                if ideal_gas_basis:
                    G_3P = sum([sln3[0][i]*sln3[2][i].G_min_criteria() for i in range(3)])
                else:
                    G_3P = sum([sln3[0][i]*sln3[2][i].G() for i in range(3)])
                new_betas = sln3[0]
                good_betas = True
                for b in new_betas:
                    if b < 0.0 or b > 1.0:
                        good_betas = False
                if self.max_phases == 3 and good_betas:
                    if G_2P < G_3P:
                        raise ValueError("Should never happen")
                    return None, sln3[2], [], sln3[0], {'iterations': sln3[3], 'err': sln3[4],
                                                        'stab_guess_name': stab_guess_name, 'G_2P': G_2P}
                if not good_betas or G_3P > G_2P:
                    try_LL_3P_failed = True # used to try to make this False but it just isn't correct
                    failed_3P = True
            except:
                try_LL_3P_failed = True
                failed_3P = True
            if VL_solved and failed_3P:
                if try_LL_3P_failed:
                    try:
                        V_over_F, xs, ys, l, g, iteration, err = self.sequential_substitution_2P(
                            T=T, P=P,
                            zs=zs, xs_guess=trial_zs,
                            ys_guess=appearing_zs,
                            liquid_phase=liquids[0],
                            gas_phase=liquids[1],
                            maxiter=self.PT_SS_POLISH_MAXITER,
                            tol=self.PT_SS_POLISH_TOL,
                            V_over_F_guess=V_over_F
                        )
                        if ideal_gas_basis:
                            new_G_2P = V_over_F*g.G_min_criteria() + (1.0 - V_over_F)*l.G_min_criteria()
                        else:
                            new_G_2P = V_over_F*g.G() + (1.0 - V_over_F)*l.G()
                        if new_G_2P < G_2P:
                            return None, [l, g], [], [1.0 - V_over_F, V_over_F], {'iterations': iteration, 'err': err,
                                         'stab_guess_name': stab_guess_name, 'G_2P': G_2P}
                            a = 1
                        else:
                            return sln_2P
                    except (TrivialSolutionError, OscillationError, PhaseCountReducedError):
                        return sln_2P
                else:
                    return sln_2P
            elif LL_solved and failed_3P:
                # We are unstable but we couldn't converge the two phase point
                # not very good, but we also don't have anything else to try, except maybe
                # converging other guesses from the stability test
                # TODO
                return sln_2P



        slnN = sln3

        if self.N == 3:
            # Cannot have a four phase system with three components (and so on)
            return None, sln3[2], [], sln3[0], {'iterations': sln3[3], 'err': sln3[4],
                                                'stab_guess_name': stab_guess_name, 'G_2P': G_2P}

        # We are here after solving three phases
        liquid_idx = 2
        while len(slnN[0]) < self.max_phases and liquid_idx < self.max_liquids:
            min_phase = slnN[2][0]
            existing_comps = slnN[1]
            # hardcoded for now - need to track
            other_phase = liquids[liquid_idx]
            stable, (trial_zs, appearing_zs, V_over_F, stab_guess_name, stab_guess_number, stab_sum_zs_test, stab_lnK_2_tot, _) = self.stability_test_Michelsen(T, P, zs, min_phase, other_phase, existing_comps=existing_comps)
        # if stable and self.unique_liquid_count > 3:
        #     for other_phase in liquids[3:]:
        #         stable, (trial_zs, appearing_zs, V_over_F, stab_guess_name, stab_guess_number, stab_sum_zs_test, stab_lnK_2_tot, _) = self.stability_test_Michelsen(T, P, zs,
        #                                                                                                     min_phase,
        #                                                                                                     other_phase, existing_comps=existing_comps)
        #         if not stable:
        #             break

            if not stable:

                flash_phases = slnN[2] + [other_phase]
                flash_comps = list(slnN[1])
                flash_comps.append(appearing_zs)
                flash_betas = list(slnN[0])
                flash_betas.append(0.0)
                try:
                    slnN = self.sequential_substitution_NP(
                        T, P, zs, flash_comps, flash_betas, flash_phases,
                        maxiter=1000, tol=1E-13,
                               trivial_solution_tol=1e-5
                    )
                    if self.max_phases == len(slnN[0]):
                        return None, slnN[2], [], slnN[0], {'iterations': slnN[3], 'err': slnN[4],
                                                                       'stab_guess_name': stab_guess_name, 'G_2P': G_2P}
                except:
                    pass

            liquid_idx += 1

        return None, slnN[2], [], slnN[0], {'iterations': slnN[3], 'err': slnN[4],
                                            'stab_guess_name': stab_guess_name, 'G_2P': G_2P}

