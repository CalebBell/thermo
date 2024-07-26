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

from chemicals.exceptions import TrivialSolutionError
from fluids.numerics import UnconvergedError, isinf, secant, trunc_log

from thermo import phases
from thermo.bulk import default_settings
from thermo.coolprop import CPiP_min
from thermo.flash.flash_base import Flash
from thermo.flash.flash_utils import (
    IDEAL_PSAT,
    IDEAL_WILSON,
    PT_SS,
    PT_SS_GDEM3,
    PT_SS_MEHRA,
    SHAW_ELEMENTAL,
    TB_TC_GUESS,
    WILSON_GUESS,
    PT_NEWTON_lNKVF,
    SS_VF_simultaneous,
    TP_solve_VF_guesses,
    TPV_solve_HSGUA_guesses_VL,
    dew_bubble_bounded_naive,
    dew_bubble_Michelsen_Mollerup,
    dew_bubble_newton_zs,
    nonlin_2P_newton,
    nonlin_spec_NP,
    sequential_substitution_2P,
    sequential_substitution_2P_functional,
    sequential_substitution_GDEM3_2P,
    sequential_substitution_Mehra_2P,
    solve_P_VF_IG_K_composition_independent,
    solve_PTV_HSGUA_1P,
    solve_T_VF_IG_K_composition_independent,
    stability_iteration_Michelsen,
)
from thermo.property_package import StabilityTester

__all__ = ['FlashVL']

class FlashVL(Flash):
    r'''Class for performing flash calculations on one and
    two phase vapor and liquid multicomponent systems. Use :obj:`FlashVLN` for
    systems which can have multiple liquid phases.

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
    liquid : :obj:`Phase <thermo.phases.Phase>`
        A single phase which can represent the liquid phase, [-]
    settings : :obj:`BulkSettings <thermo.bulk.BulkSettings>` object
        Object containing settings for calculating bulk and transport
        properties, [-]

    Attributes
    ----------
    PT_SS_MAXITER : int
        Maximum number of sequential substitution iterations to try when
        converging a two-phase solution, [-]
    PT_SS_TOL : float
        Convergence tolerance in sequential substitution [-]
    PT_SS_POLISH : bool
        When set to True, flashes which are very near a vapor fraction of 0 or
        1 are converged to a higher tolerance to ensure the solution is
        correct; without this, a flash might converge to a vapor fraction of
        -1e-7 and be called single phase, but with this the correct solution
        may be found to be 1e-8 and will be correctly returned as two phase.[-]
    PT_SS_POLISH_VF : float
        What tolerance to a vapor fraction of 0 or 1; this is an absolute
        vapor fraction value, [-]
    PT_SS_POLISH_MAXITER : int
        Maximum number of sequential substitution iterations to try when
        converging a two-phase solution that has been detected to be very
        sensitive, with a vapor fraction near 0 or 1 [-]
    PT_SS_POLISH_TOL : float
        Convergence tolerance in sequential substitution when
        converging a two-phase solution that has been detected to be very
        sensitive, with a vapor fraction near 0 or 1 [-]
    PT_STABILITY_MAXITER : int
        Maximum number of iterations to try when converging a stability test,
        [-]
    PT_STABILITY_XTOL : float
        Convergence tolerance in the stability test [-]
    DEW_BUBBLE_VF_K_COMPOSITION_INDEPENDENT_XTOL : float
        Convergence tolerance in Newton solver for bubble, dew, and vapor
        fraction spec flashes when both the liquid and gas model's K values do
        not dependent on composition, [-]
    DEW_BUBBLE_QUASI_NEWTON_XTOL : float
        Convergence tolerance in quasi-Newton bubble and dew point flashes, [-]
    DEW_BUBBLE_QUASI_NEWTON_MAXITER : int
        Maximum number of iterations to use in quasi-Newton bubble and dew
        point flashes, [-]
    DEW_BUBBLE_NEWTON_XTOL : float
        Convergence tolerance in Newton bubble and dew point flashes, [-]
    DEW_BUBBLE_NEWTON_MAXITER : int
        Maximum number of iterations to use in Newton bubble and dew
        point flashes, [-]
    TPV_HSGUA_BISECT_XTOL : float
        Tolerance in the iteration variable when converging a flash with one
        (`T`, `P`, `V`) spec and one (`H`, `S`, `G`, `U`, `A`) spec using a
        bisection-type solver, [-]
    TPV_HSGUA_BISECT_YTOL : float
        Absolute tolerance in the (`H`, `S`, `G`, `U`, `A`) spec when
        converging a flash with one (`T`, `P`, `V`) spec and one (`H`, `S`,
        `G`, `U`, `A`) spec using a bisection-type solver, [-]
    TPV_HSGUA_BISECT_YTOL_ONLY : bool
        When True, the `TPV_HSGUA_BISECT_XTOL` setting is ignored and the flash
        is considered converged once `TPV_HSGUA_BISECT_YTOL` is satisfied, [-]
    TPV_HSGUA_NEWTON_XTOL : float
        Tolerance in the iteration variable when converging a flash with one
        (`T`, `P`, `V`) spec and one (`H`, `S`, `G`, `U`, `A`) spec using a
        full newton solver, [-]
    TPV_HSGUA_NEWTON_MAXITER : float
        Maximum number of iterations when
        converging a flash with one (`T`, `P`, `V`) spec and one (`H`, `S`,
        `G`, `U`, `A`) spec using full newton solver, [-]
    TPV_HSGUA_SECANT_MAXITER : float
        Maximum number of iterations when
        converging a flash with one (`T`, `P`, `V`) spec and one (`H`, `S`,
        `G`, `U`, `A`) spec using a secant solver, [-]
    HSGUA_NEWTON_ANALYTICAL_JAC : bool
        Whether or not to calculate the full newton jacobian analytically or
        numerically; this would need to be set to False if the phase objects
        used in the flash do not have complete analytical derivatives
        implemented, [-]


    Notes
    -----
    The algorithms in this object are mostly from [1]_, [2]_ and [3]_.
    Sequential substitution without acceleration is used by default to converge
    two-phase systems.

    Quasi-newton methods are used by default to converge bubble and dew point
    calculations.

    Flashes with one (`T`, `P`, `V`) spec and one (`H`, `S`, `G`, `U`, `A`)
    spec are solved by a 1D search over PT flashes.

    Additional information that can be provided in the
    :obj:`ChemicalConstantsPackage <thermo.chemical_package.ChemicalConstantsPackage>`
    object and :obj:`PropertyCorrelationsPackage <thermo.chemical_package.PropertyCorrelationsPackage>`
    object that may help convergence is:

    * `Tc`, `Pc`, `omega`, `Tb`, and `atoms`
    * Gas heat capacity correlations
    * Liquid molar volume correlations
    * Heat of vaporization correlations

    .. warning::
        If this flasher is used on systems that can form two or more liquid
        phases, and the flash specs are in that region, there is no guarantee
        which solution is returned. Sometimes it is almost random, jumping
        back and forth and providing nasty discontinuities.

    Examples
    --------
    For the system methane-ethane-nitrogen with a composition
    [0.965, 0.018, 0.017], calculate the vapor fraction of the system and
    equilibrium phase compositions at 110 K and 1 bar. Use the Peng-Robinson
    equation of state and the chemsep sample interaction parameter database.

    >>> from thermo import ChemicalConstantsPackage, CEOSGas, CEOSLiquid, PRMIX, FlashVL
    >>> from thermo.interaction_parameters import IPDB
    >>> constants, properties = ChemicalConstantsPackage.from_IDs(['methane', 'ethane', 'nitrogen'])
    >>> kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')
    >>> kijs
    [[0.0, -0.0059, 0.0289], [-0.0059, 0.0, 0.0533], [0.0289, 0.0533, 0.0]]
    >>> eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    >>> gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    >>> liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    >>> flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
    >>> zs = [0.965, 0.018, 0.017]
    >>> PT = flasher.flash(T=110.0, P=1e5, zs=zs)
    >>> PT.VF, PT.gas.zs, PT.liquid0.zs
    (0.0890, [0.8688, 2.5765e-05, 0.13115], [0.9744, 0.01975, 0.00584])

    A few more flashes with the same system to showcase the functionality
    of the :obj:`flash <Flash.flash>` interface:

    >>> flasher.flash(P=1e5, VF=1, zs=zs).T
    133.8
    >>> flasher.flash(T=133, VF=0, zs=zs).P
    515029.6
    >>> flasher.flash(P=PT.P, H=PT.H(), zs=zs).T
    110.0
    >>> flasher.flash(P=PT.P, S=PT.S(), zs=zs).T
    110.0
    >>> flasher.flash(T=PT.T, H=PT.H(), zs=zs).T
    110.0
    >>> flasher.flash(T=PT.T, S=PT.S(), zs=zs).T
    110.0


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

    PT_SS_MAXITER = 5000
    PT_SS_TOL = 1e-13

    PT_TRIVIAL_SOLUTION_TOL = 1e-5

    # Settings for near-boundary conditions
    PT_SS_POLISH_TOL = 1e-25
    PT_SS_POLISH = True
    PT_SS_POLISH_VF = 1e-6 # 5e-8
    PT_SS_POLISH_MAXITER = 1000

    SS_2P_STAB_HIGHEST_COMP_DIFF = False
    SS_2P_STAB_COMP_DIFF_MIN = None

    PT_methods = [
        PT_SS,
        PT_SS_GDEM3,
        PT_SS_MEHRA,
        PT_NEWTON_lNKVF,
    ]
    PT_algorithms = [
        sequential_substitution_2P,
        sequential_substitution_GDEM3_2P,
        sequential_substitution_Mehra_2P,
        nonlin_2P_newton
    ]

    PT_STABILITY_MAXITER = 500 # 30 good professional default; 500 used in source DTU
    PT_STABILITY_XTOL = 5E-9 # 1e-12 was too strict; 1e-10 used in source DTU; 1e-9 set for some points near critical where convergence stopped; even some more stopped at higher Ts

    SS_ACCELERATION = False
    SS_acceleration_method = None

    VF_guess_methods = [
        WILSON_GUESS,
        IDEAL_PSAT,
        TB_TC_GUESS
    ]

    dew_bubble_flash_algos = [
        dew_bubble_Michelsen_Mollerup,
        dew_bubble_newton_zs,
        dew_bubble_bounded_naive,
        SS_VF_simultaneous,
    ]
    dew_T_flash_algos = bubble_T_flash_algos = dew_bubble_flash_algos
    dew_P_flash_algos = bubble_P_flash_algos = dew_bubble_flash_algos

    VF_flash_algos = [dew_bubble_bounded_naive, SS_VF_simultaneous]

    DEW_BUBBLE_VF_K_COMPOSITION_INDEPENDENT_XTOL = 1e-14

    DEW_BUBBLE_QUASI_NEWTON_XTOL = 1e-8
    DEW_BUBBLE_NEWTON_XTOL = 1e-5
    DEW_BUBBLE_QUASI_NEWTON_MAXITER = 200
    DEW_BUBBLE_NEWTON_MAXITER = 200

    TPV_HSGUA_BISECT_XTOL = 1e-9
    TPV_HSGUA_BISECT_YTOL = 1e-6
    TPV_HSGUA_BISECT_YTOL_ONLY = True

    TPV_HSGUA_NEWTON_XTOL = 1e-9
    TPV_HSGUA_NEWTON_MAXITER = 1000
    TPV_HSGUA_NEWTON_SOLVER = 'hybr'
    HSGUA_NEWTON_ANALYTICAL_JAC = True
    TPV_HSGUA_SECANT_MAXITER = 1000

    solids = None
    skip_solids = True
    K_composition_independent = False

    max_liquids = 1
    max_phases = 2

    supports_VF_flash = True
    supports_SF_flash = False

    ceos_gas_liquid_compatible = False

    def __init__(self, constants, correlations, gas, liquid, settings=default_settings):
        self.constants = constants
        self.correlations = correlations
        self.liquid0 = self.liquid = liquid
        self.gas = gas
        self.settings = settings
        self._finish_initialization()

    def _finish_initialization(self):
        constants, correlations = self.constants, self.correlations
        gas, liquid, settings = self.gas, self.liquid, self.settings
        self.liquids = liquids = [liquid]
        self.N = constants.N

        self.stab = StabilityTester(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)

        # self.flash_pure = FlashPureVLS(constants=constants, correlations=correlations,
        #                                gas=gas, liquids=[liquid], solids=[],
        #                                settings=settings)

        self.K_composition_independent = gas.composition_independent and liquid.composition_independent
        self.ideal_gas_basis = gas.ideal_gas_basis and liquid.ideal_gas_basis


        if gas is None:
            raise ValueError("Gas model is required")
        if liquid is None:
            raise ValueError("Liquid model is required")

        self.phases = [gas, liquid]

        self._finish_initialization_base()

    def phases_at_TP_binary(self, T, P, zs, liq, gas):
        liquid = liq.to(T=T, P=P, zs=zs)
        if self.ceos_gas_liquid_compatible:
            gas = gas.to_TP_zs(T, P, zs, other_eos=liquid.eos_mix)
        else:
            gas = gas.to(T=T, P=P, zs=zs)
        return (gas, liquid)

    def flash_TVF(self, T, VF, zs, solution=None, hot_start=None):
        return self.flash_TVF_2P(T, VF, zs, self.liquid, self.gas, solution=solution, hot_start=hot_start)

    def flash_TVF_2P(self, T, VF, zs, liquid, gas, solution=None, hot_start=None):
        if self.K_composition_independent:
            # Assume pressure independent for guess
            P, xs, ys, iterations, err = solve_T_VF_IG_K_composition_independent(VF, T, zs, gas, liquid, xtol=self.DEW_BUBBLE_VF_K_COMPOSITION_INDEPENDENT_XTOL)
            l, g = liquid.to(T=T, P=P, zs=xs), gas.to(T=T, P=P, zs=ys)
            return P, l, g, iterations, err

        constants, correlations = self.constants, self.correlations

        dew_bubble_xtol = self.DEW_BUBBLE_QUASI_NEWTON_XTOL
        dew_bubble_newton_xtol = self.DEW_BUBBLE_NEWTON_XTOL
        dew_bubble_maxiter = self.DEW_BUBBLE_QUASI_NEWTON_MAXITER

        if hot_start is not None:
            P, xs, ys = hot_start.P, hot_start.liquid0.zs, hot_start.gas.zs
        else:
            for method in self.VF_guess_methods:
                try:
                    if method is dew_bubble_newton_zs:
                        xtol = dew_bubble_newton_xtol
                    else:
                        xtol = dew_bubble_xtol
                    _, P, _, xs, ys = TP_solve_VF_guesses(zs=zs, method=method, constants=constants,
                                                           correlations=correlations, T=T, VF=VF,
                                                           xtol=xtol, maxiter=dew_bubble_maxiter)
                    break
                except Exception as e:
                    pass
                    # print(e)

        if VF == 1.0:
            dew = True
            integral_VF = True
            comp_guess = xs
            algos = self.dew_T_flash_algos
        elif VF == 0.0:
            dew = False
            integral_VF = True
            comp_guess = ys
            algos = self.bubble_T_flash_algos
        else:
            integral_VF = False
            algos = self.VF_flash_algos

        for algo in algos:
            try:
                if algo is dew_bubble_bounded_naive:
                    if self.unique_liquid_count > 1:
                        # cannott force flash each liquid easily
                        continue
                    sln = dew_bubble_bounded_naive(guess=P, fixed_val=T, zs=zs, flasher=self, iter_var='P', fixed_var='T', V_over_F=VF,
                                                   maxiter=dew_bubble_maxiter, xtol=dew_bubble_xtol)
                    return sln
                else:
                    sln = algo(P, fixed_val=T, zs=zs, liquid_phase=liquid, gas_phase=gas,
                                iter_var='P', fixed_var='T', V_over_F=VF,
                                maxiter=dew_bubble_maxiter, xtol=dew_bubble_xtol,
                                comp_guess=comp_guess)
                break
            except Exception as e:
                # print(e)
                continue

        guess, comp_guess, iter_phase, const_phase, iterations, err = sln
        if dew:
            l, g = iter_phase, const_phase
        else:
            l, g = const_phase, iter_phase

        return guess, l, g, iterations, err

        # else:
        #     raise NotImplementedError("TODO")

    def flash_PVF(self, P, VF, zs, solution=None, hot_start=None):
        return self.flash_PVF_2P(P, VF, zs, self.liquid, self.gas, solution=solution, hot_start=hot_start)

    def flash_PVF_2P(self, P, VF, zs, liquid, gas, solution=None, hot_start=None):
        if self.K_composition_independent:
            # Assume pressure independent for guess
            T, xs, ys, iterations, err = solve_P_VF_IG_K_composition_independent(VF, P, zs, gas, liquid, xtol=1e-10)
            l, g = liquid.to(T=T, P=P, zs=xs), gas.to(T=T, P=P, zs=ys)
            return T, l, g, iterations, err
        constants, correlations = self.constants, self.correlations

        dew_bubble_xtol = self.DEW_BUBBLE_QUASI_NEWTON_XTOL
        dew_bubble_maxiter = self.DEW_BUBBLE_QUASI_NEWTON_MAXITER
        dew_bubble_newton_xtol = self.DEW_BUBBLE_NEWTON_XTOL
        if hot_start is not None:
            T, xs, ys = hot_start.T, hot_start.liquid0.zs, hot_start.gas.zs
        else:
            for method in self.VF_guess_methods:
                try:
                    if method is dew_bubble_newton_zs:
                        xtol = dew_bubble_newton_xtol
                    else:
                        xtol = dew_bubble_xtol
                    T, _, _, xs, ys = TP_solve_VF_guesses(zs=zs, method=method, constants=constants,
                                                           correlations=correlations, P=P, VF=VF,
                                                           xtol=xtol, maxiter=dew_bubble_maxiter)
                    break
                except Exception as e:
                    pass
                    # print(e)

        if VF == 1.0:
            dew = True
            integral_VF = True
            comp_guess = xs
            algos = self.dew_P_flash_algos
        elif VF == 0.0:
            dew = False
            integral_VF = True
            comp_guess = ys
            algos = self.bubble_P_flash_algos
        else:
            integral_VF = False
            algos = self.VF_flash_algos

        for algo in algos:
            try:
                if algo is dew_bubble_bounded_naive:
                    if self.unique_liquid_count > 1:
                        # cannot force flash each liquid easily
                        continue
                    # This one doesn't like being low tolerance because the PT tolerance isn't there
                    sln = dew_bubble_bounded_naive(guess=T, fixed_val=P, zs=zs, flasher=self, iter_var='T', fixed_var='P', V_over_F=VF,
                                                   maxiter=dew_bubble_maxiter, xtol=max(dew_bubble_xtol, 1e-6), hot_start=hot_start)
                    # This one should never need anything else
                    # as it has its own stability test
                    #stable, info = self.stability_test_Michelsen(sln[2].value('T'), sln[2].value('P'), zs, min_phase=sln[1][0], other_phase=sln[2])
                    #if stable:
                    #    return sln
                    #else:
                    #    continue
                    return sln
                else:

                    sln = algo(T, fixed_val=P, zs=zs, liquid_phase=liquid, gas_phase=gas,
                            iter_var='T', fixed_var='P', V_over_F=VF,
                            maxiter=dew_bubble_maxiter, xtol=dew_bubble_xtol,
                            comp_guess=comp_guess)

                guess, comp_guess, iter_phase, const_phase, iterations, err = sln

                stable, info = self.stability_test_Michelsen(iter_phase.T, iter_phase.P, zs, min_phase=iter_phase, other_phase=const_phase)
                if stable:
                    break
            except Exception as e:
                # print(e)
                continue

        if dew:
            l, g = iter_phase, const_phase
        else:
            l, g = const_phase, iter_phase

        return guess, l, g, iterations, err

    def stability_test_Michelsen(self, T, P, zs, min_phase, other_phase,
                                 existing_comps=None, skip=None,
                                 expect_liquid=False, expect_aqueous=False,
                                 handle_iffy=False, lowest_dG=False,
                                 highest_comp_diff=False, min_comp_diff=None,
                                 all_solutions=False):
        if min_phase.T != T or min_phase.P != P:
            min_phase = min_phase.to_TP_zs(T=T, P=P, zs=zs)
        if other_phase.T != T or other_phase.P != P:
            other_phase = other_phase.to_TP_zs(T=T, P=P, zs=zs)


        existing_phases = len(existing_comps) if existing_comps is not None else 0
        gen = self.stab.incipient_guesses(T, P, zs, expect_liquid=expect_liquid,
                                          expect_aqueous=expect_aqueous, existing_phases=existing_phases) #random=10000 has yet to help
        always_stable = True
        stable = True

        if skip is not None:
            (gen() for i in range(skip))

        iffy_solution = None
        lowest_solution, dG_min = None, -1e100
        comp_diff_solution, comp_diff_max = None, 0.0
        if existing_comps is None:
            existing_comps = [zs]

        if all_solutions:
            all_solutions_list = []

        # The composition for the assumed-stable phase comes from the min_phase object
        # The zs is the true feed.
        fugacities_trial = min_phase.fugacities_lowest_Gibbs()
        zs_trial = min_phase.zs

        if self.supports_lnphis_args and 1:
            other_phase_arg = other_phase.lnphis_args()
            functional = True
        else:
            functional = False
            other_phase_arg = lambda zs: other_phase.lnphis_at_zs(zs, most_stable=True)


        for i, trial_comp in enumerate(gen):
                try:
                    sln = stability_iteration_Michelsen(T=T, P=P, zs_trial=zs_trial, fugacities_trial=fugacities_trial,
                                                        zs_test=trial_comp, test_phase=other_phase_arg,
                                                        maxiter=self.PT_STABILITY_MAXITER, xtol=self.PT_STABILITY_XTOL,
                                                        functional=functional)
                    sum_zs_test, Ks, zs_test, V_over_F, trial_zs, appearing_zs, dG_RT = sln
                    if zs == trial_zs:
                        continue
                    lnK_2_tot = 0.0
                    for k in range(self.N):
                        lnK = trunc_log(Ks[k])
                        lnK_2_tot += lnK*lnK
                    sum_criteria = abs(sum_zs_test - 1.0)
                    if sum_criteria < 1e-9 or lnK_2_tot < 1e-7 or sum_criteria > 1e20 or isinf(lnK_2_tot):
                        continue
                    if existing_comps:
                        existing_phase = False
                        min_diff = 1e100
                        for existing_comp in existing_comps:
                            diff = sum([abs(existing_comp[i] - appearing_zs[i]) for i in range(self.N)])/self.N
                            min_diff = min(min_diff, diff)
                            if diff < 1e-4:
                                existing_phase = True
                                break
                            diffs2 = [abs(1.0-(existing_comp[i]/(appearing_zs[i] if appearing_zs[i]!=0.0 else 1)  )) for i in range(self.N)]
                            diff2 = sum(diffs2)/self.N
                            if diff2 < .02:
                                existing_phase = True
                                break
                        # Continue stability testing if min_diff is too low?
                        if existing_phase:
                            continue
                    # some stability test-driven VFs are converged to about the right solution - but just a little on the other side
                    # For those cases, we need to let SS determine the result
                    stable = V_over_F < -1e-6 or V_over_F > (1.0 + 1e-6) #not (0.0 < V_over_F < 1.0)
                    if not stable:
                        always_stable = stable
                    if all_solutions:
                        stab_guess_name = self.stab.incipient_guess_name(i, expect_liquid=expect_liquid)
                        all_solutions_list.append((trial_zs, appearing_zs, V_over_F, stab_guess_name, i, sum_criteria, lnK_2_tot, dG_RT))
                    if not stable:
                        if highest_comp_diff:
                            if min_diff > comp_diff_max:
                                if min_comp_diff is not None and min_diff > min_comp_diff and not all_solutions:
                                    highest_comp_diff = highest_comp_diff = False
                                    break
                                comp_diff_solution = (trial_zs, appearing_zs, V_over_F, i, sum_criteria, lnK_2_tot, dG_RT)
                                comp_diff_max = min_diff
                            continue

                        if lowest_dG:
                            if dG_RT > dG_min:
                                dG_min = dG_RT
                                lowest_solution = (trial_zs, appearing_zs, V_over_F, i, sum_criteria, lnK_2_tot, dG_RT)
                            continue

                        if handle_iffy and sum_criteria < 1e-5:
                            iffy_solution = (trial_zs, appearing_zs, V_over_F, i, sum_criteria, lnK_2_tot, dG_RT)
#                        continue
                        elif all_solutions:
                            continue
                        else:
                            break

                except UnconvergedError:
                    pass
        if all_solutions:
            return all_solutions_list
        if not always_stable:
            if not lowest_dG and not highest_comp_diff and not handle_iffy:
                pass
            elif highest_comp_diff:
                trial_zs, appearing_zs, V_over_F, i, sum_criteria, lnK_2_tot, dG_RT = comp_diff_solution
            elif lowest_dG:
                trial_zs, appearing_zs, V_over_F, i, sum_criteria, lnK_2_tot, dG_RT = lowest_solution
            elif handle_iffy:
                trial_zs, appearing_zs, V_over_F, i, sum_criteria, lnK_2_tot, dG_RT = iffy_solution
            if skip is not None:
                i += skip
            stab_guess_name = self.stab.incipient_guess_name(i, expect_liquid=expect_liquid)
            return (False, (trial_zs, appearing_zs, V_over_F, stab_guess_name, i, sum_criteria, lnK_2_tot, dG_RT))
        else:
            return (stable, (None, None, None, None, None, None, None, None))


    def flash_TP_stability_test(self, T, P, zs, liquid, gas, solution=None, LL=False, phases_ready=False):
        # gen = self.stab.incipient_guesses(T, P, zs)
        if not phases_ready:
            gas, liquid = self.phases_at_TP_binary(T, P, zs, liquid, gas)
            # gas, liquid = self.phases_at(T=T, P=P, zs=zs)
            # liquid = liquid.to(T=T, P=P, zs=zs)
            # gas = gas.to(T=T, P=P, zs=zs)
        if self.ideal_gas_basis:
            G_liq, G_gas = liquid.G_dep(), gas.G_dep()
        else:
            G_liq, G_gas = liquid.G(), gas.G()
        if G_liq < G_gas: # How handle equal?
            min_phase, other_phase = liquid, gas
        elif G_liq == G_gas:
            min_phase, other_phase = (liquid, gas) if liquid.phase == 'l' else (gas, liquid)
        else:
            min_phase, other_phase = gas, liquid

        stable, (trial_zs, appearing_zs, V_over_F, stab_guess_name, stab_guess_number, stab_sum_zs_test, stab_lnK_2_tot, _) = self.stability_test_Michelsen(
                T, P, zs, min_phase, other_phase, highest_comp_diff=self.SS_2P_STAB_HIGHEST_COMP_DIFF, min_comp_diff=self.SS_2P_STAB_COMP_DIFF_MIN)
        if stable:
            ls, g = ([liquid], None) if min_phase is liquid else ([], gas)
            return g, ls, [], [1.0], {'iterations': 0, 'err': 0.0, 'stab_info': None}
        else:
            return self.flash_2P(T, P, zs, trial_zs, appearing_zs, min_phase, other_phase, gas, liquid,
                                 V_over_F_guess=None, stab_info={'stab_guess_name': stab_guess_name}, LL=LL)

#        stable = True
#        for i, trial_comp in enumerate(gen):
#                try:
#                    sln = stability_iteration_Michelsen(min_phase, trial_comp, test_phase=other_phase,
#                                 maxiter=self.PT_STABILITY_MAXITER, xtol=self.PT_STABILITY_XTOL)
#                    sum_zs_test, Ks, zs_test, V_over_F, trial_zs, appearing_zs = sln
#                    lnK_2_tot = 0.0
#                    for k in range(self.N):
#                        lnK = log(Ks[k])
#                        lnK_2_tot += lnK*lnK
#                    sum_criteria = abs(sum_zs_test - 1.0)
#                    if sum_criteria < 1e-9 or lnK_2_tot < 1e-7:
#                        continue
#                    # some stability test-driven VFs are converged to about the right solution - but just a little on the other side
#                    # For those cases, we need to let SS determine the result
#                    stable = V_over_F < -1e-6 or V_over_F > (1.0 + 1e-6) #not (0.0 < V_over_F < 1.0)
#                    if not stable:
#                        break
#
#                except UnconvergedError:
#                    pass
#        stab_guess_name = self.stab.incipient_guess_name(i)


    def sequential_substitution_2P(self, T, P, zs, xs_guess, ys_guess, liquid_phase, gas_phase, V_over_F_guess, maxiter, tol):
        if self.supports_lnphis_args and 1:

            if liquid_phase.T != T or liquid_phase.P != P:
                liquid_phase = liquid_phase.to_TP_zs(T=T, P=P, zs=xs_guess)
            if gas_phase.T != T or gas_phase.P != P:
                gas_phase = gas_phase.to_TP_zs(T=T, P=P, zs=ys_guess)

            liquid_args = liquid_phase.lnphis_args()
            gas_args = gas_phase.lnphis_args()
            # Can save one fugacity call

            V_over_F, xs, ys, iteration, err = sequential_substitution_2P_functional(T, P, zs=zs, xs_guess=xs_guess, ys_guess=ys_guess,
                               liquid_args=liquid_args, gas_args=gas_args, maxiter=maxiter, tol=tol, trivial_solution_tol=self.PT_TRIVIAL_SOLUTION_TOL,
                               V_over_F_guess=V_over_F_guess)
            l = liquid_phase.to(T=T, P=P, zs=xs)
            g = gas_phase.to(T=T, P=P, zs=ys)
        else:
            V_over_F, xs, ys, l, g, iteration, err = sequential_substitution_2P(T=T, P=P, V=None,
                                                                                zs=zs, xs_guess=xs_guess, ys_guess=ys_guess,
                                                                                liquid_phase=liquid_phase,
                                                                                gas_phase=gas_phase, maxiter=maxiter,
                                                                                tol=tol,trivial_solution_tol=self.PT_TRIVIAL_SOLUTION_TOL,
                                                                                V_over_F_guess=V_over_F_guess)


        return (V_over_F, xs, ys, l, g, iteration, err)


    def flash_2P(self, T, P, zs, trial_zs, appearing_zs, min_phase, other_phase, gas, liquid,
                 V_over_F_guess=None, stab_info=None, LL=False):
        if 0:
            self.PT_converge(T=T, P=P, zs=zs, xs_guess=trial_zs, ys_guess=appearing_zs, liquid_phase=min_phase,
                        gas_phase=other_phase, V_over_F_guess=V_over_F_guess)
        try:
            V_over_F, xs, ys, l, g, iteration, err = self.sequential_substitution_2P(T=T, P=P,
                zs=zs, xs_guess=trial_zs, ys_guess=appearing_zs,
                liquid_phase=min_phase, gas_phase=other_phase,
                V_over_F_guess=V_over_F_guess, tol=self.PT_SS_TOL,
                maxiter=self.PT_SS_MAXITER)



        except TrivialSolutionError:
            ls, g = ([liquid], None) if min_phase is liquid else ([], gas)
            return g, ls, [], [1.0], {'iterations': 0, 'err': 0.0, 'stab_info': stab_info}

        if V_over_F < self.PT_SS_POLISH_VF or V_over_F > 1.0-self.PT_SS_POLISH_VF:
            # Continue the SS, with the previous values, to a much tighter tolerance - if specified/allowed
            if (V_over_F > -self.PT_SS_POLISH_VF or V_over_F > 1.0 + self.PT_SS_POLISH_VF) and self.PT_SS_POLISH:
                V_over_F, xs, ys, l, g, iteration, err = self.sequential_substitution_2P(T=T, P=P,
                        zs=zs, xs_guess=xs, ys_guess=ys, liquid_phase=l,  gas_phase=g,
                        V_over_F_guess=V_over_F, tol=self.PT_SS_POLISH_TOL, maxiter=self.PT_SS_POLISH_MAXITER)

            if V_over_F < 0.0 or V_over_F > 1.0:

                ls, g = ([liquid], None) if min_phase is liquid else ([], gas)
                return g, ls, [], [1.0], {'iterations': iteration, 'err': err, 'stab_info': stab_info}
        if LL:
            return None, [g, l], [], [V_over_F, 1.0 - V_over_F], {'iterations': iteration, 'err': err,
                                                                    'stab_info': stab_info}

        if min_phase is liquid:
            ls, g, V_over_F = [l], g, V_over_F
        else:
            ls, g, V_over_F = [g], l, 1.0 - V_over_F

        return g, ls, [], [V_over_F, 1.0 - V_over_F], {'iterations': iteration, 'err': err, 'stab_info': stab_info}

    def PT_converge(self, T, P, zs, xs_guess, ys_guess, liquid_phase,
                    gas_phase, V_over_F_guess=0.5):
        for algo in self.PT_algorithms:
            try:
                sln = algo(T=T, P=P, zs=zs, xs_guess=xs_guess, ys_guess=ys_guess, liquid_phase=liquid_phase,
                  gas_phase=gas_phase, V_over_F_guess=V_over_F_guess)
                return sln
            except Exception as e:
                pass
                # a = 1

    def flash_PV(self, P, V, zs, solution=None, hot_start=None):
        return self.flash_TPV_HSGUA(fixed_val=P, spec_val=V, fixed_var='P', spec='V',
                            iter_var='T', zs=zs, solution=solution,
                            hot_start=hot_start)


    def flash_TV(self, T, V, zs, solution=None, hot_start=None):
        return self.flash_TPV_HSGUA(fixed_val=T, spec_val=V, fixed_var='T', spec='V',
                            iter_var='P', zs=zs, solution=solution,
                            hot_start=hot_start)



    def flash_TPV(self, T, P, V, zs=None, solution=None, hot_start=None):
        if T is None:
            return self.flash_PV(P, V, zs, solution, hot_start)
        if P is None:
            return self.flash_TV(T, V, zs, solution, hot_start)
        if hot_start is not None:
            try:
                VF_guess, xs, ys = hot_start.beta_gas, hot_start.liquid0.zs, hot_start.gas.zs
                liquid, gas = self.liquid, self.gas

                V_over_F, xs, ys, l, g, iteration, err = self.sequential_substitution_2P(
                    T=T, P=P,
                    zs=zs, xs_guess=xs, ys_guess=ys, liquid_phase=liquid,
                    gas_phase=gas, maxiter=self.PT_SS_MAXITER, tol=self.PT_SS_TOL,
                    V_over_F_guess=VF_guess
                )
                assert 0.0 <= V_over_F <= 1.0
                return g, [l], [], [V_over_F, 1.0 - V_over_F], {'iterations': iteration, 'err': err}
            except Exception as e:
                # print('FAILED from hot start TP')
                pass


        return self.flash_TP_stability_test(T, P, zs, self.liquid, self.gas, solution=solution)

    def flash_TPV_HSGUA(self, fixed_val, spec_val, fixed_var='P', spec='H',
                        iter_var='T', zs=None, solution=None,
                        selection_fun_1P=None, hot_start=None, spec_fun=None):

        constants, correlations = self.constants, self.correlations
        if solution is None:
            if fixed_var == 'P' and spec == 'H':
                fun = lambda obj: -obj.S()
            elif fixed_var == 'P' and spec == 'S':
                fun = lambda obj: obj.H() # Michaelson
            elif fixed_var == 'V' and spec == 'U':
                fun = lambda obj: -obj.S()
            elif fixed_var == 'V' and spec == 'S':
                fun = lambda obj: obj.U()
            elif fixed_var == 'P' and spec == 'U':
                fun = lambda obj: -obj.S() # promising
            else:
                fun = lambda obj: obj.G()
        else:
            if solution == 'high':
                fun = lambda obj: -obj.value(iter_var)
            elif solution == 'low':
                fun = lambda obj: obj.value(iter_var)
            elif callable(solution):
                fun = solution
            else:
                raise ValueError("Unrecognized solution")

        if selection_fun_1P is None:
            def selection_fun_1P(new, prev):
                return new[-1] < prev[-1]

        if 0:
            try:
                solutions_1P = []
                G_min = 1e100
                results_G_min_1P = None
                last_conv = None
                if hot_start is not None:
                    last_conv = hot_start.value(iter_var)
                for phase in self.unique_phases:
                    try:
                        T, P, phase, iterations, err = solve_PTV_HSGUA_1P(
                            phase, zs, fixed_val, spec_val, fixed_var=fixed_var,
                            spec=spec, iter_var=iter_var, constants=constants,
                            correlations=correlations, last_conv=last_conv
                        )
                        G = fun(phase)
                        new = [T, phase, iterations, err, G]
                        if results_G_min_1P is None or selection_fun_1P(new, results_G_min_1P):
                            G_min = G
                            results_G_min_1P = new

                        solutions_1P.append(new)
                    except Exception as e:
    #                    print(e)
                        solutions_1P.append(None)
            except:
                pass

        if 1:
            try:
                res, flash_convergence = self.solve_PT_HSGUA_NP_guess_bisect(zs, fixed_val, spec_val,
                                                               fixed_var=fixed_var, spec=spec, iter_var=iter_var,
                                                                             hot_start=hot_start, spec_fun=spec_fun)
                return None, res.phases, [], res.betas, flash_convergence
            except:
                g, ls, ss, betas, flash_convergence = self.solve_PT_HSGUA_NP_guess_newton_2P(zs, fixed_val, spec_val,
                                                                                             fixed_var=fixed_var,
                                                                                             spec=spec,
                                                                                             iter_var=iter_var)
                return g, ls, ss, betas, flash_convergence
        if 1:
            g, ls, ss, betas, flash_convergence = self.solve_PT_HSGUA_NP_guess_newton_2P(zs, fixed_val, spec_val,
                                                           fixed_var=fixed_var, spec=spec, iter_var=iter_var)
            return g, ls, ss, betas, flash_convergence

# Need to return g, ls, ss, betas, flash_convergence

    def bounds_PT_HSGUA(self, iter_var='T'):
        if iter_var == 'T':
            min_bound = phases.Phase.T_MIN_FIXED
            max_bound = phases.Phase.T_MAX_FIXED
            for p in self.phases:
                if isinstance(p, phases.CoolPropPhase):
                    min_bound = max(p.AS.Tmin(), min_bound)
                    max_bound = min(p.AS.Tmax(), max_bound)
        elif iter_var == 'P':
            min_bound = phases.Phase.P_MIN_FIXED*(1.0 - 1e-12)
            max_bound = phases.Phase.P_MAX_FIXED*(1.0 + 1e-12)
            for p in self.phases:
                if isinstance(p, phases.CoolPropPhase):
                    AS = p.AS
                    max_bound = min(AS.pmax()*(1.0 - 1e-7), max_bound)
                    min_bound = max(AS.trivial_keyed_output(CPiP_min)*(1.0 + 1e-7), min_bound)
        elif iter_var == 'V':
            min_bound = phases.Phase.V_MIN_FIXED
            max_bound = phases.Phase.V_MAX_FIXED
        return min_bound, max_bound

    def solve_PT_HSGUA_NP_guess_newton_2P(self, zs, fixed_val, spec_val,
                                          fixed_var='P', spec='H', iter_var='T',):
        phases = self.phases
        constants = self.constants
        correlations = self.correlations
        min_bound, max_bound = self.bounds_PT_HSGUA()
        init_methods = [SHAW_ELEMENTAL, IDEAL_WILSON]

        for method in init_methods:
            try:
                guess, VF, xs, ys = TPV_solve_HSGUA_guesses_VL(
                    zs, method, constants, correlations,
                    fixed_val, spec_val,
                    iter_var=iter_var, fixed_var=fixed_var, spec=spec,
                    maxiter=50, xtol=1E-5, ytol=None,
                    bounded=False, min_bound=min_bound, max_bound=max_bound,
                    user_guess=None, last_conv=None, T_ref=298.15,
                    P_ref=101325.0
                )
                break
            except Exception as e:
                # print(e)
                pass

        sln = nonlin_spec_NP(guess, fixed_val, spec_val, zs, [xs, ys], [1.0-VF, VF],
                             [self.liquids[0], self.gas], iter_var=iter_var, fixed_var=fixed_var, spec=spec,
                             maxiter=self.TPV_HSGUA_NEWTON_MAXITER, tol=self.TPV_HSGUA_NEWTON_XTOL,
                             trivial_solution_tol=1e-5, ref_phase=-1,
                             method=self.TPV_HSGUA_NEWTON_SOLVER,
                             solve_kwargs=None, debug=False,
                             analytical_jac=self.HSGUA_NEWTON_ANALYTICAL_JAC)
        iter_val, betas, compositions, phases, errs, _, iterations = sln

        return None, phases, [], betas, {'errs': errs, 'iterations': iterations}



    def solve_PT_HSGUA_NP_guess_bisect(self, zs, fixed_val, spec_val,
                                       fixed_var='P', spec='H', iter_var='T', hot_start=None, spec_fun=None):
        phases = self.phases
        constants = self.constants
        correlations = self.correlations
        min_bound, max_bound = self.bounds_PT_HSGUA()

        init_methods = [SHAW_ELEMENTAL, IDEAL_WILSON]
        guess = None

        last_conv = None
        if hot_start is not None:
            guess = hot_start.value(iter_var)
        else:
            for method in init_methods:
                try:
                    guess, VF, xs, ys = TPV_solve_HSGUA_guesses_VL(
                        zs, method, constants, correlations,
                        fixed_val, spec_val,
                        iter_var=iter_var, fixed_var=fixed_var, spec=spec,
                        maxiter=50, xtol=1E-5, ytol=None,
                        bounded=False, min_bound=min_bound, max_bound=max_bound,
                        user_guess=None, last_conv=last_conv, T_ref=298.15,
                        P_ref=101325.0
                    )
                    break
                except NotImplementedError:
                    continue
                except Exception as e:
                    #print(e)
                    pass
        if guess is None:
            if iter_var == 'T':
                guess = 298.15
            elif iter_var == 'P':
                guess = 101325.0
            elif iter_var == 'V':
                guess = 0.024465403697038125
        sln = []
        global iterations
        iterations = 0
        kwargs = {fixed_var: fixed_val, 'zs': zs}
        def to_solve(iter_val):
            global iterations
            iterations += 1
            kwargs[iter_var] = iter_val
            res = self.flash(**kwargs)
            if spec_fun is not None:
                err = getattr(res, spec)() - spec_fun(res)
            else:
                err = getattr(res, spec)() - spec_val
            sln[:] = (res, err)
            # print(f'{iter_val=}, {err=}')
            return err

        ytol = abs(spec_val)*self.TPV_HSGUA_BISECT_YTOL
        sln_val = secant(to_solve, guess, xtol=self.TPV_HSGUA_BISECT_XTOL, ytol=ytol,
                         require_xtol=self.TPV_HSGUA_BISECT_YTOL_ONLY, require_eval=True, bisection=True,
                         low=min_bound, high=max_bound, maxiter=self.TPV_HSGUA_SECANT_MAXITER)
        return sln[0], {'iterations': iterations, 'err': sln[1]}


