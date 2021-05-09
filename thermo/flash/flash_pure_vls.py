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
SOFTWARE.'''

__all__ = ['FlashPureVLS']

from thermo.flash.flash_base import Flash
from fluids.numerics import (
    numpy as np,
    secant,
    brenth,
    newton,
    linspace,
    assert_close,
    UnconvergedError,
    NoSolutionError
)
from chemicals.utils import rho_to_Vm, Vm_to_rho
from chemicals.exceptions import PhaseExistenceImpossible
from thermo.bulk import default_settings
from thermo.eos_mix import IGMIX
from thermo.coolprop import CPiP_min
from thermo.phases.coolprop_phase import (
    caching_state_CoolProp,
    CPPQ_INPUTS, 
    CPQT_INPUTS, 
    CPunknown, 
    CPiDmolar,
)
from chemicals.iapws import (
    iapws95_Psat, 
    iapws95_Tsat, 
    iapws95_rhog_sat, 
    iapws95_rhol_sat, 
    iapws95_Tc, 
    iapws95_Pc, 
    iapws95_MW, 
    iapws95_T
)
from thermo.phases import (
    Phase,
    CEOSGas, 
    CEOSLiquid, 
    IdealGas, 
    CoolPropGas, 
    CoolPropLiquid,
    IAPWS95Gas,
    IAPWS95Liquid,
    GibbsExcessLiquid,
    DryAirLemmon
)
from thermo.flash.flash_utils import (
    TVF_pure_secant,
    PVF_pure_newton,
    TSF_pure_newton,
    PSF_pure_newton,
    solve_PTV_HSGUA_1P
)


class FlashPureVLS(Flash):
    r'''Class for performing flash calculations on pure-component systems.
    This class is subtantially more robust than using multicomponent algorithms
    on pure species. It is also faster. All parameters are also attributes.

    The minimum information that is needed in addition to the :obj:`Phase`
    objects is:

    * MW
    * Vapor pressure curve if including liquids
    * Sublimation pressure curve if including solids
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
        A list of phases for representing the liquid phase; normally only one
        liquid phase is present for a pure-component system, but multiple
        liquids are allowed for the really weird cases like having both
        parahydrogen and orthohydrogen. The liquid phase which calculates a
        lower Gibbs free energy is always used. [-]
    solids : list[:obj:`Phase <thermo.phases.Phase>`]
        A list of phases for representing the solid phase; it is very common
        for multiple solid forms of a compound to exist. For water ice, the
        list is very long - normally ice is in phase Ih but other phases are Ic,
        II, III, IV, V, VI, VII, VIII, IX, X, XI, XII, XIII, XIV, XV, XVI,
        Square ice, and Amorphous ice. It is less common for there to be
        published, reliable, thermodynamic models for these different phases;
        for water there is the IAPWS-06 model for Ih, and another model
        `here <https://aip.scitation.org/doi/10.1063/1.1931662>`_
        for phases Ih, Ic, II, III, IV, V, VI, IX, XI, XII. [-]
    settings : :obj:`BulkSettings <thermo.bulk.BulkSettings>` object
        Object containing settings for calculating bulk and transport
        properties, [-]

    Attributes
    ----------
    VL_IG_hack : bool
        Whether or not to trust the saturation curve of the liquid phase;
        applied automatically to the
        :obj:`GibbsExcessLiquid <thermo.phases.GibbsExcessLiquid>`
        phase if there is a single liquid only, [-]
    VL_EOS_hacks : bool
        Whether or not to trust the saturation curve of the EOS liquid phase;
        applied automatically to the
        :obj:`CEOSLiquid <thermo.phases.CEOSLiquid>`
        phase if there is a single liquid only, [-]
    TPV_HSGUA_guess_maxiter : int
        Maximum number of iterations to try when converging a shortcut model
        for flashes with one (`T`, `P`, `V`) spec and one (`H`, `S`, `G`, `U`,
        `A`) spec, [-]
    TPV_HSGUA_guess_xtol : float
        Convergence tolerance in the iteration variable when converging a
        shortcut model for flashes with one (`T`, `P`, `V`) spec and one (`H`,
        `S`, `G`, `U`, `A`) spec, [-]
    TPV_HSGUA_maxiter : int
        Maximum number of iterations to try when converging a flashes with one
        (`T`, `P`, `V`) spec and one (`H`, `S`, `G`, `U`, `A`) spec; this is
        on a per-phase basis, so if there is a liquid and a gas phase, the
        maximum number of iterations that could end up being tried would be
        twice this, [-]
    TPV_HSGUA_xtol : float
        Convergence tolerance in the iteration variable dimension when
        converging a flash with one (`T`, `P`, `V`) spec and one (`H`, `S`,
        `G`, `U`, `A`) spec, [-]
    TVF_maxiter : int
        Maximum number of iterations to try when converging a flashes with a
        temperature and vapor fraction specification, [-]
    TVF_xtol : float
        Convergence tolerance in the temperature dimension when converging a
        flashes with a temperature and vapor fraction specification, [-]
    PVF_maxiter : int
        Maximum number of iterations to try when converging a flashes with a
        pressure and vapor fraction specification, [-]
    PVF_xtol : float
        Convergence tolerance in the pressure dimension when converging a
        flashes with a pressure and vapor fraction specification, [-]
    TSF_maxiter : int
        Maximum number of iterations to try when converging a flashes with a
        temperature and solid fraction specification, [-]
    TSF_xtol : float
        Convergence tolerance in the temperature dimension when converging a
        flashes with a temperature and solid fraction specification, [-]
    PSF_maxiter : int
        Maximum number of iterations to try when converging a flashes with a
        pressure and solid fraction specification, [-]
    PSF_xtol : float
        Convergence tolerance in the pressure dimension when converging a
        flashes with a pressure and solid fraction specification, [-]


    Notes
    -----
    The algorithms in this object are mostly from [1]_ and [2]_. They all
    boil down to newton methods with analytical derivatives. The phase with
    the lowest Gibbs energy is the most stable if there are multiple
    solutions.

    Phase input combinations which have specific simplifying assumptions
    (and thus more speed) are:

    * a :obj:`CEOSLiquid <thermo.phases.CEOSLiquid>` and a :obj:`CEOSGas <thermo.phases.CEOSGas>` with the same (consistent) parameters
    * a :obj:`CEOSGas <thermo.phases.CEOSGas>` with the :obj:`IGMIX <thermo.eos_mix.IGMIX>` eos and a :obj:`GibbsExcessLiquid <thermo.phases.GibbsExcessLiquid>`
    * a :obj:`IAPWS95Liquid <thermo.phases.IAPWS95Liquid>` and a :obj:`IAPWS95Gas <thermo.phases.IAPWS95Gas>`
    * a :obj:`CoolPropLiquid <thermo.phases.CoolPropLiquid>` and a :obj:`CoolPropGas <thermo.phases.CoolPropGas>`

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

    Create all the necessary objects using all of the default parameters for
    decane and do a flash at 300 K and 1 bar:

    >>> from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CEOSGas, FlashPureVLS
    >>> constants, correlations = ChemicalConstantsPackage.from_IDs(['decane'])
    >>> eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    >>> liquid = CEOSLiquid(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
    >>> gas = CEOSGas(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
    >>> flasher = FlashPureVLS(constants, correlations, gas=gas, liquids=[liquid], solids=[])
    >>> print(flasher.flash(T=300, P=1e5))
    <EquilibriumState, T=300.0000, P=100000.0000, zs=[1.0], betas=[1.0], phases=[<CEOSLiquid, T=300 K, P=100000 Pa>]>

    Working with steam:

    >>> from thermo import FlashPureVLS, IAPWS95Liquid, IAPWS95Gas, iapws_constants, iapws_correlations
    >>> liquid = IAPWS95Liquid(T=300, P=1e5, zs=[1])
    >>> gas = IAPWS95Gas(T=300, P=1e5, zs=[1])
    >>> flasher = FlashPureVLS(iapws_constants, iapws_correlations, gas, [liquid], [])
    >>> PT = flasher.flash(T=800.0, P=1e7)
    >>> PT.rho_mass()
    29.1071839176
    >>> print(flasher.flash(T=600, VF=.5))
    <EquilibriumState, T=600.0000, P=12344824.3572, zs=[1.0], betas=[0.5, 0.5], phases=[<IAPWS95Gas, T=600 K, P=1.23448e+07 Pa>, <IAPWS95Liquid, T=600 K, P=1.23448e+07 Pa>]>
    >>> print(flasher.flash(T=600.0, H=50802))
    <EquilibriumState, T=600.0000, P=10000469.1288, zs=[1.0], betas=[1.0], phases=[<IAPWS95Gas, T=600 K, P=1.00005e+07 Pa>]>
    >>> print(flasher.flash(P=1e7, S=104.))
    <EquilibriumState, T=599.6790, P=10000000.0000, zs=[1.0], betas=[1.0], phases=[<IAPWS95Gas, T=599.679 K, P=1e+07 Pa>]>
    >>> print(flasher.flash(V=.00061, U=55850))
    <EquilibriumState, T=800.5922, P=10144789.0899, zs=[1.0], betas=[1.0], phases=[<IAPWS95Gas, T=800.592 K, P=1.01448e+07 Pa>]>

    References
    ----------
    .. [1] Poling, Bruce E., John M. Prausnitz, and John P. O’Connell. The
       Properties of Gases and Liquids. 5th edition. New York: McGraw-Hill
       Professional, 2000.
    .. [2] Gmehling, Jürgen, Michael Kleiber, Bärbel Kolbe, and Jürgen Rarey.
       Chemical Thermodynamics for Process Simulation. John Wiley & Sons, 2019.
    '''
    VF_interpolators_built = False
    N = 1
    VL_EOS_hacks = True
    VL_IG_hack = True

    TPV_HSGUA_guess_maxiter = 50
    TPV_HSGUA_guess_xtol = 1e-7
    TPV_HSGUA_maxiter = 80
    TPV_HSGUA_xtol = 1e-10

    TVF_maxiter = 200
    TVF_xtol = 1e-10

    PVF_maxiter = 200
    PVF_xtol = 1e-10

    TSF_maxiter = 200
    TSF_xtol = 1e-10

    PSF_maxiter = 200
    PSF_xtol = 1e-10

    def __repr__(self):
        return "FlashPureVLS(gas=%s, liquids=%s, solids=%s)" %(self.gas, self.liquids, self.solids)
    def __init__(self, constants, correlations, gas, liquids, solids,
                 settings=default_settings):
        # These attributes are all that needs to be stored, then call _finish_initialization
        self.constants = constants
        self.correlations = correlations
        self.solids = solids
        self.liquids = liquids
        self.gas = gas
        self.settings = settings

        self._finish_initialization()

    def _finish_initialization(self):
        solids = self.solids
        liquids = self.liquids
        gas = self.gas


        self.gas_count = 1 if gas is not None else 0
        self.liquid_count = len(liquids)
        self.liquid = liquids[0] if len(liquids) else None
        self.solid_count = len(solids)
        
        self.supports_VF_flash = self.gas_count != 0 and self.liquid_count != 0
        self.supports_SF_flash = (self.gas_count != 0 or self.liquid_count != 0) and self.solid_count != 0

        self.skip_solids = not bool(solids)

        self.phase_count = self.gas_count + self.liquid_count + self.solid_count

        if gas is not None:
            phases = [gas] + liquids + solids

        else:
            phases = liquids + solids
        self.phases = phases


        for i, l in enumerate(self.liquids):
            setattr(self, 'liquid' + str(i), l)
        for i, s in enumerate(self.solids):
            setattr(self, 'solid' + str(i), s)

        self.VL_only = self.phase_count == 2 and self.liquid_count == 1 and self.gas is not None
        self.VL_only_CEOSs = (self.VL_only 
                              and gas 
                              and liquids 
                              and isinstance(self.liquids[0], CEOSLiquid) 
                              and isinstance(self.gas, CEOSGas))

        self.VL_only_IAPWS95 = (len(liquids) == 1
                                and (isinstance(liquids[0], IAPWS95Liquid)
                                     or liquids[0].__class__.__name__ == 'IAPWS95Liquid')
                                 and (isinstance(gas, IAPWS95Gas)
                                      or  gas.__class__.__name__ == 'IAPWS95Gas')
                                and (not solids))

        self.V_only_lemmon2000 = (len(liquids) == 0
                                 and (isinstance(gas, DryAirLemmon)
                                      or  gas.__class__.__name__ == 'DryAirLemmon')
                                and (not solids))

        # TODO implement as function of phases/or EOS
        self.VL_only_CEOSs_same = (self.VL_only_CEOSs and
                                   self.liquids[0].eos_class is self.gas.eos_class
                                   # self.liquids[0].kijs == self.gas.kijs
                                   and not (isinstance(self.liquids[0], (IGMIX,))
                                            or isinstance(self.gas, (IGMIX,))) 
                                   and self.VL_EOS_hacks)

        self.VL_only_CoolProp = (len(liquids) == 1 
                                 and isinstance(liquids[0], CoolPropLiquid)
                                 and isinstance(gas, CoolPropGas)
                                 and (not solids) and liquids[0].backend == gas.backend and
                                 liquids[0].fluid == gas.fluid)

        self.VL_IG_activity = (len(liquids) == 1
                               and isinstance(liquids[0], GibbsExcessLiquid)
                               and (isinstance(gas, IdealGas) or gas.eos_class is IGMIX)
                               and len(solids) == 0)

        if self.VL_only_CEOSs_same:
            # Two phase pure eoss are two phase up to the critical point only! Then one phase
            self.eos_pure_STP = gas.eos_mix.to_TPV_pure(T=298.15, P=101325.0, V=None, i=0)


        liquids_to_unique_liquids = []
        unique_liquids, unique_liquid_hashes = [], []
        for i, l in enumerate(liquids):
            h = l.model_hash()
            if h not in unique_liquid_hashes:
                unique_liquid_hashes.append(h)
                unique_liquids.append(l)
                liquids_to_unique_liquids.append(i)
            else:
                liquids_to_unique_liquids.append(unique_liquid_hashes.index(h))
        if gas is not None:
            gas_hash = gas.model_hash(True)

        gas_to_unique_liquid = None
        for i, l in enumerate(liquids):
            h = l.model_hash(True)
            if gas is not None and gas_hash == h:
                gas_to_unique_liquid = liquids_to_unique_liquids[i]
                break

        self.gas_to_unique_liquid = gas_to_unique_liquid
        self.liquids_to_unique_liquids = liquids_to_unique_liquids

        self.unique_liquids = unique_liquids
        self.unique_liquid_count = len(unique_liquids)
        self.unique_phases = [gas] + unique_liquids if gas is not None else unique_liquids
        if solids:
            self.unique_phases += solids
        self.unique_phase_count = (1 if gas is not None else 0) + self.unique_liquid_count + len(solids)
        self.unique_liquid_hashes = unique_liquid_hashes
        self.T_MIN_FLASH = max(p.T_MIN_FLASH for p in self.phases)



    def flash_TPV(self, T, P, V, zs=None, solution=None, hot_start=None):
        betas = [1.0]

        if solution is None:
            fun = lambda obj: obj.G()
        elif solution == 'high':
            fun = lambda obj: -obj.T
        elif solution == 'low':
            fun = lambda obj: obj.T
        elif callable(solution):
            fun = solution
        else:
            raise ValueError("Did not recognize solution %s" %(solution))

        if self.phase_count == 1:
            phase = self.phases[0].to(zs=zs, T=T, P=P, V=V)
            return None, [phase], [], betas, None
        elif self.VL_only_CoolProp:
            sln = self.gas.to(zs, T=T, P=P, V=V, prefer_phase=8)
#            if sln.phase == 'l':
#                return None, [sln], [], betas, None
            return None, [], [sln], betas, None
        elif self.VL_only_CEOSs_same and V is None and solution is None:
            gas = self.gas.to(zs=zs, T=T, P=P, V=V)
            if gas.eos_mix.phase == 'l/g':
                gas.eos_mix.solve_missing_volumes()
                if gas.eos_mix.G_dep_l < gas.eos_mix.G_dep_g:
                    l = self.liquid.to_TP_zs(T, P, zs, other_eos=gas.eos_mix)
                    return None, [l], [], betas, None
                return gas, [], [], betas, None
            elif gas.eos_mix.phase == 'g':
                return gas, [], [], betas, None
            else:
                return None, [gas], [], betas, None
        elif self.VL_IG_activity and self.VL_IG_hack and V is None and solution is None:
            l = self.liquid.to(zs=zs, T=T, P=P, V=V)
            if P > l.Psats()[0]:
                return None, [l], [], betas, None
            else:
                gas = self.gas.to(zs=zs, T=T, P=P, V=V)
                return gas, [], [], betas, None
        elif self.VL_only_CEOSs_same and V is not None and (T is not None or P is not None) and solution is None:
            gas = self.gas.to(zs=zs, T=T, P=P, V=V)
            if gas.eos_mix.phase == 'g':
                return gas, [], [], betas, None
            else:
                return None, [gas], [], betas, None
        elif self.VL_only_IAPWS95 and solution is None:
            if T is not None:
                if T > iapws95_Tc:
                    # super critical no matter what
                    gas = self.gas.to(zs=zs, T=T, P=P, V=V)
                    return gas, [], [], betas, None
                elif P is not None:
                    Psat = iapws95_Psat(T)
                    if P < Psat:
                        gas = self.gas.to(zs=zs, T=T, P=P, V=V)
                        return gas, [], [], betas, None
                    else:
                        l = self.liquid.to(zs=zs, T=T, P=P, V=V)
                        return None, [l], [], betas, None
                elif V is not None:
                    rhol_sat = iapws95_rhol_sat(T)
                    rho_mass = Vm_to_rho(V, iapws95_MW)
                    if rho_mass >= rhol_sat:
                        l = self.liquid.to(zs=zs, T=T, V=V)
                        return None, [l], [], betas, None
                    rhog_sat = iapws95_rhog_sat(T)
                    if rho_mass <= rhog_sat:
                        gas = self.gas.to(zs=zs, T=T, V=V)
                        return gas, [], [], betas, None
                    # There is no feasible solution between the two curves

            elif P is not None and V is not None:
                T = iapws95_T(P=P, rho=Vm_to_rho(V, iapws95_MW))
                try:
                    Tsat = iapws95_Tsat(P)
                    if T < Tsat:
                        l = self.liquid.to(zs=zs, T=T, V=V)
                        return None, [l], [], betas, None
                    else:
                        gas = self.gas.to(zs=zs, T=T, V=V)
                        return gas, [], [], betas, None
                except:
                    l = self.liquid.to(zs=zs, T=T, V=V)
                    return None, [l], [], betas, None
                # TODO more logic

        if self.gas_count:
            gas = self.gas.to(zs=zs, T=T, P=P, V=V)
            G_min, lowest_phase = fun(gas), gas
        else:
            G_min, lowest_phase = 1e100, None
            gas = None

        liquids = []
        for l in self.liquids:
            l = l.to(zs=zs, T=T, P=P, V=V)
            G = fun(l)
            if G < G_min:
                G_min, lowest_phase = G, l
            liquids.append(l)


        solids = []
        for s in self.solids:
            s = s.to(zs=zs, T=T, P=P, V=V)
            G = fun(s)
            if G < G_min:
                G_min, lowest_phase = G, s
            solids.append(s)

        if lowest_phase is gas:
            return lowest_phase, [], [], betas, None
        elif lowest_phase in liquids:
            return None, [lowest_phase], [], betas, None
        else:
            return None, [], [lowest_phase], betas, None

    def Psat_guess(self, T):
        if self.VL_only_CEOSs_same:
            # Two phase pure eoss are two phase up to the critical point only! Then one phase
            Psat = self.eos_pure_STP.Psat(T)
        #
        else:
            try:
                Psat = self.correlations.VaporPressures[0](T)
            except:
                # Last resort
                Psat = 1e5
        return Psat

    def flash_TVF(self, T, VF=None, zs=None, hot_start=None):
        zs = [1.0]
        if self.VL_only_CoolProp:
            sat_gas_CoolProp = caching_state_CoolProp(self.gas.backend, self.gas.fluid, 1, T, CPQT_INPUTS, CPunknown, None)
            sat_gas = self.gas.from_AS(sat_gas_CoolProp)
            sat_liq = self.liquid.to(zs=zs, T=T, V=1.0/sat_gas_CoolProp.saturated_liquid_keyed_output(CPiDmolar))
            return sat_gas.P, sat_liq, sat_gas, 0, 0.0
        elif self.VL_IG_activity:
            Psat = self.liquid.Psats_at(T)[0]
            sat_gas = self.gas.to_TP_zs(T, Psat, zs)
            sat_liq = self.liquid.to_TP_zs(T, Psat, zs)
            return Psat, sat_liq, sat_gas, 0, 0.0
        elif self.VL_only_IAPWS95:
            if T > iapws95_Tc:
                raise PhaseExistenceImpossible("Specified T is in the supercritical region", zs=zs, T=T)

            Psat = iapws95_Psat(T)
            sat_gas = self.gas.to(T=T, V=rho_to_Vm(iapws95_rhog_sat(T), self.gas._MW), zs=zs)
            sat_liq = self.liquid.to(T=T, V=rho_to_Vm(iapws95_rhol_sat(T), self.liquid._MW), zs=zs)
            return Psat, sat_liq, sat_gas, 0, 0.0
        Psat = self.Psat_guess(T)
        gas = self.gas.to_TP_zs(T, Psat, zs)

        if self.VL_only_CEOSs_same:
            if T > self.constants.Tcs[0]:
                raise PhaseExistenceImpossible("Specified T is in the supercritical region", zs=zs, T=T)

            sat_liq = self.liquids[0].to_TP_zs(T, Psat, zs, other_eos=gas.eos_mix)
            return Psat, sat_liq, gas, 0, 0.0

        liquids = [l.to_TP_zs(T, Psat, zs) for l in self.liquids]
#        return TVF_pure_newton(Psat, T, liquids, gas, maxiter=self.TVF_maxiter, xtol=self.TVF_xtol)
        Psat, l, g, iterations, err = TVF_pure_secant(Psat, T, liquids, gas, maxiter=self.TVF_maxiter, xtol=self.TVF_xtol)
        if l.Z() == g.Z():
            raise PhaseExistenceImpossible("Converged to trivial solution", zs=zs, T=T)

#        print('P', P, 'solved')
        return Psat, l, g, iterations, err

    def flash_PVF(self, P, VF=None, zs=None, hot_start=None):
        zs = [1.0]
        if self.VL_only_CoolProp:
            sat_gas_CoolProp = caching_state_CoolProp(self.gas.backend, self.gas.fluid, P, 1.0, CPPQ_INPUTS, CPunknown, None)
            sat_gas = self.gas.from_AS(sat_gas_CoolProp)
            sat_liq = self.liquids[0].to(zs=zs, T=sat_gas.T, V=1.0/sat_gas_CoolProp.saturated_liquid_keyed_output(CPiDmolar))
            return sat_gas.T, sat_liq, sat_gas, 0, 0.0
        elif self.VL_only_CEOSs_same:
            if P > self.constants.Pcs[0]:
                raise PhaseExistenceImpossible("Specified P is in the supercritical region", zs=zs, P=P)
            try:
                Tsat = self.eos_pure_STP.Tsat(P)
            except:
                raise PhaseExistenceImpossible("Failed to calculate VL equilibrium T; likely supercritical", zs=zs, P=P)
            sat_gas = self.gas.to_TP_zs(Tsat, P, zs)
            sat_liq = self.liquids[0].to_TP_zs(Tsat, P, zs, other_eos=sat_gas.eos_mix)
            return Tsat, sat_liq, sat_gas, 0, 0.0
        elif self.VL_IG_activity:
            Tsat = self.correlations.VaporPressures[0].solve_prop_poly_fit(P)
            sat_gas = self.gas.to_TP_zs(Tsat, P, zs)
            sat_liq = self.liquid.to_TP_zs(Tsat, P, zs)
            return Tsat, sat_liq, sat_gas, 0, 0.0
        elif self.VL_only_IAPWS95:
            if P > iapws95_Pc:
                raise PhaseExistenceImpossible("Specified P is in the supercritical region", zs=zs, P=P)

            Tsat = iapws95_Tsat(P)
            sat_gas = self.gas.to(T=Tsat, V=1e-3*iapws95_MW/iapws95_rhog_sat(Tsat), zs=zs)
            sat_liq = self.liquid.to(T=Tsat, V=1e-3*iapws95_MW/iapws95_rhol_sat(Tsat), zs=zs)
            return Tsat, sat_liq, sat_gas, 0, 0.0
        else:
            Tsat = self.correlations.VaporPressures[0].solve_property(P)
        gas = self.gas.to_TP_zs(Tsat, P, zs)
        liquids = [l.to_TP_zs(Tsat, P, zs) for l in self.liquids]
        Tsat, l, g, iterations, err = PVF_pure_newton(Tsat, P, liquids, gas, maxiter=self.PVF_maxiter, xtol=self.PVF_xtol)
        if l.Z() == g.Z():
            raise PhaseExistenceImpossible("Converged to trivial solution", zs=zs, P=P)
        return Tsat, l, g, iterations, err
#        return PVF_pure_secant(Tsat, P, liquids, gas, maxiter=200, xtol=1E-10)

    def flash_TSF(self, T, SF=None, zs=None, hot_start=None):
        # if under triple point search for gas - otherwise search for liquid
        # For water only there is technically two solutions at some point for both
        # liquid and gas, flag?

        # The solid-liquid interface is NOT working well...
        # Worth getting IAPWS going to compare. Maybe also other EOSs
        if T < self.constants.Tts[0]:
            Psub = self.correlations.SublimationPressures[0](T)
            try_phases = [self.gas] + self.liquids
        else:
            try_phases = self.liquids
            Psub = 1e6

        return TSF_pure_newton(Psub, T, try_phases, self.solids,
                               maxiter=self.TSF_maxiter, xtol=self.TSF_xtol)

    def flash_PSF(self, P, SF=None, zs=None, hot_start=None):
        if P < self.constants.Pts[0]:
            Tsub = self.correlations.SublimationPressures[0].solve_property(P)
            try_phases = [self.gas] + self.liquids
        else:
            try_phases = self.liquids
            Tsub = 1e6

        return PSF_pure_newton(Tsub, P, try_phases, self.solids,
                               maxiter=self.PSF_maxiter, xtol=self.PSF_xtol)


    def flash_double(self, spec_0_val, spec_1_val, spec_0_var, spec_1_var):
        pass


    def flash_TPV_HSGUA_VL_bound_first(self, fixed_var_val, spec_val, fixed_var='P',
                                 spec='H', iter_var='T', hot_start=None,
                                 selection_fun_1P=None, cubic=True):
        constants, correlations = self.constants, self.correlations
        zs = [1.0]
        VL_liq, VL_gas = None, None
        flash_convergence = {}
        has_VL = False
        need_both = True
        if fixed_var == 'T':
            if self.Psat_guess(fixed_var_val) > 1e-2:
                Psat, VL_liq, VL_gas, VL_iter, VL_err = self.flash_TVF(fixed_var_val, VF=.5, zs=zs)
                has_VL = True
        elif fixed_var == 'P':
            if fixed_var_val > 1e-2:
                Tsat, VL_liq, VL_gas, VL_iter, VL_err = self.flash_PVF(fixed_var_val, VF=.5, zs=zs)
                has_VL = True
        if has_VL:
            need_both = False
            spec_val_l = getattr(VL_liq, spec)()
            spec_val_g = getattr(VL_gas, spec)()
            VF = (spec_val - spec_val_l) / (spec_val_g - spec_val_l)
            if 0.0 <= VF <= 1.0:
                return VL_gas, [VL_liq], [], [VF, 1.0 - VF], flash_convergence
            elif VF < 0.0:
                phases = [self.liquid, self.gas]
            else:
                phases = [self.gas, self.liquid]
        else:
            phases = self.phases
        solutions_1P = []
        results_G_min_1P = None
        if hot_start is None:
            last_conv = None
        elif iter_var == 'T':
            last_conv = hot_start.T
        elif iter_var == 'P':
            last_conv = hot_start.P
        for phase in phases:
            try:
                # TODO: use has_VL to bound the solver
                T, P, phase, iterations, err = solve_PTV_HSGUA_1P(phase, zs, fixed_var_val, spec_val, fixed_var=fixed_var,
                                                                  spec=spec, iter_var=iter_var, constants=constants, correlations=correlations, last_conv=last_conv,
                                                                  oscillation_detection=cubic,
                                                                  guess_maxiter=self.TPV_HSGUA_guess_maxiter, guess_xtol=self.TPV_HSGUA_guess_xtol,
                                                                  maxiter=self.TPV_HSGUA_maxiter, xtol=self.TPV_HSGUA_xtol)
                if cubic:
                    phase.eos_mix.solve_missing_volumes()
                    if phase.eos_mix.phase == 'l/g':
                        # Check we are not metastable
                        if min(phase.eos_mix.G_dep_l, phase.eos_mix.G_dep_g) == phase.G_dep(): # If we do not have a metastable phase
                            if isinstance(phase, CEOSGas):
                                g, ls = phase, []
                            else:
                                g, ls = None, [phase]
                            flash_convergence['err'] = err
                            flash_convergence['iterations'] = iterations
                            return g, ls, [], [1.0], flash_convergence
                    else:
                        if isinstance(phase, (CEOSGas, IdealGas)):
                            g, ls = phase, []
                        else:
                            g, ls = None, [phase]
                        flash_convergence['err'] = err
                        flash_convergence['iterations'] = iterations
                        return g, ls, [], [1.0], flash_convergence
                else:
                    if isinstance(phase, (CEOSGas, IdealGas)):
                        g, ls = phase, []
                    else:
                        g, ls = None, [phase]
                    flash_convergence['err'] = err
                    flash_convergence['iterations'] = iterations
                    return g, ls, [], [1.0], flash_convergence

            except Exception as e:
                print(e)
                solutions_1P.append(None)



    def flash_TPV_HSGUA(self, fixed_var_val, spec_val, fixed_var='P', spec='H',
                        iter_var='T', zs=None, solution=None,
                        selection_fun_1P=None, hot_start=None,
                        iter_var_backup=None):
        # Be prepared to have a flag here to handle zero flow
        zs = [1.0]
        constants, correlations = self.constants, self.correlations
        if solution is None:
            if fixed_var == 'P' and spec == 'H':
                fun = lambda obj: -obj.S()
            elif fixed_var == 'P' and spec == 'S':
               # fun = lambda obj: obj.G()
                fun = lambda obj: obj.H() # Michaelson
            elif fixed_var == 'V' and spec == 'U':
                fun = lambda obj: -obj.S()
            elif fixed_var == 'V' and spec == 'S':
                fun = lambda obj: obj.U()
            elif fixed_var == 'P' and spec == 'U':
                fun = lambda obj: -obj.S() # promising
                # fun = lambda obj: -obj.H() # not bad not as good as A
                # fun = lambda obj: obj.A() # Pretty good
                # fun = lambda obj: -obj.V() # First
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

        selection_fun_1P_specified = True
        if selection_fun_1P is None:
            selection_fun_1P_specified = False
            def selection_fun_1P(new, prev):
                if fixed_var == 'P' and spec == 'S':
                    if new[-1] < prev[-1]:
                        if new[0] < 1.0 and prev[0] > 1.0:
                            # Found a very low temperature solution do not take it
                            return False
                        return True
                    elif (prev[0] < 1.0 and new[0] > 1.0):
                        return True

                else:
                    if new[-1] < prev[-1]:
                        return True
                return False

        if (self.VL_only_CEOSs_same or self.VL_IG_activity) and not selection_fun_1P_specified and solution is None and fixed_var != 'V':
            try:
                return self.flash_TPV_HSGUA_VL_bound_first(fixed_var_val=fixed_var_val, spec_val=spec_val, fixed_var=fixed_var,
                                     spec=spec, iter_var=iter_var, hot_start=hot_start, selection_fun_1P=selection_fun_1P, cubic=self.VL_only_CEOSs_same)
            except PhaseExistenceImpossible:
                pass
        elif self.V_only_lemmon2000 and not selection_fun_1P_specified and fixed_var == 'V' and iter_var == 'P':
            # Specifically allow the solution to be specified, equation goes wonky around 50000
            iter_var = 'T'
#            if sln is not None:
#                return sln
        try:
            solutions_1P = []
            G_min = 1e100
            results_G_min_1P = None
            for phase in self.phases:
                # TODO: for eoss wit boundaries, and well behaved fluids, only solve ocne instead of twice (i.e. per phase, doubling the computation.)
                try:
                    T, P, phase, iterations, err = solve_PTV_HSGUA_1P(phase, zs, fixed_var_val, spec_val, fixed_var=fixed_var,
                                                                      spec=spec, iter_var=iter_var, constants=constants, correlations=correlations,
                                                                      guess_maxiter=self.TPV_HSGUA_guess_maxiter, guess_xtol=self.TPV_HSGUA_guess_xtol,
                                                                      maxiter=self.TPV_HSGUA_maxiter, xtol=self.TPV_HSGUA_xtol)

                    G = fun(phase)
                    new = [T, phase, iterations, err, G]
                    if results_G_min_1P is None or selection_fun_1P(new, results_G_min_1P):
#                    if G < G_min:
                        G_min = G
                        results_G_min_1P = new

                    solutions_1P.append(new)
                except Exception as e:
#                    print(e)
                    solutions_1P.append(None)
        except:
            pass


        try:
            VL_liq, VL_gas = None, None
            G_VL = 1e100
            # BUG - P IS NOW KNOWN!
            if self.gas_count and self.liquid_count:
                if fixed_var == 'T' and self.Psat_guess(fixed_var_val) > 1e-2:
                    Psat, VL_liq, VL_gas, VL_iter, VL_err = self.flash_TVF(fixed_var_val, zs=zs, VF=.5)
                elif fixed_var == 'P' and fixed_var_val > 1e-2:
                    Tsat, VL_liq, VL_gas, VL_iter, VL_err = self.flash_PVF(fixed_var_val, zs=zs, VF=.5)
            elif fixed_var == 'V':
                raise NotImplementedError("Does not make sense here because there is no actual vapor frac spec")

#                VL_flash = self.flash(P=P, VF=.4)
#            print('hade it', VL_liq, VL_gas)
            spec_val_l = getattr(VL_liq, spec)()
            spec_val_g = getattr(VL_gas, spec)()
#                spec_val_l = getattr(VL_flash.liquid0, spec)()
#                spec_val_g = getattr(VL_flash.gas, spec)()
            VF = (spec_val - spec_val_l)/(spec_val_g - spec_val_l)
            if 0.0 <= VF <= 1.0:
                G_l = fun(VL_liq)
                G_g = fun(VL_gas)
                G_VL = G_g*VF + G_l*(1.0 - VF)
            else:
                VF = None
        except Exception as e:
#            print(e, spec)
            VF = None

        try:
            G_SF = 1e100
            if self.solid_count and (self.gas_count or self.liquid_count):
                VS_flash = self.flash(SF=.5, **{fixed_var: fixed_var_val})
#                VS_flash = self.flash(P=P, SF=1)
                spec_val_s = getattr(VS_flash.solid0, spec)()
                spec_other = getattr(VS_flash.phases[0], spec)()
                SF = (spec_val - spec_val_s)/(spec_other - spec_val_s)
                if SF < 0.0 or SF > 1.0:
                    raise ValueError("Not apply")
                else:
                    G_other = fun(VS_flash.phases[0])
                    G_s = fun(VS_flash.solid0)
                    G_SF = G_s*SF + G_other*(1.0 - SF)
            else:
                SF = None
        except:
            SF = None

        gas_phase = None
        ls = []
        ss = []
        betas = []

        # If a 1-phase solution arrose, set it
        if results_G_min_1P is not None:
            betas = [1.0]
            T, phase, iterations, err, _ = results_G_min_1P
            if phase.is_gas:
                gas_phase = results_G_min_1P[1]
            elif phase.is_liquid:
                ls = [results_G_min_1P[1]]
            elif phase.is_solid:
                ss = [results_G_min_1P[1]]

        flash_convergence = {}
        if G_VL < G_min:
            skip_VL = False

#            if fixed_var == 'P' and spec == 'S' and fixed_var_val < 1.0 and 0:
#                skip_VL = True

            if not skip_VL:
                G_min = G_VL
                ls = [VL_liq]
                gas_phase = VL_gas
                betas = [VF, 1.0 - VF]
                ss = [] # Ensure solid unset
                T = VL_liq.T
                iterations = 0
                err = 0.0
                flash_convergence['VF flash convergence'] = {'iterations': VL_iter, 'err': VL_err}

        if G_SF < G_min:
            try:
                ls = [SF_flash.liquid0]
                gas_phase = None
            except:
                ls = []
                gas_phase = SF_flash.gas
            ss = [SF_flash.solid0]
            betas = [1.0 - SF, SF]
            T = SF_flash.T
            iterations = 0
            err = 0.0
            flash_convergence['SF flash convergence'] = SF_flash.flash_convergence

        if G_min == 1e100:
            '''Calculate the values of val at minimum and maximum temperature
            for each phase.
            Calculate the val at the phase changes.
            Include all in the exception to prove within bounds;
            also have a self check to say whether or not the value should have
            had a converged value.
            '''
            if iter_var == 'T':
                min_bound = Phase.T_MIN_FIXED*(1.0-1e-15)
                max_bound = Phase.T_MAX_FIXED*(1.0+1e-15)
            elif iter_var == 'P':
                min_bound = Phase.P_MIN_FIXED*(1.0-1e-15)
                max_bound = Phase.P_MAX_FIXED*(1.0+1e-15)
            elif iter_var == 'V':
                min_bound = Phase.V_MIN_FIXED*(1.0-1e-15)
                max_bound = Phase.V_MAX_FIXED*(1.0+1e-15)

            phases_at_min = []
            phases_at_max = []

#            specs_at_min = []
#            specs_at_max = []

            had_solution = False
            uncertain_solution = False

            s = ''
            phase_kwargs = {fixed_var: fixed_var_val, 'zs': zs}
            for phase in self.phases:

                try:
                    phase_kwargs[iter_var] = min_bound
                    p = phase.to(**phase_kwargs)
                    phases_at_min.append(p)

                    phase_kwargs[iter_var] = max_bound
                    p = phase.to(**phase_kwargs)
                    phases_at_max.append(p)

                    low, high = getattr(phases_at_min[-1], spec)(), getattr(phases_at_max[-1], spec)()
                    low, high = min(low, high), max(low, high)
                    s += '%s 1 Phase solution: (%g, %g); ' %(p.__class__.__name__, low, high)
                    if low <= spec_val <= high:
                        had_solution = True
                except:
                    uncertain_solution = True

            if VL_liq is not None:
                s += '(%s, %s) VL 2 Phase solution: (%g, %g); ' %(
                        VL_liq.__class__.__name__, VL_gas.__class__.__name__,
                        spec_val_l, spec_val_g)
                VL_min_spec, VL_max_spec = min(spec_val_l, spec_val_g), max(spec_val_l, spec_val_g),
                if VL_min_spec <= spec_val <= VL_max_spec:
                    had_solution = True
            if SF is not None:
                s += '(%s, %s) VL 2 Phase solution: (%g, %g); ' %(
                        VS_flash.phases[0].__class__.__name__, VS_flash.solid0.__class__.__name__,
                        spec_val_s, spec_other)
                S_min_spec, S_max_spec = min(spec_val_s, spec_other), max(spec_val_s, spec_other),
                if S_min_spec <= spec_val <= S_max_spec:
                    had_solution = True
            if had_solution:
                raise UnconvergedError("Could not converge but solution detected in bounds: %s" %s)
            elif uncertain_solution:
                raise UnconvergedError("Could not converge and unable to detect if solution detected in bounds")
            else:
                raise NoSolutionError("No physical solution in bounds for %s=%s at %s=%s: %s" %(spec, spec_val, fixed_var, fixed_var_val, s))

        flash_convergence['iterations'] = iterations
        flash_convergence['err'] = err

        return gas_phase, ls, ss, betas, flash_convergence

    def compare_flashes(self, state, inputs=None):
        # do a PT
        PT = self.flash(T=state.T, P=state.P)

        if inputs is None:
            inputs = [('T', 'P'),
                                 ('T', 'V'),
                                 ('P', 'V'),

                                 ('T', 'H'),
                                 ('T', 'S'),
                                 ('T', 'U'),

                                 ('P', 'H'),
                                 ('P', 'S'),
                                 ('P', 'U'),

                                 ('V', 'H'),
                                 ('V', 'S'),
                                 ('V', 'U')]

        states = []
        for p0, p1 in inputs:
            kwargs = {}

            p0_spec = getattr(state, p0)
            try:
                p0_spec = p0_spec()
            except:
                pass
            p1_spec = getattr(state, p1)
            try:
                p1_spec = p1_spec()
            except:
                pass
            kwargs = {}
            kwargs[p0] = p0_spec
            kwargs[p1] = p1_spec
            new = self.flash(**kwargs)
            states.append(new)
        return states

    def assert_flashes_same(self, reference, states, props=['T', 'P', 'V', 'S', 'H', 'G', 'U', 'A'], rtol=1e-7):
        ref_props = [reference.value(k) for k in props]
        for i, k in enumerate(props):
            ref = ref_props[i]
            for s in states:
                assert_close(s.value(k), ref, rtol=rtol)

    def generate_VF_data(self, Pmin=None, Pmax=None, pts=100,
                         props=['T', 'P', 'V', 'S', 'H', 'G', 'U', 'A']):
        '''Could use some better algorithms for generating better data? Some of
        the solutions count on this.
        '''
        Pc = self.constants.Pcs[0]
        if Pmax is None:
            Pmax = Pc
        if Pmin is None:
            Pmin = 1e-2
        if self.VL_only_CoolProp:
            AS = self.gas.AS
            Pmin = AS.trivial_keyed_output(CPiP_min)*(1.0 + 1e-3)
            Pmax = AS.p_critical()*(1.0 - 1e-7)

        Tmin, liquid, gas, iters, flash_err = self.flash_PVF(P=Pmin, VF=.5, zs=[1.0])
        Tmax, liquid, gas, iters, flash_err = self.flash_PVF(P=Pmax, VF=.5, zs=[1.0])

        liq_props, gas_props = [[] for _ in range(len(props))], [[] for _ in range(len(props))]
        # Lots of issues near Tc - split the range into low T and high T
        T_mid = 0.1*Tmin + 0.95*Tmax
        T_next = 0.045*Tmin + 0.955*Tmax

        Ts = linspace(Tmin, T_mid, pts//2)
        Ts += linspace(T_next, Tmax, pts//2)
        Ts.insert(-1, Tmax*(1-1e-8))
        Ts.sort()
        for T in Ts:
            Psat, liquid, gas, iters, flash_err = self.flash_TVF(T, VF=.5, zs=[1.0])
            for i, prop in enumerate(props):
                liq_props[i].append(liquid.value(prop))
                gas_props[i].append(gas.value(prop))

        return liq_props, gas_props

    def build_VF_interpolators(self, T_base=True, P_base=True, pts=50):
        self.liq_VF_interpolators = liq_VF_interpolators = {}
        self.gas_VF_interpolators = gas_VF_interpolators = {}
        props = ['T', 'P', 'V', 'S', 'H', 'G', 'U', 'A',
                 'dS_dT', 'dH_dT', 'dG_dT', 'dU_dT', 'dA_dT',
                 'dS_dP', 'dH_dP', 'dG_dP', 'dU_dP', 'dA_dP',
                 'fugacity', 'dfugacity_dT', 'dfugacity_dP']

        liq_props, gas_props = self.generate_VF_data(props=props, pts=pts)
        self.liq_VF_data = liq_props
        self.gas_VF_data = gas_props
        self.props_VF_data = props

        if T_base and P_base:
            base_props, base_idxs = ('T', 'P'), (0, 1)
        elif T_base:
            base_props, base_idxs = ('T',), (0,)
        elif P_base:
            base_props, base_idxs = ('P',), (1,)

        self.VF_data_base_props = base_props
        self.VF_data_base_idxs = base_idxs

        self.VF_data_spline_kwargs = spline_kwargs = dict(bc_type='natural', extrapolate=False)

        try:
            self.build_VF_splines()
        except:
            pass

    def build_VF_splines(self):
        self.VF_interpolators_built = True
        props = self.props_VF_data
        liq_props, gas_props = self.liq_VF_data, self.gas_VF_data
        VF_data_spline_kwargs = self.VF_data_spline_kwargs
        liq_VF_interpolators = self.liq_VF_interpolators
        gas_VF_interpolators = self.gas_VF_interpolators
        from scipy.interpolate import CubicSpline

        for base_prop, base_idx in zip(self.VF_data_base_props, self.VF_data_base_idxs):
            xs = liq_props[base_idx]
            for i, k in enumerate(props):
                if i == base_idx:
                    continue
                try:
                    spline = CubicSpline(xs, liq_props[i], **VF_data_spline_kwargs)
                    liq_VF_interpolators[(base_prop, k)] = spline
                except:
                    pass

                try:
                    spline = CubicSpline(xs, gas_props[i], **VF_data_spline_kwargs)
                    gas_VF_interpolators[(base_prop, k)] = spline
                except:
                    pass



    def flash_VF_HSGUA(self, fixed_var_val, spec_val, fixed_var='VF', spec_var='H', zs=None,
                       hot_start=None, solution='high'):
        # solution at high T by default
        if not self.VF_interpolators_built:
            self.build_VF_interpolators()
        iter_var = 'T' # hardcoded -
        # to make code generic try not to use eos stuff
#        liq_obj = self.liq_VF_interpolators[(iter_var, spec_var)]
#        gas_obj = self.liq_VF_interpolators[(iter_var, spec_var)]
        # iter_var must always be T
        VF = fixed_var_val
        props = self.props_VF_data
        liq_props = self.liq_VF_data
        gas_props = self.gas_VF_data
        iter_idx = props.index(iter_var)
        spec_idx = props.index(spec_var)

        T_idx, P_idx = props.index('T'), props.index('P')
        Ts, Ps = liq_props[T_idx], liq_props[P_idx]

        dfug_dT_idx = props.index('dfugacity_dT')
        dfug_dP_idx = props.index('dfugacity_dP')

        dspec_dT_var = 'd%s_dT' %(spec_var)
        dspec_dP_var = 'd%s_dP' %(spec_var)
        dspec_dT_idx = props.index(dspec_dT_var)
        dspec_dP_idx = props.index(dspec_dP_var)

        bounding_idx, bounding_Ts = [], []

        spec_values = []
        dspec_values = []

        d_sign_changes = False
        d_sign_changes_idx = []

        for i in range(len(liq_props[0])):
            v = liq_props[spec_idx][i]*(1.0 - VF) + gas_props[spec_idx][i]*VF

            dfg_T, dfl_T = gas_props[dfug_dT_idx][i], liq_props[dfug_dT_idx][i]
            dfg_P, dfl_P = gas_props[dfug_dP_idx][i], liq_props[dfug_dP_idx][i]
            at_critical = False
            try:
                dPsat_dT = (dfg_T - dfl_T)/(dfl_P - dfg_P)
            except ZeroDivisionError:
                at_critical = True
                dPsat_dT = self.constants.Pcs[0] #

            dv_g = dPsat_dT*gas_props[dspec_dP_idx][i] + gas_props[dspec_dT_idx][i]
            dv_l = dPsat_dT*liq_props[dspec_dP_idx][i] + liq_props[dspec_dT_idx][i]
            dv = dv_l*(1.0 - VF) + dv_g*VF
            if at_critical:
                dv = dspec_values[-1]

            if i > 0:
                if ((v <= spec_val <= spec_values[-1]) or (spec_values[-1] <= spec_val <= v)):
                    bounding_idx.append((i-1, i))
                    bounding_Ts.append((Ts[i-1], Ts[i]))

                if dv*dspec_values[-1] < 0.0:
                    d_sign_changes = True
                    d_sign_changes_idx.append((i-1, i))

            spec_values.append(v)
            dspec_values.append(dv)

        # if len(bounding_idx) < 2 and d_sign_changes:
        # Might not be in the range where there are multiple solutions
        #     raise ValueError("Derivative sign changes but only found one bounding value")


        # if len(bounding_idx) == 1:
        if len(bounding_idx) == 1 and (not d_sign_changes or (bounding_idx != d_sign_changes_idx and 1)):
            # Not sure about condition
            # Go right for the root
            T_low, T_high = bounding_Ts[0][0], bounding_Ts[0][1]
            idx_low, idx_high = bounding_idx[0][0], bounding_idx[0][1]

            val_low, val_high = spec_values[idx_low], spec_values[idx_high]
            dval_low, dval_high = dspec_values[idx_low], dspec_values[idx_high]
        # elif len(bounding_idx) == 0 and d_sign_changes:
            # root must be in interval derivative changes: Go right for the root
            # idx_low, idx_high = d_sign_changes_idx[0][0], d_sign_changes_idx[0][1]
            # T_low, T_high = Ts[idx_low], Ts[idx_high]
            #
            # val_low, val_high = spec_values[idx_low], spec_values[idx_high]
            # dval_low, dval_high = dspec_values[idx_low], dspec_values[idx_high]
        elif len(bounding_idx) == 2:
            # pick range and go for it
            if solution == 'high' or solution is None:
                T_low, T_high = bounding_Ts[1][0], bounding_Ts[1][1]
                idx_low, idx_high = bounding_idx[1][0], bounding_idx[1][1]
            else:
                T_low, T_high = bounding_Ts[0][0], bounding_Ts[0][1]
                idx_low, idx_high = bounding_idx[0][0], bounding_idx[0][1]

            val_low, val_high = spec_values[idx_low], spec_values[idx_high]
            dval_low, dval_high = dspec_values[idx_low], dspec_values[idx_high]

        elif (len(bounding_idx) == 1 and d_sign_changes) or (len(bounding_idx) == 0 and d_sign_changes):
            # Gotta find where derivative root changes, then decide if we have two solutions or just one; decide which to pursue
            idx_low, idx_high = d_sign_changes_idx[0][0], d_sign_changes_idx[0][1]
            T_low, T_high = Ts[idx_low], Ts[idx_high]

            T_guess = 0.5*(T_low +T_high)
            T_der_zero, v_zero = self._VF_HSGUA_der_root(T_guess, T_low, T_high, fixed_var_val, spec_val, fixed_var=fixed_var,
                                        spec_var=spec_var)
            high, low = False, False
            if (v_zero < spec_val < spec_values[idx_high]) or (spec_values[idx_high] < spec_val < v_zero):
                high = True
            if (spec_values[idx_low] < spec_val < v_zero) or (v_zero < spec_val < spec_values[idx_low]):
                low = True
            if not low and not high:
                # There was no other solution where the derivative changed
                T_low, T_high = bounding_Ts[0][0], bounding_Ts[0][1]
                idx_low, idx_high = bounding_idx[0][0], bounding_idx[0][1]

                val_low, val_high = spec_values[idx_low], spec_values[idx_high]
                dval_low, dval_high = dspec_values[idx_low], dspec_values[idx_high]
            elif (high and solution == 'high') or not low:
                val_low, val_high = v_zero, spec_values[idx_high]
                dval_low, dval_high = dspec_values[idx_high], dspec_values[idx_high]
                T_low, T_high = T_der_zero, Ts[idx_high]
            else:
                val_low, val_high = spec_values[idx_low], v_zero
                dval_low, dval_high = dspec_values[idx_low], dspec_values[idx_low]
                T_low, T_high = Ts[idx_low], T_der_zero
        elif len(bounding_idx) >2:
            # Entropy plot has 3 solutions, two derivative changes - give up by that point
            if isinstance(solution, int):
                sln_idx = solution
            else:
                sln_idx = {'high': -1, 'mid': -2, 'low': 0}[solution]
            T_low, T_high = bounding_Ts[sln_idx][0], bounding_Ts[sln_idx][1]
            idx_low, idx_high = bounding_idx[sln_idx][0], bounding_idx[sln_idx][1]

            val_low, val_high = spec_values[idx_low], spec_values[idx_high]
            dval_low, dval_high = dspec_values[idx_low], dspec_values[idx_high]

        else:
            raise ValueError("What")

        T_guess_low  = T_low - (val_low - spec_val)/dval_low
        T_guess_high  = T_high - (val_high - spec_val)/dval_high

        if T_low < T_guess_low < T_high and T_low < T_guess_high < T_high:
            T_guess = 0.5*(T_guess_low + T_guess_high)
        else:
            T_guess = 0.5*(T_low + T_high)

        return self.flash_VF_HSGUA_bounded(T_guess, T_low, T_high, fixed_var_val, spec_val, fixed_var=fixed_var, spec_var=spec_var)

    def _VF_HSGUA_der_root(self, guess, low, high, fixed_var_val, spec_val, fixed_var='VF', spec_var='H'):
        dspec_dT_var = 'd%s_dT' % (spec_var)
        dspec_dP_var = 'd%s_dP' % (spec_var)
        VF = fixed_var_val

        val_cache = [None, 0]

        def to_solve(T):
            Psat, liquid, gas, iters, flash_err = self.flash_TVF(T=T, VF=VF, zs=[1.0])
            # Error
            calc_spec_val = getattr(gas, spec_var)()*VF + getattr(liquid, spec_var)()*(1.0 - VF)
            val_cache[0] = calc_spec_val
            val_cache[1] += 1

            dfg_T, dfl_T = gas.dfugacity_dT(), liquid.dfugacity_dT()
            dfg_P, dfl_P = gas.dfugacity_dP(), liquid.dfugacity_dP()
            dPsat_dT = (dfg_T - dfl_T) / (dfl_P - dfg_P)

            dv_g = dPsat_dT*getattr(gas, dspec_dP_var)() + getattr(gas, dspec_dT_var)()
            dv_l = dPsat_dT*getattr(liquid, dspec_dP_var)() + getattr(liquid, dspec_dT_var)()
            dv = dv_l*(1.0 - VF) + dv_g*VF

            return dv

        # import matplotlib.pyplot as plt
        # xs = linspace(low, high, 1000)
        # ys = [to_solve(x) for x in xs]
        # plt.plot(xs, ys)
        # plt.show()
        try:
            T_zero = secant(to_solve, guess, low=low, high=high, xtol=1e-12, bisection=True)
        except:
            T_zero = brenth(to_solve, low, high, xtol=1e-12)
        return T_zero, val_cache[0]

    def flash_VF_HSGUA_bounded(self, guess, low, high, fixed_var_val, spec_val, fixed_var='VF', spec_var='H'):
        dspec_dT_var = 'd%s_dT' % (spec_var)
        dspec_dP_var = 'd%s_dP' % (spec_var)
        VF = fixed_var_val

        cache = [0]
        fprime = True
        def to_solve(T):
            Psat, liquid, gas, iters, flash_err = self.flash_TVF(T=T, VF=VF, zs=[1.0])
            # Error
            calc_spec_val = getattr(gas, spec_var)()*VF + getattr(liquid, spec_var)()*(1.0 - VF)
            err = calc_spec_val - spec_val
            cache[:] = [T, Psat, liquid, gas, iters, flash_err, err, cache[-1]+1]
            if not fprime:
                return err
            # Derivative
            dfg_T, dfl_T = gas.dfugacity_dT(), liquid.dfugacity_dT()
            dfg_P, dfl_P = gas.dfugacity_dP(), liquid.dfugacity_dP()
            dPsat_dT = (dfg_T - dfl_T) / (dfl_P - dfg_P)

            dv_g = dPsat_dT*getattr(gas, dspec_dP_var)() + getattr(gas, dspec_dT_var)()
            dv_l = dPsat_dT*getattr(liquid, dspec_dP_var)() + getattr(liquid, dspec_dT_var)()
            dv = dv_l*(1.0 - VF) + dv_g*VF

            return err, dv

        #
        try:
            T_calc = newton(to_solve, guess, fprime=True, low=low, high=high, xtol=1e-12, require_eval=True)
        except:
            # Zero division error in derivative mostly
            fprime = False
            T_calc = secant(to_solve, guess, low=low, high=high, xtol=1e-12, ytol=guess*1e-5, require_eval=True)


        return cache





    def debug_TVF(self, T, VF=None, pts=2000):
        zs = [1]
        gas = self.gas
        liquids = self.liquids

        def to_solve_newton(P):
            g = gas.to_TP_zs(T, P, zs)
            fugacity_gas = g.fugacities()[0]
            dfugacities_dP_gas = g.dfugacities_dP()[0]
            ls = [l.to_TP_zs(T, P, zs) for l in liquids]
            G_min, lowest_phase = 1e100, None
            for l in ls:
                G = l.G()
                if G < G_min:
                    G_min, lowest_phase = G, l

            fugacity_liq = lowest_phase.fugacities()[0]
            dfugacities_dP_liq = lowest_phase.dfugacities_dP()[0]

            err = fugacity_liq - fugacity_gas
            derr_dP = dfugacities_dP_liq - dfugacities_dP_gas
            return err, derr_dP

        import matplotlib.pyplot as plt
        import numpy as np

        Psat = self.correlations.VaporPressures[0](T)
        Ps = np.hstack([np.logspace(np.log10(Psat/2), np.log10(Psat*2), int(pts/2)),
                        np.logspace(np.log10(1e-6), np.log10(1e9), int(pts/2))])
        Ps = np.sort(Ps)
        values = np.array([to_solve_newton(P)[0] for P in Ps])
        values[values == 0] = 1e-10 # Make them show up on the plot

        plt.loglog(Ps, values, 'x', label='Positive errors')
        plt.loglog(Ps, -values, 'o', label='Negative errors')
        plt.legend(loc='best', fancybox=True, framealpha=0.5)
        plt.show()

    def debug_PVF(self, P, VF=None, pts=2000):
        zs = [1]
        gas = self.gas
        liquids = self.liquids

        def to_solve_newton(T):
            g = gas.to_TP_zs(T, P, zs)
            fugacity_gas = g.fugacities()[0]
            dfugacities_dT_gas = g.dfugacities_dT()[0]
            ls = [l.to_TP_zs(T, P, zs) for l in liquids]
            G_min, lowest_phase = 1e100, None
            for l in ls:
                G = l.G()
                if G < G_min:
                    G_min, lowest_phase = G, l

            fugacity_liq = lowest_phase.fugacities()[0]
            dfugacities_dT_liq = lowest_phase.dfugacities_dT()[0]

            err = fugacity_liq - fugacity_gas
            derr_dT = dfugacities_dT_liq - dfugacities_dT_gas
            return err, derr_dT

        import matplotlib.pyplot as plt
        Psat_obj = self.correlations.VaporPressures[0]

        Tsat = Psat_obj.solve_property(P)
        Tmax = Psat_obj.Tmax
        Tmin = Psat_obj.Tmin


        Ts = np.hstack([np.linspace(Tmin, Tmax, int(pts/4)),
                        np.linspace(Tsat-30, Tsat+30, int(pts/4))])
        Ts = np.sort(Ts)

        values = np.array([to_solve_newton(T)[0] for T in Ts])

        plt.semilogy(Ts, values, 'x', label='Positive errors')
        plt.semilogy(Ts, -values, 'o', label='Negative errors')


        min_index = np.argmin(np.abs(values))

        T = Ts[min_index]
        Ts2 = np.linspace(T*.999, T*1.001, int(pts/2))
        values2 = np.array([to_solve_newton(T)[0] for T in Ts2])
        plt.semilogy(Ts2, values2, 'x', label='Positive Fine')
        plt.semilogy(Ts2, -values2, 'o', label='Negative Fine')

        plt.legend(loc='best', fancybox=True, framealpha=0.5)
        plt.show()



    # ph - iterate on PT
    # if oscillating, take those two phases, solve, then get VF
    # other strategy - guess phase, solve h, PT at point to vonfirm!
    # For one phase - solve each phase for H, if there is a solution.
    # Take the one with lowest Gibbs energy

