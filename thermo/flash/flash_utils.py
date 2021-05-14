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
'''

from __future__ import division
__all__ = [
    'sequential_substitution_2P', 
    'sequential_substitution_2P_functional',
    'sequential_substitution_GDEM3_2P',
    'dew_bubble_Michelsen_Mollerup', 
    'bubble_T_Michelsen_Mollerup',
    'dew_T_Michelsen_Mollerup',
    'bubble_P_Michelsen_Mollerup',
    'dew_P_Michelsen_Mollerup',
    'minimize_gibbs_2P_transformed', 
    'sequential_substitution_Mehra_2P',
    'nonlin_2P', 
    'nonlin_n_2P', 
    'sequential_substitution_NP',
    'minimize_gibbs_NP_transformed', 
    'TPV_HSGUA_guesses_1P_methods', 
    'TPV_solve_HSGUA_guesses_1P',
    'sequential_substitution_2P_HSGUAbeta',
    'sequential_substitution_2P_sat', 
    'TP_solve_VF_guesses',
    'TPV_double_solve_1P', 
    'nonlin_2P_HSGUAbeta',
    'sequential_substitution_2P_double',
    'cm_flash_tol', 
    'nonlin_2P_newton',
    'dew_bubble_newton_zs',
    'existence_3P_Michelsen_Mollerup',
    'SS_VF_simultaneous',
    'stability_iteration_Michelsen',
    'assert_stab_success_2P', 
    'nonlin_equilibrium_NP',
    'nonlin_spec_NP',
    'TPV_solve_HSGUA_guesses_VL',
    'solve_P_VF_IG_K_composition_independent',
    'solve_T_VF_IG_K_composition_independent'
]


from fluids.constants import R
from fluids.numerics import (UnconvergedError, trunc_exp, newton,
                             brenth, secant, translate_bound_f_jac,
                             numpy as np, assert_close, assert_close1d,
                             damping_maintain_sign, oscillation_checking_wrapper,
                             OscillationError, NotBoundedError, jacobian,
                             best_bounding_bounds, isclose, newton_system,
                             make_damp_initial, newton_minimize,
                             root, minimize, fsolve)
from fluids.numerics import py_solve, trunc_log

from chemicals.utils import (exp, log, copysign, normalize,
                             mixing_simple, property_mass_to_molar)
from chemicals.heat_capacity import (Dadgostar_Shaw_integral, 
                                     Dadgostar_Shaw_integral_over_T, 
                                     Lastovka_Shaw_integral,
                                     Lastovka_Shaw_integral_over_T)
from chemicals.rachford_rice import (flash_inner_loop, 
                                     Rachford_Rice_solutionN,
                                     Rachford_Rice_flash_error, 
                                     Rachford_Rice_solution_LN2)
from chemicals.phase_change import SMK
from chemicals.volume import COSTALD
from chemicals.flash_basic import flash_wilson, flash_Tb_Tc_Pc, flash_ideal
from chemicals.exceptions import TrivialSolutionError
from thermo.phases import Phase, CoolPropPhase, CEOSLiquid, CEOSGas, IAPWS95
from thermo.phases.phase_utils import lnphis_direct
from thermo.coolprop import CPiP_min

LASTOVKA_SHAW = 'Lastovka Shaw'
DADGOSTAR_SHAW_1 = 'Dadgostar Shaw 1'
STP_T_GUESS = '298.15 K'
LAST_CONVERGED = 'Last converged'
FIXED_GUESS = 'Fixed guess'
IG_ENTHALPY = 'Ideal gas'
IDEAL_LIQUID_ENTHALPY = 'Ideal liquid'
WILSON_GUESS = 'Wilson'
TB_TC_GUESS = 'Tb Tc'
IDEAL_PSAT = 'Ideal Psat'
PT_SS = 'SS'
PT_SS_MEHRA = 'SS Mehra'
PT_SS_GDEM3 = 'SS GDEM3'
PT_NEWTON_lNKVF = 'Newton lnK VF'
IDEAL_WILSON = 'Ideal Wilson'
SHAW_ELEMENTAL = 'Shaw Elemental'

PH_T_guesses_1P_methods = [LASTOVKA_SHAW, DADGOSTAR_SHAW_1, IG_ENTHALPY,
                           IDEAL_LIQUID_ENTHALPY, FIXED_GUESS, STP_T_GUESS,
                           LAST_CONVERGED]
TPV_HSGUA_guesses_1P_methods = PH_T_guesses_1P_methods

def sequential_substitution_2P(T, P, V, zs, xs_guess, ys_guess, liquid_phase,
                               gas_phase, maxiter=1000, tol=1E-13,
                               trivial_solution_tol=1e-5, V_over_F_guess=None,
                               check_G=False, check_V=False, dZ_allow=0.1):

    xs, ys = xs_guess, ys_guess
    if V_over_F_guess is None:
        V_over_F = 0.5
    else:
        V_over_F = V_over_F_guess

    cmps = range(len(zs))

    err, err1, err2, err3 = 0.0, 0.0, 0.0, 0.0
    G_old = None
    V_over_F_old = V_over_F
    restrained = 0
    restrained_switch_count = 300

    # Code for testing phis at zs
    l, g = liquid_phase, gas_phase
    if liquid_phase.T != T or liquid_phase.P != P:
        liquid_phase = liquid_phase.to_TP_zs(T=T, P=P, zs=xs)
    if gas_phase.T != T or gas_phase.P != P:
        gas_phase = gas_phase.to_TP_zs(T=T, P=P, zs=ys)

    for iteration in range(maxiter):
#        g = gas_phase.to_TP_zs(T=T, P=P, zs=ys)
#        l = liquid_phase.to_TP_zs(T=T, P=P, zs=xs)

#        l = liquid_phase.to(xs, T=T, P=P, V=V)
#        g = gas_phase.to(ys, T=T, P=P, V=V)
#        lnphis_g = g.lnphis()
#        lnphis_l = l.lnphis()
        lnphis_g = gas_phase.lnphis_at_zs(ys)
        lnphis_l = liquid_phase.lnphis_at_zs(xs)
        limited_Z = False

        try:
            Ks = [exp(lnphis_l[i] - lnphis_g[i]) for i in cmps] # K_value(phi_l=l, phi_g=g)
        except OverflowError:
            Ks = [trunc_exp(lnphis_l[i] - lnphis_g[i]) for i in cmps] # K_value(phi_l=l, phi_g=g)

        V_over_F_old = V_over_F
        try:
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F)
        except Exception as e:
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F, check=True)
#            K_low, K_high = False, False
#            for zi, Ki in zip(zs, Ks):
#                if zi != 0.0:
#                    if Ki > 1.0:
#                        K_high = True
#                    else:
#                        K_low = True
#                    if K_high and K_low:
#                        break
#            if not (K_high and K_low):
#                raise TrivialSolutionError("Converged to trivial condition, all K same phase",
#                                           comp_difference, iteration, err)
#            else:

        if check_G:
            V_over_F_G = min(max(V_over_F_old, 0), 1)
            G = g.G()*V_over_F_G + (1.0 - V_over_F_G)*l.G()
            print('new G', G, 'old G', G_old)
            if G_old is not None:
                if G > G_old:
                    step = .5
                    while G > G_old and step > 1e-4:
#                        ys_working = normalize([step*xo + (1.0 - step)*xi for xi, xo in zip(xs, xs_old)])
#                        xs_working = normalize([step*xo + (1.0 - step)*xi for xi, xo in zip(ys, ys_old)])
#                         ys_working = normalize([step*xo + (1.0 - step)*xi for xo, xi in zip(xs, xs_old)])
#                         xs_working = normalize([step*xo + (1.0 - step)*xi for xo, xi in zip(ys, ys_old)])
#                         g = gas_phase.to(ys_working, T=T, P=P, V=V)
#                         l = liquid_phase.to(xs_working, T=T, P=P, V=V)
#                         lnphis_g = g.lnphis()
#                         lnphis_l = l.lnphis()
#                         try:
#                             Ks = [exp(lnphis_l[i] - lnphis_g[i]) for i in cmps]
#                         except OverflowError:
#                             Ks = [trunc_exp(lnphis_l[i] - lnphis_g[i]) for i in cmps]
                        Ks_working = [step*xo + (1.0 - step)*xi for xo, xi in zip(Ks_old, Ks)]

                        V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks_working, guess=V_over_F)
#                        V_over_F_G = min(max(V_over_F, 0), 1)
                        g = gas_phase.to(ys_new, T=T, P=P, V=V)
                        l = liquid_phase.to(xs_new, T=T, P=P, V=V)
                        G = g.G()*V_over_F_G + (1.0 - V_over_F_G)*l.G()
                        print('step', step, G, V_over_F, Ks)
                        step *= 0.5
                    # xs, ys = xs_working, ys_working


#                    print('Gibbs increased', G/G_old)
            G_old = G
        if check_V and iteration > 2:
            big_Z_change = (abs(1.0 - l_old.Z()/l.Z()) > dZ_allow or abs(1.0 - g_old.Z()/g.Z()) > dZ_allow)
            if restrained <= restrained_switch_count and big_Z_change:
                limited_Z = True
                step = .5 #.5
                while (abs(1.0 - l_old.Z()/l.Z()) > dZ_allow  or abs(1.0 - g_old.Z()/g.Z()) > dZ_allow ) and step > 1e-8:
                    # Ks_working = [step*xo + (1.0 - step)*xi for xo, xi in zip(Ks, Ks_old)]
#                     Ks_working = [Ks[i]*(Ks_old[i]/Ks[i])**(1.0 - step) for i in cmps] # step = 0 - all new; step = 1 - all old
#                     Ks_working = [Ks_old[i]*(exp(lnphis_l[i])/exp(lnphis_g[i])/Ks_old[i])**(1.0 - step) for i in cmps]
                    ys_new = normalize([step*xo + (1.0 - step)*xi for xo, xi in zip(ys, ys_old)])
                    xs_new = normalize([step*xo + (1.0 - step)*xi for xo, xi in zip(xs, xs_old)])
                    # V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks_working, guess=V_over_F)
                    l = liquid_phase.to(xs_new, T=T, P=P, V=V)
                    g = gas_phase.to(ys_new, T=T, P=P, V=V)
                    # lnphis_g = g.lnphis()
                    # lnphis_l = l.lnphis()
                    print('step', step, V_over_F, g.Z())
                    step *= 0.5
                xs, ys = xs_new, ys_new
                lnphis_g = g.lnphis()
                lnphis_l = l.lnphis()
                Ks = [exp(lnphis_l[i] - lnphis_g[i]) for i in cmps]
                V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F)
                restrained += 1
            elif restrained > restrained_switch_count and big_Z_change:
                restrained = 0

        # Check for negative fractions - normalize only if needed
        for xi in xs_new:
            if xi < 0.0:
                xs_new_sum_inv = 1.0/sum(abs(i) for i in xs_new)
                for i in cmps:
                    xs_new[i] = abs(xs_new[i])*xs_new_sum_inv
                break
        for yi in ys_new:
            if yi < 0.0:
                ys_new_sum_inv = 1.0/sum(abs(i) for i in ys_new)
                for i in cmps:
                    ys_new[i] = abs(ys_new[i])*ys_new_sum_inv
                break

        # Calculate the error using the new Ks and old compositions
        # Claimed error function in CONVENTIONAL AND RAPID FLASH
        # CALCULATIONS FOR THE SOAVE-REDLICH-KWONG AND PENG-ROBINSON EQUATIONS OF STATE

        err = 0.0
        # Suggested tolerance 1e-15
        try:
            for Ki, xi, yi in zip(Ks, xs, ys):
                # equivalent of fugacity ratio
                # Could divide by the old Ks as well.
                err_i = Ki*xi/yi - 1.0
                err += err_i*err_i
        except ZeroDivisionError:
            err = 0.0
            for Ki, xi, yi in zip(Ks, xs, ys):
                try:
                    err_i = Ki*xi/yi - 1.0
                    err += err_i*err_i
                except ZeroDivisionError:
                    pass

        if err > 0.0 and err in (err1, err2, err3):
            raise OscillationError("Converged to cycle in errors, no progress being made")
        # Accept the new compositions
        xs_old, ys_old, Ks_old = xs, ys, Ks
        # if not limited_Z:
        #     assert xs == l.zs
        #     assert ys == g.zs
        xs, ys = xs_new, ys_new
        lnphis_g_old, lnphis_l_old = lnphis_g, lnphis_l
        l_old, g_old = l, g

#        print(err, V_over_F, Ks) # xs, ys

        # Check for
        comp_difference = sum([abs(xi - yi) for xi, yi in zip(xs, ys)])
        if comp_difference < trivial_solution_tol:
            raise TrivialSolutionError("Converged to trivial condition, compositions of both phases equal",
                                       comp_difference, iteration, err)
        if err < tol and not limited_Z:
            # Temporary!
            # err_mole_balance = 0.0
            # for i in cmps:
            #     err_mole_balance += abs(xs_old[i] * (1.0 - V_over_F_old) + ys_old[i] * V_over_F_old - zs[i])
            # if err_mole_balance < mole_balance_tol:

                # return V_over_F, xs, ys, l, g, iteration, err

            if iteration == 0:
                # We are composition independent!
                g = gas_phase.to(ys_new, T=T, P=P, V=V)
                l = liquid_phase.to(xs_new, T=T, P=P, V=V)
                return V_over_F, xs_new, ys_new, l, g, iteration, err
            else:
                g = gas_phase.to(ys_old, T=T, P=P, V=V)
                l = liquid_phase.to(xs_old, T=T, P=P, V=V)
                return V_over_F_old, xs_old, ys_old, l, g, iteration, err
        # elif err < tol and limited_Z:
        #     print(l.fugacities()/np.array(g.fugacities()))
        err1, err2, err3 = err, err1, err2
    raise UnconvergedError('End of SS without convergence')

def sequential_substitution_2P_functional(zs, xs_guess, ys_guess,
                               liquid_args, gas_args, maxiter=1000, tol=1E-13,
                               trivial_solution_tol=1e-5, V_over_F_guess=0.5):

    xs, ys = xs_guess, ys_guess
    V_over_F = V_over_F_guess
    N = len(zs)

    err = 0.0
    V_over_F_old = V_over_F
    
    Ks = [0.0]*N
    for iteration in range(maxiter):
        lnphis_g = lnphis_direct(ys, *gas_args)
        lnphis_l = lnphis_direct(xs, *liquid_args)

        for i in range(N):
            Ks[i] = exp(lnphis_l[i] - lnphis_g[i])

        V_over_F_old = V_over_F
        try:
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F)
        except:
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F, check=True)

        for xi in xs_new:
            if xi < 0.0:
                # Remove negative mole fractions - may help or may still fail
                xs_new_sum_inv = 0.0
                for xj in xs_new:
                    xs_new_sum_inv += abs(xj)
                xs_new_sum_inv = 1.0/xs_new_sum_inv
                for i in range(N):
                    xs_new[i] = abs(xs_new[i])*xs_new_sum_inv
                break
        for yi in ys_new:
            if yi < 0.0:
                ys_new_sum_inv = 0.0
                for yj in ys_new:
                    ys_new_sum_inv += abs(yj)
                ys_new_sum_inv = 1.0/ys_new_sum_inv
                for i in range(N):
                    ys_new[i] = abs(ys_new[i])*ys_new_sum_inv
                break

        err = 0.0
        for Ki, xi, yi in zip(Ks, xs, ys):
            # equivalent of fugacity ratio
            # Could divide by the old Ks as well.
            err_i = Ki*xi/yi - 1.0
            err += err_i*err_i

        xs_old, ys_old = xs, ys
        xs, ys = xs_new, ys_new

        comp_difference = 0.0
        for xi, yi in zip(xs, ys):
            comp_difference += abs(xi - yi)

        if comp_difference < trivial_solution_tol:
            raise ValueError("Converged to trivial condition, compositions of both phases equal")

        if err < tol:
            return V_over_F_old, xs_old, ys_old, iteration, err
    raise ValueError('End of SS without convergence')


def sequential_substitution_NP(T, P, zs, compositions_guesses, betas_guesses,
                               phases, maxiter=1000, tol=1E-13,
                               trivial_solution_tol=1e-5, ref_phase=2):

    compositions = compositions_guesses
    cmps = range(len(zs))
    phase_count = len(phases)
    phases_iter = range(phase_count)
    phase_iter_n1 = range(phase_count - 1)
    betas = betas_guesses
    if len(betas) < len(phases):
        betas.append(1.0 - sum(betas))

    compositions_K_order = [compositions[i] for i in phases_iter if i != ref_phase]
    compositions_ref = compositions_guesses[ref_phase]

    for iteration in range(maxiter):
        phases = [phases[i].to_TP_zs(T=T, P=P, zs=compositions[i]) for i in phases_iter]
        lnphis = [phases[i].lnphis() for i in phases_iter]

        Ks = []
        lnphis_ref = lnphis[ref_phase]
        for i in phases_iter:
            if i != ref_phase:
                lnphis_i = lnphis[i]
                try:
                    Ks.append([exp(lnphis_ref[j] - lnphis_i[j]) for j in cmps])
                except OverflowError:
                    Ks.append([trunc_exp(lnphis_ref[j] - lnphis_i[j]) for j in cmps])


        beta_guesses = [betas[i] for i in phases_iter if i != ref_phase]

        #if phase_count == 3:
        #    Rachford_Rice_solution2(zs, Ks[0], Ks[1], beta_y=beta_guesses[0], beta_z=beta_guesses[1])
        betas_new, compositions_new = Rachford_Rice_solutionN(zs, Ks, beta_guesses)
        # Sort the order back
        beta_ref_new = betas_new[-1]
        betas_new = betas_new[:-1]
        betas_new.insert(ref_phase, beta_ref_new)

        compositions_ref_new = compositions_new[-1]
        compositions_K_order_new = compositions_new[:-1]

        compositions_new = list(compositions_K_order_new)
        compositions_new.insert(ref_phase, compositions_ref_new)

        err = 0.0
        for i in phase_iter_n1:
            Ks_i = Ks[i]
            ys = compositions_K_order[i]
            try:
                for Ki, xi, yi in zip(Ks_i, compositions_ref, ys):
                    err_i = Ki*xi/yi - 1.0
                    err += err_i*err_i
            except ZeroDivisionError:
                err = 0.0
                for Ki, xi, yi in zip(Ks_i, compositions_ref, ys):
                    try:
                        err_i = Ki*xi/yi - 1.0
                        err += err_i*err_i
                    except ZeroDivisionError:
                        pass
#        print(betas, Ks, 'calculated', err)
        # print(err)

        compositions = compositions_new
        compositions_K_order = compositions_K_order_new
        compositions_ref = compositions_ref_new
        betas = betas_new

        # TODO trivial solution check - how to handle - drop phase?

        # Check for
#        comp_difference = sum([abs(xi - yi) for xi, yi in zip(xs, ys)])
#        if comp_difference < trivial_solution_tol:
#            raise ValueError("Converged to trivial condition, compositions of both phases equal")
        if err < tol:
            return betas, compositions, phases, iteration, err
        # if iteration > 100:
        #     return betas, compositions, phases, iteration, err
    raise UnconvergedError('End of SS without convergence')




def sequential_substitution_Mehra_2P(T, P, zs, xs_guess, ys_guess, liquid_phase,
                                     gas_phase, maxiter=1000, tol=1E-13,
                                     trivial_solution_tol=1e-5,
                                     acc_frequency=3, acc_delay=5,
                                     lambda_max=3, lambda_min=0.0,
                                     V_over_F_guess=None):

    xs, ys = xs_guess, ys_guess
    if V_over_F_guess is None:
        V_over_F = 0.5
    else:
        V_over_F = V_over_F_guess

    N = len(zs)
    cmps = range(N)
    lambdas = [1.0]*N

    Ks = [ys[i]/xs[i] for i in cmps]

    gs = []
    import numpy as np
    for iteration in range(maxiter):
        g = gas_phase.to_TP_zs(T=T, P=P, zs=ys)
        l = liquid_phase.to_TP_zs(T=T, P=P, zs=xs)

        fugacities_g = g.fugacities()
        fugacities_l = l.fugacities()
#        Ks = [fugacities_l[i]*ys[i]/(fugacities_g[i]*xs[i]) for i in cmps]
        lnphis_g = g.lnphis()
        lnphis_l = l.lnphis()
        phis_g = g.phis()
        phis_l = l.phis()
#        Ks = [Ks[i]*exp(-lnphis_g[i]/lnphis_l[i]) for i in cmps]
#        Ks = [Ks[i]*(phis_l[i]/phis_g[i]/Ks[i])**lambdas[i] for i in cmps]
#        Ks = [Ks[i]*fugacities_l[i]/fugacities_g[i] for i in cmps]
#        Ks = [Ks[i]*exp(-phis_g[i]/phis_l[i]) for i in cmps]
        # Mehra, R. K., R. A. Heidemann, and K. Aziz. “An Accelerated Successive Substitution Algorithm.” The Canadian Journal of Chemical Engineering 61, no. 4 (August 1, 1983): 590-96. https://doi.org/10.1002/cjce.5450610414.

        # Strongly believed correct
        gis = np.log(fugacities_g) - np.log(fugacities_l)

        if not (iteration % acc_frequency) and iteration > acc_delay:
            gis_old = np.array(gs[-1])
#            lambdas = np.abs(gis_old.T*gis_old/(gis_old.T*(gis_old - gis))*lambdas).tolist() # Alrotithm 3 also working
#            lambdas = np.abs(gis_old.T*(gis_old-gis)/((gis_old-gis).T*(gis_old - gis))*lambdas).tolist() # WORKING
            lambdas = np.abs(gis.T*gis/(gis_old.T*(gis - gis_old))).tolist() # 34, working
            lambdas = [min(max(li, lambda_min), lambda_max) for li in lambdas]
#            print(lambdas[0:5])
            print(lambdas)
#            print('Ks', Ks, )
#            print(Ks[-1], phis_l[-1], phis_g[-1], lambdas[-1], gis[-1], gis_old[-1])
            Ks = [Ks[i]*(phis_l[i]/phis_g[i]/Ks[i])**lambdas[i] for i in cmps]
#            print(Ks)
        else:
            Ks = [Ks[i]*fugacities_l[i]/fugacities_g[i] for i in cmps]
#        print(Ks[0:5])


        gs.append(gis)

#            lnKs = [lnKs[i]*1.5 for i in cmps]

        V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F)

        # Check for negative fractions - normalize only if needed
        for xi in xs_new:
            if xi < 0.0:
                xs_new_sum = sum(abs(i) for i in xs_new)
                xs_new = [abs(i)/xs_new_sum for i in xs_new]
                break
        for yi in ys_new:
            if yi < 0.0:
                ys_new_sum = sum(abs(i) for i in ys_new)
                ys_new = [abs(i)/ys_new_sum for i in ys_new]
                break

        err = 0.0
        # Suggested tolerance 1e-15
        for Ki, xi, yi in zip(Ks, xs, ys):
            # equivalent of fugacity ratio
            # Could divide by the old Ks as well.
            err_i = Ki*xi/yi - 1.0
            err += err_i*err_i
        print(err)
        # Accept the new compositions
        xs, ys = xs_new, ys_new
        # Check for
        comp_difference = sum([abs(xi - yi) for xi, yi in zip(xs, ys)])
        if comp_difference < trivial_solution_tol:
            raise TrivialSolutionError("Converged to trivial condition, compositions of both phases equal",
                                       comp_difference, iteration, err)
        if err < tol:
            return V_over_F, xs, ys, l, g, iteration, err
    raise UnconvergedError('End of SS without convergence')



def sequential_substitution_GDEM3_2P(T, P, zs, xs_guess, ys_guess, liquid_phase,
                                     gas_phase, maxiter=1000, tol=1E-13,
                                     trivial_solution_tol=1e-5, V_over_F_guess=None,
                                     acc_frequency=3, acc_delay=3,
                                     ):

    xs, ys = xs_guess, ys_guess
    if V_over_F_guess is None:
        V_over_F = 0.5
    else:
        V_over_F = V_over_F_guess

    cmps = range(len(zs))
    all_Ks = []
    all_lnKs = []

    for iteration in range(maxiter):
        g = gas_phase.to_TP_zs(T=T, P=P, zs=ys)
        l = liquid_phase.to_TP_zs(T=T, P=P, zs=xs)

        lnphis_g = g.lnphis()
        lnphis_l = l.lnphis()

        # Mehra et al. (1983) is another option

#        Ks = [exp(l - g) for l, g in zip(lnphis_l, lnphis_g)]
#        if not (iteration %3) and iteration > 3:
#            dKs = gdem(Ks, all_Ks[-1], all_Ks[-2], all_Ks[-3])
#            print(iteration, dKs)
#            Ks = [Ks[i] + dKs[i] for i in cmps]
#        all_Ks.append(Ks)

#        lnKs = [(l - g) for l, g in zip(lnphis_l, lnphis_g)]
#        if not (iteration %3) and iteration > 3:
##            dlnKs = gdem(lnKs, all_lnKs[-1], all_lnKs[-2], all_lnKs[-3])
#
#            dlnKs = gdem(lnKs, all_lnKs[-1], all_lnKs[-2], all_lnKs[-3])
#            lnKs = [lnKs[i] + dlnKs[i] for i in cmps]

        # Mehra, R. K., R. A. Heidemann, and K. Aziz. “An Accelerated Successive Substitution Algorithm.” The Canadian Journal of Chemical Engineering 61, no. 4 (August 1, 1983): 590-96. https://doi.org/10.1002/cjce.5450610414.
        lnKs = [(l - g) for l, g in zip(lnphis_l, lnphis_g)]
        if not (iteration %acc_frequency) and iteration > acc_delay:
            dlnKs = gdem(lnKs, all_lnKs[-1], all_lnKs[-2], all_lnKs[-3])
            print(dlnKs)
            lnKs = [lnKs[i] + dlnKs[i] for i in cmps]


            # Try to testaccelerated
        all_lnKs.append(lnKs)
        Ks = [exp(lnKi) for lnKi in lnKs]

        V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F)

        # Check for negative fractions - normalize only if needed
        for xi in xs_new:
            if xi < 0.0:
                xs_new_sum = sum(abs(i) for i in xs_new)
                xs_new = [abs(i)/xs_new_sum for i in xs_new]
                break
        for yi in ys_new:
            if yi < 0.0:
                ys_new_sum = sum(abs(i) for i in ys_new)
                ys_new = [abs(i)/ys_new_sum for i in ys_new]
                break

        err = 0.0
        # Suggested tolerance 1e-15
        for Ki, xi, yi in zip(Ks, xs, ys):
            # equivalent of fugacity ratio
            # Could divide by the old Ks as well.
            err_i = Ki*xi/yi - 1.0
            err += err_i*err_i

        # Accept the new compositions
        xs, ys = xs_new, ys_new
        # Check for
        comp_difference = sum([abs(xi - yi) for xi, yi in zip(xs, ys)])
        if comp_difference < trivial_solution_tol:
            raise TrivialSolutionError("Converged to trivial condition, compositions of both phases equal",
                                       comp_difference, iteration, err)
        if err < tol:
            return V_over_F, xs, ys, l, g, iteration, err
    raise UnconvergedError('End of SS without convergence')


def nonlin_equilibrium_NP(T, P, zs, compositions_guesses, betas_guesses,
                        phases, maxiter=1000, tol=1E-13,
                        trivial_solution_tol=1e-5, ref_phase=-1,
                        method='hybr', solve_kwargs=None, debug=False):
    if solve_kwargs is None:
        solve_kwargs = {}

    compositions = compositions_guesses
    N = len(zs)
    Nm1 = N - 1
    cmps = range(N)
    phase_count = len(phases)
    phase_iter = range(phase_count)
    if ref_phase < 0:
        ref_phase = phase_count + ref_phase

    phase_iter_n1 = [i for i in phase_iter if i != ref_phase]
    phase_iter_n1_0 = range(phase_count-1)
    betas = betas_guesses
    if len(betas) < len(phases):
        betas.append(1.0 - sum(betas))

    flows_guess = [compositions_guesses[j][i]*betas[j] for j in phase_iter_n1 for i in cmps]

    jac = True
    if method in ('broyden1', 'broyden2', 'anderson', 'linearmixing',
                  'diagbroyden', 'excitingmixing', 'krylov'):
        jac = False




    global iterations, info
    iterations = 0
    info = []
    def to_solve(flows, jac=jac):
        global iterations, info
        try:
            flows = flows.tolist()
        except:
            flows = list(flows)
        iterations += 1
        iter_flows = []
        iter_comps = []
        iter_betas = []
        iter_phases = []
        jac_arr = None

        remaining = zs
        for i in range(len(flows)):
            if flows[i] < 0.0:
                flows[i] = 1e-100


        for j, k in zip(phase_iter_n1, phase_iter_n1_0):
            v = flows[k*N:k*N+N]
            vs = v
            vs_sum = sum(abs(i) for i in vs)
            if vs_sum == 0.0:
                # Handle the case an optimizer takes all of all compounds already
                ys = zs
            else:
                vs_sum_inv = 1.0/vs_sum
                ys = [abs(vs[i]*vs_sum_inv) for i in cmps]
                ys = normalize(ys)
            iter_flows.append(vs)
            iter_comps.append(ys)
            iter_betas.append(vs_sum) # Would be divided by feed but feed is zs = 1
            iter_phases.append(phases[j].to_TP_zs(T=T, P=P, zs=ys))
            remaining = [remaining[i] - vs[i] for i in cmps]

        flows_ref = remaining
        iter_flows.insert(ref_phase, remaining)

        beta_ref = sum(remaining)
        iter_betas.insert(ref_phase, beta_ref)

        xs_ref = normalize([abs(i) for i in remaining])
        iter_comps.insert(ref_phase, xs_ref)

        phase_ref = phases[ref_phase].to_TP_zs(T=T, P=P, zs=xs_ref)
        iter_phases.insert(ref_phase, phase_ref)

        lnphis_ref = phase_ref.lnphis()
        dlnfugacities_ref = phase_ref.dlnfugacities_dns()

        errs = []
        for k in phase_iter_n1:
            phase = iter_phases[k]
            lnphis = phase.lnphis()
            xs = iter_comps[k]
            for i in cmps:
                # This is identical to lnfugacity(i)^j - lnfugacity(i)^ref
                gi = trunc_log(xs[i]/xs_ref[i]) + lnphis[i] - lnphis_ref[i]
                errs.append(gi)

        if jac:
            jac_arr = [[0.0]*N*(phase_count-1) for i in range(N*(phase_count-1))]
            for ni, nj in zip(phase_iter_n1, phase_iter_n1_0):
                p = iter_phases[ni]
                dlnfugacities = p.dlnfugacities_dns()
                # Begin with the first row using ni, nj;
                for i in cmps:
                    for ki, kj in zip(phase_iter_n1, phase_iter_n1_0):
                        for j in cmps:
                            delta = 1.0 if nj == kj else 0.0
                            v_ref = dlnfugacities_ref[i][j]/beta_ref
                            jac_arr[nj*N + i][kj*N + j] = dlnfugacities[i][j]*delta/iter_betas[ni] + v_ref
        info[:] = iter_betas, iter_comps, iter_phases, errs, jac_arr, flows
        if jac:
            return errs, jac_arr
        return errs

    if method == 'newton_system':
        comp_val, iterations = newton_system(to_solve, flows_guess, jac=True,
                                             xtol=tol, damping=1,
                                             damping_func=damping_maintain_sign)
    else:
        def f_jac_numpy(flows_guess):
            # needed
            ans = to_solve(flows_guess)
            if jac:
                return np.array(ans[0]), np.array(ans[1])
            return np.array(ans)
        sln = root(f_jac_numpy, flows_guess, tol=tol, jac=(True if jac else None), method=method, **solve_kwargs)
        iterations = sln['nfev']

    betas, compositions, phases, errs, jac, flows = info
    sln = (betas, compositions, phases, errs, jac, iterations)
    if debug:
        return sln, flows, to_solve
    return sln


def nonlin_spec_NP(guess, fixed_val, spec_val, zs, compositions_guesses, betas_guesses,
                    phases, iter_var='T', fixed_var='P', spec='H',
                    maxiter=1000, tol=1E-13,
                    trivial_solution_tol=1e-5, ref_phase=-1,
#                    method='hybr',
                    method='fsolve',
                    solve_kwargs=None, debug=False,
                    analytical_jac=True):
    if solve_kwargs is None:
        solve_kwargs = {}

    phase_kwargs = {fixed_var: fixed_val, iter_var: guess}
    compositions = compositions_guesses
    N = len(zs)
    Nm1 = N - 1
    cmps = range(N)
    phase_count = len(phases)
    phase_iter = range(phase_count)
    if ref_phase < 0:
        ref_phase = phase_count + ref_phase

    phase_iter_n1 = [i for i in phase_iter if i != ref_phase]
    phase_iter_n1_0 = range(phase_count-1)
    betas = betas_guesses
    if len(betas) < len(phases):
        betas.append(1.0 - sum(betas))

    guesses = [compositions_guesses[j][i]*betas[j] for j in phase_iter_n1 for i in cmps]
    guesses.append(guess)
    spec_callables = [getattr(phase.__class__, spec) for phase in phases]

    dlnphis_diter_s = 'dlnphis_d' + iter_var
    dlnphis_diter_callables = [getattr(phase.__class__, dlnphis_diter_s) for phase in phases]

    dspec_diter_s = 'd%s_d%s' %(spec, iter_var)
    dspec_diter_callables = [getattr(phase.__class__, dspec_diter_s) for phase in phases]

    dspec_dn_s = 'd%s_dns' %(spec)
    dspec_dn_callables = [getattr(phase.__class__, dspec_dn_s) for phase in phases]

    jac = True
    if method in ('broyden1', 'broyden2', 'anderson', 'linearmixing',
                  'diagbroyden', 'excitingmixing', 'krylov', 'fsolve'):
        jac = False




    global iterations, info
    iterations = 0
    info = []
    def to_solve(flows, jac=jac, skip_err=False):
        global iterations, info
        try:
            flows = flows.tolist()
        except:
            flows = list(flows)
        iter_val = flows[-1]
        phase_kwargs[iter_var] = iter_val
        flows = flows[:-1]
        iter_flows = []
        iter_comps = []
        iter_betas = []
        iter_phases = []
        jac_arr = None

        remaining = zs
        if not skip_err:
#            print(flows, iter_val)
            iterations += 1
            for i in range(len(flows)):
                if flows[i] < 0.0:
                    flows[i] = 1e-100


            for j, k in zip(phase_iter_n1, phase_iter_n1_0):
                v = flows[k*N:k*N+N]
                vs = v
                vs_sum = sum(abs(i) for i in vs)
                if vs_sum == 0.0:
                    # Handle the case an optimizer takes all of all compounds already
                    ys = zs
                else:
                    vs_sum_inv = 1.0/vs_sum
                    ys = [abs(vs[i]*vs_sum_inv) for i in cmps]
                    ys = normalize(ys)
                iter_flows.append(vs)
                iter_comps.append(ys)
                iter_betas.append(vs_sum) # Would be divided by feed but feed is zs = 1
                iter_phases.append(phases[j].to_TP_zs(zs=ys, **phase_kwargs))
                remaining = [remaining[i] - vs[i] for i in cmps]

            flows_ref = remaining
            iter_flows.insert(ref_phase, remaining)

            beta_ref = sum(remaining)
            iter_betas.insert(ref_phase, beta_ref)

            xs_ref = normalize([abs(i) for i in remaining])
            iter_comps.insert(ref_phase, xs_ref)

            phase_ref = phases[ref_phase].to_TP_zs(zs=xs_ref, **phase_kwargs)
            iter_phases.insert(ref_phase, phase_ref)

            lnphis_ref = phase_ref.lnphis()

            errs = []
            for k in phase_iter_n1:
                phase = iter_phases[k]
                lnphis = phase.lnphis()
                xs = iter_comps[k]
                for i in cmps:
                    # This is identical to lnfugacity(i)^j - lnfugacity(i)^ref
                    gi = trunc_log(xs[i]/xs_ref[i]) + lnphis[i] - lnphis_ref[i]
                    errs.append(gi)

            spec_phases = []
            spec_calc = 0.0
            for k in phase_iter:
                spec_phase = spec_callables[k](iter_phases[k])
                spec_phases.append(spec_phase)
                spec_calc += spec_phase*iter_betas[k]
            errs.append(spec_calc - spec_val)
        else:
            iter_betas, iter_comps, iter_phases, errs, jac_arr, flows, iter_val_check, spec_phases = info
            beta_ref = iter_betas[ref_phase]
            xs_ref = iter_comps[ref_phase]
            phase_ref = iter_phases[ref_phase]
            lnphis_ref = phase_ref.lnphis()

#        print(errs[-1], 'err', iter_val, 'T')


        if jac:
            dlnfugacities_ref = phase_ref.dlnfugacities_dns()
            jac_arr = [[0.0]*(N*(phase_count-1) + 1) for i in range(N*(phase_count-1)+1)]
            for ni, nj in zip(phase_iter_n1, phase_iter_n1_0):
                p = iter_phases[ni]
                dlnfugacities = p.dlnfugacities_dns()
                # Begin with the first row using ni, nj;
                for i in cmps:
                    for ki, kj in zip(phase_iter_n1, phase_iter_n1_0):
                        for j in cmps:
                            delta = 1.0 if nj == kj else 0.0
                            v_ref = dlnfugacities_ref[i][j]/beta_ref
                            jac_arr[nj*N + i][kj*N + j] = dlnfugacities[i][j]*delta/iter_betas[ni] + v_ref

            dlnphis_dspec = [dlnphis_diter_callables[i](phases[i]) for i in phase_iter]
            dlnphis_dspec_ref = dlnphis_dspec[ref_phase]
            for ni, nj in zip(phase_iter_n1, phase_iter_n1_0):
                p = iter_phases[ni]
                for i in cmps:
                    jac_arr[nj*N + i][-1] = dlnphis_dspec[ni][i] - dlnphis_dspec_ref[i]

#            last =
            dspec_calc = 0.0
            for k in phase_iter:
                dspec_calc += dspec_diter_callables[k](iter_phases[k])*iter_betas[k]
            jac_arr[-1][-1] = dspec_calc

            dspec_dns = [dspec_dn_callables[i](phases[i]) for i in phase_iter]
            dspec_dns_ref = dspec_dns[ref_phase]
            last_jac_row = jac_arr[-1]

            for ni, nj in zip(phase_iter_n1, phase_iter_n1_0):
                for i in cmps:
                    # What is wrong?
                    # H is multiplied by the phase fraction, of which this n is a part of
                    # So there must be two parts here
                    last_jac_row[nj*N + i] = ((iter_betas[ni]*dspec_dns[ni][i]/iter_betas[ni] - beta_ref*dspec_dns_ref[i]/beta_ref)
                                            + (spec_phases[ni] - spec_phases[ref_phase]))

            if skip_err:
                return jac_arr

        info[:] = iter_betas, iter_comps, iter_phases, errs, jac_arr, flows, iter_val, spec_phases
        if jac:
            return errs, jac_arr
        return errs

    if method == 'newton_system':
        comp_val, iterations = newton_system(to_solve, guesses, jac=True,
                                             xtol=tol, damping=1,
                                             damping_func=damping_maintain_sign)
    else:
        def f_jac_numpy(flows_guess):
            # needed
            ans = to_solve(flows_guess)
            if jac:
                return np.array(ans[0]), np.array(ans[1])
            return np.array(ans)
        def jac_numpy(flows_guess):
            if flows_guess.tolist() == info[5] + [info[6]]:
                a =  np.array(to_solve(flows_guess, jac=True, skip_err=True))
#                b = np.array(to_solve(flows_guess, jac=True)[1])
#                from numpy.testing import assert_allclose
#                assert_allclose(a, b, rtol=1e-10)
                return a
#            print('fail jac', tuple(flows_guess.tolist()), tuple(info[5]))
#            print('new jac')
            return np.array(to_solve(flows_guess, jac=True)[1])

        if method == 'fsolve':
            # Need a function cache! 2 wasted fevals, 1 wasted jaceval
            if analytical_jac:
                jac = False
                sln, infodict, _, _ = fsolve(f_jac_numpy, guesses, fprime=jac_numpy, xtol=tol, full_output=1, **solve_kwargs)
            else:
                sln, infodict, _, _ = fsolve(f_jac_numpy, guesses, xtol=tol, full_output=1, **solve_kwargs)
            iterations = infodict['nfev']
        else:
            sln = root(f_jac_numpy, guesses, tol=tol, jac=(True if jac else None), method=method, **solve_kwargs)
            iterations = sln['nfev']

    betas, compositions, phases, errs, jac, flows, iter_val, spec_phases = info

    sln = (iter_val, betas, compositions, phases, errs, jac, iterations)
    if debug:
        return sln, flows, to_solve
    return sln


def nonlin_2P(T, P, zs, xs_guess, ys_guess, liquid_phase,
              gas_phase, maxiter=1000, tol=1E-13,
              trivial_solution_tol=1e-5, V_over_F_guess=None,
              method='hybr'):
    # Do with just n?
    cmps = range(len(zs))
    xs, ys = xs_guess, ys_guess
    if V_over_F_guess is None:
        V_over_F = 0.5
    else:
        V_over_F = V_over_F_guess
    Ks_guess = [ys[i]/xs[i] for i in cmps]

    info = [0, None, None, None]
    def to_solve(lnKsVFTrans):
        Ks = [trunc_exp(i) for i in lnKsVFTrans[:-1]]
        V_over_F = (0.0 + (1.0 - 0.0)/(1.0 + trunc_exp(-lnKsVFTrans[-1]))) # Translation function - keep it zero to 1

        xs = [zs[i]/(1.0 + V_over_F*(Ks[i] - 1.0)) for i in cmps]
        ys = [Ks[i]*xs[i] for i in cmps]

        g = gas_phase.to_TP_zs(T=T, P=P, zs=ys)
        l = liquid_phase.to_TP_zs(T=T, P=P, zs=xs)

        lnphis_g = g.lnphis()
        lnphis_l = l.lnphis()
#        print(g.fugacities(), l.fugacities())
        new_Ks = [exp(lnphis_l[i] - lnphis_g[i]) for i in cmps]
        VF_err = Rachford_Rice_flash_error(V_over_F, zs, new_Ks)

        err = [new_Ks[i] - Ks[i] for i in cmps] + [VF_err]
        info[1:] = l, g, err
        info[0] += 1
        return err

    VF_guess_in_basis = -log((1.0-V_over_F)/(V_over_F-0.0))

    guesses = [log(i) for i in Ks_guess]
    guesses.append(VF_guess_in_basis)
#    try:
    sol = root(to_solve, guesses, tol=tol, method=method)
    # No reliable way to get number of iterations from OptimizeResult
#        solution, infodict, ier, mesg = fsolve(to_solve, guesses, full_output=True)
    solution = sol.x.tolist()
    V_over_F = (0.0 + (1.0 - 0.0)/(1.0 + exp(-solution[-1])))
    Ks = [exp(solution[i]) for i in cmps]
    xs = [zs[i]/(1.0 + V_over_F*(Ks[i] - 1.0)) for i in cmps]
    ys = [Ks[i]*xs[i] for i in cmps]
#    except Exception as e:
#        raise UnconvergedError(e)

    tot_err = 0.0
    for i in info[3]:
        tot_err += abs(i)
    return V_over_F, xs, ys, info[1], info[2], info[0], tot_err



def nonlin_2P_HSGUAbeta(spec, spec_var, iter_val, iter_var, fixed_val,
                        fixed_var, zs, xs_guess, ys_guess, liquid_phase,
                        gas_phase, maxiter=1000, tol=1E-13,
                        trivial_solution_tol=1e-5, V_over_F_guess=None,
                        method='hybr'
                        ):
    cmps = range(len(zs))
    xs, ys = xs_guess, ys_guess
    if V_over_F_guess is None:
        V_over_F = 0.5
    else:
        V_over_F = V_over_F_guess
    Ks_guess = [ys[i]/xs[i] for i in cmps]

    kwargs_l = {'zs': xs_guess, fixed_var: fixed_val}
    kwargs_g = {'zs': ys_guess, fixed_var: fixed_val}

    info = [0, None, None, None, None]
    def to_solve(lnKsVFTransHSGUABeta):
        Ks = [trunc_exp(i) for i in lnKsVFTransHSGUABeta[:-2]]
        V_over_F = (0.0 + (1.0 - 0.0)/(1.0 + trunc_exp(-lnKsVFTransHSGUABeta[-2]))) # Translation function - keep it zero to 1
        iter_val = lnKsVFTransHSGUABeta[-1]

        xs = [zs[i]/(1.0 + V_over_F*(Ks[i] - 1.0)) for i in cmps]
        ys = [Ks[i]*xs[i] for i in cmps]

        kwargs_l[iter_var] = iter_val
        kwargs_l['zs'] = xs
        kwargs_g[iter_var] = iter_val
        kwargs_g['zs'] = ys

        g = gas_phase.to(**kwargs_g)
        l = liquid_phase.to(**kwargs_l)

        lnphis_g = g.lnphis()
        lnphis_l = l.lnphis()
        new_Ks = [exp(lnphis_l[i] - lnphis_g[i]) for i in cmps]
        VF_err = Rachford_Rice_flash_error(V_over_F, zs, new_Ks)

        val_l = getattr(l, spec_var)()
        val_g = getattr(g, spec_var)()
        val = V_over_F*val_g + (1.0 - V_over_F)*val_l

        other_err = val - spec


        err = [new_Ks[i] - Ks[i] for i in cmps] + [VF_err, other_err]
        info[1:] = l, g, err, other_err
        info[0] += 1
#        print(lnKsVFTransHSGUABeta, err)
        return err

    VF_guess_in_basis = -log((1.0-V_over_F)/(V_over_F-0.0))

    guesses = [log(i) for i in Ks_guess]
    guesses.append(VF_guess_in_basis)
    guesses.append(iter_val)
#    solution, iterations = broyden2(guesses, fun=to_solve, jac=False, xtol=1e-7,
#                                    maxiter=maxiter, jac_has_fun=False, skip_J=True)

    sol = root(to_solve, guesses, tol=tol, method=method)
    solution = sol.x.tolist()
    V_over_F = (0.0 + (1.0 - 0.0)/(1.0 + exp(-solution[-2])))
    iter_val = solution[-1]
    Ks = [exp(solution[i]) for i in cmps]
    xs = [zs[i]/(1.0 + V_over_F*(Ks[i] - 1.0)) for i in cmps]
    ys = [Ks[i]*xs[i] for i in cmps]

    tot_err = 0.0
    for v in info[3]:
        tot_err += abs(v)
    return V_over_F, solution[-1], xs, ys, info[1], info[2], info[0], tot_err

#def broyden2(xs, fun, jac, xtol=1e-7, maxiter=100, jac_has_fun=False,
#             skip_J=False):



def nonlin_n_2P(T, P, zs, xs_guess, ys_guess, liquid_phase,
              gas_phase, maxiter=1000, tol=1E-13,
              trivial_solution_tol=1e-5, V_over_F_guess=None,
              method='hybr'):

    cmps = range(len(zs))
    xs, ys = xs_guess, ys_guess
    if V_over_F_guess is None:
        V_over_F = 0.45
    else:
        V_over_F = V_over_F_guess

    ns = [ys[i]*V_over_F for i in cmps]

    info = [0, None, None, None]
    def to_solve(ns):
        ys = normalize(ns)
        ns_l = [zs[i] - ns[i] for i in cmps]
#         print(sum(ns)+sum(ns_l))
        xs = normalize(ns_l)
#         print(ys, xs)

        g = gas_phase.to_TP_zs(T=T, P=P, zs=ys)
        l = liquid_phase.to_TP_zs(T=T, P=P, zs=xs)

#         print(np.array(g.dfugacities_dns()) - np.array(l.dfugacities_dns()) )

        fugacities_g = g.fugacities()
        fugacities_l = l.fugacities()


        err = [fugacities_g[i] - fugacities_l[i] for i in cmps]
        info[1:] = l, g, err
        info[0] += 1
#         print(err)
        return err

#     print(np.array(jacobian(to_solve, ns, scalar=False)))
#     print('ignore')

    sol = root(to_solve, ns, tol=tol, method=method)
    ns_sln = sol.x.tolist()
    ys = normalize(ns_sln)
    xs_sln = [zs[i] - ns_sln[i] for i in cmps]
    xs = normalize(xs_sln)

    return xs, ys

def nonlin_2P_newton(T, P, zs, xs_guess, ys_guess, liquid_phase,
              gas_phase, maxiter=1000, xtol=1E-10,
              trivial_solution_tol=1e-5, V_over_F_guess=None):
    N = len(zs)
    cmps = range(N)
    xs, ys = xs_guess, ys_guess
    if V_over_F_guess is None:
        V_over_F = 0.5
    else:
        V_over_F = V_over_F_guess

    Ks_guess = [ys[i]/xs[i] for i in cmps]

    info = []
    def to_solve(lnKsVF):
        # Jacobian verified. However, very sketchy - mole fractions may want
        # to go negative.
        lnKs = lnKsVF[:-1]
        Ks = [exp(lnKi) for lnKi in lnKs]
        VF = float(lnKsVF[-1])
        # if VF > 1:
        #     VF = 1-1e-15
        # if VF < 0:
        #     VF = 1e-15

        xs = [zi/(1.0 + VF*(Ki - 1.0)) for zi, Ki in zip(zs, Ks)]
        ys = [Ki*xi for Ki, xi in zip(Ks, xs)]

        g = gas_phase.to_TP_zs(T=T, P=P, zs=ys)
        l = liquid_phase.to_TP_zs(T=T, P=P, zs=xs)

        lnphis_g = g.lnphis()
        lnphis_l = l.lnphis()

        size = N + 1
        J = [[None]*size for i in range(size)]

        d_lnphi_dxs = l.dlnphis_dzs()
        d_lnphi_dys = g.dlnphis_dzs()


        J[N][N] = 1.0

        # Last column except last value; believed correct
        # Was not correct when compared to numerical solution
        Ksm1 = [Ki - 1.0 for Ki in Ks]
        RR_denoms_inv2 = []
        for i in cmps:
            t = 1.0 + VF*Ksm1[i]
            RR_denoms_inv2.append(1.0/(t*t))

        RR_terms = [zs[k]*Ksm1[k]*RR_denoms_inv2[k] for k in cmps]
        for i in cmps:
            value = 0.0
            d_lnphi_dxs_i, d_lnphi_dys_i = d_lnphi_dxs[i], d_lnphi_dys[i]
            for k in cmps:
                value += RR_terms[k]*(d_lnphi_dxs_i[k] - Ks[k]*d_lnphi_dys_i[k])
            J[i][-1] = value


        # Main body - expensive to compute! Lots of elements
        zsKsRRinvs2 = [zs[j]*Ks[j]*RR_denoms_inv2[j] for j in cmps]
        one_m_VF = 1.0 - VF
        for i in cmps:
            Ji = J[i]
            d_lnphi_dxs_is, d_lnphi_dys_is = d_lnphi_dxs[i], d_lnphi_dys[i]
            for j in cmps:
                value = 1.0 if i == j else 0.0
                value += zsKsRRinvs2[j]*(VF*d_lnphi_dxs_is[j] + one_m_VF*d_lnphi_dys_is[j])
                Ji[j] = value

        # Last row except last value  - good, working
        # Diff of RR w.r.t each log K
        bottom_row = J[-1]
        for j in cmps:
            bottom_row[j] = zsKsRRinvs2[j]*(one_m_VF) + VF*zsKsRRinvs2[j]

        # Last value - good, working, being overwritten
        dF_ncp1_dB = 0.0
        for i in cmps:
            dF_ncp1_dB -= RR_terms[i]*Ksm1[i]
        J[-1][-1] = dF_ncp1_dB


        err_RR = Rachford_Rice_flash_error(VF, zs, Ks)
        Fs = [lnKi - lnphi_l + lnphi_g for lnphi_l, lnphi_g, lnKi in zip(lnphis_l, lnphis_g, lnKs)]
        Fs.append(err_RR)

        info[:] = VF, xs, ys, l, g, Fs, J
        return Fs, J

    guesses = [log(i) for i in Ks_guess]
    guesses.append(V_over_F)

    # TODO trust-region
    sln, iterations = newton_system(to_solve, guesses, jac=True, xtol=xtol,
                                    maxiter=maxiter,
                                    damping_func=make_damp_initial(steps=3),
                                    damping=.5)

    VF, xs, ys, l, g, Fs, J = info

    tot_err = 0.0
    for Fi in Fs:
        tot_err += abs(Fi)
    return VF, xs, ys, l, g, tot_err, J, iterations


def gdem(x, x1, x2, x3):
    cmps = range(len(x))
    dx2 = [x[i] - x3[i] for i in cmps]
    dx1 = [x[i] - x2[i] for i in cmps]
    dx = [x[i] - x1[i] for i in cmps]

    b01, b02, b12, b11, b22 = 0.0, 0.0, 0.0, 0.0, 0.0

    for i in cmps:
        b01 += dx[i]*dx1[i]
        b02 += dx[i]*dx2[i]
        b12 += dx1[i]*dx2[i]
        b11 += dx1[i]*dx1[i]
        b22 += dx2[i]*dx2[i]

    den_inv = 1.0/(b11*b22 - b12*b12)
    mu1 = den_inv*(b02*b12 - b01*b22)
    mu2 = den_inv*(b01*b12 - b02*b11)

    factor = 1.0/(1.0 + mu1 + mu2)
    return [factor*(dx[i] - mu2*dx1[i]) for i in cmps]


def minimize_gibbs_2P_transformed(T, P, zs, xs_guess, ys_guess, liquid_phase,
                                  gas_phase, maxiter=1000, tol=1E-13,
                                  trivial_solution_tol=1e-5, V_over_F_guess=None):
    if V_over_F_guess is None:
        V_over_F = 0.5
    else:
        V_over_F = V_over_F_guess

    flows_v = [yi*V_over_F for yi in ys_guess]
    cmps = range(len(zs))

    calc_phases = []
    def G(flows_v):
        vs = [(0.0 + (zs[i] - 0.0)/(1.0 - flows_v[i])) for i in cmps]
        ls = [zs[i] - vs[i] for i in cmps]
        xs = normalize(ls)
        ys = normalize(vs)

        VF = flows_v[0]/ys[0]

        g = gas_phase.to_TP_zs(T=T, P=P, zs=ys)
        l = liquid_phase.to_TP_zs(T=T, P=P, zs=xs)

        G_l = l.G()
        G_g = g.G()
        calc_phases[:] = G_l, G_g
        GE_calc = (G_g*VF + (1.0 - VF)*G_l)/(R*T)
        return GE_calc

    ans = minimize(G, flows_v)

    flows_v = ans['x']
    vs = [(0.0 + (zs[i] - 0.0) / (1.0 - flows_v[i])) for i in cmps]
    ls = [zs[i] - vs[i] for i in cmps]
    xs = normalize(ls)
    ys = normalize(vs)

    V_over_F = flows_v[0] / ys[0]
    return V_over_F, xs, ys, calc_phases[0], calc_phases[1], ans['nfev'], ans['fun']


def minimize_gibbs_NP_transformed(T, P, zs, compositions_guesses, phases,
                                  betas, tol=1E-13,
                                  method='L-BFGS-B', opt_kwargs=None, translate=False):
    if opt_kwargs is None:
        opt_kwargs = {}
    N = len(zs)
    cmps = range(N)
    phase_count = len(phases)
    phase_iter = range(phase_count)
    phase_iter_n1 = range(phase_count-1)
    if method == 'differential_evolution':
        translate = True
#    RT_inv = 1.0/(R*T)

    # Only exist for the first n phases
    # Do not multiply by zs - we are already multiplying by a composition
    flows_guess = [compositions_guesses[j][i]*betas[j] for j in range(phase_count - 1) for i in cmps]
    # Convert the flow guesses to the basis used
    remaining = zs
    if translate:
        flows_guess_basis = []
        for j in range(phase_count-1):
            phase_guess = flows_guess[j*N:j*N+N]
            flows_guess_basis.extend([-trunc_log((remaining[i]-phase_guess[i])/(phase_guess[i]-0.0)) for i in cmps])
            remaining = [remaining[i] - phase_guess[i] for i in cmps]
    else:
        flows_guess_basis = flows_guess

    global min_G, iterations
    jac, hess = False, False
    real_min = False
    min_G = 1e100
    iterations = 0
    info = []
    last = []
    def G(flows):
        global min_G, iterations
        try:
            flows = flows.tolist()
        except:
            flows = list(flows)
        iterations += 1
        iter_flows = []
        iter_comps = []
        iter_betas = []
        iter_phases = []

        remaining = zs
        if not translate:
            for i in range(len(flows)):
                if flows[i] < 1e-10:
                    flows[i] = 1e-10

        for j in phase_iter:
            v = flows[j*N:j*N+N]

            # Mole flows of phase0/vapor
            if j == phase_count - 1:
                vs = remaining
            else:
                if translate:
                    vs = [(0.0 + (remaining[i] - 0.0)/(1.0 + trunc_exp(-v[i]))) for i in cmps]
                else:
                    vs = v
            vs_sum = sum(abs(i) for i in vs)
            if vs_sum == 0.0:
                # Handle the case an optimizer takes all of all compounds already
                ys = zs
            else:
                vs_sum_inv = 1.0/vs_sum
                ys = [abs(vs[i]*vs_sum_inv) for i in cmps]
                ys = normalize(ys)
            iter_flows.append(vs)
            iter_comps.append(ys)
            iter_betas.append(vs_sum) # Would be divided by feed but feed is zs = 1

            remaining = [remaining[i] - vs[i] for i in cmps]
        G = 0.0
        jac_array = []
        for j in phase_iter:
            comp = iter_comps[j]
            phase = phases[j].to_TP_zs(T=T, P=P, zs=comp)
            lnphis = phase.lnphis()
            if real_min:
                # fugacities = phase.fugacities()
                # fugacities = phase.phis()
                #G += sum([iter_flows[j][i]*trunc_log(fugacities[i]) for i in cmps])
                G += phase.G()*iter_betas[j]
            else:
                for i in cmps:
                    G += iter_flows[j][i]*(trunc_log(comp[i]) + lnphis[i])
            iter_phases.append(phase)

        if 0:
            fugacities_last = iter_phases[-1].fugacities()
#            G = 0.0
            for j in phase_iter_n1:
                fugacities = iter_phases[j].fugacities()
                G += sum([abs(fugacities_last[i] - fugacities[i]) for i in cmps])
#                lnphis = phase.lnphis()

#         if real_min:
#             G += G_base
# #        if not jac:
#            for j in phase_iter:
#                comp = iter_comps[j]
#            G += phase.G()*iter_betas[j]
#            if jac:
#                r = []
#                for i in cmps:
#                    v = (log())
#                jac_array.append([log()])
        jac_arr = []
        comp = iter_comps[0]
        phase = iter_phases[0]
        lnphis = phase.lnphis()
        base = [log(xi) + lnphii for xi, lnphii in zip(comp, lnphis)]
        if jac:
            for j in range(1, phase_count):
                comp = iter_comps[j]
                phase = iter_phases[j]
                lnphis = phase.lnphis()
                jac_arr.extend([ref - (log(xi) + lnphii) for ref, xi, lnphii in zip(base, comp, lnphis)])

            jac_arr = []
            comp_last = iter_comps[-1]
            phase_last = iter_phases[-1]
            flows_last = iter_flows[-1]
            lnphis_last = phase_last.lnphis()
            dlnphis_dns_last = phase_last.dlnphis_dns()

            for j in phase_iter_n1:
                comp = iter_comps[j]
                phase = iter_phases[j]
                flows = iter_flows[j]
                lnphis = phase.lnphis()
                dlnphis_dns = phase.dlnphis_dns()
                for i in cmps:
                    v = 0
                    for k in cmps:
                        v += flows[k][i]*lnphis[k][i]
                        v -= flows_last[i]*dlnphis_dns_last[k][i]
                    v += lnphis[i] + log(comp[i])





        if G < min_G:
            #  'phases', iter_phases
            print('new min G', G,  'betas', iter_betas, 'comp', iter_comps)
            info[:] = iter_betas, iter_comps, iter_phases, G
            min_G = G
        last[:] = iter_betas, iter_comps, iter_phases, G
        if hess:
            base = iter_phases[0].dlnfugacities_dns()
            p1 = iter_phases[1].dlnfugacities_dns()
            dlnphis_dns = [i.dlnphis_dns() for i in iter_phases]
            dlnphis_dns0 = iter_phases[0].dlnphis_dns()
            dlnphis_dns1 = iter_phases[1].dlnphis_dns()
            xs, ys = iter_comps[0], iter_comps[1]
            hess_arr = []
            beta = iter_betas[0]

            hess_arr = [[0.0]*N*(phase_count-1) for i in range(N*(phase_count-1))]
            for n in range(1, phase_count):
                for m in range(1, phase_count):
                    for i in cmps:
                        for j in cmps:
                            delta = 1.0 if i == j else 0.0
                            v = 1.0/iter_betas[n]*(1.0/iter_comps[n][i]*delta
                                               - 1.0 + dlnphis_dns[n][i][j])
                            v += 1.0/iter_betas[0]*(1.0/iter_comps[0][i]*delta
                                               - 1.0 + dlnphis_dns[0][i][j])
                            hess_arr[(n-1)*N+i][(m-1)*N+j] = v
#
#            for n in range(1, phase_count):
#                for i in cmps:
#                    r = []
#                    for j in cmps:
#                        v = 0.0
#                        for m in phase_iter:
#                            delta = 1.0 if i ==j else 0.0
#                            v += 1.0/iter_betas[m]*(1.0/iter_comps[m][i]*delta
#                                               - 1.0 + dlnphis_dns[m][i][j])
#
#                        # How the heck to make this multidimensional?
#                        # v = 1.0/(beta*(1.0 - beta))*(zs[i]*delta/(xs[i]*ys[i])
#                        #                              - 1.0 + (1.0 - beta)*dlnphis_dns0[i][j]
#                        #                              + beta*dlnphis_dns1[i][j])
#
#                        # v = base[i][j] + p1[i][j]
#                        r.append(v)
#                    hess_arr.append(r)
            # Going to be hard to figure out
            # for j in range(1, phase_count):
            #     comp = iter_comps[j]
            #     phase = iter_phases[j]
            #     dlnfugacities_dns = phase.dlnfugacities_dns()
            #     row = [base[i] + dlnfugacities_dns[i] for i in cmps]
            #     hess_arr = row
                # hess_arr.append(row)
            return G, jac_arr, hess_arr
        if jac:
            return G, np.array(jac_arr)
        return G
#    ans = None
    if method == 'differential_evolution':
        from scipy.optimize import differential_evolution
        real_min = True
        translate = True

        G_base = 1e100
        for p in phases:
            G_calc = p.to(T=T,P=P, zs=zs).G()
            if G_base > G_calc:
                G_base = G_calc
        jac = hess = False
#        print(G(list(flows_guess_basis)))
        ans = differential_evolution(G, [(-30.0, 30.0) for i in cmps for j in range(phase_count-1)], **opt_kwargs)
#        ans = differential_evolution(G, [(-100.0, 100.0) for i in cmps for j in range(phase_count-1)], **opt_kwargs)
        objf = float(ans['fun'])
    elif method == 'newton_minimize':
        import numdifftools as nd
        jac = True
        hess = True
        initial_hess = nd.Hessian(lambda x: G(x)[0], step=1e-4)(flows_guess_basis)
        ans, iters = newton_minimize(G, flows_guess_basis, jac=True, hess=True, xtol=tol, ytol=None, maxiter=100, damping=1.0,
                  damping_func=damping_maintain_sign)
        objf = None
    else:
        jac = True
        hess = True
        import numdifftools as nd
        def hess_fun(flows):
            return np.array(G(flows)[2])

        # hess_fun = lambda flows_guess_basis: np.array(G(flows_guess_basis)[2])
#        nd.Jacobian(G, step=1e-5)
        # trust-constr special handling to add constraints
        def fun_and_jac(x):
            x, j, _ = G(x)
            return x, np.array(j)

        ans = minimize(fun_and_jac, flows_guess_basis, jac=True, hess=hess_fun, method=method, tol=tol, **opt_kwargs)
        objf = float(ans['fun'])
#    G(ans['x']) # Make sure info has right value
#    ans['fun'] *= R*T

    betas, compositions, phases, objf = info#info
    return betas, compositions, phases, iterations, objf

def TP_solve_VF_guesses(zs, method, constants, correlations,
                        T=None, P=None, VF=None,
                        maxiter=50, xtol=1E-7, ytol=None,
                        bounded=False,
                        user_guess=None, last_conv=None):
    if method == IDEAL_PSAT:
        return flash_ideal(zs=zs, funcs=correlations.VaporPressures, Tcs=constants.Tcs, T=T, P=P, VF=VF)
    elif method == WILSON_GUESS:
        return flash_wilson(zs, Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas, T=T, P=P, VF=VF)
    elif method == TB_TC_GUESS:
        return flash_Tb_Tc_Pc(zs, Tbs=constants.Tbs, Tcs=constants.Tcs, Pcs=constants.Pcs, T=T, P=P, VF=VF)

    # Simple return values - not going through a model
    elif method == STP_T_GUESS:
        return flash_ideal(zs=zs, funcs=correlations.VaporPressures, Tcs=constants.Tcs, T=298.15, P=101325.0)
    elif method == LAST_CONVERGED:
        if last_conv is None:
            raise ValueError("No last converged")
        return last_conv
    else:
        raise ValueError("Could not converge")



def dew_P_newton(P_guess, T, zs, liquid_phase, gas_phase,
                 maxiter=200, xtol=1E-10, xs_guess=None,
                 max_step_damping=1e5,
                 trivial_solution_tol=1e-4):
    # Trial function only
    V = None
    N = len(zs)
    cmps = range(N)
    xs = zs if xs_guess is None else xs_guess


    V_over_F = 1.0
    def to_solve(lnKsP):
        # d(fl_i - fg_i)/d(ln K,i) -
        # rest is less important

        # d d(fl_i - fg_i)/d(P) should be easy
        Ks = [trunc_exp(i) for i in lnKsP[:-1]]
        P = lnKsP[-1]

        xs = [zs[i]/(1.0 + V_over_F*(Ks[i] - 1.0)) for i in cmps]
        ys = [Ks[i]*xs[i] for i in cmps]

        g = gas_phase.to(ys, T=T, P=P, V=V)
        l = liquid_phase.to(xs, T=T, P=P, V=V)

        fugacities_l = l.fugacities()
        fugacities_g = g.fugacities()
        VF_err = Rachford_Rice_flash_error(V_over_F, zs, Ks)
        errs = [fi_l - fi_g for fi_l, fi_g in zip(fugacities_l, fugacities_g)]
        errs.append(VF_err)
        return errs

    lnKs_guess = [log(zs[i]/xs[i]) for i in cmps]
    lnKs_guess.append(P_guess)
    def jac(lnKsP):
        j = jacobian(to_solve, lnKsP, scalar=False)
        return j

    lnKsP, iterations = newton_system(to_solve, lnKs_guess, jac=jac, xtol=xtol)

    xs = [zs[i]/(1.0 + V_over_F*(exp(lnKsP[i]) - 1.0)) for i in cmps]
#    ys = [exp(lnKsP[i])*xs[i] for i in cmps]
    return lnKsP[-1], xs, zs, iterations


def dew_bubble_newton_zs(guess, fixed_val, zs, liquid_phase, gas_phase,
                        iter_var='T', fixed_var='P', V_over_F=1, # 1 = dew, 0 = bubble
                        maxiter=200, xtol=1E-10, comp_guess=None,
                        max_step_damping=1e5, damping=1.0,
                        trivial_solution_tol=1e-4, debug=False,
                        method='newton', opt_kwargs=None):
    V = None
    N = len(zs)
    cmps = range(N)
    if comp_guess is None:
        comp_guess = zs

    if V_over_F == 1.0:
        iter_phase, const_phase = liquid_phase, gas_phase
    elif V_over_F == 0.0:
        iter_phase, const_phase = gas_phase, liquid_phase
    else:
        raise ValueError("Supports only VF of 0 or 1")

    lnKs = [0.0]*N

    size = N + 1
    errs = [0.0]*size
    comp_invs = [0.0]*N
    J = [[0.0]*size for i in range(size)]
    #J[N][N] = 0.0 as well
    JN = J[N]
    for i in cmps:
        JN[i] = -1.0

    s = 'dlnphis_d%s' %(iter_var)
    dlnphis_diter_var_iter = getattr(iter_phase.__class__, s)
    dlnphis_diter_var_const = getattr(const_phase.__class__, s)
    dlnphis_dzs = getattr(iter_phase.__class__, 'dlnphis_dzs')

    info = []
    kwargs = {}
    kwargs[fixed_var] = fixed_val
    kwargs['V'] = None
    def to_solve_comp(iter_vals, jac=True):
        comp = iter_vals[:-1]
        iter_val = iter_vals[-1]

        kwargs[iter_var] = iter_val
        p_iter = iter_phase.to(comp, **kwargs)
        p_const = const_phase.to(zs, **kwargs)

        lnphis_iter = p_iter.lnphis()
        lnphis_const = p_const.lnphis()
        for i in cmps:
            comp_invs[i] = comp_inv = 1.0/comp[i]
            lnKs[i] = log(zs[i]*comp_inv)
            errs[i] = lnKs[i] - lnphis_iter[i] + lnphis_const[i]
        errs[-1] = 1.0 - sum(comp)

        if jac:
            dlnphis_dxs = dlnphis_dzs(p_iter)
            dlnphis_dprop_iter = dlnphis_diter_var_iter(p_iter)
            dlnphis_dprop_const = dlnphis_diter_var_const(p_const)
            for i in cmps:
                Ji = J[i]
                Ji[-1] = dlnphis_dprop_const[i] - dlnphis_dprop_iter[i]
                for j in cmps:
                    Ji[j] = -dlnphis_dxs[i][j]
                Ji[i] -= comp_invs[i]

            info[:] = [p_iter, p_const, errs, J]
            return errs, J

        return errs
    damping = 1.0
    guesses = list(comp_guess)
    guesses.append(guess)
    if method == 'newton':
        comp_val, iterations = newton_system(to_solve_comp, guesses, jac=True,
                                             xtol=xtol, damping=damping,
                                             solve_func=py_solve,
                                             # solve_func=lambda x, y:np.linalg.solve(x, y).tolist(),
                                             damping_func=damping_maintain_sign)
    elif method == 'odeint':
        # Not even close to working
        # equations are hard
        from scipy.integrate import odeint
        def fun_and_jac(x, t):
            x, j = to_solve_comp(x.tolist() + [t])
            return np.array(x), np.array(j)
        def fun(x, t):
            x, j = to_solve_comp(x.tolist() +[t])
            return np.array(x)
        def jac(x, t):
            x, j = to_solve_comp(x.tolist() + [t])
            return np.array(j)

        ans = odeint(func=fun, y0=np.array(guesses), t=np.linspace(guess, guess*2, 5), Dfun=jac)
        return ans
    else:
        if opt_kwargs is None:
            opt_kwargs = {}
        # def fun_and_jac(x):
        #     x, j = to_solve_comp(x.tolist())
        #     return np.array(x), np.array(j)
        
        low = [.0]*N
        low.append(1.0) # guess at minimum pressure
        high = [1.0]*N
        high.append(1e10) # guess at maximum pressure
        
        f_j, into, outof = translate_bound_f_jac(to_solve_comp, jac=True, low=low, high=high, as_np=True)

        ans = root(f_j, np.array(into(guesses)), jac=True, method=method, tol=xtol, **opt_kwargs)
        comp_val = outof(ans['x']).tolist()
        iterations = ans['nfev']

    iter_val = comp_val[-1]
    comp = comp_val[:-1]

    comp_difference = 0.0
    for i in cmps: comp_difference += abs(zs[i] - comp[i])

    if comp_difference < trivial_solution_tol:
        raise ValueError("Converged to trivial condition, compositions of both phases equal")

    if iter_var == 'P' and iter_val > 1e10:
        raise ValueError("Converged to unlikely point")

    sln = [iter_val, comp]
    sln.append(info[0])
    sln.append(info[1])
    sln.append(iterations)
    tot_err = 0.0
    for err_i in info[2]:
        tot_err += abs(err_i)
    sln.append(tot_err)

    if debug:
        return sln, to_solve_comp
    return sln



l_undefined_T_msg = "Could not calculate liquid conditions at provided temperature %s K (mole fracions %s)"
g_undefined_T_msg = "Could not calculate vapor conditions at provided temperature %s K (mole fracions %s)"
l_undefined_P_msg = "Could not calculate liquid conditions at provided pressure %s Pa (mole fracions %s)"
g_undefined_P_msg = "Could not calculate vapor conditions at provided pressure %s Pa (mole fracions %s)"

def dew_bubble_Michelsen_Mollerup(guess, fixed_val, zs, liquid_phase, gas_phase,
                                  iter_var='T', fixed_var='P', V_over_F=1,
                                  maxiter=200, xtol=1E-10, comp_guess=None,
                                  max_step_damping=.25, guess_update_frequency=1,
                                  trivial_solution_tol=1e-7, V_diff=.00002, damping=1.0):
    # for near critical, V diff very wrong - .005 seen, both g as or both liquid
    kwargs = {fixed_var: fixed_val}
    N = len(zs)
    cmps = range(N)
    comp_guess = zs if comp_guess is None else comp_guess
    damping_orig = damping

    if V_over_F == 1.0:
        iter_phase, const_phase, bubble = liquid_phase, gas_phase, False
    elif V_over_F == 0.0:
        iter_phase, const_phase, bubble = gas_phase, liquid_phase, True
    else:
        raise ValueError("Supports only VF of 0 or 1")
    if iter_var == 'T':
        if V_over_F == 1.0:
            iter_msg, const_msg = l_undefined_T_msg, g_undefined_T_msg
        else:
            iter_msg, const_msg = g_undefined_T_msg, l_undefined_T_msg
    elif iter_var == 'P':
        if V_over_F == 1.0:
            iter_msg, const_msg = l_undefined_P_msg, g_undefined_P_msg
        else:
            iter_msg, const_msg = g_undefined_P_msg, l_undefined_P_msg

    s = 'dlnphis_d%s' %(iter_var)
    dlnphis_diter_var_iter = getattr(iter_phase.__class__, s)
    dlnphis_diter_var_const = getattr(const_phase.__class__, s)

    skip = 0
    guess_old = None
    V_ratio, V_ratio_last = None, None
    V_iter_last, V_const_last = None, None
    expect_phase = 'g' if V_over_F == 0.0 else 'l'
    unwanted_phase = 'l' if expect_phase == 'g' else 'g'

    successive_fails = 0
    for iteration in range(maxiter):
        kwargs[iter_var] = guess
        try:
            const_phase = const_phase.to_TP_zs(zs=zs, **kwargs)
            lnphis_const = const_phase.lnphis()
            dlnphis_dvar_const = dlnphis_diter_var_const(const_phase)
        except Exception as e:
            if guess_old is None:
                raise ValueError(const_msg %(guess, zs), e)
            successive_fails += 1
            guess = guess_old + copysign(min(max_step_damping*guess, abs(step)), step)
            continue
        try:
            skip -= 1
            iter_phase = iter_phase.to_TP_zs(zs=comp_guess, **kwargs)
            if V_diff is not None:
                V_iter, V_const = iter_phase.V(), const_phase.V()
                V_ratio = V_iter/V_const
                if 1.0 - V_diff < V_ratio < 1.0 + V_diff or skip > 0 or V_iter_last and (abs(min(V_iter, V_iter_last)/max(V_iter, V_iter_last)) < .8):
                    # Relax the constraint for the iterating on variable so two different phases exist
                    #if iter_phase.eos_mix.phase in ('l', 'g') and iter_phase.eos_mix.phase == const_phase.eos_mix.phase:
                    # Alternatively, try a stability test here

                    if iter_phase.eos_mix.phase == unwanted_phase:
                        if skip < 0:
                            skip = 4
                            damping = .15
                        if iter_var == 'P':
                            split = min(iter_phase.eos_mix.P_discriminant_zeros()) # P_discriminant_zero_l
                            if bubble:
                                split *= 0.999999999
                            else:
                                split *= 1.000000001
                        elif iter_var == 'T':
                            split = iter_phase.eos_mix.T_discriminant_zero_l()
                            if bubble:
                                split *= 0.999999999
                            else:
                                split *= 1.000000001
                        kwargs[iter_var] = guess = split
                        iter_phase = iter_phase.to(zs=comp_guess, **kwargs)
                        const_phase = const_phase.to(zs=zs, **kwargs)
                        lnphis_const = const_phase.lnphis()
                        dlnphis_dvar_const = dlnphis_diter_var_const(const_phase)
                        print('adj iter phase', split)
                    elif const_phase.eos_mix.phase == expect_phase:
                        if skip < 0:
                            skip = 4
                            damping = .15
                        if iter_var == 'P':
                            split = min(const_phase.eos_mix.P_discriminant_zeros())
                            if bubble:
                                split *= 0.999999999
                            else:
                                split *= 1.000000001
                        elif iter_var == 'T':
                            split = const_phase.eos_mix.T_discriminant_zero_l()
                            if bubble:
                                split *= 0.999999999
                            else:
                                split *= 1.000000001
                        kwargs[iter_var] = guess = split
                        const_phase = const_phase.to(zs=zs, **kwargs)
                        lnphis_const = const_phase.lnphis()
                        dlnphis_dvar_const = dlnphis_diter_var_const(const_phase)
                        iter_phase = iter_phase.to(zs=comp_guess, **kwargs)
                        # Also need to adjust the other phase to keep it in sync

                        print('adj const phase', split)

            lnphis_iter = iter_phase.lnphis()
            dlnphis_dvar_iter = dlnphis_diter_var_iter(iter_phase)
        except Exception as e:
            if guess_old is None:
                raise ValueError(iter_msg %(guess, zs), e)
            successive_fails += 1
            guess = guess_old + copysign(min(max_step_damping*guess, abs(step)), step)
            continue


        if successive_fails > 2:
            raise ValueError("Stopped convergence procedure after multiple bad steps")

        successive_fails = 0
        Ks = [exp(a - b) for a, b in zip(lnphis_const, lnphis_iter)]
        comp_guess = [zs[i]*Ks[i] for i in cmps]
        y_sum = sum(comp_guess)
        comp_guess = [y/y_sum for y in comp_guess]
        if iteration % guess_update_frequency: #  or skip > 0
            continue
        elif skip == 0:
            damping = damping_orig

        f_k = sum([zs[i]*Ks[i] for i in cmps]) - 1.0

        dfk_dvar = 0.0
        for i in cmps:
            dfk_dvar += zs[i]*Ks[i]*(dlnphis_dvar_const[i] - dlnphis_dvar_iter[i])

        guess_old = guess
        step = -f_k/dfk_dvar

#        if near_critical:
        adj_step = copysign(min(max_step_damping*guess, abs(step), abs(step)*damping), step)
        if guess + adj_step <= 0.0:
            adj_step *= 0.5
        guess = guess + adj_step
#        else:
#            guess = guess + step
        comp_difference = 0.0
        for i in cmps: comp_difference += abs(zs[i] - comp_guess[i])

        if comp_difference < trivial_solution_tol and iteration:
            for zi in zs:
                if zi == 1.0:
                    # Turn off trivial check for pure components
                    trivial_solution_tol = -1.0
            if comp_difference < trivial_solution_tol:
                raise ValueError("Converged to trivial condition, compositions of both phases equal")


        if abs(guess - guess_old) < xtol: #and not skip:
            guess = guess_old
            break
        if V_diff is not None:
            V_iter_last, V_const_last, V_ratio_last = V_iter, V_const, V_ratio

    if abs(guess - guess_old) > xtol:
        raise ValueError("Did not converge to specified tolerance")
    return guess, comp_guess, iter_phase, const_phase, iteration, abs(guess - guess_old)


l_undefined_T_msg = "Could not calculate liquid conditions at provided temperature %s K (mole fracions %s)"
g_undefined_T_msg = "Could not calculate vapor conditions at provided temperature %s K (mole fracions %s)"
l_undefined_P_msg = "Could not calculate liquid conditions at provided pressure %s Pa (mole fracions %s)"
g_undefined_P_msg = "Could not calculate vapor conditions at provided pressure %s Pa (mole fracions %s)"

def existence_3P_Michelsen_Mollerup(guess, fixed_val, zs, iter_phase, liquid0, liquid1,
                                    iter_var='T', fixed_var='P',
                                    maxiter=200, xtol=1E-10, comp_guess=None,
                                    liquid0_comp=None, liquid1_comp=None,
                                    max_step_damping=.25, SS_tol=1e-10,
                                    trivial_solution_tol=1e-7, damping=1.0,
                                    beta=0.5):
    # For convenience call the two phases that exist already liquid0, liquid1
    # But one of them can be a gas, solid, etc.
    kwargs = {fixed_var: fixed_val}
    N = len(zs)
    cmps = range(N)
    comp_guess = zs if comp_guess is None else comp_guess
    damping_orig = damping

    if iter_var == 'T':
        iter_msg, const_msg = g_undefined_T_msg, l_undefined_T_msg
    elif iter_var == 'P':
        iter_msg, const_msg = g_undefined_P_msg, l_undefined_P_msg

    s = 'dlnphis_d%s' %(iter_var)
    dlnphis_diter_var_iter = getattr(iter_phase.__class__, s)
    dlnphis_diter_var_liquid0 = getattr(liquid0.__class__, s)
#    dlnphis_diter_var_liquid1 = getattr(liquid1.__class__, s)

    skip = 0
    guess_old = None

    successive_fails = 0
    for iteration in range(maxiter):
        kwargs[iter_var] = guess
        try:
            liquid0 = liquid0.to_TP_zs(zs=liquid0_comp, **kwargs)
            lnphis_liquid0 = liquid0.lnphis()
            dlnphis_dvar_liquid0 = dlnphis_diter_var_liquid0(liquid0)
        except Exception as e:
            if guess_old is None:
                raise ValueError(const_msg %(guess, liquid0_comp), e)
            successive_fails += 1
            guess = guess_old + copysign(min(max_step_damping*guess, abs(step)), step)
            continue
        try:
            liquid1 = liquid1.to_TP_zs(zs=liquid1_comp, **kwargs)
            lnphis_liquid1 = liquid1.lnphis()
#            dlnphis_dvar_liquid1 = dlnphis_diter_var_liquid1(liquid1)
        except Exception as e:
            if guess_old is None:
                raise ValueError(const_msg %(guess, liquid0_comp), e)
            successive_fails += 1
            guess = guess_old + copysign(min(max_step_damping*guess, abs(step)), step)
            continue
        try:
            iter_phase = iter_phase.to_TP_zs(zs=comp_guess, **kwargs)
            lnphis_iter = iter_phase.lnphis()
            dlnphis_dvar_iter = dlnphis_diter_var_iter(iter_phase)
        except Exception as e:
            if guess_old is None:
                raise ValueError(iter_msg %(guess, zs), e)
            successive_fails += 1
            guess = guess_old + copysign(min(max_step_damping*guess, abs(step)), step)
            continue


        if successive_fails > 2:
            raise ValueError("Stopped convergence procedure after multiple bad steps")

        successive_fails = 0
        Ks = [exp(a - b) for a, b in zip(lnphis_liquid0, lnphis_iter)]
        comp_guess = [liquid0_comp[i]*Ks[i] for i in cmps]
        y_sum_inv = 1.0/sum(comp_guess)
        comp_guess = [y*y_sum_inv for y in comp_guess]

        f_k = sum([liquid0_comp[i]*Ks[i] for i in cmps]) - 1.0

        dfk_dvar = 0.0
        for i in cmps:
            dfk_dvar += liquid0_comp[i]*Ks[i]*(dlnphis_dvar_liquid0[i] - dlnphis_dvar_iter[i])

        guess_old = guess
        step = -f_k/dfk_dvar

        adj_step = copysign(min(max_step_damping*guess, abs(step), abs(step)*damping), step)
        if guess + adj_step <= 0.0:
            adj_step *= 0.5
        guess = guess + adj_step

        comp_difference = 0.0
        for i in cmps:
            comp_difference += abs(liquid0_comp[i] - comp_guess[i])

        if comp_difference < trivial_solution_tol and iteration:
            if comp_difference < trivial_solution_tol:
                raise ValueError("Converged to trivial condition, compositions of both phases equal")

        # Do the SS part for the two phases
        try:
            Ks_SS = [exp(lnphis_liquid0[i] - lnphis_liquid1[i]) for i in cmps]
        except OverflowError:
            Ks_SS = [trunc_exp(lnphis_liquid0[i] - lnphis_liquid1[i]) for i in cmps]
        beta, liquid0_comp_new, liquid1_comp_new = flash_inner_loop(zs, Ks_SS, guess=beta)

        for xi in liquid0_comp_new:
            if xi < 0.0:
                xs_new_sum_inv = 1.0/sum(abs(i) for i in liquid0_comp_new)
                for i in cmps:
                    liquid0_comp_new[i] = abs(liquid0_comp_new[i])*xs_new_sum_inv
                break
        for xi in liquid1_comp_new:
            if xi < 0.0:
                xs_new_sum_inv = 1.0/sum(abs(i) for i in liquid1_comp_new)
                for i in cmps:
                    liquid1_comp_new[i] = abs(liquid1_comp_new[i])*xs_new_sum_inv
                break
        err_SS = 0.0
        try:
            for Ki, xi, yi in zip(Ks_SS, liquid0_comp, liquid1_comp):
                err_i = Ki*xi/yi - 1.0
                err_SS += err_i*err_i
        except ZeroDivisionError:
            err_SS = 0.0
            for Ki, xi, yi in zip(Ks, xs, ys):
                try:
                    err_i = Ki*xi/yi - 1.0
                    err_SS += err_i*err_i
                except ZeroDivisionError:
                    pass

        liquid0_comp, liquid1_comp = liquid0_comp_new, liquid1_comp_new
        if abs(guess - guess_old) < xtol and err_SS < SS_tol:
            err_VF = abs(guess - guess_old)
            guess = guess_old
            break


    if abs(guess - guess_old) > xtol:
        raise ValueError("Did not converge to specified tolerance")

    return guess, [iter_phase, liquid0, liquid1], [0.0, 1.0-beta, beta], err_VF, err_SS, iteration


def bubble_T_Michelsen_Mollerup(T_guess, P, zs, liquid_phase, gas_phase,
                                maxiter=200, xtol=1E-10, ys_guess=None,
                                max_step_damping=5.0, T_update_frequency=1,
                                trivial_solution_tol=1e-4):
    N = len(zs)
    cmps = range(N)
    ys = zs if ys_guess is None else ys_guess


    T_guess_old = None
    successive_fails = 0
    for iteration in range(maxiter):
        try:
            g = gas_phase.to_TP_zs(T=T_guess, P=P, zs=ys)
            lnphis_g = g.lnphis()
            dlnphis_dT_g = g.dlnphis_dT()
        except Exception as e:
            if T_guess_old is None:
                raise ValueError(g_undefined_T_msg %(T_guess, ys), e)
            successive_fails += 1
            T_guess = T_guess_old + copysign(min(max_step_damping, abs(step)), step)
            continue

        try:
            l = liquid_phase.to_TP_zs(T=T_guess, P=P, zs=zs)
            lnphis_l = l.lnphis()
            dlnphis_dT_l = l.dlnphis_dT()
        except Exception as e:
            if T_guess_old is None:
                raise ValueError(l_undefined_T_msg %(T_guess, zs), e)
            successive_fails += 1
            T_guess = T_guess_old + copysign(min(max_step_damping, abs(step)), step)
            continue

        if successive_fails > 2:
            raise ValueError("Stopped convergence procedure after multiple bad steps")

        successive_fails = 0
        Ks = [exp(a - b) for a, b in zip(lnphis_l, lnphis_g)]
        ys = [zs[i]*Ks[i] for i in cmps]
        if iteration % T_update_frequency:
            continue

        f_k = sum([zs[i]*Ks[i] for i in cmps]) - 1.0

        dfk_dT = 0.0
        for i in cmps:
            dfk_dT += zs[i]*Ks[i]*(dlnphis_dT_l[i] - dlnphis_dT_g[i])

        T_guess_old = T_guess
        step = -f_k/dfk_dT


#        if near_critical:
        T_guess = T_guess + copysign(min(max_step_damping, abs(step)), step)
#        else:
#            T_guess = T_guess + step


        comp_difference = sum([abs(zi - yi) for zi, yi in zip(zs, ys)])
        if comp_difference < trivial_solution_tol:
            raise ValueError("Converged to trivial condition, compositions of both phases equal")

        y_sum = sum(ys)
        ys = [y/y_sum for y in ys]

        if abs(T_guess - T_guess_old) < xtol:
            T_guess = T_guess_old
            break

    if abs(T_guess - T_guess_old) > xtol:
        raise ValueError("Did not converge to specified tolerance")
    return T_guess, ys, l, g, iteration, abs(T_guess - T_guess_old)


def dew_T_Michelsen_Mollerup(T_guess, P, zs, liquid_phase, gas_phase,
                                maxiter=200, xtol=1E-10, xs_guess=None,
                                max_step_damping=5.0, T_update_frequency=1,
                                trivial_solution_tol=1e-4):
    N = len(zs)
    cmps = range(N)
    xs = zs if xs_guess is None else xs_guess


    T_guess_old = None
    successive_fails = 0
    for iteration in range(maxiter):
        try:
            g = gas_phase.to_TP_zs(T=T_guess, P=P, zs=zs)
            lnphis_g = g.lnphis()
            dlnphis_dT_g = g.dlnphis_dT()
        except Exception as e:
            if T_guess_old is None:
                raise ValueError(g_undefined_T_msg %(T_guess, zs), e)
            successive_fails += 1
            T_guess = T_guess_old + copysign(min(max_step_damping, abs(step)), step)
            continue

        try:
            l = liquid_phase.to_TP_zs(T=T_guess, P=P, zs=xs)
            lnphis_l = l.lnphis()
            dlnphis_dT_l = l.dlnphis_dT()
        except Exception as e:
            if T_guess_old is None:
                raise ValueError(l_undefined_T_msg %(T_guess, xs), e)
            successive_fails += 1
            T_guess = T_guess_old + copysign(min(max_step_damping, abs(step)), step)
            continue

        if successive_fails > 2:
            raise ValueError("Stopped convergence procedure after multiple bad steps")

        successive_fails = 0
        Ks = [exp(a - b) for a, b in zip(lnphis_l, lnphis_g)]
        xs = [zs[i]/Ks[i] for i in cmps]
        if iteration % T_update_frequency:
            continue


        f_k = sum(xs) - 1.0

        dfk_dT = 0.0
        for i in cmps:
            dfk_dT += xs[i]*(dlnphis_dT_g[i] - dlnphis_dT_l[i])

        T_guess_old = T_guess
        step = -f_k/dfk_dT


#        if near_critical:
        T_guess = T_guess + copysign(min(max_step_damping, abs(step)), step)
#        else:
#            T_guess = T_guess + step


        comp_difference = sum([abs(zi - xi) for zi, xi in zip(zs, xs)])
        if comp_difference < trivial_solution_tol:
            raise ValueError("Converged to trivial condition, compositions of both phases equal")

        y_sum = sum(xs)
        xs = [y/y_sum for y in xs]

        if abs(T_guess - T_guess_old) < xtol:
            T_guess = T_guess_old
            break

    if abs(T_guess - T_guess_old) > xtol:
        raise ValueError("Did not converge to specified tolerance")
    return T_guess, xs, l, g, iteration, abs(T_guess - T_guess_old)

def bubble_P_Michelsen_Mollerup(P_guess, T, zs, liquid_phase, gas_phase,
                                maxiter=200, xtol=1E-10, ys_guess=None,
                                max_step_damping=1e5, P_update_frequency=1,
                                trivial_solution_tol=1e-4):
    N = len(zs)
    cmps = range(N)
    ys = zs if ys_guess is None else ys_guess


    P_guess_old = None
    successive_fails = 0
    for iteration in range(maxiter):
        try:
            g = gas_phase = gas_phase.to_TP_zs(T=T, P=P_guess, zs=ys)
            lnphis_g = g.lnphis()
            dlnphis_dP_g = g.dlnphis_dP()
        except Exception as e:
            if P_guess_old is None:
                raise ValueError(g_undefined_P_msg %(P_guess, ys), e)
            successive_fails += 1
            P_guess = P_guess_old + copysign(min(max_step_damping, abs(step)), step)
            continue

        try:
            l = liquid_phase= liquid_phase.to_TP_zs(T=T, P=P_guess, zs=zs)
            lnphis_l = l.lnphis()
            dlnphis_dP_l = l.dlnphis_dP()
        except Exception as e:
            if P_guess_old is None:
                raise ValueError(l_undefined_P_msg %(P_guess, zs), e)
            successive_fails += 1
            T_guess = P_guess_old + copysign(min(max_step_damping, abs(step)), step)
            continue

        if successive_fails > 2:
            raise ValueError("Stopped convergence procedure after multiple bad steps")

        successive_fails = 0
        Ks = [exp(a - b) for a, b in zip(lnphis_l, lnphis_g)]
        ys = [zs[i]*Ks[i] for i in cmps]
        if iteration % P_update_frequency:
            continue

        f_k = sum([zs[i]*Ks[i] for i in cmps]) - 1.0

        dfk_dP = 0.0
        for i in cmps:
            dfk_dP += zs[i]*Ks[i]*(dlnphis_dP_l[i] - dlnphis_dP_g[i])

        P_guess_old = P_guess
        step = -f_k/dfk_dP

        P_guess = P_guess + copysign(min(max_step_damping, abs(step)), step)


        comp_difference = sum([abs(zi - yi) for zi, yi in zip(zs, ys)])
        if comp_difference < trivial_solution_tol:
            raise ValueError("Converged to trivial condition, compositions of both phases equal")

        y_sum = sum(ys)
        ys = [y/y_sum for y in ys]

        if abs(P_guess - P_guess_old) < xtol:
            P_guess = P_guess_old
            break

    if abs(P_guess - P_guess_old) > xtol:
        raise ValueError("Did not converge to specified tolerance")
    return P_guess, ys, l, g, iteration, abs(P_guess - P_guess_old)


def dew_P_Michelsen_Mollerup(P_guess, T, zs, liquid_phase, gas_phase,
                             maxiter=200, xtol=1E-10, xs_guess=None,
                             max_step_damping=1e5, P_update_frequency=1,
                             trivial_solution_tol=1e-4):
    N = len(zs)
    cmps = range(N)
    xs = zs if xs_guess is None else xs_guess


    P_guess_old = None
    successive_fails = 0
    for iteration in range(maxiter):
        try:
            g = gas_phase = gas_phase.to_TP_zs(T=T, P=P_guess, zs=zs)
            lnphis_g = g.lnphis()
            dlnphis_dP_g = g.dlnphis_dP()
        except Exception as e:
            if P_guess_old is None:
                raise ValueError(g_undefined_P_msg %(P_guess, zs), e)
            successive_fails += 1
            P_guess = P_guess_old + copysign(min(max_step_damping, abs(step)), step)
            continue

        try:
            l = liquid_phase= liquid_phase.to_TP_zs(T=T, P=P_guess, zs=xs)
            lnphis_l = l.lnphis()
            dlnphis_dP_l = l.dlnphis_dP()
        except Exception as e:
            if P_guess_old is None:
                raise ValueError(l_undefined_P_msg %(P_guess, xs), e)
            successive_fails += 1
            T_guess = P_guess_old + copysign(min(max_step_damping, abs(step)), step)
            continue

        if successive_fails > 2:
            raise ValueError("Stopped convergence procedure after multiple bad steps")

        successive_fails = 0
        Ks = [exp(a - b) for a, b in zip(lnphis_l, lnphis_g)]
        xs = [zs[i]/Ks[i] for i in cmps]
        if iteration % P_update_frequency:
            continue

        f_k = sum(xs) - 1.0

        dfk_dP = 0.0
        for i in cmps:
            dfk_dP += xs[i]*(dlnphis_dP_g[i] - dlnphis_dP_l[i])

        P_guess_old = P_guess
        step = -f_k/dfk_dP

        P_guess = P_guess + copysign(min(max_step_damping, abs(step)), step)


        comp_difference = sum([abs(zi - xi) for zi, xi in zip(zs, xs)])
        if comp_difference < trivial_solution_tol:
            raise ValueError("Converged to trivial condition, compositions of both phases equal")

        x_sum_inv = 1.0/sum(xs)
        xs = [x*x_sum_inv for x in xs]

        if abs(P_guess - P_guess_old) < xtol:
            P_guess = P_guess_old
            break

    if abs(P_guess - P_guess_old) > xtol:
        raise ValueError("Did not converge to specified tolerance")
    return P_guess, xs, l, g, iteration, abs(P_guess - P_guess_old)


# spec, iter_var, fixed_var
strs_to_ders = {('H', 'T', 'P'): 'dH_dT_P',
                ('S', 'T', 'P'): 'dS_dT_P',
                ('G', 'T', 'P'): 'dG_dT_P',
                ('U', 'T', 'P'): 'dU_dT_P',
                ('A', 'T', 'P'): 'dA_dT_P',

                ('H', 'T', 'V'): 'dH_dT_V',
                ('S', 'T', 'V'): 'dS_dT_V',
                ('G', 'T', 'V'): 'dG_dT_V',
                ('U', 'T', 'V'): 'dU_dT_V',
                ('A', 'T', 'V'): 'dA_dT_V',

                ('H', 'P', 'T'): 'dH_dP_T',
                ('S', 'P', 'T'): 'dS_dP_T',
                ('G', 'P', 'T'): 'dG_dP_T',
                ('U', 'P', 'T'): 'dU_dP_T',
                ('A', 'P', 'T'): 'dA_dP_T',

                ('H', 'P', 'V'): 'dH_dP_V',
                ('S', 'P', 'V'): 'dS_dP_V',
                ('G', 'P', 'V'): 'dG_dP_V',
                ('U', 'P', 'V'): 'dU_dP_V',
                ('A', 'P', 'V'): 'dA_dP_V',

                ('H', 'V', 'T'): 'dH_dV_T',
                ('S', 'V', 'T'): 'dS_dV_T',
                ('G', 'V', 'T'): 'dG_dV_T',
                ('U', 'V', 'T'): 'dU_dV_T',
                ('A', 'V', 'T'): 'dA_dV_T',

                ('H', 'V', 'P'): 'dH_dV_P',
                ('S', 'V', 'P'): 'dS_dV_P',
                ('G', 'V', 'P'): 'dG_dV_P',
                ('U', 'V', 'P'): 'dU_dV_P',
                ('A', 'V', 'P'): 'dA_dV_P',
}


multiple_solution_sets = set([('T', 'S'), ('T', 'H'), ('T', 'U'), ('T', 'A'), ('T', 'G'),
                              ('S', 'T'), ('H', 'T'), ('U', 'T'), ('A', 'T'), ('G', 'T'),
                              ])

def TPV_solve_HSGUA_1P(zs, phase, guess, fixed_var_val, spec_val,
                       iter_var='T', fixed_var='P', spec='H',
                       maxiter=200, xtol=1E-10, ytol=None, fprime=False,
                       minimum_progress=0.3, oscillation_detection=True,
                       bounded=False, min_bound=None, max_bound=None,
                       multi_solution=False):
    r'''Solve a single-phase flash where one of `T`, `P`, or `V` are specified
    and one of `H`, `S`, `G`, `U`, or `A` are also specified. The iteration
    (changed input variable) variable must be specified as be one of `T`, `P`,
    or `V`, but it cannot be the same as the fixed variable.

    This method is a secant or newton based solution method, optionally with
    oscillation detection to bail out of tring to solve the problem to handle
    the case where the spec cannot be met because of a phase change (as in a
    cubic eos case).

    Parameters
    ----------
    zs : list[float]
        Mole fractions of the phase, [-]
    phase : `Phase`
        The phase object of the mixture, containing the information for
        calculating properties at new conditions, [-]
    guess : float
        The guessed value for the iteration variable,
        [K or Pa or m^3/mol]
    fixed_var_val : float
        The specified value of the fixed variable (one of T, P, or V);
        [K or Pa, or m^3/mol]
    spec_val : float
        The specified value of H, S, G, U, or A, [J/(mol*K) or J/mol]
    iter_var : str
        One of 'T', 'P', 'V', [-]
    fixed_var : str
        One of 'T', 'P', 'V', [-]
    spec : str
        One of 'H', 'S', 'G', 'U', 'A', [-]
    maxiter : float
        Maximum number of iterations, [-]
    xtol : float
        Tolerance for secant-style convergence of the iteration variable,
        [K or Pa, or m^3/mol]
    ytol : float or None
        Tolerance for convergence of the spec variable,
        [J/(mol*K) or J/mol]

    Returns
    -------
    iter_var_val, phase, iterations, err

    Notes
    -----

    '''
    # Needs lots of work but the idea is here
    # Can iterate chancing any of T, P, V with a fixed other T, P, V to meet any
    # H S G U A spec.
    store = []
    global iterations
    iterations = 0
    if fixed_var == iter_var:
        raise ValueError("Fixed variable cannot be the same as iteration variable")
    if fixed_var not in ('T', 'P', 'V'):
        raise ValueError("Fixed variable must be one of `T`, `P`, `V`")
    if iter_var not in ('T', 'P', 'V'):
        raise ValueError("Iteration variable must be one of `T`, `P`, `V`")
    # Little point in enforcing the spec - might want to repurpose the function later
    if spec not in ('H', 'S', 'G', 'U', 'A'):
        raise ValueError("Spec variable must be one of `H`, `S`, `G` `U`, `A`")

    multiple_solutions = (fixed_var, spec) in multiple_solution_sets

    phase_kwargs = {fixed_var: fixed_var_val, 'zs': zs}
    spec_fun = getattr(phase.__class__, spec)
#    print('spec_fun', spec_fun)
    if fprime:
        try:
            # Gotta be a lookup by (spec, iter_var, fixed_var)
            der_attr = strs_to_ders[(spec, iter_var, fixed_var)]
        except KeyError:
            der_attr = 'd' + spec + '_d' + iter_var
        der_attr_fun = getattr(phase.__class__, der_attr)
#        print('der_attr_fun', der_attr_fun)
    def to_solve(guess, solved_phase=None):
        global iterations
        iterations += 1

        if solved_phase is not None:
            p = solved_phase
        else:
            phase_kwargs[iter_var] = guess
            p = phase.to(**phase_kwargs)

        err = spec_fun(p) - spec_val
#        err = (spec_fun(p) - spec_val)/spec_val
        store[:] = (p, err)
        if fprime:
#            print([err, guess, p.eos_mix.phase, der_attr])
            derr = der_attr_fun(p)
#            derr = der_attr_fun(p)/spec_val
            return err, derr
#        print(err)
        return err

    arg_fprime = fprime
    high = None # Optional and not often used bound for newton
    if fixed_var == 'V':
        if iter_var == 'T':
            max_phys = phase.T_max_at_V(fixed_var_val)
        elif iter_var == 'P':
            max_phys = phase.P_max_at_V(fixed_var_val)
        if max_phys is not None:
            if max_bound is None:
                max_bound = high = max_phys
            else:
                max_bound = high = min(max_phys, max_bound)

    # TV iterations
    ignore_bound_fail = (fixed_var == 'T' and iter_var == 'P')

    if fixed_var in ('T',) and ((fixed_var == 'T' and iter_var == 'P') or (fixed_var == 'P' and iter_var == 'T') or (fixed_var == 'T' and iter_var == 'V') ) and 1:
        try:
            fprime = False
            if iter_var == 'V':
                dummy_iter = 1e8
            else:
                dummy_iter = guess
            phase_kwargs[iter_var] = dummy_iter # Dummy pressure does not matter
            phase_temp = phase.to(**phase_kwargs)

            lower_phase, higher_phase = None, None
            delta = 1e-9
            if fixed_var == 'T' and iter_var == 'P':
                transitions = phase_temp.P_transitions()
                # assert len(transitions) == 1
                under_trans, above_trans = transitions[0] * (1.0 - delta), transitions[0] * (1.0 + delta)
            elif fixed_var == 'P' and iter_var == 'T':
                transitions = phase_temp.T_transitions()
                under_trans, above_trans = transitions[0] * (1.0 - delta), transitions[0] * (1.0 + delta)
                assert len(transitions) == 1

            elif fixed_var == 'T' and iter_var == 'V':
                transitions = phase_temp.P_transitions()
                delta = 1e-11
                # not_separated = True
                # while not_separated:
                P_higher = transitions[0]*(1.0 + delta)  # Dummy pressure does not matter
                lower_phase = phase.to(T=fixed_var_val, zs=zs, P=P_higher)
                P_lower = transitions[0]*(1.0 - delta)  # Dummy pressure does not matter
                higher_phase = phase.to(T=fixed_var_val, zs=zs, P=P_lower)
                under_trans, above_trans = lower_phase.V(), higher_phase.V()
                not_separated = isclose(under_trans, above_trans, rel_tol=1e-3)
                # delta *= 10

            # TODO is it possible to evaluate each limit at once, so half the work is avoided?

            bracketed_high, bracketed_low = False, False
            if min_bound is not None:
                f_min = to_solve(min_bound)
                f_low_trans = to_solve(under_trans, lower_phase)
                if f_min*f_low_trans <= 0.0:
                    bracketed_low = True
                    bounding_pair = (min(min_bound, under_trans), max(min_bound, under_trans))
            if max_bound is not None and (not bracketed_low or multiple_solutions):
                f_max = to_solve(max_bound)
                f_max_trans = to_solve(above_trans, higher_phase)
                if f_max*f_max_trans <= 0.0:
                    bracketed_high = True
                    bounding_pair = (min(max_bound, above_trans), max(max_bound, above_trans))

            if max_bound is not None and max_bound is not None and not bracketed_low and not bracketed_high:
                if not ignore_bound_fail:
                    raise NotBoundedError("Between phases")

            if bracketed_high or bracketed_low:
                oscillation_detection = False
                high = bounding_pair[1] # restrict newton/secant just in case
                min_bound, max_bound = bounding_pair
                if not (min_bound < guess < max_bound):
                    guess = 0.5*(min_bound + max_bound)
            else:
                if min_bound is not None and transitions[0] < min_bound and not ignore_bound_fail:
                    raise NotBoundedError("Not likely to bound")
                if max_bound is not None and transitions[0] > max_bound and not ignore_bound_fail:
                    raise NotBoundedError("Not likely to bound")



        except NotBoundedError as e:
            raise e
        except Exception:
            pass

    fprime = arg_fprime

    # Plot the objective function
    # tests = logspace(log10(10.6999), log10(10.70005), 15000)
    # tests = logspace(log10(10.6), log10(10.8), 15000)
    # tests = logspace(log10(min_bound), log10(max_bound), 1500)
    # values = [to_solve(t)[0] for t in tests]
    # values = [abs(t) for t in values]
    # import matplotlib.pyplot as plt
    # plt.loglog(tests, values)
    # plt.show()

    if oscillation_detection and ytol is not None:
        to_solve2, checker = oscillation_checking_wrapper(to_solve, full=True,
                                                          minimum_progress=minimum_progress,
                                                          good_err=ytol*1e6)
    else:
        to_solve2 = to_solve
        checker = None
    solve_bounded = False

    try:
        # All three variables P, T, V are positive but can grow unbounded, so
        # for the secant method, only set the one variable
        if fprime:
            iter_var_val = newton(to_solve2, guess, xtol=xtol, ytol=ytol, fprime=True,
                                  maxiter=maxiter, bisection=True, low=min_bound, high=high, gap_detection=False)
        else:
            iter_var_val = secant(to_solve2, guess, xtol=xtol, ytol=ytol,
                                  maxiter=maxiter, bisection=True, low=min_bound, high=high)
    except (UnconvergedError, OscillationError, NotBoundedError):
        solve_bounded = True
        # Unconverged - from newton/secant; oscillation - from the oscillation detector;
        # NotBounded - from when EOS needs to solve T and there is no solution
    fprime = False
    if solve_bounded:
        if bounded and min_bound is not None and max_bound is not None:
            if checker:
                min_bound_prev, max_bound_prev, fa, fb = best_bounding_bounds(min_bound, max_bound,
                                                                    f=to_solve, xs_pos=checker.xs_pos, ys_pos=checker.ys_pos,
                                                                    xs_neg=checker.xs_neg, ys_neg=checker.ys_neg)
                if abs(min_bound_prev/max_bound_prev - 1.0) > 2.5e-4:
                    # If the points are too close, odds are there is a discontinuity in the newton solution
                    min_bound, max_bound = min_bound_prev, max_bound_prev
#                    maxiter = 20
                else:
                    fa, fb = None, None

            else:
                fa, fb = None, None

            # try:
            iter_var_val = brenth(to_solve, min_bound, max_bound, xtol=xtol,
                                  ytol=ytol, maxiter=maxiter, fa=fa, fb=fb)
            # except:
            #     # Not sure at all if good idea
            #     iter_var_val = secant(to_solve, guess, xtol=xtol, ytol=ytol,
            #                           maxiter=maxiter, bisection=True, low=min_bound)
    phase, err = store

    return iter_var_val, phase, iterations, err



def solve_PTV_HSGUA_1P(phase, zs, fixed_var_val, spec_val, fixed_var,
                       spec, iter_var, constants, correlations, last_conv=None,
                       oscillation_detection=True, guess_maxiter=50,
                       guess_xtol=1e-7, maxiter=80, xtol=1e-10):
    # TODO: replace oscillation detection with bounding parameters and translation
    # The cost should be less.

    if iter_var == 'T':
        if isinstance(phase, CoolPropPhase):
            min_bound = phase.AS.Tmin()
            max_bound = phase.AS.Tmax()
        else:
            min_bound = phase.T_MIN_FIXED
            max_bound = phase.T_MAX_FIXED
#        if isinstance(phase, IAPWS95):
#            min_bound = 235.0
#            max_bound = 5000.0
    elif iter_var == 'P':
        min_bound = Phase.P_MIN_FIXED*(1.0 - 1e-12)
        max_bound = Phase.P_MAX_FIXED*(1.0 + 1e-12)
        if isinstance(phase, CoolPropPhase):
            AS = phase.AS
            max_bound = AS.pmax()*(1.0 - 1e-7)
            min_bound = AS.trivial_keyed_output(CPiP_min)*(1.0 + 1e-7)
    elif iter_var == 'V':
        min_bound = Phase.V_MIN_FIXED
        max_bound = Phase.V_MAX_FIXED
        if isinstance(phase, (CEOSLiquid, CEOSGas)):
            c2R = phase.eos_class.c2*R
            Tcs, Pcs = constants.Tcs, constants.Pcs
            b = sum([c2R*Tcs[i]*zs[i]/Pcs[i] for i in range(constants.N)])
            min_bound = b*(1.0 + 1e-15)

    if phase.is_gas:
        methods = [LAST_CONVERGED, FIXED_GUESS, STP_T_GUESS, IG_ENTHALPY,
                   LASTOVKA_SHAW]
    elif phase.is_liquid:
        methods = [LAST_CONVERGED, FIXED_GUESS, STP_T_GUESS, IDEAL_LIQUID_ENTHALPY,
                   DADGOSTAR_SHAW_1]
    else:
        methods = [LAST_CONVERGED, FIXED_GUESS, STP_T_GUESS]

    for method in methods:
        try:
            guess = TPV_solve_HSGUA_guesses_1P(zs, method, constants, correlations,
                               fixed_var_val, spec_val,
                               iter_var=iter_var, fixed_var=fixed_var, spec=spec,
                               maxiter=guess_maxiter, xtol=guess_xtol, ytol=abs(spec_val)*1e-5,
                               bounded=True, min_bound=min_bound, max_bound=max_bound,
                               user_guess=None, last_conv=last_conv, T_ref=298.15,
                               P_ref=101325.0)

            break
        except Exception:
            pass

    ytol = 1e-8*abs(spec_val)

    if iter_var == 'T' and spec in ('S', 'H'):
        ytol = ytol/100
    if isinstance(phase, IAPWS95):
        # Objective function isn't quite as nice and smooth as desired
        ytol = None

    _, phase, iterations, err = TPV_solve_HSGUA_1P(zs, phase, guess, fixed_var_val=fixed_var_val, spec_val=spec_val, ytol=ytol,
                                                   iter_var=iter_var, fixed_var=fixed_var, spec=spec, oscillation_detection=oscillation_detection,
                                                   minimum_progress=1e-4, maxiter=maxiter, fprime=True, xtol=xtol,
                                                   bounded=True, min_bound=min_bound, max_bound=max_bound)
    T, P = phase.T, phase.P
    return T, P, phase, iterations, err

def TPV_solve_HSGUA_guesses_1P(zs, method, constants, correlations,
                               fixed_var_val, spec_val,
                               iter_var='T', fixed_var='P', spec='H',
                               maxiter=20, xtol=1E-7, ytol=None,
                               bounded=False, min_bound=None, max_bound=None,
                               user_guess=None, last_conv=None, T_ref=298.15,
                               P_ref=101325.0):
    if fixed_var == iter_var:
        raise ValueError("Fixed variable cannot be the same as iteration variable")
    if fixed_var not in ('T', 'P', 'V'):
        raise ValueError("Fixed variable must be one of `T`, `P`, `V`")
    if iter_var not in ('T', 'P', 'V'):
        raise ValueError("Iteration variable must be one of `T`, `P`, `V`")
    if spec not in ('H', 'S', 'G', 'U', 'A'):
        raise ValueError("Spec variable must be one of `H`, `S`, `G` `U`, `A`")

    cmps = range(len(zs))

    iter_T = iter_var == 'T'
    iter_P = iter_var == 'P'
    iter_V = iter_var == 'V'

    fixed_P = fixed_var == 'P'
    fixed_T = fixed_var == 'T'
    fixed_V = fixed_var == 'V'

    always_S = spec in ('S', 'G', 'A')
    always_H = spec in ('H', 'G', 'U', 'A')
    always_V = spec in ('U', 'A')


    if always_S:
        P_ref_inv = 1.0/P_ref
        dS_ideal = R*sum([zi*log(zi) for zi in zs if zi > 0.0]) # ideal composition entropy composition

    def err(guess):
        # Translate the fixed variable to a local variable
        if fixed_P:
            P = fixed_var_val
        elif fixed_T:
            T = fixed_var_val
        elif fixed_V:
            V = fixed_var_val
            T = None
        # Translate the iteration variable to a local variable
        if iter_P:
            P = guess
            if not fixed_V:
                V = None
        elif iter_T:
            T = guess
            if not fixed_V:
                V = None
        elif iter_V:
            V = guess
            T = None

        if T is None:
            T = T_from_V(V, P)

        # Compute S, H, V as necessary
        if always_S:
            S = S_model(T, P) - dS_ideal - R*log(P*P_ref_inv)
        if always_H:
            H =  H_model(T, P)
        if always_V and V is None:
            V = V_model(T, P)
#        print(H, S, V, 'hi')
        # Return the objective function
        if spec == 'H':
            err = H - spec_val
        elif spec == 'S':
            err = S - spec_val
        elif spec == 'G':
            err = (H - T*S) - spec_val
        elif spec == 'U':
            err = (H - P*V) - spec_val
        elif spec == 'A':
            err = (H - P*V - T*S) - spec_val
#        print(T, P, V, 'TPV', err)
        return err

    # Precompute some things depending on the method
    if method in (LASTOVKA_SHAW, DADGOSTAR_SHAW_1):
        MW = mixing_simple(zs, constants.MWs)
        n_atoms = [sum(i.values()) for i in constants.atomss]
        sv = mixing_simple(zs, n_atoms)/MW

    if method == IG_ENTHALPY:
        HeatCapacityGases = correlations.HeatCapacityGases
        def H_model(T, P=None):
            H_calc = 0.
            for i in cmps:
                H_calc += zs[i]*HeatCapacityGases[i].T_dependent_property_integral(T_ref, T)
            return H_calc

        def S_model(T, P=None):
            S_calc = 0.
            for i in cmps:
                S_calc += zs[i]*HeatCapacityGases[i].T_dependent_property_integral_over_T(T_ref, T)
            return S_calc

        def V_model(T, P):  return R*T/P
        def T_from_V(V, P): return P*V/R

    elif method == LASTOVKA_SHAW:
        H_ref = Lastovka_Shaw_integral(T_ref, sv)
        S_ref = Lastovka_Shaw_integral_over_T(T_ref, sv)

        def H_model(T, P=None):
            H1 = Lastovka_Shaw_integral(T, sv)
            dH = H1 - H_ref
            return property_mass_to_molar(dH, MW)

        def S_model(T, P=None):
            S1 = Lastovka_Shaw_integral_over_T(T, sv)
            dS = S1 - S_ref
            return property_mass_to_molar(dS, MW)

        def V_model(T, P):  return R*T/P
        def T_from_V(V, P): return P*V/R

    elif method == DADGOSTAR_SHAW_1:
        Tc = mixing_simple(zs, constants.Tcs)
        omega = mixing_simple(zs, constants.omegas)
        H_ref = Dadgostar_Shaw_integral(T_ref, sv)
        S_ref = Dadgostar_Shaw_integral_over_T(T_ref, sv)

        def H_model(T, P=None):
            H1 = Dadgostar_Shaw_integral(T, sv)
            Hvap = SMK(T, Tc, omega)
            return (property_mass_to_molar(H1 - H_ref, MW) - Hvap)

        def S_model(T, P=None):
            S1 = Dadgostar_Shaw_integral_over_T(T, sv)
            dSvap = SMK(T, Tc, omega)/T
            return (property_mass_to_molar(S1 - S_ref, MW) - dSvap)

        Vc = mixing_simple(zs, constants.Vcs)
        def V_model(T, P=None): return COSTALD(T, Tc, Vc, omega)
        def T_from_V(V, P): secant(lambda T: COSTALD(T, Tc, Vc, omega), .65*Tc)

    elif method == IDEAL_LIQUID_ENTHALPY:
        HeatCapacityGases = correlations.HeatCapacityGases
        EnthalpyVaporizations = correlations.EnthalpyVaporizations
        def H_model(T, P=None):
            H_calc = 0.
            for i in cmps:
                H_calc += zs[i]*(HeatCapacityGases[i].T_dependent_property_integral(T_ref, T) - EnthalpyVaporizations[i](T))
            return H_calc

        def S_model(T, P=None):
            S_calc = 0.
            T_inv = 1.0/T
            for i in cmps:
                S_calc += zs[i]*(HeatCapacityGases[i].T_dependent_property_integral_over_T(T_ref, T) - T_inv*EnthalpyVaporizations[i](T))
            return S_calc

        VolumeLiquids = correlations.VolumeLiquids
        def V_model(T, P=None):
            V_calc = 0.
            for i in cmps:
                V_calc += zs[i]*VolumeLiquids[i].T_dependent_property(T)
            return V_calc
        def T_from_V(V, P):
            T_calc = 0.
            for i in cmps:
                T_calc += zs[i]*VolumeLiquids[i].solve_property(V)
            return T_calc


    # Simple return values - not going through a model
    if method == STP_T_GUESS:
        if iter_T:
            return 298.15
        elif iter_P:
            return 101325.0
        elif iter_V:
            return 0.024465403697038125
    elif method == LAST_CONVERGED:
        if last_conv is None:
            raise ValueError("No last converged")
        return last_conv
    elif method == FIXED_GUESS:
        if user_guess is None:
            raise ValueError("No user guess")
        return user_guess

    try:
        # All three variables P, T, V are positive but can grow unbounded, so
        # for the secant method, only set the one variable
        if iter_T:
            guess = 298.15
        elif iter_P:
            guess = 101325.0
        elif iter_V:
            guess = 0.024465403697038125
        return secant(err, guess, xtol=xtol, ytol=ytol,
                      maxiter=maxiter, bisection=True, low=min_bound)
    except (UnconvergedError,):
        # G and A specs are NOT MONOTONIC and the brackets will likely NOT BRACKET
        # THE ROOTS!
        return brenth(err, min_bound, max_bound, xtol=xtol, ytol=ytol, maxiter=maxiter)


def PH_secant_1P(T_guess, P, H, zs, phase, maxiter=200, xtol=1E-10,
                 minimum_progress=0.3, oscillation_detection=True):
    store = []
    global iterations
    iterations = 0
    def to_solve(T):
        global iterations
        iterations += 1
        p = phase.to_TP_zs(T, P, zs)

        err = p.H() - H
        store[:] = (p, err)
        return err
    if oscillation_detection:
        to_solve, checker = oscillation_checking_wrapper(to_solve, full=True,
                                                         minimum_progress=minimum_progress)

    T = secant(to_solve, T_guess, xtol=xtol, maxiter=maxiter)
    phase, err = store

    return T, phase, iterations, err

def PH_newton_1P(T_guess, P, H, zs, phase, maxiter=200, xtol=1E-10,
                 minimum_progress=0.3, oscillation_detection=True):
    store = []
    global iterations
    iterations = 0
    def to_solve(T):
        global iterations
        iterations += 1
        p = phase.to_TP_zs(T, P, zs)

        err = p.H() - H
        derr_dT = p.dH_dT()
        store[:] = (p, err)
        return err, derr_dT
    if oscillation_detection:
        to_solve, checker = oscillation_checking_wrapper(to_solve, full=True,
                                                         minimum_progress=minimum_progress)

    T = newton(to_solve, T_guess, fprime=True, xtol=xtol, maxiter=maxiter)
    phase, err = store

    return T, phase, iterations, err




def TVF_pure_newton(P_guess, T, liquids, gas, maxiter=200, xtol=1E-10):
    one_liquid = len(liquids)
    zs = [1.0]
    store = []
    global iterations
    iterations = 0
    def to_solve_newton(P):
        global iterations
        iterations += 1
        g = gas.to_TP_zs(T, P, zs)
        fugacity_gas = g.fugacities()[0]
        dfugacities_dP_gas = g.dfugacities_dP()[0]

        if one_liquid:
            lowest_phase = liquids[0].to_TP_zs(T, P, zs)
        else:
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
        store[:] = (lowest_phase, g, err)
        return err, derr_dP
    Psat = newton(to_solve_newton, P_guess, xtol=xtol, maxiter=maxiter,
                  low=Phase.P_MIN_FIXED,
                  require_eval=True, bisection=False, fprime=True)
    l, g, err = store

    return Psat, l, g, iterations, err

def TVF_pure_secant(P_guess, T, liquids, gas, maxiter=200, xtol=1E-10):
    one_liquid = len(liquids)
    zs = [1.0]
    store = []
    global iterations
    iterations = 0
    def to_solve_secant(P):
        global iterations
        iterations += 1
        g = gas.to_TP_zs(T, P, zs)
        fugacity_gas = g.fugacities()[0]

        if one_liquid:
            lowest_phase = liquids[0].to_TP_zs(T, P, zs)
        else:
            ls = [l.to_TP_zs(T, P, zs) for l in liquids]
            G_min, lowest_phase = 1e100, None
            for l in ls:
                G = l.G()
                if G < G_min:
                    G_min, lowest_phase = G, l

        fugacity_liq = lowest_phase.fugacities()[0]

        err = fugacity_liq - fugacity_gas
        store[:] = (lowest_phase, g, err)
        return err
    if P_guess <  Phase.P_MIN_FIXED:
        raise ValueError("Too low.")
    # if P_guess <  Phase.P_MIN_FIXED:
    #     low = None
    # else:
    #     low = Phase.P_MIN_FIXED
    Psat = secant(to_solve_secant, P_guess, xtol=xtol, maxiter=maxiter, low=Phase.P_MIN_FIXED*(1-1e-10))
    l, g, err = store

    return Psat, l, g, iterations, err


def PVF_pure_newton(T_guess, P, liquids, gas, maxiter=200, xtol=1E-10):
    one_liquid = len(liquids)
    zs = [1.0]
    store = []
    global iterations
    iterations = 0
    def to_solve_newton(T):
        global iterations
        iterations += 1
        g = gas.to_TP_zs(T, P, zs)
        fugacity_gas = g.fugacities()[0]
        dfugacities_dT_gas = g.dfugacities_dT()[0]

        if one_liquid:
            lowest_phase = liquids[0].to_TP_zs(T, P, zs)
        else:
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
        store[:] = (lowest_phase, g, err)
        return err, derr_dT
    Tsat = newton(to_solve_newton, T_guess, xtol=xtol, maxiter=maxiter,
                  low=Phase.T_MIN_FIXED,
                  require_eval=True, bisection=False, fprime=True)
    l, g, err = store

    return Tsat, l, g, iterations, err


def PVF_pure_secant(T_guess, P, liquids, gas, maxiter=200, xtol=1E-10):
    one_liquid = len(liquids)
    zs = [1.0]
    store = []
    global iterations
    iterations = 0
    def to_solve_secant(T):
        global iterations
        iterations += 1
        g = gas.to_TP_zs(T, P, zs)
        fugacity_gas = g.fugacities()[0]

        if one_liquid:
            lowest_phase = liquids[0].to_TP_zs(T, P, zs)
        else:
            ls = [l.to_TP_zs(T, P, zs) for l in liquids]
            G_min, lowest_phase = 1e100, None
            for l in ls:
                G = l.G()
                if G < G_min:
                    G_min, lowest_phase = G, l

        fugacity_liq = lowest_phase.fugacities()[0]

        err = fugacity_liq - fugacity_gas
        store[:] = (lowest_phase, g, err)
        return err
    Tsat = secant(to_solve_secant, T_guess, xtol=xtol, maxiter=maxiter,
                  low=Phase.T_MIN_FIXED)
    l, g, err = store

    return Tsat, l, g, iterations, err


def TSF_pure_newton(P_guess, T, other_phases, solids, maxiter=200, xtol=1E-10):
    one_other = len(other_phases)
    one_solid = len(solids)
    zs = [1.0]
    store = []
    global iterations
    iterations = 0
    def to_solve_newton(P):
        global iterations
        iterations += 1
        if one_solid:
            lowest_solid = solids[0].to_TP_zs(T, P, zs)
        else:
            ss = [s.to_TP_zs(T, P, zs) for s in solids]
            G_min, lowest_solid = 1e100, None
            for o in ss:
                G = o.G()
                if G < G_min:
                    G_min, lowest_solid = G, o

        fugacity_solid = lowest_solid.fugacities()[0]
        dfugacities_dP_solid = lowest_solid.dfugacities_dP()[0]

        if one_other:
            lowest_other = other_phases[0].to_TP_zs(T, P, zs)
        else:
            others = [l.to_TP_zs(T, P, zs) for l in other_phases]
            G_min, lowest_other = 1e100, None
            for o in others:
                G = o.G()
                if G < G_min:
                    G_min, lowest_other = G, o

        fugacity_other = lowest_other.fugacities()[0]
        dfugacities_dP_other = lowest_other.dfugacities_dP()[0]

        err = fugacity_other - fugacity_solid
        derr_dP = dfugacities_dP_other - dfugacities_dP_solid
        store[:] = (lowest_other, lowest_solid, err)
        return err, derr_dP

    Psub = newton(to_solve_newton, P_guess, xtol=xtol, maxiter=maxiter,
                  require_eval=True, bisection=False, fprime=True)
    other, solid, err = store

    return Psub, other, solid, iterations, err

def PSF_pure_newton(T_guess, P, other_phases, solids, maxiter=200, xtol=1E-10):
    one_other = len(other_phases)
    one_solid = len(solids)
    zs = [1.0]
    store = []
    global iterations
    iterations = 0
    def to_solve_newton(T):
        global iterations
        iterations += 1
        if one_solid:
            lowest_solid = solids[0].to_TP_zs(T, P, zs)
        else:
            ss = [s.to_TP_zs(T, P, zs) for s in solids]
            G_min, lowest_solid = 1e100, None
            for o in ss:
                G = o.G()
                if G < G_min:
                    G_min, lowest_solid = G, o

        fugacity_solid = lowest_solid.fugacities()[0]
        dfugacities_dT_solid = lowest_solid.dfugacities_dT()[0]

        if one_other:
            lowest_other = other_phases[0].to_TP_zs(T, P, zs)
        else:
            others = [l.to_TP_zs(T, P, zs) for l in other_phases]
            G_min, lowest_other = 1e100, None
            for o in others:
                G = o.G()
                if G < G_min:
                    G_min, lowest_other = G, o

        fugacity_other = lowest_other.fugacities()[0]
        dfugacities_dT_other = lowest_other.dfugacities_dT()[0]

        err = fugacity_other - fugacity_solid
        derr_dT = dfugacities_dT_other - dfugacities_dT_solid
        store[:] = (lowest_other, lowest_solid, err)
        return err, derr_dT

    Tsub = newton(to_solve_newton, T_guess, xtol=xtol, maxiter=maxiter,
                  require_eval=True, bisection=False, fprime=True)
    other, solid, err = store

    return Tsub, other, solid, iterations, err


def solve_T_VF_IG_K_composition_independent(VF, T, zs, gas, liq, xtol=1e-10):
    '''from sympy import *
    zi, P, VF = symbols('zi, P, VF')
    l_phi, g_phi = symbols('l_phi, g_phi', cls=Function)
    # g_phi = symbols('g_phi')
    # Ki = l_phi(P)/g_phi(P)
    Ki = l_phi(P)#/g_phi
    err = zi*(Ki-1)/(1+VF*(Ki-1))
    cse([diff(err, P), err], optimizations='basic')'''
    # gas phis are all one in IG model
#     gas.to(T=T, P=P, zs=zs)
    cmps = range(liq.N)
    global Ks, iterations, err
    iterations = 0
    err = 0.0
    def to_solve(lnP):
        global Ks, iterations, err
        iterations += 1
        P = exp(lnP)
        l = liq.to(T=T, P=P, zs=zs)
        Ks = liquid_phis = l.phis()
        dlnphis_dP_l = l.dphis_dP()

        err = derr = 0.0
        for i in cmps:
            x1 = liquid_phis[i] - 1.0
            x2 = VF*x1
            x3 = 1.0/(x2 + 1.0)
            x4 = x3*zs[i]
            err += x1*x4
            derr += x4*(1.0 - x2*x3)*dlnphis_dP_l[i]
        return err, P*derr

    # estimate bubble point and dew point
    # Make sure to overwrite the phase so the Psats get cached
    P_base = 1e5
    liq = liq.to(T=T, P=P_base, zs=zs)
    phis = liq.phis()
    P_bub, P_dew = 0.0, 0.0
    for i in range(liq.N):
        P_bub += phis[i]*zs[i]
        P_dew += zs[i]/(phis[i]*P_base)
    P_bub = P_bub*liq.P
    P_dew = 1.0/P_dew
    P_guess = VF*P_dew + (1.0 - VF)*P_bub

    # When Poynting is on, the are only an estimate; otherwise it is dead on
    # and there is no need for a solver
    if liq.use_Poynting or 0.0 < VF < 1.0:
        lnP = newton(to_solve, log(P_guess), xtol=xtol, fprime=True)
        P = exp(lnP)
    else:
        if VF == 0.0:
            Ks = liq.to(T=T, P=P_bub, zs=zs).phis()
            P = P_bub
        elif VF == 1.0:
            Ks = liq.to(T=T, P=P_dew, zs=zs).phis()
            P = P_dew
        else:
            raise ValueError("Vapor fraction outside range 0 to 1")
    xs = [zs[i]/(1.+VF*(Ks[i]-1.)) for i in cmps]
    for i in cmps:
        Ks[i] *= xs[i]
    ys = Ks
    return P, xs, ys, iterations, err

def solve_P_VF_IG_K_composition_independent(VF, P, zs, gas, liq, xtol=1e-10):
    # gas phis are all one in IG model
#     gas.to(T=T, P=P, zs=zs)
    cmps = range(liq.N)
    global Ks, iterations, err
    iterations = 0
    def to_solve(T):
        global Ks, iterations, err
        iterations += 1
        dlnphis_dT_l, liquid_phis = liq.dphis_dT_at(T, P, zs, phis_also=True)
        Ks = liquid_phis
#        l = liq.to(T=T, P=P, zs=zs)
#        Ks = liquid_phis = l.phis()
#        dlnphis_dT_l = l.dphis_dT()
        err = derr = 0.0
        for i in cmps:
            x1 = liquid_phis[i] - 1.0
            x2 = VF*x1
            x3 = 1.0/(x2 + 1.0)
            x4 = x3*zs[i]
            err += x1*x4
            derr += x4*(1.0 - x2*x3)*dlnphis_dT_l[i]
        return err, derr
    try:
        T = newton(to_solve, 300.0, xtol=xtol, fprime=True, low=1e-6)
    except:
        try:
            T = brenth(lambda x: to_solve(x)[0], 300, 1000)
        except:
            T = newton(to_solve, 400.0, xtol=xtol, fprime=True, low=1e-6)
    xs = [zs[i]/(1.+VF*(Ks[i]-1.)) for i in cmps]
    for i in cmps:
        Ks[i] *= xs[i]
    ys = Ks
    return T, xs, ys, iterations, err

def sequential_substitution_2P_sat(T, P, V, zs_dry, xs_guess, ys_guess, liquid_phase,
                                   gas_phase, idx, z0, z1=None, maxiter=1000, tol=1E-13,
                                   trivial_solution_tol=1e-5, damping=1.0):
    xs, ys = xs_guess, ys_guess
    V_over_F = 1.0
    cmps = range(len(zs_dry))

    if z1 is None:
        z1 = z0*1.0001 + 1e-4
        if z1 > 1:
            z1 = z0*1.0001 - 1e-4

    # secant step/solving
    p0, p1, err0, err1 = None, None, None, None
    def step(p0, p1, err0, err1):
        if p0 is None:
            return z0
        if p1 is None:
            return z1
        else:
            new = p1 - err1*(p1 - p0)/(err1 - err0)*damping
            return new


    for iteration in range(maxiter):
        p0, p1 = step(p0, p1, err0, err1), p0
        zs = list(zs_dry)
        zs[idx] = p0
        zs = normalize(zs)
#         print(zs, p0, p1)

        g = gas_phase.to(ys, T=T, P=P, V=V)
        l = liquid_phase.to(xs, T=T, P=P, V=V)
        lnphis_g = g.lnphis()
        lnphis_l = l.lnphis()

        Ks = [exp(lnphis_l[i] - lnphis_g[i]) for i in cmps]

        V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F)
        err0, err1 = 1.0 - V_over_F, err0

        # Check for negative fractions - normalize only if needed
        for xi in xs_new:
            if xi < 0.0:
                xs_new_sum = sum(abs(i) for i in xs_new)
                xs_new = [abs(i)/xs_new_sum for i in xs_new]
                break
        for yi in ys_new:
            if yi < 0.0:
                ys_new_sum = sum(abs(i) for i in ys_new)
                ys_new = [abs(i)/ys_new_sum for i in ys_new]
                break

        err, comp_diff = 0.0, 0.0
        for i in cmps:
            err_i = Ks[i]*xs[i]/ys[i] - 1.0
            err += err_i*err_i + abs(ys[i] - zs[i])
            comp_diff += abs(xs[i] - ys[i])

        # Accept the new compositions
#         xs, ys = xs_new, zs # This has worse convergence behavior?
        xs, ys = xs_new, ys_new

        if comp_diff < trivial_solution_tol:
            raise ValueError("Converged to trivial condition, compositions of both phases equal")

        if err < tol and abs(err0) < tol:
            return V_over_F, xs, zs, l, g, iteration, err, err0
    raise UnconvergedError('End of SS without convergence')

def SS_VF_simultaneous(guess, fixed_val, zs, liquid_phase, gas_phase,
                                iter_var='T', fixed_var='P', V_over_F=1,
                                maxiter=200, xtol=1E-10, comp_guess=None,
                                damping=0.8, tol_eq=1e-12, update_frequency=3):
    if comp_guess is None:
        comp_guess = zs

    if V_over_F == 1 or V_over_F > 0.5:
        dew = True
        xs, ys = comp_guess, zs
    else:
        dew = False
        xs, ys = zs, comp_guess

    sln = sequential_substitution_2P_HSGUAbeta(zs=zs, xs_guess=xs, ys_guess=ys, liquid_phase=liquid_phase,
                                     gas_phase=gas_phase, fixed_var_val=fixed_val, spec_val=V_over_F, tol_spec=xtol,
                                     iter_var_0=guess, update_frequency=update_frequency,
                                     iter_var=iter_var, fixed_var=fixed_var, spec='beta', damping=damping, tol_eq=tol_eq)
    guess, _, xs, ys, l, g, iteration, err_eq, spec_err = sln

    if dew:
        comp_guess = xs
        iter_phase, const_phase = l, g
    else:
        comp_guess = ys
        iter_phase, const_phase = g, l

    return guess, comp_guess, iter_phase, const_phase, iteration, {'err_eq': err_eq, 'spec_err': spec_err}


def sequential_substitution_2P_HSGUAbeta(zs, xs_guess, ys_guess, liquid_phase,
                                     gas_phase, fixed_var_val, spec_val,
                                     iter_var_0, iter_var_1=None,
                                     iter_var='T', fixed_var='P', spec='H',
                                     maxiter=1000, tol_eq=1E-13, tol_spec=1e-9,
                                     trivial_solution_tol=1e-5, damping=1.0,
                                     V_over_F_guess=None, fprime=True,
                                     update_frequency=1, update_eq=1e-7):
    xs, ys = xs_guess, ys_guess
    if V_over_F_guess is None:
        V_over_F = 0.5
    else:
        V_over_F = V_over_F_guess

    cmps = range(len(zs))

    if iter_var_1 is None:
        iter_var_1 = iter_var_0*1.0001 + 1e-4

    tol_spec_abs = tol_spec*abs(spec_val)
    if tol_spec_abs == 0.0:
        if spec == 'beta':
            tol_spec_abs = 1e-9
        else:
            tol_spec_abs = 1e-7

    # secant step/solving
    p0, p1, spec_err, spec_err_old = None, None, None, None
    def step(p0, p1, spec_err, spec_err_old, step_der):
        if p0 is None:
            return iter_var_0
        if p1 is None:
            return iter_var_1
        else:
            secant_step = spec_err_old*(p1 - p0)/(spec_err_old - spec_err)*damping
            if fprime and step_der is not None:
                if abs(step_der) < abs(secant_step):
                    step = step_der
                    new = p0 - step
                else:
                    step = secant_step
                    new = p1 - step
            else:
                new = p1 - secant_step
            if new < 1e-7:
                # Only handle positive values, damped steps to .5
                new = 0.5*(1e-7 + p0)
#            print(p0, p1, new)
            return new

    TPV_args = {fixed_var: fixed_var_val, iter_var: iter_var_0}

    VF_spec = spec == 'beta'
    if not VF_spec:
        spec_fun_l = getattr(liquid_phase.__class__, spec)
        spec_fun_g = getattr(gas_phase.__class__, spec)

        s_der = 'd%s_d%s_%s'%(spec, iter_var, fixed_var)
        spec_der_fun_l = getattr(liquid_phase.__class__, s_der)
        spec_der_fun_g = getattr(gas_phase.__class__, s_der)
    else:
        V_over_F = iter_var_0

    step_der = None
    for iteration in range(maxiter):
        if (not (iteration % update_frequency) or err_eq < update_eq) or iteration < 2:
            p0, p1 = step(p0, p1, spec_err, spec_err_old, step_der), p0
        TPV_args[iter_var] = p0

        g = gas_phase.to(ys, **TPV_args)
        l = liquid_phase.to(xs, **TPV_args)
        lnphis_g = g.lnphis()
        lnphis_l = l.lnphis()

        Ks = [exp(lnphis_l[i] - lnphis_g[i]) for i in cmps]

        V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F)

        if not VF_spec:
            spec_calc = spec_fun_l(l)*(1.0 - V_over_F) + spec_fun_g(g)*V_over_F
            spec_der_calc = spec_der_fun_l(l)*(1.0 - V_over_F) + spec_der_fun_g(g)*V_over_F
#            print(spec_der_calc)
        else:
            spec_calc = V_over_F
        if (not (iteration % update_frequency) or err_eq < update_eq) or iteration < 2:
            spec_err_old = spec_err # Only update old error on an update iteration
        spec_err = spec_calc - spec_val

        try:
            step_der = spec_err/spec_der_calc
            # print(spec_err, step_der, p1-p0)
        except:
            pass

        # Check for negative fractions - normalize only if needed
        for xi in xs_new:
            if xi < 0.0:
                xs_new_sum_inv = 1.0/sum(abs(i) for i in xs_new)
                xs_new = [abs(i)*xs_new_sum_inv for i in xs_new]
                break
        for yi in ys_new:
            if yi < 0.0:
                ys_new_sum_inv = 1.0/sum(abs(i) for i in ys_new)
                ys_new = [abs(i)*ys_new_sum_inv for i in ys_new]
                break

        err_eq, comp_diff = 0.0, 0.0
        for i in cmps:
            err_i = Ks[i]*xs[i]/ys[i] - 1.0
            err_eq += err_i*err_i
            comp_diff += abs(xs[i] - ys[i])

        # Accept the new compositions
#        xs, ys = xs_new, zs # This has worse convergence behavior; seems to not even converge some of the time
        xs, ys = xs_new, ys_new

        if comp_diff < trivial_solution_tol and iteration: # Allow the first iteration to start with the same composition
            raise ValueError("Converged to trivial condition, compositions of both phases equal")
        print('Guess: %g, Eq Err: %g, Spec Err: %g, VF: %g' %(p0, err_eq, spec_err, V_over_F))
#        print(p0, err_eq, spec_err, V_over_F)
#        print(p0, err, spec_err, xs, ys, V_over_F)
        if err_eq < tol_eq and abs(spec_err) < tol_spec_abs:
            return p0, V_over_F, xs, ys, l, g, iteration, err_eq, spec_err
    raise UnconvergedError('End of SS without convergence')



def sequential_substitution_2P_double(zs, xs_guess, ys_guess, liquid_phase,
                                     gas_phase, guess, spec_vals,
                                     iter_var0='T', iter_var1='P',
                                     spec_vars=['H', 'S'],
                                     maxiter=1000, tol_eq=1E-13, tol_specs=1e-9,
                                     trivial_solution_tol=1e-5, damping=1.0,
                                     V_over_F_guess=None, fprime=True):
    xs, ys = xs_guess, ys_guess
    if V_over_F_guess is None:
        V_over_F = 0.5
    else:
        V_over_F = V_over_F_guess

    cmps = range(len(zs))

    iter0_val = guess[0]
    iter1_val = guess[1]

    spec0_val = spec_vals[0]
    spec1_val = spec_vals[1]

    spec0_var = spec_vars[0]
    spec1_var = spec_vars[1]

    spec0_fun_l = getattr(liquid_phase.__class__, spec0_var)
    spec0_fun_g = getattr(gas_phase.__class__, spec0_var)

    spec1_fun_l = getattr(liquid_phase.__class__, spec1_var)
    spec1_fun_g = getattr(gas_phase.__class__, spec1_var)

    spec0_der0 = 'd%s_d%s_%s'%(spec0_var, iter_var0, iter_var1)
    spec1_der0 = 'd%s_d%s_%s'%(spec1_var, iter_var0, iter_var1)
    spec0_der1 = 'd%s_d%s_%s'%(spec0_var, iter_var1, iter_var0)
    spec1_der1 = 'd%s_d%s_%s'%(spec1_var, iter_var1, iter_var0)

    spec0_der0_fun_l = getattr(liquid_phase.__class__, spec0_der0)
    spec0_der0_fun_g = getattr(gas_phase.__class__, spec0_der0)

    spec1_der0_fun_l = getattr(liquid_phase.__class__, spec1_der0)
    spec1_der0_fun_g = getattr(gas_phase.__class__, spec1_der0)

    spec0_der1_fun_l = getattr(liquid_phase.__class__, spec0_der1)
    spec0_der1_fun_g = getattr(gas_phase.__class__, spec0_der1)

    spec1_der1_fun_l = getattr(liquid_phase.__class__, spec1_der1)
    spec1_der1_fun_g = getattr(gas_phase.__class__, spec1_der1)

    step_der = None
    for iteration in range(maxiter):
        TPV_args[iter_var0] = iter0_val
        TPV_args[iter_var1] = iter1_val

        g = gas_phase.to(zs=ys, **TPV_args)
        l = liquid_phase.to(zs=xs, **TPV_args)
        lnphis_g = g.lnphis()
        lnphis_l = l.lnphis()

        Ks = [exp(lnphis_l[i] - lnphis_g[i]) for i in cmps]

        V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F)

        spec0_calc = spec0_fun_l(l)*(1.0 - V_over_F) + spec0_fun_g(g)*V_over_F
        spec1_calc = spec1_fun_l(l)*(1.0 - V_over_F) + spec1_fun_g(g)*V_over_F

        spec0_der0_calc = spec0_der0_fun_l(l)*(1.0 - V_over_F) + spec0_der0_fun_g(g)*V_over_F
        spec0_der1_calc = spec0_der1_fun_l(l)*(1.0 - V_over_F) + spec0_der1_fun_g(g)*V_over_F

        spec1_der0_calc = spec1_der0_fun_l(l)*(1.0 - V_over_F) + spec1_der0_fun_g(g)*V_over_F
        spec1_der1_calc = spec1_der1_fun_l(l)*(1.0 - V_over_F) + spec1_der1_fun_g(g)*V_over_F

        errs = [spec0_calc - spec0_val, spec1_calc - spec1_val]
        jac = [[spec0_der0_calc, spec0_der1_calc], [spec1_der0_calc, spec1_der1_calc]]

        # Do the newton step
        dx = py_solve(jac, [-v for v in errs])
        iter0_val, iter1_val = [xi + dxi*damping for xi, dxi in zip([iter0_val, iter1_val], dx)]


        # Check for negative fractions - normalize only if needed
        for xi in xs_new:
            if xi < 0.0:
                xs_new_sum = sum(abs(i) for i in xs_new)
                xs_new = [abs(i)/xs_new_sum for i in xs_new]
                break
        for yi in ys_new:
            if yi < 0.0:
                ys_new_sum = sum(abs(i) for i in ys_new)
                ys_new = [abs(i)/ys_new_sum for i in ys_new]
                break

        err, comp_diff = 0.0, 0.0
        for i in cmps:
            err_i = Ks[i]*xs[i]/ys[i] - 1.0
            err += err_i*err_i
            comp_diff += abs(xs[i] - ys[i])

        xs, ys = xs_new, ys_new

        if comp_diff < trivial_solution_tol:
            raise ValueError("Converged to trivial condition, compositions of both phases equal")

        if err < tol_eq and abs(err0) < tol_spec_abs:
            return p0, V_over_F, xs, ys, l, g, iteration, err, err0
    raise UnconvergedError('End of SS without convergence')


def stability_iteration_Michelsen(trial_phase, zs_test, test_phase=None,
                                  maxiter=20, xtol=1E-12):
    # So long as for both trial_phase, and test_phase use the lowest Gibbs energy fugacities, no need to test two phases.
    # Very much no need to converge using acceleration - just keep a low tolerance
    # At any point, can use the Ks working, assume a drop of the new phase, and evaluate two new phases and see if G drops.
    # If it does, drop out early! This implementation does not do that.

    # Should be possible to tell if converging to trivial solution during the process - and bail out then
    # It is possible to switch this function to operated on lnphis e.g.
    #             corrections[i] = ci = zs[i]/zs_test[i]*trunc_exp(lnphis_trial[i] - lnphis_test[i])*sum_zs_test_inv
    # however numerical differences seem to be huge and operate better on fugacities with the trunc_exp function
    # then anything else.
    
    # Can this whole function be switched to the functional approach?
    # Should be possible


    if test_phase is None:
        test_phase = trial_phase
    T, P, zs = trial_phase.T, trial_phase.P, trial_phase.zs
    N = trial_phase.N
    fugacities_trial = trial_phase.fugacities_lowest_Gibbs()

    # Go through the feed composition - and the trial composition - if we have zeros, need to make them a trace;
    zs_test2 = [0.0]*N
    for i in range(N):
        zs_test2[i] = zs_test[i]
    zs_test = zs_test2
    
    for i in range(N):
        if zs_test[i] == 0.0:
            zs_test[i] = 1e-50
            # break
    for i in range(N):
        if zs[i] == 0.0:
            zs2 = [0.0]*N
            for i in range(N):
                if zs[i] == 0.0:
                    zs2[i] = 1e-50
                else:
                    zs2[i] = zs[i]
            zs = zs2
            # Requires another evaluation of the trial phase
            trial_phase = trial_phase.to(T=T, P=P, zs=zs)
            fugacities_trial = trial_phase.fugacities_lowest_Gibbs()
            break

    # Basis of equations is for the test phase being a gas, the trial phase assumed is a liquid
    # makes no real difference
    Ks = [0.0]*N
    corrections = [1.0]*N

    # Model converges towards fictional K values which, when evaluated, yield the
    # stationary point composition
    for i in range(N):
        Ks[i] = zs_test[i]/zs[i]

    sum_zs_test = sum_zs_test_inv = 1.0
    converged = False
    for _ in range(maxiter):
#        test_phase = test_phase.to(T=T, P=P, zs=zs_test)
#        fugacities_test = test_phase.fugacities_lowest_Gibbs()
        fugacities_test = test_phase.fugacities_at_zs(zs_test)

        err = 0.0
        try:
            for i in range(N):
                corrections[i] = ci = fugacities_trial[i]/fugacities_test[i]*sum_zs_test_inv
                Ks[i] *= ci
                err += (ci - 1.0)*(ci - 1.0)
        except:
            # A test fugacity became zero
            # May need special handling for this outside.
            converged = True
            break

        if err < xtol:
            converged = True
            break

        # Update compositions for the next iteration - might as well move this above the break check
        for i in range(N):
            zs_test[i] = Ks[i]*zs[i] # new test phase comp

        # Cannot move the normalization above the error check - returning
        # unnormalized sum_zs_test is used also to detect a trivial solution
        sum_zs_test = 0.0
        for i in range(N):
            sum_zs_test += zs_test[i]
        try:
            sum_zs_test_inv = 1.0/sum_zs_test
        except:
            # Fugacities are all zero
            converged = True
            break
        for i in range(N):
            zs_test[i] *= sum_zs_test_inv


    if converged:
        try:
            V_over_F, xs, ys = V_over_F, trial_zs, appearing_zs = flash_inner_loop(zs, Ks)
        except:
            # Converged to trivial solution so closely the math does not work
            V_over_F, xs, ys = V_over_F, trial_zs, appearing_zs = 0.0, zs, zs

        # Calculate the dG of the feed
        dG_RT = 0.0
        if V_over_F != 0.0:
            lnphis_test = test_phase.lnphis_at_zs(zs_test) #test_phase.lnphis()
            for i in range(N):
                dG_RT += zs_test[i]*(log(zs_test[i]) + lnphis_test[i])
            dG_RT *= V_over_F
#        print(dG_RT)


        return sum_zs_test, Ks, zs_test, V_over_F, trial_zs, appearing_zs, dG_RT
    else:
        raise UnconvergedError('End of stability_iteration_Michelsen without convergence')



def TPV_double_solve_1P(zs, phase, guesses, spec_vals,
                        goal_specs=('V', 'U'), state_specs=('T', 'P'),
                        maxiter=200, xtol=1E-10, ytol=None, spec_funs=None):
    kwargs = {'zs': zs}
    phase_cls = phase.__class__
    s00 = 'd%s_d%s_%s' %(goal_specs[0], state_specs[0], state_specs[1])
    s01 = 'd%s_d%s_%s' %(goal_specs[0], state_specs[1], state_specs[0])
    s10 = 'd%s_d%s_%s' %(goal_specs[1], state_specs[0], state_specs[1])
    s11 = 'd%s_d%s_%s' %(goal_specs[1], state_specs[1], state_specs[0])
    try:
        err0_fun = getattr(phase_cls, goal_specs[0])
        err1_fun = getattr(phase_cls, goal_specs[1])
        j00 = getattr(phase_cls, s00)
        j01 = getattr(phase_cls, s01)
        j10 = getattr(phase_cls, s10)
        j11 = getattr(phase_cls, s11)
    except:
        pass

    cache = []

    def to_solve(states):
        kwargs[state_specs[0]] = float(states[0])
        kwargs[state_specs[1]] = float(states[1])
        new = phase.to(**kwargs)
        try:
            v0, v1 = err0_fun(new), err1_fun(new)
            jac = [[j00(new), j01(new)],
                   [j10(new), j11(new)]]
        except:
            v0, v1 = new.value(goal_specs[0]), new.value(goal_specs[1])
            jac = [[new.value(s00), new.value(s01)],
                   [new.value(s10), new.value(s11)]]

        if spec_funs is not None:
            err0 = v0 - spec_funs[0](new)
            err1 = v1 - spec_funs[1](new)
        else:
            err0 = v0 - spec_vals[0]
            err1 = v1 - spec_vals[1]
        errs = [err0, err1]

        cache[:] = [new, errs, jac]
        print(kwargs, errs)
        return errs, jac

#
    states, iterations = newton_system(to_solve, x0=guesses, jac=True, xtol=xtol,
             ytol=ytol, maxiter=maxiter, damping_func=damping_maintain_sign)
    phase = cache[0]
    err = cache[1]
    jac = cache[2]

    return states, phase, iterations, err, jac



def assert_stab_success_2P(liq, gas, stab, T, P, zs, guess_name, xs=None,
                           ys=None, VF=None, SS_tol=1e-15, rtol=1e-7):
    r'''Basic function - perform a specified stability test, and then a two-phase flash using it
    Check on specified variables the method is working.
    '''
    gas = gas.to(T=T, P=P, zs=zs)
    liq = liq.to(T=T, P=P, zs=zs)
    trial_comp = stab.incipient_guess_named(T, P, zs, guess_name)
    if liq.G() < gas.G():
        min_phase, other_phase = liq, gas
    else:
        min_phase, other_phase = gas, liq

    _, _, _, V_over_F, trial_zs, appearing_zs, dG_RT = stability_iteration_Michelsen(min_phase, trial_comp, test_phase=other_phase, maxiter=100)

    V_over_F, xs_calc, ys_calc, l, g, iteration, err = sequential_substitution_2P(T=T, P=P, V=None,
                                                                        zs=zs, xs_guess=trial_zs, ys_guess=appearing_zs,
                                                                        liquid_phase=min_phase, tol=SS_tol,
                                                                        gas_phase=other_phase)
    if xs_calc is not None:
        assert_close1d(xs, xs_calc, rtol)
    if ys_calc is not None:
        assert_close1d(ys, ys_calc, rtol)
    if VF is not None:
        assert_close(V_over_F, VF, rtol)
    assert_close1d(l.fugacities(), g.fugacities(), rtol)

def TPV_solve_HSGUA_guesses_VL(zs, method, constants, correlations,
                               fixed_var_val, spec_val,
                               iter_var='T', fixed_var='P', spec='H',
                               maxiter=20, xtol=1E-7, ytol=None,
                               bounded=False, min_bound=None, max_bound=None,
                               user_guess=None, last_conv=None, T_ref=298.15,
                               P_ref=101325.0):
    global V_over_F_guess
    V_over_F_guess = 0.5

    cmps = range(constants.N)
    Tcs, Pcs, omegas = constants.Tcs, constants.Pcs, constants.omegas

    if fixed_var == iter_var:
        raise ValueError("Fixed variable cannot be the same as iteration variable")
    if fixed_var not in ('T', 'P', 'V'):
        raise ValueError("Fixed variable must be one of `T`, `P`, `V`")
    if iter_var not in ('T', 'P', 'V'):
        raise ValueError("Iteration variable must be one of `T`, `P`, `V`")
    if spec not in ('H', 'S', 'G', 'U', 'A'):
        raise ValueError("Spec variable must be one of `H`, `S`, `G` `U`, `A`")


    cmps = range(len(zs))

    iter_T = iter_var == 'T'
    iter_P = iter_var == 'P'
    iter_V = iter_var == 'V'

    fixed_P = fixed_var == 'P'
    fixed_T = fixed_var == 'T'
    fixed_V = fixed_var == 'V'
    if fixed_P:
        P = fixed_var_val
    elif fixed_T:
        T = fixed_var_val
    elif fixed_V:
        V = fixed_var_val

    always_S = spec in ('S', 'G', 'A')
    always_H = spec in ('H', 'G', 'U', 'A')
    always_V = spec in ('U', 'A')


    def H_model(T, P, xs, ys, V_over_F):
        if V_over_F >= 1.0:
            return H_model_g(T, P, zs)
        elif V_over_F <= 0.0:
            return H_model_l(T, P, zs)
        H_liq = H_model_l(T, P, xs)
        H_gas = H_model_g(T, P, ys)
        return H_liq*(1.0 - V_over_F) + V_over_F*H_gas

    def S_model(T, P, xs, ys, V_over_F):
        if V_over_F >= 1.0:
            return S_model_g(T, P, zs)
        elif V_over_F <= 0.0:
            return S_model_l(T, P, zs)
        S_liq = S_model_l(T, P, xs)
        S_gas = S_model_g(T, P, ys)
        return S_liq*(1.0 - V_over_F) + V_over_F*S_gas

    def V_model(T, P, xs, ys, V_over_F):
        if V_over_F >= 1.0:
            return V_model_g(T, P, zs)
        elif V_over_F <= 0.0:
            return V_model_l(T, P, zs)
        V_liq = V_model_l(T, P, xs)
        V_gas = V_model_g(T, P, ys)
        return V_liq*(1.0 - V_over_F) + V_over_F*V_gas

    # whhat goes in here?
    if always_S:
        P_ref_inv = 1.0/P_ref
        dS_ideal = R*sum([zi*log(zi) for zi in zs if zi > 0.0]) # ideal composition entropy composition

    info = []
    def err(guess):
        # Translate the fixed variable to a local variable
        if fixed_P:
            P = fixed_var_val
        elif fixed_T:
            T = fixed_var_val
        elif fixed_V:
            V = fixed_var_val
            T = None
        # Translate the iteration variable to a local variable
        if iter_P:
            P = guess
            if not fixed_V:
                V = None
        elif iter_T:
            T = guess
            if not fixed_V:
                V = None
        elif iter_V:
            V = guess
            T = None

        if T is None:
            T = T_from_V(V, P, zs)

        VF, xs, ys = flash_model(T, P, zs)
        info[:] = VF, xs, ys

        # Compute S, H, V as necessary
        if always_S:
            S = S_model(T, P, xs, ys, VF) - dS_ideal - R*log(P*P_ref_inv)
        if always_H:
            H =  H_model(T, P, xs, ys, VF)
        if always_V and V is None:
            V = V_model(T, P, xs, ys, VF)

        # Return the objective function
        if spec == 'H':
            err = H - spec_val
        elif spec == 'S':
            err = S - spec_val
        elif spec == 'G':
            err = (H - T*S) - spec_val
        elif spec == 'U':
            err = (H - P*V) - spec_val
        elif spec == 'A':
            err = (H - P*V - T*S) - spec_val
#         print(T, P, V, 'TPV', err)
        return err

    # Common models
    VolumeLiquids = correlations.VolumeLiquids
    def V_model_l(T, P, zs):
        V_calc = 0.
        for i in cmps:
            V_calc += zs[i]*VolumeLiquids[i].T_dependent_property(T)
        return V_calc

    def T_from_V_l(V, P, zs):
        T_calc = 0.
        for i in cmps:
            T_calc += zs[i]*VolumeLiquids[i].solve_property(V)
        return T_calc

    def V_model_g(T, P, zs):
        return R*T/P

    def T_from_V_g(V, P, zs):
        return P*V/R

    if method == IDEAL_WILSON or method == SHAW_ELEMENTAL:
        if iter_P:
            if fixed_T:
                T_inv = 1.0/T
                Ks_P = [Pcs[i]*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]*T_inv))) for i in cmps]
            def flash_model(T, P, zs):
                global V_over_F_guess
                P_inv = 1.0/P
                if not fixed_T:
                    T_inv = 1.0/T
                    Ks_P_local = [Pcs[i]*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]*T_inv))) for i in cmps]
                    Ks = [Ki*P_inv for Ki in Ks_P_local]
                else:
                    Ks = [Ki*P_inv for Ki in Ks_P]
                K_low, K_high = False, False
                for i in cmps:
                    if zs[i] != 0.0:
                        if Ks[i] > 1.0:
                            K_high = True
                        else:
                            K_low = True
                        if K_high and K_low:
                            break
                if K_high and K_low:
                    V_over_F_guess, xs, ys = Rachford_Rice_solution_LN2(zs, Ks, V_over_F_guess)
                    return V_over_F_guess, xs, ys
                elif K_high:
                    return 1.0, zs, zs
                else:
                    return 0.0, zs, zs
        else:
            P_inv  = 1.0/P
            def flash_model(T, P, zs):
                global V_over_F_guess
                T_inv = 1.0/T
                Ks = [Pcs[i]*P_inv*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]*T_inv))) for i in cmps]
                K_low, K_high = False, False
                for i in cmps:
                    if zs[i] != 0.0:
                        if Ks[i] > 1.0:
                            K_high = True
                        else:
                            K_low = True
                        if K_high and K_low:
                            break
                if K_high and K_low:
                    V_over_F_guess, xs, ys = Rachford_Rice_solution_LN2(zs, Ks, V_over_F_guess)
                    return V_over_F_guess, xs, ys
                elif K_high:
                    return 1.0, zs, zs
                else:
                    return 0.0, zs, zs

    if method == SHAW_ELEMENTAL:
        VolumeLiquids = correlations.VolumeLiquids
        MWs, n_atoms = constants.MWs, constants.n_atoms

        def H_model_g(T, P, zs):
            MW_g, sv_g = 0.0, 0.0
            for i in cmps:
                MW_g += MWs[i]*zs[i]
                sv_g += n_atoms[i]*zs[i]
            sv_g /= MW_g

            H_ref_LS = Lastovka_Shaw_integral(T_ref, sv_g)
            H1 = Lastovka_Shaw_integral(T, sv_g)
            dH = H1 - H_ref_LS
            H_gas = 1e-3*dH*MW_g  #property_mass_to_molar(dH, MW_g)
            return H_gas

        def S_model_g(T, P, zs):
            MW_g, sv_g = 0.0, 0.0
            for i in cmps:
                MW_g += MWs[i]*zs[i]
                sv_g += n_atoms[i]*zs[i]
            sv_g /= MW_g

            S_ref_LS = Lastovka_Shaw_integral_over_T(T_ref, sv_g)
            S1 = Lastovka_Shaw_integral_over_T(T, sv_g)
            dS = S1 - S_ref_LS
            S_gas = 1e-3*dS*MW_g
            return S_gas

        def H_model_l(T, P, zs):
            MW_l, sv_l, Tc_l, omega_l = 0.0, 0.0, 0.0, 0.0
            for i in cmps:
                MW_l += MWs[i]*zs[i]
                sv_l += n_atoms[i]*zs[i]
                Tc_l += Tcs[i]*zs[i]
                omega_l += omegas[i]*zs[i]
            sv_l /= MW_l

            H_ref_DS = Dadgostar_Shaw_integral(T_ref, sv_l)
            H1 = Dadgostar_Shaw_integral(T, sv_l)
            Hvap = SMK(T, Tc_l, omega_l)

            dH = H1 - H_ref_DS
            H_liq = 1e-3*dH*MW_l #property_mass_to_molar(dH, MW_l)
            return (H_liq - Hvap)

        def S_model_l(T, P, zs):
            MW_l, sv_l, Tc_l, omega_l = 0.0, 0.0, 0.0, 0.0
            for i in cmps:
                MW_l += MWs[i]*zs[i]
                sv_l += n_atoms[i]*zs[i]
                Tc_l += Tcs[i]*zs[i]
                omega_l += omegas[i]*zs[i]
            sv_l /= MW_l

            S_ref_DS = Dadgostar_Shaw_integral_over_T(T_ref, sv_l)
            S1 = Dadgostar_Shaw_integral_over_T(T, sv_l)

            Hvap = SMK(T, Tc_l, omega_l)

            dS = S1 - S_ref_DS
            S_liq = 1e-3*dS*MW_l
            return (S_liq - Hvap/T)


    elif method == IDEAL_WILSON:
        HeatCapacityGases = correlations.HeatCapacityGases
        EnthalpyVaporizations = correlations.EnthalpyVaporizations
        def flash_model(T, P, zs):
            _, _, VF, xs, ys = flash_wilson(zs, constants.Tcs, constants.Pcs, constants.omegas, T=T, P=P)
            return VF, xs, ys

        def H_model_g(T, P, zs):
            H_calc = 0.
            for i in cmps:
                H_calc += zs[i]*HeatCapacityGases[i].T_dependent_property_integral(T_ref, T)
            return H_calc

        def S_model_g(T, P, zs):
            S_calc = 0.
            for i in cmps:
                S_calc += zs[i]*HeatCapacityGases[i].T_dependent_property_integral_over_T(T_ref, T)
            return S_calc

        def H_model_l(T, P, zs):
            H_calc = 0.
            for i in cmps:
                H_calc += zs[i]*(HeatCapacityGases[i].T_dependent_property_integral(T_ref, T) - EnthalpyVaporizations[i](T))
            return H_calc

        def S_model_l(T, P, zs):
            S_calc = 0.
            T_inv = 1.0/T
            for i in cmps:
                S_calc += zs[i]*(HeatCapacityGases[i].T_dependent_property_integral_over_T(T_ref, T) - T_inv*EnthalpyVaporizations[i](T))
            return S_calc


    try:
        # All three variables P, T, V are positive but can grow unbounded, so
        # for the secant method, only set the one variable
        if iter_T:
            guess = 298.15
        elif iter_P:
            guess = 101325.0
        elif iter_V:
            guess = 0.024465403697038125
        val = secant(err, guess, xtol=xtol, ytol=ytol,
                      maxiter=maxiter, bisection=True, low=min_bound, require_xtol=False)
        return val, info[0], info[1], info[2]
    except (UnconvergedError,) as e:
        val = brenth(err, min_bound, max_bound, xtol=xtol, ytol=ytol, maxiter=maxiter)
        return val, info[0], info[1], info[2]


global cm_flash
cm_flash = None
def cm_flash_tol():
    global cm_flash
    if cm_flash is not None:
        return cm_flash
    from matplotlib.colors import ListedColormap
    N = 100
    vals = np.zeros((N, 4))
    vals[:, 3] = np.ones(N)

    # Grey for 1e-10 to 1e-7
    low = 40
    vals[:low, 0] = np.linspace(100/256, 1, low)[::-1]
    vals[:low, 1] = np.linspace(100/256, 1, low)[::-1]
    vals[:low, 2] = np.linspace(100/256, 1, low)[::-1]

    # green 1e-6 to 1e-5
    ok = 50
    vals[low:ok, 1] = np.linspace(100/256, 1, ok-low)[::-1]

    # Blue 1e-5 to 1e-3
    mid = 70
    vals[ok:mid, 2] = np.linspace(100/256, 1, mid-ok)[::-1]
    # Red 1e-3 and higher
    vals[mid:101, 0] = np.linspace(100/256, 1, 100-mid)[::-1]
    newcmp = ListedColormap(vals)

    cm_flash = newcmp
    return cm_flash

def deduplicate_stab_results(results, tol_frac_err=5e-3):
    if not results:
        return results
    N = len(results[0][0])
    cmps = range(N)
    results.sort(key=lambda x: (x[0][0], x[2]))
    good_results = [results[0]]
    for t in results[1:]:
        xs_last, ys_last = good_results[-1][0], good_results[-1][1]
        xs, ys = t[0], t[1]
        diff_x = sum([abs(xs[i] - xs_last[i]) for i in cmps])/N
        diff_y = sum([abs(ys[i] - ys_last[i]) for i in cmps])/N
        if diff_x > tol_frac_err or diff_y > tol_frac_err:
            good_results.append(t)
    return good_results

empty_flash_conv = {'iterations': 0, 'err': 0.0, 'stab_guess_name': None}
one_in_list = [1.0]
empty_list = []


