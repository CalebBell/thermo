# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
__all__ = ['sequential_substitution_2P', 'bubble_T_Michelsen_Mollerup',
           'dew_T_Michelsen_Mollerup', 'bubble_P_Michelsen_Mollerup',
           'dew_P_Michelsen_Mollerup',
           'minimize_gibbs_2P_transformed', 'sequential_substitution_Mehra_2P',
           'nonlin_2P', 'sequential_substitution_NP',
           'minimize_gibbs_NP_transformed', 'FlashVL', 'FlashPureVLS',
           'TPV_HSGUA_guesses_1P_methods', 'TPV_solve_HSGUA_guesses_1P',
           ]

from fluids.constants import R, R_inv
from fluids.numerics import (UnconvergedError, trunc_exp, py_newton as newton,
                             py_brenth as brenth, secant, numpy as np, linspace, 
                             logspace, oscillation_checker, damping_maintain_sign,
                             oscillation_checking_wrapper, OscillationError,
                             best_bounding_bounds)
from scipy.optimize import minimize, fsolve, root
from thermo.utils import (exp, log, log10, copysign, normalize, has_matplotlib,
                          mixing_simple, property_mass_to_molar)
from thermo.heat_capacity import (Lastovka_Shaw_T_for_Hm, Dadgostar_Shaw_integral,
                                  Dadgostar_Shaw_integral_over_T, Lastovka_Shaw_integral,
                                  Lastovka_Shaw_integral_over_T)
from thermo.phase_change import SMK
from thermo.volume import COSTALD
from thermo.activity import (flash_inner_loop, Rachford_Rice_solutionN,
                             Rachford_Rice_flash_error, Rachford_Rice_solution2)
from thermo.equilibrium import EquilibriumState
from thermo.phases import Phase, gas_phases, liquid_phases, solid_phases, EOSLiquid, EOSGas
from thermo.phase_identification import identify_sort_phases
from thermo.bulk import default_settings

def sequential_substitution_2P(T, P, zs, xs_guess, ys_guess, liquid_phase,
                               gas_phase, maxiter=1000, tol=1E-13,
                               trivial_solution_tol=1e-5, V_over_F_guess=None):
    
    xs, ys = xs_guess, ys_guess
    if V_over_F_guess is None:
        V_over_F = 0.5
    else:
        V_over_F = V_over_F_guess
        
    cmps = range(len(zs))
    
    for iteration in range(maxiter):
        g = gas_phase.to_TP_zs(T=T, P=P, zs=ys)
        l = liquid_phase.to_TP_zs(T=T, P=P, zs=xs)
        
        lnphis_g = g.lnphis()
        lnphis_l = l.lnphis()

        Ks = [exp(lnphis_l[i] - lnphis_g[i]) for i in cmps] # K_value(phi_l=l, phi_g=g)
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
        
        # Calculate the error using the new Ks and old compositions
        # Claimed error function in CONVENTIONAL AND RAPID FLASH
        # CALCULATIONS FOR THE SOAVE-REDLICH-KWONG AND PENG-ROBINSON EQUATIONS OF STATE
        
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
            raise ValueError("Converged to trivial condition, compositions of both phases equal")
        if err < tol:
            return V_over_F, xs, ys, l, g, iteration, err
    raise UnconvergedError('End of SS without convergence')


def sequential_substitution_NP(T, P, zs, compositions_guesses, phases,
                               betas, maxiter=1000, tol=1E-13,
                               trivial_solution_tol=1e-5, ref_phase=1):
    
    compositions = compositions_guesses
    cmps = range(len(zs))
    phases_iter = range(len(phases))
    
    for iteration in range(maxiter):
        phases = [phases[i].to_TP_zs(T=T, P=P, zs=compositions[i]) for i in phases_iter]
        lnphis = [phases[i].lnphis() for i in phases_iter]
        Ks = [[exp(lnphis[ref_phase][j] - lnphis[i][j]) for j in cmps] 
              for i in phases_iter if i != ref_phase] # 
        
        betas, compositions = Rachford_Rice_solutionN(zs, Ks, betas)
#        betas.insert(ref_phase, 1.0 - sum(betas))
        print(betas, compositions, Ks, 'calculated')
#        Rachford_Rice_solution2(zs, Ks_y, Ks_z, beta_y=0.5, beta_z=1e-6)
                
        err = 0.0
        
#        xs = compositions[ref_phase]
        # Suggested tolerance 1e-15
#        passed_K = False
#        for i in phases_iter:
#            if i == ref_phase:
#                passed_K = True
#                continue
#            K_idx = i
#            if passed_K:
#                K_idx -= 1
#            
#            Kis, ys = Ks[K_idx], compositions[i]
#            for Ki, xi, yi in zip(Kis, xs, ys):
#                err_i = Ki*xi/yi - 1.0
#                err += err_i*err_i
        
        # Check for 
#        comp_difference = sum([abs(xi - yi) for xi, yi in zip(xs, ys)])
#        if comp_difference < trivial_solution_tol:
#            raise ValueError("Converged to trivial condition, compositions of both phases equal")
#        if err < tol:
#            return betas, compositions, phases, iteration, err
        if iteration > 100:
            return betas, compositions, phases, iteration, err
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
            raise ValueError("Converged to trivial condition, compositions of both phases equal")
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
            raise ValueError("Converged to trivial condition, compositions of both phases equal")
        if err < tol:
            return V_over_F, xs, ys, l, g, iteration, err
    raise UnconvergedError('End of SS without convergence')


def nonlin_2P(T, P, zs, xs_guess, ys_guess, liquid_phase,
              gas_phase, maxiter=1000, tol=1E-13, 
              trivial_solution_tol=1e-5, V_over_F_guess=None,
              method='hybr'):
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
                                  method='L-BFGS-B', opt_kwargs=None):
    if opt_kwargs is None:
        opt_kwargs = {}
    N = len(zs)
    cmps = range(N)
    phase_count = len(phases)
    phase_iter = range(phase_count)
    RT_inv = 1.0/(R*T)
    
    # Only exist for the first n phases
    # Do not multiply by zs - we are already multiplying by a composition
    flows_guess = [compositions_guesses[j][i]*betas[j] for j in range(phase_count - 1) for i in cmps]
    # Convert the flow guesses to the basis used
    remaining = zs
    flows_guess_basis = []
    for j in range(phase_count-1):
        phase_guess = flows_guess[j*N:j*N+N]
        flows_guess_basis.extend([-log((remaining[i]-phase_guess[i])/(phase_guess[i]-0.0)) for i in cmps])
        remaining = [remaining[i] - phase_guess[i] for i in cmps]

    global min_G, iterations
    min_G = 1e100
    iterations = 0
    info = []
    def G(flows):
        global min_G, iterations
        try:
            flows = flows.tolist()
        except:
            pass
        iterations += 1
        iter_flows = []
        iter_comps = []
        iter_betas = []
        iter_phases = []
        
        remaining = zs
        
        for j in phase_iter:
            v = flows[j*N:j*N+N]
            
            # Mole flows of phase0/vapor
            if j == phase_count - 1:
                vs = remaining
            else:
                vs = [(0.0 + (remaining[i] - 0.0)/(1.0 + trunc_exp(-v[i]))) for i in cmps]
            vs_sum = sum(vs)
            vs_sum_inv = 1.0/vs_sum
            ys = [vs[i]*vs_sum_inv for i in cmps]
            ys = normalize(ys)
            iter_flows.append(vs)
            iter_comps.append(ys)
            iter_betas.append(vs_sum) # Would be divided by feed but feed is zs = 1
    
            remaining = [remaining[i] - vs[i] for i in cmps]
        G = 0.0
        for j in phase_iter:
            phase = phases[j].to_TP_zs(T=T, P=P, zs=iter_comps[j])
            iter_phases.append(phase)
            G += phase.G()*iter_betas[j]
        
        if G < min_G:
            info[:] = iter_betas, iter_comps, iter_phases
            min_G = G
        return G*RT_inv
#    ans = None
    ans = minimize(G, flows_guess_basis, method=method, tol=tol, **opt_kwargs)
#    G(ans['x']) # Make sure info has right value
    ans['fun'] /= RT_inv
    
    betas, compositions, phases = info
    return betas, compositions, phases, iterations, float(ans['fun'])
    
#    return ans, info
    
    
l_undefined_T_msg = "Could not calculate liquid conditions at provided temperature %s K (mole fracions %s)"   
g_undefined_T_msg = "Could not calculate vapor conditions at provided temperature %s K (mole fracions %s)"   
l_undefined_P_msg = "Could not calculate liquid conditions at provided pressure %s Pa (mole fracions %s)"   
g_undefined_P_msg = "Could not calculate vapor conditions at provided pressure %s Pa (mole fracions %s)"   

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



strs_to_ders = {('H', 'T'): 'dH_dT',
                ('S', 'T'): 'dS_dT',
                ('G', 'T'): 'dG_dT',
                ('U', 'T'): 'dU_dT',
                ('A', 'T'): 'dA_dT',
                ('H', 'P'): 'dH_dP',
                ('S', 'P'): 'dS_dP',
                ('G', 'P'): 'dG_dP',
                ('U', 'P'): 'dU_dP',
                ('A', 'P'): 'dA_dP',
                ('H', 'V'): 'dH_dV',
                ('S', 'V'): 'dS_dV',
                ('G', 'V'): 'dG_dV',
                ('U', 'V'): 'dU_dV',
                ('A', 'V'): 'dA_dV'}

def TPV_solve_HSGUA_1P(zs, phase, guess, fixed_var_val, spec_val,  
                       iter_var='T', fixed_var='P', spec='H',
                       maxiter=200, xtol=1E-10, ytol=None, fprime=False,
                       minimum_progress=0.3, oscillation_detection=True,
                       bounded=False, min_bound=None, max_bound=None):
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
    print(fixed_var, iter_var)
    if fixed_var == iter_var:
        raise ValueError("Fixed variable cannot be the same as iteration variable")
    if fixed_var not in ('T', 'P', 'V'):
        raise ValueError("Fixed variable must be one of `T`, `P`, `V`")
    if iter_var not in ('T', 'P', 'V'):
        raise ValueError("Iteration variable must be one of `T`, `P`, `V`")
    # Little point in enforcing the spec - might want to repurpose the function later
    if spec not in ('H', 'S', 'G', 'U', 'A'):
        raise ValueError("Spec variable must be one of `H`, `S`, `G` `U`, `A`")

    phase_kwargs = {fixed_var: fixed_var_val, 'zs': zs}
    spec_fun = getattr(phase.__class__, spec)
#    print('spec_fun', spec_fun)
    if fprime:
        try:
            # Gotta be a lookup by (spec, iter_var, fixed_var)
            der_attr = strs_to_ders[(spec, iter_var)]
        except KeyError:
            der_attr = 'd' + spec + '_d' + iter_var
        der_attr_fun = getattr(phase.__class__, der_attr)
#        print('der_attr_fun', der_attr_fun)
    def to_solve(guess):
        global iterations
        iterations += 1

        phase_kwargs[iter_var] = guess
        p = phase.to_zs_TPV(**phase_kwargs)
        
        err = spec_fun(p) - spec_val
        store[:] = (p, err)
        if fprime:
            print([err, guess])
            derr = der_attr_fun(p)
            return err, derr
        return err

    if oscillation_detection:
        to_solve2, checker = oscillation_checking_wrapper(to_solve, full=True,
                                                          minimum_progress=minimum_progress)
    else:
        to_solve2 = to_solve
        checker = None
        
    try:
        # All three variables P, T, V are positive but can grow unbounded, so
        # for the secant method, only set the one variable
        if fprime:
            iter_var_val = newton(to_solve2, guess, xtol=xtol, ytol=ytol, fprime=True,
                                  maxiter=maxiter, bisection=True, low=min_bound)
        else:
            iter_var_val = secant(to_solve2, guess, xtol=xtol, ytol=ytol,
                                  maxiter=maxiter, bisection=True, low=min_bound)
    except (UnconvergedError, OscillationError):
        fprime = False
        if bounded and min_bound is not None and max_bound is not None:
            if checker:
                min_bound, max_bound, fa, fb = best_bounding_bounds(min_bound, max_bound, 
                                                                    f=to_solve, xs_pos=checker.xs_pos, ys_pos=checker.ys_pos, 
                                                                    xs_neg=checker.xs_neg, ys_neg=checker.ys_neg)
            else:
                fa, fb = None, None
            iter_var_val = brenth(to_solve, min_bound, max_bound, xtol=xtol, 
                                  ytol=ytol, maxiter=maxiter, fa=fa, fb=fb)
    phase, err = store

    return iter_var_val, phase, iterations, err



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



def solve_PTV_HSGUA_1P(phase, zs, fixed_var_val, spec_val, fixed_var, 
                       spec, iter_var, constants, correlations):
    if iter_var == 'T':
        min_bound = Phase.T_MIN_FIXED
        max_bound = Phase.T_MAX_FIXED
    elif iter_var == 'P':
        min_bound = Phase.P_MIN_FIXED
        max_bound = Phase.P_MAX_FIXED
    elif iter_var == 'V':
        min_bound = Phase.V_MIN_FIXED
        max_bound = Phase.V_MAX_FIXED

    if isinstance(phase, gas_phases):
        methods = [LAST_CONVERGED, FIXED_GUESS, STP_T_GUESS, IG_ENTHALPY,
                   LASTOVKA_SHAW]
    elif isinstance(phase, liquid_phases):
        methods = [LAST_CONVERGED, FIXED_GUESS, STP_T_GUESS, IDEAL_LIQUID_ENTHALPY,
                   DADGOSTAR_SHAW_1]
    else:
        methods = [LAST_CONVERGED, FIXED_GUESS, STP_T_GUESS]
        
    for method in methods:
        try:
            guess = TPV_solve_HSGUA_guesses_1P(zs, method, constants, correlations, 
                               fixed_var_val, spec_val,
                               iter_var=iter_var, fixed_var=fixed_var, spec=spec,
                               maxiter=50, xtol=1E-7, ytol=abs(spec_val)*1e-5,
                               bounded=True, min_bound=min_bound, max_bound=max_bound,                    
                               user_guess=None, last_conv=None, T_ref=298.15,
                               P_ref=101325.0)
            
            break
        except Exception as e:
            pass
    
    _, phase, iterations, err = TPV_solve_HSGUA_1P(zs, phase, guess, fixed_var_val=fixed_var_val, spec_val=spec_val, ytol=1e-8*abs(spec_val),
                                                   iter_var=iter_var, fixed_var=fixed_var, spec=spec, oscillation_detection=True,
                                                   minimum_progress=0.0, maxiter=80, fprime=True,
                                                   bounded=True, min_bound=min_bound, max_bound=max_bound)
    T, P = phase.T, phase.P
    return T, P, phase, iterations, err




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
    Psat = secant(to_solve_secant, P_guess, xtol=xtol, maxiter=maxiter)
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
    Tsat = secant(to_solve_secant, T_guess, xtol=xtol, maxiter=maxiter)
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






class FlashBase(object):
    T_MAX_FIXED = Phase.T_MAX_FIXED
    T_MIN_FIXED = Phase.T_MIN_FIXED
    
    P_MAX_FIXED = Phase.P_MAX_FIXED
    P_MIN_FIXED = Phase.P_MIN_FIXED

    def vapor_scores(self, phases):
        pass

class FlashVL(FlashBase):
    PT_SS_MAXITER = 1000
    PT_SS_TOL = 1e-13
    
    def __init__(self, constants, correlations, liquid, gas):
        self.constants = constants
        self.correlations = correlations
        self.liquid = liquid
        self.gas = gas
        
    def flash(self, zs, T=None, P=None, VF=None, Hm=None, Sm=None):
        constants, correlations = self.constants, self.correlations
        
        liquid, gas = self.liquid, self.gas
        if T is not None and Sm is not None:
            # TS
            pass
        elif P is not None and Sm is not None:
            phase, xs, ys, VF, T = flash_PS_zs_2P(P, Sm, zs)
            # PS
        elif P is not None and Hm is not None:
            phase, xs, ys, VF, T = flash_PH_zs_2P(P, Hm, zs)
            # PH
        elif T is not None and P is not None:
            # PT
            try:
                _, _, VF_guess, xs_guess, ys_guess = flash_wilson(zs, constants.Tcs,
                                        constants.Pcs, constants.omegas, T=T, P=P)
            except:
                xs_guess, ys_guess, VF_guess = zs, zs, 0.5
                                                              
            V_over_F, xs, ys, l, g, iteration, err = sequential_substitution_2P(T, P, zs, xs_guess, ys_guess, liquid,
                           gas, maxiter=self.PT_SS_MAXITER, tol=self.PT_SS_TOL,
                           V_over_F_guess=VF_guess)
            
            flash_specs = {'T': T, 'P': P, 'zs': zs}
            flash_convergence = {'iterations': iteration, 'err': err}
            return EquilibriumState(T, P, zs, gas=g, liquids=[l], solids=[], 
                                    betas=[V_over_F, 1.0-V_over_F], flash_specs=flash_specs, 
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations)
            
            
        elif T is not None and VF == 1:
            dew_P_Michelsen_Mollerup(P_guess, T, zs, liquid_phase, gas_phase, 
                             maxiter=200, xtol=1E-10, xs_guess=None,
                             max_step_damping=1e5, P_update_frequency=1,
                             trivial_solution_tol=1e-4)
        elif T is not None and VF == 0:
            bubble_P_Michelsen_Mollerup(P_guess, T, zs, liquid_phase, gas_phase, 
                                maxiter=200, xtol=1E-10, ys_guess=None,
                                max_step_damping=1e5, P_update_frequency=1,
                                trivial_solution_tol=1e-4)
        elif T is not None and VF is not None:
            pass
            # T-VF flash - unimplemented.
        elif P is not None and VF == 1:
            dew_T_Michelsen_Mollerup(T_guess, P, zs, liquid_phase, gas_phase, 
                                maxiter=200, xtol=1E-10, xs_guess=None,
                                max_step_damping=5.0, T_update_frequency=1,
                                trivial_solution_tol=1e-4)
        elif P is not None and VF == 0:
            bubble_P_Michelsen_Mollerup(P_guess, T, zs, liquid_phase, gas_phase, 
                                maxiter=200, xtol=1E-10, ys_guess=None,
                                max_step_damping=1e5, P_update_frequency=1,
                                trivial_solution_tol=1e-4)
        elif P is not None and VF is not None:
            # P-VF flash - unimplemented
            pass
        else:
            raise Exception('Flash inputs unsupported')

'''
        T_spec = T is not None
        P_spec = P is not None
        V_spec = V is not None
        H_spec = H is not None
        S_spec = S is not None
        U_spec = U is not None
        
# Format - keys above, and TPV spec, HSU spec, and iter_var
        fixed_var='P', spec='H', iter_var='T',
'''
spec_to_iter_vars = {(True, False, False, True, False, False) : ('T', 'H', 'P'),
                     (True, False, False, False, True, False) : ('T', 'S', 'P'),
                     (True, False, False, False, False, True) : ('T', 'U', 'P'),

                     (False, True, False, True, False, False) : ('P', 'H', 'T'),
                     (False, True, False, False, True, False) : ('P', 'S', 'T'),
                     (False, True, False, False, False, True) : ('P', 'U', 'T'),

                     (False, False, True, True, False, False) : ('V', 'H', 'P'),
                     (False, False, True, False, True, False) : ('V', 'S', 'P'),
                     (False, False, True, False, False, True) : ('V', 'U', 'P'),
}

class FlashPureVLS(FlashBase):
    '''
    TODO: Get quick version with equivalent features so can begin 
    UNIT TESTING THE Phases and equilibrium state code. Also this code.
    
    But working on all the phases of water can wait.
    '''
    def __init__(self, constants, correlations, gas, liquids, solids, 
                 settings=default_settings):
        self.constants = constants
        self.correlations = correlations
        self.solids = solids
        self.liquids = liquids
        self.gas = gas
        
        self.gas_count = 1 if gas is not None else 0
        self.liquid_count = len(liquids)
        self.solid_count = len(solids)

        self.phase_count = self.gas_count + self.liquid_count + self.solid_count
        
        if gas is not None:
            phases = [gas] + liquids + solids
            
        else:
            phases = liquids + solids
        self.phases = phases
        
        self.settings = settings
        
        for i, l in enumerate(self.liquids):
            setattr(self, 'liquid' + str(i), l)
        for i, s in enumerate(self.solids):
            setattr(self, 'solid' + str(i), s)
            
        self.VL_only = self.phase_count == 2 and self.liquid_count == 1 and self.gas is not None
        self.VL_only_CEOSs = (self.VL_only and gas and liquids and isinstance(self.liquids[0], EOSLiquid) and isinstance(self.gas, EOSGas))


    def flash(self, zs=None, T=None, P=None, VF=None, SF=None, V=None, H=None,
              S=None, U=None):
        constants, correlations = self.constants, self.correlations
        settings = self.settings
        if zs is None:
            zs = [1.0]

        T_spec = T is not None
        P_spec = P is not None
        V_spec = V is not None
        H_spec = H is not None
        S_spec = S is not None
        U_spec = U is not None
        VF_spec = VF is not None
        SF_spec = SF is not None

        flash_specs = {'zs': zs}
        if T_spec:
            flash_specs['T'] = T
        if P_spec:
            flash_specs['P'] = P
        if V_spec:
            flash_specs['V'] = V
        if H_spec:
            flash_specs['H'] = H
        if S_spec:
            flash_specs['S'] = S
        if U_spec:
            flash_specs['U'] = U
        if VF_spec:
            flash_specs['VF'] = VF
        if SF_spec:
            flash_specs['SF'] = SF

        if ((T_spec and (P_spec or V_spec)) or (P_spec and V_spec)):            
            g, ls, ss, betas = self.flash_TPV(T=T, P=P, V=V)
            if g is not None:
                id_phases = [g] + ls + ss
            else:
                id_phases = ls + ss
            
            g, ls, ss, betas = identify_sort_phases(id_phases, betas, constants,
                                                    correlations, settings=settings)
            
            a_phase = id_phases[0]
            T, P = a_phase.T, a_phase.P
            flash_convergence = {'iterations': 0, 'err': 0}
            return EquilibriumState(T, P, zs, gas=g, liquids=ls, solids=ss, 
                                    betas=betas, flash_specs=flash_specs, 
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    flasher=self)
            
        elif T_spec and VF_spec:
            # All dew/bubble are the same with 1 component
            Psat, l, g, iterations, err = self.flash_TVF(T)
            flash_convergence = {'iterations': iterations, 'err': err}
            
            return EquilibriumState(T, Psat, zs, gas=g, liquids=[l], solids=[], 
                                    betas=[VF, 1-VF], flash_specs=flash_specs, 
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    flasher=self)
            
        elif P_spec and VF_spec:
            # All dew/bubble are the same with 1 component
            Tsat, l, g, iterations, err = self.flash_PVF(P)
            flash_convergence = {'iterations': iterations, 'err': err}

            return EquilibriumState(Tsat, P, zs, gas=g, liquids=[l], solids=[], 
                                    betas=[VF, 1-VF], flash_specs=flash_specs, 
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    flasher=self)
        elif T_spec and SF_spec:
            Psub, other_phase, s, iterations, err = self.flash_TSF(T)
            if isinstance(other_phase, gas_phases):
                g, liquids = other_phase, []
            else:
                g, liquids = None, [other_phase]
            flash_convergence = {'iterations': iterations, 'err': err}
#
            return EquilibriumState(T, Psub, zs, gas=g, liquids=liquids, solids=[s], 
                                    betas=[1-SF, SF], flash_specs=flash_specs, 
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    flasher=self)
        elif P_spec and SF_spec:
            Tsub, other_phase, s, iterations, err = self.flash_PSF(P)
            if isinstance(other_phase, gas_phases):
                g, liquids = other_phase, []
            else:
                g, liquids = None, [other_phase]
            flash_convergence = {'iterations': iterations, 'err': err}
#
            return EquilibriumState(Tsub, P, zs, gas=g, liquids=liquids, solids=[s], 
                                    betas=[1-SF, SF], flash_specs=flash_specs, 
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    flasher=self)
        
        single_iter_key = (T_spec, P_spec, V_spec, H_spec, S_spec, U_spec)
        if single_iter_key in spec_to_iter_vars:
            fixed_var, spec, iter_var = spec_to_iter_vars[single_iter_key]
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
                
                
            g, ls, ss, betas, T, P, flash_convergence = self.flash_TPV_HSGUA(fixed_var_val, spec_val, fixed_var, spec, iter_var)
            return EquilibriumState(T, P, zs, gas=g, liquids=ls, solids=ss, 
                                    betas=betas, flash_specs=flash_specs, 
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    flasher=self)

        if T is not None and S is not None:
            pass
        elif P is not None and S is not None:
            pass
        elif P is not None and H is not None:
            g, ls, ss, betas, T, flash_convergence = self.flash_PH(P, H)
#            print(g, ls, ss, betas)
            if g is not None:
                phases = [g] + ls + ss
            else:
                phases = ls + ss
            
            g, ls, ss, betas = identify_sort_phases(phases, betas, constants,
                                                    correlations, settings=settings)
            
            return EquilibriumState(T, P, zs, gas=g, liquids=ls, solids=ss, 
                                    betas=betas, flash_specs=flash_specs, 
                                    flash_convergence=flash_convergence,
                                    constants=constants, correlations=correlations,
                                    flasher=self)
            return 
        else:
            raise Exception('Flash inputs unsupported')

    def flash_TPV(self, T, P, V):
        zs = [1]
        liquids = []
        solids = []

        if self.gas_count:
            gas = self.gas.to_zs_TPV(zs=zs, T=T, P=P, V=V)
            G_min, lowest_phase = gas.G(), gas
        else:
            G_min, lowest_phase = 1e100, None
            gas = None
        for l in self.liquids:
            l = l.to_zs_TPV(zs=zs, T=T, P=P, V=V)
            G = l.G()
            if G < G_min:
                G_min, lowest_phase = G, l
            liquids.append(l)

        for s in self.solids:
            s = s.to_zs_TPV(zs=zs, T=T, P=P, V=V)
            G = s.G()
            if G < G_min:
                G_min, lowest_phase = G, s
            solids.append(s)
        
        betas = [1]
        if lowest_phase is gas:
            return lowest_phase, [], [], betas
        elif lowest_phase in liquids:
            return None, [lowest_phase], [], betas
        else:
            return None, [], [lowest_phase], betas


    
    def flash_TVF(self, T, VF=None):
        zs = [1]
        if self.VL_only_CEOSs:
            # Two phase pure eoss are two phase up to the critical point only! Then one phase
            Psat = self.gas.eos_pures_STP[0].Psat(T)
#            
        else:
            Psat = self.correlations.VaporPressures[0](T)
        
        gas = self.gas.to_TP_zs(T, Psat, zs)
        liquids = [l.to_TP_zs(T, Psat, zs) for l in self.liquids]

        if self.VL_only_CEOSs and 0:
            return Psat, liquids[0], gas, 0, 0.0
#        return TVF_pure_newton(Psat, T, liquids, gas, maxiter=200, xtol=1E-10)
        return TVF_pure_secant(Psat, T, liquids, gas, maxiter=200, xtol=1E-10)

    def flash_PVF(self, P, VF=None):
        zs = [1]
        if self.VL_only_CEOSs:
            Tsat = self.gas.eos_pures_STP[0].Tsat(P)
        else:
            Tsat = self.correlations.VaporPressures[0].solve_prop(P)
        gas = self.gas.to_TP_zs(Tsat, P, zs)
        liquids = [l.to_TP_zs(Tsat, P, zs) for l in self.liquids]
        
        if self.VL_only_CEOSs and 0:
            return Tsat, liquids[0], gas, 0, 0.0
        return PVF_pure_newton(Tsat, P, liquids, gas, maxiter=200, xtol=1E-10)
#        return PVF_pure_secant(Tsat, P, liquids, gas, maxiter=200, xtol=1E-10)

    def flash_TSF(self, T, SF=None):
        # if under triple point search for gas - otherwise search for liquid
        # For water only there is technically two solutions at some point for both
        # liquid and gas, flag?
        
        # The solid-liquid interface is NOT working well...
        # Worth getting IAPWS going to compare. Maybe also other EOSs
        zs = [1]
        if T < self.constants.Tts[0]:
            Psub = self.correlations.SublimationPressures[0](T)
            try_phases = [self.gas] + self.liquids
        else:
            try_phases = self.liquids
            Psub = 1e6
        
        return TSF_pure_newton(Psub, T, try_phases, self.solids, 
                               maxiter=200, xtol=1E-10)
        
    def flash_PSF(self, P, SF=None):
        zs = [1]
        if P < self.constants.Pts[0]:
            Tsub = self.correlations.SublimationPressures[0].solve_prop(P)
            try_phases = [self.gas] + self.liquids
        else:
            try_phases = self.liquids
            Tsub = 1e6
        
        return PSF_pure_newton(Tsub, P, try_phases, self.solids, 
                               maxiter=200, xtol=1E-10)
        
    def flash_TPV_HSGUA(self, fixed_var_val, spec_val, fixed_var='P', spec='H', iter_var='T'):
        zs = [1]
        constants, correlations = self.constants, self.correlations

        try:
            solutions_1P = []
            G_min = 1e100
            results_G_min_1P = None
            for phase in self.phases:
                try:                    
                    T, P, phase, iterations, err = solve_PTV_HSGUA_1P(phase, zs, fixed_var_val, spec_val, fixed_var=fixed_var, 
                                                                      spec=spec, iter_var=iter_var, constants=constants, correlations=correlations)
                    G = phase.G()
                    if G < G_min:
                        G_min = G
                        results_G_min_1P = (T, phase, iterations, err)
                    solutions_1P.append([T, phase, iterations, err])
                except Exception as e:
#                    print(e)
                    solutions_1P.append(None)
        except:
            pass
        
        
        try:
            G_VL = 1e100
            # BUG - P IS NOW KNOWN!
            if self.gas_count and self.liquid_count:
                if fixed_var == 'P':
                    Psat, VL_liq, VL_gas, VL_iter, VL_err = self.flash_TVF(fixed_var_val, VF=.5)
                elif fixed_var == 'T':
                    Tsat, VL_liq, VL_gas, VL_iter, VL_err = self.flash_PVF(fixed_var_val, VF=.5)
            elif fixed_var == 'V':
                raise NotImplementedError("Does not make sense here because there is no actual vapor frac spec")
                
#                
#                VL_flash = self.flash(P=P, VF=.4)
                spec_val_l = getattr(VL_liq, spec)()
                spec_val_g = getattr(VL_gas, spec)()
#                spec_val_l = getattr(VL_flash.liquid0, spec)()
#                spec_val_g = getattr(VL_flash.gas, spec)()
                VF = (spec_val - spec_val_l)/(spec_val_g - spec_val_l)
                if 0.0 <= VF <= 1.0:
                    G_l = VL_liq.G()
                    G_g = VL_gas.G()
                    G_VL = G_g*VF + G_l*(1.0 - VF)           
            else:
                VF = None
        except:
            VF = None
        
        try:
            G_SF = 1e100
            if self.solid_count and (self.gas_count or self.liquid_count):
                VS_flash = self.flash(P=P, SF=1)
                spec_val_s = getattr(VS_flash.solid0, spec)()
                spec_other = getattr(VS_flash.phases[0], spec)()
                SF = (spec_val - spec_val_s)/(spec_other - spec_val_s)
                if SF < 0.0 or SF > 1.0:
                    raise ValueError("Not apply")
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
            T, phase, iterations, err = results_G_min_1P
            if isinstance(phase, gas_phases):
                gas_phase = results_G_min_1P[1]
            elif isinstance(phase, liquid_phases):
                ls = [results_G_min_1P[1]]
            elif isinstance(phase, solid_phases):
                ss = [results_G_min_1P[1]]
        
        flash_convergence = {}
        if G_VL < G_min:
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
        flash_convergence['iterations'] = iterations
        flash_convergence['err'] = err

        return gas_phase, ls, ss, betas, T, P, flash_convergence
        

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
        
        Tsat = Psat_obj.solve_prop(P)
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
    
    
    def debug_PT(self, zs, Pmin=None, Pmax=None, Tmin=None, Tmax=None, pts=50, 
                ignore_errors=True, values=False): # pragma: no cover
        if not has_matplotlib and not values:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if Pmin is None:
            Pmin = 1e4
        if Pmax is None:
            Pmax = min(self.constants.Pcs)
        if Tmin is None:
            Tmin = min(self.constants.Tms)*.9
        if Tmax is None:
            Tmax = max(self.constants.Tcs)*1.5
            
        Ps = logspace(log10(Pmin), log10(Pmax), pts)
        Ts = linspace(Tmin, Tmax, pts)
        
        matrix = []
        for T in Ts:
            row = []
            for P in Ps:
                try:
                    state = self.flash(T=T, P=P, zs=zs)
                    row.append(state.phases_str)
                except Exception as e:
                    if ignore_errors:
                        row.append('F')
                    else:
                        raise e
            matrix.append(row)
            
        if values:
            return Ts, Ps, matrix
        
        regions = {'V': 0, 'L': 1, 'S': 2, 'VL': 3, 'LL': 4, 'VLL': 5,
                       'VLS': 6, 'VLLS': 7, 'VLLSS': 8, 'F': -1}

        used_regions = set([])
        for row in matrix:
            for v in row:
                used_regions.add(v)
        
        region_keys = list(regions.keys())
        used_keys = [i for i in region_keys if i in used_regions]
        
        regions_keys = [n for _, n in sorted(zip([regions[i] for i in used_keys], used_keys))]
        used_values = [regions[i] for i in regions_keys]

        dat = [[regions[matrix[i][j]] for j in range(pts)] for i in range(pts)]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        Ts, Ps = np.meshgrid(Ts, Ps)
        cs = ax.contourf(Ts, Ps, dat, levels=list(sorted(regions.values())))
        cbar = fig.colorbar(cs)
        
        cbar.ax.set_yticklabels([n for _, n in sorted(zip(regions.values(), regions.keys()))])
#        cbar.ax.set_yticklabels(regions_keys)
        ax.set_yscale('log')
#        plt.imshow(dat, interpolation='nearest')
#        plt.legend(loc='best', fancybox=True, framealpha=0.5)
#        return fig, ax       
        plt.show()
        
    # ph - iterate on PT
    # if oscillating, take those two phases, solve, then get VF
    # other strategy - guess phase, solve h, PT at point to vonfirm!
    # For one phase - solve each phase for H, if there is a solution.
    # Take the one with lowest Gibbs energy
    

LASTOVKA_SHAW = 'Lastovka Shaw'
DADGOSTAR_SHAW_1 = 'Dadgostar Shaw 1'
STP_T_GUESS = '298.15 K'
LAST_CONVERGED = 'Last converged'
FIXED_GUESS = 'Fixed guess'
IG_ENTHALPY = 'Ideal gas'
IDEAL_LIQUID_ENTHALPY = 'Ideal liquid'

PH_T_guesses_1P_methods = [LASTOVKA_SHAW, DADGOSTAR_SHAW_1, IG_ENTHALPY,
                           IDEAL_LIQUID_ENTHALPY, FIXED_GUESS, STP_T_GUESS,
                           LAST_CONVERGED]
TPV_HSGUA_guesses_1P_methods = PH_T_guesses_1P_methods

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
                T_calc += zs[i]*VolumeLiquids[i].solve_prop(V)
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
    except (UnconvergedError,) as e:
        # G and A specs are NOT MONOTONIC and the brackets will likely NOT BRACKET
        # THE ROOTS!
        return brenth(err, min_bound, max_bound, xtol=xtol, ytol=ytol, maxiter=maxiter)
