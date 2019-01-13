# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['GCEOSMIX', 'PRMIX', 'SRKMIX', 'PR78MIX', 'VDWMIX', 'PRSVMIX', 
'PRSV2MIX', 'TWUPRMIX', 'TWUSRKMIX', 'APISRKMIX',
'eos_Z_test_phase_stability', 'eos_Z_trial_phase_stability']

import sys
import numpy as np
from cmath import log as clog
from scipy.optimize import minimize
from scipy.misc import derivative
from fluids.numerics import IS_PYPY, newton_system
from thermo.utils import normalize, Cp_minus_Cv, isobaric_expansion, isothermal_compressibility, phase_identification_parameter
from thermo.utils import R, UnconvergedError
from thermo.utils import log, exp, sqrt
from thermo.eos import *
from thermo.activity import Wilson_K_value, K_value, flash_inner_loop, Rachford_Rice_flash_error

R2 = R*R
R_inv = 1.0/R
R2_inv = R_inv*R_inv

two_root_two = 2*2**0.5
root_two = sqrt(2.)
root_two_m1 = root_two - 1.0
root_two_p1 = root_two + 1.0
log_min = log(sys.float_info.min)


class GCEOSMIX(GCEOS):
    r'''Class for solving a generic pressure-explicit three-parameter cubic 
    equation of state for a mixture. Does not implement any parameters itself;  
    must be subclassed by a mixture equation of state class which subclasses it.
    No routines for partial molar properties for a generic cubic equation of
    state have yet been implemented, although that would be desireable. 
    The only partial molar property which is currently used is fugacity, which
    must be implemented in each mixture EOS that subclasses this.
    
    .. math::
        P=\frac{RT}{V-b}-\frac{a\alpha(T)}{V^2 + \delta V + \epsilon}

    Main methods are `fugacities`, `solve_T`, and `a_alpha_and_derivatives`.
    
    `fugacities` is a helper method intended as a common interface for setting
    fugacities of each species in each phase; it calls `fugacity_coefficients`
    to actually calculate them, but that is not implemented here. This should
    be used when performing flash calculations, where fugacities are needed 
    repeatedly. The fugacities change as a function of liquid/gas phase 
    composition, but the entire EOS need not be solved to recalculate them.
    
    `solve_T` is a wrapper around `GCEOS`'s `solve_T`; the only difference is 
    to use half the average mixture's critical temperature as the initial 
    guess.
    
    `a_alpha_and_derivatives` implements the Van der Waals mixing rules for a
    mixture. It calls `a_alpha_and_derivatives` from the pure-component EOS for 
    each species via multiple inheritance.
    '''
    nonstate_constants = ('N', 'cmps', 'Tcs', 'Pcs', 'omegas', 'kijs', 'kwargs', 'ais', 'bs')
    
    def fast_copy_base(self, a_alphas=False):
        new = self.__class__.__new__(self.__class__)
        for attr in self.nonstate_constants:
            setattr(new, attr, getattr(self, attr))
        for attr in self.nonstate_constants_specific:
            setattr(new, attr, getattr(self, attr))
        if a_alphas:
            new.a_alphas = self.a_alphas
            new.da_alpha_dTs = self.da_alpha_dTs
            new.d2a_alpha_dT2s = self.d2a_alpha_dT2s
        return new
    
    def to_TP_zs_fast(self, T, P, zs, only_l=False, only_g=False):
        copy_alphas = T == self.T
        new = self.fast_copy_base(a_alphas=copy_alphas)
        new.T = T
        new.P = P
        new.V = None
        new.zs = zs
        new.fast_init_specific()
        new.solve(pure_a_alphas=(not copy_alphas), only_l=only_l, only_g=only_g)
        return new



    def to_TP_zs(self, T, P, zs):
#        print(T, self.T, P, self.P, zs, self.zs)
        if T != self.T or P != self.P or zs != self.zs:
            return self.__class__(T=T, P=P, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, zs=zs, **self.kwargs)
        else:
            return self
    
    
    def to_TP_pure(self, T, P, i):
        kwargs = {} # TODO write function to get those
        return self.eos_pure(T=T, P=P, Tc=self.Tcs[i], Pc=self.Pcs[i],
                             omega=self.omegas[i])
    
    def a_alpha_and_derivatives_numpy(self, a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, full=True, quick=True):
        zs, kijs = self.zs, np.array(self.kijs)
        a_alphas = np.array(a_alphas)
        da_alpha_dTs = np.array(da_alpha_dTs)
        one_minus_kijs = 1.0 - kijs
        
        x0 = np.einsum('i,j', a_alphas, a_alphas)
        x0_05 = x0**0.5
        a_alpha_ijs = (one_minus_kijs)*x0_05
        z_products = np.einsum('i,j', zs, zs)
        a_alpha = np.einsum('ij,ji', a_alpha_ijs, z_products)

        self.a_alpha_ijs = a_alpha_ijs.tolist()
        
        if full:            
            term0 = np.einsum('j,i', a_alphas, da_alpha_dTs)
            term7 = (one_minus_kijs)/(x0_05)
            da_alpha_dT = (z_products*term7*(term0)).sum()
            
            term1 = -x0_05/x0*(one_minus_kijs)
            
            term2 = np.einsum('i, j', a_alphas, da_alpha_dTs)
                        
            main3 = da_alpha_dTs/(2.0*a_alphas)*term2
            main4 = -np.einsum('i, j', a_alphas, d2a_alpha_dT2s)
            main6 = -0.5*np.einsum('i, j', da_alpha_dTs, da_alpha_dTs)
            
            # Needed for fugacity temperature derivative
            self.da_alpha_dT_ijs = (0.5*(term7)*(term2 + term0))
            
            d2a_alpha_dT2 = (z_products*(term1*(main3 + main4 + main6))).sum()
        
            return float(a_alpha), float(da_alpha_dT), float(d2a_alpha_dT2)
        else:
            return float(a_alpha)



    def a_alpha_and_derivatives(self, T, full=True, quick=True,
                                pure_a_alphas=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for an EOS with the Van der Waals mixing rules. Uses the
        parent class's interface to compute pure component values. Returns
        `a_alpha`, `da_alpha_dT`, and `d2a_alpha_dT2`. Calls 
        `setup_a_alpha_and_derivatives` before calling
        `a_alpha_and_derivatives` for each species, which typically sets `a` 
        and `Tc`. Calls `cleanup_a_alpha_and_derivatives` to remove the set
        properties after the calls are done.
        
        For use in `solve_T` this returns only `a_alpha` if `full` is False.
        
        .. math::
            a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}
            
            (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}
        
        Parameters
        ----------
        T : float
            Temperature, [K]
        full : bool, optional
            If False, calculates and returns only `a_alpha`
        quick : bool, optional
            Only the quick variant is implemented; it is little faster anyhow
        pure_a_alphas : bool, optional
            Whether or not to recalculate the a_alpha terms of pure components
            (for the case of mixtures only) which stay the same as the 
            composition changes (i.e in a PT flash), [-]
        
        Returns
        -------
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific 
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by  
            EOS-specific method, [J^2/mol^2/Pa/K**2]

        Notes
        -----
        The exact expressions can be obtained with the following SymPy 
        expression below, commented out for brevity.
        
        >>> from sympy import *
        >>> a_alpha_i, a_alpha_j, kij, T = symbols('a_alpha_i, a_alpha_j, kij, T')
        >>> a_alpha_ij = (1-kij)*sqrt(a_alpha_i(T)*a_alpha_j(T))
        >>> #diff(a_alpha_ij, T)
        >>> #diff(a_alpha_ij, T, T)
        '''
        if pure_a_alphas:
            try:
                # TODO do not compute derivatives if full=False
                a_alphas, da_alpha_dTs, d2a_alpha_dT2s = self.a_alpha_and_derivatives_vectorized(T, full=True)
            except:
                a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [], [], []
                method_obj = super(type(self).__mro__[self.a_alpha_mro], self)
                for i in self.cmps:
                    self.setup_a_alpha_and_derivatives(i, T=T)
                    # Abuse method resolution order to call the a_alpha_and_derivatives
                    # method of the original pure EOS
                    # -4 goes back from object, GCEOS, SINGLEPHASEEOS, up to GCEOSMIX
                    # 
                    ds = method_obj.a_alpha_and_derivatives_pure(T)
                    a_alphas.append(ds[0])
                    da_alpha_dTs.append(ds[1])
                    d2a_alpha_dT2s.append(ds[2])
                self.cleanup_a_alpha_and_derivatives()
                
            self.a_alphas, self.da_alpha_dTs, self.d2a_alpha_dT2s = a_alphas, da_alpha_dTs, d2a_alpha_dT2s
        else:
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = self.a_alphas, self.da_alpha_dTs, self.d2a_alpha_dT2s
        
        if not IS_PYPY and self.N > 20:
            return self.a_alpha_and_derivatives_numpy(a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, full=full, quick=quick)
        return self.a_alpha_and_derivatives_py(a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, full=full, quick=quick)
        
    def a_alpha_and_derivatives_py(self, a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, full=True, quick=True):
        # For 44 components, takes 150 us in PyPy.; 95 in pythran. Much of that is type conversions.
        # 4 ms pypy for 44*4, 1.3 ms for pythran, 10 ms python with numpy
        # 2 components 1.89 pypy, pythran 1.75 us, regular python 12.7 us.
        # 10 components - regular python 148 us, 9.81 us PyPy, 8.37 pythran in PyPy (flags have no effect; 14.3 us in regular python)
        zs, kijs, cmps, N = self.zs, self.kijs, self.cmps, self.N
        da_alpha_dT, d2a_alpha_dT2 = 0.0, 0.0
        
        a_alpha_ijs = [[None]*N for _ in cmps]
        a_alpha_i_roots = [a_alpha_i**0.5 for a_alpha_i in a_alphas]
        
        if full:
            a_alpha_ij_roots = [[None]*N for _ in cmps]
            for i in cmps:
                kijs_i = kijs[i]
                a_alpha_i = a_alphas[i]
                a_alpha_ijs_is = a_alpha_ijs[i]
                a_alpha_ij_roots_i = a_alpha_ij_roots[i]
                for j in cmps:
                    if j < i:
                        continue
                    a_alpha_ij_roots_i[j] = a_alpha_i_roots[i]*a_alpha_i_roots[j]#(a_alpha_i*a_alphas[j])**0.5 
                    a_alpha_ijs_is[j] = a_alpha_ijs[j][i] = (1. - kijs_i[j])*a_alpha_ij_roots_i[j]
        else:
            for i in cmps:
                kijs_i = kijs[i]
                a_alpha_i = a_alphas[i]
                a_alpha_ijs_is = a_alpha_ijs[i]
                for j in cmps:
                    if j < i:
                        continue
                    a_alpha_ijs_is[j] = a_alpha_ijs[j][i] = (1. - kijs_i[j])*a_alpha_i_roots[i]*a_alpha_i_roots[j]
                
        # Faster than an optimized loop in pypy even
#        print(self.N, self.cmps, zs)
        z_products = [[zs[i]*zs[j] for j in cmps] for i in cmps]

        a_alpha = 0.0
        for i in cmps:
            a_alpha_ijs_i = a_alpha_ijs[i]
            z_products_i = z_products[i]
            for j in cmps:
                if j < i:
                    continue
                elif i != j:
                    a_alpha += 2.0*a_alpha_ijs_i[j]*z_products_i[j]
                else:
                    a_alpha += a_alpha_ijs_i[j]*z_products_i[j]
        
        # List comprehension tested to be faster in CPython not pypy
#        a_alpha = sum([a_alpha_ijs[i][j]*z_products[i][j]
#                      for j in self.cmps for i in self.cmps])
        self.a_alpha_ijs = a_alpha_ijs
        
        da_alpha_dT_ijs = self.da_alpha_dT_ijs = [[None]*N for _ in cmps]
        
        if full:
            for i in cmps:
                kijs_i = kijs[i]
                a_alphai = a_alphas[i]
                z_products_i = z_products[i]
                da_alpha_dT_i = da_alpha_dTs[i]
                d2a_alpha_dT2_i = d2a_alpha_dT2s[i]
                a_alpha_ij_roots_i = a_alpha_ij_roots[i]
                for j in cmps:
                    if j < i:
                        # skip the duplicates
                        continue
                    a_alphaj = a_alphas[j]
                    x0 = a_alphai*a_alphaj
                    x0_05 = a_alpha_ij_roots_i[j]
                    zi_zj = z_products_i[j]
                    
                    x1 = a_alphai*da_alpha_dTs[j]
                    x2 = a_alphaj*da_alpha_dT_i
                    x1_x2 = x1 + x2
                    x3 = 2.0*x1_x2

                    kij_m1 = kijs_i[j] - 1.0
                    
                    da_alpha_dT_ij = -0.5*kij_m1*x1_x2/x0_05
                    
                    # For temperature derivatives of fugacities 
                    da_alpha_dT_ijs[i][j] = da_alpha_dT_ijs[j][i] = da_alpha_dT_ij

                    da_alpha_dT_ij *= zi_zj
                    
                    d2a_alpha_dT2_ij = zi_zj*kij_m1*(-0.25*x0_05*(x0*(
                    2.0*(a_alphai*d2a_alpha_dT2s[j] + a_alphaj*d2a_alpha_dT2_i)
                    + 4.*da_alpha_dT_i*da_alpha_dTs[j]) - x1*x3 - x2*x3 + x1_x2*x1_x2)/(x0*x0))
                    
                    if i != j:
                        da_alpha_dT += da_alpha_dT_ij + da_alpha_dT_ij
                        d2a_alpha_dT2 += d2a_alpha_dT2_ij + d2a_alpha_dT2_ij
                    else:
                        da_alpha_dT += da_alpha_dT_ij
                        d2a_alpha_dT2 += d2a_alpha_dT2_ij
        
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
        else:
            return a_alpha
        
        
        
    def mechanical_critical_point_f_jac(self, TP):
        '''The criteria for c_goal and d_goal come from a cubic
        'roots_cubic', which uses a `f`, `g`, and `h` parameter. When all of 
        them are zero, all three roots are equal. For the eos (a=1), this
        results in the following system of equations:
    
        from sympy import *
        a = 1
        b, c, d = symbols('b, c, d')
        f = ((3* c / a) - ((b ** 2) / (a ** 2))) / 3
        g = (((2 * (b ** 3)) / (a ** 3)) - ((9* b * c) / (a **2)) + (27 * d / a)) /27
        h = ((g ** 2) / 4 + (f ** 3) / 27)z
        solve([Eq(f, 0), Eq(g, 0), Eq(h, 0)], [b, c, d])       
        
        The solution (sympy struggled) is:
        c = b^2/3
        d = b^3/27
        
        These two variables switch sign at the criteria, so they work well with
        a root finding approach.
        
        
        Derived with:
            
        from sympy import *
        P, T, V, R, b_eos, alpha = symbols('P, T, V, R, b_eos, alpha')
        Tc, Pc, omega = symbols('Tc, Pc, omega')
        delta, epsilon = symbols('delta, epsilon')
        
        a_alpha = alpha(T)
        
        eta = b_eos
        B = b_eos*P/(R*T)
        deltas = delta*P/(R*T)
        thetas = a_alpha*P/(R*T)**2
        epsilons = epsilon*(P/(R*T))**2
        etas = eta*P/(R*T)
        
        b = (deltas - B - 1)
        c = (thetas + epsilons - deltas*(B + 1))
        d = -(epsilons*(B + 1) + thetas*etas)
        
        c_goal = b*b/3
        d_goal = b*b*b/27
        
        F1 = c - c_goal
        F2 = d - d_goal
        
        cse([F1, F2, diff(F1, T), diff(F1, P), diff(F2, T), diff(F2, P)], optimizations='basic')

            
            
            
            
            
        Performance analysis:
            
        77% of this is getting a_alpha and da_alpha_dT.
        71% of the outer solver is getting f and this Jacobian.
        Limited results from optimizing the below code, which was derived with
        sympy.
        '''
        T, P = float(TP[0]), float(TP[1])
        b_eos, delta, epsilon = self.b, self.delta, self.epsilon
        eta = b_eos
        
        
        a_alpha, da_alpha_dT, _ = self.a_alpha_and_derivatives(T, full=True)
        
        
        x6 = R_inv
        x7 = 1.0/T
        x0 = a_alpha
        x1 = R_inv*R_inv
        x2 = x7*x7
        x3 = x1*x2
        x4 = P*P
        x5 = epsilon*x3*x4
        x8 = P*x6*x7
        x9 = delta*x8
        x10 = b_eos*x8
        x11 = x10 + 1.0
        x12 = x11 - x9
        x13 = x12*x12
        x14 = P*x2*x6
        x15 = da_alpha_dT
        x16 = x6*x7
        x17 = x0*x16
        x18 = 2.0*epsilon*x8
        x19 = delta*x10
        x20 = delta*x11
        x21 = b_eos - delta
        x22 = 2.0*x12*x21/3.0
        x23 = P*b_eos*x0*x1*x2
        x24 = b_eos*x5
        x25 = x11*x18
        x26 = x13*x21/9.0   
        
        
        F1 = P*x0*x3 - x11*x9 - x13/3.0 + x5
        F2 = -x11*x5 + x13*x12/27.0 - b_eos*x0*x4*x6*x1*x7*x2
        dF1_dT = x14*(x15*x6 - 2.0*x17 - x18 + x19 + x20 + x22)
        dF1_dP = x16*(x17 + x18 - x19 - x20 - x22)
        dF2_dT = x14*(-P*b_eos*x1*x15*x7 + 3.0*x23 + x24 + x25 - x26)
        dF2_dP = x16*(-2.0*x23 - x24 - x25 + x26)
        
        return [F1, F2], [[dF1_dT, dF1_dP], [dF2_dT, dF2_dP]]
        
        
    def mechanical_critical_point(self):
        r'''Method to calculate the mechanical critical point of a mixture
        of defined composition.
        
        The mechanical critical point is where:
            
        .. math::
            \frac{\partial P}{\partial \rho}|_T = 
            \frac{\partial^2 P}{\partial \rho^2}|_T =  0
            
        Returns
        ----------
        T : float
            Mechanical critical temperature, [K]
        P : float
            Mechanical critical temperature, [Pa]
            
        Notes
        -----
        One useful application of the mechanical critical temperature is that
        the pahse identification approach of Venkatarathnam is valid only up to
        it.
        
        Note that the equation of state, when solved at these conditions, will
        have fairly large (1e-3 - 1e-6) results for the derivatives; but they 
        are the minimum. This is just from floating point precision.
        
        It can also be checked looking at the calculated molar volumes - all 
        three (available with `sorted_volumes`) will be very close (1e-5
        difference in practice), again differing because of floating point
        error.
        
        The algorithm here is a custom implementation, using Newton-Raphson's
        method with the initial guesses described in [1] (mole-weighted 
        critical pressure average, critical temperature average using a 
        quadratic mixing rule). Normally ~4 iterations are needed to solve the
        system. It is relatively fast, as only one evaluation of `a_alpha`
        and `da_alpha_dT` are needed per call to function and its jacobian.        
             
        References
        ----------
        .. [1] Watson, Harry A. J., and Paul I. Barton. "Reliable Flash 
           Calculations: Part 3. A Nonsmooth Approach to Density Extrapolation 
           and Pseudoproperty Evaluation." Industrial & Engineering Chemistry 
           Research, November 11, 2017.
           https://doi.org/10.1021/acs.iecr.7b03233.
        .. [2] Mathias P. M., Boston J. F., and Watanasiri S. "Effective
           Utilization of Equations of State for Thermodynamic Properties in
           Process Simulation." AIChE Journal 30, no. 2 (June 17, 2004):
           182-86. https://doi.org/10.1002/aic.690300203.
        '''
        
        Pmc = sum([self.Pcs[i]*self.zs[i] for i in self.cmps])
        Tmc = sum([(self.Tcs[i]*self.Tcs[j])**0.5*self.zs[j]*self.zs[i] for i in self.cmps
                  for j in self.cmps])
        TP, iterations = newton_system(self.mechanical_critical_point_f_jac,
                                       x0=[Tmc, Pmc], jac=True, ytol=1e-10)
        T, P = float(TP[0]), float(TP[1])
        return T, P
        
    def to_mechanical_critical_point(self):
        T, P = self.mechanical_critical_point()
        return self.to_TP_zs(T=T, P=P, zs=self.zs)
        
        
    def fugacities(self, xs=None, ys=None):   
        r'''Helper method for calculating fugacity coefficients for any 
        phases present, using either the overall mole fractions for both phases
        or using specified mole fractions for each phase.
        
        Requires `fugacity_coefficients` to be implemented by each subclassing
        EOS.
        
        In addition to setting `fugacities_l` and/or `fugacities_g`, this also
        sets the fugacity coefficients `phis_l` and/or `phis_g`.
        
        .. math::
            \hat \phi_i^g = \frac{\hat f_i^g}{x_i P}
        
            \hat \phi_i^l = \frac{\hat f_i^l}{x_i P}
        
        Parameters
        ----------
        xs : list[float], optional
            Liquid-phase mole fractions of each species, [-]
        ys : list[float], optional
            Vapor-phase mole fractions of each species, [-]
            
        Notes
        -----
        It is helpful to check that `fugacity_coefficients` has been
        implemented correctly using the following expression, from [1]_.
        
        .. math::
            \ln \hat \phi_i = \left[\frac{\partial (n\log \phi)}{\partial 
            n_i}\right]_{T,P,n_j,V_t}
        
        For reference, several expressions for fugacity of a component are as
        follows, shown in [1]_ and [2]_.
        
        .. math::
             \ln \hat \phi_i = \int_{0}^P\left(\frac{\hat V_i}
             {RT} - \frac{1}{P}\right)dP

             \ln \hat \phi_i = \int_V^\infty \left[
             \frac{1}{RT}\frac{\partial P}{ \partial n_i}
             - \frac{1}{V}\right] d V - \ln Z
             
        References
        ----------
        .. [1] Hu, Jiawen, Rong Wang, and Shide Mao. "Some Useful Expressions 
           for Deriving Component Fugacity Coefficients from Mixture Fugacity 
           Coefficient." Fluid Phase Equilibria 268, no. 1-2 (June 25, 2008): 
           7-13. doi:10.1016/j.fluid.2008.03.007.
        .. [2] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        '''
        P = self.P
        if self.phase in ('l', 'l/g'):
            if xs is None:
                xs = self.zs
            if hasattr(self, 'Z_l'):
                self.lnphis_l = self.fugacity_coefficients(self.Z_l, zs=xs)
                self.phis_l = [exp(i) for i in self.lnphis_l]
                self.fugacities_l = [phi*x*P for phi, x in zip(self.phis_l, xs)]

        if self.phase in ('g', 'l/g'):
            if ys is None:
                ys = self.zs
            if hasattr(self, 'Z_g'):
                self.lnphis_g = self.fugacity_coefficients(self.Z_g, zs=ys)
                self.phis_g = [exp(i) for i in self.lnphis_g]
                self.fugacities_g = [phi*y*P for phi, y in zip(self.phis_g, ys)]


    def eos_fugacities_lowest_Gibbs(self):        
        try:
            if self.G_dep_l < self.G_dep_g:
                return self.fugacities_l, 'l'
            else:
                return self.fugacities_g, 'g'
        except:
            # Only one root - take it and set the prefered other phase to be a different type
            return (self.fugacities_g, 'g') if hasattr(self, 'Z_g') else (self.fugacities_l, 'l')


    def _dphi_dn(self, zi, i, phase):
        z_copy = list(self.zs)
        z_copy.pop(i)
        z_sum = sum(z_copy) + zi
        z_copy = [j/z_sum if j else 0 for j in z_copy]
        z_copy.insert(i, zi)
        
        eos = self.to_TP_zs(self.T, self.P, z_copy)
        if phase == 'g':
            return eos.phis_g[i]
        elif phase == 'l':
            return eos.phis_l[i]

    def _dfugacity_dn(self, zi, i, phase):
        z_copy = list(self.zs)
        z_copy.pop(i)
        z_sum = sum(z_copy) + zi
        z_copy = [j/z_sum if j else 0 for j in z_copy]
        z_copy.insert(i, zi)
        
        eos = self.to_TP_zs(self.T, self.P, z_copy)
        if phase == 'g':
            return eos.fugacities_g[i]
        elif phase == 'l':
            return eos.fugacities_l[i]


    def fugacities_partial_derivatives(self, xs=None, ys=None):
        if self.phase in ['l', 'l/g']:
            if xs is None:
                xs = self.zs
            self.dphis_dni_l = [derivative(self._dphi_dn, xs[i], args=[i, 'l'], dx=1E-7, n=1) for i in self.cmps]
            self.dfugacities_dni_l = [derivative(self._dfugacity_dn, xs[i], args=[i, 'l'], dx=1E-7, n=1) for i in self.cmps]
            self.dlnphis_dni_l = [dphi/phi for dphi, phi in zip(self.dphis_dni_l, self.phis_l)]
        if self.phase in ['g', 'l/g']:
            if ys is None:
                ys = self.zs
            self.dphis_dni_g = [derivative(self._dphi_dn, ys[i], args=[i, 'g'], dx=1E-7, n=1) for i in self.cmps]
            self.dfugacities_dni_g = [derivative(self._dfugacity_dn, ys[i], args=[i, 'g'], dx=1E-7, n=1) for i in self.cmps]
            self.dlnphis_dni_g = [dphi/phi for dphi, phi in zip(self.dphis_dni_g, self.phis_g)]
            # confirmed the relationship of the above 
            # There should be an easy way to get dfugacities_dn_g but I haven't figured it out

    def fugacities_partial_derivatives_2(self, xs=None, ys=None):
        if self.phase in ['l', 'l/g']:
            if xs is None:
                xs = self.zs
            self.d2phis_dni2_l = [derivative(self._dphi_dn, xs[i], args=[i, 'l'], dx=1E-5, n=2) for i in self.cmps]
            self.d2fugacities_dni2_l = [derivative(self._dfugacity_dn, xs[i], args=[i, 'l'], dx=1E-5, n=2) for i in self.cmps]
            self.d2lnphis_dni2_l = [d2phi/phi  - dphi*dphi/(phi*phi) for d2phi, dphi, phi in zip(self.d2phis_dni2_l, self.dphis_dni_l, self.phis_l)]
        if self.phase in ['g', 'l/g']:
            if ys is None:
                ys = self.zs
            self.d2phis_dni2_g = [derivative(self._dphi_dn, ys[i], args=[i, 'g'], dx=1E-5, n=2) for i in self.cmps]
            self.d2fugacities_dni2_g = [derivative(self._dfugacity_dn, ys[i], args=[i, 'g'], dx=1E-5, n=2) for i in self.cmps]
            self.d2lnphis_dni2_g = [d2phi/phi  - dphi*dphi/(phi*phi) for d2phi, dphi, phi in zip(self.d2phis_dni2_g, self.dphis_dni_g, self.phis_g)]
        # second derivative lns confirmed

    def TPD(self, Zz, Zy, zs, ys):
        r'''Helper method for calculating the Tangent Plane Distance function
        according to the original Michelsen definition. More advanced 
        transformations of the TPD function are available in the literature for
        performing calculations. This method does not alter the state of the 
        object.
        
        .. math::
            \text{TPD}(y) =  \sum_{j=1}^n y_j(\mu_j (y) - \mu_j(z))
            = RT \sum_i y_i\left(\log(y_i) + \log(\phi_i(y)) - d_i(z)\right)
            
            d_i(z) = \ln z_i + \ln \phi_i(z)
            
        Parameters
        ----------
        Zz : float
            Compressibility factor of the phase undergoing stability testing
            (`test` phase), [-]
        Zy : float
            Compressibility factor of the trial phase, [-]
        zs : list[float]
            Mole fraction composition of the phase undergoing stability 
            testing  (`test` phase), [-]
        ys : list[float]
            Mole fraction trial phase composition, [-]
        
        Returns
        -------
        TBP : float
            Original Tangent Plane Distance function, [J/mol]
            
        Notes
        -----
        A dimensionless version of this is often used as well, divided by
        RT.
        
        References
        ----------
        .. [1] Michelsen, Michael L. "The Isothermal Flash Problem. Part I. 
           Stability." Fluid Phase Equilibria 9, no. 1 (December 1982): 1-19.
        .. [2] Hoteit, Hussein, and Abbas Firoozabadi. "Simple Phase Stability
           -Testing Algorithm in the Reduction Method." AIChE Journal 52, no. 
           8 (August 1, 2006): 2909-20.
        '''
        z_log_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_log_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        
        tot = 0
        for yi, phi_yi, zi, phi_zi in zip(ys, y_log_fugacity_coefficients, zs, z_log_fugacity_coefficients):
            di = log(zi) + phi_zi
            tot += yi*(log(yi) + phi_yi - di)
        return tot*R*self.T
    
    def Stateva_Tsvetkov_TPDF(self, Zz, Zy, zs, ys):
        r'''Modified Tangent Plane Distance function according to [1]_ and
        [2]_. The stationary points of a system are all zeros of this function;
        so once all zeroes have been located, the stability can be evaluated
        at the stationary points only. It may be required to use multiple 
        guesses to find all stationary points, and there is no method of
        confirming all points have been found.
        
        This method does not alter the state of the object.
        
        .. math::
            \phi(y) = \sum_i^{N} (k_{i+1}(y) - k_i(y))^2
            
            k_i(y) = \ln \phi_i(y) + \ln(y_i) - d_i
            
            k_{N+1}(y) = k_1(y)

            d_i(z) = \ln z_i + \ln \phi_i(z)
            
        Parameters
        ----------
        Zz : float
            Compressibility factor of the phase undergoing stability testing,
             (`test` phase), [-]
        Zy : float
            Compressibility factor of the trial phase, [-]
        zs : list[float]
            Mole fraction composition of the phase undergoing stability 
            testing  (`test` phase), [-]
        ys : list[float]
            Mole fraction trial phase composition, [-]
        
        Returns
        -------
        TPDF_Stateva_Tsvetkov : float
            Modified Tangent Plane Distance function according to [1]_, [-]
            
        Notes
        -----
        In [1]_, a typo omitted the squaring of the expression. This method
        produces very interesting plots matching the shapes given in 
        literature.
        
        References
        ----------
        .. [1] Ivanov, Boyan B., Anatolii A. Galushko, and Roumiana P. Stateva.
           "Phase Stability Analysis with Equations of State-A Fresh Look from 
           a Different Perspective." Industrial & Engineering Chemistry 
           Research 52, no. 32 (August 14, 2013): 11208-23.
        .. [2] Stateva, Roumiana P., and Stefan G. Tsvetkov. "A Diverse 
           Approach for the Solution of the Isothermal Multiphase Flash 
           Problem. Application to Vapor-Liquid-Liquid Systems." The Canadian
           Journal of Chemical Engineering 72, no. 4 (August 1, 1994): 722-34.
        '''
        z_log_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_log_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        
        kis = []
        for yi, phi_yi, zi, phi_zi in zip(ys, y_log_fugacity_coefficients, zs, z_log_fugacity_coefficients):
            di = log(zi) + phi_zi
            try:
                ki = phi_yi + log(yi) - di
            except ValueError:
                ki = phi_yi + log(1e-200) - di
            kis.append(ki)
        kis.append(kis[0])

        tot = 0.0
        for i in range(self.N):
            t = kis[i+1] - kis[i]
            tot += t*t
        return tot

    def d_TPD_dy(self, Zz, Zy, zs, ys):
        # The gradient should be - for all variables
        z_log_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_log_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        gradient = []
        for yi, phi_yi, zi, phi_zi in zip(ys, y_log_fugacity_coefficients, zs, z_log_fugacity_coefficients):
            hi = di = log(zi) + phi_zi # same as di
            k = log(yi) + phi_yi - hi
            Yi = exp(-k)*yi
            gradient.append(phi_yi + log(Yi) - di)
        return gradient

    def d_TPD_Michelson_modified(self, Zz, Zy, zs, alphas):
        r'''Modified objective function for locating the minima of the
        Tangent Plane Distance function according to [1]_, also shown in [2]_
        [2]_. The stationary points of a system are all zeros of this function;
        so once all zeroes have been located, the stability can be evaluated
        at the stationary points only. It may be required to use multiple 
        guesses to find all stationary points, and there is no method of
        confirming all points have been found.
        
        This method does not alter the state of the object.
        
        .. math::
            \frac{\partial \; TPD^*}{\partial \alpha_i} = \sqrt{Y_i} \left[
            \ln \phi_i(Y) + \ln(Y_i) - h_i\right]
            
            \alpha_i = 2 \sqrt{Y_i}
            
            d_i(z) = \ln z_i + \ln \phi_i(z)
            
        Parameters
        ----------
        Zz : float
            Compressibility factor of the phase undergoing stability testing,
             (`test` phase), [-]
        Zy : float
            Compressibility factor of the trial phase, [-]
        zs : list[float]
            Mole fraction composition of the phase undergoing stability 
            testing  (`test` phase), [-]
        alphas : list[float]
            Twice the square root of the mole numbers of each component,
            [mol^0.5]
        
        Returns
        -------
        err : float
            Error in solving for stationary points according to the modified
            TPD method in [1]_, [-]
            
        Notes
        -----
        This method is particularly useful because it is not a constrained
        objective function. This has been verified to return the same roots as
        other stationary point methods.
        
        References
        ----------
        .. [1] Michelsen, Michael L. "The Isothermal Flash Problem. Part I. 
           Stability." Fluid Phase Equilibria 9, no. 1 (December 1982): 1-19.
        .. [2] Qiu, Lu, Yue Wang, Qi Jiao, Hu Wang, and Rolf D. Reitz. 
           "Development of a Thermodynamically Consistent, Robust and Efficient 
           Phase Equilibrium Solver and Its Validations." Fuel 115 (January 1, 
           2014): 1-16
        '''
        Ys = [(alpha/2.)**2 for alpha in alphas]
        ys = normalize(Ys)
        z_log_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_log_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        tot = 0
        for Yi, phi_yi, zi, phi_zi in zip(Ys, y_log_fugacity_coefficients, zs, z_log_fugacity_coefficients):
            di = log(zi) + phi_zi
            if Yi != 0:
                diff = Yi**0.5*(log(Yi) + phi_yi - di)
                tot += abs(diff)
        return tot


    def TDP_Michelsen(self, Zz, Zy, zs, ys):
        
        z_log_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_log_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        tot = 0
        for yi, phi_yi, zi, phi_zi in zip(ys, y_log_fugacity_coefficients, zs, z_log_fugacity_coefficients):
            hi = di = log(zi) + phi_zi # same as di
            
            k = log(yi) + phi_yi - hi
            # Michaelsum doesn't do the exponents.
            Yi = exp(-k)*yi
            tot += Yi*(log(Yi) + phi_yi - hi - 1.)
            
        return 1. + tot

    def TDP_Michelsen_modified(self, Zz, Zy, zs, Ys):
        # https://www.e-education.psu.edu/png520/m17_p7.html
        # Might as well continue
        Ys = [abs(float(Yi)) for Yi in Ys]
        # Ys only need to be positive
        ys = normalize(Ys)
        
        z_log_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_log_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        
        tot = 0
        for Yi, phi_yi, yi, zi, phi_zi in zip(Ys, y_log_fugacity_coefficients, ys, zs, z_log_fugacity_coefficients):
            hi = di = log(zi) + phi_zi # same as di
            tot += Yi*(log(Yi) + phi_yi - di - 1.)
        return (1. + tot)
    # Another formulation, returns the same answers.
#            tot += yi*(log(sum(Ys)) +log(yi)+ log(phi_yi) - di - 1.)
#        return (1. + sum(Ys)*tot)*1e15


    def solve_T(self, P, V, quick=True):
        r'''Generic method to calculate `T` from a specified `P` and `V`.
        Provides SciPy's `newton` solver, and iterates to solve the general
        equation for `P`, recalculating `a_alpha` as a function of temperature
        using `a_alpha_and_derivatives` each iteration.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        quick : bool, optional
            Unimplemented, although it may be possible to derive explicit 
            expressions as done for many pure-component EOS

        Returns
        -------
        T : float
            Temperature, [K]
        '''
        self.Tc = sum(self.Tcs)/self.N
        # -4 goes back from object, GCEOS
        return super(type(self).__mro__[-3], self).solve_T(P=P, V=V, quick=quick)

    
    def _err_VL_jacobian(self, lnKsVF, T, P, zs, near_critical=False):
        lnKs = lnKsVF[:-1]
        Ks = [exp(lnKi) for lnKi in lnKs]
        VF = float(lnKsVF[-1])
        
        
        xs = [zi/(1.0 + VF*(Ki - 1.0)) for zi, Ki in zip(zs, Ks)]
        ys = [Ki*xi for Ki, xi in zip(Ks, xs)]

        eos_g = self.to_TP_zs(T=T, P=P, zs=ys)
        eos_l = self.to_TP_zs(T=T, P=P, zs=xs)
        if not near_critical:
            Z_g = eos_g.Z_g
            Z_l = eos_l.Z_l
        else:
            eos_g = self.to_TP_zs(T=T, P=P, zs=ys)
            eos_l = self.to_TP_zs(T=T, P=P, zs=xs)
            try:
                Z_g = eos_g.Z_g
            except AttributeError:
                Z_g = eos_g.Z_l
            try:
                Z_l = eos_l.Z_l
            except AttributeError:
                Z_l = eos_l.Z_g
        
        size = self.N + 1
        J = [[None]*size for i in range(size)]
        
        d_lnphi_dxs = self.d_lnphi_dzs(Z_l, xs)
        d_lnphi_dys = self.d_lnphi_dzs(Z_g, ys)
        
        
        
#        # Handle the zeros and the ones
        # Half of this is probably wrong! Only gets set for one set of variables?
        # Numerical jacobian not good enough to tell
        for i in range(self.N):
            J[i][-2] = 0.0
            J[-2][i] = 0.0
            
        J[self.N][self.N] = 1.0
        
        # Last column except last value; believed correct except for d_lnphi_dzs
        for i in range(self.N):
            value = 0.0
            for k in range(self.N-1):
                RR_term = zs[k]*(Ks[k] - 1.0)/(1.0 + VF*(Ks[k] - 1.0))**2.0
                # pretty sure indexing is right in the below expression
                diff_term = d_lnphi_dxs[i][k] - Ks[k]*d_lnphi_dys[i][k]
                value += RR_term*diff_term
            J[i][-1] = value
        
        def delta(k, j):
            if k == j:
                return 1.0
            return 0.0
        
        
            
        # Main body - expensive to compute! Lots of elements
        # Can flip around the indexing of i, j on the d_lnphi_ds but still no fix
        # unsure of correct order!
        # Reveals bugs in d_lnphi_dxs though.
        for i in range(self.N - 1):
            value = 0.0
            for j in range(self.N - 1):
                value += delta(i, j)
                term = zs[j]*Ks[j]/(1.0 + VF*(Ks[j] - 1.0))**2
                value += VF*d_lnphi_dxs[i][j] - (1.0 - VF)*d_lnphi_dys[i][j]
            
                J[i][j] = value
            
        # Last row except last value  - good, working
        bottom_row = J[-1]
        for j in range(self.N):
            value = 0.0
            for k in range(self.N):
                if k == j:
                    RR_l = -Ks[j]*zs[k]*VF/(1.0 + VF*(Ks[k] - 1.0))**2.0
                    RR_g = Ks[j]*(1.0 - VF)*zs[k]/(1.0 + VF*(Ks[k] - 1.0))**2.0
                    value += RR_g - RR_l
            bottom_row[j] = value



        # Last value - good, working, being overwritten
        dF_ncp1_dB = 0.0
        for i in range(len(zs)):
            Ki = Ks[i]
            dF_ncp1_dB += -zs[i]*(Ki - 1.0)**2/(1.0 + VF*(Ki - 1.0))**2
        J[-1][-1] = dF_ncp1_dB
            
            
        return J
            
    def _err_VL(self, lnKsVF, T, P, zs, near_critical=False):
        # tried autograd without luck
        lnKs = lnKsVF[:-1]
#        Ks = np.exp(lnKs)
        Ks = [exp(lnKi) for lnKi in lnKs]
        VF = float(lnKsVF[-1])
#        VF = lnKsVF[-1]
        
        xs = [zi/(1.0 + VF*(Ki - 1.0)) for zi, Ki in zip(zs, Ks)]
        ys = [Ki*xi for Ki, xi in zip(Ks, xs)]
        
        err_RR = Rachford_Rice_flash_error(VF, zs, Ks)

        eos_g = self.to_TP_zs(T=T, P=P, zs=ys)
        eos_l = self.to_TP_zs(T=T, P=P, zs=xs)
        if not near_critical:
            lnphis_g = eos_g.lnphis_g
            lnphis_l = eos_l.lnphis_l
        else:
            eos_g = self.to_TP_zs(T=T, P=P, zs=ys)
            eos_l = self.to_TP_zs(T=T, P=P, zs=xs)
            try:
                lnphis_g = eos_g.lnphis_g
            except AttributeError:
                lnphis_g = eos_g.lnphis_l
            try:
                lnphis_l = eos_l.lnphis_l
            except AttributeError:
                lnphis_l = eos_l.lnphis_g
                
        Fs = [lnKi - lnphi_l + lnphi_g for lnphi_l, lnphi_g, lnKi in zip(lnphis_l, lnphis_g, lnKs)]
        Fs.append(err_RR)
        return Fs
        
    def sequential_substitution_3P(self, Ks_y, Ks_z, beta_y, beta_z=0.0,
                                   
                                   maxiter=1000,
                                   xtol=1E-13, near_critical=True,
                                   xs=None, ys=None, zs=None,
                                   trivial_solution_tol=1e-5):
        
        
        from thermo.activity import Rachford_Rice_solution2
        print(Ks_y, Ks_z, beta_y, beta_z)
        beta_y, beta_z, xs_new, ys_new, zs_new = Rachford_Rice_solution2(zs=self.zs, Ks_y=Ks_y, Ks_z=Ks_z, beta_y=beta_y, beta_z=beta_z)
        print(beta_y, beta_z, xs_new, ys_new, zs_new)
        
        Ks_y = [exp(lnphi_x - lnphi_y) for lnphi_x, lnphi_y in zip(lnphis_x, lnphis_y)]
        Ks_z = [exp(lnphi_x - lnphi_z) for lnphi_x, lnphi_z in zip(lnphis_x, lnphis_z)]

    def sequential_substitution_VL(self, Ks_initial=None, maxiter=1000,
                                   xtol=1E-13, near_critical=True, Ks_extra=None,
                                   xs=None, ys=None, trivial_solution_tol=1e-5):
#        print(self.zs, Ks)
        if xs is not None and ys is not None:
            pass
        else:
            if Ks_initial is None:
                Ks = [Wilson_K_value(self.T, self.P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
            else:
                Ks = Ks_initial
            xs = None
            try:
                V_over_F, xs, ys = flash_inner_loop(self.zs, Ks)
            except ValueError as e:
                if Ks_extra is not None:
                    for Ks in Ks_extra:
                        try:
                            V_over_F, xs, ys = flash_inner_loop(self.zs, Ks)
                            break
                        except ValueError as e:
                            pass
            if xs is None:
                raise(e)
        
#        print(xs, ys, 'innerloop')
        Z_l_prev = None
        Z_g_prev = None

        for i in range(maxiter):
            if not near_critical:
                eos_g = self.to_TP_zs_fast(T=self.T, P=self.P, zs=ys, only_l=False, only_g=True)
                eos_l = self.to_TP_zs_fast(T=self.T, P=self.P, zs=xs, only_l=True, only_g=False)
                eos_g.fugacities()
                eos_l.fugacities()
#                eos_g = self.to_TP_zs(T=self.T, P=self.P, zs=ys)
#                eos_l = self.to_TP_zs(T=self.T, P=self.P, zs=xs)
    
                lnphis_g = eos_g.lnphis_g#fugacity_coefficients(eos_g.Z_g, ys)
                lnphis_l = eos_l.lnphis_l#fugacity_coefficients(eos_l.Z_l, xs)
                fugacities_l = eos_l.fugacities_l
                fugacities_g = eos_g.fugacities_g
            else:
                eos_g = self.to_TP_zs_fast(T=self.T, P=self.P, zs=ys, only_l=False, only_g=True)
                eos_l = self.to_TP_zs_fast(T=self.T, P=self.P, zs=xs, only_l=True, only_g=False)
                eos_g.fugacities()
                eos_l.fugacities()
#                eos_g = self.to_TP_zs(T=self.T, P=self.P, zs=ys)
#                eos_l = self.to_TP_zs(T=self.T, P=self.P, zs=xs)
                if 0:
                    if hasattr(eos_g, 'lnphis_g') and hasattr(eos_g, 'lnphis_l'):
                        if Z_l_prev is not None and Z_g_prev is not None:
                            if abs(eos_g.Z_g - Z_g_prev) < abs(eos_g.Z_l - Z_g_prev):
                                lnphis_g = eos_g.lnphis_g
                                fugacities_g = eos_g.fugacities_g
                                Z_g_prev = eos_g.Z_g
                            else:
                                lnphis_g = eos_g.lnphis_l
                                fugacities_g = eos_g.fugacities_l
                                Z_g_prev = eos_g.Z_l
                        else:
                            if eos_g.G_dep_g < eos_g.lnphis_l:
                                lnphis_g = eos_g.lnphis_g
                                fugacities_g = eos_g.fugacities_g
                                Z_g_prev = eos_g.Z_g
                            else:
                                lnphis_g = eos_g.lnphis_l
                                fugacities_g = eos_g.fugacities_l
                                Z_g_prev = eos_g.Z_l
                    else:
                        try:
                            lnphis_g = eos_g.lnphis_g#fugacity_coefficients(eos_g.Z_g, ys)
                            fugacities_g = eos_g.fugacities_g
                            Z_g_prev = eos_g.Z_g
                        except AttributeError:
                            lnphis_g = eos_g.lnphis_l#fugacity_coefficients(eos_g.Z_l, ys)
                            fugacities_g = eos_g.fugacities_l
                            Z_g_prev = eos_g.Z_l
                    if hasattr(eos_l, 'lnphis_g') and hasattr(eos_l, 'lnphis_l'):
                        if Z_l_prev is not None and Z_g_prev is not None:
                            if abs(eos_l.Z_l - Z_l_prev) < abs(eos_l.Z_g - Z_l_prev):
                                lnphis_l = eos_l.lnphis_g
                                fugacities_l = eos_l.fugacities_g
                                Z_l_prev = eos_l.Z_g
                            else:
                                lnphis_l = eos_l.lnphis_l
                                fugacities_l = eos_l.fugacities_l
                                Z_l_prev = eos_l.Z_l
                        else:
                            if eos_l.G_dep_g < eos_l.lnphis_l:
                                lnphis_l = eos_l.lnphis_g
                                fugacities_l = eos_l.fugacities_g
                                Z_l_prev = eos_l.Z_g
                            else:
                                lnphis_l = eos_l.lnphis_l
                                fugacities_l = eos_l.fugacities_l
                                Z_l_prev = eos_l.Z_l
                    else:
                        try:
                            lnphis_l = eos_l.lnphis_g#fugacity_coefficients(eos_l.Z_g, ys)
                            fugacities_l = eos_l.fugacities_g
                            Z_l_prev = eos_l.Z_g
                        except AttributeError:
                            lnphis_l = eos_l.lnphis_l#fugacity_coefficients(eos_l.Z_l, ys)
                            fugacities_l = eos_l.fugacities_l
                            Z_l_prev = eos_l.Z_l
                elif 0:
                    if hasattr(eos_g, 'lnphis_g') and hasattr(eos_g, 'lnphis_l'):
                        if eos_g.G_dep_g < eos_g.lnphis_l:
                            lnphis_g = eos_g.lnphis_g
                            fugacities_g = eos_g.fugacities_g
                        else:
                            lnphis_g = eos_g.lnphis_l
                            fugacities_g = eos_g.fugacities_l
                    else:
                        try:
                            lnphis_g = eos_g.lnphis_g#fugacity_coefficients(eos_g.Z_g, ys)
                            fugacities_g = eos_g.fugacities_g
                        except AttributeError:
                            lnphis_g = eos_g.lnphis_l#fugacity_coefficients(eos_g.Z_l, ys)
                            fugacities_g = eos_g.fugacities_l
                    
                    if hasattr(eos_l, 'lnphis_g') and hasattr(eos_l, 'lnphis_l'):
                        if eos_l.G_dep_g < eos_l.lnphis_l:
                            lnphis_l = eos_l.lnphis_g
                            fugacities_l = eos_l.fugacities_g
                        else:
                            lnphis_l = eos_l.lnphis_l
                            fugacities_l = eos_l.fugacities_l
                    else:
                        try:
                            lnphis_l = eos_l.lnphis_g#fugacity_coefficients(eos_l.Z_g, ys)
                            fugacities_l = eos_l.fugacities_g
                        except AttributeError:
                            lnphis_l = eos_l.lnphis_l#fugacity_coefficients(eos_l.Z_l, ys)
                            fugacities_l = eos_l.fugacities_l
                    
                else:
                    try:
                        lnphis_g = eos_g.lnphis_g#fugacity_coefficients(eos_g.Z_g, ys)
                        fugacities_g = eos_g.fugacities_g
                    except AttributeError:
                        lnphis_g = eos_g.lnphis_l#fugacity_coefficients(eos_g.Z_l, ys)
                        fugacities_g = eos_g.fugacities_l
                    try:
                        lnphis_l = eos_l.lnphis_l#fugacity_coefficients(eos_l.Z_l, xs)
                        fugacities_l = eos_l.fugacities_l
                    except AttributeError:
                        lnphis_l = eos_l.lnphis_g#fugacity_coefficients(eos_l.Z_g, xs)
                        fugacities_l = eos_l.fugacities_g

#            print(phis_l, phis_g, 'phis')
#             Ks = [exp(a - b) for a, b in zip(ln_phis_l, ln_phis_g)]
            Ks = [exp(l - g) for l, g in zip(lnphis_l, lnphis_g)] # K_value(phi_l=l, phi_g=g)
#            print(Ks)
            # Hack - no idea if this will work
#            maxK = max(Ks)
#            if maxK < 1:
#                Ks[Ks.index(maxK)] = 1.1
#            minK = min(Ks)
#            if minK >= 1:
#                Ks[Ks.index(minK)] = .9
                
            
#            print(Ks, 'Ks into RR')
            V_over_F, xs_new, ys_new = flash_inner_loop(self.zs, Ks)
#            if any(i < 0 for i in xs_new):
#                print('hil', xs_new)
#            
#            if any(i < 0 for i in ys_new):
#                print('hig', ys_new)
            
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
            
            # Claimed error function in CONVENTIONAL AND RAPID FLASH CALCULATIONS FOR THE SOAVE-REDLICH-KWONG AND PENG-ROBINSON EQUATIONS OF STATE
            
#            err3 = 0.0
#            for l, g, xi, yi in zip(lnphis_l, lnphis_g, xs_new, ys_new):
#                print(xi/yi, exp(l-g), l-g)
#                err_i = (expm1(l-g)*xi/yi) - 1.0 # Note: expm1 is slower
#                print(err_i, err_i*err_i, 'hi')
#                err3 += err_i*err_i
            
#            err2 = sum([(exp(l-g)-1.0)**2  ]) # Suggested tolerance 1e-15
            err2 = 0.0
            for l, g in zip(fugacities_l, fugacities_g):
                err_i = (l/g-1.0)
                err2 += err_i*err_i
           # Suggested tolerance 1e-15
            # This is a better metric because it does not involve  hysterisis
#            print(err3, err2)
            
            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
#            print(err, err2)
            xs, ys = xs_new, ys_new
#            print(i, 'err', err, err2, 'xs, ys', xs, ys, 'VF', V_over_F)
            if near_critical:
                comp_difference = sum([abs(xi - yi) for xi, yi in zip(xs, ys)])
                if comp_difference < trivial_solution_tol:
                    raise ValueError("Converged to trivial condition, compositions of both phases equal")
#            print(xs)
            if err2 < xtol:
                break
            if i == maxiter-1:
                raise ValueError('End of SS without convergence')
        return V_over_F, xs, ys, eos_l, eos_g

    def stabiliy_iteration_Michelsen(self, T, P, zs, Ks_initial=None, 
                                     maxiter=20, xtol=1E-12, liq=True):
        # checks stability vs. the current zs, mole fractions
        
        eos_ref = self.to_TP_zs(T=T, P=P, zs=zs)
        # If one phase is present - use that phase as the reference phase.
        # Otherwise, consider the phase with the lowest Gibbs excess energy as
        # the stable phase
        fugacities_ref, fugacities_ref_phase = eos_ref.eos_fugacities_lowest_Gibbs()
#        print(fugacities_ref, fugacities_ref_phase, 'fugacities_ref, fugacities_ref_phase')

        if Ks_initial is None:
            Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        else:
            Ks = Ks_initial
            
        same_phase_count = 0.0
        for _ in range(maxiter):
            if liq:
                zs_test = [zi/Ki for zi, Ki in zip(zs, Ks)]
            else:
                zs_test = [zi*Ki for zi, Ki in zip(zs, Ks)]
                
            sum_zs_test = sum(zs_test)
            zs_test_normalized = [zi/sum_zs_test for zi in zs_test]
            
            eos_test = self.to_TP_zs(T=T, P=P, zs=zs_test_normalized)
            fugacities_test, fugacities_phase = eos_test.eos_fugacities_lowest_Gibbs()
            
            if fugacities_ref_phase == fugacities_phase:
                same_phase_count += 1.0
            else:
                same_phase_count = 0
            
#            print(fugacities_ref_phase, fugacities_phase)
            
            if liq:
                corrections = [fi/f_ref*sum_zs_test for fi, f_ref in zip(fugacities_test, fugacities_ref)]
            else:
                corrections = [f_ref/(fi*sum_zs_test) for fi, f_ref in zip(fugacities_test, fugacities_ref)]
            Ks = [Ki*corr for Ki, corr in zip(Ks, corrections)]
            
            corrections_minus_1 = [corr - 1.0 for corr in corrections]
            err = sum([ci*ci for ci in corrections_minus_1])
#            print(err, xtol, Ks, corrections)
#            print('MM iter Ks =', Ks, 'zs', zs_test_normalized, 'MM err', err, xtol, _)
            if err < xtol:
                break
            elif same_phase_count > 5:
                break
            # It is possible to break if the trivial solution is being approached here also
            if _ == maxiter-1 and fugacities_ref_phase != fugacities_phase:
                raise UnconvergedError('End of stabiliy_iteration_Michelsen without convergence')
        # Fails directly if fugacities_ref_phase == fugacities_phase
        return sum_zs_test, Ks, fugacities_ref_phase == fugacities_phase
            
    def stability_Michelsen(self, T, P, zs, Ks_initial=None, maxiter=20,
                            xtol=1E-12, trivial_criteria=1E-4, 
                            stable_criteria=1E-7):
#        print('MM starting, Ks=', Ks_initial)
        if Ks_initial is None:
            Ks = [Wilson_K_value(T, P, Tci, Pci, omega)  for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        else:
            Ks = Ks_initial
        
        zs_sum_g, Ks_g, phase_failure_g = self.stabiliy_iteration_Michelsen(T=T, P=P, zs=zs, Ks_initial=Ks,
                                                                            maxiter=maxiter, xtol=xtol, liq=False)
        zs_sum_l, Ks_l, phase_failure_l = self.stabiliy_iteration_Michelsen(T=T, P=P, zs=zs, Ks_initial=Ks, 
                                                                            maxiter=maxiter, xtol=xtol, liq=True)
        
        log_Ks_g = [log(Ki) for Ki in Ks_g]
        log_Ks_l = [log(Ki) for Ki in Ks_l]        
        
        lnK_2_tot_g = sum(log_Ki*log_Ki for log_Ki in log_Ks_g)
        lnK_2_tot_l = sum(log_Ki*log_Ki for log_Ki in log_Ks_l)        
        
        sum_g_criteria = zs_sum_g - 1.0
        sum_l_criteria = zs_sum_l - 1.0
        
        trivial_g, trivial_l = False, False
        if lnK_2_tot_g < trivial_criteria:
            trivial_g = True
        if lnK_2_tot_l < trivial_criteria:
            trivial_l = True
            
        stable = False
                    
        # Table 4.6 Summary of Possible Phase Stability Test Results, 
        # Phase Behavior, Whitson and Brule
        # There is a typo where Sl appears in the vapor column; this should be
        # liquid; as shown in https://www.e-education.psu.edu/png520/m17_p7.html
        
#        print('phase_failure_g', phase_failure_g, 'phase_failure_l', phase_failure_l,
#              'sum_g_criteria', sum_g_criteria, 'sum_l_criteria', sum_l_criteria)
        if phase_failure_g and phase_failure_l:
            stable = True
        elif trivial_g and trivial_l:
            stable = True
        elif sum_g_criteria < stable_criteria and trivial_l:
            stable = True
        elif trivial_g and sum_l_criteria < stable_criteria:
            stable = True
        elif sum_g_criteria < stable_criteria and sum_l_criteria < stable_criteria:
            stable = True
        # These last two are custom, and it is apparent since they are bad
        # Also did not document well enough the cases they fail in
        # Disabled 2018-12-29
#        elif trivial_l and sum_l_criteria < stable_criteria:
#            stable = True
#        elif trivial_g and sum_g_criteria < stable_criteria:
#            stable = True
#        else:
#            print('lnK_2_tot_g', lnK_2_tot_g , 'lnK_2_tot_l', lnK_2_tot_l,
#                  'sum_g_criteria', sum_g_criteria, 'sum_l_criteria', sum_l_criteria)

            
            
        # No need to enumerate unstable results
        
        if not stable:
            Ks = [K_g*K_l for K_g, K_l in zip(Ks_g, Ks_l)]
#        print('MM ended', Ks, stable, Ks_g, Ks_l)
        return stable, Ks, [Ks_g, Ks_l]
        

    def _V_over_F_bubble_T_inner(self, T, P, zs, maxiter=20, xtol=1E-3):
        eos_l = self.to_TP_zs(T=T, P=P, zs=zs)
        
        if not hasattr(eos_l, 'V_l'):
            raise ValueError('At the specified temperature, there is no liquid root')
    
        Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        V_over_F, xs, ys = flash_inner_loop(zs, Ks)
        for i in range(maxiter):
            eos_g = self.to_TP_zs(T=T, P=P, zs=ys)
    
            if not hasattr(eos_g, 'V_g'):
                phis_g = eos_g.phis_l
                fugacities_g = eos_g.fugacities_l
            else:
                phis_g = eos_g.phis_g
                fugacities_g = eos_g.fugacities_g
            
            Ks = [K_value(phi_l=l, phi_g=g) for l, g in zip(eos_l.phis_l, phis_g)]
            V_over_F, xs, ys = flash_inner_loop(zs, Ks)
            err = sum([abs(i-j) for i, j in zip(eos_l.fugacities_l, fugacities_g)])
            if err < xtol:
                break
        if not hasattr(eos_g, 'V_g'):
            raise ValueError('At the specified temperature, the solver did not converge to a vapor root')
        return V_over_F
#        raise Exception('Could not converge to desired tolerance')

    def _V_over_F_dew_T_inner(self, T, P, zs, maxiter=20, xtol=1E-10):
        eos_g = self.to_TP_zs(T=T, P=P, zs=zs)
        if not hasattr(eos_g, 'V_g'):
            raise ValueError('At the specified temperature, there is no vapor root')
        
        Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        V_over_F, xs, ys = flash_inner_loop(zs, Ks)
        for i in range(maxiter):
            eos_l = self.to_TP_zs(T=T, P=P, zs=xs)
    
            if not hasattr(eos_l, 'V_l'):
                phis_l = eos_l.phis_g
                fugacities_l = eos_l.fugacities_g
            else:
                phis_l = eos_l.phis_l
                fugacities_l = eos_l.fugacities_l
    
    
            Ks = [K_value(phi_l=l, phi_g=g) for l, g in zip(phis_l, eos_g.phis_g)]
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks)
            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
            xs, ys = xs_new, ys_new
            if xtol < 1E-10:
                break
        if not hasattr(eos_l, 'V_l'):
            raise ValueError('At the specified temperature, the solver did not converge to a liquid root')
        return V_over_F-1.0
#        return abs(V_over_F-1)

    def _V_over_F_dew_T_inner_accelerated(self, T, P, zs, maxiter=20, xtol=1E-10):
        '''This is not working.
        '''
        eos_g = self.to_TP_zs(T=T, P=P, zs=zs)
        if not hasattr(eos_g, 'V_g'):
            raise ValueError('At the specified temperature, there is no vapor root')
        
        Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        V_over_F_new, xs, ys = flash_inner_loop(zs, Ks)
        for i in range(maxiter):
            eos_l = self.to_TP_zs(T=T, P=P, zs=xs)
    
            if not hasattr(eos_l, 'V_l'):
                phis_l = eos_l.phis_g
                fugacities_l = eos_l.fugacities_g
            else:
                phis_l = eos_l.phis_l
                fugacities_l = eos_l.fugacities_l
            
            if 0.0 < V_over_F_new < 1.0 and i > 2:
                Rs = [K_value(phi_l=l, phi_g=g) for l, g in zip(phis_l, eos_g.phis_g)]
                lambdas = [(Ki - 1.0)/(Ki - Rri) for Rri, Ki in zip(Rs, Ks)]
                Ks = [Ki*Ri**lambda_i for Ki, Ri, lambda_i in zip(Ks, Rs, lambdas)]
            else:
                Ks = [K_value(phi_l=l, phi_g=g) for l, g in zip(phis_l, eos_g.phis_g)]
            
            V_over_F_new, xs_new, ys_new = flash_inner_loop(zs, Ks)
            err_new = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
            xs, ys = xs_new, ys_new
            V_over_F_old = V_over_F_new
            if i == 0:
                err_old = err_new
            
            err_old = err_new
            if err_new < xtol:
                break
        if not hasattr(eos_l, 'V_l'):
            raise ValueError('At the specified temperature, the solver did not converge to a liquid root')
        return V_over_F_new-1.0
#        return abs(V_over_F-1)




class PRMIX(GCEOSMIX, PR):
    r'''Class for solving the Peng-Robinson cubic equation of state for a 
    mixture of any number of compounds. Subclasses `PR`. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.

    The implemented method here is `fugacity_coefficients`, which implements
    the formula for fugacity coefficients in a mixture as given in [1]_.
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}
        
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}
        
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}
        
        b = \sum_i z_i b_i

        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}
        
	  b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}

        \alpha(T)_i=[1+\kappa_i(1-\sqrt{T_{r,i}})]^2
        
        \kappa_i=0.37464+1.54226\omega_i-0.26992\omega^2_i
        
    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:
    
    >>> eos = PRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (3.625735065042031e-05, 0.0007006656856469095)
    >>> eos.fugacities_l, eos.fugacities_g
    ([793860.8382114634, 73468.55225303846], [436530.9247009119, 358114.63827532396])
    
    Notes
    -----
    For P-V initializations, SciPy's `newton` solver is used to find T.

    References
    ----------
    .. [1] Peng, Ding-Yu, and Donald B. Robinson. "A New Two-Constant Equation 
       of State." Industrial & Engineering Chemistry Fundamentals 15, no. 1 
       (February 1, 1976): 59-64. doi:10.1021/i160057a011.
    .. [2] Robinson, Donald B., Ding-Yu Peng, and Samuel Y-K Chung. "The 
       Development of the Peng - Robinson Equation and Its Application to Phase
       Equilibrium in a System Containing Methanol." Fluid Phase Equilibria 24,
       no. 1 (January 1, 1985): 25-41. doi:10.1016/0378-3812(85)87035-7. 
    '''
    a_alpha_mro = -4
    eos_pure = PR
    
    nonstate_constants_specific = ('kappas', )
    
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None):
        self.N = N = len(Tcs)
        self.cmps = range(self.N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        if kijs is None:
            kijs = [[0.0]*N for i in self.cmps]
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        # optimization, unfortunately
        c1R2, c2R = self.c1*R2, self.c2*R
        # Also tried to store the inverse of Pcs, without success - slows it down
        self.ais = [c1R2*Tcs[i]*Tcs[i]/Pcs[i] for i in self.cmps]
        self.bs = [c2R*Tcs[i]/Pcs[i] for i in self.cmps]
        self.b = b = sum([bi*zi for bi, zi in zip(self.bs, zs)])
        self.kappas = [omega*(-0.26992*omega + 1.54226) + 0.37464 for omega in omegas]
        
        self.delta = 2.0*b
        self.epsilon = -b*b

        self.solve()
        self.fugacities()

    def fast_init_specific(self):
        self.b = b = sum([bi*zi for bi, zi in zip(self.bs, self.zs)])
        self.delta = 2.0*b
        self.epsilon = -b*b


    def a_alpha_and_derivatives_vectorized(self, T, full=False, quick=True):
        if not full:
            a_alphas = []
            for a, kappa, Tc in zip(self.ais, self.kappas, self.Tcs):
                x1 = Tc**-0.5
                x2 = 1.0 + kappa*(1.0 - T*x1)
                a_alphas.append(a*x2*x2)
            
            return a_alphas
        else:
            T_inv = 1.0/T
            x0 = T**0.5
            x0_inv = 1.0/x0
            x0T_inv = x0_inv*T_inv
            a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [], [], []
            
            for a, kappa, Tc in zip(self.ais, self.kappas, self.Tcs):
                x1 = Tc**-0.5
                x2 = kappa*(x0*x1 - 1.) - 1.
                x3 = a*kappa
                x4 = x1*x2
                a_alphas.append(a*x2*x2)
                da_alpha_dTs.append(x4*x3*x0_inv)
                d2a_alpha_dT2s.append(0.5*x3*(T_inv*x1*x1*kappa - x4*x0T_inv))

            return a_alphas, da_alpha_dTs, d2a_alpha_dT2s






    def setup_a_alpha_and_derivatives(self, i, T=None):
        r'''Sets `a`, `kappa`, and `Tc` for a specific component before the 
        pure-species EOS's `a_alpha_and_derivatives` method is called. Both are 
        called by `GCEOSMIX.a_alpha_and_derivatives` for every component.'''
        self.a, self.kappa, self.Tc = self.ais[i], self.kappas[i], self.Tcs[i]

    def cleanup_a_alpha_and_derivatives(self):
        r'''Removes properties set by `setup_a_alpha_and_derivatives`; run by
        `GCEOSMIX.a_alpha_and_derivatives` after `a_alpha` is calculated for 
        every component'''
        del(self.a, self.kappa, self.Tc)
        
    def fugacity_coefficients(self, Z, zs):
        r'''Literature formula for calculating fugacity coefficients for each
        species in a mixture. Verified numerically. Applicable to most 
        derivatives of the Peng-Robinson equation of state as well.
        Called by `fugacities` on initialization, or by a solver routine 
        which is performing a flash calculation.
        
        .. math::
            \ln \hat \phi_i = \frac{B_i}{B}(Z-1)-\ln(Z-B) + \frac{A}{2\sqrt{2}B}
            \left[\frac{B_i}{B} - \frac{2}{a\alpha}\sum_i y_i(a\alpha)_{ij}\right]
            \log\left[\frac{Z + (1+\sqrt{2})B}{Z-(\sqrt{2}-1)B}\right]
            
            A = \frac{(a\alpha)P}{R^2 T^2}
            
            B = \frac{b P}{RT}
        
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        log_phis : float
            Log fugacity coefficient for each species, [-]
                         
        References
        ----------
        .. [1] Peng, Ding-Yu, and Donald B. Robinson. "A New Two-Constant  
           Equation of State." Industrial & Engineering Chemistry Fundamentals 
           15, no. 1 (February 1, 1976): 59-64. doi:10.1021/i160057a011.
        .. [2] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        '''
        bs, b = self.bs, self.b
        A = self.a_alpha*self.P/(R2*self.T*self.T)
        B = b*self.P/(R*self.T)

        # The two log terms need to use a complex log; typically these are
        # calculated at "liquid" volume solutions which are unstable
        # and cannot exist
        x0 = clog(Z - B).real
        Zm1 = Z - 1.0
        
        x1 = 2./self.a_alpha
        x2 = A/(two_root_two*B)
        x3 = clog((Z + (root_two + 1.)*B)/(Z - (root_two - 1.)*B)).real
        
        cmps = self.cmps
        phis = []
        fugacity_sum_terms = []
        
        for i in cmps:
            a_alpha_js = self.a_alpha_ijs[i]
            b_ratio = bs[i]/b
            t1 = b_ratio*Zm1 - x0
            
            sum_term = sum([zs[j]*a_alpha_js[j] for j in cmps])

            t3 = t1 - x2*(x1*sum_term - b_ratio)*x3
            # Temp
            if t3 > 700.0:
                t3 = 700
            phis.append(t3)
            fugacity_sum_terms.append(sum_term)
        self.fugacity_sum_terms = fugacity_sum_terms
        return phis


    def d_lnphis_dT(self, Z, dZ_dT, zs):
        a_alpha_ijs, da_alpha_dT_ijs = self.a_alpha_ijs, self.da_alpha_dT_ijs
        cmps = self.cmps
        bs, b = self.bs, self.b
        
        T_inv = 1.0/self.T

        A = self.a_alpha*self.P*R2_inv*T_inv*T_inv
        
        B = b*self.P*R_inv*T_inv
        
        x2 = T_inv*T_inv
        x3 = R_inv
        
        x4 = self.P*b*x3
        x5 = x2*x4
        x8 = x4*T_inv
        
        x10 = self.a_alpha
        x11 = 1.0/self.a_alpha
        x12 = self.da_alpha_dT
        
        x13 = root_two
        x14 = 1.0/b
        
        x15 = x13 + 1.0 # root_two plus 1
        x16 = Z + x15*x8
        x17 = x13 - 1.0 # root two minus one
        x18 = x16/(x17*x8 - Z)
        x19 = log(-x18)
        
        x24 = 0.25*x10*x13*x14*x19*x2*x3
        x25 = 0.25*x12*x13*x14*x19*x3*T_inv
        x26 = 0.25*x10*x13*x14*x3*T_inv*(-dZ_dT + x15*x5 - x18*(dZ_dT + x17*x5))/(x16)
        x50 = -0.5*x13*x14*x19*x3*T_inv
        x51 = -x11*x12
        x52 = (dZ_dT + x5)/(x8 - Z)
        x53 = 2.0*x11
        
        # Composition stuff
        
        try:
            fugacity_sum_terms = self.fugacity_sum_terms
        except AttributeError:
            fugacity_sum_terms = [sum([zs[j]*a_alpha_ijs[i][j] for j in cmps]) for i in cmps]
        
        d_lnphis_dTs = []
        for i in cmps:
            x9 = fugacity_sum_terms[i]
#            x9 = sum([zs[j]*a_alpha_ijs[i][j] for j in cmps])
            der_sum = sum([zs[j]*da_alpha_dT_ijs[i][j] for j in cmps])
            
            x20 = x50*(x51*x9 + der_sum) + x52
            x21 = self.bs[i]*x14
            x23 = x53*x9 - x21
            
            d_lhphi_dT = dZ_dT*x21 + x20 + x23*(x24 - x25 + x26)
            d_lnphis_dTs.append(d_lhphi_dT)
        return d_lnphis_dTs
         
    def d_lnphis_dP(self, Z, dZ_dP, zs):
        a_alpha = self.a_alpha
        cmps = self.cmps
        bs, b = self.bs, self.b
        T_inv = 1.0/self.T

        x2 = 1.0/b
        x6 = b*R_inv*T_inv
        x8 = self.P*x6
        x9 = (dZ_dP - x6)/(x8 - Z)
        x13 = Z + root_two_p1*x8
        x15 = (a_alpha*root_two*x2*R_inv*T_inv*(dZ_dP + root_two_p1*x6 
                + x13*(dZ_dP - root_two_m1*x6)/(root_two_m1*x8 - Z))/(4.0*x13))

        try:
            fugacity_sum_terms = self.fugacity_sum_terms
        except AttributeError:
            a_alpha_ijs = self.a_alpha_ijs
            fugacity_sum_terms = [sum([zs[j]*a_alpha_ijs[i][j] for j in cmps]) for i in cmps]

        x50 = -2.0/a_alpha
        d_lnphi_dPs = []
        for i in cmps:
            x3 = bs[i]*x2
            x10 = x50*fugacity_sum_terms[i]
            d_lnphi_dP = dZ_dP*x3 + x15*(x10 + x3) + x9
            d_lnphi_dPs.append(d_lnphi_dP)
        return d_lnphi_dPs                            

    def d_lnphi_dzs(self, Z, zs):
        
        # TODO try to follow "B.5.2.1 Derivatives of Fugacity Coefficient with Respect to Mole Fraction"
        # "Development of an Equation-of-State Thermal Flooding Simulator"
        cmps_m1 = range(self.N-1)
        
        a_alpha = self.a_alpha
        a_alpha_ijs = self.a_alpha_ijs
        T2 = self.T*self.T
        b = self.b

        A = a_alpha*self.P/(R2*T2)
        B = b*self.P/(R*self.T)
        B2 = B*B
        Z2 = Z*Z
        A_B = A/B
        ZmB = Z - B
        
        
        dZ_dA = (B - Z)/(3.0*Z2 - 2.0*(1.0 - B)*Z + (A - 2.0*B - 3.0*B2))
        
        # 2*(3.0*B + 1)*Z may or may not have Z
        # Simple phase stability-testing algorithm in the reduction method.
        dZ_dB = ((-Z2 + 2*(3.0*B + 1)*Z) + (A - 2.0*B - 3.0*B2))/(
                3.0*Z2 - 2.0*(1.0 - B)*Z + (A - 2.0*B - 3.0*B2))


        Sis = []
        for i in range(len(zs)):
            tot = 0.0
            for j in range(len(zs)):
                tot += zs[j]*a_alpha_ijs[i][j]
            Sis.append(tot)
            
        Sais = [val/a_alpha for val in Sis]
        Sbis = [bi/b for bi in self.bs]
        
        Snc = Sis[-1]
        const_A = 2.0*self.P/(R2*T2)
        dA_dzis = [const_A*(Si - Snc) for Si in Sis[:-1]]
        
        const_B = 2.0*self.P/(R*self.T)
        bnc = self.bs[-1]
        dB_dzis = [const_B*(self.bs[i] - bnc) for i in self.cmps] # Probably wrong, missing
        
        dZ_dzs = [dZ_dA*dA_dz_i + dZ_dB*dB_dzi for dA_dz_i, dB_dzi in zip(dA_dzis, dB_dzis)]
        
        t1 = (Z2 + 2.0*Z*B - B2)
        t2 = clog((Z + (root_two + 1.)*B)/(Z - (root_two - 1.)*B)).real
        t3 = t2*-A/(B*two_root_two)
        t4 = -t2/(two_root_two*B)
            
        a_nc = a_alpha_ijs[-1][-1] # no idea if this is right

        # Have some converns of what Snc really is
        dlnphis_dzs_all = []
        for i in range(self.N):
            Diks = [-A_B*(2.0*Sais[i] - Sbis[i])*(Z*dB_dzis[k] - B*dZ_dzs[k])/t1
                    for k in cmps_m1]
            
            Ciks = [t3*(2.0*(a_alpha_ijs[i][k] - a_nc)/a_alpha
                        - 4.0*Sais[i]*(Sais[k] - Snc) 
                        + Sbis[i]*(Sbis[k] - Snc))   
                    for k in cmps_m1]
            
            
            x5 = t4*(2.0*Sais[i] - Sbis[i])
            Biks = [x5*(dA_dzis[k] - A_B*dB_dzis[k])
                    for k in cmps_m1 ]
            
            Aiks = [Sbis[i]*(dZ_dzs[k] - (Sbis[k] - Snc)*(Z - 1.0))
                    - (dZ_dzs[k] - dB_dzis[k])/ZmB 
                    for k in cmps_m1 ]
            
            dlnphis_dzs = [Aik + Bik + Cik + Dik for Aik, Bik, Cik, Dik in zip(Aiks, Biks, Ciks, Diks)]
            dlnphis_dzs_all.append(dlnphis_dzs)
        return dlnphis_dzs_all
        


class SRKMIX(GCEOSMIX, SRK):    
    r'''Class for solving the Soave-Redlich-Kwong cubic equation of state for a 
    mixture of any number of compounds. Subclasses `SRK`. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.

    The implemented method here is `fugacity_coefficients`, which implements
    the formula for fugacity coefficients in a mixture as given in [1]_.
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}
        
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}
        
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

        b = \sum_i z_i b_i

        a_i =\left(\frac{R^2(T_{c,i})^{2}}{9(\sqrt[3]{2}-1)P_{c,i}} \right)
        =\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}
    
        b_i =\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_{c,i}}{P_{c,i}}
        =\frac{0.08664\cdot R T_{c,i}}{P_{c,i}}
        
        \alpha(T)_i = \left[1 + m_i\left(1 - \sqrt{\frac{T}{T_{c,i}}}\right)\right]^2
        
        m_i = 0.480 + 1.574\omega_i - 0.176\omega_i^2
            
    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:
    
    >>> SRK_mix = SRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> SRK_mix.V_l, SRK_mix.V_g
    (4.104755570185169e-05, 0.0007110155639819185)
    >>> SRK_mix.fugacities_l, SRK_mix.fugacities_g
    ([817841.6430546861, 72382.81925202614], [442137.12801246037, 361820.79211909405])
    
    Notes
    -----
    For P-V initializations, SciPy's `newton` solver is used to find T.

    References
    ----------
    .. [1] Soave, Giorgio. "Equilibrium Constants from a Modified Redlich-Kwong
       Equation of State." Chemical Engineering Science 27, no. 6 (June 1972): 
       1197-1203. doi:10.1016/0009-2509(72)80096-4.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th 
       edition. New York: McGraw-Hill Professional, 2000.
    .. [3] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
       Butterworth-Heinemann, 1985.
    '''
    a_alpha_mro = -4
    eos_pure = SRK
    nonstate_constants_specific = ('ms',)
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None):
        self.N = len(Tcs)
        self.cmps = range(self.N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        if kijs is None:
            kijs = [[0]*self.N for i in range(self.N)]
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        self.ais = [self.c1*R2*Tc*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.bs = [self.c2*R*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        self.ms = [0.480 + 1.574*omega - 0.176*omega*omega for omega in omegas]
        self.delta = self.b

        self.solve()
        self.fugacities()

    def fast_init_specific(self):
        self.b = b = sum([bi*zi for bi, zi in zip(self.bs, self.zs)])
        self.delta = self.b

    def setup_a_alpha_and_derivatives(self, i, T=None):
        r'''Sets `a`, `m`, and `Tc` for a specific component before the 
        pure-species EOS's `a_alpha_and_derivatives` method is called. Both are 
        called by `GCEOSMIX.a_alpha_and_derivatives` for every component.'''
        self.a, self.m, self.Tc = self.ais[i], self.ms[i], self.Tcs[i]

    def cleanup_a_alpha_and_derivatives(self):
        r'''Removes properties set by `setup_a_alpha_and_derivatives`; run by
        `GCEOSMIX.a_alpha_and_derivatives` after `a_alpha` is calculated for 
        every component'''
        del(self.a, self.m, self.Tc)
        
    def fugacity_coefficients(self, Z, zs):
        r'''Literature formula for calculating fugacity coefficients for each
        species in a mixture. Verified numerically. Applicable to most 
        derivatives of the SRK equation of state as well.
        Called by `fugacities` on initialization, or by a solver routine 
        which is performing a flash calculation.
        
        .. math::
            \ln \hat \phi_i = \frac{B_i}{B}(Z-1) - \ln(Z-B) + \frac{A}{B}
            \left[\frac{B_i}{B} - \frac{2}{a \alpha}\sum_i y_i(a\alpha)_{ij}
            \right]\ln\left(1+\frac{B}{Z}\right)
            
            A=\frac{a\alpha P}{R^2T^2}
            
            B = \frac{bP}{RT}
        
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        log_phis : float
            Log fugacity coefficient for each species, [-]
        
        References
        ----------
        .. [1] Soave, Giorgio. "Equilibrium Constants from a Modified 
           Redlich-Kwong Equation of State." Chemical Engineering Science 27,
           no. 6 (June 1972): 1197-1203. doi:10.1016/0009-2509(72)80096-4.
        .. [2] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        '''
        
        RT = self.T*R
        P_RT = self.P/RT
        A = self.a_alpha*self.P/(RT*RT)
        B = self.b*self.P/RT
        A_B = A/B
        t0 = log(Z - B)
        t3 = log(1. + B/Z)
        Z_minus_one_over_B = (Z - 1.0)/B
        two_over_a_alpha = 2./self.a_alpha
        phis = []
        fugacity_sum_terms = []
        for i in self.cmps:
            Bi = self.bs[i]*P_RT
            t1 = Bi*Z_minus_one_over_B - t0
            l = self.a_alpha_ijs[i]
            sum_term = sum([zs[j]*l[j] for j in self.cmps])
            t2 = A_B*(Bi/B - two_over_a_alpha*sum_term)
            phis.append(t1 + t2*t3)
            fugacity_sum_terms.append(sum_term)
        self.fugacity_sum_terms = fugacity_sum_terms
        return phis
        

    def d_lnphis_dT(self, Z, dZ_dT, zs):
        a_alpha_ijs, da_alpha_dT_ijs = self.a_alpha_ijs, self.da_alpha_dT_ijs
        cmps = self.cmps
        P, bs, b = self.P, self.bs, self.b
        
        T_inv = 1.0/self.T
        A = self.a_alpha*P*R2_inv*T_inv*T_inv
        B = b*P*R_inv*T_inv

        x2 = T_inv*T_inv
        x4 = P*b*R_inv
        x6 = x4*T_inv
        
        x8 = self.a_alpha
        x9 = 1.0/x8
        x10 = self.da_alpha_dT
        x11 = 1.0/b
        x12 = 1.0/Z
        x13 = x12*x6 + 1.0
        x14 = log(x13)
        x19 = x11*x14*x2*R_inv*x8
        x20 = x10*x11*x14*R_inv*T_inv
        x21 = P*x12*x2*x8*(dZ_dT*x12 + T_inv)/(R2*x13)
        
        x50 = -x11*x14*R_inv*T_inv
        x51 = -2.0*x10
        x52 = (dZ_dT + x2*x4)/(x6 - Z)

        # Composition stuff
        d_lnphis_dTs = []
        
        try:
            fugacity_sum_terms = self.fugacity_sum_terms
        except AttributeError:
            fugacity_sum_terms = [sum([zs[j]*a_alpha_ijs[i][j] for j in cmps]) for i in cmps]

        for i in cmps:
            x7 = fugacity_sum_terms[i]
#            x7 = sum([zs[j]*a_alpha_ijs[i][j] for j in cmps])
            der_sum = sum([zs[j]*da_alpha_dT_ijs[i][j] for j in cmps])
    
            x15 = (x50*(x51*x7*x9 + 2.0*der_sum) + x52)

            x16 = bs[i]*x11
            x18 = -x16 + 2.0*x7*x9
        
            d_lhphi_dT = dZ_dT*x16 + x15 + x18*(x19 - x20 + x21)
            d_lnphis_dTs.append(d_lhphi_dT)
        return d_lnphis_dTs

    def d_lnphis_dP(self, Z, dZ_dP, zs):
        a_alpha = self.a_alpha
        cmps = self.cmps
        bs, b = self.bs, self.b
        T_inv = 1.0/self.T

        RT_inv = T_inv*R_inv
        x0 = Z
        x1 = dZ_dP
        x2 = 1.0/b
        x4 = b*RT_inv
        x5 = self.P*x4
        x6 = (dZ_dP - x4)/(x5 - Z)
        x7 = a_alpha
        x9 = 1./Z
        x10 = a_alpha*x9*(self.P*dZ_dP*x9 - 1)*RT_inv*RT_inv/((x5*x9 + 1.0))
        try:
            fugacity_sum_terms = self.fugacity_sum_terms
        except AttributeError:
            a_alpha_ijs = self.a_alpha_ijs
            fugacity_sum_terms = [sum([zs[j]*a_alpha_ijs[i][j] for j in cmps]) for i in cmps]

        x50 = 2.0/a_alpha
        d_lnphi_dPs = []
        for i in cmps:
            x8 = x50*fugacity_sum_terms[i]
            x3 = bs[i]*x2
            d_lnphi_dP = dZ_dP*x3 + x10*(x8 - x3) + x6
            d_lnphi_dPs.append(d_lnphi_dP)
        return d_lnphi_dPs                            



class PR78MIX(PRMIX):
    r'''Class for solving the Peng-Robinson cubic equation of state for a 
    mixture of any number of compounds according to the 1978 variant. 
    Subclasses `PR`. Solves the EOS on initialization and calculates fugacities  
    for all components in all phases.

    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}
        
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}
        
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}
        
        b = \sum_i z_i b_i

        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}
        
	  b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}

        \alpha(T)_i=[1+\kappa_i(1-\sqrt{T_{r,i}})]^2
        
        \kappa_i = 0.37464+1.54226\omega_i-0.26992\omega_i^2 \text{ if } \omega_i
        \le 0.491
        
        \kappa_i = 0.379642 + 1.48503 \omega_i - 0.164423\omega_i^2 + 0.016666
        \omega_i^3 \text{ if } \omega_i > 0.491
        
    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa, with modified
    acentric factors to show the difference between `PRMIX`
    
    >>> eos = PR78MIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.6, 0.7], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (3.239642793468725e-05, 0.0005043378493002219)
    >>> eos.fugacities_l, eos.fugacities_g
    ([833048.4511980312, 6160.908815331656], [460717.2776793945, 279598.90103207604])
    
    Notes
    -----
    This variant is recommended over the original.

    References
    ----------
    .. [1] Peng, Ding-Yu, and Donald B. Robinson. "A New Two-Constant Equation 
       of State." Industrial & Engineering Chemistry Fundamentals 15, no. 1 
       (February 1, 1976): 59-64. doi:10.1021/i160057a011.
    .. [2] Robinson, Donald B., Ding-Yu Peng, and Samuel Y-K Chung. "The 
       Development of the Peng - Robinson Equation and Its Application to Phase
       Equilibrium in a System Containing Methanol." Fluid Phase Equilibria 24,
       no. 1 (January 1, 1985): 25-41. doi:10.1016/0378-3812(85)87035-7. 
    '''
    a_alpha_mro = -4
    eos_pure = PR78
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None):
        self.N = len(Tcs)
        self.cmps = range(self.N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        if kijs is None:
            kijs = [[0]*self.N for i in range(self.N)]
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        self.ais = [self.c1*R*R*Tc*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.bs = [self.c2*R*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        self.kappas = []
        for omega in omegas:
            if omega <= 0.491:
                self.kappas.append(0.37464 + 1.54226*omega - 0.26992*omega*omega)
            else:
                self.kappas.append(0.379642 + 1.48503*omega - 0.164423*omega**2 + 0.016666*omega**3)
        
        self.delta = 2.*self.b
        self.epsilon = -self.b*self.b

        self.solve()
        self.fugacities()



class VDWMIX(GCEOSMIX, VDW):
    r'''Class for solving the Van der Waals cubic equation of state for a 
    mixture of any number of compounds. Subclasses `VDW`. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.

    The implemented method here is `fugacity_coefficients`, which implements
    the formula for fugacity coefficients in a mixture as given in [1]_.
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P=\frac{RT}{V-b}-\frac{a}{V^2}
        
        a = \sum_i \sum_j z_i z_j {a}_{ij}
            
        b = \sum_i z_i b_i

        a_{ij} = (1-k_{ij})\sqrt{a_{i}a_{j}}

        a_i=\frac{27}{64}\frac{(RT_{c,i})^2}{P_{c,i}}

        b_i=\frac{RT_{c,i}}{8P_{c,i}}
            
    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    omegas : float, optional
        Acentric factors of all compounds - Not used in equation of state!, [-]
        
    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:
    
    >>> eos = VDWMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (5.881367851416652e-05, 0.0007770869741895236)
    >>> eos.fugacities_l, eos.fugacities_g
    ([854533.2669205057, 207126.84972762014], [448470.7363380735, 397826.543999929])

    Notes
    -----
    For P-V initializations, SciPy's `newton` solver is used to find T.

    References
    ----------
    .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
       Butterworth-Heinemann, 1985.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th 
       edition. New York: McGraw-Hill Professional, 2000.
    '''
    a_alpha_mro = -4
    eos_pure = VDW
    
    nonstate_constants_specific = tuple()
    
    def __init__(self, Tcs, Pcs, zs, kijs=None, T=None, P=None, V=None, 
                 omegas=None):
        self.N = len(Tcs)
        self.cmps = range(self.N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.zs = zs
        if kijs is None:
            kijs = [[0]*self.N for i in range(self.N)]
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        self.ais = [27.0/64.0*(R*Tc)**2/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.bs = [R*Tc/(8.*Pc) for Tc, Pc in zip(Tcs, Pcs)]
        self.b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        
        self.omegas = omegas
        self.solve()
        self.fugacities()

    def fast_init_specific(self):
        self.b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        
    def setup_a_alpha_and_derivatives(self, i, T=None):
        r'''Sets `a` for a specific component before the 
        pure-species EOS's `a_alpha_and_derivatives` method is called. Both are 
        called by `GCEOSMIX.a_alpha_and_derivatives` for every component.'''
        self.a = self.ais[i]
        
    def cleanup_a_alpha_and_derivatives(self):
        r'''Removes properties set by `setup_a_alpha_and_derivatives`; run by
        `GCEOSMIX.a_alpha_and_derivatives` after `a_alpha` is calculated for 
        every component'''
        del(self.a)
        
    def a_alpha_and_derivatives_vectorized(self, T, full=False, quick=True):
        if not full:
            return self.ais
        else:
            zeros = [0.0]*self.N
            return self.ais, zeros, zeros
        
    def fugacity_coefficients(self, Z, zs):
        r'''Literature formula for calculating fugacity coefficients for each
        species in a mixture. Verified numerically.
        Called by `fugacities` on initialization, or by a solver routine 
        which is performing a flash calculation.
        
        .. math::
            \ln \hat \phi_i = \frac{b_i}{V-b} - \ln\left[Z\left(1
            - \frac{b}{V}\right)\right] - \frac{2\sqrt{aa_i}}{RTV}
        
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        log_phis : float
            Log fugacity coefficient for each species, [-]
                         
        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        '''
        phis = []
        V = Z*R*self.T/self.P
        
        t1 = log(Z*(1. - self.b/V))
        t2 = 2.0/(R*self.T*V)
        t3 = 1.0/(V - self.b)
        a_alpha = self.a_alpha
        for ai, bi in zip(self.ais, self.bs):
            phi = (bi*t3 - t1 - t2*(a_alpha*ai)**0.5)
            phis.append(phi)
        return phis

    def d_lnphis_dT(self, Z, dZ_dT, zs):
        a_alpha_ijs, da_alpha_dT_ijs = self.a_alpha_ijs, self.da_alpha_dT_ijs
        cmps = self.cmps
        T, P, ais, bs, b = self.T, self.P, self.ais, self.bs, self.b
        
        T_inv = 1.0/T
        T_inv2 = T_inv*T_inv
        A = self.a_alpha*P*R2_inv*T_inv2
        B = b*P*R_inv*T_inv

        x0 = self.a_alpha
        x4 = 1.0/Z
        x5 = 4.0*P*R2_inv*x4*T_inv2*T_inv
        
        x8 = 2*P*R2_inv*T_inv2*dZ_dT/Z**2
        x9 = P*R2_inv*x4*T_inv2*self.da_alpha_dT/x0
        x10 = 1.0/P
        x11 = R*x10*(T*dZ_dT + Z)/(-R*T*x10*Z + b)**2
        x13 = b*T_inv*R_inv
        x14 = P*x13*x4 - 1.0
        x15 = x4*(P*x13*(T_inv + x4*dZ_dT) - x14*dZ_dT)/x14
        
        # Composition stuff
        d_lnphis_dTs = []
        for i in cmps:
            x1 = (ais[i]*x0)**0.5
            d_lhphi_dT = -bs[i]*x11 + x1*x5 + x1*x8 - x1*x9 + x15
            d_lnphis_dTs.append(d_lhphi_dT)
        return d_lnphis_dTs

    def d_lnphis_dP(self, Z, dZ_dP, zs):
        a_alpha = self.a_alpha
        cmps = self.cmps
        T, P, bs, b, ais = self.T, self.P, self.bs, self.b, self.ais
        
        T_inv = 1.0/T
        RT_inv = T_inv*R_inv

        x3 = T_inv*T_inv
        x5 = 1.0/Z
        x6 = 2.0*R2_inv*x3*x5
        x8 = 2.0*P*R2_inv*x3*dZ_dP*x5*x5
        x9 = 1./P
        x10 = Z*x9
        x11 = R*T*x9*(-x10 + dZ_dP)/(-R*T*x10 + b)**2
        x12 = P*x5
        x13 = b*RT_inv
        x14 = x12*x13 - 1.0
        x15 = -x5*(-x13*(x12*dZ_dP - 1.0) + x14*dZ_dP)/x14

        d_lnphi_dPs = []
        for i in cmps:
            x1 = (ais[i]*a_alpha)**0.5
            d_lnphi_dP = -bs[i]*x11 - x1*x6 + x1*x8 + x15
            d_lnphi_dPs.append(d_lnphi_dP)
        return d_lnphi_dPs                            


class PRSVMIX(PRMIX, PRSV):
    r'''Class for solving the Peng-Robinson-Stryjek-Vera equations of state for
    a mixture as given in [1]_.  Subclasses `PRMIX` and `PRSV`.
    Solves the EOS on initialization and calculates fugacities for all 
    components in all phases.
    
    Inherits the method of calculating fugacity coefficients from `PRMIX`.
    Two of `T`, `P`, and `V` are needed to solve the EOS.
    
    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}
        
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}
        
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}
        
        b = \sum_i z_i b_i

        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}
        
	  b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}
        
        \alpha(T)_i=[1+\kappa_i(1-\sqrt{T_{r,i}})]^2
        
        \kappa_i = \kappa_{0,i} + \kappa_{1,i}(1 + T_{r,i}^{0.5})(0.7 - T_{r,i})
        
        \kappa_{0,i} = 0.378893 + 1.4897153\omega_i - 0.17131848\omega_i^2 
        + 0.0196554\omega_i^3
        
    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    kappa1s : list[float], optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]

    Examples
    --------
    P-T initialization, two-phase, nitrogen and methane
    
    >>> eos = PRSVMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l/g', 3.6235523883756384e-05, -6349.003406339954, -49.12403359687132)
    
    Notes
    -----
    [1]_ recommends that `kappa1` be set to 0 for Tr > 0.7. This is not done by 
    default; the class boolean `kappa1_Tr_limit` may be set to True and the
    problem re-solved with that specified if desired. `kappa1_Tr_limit` is not
    supported for P-V inputs.
    
    For P-V initializations, SciPy's `newton` solver is used to find T.

    [2]_ and [3]_ are two more resources documenting the PRSV EOS. [4]_ lists
    `kappa` values for 69 additional compounds. See also `PRSV2`. Note that
    tabulated `kappa` values should be used with the critical parameters used
    in their fits. Both [1]_ and [4]_ only considered vapor pressure in fitting
    the parameter.

    References
    ----------
    .. [1] Stryjek, R., and J. H. Vera. "PRSV: An Improved Peng-Robinson 
       Equation of State for Pure Compounds and Mixtures." The Canadian Journal
       of Chemical Engineering 64, no. 2 (April 1, 1986): 323-33. 
       doi:10.1002/cjce.5450640224. 
    .. [2] Stryjek, R., and J. H. Vera. "PRSV - An Improved Peng-Robinson 
       Equation of State with New Mixing Rules for Strongly Nonideal Mixtures."
       The Canadian Journal of Chemical Engineering 64, no. 2 (April 1, 1986): 
       334-40. doi:10.1002/cjce.5450640225.  
    .. [3] Stryjek, R., and J. H. Vera. "Vapor-liquid Equilibrium of 
       Hydrochloric Acid Solutions with the PRSV Equation of State." Fluid 
       Phase Equilibria 25, no. 3 (January 1, 1986): 279-90. 
       doi:10.1016/0378-3812(86)80004-8. 
    .. [4] Proust, P., and J. H. Vera. "PRSV: The Stryjek-Vera Modification of 
       the Peng-Robinson Equation of State. Parameters for Other Pure Compounds
       of Industrial Interest." The Canadian Journal of Chemical Engineering 
       67, no. 1 (February 1, 1989): 170-73. doi:10.1002/cjce.5450670125.
    '''
    a_alpha_mro = -5
    eos_pure = PRSV
    nonstate_constants_specific = ('kappa0s', 'kappa1s', 'kappas')
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None, kappa1s=None):
        self.N = len(Tcs)
        self.cmps = range(self.N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs

        if kijs is None:
            kijs = [[0]*self.N for i in range(self.N)]
        self.kijs = kijs

        if kappa1s is None:
            kappa1s = [0.0 for i in self.cmps]
        self.kwargs = {'kijs': kijs, 'kappa1s': kappa1s}
        self.T = T
        self.P = P
        self.V = V

        self.ais = [self.c1*R*R*Tc*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.bs = [self.c2*R*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        
        self.kappa0s = [0.378893 + 1.4897153*omega - 0.17131848*omega**2 + 0.0196554*omega**3 for omega in omegas]
        
        self.delta = 2*self.b
        self.epsilon = -self.b*self.b

        self.check_sufficient_inputs()
        if self.V and self.P:
            # Deal with T-solution here; does NOT support kappa1_Tr_limit.
            self.kappa1s = kappa1s
            self.T = self.solve_T(self.P, self.V)
        else:
            self.kappa1s = [(0 if (T/Tc > 0.7 and self.kappa1_Tr_limit) else kappa1) for kappa1, Tc in zip(kappa1s, Tcs)]
            
        self.kappas = [kappa0 + kappa1*(1 + (self.T/Tc)**0.5)*(0.7 - (self.T/Tc)) for kappa0, kappa1, Tc in zip(self.kappa0s, self.kappa1s, self.Tcs)]
        self.solve()

        self.fugacities()


    def a_alpha_and_derivatives_vectorized(self, T, full=False, quick=True):
        raise NotImplementedError("Not Implemented")

    def setup_a_alpha_and_derivatives(self, i, T=None):
        r'''Sets `a`, `kappa0`, `kappa1`, and `Tc` for a specific component before the 
        pure-species EOS's `a_alpha_and_derivatives` method is called. Both are 
        called by `GCEOSMIX.a_alpha_and_derivatives` for every component.'''
        if not hasattr(self, 'kappas'):
            self.kappas = [kappa0 + kappa1*(1 + (T/Tc)**0.5)*(0.7 - (T/Tc)) for kappa0, kappa1, Tc in zip(self.kappa0s, self.kappa1s, self.Tcs)]
        self.a, self.kappa, self.kappa0, self.kappa1, self.Tc = self.ais[i], self.kappas[i], self.kappa0s[i], self.kappa1s[i], self.Tcs[i]

    def cleanup_a_alpha_and_derivatives(self):
        r'''Removes properties set by `setup_a_alpha_and_derivatives`; run by
        `GCEOSMIX.a_alpha_and_derivatives` after `a_alpha` is calculated for 
        every component'''
        del(self.a, self.kappa, self.kappa0, self.kappa1, self.Tc)
        

class PRSV2MIX(PRMIX, PRSV2):
    r'''Class for solving the Peng-Robinson-Stryjek-Vera 2 equations of state 
    for a Mixture as given in [1]_.  Subclasses `PRMIX` and `PRSV2`.
    Solves the EOS on initialization and calculates fugacities for all 
    components in all phases.

    Inherits the method of calculating fugacity coefficients from `PRMIX`.
    Two of `T`, `P`, and `V` are needed to solve the EOS.
    
    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}
        
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}
        
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}
        
        b = \sum_i z_i b_i

        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}
        
	  b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}
        
        \alpha(T)_i=[1+\kappa_i(1-\sqrt{T_{r,i}})]^2
        
        \kappa_i = \kappa_{0,i} + [\kappa_{1,i} + \kappa_{2,i}(\kappa_{3,i} - T_{r,i})(1-T_{r,i}^{0.5})]
        (1 + T_{r,i}^{0.5})(0.7 - T_{r,i})
        
        \kappa_{0,i} = 0.378893 + 1.4897153\omega_i - 0.17131848\omega_i^2 
        + 0.0196554\omega_i^3
        
    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    kappa1s : list[float], optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]
    kappa2s : list[float], optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]
    kappa3s : list[float], optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:
    
    >>> eos = PRSV2MIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (3.6235523883756384e-05, 0.0007002421492037558)
    >>> eos.fugacities_l, eos.fugacities_g
    ([794057.5831840535, 72851.22327178411], [436553.65618350444, 357878.1106688994])
    
    Notes
    -----    
    For P-V initializations, SciPy's `newton` solver is used to find T.

    Note that tabulated `kappa` values should be used with the critical 
    parameters used in their fits. [1]_ considered only vapor 
    pressure in fitting the parameter.

    References
    ----------
    .. [1] Stryjek, R., and J. H. Vera. "PRSV2: A Cubic Equation of State for 
       Accurate Vapor-liquid Equilibria Calculations." The Canadian Journal of 
       Chemical Engineering 64, no. 5 (October 1, 1986): 820-26. 
       doi:10.1002/cjce.5450640516. 
    '''
    a_alpha_mro = -5
    eos_pure = PRSV2
    nonstate_constants_specific = ('kappa1s', 'kappa2s', 'kappa3s', 'kappa0s', 'kappas')
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 kappa1s=None, kappa2s=None, kappa3s=None):
        self.N = len(Tcs)
        self.cmps = range(self.N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs

        if kijs is None:
            kijs = [[0]*self.N for i in range(self.N)]
        self.kijs = kijs

        if kappa1s is None:
            kappa1s = [0 for i in self.cmps]
        if kappa2s is None:
            kappa2s = [0 for i in self.cmps]
        if kappa3s is None:
            kappa3s = [0 for i in self.cmps]
        self.kwargs = {'kijs': kijs, 'kappa1s': kappa1s, 'kappa2s': kappa2s, 'kappa3s': kappa3s}
        self.kappa1s = kappa1s
        self.kappa2s = kappa2s
        self.kappa3s = kappa3s

        self.T = T
        self.P = P
        self.V = V

        self.ais = [self.c1*R*R*Tc*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.bs = [self.c2*R*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        
        self.kappa0s = [0.378893 + 1.4897153*omega - 0.17131848*omega**2 + 0.0196554*omega**3 for omega in omegas]
        
        self.delta = 2*self.b
        self.epsilon = -self.b*self.b

        
        if self.V and self.P:
            self.T = self.solve_T(self.P, self.V)
    
        self.kappas = []
        for Tc, kappa0, kappa1, kappa2, kappa3 in zip(Tcs, self.kappa0s, self.kappa1s, self.kappa2s, self.kappa3s):
            Tr = self.T/Tc
            kappa = kappa0 + ((kappa1 + kappa2*(kappa3 - Tr)*(1. - Tr**0.5))*(1. + Tr**0.5)*(0.7 - Tr))
            self.kappas.append(kappa)
        self.solve()
        self.fugacities()
        
    def setup_a_alpha_and_derivatives(self, i, T=None):
        r'''Sets `a`, `kappa`, `kappa0`, `kappa1`, `kappa2`, `kappa3` and `Tc`
        for a specific component before the 
        pure-species EOS's `a_alpha_and_derivatives` method is called. Both are 
        called by `GCEOSMIX.a_alpha_and_derivatives` for every component.'''
        if not hasattr(self, 'kappas'):
            self.kappas = []
            for Tc, kappa0, kappa1, kappa2, kappa3 in zip(self.Tcs, self.kappa0s, self.kappa1s, self.kappa2s, self.kappa3s):
                Tr = T/Tc
                kappa = kappa0 + ((kappa1 + kappa2*(kappa3 - Tr)*(1. - Tr**0.5))*(1. + Tr**0.5)*(0.7 - Tr))
                self.kappas.append(kappa)

        (self.a, self.kappa, self.kappa0, self.kappa1, self.kappa2, 
         self.kappa3, self.Tc) = (self.ais[i], self.kappas[i], self.kappa0s[i],
         self.kappa1s[i], self.kappa2s[i], self.kappa3s[i], self.Tcs[i])

    def cleanup_a_alpha_and_derivatives(self):
        r'''Removes properties set by `setup_a_alpha_and_derivatives`; run by
        `GCEOSMIX.a_alpha_and_derivatives` after `a_alpha` is calculated for 
        every component'''
        del(self.a, self.kappa, self.kappa0, self.kappa1, self.kappa2, self.kappa3, self.Tc)

    def a_alpha_and_derivatives_vectorized(self, T, full=False, quick=True):
        raise NotImplementedError("Not Implemented")

class TWUPRMIX(PRMIX, TWUPR):
    r'''Class for solving the Twu [1]_ variant of the Peng-Robinson cubic 
    equation of state for a mixture. Subclasses `TWUPR`. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.
    
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}
        
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}
        
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}
        
        b = \sum_i z_i b_i

        a_i=0.45724\frac{R^2T_{c,i}^2}{P_{c,i}}
        
	  b_i=0.07780\frac{RT_{c,i}}{P_{c,i}}
   
       \alpha_i = \alpha_i^{(0)} + \omega_i(\alpha_i^{(1)}-\alpha_i^{(0)})
       
       \alpha^{(\text{0 or 1})} = T_{r,i}^{N(M-1)}\exp[L(1-T_{r,i}^{NM})]
      
    For sub-critical conditions:
    
    L0, M0, N0 =  0.125283, 0.911807,  1.948150;
    
    L1, M1, N1 = 0.511614, 0.784054, 2.812520
    
    For supercritical conditions:
    
    L0, M0, N0 = 0.401219, 4.963070, -0.2;
    
    L1, M1, N1 = 0.024955, 1.248089, -8.  
        
    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:
    
    >>> eos = TWUPRMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (3.624569813157017e-05, 0.0007004398944116553)
    >>> eos.fugacities_l, eos.fugacities_g
    ([792155.022163319, 73305.88829726777], [436468.9677642441, 358049.24955730926])
    
    Notes
    -----
    For P-V initializations, SciPy's `newton` solver is used to find T.
    Claimed to be more accurate than the PR, PR78 and PRSV equations.

    References
    ----------
    .. [1] Twu, Chorng H., John E. Coon, and John R. Cunningham. "A New 
       Generalized Alpha Function for a Cubic Equation of State Part 1. 
       Peng-Robinson Equation." Fluid Phase Equilibria 105, no. 1 (March 15, 
       1995): 49-59. doi:10.1016/0378-3812(94)02601-V.
    '''
    a_alpha_mro = -5
    eos_pure = TWUPR
    
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None):
        self.N = len(Tcs)
        self.cmps = range(self.N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        if kijs is None:
            kijs = [[0]*self.N for i in range(self.N)]
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        self.ais = [self.c1*R*R*Tc*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.bs = [self.c2*R*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        
        self.delta = 2.*self.b
        self.epsilon = -self.b*self.b
        self.check_sufficient_inputs()

        self.solve()
        self.fugacities()

    def setup_a_alpha_and_derivatives(self, i, T=None):
        r'''Sets `a`, `omega`, and `Tc` for a specific component before the 
        pure-species EOS's `a_alpha_and_derivatives` method is called. Both are 
        called by `GCEOSMIX.a_alpha_and_derivatives` for every component.'''
        self.a, self.Tc, self.omega  = self.ais[i], self.Tcs[i], self.omegas[i]
    def cleanup_a_alpha_and_derivatives(self):
        r'''Removes properties set by `setup_a_alpha_and_derivatives`; run by
        `GCEOSMIX.a_alpha_and_derivatives` after `a_alpha` is calculated for 
        every component'''
        del(self.a, self.Tc, self.omega)

    def a_alpha_and_derivatives_vectorized(self, T, full=False, quick=True):
        raise NotImplementedError("Not Implemented")

class TWUSRKMIX(SRKMIX, TWUSRK):
    r'''Class for solving the Twu variant of the Soave-Redlich-Kwong cubic 
    equation of state for a mixture. Subclasses `TWUSRK`. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.
    
    Two of `T`, `P`, and `V` are needed to solve the EOS.
    
    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}
        
        a_i =\left(\frac{R^2(T_{c,i})^{2}}{9(\sqrt[3]{2}-1)P_{c,i}} \right)
        =\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}
    
        b_i =\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_{c,i}}{P_{c,i}}
        =\frac{0.08664\cdot R T_{c,i}}{P_{c,i}}
        
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}
        
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}
        
        b = \sum_i z_i b_i
        
        \alpha_i = \alpha^{(0,i)} + \omega_i(\alpha^{(1,i)}-\alpha^{(0,i)})
       
        \alpha^{(\text{0 or 1, i})} = T_{r,i}^{N(M-1)}\exp[L(1-T_{r,i}^{NM})]
      
    For sub-critical conditions:
    
    L0, M0, N0 =  0.141599, 0.919422, 2.496441
    
    L1, M1, N1 = 0.500315, 0.799457, 3.291790
    
    For supercritical conditions:
    
    L0, M0, N0 = 0.441411, 6.500018, -0.20
    
    L1, M1, N1 = 0.032580,  1.289098, -8.0
    
    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------    
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:
    
    >>> eos = TWUSRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (4.1087913616390855e-05, 0.000711707084027679)
    >>> eos.fugacities_l, eos.fugacities_g
    ([809692.8308266959, 74093.63881572774], [441783.43148985505, 362470.31741077645])
    
    Notes
    -----
    For P-V initializations, SciPy's `newton` solver is used to find T.
    Claimed to be more accurate than the SRK equation.

    References
    ----------
    .. [1] Twu, Chorng H., John E. Coon, and John R. Cunningham. "A New 
       Generalized Alpha Function for a Cubic Equation of State Part 2. 
       Redlich-Kwong Equation." Fluid Phase Equilibria 105, no. 1 (March 15, 
       1995): 61-69. doi:10.1016/0378-3812(94)02602-W.
    '''
    a_alpha_mro = -5
    eos_pure = TWUSRK
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None):
        self.N = len(Tcs)
        self.cmps = range(self.N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        if kijs is None:
            kijs = [[0]*self.N for i in range(self.N)]
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        self.ais = [self.c1*R*R*Tc*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.bs = [self.c2*R*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        
        self.delta = self.b
        self.check_sufficient_inputs()

        self.solve()
        self.fugacities()

    def setup_a_alpha_and_derivatives(self, i, T=None):
        r'''Sets `a`, `omega`, and `Tc` for a specific component before the 
        pure-species EOS's `a_alpha_and_derivatives` method is called. Both are 
        called by `GCEOSMIX.a_alpha_and_derivatives` for every component.'''
        self.a, self.Tc, self.omega  = self.ais[i], self.Tcs[i], self.omegas[i]

    def cleanup_a_alpha_and_derivatives(self):
        r'''Removes properties set by `setup_a_alpha_and_derivatives`; run by
        `GCEOSMIX.a_alpha_and_derivatives` after `a_alpha` is calculated for 
        every component'''
        del(self.a, self.Tc, self.omega)


class APISRKMIX(SRKMIX, APISRK):
    r'''Class for solving the Refinery Soave-Redlich-Kwong cubic 
    equation of state for a mixture of any number of compounds, as shown in the
    API Databook [1]_. Subclasses `APISRK`. Solves the EOS on
    initialization and calculates fugacities for all components in all phases.
        
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}
        
        a \alpha = \sum_i \sum_j z_i z_j {(a\alpha)}_{ij}
        
        (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

        b = \sum_i z_i b_i

        a_i =\left(\frac{R^2(T_{c,i})^{2}}{9(\sqrt[3]{2}-1)P_{c,i}} \right)
        =\frac{0.42748\cdot R^2(T_{c,i})^{2}}{P_{c,i}}
    
        b_i =\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_{c,i}}{P_{c,i}}
        =\frac{0.08664\cdot R T_{c,i}}{P_{c,i}}
        
        \alpha(T)_i = \left[1 + S_{1,i}\left(1-\sqrt{T_{r,i}}\right) + S_{2,i}
        \frac{1- \sqrt{T_{r,i}}}{\sqrt{T_{r,i}}}\right]^2
        
        S_{1,i} = 0.48508 + 1.55171\omega_i - 0.15613\omega_i^2 \text{ if S1 is not tabulated }
        
    Parameters
    ----------
    Tcs : float
        Critical temperatures of all compounds, [K]
    Pcs : float
        Critical pressures of all compounds, [Pa]
    omegas : float
        Acentric factors of all compounds, [-]
    zs : float
        Overall mole fractions of all species, [-]
    kijs : list[list[float]], optional
        n*n size list of lists with binary interaction parameters for the
        Van der Waals mixing rules, default all 0 [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    S1s : float, optional
        Fit constant or estimated from acentric factor if not provided [-]
    S2s : float, optional
        Fit constant or 0 if not provided [-]

    Notes
    -----
    For P-V initializations, SciPy's `newton` solver is used to find T.

    Examples
    --------    
    T-P initialization, nitrogen-methane at 115 K and 1 MPa:
    
    >>> eos = APISRKMIX(T=115, P=1E6, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]])
    >>> eos.V_l, eos.V_g
    (4.1015909205567394e-05, 0.0007104685894929316)
    >>> eos.fugacities_l, eos.fugacities_g
    ([817882.3033490371, 71620.48238123357], [442158.29113191745, 361519.7987757053])

    References
    ----------
    .. [1] API Technical Data Book: General Properties & Characterization.
       American Petroleum Institute, 7E, 2005.
    '''
    a_alpha_mro = -5
    eos_pure = APISRK
    nonstate_constants_specific = ('S1s', 'S2s')
    def __init__(self, Tcs, Pcs, zs, omegas=None, kijs=None, T=None, P=None, V=None,
                 S1s=None, S2s=None):
        self.N = len(Tcs)
        self.cmps = range(self.N)
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.zs = zs
        if kijs is None:
            kijs = [[0.0]*self.N for i in range(self.N)]
        self.kijs = kijs
        self.kwargs = {'kijs': kijs}
        self.T = T
        self.P = P
        self.V = V

        self.check_sufficient_inputs()

        # Setup S1s and S2s
        if S1s is None and omegas is None:
            raise Exception('Either acentric factor of S1 is required')
        if S1s is None:
            self.S1s = [0.48508 + 1.55171*omega - 0.15613*omega*omega for omega in omegas]
        else:
            self.S1s = S1s
        if S2s is None:
            S2s = [0.0 for i in self.cmps]
        self.S2s = S2s
        
        self.ais = [self.c1*R*R*Tc*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.bs = [self.c2*R*Tc/Pc for Tc, Pc in zip(Tcs, Pcs)]
        self.b = sum(bi*zi for bi, zi in zip(self.bs, self.zs))
        self.delta = self.b

        self.solve()
        self.fugacities()

    def setup_a_alpha_and_derivatives(self, i, T=None):
        r'''Sets `a`, `S1`, `S2` and `Tc` for a specific component before the 
        pure-species EOS's `a_alpha_and_derivatives` method is called. Both are 
        called by `GCEOSMIX.a_alpha_and_derivatives` for every component.'''
        self.a, self.Tc, self.S1, self.S2  = self.ais[i], self.Tcs[i], self.S1s[i], self.S2s[i]

    def cleanup_a_alpha_and_derivatives(self):
        r'''Removes properties set by `setup_a_alpha_and_derivatives`; run by
        `GCEOSMIX.a_alpha_and_derivatives` after `a_alpha` is calculated for 
        every component'''
        del(self.a, self.Tc, self.S1, self.S2)


def eos_Z_test_phase_stability(eos):        
    try:
        if eos.G_dep_l < eos.G_dep_g:
            Z_eos = eos.Z_l
            prefer, alt = 'Z_g', 'Z_l'
        else:
            Z_eos = eos.Z_g
            prefer, alt =  'Z_l', 'Z_g'
    except:
        # Only one root - take it and set the prefered other phase to be a different type
        Z_eos = eos.Z_g if hasattr(eos, 'Z_g') else eos.Z_l
        prefer = 'Z_l' if hasattr(eos, 'Z_g') else 'Z_g'
        alt = 'Z_g' if hasattr(eos, 'Z_g') else 'Z_l'
    return Z_eos, prefer, alt


def eos_Z_trial_phase_stability(eos, prefer, alt):
    try:
        if eos.G_dep_l < eos.G_dep_g:
            Z_trial = eos.Z_l
        else:
            Z_trial = eos.Z_g
    except:
        # Only one phase, doesn't matter - only that phase will be returned
        try:
            Z_trial = getattr(eos, alt)
        except:
            Z_trial = getattr(eos, prefer)
    return Z_trial
