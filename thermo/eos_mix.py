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
'eos_Z_test_phase_stability', 'eos_Z_trial_phase_stability',
'eos_mix_list']

import sys
import numpy as np
from cmath import log as clog, atanh as catanh
from scipy.optimize import minimize
from scipy.misc import derivative
from fluids.numerics import IS_PYPY, newton_system, UnconvergedError
from thermo.utils import normalize, Cp_minus_Cv, isobaric_expansion, isothermal_compressibility, phase_identification_parameter, dxs_to_dn_partials, dxs_to_dns, dns_to_dn_partials
from thermo.utils import R
from thermo.utils import log, exp, sqrt
from thermo.eos import *
from thermo.activity import Wilson_K_value, K_value, flash_inner_loop, Rachford_Rice_flash_error, Rachford_Rice_solution2

R2 = R*R
R_inv = 1.0/R
R2_inv = R_inv*R_inv

two_root_two = 2*2**0.5
root_two = sqrt(2.)
root_two_m1 = root_two - 1.0
root_two_p1 = root_two + 1.0
log_min = log(sys.float_info.min)


def a_alpha_aijs_composition_independent(a_alphas, kijs):
    N = len(a_alphas)
    cmps = range(N)
    
    a_alpha_ijs = [[0.0]*N for _ in cmps]
    a_alpha_i_roots = [a_alpha_i**0.5 for a_alpha_i in a_alphas]
#    a_alpha_i_roots_inv = [1.0/i for i in a_alpha_i_roots] # Storing this to avoid divisions was not faster when tested
    # Tried optimization - skip the divisions - can just store the inverses of a_alpha_i_roots and do another multiplication
    # Store the inverses of a_alpha_ij_roots
    a_alpha_ij_roots_inv = [[0.0]*N for _ in cmps]
    
    for i in cmps:
        kijs_i = kijs[i]
        a_alpha_i = a_alphas[i]
        a_alpha_ijs_is = a_alpha_ijs[i]
        a_alpha_ij_roots_i_inv = a_alpha_ij_roots_inv[i]
        # Using range like this saves 20% of the comp time for 44 components!
        a_alpha_i_root_i = a_alpha_i_roots[i]
        for j in range(i, N):
#        for j in cmps:
#            # TODo range
#            if j < i:
#                continue
            term = a_alpha_i_root_i*a_alpha_i_roots[j]
#            a_alpha_ij_roots_i_inv[j] = a_alpha_i_roots_inv[i]*a_alpha_i_roots_inv[j]#1.0/term
            a_alpha_ij_roots_i_inv[j] = 1.0/term
            a_alpha_ijs_is[j] = a_alpha_ijs[j][i] = (1. - kijs_i[j])*term
    return a_alpha_ijs, a_alpha_i_roots, a_alpha_ij_roots_inv

def a_alpha_and_derivatives(a_alphas, T, zs, kijs, a_alpha_ijs=None,
                            a_alpha_i_roots=None, a_alpha_ij_roots_inv=None):
    N = len(a_alphas)
    da_alpha_dT, d2a_alpha_dT2 = 0.0, 0.0
    
    if a_alpha_ijs is None or a_alpha_i_roots is None or a_alpha_ij_roots_inv is None:
        a_alpha_ijs, a_alpha_i_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, kijs)
            
    a_alpha = 0.0
    for i in range(N):
        a_alpha_ijs_i = a_alpha_ijs[i]
        zi = zs[i]
        for j in range(i+1, N):
            term = a_alpha_ijs_i[j]*zi*zs[j]
            a_alpha += term + term
                
        a_alpha += a_alpha_ijs_i[i]*zi*zi
                
    return a_alpha, None, a_alpha_ijs


def a_alpha_and_derivatives_full(a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, zs, 
                                 kijs, a_alpha_ijs=None, a_alpha_i_roots=None,
                                 a_alpha_ij_roots_inv=None,
                                 second_derivative=True):
    # For 44 components, takes 150 us in PyPy.
    
    N = len(a_alphas)
    cmps = range(N)
    da_alpha_dT, d2a_alpha_dT2 = 0.0, 0.0
    
    if a_alpha_ijs is None or a_alpha_i_roots is None or a_alpha_ij_roots_inv is None:
        a_alpha_ijs, a_alpha_i_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, kijs)
            
    z_products = [[zs[i]*zs[j] for j in cmps] for i in cmps]

    a_alpha = 0.0
    for i in cmps:
        a_alpha_ijs_i = a_alpha_ijs[i]
        z_products_i = z_products[i]
        for j in cmps:
            if j < i:
                continue
            term = a_alpha_ijs_i[j]*z_products_i[j]
            if i != j:
                a_alpha += term + term
            else:
                a_alpha += term
    
    da_alpha_dT_ijs = [[0.0]*N for _ in cmps]
    
    d2a_alpha_dT2_ij = 0.0
    
    for i in cmps:
        kijs_i = kijs[i]
        a_alphai = a_alphas[i]
        z_products_i = z_products[i]
        da_alpha_dT_i = da_alpha_dTs[i]
        d2a_alpha_dT2_i = d2a_alpha_dT2s[i]
        a_alpha_ij_roots_inv_i = a_alpha_ij_roots_inv[i]
        da_alpha_dT_ijs_i = da_alpha_dT_ijs[i]
        
        for j in cmps:
#        for j in range(0, i+1):
            if j < i:
#                # skip the duplicates
                continue
            a_alphaj = a_alphas[j]
            x0_05_inv = a_alpha_ij_roots_inv_i[j]
            zi_zj = z_products_i[j]
            da_alpha_dT_j = da_alpha_dTs[j]
            
            x1 = a_alphai*da_alpha_dT_j
            x2 = a_alphaj*da_alpha_dT_i
            x1_x2 = x1 + x2
            x3 = x1_x2 + x1_x2

            kij_m1 = kijs_i[j] - 1.0
            
            da_alpha_dT_ij = -0.5*kij_m1*x1_x2*x0_05_inv
            # For temperature derivatives of fugacities 
            da_alpha_dT_ijs_i[j] = da_alpha_dT_ijs[j][i] = da_alpha_dT_ij

            da_alpha_dT_ij *= zi_zj
            
            
            x0 = a_alphai*a_alphaj
        
            d2a_alpha_dT2_ij = zi_zj*kij_m1*(  (x0*(
            -0.5*(a_alphai*d2a_alpha_dT2s[j] + a_alphaj*d2a_alpha_dT2_i)
            - da_alpha_dT_i*da_alpha_dT_j) +.25*x1_x2*x1_x2)/(x0_05_inv*x0*x0))
            
            
            if i != j:
                da_alpha_dT += da_alpha_dT_ij + da_alpha_dT_ij
                d2a_alpha_dT2 += d2a_alpha_dT2_ij + d2a_alpha_dT2_ij
            else:
                da_alpha_dT += da_alpha_dT_ij
                d2a_alpha_dT2 += d2a_alpha_dT2_ij

    return a_alpha, da_alpha_dT, d2a_alpha_dT2, da_alpha_dT_ijs, a_alpha_ijs



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
    multicomponent = True
#    def fast_copy_base(self, a_alphas=False):
#        new = self.__class__.__new__(self.__class__)
##        for attr in self.nonstate_constants:
##            setattr(new, attr, getattr(self, attr))
#        new.N = self.N
#        new.cmps = self.cmps
#        new.Tcs = self.Tcs
#        new.Pcs = self.Pcs
#        new.omegas = self.omegas
#        new.kijs = self.kijs
#        new.kwargs = self.kwargs
#        new.ais = self.ais
#        new.bs = self.bs
#
##        for attr in self.nonstate_constants_specific:
##            setattr(new, attr, getattr(self, attr))
#        if a_alphas:
#            new.a_alphas = self.a_alphas
#            new.da_alpha_dTs = self.da_alpha_dTs
#            new.d2a_alpha_dT2s = self.d2a_alpha_dT2s
#        return new
    
    def to_TP_zs_fast(self, T, P, zs, only_l=False, only_g=False, full_alphas=True):
        
        new = self.__class__.__new__(self.__class__)
        new.N = self.N
        new.cmps = self.cmps
        new.Tcs = self.Tcs
        new.Pcs = self.Pcs
        new.omegas = self.omegas
        new.kijs = self.kijs
        new.kwargs = self.kwargs
        new.ais = self.ais
        new.bs = self.bs
        
        copy_alphas = T == self.T
        if copy_alphas:
            new.a_alphas = self.a_alphas
            new.da_alpha_dTs = self.da_alpha_dTs
            new.d2a_alpha_dT2s = self.d2a_alpha_dT2s
            try:
                new.a_alpha_ijs = self.a_alpha_ijs
                new.a_alpha_i_roots = self.a_alpha_i_roots
                new.a_alpha_ij_roots_inv = self.a_alpha_ij_roots_inv
            except:
                pass
        
        new.zs = zs
        new.T = T
        new.P = P
        new.V = None
        new.fast_init_specific(self)
        new.solve(pure_a_alphas=(not copy_alphas), only_l=only_l, only_g=only_g, full_alphas=full_alphas)
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
        
        if quick:
            try:
                a_alpha_ijs, a_alpha_i_roots, a_alpha_ij_roots_inv = self.a_alpha_ijs, self.a_alpha_i_roots, self.a_alpha_ij_roots_inv
            except AttributeError:
                a_alpha_ijs, a_alpha_i_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, kijs)
                self.a_alpha_ijs, self.a_alpha_i_roots, self.a_alpha_ij_roots_inv = a_alpha_ijs, a_alpha_i_roots, a_alpha_ij_roots_inv
        else:
            a_alpha_ijs, a_alpha_i_roots, a_alpha_ij_roots_inv = a_alpha_aijs_composition_independent(a_alphas, kijs)
            self.a_alpha_ijs, self.a_alpha_i_roots, self.a_alpha_ij_roots_inv = a_alpha_ijs, a_alpha_i_roots, a_alpha_ij_roots_inv
        
        if full:
            a_alpha, da_alpha_dT, d2a_alpha_dT2, da_alpha_dT_ijs, a_alpha_ijs = a_alpha_and_derivatives_full(a_alphas, da_alpha_dTs, d2a_alpha_dT2s, T, zs, kijs,
                                                                                                             a_alpha_ijs, a_alpha_i_roots, a_alpha_ij_roots_inv)
            self.da_alpha_dT_ijs = da_alpha_dT_ijs
            self.a_alpha_ijs = a_alpha_ijs
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
        else:
            # Priority - test, fix, and validate
            a_alpha, _, a_alpha_ijs = a_alpha_and_derivatives(a_alphas, T, zs, kijs, a_alpha_ijs, a_alpha_i_roots, a_alpha_ij_roots_inv)
            self.da_alpha_dT_ijs = []
            self.a_alpha_ijs = a_alpha_ijs
            return a_alpha

        # DO NOT REMOVE THIS CODE! IT MAKES TIHNGS SLOWER IN PYPY, even though it never runs
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
        
        try:
            del self.a_alpha_ijs
            del self.a_alpha_i_roots
            del self.a_alpha_ij_roots_inv
        except:
            pass
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
        zs, Tcs, Pcs = self.zs, self.Tcs, self.Pcs
        Pmc = sum([Pcs[i]*zs[i] for i in self.cmps])
        Tmc = sum([(Tcs[i]*Tcs[j])**0.5*zs[j]*zs[i] for i in self.cmps
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
        if self.phase in ('l', 'l/g') and hasattr(self, 'Z_l'):
            if xs is None:
                xs = self.zs
            self.lnphis_l = self.fugacity_coefficients(self.Z_l, zs=xs)
            self.phis_l = [exp(i) for i in self.lnphis_l]
            self.fugacities_l = [phi*x*P for phi, x in zip(self.phis_l, xs)]

        if self.phase in ('g', 'l/g') and hasattr(self, 'Z_g'):
            if ys is None:
                ys = self.zs
            self.lnphis_g = self.fugacity_coefficients(self.Z_g, zs=ys)
            self.phis_g = [exp(i) for i in self.lnphis_g]
            self.fugacities_g = [phi*y*P for phi, y in zip(self.phis_g, ys)]

    
    def eos_lnphis_lowest_Gibbs(self):
        try:        
            try:
                if self.G_dep_l < self.G_dep_g:
                    return self.lnphis_l, 'l'
                else:
                    return self.lnphis_g, 'g'
            except:
                # Only one root - take it and set the prefered other phase to be a different type
                return (self.lnphis_g, 'g') if hasattr(self, 'Z_g') else (self.lnphis_l, 'l')
        except:
            self.fugacities()
            return self.eos_fugacities_lowest_Gibbs()


    def eos_fugacities_lowest_Gibbs(self):
        try:        
            try:
                if self.G_dep_l < self.G_dep_g:
                    return self.fugacities_l, 'l'
                else:
                    return self.fugacities_g, 'g'
            except:
                # Only one root - take it and set the prefered other phase to be a different type
                return (self.fugacities_g, 'g') if hasattr(self, 'Z_g') else (self.fugacities_l, 'l')
        except:
            self.fugacities()
            return self.eos_fugacities_lowest_Gibbs()


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
    
    @staticmethod
    def Stateva_Tsvetkov_TPDF_fixed(log_phi_zs, log_phi_ys, zs, ys):
        kis = []
        for yi, log_phi_yi, zi, log_phi_zi in zip(ys, log_phi_ys, zs, log_phi_zs):
            di = log_phi_zi + (log(zi) if zi > 0.0 else -690.0)
            
            
            try:
                ki = log_phi_yi + log(yi) - di
            except ValueError:
                # log - yi is negative; convenient to handle it to make the optimization take negative comps
                ki = log_phi_yi + -690.0 - di
            kis.append(ki)
        kis.append(kis[0])

        tot = 0.0
        for i in range(len(zs)):
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

    
    def _err_VL_jacobian(self, lnKsVF, T, P, zs, near_critical=False,
                         err_also=False, info=None):
        if info is None:
            info = []
        N, cmps = self.N, self.cmps
        lnKs = lnKsVF[:-1]
        Ks = [exp(lnKi) for lnKi in lnKs]
        VF = float(lnKsVF[-1])
        
        
        xs = [zi/(1.0 + VF*(Ki - 1.0)) for zi, Ki in zip(zs, Ks)]
        ys = [Ki*xi for Ki, xi in zip(Ks, xs)]

        eos_g = self.to_TP_zs_fast(T=T, P=P, zs=ys, only_g=True) # 
        eos_l = self.to_TP_zs_fast(T=T, P=P, zs=xs, only_l=True) # 

#        eos_g = self.to_TP_zs(T=T, P=P, zs=ys)
#        eos_l = self.to_TP_zs(T=T, P=P, zs=xs)
        if not near_critical:
#            lnphis_g = eos_g.lnphis_g
#            lnphis_l = eos_l.lnphis_l
            Z_g = eos_g.Z_g
            Z_l = eos_l.Z_l
        else:
            try:
#                lnphis_g = eos_g.lnphis_g
                Z_g = eos_g.Z_g
            except AttributeError:
#                lnphis_g = eos_g.lnphis_l
                Z_g = eos_g.Z_l
            try:
#                lnphis_l = eos_l.lnphis_l
                Z_l = eos_l.Z_l
            except AttributeError:
#                lnphis_l = eos_l.lnphis_g
                Z_l = eos_l.Z_g

        lnphis_g = eos_g.fugacity_coefficients(Z_g, ys)
        lnphis_l = eos_l.fugacity_coefficients(Z_l, xs)

        size = N + 1
        J = [[None]*size for i in range(size)]
        
        
#        d_lnphi_dzs_basic_num
#        d_lnphi_dxs = eos_l.d_lnphi_dzs_basic_num(Z_l, xs)
#        d_lnphi_dys = eos_g.d_lnphi_dzs_basic_num(Z_g, ys)
        d_lnphi_dxs = eos_l.d_lnphi_dzs(Z_l, xs)
        d_lnphi_dys = eos_g.d_lnphi_dzs(Z_g, ys)
        
        
        
#        # Handle the zeros and the ones
        # Half of this is probably wrong! Only gets set for one set of variables?
        # Numerical jacobian not good enough to tell
#        for i in range(self.N):
#            J[i][-2] = 0.0
#            J[-2][i] = 0.0
            
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
                # pretty sure indexing is right in the below expression
                value += RR_terms[k]*(d_lnphi_dxs_i[k] - Ks[k]*d_lnphi_dys_i[k])
            J[i][-1] = value
#            print(value)
        
#        def delta(k, j):
#            if k == j:
#                return 1.0
#            return 0.0
        
        
            
        # Main body - expensive to compute! Lots of elements
        # Can flip around the indexing of i, j on the d_lnphi_ds but still no fix
        # unsure of correct order!
        # Reveals bugs in d_lnphi_dxs though.
        zsKsRRinvs2 = [zs[j]*Ks[j]*RR_denoms_inv2[j] for j in cmps]
        one_m_VF = 1.0 - VF
        for i in cmps: # to N is CORRECT/MATCHES JACOBIAN NUMERICALLY
            Ji = J[i]
            d_lnphi_dxs_is, d_lnphi_dys_is = d_lnphi_dxs[i], d_lnphi_dys[i]
            for j in cmps: # to N is CORRECT/MATCHES JACOBIAN NUMERICALLY
                value = 1.0 if i == j else 0.0
#                value = 0.0
#                value += delta(i, j)
#                print(i, j, value)
                # Maybe if i == j, can skip the bit below? Tried it once and the solver never converged
#                term = zs[j]*Ks[j]*RR_denoms_inv2[j]
                value += zsKsRRinvs2[j]*(VF*d_lnphi_dxs_is[j] + one_m_VF*d_lnphi_dys_is[j])
                Ji[j] = value
            
        # Last row except last value  - good, working
        # Diff of RR w.r.t each log K
        bottom_row = J[-1]
        for j in cmps:
#            value = 0.0
#            RR_l = 
#            RR_l = -Ks[j]*zs[j]*VF/(1.0 + VF*(Ks[j] - 1.0))**2.0
#            RR_g = Ks[j]*(1.0 - VF)*zs[j]/(1.0 + VF*(Ks[j] - 1.0))**2.0
#            value +=  #  -RR_l
            bottom_row[j] = zsKsRRinvs2[j]*(one_m_VF) + VF*zsKsRRinvs2[j]
        # Last row except last value  - good, working
#        bottom_row = J[-1]
#        for j in range(self.N):
#            value = 0.0
#            for k in range(self.N):
#                if k == j:
#                    RR_l = -Ks[j]*zs[k]*VF/(1.0 + VF*(Ks[k] - 1.0))**2.0
#                    RR_g = Ks[j]*(1.0 - VF)*zs[k]/(1.0 + VF*(Ks[k] - 1.0))**2.0
#                    value += RR_g - RR_l
#            bottom_row[j] = value
#
        # Last value - good, working, being overwritten
        dF_ncp1_dB = 0.0
        for i in cmps:
            dF_ncp1_dB -= RR_terms[i]*Ksm1[i]
        J[-1][-1] = dF_ncp1_dB
        
        info[:] = VF, xs, ys, eos_l, eos_g
        
        if err_also:
            err_RR = Rachford_Rice_flash_error(VF, zs, Ks)
            Fs = [lnKi - lnphi_l + lnphi_g for lnphi_l, lnphi_g, lnKi in zip(lnphis_l, lnphis_g, lnKs)]
            Fs.append(err_RR)
            return Fs, J

        return J
            
    def _err_VL(self, lnKsVF, T, P, zs, near_critical=False):
        import numpy as np
        # tried autograd without luck
        lnKs = lnKsVF[:-1]
        if isinstance(lnKs, np.ndarray):
            lnKs = lnKs.tolist()
#        Ks = np.exp(lnKs)
        Ks = [exp(lnKi) for lnKi in lnKs]
        VF = float(lnKsVF[-1])
#        VF = lnKsVF[-1]
        
        xs = [zi/(1.0 + VF*(Ki - 1.0)) for zi, Ki in zip(zs, Ks)]
        ys = [Ki*xi for Ki, xi in zip(Ks, xs)]
        
        err_RR = Rachford_Rice_flash_error(VF, zs, Ks)

        eos_g = self.to_TP_zs_fast(T=T, P=P, zs=ys, only_g=True) # 
        eos_g.fugacities()
        eos_l = self.to_TP_zs_fast(T=T, P=P, zs=xs, only_l=True) # 
        eos_l.fugacities()
        
        if not near_critical:
            lnphis_g = eos_g.lnphis_g
            lnphis_l = eos_l.lnphis_l
        else:
            try:
                lnphis_g = eos_g.lnphis_g
            except AttributeError:
                lnphis_g = eos_g.lnphis_l
            try:
                lnphis_l = eos_l.lnphis_l
            except AttributeError:
                lnphis_l = eos_l.lnphis_g
#        Fs = [fl/fg-1.0 for fl, fg in zip(fugacities_l, fugacities_g)]
        Fs = [lnKi - lnphi_l + lnphi_g for lnphi_l, lnphi_g, lnKi in zip(lnphis_l, lnphis_g, lnKs)]
        Fs.append(err_RR)
        return Fs
        
    def sequential_substitution_3P(self, Ks_y, Ks_z, beta_y, beta_z=0.0,
                                   
                                   maxiter=1000,
                                   xtol=1E-13, near_critical=True,
                                   xs=None, ys=None, zs=None,
                                   trivial_solution_tol=1e-5):
        
        
        print(Ks_y, Ks_z, beta_y, beta_z)
        beta_y, beta_z, xs_new, ys_new, zs_new = Rachford_Rice_solution2(ns=self.zs, 
                                                                         Ks_y=Ks_y, Ks_z=Ks_z,
                                                                         beta_y=beta_y, beta_z=beta_z)
        print(beta_y, beta_z, xs_new, ys_new, zs_new)
        
        Ks_y = [exp(lnphi_x - lnphi_y) for lnphi_x, lnphi_y in zip(lnphis_x, lnphis_y)]
        Ks_z = [exp(lnphi_x - lnphi_z) for lnphi_x, lnphi_z in zip(lnphis_x, lnphis_z)]

    def newton_VL(self, Ks_initial=None, maxiter=30,
                  ytol=1E-7, near_critical=True,
                  xs=None, ys=None, V_over_F=None):
        T, P, zs = self.T, self.P, self.zs
        if xs is not None and ys is not None and V_over_F is not None:
            pass
        else:
            if Ks_initial is None:
                Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
            else:
                Ks = Ks_initial
            V_over_F, xs, ys = flash_inner_loop(zs, Ks)
        
        
        
        
        lnKs_guess = [log(yi/xi) for yi, xi in zip(ys, xs)] + [V_over_F]
        
        info = []
        def err_and_jacobian(lnKs_guess):
            err =  self._err_VL_jacobian(lnKs_guess, T, P, zs, near_critical=True, err_also=True, info=info)
#            print(lnKs_guess[-1], err)
            return err

        ans, count = newton_system(err_and_jacobian, jac=True, x0=lnKs_guess, ytol=ytol, maxiter=maxiter)
        V_over_F, xs, ys, eos_l, eos_g = info
        return V_over_F, xs, ys, eos_l, eos_g
        

    def sequential_substitution_VL(self, Ks_initial=None, maxiter=1000,
                                   xtol=1E-13, near_critical=True, Ks_extra=None,
                                   xs=None, ys=None, trivial_solution_tol=1e-5, info=None,
                                   full_alphas=False):
#        print(self.zs, Ks)
        T, P, zs = self.T, self.P, self.zs
        V_over_F = None
        if xs is not None and ys is not None:
            pass
        else:
            # TODO use flash_wilson here
            if Ks_initial is None:
                Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
            else:
                Ks = Ks_initial
            xs = None
            try:
                V_over_F, xs, ys = flash_inner_loop(zs, Ks)
            except ValueError as e:
                if Ks_extra is not None:
                    for Ks in Ks_extra:
                        try:
                            V_over_F, xs, ys = flash_inner_loop(zs, Ks)
                            break
                        except ValueError as e:
                            pass
            if xs is None:
                raise(e)
        
#        print(xs, ys, 'innerloop')
#        Z_l_prev = None
#        Z_g_prev = None

        for i in range(maxiter):
            if not near_critical:
                eos_g = self.to_TP_zs_fast(T=T, P=P, zs=ys, only_l=False, only_g=True, full_alphas=full_alphas)
                eos_l = self.to_TP_zs_fast(T=T, P=P, zs=xs, only_l=True, only_g=False, full_alphas=full_alphas)
                lnphis_g = eos_g.fugacity_coefficients(eos_g.Z_g, ys)
                lnphis_l = eos_l.fugacity_coefficients(eos_l.Z_l, xs)
            else:
                eos_g = self.to_TP_zs_fast(T=T, P=P, zs=ys, only_l=False, only_g=True, full_alphas=full_alphas)
                eos_l = self.to_TP_zs_fast(T=T, P=P, zs=xs, only_l=True, only_g=False, full_alphas=full_alphas)                
                try:
                    lnphis_g = eos_g.fugacity_coefficients(eos_g.Z_g, ys)
                except AttributeError:
                    lnphis_g = eos_g.fugacity_coefficients(eos_g.Z_l, ys)
                try:
                    lnphis_l = eos_l.fugacity_coefficients(eos_l.Z_l, xs)
                except AttributeError:
                    lnphis_l = eos_l.fugacity_coefficients(eos_l.Z_g, xs)

                
#                eos_g = self.to_TP_zs(T=self.T, P=self.P, zs=ys)
#                eos_l = self.to_TP_zs(T=self.T, P=self.P, zs=xs)
#                if 0:
#                    if hasattr(eos_g, 'lnphis_g') and hasattr(eos_g, 'lnphis_l'):
#                        if Z_l_prev is not None and Z_g_prev is not None:
#                            if abs(eos_g.Z_g - Z_g_prev) < abs(eos_g.Z_l - Z_g_prev):
#                                lnphis_g = eos_g.lnphis_g
#                                fugacities_g = eos_g.fugacities_g
#                                Z_g_prev = eos_g.Z_g
#                            else:
#                                lnphis_g = eos_g.lnphis_l
#                                fugacities_g = eos_g.fugacities_l
#                                Z_g_prev = eos_g.Z_l
#                        else:
#                            if eos_g.G_dep_g < eos_g.lnphis_l:
#                                lnphis_g = eos_g.lnphis_g
#                                fugacities_g = eos_g.fugacities_g
#                                Z_g_prev = eos_g.Z_g
#                            else:
#                                lnphis_g = eos_g.lnphis_l
#                                fugacities_g = eos_g.fugacities_l
#                                Z_g_prev = eos_g.Z_l
#                    else:
#                        try:
#                            lnphis_g = eos_g.lnphis_g#fugacity_coefficients(eos_g.Z_g, ys)
#                            fugacities_g = eos_g.fugacities_g
#                            Z_g_prev = eos_g.Z_g
#                        except AttributeError:
#                            lnphis_g = eos_g.lnphis_l#fugacity_coefficients(eos_g.Z_l, ys)
#                            fugacities_g = eos_g.fugacities_l
#                            Z_g_prev = eos_g.Z_l
#                    if hasattr(eos_l, 'lnphis_g') and hasattr(eos_l, 'lnphis_l'):
#                        if Z_l_prev is not None and Z_g_prev is not None:
#                            if abs(eos_l.Z_l - Z_l_prev) < abs(eos_l.Z_g - Z_l_prev):
#                                lnphis_l = eos_l.lnphis_g
#                                fugacities_l = eos_l.fugacities_g
#                                Z_l_prev = eos_l.Z_g
#                            else:
#                                lnphis_l = eos_l.lnphis_l
#                                fugacities_l = eos_l.fugacities_l
#                                Z_l_prev = eos_l.Z_l
#                        else:
#                            if eos_l.G_dep_g < eos_l.lnphis_l:
#                                lnphis_l = eos_l.lnphis_g
#                                fugacities_l = eos_l.fugacities_g
#                                Z_l_prev = eos_l.Z_g
#                            else:
#                                lnphis_l = eos_l.lnphis_l
#                                fugacities_l = eos_l.fugacities_l
#                                Z_l_prev = eos_l.Z_l
#                    else:
#                        try:
#                            lnphis_l = eos_l.lnphis_g#fugacity_coefficients(eos_l.Z_g, ys)
#                            fugacities_l = eos_l.fugacities_g
#                            Z_l_prev = eos_l.Z_g
#                        except AttributeError:
#                            lnphis_l = eos_l.lnphis_l#fugacity_coefficients(eos_l.Z_l, ys)
#                            fugacities_l = eos_l.fugacities_l
#                            Z_l_prev = eos_l.Z_l
#                elif 0:
#                    if hasattr(eos_g, 'lnphis_g') and hasattr(eos_g, 'lnphis_l'):
#                        if eos_g.G_dep_g < eos_g.lnphis_l:
#                            lnphis_g = eos_g.lnphis_g
#                            fugacities_g = eos_g.fugacities_g
#                        else:
#                            lnphis_g = eos_g.lnphis_l
#                            fugacities_g = eos_g.fugacities_l
#                    else:
#                        try:
#                            lnphis_g = eos_g.lnphis_g#fugacity_coefficients(eos_g.Z_g, ys)
#                            fugacities_g = eos_g.fugacities_g
#                        except AttributeError:
#                            lnphis_g = eos_g.lnphis_l#fugacity_coefficients(eos_g.Z_l, ys)
#                            fugacities_g = eos_g.fugacities_l
#                    
#                    if hasattr(eos_l, 'lnphis_g') and hasattr(eos_l, 'lnphis_l'):
#                        if eos_l.G_dep_g < eos_l.lnphis_l:
#                            lnphis_l = eos_l.lnphis_g
#                            fugacities_l = eos_l.fugacities_g
#                        else:
#                            lnphis_l = eos_l.lnphis_l
#                            fugacities_l = eos_l.fugacities_l
#                    else:
#                        try:
#                            lnphis_l = eos_l.lnphis_g#fugacity_coefficients(eos_l.Z_g, ys)
#                            fugacities_l = eos_l.fugacities_g
#                        except AttributeError:
#                            lnphis_l = eos_l.lnphis_l#fugacity_coefficients(eos_l.Z_l, ys)
#                            fugacities_l = eos_l.fugacities_l
#                    
#                else:
#            print(phis_l, phis_g, 'phis')
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
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks, guess=V_over_F)
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
            
            err3 = 0.0
            # Suggested tolerance 1e-15
            for Ki, xi, yi in zip(Ks, xs, ys):
                # equivalent of fugacity ratio
                # Could divide by the old Ks as well.
                err_i = Ki*xi/yi - 1.0
                err3 += err_i*err_i
                # or use absolute for tolerance...
            
#            err2 = sum([(exp(l-g)-1.0)**2  ]) 
#            err2 = 0.0
#            for l, g in zip(fugacities_l, fugacities_g):
#                err_i = (l/g-1.0)
#                err2 += err_i*err_i
           # Suggested tolerance 1e-15
            # This is a better metric because it does not involve  hysterisis
#            print(err3, err2)
            
#            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
#                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
#            print(err, err2)
            xs, ys = xs_new, ys_new
#            print(i, 'err', err, err2, 'xs, ys', xs, ys, 'VF', V_over_F)
            if near_critical:
                comp_difference = sum([abs(xi - yi) for xi, yi in zip(xs, ys)])
                if comp_difference < trivial_solution_tol:
                    raise ValueError("Converged to trivial condition, compositions of both phases equal")
#            print(xs)
            if err3 < xtol:
                break
            if i == maxiter-1:
                raise ValueError('End of SS without convergence')
        
        if info is not None:
            info[:] = (i, err3)
        return V_over_F, xs, ys, eos_l, eos_g

    def stabiliy_iteration_Michelsen(self, T, P, zs, Ks_initial=None, 
                                     maxiter=20, xtol=1E-12, liq=True):
        # checks stability vs. the current zs, mole fractions
        # liq: whether adding a test liquid phase to see if is stable or not
        
        eos_ref = self#.to_TP_zs(T=T, P=P, zs=zs)
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
#            if liq:
#                print(zs_test_normalized, sum_zs_test)
            
#            to_TP_zs_fast(self, T, P, zs, only_l=False, only_g=False)

            # IT IS NOT PERMISSIBLE TO DO ONLY ONE ROOT! 2019-03-20
            # Breaks lots of stabilities.
            eos_test = self.to_TP_zs_fast(T=T, P=P, zs=zs_test_normalized, only_l=False, only_g=False, full_alphas=False)
            fugacities_test, fugacities_phase = eos_test.eos_fugacities_lowest_Gibbs()
            
            if fugacities_ref_phase == fugacities_phase:
                same_phase_count += 1.0
            else:
                same_phase_count = 0
            
#            if liq:
#                print(fugacities_test, fugacities_ref_phase, fugacities_phase)
            
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
#            elif same_phase_count > 5:
#                break
            # It is possible to break if the trivial solution is being approached here also
            if _ == maxiter-1 and fugacities_ref_phase != fugacities_phase:
                raise UnconvergedError('End of stabiliy_iteration_Michelsen without convergence')

        # Fails directly if fugacities_ref_phase == fugacities_phase
        # Fugacity error:
        # no, the fugacities are not supposed to be equal
#        err_equifugacity = 0
#        for fi, fref in zip(fugacities_test, fugacities_ref):
#            err_equifugacity += abs(fi - fref)
#        if err_equifugacity/P > 1e-3:
#            sum_zs_test = 1
        
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
#        print(Ks_l, Ks_g, 'Ks_l, Ks_g')
                    
        # Table 4.6 Summary of Possible Phase Stability Test Results, 
        # Phase Behavior, Whitson and Brule
        # There is a typo where Sl appears in the vapor column; this should be
        # liquid; as shown in https://www.e-education.psu.edu/png520/m17_p7.html
        g_pass, l_pass = False, False # pass means this phase cannot form another phase
        
        if phase_failure_g:
            g_pass = True
        if phase_failure_l:
            l_pass = True
        if trivial_g:
            g_pass = True
        if trivial_l:
            l_pass = True
        if sum_g_criteria < stable_criteria:
            g_pass = True
        if sum_l_criteria < stable_criteria:
            l_pass = True
#        print(l_pass, g_pass, 'l, g test show stable')
            

        
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
#        print('stable', stable, 'phase_failure_g', phase_failure_g, 'phase_failure_l', phase_failure_l,
#              'sum_g_criteria', sum_g_criteria, 'sum_l_criteria', sum_l_criteria,
#              'trivial_g', trivial_g, 'trivial_l', trivial_l)
            
            
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
    
    def _fugacity_sum_terms(self):
        zs = self.zs
        cmps = self.cmps
        a_alpha_ijs = self.a_alpha_ijs
        fugacity_sum_terms = []
        for i in cmps:
            l = a_alpha_ijs[i]
            sum_term = 0.0
            for j in cmps:
                sum_term += zs[j]*l[j]
            fugacity_sum_terms.append(sum_term)
        self.fugacity_sum_terms = fugacity_sum_terms
        return fugacity_sum_terms

    def _da_alpha_dT_j_rows(self):
        zs = self.zs
        cmps = self.cmps
        da_alpha_dT_ijs = self.da_alpha_dT_ijs
        da_alpha_dT_j_rows = []
        for i in cmps:
            l = da_alpha_dT_ijs[i]
            sum_term = 0.0
            for j in cmps:
                sum_term += zs[j]*l[j]
            da_alpha_dT_j_rows.append(sum_term)
        self.da_alpha_dT_j_rows = da_alpha_dT_j_rows
        return da_alpha_dT_j_rows
    
    def _d2a_alpha_dT2_j_rows(self):
        # Does not seem to have worked
        a_alpha, da_alpha_dT, d2a_alpha_dT2, da_alpha_dT_ijs, a_alpha_ijs, d2a_alpha_dT2_ijs = a_alpha_and_derivatives_full(
                    self.a_alphas, self.da_alpha_dTs, self.d2a_alpha_dT2s, self.T, self.zs, self.kijs,
                     self.a_alpha_ijs, self.a_alpha_i_roots, self.a_alpha_ij_roots_inv, second_matrix=True)

        zs = self.zs
        cmps = self.cmps
        d2a_alpha_dT2_j_rows = []
        for i in cmps:
            l = d2a_alpha_dT2_ijs[i]
            sum_term = 0.0
            for j in cmps:
                sum_term += zs[j]*l[j]
            d2a_alpha_dT2_j_rows.append(sum_term)
        self.d2a_alpha_dT2_j_rows = d2a_alpha_dT2_j_rows
        return d2a_alpha_dT2_j_rows


    
    @property
    def db_dzs(self):   
        r'''Helper method for calculating the composition derivatives of `b`.
        Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial b}{\partial x_i}\right)_{T, P, x_{i\ne j}} 
            = b_i

        Returns
        -------
        db_dzs : list[float]
            Composition derivative of `b` of each component, [m^3/mol]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return self.bs

    @property
    def db_dns(self):   
        r'''Helper method for calculating the mole number derivatives of `b`.
        Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial b}{\partial n_i}\right)_{T, P, n_{i\ne j}} 
            = b_i - b

        Returns
        -------
        db_dns : list[float]
            Composition derivative of `b` of each component, [m^3/mol^2]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        b = self.b
        return [bi - b for bi in self.bs]
    
    @property
    def dnb_dns(self):
        r'''Helper method for calculating the partial molar derivative of `b`.
        Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial n \cdot b}{\partial n_i}\right)_{T, P, n_{i\ne j}} 
            = b_i 

        Returns
        -------
        dnb_dns : list[float]
            Partial molar derivative of `b` of each component, [m^3/mol]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return self.bs

    @property
    def d2b_dzizjs(self):
        return [[0.0]*self.N for i in self.cmps]

    @property
    def d2b_dninjs(self):   
        r'''Helper method for calculating the second partial mole number
        derivatives of `b`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 b}{\partial n_i \partial n_j}\right)_{T, P, 
            n_{k\ne i,k}} = 2b - b_i - b_j

        Returns
        -------
        d2b_dninjs : list[list[float]]
            Second Composition derivative of `b` of each component, 
            [m^3/mol^3]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        bb = 2.0*self.b
        bs = self.bs
        cmps = self.cmps
        d2b_dninjs = []
        for bi in self.bs:
            d2b_dninjs.append([bb - bi - bj for bj in bs])
        return d2b_dninjs

    @property
    def d3b_dzizjzks(self):   
        # All zeros
        return 0.0

    @property
    def d3b_dninjnks(self):   
        r'''Helper method for calculating the third partial mole number
        derivatives of `b`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^3 b}{\partial n_i \partial n_j \partial n_k }
            \right)_{T, P, 
            n_{m \ne i,j,k}} = 2(-3b + b_i + b_j + b_k)

        Returns
        -------
        d2b_dninjs : list[list[list[float]]]
            Second Composition derivative of `b` of each component, 
            [m^3/mol^4]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        m3b = -3.0*self.b
        bs = self.bs
        cmps = self.cmps
        d3b_dninjnks = []
        for bi in bs:
            d3b_dnjnks = []
            for bj in bs:
                d3b_dnjnks.append([2.0*(m3b + bi + bj + bk) for bk in bs])
            d3b_dninjnks.append(d3b_dnjnks)
        return d3b_dninjnks


    @property
    def da_alpha_dzs(self):   
        r'''Helper method for calculating the composition derivatives of
        `a_alpha`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial a \alpha}{\partial x_i}\right)_{T, P, x_{i\ne j}} 
            = 2 \cdot \sum_j z_{j} (1 - k_{ij}) \sqrt{ (a \alpha)_i (a \alpha)_j}

        Returns
        -------
        da_alpha_dzs : list[float]
            Composition derivative of `alpha` of each component, 
            [kg*m^5/(mol^2*s^2)]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            fugacity_sum_terms = self.fugacity_sum_terms
        except:
            fugacity_sum_terms = self._fugacity_sum_terms()
        return [i + i for i in fugacity_sum_terms]

    @property
    def da_alpha_dns(self):   
        r'''Helper method for calculating the mole number derivatives of
        `a_alpha`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial a \alpha}{\partial n_i}\right)_{T, P, n_{i\ne j}} 
            = 2 (-a\alpha + \sum_j z_{j} (1 - k_{ij}) \sqrt{ (a \alpha)_i (a \alpha)_j})

        Returns
        -------
        da_alpha_dns : list[float]
            Mole number derivative of `alpha` of each component, 
            [kg*m^5/(mol^3*s^2)]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            fugacity_sum_terms = self.fugacity_sum_terms
        except:
            fugacity_sum_terms = self._fugacity_sum_terms()
        a_alpha = self.a_alpha
        return [2.0*(t - a_alpha) for t in fugacity_sum_terms]

    @property
    def dna_alpha_dns(self):   
        r'''Helper method for calculating the partial molar derivatives of
        `a_alpha`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial a \alpha}{\partial n_i}\right)_{T, P, n_{i\ne j}} 
            = 2 (-0.5 a\alpha + \sum_j z_{j} (1 - k_{ij}) \sqrt{ (a \alpha)_i (a \alpha)_j})

        Returns
        -------
        dna_alpha_dns : list[float]
            Partial molar derivative of `alpha` of each component, 
            [kg*m^5/(mol^2*s^2)]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            fugacity_sum_terms = self.fugacity_sum_terms
        except:
            fugacity_sum_terms = self._fugacity_sum_terms()
        a_alpha = self.a_alpha
        return [t + t - a_alpha for t in fugacity_sum_terms]

    @property
    def d2a_alpha_dzizjs(self):   
        r'''Helper method for calculating the second composition derivatives of
        `a_alpha` (hessian). Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 a \alpha}{\partial x_i x_j}\right)_{T, P, x_{k\ne i,j}} 
            = 2 (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

        Returns
        -------
        d2a_alpha_dzizjs : list[float]
            Second composition derivative of `alpha` of each component, 
            [kg*m^5/(mol^2*s^2)]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        a_alpha_ijs = self.a_alpha_ijs
        return [[i+i for i in row] for row in a_alpha_ijs]

    @property
    def d2a_alpha_dninjs(self):   
        r'''Helper method for calculating the second partial molar derivatives 
        of `a_alpha` (hessian). Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 a \alpha}{\partial n_i \partial n_j }\right)_{T, P, n_{k\ne i,j}} 
            = 2\left[3(a \alpha) + (a\alpha)_{ij}  -2 (\text{term}_{i,j})
            \right]
            
        .. math::
            \text{term}_{i,j} = \sum_k z_k\left((a\alpha)_{ik} + (a\alpha)_{jk}
            \right)
            
        .. math::
            (a\alpha)_{ij} = (1-k_{ij})\sqrt{(a\alpha)_{i}(a\alpha)_{j}}

            
        Returns
        -------
        d2a_alpha_dninjs : list[float]
            Second partial molar derivative of `alpha` of each component, 
            [kg*m^5/(mol^3*s^2)]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            fugacity_sum_terms = self.fugacity_sum_terms
        except:
            fugacity_sum_terms = self._fugacity_sum_terms()
        a_alpha = self.a_alpha
        a_alpha_ijs = self.a_alpha_ijs
        cmps = self.cmps
        zs = self.zs
        a_alpha3 = 3.0*a_alpha
        
        hessian = []
        for i in cmps:
            row = []
            for j in cmps:
                if i == j:
                    term = 2.0*fugacity_sum_terms[i]
                else:
                    term = 0.0
                    for k in cmps:
                        term += zs[k]*(a_alpha_ijs[i][k] + a_alpha_ijs[j][k])
                row.append(2.0*(a_alpha3 + a_alpha_ijs[i][j] -2.0*term))
            hessian.append(row)
        return hessian

    @property
    def da_alpha_dT_dzs(self):   
        r'''Helper method for calculating the composition derivatives of
        `da_alpha_dT`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 a \alpha}{\partial x_i \partial T}
            \right)_{P, x_{i\ne j}} 
            = 2 \sum_j -z_{j} (k_{ij} - 1) (a \alpha)_i (a \alpha)_j
            \frac{\partial (a \alpha)_i}{\partial T} \frac{\partial (a \alpha)_j}{\partial T}
            \left({ (a \alpha)_i (a \alpha)_j}\right)^{-0.5}

        Returns
        -------
        da_alpha_dT_dzs : list[float]
            Composition derivative of `da_alpha_dT` of each component, 
            [kg*m^5/(mol^2*s^2*K)]
        
        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            da_alpha_dT_j_rows = self.da_alpha_dT_j_rows
        except:
            da_alpha_dT_j_rows = self._da_alpha_dT_j_rows()
        return [i + i for i in da_alpha_dT_j_rows]

    @property
    def da_alpha_dT_dns(self):   
        r'''Helper method for calculating the mole number derivatives of
        `da_alpha_dT`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 a \alpha}{\partial n_i \partial T}
            \right)_{P, n_{i\ne j}} 
            = 2 \left[\sum_j -z_{j} (k_{ij} - 1) (a \alpha)_i (a \alpha)_j
            \frac{\partial (a \alpha)_i}{\partial T} \frac{\partial (a \alpha)_j}{\partial T}
            \left({ (a \alpha)_i (a \alpha)_j}\right)^{-0.5} 
             - \frac{\partial a \alpha}{\partial T} \right]

        Returns
        -------
        da_alpha_dT_dzs : list[float]
            Composition derivative of `da_alpha_dT` of each component, 
            [kg*m^5/(mol^2*s^2*K)]
        
        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            da_alpha_dT_j_rows = self.da_alpha_dT_j_rows
        except:
            da_alpha_dT_j_rows = self._da_alpha_dT_j_rows()
        da_alpha_dT = self.da_alpha_dT
        return [2.0*(t - da_alpha_dT) for t in da_alpha_dT_j_rows]

    @property
    def dna_alpha_dT_dns(self):   
        r'''Helper method for calculating the mole number derivatives of
        `da_alpha_dT`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 n a \alpha}{\partial n_i \partial T}
            \right)_{P, n_{i\ne j}} 
            = 2 \left[\sum_j -z_{j} (k_{ij} - 1) (a \alpha)_i (a \alpha)_j
            \frac{\partial (a \alpha)_i}{\partial T} \frac{\partial (a \alpha)_j}{\partial T}
            \left({ (a \alpha)_i (a \alpha)_j}\right)^{-0.5} 
             - 0.5 \frac{\partial a \alpha}{\partial T} \right]

        Returns
        -------
        dna_alpha_dT_dns : list[float]
            Composition derivative of `da_alpha_dT` of each component, 
            [kg*m^5/(mol*s^2*K)]
        
        Notes
        -----
        This derivative is checked numerically.
        '''
        try:
            da_alpha_dT_j_rows = self.da_alpha_dT_j_rows
        except:
            da_alpha_dT_j_rows = self._da_alpha_dT_j_rows()
        da_alpha_dT = self.da_alpha_dT
        return [t + t - da_alpha_dT for t in da_alpha_dT_j_rows]
    
    def dV_dzs(self, Z, zs):
        r'''Calculates the molar volume composition derivative
        (where the mole fractions do not sum to 1). Verified numerically. 
        Used in many other derivatives, and for the molar volume mole number 
        derivative and partial molar volume calculation.
        
        .. math::
            \left(\frac{\partial V}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}  =
            \frac{- R T \left(V^{2}{\left(x \right)} + V{\left(x \right)} \delta{\left(x \right)} 
            + \epsilon{\left(x \right)}\right)^{3} \frac{d}{d x} b{\left(x \right)} + \left(V{\left(x \right)} 
            - b{\left(x \right)}\right)^{2} \left(V^{2}{\left(x \right)} + V{\left(x \right)} \delta{\left(x \right)} 
            + \epsilon{\left(x \right)}\right)^{2} \frac{d}{d x} \operatorname{a \alpha}{\left(x \right)} 
            - \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} V^{3}{\left(x \right)} \operatorname{a 
            \alpha}{\left(x \right)} \frac{d}{d x} \delta{\left(x \right)} - \left(V{\left(x \right)} - b{\left(x
            \right)}\right)^{2} V^{2}{\left(x \right)} \operatorname{a \alpha}{\left(x \right)} \delta{\left(x 
            \right)} \frac{d}{d x} \delta{\left(x \right)} - \left(V{\left(x \right)} - b{\left(x \right)}
            \right)^{2} V^{2}{\left(x \right)} \operatorname{a \alpha}{\left(x \right)} \frac{d}{d x} \epsilon{
            \left(x \right)} - \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} V{\left(x \right)}
            \operatorname{a \alpha}{\left(x \right)} \delta{\left(x \right)} \frac{d}{d x} \epsilon{\left(x 
            \right)} - \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} V{\left(x \right)} \operatorname{a
            \alpha}{\left(x \right)} \epsilon{\left(x \right)} \frac{d}{d x} \delta{\left(x \right)} 
            - \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} \operatorname{a \alpha}{\left(x \right)}
            \epsilon{\left(x \right)} \frac{d}{d x} \epsilon{\left(x \right)}}{- R T \left(V^{2}{\left(x \right)}
            + V{\left(x \right)} \delta{\left(x \right)} + \epsilon{\left(x \right)}\right)^{3} 
            + 2 \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} V^{3}{\left(x \right)} 
            \operatorname{a \alpha}{\left(x \right)} + 3 \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2}
            V^{2}{\left(x \right)} \operatorname{a \alpha}{\left(x \right)} \delta{\left(x \right)} 
            + \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} V{\left(x \right)} \operatorname{a 
            \alpha}{\left(x \right)} \delta^{2}{\left(x \right)} + 2 \left(V{\left(x \right)} - b{\left(x 
            \right)}\right)^{2} V{\left(x \right)} \operatorname{a \alpha}{\left(x \right)} \epsilon{\left(x
            \right)} + \left(V{\left(x \right)} - b{\left(x \right)}\right)^{2} \operatorname{a \alpha}{\left(x
            \right)} \delta{\left(x \right)} \epsilon{\left(x \right)}}
            
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        dV_dzs : float
            Molar volume composition derivatives, [m^3/mol]
            
        Notes
        -----
        The derivation for the derivative is performed as follows using SymPy.
        The function source code is an optimized variant created with the `cse`
        SymPy function, and hand optimized further.
        
        >>> from sympy import * # doctest:+SKIP
        >>> P, T, R, x = symbols('P, T, R, x') # doctest:+SKIP
        >>> V, delta, epsilon, a_alpha, b = symbols('V, delta, epsilon, a\ \\alpha, b', cls=Function) # doctest:+SKIP
        >>> CUBIC = R*T/(V(x) - b(x)) - a_alpha(x)/(V(x)*V(x) + delta(x)*V(x) + epsilon(x)) - P # doctest:+SKIP
        >>> solve(diff(CUBIC, x), Derivative(V(x), x)) # doctest:+SKIP
        [(-R*T*(V(x)**2 + V(x)*delta(x) + epsilon(x))**3*Derivative(b(x), x) + (V(x) - b(x))**2*(V(x)**2 + V(x)*delta(x) + epsilon(x))**2*Derivative(a \alpha(x), x) - (V(x) - b(x))**2*V(x)**3*a \alpha(x)*Derivative(delta(x), x) - (V(x) - b(x))**2*V(x)**2*a \alpha(x)*delta(x)*Derivative(delta(x), x) - (V(x) - b(x))**2*V(x)**2*a \alpha(x)*Derivative(epsilon(x), x) - (V(x) - b(x))**2*V(x)*a \alpha(x)*delta(x)*Derivative(epsilon(x), x) - (V(x) - b(x))**2*V(x)*a \alpha(x)*epsilon(x)*Derivative(delta(x), x) - (V(x) - b(x))**2*a \alpha(x)*epsilon(x)*Derivative(epsilon(x), x))/(-R*T*(V(x)**2 + V(x)*delta(x) + epsilon(x))**3 + 2*(V(x) - b(x))**2*V(x)**3*a \alpha(x) + 3*(V(x) - b(x))**2*V(x)**2*a \alpha(x)*delta(x) + (V(x) - b(x))**2*V(x)*a \alpha(x)*delta(x)**2 + 2*(V(x) - b(x))**2*V(x)*a \alpha(x)*epsilon(x) + (V(x) - b(x))**2*a \alpha(x)*delta(x)*epsilon(x))]
        '''
        T = self.T
        RT = R*T
        V = Z*RT/self.P
        ddelta_dzs = self.ddelta_dzs
        depsilon_dzs = self.depsilon_dzs
        db_dzs = self.db_dzs
        da_alpha_dzs = self.da_alpha_dzs

        x0 = self.delta
        x1 = a_alpha = self.a_alpha
        x2 = epsilon = self.epsilon
        b = self.b

        x0V = x0*V
        Vmb = V - b
        x5 = Vmb*Vmb
        x1x5 = x1*x5
        x0x1x5 = x0*x1x5
        t0 = V*x1x5
        x6 = x2*x1x5
        x9 = V*V
        x7 = x9*t0
        x8 = x2*t0
        x10 = x0V + x2 + x9
        x10x10 = x10*x10
        x11 = R*T*x10*x10x10
        x13 = x0x1x5*x9
        x7x8 = x7 + x8
        
        t2 = -1.0/(x0V*x0x1x5 + x0*x6 - x11 + 3.0*x13 + x7x8 + x7x8)
        t1 = t2*x10x10*x5
        t3 = x0V*x1x5
        t4 = x1x5*x9
        t5 = t2*(t3 + t4 + x6)
        t6 = t2*(x13 + x7x8)
        x11t2 = x11*t2
        
        return [t5*depsilon_dzs[i] - t1*da_alpha_dzs[i] + x11t2*db_dzs[i] + t6*ddelta_dzs[i]
                for i in self.cmps]

    def dV_dns(self, Z, zs):
        r'''Calculates the molar volume mole number derivatives
        (where the mole fractions sum to 1). No specific formula is implemented
        for this property - it is calculated from the mole fraction derivative.
        
        .. math::
            \left(\frac{\partial V}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left( \left(\frac{\partial V}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)
            
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        dV_dns : float
            Molar volume mole number derivatives, [m^3/mol^2]
        '''
        return dxs_to_dns(self.dV_dzs(Z, zs), zs)

    def dnV_dns(self, Z, zs):
        r'''Calculates the partial molar volume of the specified phase
        No specific formula is implemented
        for this property - it is calculated from the molar
        volume mole fraction derivative.
        
        .. math::
            \left(\frac{\partial n V}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left( \left(\frac{\partial V}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)
            
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        dnV_dns : float
            Partial molar volume of the mixture of the specified phase,
            [m^3/mol]
        '''
        V = Z*R*self.T/self.P
        return dxs_to_dn_partials(self.dV_dzs(Z, zs), zs, V)

    def d2V_dzizjs(self, Z, zs):
        V = Z*self.T*R/self.P
        dV_dzs = self.dV_dzs(Z, zs)
        
        depsilon_dzs = self.depsilon_dzs
        d2epsilon_dzizjs = self.d2epsilon_dzizjs
        
        ddelta_dzs = self.ddelta_dzs
        d2delta_dzizjs = self.d2delta_dzizjs

        db_dzs = self.db_dzs
        d2bs = self.d2b_dzizjs
        da_alpha_dzs = self.da_alpha_dzs
        d2a_alpha_dzizjs = self.d2a_alpha_dzizjs
        
        return self._d2V_dij_wrapper(V=V, d_Vs=dV_dzs, dbs=db_dzs, d2bs=d2bs,
                                     d_epsilons=depsilon_dzs, d2_epsilons=d2epsilon_dzizjs,
                                     d_deltas=ddelta_dzs, d2_deltas=d2delta_dzizjs,
                                     da_alphas=da_alpha_dzs, d2a_alphas=d2a_alpha_dzizjs
                                     )

    def d2V_dninjs(self, Z, zs):
        V = Z*self.T*R/self.P
        dV_dns = self.dV_dns(Z, zs)
        
        depsilon_dns = self.depsilon_dns
        d2epsilon_dninjs = self.d2epsilon_dninjs
        
        ddelta_dns = self.ddelta_dns
        d2delta_dninjs = self.d2delta_dninjs

        db_dns = self.db_dns
        d2bs = self.d2b_dninjs
        da_alpha_dns = self.da_alpha_dns
        d2a_alpha_dninjs = self.d2a_alpha_dninjs
        
        return self._d2V_dij_wrapper(V=V, d_Vs=dV_dns, dbs=db_dns, d2bs=d2bs,
                                     d_epsilons=depsilon_dns, d2_epsilons=d2epsilon_dninjs,
                                     d_deltas=ddelta_dns, d2_deltas=d2delta_dninjs,
                                     da_alphas=da_alpha_dns, d2a_alphas=d2a_alpha_dninjs
                                     )

    def _d2V_dij_wrapper(self, V, d_Vs, dbs, d2bs, d_epsilons, d2_epsilons,
                         d_deltas, d2_deltas, da_alphas, d2a_alphas):
        T = self.T
        cmps = self.cmps

        x0 = V
        x3 = self.b
        x4 = x0 - x3
        x5 = self.epsilon
        x6 = x0**2
        x7 = self.delta
        x8 = x0*x7
        x9 = x5 + x6 + x8
        x10 = self.a_alpha
        x11 = x10*x4**2
        x12 = 2*x0
        x13 = x9**2
        x14 = R*T
        x17 = x4**3
        x18 = x10*x17
        x19 = 2*x18
        x22 = 4*x18
        x27 = x12*x18
        x33 = x14*x9**3
        x34 = 2*x33
        x37 = x19*x8
        x38 = x17*x9
        x39 = x10*x38
        
        hessian = []
        for i in cmps:

            
            row = []
            for j in cmps:
                
                # TODO optimize this
                x15 = d_epsilons[i]
                x16 = d_epsilons[j]
                x20 = x16*x19
                
                x21 = d_Vs[i]
                x24 = d_Vs[j]
                
                x23 = x21*x22
                x25 = x15*x24
                x26 = d_deltas[i]
                x28 = d_deltas[j]
                x29 = x21*x24
                x30 = 8*x18*x29
                x31 = x28*x6
                
                x32 = x24*x26
                x35 = x34*dbs[j]
                x36 = dbs[i]
                x40 = x38*da_alphas[i]
                x41 = x38*da_alphas[j]
                x42 = x21*x41
                x43 = x24*x40
                x44 = x21*x39
                
                
                
                d1 = d2_deltas[i][j] # Derivative(x7, x1, x2)
                d2 = d2a_alphas[i][j] # Derivative(x10, x1, x2)
                d3 = d2bs[i][j] # Derivative(x3, x1, x2)
                d4 = d2_epsilons[i][j] # Derivative(x5, x1, x2)
                
                v = ((x0*x16*x23 + x0*x22*x25 - x0*x26*x41 - x0*x28*x40 
                      - x0*x39*d1 - x12*x42 - x12*x43 + x13*x17*d2 + x15*x20
                      + x15*x27*x28 - x15*x41 + x16*x26*x27 - x16*x40 
                      + x19*x25*x7 + x19*x26*x31 + x19*x29*x7**2 + x20*x21*x7 
                      + x21*x28*x37 + x21*x35 + x22*x32*x6 + x23*x31 
                      + x24*x34*x36 - 2*x24*x44 - x28*x44 - x29*x34 + x30*x6 
                      + x30*x8 + x32*x37 - x32*x39 - x33*x4*d3 - x35*x36 
                      - x39*d4 - x42*x7 - x43*x7)/(x4*x9*(x11*x12 + x11*x7 - x13*x14)))
                row.append(v)
                
            hessian.append(row)
        return hessian
                


    def _d2V_dij_wrapper_broken(self, V, d_Vs, dbs, d2bs, d_epsilons, d2_epsilons,
                         d_deltas, d2_deltas, da_alphas, d2a_alphas):
        # TODO remove
        T = self.T
        cmps = self.cmps
        x0 = V
        x1 = self.b
        x2 = x0 - x1
        x3 = self.epsilon
        x4 = self.delta
        x5 = x0**2 + x0*x4 + x3
        x6 = self.a_alpha
        x7 = x2**2*x6
        x8 = 2*x0
        x9 = x5**2
        x10 = R*T
        x11 = x2**3
        x12 = x11*x6
        x13 = x12*x5
        x14 = x10*x5**3
        x16 = 2*x12
        x17 = x16*x5

        hessian = []
        for i in cmps:
            x15 = d_Vs[i]
            x23 = x15*x4
            x18 = d_epsilons[i]
            x19 = x11*x5*da_alphas[i]
            x20 = 2*x19
            x21 = d_deltas[i]
            x22 = x0*x21
            
            x50 = dbs[i]
            
            row = []
            
            for j in cmps:
                d1 = d2_deltas[j][i] # Derivative(x4, (x, 2))
                d2 = d2a_alphas[j][i] # Derivative(x6, (x, 2))
                d3 = d2_epsilons[j][i] # Derivative(x3, (x, 2))
                d4 = d2bs[j][i] #Derivative(x1, (x, 2))
                
                v = (-(x0*x13*d1 + 4*x0*x15*x19 - x11*x9*d2 + x13*d3 + x14*x2*d4
                        + 2*x14*(x15 - x50)**2 + x15**2*x17 + x15*x17*x21 
                        - x16*(x15*x8 + x18 + x22 + x23)**2 + x18*x20 + x20*x22 + x20*x23)
                        /(x2*x5*(-x10*x9 + x4*x7 + x7*x8)))
                row.append(v)
            hessian.append(row)
        return hessian
        
        
                


    def dZ_dzs(self, Z, zs):
        r'''Calculates the compressibility composition derivatives
        (where the mole fractions do not sum to 1). No specific formula is 
        implemented for this property - it is calculated from the 
        composition derivative of molar volume, which does have its formula
        implemented.
        
        .. math::
            \left(\frac{\partial Z}{\partial x_i}\right)_{T, P,
            x_{i\ne j}} = \frac{P }{RT} 
            \left(\frac{\partial V}{\partial x_i}\right)_{T, P, x_{i\ne j}}
            
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        dZ_dzs : float
            Compressibility composition derivative, [-]
        '''
        factor = self.P/(self.T*R)
        return [dV*factor for dV in self.dV_dzs(Z, zs)]

    def dZ_dns(self, Z, zs):
        r'''Calculates the compressibility mole number derivatives
        (where the mole fractions sum to 1). No specific formula is implemented
        for this property - it is calculated from the mole fraction derivative.
        
        .. math::
            \left(\frac{\partial Z}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left( \left(\frac{\partial Z}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)
            
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        dZ_dns : float
            Compressibility number derivatives, [1/mol]
        '''
        return dxs_to_dns(self.dZ_dzs(Z, zs), zs)

    def dnZ_dns(self, Z, zs):
        r'''Calculates the partial compressibility of the specified phase
        No specific formula is implemented
        for this property - it is calculated from the compressibility
        mole fraction derivative.
        
        .. math::
            \left(\frac{\partial n Z}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left( \left(\frac{\partial Z}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)
            
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        dnZ_dns : float
            Partial compressibility of the mixture of the specified phase,
            [-]
        '''
        return dxs_to_dn_partials(self.dZ_dzs(Z, zs), zs, Z)
    
    def dH_dep_dzs(self, Z, zs):
        r'''Calculates the molar departure enthalpy composition derivative
        (where the mole fractions do not sum to 1). Verified numerically. 
        Useful in solving for enthalpy specifications in newton-type methods,
        and forms the basis for the molar departure enthalpy mole number
        derivative and molar partial departure enthalpy.
        
        .. math::
            \left(\frac{\partial H_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}  =
            P \frac{d}{d x} V{\left(x \right)} + \frac{2 \left(T \frac{\partial}{\partial T}
            \operatorname{a \alpha}{\left(T,x \right)} - \operatorname{a \alpha}{\left(x
            \right)}\right) \left(- \delta{\left(x \right)} \frac{d}{d x} \delta{\left(x
            \right)} + 2 \frac{d}{d x} \epsilon{\left(x \right)}\right) \operatorname{atanh}
            {\left(\frac{2 V{\left(x \right)} + \delta{\left(x \right)}}{\sqrt{\delta^{2}
            {\left(x \right)} - 4 \epsilon{\left(x \right)}}} \right)}}{\left(\delta^{2}
            {\left(x \right)} - 4 \epsilon{\left(x \right)}\right)^{\frac{3}{2}}}
            + \frac{2 \left(T \frac{\partial}{\partial T} \operatorname{a \alpha}
            {\left(T,x \right)} - \operatorname{a \alpha}{\left(x \right)}\right) 
            \left(\frac{\left(- \delta{\left(x \right)} \frac{d}{d x} \delta{\left(x 
            \right)} + 2 \frac{d}{d x} \epsilon{\left(x \right)}\right) \left(2
            V{\left(x \right)} + \delta{\left(x \right)}\right)}{\left(\delta^{2}{\left(x
            \right)} - 4 \epsilon{\left(x \right)}\right)^{\frac{3}{2}}} + \frac{2 
            \frac{d}{d x} V{\left(x \right)} + \frac{d}{d x} \delta{\left(x \right)}}
            {\sqrt{\delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}}}\right)}{\left(
            - \frac{\left(2 V{\left(x \right)} + \delta{\left(x \right)}\right)^{2}}{
            \delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}} + 1\right) \sqrt{
            \delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}}} + \frac{2 
            \left(T \frac{\partial^{2}}{\partial x\partial T} \operatorname{a \alpha}
            {\left(T,x \right)} - \frac{d}{d x} \operatorname{a \alpha}{\left(x \right)}
            \right) \operatorname{atanh}{\left(\frac{2 V{\left(x \right)} + \delta{\left(x
            \right)}}{\sqrt{\delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}}}
            \right)}}{\sqrt{\delta^{2}{\left(x \right)} - 4 \epsilon{\left(x \right)}}}
            
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        dH_dep_dzs : float
            Departure enthalpy composition derivatives, [J/mol]
            
        Notes
        -----
        The derivation for the derivative is performed as follows using SymPy.
        The function source code is an optimized variant created with the `cse`
        SymPy function, and hand optimized further.
            
        >>> from sympy import * # doctest:+SKIP
        >>> P, T, V, R, b, a, delta, epsilon, x = symbols('P, T, V, R, b, a, delta, epsilon, x') # doctest:+SKIP
        >>> V, delta, epsilon, a_alpha, b = symbols('V, delta, epsilon, a_alpha, b', cls=Function) # doctest:+SKIP
        >>> H_dep = (P*V(x) - R*T + 2/sqrt(delta(x)**2 - 4*epsilon(x))*(T*Derivative(a_alpha(T, x), T) # doctest:+SKIP
        ... - a_alpha(x))*atanh((2*V(x)+delta(x))/sqrt(delta(x)**2-4*epsilon(x))))
        >>> diff(H_dep, x) # doctest:+SKIP
        P*Derivative(V(x), x) + 2*(T*Derivative(a \alpha(T, x), T) - a \alpha(x))*(-delta(x)*Derivative(delta(x), x) + 2*Derivative(epsilon(x), x))*atanh((2*V(x) + delta(x))/sqrt(delta(x)**2 - 4*epsilon(x)))/(delta(x)**2 - 4*epsilon(x))**(3/2) + 2*(T*Derivative(a \alpha(T, x), T) - a \alpha(x))*((-delta(x)*Derivative(delta(x), x) + 2*Derivative(epsilon(x), x))*(2*V(x) + delta(x))/(delta(x)**2 - 4*epsilon(x))**(3/2) + (2*Derivative(V(x), x) + Derivative(delta(x), x))/sqrt(delta(x)**2 - 4*epsilon(x)))/((-(2*V(x) + delta(x))**2/(delta(x)**2 - 4*epsilon(x)) + 1)*sqrt(delta(x)**2 - 4*epsilon(x))) + 2*(T*Derivative(a \alpha(T, x), T, x) - Derivative(a \alpha(x), x))*atanh((2*V(x) + delta(x))/sqrt(delta(x)**2 - 4*epsilon(x)))/sqrt(delta(x)**2 - 4*epsilon(x))
        '''
        P = self.P
        T = self.T
        ddelta_dzs = self.ddelta_dzs
        depsilon_dzs = self.depsilon_dzs
        da_alpha_dzs = self.da_alpha_dzs
        da_alpha_dT_dzs = self.da_alpha_dT_dzs
        dV_dzs = self.dV_dzs(Z, zs)
        
        x0 = V = Z*R*T/P
        x2 = self.delta
        x3 = x0 + x0 + x2
        x4 = self.epsilon
        x5 = x2*x2 - 4.0*x4
        try:
            x6 = x5**-0.5
        except:
            # VDW has x5 as zero as delta, epsilon = 0
            x6 = 1e50
        x7 = 2.0*catanh(x3*x6).real
        x8 = x9 = self.a_alpha
        
        x10 = T*self.da_alpha_dT - x8
        x13 = x6*x6# 1.0/x5

        t0 = x6*x7
        t1 = x10*t0*x13
        t2 = 2.0*x10*x13/(x13*x3*x3 - 1.0)
        dH_dzs = []
        for i in self.cmps:
            x1 = dV_dzs[i]
            x11 = ddelta_dzs[i]
            x12 = x11*x2 - 2.0*depsilon_dzs[i]
                
            value = (P*x1 - x12*t1 + t2*(x12*x13*x3 -x1 - x1 - x11)
                     + t0*(T*da_alpha_dT_dzs[i] - da_alpha_dzs[i]))
            dH_dzs.append(value)
        return dH_dzs

    def dH_dep_dns(self, Z, zs):
        r'''Calculates the molar departure enthalpy mole number derivatives
        (where the mole fractions sum to 1). No specific formula is implemented
        for this property - it is calculated from the mole fraction derivative.
        
        .. math::
            \left(\frac{\partial H_{dep}}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left(  \left(\frac{\partial H_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)
            
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        dH_dep_dns : float
            Departure enthalpy mole number derivatives, [J/mol^2]
        '''
        return dxs_to_dns(self.dH_dep_dzs(Z, zs), zs)

    def dnH_dep_dns(self, Z, zs):
        r'''Calculates the partial molar departure enthalpy. No specific 
        formula is implemented for this property - it is calculated from the
        mole fraction derivative.
        
        .. math::
            \left(\frac{\partial n H_{dep}}{\partial n_i}\right)_{T, P,
            n_{i\ne j}} = f\left(  \left(\frac{\partial H_{dep}}{\partial x_i}\right)_{T, P,
            x_{i\ne j}}
            \right)
            
        Parameters
        ----------
        Z : float
            Compressibility of the mixture for a desired phase, [-]
        zs : list[float], optional
            List of mole factions, either overall or in a specific phase, [-]
        
        Returns
        -------
        dnH_dep_dns : float
            Partial molar departure enthalpies of the phase, [J/mol]
        '''
        try:
            if Z == self.Z_l:
                F = self.H_dep_l
            else:
                F = self.H_dep_g
        except:
            F = self.H_dep_g
        return dxs_to_dn_partials(self.dH_dep_dzs(Z, zs), zs, F)

    def _lnphi_d_helper(self, Z, dbs, depsilons, ddelta, dVs, da_alphas):
        # Quite a bit of optimization remains here - only do once tested
        T = self.T
        P = self.P
        x3 = self.b
        x4 = self.delta
        x5 = self.epsilon
        RT = R*T
        x0 = V = Z*RT/P

        x2 = 1.0/(RT)
        x6 = x4*x4 - 4.0*x5
        x7 = x6**-0.5
        x8 = self.a_alpha
        x9 = x0 + x0
        x10 = x4 + x9
        x11 = x2 + x2
        x12 = x11*catanh(x10*x7).real
        x15 = 1.0/x6

        db_dns = dbs
        depsilon_dns = depsilons
        ddelta_dns = ddelta
        dV_dns = dVs
        da_alpha_dns = da_alphas
        
        t1 = P*x2
        t2 = x11*x15*x8/(x10*x10*x15 - 1.0) 
        t3 = x12*x8*x6**(-1.5)
        t4 = x12*x7
        t5 = 1.0/(x0 - x3)
        t6 = x4 + x9
        
        dfugacity_dns = []
        for i in self.cmps:
            x13 = ddelta_dns[i]
            x14 = x13*x4 - 2.0*depsilon_dns[i]
            x16 = x14*x15
            x1 = dV_dns[i]
            diff = (x1*t1 + t2*(x1 + x1 + x13 - x16*t6) + x14*t3 - t4*da_alpha_dns[i] - t5*(x1 - db_dns[i]))
            dfugacity_dns.append(diff)
        return dfugacity_dns

    def dlnphi_dzs(self, Z, zs):
        # Good to go - doc and test
        return self._lnphi_d_helper(Z, dbs=self.db_dzs, depsilons=self.depsilon_dzs, 
                                    ddelta=self.ddelta_dzs, dVs=self.dV_dzs(Z, zs),
                                    da_alphas=self.da_alpha_dzs)

    def dlnphi_dns(self, Z, zs):
        # Good to go - doc and test
        return self._lnphi_d_helper(Z, dbs=self.db_dns, depsilons=self.depsilon_dns, 
                                    ddelta=self.ddelta_dns, dVs=self.dV_dns(Z, zs),
                                    da_alphas=self.da_alpha_dns)
        
    def _d2lnphi_d2_helper(self, V, d_Vs, d2Vs, dbs, d2bs, d_epsilons, d2_epsilons,
                          d_deltas, d2_deltas, da_alphas, d2a_alphas):
        T, P = self.T, self.P
        cmps = self.cmps
        x0 = V
        hess = []
        
        x2 = 1/(R*T)
        x3 = self.b
        x4 = x0 - x3
        x7 = self.delta
        x8 = self.epsilon
        x11 = self.a_alpha
        for i in cmps:
            x5 = d_Vs[i]
            x17 = d_deltas[i]
            x18 = x17*x7 - 2*d_epsilons[i]
            x23 = da_alphas[i]
            bi = dbs[i]
            
            row = []
            for j in cmps:
                x6 = d_Vs[j]
                x9 = x7**2 - 4*x8
                x10 = 1/sqrt(x9)
                x12 = 2*x0
                x13 = x12 + x7
                x14 = catanh(x10*x13).real
                x15 = 2*x2
                x16 = x14*x15
                x19 = da_alphas[j]
                x20 = x16/x9**(3/2)
                x21 = d_deltas[j]
                x22 = x21*x7 - 2*d_epsilons[i]
                x24 = d2_deltas[i][j]
                x25 = x17*x21 + x24*x7 - 2*d2_epsilons[i][j]
                x26 = x11*x22
                x27 = 2*x5
                x28 = 1/x9
                x29 = x28*x7
                x30 = x18*x28
                x31 = x12*x30 - x17 + x18*x29 - x27
                x32 = x13**2*x28 - 1
                x33 = x15/x32
                x34 = x28*x33
                x35 = 2*x6
                x36 = x22*x28
                x37 = x12*x36 - x21 + x22*x29 - x35
                x38 = x9**(-2)
                x39 = x18*x38
                x40 = x11*x37
                x41 = x31*x38
                x42 = x22*x39
                
                x1 = d2Vs[i][j]
                v = (P*x1*x2 - x10*x16*d2a_alphas[i][j] + x11*x20*x25 - x11*x34*(-6*x0*x42
                     - 2*x1 + x12*x25*x28 + x17*x36 + x21*x30 - x24 + x25*x29 + x27*x36 + x30*x35 
                     - 3*x42*x7) - 4*x13*x2*x40*x41/x32**2 - 6*x14*x18*x2*x26/x9**(5/2)
                    + x18*x19*x20 - x19*x31*x34 + x20*x22*x23 - x23*x34*x37 + x26*x33*x41 + x33*x39*x40
                    - (x1 - d2bs[i][j])/x4 + (x5 - bi)*(x6 - dbs[j])/x4**2)
                row.append(v)
            hess.append(row)
        return hess
        
        
    def d2lnphi_dninjs(self, Z, zs):
        V = Z*self.T*R/self.P
        dV_dns = self.dV_dns(Z, zs)
        d2Vs = self.d2V_dninjs(Z, zs)
        
        depsilon_dns = self.depsilon_dns
        d2epsilon_dninjs = self.d2epsilon_dninjs
        
        ddelta_dns = self.ddelta_dns
        d2delta_dninjs = self.d2delta_dninjs

        db_dns = self.db_dns
        d2bs = self.d2b_dninjs
        da_alpha_dns = self.da_alpha_dns
        d2a_alpha_dninjs = self.d2a_alpha_dninjs
        return self._d2lnphi_d2_helper(V=V, d2Vs=d2Vs, d_Vs=dV_dns, dbs=db_dns, d2bs=d2bs,
                                     d_epsilons=depsilon_dns, d2_epsilons=d2epsilon_dninjs,
                                     d_deltas=ddelta_dns, d2_deltas=d2delta_dninjs,
                                     da_alphas=da_alpha_dns, d2a_alphas=d2a_alpha_dninjs)

    def d2lnphi_dzizjs(self, Z, zs):
        V = Z*self.T*R/self.P
        dV_dzs = self.dV_dzs(Z, zs)
        d2Vs = self.d2V_dzizjs(Z, zs)

        depsilon_dzs = self.depsilon_dzs
        d2epsilon_dzizjs = self.d2epsilon_dzizjs
        
        ddelta_dzs = self.ddelta_dzs
        d2delta_dzizjs = self.d2delta_dzizjs

        db_dzs = self.db_dzs
        d2bs = self.d2b_dzizjs
        da_alpha_dzs = self.da_alpha_dzs
        d2a_alpha_dzizjs = self.d2a_alpha_dzizjs
        return self._d2lnphi_d2_helper(V=V, d_Vs=dV_dzs, d2Vs=d2Vs, dbs=db_dzs, d2bs=d2bs,
                                     d_epsilons=depsilon_dzs, d2_epsilons=d2epsilon_dzizjs,
                                     d_deltas=ddelta_dzs, d2_deltas=d2delta_dzizjs,
                                     da_alphas=da_alpha_dzs, d2a_alphas=d2a_alpha_dzizjs)

    def fugacity_coefficients(self, Z, zs):
        try:
            if Z == self.Z_l:
                F = self.phi_l
            else:
                F = self.phi_g
        except:
            F = self.phi_g
        # This conversion seems numerically safe anyway
        return dns_to_dn_partials(self.dlnphi_dns(Z, zs), log(F))

    def d_main_derivatives_and_departures_dn(self, g=True):
        Z = self.Z_g if g else self.Z_l
        V = self.V_g if g else self.V_l
        
        T = self.T
        
        x0 = self.a_alpha
        x2 = self.epsilon
        x3 = V
        x4 = self.delta
        x5 = x2 + x3**2 + x3*x4
        x6 = 1/x5
        x7 = self.b
        x8 = x3 - x7
        x14 = x5**(-2)
        x15 = self.da_alpha_dT
        x16 = x14*x15
        x18 = 2*x3 + x4
        x23 = x5**(-3)
        x24 = 2*x23
        x27 = x18**2
        x28 = x18*x24
        
        

        da_alpha_dT_dns = self.da_alpha_dT_dns
        db_dns = self.db_dns
        ddelta_dns = self.ddelta_dns
        depsilon_dns = self.depsilon_dns
        da_alpha_dns = self.da_alpha_dns
        dV_dns = self.dV_dns(Z, self.zs)
        
        dndP_dT_dsn = []
        dndP_dV_dns = []
        dnd2P_dT2_dns = []
        dnd2P_dV2_dns = []
        dnd2P_dTdV_dns = []

        for i in self.cmps:
            x1 = da_alpha_dT_dns[i]
            x9 = dV_dns[i]
            x10 = R*(x9 - db_dns[i])
            x17 = 2*x10/x8**3
            x12 = 2*x9
    
            x11 = ddelta_dns[i]
            x21 = x11 + x12
            x22 = x0*x21
            
            x13 = x11*x3 + x12*x3 + x4*x9 + depsilon_dns[i]
            x25 = x0*x13
            x26 = x24*x25
    
            x19 = da_alpha_dns[i]
            x20 = x14*x19
            
            dndP_dT = -x1*x6 - x10/x8**2 + x13*x16
            dndP_dT_dsn.append(dndP_dT)
            
            dndP_dV = T*x17 + x14*x22 + x18*x20 - x18*x26
            dndP_dV_dns.append(dndP_dV)
            
            d2a_alpha_dT2_dn = 0 # TODO, vector
            dnd2P_dT2 = x6*(x13*x6*self.d2a_alpha_dT2 - d2a_alpha_dT2_dn)
            dnd2P_dT2_dns.append(dnd2P_dT2)
            
            dnd2P_dV2 = -6*T*x10/x8**4 - 2*x19*x23*x27 + 2*x20 - 2*x22*x28 + 6*x25*x27/x5**4 - 2*x26
            dnd2P_dV2_dns.append(dnd2P_dV2)
            
            dnd2P_dTdV = x1*x14*x18 - x13*x15*x28 + x16*x21 + x17
            dnd2P_dTdV_dns.append(dnd2P_dTdV)
            
        return dndP_dT_dsn, dndP_dV_dns, dnd2P_dT2_dns, dnd2P_dV2_dns, dnd2P_dTdV_dns


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
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optoinal
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

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
    
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
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

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def fast_init_specific(self, other):
        self.kappas = other.kappas
        b = 0.0
        for bi, zi in zip(self.bs, self.zs):
            b += bi*zi
        self.b = b
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
        a_alpha = self.a_alpha
        cmps = self.cmps
        a_alpha_ijs = self.a_alpha_ijs
        T_inv = 1.0/self.T
        bs, b = self.bs, self.b
        A = a_alpha*self.P*R2_inv*T_inv*T_inv
        B = b*self.P*T_inv*R_inv
        b_inv = 1.0/b
        
        # The two log terms need to use a complex log; typically these are
        # calculated at "liquid" volume solutions which are unstable
        # and cannot exist
        try:
            x0 = log(Z - B)
        except ValueError:
            # less than zero
            x0 = 0.0
            
        Zm1 = Z - 1.0
        x1 = 2./a_alpha
        x2 = A/(two_root_two*B)
        t0 = (Z + root_two_p1*B)/(Z - root_two_m1*B)
        try:
            x3 = log(t0)
        except ValueError:
            # less than zero
            x3 = 0.0
        x4 = x2*x3

        phis = []
        fugacity_sum_terms = []
        for i in cmps:
            a_alpha_js = a_alpha_ijs[i]
            b_ratio = bs[i]*b_inv 
            
            sum_term = 0.0
            for zi, a_alpha_j_i in zip(zs, a_alpha_js):
                sum_term += zi*a_alpha_j_i
#            sum_term = sum([zs[j]*a_alpha_js[j] for j in cmps])

            t3 = b_ratio*Zm1 - x0 - x4*(x1*sum_term - b_ratio)

            # Let wherever calls the exp deal with overflow
#            if t3 > 700.0:
#                t3 = 700.0
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

            der_sum = 0.0
            da_alpha_dT_ijs_i = da_alpha_dT_ijs[i]
            for j in cmps:
                der_sum += zs[j]*da_alpha_dT_ijs_i[j]
#            der_sum = sum([zs[j]*da_alpha_dT_ijs[i][j] ])
            
            x20 = x50*(x51*x9 + der_sum) + x52
            x21 = bs[i]*x14
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



    def d_lnphi_dzs_analytical0(self, Z, zs):
        
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
        
    
    def d_lnphi_dzs_basic_num(self, Z, zs):
        from thermo import normalize
        all_diffs = []
        
        try:
            if self.G_dep_l < self.G_dep_g:
                lnphis_ref = self.lnphis_l
            else:
                lnphis_ref = self.lnphis_g
        except:
           lnphis_ref = self.lnphis_l if hasattr(self, 'G_dep_l') else self.lnphis_g
        
        
        
        
        for i in range(len(zs)):
            zs2 = list(zs)
            dz = 1e-7#zs2[i]*3e-
            zs2[i] = zs2[i]+dz
#            sum_one = sum(zs2)
#            zs2 = normalize(zs2)
            eos2 = self.to_TP_zs(T=self.T, P=self.P, zs=zs2)
            
            

            diffs = []
            for j in range(len(zs)):
                try:
                    dlnphis = (eos2.lnphis_g[j] - lnphis_ref[j])/dz
                except:
                    dlnphis = (eos2.lnphis_l[j] - lnphis_ref[j])/dz
                diffs.append(dlnphis)
            all_diffs.append(diffs)
        import numpy as np
        return np.array(all_diffs).T.tolist()
    
    
    def d_lnphi_dzs_numdifftools(self, Z, zs):
        import numpy as np
        import numdifftools as nd
        
        def lnphis_from_zs(zs2):
            if isinstance(zs2, np.ndarray):
                zs2 = zs2.tolist()
            # Last row suggests the normalization breaks everything!
#            zs2 = normalize(zs2)
            
#            if Z == self.Z_l


            try:
                return np.array(self.to_TP_zs(T=self.T, P=self.P, zs=zs2).lnphis_l)
            except:
                return np.array(self.to_TP_zs(T=self.T, P=self.P, zs=zs2).lnphis_g)
    
        Jfun_partial = nd.Jacobian(lnphis_from_zs, step=1e-4, order=2, method='central')
        return Jfun_partial(zs)
    
    def d_lnphi_dzs(self, Z, zs):
        # Is it possible to numerically evaluate different parts to try to find the problems?
        T, P = self.T, self.P
        T2 = T*T
        T_inv = 1.0/T
        RT_inv = R_inv*T_inv
        bs, b = self.bs, self.b
        a_alpha = self.a_alpha
        a_alpha_ijs = self.a_alpha_ijs
        a_alphas = self.a_alphas
        fugacity_sum_terms = self.fugacity_sum_terms
        N = len(zs)
        cmps = range(N)

        b2 = b*b
        b_inv = 1.0/b
        b2_inv = b_inv*b_inv
        a_alpha2 = a_alpha*a_alpha

        A = a_alpha*P*RT_inv*RT_inv
        B = b*P*RT_inv
        B_inv = 1.0/B
        C = 1.0/(Z - B)
        
        Zm1 = Z - 1.0
#        Dis = [bi/b*Zm1 for bi in self.bs] # not needed

        G = (Z + (1.0 + root_two)*B)/(Z + (1.0 - root_two)*B)
        
        
        t4 = 2.0/a_alpha
        t5 = -A/(two_root_two*B)
        Eis = [t5*(t4*fugacity_sum_terms[i] - bs[i]*b_inv) for i in cmps]
#        ln_phis = []
#        for i in cmps:
#            ln_phis.append(log(C) + Dis[i] + Eis[i]*log(G))
#        return ln_phis

#        Bis = [bi*P/(R*T) for bi in bs]
        # maybe with a 2 constant? 
        t6 = P*RT_inv
        dB_dxks = [t6*bk for bk in bs]
        
        
        # THIS IS WRONG - the sum changes w.r.t (or does it?)
        # Believed right now?
        const = (P+P)*RT_inv*RT_inv
        dA_dxks = [const*term_i for term_i in fugacity_sum_terms]
            
        dF_dZ_inv = 1.0/(3.0*Z*Z - 2.0*Z*(1.0 - B) + (A - 3.0*B*B - 2.0*B))
        
        t15 = (A - 2.0*B - 3.0*B*B + 2.0*(3.0*B + 1.0)*Z - Z*Z)
        BmZ = (B - Z)
        dZ_dxs = [(BmZ*dA_dxks[i] + t15*dB_dxks[i])*dF_dZ_inv for i in cmps]
        
        # function only of k
        ZmB = Z - B
        t20 = -1.0/(ZmB*ZmB)
        dC_dxs = [t20*(dZ_dxs[k] - dB_dxks[k]) for k in cmps]
        
        dD_dxs = []
#        dD_dxs = [[0.0]*N for _ in cmps]
        t55s = [b*dZ_dxs[k] - bs[k]*Zm1 for k in cmps]
        for i in cmps:
#            dD_dxs_i = dD_dxs[i]
            b_term_ratio = bs[i]*b2_inv
            dD_dxs.append([b_term_ratio*t55s[k] for k in cmps])
#            for k in cmps:
#                dD_dxs_i[k] = b_term_ratio*t55s[k]
#        dD_dxs = []
#        for i in cmps:
#            term = bs[i]/(b*b)*(b*dZ_dxs[i] - b*(Z - 1.0))
#            dD_dxs.append(term)
            
        # ? Believe this is the only one with multi indexes?
        t1 = 1.0/(two_root_two*a_alpha*b*B)
        t2 = t1*A/(a_alpha*b)
        t50s = [B*dA_dxks[k] - A*dB_dxks[k] for k in cmps]
        
        # problem is in here, tested numerically
        b_two = b + b
        t32 = 2.0*a_alpha*b2
        t33 = 4.0*b2
        t34 = t1*B_inv*a_alpha
        t35 = -t1*B_inv*b_two
        
        # Symmetric matrix!
        dE_dxs = [[0.0]*N for _ in cmps] # TODO - makes little sense. Too many i indexes.
        for i in cmps:
            zm_aim_tot = fugacity_sum_terms[i]
            t30 = t34*bs[i] + t35*zm_aim_tot
            t31 = t33*zm_aim_tot
            
            dE_dxs_i = []
            a_alpha_ijs_i = a_alpha_ijs[i]
            for k in range(0, i+1):
                # Sign was wrong in article - should be a plus
                second = t2*(t31*fugacity_sum_terms[k] - t32*a_alpha_ijs_i[k] - bs[i]*bs[k]*a_alpha2)
                dE_dxs[i][k] = dE_dxs[k][i] = t30*t50s[k] + second
                
#                dE_dxs_i.append(t1*(first + second))
#            dE_dxs.append(dE_dxs_i)
                
        t59 = (Z + (1.0 - root_two)*B)
        t60 = two_root_two/(t59*t59)
        dG_dxs = [t60*(Z*dB_dxks[k] - B*dZ_dxs[k]) for k in cmps]
            
        
        G_inv = 1.0/G
        logG = log(G)
        C_inv = 1.0/C
        dlnphis_dxs = []
#        dlnphis_dxs = [[0.0]*N for _ in cmps]
        t61s = [C_inv*dC_dxi for dC_dxi in dC_dxs]
        for i in cmps:
            dD_dxs_i = dD_dxs[i]
            dE_dxs_i = dE_dxs[i]
            E_G = Eis[i]*G_inv
#            dlnphis_dxs_i = dlnphis_dxs[i]
            dlnphis_dxs_i = [t61s[k] + dD_dxs_i[k] + logG*dE_dxs_i[k] + E_G*dG_dxs[k]
                             for k in cmps]
            dlnphis_dxs.append(dlnphis_dxs_i)

#        return dlnphis_dxs
        return dlnphis_dxs#, dZ_dxs, dA_dxks, dB_dxks, dC_dxs, dD_dxs, dE_dxs, dG_dxs
        

#        d_lnphi_dzs = d_lnphi_dzs_Varavei

#    def dZ_dzs(self, Z, zs):
#        '''
#        from fluids.numerics import derivative
#        Tcs = [126.2, 304.2, 373.2]
#        Pcs = [3394387.5, 7376460.0, 8936865.0]
#        omegas = [0.04, 0.2252, 0.1]
#        zs = [.7, .2, .1]
#        eos = PRMIX(T=300, P=1e5, zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas)
#
#        def dZ_dn(ni, i):
#            zs = [.7, .2, .1]
#            zs[i] = ni
#            eos = PRMIX(T=300, P=1e5, zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas)
#            return eos.Z_g
#        [derivative(dZ_dn, ni, dx=1e-3, order=17, args=(i,)) for i, ni in zip((0, 1, 2), (.7, .2, .1))], eos.d_Z_dzs(eos.Z_g, zs) 
#        '''
#        # Not even correct for SRK eos, needs to be re derived there
#        T, P = self.T, self.P
#        bs, b = self.bs, self.b
#        RT_inv = R_inv/T
#        fugacity_sum_terms = self.fugacity_sum_terms
#        A = self.a_alpha*P*RT_inv*RT_inv
#        B = b*P*RT_inv
#        C = 1.0/(Z - B)
#        
#        Zm1 = Z - 1.0
#        
#        t6 = P*RT_inv
#        dB_dxks = [t6*bk for bk in bs]
#        
#        const = (P+P)*RT_inv*RT_inv
#        dA_dxks = [const*term_i for term_i in fugacity_sum_terms]
#        
#        dF_dZ_inv = 1.0/(3.0*Z*Z - 2.0*Z*(1.0 - B) + (A - 3.0*B*B - 2.0*B))
#        
#        t15 = (A - 2.0*B - 3.0*B*B + 2.0*(3.0*B + 1.0)*Z - Z*Z)
#        BmZ = (B - Z)
#        dZ_dxs = [(BmZ*dA_dxks[i] + t15*dB_dxks[i])*dF_dZ_inv for i in self.cmps]
#        return dZ_dxs

#    def dV_dzs(self, Z, zs):
#        # This one is fine for all EOSs
#        factor = self.T*R/self.P
#        return [i*factor for i in self.dZ_dzs(Z, zs)]
#    
#        
        
    @property
    def ddelta_dzs(self):   
        r'''Helper method for calculating the composition derivatives of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \delta}{\partial x_i}\right)_{T, P, x_{i\ne j}} 
            = 2 b_i

        Returns
        -------
        ddelta_dzs : list[float]
            Composition derivative of `delta` of each component, [m^3/mol]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return [i + i for i in self.bs]
    
    @property
    def ddelta_dns(self):   
        r'''Helper method for calculating the mole number derivatives of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \delta}{\partial n_i}\right)_{T, P, n_{i\ne j}} 
            = 2 (b_i - b)

        Returns
        -------
        ddelta_dns : list[float]
            Mole number derivative of `delta` of each component, [m^3/mol^2]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        b = self.b
        return [2.0*(bi - b) for bi in self.bs]

    @property
    def d2delta_dzizjs(self):   
        r'''Helper method for calculating the second composition derivatives (hessian) of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \delta}{\partial x_i\partial x_j}\right)_{T, P, x_{k\ne i,j}} 
            = 0

        Returns
        -------
        d2delta_dzizjs : list[float]
            Second Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        return [[0.0]*self.N for i in self.cmps]

    @property
    def d2delta_dninjs(self):   
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \delta}{\partial n_i \partial n_j}\right)_{T, P, n_{k\ne i,j}} 
            = 4b - 2b_i - 2b_j

        Returns
        -------
        d2delta_dninjs : list[list[float]]
            Second mole number derivative of `delta` of each component, [m^3/mol^3]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        bb = 2.0*self.b
        bs = self.bs
        cmps = self.cmps
        d2b_dninjs = []
        for bi in self.bs:
            d2b_dninjs.append([2.0*(bb - bi - bj) for bj in bs])
        return d2b_dninjs
    
    @property
    def depsilon_dzs(self):       
        r'''Helper method for calculating the composition derivatives of
        `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \epsilon}{\partial x_i}\right)_{T, P, x_{i\ne j}} 
            = -2 b_i\cdot b

        Returns
        -------
        depsilon_dzs : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        b2n = -2.0*self.b
        return [bi*b2n for bi in self.bs]

    @property    
    def depsilon_dns(self):       
        r'''Helper method for calculating the mole number derivatives of
        `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \epsilon}{\partial n_i}\right)_{T, P, n_{i\ne j}} 
            = 2b(b - b_i)

        Returns
        -------
        depsilon_dns : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^3]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        b = self.b
        b2 = b + b
        return [b2*(b - bi) for bi in self.bs]
    
    @property
    def d2epsilon_dzizjs(self):       
        r'''Helper method for calculating the second composition derivatives (hessian) 
        of `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial x_i \partial x_j}\right)_{T, P, x_{k\ne i,j}} 
            =  2 b_i b_j

        Returns
        -------
        d2epsilon_dzizjs : list[list[float]]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        bs = self.bs
        return [[-2.0*bi*bj for bi in bs] for bj in bs]

    @property
    def d2epsilon_dninjs(self):       
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial n_i n_j}\right)_{T, P, n_{k\ne i,j}} 
            = -2b(2b - b_i - b_j) - 2(b - b_i)(b - b_j)
        Returns
        -------
        d2epsilon_dninjs : list[list[float]]
            Second composition derivative of `epsilon` of each component, [m^6/mol^4]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        bs = self.bs
        b = self.b
        cmps = self.cmps
        bb = b + b
        d2epsilon_dninjs = []
        for i in cmps:
            l = []
            for j in cmps:
                bi, bj = bs[i], bs[j]
#                if i != j:
                v = -bb*(bb - (bi + bj))  -2.0*(b - bi)*(b - bj)
#                else:
#                    v = -2.0*(b - bi)*(3.0*b - bj)
                l.append(v)
            d2epsilon_dninjs.append(l)
        return d2epsilon_dninjs


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
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optoinal
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

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
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
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

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()
        
    def fast_init_specific(self, other):
        self.ms = other.ms
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

    @property
    def ddelta_dzs(self):   
        r'''Helper method for calculating the composition derivatives of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \delta}{\partial x_i}\right)_{T, P, x_{i\ne j}} 
            = b_i

        Returns
        -------
        ddelta_dzs : list[float]
            Composition derivative of `delta` of each component, [m^3/mol]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return self.bs

    @property
    def ddelta_dns(self):   
        r'''Helper method for calculating the mole number derivatives of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \delta}{\partial n_i}\right)_{T, P, n_{i\ne j}} 
            = (b_i - b)

        Returns
        -------
        ddelta_dns : list[float]
            Mole number derivative of `delta` of each component, [m^3/mol^2]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        b = self.b
        return [(bi - b) for bi in self.bs]

    @property
    def d2delta_dzizjs(self):   
        r'''Helper method for calculating the second composition derivatives (hessian) of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \delta}{\partial x_i\partial x_j}\right)_{T, P, x_{k\ne i,j}} 
            = 0

        Returns
        -------
        d2delta_dzizjs : list[float]
            Second Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        return [[0.0]*self.N for i in self.cmps]

    @property
    def d2delta_dninjs(self):   
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \delta}{\partial n_i \partial n_j}\right)_{T, P, n_{k\ne i,j}} 
            = 2b - b_i - b_j

        Returns
        -------
        d2delta_dninjs : list[list[float]]
            Second mole number derivative of `delta` of each component, [m^3/mol^3]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return self.d2b_dninjs

    @property
    def depsilon_dzs(self):       
        r'''Helper method for calculating the composition derivatives of
        `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \epsilon}{\partial x_i}\right)_{T, P, x_{i\ne j}} 
            = 0

        Returns
        -------
        depsilon_dzs : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return [0.0]*self.N

    @property
    def depsilon_dns(self):       
        r'''Helper method for calculating the mole number derivatives of
        `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \epsilon}{\partial n_i}\right)_{T, P, n_{i\ne j}} 
            = 0

        Returns
        -------
        depsilon_dns : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^3]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return [0.0]*self.N

    @property
    def d2epsilon_dzizjs(self):       
        r'''Helper method for calculating the second composition derivatives (hessian) 
        of `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial x_i \partial x_j}\right)_{T, P, x_{k\ne i,j}} 
            = 0

        Returns
        -------
        d2epsilon_dzizjs : list[list[float]]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return [[0.0]*self.N for i in self.cmps]

    @property
    def d2epsilon_dninjs(self):       
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial n_i n_j}\right)_{T, P, n_{k\ne i,j}} 
            = 0

        Returns
        -------
        d2epsilon_dninjs : list[list[float]]
            Second composition derivative of `epsilon` of each component, [m^6/mol^4]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return [[0.0]*self.N for i in self.cmps]


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
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optoinal
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

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
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
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

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
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
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optoinal
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]
        
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
                 omegas=None, fugacities=True, only_l=False, only_g=False):
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
        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def fast_init_specific(self, other):
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

    @property
    def ddelta_dzs(self):   
        r'''Helper method for calculating the composition derivatives of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \delta}{\partial x_i}\right)_{T, P, x_{i\ne j}} 
            = 0

        Returns
        -------
        ddelta_dzs : list[float]
            Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        return [0.0]*self.N

    @property
    def ddelta_dns(self):   
        r'''Helper method for calculating the mole number derivatives of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \delta}{\partial n_i}\right)_{T, P, n_{i\ne j}} 
            = 0

        Returns
        -------
        ddelta_dns : list[float]
            Mole number derivative of `delta` of each component, [m^3/mol^2]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return [0.0]*self.N

    @property
    def depsilon_dzs(self):       
        r'''Helper method for calculating the composition derivatives of
        `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \epsilon}{\partial x_i}\right)_{T, P, x_{i\ne j}} 
            = 0

        Returns
        -------
        depsilon_dzs : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return [0.0]*self.N

    @property
    def depsilon_dns(self):       
        r'''Helper method for calculating the mole number derivatives of
        `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial \epsilon}{\partial n_i}\right)_{T, P, n_{i\ne j}} 
            = 0

        Returns
        -------
        depsilon_dns : list[float]
            Composition derivative of `epsilon` of each component, [m^6/mol^3]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return [0.0]*self.N

    @property
    def d2delta_dzizjs(self):   
        r'''Helper method for calculating the second composition derivatives (hessian) of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \delta}{\partial x_i\partial x_j}\right)_{T, P, x_{k\ne i,j}} 
            = 0

        Returns
        -------
        d2delta_dzizjs : list[float]
            Second Composition derivative of `delta` of each component, [m^3/mol]

        Notes
        -----
        This derivative is checked numerically.
        '''
        return [[0.0]*self.N for i in self.cmps]

    @property
    def d2delta_dninjs(self):   
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `delta`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \delta}{\partial n_i \partial n_j}\right)_{T, P, n_{k\ne i,j}} 
            = 0

        Returns
        -------
        d2delta_dninjs : list[list[float]]
            Second mole number derivative of `delta` of each component, [m^3/mol^3]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return [[0.0]*self.N for i in self.cmps]

    @property
    def d2epsilon_dzizjs(self):       
        r'''Helper method for calculating the second composition derivatives (hessian) 
        of `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial x_i \partial x_j}\right)_{T, P, x_{k\ne i,j}} 
            = 0

        Returns
        -------
        d2epsilon_dzizjs : list[list[float]]
            Composition derivative of `epsilon` of each component, [m^6/mol^2]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return [[0.0]*self.N for i in self.cmps]

    @property
    def d2epsilon_dninjs(self):       
        r'''Helper method for calculating the second mole number derivatives (hessian) of
        `epsilon`. Note this is independent of the phase.
        
        .. math::
            \left(\frac{\partial^2 \epsilon}{\partial n_i n_j}\right)_{T, P, n_{k\ne i,j}} 
            = 0

        Returns
        -------
        d2epsilon_dninjs : list[list[float]]
            Second composition derivative of `epsilon` of each component, [m^6/mol^4]
            
        Notes
        -----
        This derivative is checked numerically.
        '''
        return [[0.0]*self.N for i in self.cmps]


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
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optoinal
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

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
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 kappa1s=None, fugacities=True, only_l=False, only_g=False):
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

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()



    def fast_init_specific(self, other):
        self.kappa0s = other.kappa0s
        self.kappa1s = other.kappa1s
        self.kappas = other.kappas
        b = 0.0
        for bi, zi in zip(self.bs, self.zs):
            b += bi*zi
        self.b = b
        self.delta = 2.0*b
        self.epsilon = -b*b


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
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optoinal
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

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
                 kappa1s=None, kappa2s=None, kappa3s=None,
                 fugacities=True, only_l=False, only_g=False):
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
        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def fast_init_specific(self, other):
        self.kappa0s = other.kappa0s
        self.kappa1s = other.kappa1s
        self.kappa2s = other.kappa2s
        self.kappa3s = other.kappa3s
        self.kappas = other.kappas
        b = 0.0
        for bi, zi in zip(self.bs, self.zs):
            b += bi*zi
        self.b = b
        self.delta = 2.0*b
        self.epsilon = -b*b

        
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
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optoinal
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

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
    
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
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

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
            self.fugacities()

    def fast_init_specific(self, other):
        b = 0.0
        for bi, zi in zip(self.bs, self.zs):
            b += bi*zi
        self.b = b
        self.delta = 2.0*b
        self.epsilon = -b*b


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
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optoinal
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

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
    def __init__(self, Tcs, Pcs, omegas, zs, kijs=None, T=None, P=None, V=None,
                 fugacities=True, only_l=False, only_g=False):
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

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
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
    fugacities : bool, optional
        Whether or not to calculate fugacity related values (phis, log phis,
        and fugacities); default True, [-]
    only_l : bool, optional
        When true, if there is a liquid and a vapor root, only the liquid
        root (and properties) will be set; default False, [-]
    only_g : bool, optoinal
        When true, if there is a liquid and a vapor root, only the vapor
        root (and properties) will be set; default False, [-]

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
                 S1s=None, S2s=None, fugacities=True, only_l=False, only_g=False):
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

        self.solve(only_l=only_l, only_g=only_g)
        if fugacities:
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


def eos_lnphis_test_phase_stability(eos):        
    try:
        if eos.G_dep_l < eos.G_dep_g:
            lnphis_eos = eos.lnphis_l
            prefer, alt = 'lnphis_g', 'lnphis_l'
        else:
            lnphis_eos = eos.lnphis_g
            prefer, alt =  'lnphis_l', 'lnphis_g'
    except:
        # Only one root - take it and set the prefered other phase to be a different type
        lnphis_eos = eos.lnphis_g if hasattr(eos, 'lnphis_g') else eos.lnphis_l
        prefer = 'lnphis_l' if hasattr(eos, 'lnphis_g') else 'lnphis_g'
        alt = 'lnphis_g' if hasattr(eos, 'lnphis_g') else 'lnphis_l'
    return lnphis_eos, prefer, alt


def eos_lnphis_trial_phase_stability(eos, prefer, alt):
    try:
        if eos.G_dep_l < eos.G_dep_g:
            lnphis_trial = eos.lnphis_l
        else:
            lnphis_trial = eos.lnphis_g
    except:
        # Only one phase, doesn't matter - only that phase will be returned
        try:
            lnphis_trial = getattr(eos, alt)
        except:
            lnphis_trial = getattr(eos, prefer)
    return lnphis_trial

eos_mix_list = [PRMIX, SRKMIX, PR78MIX, VDWMIX, PRSVMIX, PRSV2MIX, TWUPRMIX, TWUSRKMIX, APISRKMIX]
