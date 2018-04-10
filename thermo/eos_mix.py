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
'PRSV2MIX', 'TWUPRMIX', 'TWUSRKMIX', 'APISRKMIX']
import numpy as np
from scipy.optimize import newton, minimize
from scipy.misc import derivative
from thermo.utils import normalize, Cp_minus_Cv, isobaric_expansion, isothermal_compressibility, phase_identification_parameter
from thermo.utils import R
from thermo.utils import log, exp, sqrt
from thermo.eos import *
import sys

R2 = R*R
two_root_two = 2*2**0.5
root_two = sqrt(2.)
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
    def a_alpha_and_derivatives(self, T, full=True, quick=True):
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
        zs, kijs = self.zs, self.kijs
        a_alphas, da_alpha_dTs, d2a_alpha_dT2s = [], [], []
        
        for i in self.cmps:
            self.setup_a_alpha_and_derivatives(i, T=T)
            # Abuse method resolution order to call the a_alpha_and_derivatives
            # method of the original pure EOS
            # -4 goes back from object, GCEOS, SINGLEPHASEEOS, up to GCEOSMIX
            ds = super(type(self).__mro__[self.a_alpha_mro], self).a_alpha_and_derivatives(T)
            a_alphas.append(ds[0])
            da_alpha_dTs.append(ds[1])
            d2a_alpha_dT2s.append(ds[2])
        self.cleanup_a_alpha_and_derivatives()
        
        da_alpha_dT, d2a_alpha_dT2 = 0.0, 0.0
        
        a_alpha_ijs = [[(1. - kijs[i][j])*(a_alphas[i]*a_alphas[j])**0.5 
                              for j in self.cmps] for i in self.cmps]
                                
        # Needed in calculation of fugacity coefficients
        z_products = [[zs[i]*zs[j] for j in self.cmps] for i in self.cmps]
        
        
        a_alpha = sum([a_alpha_ijs[i][j]*z_products[i][j]
                      for j in self.cmps for i in self.cmps])
        self.a_alpha_ijs = a_alpha_ijs
        
        if full:
            for i in self.cmps:
                for j in self.cmps:
                    a_alphai, a_alphaj = a_alphas[i], a_alphas[j]
                    x0 = a_alphai*a_alphaj
                    x0_05 = x0**0.5
                    zi_zj = z_products[i][j]

                    da_alpha_dT += zi_zj*((1. - kijs[i][j])/(2.*x0_05)
                    *(a_alphai*da_alpha_dTs[j] + a_alphaj*da_alpha_dTs[i]))
                    
                    x1 = a_alphai*da_alpha_dTs[j]
                    x2 = a_alphaj*da_alpha_dTs[i]
                    x3 = 2.*a_alphai*da_alpha_dTs[j] + 2.*a_alphaj*da_alpha_dTs[i]
                    d2a_alpha_dT2 += (-x0_05*(kijs[i][j] - 1.)*(x0*(
                    2.*a_alphai*d2a_alpha_dT2s[j] + 2.*a_alphaj*d2a_alpha_dT2s[i]
                    + 4.*da_alpha_dTs[i]*da_alpha_dTs[j]) - x1*x3 - x2*x3 + (x1 
                    + x2)**2)/(4.*x0*x0))*zi_zj
        
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
        else:
            return a_alpha
        
    def to_mechanical_critical_point(self):
        Pmc = sum([self.Pcs[i]*self.zs[i] for i in self.cmps])
        Tmc = sum([(self.Tcs[i]*self.Tcs[j])**0.5*self.zs[j]*self.zs[i] for i in self.cmps
                  for j in self.cmps])
        
        # Calculate Vc using Zc at the eos level. 
        def to_minimize(TP):
            # Sometimes there are zero-division errors when dP_dV is literally zero
            eos = self.to_TP_zs(T=float(TP[0]), P=float(TP[1]), zs=self.zs)
            err = 0
            err += abs(eos.raw_volumes[0] - eos.raw_volumes[1])
            err += abs(eos.raw_volumes[1] - eos.raw_volumes[2])
            return err
        
        # differential_evolution(to_maximize,bounds=[(400, 500), [3380688/2, 3380688*2]], popsize=100, maxiter=100)
        
        ans = minimize(to_minimize, [Tmc, Pmc], tol=1e-12, method='Nelder-Mead')
        if ans['fun'] < 1E-5:
            T, P = ans['x']
            return self.to_TP_zs(T=float(T), P=float(P), zs=self.zs)
        else:
            raise Exception('Not Found', ans)
        
        
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
        if self.phase in ('l', 'l/g'):
            if xs is None:
                xs = self.zs
            if hasattr(self, 'Z_l'):
                self.phis_l = self.fugacity_coefficients(self.Z_l, zs=xs)
                self.fugacities_l = [phi*x*self.P for phi, x in zip(self.phis_l, xs)]
                self.lnphis_l = [log(i) for i in self.phis_l]
        if self.phase in ('g', 'l/g'):
            if ys is None:
                ys = self.zs
            if hasattr(self, 'Z_g'):
                self.phis_g = self.fugacity_coefficients(self.Z_g, zs=ys)
                self.fugacities_g = [phi*y*self.P for phi, y in zip(self.phis_g, ys)]
                self.lnphis_g = [log(i) for i in self.phis_g]


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
        z_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        
        tot = 0
        for yi, phi_yi, zi, phi_zi in zip(ys, y_fugacity_coefficients, zs, z_fugacity_coefficients):
            di = log(zi) + log(phi_zi)
            tot += yi*(log(yi) + log(phi_yi) - di)
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
        z_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        
        kis = []
        for yi, phi_yi, zi, phi_zi in zip(ys, y_fugacity_coefficients, zs, z_fugacity_coefficients):
            di = log(zi*phi_zi)
            try:
                ki = log(yi*phi_yi) - di
            except ValueError:
                ki = log(1e-200*phi_yi) - di
            kis.append(ki)
        kis.append(kis[0])

        tot = 0.0
        for i in range(self.N):
            t = kis[i+1] - kis[i]
            tot += t*t
        return tot

    def d_TPD_dy(self, Zz, Zy, zs, ys):
        # The gradient should be - for all variables
        z_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        gradient = []
        for yi, phi_yi, zi, phi_zi in zip(ys, y_fugacity_coefficients, zs, z_fugacity_coefficients):
            hi = di = log(zi) + log(phi_zi) # same as di
            k = log(yi) + log(phi_yi) - hi
            Yi = exp(-k)*yi
            gradient.append(log(phi_yi) + log(Yi) - di)
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
        z_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        tot = 0
        for Yi, phi_yi, zi, phi_zi in zip(Ys, y_fugacity_coefficients, zs, z_fugacity_coefficients):
            di = log(zi) + log(phi_zi)
            if Yi != 0:
                diff = Yi**0.5*(log(Yi) + log(phi_yi) - di)
                tot += abs(diff)
        return tot


    def TDP_Michelsen(self, Zz, Zy, zs, ys):
        
        z_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        tot = 0
        for yi, phi_yi, zi, phi_zi in zip(ys, y_fugacity_coefficients, zs, z_fugacity_coefficients):
            hi = di = log(zi) + log(phi_zi) # same as di
            
            k = log(yi) + log(phi_yi) - hi
            # Michaelsum doesn't do the exponents.
            Yi = exp(-k)*yi
            tot += Yi*(log(Yi) + log(phi_yi) - hi - 1.)
            
        return 1. + tot

    def TDP_Michelsen_modified(self, Zz, Zy, zs, Ys):
        # https://www.e-education.psu.edu/png520/m17_p7.html
        # Might as well continue
        Ys = [abs(float(Yi)) for Yi in Ys]
        # Ys only need to be positive
        ys = normalize(Ys)
        
        z_fugacity_coefficients = self.fugacity_coefficients(Zz, zs)
        y_fugacity_coefficients = self.fugacity_coefficients(Zy, ys)
        
        tot = 0
        for Yi, phi_yi, yi, zi, phi_zi in zip(Ys, y_fugacity_coefficients, ys, zs, z_fugacity_coefficients):
            hi = di = log(zi) + log(phi_zi) # same as di
            tot += Yi*(log(Yi) + log(phi_yi) - di - 1.)
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

    def to_TP_zs(self, T, P, zs):
        if T != self.T or P != self.P or zs != self.zs:
            return self.__class__(T=T, P=P, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, zs=zs, **self.kwargs)
        else:
            return self



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
        self.kappas = [0.37464 + 1.54226*omega - 0.26992*omega*omega for omega in omegas]
        
        self.delta = 2.*self.b
        self.epsilon = -self.b*self.b

        self.solve()
        self.fugacities()
        
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
        phis : float
            Fugacity coefficient for each species, [-]
                         
        References
        ----------
        .. [1] Peng, Ding-Yu, and Donald B. Robinson. "A New Two-Constant  
           Equation of State." Industrial & Engineering Chemistry Fundamentals 
           15, no. 1 (February 1, 1976): 59-64. doi:10.1021/i160057a011.
        .. [2] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        '''
        from cmath import log
        A = self.a_alpha*self.P/(R2*self.T*self.T)
        B = self.b*self.P/(R*self.T)
        phis = []
        for i in self.cmps:
            # The two log terms need to use a complex log; typically these are
            # calculated at "liquid" volume solutions which are unstable
            # and cannot exist
            t1 = self.bs[i]/self.b*(Z - 1.) - log(Z - B).real
            t2 = 2./self.a_alpha*sum([zs[j]*self.a_alpha_ijs[i][j] for j in self.cmps])
            t3 = t1 - A/(two_root_two*B)*(t2 - self.bs[i]/self.b)*log((Z + (root_two + 1.)*B)/(Z - (root_two - 1.)*B)).real
            phis.append(exp(t3))
        return phis


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
        self.ms = [0.480 + 1.574*omega - 0.176*omega*omega for omega in omegas]
        self.delta = self.b

        self.solve()
        self.fugacities()

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
        phis : float
            Fugacity coefficient for each species, [-]
                         
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
        for i in self.cmps:
            Bi = self.bs[i]*P_RT
            t1 = Bi*Z_minus_one_over_B - t0
            l = self.a_alpha_ijs[i]
            t2 = A_B*(Bi/B - two_over_a_alpha*sum([zs[j]*l[j] for j in self.cmps]))
            phis.append(exp(t1 + t2*t3))
        return phis
        

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
    def __init__(self, Tcs, Pcs, zs, kijs=None, T=None, P=None, V=None):
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
        
        self.solve()
        self.fugacities()
        
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
        phis : float
            Fugacity coefficient for each species, [-]
                         
        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        '''
        phis = []
        V = Z*R*self.T/self.P
        for i in self.cmps:
            phi = self.bs[i]/(V-self.b) - log(Z*(1. - self.b/V)) - 2.*(self.a_alpha*self.ais[i])**0.5/(R*self.T*V)
            phis.append(exp(phi))
        return phis


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
            kappa1s = [0 for i in self.cmps]
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
    def __init__(self, Tcs, Pcs, zs, omegas=None, kijs=None, T=None, P=None, V=None,
                 S1s=None, S2s=None):
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

        self.check_sufficient_inputs()

        # Setup S1s and S2s
        if S1s is None and omegas is None:
            raise Exception('Either acentric factor of S1 is required')
        if S1s is None:
            self.S1s = [0.48508 + 1.55171*omega - 0.15613*omega*omega for omega in omegas]
        else:
            self.S1s = S1s
        if S2s is None:
            S2s = [0 for i in self.cmps]
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


