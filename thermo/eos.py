# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['GCEOS', 'PR', 'SRK', 'PR78', 'PRSV', 'PRSV2', 'VDW', 'RK',  
'APISRK', 'TWUPR', 'TWUSRK', 'ALPHA_FUNCTIONS', 'eos_list', 'GCEOS_DUMMY',
'IG']

from cmath import atanh as catanh
from fluids.numerics import (chebval, brenth, third, sixth, roots_cubic,
                             roots_cubic_a1, numpy as np, py_newton as newton,
                             py_bisect as bisect, inf, polyder, chebder, 
                             trunc_exp, secant)
from thermo.utils import R
from thermo.utils import (Cp_minus_Cv, isobaric_expansion, 
                          isothermal_compressibility, 
                          phase_identification_parameter)
from thermo.utils import log, exp, sqrt, copysign, horner

R2 = R*R
R_2 = 0.5*R
R_inv = 1.0/R


class GCEOS(object):
    r'''Class for solving a generic Pressure-explicit three-parameter cubic 
    equation of state. Does not implement any parameters itself; must be 
    subclassed by an equation of state class which uses it. Works for mixtures
    or pure species for all properties except fugacity. All properties are 
    derived with the CAS SymPy, not relying on any derivations previously 
    published.

    .. math::
        P=\frac{RT}{V-b}-\frac{a\alpha(T)}{V^2 + \delta V + \epsilon}

    Main methods (in order they are called) are `solve`, `set_from_PT`,
    `volume_solutions`, `set_properties_from_solution`,  and
    `derivatives_and_departures`. 

    `solve` calls `check_sufficient_input`, which checks if two of `T`, `P`, 
    and `V` were set. It then solves for the 
    remaining variable. If `T` is missing, method `solve_T` is used; it is
    parameter specific, and so must be implemented in each specific EOS. 
    If `P` is missing, it is directly calculated. If `V` is missing, it
    is calculated with the method `volume_solutions`. At this point, either
    three possible volumes or one user specified volume are known. The
    value of `a_alpha`, and its first and second temperature derivative are
    calculated with the EOS-specific method `a_alpha_and_derivatives`. 

    If `V` is not provided, `volume_solutions` calculates the three 
    possible molar volumes which are solutions to the EOS; in the single-phase 
    region, only one solution is real and correct. In the two-phase region, all 
    volumes are real, but only the largest and smallest solution are physically 
    meaningful, with the largest being that of the gas and the smallest that of
    the liquid.

    `set_from_PT` is called to sort out the possible molar volumes. For the 
    case of a user-specified `V`, the possibility of there existing another 
    solution is ignored for speed. If there is only one real volume, the 
    method `set_properties_from_solution` is called with it. If there are
    two real volumes, `set_properties_from_solution` is called once with each 
    volume. The phase is returned by `set_properties_from_solution`, and the
    volumes is set to either `V_l` or `V_g` as appropriate. 
    
    `set_properties_from_solution` is a beast which calculates all relevant
    partial derivatives and properties of the EOS. 15 derivatives and excess
    enthalpy and entropy are calculated first. If the method was called with 
    the `quick` flag, the method `derivatives_and_departures` uses a mess 
    derived with SymPy's `cse` function to perform the calculation as quickly
    as possible. Otherwise, the independent formulas for each property are used.

    `set_properties_from_solution` next calculates `beta` (isobaric expansion
    coefficient), `kappa` (isothermal compressibility), `Cp_minus_Cv`, `Cv_dep`,
    `Cp_dep`, `V_dep` molar volume departure, `U_dep` internal energy departure,
    `G_dep` Gibbs energy departure, `A_dep` Helmholtz energy departure,
    `fugacity`, and `phi` (fugacity coefficient). It then calculates
    `PIP` or phase identification parameter, and determines the fluid phase
    with it. Finally, it sets all these properties as attibutes or either 
    the liquid or gas phase with the convention of adding on `_l` or `_g` to
    the variable names.
    '''
    # Slots does not help performance in either implementation
    kwargs = {}
    N = 1
    multicomponent = False
    
    def __repr__(self):
        s = '%s(Tc=%s, Pc=%s, omega=%s, ' %(self.__class__.__name__, repr(self.Tc), repr(self.Pc), repr(self.omega))
        if hasattr(self, 'no_T_spec') and self.no_T_spec:
            s += 'P=%s, V=%s' %(repr(self.P), repr(self.V))
        elif self.V is not None:
            s += 'T=%s, V=%s' %(repr(self.T), repr(self.V))
        else:
            s += 'T=%s, P=%s' %(repr(self.T), repr(self.P))
        s += ')'
        return s
    
    def check_sufficient_inputs(self):
        '''Method to an exception if none of the pairs (T, P), (T, V), or 
        (P, V) are given. '''
        if not ((self.T is not None and self.P is not None) or
                (self.T is not None and self.V is not None) or 
                (self.P is not None and self.V is not None)):
            raise Exception('Either T and P, or T and V, or P and V are required')


    def solve(self, pure_a_alphas=True, only_l=False, only_g=False, full_alphas=True):
        '''First EOS-generic method; should be called by all specific EOSs.
        For solving for `T`, the EOS must provide the method `solve_T`.
        For all cases, the EOS must provide `a_alpha_and_derivatives`.
        Calls `set_from_PT` once done.
        '''
        self.check_sufficient_inputs()
        
        if self.V is not None:
            if self.P is not None:
                self.T = self.solve_T(self.P, self.V)
                self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2 = self.a_alpha_and_derivatives(self.T, pure_a_alphas=pure_a_alphas)
            else:
                self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2 = self.a_alpha_and_derivatives(self.T, pure_a_alphas=pure_a_alphas)
                self.P = R*self.T/(self.V-self.b) - self.a_alpha/(self.V*self.V + self.delta*self.V + self.epsilon)
            Vs = [self.V, 1.0j, 1.0j]
        else:
            if full_alphas:
                self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2 = self.a_alpha_and_derivatives(self.T, pure_a_alphas=pure_a_alphas)
            else:
                self.a_alpha = self.a_alpha_and_derivatives(self.T, full=False, pure_a_alphas=pure_a_alphas)
                self.da_alpha_dT, self.d2a_alpha_dT2 = -5e-3, 1.5e-5
            self.raw_volumes = Vs = self.volume_solutions(self.T, self.P, self.b, self.delta, self.epsilon, self.a_alpha)
        self.set_from_PT(Vs, only_l=only_l, only_g=only_g)

    def resolve_full_alphas(self):
        '''Generic method to resolve the eos with fully calculated alpha
        derviatives.
        '''
        self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2 = self.a_alpha_and_derivatives(self.T, full=True, pure_a_alphas=False)
        self.set_from_PT(self.raw_volumes, only_l=hasattr(self, 'V_l'), only_g=hasattr(self, 'V_g'))
        
    def set_from_PT(self, Vs, only_l=False, only_g=False):
        '''Counts the number of real volumes in `Vs`, and determines what to do.
        If there is only one real volume, the method 
        `set_properties_from_solution` is called with it. If there are
        two real volumes, `set_properties_from_solution` is called once with  
        each volume. The phase is returned by `set_properties_from_solution`, 
        and the volumes is set to either `V_l` or `V_g` as appropriate. 

        Parameters
        ----------
        Vs : list[float]
            Three possible molar volumes, [m^3/mol]
        only_l : bool
            When true, if there is a liquid and a vapor root, only the liquid
            root (and properties) will be set.
        only_g : bool
            When true, if there is a liquid and a vapor root, only the vapor
            root (and properties) will be set.
        
        Notes
        -----
        An optimizatino attempt was made to remove min() and max() from this
        function; that is indeed possible, but the check for handling if there
        are two or three roots makes it not worth it.
        '''
        good_roots = [i.real for i in Vs if i.imag == 0.0 and i.real > 0.0]
        good_root_count = len(good_roots)
            # All roots will have some imaginary component; ignore them if > 1E-9 (when using a solver that does not strip them)
#        good_roots = [i.real for i in Vs if abs(i.imag) < 1E-9 and i.real > 0.0]
#        good_root_count = len(good_roots)
            
        if good_root_count == 1: 
            self.phase = self.set_properties_from_solution(self.T, self.P,
                                                           good_roots[0], self.b, 
                                                           self.delta, self.epsilon, 
                                                           self.a_alpha, self.da_alpha_dT,
                                                           self.d2a_alpha_dT2)
            
            if self.N == 1 and (
                    (self.multicomponent and (self.Tcs[0] == self.T and self.Pcs[0] == self.P))
                    or (not self.multicomponent and self.Tc == self.T and self.Pc == self.P)):
                
                force_l = not self.phase == 'l'
                force_g = not self.phase == 'g'
                self.set_properties_from_solution(self.T, self.P,
                                                  good_roots[0], self.b, 
                                                  self.delta, self.epsilon, 
                                                  self.a_alpha, self.da_alpha_dT,
                                                  self.d2a_alpha_dT2,
                                                  force_l=force_l,
                                                  force_g=force_g)
                self.phase = 'l/g'
        elif good_root_count > 1:
            V_l, V_g = min(good_roots), max(good_roots)
            
            if not only_g:
                self.set_properties_from_solution(self.T, self.P, V_l, self.b, 
                                                   self.delta, self.epsilon,
                                                   self.a_alpha, self.da_alpha_dT,
                                                   self.d2a_alpha_dT2,
                                                   force_l=True)
            if not only_l:
                self.set_properties_from_solution(self.T, self.P, V_g, self.b, 
                                                   self.delta, self.epsilon,
                                                   self.a_alpha, self.da_alpha_dT,
                                                   self.d2a_alpha_dT2, force_g=True)
            self.phase = 'l/g'
        else:
            # Even in the case of three real roots, it is still the min/max that make sense
            raise Exception('No acceptable roots were found; the roots are %s, T is %s K, P is %s Pa, a_alpha is %s, b is %s' %(str(Vs), str(self.T), str(self.P), str([self.a_alpha]), str([self.b])))


    def set_properties_from_solution(self, T, P, V, b, delta, epsilon, a_alpha, 
                                     da_alpha_dT, d2a_alpha_dT2, quick=True,
                                     force_l=False, force_g=False):
        r'''Sets all interesting properties which can be calculated from an
        EOS alone. Determines which phase the fluid is on its own; for details,
        see `phase_identification_parameter`.
        
        The list of properties set is as follows, with all properties suffixed
        with '_l' or '_g'.
        
        dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP, d2P_dT2, d2P_dV2, d2V_dT2, 
        d2V_dP2, d2T_dV2, d2T_dP2, d2V_dPdT, d2P_dTdV, d2T_dPdV, H_dep, S_dep, 
        beta, kappa, Cp_minus_Cv, V_dep, U_dep, G_dep, A_dep, fugacity, phi, 
        and PIP.

        Parameters
        ----------
        T : float
            Temperature, [K]
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        b : float
            Coefficient calculated by EOS-specific method, [m^3/mol]
        delta : float
            Coefficient calculated by EOS-specific method, [m^3/mol]
        epsilon : float
            Coefficient calculated by EOS-specific method, [m^6/mol^2]
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        da_alpha_dT : float
            Temperature derivative of coefficient calculated by EOS-specific 
            method, [J^2/mol^2/Pa/K]
        d2a_alpha_dT2 : float
            Second temperature derivative of coefficient calculated by  
            EOS-specific method, [J^2/mol^2/Pa/K**2]
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (3x faster) or 
            individual formulas
        
        Returns
        -------
        phase : str
            Either 'l' or 'g'
            
        Notes
        -----
        The individual formulas for the derivatives and excess properties are 
        as follows. For definitions of `beta`, see `isobaric_expansion`;
        for `kappa`, see isothermal_compressibility; for `Cp_minus_Cv`, see
        `Cp_minus_Cv`; for `phase_identification_parameter`, see 
        `phase_identification_parameter`.
        
        First derivatives; in part using the Triple Product Rule [2]_, [3]_:
        
        .. math::
            \left(\frac{\partial P}{\partial T}\right)_V = \frac{R}{V - b}
            - \frac{a \frac{d \alpha{\left (T \right )}}{d T}}{V^{2} + V \delta
            + \epsilon}
            
            \left(\frac{\partial P}{\partial V}\right)_T = - \frac{R T}{\left(
            V - b\right)^{2}} - \frac{a \left(- 2 V - \delta\right) \alpha{
            \left (T \right )}}{\left(V^{2} + V \delta + \epsilon\right)^{2}}
            
            \left(\frac{\partial V}{\partial T}\right)_P =-\frac{
            \left(\frac{\partial P}{\partial T}\right)_V}{
            \left(\frac{\partial P}{\partial V}\right)_T}
            
            \left(\frac{\partial V}{\partial P}\right)_T =-\frac{
            \left(\frac{\partial V}{\partial T}\right)_P}{
            \left(\frac{\partial P}{\partial T}\right)_V}            

            \left(\frac{\partial T}{\partial V}\right)_P = \frac{1}
            {\left(\frac{\partial V}{\partial T}\right)_P}
            
            \left(\frac{\partial T}{\partial P}\right)_V = \frac{1}
            {\left(\frac{\partial P}{\partial T}\right)_V}
            
        Second derivatives with respect to one variable; those of `T` and `V`
        use identities shown in [1]_ and verified numerically:
        
        .. math::
            \left(\frac{\partial^2  P}{\partial T^2}\right)_V =  - \frac{a 
            \frac{d^{2} \alpha{\left (T \right )}}{d T^{2}}}{V^{2} + V \delta 
            + \epsilon}
            
            \left(\frac{\partial^2  P}{\partial V^2}\right)_T = 2 \left(\frac{
            R T}{\left(V - b\right)^{3}} - \frac{a \left(2 V + \delta\right)^{
            2} \alpha{\left (T \right )}}{\left(V^{2} + V \delta + \epsilon
            \right)^{3}} + \frac{a \alpha{\left (T \right )}}{\left(V^{2} + V 
            \delta + \epsilon\right)^{2}}\right)
            
            \left(\frac{\partial^2 T}{\partial P^2}\right)_V = -\left(\frac{
            \partial^2 P}{\partial T^2}\right)_V \left(\frac{\partial P}{
            \partial T}\right)^{-3}_V
            
            \left(\frac{\partial^2 V}{\partial P^2}\right)_T = -\left(\frac{
            \partial^2 P}{\partial V^2}\right)_T \left(\frac{\partial P}{
            \partial V}\right)^{-3}_T
            
            \left(\frac{\partial^2 T}{\partial V^2}\right)_P = -\left[
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) \right]
            \left(\frac{\partial P}{\partial T}\right)^{-2}_V
            + \left[\left(\frac{\partial^2 P}{\partial T\partial V}\right)
            \left(\frac{\partial P}{\partial T}\right)_V 
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V\right]
            \left(\frac{\partial P}{\partial T}\right)_V^{-3}
            \left(\frac{\partial P}{\partial V}\right)_T

            \left(\frac{\partial^2 V}{\partial T^2}\right)_P = -\left[
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) \right]
            \left(\frac{\partial P}{\partial V}\right)^{-2}_T
            + \left[\left(\frac{\partial^2 P}{\partial T\partial V}\right)
            \left(\frac{\partial P}{\partial V}\right)_T 
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T\right]
            \left(\frac{\partial P}{\partial V}\right)_T^{-3}
            \left(\frac{\partial P}{\partial T}\right)_V

                        
        Second derivatives with respect to the other two variables; those of 
        `T` and `V` use identities shown in [1]_ and verified numerically:

        .. math::
            \left(\frac{\partial^2 P}{\partial T \partial V}\right) = - \frac{
            R}{\left(V - b\right)^{2}} + \frac{a \left(2 V + \delta\right) 
            \frac{d \alpha{\left (T \right )}}{d T}}{\left(V^{2} + V \delta 
            + \epsilon\right)^{2}}
           
           \left(\frac{\partial^2 T}{\partial P\partial V}\right) = 
            - \left[\left(\frac{\partial^2 P}{\partial T \partial V}\right)
            \left(\frac{\partial P}{\partial T}\right)_V
            - \left(\frac{\partial P}{\partial V}\right)_T
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V
            \right]\left(\frac{\partial P}{\partial T}\right)_V^{-3}

            \left(\frac{\partial^2 V}{\partial T\partial P}\right) = 
            - \left[\left(\frac{\partial^2 P}{\partial T \partial V}\right)
            \left(\frac{\partial P}{\partial V}\right)_T
            - \left(\frac{\partial P}{\partial T}\right)_V
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T
            \right]\left(\frac{\partial P}{\partial V}\right)_T^{-3}

        Excess properties
            
        .. math::
            H_{dep} = \int_{\infty}^V \left[T\frac{\partial P}{\partial T}_V 
            - P\right]dV + PV - RT= P V - R T + \frac{2}{\sqrt{
            \delta^{2} - 4 \epsilon}} \left(T a \frac{d \alpha{\left (T \right 
            )}}{d T}  - a \alpha{\left (T \right )}\right) \operatorname{atanh}
            {\left (\frac{2 V + \delta}{\sqrt{\delta^{2} - 4 \epsilon}} 
            \right)}

            S_{dep} = \int_{\infty}^V\left[\frac{\partial P}{\partial T} 
            - \frac{R}{V}\right] dV + R\log\frac{PV}{RT} = - R \log{\left (V 
            \right )} + R \log{\left (\frac{P V}{R T} \right )} + R \log{\left
            (V - b \right )} + \frac{2 a \frac{d\alpha{\left (T \right )}}{d T}
            }{\sqrt{\delta^{2} - 4 \epsilon}} \operatorname{atanh}{\left (\frac
            {2 V + \delta}{\sqrt{\delta^{2} - 4 \epsilon}} \right )}
        
            V_{dep} = V - \frac{RT}{P}
            
            U_{dep} = H_{dep} - P V_{dep}
            
            G_{dep} = H_{dep} - T S_{dep}
            
            A_{dep} = U_{dep} - T S_{dep}
            
            \text{fugacity} = P\exp\left(\frac{G_{dep}}{RT}\right)
            
            \phi = \frac{\text{fugacity}}{P}
            
            C_{v, dep} = T\int_\infty^V \left(\frac{\partial^2 P}{\partial 
            T^2}\right) dV = - T a \left(\sqrt{\frac{1}{\delta^{2} - 4 
            \epsilon}} \log{\left (V - \frac{\delta^{2}}{2} \sqrt{\frac{1}{
            \delta^{2} - 4 \epsilon}} + \frac{\delta}{2} + 2 \epsilon \sqrt{
            \frac{1}{\delta^{2} - 4 \epsilon}} \right )} - \sqrt{\frac{1}{
            \delta^{2} - 4 \epsilon}} \log{\left (V + \frac{\delta^{2}}{2} 
            \sqrt{\frac{1}{\delta^{2} - 4 \epsilon}} + \frac{\delta}{2} 
            - 2 \epsilon \sqrt{\frac{1}{\delta^{2} - 4 \epsilon}} \right )}
            \right) \frac{d^{2} \alpha{\left (T \right )} }{d T^{2}}  
            
            C_{p, dep} = (C_p-C_v)_{\text{from EOS}} + C_{v, dep} - R
            
            
        References
        ----------
        .. [1] Thorade, Matthis, and Ali Saadat. "Partial Derivatives of 
           Thermodynamic State Properties for Dynamic Simulation." 
           Environmental Earth Sciences 70, no. 8 (April 10, 2013): 3497-3503.
           doi:10.1007/s12665-013-2394-z.
        .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th 
           edition. New York: McGraw-Hill Professional, 2000.
        .. [3] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        '''
        (dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP, 
            d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2,
            d2V_dPdT, d2P_dTdV, d2T_dPdV,
            H_dep, S_dep, Cv_dep) = self.derivatives_and_departures(T, P, V, b, delta, epsilon, a_alpha, da_alpha_dT, d2a_alpha_dT2, quick=quick)
        
        RT = R*T
        RT_inv = 1.0/RT
        P_inv = 1.0/P
        V_inv = 1.0/V
        Z = P*V*RT_inv
        
        beta = dV_dT*V_inv # isobaric_expansion(V, dV_dT)
        kappa = -dV_dP*V_inv # isothermal_compressibility(V, dV_dP)
        Cp_m_Cv = -T*dP_dT*dP_dT*dV_dP # Cp_minus_Cv(T, dP_dT, dP_dV)
        
        Cp_dep = Cp_m_Cv + Cv_dep - R
        
        TS_dep = T*S_dep
        V_dep = V - RT*P_inv      
        U_dep = H_dep - P*V_dep
        G_dep = H_dep - TS_dep
        A_dep = U_dep - TS_dep
        try:
            fugacity = P*exp(G_dep*RT_inv)
        except OverflowError:
            fugacity = P*trunc_exp(G_dep*RT_inv, trunc=1e308)
        phi = fugacity*P_inv
  
        PIP = V*(d2P_dTdV*dT_dP - d2P_dV2*dV_dP) # phase_identification_parameter(V, dP_dT, dP_dV, d2P_dV2, d2P_dTdV)

      
         # 1 + 1e-14 - allow a few dozen unums of toleranve to keep ideal gas model a gas
        if force_l or (not force_g and PIP > 1.00000000000001):
            self.V_l, self.Z_l = V, Z
            self.beta_l, self.kappa_l = beta, kappa
            self.PIP_l, self.Cp_minus_Cv_l = PIP, Cp_m_Cv
            
            self.dP_dT_l, self.dP_dV_l, self.dV_dT_l = dP_dT, dP_dV, dV_dT
            self.dV_dP_l, self.dT_dV_l, self.dT_dP_l = dV_dP, dT_dV, dT_dP
            
            self.d2P_dT2_l, self.d2P_dV2_l = d2P_dT2, d2P_dV2
            self.d2V_dT2_l, self.d2V_dP2_l = d2V_dT2, d2V_dP2
            self.d2T_dV2_l, self.d2T_dP2_l = d2T_dV2, d2T_dP2
                        
            self.d2V_dPdT_l, self.d2P_dTdV_l, self.d2T_dPdV_l = d2V_dPdT, d2P_dTdV, d2T_dPdV
            
            self.H_dep_l, self.S_dep_l, self.V_dep_l = H_dep, S_dep, V_dep, 
            self.U_dep_l, self.G_dep_l, self.A_dep_l = U_dep, G_dep, A_dep, 
            self.fugacity_l, self.phi_l = fugacity, phi
            self.Cp_dep_l, self.Cv_dep_l = Cp_dep, Cv_dep
            return 'l'
        else:
            self.V_g, self.Z_g = V, Z
            self.beta_g, self.kappa_g = beta, kappa
            self.PIP_g, self.Cp_minus_Cv_g = PIP, Cp_m_Cv
            
            self.dP_dT_g, self.dP_dV_g, self.dV_dT_g = dP_dT, dP_dV, dV_dT
            self.dV_dP_g, self.dT_dV_g, self.dT_dP_g = dV_dP, dT_dV, dT_dP
            
            self.d2P_dT2_g, self.d2P_dV2_g = d2P_dT2, d2P_dV2
            self.d2V_dT2_g, self.d2V_dP2_g = d2V_dT2, d2V_dP2
            self.d2T_dV2_g, self.d2T_dP2_g = d2T_dV2, d2T_dP2
            
            self.d2V_dPdT_g, self.d2P_dTdV_g, self.d2T_dPdV_g = d2V_dPdT, d2P_dTdV, d2T_dPdV
            
            self.H_dep_g, self.S_dep_g, self.V_dep_g = H_dep, S_dep, V_dep, 
            self.U_dep_g, self.G_dep_g, self.A_dep_g = U_dep, G_dep, A_dep, 
            self.fugacity_g, self.phi_g = fugacity, phi
            self.Cp_dep_g, self.Cv_dep_g = Cp_dep, Cv_dep
            return 'g'

    def a_alpha_and_derivatives(self, T, full=True, quick=True,
                                pure_a_alphas=True):
        '''Dummy method to calculate `a_alpha` and its first and second
        derivatives. Should be implemented with the same function signature in 
        each EOS variant; this only raises a NotImplemented Exception.
        Should return 'a_alpha', 'da_alpha_dT', and 'd2a_alpha_dT2'.

        For use in `solve_T`, returns only `a_alpha` if `full` is False.
        
        Parameters
        ----------
        T : float
            Temperature, [K]
        full : bool, optional
            If False, calculates and returns only `a_alpha`, [-]
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (3x faster) or 
            individual formulas, [-]
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
        '''
        return self.a_alpha_and_derivatives_pure(T=T, full=full, quick=quick)
    
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        raise NotImplemented('a_alpha and its first and second derivatives \
should be calculated by this method, in a user subclass.')

    def solve_T(self, P, V, quick=True):
        '''Generic method to calculate `T` from a specified `P` and `V`.
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
            Whether to use a SymPy cse-derived expression (3x faster) or 
            individual formulas - not applicable where a numerical solver is
            used.

        Returns
        -------
        T : float
            Temperature, [K]
        '''
        denominator_inv = 1.0/(V*V + self.delta*V + self.epsilon)
        V_minus_b_inv = 1.0/(V-self.b)
        self.no_T_spec = True
        
        # dP_dT could be added to use a derivative-based method, however it is
        # quite costly in comparison to the extra evaluations because it
        # requires the temperature derivative of da_alpha_dT
        def to_solve(T):
            a_alpha = self.a_alpha_and_derivatives(T, full=False, quick=False)
            P_calc = R*T*V_minus_b_inv - a_alpha*denominator_inv
            err = P_calc - P
            return err
        T_guess_ig = P*V*R_inv
        T_guess_liq = P*V*R_inv*1000.0 # Compressibility factor of 0.001 for liquids
        err_ig = to_solve(T_guess_ig)
        err_liq = to_solve(T_guess_liq)
        
        if err_ig*err_liq < 0.0:
            return brenth(to_solve, T_guess_ig, T_guess_liq, xtol=1e-12,
                          fa=err_ig, fb=err_liq)
        else:
            if abs(err_ig) < abs(err_liq):
                T_guess = T_guess_ig
                f0 = err_ig
            else:
                T_guess = T_guess_liq
                f0 = err_liq
            # T_guess = self.Tc*0.5
            return secant(to_solve, T_guess, low=1e-12, xtol=1e-12, f0=f0)

    @staticmethod
    def volume_solutions(T, P, b, delta, epsilon, a_alpha, quick=True):
        r'''Solution of this form of the cubic EOS in terms of volumes. Returns
        three values, all with some complex part.  

        Parameters
        ----------
        T : float
            Temperature, [K]
        P : float
            Pressure, [Pa]
        b : float
            Coefficient calculated by EOS-specific method, [m^3/mol]
        delta : float
            Coefficient calculated by EOS-specific method, [m^3/mol]
        epsilon : float
            Coefficient calculated by EOS-specific method, [m^6/mol^2]
        a_alpha : float
            Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (3x faster) or 
            individual formulas

        Returns
        -------
        Vs : list[float]
            Three possible molar volumes, [m^3/mol]
            
        Notes
        -----
        Using explicit formulas, as can be derived in the following example,
        is faster than most numeric root finding techniques, and
        finds all values explicitly. It takes several seconds.
        
        >>> from sympy import *
        >>> P, T, V, R, b, a, delta, epsilon, alpha = symbols('P, T, V, R, b, a, delta, epsilon, alpha')
        >>> Tc, Pc, omega = symbols('Tc, Pc, omega')
        >>> CUBIC = R*T/(V-b) - a*alpha/(V*V + delta*V + epsilon) - P
        >>> #solve(CUBIC, V)
        
        Note this approach does not have the same issues as formulas using trig
        functions or numerical routines.
        
        References
        ----------
        .. [1] Zhi, Yun, and Huen Lee. "Fallibility of Analytic Roots of Cubic 
           Equations of State in Low Temperature Region." Fluid Phase 
           Equilibria 201, no. 2 (September 30, 2002): 287-94. 
           https://doi.org/10.1016/S0378-3812(02)00072-9.

        '''
#        RT_inv = R_inv/T
#        P_RT_inv = P*RT_inv
#        eta = b
#        B = b*P_RT_inv
#        deltas = delta*P_RT_inv
#        thetas = a_alpha*P_RT_inv*RT_inv
#        epsilons = epsilon*P_RT_inv*P_RT_inv
#        etas = eta*P_RT_inv
#        
#        a = 1.0
#        b2 = (deltas - B - 1.0)
#        c = (thetas + epsilons - deltas*(B + 1.0))
#        d = -(epsilons*(B + 1.0) + thetas*etas)
#        open('bcd.txt', 'a').write('\n%s' %(str([float(b2), float(c), float(d)])))
        
        
        
        
        x24 = 1.73205080756887729352744634151j + 1.
        x24_inv = 0.25 - 0.433012701892219323381861585376j
        x26 = -1.73205080756887729352744634151j + 1.
        x26_inv = 0.25 + 0.433012701892219323381861585376j
        # Changing over to the inverse constants changes some dew point results
        if quick:
            x0 = 1./P
            x1 = P*b
            x2 = R*T
            x3 = P*delta
            x4 = x1 + x2 - x3
            x5 = x0*x4
            x6 = a_alpha*b
            x7 = epsilon*x1
            x8 = epsilon*x2
            x9 = x0*x0
            x10 = P*epsilon
            x11 = delta*x1
            x12 = delta*x2
#            x13 = 3.*a_alpha
#            x14 = 3.*x10
#            x15 = 3.*x11
#            x16 = 3.*x12
            x17 = -x4
            x17_2 = x17*x17
            x18 = x0*x17_2
            tm1 = x12 - a_alpha + (x11  - x10)
#            print(x11, x12, a_alpha, x10)
            t0 = x6 + x7 + x8
            t1 = (3.0*tm1  + x18) # custom vars
#            t1 = (-x13 - x14 + x15 + x16 + x18) # custom vars
            t2 = (9.*x0*x17*tm1 + 2.0*x17_2*x17*x9
                     - 27.*t0)
            
            x4x9  = x4*x9
            x19 = ((-13.5*x0*t0 - 4.5*x4x9*tm1
                   - x4*x4x9*x5
                    + 0.5*((x9*(-4.*x0*t1*t1*t1 + t2*t2))+0.0j)**0.5
                    )+0.0j)**third
            
            x20 = -t1/x19#
            x22 = x5 + x5
            x25 = 4.*x0*x20
            return ((x0*x20 - x19 + x5)*third,
                    (x19*x24 + x22 - x25*x24_inv)*sixth,
                    (x19*x26 + x22 - x25*x26_inv)*sixth)
        else:
            return (-(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)/(3*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)) - (sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)/3 - (-P*b + P*delta - R*T)/(3*P),
                     -(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)/(3*(-1/2 - sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)) - (-1/2 - sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)/3 - (-P*b + P*delta - R*T)/(3*P),
                     -(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)/(3*(-1/2 + sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)) - (-1/2 + sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)/3 - (-P*b + P*delta - R*T)/(3*P))

    # volume_solutions_Cardano
    @staticmethod
    def volume_solutions(T, P, b, delta, epsilon, a_alpha, quick=True):
        RT_inv = R_inv/T
        P_RT_inv = P*RT_inv
#        eta = b
        B = etas = b*P_RT_inv
        deltas = delta*P_RT_inv
        thetas = a_alpha*P_RT_inv*RT_inv
        epsilons = epsilon*P_RT_inv*P_RT_inv
        
#        a = 1.0
        b = (deltas - B - 1.0)
        c = (thetas + epsilons - deltas*(B + 1.0))
        d = -(epsilons*(B + 1.0) + thetas*etas)
#        print(b, c, d)
        roots = roots_cubic(1.0, b, c, d)
#        roots = np.roots([a, b, c, d]).tolist()
        RT_P = R*T/P
        return [V*RT_P for V in roots]

    # validation method
    @staticmethod
    def volume_solutions_bench(T, P, b, delta, epsilon, a_alpha, quick=True):
        RT_inv = R_inv/T
        P_RT_inv = P*RT_inv
        eta = b
        B = b*P_RT_inv
        deltas = delta*P_RT_inv
        thetas = a_alpha*P_RT_inv*RT_inv
        epsilons = epsilon*P_RT_inv*P_RT_inv
        etas = eta*P_RT_inv
        
        a = 1.0
        b = (deltas - B - 1.0)
        c = (thetas + epsilons - deltas*(B + 1.0))
        d = -(epsilons*(B + 1.0) + thetas*etas)
        RT_P = R*T/P
        roots = roots_cubic(a, b, c, d)
        
        def trim_root(x, tol=1e-6):
            x = np.array(x)
            vals = abs(x.imag) < abs(x.real)*tol
            try:
                x.imag[vals] = 0
            except:
                pass
            return x     
        
        fast = trim_root(roots)
        slow = trim_root(np.roots([a, b, c, d]))

        fast = np.sort(fast)
        slow = np.sort(slow)
        if np.sign(slow[1].imag) != np.sign(fast[1].imag):
            fast[1], fast[2] = fast[2], fast[1]
        try:
            from numpy.testing import assert_allclose
            assert_allclose(fast, slow, rtol=1e-7)
        except:
            ratio = np.real_if_close(np.array(fast)/np.array(slow), tol=1e6)
            print('root fail', ratio, [b, c, d])
                
        return [V*RT_P for V in roots]


    def derivatives_and_departures(self, T, P, V, b, delta, epsilon, a_alpha, da_alpha_dT, d2a_alpha_dT2, quick=True):
        
        dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep = (
        self.main_derivatives_and_departures(T, P, V, b, delta, epsilon, 
                                             a_alpha, da_alpha_dT, 
                                             d2a_alpha_dT2, quick=quick))
        try:
            inverse_dP_dV = 1.0/dP_dV
        except ZeroDivisionError:
            inverse_dP_dV = inf
        dT_dP = 1./dP_dT

        dV_dT = -dP_dT*inverse_dP_dV
        dV_dP = -dV_dT*dT_dP 
        dT_dV = 1./dV_dT
                
        
        inverse_dP_dV2 = inverse_dP_dV*inverse_dP_dV
        inverse_dP_dV3 = inverse_dP_dV*inverse_dP_dV2
        
        inverse_dP_dT2 = dT_dP*dT_dP
        inverse_dP_dT3 = inverse_dP_dT2*dT_dP
        
        d2V_dP2 = -d2P_dV2*inverse_dP_dV3
        d2T_dP2 = -d2P_dT2*inverse_dP_dT3
        
        d2T_dV2 = (-(d2P_dV2*dP_dT - dP_dV*d2P_dTdV)*inverse_dP_dT2
                   +(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3*dP_dV)
        d2V_dT2 = (-(d2P_dT2*dP_dV - dP_dT*d2P_dTdV)*inverse_dP_dV2
                   +(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*inverse_dP_dV3*dP_dT)

        d2V_dPdT = -(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*inverse_dP_dV3
        d2T_dPdV = -(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*inverse_dP_dT3

        # TODO return one large tuple - quicker, constructing the lists is slow
#        return ([dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP], 
#                [d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2],
#                [d2V_dPdT, d2P_dTdV, d2T_dPdV],
#                [H_dep, S_dep, Cv_dep])
        return (dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP, 
                d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2,
                d2V_dPdT, d2P_dTdV, d2T_dPdV,
                H_dep, S_dep, Cv_dep)



    @property
    def sorted_volumes(self):
        r'''List of lexicographically-sorted molar volumes available from the
        root finding algorithm used to solve the PT point. The convention of 
        sorting lexicographically comes from numpy's handling of complex 
        numbers, which python does not define. This method was added to 
        facilitate testing, as the volume solution method changes over time 
        and the ordering does as well.

        Examples
        --------
        >>> PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6).sorted_volumes
        [(0.00013022212513965896+0j), (0.0011236313134682665-0.0012926967234386064j), (0.0011236313134682665+0.0012926967234386064j)]
        '''
        sort_fun = lambda x: (x.real, x.imag)
        return sorted(self.raw_volumes, key=sort_fun)
    
    
    @staticmethod
    def main_derivatives_and_departures(T, P, V, b, delta, epsilon, a_alpha,
                                        da_alpha_dT, d2a_alpha_dT2, quick=True):
        if not quick:
            return GCEOS.main_derivatives_and_departures(T, P, V, b, delta, 
                                                         epsilon, a_alpha,
                                                         da_alpha_dT,
                                                         d2a_alpha_dT2)
        epsilon2 = epsilon + epsilon
        x0 = 1.0/(V - b)
        x1 = 1.0/(V*(V + delta) + epsilon)
        x3 = R*T
        x4 = x0*x0
        x5 = V + V + delta
        x6 = x1*x1
        x7 = a_alpha*x6
        x8 = P*V
        x9 = delta*delta
        x10 = x9 - epsilon2 - epsilon2
        try:
            x11 = x10**-0.5
        except ZeroDivisionError:
            # Needed for ideal gas model
            x11 = 0.0
        x11_half = 0.5*x11
        x12 = 2.*x11*catanh(x11*x5).real # Possible to use a catan, but then a complex division and sq root is needed too
        x14 = 0.5*x5
        x15 = epsilon2*x11
        x16 = x11_half*x9
        x17 = x5*x6
        dP_dT = R*x0 - da_alpha_dT*x1
        dP_dV = x5*x7 - x3*x4
        d2P_dT2 = -d2a_alpha_dT2*x1
        
        d2P_dV2 = (x7 + x3*x4*x0 - a_alpha*x5*x17*x1)
        d2P_dV2 += d2P_dV2
        
        d2P_dTdV = da_alpha_dT*x17 - R*x4
        H_dep = x12*(T*da_alpha_dT - a_alpha) - x3 + x8
        
        t1 = (x3*x0/P)
        S_dep = -R_2*log(t1*t1) + da_alpha_dT*x12  # Consider Real part of the log only via log(x**2)/2 = Re(log(x))
        
        x18 = x16 - x15
        x19 = (x14 + x18)/(x14 - x18)
        Cv_dep = T*d2a_alpha_dT2*x11_half*(log(x19*x19)) # Consider Real part of the log only via log(x**2)/2 = Re(log(x))
        return dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep

    @staticmethod
    def main_derivatives_and_departures_slow(T, P, V, b, delta, epsilon, a_alpha,
                                        da_alpha_dT, d2a_alpha_dT2):
        dP_dT = R/(V - b) - da_alpha_dT/(V**2 + V*delta + epsilon)
        dP_dV = -R*T/(V - b)**2 - (-2*V - delta)*a_alpha/(V**2 + V*delta + epsilon)**2
        d2P_dT2 = -d2a_alpha_dT2/(V**2 + V*delta + epsilon)
        d2P_dV2 = 2*(R*T/(V - b)**3 - (2*V + delta)**2*a_alpha/(V**2 + V*delta + epsilon)**3 + a_alpha/(V**2 + V*delta + epsilon)**2)
        d2P_dTdV = -R/(V - b)**2 + (2*V + delta)*da_alpha_dT/(V**2 + V*delta + epsilon)**2
        H_dep = P*V - R*T + 2*(T*da_alpha_dT - a_alpha)*catanh((2*V + delta)/sqrt(delta**2 - 4*epsilon)).real/sqrt(delta**2 - 4*epsilon)
        S_dep = -R*log(V) + R*log(P*V/(R*T)) + R*log(V - b) + 2*da_alpha_dT*catanh((2*V + delta)/sqrt(delta**2 - 4*epsilon)).real/sqrt(delta**2 - 4*epsilon)
        Cv_dep = -T*(sqrt(1/(delta**2 - 4*epsilon))*log(V - delta**2*sqrt(1/(delta**2 - 4*epsilon))/2 + delta/2 + 2*epsilon*sqrt(1/(delta**2 - 4*epsilon))) - sqrt(1/(delta**2 - 4*epsilon))*log(V + delta**2*sqrt(1/(delta**2 - 4*epsilon))/2 + delta/2 - 2*epsilon*sqrt(1/(delta**2 - 4*epsilon))))*d2a_alpha_dT2
        return dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep

    def Tsat(self, P, polish=False):
        r'''Generic method to calculate the temperature for a specified 
        vapor pressure of the pure fluid.
        This is simply a bounded solver running between `0.2Tc` and `Tc` on the
        `Psat` method.
        
        Parameters
        ----------
        P : float
            Vapor pressure, [Pa]
        polish : bool, optional
            Whether to attempt to use a numerical solver to make the solution
            more precise or not

        Returns
        -------
        Tsat : float
            Temperature of saturation, [K]
            
        Notes
        -----
        It is recommended not to run with `polish=True`, as that will make the
        calculation much slower.
        '''
        def to_solve(T):
            err = self.Psat(T, polish=polish) - P
#            print(err, T)
#            derr_dT = self.dPsat_dT(T)
            return err#, derr_dT
#            return copysign(log(abs(err)), err)
        # Outstanding improvements to do: Better guess; get NR working;
        # see if there is a general curve
        
        guess = -5.4*self.Tc/(1.0*log(P/self.Pc) - 5.4)
#        return newton(to_solve, guess, fprime=True, ytol=1e-6, high=self.Pc)
#        return newton(to_solve, guess, ytol=1e-6, high=self.Pc)
        try:
            return brenth(to_solve, max(guess*.7, 0.2*self.Tc), min(self.Tc, guess*1.3))
        except:
            try:
                return brenth(to_solve, 0.2*self.Tc, self.Tc)
            except:
                return brenth(to_solve, 0.2*self.Tc, self.Tc*1.5)
            
    def Psat(self, T, polish=False):
        r'''Generic method to calculate vapor pressure for a specified `T`.
        
        From Tc to 0.32Tc, uses a 10th order polynomial of the following form:
        
        .. math::
            \ln\frac{P_r}{T_r} = \sum_{k=0}^{10} C_k\left(\frac{\alpha}{T_r}
            -1\right)^{k}
                    
        If `polish` is True, SciPy's `newton` solver is launched with the 
        calculated vapor pressure as an initial guess in an attempt to get more
        accuracy. This may not converge however.
        
        Results above the critical temperature are meaningless. A first-order 
        polynomial is used to extrapolate under 0.32 Tc; however, there is 
        normally not a volume solution to the EOS which can produce that
        low of a pressure.
        
        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to attempt to use a numerical solver to make the solution
            more precise or not

        Returns
        -------
        Psat : float
            Vapor pressure, [Pa]
            
        Notes
        -----
        EOSs sharing the same `b`, `delta`, and `epsilon` have the same
        coefficient sets.
                
        Form for the regression is inspired from [1]_.
        
        No volume solution is needed when `polish=False`; the only external 
        call is for the value of `a_alpha`.
                    
        References
        ----------
        .. [1] Soave, G. "Direct Calculation of Pure-Compound Vapour Pressures 
           through Cubic Equations of State." Fluid Phase Equilibria 31, no. 2 
           (January 1, 1986): 203-7. doi:10.1016/0378-3812(86)90013-0. 
        '''
        # WARNING - For compounds whose a_alpha (x)values extend too high,
        # this method is inaccurate.
        # TODO: find way to extend the range? Multiple compounds?
        
        if T == self.Tc:
            return self.Pc
        alpha = self.a_alpha_and_derivatives(T, full=False)/self.a
        Tr = T/self.Tc
        x = alpha/Tr - 1.
                        
        if Tr > 0.999:
            y = horner(self.Psat_coeffs_critical, x)
            Psat = y*Tr*self.Pc
        else:
            if Tr < 0.32:
                y = horner(self.Psat_coeffs_limiting, x)
            else:
                y = chebval(self.Psat_cheb_constant_factor[1]*(x + self.Psat_cheb_constant_factor[0]), self.Psat_cheb_coeffs)
            try:
                Psat = exp(y)*Tr*self.Pc
            except OverflowError:
                # coefficients sometimes overflow before T is lowered to 0.32Tr
                polish = False # There is no solution available to polish
                Psat = 0
        
        if polish:
            if T > self.Tc:
                raise ValueError("Cannot solve for equifugacity condition "
                                 "beyond critical temperature")
            def to_solve_newton(P):
                # For use by newton. Only supports initialization with Tc, Pc and omega
                # ~200x slower and not guaranteed to converge (primary issue is one phase)
                # not existing
                e = self.__class__(Tc=self.Tc, Pc=self.Pc, omega=self.omega, T=T, P=P)
                try:
                    fugacity_l = e.fugacity_l
                except AttributeError as e:
                    raise e
                
                try:
                    fugacity_g = e.fugacity_g
                except AttributeError as e:
                    raise e
                
                err = fugacity_l - fugacity_g
                
                d_err_d_P = e.dfugacity_dP_l - e.dfugacity_dP_g
#                print('err', err, 'd_err_d_P', d_err_d_P, 'P', P)
                return err, d_err_d_P
            try:
                Psat = newton(to_solve_newton, Psat, high=self.Pc, fprime=True, 
                              xtol=1e-12, ytol=1e-6, require_eval=False)
            except:
                def to_solve_bisect(P):
                    e = self.__class__(Tc=self.Tc, Pc=self.Pc, omega=self.omega, T=T, P=P)
                    try:
                        fugacity_l = e.fugacity_l
                    except AttributeError as e:
                        return 1e20
                    
                    try:
                        fugacity_g = e.fugacity_g
                    except AttributeError as e:
                        return -1e20
                    err = fugacity_l - fugacity_g
#                    print(err, 'err', 'P', P)
                    return err
                Psat = bisect(to_solve_bisect, .98*Psat, 1.02*Psat, 
                              maxiter=1000)
                    
        return Psat

    def dPsat_dT(self, T):
        r'''Generic method to calculate the temperature derivative of vapor 
        pressure for a specified `T`. Implements the analytical derivative
        of the three polynomials described in `Psat`.
        
        As with `Psat`, results above the critical temperature are meaningless. 
        The first-order polynomial which is used to calculate it under 0.32 Tc
        may not be physicall meaningful, due to there normally not being a 
        volume solution to the EOS which can produce that low of a pressure.
        
        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        dPsat_dT : float
            Derivative of vapor pressure with respect to temperature, [Pa/K]
            
        Notes
        -----
        There is a small step change at 0.32 Tc for all EOS due to the two
        switch between polynomials at that point.
        
        Useful for calculating enthalpy of vaporization with the Clausius
        Clapeyron Equation. Derived with SymPy's diff and cse.
        '''
        # WARNING - For compounds whose a_alpha (x)values extend too high,
        # this method is inaccurate.
        # TODO: find way to extend the range? Multiple compounds?
        a_alphas = self.a_alpha_and_derivatives(T)
        Tc, alpha, d_alpha_dT = self.Tc, a_alphas[0]/self.a, a_alphas[1]/self.a
        Tc_inv = 1.0/Tc
        T_inv = 1.0/T
        Tr = T*Tc_inv
        Pc = self.Pc
        if Tr < 0.32:
            c = self.Psat_coeffs_limiting
            return self.Pc*T*c[0]*(self.Tc*d_alpha_dT/T - self.Tc*alpha/(T*T)
                              )*exp(c[0]*(-1. + self.Tc*alpha/T) + c[1]
                              )/self.Tc + self.Pc*exp(c[0]*(-1.
                              + self.Tc*alpha/T) + c[1])/self.Tc
        elif Tr > 0.999:
            x = alpha/Tr - 1.
            y = horner(self.Psat_coeffs_critical, x)
            dy_dT = T_inv*(Tc*d_alpha_dT - Tc*alpha*T_inv)*horner(self.Psat_coeffs_critical_der, x)
            return self.Pc*(T*dy_dT*Tc_inv + y*Tc_inv)
        else:
            x = alpha/Tr - 1.
            arg = (self.Psat_cheb_constant_factor[1]*(x + self.Psat_cheb_constant_factor[0]))
            y = chebval(arg, self.Psat_cheb_coeffs)
            
            exp_y = exp(y)
            dy_dT = T_inv*(Tc*d_alpha_dT - Tc*alpha*T_inv)*chebval(arg,
                     self.Psat_cheb_coeffs_der)*self.Psat_cheb_constant_factor[1]
            Psat = Pc*T*exp_y*dy_dT*Tc_inv + Pc*exp_y*Tc_inv
            return Psat
        
    def phi_sat(self, T, polish=True):
        r'''Method to calculate the saturation fugacity coefficient of the
        compound. This does not require solving the EOS itself.
        
        Parameters
        ----------
        T : float
            Temperature, [K]
        polish : bool, optional
            Whether to perform a rigorous calculation or to use a polynomial
            fit, [-]

        Returns
        -------
        phi_sat : float
            Fugacity coefficient along the liquid-vapor saturation line, [-]
            
        Notes
        -----
        Accuracy is generally around 1e-7. If Tr is under 0.32, the rigorous
        method is always used, but a solution may not exist if both phases
        cannot coexist. If Tr is above 1, likewise a solution does not exist.
        '''
        # WARNING - For compounds whose a_alpha (x)values extend too high,
        # this method is inaccurate.
        # TODO: find way to extend the range? Multiple compounds?
        Tr = T/self.Tc
        if polish or not 0.32 <= Tr <= 1.0:
            e = self.to_TP(T=T, P=self.Psat(T), polish=True) # True
            try:
                return e.phi_l
            except:
                return e.phi_g

        alpha = self.a_alpha_and_derivatives(T, full=False)/self.a
        x = alpha/Tr - 1.
        return horner(self.phi_sat_coeffs, x)
        
    def V_l_sat(self, T):
        r'''Method to calculate molar volume of the liquid phase along the
        saturation line.
        
        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        V_l_sat : float
            Liquid molar volume along the saturation line, [m^3/mol]
            
        Notes
        -----
        Computes `Psat`, and then uses `volume_solutions` to obtain the three
        possible molar volumes. The lowest value is returned.
        '''
        Psat = self.Psat(T)
        a_alpha = self.a_alpha_and_derivatives(T, full=False)
        Vs = self.volume_solutions(T, Psat, self.b, self.delta, self.epsilon, a_alpha)
        # Assume we can safely take the Vmax as gas, Vmin as l on the saturation line
        return min([i.real for i in Vs])
    
    def V_g_sat(self, T):
        r'''Method to calculate molar volume of the vapor phase along the
        saturation line.
        
        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        V_g_sat : float
            Gas molar volume along the saturation line, [m^3/mol]
            
        Notes
        -----
        Computes `Psat`, and then uses `volume_solutions` to obtain the three
        possible molar volumes. The highest value is returned.
        '''
        Psat = self.Psat(T)
        a_alpha = self.a_alpha_and_derivatives(T, full=False)
        Vs = self.volume_solutions(T, Psat, self.b, self.delta, self.epsilon, a_alpha)
        # Assume we can safely take the Vmax as gas, Vmin as l on the saturation line
        return max([i.real for i in Vs])
    
    def Hvap(self, T):
        r'''Method to calculate enthalpy of vaporization for a pure fluid from
        an equation of state, without iteration.
        
        .. math::
            \frac{dP^{sat}}{dT}=\frac{\Delta H_{vap}}{T(V_g - V_l)}
        
        Results above the critical temperature are meaningless. A first-order 
        polynomial is used to extrapolate under 0.32 Tc; however, there is 
        normally not a volume solution to the EOS which can produce that
        low of a pressure.
        
        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        Hvap : float
            Increase in enthalpy needed for vaporization of liquid phase along 
            the saturation line, [J/mol]
            
        Notes
        -----
        Calculates vapor pressure and its derivative with `Psat` and `dPsat_dT`
        as well as molar volumes of the saturation liquid and vapor phase in
        the process.
        
        Very near the critical point this provides unrealistic results due to
        `Psat`'s polynomials being insufficiently accurate.
                    
        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        '''
        Psat = self.Psat(T)
        dPsat_dT = self.dPsat_dT(T)
        a_alpha = self.a_alpha_and_derivatives(T, full=False)
        Vs = self.volume_solutions(T, Psat, self.b, self.delta, self.epsilon, a_alpha)
        # Assume we can safely take the Vmax as gas, Vmin as l on the saturation line
        Vs = [i.real for i in Vs]
        V_l, V_g = min(Vs), max(Vs)
        return dPsat_dT*T*(V_g - V_l)

    def to_TP(self, T, P):
        if T != self.T or P != self.P:
            return self.__class__(T=T, P=P, Tc=self.Tc, Pc=self.Pc, omega=self.omega, **self.kwargs)
        else:
            return self

    def to_TV(self, T, V):
        if T != self.T or V != self.V:
            # Only allow creation of new class if volume actually specified
            return self.__class__(T=T, V=V, Tc=self.Tc, Pc=self.Pc, omega=self.omega, **self.kwargs)
        else:
            return self
        
    def to_PV(self, P, V):
        if P != self.P or V != self.V:
            return self.__class__(V=V, P=P, Tc=self.Tc, Pc=self.Pc, omega=self.omega, **self.kwargs)
        else:
            return self
    
    def to(self, T=None, P=None, V=None):
        if T is not None and P is not None:
            return self.to_TP(T, P)
        elif T is not None and V is not None:
            return self.to_TV(T, V)
        elif P is not None and V is not None:
            return self.to_PV(P, V)
        else:
            # Error message
            return self.__class__(T=T, V=V, P=P, Tc=self.Tc, Pc=self.Pc, omega=self.omega, **self.kwargs)
        
        
    @property
    def more_stable_phase(self):
        try:
            if self.G_dep_l < self.G_dep_g:
                return 'l'
            else:
                return 'g'
        except:
            try:
                self.Z_g
                return 'g'
            except:
                return 'l'

        

    def discriminant_at_T_zs(self, P):
        # Only P is allowed to be varied
        RT = R*self.T
        RT6 = RT**6
        x0 = P*P
        x1 = P*self.b + RT
        x2 = self.a_alpha*self.b + self.epsilon*x1
        x3 = P*self.epsilon
        x4 = self.delta*x1
        x5 = -P*self.delta + x1
        x6 = self.a_alpha + x3 - x4
        x2_2 = x2*x2
        x5_2 = x5*x5
        x6_2 = x6*x6
        return x0*(18.0*P*x2*x5*x6 + 4.0*P*(-self.a_alpha - x3 + x4)**3 
                   - 27.0*x0*x2_2 - 4.0*x2*x5_2*x5 + x5_2*x6_2)/RT6

    def V_g_extrapolated(self):
        P_pseudo_mc = sum([self.Pcs[i]*self.zs[i] for i in self.cmps])
        T_pseudo_mc = sum([(self.Tcs[i]*self.Tcs[j])**0.5*self.zs[j]*self.zs[i] 
                           for i in self.cmps for j in self.cmps])
        V_pseudo_mc = (self.Zc*R*T_pseudo_mc)/P_pseudo_mc
        rho_pseudo_mc = 1.0/V_pseudo_mc
        
        # Can take a while to converge
        P_disc = newton(self.discriminant_at_T_zs, self.P, tol=1e-7, maxiter=200)
        if P_disc <= 0.0:
            P_disc = newton(self.discriminant_at_T_zs, self.P*100, tol=1e-7, maxiter=200)
#            P_max = self.P*1000
#            P_disc = brenth(self.discriminant_at_T_zs, self.P*1e-3, P_max, rtol=1e-7, maxiter=200)

        try:
            P_low = max(P_disc - 10.0, 1e-3)
            eos_low = self.to_TP_zs(T=self.T, P=P_low, zs=self.zs)
            rho_low = 1.0/eos_low.V_g
        except:
            P_low = max(P_disc + 10.0, 1e-3)
            eos_low = self.to_TP_zs(T=self.T, P=P_low, zs=self.zs)
            rho_low = 1.0/eos_low.V_g
        
        rho0 = (rho_low + 1.4*rho_pseudo_mc)*0.5
        
        dP_drho = eos_low.dP_drho_g
        rho1 = P_low*((rho_low - 1.4*rho_pseudo_mc) + P_low/dP_drho)
        
        rho2 = -P_low*P_low*((rho_low - 1.4*rho_pseudo_mc)*0.5 + P_low/dP_drho)
        rho_ans = rho0 + rho1/eos_low.P + rho2/(eos_low.P*eos_low.P)
        return 1.0/rho_ans
        
    @property
    def rho_l(self):
        return 1.0/self.V_l
    
    @property
    def rho_g(self):
        return 1.0/self.V_g


    @property
    def dZ_dT_l(self):
        T_inv = 1.0/self.T
        return self.P*R_inv*T_inv*(self.dV_dT_l - self.V_l*T_inv)

    @property
    def dZ_dT_g(self):
        T_inv = 1.0/self.T
        return self.P*R_inv*T_inv*(self.dV_dT_g - self.V_g*T_inv)
    
    @property
    def dZ_dP_l(self):
        return 1.0/(self.T*R)*(self.V_l + self.P*self.dV_dP_l)
    
    @property
    def dZ_dP_g(self):
        return 1.0/(self.T*R)*(self.V_g + self.P*self.dV_dP_g)

    @property
    def d2V_dTdP_l(self):
        return self.d2V_dPdT_l
    
    @property
    def d2V_dTdP_g(self):
        return self.d2V_dPdT_g
    
    @property
    def d2P_dVdT_l(self):
        return self.d2P_dTdV_l

    @property
    def d2P_dVdT_g(self):
        return self.d2P_dTdV_g
    
    @property
    def d2T_dVdP_l(self):
        return self.d2T_dPdV_l
    
    @property
    def d2T_dVdP_g(self):
        return self.d2T_dPdV_g
    
    
    @property
    def dP_drho_l(self):
        r'''Derivative of pressure with respect to molar density for the liquid
        phase, [Pa/(mol/m^3)]
        
        .. math::
            \frac{\partial P}{\partial \rho} = -V^2 \frac{\partial P}{\partial V}        
        '''
        return -self.V_l*self.V_l*self.dP_dV_l 
    
    @property
    def dP_drho_g(self):
        r'''Derivative of pressure with respect to molar density for the gas
        phase, [Pa/(mol/m^3)]
        
        .. math::
            \frac{\partial P}{\partial \rho} = -V^2 \frac{\partial P}{\partial V}        
        '''
        return -self.V_g*self.V_g*self.dP_dV_g 
    
    @property
    def drho_dP_l(self):
        r'''Derivative of molar density with respect to pressure for the liquid
        phase, [(mol/m^3)/Pa]
        
        .. math::
            \frac{\partial \rho}{\partial P} = \frac{-1}{V^2} \frac{\partial V}{\partial P}        
        '''
        return -self.dV_dP_l/(self.V_l*self.V_l)
    
    @property
    def drho_dP_g(self):
        r'''Derivative of molar density with respect to pressure for the gas
        phase, [(mol/m^3)/Pa]
        
        .. math::
            \frac{\partial \rho}{\partial P} = \frac{-1}{V^2} \frac{\partial V}{\partial P}        
        '''
        return -self.dV_dP_g/(self.V_g*self.V_g)
    
    @property
    def d2P_drho2_l(self):
        r'''Second derivative of pressure with respect to molar density for the 
        liquid phase, [Pa/(mol/m^3)^2]
        
        .. math::
            \frac{\partial^2 P}{\partial \rho^2} = -V^2\left(
            -V^2\frac{\partial^2 P}{\partial V^2} - 2V \frac{\partial P}{\partial V}
            \right)
        '''
        return -self.V_l**2*(-self.V_l**2*self.d2P_dV2_l - 2*self.V_l*self.dP_dV_l)

    @property
    def d2P_drho2_g(self):
        r'''Second derivative of pressure with respect to molar density for the 
        gas phase, [Pa/(mol/m^3)^2]
        
        .. math::
            \frac{\partial^2 P}{\partial \rho^2} = -V^2\left(
            -V^2\frac{\partial^2 P}{\partial V^2} - 2V \frac{\partial P}{\partial V}
            \right)
        '''
        return -self.V_g**2*(-self.V_g**2*self.d2P_dV2_g - 2*self.V_g*self.dP_dV_g)

    @property
    def d2rho_dP2_l(self):
        r'''Second derivative of molar density with respect to pressure for the 
        liquid phase, [(mol/m^3)/Pa^2]
        
        .. math::
            \frac{\partial^2 \rho}{\partial P^2} = 
            -\frac{\partial^2 V}{\partial P^2}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial P}\right)^2\frac{1}{V^3}
        '''
        return -self.d2V_dP2_l/self.V_l**2 + 2*self.dV_dP_l**2/self.V_l**3

    @property
    def d2rho_dP2_g(self):
        r'''Second derivative of molar density with respect to pressure for the 
        gas phase, [(mol/m^3)/Pa^2]
        
        .. math::
            \frac{\partial^2 \rho}{\partial P^2} = 
            -\frac{\partial^2 V}{\partial P^2}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial P}\right)^2\frac{1}{V^3}
        '''
        return -self.d2V_dP2_g/self.V_g**2 + 2*self.dV_dP_g**2/self.V_g**3
    
    
    @property
    def dT_drho_l(self):
        r'''Derivative of temperature with respect to molar density for the 
        liquid phase, [K/(mol/m^3)]
        
        .. math::
            \frac{\partial \T}{\partial \rho} = V^2 \frac{\partial T}{\partial V}        
        '''
        return -self.V_l*self.V_l*self.dT_dV_l

    @property
    def dT_drho_g(self):
        r'''Derivative of temperature with respect to molar density for the 
        gas phase, [K/(mol/m^3)]
        
        .. math::
            \frac{\partial \T}{\partial \rho} = V^2 \frac{\partial T}{\partial V}        
        '''
        return -self.V_g*self.V_g*self.dT_dV_g
    
    @property
    def d2T_drho2_l(self):
        r'''Second derivative of temperature with respect to molar density for  
        the liquid phase, [K/(mol/m^3)^2]
        
        .. math::
            \frac{\partial^2 T}{\partial \rho^2} = 
            -V^2(-V^2 \frac{\partial^2 T}{\partial V^2} -2V \frac{\partial T}{\partial V}  )
        '''
        return -self.V_l**2*(-self.V_l**2*self.d2T_dV2_l - 2*self.V_l*self.dT_dV_l)
    
    @property
    def d2T_drho2_g(self):
        r'''Second derivative of temperature with respect to molar density for  
        the gas phase, [K/(mol/m^3)^2]
        
        .. math::
            \frac{\partial^2 T}{\partial \rho^2} = 
            -V^2(-V^2 \frac{\partial^2 T}{\partial V^2} -2V \frac{\partial T}{\partial V}  )
        '''
        return -self.V_g**2*(-self.V_g**2*self.d2T_dV2_g - 2*self.V_g*self.dT_dV_g)


    @property
    def drho_dT_l(self):
        r'''Derivative of molar density with respect to temperature for the 
        liquid phase, [(mol/m^3)/K]
        
        .. math::
            \frac{\partial \rho}{\partial T} = - \frac{1}{V^2}
            \frac{\partial V}{\partial T}        
        '''
        return -self.dV_dT_l/(self.V_l*self.V_l)

    @property
    def drho_dT_g(self):
        r'''Derivative of molar density with respect to temperature for the 
        gas phase, [(mol/m^3)/K]
        
        .. math::
            \frac{\partial \rho}{\partial T} = - \frac{1}{V^2}
            \frac{\partial V}{\partial T}        
        '''
        return -self.dV_dT_g/(self.V_g*self.V_g)
    
    @property
    def d2rho_dT2_l(self):
        r'''Second derivative of molar density with respect to temperature for  
        the liquid phase, [(mol/m^3)/K^2]
        
        .. math::
            \frac{\partial^2 \rho}{\partial T^2} = 
            -\frac{\partial^2 V}{\partial T^2}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial T}\right)^2\frac{1}{V^3}
        '''
        return -self.d2V_dT2_l/self.V_l**2 + 2*self.dV_dT_l**2/self.V_l**3
    
    @property
    def d2rho_dT2_g(self):
        r'''Second derivative of molar density with respect to temperature for  
        the gas phase, [(mol/m^3)/K^2]
        
        .. math::
            \frac{\partial^2 \rho}{\partial T^2} = 
            -\frac{\partial^2 V}{\partial T^2}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial T}\right)^2\frac{1}{V^3}
        '''
        return -self.d2V_dT2_g/self.V_g**2 + 2*self.dV_dT_g**2/self.V_g**3
    
    @property
    def d2P_dTdrho_l(self):
        r'''Derivative of pressure with respect to molar density, and  
        temperature for the liquid phase, [Pa/(K*mol/m^3)]
        
        .. math::
            \frac{\partial^2 P}{\partial \rho\partial T} 
            = -V^2 \frac{\partial^2 P}{\partial T \partial V}        
        '''
        return -(self.V_l*self.V_l)*self.d2P_dTdV_l

    @property
    def d2P_dTdrho_g(self):
        r'''Derivative of pressure with respect to molar density, and  
        temperature for the gas phase, [Pa/(K*mol/m^3)]
        
        .. math::
            \frac{\partial^2 P}{\partial \rho\partial T} 
            = -V^2 \frac{\partial^2 P}{\partial T \partial V}        
        '''
        return -(self.V_g*self.V_g)*self.d2P_dTdV_g

    @property
    def d2T_dPdrho_l(self):
        r'''Derivative of temperature with respect to molar density, and  
        pressure for the liquid phase, [K/(Pa*mol/m^3)]
        
        .. math::
            \frac{\partial^2 T}{\partial \rho\partial P} 
            = -V^2 \frac{\partial^2 T}{\partial P \partial V}        
        '''
        return -(self.V_l*self.V_l)*self.d2T_dPdV_l
    
    @property
    def d2T_dPdrho_g(self):
        r'''Derivative of temperature with respect to molar density, and  
        pressure for the gas phase, [K/(Pa*mol/m^3)]
        
        .. math::
            \frac{\partial^2 T}{\partial \rho\partial P} 
            = -V^2 \frac{\partial^2 T}{\partial P \partial V}        
        '''
        return -(self.V_g*self.V_g)*self.d2T_dPdV_g
    
    @property
    def d2rho_dPdT_l(self):
        r'''Second derivative of molar density with respect to pressure
        and temperature for the liquid phase, [(mol/m^3)/(K*Pa)]
        
        .. math::
            \frac{\partial^2 \rho}{\partial T \partial P} = 
            -\frac{\partial^2 V}{\partial T \partial P}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial T}\right)
            \left(\frac{\partial V}{\partial P}\right)
            \frac{1}{V^3}
        '''
        return -self.d2V_dPdT_l/self.V_l**2 + 2*self.dV_dT_l*self.dV_dP_l/self.V_l**3

    @property
    def d2rho_dPdT_g(self):
        r'''Second derivative of molar density with respect to pressure
        and temperature for the gas phase, [(mol/m^3)/(K*Pa)]
        
        .. math::
            \frac{\partial^2 \rho}{\partial T \partial P} = 
            -\frac{\partial^2 V}{\partial T \partial P}\frac{1}{V^2}
            + 2 \left(\frac{\partial V}{\partial T}\right)
            \left(\frac{\partial V}{\partial P}\right)
            \frac{1}{V^3}
        '''
        return -self.d2V_dPdT_g/self.V_g**2 + 2*self.dV_dT_g*self.dV_dP_g/self.V_g**3
    
    @property
    def dH_dep_dT_l(self):
        r'''Derivative of departure enthalpy with respect to 
        temeprature for the liquid phase, [(J/mol)/K]
        
        .. math::
            \frac{\partial H_{dep, l}}{\partial T} = P \frac{d}{d T} V{\left (T
            \right )} - R + \frac{2 T}{\sqrt{\delta^{2} - 4 \epsilon}} 
                \operatorname{atanh}{\left (\frac{\delta + 2 V{\left (T \right
                )}}{\sqrt{\delta^{2} - 4 \epsilon}} \right )} \frac{d^{2}}{d 
                T^{2}}  \operatorname{a \alpha}{\left (T \right )} + \frac{4
                \left(T \frac{d}{d T} \operatorname{a \alpha}{\left (T \right
                )} - \operatorname{a \alpha}{\left (T \right )}\right) \frac{d}
                {d T} V{\left (T \right )}}{\left(\delta^{2} - 4 \epsilon
                \right) \left(- \frac{\left(\delta + 2 V{\left (T \right )}
                \right)^{2}}{\delta^{2} - 4 \epsilon} + 1\right)}
        '''
        x0 = self.V_l
        x1 = self.dV_dT_l
        x2 = self.a_alpha
        x3 = self.delta*self.delta - 4.0*self.epsilon
        if x3 == 0.0:
            x3 = 1e-100

        x4 = x3**-0.5
        x5 = self.delta + x0 + x0
        x6 = 1.0/x3
        return (self.P*x1 - R + 2.0*self.T*x4*catanh(x4*x5).real*self.d2a_alpha_dT2 
                - 4.0*x1*x6*(self.T*self.da_alpha_dT - x2)/(x5*x5*x6 - 1.0))

    @property
    def dH_dep_dT_g(self):
        r'''Derivative of departure enthalpy with respect to 
        temeprature for the gas phase, [(J/mol)/K]
        
        .. math::
            \frac{\partial H_{dep, g}}{\partial T} = P \frac{d}{d T} V{\left (T
            \right )} - R + \frac{2 T}{\sqrt{\delta^{2} - 4 \epsilon}} 
                \operatorname{atanh}{\left (\frac{\delta + 2 V{\left (T \right
                )}}{\sqrt{\delta^{2} - 4 \epsilon}} \right )} \frac{d^{2}}{d 
                T^{2}}  \operatorname{a \alpha}{\left (T \right )} + \frac{4
                \left(T \frac{d}{d T} \operatorname{a \alpha}{\left (T \right
                )} - \operatorname{a \alpha}{\left (T \right )}\right) \frac{d}
                {d T} V{\left (T \right )}}{\left(\delta^{2} - 4 \epsilon
                \right) \left(- \frac{\left(\delta + 2 V{\left (T \right )}
                \right)^{2}}{\delta^{2} - 4 \epsilon} + 1\right)}
        '''
        x0 = self.V_g
        x1 = self.dV_dT_g
        x2 = self.a_alpha
        x3 = self.delta*self.delta - 4.0*self.epsilon
        if x3 == 0.0:
            x3 = 1e-100
        x4 = x3**-0.5
        x5 = self.delta + x0 + x0
        x6 = 1.0/x3
        return (self.P*x1 - R + 2.0*self.T*x4*catanh(x4*x5).real*self.d2a_alpha_dT2 
                - 4.0*x1*x6*(self.T*self.da_alpha_dT - x2)/(x5*x5*x6 - 1.0))
        
    @property
    def dH_dep_dT_l_V(self):
        r'''Derivative of departure enthalpy with respect to 
        temeprature at constant volume for the liquid phase, [(J/mol)/K]
        
        .. math::
            \left(\frac{\partial H_{dep, l}}{\partial T}\right)_{V} = 
            - R + \frac{2 T 
            \operatorname{atanh}{\left(\frac{2 V_l + \delta}{\sqrt{\delta^{2}
            - 4 \epsilon}} \right)} \frac{d^{2}}{d T^{2}} \operatorname{
                a_{\alpha}}{\left(T \right)}}{\sqrt{\delta^{2} - 4 \epsilon}} 
                + V_l \frac{\partial}{\partial T} P{\left(T,V \right)}
        '''
        T = self.T
        delta, epsilon = self.delta, self.epsilon
        V = self.V_l
        dP_dT = self.dP_dT_l
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100
        return -R + 2.0*T*x0*catanh(x0*(V + V + delta)).real*self.d2a_alpha_dT2 + V*dP_dT

    @property
    def dH_dep_dT_g_V(self):
        r'''Derivative of departure enthalpy with respect to 
        temeprature at constant volume for the gas phase, [(J/mol)/K]
        
        .. math::
            \left(\frac{\partial H_{dep, g}}{\partial T}\right)_{V} = 
            - R + \frac{2 T 
            \operatorname{atanh}{\left(\frac{2 V_g + \delta}{\sqrt{\delta^{2}
            - 4 \epsilon}} \right)} \frac{d^{2}}{d T^{2}} \operatorname{
                a_{\alpha}}{\left(T \right)}}{\sqrt{\delta^{2} - 4 \epsilon}} 
                + V_g \frac{\partial}{\partial T} P{\left(T,V \right)}
        '''

        T = self.T
        delta, epsilon = self.delta, self.epsilon
        V = self.V_g
        dP_dT = self.dP_dT_g
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100
        return -R + 2.0*T*x0*catanh(x0*(V + V + delta)).real*self.d2a_alpha_dT2 + V*dP_dT
        
    @property
    def dH_dep_dP_l(self):
        r'''Derivative of departure enthalpy with respect to 
        pressure for the liquid phase, [(J/mol)/Pa]
        
        .. math::
            \frac{\partial H_{dep, l}}{\partial P} = P \frac{d}{d P} V{\left (P
            \right )} + V{\left (P \right )} + \frac{4 \left(T \frac{d}{d T} 
            \operatorname{a \alpha}{\left (T \right )} - \operatorname{a 
            \alpha}{\left (T \right )}\right) \frac{d}{d P} V{\left (P \right 
            )}}{\left(\delta^{2} - 4 \epsilon\right) \left(- \frac{\left(\delta
            + 2 V{\left (P \right )}\right)^{2}}{\delta^{2} - 4 \epsilon} 
            + 1\right)}
        '''
        delta = self.delta
        x0 = self.V_l
        x2 = delta*delta - 4.0*self.epsilon
        x4 = (delta + x0 + x0)
        return (x0 + self.dV_dP_l*(self.P - 4.0*(self.T*self.da_alpha_dT
                - self.a_alpha)/(x4*x4 - x2)))
        
    @property
    def dH_dep_dP_g(self):
        r'''Derivative of departure enthalpy with respect to 
        pressure for the gas phase, [(J/mol)/Pa]
        
        .. math::
            \frac{\partial H_{dep, g}}{\partial P} = P \frac{d}{d P} V{\left (P
            \right )} + V{\left (P \right )} + \frac{4 \left(T \frac{d}{d T} 
            \operatorname{a \alpha}{\left (T \right )} - \operatorname{a 
            \alpha}{\left (T \right )}\right) \frac{d}{d P} V{\left (P \right 
            )}}{\left(\delta^{2} - 4 \epsilon\right) \left(- \frac{\left(\delta
            + 2 V{\left (P \right )}\right)^{2}}{\delta^{2} - 4 \epsilon} 
            + 1\right)}
        '''
        delta = self.delta
        x0 = self.V_g
        x2 = delta*delta - 4.0*self.epsilon
        x4 = (delta + x0 + x0)
        return (x0 + self.dV_dP_g*(self.P - 4.0*(self.T*self.da_alpha_dT
                - self.a_alpha)/(x4*x4 - x2)))

    @property
    def dH_dep_dP_l_V(self):
        r'''Derivative of departure enthalpy with respect to 
        pressure at constant volume for the gas phase, [(J/mol)/Pa]
        
        .. math::
            \left(\frac{\partial H_{dep, g}}{\partial P}\right)_{V} = 
            - R \left(\frac{\partial T}{\partial P}\right)_V + V + \frac{2 \left(T 
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}
            + \left(\frac{\partial a \alpha}{\partial T}\right)_P
            \left(\frac{\partial T}{\partial P}\right)_V - \left(\frac{
            \partial a \alpha}{\partial P}\right)_{V} \right) 
            \operatorname{atanh}{\left(\frac{2 V + \delta}
            {\sqrt{\delta^{2} - 4 \epsilon}} \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}}
        '''

        T, V, delta, epsilon = self.T, self.V_l, self.delta, self.epsilon
        da_alpha_dT, d2a_alpha_dT2 = self.da_alpha_dT, self.d2a_alpha_dT2 
        dT_dP = self.dT_dP_l
        
        d2a_alpha_dTdP_V = d2a_alpha_dT2*dT_dP
        da_alpha_dP_V = da_alpha_dT*dT_dP
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100
            
        return (-R*dT_dP + V + 2.0*x0*(
                T*d2a_alpha_dTdP_V + dT_dP*da_alpha_dT - da_alpha_dP_V)
                *catanh(x0*(V + V + delta)).real)

    @property
    def dH_dep_dP_g_V(self):
        r'''Derivative of departure enthalpy with respect to 
        pressure at constant volume for the liquid phase, [(J/mol)/Pa]
        
        .. math::
            \left(\frac{\partial H_{dep, g}}{\partial P}\right)_{V} = 
            - R \left(\frac{\partial T}{\partial P}\right)_V + V + \frac{2 \left(T 
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}
            + \left(\frac{\partial a \alpha}{\partial T}\right)_P
            \left(\frac{\partial T}{\partial P}\right)_V - \left(\frac{
            \partial a \alpha}{\partial P}\right)_{V} \right) 
            \operatorname{atanh}{\left(\frac{2 V + \delta}
            {\sqrt{\delta^{2} - 4 \epsilon}} \right)}}{\sqrt{\delta^{2}
            - 4 \epsilon}}
        '''
        T, V, delta, epsilon = self.T, self.V_g, self.delta, self.epsilon
        da_alpha_dT, d2a_alpha_dT2 = self.da_alpha_dT, self.d2a_alpha_dT2 
        dT_dP = self.dT_dP_g
        
        d2a_alpha_dTdP_V = d2a_alpha_dT2*dT_dP
        da_alpha_dP_V = da_alpha_dT*dT_dP
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100
            
        return (-R*dT_dP + V + 2.0*x0*(
                T*d2a_alpha_dTdP_V + dT_dP*da_alpha_dT - da_alpha_dP_V)
                *catanh(x0*(V + V + delta)).real)

    @property
    def dH_dep_dV_g_T(self):
        r'''Derivative of departure enthalpy with respect to 
        volume at constant temperature for the gas phase, [J/m^3]
        
        .. math::
            \left(\frac{\partial H_{dep, g}}{\partial V}\right)_{T} = 
            \left(\frac{\partial H_{dep, g}}{\partial P}\right)_{T} \cdot
            \left(\frac{\partial P}{\partial V}\right)_{T} 
        '''
        return self.dH_dep_dP_g*self.dP_dV_g

    @property
    def dH_dep_dV_l_T(self):
        r'''Derivative of departure enthalpy with respect to 
        volume at constant temperature for the gas phase, [J/m^3]
        
        .. math::
            \left(\frac{\partial H_{dep, l}}{\partial V}\right)_{T} = 
            \left(\frac{\partial H_{dep, l}}{\partial P}\right)_{T} \cdot
            \left(\frac{\partial P}{\partial V}\right)_{T} 
        '''
        return self.dH_dep_dP_l*self.dP_dV_l

    @property
    def dH_dep_dV_g_P(self):
        r'''Derivative of departure enthalpy with respect to 
        volume at constant pressure for the gas phase, [J/m^3]
        
        .. math::
            \left(\frac{\partial H_{dep, g}}{\partial V}\right)_{P} = 
            \left(\frac{\partial H_{dep, g}}{\partial T}\right)_{P} \cdot
            \left(\frac{\partial T}{\partial V}\right)_{P} 
        '''
        return self.dH_dep_dT_g*self.dT_dV_g

    @property
    def dH_dep_dV_l_P(self):
        r'''Derivative of departure enthalpy with respect to 
        volume at constant pressure for the liquid phase, [J/m^3]
        
        .. math::
            \left(\frac{\partial H_{dep, l}}{\partial V}\right)_{P} = 
            \left(\frac{\partial H_{dep, l}}{\partial T}\right)_{P} \cdot
            \left(\frac{\partial T}{\partial V}\right)_{P} 
        '''
        return self.dH_dep_dT_l*self.dT_dV_l

    @property
    def dS_dep_dT_l(self):
        r'''Derivative of departure entropy with respect to 
        temperature for the liquid phase, [(J/mol)/K^2]
        
        .. math::
            \frac{\partial S_{dep, l}}{\partial T} = - \frac{R \frac{d}{d T}
            V{\left (T \right )}}{V{\left (T \right )}} + \frac{R \frac{d}{d T}
            V{\left (T \right )}}{- b + V{\left (T \right )}} + \frac{4
            \frac{d}{d T} V{\left (T \right )} \frac{d}{d T} \operatorname{a
            \alpha}{\left (T \right )}}{\left(\delta^{2} - 4 \epsilon\right) 
            \left(- \frac{\left(\delta + 2 V{\left (T \right )}\right)^{2}}
            {\delta^{2} - 4 \epsilon} + 1\right)} + \frac{2 \frac{d^{2}}{d 
            T^{2}}  \operatorname{a \alpha}{\left (T \right )}}
            {\sqrt{\delta^{2} - 4 \epsilon}} \operatorname{atanh}{\left (\frac{
            \delta + 2 V{\left (T \right )}}{\sqrt{\delta^{2} - 4 \epsilon}} 
            \right )} + \frac{R^{2} T}{P V{\left (T \right )}} \left(\frac{P}
            {R T} \frac{d}{d T} V{\left (T \right )} - \frac{P}{R T^{2}} 
            V{\left (T \right )}\right)
        '''
        x0 = self.V_l
        x1 = 1./x0
        x2 = self.dV_dT_l
        x3 = R*x2
        x4 = self.a_alpha
        x5 = self.delta*self.delta - 4.0*self.epsilon
        if x5 == 0.0:
            x5 = 1e-100
        x6 = x5**-0.5
        x7 = self.delta + 2.0*x0
        x8 = 1.0/x5
        return (R*x1*(x2 - x0/self.T) - x1*x3 - 4.0*x2*x8*self.da_alpha_dT
                /(x7*x7*x8 - 1.0) - x3/(self.b - x0) 
                + 2.0*x6*catanh(x6*x7).real*self.d2a_alpha_dT2)
    
    @property
    def dS_dep_dT_g(self):
        r'''Derivative of departure entropy with respect to 
        temperature for the gas phase, [(J/mol)/K^2]
        
        .. math::
            \frac{\partial S_{dep, g}}{\partial T} = - \frac{R \frac{d}{d T}
            V{\left (T \right )}}{V{\left (T \right )}} + \frac{R \frac{d}{d T}
            V{\left (T \right )}}{- b + V{\left (T \right )}} + \frac{4
            \frac{d}{d T} V{\left (T \right )} \frac{d}{d T} \operatorname{a
            \alpha}{\left (T \right )}}{\left(\delta^{2} - 4 \epsilon\right) 
            \left(- \frac{\left(\delta + 2 V{\left (T \right )}\right)^{2}}
            {\delta^{2} - 4 \epsilon} + 1\right)} + \frac{2 \frac{d^{2}}{d 
            T^{2}}  \operatorname{a \alpha}{\left (T \right )}}
            {\sqrt{\delta^{2} - 4 \epsilon}} \operatorname{atanh}{\left (\frac{
            \delta + 2 V{\left (T \right )}}{\sqrt{\delta^{2} - 4 \epsilon}} 
            \right )} + \frac{R^{2} T}{P V{\left (T \right )}} \left(\frac{P}
            {R T} \frac{d}{d T} V{\left (T \right )} - \frac{P}{R T^{2}} 
            V{\left (T \right )}\right)
        '''
        x0 = self.V_g
        x1 = 1./x0
        x2 = self.dV_dT_g
        x3 = R*x2
        x4 = self.a_alpha
        
        x5 = self.delta*self.delta - 4.0*self.epsilon
        if x5 == 0.0:
            x5 = 1e-100
        x6 = x5**-0.5
        x7 = self.delta + 2.0*x0
        x8 = 1.0/x5
        return (R*x1*(x2 - x0/self.T) - x1*x3 - 4.0*x2*x8*self.da_alpha_dT
                /(x7*x7*x8 - 1.0) - x3/(self.b - x0) 
                + 2.0*x6*catanh(x6*x7).real*self.d2a_alpha_dT2)

    @property
    def dS_dep_dT_l_V(self):
        r'''Derivative of departure entropy with respect to 
        temeprature at constant volume for the liquid phase, [(J/mol)/K^2]
        
        .. math::
            \left(\frac{\partial S_{dep, l}}{\partial T}\right)_{V} = 
            \frac{R^{2} T \left(\frac{V \frac{\partial}{\partial T} P{\left(T,V 
            \right)}}{R T} - \frac{V P{\left(T,V \right)}}{R T^{2}}\right)}{
            V P{\left(T,V \right)}} + \frac{2 \operatorname{atanh}{\left(
            \frac{2 V + \delta}{\sqrt{\delta^{2} - 4 \epsilon}} \right)}
            \frac{d^{2}}{d T^{2}} \operatorname{a \alpha}{\left(T \right)}}
            {\sqrt{\delta^{2} - 4 \epsilon}}
        '''
        T, P = self.T, self.P
        delta, epsilon = self.delta, self.epsilon
        V = self.V_l
        dP_dT = self.dP_dT_l
        try:
            x1 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x1 = 1e100
        return (R*(dP_dT/P - 1.0/T) + 2.0*x1*catanh(x1*(V + V + delta)).real*self.d2a_alpha_dT2)
            
    @property
    def dS_dep_dT_g_V(self):
        r'''Derivative of departure entropy with respect to 
        temeprature at constant volume for the gas phase, [(J/mol)/K^2]
        
        .. math::
            \left(\frac{\partial S_{dep, g}}{\partial T}\right)_{V} = 
            \frac{R^{2} T \left(\frac{V \frac{\partial}{\partial T} P{\left(T,V 
            \right)}}{R T} - \frac{V P{\left(T,V \right)}}{R T^{2}}\right)}{
            V P{\left(T,V \right)}} + \frac{2 \operatorname{atanh}{\left(
            \frac{2 V + \delta}{\sqrt{\delta^{2} - 4 \epsilon}} \right)}
            \frac{d^{2}}{d T^{2}} \operatorname{a \alpha}{\left(T \right)}}
            {\sqrt{\delta^{2} - 4 \epsilon}}
        '''
        T, P = self.T, self.P
        delta, epsilon = self.delta, self.epsilon
        V = self.V_g
        dP_dT = self.dP_dT_g
        try:
            x1 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x1 = 1e100
        return (R*(dP_dT/P - 1.0/T) + 2.0*x1*catanh(x1*(V + V + delta)).real*self.d2a_alpha_dT2)

    @property
    def dS_dep_dP_l(self):
        r'''Derivative of departure entropy with respect to 
        pressure for the liquid phase, [(J/mol)/K/Pa]
        
        .. math::
            \frac{\partial S_{dep, l}}{\partial P} = - \frac{R \frac{d}{d P}
            V{\left (P \right )}}{V{\left (P \right )}} + \frac{R \frac{d}{d P}
            V{\left (P \right )}}{- b + V{\left (P \right )}} + \frac{4 \frac{
            d}{d P} V{\left (P \right )} \frac{d}{d T} \operatorname{a \alpha}
            {\left (T \right )}}{\left(\delta^{2} - 4 \epsilon\right) \left(
            - \frac{\left(\delta + 2 V{\left (P \right )}\right)^{2}}{
            \delta^{2} - 4 \epsilon} + 1\right)} + \frac{R^{2} T}{P V{\left (P
            \right )}} \left(\frac{P}{R T} \frac{d}{d P} V{\left (P \right )} 
            + \frac{V{\left (P \right )}}{R T}\right)
        '''
        x0 = self.V_l
        x1 = 1.0/x0
        x2 = self.dV_dP_l
        x3 = R*x2
        try:
            x4 = 1.0/(self.delta*self.delta - 4.0*self.epsilon)
        except ZeroDivisionError:
            x4 = 1e50
        return (-x1*x3 - 4.0*x2*x4*self.da_alpha_dT/(x4*(self.delta + 2*x0)**2 
                - 1) - x3/(self.b - x0) + R*x1*(self.P*x2 + x0)/self.P)
        
    @property
    def dS_dep_dP_g(self):
        r'''Derivative of departure entropy with respect to 
        pressure for the gas phase, [(J/mol)/K/Pa]
        
        .. math::
            \frac{\partial S_{dep, g}}{\partial P} = - \frac{R \frac{d}{d P}
            V{\left (P \right )}}{V{\left (P \right )}} + \frac{R \frac{d}{d P}
            V{\left (P \right )}}{- b + V{\left (P \right )}} + \frac{4 \frac{
            d}{d P} V{\left (P \right )} \frac{d}{d T} \operatorname{a \alpha}
            {\left (T \right )}}{\left(\delta^{2} - 4 \epsilon\right) \left(
            - \frac{\left(\delta + 2 V{\left (P \right )}\right)^{2}}{
            \delta^{2} - 4 \epsilon} + 1\right)} + \frac{R^{2} T}{P V{\left (P
            \right )}} \left(\frac{P}{R T} \frac{d}{d P} V{\left (P \right )} 
            + \frac{V{\left (P \right )}}{R T}\right)
        '''
        x0 = self.V_g
        x1 = 1.0/x0
        x2 = self.dV_dP_g
        x3 = R*x2
        try:
            x4 = 1.0/(self.delta*self.delta - 4.0*self.epsilon)
        except ZeroDivisionError:
            x4 = 1e200
        return (-x1*x3 - 4.0*x2*x4*self.da_alpha_dT/(x4*(self.delta + 2*x0)**2 
                - 1) - x3/(self.b - x0) + R*x1*(self.P*x2 + x0)/self.P)

    @property
    def dS_dep_dP_g_V(self):
        r'''Derivative of departure entropy with respect to 
        pressure at constant volume for the gas phase, [(J/mol)/K/Pa]
        
        .. math::
            \left(\frac{\partial S_{dep, g}}{\partial P}\right)_{V} = 
            \frac{2 \operatorname{atanh}{\left(\frac{2 V + \delta}{
            \sqrt{\delta^{2} - 4 \epsilon}} \right)} 
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}}{\sqrt{\delta^{2} - 4 \epsilon}} 
            + \frac{R^{2} \left(- \frac{P V \frac{d}{d P} T{\left(P \right)}}
            {R T^{2}{\left(P \right)}}
             + \frac{V}{R T{\left(P \right)}}\right) T{\left(P \right)}}{P V}
        '''
        T, P, delta, epsilon = self.T, self.P, self.delta, self.epsilon
        d2a_alpha_dT2 = self.d2a_alpha_dT2 
        V, dT_dP = self.V_g, self.dT_dP_g
        d2a_alpha_dTdP_V = d2a_alpha_dT2*dT_dP
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100
        return (2.0*x0*catanh(x0*(V + V + delta)).real*d2a_alpha_dTdP_V
                - R*(P*dT_dP/T - 1.0)/P)

    @property
    def dS_dep_dP_l_V(self):
        r'''Derivative of departure entropy with respect to 
        pressure at constant volume for the liquid phase, [(J/mol)/K/Pa]
        
        .. math::
            \left(\frac{\partial S_{dep, l}}{\partial P}\right)_{V} = 
            \frac{2 \operatorname{atanh}{\left(\frac{2 V + \delta}{
            \sqrt{\delta^{2} - 4 \epsilon}} \right)} 
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}}{\sqrt{\delta^{2} - 4 \epsilon}} 
            + \frac{R^{2} \left(- \frac{P V \frac{d}{d P} T{\left(P \right)}}
            {R T^{2}{\left(P \right)}}
             + \frac{V}{R T{\left(P \right)}}\right) T{\left(P \right)}}{P V}
        '''
        T, P, delta, epsilon = self.T, self.P, self.delta, self.epsilon
        d2a_alpha_dT2 = self.d2a_alpha_dT2 
        V, dT_dP = self.V_l, self.dT_dP_l
        d2a_alpha_dTdP_V = d2a_alpha_dT2*dT_dP
        try:
            x0 = (delta*delta - 4.0*epsilon)**-0.5
        except ZeroDivisionError:
            x0 = 1e100
        return (2.0*x0*catanh(x0*(V + V + delta)).real*d2a_alpha_dTdP_V
                - R*(P*dT_dP/T - 1.0)/P)

    @property
    def dS_dep_dV_g_T(self):
        r'''Derivative of departure entropy with respect to 
        volume at constant temperature for the gas phase, [J/K/m^3]
        
        .. math::
            \left(\frac{\partial S_{dep, g}}{\partial V}\right)_{T} = 
            \left(\frac{\partial S_{dep, g}}{\partial P}\right)_{T} \cdot
            \left(\frac{\partial P}{\partial V}\right)_{T} 
        '''
        return self.dS_dep_dP_g*self.dP_dV_g

    @property
    def dS_dep_dV_l_T(self):
        r'''Derivative of departure entropy with respect to 
        volume at constant temperature for the gas phase, [J/K/m^3]
        
        .. math::
            \left(\frac{\partial S_{dep, l}}{\partial V}\right)_{T} = 
            \left(\frac{\partial S_{dep, l}}{\partial P}\right)_{T} \cdot
            \left(\frac{\partial P}{\partial V}\right)_{T} 
        '''
        return self.dS_dep_dP_l*self.dP_dV_l

    @property
    def dS_dep_dV_g_P(self):
        r'''Derivative of departure entropy with respect to 
        volume at constant pressure for the gas phase, [J/K/m^3]
        
        .. math::
            \left(\frac{\partial S_{dep, g}}{\partial V}\right)_{P} = 
            \left(\frac{\partial S_{dep, g}}{\partial T}\right)_{P} \cdot
            \left(\frac{\partial T}{\partial V}\right)_{P} 
        '''
        return self.dS_dep_dT_g*self.dT_dV_g

    @property
    def dS_dep_dV_l_P(self):
        r'''Derivative of departure entropy with respect to 
        volume at constant pressure for the liquid phase, [J/K/m^3]
        
        .. math::
            \left(\frac{\partial S_{dep, l}}{\partial V}\right)_{P} = 
            \left(\frac{\partial S_{dep, l}}{\partial T}\right)_{P} \cdot
            \left(\frac{\partial T}{\partial V}\right)_{P} 
        '''
        return self.dS_dep_dT_l*self.dT_dV_l
        
    @property
    def dfugacity_dT_l(self):
        r'''Derivative of fugacity with respect to temperature for the liquid 
        phase, [Pa/K]
        
        .. math::
            \frac{\partial (\text{fugacity})_{l}}{\partial T} = P \left(\frac{1}
            {R T} \left(- T \frac{\partial}{\partial T} \operatorname{S_{dep}}
            {\left (T,P \right )} - \operatorname{S_{dep}}{\left (T,P \right )}
            + \frac{\partial}{\partial T} \operatorname{H_{dep}}{\left (T,P
            \right )}\right) - \frac{1}{R T^{2}} \left(- T \operatorname{
                S_{dep}}{\left (T,P \right )} + \operatorname{H_{dep}}{\left
                (T,P \right )}\right)\right) e^{\frac{1}{R T} \left(- T 
                \operatorname{S_{dep}}{\left (T,P \right )} + \operatorname
                {H_{dep}}{\left (T,P \right )}\right)}
        '''
        T, P = self.T, self.P
        T_inv = 1.0/T
        S_dep_l = self.S_dep_l
        x4 = R_inv*(self.H_dep_l - T*S_dep_l)
        return P*(T_inv*R_inv*(self.dH_dep_dT_l - T*self.dS_dep_dT_l - S_dep_l) 
                  - x4*T_inv*T_inv)*exp(T_inv*x4)
 
    @property
    def dfugacity_dT_g(self):
        r'''Derivative of fugacity with respect to temperature for the gas 
        phase, [Pa/K]
        
        .. math::
            \frac{\partial (\text{fugacity})_{g}}{\partial T} = P \left(\frac{1}
            {R T} \left(- T \frac{\partial}{\partial T} \operatorname{S_{dep}}
            {\left (T,P \right )} - \operatorname{S_{dep}}{\left (T,P \right )}
            + \frac{\partial}{\partial T} \operatorname{H_{dep}}{\left (T,P
            \right )}\right) - \frac{1}{R T^{2}} \left(- T \operatorname{
                S_{dep}}{\left (T,P \right )} + \operatorname{H_{dep}}{\left
                (T,P \right )}\right)\right) e^{\frac{1}{R T} \left(- T 
                \operatorname{S_{dep}}{\left (T,P \right )} + \operatorname
                {H_{dep}}{\left (T,P \right )}\right)}
        '''
        T, P = self.T, self.P
        T_inv = 1.0/T
        S_dep_g = self.S_dep_g
        x4 = R_inv*(self.H_dep_g - T*S_dep_g)
        return P*(T_inv*R_inv*(self.dH_dep_dT_g - T*self.dS_dep_dT_g - S_dep_g) 
                  - x4*T_inv*T_inv)*exp(T_inv*x4)

    @property
    def dfugacity_dP_l(self):
        r'''Derivative of fugacity with respect to pressure for the liquid 
        phase, [-]
        
        .. math::
            \frac{\partial (\text{fugacity})_{l}}{\partial P} = \frac{P}{R T} 
            \left(- T \frac{\partial}{\partial P} \operatorname{S_{dep}}{\left
            (T,P \right )} + \frac{\partial}{\partial P} \operatorname{H_{dep}}
            {\left (T,P \right )}\right) e^{\frac{1}{R T} \left(- T
            \operatorname{S_{dep}}{\left (T,P \right )} + \operatorname{
            H_{dep}}{\left (T,P \right )}\right)} + e^{\frac{1}{R T}
            \left(- T \operatorname{S_{dep}}{\left (T,P \right )} 
            + \operatorname{H_{dep}}{\left (T,P \right )}\right)}
        '''
        T, P = self.T, self.P
        x0 = 1.0/(R*T)
        return (1.0 - P*x0*(T*self.dS_dep_dP_l - self.dH_dep_dP_l))*exp(
                -x0*(T*self.S_dep_l - self.H_dep_l))

    @property
    def dfugacity_dP_g(self):
        r'''Derivative of fugacity with respect to pressure for the gas 
        phase, [-]
        
        .. math::
            \frac{\partial (\text{fugacity})_{g}}{\partial P} = \frac{P}{R T} 
            \left(- T \frac{\partial}{\partial P} \operatorname{S_{dep}}{\left
            (T,P \right )} + \frac{\partial}{\partial P} \operatorname{H_{dep}}
            {\left (T,P \right )}\right) e^{\frac{1}{R T} \left(- T
            \operatorname{S_{dep}}{\left (T,P \right )} + \operatorname{
            H_{dep}}{\left (T,P \right )}\right)} + e^{\frac{1}{R T}
            \left(- T \operatorname{S_{dep}}{\left (T,P \right )} 
            + \operatorname{H_{dep}}{\left (T,P \right )}\right)}
        '''
        T, P = self.T, self.P
        x0 = 1.0/(R*T)
        return (1.0 - P*x0*(T*self.dS_dep_dP_g - self.dH_dep_dP_g))*exp(
                -x0*(T*self.S_dep_g - self.H_dep_g))

    @property
    def dphi_dT_l(self):
        r'''Derivative of fugacity coefficient with respect to temperature for 
        the liquid phase, [1/K]
        
        .. math::
            \frac{\partial \phi}{\partial T} = \left(\frac{- T \frac{\partial}
            {\partial T} \operatorname{S_{dep}}{\left(T,P \right)} 
            - \operatorname{S_{dep}}{\left(T,P \right)} + \frac{\partial}
            {\partial T} \operatorname{H_{dep}}{\left(T,P \right)}}{R T} 
            - \frac{- T \operatorname{S_{dep}}{\left(T,P \right)}
            + \operatorname{H_{dep}}{\left(T,P \right)}}{R T^{2}}\right) 
            e^{\frac{- T \operatorname{S_{dep}}{\left(T,P \right)} 
            + \operatorname{H_{dep}}{\left(T,P \right)}}{R T}}
        '''
        T, P = self.T, self.P
        T_inv = 1.0/T
        x4 = T_inv*(T*self.S_dep_l - self.H_dep_l)
        return (-R_inv*T_inv*(T*self.dS_dep_dT_l + self.S_dep_l - x4 
                             - self.dH_dep_dT_l)*exp(-R_inv*x4))
        
    @property
    def dphi_dT_g(self):
        r'''Derivative of fugacity coefficient with respect to temperature for 
        the gas phase, [1/K]
        
        .. math::
            \frac{\partial \phi}{\partial T} = \left(\frac{- T \frac{\partial}
            {\partial T} \operatorname{S_{dep}}{\left(T,P \right)} 
            - \operatorname{S_{dep}}{\left(T,P \right)} + \frac{\partial}
            {\partial T} \operatorname{H_{dep}}{\left(T,P \right)}}{R T} 
            - \frac{- T \operatorname{S_{dep}}{\left(T,P \right)}
            + \operatorname{H_{dep}}{\left(T,P \right)}}{R T^{2}}\right) 
            e^{\frac{- T \operatorname{S_{dep}}{\left(T,P \right)} 
            + \operatorname{H_{dep}}{\left(T,P \right)}}{R T}}
        '''
        T, P = self.T, self.P
        T_inv = 1.0/T
        x4 = T_inv*(T*self.S_dep_g - self.H_dep_g)
        return (-R_inv*T_inv*(T*self.dS_dep_dT_g + self.S_dep_g - x4 
                             - self.dH_dep_dT_g)*exp(-R_inv*x4))

    @property
    def dphi_dP_l(self):
        r'''Derivative of fugacity coefficient with respect to pressure for 
        the liquid phase, [1/Pa]
        
        .. math::
            \frac{\partial \phi}{\partial P} = \frac{\left(- T \frac{\partial}
            {\partial P} \operatorname{S_{dep}}{\left(T,P \right)}
            + \frac{\partial}{\partial P} \operatorname{H_{dep}}{\left(T,P 
            \right)}\right) e^{\frac{- T \operatorname{S_{dep}}{\left(T,P 
            \right)} + \operatorname{H_{dep}}{\left(T,P \right)}}{R T}}}{R T}
        '''
        T = self.T
        x0 = self.S_dep_l
        x1 = self.H_dep_l
        x2 = 1.0/(R*T)
        return -x2*(T*self.dS_dep_dP_l - self.dH_dep_dP_l)*exp(-x2*(T*x0 - x1))

    @property
    def dphi_dP_g(self):
        r'''Derivative of fugacity coefficient with respect to pressure for 
        the gas phase, [1/Pa]
        
        .. math::
            \frac{\partial \phi}{\partial P} = \frac{\left(- T \frac{\partial}
            {\partial P} \operatorname{S_{dep}}{\left(T,P \right)}
            + \frac{\partial}{\partial P} \operatorname{H_{dep}}{\left(T,P 
            \right)}\right) e^{\frac{- T \operatorname{S_{dep}}{\left(T,P 
            \right)} + \operatorname{H_{dep}}{\left(T,P \right)}}{R T}}}{R T}
        '''
        T = self.T
        x0 = self.S_dep_g
        x1 = self.H_dep_g
        x2 = 1.0/(R*T)
        return -x2*(T*self.dS_dep_dP_g - self.dH_dep_dP_g)*exp(-x2*(T*x0 - x1))
    
    @property
    def dbeta_dT_g(self):
        r'''Derivative of isobaric expansion coefficient with respect to 
        temeprature for the gas phase, [1/K^2]
        
        .. math::
            \frac{\partial \beta_g}{\partial T} = \frac{\frac{\partial^{2}}
            {\partial T^{2}} V{\left (T,P \right )_g}}{V{\left (T,P \right )_g}} -
            \frac{\left(\frac{\partial}{\partial T} V{\left (T,P \right )_g}
            \right)^{2}}{V^{2}{\left (T,P \right )_g}}
        '''
        V_inv = 1.0/self.V_g
        dV_dT = self.dV_dT_g
        return V_inv*(self.d2V_dT2_g - dV_dT*dV_dT*V_inv)

    @property
    def dbeta_dT_l(self):
        r'''Derivative of isobaric expansion coefficient with respect to 
        temeprature for the liquid phase, [1/K^2]
        
        .. math::
            \frac{\partial \beta_l}{\partial T} = \frac{\frac{\partial^{2}}
            {\partial T^{2}} V{\left (T,P \right )_l}}{V{\left (T,P \right )_l}} -
            \frac{\left(\frac{\partial}{\partial T} V{\left (T,P \right )_l}
            \right)^{2}}{V^{2}{\left (T,P \right )_l}}
        '''
        V_inv = 1.0/self.V_l
        dV_dT = self.dV_dT_l
        return V_inv*(self.d2V_dT2_l - dV_dT*dV_dT*V_inv)

    @property
    def dbeta_dP_g(self):
        r'''Derivative of isobaric expansion coefficient with respect to 
        pressure for the gas phase, [1/(Pa*K)]
        
        .. math::
            \frac{\partial \beta_g}{\partial P} = \frac{\frac{\partial^{2}}
            {\partial T\partial P} V{\left (T,P \right )_g}}{V{\left (T,
            P \right )_g}} - \frac{\frac{\partial}{\partial P} V{\left (T,P 
            \right )_g} \frac{\partial}{\partial T} V{\left (T,P \right )_g}}
            {V^{2}{\left (T,P \right )_g}}
        '''
        V_inv = 1.0/self.V_g
        dV_dT = self.dV_dT_g
        dV_dP = self.dV_dP_g
        return V_inv*(self.d2V_dTdP_g - dV_dT*dV_dP*V_inv)

    @property
    def dbeta_dP_l(self):
        r'''Derivative of isobaric expansion coefficient with respect to 
        pressure for the liquid phase, [1/(Pa*K)]
        
        .. math::
            \frac{\partial \beta_g}{\partial P} = \frac{\frac{\partial^{2}}
            {\partial T\partial P} V{\left (T,P \right )_l}}{V{\left (T,
            P \right )_l}} - \frac{\frac{\partial}{\partial P} V{\left (T,P 
            \right )_l} \frac{\partial}{\partial T} V{\left (T,P \right )_l}}
            {V^{2}{\left (T,P \right )_l}}
        '''
        V_inv = 1.0/self.V_l
        dV_dT = self.dV_dT_l
        dV_dP = self.dV_dP_l
        return V_inv*(self.d2V_dTdP_l - dV_dT*dV_dP*V_inv)

    @property
    def da_alpha_dP_g_V(self):
        r'''Derivative of the `a_alpha` with respect to 
        pressure at constant volume (varying T) for the gas phase, 
        [J^2/mol^2/Pa^2]
        
        .. math::
            \left(\frac{\partial a \alpha}{\partial P}\right)_{V}
            = \left(\frac{\partial a \alpha}{\partial T}\right)_{P}
            \cdot\left( \frac{\partial T}{\partial P}\right)_V
        '''
        return self.da_alpha_dT*self.dT_dP_g
        
    @property
    def da_alpha_dP_l_V(self):
        r'''Derivative of the `a_alpha` with respect to 
        pressure at constant volume (varying T) for the liquid phase, 
        [J^2/mol^2/Pa^2]
        
        .. math::
            \left(\frac{\partial a \alpha}{\partial P}\right)_{V}
            = \left(\frac{\partial a \alpha}{\partial T}\right)_{P}
            \cdot\left( \frac{\partial T}{\partial P}\right)_V
        '''
        return self.da_alpha_dT*self.dT_dP_l

    @property
    def d2a_alpha_dTdP_g_V(self):
        r'''Derivative of the temperature derivative of `a_alpha` with respect  
        to pressure at constant volume (varying T) for the gas phase, 
        [J^2/mol^2/Pa^2/K]
        
        .. math::
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}
            = \left(\frac{\partial^2 a \alpha}{\partial T^2}\right)_{P}
            \cdot\left( \frac{\partial T}{\partial P}\right)_V
            '''
        return self.d2a_alpha_dT2*self.dT_dP_g

    @property
    def d2a_alpha_dTdP_l_V(self):
        r'''Derivative of the temperature derivative of `a_alpha` with respect  
        to pressure at constant volume (varying T) for the liquid phase, 
        [J^2/mol^2/Pa^2/K]
        
        .. math::
            \left(\frac{\partial \left(\frac{\partial a \alpha}{\partial T}
            \right)_P}{\partial P}\right)_{V}
            = \left(\frac{\partial^2 a \alpha}{\partial T^2}\right)_{P}
            \cdot\left( \frac{\partial T}{\partial P}\right)_V
            '''
        return self.d2a_alpha_dT2*self.dT_dP_l

class GCEOS_DUMMY(GCEOS):
    Tc = None
    Pc = None
    omega = None
    def __init__(self, T=None, P=None, **kwargs):
        self.T = T
        self.P = P


# No named parameters
class ALPHA_FUNCTIONS(GCEOS):
    r'''Basic class with a number of attached alpha functions for different
    applications, all of which have no parameters attached. These alpha 
    functions should be used for fitting purposes; new EOSs should have their
    alpha functions added here. The first and second derivatives should also
    be implemented. Efficient implementations are discouraged but possible.
    
    All parameters should be in `self.alpha_function_coeffs`. This object is
    inspired by the work of [1]_, where most of the alpha functions have been
    found.

    Examples
    --------
    Swap out the default alpha function from the SRK EOS, replace it the same,
    a new method that takes a manually specified coefficient.
    
    >>> eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    >>> [eos.m, eos.a_alpha_and_derivatives(299)]
    [0.9326878999999999, (3.7271789178606376, -0.007332989159328508, 1.947612023379061e-05)]
    
    >>> class SRK_Soave_1972(SRK):
    ...     a_alpha_and_derivatives = ALPHA_FUNCTIONS.Soave_1972
    >>> SRK_Soave_1972.alpha_function_coeffs = [0.9326878999999999]
    >>> a = SRK_Soave_1972(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    >>> a.a_alpha_and_derivatives(299)
    (3.7271789178606376, -0.007332989159328508, 1.947612023379061e-05)
        


    References
    ----------
    .. [1] Young, André F., Fernando L. P. Pessoa, and Victor R. R. Ahón. 
       "Comparison of 20 Alpha Functions Applied in the Peng–Robinson Equation 
       of State for Vapor Pressure Estimation." Industrial & Engineering 
       Chemistry Research 55, no. 22 (June 8, 2016): 6506-16. 
       doi:10.1021/acs.iecr.6b00721.
    '''
    
    @staticmethod
    def Soave_1972(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Soave (1972) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Same as `SRK.a_alpha_and_derivatives` but slower and
        requiring `alpha_function_coeffs` to be set. One coefficient needed.
        
        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{Tc}} + 1\right)
            + 1\right)^{2}

        References
        ----------
        .. [1] Soave, Giorgio. "Equilibrium Constants from a Modified Redlich-
           Kwong Equation of State." Chemical Engineering Science 27, no. 6 
           (June 1972): 1197-1203. doi:10.1016/0009-2509(72)80096-4.
        '''
        c1 = self.alpha_function_coeffs[0]
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*(c1*(-sqrt(T/Tc) + 1) + 1)**2
        if not full:
            return a_alpha
        else:
            da_alpha_dT = -a*c1*sqrt(T/Tc)*(c1*(-sqrt(T/Tc) + 1) + 1)/T
            d2a_alpha_dT2 = a*c1*(c1/Tc - sqrt(T/Tc)*(c1*(sqrt(T/Tc) - 1) - 1)/T)/(2*T)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Heyen(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Heyen (1980) [1]_. Returns `a_alpha`, `da_alpha_dT`,  
        and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Two coefficients needed.
        
        .. math::
            \alpha = e^{c_{1} \left(- \left(\frac{T}{Tc}\right)^{c_{2}}
            + 1\right)}

        References
        ----------
        .. [1] Heyen, G. Liquid and Vapor Properties from a Cubic Equation of 
           State. In "Proceedings of the 2nd International Conference on Phase 
           Equilibria and Fluid Properties in the Chemical Industry". DECHEMA: 
           Frankfurt, 1980; p 9-13.
        '''
        c1, c2 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*exp(c1*(1 -(T/Tc)**c2))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = -a*c1*c2*(T/Tc)**c2*exp(c1*(-(T/Tc)**c2 + 1))/T
            d2a_alpha_dT2 = a*c1*c2*(T/Tc)**c2*(c1*c2*(T/Tc)**c2 - c2 + 1)*exp(-c1*((T/Tc)**c2 - 1))/T**2
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
        
    @staticmethod
    def Harmens_Knapp(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Harmens and Knapp (1980) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Two coefficients needed.
        
        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{Tc}} + 1\right) 
            - c_{2} \left(1 - \frac{Tc}{T}\right) + 1\right)^{2}

        References
        ----------
        .. [1] Harmens, A., and H. Knapp. "Three-Parameter Cubic Equation of 
           State for Normal Substances." Industrial & Engineering Chemistry 
           Fundamentals 19, no. 3 (August 1, 1980): 291-94. 
           doi:10.1021/i160075a010. 
        '''
        c1, c2 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*(c1*(-sqrt(T/Tc) + 1) - c2*(1 - Tc/T) + 1)**2
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-c1*sqrt(T/Tc)/T - 2*Tc*c2/T**2)*(c1*(-sqrt(T/Tc) + 1) - c2*(1 - Tc/T) + 1)
            d2a_alpha_dT2 = a*((c1*sqrt(T/Tc) + 2*Tc*c2/T)**2 - (c1*sqrt(T/Tc) + 8*Tc*c2/T)*(c1*(sqrt(T/Tc) - 1) + c2*(1 - Tc/T) - 1))/(2*T**2)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Mathias(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Mathias (1983) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Two coefficients needed.
        
        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{Tc}} + 1\right)
            - c_{2} \left(- \frac{T}{Tc} + 0.7\right) \left(- \frac{T}{Tc} 
            + 1\right) + 1\right)^{2}

        References
        ----------
        .. [1] Mathias, Paul M. "A Versatile Phase Equilibrium Equation of 
           State." Industrial & Engineering Chemistry Process Design and 
           Development 22, no. 3 (July 1, 1983): 385-91. 
           doi:10.1021/i200022a008. 
        '''
        c1, c2 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*(1 + c1*(1-sqrt(Tr)) -c2*(1-Tr)*(0.7-Tr))**2
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(c1*(-sqrt(T/Tc) + 1) - c2*(-T/Tc + 0.7)*(-T/Tc + 1) + 1)*(2*c2*(-T/Tc + 0.7)/Tc + 2*c2*(-T/Tc + 1)/Tc - c1*sqrt(T/Tc)/T)
            d2a_alpha_dT2 = a*((8*c2/Tc**2 - c1*sqrt(T/Tc)/T**2)*(c1*(sqrt(T/Tc) - 1) + c2*(T/Tc - 1)*(T/Tc - 0.7) - 1) + (2*c2*(T/Tc - 1)/Tc + 2*c2*(T/Tc - 0.7)/Tc + c1*sqrt(T/Tc)/T)**2)/2
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Mathias_Copeman(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Mathias and Copeman (1983) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Three coefficients needed.
        
        .. math::
            \alpha = \left(c_{1} \left(- \sqrt{\frac{T}{Tc}} + 1\right)
            + c_{2} \left(- \sqrt{\frac{T}{Tc}} + 1\right)^{2} + c_{3} \left(
            - \sqrt{\frac{T}{Tc}} + 1\right)^{3} + 1\right)^{2}

        References
        ----------
        .. [1] Mathias, Paul M., and Thomas W. Copeman. "Extension of the 
           Peng-Robinson Equation of State to Complex Mixtures: Evaluation of 
           the Various Forms of the Local Composition Concept." Fluid Phase 
           Equilibria 13 (January 1, 1983): 91-108. 
           doi:10.1016/0378-3812(83)80084-3. 
        '''
        c1, c2, c3 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*(c1*(-sqrt(T/Tc) + 1) + c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-c1*sqrt(T/Tc)/T - 2*c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T - 3*c3*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)**2/T)*(c1*(-sqrt(T/Tc) + 1) + c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)
            d2a_alpha_dT2 = a*(T*(c1 - 2*c2*(sqrt(T/Tc) - 1) + 3*c3*(sqrt(T/Tc) - 1)**2)**2 - (2*T*(c2 - 3*c3*(sqrt(T/Tc) - 1)) + Tc*sqrt(T/Tc)*(c1 - 2*c2*(sqrt(T/Tc) - 1) + 3*c3*(sqrt(T/Tc) - 1)**2))*(c1*(sqrt(T/Tc) - 1) - c2*(sqrt(T/Tc) - 1)**2 + c3*(sqrt(T/Tc) - 1)**3 - 1))/(2*T**2*Tc)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Gibbons_Laughton(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Gibbons and Laughton (1984) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Two coefficients needed.
        
        .. math::
            \alpha = c_{1} \left(\frac{T}{Tc} - 1\right) + c_{2} 
            \left(\sqrt{\frac{T}{Tc}} - 1\right) + 1

        References
        ----------
        .. [1] Gibbons, Richard M., and Andrew P. Laughton. "An Equation of 
           State for Polar and Non-Polar Substances and Mixtures" 80, no. 9 
           (January 1, 1984): 1019-38. doi:10.1039/F29848001019.
        '''
        c1, c2 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*(c1*(T/Tc - 1) + c2*(sqrt(T/Tc) - 1) + 1)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(c1/Tc + c2*sqrt(T/Tc)/(2*T))
            d2a_alpha_dT2 = a*(-c2*sqrt(T/Tc)/(4*T**2))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Soave_1984(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Soave (1984) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Two coefficients needed.
        
        .. math::
            \alpha = c_{1} \left(- \frac{T}{Tc} + 1\right) + c_{2} \left(-1
            + \frac{Tc}{T}\right) + 1

        References
        ----------
        .. [1] Soave, G. "Improvement of the Van Der Waals Equation of State." 
           Chemical Engineering Science 39, no. 2 (January 1, 1984): 357-69. 
           doi:10.1016/0009-2509(84)80034-2.
        '''
        c1, c2 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*(c1*(-T/Tc + 1) + c2*(-1 + Tc/T) + 1)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-c1/Tc - Tc*c2/T**2)
            d2a_alpha_dT2 = a*(2*Tc*c2/T**3)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
    # "Stryjek-Vera" skipped, doesn't match PRSV or PRSV2

    @staticmethod
    def Yu_Lu(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Yu and Lu (1987) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Four coefficients needed.
        
        .. math::
            \alpha = 10^{c_{4} \left(- \frac{T}{Tc} + 1\right) \left(
            \frac{T^{2} c_{3}}{Tc^{2}} + \frac{T c_{2}}{Tc} + c_{1}\right)}

        References
        ----------
        .. [1] Yu, Jin-Min, and Benjamin C. -Y. Lu. "A Three-Parameter Cubic 
           Equation of State for Asymmetric Mixture Density Calculations." 
           Fluid Phase Equilibria 34, no. 1 (January 1, 1987): 1-19. 
           doi:10.1016/0378-3812(87)85047-1. 
        '''
        c1, c2, c3, c4 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*10**(c4*(-T/Tc + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(10**(c4*(-T/Tc + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1))*(c4*(-T/Tc + 1)*(2*T*c3/Tc**2 + c2/Tc) - c4*(T**2*c3/Tc**2 + T*c2/Tc + c1)/Tc)*log(10))
            d2a_alpha_dT2 = a*(10**(-c4*(T/Tc - 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1))*c4*(-4*T*c3/Tc - 2*c2 - 2*c3*(T/Tc - 1) + c4*(T**2*c3/Tc**2 + T*c2/Tc + c1 + (T/Tc - 1)*(2*T*c3/Tc + c2))**2*log(10))*log(10)/Tc**2)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Trebble_Bishnoi(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Trebble and Bishnoi (1987) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. One coefficient needed.
        
        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{Tc} + 1\right)}

        References
        ----------
        .. [1] Trebble, M. A., and P. R. Bishnoi. "Development of a New Four-
           Parameter Cubic Equation of State." Fluid Phase Equilibria 35, no. 1
           (September 1, 1987): 1-18. doi:10.1016/0378-3812(87)80001-8.
        '''
        c1 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*exp(c1*(-T/Tc + 1))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*-c1*exp(c1*(-T/Tc + 1))/Tc
            d2a_alpha_dT2 = a*c1**2*exp(-c1*(T/Tc - 1))/Tc**2
            return a_alpha, da_alpha_dT, d2a_alpha_dT2
            
            
    @staticmethod
    def Melhem(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Melhem et al. (1989) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Two coefficients needed.
        
        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{Tc} + 1\right) + c_{2} 
            \left(- \sqrt{\frac{T}{Tc}} + 1\right)^{2}}

        References
        ----------
        .. [1] Melhem, Georges A., Riju Saini, and Bernard M. Goodwin. "A 
           Modified Peng-Robinson Equation of State." Fluid Phase Equilibria 
           47, no. 2 (August 1, 1989): 189-237. 
           doi:10.1016/0378-3812(89)80176-1. 
        '''
        c1, c2 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*exp(c1*(-T/Tc + 1) + c2*(-sqrt(T/Tc) + 1)**2)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((-c1/Tc - c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T)*exp(c1*(-T/Tc + 1) + c2*(-sqrt(T/Tc) + 1)**2))
            d2a_alpha_dT2 = a*(((c1/Tc - c2*sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T)**2 + c2*(1/Tc - sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T)/(2*T))*exp(-c1*(T/Tc - 1) + c2*(sqrt(T/Tc) - 1)**2))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Androulakis(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Androulakis et al. (1989) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Three coefficients needed.
        
        .. math::
            \alpha = c_{1} \left(- \left(\frac{T}{Tc}\right)^{\frac{2}{3}}
            + 1\right) + c_{2} \left(- \left(\frac{T}{Tc}\right)^{\frac{2}{3}} 
            + 1\right)^{2} + c_{3} \left(- \left(\frac{T}{Tc}\right)^{
            \frac{2}{3}} + 1\right)^{3} + 1

        References
        ----------
        .. [1] Androulakis, I. P., N. S. Kalospiros, and D. P. Tassios. 
           "Thermophysical Properties of Pure Polar and Nonpolar Compounds with
           a Modified VdW-711 Equation of State." Fluid Phase Equilibria 45, 
           no. 2 (April 1, 1989): 135-63. doi:10.1016/0378-3812(89)80254-7. 
        '''
        c1, c2, c3 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*(c1*(-(T/Tc)**(2/3) + 1) + c2*(-(T/Tc)**(2/3) + 1)**2 + c3*(-(T/Tc)**(2/3) + 1)**3 + 1)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-2*c1*(T/Tc)**(2/3)/(3*T) - 4*c2*(T/Tc)**(2/3)*(-(T/Tc)**(2/3) + 1)/(3*T) - 2*c3*(T/Tc)**(2/3)*(-(T/Tc)**(2/3) + 1)**2/T)
            d2a_alpha_dT2 = a*(2*(T/Tc)**(2/3)*(c1 + 4*c2*(T/Tc)**(2/3) - 2*c2*((T/Tc)**(2/3) - 1) - 12*c3*(T/Tc)**(2/3)*((T/Tc)**(2/3) - 1) + 3*c3*((T/Tc)**(2/3) - 1)**2)/(9*T**2))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Schwartzentruber(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Schwartzentruber et al. (1990) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Three coefficients needed.
        
        .. math::
            \alpha = \left(c_{4} \left(- \sqrt{\frac{T}{Tc}} + 1\right) 
            - \left(- \sqrt{\frac{T}{Tc}} + 1\right) \left(\frac{T^{2} c_{3}}
            {Tc^{2}} + \frac{T c_{2}}{Tc} + c_{1}\right) + 1\right)^{2}

        References
        ----------
        .. [1] J. Schwartzentruber, H. Renon, and S. Watanasiri, "K-values for 
           Non-Ideal Systems:An Easier Way," Chem. Eng., March 1990, 118-124.
        '''
        c1, c2, c3 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*((c4*(-sqrt(T/Tc) + 1) - (-sqrt(T/Tc) + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1) + 1)**2)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((c4*(-sqrt(T/Tc) + 1) - (-sqrt(T/Tc) + 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1) + 1)*(-2*(-sqrt(T/Tc) + 1)*(2*T*c3/Tc**2 + c2/Tc) - c4*sqrt(T/Tc)/T + sqrt(T/Tc)*(T**2*c3/Tc**2 + T*c2/Tc + c1)/T))
            d2a_alpha_dT2 = a*(((-c4*(sqrt(T/Tc) - 1) + (sqrt(T/Tc) - 1)*(T**2*c3/Tc**2 + T*c2/Tc + c1) + 1)*(8*c3*(sqrt(T/Tc) - 1)/Tc**2 + 4*sqrt(T/Tc)*(2*T*c3/Tc + c2)/(T*Tc) + c4*sqrt(T/Tc)/T**2 - sqrt(T/Tc)*(T**2*c3/Tc**2 + T*c2/Tc + c1)/T**2) + (2*(sqrt(T/Tc) - 1)*(2*T*c3/Tc + c2)/Tc - c4*sqrt(T/Tc)/T + sqrt(T/Tc)*(T**2*c3/Tc**2 + T*c2/Tc + c1)/T)**2)/2)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Almeida(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Almeida et al. (1991) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Three coefficients needed.
        
        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{Tc} + 1\right) \left|{
            \frac{T}{Tc} - 1}\right|^{c_{2} - 1} + c_{3} \left(-1 
            + \frac{Tc}{T}\right)}

        References
        ----------
        .. [1] Almeida, G. S., M. Aznar, and A. S. Telles. "Uma Nova Forma de 
           Dependência Com a Temperatura Do Termo Atrativo de Equações de 
           Estado Cúbicas." RBE, Rev. Bras. Eng., Cad. Eng. Quim 8 (1991): 95.
        '''
        # Note: For the second derivative, requires the use a CAS which can 
        # handle the assumption that Tr-1 != 0.
        c1, c2, c3 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*exp(c1*(-T/Tc + 1)*abs(T/Tc - 1)**(c2 - 1) + c3*(-1 + Tc/T))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((c1*(c2 - 1)*(-T/Tc + 1)*abs(T/Tc - 1)**(c2 - 1)*copysign(1, T/Tc - 1)/(Tc*Abs(T/Tc - 1)) - c1*abs(T/Tc - 1)**(c2 - 1)/Tc - Tc*c3/T**2)*exp(c1*(-T/Tc + 1)*abs(T/Tc - 1)**(c2 - 1) + c3*(-1 + Tc/T)))
            d2a_alpha_dT2 = a*exp(c3*(Tc/T - 1) - c1*abs(T/Tc - 1)**(c2 - 1)*(T/Tc - 1))*((c1*abs(T/Tc - 1)**(c2 - 1))/Tc + (Tc*c3)/T**2 + (c1*abs(T/Tc - 1)**(c2 - 2)*copysign(1, T/Tc - 1)*(c2 - 1)*(T/Tc - 1))/Tc)**2 - exp(c3*(Tc/T - 1) - c1*abs(T/Tc - 1)**(c2 - 1)*(T/Tc - 1))*((2*c1*abs(T/Tc - 1)**(c2 - 2)*copysign(1, T/Tc - 1)*(c2 - 1))/Tc**2 - (2*Tc*c3)/T**3 + (c1*abs(T/Tc - 1)**(c2 - 3)*copysign(1, T/Tc - 1)**2*(c2 - 1)*(c2 - 2)*(T/Tc - 1))/Tc**2)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Twu(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Twu et al. (1991) [1]_. Returns `a_alpha`, 
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`  
        for more documentation. Three coefficients needed.
        
        .. math::
            \alpha = \left(\frac{T}{Tc}\right)^{c_{3} \left(c_{2} 
            - 1\right)} e^{c_{1} \left(- \left(\frac{T}{Tc}
            \right)^{c_{2} c_{3}} + 1\right)}

        References
        ----------
        .. [1] Twu, Chorng H., David Bluck, John R. Cunningham, and John E. 
           Coon. "A Cubic Equation of State with a New Alpha Function and a 
           New Mixing Rule." Fluid Phase Equilibria 69 (December 10, 1991): 
           33-50. doi:10.1016/0378-3812(91)90024-2.
        '''
        c1, c2, c3 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*((T/Tc)**(c3*(c2 - 1))*exp(c1*(-(T/Tc)**(c2*c3) + 1)))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-c1*c2*c3*(T/Tc)**(c2*c3)*(T/Tc)**(c3*(c2 - 1))*exp(c1*(-(T/Tc)**(c2*c3) + 1))/T + c3*(T/Tc)**(c3*(c2 - 1))*(c2 - 1)*exp(c1*(-(T/Tc)**(c2*c3) + 1))/T)
            d2a_alpha_dT2 = a*(c3*(T/Tc)**(c3*(c2 - 1))*(c1**2*c2**2*c3*(T/Tc)**(2*c2*c3) - c1*c2**2*c3*(T/Tc)**(c2*c3) - 2*c1*c2*c3*(T/Tc)**(c2*c3)*(c2 - 1) + c1*c2*(T/Tc)**(c2*c3) - c2 + c3*(c2 - 1)**2 + 1)*exp(-c1*((T/Tc)**(c2*c3) - 1))/T**2)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Soave_1993(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Soave (1983) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Two coefficient needed.
        
        .. math::
            \alpha = c_{1} \left(- \frac{T}{Tc} + 1\right) + c_{2} 
            \left(- \sqrt{\frac{T}{Tc}} + 1\right)^{2} + 1

        References
        ----------
        .. [1] Soave, G. "Improving the Treatment of Heavy Hydrocarbons by the 
           SRK EOS." Fluid Phase Equilibria 84 (April 1, 1993): 339-42. 
           doi:10.1016/0378-3812(93)85131-5.
        '''
        c1, c2 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*(c1*(-T/Tc + 1) + c2*(-sqrt(T/Tc) + 1)**2 + 1)
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-c1/Tc - c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T)
            d2a_alpha_dT2 = a*(c2*(1/Tc - sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T)/(2*T))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Gasem(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Gasem (2001) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Three coefficients needed.
        
        .. math::
            \alpha = e^{\left(- \left(\frac{T}{Tc}\right)^{c_{3}} + 1\right) 
            \left(\frac{T c_{2}}{Tc} + c_{1}\right)}

        References
        ----------
        .. [1] Gasem, K. A. M, W Gao, Z Pan, and R. L Robinson Jr. "A Modified 
           Temperature Dependence for the Peng-Robinson Equation of State." 
           Fluid Phase Equilibria 181, no. 1–2 (May 25, 2001): 113-25. 
           doi:10.1016/S0378-3812(01)00488-5.
        '''
        c1, c2, c3 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*(exp((-(T/Tc)**c3 + 1)*(T*c2/Tc + c1)))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((c2*(-(T/Tc)**c3 + 1)/Tc - c3*(T/Tc)**c3*(T*c2/Tc + c1)/T)*exp((-(T/Tc)**c3 + 1)*(T*c2/Tc + c1)))
            d2a_alpha_dT2 = a*(((c2*((T/Tc)**c3 - 1)/Tc + c3*(T/Tc)**c3*(T*c2/Tc + c1)/T)**2 - c3*(T/Tc)**c3*(2*c2/Tc + c3*(T*c2/Tc + c1)/T - (T*c2/Tc + c1)/T)/T)*exp(-((T/Tc)**c3 - 1)*(T*c2/Tc + c1)))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Coquelet(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Coquelet et al. (2004) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Three coefficients needed.
        
        .. math::
            \alpha = e^{c_{1} \left(- \frac{T}{Tc} + 1\right) \left(c_{2} 
            \left(- \sqrt{\frac{T}{Tc}} + 1\right)^{2} + c_{3} 
            \left(- \sqrt{\frac{T}{Tc}} + 1\right)^{3} + 1\right)^{2}}

        References
        ----------
        .. [1] Coquelet, C., A. Chapoy, and D. Richon. "Development of a New 
           Alpha Function for the Peng–Robinson Equation of State: Comparative 
           Study of Alpha Function Models for Pure Gases (Natural Gas 
           Components) and Water-Gas Systems." International Journal of 
           Thermophysics 25, no. 1 (January 1, 2004): 133-58. 
           doi:10.1023/B:IJOT.0000022331.46865.2f.
        '''
        c1, c2, c3 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*(exp(c1*(-T/Tc + 1)*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((c1*(-T/Tc + 1)*(-2*c2*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)/T - 3*c3*sqrt(T/Tc)*(-sqrt(T/Tc) + 1)**2/T)*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1) - c1*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2/Tc)*exp(c1*(-T/Tc + 1)*(c2*(-sqrt(T/Tc) + 1)**2 + c3*(-sqrt(T/Tc) + 1)**3 + 1)**2))
            d2a_alpha_dT2 = a*(c1*(c1*(-(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)/Tc + sqrt(T/Tc)*(-2*c2 + 3*c3*(sqrt(T/Tc) - 1))*(sqrt(T/Tc) - 1)*(T/Tc - 1)/T)**2*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)**2 - ((T/Tc - 1)*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)*(2*c2/Tc - 6*c3*(sqrt(T/Tc) - 1)/Tc - 2*c2*sqrt(T/Tc)*(sqrt(T/Tc) - 1)/T + 3*c3*sqrt(T/Tc)*(sqrt(T/Tc) - 1)**2/T) + 4*sqrt(T/Tc)*(2*c2 - 3*c3*(sqrt(T/Tc) - 1))*(sqrt(T/Tc) - 1)*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)/Tc + (2*c2 - 3*c3*(sqrt(T/Tc) - 1))**2*(sqrt(T/Tc) - 1)**2*(T/Tc - 1)/Tc)/(2*T))*exp(-c1*(T/Tc - 1)*(c2*(sqrt(T/Tc) - 1)**2 - c3*(sqrt(T/Tc) - 1)**3 + 1)**2))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Haghtalab(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Haghtalab et al. (2010) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Three coefficients needed.
        
        .. math::
            \alpha = e^{\left(- c_{3}^{\log{\left (\frac{T}{Tc} \right )}} 
            + 1\right) \left(- \frac{T c_{2}}{Tc} + c_{1}\right)}

        References
        ----------
        .. [1] Haghtalab, A., M. J. Kamali, S. H. Mazloumi, and P. Mahmoodi. 
           "A New Three-Parameter Cubic Equation of State for Calculation 
           Physical Properties and Vapor-liquid Equilibria." Fluid Phase 
           Equilibria 293, no. 2 (June 25, 2010): 209-18. 
           doi:10.1016/j.fluid.2010.03.029.
        '''
        c1, c2, c3 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*exp((-c3**log(T/Tc) + 1)*(-T*c2/Tc + c1))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((-c2*(-c3**log(T/Tc) + 1)/Tc - c3**log(T/Tc)*(-T*c2/Tc + c1)*log(c3)/T)*exp((-c3**log(T/Tc) + 1)*(-T*c2/Tc + c1)))
            d2a_alpha_dT2 = a*(((c2*(c3**log(T/Tc) - 1)/Tc + c3**log(T/Tc)*(T*c2/Tc - c1)*log(c3)/T)**2 + c3**log(T/Tc)*(2*c2/Tc + (T*c2/Tc - c1)*log(c3)/T - (T*c2/Tc - c1)/T)*log(c3)/T)*exp((c3**log(T/Tc) - 1)*(T*c2/Tc - c1)))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Saffari(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Saffari and Zahedi (2013) [1]_. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Three coefficients needed.
        
        .. math::
            \alpha = e^{\frac{T c_{1}}{Tc} + c_{2} \log{\left (\frac{T}{Tc} 
            \right )} + c_{3} \left(- \sqrt{\frac{T}{Tc}} + 1\right)}

        References
        ----------
        .. [1] Saffari, Hamid, and Alireza Zahedi. "A New Alpha-Function for 
           the Peng-Robinson Equation of State: Application to Natural Gas." 
           Chinese Journal of Chemical Engineering 21, no. 10 (October 1, 
           2013): 1155-61. doi:10.1016/S1004-9541(13)60581-9.
        '''
        c1, c2, c3 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*(exp(T*c1/Tc + c2*log(T/Tc) + c3*(-sqrt(T/Tc) + 1)))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*((c1/Tc + c2/T - c3*sqrt(T/Tc)/(2*T))*exp(T*c1/Tc + c2*log(T/Tc) + c3*(-sqrt(T/Tc) + 1)))
            d2a_alpha_dT2 = a*(((2*c1/Tc + 2*c2/T - c3*sqrt(T/Tc)/T)**2 - (4*c2 - c3*sqrt(T/Tc))/T**2)*exp(T*c1/Tc + c2*log(T/Tc) - c3*(sqrt(T/Tc) - 1))/4)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    @staticmethod
    def Chen_Yang(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives according to Hamid and Yang (2017) [1]_. Returns `a_alpha`,
        `da_alpha_dT`, and `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives`
        for more documentation. Seven coefficients needed.
        
        .. math::
            \alpha = e^{\left(- c_{3}^{\log{\left (\frac{T}{Tc} \right )}} 
            + 1\right) \left(- \frac{T c_{2}}{Tc} + c_{1}\right)}

        References
        ----------
        .. [1] Chen, Zehua, and Daoyong Yang. "Optimization of the Reduced 
           Temperature Associated with Peng–Robinson Equation of State and 
           Soave-Redlich-Kwong Equation of State To Improve Vapor Pressure 
           Prediction for Heavy Hydrocarbon Compounds." Journal of Chemical & 
           Engineering Data, August 31, 2017. doi:10.1021/acs.jced.7b00496.
        '''
        c1, c2, c3, c4, c5, c6, c7 = self.alpha_function_coeffs
        T, Tc, a = self.T, self.Tc, self.a
        a_alpha = a*exp(c4*log((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)**2 + (-T/Tc + 1)*(c1 + c2*omega + c3*omega**2))
        if not full:
            return a_alpha
        else:
            da_alpha_dT = a*(-(c1 + c2*omega + c3*omega**2)/Tc - c4*sqrt(T/Tc)*(c5 + c6*omega + c7*omega**2)*log((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)/(T*((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)))*exp(c4*log((-sqrt(T/Tc) + 1)*(c5 + c6*omega + c7*omega**2) + 1)**2 + (-T/Tc + 1)*(c1 + c2*omega + c3*omega**2))
            d2a_alpha_dT2 = a*(((c1 + c2*omega + c3*omega**2)/Tc - c4*sqrt(T/Tc)*(c5 + c6*omega + c7*omega**2)*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)/(T*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)))**2 - c4*(c5 + c6*omega + c7*omega**2)*((c5 + c6*omega + c7*omega**2)*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)/(Tc*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)) - (c5 + c6*omega + c7*omega**2)/(Tc*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)) + sqrt(T/Tc)*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)/T)/(2*T*((sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) - 1)))*exp(c4*log(-(sqrt(T/Tc) - 1)*(c5 + c6*omega + c7*omega**2) + 1)**2 - (T/Tc - 1)*(c1 + c2*omega + c3*omega**2))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class IG(GCEOS):
    r'''Class for solving the ideal gas equation in the `GCEOS` framework.
    This provides access to a number of derivatives and properties easily.
    It also keeps a common interface for all gas models. However, it is 
    somewhat slow.
    
    Subclasses `GCEOS`, which 
    provides the methods for solving the EOS and calculating its assorted 
    relevant thermodynamic properties. Solves the EOS on initialization. 

    Implemented methods here are `a_alpha_and_derivatives`, which calculates 
    a_alpha and its first and second derivatives (all zero), and `solve_T`, 
    which from a specified `P` and `V` obtains `T`.
    
    Two of `T`, `P`, and `V` are needed to solve the EOS; values for `Tc` and
    `Pc` and `omega`, which are not used in the calculates, are set to those of
    methane by default to allow use without specifying them.

    .. math::
        P = \frac{RT}{V}
        
    Parameters
    ----------
    Tc : float, optional
        Critical temperature, [K]
    Pc : float, optional
        Critical pressure, [Pa]
    omega : float, optional
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    T-P initialization, and exploring each phase's properties:
    
    >>> eos = IG(T=400., P=1E6)
    >>> eos.V_g, eos.phase
    (0.003325785047261296, 'g')
    >>> eos.H_dep_g, eos.S_dep_g, eos.U_dep_g, eos.G_dep_g, eos.A_dep_g
    (0.0, 0.0, 0.0, 0.0, 0.0)
    >>> eos.beta_g, eos.kappa_g, eos.Cp_dep_g, eos.Cv_dep_g
    (0.0024999999999999996, 1e-06, -1.7763568394002505e-15, 0.0)
    >>> eos.fugacity_g, eos.PIP_g, eos.Z_g, eos.dP_dT_g
    (1000000.0, 0.9999999999999999, 1.0, 2500.0)
    
    Notes
    -----

    References
    ----------
    .. [1] Smith, J. M, H. C Van Ness, and Michael M Abbott. Introduction to 
       Chemical Engineering Thermodynamics. Boston: McGraw-Hill, 2005.
    '''
    Zc = 1.0
    a = 0.0
    b = 0.0
    delta = 0.0
    epsilon = 0.0
    
    # Handle the properties where numerical error puts values - but they should
    # be zero. Not all of them are non-zero all the time - but some times
    # they are
    def _zero(self): return 0.0
    def _set_nothing(self, thing): return
    
    d2T_dV2_g = property(_zero, _set_nothing)
    d2V_dT2_g = property(_zero, _set_nothing)
    G_dep_g = property(_zero, _set_nothing)
    H_dep_g = property(_zero, _set_nothing)
    S_dep_g = property(_zero, _set_nothing)
    U_dep_g = property(_zero, _set_nothing)
    A_dep_g = property(_zero, _set_nothing)
    V_dep_g = property(_zero, _set_nothing)
    Cp_dep_g = property(_zero, _set_nothing)
    
    # Replace methods
    dH_dep_dP_g = property(_zero, doc=GCEOS.dH_dep_dP_g)
    dH_dep_dT_g = property(_zero, doc=GCEOS.dH_dep_dT_g)
    dS_dep_dP_g = property(_zero, doc=GCEOS.dS_dep_dP_g)
    dS_dep_dT_g = property(_zero, doc=GCEOS.dS_dep_dT_g)
    dfugacity_dT_g = property(_zero, doc=GCEOS.dfugacity_dT_g)
    dphi_dP_g = property(_zero, doc=GCEOS.dphi_dP_g)
    dphi_dT_g = property(_zero, doc=GCEOS.dphi_dT_g)
 

    def __init__(self, Tc=190.564, Pc=4599000.0, omega=0.008, T=None, P=None, 
                 V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V
        self.Vc = self.Zc*R*Tc/Pc
        
        self.solve()

    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        if not full:
            return 0.0
        else:
            return (0.0, 0.0, 0.0)

    def solve_T(self, P, V, quick=True):
        self.no_T_spec = True
        return P*V*R_inv
            
class PR(GCEOS):
    r'''Class for solving the Peng-Robinson cubic 
    equation of state for a pure compound. Subclasses `GCEOS`, which 
    provides the methods for solving the EOS and calculating its assorted 
    relevant thermodynamic properties. Solves the EOS on initialization. 

    Implemented methods here are `a_alpha_and_derivatives`, which calculates 
    a_alpha and its first and second derivatives, and `solve_T`, which from a 
    specified `P` and `V` obtains `T`.
    
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

        a=0.45724\frac{R^2T_c^2}{P_c}
        
	     b=0.07780\frac{RT_c}{P_c}

        \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2
        
        \kappa=0.37464+1.54226\omega-0.26992\omega^2
        
    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    T-P initialization, and exploring each phase's properties:
    
    >>> eos = PR(Tc=507.6, Pc=3025000.0, omega=0.2975, T=400., P=1E6)
    >>> eos.V_l, eos.V_g
    (0.00015607313188529268, 0.0021418760907613724)
    >>> eos.phase
    'l/g'
    >>> eos.H_dep_l, eos.H_dep_g
    (-26111.868721160878, -3549.2993749373945)
    >>> eos.S_dep_l, eos.S_dep_g
    (-58.09842815106099, -6.439449710478305)
    >>> eos.U_dep_l, eos.U_dep_g
    (-22942.157933046172, -2365.391545698767)
    >>> eos.G_dep_l, eos.G_dep_g
    (-2872.497460736482, -973.5194907460723)
    >>> eos.A_dep_l, eos.A_dep_g
    (297.21332737822377, 210.38833849255525)
    >>> eos.beta_l, eos.beta_g
    (0.0026933709177837514, 0.01012322391117497)
    >>> eos.kappa_l, eos.kappa_g
    (9.33572154382935e-09, 1.9710669809793307e-06)
    >>> eos.Cp_minus_Cv_l, eos.Cp_minus_Cv_g
    (48.510145807408, 44.54414603000346)
    >>> eos.Cv_dep_l, eos.Cp_dep_l
    (18.89210627002112, 59.08779227742912)

    P-T initialization, liquid phase, and round robin trip:
    
    >>> eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.00013022208100139945, -31134.740290463425, -72.47559475426019)
    
    T-V initialization, liquid phase:
    
    >>> eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., V=0.00013022208100139953)
    >>> eos.P, eos.phase
    (1000000.0000020266, 'l')
    
    P-V initialization at same state:
    
    >>> eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00013022208100139953, P=1E6)
    >>> eos.T, eos.phase
    (298.99999999999926, 'l')
    
    Notes
    -----
    The constants in the expresions for `a` and `b` are given to full precision
    in the actual code, as derived in [3]_.
    
    The full expression for critical compressibility is:
        
    .. math::
        Z_c = \frac{1}{32} \left(\sqrt[3]{16 \sqrt{2}-13}-\frac{7}{\sqrt[3]
        {16 \sqrt{2}-13}}+11\right)

    References
    ----------
    .. [1] Peng, Ding-Yu, and Donald B. Robinson. "A New Two-Constant Equation 
       of State." Industrial & Engineering Chemistry Fundamentals 15, no. 1 
       (February 1, 1976): 59-64. doi:10.1021/i160057a011.
    .. [2] Robinson, Donald B., Ding-Yu Peng, and Samuel Y-K Chung. "The 
       Development of the Peng - Robinson Equation and Its Application to Phase
       Equilibrium in a System Containing Methanol." Fluid Phase Equilibria 24,
       no. 1 (January 1, 1985): 25-41. doi:10.1016/0378-3812(85)87035-7. 
    .. [3] Privat, R., and J.-N. Jaubert. "PPR78, a Thermodynamic Model for the
       Prediction of Petroleum Fluid-Phase Behaviour," 11. EDP Sciences, 2011. 
       doi:10.1051/jeep/201100011.
    '''
    # constant part of `a`, 
    # X = (-1 + (6*sqrt(2)+8)**Rational(1,3) - (6*sqrt(2)-8)**Rational(1,3))/3
    # (8*(5*X+1)/(49-37*X)).evalf(40)
    c1 = 0.4572355289213821893834601962251837888504
    
    # Constant part of `b`, (X/(X+3)).evalf(40)
    c2 = 0.0777960739038884559718447100373331839711

#    c1, c2 = 0.45724, 0.07780
    
    # Zc is the mechanical compressibility for mixtures as well.
    Zc = 0.3074013086987038480093850966542222720096

    Psat_coeffs_limiting = [-3.4758880164801873, 0.7675486448347723]
    
    Psat_coeffs_critical = [13.906174756604267, -8.978515559640332, 
                            6.191494729386664, -3.3553014047359286,
                            1.0000000000011509]
    
    Psat_cheb_coeffs = [-7.693430141477579, -7.792157693145173, -0.12584439451814622, 0.0045868660863990305,
                        0.011902728116315585, -0.00809984848593371, 0.0035807374586641324, -0.001285457896498948,
                        0.0004379441379448949, -0.0001701325511665626, 7.889450459420399e-05, -3.842330780886875e-05, 
                        1.7884847876342805e-05, -7.9432179091441e-06, 3.51726370898656e-06, -1.6108797741557683e-06, 
                        7.625638345550717e-07, -3.6453554523813245e-07, 1.732454904858089e-07, -8.195124459058523e-08, 
                        3.8929380082904216e-08, -1.8668536344161905e-08, 9.021955971552252e-09, -4.374277331168795e-09,
                        2.122697092724708e-09, -1.0315557015083254e-09, 5.027805333255708e-10, -2.4590905784642285e-10, 
                        1.206301486380689e-10, -5.932583414867791e-11, 2.9274476912683964e-11, -1.4591650777202522e-11, 
                        7.533835507484918e-12, -4.377200831613345e-12, 1.7413208326438542e-12]
    Psat_cheb_coeffs_der = chebder(Psat_cheb_coeffs)
    Psat_coeffs_critical_der = polyder(Psat_coeffs_critical[::-1])[::-1]
    Psat_cheb_constant_factor = (-2.355355160853182, 0.42489124941587103)
    
    phi_sat_coeffs = [4.040440857039882e-09, -1.512382901024055e-07, 2.5363900091436416e-06,
                      -2.4959001060510725e-05, 0.00015714708105355206, -0.0006312347348814933,
                      0.0013488647482434379, 0.0008510254890166079, -0.017614759099592196,
                      0.06640627813169839, -0.13427456425899886, 0.1172205279608668, 
                      0.13594473870160448, -0.5560225934266592, 0.7087599054079694, 
                      0.6426353018023558]

    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        Tc_Pc = Tc/Pc
        self.a = self.c1*R2*Tc*Tc_Pc
        self.b = self.c2*R*Tc_Pc
        self.kappa = omega*(-0.26992*omega + 1.54226) + 0.37464
        self.delta = 2.*self.b
        self.epsilon = -self.b*self.b
        self.Vc = self.Zc*R*Tc_Pc
        
        self.solve()

    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `Tc`, `kappa`, and `a`. 
        
        For use in `solve_T`, returns only `a_alpha` if full is False.

        .. math::
            a\alpha = a \left(\kappa \left(- \frac{T^{0.5}}{Tc^{0.5}} 
            + 1\right) + 1\right)^{2}
        
            \frac{d a\alpha}{dT} = - \frac{1.0 a \kappa}{T^{0.5} Tc^{0.5}}
            \left(\kappa \left(- \frac{T^{0.5}}{Tc^{0.5}} + 1\right) + 1\right)

            \frac{d^2 a\alpha}{dT^2} = 0.5 a \kappa \left(- \frac{1}{T^{1.5} 
            Tc^{0.5}} \left(\kappa \left(\frac{T^{0.5}}{Tc^{0.5}} - 1\right)
            - 1\right) + \frac{\kappa}{T^{1.0} Tc^{1.0}}\right)
        '''
        if not full:
            return self.a*(1.0 + self.kappa*(1.0 - (T/self.Tc)**0.5))**2
        else:
            if quick:
                Tc, kappa = self.Tc, self.kappa
                x0 = T**0.5
                x1 = Tc**-0.5
                x2 = kappa*(x0*x1 - 1.) - 1.
                x3 = self.a*kappa
                x4 = x1*x2
                
                a_alpha = self.a*x2*x2
                da_alpha_dT = x4*x3/x0
                d2a_alpha_dT2 = 0.5*x3*(kappa/(T*Tc) - x4/(x0*T))
            else:
                a_alpha = self.a*(1 + self.kappa*(1-(T/self.Tc)**0.5))**2
                da_alpha_dT = -self.a*self.kappa*sqrt(T/self.Tc)*(self.kappa*(-sqrt(T/self.Tc) + 1.) + 1.)/T
                d2a_alpha_dT2 = self.a*self.kappa*(self.kappa/self.Tc - sqrt(T/self.Tc)*(self.kappa*(sqrt(T/self.Tc) - 1.) - 1.)/T)/(2.*T)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def solve_T(self, P, V, quick=True):
        r'''Method to calculate `T` from a specified `P` and `V` for the PR
        EOS. Uses `Tc`, `a`, `b`, and `kappa` as well, obtained from the 
        class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (3x faster) or 
            individual formulas

        Returns
        -------
        T : float
            Temperature, [K]
        
        Notes
        -----
        The exact solution can be derived as follows, and is excluded for 
        breviety.
        
        >>> from sympy import *
        >>> P, T, V = symbols('P, T, V')
        >>> Tc, Pc, omega = symbols('Tc, Pc, omega')
        >>> R, a, b, kappa = symbols('R, a, b, kappa')
        
        >>> a_alpha = a*(1 + kappa*(1-sqrt(T/Tc)))**2
        >>> PR_formula = R*T/(V-b) - a_alpha/(V*(V+b)+b*(V-b)) - P
        >>> #solve(PR_formula, T)
        '''
        self.no_T_spec = True
        Tc, a, b, kappa = self.Tc, self.a, self.b, self.kappa
        if quick:
            x0 = V*V
            x1 = R*Tc
            x2 = x0*x1
            x3 = kappa*kappa
            x4 = a*x3
            x5 = b*x4
            x6 = 2.*V*b
            x7 = x1*x6
            x8 = b*b
            x9 = x1*x8
            x10 = V*x4
            x11 = (-x10 + x2 + x5 + x7 - x9)**2
            x12 = x0*x0
            x13 = R*R
            x14 = Tc*Tc
            x15 = x13*x14
            x16 = x8*x8
            x17 = a*a
            x18 = x3*x3
            x19 = x17*x18
            x20 = x0*V
            x21 = 2.*R*Tc*a*x3
            x22 = x8*b
            x23 = 4.*V*x22
            x24 = 4.*b*x20
            x25 = a*x1
            x26 = x25*x8
            x27 = x26*x3
            x28 = x0*x25
            x29 = x28*x3
            x30 = 2.*x8
            x31 = 6.*V*x27 - 2.*b*x29 + x0*x13*x14*x30 + x0*x19 + x12*x15 + x15*x16 - x15*x23 + x15*x24 - x19*x6 + x19*x8 - x20*x21 - x21*x22
            x32 = V - b
            x33 = 2.*(R*Tc*a*kappa)
            x34 = P*x2
            x35 = P*x5
            x36 = x25*x3
            x37 = P*x10
            x38 = P*R*Tc
            x39 = V*x17
            x40 = 2.*kappa*x3
            x41 = b*x17
            x42 = P*a*x3
            return -Tc*(2.*a*kappa*x11*sqrt(x32**3*(x0 + x6 - x8)*(P*x7 - P*x9 + x25 + x33 + x34 + x35 + x36 - x37))*(kappa + 1.) - x31*x32*((4.*V)*(R*Tc*a*b*kappa) + x0*x33 - x0*x35 + x12*x38 + x16*x38 + x18*x39 - x18*x41 - x20*x42 - x22*x42 - x23*x38 + x24*x38 + x25*x6 - x26 - x27 + x28 + x29 + x3*x39 - x3*x41 + x30*x34 - x33*x8 + x36*x6 + 3*x37*x8 + x39*x40 - x40*x41))/(x11*x31)
        else:
            return Tc*(-2*a*kappa*sqrt((V - b)**3*(V**2 + 2*V*b - b**2)*(P*R*Tc*V**2 + 2*P*R*Tc*V*b - P*R*Tc*b**2 - P*V*a*kappa**2 + P*a*b*kappa**2 + R*Tc*a*kappa**2 + 2*R*Tc*a*kappa + R*Tc*a))*(kappa + 1)*(R*Tc*V**2 + 2*R*Tc*V*b - R*Tc*b**2 - V*a*kappa**2 + a*b*kappa**2)**2 + (V - b)*(R**2*Tc**2*V**4 + 4*R**2*Tc**2*V**3*b + 2*R**2*Tc**2*V**2*b**2 - 4*R**2*Tc**2*V*b**3 + R**2*Tc**2*b**4 - 2*R*Tc*V**3*a*kappa**2 - 2*R*Tc*V**2*a*b*kappa**2 + 6*R*Tc*V*a*b**2*kappa**2 - 2*R*Tc*a*b**3*kappa**2 + V**2*a**2*kappa**4 - 2*V*a**2*b*kappa**4 + a**2*b**2*kappa**4)*(P*R*Tc*V**4 + 4*P*R*Tc*V**3*b + 2*P*R*Tc*V**2*b**2 - 4*P*R*Tc*V*b**3 + P*R*Tc*b**4 - P*V**3*a*kappa**2 - P*V**2*a*b*kappa**2 + 3*P*V*a*b**2*kappa**2 - P*a*b**3*kappa**2 + R*Tc*V**2*a*kappa**2 + 2*R*Tc*V**2*a*kappa + R*Tc*V**2*a + 2*R*Tc*V*a*b*kappa**2 + 4*R*Tc*V*a*b*kappa + 2*R*Tc*V*a*b - R*Tc*a*b**2*kappa**2 - 2*R*Tc*a*b**2*kappa - R*Tc*a*b**2 + V*a**2*kappa**4 + 2*V*a**2*kappa**3 + V*a**2*kappa**2 - a**2*b*kappa**4 - 2*a**2*b*kappa**3 - a**2*b*kappa**2))/((R*Tc*V**2 + 2*R*Tc*V*b - R*Tc*b**2 - V*a*kappa**2 + a*b*kappa**2)**2*(R**2*Tc**2*V**4 + 4*R**2*Tc**2*V**3*b + 2*R**2*Tc**2*V**2*b**2 - 4*R**2*Tc**2*V*b**3 + R**2*Tc**2*b**4 - 2*R*Tc*V**3*a*kappa**2 - 2*R*Tc*V**2*a*b*kappa**2 + 6*R*Tc*V*a*b**2*kappa**2 - 2*R*Tc*a*b**3*kappa**2 + V**2*a**2*kappa**4 - 2*V*a**2*b*kappa**4 + a**2*b**2*kappa**4))
    
        
    
    
    
#a = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=400., P=1E6)
#print(a.d2V_dPdT_g, a.V_g)
##
#b = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., V=0.00013022208100139953)
#print(b.d2V_dPdT_l, b.PIP_l, b.V_l, b.P)
#
#c = PR(Tc=507.6, Pc=3025000, omega=0.2975, V=0.00013022208100139953, P=1E6)
#print(c.d2V_dPdT_l, c.PIP_l, c.V_l, c.T)


class PR78(PR):
    r'''Class for solving the Peng-Robinson cubic 
    equation of state for a pure compound according to the 1978 variant.
    Subclasses `PR`, which provides everything except the variable `kappa`.
    Solves the EOS on initialization. See `PR` for further documentation.
    
    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

        a=0.45724\frac{R^2T_c^2}{P_c}
        
	  b=0.07780\frac{RT_c}{P_c}

        \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2
        
        \kappa_i = 0.37464+1.54226\omega_i-0.26992\omega_i^2 \text{ if } \omega_i
        \le 0.491
        
        \kappa_i = 0.379642 + 1.48503 \omega_i - 0.164423\omega_i^2 + 0.016666
        \omega_i^3 \text{ if } \omega_i > 0.491
        
    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    P-T initialization (furfuryl alcohol), liquid phase:
    
    >>> eos = PR78(Tc=632, Pc=5350000, omega=0.734, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 8.351960066075009e-05, -63764.649480508735, -130.73710891262687)
    
    Notes
    -----
    This variant is recommended over the original.

    References
    ----------
    .. [1] Robinson, Donald B, and Ding-Yu Peng. The Characterization of the 
       Heptanes and Heavier Fractions for the GPA Peng-Robinson Programs. 
       Tulsa, Okla.: Gas Processors Association, 1978.
    .. [2] Robinson, Donald B., Ding-Yu Peng, and Samuel Y-K Chung. "The 
       Development of the Peng - Robinson Equation and Its Application to Phase
       Equilibrium in a System Containing Methanol." Fluid Phase Equilibria 24,
       no. 1 (January 1, 1985): 25-41. doi:10.1016/0378-3812(85)87035-7.  
    '''
    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = 2*self.b
        self.epsilon = -self.b*self.b
        self.Vc = self.Zc*R*self.Tc/self.Pc

        if omega <= 0.491:
            self.kappa = 0.37464 + 1.54226*omega - 0.26992*omega*omega
        else:
            self.kappa = 0.379642 + 1.48503*omega - 0.164423*omega**2 + 0.016666*omega**3

        self.solve()


class PRSV(PR):
    r'''Class for solving the Peng-Robinson-Stryjek-Vera equations of state for
    a pure compound as given in [1]_. The same as the Peng-Robinson EOS,
    except with a different `kappa` formula and with an optional fit parameter.
    Subclasses `PR`, which provides only several constants. See `PR` for 
    further documentation and examples.
    
    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

        a=0.45724\frac{R^2T_c^2}{P_c}
        
        b=0.07780\frac{RT_c}{P_c}

        \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2
        
        \kappa = \kappa_0 + \kappa_1(1 + T_r^{0.5})(0.7 - T_r)
        
        \kappa_0 = 0.378893 + 1.4897153\omega - 0.17131848\omega^2 
        + 0.0196554\omega^3
        
    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    kappa1 : float, optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]

    Examples
    --------
    P-T initialization (hexane, with fit parameter in [1]_), liquid phase:
    
    >>> eos = PRSV(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.000130126869448406, -31698.916002476693, -74.16749024350415)
    
    Notes
    -----
    [1]_ recommends that `kappa1` be set to 0 for Tr > 0.7. This is not done by 
    default; the class boolean `kappa1_Tr_limit` may be set to True and the
    problem re-solved with that specified if desired. `kappa1_Tr_limit` is not
    supported for P-V inputs.
    
    Solutions for P-V solve for `T` with SciPy's `newton` solver, as there is no
    analytical solution for `T`
    
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
    kappa1_Tr_limit = False
    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None, kappa1=0):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V
        self.kwargs = {'kappa1': kappa1}
        
        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = 2*self.b
        self.epsilon = -self.b*self.b
        self.kappa0 = 0.378893 + 1.4897153*omega - 0.17131848*omega**2 + 0.0196554*omega**3
        self.Vc = self.Zc*R*self.Tc/self.Pc

        self.check_sufficient_inputs()
        if self.V and self.P:
            # Deal with T-solution here; does NOT support kappa1_Tr_limit.
            self.kappa1 = kappa1
            self.T = self.solve_T(self.P, self.V)
            Tr = self.T/Tc
        else:
            Tr = self.T/Tc
            if self.kappa1_Tr_limit and Tr > 0.7:
                self.kappa1 = 0
            else:
                self.kappa1 = kappa1
    
        self.kappa = self.kappa0 + self.kappa1*(1 + Tr**0.5)*(0.7 - Tr)
        self.solve()

    def solve_T(self, P, V, quick=True):
        r'''Method to calculate `T` from a specified `P` and `V` for the PRSV
        EOS. Uses `Tc`, `a`, `b`, `kappa0`  and `kappa` as well, obtained from  
        the class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (somewhat faster) or 
            individual formulas.

        Returns
        -------
        T : float
            Temperature, [K]
        
        Notes
        -----
        Not guaranteed to produce a solution. There are actually two solution,
        one much higher than normally desired; it is possible the solver could
        converge on this.        
        '''
        Tc, a, b, kappa0, kappa1 = self.Tc, self.a, self.b, self.kappa0, self.kappa1
        self.no_T_spec = True
        if quick:
            x0 = V - b
            R_x0 = R/x0
            x3 = (100.*(V*(V + b) + b*x0))
            x4 = 10.*kappa0
            kappa110 = kappa1*10.
            kappa17 = kappa1*7.
            def to_solve(T):
                x1 = T/Tc
                x2 = x1**0.5
                return (T*R_x0 - a*((x4 - (kappa110*x1 - kappa17)*(x2 + 1.))*(x2 - 1.) - 10.)**2/x3) - P
        else:
            def to_solve(T):
                P_calc = R*T/(V - b) - a*((kappa0 + kappa1*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))*(-sqrt(T/Tc) + 1) + 1)**2/(V*(V + b) + b*(V - b))
                return P_calc - P
        return newton(to_solve, Tc*0.5)

    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `Tc`, `kappa0`, `kappa1`, and 
        `a`. 
        
        For use in root-finding, returns only `a_alpha` if full is False.

        The `a_alpha` function is shown below; its first and second derivatives
        are long available through the SymPy expression under it.

        .. math::
            a\alpha = a \left(\left(\kappa_{0} + \kappa_{1} \left(\sqrt{\frac{
            T}{Tc}} + 1\right) \left(- \frac{T}{Tc} + \frac{7}{10}\right)
            \right) \left(- \sqrt{\frac{T}{Tc}} + 1\right) + 1\right)^{2}
            
        >>> from sympy import *
        >>> P, T, V = symbols('P, T, V')
        >>> Tc, Pc, omega = symbols('Tc, Pc, omega')
        >>> R, a, b, kappa0, kappa1 = symbols('R, a, b, kappa0, kappa1')
        >>> kappa = kappa0 + kappa1*(1 + sqrt(T/Tc))*(Rational(7, 10)-T/Tc)
        >>> a_alpha = a*(1 + kappa*(1-sqrt(T/Tc)))**2
        >>> # diff(a_alpha, T)
        >>> # diff(a_alpha, T, 2)
        '''
        Tc, a, kappa0, kappa1 = self.Tc, self.a, self.kappa0, self.kappa1
        if not full:
            return a*((kappa0 + kappa1*(sqrt(T/Tc) + 1)*(-T/Tc + 0.7))*(-sqrt(T/Tc) + 1) + 1)**2
        else:
            if quick:
                x1 = T/Tc
                x2 = x1**0.5
                x3 = x2 - 1.
                x4 = 10.*x1 - 7.
                x5 = x2 + 1.
                x6 = 10.*kappa0 - kappa1*x4*x5
                x7 = x3*x6
                x8 = x7*0.1 - 1.
                x10 = x6/T
                x11 = kappa1*x3
                x12 = x4/T
                x13 = 20./Tc*x5 + x12*x2
                x14 = -x10*x2 + x11*x13
                a_alpha = a*x8*x8
                da_alpha_dT = -a*x14*x8*0.1
                d2a_alpha_dT2 = a*(x14*x14 - x2/T*(x7 - 10.)*(2.*kappa1*x13 + x10 + x11*(40./Tc - x12)))/200.
            else:
                a_alpha = a*((kappa0 + kappa1*(sqrt(T/Tc) + 1)*(-T/Tc + 0.7))*(-sqrt(T/Tc) + 1) + 1)**2
                da_alpha_dT = a*((kappa0 + kappa1*(sqrt(T/Tc) + 1)*(-T/Tc + 0.7))*(-sqrt(T/Tc) + 1) + 1)*(2*(-sqrt(T/Tc) + 1)*(-kappa1*(sqrt(T/Tc) + 1)/Tc + kappa1*sqrt(T/Tc)*(-T/Tc + 0.7)/(2*T)) - sqrt(T/Tc)*(kappa0 + kappa1*(sqrt(T/Tc) + 1)*(-T/Tc + 0.7))/T)
                d2a_alpha_dT2 = a*((kappa1*(sqrt(T/Tc) - 1)*(20*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(10*T/Tc - 7)/T) - sqrt(T/Tc)*(10*kappa0 - kappa1*(sqrt(T/Tc) + 1)*(10*T/Tc - 7))/T)**2 - sqrt(T/Tc)*((10*kappa0 - kappa1*(sqrt(T/Tc) + 1)*(10*T/Tc - 7))*(sqrt(T/Tc) - 1) - 10)*(kappa1*(40/Tc - (10*T/Tc - 7)/T)*(sqrt(T/Tc) - 1) + 2*kappa1*(20*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(10*T/Tc - 7)/T) + (10*kappa0 - kappa1*(sqrt(T/Tc) + 1)*(10*T/Tc - 7))/T)/T)/200
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

            
class PRSV2(PR):
    r'''Class for solving the Peng-Robinson-Stryjek-Vera 2 equations of state 
    for a pure compound as given in [1]_. The same as the Peng-Robinson EOS,
    except with a different `kappa` formula and with three fit parameters.
    Subclasses `PR`, which provides only several constants. See `PR` for 
    further documentation and examples. PRSV provides only one constant.
    
    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

        a=0.45724\frac{R^2T_c^2}{P_c}
        
	  b=0.07780\frac{RT_c}{P_c}

        \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2
        
        \kappa = \kappa_0 + [\kappa_1 + \kappa_2(\kappa_3 - T_r)(1-T_r^{0.5})]
        (1 + T_r^{0.5})(0.7 - T_r)
        
        \kappa_0 = 0.378893 + 1.4897153\omega - 0.17131848\omega^2 
        + 0.0196554\omega^3
        
    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    kappa1 : float, optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]
    kappa2 : float, optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]
    kappa : float, optional
        Fit parameter; available in [1]_ for over 90 compounds, [-]

    Examples
    --------
    P-T initialization (hexane, with fit parameter in [1]_), liquid phase:
    
    >>> eos = PRSV2(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6, kappa1=0.05104, kappa2=0.8634, kappa3=0.460)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.00013018821346475235, -31496.173493225797, -73.61525801151421)
    
    Notes
    -----
    Solutions for P-V solve for `T` with SciPy's `newton` solver, as there is 
    no analytical solution for `T`
    
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
    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None, kappa1=0, kappa2=0, kappa3=0):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V
        self.check_sufficient_inputs()
        self.kwargs = {'kappa1': kappa1, 'kappa2': kappa2, 'kappa3': kappa3}
        
        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = 2*self.b
        self.epsilon = -self.b*self.b
        self.kappa0 = 0.378893 + 1.4897153*omega - 0.17131848*omega*omega + 0.0196554*omega*omega*omega
        self.kappa1, self.kappa2, self.kappa3 = kappa1, kappa2, kappa3
        self.Vc = self.Zc*R*self.Tc/self.Pc

        if self.V and self.P:
            # Deal with T-solution here
            self.T = self.solve_T(self.P, self.V)
        Tr = self.T/Tc
    
        self.kappa = self.kappa0 + ((self.kappa1 + self.kappa2*(self.kappa3 
                                     - Tr)*(1 - Tr**0.5))*(1 + Tr**0.5)*(0.7 - Tr))
        self.solve()

    def solve_T(self, P, V, quick=True):
        r'''Method to calculate `T` from a specified `P` and `V` for the PRSV2
        EOS. Uses `Tc`, `a`, `b`, `kappa0`, `kappa1`, `kappa2`, and `kappa3`
        as well, obtained from the class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (somewhat faster) or 
            individual formulas.

        Returns
        -------
        T : float
            Temperature, [K]
        
        Notes
        -----
        Not guaranteed to produce a solution. There are actually 8 solutions,
        six with an imaginary component at a tested point. The two temperature
        solutions are quite far apart, with one much higher than the other;
        it is possible the solver could converge on the higher solution, so use
        `T` inputs with care. This extra solution is a perfectly valid one
        however.
        '''
        # Generic solution takes 72 vs 56 microseconds for the optimized version below
#        return super(PR, self).solve_T(P, V, quick=quick) 
        self.no_T_spec = True
        Tc, a, b, kappa0, kappa1, kappa2, kappa3 = self.Tc, self.a, self.b, self.kappa0, self.kappa1, self.kappa2, self.kappa3
        if quick:
            x0 = V - b
            R_x0 = R/x0
            x5 = (100.*(V*(V + b) + b*x0))
            x4 = 10.*kappa0
            def to_solve(T):
                x1 = T/Tc
                x2 = x1**0.5
                x3 = x2 - 1.
                return (R_x0*T - a*(x3*(x4 - (kappa1 + kappa2*x3*(-kappa3 + x1))*(10.*x1 - 7.)*(x2 + 1.)) - 10.)**2/x5) - P
        else:
            def to_solve(T):
                P_calc = R*T/(V - b) - a*((kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))*(-sqrt(T/Tc) + 1) + 1)**2/(V*(V + b) + b*(V - b))
                return P_calc - P
        return newton(to_solve, Tc*0.5)


    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `Tc`, `kappa0`, `kappa1`,
        `kappa2`, `kappa3`, and `a`. 
        
        For use in `solve_T`, returns only `a_alpha` if full is False.
        
        The first and second derivatives of `a_alpha` are available through the
        following SymPy expression.

        >>> from sympy import *
        >>> P, T, V = symbols('P, T, V')
        >>> Tc, Pc, omega = symbols('Tc, Pc, omega')
        >>> R, a, b, kappa0, kappa1, kappa2, kappa3 = symbols('R, a, b, kappa0, kappa1, kappa2, kappa3')
        >>> Tr = T/Tc
        >>> kappa = kappa0 + (kappa1 + kappa2*(kappa3-Tr)*(1-sqrt(Tr)))*(1+sqrt(Tr))*(Rational('0.7')-Tr)
        >>> a_alpha = a*(1 + kappa*(1-sqrt(T/Tc)))**2
        >>> # diff(a_alpha, T)
        >>> # diff(a_alpha, T, 2)
        '''
        Tc, a, kappa0, kappa1, kappa2, kappa3 = self.Tc, self.a, self.kappa0, self.kappa1, self.kappa2, self.kappa3
        
        if not full:
            Tr = T/Tc
            kappa = kappa0 + ((kappa1 + kappa2*(kappa3 - Tr)*(1 - Tr**0.5))*(1 + Tr**0.5)*(0.7 - Tr))
            return a*(1 + kappa*(1-sqrt(T/Tc)))**2
        else:
            if quick:
                x1 = T/Tc
                x2 = sqrt(x1)
                x3 = x2 - 1.
                x4 = x2 + 1.
                x5 = 10.*x1 - 7.
                x6 = -kappa3 + x1
                x7 = kappa1 + kappa2*x3*x6
                x8 = x5*x7
                x9 = 10.*kappa0 - x4*x8
                x10 = x3*x9
                x11 = x10*0.1 - 1.
                x13 = x2/T
                x14 = x7/Tc
                x15 = kappa2*x4*x5
                x16 = 2.*(-x2 + 1.)/Tc + x13*(kappa3 - x1)
                x17 = -x13*x8 - x14*(20.*x2 + 20.) + x15*x16
                x18 = x13*x9 + x17*x3
                x19 = x2/(T*T)
                x20 = 2.*x2/T
                
                a_alpha = a*x11*x11
                da_alpha_dT = a*x11*x18*0.1
                d2a_alpha_dT2 = a*(x18*x18 + (x10 - 10.)*(x17*x20 - x19*x9 + x3*(40.*kappa2/Tc*x16*x4 + kappa2*x16*x20*x5 - 40./T*x14*x2 - x15/T*x2*(4./Tc - x6/T) + x19*x8)))/200.
            else:
                a_alpha = a*(1 + self.kappa*(1-sqrt(T/Tc)))**2
                da_alpha_dT = a*((kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))*(-sqrt(T/Tc) + 1) + 1)*(2*(-sqrt(T/Tc) + 1)*((sqrt(T/Tc) + 1)*(-T/Tc + 7/10)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T)) - (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(-T/Tc + 7/10)/(2*T)) - sqrt(T/Tc)*(kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))/T)
                d2a_alpha_dT2 = a*((kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))*(-sqrt(T/Tc) + 1) + 1)*(2*(-sqrt(T/Tc) + 1)*((sqrt(T/Tc) + 1)*(-T/Tc + 7/10)*(kappa2*sqrt(T/Tc)/(T*Tc) + kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(4*T**2)) - 2*(sqrt(T/Tc) + 1)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T))/Tc + sqrt(T/Tc)*(-T/Tc + 7/10)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T))/T - sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))/(T*Tc) - sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(-T/Tc + 7/10)/(4*T**2)) - 2*sqrt(T/Tc)*((sqrt(T/Tc) + 1)*(-T/Tc + 7/10)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T)) - (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(-T/Tc + 7/10)/(2*T))/T + sqrt(T/Tc)*(kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))/(2*T**2)) + a*((-sqrt(T/Tc) + 1)*((sqrt(T/Tc) + 1)*(-T/Tc + 7/10)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T)) - (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(-T/Tc + 7/10)/(2*T)) - sqrt(T/Tc)*(kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))/(2*T))*(2*(-sqrt(T/Tc) + 1)*((sqrt(T/Tc) + 1)*(-T/Tc + 7/10)*(-kappa2*(-sqrt(T/Tc) + 1)/Tc - kappa2*sqrt(T/Tc)*(-T/Tc + kappa3)/(2*T)) - (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)/Tc + sqrt(T/Tc)*(kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(-T/Tc + 7/10)/(2*T)) - sqrt(T/Tc)*(kappa0 + (kappa1 + kappa2*(-sqrt(T/Tc) + 1)*(-T/Tc + kappa3))*(sqrt(T/Tc) + 1)*(-T/Tc + 7/10))/T)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2


class VDW(GCEOS):
    r'''Class for solving the Van der Waals cubic 
    equation of state for a pure compound. Subclasses `GCEOS`, which 
    provides the methods for solving the EOS and calculating its assorted 
    relevant thermodynamic properties. Solves the EOS on initialization. 

    Implemented methods here are `a_alpha_and_derivatives`, which sets 
    a_alpha and its first and second derivatives, and `solve_T`, which from a 
    specified `P` and `V` obtains `T`. `main_derivatives_and_departures` is
    a re-implementation with VDW specific methods, as the general solution
    has ZeroDivisionError errors.
    
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P=\frac{RT}{V-b}-\frac{a}{V^2}
        
        a=\frac{27}{64}\frac{(RT_c)^2}{P_c}

        b=\frac{RT_c}{8P_c}
    
    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    omega : float, optional
        Acentric factor - not used in equation of state!, [-]
        
    Examples
    --------    
    >>> eos = VDW(Tc=507.6, Pc=3025000, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.00022332978038490077, -13385.722837649315, -32.65922018109096)

    Notes
    -----
    `omega` is allowed as an input for compatibility with the other EOS forms,
    but is not used.

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th 
       edition. New York: McGraw-Hill Professional, 2000.
    .. [2] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
       Butterworth-Heinemann, 1985.
    '''
    delta = 0
    epsilon = 0
    omega = None
    Zc = 3/8.
    
    Psat_coeffs_limiting = [-3.0232164484175756, 0.20980668241160666]
    
    Psat_coeffs_critical = [9.575399398167086, -5.742004486758378, 
                            4.8000085098196745, -3.000000002903554,
                            1.0000000000002651]

    Psat_cheb_coeffs = [-3.0938407448693392, -3.095844800654779, -0.01852425171597184, -0.009132810281704463,
                        0.0034478548769173167, -0.0007513250489879469, 0.0001425235859202672, -3.18455900032599e-05, 
                        8.318773833859442e-06, -2.125810773856036e-06, 5.171012493290658e-07, -1.2777009201877978e-07, 
                        3.285945705657834e-08, -8.532047244427343e-09, 2.196978792832582e-09, -5.667409821199761e-10,
                        1.4779624173003134e-10, -3.878590467732996e-11, 1.0181633097391951e-11, -2.67662653922595e-12, 
                        7.053635426397184e-13, -1.872821965868618e-13, 4.9443291800198297e-14, -1.2936198878592264e-14,
                        2.9072203628840998e-15, -4.935694864968698e-16, 2.4160767787481663e-15, 8.615748088927622e-16, 
                        -5.198342841253312e-16, -2.19739320055784e-15, -1.0876309618559898e-15, 7.727786509661994e-16,
                        7.958450521858285e-16, 2.088444434750203e-17, -1.3864912907016191e-16]
    Psat_cheb_constant_factor = (-1.0630005005005003, 0.9416200294550813)
    Psat_cheb_coeffs_der = chebder(Psat_cheb_coeffs)
    Psat_coeffs_critical_der = polyder(Psat_coeffs_critical[::-1])[::-1]
    
    phi_sat_coeffs = [-4.703247660146169e-06, 7.276853488756492e-05, -0.0005008397610615123,
                      0.0019560274384829595, -0.004249875101260566, 0.001839985687730564,
                      0.02021191780955066, -0.07056928933569773, 0.09941120467466309, 
                      0.021295687530901747, -0.32582447905247514, 0.521321793740683,
                      0.6950957738017804]

    def __init__(self, Tc, Pc, T=None, P=None, V=None, omega=None):
        self.Tc = Tc
        self.Pc = Pc
        self.T = T
        self.P = P
        self.V = V

        self.a = 27.0/64.0*(R*Tc)**2/Pc
        self.b = R*Tc/(8.*Pc)
        self.Vc = self.Zc*R*self.Tc/self.Pc
        self.solve()

    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `a`.
        
        .. math::
            a\alpha = a
        
            \frac{d a\alpha}{dT} = 0

            \frac{d^2 a\alpha}{dT^2} = 0
        '''
        if not full:
            return self.a
        else:
            a_alpha = self.a
            da_alpha_dT = 0.0
            d2a_alpha_dT2 = 0.0
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def solve_T(self, P, V):
        r'''Method to calculate `T` from a specified `P` and `V` for the VDW
        EOS. Uses `a`, and `b`, obtained from the class's namespace.

        .. math::
            T =  \frac{1}{R V^{2}} \left(P V^{2} \left(V - b\right)
            + V a - a b\right)

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]

        Returns
        -------
        T : float
            Temperature, [K]
        '''
        self.no_T_spec = True
        return (P*V**2*(V - self.b) + V*self.a - self.a*self.b)/(R*V**2)
    
    @staticmethod
    def main_derivatives_and_departures(T, P, V, b, delta, epsilon, a_alpha,
                                        da_alpha_dT, d2a_alpha_dT2, quick=True):
        '''Re-implementation of derivatives and excess property calculations, 
        as ZeroDivisionError errors occur with the general solution. The 
        following derivation is the source of these formulas.
        
        >>> from sympy import *
        >>> P, T, V, R, b, a = symbols('P, T, V, R, b, a')
        >>> P_vdw = R*T/(V-b) - a/(V*V)
        >>> vdw = P_vdw - P
        >>> 
        >>> dP_dT = diff(vdw, T)
        >>> dP_dV = diff(vdw, V)
        >>> d2P_dT2 = diff(vdw, T, 2)
        >>> d2P_dV2 = diff(vdw, V, 2)
        >>> d2P_dTdV = diff(vdw, T, V)
        >>> H_dep = integrate(T*dP_dT - P_vdw, (V, oo, V))
        >>> H_dep += P*V - R*T
        >>> S_dep = integrate(dP_dT - R/V, (V,oo,V))
        >>> S_dep += R*log(P*V/(R*T))
        >>> Cv_dep = T*integrate(d2P_dT2, (V,oo,V))
        >>> 
        >>> dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep
        (R/(V - b), -R*T/(V - b)**2 + 2*a/V**3, 0, 2*(R*T/(V - b)**3 - 3*a/V**4), -R/(V - b)**2, P*V - R*T - a/V, R*(-log(V) + log(V - b)) + R*log(P*V/(R*T)), 0)
        '''
        dP_dT = R/(V - b)
        dP_dV = -R*T/(V - b)**2 + 2*a_alpha/V**3
        d2P_dT2 = 0
        d2P_dV2 = 2*(R*T/(V - b)**3 - 3*a_alpha/V**4)
        d2P_dTdV = -R/(V - b)**2
        H_dep = P*V - R*T - a_alpha/V
        S_dep = R*(-log(V) + log(V - b)) + R*log(P*V/(R*T))
        Cv_dep = 0
        return [dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep]

        

class RK(GCEOS):
    r'''Class for solving the Redlich-Kwong cubic 
    equation of state for a pure compound. Subclasses `GCEOS`, which 
    provides the methods for solving the EOS and calculating its assorted 
    relevant thermodynamic properties. Solves the EOS on initialization. 

    Implemented methods here are `a_alpha_and_derivatives`, which sets 
    a_alpha and its first and second derivatives, and `solve_T`, which from a 
    specified `P` and `V` obtains `T`. 
    
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P =\frac{RT}{V-b}-\frac{a}{V\sqrt{T}(V+b)}
        
        a=\left(\frac{R^2(T_c)^{2.5}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2.5}}{P_c}
        
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}
    
    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------    
    >>> eos = RK(Tc=507.6, Pc=3025000, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.00015189341729751862, -26160.833620674086, -63.01311649400544)
    
    Notes
    -----
    `omega` is allowed as an input for compatibility with the other EOS forms,
    but is not used.

    References
    ----------
    .. [1] Redlich, Otto., and J. N. S. Kwong. "On the Thermodynamics of 
       Solutions. V. An Equation of State. Fugacities of Gaseous Solutions." 
       Chemical Reviews 44, no. 1 (February 1, 1949): 233-44. 
       doi:10.1021/cr60137a013.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th 
       edition. New York: McGraw-Hill Professional, 2000.
    .. [3] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
       Butterworth-Heinemann, 1985.
    '''
    c1 = 0.4274802335403414043909906940611707345513 # 1/(9*(2**(1/3.)-1)) 
    c2 = 0.08664034996495772158907020242607611685675 # (2**(1/3.)-1)/3 
    epsilon = 0
    omega = None
    Zc = 1/3.

    Psat_coeffs_limiting = [-72.700288369511583, -68.76714163049]
    Psat_coeffs_critical = [1129250.3276866912, 4246321.053155941,
                            5988691.4873851035, 3754317.4112657467, 
                            882716.2189281426]

    Psat_cheb_coeffs = [-6.8488798834192215, -6.93992806360099, -0.11216113842675507, 0.0022494496508455135, 
                        0.00995148012561513, -0.005789786392208277, 0.0021454644555051177, -0.0006192510387981658,
                        0.00016870584348326536, -5.828094356536212e-05, 2.5829410448955883e-05, -1.1312372380559225e-05,
                        4.374040785359406e-06, -1.5546789700246184e-06, 5.666723613325655e-07, -2.2701147218271074e-07,
                        9.561199996134724e-08, -3.934646467524511e-08, 1.55272396700466e-08, -6.061097474369418e-09,
                        2.4289648176102022e-09, -1.0031987621530753e-09, 4.168016003137324e-10, -1.7100917451312765e-10,
                        6.949731049432813e-11, -2.8377758503521713e-11, 1.1741734564892428e-11, -4.891469634936765e-12, 
                        2.0373765879672795e-12, -8.507821454718095e-13, 3.4975627537410514e-13, -1.4468659018281038e-13,
                        6.536766028637786e-14, -2.7636123641275323e-14, 1.105377996166862e-14]
    Psat_cheb_constant_factor = (0.8551757791729341, 9.962912449541513)
    Psat_cheb_coeffs_der = chebder(Psat_cheb_coeffs)
    Psat_coeffs_critical_der = polyder(Psat_coeffs_critical[::-1])[::-1]
    
    phi_sat_coeffs = [156707085.9178746, 1313005585.0874271, 4947242291.244957, 
                      11038959845.808495, 16153986262.1129, 16199294577.496677, 
                      11273931409.81048, 5376831929.990161, 1681814895.2875218, 
                      311544335.80653775, 25954329.68176187]

    def __init__(self, Tc, Pc, T=None, P=None, V=None, omega=None):
        self.Tc = Tc
        self.Pc = Pc
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1*R*R*Tc**2.5/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = self.b
        self.Vc = self.Zc*R*self.Tc/self.Pc
        self.solve()

    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `a`.
        
        .. math::
            a\alpha = \frac{a}{\sqrt{T}}
        
            \frac{d a\alpha}{dT} = - \frac{a}{2 T^{\frac{3}{2}}}

            \frac{d^2 a\alpha}{dT^2} = \frac{3 a}{4 T^{\frac{5}{2}}}
        '''
        a_alpha = self.a*T**-0.5
        if not full:
            return a_alpha
        else:
            da_alpha_dT = -0.5*self.a*T**(-1.5)
            d2a_alpha_dT2 = 0.75*self.a*T**(-2.5)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def solve_T(self, P, V, quick=True):
        r'''Method to calculate `T` from a specified `P` and `V` for the RK
        EOS. Uses `a`, and `b`, obtained from the class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (3x faster) or 
            individual formulas

        Returns
        -------
        T : float
            Temperature, [K]

        Notes
        -----
        The exact solution can be derived as follows; it is excluded for 
        breviety.
        
        >>> from sympy import *
        >>> P, T, V, R = symbols('P, T, V, R')
        >>> Tc, Pc = symbols('Tc, Pc')
        >>> a, b = symbols('a, b')

        >>> RK = Eq(P, R*T/(V-b) - a/sqrt(T)/(V*V + b*V))
        >>> # solve(RK, T)
        '''
        self.no_T_spec = True
        a, b = self.a, self.b
        if quick:
            x1 = -1.j*1.7320508075688772 + 1.
            x2 = V - b
            x3 = x2/R
            x4 = V + b
            x5 = (1.7320508075688772*(x2*x2*(-4.*P*P*P*x3 + 27.*a*a/(V*V*x4*x4))/(R*R))**0.5 - 9.*a*x3/(V*x4) +0j)**(1./3.)
            return (3.3019272488946263*(11.537996562459266*P*x3/(x1*x5) + 1.2599210498948732*x1*x5)**2/144.0).real
        else:
            return ((-(-1/2 + sqrt(3)*1j/2)*(sqrt(729*(-V*a + a*b)**2/(R*V**2 + R*V*b)**2 + 108*(-P*V + P*b)**3/R**3)/2 + 27*(-V*a + a*b)/(2*(R*V**2 + R*V*b))+0j)**(1/3)/3 + (-P*V + P*b)/(R*(-1/2 + sqrt(3)*1j/2)*(sqrt(729*(-V*a + a*b)**2/(R*V**2 + R*V*b)**2 + 108*(-P*V + P*b)**3/R**3)/2 + 27*(-V*a + a*b)/(2*(R*V**2 + R*V*b))+0j)**(1/3)))**2).real


class SRK(GCEOS):
    r'''Class for solving the Soave-Redlich-Kwong cubic 
    equation of state for a pure compound. Subclasses `GCEOS`, which 
    provides the methods for solving the EOS and calculating its assorted 
    relevant thermodynamic properties. Solves the EOS on initialization. 

    Implemented methods here are `a_alpha_and_derivatives`, which sets 
    a_alpha and its first and second derivatives, and `solve_T`, which from a 
    specified `P` and `V` obtains `T`. 
    
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}
        
        a=\left(\frac{R^2(T_c)^{2}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2}}{P_c}
    
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}
        
        \alpha(T) = \left[1 + m\left(1 - \sqrt{\frac{T}{T_c}}\right)\right]^2
        
        m = 0.480 + 1.574\omega - 0.176\omega^2
    
    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------    
    >>> eos = SRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.00014682102759032003, -31754.65309653571, -74.3732468359525)

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
    c1 = 0.4274802335403414043909906940611707345513 # 1/(9*(2**(1/3.)-1)) 
    c2 = 0.08664034996495772158907020242607611685675 # (2**(1/3.)-1)/3 
    epsilon = 0
    Zc = 1/3.

    Psat_coeffs_limiting = [-3.2308843103522107, 0.7210534170705403]
    
    Psat_coeffs_critical = [9.374273428735918, -6.15924292062784,
                            4.995561268009732, -3.0536215892966374, 
                            1.0000000000023588]

    Psat_cheb_coeffs = [-7.871741490227961, -7.989748461289071, -0.1356344797770207, 0.009506579247579184,
                        0.009624489219138763, -0.007066708482598217, 0.003075503887853841, -0.001012177935988426,
                        0.00028619693856193646, -8.960150789432905e-05, 3.8678642545223406e-05, -1.903594210476056e-05,
                        8.531492278109217e-06, -3.345456890803595e-06, 1.2311165149343946e-06, -4.784033464026011e-07,
                        2.0716513992539553e-07, -9.365210448247373e-08, 4.088078067054522e-08, -1.6950725229317957e-08,
                        6.9147476960875615e-09, -2.9036036947212296e-09, 1.2683728020787197e-09, -5.610046772833513e-10,
                        2.444858416194781e-10, -1.0465240317131946e-10, 4.472305869824417e-11, -1.9380782026977295e-11,
                        8.525075935982007e-12, -3.770209730351304e-12, 1.6512636527230007e-12, -7.22057288092548e-13,
                        3.2921267708457824e-13, -1.616661448808343e-13, 6.227456701354828e-14]
    Psat_cheb_constant_factor = (-2.5857326352412238, 0.38702722494279784)
    Psat_cheb_coeffs_der = chebder(Psat_cheb_coeffs)
    Psat_coeffs_critical_der = polyder(Psat_coeffs_critical[::-1])[::-1]
    
    phi_sat_coeffs = [4.883976406433718e-10, -2.00532968010467e-08, 3.647765457046907e-07,
                      -3.794073186960753e-06, 2.358762477641146e-05, -7.18419726211543e-05,
                      -0.00013493130050539593, 0.002716443506003684, -0.015404883730347763,
                      0.05251643616017714, -0.11346125895127993, 0.12885073074459652,
                      0.0403144920149403, -0.39801902918654086, 0.5962308106352003, 
                      0.6656153310272716]

    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.m = 0.480 + 1.574*omega - 0.176*omega*omega
        self.Vc = self.Zc*R*self.Tc/self.Pc
        self.delta = self.b
        self.solve()

    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `Tc`, `m`, and `a`.
        
        .. math::
            a\alpha = a \left(m \left(- \sqrt{\frac{T}{Tc}} + 1\right)
            + 1\right)^{2}
        
            \frac{d a\alpha}{dT} = \frac{a m}{T} \sqrt{\frac{T}{Tc}} \left(m
            \left(\sqrt{\frac{T}{Tc}} - 1\right) - 1\right)

            \frac{d^2 a\alpha}{dT^2} = \frac{a m \sqrt{\frac{T}{Tc}}}{2 T^{2}}
            \left(m + 1\right)
        '''
        a, Tc, m = self.a, self.Tc, self.m
        sqTr = (T/Tc)**0.5
        a_alpha = a*(m*(1. - sqTr) + 1.)**2
        if not full:
            return a_alpha
        else:
            da_alpha_dT = -a*m*sqTr*(m*(-sqTr + 1.) + 1.)/T
            d2a_alpha_dT2 =  a*m*sqTr*(m + 1.)/(2.*T*T)
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def solve_T(self, P, V, quick=True):
        r'''Method to calculate `T` from a specified `P` and `V` for the SRK
        EOS. Uses `a`, `b`, and `Tc` obtained from the class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (3x faster) or 
            individual formulas

        Returns
        -------
        T : float
            Temperature, [K]

        Notes
        -----
        The exact solution can be derived as follows; it is excluded for 
        breviety.
        
        >>> from sympy import *
        >>> P, T, V, R, a, b, m = symbols('P, T, V, R, a, b, m')
        >>> Tc, Pc, omega = symbols('Tc, Pc, omega')
        >>> a_alpha = a*(1 + m*(1-sqrt(T/Tc)))**2
        >>> SRK = R*T/(V-b) - a_alpha/(V*(V+b)) - P
        >>> # solve(SRK, T)
        '''
        self.no_T_spec = True
        a, b, Tc, m = self.a, self.b, self.Tc, self.m
        if quick:
            x0 = R*Tc
            x1 = V*b
            x2 = x0*x1
            x3 = V*V
            x4 = x0*x3
            x5 = m*m
            x6 = a*x5
            x7 = b*x6
            x8 = V*x6
            x9 = (x2 + x4 + x7 - x8)**2
            x10 = x3*x3
            x11 = R*R*Tc*Tc
            x12 = a*a
            x13 = x5*x5
            x14 = x12*x13
            x15 = b*b
            x16 = x3*V
            x17 = a*x0
            x18 = x17*x5
            x19 = 2.*b*x16
            x20 = -2.*V*b*x14 + 2.*V*x15*x18 + x10*x11 + x11*x15*x3 + x11*x19 + x14*x15 + x14*x3 - 2*x16*x18
            x21 = V - b
            x22 = 2*m*x17
            x23 = P*x4
            x24 = P*x8
            x25 = x1*x17
            x26 = P*R*Tc
            x27 = x17*x3
            x28 = V*x12
            x29 = 2.*m*m*m
            x30 = b*x12
            return -Tc*(2.*a*m*x9*(V*x21*x21*x21*(V + b)*(P*x2 + P*x7 + x17 + x18 + x22 + x23 - x24))**0.5*(m + 1.) - x20*x21*(-P*x16*x6 + x1*x22 + x10*x26 + x13*x28 - x13*x30 + x15*x23 + x15*x24 + x19*x26 + x22*x3 + x25*x5 + x25 + x27*x5 + x27 + x28*x29 + x28*x5 - x29*x30 - x30*x5))/(x20*x9)
        else:
            return Tc*(-2*a*m*sqrt(V*(V - b)**3*(V + b)*(P*R*Tc*V**2 + P*R*Tc*V*b - P*V*a*m**2 + P*a*b*m**2 + R*Tc*a*m**2 + 2*R*Tc*a*m + R*Tc*a))*(m + 1)*(R*Tc*V**2 + R*Tc*V*b - V*a*m**2 + a*b*m**2)**2 + (V - b)*(R**2*Tc**2*V**4 + 2*R**2*Tc**2*V**3*b + R**2*Tc**2*V**2*b**2 - 2*R*Tc*V**3*a*m**2 + 2*R*Tc*V*a*b**2*m**2 + V**2*a**2*m**4 - 2*V*a**2*b*m**4 + a**2*b**2*m**4)*(P*R*Tc*V**4 + 2*P*R*Tc*V**3*b + P*R*Tc*V**2*b**2 - P*V**3*a*m**2 + P*V*a*b**2*m**2 + R*Tc*V**2*a*m**2 + 2*R*Tc*V**2*a*m + R*Tc*V**2*a + R*Tc*V*a*b*m**2 + 2*R*Tc*V*a*b*m + R*Tc*V*a*b + V*a**2*m**4 + 2*V*a**2*m**3 + V*a**2*m**2 - a**2*b*m**4 - 2*a**2*b*m**3 - a**2*b*m**2))/((R*Tc*V**2 + R*Tc*V*b - V*a*m**2 + a*b*m**2)**2*(R**2*Tc**2*V**4 + 2*R**2*Tc**2*V**3*b + R**2*Tc**2*V**2*b**2 - 2*R*Tc*V**3*a*m**2 + 2*R*Tc*V*a*b**2*m**2 + V**2*a**2*m**4 - 2*V*a**2*b*m**4 + a**2*b**2*m**4))


class APISRK(SRK):
    r'''Class for solving the Refinery Soave-Redlich-Kwong cubic 
    equation of state for a pure compound shown in the API Databook [1]_.
    Subclasses `GCEOS`, which 
    provides the methods for solving the EOS and calculating its assorted 
    relevant thermodynamic properties. Solves the EOS on initialization. 

    Implemented methods here are `a_alpha_and_derivatives`, which sets 
    a_alpha and its first and second derivatives, and `solve_T`, which from a 
    specified `P` and `V` obtains `T`. Two fit constants are used in this 
    expresion, with an estimation scheme for the first if unavailable and the
    second may be set to zero.
    
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}
        
        a=\left(\frac{R^2(T_c)^{2}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2}}{P_c}
    
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}
        
        \alpha(T) = \left[1 + S_1\left(1-\sqrt{T_r}\right) + S_2\frac{1
        - \sqrt{T_r}}{\sqrt{T_r}}\right]^2
        
        S_1 = 0.48508 + 1.55171\omega - 0.15613\omega^2 \text{ if S1 is not tabulated }
        
    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float, optional
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]
    S1 : float, optional
        Fit constant or estimated from acentric factor if not provided [-]
    S2 : float, optional
        Fit constant or 0 if not provided [-]

    Examples
    --------    
    >>> eos = APISRK(Tc=514.0, Pc=6137000.0, S1=1.678665, S2=-0.216396, P=1E6, T=299)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 7.045692682173235e-05, -42826.271630638774, -103.62694391379836)

    References
    ----------
    .. [1] API Technical Data Book: General Properties & Characterization.
       American Petroleum Institute, 7E, 2005.
    '''
    def __init__(self, Tc, Pc, omega=None, T=None, P=None, V=None, S1=None, S2=0):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V
        self.kwargs = {'S1': S1, 'S2': S2}
        self.check_sufficient_inputs()

        if S1 is None and omega is None:
            raise Exception('Either acentric factor of S1 is required')

        if S1 is None:
            self.S1 = 0.48508 + 1.55171*omega - 0.15613*omega*omega
        else:
            self.S1 = S1
        self.S2 = S2
        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = self.b
        self.Vc = self.Zc*R*self.Tc/self.Pc
        
        self.solve()

    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `Tc`, `a`, `S1`, and `S2`. 
        
        .. math::
            a\alpha(T) = a\left[1 + S_1\left(1-\sqrt{T_r}\right) + S_2\frac{1
            - \sqrt{T_r}}{\sqrt{T_r}}\right]^2
        
            \frac{d a\alpha}{dT} = a\frac{Tc}{T^{2}} \left(- S_{2} \left(\sqrt{
            \frac{T}{Tc}} - 1\right) + \sqrt{\frac{T}{Tc}} \left(S_{1} \sqrt{
            \frac{T}{Tc}} + S_{2}\right)\right) \left(S_{2} \left(\sqrt{\frac{
            T}{Tc}} - 1\right) + \sqrt{\frac{T}{Tc}} \left(S_{1} \left(\sqrt{
            \frac{T}{Tc}} - 1\right) - 1\right)\right)

            \frac{d^2 a\alpha}{dT^2} = a\frac{1}{2 T^{3}} \left(S_{1}^{2} T
            \sqrt{\frac{T}{Tc}} - S_{1} S_{2} T \sqrt{\frac{T}{Tc}} + 3 S_{1}
            S_{2} Tc \sqrt{\frac{T}{Tc}} + S_{1} T \sqrt{\frac{T}{Tc}} 
            - 3 S_{2}^{2} Tc \sqrt{\frac{T}{Tc}} + 4 S_{2}^{2} Tc + 3 S_{2} 
            Tc \sqrt{\frac{T}{Tc}}\right)
        '''
        a, Tc, S1, S2 = self.a, self.Tc, self.S1, self.S2
        if not full:
            return a*(S1*(-(T/Tc)**0.5 + 1.) + S2*(-(T/Tc)**0.5 + 1)*(T/Tc)**-0.5 + 1)**2
        else:
            if quick:
                x0 = (T/Tc)**0.5
                x1 = x0 - 1.
                x2 = x1/x0
                x3 = S2*x2
                x4 = S1*x1 + x3 - 1.
                x5 = S1*x0
                x6 = S2 - x3 + x5
                x7 = 3.*S2
                a_alpha = a*x4*x4
                da_alpha_dT = a*x4*x6/T
                d2a_alpha_dT2 = a*(-x4*(-x2*x7 + x5 + x7) + x6*x6)/(2.*T*T)
            else:
                a_alpha = a*(S1*(-sqrt(T/Tc) + 1) + S2*(-sqrt(T/Tc) + 1)/sqrt(T/Tc) + 1)**2
                da_alpha_dT = a*((S1*(-sqrt(T/Tc) + 1) + S2*(-sqrt(T/Tc) + 1)/sqrt(T/Tc) + 1)*(-S1*sqrt(T/Tc)/T - S2/T - S2*(-sqrt(T/Tc) + 1)/(T*sqrt(T/Tc))))
                d2a_alpha_dT2 = a*(((S1*sqrt(T/Tc) + S2 - S2*(sqrt(T/Tc) - 1)/sqrt(T/Tc))**2 - (S1*sqrt(T/Tc) + 3*S2 - 3*S2*(sqrt(T/Tc) - 1)/sqrt(T/Tc))*(S1*(sqrt(T/Tc) - 1) + S2*(sqrt(T/Tc) - 1)/sqrt(T/Tc) - 1))/(2*T**2))
            return a_alpha, da_alpha_dT, d2a_alpha_dT2

    def solve_T(self, P, V, quick=True):
        r'''Method to calculate `T` from a specified `P` and `V` for the API 
        SRK EOS. Uses `a`, `b`, and `Tc` obtained from the class's namespace.

        Parameters
        ----------
        P : float
            Pressure, [Pa]
        V : float
            Molar volume, [m^3/mol]
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (3x faster) or 
            individual formulas

        Returns
        -------
        T : float
            Temperature, [K]

        Notes
        -----
        If S2 is set to 0, the solution is the same as in the SRK EOS, and that
        is used. Otherwise, newton's method must be used to solve for `T`. 
        There are 8 roots of T in that case, six of them real. No guarantee can
        be made regarding which root will be obtained.
        '''
        self.no_T_spec = True
        if self.S2 == 0:
            self.m = self.S1
            return SRK.solve_T(self, P, V, quick=quick)
        else:
            # Previously coded method is  63 microseconds vs 47 here
#            return super(SRK, self).solve_T(P, V, quick=quick) 
            Tc, a, b, S1, S2 = self.Tc, self.a, self.b, self.S1, self.S2
            if quick:
                x2 = R/(V-b)
                x3 = (V*(V + b))
                def to_solve(T):
                    x0 = (T/Tc)**0.5
                    x1 = x0 - 1.
                    return (x2*T - a*(S1*x1 + S2*x1/x0 - 1.)**2/x3) - P
            else:
                def to_solve(T):
                    P_calc = R*T/(V - b) - a*(S1*(-sqrt(T/Tc) + 1) + S2*(-sqrt(T/Tc) + 1)/sqrt(T/Tc) + 1)**2/(V*(V + b))
                    return P_calc - P
            return newton(to_solve, Tc*0.5)


class TWUPR(PR):
    r'''Class for solving the Twu [1]_ variant of the Peng-Robinson cubic 
    equation of state for a pure compound. Subclasses `PR`, which 
    provides the methods for solving the EOS and calculating its assorted 
    relevant thermodynamic properties. Solves the EOS on initialization. 

    Implemented methods here are `a_alpha_and_derivatives`, which sets 
    a_alpha and its first and second derivatives, and `solve_T`, which from a 
    specified `P` and `V` obtains `T`. 
    
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

        a=0.45724\frac{R^2T_c^2}{P_c}
        
	  b=0.07780\frac{RT_c}{P_c}
   
       \alpha = \alpha^{(0)} + \omega(\alpha^{(1)}-\alpha^{(0)})
       
       \alpha^{(i)} = T_r^{N(M-1)}\exp[L(1-T_r^{NM})]
      
    For sub-critical conditions:
    
    L0, M0, N0 =  0.125283, 0.911807,  1.948150;
    
    L1, M1, N1 = 0.511614, 0.784054, 2.812520
    
    For supercritical conditions:
    
    L0, M0, N0 = 0.401219, 4.963070, -0.2;
    
    L1, M1, N1 = 0.024955, 1.248089, -8.  
        
    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------
    >>> eos = TWUPR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    >>> eos.V_l, eos.H_dep_l, eos.S_dep_l
    (0.0001301754975832378, -31652.72639160809, -74.11282530917981)
    
    Notes
    -----
    Claimed to be more accurate than the PR, PR78 and PRSV equations.

    There is no analytical solution for `T`. There are multiple possible 
    solutions for `T` under certain conditions; no guaranteed are provided
    regarding which solution is obtained.

    References
    ----------
    .. [1] Twu, Chorng H., John E. Coon, and John R. Cunningham. "A New 
       Generalized Alpha Function for a Cubic Equation of State Part 1. 
       Peng-Robinson Equation." Fluid Phase Equilibria 105, no. 1 (March 15, 
       1995): 49-59. doi:10.1016/0378-3812(94)02601-V.
    '''
    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V
        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = 2.*self.b
        self.epsilon = -self.b*self.b
        self.check_sufficient_inputs()
        self.Vc = self.Zc*R*self.Tc/self.Pc

        self.solve_T = super(PR, self).solve_T        
        self.solve()

    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `Tc`, `omega`, and `a`.
        
        Because of its similarity for the TWUSRK EOS, this has been moved to an 
        external `TWU_a_alpha_common` function. See it for further 
        documentation.
        '''
        return TWU_a_alpha_common(T, self.Tc, self.omega, self.a, full=full, quick=quick, method='PR')


def TWU_a_alpha_common(T, Tc, omega, a, full=True, quick=True, method='PR'):
    r'''Function to calculate `a_alpha` and optionally its first and second
    derivatives for the TWUPR or TWUSRK EOS. Returns 'a_alpha', and 
    optionally 'da_alpha_dT' and 'd2a_alpha_dT2'.
    Used by `TWUPR` and `TWUSRK`; has little purpose on its own.
    See either class for the correct reference, and examples of using the EOS.

    Parameters
    ----------
    T : float
        Temperature, [K]
    Tc : float
        Critical temperature, [K]
    omega : float
        Acentric factor, [-]
    a : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
    full : float
        Whether or not to return its first and second derivatives
    quick : bool, optional
        Whether to use a SymPy cse-derived expression (3x faster) or 
        individual formulas
    method : str
        Either 'PR' or 'SRK'
        
    Notes
    -----
    The derivatives are somewhat long and are not described here for 
    brevity; they are obtainable from the following SymPy expression.
    
    >>> from sympy import *
    >>> T, Tc, omega, N1, N0, M1, M0, L1, L0 = symbols('T, Tc, omega, N1, N0, M1, M0, L1, L0')
    >>> Tr = T/Tc
    >>> alpha0 = Tr**(N0*(M0-1))*exp(L0*(1-Tr**(N0*M0)))
    >>> alpha1 = Tr**(N1*(M1-1))*exp(L1*(1-Tr**(N1*M1)))
    >>> alpha = alpha0 + omega*(alpha1-alpha0)
    >>> # diff(alpha, T)
    >>> # diff(alpha, T, T)
    '''
    Tr = T/Tc
    if method == 'PR':
        if Tr < 1:
            L0, M0, N0 = 0.125283, 0.911807, 1.948150
            L1, M1, N1 = 0.511614, 0.784054, 2.812520
        else:
            L0, M0, N0 = 0.401219, 4.963070, -0.2
            L1, M1, N1 = 0.024955, 1.248089, -8.  
    elif method == 'SRK':
        if Tr < 1:
            L0, M0, N0 = 0.141599, 0.919422, 2.496441
            L1, M1, N1 = 0.500315, 0.799457, 3.291790
        else:
            L0, M0, N0 = 0.441411, 6.500018, -0.20
            L1, M1, N1 = 0.032580,  1.289098, -8.0
    else:
        raise Exception('Only `PR` and `SRK` are accepted as method')
    
    if not full:
        alpha0 = Tr**(N0*(M0-1.))*exp(L0*(1.-Tr**(N0*M0)))
        alpha1 = Tr**(N1*(M1-1.))*exp(L1*(1.-Tr**(N1*M1)))
        alpha = alpha0 + omega*(alpha1 - alpha0)
        return a*alpha
    else:
        if quick:
            x0 = T/Tc
            x1 = M0 - 1
            x2 = N0*x1
            x3 = x0**x2
            x4 = M0*N0
            x5 = x0**x4
            x6 = exp(-L0*(x5 - 1.))
            x7 = x3*x6
            x8 = M1 - 1.
            x9 = N1*x8
            x10 = x0**x9
            x11 = M1*N1
            x12 = x0**x11
            x13 = x2*x7
            x14 = L0*M0*N0*x3*x5*x6
            x15 = x13 - x14
            x16 = exp(-L1*(x12 - 1))
            x17 = -L1*M1*N1*x10*x12*x16 + x10*x16*x9 - x13 + x14
            x18 = N0*N0
            x19 = x18*x3*x6
            x20 = x1**2*x19
            x21 = M0**2
            x22 = L0*x18*x3*x5*x6
            x23 = x21*x22
            x24 = 2*M0*x1*x22
            x25 = L0**2*x0**(2*x4)*x19*x21
            x26 = N1**2
            x27 = x10*x16*x26
            x28 = M1**2
            x29 = L1*x10*x12*x16*x26
            a_alpha = a*(-omega*(-x10*exp(L1*(-x12 + 1)) + x3*exp(L0*(-x5 + 1))) + x7)
            da_alpha_dT = a*(omega*x17 + x15)/T
            d2a_alpha_dT2 = a*(-(omega*(-L1**2*x0**(2.*x11)*x27*x28 + 2.*M1*x29*x8 + x17 + x20 - x23 - x24 + x25 - x27*x8**2 + x28*x29) + x15 - x20 + x23 + x24 - x25)/T**2)
        else:
            a_alpha = TWU_a_alpha_common(T=T, Tc=Tc, omega=omega, a=a, full=False, quick=quick, method=method)
            da_alpha_dT = a*(-L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T + N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T + omega*(L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T - L1*M1*N1*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(L1*(-(T/Tc)**(M1*N1) + 1))/T - N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(L0*(-(T/Tc)**(M0*N0) + 1))/T + N1*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)*exp(L1*(-(T/Tc)**(M1*N1) + 1))/T))
            d2a_alpha_dT2 = a*((L0**2*M0**2*N0**2*(T/Tc)**(2*M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - L0*M0**2*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - 2*L0*M0*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) + L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) + N0**2*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)**2*exp(-L0*((T/Tc)**(M0*N0) - 1)) - N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) - omega*(L0**2*M0**2*N0**2*(T/Tc)**(2*M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - L0*M0**2*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - 2*L0*M0*N0**2*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) + L0*M0*N0*(T/Tc)**(M0*N0)*(T/Tc)**(N0*(M0 - 1))*exp(-L0*((T/Tc)**(M0*N0) - 1)) - L1**2*M1**2*N1**2*(T/Tc)**(2*M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(-L1*((T/Tc)**(M1*N1) - 1)) + L1*M1**2*N1**2*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(-L1*((T/Tc)**(M1*N1) - 1)) + 2*L1*M1*N1**2*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)*exp(-L1*((T/Tc)**(M1*N1) - 1)) - L1*M1*N1*(T/Tc)**(M1*N1)*(T/Tc)**(N1*(M1 - 1))*exp(-L1*((T/Tc)**(M1*N1) - 1)) + N0**2*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)**2*exp(-L0*((T/Tc)**(M0*N0) - 1)) - N0*(T/Tc)**(N0*(M0 - 1))*(M0 - 1)*exp(-L0*((T/Tc)**(M0*N0) - 1)) - N1**2*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)**2*exp(-L1*((T/Tc)**(M1*N1) - 1)) + N1*(T/Tc)**(N1*(M1 - 1))*(M1 - 1)*exp(-L1*((T/Tc)**(M1*N1) - 1))))/T**2)
        return a_alpha, da_alpha_dT, d2a_alpha_dT2


class TWUSRK(SRK):
    r'''Class for solving the Soave-Redlich-Kwong cubic 
    equation of state for a pure compound. Subclasses `GCEOS`, which 
    provides the methods for solving the EOS and calculating its assorted 
    relevant thermodynamic properties. Solves the EOS on initialization. 

    Implemented methods here are `a_alpha_and_derivatives`, which sets 
    a_alpha and its first and second derivatives, and `solve_T`, which from a 
    specified `P` and `V` obtains `T`. 
    
    Two of `T`, `P`, and `V` are needed to solve the EOS.

    .. math::
        P = \frac{RT}{V-b} - \frac{a\alpha(T)}{V(V+b)}
        
        a=\left(\frac{R^2(T_c)^{2}}{9(\sqrt[3]{2}-1)P_c} \right)
        =\frac{0.42748\cdot R^2(T_c)^{2}}{P_c}
    
        b=\left( \frac{(\sqrt[3]{2}-1)}{3}\right)\frac{RT_c}{P_c}
        =\frac{0.08664\cdot R T_c}{P_c}
        
        \alpha = \alpha^{(0)} + \omega(\alpha^{(1)}-\alpha^{(0)})
       
        \alpha^{(i)} = T_r^{N(M-1)}\exp[L(1-T_r^{NM})]
      
    For sub-critical conditions:
    
    L0, M0, N0 =  0.141599, 0.919422, 2.496441
    
    L1, M1, N1 = 0.500315, 0.799457, 3.291790
    
    For supercritical conditions:
    
    L0, M0, N0 = 0.441411, 6.500018, -0.20
    
    L1, M1, N1 = 0.032580,  1.289098, -8.0
    
    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [Pa]
    omega : float
        Acentric factor, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    V : float, optional
        Molar volume, [m^3/mol]

    Examples
    --------    
    >>> eos = TWUSRK(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.00014689217317770398, -31612.591872087483, -74.02294100343829)
    
    Notes
    -----
    There is no analytical solution for `T`. There are multiple possible 
    solutions for `T` under certain conditions; no guaranteed are provided
    regarding which solution is obtained.

    References
    ----------
    .. [1] Twu, Chorng H., John E. Coon, and John R. Cunningham. "A New 
       Generalized Alpha Function for a Cubic Equation of State Part 2. 
       Redlich-Kwong Equation." Fluid Phase Equilibria 105, no. 1 (March 15, 
       1995): 61-69. doi:10.1016/0378-3812(94)02602-W.
    '''
    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = self.b
        self.check_sufficient_inputs()
        self.Vc = self.Zc*R*self.Tc/self.Pc
        self.solve_T = super(SRK, self).solve_T
        self.solve()
        
    def a_alpha_and_derivatives_pure(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `Tc`, `omega`, and `a`.
        
        Because of its similarity for the TWUPR EOS, this has been moved to an 
        external `TWU_a_alpha_common` function. See it for further 
        documentation.
        '''
        return TWU_a_alpha_common(T, self.Tc, self.omega, self.a, full=full, quick=quick, method='SRK')


eos_list = [IG, PR, PR78, PRSV, PRSV2, VDW, RK, SRK, APISRK, TWUPR, TWUSRK]
