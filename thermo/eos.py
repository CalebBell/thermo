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

#__all__ = ['CUBIC_EOS', 'PR']
from cmath import atanh as catanh
from scipy.constants import R
from scipy.optimize import newton
from thermo.utils import Cp_minus_Cv, isothermal_compressibility, phase_identification_parameter, phase_identification_parameter_phase
from thermo.utils import log, exp, sqrt
from thermo.utils import _isobaric_expansion as isobaric_expansion 




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
    def check_sufficient_inputs(self):
        '''Method to an exception if none of the pairs (T, P), (T, V), or 
        (P, V) are given. '''
        if not ((self.T and self.P) or (self.T and self.V) or (self.P and self.V)):
            raise Exception('Either T and P, or T and V, or P and V are required')


    def solve(self):
        '''First EOS-generic method; should be called by all specific EOSs.
        For solving for `T`, the EOS must provide the method `solve_T`.
        For all cases, the EOS must provide `a_alpha_and_derivatives`.
        Calls `set_from_PT` once done.
        '''
        self.check_sufficient_inputs()
        
        if self.V:
            if self.P:
                self.T = self.solve_T(self.P, self.V)
                self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2 = self.a_alpha_and_derivatives(self.T)
            else:
                self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2 = self.a_alpha_and_derivatives(self.T)
                self.P = R*self.T/(self.V-self.b) - self.a_alpha/(self.V*self.V + self.delta*self.V + self.epsilon)
            Vs = [self.V, 1j, 1j]
        else:
            self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2 = self.a_alpha_and_derivatives(self.T)
            Vs = self.volume_solutions(self.T, self.P, self.b, self.delta, self.epsilon, self.a_alpha)
        self.set_from_PT(Vs)

    def set_from_PT(self, Vs):
        '''Counts the number of real volumes in `Vs`, and determins what to do.
        If there is only one real volume, the method 
        `set_properties_from_solution` is called with it. If there are
        two real volumes, `set_properties_from_solution` is called once with  
        each volume. The phase is returned by `set_properties_from_solution`, 
        and the volumes is set to either `V_l` or `V_g` as appropriate. 

        Parameters
        ----------
        Vs : list[float]
            Three possible molar volumes, [m^3/mol]
        '''
        # All roots will have some imaginary component; ignore them if > 1E-9
        imaginary_roots_count = len([True for i in Vs if abs(i.imag) > 1E-9]) 
        if imaginary_roots_count == 2: 
            V = [i for i in Vs if abs(i.imag) < 1E-9][0].real
            self.phase = self.set_properties_from_solution(self.T, self.P, V, self.b, self.delta, self.epsilon, self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2)
            if self.phase == 'l':
                self.V_l = V
            else:
                self.V_g = V
        elif imaginary_roots_count == 0:
            Vs = [i.real for i in Vs]
            self.V_l, self.V_g = min(Vs), max(Vs)
            [self.set_properties_from_solution(self.T, self.P, V, self.b, self.delta, self.epsilon, self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2) for V in [self.V_l, self.V_g]]
            self.phase = 'l/g'
        else:  # pragma: no cover
            raise Exception('No real volumes calculated - look into numerical issues.')

    def set_properties_from_solution(self, T, P, V, b, delta, epsilon, a_alpha, 
                                     da_alpha_dT, d2a_alpha_dT2, quick=True):
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
        ([dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP], 
            [d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2],
            [d2V_dPdT, d2P_dTdV, d2T_dPdV],
            [H_dep, S_dep, Cv_dep]) = self.derivatives_and_departures(T, P, V, b, delta, epsilon, a_alpha, da_alpha_dT, d2a_alpha_dT2, quick=quick)
                
        beta = dV_dT/V # isobaric_expansion(V, dV_dT)
        kappa = -dV_dP/V # isothermal_compressibility(V, dV_dP)
        Cp_m_Cv = -T*dP_dT**2/dP_dV # Cp_minus_Cv(T, dP_dT, dP_dV)
        
        Cp_dep = Cp_m_Cv + Cv_dep - R
                
        V_dep = (V - R*T/P)        
        U_dep = H_dep - P*V_dep
        G_dep = H_dep - T*S_dep
        A_dep = U_dep - T*S_dep
        fugacity = P*exp(G_dep/(R*T))
        phi = fugacity/P
  
        PIP = V*(d2P_dTdV/dP_dT - d2P_dV2/dP_dV) # phase_identification_parameter(V, dP_dT, dP_dV, d2P_dV2, d2P_dTdV)
        phase = 'l' if PIP > 1 else 'g' # phase_identification_parameter_phase(PIP)
      
        if phase == 'l':
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
        else:
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
        return phase            

    def a_alpha_and_derivatives(self, T, full=True, quick=True):
        '''Dummy method to calculate `a_alpha` and its first and second
        derivatives. Should be implemented with the same function signature in 
        each EOS variant; this only raises a NotImplemented Exception.
        Should return 'a_alpha', 'da_alpha_dT', and 'd2a_alpha_dT2'.

        For use in `solve_T`, returns only `a_alpha` if full is False.
        
        Parameters
        ----------
        T : float
            Temperature, [K]
        full : bool, optional
            If False, calculates and returns only `a_alpha`
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (3x faster) or 
            individual formulas
        
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
        raise NotImplemented('a_alpha and its first and second derivatives \
should be calculated by this method, in a user subclass.')
    
    def solve_T(self, P, V, quick=True):
        '''Generic method to calculate `T` from a specified `P` and `V`.
        Provides a SciPy's `newton` solver, and iterates to solve the general
        equation for `P`, resolving for `a_alpha` as a function of temperature
        using `a_alpha_and_derivatives` each time.

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
        def to_solve(T):
            a_alpha = self.a_alpha_and_derivatives(T, full=False)
            P_calc = R*T/(V-self.b) - a_alpha/(V*V + self.delta*V + self.epsilon)
            return P_calc - P
        return newton(to_solve, self.Tc*0.5)

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
        '''
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
            x9 = P**-2
            x10 = P*epsilon
            x11 = delta*x1
            x12 = delta*x2
            x13 = 3.*a_alpha
            x14 = 3.*x10
            x15 = 3.*x11
            x16 = 3.*x12
            x17 = -x1 - x2 + x3
            x18 = x0*x17*x17
            x19 = ((-13.5*x0*(x6 + x7 + x8) - 4.5*x4*x9*(-a_alpha - x10 + x11 + x12) + ((x9*(-4.*x0*(-x13 - x14 + x15 + x16 + x18)**3 + (-9.*x0*x17*(a_alpha + x10 - x11 - x12) + 2.*x17**3*x9 - 27.*x6 - 27.*x7 - 27.*x8)**2))+0j)**0.5*0.5 - x4**3*P**-3)+0j)**(1./3.)
            x20 = x13 + x14 - x15 - x16 - x18
            x22 = 2.*x5
            x23 = 1.7320508075688772j
            x24 = x23 + 1.
            x25 = 4.*x0*x20/x19
            x26 = -x23 + 1.
            return [x0*x20/(x19*3.) - x19/3. + x5/3.,
                    x19*x24/6. + x22/6. - x25/(6.*x24),
                    x19*x26/6. + x22/6. - x25/(6.*x26)]
        else:
            return [-(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)/(3*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)) - (sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)/3 - (-P*b + P*delta - R*T)/(3*P),
                     -(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)/(3*(-1/2 - sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)) - (-1/2 - sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)/3 - (-P*b + P*delta - R*T)/(3*P),
                     -(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)/(3*(-1/2 + sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)) - (-1/2 + sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)/3 - (-P*b + P*delta - R*T)/(3*P)]

    def derivatives_and_departures(self, T, P, V, b, delta, epsilon, a_alpha, da_alpha_dT, d2a_alpha_dT2, quick=True):
        
        dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep = (
        self.main_derivatives_and_departures(T, P, V, b, delta, epsilon, 
                                             a_alpha, da_alpha_dT, 
                                             d2a_alpha_dT2, quick=quick))

        dV_dT = -dP_dT/dP_dV
        dV_dP = -dV_dT/dP_dT 
        dT_dV = 1./dV_dT
        dT_dP = 1./dP_dT
        
        d2V_dP2 = -d2P_dV2*dP_dV**-3
        d2T_dP2 = -d2P_dT2*dP_dT**-3
        
        d2T_dV2 = (-(d2P_dV2*dP_dT - dP_dV*d2P_dTdV)*dP_dT**-2 
                   +(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*dP_dT**-3*dP_dV)
        d2V_dT2 = (-(d2P_dT2*dP_dV - dP_dT*d2P_dTdV)*dP_dV**-2
                   +(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*dP_dV**-3*dP_dT)

        d2V_dPdT = -(d2P_dTdV*dP_dV - dP_dT*d2P_dV2)*dP_dV**-3
        d2T_dPdV = -(d2P_dTdV*dP_dT - dP_dV*d2P_dT2)*dP_dT**-3

        
        return ([dP_dT, dP_dV, dV_dT, dV_dP, dT_dV, dT_dP], 
                [d2P_dT2, d2P_dV2, d2V_dT2, d2V_dP2, d2T_dV2, d2T_dP2],
                [d2V_dPdT, d2P_dTdV, d2T_dPdV],
                [H_dep, S_dep, Cv_dep])

    @staticmethod
    def main_derivatives_and_departures(T, P, V, b, delta, epsilon, a_alpha,
                                        da_alpha_dT, d2a_alpha_dT2, quick=True):
        if quick:
            x0 = V - b
            x1 = V*V + V*delta + epsilon
            x3 = R*T
            x4 = 1./(x0*x0)
            x5 = 2*V + delta
            x6 = 1./(x1*x1)
            x7 = a_alpha*x6
            x8 = P*V
            x9 = delta*delta
            x10 = -4*epsilon + x9
            x11 = x10**-0.5
            x12 = 2.*x11*catanh(x11*x5).real
            x13 = x10**-0.5 
            x14 = V + delta*0.5
            x15 = 2.*epsilon*x13
            x16 = x13*x9*0.5
            dP_dT = R/x0 - da_alpha_dT/x1
            dP_dV = -x3*x4 + x5*x7
            d2P_dT2 = -d2a_alpha_dT2/x1
            d2P_dV2 = -2.*a_alpha*x5*x5*x1**-3 + 2.*x7 + 2.*x3*x0**-3
            d2P_dTdV = -R*x4 + da_alpha_dT*x5*x6
            H_dep = x12*(T*da_alpha_dT - a_alpha) - x3 + x8
            S_dep = -R*log(V*x3/(x0*x8)) + da_alpha_dT*x12 
            Cv_dep = -T*d2a_alpha_dT2*x13*(-log((x14 - x15 + x16)/(x14 + x15 - x16)))
        else:
            dP_dT = R/(V - b) - da_alpha_dT/(V**2 + V*delta + epsilon)
            dP_dV = -R*T/(V - b)**2 - (-2*V - delta)*a_alpha/(V**2 + V*delta + epsilon)**2
            d2P_dT2 = -d2a_alpha_dT2/(V**2 + V*delta + epsilon)
            d2P_dV2 = 2*(R*T/(V - b)**3 - (2*V + delta)**2*a_alpha/(V**2 + V*delta + epsilon)**3 + a_alpha/(V**2 + V*delta + epsilon)**2)
            d2P_dTdV = -R/(V - b)**2 + (2*V + delta)*da_alpha_dT/(V**2 + V*delta + epsilon)**2
            H_dep = P*V - R*T + 2*(T*da_alpha_dT - a_alpha)*catanh((2*V + delta)/sqrt(delta**2 - 4*epsilon)).real/sqrt(delta**2 - 4*epsilon)
            S_dep = -R*log(V) + R*log(P*V/(R*T)) + R*log(V - b) + 2*da_alpha_dT*catanh((2*V + delta)/sqrt(delta**2 - 4*epsilon)).real/sqrt(delta**2 - 4*epsilon)
            Cv_dep = -T*(sqrt(1/(delta**2 - 4*epsilon))*log(V - delta**2*sqrt(1/(delta**2 - 4*epsilon))/2 + delta/2 + 2*epsilon*sqrt(1/(delta**2 - 4*epsilon))) - sqrt(1/(delta**2 - 4*epsilon))*log(V + delta**2*sqrt(1/(delta**2 - 4*epsilon))/2 + delta/2 - 2*epsilon*sqrt(1/(delta**2 - 4*epsilon))))*d2a_alpha_dT2
        return [dP_dT, dP_dV, d2P_dT2, d2P_dV2, d2P_dTdV, H_dep, S_dep, Cv_dep]


class PR(GCEOS):
    r'''Class for solving a the Peng-Robinson cubic 
    equation of state for a pure compound. Subclasses `CUBIC_EOS`, which 
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
    
    >>> eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=400., P=1E6)
    >>> eos.V_l, eos.V_g
    (0.0001560731318852931, 0.0021418760907613724)
    >>> eos.phase
    'l/g'
    >>> eos.H_dep_l, eos.H_dep_g
    (-26111.86872116082, -3549.2993749373945)
    >>> eos.S_dep_l, eos.S_dep_g
    (-58.09842815106086, -6.439449710478305)
    >>> eos.U_dep_l, eos.U_dep_g
    (-22942.157933046114, -2365.391545698767)
    >>> eos.G_dep_l, eos.G_dep_g
    (-2872.4974607364747, -973.5194907460736)
    >>> eos.A_dep_l, eos.A_dep_g
    (297.21332737823104, 210.38833849255388)
    >>> eos.beta_l, eos.beta_g
    (0.0026933709177838043, 0.010123223911174959)
    >>> eos.kappa_l, eos.kappa_g
    (9.335721543829601e-09, 1.9710669809793286e-06)
    >>> eos.Cp_minus_Cv_l, eos.Cp_minus_Cv_g
    (48.51014580740871, 44.54414603000341)
    >>> eos.Cv_dep_l, eos.Cp_dep_l
    (25.165377505266747, 44.50559908690951)

    P-T initialization, liquid phase, and round robin trip:
    
    >>> eos = PR(Tc=507.6, Pc=3025000, omega=0.2975, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.00013022208100139964, -31134.740290463385, -72.47559475426007)
    
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

    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.kappa = 0.37464 + 1.54226*omega - 0.26992*omega*omega
        self.delta = 2*self.b
        self.epsilon = -self.b*self.b
        
        
        self.solve()

    def a_alpha_and_derivatives(self, T, full=True, quick=True):
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
            return self.a*(1 + self.kappa*(1-(T/self.Tc)**0.5))**2
        else:
            if quick:
                Tc, kappa = self.Tc, self.kappa
                x0 = T**0.5
                x1 = Tc**-0.5
                x2 = kappa*(x0*x1 - 1.) - 1.
                x3 = self.a*kappa
                
                a_alpha = self.a*x2*x2
                da_alpha_dT = x1*x2*x3/x0
                d2a_alpha_dT2 = x3*(-0.5*T**-1.5*x1*x2 + 0.5/(T*Tc)*kappa)
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
    r'''Class for solving a the Peng-Robinson cubic 
    equation of state for a pure compound according to the 1978 variant.
    Subclasses `PR`, which provides everything except the variable `kappa`.
    Solves the EOS on initialization. See `PR` for further documentation.
    
    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

        a=0.45724\frac{R^2T_c^2}{P_c}
        
	  b=0.07780\frac{RT_c}{P_c}

        \alpha(T)=[1+\kappa(1-\sqrt{T_r})]^2
        
        m_i = 0.37464+1.54226\omega-0.26992\omega^2 \text{ if } \omega_i
        \le 0.491
        
        m_i = 0.379642 + 1.48503 \omega_i - 0.164423\omega_i^2 + 0.016666
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
    ('l', 8.351960066075052e-05, -63764.64948050847, -130.737108912626)
    
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
    ('l', 0.0001301268694484059, -31698.916002476708, -74.1674902435042)
    
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
        
        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = 2*self.b
        self.epsilon = -self.b*self.b
        self.kappa0 = 0.378893 + 1.4897153*omega - 0.17131848*omega**2 + 0.0196554*omega**3

        self.check_sufficient_inputs()
        if self.V and self.P:
            # Deal with T-solution here; does NOT support kappa1_Tr_limit.
            self.kappa1 = kappa1
            self.T = self.solve_T(self.P, self.V)
            Tr = self.T/Tc
        else:
            Tr = self.T/Tc
            if self.kappa1_Tr_limit and Tc > 0.7:
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

    def a_alpha_and_derivatives(self, T, full=True, quick=True):
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
    ('l', 0.00013018821346475254, -31496.173493225753, -73.6152580115141)
    
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
        
        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = 2*self.b
        self.epsilon = -self.b*self.b
        self.kappa0 = 0.378893 + 1.4897153*omega - 0.17131848*omega*omega + 0.0196554*omega*omega*omega
        self.kappa1, self.kappa2, self.kappa3 = kappa1, kappa2, kappa3
        
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


    def a_alpha_and_derivatives(self, T, full=True, quick=True):
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
    equation of state for a pure compound. Subclasses `CUBIC_EOS`, which 
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

    Examples
    --------    
    >>> eos = VDW(Tc=507.6, Pc=3025000, T=299., P=1E6)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 0.00022332978038490077, -13385.722837649315, -32.65922018109096)

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th 
       edition. New York: McGraw-Hill Professional, 2000.
    .. [2] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
       Butterworth-Heinemann, 1985.
    '''
    def __init__(self, Tc, Pc, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.T = T
        self.P = P
        self.V = V

        self.a = 27.0/64.0*(R*Tc)**2/Pc
        self.b = R*Tc/(8.*Pc)
        self.delta = 0
        self.epsilon = 0
        self.solve()

    def a_alpha_and_derivatives(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `a`.
        
        .. math::
            a\alpha = a
        
            \frac{d a\alpha}{dT} = 0

            \frac{d^2 a\alpha}{dT^2} = 0
        '''
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
    equation of state for a pure compound. Subclasses `CUBIC_EOS`, which 
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
    ('l', 0.00015189341729751865, -26160.833620674082, -63.01311649400543)

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
    
    def __init__(self, Tc, Pc, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1*R*R*Tc**2.5/Pc
        self.b = self.c2*R*Tc/Pc
        self.delta = self.b
        self.solve()

    def a_alpha_and_derivatives(self, T, full=True, quick=True):
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
    equation of state for a pure compound. Subclasses `CUBIC_EOS`, which 
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
    ('l', 0.0001473238480377508, -30917.940322270817, -72.44137873264924)

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
   
    def __init__(self, Tc, Pc, omega, T=None, P=None, V=None):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V

        self.a = self.c1*R*R*Tc*Tc/Pc
        self.b = self.c2*R*Tc/Pc
        self.m = 0.480 + 1.574*omega - 0.176*omega
        self.delta = self.b
        self.solve()

    def a_alpha_and_derivatives(self, T, full=True, quick=True):
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
    Subclasses `CUBIC_EOS`, which 
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
    S1 : float, optional
        Fit constant or 0 if not provided [-]

    Examples
    --------    
    >>> eos = APISRK(Tc=514.0, Pc=6137000.0, S1=1.678665, S2=-0.216396, P=1E6, T=299)
    >>> eos.phase, eos.V_l, eos.H_dep_l, eos.S_dep_l
    ('l', 7.045692682173252e-05, -42826.2716306387, -103.6269439137981)

    References
    ----------
    .. [1] API Technical Data Book: General Properties & Characterization.
       American Petroleum Institute, 7E, 2005.
    '''
    epsilon = 0
    
    def __init__(self, Tc, Pc, omega=None, T=None, P=None, V=None, S1=None, S2=0):
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.T = T
        self.P = P
        self.V = V
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
        
        self.solve()

    def a_alpha_and_derivatives(self, T, full=True, quick=True):
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
            return a*(S1*(-sqrt(T/Tc) + 1) + S2*(-sqrt(T/Tc) + 1)/sqrt(T/Tc) + 1)**2
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
    r'''Class for solving a the Peng-Robinson cubic 
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
    (0.0001301754975832377, -25137.048959073203, -52.32126198966514)
    
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
        self.delta = 2*self.b
        self.epsilon = -self.b*self.b
        self.check_sufficient_inputs()

        self.solve_T = super(PR, self).solve_T        
        self.solve()

    def a_alpha_and_derivatives(self, T, full=True, quick=True):
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
    >>> N1, N0, M1, M0, L1, L0 = symbols('N1, N0, M1, M0, L1, L0')
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
    equation of state for a pure compound. Subclasses `CUBIC_EOS`, which 
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
        
        self.solve_T = super(SRK, self).solve_T
        self.solve()
        
    def a_alpha_and_derivatives(self, T, full=True, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for this EOS. Returns `a_alpha`, `da_alpha_dT`, and 
        `d2a_alpha_dT2`. See `GCEOS.a_alpha_and_derivatives` for more 
        documentation. Uses the set values of `Tc`, `omega`, and `a`.
        
        Because of its similarity for the TWUPR EOS, this has been moved to an 
        external `TWU_a_alpha_common` function. See it for further 
        documentation.
        '''
        return TWU_a_alpha_common(T, self.Tc, self.omega, self.a, full=full, quick=quick, method='SRK')



