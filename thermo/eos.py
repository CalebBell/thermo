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
from thermo.utils import Cp_minus_Cv, isothermal_compressibility, phase_identification_parameter, phase_identification_parameter_phase
from math import log, exp, sqrt
from thermo.utils import _isobaric_expansion as isobaric_expansion 


class CUBIC_EOS(object):
    r'''Class for solving a generic Pressure-explicit three-parameter cubic 
    equation of state. Does not implement any parameters itself; must be 
    subclassed by an equation of state class which uses it. Works for mixtures
    or pure species for all properties except fugacity. All properties are 
    derived with the CAS SymPy, not relying on any derivations previously 
    published.

    .. math::
        P = \frac{RT}{v-b}-\frac{a\alpha(T)}{v(v+b)+b(v-b)}

    Main methods (in order they are called) are `solve`, `set_from_PT`,
    `volume_solutions`, `set_properties_from_solution`,  and
    `derivatives_and_departures`. 

    `solve` checks if two of `T`, `P`, and `V` were set. It solves for the 
    remaining variable. If `T` is missing, method `solve_T` is used; it is
    parameter specific, and so must be implemented in each specific EOS.
    If `P` is missing, it is directly calculated. If `V` is missing, it
    is calculated with the method `volume_solutions`. At this point, either
    three possible volumes or one user specified volume are known. The
    value of `a_alpha`, and its first and second temperature derivative are
    calculated with the EOS-specific method `set_a_alpha_and_derivatives`. 

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
    the `quick` flag, the method `derivatives_and_departures` is used; it is
    a mess derived with SymPy's `cse` function to perform the calculation
    as quickly as possible. Otherwise, three seperate methods 
    `first_derivatives`, `second_derivatives`, and `second_derivatives_mixed`
    are used to perform the calculation with independent solutions derived
    from SymPy. 

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
    
    def solve(self):
        '''First EOS-generic method; should be called by all specific EOSs.
        For solving for `T`, the EOS must provide the method `solve_T`.
        For all cases, the EOS must provide `set_a_alpha_and_derivatives`.
        Calls `set_from_PT` once done.
        '''
        if not ((self.T and self.P) or (self.T and self.V) or (self.P and self.V)):
            raise Exception('Either T and P, or T and V, or P and V are required')
        
        if self.V:
            if self.P:
                self.T = self.solve_T(self.P, self.V)
                self.set_a_alpha_and_derivatives(self.T)
            else:
                self.set_a_alpha_and_derivatives(self.T)
                self.P = R*self.T/(self.V-self.b) - self.a_alpha/(self.V*(self.V+self.b)+self.b*(self.V-self.b))
            Vs = [self.V, 1j, 1j]
        else:
            self.set_a_alpha_and_derivatives(self.T)
            Vs = self.volume_solutions(self.T, self.P, self.b, self.a_alpha)
        
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
            self.phase = self.set_properties_from_solution(self.T, self.P, V, self.b, self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2)
            if self.phase == 'l':
                self.V_l = V
            else:
                self.V_g = V
        elif imaginary_roots_count == 0:
            Vs = [i.real for i in Vs]
            self.V_l, self.V_g = min(Vs), max(Vs)
            [self.set_properties_from_solution(self.T, self.P, V, self.b, self.a_alpha, self.da_alpha_dT, self.d2a_alpha_dT2) for V in [self.V_l, self.V_g]]
            self.phase = 'l/g'
        else:  # pragma: no cover
            raise Exception('No real volumes calculated - look into numerical issues.')

    def set_properties_from_solution(self, T, P, V, b, a_alpha, da_alpha_dT, 
                                     d2a_alpha_dT2, quick=True):
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
            - \frac{\frac{d {a \alpha}{\left (T \right )}}{d T} }{V \left(V 
            + b\right) + b \left(V - b\right)}
            
            \left(\frac{\partial P}{\partial V}\right)_T = - \frac{R T}{\left(
            V - b\right)^{2}} - \frac{\left(- 2 V - 2 b\right) \operatorname{a
            \alpha}{\left (T \right )}}{\left(V \left(V + b\right) 
            + b \left(V - b\right)\right)^{2}}
            
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
            \left(\frac{\partial^2 P}{\partial T^2}\right)_V =- \frac{\frac{
            d^{2}}{d T^{2}}  \operatorname{a \alpha}{\left (T \right )}}{V 
            \left(V + b\right) + b \left(V - b\right)}
            
            \left(\frac{\partial^2 P}{\partial V^2}\right)_T =\frac{2 R T}{
            \left(V - b\right)^{3}} - \frac{\left(- 4 V - 4 b\right) \left(- 
            2 V - 2 b\right) \operatorname{a \alpha}{\left (T \right )}}{\left(
            V \left(V + b\right) + b \left(V - b\right)\right)^{3}} + \frac{2 
            \operatorname{a \alpha}{\left (T \right )}}{\left(V \left(V + 
            b\right) + b \left(V - b\right)\right)^{2}}
            
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
           \left(\frac{\partial^2 P}{\partial T \partial V}\right) = - \frac{R}
           {\left(V - b\right)^{2}} - \frac{\left(- 2 V - 2 b\right) \frac{d}
           {d T} \operatorname{a \alpha}{\left (T \right )}}{\left(V \left(V +
           b\right) + b \left(V - b\right)\right)^{2}}
           
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
            - P\right]dV + PV - RT= \frac{1}{2}\,{\frac {\sqrt {2} \left(\left(
            {\frac {\rm d}{{\rm d}T}}{\it a\alpha} \left( T \right)  \right) 
            T-{\it a\alpha} \left( T \right)  \right) }{b}
            {\rm arctanh} \left(1/2\,{\frac { \left( V+b \right) \sqrt {2}}{b}}
            \right)} + PV - RT

            S_{dep} = \int_{\infty}^V\left[\frac{\partial P}{\partial T} 
            - \frac{R}{V}\right] dV + R\log\frac{PV}{RT} = 0.5\,{\frac {1}{b} 
            \left(  \left( {\frac 
            {\rm d}{{\rm d}T}}{\it a\alpha}\left( T \right)  \right) \sqrt {2}
            {\rm arctanh} \left(0.5\,{\frac {\left( V+b \right) \sqrt {2}}{b}}
            \right)-2\,Rb \left( \ln  \left( V\right) -\ln  \left( V-b \right)
            \right)  \right) }+ R\log\frac{PV}{RT}
        
            V_{dep} = V - \frac{RT}{P}
            
            U_{dep} = H_{dep} - P V_{dep}
            
            G_{dep} = H_{dep} - T S_{dep}
            
            A_{dep} = U_{dep} - T S_{dep}
            
            \text{fugacity} = P\exp(\frac{G_{dep}}{RT})
            
            \phi = \frac{\text{fugacity}}{P}
            
            C_{v, dep} = T\int_\infty^V \left(\frac{\partial^2 P}{\partial 
            T^2}\right) dV = - \frac{T}{b} \left(- \frac{\sqrt{2}}{4} \log{
            \left (V + b + \sqrt{2} b \right )} + \frac{\sqrt{2}}{4} \log{\left
            (V - \sqrt{2} b + b \right )}\right) \frac{d^{2} \operatorname{a 
            \alpha}{\left (T \right )}}{d T^{2}}  
            
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
            [H_dep, S_dep]) = self.derivatives_and_departures(T, P, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2, quick=quick)
                
        beta = isobaric_expansion(V, dV_dT)
        kappa = isothermal_compressibility(V, dV_dP)
        Cp_m_Cv = Cp_minus_Cv(T, dP_dT, dP_dV)
        
        Cv_dep = -T*(-sqrt(2)*log(V + b + sqrt(2)*b)/4 + sqrt(2)*log(V - sqrt(2)*b + b)/4)*d2a_alpha_dT2/b
        Cp_dep = Cp_m_Cv + Cv_dep - R
                
        V_dep = (V - R*T/P)        
        U_dep = H_dep - P*V_dep
        G_dep = H_dep - T*S_dep
        A_dep = U_dep - T*S_dep
        fugacity = P*exp(G_dep/(R*T))
        phi = fugacity/P
  
        PIP = phase_identification_parameter(V, dP_dT, dP_dV, d2P_dV2, d2P_dTdV)
        phase = phase_identification_parameter_phase(PIP)
      
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

    def set_a_alpha_and_derivatives(self, T, quick=True):
        '''Dummy method to calculate `a_alpha` and its first and second
        derivatives. Should be implemented with the same function signature in 
        each EOS variant; this only raises a NotImplemented Exception.
        Should set 'a_alpha', 'da_alpha_dT', and 'd2a_alpha_dT2'.

        Parameters
        ----------
        T : float
            Temperature, [K]
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (3x faster) or 
            individual formulas
        '''
        raise NotImplemented('a_alpha and its first and second derivatives \
should be calculated by this method, in a user subclass.')
    
    def solve_T(self, P, V, quick=True):
        '''Dummy method to calculate `T` from a specified `P` and `V`.
        Should be implemented with the same function signature in 
        each EOS variant; this only raises a NotImplemented Exception.
        This will use at least `Tc` and `Pc` as well, obtained from the class's
        namespace.

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
        '''
        raise NotImplemented('A method to solve the EOS for T should be \
calculated by this method, in a user subclass.')
    
    @staticmethod
    def volume_solutions(T, P, b, a_alpha, quick=True):
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
        The very-ugly solutions are as follows. Note that using the expressions
        directly is still faster than most numeric root finding techniques, and
        finds all values explicitly.
        
        .. math::
            V_1 = - \frac{- \frac{1}{P Tc} \left(- 9 P Tc b^{2} - 6 R T Tc b 
            - 6 \sqrt{T} \sqrt{Tc} a \kappa^{2} - 6 \sqrt{T} \sqrt{Tc} a \kappa
            + 3 T a \kappa^{2} + 3 Tc a \kappa^{2} + 6 Tc a \kappa + 3 Tc a
            \right) + \frac{1}{P^{2}} \left(P b - R T\right)^{2}}{3 \sqrt[3]{
            \frac{1}{2} \sqrt{- 4 \left(- \frac{1}{P Tc} \left(- 9 P Tc b^{2} 
            - 6 R T Tc b - 6 \sqrt{T} \sqrt{Tc} a \kappa^{2} - 6 \sqrt{T} 
            \sqrt{Tc} a \kappa + 3 T a \kappa^{2} + 3 Tc a \kappa^{2} + 6 Tc a 
            \kappa + 3 Tc a\right) + \frac{1}{P^{2}} \left(P b - R T\right)^{2}
            \right)^{3} + \left(\frac{1}{P Tc} \left(27 P Tc b^{3} + 27 R T Tc 
            b^{2} + 54 \sqrt{T} \sqrt{Tc} a b \kappa^{2} + 54 \sqrt{T} \sqrt{
            Tc} a b \kappa - 27 T a b \kappa^{2} - 27 Tc a b \kappa^{2} - 54 Tc
            a b \kappa - 27 Tc a b\right) - \frac{9}{P^{2} Tc} \left(P b - R T
            \right) \left(- 3 P Tc b^{2} - 2 R T Tc b - 2 \sqrt{T} \sqrt{Tc} a
            \kappa^{2} - 2 \sqrt{T} \sqrt{Tc} a \kappa + T a \kappa^{2} + Tc a 
            \kappa^{2} + 2 Tc a \kappa + Tc a\right) + \frac{2}{P^{3}} \left(P
            b - R T\right)^{3}\right)^{2}} + \frac{1}{2 P Tc} \left(27 P Tc 
            b^{3} + 27 R T Tc b^{2} + 54 \sqrt{T} \sqrt{Tc} a b \kappa^{2} + 54
            \sqrt{T} \sqrt{Tc} a b \kappa - 27 T a b \kappa^{2} - 27 Tc a b 
            \kappa^{2} - 54 Tc a b \kappa - 27 Tc a b\right) - \frac{9}{2 P^{2}
            Tc} \left(P b - R T\right) \left(- 3 P Tc b^{2} - 2 R T Tc b - 2 
            \sqrt{T} \sqrt{Tc} a \kappa^{2} - 2 \sqrt{T} \sqrt{Tc} a \kappa + T
            a \kappa^{2} + Tc a \kappa^{2} + 2 Tc a \kappa + Tc a\right) + 
            \frac{1}{P^{3}} \left(P b - R T\right)^{3}}} - \frac{1}{3} \sqrt[3]
            {\frac{1}{2} \sqrt{- 4 \left(- \frac{1}{P Tc} \left(- 9 P Tc b^{2} 
            - 6 R T Tc b - 6 \sqrt{T} \sqrt{Tc} a \kappa^{2} - 6 \sqrt{T} 
            \sqrt{Tc} a \kappa + 3 T a \kappa^{2} + 3 Tc a \kappa^{2} + 6 Tc a 
            \kappa + 3 Tc a\right) + \frac{1}{P^{2}} \left(P b - R T\right)^{2}
            \right)^{3} + \left(\frac{1}{P Tc} \left(27 P Tc b^{3} + 27 R T Tc 
            b^{2} + 54 \sqrt{T} \sqrt{Tc} a b \kappa^{2} + 54 \sqrt{T} 
            \sqrt{Tc} a b \kappa - 27 T a b \kappa^{2} - 27 Tc a b \kappa^{2} 
            - 54 Tc a b \kappa - 27 Tc a b\right) - \frac{9}{P^{2} Tc} \left(P
            b - R T\right) \left(- 3 P Tc b^{2} - 2 R T Tc b - 2 \sqrt{T} 
            \sqrt{Tc} a \kappa^{2} - 2 \sqrt{T} \sqrt{Tc} a \kappa + T a 
            \kappa^{2} + Tc a \kappa^{2} + 2 Tc a \kappa + Tc a\right) 
            + \frac{2}{P^{3}} \left(P b - R T\right)^{3}\right)^{2}} + \frac{1}
            {2 P Tc} \left(27 P Tc b^{3} + 27 R T Tc b^{2} + 54 \sqrt{T} 
            \sqrt{Tc} a b \kappa^{2} + 54 \sqrt{T} \sqrt{Tc} a b \kappa - 27 T 
            a b \kappa^{2} - 27 Tc a b \kappa^{2} - 54 Tc a b \kappa - 27 Tc a 
            b\right) - \frac{9}{2 P^{2} Tc} \left(P b - R T\right) \left(- 3 P
            Tc b^{2} - 2 R T Tc b - 2 \sqrt{T} \sqrt{Tc} a \kappa^{2} - 2 
            \sqrt{T} \sqrt{Tc} a \kappa + T a \kappa^{2} + Tc a \kappa^{2} + 2 
            Tc a \kappa + Tc a\right) + \frac{1}{P^{3}} \left(P b - R T\right
            )^{3}} - \frac{1}{3 P} \left(P b - R T\right)
            
            V_2 = - \frac{- \frac{1}{P Tc} \left(- 9 P Tc b^{2} - 6 R T Tc b 
            - 6 \sqrt{T} \sqrt{Tc} a \kappa^{2} - 6 \sqrt{T} \sqrt{Tc} a \kappa
            + 3 T a \kappa^{2} + 3 Tc a \kappa^{2} + 6 Tc a \kappa + 3 Tc a
            \right) + \frac{1}{P^{2}} \left(P b - R T\right)^{2}}{3 \left(- 
            \frac{1}{2} - \frac{\sqrt{3} i}{2}\right) \sqrt[3]{\frac{1}{2} 
            \sqrt{- 4 \left(- \frac{1}{P Tc} \left(- 9 P Tc b^{2} - 6 R T Tc b
            - 6 \sqrt{T} \sqrt{Tc} a \kappa^{2} - 6 \sqrt{T} \sqrt{Tc} a \kappa
            + 3 T a \kappa^{2} + 3 Tc a \kappa^{2} + 6 Tc a \kappa + 3 Tc a
            \right) + \frac{1}{P^{2}} \left(P b - R T\right)^{2}\right)^{3} 
            + \left(\frac{1}{P Tc} \left(27 P Tc b^{3} + 27 R T Tc b^{2} + 54 
            \sqrt{T} \sqrt{Tc} a b \kappa^{2} + 54 \sqrt{T} \sqrt{Tc} a b 
            \kappa - 27 T a b \kappa^{2} - 27 Tc a b \kappa^{2} - 54 Tc a b 
            \kappa - 27 Tc a b\right) - \frac{9}{P^{2} Tc} \left(P b - R T
            \right) \left(- 3 P Tc b^{2} - 2 R T Tc b - 2 \sqrt{T} \sqrt{Tc} a 
            \kappa^{2} - 2 \sqrt{T} \sqrt{Tc} a \kappa + T a \kappa^{2} + Tc a
            \kappa^{2} + 2 Tc a \kappa + Tc a\right) + \frac{2}{P^{3}} \left(P 
            b - R T\right)^{3}\right)^{2}} + \frac{1}{2 P Tc} \left(27 P Tc 
            b^{3} + 27 R T Tc b^{2} + 54 \sqrt{T} \sqrt{Tc} a b \kappa^{2} + 54
            \sqrt{T} \sqrt{Tc} a b \kappa - 27 T a b \kappa^{2} - 27 Tc a b 
            \kappa^{2} - 54 Tc a b \kappa - 27 Tc a b\right) - \frac{9}{2 P^{2} 
            Tc} \left(P b - R T\right) \left(- 3 P Tc b^{2} - 2 R T Tc b - 2 
            \sqrt{T} \sqrt{Tc} a \kappa^{2} - 2 \sqrt{T} \sqrt{Tc} a \kappa 
            + T a \kappa^{2} + Tc a \kappa^{2} + 2 Tc a \kappa + Tc a\right) 
            + \frac{1}{P^{3}} \left(P b - R T\right)^{3}}} - \frac{1}{3} \left(
            - \frac{1}{2} - \frac{\sqrt{3} i}{2}\right) \sqrt[3]{\frac{1}{2} 
            \sqrt{- 4 \left(- \frac{1}{P Tc} \left(- 9 P Tc b^{2} - 6 R T Tc b 
            - 6 \sqrt{T} \sqrt{Tc} a \kappa^{2} - 6 \sqrt{T} \sqrt{Tc} a \kappa
            + 3 T a \kappa^{2} + 3 Tc a \kappa^{2} + 6 Tc a \kappa + 3 Tc a
            \right) + \frac{1}{P^{2}} \left(P b - R T\right)^{2}\right)^{3} 
            + \left(\frac{1}{P Tc} \left(27 P Tc b^{3} + 27 R T Tc b^{2} + 54 
            \sqrt{T} \sqrt{Tc} a b \kappa^{2} + 54 \sqrt{T} \sqrt{Tc} a b 
            \kappa - 27 T a b \kappa^{2} - 27 Tc a b \kappa^{2} - 54 Tc a b 
            \kappa - 27 Tc a b\right) - \frac{9}{P^{2} Tc} \left(P b - R T
            \right) \left(- 3 P Tc b^{2} - 2 R T Tc b - 2 \sqrt{T} \sqrt{Tc} a 
            \kappa^{2} - 2 \sqrt{T} \sqrt{Tc} a \kappa + T a \kappa^{2} + Tc a
            \kappa^{2} + 2 Tc a \kappa + Tc a\right) + \frac{2}{P^{3}} \left(P
            b - R T\right)^{3}\right)^{2}} + \frac{1}{2 P Tc} \left(27 P Tc 
            b^{3} + 27 R T Tc b^{2} + 54 \sqrt{T} \sqrt{Tc} a b \kappa^{2} + 54
            \sqrt{T} \sqrt{Tc} a b \kappa - 27 T a b \kappa^{2} - 27 Tc a b 
            \kappa^{2} - 54 Tc a b \kappa - 27 Tc a b\right) - \frac{9}{2 P^{2}
            Tc} \left(P b - R T\right) \left(- 3 P Tc b^{2} - 2 R T Tc b - 2 
            \sqrt{T} \sqrt{Tc} a \kappa^{2} - 2 \sqrt{T} \sqrt{Tc} a \kappa + T
            a \kappa^{2} + Tc a \kappa^{2} + 2 Tc a \kappa + Tc a\right) 
            + \frac{1}{P^{3}} \left(P b - R T\right)^{3}} - \frac{1}{3 P} 
            \left(P b - R T\right)
            
            V_3 = - \frac{- \frac{1}{P Tc} \left(- 9 P Tc b^{2} - 6 R T Tc b 
            - 6 \sqrt{T} \sqrt{Tc} a \kappa^{2} - 6 \sqrt{T} \sqrt{Tc} a \kappa
            + 3 T a \kappa^{2} + 3 Tc a \kappa^{2} + 6 Tc a \kappa + 3 Tc a
            \right) + \frac{1}{P^{2}} \left(P b - R T\right)^{2}}{3 \left(- 
            \frac{1}{2} + \frac{\sqrt{3} i}{2}\right) \sqrt[3]{\frac{1}{2} 
            \sqrt{- 4 \left(- \frac{1}{P Tc} \left(- 9 P Tc b^{2} - 6 R T Tc b
            - 6 \sqrt{T} \sqrt{Tc} a \kappa^{2} - 6 \sqrt{T} \sqrt{Tc} a \kappa
            + 3 T a \kappa^{2} + 3 Tc a \kappa^{2} + 6 Tc a \kappa + 3 Tc a
            \right) + \frac{1}{P^{2}} \left(P b - R T\right)^{2}\right)^{3} 
            + \left(\frac{1}{P Tc} \left(27 P Tc b^{3} + 27 R T Tc b^{2} + 54 
            \sqrt{T} \sqrt{Tc} a b \kappa^{2} + 54 \sqrt{T} \sqrt{Tc} a b 
            \kappa - 27 T a b \kappa^{2} - 27 Tc a b \kappa^{2} - 54 Tc a b 
            \kappa - 27 Tc a b\right) - \frac{9}{P^{2} Tc} \left(P b - R T
            \right) \left(- 3 P Tc b^{2} - 2 R T Tc b - 2 \sqrt{T} \sqrt{Tc} a 
            \kappa^{2} - 2 \sqrt{T} \sqrt{Tc} a \kappa + T a \kappa^{2} + Tc a
            \kappa^{2} + 2 Tc a \kappa + Tc a\right) + \frac{2}{P^{3}} \left(P 
            b - R T\right)^{3}\right)^{2}} + \frac{1}{2 P Tc} \left(27 P Tc 
            b^{3} + 27 R T Tc b^{2} + 54 \sqrt{T} \sqrt{Tc} a b \kappa^{2} + 54
            \sqrt{T} \sqrt{Tc} a b \kappa - 27 T a b \kappa^{2} - 27 Tc a b 
            \kappa^{2} - 54 Tc a b \kappa - 27 Tc a b\right) - \frac{9}{2 P^{2} 
            Tc} \left(P b - R T\right) \left(- 3 P Tc b^{2} - 2 R T Tc b - 2 
            \sqrt{T} \sqrt{Tc} a \kappa^{2} - 2 \sqrt{T} \sqrt{Tc} a \kappa 
            + T a \kappa^{2} + Tc a \kappa^{2} + 2 Tc a \kappa + Tc a\right) 
            + \frac{1}{P^{3}} \left(P b - R T\right)^{3}}} - \frac{1}{3} \left(
            - \frac{1}{2} + \frac{\sqrt{3} i}{2}\right) \sqrt[3]{\frac{1}{2} 
            \sqrt{- 4 \left(- \frac{1}{P Tc} \left(- 9 P Tc b^{2} - 6 R T Tc b 
            - 6 \sqrt{T} \sqrt{Tc} a \kappa^{2} - 6 \sqrt{T} \sqrt{Tc} a \kappa
            + 3 T a \kappa^{2} + 3 Tc a \kappa^{2} + 6 Tc a \kappa + 3 Tc a
            \right) + \frac{1}{P^{2}} \left(P b - R T\right)^{2}\right)^{3} 
            + \left(\frac{1}{P Tc} \left(27 P Tc b^{3} + 27 R T Tc b^{2} + 54 
            \sqrt{T} \sqrt{Tc} a b \kappa^{2} + 54 \sqrt{T} \sqrt{Tc} a b 
            \kappa - 27 T a b \kappa^{2} - 27 Tc a b \kappa^{2} - 54 Tc a b 
            \kappa - 27 Tc a b\right) - \frac{9}{P^{2} Tc} \left(P b - R T
            \right) \left(- 3 P Tc b^{2} - 2 R T Tc b - 2 \sqrt{T} \sqrt{Tc} a 
            \kappa^{2} - 2 \sqrt{T} \sqrt{Tc} a \kappa + T a \kappa^{2} + Tc a
            \kappa^{2} + 2 Tc a \kappa + Tc a\right) + \frac{2}{P^{3}} \left(P
            b - R T\right)^{3}\right)^{2}} + \frac{1}{2 P Tc} \left(27 P Tc 
            b^{3} + 27 R T Tc b^{2} + 54 \sqrt{T} \sqrt{Tc} a b \kappa^{2} + 54
            \sqrt{T} \sqrt{Tc} a b \kappa - 27 T a b \kappa^{2} - 27 Tc a b 
            \kappa^{2} - 54 Tc a b \kappa - 27 Tc a b\right) - \frac{9}{2 P^{2}
            Tc} \left(P b - R T\right) \left(- 3 P Tc b^{2} - 2 R T Tc b - 2 
            \sqrt{T} \sqrt{Tc} a \kappa^{2} - 2 \sqrt{T} \sqrt{Tc} a \kappa + T
            a \kappa^{2} + Tc a \kappa^{2} + 2 Tc a \kappa + Tc a\right) 
            + \frac{1}{P^{3}} \left(P b - R T\right)^{3}} - \frac{1}{3 P} 
            \left(P b - R T\right)
        '''
        T, P = complex(T), complex(P)
        if quick:
            x0 = 1./P
            x1 = R*T
            x2 = P*b - x1
            x3 = x0*x2
            x4 = x2*x2*x2
            x5 = b*x1
            x6 = a_alpha
            x7 = -x6
            x8 = b*b
            x9 = P*x8
            x10 = b*(x5 + x7 + x9)
            x11 = 1./(P*P)
            x12 = 3.*x6
            x13 = 6.*x5
            x14 = 9.*x9
            x15 = x0*x2*x2
            x16 = (13.5*x0*x10 + 4.5*x11*x2*(2.*x5 + x7 + 3.*x9) + (x11*(-4.*x0*(-x12 + x13 + x14 + x15)**3 + (27.*x10 + 2.*x11*x4 - x3*(-27.*P*x8 - 18.*R*T*b + 9.*x6))**2))**0.5/2. + x4/P**3)**(1./3.)
            x17 = x12 - x13 - x14 - x15
            x18 = 1./x16
            x19 = -2.*x3
            x20 = 1.7320508075688772j
            x21 = x20 + 1.
            x22 = 4.*x0*x17*x18
            x23 = -x20 + 1.
            return [x0*x17*x18/3. - x16/3. - x3/3.,
                    x16*x21/6. + x19/6. - x22/(6.*x21),
                    x16*x23/6. + x19/6. - x22/(6.*x23)]
        else:
            return [-(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)/(3*((-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)**0.5/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)) - ((-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)**0.5/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)/3 - (P*b - R*T)/(3*P),
                    -(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)/(3*(-1/2 - sqrt(3)*1j/2)*((-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)**0.5/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)) - (-1/2 - sqrt(3)*1j/2)*((-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)**0.5/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)/3 - (P*b - R*T)/(3*P),
                    -(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)/(3*(-1/2 + sqrt(3)*1j/2)*((-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)**0.5/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)) - (-1/2 + sqrt(3)*1j/2)*((-4*(-3*(-3*P*b**2 - 2*R*T*b + a_alpha)/P + (P*b - R*T)**2/P**2)**3 + (27*(P*b**3 + R*T*b**2 - b*a_alpha)/P - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/P**2 + 2*(P*b - R*T)**3/P**3)**2)**0.5/2 + 27*(P*b**3 + R*T*b**2 - b*a_alpha)/(2*P) - 9*(P*b - R*T)*(-3*P*b**2 - 2*R*T*b + a_alpha)/(2*P**2) + (P*b - R*T)**3/P**3)**(1/3)/3 - (P*b - R*T)/(3*P)]
    
    @staticmethod
    def derivatives_and_departures(T, P, V, b, a_alpha, da_alpha_dT, d2a_alpha_dT2, quick=True):
        if quick:
            x0 = V - b
            x1 = V + b
            x2 = V*x1 + b*x0
            x3 = 1./x2
            x4 = R/x0 - da_alpha_dT*x3
            x5 = R*T
            x6 = 1./(x0*x0)
            x7 = x5*x6
            x8 = 2.*x1
            x9 = 1./(x2*x2)
            x10 = a_alpha*x9
            x11 = x10*x8
            x12 = d2a_alpha_dT2*x3
            x13 = 1./(-x11 + x7)
            x14 = R*x6
            x15 = da_alpha_dT*x8*x9
            x16 = P*V
            x17 = 1.414213562373095048801688724209698078570 # sqrt(2)
            x18 = 1./b
            x19 = x17*x18/2.
            x20 = catanh(x1*x19).real
            
            dP_dT = x4
            dP_dV = x11 - x7
            d2P_dT2 = -x12
            d2P_dV2 = -8.*a_alpha*x1*x1/(x2*x2*x2) + 2.*x10 + 2.*x5/(x0*x0*x0)
            d2P_dTdV = -x14 + x15
            H_dep = x16 + x19*x20*(T*da_alpha_dT - a_alpha) - x5
            S_dep = R*log(x16/(R*T)) - x18*(2*R*b*(log(V) - log(x0)) - da_alpha_dT*x17*x20)/2.        
        else:
            dP_dT = R/(V - b) - da_alpha_dT/(V*(V + b) + b*(V - b))
            dP_dV = -R*T/(V - b)**2 - (-2*V - 2*b)*a_alpha/(V*(V + b) + b*(V - b))**2
            d2P_dT2 = -d2a_alpha_dT2/(V*(V + b) + b*(V - b))
            d2P_dV2 = 2*R*T/(V - b)**3 - (-4*V - 4*b)*(-2*V - 2*b)*a_alpha/(V*(V + b) + b*(V - b))**3 + 2*a_alpha/(V*(V + b) + b*(V - b))**2
            d2P_dTdV = -R/(V - b)**2 - (-2*V - 2*b)*da_alpha_dT/(V*(V + b) + b*(V - b))**2
            H_dep = P*V - R*T + sqrt(2)*catanh((V + b)*sqrt(2)/b/2).real * (da_alpha_dT*T-a_alpha)/b/2
            S_dep = R*log(P*V/(R*T)) + (da_alpha_dT*sqrt(2)*catanh((V + b)*sqrt(2)/b/2).real - 2*R*b*(log(V) - log(V - b)))/b/2


        dV_dT = -dP_dT/dP_dV
        dV_dP = -dV_dT/dP_dT # or same as dP_dV
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
                [H_dep, S_dep])



        
class PR(CUBIC_EOS):
    r'''Class for solving a the Peng-Robinson cubic 
    equation of state for a pure compound. Subclasses `CUBIC_EOS`, which 
    provides the methods for solving the EOS and calculating its assorted 
    relevant thermodynamic properties. Solves the EOS on initialization. 

    Implemented methods here are `set_a_alpha_and_derivatives`, which sets 
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
    (-58.09842815106086, -6.439449710478302)
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
        
        self.solve()

    def set_a_alpha_and_derivatives(self, T, quick=True):
        r'''Method to calculate `a_alpha` and its first and second
        derivatives for the PR EOS.  Sets 'a_alpha', 'da_alpha_dT', and 
        'd2a_alpha_dT2'.

        .. math::
            a\alpha = a \left(\kappa \left(- \frac{T^{0.5}}{Tc^{0.5}} 
            + 1\right) + 1\right)^{2}
        
            \frac{d a\alpha}{dT} = - \frac{1.0 a \kappa}{T^{0.5} Tc^{0.5}}
            \left(\kappa \left(- \frac{T^{0.5}}{Tc^{0.5}} + 1\right) + 1\right)

            \frac{d^2 a\alpha}{dT^2} = 0.5 a \kappa \left(- \frac{1}{T^{1.5} 
            Tc^{0.5}} \left(\kappa \left(\frac{T^{0.5}}{Tc^{0.5}} - 1\right)
            - 1\right) + \frac{\kappa}{T^{1.0} Tc^{1.0}}\right)

        Uses the set values of `Tc`, `kappa, and `a`.

        Parameters
        ----------
        T : float
            Temperature, [K]
        quick : bool, optional
            Whether to use a SymPy cse-derived expression (3x faster) or 
            individual formulas
        '''
        if quick:
            x0 = T**0.5
            x1 = self.Tc**(-0.5)
            x2 = self.kappa*(x0*x1 - 1.) - 1.
            x3 = self.a*self.kappa
            
            self.a_alpha = self.a*x2*x2
            self.da_alpha_dT = x1*x2*x3/x0
            self.d2a_alpha_dT2 = x3*(-0.5*T**(-1.5)*x1*x2 + 0.5/T/self.Tc*self.kappa)
        else:
            self.a_alpha = self.a*(1 + self.kappa*(1-(T/self.Tc)**0.5))**2
            self.da_alpha_dT = -self.a*self.kappa*sqrt(T/self.Tc)*(self.kappa*(-sqrt(T/self.Tc) + 1.) + 1.)/T
            self.d2a_alpha_dT2 = self.a*self.kappa*(self.kappa/self.Tc - sqrt(T/self.Tc)*(self.kappa*(sqrt(T/self.Tc) - 1.) - 1.)/T)/(2.*T)

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
        The exact solution is:
        
        .. math::
            T = \frac{Tc}{\left(R Tc V^{2} + 2 R Tc V b - R Tc b^{2} - V a 
            \kappa^{2} + a b \kappa^{2}\right)^{2} \left(R^{2} Tc^{2} V^{4} 
            + 4 R^{2} Tc^{2} V^{3} b + 2 R^{2} Tc^{2} V^{2} b^{2} - 4 R^{2} 
            Tc^{2} V b^{3} + R^{2} Tc^{2} b^{4} - 2 R Tc V^{3} a \kappa^{2} 
            - 2 R Tc V^{2} a b \kappa^{2} + 6 R Tc V a b^{2} \kappa^{2} - 2 R 
            Tc a b^{3} \kappa^{2} + V^{2} a^{2} \kappa^{4} - 2 V a^{2} b 
            \kappa^{4} + a^{2} b^{2} \kappa^{4}\right)} \left(- 2 a \kappa 
            \sqrt{\left(V - b\right)^{3} \left(V^{2} + 2 V b - b^{2}\right) 
            \left(P R Tc V^{2} + 2 P R Tc V b - P R Tc b^{2} - P V a \kappa^{2}
            + P a b \kappa^{2} + R Tc a \kappa^{2} + 2 R Tc a \kappa + R Tc 
            a\right)} \left(\kappa + 1\right) \left(R Tc V^{2} + 2 R Tc V b 
            - R Tc b^{2} - V a \kappa^{2} + a b \kappa^{2}\right)^{2} 
            + \left(V - b\right) \left(R^{2} Tc^{2} V^{4} + 4 R^{2} Tc^{2} 
            V^{3} b + 2 R^{2} Tc^{2} V^{2} b^{2} - 4 R^{2} Tc^{2} V b^{3} 
            + R^{2} Tc^{2} b^{4} - 2 R Tc V^{3} a \kappa^{2} - 2 R Tc V^{2} 
            a b \kappa^{2} + 6 R Tc V a b^{2} \kappa^{2} - 2 R Tc a b^{3} 
            \kappa^{2} + V^{2} a^{2} \kappa^{4} - 2 V a^{2} b \kappa^{4} 
            + a^{2} b^{2} \kappa^{4}\right) \left(P R Tc V^{4} + 4 P R Tc 
            V^{3} b + 2 P R Tc V^{2} b^{2} - 4 P R Tc V b^{3} + P R Tc b^{4} 
            - P V^{3} a \kappa^{2} - P V^{2} a b \kappa^{2} + 3 P V a b^{2} 
            \kappa^{2} - P a b^{3} \kappa^{2} + R Tc V^{2} a \kappa^{2} + 2 R 
            Tc V^{2} a \kappa + R Tc V^{2} a + 2 R Tc V a b \kappa^{2} + 4 R 
            Tc V a b \kappa + 2 R Tc V a b - R Tc a b^{2} \kappa^{2} - 2 R Tc 
            a b^{2} \kappa - R Tc a b^{2} + V a^{2} \kappa^{4} + 2 V a^{2} 
            \kappa^{3} + V a^{2} \kappa^{2} - a^{2} b \kappa^{4} - 2 a^{2} b 
            \kappa^{3} - a^{2} b \kappa^{2}\right)\right)
        
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
