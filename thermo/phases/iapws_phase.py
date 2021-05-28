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

__all__ = ['IAPWS95', 'IAPWS95Gas', 'IAPWS95Liquid', 'IAPWS97']

from chemicals import iapws
from chemicals.viscosity import mu_IAPWS
from chemicals.thermal_conductivity import k_IAPWS
from .helmholtz_eos import HelmholtzEOS
from chemicals.utils import rho_to_Vm, Vm_to_rho
from .phase import Phase

class IAPWS95(HelmholtzEOS):
    model_name = 'iapws95'
    _MW = iapws.iapws95_MW
    Tc = iapws.iapws95_Tc
    Pc = iapws.iapws95_Pc
    rhoc_mass = iapws.iapws95_rhoc
    rhoc_mass_inv = 1.0/rhoc_mass

    rhoc_inv = rho_to_Vm(rhoc_mass, _MW)
    rhoc = 1.0/rhoc_inv

    rho_red = rhoc
    rho_red_inv = rhoc_inv

    T_red = Tc
    T_fixed_transport = 1.5*T_red

    _MW_kg = _MW*1e-3
    R = _MW_kg*iapws.iapws95_R # This is just the gas constant 8.314... but matching iapws to their decimals
    R_inv = 1.0/R
    R2 = R*R

    #R = property_mass_to_molar(iapws95_R, iapws95_MW)
    zs = [1.0]
    cmps = [0]
#    HeatCapacityGases = iapws_correlations.HeatCapacityGases

    T_MAX_FIXED = 5000.0
    T_MIN_FIXED = 243.0 # PU has flash failures at < 242 ish K

    _d4Ar_ddelta2dtau2_func = staticmethod(iapws.iapws95_d4Ar_ddelta2dtau2)
    _d3Ar_ddeltadtau2_func = staticmethod(iapws.iapws95_d3Ar_ddeltadtau2)
    _d3Ar_ddelta2dtau_func = staticmethod(iapws.iapws95_d3Ar_ddelta2dtau)
    _d2Ar_ddeltadtau_func = staticmethod(iapws.iapws95_d2Ar_ddeltadtau)
    _d2Ar_dtau2_func = staticmethod(iapws.iapws95_d2Ar_dtau2)
    _dAr_dtau_func = staticmethod(iapws.iapws95_dAr_dtau)
    _d3Ar_ddelta3_func = staticmethod(iapws.iapws95_d3Ar_ddelta3)
    _d2Ar_ddelta2_func = staticmethod(iapws.iapws95_d2Ar_ddelta2)
    _dAr_ddelta_func = staticmethod(iapws.iapws95_dAr_ddelta)
    _Ar_func = staticmethod(iapws.iapws95_Ar)


    def __init__(self, T=None, P=None, zs=None):
        self.T = T
        self.P = P
        self._rho_mass = rho_mass = iapws.iapws95_rho(T, P)
        self._V = rho_to_Vm(rho=rho_mass, MW=self._MW)
        self.tau = tau = self.Tc/T
        self.delta = delta = rho_mass*self.rhoc_mass_inv
        self.A0, self.dA0_dtau, self.d2A0_dtau2, self.d3A0_dtau3 = iapws.iapws95_A0_tau_derivatives(tau, delta)

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.T = T
        new.P = P
        new._rho_mass = rho_mass = iapws.iapws95_rho(T, P)
        new._V = rho_to_Vm(rho=rho_mass, MW=self._MW)
        new.tau = tau = new.Tc/T
        new.delta = delta = rho_mass*new.rhoc_mass_inv
        new.A0, new.dA0_dtau, new.d2A0_dtau2, new.d3A0_dtau3 = iapws.iapws95_A0_tau_derivatives(tau, delta)
        return new

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        if T is not None and P is not None:
            new.T = T
            new._rho_mass = rho_mass = iapws.iapws95_rho(T, P)
            new._V = rho_to_Vm(rho=rho_mass, MW=self._MW)
            new.P = P
        elif T is not None and V is not None:
            new.T = T
            new._rho_mass = rho_mass = 1e-3*self._MW/V
            P = iapws.iapws95_P(T, rho_mass)
            new._V = V
            new.P = P
        elif P is not None and V is not None:
            new._rho_mass = rho_mass = Vm_to_rho(V, MW=self._MW)
            T = new.T = iapws.iapws95_T(P, rho_mass)
            new._V = V
            new.P = P
        else:
            raise ValueError("Two of T, P, or V are needed")

        new.P = P
        new.T = T
        new.tau = tau = new.Tc/T
        new.delta = delta = rho_mass*new.rhoc_mass_inv
        new.A0, new.dA0_dtau, new.d2A0_dtau2, new.d3A0_dtau3 = iapws.iapws95_A0_tau_derivatives(tau, delta)

        return new


    def mu(self):
        r'''Calculate and return the viscosity of water according to the IAPWS.
        For details, see :obj:`chemicals.viscosity.mu_IAPWS`.

        Returns
        -------
        mu : float
            Viscosity of water, [Pa*s]
        '''
        try:
            return self._mu
        except:
            pass
        self.__mu_k()
        return self._mu

    def k(self):
        r'''Calculate and return the thermal conductivity of water according to the IAPWS.
        For details, see :obj:`chemicals.thermal_conductivity.k_IAPWS`.

        Returns
        -------
        k : float
            Thermal conductivity of water, [W/m/K]
        '''
        try:
            return self._k
        except:
            pass
        self.__mu_k()
        return self._k

    def __mu_k(self):
        drho_mass_dP = self.drho_mass_dP()

        # TODO: curve fit drho_dP_Tr better than IAPWS did (mpmath)
        drho_dP_Tr = self.to(T=self.T_fixed_transport, V=self._V, zs=self.zs).drho_mass_dP()
        self._mu = mu_IAPWS(T=self.T, rho=self._rho_mass, drho_dP=drho_mass_dP,
                        drho_dP_Tr=drho_dP_Tr)

        self._k = k_IAPWS(T=self.T, rho=self._rho_mass, Cp=self.Cp_mass(), Cv=self.Cv_mass(),
                       mu=self._mu, drho_dP=drho_mass_dP, drho_dP_Tr=drho_dP_Tr)



class IAPWS95Gas(IAPWS95):
    is_gas = True
    is_liquid = False
    force_phase = 'g'

class IAPWS95Liquid(IAPWS95):
    force_phase = 'l'
    is_gas = False
    is_liquid = True

class IAPWS97(Phase):
    model_name = 'iapws97'
    model_attributes = ('model_name',)
    _MW = 18.015268
    R = 461.526
    Tc = 647.096
    Pc = 22.064E6
    rhoc = 322.
    zs = [1.0]
    cmps = [0]
    def mu(self):
        return mu_IAPWS(T=self.T, rho=self._rho_mass)

    def k(self):
        # TODO add properties; even industrial formulation recommends them
        return k_IAPWS(T=self.T, rho=self._rho_mass)

    ### Region 1,2,5 Gibbs
    def G(self):
        try:
            return self._G
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            G = iapws.iapws97_G_region1(tau, pi)
        elif region == 2:
            G = iapws.iapws97_Gr_region2(tau, pi) + iapws.iapws97_G0_region2(tau, pi)
        elif region == 5:
            G = iapws.iapws97_Gr_region5(tau, pi) + iapws.iapws97_G0_region5(tau, pi)
        elif region == 4:
            G = self.H() - self.T*self.S()
        self._G = G
        return G


    def dG_dpi(self):
        try:
            return self._dG_dpi
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            dG_dpi = iapws.iapws97_dG_dpi_region1(tau, pi)
        elif region == 2:
            dG_dpi = 1.0/pi + iapws.iapws97_dGr_dpi_region2(tau, pi)
        elif region == 5:
            dG_dpi = 1.0/pi + iapws.iapws97_dGr_dpi_region5(tau, pi)
        self._dG_dpi = dG_dpi
        return dG_dpi

    def d2G_d2pi(self):
        try:
            return self._d2G_d2pi
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            d2G_d2pi = iapws.iapws97_d2G_dpi2_region1(tau, pi)
        elif region == 2:
            d2G_d2pi = -1.0/(pi*pi) + iapws.iapws97_d2Gr_dpi2_region2(tau, pi)
        elif region == 5:
            d2G_d2pi = -1.0/(pi*pi) + iapws.iapws97_d2Gr_dpi2_region5(tau, pi)
        self._d2G_d2pi = d2G_d2pi
        return d2G_d2pi

    def dG_dtau(self):
        try:
            return self._dG_dtau
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            dG_dtau = iapws.iapws97_dG_dtau_region1(tau, pi)
        elif region == 2:
            dG_dtau = iapws.iapws97_dG0_dtau_region2(tau, pi) + iapws.iapws97_dGr_dtau_region2(tau, pi)
        elif region == 5:
            dG_dtau = iapws.iapws97_dG0_dtau_region5(tau, pi) + iapws.iapws97_dGr_dtau_region5(tau, pi)
        self._dG_dtau = dG_dtau
        return dG_dtau

    def d2G_d2tau(self):
        try:
            return self._d2G_d2tau
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            d2G_d2tau = iapws.iapws97_d2G_dtau2_region1(tau, pi)
        elif region == 2:
            d2G_d2tau = (iapws.iapws97_d2Gr_dtau2_region2(tau, pi)
                         + iapws.iapws97_d2G0_dtau2_region2(tau, pi))
        elif region == 5:
            d2G_d2tau = (iapws.iapws97_d2Gr_dtau2_region5(tau, pi)
                         + iapws.iapws97_d2G0_dtau2_region5(tau, pi))
        self._d2G_d2tau = d2G_d2tau
        return d2G_d2tau

    def d2G_dpidtau(self):
        try:
            return self._d2G_dpidtau
        except:
            pass
        tau, pi, region = self.tau, self.pi, self.region
        if region == 1:
            d2G_dpidtau = iapws.iapws97_d2G_dpidtau_region1(tau, pi)
        elif region == 2:
            d2G_dpidtau = iapws.iapws97_d2Gr_dpidtau_region2(tau, pi)
        elif region == 5:
            d2G_dpidtau = iapws.iapws97_d2Gr_dpidtau_region5(tau, pi)
        self._d2G_dpidtau = d2G_dpidtau
        return d2G_dpidtau


    ### Region 3 Helmholtz
    def A_region3(self):
        try:
            return self._A_region3
        except:
            pass
        self._A_region3 = A_region3 = iapws.iapws97_A_region3_region3(self.tau, self.delta)
        return A_region3

    def dA_ddelta(self):
        try:
            return self._dA_ddelta
        except:
            pass
        self._dA_ddelta = dA_ddelta = iapws.iapws97_dA_ddelta_region3(self.tau, self.delta)
        return dA_ddelta

    def d2A_d2delta(self):
        try:
            return self._d2A_d2delta
        except:
            pass
        self._d2A_d2delta = d2A_d2delta = iapws.iapws97_d2A_d2delta_region3(self.tau, self.delta)
        return d2A_d2delta

    def dA_dtau(self):
        try:
            return self._dA_dtau
        except:
            pass
        self._dA_dtau = dA_dtau = iapws.iapws97_dA_dtau_region3(self.tau, self.delta)
        return dA_dtau

    def d2A_d2tau(self):
        try:
            return self._d2A_d2tau
        except:
            pass
        self._d2A_d2tau = d2A_d2tau = iapws.iapws97_d2A_d2tau_region3(self.tau, self.delta)
        return d2A_d2tau

    def d2A_ddeltadtau(self):
        try:
            return self._d2A_ddeltadtau
        except:
            pass
        self._d2A_ddeltadtau = d2A_ddeltadtau = iapws.iapws97_d2A_ddeltadtau_region3(self.tau, self.delta)
        return d2A_ddeltadtau

    def __init__(self, T=None, P=None, zs=None):
        self.T = T
        self.P = P
        self._rho_mass = iapws.iapws97_rho(T, P)
        self._V = rho_to_Vm(rho=self._rho_mass, MW=self._MW)
        self.region = region = iapws.iapws97_identify_region_TP(T, P)
        if region == 1:
            self.pi = P*6.049606775559589e-08 #1/16.53E6
            self.tau = 1386.0/T
            self.Pref = 16.53E6
            self.Tref = 1386.0
        elif region == 2:
            self.pi = P*1e-6
            self.tau = 540.0/T
            self.Pref = 1e6
            self.Tref = 540.0
        elif region == 3:
            self.tau = self.Tc/T
            self.Tref = self.Tc
            self.delta = self._rho_mass*0.003105590062111801 # 1/322.0
            self.rhoref = 322.0
        elif region == 5:
            self.pi = P*1e-6
            self.tau = 1000.0/T
            self.Tref = 1000.0
            self.Pref = 1e6



    def to_TP_zs(self, T, P, zs, other_eos=None):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        self._rho_mass = iapws.iapws97_rho(T, P)
        self._V = rho_to_Vm(rho=self._rho_mass, MW=self._MW)
        self.region = region = iapws.iapws97_identify_region_TP(T, P)
        if region == 1:
            self.pi = P*6.049606775559589e-08 #1/16.53E6
            self.tau = 1386.0/T
        elif region == 2:
            self.pi = P*1e-6
            self.tau = 540.0/T
        elif region == 3:
            self.tau = self.Tc/T
            self.delta = self._rho_mass*0.003105590062111801 # 1/322.0
        elif region == 5:
            self.pi = P*1e-6
            self.tau = 1000.0/T

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs

        if T is not None:
            new.T = T
            if P is not None:
                new._rho_mass = rho_mass = iapws.iapws97_rho(T, P)
                new._V = rho_to_Vm(rho=rho_mass, MW=self._MW)
                new.P = P
            elif V is not None:
                new._rho_mass = rho_mass = Vm_to_rho(V, MW=self._MW)
                P = iapws.iapws97_P(T, rho_mass)
                new.V = V
                new.P = P
        elif P is not None and V is not None:
            new._rho_mass = rho_mass = Vm_to_rho(V, MW=self._MW)
            T = new.T = iapws.iapws97_T(P, rho_mass)
            new.V = V
            new.P = P
        else:
            raise ValueError("Two of T, P, or V are needed")

        new.region = region = iapws.iapws97_identify_region_TP(new.T, new.P)
        if region == 1:
            new.pi = P*6.049606775559589e-08 #1/16.53E6
            new.tau = 1386.0/T
            new.Pref = 16.53E6
            new.Tref = 1386.0
        elif region == 2:
            new.pi = P*1e-6
            new.tau = 540.0/T
            new.Pref = 1e6
            new.Tref = 540.0
        elif region == 3:
            new.tau = new.Tc/T
            new.Tref = new.Tc
            new.delta = new._rho_mass*0.003105590062111801 # 1/322.0
            new.rhoref = 322.0
        elif region == 5:
            new.pi = P*1e-6
            new.tau = 1000.0/T
            new.Tref = 1000.0
            new.Pref = 1e6

        new.P = P
        new.T = T

        return new

    def V(self):
        return self._V

    def U(self):
        try:
            return self._U
        except:
            pass

        if self.region != 3:
            try:
                dG_dtau = self._dG_dtau
            except:
                dG_dtau = self.dG_dtau()
            try:
                dG_dpi = self._dG_dpi
            except:
                dG_dpi = self.dG_dpi()
            U = self.R*self.T(*self.tau*dG_dtau - self.pi*dG_dpi)
        self._U = U
        return U

    def S(self):
        try:
            return self._S
        except:
            pass
        if self.region != 3:
            try:
                G = self._G
            except:
                G = self.G()
            try:
                dG_dtau = self._dG_dtau
            except:
                dG_dtau = self.dG_dtau()
            S = self.R*(self.tau*dG_dtau - G)
        self._S = S
        return S

    def H(self):
        try:
            return self._H
        except:
            pass
        if self.region != 3:
            try:
                dG_dtau = self._dG_dtau
            except:
                dG_dtau = self.dG_dtau()
            H = self.R*self.T*self.tau*dG_dtau
        self._H = H
        return H

    def Cv(self):
        try:
            return self._Cv
        except:
            pass
        if self.region != 3:
            try:
                d2G_d2tau = self._d2G_d2tau
            except:
                d2G_d2tau = self.d2G_d2tau()
            try:
                dG_dpi = self._dG_dpi
            except:
                dG_dpi = self.dG_dpi()
            try:
                d2G_dpidtau = self._d2G_dpidtau
            except:
                d2G_dpidtau = self.d2G_dpidtau()
            try:
                d2G_d2pi = self._d2G_d2pi
            except:
                d2G_d2pi = self.d2G_d2pi()

            tau = self.tau
            x0 = (dG_dpi - tau*d2G_dpidtau)
            Cv = self.R*(-tau*tau*d2G_d2tau + x0*x0/d2G_d2pi)


    def Cp(self):
        try:
            return self._Cp
        except:
            pass

        if self.region == 3:
            tau, delta = self.tau, self.delta # attributes set on init
            try:
                dA_ddelta = self._dA_ddelta
            except:
                dA_ddelta = self.dA_ddelta()
            try:
                d2A_ddeltadtau = self._d2A_ddeltadtau
            except:
                d2A_ddeltadtau = self.d2A_ddeltadtau()
            try:
                d2A_d2delta = self._d2A_d2delta
            except:
                d2A_d2delta = self.d2A_d2delta()
            try:
                d2A_d2tau = self._d2A_d2tau
            except:
                d2A_d2tau = self.d2A_d2tau()

            x0 = (delta*dA_ddelta - delta*tau*d2A_ddeltadtau)
            Cp = self.R*(-tau*tau*d2A_d2tau + x0*x0/(delta*(2.0*dA_ddelta + delta*d2A_d2delta)))

#        self.Cp = (-self.tau**2*self.ddA_ddtau + (self.delta*self.dA_ddelta - self.delta*self.tau*self.ddA_ddelta_dtau)**2\
#                  /(2*self.delta*self.dA_ddelta + self.delta**2*self.ddA_dddelta))*R

        else:
            tau = self.tau
            Cp = -self.R*tau*tau*self.d2G_d2tau()
        Cp *= self._MW*1e-3
        self._Cp = Cp
        return Cp

    dH_dT = dH_dT_P = Cp

    ### Derivatives
    def dV_dP(self):
        '''
        from sympy import *
        R, T, MW, P, Pref, Tref = symbols('R, T, MW, P, Pref, Tref')
        dG_dpif = symbols('dG_dpi', cls=Function)
        pi = P/Pref
        tau = Tref/T
        dG_dpi = dG_dpif(tau, pi)
        V = (R*T*pi*dG_dpi*MW)/(1000*P)
        print(diff(V, P))

        MW*R*T*Subs(Derivative(dG_dpi(Tref/T, _xi_2), _xi_2), _xi_2, P/Pref)/(1000*Pref**2)
        '''
        try:
            return self._dV_dP
        except:
            pass
        if self.region != 3:
            try:
                d2G_d2pi = self._d2G_d2pi
            except:
                d2G_d2pi = self.d2G_d2pi()
            dV_dP = self._MW*self.R*self.T*d2G_d2pi/(1000.0*self.Pref*self.Pref)

        self._dV_dP = dV_dP
        return dV_dP

    def dV_dT(self):
        # similar to dV_dP
        try:
            return self._dV_dT
        except:
            pass
        if self.region != 3:
            try:
                dG_dpi = self._dG_dpi
            except:
                dG_dpi = self.dG_dpi()
            try:
                d2G_dpidtau = self._d2G_dpidtau
            except:
                d2G_dpidtau = self.d2G_dpidtau()


            dV_dT = (self._MW*self.R*dG_dpi/(1000*self.Pref)
            - self._MW*self.R*self.Tref*d2G_dpidtau/(1000*self.Pref*self.T))
        self._dV_dT = dV_dT
        return dV_dT

    def dP_dT(self):
        try:
            return self._dP_dT
        except:
            pass
        if self.region != 3:
            dP_dT = -self.dV_dT()/self.dV_dP()
        self._dP_dT = dP_dT
        return dP_dT

    def dP_dV(self):
        return 1.0/self.dV_dP()
