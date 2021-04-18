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
__all__ = ['HelmholtzEOS',]

from chemicals.utils import log
from .phase import Phase

class HelmholtzEOS(Phase):

    model_attributes = ('model_name',)

    def __repr__(self):
        r'''Method to create a string representation of the phase object, with
        the goal of making it easy to obtain standalone code which reproduces
        the current state of the phase. This is extremely helpful in creating
        new test cases.

        Returns
        -------
        recreation : str
            String which is valid Python and recreates the current state of
            the object if ran, [-]

        Examples
        --------
        >>> from thermo import IAPWS95Gas
        >>> phase = IAPWS95Gas(T=300, P=1e5, zs=[1])
        >>> phase
        IAPWS95Gas(T=300, P=100000.0, zs=[1.0])
        '''
        base = '%s('  %(self.__class__.__name__)
        for s in ('Hfs', 'Gfs', 'Sfs', 'T', 'P', 'zs'):
            if hasattr(self, s) and getattr(self, s) is not None:
                base += '%s=%s, ' %(s, getattr(self, s))
        if base[-2:] == ', ':
            base = base[:-2]
        base += ')'
        return base

    def V(self):
        return self._V

    def A(self):
        try:
            return self._A
        except:
            pass
        A = self._A = self.A0 + self._Ar_func(self.tau, self.delta)
        return A

    def dA_ddelta(self):
        try:
            return self._dA_ddelta
        except:
            pass
        delta = self.delta
        dA_ddelta = self._dA_ddelta = self._dAr_ddelta_func(self.tau, delta) + 1./delta
        return dA_ddelta

    def d2A_ddelta2(self):
        try:
            return self._d2A_ddelta2
        except:
            pass
        delta = self.delta
        self._d2A_ddelta2 = d2A_ddelta2 = self._d2Ar_ddelta2_func(self.tau, delta) - 1./(delta*delta)
        return d2A_ddelta2


    def d3A_ddelta3(self):
        try:
            return self._d3A_ddelta3
        except:
            pass
        delta = self.delta
        self._d3A_ddelta3 = d3A_ddelta3 = self._d3Ar_ddelta3_func(self.tau, delta) + 2./(delta*delta*delta)
        return d3A_ddelta3

    def dA_dtau(self):
        try:
            return self._dA_dtau
        except:
            pass
        dA_dtau = self._dA_dtau = self._dAr_dtau_func(self.tau, self.delta) + self.dA0_dtau
        return dA_dtau

    def d2A_dtau2(self):
        try:
            return self._d2A_dtau2
        except:
            pass
        self._d2A_dtau2 = d2A_dtau2 = self._d2Ar_dtau2_func(self.tau, self.delta) + self.d2A0_dtau2
        return d2A_dtau2

    def d2A_ddeltadtau(self):
        try:
            return self._d2A_ddeltadtau
        except:
            pass
        self._d2A_ddeltadtau = d2A_ddeltadtau = self._d2Ar_ddeltadtau_func(self.tau, self.delta)
        return d2A_ddeltadtau

    def d3A_ddelta2dtau(self):
        try:
            return self._d3A_ddelta2dtau
        except:
            pass
        self._d3A_ddelta2dtau = d3A_ddelta2dtau = self._d3Ar_ddelta2dtau_func(self.tau, self.delta)
        return d3A_ddelta2dtau

    def d3A_ddeltadtau2(self):
        try:
            return self._d3A_ddeltadtau2
        except:
            pass
        self._d3A_ddeltadtau2 = d3A_ddeltadtau2 = self._d3Ar_ddeltadtau2_func(self.tau, self.delta)
        return d3A_ddeltadtau2

    def lnphis(self):
        delta = self.delta
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            A = self._A
        except:
            A = self.A()
        x0 = delta*(dA_ddelta - 1.0/delta)
        lnphi = A - self.A0 + x0 - log(x0 + 1.0)

        return [lnphi]

    def dlnphis_dV_T(self):
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        delta = self.delta
        rho, rho_red = 1.0/self._V, self.rho_red
        x0 = self.rho_red_inv
        rho_inv = 1.0/rho

        x1 = -rho_red*rho_inv
        x2 = rho*(d2A_ddelta2*x0 + rho_red*rho_inv*rho_inv)
        x3 = dA_ddelta + x1
        dlnphi_dV_T = x0*(2.0*dA_ddelta + x1 + x2 - (x2 + x3)/(rho*x0*x3 + 1.0) - 1.0/delta)
#
        dlnphi_dV_T *= -1.0/(self._V*self._V)

        return [dlnphi_dV_T]

    def dlnphis_dT_V(self):
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()
        try:
            dA_dtau = self._dA_dtau
        except:
            dA_dtau = self.dA_dtau()

        rho = 1.0/self._V
        rho_red_inv = self.rho_red_inv
        rho_red = self.rho_red

        T_red, T = self.T_red, self.T

        dlnphi_dT_V = T_red/(T*T)*(rho_red_inv*rho*d2A_ddeltadtau*(-1.0
                        + 1.0/(rho*(dA_ddelta - rho_red/rho)*rho_red_inv + 1.0))
                        - dA_dtau
                        + self.dA0_dtau)
        return [dlnphi_dT_V]

    def dlnphis_dT_P(self):
        try:
            return self._dlnphis_dT_P
        except:
            pass

        self._dlnphis_dT_P = dlnphis_dT_P = [self.dlnphis_dT_V()[0] - self.dlnphis_dV_T()[0]*self.dP_dT_V()/self.dP_dV_T()]
        return dlnphis_dT_P
    dlnphis_dT = dlnphis_dT_P

    def dlnphis_dP_V(self):
        return [self.dlnphis_dT_V()[0]/self.dP_dT_V()]

    def dlnphis_dV_P(self):
        return [self.dlnphis_dT_P()[0]/self.dV_dT_P()]

    def dlnphis_dP_T(self):
        return [self.dlnphis_dP_V()[0] - self.dlnphis_dV_P()[0]*self.dT_dP_V()/self.dT_dV_P()]

    dlnphis_dP = dlnphis_dP_T

    def S(self):
        try:
            return self._S
        except:
            pass
        try:
            dA_dtau = self._dA_dtau
        except:
            dA_dtau = self.dA_dtau()
        try:
            A = self._A
        except:
            A = self.A()
        self._S = S = (self.tau*dA_dtau - A)*self.R
        return S

    def S_dep(self):
        S_ideal = (self.tau*self.dA0_dtau - self.A0)*self.R
        return self.S() - S_ideal


    def dS_dT_V(self):
        try:
            return self._dS_dT_V
        except:
            pass
        try:
            d2A_dtau2 = self._d2A_dtau2
        except:
            d2A_dtau2 = self.d2A_dtau2()

        T, T_red = self.T, self.T_red
        self._dS_dT_V = dS_dT_V = (-T_red*T_red*d2A_dtau2)*self.R/(T*T*T)
        return dS_dT_V

    def dS_dV_T(self):
        try:
            return self._dS_dV_T
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()

        R, T, V, rho_red, T_red = self.R, self.T, self._V, self.rho_red, self.T_red

        self._dS_dV_T = dS_dV_T = R*(dA_ddelta - T_red*d2A_ddeltadtau/(T))/(V*V*rho_red)
        return dS_dV_T

    def dS_dP_T(self):
        # From chain rule
        return self.dS_dV_T()/self.dP_dV()
    dS_dP = dS_dP_T


    def dS_dT_P(self):
        try:
            return self._dS_dT_P
        except:
            pass

        self._dS_dT_P = dS_dT_P = self.dS_dT_V() - self.dS_dV_T()*self.dP_dT_V()/self.dP_dV_T()
        return dS_dT_P
    dS_dT = dS_dT_P

    def dS_dP_V(self):
        return self.dS_dT_V()/self.dP_dT_V()

    def H(self):
        try:
            return self._H
        except:
            pass
        try:
            dA_dtau = self._dA_dtau
        except:
            dA_dtau = self.dA_dtau()
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        self._H = H = (self.tau*dA_dtau + dA_ddelta*self.delta)*self.T*self.R
        return H

    def H_dep(self):
        try:
            return self._H_dep
        except:
            pass
        # Might be right, might not - hard to check due to reference state
        H = self.H()
        dA_dtau = self.dA0_dtau
        dA_ddelta = 1./self.delta

        H_ideal = (self.tau*dA_dtau + dA_ddelta*self.delta)*self.T*self.R
        self._H_dep = H - H_ideal
        return self._H_dep

    def dH_dT_V(self):
        try:
            return self._dH_dT_V
        except:
            pass
        try:
            dA_dtau = self._dA_dtau
        except:
            dA_dtau = self.dA_dtau()
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()
        try:
            d2A_dtau2 = self._d2A_dtau2
        except:
            d2A_dtau2 = self.d2A_dtau2()


        T, T_red, V, R, rho_red = self.T, self.T_red, self._V, self.R, self.rho_red
        T_inv = 1.0/T

        self._dH_dT_V = dH_dT_V = R*T*(-T_red*dA_dtau*T_inv*T_inv - T_inv*T_inv*T_red*d2A_ddeltadtau/(V*rho_red) - T_red*T_red*T_inv*T_inv*T_inv*d2A_dtau2) + R*(dA_ddelta/(V*rho_red) + T_red*dA_dtau/T)
        return dH_dT_V

    def dH_dV_T(self):
        try:
            return self._dH_dV_T
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()

        T, T_red, V, R, rho_red = self.T, self.T_red, self._V, self.R, self.rho_red

        self._dH_dV_T = dH_dV_T = R*T*(-dA_ddelta/(V*V*rho_red) - d2A_ddelta2/(V*V*V*rho_red*rho_red) - T_red*d2A_ddeltadtau/(T*V*V*rho_red))
        return dH_dV_T

    def dH_dP_V(self):
        return self.dH_dT_V()/self.dP_dT_V()

    def dH_dP_T(self):
        # From chain rule
        return self.dH_dV_T()/self.dP_dV()
    dH_dP = dH_dP_T

    def dH_dV_P(self):
        return self.dH_dT_P()/self.dV_dT_P()



    def Cv(self):
        try:
            return self._Cv
        except:
            pass
        try:
            d2A_dtau2 = self._d2A_dtau2
        except:
            d2A_dtau2 = self.d2A_dtau2()
        tau = self.tau
        self._Cv = Cv = -tau*tau*d2A_dtau2*self.R
        return Cv

    def Cp(self):
        try:
            return self._Cp
        except:
            pass
        tau, delta = self.tau, self.delta
        try:
            d2A_dtau2 = self._d2A_dtau2
        except:
            d2A_dtau2 = self.d2A_dtau2()
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()

        x0 = delta*dA_ddelta - delta*tau*d2A_ddeltadtau
        den = delta*(dA_ddelta + dA_ddelta + delta*d2A_ddelta2)
        self._Cp = Cp = (-tau*tau*d2A_dtau2 + x0*x0/den)*self.R
        return Cp

    dH_dT = dH_dT_P = Cp


    def dP_dT(self):
        try:
            return self._dP_dT
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()
        self._dP_dT = dP_dT = self.R/self._V*self.delta*(dA_ddelta - self.tau*d2A_ddeltadtau)
        return dP_dT

    def dP_dV(self):
        try:
            return self._dP_dV
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        delta = self.delta

        rho = self.rho()
        self._dP_dV = dP_dV = -self.R*rho*rho*self.T*delta*(2.0*dA_ddelta + delta*d2A_ddelta2)
        return dP_dV

    dP_dT_V = dP_dT
    dP_dV_T = dP_dV

    def d2P_dT2(self):
        try:
            return self._d2P_dT2
        except:
            pass
        d3A_ddeltadtau2 = self.d3A_ddeltadtau2()
        T, T_red, delta, V, rho_red, R = self.T, self.T_red, self.delta, self._V, self.rho_red, self.R
        self._d2P_dT2 = d2P_dT2 = R*T_red*T_red*d3A_ddeltadtau2/(T*T*T*V*V*rho_red)
        return d2P_dT2

    def d2P_dV2(self):
        try:
            return self._d2P_dV2
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        try:
            d3A_ddelta3 = self._d3A_ddelta3
        except:
            d3A_ddelta3 = self.d3A_ddelta3()
        '''
        from sympy import *
        from chemicals import Vm_to_rho
        T_red, rho_red, V, R = symbols('T_red, rho_red, V, R')
        T = symbols('T')
        rho = 1/V

        iapws95_dA_ddelta = symbols('ddelta', cls=Function)

        rho_red_inv = (1/rho_red)
        tau = T_red/T
        delta = rho*rho_red_inv
        dA_ddelta = iapws95_dA_ddelta(tau, delta)
        P = (dA_ddelta*delta)*rho*R*T
        print(diff(P,V, 2))
        '''
        R, T, V, rho_red = self.R, self.T, self._V, self.rho_red

        self._d2P_dV2 = d2P_dV2 = R*T*(6.0*dA_ddelta + (2*d2A_ddelta2 + d3A_ddelta3/(V*rho_red))/(V*rho_red) + 4*d2A_ddelta2/(V*rho_red))/(V**4*rho_red)
        return d2P_dV2

    def d2P_dTdV(self):
        try:
            return self._d2P_dTdV
        except:
            pass
        try:
            dA_ddelta = self._dA_ddelta
        except:
            dA_ddelta = self.dA_ddelta()
        try:
            d2A_ddelta2 = self._d2A_ddelta2
        except:
            d2A_ddelta2 = self.d2A_ddelta2()
        try:
            d2A_ddeltadtau = self._d2A_ddeltadtau
        except:
            d2A_ddeltadtau = self.d2A_ddeltadtau()
        d3A_ddelta2dtau = self.d3A_ddelta2dtau()
        R, T, T_red, V, rho_red = self.R, self.T, self.T_red, self._V, self.rho_red

        self._d2P_dTdV = d2P_dTdV = R*(-2.0*dA_ddelta - d2A_ddelta2/(V*rho_red) + 2.0*T_red*d2A_ddeltadtau/T
                                       + T_red*d3A_ddelta2dtau/(T*V*rho_red))/(V*V*V*rho_red)
        return d2P_dTdV

    def B_virial(self):
        try:
            f0 = self._dAr_ddelta_func(self.tau, 0.0)
        except:
            f0 = self._dAr_ddelta_func(self.tau, 1e-20)
        return f0/self.rho_red

    def dB_virial_dT(self):
        tau = self.tau
        try:
            f0 = self._d2Ar_ddeltadtau_func(tau, 0.0)
        except:
            f0 = self._d2Ar_ddeltadtau_func(tau, 1e-20)
        return -tau*tau*f0/(self.rho_red*self.T_red)

    def d2B_virial_dT2(self):
        T, T_red, tau = self.T, self.T_red, self.tau

        try:
            f0 = self._d2Ar_ddeltadtau_func(tau, 0.0)
        except:
            f0 = self._d2Ar_ddeltadtau_func(tau, 1e-20)
        try:
            f1 = self._d3Ar_ddeltadtau2_func(tau, 0.0)
        except:
            f1 = self._d3Ar_ddeltadtau2_func(tau, 1e-20)
        return T_red*(2.0*f0 + tau*f1)/(T*T*T*self.rho_red)

    def d3B_virial_dT3(self):
        T, T_red, tau = self.T, self.T_red, self.tau

        try:
            f0 = self._d2Ar_ddeltadtau_func(tau, 0.0)
        except:
            f0 = self._d2Ar_ddeltadtau_func(tau, 1e-20)
        try:
            f1 = self._d3Ar_ddeltadtau2_func(tau, 0.0)
        except:
            f1 = self._d3Ar_ddeltadtau2_func(tau, 1e-20)
        try:
            f2 = self._d4Ar_ddeltadtau3_func(tau, 0.0)
        except:
            f2 = self._d4Ar_ddeltadtau3_func(tau, 1e-20)
        return (-T_red*(6.0*f0 + 6.0*tau*f1 + tau*tau*f2)/(T*T*T*T*self.rho_red))

    def C_virial(self):
        try:
            f0 = self._d2Ar_ddelta2_func(self.tau, 0.0)
        except:
            f0 = self._d2Ar_ddelta2_func(self.tau, 1e-20)
        return f0/(self.rho_red*self.rho_red)

    def dC_virial_dT(self):
        tau = self.tau
        try:
            f0 = self._d3Ar_ddelta2dtau_func(tau, 0.0)
        except:
            f0 = self._d3Ar_ddelta2dtau_func(tau, 1e-20)
        return -tau*tau*f0/(self.rho_red*self.rho_red*self.T_red)

    def d2C_virial_dT2(self):
        T, T_red, tau = self.T, self.T_red, self.tau
        try:
            f0 = self._d3Ar_ddelta2dtau_func(tau, 0.0)
        except:
            f0 = self._d3Ar_ddelta2dtau_func(tau, 1e-20)

        try:
            f1 = self._d4Ar_ddelta2dtau2_func(tau, 0.0)
        except:
            f1 = self._d4Ar_ddelta2dtau2_func(tau, 1e-20)
        return T_red*(2.0*f0 + tau*f1)/(T*T*T*self.rho_red*self.rho_red)

