'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2022 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
__all__ = ['GibbsEOS',]

from thermo.phases.phase import Phase


class GibbsEOS(Phase):
    # http://www.iapws.org/relguide/Advise3.pdf is useful
    def V(self):
        return self.dG_dP()
        # return self._V

    def dV_dP(self):
        # easy
        '''from sympy import *
        T, P = symbols('T, P')
        G = symbols('G', cls=Function)
        V = diff(G(T, P), P)
        diff(V, P)
        '''
        return self.d2G_dP2()

    dV_dP_T = dV_dP

    def dP_dV(self):
        return 1.0/self.d2G_dP2()

    dP_dV_T = dP_dV

    def d2V_dP2(self):
        return self.d3G_dP3()

    def dV_dT(self):
        return self.d2G_dTdP()

    def dT_dV(self):
        return 1.0/self.d2G_dTdP()

    dT_dV_P = dT_dV

    def d2V_dT2(self):
        return self.d3G_dT2dP()

    def d2V_dTdP(self):
        return self.d3G_dTdP2()


    def dP_dT(self):
        return -self.dV_dT()/self.dV_dP()

    dP_dT_V = dP_dT

    def dT_dP(self):
        return 1.0/self.dP_dT()

    dT_dP_V = dT_dP


    def H(self):
        return self.G() - self.T*self.dG_dT()

    def S(self):
        return -self.dG_dT()

    def Cp(self):
        return -self.T*self.d2G_dT2()

    def U(self):
        return self.G() - self.T*self.dG_dT() - self.P*self.dG_dP()

    def A(self):
        return self.G() - self.P*self.dG_dP()

    def dS_dT(self):
        return -self.d2G_dT2()
    dS_dT_P = dS_dT

    def d2S_dT2(self):
        return -self.d3G_dT3()
    d2S_dT2_P = d2S_dT2

    def dS_dP(self):
        return -self.d2G_dTdP()

    dS_dP_T = dS_dP

    def d2S_dP2(self):
        return -self.d3G_dTdP2()

    def d2S_dTdP(self):
        return -self.d3G_dT2dP()



