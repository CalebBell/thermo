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
__all__ = ['GraysonStreed', 'ChaoSeader']

from chemicals.utils import log, log10
from thermo.activity import IdealSolution
from .phase import Phase

# hydrogen, methane
Grayson_Streed_special_CASs = set(['1333-74-0', '74-82-8'])

class GraysonStreed(Phase):
    phase = force_phase = 'l'
    is_gas = False
    is_liquid = True
    # revised one
    hydrogen_coeffs = (1.50709, 2.74283, -0.0211, 0.00011, 0.0, 0.008585, 0.0, 0.0, 0.0, 0.0)
    methane_coeffs = (1.36822, -1.54831, 0.0, 0.02889, -0.01076, 0.10486, -0.02529, 0.0, 0.0, 0.0)
    simple_coeffs = (2.05135, -2.10889, 0.0, -0.19396, 0.02282, 0.08852, 0.0, -0.00872, -0.00353, 0.00203)
    version = 1

    pure_references = tuple()
    model_attributes = ('Tcs', 'Pcs', 'omegas', '_CASs',
                        'GibbsExcessModel') + pure_references

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs

        new._Tcs = self._Tcs
        new._Pcs = self._Pcs
        new._omegas = self._omegas
        new._CASs = self._CASs
        new.regular = self.regular
        new.GibbsExcessModel = self.GibbsExcessModel.to_T_xs(T, zs)
        new.version = self.version

        try:
            new.N = self.N
        except:
            pass

        return new

    def to(self, zs, T=None, P=None, V=None):
        if T is not None:
            if P is not None:
                return self.to_TP_zs(T=T, P=P, zs=zs)
            elif V is not None:
                raise ValueError("Model does not implement volume")
        elif P is not None and V is not None:
            raise ValueError("Model does not implement volume")
        else:
            raise ValueError("Two of T, P, or V are needed")

    def __init__(self, Tcs, Pcs, omegas, CASs,
                 GibbsExcessModel=IdealSolution(),
                 T=None, P=None, zs=None,
                 ):

        self.T = T
        self.P = P
        self.zs = zs

        self.N = len(zs)
        self._Tcs = Tcs
        self._Pcs = Pcs
        self._omegas = omegas
        self._CASs = CASs
        self.regular = [i not in Grayson_Streed_special_CASs for i in CASs]

        self.GibbsExcessModel = GibbsExcessModel

    def gammas(self):
        try:
            return self.GibbsExcessModel._gammas
        except AttributeError:
            return self.GibbsExcessModel.gammas()

    def phis(self):
        try:
            return self._phis
        except AttributeError:
            pass
        try:
            gammas = self._gammas
        except AttributeError:
            gammas = self.gammas()
        fugacity_coeffs_pure = self.nus()

        self._phis = [gammas[i]*fugacity_coeffs_pure[i]
                for i in range(self.N)]
        return self._phis


    def lnphis(self):
        try:
            return self._lnphis
        except AttributeError:
            pass
        self._lnphis = [log(i) for i in self.phis()]
        return self._lnphis

    lnphis_G_min = lnphis

    def nus(self):
        T, P = self.T, self.P
        Tcs, Pcs, omegas = self._Tcs, self._Pcs, self._omegas
        regular, CASs = self.regular, self._CASs
        nus = []
        limit_Tr = self.version > 0

        for i in range(self.N):
            # TODO validate and take T, P derivatives; n derivatives are from regular solution only
            Tr = T/Tcs[i]
            Pr = P/Pcs[i]

            if regular[i]:
                coeffs = self.simple_coeffs
            elif CASs[i] == '1333-74-0':
                coeffs = self.hydrogen_coeffs
            elif CASs[i] == '74-82-8':
                coeffs = self.methane_coeffs
            else:
                raise ValueError("Fail")
            A0, A1, A2, A3, A4, A5, A6, A7, A8, A9 = coeffs

            log10_v0 = A0 + A1/Tr + A2*Tr + A3*Tr**2 + A4*Tr**3 + (A5 + A6*Tr + A7*Tr**2)*Pr + (A8 + A9*Tr)*Pr**2 - log10(Pr)
            log10_v1 = -4.23893 + 8.65808*Tr - 1.2206/Tr - 3.15224*Tr**3 - 0.025*(Pr - 0.6)
            if Tr > 1.0 and limit_Tr:
                log10_v1 = 1.0

            if regular[i]:
                v = 10.0**(log10_v0 + omegas[i]*log10_v1)
            else:
                # Chao Seader mentions
                v = 10.0**(log10_v0)
            nus.append(v)
        return nus

class ChaoSeader(GraysonStreed):
    # original one
    hydrogen_coeffs = (1.96718, 1.02972, -0.054009, 0.0005288, 0.0, 0.008585, 0.0, 0.0, 0.0, 0.0)
    methane_coeffs = (2.4384, -2.2455, -0.34084, 0.00212, -0.00223, 0.10486, -0.03691, 0.0, 0.0, 0.0)
    simple_coeffs = (5.75748, -3.01761, -4.985, 2.02299, 0.0, 0.08427, 0.26667, -0.31138, -0.02655, 0.02883)
    version = 0
