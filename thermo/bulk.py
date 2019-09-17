# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
__all__ = ['Bulk']

from fluids.constants import R, R_inv
from thermo.utils import log, exp
from thermo.phases import Phase

'''Class designed to have multiple phases.

Calculates dew, bubble points as properties (going to call back to property package)
I guess it's going to need MW as well.

Does not have any flow property.
'''


class Bulk(Phase):
    def __init__(self, T, P,
                 zs, phases, phase_fractions):
        self.T = T
        self.P = P
        self.zs = zs
        self.phases = phases
        self.phase_fractions = phase_fractions

    def MW(self):
        MWs = self.constants.MWs
        zs = self.zs
        MW = 0.0
        for i in range(len(MWs)):
            MW += zs[i]*MWs[i]
        return MW
        
    def V(self):
        # Is there a point to anything else?
        try:
            return self._V
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        V = 0.0
        for i in range(len(betas)):
            V += betas[i]*phases[i].V()
        self._V = V
        return V
    
    def Cp(self):
        try:
            return self._Cp
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        Cp = 0.0
        for i in range(len(betas)):
            Cp += betas[i]*phases[i].Cp()
        self._Cp = Cp
        return Cp

    def H(self):
        try:
            return self._H
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        H = 0.0
        for i in range(len(betas)):
            H += betas[i]*phases[i].H()
        self._H = H
        return H

    def S(self):
        try:
            return self._S
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        S = 0.0
        for i in range(len(betas)):
            S += betas[i]*phases[i].S()
        self._S = S
        return S
    
    def dH_dT(self):
        try:
            return self._dH_dT
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dH_dT = 0.0
        for i in range(len(betas)):
            dH_dT += betas[i]*phases[i].dH_dT()
        self._dH_dT = dH_dT
        return dH_dT

    def dH_dP(self):
        try:
            return self._dH_dP
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dH_dP = 0.0
        for i in range(len(betas)):
            dH_dP += betas[i]*phases[i].dH_dP()
        self._dH_dP = dH_dP
        return dH_dP

    def dS_dP(self):
        try:
            return self._dS_dP
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dS_dP = 0.0
        for i in range(len(betas)):
            dS_dP += betas[i]*phases[i].dS_dP()
        self._dS_dP = dS_dP
        return dS_dP

    def dS_dT(self):
        try:
            return self._dS_dT
        except AttributeError:
            pass
        
        betas, phases = self.phase_fractions, self.phases
        dS_dT = 0.0
        for i in range(len(betas)):
            dS_dT += betas[i]*phases[i].dS_dT()
        self._dS_dT = dS_dT
        return dS_dT