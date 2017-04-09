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

__all__ = ['Ideal_PP', 'UNIFAC_PP']

import numpy as np
from scipy.optimize import brenth, ridder, golden, brent

from thermo.utils import log
from thermo.utils import R, pi, N_A

from thermo.activity import K_value, flash_inner_loop, dew_at_T, bubble_at_T
from thermo.unifac import UNIFAC


class Ideal_PP(object):
    def _T_VF_err(self, P, VF, zs, Psats):
        Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
        return flash_inner_loop(zs=zs, Ks=Ks)[0] - VF
        
    def _P_VF_err(self, T, P, VF, zs):
        Psats = self._Psats(T)
        Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
        return flash_inner_loop(zs=zs, Ks=Ks)[0] - VF
    
    def _Psats(self, T):
        # Need to reset the method because for the T bounded solver,
        # will normally get a different than prefered method as it starts
        # at the boundaries
        Psats = []
        for i in self.VaporPressures:
            if T < i.Tmax:
                i.method = None
                Psats.append(i(T))
            else:
                Psats.append(i.extrapolate_tabular(T))
        return Psats
        
    def _Tsats(self, P):
        Tsats = []
        for i in self.VaporPressures:
            try: 
                Tsats.append(i.solve_prop(P))
            except:
                error = lambda T: i.extrapolate_tabular(T) - P
                Tsats.append(brenth(error, i.Tmax, i.Tmax*5))
        return Tsats
                
    def __init__(self, VaporPressures=None, Tms=None, Tcs=None, Pcs=None):
        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tcs = Tcs
        self.Pcs = Pcs
        
        
    def flash(self, zs, T=None, P=None, VF=None):
        if T is not None and P is not None:
            phase, xs, ys, V_over_F = self.flash_TP_zs(T=T, P=P, zs=zs)
        elif T is not None and VF is not None:
            phase, xs, ys, V_over_F, P = self.flash_TVF_zs(T=T, VF=VF, zs=zs)
        elif P is not None and VF is not None:
            phase, xs, ys, V_over_F, T = self.flash_PVF_zs(P=P, VF=VF, zs=zs)
        else:
            raise Exception('Unsupported flash requested')
        self.T = T
        self.P = P
        self.V_over_F = V_over_F
        self.phase = phase
        self.xs = xs
        self.ys = ys
        self.zs = zs
        

    def flash_TP_zs(self, T, P, zs):
        Psats = self._Psats(T)
        Pdew = dew_at_T(zs, Psats)
        Pbubble = bubble_at_T(zs, Psats)
        if P <= Pdew:
            # phase, ys, xs, quality
            return 'g', None, zs, 1
        elif P >= Pbubble:
            return 'l', zs, None, 0
        else:
            Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
            V_over_F, xs, ys = flash_inner_loop(zs=zs, Ks=Ks)
            return 'l/g', xs, ys, V_over_F
        
    def flash_TVF_zs(self, T, VF, zs):
        assert 0 <= VF <= 1
        Psats = self._Psats(T)
        if VF == 0:
            P = bubble_at_T(zs, Psats)
        elif VF == 1:
            P = dew_at_T(zs, Psats)
        else:
            P = brenth(self._T_VF_err, min(Psats)*(1+1E-7), max(Psats)*(1-1E-7), args=(VF, zs, Psats))
        Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
        V_over_F, xs, ys = flash_inner_loop(zs=zs, Ks=Ks)
        return 'l/g', xs, ys, V_over_F, P
    
    def flash_PVF_zs(self, P, VF, zs):
        assert 0 <= VF <= 1
        Tsats = self._Tsats(P)
        T = brenth(self._P_VF_err, min(Tsats)*(1+1E-7), max(Tsats)*(1-1E-7), args=(P, VF, zs))
        Psats = self._Psats(T)
        Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
        V_over_F, xs, ys = flash_inner_loop(zs=zs, Ks=Ks)
        return 'l/g', xs, ys, V_over_F, T


class UNIFAC_PP(object):
    __TP_cache = None
    __TVF_solve_cache = None

    def _P_VF_err(self, T, P, VF, zs):
        P_calc = self.flash_TVF_zs(T=T, VF=VF, zs=zs)[-1]
        return P_calc- P

    def __init__(self, UNIFAC_groups, VaporPressures, Tms=None, Tcs=None, Pcs=None):
        self.UNIFAC_groups = UNIFAC_groups
        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.cmps = range(len(VaporPressures))

    def _dew_P_UNIFAC_err(self, P, T, zs, Psats, Pmax):
        # returns 0 at Pdew, higher than that everywhere else. 
        # Raises exceptions at values of P somewhat higher than Pbubble.
        if P < 0:
            return 1 # Ensure P does not go negative
        if P > Pmax:
            return 1
        # Ensure P does not raise an exception 
        # by always keeping P under Pmax- golden handles this change well
        try:
            V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs, Psats=Psats)
            ans = -(V_over_F-1)
            
#            if any(i < 0 for i in xs):
#                return -100
#            if any(i < 0 for i in ys):
#                return -100
            
            if ans < 0:
                ans = -ans
            return ans
        except:
            # Return 1 if T went too low and the activity coefficients calculated are causing
            # negative V_over_F and negative mole fractions
            return 1

    def _T_VF_err(self, P, T, zs, Psats, Pmax, V_over_F_goal=1):
        if P < 0 or P > Pmax:
            return 1 
        try:
            
            V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs, Psats=Psats, restart=self.__TVF_solve_cache)

            if any(i < 0 for i in xs) or any(i < 0 for i in ys):
                return -10000*(Pmax-P)/Pmax
            self.__TVF_solve_cache = (V_over_F, xs, ys)
            ans = -(V_over_F-V_over_F_goal)
            return ans
        except:
            return 1

    def Ks(self, P, Psats, gammas):
        return [K_value(P=P, Psat=Psat, gamma=gamma) for Psat, gamma in zip(Psats, gammas)]

    def _Psats(self, Psats=None, T=None):
        if Psats is None:
            Psats = []
            for i in self.VaporPressures:
                if T < i.Tmax:
                    i.method = None
                    Psats.append(i(T))
                else:
                    Psats.append(i.extrapolate_tabular(T))
            return Psats
        else:
            return Psats
#
    def _flash_sequential_substitution_TP(self, T, P, zs, Psats=None, restart=None):
        Psats = self._Psats(Psats=Psats, T=T)
        if restart:
            V_over_F, xs, ys = restart
        else:
            gammas = UNIFAC(chemgroups=self.UNIFAC_groups, T=T, xs=zs)
            Ks = self.Ks(P, Psats, gammas)
            V_over_F, xs, ys = flash_inner_loop(zs, Ks)
        for i in range(100):
            if any(i < 0 for i in xs):
                xs = zs
            if any(i < 0 for i in ys):
                ys = zs
            gammas = UNIFAC(chemgroups=self.UNIFAC_groups, T=T, xs=xs)
            Ks = self.Ks(P, Psats, gammas)
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks)
            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
            xs, ys = xs_new, ys_new
            if err < 1E-7:
                break
        return V_over_F, xs, ys

    def flash(self, zs, T=None, P=None, VF=None):
        if T is not None and P is not None:
            phase, xs, ys, V_over_F = self.flash_TP_zs(T=T, P=P, zs=zs)
        elif T is not None and VF is not None:
            phase, xs, ys, V_over_F, P = self.flash_TVF_zs(T=T, VF=VF, zs=zs)
        elif P is not None and VF is not None:
            phase, xs, ys, V_over_F, T = self.flash_PVF_zs(P=P, VF=VF, zs=zs)
        else:
            raise Exception('Unsupported flash requested')
        self.T = T
        self.P = P
        self.V_over_F = V_over_F
        self.phase = phase
        self.xs = xs
        self.ys = ys
        self.zs = zs
        
    def P_bubble_at_T(self, T, zs, Psats=None):
        # Returns P_bubble; only thing easy to calculate
        Psats = self._Psats(Psats, T)
        gammas = UNIFAC(chemgroups=self.UNIFAC_groups, T=T, xs=zs)
        return sum([gammas[i]*zs[i]*Psats[i] for i in self.cmps])

    def P_dew_at_T(self, T, zs, Psats=None):
        Psats = self._Psats(Psats, T)
        Pmax = self.P_bubble_at_T(T, zs, Psats)
        diff = 1E-7
        P_dew = brenth(self._T_VF_err, Pmax*diff, Pmax, args=(T, zs, Psats, Pmax, 1))
        self.__TVF_solve_cache = None
        return P_dew
#        try:
#            return brent(self._dew_P_UNIFAC_err, args=(T, zs, Psats, Pmax), brack=(Pmax*diff, Pmax*(1-diff), Pmax))
#        except:
#        return golden(self._dew_P_UNIFAC_err, args=(T, zs, Psats, Pmax), brack=(Pmax, Pmax*(1-diff)))
#        
    def flash_TVF_zs(self, T, VF, zs):
        assert 0 <= VF <= 1
        Psats = self._Psats(T=T)
        Pbubble = self.P_bubble_at_T(T=T, zs=zs, Psats=Psats)
        if VF == 0:
            P = Pbubble
        else:
            diff = 1E-7
            Pmax = Pbubble
            P = brenth(self._T_VF_err, Pmax*diff, Pmax, args=(T, zs, Psats, Pmax, VF))
            self.__TVF_solve_cache = None
#            P = brenth(self._T_VF_err, Pdew, Pbubble, args=(T, VF, zs, Psats))
        V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs, Psats=Psats)
        return 'l/g', xs, ys, V_over_F, P
    
    def flash_TP_zs(self, T, P, zs):
        Psats = self._Psats(T=T)
        Pbubble = self.P_bubble_at_T(T=T, zs=zs, Psats=Psats)
        if P >= Pbubble:
            return 'l', zs, None, 0
        Pdew = self.P_dew_at_T(T=T, zs=zs, Psats=Psats)
        if P <= Pdew:
            # phase, ys, xs, quality
            return 'g', None, zs, 1
        else:
            V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs, Psats=Psats)
            return 'l/g', xs, ys, V_over_F
    
   
    def flash_PVF_zs(self, P, VF, zs):
        T = ridder(self._P_VF_err, min(self.Tms), min(self.Tcs), args=(P, VF, zs))
        V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs)
        return 'l/g', xs, ys, V_over_F, T

