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
__all__ = ['CombinedPhase']

from .phase import Phase


class CombinedPhase(Phase):
    def __init__(self, phases, equilibrium=None, thermal=None, volume=None,
                 other_props=None,
                 T=None, P=None, zs=None,
                 ):
        # phases : list[other phases]
        # equilibrium: index
        # thermal: index
        # volume: index
        # other_props: dict[prop] = phase index

        # may be missing S_formation_ideal_gas Hfs arg
        self.equilibrium = equilibrium
        self.thermal = thermal
        self.volume = volume
        self.other_props = other_props

        for i, p in enumerate(phases):
            if p.T != T or p.P != P or p.zs != zs:
                phases[i] = p.to(T=T, P=P, zs=zs)
        self.phases = phases

    def lnphis(self):
        # This style will save the getattr call but takes more time to code
        if 'lnphis' in self.other_props:
            return self.phases[self.other_props['lnphis']].lnphis()
        if self.equilibrium is not None:
            return self.phases[self.equilibrium].lnphis()
        raise ValueError("No method specified")

    def lnphis_G_min(self):
        if 'lnphis' in self.other_props:
            return self.phases[self.other_props['lnphis']].lnphis_G_min()
        if self.equilibrium is not None:
            return self.phases[self.equilibrium].lnphis_G_min()
        raise ValueError("No method specified")

    def makeeqfun(prop_name):
        def fun(self):
            if prop_name in self.other_props:
                return getattr(self.phases[self.other_props[prop_name]], prop_name)()
            if self.equilibrium is not None:
                return getattr(self.phases[self.equilibrium], prop_name)()
            raise ValueError("No method specified")
        return fun

    def makethermalfun(prop_name):
        def fun(self):
            if prop_name in self.other_props:
                return getattr(self.phases[self.other_props[prop_name]], prop_name)()
            if self.thermal is not None:
                return getattr(self.phases[self.thermal], prop_name)()
            raise ValueError("No method specified")
        return fun

    def makevolumefun(prop_name):
        def fun(self):
            if prop_name in self.other_props:
                return getattr(self.phases[self.other_props[prop_name]], prop_name)()
            if self.volume is not None:
                return getattr(self.phases[self.volume], prop_name)()
            raise ValueError("No method specified")
        return fun

    lnphis = makeeqfun("lnphis")
    dlnphis_dT = makeeqfun("dlnphis_dT")
    dlnphis_dP = makeeqfun("dlnphis_dP")
    dlnphis_dns = makeeqfun("dlnphis_dns")

    V = makevolumefun("V")
    dP_dT = makevolumefun("dP_dT")
    dP_dT_V = dP_dT
    dP_dV = makevolumefun("dP_dV")
    dP_dV_T = dP_dV
    d2P_dT2 = makevolumefun("d2P_dT2")
    d2P_dT2_V = d2P_dT2
    d2P_dV2 = makevolumefun("d2P_dV2")
    d2P_dV2_T = d2P_dV2
    d2P_dTdV = makevolumefun("d2P_dTdV")

    H = makethermalfun("H")
    S = makethermalfun("S")
    Cp = makethermalfun("Cp")
    dH_dT = makethermalfun("dH_dT")
    dH_dP = makethermalfun("dH_dP")
    dH_dT_V = makethermalfun("dH_dT_V")
    dH_dP_V = makethermalfun("dH_dP_V")
    dH_dV_T = makethermalfun("dH_dV_T")
    dH_dV_P = makethermalfun("dH_dV_P")
    dH_dzs = makethermalfun("dH_dzs")
    dS_dT = makethermalfun("dS_dT")
    dS_dP = makethermalfun("dS_dP")
    dS_dT_P = makethermalfun("dS_dT_P")
    dS_dT_V = makethermalfun("dS_dT_V")
    dS_dP_V = makethermalfun("dS_dP_V")
    dS_dzs = makethermalfun("dS_dzs")

