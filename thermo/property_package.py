  # -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

__all__ = ['Property_Package', 'Ideal_PP', 'UNIFAC_PP', 'Activity_PP', 
           'UNIFAC_Dortmund_PP', 'IdealPPThermodynamic']

import numpy as np
from scipy.optimize import brenth, ridder, golden, brent

from thermo.utils import log, exp
from thermo.utils import has_matplotlib, R, pi, N_A

from thermo.activity import K_value, flash_inner_loop, dew_at_T, bubble_at_T
from thermo.unifac import UNIFAC, UFSG, DOUFSG, DOUFIP2006

if has_matplotlib:
    import matplotlib
    import matplotlib.pyplot as plt


class Property_Package(object):
    # Constant - if the phase fraction is this close to either the liquid or 
    # vapor phase, round it to it
    PHASE_ROUNDING_TOL = 1E-9 
    
    def to(self, zs, T=None, P=None, VF=None):
        obj = self.__class__(**self.kwargs)
        obj.flash(T=T, P=P, VF=VF, zs=zs)
        return obj    
    
    def __copy__(self):
        obj = self.__class__(**self.kwargs)
        return obj
    
    def Tdew(self, P, zs):
        return self.to(P=P, VF=1, zs=zs).T
    
    def Pdew(self, T, zs):
        return self.to(T=T, VF=1, zs=zs).P
    
    def Tbubble(self, P, zs):
        return self.to(P=P, VF=0, zs=zs).T
    
    def Pbubble(self, T, zs):
        return self.to(T=T, VF=0, zs=zs).P
    
    def _post_flash(self):
        pass
    
    def flash(self, zs, T=None, P=None, VF=None):
        if any(i == 0 for i in zs):
            zs = [i if i != 0 else 1E-11 for i in zs]
            z_tot = sum(zs)
            zs = [i/z_tot for i in zs]
            
        if T is not None and P is not None:
            phase, xs, ys, V_over_F = self.flash_TP_zs(T=T, P=P, zs=zs)
        elif T is not None and VF is not None:
            phase, xs, ys, V_over_F, P = self.flash_TVF_zs(T=T, VF=VF, zs=zs)
        elif P is not None and VF is not None:
            phase, xs, ys, V_over_F, T = self.flash_PVF_zs(P=P, VF=VF, zs=zs)
        else:
            raise Exception('Unsupported flash requested')
        # Truncate 
        if phase  == 'l/g':
            if V_over_F < self.PHASE_ROUNDING_TOL: # liquid
                phase, xs, ys, V_over_F = 'l', zs, None, 0.
            elif V_over_F > 1. - self.PHASE_ROUNDING_TOL:
                phase, xs, ys, V_over_F = 'g', None, zs, 1.
                
        self.T = T
        self.P = P
        self.V_over_F = V_over_F
        self.phase = phase
        self.xs = xs
        self.ys = ys
        self.zs = zs
        
        self._post_flash()
        
    def plot_Pxy(self, T, pts=30): # pragma: no cover
        if not has_matplotlib:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if self.N != 2:
            raise Exception('Pxy plotting requires a mixture of exactly two components')
        z1 = np.linspace(0, 1, pts)
        z2 = 1 - z1
        Ps_dew = []
        Ps_bubble = []
        
        for i in range(pts):
            self.flash(T=T, VF=0, zs=[z1[i], z2[i]])
            Ps_bubble.append(self.P)
            self.flash(T=T, VF=1, zs=[z1[i], z2[i]])
            Ps_dew.append(self.P)
            
        plt.title('Pxy diagram at T=%s K' %T)
        plt.plot(z1, Ps_dew, label='Dew pressure')
        plt.plot(z1, Ps_bubble, label='Bubble pressure')
        plt.xlabel('Mole fraction x1')
        plt.ylabel('System pressure, Pa')
        plt.legend(loc='best')
        plt.show()
        
    def plot_Txy(self, P, pts=30): # pragma: no cover
        if not has_matplotlib:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if self.N != 2:
            raise Exception('Txy plotting requires a mixture of exactly two components')
        z1 = np.linspace(0, 1, pts)
        z2 = 1 - z1
        Ts_dew = []
        Ts_bubble = []
        
        for i in range(pts):
            self.flash(P=P, VF=0, zs=[z1[i], z2[i]])
            Ts_bubble.append(self.T)
            self.flash(P=P, VF=1, zs=[z1[i], z2[i]])
            Ts_dew.append(self.T)
            
        plt.title('Txy diagram at P=%s Pa' %P)
        plt.plot(z1, Ts_dew, label='Dew temperature, K')
        plt.plot(z1, Ts_bubble, label='Bubble temperature, K')
        plt.xlabel('Mole fraction x1')
        plt.ylabel('System temperature, K')
        plt.legend(loc='best')
        plt.show()
        
    def plot_xy(self, P=None, T=None, pts=30): # pragma: no cover
        if not has_matplotlib:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if self.N != 2:
            raise Exception('xy plotting requires a mixture of exactly two components')
        
        z1 = np.linspace(0, 1, pts)
        z2 = 1 - z1
        y1_bubble = []
        x1_bubble = []
        for i in range(pts):
            if T:
                self.flash(T=T, VF=0, zs=[z1[i], z2[i]])
            elif P:
                self.flash(P=P, VF=0, zs=[z1[i], z2[i]])
            x1_bubble.append(self.xs[0])
            y1_bubble.append(self.ys[0])
        if T:
            plt.title('xy diagram at T=%s K (varying P)' %T)
        else:
            plt.title('xy diagram at P=%s Pa (varying T)' %P)
        plt.xlabel('Liquid mole fraction x1')
        plt.ylabel('Vapor mole fraction x1')
        plt.plot(x1_bubble, y1_bubble, '-', label='liquid vs vapor composition')
        plt.legend(loc='best')
        plt.plot([0, 1], [0, 1], '--')
        plt.axis((0,1,0,1))
        plt.show()
        
    def plot_TP(self, zs, Tmin=None, Tmax=None, pts=50, branches=[]): # pragma: no cover
        if not has_matplotlib:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if not Tmin:
            Tmin = min(self.Tms)
        if not Tmax:
            Tmax = min(self.Tcs)
        Ts = np.linspace(Tmin, Tmax, pts)
        P_dews = []
        P_bubbles = []
        branch = bool(len(branches))
        if branch:
            branch_Ps = [[] for i in range(len(branches))]
        for T in Ts:
            self.flash(T=T, VF=0, zs=zs)
            P_bubbles.append(self.P)
            self.flash(T=T, VF=1, zs=zs)
            P_dews.append(self.P)
            if branch:
                for VF, Ps in zip(branches, branch_Ps):
                    self.flash(T=T, VF=VF, zs=zs)
                    Ps.append(self.P)
        
        plt.plot(Ts, P_dews, label='PT dew point curve')
        plt.plot(Ts, P_bubbles, label='PT bubble point curve')
        plt.xlabel('System temperature, K')
        plt.ylabel('System pressure, Pa')
        plt.title('PT system curve, zs=%s' %zs)
        if branch:
            for VF, Ps in zip(branches, branch_Ps):
                plt.plot(Ts, Ps, label='PT curve for VF=%s'%VF)
        plt.legend(loc='best')
        plt.show()
        
    @staticmethod
    def un_zero_zs(zs):
        if any(i == 0 for i in zs):
            zs = [i if i != 0 else 1E-6 for i in zs]
            z_tot = sum(zs)
            zs = [i/z_tot for i in zs]
        return zs

                
    def plot_ternary(self, T, scale=10): # pragma: no cover
        if not has_matplotlib:
            raise Exception('Optional dependency matplotlib is required for plotting')
        try:
            import ternary
        except:
            raise Exception('Optional dependency ternary is required for ternary plotting')
        if self.N != 3:
            raise Exception('Ternary plotting requires a mixture of exactly three components')

        P_values = []

        def P_dew_at_T_zs(zs):
            zs = self.un_zero_zs(zs)
            self.flash(T=T, zs=zs, VF=0)
            P_values.append(self.P)
            return self.P
        
        def P_bubble_at_T_zs(zs):
            zs = self.un_zero_zs(zs)
            self.flash(T=T, zs=zs, VF=1)
            return self.P
        
        
        axes_colors = {'b': 'g', 'l': 'r', 'r':'b'}
        ticks = [round(i / float(10), 1) for i in range(10+1)]
        
        fig, ax = plt.subplots(1, 3, gridspec_kw = {'width_ratios':[4, 4, 1]})
        ax[0].axis("off") ; ax[1].axis("off")  ; ax[2].axis("off")
        
        for axis, f, i in zip(ax[0:2], [P_dew_at_T_zs, P_bubble_at_T_zs], [0, 1]):
            figure, tax = ternary.figure(ax=axis, scale=scale)
            figure.set_size_inches(12, 4)
            if not i:
                tax.heatmapf(f, boundary=True, colorbar=False, vmin=0)
            else:
                tax.heatmapf(f, boundary=True, colorbar=False, vmin=0, vmax=max(P_values))
        
            tax.boundary(linewidth=2.0)
            tax.left_axis_label("mole fraction $x_2$", offset=0.16, color=axes_colors['l'])
            tax.right_axis_label("mole fraction $x_1$", offset=0.16, color=axes_colors['r'])
            tax.bottom_axis_label("mole fraction $x_3$", offset=-0.06, color=axes_colors['b'])
        
            tax.ticks(ticks=ticks, axis='rlb', linewidth=1, clockwise=True,
                      axes_colors=axes_colors, offset=0.03)
        
            tax.gridlines(multiple=scale/10., linewidth=2,
                          horizontal_kwargs={'color':axes_colors['b']},
                          left_kwargs={'color':axes_colors['l']},
                          right_kwargs={'color':axes_colors['r']},
                          alpha=0.5)
        
        norm = plt.Normalize(vmin=0, vmax=max(P_values))
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
        sm._A = []
        cb = plt.colorbar(sm, ax=ax[2])
        cb.locator = matplotlib.ticker.LinearLocator(numticks=7)
        cb.formatter = matplotlib.ticker.ScalarFormatter()
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        plt.tight_layout()
        fig.suptitle("Bubble pressure vs composition (left) and dew pressure vs composition (right) at %s K, in Pa" %T, fontsize=14); 
        fig.subplots_adjust(top=0.85)
        plt.show()


class Ideal_PP(Property_Package):
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
        self.N = len(VaporPressures)
        self.cmps = range(self.N)
        
        self.kwargs = {'VaporPressures': VaporPressures,
                       'Tms': Tms, 'Tcs': Tcs, 'Pcs': Pcs}

    def flash_TP_zs(self, T, P, zs):
        Psats = self._Psats(T)
        if self.N == 1:
            Pdew = Pbubble = Psats[0]
        else:
            Pdew = dew_at_T(zs, Psats)
            Pbubble = bubble_at_T(zs, Psats)
        if P <= Pdew:
            # phase, ys, xs, quality - works for 1 comps too
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
        # handle one component
        if self.N == 1:
            return 'l/g', [1.0], [1.0], VF, Psats[0]
        
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
        # handle one component
        if self.N == 1:
            return 'l/g', [1.0], [1.0], VF, Tsats[0]
        
        T = brenth(self._P_VF_err, min(Tsats)*(1+1E-7), max(Tsats)*(1-1E-7), args=(P, VF, zs))
        Psats = self._Psats(T)
        Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
        V_over_F, xs, ys = flash_inner_loop(zs=zs, Ks=Ks)
        return 'l/g', xs, ys, V_over_F, T


class IdealPPThermodynamic(Ideal_PP):
    T_REF_IG = 298.15
    P_REF_IG = 101325.
    
    def __init__(self, VaporPressures=None, Tms=None, Tbs=None, Tcs=None, Pcs=None, 
                 HeatCapacityLiquids=None, HeatCapacityGases=None,
                EnthalpyVaporizations=None):
        self.cmps = range(len(VaporPressures))
        self.N = len(VaporPressures)
        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tbs = Tbs
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.HeatCapacityLiquids = HeatCapacityLiquids
        self.HeatCapacityGases = HeatCapacityGases
        self.EnthalpyVaporizations = EnthalpyVaporizations
        
        self.kwargs = {'VaporPressures': VaporPressures,
                       'Tms': Tms, 'Tbs': Tbs, 'Tcs': Tcs, 'Pcs': Pcs,
                       'HeatCapacityLiquids': HeatCapacityLiquids, 
                       'HeatCapacityGases': HeatCapacityGases,
                       'EnthalpyVaporizations': EnthalpyVaporizations}
        
    def _post_flash(self):
        # Cannot derive other properties with this
        self.Hm = self.enthalpy_Cpg_Hvap()
        self.Sm = self.entropy_Cpg_Hvap()
        self.Gm = self.Hm - self.T*self.Sm if (self.Hm is not None and self.Sm is not None) else None

#        self.G = self.H - self.T*self.S if (self.H is not None and self.S is not None) else None


    def enthalpy_Cpg_Hvap(self):
        r'''Method to calculate the enthalpy of an ideal mixture (no pressure
        effects). This routine is based on "route A", where only the gas heat
        capacity and enthalpy of vaporization are used.
        
        The reference temperature is a property of the class; it defaults to
        298.15 K.
        
        For a pure gas mixture:
            
        .. math::
             H = \sum_i z_i \cdot \int_{T_{ref}}^T C_{p}^{ig}(T) dT
             
        For a pure liquid mixture:
            
        .. math::
             H = \sum_i z_i \left( \int_{T_{ref}}^T C_{p}^{ig}(T) dT + H_{vap, i}(T) \right)
             
        For a vapor-liquid mixture:
            
        .. math::
             H = \sum_i z_i \cdot \int_{T_{ref}}^T C_{p}^{ig}(T) dT
                 + \sum_i x_i\left(1 - \frac{V}{F}\right)H_{vap, i}(T)

        Returns
        -------
        H : float
            Enthalpy of the mixture with respect to the reference temperature,
            [J/mol]
            
        Notes
        -----
        The object must be flashed before this routine can be used. It 
        depends on the properties T, zs, xs, V_over_F, HeatCapacityGases, 
        EnthalpyVaporizations, and.
        '''
        H = 0
        T = self.T
        if self.phase == 'g':
            for i in self.cmps:
                H += self.zs[i]*self.HeatCapacityGases[i].T_dependent_property_integral(self.T_REF_IG, T)
        elif self.phase == 'l':
            for i in self.cmps:
                # No further contribution needed
                Hg298_to_T = self.HeatCapacityGases[i].T_dependent_property_integral(self.T_REF_IG, T)
                Hvap = -self.EnthalpyVaporizations[i](T) # Do the transition at the temperature of the liquid
                H += self.zs[i]*(Hg298_to_T + Hvap)
        elif self.phase == 'l/g':
            for i in self.cmps:
                Hg298_to_T_zi = self.zs[i]*self.HeatCapacityGases[i].T_dependent_property_integral(self.T_REF_IG, T)
                Hvap_contrib = -self.xs[i]*(1-self.V_over_F)*self.EnthalpyVaporizations[i](T)
                H += (Hg298_to_T_zi + Hvap_contrib)
        return H

    def set_T_transitions(self, Ts):
        if Ts == 'Tb':
            self.T_trans = self.Tbs
        elif Ts == 'Tc':
            self.T_trans = self.Tcs
        elif isinstance(Ts, float):
            self.T_trans = [Ts]*self.N
        else:
            self.T_trans = Ts

    def enthalpy_Cpl_Cpg_Hvap(self):
        H = 0
        T = self.T
        T_trans = self.T_trans
        
        if self.phase == 'l':
            for i in self.cmps:
                H += self.zs[i]*self.HeatCapacityLiquids[i].T_dependent_property_integral(self.T_REF_IG, T)
        elif self.phase == 'g':
            for i in self.cmps:
                H_to_trans = self.HeatCapacityLiquids[i].T_dependent_property_integral(self.T_REF_IG, self.T_trans[i])
                H_trans = self.EnthalpyVaporizations[i](self.T_trans[i]) 
                H_to_T_gas = self.HeatCapacityGases[i].T_dependent_property_integral(self.T_trans[i], T)
                H += self.zs[i]*(H_to_trans + H_trans + H_to_T_gas)
        elif self.phase == 'l/g':
            for i in self.cmps:
                H_to_T_liq = self.HeatCapacityLiquids[i].T_dependent_property_integral(self.T_REF_IG, T)
                H_to_trans = self.HeatCapacityLiquids[i].T_dependent_property_integral(self.T_REF_IG, self.T_trans[i])
                H_trans = self.EnthalpyVaporizations[i](self.T_trans[i])
                H_to_T_gas = self.HeatCapacityGases[i].T_dependent_property_integral(self.T_trans[i], T)
                H += self.V_over_F*self.ys[i]*(H_to_trans + H_trans + H_to_T_gas)
                H += (1-self.V_over_F)*self.xs[i]*(H_to_T_liq)
        return H


    def entropy_Cpg_Hvap(self):
        r'''Method to calculate the entropy of an ideal mixture. This routine 
        is based on "route A", where only the gas heat capacity and enthalpy of
        vaporization are used.
        
        The reference temperature and pressure are properties of the class; it 
        defaults to 298.15 K and 101325 Pa.
        
        There is a contribution due to mixing:
            
        .. math::
            \Delta S_{mixing} = -R\sum_i z_i \log(z_i) 
            
        The ideal gas pressure contribution is:
            
        .. math::
            \Delta S_{P} = -R\log\left(\frac{P}{P_{ref}}\right)
            
        For a liquid mixture or a partially liquid mixture, the entropy
        contribution is not so strong - all such pressure effects find that
        expression capped at the vapor pressure, as shown in [1]_.
        
        .. math::
            \Delta S_{P} = - \sum_i x_i\left(1 - \frac{V}{F}\right) 
            R\log\left(\frac{P_{sat, i}}{P_{ref}}\right) - \sum_i y_i\left(
            \frac{V}{F}\right) R\log\left(\frac{P}{P_{ref}}\right)
            
        These expressions are combined with the standard heat capacity and 
        enthalpy of vaporization expressions to calculate the total entropy:
        
        For a pure gas mixture:
            
        .. math::
             S = \Delta S_{mixing} + \Delta S_{P} + \sum_i z_i \cdot
             \int_{T_{ref}}^T \frac{C_{p}^{ig}(T)}{T} dT
                          
        For a pure liquid mixture:
            
        .. math::
             S = \Delta S_{mixing} + \Delta S_{P} + \sum_i z_i \left( 
             \int_{T_{ref}}^T \frac{C_{p}^{ig}(T)}{T} dT + \frac{H_{vap, i}
             (T)}{T} \right)
            
        For a vapor-liquid mixture:
            
        .. math::
             S = \Delta S_{mixing} + \Delta S_{P} + \sum_i z_i \cdot 
             \int_{T_{ref}}^T \frac{C_{p}^{ig}(T)}{T} dT + \sum_i x_i\left(1
             - \frac{V}{F}\right)\frac{H_{vap, i}(T)}{T}
             
        Returns
        -------
        S : float
            Entropy of the mixture with respect to the reference temperature,
            [J/mol/K]
            
        Notes
        -----
        The object must be flashed before this routine can be used. It 
        depends on the properties T, P, zs, V_over_F, HeatCapacityGases, 
        EnthalpyVaporizations, VaporPressures, and xs.
        
        References
        ----------
        .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
           New York: McGraw-Hill Professional, 2000.
        '''
        S = 0
        T = self.T
        P = self.P
        S -= R*sum([zi*log(zi) for zi in self.zs if zi > 0]) # ideal composition entropy composition; chemsep checked
        # Both of the mixing and vapor pressure terms have negative signs
        # Equation 6-4.4b in Poling for the vapor pressure component
        # For liquids above their critical temperatures, Psat is equal to the system P (COCO).
        if self.phase == 'g':
            S -= R*log(P/101325.) # Gas-phase ideal pressure contribution (checked repeatedly)
            for i in self.cmps:
                S += self.HeatCapacityGases[i].T_dependent_property_integral_over_T(298.15, T)
        elif self.phase == 'l':
            Psats = self._Psats(T)
            for i in self.cmps:
                Sg298_to_T = self.HeatCapacityGases[i].T_dependent_property_integral_over_T(298.15, T)
                Svap = -self.EnthalpyVaporizations[i](T)/T # Do the transition at the temperature of the liquid
                S_P = -R*log(Psats[i]/101325.)
                S += self.zs[i]*(Sg298_to_T + Svap + S_P)
        elif self.phase == 'l/g':
            Psats = self._Psats(T)
            S_P_vapor = -R*log(P/101325.) # Gas-phase ideal pressure contribution (checked repeatedly)
            for i in self.cmps:
                Sg298_to_T_zi = self.zs[i]*self.HeatCapacityGases[i].T_dependent_property_integral_over_T(298.15, T)
                Svap_contrib = -self.xs[i]*(1-self.V_over_F)*self.EnthalpyVaporizations[i](T)/T
                # Pressure contributions from both phases
                S_P_vapor_i = self.V_over_F*self.ys[i]*S_P_vapor
                S_P_liquid_i = -R*log(Psats[i]/101325.)*(1-self.V_over_F)*self.xs[i]
                S += (Sg298_to_T_zi + Svap_contrib + S_P_vapor_i + S_P_liquid_i)
        return S


class Activity_PP(Property_Package):
    __TP_cache = None
    __TVF_solve_cache = None
    retention = False
    use_Poynting = False
    use_phis = False

    def _P_VF_err(self, T, P, VF, zs):
        P_calc = self.flash_TVF_zs(T=T, VF=VF, zs=zs)[-1]
        return P_calc- P

    def _P_VF_err_2(self, T, P, VF, zs):
        V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs)
        if V_over_F < 0:
            if any(i < 0 for i in xs) or any(i < 0 for i in ys):
                return 5
            return -5
        if any(i < 0 for i in xs) or any(i < 0 for i in ys):
            return -5
        return V_over_F - VF

    def _T_VF_err(self, P, T, zs, Psats, Pmax, V_over_F_goal=1):
        if P < 0 or P > Pmax:
            return 1 
        V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs, Psats=Psats, restart=self.__TVF_solve_cache)
        if any(i < 0 for i in xs) or any(i < 0 for i in ys):
            return -100000*(Pmax-P)/Pmax
        self.__TVF_solve_cache = (V_over_F, xs, ys)
        ans = -(V_over_F-V_over_F_goal)
        return ans

    def Ks(self, T, P, xs, ys, Psats):
        gammas = self.gammas(T=T, xs=xs)
        if self.use_phis:
            phis_g = self.phis_g(T=T, P=P, ys=ys)
            phis_l = self.phis_l(T=T, P=P, xs=xs)
            if self.use_Poynting:
                Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
                return [K_value(P=P, Psat=Psats[i], gamma=gammas[i], 
                                phi_l=phis_l[i], phi_g=phis_g[i], Poynting=Poyntings[i]) for i in self.cmps]
            return [K_value(P=P, Psat=Psats[i], gamma=gammas[i], 
                            phi_l=phis_l[i], phi_g=phis_g[i]) for i in self.cmps]
        if self.use_Poynting:
            Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
            Ks = [K_value(P=P, Psat=Psats[i], gamma=gammas[i], Poynting=Poyntings[i]) for i in self.cmps]
            return Ks
        Ks = [K_value(P=P, Psat=Psats[i], gamma=gammas[i]) for i in self.cmps]
        return Ks
    
    def Poyntings(self, T, P, Psats):
        Vmls = [VolumeLiquid(T=T, P=P) for VolumeLiquid in self.VolumeLiquids]
        return [exp(Vml*(P-Psat)/(R*T)) for Psat, Vml in zip(Psats, Vmls)]

    def phis_g(self, T, P, ys):
        return self.eos_mix(T=T, P=P, zs=ys, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas).phis_g

    def phis_l(self, T, P, xs):
        P_sat_eos = [i.Psat(T) for i in self.eos_pure_instances]
        return [i.to_TP(T=T, P=Psat).phi_l for i, Psat in zip(self.eos_pure_instances, P_sat_eos)]    


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

    def _flash_sequential_substitution_TP(self, T, P, zs, Psats=None, restart=None):
        Psats = self._Psats(Psats=Psats, T=T)
        if self.retention and restart:
            V_over_F, xs, ys = restart
        else:
            Ks = self.Ks(T, P, zs, zs, Psats)
            V_over_F, xs, ys = flash_inner_loop(zs, Ks)
        for i in range(100):
            if any(i < 0 for i in xs):
                xs = zs
            if any(i < 0 for i in ys):
                ys = zs
            Ks = self.Ks(T, P, xs, ys, Psats)
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks)
            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
            xs, ys = xs_new, ys_new
            if err < 1E-7:
                break
        return V_over_F, xs, ys
    

    
    def gammas(self, T, xs):
        return [1 for i in self.N]
#        raise Exception(NotImplemented)

    def P_bubble_at_T(self, T, zs, Psats=None):
        # Returns P_bubble; only thing easy to calculate
        Psats = self._Psats(Psats, T)
        gammas = self.gammas(T=T, xs=zs)
        P = sum([gammas[i]*zs[i]*Psats[i] for i in self.cmps])
        if self.use_Poynting and not self.use_phis:
            # This is not really necessary; and 3 is more than enough iterations
            for i in range(3):
                Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
                P = sum([gammas[i]*zs[i]*Psats[i]*Poyntings[i] for i in self.cmps])
        elif self.use_phis:
            for i in range(5):
                phis_l = self.phis_l(T=T, P=P, xs=zs)
                if self.use_Poynting:
                    Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
                    P = sum([gammas[i]*zs[i]*Psats[i]*Poyntings[i]*phis_l[i] for i in self.cmps])
                else:
                    P = sum([gammas[i]*zs[i]*Psats[i]*phis_l[i] for i in self.cmps])
        # TODO: support equations of state, once you get that figured out.
        return P

    def P_dew_at_T(self, T, zs, Psats=None):
        Psats = self._Psats(Psats, T)
        Pmax = self.P_bubble_at_T(T, zs, Psats)
        diff = 1E-7
        # EOSs do not solve at very low pressure
        if self.use_phis:
            Pmin = max(Pmax*diff, 1)
        else:
            Pmin = Pmax*diff
        P_dew = brenth(self._T_VF_err, Pmin, Pmax, args=(T, zs, Psats, Pmax, 1))
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
            # EOSs do not solve at very low pressure
            if self.use_phis:
                Pmin = max(Pmax*diff, 1)
            else:
                Pmin = Pmax*diff

            P = brenth(self._T_VF_err, Pmin, Pmax, args=(T, zs, Psats, Pmax, VF))
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
        try:
            # In some caases, will find a false root - resort to iterations which
            # are always between Pdew and Pbubble if this happens
            T = brenth(self._P_VF_err_2, min(self.Tms), min(self.Tcs), args=(P, VF, zs), maxiter=500)
            V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs)
            assert abs(V_over_F-VF) < 1E-6
        except:
            T = ridder(self._P_VF_err, min(self.Tms), min(self.Tcs), args=(P, VF, zs))
            V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs)
        return 'l/g', xs, ys, V_over_F, T



class UNIFAC_PP(Activity_PP):
    subgroup_data = UFSG

    def __init__(self, UNIFAC_groups, VaporPressures, Tms=None, Tcs=None, Pcs=None,
                 omegas=None, VolumeLiquids=None, eos=None, eos_mix=None):
        self.UNIFAC_groups = UNIFAC_groups
        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.VolumeLiquids = VolumeLiquids
        self.omegas = omegas
        self.eos = eos
        self.eos_mix = eos_mix
        self.N = len(VaporPressures)
        self.cmps = range(self.N)

        if eos:
            self.eos_pure_instances = [eos(Tc=Tcs[i], Pc=Pcs[i], omega=omegas[i], T=Tcs[i]*0.5, P=Pcs[i]*0.1) for i in self.cmps]
        
        # Pre-calculate some of the inputs UNIFAC uses
        self.rs = []
        self.qs = []
        for groups in self.UNIFAC_groups:
            ri = 0.
            qi = 0.
            for group, count in groups.items():
                ri += self.subgroup_data[group].R*count
                qi += self.subgroup_data[group].Q*count
            self.rs.append(ri)
            self.qs.append(qi)
        

        self.group_counts = {}
        for groups in self.UNIFAC_groups:
            for group, count in groups.items():
                if group in self.group_counts:
                    self.group_counts[group] += count
                else:
                    self.group_counts[group] = count
        self.UNIFAC_cached_inputs = (self.rs, self.qs, self.group_counts)

    def gammas(self, T, xs, cached=None):
        return UNIFAC(chemgroups=self.UNIFAC_groups, T=T, xs=xs, cached=self.UNIFAC_cached_inputs)



class UNIFAC_Dortmund_PP(UNIFAC_PP):
    subgroup_data = DOUFSG

    def gammas(self, T, xs, cached=None):
        return UNIFAC(chemgroups=self.UNIFAC_groups, T=T, xs=xs, 
                      cached=self.UNIFAC_cached_inputs,
                      subgroup_data=DOUFSG, interaction_data=DOUFIP2006, modified=True)

