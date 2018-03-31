  # -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

.. contents:: :local:

Property Package Base Class
---------------------------
.. autoclass:: PropertyPackage
    :members:

Ideal Property Package
----------------------
.. autoclass:: Ideal
    :members:

.. autoclass:: IdealCaloric
    :members:
'''

from __future__ import division

__all__ = ['PropertyPackage', 'Ideal', 'Unifac', 'GammaPhi', 
           'UnifacDortmund', 'IdealCaloric', 'GammaPhiCaloric',
           'UnifacCaloric', 'UnifacDortmundCaloric', 'Nrtl']

from copy import copy
import numpy as np
from scipy.optimize import brenth, ridder, golden, brent
from scipy.misc import derivative

from thermo.utils import log, exp
from thermo.utils import has_matplotlib, R, pi, N_A
from thermo.utils import remove_zeros, normalize

from thermo.activity import K_value, flash_inner_loop, dew_at_T, bubble_at_T, NRTL
from thermo.unifac import UNIFAC, UFSG, DOUFSG, DOUFIP2006

if has_matplotlib:
    import matplotlib
    import matplotlib.pyplot as plt


class PropertyPackage(object):
    # Constant - if the phase fraction is this close to either the liquid or 
    # vapor phase, round it to it
    PHASE_ROUNDING_TOL = 1E-9 
    SUPPORTS_ZERO_FRACTIONS = True
    zero_fraction = 1E-6

    def to(self, zs, T=None, P=None, VF=None):
        obj = copy(self)
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
        if not self.SUPPORTS_ZERO_FRACTIONS:
            zs = remove_zeros(zs, 1e-11)
            
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
        
    def plot_Pxy(self, T, pts=30, display=True): # pragma: no cover
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
        if display:
            plt.show()
        else:
            return plt
        
    def plot_Txy(self, P, pts=30, display=True): # pragma: no cover
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
        if display:
            plt.show()
        else:
            return plt
        
    def plot_xy(self, P=None, T=None, pts=30, display=True): # pragma: no cover
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
                self.flash(T=T, VF=self.PHASE_ROUNDING_TOL*2, zs=[z1[i], z2[i]])
            elif P:
                self.flash(P=P, VF=self.PHASE_ROUNDING_TOL*2, zs=[z1[i], z2[i]])
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
        if display:
            plt.show()
        else:
            return plt
        
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
            zs = remove_zeros(zs, self.zero_fraction)
            self.flash(T=T, zs=zs, VF=0)
            P_values.append(self.P)
            return self.P
        
        def P_bubble_at_T_zs(zs):
            zs = remove_zeros(zs, 1e-6)
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

    def plot_TP_caloric(self, zs, Tmin=None, Tmax=None, Pmin=None, Pmax=None, 
                        pts=15, prop='Hm'):  # pragma: no cover
        if prop not in ['Sm', 'Gm', 'Hm']:
            raise Exception("The only supported property plots are enthalpy "
                            "('Hm'), entropy ('Sm'), and Gibbe energy ('Gm')")
        prop_name = {'Hm': 'enthalpy', 'Sm': 'entropy', 'Gm': 'Gibbs energy'}[prop]
        prop_units = {'Hm': 'J/mol', 'Sm': 'J/mol/K', 'Gm': 'J/mol'}[prop]

        if not has_matplotlib:
            raise Exception('Optional dependency matplotlib is required for plotting')
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib.ticker import FormatStrFormatter
        import numpy.ma as ma

        if Pmin is None:
            raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Pmax is None:
            raise Exception('Maximum pressure could not be auto-detected; please provide it')
        if Tmin is None:
            raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Tmax is None:
            raise Exception('Maximum pressure could not be auto-detected; please provide it')

        Ps = np.linspace(Pmin, Pmax, pts)
        Ts = np.linspace(Tmin, Tmax, pts)
        Ts_mesh, Ps_mesh = np.meshgrid(Ts, Ps)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        
        properties = []
        for T in Ts:
            properties2 = []
            for P in Ps:
                self.flash_caloric(zs=zs, T=T, P=P)
                properties2.append(getattr(self, prop))
            properties.append(properties2)
                
        ax.plot_surface(Ts_mesh, Ps_mesh, properties, cstride=1, rstride=1, alpha=0.5)
        
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.set_xlabel('Temperature, K')
        ax.set_ylabel('Pressure, Pa')
        ax.set_zlabel('%s, %s' %(prop_name, prop_units))
        plt.title('Temperature-pressure %s plot' %prop_name)
        plt.show(block=False)


class Ideal(PropertyPackage):
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
                Psat = i(T)
                if Psat is None:
                    Psat = i.extrapolate_tabular(T)
                Psats.append(Psat)
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
                
    def __init__(self, VaporPressures=None, Tms=None, Tcs=None, Pcs=None, 
                 **kwargs):
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
        elif 1.0 in zs:
            return 'l/g', list(zs), list(zs), VF, Psats[zs.index(1.0)]

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
        if self.N == 1:
            return 'l/g', [1.0], [1.0], VF, Tsats[0]
        elif 1.0 in zs:
            return 'l/g', list(zs), list(zs), VF, Tsats[zs.index(1.0)]

        T = brenth(self._P_VF_err, min(Tsats)*(1+1E-7), max(Tsats)*(1-1E-7), args=(P, VF, zs))
        Psats = self._Psats(T)
        Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
        V_over_F, xs, ys = flash_inner_loop(zs=zs, Ks=Ks)
        return 'l/g', xs, ys, V_over_F, T


class IdealCaloric(Ideal):
    T_REF_IG = 298.15
    P_REF_IG = 101325.
    P_DEPENDENT_H_LIQ = True
    
    def __init__(self, VaporPressures=None, Tms=None, Tbs=None, Tcs=None, Pcs=None, 
                 HeatCapacityLiquids=None, HeatCapacityGases=None,
                 EnthalpyVaporizations=None, VolumeLiquids=None, **kwargs):
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
        self.VolumeLiquids = VolumeLiquids
        
        self.kwargs = {'VaporPressures': VaporPressures,
                       'Tms': Tms, 'Tbs': Tbs, 'Tcs': Tcs, 'Pcs': Pcs,
                       'HeatCapacityLiquids': HeatCapacityLiquids, 
                       'HeatCapacityGases': HeatCapacityGases,
                       'EnthalpyVaporizations': EnthalpyVaporizations, 
                       'VolumeLiquids': VolumeLiquids}
        
    def flash_caloric(self, zs, T=None, P=None, VF=None, Hm=None, Sm=None):
        if not self.SUPPORTS_ZERO_FRACTIONS:
            zs = remove_zeros(zs, self.zero_fraction)
            
        try:
            if T is not None and Sm is not None:
                P = self.flash_TS_zs_bounded(T=T, Sm=Sm, zs=zs)
            elif P is not None and Sm is not None:
                T = self.flash_PS_zs_bounded(P=P, Sm=Sm, zs=zs)
            elif P is not None and Hm is not None:
                T = self.flash_PH_zs_bounded(P=P, Hm=Hm, zs=zs)
            elif ((T is not None and P is not None) or
                (T is not None and VF is not None) or
                (P is not None and VF is not None)):
                pass
            else:
                raise Exception('Flash inputs unsupported')

            self.flash(zs=zs, T=T, P=P, VF=VF)
            self._post_flash()
            self.status = True
        except Exception as e:
            # Write Nones for everything here
            self.status = e
            self._set_failure()
            
    def _set_failure(self):
        self.Hm = None
        self.Sm = None
        self.Gm = None
        self.chemical_potential = None
        self.T = None
        self.P = None
        self.phase = None
        self.V_over_F = None
        self.xs = None
        self.ys = None
            


    def _post_flash(self):
        # Cannot derive other properties with this
        self.Hm = self.enthalpy_Cpg_Hvap()
        self.Sm = self.entropy_Cpg_Hvap()
        self.Gm = self.Hm - self.T*self.Sm if (self.Hm is not None and self.Sm is not None) else None

    def partial_property(self, T, P, i, zs, prop='Hm'):
        r'''Method to calculate the partial molar property for entropy,
        enthalpy, or gibbs energy. Note the partial gibbs energy is known
        as chemical potential as well.
        
        .. math::
            \bar m_i = \left( \frac{\partial (n_T m)} {\partial n_i}
            \right)_{T, P, n_{j\ne i}}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the partial property, [K]
        P : float
            Pressure at which to calculate the partial property, [Pa]
        i : int
            Compound index, [-]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]

        Returns
        -------
        partial_prop : float
            Calculated partial property, [`units`]
        '''
        if prop not in ('Sm', 'Gm', 'Hm'):
            raise Exception("The only supported property plots are enthalpy "
                            "('Hm'), entropy ('Sm'), and Gibbe energy ('Gm')")
        
        def prop_extensive(ni, ns, i):
            ns[i] = ni
            n_tot = sum(ns)
            zs = normalize(ns)
            obj = self.to(T=T, P=P, zs=zs)
            obj.flash_caloric(T=T, P=P, zs=zs)
            property_value = getattr(obj, prop)
            return property_value*n_tot
        return derivative(prop_extensive, zs[i], dx=1E-6, args=[list(zs), i])



    def enthalpy_Cpg_Hvap(self):
        r'''Method to calculate the enthalpy of an ideal mixture. This routine
        is based on "route A", where the gas heat
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
                 
        For liquids, the enthalpy contribution of pressure is:
        
        .. math::
            \Delta H = \sum_i z_i (P - P_{sat, i}) V_{m, i}

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
        P = self.P
        if self.phase == 'g':
            for i in self.cmps:
                H += self.zs[i]*self.HeatCapacityGases[i].T_dependent_property_integral(self.T_REF_IG, T)
        elif self.phase == 'l':
            Psats = self._Psats(T=T)
            for i in self.cmps:
                # No further contribution needed
                Hg298_to_T = self.HeatCapacityGases[i].T_dependent_property_integral(self.T_REF_IG, T)
                Hvap = self.EnthalpyVaporizations[i](T) # Do the transition at the temperature of the liquid
                if Hvap is None:
                    Hvap = 0 # Handle the case of a package predicting a transition past the Tc
                H_i = Hg298_to_T - Hvap 
                if self.P_DEPENDENT_H_LIQ:
                    Vl = self.VolumeLiquids[i](T, P)
                    if Vl is None:
                        # Handle an inability to get a liquid volume by taking
                        # one at the boiling point (and system P)
                        Vl = self.VolumeLiquids[i](self.Tbs[i], P)
                    H_i += (P - Psats[i])*Vl
                H += self.zs[i]*(H_i) 
        elif self.phase == 'l/g':
            for i in self.cmps:
                Hg298_to_T_zi = self.zs[i]*self.HeatCapacityGases[i].T_dependent_property_integral(self.T_REF_IG, T)
                Hvap = self.EnthalpyVaporizations[i](T) 
                if Hvap is None:
                    Hvap = 0 # Handle the case of a package predicting a transition past the Tc
                Hvap_contrib = -self.xs[i]*(1-self.V_over_F)*Hvap
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
            Psats = self._Psats(T=T)
            for i in self.cmps:
                Sg298_to_T = self.HeatCapacityGases[i].T_dependent_property_integral_over_T(298.15, T)
                Hvap = self.EnthalpyVaporizations[i](T)
                if Hvap is None:
                    Hvap = 0 # Handle the case of a package predicting a transition past the Tc
                Svap = -Hvap/T # Do the transition at the temperature of the liquid
                S_P = -R*log(Psats[i]/101325.)
                S += self.zs[i]*(Sg298_to_T + Svap + S_P)
        elif self.phase == 'l/g':
            Psats = self._Psats(T=T)
            S_P_vapor = -R*log(P/101325.) # Gas-phase ideal pressure contribution (checked repeatedly)
            for i in self.cmps:
                Sg298_to_T_zi = self.zs[i]*self.HeatCapacityGases[i].T_dependent_property_integral_over_T(298.15, T)
                Hvap = self.EnthalpyVaporizations[i](T) 
                if Hvap is None:
                    Hvap = 0 # Handle the case of a package predicting a transition past the Tc

                Svap_contrib = -self.xs[i]*(1-self.V_over_F)*Hvap/T
                # Pressure contributions from both phases
                S_P_vapor_i = self.V_over_F*self.ys[i]*S_P_vapor
                S_P_liquid_i = -R*log(Psats[i]/101325.)*(1-self.V_over_F)*self.xs[i]
                S += (Sg298_to_T_zi + Svap_contrib + S_P_vapor_i + S_P_liquid_i)
        return S

    def flash_PH_zs_bounded(self, P, Hm, zs, T_low=None, T_high=None, 
                            Hm_low=None, Hm_high=None):
        # Begin the search at half the lowest chemical's melting point
        if T_low is None:
            T_low = min(self.Tms)/2 
                
        # Cap the T high search at 8x the highest critical point
        # (will not work well for helium, etc.)
        if T_high is None:
            max_Tc = max(self.Tcs)
            if max_Tc < 100:
                T_high = 4000
            else:
                T_high = max_Tc*8
    
        temp_pkg_cache = []
        def PH_error(T, P, zs, H_goal):
            if not temp_pkg_cache:
                temp_pkg = self.to(T=T, P=P, zs=zs)
                temp_pkg_cache.append(temp_pkg)
            else:
                temp_pkg = temp_pkg_cache[0]
                temp_pkg.flash(T=T, P=P, zs=zs)
            temp_pkg._post_flash()
            return temp_pkg.Hm - H_goal
    
        try:
            T_goal = brenth(PH_error, T_low, T_high, args=(P, zs, Hm))
            return T_goal

        except ValueError:
            if Hm_low is None:
                pkg_low = self.to(T=T_low, P=P, zs=zs)
                pkg_low._post_flash()
                Hm_low = pkg_low.Hm
            if Hm < Hm_low:
                raise ValueError('The requested molar enthalpy cannot be found'
                                 ' with this bounded solver because the lower '
                                 'temperature bound %g K has an enthalpy (%g '
                                 'J/mol) higher than that requested (%g J/mol)' %(
                                                             T_low, Hm_low, Hm))
            if Hm_high is None:
                pkg_high = self.to(T=T_high, P=P, zs=zs)
                pkg_high._post_flash()
                Hm_high = pkg_high.Hm
            if Hm > Hm_high:
                raise ValueError('The requested molar enthalpy cannot be found'
                                 ' with this bounded solver because the upper '
                                 'temperature bound %g K has an enthalpy (%g '
                                 'J/mol) lower than that requested (%g J/mol)' %(
                                                             T_high, Hm_high, Hm))


    def flash_PS_zs_bounded(self, P, Sm, zs, T_low=None, T_high=None, 
                            Sm_low=None, Sm_high=None):
        # Begin the search at half the lowest chemical's melting point
        if T_low is None:
            T_low = min(self.Tms)/2 
                
        # Cap the T high search at 8x the highest critical point
        # (will not work well for helium, etc.)
        if T_high is None:
            max_Tc = max(self.Tcs)
            if max_Tc < 100:
                T_high = 4000
            else:
                T_high = max_Tc*8
    
        temp_pkg_cache = []
        def PS_error(T, P, zs, S_goal):
            if not temp_pkg_cache:
                temp_pkg = self.to(T=T, P=P, zs=zs)
                temp_pkg_cache.append(temp_pkg)
            else:
                temp_pkg = temp_pkg_cache[0]
                temp_pkg.flash(T=T, P=P, zs=zs)
            temp_pkg._post_flash()
            return temp_pkg.Sm - S_goal
    
        try:
            T_goal = brenth(PS_error, T_low, T_high, args=(P, zs, Sm))
            return T_goal

        except ValueError:
            if Sm_low is None:
                pkg_low = self.to(T=T_low, P=P, zs=zs)
                pkg_low._post_flash()
                Sm_low = pkg_low.Sm
            if Sm < Sm_low:
                raise ValueError('The requested molar entropy cannot be found'
                                 ' with this bounded solver because the lower '
                                 'temperature bound %g K has an entropy (%g '
                                 'J/mol/K) higher than that requested (%g J/mol/K)' %(
                                                             T_low, Sm_low, Sm))
            if Sm_high is None:
                pkg_high = self.to(T=T_high, P=P, zs=zs)
                pkg_high._post_flash()
                Sm_high = pkg_high.Sm
            if Sm > Sm_high:
                raise ValueError('The requested molar entropy cannot be found'
                                 ' with this bounded solver because the upper '
                                 'temperature bound %g K has an entropy (%g '
                                 'J/mol/K) lower than that requested (%g J/mol/K)' %(
                                                             T_high, Sm_high, Sm))

    def flash_TS_zs_bounded(self, T, Sm, zs, P_low=None, P_high=None, 
                            Sm_low=None, Sm_high=None):
        # Begin the search at half the lowest chemical's melting point
        if P_high is None:
            P_high = self.Pbubble(T, zs)
        if P_low is None:
            P_low = 1E-5 # min pressure
#            P_high = max(self.Pcs)*10

        temp_pkg_cache = []
        def TS_error(P, T, zs, S_goal):
            if not temp_pkg_cache:
                temp_pkg = self.to(T=T, P=P, zs=zs)
                temp_pkg_cache.append(temp_pkg)
            else:
                temp_pkg = temp_pkg_cache[0]
                temp_pkg.flash(T=T, P=P, zs=zs)
            temp_pkg._post_flash()
            return temp_pkg.Sm - S_goal
    
        try:
            P_goal = brenth(TS_error, P_low, P_high, args=(T, zs, Sm))
            return P_goal

        except ValueError:
            if Sm_low is None:
                pkg_low = self.to(T=T, P=P_low, zs=zs)
                pkg_low._post_flash()
                Sm_low = pkg_low.Sm
            if Sm > Sm_low:
                raise ValueError('The requested molar entropy cannot be found'
                                 ' with this bounded solver because the lower '
                                 'pressure bound %g Pa has an entropy (%g '
                                 'J/mol/K) lower than that requested (%g J/mol/K)' %(
                                                             P_low, Sm_low, Sm))
            if Sm_high is None:
                pkg_high = self.to(T=T, P=P_high, zs=zs)
                pkg_high._post_flash()
                Sm_high = pkg_high.Sm
            if Sm < Sm_high:
                raise ValueError('The requested molar entropy cannot be found'
                                 ' with this bounded solver because the upper '
                                 'pressure bound %g Pa has an entropy (%g '
                                 'J/mol/K) upper than that requested (%g J/mol/K)' %(
                                                             P_high, Sm_high, Sm))


class GammaPhi(PropertyPackage):
    __TP_cache = None
    __TVF_solve_cache = None
    retention = False
    use_Poynting = False
    use_phis = False
    SUPPORTS_ZERO_FRACTIONS = False

    def __init__(self, VaporPressures=None, Tms=None, Tcs=None, Pcs=None, 
                 **kwargs):
        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.N = len(VaporPressures)
        self.cmps = range(self.N)
        
        self.kwargs = {'VaporPressures': VaporPressures,
                       'Tms': Tms, 'Tcs': Tcs, 'Pcs': Pcs}

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
#        return [1 for i in xs] # Most models seem to assume this
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
        return [1 for i in self.cmps]

    def VE_l(self):
        r'''Calculates the excess volume of a liquid phase using an
        activity coefficient model as shown in [1]_ and [2]_.
            
        .. math::
            v^E = \left(\frac{\partial g^E}{\partial P}\right)_{T, xi, xj...} 
            
        In practice, this returns 0 as no pressure-dependent activity models
        are available.
            
        Returns
        -------
        VE : float
            Excess volume of the liquid phase (0), [m^3/mol]
    
        Notes
        -----
        The relationship for partial excess molar volume is as follows:
            
        .. math::
            \frac{\bar v_i^E}{RT} = \left(\frac{\partial \ln \gamma_i}
            {\partial P}\right)_T
            
            
        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process 
           Simulation. Weinheim, Germany: Wiley-VCH, 2012.
        '''
        return 0.0
    
    def GE_l(self, T, xs):
        r'''Calculates the excess Gibbs energy of a liquid phase using an
        activity coefficient model as shown in [1]_ and [2]_.
            
        .. math::
            g_E = RT\sum_i x_i \ln \gamma_i
            
        Parameters
        ----------
        T : float
            Temperature of the system, [K]
        xs : list[float]
            Mole fractions of the liquid phase of the system, [-]
    
        Returns
        -------
        GE : float
            Excess Gibbs energy of the liquid phase, [J/mol]
    
        Notes
        -----
        It is possible to directly calculate GE in some activity coefficient 
        models, without calculating individual activity coefficients of the
        species.
        
        Note also the relationship of the expressions for partial excess Gibbs 
        energies:
            
        .. math::
            \bar g_i^E = RT\ln(\gamma_i)
        
            g^E = \sum_i x_i \bar g_i^E
            
        Most activity coefficient models are pressure independent, which leads
        to the relationship where excess Helmholtz energy is the same as the
        excess Gibbs energy.
        
        .. math::
            G^E = A^E
            
        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process 
           Simulation. Weinheim, Germany: Wiley-VCH, 2012.
        '''
        gammas = self.gammas(T=T, xs=xs)
        return R*T*sum(xi*log(gamma) for xi, gamma in zip(xs, gammas))

    def HE_l(self, T, xs):
        r'''Calculates the excess enthalpy of a liquid phase using an
        activity coefficient model as shown in [1]_ and [2]_. This is an 
        expression of the Gibbs-Helmholz relation.
            
        .. math::
            \frac{-h^E}{T^2} = \frac{\partial (g^E/T)}{\partial T}
            
        Parameters
        ----------
        T : float
            Temperature of the system, [K]
        xs : list[float]
            Mole fractions of the liquid phase of the system, [-]
    
        Returns
        -------
        HE : float
            Excess enthalpy of the liquid phase, [J/mol]
    
        Notes
        -----
        It is possible to obtain analytical results for some activity 
        coefficient models; this method provides only the `derivative`
        method of scipy with its default parameters to obtain a numerical
        result.
        
        Note also the relationship of the expressions for partial excess 
        enthalpy:
            
        .. math::
            \left(\frac{\partial \ln \gamma_i}{\partial (1/T))}\right) 
                = \frac{\bar h_i^E}{R}    
                
            \left(\frac{\partial \ln \gamma_i}{\partial T}\right)
            = -\frac{\bar h_i^E}{RT^2}

            
        Most activity coefficient models are pressure independent, so the Gibbs
        Duhem expression only has a temperature relevance.
        
        .. math::
            \sum_i x_i d \ln \gamma_i = - \frac{H^{E}}{RT^2} dT 
            + \frac{V^E}{RT} dP
            
        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process 
           Simulation. Weinheim, Germany: Wiley-VCH, 2012.
        '''
        to_diff = lambda T: self.GE_l(T, xs)/T
        return -derivative(to_diff, T)*T**2
    
    def SE_l(self, T, xs):
        r'''Calculates the excess entropy of a liquid phase using an
        activity coefficient model as shown in [1]_ and [2]_.
            
        .. math::
            s^E = \frac{h^E - g^E }{T} 
            
        Parameters
        ----------
        T : float
            Temperature of the system, [K]
        xs : list[float]
            Mole fractions of the liquid phase of the system, [-]
    
        Returns
        -------
        SE : float
            Excess entropy of the liquid phase, [J/mol/K]
    
        Notes
        -----
        It is possible to obtain analytical results for some activity 
        coefficient models; this method provides only the `derivative`
        method of scipy with its default parameters to obtain a numerical
        result for the excess enthalpy, although the excess Gibbs energy
        is exact.
        
        Note also the relationship of the expressions for partial excess 
        entropy: 
            
        .. math::
            S_i^E = -R\left(T \frac{\partial \ln \gamma_i}{\partial T}
            + \ln \gamma_i\right)

            
        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process 
           Simulation. Weinheim, Germany: Wiley-VCH, 2012.
        '''
        return (self.HE_l(T, xs) - self.GE_l(T, xs))/T
    
    def CpE_l(self, T, xs):
        r'''Calculates the excess heat capacity of a liquid phase using an
        activity coefficient model as shown in [1]_ and [2]_.
            
        .. math::
             C_{p,l}^E = \left(\frac{\partial H^E}{\partial T}\right)_{p, x}
            
        Parameters
        ----------
        T : float
            Temperature of the system, [K]
        xs : list[float]
            Mole fractions of the liquid phase of the system, [-]
    
        Returns
        -------
        CpE : float
            Excess heat capacity of the liquid phase, [J/mol/K]
    
        Notes
        -----
        This method provides only the `derivative`
        method of scipy with its default parameters to obtain a numerical
        result for the excess enthalpy as well as the derivative of excess
        enthalpy.
                    
        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering. 
           Butterworth-Heinemann, 1985.
        .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process 
           Simulation. Weinheim, Germany: Wiley-VCH, 2012.
        '''
        to_diff = lambda T : self.HE_l(T, xs)
        return derivative(to_diff, T)
    
    def gammas_infinite_dilution(self, T):
        gamma_infs = []
        for i in self.cmps:
            xs = [1./(self.N - 1)]*self.N
            xs[i] = 0
            gamma_inf = self.gammas(T=T, xs=xs)[i]
            gamma_infs.append(gamma_inf)
        return gamma_infs
    
    def H_dep_g(self, T, P, ys):
        if not self.use_phis:
            return 0.0
        e = self.eos_mix(T=T, P=P, zs=ys, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas)
        try:
            return e.H_dep_g
        except AttributeError:
            # This really is the correct approach
            return e.H_dep_l

    
    def S_dep_g(self, T, P, ys):
        if not self.use_phis:
            return 0.0
        e = self.eos_mix(T=T, P=P, zs=ys, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas)
        try:
            return e.S_dep_g
        except AttributeError:
            # This really is the correct approach
            return e.S_dep_l
    
    def enthalpy_excess(self, T, P, V_over_F, xs, ys):
        # Does this handle the transition without a discontinuity?
        H = 0
        if self.phase == 'g':
            H += self.H_dep_g(T=T, P=P, ys=ys)
        elif self.phase == 'l':
            H += self.HE_l(T=T, xs=xs)
        elif self.phase == 'l/g':
            HE_l = self.HE_l(T=T, xs=xs)
            HE_g = self.H_dep_g(T=T, P=P, ys=ys)
            H += (1. - V_over_F)*HE_l + HE_g*V_over_F
        return H


    def entropy_excess(self, T, P, V_over_F, xs, ys):
        # Does this handle the transition without a discontinuity?
        S = 0
        if self.phase == 'g':
            S += self.S_dep_g(T=T, P=P, ys=ys)
        elif self.phase == 'l':
            S += self.SE_l(T=T, xs=xs)
        elif self.phase == 'l/g':
            SE_l = self.SE_l(T=T, xs=xs)
            SE_g = self.S_dep_g(T=T, P=P, ys=ys)
            S += (1. - V_over_F)*SE_l + SE_g*V_over_F
        return S


    def P_bubble_at_T(self, T, zs, Psats=None):
        # Returns P_bubble; only thing easy to calculate
        Psats = self._Psats(Psats, T)
        # If there is one component, return at the saturation line
        if self.N == 1:
            return Psats[0]
        
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
        
        # If there is one component, return at the saturation line
        if self.N == 1:
            return Psats[0]
        
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
        
        # handle one component
        if self.N == 1:
            return 'l/g', [1.0], [1.0], VF, Psats[0]
        
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
        if self.N == 1:
            Tsats = self._Tsats(P)
            return 'l/g', [1.0], [1.0], VF, Tsats[0]
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


class GammaPhiCaloric(GammaPhi, IdealCaloric):
    
    def _post_flash(self):
        # Cannot derive other properties with this
        self.Hm = self.enthalpy_Cpg_Hvap() + self.enthalpy_excess(T=self.T, P=self.P, V_over_F=self.V_over_F, xs=self.xs, ys=self.ys)
        self.Sm = self.entropy_Cpg_Hvap() + self.entropy_excess(T=self.T, P=self.P, V_over_F=self.V_over_F, xs=self.xs, ys=self.ys)
        
        self.Gm = self.Hm - self.T*self.Sm if (self.Hm is not None and self.Sm is not None) else None



    
    def __init__(self, VaporPressures=None, Tms=None, Tbs=None, Tcs=None, 
                 Pcs=None, omegas=None, VolumeLiquids=None, eos=None, 
                 eos_mix=None, HeatCapacityLiquids=None, HeatCapacityGases=None,
                 EnthalpyVaporizations=None, **kwargs):
        self.cmps = range(len(VaporPressures))
        self.N = len(VaporPressures)
        self.VaporPressures = VaporPressures
        self.omegas = omegas
        
        self.Tms = Tms
        self.Tbs = Tbs
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.HeatCapacityLiquids = HeatCapacityLiquids
        self.HeatCapacityGases = HeatCapacityGases
        self.EnthalpyVaporizations = EnthalpyVaporizations
        self.VolumeLiquids = VolumeLiquids
        self.eos = eos
        self.eos_mix = eos_mix
        
        if eos:
            self.eos_pure_instances = [eos(Tc=Tcs[i], Pc=Pcs[i], omega=omegas[i], T=Tcs[i]*0.5, P=Pcs[i]*0.1) for i in self.cmps]
        
        self.kwargs = {'VaporPressures': VaporPressures,
                       'Tms': Tms, 'Tbs': Tbs, 'Tcs': Tcs, 'Pcs': Pcs,
                       'HeatCapacityLiquids': HeatCapacityLiquids, 
                       'HeatCapacityGases': HeatCapacityGases,
                       'EnthalpyVaporizations': EnthalpyVaporizations,
                       'VolumeLiquids': VolumeLiquids,
                       'eos': eos, 'eos_mix': eos_mix,
                       'omegas': omegas}


class Nrtl(GammaPhi):
    def taus(self, T):
        # initialize the matrix to be zero
        taus = [[0.0]*self.N for i in self.cmps]
        T2 = T*T
        logT = log(T)
        for i in self.cmps:
            for j in range(self.N - i):
                if i == j:
                    tau = 0.0
                else:
                    coeffs = self.tau_coeffs[i][j]
                    tau = coeffs[0] + coeffs[1]/T + coeffs[2]*logT + coeffs[3]*T + coeffs[4]/T2  + coeffs[5]*T2
                taus[i][j] = tau #  = taus[j][i]
        return taus
    
    def alphas(self, T):
        alphas = [[0.0]*self.N for i in self.cmps]
        for i in self.cmps:
            for j in range(self.N - i):
                if i == j:
                    alpha = 0.0
                else:
                    c, d = self.alpha_coeffs[i][j]
                    alpha = c + d*T
                alphas[i][j] = alpha #  = alphas[j][i]
        return alphas
    
    
    def __init__(self, tau_coeffs, alpha_coeffs, VaporPressures, Tms=None,
                 Tcs=None, Pcs=None, omegas=None, VolumeLiquids=None, eos=None,
                 eos_mix=None):
        self.tau_coeffs = tau_coeffs
        self.alpha_coeffs = alpha_coeffs
        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.VolumeLiquids = VolumeLiquids
        self.eos = eos
        self.eos_mix = eos_mix
        self.N = len(VaporPressures)
        self.cmps = range(self.N)

        if eos:
            self.eos_pure_instances = [eos(Tc=Tcs[i], Pc=Pcs[i], omega=omegas[i], T=Tcs[i]*0.5, P=Pcs[i]*0.1) for i in self.cmps]

    def gammas(self, T, xs, cached=None):
        alphas = self.alphas(T)
        taus = self.taus(T)
        return NRTL(xs=xs, taus=taus, alphas=alphas)
    
    
class Unifac(GammaPhi):
    subgroup_data = UFSG

    def __init__(self, UNIFAC_groups, VaporPressures, Tms=None, Tcs=None, Pcs=None,
                 omegas=None, VolumeLiquids=None, eos=None, eos_mix=None, **kwargs):
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
        
        self.cache_unifac_inputs()
        
    def cache_unifac_inputs(self):
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



class UnifacDortmund(Unifac):
    subgroup_data = DOUFSG

    def gammas(self, T, xs, cached=None):
        return UNIFAC(chemgroups=self.UNIFAC_groups, T=T, xs=xs, 
                      cached=self.UNIFAC_cached_inputs,
                      subgroup_data=DOUFSG, interaction_data=DOUFIP2006, modified=True)


class UnifacCaloric(Unifac, GammaPhiCaloric):
        
    def __init__(self, VaporPressures, eos=None, **kwargs):
        self.cmps = range(len(VaporPressures))
        self.N = len(VaporPressures)
        self.VaporPressures = VaporPressures
        self.eos = eos
        self.__dict__.update(kwargs)
        self.kwargs = {'VaporPressures': VaporPressures, 'eos': eos}
        self.kwargs.update(kwargs)
        
        if eos:
            self.eos_pure_instances = [eos(Tc=self.Tcs[i], Pc=self.Pcs[i], omega=self.omegas[i], T=self.Tcs[i]*0.5, P=self.Pcs[i]*0.1) for i in self.cmps]
        
        self.cache_unifac_inputs()

class UnifacDortmundCaloric(UnifacDortmund, GammaPhiCaloric):
        
    def __init__(self, VaporPressures, eos=None, **kwargs):
        self.cmps = range(len(VaporPressures))
        self.N = len(VaporPressures)
        self.VaporPressures = VaporPressures
        self.eos = eos
        self.__dict__.update(kwargs)
        self.kwargs = {'VaporPressures': VaporPressures, 'eos': eos}
        self.kwargs.update(kwargs)
        
        if eos:
            self.eos_pure_instances = [eos(Tc=self.Tcs[i], Pc=self.Pcs[i], omega=self.omegas[i], T=self.Tcs[i]*0.5, P=self.Pcs[i]*0.1) for i in self.cmps]
        
        self.cache_unifac_inputs()
