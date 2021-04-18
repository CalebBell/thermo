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
__all__ = ['VirialCorrelationsPitzerCurl', 'VirialGas']

from fluids.numerics import newton
from chemicals.utils import log 
from thermo.heat_capacity import HeatCapacityGas
from .phase import Phase

from chemicals.virial import BVirial_Pitzer_Curl, Z_from_virial_density_form
class VirialCorrelationsPitzerCurl(object):

    def __init__(self, Tcs, Pcs, omegas):
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.N = len(Tcs)

    def C_pures(self, T):
        return [0.0]*self.N

    def dC_dT_pures(self, T):
        return [0.0]*self.N

    def d2C_dT2_pures(self, T):
        return [0.0]*self.N

    def C_interactions(self, T):
        N = self.N
        Ciij = [[0.0]*N for i in range(N)]
        Cijj = [[0.0]*N for i in range(N)]

#        Full return should be (Ciij, Ciji, Cjii), (Cijj, Cjij, Cjji)
#        but due to symmetry there is only those two matrices
        return Ciij, Cijj

    def dC_dT_interactions(self, T):
        N = self.N
        Ciij = [[0.0]*N for i in range(N)]
        Cijj = [[0.0]*N for i in range(N)]
        return Ciij, Cijj

    def d2C_dT2_interactions(self, T):
        N = self.N
        Ciij = [[0.0]*N for i in range(N)]
        Cijj = [[0.0]*N for i in range(N)]
        return Ciij, Cijj

    def B_pures(self, T):
        Tcs, Pcs, omegas = self.Tcs, self.Pcs, self.omegas
        return [BVirial_Pitzer_Curl(T, Tcs[i], Pcs[i], omegas[i]) for i in range(self.N)]

    def dB_dT_pures(self, T):
        Tcs, Pcs, omegas = self.Tcs, self.Pcs, self.omegas
        return [BVirial_Pitzer_Curl(T, Tcs[i], Pcs[i], omegas[i], 1) for i in range(self.N)]

    def B_interactions(self, T):
        N = self.N
        return [[0.0]*N for i in range(N)]

    def dB_dT_interactions(self, T):
        N = self.N
        return [[0.0]*N for i in range(N)]

    def B_matrix(self, T):
        N = self.N
        B_mat = [[0.0]*N for i in range(N)]
        pures = self.B_pures(T)
        B_interactions = self.B_interactions(T)
        for i in range(N):
            B_mat[i][i] = pures[i]
        for i in range(N):
            for j in range(i):
                B_mat[i][j] = B_interactions[i][j]
                B_mat[j][i] = B_interactions[j][i]

        return B_mat

    def dB_dT_matrix(self, T):
        N = self.N
        B_mat = [[0.0]*N for i in range(N)]
        pures = self.dB_dT_pures(T)
        B_interactions = self.dB_dT_interactions(T)
        for i in range(N):
            B_mat[i][i] = pures[i]
        for i in range(N):
            for j in range(i):
                B_mat[i][j] = B_interactions[i][j]
                B_mat[j][i] = B_interactions[j][i]

        return B_mat


    def d2B_dT2_pures(self, T):
        Tcs, Pcs, omegas = self.Tcs, self.Pcs, self.omegas
        return [BVirial_Pitzer_Curl(T, Tcs[i], Pcs[i], omegas[i], 2) for i in range(self.N)]
    def d2B_dT2_interactions(self, T):
        N = self.N
        return [[0.0]*N for i in range(N)]

    def d2B_dT2_matrix(self, T):
        N = self.N
        B_mat = [[0.0]*N for i in range(N)]
        pures = self.d2B_dT2_pures(T)
        B_interactions = self.d2B_dT2_interactions(T)
        for i in range(N):
            B_mat[i][i] = pures[i]
        for i in range(N):
            for j in range(i):
                B_mat[i][j] = B_interactions[i][j]
                B_mat[j][i] = B_interactions[j][i]

        return B_mat


class VirialGas(Phase):
    phase = 'g'
    force_phase = 'g'
    is_gas = True
    is_liquid = False
    ideal_gas_basis = True
    pure_references = ('HeatCapacityGases',)
    pure_reference_types = (HeatCapacityGas, )

    def __init__(self, model, HeatCapacityGases=None, Hfs=None, Gfs=None, 
                 T=None, P=None, zs=None):
        self.model = model
        self.HeatCapacityGases = HeatCapacityGases
        self.Hfs = Hfs
        self.Gfs = Gfs
        if Hfs is not None and Gfs is not None and None not in Hfs and None not in Gfs:
            self.Sfs = [(Hfi - Gfi)/298.15 for Hfi, Gfi in zip(Hfs, Gfs)]
        else:
            self.Sfs = None

        for i in (zs, HeatCapacityGases, Hfs, Gfs):
            if i is not None:
                self.N = len(i)
                break
        if zs is not None:
            self.zs = zs
        if T is not None:
            self.T = T
        if P is not None:
            self.P = P
        if T is not None and P is not None and zs is not None:
            Z = Z_from_virial_density_form(T, P, self.B(), self.C())
            self._V = Z*self.R*T/P

    def V(self):
        return self._V

    def dP_dT(self):
        r'''

        .. math::
            \left(\frac{\partial P}{\partial T}\right)_{V} = \frac{R \left(T
            \left(V \frac{d}{d T} B{\left(T \right)} + \frac{d}{d T} C{\left(T
            \right)}\right) + V^{2} + V B{\left(T \right)} + C{\left(T \right)}
            \right)}{V^{3}}

        '''
        try:
            return self._dP_dT
        except:
            pass
        T, V = self.T, self._V
        self._dP_dT = dP_dT = self.R*(T*(V*self.dB_dT() + self.dC_dT()) + V*(V + self.B()) + self.C())/(V*V*V)
        return dP_dT

    def dP_dV(self):
        r'''

        .. math::
            \left(\frac{\partial P}{\partial V}\right)_{T} =
            - \frac{R T \left(V^{2} + 2 V B{\left(T \right)} + 3 C{\left(T
            \right)}\right)}{V^{4}}

        '''
        try:
            return self._dP_dV
        except:
            pass
        T, V = self.T, self._V
        self._dP_dV = dP_dV = -self.R*T*(V*V + 2.0*V*self.B() + 3.0*self.C())/(V*V*V*V)
        return dP_dV

    def d2P_dTdV(self):
        r'''

        .. math::
            \left(\frac{\partial^2 P}{\partial V\partial T}\right)_{T} =
            - \frac{R \left(2 T V \frac{d}{d T} B{\left(T \right)} + 3 T
            \frac{d}{d T} C{\left(T \right)} + V^{2} + 2 V B{\left(T \right)}
            + 3 C{\left(T \right)}\right)}{V^{4}}

        '''
        try:
            return self._d2P_dTdV
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V
        self._d2P_dTdV = d2P_dTdV = -self.R*(2.0*T*V*self.dB_dT() + 3.0*T*self.dC_dT()
        + V2 + 2.0*V*self.B() + 3.0*self.C())/(V2*V2)

        return d2P_dTdV

    def d2P_dV2(self):
        r'''

        .. math::
            \left(\frac{\partial^2 P}{\partial V^2}\right)_{T} =
            \frac{2 R T \left(V^{2} + 3 V B{\left(T \right)}
            + 6 C{\left(T \right)}\right)}{V^{5}}

        '''
        try:
            return self._d2P_dV2
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V
        self._d2P_dV2 = d2P_dV2 = 2.0*self.R*T*(V2 + 3.0*V*self.B() + 6.0*self.C())/(V2*V2*V)
        return d2P_dV2

    def d2P_dT2(self):
        r'''

        .. math::
            \left(\frac{\partial^2 P}{\partial T^2}\right)_{V} =
            \frac{R \left(T \left(V \frac{d^{2}}{d T^{2}} B{\left(T \right)}
            + \frac{d^{2}}{d T^{2}} C{\left(T \right)}\right) + 2 V \frac{d}{d T}
            B{\left(T \right)} + 2 \frac{d}{d T} C{\left(T \right)}\right)}{V^{3}}

        '''
        try:
            return self._d2P_dT2
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V
        self._d2P_dT2 = d2P_dT2 = self.R*(T*(V*self.d2B_dT2() + self.d2C_dT2())
                              + 2.0*V*self.dB_dT() + 2.0*self.dC_dT())/(V*V*V)
        return d2P_dT2

    def H_dep(self):
        r'''

        .. math::
           H_{dep} = \frac{R T^{2} \left(2 V \frac{d}{d T} B{\left(T \right)}
           + \frac{d}{d T} C{\left(T \right)}\right)}{2 V^{2}} - R T \left(-1
            + \frac{V^{2} + V B{\left(T \right)} + C{\left(T \right)}}{V^{2}}
            \right)

        '''
        '''
        from sympy import *
        Z, R, T, V, P = symbols('Z, R, T, V, P')
        B, C = symbols('B, C', cls=Function)
        base =Eq(P*V/(R*T), 1 + B(T)/V + C(T)/V**2)
        P_sln = solve(base, P)[0]
        Z = P_sln*V/(R*T)

        # Two ways to compute H_dep
        Hdep2 = R*T - P_sln*V + integrate(P_sln - T*diff(P_sln, T), (V, oo, V))
        Hdep = -R*T*(Z-1) -integrate(diff(Z, T)/V, (V, oo, V))*R*T**2
        '''
        try:
            return self._H_dep
        except:
            pass

        T, V = self.T, self._V
        V2 = V*V
        RT = self.R*T
        self._H_dep = H_dep = RT*(T*(2.0*V*self.dB_dT() + self.dC_dT())/(2.0*V2)
               - (-1.0 + (V2 + V*self.B() + self.C())/V2))
        return H_dep

    def dH_dep_dT(self):
        r'''

        .. math::
           \frac{\partial H_{dep}}{\partial T} = \frac{R \left(2 T^{2} V
           \frac{d^{2}}{d T^{2}} B{\left(T \right)} + T^{2} \frac{d^{2}}{d T^{2}}
           C{\left(T \right)} + 2 T V \frac{d}{d T} B{\left(T \right)}
           - 2 V B{\left(T \right)} - 2 C{\left(T \right)}\right)}{2 V^{2}}

        '''
        try:
            return self._dH_dep_dT
        except:
            pass
        T, V = self.T, self._V
        self._dH_dep_dT = dH_dep_dT = (self.R*(2.0*T*T*V*self.d2B_dT2() + T*T*self.d2C_dT2()
            + 2.0*T*V*self.dB_dT() - 2.0*V*self.B() - 2.0*self.C())/(2.0*V*V))
        return dH_dep_dT

    def S_dep(self):
        r'''

        .. math::
           S_{dep} = \frac{R \left(- T \frac{d}{d T} C{\left(T \right)} + 2 V^{2}
           \ln{\left(\frac{V^{2} + V B{\left(T \right)} + C{\left(T \right)}}
           {V^{2}} \right)} - 2 V \left(T \frac{d}{d T} B{\left(T \right)}
            + B{\left(T \right)}\right) - C{\left(T \right)}\right)}{2 V^{2}}

        '''
        '''
        dP_dT = diff(P_sln, T)
        S_dep = integrate(dP_dT - R/V, (V, oo, V)) + R*log(Z)

        '''
        try:
            return self._S_dep
        except:
            pass

        T, V = self.T, self._V
        V2 = V*V
        self._S_dep = S_dep = (self.R*(-T*self.dC_dT() + 2*V2*log((V2 + V*self.B() + self.C())/V**2)
        - 2*V*(T*self.dB_dT() + self.B()) - self.C())/(2*V2))
        return S_dep

    def dS_dep_dT(self):
        r'''

        .. math::
           \frac{\partial S_{dep}}{\partial T} = \frac{R \left(2 V^{2} \left(V
           \frac{d}{d T} B{\left(T \right)} + \frac{d}{d T} C{\left(T \right)}
           \right) - \left(V^{2} + V B{\left(T \right)} + C{\left(T \right)}
           \right) \left(T \frac{d^{2}}{d T^{2}} C{\left(T \right)} + 2 V
           \left(T \frac{d^{2}}{d T^{2}} B{\left(T \right)} + 2 \frac{d}{d T}
           B{\left(T \right)}\right) + 2 \frac{d}{d T} C{\left(T \right)}
           \right)\right)}{2 V^{2} \left(V^{2} + V B{\left(T \right)}
           + C{\left(T \right)}\right)}

        '''
        try:
            return self._dS_dep_dT
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V

        self._dS_dep_dT = dS_dep_dT = (self.R*(2.0*V2*(V*self.dB_dT() + self.dC_dT()) - (V2 + V*self.B() + self.C())*(T*self.d2C_dT2()
        + 2.0*V*(T*self.d2B_dT2() + 2.0*self.dB_dT()) + 2.0*self.dC_dT()))/(2.0*V2*(V2 + V*self.B() + self.C())))
        return dS_dep_dT

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        new.N = self.N

        new.HeatCapacityGases = self.HeatCapacityGases
        new.model = self.model
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        Z = Z_from_virial_density_form(T, P, new.B(), new.C())
        new._V = Z*self.R*T/P
        return new

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.N = self.N
        new.HeatCapacityGases = self.HeatCapacityGases
        new.model = model = self.model
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        if T is not None:
            new.T = T
            if P is not None:
                new.P = P
                Z = Z_from_virial_density_form(T, P, new.B(), new.C())
                new._V = Z*self.R*T/P
            elif V is not None:
                P = new.P = self.R*T*(V*V + V*new.B() + new.C())/(V*V*V)
                new._V = V
        elif P is not None and V is not None:
            new.P = P
            # PV specified, solve for T
            def err(T):
                # Solve for P matching; probably there is a better solution here that does not
                # require the cubic solution but this works for now
                # TODO: instead of using self.to_TP_zs to allow calculating B and C,
                # they should be functional
                new_tmp = self.to_TP_zs(T=T, P=P, zs=zs)
                B = new_tmp.B()
                C = new_tmp.C()
                x2 = V*V + V*B + C
                x3 = self.R/(V*V*V)

                P_err = T*x2*x3 - P
                dP_dT = x3*(T*(V*new_tmp.dB_dT() + new_tmp.dC_dT()) + x2)
                return P_err, dP_dT

            T_ig = P*V/self.R # guess
            T = newton(err, T_ig, fprime=True, xtol=1e-15)
            new.T = T
        else:
            raise ValueError("Two of T, P, or V are needed")

        return new

    def B(self):
        try:
            return self._B
        except:
            pass
        N = self.N
        T = self.T
        if N == 1:
            return self.model.B_pures(T)[0]
        zs = self.zs
        B_matrix = self.model.B_matrix(T)
        B = 0.0
        for i in range(N):
            B_tmp = 0.0
            row = B_matrix[i]
            for j in range(N):
                B += zs[j]*row[j]
            B += zs[i]*B_tmp

        self._B = B
        return B

    def dB_dT(self):
        try:
            return self._dB_dT
        except:
            pass
        N = self.N
        T = self.T
        if N == 1:
            return self.model.dB_dT_pures(T)[0]
        zs = self.zs
        dB_dT_matrix = self.model.dB_dT_matrix(T)
        dB_dT = 0.0
        for i in range(N):
            dB_dT_tmp = 0.0
            row = dB_dT_matrix[i]
            for j in range(N):
                dB_dT += zs[j]*row[j]
            dB_dT += zs[i]*dB_dT_tmp

        self._dB_dT = dB_dT
        return dB_dT

    def d2B_dT2(self):
        try:
            return self._d2B_dT2
        except:
            pass
        N = self.N
        T = self.T
        if N == 1:
            return self.model.d2B_dT2_pures(T)[0]
        zs = self.zs
        d2B_dT2_matrix = self.model.d2B_dT2_matrix(T)
        d2B_dT2 = 0.0
        for i in range(N):
            d2B_dT2_tmp = 0.0
            row = d2B_dT2_matrix[i]
            for j in range(N):
                d2B_dT2 += zs[j]*row[j]
            d2B_dT2 += zs[i]*d2B_dT2_tmp

        self._d2B_dT2 = d2B_dT2
        return d2B_dT2

    def C(self):
        try:
            return self._C
        except:
            pass
        T = self.T
        zs = self.zs
        C_pures = self.model.C_pures(T)
        Ciij, Cijj = self.model.C_interactions(T)
        C = 0.0
        N = self.N
        for i in range(N):
            for j in range(N):
                # poling 5-4.3b should be able to be used to take out the k loop?
                for k in range(N):
                    if i == j == k:
                        Cval = C_pures[i]
                    elif i == j:
                        Cval = Ciij[i][j]
                    else:
                        Cval = Cijj[i][j]
                    C += zs[i]*zs[j]*zs[k]*Cval
        self._C = C
        return C

    def dC_dT(self):
        try:
            return self._dC_dT
        except:
            pass
        T = self.T
        zs = self.zs
        dC_dT_pures = self.model.dC_dT_pures(T)
        dC_dTiij, dC_dTijj = self.model.dC_dT_interactions(T)
        dC_dT = 0.0
        N = self.N
        for i in range(N):
            for j in range(N):
                # poling 5-4.3b should be able to be used to take out the k loop?
                for k in range(N):
                    if i == j == k:
                        dC_dTval = dC_dT_pures[i]
                    elif i == j:
                        dC_dTval = dC_dTiij[i][j]
                    else:
                        dC_dTval = dC_dTijj[i][j]
                    dC_dT += zs[i]*zs[j]*zs[k]*dC_dTval
        self._dC_dT = dC_dT
        return dC_dT

    def d2C_dT2(self):
        try:
            return self._d2C_dT2
        except:
            pass
        T = self.T
        zs = self.zs
        d2C_dT2_pures = self.model.d2C_dT2_pures(T)
        d2C_dT2iij, d2C_dT2ijj = self.model.d2C_dT2_interactions(T)
        d2C_dT2 = 0.0
        N = self.N
        for i in range(N):
            for j in range(N):
                # poling 5-4.3b should be able to be used to take out the k loop?
                for k in range(N):
                    if i == j == k:
                        d2C_dT2val = d2C_dT2_pures[i]
                    elif i == j:
                        d2C_dT2val = d2C_dT2iij[i][j]
                    else:
                        d2C_dT2val = d2C_dT2ijj[i][j]
                    d2C_dT2 += zs[i]*zs[j]*zs[k]*d2C_dT2val
        self._d2C_dT2 = d2C_dT2
        return d2C_dT2
