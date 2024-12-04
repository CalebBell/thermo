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
__all__ = ['DryAirLemmon', 'HumidAirRP1485']

from chemicals import air
from chemicals.thermal_conductivity import k_air_lemmon
from chemicals.virial import Z_from_virial_density_form
from chemicals.viscosity import mu_air_lemmon

from thermo.phases.helmholtz_eos import HelmholtzEOS
from thermo.phases.iapws_phase import IAPWS95
from thermo.phases.virial_phase import VirialGas


class DryAirLemmon(HelmholtzEOS):
    model_name = 'lemmon2000'
    is_gas = True
    is_liquid = False
    force_phase = 'g'

    # _MW = air.lemmon2000_air_MW = 28.9586
    _MW = 28.96546 # CoolProp
    rho_red = air.lemmon2000_air_rho_reducing
    rho_red_inv = 1.0/rho_red
    T_red = air.lemmon2000_air_T_reducing
    R = air.lemmon2000_air_R

    zs = [1.0]
    cmps = [0]
    N = 1
    T_MAX_FLASH = T_MAX_FIXED = 2000.0
    T_MIN_FLASH = T_MIN_FIXED = 132.6313 # For now gas only.
    T_fixed_transport = 265.262

    _Ar_func = staticmethod(air.lemmon2000_air_Ar)

    _d3Ar_ddeltadtau2_func = staticmethod(air.lemmon2000_air_d3Ar_ddeltadtau2)
    _d3Ar_ddelta2dtau_func = staticmethod(air.lemmon2000_air_d3Ar_ddelta2dtau)
    _d2Ar_ddeltadtau_func = staticmethod(air.lemmon2000_air_d2Ar_ddeltadtau)

    _dAr_dtau_func = staticmethod(air.lemmon2000_air_dAr_dtau)
    _d2Ar_dtau2_func = staticmethod(air.lemmon2000_air_d2Ar_dtau2)
    _d3Ar_dtau3_func = staticmethod(air.lemmon2000_air_d3Ar_dtau3)
    _d4Ar_dtau4_func = staticmethod(air.lemmon2000_air_d4Ar_dtau4)

    _dAr_ddelta_func = staticmethod(air.lemmon2000_air_dAr_ddelta)
    _d2Ar_ddelta2_func = staticmethod(air.lemmon2000_air_d2Ar_ddelta2)
    _d3Ar_ddelta3_func = staticmethod(air.lemmon2000_air_d3Ar_ddelta3)
    _d4Ar_ddelta4_func = staticmethod(air.lemmon2000_air_d4Ar_ddelta4)

    _d4Ar_ddelta2dtau2_func = staticmethod(air.lemmon2000_air_d4Ar_ddelta2dtau2)
    _d4Ar_ddelta3dtau_func = staticmethod(air.lemmon2000_air_d4Ar_ddelta3dtau)
    _d4Ar_ddeltadtau3_func = staticmethod(air.lemmon2000_air_d4Ar_ddeltadtau3)

    def __init__(self, T=HelmholtzEOS.T_DEFAULT, P=HelmholtzEOS.T_DEFAULT, zs=None):
        self.T = T
        self.P = P
        self._rho = rho = air.lemmon2000_rho(T, P)
        self._V = 1.0/rho
        self.tau = tau = self.T_red/T
        self.delta = delta = rho*self.rho_red_inv
        self.A0 = air.lemmon2000_air_A0(tau, delta)
        self.dA0_dtau = air.lemmon2000_air_dA0_dtau(tau, delta)
        self.d2A0_dtau2 = air.lemmon2000_air_d2A0_dtau2(tau, delta)
        self.d3A0_dtau3 = air.lemmon2000_air_d3A0_dtau3(tau, delta)

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.T = T
        new.P = P
        new._rho = rho = air.lemmon2000_rho(T, P)
        new._V = 1.0/rho
        new.tau = tau = new.T_red/T
        new.delta = delta = rho*new.rho_red_inv
        new.A0 = air.lemmon2000_air_A0(tau, delta)
        new.dA0_dtau = air.lemmon2000_air_dA0_dtau(tau, delta)
        new.d2A0_dtau2 = air.lemmon2000_air_d2A0_dtau2(tau, delta)
        new.d3A0_dtau3 = air.lemmon2000_air_d3A0_dtau3(tau, delta)
        return new

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        if T is not None and P is not None:
            new.T = T
            new._rho = air.lemmon2000_rho(T, P)
            new._V = 1.0/new._rho
            new.P = P
        elif T is not None and V is not None:
            new._rho = 1.0/V
            new._V = V
            P = air.lemmon2000_P(T, new._rho)
        elif P is not None and V is not None:
            T = air.lemmon2000_T(P=P, rho=1.0/V)
            new._rho = 1.0/V
            new._V = V
        else:
            raise ValueError("Two of T, P, or V are needed")

        new.P = P
        new.T = T
        new.tau = tau = new.T_red/T
        new.delta = delta = new._rho*new.rho_red_inv

        new.A0 = air.lemmon2000_air_A0(tau, delta)
        new.dA0_dtau = air.lemmon2000_air_dA0_dtau(tau, delta)
        new.d2A0_dtau2 = air.lemmon2000_air_d2A0_dtau2(tau, delta)
        new.d3A0_dtau3 = air.lemmon2000_air_d3A0_dtau3(tau, delta)
        return new

    def mu(self):
        r'''Calculate and return the viscosity of air according to the Lemmon
        and Jacobsen (2003) .
        For details, see :obj:`chemicals.viscosity.mu_air_lemmon`.

        Returns
        -------
        mu : float
            Viscosity of air, [Pa*s]
        '''
        try:
            return self._mu
        except:
            pass
        self._mu = mu = mu_air_lemmon(self.T, self._rho)
        return mu

    def k(self):
        r'''Calculate and return the thermal conductivity of air according to
        Lemmon and Jacobsen (2004)
        For details, see :obj:`chemicals.thermal_conductivity.k_air_lemmon`.

        Returns
        -------
        k : float
            Thermal conductivity of air, [W/m/K]
        '''
        try:
            return self._k
        except:
            pass
        # We require viscosity to calculate thermal conductivity
        self.mu()

        # This call is very expensive; this could be curve-fit as it is a 1D function
        drho_dP_Tr = self.to(T=self.T_fixed_transport, V=self._V, zs=self.zs).drho_dP()

        self._k = k = k_air_lemmon(T=self.T, rho=self._rho, Cp=self.Cp(), Cv=self.Cv(),
                       mu=self._mu, drho_dP=self.drho_dP(), drho_dP_Tr=drho_dP_Tr)
        return k



class HumidAirRP1485(VirialGas):
    is_gas = True
    is_liquid = False
    def __init__(self, Hfs=None, Gfs=None, T=None, P=None, zs=None,
                 ):
        # Although in put is zs, it is required to be in the order of
        # (air, water) mole fraction
        self.Hfs = Hfs
        self.Gfs = Gfs
        if Hfs is not None and Gfs is not None and None not in Hfs and None not in Gfs:
            self.Sfs = [(Hfi - Gfi)/298.15 for Hfi, Gfi in zip(Hfs, Gfs)]
        else:
            self.Sfs = None

        for i in (Hfs, Gfs, zs):
            if i is not None:
                self.N = len(i)
                break
        if zs is not None:
            self.psi_w = psi_w = zs[1]
            self.psi_a = psi_a = zs[0]
            self.zs = zs
        if T is not None:
            self.T = T
        if P is not None:
            self.P = P
        if T is not None and P is not None and zs is not None:
            self.air = DryAirLemmon(T=T, P=P)
            self.water = IAPWS95(T=T, P=P)
            Z = Z_from_virial_density_form(T, P, [self.B(), self.C()])
            self._V = Z*self.R*T/P
            self._MW = DryAirLemmon._MW*psi_a + IAPWS95._MW*psi_w

    def B(self):
        try:
            return self._B
        except:
            pass
        Baa = self.air.B_virial()
        Baw = air.TEOS10_BAW_derivatives(self.T)[0]
        Bww = self.water.B_virial()
        psi_a, psi_w = self.psi_a, self.psi_w

        self._B = B = psi_a*psi_a*Baa + 2.0*psi_a*psi_w*Baw + psi_w*psi_w*Bww
        return B

    def C(self):
        try:
            return self._C
        except:
            pass
        T = self.T
        Caaa = self.air.C_virial()
        Cwww = self.water.C_virial()
        Caww = air.TEOS10_CAWW_derivatives(T)[0]
        Caaw = air.TEOS10_CAAW_derivatives(T)[0]
        psi_a, psi_w = self.psi_a, self.psi_w
        self._C = C = (psi_a*psi_a*(Caaa + 3.0*psi_w*Caaw)
                       + psi_w*psi_w*(3.0*psi_a*Caww + psi_w*Cwww))
        return C
