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

__all__ = ['IdealGas']

from fluids.numerics import log
from fluids.numerics import numpy as np

from thermo.heat_capacity import HeatCapacityGas
from thermo.phases.phase import Phase

try:
    ndarray, array, zeros, ones = np.ndarray, np.array, np.zeros, np.ones
except:
    pass

class IdealGas(Phase):
    r'''Class for representing an ideal gas as a phase object. All departure
    properties are zero.

    .. math::
        P = \frac{RT}{V}

    Parameters
    ----------
    HeatCapacityGases : list[HeatCapacityGas]
        Objects proiding pure-component heat capacity correlations, [-]
    Hfs : list[float]
        Molar ideal-gas standard heats of formation at 298.15 K and 1 atm,
        [J/mol]
    Gfs : list[float]
        Molar ideal-gas standard Gibbs energies of formation at 298.15 K and
        1 atm, [J/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    zs : list[float], optional
        Mole fractions of each component, [-]

    Examples
    --------
    T-P initialization for oxygen and nitrogen, using Poling's polynomial heat
    capacities:

    >>> from scipy.constants import R
    >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
    ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
    >>> phase = IdealGas(T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
    >>> phase.Cp()
    29.1733530

    '''

    """DO NOT DELETE - EOS CLASS IS TOO SLOW!
    This will be important for fitting.

    """
    phase = 'g'
    force_phase = 'g'
    is_gas = True
    is_liquid = False
    composition_independent = True
    ideal_gas_basis = True

    pure_references = ('HeatCapacityGases',)
    pure_reference_types = (HeatCapacityGas,)

    obj_references = ('HeatCapacityGases', 'result', 'constants', 'correlations')

    model_attributes = ('Hfs', 'Gfs', 'Sfs') + pure_references

    def __init__(self, HeatCapacityGases=None, Hfs=None, Gfs=None, T=Phase.T_DEFAULT, P=Phase.P_DEFAULT, zs=None):
        self.HeatCapacityGases = HeatCapacityGases
        self.Hfs = Hfs
        self.Gfs = Gfs
        self.vectorized = vectorized = any(type(v) is ndarray for v in (zs, Hfs, Gfs))

        if Hfs is not None and Gfs is not None and None not in Hfs and None not in Gfs:
            T_ref_inv = 1.0/298.15
            self.Sfs = [(Hfi - Gfi)*T_ref_inv for Hfi, Gfi in zip(Hfs, Gfs)]
        else:
            self.Sfs = None

        if zs is not None:
            self.N = N = len(zs)
            self.zeros1d = zeros(N) if vectorized else [0.0]*N
            self.ones1d = ones(N) if vectorized else [1.0]*N
        elif HeatCapacityGases is not None:
            self.N = N = len(HeatCapacityGases)
            self.zeros1d = zeros(N) if vectorized else [0.0]*N
            self.ones1d = ones(N) if vectorized else [1.0]*N
        if zs is not None:
            self.zs = zs
        if T is not None:
            self.T = T
        if P is not None:
            self.P = P

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
        >>> from thermo import HeatCapacityGas, IdealGas
        >>> from scipy.constants import R
        >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
        ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
        >>> phase = IdealGas(T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
        >>> phase
        IdealGas(HeatCapacityGases=[HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-8.23131799e-12, 1.30537063e-08, 5.82012383e-07, -0.0021700747, 29.42488320])), HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [1.48828880e-11, -4.988677570e-08, 5.470916402e-05, -0.01491614593, 30.1814993]))], T=300, P=100000.0, zs=[0.79, 0.21])

        '''
        Cpgs = ', '.join(str(o) for o in self.HeatCapacityGases)
        base = f'IdealGas(HeatCapacityGases=[{Cpgs}], '
        for s in ('Hfs', 'Gfs', 'Sfs', 'T', 'P', 'zs'):
            if hasattr(self, s) and getattr(self, s) is not None:
                base += f'{s}={getattr(self, s)}, '
        if base[-2:] == ', ':
            base = base[:-2]
        base += ')'
        return base

    def fugacities(self):
        r'''Method to calculate and return the fugacities of each
        component in the phase.

        .. math::
            \text{fugacitiy}_i = z_i P

        Returns
        -------
        fugacities : list[float]
            Fugacities, [Pa]

        Examples
        --------
        >>> from scipy.constants import R
        >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
        ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63]))]
        >>> phase = IdealGas(T=300, P=1e5, zs=[.79, .21], HeatCapacityGases=HeatCapacityGases)
        >>> phase.fugacities()
        [79000.0, 21000.0]
        '''
        P = self.P
        return [P*zi for zi in self.zs]

    def lnphis(self):
        r'''Method to calculate and return the log of fugacity coefficients of
        each component in the phase.

        .. math::
            \ln \phi_i = 0.0

        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-]
        '''
        return self.zeros1d

    lnphis_G_min = lnphis
    lnphis_lowest_Gibbs = lnphis

    def phis(self):
        r'''Method to calculate and return the fugacity coefficients of
        each component in the phase.

        .. math::
             \phi_i = 1

        Returns
        -------
        phis : list[float]
            Fugacity fugacity coefficients, [-]
        '''
        return self.ones1d

    def dphis_dT(self):
        r'''Method to calculate and return the temperature derivative of
        fugacity coefficients of each component in the phase.

        .. math::
             \frac{\partial \phi_i}{\partial T} = 0

        Returns
        -------
        dphis_dT : list[float]
            Temperature derivative of fugacity fugacity coefficients, [1/K]
        '''
        return self.zeros1d

    def dphis_dP(self):
        r'''Method to calculate and return the pressure derivative of
        fugacity coefficients of each component in the phase.

        .. math::
             \frac{\partial \phi_i}{\partial P} = 0

        Returns
        -------
        dphis_dP : list[float]
            Pressure derivative of fugacity fugacity coefficients, [1/Pa]
        '''
        return self.zeros1d

    def dlnphis_dT(self):
        r'''Method to calculate and return the temperature derivative of the
        log of fugacity coefficients of each component in the phase.

        .. math::
             \frac{\partial \ln \phi_i}{\partial T} = 0

        Returns
        -------
        dlnphis_dT : list[float]
            Log fugacity coefficients, [1/K]
        '''
        return self.zeros1d

    def dlnphis_dP(self):
        r'''Method to calculate and return the pressure derivative of the
        log of fugacity coefficients of each component in the phase.

        .. math::
             \frac{\partial \ln \phi_i}{\partial P} = 0

        Returns
        -------
        dlnphis_dP : list[float]
            Log fugacity coefficients, [1/Pa]
        '''
        return self.zeros1d

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        new.N = self.N
        new.zeros1d = self.zeros1d
        new.ones1d = self.ones1d
        new.vectorized = self.vectorized

        new.HeatCapacityGases = self.HeatCapacityGases
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        return new

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        R = self.R
        if T is not None and V is not None:
            P = R*T/V
        elif P is not None and V is not None:
            T = P*V/R
        elif T is not None and P is not None:
            pass
        else:
            raise ValueError("Two of T, P, or V are needed")
        new.P = P
        new.T = T

        new.zs = zs
        new.N = self.N
        new.zeros1d = self.zeros1d
        new.ones1d = self.ones1d

        new.HeatCapacityGases = self.HeatCapacityGases
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        new.vectorized = self.vectorized

        return new


    ### Volumetric properties
    def V(self):
        r'''Method to calculate and return the molar volume of the phase.

        .. math::
             V = \frac{RT}{P}

        Returns
        -------
        V : float
            Molar volume, [m^3/mol]
        '''
        return self.R*self.T/self.P

    def dP_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        pressure of the phase.

        .. math::
             \frac{\partial P}{\partial T} = \frac{P}{T}

        Returns
        -------
        dP_dT : float
            First temperature derivative of pressure, [Pa/K]
        '''
        return self.P/self.T
    dP_dT_V = dP_dT

    def dP_dV(self):
        r'''Method to calculate and return the first volume derivative of
        pressure of the phase.

        .. math::
             \frac{\partial P}{\partial V} = \frac{-P^2}{RT}

        Returns
        -------
        dP_dV : float
            First volume derivative of pressure, [Pa*mol/m^3]
        '''
        return -self.P*self.P/(self.R*self.T)

    dP_dV_T = dP_dV

    def d2P_dT2(self):
        r'''Method to calculate and return the second temperature derivative of
        pressure of the phase.

        .. math::
             \frac{\partial^2 P}{\partial T^2} = 0

        Returns
        -------
        d2P_dT2 : float
            Second temperature derivative of pressure, [Pa/K^2]
        '''
        return 0.0
    d2P_dT2_V = d2P_dT2

    def d2P_dV2(self):
        r'''Method to calculate and return the second volume derivative of
        pressure of the phase.

        .. math::
             \frac{\partial^2 P}{\partial V^2} = \frac{2P^3}{R^2T^2}

        Returns
        -------
        d2P_dV2 : float
            Second volume derivative of pressure, [Pa*mol^2/m^6]
        '''
        P, T = self.P, self.T
        return 2.0*P*P*P/(self.R2*T*T)

    d2P_dV2_T = d2P_dV2

    def d2P_dTdV(self):
        r'''Method to calculate and return the second derivative of
        pressure with respect to temperature and volume of the phase.

        .. math::
             \frac{\partial^2 P}{\partial V \partial T} = \frac{-P^2}{RT^2}

        Returns
        -------
        d2P_dTdV : float
            Second volume derivative of pressure, [mol*Pa^2/(J*K)]
        '''
        P, T = self.P, self.T
        return -P*P/(self.R*T*T)

    def d2T_dV2(self):
        return 0.0

    def d2V_dT2(self):
        return 0.0

    def dV_dT(self):
        return self.R/self.P

    def PIP(self):
        return 1.0 # For speed

    def d2V_dP2(self):
        P, T = self.P, self.T
        return 2.0*self.R*T/(P*P*P)

    def d2T_dP2(self):
        return 0.0

    def dV_dP(self):
        P, T = self.P, self.T
        return -self.R*T/(P*P)

    def dT_dP(self):
        return self.T/self.P

    def dT_dV(self):
        return self.P*self.R_inv

    def dV_dzs(self):
        return self.zeros1d


    d2T_dV2_P = d2T_dV2
    d2V_dT2_P = d2V_dT2
    d2V_dP2_T = d2V_dP2
    d2T_dP2_V = d2T_dP2
    dV_dP_T = dV_dP
    dV_dT_P = dV_dT
    dT_dP_V = dT_dP
    dT_dV_P = dT_dV

    ### Thermodynamic properties

    def H(self):
        r'''Method to calculate and return the enthalpy of the phase.

        .. math::
            H = \sum_i z_i H_{i}^{ig}

        Returns
        -------
        H : float
            Molar enthalpy, [J/(mol)]
        '''
        try:
            return self._H
        except AttributeError:
            pass
        zs = self.zs
        try:
            Cpig_integrals_pure = self._Cpig_integrals_pure
        except AttributeError:
            Cpig_integrals_pure = self.Cpig_integrals_pure()
        H = 0.0
        for i in range(self.N):
            H += zs[i]*Cpig_integrals_pure[i]
        self._H = H
        return H

    def S(self):
        r'''Method to calculate and return the entropy of the phase.

        .. math::
            S = \sum_i z_i S_{i}^{ig} - R\ln\left(\frac{P}{P_{ref}}\right)
            - R\sum_i z_i \ln(z_i)

        Returns
        -------
        S : float
            Molar entropy, [J/(mol*K)]
        '''
        try:
            return self._S
        except AttributeError:
            pass
        Cpig_integrals_over_T_pure = self.Cpig_integrals_over_T_pure()
        log_zs = self.log_zs()
        R, P, zs = self.R, self.P, self.zs
        cmps = range(self.N)
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        S -= R*sum([zs[i]*log_zs[i] for i in cmps]) # ideal composition entropy composition
        S -= R*log(P*P_REF_IG_INV)
        for i in cmps:
            S += zs[i]*Cpig_integrals_over_T_pure[i]
        self._S = S
        return S

    def Cp(self):
        r'''Method to calculate and return the molar heat capacity of the
        phase.

        .. math::
            C_p = \sum_i z_i C_{p,i}^{ig}

        Returns
        -------
        Cp : float
            Molar heat capacity, [J/(mol*K)]
        '''
        try:
            return self._Cp
        except AttributeError:
            pass
        Cpigs_pure = self.Cpigs_pure()
        Cp, zs = 0.0, self.zs
        for i in range(self.N):
            Cp += zs[i]*Cpigs_pure[i]
        self._Cp = Cp
        return Cp

    dH_dT = Cp
    dH_dT_V = Cp # H does not depend on P, so the P is increased without any effect on H

    def dH_dP(self):
        r'''Method to calculate and return the first pressure derivative of
        molar enthalpy of the phase.

        .. math::
            \frac{\partial H}{\partial P} = 0

        Returns
        -------
        dH_dP : float
            First pressure derivative of molar enthalpy, [J/(mol*Pa)]
        '''
        return 0.0

    def d2H_dT2(self):
        r'''Method to calculate and return the first temperature derivative of
        molar heat capacity of the phase.

        .. math::
            \frac{\partial C_p}{\partial T} = \sum_i z_i \frac{\partial
            C_{p,i}^{ig}}{\partial T}

        Returns
        -------
        d2H_dT2 : float
            Second temperature derivative of enthalpy, [J/(mol*K^2)]
        '''
        try:
            return self._d2H_dT2
        except AttributeError:
            pass
        dCpigs_pure = self.dCpigs_dT_pure()
        dCp, zs = 0.0, self.zs
        for i in range(self.N):
            dCp += zs[i]*dCpigs_pure[i]
        self._d2H_dT2 = dCp
        return dCp

    def d2H_dP2(self):
        r'''Method to calculate and return the second pressure derivative of
        molar enthalpy of the phase.

        .. math::
            \frac{\partial^2 H}{\partial P^2} = 0

        Returns
        -------
        d2H_dP2 : float
            Second pressure derivative of molar enthalpy, [J/(mol*Pa^2)]
        '''
        return 0.0

    def d2H_dTdP(self):
        r'''Method to calculate and return the pressure derivative of
        molar heat capacity of the phase.

        .. math::
            \frac{\partial C_p}{\partial P} = 0

        Returns
        -------
        d2H_dTdP : float
            First pressure derivative of heat capacity, [J/(mol*K*Pa)]
        '''
        return 0.0

    def dH_dP_V(self):
        r'''Method to calculate and return the pressure derivative of
        molar enthalpy at constant volume of the phase.

        .. math::
            \left(\frac{\partial H}{\partial P}\right)_{V} = C_p
            \left(\frac{\partial T}{\partial P}\right)_{V}

        Returns
        -------
        dH_dP_V : float
            First pressure derivative of molar enthalpy at constant volume,
            [J/(mol*Pa)]
        '''
        dH_dP_V = self.Cp()*self.dT_dP()
        return dH_dP_V

    def dH_dV_T(self):
        r'''Method to calculate and return the volume derivative of
        molar enthalpy at constant temperature of the phase.

        .. math::
            \left(\frac{\partial H}{\partial V}\right)_{T} = 0

        Returns
        -------
        dH_dV_T : float
            First pressure derivative of molar enthalpy at constant volume,
            [J/(m^3)]
        '''
        return 0.0

    def dH_dV_P(self):
        r'''Method to calculate and return the volume derivative of
        molar enthalpy at constant pressure of the phase.

        .. math::
            \left(\frac{\partial H}{\partial V}\right)_{P} = C_p
            \left(\frac{\partial T}{\partial V}\right)_{P}

        Returns
        -------
        dH_dV_T : float
            First pressure derivative of molar enthalpy at constant volume,
            [J/(m^3)]
        '''
        dH_dV_P = self.dT_dV()*self.Cp()
        return dH_dV_P

    def dH_dzs(self):
        return self.Cpig_integrals_pure()

    def dS_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        molar entropy of the phase.

        .. math::
            \frac{\partial S}{\partial T} = \frac{C_p}{T}

        Returns
        -------
        dS_dT : float
            First temperature derivative of molar entropy, [J/(mol*K^2)]
        '''
        dS_dT = self.Cp()/self.T
        return dS_dT
    dS_dT_P = dS_dT

    def dS_dP(self):
        r'''Method to calculate and return the first pressure derivative of
        molar entropy of the phase.

        .. math::
            \frac{\partial S}{\partial P} = -\frac{R}{P}

        Returns
        -------
        dS_dP : float
            First pressure derivative of molar entropy, [J/(mol*K*Pa)]
        '''
        return -self.R/self.P

    def d2S_dP2(self):
        r'''Method to calculate and return the second pressure derivative of
        molar entropy of the phase.

        .. math::
            \frac{\partial^2 S}{\partial P^2} = \frac{R}{P^2}

        Returns
        -------
        d2S_dP2 : float
            Second pressure derivative of molar entropy, [J/(mol*K*Pa^2)]
        '''
        P = self.P
        return self.R/(P*P)

    def dS_dT_V(self):
        r'''Method to calculate and return the first temperature derivative of
        molar entropy at constant volume of the phase.

        .. math::
            \left(\frac{\partial S}{\partial T}\right)_V =
            \frac{C_p}{T} - \frac{R}{P}\frac{\partial P}{\partial T}

        Returns
        -------
        dS_dT_V : float
            First temperature derivative of molar entropy at constant volume,
            [J/(mol*K^2)]
        '''
        dS_dT_V = self.Cp()/self.T - self.R/self.P*self.dP_dT()
        return dS_dT_V

    def dS_dP_V(self):
        r'''Method to calculate and return the first pressure derivative of
        molar entropy at constant volume of the phase.

        .. math::
            \left(\frac{\partial S}{\partial P}\right)_V =
            \frac{-R}{P} + \frac{C_p}{T}\frac{\partial T}{\partial P}

        Returns
        -------
        dS_dP_V : float
            First pressure derivative of molar entropy at constant volume,
            [J/(mol*K*Pa)]
        '''
        dS_dP_V = -self.R/self.P + self.Cp()/self.T*self.dT_dP()
        return dS_dP_V

    def d2P_dTdP(self):
        return 0.0

    def d2P_dVdP(self):
        return 0.0

    def d2P_dVdT_TP(self):
        return 0.0

    def d2P_dT2_PV(self):
        return 0.0

    def H_dep(self):
        return 0.0

    G_dep = S_dep = U_dep = A_dep = H_dep

    def dS_dzs(self):
        try:
            return self._dS_dzs
        except AttributeError:
            pass
        log_zs = self.log_zs()
        integrals = self.Cpig_integrals_over_T_pure()
        R = self.R
        self._dS_dzs = [integrals[i] - R*(log_zs[i] + 1.0)
                        for i in range(self.N)]
        return self._dS_dzs

    # Properties using constants, correlations
    def mu(self):
        try:
            return self._mu
        except AttributeError:
            pass
        mu = self.correlations.ViscosityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._mu = mu
        return mu

    def k(self):
        try:
            return self._k
        except AttributeError:
            pass
        k = self.correlations.ThermalConductivityGasMixture.mixture_property(self.T, self.P, self.zs, self.ws())
        self._k = k
        return k
