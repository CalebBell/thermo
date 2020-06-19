# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from math import exp, log

from chemicals import none_and_length_check
from fluids.numerics import newton, brenth, oscillation_checker, secant, NotBoundedError

from chemicals.rachford_rice import flash_inner_loop

__all__ = ['K_value','Wilson_K_value', 'PR_water_K_value', 'flash_wilson', 
           'flash_Tb_Tc_Pc', 'flash_ideal', 'flash_ideal_basic', 'dew_at_T',
           'bubble_at_T', 'identify_phase', 'identify_phase_mixture',
           'Pbubble_mixture', 'bubble_at_P', 'Pdew_mixture', 'get_T_bub_est',
           'get_T_dew_est', 'get_P_dew_est', 'get_P_bub_est']


def K_value(P=None, Psat=None, phi_l=None, phi_g=None, gamma=None, Poynting=1):
    r'''Calculates the equilibrium K-value assuming Raoult's law,
    or an equation of state model, or an activity coefficient model,
    or a combined equation of state-activity model.

    The calculation procedure will use the most advanced approach with the
    provided inputs:

        * If `P`, `Psat`, `phi_l`, `phi_g`, and `gamma` are provided, use the
          combined approach.
        * If `P`, `Psat`, and `gamma` are provided, use the modified Raoult's
          law.
        * If `phi_l` and `phi_g` are provided, use the EOS only method.
        * If `P` and `Psat` are provided, use Raoult's law.

    Definitions:

    .. math::
        K_i=\frac{y_i}{x_i}

    Raoult's law:

    .. math::
        K_i = \frac{P_{i}^{sat}}{P}

    Activity coefficient, no EOS (modified Raoult's law):

    .. math::
        K_i = \frac{\gamma_i P_{i}^{sat}}{P}

    Equation of state only:

    .. math::
        K_i = \frac{\phi_i^l}{\phi_i^v} = \frac{f_i^l y_i}{f_i^v x_i}

    Combined approach (liquid reference fugacity coefficient is normally
    calculated the saturation pressure for it as a pure species; vapor fugacity
    coefficient calculated normally):

    .. math::
        K_i = \frac{\gamma_i P_i^{sat} \phi_i^{l,ref}}{\phi_i^v P}

    Combined approach, with Poynting Correction Factor (liquid molar volume in
    the integral is for i as a pure species only):

    .. math::
        K_i = \frac{\gamma_i P_i^{sat} \phi_i^{l, ref} \exp\left[\frac{
        \int_{P_i^{sat}}^P V_i^l dP}{RT}\right]}{\phi_i^v P}

    Parameters
    ----------
    P : float
        System pressure, optional
    Psat : float
        Vapor pressure of species i, [Pa]
    phi_l : float
        Fugacity coefficient of species i in the liquid phase, either
        at the system conditions (EOS-only case) or at the saturation pressure
        of species i as a pure species (reference condition for the combined
        approach), optional [-]
    phi_g : float
        Fugacity coefficient of species i in the vapor phase at the system
        conditions, optional [-]
    gamma : float
        Activity coefficient of species i in the liquid phase, optional [-]
    Poynting : float
        Poynting correction factor, optional [-]

    Returns
    -------
    K : float
        Equilibrium K value of component i, calculated with an approach
        depending on the provided inputs [-]

    Notes
    -----
    The Poynting correction factor is normally simplified as follows, due to
    a liquid's low pressure dependency:

    .. math::
        K_i = \frac{\gamma_i P_i^{sat} \phi_i^{l, ref} \exp\left[\frac{V_l
        (P-P_i^{sat})}{RT}\right]}{\phi_i^v P}

    Examples
    --------
    Raoult's law:

    >>> K_value(101325, 3000.)
    0.029607698001480384

    Modified Raoult's law:

    >>> K_value(P=101325, Psat=3000, gamma=0.9)
    0.026646928201332347

    EOS-only approach:

    >>> K_value(phi_l=1.6356, phi_g=0.88427)
    1.8496613025433408

    Gamma-phi combined approach:

    >>> K_value(P=1E6, Psat=1938800, phi_l=1.4356, phi_g=0.88427, gamma=0.92)
    2.8958055544121137

    Gamma-phi combined approach with a Poynting factor:

    >>> K_value(P=1E6, Psat=1938800, phi_l=1.4356, phi_g=0.88427, gamma=0.92,
    ... Poynting=0.999)
    2.8929097488577016

    References
    ----------
    .. [1] Gmehling, Jurgen, Barbel Kolbe, Michael Kleiber, and Jurgen Rarey.
       Chemical Thermodynamics for Process Simulation. 1st edition. Weinheim:
       Wiley-VCH, 2012.
    .. [2] Skogestad, Sigurd. Chemical and Energy Process Engineering. 1st
       edition. Boca Raton, FL: CRC Press, 2008.
    '''
    try:
        if gamma is not None:
            if phi_l is not None:
                return gamma*Psat*phi_l*Poynting/(phi_g*P)
            return gamma*Psat*Poynting/P
        elif phi_l is not None:
            return phi_l/phi_g
        return Psat/P
    except TypeError:
        raise Exception('Input must consist of one set from (P, Psat, phi_l, \
phi_g, gamma), (P, Psat, gamma), (phi_l, phi_g), (P, Psat)')


def Wilson_K_value(T, P, Tc, Pc, omega):
    r'''Calculates the equilibrium K-value for a component using Wilson's
    heuristic mode. This is very useful for initialization of stability tests
    and flashes.

    .. math::
        K_i = \frac{P_c}{P} \exp\left(5.37(1+\omega)\left[1 - \frac{T_c}{T}
        \right]\right)

    Parameters
    ----------
    T : float
        System temperature, [K]
    P : float
        System pressure, [Pa]
    Tc : float
        Critical temperature of fluid [K]
    Pc : float
        Critical pressure of fluid [Pa]
    omega : float
        Acentric factor for fluid, [-]

    Returns
    -------
    K : float
        Equilibrium K value of component, calculated via the Wilson heuristic
        [-]

    Notes
    -----
    There has been little literature exploration of other formlulas for the
    same purpose. This model may be useful even for activity coefficient
    models.

    Note the K-values are independent of composition; the correlation is
    applicable up to 3.5 MPa.

    A description for how this function was generated can be found in [2]_.

    Examples
    --------
    Ethane at 270 K and 76 bar:

    >>> Wilson_K_value(270.0, 7600000.0, 305.4, 4880000.0, 0.098)
    0.2963932297479371

    The "vapor pressure" predicted by this equation can be calculated by
    multiplying by pressure:

    >>> Wilson_K_value(270.0, 7600000.0, 305.4, 4880000.0, 0.098)*7600000.0
    2252588.546084322

    References
    ----------
    .. [1] Wilson, Grant M. "A Modified Redlich-Kwong Equation of State,
       Application to General Physical Data Calculations." In 65th National
       AIChE Meeting, Cleveland, OH, 1969.
    .. [2] Peng, Ding-Yu, and Donald B. Robinson. "Two and Three Phase
       Equilibrium Calculations for Systems Containing Water." The Canadian
       Journal of Chemical Engineering, December 1, 1976.
       https://doi.org/10.1002/cjce.5450540620.
    '''
    return Pc/P*exp((5.37*(1.0 + omega)*(1.0 - Tc/T)))


def PR_water_K_value(T, P, Tc, Pc):
    r'''Calculates the equilibrium K-value for a component against water
    according to the Peng and Robinson (1976) heuristic.

    .. math::
        K_i = 10^6 \frac{P_{ri}}{T_{ri}}

    Parameters
    ----------
    T : float
        System temperature, [K]
    P : float
        System pressure, [Pa]
    Tc : float
        Critical temperature of chemical [K]
    Pc : float
        Critical pressure of chemical [Pa]

    Returns
    -------
    K : float
        Equilibrium K value of component with water as the other phase (
        not as the reference), calculated via this heuristic [-]

    Notes
    -----
    Note the K-values are independent of composition.

    Examples
    --------
    Octane at 300 K and 1 bar:

    >>> PR_water_K_value(300, 1e5, 568.7, 2490000.0)
    76131.19143239626

    References
    ----------
    .. [1] Peng, Ding-Yu, and Donald B. Robinson. "Two and Three Phase
       Equilibrium Calculations for Systems Containing Water." The Canadian
       Journal of Chemical Engineering, December 1, 1976.
       https://doi.org/10.1002/cjce.5450540620.
    '''
    Tr = T/Tc
    Pr = P/Pc
    return 1e6*Pr/Tr


def flash_wilson(zs, Tcs, Pcs, omegas, T=None, P=None, VF=None):
    r'''PVT flash model using Wilson's equation - useful for obtaining initial
    guesses for more rigorous models, or it can be used as its own model.
    Capable of solving with two of `T`, `P`, and `VF` for the other one;
    that results in three solve modes, but for `VF=1` and `VF=0`, there are
    additional solvers; for a total of seven solvers implemented.

    This model uses `flash_inner_loop` to solve the Rachford-Rice problem.

    .. math::
        K_i = \frac{P_c}{P} \exp\left(5.37(1+\omega)\left[1 - \frac{T_c}{T}
        \right]\right)

    Parameters
    ----------
    zs : list[float]
        Mole fractions of the phase being flashed, [-]
    Tcs : list[float]
        Critical temperatures of all species, [K]
    Pcs : list[float]
        Critical pressures of all species, [Pa]
    omegas : list[float]
        Acentric factors of all species, [-]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    VF : float, optional
        Molar vapor fraction, [-]

    Returns
    -------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    VF : float
        Molar vapor fraction, [-]
    xs : list[float]
        Mole fractions of liquid phase, [-]
    ys : list[float]
        Mole fractions of vapor phase, [-]

    Notes
    -----
    For the cases where `VF` is 1 or 0 and T is known, an explicit solution is
    used. For the same cases where `P` and `VF` are known, there is no explicit
    solution available.

    There is an internal `Tmax` parameter, set to 50000 K; which, in the event
    of convergence of the Secant method, is used as a bounded for a bounded
    solver. It is used in the PVF solvers. This typically allows pressures
    up to 2 GPa to be converged to. However, for narrow-boiling mixtures, the
    PVF failure may occur at much lower pressures.

    Examples
    --------
    >>> Tcs = [305.322, 540.13]
    >>> Pcs = [4872200.0, 2736000.0]
    >>> omegas = [0.099, 0.349]
    >>> zs = [0.4, 0.6]
    >>> flash_wilson(zs=zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=300, P=1e5)
    (300, 100000.0, 0.42219453293637355, [0.020938815080034565, 0.9790611849199654], [0.9187741856225791, 0.08122581437742094])
    '''
    T_MAX = 50000.0
    N = len(zs)
    cmps = range(N)
    # Assume T and P to begin with
    if T is not None and P is not None:
        P_inv, T_inv = 1.0/P, 1.0/T
        Ks = [Pcs[i]*P_inv*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]*T_inv))) for i in cmps]
#        all_under_1, all_over_1 = True, True
#        for K in Ks:
#            if K < 1.0:
#                all_over_1 = False
#            else:
#                all_under_1 = False
#            if all_over_1:
#                raise ValueError("Fail")
#            elif all_under_1:
#                raise ValueError("Fail")
        ans = (T, P) + flash_inner_loop(zs=zs, Ks=Ks)
        return ans
    elif T is not None and VF == 0.0:
        ys = []
        P_bubble = 0.0
        T_inv = 1.0/T
        for i in cmps:
            v = zs[i]*Pcs[i]*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]*T_inv)))
            P_bubble += v
            ys.append(v)
        P_inv = 1.0/P_bubble
        for i in cmps:
            ys[i] *= P_inv
        return (T, P_bubble, 0.0, zs, ys)
    elif T is not None and VF == 1.0:
        xs = []
        P_dew = 0.
        T_inv = 1.0/T
        for i in cmps:
            v = zs[i]/(Pcs[i]*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]*T_inv))))
            P_dew += v
            xs.append(v)
        P_dew = 1./P_dew
        for i in cmps:
            xs[i] *= P_dew
        return (T, P_dew, 1.0, xs, zs)
    elif T is not None and VF is not None:
        # Solve for the pressure to create the desired vapor fraction
        P_bubble = 0.0
        P_dew = 0.
        T_inv = 1.0/T
        K_Ps = []
        for i in cmps:
            K_P = Pcs[i]*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]*T_inv)))
            P_bubble += zs[i]*K_P
            P_dew += zs[i]/K_P
            K_Ps.append(K_P)
        P_dew = 1./P_dew
#        try:
        '''Rachford-Rice esque solution in terms of pressure.
        from sympy import *
        N = 1
        cmps = range(N)
        zs = z0, z1, z2, z3 = symbols('z0, z1, z2, z3')
        Ks_P = K0_P, K1_P, K2_P, K3_P = symbols('K0_P, K1_P, K2_P, K3_P')
        VF, P = symbols('VF, P')
        tot = 0
        for i in cmps:
            tot += zs[i]*(Ks_P[i]/P - 1)/(1 + VF*(Ks_P[i]/P - 1))
        cse([tot, diff(tot, P)], optimizations='basic')
        '''
        def err(P):
            P_inv = 1.0/P
            err, derr = 0.0, 0.0
            for i in cmps:
                x50 = K_Ps[i]*P_inv
                x0 = x50 - 1.0
                x1 = VF*x0
                x2 = 1.0/(x1 + 1.0)
                x3 = x2*zs[i]
                err += x0*x3
                derr += x50*P_inv*x3*(x1*x2 - 1.0)
            return err, derr
        P_guess = P_bubble + VF*(P_dew - P_bubble) # Linear interpolation
        P = newton(err, P_guess, fprime=True, bisection=True,
                   low=P_dew, high=P_bubble)
        P_inv = 1.0/P
        xs, ys = [], []
        for i in cmps:
            Ki = K_Ps[i]*P_inv
            xi = zs[i]/(1.0 + VF*(Ki - 1.0))
            ys.append(Ki*xi)
            xs.append(xi)
        return (T, P, VF, xs, ys)
#        except:
#            info = []
#            def to_solve(P):
#                T_calc, P_calc, VF_calc, xs, ys = flash_wilson(zs, Tcs, Pcs, omegas, T=T, P=P)
#                info[:] = T_calc, P_calc, VF_calc, xs, ys
#                err = VF_calc - VF
#                return err
#            P = brenth(to_solve, P_dew, P_bubble)
#            return tuple(info)
    elif P is not None:
        P_inv = 1.0/P
        Ks, xs = [0.0]*N, [0.0]*N
        x50s = [5.37*(omegai + 1.0) for omegai in omegas]
        def to_solve(T_guess):
            err, derr = 0.0, 0.0
            T_inv = 1.0/T_guess
            T_inv2 = T_inv*T_inv
            for i in cmps:
                Ks[i] = Pcs[i]*exp(x50s[i]*(1.0 - Tcs[i]*T_inv))*P_inv
                dKi_dT = Ks[i]*x50s[i]*T_inv2*Tcs[i]
                x1 = Ks[i] - 1.0
                x2 = VF*x1
                x3 = 1.0/(x2 + 1.0)
                xs[i] = x3*zs[i]
                err += x1*xs[i]
                derr += xs[i]*(1.0 - x2*x3)*dKi_dT
            return err, derr


        T_low, T_high = 1e100, 0.0
        logP = log(P)
        for i in cmps:
            T_K_1 = Tcs[i]*x50s[i]/(x50s[i] - logP + log(Pcs[i]))
            if T_K_1 < T_low:
                T_low = T_K_1
            if T_K_1 > T_high:
                T_high = T_K_1
        if T_low < 0.0:
            T_low = 1e-12
        if T_high <= 0.0:
            raise ValueError("No temperature exists which makes Wilson K factor above 1 - decrease pressure")
        if T_high < 0.1*T_MAX:
            T = 0.5*(T_low + T_high)
        else:
            T = 0.0
            for i in cmps:
                T += zs[i]*Tcs[i]
            T *= 0.666666
            if T < T_low:
                T = T_low + 1.0 # Take a nominal step
        #try:
        T = newton(to_solve, T, fprime=True, low=T_low, xtol=1e-13, bisection=True) # High bound not actually a bound, only low bound
        #except Exception as e:
        #    print(e, [P, VF])
        #    T = 1e100
        if 1e-10 < T < T_MAX:
            ys = x50s
            for i in cmps:
                ys[i] = xs[i]*Ks[i]
            return (T, P, VF, xs, ys)

#    # Old code - may converge where the other will not
#    if P is not None and VF == 1.0:
#        def to_solve(T_guess):
#            # Avoid some nasty unpleasantness in newton
#            T_guess = abs(T_guess)
#            P_dew = 0.
#            for i in range(len(zs)):
#                P_dew += zs[i]/(Pcs[i]*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]/T_guess))))
#            P_dew = 1./P_dew
##            print(P_dew - P, T_guess)
#            return P_dew - P
#        # 2/3 average critical point
#        T_guess = sum([.666*Tcs[i]*zs[i] for i in cmps])
#        try:
#            T_dew = abs(newton(to_solve, T_guess, maxiter=50, ytol=1e-2))
#        except Exception as e:
##            print(e)
#            T_dew = None
#        if T_dew is None or T_dew > T_MAX*5.0:
#            # Went insanely high T, bound it with brenth
#            T_low_guess = sum([.1*Tcs[i]*zs[i] for i in cmps])
#            try:
#                T_dew = brenth(to_solve, T_MAX, T_low_guess)
#            except NotBoundedError:
#                raise Exception("Bisecting solver could not find a solution between %g K and %g K" %(T_MAX, T_low_guess))
#        return flash_wilson(zs, Tcs, Pcs, omegas, T=T_dew, P=P)
#    elif P is not None and VF == 0.0:
#        def to_solve(T_guess):
#            T_guess = abs(T_guess)
#            P_bubble = 0.0
#            for i in cmps:
#                P_bubble += zs[i]*Pcs[i]*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]/T_guess)))
#            return P_bubble - P
#        # 2/3 average critical point
#        T_guess = sum([.55*Tcs[i]*zs[i] for i in cmps])
#        try:
#            T_bubble = abs(newton(to_solve, T_guess, maxiter=50, ytol=1e-2))
#        except Exception as e:
#            T_bubble = None
#        if T_bubble is None or T_bubble > T_MAX*5.0:
#            # Went insanely high T, bound it with brenth
#            T_low_guess = sum([.1*Tcs[i]*zs[i] for i in cmps])
#            try:
#                T_bubble = brenth(to_solve, T_MAX, T_low_guess)
#            except NotBoundedError:
#                raise Exception("Bisecting solver could not find a solution between %g K and %g K" %(T_MAX, T_low_guess))
#
#        return flash_wilson(zs, Tcs, Pcs, omegas, T=T_bubble, P=P)
#    elif P is not None and VF is not None:
#        # Solve for in the middle of Pdew
#        T_low = flash_wilson(zs, Tcs, Pcs, omegas, P=P, VF=1)[0]
#        T_high = flash_wilson(zs, Tcs, Pcs, omegas, P=P, VF=0)[0]
##        print(T_low, T_high)
#        info = []
#        def err(T):
#            T_calc, P_calc, VF_calc, xs, ys = flash_wilson(zs, Tcs, Pcs, omegas, T=T, P=P)
##            if abs(VF_calc) > 100: # Did not work at all
##                VF_calc = abs(VF_calc)
#            info[:] = T_calc, P_calc, VF_calc, xs, ys
##            print(T, VF_calc - VF)
#            return VF_calc - VF
#        # Nasty function for tolerance; the default works and is good enough, could remove some
#        # iterations in the fuure
#        P = brenth(err, T_low, T_high, xtol=1e-14)
#        return tuple(info)
    else:
        raise ValueError("Provide two of P, T, and VF")


def flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, T=None, P=None, VF=None):
    r'''PVT flash model using a model published in [1]_, which provides a PT
    surface using only each compound's boiling temperature and critical
    temperature and pressure. This is useful for obtaining initial
    guesses for more rigorous models, or it can be used as its own model.
    Capable of solving with two of `T`, `P`, and `VF` for the other one;
    that results in three solve modes, but for `VF=1` and `VF=0`, there are
    additional solvers; for a total of seven solvers implemented.

    This model uses `flash_inner_loop` to solve the Rachford-Rice problem.

    .. math::
        K_i = \frac{P_{c,i}^{\left(\frac{1}{T} - \frac{1}{T_{b,i}} \right) /
        \left(\frac{1}{T_{c,i}} - \frac{1}{T_{b,i}} \right)}}{P}

    Parameters
    ----------
    zs : list[float]
        Mole fractions of the phase being flashed, [-]
    Tbs : list[float]
        Boiling temperatures of all species, [K]
    Tcs : list[float]
        Critical temperatures of all species, [K]
    Pcs : list[float]
        Critical pressures of all species, [Pa]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    VF : float, optional
        Molar vapor fraction, [-]

    Returns
    -------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    VF : float
        Molar vapor fraction, [-]
    xs : list[float]
        Mole fractions of liquid phase, [-]
    ys : list[float]
        Mole fractions of vapor phase, [-]

    Notes
    -----
    For the cases where `VF` is 1 or 0 and T is known, an explicit solution is
    used. For the same cases where `P` and `VF` are known, there is no explicit
    solution available.

    There is an internal `Tmax` parameter, set to 50000 K; which, in the event
    of convergence of the Secant method, is used as a bounded for a bounded
    solver. It is used in the PVF solvers. This typically allows pressures
    up to 2 MPa to be converged to. Failures may still occur for other
    conditions.

    Examples
    --------
    >>> Tcs = [305.322, 540.13]
    >>> Pcs = [4872200.0, 2736000.0]
    >>> Tbs = [184.55, 371.53]
    >>> zs = [0.4, 0.6]
    >>> flash_Tb_Tc_Pc(zs=zs, Tcs=Tcs, Pcs=Pcs, Tbs=Tbs, T=300, P=1e5)
    (300, 100000.0, 0.3807040748145384, [0.031157843036568357, 0.9688421569634317], [0.9999999998827085, 1.1729141887515062e-10])
    '''
    T_MAX = 50000
    N = len(zs)
    cmps = range(N)
    # Assume T and P to begin with
    if T is not None and P is not None:
        Ks = [Pcs[i]**((1.0/T - 1.0/Tbs[i])/(1.0/Tcs[i] - 1.0/Tbs[i]))/P for i in cmps]
        return (T, P) + flash_inner_loop(zs=zs, Ks=Ks, check=True)

    if T is not None and VF == 0:
        P_bubble = 0.0
        for i in cmps:
            P_bubble += zs[i]*Pcs[i]**((1.0/T - 1.0/Tbs[i])/(1.0/Tcs[i] - 1.0/Tbs[i]))
        return flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, T=T, P=P_bubble)
    if T is not None and VF == 1:
        # Checked to be working vs. PT implementation.
        P_dew = 0.
        for i in cmps:
            P_dew += zs[i]/( Pcs[i]**((1.0/T - 1.0/Tbs[i])/(1.0/Tcs[i] - 1.0/Tbs[i])) )
        P_dew = 1./P_dew
        return flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, T=T, P=P_dew)
    elif T is not None and VF is not None:
        # Solve for in the middle of Pdew
        P_low = flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, T=T, VF=1)[1]
        P_high = flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, T=T, VF=0)[1]
        info = []
        def err(P):
            T_calc, P_calc, VF_calc, xs, ys = flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, T=T, P=P)
            info[:] = T_calc, P_calc, VF_calc, xs, ys
            return VF_calc - VF
        P = brenth(err, P_low, P_high)
        return tuple(info)

    elif P is not None and VF == 1:
        checker = oscillation_checker()
        def to_solve(T_guess):
            T_guess = abs(T_guess)
            P_dew = 0.
            for i in range(len(zs)):
                P_dew += zs[i]/( Pcs[i]**((1.0/T_guess - 1.0/Tbs[i])/(1.0/Tcs[i] - 1.0/Tbs[i])) )
            P_dew = 1./P_dew
            err = P_dew - P
            if checker(T_guess, err):
                raise ValueError("Oscillation")
#            print(T_guess, err)
            return err

        Tc_pseudo = sum([Tcs[i]*zs[i] for i in cmps])
        T_guess = 0.666*Tc_pseudo
        try:
            T_dew = abs(secant(to_solve, T_guess, maxiter=50, ytol=1e-2)) # , high=Tc_pseudo*3
        except:
            T_dew = None
        if T_dew is None or T_dew > T_MAX*5.0:
            # Went insanely high T, bound it with brenth
            T_low_guess = sum([.1*Tcs[i]*zs[i] for i in cmps])
            checker = oscillation_checker(both_sides=True, minimum_progress=.05)
            try:
                T_dew = brenth(to_solve, T_MAX, T_low_guess)
            except NotBoundedError:
                raise Exception("Bisecting solver could not find a solution between %g K and %g K" %(T_MAX, T_low_guess))
        return flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, T=T_dew, P=P)

    elif P is not None and VF == 0:
        checker = oscillation_checker()
        def to_solve(T_guess):
            T_guess = abs(T_guess)
            P_bubble = 0.0
            for i in cmps:
                P_bubble += zs[i]*Pcs[i]**((1.0/T_guess - 1.0/Tbs[i])/(1.0/Tcs[i] - 1.0/Tbs[i]))

            err = P_bubble - P
            if checker(T_guess, err):
                raise ValueError("Oscillation")

#            print(T_guess, err)
            return err
        # 2/3 average critical point
        Tc_pseudo = sum([Tcs[i]*zs[i] for i in cmps])
        T_guess = 0.55*Tc_pseudo
        try:
            T_bubble = abs(secant(to_solve, T_guess, maxiter=50, ytol=1e-2)) # , high=Tc_pseudo*4
        except Exception as e:
#            print(e)
            checker = oscillation_checker(both_sides=True, minimum_progress=.05)
            T_bubble = None
        if T_bubble is None or T_bubble > T_MAX*5.0:
            # Went insanely high T (or could not converge because went too high), bound it with brenth
            T_low_guess = 0.1*Tc_pseudo
            try:
                T_bubble = brenth(to_solve, T_MAX, T_low_guess)
            except NotBoundedError:
                raise Exception("Bisecting solver could not find a solution between %g K and %g K" %(T_MAX, T_low_guess))

        return flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, T=T_bubble, P=P)
    elif P is not None and VF is not None:
        T_low = flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, P=P, VF=1)[0]
        T_high = flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, P=P, VF=0)[0]
        info = []
        def err(T):
            T_calc, P_calc, VF_calc, xs, ys = flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, T=T, P=P)
            info[:] = T_calc, P_calc, VF_calc, xs, ys
            return VF_calc - VF
        P = brenth(err, T_low, T_high)
        return tuple(info)
    else:
        raise ValueError("Provide two of P, T, and VF")


def flash_ideal(zs, funcs, Tcs=None, T=None, P=None, VF=None):
    T_MAX = 50000.0
    N = len(zs)
    cmps = range(N)
    if Tcs is None:
        Tcs = [fi.solve_prop(1e6) for fi in funcs]
    if T is not None and P is not None:
        P_inv = 1.0/P
        Ks = [P_inv*funcs[i](T) for i in cmps]
        ans = (T, P) + flash_inner_loop(zs=zs, Ks=Ks)
        return ans
    if T is not None and VF == 0.0:
        ys = []
        P_bubble = 0.0
        for i in cmps:
            v = funcs[i](T)*zs[i]
            P_bubble += v
            ys.append(v)

        P_inv = 1.0/P_bubble
        for i in cmps:
            ys[i] *= P_inv
        return (T, P_bubble, 0.0, zs, ys)
    if T is not None and VF == 1.0:
        xs = []
        P_dew = 0.
        for i in cmps:
            v = zs[i]/funcs[i](T)
            P_dew += v
            xs.append(v)
        P_dew = 1./P_dew
        for i in cmps:
            xs[i] *= P_dew
        return (T, P_dew, 1.0, xs, zs)
    elif T is not None and VF is not None:
        # Solve for in the middle of Pdew
        P_low = flash_ideal(zs, funcs, Tcs, T=T, VF=1)[1]
        P_high = flash_ideal(zs, funcs, Tcs, T=T, VF=0)[1]
        info = []
        def to_solve(P):
            T_calc, P_calc, VF_calc, xs, ys = flash_ideal(zs, funcs, Tcs, T=T, P=P)
            info[:] = T_calc, P_calc, VF_calc, xs, ys
            err = VF_calc - VF
            return err
        P = brenth(to_solve, P_low, P_high)
        return tuple(info)
    elif P is not None and VF == 1:
        def to_solve(T_guess):
            T_guess = abs(T_guess)
            P_dew = 0.
            for i in cmps:
                P_dew += zs[i]/funcs[i](T_guess)
            P_dew = 1./P_dew
            return P_dew - P

        # 2/3 average critical point
        T_guess = .66666*sum([Tcs[i]*zs[i] for i in cmps])
        try:
            T_dew = abs(secant(to_solve, T_guess, maxiter=50, ytol=1e-2))
        except Exception as e:
            T_dew = None
        if T_dew is None or T_dew > T_MAX*5.0:
            # Went insanely high T, bound it with brenth
            T_low_guess = sum([.1*Tcs[i]*zs[i] for i in cmps])
            try:
                T_dew = brenth(to_solve, T_MAX, T_low_guess)
            except NotBoundedError:
                raise Exception("Bisecting solver could not find a solution between %g K and %g K" %(T_MAX, T_low_guess))
        return flash_ideal(zs, funcs, Tcs, T=T_dew, P=P)

    elif P is not None and VF == 0:
        def to_solve(T_guess):
            # T_guess = abs(T_guess)
            P_bubble = 0.0
            for i in cmps:
                P_bubble += zs[i]*funcs[i](T_guess)
            return P_bubble - P
        # 2/3 average critical point
        T_guess = sum([.55*Tcs[i]*zs[i] for i in cmps])
        try:
            T_bubble = abs(secant(to_solve, T_guess, maxiter=50, ytol=1e-2, bisection=True))
        except Exception as e:
            T_bubble = None
        if T_bubble is None or T_bubble > T_MAX*5.0:
            # Went insanely high T, bound it with brenth
            T_low_guess = sum([.1*Tcs[i]*zs[i] for i in cmps])
            try:
                T_bubble = brenth(to_solve, T_MAX, T_low_guess)
            except NotBoundedError:
                raise Exception("Bisecting solver could not find a solution between %g K and %g K" %(T_MAX, T_low_guess))

        return flash_ideal(zs, funcs, Tcs, T=T_bubble, P=P)
    elif P is not None and VF is not None:
        T_low = flash_ideal(zs, funcs, Tcs, P=P, VF=1)[0]
        T_high = flash_ideal(zs, funcs, Tcs, P=P, VF=0)[0]
        info = []
        def err(T):
            T_calc, P_calc, VF_calc, xs, ys = flash_ideal(zs, funcs, Tcs, T=T, P=P)
            info[:] = T_calc, P_calc, VF_calc, xs, ys
            return VF_calc - VF
        P = brenth(err, T_low, T_high, xtol=1e-14)
        return tuple(info)
    else:
        raise ValueError("Provide two of P, T, and VF")


def flash_ideal_basic(P, zs, Psats):
#    if not fugacities:
#        fugacities = [1 for i in range(len(zs))]
#    if not gammas:
#        gammas = [1 for i in range(len(zs))]
    if not none_and_length_check((zs, Psats)):
        raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
    Ks = [K_value(P=P, Psat=Psats[i]) for i in range(len(zs))]
    def valid_range(zs, Ks):
        valid = True
        if sum([zs[i]*Ks[i] for i in range(len(Ks))]) < 1:
            valid = False
        if sum([zs[i]/Ks[i] for i in range(len(Ks))]) < 1:
            valid = False
        return valid
    if not valid_range(zs, Ks):
        raise Exception('Solution does not exist')

    V_over_F, xs, ys = flash_inner_loop(zs=zs, Ks=Ks)
    if V_over_F < 0:
        raise Exception('V_over_F is negative!')
    return xs, ys, V_over_F


def dew_at_T(zs, Psats, fugacities=None, gammas=None):
    '''
    >>> dew_at_T([0.5, 0.5], [1400, 7000])
    2333.3333333333335
    >>> dew_at_T([0.5, 0.5], [1400, 7000], gammas=[1.1, .75])
    2381.443298969072
    >>> dew_at_T([0.5, 0.5], [1400, 7000], gammas=[1.1, .75], fugacities=[.995, 0.98])
    2401.621874512658
    '''
    if fugacities is None and gammas is None:
        try:
            return 1.0/sum([zs[i]/Psats[i] for i in range(len(zs))])
        except Exception as e:
            if not none_and_length_check((Psats,)):
                raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
            raise e
    elif gammas is not None and fugacities is None:
        try:
            return 1.0/sum([zs[i]/(Psats[i]*gammas[i]) for i in range(len(zs))])
        except Exception as e:
            if not none_and_length_check((Psats, gammas)):
                raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
            raise e
    elif fugacities is not None and gammas is None:
        try:
            return 1.0/sum([zs[i]*fugacities[i]/Psats[i] for i in range(len(zs))])
        except Exception as e:
            if not none_and_length_check((Psats, fugacities)):
                raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
            raise e
    elif fugacities is not None and gammas is not None:
        try:
            return 1.0/sum([zs[i]*fugacities[i]/(Psats[i]*gammas[i]) for i in range(len(zs))])
        except Exception as e:
            if not none_and_length_check((zs, Psats, fugacities, gammas)):
                raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
            raise e
    return ValueError


def bubble_at_T(zs, Psats, fugacities=None, gammas=None):
    '''
    >>> bubble_at_T([0.5, 0.5], [1400, 7000])
    4200.0
    >>> bubble_at_T([0.5, 0.5], [1400, 7000], gammas=[1.1, .75])
    3395.0
    >>> bubble_at_T([0.5, 0.5], [1400, 7000], gammas=[1.1, .75], fugacities=[.995, 0.98])
    3452.440775305097
    '''
    l = len(zs)
    if fugacities is None and gammas is None:
        try:
            return sum([zs[i]*Psats[i] for i in range(l)])
        except Exception as e:
            if not none_and_length_check((Psats,)):
                raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
            raise e
    elif gammas is not None and fugacities is None:
        try:
            return sum([zs[i]*Psats[i]*gammas[i] for i in range(l)])
        except Exception as e:
            if not none_and_length_check((Psats, gammas)):
                raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
            raise e
    elif fugacities is not None and gammas is None:
        try:
            return sum([zs[i]*Psats[i]/fugacities[i] for i in range(l)])
        except Exception as e:
            if not none_and_length_check((Psats, fugacities)):
                raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
            return e
    elif fugacities is not None and gammas is not None:
        try:
            return sum([zs[i]*Psats[i]*gammas[i]/fugacities[i] for i in range(l)])
        except Exception as e:
            if not none_and_length_check((zs, Psats, fugacities, gammas)):
                raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
            return e
    else:
        raise ValueError


def identify_phase(T, P, Tm=None, Tb=None, Tc=None, Psat=None):
    r'''Determines the phase of a one-species chemical system according to
    basic rules, using whatever information is available. Considers only the
    phases liquid, solid, and gas; does not consider two-phase
    scenarios, as should occurs between phase boundaries.

    * If the melting temperature is known and the temperature is under or equal
      to it, consider it a solid.
    * If the critical temperature is known and the temperature is greater or
      equal to it, consider it a gas.
    * If the vapor pressure at `T` is known and the pressure is under or equal
      to it, consider it a gas. If the pressure is greater than the vapor
      pressure, consider it a liquid.
    * If the melting temperature, critical temperature, and vapor pressure are
      not known, attempt to use the boiling point to provide phase information.
      If the pressure is between 90 kPa and 110 kPa (approximately normal),
      consider it a liquid if it is under the boiling temperature and a gas if
      above the boiling temperature.
    * If the pressure is above 110 kPa and the boiling temperature is known,
      consider it a liquid if the temperature is under the boiling temperature.
    * Return None otherwise.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    Tm : float, optional
        Normal melting temperature, [K]
    Tb : float, optional
        Normal boiling point, [K]
    Tc : float, optional
        Critical temperature, [K]
    Psat : float, optional
        Vapor pressure of the fluid at `T`, [Pa]

    Returns
    -------
    phase : str
        Either 's', 'l', 'g', or None if the phase cannot be determined

    Notes
    -----
    No special attential is paid to any phase transition. For the case where
    the melting point is not provided, the possibility of the fluid being solid
    is simply ignored.

    Examples
    --------
    >>> identify_phase(T=280, P=101325, Tm=273.15, Psat=991)
    'l'
    '''
    if Tm and T <= Tm:
        return 's'
    elif Tc and T >= Tc:
        # No special return value for the critical point
        return 'g'
    elif Psat:
        # Do not allow co-existence of phases; transition to 'l' directly under
        if P <= Psat:
            return 'g'
        elif P > Psat:
            return 'l'
    elif Tb:
        # Crude attempt to model phases without Psat
        # Treat Tb as holding from 90 kPa to 110 kPa
        if 9E4 < P < 1.1E5:
            if T < Tb:
                return  'l'
            else:
                return 'g'
        elif P > 1.1E5 and T <= Tb:
            # For the higher-pressure case, it is definitely liquid if under Tb
            # Above the normal boiling point, impossible to say - return None
            return 'l'
        else:
            return None
    else:
        return None


mixture_phase_methods = ['IDEAL_VLE', 'SUPERCRITICAL_T', 'SUPERCRITICAL_P', 'IDEAL_VLE_SUPERCRITICAL']


def identify_phase_mixture(T=None, P=None, zs=None, Tcs=None, Pcs=None,
                           Psats=None, CASRNs=None,
                           get_methods=False, method=None):  # pragma: no cover
    '''
    >>> identify_phase_mixture(T=280, P=5000., zs=[0.5, 0.5], Psats=[1400, 7000])
    ('l', [0.5, 0.5], None, 0)
    >>> identify_phase_mixture(T=280, P=3000., zs=[0.5, 0.5], Psats=[1400, 7000])
    ('two-phase', [0.7142857142857143, 0.2857142857142857], [0.33333333333333337, 0.6666666666666666], 0.5625000000000001)
    >>> identify_phase_mixture(T=280, P=800., zs=[0.5, 0.5], Psats=[1400, 7000])
    ('g', None, [0.5, 0.5], 1)
    >>> identify_phase_mixture(T=280, P=800., zs=[0.5, 0.5])
    (None, None, None, None)
    '''
    def list_methods():
        methods = []
        if Psats and none_and_length_check((Psats, zs)):
            methods.append('IDEAL_VLE')
        if Tcs and none_and_length_check([Tcs]) and all([T >= i for i in Tcs]):
            methods.append('SUPERCRITICAL_T')
        if Pcs and none_and_length_check([Pcs]) and all([P >= i for i in Pcs]):
            methods.append('SUPERCRITICAL_P')
        if Tcs and none_and_length_check([zs, Tcs]) and any([T > Tc for Tc in Tcs]):
            methods.append('IDEAL_VLE_SUPERCRITICAL')
        methods.append('NONE')
        return methods
    if get_methods:
        return list_methods()
    if not method:
        method = list_methods()[0]
    # This is the calculate, given the method section
    xs, ys, phase, V_over_F = None, None, None, None
    if method == 'IDEAL_VLE':
        Pdew = dew_at_T(zs, Psats)
        Pbubble = bubble_at_T(zs, Psats)
        if P >= Pbubble:
            phase = 'l'
            ys = None
            xs = zs
            V_over_F = 0
        elif P <= Pdew:
            phase = 'g'
            ys = zs
            xs = None
            V_over_F = 1
        elif Pdew < P < Pbubble:
            xs, ys, V_over_F = flash_ideal_basic(P, zs, Psats)
            phase = 'two-phase'
    elif method == 'SUPERCRITICAL_T':
        if all([T >= i for i in Tcs]):
            phase = 'g'
        else: # The following is nonsensical
            phase = 'two-phase'
    elif method == 'SUPERCRITICAL_P':
        if all([P >= i for i in Pcs]):
            phase = 'g'
        else: # The following is nonsensical
            phase = 'two-phase'
    elif method == 'IDEAL_VLE_SUPERCRITICAL':
        Psats = list(Psats)
        for i in range(len(Psats)):
            if not Psats[i] and Tcs[i] and Tcs[i] <= T:
                Psats[i] = 1E8
        Pdew = dew_at_T(zs, Psats)
        Pbubble = 1E99
        if P >= Pbubble:
            phase = 'l'
            ys = None
            xs = zs
            V_over_F = 0
        elif P <= Pdew:
            phase = 'g'
            ys = zs
            xs = None
            V_over_F = 1
        elif Pdew < P < Pbubble:
            xs, ys, V_over_F = flash_ideal_basic(P, zs, Psats)
            phase = 'two-phase'

    elif method == 'NONE':
        pass
    else:
        raise Exception('Failure in in function')
    return phase, xs, ys, V_over_F


def Pbubble_mixture(T=None, zs=None, Psats=None, CASRNs=None,
                   get_methods=False, method=None):  # pragma: no cover
    '''
    >>> Pbubble_mixture(zs=[0.5, 0.5], Psats=[1400, 7000])
    4200.0
    '''
    def list_methods():
        methods = []
        if none_and_length_check((Psats, zs)):
            methods.append('IDEAL_VLE')
        methods.append('NONE')
        return methods
    if get_methods:
        return list_methods()
    if not method:
        method = list_methods()[0]
    # This is the calculate, given the method section
    if method == 'IDEAL_VLE':
        Pbubble = bubble_at_T(zs, Psats)
    elif method == 'NONE':
        Pbubble = None
    else:
        raise Exception('Failure in in function')
    return Pbubble


def bubble_at_P(P, zs, vapor_pressure_eqns, fugacities=None, gammas=None):
    '''Calculates bubble point for a given pressure

    Parameters
    ----------
    P : float
        Pressure, [Pa]
    zs : list[float]
        Overall mole fractions of all species, [-]
    vapor_pressure_eqns : list[functions]
        Temperature dependent function for each specie, Returns Psat, [Pa]
    fugacities : list[float], optional
        fugacities of each species, defaults to list of ones, [-]
    gammas : list[float], optional
        gammas of each species, defaults to list of ones, [-]

    Returns
    -------
    Tbubble : float, optional
        Temperature of bubble point at pressure `P`, [K]

    '''

    def bubble_P_error(T):
        Psats = [VP(T) for VP in vapor_pressure_eqns]
        Pcalc = bubble_at_T(zs, Psats, fugacities, gammas)

        return P - Pcalc

    T_bubble = secant(bubble_P_error, 300)

    return T_bubble


def Pdew_mixture(T=None, zs=None, Psats=None, CASRNs=None,
                 get_methods=False, method=None):  # pragma: no cover
    '''
    >>> Pdew_mixture(zs=[0.5, 0.5], Psats=[1400, 7000])
    2333.3333333333335
    '''
    def list_methods():
        methods = []
        if none_and_length_check((Psats, zs)):
            methods.append('IDEAL_VLE')
        methods.append('NONE')
        return methods
    if get_methods:
        return list_methods()
    if not method:
        method = list_methods()[0]
    # This is the calculate, given the method section
    if method == 'IDEAL_VLE':
        Pdew = dew_at_T(zs, Psats)
    elif method == 'NONE':
        Pdew = None
    else:
        raise Exception('Failure in in function')
    return Pdew


def get_T_bub_est(P, zs, Tbs, Tcs, Pcs):
    T_bub_est = 0.7*sum([zi*Tci for zi, Tci in zip(zs, Tcs)])
    T_LO = T_HI = 0.0
    for i in range(1000):
        Ks = [Pc**((1./T_bub_est - 1./Tb)/(1./Tc - 1./Tb))/P for Tb, Tc, Pc in zip(Tbs, Tcs, Pcs)]
        y_bub_sum = sum([zi*Ki for zi, Ki in zip(zs, Ks)])
        if y_bub_sum < 1.:
            T_LO = T_bub_est
            y_LO_sum = y_bub_sum - 1.
            T_new = T_bub_est*1.1
        elif y_bub_sum > 1.:
            T_HI = T_bub_est
            y_HI_sum = y_bub_sum - 1.
            T_new = T_bub_est/1.1
        else:
            return T_bub_est
        if T_LO*T_HI > 0.0:
            T_new = (y_HI_sum*T_LO - y_LO_sum*T_HI)/(y_HI_sum - y_LO_sum)

        if abs(T_bub_est - T_new) < 1E-3:
            return T_bub_est
        elif abs(y_bub_sum - 1.) < 1E-5:
            return T_bub_est
        else:
            T_bub_est = T_new


def get_T_dew_est(P, zs, Tbs, Tcs, Pcs, T_bub_est=None):
    if T_bub_est is None:
        T_bub_est = get_T_bub_est(P, zs, Tbs, Tcs, Pcs)
    T_dew_est = 1.1*T_bub_est
    T_LO = T_HI = 0.0
    for i in range(10000):
        Ks = [Pc**((1./T_dew_est - 1./Tb)/(1./Tc - 1./Tb))/P for Tb, Tc, Pc in zip(Tbs, Tcs, Pcs)]
        x_dew_sum = sum([zi/Ki for zi, Ki in zip(zs, Ks)])
        if x_dew_sum < 1.:
            T_LO = T_dew_est
            x_LO_sum = x_dew_sum - 1.
            T_new = T_dew_est/1.1
        elif x_dew_sum > 1.:
            T_HI = T_dew_est
            x_HI_sum = x_dew_sum - 1.
            T_new = T_dew_est*1.1
        else:
            return T_dew_est
        if T_LO*T_HI > 0.0:
            T_new = (x_HI_sum*T_LO - x_LO_sum*T_HI)/(x_HI_sum - x_LO_sum)

        if abs(T_dew_est - T_new) < 1E-3:
            return T_dew_est
        elif abs(x_dew_sum - 1.) < 1E-5:
            return T_dew_est
        else:
            T_dew_est = T_new


def get_P_dew_est(T, zs, Tbs, Tcs, Pcs):
    def err(P):
        e = get_T_dew_est(P, zs, Tbs, Tcs, Pcs) - T
        return e
    try:
        return brenth(err, 1E-2, 1e8)
    except:
        return brenth(err, 1E-3, 1E12)


def get_P_bub_est(T, zs, Tbs, Tcs, Pcs):
    def err(P):
        e = get_T_bub_est(P, zs, Tbs, Tcs, Pcs) - T
        return e
    try:
        return brenth(err, 1E-2, 1e8)
    except:
        return brenth(err, 1E-3, 1E12)