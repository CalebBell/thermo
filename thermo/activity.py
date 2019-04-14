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

__all__ = ['K_value', 'Wilson_K_value', 'flash_wilson', 'flash_Tb_Tc_Pc',
           'Rachford_Rice_flash_error', 
           'Rachford_Rice_solution', 'Rachford_Rice_polynomial',
           'Rachford_Rice_solution_polynomial', 'Rachford_Rice_solution_LN2',
           'Rachford_Rice_solution2', 'Rachford_Rice_solutionN',
           'Rachford_Rice_flashN_f_jac', 'Rachford_Rice_flash2_f_jac',
           'Li_Johns_Ahmadi_solution', 'flash_inner_loop', 'NRTL', 'Wilson',
           'UNIQUAC', 'flash', 'dew_at_T',
           'bubble_at_T', 'identify_phase', 'mixture_phase_methods',
           'identify_phase_mixture', 'Pbubble_mixture', 'bubble_at_P',
           'Pdew_mixture']

from fluids.numerics import IS_PYPY, one_epsilon_larger, one_epsilon_smaller
from fluids.numerics import newton_system, roots_cubic, roots_quartic, horner, py_brenth as brenth, py_newton as newton, oscillation_checker # Always use this method for advanced features
from thermo.utils import exp, log
from thermo.utils import none_and_length_check
from thermo.utils import R
import numpy as np




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
        K_i = \frac{\phi_i^l}{\phi_i^v} = \frac{f_i^l}{f_i^v}

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
        if gamma:
            if phi_l:
                return gamma*Psat*phi_l*Poynting/(phi_g*P)
            return gamma*Psat*Poynting/P
        elif phi_l:
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

    Examples
    --------
    Ethane at 270 K and 76 bar:
        
    >>> Wilson_K_value(270.0, 7600000.0, 305.4, 4880000.0, 0.098)
    0.2963932297479371
    
    References
    ----------
    .. [1] Wilson, Grant M. "A Modified Redlich-Kwong Equation of State, 
       Application to General Physical Data Calculations." In 65th National 
       AIChE Meeting, Cleveland, OH, 1969.
    '''
    return Pc/P*exp((5.37*(1.0 + omega)*(1.0 - Tc/T)))


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
    T_MAX = 50000
    N = len(zs)
    cmps = range(N)
    # Assume T and P to begin with
    if T is not None and P is not None:
        P_inv, T_inv = 1.0/P, 1.0/T
        Ks = [Pcs[i]*P_inv*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]*T_inv))) for i in cmps]
        ans = (T, P) + flash_inner_loop(zs=zs, Ks=Ks)
        return ans
    if T is not None and VF == 0:
        P_bubble = 0.0
        for i in cmps:
            P_bubble += zs[i]*Pcs[i]*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]/T)))
        return flash_wilson(zs, Tcs, Pcs, omegas, T=T, P=P_bubble)
    if T is not None and VF == 1:
        P_dew = 0.
        for i in cmps:
            P_dew += zs[i]/(Pcs[i]*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]/T))))
        P_dew = 1./P_dew
#        print(P_dew)
        return flash_wilson(zs, Tcs, Pcs, omegas, T=T, P=P_dew)
    elif T is not None and VF is not None:
        # Solve for in the middle of Pdew
        P_low = flash_wilson(zs, Tcs, Pcs, omegas, T=T, VF=1)[1]
        P_high = flash_wilson(zs, Tcs, Pcs, omegas, T=T, VF=0)[1]
        info = []
        def to_solve(P):
            T_calc, P_calc, VF_calc, xs, ys = flash_wilson(zs, Tcs, Pcs, omegas, T=T, P=P)
            info[:] = T_calc, P_calc, VF_calc, xs, ys
            err = VF_calc - VF
            return err
        P = brenth(to_solve, P_low, P_high)
        return tuple(info)
    elif P is not None and VF == 1:
        def to_solve(T_guess):
            # Avoid some nasty unpleasantness in newton
            T_guess = abs(T_guess)
            P_dew = 0.
            for i in range(len(zs)):
                P_dew += zs[i]/(Pcs[i]*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]/T_guess))))
            P_dew = 1./P_dew
#            print(P_dew - P, T_guess)
            return P_dew - P
        # 2/3 average critical point
        T_guess = sum([.666*Tcs[i]*zs[i] for i in cmps])
        try:
            T_dew = abs(newton(to_solve, T_guess, maxiter=50, ytol=1e-2))
        except Exception as e:
#            print(e)
            T_dew = None
        if T_dew is None or T_dew > T_MAX*5.0: 
            # Went insanely high T, bound it with brenth
            T_low_guess = sum([.1*Tcs[i]*zs[i] for i in cmps])
            try:
                T_dew = brenth(to_solve, T_MAX, T_low_guess)
            except ValueError:
                raise Exception("Bisecting solver could not find a solution between %g K and %g K" %(T_MAX, T_low_guess))
        return flash_wilson(zs, Tcs, Pcs, omegas, T=T_dew, P=P)
    elif P is not None and VF == 0:
        def to_solve(T_guess):
            T_guess = abs(T_guess)
            P_bubble = 0.0
            for i in cmps:
                P_bubble += zs[i]*Pcs[i]*exp((5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]/T_guess)))
            return P_bubble - P
        # 2/3 average critical point
        T_guess = sum([.55*Tcs[i]*zs[i] for i in cmps])
        try:
            T_bubble = abs(newton(to_solve, T_guess, maxiter=50, ytol=1e-2))
        except Exception as e:
            T_bubble = None
        if T_bubble is None or T_bubble > T_MAX*5.0: 
            # Went insanely high T, bound it with brenth
            T_low_guess = sum([.1*Tcs[i]*zs[i] for i in cmps])
            try:
                T_bubble = brenth(to_solve, T_MAX, T_low_guess)
            except ValueError:
                raise Exception("Bisecting solver could not find a solution between %g K and %g K" %(T_MAX, T_low_guess))
            
        return flash_wilson(zs, Tcs, Pcs, omegas, T=T_bubble, P=P)
    elif P is not None and VF is not None:
        # Solve for in the middle of Pdew
        T_low = flash_wilson(zs, Tcs, Pcs, omegas, P=P, VF=1)[0]
        T_high = flash_wilson(zs, Tcs, Pcs, omegas, P=P, VF=0)[0]
#        print(T_low, T_high)
        info = []
        def err(T):
            T_calc, P_calc, VF_calc, xs, ys = flash_wilson(zs, Tcs, Pcs, omegas, T=T, P=P)
#            if abs(VF_calc) > 100: # Did not work at all
#                VF_calc = abs(VF_calc)
            info[:] = T_calc, P_calc, VF_calc, xs, ys
#            print(T, VF_calc - VF)
            return VF_calc - VF
        # Nasty function for tolerance; the default works and is good enough, could remove some
        # iterations in the fuure
        P = brenth(err, T_low, T_high, xtol=1e-14)
        return tuple(info)
    else:
        raise ValueError("Provide two of P, T, and VF")


def flash_Tb_Tc_Pc(zs, Tbs, Tcs, Pcs, T=None, P=None, VF=None):
    r'''PVT flash model using a model published in [1]_, which provides a PT
    surface  using only each compound's boiling temperature and critical 
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
            T_dew = abs(newton(to_solve, T_guess, maxiter=50, ytol=1e-2)) # , high=Tc_pseudo*3
        except:
            T_dew = None
        if T_dew is None or T_dew > T_MAX*5.0: 
            # Went insanely high T, bound it with brenth
            T_low_guess = sum([.1*Tcs[i]*zs[i] for i in cmps])
            checker = oscillation_checker(both_sides=True, minimum_progress=.05)
            try:
                T_dew = brenth(to_solve, T_MAX, T_low_guess)
            except ValueError:
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
            T_bubble = abs(newton(to_solve, T_guess, maxiter=50, ytol=1e-2)) # , high=Tc_pseudo*4
        except Exception as e:
#            print(e)
            checker = oscillation_checker(both_sides=True, minimum_progress=.05)
            T_bubble = None
        if T_bubble is None or T_bubble > T_MAX*5.0: 
            # Went insanely high T (or could not converge because went too high), bound it with brenth
            T_low_guess = 0.1*Tc_pseudo
            try:
                T_bubble = brenth(to_solve, T_MAX, T_low_guess)
            except ValueError:
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


def Rachford_Rice_polynomial_3(zs, Cs):
    z0, z1, z2 = zs
    C0, C1, C2 = Cs
    x0 = C0*z0
    x1 = C1*z1
    x2 = C2*z2
    a = C0*C1*C2*(z0 + z1 + z2)
    return [1.0,
            (C0*x1 + C0*x2 + C1*x0 + C1*x2 + C2*x0 + C2*x1)/a,
            (x0 + x1 + x2)/a]

def Rachford_Rice_polynomial_4(zs, Cs):
    z0, z1, z2, z3 = zs
    C0, C1, C2, C3 = Cs
    x0 = C0*z0
    x1 = C1*x0
    x2 = C1*z1
    x3 = C0*x2
    x4 = C2*z2
    x5 = C0*x4
    x6 = C3*z3
    x7 = C0*x6
    x8 = C2*x0
    x9 = C2*x2
    x10 = C1*x4
    x11 = C1*x6
    a = C0*C1*C2*C3*(z0 + z1 + z2 + z3)
    coeffs = [1.0,
              (C1*x5 + C1*x7 + C2*x1 + C2*x11 + C2*x3 + C2*x7 
               + C3*x1 + C3*x10 + C3*x3 + C3*x5 + C3*x8 + C3*x9)/a,
              (C2*x6 + C3*x0 + C3*x2 + C3*x4 + x1 + x10
               + x11 + x3 + x5 + x7 + x8 + x9)/a,
              (x0 + x2 + x4 + x6)/a]
    return coeffs

def Rachford_Rice_polynomial_5(zs, Cs):
    z0, z1, z2, z3, z4 = zs
    C0, C1, C2, C3, C4 = Cs
    x0 = C0*z0
    x1 = C1*x0
    x2 = C2*x1
    x3 = C1*z1
    x4 = C0*x3
    x5 = C2*x4
    x6 = C2*z2
    x7 = C0*x6
    x8 = C1*x7
    x9 = C3*z3
    x10 = C0*x9
    x11 = C1*x10
    x12 = C4*z4
    x13 = C0*x12
    x14 = C1*x13
    x15 = C3*x1
    x16 = C3*x4
    x17 = C2*x0
    x18 = C3*x17
    x19 = C3*x7
    x20 = C2*x10
    x21 = C2*x13
    x22 = C2*x3
    x23 = C3*x22
    x24 = C1*x6
    x25 = C3*x24
    x26 = C1*x9
    x27 = C2*x26
    x28 = C1*x12
    x29 = C2*x28
    x30 = C3*x0
    x31 = C3*x3
    x32 = C3*x6
    x33 = C2*x9
    x34 = C2*x12    
    a = C0*C1*C2*C3*C4*(z0 + z1 + z2 + z3 + z4)
    b = (C2*x11 + C2*x14 + C3*x14 + C3*x2 + C3*x21 + C3*x29 + C3*x5 + C3*x8
         + C4*x11 + C4*x15 + C4*x16 + C4*x18 + C4*x19 + C4*x2 + C4*x20 + C4*x23
         + C4*x25 + C4*x27 + C4*x5 + C4*x8)/a
    c = (C3*x13 + C3*x28 + C3*x34 + C4*x1 + C4*x10 + C4*x17 + C4*x22 + C4*x24 
         + C4*x26 + C4*x30 + C4*x31 + C4*x32 + C4*x33 + C4*x4 + C4*x7 + x11
         + x14 + x15 + x16 + x18 + x19 + x2 + x20 + x21 + x23 + x25 + x27 
         + x29 + x5 + x8)/a
    d = (C3*x12 + C4*x0 + C4*x3 + C4*x6 + C4*x9 + x1 + x10 + x13 + x17 + x22 
         + x24 + x26 + x28 + x30 + x31 + x32 + x33 + x34 + x4 + x7)/a
    e = (x0 + x12 + x3 + x6 + x9)/a
    return [1.0, b, c, d, e]

    
_RR_poly_idx_cache = {}
def _Rachford_Rice_polynomial_coeff(value, zs, Cs, N):
    global_list = []
    # This part can be cached, so its performance implication is small
    # I believe for high-N, this is causing out of memory errors
    # However, even when using yield, still out-of-memories
    def better_recurse(prev_value, max_value, working=None):
        if working is None:
            working = []
        for i in range(prev_value, max_value):
            if N == max_value:
#                yield working + [i]
#                return
                global_list.append(working + [i])
            else:
                better_recurse(i + 1, max_value + 1, working + [i])
#        return global_list
    
    if (value, N) in _RR_poly_idx_cache:
        global_list = _RR_poly_idx_cache[(value, N)]
    else:
        better_recurse(0, value)
        _RR_poly_idx_cache[(value, N)] = global_list
    
#     zs_sum_mat = []
#     Cs_inv_mat = []
#     for i in range(N):
#         Cs_inv_list = []
#         zs_sum_list = []
#         for j in range(N):
#             if j > i:
#                 Cs_inv_list.append(None)
#                 zs_sum_list.append(None)
#             else:
#                 Cs_inv_list.append(Cs[i]*Cs[j])
#                 zs_sum_list.append(zs[i] + zs[j])
#         Cs_inv_mat.append(Cs_inv_list)
#         zs_sum_mat.append(zs_sum_list)
#     print(Cs_inv_mat)
#     Cs_inv_mat = [[Ci*Cj for Cj in Cs] for Ci in Cs]
#     zs_sum_mat = [[zi + zj for zj in zs] for zi in zs]

    # If there were some way to use cse this might work much faster
    c = 0.0
    for idxs in global_list:
        C_msum = 1.0
        z_tot = 1.0
        for i in idxs:
            z_tot -= zs[i]
            C_msum *= Cs[i]
#         print(z_tot, C_msum, idxs)
#         C_msum = 1.0
#         z_tot = 1.0
#         l_idxs = len(idxs)
# #         # j is always larger than i only need half the matrixes
#         for i in range(0, l_idxs-1, 2):
#             i, j = idxs[i], idxs[i+1]
# #             print(j, i)
# #             print(j > i)
#             z_tot -= zs_sum_mat[j][i]
#             C_msum *= Cs_inv_mat[j][i]
#         if l_idxs & 1:
#             j = idxs[-1]
#             z_tot -= zs[j]
#             C_msum *= Cs[j]
        c += z_tot*C_msum
    return c


def Rachford_Rice_polynomial(zs, Ks):
    r'''Transforms the Rachford-Rice equation into a polynomial and returns
    its coefficients.
    A spelled-out solution is used for N from 2 to 5, derived with SymPy and
    optimized with the common sub expression approach.
    
    .. warning:: For large numbers of components (>20) this model performs 
       terribly, though with future optimization it may be possible to have 
       better performance.
    
    .. math::
        \sum_{i=1}^N z_i C_i\left[ \Pi_{j\ne i}^N \left(1 + \frac{V}{F} 
        C_j\right)\right] = 0

    .. math::        
        C_i = K_i - 1.0
        
    Once the above calculation is performed, it must be rearranged into 
    polynomial form.
    
    Parameters
    ----------
    zs : list[float]
        Overall mole fractions of all species, [-]
    Ks : list[float]
        Equilibrium K-values, [-]
        
    Returns
    -------
    coeffs : float
        Coefficients, with earlier coefficients corresponding to higher powers,
        [-]

    Notes
    -----
    Explicit calculations for any degree can be obtained with SymPy, changing
    N as desired:
        

    >>> from sympy import * # doctest: +SKIP
    >>> N = 4
    >>> Cs = symbols('C0:' + str(N)) # doctest: +SKIP
    >>> zs = symbols('z0:' + str(N)) # doctest: +SKIP
    >>> alpha = symbols('alpha') # doctest: +SKIP
    >>> tot = 0
    >>> for i in range(N): # doctest: +SKIP
    ...     mult_sum = 1
    >>> for j in range(N): # doctest: +SKIP
    ...     if j != i:
    ...         mult_sum *= (1 + alpha*Cs[j])
    ...     tot += zs[i]*Cs[i]*mult_sum 
    
    poly_expr = poly(expand(tot), alpha)
    coeff_list = poly_expr.all_coeffs()
    cse(coeff_list, optimizations='basic')
    
    [1]_ suggests a matrix-math based approach for solving the model, but that
    has not been performed here. [1]_ also has explicit equations for 
    up to N = 7 to derive the coefficients.
    
    The general form was derived to be slightly different than that in [1]_,
    but is confirmed to also be correct as it matches other methods for solving
    the Rachford-Rice equation.
    
    The first coefficient is always 1.
    
    The approach is also discussed in [3]_, with one example.
    
    Examples
    --------
    >>> Rachford_Rice_polynomial(zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    [1.0, -3.692652996676083, 2.073518878815093]

    References
    ----------
    .. [1] Weigle, Brett D. "A Generalized Polynomial Form of the Objective 
       Function in Flash Calculations." Pennsylvania State University, 1992.
    .. [2] Warren, John H. "Explicit Determination of the Vapor Fraction in 
       Flash Calculations." Pennsylvania State University, 1991.
    .. [3] Monroy-Loperena, Rosendo, and Felipe D. Vargas-Villamil. "On the
       Determination of the Polynomial Defining of Vapor-Liquid Split of 
       Multicomponent Mixtures." Chemical Engineering Science 56, no. 20 
       (October 1, 2001): 5865-68.
       https://doi.org/10.1016/S0009-2509(01)00267-6.
    '''
    N = len(zs)
    Cs = [Ki - 1.0 for Ki in Ks]
    if N == 2:
        C0, C1 = Cs
        z0, z1 = zs
        return [1.0, (C0*z0 + C1*z1)/(C0*C1*(z0 + z1))]
    elif N == 3:
        return Rachford_Rice_polynomial_3(zs, Cs)
    elif N == 4:
        return Rachford_Rice_polynomial_4(zs, Cs)
    elif N == 5:
        return Rachford_Rice_polynomial_5(zs, Cs)
    
    
    Cs_inv = [1.0/Ci for Ci in Cs]
    coeffs = [1.0]
    
#    if N > 2:
    c = 0.0
    for i in range(0, N):
        c += (1.0 - zs[i])*Cs_inv[i]
    coeffs.append(c)
    
    coeffs.extend([_Rachford_Rice_polynomial_coeff(v, zs, Cs_inv, N) 
                    for v in range(N-1, 2, -1)])
        
    c = 0.0
    for i in range(0, N):
        C_sumprod = 1.0
        for j, C in enumerate(Cs_inv):
            if j != i:
                C_sumprod *= C
        c += zs[i]*C_sumprod
    coeffs.append(c)
    return coeffs


def Rachford_Rice_solution_polynomial(zs, Ks):
    r'''Solves the Rachford-Rice equation by transforming it into a polynomial,
    and then either analytically calculating the roots, or, using the known 
    range the correct root is in, numerically solving for the correct
    polynomial root. The analytical solutions are used for N from 2 to 4.
    
    Uses the method proposed in [2]_ to obtain an initial guess when solving
    the polynomial for the root numerically.

    .. math::
        \sum_i \frac{z_i(K_i-1)}{1 + \frac{V}{F}(K_i-1)} = 0
        
    .. warning:: : Using this function with more than 20 components is likely  
       to crash Python! This model does not work well with many components!
    
    This method, developed first in [3]_ and expanded in [1]_, is clever but
    of little use for large numbers of components.

    Parameters
    ----------
    zs : list[float]
        Overall mole fractions of all species, [-]
    Ks : list[float]
        Equilibrium K-values, [-]
        
    Returns
    -------
    V_over_F : float
        Vapor fraction solution [-]
    xs : list[float]
        Mole fractions of each species in the liquid phase, [-]
    ys : list[float]
        Mole fractions of each species in the vapor phase, [-]

    Notes
    -----
    This approach has mostly been ignored by academia, despite some of its 
    advantages.
    
    The initial guess is the average of the following, as described in [2]_.

    .. math::
        \left(\frac{V}{F}\right)_{min} = \frac{(K_{max}-K_{min})z_{of\;K_{max}}
        - (1-K_{min})}{(1-K_{min})(K_{max}-1)}

        \left(\frac{V}{F}\right)_{max} = \frac{1}{1-K_{min}}

    If the `newton` method does not converge, a bisection method (brenth) is
    used instead. However, it is somewhat slower, especially as newton will
    attempt 50 iterations before giving up.
    
    
    Examples
    --------
    >>> Rachford_Rice_solution_polynomial(zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    (0.6907302627738541, [0.3394086969663436, 0.3650560590371706, 0.29553524399648573], [0.571903654388289, 0.27087159580558057, 0.1572247498061304])

    References
    ----------
    .. [1] Weigle, Brett D. "A Generalized Polynomial Form of the Objective 
       Function in Flash Calculations." Pennsylvania State University, 1992.
    .. [2] Li, Yinghui, Russell T. Johns, and Kaveh Ahmadi. "A Rapid and Robust
       Alternative to Rachford-Rice in Flash Calculations." Fluid Phase
       Equilibria 316 (February 25, 2012): 85-97.
       doi:10.1016/j.fluid.2011.12.005.
    .. [3] Warren, John H. "Explicit Determination of the Vapor Fraction in 
       Flash Calculations." Pennsylvania State University, 1991.
    '''
    N = len(zs)
    if N > 30:
        raise ValueError("Unlikely to solve")
    poly = Rachford_Rice_polynomial(zs, Ks)

    Kmin = min(Ks)
    Kmax = max(Ks)
    z_of_Kmax = zs[Ks.index(Kmax)]
    V_over_F_min = ((Kmax-Kmin)*z_of_Kmax - (1.- Kmin))/((1.- Kmin)*(Kmax- 1.))
    V_over_F_max = 1./(1.-Kmin)
    
    if V_over_F_min < 0.0:
        V_over_F_min *= one_epsilon_larger
    else:
        V_over_F_min *= one_epsilon_smaller

    if V_over_F_max < 0.0:
        V_over_F_max *= one_epsilon_larger
    else:
        V_over_F_max *= one_epsilon_smaller
    

    if N > 5:
        # For safety, obtain limits of K 
        x0 = 0.5*(V_over_F_min + V_over_F_max)
        def err(VF):
            return horner(poly, VF)
        
        try:
            V_over_F = newton(err, x0)
            if V_over_F < V_over_F_min or V_over_F > V_over_F_max:
                raise ValueError("Newton converged to another root")
        except:
            V_over_F = brenth(err, V_over_F_min, V_over_F_max)
    else:
        if N == 4:
            coeffs = poly
        elif N == 3:
            coeffs = (0.0,) + tuple(poly)
        elif N == 2:
            coeffs = (0.0, 0.0) + tuple(poly)
        if N == 5:
            roots = roots_quartic(*poly)
        else:
            roots = roots_cubic(*coeffs)
        if N == 2:
            V_over_F = roots[0]
        else:
            V_over_F = None
            for root in roots:
                if abs(root.imag) < 1e-9 and V_over_F_min <= root.real <= V_over_F_max:
#                if root.imag == 0.0 and V_over_F_min <= root <= V_over_F_max:
                    V_over_F = root.real
                    break
            if V_over_F is None:
                raise ValueError("Bad roots", roots, "Root should be between V_over_F_min and V_over_F_max")
    
    xs = [zi/(1.+V_over_F*(Ki-1.)) for zi, Ki in zip(zs, Ks)]
    ys = [Ki*xi for xi, Ki in zip(xs, Ks)]
    return V_over_F, xs, ys
        

def Rachford_Rice_flash_error(V_over_F, zs, Ks):
    r'''Calculates the objective function of the Rachford-Rice flash equation.
    This function should be called by a solver seeking a solution to a flash
    calculation. The unknown variable is `V_over_F`, for which a solution
    must be between 0 and 1.

    .. math::
        \sum_i \frac{z_i(K_i-1)}{1 + \frac{V}{F}(K_i-1)} = 0

    Parameters
    ----------
    V_over_F : float
        Vapor fraction guess [-]
    zs : list[float]
        Overall mole fractions of all species, [-]
    Ks : list[float]
        Equilibrium K-values, [-]

    Returns
    -------
    error : float
        Deviation between the objective function at the correct V_over_F
        and the attempted V_over_F, [-]

    Notes
    -----
    The derivation is as follows:

    .. math::
        F z_i = L x_i + V y_i

        x_i = \frac{z_i}{1 + \frac{V}{F}(K_i-1)}

        \sum_i y_i = \sum_i K_i x_i = 1

        \sum_i(y_i - x_i)=0

        \sum_i \frac{z_i(K_i-1)}{1 + \frac{V}{F}(K_i-1)} = 0

    Examples
    --------
    >>> Rachford_Rice_flash_error(0.5, zs=[0.5, 0.3, 0.2],
    ... Ks=[1.685, 0.742, 0.532])
    0.04406445591174976

    References
    ----------
    .. [1] Rachford, H. H. Jr, and J. D. Rice. "Procedure for Use of Electronic
       Digital Computers in Calculating Flash Vaporization Hydrocarbon
       Equilibrium." Journal of Petroleum Technology 4, no. 10 (October 1,
       1952): 19-3. doi:10.2118/952327-G.
    '''
    return sum([zi*(Ki-1.)/(1.+V_over_F*(Ki-1.)) for Ki, zi in zip(Ks, zs)])


def Rachford_Rice_flashN_f_jac(betas, ns, Ks):
    N = len(betas)
    betas = [float(i) for i in betas]
    Fs = [0.0]*N
    dFs_dBetas = [[0.0]*N for i in range(N)]
    
    Ksm1 = [[i-1.0 for i in Ks_i] for Ks_i in Ks]
    zsKsm1 = [[zi*Ksim1 for zi, Ksim1 in zip(ns, Ksm1i)] for Ksm1i in Ksm1]
    
    for i, zi in enumerate(ns):
        denom = 1.0
        for j, beta_i in enumerate(betas):
            denom += beta_i*Ksm1[j][i]
        denom_inv = 1.0/denom
        denom_inv2 = denom_inv*denom_inv
        
        for j in range(N):
            Fs[j] += zsKsm1[j][i]*denom_inv

        for j in range(N):
            for k in range(N):
                if k <= j:
                    term = zsKsm1[j][i]*Ksm1[k][i]*denom_inv2
                    if k == j:
                        dFs_dBetas[k][j] -= term
                    else:
                        dFs_dBetas[k][j] -= term
                        dFs_dBetas[j][k] -= term

    return Fs, dFs_dBetas


def Rachford_Rice_flash2_f_jac(betas, zs, Ks_y, Ks_z):
    # In a more clever system like RR 2, can compute entire numerators before hand.
    beta_y, beta_z = float(betas[0]), float(betas[1])
    F0 = 0.0
    F1 = 0.0
    dF0_dy = 0.0
    dF0_dz = 0.0
    dF1_dz = 0.0

    for zi, Ky_i, Kz_i in zip(zs, Ks_y, Ks_z):
        Ky_m1 = (Ky_i - 1.0)
        ziKy_m1 = zi*Ky_m1

        Kz_m1 = (Kz_i - 1.0)
        ziKz_m1 = zi*Kz_m1

        denom_inv = 1.0/(1.0 + beta_y*Ky_m1 + beta_z*Kz_m1) # same in all
        delta_F0 = ziKy_m1*denom_inv
        delta_F1 = ziKz_m1*denom_inv

        F0 += delta_F0
        F1 += delta_F1
        dF0_dy -= delta_F0*Ky_m1*denom_inv
        dF0_dz -= delta_F0*Kz_m1*denom_inv
        dF1_dz -= delta_F1*Kz_m1*denom_inv

    return [F0, F1], [[dF0_dy, dF0_dz], [dF0_dz, dF1_dz]]


def Rachford_Rice_valid_solution_naive(ns, betas, Ks, limit_betas=False):
    if limit_betas:
        for beta in betas:
            if beta < 0.0 or beta > 1.0:
                return False
    
    for i, ni in enumerate(ns):
        sum_critiria = 1.0
        for j, beta_i in enumerate(betas):
            sum_critiria += beta_i*(Ks[j][i] - 1.0)
        if sum_critiria < 0.0:
            # Will result in negative composition for xi, yi, and zi
            return False
    return True


def Rachford_Rice_solutionN(ns, Ks, betas):
    r'''Solves the (phases -1) objectives functions of the Rachford-Rice flash 
    equation for an N-phase system. Initial guesses are required for all phase 
    fractions except the last`. The Newton method is used, with an
    analytical Jacobian.
        
    Parameters
    ----------
    ns : list[float]
        Overall mole fractions of all species, [-]
    Ks : list[list[float]]
        Equilibrium K-values of all phases phase to the `x` phase, [-]
    betas : list[float]
        Phase fraction guesses for the first N - 1 phases, [-]
        
    Returns
    -------
    betas : list[float]
        Phase fractions of the first N-1 phases, [-]
    compositions : list[list[float]]
        Mole fractions of each species in each phase, [-]
    
    Notes
    -----
    Besides testing with three phases, only one 5 phase problem has been solved
    with this algorithm, from [1]_.
    
    Examples
    --------
    >>> ns = [0.204322076984, 0.070970999150, 0.267194323384, 0.296291964579, 0.067046080882, 0.062489248292, 0.031685306730]
    >>> Ks_y = [1.23466988745, 0.89727701141, 2.29525708098, 1.58954899888, 0.23349348597, 0.02038108640, 1.40715641002]
    >>> Ks_z = [1.52713341421, 0.02456487977, 1.46348240453, 1.16090546194, 0.24166289908, 0.14815282572, 14.3128010831]
    >>> Rachford_Rice_solutionN(ns, [Ks_y, Ks_z], [.1, .6])
    ... [0.6868328915094766, 0.06019424397668606], [[0.1712804659711611, 0.08150738616425436, 0.1393433949193188, 0.20945175387703213, 0.15668977784027893, 0.22650123851718007, 0.015225982711774586], [0.21147483364299702, 0.07313470386530294, 0.31982891387635903, 0.33293382568889657, 0.036586042443791586, 0.004616341311925655, 0.02142533917172731], [0.26156812278601893, 0.00200221914149187, 0.20392660665189805, 0.2431536850887592, 0.03786610596908295, 0.03355679851539993, 0.21792646184834918]])
    
    References
    ----------
    .. [1] Gao, Ran, Xiaolong Yin, and Zhiping Li. "Hybrid Newton-Successive 
       Substitution Method for Multiphase Rachford-Rice Equations." Entropy 20,
       no. 6 (June 2018): 452. https://doi.org/10.3390/e20060452.
    '''
    limit_betas = False
    def new_betas(betas, d_betas, damping):
        betas_test = [beta_i + d_beta*damping for beta_i, d_beta in zip(betas, d_betas)]
        for i in range(20):
            is_valid = Rachford_Rice_valid_solution_naive(ns, betas_test, Ks, limit_betas=limit_betas)
            if is_valid:
                break
            
            damping = 0.5*damping
            for i in range(len(betas_test)):
                betas_test[i] = betas_test[i]  + d_betas[i]*damping
#            betas_test = [beta_i + d_beta*damping for beta_i, d_beta in zip(betas, d_betas)]
#            print('out of bounds', damping, betas_test)
        if not is_valid:
            raise ValueError("Should never happen - N phase RR still out of bounds after 20 iterations")
        return betas_test
    
    if not Rachford_Rice_valid_solution_naive(ns, betas, Ks, limit_betas=limit_betas):
        raise ValueError("Initial guesses will not lead to convergence")
    
    betas, iter = newton_system(Rachford_Rice_flashN_f_jac, jac=True, 
                                           x0=betas, args=(ns, Ks),
                                           ytol=1e-14, damping_func=new_betas)
#    print(betas, iter, 'current progress')
    comps = [[]]
    for i, ni in enumerate(ns):
        denom = 1.0
        for j, beta_i in enumerate(betas):
            denom += beta_i*(Ks[j][i]-1.0)
        denom_inv = 1.0/denom
        comps[0].append(ni*denom_inv)
    
    for Ks_j in Ks:
        comp = [Ki*xi for Ki, xi in zip(Ks_j, comps[0])]
        comps.append(comp)

    return betas, comps


def Rachford_Rice_solution2(ns, Ks_y, Ks_z, beta_y=0.5, beta_z=1e-6):
    r'''Solves the two objective functions of the Rachford-Rice flash equation
    for a three-phase system. Initial guesses are required for both phase 
    fractions, `beta_y` and `beta_z`. The Newton method is used, with an
    analytical Jacobian.

    .. math::
        F_0 = \sum_i \frac{z_i (K_y -1)}{1 + \beta_y(K_y-1) + \beta_z(K_z-1)} = 0

    .. math::
        F_1 = \sum_i \frac{z_i (K_z -1)}{1 + \beta_y(K_y-1) + \beta_z(K_z-1)} = 0
        
    Parameters
    ----------
    ns : list[float]
        Overall mole fractions of all species (would be `zs` except that is
        conventially used for one of the three phases), [-]
    Ks_y : list[float]
        Equilibrium K-values of `y` phase to `x` phase, [-]
    Ks_z : list[float]
        Equilibrium K-values of `z` phase to `x` phase, [-]
    beta_y : float, optional
        Initial guess for `y` phase (between 0 and 1), [-]
    beta_z : float, optional
        Initial guess for `z` phase (between 0 and 1), [-]
        
    Returns
    -------
    beta_y : float
        Phase fraction of `y` phase, [-]
    beta_z : float
        Phase fraction of `z` phase, [-]
    xs : list[float]
        Mole fractions of each species in the `x` phase, [-]
    ys : list[float]
        Mole fractions of each species in the `y` phase, [-]
    zs : list[float]
        Mole fractions of each species in the `z` phase, [-]
    
    Notes
    -----
    The elements of the Jacobian are calculated as follows:

    .. math::
        \frac{\partial F_0}{\partial \beta_y} = \sum_i \frac{-z_i (K_y -1)^2}
        {\left(1 + \beta_y(K_y-1) + \beta_z(K_z-1)\right)^2}

    .. math::
        \frac{\partial F_1}{\partial \beta_z} = \sum_i \frac{-z_i (K_z -1)^2}
        {\left(1  + \beta_y(K_y-1) + \beta_z(K_z-1)\right)^2}

    .. math::
        \frac{\partial F_1}{\partial \beta_y} = \sum_i \frac{\partial F_0}
        {\partial \beta_z}  = \frac{-z_i (K_z -1)(K_y - 1)}{\left(1 
        + \beta_y(K_y-1) + \beta_z(K_z-1)\right)^2}
        
    In general, the solution which Newton's method converges to may not be the 
    desired one, so further constraints are required.
    
    Okuno's method in [1]_ provides a polygonal region where the correct answer
    lies. It has not been implemented.
    
    The Leibovici and Neoschil method [4]_ provides a method to compute/update 
    the damping parameter, which is suposed to ensure convergence. It claims to
    be able to calculate the maximum damping factor for Newton's method, if it
    tries to go out of bounds.
    
    A custom region which is believed to be the same as that of Okuno is 
    implemented instead - the region which ensures positive compositions for
    all compounds in all phases, but does not restrict the phase fractions to
    be between 0 and 1 or even positive.
    
    With the convergence restraint, it is believed if a solution lies within
    (0, 1) for both variables, the correct solution will be converged to so long
    as the initial guesses are within the correct region.
        
    Examples
    --------
    >>> ns = [0.204322076984, 0.070970999150, 0.267194323384, 0.296291964579, 0.067046080882, 0.062489248292, 0.031685306730]
    >>> Ks_y = [1.23466988745, 0.89727701141, 2.29525708098, 1.58954899888, 0.23349348597, 0.02038108640, 1.40715641002]
    >>> Ks_z = [1.52713341421, 0.02456487977, 1.46348240453, 1.16090546194, 0.24166289908, 0.14815282572, 14.3128010831]
    >>> Rachford_Rice_solution2(ns, Ks_y, Ks_z, beta_y=.1, beta_z=.6)
    ... (0.6868328915094766, 0.06019424397668606, [0.1712804659711611, 0.08150738616425436, 0.1393433949193188, 0.20945175387703213, 0.15668977784027893, 0.22650123851718007, 0.015225982711774586], [0.21147483364299702, 0.07313470386530294, 0.31982891387635903, 0.33293382568889657, 0.036586042443791586, 0.004616341311925655, 0.02142533917172731], [0.26156812278601893, 0.00200221914149187, 0.20392660665189805, 0.2431536850887592, 0.03786610596908295, 0.03355679851539993, 0.21792646184834918])
    
    References
    ----------
    .. [1] Okuno, Ryosuke, Russell Johns, and Kamy Sepehrnoori. "A New 
       Algorithm for Rachford-Rice for Multiphase Compositional Simulation." 
       SPE Journal 15, no. 02 (June 1, 2010): 313-25. 
       https://doi.org/10.2118/117752-PA.
    .. [2] Li, Zhidong, and Abbas Firoozabadi. "Initialization of Phase 
       Fractions in Rachford–Rice Equations for Robust and Efficient 
       Three-Phase Split Calculation." Fluid Phase Equilibria 332 (October 25,
       2012): 21-27. https://doi.org/10.1016/j.fluid.2012.06.021.
    .. [3] Gao, Ran, Xiaolong Yin, and Zhiping Li. "Hybrid Newton-Successive 
       Substitution Method for Multiphase Rachford-Rice Equations." Entropy 20,
       no. 6 (June 2018): 452. https://doi.org/10.3390/e20060452.
    .. [4] Leibovici, Claude F., and Jean Neoschil. "A Solution of 
       Rachford-Rice Equations for Multiphase Systems." Fluid Phase Equilibria 
       112, no. 2 (December 1, 1995): 217-21. 
       https://doi.org/10.1016/0378-3812(95)02797-I.
    '''
    limit_betas = False
    Ks = [Ks_y, Ks_z]
    def new_betas(betas, d_betas, damping):
        betas_test = [beta_i + d_beta*damping for beta_i, d_beta in zip(betas, d_betas)]
        for i in range(20):
            is_valid = Rachford_Rice_valid_solution_naive(ns, betas_test, Ks, limit_betas=limit_betas)
            if is_valid:
                break
            
            damping = 0.5*damping
            betas_test = [beta_i + d_beta*damping for beta_i, d_beta in zip(betas, d_betas)]
#            print('out of bounds', damping, betas_test)
        if not is_valid:
            raise ValueError("Should never happen - 3 phase RR still out of bounds after 20 iterations")
        return betas_test
    
    if not Rachford_Rice_valid_solution_naive(ns, [beta_y, beta_z], Ks, limit_betas=limit_betas):
        raise ValueError("Initial guesses will not lead to convergence")
    
    (beta_y, beta_z), iter = newton_system(Rachford_Rice_flashN_f_jac, jac=True, 
                                           x0=[beta_y, beta_z], args=(ns, Ks),
                                           ytol=1e-14, damping_func=new_betas)

#    (beta_y, beta_z), iter = newton_system(Rachford_Rice_flash2_f_jac, jac=True, 
#                                           x0=[beta_y, beta_z], args=(ns, Ks_y, Ks_z),
#                                           ytol=1e-14, damping_func=new_betas)

    xs = [zi/(1.+beta_y*(Ky-1.) + beta_z*(Kz-1.)) for Ky, Kz, zi in zip(Ks_y, Ks_z, ns)]
    ys = [Ky*xi for xi, Ky in zip(xs, Ks_y)]
    zs = [Kz*xi for xi, Kz in zip(xs, Ks_z)]
    return beta_y, beta_z, xs, ys, zs


def Rachford_Rice_solution(zs, Ks, fprime=False, fprime2=False,
                           limit=True):
    r'''Solves the objective function of the Rachford-Rice flash equation.
    Uses the method proposed in [2]_ to obtain an initial guess.

    .. math::
        \sum_i \frac{z_i(K_i-1)}{1 + \frac{V}{F}(K_i-1)} = 0

    Parameters
    ----------
    zs : list[float]
        Overall mole fractions of all species, [-]
    Ks : list[float]
        Equilibrium K-values, [-]
    fprime : bool, optional
        Whether or not to use the first derivative of the objective function
        in the solver (Newton-Raphson is used) or not (secant is used), [-]
    fprime2 : bool, optional
        Whether or not to use the second derivative of the objective function
        in the solver (parabolic Halley’s method is used if True) or not, [-]
        
    Returns
    -------
    V_over_F : float
        Vapor fraction solution [-]
    xs : list[float]
        Mole fractions of each species in the liquid phase, [-]
    ys : list[float]
        Mole fractions of each species in the vapor phase, [-]

    Notes
    -----
    The initial guess is the average of the following, as described in [2]_.

    .. math::
        \left(\frac{V}{F}\right)_{min} = \frac{(K_{max}-K_{min})z_{of\;K_{max}}
        - (1-K_{min})}{(1-K_{min})(K_{max}-1)}

        \left(\frac{V}{F}\right)_{max} = \frac{1}{1-K_{min}}

    Another algorithm for determining the range of the correct solution is
    given in [3]_; [2]_ provides a narrower range however. For both cases,
    each guess should be limited to be between 0 and 1 as they are often
    negative or larger than 1.

    .. math::
        \left(\frac{V}{F}\right)_{min} = \frac{1}{1-K_{max}}

        \left(\frac{V}{F}\right)_{max} = \frac{1}{1-K_{min}}

    If the `newton` method does not converge, a bisection method (brenth) is
    used instead. However, it is somewhat slower, especially as newton will
    attempt 50 iterations before giving up.
    
    In all benchmarks attempted, secant method provides better performance than
    Newton-Raphson or parabolic Halley’s method. This may not be generally
    true; but it is for Python and SciPy's implementation. They are implemented
    for benchmarking purposes.
    
    The first and second derivatives are:
        
    .. math::
        \frac{d \text{ obj}}{d \frac{V}{F}} = \sum_i \frac{-z_i(K_i-1)^2}
        {(1 + \frac{V}{F}(K_i-1))^2} 
        
        \frac{d^2 \text{ obj}}{d (\frac{V}{F})^2} = \sum_i \frac{2z_i(K_i-1)^3}
        {(1 + \frac{V}{F}(K_i-1))^3} 

    Examples
    --------
    >>> Rachford_Rice_solution(zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    (0.6907302627738542, [0.3394086969663436, 0.3650560590371706, 0.2955352439964858], [0.571903654388289, 0.27087159580558057, 0.15722474980613044])

    References
    ----------
    .. [1] Rachford, H. H. Jr, and J. D. Rice. "Procedure for Use of Electronic
       Digital Computers in Calculating Flash Vaporization Hydrocarbon
       Equilibrium." Journal of Petroleum Technology 4, no. 10 (October 1,
       1952): 19-3. doi:10.2118/952327-G.
    .. [2] Li, Yinghui, Russell T. Johns, and Kaveh Ahmadi. "A Rapid and Robust
       Alternative to Rachford-Rice in Flash Calculations." Fluid Phase
       Equilibria 316 (February 25, 2012): 85-97.
       doi:10.1016/j.fluid.2011.12.005.
    .. [3] Whitson, Curtis H., and Michael L. Michelsen. "The Negative Flash."
       Fluid Phase Equilibria, Proceedings of the Fifth International
       Conference, 53 (December 1, 1989): 51-71.
       doi:10.1016/0378-3812(89)80072-X.
    '''
    Kmin = min(Ks)
    Kmax = max(Ks)
    z_of_Kmax = zs[Ks.index(Kmax)]

    V_over_F_min = ((Kmax-Kmin)*z_of_Kmax - (1.- Kmin))/((1.- Kmin)*(Kmax- 1.))
    V_over_F_max = 1./(1.-Kmin)

    if limit:
        V_over_F_min2 = V_over_F_min if V_over_F_min > 0.0 else 0.0
        V_over_F_max2 = V_over_F_max if V_over_F_max < 1.0 else 1.0
    else:
        V_over_F_min2 = V_over_F_min
        V_over_F_max2 = V_over_F_max
#    print(V_over_F_min2, V_over_F_max2)
    
    x0 = (V_over_F_min2 + V_over_F_max2)*0.5
    
    K_minus_1 = [Ki - 1.0 for Ki in Ks]
    zs_k_minus_1 = [zi*Kim1 for zi, Kim1 in zip(zs, K_minus_1)]

    def err(V_over_F):
        diff =  sum([num/(1. + V_over_F*Kim1) for num, Kim1 in zip(zs_k_minus_1, K_minus_1)])
        return diff
    
    if fprime or fprime2:
        zs_k_minus_1_2 = [-first*Kim1 for first, Kim1 in zip(zs_k_minus_1, K_minus_1)]
    
    if fprime2:
        zs_k_minus_1_3 = [-2.0*second*Kim1 for second, Kim1 in zip(zs_k_minus_1_2, K_minus_1)]
        def err2(V_over_F):
            err0, err1, err2 = 0.0, 0.0, 0.0
            for num0, num1, num2, Kim1 in zip(zs_k_minus_1, zs_k_minus_1_2, zs_k_minus_1_3, K_minus_1):
                VF_kim1_1_inv = 1.0/(1. + V_over_F*Kim1)
                t2 = VF_kim1_1_inv*VF_kim1_1_inv
                err0 += num0*VF_kim1_1_inv
                err1 += num1*t2
                err2 += num2*t2*VF_kim1_1_inv
#            print(err0, err1, err2)
            return err0, err1, err2
    elif fprime:
        def err1(V_over_F):
            err0, err1 = 0.0, 0.0
            for num0, num1, Kim1 in zip(zs_k_minus_1, zs_k_minus_1_2, K_minus_1):
                VF_kim1_1_inv = 1.0/(1. + V_over_F*Kim1)
                err0 += num0*VF_kim1_1_inv
                err1 += num1*VF_kim1_1_inv*VF_kim1_1_inv
#            print(err0, V_over_F)
            return err0, err1

            
#    if not fprime and not fprime2:
#    def err(V_over_F):
##        print(V_over_F)
#        return sum([num/(1. + V_over_F*Kim1) for num, Kim1 in zip(zs_k_minus_1, K_minus_1)])
#    
#    if fprime or fprime2:
#        zs_k_minus_1_2 = [-first*Kim1 for first, Kim1 in zip(zs_k_minus_1, K_minus_1)]
#        def fprime_obj(V_over_F):
#            denom = [V_over_F*Kim1 + 1.0 for Kim1 in K_minus_1]
#            denom2 = [d1*d1 for d1 in denom]
#            return sum([num/deno for num, deno in zip(zs_k_minus_1_2, denom2)])
#        
#    if fprime2:
#        zs_k_minus_1_3 = [-2.0*second*Kim1 for second, Kim1 in zip(zs_k_minus_1_2, K_minus_1)]
#        def fprime2_obj(V_over_F):
#            denom = [V_over_F*Kim1 + 1.0 for Kim1 in K_minus_1]
#            denom2 = [d1*d1 for d1 in denom]
#            denom3 = [d2*d1 for d1, d2 in zip(denom, denom2)]
#            return sum([num/deno for num, deno in zip(zs_k_minus_1_3, denom3)])
        
    try:
        low, high = V_over_F_min*one_epsilon_larger, V_over_F_max*one_epsilon_smaller
        if fprime2:
            V_over_F = newton(err2, x0, ytol=1e-5, fprime=True, fprime2=True,
                              high=high, low=low)
        elif fprime:
            V_over_F = newton(err1, x0, ytol=1e-12, fprime=True, high=high,
                              low=low)
        else:
#            print(V_over_F_max, V_over_F_min)
            V_over_F = newton(err, x0, ytol=1e-5, high=high,
                              low=low)
        
#        assert V_over_F >= V_over_F_min2
#        assert V_over_F <= V_over_F_max2
    except Exception as e:
        V_over_F = brenth(err, low, high)
                
    xs = [zi/(1.+V_over_F*(Ki-1.)) for zi, Ki in zip(zs, Ks)]
    ys = [Ki*xi for xi, Ki in zip(xs, Ks)]
    return V_over_F, xs, ys


def Rachford_Rice_solution_numpy(zs, Ks, limit=True):
    '''Undocumented version of Rachford_Rice_solution which works with numpy
    instead. Can be up to 15x faster for cases of 30000+ compounds;
    typically 7-10 x faster.
    '''
    zs, Ks = np.array(zs), np.array(Ks)

    Kmin = Ks.min()
    Kmax = Ks.max()
    
    z_of_Kmax = zs[Ks == Kmax][0]

    V_over_F_min = ((Kmax-Kmin)*z_of_Kmax - (1.-Kmin))/((1.-Kmin)*(Kmax-1.))
    V_over_F_max = 1./(1.-Kmin)

    if V_over_F_min < 0.0:
        V_over_F_min *= one_epsilon_larger
    else:
        V_over_F_min *= one_epsilon_smaller

    if V_over_F_max < 0.0:
        V_over_F_max *= one_epsilon_larger
    else:
        V_over_F_max *= one_epsilon_smaller

#    , one_epsilon_larger

    if limit:
        # Range will cover a region which has the solution for 0 < VF < 1
        V_over_F_min2 = max(0., V_over_F_min)
        V_over_F_max2 = min(1., V_over_F_max)
    else:
        V_over_F_min2 = V_over_F_min
        V_over_F_max2 = V_over_F_max

    x0 = (V_over_F_min2 + V_over_F_max2)*0.5
    
    K_minus_1 = Ks - 1.0
    zs_k_minus_1 = zs*K_minus_1
    def err(V_over_F):
        err = float((zs_k_minus_1/(1.0 + V_over_F*K_minus_1)).sum())
        return err
    try:
        V_over_F = newton(err, x0, high=V_over_F_max*one_epsilon_smaller,
                          low=V_over_F_min*one_epsilon_larger, ytol=1e-5)
    except Exception as e:
        V_over_F = brenth(err, V_over_F_max*one_epsilon_smaller, V_over_F_min*one_epsilon_larger)
        
    xs = zs/(1.0 + V_over_F*K_minus_1)
    ys = Ks*xs
    return float(V_over_F), xs.tolist(), ys.tolist()


def Rachford_Rice_solution_LN2(zs, Ks, guess=None):
    r'''Solves the a objective function for the Rachford-Rice flash equation
    according to the Leibovici and Nichita (2010) transformation (method 2).
    This transformation makes the only zero of the function be the desired one.
    Consequently, higher-order methods may be used to solve this equation.
    Halley's (second derivative) method is found to be the best; typically 
    needing ~50% fewer iterations than the RR formulation with Secant method. 

    .. math::
        H(y) = \sum_i^n \frac{z_i}{\lambda - c_i} = 0
        
    .. math::
        \lambda = c_k + \frac{c_{k+1} - c_k}{1 + e^{-y}}
        
    .. math::
        c_i = \frac{1}{1-K_i}
        
    .. math::
        c_{k} = \left(\frac{V}{F}\right)_{min}
        
    .. math::
        c_{k+1} = \left(\frac{V}{F}\right)_{max}
        
    Note the two different uses of `c` in the above equation, confusingly
    given in [1]_. `lambda` is the vapor fraction.
    
    Once the equation has been solved for `y`, the vapor fraction can be
    calculated outside the solver.
        

    Parameters
    ----------
    zs : list[float]
        Overall mole fractions of all species, [-]
    Ks : list[float]
        Equilibrium K-values, [-]
    guess : float, optional
        Optional initial guess for vapor fraction, [-]
        
    Returns
    -------
    V_over_F : float
        Vapor fraction solution [-]
    xs : list[float]
        Mole fractions of each species in the liquid phase, [-]
    ys : list[float]
        Mole fractions of each species in the vapor phase, [-]

    Notes
    -----
    The initial guess is the average of the following, as described in [2]_.

    .. math::
        \left(\frac{V}{F}\right)_{min} = \frac{(K_{max}-K_{min})z_{of\;K_{max}}
        - (1-K_{min})}{(1-K_{min})(K_{max}-1)}

        \left(\frac{V}{F}\right)_{max} = \frac{1}{1-K_{min}}

    The first and second derivatives are derived with sympy as follows:
        
    >>> from sympy import * # doctest: +SKIP
    >>> VF_min, VF_max, ai, ci, y = symbols('VF_min, VF_max, ai, ci, y') # doctest: +SKIP
    >>> V_over_F = (VF_min + (VF_max - VF_min)/(1 + exp(-y))) # doctest: +SKIP
    >>> F = ai/(V_over_F - ci) # doctest: +SKIP
    >>> terms = [F, diff(F, y), diff(F, y, 2)] # doctest: +SKIP
    >>> cse(terms, optimizations='basic') # doctest: +SKIP
        
    Examples
    --------
    >>> Rachford_Rice_solution_LN2(zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    (0.6907302627738541, [0.3394086969663436, 0.3650560590371706, 0.29553524399648573], [0.571903654388289, 0.27087159580558057, 0.1572247498061304])

    References
    ----------
    .. [1] Leibovici, Claude F., and Dan Vladimir Nichita. "Iterative Solutions
       for ∑iaiλ-ci=1 Equations." Chemical Engineering Research and Design 88, 
       no. 5 (May 1, 2010): 602-5. https://doi.org/10.1016/j.cherd.2009.10.012.
    .. [2] Li, Yinghui, Russell T. Johns, and Kaveh Ahmadi. "A Rapid and Robust
       Alternative to Rachford-Rice in Flash Calculations." Fluid Phase
       Equilibria 316 (February 25, 2012): 85-97.
       doi:10.1016/j.fluid.2011.12.005.
    .. [3] Billingsley, D. S. "Iterative Solution for ∑iaiλ-ci Equations." 
       Computers & Chemical Engineering 26, no. 3 (March 15, 2002): 457-60. 
       https://doi.org/10.1016/S0098-1354(01)00767-0.
    '''
    Kmin = min(Ks)
    Kmax = max(Ks)
    z_of_Kmax = zs[Ks.index(Kmax)]
    
    one_m_Kmin = 1.0 - Kmin
    
    V_over_F_min = ((Kmax-Kmin)*z_of_Kmax - one_m_Kmin)/((one_m_Kmin)*(Kmax - 1.))
    V_over_F_max = 1./one_m_Kmin
    
    guess = 0.5*(V_over_F_min + V_over_F_max) if guess is None else guess
    cis = [1.0/(1.0 - Ki) for Ki in Ks]
    
    x0 = V_over_F_max - V_over_F_min
    def err(y):
        x1 = exp(-y)
        x3 = 1.0/(x1 + 1.0)
        x0x3 = x0*x3
        
        x6 = x0x3*x3
        x1x6 = x1*x6
        t50 = V_over_F_min + x0x3
        t51 = 1.0 - 2.0*x1*x3
        
        F0, dF0, ddF0 = 0.0, 0.0, 0.0
        for zi, ci in zip(zs, cis):
            x5 = 1.0/(t50 - ci)
            zix5 = zi*x5
            F0 += zix5
            # Func requires 1 division, 1 multiplication, 2 add
            # 1st Deriv adds 2 mult, 1 add
            # 3rd deriv adds 1 mult, 3 add
            
            x5x1x6 = x5*x1x6
            x7 = zix5*x5x1x6
            dF0 -= x7
            ddF0 += x7*(t51 + x5x1x6 + x5x1x6)      
        
        return F0, dF0, ddF0
    
    # Suggests guess V_over_F_min, not using
    guess = -log((V_over_F_max-guess)/(guess-V_over_F_min))
    
    # Should always converge - no poles
    try:
        V_over_F = newton(err, guess, fprime=True, fprime2=True, ytol=1e-8)
    except Exception as e:
        low, high = V_over_F_min + 1e-8, V_over_F_max - 1e-8
        low = -log((V_over_F_max-low)/(low-V_over_F_min))
        high = -log((V_over_F_max-high)/(high-V_over_F_min))
        
        V_over_F = brenth(lambda x: err(x)[0], low, high)
    V_over_F = (V_over_F_min + (V_over_F_max - V_over_F_min)/(1.0 + exp(-V_over_F)))
    
    xs = [zi/(1.+V_over_F*(Ki-1.)) for zi, Ki in zip(zs, Ks)]
    ys = [Ki*xi for xi, Ki in zip(xs, Ks)]
    return V_over_F, xs, ys


def Li_Johns_Ahmadi_solution(zs, Ks, guess=None):
    r'''Solves the objective function of the Li-Johns-Ahmadi flash equation.
    Uses the method proposed in [1]_ to obtain an initial guess.

    .. math::
        0 = 1 + \left(\frac{K_{max}-K_{min}}{K_{min}-1}\right)x_{max} + \sum_{i=2}^{n-1}
        \frac{K_i-K_{min}}{K_{min}-1}
        \left[\frac{z_i(K_{max}-1)x_{max}}{(K_i-1)z_{max} + (K_{max}-K_i)x_{max}}\right]
        
    Parameters
    ----------
    zs : list[float]
        Overall mole fractions of all species, [-]
    Ks : list[float]
        Equilibrium K-values, [-]

    Returns
    -------
    V_over_F : float
        Vapor fraction solution [-]
    xs : list[float]
        Mole fractions of each species in the liquid phase, [-]
    ys : list[float]
        Mole fractions of each species in the vapor phase, [-]

    Notes
    -----
    The initial guess is the average of the following, as described in [1]_.
    Each guess should be limited to be between 0 and 1 as they are often
    negative or larger than 1. `max` refers to the corresponding mole fractions
    for the species with the largest K value.

    .. math::
        \left(\frac{1-K_{min}}{K_{max}-K_{min}}\right)z_{max}\le x_{max} \le
        \left(\frac{1-K_{min}}{K_{max}-K_{min}}\right)

    If the `newton` method does not converge, a bisection method (brenth) is
    used instead. However, it is somewhat slower, especially as newton will
    attempt 50 iterations before giving up.

    This method does not work for problems of only two components.
    K values are sorted internally. Has not been found to be quicker than the
    Rachford-Rice equation.

    Examples
    --------
    >>> Li_Johns_Ahmadi_solution(zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    (0.6907302627738544, [0.33940869696634357, 0.3650560590371706, 0.2955352439964858], [0.5719036543882889, 0.27087159580558057, 0.15722474980613044])

    References
    ----------
    .. [1] Li, Yinghui, Russell T. Johns, and Kaveh Ahmadi. "A Rapid and Robust
       Alternative to Rachford-Rice in Flash Calculations." Fluid Phase
       Equilibria 316 (February 25, 2012): 85-97.
       doi:10.1016/j.fluid.2011.12.005.
    '''
    # Re-order both Ks and Zs by K value, higher coming first
    p = sorted(zip(Ks,zs), reverse=True)
    Ks_sorted, zs_sorted = [K for (K,z) in p], [z for (K,z) in p]


    # Largest K value and corresponding overall mole fraction
    k1 = Ks_sorted[0]
    z1 = zs_sorted[0]
    # Smallest K value
    kn = Ks_sorted[-1]

    x_max = (1. - kn)/(k1 - kn)
    x_min = x_max*z1

    if x_min < 0.0:
        x_min2 = 0.0
    else:
        x_min2 = x_min
        
    if x_max > 1.0:
        x_max2 = 1.0
    else:
        x_max2 = x_max
    
    length = len(zs)-1
    kn_m_1 = kn - 1.0
    k1_m_1 = (k1 - 1.0)
    kn_m_1_inv = 1.0/kn_m_1
    t1 = (k1 - kn)*kn_m_1_inv

    x_guess = (x_min2 + x_max2)*0.5 if guess is None else z1/(guess*k1_m_1 + 1.0)

    Ks_iter = Ks_sorted[1:length]
    zs_iter = zs_sorted[1:length]
    
    
    terms_2, terms_3 = [], []
    for ki, zi in zip(Ks_iter, zs_iter):
        term_1 = 1.0/((ki-kn)*kn_m_1_inv*zi*k1_m_1)
        terms_2.append((ki - 1.0)*z1*term_1)
        terms_3.append((k1 - ki)*term_1)
        
        
        
#    terms_1 = [(ki-kn)*kn_m_1_inv*zi*k1_m_1 for ki, zi in zip(Ks_iter, zs_iter)]
#    terms_2 = [(ki - 1.0)*z1 for ki in Ks_iter]
#    terms_3 = []
    
    def objective(x1):
        err = 1. + t1*x1
        for term2, term3 in zip(terms_2, terms_3):
            # evaluations: 2 mult, 1 div, 2 add
            err += x1/(term2 + term3*x1)
#        for ki, zi, term1, term2 in zip(Ks_iter, zs_iter, terms_1, terms_2):
#            # evaluations: 2 mult, 1 div, 2 add
#            err += term1*x1/(term2 + (k1-ki)*x1)
#        print(err, x1)
        return err

    try:
        x1 = newton(objective, x_guess, low=x_min, high=x_max, ytol=1e-13)
        # newton skips out of its specified range in some cases, finding another solution
        # Check for that with asserts, and use brenth if it did
        # Must also check that V_over_F is right.
        V_over_F = (z1 - x1)/(x1*k1_m_1)
#        print('V_over_F', V_over_F)
        
        assert x1 >= x_min
        assert x1 <= x_max
#        assert 0.0 <= V_over_F <= 1.0
    except Exception as e:
#        print('using bounding')
#        from fluids.numerics import py_bisect as bisect
#        x1 = bisect(objective, x_min, x_max, ytol=1e-12)
        x1 = brenth(objective, x_min, x_max) # , xtol=1e-12, rtol=0
        V_over_F = (-x1 + z1)/(x1*(k1 - 1.))
    xs = [zi/(1.+V_over_F*(Ki-1.)) for zi, Ki in zip(zs, Ks)]
    ys = [Ki*xi for xi, Ki in zip(xs, Ks)]
    return V_over_F, xs, ys


FLASH_INNER_ANALYTICAL = 'Analytical'
FLASH_INNER_SECANT = 'Rachford-Rice (Secant)'
FLASH_INNER_NR = 'Rachford-Rice (Newton-Raphson)'
FLASH_INNER_HALLEY = 'Rachford-Rice (Halley)'
FLASH_INNER_NUMPY = 'Rachford-Rice (NumPy)'
FLASH_INNER_LJA = 'Li-Johns-Ahmadi'
FLASH_INNER_POLY = 'Rachford-Rice (polynomial)'
FLASH_INNER_LN2 = 'Leibovici and Nichita 2'


flash_inner_loop_methods = [FLASH_INNER_ANALYTICAL, 
                            FLASH_INNER_SECANT,
                            FLASH_INNER_NR, FLASH_INNER_HALLEY,
                            FLASH_INNER_NUMPY, FLASH_INNER_LJA,
                            FLASH_INNER_POLY, FLASH_INNER_LN2]

def flash_inner_loop_list_methods(l):
    methods = []
    if l in (2, 3, 4, 5):
        methods.append(FLASH_INNER_ANALYTICAL)
    if l >= 10 and not IS_PYPY:
        methods.append(FLASH_INNER_NUMPY)
    if l >= 2:
        methods.extend([FLASH_INNER_LN2, FLASH_INNER_SECANT, FLASH_INNER_NR, FLASH_INNER_HALLEY])
        if l < 10 and not IS_PYPY:
            methods.append(FLASH_INNER_NUMPY)
    if l >= 3:
        methods.append(FLASH_INNER_LJA)
    if l < 20:
        methods.append(FLASH_INNER_POLY)
    return methods


def flash_inner_loop(zs, Ks, AvailableMethods=False, Method=None,
                     limit=True, guess=None, check=False):
    r'''This function handles the solution of the inner loop of a flash
    calculation, solving for liquid and gas mole fractions and vapor fraction
    based on specified overall mole fractions and K values. As K values are
    weak functions of composition, this should be called repeatedly by an outer
    loop. Will automatically select an algorithm to use if no Method is
    provided. Should always provide a solution.

    The automatic algorithm selection will try an analytical solution, and use
    the Rachford-Rice method if there are 6 or more components in the mixture.

    Parameters
    ----------
    zs : list[float]
        Overall mole fractions of all species, [-]
    Ks : list[float]
        Equilibrium K-values, [-]
    guess : float, optional
        Optional initial guess for vapor fraction, [-]
    check : bool, optional
        Whether or not to check the K values to ensure a positive-composition
        solution exists, [-]

    Returns
    -------
    V_over_F : float
        Vapor fraction solution [-]
    xs : list[float]
        Mole fractions of each species in the liquid phase, [-]
    ys : list[float]
        Mole fractions of each species in the vapor phase, [-]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain a solution with the given
        inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'Analytical',
        'Rachford-Rice (Secant)', 'Rachford-Rice (Newton-Raphson)', 
        'Rachford-Rice (Halley)', 'Rachford-Rice (NumPy)', 
        'Leibovici and Nichita 2', 'Rachford-Rice (polynomial)', and 
        'Li-Johns-Ahmadi'. All valid values are also held
        in the list `flash_inner_loop_methods`.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        a solution for the desired chemical, and will return methods instead of
        `V_over_F`, `xs`, and `ys`.

    Notes
    -----
    A total of eight methods are available for this function. They are:

        * 'Analytical', an exact solution derived with SymPy, applicable only
          only to mixtures of two, three, or four components
        * 'Rachford-Rice (Secant)', 'Rachford-Rice (Newton-Raphson)', 
          'Rachford-Rice (Halley)', or 'Rachford-Rice (NumPy)',
          which numerically solves an objective function
          described in :obj:`Rachford_Rice_solution`.
        * 'Leibovici and Nichita 2', a transformation of the RR equation
           described in :obj:`Rachford_Rice_solution_LN2`.
        * 'Li-Johns-Ahmadi', which numerically solves an objective function
          described in :obj:`Li_Johns_Ahmadi_solution`.

    Examples
    --------
    >>> flash_inner_loop(zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    (0.6907302627738537, [0.3394086969663437, 0.36505605903717053, 0.29553524399648573], [0.5719036543882892, 0.2708715958055805, 0.1572247498061304])
    '''
    if AvailableMethods:
        l = len(zs)
        return flash_inner_loop_list_methods(l)
    if Method is None:
        l = len(zs)
        Method = FLASH_INNER_ANALYTICAL if l < 5 else (FLASH_INNER_NUMPY if (not IS_PYPY and l >= 10) else FLASH_INNER_LN2)    
    if check:
        K_low, K_high = False, False
        for K in Ks:
            if K > 1.0:
                K_high = True
            else:
                K_low = True
            if K_high and K_low:
                break
        if not K_low or not K_high:
            raise ValueError("For provided K values, there is no positive-composition solution; Ks=%s" %(Ks))

    if Method == FLASH_INNER_LN2:
        return Rachford_Rice_solution_LN2(zs, Ks, guess)
    elif Method == FLASH_INNER_SECANT:
        return Rachford_Rice_solution(zs, Ks, limit=limit)
    elif Method == FLASH_INNER_ANALYTICAL:
        l = len(zs)
        if l == 2:
            z1, z2 = zs
            K1, K2 = Ks
            try:
                z1z2 = z1 + z2
                K1z1 = K1*z1
                K2z2 = K2*z2
                t1 = z1z2 - K1z1 - K2z2 
                V_over_F = (t1)/(t1 + K2*K1z1 + K1*K2z2 - K1*z2 - K2*z1)
            except ZeroDivisionError:
                return Rachford_Rice_solution(zs=zs, Ks=Ks)
        elif l == 3:
            # TODO disable
            z1, z2, z3 = zs
            K1, K2, K3 = Ks
            V_over_F = (-K1*K2*z1/2 - K1*K2*z2/2 - K1*K3*z1/2 - K1*K3*z3/2 + K1*z1 + K1*z2/2 + K1*z3/2 - K2*K3*z2/2 - K2*K3*z3/2 + K2*z1/2 + K2*z2 + K2*z3/2 + K3*z1/2 + K3*z2/2 + K3*z3 - z1 - z2 - z3 - (K1**2*K2**2*z1**2 + 2*K1**2*K2**2*z1*z2 + K1**2*K2**2*z2**2 - 2*K1**2*K2*K3*z1**2 - 2*K1**2*K2*K3*z1*z2 - 2*K1**2*K2*K3*z1*z3 + 2*K1**2*K2*K3*z2*z3 - 2*K1**2*K2*z1*z2 + 2*K1**2*K2*z1*z3 - 2*K1**2*K2*z2**2 - 2*K1**2*K2*z2*z3 + K1**2*K3**2*z1**2 + 2*K1**2*K3**2*z1*z3 + K1**2*K3**2*z3**2 + 2*K1**2*K3*z1*z2 - 2*K1**2*K3*z1*z3 - 2*K1**2*K3*z2*z3 - 2*K1**2*K3*z3**2 + K1**2*z2**2 + 2*K1**2*z2*z3 + K1**2*z3**2 - 2*K1*K2**2*K3*z1*z2 + 2*K1*K2**2*K3*z1*z3 - 2*K1*K2**2*K3*z2**2 - 2*K1*K2**2*K3*z2*z3 - 2*K1*K2**2*z1**2 - 2*K1*K2**2*z1*z2 - 2*K1*K2**2*z1*z3 + 2*K1*K2**2*z2*z3 + 2*K1*K2*K3**2*z1*z2 - 2*K1*K2*K3**2*z1*z3 - 2*K1*K2*K3**2*z2*z3 - 2*K1*K2*K3**2*z3**2 + 4*K1*K2*K3*z1**2 + 4*K1*K2*K3*z1*z2 + 4*K1*K2*K3*z1*z3 + 4*K1*K2*K3*z2**2 + 4*K1*K2*K3*z2*z3 + 4*K1*K2*K3*z3**2 + 2*K1*K2*z1*z2 - 2*K1*K2*z1*z3 - 2*K1*K2*z2*z3 - 2*K1*K2*z3**2 - 2*K1*K3**2*z1**2 - 2*K1*K3**2*z1*z2 - 2*K1*K3**2*z1*z3 + 2*K1*K3**2*z2*z3 - 2*K1*K3*z1*z2 + 2*K1*K3*z1*z3 - 2*K1*K3*z2**2 - 2*K1*K3*z2*z3 + K2**2*K3**2*z2**2 + 2*K2**2*K3**2*z2*z3 + K2**2*K3**2*z3**2 + 2*K2**2*K3*z1*z2 - 2*K2**2*K3*z1*z3 - 2*K2**2*K3*z2*z3 - 2*K2**2*K3*z3**2 + K2**2*z1**2 + 2*K2**2*z1*z3 + K2**2*z3**2 - 2*K2*K3**2*z1*z2 + 2*K2*K3**2*z1*z3 - 2*K2*K3**2*z2**2 - 2*K2*K3**2*z2*z3 - 2*K2*K3*z1**2 - 2*K2*K3*z1*z2 - 2*K2*K3*z1*z3 + 2*K2*K3*z2*z3 + K3**2*z1**2 + 2*K3**2*z1*z2 + K3**2*z2**2)**0.5/2)/(K1*K2*K3*z1 + K1*K2*K3*z2 + K1*K2*K3*z3 - K1*K2*z1 - K1*K2*z2 - K1*K2*z3 - K1*K3*z1 - K1*K3*z2 - K1*K3*z3 + K1*z1 + K1*z2 + K1*z3 - K2*K3*z1 - K2*K3*z2 - K2*K3*z3 + K2*z1 + K2*z2 + K2*z3 + K3*z1 + K3*z2 + K3*z3 - z1 - z2 - z3)
        elif l == 3 or l == 4 or l == 5:
            return Rachford_Rice_solution_polynomial(zs, Ks)
        elif l == 1:
            raise ValueError("Input dimensions are for one component! Rachford-Rice does not aply")
        else:
            raise Exception('Only solutions for components counts 2, 3, and 4 are available analytically')
        # Need to avoid zero divisions here
        xs = [zi/(1.+V_over_F*(Ki-1.)) for zi, Ki in zip(zs, Ks)] # if zi != 0.0
        ys = [Ki*xi for xi, Ki in zip(xs, Ks)]
        return V_over_F, xs, ys
    
    elif Method == FLASH_INNER_NUMPY:
        try:
            return Rachford_Rice_solution_numpy(zs=zs, Ks=Ks, limit=limit)
        except:
            return Rachford_Rice_solution(zs=zs, Ks=Ks, limit=limit)
    elif Method == FLASH_INNER_NR:
        return Rachford_Rice_solution(zs=zs, Ks=Ks, limit=limit, fprime=True)
    elif Method == FLASH_INNER_HALLEY:
        return Rachford_Rice_solution(zs=zs, Ks=Ks, limit=limit, fprime=True, 
                                      fprime2=True)
    
    elif Method == FLASH_INNER_LJA:
        return Li_Johns_Ahmadi_solution(zs=zs, Ks=Ks)
    elif Method == FLASH_INNER_POLY:
        return Rachford_Rice_solution_polynomial(zs=zs, Ks=Ks)
    else:
        raise Exception('Incorrect Method input')


def NRTL(xs, taus, alphas):
    r'''Calculates the activity coefficients of each species in a mixture
    using the Non-Random Two-Liquid (NRTL) method, given their mole fractions,
    dimensionless interaction parameters, and nonrandomness constants. Those
    are normally correlated with temperature in some form, and need to be
    calculated separately.

    .. math::
        \ln(\gamma_i)=\frac{\displaystyle\sum_{j=1}^{n}{x_{j}\tau_{ji}G_{ji}}}
        {\displaystyle\sum_{k=1}^{n}{x_{k}G_{ki}}}+\sum_{j=1}^{n}
        {\frac{x_{j}G_{ij}}{\displaystyle\sum_{k=1}^{n}{x_{k}G_{kj}}}}
        {\left ({\tau_{ij}-\frac{\displaystyle\sum_{m=1}^{n}{x_{m}\tau_{mj}
        G_{mj}}}{\displaystyle\sum_{k=1}^{n}{x_{k}G_{kj}}}}\right )}

        G_{ij}=\text{exp}\left ({-\alpha_{ij}\tau_{ij}}\right )

    Parameters
    ----------
    xs : list[float]
        Liquid mole fractions of each species, [-]
    taus : list[list[float]]
        Dimensionless interaction parameters of each compound with each other,
        [-]
    alphas : list[list[float]]
        Nonrandomness constants of each compound interacting with each other, [-]

    Returns
    -------
    gammas : list[float]
        Activity coefficient for each species in the liquid mixture, [-]

    Notes
    -----
    This model needs N^2 parameters.

    One common temperature dependence of the nonrandomness constants is:

    .. math::
        \alpha_{ij}=c_{ij}+d_{ij}T

    Most correlations for the interaction parameters include some of the terms
    shown in the following form:

    .. math::
        \tau_{ij}=A_{ij}+\frac{B_{ij}}{T}+\frac{C_{ij}}{T^{2}}+D_{ij}
        \ln{\left ({T}\right )}+E_{ij}T^{F_{ij}}
        
    The original form of this model used the temperature dependence of taus in 
    the form (values can be found in the literature, often with units of
    calories/mol):
        
    .. math::
        \tau_{ij}=\frac{b_{ij}}{RT}

    For this model to produce ideal acitivty coefficients (gammas = 1),
    all interaction parameters should be 0; the value of alpha does not impact
    the calculation when that is the case.

    Examples
    --------
    Ethanol-water example, at 343.15 K and 1 MPa:

    >>> NRTL(xs=[0.252, 0.748], taus=[[0, -0.178], [1.963, 0]],
    ... alphas=[[0, 0.2974],[.2974, 0]])
    [1.9363183763514304, 1.1537609663170014]

    References
    ----------
    .. [1] Renon, Henri, and J. M. Prausnitz. "Local Compositions in
       Thermodynamic Excess Functions for Liquid Mixtures." AIChE Journal 14,
       no. 1 (1968): 135-144. doi:10.1002/aic.690140124.
    .. [2] Gmehling, Jurgen, Barbel Kolbe, Michael Kleiber, and Jurgen Rarey.
       Chemical Thermodynamics for Process Simulation. 1st edition. Weinheim:
       Wiley-VCH, 2012.
    '''
    gammas = []
    cmps = range(len(xs))
    Gs = [[exp(-alphas[i][j]*taus[i][j]) for j in cmps] for i in cmps]
    for i in cmps:
        tn1, td1, total2 = 0., 0., 0.
        Gsi = Gs[i]
        tausi = taus[i]
        for j in cmps:
            # Term 1, numerator and denominator
            tn1 += xs[j]*taus[j][i]*Gs[j][i]
            td1 +=  xs[j]*Gs[j][i]
            # Term 2
            tn2 = xs[j]*Gsi[j]
            
            # TODO: Combine these two to save some multiplications
            td2 = td3 = sum([xs[k]*Gs[k][j] for k in cmps])
            tn3 = sum([xs[m]*taus[m][j]*Gs[m][j] for m in cmps])
            
            total2 += tn2/td2*(tausi[j] - tn3/td3)
        gamma = exp(tn1/td1 + total2)
        gammas.append(gamma)
    return gammas


def Wilson(xs, params):
    r'''Calculates the activity coefficients of each species in a mixture
    using the Wilson method, given their mole fractions, and
    dimensionless interaction parameters. Those are normally correlated with
    temperature, and need to be calculated separately.

    .. math::
        \ln \gamma_i = 1 - \ln \left(\sum_j^N \Lambda_{ij} x_j\right)
        -\sum_j^N \frac{\Lambda_{ji}x_j}{\displaystyle\sum_k^N \Lambda_{jk}x_k}

    Parameters
    ----------
    xs : list[float]
        Liquid mole fractions of each species, [-]
    params : list[list[float]]
        Dimensionless interaction parameters of each compound with each other,
        [-]

    Returns
    -------
    gammas : list[float]
        Activity coefficient for each species in the liquid mixture, [-]

    Notes
    -----
    This model needs N^2 parameters.

    The original model correlated the interaction parameters using the standard
    pure-component molar volumes of each species at 25°C, in the following form:

    .. math::
        \Lambda_{ij} = \frac{V_j}{V_i} \exp\left(\frac{-\lambda_{i,j}}{RT}\right)

    However, that form has less flexibility and offered no advantage over
    using only regressed parameters.

    Most correlations for the interaction parameters include some of the terms
    shown in the following form:

    .. math::
        \ln \Lambda_{ij} =a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T + d_{ij}T
        + \frac{e_{ij}}{T^2} + h_{ij}{T^2}

    The Wilson model is not applicable to liquid-liquid systems.
    
    For this model to produce ideal acitivty coefficients (gammas = 1),
    all interaction parameters should be 1.

    Examples
    --------
    Ethanol-water example, at 343.15 K and 1 MPa:

    >>> Wilson([0.252, 0.748], [[1, 0.154], [0.888, 1]])
    [1.8814926087178843, 1.1655774931125487]

    References
    ----------
    .. [1] Wilson, Grant M. "Vapor-Liquid Equilibrium. XI. A New Expression for
       the Excess Free Energy of Mixing." Journal of the American Chemical
       Society 86, no. 2 (January 1, 1964): 127-130. doi:10.1021/ja01056a002.
    .. [2] Gmehling, Jurgen, Barbel Kolbe, Michael Kleiber, and Jurgen Rarey.
       Chemical Thermodynamics for Process Simulation. 1st edition. Weinheim:
       Wiley-VCH, 2012.
    '''
    gammas = []
    cmps = range(len(xs))
    for i in cmps:
        tot1 = log(sum([params[i][j]*xs[j] for j in cmps]))
        tot2 = 0.
        for j in cmps:
            tot2 += params[j][i]*xs[j]/sum([params[j][k]*xs[k] for k in cmps])

        gamma = exp(1. - tot1 - tot2)
        gammas.append(gamma)
    return gammas


def UNIQUAC(xs, rs, qs, taus):
    r'''Calculates the activity coefficients of each species in a mixture
    using the Universal quasi-chemical (UNIQUAC) equation, given their mole
    fractions, `rs`, `qs`, and dimensionless interaction parameters. The
    interaction parameters are normally correlated with temperature, and need
    to be calculated separately.

    .. math::
        \ln \gamma_i = \ln \frac{\Phi_i}{x_i} + \frac{z}{2} q_i \ln
        \frac{\theta_i}{\Phi_i}+ l_i - \frac{\Phi_i}{x_i}\sum_j^N x_j l_j
        - q_i \ln\left( \sum_j^N \theta_j \tau_{ji}\right)+ q_i - q_i\sum_j^N
        \frac{\theta_j \tau_{ij}}{\sum_k^N \theta_k \tau_{kj}}

        \theta_i = \frac{x_i q_i}{\displaystyle\sum_{j=1}^{n} x_j q_j}

         \Phi_i = \frac{x_i r_i}{\displaystyle\sum_{j=1}^{n} x_j r_j}

         l_i = \frac{z}{2}(r_i - q_i) - (r_i - 1)

    Parameters
    ----------
    xs : list[float]
        Liquid mole fractions of each species, [-]
    rs : list[float]
        Van der Waals volume parameters for each species, [-]
    qs : list[float]
        Surface area parameters for each species, [-]
    taus : list[list[float]]
        Dimensionless interaction parameters of each compound with each other,
        [-]

    Returns
    -------
    gammas : list[float]
        Activity coefficient for each species in the liquid mixture, [-]

    Notes
    -----
    This model needs N^2 parameters.

    The original expression for the interaction parameters is as follows:

    .. math::
        \tau_{ji} = \exp\left(\frac{-\Delta u_{ij}}{RT}\right)

    However, it is seldom used. Most correlations for the interaction
    parameters include some of the terms shown in the following form:

    .. math::
        \ln \tau{ij} =a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T + d_{ij}T
        + \frac{e_{ij}}{T^2}

    This model is recast in a slightly more computationally efficient way in
    [2]_, as shown below:

    .. math::
        \ln \gamma_i = \ln \gamma_i^{res} + \ln \gamma_i^{comb}

        \ln \gamma_i^{res} = q_i \left(1 - \ln\frac{\sum_j^N q_j x_j \tau_{ji}}
        {\sum_j^N q_j x_j}- \sum_j \frac{q_k x_j \tau_{ij}}{\sum_k q_k x_k
        \tau_{kj}}\right)

        \ln \gamma_i^{comb} = (1 - V_i + \ln V_i) - \frac{z}{2}q_i\left(1 -
        \frac{V_i}{F_i} + \ln \frac{V_i}{F_i}\right)

        V_i = \frac{r_i}{\sum_j^N r_j x_j}

        F_i = \frac{q_i}{\sum_j q_j x_j}


    There is no global set of parameters which will make this model yield
    ideal acitivty coefficients (gammas = 1) for this model.
    
    Examples
    --------
    Ethanol-water example, at 343.15 K and 1 MPa:

    >>> UNIQUAC(xs=[0.252, 0.748], rs=[2.1055, 0.9200], qs=[1.972, 1.400],
    ... taus=[[1.0, 1.0919744384510301], [0.37452902779205477, 1.0]])
    [2.35875137797083, 1.2442093415968987]

    References
    ----------
    .. [1] Abrams, Denis S., and John M. Prausnitz. "Statistical Thermodynamics
       of Liquid Mixtures: A New Expression for the Excess Gibbs Energy of
       Partly or Completely Miscible Systems." AIChE Journal 21, no. 1 (January
       1, 1975): 116-28. doi:10.1002/aic.690210115.
    .. [2] Gmehling, Jurgen, Barbel Kolbe, Michael Kleiber, and Jurgen Rarey.
       Chemical Thermodynamics for Process Simulation. 1st edition. Weinheim:
       Wiley-VCH, 2012.
    .. [3] Maurer, G., and J. M. Prausnitz. "On the Derivation and Extension of
       the Uniquac Equation." Fluid Phase Equilibria 2, no. 2 (January 1,
       1978): 91-99. doi:10.1016/0378-3812(78)85002-X.
    '''
    cmps = range(len(xs))
    rsxs = [rs[i]*xs[i] for i in cmps]
    rsxs_sum = sum(rsxs)
    phis = [rsxs[i]/rsxs_sum for i in cmps]
    qsxs = [qs[i]*xs[i] for i in cmps]
    qsxs_sum = sum(qsxs)
    vs = [qsxs[i]/qsxs_sum for i in cmps]

    Ss = [sum([vs[j]*taus[j][i] for j in cmps]) for i in cmps]
    VsSs = [vs[j]/Ss[j] for j in cmps]

    ans = []
    for i in cmps:
        x1 = phis[i]/xs[i]
        x2 = phis[i]/vs[i]
        loggammac = log(x1) + 1.0 - x1 - 5.0*qs[i]*(log(x2) + 1.0 - x2)
        loggammar = qs[i]*(1.0 - log(Ss[i]) - sum([taus[i][j]*VsSs[j] for j in cmps]))
        ans.append(exp(loggammac + loggammar))
    return ans


def flash(P, zs, Psats):
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
    
    
    
    
    
#    if not fugacities:
#        fugacities = [1 for i in range(len(Psats))]
#    if not gammas:
#        gammas = [1 for i in range(len(Psats))]
#    if not none_and_length_check((zs, Psats, fugacities, gammas)):
#        raise Exception('Input dimentions are inconsistent or some input parameters are missing.')
#    P = 1/sum(zs[i]*fugacities[i]/Psats[i]/gammas[i] for i in range(len(zs)))
#    return P
#

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
                           AvailableMethods=False, Method=None):  # pragma: no cover
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
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    xs, ys, phase, V_over_F = None, None, None, None
    if Method == 'IDEAL_VLE':
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
            xs, ys, V_over_F = flash(P, zs, Psats)
            phase = 'two-phase'
    elif Method == 'SUPERCRITICAL_T':
        if all([T >= i for i in Tcs]):
            phase = 'g'
        else: # The following is nonsensical
            phase = 'two-phase'
    elif Method == 'SUPERCRITICAL_P':
        if all([P >= i for i in Pcs]):
            phase = 'g'
        else: # The following is nonsensical
            phase = 'two-phase'
    elif Method == 'IDEAL_VLE_SUPERCRITICAL':
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
            xs, ys, V_over_F = flash(P, zs, Psats)
            phase = 'two-phase'

    elif Method == 'NONE':
        pass
    else:
        raise Exception('Failure in in function')
    return phase, xs, ys, V_over_F


def Pbubble_mixture(T=None, zs=None, Psats=None, CASRNs=None,
                   AvailableMethods=False, Method=None):  # pragma: no cover
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
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == 'IDEAL_VLE':
        Pbubble = bubble_at_T(zs, Psats)
    elif Method == 'NONE':
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

    T_bubble = newton(bubble_P_error, 300)

    return T_bubble


def Pdew_mixture(T=None, zs=None, Psats=None, CASRNs=None,
                 AvailableMethods=False, Method=None):  # pragma: no cover
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
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == 'IDEAL_VLE':
        Pdew = dew_at_T(zs, Psats)
    elif Method == 'NONE':
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
#get_T_bub_est(1E6, zs=[0.5, 0.5], Tbs=[194.67, 341.87], Tcs=[304.2, 507.4], Pcs=[7.38E6, 3.014E6])


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
#get_T_dew_est(1E6, zs=[0.5, 0.5], Tbs=[194.67, 341.87], Tcs=[304.2, 507.4], Pcs=[7.38E6, 3.014E6], T_bub_est = 290.6936541653881)


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
