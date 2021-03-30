# -*- coding: utf-8 -*-
r'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
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
SOFTWARE.

Some of the methods implemented here are numerical while others are analytical.


The cubic EOS can be rearranged into the following polynomial form:

.. math::
    0 = Z^3 + (\delta' - B' - 1)Z^2 + [\theta' + \epsilon' - \delta(B'+1)]Z
    - [\epsilon'(B'+1) + \theta'\eta']

.. math::
    B' = \frac{bP}{RT}

.. math::
    \delta' = \frac{\delta P}{RT}

.. math::
    \theta' = \frac{a\alpha P}{(RT)^2}

.. math::
    \epsilon' = \epsilon\left(\frac{P}{RT}\right)^2

The range of pressures, temperatures, and :math:`a \alpha` values is so large
that almost all analytical solutions produce huge errors in some conditions.
Because the EOS volume cannot be under `b`, this often results in a root being
ignored where there should have been a liquid-like root detected.

A number of plots showing the relative error in volume calculation are shown
below to demonstrate how different methods work.

.. contents:: :local:

Analytical Solvers
------------------
.. autofunction:: volume_solutions_Cardano
.. autofunction:: volume_solutions_fast
.. autofunction:: volume_solutions_a1
.. autofunction:: volume_solutions_a2
.. autofunction:: volume_solutions_numpy
.. autofunction:: volume_solutions_ideal

Numerical Solvers
-----------------
.. autofunction:: volume_solutions_halley
.. autofunction:: volume_solutions_NR
.. autofunction:: volume_solutions_NR_low_P

Higher-Precision Solvers
------------------------
.. autofunction:: volume_solutions_mpmath
.. autofunction:: volume_solutions_mpmath_float

'''

from __future__ import division, print_function
__all__ = ['volume_solutions_mpmath', 'volume_solutions_mpmath_float',
           'volume_solutions_NR', 'volume_solutions_NR_low_P', 'volume_solutions_halley',
           'volume_solutions_fast', 'volume_solutions_Cardano', 'volume_solutions_a1',
           'volume_solutions_a2', 'volume_solutions_numpy', 'volume_solutions_ideal']


from cmath import sqrt as csqrt

from fluids.numerics import (brenth, third, sixth, roots_cubic,
                             roots_cubic_a1, numpy as np,
                             roots_cubic_a2,
                             deflate_cubic_real_roots)

from fluids.constants import R, R_inv



def volume_solutions_mpmath(T, P, b, delta, epsilon, a_alpha, dps=30):
    r'''Solution of this form of the cubic EOS in terms of volumes, using the
    `mpmath` arbitrary precision library. The number of decimal places returned
    is controlled by the `dps` parameter.

    This function is the reference implementation which provides exactly
    correct solutions; other algorithms are compared against this one.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    b : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
    dps : int
        Number of decimal places in the result by `mpmath`, [-]

    Returns
    -------
    Vs : tuple[complex]
        Three possible molar volumes, [m^3/mol]

    Notes
    -----
    Although `mpmath` has a cubic solver, it has been found to fail to solve in
    some cases. Accordingly, the algorithm is as follows:

    Working precision is `dps` plus 40 digits; and if P < 1e-10 Pa, it is
    `dps` plus 400 digits. The input parameters are converted exactly to `mpf`
    objects on input.

    `polyroots` from mpmath is used with `maxsteps=2000`, and extra precision
    of 15 digits. If the solution does not converge, 20 extra digits are added
    up to 8 times. If no solution is found, mpmath's `findroot` is called on
    the pressure error function using three initial guesses from another solver.

    Needless to say, this function is quite slow.

    Examples
    --------
    Test case which presented issues for PR EOS (three roots were not being returned):

    >>> volume_solutions_mpmath(0.01, 1e-05, 2.5405184201558786e-05, 5.081036840311757e-05, -6.454233843151321e-10, 0.3872747173781095)
    (mpf('0.0000254054613415548712260258773060137'), mpf('4.66038025602155259976574392093252'), mpf('8309.80218708657190094424659859346'))

    References
    ----------
    .. [1] Johansson, Fredrik. Mpmath: A Python Library for Arbitrary-Precision
       Floating-Point Arithmetic, 2010.
    '''
    # Tried to remove some green on physical TV with more than 30, could not
    # 30 is fine, but do not dercease further!
    # No matter the precision, still cannot get better
    # Need to switch from `rindroot` to an actual cubic solution in mpmath
    # Three roots not found in some cases
    # PRMIX(T=1e-2, P=1e-5, Tcs=[126.1, 190.6], Pcs=[33.94E5, 46.04E5], omegas=[0.04, 0.011], zs=[0.5, 0.5], kijs=[[0,0],[0,0]]).volume_error()
    # Once found it possible to compute VLE down to 0.03 Tc with ~400 steps and ~500 dps.
    # need to start with a really high dps to get convergence or it is discontinuous
    if P == 0.0 or T == 0.0:
        raise ValueError("Bad P or T; issue is not the algorithm")

    import mpmath as mp
    mp.mp.dps = dps + 40#400#400
    if P < 1e-10:
        mp.mp.dps = dps + 400
    b, T, P, epsilon, delta, a_alpha = [mp.mpf(i) for i in [b, T, P, epsilon, delta, a_alpha]]
    roots = None
    if 1:
        RT_inv = 1/(mp.mpf(R)*T)
        P_RT_inv = P*RT_inv
        B = etas = b*P_RT_inv
        deltas = delta*P_RT_inv
        thetas = a_alpha*P_RT_inv*RT_inv
        epsilons = epsilon*P_RT_inv*P_RT_inv

        b = (deltas - B - 1)
        c = (thetas + epsilons - deltas*(B + 1))
        d = -(epsilons*(B + 1) + thetas*etas)

        extraprec = 15
        # extraprec alone is not enough to converge everything
        try:
            # found case 20 extrapec not enough, increased to 30
            # Found another case needing 40
            for i in range(8):
                try:
                    # Found 1 case 100 steps not enough needed 200; then found place 400 was not enough
                    roots = mp.polyroots([mp.mpf(1.0), b, c, d], extraprec=extraprec, maxsteps=2000)
                    break
                except Exception as e:
                    extraprec += 20
#                        print(e, extraprec)
                    if i == 7:
#                            print(e, 'failed')
                        raise e

            if all(i == 0 or i == 1 for i in roots):
                return volume_solutions_mpmath(T, P, b, delta, epsilon, a_alpha, dps=dps*2)
        except:
            try:
                guesses = volume_solutions_fast(T, P, b, delta, epsilon, a_alpha)
                roots = mp.polyroots([mp.mpf(1.0), b, c, d], extraprec=40, maxsteps=100, roots_init=guesses)
            except:
                pass
#            roots = np.roots([1.0, b, c, d]).tolist()
        if roots is not None:
            RT_P = mp.mpf(R)*T/P
            hits = [V*RT_P for V in roots]

    if roots is None:
#        print('trying numerical mpmath')
        guesses = volume_solutions_fast(T, P, b, delta, epsilon, a_alpha)
        RT = T*R
        def err(V):
            return(RT/(V-b) - a_alpha/(V*(V + delta) + epsilon)) - P

        hits = []
        for Vi in guesses:
            try:
                V_calc = mp.findroot(err, Vi, solver='newton')
                hits.append(V_calc)
            except Exception as e:
                pass
        if not hits:
            raise ValueError("Could not converge any mpmath volumes")
    # Return in the specified precision
    mp.mp.dps = dps
    sort_fun = lambda x: (x.real, x.imag)
    return tuple(sorted(hits, key=sort_fun))

def volume_solutions_mpmath_float(T, P, b, delta, epsilon, a_alpha):
    r'''Simple wrapper around :obj:`volume_solutions_mpmath` which uses the
    default parameters and returns the values as floats.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    b : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
    dps : int
        Number of decimal places in the result by `mpmath`, [-]

    Returns
    -------
    Vs : tuple[complex]
        Three possible molar volumes, [m^3/mol]

    Notes
    -----

    Examples
    --------
    Test case which presented issues for PR EOS (three roots were not being returned):

    >>> volume_solutions_mpmath_float(0.01, 1e-05, 2.5405184201558786e-05, 5.081036840311757e-05, -6.454233843151321e-10, 0.3872747173781095)
    ((2.540546134155487e-05+0j), (4.660380256021552+0j), (8309.802187086572+0j))
    '''
    Vs = volume_solutions_mpmath(T, P, b, delta, epsilon, a_alpha)
    return tuple(float(Vi.real) + float(Vi.imag)*1.0j for Vi in Vs)

def volume_solutions_NR(T, P, b, delta, epsilon, a_alpha, tries=0):
    r'''Newton-Raphson based solver for cubic EOS volumes based on the idea
    of initializing from an analytical solver. This algorithm can only be
    described as a monstrous mess. It is fairly fast for most cases, but about
    3x slower than :obj:`volume_solutions_halley`. In the worst case this
    will fall back to `mpmath`.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    b : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
    tries : int, optional
        Internal parameter as this function will call itself if it needs to;
        number of previous solve attempts, [-]

    Returns
    -------
    Vs : tuple[complex]
        Three possible molar volumes, [m^3/mol]

    Notes
    -----

    Sample regions where this method works perfectly are shown below:

    .. figure:: eos/volume_error_NR_PR_methanol_high.png
       :scale: 70 %
       :alt: PR EOS methanol volume error high pressure

    .. figure:: eos/volume_error_NR_PR_methanol_low.png
       :scale: 70 %
       :alt: PR EOS methanol volume error low pressure

    '''


    '''Even if mpmath is used for greater precision in the calculated root,
    it gets rounded back to a float - and then error occurs.
    Cannot beat numerical method or numpy roots!

    The only way out is to keep volume as many decimals, to pass back in
    to initialize the TV state.
    '''
    # Initial calculation - could use any method, however this is fastest
    # 2 divisions, 2 powers in here
    # First bit is top left corner
    if a_alpha == 0.0:
        '''from sympy import *
            R, T, P, b, V = symbols('R, T, P, b, V')
            solve(Eq(P, R*T/(V-b)), V)
        '''
        # EOS has devolved into having the first term solution only
        return [b + R*T/P, -1j, -1j]
    if P < 1e-2:
    # if 0 or (0 and ((T < 1e-2 and P > 1e6) or (P < 1e-3 and T < 1e-2) or (P < 1e-1 and T < 1e-4) or P < 1)):
        # Not perfect but so much wasted dev time need to move on, try other fluids and move this tolerance up if needed
        # if P < min(GCEOS.P_discriminant_zeros_analytical(T=T, b=b, delta=delta, epsilon=epsilon, a_alpha=a_alpha, valid=True)):
            # TODO - need function that returns range two solutions are available!
            # Very important because the below strategy only works for that regime.
        if T > 1e-2 or 1:
            try:
                return volume_solutions_NR_low_P(T, P, b, delta, epsilon, a_alpha)
            except Exception as e:
                pass
#                print(e, 'was not 2 phase')

        try:
            return volume_solutions_mpmath_float(T, P, b, delta, epsilon, a_alpha)
        except:
            pass
    try:
        if tries == 0:
            Vs = list(volume_solutions_Cardano(T, P, b, delta, epsilon, a_alpha))
#                Vs = [Vi+1e-45j for Vi in volume_solutions_Cardano(T, P, b, delta, epsilon, a_alpha, quick=True)]
        elif tries == 1:
            Vs = list(volume_solutions_fast(T, P, b, delta, epsilon, a_alpha))
        elif tries == 2:
            # sometimes used successfully
            Vs = list(volume_solutions_a1(T, P, b, delta, epsilon, a_alpha))
        # elif tries == 3:
        #     # never used successfully
        #     Vs = GCEOS.volume_solutions_a2(T, P, b, delta, epsilon, a_alpha)

        # TODO fall back to tlow T
    except:
#            Vs = GCEOS.volume_solutions_Cardano(T, P, b, delta, epsilon, a_alpha)
        if tries == 0:
            Vs = list(volume_solutions_fast(T, P, b, delta, epsilon, a_alpha))
        else:
            Vs = list(volume_solutions_Cardano(T, P, b, delta, epsilon, a_alpha))
        # Zero division error is possible above

    RT = R*T
    P_inv = 1.0/P
#        maxiter = range(3)
    # The case for a fixed number of iterations has pretty much gone.
    # On 1 occasion
    failed = False
    max_err, rel_err = 0.0, 0.0
    try:
        for i in (0, 1, 2):
            V = Vi = Vs[i]
            err = 0.0
            for _ in range(11):
                # More iterations seems to create problems. No, 11 is just lucky for particular problem.
#            for _ in (0, 1, 2):
                # 3 divisions each iter = 15, triple the duration of the solve
                denom1 = 1.0/(V*(V + delta) + epsilon)
                denom0 = 1.0/(V-b)
                w0 = RT*denom0
                w1 = a_alpha*denom1
                if w0 - w1 - P == err:
                    break # No change in error
                err = w0 - w1 - P
#                print(abs(err), V, _)
                derr_dV = (V + V + delta)*w1*denom1 - w0*denom0
                V = V - err/derr_dV
                rel_err = abs(err*P_inv)
                if rel_err < 1e-14 or V == Vi:
                    # Conditional check probably not worth it
                    break
#                if _ > 5:
#                    print(_, V)
            # This check can get rid of the noise
            if rel_err > 1e-2: # originally 1e-2; 1e-5 did not change; 1e-10 to far
#            if abs(err*P_inv) > 1e-2 and (i.real != 0.0 and abs(i.imag/i.real) < 1E-10 ):
                failed = True
#                    break
            if not (.95 < (Vi/V).real < 1.05):
                # Cannot let a root become another root
                failed = True
                max_err = 1e100
                break
            Vs[i] = V
            max_err = max(max_err, rel_err)
    except:
        failed = True

#            def to_sln(V):
#                denom1 = 1.0/(V*(V + delta) + epsilon)
#                denom0 = 1.0/(V-b)
#                w0 = x2*denom0
#                w1 = a_alpha*denom1
#                err = w0 - w1 - P
##                print(err*P_inv, V)
#                return err#*P_inv
#            try:
#                from fluids.numerics import py_bisect as bisect, secant, linspace
##                Vs[i] = secant(to_sln, Vs[i].real, x1=Vs[i].real*1.0001, ytol=1e-12, damping=.6)
#                import matplotlib.pyplot as plt
#
#                plt.figure()
#                xs = linspace(Vs[i].real*.9999999999, Vs[i].real*1.0000000001, 2000000) + [Vs[i]]
#                ys = [abs(to_sln(V)) for V in xs]
#                plt.semilogy(xs, ys)
#                plt.show()
#
##                Vs[i] = bisect(to_sln, Vs[i].real*.999, Vs[i].real*1.001)
#            except Exception as e:
#                print(e)
    root_failed = not [i.real for i in Vs if i.real > b and (i.real == 0.0 or abs(i.imag/i.real) < 1E-12)]
    if not failed:
        failed = root_failed

    if failed and tries < 2:
        return volume_solutions_NR(T, P, b, delta, epsilon, a_alpha, tries=tries+1)
    elif root_failed:
#            print('%g, %g; ' %(T, P), end='')
        return volume_solutions_mpmath_float(T, P, b, delta, epsilon, a_alpha)
    elif failed and tries == 2:
        # Are we at least consistent? Diitch the NR and try to be OK with the answer
#            Vs0 = GCEOS.volume_solutions_Cardano(T, P, b, delta, epsilon, a_alpha, quick=True)
#            Vs1 = GCEOS.volume_solutions_a1(T, P, b, delta, epsilon, a_alpha, quick=True)
#            if sum(abs((i -j)/i) for i, j in zip(Vs0, Vs1)) < 1e-6:
#                return Vs0
        if max_err < 5e3:
        # if max_err < 1e6:
            # Try to catch floating point error
            return Vs
        return volume_solutions_NR_low_P(T, P, b, delta, epsilon, a_alpha)
#        print('%g, %g; ' %(T, P), end='')
#            print(T, P, b, delta, a_alpha)
#            if root_failed:
        return volume_solutions_mpmath_float(T, P, b, delta, epsilon, a_alpha)
        # return Vs
#        if tries == 3 or tries == 2:
#            print(tries)
    return Vs

def volume_solutions_NR_low_P(T, P, b, delta, epsilon, a_alpha):
    r'''Newton-Raphson based solver for cubic EOS volumes designed specifically
    for the low-pressure regime. Seeks only two possible solutions - an ideal
    gas like one, and one near the eos covolume `b` - as the initializations are
    `R*T/P` and `b*1.000001` .

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    b : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]
    tries : int, optional
        Internal parameter as this function will call itself if it needs to;
        number of previous solve attempts, [-]

    Returns
    -------
    Vs : tuple[complex]
        Three possible molar volumes (third one is hardcoded to 1j), [m^3/mol]

    Notes
    -----
    The algorithm is NR, with some checks that will switch the solver to
    `brenth` some of the time.
    '''

    P_inv = 1.0/P
    def err_fun(V):
        denom1 = 1.0/(V*(V + delta) + epsilon)
        denom0 = 1.0/(V-b)
        w0 = R*T*denom0
        w1 = a_alpha*denom1
        err = w0 - w1 - P
        return err

#        failed = False
    Vs = [R*T/P, b*1.000001]
    max_err, rel_err = 0.0, 0.0
    for i, damping in zip((0, 1), (1.0, 1.0)):
        V = Vi = Vs[i]
        err = 0.0
        for _ in range(31):
            denom1 = 1.0/(V*(V + delta) + epsilon)
            denom0 = 1.0/(V-b)
            w0 = R*T*denom0
            w1 = a_alpha*denom1
            if w0 - w1 - P == err:
                break # No change in error
            err = w0 - w1 - P
            derr_dV = (V + V + delta)*w1*denom1 - w0*denom0
            if derr_dV != 0.0:
                V = V - err/derr_dV*damping
            rel_err = abs(err*P_inv)
            if rel_err < 1e-14 or V == Vi:
                # Conditional check probably not worth it
                break
        if i == 1 and V > 1.5*b or V < b:
            # try:
                # try:
            try:
                try:
                    V = brenth(err_fun, b*(1.0+1e-12), b*(1.5), xtol=1e-14)
                except Exception as e:
                    if a_alpha < 1e-5:
                        V = brenth(err_fun, b*1.5, b*5.0, xtol=1e-14)
                    else:
                        raise e

                denom1 = 1.0/(V*(V + delta) + epsilon)
                denom0 = 1.0/(V-b)
                w0 = R*T*denom0
                w1 = a_alpha*denom1
                err = w0 - w1 - P
                derr_dV = (V + V + delta)*w1*denom1 - w0*denom0
                V_1NR = V - err/derr_dV*damping
                if abs((V_1NR-V)/V) < 1e-10:
                    V = V_1NR

            except:
                V = 1j
        if i == 0 and rel_err > 1e-8:
            V = 1j
#                    failed = True
                # except:
                #     V = brenth(err_fun, b*(1.0+1e-12), b*(1.5))
            # except:
            #     pass
                # print([T, P, 'fail on brenth low P root'])
        Vs[i] = V
#            max_err = max(max_err, rel_err)
    Vs.append(1j)
#        if failed:
    return Vs

def volume_solutions_halley(T, P, b, delta, epsilon, a_alpha):
    r'''Halley's method based solver for cubic EOS volumes based on the idea
    of initializing from a single liquid-like guess which is solved precisely,
    deflating the cubic analytically, solving the quadratic equation for the
    next two volumes, and then performing two halley steps on each of them
    to obtain the final solutions. This method does not calculate imaginary
    roots - they are set to zero on detection. This method has been rigorously
    tested over a wide range of conditions.

    One limitation is that if `P < 1e-2` or `a_alpha < 1e-9` the NR solution
    is called as this method has not been found to be completely suitable
    for those conditions.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    b : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

    Returns
    -------
    Vs : tuple[complex]
        Three possible molar volumes, [m^3/mol]

    Notes
    -----
    A sample region where this method works perfectly is shown below:

    .. figure:: eos/volume_error_halley_PR_methanol_low.png
       :scale: 70 %
       :alt: PR EOS methanol volume error low pressure

    '''
    '''
    Cases known to be failing:
        (nitrogen); goes to the other solver and it reports a wrong duplicate root
        obj = PRTranslatedConsistent(Tc=126.2, Pc=3394387.5, omega=0.04, T=3204.081632653062, P=1e9)
    '''
    # Test the case where a_alpha is so low, even with the lowest possible volume `b`,
    # the value of the second term plus P is equal to P.
#        if a_alpha == 0.0:
#            return (b + R*T/P, 0.0, 0.0)
    if a_alpha/(b*(b + delta) + epsilon) + P == P:
        return (b + R*T/P, 0.0, 0.0)
    if P < 1e-2 or a_alpha < 1e-9:  # numba: delete
    # if 0 or (0 and ((T < 1e-2 and P > 1e6) or (P < 1e-3 and T < 1e-2) or (P < 1e-1 and T < 1e-4) or P < 1)):
        # Not perfect but so much wasted dev time need to move on, try other fluids and move this tolerance up if needed
        # if P < min(GCEOS.P_discriminant_zeros_analytical(T=T, b=b, delta=delta, epsilon=epsilon, a_alpha=a_alpha, valid=True)):
            # TODO - need function that returns range two solutions are available!
            # Very important because the below strategy only works for that regime.
        try: # numba: delete
            return volume_solutions_NR(T, P, b, delta, epsilon, a_alpha)  # numba: delete
        except:  # numba: delete
            return (0.0, 0.0, 0.0)  # numba: delete

#        try:
#            return volume_solutions_mpmath_float(T, P, b, delta, epsilon, a_alpha)
#        except:
#            pass

    RT = R*T
    RT_2 = RT + RT
    a_alpha_2 = a_alpha + a_alpha
    P_inv = 1.0/P

    RT_inv = R_inv/T
    P_RT_inv = P*RT_inv
    B = etas = b*P_RT_inv
    deltas = delta*P_RT_inv
    thetas = a_alpha*P_RT_inv*RT_inv
    epsilons = epsilon*P_RT_inv*P_RT_inv

    b2 = (deltas - B - 1.0)
    c2 = (thetas + epsilons - deltas*(B + 1.0))
    d2 = -(epsilons*(B + 1.0) + thetas*etas)
    RT_P = RT*P_inv

    V0, V1 = 0.0, 0.0
    for i in range(3):
        if i == 0:
            V = Vi = -RT_P*d2/c2#R*T*P_inv
            if V <= b:
                V = b*1.000001 # avoid a division by zero
        elif i == 1:
            V = Vi = b*1.000001
        elif i == 2:
            V = Vi = b*20.0
        fval_oldold = 1.0
        fval_old = 0.0
        for j in range(50):
#             print(j, V)
            x0_inv = 1.0/(V - b)
            x1_inv = 1.0/(V*(V + delta) + epsilon)
            x2 = V + V + delta
            fval = RT*x0_inv - P - a_alpha*x1_inv
            x0_inv2 = x0_inv*x0_inv # make it 1/x0^2
            x1_inv2 = x1_inv*x1_inv # make it 1/x1^2
            x3 = a_alpha*x1_inv2
            fder = x2*x3 - RT*x0_inv2
            fder2 = RT_2*x0_inv2*x0_inv - a_alpha_2*x2*x2*x1_inv2*x1_inv + x3 + x3

            fder_inv = 1.0/fder
            step = fval*fder_inv
            rel_err = abs(fval*P_inv)
#                print(fval, rel_err, step, j, i, V)
            step_den = 1.0 - 0.5*step*fder2*fder_inv
            if step_den == 0.0:
#                    if fval == 0.0:
#                        break # got a perfect answer
                continue
            V = V - step/step_den

            if (rel_err < 3e-15 or V == Vi or fval_old == fval or fval == fval_oldold
                or (j > 10 and rel_err < 1e-12)):
                # Conditional check probably not worth it
                break
            fval_oldold, fval_old = fval_old, fval

#         if i == 0:
#             V0 = V
#         elif i == 1:
#             V1 = V
        if j != 49:
            V0 = V

            x1, x2 = deflate_cubic_real_roots(b2, c2, d2, V*P_RT_inv)
            if x1 == 0.0:
                return (V0, 0.0, 0.0)
            # 8 divisions only for polishing
            V1 = x1*RT_P
            V2 = x2*RT_P
#             print(V1, V2, 'deflated Vs')

            # Fixed a lot of really bad points in the plots with these.
            # Article suggests they are not needed, but 1 is better than 11 iterations!
            V = V1
            x0_inv = 1.0/(V - b)
            t90 = V*(V + delta) + epsilon
            if t90 != 0.0:
                x1_inv = 1.0/(V*(V + delta) + epsilon)
                x2 = V + V + delta
                fval = -P + RT*x0_inv - a_alpha*x1_inv
                x0_inv2 = x0_inv*x0_inv # make it 1/x0^2
                x1_inv2 = x1_inv*x1_inv # make it 1/x1^2
                x3 = a_alpha*x1_inv2
                fder = x2*x3 - RT*x0_inv2
                fder2 = RT_2*x0_inv2*x0_inv - a_alpha_2*x2*x2*x1_inv2*x1_inv + x3 + x3

                fder_inv = 1.0/fder
                step = fval*fder_inv
                V1 = V - step/(1.0 - 0.5*step*fder2*fder_inv)

            # Take a step with V2
            V = V2
            x0_inv = 1.0/(V - b)
            t90 = V*(V + delta) + epsilon
            if t90 != 0.0:
                x1_inv = 1.0/(t90)
                x2 = V + V + delta
                fval = -P + RT*x0_inv - a_alpha*x1_inv
                x0_inv2 = x0_inv*x0_inv # make it 1/x0^2
                x1_inv2 = x1_inv*x1_inv # make it 1/x1^2
                x3 = a_alpha*x1_inv2
                fder = x2*x3 - RT*x0_inv2
                fder2 = RT_2*x0_inv2*x0_inv - a_alpha_2*x2*x2*x1_inv2*x1_inv + x3 + x3

                fder_inv = 1.0/fder
                step = fval*fder_inv
                V2 = V - step/(1.0 - 0.5*step*fder2*fder_inv)
            return (V0, V1, V2)
    return (0.0, 0.0, 0.0)

def volume_solutions_fast(T, P, b, delta, epsilon, a_alpha):
    r'''Solution of this form of the cubic EOS in terms of volumes. Returns
    three values, all with some complex part. This is believed to be the
    fastest analytical formula, and while it does not suffer from the same
    errors as Cardano's formula, it has plenty of its own numerical issues.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    b : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

    Returns
    -------
    Vs : tuple[complex]
        Three possible molar volumes, [m^3/mol]

    Notes
    -----
    Using explicit formulas, as can be derived in the following example,
    is faster than most numeric root finding techniques, and
    finds all values explicitly. It takes several seconds.

    >>> from sympy import *
    >>> P, T, V, R, b, a, delta, epsilon, alpha = symbols('P, T, V, R, b, a, delta, epsilon, alpha')
    >>> Tc, Pc, omega = symbols('Tc, Pc, omega')
    >>> CUBIC = R*T/(V-b) - a*alpha/(V*V + delta*V + epsilon) - P
    >>> #solve(CUBIC, V)

    A sample region where this method does not obtain the correct solution
    (PR EOS for methanol) is as follows:

    .. figure:: eos/volume_error_sympy_PR_methanol_high.png
       :scale: 70 %
       :alt: PR EOS methanol volume error high pressure

    References
    ----------
    .. [1] Zhi, Yun, and Huen Lee. "Fallibility of Analytic Roots of Cubic
       Equations of State in Low Temperature Region." Fluid Phase
       Equilibria 201, no. 2 (September 30, 2002): 287-94.
       https://doi.org/10.1016/S0378-3812(02)00072-9.
    '''
    x24 = 1.73205080756887729352744634151j + 1.
    x24_inv = 0.25 - 0.433012701892219323381861585376j
    x26 = -1.73205080756887729352744634151j + 1.
    x26_inv = 0.25 + 0.433012701892219323381861585376j
    # Changing over to the inverse constants changes some dew point results
#        if quick:
    x0 = 1./P
    x1 = P*b
    x2 = R*T
    x3 = P*delta
    x4 = x1 + x2 - x3
    x5 = x0*x4
    x6 = a_alpha*b
    x7 = epsilon*x1
    x8 = epsilon*x2
    x9 = x0*x0
    x10 = P*epsilon
    x11 = delta*x1
    x12 = delta*x2
#            x13 = 3.*a_alpha
#            x14 = 3.*x10
#            x15 = 3.*x11
#            x16 = 3.*x12
    x17 = -x4
    x17_2 = x17*x17
    x18 = x0*x17_2
    tm1 = x12 - a_alpha + (x11  - x10)
#            print(x11, x12, a_alpha, x10)
    t0 = x6 + x7 + x8
    t1 = (3.0*tm1  + x18) # custom vars
#            t1 = (-x13 - x14 + x15 + x16 + x18) # custom vars
    t2 = (9.*x0*x17*tm1 + 2.0*x17_2*x17*x9
             - 27.*t0)

    x4x9  = x4*x9
    x19 = ((-13.5*x0*t0 - 4.5*x4x9*tm1
           - x4*x4x9*x5
            + 0.5*csqrt((x9*(-4.*x0*t1*t1*t1 + t2*t2))+0.0j)
            )+0.0j)**third

    x20 = -t1/x19#
    x22 = x5 + x5
    x25 = 4.*x0*x20
    return ((x0*x20 - x19 + x5)*third,
            (x19*x24 + x22 - x25*x24_inv)*sixth,
            (x19*x26 + x22 - x25*x26_inv)*sixth)


def volume_solutions_Cardano(T, P, b, delta, epsilon, a_alpha):
    r'''Calculate the molar volume solutions to a cubic equation of state using
    Cardano's formula, and a few tweaks to improve numerical precision.
    This solution is quite fast in general although it involves powers or
    trigonometric functions. However, it has numerical issues at many
    seemingly random areas in the low pressure region.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    b : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

    Returns
    -------
    Vs : list[float]
        Three possible molar volumes, [m^3/mol]

    Notes
    -----

    Two sample regions where this method does not obtain the correct solution
    (PR EOS for hydrogen) are as follows:

    .. figure:: eos/volume_error_cardano_PR_hydrogen_high.png
       :scale: 100 %
       :alt: PR EOS hydrogen volume error high pressure

    .. figure:: eos/volume_error_cardano_PR_hydrogen_low.png
       :scale: 100 %
       :alt: PR EOS hydrogen volume error low pressure

    References
    ----------
    .. [1] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    RT_inv = R_inv/T
    P_RT_inv = P*RT_inv
    B = etas = b*P_RT_inv
    deltas = delta*P_RT_inv
    thetas = a_alpha*P_RT_inv*RT_inv
    epsilons = epsilon*P_RT_inv*P_RT_inv

    b = (deltas - B - 1.0)
    c = (thetas + epsilons - deltas*(B + 1.0))
    d = -(epsilons*(B + 1.0) + thetas*etas)
    roots = list(roots_cubic(1.0, b, c, d))
    RT_P = R*T/P
    return [V*RT_P for V in roots]

def volume_solutions_a1(T, P, b, delta, epsilon, a_alpha):
    r'''Solution of this form of the cubic EOS in terms of volumes. Returns
    three values, all with some complex part. This uses an analytical solution
    for the cubic equation with the leading coefficient set to 1 as in the EOS
    case; and the analytical solution is the one recommended by Mathematica.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    b : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

    Returns
    -------
    Vs : tuple[complex]
        Three possible molar volumes, [m^3/mol]

    Notes
    -----
    A sample region where this method does not obtain the correct solution
    (PR EOS for methanol) is as follows:

    .. figure:: eos/volume_error_mathematica_PR_methanol_high.png
       :scale: 70 %
       :alt: PR EOS methanol volume error high pressure

    Examples
    --------
    >>> volume_solutions_a1(8837.07874361444, 216556124.0631852, 0.0003990176625589891, 0.0010590390565805598, -1.5069972655436541e-07, 7.20417995032918e-15)
    ((0.000738308-7.5337e-20j), (-0.001186094-6.52444e-20j), (0.000127055+6.52444e-20j))
    '''
    RT_inv = R_inv/T
    P_RT_inv = P*RT_inv
    B = etas = b*P_RT_inv
    deltas = delta*P_RT_inv
    thetas = a_alpha*P_RT_inv*RT_inv
    epsilons = epsilon*P_RT_inv*P_RT_inv

    b = (deltas - B - 1.0)
    c = (thetas + epsilons - deltas*(B + 1.0))
    d = -(epsilons*(B + 1.0) + thetas*etas)
#        roots_cubic_a1, roots_cubic_a2
    RT_P = R*T/P
    return tuple(V*RT_P for V in roots_cubic_a1(b, c, d))

def volume_solutions_a2(T, P, b, delta, epsilon, a_alpha):
    r'''Solution of this form of the cubic EOS in terms of volumes. Returns
    three values, all with some complex part. This uses an analytical solution
    for the cubic equation with the leading coefficient set to 1 as in the EOS
    case; and the analytical solution is the one recommended by Maple.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    b : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

    Returns
    -------
    Vs : tuple[complex]
        Three possible molar volumes, [m^3/mol]

    Notes
    -----
    A sample region where this method does not obtain the correct solution
    (SRK EOS for decane) is as follows:

    .. figure:: eos/volume_error_maple_SRK_decane_high.png
       :scale: 70 %
       :alt: SRK EOS decane volume error high pressure
    '''
    #
    RT_inv = R_inv/T
    P_RT_inv = P*RT_inv
    B = etas = b*P_RT_inv
    deltas = delta*P_RT_inv
    thetas = a_alpha*P_RT_inv*RT_inv
    epsilons = epsilon*P_RT_inv*P_RT_inv

    b = (deltas - B - 1.0)
    c = (thetas + epsilons - deltas*(B + 1.0))
    d = -(epsilons*(B + 1.0) + thetas*etas)
#        roots_cubic_a1, roots_cubic_a2
    roots = list(roots_cubic_a2(1.0, b, c, d))

    RT_P = R*T/P
    return [V*RT_P for V in roots]



def volume_solutions_numpy(T, P, b, delta, epsilon, a_alpha):
    r'''Calculate the molar volume solutions to a cubic equation of state using
    NumPy's `roots` function, which is a power series iterative matrix solution
    that is very stable but does not have full precision in some cases.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    b : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

    Returns
    -------
    Vs : list[float]
        Three possible molar volumes, [m^3/mol]

    Notes
    -----

    A sample region where this method does not obtain the correct solution
    (SRK EOS for ethane) is as follows:

    .. figure:: eos/volume_error_numpy_SRK_ethane.png
       :scale: 100 %
       :alt: numpy.roots error for SRK eos using ethane_


    References
    ----------
    .. [1] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    '''
    RT_inv = R_inv/T
    P_RT_inv = P*RT_inv
    B = etas = b*P_RT_inv
    deltas = delta*P_RT_inv
    thetas = a_alpha*P_RT_inv*RT_inv
    epsilons = epsilon*P_RT_inv*P_RT_inv

    b = (deltas - B - 1.0)
    c = (thetas + epsilons - deltas*(B + 1.0))
    d = -(epsilons*(B + 1.0) + thetas*etas)

    roots = np.roots([1.0, b, c, d]).tolist()
    RT_P = R*T/P
    return [V*RT_P for V in roots]

def volume_solutions_ideal(T, P, b=0.0, delta=0.0, epsilon=0.0, a_alpha=0.0):
    r'''Calculate the ideal-gas molar volume in a format compatible with the
    other cubic EOS solvers. The ideal gas volume is the first element; and the
    secodn and third elements are zero. This is implemented to allow the
    ideal-gas model to be compatible with the cubic models, whose equations
    do not work with parameters of zero.

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [Pa]
    b : float, optional
        Coefficient calculated by EOS-specific method, [m^3/mol]
    delta : float, optional
        Coefficient calculated by EOS-specific method, [m^3/mol]
    epsilon : float, optional
        Coefficient calculated by EOS-specific method, [m^6/mol^2]
    a_alpha : float, optional
        Coefficient calculated by EOS-specific method, [J^2/mol^2/Pa]

    Returns
    -------
    Vs : list[float]
        Three possible molar volumes, [m^3/mol]

    Examples
    --------
    >>> volume_solutions_ideal(T=300, P=1e7)
    (0.0002494338785445972, 0.0, 0.0)
    '''
    return (R*T/P, 0.0, 0.0)