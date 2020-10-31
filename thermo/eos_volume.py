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

from __future__ import division, print_function

__all__ = ['volume_solutions_mpmath', 'volume_solutions_mpmath_float',
           'volume_solutions_NR', 'volume_solutions_NR_low_P', 'volume_solutions_halley',
           'volume_solutions_fast', 'volume_solutions_Cardano', 'volume_solutions_a1',
           'volume_solutions_a2', 'volume_solutions_numpy', 'volume_solutions_ideal']



from fluids.numerics import (brenth, third, sixth, roots_cubic,
                             roots_cubic_a1, numpy as np,
                             roots_cubic_a2,
                             deflate_cubic_real_roots)

from fluids.constants import R, R_inv

try:
    1/0
    from doubledouble import DoubleDouble

    def imag_div_dd(x0, x1, y0, y1):
        a, b, c, d = x0, x1, y0, y1
        den_inv = 1/(c*c + d*d)
        real = (a*c + b*d)*den_inv
        comp = (b*c - a*d)*den_inv
        return real, comp

    def imag_mult_dd(x0, x1, y0, y1):
        x, y, u, v = x0, x1, y0, y1
        return x*u - y*v, x*v + y*u

    def imag_add_dd(x0, x1, y0, y1):
        return x0 + y0, x1+y1

    third_dd = (DoubleDouble(1)/DoubleDouble(3))
    sqrt3_dd = DoubleDouble(1.7320508075688772, 1.0035084221806902e-16) # DoubleDouble(3).root(2)
    sqrt3_quarter_dd = DoubleDouble(0.4330127018922193, 2.5087710554517254e-17)
    quarter_dd = DoubleDouble(1)/DoubleDouble(4)


    def cbrt_dd(x):
        # http://web.mit.edu/tabbott/Public/quaddouble-debian/qd-2.3.4-old/docs/qd.pdf
        # start off with a "good" guess
        y = 1/DoubleDouble(float(x)**(1.0/3.))
        y = y + third_dd*y*(1-x*y*y*y)
        y = y + third_dd*y*(1-x*y*y*y)
    #     y = y + third_dd*y*(1-x*y*y*y)
        return 1/y


    def imag_cbrt_dd(xr, xc):
        y_guess = (float(xr)+float(xc)*1.0j)**(-1.0/3.)

        yr, yc = DoubleDouble(y_guess.real), DoubleDouble(y_guess.imag)
    #     print(repr(yr), repr(yc))
        t0r, t0c = imag_mult_dd(yr, yc, yr, yc) # have y*y
        t0r, t0c = imag_mult_dd(t0r, t0c, yr, yc) # have y*y*y
        t0r, t0c = imag_mult_dd(xr, xc, t0r, t0c) # have x*y*y*y
        t0r, t0c = imag_add_dd(1.0, 0.0, -t0r, -t0c) # have 1-x*y*y*y
        t0r, t0c = imag_mult_dd(yr, yc, t0r, t0c) # have y*(1-x*y*y*y)
        t0r, t0c = imag_mult_dd(third_dd, 0.0, t0r, t0c) # have third_dd*y*(1-x*y*y*y)
        yr, yc = imag_add_dd(yr, yc, t0r, t0c) # have y

    #     print(repr(yr), repr(yc))

#        t0r, t0c = imag_mult_dd(yr, yc, yr, yc) # have y*y
#        t0r, t0c = imag_mult_dd(t0r, t0c, yr, yc) # have y*y*y
#        t0r, t0c = imag_mult_dd(xr, xc, t0r, t0c) # have x*y*y*y
#        t0r, t0c = imag_add_dd(1.0, 0.0, -t0r, -t0c) # have 1-x*y*y*y
#        t0r, t0c = imag_mult_dd(yr, yc, t0r, t0c) # have y*(1-x*y*y*y)
#        t0r, t0c = imag_mult_dd(third_dd, 0.0, t0r, t0c) # have third_dd*y*(1-x*y*y*y)
#        yr, yc = imag_add_dd(yr, yc, t0r, t0c) # have y

        return imag_div_dd(1.0, 0.0, yr, yc)
except:
    pass


def volume_solutions_mpmath(T, P, b, delta, epsilon, a_alpha, dps=30):
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
        print('trying numerical mpmath')
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

    sort_fun = lambda x: (x.real, x.imag)
    return list(sorted(hits, key=sort_fun))

def volume_solutions_mpmath_float(T, P, b, delta, epsilon, a_alpha):
    Vs = volume_solutions_mpmath(T, P, b, delta, epsilon, a_alpha)
    return [float(Vi.real) + float(Vi.imag)*1.0j for Vi in Vs]

def volume_solutions_NR(T, P, b, delta, epsilon, a_alpha, tries=0):
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
                print(e, 'was not 2 phase')

        try:
            return volume_solutions_mpmath_float(T, P, b, delta, epsilon, a_alpha)
        except:
            pass
    try:
        if tries == 0:
            Vs = volume_solutions_Cardano(T, P, b, delta, epsilon, a_alpha)
#                Vs = [Vi+1e-45j for Vi in volume_solutions_Cardano(T, P, b, delta, epsilon, a_alpha, quick=True)]
        elif tries == 1:
            Vs = volume_solutions_fast(T, P, b, delta, epsilon, a_alpha)
        elif tries == 2:
            # sometimes used successfully
            Vs = volume_solutions_a1(T, P, b, delta, epsilon, a_alpha)
        # elif tries == 3:
        #     # never used successfully
        #     Vs = GCEOS.volume_solutions_a2(T, P, b, delta, epsilon, a_alpha)

        # TODO fall back to tlow T
    except:
#            Vs = GCEOS.volume_solutions_Cardano(T, P, b, delta, epsilon, a_alpha)
        if tries == 0:
            Vs = volume_solutions_fast(T, P, b, delta, epsilon, a_alpha)
        else:
            Vs = volume_solutions_Cardano(T, P, b, delta, epsilon, a_alpha)
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

def volume_solutions_NR_low_P(T, P, b, delta, epsilon, a_alpha,
                              tries=0):

    P_inv = 1/P
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
            V = Vi = b*20
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
    three values, all with some complex part.

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
    Using explicit formulas, as can be derived in the following example,
    is faster than most numeric root finding techniques, and
    finds all values explicitly. It takes several seconds.

    >>> from sympy import *
    >>> P, T, V, R, b, a, delta, epsilon, alpha = symbols('P, T, V, R, b, a, delta, epsilon, alpha')
    >>> Tc, Pc, omega = symbols('Tc, Pc, omega')
    >>> CUBIC = R*T/(V-b) - a*alpha/(V*V + delta*V + epsilon) - P
    >>> #solve(CUBIC, V)

    Note this approach does not have the same issues as formulas using trig
    functions or numerical routines.

    References
    ----------
    .. [1] Zhi, Yun, and Huen Lee. "Fallibility of Analytic Roots of Cubic
       Equations of State in Low Temperature Region." Fluid Phase
       Equilibria 201, no. 2 (September 30, 2002): 287-94.
       https://doi.org/10.1016/S0378-3812(02)00072-9.

    '''
#        RT_inv = R_inv/T
#        P_RT_inv = P*RT_inv
#        eta = b
#        B = b*P_RT_inv
#        deltas = delta*P_RT_inv
#        thetas = a_alpha*P_RT_inv*RT_inv
#        epsilons = epsilon*P_RT_inv*P_RT_inv
#        etas = eta*P_RT_inv
#
#        a = 1.0
#        b2 = (deltas - B - 1.0)
#        c = (thetas + epsilons - deltas*(B + 1.0))
#        d = -(epsilons*(B + 1.0) + thetas*etas)
#        open('bcd.txt', 'a').write('\n%s' %(str([float(b2), float(c), float(d)])))




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
            + 0.5*((x9*(-4.*x0*t1*t1*t1 + t2*t2))+0.0j)**0.5
            )+0.0j)**third

    x20 = -t1/x19#
    x22 = x5 + x5
    x25 = 4.*x0*x20
    return [(x0*x20 - x19 + x5)*third,
            (x19*x24 + x22 - x25*x24_inv)*sixth,
            (x19*x26 + x22 - x25*x26_inv)*sixth]
#        else:
#            return [-(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)/(3*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)) - (sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)/3 - (-P*b + P*delta - R*T)/(3*P),
#                     -(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)/(3*(-1/2 - sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)) - (-1/2 - sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)/3 - (-P*b + P*delta - R*T)/(3*P),
#                     -(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)/(3*(-1/2 + sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)) - (-1/2 + sqrt(3)*1j/2)*(sqrt(-4*(-3*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P + (-P*b + P*delta - R*T)**2/P**2)**3 + (27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/P - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/P**2 + 2*(-P*b + P*delta - R*T)**3/P**3)**2)/2 + 27*(-P*b*epsilon - R*T*epsilon - a_alpha*b)/(2*P) - 9*(-P*b + P*delta - R*T)*(-P*b*delta + P*epsilon - R*T*delta + a_alpha)/(2*P**2) + (-P*b + P*delta - R*T)**3/P**3)**(1/3)/3 - (-P*b + P*delta - R*T)/(3*P)]



def volume_solutions_doubledouble(T, P, b, delta, epsilon, a_alpha):
#        print(T, P, b, delta, epsilon, a_alpha)
    T = DoubleDouble(T)
    P = DoubleDouble(P)
    b = DoubleDouble(b)
    delta = DoubleDouble(delta)
    a_alpha = DoubleDouble(a_alpha)
    epsilon = DoubleDouble(epsilon)
    delta = DoubleDouble(delta)
    R = DoubleDouble(8.31446261815324)
    x0 = 1/P
    x1 = P*b
    x2 = R*T
    x3 = P*delta
    x4 = x1 + x2 - x3
    x5 = x0*x4
    x22 = x5 + x5
    x6 = a_alpha*b
    x7 = epsilon*x1
    x8 = epsilon*x2
    x9 = x0*x0
    x10 = P*epsilon
    x11 = delta*x1
    x12 = delta*x2
    x17 = -x4
    x17_2 = x17*x17
    x18 = x0*x17_2
    tm1 = x12 - a_alpha + (x11  - x10)
    t0 = x6 + x7 + x8
    t1 = (3*tm1  + x18)
    t2 = ((9*x0*x17*tm1) + 2*x17_2*x17*x9  - 27*t0)

    x4x9  = x4*x9
#        print('x9, x0, t1, t2', float(x9), float(x0), float(t1), float(t2))

    to_sqrt = x9*(-4*t1*t1*t1*x0 + t2*t2) # For low P, this value overflows. No ideas how to compute and not interested in it.
#        print('to_sqrt', float(to_sqrt))
    if to_sqrt < 0.0:
        sqrted = (-to_sqrt).sqrt()
        imag_rt = True
    else:
        sqrted = (to_sqrt).sqrt()
        imag_rt = False

    easy_adds = -13.5*x0*t0 - 4.5*x4x9*tm1 - x4*x4x9*x5
    if imag_rt:
        v0r, v0c = easy_adds, 0.5*sqrted
    else:
        v0r, v0c = easy_adds + 0.5*sqrted, 0.0

    x19r, x19c = imag_cbrt_dd(v0r, v0c)
    x20r, x20c = imag_div_dd(-t1, -0.0, x19r, x19c)

    f0r, f0c = imag_mult_dd(x20r, x20c, x0, 0.0)
    x25r, x25c = 4.0*f0r, 4.0*f0c

    x24r, x24c = 1, sqrt3_dd
    x24_invr, x24_invc = quarter_dd, -sqrt3_quarter_dd
#        x26 = -1.73205080756887729352744634151j + 1.
    x26r, x26c = 1, -sqrt3_dd
#        x26_inv = 0.25 + 0.433012701892219323381861585376j
    x26_invr, x26_invc = quarter_dd, sqrt3_quarter_dd

    g0 = float(f0r - x19r + x5)
    g1 = float(f0c - x19c)

    f1r, f1c = imag_mult_dd(x19r, x19c, x24r, x24c)
    f2r, f2c = imag_mult_dd(x25r, x25c, x24_invr, x24_invc)
#        f2 = x25*x24_inv
    g2 = float(f1r + x22 - f2r) # try to take away decimals anywhere and more error appears.
    g3 = float(f1c - f2c)

#        f3 = x19*x26
    f3r, f3c = imag_mult_dd(x19r, x19c, x26r, x26c)
    f4r, f4c = imag_mult_dd(x25r, x25c, x26_invr, x26_invc) #x25*x26_inv

    g4 = float(f3r + x22 - f4r)
    g5 = float(f3c - f4c)
    return ((g0 + g1*1j)*third,
            (g2 + g3*1j)*sixth,
            (g4 + g5*1j)*sixth)
#        return [(g0 + g1*1j)*third,
#                (g2 + g3*1j)*sixth,
##                (x19*x24 + x22 - x25*x24_inv)*sixth,
#                (x19*x26 + x22 - x25*x26_inv)*sixth]

def volume_solutions_Cardano(T, P, b, delta, epsilon, a_alpha):
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



#        if 0:
#            for i in range(3):
#                from fluids.numerics import bisect
#                def err(Z):
#                    err = Z*(Z*(Z + b) + c) + d
#                    return err
#                for fact in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-4, 1e-3):
#                    try:
#                        roots[i] = bisect(err, roots[i].real*(1+fact), roots[i].real*(1-fact), xtol=1e-15)
#                        break
#                    except Exception as e:
##                        print(e)
#                        pass
#                for _ in range(3):
#                    Z = roots[i]
##                    x0 = Z*(Z + b) + c
##                    err = Z*x0 + d
##                    derr = Z*(Z + Z + b) + x0
##
##                    roots[i] = Z - err/derr
##
#
#                    x0 = Z*(Z + b) + c
#                    err = Z*x0 + d
#                    derr = Z*(Z + Z + b) + x0
#                    d2err = 2.0*(3.0*Z + b)
#
#                    step = err/derr
#                    step = step/(1.0 - 0.5*step*d2err/derr)
#                    roots[i] = Z - step


    RT_P = R*T/P
    return [V*RT_P for V in roots]

def volume_solutions_a1(T, P, b, delta, epsilon, a_alpha):
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
    roots = list(roots_cubic_a1(b, c, d))

    RT_P = R*T/P
    return [V*RT_P for V in roots]

def volume_solutions_a2(T, P, b, delta, epsilon, a_alpha):
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

def volume_solutions_ideal(T, P, b, delta, epsilon, a_alpha):
    # Saves some time
    return [R*T/P, 0.0, 0.0]