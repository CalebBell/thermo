from fluids.numerics import IS_PYPY

import thermo

if not IS_PYPY:

    import thermo.numba
from random import random

import numpy as np
from fluids.numerics import normalize

from thermo import UNIQUAC
from thermo.eos_mix import *


class BaseTimeSuite:
    def setup(self):
        for k in dir(self.__class__):
            if 'time' in k and 'numba' in k:
                c = getattr(self, k)
                c()

class UNIQUACTimeSuite(BaseTimeSuite):
    def setup(self):

        N = 3
        T = 331.42
        xs = [0.229, 0.175, 0.596]
        rs = [2.5735, 2.87, 1.4311]
        qs = [2.336, 2.41, 1.432]

        # madeup numbers to match Wilson example roughly
        tausA = [[0.0, -1.05e-4, -2.5e-4], [3.9e-4, 0.0, 1.6e-4], [-1.123e-4, 6.5e-4, 0]]
        tausB = [[0.0, 235.0, -169.0], [-160, 0.0, -715.0], [11.2, 144.0, 0.0]]
        tausC = [[0.0, -4.23e-4, 2.9e-4], [6.1e-4, 0.0, 8.2e-5], [-7.8e-4, 1.11e-4, 0]]
        tausD = [[0.0, -3.94e-5, 2.22e-5], [8.5e-5, 0.0, 4.4e-5], [-7.9e-5, 3.22e-5, 0]]
        tausE = [[0.0, -4.2e2, 8.32e2], [2.7e2, 0.0, 6.8e2], [3.7e2, 7.43e2, 0]]
        tausF = [[0.0, 9.64e-8, 8.94e-8], [1.53e-7, 0.0, 1.11e-7], [7.9e-8, 2.276e-8, 0]]
        ABCDEF = (tausA, tausB, tausC, tausD, tausE, tausF)


        self.GE2 = UNIQUAC(T=T, xs=xs, rs=rs, qs=qs, ABCDEF=ABCDEF)
        if not IS_PYPY:
            self.GE2nb = UNIQUAC(T=T, xs=np.array(xs), rs=np.array(rs), qs=np.array(qs), ABCDEF=(np.array(v) for v in ABCDEF))
            self.xs2nb = np.array([.1, .2, .7])


        def make_rsqs(N):
            cmps = range(N)
            rs = [float('%.3g'%(random()*2.5)) for _ in cmps]
            qs = [float('%.3g'%(random()*1.3)) for _ in cmps]
            return rs, qs

        def make_taus(N):
            cmps = range(N)
            data = []
            base = [1e-4, 200.0, -5e-4, -7e-5, 300, 9e-8]

            for i in cmps:
                row = []
                for j in cmps:
                    if i == j:
                        row.append([0.0]*6)
                    else:
                        row.append([float('%.3g'%(random()*n)) for n in base])
                data.append(row)
            return data

        N = 20
        rs, qs = make_rsqs(N)
        taus = make_taus(N)
        xs = normalize([random() for i in range(N)])
        T = 350.0
        self.GE20 = UNIQUAC(T=T, xs=xs, rs=rs, qs=qs, tau_coeffs=taus)
        self.xs20 = normalize([random() for i in range(N)])
        if not IS_PYPY:
            self.GE20nb = UNIQUAC(T=T, xs=np.array(xs), rs=np.array(rs), qs=np.array(qs), tau_coeffs=np.array(taus))
            self.xs20nb = np.array(self.xs20)

    def time_to_T_xs2(self):
        return self.GE2.to_T_xs(T=340.0, xs=[.1, .2, .7])

    def time_gammas2(self):
        return self.GE2.to_T_xs(T=340.0, xs=[.1, .2, .7]).gammas()

    def time_dHE_dT20(self):
        return self.GE2.to_T_xs(T=340.0, xs=[.1, .2, .7]).dHE_dT()

    def time_dgammas_dns2(self):
        return self.GE2.to_T_xs(T=340.0, xs=[.1, .2, .7]).dgammas_dns()



    def time_to_T_xs2_nb(self):
        return self.GE2nb.to_T_xs(T=340.0, xs=self.xs2nb)

    def time_gammas2_nb(self):
        return self.GE2nb.to_T_xs(T=340.0, xs=self.xs2nb).gammas()

    def time_dHE_dT20_nb(self):
        return self.GE2nb.to_T_xs(T=340.0, xs=self.xs2nb).dHE_dT()

    def time_dgammas_dns2(self):
        return self.GE2nb.to_T_xs(T=340.0, xs=self.xs2nb).dgammas_dns()



    def time_to_T_xs20(self):
        return self.GE20.to_T_xs(T=340.0, xs=self.xs20)

    def time_gammas20(self):
        return self.GE20.to_T_xs(T=340.0, xs=self.xs20).gammas()

    def time_dHE_dT20(self):
        return self.GE20.to_T_xs(T=340.0, xs=self.xs20).dHE_dT()

    def time_dgammas_dns20(self):
        return self.GE20.to_T_xs(T=340.0, xs=self.xs20).dgammas_dns()
del UNIQUACTimeSuite

class EOSTimeSuite(BaseTimeSuite):
    def setup(self):
        pass

def eos_ternary_args(eos, numpy=False):
    Tcs = [126.2, 304.2, 373.2]

    Pcs = [3394387.5, 7376460.0, 8936865.0]
    omegas = [0.04, 0.2252, 0.1]
    kijs = [[0.0, -0.0122, 0.1652], [-0.0122, 0.0, 0.0967], [0.1652, 0.0967, 0.0]]

    kwargs = dict(Tcs=Tcs, Pcs=Pcs, omegas=omegas, kijs=kijs)

    ans = kwargs.copy()
    if 'Translated' in eos.__name__:
        ans['cs'] = [3.1802632895165143e-06, 4.619807672093997e-06, 3.930402699800546e-06]
    if eos is APISRKMIX:
        ans['S1s'] = [1.678665, 1.2, 1.5]
        ans['S2s'] = [-0.216396, -.2, -.1]
    if eos in (PRSVMIX, PRSV2MIX):
        ans['kappa1s'] = [0.05104, .025, .035]
    if eos is PRSV2MIX:
        ans['kappa2s'] = [.8, .9, 1.1]
        ans['kappa3s'] = [.46, .47, .48]
    if numpy:
        for k, v in ans.items():
            if type(v) is list:
                ans[k] = np.array(v)
    return ans


class EOSMIXTernaryTimeSuite(BaseTimeSuite):
    zs = [.7, .3, .1]
    zs_np = np.array(zs)

    zs2 = [.6, .1, .3]
    zs2_np = np.array(zs)

    eos_args_np = {obj: eos_ternary_args(obj, True) for obj in eos_mix_list}
    eos_args = {obj: eos_ternary_args(obj, False) for obj in eos_mix_list}
    if not IS_PYPY:
        eos_to_eos_numba = {eos_mix_list[i]: thermo.numba.eos_mix.eos_mix_list[i] for i in range(len(eos_mix_list))}
    params = eos_mix_list

    eos_instances_PT = {}
    for eos in eos_mix_list:
        eos_instances_PT[eos] = eos(T=300.0, P=1e5, zs=zs, **eos_args[eos])

    eos_instances_np_PT = {}
    for eos in eos_mix_list:
        eos_instances_np_PT[eos] = eos(T=300.0, P=1e5, zs=zs_np, **eos_args_np[eos])

    eos_instances_numba_PT = {}
    for eos in eos_mix_list:
        eos_instances_numba_PT[eos] = eos_to_eos_numba[eos](T=300.0, P=1e5, zs=zs_np, **eos_args_np[eos])


    def setup(self, eos):
        self.time_eos_PT_numba(eos)
        self.time_eos_PV_numba(eos)
        self.time_eos_TV_numba(eos)


    def time_eos_PT(self, eos):
        return eos(T=300.0, P=1e5, zs=self.zs, **self.eos_args[eos])

    def time_eos_PT_numpy(self, eos):
        return eos(T=300.0, P=1e5, zs=self.zs_np, **self.eos_args_np[eos])

    def time_eos_PT_numba(self, eos):
        return self.eos_to_eos_numba[eos](T=300.0, P=1e5, zs=self.zs_np, **self.eos_args_np[eos])


    def time_eos_PT_fast_same_T(self, eos):
        return self.eos_instances_PT[eos].to_TP_zs_fast(T=300.0, P=1e5, zs=self.zs2, only_g=True, full_alphas=True)

    def time_eos_PT_numpy_fast_same_T(self, eos):
        return self.eos_instances_np_PT[eos].to_TP_zs_fast(T=300.0, P=1e5, zs=self.zs2_np, only_g=True, full_alphas=True)

    def time_eos_PT_numba_fast_same_T(self, eos):
        return self.eos_instances_numba_PT[eos].to_TP_zs_fast(T=300.0, P=1e5, zs=self.zs2_np, only_g=True, full_alphas=True)


    def time_eos_PT_faster_same_T(self, eos):
        return self.eos_instances_PT[eos].to_TP_zs_fast(T=300.0, P=1e5, zs=self.zs2, only_g=True, full_alphas=False)

    def time_eos_PT_numpy_faster_same_T(self, eos):
        return self.eos_instances_np_PT[eos].to_TP_zs_fast(T=300.0, P=1e5, zs=self.zs2_np, only_g=True, full_alphas=False)

    def time_eos_PT_numba_faster_same_T(self, eos):
        return self.eos_instances_numba_PT[eos].to_TP_zs_fast(T=300.0, P=1e5, zs=self.zs2_np, only_g=True, full_alphas=False)


    def time_eos_PT_fast_diff_T(self, eos):
        return self.eos_instances_PT[eos].to_TP_zs_fast(T=301.0, P=1e5, zs=self.zs2, only_g=True, full_alphas=True)

    def time_eos_PT_numpy_fast_diff_T(self, eos):
        return self.eos_instances_np_PT[eos].to_TP_zs_fast(T=301.0, P=1e5, zs=self.zs2_np, only_g=True, full_alphas=True)

    def time_eos_PT_numba_fast_diff_T(self, eos):
        return self.eos_instances_numba_PT[eos].to_TP_zs_fast(T=301.0, P=1e5, zs=self.zs2_np, only_g=True, full_alphas=True)


    def time_eos_PT_faster_diff_T(self, eos):
        return self.eos_instances_PT[eos].to_TP_zs_fast(T=301.0, P=1e5, zs=self.zs2, only_g=True, full_alphas=False)

    def time_eos_PT_numpy_faster_diff_T(self, eos):
        return self.eos_instances_np_PT[eos].to_TP_zs_fast(T=301.0, P=1e5, zs=self.zs2_np, only_g=True, full_alphas=False)

    def time_eos_PT_numba_faster_diff_T(self, eos):
        return self.eos_instances_numba_PT[eos].to_TP_zs_fast(T=301.0, P=1e5, zs=self.zs2_np, only_g=True, full_alphas=False)



    def time_eos_PV(self, eos):
        return eos(V=.025, P=1e5, zs=self.zs, **self.eos_args[eos])

    def time_eos_PV_numpy(self, eos):
        return eos(V=.025, P=1e5, zs=self.zs_np, **self.eos_args_np[eos])

    def time_eos_PV_numba(self, eos):
        return self.eos_to_eos_numba[eos](V=.025, P=1e5, zs=self.zs_np, **self.eos_args_np[eos])


    def time_eos_PV_to(self, eos):
        return self.eos_instances_PT[eos].to(V=.025, P=1e5, zs=self.zs2)

    def time_eos_PV_numpy_to(self, eos):
        return self.eos_instances_np_PT[eos].to(V=.025, P=1e5, zs=self.zs2_np)

    def time_eos_PV_numba_to(self, eos):
        return self.eos_instances_numba_PT[eos].to(V=.025, P=1e5, zs=self.zs2_np)



    def time_eos_TV(self, eos):
        return eos(T=300.0, V=.025, zs=self.zs, **self.eos_args[eos])

    def time_eos_TV_numpy(self, eos):
        return eos(T=300.0, V=.025, zs=self.zs_np, **self.eos_args_np[eos])

    def time_eos_TV_numba(self, eos):
        return self.eos_to_eos_numba[eos](T=300.0, V=.025, zs=self.zs_np, **self.eos_args_np[eos])


    def time_eos_TV_to(self, eos):
        return self.eos_instances_PT[eos].to(V=.025, T=301.0, zs=self.zs2)

    def time_eos_TV_numpy_to(self, eos):
        return self.eos_instances_np_PT[eos].to(V=.025, T=301.0, zs=self.zs2_np)

    def time_eos_TV_numba_to(self, eos):
        return self.eos_instances_numba_PT[eos].to(V=.025, T=301.0, zs=self.zs2_np)

