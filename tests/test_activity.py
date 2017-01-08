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

from numpy.testing import assert_allclose
import pytest
import pandas as pd
from thermo.activity import *

def test_Rachford_Rice_flash_error():
    err = Rachford_Rice_flash_error(0.5, zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    assert_allclose(err, 0.04406445591174976)

def test_Rachford_Rice_solution():
    V_over_F, xs, ys = Rachford_Rice_solution(zs=[0.5, 0.3, 0.2], Ks=[1.685, 0.742, 0.532])
    xs_expect = [0.33940869696634357, 0.3650560590371706, 0.2955352439964858]
    ys_expect = [0.5719036543882889, 0.27087159580558057, 0.15722474980613044]
    assert_allclose(V_over_F, 0.6907302627738544)
    assert_allclose(xs, xs_expect)
    assert_allclose(ys, ys_expect)
    
    

def test_flash_solution_algorithms():
    algorithms = [Rachford_Rice_solution, Li_Johns_Ahmadi_solution]
    for algo in algorithms:
        # Said to be in:  J.D. Seader, E.J. Henley, D.K. Roper, Separation Process Principles, third ed., John Wiley & Sons, New York, 2010.
        zs = [0.1, 0.2, 0.3, 0.4]
        Ks = [4.2, 1.75, 0.74, 0.34]
        V_over_F_expect = 0.12188885
        xs_expect = [0.07194015, 0.18324807, 0.30981849, 0.43499379]
        
        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-4)
        assert_allclose(xs, xs_expect, rtol=1E-4)
                    
        # Said to be in:  B.A. Finlayson, Introduction to Chemical Engineering Computing, second ed., John Wiley & Sons, New York, 2012.
        zs = [0.1, 0.3, 0.4, 0.2]
        Ks = [6.8, 2.2, 0.8, 0.052]
        V_over_F_expect = 0.42583973
        xs_expect = [0.02881952, 0.19854300, 0.43723872, 0.33539943]
        
        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-5)
        assert_allclose(xs, xs_expect, rtol=1E-5)
    
        # Said to be in: J. Vidal, Thermodynamics: Applications in Chemical Engineering and the Petroleum Industry, Technip, Paris, 2003.
        zs = [0.2, 0.3, 0.4, 0.05, 0.05]
        Ks = [2.5250, 0.7708, 1.0660, 0.2401, 0.3140]
        V_over_F_expect = 0.52360688
        xs_expect = [0.11120375, 0.34091324, 0.38663852, 0.08304114, 0.07802677]
        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-2)
        assert_allclose(xs, xs_expect, rtol=1E-2)
    
    
        # Said to be in: R. Monroy-Loperena, F.D. Vargas-Villamil, On the determination of the polynomial defining of vapor-liquid split of multicomponent mixtures, Chem.Eng. Sci. 56 (2001) 5865–5868.
        zs = [0.05, 0.10, 0.15, 0.30, 0.30, 0.10]
        Ks = [6.0934, 2.3714, 1.3924, 1.1418, 0.6457, 0.5563]
        V_over_F_expect = 0.72073810
        xs_expect = [0.01070433, 0.05029118, 0.11693011, 0.27218275, 0.40287788, 0.14701374]
        
        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-6)
        assert_allclose(xs, xs_expect, rtol=1E-6)
        
        # Said to be in: R. Monroy-Loperena, F.D. Vargas-Villamil, On the determination of the polynomial defining of vapor-liquid split of multicomponent mixtures, Chem.Eng. Sci. 56 (2001) 5865–5868.
        zs = [0.3727, 0.0772, 0.0275, 0.0071, 0.0017, 0.0028, 0.0011, 0.0015, 0.0333, 0.0320, 0.0608, 0.0571, 0.0538, 0.0509, 0.0483, 0.0460, 0.0439, 0.0420, 0.0403]
        Ks = [7.11, 4.30, 3.96, 1.51, 1.20, 1.27, 1.16, 1.09, 0.86, 0.80, 0.73, 0.65, 0.58, 0.51, 0.45, 0.39, 0.35, 0.30, 0.26]
        V_over_F_expect = 0.84605135
        xs_expect = [0.06041132, 0.02035881, 0.00784747, 0.00495988, 0.00145397, 0.00227932, 0.00096884, 0.00139386, 0.03777425, 0.03851756, 0.07880076, 0.08112154, 0.08345504, 0.08694391, 0.09033579, 0.09505925, 0.09754111, 0.10300074, 0.10777648]
        
        V_over_F, xs, ys = algo(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-6)
        assert_allclose(xs, xs_expect, rtol=1E-5)
        
        # Random example from MultiComponentFlash.xlsx, https://6507a56d-a-62cb3a1a-s-sites.googlegroups.com/site/simulationsmodelsandworksheets/MultiComponentFlash.xlsx?attachauth=ANoY7coiZq4OX8HjlI75HGTWiegJ9Tqz6cyqmmmH9ib-dhcNL89TIUTmQw3HxrnolKHgYuL66drYGasDgTkf4_RrWlciyRKwJCbSi5YgTG1GfZR_UhlBuaoKQvrW_L8HdboB3PYejRbzVQaCshwzYcOeGCZycdXQdF9scxoiZLpy7wbUA0xx8j9e4nW1D9PjyApC-MjsjqjqL10HFcw1KVr5sD0LZTkZCqFYA1HReqLzOGZE01_b9sfk351BB33mwSgWQlo3DLVe&attredirects=0&d=1
        Ks = [0.90000, 2.70000, 0.38000, 0.09800, 0.03800, 0.02400, 0.07500, 0.00019, 0.00070]
        zs = [0.0112, 0.8957, 0.0526, 0.0197, 0.0068, 0.0047, 0.0038, 0.0031, 0.0024]
        V_over_F_expect = 0.964872854762834
        V_over_F, xs, ys = Rachford_Rice_solution(zs=zs, Ks=Ks)
        assert_allclose(V_over_F, V_over_F_expect, rtol=1E-11)


#        zs = [0.064995651826382228, 0.0019192147863876529, 0.022626144636592599, 0.0090710295908533606, 0.067574153762178582, 0.0030785453880062706, 0.070693047991453636, 0.03152261521326441, 0.029224025275160968, 0.046600689685048095, 0.0030751343207852857, 0.032335598119219516, 0.019459194337798733, 0.038445699756423687, 0.038072990200753005, 0.009264685474861218, 0.035704603662097371, 0.02949585716137676, 0.027722060064798125, 0.036013457809228072, 0.064697852820750523, 0.07091200032558817, 0.067778533144331937, 0.0093228561600868093, 0.056325981859098359, 0.0032604054280783285, 0.024251303134501322, 0.053090564205998884, 0.033466103858896021]
#        Ks = [1.5652651247667293, 1.6374962916199021, 1.8785772003024455, 0.94479207838626045, 1.0939709076894193, 0.8453740416937745, 1.1804627111769761, 1.9733010920626894, 1.2896141019705114, 1.6621232295714163, 0.85730344646644974, 1.6991745443812776, 0.83973069781006804, 1.309532265031327, 0.93185789102038497, 1.79605818255131, 1.4949445828443497, 0.85103746762982868, 0.66034048706407034, 0.075067714748298986, 1.7305657580590847, 1.967975188102125, 0.65966765065643518, 0.79464595149530504, 1.1893959488040962, 0.67623202134735672, 1.9245957955761228, 1.3811654298922205, 1.7899045224630479]
