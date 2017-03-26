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
from thermo.chemical import *
from thermo.identifiers import pubchem_dict
from scipy.integrate import quad
from math import *
from scipy.constants import R

def test_Chemical_properties():
    w = Chemical('water')
    assert_allclose(w.Tm, 273.15)
    assert_allclose(w.Tb, 373.124)
    
    assert_allclose(w.Tc, 647.14)
    assert_allclose(w.Pc, 22048320.0)
    assert_allclose(w.Vc, 5.6e-05)
    assert_allclose(w.omega, 0.344)
    
    assert_allclose(w.Zc, 0.2294728175007233)
    assert_allclose(w.rhoc, 321.7014285714285)
    assert_allclose(w.rhocm, 17857.142857142855)
    
    assert_allclose(w.StielPolar, 0.023222134391615246)
    
    pentane = Chemical('pentane')
    assert_allclose(pentane.Tt, 143.47)
    assert_allclose(pentane.Pt, 0.07098902774226569)

    assert_allclose(pentane.Hfus, 116426.08509804323)
    assert_allclose(pentane.Hfusm, 8400.0)
    
    phenol = Chemical('phenol')
    assert_allclose(phenol.Hsub, 740612.9172243401)
    assert_allclose(phenol.Hsubm, 69700.0)

    # Hf test - always molar, TODO mass-based
    assert_allclose(phenol.Hf, -96400.0)
    assert_allclose(phenol.Hc, -3121919.0)

    assert_allclose(phenol.Tflash, 348.15)
    assert_allclose(phenol.Tautoignition, 868.15)
    
    assert_allclose(phenol.LFL, 0.013000000000000001)
    assert_allclose(phenol.UFL, 0.095)

    benzene = Chemical('benzene')
    assert benzene.STEL == (2.5, 'ppm')
    assert benzene.TWA == (0.5, 'ppm')
    assert benzene.Skin
    
    assert Chemical('acetaldehyde').Ceiling == (25.0, 'ppm')
    
    d = benzene.Carcinogen 
    assert len(d) == 2
    assert d['National Toxicology Program 13th Report on Carcinogens'] == 'Known'
    assert d['International Agency for Research on Cancer'] == 'Carcinogenic to humans (1)'

    assert_allclose(w.dipole, 1.85)
    assert_allclose(w.Stockmayer, 501.01)
    
    CH4 = Chemical('methane')
    assert_allclose(CH4.GWP, 25.0)
    
    assert Chemical('Bromochlorodifluoromethane').ODP == 7.9
    
    assert_allclose(phenol.logP, 1.48)

    l = phenol.legal_status
    assert sorted(l.keys()) == ['DSL', 'EINECS', 'NLP', 'SPIN', 'TSCA']
    assert sorted(l.values()) == ['LISTED', 'LISTED', 'LISTED', 'LISTED', 'UNLISTED']

    phenol.economic_status

    assert_allclose(benzene.conductivity, 7.6e-06)
    assert_allclose(benzene.RI, 1.5011)


def test_Chemical_properties_T_dependent_constants():
    w = Chemical('water')
    assert_allclose(w.Psat_298, 3167.418523735963)
    
    assert_allclose(w.Vml_Tb, 1.8829559687798784e-05)
    assert_allclose(w.Vml_Tm, 1.7908144191247533e-05)
    assert_allclose(w.Vml_STP, 1.8069338439592963e-05)
    assert_allclose(w.Vmg_STP, 0.023505766772305356)
    
    
    assert_allclose(w.Hvap_Tb, 2256470.870516969)
    assert_allclose(w.Hvap_Tbm, 40650.95454420694)
    
    assert w.phase_STP == 'l'
    
    assert_allclose(w.molecular_diameter, 3.24681)

def test_Chemical_properties_T_dependent():
    # T-only dependent properties (always or at the moment)
    # Keep the order of the tests matching the order of the code
    w = Chemical('water', T=300, P=1E5)
    Pd = Chemical('palladium')
    
    assert_allclose(w.Psat, 3533.918074415897)
    assert_allclose(w.Hvapm, 43908.418874478055)
    assert_allclose(w.Hvap, 2437287.6177599267)
    
    assert_allclose(Pd.Cpsm, 24.930765664000003) 
    assert_allclose(w.Cplm, 75.2955317728452)
    assert_allclose(w.Cpgm, 33.590714617128235)
    
    assert_allclose(Pd.Cps, 234.26767209171211)
    assert_allclose(w.Cpl, 4179.537135856072)
    assert_allclose(w.Cpg, 1864.5680010040496)
    
    assert_allclose(w.Cvgm, 25.276254817128233)
    assert_allclose(w.Cvg, 1403.0453491218693)
    assert_allclose(w.isentropic_exponent, 1.3289435029103198)
    
    assert_allclose(Pd.Vms, 8.86833333333333e-06)
    assert_allclose(w.Vml, 1.8077520828345428e-05)
    assert_allclose(w.Vmg, 0.02401190487463453)
    
    assert_allclose(Pd.rhos, 12000.000000000005)
    assert_allclose(w.rhol, 996.5570041967351)
    assert_allclose(w.rhog, 0.7502645081286664)
    
    assert_allclose(Pd.rhosm, 112760.75925577903)
    assert_allclose(w.rholm, 55317.319752828436)
    assert_allclose(w.rhogm, 41.646008728627386)
    
    assert_allclose(Pd.Zs, 0.00036248477437931853)
    assert_allclose(w.Zl, 0.0007247422467681115)
    assert_allclose(w.Zg, 0.9626564423998831)
    
    assert_allclose(w.Bvirial, -0.0009314745253654686)

    assert_allclose(w.isobaric_expansion_l, 0.00027479530461365189, rtol=1E-3)
    assert_allclose(w.isobaric_expansion_g, 0.004082110714805371, rtol=1E-3)
    
    assert_allclose(w.mul, 0.0008537426062537152)
    assert_allclose(w.mug, 9.759577077891826e-06)
    
    assert_allclose(w.kl, 0.6094991151038377)
    assert_allclose(w.kg, 0.018984360775888904)
    
    assert_allclose(w.sigma, 0.07176932405246211)
    
    assert_allclose(w.permittivity, 77.70030000000001)
    
    assert_allclose(w.JTl, -2.2029508371866032e-07, rtol=1E-3)
    assert_allclose(w.JTg, 0.00016057626157512468, rtol=1E-3)
    
    assert_allclose(w.nul, 8.566921938819405e-07, rtol=1E-3)
    assert_allclose(w.nug, 1.3008181744108452e-05, rtol=1E-3)
    
    assert_allclose(w.Prl, 5.854395582989558, rtol=1E-3)
    assert_allclose(w.Prg, 0.9585466341264076, rtol=1E-3)
    
    assert_allclose(w.solubility_parameter, 47863.51384219548)
    assert_allclose(w.Parachor, 9.363768522707514e-06)
    
    # Poynting factor
    assert_allclose(Chemical('pentane', T=300, P=1E7).Poynting, 1.5743051250679803)
    
    c = Chemical('pentane', T=300, P=1E7)
    Poy = exp(quad(lambda P : c.VolumeLiquid(c.T, P), c.Psat, c.P)[0]/R/c.T)
    assert_allclose(Poy, 1.5821826990975127)


def test_Chemical_properties_T_phase():
    # T-only dependent properties (always or at the moment)
    # Keep the order of the tests matching the order of the code
    w = Chemical('water', T=300, P=1E5)
    
    assert_allclose(w.Cp, 4179.537135856072)
    assert_allclose(w.Cpm, 75.2955317728452)
    
    assert_allclose(w.Vm, 1.8077520828345428e-05)
    assert_allclose(w.rho, 996.5570041967351)
    assert_allclose(w.rhom, 55317.319752828436)
    assert_allclose(w.Z, 0.0007247422467681115)
    
    assert_allclose(w.isobaric_expansion, 0.00027479530461365189, rtol=1E-3)
    assert_allclose(w.JT, -2.2029508371866032e-07, rtol=1E-3)

    assert_allclose(w.mu, 0.0008537426062537152)
    assert_allclose(w.k, 0.6094991151038377)
    
    assert_allclose(w.nu, 8.566921938819405e-07)
    assert_allclose(w.alpha, 1.4633315800714463e-07)

    assert_allclose(w.Pr, 5.854395582989558)

def test_Mixture():
    Mixture(['water', 'ethanol'], ws=[.5, .5], T=320, P=1E5)
    Mixture(['water', 'phosphoric acid'], ws=[.5, .5], T=320, P=1E5)
    Mixture('air', T=320, P=1E5)
    
    Mixture(['ethanol', 'water'], ws=[0.5, 0.5], T=500)
 
def test_Stream():   
    Stream(['H2', 'NH3', 'CO', 'Ar', 'CH4', 'N2'],
           zs=[.7371, 0, .024, .027, .013, .2475], 
    T=500, P=20.5E5, m=300)



def test_H_Chemical():
    from thermo import chemical
    chemical.caching = False
    w = Chemical('water', T=298.15, P=101325.0)
    w.set_ref(T_ref=298.15, P_ref=101325, phase_ref='l', H_ref=0, S_ref=0)
    assert 0 == w.H
    assert 0 == w.S
    w.calculate(297.15, w.P)
    assert_allclose(w.Hm, 1000*(1.814832712-1.890164074), rtol=1E-3)
    w.calculate(274.15, w.P)
    assert_allclose(w.Hm, 1000*(0.07708322535-1.890164074), rtol=1E-4)
    w.calculate(273.15001, w.P) 
    assert w.phase == 'l'
    H_pre_transition = w.Hm
    w.calculate(273.15, w.P)
    assert w.phase == 's'
    dH_transition = w.Hm - H_pre_transition
    assert_allclose(dH_transition, -6010.0, rtol=1E-5)
    # There is not solid heat capacity for water in the database


    w = Chemical('water', T=273.0, P=101325.0)
    w.set_ref(T_ref=273.0, P_ref=101325, phase_ref='s', H_ref=0, S_ref=0)
    w.set_thermo()
    assert 0 == w.H
    assert 0 == w.S
    assert w.phase == 's'
    w.calculate(273.15)
    H_pre_transition = w.Hm
    w.calculate(273.15001, w.P)
    dH_transition = w.Hm - H_pre_transition
    assert_allclose(dH_transition, 6010.0, rtol=1E-5)
    assert w.phase == 'l'
    w.calculate(274.15)
    H_initial_liquid = w.Hm
    w.calculate(298.15, w.P)
    initial_liq_to_STP = w.Hm - H_initial_liquid
    assert_allclose(initial_liq_to_STP, -1000*(0.07708322535-1.890164074), rtol=1E-3)


    w = Chemical('water', T=373, P=101325.0)
    H_pre_vap = w.Hm
    w.calculate(w.Tb+1E-1)
    dH = w.Hm - H_pre_vap
    assert_allclose(dH, 40650, 1E-3) # google search answer


    Hm_as_vapor = w.Hm
    w.calculate(w.T+20)
    dH_20K_gas = w.Hm - Hm_as_vapor
    assert_allclose(dH_20K_gas, 1000*(48.9411675-48.2041134), rtol=1E-1) # Web tables, but hardly matches because of the excess


@pytest.mark.slow
@pytest.mark.meta_Chemical
def test_all_chemicals():
    for i in pubchem_dict.keys():
        c = Chemical(i)
        
        # T and P dependent properties - just test they can be called
        c.Psat
        c.Hvapm
        c.Hvap
        c.Cpsm
        c.Cplm
        c.Cpgm
        c.Cps
        c.Cpl
        c.Cpg
        c.Cvgm
        c.Cvg
        c.isentropic_exponent
        c.Vms
        c.Vml
        c.Vmg
        c.rhos
        c.rhol
        c.rhog
        c.rhosm
        c.rholm
        c.rhogm
        c.Zs
        c.Zl
        c.Zg
        c.Bvirial
        c.isobaric_expansion_l
        c.isobaric_expansion_g
        c.mul
        c.mug
        c.kl
        c.kg
        c.sigma
        c.permittivity
        c.JTl
        c.JTg
        c.nul
        c.nug
        c.alphal
        c.alphag
        c.Prl
        c.Prg
        c.solubility_parameter
        c.Parachor
        
        # Any phase dependent property
        c.Cp
        c.Cpm
        c.Vm
        c.rho
        c.rhom
        c.Z
        c.isobaric_expansion
        c.JT
        c.mu
        c.k
        c.nu
        c.alpha
        c.Pr
        # Some constant stuff
        c.Van_der_Waals_area
        c.Van_der_Waals_volume
        c.UNIFAC_R
        c.UNIFAC_Q
        c.UNIFAC_groups
        c.UNIFAC_Dortmund_groups
        c.PSRK_groups