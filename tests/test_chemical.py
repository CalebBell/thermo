# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from fluids.numerics import assert_close, assert_close1d, assert_close2d
import pytest
from thermo.chemical import *
from chemicals.elements import periodic_table
import thermo
from chemicals.identifiers import pubchem_db
from scipy.integrate import quad
from math import *
from fluids.constants import R

@pytest.mark.deprecated
def test_Chemical_properties():
    w = Chemical('water')
    assert_close(w.Tm, 273.15)
    assert_close(w.Tb, 373.124)

    assert_close(w.Tc, 647.14)
    assert_close(w.Pc, 22048320.0)
    assert_close(w.Vc, 5.6e-05)
    assert_close(w.omega, 0.344)

    assert_close(w.Zc, 0.2294728175007233, rtol=1E-5)
    assert_close(w.rhoc, 321.7014285714285, rtol=1E-4)
    assert_close(w.rhocm, 17857.142857142855, rtol=1E-4)

    assert_close(w.StielPolar, 0.023222134391615246, rtol=1E-3)

    pentane = Chemical('pentane')
    assert_close(pentane.Tt, 143.47)

    # Vapor pressure correlation did not extend down far enough once made strict
    assert_close(pentane.Pt, 0.12218586819726474)

    assert_close(pentane.Hfus, 116426.08509804323, rtol=1E-3)
    assert_close(pentane.Hfusm, 8400.0, rtol=1E-3)

    phenol = Chemical('phenol')
    assert_close(phenol.Hsub, 736964.419015, rtol=1E-3)
    assert_close(phenol.Hsubm, 69356.6353093363)

    assert_close(phenol.Hfm, -165100.)
    assert_close(phenol.Hcm, -3053219.)

    assert_close(phenol.Tflash, 348.15)
    assert_close(phenol.Tautoignition, 868.15)

    assert_close(phenol.LFL, 0.013000000000000001)
    assert_close(phenol.UFL, 0.095)

    assert_close(phenol.R_specific, 88.34714960720952, rtol=1E-4)

    benzene = Chemical('benzene')
    assert benzene.STEL == (2.5, 'ppm')
    assert benzene.TWA == (0.5, 'ppm')
    assert benzene.Skin

    assert Chemical('acetaldehyde').Ceiling == (25.0, 'ppm')

    d = benzene.Carcinogen
    assert len(d) == 2
    assert d['National Toxicology Program 13th Report on Carcinogens'] == 'Known'
    assert d['International Agency for Research on Cancer'] == 'Carcinogenic to humans (1)'

    assert_close(w.dipole, 1.85)
    assert_close(w.Stockmayer, 501.01)

    CH4 = Chemical('methane')
    assert_close(CH4.GWP, 25.0)

    assert Chemical('Bromochlorodifluoromethane').ODP == 7.9

    assert_close(phenol.logP, 1.48)

    l = phenol.legal_status
    assert sorted(l.keys()) == ['DSL', 'EINECS', 'NLP', 'SPIN', 'TSCA']
    assert sorted(l.values()) == ['LISTED', 'LISTED', 'LISTED', 'LISTED', 'UNLISTED']

    phenol.economic_status

    assert_close(benzene.conductivity, 7.6e-06)
    assert_close(benzene.RI, 1.5011)



@pytest.mark.deprecated
def test_Chemical_properties_T_dependent_constants():
    w = Chemical('water')
    assert_close(w.Psat_298, 3167.418523735963, rtol=1e-4)

    assert_close(w.Vml_Tb, 1.8829559687798784e-05, rtol=1e-4)
    assert_close(w.Vml_Tm, 1.7908144191247533e-05, rtol=1e-4)
    assert_close(w.Vml_STP, 1.8069338439592963e-05, rtol=1e-4)

    assert_close(w.Vmg_STP, 0.024465403697038125, rtol=1e-4)


    assert_close(w.Hvap_Tb, 2256470.870516969, rtol=1e-4)
    assert_close(w.Hvap_Tbm, 40650.95454420694, rtol=1e-4)

    assert w.phase_STP == 'l'

    assert_close(w.molecular_diameter, 3.24681, rtol=1e-4)

@pytest.mark.deprecated
def test_Chemical_properties_T_dependent():
    # T-only dependent properties (always or at the moment)
    # Keep the order of the tests matching the order of the code
    w = Chemical('water', T=300, P=1E5)
    s = Chemical('water', T=500, P=1E5)
    Pd = Chemical('palladium')

    assert_close(w.Psat, 3533.918074415897, rtol=1e-4)
    assert_close(w.Hvapm, 43908.418874478055, rtol=1e-4)
    assert_close(w.Hvap, 2437287.6177599267, rtol=1e-4)

    assert_close(Pd.Cpsm, 24.930765664000003, rtol=1e-4)
    assert_close(w.Cplm, 75.2955317728452, rtol=1e-4)
    assert_close(w.Cpgm, 33.590714617128235, rtol=1e-4)

    assert_close(Pd.Cps, 234.26767209171211, rtol=1e-4)
    assert_close(w.Cpl, 4179.537135856072, rtol=1e-4)
    assert_close(w.Cpg, 1864.5680010040496, rtol=1e-4)

    assert_close(w.Cvgm, 25.276254817128233, rtol=1e-4)
    assert_close(w.Cvg, 1403.0453491218693, rtol=1e-4)
    assert_close(w.isentropic_exponent, 1.3289435029103198, rtol=1e-4)

    assert_close(Pd.Vms, 8.86833333333333e-06, rtol=1e-4)
    assert_close(w.Vml, 1.8077520828345428e-05, rtol=1e-4)
    assert_close(w.Vmg_ideal, 0.02494338785445972, rtol=1e-4)

    assert_close(Pd.rhos, 12000.000000000005, rtol=1e-4)
    assert_close(w.rhol, 996.5570041967351, rtol=1e-4)
    assert_close(s.rhog, 0.4351403649367513, rtol=1e-4)

    assert_close(Pd.rhosm, 112760.75925577903, rtol=1e-4)
    assert_close(w.rholm, 55317.319752828436, rtol=1e-4)
    assert_close(s.rhogm, 24.153960689856124, rtol=1e-4)

    assert_close(Pd.Zs, 0.00036248477437931853, rtol=1e-4)
    assert_close(w.Zl, 0.0007247422467681115, rtol=1e-4)
    assert_close(s.Zg, 0.9958810199872231, rtol=1e-4)

    assert_close(s.Bvirial, -0.0001712355267057688, rtol=1e-4)

    assert_close(w.isobaric_expansion_l, 0.00027479530461365189, rtol=1E-3)
    assert_close(s.isobaric_expansion_g, 0.0020332741204046327, rtol=1E-3)

    assert_close(w.mul, 0.0008537426062537152, rtol=1e-4)
    assert_close(s.mug, 1.729908278164999e-05, rtol=1e-4)

    assert_close(w.kl, 0.6094991151038377, rtol=1e-4)
    assert_close(s.kg, 0.036031817846801754, rtol=1e-4)

    assert_close(w.sigma, 0.07176932405246211, rtol=1e-4)

    assert_close(w.permittivity, 77.70030000000001, rtol=1e-4)
    assert_close(w.absolute_permittivity, 6.879730496854497e-10, rtol=1e-4)

    assert_close(w.JTl, -2.2029508371866032e-07, rtol=1E-3)
    assert_close(s.JTg, 1.9548097005716312e-05, rtol=1E-3)

    assert_close(w.nul, 8.566921938819405e-07, rtol=1E-3)
    assert_close(s.nug, 3.975517827256603e-05, rtol=1E-3)

    assert_close(w.Prl, 5.854395582989558, rtol=1E-3)
    assert_close(s.Prg, 0.9390303617687221, rtol=1E-3)

    assert_close(w.solubility_parameter, 47863.51384219548, rtol=1e-4)
    assert_close(w.Parachor, 9.363505073270296e-06, rtol=5e-4)

    # Poynting factor
    assert_close(Chemical('pentane', T=300, P=1E7).Poynting, 1.5743051250679803, atol=.02)

    c = Chemical('pentane', T=300, P=1E7)
    Poy = exp(quad(lambda P : c.VolumeLiquid(c.T, P), c.Psat, c.P)[0]/R/c.T)
    assert_close(Poy, 1.5821826990975127, atol=.02)


@pytest.mark.deprecated
def test_Chemical_properties_T_phase():
    # T-only dependent properties (always or at the moment)
    # Keep the order of the tests matching the order of the code
    w = Chemical('water', T=300, P=1E5)

    assert_close(w.Cp, 4179.537135856072, rtol=1e-4)
    assert_close(w.Cpm, 75.2955317728452, rtol=1e-4)

    assert_close(w.Vm, 1.8077520828345428e-05, rtol=1e-4)
    assert_close(w.rho, 996.5570041967351, rtol=1e-4)
    assert_close(w.rhom, 55317.319752828436, rtol=1e-4)
    assert_close(w.Z, 0.0007247422467681115, rtol=1e-4)

    assert_close(w.isobaric_expansion, 0.00027479530461365189, rtol=1E-3)
    assert_close(w.JT, -2.2029508371866032e-07, rtol=1E-3)

    assert_close(w.mu, 0.0008537426062537152, rtol=1e-4)
    assert_close(w.k, 0.6094991151038377, rtol=1e-4)

    assert_close(w.nu, 8.566921938819405e-07, rtol=1e-4)
    assert_close(w.alpha, 1.4633315800714463e-07, rtol=1e-4)

    assert_close(w.Pr, 5.854395582989558, rtol=1e-4)

@pytest.mark.deprecated
def test_H_Chemical():
    from thermo import chemical
    chemical.caching = False
    w = Chemical('water', T=298.15, P=101325.0)
    w.set_ref(T_ref=298.15, P_ref=101325, phase_ref='l', H_ref=0, S_ref=0)
    assert 0 == w.H
    assert 0 == w.S
    w.calculate(297.15, w.P)
    assert_close(w.Hm, 1000*(1.814832712-1.890164074), rtol=1E-3)
    w.calculate(274.15, w.P)
    assert_close(w.Hm, 1000*(0.07708322535-1.890164074), rtol=1E-4)
    w.calculate(273.15001, w.P)
    assert w.phase == 'l'
    H_pre_transition = w.Hm
    w.calculate(273.15, w.P)
    assert w.phase == 's'
    dH_transition = w.Hm - H_pre_transition
    assert_close(dH_transition, -6010.0, rtol=1E-5)
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
    assert_close(dH_transition, 6010.0, rtol=1E-5)
    assert w.phase == 'l'
    w.calculate(274.15)
    H_initial_liquid = w.Hm
    w.calculate(298.15, w.P)
    initial_liq_to_STP = w.Hm - H_initial_liquid
    assert_close(initial_liq_to_STP, -1000*(0.07708322535-1.890164074), rtol=1E-3)


    w = Chemical('water', T=373, P=101325.0)
    H_pre_vap = w.Hm
    w.calculate(w.Tb+1E-1)
    dH = w.Hm - H_pre_vap
    assert_close(dH, 40650, 1E-3) # google search answer


    Hm_as_vapor = w.Hm
    w.calculate(w.T+20)
    dH_20K_gas = w.Hm - Hm_as_vapor
    assert_close(dH_20K_gas, 1000*(48.9411675-48.2041134), rtol=1E-1) # Web tables, but hardly matches because of the excess





@pytest.mark.fuzz
@pytest.mark.slow
@pytest.mark.meta_Chemical
@pytest.mark.deprecated
def test_all_chemicals():
    for i in pubchem_db.CAS_index.values():
        c = Chemical(i.CASs)

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
        c.SGs
        c.SGl
        c.SGg
        c.SG
        c.Bvirial
        c.isobaric_expansion_l
        c.isobaric_expansion_g
        c.mul
        c.mug
        c.kl
        c.kg
        c.sigma
        c.permittivity
        c.absolute_permittivity
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
        c.R_specific

@pytest.mark.deprecated
def test_specific_chemical_failures():
    # D2O - failed on Hf, Gf
    Chemical('7789-20-0')

    # Tc failure
    Chemical('132259-10-0')

@pytest.mark.fuzz
@pytest.mark.slow
@pytest.mark.deprecated
def test_all_element_Chemicals():
    things = [periodic_table.CAS_to_elements, periodic_table.name_to_elements, periodic_table.symbol_to_elements]
    failed_CASs = []
    for thing in things:
        for i in thing.keys():
            try:
                Chemical(i)
            except:
                failed_CASs.append(periodic_table[i].name)
    assert 0 == len(set(failed_CASs))
