# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids.numerics import derivative

from thermo import Chemical, Mixture
from thermo.phases import *
from thermo.eos_mix import *
from thermo.eos import *
from thermo.vapor_pressure import VaporPressure, SublimationPressure
from thermo.volume import *
from thermo.heat_capacity import *
from thermo.phase_change import *
from thermo import ChemicalConstantsPackage, PropertyCorrelationPackage
from thermo.flash import FlashPureVLS

def test_two_eos_pure_flash_all_properties():
    # Methanol
    constants = ChemicalConstantsPackage(atom_fractions={'H': 0.6666666666666666, 'C': 0.16666666666666666, 'O': 0.16666666666666666}, atomss=[{'H': 4, 'C': 1, 'O': 1}],
                                         CASs=['67-56-1'], charges=[0], conductivities=[4.4e-05], dipoles=[1.7], 
                                         formulas=['CH4O'], Gfgs=[-162000.13000000003], Gfgs_mass=[-5055890.325967345], Hcs=[-764464.0], Hcs_lower=[-676489.1], Hcs_lower_mass=[-21112666.368306957], Hcs_mass=[-23858290.373904638], 
                                         Hfgs=[-200700.0], Hfgs_mass=[-6263681.321870828], Hfus_Tms=[3215.0000000000005], Hfus_Tms_mass=[100337.49601302797], Hvap_298s=[37486.47944178592], Hvap_298s_mass=[1169922.078237216],
                                         Hvap_Tbs=[35170.873821707064], Hvap_Tbs_mass=[1097653.9383702152], LFLs=[0.06], logPs=[-0.74], molecular_diameters=[3.7995699999999997],
                                         MWs=[32.04186], names=['methanol'], omegas=[0.5589999999999999], Parachors=[1.574169046057019e-05], Pcs=[8084000.0], phase_STPs=['l'], Psat_298s=[16905.960312551426], PSRK_groups=[{15: 1}], 
                                         Pts=[0.1758862695025245], PubChems=[887], rhocs=[8547.008547008547], rhocs_mass=[273.86205128205125], rhol_STPs=[24494.78614922483], rhol_STPs_mass=[784.8585085234012], RIs=[1.3288], S0gs=[239.9], 
                                         S0gs_mass=[7487.080962216301], Sfgs=[-129.7999999999999], Sfgs_mass=[-4050.950849919446], similarity_variables=[0.18725504699165404], Skins=[True], smiless=['CO'], STELs=[(250.0, 'ppm')], 
                                         StielPolars=[0.027243902847492674], Stockmayers=[685.96], Tautoignitions=[713.15], Tbs=[337.65], Tcs=[512.5], Tflashs=[282.15], Tms=[175.15], Tts=[175.59], UFLs=[0.36], 
                                         UNIFAC_Dortmund_groups=[{15: 1}], UNIFAC_groups=[{15: 1}], Van_der_Waals_areas=[358000.0], Van_der_Waals_volumes=[2.1709787e-05], Vcs=[0.000117], Vml_STPs=[4.0825014511573776e-05], 
                                         Vml_Tms=[3.541058756059562e-05], Zcs=[0.22196480200068586], UNIFAC_Rs=[1.4311], UNIFAC_Qs=[1.432], rhos_Tms=[1013.0221439813405], Vms_Tms=[3.162996997683616e-05], solubility_parameters=[29315.58469262365],
                                         Vml_60Fs=[4.033573571273147e-05], rhol_60Fs_mass=[794.3789652976724], rhol_60Fs=[24791.91174599953])
    
    VaporPressures = [VaporPressure(best_fit=(175.7, 512.49, [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10, 
                                                              -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708])), ]
    HeatCapacityGases = [HeatCapacityGas(best_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 
                                                                  4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924])), ]
    HeatCapacityLiquids = [HeatCapacityLiquid(best_fit=(180.0, 503.1, [-5.042130764341761e-17, 1.3174414379504284e-13, -1.472202211288266e-10, 9.19934288272021e-08,
                                                                       -3.517841445216993e-05, 0.008434516406617465, -1.2381765320848312, 101.71442569958393, -3508.6245143327947])),]
    VolumeLiquids = [VolumeLiquid(best_fit=(175.7, 502.5, [3.5725079384600736e-23, -9.031033742820083e-20, 9.819637959370411e-17, -5.993173551565636e-14, 
                                                           2.2442465416964825e-11, -5.27776114586072e-09, 7.610461006178106e-07, -6.148574498547711e-05, 0.00216398089328537])), ]
    EnthalpyVaporizations = [EnthalpyVaporization(best_fit=(175.7, 512.499, 512.5, [-0.004536133852590396, -0.2817551666837462, -7.344529282245696, -104.02286881045083,
                                                                                    -860.5796142607192, -4067.8897875259267, -8952.300062896637, 2827.0089241465225, 44568.12528999141])),]
    HeatCapacitySolids = [HeatCapacitySolid(best_fit=(1.0, 5000.0, [-4.2547351607351175e-26, 9.58204543572984e-22, -8.928062818728625e-18, 4.438942190507877e-14,
                                                                    -1.2656161406049876e-10, 2.0651464217978594e-07, -0.0001691371394823046, 0.2038633833421581, -0.07254973910767148])),]
    SublimationPressures = [SublimationPressure(best_fit=(61.5, 179.375, [-1.9972190661146383e-15, 2.1648606414769645e-12, -1.0255776193312338e-09, 2.7846062954442135e-07,
                                                                          -4.771529410705124e-05, 0.005347189071525987, -0.3916553642749777, 18.072103851054266, -447.1556383160345])),]
    EnthalpySublimations = [EnthalpySublimation(best_fit=(1.7515, 175.59, [2.3382707698778188e-17, -2.03890965442551e-12, 1.5374109464154768e-09, -4.640933157748743e-07,
                                                                           6.931187040484687e-05, -0.004954625422589015, 0.045058888152305354, 32.52432385785916, 42213.605713250145])),]
    VolumeSolids = [VolumeSolid(best_fit=(52.677, 175.59, [3.9379562779372194e-30, 1.4859309728437516e-27, 3.897856765862211e-24, 5.012758300685479e-21, 7.115820892078097e-18, 
                                                           9.987967202910477e-15, 1.4030825662633013e-11, 1.970935889948393e-08, 2.7686131179275174e-05])),]
    
    correlations = PropertyCorrelationPackage(constants, VaporPressures=VaporPressures, HeatCapacityGases=HeatCapacityGases, HeatCapacityLiquids=HeatCapacityLiquids, VolumeLiquids=VolumeLiquids,
                              EnthalpyVaporizations=EnthalpyVaporizations, HeatCapacitySolids=HeatCapacitySolids, SublimationPressures=SublimationPressures, 
                              EnthalpySublimations=EnthalpySublimations, VolumeSolids=VolumeSolids)
    
    eos_liquid = EOSLiquid(PRMIX, dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas), HeatCapacityGases=HeatCapacityGases,
                          Hfs=constants.Hfgs, Sfs=constants.Sfgs, Gfs=constants.Gfgs)
    
    gas = EOSGas(PRMIX, dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas), HeatCapacityGases=HeatCapacityGases,
                Hfs=constants.Hfgs, Sfs=constants.Sfgs, Gfs=constants.Gfgs)
    
    # Pseudo vapor volume for eos gas?
    flasher = FlashPureVLS(constants, correlations, gas, [eos_liquid], [])
    
    # Test all the results
    T, VF = 300.0, 0.4
    eq = flasher.flash(T=T, VF=VF)
    
    # Meta
    assert eq.phases[0] is eq.gas
    assert eq.phases[1] is eq.liquid0
    assert eq.phase_count == 2
    assert eq.liquid_count ==1 
    assert eq.gas_count == 1 
    assert eq.solid_count == 0
    assert eq.zs == [1.0]
    
    # Phase fractions
    assert_allclose(eq.betas, [.4, .6], rtol=1e-12)
    assert_allclose(eq.betas_mass, [.4, .6], rtol=1e-12)
    assert_allclose(eq.betas_volume, [0.9994907285990579, 0.0005092714009421547], rtol=1e-12)
    
    assert_allclose(eq.T, T, rtol=1e-12)
    for phase in eq.phases:
        assert_allclose(phase.T, T, rtol=1e-12)
    
    P_eq_expect = 17641.849717291494 # Might change slightly in the future due to tolerance
    assert_allclose(eq.P, P_eq_expect, rtol=1e-9)
    for phase in eq.phases:
        assert_allclose(phase.P, eq.P, rtol=1e-12)
        
    # Mass and volume (liquid) fractions
    # Since liquid fraction does not make any sense, might as well use pure volumes.
    assert_allclose(eq.Vfls(), [1])
    assert_allclose(eq.bulk.Vfls(), [1])
    assert_allclose([eq.Vfls(phase) for phase in eq.phases], [[1], [1]])
    
    assert_allclose(eq.Vfgs(), [1])
    assert_allclose(eq.bulk.Vfgs(), [1])
    assert_allclose([eq.Vfgs(phase) for phase in eq.phases], [[1], [1]])
    
    assert_allclose(eq.ws(), [1])
    assert_allclose(eq.bulk.ws(), [1])
    assert_allclose([eq.ws(phase) for phase in eq.phases], [[1], [1]])
    
    
    # H S G U A
    assert_allclose(eq.H(), -24104.760566193883, rtol=1e-12)
    assert_allclose(eq.bulk.H(), -24104.760566193883, rtol=1e-12)
    assert_allclose([i.H() for i in eq.phases], [51.18035762282534, -40208.72118207169], rtol=1e-12)
    
    assert_allclose(eq.S(), -65.77742662151137, rtol=1e-12)
    assert_allclose(eq.bulk.S(), -65.77742662151137, rtol=1e-12)
    assert_allclose([i.S() for i in eq.phases], [14.742376457877638, -119.45729534110404], rtol=1e-12)
    
    assert_allclose(eq.G(), -4371.532579740473, rtol=1e-12)
    assert_allclose(eq.bulk.G(), -4371.532579740473, rtol=1e-12)
    assert_allclose([i.G() for i in eq.phases], [-4371.532579740466, -4371.53257974048], rtol=1e-12)
    
    assert_allclose(eq.U(), -25098.583314964242, rtol=1e-12)
    assert_allclose(eq.bulk.U(), -25098.583314964242, rtol=1e-12)
    assert_allclose([i.U() for i in eq.phases], [-2432.11120054419, -40209.56472457761], rtol=1e-12)
    
    assert_allclose(eq.A(), -5365.355328510832, rtol=1e-12)
    assert_allclose(eq.bulk.A(), -5365.355328510832, rtol=1e-12)
    assert_allclose([i.A() for i in eq.phases], [-6854.824137907482, -4372.376122246402], rtol=1e-12)
    
    # Reactive H S G U A
    assert_allclose(eq.H_reactive(), -224804.7605661939, rtol=1e-12)
    assert_allclose(eq.bulk.H_reactive(), -224804.7605661939, rtol=1e-12)
    assert_allclose([i.H_reactive() for i in eq.phases], [-200648.81964237717, -240908.7211820717], rtol=1e-12)
    
    assert_allclose(eq.S_reactive(), -195.57742662151128, rtol=1e-12)
    assert_allclose(eq.bulk.S_reactive(), -195.57742662151128, rtol=1e-12)
    assert_allclose([i.S_reactive() for i in eq.phases], [-115.05762354212226, -249.25729534110394], rtol=1e-12)
    
    assert_allclose(eq.G_reactive(), -166131.5325797405, rtol=1e-12)
    assert_allclose(eq.bulk.G_reactive(), -166131.5325797405, rtol=1e-12)
    assert_allclose([i.G_reactive() for i in eq.phases], [-166131.5325797405]*2, rtol=1e-12)
    
    assert_allclose(eq.U_reactive(), -225798.58331496426, rtol=1e-12)
    assert_allclose(eq.bulk.U_reactive(), -225798.58331496426, rtol=1e-12)
    assert_allclose([i.U_reactive() for i in eq.phases], [-203132.1112005442, -240909.56472457762], rtol=1e-12)
    
    assert_allclose(eq.A_reactive(), -167125.35532851086, rtol=1e-12)
    assert_allclose(eq.bulk.A_reactive(), -167125.35532851086, rtol=1e-12)
    assert_allclose([i.A_reactive() for i in eq.phases], [-168614.82413790753, -166132.37612224644], rtol=1e-12)
    
    # Mass H S G U A
    assert_allclose(eq.H_mass(), -752289.6787575341, rtol=1e-12)
    assert_allclose(eq.bulk.H_mass(), -752289.6787575341, rtol=1e-12)
    assert_allclose([i.H_mass() for i in eq.phases], [1597.2967119519697, -1254880.9957371918], rtol=1e-12)
    
    assert_allclose(eq.S_mass(), -2052.859185500198, rtol=1e-12)
    assert_allclose(eq.bulk.S_mass(), -2052.859185500198, rtol=1e-12)
    assert_allclose([i.S_mass() for i in eq.phases], [460.0973993980886, -3728.163575432389], rtol=1e-12)
    
    assert_allclose(eq.G_mass(), -136431.9231074748, rtol=1e-12)
    assert_allclose(eq.bulk.G_mass(), -136431.9231074748, rtol=1e-12)
    assert_allclose([i.G_mass() for i in eq.phases], [-136431.92310747458]*2, rtol=1e-12)
    
    assert_allclose(eq.A_mass(), -167448.31069453622, rtol=1e-12)
    assert_allclose(eq.bulk.A_mass(), -167448.31069453622, rtol=1e-12)
    assert_allclose([i.A_mass() for i in eq.phases], [-213933.40267723164, -136458.24937273934], rtol=1e-12)
    
    assert_allclose(eq.U_mass(), -783306.0663445955, rtol=1e-12)
    assert_allclose(eq.bulk.U_mass(), -783306.0663445955, rtol=1e-12)
    assert_allclose([i.U_mass() for i in eq.phases], [-75904.18285780508, -1254907.322002456], rtol=1e-12)
    
    # Other
    assert_allclose(eq.MW(), 32.04186, rtol=1e-12)
    assert_allclose(eq.bulk.MW(), 32.04186, rtol=1e-12)
    assert_allclose([i.MW() for i in eq.phases], [32.04186, 32.04186], rtol=1e-12)
    
    # Volumetric
    assert_allclose(eq.rho_mass(), 0.568791245201322, rtol=1e-12)
    assert_allclose(eq.bulk.rho_mass(), 0.568791245201322, rtol=1e-12)
    assert_allclose([i.rho_mass() for i in eq.phases], [0.2276324247643884, 670.1235264525617], rtol=1e-12)
    
    assert_allclose(eq.V(), 0.05633325103071669, rtol=1e-12)
    assert_allclose(eq.bulk.V(), 0.05633325103071669, rtol=1e-12)
    assert_allclose([i.V() for i in eq.phases], [0.1407614052926116, 4.781485612006529e-05], rtol=1e-12)
    
    assert_allclose(eq.rho(), 17.75150522476916, rtol=1e-12)
    assert_allclose(eq.bulk.rho(), 17.75150522476916, rtol=1e-12)
    assert_allclose([i.rho() for i in eq.phases], [7.104220066013285, 20914.002072681225], rtol=1e-12)
    
    assert_allclose(eq.Z(), 0.3984313416321555, rtol=1e-12)
    assert_allclose(eq.bulk.Z(), 0.3984313416321555, rtol=1e-12)
    assert_allclose([i.Z() for i in eq.phases],  [0.9955710798615588, 0.00033818281255378383], rtol=1e-12)
    
    assert_allclose(eq.V_mass(), 1.7581142614915828, rtol=1e-12)
    assert_allclose(eq.bulk.V_mass(), 1.7581142614915828, rtol=1e-12)
    assert_allclose([i.V_mass() for i in eq.phases], [4.393047260446541, 0.0014922621882770006], rtol=1e-12)
    
    # MW air may be adjusted in the future
    SG_gas_expect = 1.10647130731458
    assert_allclose(eq.SG_gas(), SG_gas_expect, rtol=1e-5)
    assert_allclose(eq.bulk.SG_gas(), SG_gas_expect, rtol=1e-5)
    assert_allclose([i.SG_gas() for i in eq.phases], [SG_gas_expect]*2, rtol=1e-5)
    
    assert_allclose(eq.SG(), [0.7951605425835767], rtol=1e-5)
    assert_allclose(eq.bulk.SG(), [0.7951605425835767], rtol=1e-5)
    assert_allclose([i.SG() for i in eq.phases], [0.7951605425835767]*2, rtol=1e-5)
    
    # Cp and Cv related
    assert_allclose(eq.Cp(), 85.30634675113457, rtol=1e-12)
    assert_allclose(eq.bulk.Cp(), 85.30634675113457, rtol=1e-12)
    assert_allclose([i.Cp() for i in eq.phases],  [44.63182543018587, 112.42269429843371], rtol=1e-12)
    
    assert_allclose(eq.Cp_mass(), 2662.340661595006, rtol=1e-12)
    assert_allclose(eq.bulk.Cp_mass(), 2662.340661595006, rtol=1e-12)
    assert_allclose([i.Cp_mass() for i in eq.phases], [1392.9224280421258, 3508.6194839635937], rtol=1e-12)
    
    assert_allclose(eq.Cv(), 65.77441970078021, rtol=1e-12)
    assert_allclose(eq.bulk.Cv(), 65.77441970078021, rtol=1e-12)
    assert_allclose([i.Cv() for i in eq.phases], [36.18313355107219, 79.87072617335762], rtol=1e-12)
    
    assert_allclose(eq.Cp_Cv_ratio(), 1.2969532401685737, rtol=1e-12)
    assert_allclose(eq.bulk.Cp_Cv_ratio(), 1.2969532401685737, rtol=1e-12)
    assert_allclose([i.Cp_Cv_ratio() for i in eq.phases], [1.2334980707845113, 1.4075581841389895], rtol=1e-12)
    
    assert_allclose(eq.Cv_mass(), 2052.765341986396, rtol=1e-12)
    assert_allclose(eq.bulk.Cv_mass(), 2052.765341986396, rtol=1e-12)
    assert_allclose([i.Cv_mass() for i in eq.phases], [1129.2457289018862, 2492.699430474936], rtol=1e-12)
    
    # ideal gas properties
    assert_allclose(eq.V_ideal_gas(), 0.14138759968016104, rtol=1e-12)
    assert_allclose(eq.bulk.V_ideal_gas(), 0.14138759968016104, rtol=1e-12)
    assert_allclose([i.V_ideal_gas() for i in eq.phases], [0.14138759968016104]*2, rtol=1e-12)
    
    assert_allclose(eq.Cp_ideal_gas(), 44.47452555993428, rtol=1e-12)
    assert_allclose(eq.bulk.Cp_ideal_gas(), 44.47452555993428, rtol=1e-12)
    assert_allclose([i.Cp_ideal_gas() for i in eq.phases], [44.47452555993428]*2, rtol=1e-12)
    
    assert_allclose(eq.Cv_ideal_gas(), 36.160062941781035, rtol=1e-12)
    assert_allclose(eq.bulk.Cv_ideal_gas(), 36.160062941781035, rtol=1e-12)
    assert_allclose([i.Cv_ideal_gas() for i in eq.phases], [36.160062941781035]*2, rtol=1e-12)
    
    assert_allclose(eq.Cp_Cv_ratio_ideal_gas(), 1.229934959779794, rtol=1e-12)
    assert_allclose(eq.bulk.Cp_Cv_ratio_ideal_gas(), 1.229934959779794, rtol=1e-12)
    assert_allclose([i.Cp_Cv_ratio_ideal_gas() for i in eq.phases], [1.229934959779794]*2, rtol=1e-12)
    
    assert_allclose(eq.H_ideal_gas(), 82.17715909331491, rtol=1e-12)
    assert_allclose(eq.bulk.H_ideal_gas(), 82.17715909331491, rtol=1e-12)
    assert_allclose([i.H_ideal_gas() for i in eq.phases], [82.17715909331491]*2, rtol=1e-12)
    
    assert_allclose(eq.S_ideal_gas(), 14.808945043469695, rtol=1e-12)
    assert_allclose(eq.bulk.S_ideal_gas(), 14.808945043469695, rtol=1e-12)
    assert_allclose([i.S_ideal_gas() for i in eq.phases], [14.808945043469695]*2, rtol=1e-12)
    
    assert_allclose(eq.G_ideal_gas(), -4360.506353947593, rtol=1e-12)
    assert_allclose(eq.bulk.G_ideal_gas(), -4360.506353947593, rtol=1e-12)
    assert_allclose([i.G_ideal_gas() for i in eq.phases], [-4360.506353947593]*2, rtol=1e-12)
    
    assert_allclose(eq.A_ideal_gas(), -6854.845139393565, rtol=1e-12)
    assert_allclose(eq.bulk.A_ideal_gas(), -6854.845139393565, rtol=1e-12)
    assert_allclose([i.A_ideal_gas() for i in eq.phases], [-6854.845139393565]*2, rtol=1e-12)
    
    assert_allclose(eq.U_ideal_gas(), -2412.161626352657, rtol=1e-12)
    assert_allclose(eq.bulk.U_ideal_gas(), -2412.161626352657, rtol=1e-12)
    assert_allclose([i.U_ideal_gas() for i in eq.phases], [-2412.161626352657]*2, rtol=1e-12)
    
    # Ideal formation basis
    
    assert_allclose(eq.H_formation_ideal_gas(), -200700.0, rtol=1e-12)
    assert_allclose(eq.bulk.H_formation_ideal_gas(), -200700.0, rtol=1e-12)
    assert_allclose([i.H_formation_ideal_gas() for i in eq.phases], [-200700.0]*2, rtol=1e-12)
    
    assert_allclose(eq.S_formation_ideal_gas(), -129.8, rtol=1e-12)
    assert_allclose(eq.bulk.S_formation_ideal_gas(), -129.8, rtol=1e-12)
    assert_allclose([i.S_formation_ideal_gas() for i in eq.phases], [-129.8]*2, rtol=1e-12)
    
    assert_allclose(eq.G_formation_ideal_gas(), -162000.13, rtol=1e-12)
    assert_allclose(eq.bulk.G_formation_ideal_gas(), -162000.13, rtol=1e-12)
    assert_allclose([i.G_formation_ideal_gas() for i in eq.phases], [-162000.13]*2, rtol=1e-12)
    
    assert_allclose(eq.U_formation_ideal_gas(), -215026.0985375923, rtol=1e-12)
    assert_allclose(eq.bulk.U_formation_ideal_gas(), -215026.0985375923, rtol=1e-12)
    assert_allclose([i.U_formation_ideal_gas() for i in eq.phases], [-215026.0985375923]*2, rtol=1e-12)
    
    assert_allclose(eq.A_formation_ideal_gas(), -176326.22853759234, rtol=1e-12)
    assert_allclose(eq.bulk.A_formation_ideal_gas(), -176326.22853759234, rtol=1e-12)
    assert_allclose([i.A_formation_ideal_gas() for i in eq.phases], [-176326.22853759234]*2, rtol=1e-12)
    
    # Pseudo critical properties
    assert_allclose(eq.pseudo_Tc(), constants.Tcs[0], rtol=1e-12)
    assert_allclose(eq.bulk.pseudo_Tc(), constants.Tcs[0], rtol=1e-12)
    assert_allclose([i.pseudo_Tc() for i in eq.phases], [constants.Tcs[0]]*2, rtol=1e-12)
    
    assert_allclose(eq.pseudo_Pc(), constants.Pcs[0], rtol=1e-12)
    assert_allclose(eq.bulk.pseudo_Pc(), constants.Pcs[0], rtol=1e-12)
    assert_allclose([i.pseudo_Pc() for i in eq.phases], [constants.Pcs[0]]*2, rtol=1e-12)
    
    assert_allclose(eq.pseudo_Vc(), constants.Vcs[0], rtol=1e-12)
    assert_allclose(eq.bulk.pseudo_Vc(), constants.Vcs[0], rtol=1e-12)
    assert_allclose([i.pseudo_Vc() for i in eq.phases], [constants.Vcs[0]]*2, rtol=1e-12)
    
    assert_allclose(eq.pseudo_Zc(), constants.Zcs[0], rtol=1e-12)
    assert_allclose(eq.bulk.pseudo_Zc(), constants.Zcs[0], rtol=1e-12)
    assert_allclose([i.pseudo_Zc() for i in eq.phases], [constants.Zcs[0]]*2, rtol=1e-12)
    
    # Standard volumes
    V_std_expect = 0.023690417461829063
    assert_allclose(eq.V_gas_standard(), V_std_expect, rtol=1e-12)
    assert_allclose(eq.bulk.V_gas_standard(), V_std_expect, rtol=1e-12)
    assert_allclose([i.V_gas_standard() for i in eq.phases], [V_std_expect]*2, rtol=1e-12)
    
    V_std_expect = 0.02364483003622853
    assert_allclose(eq.V_gas_normal(), V_std_expect, rtol=1e-12)
    assert_allclose(eq.bulk.V_gas_normal(), V_std_expect, rtol=1e-12)
    assert_allclose([i.V_gas_normal() for i in eq.phases], [V_std_expect]*2, rtol=1e-12)
    
    # Combustion properties
    Hc_expect = -764464.0
    assert_allclose(eq.Hc(), Hc_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Hc(), Hc_expect, rtol=1e-12)
    assert_allclose([i.Hc() for i in eq.phases], [Hc_expect]*2, rtol=1e-12)
    
    Hc_mass_expect = -23858290.373904638
    assert_allclose(eq.Hc_mass(), Hc_mass_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Hc_mass(), Hc_mass_expect, rtol=1e-12)
    assert_allclose([i.Hc_mass() for i in eq.phases], [Hc_mass_expect]*2, rtol=1e-12)
    
    Hc_lower_expect = -676489.1
    assert_allclose(eq.Hc_lower(), Hc_lower_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Hc_lower(), Hc_lower_expect, rtol=1e-12)
    assert_allclose([i.Hc_lower() for i in eq.phases], [Hc_lower_expect]*2, rtol=1e-12)
    
    Hc_lower_mass_expect =  -21112666.368306957
    assert_allclose(eq.Hc_lower_mass(), Hc_lower_mass_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Hc_lower_mass(), Hc_lower_mass_expect, rtol=1e-12)
    assert_allclose([i.Hc_lower_mass() for i in eq.phases], [Hc_lower_mass_expect]*2, rtol=1e-12)
    
    # Volume combustion properties
    Hc_normal_expect = -32331126.881804217
    assert_allclose(eq.Hc_normal(), Hc_normal_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Hc_normal(), Hc_normal_expect, rtol=1e-12)
    assert_allclose([i.Hc_normal() for i in eq.phases], [Hc_normal_expect]*2, rtol=1e-12)
    
    Hc_standard_expect = -32268912.155378208
    assert_allclose(eq.Hc_standard(), Hc_standard_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Hc_standard(), Hc_standard_expect, rtol=1e-12)
    assert_allclose([i.Hc_standard() for i in eq.phases], [Hc_standard_expect]*2, rtol=1e-12)
    
    Hc_lower_normal_expect = -28610444.607277177
    assert_allclose(eq.Hc_lower_normal(), Hc_lower_normal_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Hc_lower_normal(), Hc_lower_normal_expect, rtol=1e-12)
    assert_allclose([i.Hc_lower_normal() for i in eq.phases], [Hc_lower_normal_expect]*2, rtol=1e-12)
    
    Hc_lower_standard_expect = -28555389.582728375
    assert_allclose(eq.Hc_lower_standard(), Hc_lower_standard_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Hc_lower_standard(), Hc_lower_standard_expect, rtol=1e-12)
    assert_allclose([i.Hc_lower_standard() for i in eq.phases], [Hc_lower_standard_expect]*2, rtol=1e-12)
    
    # Wobbe index
    Wobbe_index_expect = 726753.2127139702
    assert_allclose(eq.Wobbe_index(), Wobbe_index_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Wobbe_index(), Wobbe_index_expect, rtol=1e-12)
    assert_allclose([i.Wobbe_index() for i in eq.phases], [Wobbe_index_expect]*2, rtol=1e-12)
    
    Wobbe_index_lower_expect = 643118.0890022058
    assert_allclose(eq.Wobbe_index_lower(), Wobbe_index_lower_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Wobbe_index_lower(), Wobbe_index_lower_expect, rtol=1e-12)
    assert_allclose([i.Wobbe_index_lower() for i in eq.phases], [Wobbe_index_lower_expect]*2, rtol=1e-12)
    
    Wobbe_index_mass_expect = 22681367.83301501
    assert_allclose(eq.Wobbe_index_mass(), Wobbe_index_mass_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Wobbe_index_mass(), Wobbe_index_mass_expect, rtol=1e-12)
    assert_allclose([i.Wobbe_index_mass() for i in eq.phases], [Wobbe_index_mass_expect]*2, rtol=1e-12)
    
    Wobbe_index_lower_mass_expect = 20071184.6628818
    assert_allclose(eq.Wobbe_index_lower_mass(), Wobbe_index_lower_mass_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Wobbe_index_lower_mass(), Wobbe_index_lower_mass_expect, rtol=1e-12)
    assert_allclose([i.Wobbe_index_lower_mass() for i in eq.phases], [Wobbe_index_lower_mass_expect]*2, rtol=1e-12)
    
    # Wobbe index volume properties
    Wobbe_index_standard_expect = 30677096.082622595
    assert_allclose(eq.Wobbe_index_standard(), Wobbe_index_standard_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Wobbe_index_standard(), Wobbe_index_standard_expect, rtol=1e-12)
    assert_allclose([i.Wobbe_index_standard() for i in eq.phases], [Wobbe_index_standard_expect]*2, rtol=1e-12)
    
    Wobbe_index_normal_expect = 30736241.774647623
    assert_allclose(eq.Wobbe_index_normal(), Wobbe_index_normal_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Wobbe_index_normal(), Wobbe_index_normal_expect, rtol=1e-12)
    assert_allclose([i.Wobbe_index_normal() for i in eq.phases], [Wobbe_index_normal_expect]*2, rtol=1e-12)
    
    Wobbe_index_lower_standard_expect = 27146760.50088282
    assert_allclose(eq.Wobbe_index_lower_standard(), Wobbe_index_lower_standard_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Wobbe_index_lower_standard(), Wobbe_index_lower_standard_expect, rtol=1e-12)
    assert_allclose([i.Wobbe_index_lower_standard() for i in eq.phases], [Wobbe_index_lower_standard_expect]*2, rtol=1e-12)
    
    Wobbe_index_lower_normal_expect = 27199099.677046627
    assert_allclose(eq.Wobbe_index_lower_normal(), Wobbe_index_lower_normal_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Wobbe_index_lower_normal(), Wobbe_index_lower_normal_expect, rtol=1e-12)
    assert_allclose([i.Wobbe_index_lower_normal() for i in eq.phases], [Wobbe_index_lower_normal_expect]*2, rtol=1e-12)
    
    # Mechanical critical point - these have an inner solver
    Tmc_expect = 512.5
    assert_allclose(eq.Tmc(), Tmc_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Tmc(), Tmc_expect, rtol=1e-12)
    assert_allclose([i.Tmc() for i in eq.phases], [Tmc_expect]*2, rtol=1e-12)
    
    Pmc_expect = 8084000.0
    assert_allclose(eq.Pmc(), Pmc_expect, rtol=1e-12)
    assert_allclose(eq.bulk.Pmc(), Pmc_expect, rtol=1e-12)
    assert_allclose([i.Pmc() for i in eq.phases], [Pmc_expect]*2, rtol=1e-12)
    
    Vmc_expect = 0.00016203642168563802
    assert_allclose(eq.Vmc(), Vmc_expect, rtol=1e-5)
    assert_allclose(eq.bulk.Vmc(), Vmc_expect, rtol=1e-5)
    assert_allclose([i.Vmc() for i in eq.phases], [Vmc_expect]*2, rtol=1e-5)
    
    Zmc_expect = 0.30740497655001947
    assert_allclose(eq.Zmc(), Zmc_expect, rtol=1e-5)
    assert_allclose(eq.bulk.Zmc(), Zmc_expect, rtol=1e-5)
    assert_allclose([i.Zmc() for i in eq.phases], [Zmc_expect]*2, rtol=1e-5)
    
    # Properties calculated form derivatives
    assert_allclose(eq.beta(), 0.002008218029645217, rtol=1e-12)
    assert_allclose(eq.bulk.beta(), 0.002008218029645217, rtol=1e-12)
    assert_allclose([i.beta() for i in eq.phases], [0.0033751089225799308, 0.0010969574343554077], rtol=1e-12)
    
    assert_allclose(eq.kappa(), 2.277492845010776e-05, rtol=1e-12)
    assert_allclose(eq.bulk.kappa(), 2.277492845010776e-05, rtol=1e-12)
    assert_allclose([i.kappa() for i in eq.phases], [5.693652573977374e-05, 5.302569971013028e-10], rtol=1e-12)
    
    assert_allclose(eq.Joule_Thomson(), 1.563918814309498e-05, rtol=1e-12)
    assert_allclose(eq.bulk.Joule_Thomson(), 1.563918814309498e-05, rtol=1e-12)
    assert_allclose([i.Joule_Thomson() for i in eq.phases], [3.9525992445522226e-05, -2.853480585231828e-07], rtol=1e-12)
    
    assert_allclose(eq.speed_of_sound(), 235.8471087474984, rtol=1e-12)
    assert_allclose(eq.bulk.speed_of_sound(), 235.8471087474984, rtol=1e-12)
    assert_allclose([i.speed_of_sound() for i in eq.phases], [55.22243501081154, 356.26355790528964], rtol=1e-12)
    
    assert_allclose(eq.speed_of_sound_mass(), 1317.5639311230432, rtol=1e-12)
    assert_allclose(eq.bulk.speed_of_sound_mass(), 1317.5639311230432, rtol=1e-12)
    assert_allclose([i.speed_of_sound_mass() for i in eq.phases], [308.5010833731638, 1990.2724962896298], rtol=1e-12)
    
    # Departure properties
    assert_allclose(eq.H_dep(), -24186.9377252872, rtol=1e-12)
    assert_allclose(eq.bulk.H_dep(), -24186.9377252872, rtol=1e-12)
    assert_allclose([i.H_dep() for i in eq.phases], [-30.99680147049139, -40290.898341165004], rtol=1e-12)
    
    assert_allclose(eq.S_dep(), -80.58637166498106, rtol=1e-12)
    assert_allclose(eq.bulk.S_dep(), -80.58637166498106, rtol=1e-12)
    assert_allclose([i.S_dep() for i in eq.phases], [-0.0665685855920503, -134.26624038457373], rtol=1e-12)
    
    assert_allclose(eq.G_dep(), -11.026225792880723, rtol=1e-9)
    assert_allclose(eq.bulk.G_dep(), -11.026225792880723, rtol=1e-9)
    assert_allclose([i.G_dep() for i in eq.phases], [-11.026225792876303, -11.026225792884361], rtol=1e-9)
    
    assert_allclose(eq.U_dep(), -22686.421688611586, rtol=1e-9)
    assert_allclose(eq.bulk.U_dep(), -22686.421688611586, rtol=1e-9)
    assert_allclose([i.U_dep() for i in eq.phases], [-19.949574191534843, -37797.40309822495], rtol=1e-9)
    
    assert_allclose(eq.A_dep(), 1489.4898108827329, rtol=1e-9)
    assert_allclose(eq.bulk.A_dep(), 1489.4898108827329, rtol=1e-9)
    assert_allclose([i.A_dep() for i in eq.phases], [0.021001486080244547, 2482.4690171471666], rtol=1e-9)
    
    assert_allclose(eq.Cp_dep(), 40.83182119120029, rtol=1e-9)
    assert_allclose(eq.bulk.Cp_dep(), 40.83182119120029, rtol=1e-9)
    assert_allclose([i.Cp_dep() for i in eq.phases], [0.1572998702515953, 67.94816873849943], rtol=1e-9)
    
    assert_allclose(eq.Cv_dep(), 29.61435675899918, rtol=1e-12)
    assert_allclose(eq.bulk.Cv_dep(), 29.61435675899918, rtol=1e-12)
    assert_allclose([i.Cv_dep() for i in eq.phases], [0.023070609291153232, 43.71066323157659], rtol=1e-12)
    
    # Standard liquid density
    rho_liquid_ref_expect = 784.8585085234012
    assert_allclose(eq.rho_liquid_ref(), rho_liquid_ref_expect, rtol=1e-12)
    assert_allclose(eq.bulk.rho_liquid_ref(), rho_liquid_ref_expect, rtol=1e-12)
    assert_allclose([i.rho_liquid_ref() for i in eq.phases], [rho_liquid_ref_expect]*2, rtol=1e-12)
    
    V_liquid_ref_expect = 4.0825014511573776e-05
    assert_allclose(eq.V_liquid_ref(), V_liquid_ref_expect, rtol=1e-12)
    assert_allclose(eq.bulk.V_liquid_ref(), V_liquid_ref_expect, rtol=1e-12)
    assert_allclose([i.V_liquid_ref() for i in eq.phases], [V_liquid_ref_expect]*2, rtol=1e-12)
    
    # Water contect
    assert_allclose(eq.molar_water_content(), 0, atol=0)
    assert_allclose(eq.bulk.molar_water_content(), 0, atol=0)
    assert_allclose([i.molar_water_content() for i in eq.phases], [0]*2, atol=0)
    
    assert_allclose(eq.zs_no_water(), [1], atol=0)
    assert_allclose(eq.bulk.zs_no_water(), [1], atol=0)
    assert_allclose([i.zs_no_water() for i in eq.phases], [[1.0]]*2, atol=0)
    
    assert_allclose(eq.ws_no_water(), [1], atol=0)
    assert_allclose(eq.bulk.ws_no_water(), [1], atol=0)
    assert_allclose([i.ws_no_water() for i in eq.phases], [[1.0]]*2, atol=0)
    
    # H/C ratio
    assert_allclose(eq.H_C_ratio(), 4, atol=0)
    assert_allclose(eq.bulk.H_C_ratio(), 4, atol=0)
    assert_allclose([i.H_C_ratio() for i in eq.phases], [4]*2, atol=0)
    
    assert_allclose(eq.H_C_ratio_mass(), 0.3356806847227889, rtol=1e-12)
    assert_allclose(eq.bulk.H_C_ratio_mass(), 0.3356806847227889, rtol=1e-12)
    assert_allclose([i.H_C_ratio_mass() for i in eq.phases], [0.3356806847227889]*2, rtol=1e-12)
