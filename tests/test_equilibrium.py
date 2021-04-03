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

import pytest
from fluids.numerics import derivative, assert_close, assert_close1d, assert_close2d

from thermo import Chemical, Mixture
from thermo.phases import *
from thermo.eos_mix import *
from chemicals.utils import *
from thermo.eos import *
from thermo.vapor_pressure import VaporPressure, SublimationPressure
from thermo.volume import *
from thermo.heat_capacity import *
from thermo.phase_change import *
from thermo import ChemicalConstantsPackage, PropertyCorrelationsPackage
from thermo.flash import FlashPureVLS, FlashVLN, FlashVL
from thermo.bulk import *
from thermo.equilibrium import EquilibriumState


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

    VaporPressures = [VaporPressure(poly_fit=(175.7, 512.49, [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10,
                                                              -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708])), ]
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12,
                                                                  4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924])), ]
    HeatCapacityLiquids = [HeatCapacityLiquid(poly_fit=(180.0, 503.1, [-5.042130764341761e-17, 1.3174414379504284e-13, -1.472202211288266e-10, 9.19934288272021e-08,
                                                                       -3.517841445216993e-05, 0.008434516406617465, -1.2381765320848312, 101.71442569958393, -3508.6245143327947])),]
    VolumeLiquids = [VolumeLiquid(poly_fit=(175.7, 502.5, [3.5725079384600736e-23, -9.031033742820083e-20, 9.819637959370411e-17, -5.993173551565636e-14,
                                                           2.2442465416964825e-11, -5.27776114586072e-09, 7.610461006178106e-07, -6.148574498547711e-05, 0.00216398089328537])), ]
    EnthalpyVaporizations = [EnthalpyVaporization(poly_fit=(175.7, 512.499, 512.5, [-0.004536133852590396, -0.2817551666837462, -7.344529282245696, -104.02286881045083,
                                                                                    -860.5796142607192, -4067.8897875259267, -8952.300062896637, 2827.0089241465225, 44568.12528999141])),]
    HeatCapacitySolids = [HeatCapacitySolid(poly_fit=(1.0, 5000.0, [-4.2547351607351175e-26, 9.58204543572984e-22, -8.928062818728625e-18, 4.438942190507877e-14,
                                                                    -1.2656161406049876e-10, 2.0651464217978594e-07, -0.0001691371394823046, 0.2038633833421581, -0.07254973910767148])),]
    SublimationPressures = [SublimationPressure(poly_fit=(61.5, 179.375, [-1.9972190661146383e-15, 2.1648606414769645e-12, -1.0255776193312338e-09, 2.7846062954442135e-07,
                                                                          -4.771529410705124e-05, 0.005347189071525987, -0.3916553642749777, 18.072103851054266, -447.1556383160345])),]
    EnthalpySublimations = [EnthalpySublimation(poly_fit=(1.7515, 175.59, [2.3382707698778188e-17, -2.03890965442551e-12, 1.5374109464154768e-09, -4.640933157748743e-07,
                                                                           6.931187040484687e-05, -0.004954625422589015, 0.045058888152305354, 32.52432385785916, 42213.605713250145])),]
    VolumeSolids = [VolumeSolid(poly_fit=(52.677, 175.59, [3.9379562779372194e-30, 1.4859309728437516e-27, 3.897856765862211e-24, 5.012758300685479e-21, 7.115820892078097e-18,
                                                           9.987967202910477e-15, 1.4030825662633013e-11, 1.970935889948393e-08, 2.7686131179275174e-05])),]

    correlations = PropertyCorrelationsPackage(constants, VaporPressures=VaporPressures, HeatCapacityGases=HeatCapacityGases, HeatCapacityLiquids=HeatCapacityLiquids, VolumeLiquids=VolumeLiquids,
                                               EnthalpyVaporizations=EnthalpyVaporizations, HeatCapacitySolids=HeatCapacitySolids, SublimationPressures=SublimationPressures,
                                               EnthalpySublimations=EnthalpySublimations, VolumeSolids=VolumeSolids)

    eos_liquid = CEOSLiquid(PRMIX, dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas), HeatCapacityGases=HeatCapacityGases,
                            Hfs=constants.Hfgs, Sfs=constants.Sfgs, Gfs=constants.Gfgs)

    gas = CEOSGas(PRMIX, dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas), HeatCapacityGases=HeatCapacityGases,
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
    assert_close1d(eq.betas, [.4, .6], rtol=1e-12)
    assert_close1d(eq.betas_mass, [.4, .6], rtol=1e-12)
    assert_close1d(eq.betas_volume, [0.9994907285990579, 0.0005092714009421547], rtol=1e-12)

    assert_close(eq.T, T, rtol=1e-12)
    for phase in eq.phases:
        assert_close(phase.T, T, rtol=1e-12)

    P_eq_expect = 17641.849717291494 # Might change slightly in the future due to tolerance
    assert_close(eq.P, P_eq_expect, rtol=1e-9)
    for phase in eq.phases:
        assert_close(phase.P, eq.P, rtol=1e-12)

    # Mass and volume (liquid) fractions
    # Since liquid fraction does not make any sense, might as well use pure volumes.
    assert_close1d(eq.Vfls(), [1])
    assert_close1d(eq.bulk.Vfls(), [1])
    assert_close2d([eq.Vfls(phase) for phase in eq.phases], [[1], [1]])

    assert_close1d(eq.Vfgs(), [1])
    assert_close1d(eq.bulk.Vfgs(), [1])
    assert_close2d([eq.Vfgs(phase) for phase in eq.phases], [[1], [1]])

    assert_close1d(eq.ws(), [1])
    assert_close1d(eq.bulk.ws(), [1])
    assert_close2d([eq.ws(phase) for phase in eq.phases], [[1], [1]])


    # H S G U A
    assert_close(eq.H(), -24104.760566193883, rtol=1e-12)
    assert_close(eq.bulk.H(), -24104.760566193883, rtol=1e-12)
    assert_close1d([i.H() for i in eq.phases], [51.18035762282534, -40208.72118207169], rtol=1e-12)

    assert_close(eq.S(), -65.77742662151137, rtol=1e-12)
    assert_close(eq.bulk.S(), -65.77742662151137, rtol=1e-12)
    assert_close1d([i.S() for i in eq.phases], [14.742376457877638, -119.45729534110404], rtol=1e-12)

    assert_close(eq.G(), -4371.532579740473, rtol=1e-12)
    assert_close(eq.bulk.G(), -4371.532579740473, rtol=1e-12)
    assert_close1d([i.G() for i in eq.phases], [-4371.532579740466, -4371.53257974048], rtol=1e-12)

    assert_close(eq.U(), -25098.583314964242, rtol=1e-12)
    assert_close(eq.bulk.U(), -25098.583314964242, rtol=1e-12)
    assert_close1d([i.U() for i in eq.phases], [-2432.11120054419, -40209.56472457761], rtol=1e-12)

    assert_close(eq.A(), -5365.355328510832, rtol=1e-12)
    assert_close(eq.bulk.A(), -5365.355328510832, rtol=1e-12)
    assert_close1d([i.A() for i in eq.phases], [-6854.824137907482, -4372.376122246402], rtol=1e-12)

    # Reactive H S G U A
    assert_close(eq.H_reactive(), -224804.7605661939, rtol=1e-12)
    assert_close(eq.bulk.H_reactive(), -224804.7605661939, rtol=1e-12)
    assert_close1d([i.H_reactive() for i in eq.phases], [-200648.81964237717, -240908.7211820717], rtol=1e-12)

    assert_close(eq.S_reactive(), -195.57742662151128, rtol=1e-12)
    assert_close(eq.bulk.S_reactive(), -195.57742662151128, rtol=1e-12)
    assert_close1d([i.S_reactive() for i in eq.phases], [-115.05762354212226, -249.25729534110394], rtol=1e-12)

    assert_close(eq.G_reactive(), -166131.5325797405, rtol=1e-12)
    assert_close(eq.bulk.G_reactive(), -166131.5325797405, rtol=1e-12)
    assert_close1d([i.G_reactive() for i in eq.phases], [-166131.5325797405]*2, rtol=1e-12)

    assert_close(eq.U_reactive(), -225798.58331496426, rtol=1e-12)
    assert_close(eq.bulk.U_reactive(), -225798.58331496426, rtol=1e-12)
    assert_close1d([i.U_reactive() for i in eq.phases], [-203132.1112005442, -240909.56472457762], rtol=1e-12)

    assert_close(eq.A_reactive(), -167125.35532851086, rtol=1e-12)
    assert_close(eq.bulk.A_reactive(), -167125.35532851086, rtol=1e-12)
    assert_close1d([i.A_reactive() for i in eq.phases], [-168614.82413790753, -166132.37612224644], rtol=1e-12)

    # Mass H S G U A
    assert_close(eq.H_mass(), -752289.6787575341, rtol=1e-12)
    assert_close(eq.bulk.H_mass(), -752289.6787575341, rtol=1e-12)
    assert_close1d([i.H_mass() for i in eq.phases], [1597.2967119519697, -1254880.9957371918], rtol=1e-12)

    assert_close(eq.S_mass(), -2052.859185500198, rtol=1e-12)
    assert_close(eq.bulk.S_mass(), -2052.859185500198, rtol=1e-12)
    assert_close1d([i.S_mass() for i in eq.phases], [460.0973993980886, -3728.163575432389], rtol=1e-12)

    assert_close(eq.G_mass(), -136431.9231074748, rtol=1e-12)
    assert_close(eq.bulk.G_mass(), -136431.9231074748, rtol=1e-12)
    assert_close1d([i.G_mass() for i in eq.phases], [-136431.92310747458]*2, rtol=1e-12)

    assert_close(eq.A_mass(), -167448.31069453622, rtol=1e-12)
    assert_close(eq.bulk.A_mass(), -167448.31069453622, rtol=1e-12)
    assert_close1d([i.A_mass() for i in eq.phases], [-213933.40267723164, -136458.24937273934], rtol=1e-12)

    assert_close(eq.U_mass(), -783306.0663445955, rtol=1e-12)
    assert_close(eq.bulk.U_mass(), -783306.0663445955, rtol=1e-12)
    assert_close1d([i.U_mass() for i in eq.phases], [-75904.18285780508, -1254907.322002456], rtol=1e-12)

    # Other
    assert_close(eq.MW(), 32.04186, rtol=1e-12)
    assert_close(eq.bulk.MW(), 32.04186, rtol=1e-12)
    assert_close1d([i.MW() for i in eq.phases], [32.04186, 32.04186], rtol=1e-12)

    # Volumetric
    assert_close(eq.rho_mass(), 0.568791245201322, rtol=1e-12)
    assert_close(eq.bulk.rho_mass(), 0.568791245201322, rtol=1e-12)
    assert_close1d([i.rho_mass() for i in eq.phases], [0.2276324247643884, 670.1235264525617], rtol=1e-12)

    assert_close(eq.V(), 0.05633325103071669, rtol=1e-12)
    assert_close(eq.bulk.V(), 0.05633325103071669, rtol=1e-12)
    assert_close1d([i.V() for i in eq.phases], [0.1407614052926116, 4.781485612006529e-05], rtol=1e-12)

    assert_close(eq.rho(), 17.75150522476916, rtol=1e-12)
    assert_close(eq.bulk.rho(), 17.75150522476916, rtol=1e-12)
    assert_close1d([i.rho() for i in eq.phases], [7.104220066013285, 20914.002072681225], rtol=1e-12)

    assert_close(eq.Z(), 0.3984313416321555, rtol=1e-12)
    assert_close(eq.bulk.Z(), 0.3984313416321555, rtol=1e-12)
    assert_close1d([i.Z() for i in eq.phases],  [0.9955710798615588, 0.00033818281255378383], rtol=1e-12)

    assert_close(eq.V_mass(), 1.7581142614915828, rtol=1e-12)
    assert_close(eq.bulk.V_mass(), 1.7581142614915828, rtol=1e-12)
    assert_close1d([i.V_mass() for i in eq.phases], [4.393047260446541, 0.0014922621882770006], rtol=1e-12)

    # MW air may be adjusted in the future
    SG_gas_expect = 1.10647130731458
    assert_close(eq.SG_gas(), SG_gas_expect, rtol=1e-5)
    assert_close(eq.bulk.SG_gas(), SG_gas_expect, rtol=1e-5)
    assert_close1d([i.SG_gas() for i in eq.phases], [SG_gas_expect]*2, rtol=1e-5)

    assert_close(eq.SG(), 0.7951605425835767, rtol=1e-5)
    assert_close(eq.bulk.SG(), 0.7951605425835767, rtol=1e-5)
    assert_close1d([i.SG() for i in eq.phases], [0.7951605425835767]*2, rtol=1e-5)

    # Cp and Cv related
    assert_close(eq.Cp(), 85.30634675113457, rtol=1e-12)
    assert_close(eq.bulk.Cp(), 85.30634675113457, rtol=1e-12)
    assert_close1d([i.Cp() for i in eq.phases],  [44.63182543018587, 112.42269429843371], rtol=1e-12)

    assert_close(eq.Cp_mass(), 2662.340661595006, rtol=1e-12)
    assert_close(eq.bulk.Cp_mass(), 2662.340661595006, rtol=1e-12)
    assert_close1d([i.Cp_mass() for i in eq.phases], [1392.9224280421258, 3508.6194839635937], rtol=1e-12)

    assert_close(eq.Cv(), 65.77441970078021, rtol=1e-12)
    assert_close(eq.bulk.Cv(), 65.77441970078021, rtol=1e-12)
    assert_close1d([i.Cv() for i in eq.phases], [36.18313355107219, 79.87072617335762], rtol=1e-12)

    assert_close(eq.Cp_Cv_ratio(), 1.2969532401685737, rtol=1e-12)
    assert_close(eq.bulk.Cp_Cv_ratio(), 1.2969532401685737, rtol=1e-12)
    assert_close1d([i.Cp_Cv_ratio() for i in eq.phases], [1.2334980707845113, 1.4075581841389895], rtol=1e-12)

    assert_close(eq.Cv_mass(), 2052.765341986396, rtol=1e-12)
    assert_close(eq.bulk.Cv_mass(), 2052.765341986396, rtol=1e-12)
    assert_close1d([i.Cv_mass() for i in eq.phases], [1129.2457289018862, 2492.699430474936], rtol=1e-12)

    # ideal gas properties
    assert_close(eq.V_ideal_gas(), 0.14138759968016104, rtol=1e-12)
    assert_close(eq.bulk.V_ideal_gas(), 0.14138759968016104, rtol=1e-12)
    assert_close1d([i.V_ideal_gas() for i in eq.phases], [0.14138759968016104]*2, rtol=1e-12)

    assert_close(eq.Cp_ideal_gas(), 44.47452555993428, rtol=1e-12)
    assert_close(eq.bulk.Cp_ideal_gas(), 44.47452555993428, rtol=1e-12)
    assert_close1d([i.Cp_ideal_gas() for i in eq.phases], [44.47452555993428]*2, rtol=1e-12)

    assert_close(eq.Cv_ideal_gas(), 36.160062941781035, rtol=1e-12)
    assert_close(eq.bulk.Cv_ideal_gas(), 36.160062941781035, rtol=1e-12)
    assert_close1d([i.Cv_ideal_gas() for i in eq.phases], [36.160062941781035]*2, rtol=1e-12)

    assert_close(eq.Cp_Cv_ratio_ideal_gas(), 1.229934959779794, rtol=1e-12)
    assert_close(eq.bulk.Cp_Cv_ratio_ideal_gas(), 1.229934959779794, rtol=1e-12)
    assert_close1d([i.Cp_Cv_ratio_ideal_gas() for i in eq.phases], [1.229934959779794]*2, rtol=1e-12)

    assert_close(eq.H_ideal_gas(), 82.17715909331491, rtol=1e-12)
    assert_close(eq.bulk.H_ideal_gas(), 82.17715909331491, rtol=1e-12)
    assert_close1d([i.H_ideal_gas() for i in eq.phases], [82.17715909331491]*2, rtol=1e-12)

    assert_close(eq.S_ideal_gas(), 14.808945043469695, rtol=1e-12)
    assert_close(eq.bulk.S_ideal_gas(), 14.808945043469695, rtol=1e-12)
    assert_close1d([i.S_ideal_gas() for i in eq.phases], [14.808945043469695]*2, rtol=1e-12)

    assert_close(eq.G_ideal_gas(), -4360.506353947593, rtol=1e-12)
    assert_close(eq.bulk.G_ideal_gas(), -4360.506353947593, rtol=1e-12)
    assert_close1d([i.G_ideal_gas() for i in eq.phases], [-4360.506353947593]*2, rtol=1e-12)

    assert_close(eq.A_ideal_gas(), -6854.845139393565, rtol=1e-12)
    assert_close(eq.bulk.A_ideal_gas(), -6854.845139393565, rtol=1e-12)
    assert_close1d([i.A_ideal_gas() for i in eq.phases], [-6854.845139393565]*2, rtol=1e-12)

    assert_close(eq.U_ideal_gas(), -2412.161626352657, rtol=1e-12)
    assert_close(eq.bulk.U_ideal_gas(), -2412.161626352657, rtol=1e-12)
    assert_close1d([i.U_ideal_gas() for i in eq.phases], [-2412.161626352657]*2, rtol=1e-12)

    # Ideal formation basis

    assert_close(eq.H_formation_ideal_gas(), -200700.0, rtol=1e-12)
    assert_close(eq.bulk.H_formation_ideal_gas(), -200700.0, rtol=1e-12)
    assert_close1d([i.H_formation_ideal_gas() for i in eq.phases], [-200700.0]*2, rtol=1e-12)

    assert_close(eq.S_formation_ideal_gas(), -129.8, rtol=1e-12)
    assert_close(eq.bulk.S_formation_ideal_gas(), -129.8, rtol=1e-12)
    assert_close1d([i.S_formation_ideal_gas() for i in eq.phases], [-129.8]*2, rtol=1e-12)

    assert_close(eq.G_formation_ideal_gas(), -162000.13, rtol=1e-12)
    assert_close(eq.bulk.G_formation_ideal_gas(), -162000.13, rtol=1e-12)
    assert_close1d([i.G_formation_ideal_gas() for i in eq.phases], [-162000.13]*2, rtol=1e-12)

    assert_close(eq.U_formation_ideal_gas(), -215026.0985375923, rtol=1e-12)
    assert_close(eq.bulk.U_formation_ideal_gas(), -215026.0985375923, rtol=1e-12)
    assert_close1d([i.U_formation_ideal_gas() for i in eq.phases], [-215026.0985375923]*2, rtol=1e-12)

    assert_close(eq.A_formation_ideal_gas(), -176326.22853759234, rtol=1e-12)
    assert_close(eq.bulk.A_formation_ideal_gas(), -176326.22853759234, rtol=1e-12)
    assert_close1d([i.A_formation_ideal_gas() for i in eq.phases], [-176326.22853759234]*2, rtol=1e-12)

    # Pseudo critical properties
    assert_close(eq.pseudo_Tc(), constants.Tcs[0], rtol=1e-12)
    assert_close(eq.bulk.pseudo_Tc(), constants.Tcs[0], rtol=1e-12)
    assert_close1d([i.pseudo_Tc() for i in eq.phases], [constants.Tcs[0]]*2, rtol=1e-12)

    assert_close(eq.pseudo_Pc(), constants.Pcs[0], rtol=1e-12)
    assert_close(eq.bulk.pseudo_Pc(), constants.Pcs[0], rtol=1e-12)
    assert_close1d([i.pseudo_Pc() for i in eq.phases], [constants.Pcs[0]]*2, rtol=1e-12)

    assert_close(eq.pseudo_Vc(), constants.Vcs[0], rtol=1e-12)
    assert_close(eq.bulk.pseudo_Vc(), constants.Vcs[0], rtol=1e-12)
    assert_close1d([i.pseudo_Vc() for i in eq.phases], [constants.Vcs[0]]*2, rtol=1e-12)

    assert_close(eq.pseudo_Zc(), constants.Zcs[0], rtol=1e-12)
    assert_close(eq.bulk.pseudo_Zc(), constants.Zcs[0], rtol=1e-12)
    assert_close1d([i.pseudo_Zc() for i in eq.phases], [constants.Zcs[0]]*2, rtol=1e-12)

    # Standard volumes
    V_std_expect = 0.02364483003622853
    assert_close(eq.V_gas_standard(), V_std_expect, rtol=1e-12)
    assert_close(eq.bulk.V_gas_standard(), V_std_expect, rtol=1e-12)
    assert_close1d([i.V_gas_standard() for i in eq.phases], [V_std_expect]*2, rtol=1e-12)

    V_std_expect = 0.022413969545014137
    assert_close(eq.V_gas_normal(), V_std_expect, rtol=1e-12)
    assert_close(eq.bulk.V_gas_normal(), V_std_expect, rtol=1e-12)
    assert_close1d([i.V_gas_normal() for i in eq.phases], [V_std_expect]*2, rtol=1e-12)

    # Combustion properties
    Hc_expect = -764464.0
    assert_close(eq.Hc(), Hc_expect, rtol=1e-12)
    assert_close(eq.bulk.Hc(), Hc_expect, rtol=1e-12)
    assert_close1d([i.Hc() for i in eq.phases], [Hc_expect]*2, rtol=1e-12)

    Hc_mass_expect = -23858290.373904638
    assert_close(eq.Hc_mass(), Hc_mass_expect, rtol=1e-12)
    assert_close(eq.bulk.Hc_mass(), Hc_mass_expect, rtol=1e-12)
    assert_close1d([i.Hc_mass() for i in eq.phases], [Hc_mass_expect]*2, rtol=1e-12)

    Hc_lower_expect = -676489.1
    assert_close(eq.Hc_lower(), Hc_lower_expect, rtol=1e-12)
    assert_close(eq.bulk.Hc_lower(), Hc_lower_expect, rtol=1e-12)
    assert_close1d([i.Hc_lower() for i in eq.phases], [Hc_lower_expect]*2, rtol=1e-12)

    Hc_lower_mass_expect =  -21112666.368306957
    assert_close(eq.Hc_lower_mass(), Hc_lower_mass_expect, rtol=1e-12)
    assert_close(eq.bulk.Hc_lower_mass(), Hc_lower_mass_expect, rtol=1e-12)
    assert_close1d([i.Hc_lower_mass() for i in eq.phases], [Hc_lower_mass_expect]*2, rtol=1e-12)

    # Volume combustion properties
    Hc_normal_expect = -34106586.897279456
    assert_close(eq.Hc_normal(), Hc_normal_expect, rtol=1e-12)
    assert_close(eq.bulk.Hc_normal(), Hc_normal_expect, rtol=1e-12)
    assert_close1d([i.Hc_normal() for i in eq.phases], [Hc_normal_expect]*2, rtol=1e-12)

    Hc_standard_expect = -32331126.881804217
    assert_close(eq.Hc_standard(), Hc_standard_expect, rtol=1e-12)
    assert_close(eq.bulk.Hc_standard(), Hc_standard_expect, rtol=1e-12)
    assert_close1d([i.Hc_standard() for i in eq.phases], [Hc_standard_expect]*2, rtol=1e-12)

    Hc_lower_normal_expect = -30181583.79493655
    assert_close(eq.Hc_lower_normal(), Hc_lower_normal_expect, rtol=1e-12)
    assert_close(eq.bulk.Hc_lower_normal(), Hc_lower_normal_expect, rtol=1e-12)
    assert_close1d([i.Hc_lower_normal() for i in eq.phases], [Hc_lower_normal_expect]*2, rtol=1e-12)

    Hc_lower_standard_expect = -28610444.607277177
    assert_close(eq.Hc_lower_standard(), Hc_lower_standard_expect, rtol=1e-12)
    assert_close(eq.bulk.Hc_lower_standard(), Hc_lower_standard_expect, rtol=1e-12)
    assert_close1d([i.Hc_lower_standard() for i in eq.phases], [Hc_lower_standard_expect]*2, rtol=1e-12)

    # Wobbe index
    Wobbe_index_expect = 726753.2127139702
    assert_close(eq.Wobbe_index(), Wobbe_index_expect, rtol=1e-12)
    assert_close(eq.bulk.Wobbe_index(), Wobbe_index_expect, rtol=1e-12)
    assert_close1d([i.Wobbe_index() for i in eq.phases], [Wobbe_index_expect]*2, rtol=1e-12)

    Wobbe_index_lower_expect = 643118.0890022058
    assert_close(eq.Wobbe_index_lower(), Wobbe_index_lower_expect, rtol=1e-12)
    assert_close(eq.bulk.Wobbe_index_lower(), Wobbe_index_lower_expect, rtol=1e-12)
    assert_close1d([i.Wobbe_index_lower() for i in eq.phases], [Wobbe_index_lower_expect]*2, rtol=1e-12)

    Wobbe_index_mass_expect = 22681367.83301501
    assert_close(eq.Wobbe_index_mass(), Wobbe_index_mass_expect, rtol=1e-12)
    assert_close(eq.bulk.Wobbe_index_mass(), Wobbe_index_mass_expect, rtol=1e-12)
    assert_close1d([i.Wobbe_index_mass() for i in eq.phases], [Wobbe_index_mass_expect]*2, rtol=1e-12)

    Wobbe_index_lower_mass_expect = 20071184.6628818
    assert_close(eq.Wobbe_index_lower_mass(), Wobbe_index_lower_mass_expect, rtol=1e-12)
    assert_close(eq.bulk.Wobbe_index_lower_mass(), Wobbe_index_lower_mass_expect, rtol=1e-12)
    assert_close1d([i.Wobbe_index_lower_mass() for i in eq.phases], [Wobbe_index_lower_mass_expect]*2, rtol=1e-12)

    # Wobbe index volume properties
    Wobbe_index_standard_expect = 30736241.774647623
    assert_close(eq.Wobbe_index_standard(), Wobbe_index_standard_expect, rtol=1e-12)
    assert_close(eq.bulk.Wobbe_index_standard(), Wobbe_index_standard_expect, rtol=1e-12)
    assert_close1d([i.Wobbe_index_standard() for i in eq.phases], [Wobbe_index_standard_expect]*2, rtol=1e-12)

    Wobbe_index_normal_expect = 32424118.862766653
    assert_close(eq.Wobbe_index_normal(), Wobbe_index_normal_expect, rtol=1e-12)
    assert_close(eq.bulk.Wobbe_index_normal(), Wobbe_index_normal_expect, rtol=1e-12)
    assert_close1d([i.Wobbe_index_normal() for i in eq.phases], [Wobbe_index_normal_expect]*2, rtol=1e-12)

    Wobbe_index_lower_standard_expect = 27199099.677046627
    assert_close(eq.Wobbe_index_lower_standard(), Wobbe_index_lower_standard_expect, rtol=1e-12)
    assert_close(eq.bulk.Wobbe_index_lower_standard(), Wobbe_index_lower_standard_expect, rtol=1e-12)
    assert_close1d([i.Wobbe_index_lower_standard() for i in eq.phases], [Wobbe_index_lower_standard_expect]*2, rtol=1e-12)

    Wobbe_index_lower_normal_expect = 28692735.024495646
    assert_close(eq.Wobbe_index_lower_normal(), Wobbe_index_lower_normal_expect, rtol=1e-12)
    assert_close(eq.bulk.Wobbe_index_lower_normal(), Wobbe_index_lower_normal_expect, rtol=1e-12)
    assert_close1d([i.Wobbe_index_lower_normal() for i in eq.phases], [Wobbe_index_lower_normal_expect]*2, rtol=1e-12)

    # Mechanical critical point - these have an inner solver
    Tmc_expect = 512.5
    assert_close(eq.Tmc(), Tmc_expect, rtol=1e-12)
    assert_close(eq.bulk.Tmc(), Tmc_expect, rtol=1e-12)
    assert_close1d([i.Tmc() for i in eq.phases], [Tmc_expect]*2, rtol=1e-12)

    Pmc_expect = 8084000.0
    assert_close(eq.Pmc(), Pmc_expect, rtol=1e-12)
    assert_close(eq.bulk.Pmc(), Pmc_expect, rtol=1e-12)
    assert_close1d([i.Pmc() for i in eq.phases], [Pmc_expect]*2, rtol=1e-12)

    # The solver tolerance here is loose
    Vmc_expect = 0.00016203642168563802
    assert_close(eq.Vmc(), Vmc_expect, rtol=1e-4)
    assert_close(eq.bulk.Vmc(), Vmc_expect, rtol=1e-4)
    assert_close1d([i.Vmc() for i in eq.phases], [Vmc_expect]*2, rtol=1e-4)

    # The solver tolerance here is loose
    Zmc_expect = 0.30740497655001947
    assert_close(eq.Zmc(), Zmc_expect, rtol=1e-4)
    assert_close(eq.bulk.Zmc(), Zmc_expect, rtol=1e-4)
    assert_close1d([i.Zmc() for i in eq.phases], [Zmc_expect]*2, rtol=1e-4)

    # Properties calculated form derivatives
    assert_close(eq.isobaric_expansion(), 0.002008218029645217, rtol=1e-12)
    assert_close(eq.bulk.isobaric_expansion(), 0.002008218029645217, rtol=1e-12)
    assert_close1d([i.isobaric_expansion() for i in eq.phases], [0.0033751089225799308, 0.0010969574343554077], rtol=1e-12)

    assert_close(eq.kappa(), 2.277492845010776e-05, rtol=1e-12)
    assert_close(eq.bulk.kappa(), 2.277492845010776e-05, rtol=1e-12)
    assert_close1d([i.kappa() for i in eq.phases], [5.693652573977374e-05, 5.302569971013028e-10], rtol=1e-12)

    assert_close(eq.Joule_Thomson(), 1.563918814309498e-05, rtol=1e-12)
    assert_close(eq.bulk.Joule_Thomson(), 1.563918814309498e-05, rtol=1e-12)
    assert_close1d([i.Joule_Thomson() for i in eq.phases], [3.9525992445522226e-05, -2.853480585231828e-07], rtol=1e-12)

    assert_close(eq.speed_of_sound(), 235.8471087474984, rtol=1e-12)
    assert_close(eq.bulk.speed_of_sound(), 235.8471087474984, rtol=1e-12)
    assert_close1d([i.speed_of_sound() for i in eq.phases], [55.22243501081154, 356.26355790528964], rtol=1e-12)

    assert_close(eq.speed_of_sound_mass(), 1317.5639311230432, rtol=1e-12)
    assert_close(eq.bulk.speed_of_sound_mass(), 1317.5639311230432, rtol=1e-12)
    assert_close1d([i.speed_of_sound_mass() for i in eq.phases], [308.5010833731638, 1990.2724962896298], rtol=1e-12)

    # Departure properties
    assert_close(eq.H_dep(), -24186.9377252872, rtol=1e-12)
    assert_close(eq.bulk.H_dep(), -24186.9377252872, rtol=1e-12)
    assert_close1d([i.H_dep() for i in eq.phases], [-30.99680147049139, -40290.898341165004], rtol=1e-12)

    assert_close(eq.S_dep(), -80.58637166498106, rtol=1e-12)
    assert_close(eq.bulk.S_dep(), -80.58637166498106, rtol=1e-12)
    assert_close1d([i.S_dep() for i in eq.phases], [-0.0665685855920503, -134.26624038457373], rtol=1e-12)

    assert_close(eq.G_dep(), -11.026225792880723, rtol=1e-9)
    assert_close(eq.bulk.G_dep(), -11.026225792880723, rtol=1e-9)
    assert_close1d([i.G_dep() for i in eq.phases], [-11.026225792876303, -11.026225792884361], rtol=1e-9)

    assert_close(eq.U_dep(), -22686.421688611586, rtol=1e-9)
    assert_close(eq.bulk.U_dep(), -22686.421688611586, rtol=1e-9)
    assert_close1d([i.U_dep() for i in eq.phases], [-19.949574191534843, -37797.40309822495], rtol=1e-9)

    assert_close(eq.A_dep(), 1489.4898108827329, rtol=1e-9)
    assert_close(eq.bulk.A_dep(), 1489.4898108827329, rtol=1e-9)
    assert_close1d([i.A_dep() for i in eq.phases], [0.021001486080244547, 2482.4690171471666], rtol=1e-9)

    assert_close(eq.Cp_dep(), 40.83182119120029, rtol=1e-9)
    assert_close(eq.bulk.Cp_dep(), 40.83182119120029, rtol=1e-9)
    assert_close1d([i.Cp_dep() for i in eq.phases], [0.1572998702515953, 67.94816873849943], rtol=1e-9)

    assert_close(eq.Cv_dep(), 29.61435675899918, rtol=1e-12)
    assert_close(eq.bulk.Cv_dep(), 29.61435675899918, rtol=1e-12)
    assert_close1d([i.Cv_dep() for i in eq.phases], [0.023070609291153232, 43.71066323157659], rtol=1e-12)

    # Standard liquid density
    rho_mass_liquid_ref_expect = 784.8585085234012
    assert_close(eq.rho_mass_liquid_ref(), rho_mass_liquid_ref_expect, rtol=1e-12)
    assert_close(eq.bulk.rho_mass_liquid_ref(), rho_mass_liquid_ref_expect, rtol=1e-12)
    assert_close1d([i.rho_mass_liquid_ref() for i in eq.phases], [rho_mass_liquid_ref_expect]*2, rtol=1e-12)

    V_liquid_ref_expect = 4.0825014511573776e-05
    assert_close(eq.V_liquid_ref(), V_liquid_ref_expect, rtol=1e-12)
    assert_close(eq.bulk.V_liquid_ref(), V_liquid_ref_expect, rtol=1e-12)
    assert_close1d([i.V_liquid_ref() for i in eq.phases], [V_liquid_ref_expect]*2, rtol=1e-12)

    # Water contect
    assert_close(eq.molar_water_content(), 0, atol=0)
    assert_close(eq.bulk.molar_water_content(), 0, atol=0)
    assert_close1d([i.molar_water_content() for i in eq.phases], [0]*2, atol=0)

    assert_close1d(eq.zs_no_water(), [1], atol=0)
    assert_close1d(eq.bulk.zs_no_water(), [1], atol=0)
    assert_close2d([i.zs_no_water() for i in eq.phases], [[1.0]]*2, atol=0)

    assert_close1d(eq.ws_no_water(), [1], atol=0)
    assert_close1d(eq.bulk.ws_no_water(), [1], atol=0)
    assert_close2d([i.ws_no_water() for i in eq.phases], [[1.0]]*2, atol=0)

    # H/C ratio
    assert_close(eq.H_C_ratio(), 4, atol=0)
    assert_close(eq.bulk.H_C_ratio(), 4, atol=0)
    assert_close1d([i.H_C_ratio() for i in eq.phases], [4]*2, atol=0)

    assert_close(eq.H_C_ratio_mass(), 0.3356806847227889, rtol=1e-12)
    assert_close(eq.bulk.H_C_ratio_mass(), 0.3356806847227889, rtol=1e-12)
    assert_close1d([i.H_C_ratio_mass() for i in eq.phases], [0.3356806847227889]*2, rtol=1e-12)


def test_thermodynamic_derivatives_settings_with_flash():
    # Slow - any way to mock? Ideally start with a hot start, TODO
    T, P = 298.15, 1e5
    zs = [.25, 0.7, .05]
    # m = Mixture(['butanol', 'water', 'ethanol'], zs=zs)
    constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Pcs=[4414000.0, 22048320.0, 6137000.0],
                                         omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844],
                                         CASs=['71-36-3', '7732-18-5', '64-17-5'])
    HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),]

    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True, HeatCapacityGases=HeatCapacityGases)
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)

    settings = BulkSettings(dP_dT=EQUILIBRIUM_DERIVATIVE, dP_dV=EQUILIBRIUM_DERIVATIVE,
                            d2P_dV2=EQUILIBRIUM_DERIVATIVE, d2P_dT2=EQUILIBRIUM_DERIVATIVE,
                            d2P_dTdV=EQUILIBRIUM_DERIVATIVE)

    flashN = FlashVLN(constants, correlations, liquids=[liq, liq], gas=gas, settings=settings)
    res = flashN.flash(T=361.0, P=P, zs=zs)
    assert res.settings is settings
    assert res.bulk.settings is settings
    assert res.liquid_bulk.settings is settings

    # This will be taken care of in the future
    with pytest.raises(NotImplementedError):
        res.dP_dT()
    with pytest.raises(NotImplementedError):
        flashN.flash(T=361.0, V=res.V(), zs=zs)

def test_thermodynamic_derivatives_settings_with_flash_binary():
    # Tests that only need two phases
    T, P = 200.0, 1e5
    constants = ChemicalConstantsPackage(Tcs=[305.32, 469.7], Pcs=[4872000.0, 3370000.0],
                                         omegas=[0.098, 0.251], Tms=[90.3, 143.15],
                                         Tbs=[184.55, 309.21], CASs=['74-84-0', '109-66-0'],
                                         names=['ethane', 'pentane'], MWs=[30.06904, 72.14878])
    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                         HeatCapacityGas(poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866]))]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=HeatCapacityGases, skip_missing=True)
    zs = ws_to_zs(MWs=constants.MWs, ws=[.5, .5])

    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas}
    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)

    settings = BulkSettings(equilibrium_perturbation=1e-7,Joule_Thomson=EQUILIBRIUM_DERIVATIVE)

    flasher = FlashVL(constants, correlations, liquid=liq, gas=gas, settings=settings)
    res = flasher.flash(P=P, T=T, zs=zs)
    assert res.settings is settings

    # Numerical derivative
    assert_close(res.Joule_Thomson(), 0.00018067735521980137, rtol=1e-7)


def test_thermodynamic_derivatives_named_settings_with_flash():
    T, P = 298.15, 1e5
    zs = [.25, 0.7, .05]
    # m = Mixture(['butanol', 'water', 'ethanol'], zs=zs)
    constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Pcs=[4414000.0, 22048320.0, 6137000.0],
                                         omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844],
                                         CASs=['71-36-3', '7732-18-5', '64-17-5'])
    HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),]

    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True, HeatCapacityGases=HeatCapacityGases)
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)

    settings = BulkSettings(isobaric_expansion=EQUILIBRIUM_DERIVATIVE, equilibrium_perturbation=1e-7,
                            kappa=EQUILIBRIUM_DERIVATIVE)

    flashN = FlashVLN(constants, correlations, liquids=[liq, liq], gas=gas, settings=settings)

    res = flashN.flash(T=361.0, P=P, zs=zs)
    # Numeric derivative
    assert_close(res.bulk.isobaric_expansion(), 3.9202893172854045, rtol=1e-5)
    assert_close(res.isobaric_expansion(), 3.9202893172854045, rtol=1e-5)
    assert res.liquid_bulk.isobaric_expansion() is None

    assert res.liquid_bulk.kappa() is None
    assert_close(res.kappa(), 0.0010137530158341767, rtol=1e-5)
    assert_close(res.bulk.kappa(), 0.0010137530158341767, rtol=1e-5)



def test_thermodynamic_derivatives_settings():
    T, P = 298.15, 1e5
    zs = [.25, 0.7, .05]
    # m = Mixture(['butanol', 'water', 'ethanol'], zs=zs)
    constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Pcs=[4414000.0, 22048320.0, 6137000.0],
                                         omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844],
                                         CASs=['71-36-3', '7732-18-5', '64-17-5'])
    HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),]

    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True, HeatCapacityGases=HeatCapacityGases)
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    T_VLL = 361.0
    VLL_betas = [0.027939322463013245, 0.6139152961492603, 0.35814538138772645]
    VLL_zs_gas = [0.23840099709086618, 0.5786839935180893, 0.18291500939104433]
    VLL_zs_l0 = [7.619975052238078e-05, 0.9989622883894993, 0.0009615118599781685]
    VLL_zs_l1 = [0.6793120076703765, 0.19699746328631032, 0.12369052904331329]
    gas_VLL = gas.to(T=T_VLL, P=P, zs=VLL_zs_gas)
    l0_VLL = liq.to(T=T_VLL, P=P, zs=VLL_zs_l0)
    l1_VLL = liq.to(T=T_VLL, P=P, zs=VLL_zs_l1)

    VLL_kwargs = dict(T=T_VLL, P=P, zs=zs,
                     gas=gas_VLL, liquids=[l0_VLL, l1_VLL], solids=[], betas=VLL_betas,
                     flash_specs=None, flash_convergence=None,
                     constants=constants, correlations=correlations, flasher=None)

    # MOLE weighted derivatives
    settings = BulkSettings(dP_dT=MOLE_WEIGHTED, dP_dV=MOLE_WEIGHTED,
                            d2P_dV2=MOLE_WEIGHTED, d2P_dT2=MOLE_WEIGHTED,
                            d2P_dTdV=MOLE_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = 2368612.801863535, 2368604.823694247
    assert_close(res.dP_dT_frozen(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dT_frozen(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dT_frozen(), v2)

    assert_close(res.dP_dT(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dT(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dT(), v2)

    v, v2 = -91302146519714.0, -91302146426651.06
    assert_close(res.dP_dV_frozen(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dV_frozen(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dV_frozen(), v2)
    assert_close(res.dP_dV(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dV(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dV(), v2)

    v, v2 = -2059.854933409936, -2059.854681581265
    assert_close(res.d2P_dT2_frozen(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dT2_frozen(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dT2_frozen(), v2)
    assert_close(res.d2P_dT2(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dT2(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dT2(), v2)

    v, v2 =  5.690848954200077e+19, 5.690848954199457e+19
    assert_close(res.d2P_dV2_frozen(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dV2_frozen(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dV2_frozen(), v2, rtol=1e-8)
    assert_close(res.d2P_dV2(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dV2(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dV2(), v2, rtol=1e-8)

    v, v2 = -384372661273.9939, -384372661000.05206
    assert_close(res.d2P_dTdV_frozen(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dTdV_frozen(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dTdV_frozen(), v2, rtol=1e-8)
    assert_close(res.d2P_dTdV(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dTdV(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dTdV(), v2, rtol=1e-8)

    # Mass weighted derivatives
    settings = BulkSettings(dP_dT=MASS_WEIGHTED, dP_dV=MASS_WEIGHTED,
                        d2P_dV2=MASS_WEIGHTED, d2P_dT2=MASS_WEIGHTED,
                        d2P_dTdV=MASS_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = 1665265.7297690287, 1717663.9516905276
    assert_close(res.dP_dT(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dT(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dT(), v2)

    v, v2 = (-52328362725814.84, -53975176920556.54)
    assert_close(res.dP_dV(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dV(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dV(), v2)

    v, v2 = (-1558.12160618414, -1607.156622080879)
    assert_close(res.d2P_dT2(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dT2(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dT2(), v2)

    v, v2 =  (3.1171134384979878e+19, 3.2152114243026096e+19)
    assert_close(res.d2P_dV2(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dV2(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dV2(), v2, rtol=1e-8)

    v, v2 = (-222452748573.21567, -229453509386.25995)
    assert_close(res.d2P_dTdV(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dTdV(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dTdV(), v2, rtol=1e-8)

    # Volume derivatives
    settings = BulkSettings(dP_dT=VOLUME_WEIGHTED, dP_dV=VOLUME_WEIGHTED,
                            d2P_dV2=VOLUME_WEIGHTED, d2P_dT2=VOLUME_WEIGHTED,
                            d2P_dTdV=VOLUME_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)

    v, v2 = (92138.35889956845, 1664153.2082311283)
    assert_close(res.dP_dT(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dT(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dT(), v2)

    v, v2 = (-2815534213858.3413, -51001937819247.34)
    assert_close(res.dP_dV(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dV(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dV(), v2)

    v, v2 = (-86.62751615902027, -1569.0598908974252)
    assert_close(res.d2P_dT2(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dT2(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dT2(), v2)

    v, v2 = (1.666508633812045e+18, 3.0187970826232173e+19)
    assert_close(res.d2P_dV2(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dV2(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dV2(), v2, rtol=1e-8)

    v, v2 = (-11984991337.816126, -217101959098.11475)
    assert_close(res.d2P_dTdV(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dTdV(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dTdV(), v2, rtol=1e-8)




    # minimum phase prop
    settings = BulkSettings(dP_dT=MINIMUM_PHASE_PROP, dP_dV=MINIMUM_PHASE_PROP,
                            d2P_dV2=MINIMUM_PHASE_PROP, d2P_dT2=MINIMUM_PHASE_PROP,
                            d2P_dTdV=MINIMUM_PHASE_PROP)
    res = EquilibriumState(settings=settings, **VLL_kwargs)

    v, v2 = (285.5534273695696, 870098.7910584641)
    assert_close(res.dP_dT(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dT(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dT(), v2)

    v, v2 = (-144706516013762.3, -144706516013762.3)
    assert_close(res.dP_dV(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dV(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dV(), v2)

    v, v2 = (-2769.716146414676, -2769.716146414676)
    assert_close(res.d2P_dT2(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dT2(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dT2(), v2)

    v, v2 = (221845713.3633925, 1.0417348512232099e+18)
    assert_close(res.d2P_dV2(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dV2(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dV2(), v2, rtol=1e-8)

    v, v2 = (-606373312784.1843, -606373312784.1843)
    assert_close(res.d2P_dTdV(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dTdV(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dTdV(), v2, rtol=1e-8)

    # Maximum phase prop
    settings = BulkSettings(dP_dT=MAXIMUM_PHASE_PROP, dP_dV=MAXIMUM_PHASE_PROP,
                            d2P_dV2=MAXIMUM_PHASE_PROP, d2P_dT2=MAXIMUM_PHASE_PROP,
                            d2P_dTdV=MAXIMUM_PHASE_PROP)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (3350597.34335973, 3350597.34335973)
    assert_close(res.dP_dT(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dT(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dT(), v2)

    v, v2 = (-3330894.562497298, -6881570784983.291)
    assert_close(res.dP_dV(), v, rtol=1e-8)
    assert_close(res.bulk.dP_dV(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.dP_dV(), v2)

    v, v2 = (-0.00901341368309402, -1003.73644890463)
    assert_close(res.d2P_dT2(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dT2(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dT2(), v2)

    v, v2 = (9.208989802178342e+19, 9.208989802178342e+19)
    assert_close(res.d2P_dV2(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dV2(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dV2(), v2, rtol=1e-8)

    v, v2 = (-9804.883887467768, -33815343529.532673)
    assert_close(res.d2P_dTdV(), v, rtol=1e-8)
    assert_close(res.bulk.d2P_dTdV(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.d2P_dTdV(), v2, rtol=1e-8)



def test_thermodynamic_derivatives_named_settings():
    T, P = 298.15, 1e5
    zs = [.25, 0.7, .05]
    # m = Mixture(['butanol', 'water', 'ethanol'], zs=zs)
    constants = ChemicalConstantsPackage(Tcs=[563.0, 647.14, 514.0], Pcs=[4414000.0, 22048320.0, 6137000.0],
                                         omegas=[0.59, 0.344, 0.635], MWs=[74.1216, 18.01528, 46.06844],
                                         CASs=['71-36-3', '7732-18-5', '64-17-5'])
    HeatCapacityGases=[HeatCapacityGas(poly_fit=(50.0, 1000.0, [-3.787200194613107e-20, 1.7692887427654656e-16, -3.445247207129205e-13, 3.612771874320634e-10, -2.1953250181084466e-07, 7.707135849197655e-05, -0.014658388538054169, 1.5642629364740657, -7.614560475001724])),
                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                    HeatCapacityGas(poly_fit=(50.0, 1000.0, [-1.162767978165682e-20, 5.4975285700787494e-17, -1.0861242757337942e-13, 1.1582703354362728e-10, -7.160627710867427e-08, 2.5392014654765875e-05, -0.004732593693568646, 0.5072291035198603, 20.037826650765965])),]

    correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True, HeatCapacityGases=HeatCapacityGases)
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    gas = CEOSGas(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(SRKMIX, eos_kwargs, HeatCapacityGases=correlations.HeatCapacityGases, T=T, P=P, zs=zs)
    T_VLL = 361.0
    VLL_betas = [0.027939322463013245, 0.6139152961492603, 0.35814538138772645]
    VLL_zs_gas = [0.23840099709086618, 0.5786839935180893, 0.18291500939104433]
    VLL_zs_l0 = [7.619975052238078e-05, 0.9989622883894993, 0.0009615118599781685]
    VLL_zs_l1 = [0.6793120076703765, 0.19699746328631032, 0.12369052904331329]
    gas_VLL = gas.to(T=T_VLL, P=P, zs=VLL_zs_gas)
    l0_VLL = liq.to(T=T_VLL, P=P, zs=VLL_zs_l0)
    l1_VLL = liq.to(T=T_VLL, P=P, zs=VLL_zs_l1)

    VLL_kwargs = dict(T=T_VLL, P=P, zs=zs,
                     gas=gas_VLL, liquids=[l0_VLL, l1_VLL], solids=[], betas=VLL_betas,
                     flash_specs=None, flash_convergence=None,
                     constants=constants, correlations=correlations, flasher=None)



    settings = BulkSettings(isobaric_expansion=FROM_DERIVATIVE_SETTINGS,
                            dP_dT=VOLUME_WEIGHTED, dP_dV=VOLUME_WEIGHTED,
                            d2P_dV2=VOLUME_WEIGHTED, d2P_dT2=VOLUME_WEIGHTED,
                            d2P_dTdV=VOLUME_WEIGHTED)

    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (3.7457177005192036e-05, 0.0006765320296813641)
    assert_close(res.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.bulk.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.isobaric_expansion(), v2)

    settings = BulkSettings(isobaric_expansion=MOLE_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (0.0011408916410772427, 0.001059818733856218)
    assert_close(res.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.bulk.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.isobaric_expansion(), v2)

    settings = BulkSettings(isobaric_expansion=MASS_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (0.0012749042097648041, 0.0012237061311355878)
    assert_close(res.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.bulk.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.isobaric_expansion(), v2)

    settings = BulkSettings(isobaric_expansion=VOLUME_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (0.0028096620373989126, 0.0012336359059691522)
    assert_close(res.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.bulk.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.isobaric_expansion(), v2)

    settings = BulkSettings(isobaric_expansion=MAXIMUM_PHASE_PROP)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (0.0029017492220275273, 0.0013809854111036487)
    assert_close(res.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.bulk.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.isobaric_expansion(), v2)

    settings = BulkSettings(isobaric_expansion=MINIMUM_PHASE_PROP)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (0.0009206892068839951, 0.0009206892068839951)
    assert_close(res.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.bulk.isobaric_expansion(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.isobaric_expansion(), v2)

    # Kappa setting

    settings = BulkSettings(kappa=FROM_DERIVATIVE_SETTINGS,
                            dP_dT=VOLUME_WEIGHTED, dP_dV=VOLUME_WEIGHTED,
                            d2P_dV2=VOLUME_WEIGHTED, d2P_dT2=VOLUME_WEIGHTED,
                            d2P_dTdV=VOLUME_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (4.065318446361809e-10, 4.0653229903060884e-10)
    assert_close(res.kappa(), v, rtol=1e-8)
    assert_close(res.bulk.kappa(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.kappa(), v2)

    settings = BulkSettings(kappa=MOLE_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (2.8465214826316303e-07, 7.371276386739713e-10)
    assert_close(res.kappa(), v, rtol=1e-8)
    assert_close(res.bulk.kappa(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.kappa(), v2)

    settings = BulkSettings(kappa=MASS_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (3.11147790096034e-07, 1.1387317549953479e-09)
    assert_close(res.kappa(), v, rtol=1e-8)
    assert_close(res.bulk.kappa(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.kappa(), v2)

    settings = BulkSettings(kappa=VOLUME_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (9.60092945535878e-06, 1.1670430800087437e-09)
    assert_close(res.kappa(), v, rtol=1e-8)
    assert_close(res.bulk.kappa(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.kappa(), v2)

    settings = BulkSettings(kappa=MAXIMUM_PHASE_PROP)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (1.0161843437697627e-05, 1.5871593263837287e-09)
    assert_close(res.kappa(), v, rtol=1e-8)
    assert_close(res.bulk.kappa(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.kappa(), v2)

    settings = BulkSettings(kappa=MINIMUM_PHASE_PROP)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (2.747836020071562e-10, 2.747836020071562e-10)
    assert_close(res.kappa(), v, rtol=1e-8)
    assert_close(res.bulk.kappa(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.kappa(), v2)

    # Joule Thomson Coefficient
    settings = BulkSettings(Joule_Thomson=FROM_DERIVATIVE_SETTINGS,
                            dP_dT=VOLUME_WEIGHTED, dP_dV=VOLUME_WEIGHTED,
                            d2P_dV2=VOLUME_WEIGHTED, d2P_dT2=VOLUME_WEIGHTED,
                            d2P_dTdV=VOLUME_WEIGHTED)

    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (-7.240388523395516e-06, -3.1091771780270116e-07)
    assert_close(res.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.bulk.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.Joule_Thomson(), v2)

    settings = BulkSettings(Joule_Thomson=MOLE_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (3.993637901448662e-07, -2.1061721898004138e-07)
    assert_close(res.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.bulk.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.Joule_Thomson(), v2)

    settings = BulkSettings(Joule_Thomson=MASS_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (4.3752173371294737e-07, -2.3579009743036363e-07)
    assert_close(res.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.bulk.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.Joule_Thomson(), v2)

    settings = BulkSettings(Joule_Thomson=VOLUME_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (2.061401120148966e-05, -2.37212984942978e-07)
    assert_close(res.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.bulk.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.Joule_Thomson(), v2)

    settings = BulkSettings(Joule_Thomson=MAXIMUM_PHASE_PROP)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 =(2.1832347936583476e-05, -1.9236927421380777e-07)
    assert_close(res.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.bulk.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.Joule_Thomson(), v2)

    settings = BulkSettings(Joule_Thomson=MINIMUM_PHASE_PROP)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (-2.583274386299929e-07, -2.583274386299929e-07)
    assert_close(res.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.bulk.Joule_Thomson(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.Joule_Thomson(), v2)

    # Speed of Sound
    settings = BulkSettings(speed_of_sound=FROM_DERIVATIVE_SETTINGS,
                            dP_dT=VOLUME_WEIGHTED, dP_dV=VOLUME_WEIGHTED,
                            d2P_dV2=VOLUME_WEIGHTED, d2P_dT2=VOLUME_WEIGHTED,
                            d2P_dTdV=VOLUME_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (1472.7188644943863, 377.43483503795505)
    assert_close(res.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.bulk.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.speed_of_sound(), v2)

    settings = BulkSettings(speed_of_sound=MOLE_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (324.60817499927214, 322.9861514323374)
    assert_close(res.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.bulk.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.speed_of_sound(), v2)

    settings = BulkSettings(speed_of_sound=MASS_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (297.3316752171302, 304.86189141828106)
    assert_close(res.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.bulk.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.speed_of_sound(), v2)

    settings = BulkSettings(speed_of_sound=VOLUME_WEIGHTED)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (71.56743277633015, 302.82216631626983)
    assert_close(res.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.bulk.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.speed_of_sound(), v2)

    settings = BulkSettings(speed_of_sound=MAXIMUM_PHASE_PROP)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (367.1061252594205, 367.1061252594205)
    assert_close(res.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.bulk.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.speed_of_sound(), v2)

    settings = BulkSettings(speed_of_sound=MINIMUM_PHASE_PROP)
    res = EquilibriumState(settings=settings, **VLL_kwargs)
    v, v2 = (58.05522195758289, 272.55436171551884)
    assert_close(res.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.bulk.speed_of_sound(), v, rtol=1e-8)
    assert_close(res.liquid_bulk.speed_of_sound(), v2)