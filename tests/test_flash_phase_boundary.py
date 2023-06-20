'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2022, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
'''

from fluids.numerics import assert_close, normalize

from thermo import PRMIX, CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashVLN, HeatCapacityGas, PropertyCorrelationsPackage, VaporPressure
from thermo.flash.flash_utils import generate_pure_phase_boolean_check


def test_water_C1_C8_water_formation_temperature():
    T = 298.15
    P = 101325.0
    omegas = [0.344, 0.008, 0.394]
    Tcs = [647.14, 190.564, 568.7]
    Pcs = [22048320.0, 4599000.0, 2490000.0]
    kijs=[[0,0, 0],[0,0, 0.0496], [0,0.0496,0]]
    zs = [1.0/3.0]*3
    N = len(zs)

    HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                         HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                         HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.069661592422583e-22, -1.2992882995593864e-18, 8.808066659263286e-15, -2.1690080247294972e-11, 2.8519221306107026e-08, -2.187775092823544e-05, 0.009432620102532702, -1.5719488702446165, 217.60587499269303]))]
    constants = ChemicalConstantsPackage(Tcs=Tcs, Pcs=Pcs, omegas=omegas, MWs=[18.01528, 16.04246, 114.22852],
                                         CASs=['7732-18-5', '74-82-8', '111-65-9'])
    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases)
    eos_kwargs = dict(Tcs=Tcs, Pcs=Pcs, omegas=omegas, kijs=kijs)

    gas = CEOSGas(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)
    liq = CEOSLiquid(PRMIX, eos_kwargs, HeatCapacityGases=HeatCapacityGases, T=T, P=P, zs=zs)


    flashN = FlashVLN(constants, properties, liquids=[liq, liq], gas=gas)

    # find some temperatures and pressures at which water forms a phase
    check = generate_pure_phase_boolean_check(idx=0, required_zi=.5)

    # temperature
    res = flashN.flash_phase_boundary(zs=zs, P=5e5, phase_frac_check=check)
    assert_close(res.T, 393.23623952686864, rtol=3e-5)

    res = flashN.flash_phase_boundary(zs=zs, P=1e7, phase_frac_check=check)
    assert_close(res.T, 479.5159347779361, rtol=3e-5)
    # pressure
    res = flashN.flash_phase_boundary(zs=zs, T=400, phase_frac_check=check)
    assert_close(res.P, 624775.8877570137, rtol=3e-5)

    res = flashN.flash_phase_boundary(zs=zs, T=300, phase_frac_check=check)
    assert_close(res.P, 8240.1808789621, rtol=3e-5)

    res = flashN.flash_phase_boundary(zs=zs, T=265, phase_frac_check=check)
    assert_close(res.P, 762.2869779418065, rtol=3e-5)

    res = flashN.flash_phase_boundary(zs=zs, T=230, phase_frac_check=check)
    assert_close(res.P, 27.55802639091696, rtol=3e-5)

    # find some temperatures and pressures at which C8 forms a phase
    check = generate_pure_phase_boolean_check(idx=2, required_zi=.3)
    zs = [.5, .4, .1]


    T_expect_1e5 = 340.99952412066443
    res = flashN.flash_phase_boundary(zs=zs, P=1e5, phase_frac_check=check)
    assert_close(res.T, T_expect_1e5, rtol=3e-5)

    # do one with H/S/U, on the same point as before
    res = flashN.flash_phase_boundary(zs=zs, H=res.H(), phase_frac_check=check)
    assert_close(res.T, T_expect_1e5, rtol=3e-5)

    res = flashN.flash_phase_boundary(zs=zs, S=res.S(), phase_frac_check=check)
    assert_close(res.T, T_expect_1e5, rtol=3e-5)

    res = flashN.flash_phase_boundary(zs=zs, U=res.U(), phase_frac_check=check)
    assert_close(res.T, T_expect_1e5, rtol=3e-5)

    # do a few more points
    res = flashN.flash_phase_boundary(zs=zs, P=1243, phase_frac_check=check)
    assert_close(res.T, 262.4209165306811, rtol=3e-5)

    res = flashN.flash_phase_boundary(zs=zs, P=47, phase_frac_check=check)
    assert_close(res.T, 225.7589560802028, rtol=3e-5)

    res = flashN.flash_phase_boundary(zs=zs, P=794923, phase_frac_check=check)
    assert_close(res.T, 400.5281447689433, rtol=3e-5)



def test_water_natural_gas_formation_temperature():

    # constants, properties = ChemicalConstantsPackage.from_IDs(['nitrogen', 'methane', 'ethane', 'propane',
    #                                                            'n-butane', 'n-pentane','isopentane', 'hexane',
    #                                                            'decane', 'hydrogen sulfide', 'CO2', 'water',
    #                                                            'oxygen', 'argon'])

    constants = ChemicalConstantsPackage(atomss=[{'N': 2}, {'C': 1, 'H': 4}, {'C': 2, 'H': 6}, {'C': 3, 'H': 8}, {'C': 4, 'H': 10}, {'C': 5, 'H': 12}, {'C': 5, 'H': 12}, {'C': 6, 'H': 14}, {'C': 10, 'H': 22}, {'H': 2, 'S': 1}, {'C': 1, 'O': 2}, {'H': 2, 'O': 1}, {'O': 2}, {'Ar': 1}], CASs=['7727-37-9', '74-82-8', '74-84-0', '74-98-6', '106-97-8', '109-66-0', '78-78-4', '110-54-3', '124-18-5', '7783-06-4', '124-38-9', '7732-18-5', '7782-44-7', '7440-37-1'], MWs=[28.0134, 16.04246, 30.06904, 44.09562, 58.1222, 72.14878, 72.14878, 86.17536, 142.28168, 34.08088, 44.0095, 18.01528, 31.9988, 39.948], names=['nitrogen', 'methane', 'ethane', 'propane', 'butane', 'pentane', '2-methylbutane', 'hexane', 'decane', 'hydrogen sulfide', 'carbon dioxide', 'water', 'oxygen', 'argon'], omegas=[0.04, 0.008, 0.098, 0.152, 0.193, 0.251, 0.227, 0.2975, 0.49, 0.1, 0.2252, 0.344, 0.021, -0.004], Pcs=[3394387.5, 4599000.0, 4872000.0, 4248000.0, 3796000.0, 3370000.0, 3380000.0, 3025000.0, 2110000.0, 8936865.0, 7376460.0, 22048320.0, 5042945.25, 4873732.5], Tbs=[77.355, 111.65, 184.55, 231.04, 272.65, 309.21, 300.98, 341.87, 447.25, 213.6, 194.67, 373.124, 90.188, 87.302], Tcs=[126.2, 190.564, 305.32, 369.83, 425.12, 469.7, 460.4, 507.6, 611.7, 373.2, 304.2, 647.14, 154.58, 150.8], Tms=[63.15, 90.75, 90.3, 85.5, 135.05, 143.15, 113.15, 178.075, 243.225, 187.65, 216.65, 273.15, 54.36, 83.81], Vcs=[8.95e-05, 9.86e-05, 0.0001455, 0.0002, 0.000255, 0.000311, 0.000306, 0.000368, 0.000624, 9.85e-05, 9.4e-05, 5.6e-05, 7.34e-05, 7.49e-05])

    HeatCapacityGases = [HeatCapacityGas(CASRN="7727-37-9", MW=28.0134, similarity_variable=0.07139440410660612, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
     HeatCapacityGas(CASRN="74-82-8", MW=16.04246, similarity_variable=0.3116728980468083, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
     HeatCapacityGas(CASRN="74-84-0", MW=30.06904, similarity_variable=0.26605438683775734, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
     HeatCapacityGas(CASRN="74-98-6", MW=44.09562, similarity_variable=0.24945788266499033, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [7.008452174279456e-22, -1.7927920989992578e-18, 1.1218415948991092e-17, 4.23924157032547e-12, -5.279987063309569e-09, 2.5119646468572195e-06, -0.0004080663744697597, 0.1659704314379956, 26.107282495650367])),
     HeatCapacityGas(CASRN="106-97-8", MW=58.1222, similarity_variable=0.24087181834135665, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-2.608494166540452e-21, 1.3127902917979555e-17, -2.7500977814441112e-14, 3.0563338307642794e-11, -1.866070373718589e-08, 5.4505831355984375e-06, -0.00024022110003950325, 0.04007078628096955, 55.70646822218319])),
     HeatCapacityGas(CASRN="109-66-0", MW=72.14878, similarity_variable=0.235624219841278, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [7.537198394065234e-22, -4.946850205122326e-18, 1.4223747507170372e-14, -2.3451318313798008e-11, 2.4271676873997662e-08, -1.6055220805830093e-05, 0.006379734000450042, -1.0360272314628292, 141.84695243411866])),
     HeatCapacityGas(CASRN="78-78-4", MW=72.14878, similarity_variable=0.235624219841278, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-3.3940692195566165e-21, 1.7702350857101215e-17, -3.908639931077815e-14, 4.7271630395909585e-11, -3.371762950874771e-08, 1.4169051870903116e-05, -0.003447090979499268, 0.7970786116058249, -8.8294795805881])),
     HeatCapacityGas(CASRN="110-54-3", MW=86.17536, similarity_variable=0.2320849022272724, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [1.3740654453881647e-21, -8.344496203280677e-18, 2.2354782954548568e-14, -3.4659555330048226e-11, 3.410703030634579e-08, -2.1693611029230923e-05, 0.008373280796376588, -1.356180511425385, 175.67091124888998])),
     HeatCapacityGas(CASRN="124-18-5", MW=142.28168, similarity_variable=0.22490597524572384, extrapolation="linear", method="POLY_FIT", poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735])),
     HeatCapacityGas(CASRN="7783-06-4", MW=34.08088, similarity_variable=0.08802589604493781, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.4967172940791855e-22, 1.7086617923088487e-18, -3.505442235019261e-15, 3.911995832871371e-12, -2.56012228400194e-09, 9.620884103239162e-07, -0.00016570643705524543, 0.011886900701175745, 32.972342195898534])),
     HeatCapacityGas(CASRN="124-38-9", MW=44.0095, similarity_variable=0.0681671002851657, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-3.1115474168865828e-21, 1.39156078498805e-17, -2.5430881416264243e-14, 2.4175307893014295e-11, -1.2437314771044867e-08, 3.1251954264658904e-06, -0.00021220221928610925, 0.000884685506352987, 29.266811602924644])),
     HeatCapacityGas(CASRN="7732-18-5", MW=18.01528, similarity_variable=0.16652530518537598, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
     HeatCapacityGas(CASRN="7782-44-7", MW=31.9988, similarity_variable=0.06250234383789392, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
     HeatCapacityGas(CASRN="7440-37-1", MW=39.948, similarity_variable=0.025032542304996495, extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-1.0939921922581918e-31, 4.144146614006628e-28, -6.289942296644484e-25, 4.873620648503505e-22, -2.0309301195845294e-19, 4.3863747689727484e-17, -4.29308508081826e-15, 20.786156545383236]))]

    VaporPressures = [VaporPressure(CASRN="7727-37-9", Tb=77.355, Tc=126.2, Pc=3394387.5, omega=0.04, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(63.2, 126.18199999999999, [5.490876411024536e-15, -3.709517805130509e-12, 1.0593254238679989e-09, -1.6344291780087318e-07, 1.4129990091975526e-05, -0.0005776268289835264, -0.004489180523814208, 1.511854256824242, -36.95425216567675])),
     VaporPressure(CASRN="74-82-8", Tb=111.65, Tc=190.564, Pc=4599000.0, omega=0.008, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(90.8, 190.554, [1.2367137894255505e-16, -1.1665115522755316e-13, 4.4703690477414014e-11, -8.405199647262538e-09, 5.966277509881474e-07, 5.895879890001534e-05, -0.016577129223752325, 1.502408290283573, -42.86926854012409])),
     VaporPressure(CASRN="74-84-0", Tb=184.55, Tc=305.32, Pc=4872000.0, omega=0.098, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(90.4, 305.312, [-1.1908381885079786e-17, 2.1355746620587145e-14, -1.66363909858873e-11, 7.380706042464946e-09, -2.052789573477409e-06, 0.00037073086909253047, -0.04336716238170919, 3.1418840094903784, -102.75040650505277])),
     VaporPressure(CASRN="74-98-6", Tb=231.04, Tc=369.83, Pc=4248000.0, omega=0.152, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(85.53500000000001, 369.88, [-6.614459112569553e-18, 1.3568029167021588e-14, -1.2023152282336466e-11, 6.026039040950274e-09, -1.877734093773071e-06, 0.00037620249872919755, -0.048277894617307984, 3.790545023359657, -137.90784855852178])),
     VaporPressure(CASRN="106-97-8", Tb=272.65, Tc=425.12, Pc=3796000.0, omega=0.193, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(135, 425.115, [-7.814862594786785e-19, 1.9990087338619667e-15, -2.222658378316964e-12, 1.4088883433786636e-09, -5.606069245923512e-07, 0.00014504634895991808, -0.024330255528558123, 2.5245680747528114, -119.71493146261565])),
     VaporPressure(CASRN="109-66-0", Tb=309.21, Tc=469.7, Pc=3370000.0, omega=0.251, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(144, 469.6900000000001, [-4.615267824440021e-19, 1.2871894720673305e-15, -1.559590276548174e-12, 1.0761461914304972e-09, -4.6539407107989163e-07, 0.0001305817686830925, -0.023686309296601378, 2.64766854437685, -136.66909337025592])),
     VaporPressure(CASRN="78-78-4", Tb=300.98, Tc=460.4, Pc=3380000.0, omega=0.227, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(112.66000000000001, 460.34000000000003, [-1.225276030507864e-18, 3.156674603724423e-15, -3.515687989427183e-12, 2.216101778567771e-09, -8.688771638626119e-07, 0.0002190280669088704, -0.035326967454100376, 3.4738307292778865, -159.25081271644115])),
     VaporPressure(CASRN="110-54-3", Tb=341.87, Tc=507.6, Pc=3025000.0, omega=0.2975, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(177.84, 507.81, [-1.604637831500919e-19, 4.982429526452368e-16, -6.744889606698199e-13, 5.223525465352799e-10, -2.5493665759436215e-07, 8.125588912741348e-05, -0.01686860265374473, 2.174938146922705, -129.28820562410874])),
     VaporPressure(CASRN="124-18-5", Tb=447.25, Tc=611.7, Pc=2110000.0, omega=0.49, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(243.51, 617.69, [-1.9653193622863184e-20, 8.32071200890499e-17, -1.5159284607404818e-13, 1.5658305222329732e-10, -1.0129531274368712e-07, 4.2609908802380584e-05, -0.01163326014833186, 1.962044867057741, -153.15601192906817])),
     VaporPressure(CASRN="7783-06-4", Tb=213.6, Tc=373.2, Pc=8936865.0, omega=0.1, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(187.70999999999998, 373.09000000000003, [1.6044335409399552e-18, -3.2731123710759124e-15, 2.851504830400211e-12, -1.369466760280872e-09, 3.867598663620643e-07, -6.147439138821642e-05, 0.003971709848411946, 0.2532292056665513, -32.18677016824108])),
     VaporPressure(CASRN="124-38-9", Tb=194.67, Tc=304.2, Pc=7376460.0, omega=0.2252, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(216.602, 304.1182, [3.141153183434152e-16, -6.380334249054363e-13, 5.659255411862981e-10, -2.862785039666638e-07, 9.032467874853838e-05, -0.018198707888890574, 2.285855947253462, -163.51141480675344, 5103.172975291208])),
     VaporPressure(CASRN="7732-18-5", Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317])),
     VaporPressure(CASRN="7782-44-7", Tb=90.188, Tc=154.58, Pc=5042945.25, omega=0.021, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(54.370999999999995, 154.57100000000003, [-9.865296960381724e-16, 9.716055729011619e-13, -4.163287834047883e-10, 1.0193358930366495e-07, -1.57202974507404e-05, 0.0015832482627752501, -0.10389607830776562, 4.24779829961549, -74.89465804494587])),
     VaporPressure(CASRN="7440-37-1", Tb=87.302, Tc=150.8, Pc=4873732.5, omega=-0.004, extrapolation="AntoineAB|DIPPR101_ABC", method="POLY_FIT", poly_fit=(83.816, 150.67700000000002, [3.156255133278695e-15, -2.788016448186089e-12, 1.065580375727257e-09, -2.2940542608809444e-07, 3.024735996501385e-05, -0.0024702132398995436, 0.11819673125756014, -2.684020790786307, 20.312746972164785]))]


    properties = PropertyCorrelationsPackage(constants=constants, HeatCapacityGases=HeatCapacityGases, VaporPressures=VaporPressures, skip_missing=True)

    kijs = [[0.0, 0.0289, 0.0533, 0.0878, 0.0711, 0.1, 0.0922, 0.1496, 0.1122, 0.1652, -0.0122, 0, -0.0159, -0.0004], [0.0289, 0.0, -0.0059, 0.0119, 0.0185, 0.023, -0.0056, 0.04, 0.0411, 0, 0.0978, 0, 0, 0.0152], [0.0533, -0.0059, 0.0, 0.0011, 0.0089, 0.0078, 0, -0.04, 0.0144, 0.0952, 0.13, 0, 0, 0], [0.0878, 0.0119, 0.0011, 0.0, 0.0033, 0.0267, 0.0111, 0.0007, 0.0, 0.0878, 0.1315, 0, 0, 0], [0.0711, 0.0185, 0.0089, 0.0033, 0.0, 0.0174, 0, -0.0056, 0.0078, 0, 0.1352, 0, 0, 0], [0.1, 0.023, 0.0078, 0.0267, 0.0174, 0.0, 0, 0, 0, 0.063, 0.1252, 0, 0, 0], [0.0922, -0.0056, 0, 0.0111, 0, 0, 0.0, 0, 0, 0, 0.1219, 0, 0, 0], [0.1496, 0.04, -0.04, 0.0007, -0.0056, 0, 0, 0.0, 0, 0, 0.11, 0, 0, 0], [0.1122, 0.0411, 0.0144, 0.0, 0.0078, 0, 0, 0, 0.0, 0.0333, 0.1141, 0, 0, 0], [0.1652, 0, 0.0952, 0.0878, 0, 0.063, 0, 0, 0.0333, 0.0, 0.0967, 0.0394, 0, 0], [-0.0122, 0.0978, 0.13, 0.1315, 0.1352, 0.1252, 0.1219, 0.11, 0.1141, 0.0967, 0.0, 0.0952, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0394, 0.0952, 0.0, 0, 0], [-0.0159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0089], [-0.0004, 0.0152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0089, 0.0]]

    check = generate_pure_phase_boolean_check(idx=11, required_zi=.5)


    eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
    gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
    flasher = FlashVLN(constants, properties, liquids=[liquid, liquid], gas=gas)

    zs =      [.03, .816, .05, .03,
              .01, .0123, .01204, .00653,
              .0053, .01, .004224, 1e-2,
              .0012, .012306]
    zs = normalize(zs)

    res = flasher.flash_phase_boundary(zs=zs, P=1.1e5, phase_frac_check=check)
    assert_close(res.T, 284.12615204391284, atol=.2)

    res = flasher.flash_phase_boundary(zs=zs, P=1e7, phase_frac_check=check)
    assert_close(res.T, 350.5751004412331, rtol=1e-4)
