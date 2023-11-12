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

import os
from math import *

import numpy as np
import pytest
from fluids.numerics import *

import thermo
from thermo import *
from thermo.coolprop import *
from thermo.test_utils import mark_plot_unsupported
from thermo.flash.flash_utils import cm_flash_tol

try:
    import matplotlib.pyplot as plt
except:
    pass



# Delete this
# from .test_flash_pure import (pure_fluids, constants, HeatCapacityGases, VaporPressures, VolumeLiquids, correlations)

pure_fluids = ['water', 'methane', 'ethane', 'decane', 'ammonia', 'nitrogen', 'oxygen', 'methanol', 'eicosane', 'hydrogen']

"""# Recreate the below with the following:
N = len(pure_fluids)
m = Mixture(pure_fluids, zs=[1/N]*N, T=298.15, P=1e5)
print(m.constants.make_str(delim=', \n', properties=('Tcs', 'Pcs', 'omegas', 'MWs', "CASs")))
correlations = m.properties()
print(correlations.as_poly_fit(['HeatCapacityGases']))
"""
constants = ChemicalConstantsPackage(Tcs=[647.14, 190.56400000000002, 305.32, 611.7, 405.6, 126.2, 154.58, 512.5,
                                          768.0,
                                          33.2
                                          ],
            Pcs=[22048320.0, 4599000.0, 4872000.0, 2110000.0, 11277472.5, 3394387.5, 5042945.25, 8084000.0,
                 1070000.0,
                 1296960.0
                 ],
            omegas=[0.344, 0.008, 0.098, 0.49, 0.25, 0.04, 0.021, 0.559,
                    0.8805,
                    -0.22,
                    ],
            MWs=[18.01528, 16.04246, 30.06904, 142.28168, 17.03052, 28.0134, 31.9988, 32.04186,
                 282.54748,
                 2.01588
                 ],
            Vcs=[5.6e-05, 9.86e-05, 0.0001455, 0.000624, 7.25e-05, 8.95e-05, 7.34e-05, 0.000117, 0.001325, 6.5e-05],
            CASs=['7732-18-5', '74-82-8', '74-84-0', '124-18-5', '7664-41-7', '7727-37-9', '7782-44-7', '67-56-1',
                  '112-95-8',
                  '1333-74-0'
                  ])

HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [5.543665000518528e-22, -2.403756749600872e-18, 4.2166477594350336e-15, -3.7965208514613565e-12, 1.823547122838406e-09, -4.3747690853614695e-07, 5.437938301211039e-05, -0.003220061088723078, 33.32731489750759])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [6.7703235945157e-22, -2.496905487234175e-18, 3.141019468969792e-15, -8.82689677472949e-13, -1.3709202525543862e-09, 1.232839237674241e-06, -0.0002832018460361874, 0.022944239587055416, 32.67333514157593])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.115386645067898e-21, -3.2034776773408394e-17, 5.957592282542187e-14, -5.91169369931607e-11, 3.391209091071677e-08, -1.158730780040934e-05, 0.002409311277400987, -0.18906638711444712, 37.94602410497228])),
                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-1.702672546011891e-21, 6.6751002084997075e-18, -7.624102919104147e-15, -4.071140876082743e-12, 1.863822577724324e-08, -1.9741705032236747e-05, 0.009781408958916831, -1.6762677829939379, 252.8975930305735])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.444966286051841e-23, 9.444106746563928e-20, -1.2490299714587002e-15, 2.6693560979905865e-12, -2.5695131746723413e-09, 1.2022442523089315e-06, -0.00021492132731007108, 0.016616385291696574, 32.84274656062226])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [-6.496329615255804e-23, 2.1505678500404716e-19, -2.2204849352453665e-16, 1.7454757436517406e-14, 9.796496485269412e-11, -4.7671178529502835e-08, 8.384926355629239e-06, -0.0005955479316119903, 29.114778709934264])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [7.682842888382947e-22, -3.3797331490434755e-18, 6.036320672021355e-15, -5.560319277907492e-12, 2.7591871443240986e-09, -7.058034933954475e-07, 9.350023770249747e-05, -0.005794412013028436, 29.229215579932934])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [2.3511458696647882e-21, -9.223721411371584e-18, 1.3574178156001128e-14, -8.311274917169928e-12, 4.601738891380102e-10, 1.78316202142183e-06, -0.0007052056417063217, 0.13263597297874355, 28.44324970462924])),
                     HeatCapacityGas(poly_fit=(200.0, 1000.0, [-2.075118433508619e-20, 1.0383055980949049e-16, -2.1577805903757125e-13, 2.373511052680461e-10, -1.4332562489496906e-07, 4.181755403465859e-05, -0.0022544761674344544, -0.15965342941876415, 303.71771182550816])),
                     HeatCapacityGas(poly_fit=(50.0, 1000.0, [1.1878323802695824e-20, -5.701277266842367e-17, 1.1513022068830274e-13, -1.270076105261405e-10, 8.309937583537026e-08, -3.2694889968431594e-05, 0.007443050245274358, -0.8722920255910297, 66.82863369121873]))]

VaporPressures = [VaporPressure(exp_poly_fit=(273.17, 647.086, [-2.8478502840358144e-21, 1.7295186670575222e-17, -4.034229148562168e-14, 5.0588958391215855e-11, -3.861625996277003e-08, 1.886271475957639e-05, -0.005928371869421494, 1.1494956887882308, -96.74302379151317])),
                  VaporPressure(exp_poly_fit=(90.8, 190.554, [1.2367137894255505e-16, -1.1665115522755316e-13, 4.4703690477414014e-11, -8.405199647262538e-09, 5.966277509881474e-07, 5.895879890001534e-05, -0.016577129223752325, 1.502408290283573, -42.86926854012409])),
                  VaporPressure(exp_poly_fit=(90.4, 305.312, [-1.1908381885079786e-17, 2.1355746620587145e-14, -1.66363909858873e-11, 7.380706042464946e-09, -2.052789573477409e-06, 0.00037073086909253047, -0.04336716238170919, 3.1418840094903784, -102.75040650505277])),
                  VaporPressure(exp_poly_fit=(243.51, 617.69, [-1.9653193622863184e-20, 8.32071200890499e-17, -1.5159284607404818e-13, 1.5658305222329732e-10, -1.0129531274368712e-07, 4.2609908802380584e-05, -0.01163326014833186, 1.962044867057741, -153.15601192906817])),
                  VaporPressure(exp_poly_fit=(195.505, 405.39, [1.8775319752114198e-19, -3.2834459725160406e-16, 1.9723813042226462e-13, -1.3646182471796847e-11, -4.348131713052942e-08, 2.592796525478491e-05, -0.007322263143033041, 1.1431876410642319, -69.06950797691312])),
                  VaporPressure(exp_poly_fit=(63.2, 126.18199999999999, [5.490876411024536e-15, -3.709517805130509e-12, 1.0593254238679989e-09, -1.6344291780087318e-07, 1.4129990091975526e-05, -0.0005776268289835264, -0.004489180523814208, 1.511854256824242, -36.95425216567675])),
                  VaporPressure(exp_poly_fit=(54.370999999999995, 154.57100000000003, [-9.865296960381724e-16, 9.716055729011619e-13, -4.163287834047883e-10, 1.0193358930366495e-07, -1.57202974507404e-05, 0.0015832482627752501, -0.10389607830776562, 4.24779829961549, -74.89465804494587])),
                  VaporPressure(exp_poly_fit=(175.7, 512.49, [-1.446088049406911e-19, 4.565038519454878e-16, -6.278051259204248e-13, 4.935674274379539e-10, -2.443464113936029e-07, 7.893819658700523e-05, -0.016615779444332356, 2.1842496316772264, -134.19766175812708])),
                  VaporPressure(exp_poly_fit=(309.58, 768.0, [-7.623520807550015e-21, 3.692055956856486e-17, -7.843475114100242e-14, 9.586109955308152e-11, -7.420461759412686e-08, 3.766437656499104e-05, -0.012473026309766598, 2.5508091747355173, -247.8254681597346])),
                  VaporPressure(exp_poly_fit=(13.967, 33.135000000000005, [2.447649824484286e-11, -1.9092317903068513e-09, -9.871694369510465e-08, 1.8738775057921993e-05, -0.0010528586872742638, 0.032004931729483245, -0.5840683278086365, 6.461815322744102, -23.604507906951046])),]

VolumeLiquids = [VolumeLiquid(poly_fit=(273.17, 637.096, [9.00307261049824e-24, -3.097008950027417e-20, 4.608271228765265e-17, -3.8726692841874345e-14, 2.0099220218891486e-11, -6.596204729785676e-09, 1.3368112879131157e-06, -0.00015298762503607717, 0.007589247005014652])),
                 VolumeLiquid(poly_fit=(90.8, 180.564, [7.730541828225242e-20, -7.911042356530585e-17, 3.51935763791471e-14, -8.885734012624568e-12, 1.3922694980104743e-09, -1.3860056394382538e-07, 8.560110533953199e-06, -0.00029978743425740123, 0.004589555868318768])),
                 VolumeLiquid(poly_fit=(90.4, 295.322, [8.94814934875547e-22, -1.2714020587822148e-18, 7.744406810864275e-16, -2.6371441222674234e-13, 5.484329710203708e-11, -7.125679072517415e-09, 5.646423685845822e-07, -2.487300046509999e-05, 0.0005100376035218567])),
                 VolumeLiquid(poly_fit=(243.51, 607.7, [1.0056823442253386e-22, -3.2166293088353376e-19, 4.442027873447809e-16, -3.4574825216883073e-13, 1.6583965814129937e-10, -5.018203505211133e-08, 9.353680499788552e-06, -0.0009817356348626736, 0.04459313654596568])),
                 VolumeLiquid(poly_fit=(195.505, 395.4, [5.103835649289192e-22, -1.1451165900302792e-18, 1.1154748069727923e-15, -6.160223429581022e-13, 2.1091176437184486e-10, -4.583765753427699e-08, 6.175086699295002e-06, -0.00047144103245914243, 0.015638233047208582])),
                 VolumeLiquid(poly_fit=(63.2, 116.192, [9.50261462694019e-19, -6.351064785670885e-16, 1.8491415360234833e-13, -3.061531642102745e-11, 3.151588109585604e-09, -2.0650965261816766e-07, 8.411110954342014e-06, -0.00019458305886755787, 0.0019857193167955463])),
                 VolumeLiquid(poly_fit=(54.370999999999995, 144.58100000000002, [6.457909929992152e-20, -4.7825644162085234e-17, 1.5319533644419177e-14, -2.7692511820542383e-12, 3.088256295705142e-10, -2.1749171236451626e-08, 9.448300475893009e-07, -2.3081894336450133e-05, 0.00026558114294435354])),
                 VolumeLiquid(poly_fit=(175.7, 502.5, [3.5725079384600736e-23, -9.031033742820083e-20, 9.819637959370411e-17, -5.993173551565636e-14, 2.2442465416964825e-11, -5.27776114586072e-09, 7.610461006178106e-07, -6.148574498547711e-05, 0.00216398089328537])),
                 VolumeLiquid(poly_fit=(309.58, 729.5999999999999, [7.45473064887452e-24, -2.89457102830013e-20, 4.867041417017972e-17, -4.6252122183399004e-14, 2.7157887108452537e-11, -1.0085443480134824e-08, 2.3130153268044497e-06, -0.0002992756488164552, 0.01705648133237398])),
                 VolumeLiquid(poly_fit=(13.967, 29.3074, [1.338998655322118e-14, -2.2300738749278554e-12, 1.6125645123435388e-10, -6.603182508985557e-09, 1.6732222054898376e-07, -2.6846339878160216e-06, 2.6629092007736217e-05, -0.0001490816582989168, 0.0003852732680036591]))]

correlations = PropertyCorrelationsPackage(constants=constants, skip_missing=True, VolumeLiquids=VolumeLiquids, VaporPressures=VaporPressures, HeatCapacityGases=HeatCapacityGases)




pure_surfaces_dir = os.path.join(thermo.thermo_dir, '..', 'surfaces', 'virial')

@pytest.mark.plot
@pytest.mark.slow
@pytest.mark.parametric
@pytest.mark.parametrize("fluid", pure_fluids)
@pytest.mark.parametrize("variables", ['VPT',
                                       #'VTP',
                                       # 'PHT', 'PST', 'PUT',
                                       # 'VUT', 'VST', 'VHT',
                                          # 'TSV',
                                            # 'THP', 'TUP',
                                       ])
def test_plot_fluid_virial_pure(fluid, variables):
    spec0, spec1, check_prop = variables
    plot_name = variables[0:2]
    T, P = 298.15, 101325.0
    zs = [1.0]
    fluid_idx = pure_fluids.index(fluid)

    pure_const, pure_props = constants.subset([fluid_idx]), correlations.subset([fluid_idx])


    model = VirialCSP(Tcs=pure_const.Tcs, Pcs=pure_const.Pcs, Vcs=pure_const.Vcs, omegas=pure_const.omegas,
                      B_model=VIRIAL_B_PITZER_CURL, C_model=VIRIAL_C_ORBEY_VERA)
    gas = VirialGas(model, HeatCapacityGases=pure_props.HeatCapacityGases, T=T, P=P, zs=zs)



    flasher = FlashPureVLS(constants=pure_const, correlations=pure_props,
                       gas=gas, liquids=[], solids=[])
    # print(repr(gas))
    # print(pure_const)
    # print(pure_props)
    # 1/0
    # flasher.TPV_HSGUA_xtol = 1e-14

    inconsistent = frozenset([spec0, spec1]) in (frozenset(['T', 'H']), frozenset(['T', 'U']))

    res = flasher.TPV_inputs(zs=[1.0], pts=200, spec0='T', spec1='P',
                             check0=spec0, check1=spec1, prop0=check_prop,
                           trunc_err_low=1e-13,
                           trunc_err_high=1, color_map=cm_flash_tol(),
                           show=False, verbose=not inconsistent)

    matrix_spec_flashes, matrix_flashes, errs, plot_fig = res

    path = os.path.join(pure_surfaces_dir, fluid, plot_name)
    if not os.path.exists(path):
        os.makedirs(path)

    tol = 1e-13

    key = f'{plot_name} - {eos.__name__} - {fluid}'

    if inconsistent:
        spec_name = spec0 + spec1
        mark_plot_unsupported(plot_fig, reason='EOS is inconsistent for %s inputs' %(spec_name))
        tol = 1e300

    plot_fig.savefig(os.path.join(path, key + '.png'))
    plt.close()

    max_err = np.max(np.abs(errs))
    assert max_err < tol

# test_plot_fluid_virial_pure('water', 'VPT')
# Not sure where to go from here
# Definitely not as continuous as desired!
del test_plot_fluid_virial_pure
