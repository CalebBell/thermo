  # -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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


.. warning::
    These classes were a first attempt at rigorous multiphase equilibrium.
    They may be useful in some special cases but they are not complete and
    further development will not happen. They were never documented as well.

    It is recommended to switch over to the :obj:`thermo.flash` interface
    which seeks to be more modular, easier to maintain and extend,
    higher-performance, and easier to modify.

'''

from __future__ import division

__all__ = ['PropertyPackage', 'Ideal', 'Unifac', 'GammaPhi',
           'UnifacDortmund', 'IdealCaloric', 'GammaPhiCaloric',
           'UnifacCaloric', 'UnifacDortmundCaloric', 'Nrtl', 'WilsonPP',
           'StabilityTester',
           'eos_Z_test_phase_stability', 'eos_Z_trial_phase_stability',
           'Stateva_Tsvetkov_TPDF_eos', 'd_TPD_Michelson_modified_eos',
           'GceosBase']

try:
    from random import uniform, shuffle, seed
except:
    pass
from fluids.numerics import (OscillationError, UnconvergedError,
                             ridder, derivative, caching_decorator,
                             newton, linspace, logspace,
                             brenth, py_solve,
                             oscillation_checker, secant, damping_maintain_sign,
                             oscillation_checking_wrapper, numpy as np)
from fluids.constants import R, pi, N_A
from chemicals.utils import log, log10, exp, copysign, dxs_to_dn_partials, dxs_to_dns, dns_to_dn_partials, d2xs_to_dxdn_partials, remove_zeros, normalize, Cp_minus_Cv, mixing_simple, property_mass_to_molar
from thermo.utils import has_matplotlib
from chemicals.elements import mixture_atomic_composition, similarity_variable
from chemicals.identifiers import IDs_to_CASs
from chemicals.rachford_rice import flash_inner_loop, Rachford_Rice_solution2
from chemicals.flash_basic import K_value, Wilson_K_value, flash_wilson, flash_Tb_Tc_Pc, flash_ideal
from thermo.wilson import Wilson_gammas as Wilson
from thermo.nrtl import NRTL_gammas
from chemicals.rachford_rice import Rachford_Rice_flash_error
from thermo.unifac import UNIFAC_gammas, UFSG, DOUFSG
from thermo import unifac
from thermo.eos_mix import *
from thermo.eos import *
from chemicals.heat_capacity import Lastovka_Shaw_T_for_Hm, Lastovka_Shaw_T_for_Sm, Dadgostar_Shaw_integral_over_T, Dadgostar_Shaw_integral, Lastovka_Shaw_integral
from thermo.phase_change import SMK


DIRECT_1P = 'Direct 1 Phase'
DIRECT_2P = 'Direct 2 Phase'
RIGOROUS_BISECTION = 'Bisection'
CAS_H2O = '7732-18-5'

def Rachford_Rice_solution_negative(zs, Ks):
    try:
        return flash_inner_loop(zs, Ks)
    except:
        pass
    # Only here for backwards compatibility
    # Works when component compositions go negative.


    Kmin = min(Ks)
    Kmax = max(Ks)
    z_of_Kmax = zs[Ks.index(Kmax)]

    V_over_F_min = ((Kmax-Kmin)*z_of_Kmax - (1.-Kmin))/((1.-Kmin)*(Kmax-1.))
    V_over_F_max = 1./(1.-Kmin)

    V_over_F_min2 = max(0., V_over_F_min)
    V_over_F_max2 = min(1., V_over_F_max)

    x0 = (V_over_F_min2 + V_over_F_max2)*0.5
    try:
        # Newton's method is marginally faster than brenth
        V_over_F = secant(Rachford_Rice_flash_error, x0, args=(zs, Ks))
        # newton skips out of its specified range in some cases, finding another solution
        # Check for that with asserts, and use brenth if it did
        assert V_over_F >= V_over_F_min2
        assert V_over_F <= V_over_F_max2
    except:
        V_over_F = brenth(Rachford_Rice_flash_error, V_over_F_max-1E-7, V_over_F_min+1E-7, args=(zs, Ks))
    # Cases not covered by the above solvers: When all components have K > 1, or all have K < 1
    # Should get a solution for all other cases.
    xs = [zi/(1.+V_over_F*(Ki-1.)) for zi, Ki in zip(zs, Ks)]
    ys = [Ki*xi for xi, Ki in zip(xs, Ks)]
    return V_over_F, xs, ys

#from random import uniform, seed
#seed(0)
#print([uniform(0, 1) for _ in range(1000)])
# 250 values between 0 and 1.
random_values = [0.8444218515250481, 0.7579544029403025, 0.420571580830845, 0.25891675029296335, 0.5112747213686085, 0.4049341374504143, 0.7837985890347726,
    0.30331272607892745, 0.4765969541523558, 0.5833820394550312, 0.9081128851953352, 0.5046868558173903, 0.28183784439970383, 0.7558042041572239,
    0.6183689966753316, 0.25050634136244054, 0.9097462559682401, 0.9827854760376531, 0.8102172359965896, 0.9021659504395827, 0.3101475693193326,
    0.7298317482601286, 0.8988382879679935, 0.6839839319154413, 0.47214271545271336, 0.1007012080683658, 0.4341718354537837, 0.6108869734438016,
    0.9130110532378982, 0.9666063677707588, 0.47700977655271704, 0.8653099277716401, 0.2604923103919594, 0.8050278270130223, 0.5486993038355893,
    0.014041700164018955, 0.7197046864039541, 0.39882354222426875, 0.824844977148233, 0.6681532012318508, 0.0011428193144282783, 0.49357786646532464,
    0.8676027754927809, 0.24391087688713198, 0.32520436274739006, 0.8704712321086546, 0.19106709150239054, 0.5675107406206719, 0.23861592861522019,
    0.9675402502901433, 0.80317946927987, 0.44796957143557037, 0.08044581855253541, 0.32005460467254576, 0.5079406425205739, 0.9328338242269067,
    0.10905784593110368, 0.5512672460905512, 0.7065614098668896, 0.5474409113284238, 0.814466863291336, 0.540283606970324, 0.9638385459738009,
    0.603185627961383, 0.5876170641754364, 0.4449890262755162, 0.5962868615831063, 0.38490114597266045, 0.5756510141648885, 0.290329502402758,
    0.18939132855435614, 0.1867295282555551, 0.6127731798686067, 0.6566593889896288, 0.47653099200938076, 0.08982436119559367, 0.7576039219664368,
    0.8767703708227748, 0.9233810159462806, 0.8424602231401824, 0.898173121357879, 0.9230824398201768, 0.5405999249480544, 0.3912960502346249,
    0.7052833998544062, 0.27563412131212717, 0.8116287085078785, 0.8494859651863671, 0.8950389674266752, 0.5898011835311598, 0.9497648732321206,
    0.5796950107456059, 0.4505631066311552, 0.660245378622389, 0.9962578393535727, 0.9169412179474561, 0.7933250841302242, 0.0823729881966474,
    0.6127831050407122, 0.4864442019691668, 0.6301473404114728, 0.8450775756715152, 0.24303562206185625, 0.7314892207908478, 0.11713429320851798,
    0.22046053686782852, 0.7945829717105759, 0.33253614921965546, 0.8159130965336595, 0.1006075202160962, 0.14635848891230385, 0.6976706401912388,
    0.04523406786561235, 0.5738660367891669, 0.9100160146990397, 0.534197968260724, 0.6805891325622565, 0.026696794662205203, 0.6349999099114583,
    0.6063384177542189, 0.5759529480315407, 0.3912094093228269, 0.3701399403351875, 0.9805166506472687, 0.036392037611485795, 0.021636509855024078,
    0.9610312802396112, 0.18497194139743833, 0.12389516442443171, 0.21057650988664645, 0.8007465903541809, 0.9369691586445807, 0.022782575668658378,
    0.42561883196681716, 0.10150021937416975, 0.259919889792832, 0.22082927131631735, 0.6469257198353225, 0.3502939673965323, 0.18031790152968785,
    0.5036365052098872, 0.03937870708469238, 0.10092124118896661, 0.9882351487225011, 0.19935579046706298, 0.35855530131160185, 0.7315983062253606,
    0.8383265651934163, 0.9184820619953314, 0.16942460609746768, 0.6726405635730526, 0.9665489030431832, 0.05805094382649867, 0.6762017842993783,
    0.8454245937016164, 0.342312541078584, 0.25068733928511167, 0.596791393469411, 0.44231403369907896, 0.17481948445144113, 0.47162541509628797,
    0.40990539565755457, 0.5691127395242802, 0.5086001300626332, 0.3114460010002068, 0.35715168259026286, 0.837661174368979, 0.25093266482213705,
    0.560600218853524, 0.012436318829314397, 0.7415743774106636, 0.3359165544734606, 0.04569649356841665, 0.28088316421834825, 0.24013040782635398,
    0.9531293398277989, 0.35222556151550743, 0.2878779148564, 0.35920119725374633, 0.9469058356578911, 0.6337478522492526, 0.6210768456186673,
    0.7156193503014563, 0.38801723531250565, 0.4144179882772473, 0.650832862263345, 0.001524221856720187, 0.1923095412446758, 0.3344016906625016,
    0.23941596018595857, 0.6373994011293003, 0.37864807032309444, 0.8754233917130172, 0.5681514209101919, 0.4144063966836443, 0.40226707511907955,
    0.7018296239336754, 0.41822655329246605, 0.6621958889738174, 0.04677968595679827, 0.44535218971882984, 0.25922692344722276, 0.15768657212231085,
    0.5275731301676146, 0.48726560106903205, 0.5614049256144269, 0.7554847672586825, 0.8838751542487009, 0.4945826703752868, 0.31205824641687296,
    0.46689223535252355, 0.8090458573603624, 0.8750163314802711, 0.8124149323637591, 0.188001294050828, 0.9994203594553304, 0.6330887599183004,
    0.08346705017572931, 0.7255543554613124, 0.9868214802051282, 0.40181682221254356, 0.6785150052419683, 0.31617713722134233, 0.2135246620646961,
    0.7173241433110372, 0.0023575647193538884, 0.8227314105314157, 0.5283459768597928, 0.09778434180065931, 0.11890389478474583, 0.6492654248961536,
    0.8736538239003423, 0.27998274332687256, 0.9785151867733981, 0.10018068906370903, 0.8539381095973382, 0.39669617733090445, 0.08134541676823415,
    0.2747138434192621, 0.4529781848179143, 0.7923415311856522, 0.8613599036372361, 0.13342055420254906, 0.5208655284141989, 0.6507832381497373,
    0.3470530145996015, 0.8718638357105861, 0.27840981521636055, 0.01857432754559518, 0.0406632736752609, 0.6809967701112433, 0.5583557360970469,
    0.946502554169996, 0.9384387997349186, 0.9098511774051025, 0.04200453196734122, 0.7491348233908631, 0.7013248175948597, 0.6553618646747296,
    0.7123576525162417, 0.9027101506193307, 0.6401411997932241, 0.372449262972256, 0.5379287837318205, 0.20784410369082473, 0.5871255046951435,
    0.008897082049078797, 0.15102317386398778, 0.3334083880298664, 0.7896231589257826, 0.7184994227715396, 0.3382559700266786, 0.6205381083165517,
    0.041202949506209285, 0.16386054567557595, 0.9819140701253054, 0.28953085363586695, 0.39479198298829066, 0.5484842965725134, 0.29340700145733656,
    0.47806466915102097, 0.2397060836386239, 0.04825636228829444, 0.17958684904155564, 0.5230502317000981, 0.07086288409434749, 0.4031691464450935,
    0.3285207100154869, 0.4147216089714424, 0.09940033823870109, 0.9086575543967805, 0.4740046511372964, 0.8408483326276716, 0.976229457649057,
    0.34365159365776776, 0.4790865191519861, 0.6995952911506185, 0.42653532354402823, 0.30190311621935595, 0.7347509912186152, 0.8943997782145745,
    0.9196888444316101, 0.6267420468068673, 0.3755713463285453, 0.9745605214796941, 0.6388785175004733, 0.06583467727730097, 0.08466956912011114,
    0.749869571783086, 0.06115615654596607, 0.007851005331251826, 0.39380795178170946, 0.5190037287013293, 0.44854428559655457, 0.48861880442715255,
    0.5848887019932744, 0.6793025673721249, 0.4230380735074225, 0.3683314563344259, 0.9884590580992895, 0.26091653544625626, 0.7771001545085096,
    0.43122102463204415, 0.35852038200953895, 0.06385794894382868, 0.8635789443020424, 0.7020041497619371, 0.9030107075409272, 0.4516117926868677,
    0.6769209668166035, 0.11891028655385572, 0.3979536016023134, 0.20723197341708288, 0.04210142789066196, 0.94796135125632, 0.21589436846535714,
    0.1463544898080057, 0.19797004355794223, 0.37803196431429753, 0.5463912623151137, 0.15133436847289106, 0.9886898889857565, 0.9829892105452821,
    0.14840201708602985, 0.4059068831679489, 0.6799294831100022, 0.8776565829010952, 0.49540592491118873, 0.9170466727598151, 0.3224603148813061,
    0.4984408914907503, 0.4986465918650089, 0.6700681513152942, 0.2019913087994536, 0.6097706104167804, 0.21877309687215574, 0.340220315051032,
    0.9625664632546818, 0.8990080380310076, 0.8181183809177941, 0.035468261876012264, 0.14836688246192975, 0.2568819120719038, 0.7841665681891542,
    0.8423333270773672, 0.5829481802462215, 0.7181316517768294, 0.8070553799750758, 0.06635913103778524, 0.08464313683307012, 0.8688953140043785,
    0.03941582937802879, 0.22509065367649606, 0.04063202664590093, 0.015285139969726802, 0.8439546856924078, 0.3305943672500803, 0.1606900602627206,
    0.1488194902889095, 0.656083661770337, 0.9685982716927071, 0.5049996926056783, 0.9010904768840049, 0.5024285989524275, 0.5738724774915492,
    0.6785713567893591, 0.805109989032137, 0.7578463822613826, 0.9905325627055622, 0.7469653891501328, 0.9057807233528663, 0.20610483206558328,
    0.535416304328581, 0.5986142636674691, 0.8256966171603538, 0.4822135630659161, 0.7910402117090956, 0.3885688901501142, 0.5863884555814496,
    0.8513166074810679, 0.7980594711041583, 0.6569845518861341, 0.00024069652516689466, 0.18196892218621108, 0.5068577868511277, 0.2544593984833793,
    0.06562084327273077, 0.8598834221214616, 0.9429470213131631, 0.3028048781490337, 0.40807316738486077, 0.8100375338172869, 0.06225875887122312,
    0.6409848625624502, 0.12732081293278708, 0.2870883399952252, 0.829940686628406, 0.0555270458896614, 0.035933833430188966, 0.4178660447962945,
    0.49183095909626395, 0.8633251831082008, 0.7171887463451895, 0.6735438085995347, 0.15137377239978678, 0.9867059242186832, 0.41114019628748133,
    0.6117708643248599, 0.38668300553323576, 0.04703291581184044, 0.4708892090480652, 0.15136775389483625, 0.03246546237394399, 0.6174004236810055,
    0.6299662912183356, 0.10529282465636491, 0.5491437662317772, 0.3466679766399683, 0.3834140731648874, 0.7764198986996783, 0.49031967752424566,
    0.8812766154122413, 0.6101197429062234, 0.4671884150380703, 0.6323126400553846, 0.3378653798287524, 0.12432379252825243, 0.6825296186925238,
    0.622037442746657, 0.7885664913738635, 0.1271091249471088, 0.9117833181295222, 0.799341211421814, 0.9168874080910093, 0.8725347217734669, 0.681006446357057,
    0.8102508494373589, 0.5190073092314018, 0.7854891493606652, 0.18912746785718504, 0.7821141063572942, 0.44457960405634067, 0.756616221297365,
    0.4554702368121878, 0.7895587282777832, 0.07533958521856021, 0.04464090542441246, 0.9342895823715677, 0.4861651007487351, 0.9010713996489047,
    0.9447832518820701, 0.6665111524556335, 0.5717968260934746, 0.21597938410680917, 0.09347621929900818, 0.8193942150822732, 0.8887720676319878,
    0.7793957106948857, 0.6985024327316249, 0.42011111607482077, 0.3053115900269564, 0.11344489563770899, 0.425970248072163, 0.5660129742477574,
    0.9228805831375125, 0.9357547693309531, 0.41564119654091314, 0.0992109880980957, 0.7738187324714434, 0.7342793416571158, 0.03070084595190614,
    0.4467185991338365, 0.6864181042985581, 0.030134234552269934, 0.9192823534016137, 0.9622424865104192, 0.72254277208884, 0.0785385396518038,
    0.07032946587635569, 0.3592533148212369, 0.029377507756986443, 0.3478777272843395, 0.009964241312966027, 0.9743235128409679, 0.8190066990688627,
    0.07051761147818736, 0.8934350918478603, 0.20797804000401565, 0.20479079826934998, 0.6737591455288341, 0.9382622681625481, 0.12318812122923739,
    0.007184567252270457, 0.3691301471700257, 0.024650014436155776, 0.6048482375805311, 0.8591756086192088, 0.1869917024228578, 0.11239103583018406,
    0.34444960733861085, 0.9591715206073138, 0.13015769442868408, 0.9665192604669938, 0.36223986994484925, 0.47337040276011155, 0.29263198596497353,
    0.9371268442154698, 0.9581478949874975, 0.6359157065077434, 0.18404555017515556, 0.9929517886102871, 0.10258043954691198, 0.5808493815940804,
    0.15640306008300875, 0.8976753141502056, 0.9456783914956152, 0.8043902980001079, 0.3158914186681244, 0.2428386899579852, 0.7548584132190378,
    0.291059519145354, 0.4197853778540753, 0.04625567690264132, 0.13223381043380655, 0.020549620641776678, 0.0779211200935358, 0.07321114936486084,
    0.42023170217414685, 0.5507771776374378, 0.740878819870922, 0.14228347384241602, 0.4221887461694188, 0.6369660374117204, 0.08455569481893255,
    0.44481115514620384, 0.3692560392397978, 0.9489319289416618, 0.05785711390101722, 0.40862622118314806, 0.41722547979620506, 0.728180504599678,
    0.3206710028745039, 0.20399027594623398, 0.2933116551663051, 0.4708875424493587, 0.9502683295716211, 0.7965170227633064, 0.2769702457797433,
    0.5581815883930463, 0.6882003035685332, 0.7956571556821322, 0.4461643839498476, 0.398776905129706, 0.7676407428212785, 0.43171649556411207,
    0.2479576688970051, 0.4534470315306477, 0.9371046462904561, 0.14256748821860132, 0.4624353545272121, 0.6373035243637815, 0.48328798826810027,
    0.20363990437036994, 0.0018431606156659175, 0.698991711803439, 0.6187355180234525, 0.007776649435864202, 0.2985601210181208, 0.7686342595428415,
    0.6289203785446209, 0.5452081159439722, 0.1562211098090489, 0.7062940429996885, 0.4714349217158037, 0.6781787462359636, 0.7600898367234922,
    0.23236272144124515, 0.7619950130977117, 0.28008838468838926, 0.9840151371182455, 0.12083161078451865, 0.8837180187440564, 0.040547125043371324,
    0.256575818348144, 0.5261019087624684, 0.5816161834445946, 0.3962349850280922, 0.10203172822707107, 0.2526080858247133, 0.28339650386048865,
    0.7552228545587315, 0.9087743252220071, 0.5954099154864194, 0.03545096569102746, 0.7922364716417103, 0.30560393283991993, 0.33989040641624346,
    0.5301854376454147, 0.24904704757555507, 0.9199780878573697, 0.1635547583408129, 0.41483040050373277, 0.2896919495072058, 0.5198341022016146,
    0.5739818030823766, 0.6271396891048426, 0.5313758038379728, 0.4108045023355995, 0.634594012376466, 0.40341287658681757, 0.7785502590540477,
    0.7881774252549901, 0.29225416811082217, 0.37180432355577453, 0.6288109059468862, 0.15706996711565713, 0.6970319309869248, 0.3814277529807131,
    0.591062474757007, 0.1395330992312218, 0.6682583860975598, 0.3540578606136997, 0.4726655762072315, 0.4151074008495357, 0.47671524799509457,
    0.6946956329164442, 0.31824017683207795, 0.6520544808985483, 0.060222107499701916, 0.3001851524622099, 0.7452096901500458, 0.05240587806206365,
    0.6211421952822352, 0.025546799267838538, 0.4715288683099005, 0.8885450437134765, 0.010110093997603875, 0.5268280206539229, 0.06645682965886301,
    0.8671097761494883, 0.6862965222396646, 0.7419538566814291, 0.669007579945888, 0.006423453698145676, 0.041177862257898545, 0.6208768040220466,
    0.9996851255769114, 0.8731472390917929, 0.699685806725371, 0.7270999543422898, 0.2266870226016624, 0.751613934135812, 0.28792410486343756,
    0.10546026702239297, 0.4608948954667579, 0.33019577252961807, 0.168255398651179, 0.42170989251140467, 0.8972009769638755, 0.4352702732981688,
    0.4472918952497248, 0.708827757444238, 0.5241618701522923, 0.12922303534199353, 0.91039239754397, 0.4441243361619651, 0.7893377392253591,
    0.38887513002224416, 0.806846018820692, 0.3895364160074527, 0.2201595216660458, 0.19619466691666865, 0.9400346443375104, 0.58653025858102,
    0.04979326505826487, 0.38834759617804915, 0.234029260524927, 0.08465706460929934, 0.18675586852140846, 0.05699047999950346, 0.6380736282281027,
    0.17337386483746886, 0.6107798762435255, 0.6125067478912297, 0.7049237107399368, 0.5121186506114312, 0.28442399033479826, 0.8774574539285279,
    0.35307108172351365, 0.4582943249787391, 0.6318794317305464, 0.5161242981674495, 0.9564683485665337, 0.9547176774381221, 0.9297598506094263,
    0.9340763496652581, 0.580960135568696, 0.49020206373000297, 0.7041168173823689, 0.21541959298546798, 0.26587203921552827, 0.04380725363309168,
    0.16285754255803098, 0.0038745499388105342, 0.6546275765234981, 0.14040698903568194, 0.7866793455760521, 0.680503995881725, 0.9706757933544957,
    0.3965144869518913, 0.9213919134510528, 0.4537041723195332, 0.3395037398362071, 0.10233886991705377, 0.8828321850718597, 0.7947901585625868,
    0.3229289765350606, 0.45574438492562896, 0.32514346581324827, 0.028829116538094723, 0.04435252539911694, 0.3687041258820589, 0.20959132812878367,
    0.5245146032105923, 0.1877850356496189, 0.2016215864664097, 0.6726678813176303, 0.7356026567617159, 0.31223209587410494, 0.8599943994333726,
    0.2546391746557106, 0.34394037628155716, 0.712480390369609, 0.04450290132920964, 0.934183460116191, 0.07233773178762537, 0.4609310589380602,
    0.7246048259600892, 0.04746853498479808, 0.8090026856371774, 0.9788933433114139, 0.460511672795628, 0.11812363628756806, 0.08147699565547994,
    0.09873043616313526, 0.7654413741364753, 0.4140128484685186, 0.9192341581990311, 0.4406397760864845, 0.07714331014460807, 0.42693558751800065,
    0.7548278934255565, 0.8293384268467949, 0.039351686529191854, 0.1803893912563338, 0.490013452023644, 0.12808547795160863, 0.8710926419421733,
    0.9344608884461488, 0.3195969983538176, 0.43484368255202, 0.5570540644200566, 0.2855057910835891, 0.5410756974595614, 0.2011850454737838,
    0.2966412512769129, 0.44178363318767744, 0.604669902191143, 0.5361650260862432, 0.2609879767339395, 0.23178787541805523, 0.11873023670071103,
    0.7834936358921726, 0.09890076646638046, 0.7328850061793606, 0.2487736956630997, 0.28455698400578255, 0.7360834330107994, 0.6596207917216363,
    0.7419215555155583, 0.5152830587943614, 0.8590958196652707, 0.12179389137547159, 0.6451969614065052, 0.11824431248865597, 0.7372833681454282,
    0.3589046614584527, 0.67488210437111, 0.7034839134412817, 0.6606084576410584, 0.22155798032782648, 0.8317998863873537, 0.24013608742346748,
    0.5181532972121122, 0.6746457541533513, 0.23360317478475656, 0.628511722983939, 0.2868310479973286, 0.1713823760843869, 0.809748828526577,
    0.5531227700773604, 0.32788470660885605, 0.5854309472055399, 0.025286397427288332, 0.12982285676032723, 0.3955808516982431, 0.9757565794644123,
    0.5104745178761232, 0.07645620506689521, 0.7650406152494567, 0.7814438709253152, 0.7748021743948562, 0.5694980380479538, 0.6956987378694627,
    0.21345793631163135, 0.7325605908939883, 0.8161739873415944, 0.7599665402219192, 0.353462402585887, 0.5910280505757086, 0.6289893574898388,
    0.9008098536570839, 0.1080138952733335, 0.8339337708504084, 0.5264355584690392, 0.3586141205519373, 0.4556029014937524, 0.012635498930738787,
    0.22007359233142765, 0.6527634200680049, 0.660849279754449, 0.4946989402863131, 0.9533258805973196, 0.4809150885494712, 0.3139436595456605,
    0.8477808391956414, 0.259158299397262, 0.6043059930343495, 0.7034188523223, 0.8216962986917842, 0.7853687501827489, 0.3840923305137113,
    0.059180305962736934, 0.03828786548344276, 0.7264603879084595, 0.9616913814068508, 0.3431653742712939, 0.44119509807551416, 0.7257980157417766,
    0.6578312458538799, 0.26010658848413604, 0.6715848457987025, 0.3049024195743838, 0.3563579065620385, 0.5395133052630944, 0.7323138239267305,
    0.15121621156796483, 0.021987210892938758, 0.6278299544850219, 0.024564677785836264, 0.04496324071616853, 0.22577557672213355, 0.6538768733044555,
    0.06654509768602879, 0.06240576762652772, 0.9720932443736168, 0.4226528937805498, 0.8924289339928592, 0.21652428395276402, 0.4352131794546169,
    0.35803513461315506, 0.17693553603496914, 0.32881318575191665, 0.9867958186960467, 0.7473090097951195, 0.3826682791831585, 0.40928443439993156,
    0.2637409011550663, 0.531336678598825, 0.7356369121419466, 0.686646615750601, 0.46264983534131954, 0.041939046716157, 0.9215078064992686,
    0.4089338030960661, 0.3902988670119316, 0.0031101144891549914, 0.13822721408191307, 0.8688534175006787, 0.513934596181303, 0.7324348442226767,
    0.14816788643335854, 0.33005100665524945, 0.8401365565378639, 0.8206585211774247, 0.2467942680862406, 0.021975308333072263, 0.8064669735456029,
    0.16884400503942165, 0.7876813921208954, 0.6836592298851071, 0.1683147603108942, 0.0784886436699127, 0.9276494299222889, 0.5978783972833935,
    0.620510173056511, 0.4575118028380537, 0.15007097732228858, 0.6019699129465877, 0.2524728800375037, 0.8058946560175415, 0.732718954805416,
    0.027267185045511733, 0.9324230096450348, 0.03631604832667812, 0.0896193188307074, 0.2927345609042453, 0.1508090604701401, 0.2361450829166024,
    0.3558094886115547, 0.7354997154547138, 0.4047113607648444, 0.2698397547254259, 0.4923131536276696, 0.39259324978876053, 0.310764197486207,
    0.900541657866744, 0.5504484509596044, 0.9773275109747672, 0.7729124093934382, 0.570499297619577, 0.26244658927686404, 0.6868436562888387,
    0.45591771896977173, 0.7213877150417534, 0.40377880891106155, 0.49600503631794757, 0.02068376744575562, 0.739958502320053, 0.03427354435563068,
    0.6807253858476396, 0.5820036955379622, 0.7759176114881267, 0.28977759923741564, 0.6861108151233298, 0.20709797563103816, 0.5292720013578311,
    0.34028037925118015, 0.9784545513570129, 0.9718665573793185, 0.20896973547336006, 0.5660382358858294, 0.3294426858782725, 0.9685381870202809,
    0.9245259481865659, 0.5861458530564896, 0.7200844551084937, 0.6813247567090696, 0.353355632443361, 0.91636156937516, 0.899453536816357, 0.33065846447807934,
    0.7473949106043586, 0.009092126674448586, 0.8163591105584419, 0.5648693453979996, 0.9523067127509502, 0.3631930745481745, 0.6257130749033707,
    0.3230024315033787, 0.7827853814039997, 0.6007029967830003, 0.9874710229786893, 0.0010127930964535237, 0.14075874215813544, 0.043601382090813434,
    0.1258478488128345, 0.9293852970698306, 0.9486082995058949, 0.4804125346981437
]


def Stateva_Tsvetkov_TPDF_eos(eos):
    Z_eos, prefer, alt = eos_Z_test_phase_stability(eos)

    def obj_func_constrained(zs):
        # zs is N -1 length
        zs_trial = [abs(float(i)) for i in zs]
        if sum(zs_trial) >= 1:
            zs_trial = normalize(zs_trial)

        # In some cases, 1 - x < 0
        zs_trial.append(abs(1.0 - sum(zs_trial)))

        eos2 = eos.to_TP_zs(T=eos.T, P=eos.P, zs=zs_trial)
        Z_trial = eos_Z_trial_phase_stability(eos2, prefer, alt)
        TPD = eos.Stateva_Tsvetkov_TPDF(Z_eos, Z_trial, eos.zs, zs_trial)
        return TPD
    return obj_func_constrained


def d_TPD_Michelson_modified_eos(eos):
    Z_eos, prefer, alt = eos_Z_test_phase_stability(eos)

    def obj_func_unconstrained(alphas):
        # zs is N -1 length
        Ys = [(alpha/2.)**2 for alpha in alphas]
        zs_trial = normalize(Ys)

        eos2 = eos.to_TP_zs(T=eos.T, P=eos.P, zs=zs_trial)
        Z_trial = eos_Z_trial_phase_stability(eos2, prefer, alt)
        TPD = eos.d_TPD_Michelson_modified(Z_eos, Z_trial, eos.zs, alphas)
        return TPD
    return obj_func_unconstrained


class StabilityTester(object):

    def __init__(self, Tcs, Pcs, omegas, aqueous_check=False, CASs=None):
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.N = len(Tcs)
        self.cmps = range(self.N)
        self.aqueous_check = aqueous_check
        self.CASs = CASs

        try:
            self.water_index = CASs.index(CAS_H2O)
        except:
            self.water_index = None

    def set_d_TPD_obj_unconstrained(self, f, T, P, zs):
        self.f_unconstrained = f
        self.T = T
        self.P = P
        self.zs = zs

    def set_d_TPD_obj_constrained(self, f, T, P, zs):
        self.f_constrained = f
        self.T = T
        self.P = P
        self.zs = zs

    def stationary_points_unconstrained(self, random=True, guesses=None, raw_guesses=None,
                                        fmin=1e-7, tol=1e-12, method='Nelder-Mead'):
        if not raw_guesses:
            raw_guesses = []
        if not guesses:
            guesses = self.guesses(T=self.T, P=self.P, zs=self.zs, random=random)
        results = []
        results2 = []
        for guesses, convert in zip((raw_guesses, guesses), (False, True)):
            for guess in guesses:
                if convert:
                # Convert the guess to a basis squared
                    guess = [i**0.5*2.0 for i in guess]
                from scipy.optimize import minimize
                ans = minimize(self.f_unconstrained, guess, method=method, tol=tol)
                # Convert the answer to a normal basis
                Ys = [(alpha/2.)**2 for alpha in ans['x']]
                ys = normalize(Ys)
                if ans['fun'] <= fmin:
                    results.append(ys)
                results2.append(ans)
        return results, results2


    def stationary_points_constrained(self, random=True, guesses=None,
                                      fmin=1e-7, iter=1000, tol=1e-12, method='fmin_slsqp'):
        from scipy.optimize import fmin_slsqp
        if not guesses:
            guesses = self.guesses(T=self.T, P=self.P, zs=self.zs, random=random)
        results = []
        def f_ieqcons(guess):
            return 1.0 - sum(guess)

        arr = -np.ones((len(guesses[0]) - 1))
        def fprime_ieqcons(guess):
            return arr
#            return [[0.0]*len(guess)]
#            return np.ones([1, len(guess)])

        for guess in guesses:

            ans, err, _, _, _ = fmin_slsqp(self.f_constrained, x0=guess[0:-1], f_ieqcons=f_ieqcons,
                                          acc=tol, full_output=True, disp=False,
                                          fprime_ieqcons=fprime_ieqcons)
            # Convert the answer to a normal basis
            zs = np.abs(ans).tolist()
            zs.append(1.0 - sum(zs))
            if err <= fmin:
                results.append(zs)
        return results


    def random_guesses(self, N=None):
        if N is None:
            N = self.N
        seed(0)
        random_guesses = [normalize([uniform(0, 1) for _ in range(N)])
                          for k in range(N)]
        return random_guesses

    def pure_guesses(self, zero_fraction=1E-6):
        pure_guesses = [normalize([zero_fraction if j != k else 1 for j in self.cmps])
                       for k in self.cmps]
        return pure_guesses

    def Wilson_guesses(self, T, P, zs, powers=(1, -1, 1/3., -1/3.)): #
        # First K is vapor-like phase; second, liquid like
        Ks_Wilson = [Wilson_K_value(T=T, P=P, Tc=self.Tcs[i], Pc=self.Pcs[i], omega=self.omegas[i]) for i in self.cmps]
        Wilson_guesses = []
        for power in powers:
            Ys_Wilson = [Ki**power*zi for Ki, zi in zip(Ks_Wilson, zs)]
            Wilson_guesses.append(normalize(Ys_Wilson))
#            print(Ys_Wilson, normalize(Ys_Wilson))
        return Wilson_guesses

    def incipient_guess_name(self, idx, expect_liquid=False,
                             expect_aqueous=False, existing_phases=0):
        if idx < 4:
            if not expect_liquid:
                return ('Wilson gas', 'Wilson liquid', 'Wilson gas third', 'Wilson liquid third')[idx]
            else:
                if expect_aqueous:
                    if idx == 0:
                        return 'pure%d' %(self.water_index)
                    return ('Wilson liquid',  'Wilson liquid third', 'Wilson gas third', 'Wilson gas')[idx-1]
                else:
                    return ('Wilson liquid',  'Wilson liquid third', 'Wilson gas third', 'Wilson gas')[idx]
        if expect_aqueous:
            idx -= 1
        elif idx > 3 and idx <= 3 + self.N:
            return 'pure%d' %(idx-3)
        elif idx > 3+self.N:
            return 'random%d' %(idx-(3+self.N))

    def incipient_guess_named(self, T, P, zs, name, zero_fraction=1E-6):
        N, cmps = self.N, self.cmps
        Ks_Wilson = [Wilson_K_value(T=T, P=P, Tc=self.Tcs[i], Pc=self.Pcs[i], omega=self.omegas[i]) for i in self.cmps]
        if name == 'Wilson gas':
            # Where the wilson guess leads to an incipient gas
            Ys_Wilson = [Ki*zi for Ki, zi in zip(Ks_Wilson, zs)]
            return normalize(Ys_Wilson)
        elif name == 'Wilson liquid':
            Ys_Wilson = [zi/Ki for Ki, zi in zip(Ks_Wilson, zs)]
            return normalize(Ys_Wilson)
        elif name == 'Wilson gas third':
            Ys_Wilson = [Ki**(1.0/3.0)*zi for Ki, zi in zip(Ks_Wilson, zs)]
            return normalize(Ys_Wilson)
        elif name == 'Wilson liquid third':
            Ys_Wilson = [Ki**(-1.0/3.0)*zi for Ki, zi in zip(Ks_Wilson, zs)]
            return normalize(Ys_Wilson)
        elif name[0:4] == 'pure':
            k = int(name[4:])
            main_frac = 1.0 - zero_fraction
            remaining = zero_fraction/(N-1.0)
            guess = [remaining]*N
            guess[k] = main_frac
            return guess

    def incipient_guesses(self, T, P, zs, pure=True, Wilson=True, random=True,
                zero_fraction=1E-6, expect_liquid=False, expect_aqueous=False,
                existing_phases=0):
        N, cmps = self.N, self.cmps
        Tcs, Pcs, omegas = self.Tcs, self.Pcs, self.omegas

        WILSON_MAX_GUESSES = 4
        PURE_MAX_GUESSES = N
        PURE_MAX = WILSON_MAX_GUESSES + PURE_MAX_GUESSES
        WILSON_UNLIKELY_K = 1e-100

        if random is True:
            RANDOM_MAX_GUESSES = N
        else:
            RANDOM_MAX_GUESSES = random

        if Wilson:
            Ks_Wilson = [0.0]*N
            P_inv, T_inv = 1.0/P, 1.0/T
            all_wilson_zero = True
            any_wilson_zero = False
            wilson_unlikely = True
            for i in cmps:
                Ks_Wilson[i] = Pcs[i]*P_inv*exp(5.37*(1.0 + omegas[i])*(1.0 - Tcs[i]*T_inv))
            for i in cmps:
                if Ks_Wilson[i] != 0.0:
                    all_wilson_zero = False
                else:
                    any_wilson_zero = True
                if Ks_Wilson[i] > WILSON_UNLIKELY_K:
                    wilson_unlikely = False


            if expect_liquid and not all_wilson_zero and not wilson_unlikely:
                if expect_aqueous:
                    main_frac = 1.0 - zero_fraction
                    remaining = zero_fraction/(N-1.0)
                    guess = [remaining]*N
                    guess[self.water_index] = main_frac
                    yield guess
                # if existing_phases:
                #     yield zs

                if not any_wilson_zero:
                    yield normalize([zs[i]/Ks_Wilson[i] for i in cmps]) # liquid composition estimate
                    yield normalize([Ks_Wilson[i]**(-1.0/3.0)*zs[i] for i in cmps])

                yield normalize([Ks_Wilson[i]**(1.0/3.0)*zs[i] for i in cmps])
                yield normalize([Ks_Wilson[i]*zs[i] for i in cmps]) # gas composition estimate

            elif not all_wilson_zero and not wilson_unlikely:
                yield normalize([Ks_Wilson[i]*zs[i] for i in cmps]) # gas composition estimate
                # if existing_phases:
                    # yield zs
                if not any_wilson_zero:
                    yield normalize([zs[i]/Ks_Wilson[i] for i in cmps]) # liquid composition estimate
                # TODO optimization - can cache Ks_Wilson power, and cache it.
                yield normalize([Ks_Wilson[i]**(1.0/3.0)*zs[i] for i in cmps])
                if not any_wilson_zero:
                    yield normalize([Ks_Wilson[i]**(-1.0/3.0)*zs[i] for i in cmps])

        if pure: # these could be pre-allocated based on zero_fraction
            # A pure phase is more likely to be a liquid than a gas - gases have no polar effects
            main_frac = 1.0 - zero_fraction
            remaining = zero_fraction/(N-1.0)
            for k in cmps:
                guess = [remaining]*N
                guess[k] = main_frac
                yield guess
        if random: # these could be pre-allocated based on N, although changing number is hard
            # Probably bad for cache access however
            # More likely to predict a liquid phase than a gas phase
            idx = 0
            if RANDOM_MAX_GUESSES*N > 1000: # len(random_values)
                for i in range(RANDOM_MAX_GUESSES):
                    seed(0)
                    for k in cmps:
                        yield normalize([uniform(0.0, 1.0) for _ in cmps])
            else:
                for i in range(RANDOM_MAX_GUESSES):
                    guess = random_values[idx:idx+N]
                    tot_inv = 1.0/sum(guess)
                    for j in cmps:
                        guess[j] *= tot_inv
                    yield guess
                    idx += N

    def guess_generator(self, T, P, zs, pure=True, Wilson=True, random=True,
                zero_fraction=1E-6):
        WILSON_MAX_GUESSES = 4
        PURE_MAX_GUESSES = self.N
        PURE_MAX = WILSON_MAX_GUESSES + PURE_MAX_GUESSES

        if random is True:
            RANDOM_MAX_GUESSES = self.N
        else:
            RANDOM_MAX_GUESSES = random

        RANDOM_MAX = RANDOM_MAX_GUESSES + PURE_MAX

        max_guesses = RANDOM_MAX

        guesses = []
        if Wilson:
            guesses.extend(self.Wilson_guesses(T, P, zs))
        if pure:
            guesses.extend(self.pure_guesses(zero_fraction))
        if random:
            if random is True:
                guesses.extend(self.random_guesses())
            else:
                guesses.extend(self.random_guesses(random))

        i = 0
        while i < max_guesses:
            if i < WILSON_MAX_GUESSES:
                yield guesses.pop(0)
            elif i < PURE_MAX:
                yield guesses.pop(0)
            elif i < RANDOM_MAX:
                yield guesses.pop(0)
            i += 1



    def guesses(self, T, P, zs, pure=True, Wilson=True, random=True,
                zero_fraction=1E-6):
        '''Returns mole fractions, not Ks.
        '''
        guesses = []
        if Wilson:
            guesses.extend(self.Wilson_guesses(T, P, zs))
#            print('Wilson_guesses raw', guesses)
        if pure:
            guesses.extend(self.pure_guesses(zero_fraction))
        if random:
            if random is True:
                guesses.extend(self.random_guesses())
            else:
                guesses.extend(self.random_guesses(random))

        return guesses


class PropertyPackage(object):


    # Constant - if the phase fraction is this close to either the liquid or
    # vapor phase, round it to it
    PHASE_ROUNDING_TOL = 1E-9
    SUPPORTS_ZERO_FRACTIONS = True
    zero_fraction = 1E-6
    FLASH_VF_TOL = 1e-6

    T_REF_IG = 298.15
    P_REF_IG = 101325.
    P_REF_IG_INV = 1.0/P_REF_IG

    T_MAX_FIXED = 10000.0
    T_MIN_FIXED = 1e-3

    P_MAX_FIXED = 1e9
    P_MIN_FIXED = 1e-3

    def to(self, zs, T=None, P=None, VF=None):
        from copy import copy
        obj = copy(self)
        obj.flash(T=T, P=P, VF=VF, zs=zs)
        return obj

    def __copy__(self):
        obj = self.__class__(**self.kwargs)
        return obj

    def Tdew(self, P, zs):
        return self.to(P=P, VF=1, zs=zs).T

    def Pdew(self, T, zs):
        return self.to(T=T, VF=1, zs=zs).P

    def Tbubble(self, P, zs):
        return self.to(P=P, VF=0, zs=zs).T

    def Pbubble(self, T, zs):
        return self.to(T=T, VF=0, zs=zs).P

    def _post_flash(self):
        pass

    def flash(self, zs, T=None, P=None, VF=None):
        '''Note: There is no caching at this layer
        '''
        if not self.SUPPORTS_ZERO_FRACTIONS:
            zs = remove_zeros(zs, 1e-11)
        if T is not None and P is not None:
            phase, xs, ys, V_over_F = self.flash_TP_zs(T=T, P=P, zs=zs)
        elif T is not None and VF is not None:
            phase, xs, ys, V_over_F, P = self.flash_TVF_zs(T=T, VF=VF, zs=zs)
        elif P is not None and VF is not None:
            phase, xs, ys, V_over_F, T = self.flash_PVF_zs(P=P, VF=VF, zs=zs)
        else:
            raise Exception('Unsupported flash requested')

        if VF is not None:
            # Handle the case that a non-zero VF was specified, but the flash's
            # tolerance results in the phase being rounded.
            if V_over_F < self.PHASE_ROUNDING_TOL and VF > self.PHASE_ROUNDING_TOL:
                V_over_F = VF
            elif V_over_F > 1. - self.PHASE_ROUNDING_TOL and VF < 1. - self.PHASE_ROUNDING_TOL:
                V_over_F = VF
        # Truncate
        if phase  == 'l/g':
            if V_over_F < self.PHASE_ROUNDING_TOL: # liquid
                phase, xs, ys, V_over_F = 'l', zs, None, 0.
            elif V_over_F > 1. - self.PHASE_ROUNDING_TOL:
                phase, xs, ys, V_over_F = 'g', None, zs, 1.

        self.T = T
        self.P = P
        self.V_over_F = V_over_F
        self.phase = phase
        self.xs = xs
        self.ys = ys
        self.zs = zs

        self._post_flash()

    def plot_Pxy(self, T, pts=30, display=True, ignore_errors=True,
                 values=False): # pragma: no cover
        if not has_matplotlib() and values is not False:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if self.N != 2:
            raise Exception('Pxy plotting requires a mixture of exactly two components')
        z1 = linspace(0, 1, pts)
        z2 = [1.0 - zi for zi in z1]
        Ps_dew = []
        Ps_bubble = []

        for i in range(pts):
            try:
                self.flash(T=T, VF=0, zs=[z1[i], z2[i]])
                Ps_bubble.append(self.P)
            except Exception as e:
                if ignore_errors:
                    Ps_bubble.append(None)
                else:
                    raise e
            try:
                self.flash(T=T, VF=1, zs=[z1[i], z2[i]])
                Ps_dew.append(self.P)
            except Exception as e:
                if ignore_errors:
                    Ps_dew.append(None)
                else:
                    raise e
        if values:
            return z1, z2, Ps_bubble, Ps_dew

        import matplotlib.pyplot as plt
        plt.title('Pxy diagram at T=%s K' %T)
        plt.plot(z1, Ps_dew, label='Dew pressure')
        plt.plot(z1, Ps_bubble, label='Bubble pressure')
        plt.xlabel('Mole fraction x1')
        plt.ylabel('System pressure, Pa')
        plt.legend(loc='best')
        if display:
            plt.show()
        else:
            return plt

    def plot_Txy(self, P, pts=30, display=True, ignore_errors=True,
                 values=False): # pragma: no cover
        if not has_matplotlib() and values is not False:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if self.N != 2:
            raise Exception('Txy plotting requires a mixture of exactly two components')
        z1 = linspace(0, 1, pts)
        z2 = [1.0 - zi for zi in z1]
        Ts_dew = []
        Ts_bubble = []

        for i in range(pts):
            try:
                self.flash(P=P, VF=0, zs=[z1[i], z2[i]])
                Ts_bubble.append(self.T)
            except Exception as e:
                if ignore_errors:
                    Ts_bubble.append(None)
                else:
                    raise e
            try:
                self.flash(P=P, VF=1, zs=[z1[i], z2[i]])
                Ts_dew.append(self.T)
            except Exception as e:
                if ignore_errors:
                    Ts_dew.append(None)
                else:
                    raise e
        if values:
            return z1, z2, Ts_bubble, Ts_dew
        import matplotlib.pyplot as plt
        plt.title('Txy diagram at P=%s Pa' %P)
        plt.plot(z1, Ts_dew, label='Dew temperature, K')
        plt.plot(z1, Ts_bubble, label='Bubble temperature, K')
        plt.xlabel('Mole fraction x1')
        plt.ylabel('System temperature, K')
        plt.legend(loc='best')
        if display:
            plt.show()
        else:
            return plt

    def plot_xy(self, P=None, T=None, pts=30, display=True): # pragma: no cover
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        if self.N != 2:
            raise Exception('xy plotting requires a mixture of exactly two components')
        import matplotlib.pyplot as plt
        z1 = np.linspace(1e-8, 1-1e-8, pts)
        z2 = 1 - z1
        y1_bubble = []
        x1_bubble = []
        for i in range(pts):
            try:
                if T is not None:
                    self.flash(T=T, VF=self.PHASE_ROUNDING_TOL*2, zs=[float(z1[i]), float(z2[i])])
    #                print(T, self.xs, self.ys, self.V_over_F)
                elif P is not None:
                    self.flash(P=P, VF=self.PHASE_ROUNDING_TOL*2, zs=[float(z1[i]), float(z2[i])])
    #                print(P, self.xs, self.ys, self.V_over_F, self.zs)
                x1_bubble.append(self.xs[0])
                y1_bubble.append(self.ys[0])
            except Exception as e:
                print('Failed on pt %d' %(i), e)
        if T:
            plt.title('xy diagram at T=%s K (varying P)' %T)
        else:
            plt.title('xy diagram at P=%s Pa (varying T)' %P)
        plt.xlabel('Liquid mole fraction x1')
        plt.ylabel('Vapor mole fraction x1')
        plt.plot(x1_bubble, y1_bubble, '-', label='liquid vs vapor composition')
        plt.legend(loc='best')
        plt.plot([0, 1], [0, 1], '--')
        plt.axis((0,1,0,1))
        if display:
            plt.show()
        else:
            return plt

    def plot_PT(self, zs, Pmin=None, Pmax=None, pts=50, branches=[],
                ignore_errors=True, values=False): # pragma: no cover
        if not has_matplotlib() and not values:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if not Pmin:
            Pmin = 1e4
        if not Pmax:
            Pmax = min(self.Pcs)
        import matplotlib.pyplot as plt
        Ps = logspace(log10(Pmin), log10(Pmax), pts)
        T_dews = []
        T_bubbles = []
        branch = bool(len(branches))
        if branch:
            branch_Ts = [[] for i in range(len(branches))]
        else:
            branch_Ts = None
        for P in Ps:
            try:
                self.flash(P=P, VF=0, zs=zs)
                T_bubbles.append(self.T)
            except Exception as e:
                if ignore_errors:
                    T_bubbles.append(None)
                else:
                    raise e
            try:
                self.flash(P=P, VF=1, zs=zs)
                T_dews.append(self.T)
            except Exception as e:
                if ignore_errors:
                    T_dews.append(None)
                else:
                    raise e

            if branch:
                for VF, Ts in zip(branches, branch_Ts):
                    try:
                        self.flash(P=P, VF=VF, zs=zs)
                        Ts.append(self.T)
                    except Exception as e:
                        if ignore_errors:
                            Ts.append(None)
                        else:
                            raise e
        if values:
            return Ps, T_dews, T_bubbles, branch_Ts
        import matplotlib.pyplot as plt
        plt.plot(Ps, T_dews, label='PT dew point curve')
        plt.plot(Ps, T_bubbles, label='PT bubble point curve')
        plt.xlabel('System pressure, Pa')
        plt.ylabel('System temperature, K')
        plt.title('PT system curve, zs=%s' %zs)
        if branch:
            for VF, Ts in zip(branches, branch_Ts):
                plt.plot(Ps, Ts, label='PT curve for VF=%s'%VF)
        plt.legend(loc='best')
        plt.show()


    def plot_TP(self, zs, Tmin=None, Tmax=None, pts=50, branches=[],
                ignore_errors=True, values=False): # pragma: no cover
        if not has_matplotlib() and not values:
            raise Exception('Optional dependency matplotlib is required for plotting')
        if not Tmin:
            Tmin = min(self.Tms)
        if not Tmax:
            Tmax = min(self.Tcs)
        Ts = np.linspace(Tmin, Tmax, pts)
        P_dews = []
        P_bubbles = []
        branch = bool(len(branches))
        if branch:
            branch_Ps = [[] for i in range(len(branches))]
        else:
            branch_Ps = None
        for T in Ts:
            try:
                self.flash(T=T, VF=0, zs=zs)
                P_bubbles.append(self.P)
            except Exception as e:
                if ignore_errors:
                    P_bubbles.append(None)
                else:
                    raise e
            try:
                self.flash(T=T, VF=1, zs=zs)
                P_dews.append(self.P)
            except Exception as e:
                if ignore_errors:
                    P_dews.append(None)
                else:
                    raise e

            if branch:
                for VF, Ps in zip(branches, branch_Ps):
                    try:
                        self.flash(T=T, VF=VF, zs=zs)
                        Ps.append(self.P)
                    except Exception as e:
                        if ignore_errors:
                            Ps.append(None)
                        else:
                            raise e
        if values:
            return Ts, P_dews, P_bubbles, branch_Ps
        import matplotlib.pyplot as plt
        plt.plot(Ts, P_dews, label='TP dew point curve')
        plt.plot(Ts, P_bubbles, label='TP bubble point curve')
        plt.xlabel('System temperature, K')
        plt.ylabel('System pressure, Pa')
        plt.title('PT system curve, zs=%s' %zs)
        if branch:
            for VF, Ps in zip(branches, branch_Ps):
                plt.plot(Ts, Ps, label='TP curve for VF=%s'%VF)
        plt.legend(loc='best')
        plt.show()


    def plot_ternary(self, T, scale=10): # pragma: no cover
        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        try:
            import ternary
        except:
            raise Exception('Optional dependency ternary is required for ternary plotting')
        if self.N != 3:
            raise Exception('Ternary plotting requires a mixture of exactly three components')

        P_values = []

        def P_dew_at_T_zs(zs):
            zs = remove_zeros(zs, self.zero_fraction)
            self.flash(T=T, zs=zs, VF=0)
            P_values.append(self.P)
            return self.P

        def P_bubble_at_T_zs(zs):
            zs = remove_zeros(zs, 1e-6)
            self.flash(T=T, zs=zs, VF=1)
            return self.P

        import matplotlib
        import matplotlib.pyplot as plt


        axes_colors = {'b': 'g', 'l': 'r', 'r':'b'}
        ticks = [round(i / float(10), 1) for i in range(10+1)]

        fig, ax = plt.subplots(1, 3, gridspec_kw = {'width_ratios':[4, 4, 1]})
        ax[0].axis("off") ; ax[1].axis("off")  ; ax[2].axis("off")

        for axis, f, i in zip(ax[0:2], [P_dew_at_T_zs, P_bubble_at_T_zs], [0, 1]):
            figure, tax = ternary.figure(ax=axis, scale=scale)
            figure.set_size_inches(12, 4)
            if not i:
                tax.heatmapf(f, boundary=True, colorbar=False, vmin=0)
            else:
                tax.heatmapf(f, boundary=True, colorbar=False, vmin=0, vmax=max(P_values))

            tax.boundary(linewidth=2.0)
            tax.left_axis_label("mole fraction $x_2$", offset=0.16, color=axes_colors['l'])
            tax.right_axis_label("mole fraction $x_1$", offset=0.16, color=axes_colors['r'])
            tax.bottom_axis_label("mole fraction $x_3$", offset=-0.06, color=axes_colors['b'])

            tax.ticks(ticks=ticks, axis='rlb', linewidth=1, clockwise=True,
                      axes_colors=axes_colors, offset=0.03)

            tax.gridlines(multiple=scale/10., linewidth=2,
                          horizontal_kwargs={'color':axes_colors['b']},
                          left_kwargs={'color':axes_colors['l']},
                          right_kwargs={'color':axes_colors['r']},
                          alpha=0.5)

        norm = plt.Normalize(vmin=0, vmax=max(P_values))
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
        sm._A = []
        cb = plt.colorbar(sm, ax=ax[2])
        cb.locator = matplotlib.ticker.LinearLocator(numticks=7)
        cb.formatter = matplotlib.ticker.ScalarFormatter()
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        plt.tight_layout()
        fig.suptitle("Bubble pressure vs composition (left) and dew pressure vs composition (right) at %s K, in Pa" %T, fontsize=14);
        fig.subplots_adjust(top=0.85)
        plt.show()

    def plot_TP_caloric(self, zs, Tmin=None, Tmax=None, Pmin=None, Pmax=None,
                        pts=15, prop='Hm'):  # pragma: no cover
        if prop not in ['Sm', 'Gm', 'Hm']:
            raise Exception("The only supported property plots are enthalpy "
                            "('Hm'), entropy ('Sm'), and Gibbe energy ('Gm')")
        prop_name = {'Hm': 'enthalpy', 'Sm': 'entropy', 'Gm': 'Gibbs energy'}[prop]
        prop_units = {'Hm': 'J/mol', 'Sm': 'J/mol/K', 'Gm': 'J/mol'}[prop]

        if not has_matplotlib():
            raise Exception('Optional dependency matplotlib is required for plotting')
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib.ticker import FormatStrFormatter
        import numpy.ma as ma

        if Pmin is None:
            raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Pmax is None:
            raise Exception('Maximum pressure could not be auto-detected; please provide it')
        if Tmin is None:
            raise Exception('Minimum pressure could not be auto-detected; please provide it')
        if Tmax is None:
            raise Exception('Maximum pressure could not be auto-detected; please provide it')

        Ps = np.linspace(Pmin, Pmax, pts)
        Ts = np.linspace(Tmin, Tmax, pts)
        Ts_mesh, Ps_mesh = np.meshgrid(Ts, Ps)
        fig = plt.figure()
        ax = fig.gca(projection='3d')


        properties = []
        for T in Ts:
            properties2 = []
            for P in Ps:
                self.flash_caloric(zs=zs, T=T, P=P)
                properties2.append(getattr(self, prop))
            properties.append(properties2)

        ax.plot_surface(Ts_mesh, Ps_mesh, properties, cstride=1, rstride=1, alpha=0.5)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.4g'))
        ax.set_xlabel('Temperature, K')
        ax.set_ylabel('Pressure, Pa')
        ax.set_zlabel('%s, %s' %(prop_name, prop_units))
        plt.title('Temperature-pressure %s plot' %prop_name)
        plt.show(block=False)



    def flash_caloric(self, zs, T=None, P=None, VF=None, Hm=None, Sm=None):
        if not self.SUPPORTS_ZERO_FRACTIONS:
            zs = remove_zeros(zs, self.zero_fraction)

        kwargs = {'zs': zs}
        try:
            if T is not None and Sm is not None:
                kwargs['T'] = T
                kwargs.update(self.flash_TS_zs_bounded(T=T, Sm=Sm, zs=zs))
            elif P is not None and Sm is not None:
                kwargs['P'] = P
                kwargs.update(self.flash_PS_zs_bounded(P=P, Sm=Sm, zs=zs))
            elif P is not None and Hm is not None:
                kwargs['P'] = P
                kwargs.update(self.flash_PH_zs_bounded(P=P, Hm=Hm, zs=zs))
            elif ((T is not None and P is not None) or
                (T is not None and VF is not None) or
                (P is not None and VF is not None)):
                kwargs['P'] = P
                kwargs['T'] = T
                kwargs['VF'] = VF
            else:
                raise Exception('Flash inputs unsupported')


#            ''' The routine needs to be upgraded to set these properties
#                self.T = T
#                self.P = P
#                self.V_over_F = V_over_F
#                self.phase = phase
#                self.xs = xs
#                self.ys = ys
#                self.zs = zs
#            '''
#            self.__dict__.update(kwargs)

            self.flash(**kwargs)
            self._post_flash()
            self.status = True
        except Exception as e:
            # Write Nones for everything here
            self.status = e
            self._set_failure()

    def _set_failure(self):
        self.Hm = None
        self.Sm = None
        self.Gm = None
        self.chemical_potential = None
        self.T = None
        self.P = None
        self.phase = None
        self.V_over_F = None
        self.xs = None
        self.ys = None




    def flash_PH_zs_bounded(self, P, Hm, zs, T_low=None, T_high=None,
                            Hm_low=None, Hm_high=None):
        '''THIS DOES NOT WORK FOR PURE COMPOUNDS!!!!!!!!!!!!!
        '''
        # Begin the search at half the lowest chemical's melting point
        if T_low is None:
            T_low = min(self.Tms)/2

        # Cap the T high search at 8x the highest critical point
        # (will not work well for helium, etc.)
        if T_high is None:
            max_Tc = max(self.Tcs)
            if max_Tc < 100:
                T_high = 4000.0
            else:
                T_high = max_Tc*8.0

#        print('T_low, T_high', T_low, T_high)
        temp_pkg_cache = []
        def PH_error(T, P, zs, H_goal):
#            print(T, P, H_goal)
            if not temp_pkg_cache:
                temp_pkg = self.to(T=T, P=P, zs=zs)
                temp_pkg_cache.append(temp_pkg)
            else:
                temp_pkg = temp_pkg_cache[0]
                temp_pkg.flash(T=T, P=P, zs=zs)
            temp_pkg._post_flash()
#            print(temp_pkg.Hm - H_goal, T, P)
            err = temp_pkg.Hm - H_goal
#            print(T, err)
            return err

        def PH_VF_error(VF, P, zs, H_goal):
            if not temp_pkg_cache:
                temp_pkg = self.to(VF=VF, P=P, zs=zs)
                temp_pkg_cache.append(temp_pkg)
            else:
                temp_pkg = temp_pkg_cache[0]
                temp_pkg.flash(VF=VF, P=P, zs=zs)
            temp_pkg._post_flash()
            return temp_pkg.Hm - H_goal
        try:
            try:
                T_goal = brenth(PH_error, T_low, T_high, ytol=1e-6, args=(P, zs, Hm))
            except UnconvergedError:
                if self.N == 1:
                    VF_goal = brenth(PH_VF_error, 0, 1, args=(P, zs, Hm))
                    return {'VF': VF_goal}
            return {'T': T_goal}

        except ValueError:
            if Hm_low is None:
                pkg_low = self.to(T=T_low, P=P, zs=zs)
                pkg_low._post_flash()
                Hm_low = pkg_low.Hm
            if Hm < Hm_low:
                raise ValueError('The requested molar enthalpy cannot be found'
                                 ' with this bounded solver because the lower '
                                 'temperature bound %g K has an enthalpy (%g '
                                 'J/mol) higher than that requested (%g J/mol)' %(
                                                             T_low, Hm_low, Hm))
            if Hm_high is None:
                pkg_high = self.to(T=T_high, P=P, zs=zs)
                pkg_high._post_flash()
                Hm_high = pkg_high.Hm
            if Hm > Hm_high:
                raise ValueError('The requested molar enthalpy cannot be found'
                                 ' with this bounded solver because the upper '
                                 'temperature bound %g K has an enthalpy (%g '
                                 'J/mol) lower than that requested (%g J/mol)' %(
                                                             T_high, Hm_high, Hm))


    def flash_PS_zs_bounded(self, P, Sm, zs, T_low=None, T_high=None,
                            Sm_low=None, Sm_high=None):
        '''THIS DOES NOT WORK FOR PURE COMPOUNDS!!!!!!!!!!!!!
        '''
        # Begin the search at half the lowest chemical's melting point
        if T_low is None:
            T_low = min(self.Tms)/2

        # Cap the T high search at 8x the highest critical point
        # (will not work well for helium, etc.)
        if T_high is None:
            max_Tc = max(self.Tcs)
            if max_Tc < 100:
                T_high = 4000
            else:
                T_high = max_Tc*8

        temp_pkg_cache = []
        def PS_error(T, P, zs, S_goal):
            if not temp_pkg_cache:
                temp_pkg = self.to(T=T, P=P, zs=zs)
                temp_pkg_cache.append(temp_pkg)
            else:
                temp_pkg = temp_pkg_cache[0]
                temp_pkg.flash(T=T, P=P, zs=zs)
            temp_pkg._post_flash()
            return temp_pkg.Sm - S_goal

        def PS_VF_error(VF, P, zs, S_goal):
            if not temp_pkg_cache:
                temp_pkg = self.to(VF=VF, P=P, zs=zs)
                temp_pkg_cache.append(temp_pkg)
            else:
                temp_pkg = temp_pkg_cache[0]
                temp_pkg.flash(VF=VF, P=P, zs=zs)
            temp_pkg._post_flash()
            return temp_pkg.Sm - S_goal
        try:
            T_goal = brenth(PS_error, T_low, T_high, args=(P, zs, Sm))
            if self.N == 1:
                err = abs(PS_error(T_goal, P, zs, Sm))
                if err > 1E-3:
                    VF_goal = brenth(PS_VF_error, 0, 1, args=(P, zs, Sm))
                    return {'VF': VF_goal}


            return {'T': T_goal}

        except ValueError:
            if Sm_low is None:
                pkg_low = self.to(T=T_low, P=P, zs=zs)
                pkg_low._post_flash()
                Sm_low = pkg_low.Sm
            if Sm < Sm_low:
                raise ValueError('The requested molar entropy cannot be found'
                                 ' with this bounded solver because the lower '
                                 'temperature bound %g K has an entropy (%g '
                                 'J/mol/K) higher than that requested (%g J/mol/K)' %(
                                                             T_low, Sm_low, Sm))
            if Sm_high is None:
                pkg_high = self.to(T=T_high, P=P, zs=zs)
                pkg_high._post_flash()
                Sm_high = pkg_high.Sm
            if Sm > Sm_high:
                raise ValueError('The requested molar entropy cannot be found'
                                 ' with this bounded solver because the upper '
                                 'temperature bound %g K has an entropy (%g '
                                 'J/mol/K) lower than that requested (%g J/mol/K)' %(
                                                             T_high, Sm_high, Sm))

    def flash_TS_zs_bounded(self, T, Sm, zs, P_low=None, P_high=None,
                            Sm_low=None, Sm_high=None):
        # Begin the search at half the lowest chemical's melting point
        if P_high is None:
            if self.N == 1:
                P_high = self.Pcs[0]
            else:
                P_high = self.Pdew(T, zs)*100
        if P_low is None:
            P_low = 1E-5 # min pressure

        temp_pkg_cache = []
        def TS_error(P, T, zs, S_goal):
            if not temp_pkg_cache:
                temp_pkg = self.to(T=T, P=P, zs=zs)
                temp_pkg_cache.append(temp_pkg)
            else:
                temp_pkg = temp_pkg_cache[0]
                temp_pkg.flash(T=T, P=P, zs=zs)
            temp_pkg._post_flash()
            return temp_pkg.Sm - S_goal
        def TS_VF_error(VF, T, zs, S_goal):
            if not temp_pkg_cache:
                temp_pkg = self.to(VF=VF, T=T, zs=zs)
                temp_pkg_cache.append(temp_pkg)
            else:
                temp_pkg = temp_pkg_cache[0]
                temp_pkg.flash(VF=VF, T=T, zs=zs)
            temp_pkg._post_flash()
            return temp_pkg.Sm - S_goal
        try:
            P_goal = brenth(TS_error, P_low, P_high, args=(T, zs, Sm))
            if self.N == 1:
                err = abs(TS_error(P_goal, T, zs, Sm))
                if err > 1E-3:
                    VF_goal = brenth(TS_VF_error, 0, 1, args=(T, zs, Sm))
                    return {'VF': VF_goal}
            return {'P': P_goal}

        except ValueError:
            if Sm_low is None:
                pkg_low = self.to(T=T, P=P_low, zs=zs)
                pkg_low._post_flash()
                Sm_low = pkg_low.Sm
            if Sm > Sm_low:
                raise ValueError('The requested molar entropy cannot be found'
                                 ' with this bounded solver because the lower '
                                 'pressure bound %g Pa has an entropy (%g '
                                 'J/mol/K) lower than that requested (%g J/mol/K)' %(
                                                             P_low, Sm_low, Sm))
            if Sm_high is None:
                pkg_high = self.to(T=T, P=P_high, zs=zs)
                pkg_high._post_flash()
                Sm_high = pkg_high.Sm
            if Sm < Sm_high:
                raise ValueError('The requested molar entropy cannot be found'
                                 ' with this bounded solver because the upper '
                                 'pressure bound %g Pa has an entropy (%g '
                                 'J/mol/K) upper than that requested (%g J/mol/K)' %(
                                                             P_high, Sm_high, Sm))
    @property
    def Hm_reactive(self):
        Hm = self.Hm
        for zi, Hf in zip(self.zs, self.Hfs):
            Hm += zi*Hf
        return Hm

    @property
    def Sm_reactive(self):
        Sm = self.Sm
        for zi, Sf in zip(self.zs, self.Sfs):
            Sm += zi*Sf
        return Sm

    @property
    def Gm_reactive(self):
        Gm = self.Hm_reactive - self.T*self.Sm_reactive
        return Gm


class Ideal(PropertyPackage):
    def Ks(self, T, P, zs=None):
        Psats = self._Psats(T)
        Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
        return Ks


    def _T_VF_err_ideal(self, P, VF, zs, Psats):
        Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
        return Rachford_Rice_solution_negative(zs=zs, Ks=Ks)[0] - VF

    def _P_VF_err_ideal(self, T, P, VF, zs):
        Psats = self._Psats(T)
        Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
        return Rachford_Rice_solution_negative(zs=zs, Ks=Ks)[0] - VF

    def _Psats(self, T):
        # Need to reset the method because for the T bounded solver,
        # will normally get a different than prefered method as it starts
        # at the boundaries
        Psats = []
        for i in self.VaporPressures:
            Psats.append(i(T))
#            if i.locked:
#            else:
#                if T < i.Tmax:
#                    #i.method = None
#                    Psat = i(T)
#                    if Psat is None:
#                        Psat = i.extrapolate_tabular(T)
#                    Psats.append(Psat)
#                else:
#    #                print(i.CASRN)
#                    Psats.append(i.extrapolate_tabular(T))
        return Psats

    def _Tsats(self, P):
        Tsats = []
        for i in self.VaporPressures:
#            try:
            Tsats.append(i.solve_property(P))
#            except:
#                error = lambda T: i.extrapolate_tabular(T) - P
#                Tsats.append(brenth(error, i.Tmax, i.Tmax*5))
        return Tsats

    def _d_Psats_dT(self, T):
        dPsats_dT = []
        for i in self.VaporPressures:
            dPsats_dT.append(i.T_dependent_property_derivative(T))
        return dPsats_dT

    def __init__(self, VaporPressures=None, Tms=None, Tcs=None, Pcs=None,
                 **kwargs):
        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.N = len(VaporPressures)
        self.cmps = range(self.N)

        self.kwargs = {'VaporPressures': VaporPressures,
                       'Tms': Tms, 'Tcs': Tcs, 'Pcs': Pcs}

    def Ks_and_dKs_dT(self, T, P, zs=None):
        Psats = self._Psats(T)
        dPsats_dT = self._d_Psats_dT(T)
        P_inv = 1.0/P
        Ks = [Psat*P_inv for Psat in Psats]
        dKs_dT = [dPsat_dTi*P_inv for dPsat_dTi in dPsats_dT]
        return Ks, dKs_dT

    def d_VF_dT(self, delta=1e-4, full=True):
        # Not accurate enough
        VF1 = self.V_over_F
        zs = self.zs
        Ks, dKs_dT = self.Ks_and_dKs_dT(self.T, self.P, zs)
        # Perturb the Ks
        Ks2 = [Ki + dKi*delta for Ki, dKi in zip(Ks, dKs_dT)]
        VF2, _, _ = Rachford_Rice_solution_negative(zs, Ks2, guess=VF1)
        return (VF2 - VF1)/delta



    def flash_TP_zs(self, T, P, zs):
        return self.flash_TP_zs_ideal(T=T, P=P, zs=zs)

    def flash_TP_zs_ideal(self, T, P, zs):
        Psats = self._Psats(T)
        if self.N == 1:
            Pdew = Pbubble = Psats[0]
        else:
            Pdew = 1.0/sum([zs[i]/Psats[i] for i in range(self.N)])
            Pbubble = sum([zs[i]*Psats[i] for i in range(self.N)])
        if P <= Pdew:
            # phase, ys, xs, quality - works for 1 comps too
            return 'g', None, zs, 1
        elif P >= Pbubble:
            return 'l', zs, None, 0
        else:
            Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
            V_over_F, xs, ys = Rachford_Rice_solution_negative(zs=zs, Ks=Ks)
            return 'l/g', xs, ys, V_over_F


    def flash_TVF_zs(self, T, VF, zs):
        return self.flash_TVF_zs_ideal(T, VF, zs)

    def flash_TVF_zs_ideal(self, T, VF, zs):
        assert 0 <= VF <= 1
        Psats = self._Psats(T)
        # handle one component
        if self.N == 1:
            return 'l/g', [1.0], [1.0], VF, Psats[0]
        elif 1.0 in zs:
            return 'l/g', list(zs), list(zs), VF, Psats[zs.index(1.0)]

        if VF == 0:
            P = sum([zs[i]*Psats[i] for i in range(self.N)])
        elif VF == 1:
            P = 1.0/sum([zs[i]/Psats[i] for i in range(self.N)])
        else:
            P = brenth(self._T_VF_err_ideal, min(Psats)*(1+1E-7), max(Psats)*(1-1E-7), args=(VF, zs, Psats))
        Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
        V_over_F, xs, ys = Rachford_Rice_solution_negative(zs=zs, Ks=Ks)
        return 'l/g', xs, ys, V_over_F, P

    def flash_PVF_zs(self, P, VF, zs):
        return self.flash_PVF_zs_ideal(P, VF, zs)

    def flash_PVF_zs_ideal(self, P, VF, zs):
        assert 0 <= VF <= 1
        Tsats = self._Tsats(P)
        if self.N == 1:
            return 'l/g', [1.0], [1.0], VF, Tsats[0]
        elif 1.0 in zs:
            return 'l/g', list(zs), list(zs), VF, Tsats[zs.index(1.0)]

        T = brenth(self._P_VF_err_ideal, min(Tsats)*(1+1E-7), max(Tsats)*(1-1E-7), args=(P, VF, zs))
        Psats = self._Psats(T)
        Ks = [K_value(P=P, Psat=Psat) for Psat in Psats]
        V_over_F, xs, ys = Rachford_Rice_solution_negative(zs=zs, Ks=Ks)
        return 'l/g', xs, ys, V_over_F, T


class IdealCaloric(Ideal):
    P_DEPENDENT_H_LIQ = True

    @property
    def Cplm_dep(self):
        return 0.0

    @property
    def Cpgm_dep(self):
        return 0.0

    @property
    def Cvlm_dep(self):
        return 0.0

    @property
    def Cvgm_dep(self):
        return 0.0

    @property
    def Cplm(self):
        Cp = 0.0
        for i in self.cmps:
            Cp += self.zs[i]*self.HeatCapacityLiquids[i].T_dependent_property(self.T)
        return Cp

    @property
    def Cpgm(self):
        Cp = 0.0
        for i in self.cmps:
            Cp += self.zs[i]*self.HeatCapacityGases[i].T_dependent_property(self.T)
        return Cp

    @property
    def Cvgm(self):
        return self.Cpgm - R

    @property
    def Cvlm(self):
        return self.Cplm


    def Hg_ideal(self, T, zs):
        H = 0.0
        T_REF_IG = self.T_REF_IG
        HeatCapacityGases = self.HeatCapacityGases
        cmps = self.cmps
        for zi, obj in zip(zs, HeatCapacityGases):
            H += zi*obj.T_dependent_property_integral(T_REF_IG, T)
        return H


    def Cpg_ideal(self, T, zs):
        Cp = 0.0
        HeatCapacityGases = self.HeatCapacityGases
        for zi, HeatCapacityGas in zip(zs, HeatCapacityGases):
            Cp += zi*HeatCapacityGas.T_dependent_property(T)
        return Cp


    def __init__(self, VaporPressures=None, Tms=None, Tbs=None, Tcs=None, Pcs=None,
                 HeatCapacityLiquids=None, HeatCapacityGases=None,
                 EnthalpyVaporizations=None, VolumeLiquids=None, Hfs=None,
                 Gfs=None, **kwargs):
        self.cmps = range(len(VaporPressures))
        self.N = len(VaporPressures)
        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tbs = Tbs
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.HeatCapacityLiquids = HeatCapacityLiquids
        self.HeatCapacityGases = HeatCapacityGases
        self.EnthalpyVaporizations = EnthalpyVaporizations
        self.VolumeLiquids = VolumeLiquids

        self.Hfs = Hfs
        self.Gfs = Gfs
        if Hfs is not None and Gfs is not None and None not in Hfs and None not in Gfs:
            self.Sfs = [(Hfi - Gfi)/298.15 for Hfi, Gfi in zip(Hfs, Gfs)]

        self.kwargs = {'VaporPressures': VaporPressures,
                       'Tms': Tms, 'Tbs': Tbs, 'Tcs': Tcs, 'Pcs': Pcs,
                       'HeatCapacityLiquids': HeatCapacityLiquids,
                       'HeatCapacityGases': HeatCapacityGases,
                       'EnthalpyVaporizations': EnthalpyVaporizations,
                       'VolumeLiquids': VolumeLiquids}



    def _post_flash(self):
        # Cannot derive other properties with this
        self.Hm = self.enthalpy_Cpg_Hvap()
        self.Sm = self.entropy_Cpg_Hvap()
        self.Gm = self.Hm - self.T*self.Sm if (self.Hm is not None and self.Sm is not None) else None

    def partial_property(self, T, P, i, zs, prop='Hm'):
        r'''Method to calculate the partial molar property for entropy,
        enthalpy, or gibbs energy. Note the partial gibbs energy is known
        as chemical potential as well.

        .. math::
            \bar m_i = \left( \frac{\partial (n_T m)} {\partial n_i}
            \right)_{T, P, n_{j\ne i}}

        Parameters
        ----------
        T : float
            Temperature at which to calculate the partial property, [K]
        P : float
            Pressure at which to calculate the partial property, [Pa]
        i : int
            Compound index, [-]
        zs : list[float], optional
            Mole fractions of all species in the mixture, [-]

        Returns
        -------
        partial_prop : float
            Calculated partial property, [`units`]
        '''
        if prop not in ('Sm', 'Gm', 'Hm'):
            raise Exception("The only supported property plots are enthalpy "
                            "('Hm'), entropy ('Sm'), and Gibbe energy ('Gm')")

        def prop_extensive(ni, ns, i):
            ns[i] = ni
            n_tot = sum(ns)
            zs = normalize(ns)
            obj = self.to(T=T, P=P, zs=zs)
            obj.flash_caloric(T=T, P=P, zs=zs)
            property_value = getattr(obj, prop)
            return property_value*n_tot
        return derivative(prop_extensive, zs[i], dx=1E-6, args=[list(zs), i])



    def enthalpy_Cpg_Hvap(self):
        r'''Method to calculate the enthalpy of an ideal mixture. This routine
        is based on "route A", where the gas heat
        capacity and enthalpy of vaporization are used.

        The reference temperature is a property of the class; it defaults to
        298.15 K.

        For a pure gas mixture:

        .. math::
             H = \sum_i z_i \cdot \int_{T_{ref}}^T C_{p}^{ig}(T) dT

        For a pure liquid mixture:

        .. math::
             H = \sum_i z_i \left( \int_{T_{ref}}^T C_{p}^{ig}(T) dT + H_{vap, i}(T) \right)

        For a vapor-liquid mixture:

        .. math::
             H = \sum_i z_i \cdot \int_{T_{ref}}^T C_{p}^{ig}(T) dT
                 + \sum_i x_i\left(1 - \frac{V}{F}\right)H_{vap, i}(T)

        For liquids, the enthalpy contribution of pressure is:

        .. math::
            \Delta H = \sum_i z_i (P - P_{sat, i}) V_{m, i}

        Returns
        -------
        H : float
            Enthalpy of the mixture with respect to the reference temperature,
            [J/mol]

        Notes
        -----
        The object must be flashed before this routine can be used. It
        depends on the properties T, zs, xs, V_over_F, HeatCapacityGases,
        EnthalpyVaporizations, and.
        '''
        H = 0
        T = self.T
        P = self.P
        if self.phase == 'g':
            for i in self.cmps:
                H += self.zs[i]*self.HeatCapacityGases[i].T_dependent_property_integral(self.T_REF_IG, T)
        elif self.phase == 'l':
            Psats = self._Psats(T=T)
            for i in self.cmps:
                # No further contribution needed
                Hg298_to_T = self.HeatCapacityGases[i].T_dependent_property_integral(self.T_REF_IG, T)
                Hvap = self.EnthalpyVaporizations[i](T) # Do the transition at the temperature of the liquid
                if Hvap is None:
                    Hvap = 0 # Handle the case of a package predicting a transition past the Tc
                H_i = Hg298_to_T - Hvap
                if self.P_DEPENDENT_H_LIQ:
                    Vl = self.VolumeLiquids[i](T, P)
                    if Vl is None:
                        # Handle an inability to get a liquid volume by taking
                        # one at the boiling point (and system P)
                        Vl = self.VolumeLiquids[i].TP_or_T_dependent_property(self.Tbs[i], P)
                    H_i += (P - Psats[i])*Vl
                H += self.zs[i]*(H_i)
        elif self.phase == 'l/g':
            for i in self.cmps:
                Hg298_to_T_zi = self.zs[i]*self.HeatCapacityGases[i].T_dependent_property_integral(self.T_REF_IG, T)
                Hvap = self.EnthalpyVaporizations[i](T)
                if Hvap is None:
                    Hvap = 0 # Handle the case of a package predicting a transition past the Tc
                Hvap_contrib = -self.xs[i]*(1-self.V_over_F)*Hvap
                H += (Hg298_to_T_zi + Hvap_contrib)
        return H

    def set_T_transitions(self, Ts):
        if Ts == 'Tb':
            self.T_trans = self.Tbs
        elif Ts == 'Tc':
            self.T_trans = self.Tcs
        elif isinstance(Ts, float):
            self.T_trans = [Ts]*self.N
        else:
            self.T_trans = Ts

    def enthalpy_Cpl_Cpg_Hvap(self):
        H = 0
        T = self.T
        T_trans = self.T_trans

        if self.phase == 'l':
            for i in self.cmps:
                H += self.zs[i]*self.HeatCapacityLiquids[i].T_dependent_property_integral(self.T_REF_IG, T)
        elif self.phase == 'g':
            for i in self.cmps:
                H_to_trans = self.HeatCapacityLiquids[i].T_dependent_property_integral(self.T_REF_IG, self.T_trans[i])
                H_trans = self.EnthalpyVaporizations[i](self.T_trans[i])
                H_to_T_gas = self.HeatCapacityGases[i].T_dependent_property_integral(self.T_trans[i], T)
                H += self.zs[i]*(H_to_trans + H_trans + H_to_T_gas)
        elif self.phase == 'l/g':
            for i in self.cmps:
                H_to_T_liq = self.HeatCapacityLiquids[i].T_dependent_property_integral(self.T_REF_IG, T)
                H_to_trans = self.HeatCapacityLiquids[i].T_dependent_property_integral(self.T_REF_IG, self.T_trans[i])
                H_trans = self.EnthalpyVaporizations[i](self.T_trans[i])
                H_to_T_gas = self.HeatCapacityGases[i].T_dependent_property_integral(self.T_trans[i], T)
                H += self.V_over_F*self.ys[i]*(H_to_trans + H_trans + H_to_T_gas)
                H += (1-self.V_over_F)*self.xs[i]*(H_to_T_liq)
        return H


    def entropy_Cpg_Hvap(self):
        r'''Method to calculate the entropy of an ideal mixture. This routine
        is based on "route A", where only the gas heat capacity and enthalpy of
        vaporization are used.

        The reference temperature and pressure are properties of the class; it
        defaults to 298.15 K and 101325 Pa.

        There is a contribution due to mixing:

        .. math::
            \Delta S_{mixing} = -R\sum_i z_i \ln(z_i)

        The ideal gas pressure contribution is:

        .. math::
            \Delta S_{P} = -R\ln\left(\frac{P}{P_{ref}}\right)

        For a liquid mixture or a partially liquid mixture, the entropy
        contribution is not so strong - all such pressure effects find that
        expression capped at the vapor pressure, as shown in [1]_.

        .. math::
            \Delta S_{P} = - \sum_i x_i\left(1 - \frac{V}{F}\right)
            R\ln\left(\frac{P_{sat, i}}{P_{ref}}\right) - \sum_i y_i\left(
            \frac{V}{F}\right) R\ln\left(\frac{P}{P_{ref}}\right)

        These expressions are combined with the standard heat capacity and
        enthalpy of vaporization expressions to calculate the total entropy:

        For a pure gas mixture:

        .. math::
             S = \Delta S_{mixing} + \Delta S_{P} + \sum_i z_i \cdot
             \int_{T_{ref}}^T \frac{C_{p}^{ig}(T)}{T} dT

        For a pure liquid mixture:

        .. math::
             S = \Delta S_{mixing} + \Delta S_{P} + \sum_i z_i \left(
             \int_{T_{ref}}^T \frac{C_{p}^{ig}(T)}{T} dT + \frac{H_{vap, i}
             (T)}{T} \right)

        For a vapor-liquid mixture:

        .. math::
             S = \Delta S_{mixing} + \Delta S_{P} + \sum_i z_i \cdot
             \int_{T_{ref}}^T \frac{C_{p}^{ig}(T)}{T} dT + \sum_i x_i\left(1
             - \frac{V}{F}\right)\frac{H_{vap, i}(T)}{T}

        Returns
        -------
        S : float
            Entropy of the mixture with respect to the reference temperature,
            [J/mol/K]

        Notes
        -----
        The object must be flashed before this routine can be used. It
        depends on the properties T, P, zs, V_over_F, HeatCapacityGases,
        EnthalpyVaporizations, VaporPressures, and xs.

        References
        ----------
        .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
           New York: McGraw-Hill Professional, 2000.
        '''
        S = 0.0
        T = self.T
        P = self.P
        S -= R*sum([zi*log(zi) for zi in self.zs if zi > 0.0]) # ideal composition entropy composition; chemsep checked
        # Both of the mixing and vapor pressure terms have negative signs
        # Equation 6-4.4b in Poling for the vapor pressure component
        # For liquids above their critical temperatures, Psat is equal to the system P (COCO).
        if self.phase == 'g':
            S -= R*log(P/101325.) # Gas-phase ideal pressure contribution (checked repeatedly)
            for i in self.cmps:
                S += self.zs[i]*self.HeatCapacityGases[i].T_dependent_property_integral_over_T(298.15, T)
        elif self.phase == 'l':
            Psats = self._Psats(T=T)
            T_inv = 1.0/T
            for i in self.cmps:
                Sg298_to_T = self.HeatCapacityGases[i].T_dependent_property_integral_over_T(298.15, T)
                Hvap = self.EnthalpyVaporizations[i](T)
                if Hvap is None:
                    Hvap = 0.0 # Handle the case of a package predicting a transition past the Tc
                Svap = -Hvap*T_inv # Do the transition at the temperature of the liquid
                S_P = -R*log(Psats[i]/101325.)
                S += self.zs[i]*(Sg298_to_T + Svap + S_P)
        elif self.phase == 'l/g':
            Psats = self._Psats(T=T)
            S_P_vapor = -R*log(P/101325.) # Gas-phase ideal pressure contribution (checked repeatedly)
            for i in self.cmps:
                Sg298_to_T_zi = self.zs[i]*self.HeatCapacityGases[i].T_dependent_property_integral_over_T(298.15, T)
                Hvap = self.EnthalpyVaporizations[i](T)
                if Hvap is None:
                    Hvap = 0 # Handle the case of a package predicting a transition past the Tc

                Svap_contrib = -self.xs[i]*(1-self.V_over_F)*Hvap/T
                # Pressure contributions from both phases
                S_P_vapor_i = self.V_over_F*self.ys[i]*S_P_vapor
                S_P_liquid_i = -R*log(Psats[i]/101325.)*(1-self.V_over_F)*self.xs[i]
                S += (Sg298_to_T_zi + Svap_contrib + S_P_vapor_i + S_P_liquid_i)
        return S

        # TODO
        '''Cp_ideal, Cp_real, speed of sound -- or come up with a way for
        mixture to better make calls to the property package. Probably both.
        '''

class GammaPhi(PropertyPackage):
    __TP_cache = None
    __TVF_solve_cache = None
    retention = False
    use_Poynting = False
    use_phis = False
    SUPPORTS_ZERO_FRACTIONS = False

    def __init__(self, VaporPressures=None, Tms=None, Tcs=None, Pcs=None,
                 **kwargs):
        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.N = len(VaporPressures)
        self.cmps = range(self.N)

        self.kwargs = {'VaporPressures': VaporPressures,
                       'Tms': Tms, 'Tcs': Tcs, 'Pcs': Pcs}

    def _P_VF_err(self, T, P, VF, zs):
        P_calc = self.flash_TVF_zs(T=T, VF=VF, zs=zs)[-1]
        return P_calc- P

    def _P_VF_err_2(self, T, P, VF, zs):
        V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs)
        if V_over_F < 0:
            if any(i < 0 for i in xs) or any(i < 0 for i in ys):
                return 5
            return -5
        if any(i < 0 for i in xs) or any(i < 0 for i in ys):
            return -5
        return V_over_F - VF

    def _T_VF_err(self, P, T, zs, Psats, Pmax, V_over_F_goal=1):
        if P < 0 or P > Pmax:
            return 1
        V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs, Psats=Psats, restart=self.__TVF_solve_cache)
        if any(i < 0 for i in xs) or any(i < 0 for i in ys):
            return -100000*(Pmax-P)/Pmax
        self.__TVF_solve_cache = (V_over_F, xs, ys)
        ans = -(V_over_F-V_over_F_goal)
        return ans

    def Ks(self, T, P, xs, ys, Psats):
        gammas = self.gammas(T=T, xs=xs)
        if self.use_phis:
            phis_g = self.phis_g(T=T, P=P, ys=ys)
            phis_l = self.phis_l(T=T, xs=xs)
            if self.use_Poynting:
                Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
                return [K_value(P=P, Psat=Psats[i], gamma=gammas[i],
                                phi_l=phis_l[i], phi_g=phis_g[i], Poynting=Poyntings[i]) for i in self.cmps]
            return [K_value(P=P, Psat=Psats[i], gamma=gammas[i],
                            phi_l=phis_l[i], phi_g=phis_g[i]) for i in self.cmps]
        if self.use_Poynting:
            Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
            Ks = [K_value(P=P, Psat=Psats[i], gamma=gammas[i], Poynting=Poyntings[i]) for i in self.cmps]
            return Ks
        Ks = [K_value(P=P, Psat=Psats[i], gamma=gammas[i]) for i in self.cmps]
        return Ks

    def Poyntings(self, T, P, Psats):
        Vmls = [VolumeLiquid.T_dependent_property(T=T) for VolumeLiquid in self.VolumeLiquids]
#        Vmls = [VolumeLiquid(T=T, P=P) for VolumeLiquid in self.VolumeLiquids]
        return [exp(Vml*(P-Psat)/(R*T)) for Psat, Vml in zip(Psats, Vmls)]

    def dPoyntings_dT(self, T, P, Psats=None):
        if Psats is None:
            Psats = self._Psats(T=T)

        dPsats_dT = [VaporPressure.T_dependent_property_derivative(T=T)
                     for VaporPressure in self.VaporPressures]

        Vmls = [VolumeLiquid.T_dependent_property(T=T) for VolumeLiquid in self.VolumeLiquids]
        dVml_dTs = [VolumeLiquid.T_dependent_property_derivative(T=T)
                    for VolumeLiquid in self.VolumeLiquids]
#        Vmls = [VolumeLiquid(T=T, P=P) for VolumeLiquid in self.VolumeLiquids]
#        dVml_dTs = [VolumeLiquid.TP_dependent_property_derivative_T(T=T, P=P)
#                    for VolumeLiquid in self.VolumeLiquids]

        x0 = 1.0/R
        x1 = 1.0/T

        dPoyntings_dT = []
        for i in self.cmps:
            x2 = Vmls[i]
            x3 = Psats[i]

            x4 = P - x3
            x5 = x1*x2*x4
            dPoyntings_dTi = -x0*x1*(x2*dPsats_dT[i] - x4*dVml_dTs[i] + x5)*exp(x0*x5)
            dPoyntings_dT.append(dPoyntings_dTi)
        return dPoyntings_dT


    def dPoyntings_dP(self, T, P, Psats=None):
        '''from sympy import *
        R, T, P, zi = symbols('R, T, P, zi')
        Vml = symbols('Vml', cls=Function)
        cse(diff(exp(Vml(T)*(P - Psati(T))/(R*T)), P), optimizations='basic')
        '''
        Vmls = [VolumeLiquid(T=T, P=P) for VolumeLiquid in self.VolumeLiquids]
        if Psats is None:
            Psats = self._Psats(T=T)

        dPoyntings_dPs = []
        for i in self.cmps:
            x0 = Vmls[i]/(R*T)
            dPoyntings_dPs.append(x0*exp(x0*(P - Psats[i])))
        return dPoyntings_dPs


    def phis_g(self, T, P, ys):
        return self.eos_mix(T=T, P=P, zs=ys, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas).phis_g

    def phis_l(self, T, xs):
        # TODO change name to include sat in it
#        return [1 for i in xs] # Most models seem to assume this
        # This was correct; could also return phi_g instead of phi_l.
        phis_sat = []
        for obj in self.eos_pure_instances:
            Psat = obj.Psat(T)
            obj = obj.to_TP(T=T, P=Psat)
            # Along the saturation line, may not exist for one phase or the other even though incredibly precise
            try:
                phi = obj.phi_l
            except:
                phi = obj.phi_g
            phis_sat.append(phi)
        return phis_sat


    def fugacity_coefficients_l(self, T, P, xs):
        # DO NOT EDIT _ CORRECT
        gammas = self.gammas(T, xs)
        Psats = self._Psats(T=T)

        if self.use_phis:
            phis = self.phis_l(T=T, xs=xs)
        else:
            phis = [1.0]*self.N

        if self.use_Poynting:
            Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
        else:
            Poyntings = [1.0]*self.N

        P_inv = 1.0/P
        return [gammas[i]*Psats[i]*Poyntings[i]*phis[i]*P_inv
                for i in self.cmps]


    def lnphis_l(self, T, P, xs):
        # DO NOT EDIT _ CORRECT
        return [log(i) for i in self.fugacity_coefficients_l(T, P, xs)]

    def fugacities_l(self, T, P, xs):
        # DO NOT EDIT _ CORRECT
        gammas = self.gammas(T, xs)
        Psats = self._Psats(T=T)
        if self.use_phis:
            phis = self.phis_l(T=T, xs=xs)
        else:
            phis = [1.0]*self.N

        if self.use_Poynting:
            Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
        else:
            Poyntings = [1.0]*self.N
        return [xs[i]*gammas[i]*Psats[i]*Poyntings[i]*phis[i]
                for i in self.cmps]
#        return xs[i]*gammas[i]*Psats[i]*Poynting*phil

    def dphis_dT(self, T, P, xs):
        Psats = self._Psats(T=T)
        gammas = self.gammas(T, xs)


        if self.use_Poynting:
            # Evidence suggests poynting derivatives are not worth calculating
            dPoyntings_dT = [0.0]*self.N#self.dPoyntings_dT(T, P, Psats=Psats)
            Poyntings = self.Poyntings(T, P, Psats)
        else:
            dPoyntings_dT = [0.0]*self.N
            Poyntings = [1.0]*self.N

        dPsats_dT = [VaporPressure.T_dependent_property_derivative(T=T)
                     for VaporPressure in self.VaporPressures]
        dgammas_dT = self.dgammas_dT(T, xs)

        if self.use_phis:
            dphis_l_sat_dT = 0.0
            phis_l_sat = self.phis_l(T, xs)
        else:
            dphis_l_sat_dT = 0.0
            phis_l_sat = [1.0]*self.N

        dphis_dTl = []
        for i in self.cmps:
            x0 = gammas[i]
            x1 = phis_l_sat[i]
            x2 = Psats[i]
            x3 = Poyntings[i]
            x4 = x2*x3
            x5 = x0*x1
            v = (x0*x4*dphis_l_sat_dT + x1*x4*dgammas_dT[i] + x2*x5*dPoyntings_dT[i] + x3*x5*dPsats_dT[i])/P
            dphis_dTl.append(v)
        return dphis_dTl

    def dlnphis_dT(self, T, P, xs):
        dphis_dT = self.dphis_dT(T, P, xs)
        phis = self.fugacity_coefficients_l(T, P, xs)
        return [i/j for i, j in zip(dphis_dT, phis)]

    def _Psats(self, Psats=None, T=None):
        if Psats is None:
            Psats = []
            for i in self.VaporPressures:
#                if i.locked:
                Psats.append(i(T))
#                else:
#                    if T < i.Tmax:
#                        #i.method = None
#                        Psats.append(i(T))
#                    else:
#                        Psats.append(i.extrapolate_tabular(T))
            return Psats
        else:
            return Psats

    def _flash_sequential_substitution_TP(self, T, P, zs, Psats=None, restart=None):
        Psats = self._Psats(Psats=Psats, T=T)
        if self.retention and restart:
            V_over_F, xs, ys = restart
        else:
            Ks = self.Ks(T, P, zs, zs, Psats)
            V_over_F, xs, ys = Rachford_Rice_solution_negative(zs, Ks)
        for i in range(100):
            if any(i < 0 for i in xs):
                xs = zs
            if any(i < 0 for i in ys):
                ys = zs
            Ks = self.Ks(T, P, xs, ys, Psats)
            V_over_F, xs_new, ys_new = Rachford_Rice_solution_negative(zs, Ks)
            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
            xs, ys = xs_new, ys_new
            if err < 1E-7:
                break
        return V_over_F, xs, ys



    def gammas(self, T, xs):
        return [1 for i in self.cmps]

    def VE_l(self):
        r'''Calculates the excess volume of a liquid phase using an
        activity coefficient model as shown in [1]_ and [2]_.

        .. math::
            v^E = \left(\frac{\partial g^E}{\partial P}\right)_{T, xi, xj...}

        In practice, this returns 0 as no pressure-dependent activity models
        are available.

        Returns
        -------
        VE : float
            Excess volume of the liquid phase (0), [m^3/mol]

        Notes
        -----
        The relationship for partial excess molar volume is as follows:

        .. math::
            \frac{\bar v_i^E}{RT} = \left(\frac{\partial \ln \gamma_i}
            {\partial P}\right)_T


        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
           Butterworth-Heinemann, 1985.
        .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process
           Simulation. Weinheim, Germany: Wiley-VCH, 2012.
        '''
        return 0.0

    def GE_l(self, T, xs):
        r'''Calculates the excess Gibbs energy of a liquid phase using an
        activity coefficient model as shown in [1]_ and [2]_.

        .. math::
            g_E = RT\sum_i x_i \ln \gamma_i

        Parameters
        ----------
        T : float
            Temperature of the system, [K]
        xs : list[float]
            Mole fractions of the liquid phase of the system, [-]

        Returns
        -------
        GE : float
            Excess Gibbs energy of the liquid phase, [J/mol]

        Notes
        -----
        It is possible to directly calculate GE in some activity coefficient
        models, without calculating individual activity coefficients of the
        species.

        Note also the relationship of the expressions for partial excess Gibbs
        energies:

        .. math::
            \bar g_i^E = RT\ln(\gamma_i)

            g^E = \sum_i x_i \bar g_i^E

        Most activity coefficient models are pressure independent, which leads
        to the relationship where excess Helmholtz energy is the same as the
        excess Gibbs energy.

        .. math::
            G^E = A^E

        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
           Butterworth-Heinemann, 1985.
        .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process
           Simulation. Weinheim, Germany: Wiley-VCH, 2012.
        '''
        gammas = self.gammas(T=T, xs=xs)
        return R*T*sum(xi*log(gamma) for xi, gamma in zip(xs, gammas))

    def HE_l(self, T, xs):
        r'''Calculates the excess enthalpy of a liquid phase using an
        activity coefficient model as shown in [1]_ and [2]_. This is an
        expression of the Gibbs-Helmholz relation.

        .. math::
            \frac{-h^E}{T^2} = \frac{\partial (g^E/T)}{\partial T}

        Parameters
        ----------
        T : float
            Temperature of the system, [K]
        xs : list[float]
            Mole fractions of the liquid phase of the system, [-]

        Returns
        -------
        HE : float
            Excess enthalpy of the liquid phase, [J/mol]

        Notes
        -----
        It is possible to obtain analytical results for some activity
        coefficient models; this method provides only the `derivative`
        method of scipy with its default parameters to obtain a numerical
        result.

        Note also the relationship of the expressions for partial excess
        enthalpy:

        .. math::
            \left(\frac{\partial \ln \gamma_i}{\partial (1/T))}\right)
                = \frac{\bar h_i^E}{R}

            \left(\frac{\partial \ln \gamma_i}{\partial T}\right)
            = -\frac{\bar h_i^E}{RT^2}


        Most activity coefficient models are pressure independent, so the Gibbs
        Duhem expression only has a temperature relevance.

        .. math::
            \sum_i x_i d \ln \gamma_i = - \frac{H^{E}}{RT^2} dT
            + \frac{V^E}{RT} dP

        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
           Butterworth-Heinemann, 1985.
        .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process
           Simulation. Weinheim, Germany: Wiley-VCH, 2012.
        '''
        to_diff = lambda T: self.GE_l(T, xs)/T
        return -derivative(to_diff, T)*T**2

    def SE_l(self, T, xs):
        r'''Calculates the excess entropy of a liquid phase using an
        activity coefficient model as shown in [1]_ and [2]_.

        .. math::
            s^E = \frac{h^E - g^E }{T}

        Parameters
        ----------
        T : float
            Temperature of the system, [K]
        xs : list[float]
            Mole fractions of the liquid phase of the system, [-]

        Returns
        -------
        SE : float
            Excess entropy of the liquid phase, [J/mol/K]

        Notes
        -----
        It is possible to obtain analytical results for some activity
        coefficient models; this method provides only the `derivative`
        method of scipy with its default parameters to obtain a numerical
        result for the excess enthalpy, although the excess Gibbs energy
        is exact.

        Note also the relationship of the expressions for partial excess
        entropy:

        .. math::
            S_i^E = -R\left(T \frac{\partial \ln \gamma_i}{\partial T}
            + \ln \gamma_i\right)


        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
           Butterworth-Heinemann, 1985.
        .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process
           Simulation. Weinheim, Germany: Wiley-VCH, 2012.
        '''
        return (self.HE_l(T, xs) - self.GE_l(T, xs))/T

    def CpE_l(self, T, xs):
        r'''Calculates the excess heat capacity of a liquid phase using an
        activity coefficient model as shown in [1]_ and [2]_.

        .. math::
             C_{p,l}^E = \left(\frac{\partial H^E}{\partial T}\right)_{p, x}

        Parameters
        ----------
        T : float
            Temperature of the system, [K]
        xs : list[float]
            Mole fractions of the liquid phase of the system, [-]

        Returns
        -------
        CpE : float
            Excess heat capacity of the liquid phase, [J/mol/K]

        Notes
        -----
        This method provides only the `derivative`
        method of scipy with its default parameters to obtain a numerical
        result for the excess enthalpy as well as the derivative of excess
        enthalpy.

        References
        ----------
        .. [1] Walas, Stanley M. Phase Equilibria in Chemical Engineering.
           Butterworth-Heinemann, 1985.
        .. [2] Gmehling, Jurgen. Chemical Thermodynamics: For Process
           Simulation. Weinheim, Germany: Wiley-VCH, 2012.
        '''
        to_diff = lambda T : self.HE_l(T, xs)
        return derivative(to_diff, T)

    def gammas_infinite_dilution(self, T):
        gamma_infs = []
        for i in self.cmps:
            xs = [1./(self.N - 1)]*self.N
            xs[i] = 0
            gamma_inf = self.gammas(T=T, xs=xs)[i]
            gamma_infs.append(gamma_inf)
        return gamma_infs

    def H_dep_g(self, T, P, ys):
        if not self.use_phis:
            return 0.0
        e = self.eos_mix(T=T, P=P, zs=ys, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas)
        try:
            return e.H_dep_g
        except AttributeError:
            # This really is the correct approach
            return e.H_dep_l


    def S_dep_g(self, T, P, ys):
        if not self.use_phis:
            return 0.0
        e = self.eos_mix(T=T, P=P, zs=ys, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas)
        try:
            return e.S_dep_g
        except AttributeError:
            # This really is the correct approach
            return e.S_dep_l

    def enthalpy_excess(self, T, P, V_over_F, xs, ys):
        # Does this handle the transition without a discontinuity?
        H = 0
        if self.phase == 'g':
            H += self.H_dep_g(T=T, P=P, ys=ys)
        elif self.phase == 'l':
            H += self.HE_l(T=T, xs=xs)
        elif self.phase == 'l/g':
            HE_l = self.HE_l(T=T, xs=xs)
            HE_g = self.H_dep_g(T=T, P=P, ys=ys)
            H += (1. - V_over_F)*HE_l + HE_g*V_over_F
        return H


    def entropy_excess(self, T, P, V_over_F, xs, ys):
        # Does this handle the transition without a discontinuity?
        S = 0
        if self.phase == 'g':
            S += self.S_dep_g(T=T, P=P, ys=ys)
        elif self.phase == 'l':
            S += self.SE_l(T=T, xs=xs)
        elif self.phase == 'l/g':
            SE_l = self.SE_l(T=T, xs=xs)
            SE_g = self.S_dep_g(T=T, P=P, ys=ys)
            S += (1. - V_over_F)*SE_l + SE_g*V_over_F
        return S


    def P_bubble_at_T(self, T, zs, Psats=None):
        # Returns P_bubble; only thing easy to calculate
        Psats = self._Psats(Psats, T)
        cmps, N = self.cmps, self.N

        # If there is one component, return at the saturation line
        if self.N == 1:
            return Psats[0]

        gammas = self.gammas(T=T, xs=zs)
        P = sum([gammas[i]*zs[i]*Psats[i] for i in cmps])
        if self.use_Poynting and not self.use_phis:
            # This is not really necessary; and 3 is more than enough iterations
            for i in range(3):
                Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
                P = sum([gammas[i]*zs[i]*Psats[i]*Poyntings[i] for i in cmps])
        elif self.use_phis:
            # should use pressure derivatives to get solution here
            # with the MM method
            for i in range(5):
                phis_l = self.phis_l(T=T, xs=zs)
                if self.use_Poynting:
                    Poyntings = self.Poyntings(T=T, P=P, Psats=Psats)
                    P = sum([gammas[i]*zs[i]*Psats[i]*Poyntings[i]*phis_l[i] for i in cmps])
                else:
                    P = sum([gammas[i]*zs[i]*Psats[i]*phis_l[i] for i in cmps])



        # TODO: support equations of state, once you get that figured out.
        P_inv = 1.0/P
        Ks = [gammas[i]*Psats[i]*P_inv for i in cmps]
        ys = [zs[i]*Ks[i] for i in cmps]

        return P, zs, ys, Ks

    def P_dew_at_T(self, T, zs, Psats=None):
        Psats = self._Psats(Psats, T)

        # If there is one component, return at the saturation line
        if self.N == 1:
            return Psats[0]

        Pmax, _, _, _ = self.P_bubble_at_T(T, zs, Psats)
        diff = 1E-7
        # EOSs do not solve at very low pressure
        if self.use_phis:
            Pmin = max(Pmax*diff, 1)
        else:
            Pmin = Pmax*diff
        P_dew = brenth(self._T_VF_err, Pmin, Pmax, args=(T, zs, Psats, Pmax, 1))
        self.__TVF_solve_cache = None
        return P_dew
#        try:
#            return brent(self._dew_P_UNIFAC_err, args=(T, zs, Psats, Pmax), brack=(Pmax*diff, Pmax*(1-diff), Pmax))
#        except:
#        return golden(self._dew_P_UNIFAC_err, args=(T, zs, Psats, Pmax), brack=(Pmax, Pmax*(1-diff)))
#
    def flash_TVF_zs(self, T, VF, zs):
        assert 0 <= VF <= 1
        Psats = self._Psats(T=T)

        # handle one component
        if self.N == 1:
            return 'l/g', [1.0], [1.0], VF, Psats[0]

        Pbubble, _, ys, Ks = self.P_bubble_at_T(T=T, zs=zs, Psats=Psats)
        if VF == 0:
            P = Pbubble
            V_over_F = VF
            xs = zs
        else:
            diff = 1E-7
            Pmax = Pbubble
            # EOSs do not solve at very low pressure
            if self.use_phis:
                Pmin = max(Pmax*diff, 1)
            else:
                Pmin = Pmax*diff

            P = brenth(self._T_VF_err, Pmin, Pmax, args=(T, zs, Psats, Pmax, VF))
            self.__TVF_solve_cache = None
#            P = brenth(self._T_VF_err, Pdew, Pbubble, args=(T, VF, zs, Psats))
            V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs, Psats=Psats)
        return 'l/g', xs, ys, V_over_F, P

    def flash_TP_zs(self, T, P, zs):
        Psats = self._Psats(T=T)
        Pbubble, _, ys, Ks = self.P_bubble_at_T(T=T, zs=zs, Psats=Psats)
        if P >= Pbubble:
            return 'l', zs, ys, 0 # return ys
        Pdew = self.P_dew_at_T(T=T, zs=zs, Psats=Psats)
        if P <= Pdew:
            # phase, ys, xs, quality
            return 'g', None, zs, 1
        else:
            V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs, Psats=Psats)
            return 'l/g', xs, ys, V_over_F


    def flash_PVF_zs(self, P, VF, zs):
        if self.N == 1:
            Tsats = self._Tsats(P)
            return 'l/g', [1.0], [1.0], VF, Tsats[0]
        try:
            # In some caases, will find a false root - resort to iterations which
            # are always between Pdew and Pbubble if this happens
            T = brenth(self._P_VF_err_2, min(self.Tms), min(self.Tcs), args=(P, VF, zs), maxiter=500)
            V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs)
            assert abs(V_over_F-VF) < 1E-6
        except:
            T = ridder(self._P_VF_err, min(self.Tms), min(self.Tcs), args=(P, VF, zs))
            V_over_F, xs, ys = self._flash_sequential_substitution_TP(T=T, P=P, zs=zs)
        return 'l/g', xs, ys, V_over_F, T


    def dew_T_Michelsen_Mollerup(self, T_guess, P, zs, maxiter=200,
                                 xtol=1E-10, info=None, xs_guess=None,
                                 max_step_damping=100.0, near_critical=False,
                                 trivial_solution_tol=1e-4):
        # Does not have any formulation available
        # According to the following, convergence does not occur with newton's method near the critical point
        # It recommends some sort of substitution method
        # Accelerated successive substitution schemes for bubble-point and dew-point calculations
        N = len(zs)
        cmps = range(N)
        xs = zs if xs_guess is None else xs_guess
        def lnphis_and_derivatives(T_guess):

            if self.use_phis:
                ln_phis_g = [log(i) for i in self.phis_g(T=T_guess, P=P, ys=zs)]
            else:
                ln_phis_g = [1.0]*N
            dlnphis_dT_g = [0.0]*N

            ln_phis_l = self.lnphis_l(T_guess, P, xs)
            dlnphis_dT_l = self.dlnphis_dT(T_guess, P, xs)


            return ln_phis_l, ln_phis_g, dlnphis_dT_l, dlnphis_dT_g

        T_guess_old = None
        successive_fails = 0
        for i in range(maxiter):
            try:
                ln_phis_l, ln_phis_g, dlnphis_dT_l, dlnphis_dT_g = lnphis_and_derivatives(T_guess)
                successive_fails = 0
            except Exception as e:
                print(e)
                if T_guess_old is None:
                    raise ValueError("Could not calculate liquid and vapor conditions at provided initial temperature %s K" %(T_guess))
                successive_fails += 1
                if successive_fails >= 2:
                    raise ValueError("Stopped convergence procedure after multiple bad steps")
                T_guess = T_guess_old + copysign(min(max_step_damping, abs(step)), step)
#                print('fail - new T guess', T_guess)
                ln_phis_l, ln_phis_g, dlnphis_dT_l, dlnphis_dT_g = lnphis_and_derivatives(T_guess)

            Ks = [exp(a - b) for a, b in zip(ln_phis_l, ln_phis_g)]
            xs = [zs[i]/Ks[i] for i in cmps]
            f_k = sum([xs[i] for i in cmps]) - 1.0

            dfk_dT = 0.0
            for i in cmps:
                dfk_dT += xs[i]*(dlnphis_dT_g[i] - dlnphis_dT_l[i])

            T_guess_old = T_guess
            step = -f_k/dfk_dT

#            print(xs, T_guess, step, dfk_dT)

            if near_critical:
                T_guess = T_guess + copysign(min(max_step_damping, abs(step)), step)
            else:
                T_guess = T_guess + step

            if near_critical:
                comp_difference = sum([abs(zi - xi) for zi, xi in zip(zs, xs)])
                if comp_difference < trivial_solution_tol:
                    raise ValueError("Converged to trivial condition, compositions of both phases equal")

            x_sum = sum(xs)
            xs = [x/x_sum for x in xs]

            if info is not None:
                info[:] = xs, zs, Ks, 1.0
            if abs(T_guess - T_guess_old) < xtol:
                T_guess = T_guess_old
                break


        if abs(T_guess - T_guess_old) > xtol:
            raise ValueError("Did not converge to specified tolerance")

        return T_guess

    def HE_l2(self, T, xs):
        # Just plain excess enthalpy here
        '''f = symbols('f', cls=Function)
        T = symbols('T')
        simplify(-T**2*diff(f(T)/T, T))
        '''
        dGE_dT = self.dGE_dT(T, xs)
        GE = self.GE_l2(T, xs)
        return -T*dGE_dT + GE

    def dHE_dx(self, T, xs):
        # Derived by hand taking into account the expression for excess enthalpy
        d2GE_dTdxs = self.d2GE_dTdxs(T, xs)
        dGE_dxs = self.dGE_dxs(T, xs)
        return [-T*d2GE_dTdxs[i] + dGE_dxs[i] for i in self.cmps]

    def dHE_dn(self, T, xs):
        return dxs_to_dns(self.dHE_dx(T, xs), xs)

    def dnHE_dn(self, T, xs):
        return dxs_to_dn_partials(self.dHE_dx(T, xs), xs, self.HE_l2(T, xs))

    def dSE_dx(self, T, xs):
        # Derived by hand.
        dGE_dxs = self.dGE_dxs(T, xs)
        dHE_dx = self.dHE_dx(T, xs)
        T_inv = 1.0/T
        return [T_inv*(dHE_dx[i] - dGE_dxs[i]) for i in self.cmps]

    def dSE_dn(self, T, xs):
        return dxs_to_dns(self.dSE_dx(T, xs), xs)

    def dnSE_dn(self, T, xs):
        return dxs_to_dn_partials(self.dSE_dx(T, xs), xs, self.SE_l(T, xs))

    def dGE_dns(self, T, xs):
        # Mole number derivatives
        return dxs_to_dns(self.dGE_dxs(T, xs), xs)

    def dnGE_dns(self, T, xs):
        return dxs_to_dn_partials(self.dGE_dxs(T, xs), xs, self.GE_l2(T, xs))

    def gammas2(self, T, xs):
        r'''
        .. math::
            \gamma_i = \exp\left(\frac{\frac{\partial n_i G^E}{\partial n_i }}{RT}\right)
        '''
        # Matches the gamma formulation perfectly
        dG_dxs = self.dGE_dxs(T, xs)
        GE = self.GE_l2(T, xs)
        dG_dns = dxs_to_dn_partials(dG_dxs, xs, GE)
        RT_inv = 1.0/(R*T)
        return [exp(i*RT_inv) for i in dG_dns]

    def dgammas_dx(self, T, xs):
#        dnGE_dni = self.dnGE_dns(T, xs)
        gammas = self.gammas2(T, xs)

        cmps = self.cmps

        d2GE_dxixjs = self.d2GE_dxixjs(T, xs)
        d2nGE_dxjnis = d2xs_to_dxdn_partials(d2GE_dxixjs, xs)

        RT_inv = 1.0/(R*T)

        matrix = []
        for i in cmps:
            row = []
            gammai = gammas[i]
            for j in cmps:
                v = gammai*d2nGE_dxjnis[i][j]*RT_inv
                row.append(v)
            matrix.append(row)
        return matrix

    def dgammas_dT(self, T, xs):
        r'''
        .. math::
            \frac{\partial \gamma_i}{\partial T} =
            \left(\frac{\frac{\partial^2 n G^E}{\partial T \partial n_i}}{RT} -
            \frac{{\frac{\partial n_i G^E}{\partial n_i }}}{RT^2}\right)
             \exp\left(\frac{\frac{\partial n_i G^E}{\partial n_i }}{RT}\right)

        from sympy import *
        R, T = symbols('R, T')
        f = symbols('f', cls=Function)
        diff(exp(f(T)/(R*T)), T)
        '''
        d2nGE_dTdns = self.d2nGE_dTdns(T, xs)

        dG_dxs = self.dGE_dxs(T, xs)
        GE = self.GE_l2(T, xs)
        dG_dns = dxs_to_dn_partials(dG_dxs, xs, GE)

        RT_inv = 1.0/(R*T)
        return [(d2nGE_dTdns[i]*RT_inv - dG_dns[i]*RT_inv/T)*exp(dG_dns[i]*RT_inv)
                for i in self.cmps]

    def d2GE_dTdns(self, T, xs):
        return dxs_to_dns(self.d2GE_dTdxs(T, xs), xs)


    def d2nGE_dTdns(self, T, xs):
        # needed in gammas temperature derivatives
        dGE_dT = self.dGE_dT(T, xs)
        d2GE_dTdns = self.d2GE_dTdns(T, xs)
        return dns_to_dn_partials(d2GE_dTdns, dGE_dT)

    def dHE_dT(self, T, xs):
        # excess enthalpy temperature derivative
        '''from sympy import *
        f = symbols('f', cls=Function)
        T = symbols('T')
        diff(simplify(-T**2*diff(f(T)/T, T)), T)
        '''
        return -T*self.dGE2_dT2(T, xs)

    def dSE_dT(self, T, xs):
        '''from sympy import *
        T = symbols('T')
        G, H = symbols('G, H', cls=Function)
        S = (H(T) - G(T))/T
        print(diff(S, T))
        # (-Derivative(G(T), T) + Derivative(H(T), T))/T - (-G(T) + H(T))/T**2
        '''
        # excess entropy temperature derivative
        H = self.HE_l2(T, xs)
        dHdT = self.dHE_dT(T, xs)
        dGdT = self.dGE_dT(T, xs)
        G = self.GE2(T, xs)
        return (-dGdT + dHdT)/T - (-G + H)/(T*T)


class GammaPhiCaloric(GammaPhi, IdealCaloric):

    def _post_flash(self):
        # Cannot derive other properties with this
        self.Hm = self.enthalpy_Cpg_Hvap() + self.enthalpy_excess(T=self.T, P=self.P, V_over_F=self.V_over_F, xs=self.xs, ys=self.ys)
        self.Sm = self.entropy_Cpg_Hvap() + self.entropy_excess(T=self.T, P=self.P, V_over_F=self.V_over_F, xs=self.xs, ys=self.ys)

        self.Gm = self.Hm - self.T*self.Sm if (self.Hm is not None and self.Sm is not None) else None




    def __init__(self, VaporPressures=None, Tms=None, Tbs=None, Tcs=None,
                 Pcs=None, omegas=None, VolumeLiquids=None, eos=None,
                 eos_mix=None, HeatCapacityLiquids=None, HeatCapacityGases=None,
                 EnthalpyVaporizations=None, **kwargs):
        self.cmps = range(len(VaporPressures))
        self.N = len(VaporPressures)
        self.VaporPressures = VaporPressures
        self.omegas = omegas

        self.Tms = Tms
        self.Tbs = Tbs
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.HeatCapacityLiquids = HeatCapacityLiquids
        self.HeatCapacityGases = HeatCapacityGases
        self.EnthalpyVaporizations = EnthalpyVaporizations
        self.VolumeLiquids = VolumeLiquids
        self.eos = eos
        self.eos_mix = eos_mix

        if eos:
            self.eos_pure_instances = [eos(Tc=Tcs[i], Pc=Pcs[i], omega=omegas[i], T=Tcs[i]*0.5, P=Pcs[i]*0.1) for i in self.cmps]

        self.kwargs = {'VaporPressures': VaporPressures,
                       'Tms': Tms, 'Tbs': Tbs, 'Tcs': Tcs, 'Pcs': Pcs,
                       'HeatCapacityLiquids': HeatCapacityLiquids,
                       'HeatCapacityGases': HeatCapacityGases,
                       'EnthalpyVaporizations': EnthalpyVaporizations,
                       'VolumeLiquids': VolumeLiquids,
                       'eos': eos, 'eos_mix': eos_mix,
                       'omegas': omegas}


class Nrtl(GammaPhiCaloric):
    def GE2(self, T, xs):
        cmps = self.cmps
        taus = self.taus(T)
        Gs = self.Gs(T, taus=taus)

        tot = 0
        for i in self.cmps:
            sum1 = 0.0
            sum2 = 0.0
            for j in self.cmps:
                sum1 += Gs[j][i]*xs[j]
                sum2 += taus[j][i]*Gs[j][i]*xs[j] # dup
            t1 = sum2/sum1
            tot += xs[i]*t1
        return T*R*tot



    def dGE_dxs(self, T, xs):
        '''
        from sympy import *
        N = 3
        R, T = symbols('R, T')
        x0, x1, x2 = symbols('x0, x1, x2')
        xs = [x0, x1, x2]

        tau00, tau01, tau02, tau10, tau11, tau12, tau20, tau21, tau22 = symbols(
            'tau00, tau01, tau02, tau10, tau11, tau12, tau20, tau21, tau22', cls=Function)
        tau_ijs = [[tau00(T), tau01(T), tau02(T)],
                   [tau10(T), tau11(T), tau12(T)],
                   [tau20(T), tau21(T), tau22(T)]]


        G00, G01, G02, G10, G11, G12, G20, G21, G22 = symbols(
            'G00, G01, G02, G10, G11, G12, G20, G21, G22', cls=Function)
        G_ijs = [[G00(T), G01(T), G02(T)],
                   [G10(T), G11(T), G12(T)],
                   [G20(T), G21(T), G22(T)]]
        ge = 0
        for i in [2]:#range(0):
            num = 0
            den = 0
            for j in range(N):
                num += tau_ijs[j][i]*G_ijs[j][i]*xs[j]
                den += G_ijs[j][i]*xs[j]
            ge += xs[i]*num/den
        ge = ge#*R*T
        diff(ge, x1), diff(ge, x2)
        '''
        cmps = self.cmps
        taus = self.taus(T)
        alphas = self.alphas(T)
        Gs = self.Gs(T)

        dGE_dxs = []

        for k in cmps:
            # k is what is being differentiated
            tot = 0
            for i in cmps:

                # sum1 in other places
                sum1 = 0.0
                sum2 = 0.0
                for j in cmps:
                    sum1 += xs[j]*Gs[j][i]
                    sum2 += xs[j]*taus[j][i]*Gs[j][i] # sum2 in other places

                term0 = xs[i]*Gs[k][i]*taus[k][i]/sum1
                term1 = -xs[i]*Gs[k][i]*sum2/(sum1*sum1)


                tot += term0 + term1
                if i == k:
                    tot += sum2/sum1
            tot *= R*T
            dGE_dxs.append(tot)
        return dGE_dxs



    def dGE_dT(self, T, xs):
        '''from sympy import *
        R, T, x = symbols('R, T, x')
        g, tau = symbols('g, tau', cls=Function)
        m, n, o = symbols('m, n, o', cls=Function)
        r, s, t = symbols('r, s, t', cls=Function)
        u, v, w = symbols('u, v, w', cls=Function)
        diff(T* (m(T)*n(T) + r(T)*s(T) + u(T)*v(T))/(o(T) + t(T) + w(T)), T)
        '''
        # DO NOT EDIT _ WORKING
        cmps = self.cmps
        taus = self.taus(T)
        dtaus_dT = self.dtaus_dT(T)

        alphas = self.alphas(T)
        dalphas_dT = self.dalphas_dT(T)

        Gs = self.Gs(T)
        dGs_dT = self.dGs_dT(T)

        tot = 0
        for i in self.cmps:
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            sum4 = 0.0
            sum5 = 0.0
            for j in self.cmps:
                tauji = taus[j][i]

                Gjixj = Gs[j][i]*xs[j]
                dGjidTxj = dGs_dT[j][i]*xs[j]

                sum1 += Gjixj
                sum2 += tauji*Gjixj
                sum3 += dGjidTxj
                sum4 += tauji*dGjidTxj
                sum5 += dtaus_dT[j][i]*Gjixj

            t1 = sum2/sum1 - T*(sum2*sum3)/(sum1*sum1) + T*(sum4 + sum5)/sum1
            tot += xs[i]*t1
        return R*tot

    def dGE2_dT2(self, T, xs):
        '''from sympy import *
        R, T, x = symbols('R, T, x')
        g, tau = symbols('g, tau', cls=Function)
        m, n, o = symbols('m, n, o', cls=Function)
        r, s, t = symbols('r, s, t', cls=Function)
        u, v, w = symbols('u, v, w', cls=Function)

        (diff(T*(m(T)*n(T) + r(T)*s(T))/(o(T) + t(T)), T, 2))
        '''
        cmps = self.cmps
        taus = self.taus(T)
        dtaus_dT = self.dtaus_dT(T)
        d2taus_dT2 = self.d2taus_dT2(T)

        alphas = self.alphas(T)
        dalphas_dT = self.dalphas_dT(T)

        Gs = self.Gs(T)
        dGs_dT = self.dGs_dT(T)
        d2Gs_dT2 = self.d2Gs_dT2(T)

        tot = 0
        for i in self.cmps:
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            sum4 = 0.0
            sum5 = 0.0

            sum6 = 0.0
            sum7 = 0.0
            sum8 = 0.0
            sum9 = 0.0
            for j in self.cmps:
                tauji = taus[j][i]
                dtaus_dTji = dtaus_dT[j][i]

                Gjixj = Gs[j][i]*xs[j]
                dGjidTxj = dGs_dT[j][i]*xs[j]
                d2GjidT2xj = xs[j]*d2Gs_dT2[j][i]

                sum1 += Gjixj
                sum2 += tauji*Gjixj
                sum3 += dGjidTxj

                sum4 += tauji*dGjidTxj
                sum5 += dtaus_dTji*Gjixj

                sum6 += d2GjidT2xj

                sum7 += tauji*d2GjidT2xj

                sum8 += Gjixj*d2taus_dT2[j][i]

                sum9 += dGjidTxj*dtaus_dTji

            term1 = -T*sum2*(sum6 - 2.0*sum3*sum3/sum1)/sum1
            term2 = T*(sum7 + sum8 + 2.0*sum9)
            term3 = -2*T*(sum3*(sum4 + sum5))/sum1
            term4 = -2.0*(sum2*sum3)/sum1
            term5 = 2*(sum4 + sum5)

            tot += xs[i]*(term1 + term2 + term3 + term4 + term5)/sum1
        return R*tot


    def taus(self, T):
        r'''Calculate the `tau` terms for the NRTL model for a specified
        temperature.

        .. math::
            \tau_{ij}=A_{ij}+\frac{B_{ij}}{T}+E_{ij}\ln T + F_{ij}T
            + \frac{G_{ij}}{T^2} + H_{ij}{T^2}


        These `tau ij` values (and the coefficients) are NOT symmetric
        normally.
        '''
#        tau_coeffs = self.tau_coeffs
        # Zero coefficients if not specified by user
#        if tau_coeffs is None:
#            return self.zero_coeffs


        tau_coeffs_A = self.tau_coeffs_A
        tau_coeffs_B = self.tau_coeffs_B
        tau_coeffs_E = self.tau_coeffs_E
        tau_coeffs_F = self.tau_coeffs_F
        tau_coeffs_G = self.tau_coeffs_G
        tau_coeffs_H = self.tau_coeffs_H

        if tau_coeffs_A is None:
            return self.zero_tau_coeffs_Acoeffs

        N, cmps = self.N, self.cmps
        T2 = T*T
        Tinv = 1.0/T
        T2inv = Tinv*Tinv
        logT = log(T)

        # initialize the matrix to be A
        taus = [list(l) for l in tau_coeffs_A]
        for i in cmps:
            tau_coeffs_Bi = tau_coeffs_B[i]
            tau_coeffs_Ei = tau_coeffs_E[i]
            tau_coeffs_Fi = tau_coeffs_F[i]
            tau_coeffs_Gi = tau_coeffs_G[i]
            tau_coeffs_Hi = tau_coeffs_H[i]
            tausi = taus[i]
            for j in cmps:
                tausi[j] += tau_coeffs_Bi[j]*Tinv + tau_coeffs_Ei[j]*logT + tau_coeffs_Fi[j]*T + tau_coeffs_Gi[j]*T2inv + tau_coeffs_Hi[j]*T2
#                tausi[j] = tau_coeffs_Ai[j] + tau_coeffs_Bi[j]*Tinv + tau_coeffs_Ei[j]*logT + tau_coeffs_Fi[j]*T + tau_coeffs_Gi[j]*T2inv + tau_coeffs_Hi[j]*T2
#                coeffs = coeffsi[j]
#                tausi[j] = coeffs[0] + coeffs[1]*Tinv + coeffs[2]*logT + coeffs[3]*T + coeffs[4]*T2inv + coeffs[5]*T2
        # This approach may be better for the cache

#        for i in cmps:
#            tausi = taus[i]
#            tau_coeffs_Bi = tau_coeffs_B[i]
#            for j in cmps:
#                tausi[j] += tau_coeffs_Bi[j]*Tinv
#        for i in cmps:
#            tausi = taus[i]
#            tau_coeffs_Ei = tau_coeffs_E[i]
#            for j in cmps:
#                tausi[j] += tau_coeffs_Ei[j]*logT
#        for i in cmps:
#            tausi = taus[i]
#            tau_coeffs_Fi = tau_coeffs_F[i]
#            for j in cmps:
#                tausi[j] += tau_coeffs_Fi[j]*T
#        for i in cmps:
#            tausi = taus[i]
#            tau_coeffs_Gi = tau_coeffs_G[i]
#            for j in cmps:
#                tausi[j] += tau_coeffs_Gi[j]*T2inv
#        for i in cmps:
#            tausi = taus[i]
#            tau_coeffs_Hi = tau_coeffs_H[i]
#            for j in cmps:
#                tausi[j] += tau_coeffs_Hi[j]*T2
        return taus

    def dtaus_dT(self, T):
        r'''Calculate the temperature derivative of the `tau` terms for the
        NRTL model for a specified temperature.

        .. math::
            \frac{\partial \tau_{ij}} {\partial T}_{P, x_i} =
            - \frac{B_{ij}}{T^{2}} + \frac{E_{ij}}{T} + F_{ij}
            - \frac{2 G_{ij}}{T^{3}} + 2 H_{ij} T

        These `tau ij` values (and the coefficients) are NOT symmetric
        normally.
        '''
        # Believed all correct but not tested
        tau_coeffs_B = self.tau_coeffs_B
        tau_coeffs_E = self.tau_coeffs_E
        tau_coeffs_F = self.tau_coeffs_F
        tau_coeffs_G = self.tau_coeffs_G
        tau_coeffs_H = self.tau_coeffs_H
        cmps = self.cmps

        Tinv = 1.0/T
        nT2inv = -Tinv*Tinv
        n2T3inv = 2.0*nT2inv*Tinv
        T2 = T + T

        dtaus_dT = [list(l) for l in tau_coeffs_F]
        for i in cmps:
            tau_coeffs_Bi = tau_coeffs_B[i]
            tau_coeffs_Ei = tau_coeffs_E[i]
            tau_coeffs_Fi = tau_coeffs_F[i]
            tau_coeffs_Gi = tau_coeffs_G[i]
            tau_coeffs_Hi = tau_coeffs_H[i]
            dtaus_dTi = dtaus_dT[i]
            for j in cmps:
                dtaus_dTi[j] += (nT2inv*tau_coeffs_Bi[j] + Tinv*tau_coeffs_Ei[j]
                + n2T3inv*tau_coeffs_Gi[j] + T2*tau_coeffs_Hi[j])

        return dtaus_dT

    def d2taus_dT2(self, T):
        r'''Calculate the second temperature derivative of the `tau` terms for
        the NRTL model for a specified temperature.

        .. math::
            \frac{\partial^2 \tau_{ij}} {\partial T^2}_{P, x_i} =
            \frac{2 B_{ij}}{T^{3}} - \frac{E_{ij}}{T^{2}} + \frac{6 G_{ij}}
            {T^{4}} + 2 H_{ij}

        These `tau ij` values (and the coefficients) are NOT symmetric
        normally.
        '''
        tau_coeffs_B = self.tau_coeffs_B
        tau_coeffs_E = self.tau_coeffs_E
        tau_coeffs_G = self.tau_coeffs_G
        tau_coeffs_H = self.tau_coeffs_H
        cmps = self.cmps

        d2taus_dT2 = [[h + h for h in l] for l in tau_coeffs_H]

        Tinv = 1.0/T
        Tinv2 = Tinv*Tinv

        T3inv2 = 2.0*(Tinv2*Tinv)
        nT2inv = -Tinv*Tinv
        T4inv6 = 6.0*(Tinv2*Tinv2)
        for i in cmps:
            tau_coeffs_Bi = tau_coeffs_B[i]
            tau_coeffs_Ei = tau_coeffs_E[i]
            tau_coeffs_Gi = tau_coeffs_G[i]
            d2taus_dT2i = d2taus_dT2[i]
            for j in cmps:
                d2taus_dT2i[j] += (T3inv2*tau_coeffs_Bi[j]
                                   + nT2inv*tau_coeffs_Ei[j]
                                   + T4inv6*tau_coeffs_Gi[j])
        return d2taus_dT2

    def d3taus_dT3(self, T):
        r'''Calculate the third temperature derivative of the `tau` terms for
        the NRTL model for a specified temperature.

        .. math::
            \frac{\partial^3 \tau_{ij}} {\partial T^3}_{P, x_i} =
            - \frac{6 B_{ij}}{T^{4}} + \frac{2 E_{ij}}{T^{3}}
            - \frac{24 G_{ij}}{T^{5}}

        These `tau ij` values (and the coefficients) are NOT symmetric
        normally.
        '''
        tau_coeffs_B = self.tau_coeffs_B
        tau_coeffs_E = self.tau_coeffs_E
        tau_coeffs_G = self.tau_coeffs_G
        N, cmps = self.N, self.cmps

        d3taus_dT3 = [[0.0]*N for i in cmps]

        Tinv = 1.0/T
        T2inv = Tinv*Tinv

        nT4inv6 = -6.0*T2inv*T2inv
        T3inv2 = 2.0*T2inv*Tinv
        T5inv24 = -24.0*(T2inv*T2inv*Tinv)

        for i in cmps:
            tau_coeffs_Bi = tau_coeffs_B[i]
            tau_coeffs_Ei = tau_coeffs_E[i]
            tau_coeffs_Gi = tau_coeffs_G[i]
            d3taus_dT3i = d3taus_dT3[i]
            for j in cmps:
                d3taus_dT3i[j] = (nT4inv6*tau_coeffs_Bi[j]
                                  + T3inv2*tau_coeffs_Ei[j]
                                  + T5inv24*tau_coeffs_Gi[j])
        return d3taus_dT3



    def alphas(self, T):
        '''Calculates the `alpha` terms in the NRTL model for a specified
        temperature.

        .. math::
            \alpha_{ij}=c_{ij}+d_{ij}T

        `alpha` values (and therefore `cij` and `dij` are normally symmetrical;
        but this is not strictly required.

        Some sources suggest the c term should be fit to a given system; but
        the `d` term should be fit for an entire chemical family to avoid
        overfitting.

        Recommended values for `cij` according to one source are:

        0.30 Nonpolar substances with nonpolar substances; low deviation from ideality.
        0.20 Hydrocarbons that are saturated interacting with polar liquids that do not associate, or systems that for multiple liquid phases which are immiscible
        0.47 Strongly self associative systems, interacting with non-polar substances

        `alpha_coeffs` should be a list[list[cij, dij]] so a 3d array
        '''
        # Zero coefficients if not specified by user

#        N = self.N
        cmps = self.cmps
#        alpha_coeffs = self.alpha_coeffs
#        if alpha_coeffs is None:
#            return self.zero_coeffs

        alpha_coeffs_c, alpha_coeffs_d = self.alpha_coeffs_c, self.alpha_coeffs_d

#        alphas = [[0.0]*N for i in cmps]
        alphas = []
        for i in cmps:
#            alphas_i = alphas[i]
#            alpha_coeffs_i = alpha_coeffs[i]
            alpha_coeffs_ci = alpha_coeffs_c[i]
            alpha_coeffs_di = alpha_coeffs_d[i]
            alphas.append([alpha_coeffs_ci[j] + alpha_coeffs_di[j]*T for j in cmps])
#            for j in cmps:
#                alphas_i[j] = alpha_coeffs_ci[j] + alpha_coeffs_di[j]*T
#            for j in range(N - i):
#                # TODO - handle asymmetry
#                if i == j:
#                    alphas_i[j] = 0.0
#                else:
#                    c, d = alpha_coeffs_i[j]
#                    alphas_i[j] = c + d*T

        return alphas

    def dalphas_dT(self, T):
        return self.alpha_coeffs_d

    def d2alphas_dT2(self, T):
        return self.zero_coeffs

    def Gs(self, T, alphas=None, taus=None):
        if alphas is None:
            alphas = self.alphas(T)
        if taus is None:
            taus = self.taus(T)
        cmps = self.cmps

        Gs = []
        for i in cmps:
            alphasi = alphas[i]
            tausi = taus[i]
            Gs.append([exp(-alphasi[j]*tausi[j]) for j in cmps])
        return Gs

    def dGs_dT(self, T, alphas=None, dalphas_dT=None, taus=None, dtaus_dT=None,
               Gs=None):
        '''from sympy import *
        T = symbols('T')
        alpha, tau = symbols('alpha, tau', cls=Function)

        diff(exp(-alpha(T)*tau(T)), T)
        '''
        if alphas is None:
            alphas = self.alphas(T)
        if dalphas_dT is None:
            dalphas_dT = self.dalphas_dT(T)
        if taus is None:
            taus = self.taus(T)
        if dtaus_dT is None:
            dtaus_dT = self.dtaus_dT(T)
        if Gs is None:
            Gs = self.Gs(T, alphas, taus)
        cmps = self.cmps

        dGs_dT = []
        for i in cmps:
            alphasi = alphas[i]
            tausi = taus[i]
            dalphasi = dalphas_dT[i]
            dtausi = dtaus_dT[i]
            Gsi = Gs[i]

            dGs_dT.append([(-alphasi[j]*dtausi[j] - tausi[j]*dalphasi[j])*Gsi[j]
                    for j in cmps])
        return dGs_dT

    def d2Gs_dT2(self, T, alphas=None, dalphas_dT=None, taus=None,
                 dtaus_dT=None, d2taus_dT2=None, Gs=None, dGs_dT=None):
        '''from sympy import *
        T = symbols('T')
        alpha, tau = symbols('alpha, tau', cls=Function)
        expr = diff(exp(-alpha(T)*tau(T)), T, 2)
        expr = ((alpha(T)*Derivative(tau(T), T) + tau(T)*Derivative(alpha(T), T))**2 - alpha(T)*Derivative(tau(T), (T, 2)) - 2*Derivative(alpha(T), T)*Derivative(tau(T), T))*exp(-alpha(T)*tau(T))
        simplify(expr)
        '''
        if alphas is None:
            alphas = self.alphas(T)
        if dalphas_dT is None:
            dalphas_dT = self.dalphas_dT(T)
        if taus is None:
            taus = self.taus(T)
        if dtaus_dT is None:
            dtaus_dT = self.dtaus_dT(T)
        if d2taus_dT2 is None:
            d2taus_dT2 = self.d2taus_dT2(T)
        if Gs is None:
            Gs = self.Gs(T, alphas, taus)
#        if dGs_dT is None:
#            dGs_dT = self.dGs_dT(T, alphas=alphas, dalphas_dT=dalphas_dT,
#                                 taus=taus, dtaus_dT=dtaus_dT, Gs=Gs)
        cmps = self.cmps

        d2Gs_dT2 = []
        for i in cmps:
            alphasi = alphas[i]
            tausi = taus[i]
            dalphasi = dalphas_dT[i]
            dtausi = dtaus_dT[i]
            d2taus_dT2i = d2taus_dT2[i]
            Gsi = Gs[i]
#            dGs_dTi = dGs_dT[i]


            d2Gs_dT2_row = []
            for j in cmps:
                t1 = alphasi[j]*dtausi[j] + tausi[j]*dalphasi[j]

                d2Gs_dT2_row.append((t1*t1 - alphasi[j]*d2taus_dT2i[j] - 2.0*dalphasi[j]*dtausi[j])*exp(-tausi[j]*alphasi[j])) # Gsi[j]

            d2Gs_dT2.append(d2Gs_dT2_row)
        return d2Gs_dT2


    def __init__(self, VaporPressures, tau_coeffs=None, alpha_coeffs=None, Tms=None,
                 Tcs=None, Pcs=None, omegas=None, VolumeLiquids=None, eos=None,
                 eos_mix=None, HeatCapacityLiquids=None,
                 HeatCapacityGases=None,
                 EnthalpyVaporizations=None,

                 **kwargs):



        self.tau_coeffs = tau_coeffs
        if tau_coeffs is not None:
            self.tau_coeffs_A = [[i[0] for i in l] for l in tau_coeffs]
            self.tau_coeffs_B = [[i[1] for i in l] for l in tau_coeffs]
            self.tau_coeffs_E = [[i[2] for i in l] for l in tau_coeffs]
            self.tau_coeffs_F = [[i[3] for i in l] for l in tau_coeffs]
            self.tau_coeffs_G = [[i[4] for i in l] for l in tau_coeffs]
            self.tau_coeffs_H = [[i[5] for i in l] for l in tau_coeffs]
        else:
            self.tau_coeffs_A = None
            self.tau_coeffs_B = None
            self.tau_coeffs_E = None
            self.tau_coeffs_F = None
            self.tau_coeffs_G = None
            self.tau_coeffs_H = None

        self.alpha_coeffs = alpha_coeffs
        if alpha_coeffs is not None:
            self.alpha_coeffs_c = [[i[0] for i in l] for l in alpha_coeffs]
            self.alpha_coeffs_d = [[i[1] for i in l] for l in alpha_coeffs]
        else:
            self.alpha_coeffs_c = None
            self.alpha_coeffs_d = None

        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.VolumeLiquids = VolumeLiquids
        self.eos = eos
        self.eos_mix = eos_mix
        self.N = N = len(VaporPressures)
        self.cmps = range(self.N)
        self.kwargs = kwargs


        self.HeatCapacityLiquids = HeatCapacityLiquids
        self.HeatCapacityGases = HeatCapacityGases
        self.EnthalpyVaporizations = EnthalpyVaporizations

        self.zero_coeffs_linear = [0.0]*self.N

        self.zero_coeffs = [[0.0]*N for _ in range(N)]

        if eos:
            self.eos_pure_instances = [eos(Tc=Tcs[i], Pc=Pcs[i], omega=omegas[i], T=Tcs[i]*0.5, P=Pcs[i]*0.1) for i in self.cmps]

    def gammas(self, T, xs, cached=None):
        alphas = self.alphas(T)
        taus = self.taus(T)
        return NRTL_gammas(xs=xs, taus=taus, alphas=alphas)

class WilsonPP(GammaPhiCaloric):

    def lambdas(self, T):
        r'''Calculate the `lambda` terms for the Wilson model for a specified
        temperature.

        .. math::
            \Lambda_{ij} = \exp\left[a_{ij}+\frac{b_{ij}}{T}+c_{ij}\ln T
+ d_{ij}T + \frac{e_{ij}}{T^2} + f_{ij}{T^2}\right]


        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        # 87% of the time of this routine is the exponential.
        lambda_coeffs_A = self.lambda_coeffs_A
        lambda_coeffs_B = self.lambda_coeffs_B
        lambda_coeffs_C = self.lambda_coeffs_C
        lambda_coeffs_D = self.lambda_coeffs_D
        lambda_coeffs_E = self.lambda_coeffs_E
        lambda_coeffs_F = self.lambda_coeffs_F


        N, cmps = self.N, self.cmps
        T2 = T*T
        Tinv = 1.0/T
        T2inv = Tinv*Tinv
        logT = log(T)

        lambdas = []
        for i in cmps:
            lambda_coeffs_Ai = lambda_coeffs_A[i]
            lambda_coeffs_Bi = lambda_coeffs_B[i]
            lambda_coeffs_Ci = lambda_coeffs_C[i]
            lambda_coeffs_Di = lambda_coeffs_D[i]
            lambda_coeffs_Ei = lambda_coeffs_E[i]
            lambda_coeffs_Fi = lambda_coeffs_F[i]
            lambdasi = [exp(lambda_coeffs_Ai[j] + lambda_coeffs_Bi[j]*Tinv
                        + lambda_coeffs_Ci[j]*logT + lambda_coeffs_Di[j]*T
                        + lambda_coeffs_Ei[j]*T2inv + lambda_coeffs_Fi[j]*T2)
                        for j in cmps]
            lambdas.append(lambdasi)
        return lambdas

    def dlambdas_dT(self, T):
        r'''Calculate the temperature derivative of the `lambda` terms for the
        Wilson model for a specified temperature.

        .. math::
            \frac{\partial \Lambda_{ij}}{\partial T} =
            \left(2 T h_{ij} + d_{ij} + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}}
            - \frac{2 e_{ij}}{T^{3}}\right) e^{T^{2} h_{ij} + T d_{ij} + a_{ij}
            + c_{ij} \ln{\left(T \right)} + \frac{b_{ij}}{T}
            + \frac{e_{ij}}{T^{2}}}


        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        # 87% of the time of this routine is the exponential.
#        lambda_coeffs_A = self.lambda_coeffs_A
        lambda_coeffs_B = self.lambda_coeffs_B
        lambda_coeffs_C = self.lambda_coeffs_C
        lambda_coeffs_D = self.lambda_coeffs_D
        lambda_coeffs_E = self.lambda_coeffs_E
        lambda_coeffs_F = self.lambda_coeffs_F

        cmps = self.cmps
        lambdas = self.lambdas(T)
        dlambdas_dT = []

        T2 = T+T
        Tinv = 1.0/T
        nT2inv = -Tinv*Tinv
        nT3inv2 = 2.0*nT2inv*Tinv

        for i in cmps:
            lambdasi = lambdas[i]
#            lambda_coeffs_Ai = lambda_coeffs_A[i]
            lambda_coeffs_Bi = lambda_coeffs_B[i]
            lambda_coeffs_Ci = lambda_coeffs_C[i]
            lambda_coeffs_Di = lambda_coeffs_D[i]
            lambda_coeffs_Ei = lambda_coeffs_E[i]
            lambda_coeffs_Fi = lambda_coeffs_F[i]
            dlambdas_dTi = [(T2*lambda_coeffs_Fi[j] + lambda_coeffs_Di[j]
                             + lambda_coeffs_Ci[j]*Tinv + lambda_coeffs_Bi[j]*nT2inv
                             + lambda_coeffs_Ei[j]*nT3inv2)*lambdasi[j]
                            for j in cmps]
            dlambdas_dT.append(dlambdas_dTi)
        return dlambdas_dT

    def d2lambdas_dT2(self, T):
        r'''Calculate the second temperature derivative of the `lambda` terms
         for the Wilson model for a specified temperature.

        .. math::
            \frac{\partial^2 \Lambda_{ij}}{\partial^2 T} =
            \left(2 f_{ij} + \left(2 T f_{ij} + d_{ij} + \frac{c_{ij}}{T}
            - \frac{b_{ij}}{T^{2}} - \frac{2 e_{ij}}{T^{3}}\right)^{2}
                - \frac{c_{ij}}{T^{2}} + \frac{2 b_{ij}}{T^{3}}
                + \frac{6 e_{ij}}{T^{4}}\right) e^{T^{2} f_{ij} + T d_{ij}
                + a_{ij} + c_{ij} \ln{\left(T \right)} + \frac{b_{ij}}{T}
                + \frac{e_{ij}}{T^{2}}}


        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        lambda_coeffs_B = self.lambda_coeffs_B
        lambda_coeffs_C = self.lambda_coeffs_C
        lambda_coeffs_E = self.lambda_coeffs_E
        lambda_coeffs_F = self.lambda_coeffs_F

        cmps = self.cmps
        lambdas = self.lambdas(T)
        dlambdas_dT = self.dlambdas_dT(T)

        Tinv = 1.0/T
        nT2inv = -Tinv*Tinv
        T3inv2 = 2.0*Tinv*Tinv*Tinv
#        T4inv6 = 3.0*T3inv2*Tinv

        T4inv6 = 6.0*Tinv*Tinv*Tinv*Tinv

        d2lambdas_dT2s = []
        for i in cmps:
            lambdasi = lambdas[i]
            dlambdas_dTi = dlambdas_dT[i]
            lambda_coeffs_Bi = lambda_coeffs_B[i]
            lambda_coeffs_Ci = lambda_coeffs_C[i]
            lambda_coeffs_Ei = lambda_coeffs_E[i]
            lambda_coeffs_Fi = lambda_coeffs_F[i]
            d2lambdas_dT2i = [(2.0*lambda_coeffs_Fi[j] + nT2inv*lambda_coeffs_Ci[j]
                             + T3inv2*lambda_coeffs_Bi[j] + T4inv6*lambda_coeffs_Ei[j]
                             + dlambdas_dTi[j]*dlambdas_dTi[j]/(lambdasi[j]*lambdasi[j])
                               )*lambdasi[j] for j in cmps]
            d2lambdas_dT2s.append(d2lambdas_dT2i)
        return d2lambdas_dT2s

    def d3lambdas_dT3(self, T):
        r'''Calculate the third temperature derivative of the `lambda` terms
         for the Wilson model for a specified temperature.

        .. math::
            \frac{\partial^3 \Lambda_{ij}}{\partial^3 T} =
            \left(3 \left(2 f_{ij} - \frac{c_{ij}}{T^{2}} + \frac{2 b_{ij}}{T^{3}}
            + \frac{6 e_{ij}}{T^{4}}\right) \left(2 T f_{ij} + d_{ij}
            + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}} - \frac{2 e_{ij}}{T^{3}}\right)
            + \left(2 T f_{ij} + d_{ij} + \frac{c_{ij}}{T} - \frac{b_{ij}}{T^{2}}
            - \frac{2 e_{ij}}{T^{3}}\right)^{3} - \frac{2 \left(- c_{ij}
            + \frac{3 b_{ij}}{T} + \frac{12 e_{ij}}{T^{2}}\right)}{T^{3}}\right)
            e^{T^{2} f_{ij} + T d_{ij} + a_{ij} + c_{ij} \ln{\left(T \right)}
            + \frac{b_{ij}}{T} + \frac{e_{ij}}{T^{2}}}

        These `Lambda ij` values (and the coefficients) are NOT symmetric.
        '''
        lambda_coeffs_B = self.lambda_coeffs_B
        lambda_coeffs_C = self.lambda_coeffs_C
        lambda_coeffs_E = self.lambda_coeffs_E
        lambda_coeffs_F = self.lambda_coeffs_F

        cmps = self.cmps
        lambdas = self.lambdas(T)
        dlambdas_dT = self.dlambdas_dT(T)

        Tinv = 1.0/T
        nT2inv = -Tinv*Tinv
        T3inv2 = 2.0*Tinv*Tinv*Tinv
#        T4inv6 = 3.0*T3inv2*Tinv

        T4inv6 = 6.0*Tinv*Tinv*Tinv*Tinv

        T2_12 = 12.0*Tinv*Tinv

        d3lambdas_dT3s = []
        for i in cmps:
            lambdasi = lambdas[i]
            dlambdas_dTi = dlambdas_dT[i]
            lambda_coeffs_Bi = lambda_coeffs_B[i]
            lambda_coeffs_Ci = lambda_coeffs_C[i]
            lambda_coeffs_Ei = lambda_coeffs_E[i]
            lambda_coeffs_Fi = lambda_coeffs_F[i]
            d3lambdas_dT3is = []
            for j in cmps:
                term2 = (2.0*lambda_coeffs_Fi[j] + nT2inv*lambda_coeffs_Ci[j]
                         + T3inv2*lambda_coeffs_Bi[j] + T4inv6*lambda_coeffs_Ei[j])

                term3 = dlambdas_dTi[j]/lambdasi[j]

                term4 = (T3inv2*(lambda_coeffs_Ci[j] - 3.0*lambda_coeffs_Bi[j]*Tinv
                         - T2_12*lambda_coeffs_Ei[j]))

                d3lambdas_dT3is.append((3.0*term2*term3 + term3*term3*term3 + term4)*lambdasi[j])

            d3lambdas_dT3s.append(d3lambdas_dT3is)
        return d3lambdas_dT3s

    def GE_l2(self, T, xs):
        # Works great already!
        lambdas = self.lambdas(T)
        cmps = self.cmps
        main_tot = 0.0
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*lambdas[i][j]
            main_tot += xs[i]*log(tot)
        return -main_tot*R*T

    def dGE_dT(self, T, xs):
        r'''

        .. math::
            \frac{\partial G^E}{\partial T} = -R\sum_i x_i \ln\left(\sum_j x_i \Lambda_{ij}\right)
            -RT\sum_i \frac{x_i \sum_j x_j \frac{\Lambda _{ij}}{\partial T}}{\sum_j x_j \Lambda_{ij}}
        '''

        '''from sympy import *
        N = 4
        R, T = symbols('R, T')
        x1, x2, x3, x4 = symbols('x1, x2, x3, x4')
        xs = [x1, x2, x3, x4]

        Lambda11, Lambda12, Lambda13, Lambda14, Lambda21, Lambda22, Lambda23, Lambda24, Lambda31, Lambda32, Lambda33, Lambda34, Lambda41, Lambda42, Lambda43, Lambda44 = symbols(
            'Lambda11, Lambda12, Lambda13, Lambda14, Lambda21, Lambda22, Lambda23, Lambda24, Lambda31, Lambda32, Lambda33, Lambda34, Lambda41, Lambda42, Lambda43, Lambda44', cls=Function)
        Lambda_ijs = [[Lambda11(T), Lambda12(T), Lambda13(T), Lambda14(T)],
                   [Lambda21(T), Lambda22(T), Lambda23(T), Lambda24(T)],
                   [Lambda31(T), Lambda32(T), Lambda33(T), Lambda34(T)],
                   [Lambda41(T), Lambda42(T), Lambda43(T), Lambda44(T)]]
        ge = 0
        for i in range(N):
            num = 0
            for j in range(N):
                num += Lambda_ijs[i][j]*xs[j]
            ge -= xs[i]*log(num)
        ge = ge*R*T


        diff(ge, T)
        '''
        cmps = self.cmps
        lambdas = self.lambdas(T)
        dlambdas_dT = self.dlambdas_dT(T)
        RT = T*R

        xj_Lambdas_ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*lambdas[i][j]
            xj_Lambdas_ijs.append(tot)

        xj_dLambdas_dTijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*dlambdas_dT[i][j]
            xj_dLambdas_dTijs.append(tot)


        # First term, with log
        tot = 0.0
        for i in cmps:
            tot += xs[i]*log(xj_Lambdas_ijs[i])
        tot *= -R

        # Second term
        sum1 = 0.0
        for i in cmps:
            sum1 += xs[i]*xj_dLambdas_dTijs[i]/xj_Lambdas_ijs[i]
        tot -= RT*sum1
        return tot

    def d2GE_dT2(self, T, xs):
        r'''
        .. math::
            \frac{\partial^2 G^E}{\partial T^2} = -R\left[T\sum_i \left(\frac{x_i \sum_j (x_j \frac{\partial^2 \Lambda_{ij}}{\partial T^2} )}{\sum_j x_j \Lambda_{ij}}
        - \frac{x_i (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T}  )^2}{(\sum_j x_j \Lambda_{ij})^2}
        \right)
        + 2\sum_i \left(\frac{x_i \sum_j  x_j \frac{\partial \Lambda_{ij}}{\partial T}}{\sum_j x_j \Lambda_{ij}}
        \right)
        \right]
        '''
        cmps = self.cmps
        lambdas = self.lambdas(T)
        dlambdas_dT = self.dlambdas_dT(T)
        d2lambdas_dT2 = self.d2lambdas_dT2(T)


        xj_Lambdas_ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*lambdas[i][j]
            xj_Lambdas_ijs.append(tot)

        xj_dLambdas_dTijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*dlambdas_dT[i][j]
            xj_dLambdas_dTijs.append(tot)

        xj_d2Lambdas_dT2ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*d2lambdas_dT2[i][j]
            xj_d2Lambdas_dT2ijs.append(tot)

        # Last term, also the same term as last term of dGE_dT
        sum1 = 0.0
        for i in cmps:
            sum1 += xs[i]*xj_dLambdas_dTijs[i]/xj_Lambdas_ijs[i]

        sum0 = 0.0
        for i in cmps:
            sum0 += (xs[i]*xj_d2Lambdas_dT2ijs[i]/xj_Lambdas_ijs[i]
                    - xs[i]*(xj_dLambdas_dTijs[i]*xj_dLambdas_dTijs[i])/(xj_Lambdas_ijs[i]*xj_Lambdas_ijs[i]))

        return -R*(T*sum0 + 2.0*sum1)


    def d3GE_dT3(self, T, xs):
        r'''
        .. math::
            \frac{\partial^3 G^E}{\partial T^3} = -R\left[3\left(\frac{x_i \sum_j (x_j \frac{\partial^2 \Lambda_{ij}}{\partial T^2} )}{\sum_j x_j \Lambda_{ij}}
            - \frac{x_i (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T}  )^2}{(\sum_j x_j \Lambda_{ij})^2}
            \right)
            +T\left(
            \sum_i \frac{x_i (\sum_j x_j \frac{\partial^3 \Lambda _{ij}}{\partial T^3})}{\sum_j x_j \Lambda_{ij}}
            - \frac{3x_i (\sum_j x_j \frac{\partial \Lambda_{ij}^2}{\partial T^2})  (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T})}{(\sum_j x_j \Lambda_{ij})^2}
            + 2\frac{x_i(\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T})^3}{(\sum_j x_j \Lambda_{ij})^3}
            \right)\right]

        '''
        cmps = self.cmps
        lambdas = self.lambdas(T)
        dlambdas_dT = self.dlambdas_dT(T)
        d2lambdas_dT2 = self.d2lambdas_dT2(T)
        d3lambdas_dT3 = self.d3lambdas_dT3(T)

        xj_Lambdas_ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*lambdas[i][j]
            xj_Lambdas_ijs.append(tot)

        xj_dLambdas_dTijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*dlambdas_dT[i][j]
            xj_dLambdas_dTijs.append(tot)

        xj_d2Lambdas_dT2ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*d2lambdas_dT2[i][j]
            xj_d2Lambdas_dT2ijs.append(tot)

        xj_d3Lambdas_dT3ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*d3lambdas_dT3[i][j]
            xj_d3Lambdas_dT3ijs.append(tot)

        #Term is directly from the one above it
        sum0 = 0.0
        for i in cmps:
            sum0 += (xs[i]*xj_d2Lambdas_dT2ijs[i]/xj_Lambdas_ijs[i]
                    - xs[i]*(xj_dLambdas_dTijs[i]*xj_dLambdas_dTijs[i])/(xj_Lambdas_ijs[i]*xj_Lambdas_ijs[i]))

        sum_d3 = 0.0
        for i in cmps:
            sum_d3 += xs[i]*xj_d3Lambdas_dT3ijs[i]/xj_Lambdas_ijs[i]

        sum_comb = 0.0
        for i in cmps:
            sum_comb += 3.0*xs[i]*xj_d2Lambdas_dT2ijs[i]*xj_dLambdas_dTijs[i]/(xj_Lambdas_ijs[i]*xj_Lambdas_ijs[i])

        sum_last = 0.0
        for i in cmps:
            sum_last += 2.0*xs[i]*(xj_dLambdas_dTijs[i])**3/(xj_Lambdas_ijs[i]*xj_Lambdas_ijs[i]*xj_Lambdas_ijs[i])

        return -R*(3.0*sum0 + T*(sum_d3 - sum_comb + sum_last))

    def d2GE_dTdxs(self, T, xs):
        r'''

        .. math::
            \frac{\partial^2 G^E}{\partial x_k \partial T} = -R\left[T\left(
            \sum_i  \left(\frac{x_i \frac{\partial n_{ik}}{\partial T}}{\sum_j x_j \Lambda_{ij}}
            - \frac{x_i \Lambda_{ik} (\sum_j x_j \frac{\partial \Lambda_{ij}}{\partial T} )}{(\partial_j x_j \Lambda_{ij})^2}
            \right) + \frac{\sum_i x_i \frac{\partial \Lambda_{ki}}{\partial T}}{\sum_j x_j \Lambda_{kj}}
            \right)
            + \ln\left(\sum_i x_i \Lambda_{ki}\right)
            + \sum_i \frac{x_i \Lambda_{ik}}{\sum_j x_j \Lambda_{ij}}
            \right]
        '''
        cmps = self.cmps
        lambdas = self.lambdas(T)
        dlambdas_dT = self.dlambdas_dT(T)
        d2lambdas_dT2 = self.d2lambdas_dT2(T)


        xj_Lambdas_ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*lambdas[i][j]
            xj_Lambdas_ijs.append(tot)

        xj_dLambdas_dTijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*dlambdas_dT[i][j]
            xj_dLambdas_dTijs.append(tot)


        d2GE_dTdxs = []
        for k in cmps:
            tot1 = 0.0
            for i in cmps:
                tot1 += (xs[i]*dlambdas_dT[i][k]/xj_Lambdas_ijs[i]
                - xs[i]*xj_dLambdas_dTijs[i]*lambdas[i][k]/(xj_Lambdas_ijs[i]*xj_Lambdas_ijs[i]))

            tot1 += xj_dLambdas_dTijs[k]/xj_Lambdas_ijs[k]

            tot2 = 0.0
            for i in cmps:
                tot2 += xs[i]*lambdas[i][k]/xj_Lambdas_ijs[i]

            dG = -R*(T*tot1 + log(xj_Lambdas_ijs[k]) + tot2)

            d2GE_dTdxs.append(dG)
        return d2GE_dTdxs



    def dGE_dxs(self, T, xs):
        r'''

        .. math::
            \frac{\partial G^E}{\partial x_k} = -RT\left[
            \sum_i \frac{x_i \Lambda_{ik}}{\sum_j \Lambda_{ij}x_j }
            + \ln\left(\sum_j x_j \Lambda_{kj}\right)
            \right]
        '''
        '''
        from sympy import *
        N = 4
        R, T = symbols('R, T')
        x1, x2, x3, x4 = symbols('x1, x2, x3, x4')
        xs = [x1, x2, x3, x4]

        Lambda11, Lambda12, Lambda13, Lambda14, Lambda21, Lambda22, Lambda23, Lambda24, Lambda31, Lambda32, Lambda33, Lambda34, Lambda41, Lambda42, Lambda43, Lambda44 = symbols(
            'Lambda11, Lambda12, Lambda13, Lambda14, Lambda21, Lambda22, Lambda23, Lambda24, Lambda31, Lambda32, Lambda33, Lambda34, Lambda41, Lambda42, Lambda43, Lambda44', cls=Function)
        Lambda_ijs = [[Lambda11(T), Lambda12(T), Lambda13(T), Lambda14(T)],
                   [Lambda21(T), Lambda22(T), Lambda23(T), Lambda24(T)],
                   [Lambda31(T), Lambda32(T), Lambda33(T), Lambda34(T)],
                   [Lambda41(T), Lambda42(T), Lambda43(T), Lambda44(T)]]
        ge = 0
        for i in range(N):
            num = 0
            for j in range(N):
                num += Lambda_ijs[i][j]*xs[j]
            ge -= xs[i]*log(num)
        ge = ge*R*T


        diff(ge, x1)#, diff(ge, x1, x2), diff(ge, x1, x2, x3)
        '''
        cmps = self.cmps
        lambdas = self.lambdas(T)
        mRT = -T*R

        xj_Lambdas_ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*lambdas[i][j]
            xj_Lambdas_ijs.append(tot)

        # k = what is being differentiated with respect to
        dGE_dxs = []
        for k in cmps:
            tot = 0.0
            for i in cmps:
                tot += xs[i]*lambdas[i][k]/xj_Lambdas_ijs[i]
            tot += log(xj_Lambdas_ijs[k])

            dGE_dxs.append(mRT*tot)

        return dGE_dxs


    def d2GE_dxixjs(self, T, xs):
        r'''
        .. math::
            \frac{\partial^2 G^E}{\partial x_k \partial x_m} = RT\left(
            \sum_i \frac{x_i \Lambda_{ik} \Lambda_{im}}{(\sum_j x_j \Lambda_{ij})^2}
            -\frac{\Lambda_{km}}{\sum_j x_j \Lambda_{kj}}
            -\frac{\Lambda_{mk}}{\sum_j x_j \Lambda_{mj}}
            \right)
        '''
        # Correct, tested with hessian
        cmps = self.cmps
        RT = R*T
        lambdas = self.lambdas(T)

        xj_Lambdas_ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*lambdas[i][j]
            xj_Lambdas_ijs.append(tot)



        d2GE_dxixjs = []
        for k in cmps:
            dG_row = []
            for m in cmps:
                tot = 0.0
                for i in cmps:
                    tot += xs[i]*lambdas[i][k]*lambdas[i][m]/(xj_Lambdas_ijs[i]*xj_Lambdas_ijs[i])
                tot -= lambdas[k][m]/xj_Lambdas_ijs[k]
                tot -= lambdas[m][k]/xj_Lambdas_ijs[m]
                dG_row.append(RT*tot)
            d2GE_dxixjs.append(dG_row)
        return d2GE_dxixjs

    def d3GE_dxixjxks(self, T, xs):
        r'''
        .. math::
            \frac{\partial^3 G^E}{\partial x_k \partial x_m \partial x_n}
            = -RT\left[
            \sum_i \left(\frac{2x_i \Lambda_{ik}\Lambda_{im}\Lambda_{in}} {(\sum x_j \Lambda_{ij})^3}\right)
            - \frac{\Lambda_{km} \Lambda_{kn}}{(\sum_j x_j \Lambda_{kj})^2}
            - \frac{\Lambda_{mk} \Lambda_{mn}}{(\sum_j x_j \Lambda_{mj})^2}
            - \frac{\Lambda_{nk} \Lambda_{nm}}{(\sum_j x_j \Lambda_{nj})^2}

            \right]
        '''
        # Correct, tested with sympy expanding
        cmps = self.cmps
        nRT = -R*T
        lambdas = self.lambdas(T)

        xj_Lambdas_ijs = []
        for i in cmps:
            tot = 0.0
            for j in cmps:
                tot += xs[j]*lambdas[i][j]
            xj_Lambdas_ijs.append(tot)

        xj_Lambdas_ijs_invs = [1.0/i for i in xj_Lambdas_ijs]

        # all the same: analytical[i][j][k] = analytical[i][k][j] = analytical[j][i][k] = analytical[j][k][i] = analytical[k][i][j] = analytical[k][j][i] = float(v)
        d2GE_dxixjs = []
        for k in cmps:
            dG_matrix = []
            for m in cmps:
                dG_row = []
                for n in cmps:
                    tot = 0.0
                    for i in cmps:
                        num = 2.0*xs[i]*lambdas[i][k]*lambdas[i][m]*lambdas[i][n]
                        den = xj_Lambdas_ijs_invs[i]*xj_Lambdas_ijs_invs[i]*xj_Lambdas_ijs_invs[i]
                        tot += num*den

                    tot -= lambdas[k][m]*lambdas[k][n]*xj_Lambdas_ijs_invs[k]*xj_Lambdas_ijs_invs[k]
                    tot -= lambdas[m][k]*lambdas[m][n]*xj_Lambdas_ijs_invs[m]*xj_Lambdas_ijs_invs[m]
                    tot -= lambdas[n][m]*lambdas[n][k]*xj_Lambdas_ijs_invs[n]*xj_Lambdas_ijs_invs[n]
                    dG_row.append(nRT*tot)
                dG_matrix.append(dG_row)
            d2GE_dxixjs.append(dG_matrix)
        return d2GE_dxixjs

    def gammas(self, T, xs, cached=None):
        lambdas = self.lambdas(T)
        return Wilson(xs=xs, params=lambdas)








    def __init__(self, VaporPressures, lambda_coeffs=None, Tms=None,
                 Tcs=None, Pcs=None, omegas=None, VolumeLiquids=None, eos=None,
                 eos_mix=None, HeatCapacityLiquids=None,
                 HeatCapacityGases=None,
                 EnthalpyVaporizations=None,

                 **kwargs):



        self.lambda_coeffs = lambda_coeffs
        if lambda_coeffs is not None:
            self.lambda_coeffs_A = [[i[0] for i in l] for l in lambda_coeffs]
            self.lambda_coeffs_B = [[i[1] for i in l] for l in lambda_coeffs]
            self.lambda_coeffs_C = [[i[2] for i in l] for l in lambda_coeffs]
            self.lambda_coeffs_D = [[i[3] for i in l] for l in lambda_coeffs]
            self.lambda_coeffs_E = [[i[4] for i in l] for l in lambda_coeffs]
            self.lambda_coeffs_F = [[i[5] for i in l] for l in lambda_coeffs]
        else:
            self.lambda_coeffs_A = None
            self.lambda_coeffs_B = None
            self.lambda_coeffs_C = None
            self.lambda_coeffs_D = None
            self.lambda_coeffs_E = None
            self.lambda_coeffs_F = None

        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.VolumeLiquids = VolumeLiquids
        self.eos = eos
        self.eos_mix = eos_mix
        self.N = N = len(VaporPressures)
        self.cmps = range(self.N)
        self.kwargs = kwargs


        self.HeatCapacityLiquids = HeatCapacityLiquids
        self.HeatCapacityGases = HeatCapacityGases
        self.EnthalpyVaporizations = EnthalpyVaporizations

        self.zero_coeffs_linear = [0.0]*self.N

        self.zero_coeffs = [[0.0]*N for _ in range(N)]

        if eos:
            self.eos_pure_instances = [eos(Tc=Tcs[i], Pc=Pcs[i], omega=omegas[i], T=Tcs[i]*0.5, P=Pcs[i]*0.1) for i in self.cmps]



class Unifac(GammaPhi):
    '''
    '''
    # TODO: Calculate derivatives analytically via the derivatives given in the supporting information of
    # Jäger, Andreas, Ian H. Bell, and Cornelia Breitkopf. "A Theoretically Based Departure Function for Multi-Fluid Mixture Models." Fluid Phase Equilibria 469 (August 15, 2018): 56-69. https://doi.org/10.1016/j.fluid.2018.04.015.

    subgroup_data = UFSG

    def __init__(self, UNIFAC_groups, VaporPressures, Tms=None, Tcs=None, Pcs=None,
                 omegas=None, VolumeLiquids=None, eos=None, eos_mix=None, **kwargs):
        self.UNIFAC_groups = UNIFAC_groups
        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.VolumeLiquids = VolumeLiquids
        self.omegas = omegas
        self.eos = eos
        self.eos_mix = eos_mix
        self.N = len(VaporPressures)
        self.cmps = range(self.N)

        if eos:
            self.eos_pure_instances = [eos(Tc=Tcs[i], Pc=Pcs[i], omega=omegas[i], T=Tcs[i]*0.5, P=Pcs[i]*0.1) for i in self.cmps]

        self.cache_unifac_inputs()

    def cache_unifac_inputs(self):
        # Pre-calculate some of the inputs UNIFAC uses
        self.rs = []
        self.qs = []
        for groups in self.UNIFAC_groups:
            ri = 0.
            qi = 0.
            for group, count in groups.items():
                ri += self.subgroup_data[group].R*count
                qi += self.subgroup_data[group].Q*count
            self.rs.append(ri)
            self.qs.append(qi)


        self.group_counts = {}
        for groups in self.UNIFAC_groups:
            for group, count in groups.items():
                if group in self.group_counts:
                    self.group_counts[group] += count
                else:
                    self.group_counts[group] = count
        self.UNIFAC_cached_inputs = (self.rs, self.qs, self.group_counts)

    def gammas(self, T, xs, cached=None):
        return UNIFAC_gammas(chemgroups=self.UNIFAC_groups, T=T, xs=xs, cached=self.UNIFAC_cached_inputs)



class UnifacDortmund(Unifac):
    subgroup_data = DOUFSG

    def gammas(self, T, xs, cached=None):
        return UNIFAC_gammas(chemgroups=self.UNIFAC_groups, T=T, xs=xs,
                      cached=self.UNIFAC_cached_inputs,
                      subgroup_data=DOUFSG, interaction_data=unifac.DOUFIP2006, modified=True)


class UnifacCaloric(Unifac, GammaPhiCaloric):

    def __init__(self, VaporPressures, eos=None, **kwargs):
        self.cmps = range(len(VaporPressures))
        self.N = len(VaporPressures)
        self.VaporPressures = VaporPressures
        self.eos = eos
        self.__dict__.update(kwargs)
        self.kwargs = {'VaporPressures': VaporPressures, 'eos': eos}
        self.kwargs.update(kwargs)

        if eos:
            self.eos_pure_instances = [eos(Tc=self.Tcs[i], Pc=self.Pcs[i], omega=self.omegas[i], T=self.Tcs[i]*0.5, P=self.Pcs[i]*0.1) for i in self.cmps]

        self.cache_unifac_inputs()

class UnifacDortmundCaloric(UnifacDortmund, GammaPhiCaloric):

    def __init__(self, VaporPressures, eos=None, **kwargs):
        self.cmps = range(len(VaporPressures))
        self.N = len(VaporPressures)
        self.VaporPressures = VaporPressures
        self.eos = eos
        self.__dict__.update(kwargs)
        self.kwargs = {'VaporPressures': VaporPressures, 'eos': eos}
        self.kwargs.update(kwargs)

        if eos:
            self.eos_pure_instances = [eos(Tc=self.Tcs[i], Pc=self.Pcs[i], omega=self.omegas[i], T=self.Tcs[i]*0.5, P=self.Pcs[i]*0.1) for i in self.cmps]

        self.cache_unifac_inputs()

def eos_Z_test_phase_stability(eos):
    try:
        if eos.G_dep_l < eos.G_dep_g:
            Z_eos = eos.Z_l
            prefer, alt = 'Z_g', 'Z_l'
        else:
            Z_eos = eos.Z_g
            prefer, alt =  'Z_l', 'Z_g'
    except:
        # Only one root - take it and set the prefered other phase to be a different type
        Z_eos = eos.Z_g if hasattr(eos, 'Z_g') else eos.Z_l
        prefer = 'Z_l' if hasattr(eos, 'Z_g') else 'Z_g'
        alt = 'Z_g' if hasattr(eos, 'Z_g') else 'Z_l'
    return Z_eos, prefer, alt


def eos_Z_trial_phase_stability(eos, prefer, alt):
    try:
        if eos.G_dep_l < eos.G_dep_g:
            Z_trial = eos.Z_l
        else:
            Z_trial = eos.Z_g
    except:
        # Only one phase, doesn't matter - only that phase will be returned
        try:
            Z_trial = getattr(eos, alt)
        except:
            Z_trial = getattr(eos, prefer)
    return Z_trial

class GceosBase(Ideal):
    # TodO move to own class

    pure_guesses = True
    Wilson_guesses = True
    random_guesses = False
    zero_fraction_guesses = 1E-6
    stability_maxiter = 500 # 30 good professional default; 500 used in source DTU
#    PT_STABILITY_XTOL = 5E-9 # 1e-12 was too strict; 1e-10 used in source DTU; 1e-9 set for some points near critical where convergence stopped; even some more stopped at higher Ts
    stability_xtol = 1e-10
    substitution_maxiter =  100 # 1000 #
#    substitution_xtol = 1e-7 # 1e-10 too strict
    substitution_xtol = 1e-12 #new, fugacity ratio - root based tolerance
    def __init__(self, eos_mix=PRMIX, VaporPressures=None, Tms=None, Tbs=None,
                 Tcs=None, Pcs=None, omegas=None, kijs=None, eos_kwargs=None,
                 HeatCapacityGases=None, MWs=None, atomss=None,
                 eos=None, Hfs=None, Gfs=None,
                 **kwargs):
        self.eos_mix = eos_mix
        self.VaporPressures = VaporPressures
        self.Tms = Tms
        self.Tbs = Tbs
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.kijs = kijs
        self.eos_kwargs = eos_kwargs if eos_kwargs is not None else {}
        self.eos = eos
        self.N = len(VaporPressures)
        self.cmps = range(self.N)
        self.HeatCapacityGases = HeatCapacityGases

        self.Hfs = Hfs
        self.Gfs = Gfs
        if Hfs is not None and Gfs is not None and None not in Hfs and None not in Gfs:
            self.Sfs = [(Hfi - Gfi)/298.15 for Hfi, Gfi in zip(Hfs, Gfs)]

        self.stability_tester = StabilityTester(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas)

        self.kwargs = kwargs
        self.kwargs['HeatCapacityGases'] = HeatCapacityGases
        self.kwargs['VaporPressures'] = VaporPressures
        self.kwargs['Tms'] = Tms
        self.kwargs['Tcs'] = Tcs
        self.kwargs['Pcs'] = Pcs
        self.kwargs['omegas'] = omegas
        self.kwargs['kijs'] = kijs

        self.MWs = MWs
        self.atomss = atomss

        if atomss is not None:
            self.n_atoms = [sum(i.values()) for i in atomss]

        # No `zs`
#        self.eos_mix_ref = self.eos_mix(T=self.T_REF_IG, P=self.P_REF_IG, Tcs=self.Tcs, Pcs=self.Pcs, kijs=self.kijs, **self.eos_kwargs)
    def eos_pure_ref(self, i=None):
        try:
            if i is None:
                return self.eos_pure_refs
            else:
                return self.eos_pure_refs[i]
        except:
            fake_T = 298.15
            fake_P = 101325.0
            fake_zs = [1./self.N]*self.N
            eos_base = self.to_TP_zs(T=fake_T, P=fake_P, zs=fake_zs)
            eos_pure_refs = []
            for idx in self.cmps:
                eos_pure = eos_base.to_TPV_pure(T=fake_T, P=fake_P, V=None, i=idx)

                eos_pure_refs.append(eos_pure)
            self.eos_pure_refs = eos_pure_refs

            if i is None:
                return self.eos_pure_refs
            else:
                return self.eos_pure_refs[i]

    def _Psats(self, T):
#        fake_P = 1e7 # Try to avoid two solutions (slower)
#        fake_zs = [1./self.N]*self.N
#        Psats = []
#        try:
#            try:
#                assert self.eos_l is not None
#                eos_base = self.eos_l
#            except AttributeError:
#                assert self.eos_g is not None
#                eos_base = self.eos_g
#        except:
#            eos_base = self.to_TP_zs(T=T, P=fake_P, zs=fake_zs)

#        for i in self.cmps:
#            eos_pure = eos_base.to_TP_pure(T, fake_P, i)
#            Psats.append(eos_pure.Psat(T))
        return [eos.Psat(T) for eos in self.eos_pure_ref()]
#        return Psats

    def _Tsats(self, P):
        try:
            vs = [eos.Tsat(P) for eos in self.eos_pure_ref()]
            print(vs,P, 'values')
            print(self.eos_pure_ref())
            return vs
        except Exception as e:
            print('Fail')
            print(e)
            raise e

#        fake_T = 300
#        fake_zs = [1./self.N]*self.N
#        Tsats = []
#        try:
#            try:
#                assert self.eos_l is not None
#                eos_base = self.eos_l
#            except AttributeError:
#                assert self.eos_g is not None
#                eos_base = self.eos_g
#        except:
#            eos_base = self.to_TP_zs(T=fake_T, P=P, zs=fake_zs)
#
#        for i in self.cmps:
#            eos_pure = eos_base.to_TP_pure(fake_T, P, i)
#            Tsats.append(eos_pure.Tsat(P))
#        return Tsats

    def dH_dT(self, T, P, V_over_F, zs, xs, ys, eos_l, eos_g, phase):
        # Believed correct
        H = 0.0
#        T = self.T
#        P = self.P
        T_REF_IG = self.T_REF_IG
        HeatCapacityGases = self.HeatCapacityGases
        cmps = self.cmps

        if phase == 'g':
            for zi, obj in zip(zs, HeatCapacityGases):
                H += zi*obj.T_dependent_property(T)
#                print(H, 'Cpg')
            try:
#                print('returning liquid', eos_g.dH_dep_dT_g)
                H += eos_g.dH_dep_dT_g
            except AttributeError:
                H += eos_g.dH_dep_dT_l
        elif phase == 'l':
            for zi, obj in zip(zs, HeatCapacityGases):
                H += zi*obj.T_dependent_property(T)
                # Be careful Cp is continuous
#                print(H, 'Cpg')
            try:
                H += eos_l.dH_dep_dT_l
            except AttributeError:
                H += eos_l.dH_dep_dT_g
        elif phase == 'l/g':
            raise ValueError
        return H


    def dH_dP(self, P, T, V_over_F, zs, xs, ys, eos_l, eos_g, phase):
        dH = 0.0
        if phase == 'g':
            try:
                dH += eos_g.dH_dep_dP_g
            except AttributeError:
                dH += eos_g.dH_dep_P_l
        elif phase == 'l':
            try:
                dH += eos_l.dH_dep_dP_l
            except AttributeError:
                dH += eos_l.dH_dep_dP_g
        elif phase == 'l/g':
            raise ValueError
        return dH

    def enthalpy_eosmix(self, T, P, V_over_F, zs, xs, ys, eos_l, eos_g, phase):
        # Believed correct
        H = 0.0
#        T = self.T
#        P = self.P
        T_REF_IG = self.T_REF_IG
        HeatCapacityGases = self.HeatCapacityGases
        cmps = self.cmps

        if phase == 'g' or V_over_F == 1.0:
            for zi, obj in zip(zs, HeatCapacityGases):
                H += zi*obj.T_dependent_property_integral(T_REF_IG, T)
            try:
                H += eos_g.H_dep_g
            except AttributeError:
                H += eos_g.H_dep_l
        elif phase == 'l' or V_over_F == 0.0:
            for zi, obj in zip(zs, HeatCapacityGases):
                H += zi*obj.T_dependent_property_integral(T_REF_IG, T)
            try:
                H += eos_l.H_dep_l
            except AttributeError:
                H += eos_l.H_dep_g
        elif phase == 'l/g':
            try:
                H_l = eos_l.H_dep_l
            except AttributeError:
                H_l = eos_l.H_dep_g

            try:
                H_g = eos_g.H_dep_g
            except AttributeError:
                H_g = eos_g.H_dep_l

            dH_integrals = [obj.T_dependent_property_integral(T_REF_IG, T)for obj in HeatCapacityGases]
            for xi, yi, dH in zip(xs, ys, dH_integrals):
                H_g += yi*dH
                H_l += xi*dH
            H = H_g*V_over_F + H_l*(1.0 - V_over_F)
        return H


    def dS_dT(self, T, P, V_over_F, zs, xs, ys, eos_l, eos_g, phase):
        HeatCapacityGases = self.HeatCapacityGases
        cmps = self.cmps
        T_REF_IG = self.T_REF_IG
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        if phase == 'g' or V_over_F == 1.0:
            dS_pure_zum = 0.0
            for zi, obj in zip(zs, HeatCapacityGases):
                dS_pure_zum += zi*obj.T_dependent_property(T)
            S += dS_pure_zum/T
            try:
                S += eos_g.dS_dep_dT_g
            except AttributeError:
                S += eos_g.dS_dep_dT_l
        elif phase == 'l' or V_over_F == 0.0:
            dS_pure_zum = 0.0
            for zi, obj in zip(zs, HeatCapacityGases):
                dS_pure_zum += zi*obj.T_dependent_property(T)
            S += dS_pure_zum/T
            try:
                S += eos_l.dS_dep_dT_l
            except AttributeError:
                S += eos_l.dS_dep_dT_g
        elif phase == 'l/g':
            raise ValueError

        return S

    def dS_dP(self, P, T, V_over_F, zs, xs, ys, eos_l, eos_g, phase):
        dS = 0.0
        if phase == 'g' or V_over_F == 1.0:
            dS -= R/P
            try:
                dS += eos_g.dS_dep_dP_g
            except AttributeError:
                dS += eos_g.dS_dep_dP_l
        elif phase == 'l' or V_over_F == 0.0:
            dS -= R/P
            try:
                dS += eos_l.dS_dep_dP_l
            except AttributeError:
                dS += eos_l.dS_dep_dP_g
        elif phase == 'l/g':
            raise ValueError

        return dS


    def entropy_eosmix(self, T, P, V_over_F, zs, xs, ys, eos_l, eos_g, phase):
        HeatCapacityGases = self.HeatCapacityGases
        cmps = self.cmps
        T_REF_IG = self.T_REF_IG
        P_REF_IG_INV = self.P_REF_IG_INV
        S = 0.0
        # Could be ordered as log(zi^zi*zj^zj... and so on.)
        # worth checking/trying!
        # It is faster in CPython, slower or maybe faster in PyPy - very very hard to tell.
        if phase == 'g' or V_over_F == 1.0:
            S -= R*sum([zi*log(zi) for zi in zs if zi > 0.0]) # ideal composition entropy composition
            S -= R*log(P*P_REF_IG_INV) # Not sure, but for delta S - doesn't impact what is divided by.
            for i in cmps:
                dS = HeatCapacityGases[i].T_dependent_property_integral_over_T(T_REF_IG, T)
                S += zs[i]*dS
            try:
                S += eos_g.S_dep_g
            except AttributeError:
                S += eos_g.S_dep_l

        elif phase == 'l' or V_over_F == 0.0:
            S -= R*sum([zi*log(zi) for zi in zs if zi > 0.0]) # ideal composition entropy composition
            S -= R*log(P*P_REF_IG_INV)
            for i in cmps:
                dS = HeatCapacityGases[i].T_dependent_property_integral_over_T(T_REF_IG, T)
                S += zs[i]*dS
            try:
                S += eos_l.S_dep_l
            except AttributeError:
                S += eos_l.S_dep_g

        elif phase == 'l/g':
            S_l = 0.0
            for xi in xs:
                try:
                    S_l += xi*log(xi)
                except ValueError:
                    pass
            S_l *= -R

            S_g = 0.0
            for yi in ys:
                try:
                    S_g += yi*log(yi)
                except ValueError:
                    pass
            S_g *= -R

            mRlogPratio = -R*log(P*P_REF_IG_INV)
            S_l += mRlogPratio
            S_g += mRlogPratio
            for i in cmps:
                dS = HeatCapacityGases[i].T_dependent_property_integral_over_T(T_REF_IG, T)
                S_g += ys[i]*dS
                S_l += xs[i]*dS

            S_g += eos_g.S_dep_g
            S_l += eos_l.S_dep_l
            S = S_g*V_over_F + S_l*(1.0 - V_over_F)
        return S

    @property
    def Hlm_dep(self):
        return self.eos_l.H_dep_l

    @property
    def Hgm_dep(self):
        return self.eos_g.H_dep_g

    @property
    def Slm_dep(self):
        return self.eos_l.S_dep_l

    @property
    def Sgm_dep(self):
        return self.eos_g.S_dep_g

    @property
    def Cplm_dep(self):
        return self.eos_l.Cp_dep_l

    @property
    def Cpgm_dep(self):
        return self.eos_g.Cp_dep_g

    @property
    def Cvlm_dep(self):
        return self.eos_l.Cv_dep_l

    @property
    def Cvgm_dep(self):
        return self.eos_g.Cv_dep_g

    @property
    def Cpgm(self):
        Cp = 0.0
        for i in self.cmps:
            Cp += self.zs[i]*self.HeatCapacityGases[i].T_dependent_property(self.T)
        return Cp + self.Cpgm_dep

    @property
    def Cplm(self):
        Cp = 0.0
        for i in self.cmps:
            Cp += self.zs[i]*self.HeatCapacityGases[i].T_dependent_property(self.T)
        return Cp + self.Cplm_dep

    @property
    def Cvgm(self):
        return self.Cpgm - Cp_minus_Cv(T=self.T, dP_dT=self.eos_g.dP_dT_g, dP_dV=self.eos_g.dP_dV_g)

    @property
    def Cvlm(self):
        return self.Cplm - Cp_minus_Cv(T=self.T, dP_dT=self.eos_l.dP_dT_l, dP_dV=self.eos_l.dP_dV_l)

    def flash_caloric(self, zs, T=None, P=None, VF=None, Hm=None, Sm=None):
        if not self.SUPPORTS_ZERO_FRACTIONS:
            zs = remove_zeros(zs, self.zero_fraction)
        self.zs = zs

        try:
            if T is not None and Sm is not None:
                # TS
                pass
            elif P is not None and Sm is not None:
                phase, xs, ys, VF, T = self.flash_PS_zs(P, Sm, zs)
                # PS
            elif P is not None and Hm is not None:
                phase, xs, ys, VF, T = self.flash_PH_zs(P, Hm, zs)
                # PH

            elif ((T is not None and P is not None) or
                (T is not None and VF is not None) or
                (P is not None and VF is not None)):
                  # non thermo flashes
                  self.flash(zs=zs, T=T, P=P, VF=VF)
                  xs = self.xs
                  ys = self.ys
                  phase = self.phase
                  VF = self.V_over_F
                  pass
            else:
                raise Exception('Flash inputs unsupported')

            self.P = P
            self.T = T
            self.V_over_F = VF
            self.xs = xs
            self.ys = ys
            self.phase = phase
            self.Hm = Hm
            self.Sm = Sm

#            ''' The routine needs to be upgraded to set these properties
#                self.T = T
#                self.P = P
#                self.V_over_F = V_over_F
#                self.phase = phase
#                self.xs = xs
#                self.ys = ys
#                self.zs = zs
#            '''
            self._post_flash()
            self.status = True
        except Exception as e:
            # Write Nones for everything here
            self.status = e
            self._set_failure()


    def _post_flash(self):
        # Note: compositions are not being checked!
        if self.xs is not None: #  and self.V_over_F != 1.0
            try:
                if self.eos_l.T == self.T and self.eos_l.P == self.P:
                    if self.eos_l.da_alpha_dT == -5e-3:
                        # finish, calculate a_alphas
                        self.eos_l.resolve_full_alphas()
                    pass
                else:
                    raise ValueError
            except:
                self.eos_l = self.to_TP_zs(self.T, self.P, self.xs)
        if self.ys is not None: #  and self.V_over_F != 0.0
            try:
                if self.eos_g.T == self.T and self.eos_g.P == self.P:
                    if self.eos_g.da_alpha_dT == -5e-3:
                        # finish, calculate a_alphas
                        self.eos_g.resolve_full_alphas()
                    pass
                else:
                    raise ValueError
            except:
                self.eos_g = self.to_TP_zs(self.T, self.P, self.ys)
        # Cannot derive other properties with this
        try:
            self.Hm = self.enthalpy_eosmix(T=self.T, P=self.P, V_over_F=self.V_over_F,
                                          zs=self.zs, xs=self.xs, ys=self.ys,
                                          eos_l=self.eos_l, eos_g=self.eos_g,
                                          phase=self.phase)
            self.Sm = self.entropy_eosmix(T=self.T, P=self.P, V_over_F=self.V_over_F,
                                          zs=self.zs, xs=self.xs, ys=self.ys,
                                          eos_l=self.eos_l, eos_g=self.eos_g,
                                          phase=self.phase)
            self.Gm = self.Hm - self.T*self.Sm if (self.Hm is not None and self.Sm is not None) else None
        except Exception as e:
            print(e)
            pass

    def to_TP_zs(self, T, P, zs, fugacities=True, only_l=False, only_g=False):
        return self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                            zs=zs, kijs=self.kijs, T=T, P=P,
                            fugacities=fugacities, only_l=only_l,
                            only_g=only_g, **self.eos_kwargs)


    def flash_TP_zs_3P(self, T, P, zs):
        "From 5.9: Multiphase Split and Stability Analysis"
        phase, xs, ys, beta_y = self.flash_TP_zs(T=T, P=P, zs=zs)
        eos_l, eos_g = self.eos_l, self.eos_g

        Ks_y = [yi/xi for yi, xi in zip(ys, xs)]
        print('2PHASE KS', Ks_y)

        # TODO only call stability test on heavier MW phase
        def is_stable():
            stable, Ks_initial, Ks_extra = eos_l.stability_Michelsen(T=T, P=P, zs=zs,
                                                      Ks_initial=None,
                                                      maxiter=self.stability_maxiter,
                                                      xtol=self.stability_xtol)
            if not stable:
                return stable, Ks_extra[-1]

#            print('RESULTS ZS', stable, Ks_initial, Ks_extra)

            stable, Ks_initial, Ks_extra = eos_l.stability_Michelsen(T=T, P=P, zs=ys,
                                                      Ks_initial=None,
                                                      maxiter=self.stability_maxiter,
                                                      xtol=self.stability_xtol)
#            print('RESULTS YS', stable, Ks_initial, Ks_extra)
            if not stable:
                return stable, Ks_extra[-1]

            stable, Ks_initial, Ks_extra = eos_l.stability_Michelsen(T=T, P=P, zs=xs,
                                                      Ks_initial=None,
                                                      maxiter=self.stability_maxiter,
                                                      xtol=self.stability_xtol)
#            print('RESULTS XS', stable, Ks_initial, Ks_extra)
#            print('DONE STABLE PART')

            return stable, Ks_extra[-1]
        stable, Ks_z = is_stable()
#        print(stable, Ks_z, comp_test)
#        Ks_z = [zi/xi for xi, zi in zip(xs, comp_test)]
#        print(Ks_z, Ks_y)

        print('DOING RR2 with Ks', Ks_y, Ks_z)
        beta_y, beta_z, xs, ys, zs = Rachford_Rice_solution2(zs, Ks_y=Ks_y,
                                                             Ks_z=Ks_z)
        print('done RR2; betas', beta_y, beta_z, 'fractions', xs, ys, zs)
#
##        beta_z, _, _ = flash_inner_loop(comp_test, Ks_z)
#        print(beta_z, 'beta_z', 'beta_y', beta_y)

# Try to hardcode them
#        Ks_y = []

        print(eos_l.sequential_substitution_3P(Ks_y, Ks_z, beta_y, beta_z=beta_z))


    def stability_test_VL(self, T, P, zs, require_convergence=False,
                          eos=None):
        if eos is None:
            eos = self.to_TP_zs(T=T, P=P, zs=zs, fugacities=False)

        stable = True
        guess_generator = self.stability_tester.guess_generator(T=T, P=P, zs=zs,
                                                   pure=self.pure_guesses,
                                                   Wilson=self.Wilson_guesses,
                                                   random=self.random_guesses,
                                                   zero_fraction=self.zero_fraction_guesses)
        for trial_comp in guess_generator:
            for Ks in ([comp_i/zi for comp_i, zi in zip(trial_comp, zs)],
                        [zi/comp_i for comp_i, zi in zip(trial_comp, zs)]):
                # No Ks are duplicated as of 2019-02-18
                try:
                    stable, Ks_initial, Ks_extra = eos.stability_Michelsen(T=T, P=P, zs=zs,
                                                              Ks_initial=Ks,
                                                              maxiter=self.stability_maxiter,
                                                              xtol=self.stability_xtol)
                except UnconvergedError as e:
                    if require_convergence:
                        raise e
                if not stable:
                    return stable, Ks_initial, Ks_extra
        return stable, None, None


    def flash_TP_zs(self, T, P, zs, Wilson_first=True):
        info = []
        if hasattr(self, 'eos_l') and self.eos_l is not None:
            eos = self.eos_l.to_TP_zs_fast(T=T, P=P, zs=zs, full_alphas=False)
        elif hasattr(self, 'eos_g') and self.eos_g is not None:
            eos = self.eos_g.to_TP_zs_fast(T=T, P=P, zs=zs, full_alphas=False)
        else:
            eos = self.to_TP_zs(T=T, P=P, zs=zs, fugacities=False)
        if self.N == 1:
            if eos.phase == 'l/g':
                liq = eos.G_dep_l < eos.G_dep_g
            else:
                liq = eos.phase == 'l'
            if liq:
                self.eos_l = eos
                self.eos_g = None
                self.info = info
                return 'l', [1.0], None, 0.0
            else:
                self.eos_l = None
                self.eos_g = eos
                self.info = info
                return 'g', None, [1.0], 1.0

        try:
            G_dep_eos = min(eos.G_dep_l, eos.G_dep_g)
        except:
            G_dep_eos = eos.G_dep_g if hasattr(eos, 'G_dep_g') else eos.G_dep_l

        # Fast path - try the flash
        if Wilson_first:
            try:
                _, _, VF_wilson, xs_wilson, ys_wilson = flash_wilson(zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             P=P, T=T)
            except Exception as e:
#                print('wilson flash init failed')
                VF_wilson = -1
#            print(VF_wilson, xs_wilson, ys_wilson, 'VF_wilson, xs_wilson, ys_wilson')
            if 1e-7 < VF_wilson < 1.0 - 1e-7:
                try:
                    VF, xs, ys, eos_l, eos_g = eos.sequential_substitution_VL(
                                                maxiter=self.substitution_maxiter,
                                                xtol=self.substitution_xtol,
                                                near_critical=True,
                                                xs=xs_wilson, ys=ys_wilson,
                                                info=info
                                                )
                    phase = 'l/g'
                    try:
                        G_dep_l = eos_l.G_dep_l
                    except AttributeError:
                        G_dep_l = eos_l.G_dep_g
                    try:
                        G_dep_g = eos_g.G_dep_g
                    except AttributeError:
                        G_dep_g = eos_g.G_dep_l

                    # This comparison is not correct
                    G_TP = G_dep_l*(1.0 - VF) + G_dep_g*VF

#                    print(VF)
                    if VF < 0.0 or VF > 1.0 or G_TP > G_dep_eos:
                        raise ValueError("Wilson flash converged but VF unfeasible or Gibbs energy lower than stable phase")

                    self.eos_l = eos_l
                    self.eos_g = eos_g

                    eos_l.resolve_full_alphas()
                    eos_g.resolve_full_alphas()

                    self.info = info
                    return phase, xs, ys, VF


                except Exception as e:
#                    print(e, 'Wilson flash fail')
                    pass



        stable = True
        guess_generator = self.stability_tester.guess_generator(T=T, P=P, zs=zs,
                                                   pure=self.pure_guesses,
                                                   Wilson=self.Wilson_guesses,
                                                   random=self.random_guesses,
                                                   zero_fraction=self.zero_fraction_guesses)
        unstable_and_failed_SS = False
        for trial_comp in guess_generator:
            if not stable:
                break
#            print(trial_comp)
            for Ks in ([comp_i/zi for comp_i, zi in zip(trial_comp, zs)],
                        [zi/comp_i for comp_i, zi in zip(trial_comp, zs)]):

#                print(Ks)
                try:
                    stable, Ks_initial, Ks_extra = eos.stability_Michelsen(T=T, P=P, zs=zs,
                                                              Ks_initial=Ks,
                                                              maxiter=self.stability_maxiter,
                                                              xtol=self.stability_xtol)
#                    print(stable, Ks_initial)

                except UnconvergedError as e:
                    print(e, 'stability failed')

                    pass
                if not stable:
#                    print('found not stable with Ks:', Ks)
                    # two phase flash with init Ks

                    try:
                        VF, xs, ys, eos_l, eos_g = eos.sequential_substitution_VL(Ks_initial=Ks_initial,
                                            maxiter=self.substitution_maxiter,
                                            xtol=self.substitution_xtol,
                                            near_critical=True,
                                            Ks_extra=Ks_extra,
                                            info=info)


                        G_dep_l = eos_l.G_dep_l if hasattr(eos_l, 'G_dep_l') else eos_l.G_dep_g
                        G_dep_g = eos_g.G_dep_g if hasattr(eos_g, 'G_dep_g') else eos_g.G_dep_l

                        G_TP = G_dep_l*(1.0 - VF) + G_dep_g*VF

                        if VF < 0 or VF > 1 or G_TP > G_dep_eos:
                            raise ValueError("Stability test Ks flash converged but VF unfeasible or Gibbs energy higher than stable phase")

                        self.eos_l = eos_l
                        self.eos_g = eos_g
                        eos_l.resolve_full_alphas()
                        eos_g.resolve_full_alphas()
                        phase = 'l/g'
                        break
                    except Exception as e:
                        # K guesses were not close enough to convege or some other error happened
#                        print('failed convergence of SS with Ks', Ks_initial, e, 'T=%g, P=%g, zs=%s' %(T, P, zs))
                        unstable_and_failed_SS = True
                        stable = True

#                print('found stable with Ks:', Ks)
#        try:
#            print('liquid gibbs (single phase)', eos.G_dep_l)
#        except:
#            print('No liquid phase (pure)')

#        try:
#            print('vapor gibbs (single phase)', eos.G_dep_g)
#        except:
#            print('No vapor phase (pure)')
#
#        print('After stability test, stable=%g' %(stable))

        if unstable_and_failed_SS:
            raise ValueError("Flash failed - single phase detected to be unstable, but could not converge with SS")
        if stable:
            try:
                if eos.G_dep_l < eos.G_dep_g:
                    phase, xs, ys, VF = 'l', zs, None, 0
                    self.eos_l = eos
                    self.eos_g = None
                else:
                    phase, xs, ys, VF = 'g', None, zs, 1
                    self.eos_g = None
                    self.eos_l = eos
            except:
                # Only one root - take it and set the prefered other phase to be a different type
                if hasattr(eos, 'Z_l'):
                    phase, xs, ys, VF = 'l', zs, None, 0
                    self.eos_l = eos
                    self.eos_g = None
                else:
                    phase, xs, ys, VF = 'g', None, zs, 1
                    self.eos_g = None
                    self.eos_l = eos
            eos.resolve_full_alphas()
        self.info = info

        return phase, xs, ys, VF

    def flash_PVF_zs(self, P, VF, zs):
        if not 0 <= VF <= 1:
            raise ValueError("Vapor fraction out of range 0 to 1")
        if self.N == 1:
            # Do not allow above critical point!
            if P > self.Pcs[0]:
                raise ValueError("Pressure is greater than critical pressure")
            Tsat = self._Tsats(P)[0]
            self.eos_l = self.eos_g = self.eos_pure_ref(0).to_TP(T=Tsat, P=P)
#            self.eos_l = self.eos_g = self.to_TP_zs(T=Tsat, P=P, zs=zs, fugacities=False)
            return 'l/g', [1.0], [1.0], VF, Tsat
#        elif 1.0 in zs:
#            raise NotImplemented

        if VF == 0:
            xs, ys, VF, T, eos_l, eos_g = self.bubble_T(P=P, zs=zs)
        elif VF == 1:
            xs, ys, VF, T, eos_l, eos_g = self.dew_T(P=P, zs=zs)
        else:
            res = [None]
            def err(T):
                eos = self.to_TP_zs(T=T, P=P, zs=zs)
#                print(eos.zs, 'eos.zs before loop')
                VF_calc, xs, ys, eos_l, eos_g = eos.sequential_substitution_VL(Ks_initial=None,
                                                                 maxiter=self.substitution_maxiter,
                                                                 xtol=self.substitution_xtol,
                                                                 near_critical=True,
                                                                 xs=xs_guess, ys=ys_guess)
                res[0] = (VF_calc, xs, ys, eos_l, eos_g)
#                print(P, VF_calc - VF, res)
                return VF_calc - VF

            _, xs_guess, ys_guess, _, T_guess_as_pure = self.flash_PVF_zs_ideal(P=P, VF=VF, zs=zs)
#            print(xs_guess, ys_guess, 'hi')
            T = None
            try:
                T = float(newton(err, T_guess_as_pure, tol=self.FLASH_VF_TOL))
            except:
                pass
            if T is None:
                from scipy.optimize import fsolve
                try:
                    T = float(fsolve(err, T_guess_as_pure))
                except:
                    pass
#            print(P, 'worked!')
            if T is None:
                T = float(brenth(err, .9*T_guess_as_pure, 1.1*T_guess_as_pure))
            VF, xs, ys, eos_l, eos_g = res[0]

        self.eos_l = eos_l
        self.eos_g = eos_g
        return 'l/g', xs, ys, VF, T


    def flash_TVF_zs(self, T, VF, zs):
        if not 0 <= VF <= 1:
            raise ValueError("Vapor fraction out of range 0 to 1")
        if self.N == 1:
            if T > self.Tcs[0]:
                raise ValueError("Temperature is greater than critical Temperature")
            Psat = self._Psats(T)[0]
            self.eos_l = self.eos_g = self.to_TP_zs(T=T, P=Psat, zs=zs, fugacities=False)
            return 'l/g', [1.0], [1.0], VF, Psat
#        elif 1.0 in zs:
#            raise NotImplemented
#            return 'l/g', list(zs), list(zs), VF, Psats[zs.index(1.0)]
        if VF == 0:
            xs, ys, VF, P, eos_l, eos_g = self.bubble_P(T=T, zs=zs)
        elif VF == 1:
            xs, ys, VF, P, eos_l, eos_g = self.dew_P(T=T, zs=zs)
        else:
            res = [None]
            def err(P):
                P = float(P)
#                print('P guess', P)
                eos = self.to_TP_zs(T=T, P=P, zs=zs)
                VF_calc, xs, ys, eos_l, eos_g = eos.sequential_substitution_VL(Ks_initial=None,
                                                                 maxiter=self.substitution_maxiter,
                                                                 xtol=self.substitution_xtol,
                                                                 near_critical=True,
                                                                 xs=xs_guess, ys=ys_guess
                                                                 )
                res[0] = (VF_calc, xs, ys, eos_l, eos_g)
#                print(P, VF_calc - VF, res)
                return VF_calc - VF

            _, xs_guess, ys_guess, _, P_guess_as_pure = self.flash_TVF_zs_ideal(T=T, VF=VF, zs=zs)
            P = None
#            print('P_guess_as_pure', P_guess_as_pure)
            try:
                P = newton(err, P_guess_as_pure, tol=self.FLASH_VF_TOL)
            except Exception as e:
#                print(e, 'newton failed')
                pass
            if P is None:
                try:
                    from scipy.optimize import fsolve
                    P = float(fsolve(err, P_guess_as_pure, xtol=self.FLASH_VF_TOL))
                except Exception as e:
#                    print(e, 'fsolve failed')
                    pass
#            print(P, 'worked!')
            if P is None:
                try:
                    P = float(brenth(err, .9*P_guess_as_pure, 1.1*P_guess_as_pure))
                except Exception as e:
                    pass
                    # Compute the dew and bubble points
            VF, xs, ys, eos_l, eos_g = res[0]

        self.eos_l = eos_l
        self.eos_g = eos_g
        return 'l/g', xs, ys, VF, P

    def PH_Michelson(self, T_guess, P, zs, H_goal, maxiter=100, tol=1e-6,
                     VF_guess_init=None, damping=1.0, analytical=True,
                     max_T_step=30):

        Ks = [Wilson_K_value(T_guess, P, Tci, Pci, omega) for Pci, Tci, omega in
              zip(self.Pcs, self.Tcs, self.omegas)]
        V_over_F, xs, ys = flash_inner_loop(zs=zs, Ks=Ks)
        if VF_guess_init is not None:
            V_over_F = VF_guess_init

        store = {}
        store['xs'] = xs
        store['ys'] = ys
        def err_fun(T_V_over_F, zs, Ks, jac=True):
            T, V_over_F = float(T_V_over_F[0]), float(T_V_over_F[1])
#            print('calling', T, V_over_F)
#            print('Ks', Ks)
            try:
                V_over_F2, xs, ys = flash_inner_loop(zs=zs, Ks=Ks, check=True)
            except:
                # Likely proceeded to Ks which have no solution
                Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in
                      zip(self.Pcs, self.Tcs, self.omegas)]
                V_over_F2, xs, ys = flash_inner_loop(zs=zs, Ks=Ks, check=True)
#            print(Ks)
#            xs = [zi/(1.+V_over_F*(Ki-1.)) for zi, Ki in zip(zs, Ks)] # if zi != 0.0
#            ys = [Ki*xi for xi, Ki in zip(xs, Ks)]

#            print(xs, ys, 'xs and ys')
            eos_l = self.to_TP_zs(T=T, P=P, zs=xs, fugacities=True)
            eos_g = self.to_TP_zs(T=T, P=P, zs=ys, fugacities=True)

            # Calculate the enthalpy - TODO do it here!
            T_REF_IG = self.T_REF_IG
            HeatCapacityGases = self.HeatCapacityGases
            dH_integrals = Hils = Higs = [obj.T_dependent_property_integral(T_REF_IG, T) for obj in HeatCapacityGases]
            Cpls = Cpgs = [obj.T_dependent_property(T) for obj in HeatCapacityGases]

            try:
                H_dep_g = eos_g.H_dep_g
            except AttributeError:
                H_dep_g = eos_g.H_dep_l
            try:
                H_dep_l = eos_l.H_dep_l
            except AttributeError:
                H_dep_l = eos_l.H_dep_g

            try:
                dH_dep_dT_g = eos_g.dH_dep_dT_g
            except AttributeError:
                dH_dep_dT_g = eos_g.dH_dep_dT_l
            try:
                dH_dep_dT_l = eos_l.dH_dep_dT_l
            except AttributeError:
                dH_dep_dT_l = eos_l.dH_dep_dT_g

            H_g, H_l = H_dep_g, H_dep_l
            for xi, yi, dH in zip(xs, ys, dH_integrals):
                H_g += yi*dH
                H_l += xi*dH
            H_calc = H_g*V_over_F + H_l*(1.0 - V_over_F)



#            H_calc = self.enthalpy_eosmix(T, P, V_over_F, zs, xs, ys, eos_l, eos_g, 'l/g')
#            print(H_calc2, H_calc, T)
            g2 = H_calc - H_goal

            lnphis_l = eos_l._eos_lnphis_lowest_Gibbs()[0]
            lnphis_g = eos_g._eos_lnphis_lowest_Gibbs()[0]

            Ks = [exp(l - g) for l, g in zip(lnphis_l, lnphis_g)]
            g1 = Rachford_Rice_flash_error(V_over_F, zs, Ks)

#            print(T, V_over_F)

            store['eos_l'] = eos_l
            store['eos_g'] = eos_g
            store['Ks'] = Ks
            store['xs'] = xs
            store['ys'] = ys

            ts = [1.0 + V_over_F*(Ki - 1.0) for Ki in Ks]


            d_RR1_d_beta1 = 0.0
            for ti, zi, Ki in zip(ts, zs, Ks):
                d_RR1_d_beta1 -= zi/(ti*ti)*(Ki - 1.0)*(Ki - 1.0)

            _, dKs_dT = self.Ks_and_dKs_dT(eos_l, eos_g, xs, ys)

            # Derived as follows:
            '''
            from sympy import *
            zi, Ki, VF, T = symbols('zi, Ki, VF, T ')
            expr = zi*(Ki(T)-1)/(1+VF*(Ki(T)-1))
            simplify(diff(expr, T))
            '''
            d_RR1_dT = 0.0
            for dK_dTi, ti, zi, Ki in zip(dKs_dT, ts, zs, Ks):
                den_inv = 1.0/(V_over_F*(Ki - 1.0) + 1.0)
#                zi*dK_dTi/(VF*(Ki - 1) + 1)**2
                d_RR1_dT +=zi*dK_dTi*den_inv*den_inv

#                d_RR1_dT += zi/(ti*ti)*(ti*Ki*dK_dTi - V_over_F*Ki*(Ki - 1.0)*dK_dTi)

#                d_RR1_dT += -V_over_F*zi*(Ki - 1.)*dK_dTi/(V_over_F*(Ki - 1.) + 1.)**2 + zi*dK_dTi/(V_over_F*(Ki - 1.) + 1.)

#            print('start2')
            '''Confirmed easily!
            from sympy import *
            zi, Ki, VF, T = symbols('zi, Ki, VF, T ')
            x1, x2, y1, y2, H1l, H2l, H1g, H2g, Hdepl, Hdepg = symbols('x1, x2, y1, y2, H1l, H2l, H1g, H2g, Hdepl, Hdepg')
            expr = VF(T)*(Hdepg(T) + H1g(T)*y1(T) + H2g(T)*y2(T)) + (1 - VF(T))*(Hdepl(T) + H1l(T)*x1(T) + H2l(T)*x2(T))
            diff(expr, VF(T))
            '''
#            d_H_d_beta = (self.enthalpy_eosmix(T, P, V_over_F, ys, ys, ys, eos_l, eos_g, 'g')
#                          - self.enthalpy_eosmix(T, P, V_over_F, xs, xs, xs, eos_l, eos_g, 'l'))
#            print('end2')
            d_H_d_beta = H_g - H_l




            # This is simply not correct!
            # Actually, this is not even used.
#            tot1, tot2 = 0.0, 0.0
##            d_beta_d_T = 0.0
#            for dK_dT, zi, Ki in zip(dKs_dT, zs, Ks):
#                tot1 += zi*dK_dT
#                tot2 += (1.0 - Ki)*(1.0 - Ki)*zi
#
#            d_beta_d_T = tot1/tot2
#            d_beta_d_T2 = -d_beta_d_T

            delta = 1e-3
            Ks_perturb = [Ki + dKi*delta for Ki, dKi in zip(Ks, dKs_dT)]
            VF_perturb, xs2, ys2 = flash_inner_loop(zs, Ks_perturb, guess=V_over_F)
            d_beta_d_T = (VF_perturb - V_over_F)/delta
            d_beta_d_T2 = -d_beta_d_T



            # Take the composition derivatives from the RR
            dx_dTs = [(x2 - x1)/delta for x1, x2 in zip(xs, xs2)]
            dy_dTs = [(y2 - y1)/delta for y1, y2 in zip(ys, ys2)]


            # It is believed this is wrong! Need a set of dx_dT for liquid phase!
#            print('TO TEST', Ks, dKs_dT, V_over_F, d_beta_d_T)
#            dy_dTs = []
#            for dK_dT, ti, zi, Ki in zip(dKs_dT, ts, zs, Ks):
#                dyi_dT = (zi*dK_dT*ti - zi*Ki*(d_beta_d_T*(Ki - 1.0) + V_over_F*dK_dT))/(ti*ti)
#                dy_dTs.append(dyi_dT)
#
#            dx_dTs = []
#            for dK_dT, ti, zi, Ki in zip(dKs_dT, ts, zs, Ks):
#                dxi_dT = (zi*dK_dT*ti - zi*Ki*(d_beta_d_T2*(Ki - 1.0) + V_over_F*dK_dT))/(ti*ti)
#                dx_dTs.append(dxi_dT)
##
#            dx_dTs2 = []
#            dy_dTs2 = []
#            for Ki, dK_dTi, zi in zip(Ks, dKs_dT, zs):
#                x0 = V_over_F
#                x1 = Ki
#                x2 = x1 - 1.0
#                x3 = x0*x2 + 1.0
#                x4 = zi/x3**2
#                x5 = dK_dTi
#                x6 = x0*x5 + x2*d_beta_d_T
#                dx_dTs2.append(-x4*x6)
#                dy_dTs2.append(-x1*x6 + x3*x5)
#            dx_dTs, dy_dTs = dx_dTs2, dy_dTs2
#            print(dx_dTs, dx_dTs2)
#            print(dy_dTs, dy_dTs2)








            tot1 = 0.0 # sum Hi(L)*dxi/dT
            tot2 = 0.0 # sum xi*dHil_dT = xi*Cpl

            tot3 = 0.0 # sum Hi(g)*dyi_dT
            tot4 = 0.0 # sum yi*dHig_dT = yi*Cpg

            # tot5 and tot6 are zero in the first point because of starting at the reference point
            tot5 = 0.0 # sum Hi(g)*yi
            tot6 = 0.0 # sum Hil*xi
            for xi, yi, Hil, Hig, Cpl, Cpg, dxi_dT, dyi_dT in zip(xs, ys, Hils, Higs, Cpls, Cpgs, dx_dTs, dy_dTs):
                tot1 += Hil*dxi_dT
                tot2 += xi*Cpl

                tot3 += Hig*dyi_dT
                tot4 += yi*Cpg

#                print(Hig*yi, Hil*xi, xi, yi, Hig, Hil, 'in loop')
                tot5 += Hig*yi
                tot6 += Hil*xi

#            print(d_beta_d_T*(tot5 + H_dep_g), d_beta_d_T*(tot6 + H_dep_l), tot5, H_dep_g, tot6, H_dep_l)
            d_H_dT = ((1.0 - V_over_F)*(tot1 + tot2 + dH_dep_dT_l)
#                       + d_beta_d_T*(tot5 + H_dep_g)
#                       - d_beta_d_T*(tot6 + H_dep_l)
                       + V_over_F*(tot3 + tot4 + dH_dep_dT_g))

            # Can I finite difference d_H_dT using the xs2, ys2, dH_dep_l and g?
            tot1, tot2, tot3, tot4  = 0.0, 0.0, 0.0, 0.0

            # If H_goal is not used as part of the expression, it is the same as one of the ones above
            H_goal_inv = 1.0/H_goal
#            H_goal_inv = 1.0
            for dK_dT, ti, zi, Ki, dx_dT, dy_dT, obj, xi, yi, dH, Cp in zip(dKs_dT, ts, zs, Ks, dx_dTs, dy_dTs, HeatCapacityGases, xs, ys, dH_integrals, Cpls):
                x1 = dH*H_goal_inv
                x2 = Cp

                tot1 += dx_dT*x1
                tot2 += xi*H_goal_inv*x2

                tot3 += dy_dT*x1
                tot4 += yi*H_goal_inv*x2



            d_H_dT = (V_over_F*(tot3 + tot4 + dH_dep_dT_g)

#                       + d_beta_d_T*(tot5 + H_dep_g)
#                       - d_beta_d_T*(tot6 + H_dep_l)

                      + (1.0 - V_over_F)*(tot1 + tot2 + dH_dep_dT_l))
#            print(d_H_dT, d_H_dT2, 'two values')

            store['dKs_dT'] = dKs_dT
#            Hg_ideal(self, T, zs), Cpg_ideal(self, T, zs)



#            jacobian = [[d_RR1_d_beta1, d_RR1_dT], [d_H_d_beta, d_H_dT]]
            jacobian = [[d_RR1_dT, d_RR1_d_beta1], [d_H_dT, d_H_d_beta]]
#            print(jacobian)
            if jac:
                return [g1, g2], jacobian

            return [g1, g2]

        def to_Jac(T_V_over_F):
            import numpy as np
#            print('calling for jac')
            return np.array(err_fun(T_V_over_F, zs, Ks, jac=False))

        Ts_attempt = [T_guess]
        VFs_attempt = [V_over_F]
        iter = 0
#        analytical = False

        while iter < maxiter:
            fcur, j_analytical = err_fun([T_guess, V_over_F], zs, Ks, jac=True)

            err =  abs(fcur[0]) + abs(fcur[1])
#            print(T_guess, V_over_F, fcur)

            if err < tol:
                break

#            if not analytical:
#            try:
#                from numdifftools.core import Jacobian
#            except:
#                pass
#
#            j_obj = Jacobian(to_Jac, step=1e-5)
#            j = j_obj([T_guess, V_over_F])
#            print('CURRENT ERROR', fcur)
#            print('ANALYTICAL JACOBIAN', j_analytical)
#            print('NUMERICAL JACOBIAN', j.tolist())
#            print('JACOBIAN RATIO', (j/j_analytical).tolist())
#    #            print(j)
    #            print(j_analytical)

            if analytical:
                j = j_analytical


#            break

            dx = py_solve(j, [-v for v in fcur])
            # The damping actually makes it take fewer iterations
#            if abs(dx[0]) < 2 :
#                damping = 1
            T_step = dx[0]*damping
            if abs(T_step) > max_T_step:
                T_step = copysign(max_T_step, T_step)
            T_guess = T_guess + T_step

            V_over_F = V_over_F + dx[1]
            # Try to reduce overstepping?
#            if V_over_F < 0:
#                V_over_F = 0.5*V_over_F
#            elif V_over_F > 1:
#                V_over_F = 1 + 0.5*(V_over_F % 1)



#            T_guess2, V_over_F = [xi + dxi*damping for xi, dxi in zip([T_guess, V_over_F], dx)]
#            if abs(T_guess - T_guess2) > max_T_step:
#                T_guess2 =


            dKs_dT = store['dKs_dT']
#            _, dKs_dT = self.Ks_and_dKs_dT(store['eos_l'], store['eos_g'], store['xs'], store['ys'])

            dT = T_guess - Ts_attempt[-1]
            Ks = store['Ks']


            # diff(log(f(x)), y) = Derivative(f(x), x)/f(x)
            Ks = [exp(log(K) + dK_dT/K*dT) for K, dK_dT in zip(Ks, dKs_dT)]

            Ts_attempt.append(T_guess)
            VFs_attempt.append(V_over_F)

            iter += 1
        if iter == maxiter:
            raise ValueError("Did not converge PH 2 phase flash")
        # What needs to be returned?
        eos_l = store['eos_l']
        eos_g = store['eos_g']
        xs = store['xs']
        ys = store['ys']

        self.eos_l, self.eos_g = eos_l, eos_g
#        print(iter)
        return 'l/g', xs, ys, V_over_F, T_guess

    def PH_Agarwal(self, T_guess, P, zs, H_goal, maxiter=100, tol=1e-6,
                     VF_guess_init=None, damping=1.0):
        import numpy as np
        from numdifftools.core import Jacobian
        from fluids.numerics import py_solve

        Ks = [Wilson_K_value(T_guess, P, Tci, Pci, omega) for Pci, Tci, omega in
              zip(self.Pcs, self.Tcs, self.omegas)]
        V_over_F, xs, ys = flash_inner_loop(zs=zs, Ks=Ks)
        if VF_guess_init is not None:
            V_over_F = VF_guess_init

        store = {}
        Ks_prev = []
        fs_prev = []

        eos_l = self.to_TP_zs(T=T, P=P, zs=xs, fugacities=True)
        eos_g = self.to_TP_zs(T=T, P=P, zs=ys, fugacities=True)

        # Calculate the enthalpy
        H_calc = self.enthalpy_eosmix(T_guess, P, V_over_F, zs, xs, ys, eos_l, eos_g, 'l/g')
        g2 = H_calc - H_goal

        # Step 5
        lnphis_l = eos_l._eos_lnphis_lowest_Gibbs()[0]
        lnphis_g = eos_g._eos_lnphis_lowest_Gibbs()[0]
        Cp_l = self.dH_dT(T, P, V_over_F, zs, xs, ys, eos_l, eos_g, 'l')
        Cp_g = self.dH_dT(T, P, V_over_F, zs, xs, ys, eos_l, eos_g, 'g')

        # Step 6
        fs = []
        for lnphi_l, lnphi_g, xi, yi in zip(lnphis_l, lnphis_g, xs, ys):
            fs.append(lnphi_g + log(yi) - lnphi_l - log(xi))

        # Step 7
        T_guess = T_guess - g2/(V_over_F*Cp_g + (1.0 - V_over_F)*Cp_l)

        Ks = [exp(l - g) for l, g in zip(lnphis_l, lnphis_g)]
        V_over_F, xs, ys = flash_inner_loop(zs=zs, Ks=Ks)

        def err_fun(T_V_over_F, zs, Ks):
#            print('calling', Ks)
            T, V_over_F = float(T_V_over_F[0]), float(T_V_over_F[1])

            # step 8
            V_over_F, xs, ys = flash_inner_loop(zs=zs, Ks=Ks)

            # Step 9
            eos_l = self.to_TP_zs(T=T, P=P, zs=xs, fugacities=True)
            eos_g = self.to_TP_zs(T=T, P=P, zs=ys, fugacities=True)
            H_calc = self.enthalpy_eosmix(T_guess, P, V_over_F, zs, xs, ys, eos_l, eos_g, 'l/g')
            g2 = H_calc - H_goal

            # Step 10
            lnphis_l = eos_l._eos_lnphis_lowest_Gibbs()[0]
            lnphis_g = eos_g._eos_lnphis_lowest_Gibbs()[0]

            fs_prev = fs
            fs = []
            for lnphi_l, lnphi_g, xi, yi in zip(lnphis_l, lnphis_g, xs, ys):
                fs.append(lnphi_g + log(yi) - lnphi_l - log(xi))

            Ks_prev = Ks
            Ks = [exp(l - g) for l, g in zip(lnphis_l, lnphis_g)]
            # Step 11
            Ks05 = []
            for K, K_prev, dK_dT, f, f_prev in zip(Ks, Ks_prev, dKs_dT, fs, fs_prev):
                lnK05 = log(K) + (log(K) - log(K_prev))*f_prev/((log(K) - log(K_prev))*(f - f_prev))*f
                Ks05.append(exp(lnK05))

            # Jacobian is supposed to be constructed based on the Ks05 and so on
            # Nothing fundamentally better anout this update; just a little cleverer.
            V_over_F_05, xs_05, ys_05 = flash_inner_loop(zs=zs, Ks=Ks05)


            g1 = Rachford_Rice_flash_error(V_over_F_05, zs, Ks05)

            store['eos_l'] = eos_l
            store['eos_g'] = eos_g
            store['Ks'] = Ks
            store['xs'] = xs
            store['ys'] = ys

            return [g1, g2]

        def to_Jac(T_V_over_F):
#            print('calling for jac')
            return np.array(err_fun(T_V_over_F, zs, Ks))

        Ts_attempt = [T_guess]
        VFs_attempt = [V_over_F]
        iter = 0

        while iter < maxiter:
            fcur = err_fun([T_guess, V_over_F], zs, Ks)

            err =  abs(fcur[0]) + abs(fcur[1])
#            print(T_guess, V_over_F, fcur)

            if err < tol:
                break

            j_obj = Jacobian(to_Jac, step=1e-4)
            j = j_obj([T_guess, V_over_F])


            break



            dx = py_solve(j, [-v for v in fcur])
            T_guess, V_over_F = [xi + dxi*damping for xi, dxi in zip([T_guess, V_over_F], dx)]

            _, dKs_dT = self.Ks_and_dKs_dT(store['eos_l'], store['eos_g'], store['xs'], store['ys'])

            dT = T_guess - Ts_attempt[-1]
            Ks = store['Ks']


            # diff(log(f(x)), y) = Derivative(f(x), x)/f(x)
            Ks = [exp(log(K) + dK_dT/K*dT) for K, dK_dT in zip(Ks, dKs_dT)]

            Ts_attempt.append(T_guess)
            VFs_attempt.append(V_over_F)


    def P_HS_error_1P(self, T, P, zs, H_goal=None, S_goal=None, info=None):
        if self.N == 1:
            eos_phase = self.eos_pure_ref(0).to_TP(T=T, P=P)
        else:
            if hasattr(self, 'eos_l') and self.eos_l is not None:
                eos_phase = self.eos_l.to_TP_zs_fast(T=T, P=P, zs=zs)
            elif hasattr(self, 'eos_g') and self.eos_g is not None:
                eos_phase = self.eos_g.to_TP_zs_fast(T=T, P=P, zs=zs)
            else:
                eos_phase = self.to_TP_zs(T=T, P=P, zs=zs, fugacities=False)
        phase = eos_phase.more_stable_phase
        if phase == 'l':
            eos_l, eos_g = eos_phase, None
        else:
            eos_l, eos_g = None, eos_phase
        if H_goal is not None:
            H_calc = self.enthalpy_eosmix(T, P, None, zs, None, None, eos_l, eos_g, phase)
            err = H_calc - H_goal
        else:
            S_calc = self.entropy_eosmix(T, P, None, zs, None, None, eos_l, eos_g, phase)
            err = S_calc - S_goal

        if info is not None:
            info[:] = (eos_phase, phase, err)
#        print(T, err, info)
        return err

    def P_HS_error_and_der_1P(self, T, P, zs, H_goal=None, S_goal=None, info=None):
#        print(T, 'T')
        if self.N == 1:
            eos_phase = self.eos_pure_ref(0).to_TP(T=T, P=P)
        else:
            if hasattr(self, 'eos_l') and self.eos_l is not None:
                eos_phase = self.eos_l.to_TP_zs_fast(T=T, P=P, zs=zs)
            elif hasattr(self, 'eos_g') and self.eos_g is not None:
                eos_phase = self.eos_g.to_TP_zs_fast(T=T, P=P, zs=zs)
            else:
                eos_phase = self.to_TP_zs(T=T, P=P, zs=zs, fugacities=False)
        phase = eos_phase.more_stable_phase
        if phase == 'l':
            eos_l, eos_g = eos_phase, None
        else:
            eos_l, eos_g = None, eos_phase
        if H_goal is not None:
            H_calc = self.enthalpy_eosmix(T, P, None, zs, None, None, eos_l, eos_g, phase)
            err = H_calc - H_goal
            dErr_dT = self.dH_dT(T, P, None, zs, zs, zs, eos_l, eos_g, phase)
        else:
            S_calc = self.entropy_eosmix(T, P, None, zs, None, None, eos_l, eos_g, phase)
            err = S_calc - S_goal
            dErr_dT = self.dS_dT(T, P, None, zs, zs, zs, eos_l, eos_g, phase)
#        print(T, err, dErr_dT, 'dErr_dT')
        if info is not None:
            info[:] = (eos_phase, phase, err, dErr_dT)
        return err, dErr_dT


    def flash_PH_zs_bounded_1P(self, P, Hm, zs, T_low=None, T_high=None):
        # Begin the search at half the lowest chemical's melting point
        if T_low is None:
            T_low = 0.5*min(self.Tms)

        # Cap the T high search at 8x the highest critical point
        # (will not work well for helium, etc.)
        if T_high is None:
            max_Tc = max(self.Tcs)
            if max_Tc < 100:
                T_high = 4000.0
            else:
                T_high = max_Tc*8.0
        info = []

        T = brenth(self.P_HS_error_1P, T_low, T_high, rtol=1e-8, args=(P, zs, Hm, None, info))
            # TODO stability test, 1 component
        return T, info[1], info[0], None


    def PH_T_guesses_1P(self, P, Hm, zs, T_guess=None):
        i = -1 if T_guess is not None else 0
        while i < 3:
            try:
                if i == -1:
                    yield T_guess
                elif i == 0:
                    MW = mixing_simple(zs, self.MWs)
                    sv = mixing_simple(zs, self.n_atoms)/MW
                    T_calc = Lastovka_Shaw_T_for_Hm(Hm=Hm, MW=MW, similarity_variable=sv)
                    if 1 or T_calc > 0.0:
                        yield T_calc
                elif i == 1:
                    Tc = mixing_simple(zs, self.Tcs)
                    omega = mixing_simple(zs, self.omegas)
                    def Dadgostar_Shaw_T_guess(Hm, MW, similarity_variable, Tc, omega,
                                               T_ref=298.15, factor=1.0):
                        H_ref = Dadgostar_Shaw_integral(T_ref, similarity_variable)
                        def err(T):
                            Hvap = SMK(T, Tc, omega)
                            H1 = Dadgostar_Shaw_integral(T, similarity_variable)
                            dH = H1 - H_ref
                            # Limitation to higher T unfortunately
                            return ((property_mass_to_molar(dH, MW)*factor - Hvap) - Hm)
                        return newton(err, 100, high=Tc)

                    T_calc = Dadgostar_Shaw_T_guess(Hm, MW, sv, Tc, omega, factor=1)
                    if 1 or T_calc > 0.0:
                        yield T_calc
                elif i == 2:
                    yield 298.15

            # with Hvap and Cpl can get another guess - but is a little more complicated
            except Exception as e:
#                print(e)
                pass
            i += 1

    def PS_T_guesses_1P(self, P, Sm, zs, T_guess=None):
        i = -2 if T_guess is not None else -1

        Sm -= R*sum([zi*log(zi) for zi in zs if zi > 0.0]) # ideal composition entropy composition
        Sm -= R*log(P*self.P_REF_IG_INV)
        # It would be nice to include a non-ideal component - maybe from a virial correlation

        while i < 3:
            try:
                if i == -2:
                    yield T_guess
                elif i == -1 and self.N == 1:
                    yield 298.15
                elif i == -100:
                    MW = mixing_simple(zs, self.MWs)
                    sv = mixing_simple(zs, self.n_atoms)/MW
                    Tc = mixing_simple(zs, self.Tcs)
                    Pc = mixing_simple(zs, self.Pcs)
                    omega = mixing_simple(zs, self.omegas)


#                    from thermo.heat_capacity import Lastovka_Shaw_integral_over_T
#                    def Lastovka_Shaw_T_for_Sm(Sm, MW, similarity_variable, T_ref=298.15,
#                                               factor=1.0):
#                        S_ref = Lastovka_Shaw_integral_over_T(T_ref, similarity_variable)
#                        def err(T):
#                            S1 = Lastovka_Shaw_integral_over_T(T, similarity_variable)
#                            dS = S1 - S_ref
#
#                            # May save an iteration
#                            Tr = T/Tc
#                            Pr = P/Pc
#                            dS_virial = R*Pr/Tr**2.6*(.675 + .722*omega/Tr**2.6)
##                            dS_virial = R*Pr/Tr**2*(.33 + .277/Tr + .036/Tr**2 + .0049/Tr**7 + omega*(-0.662 + 1.269/Tr**2 + .064/Tr**7))
#                            print(property_mass_to_molar(dS,  MW)*factor, dS_virial, Sm, 'T', T)
#                            err = (property_mass_to_molar(dS,  MW)*factor - dS_virial - Sm)
#                    #         print(T, err)
#                            return err
#                        try:
#                            return newton(err, 500, ytol=1e-4)
#                        except Exception as e:
#                            try:
#                                return brenth(err, 1e-3, 1e5)
#                            except Exception as e:
#                                if err(1e-11) > 0:
#                                    raise ValueError("For gas only entropy spec to be correct, "
#                                                     "model requires negative temperature")
#                                raise e
#                    T = Lastovka_Shaw_T_for_Sm(Sm=Sm, MW=MW, similarity_variable=sv)
#                    print(T, 'guess')
#                    if T > 10:
#                        yield T



                elif i == 0:
                    MW = mixing_simple(zs, self.MWs)
                    sv = mixing_simple(zs, self.n_atoms)/MW
                    T = Lastovka_Shaw_T_for_Sm(Sm=Sm, MW=MW, similarity_variable=sv)
#                    print(T, 'guess')
                    if T > 10:
                        yield T
                elif i == 1:
                    Tc = mixing_simple(zs, self.Tcs)
                    omega = mixing_simple(zs, self.omegas)
                    def Dadgostar_Shaw_T_guess(Sm, MW, similarity_variable, Tc, omega,
                                               T_ref=298.15, factor=1.0):
                        S_ref = Dadgostar_Shaw_integral_over_T(T_ref, similarity_variable)
                        def err(T):
                            Hvap = SMK(T, Tc, omega)
                            S1 = Dadgostar_Shaw_integral_over_T(T, similarity_variable)
                            dS = S1 - S_ref
                            # Limitation to higher T unfortunately
                            return ((property_mass_to_molar(dS, MW)*factor - Hvap/T) - Sm)
                        return newton(err, 100, high=Tc)

                    T = Dadgostar_Shaw_T_guess(Sm, MW, sv, Tc, omega, factor=1)
#                    print(T, 'T guess liquid')
                    if T > 0.0:
                        yield T
                elif i == 2:
                    yield 298.15

            except Exception as e:
#                print(e)
                pass
            i += 1


    def PH_T_guesses_2P(self, P, Hm, zs, T_guess=None):
        i = -1 if T_guess is not None else 0
        while i < 3:
            try:
                if i == -1:
                    yield T_guess
                elif i == 0 and self.N > 1:
                    MW = mixing_simple(zs, self.MWs)
                    sv = mixing_simple(zs, self.n_atoms)/MW

                    def approx_H(Hm, P, MW, similarity_variable, Tcs, Pcs, omegas,
                                 T_ref=298.15,
                                 factor=1.0):
                        Tc = mixing_simple(zs, Tcs)
                        omega = mixing_simple(zs, omegas)
                        H_ref_LS = Lastovka_Shaw_integral(T_ref, similarity_variable)
                        H_ref_DS = Dadgostar_Shaw_integral(T_ref, similarity_variable)
                        def Hm_approx_basic(T, V_over_F):
                            if V_over_F > 1:
                                V_over_F = 1
                            elif V_over_F < 0:
                                V_over_F = 0
                            H1 = Lastovka_Shaw_integral(T, similarity_variable)
                            dH = H1 - H_ref_LS
                            H_gas = property_mass_to_molar(dH, MW)*factor

                            if V_over_F < 1:
                                Hvap = SMK(T, Tc, omega)
                                H1 = Dadgostar_Shaw_integral(T, similarity_variable)
                                dH = H1 - H_ref_DS
                                H_liq = property_mass_to_molar(dH, MW)*factor
                            else:
                                H_liq = 0
                                Hvap = 0

                            return H_gas*V_over_F + (1.0 - V_over_F)*(H_liq - Hvap)

                        def to_solve(T):
                            _, _, VF, _, _ = flash_wilson(zs, Tcs=Tcs, Pcs=Pcs, omegas=omegas, T=T, P=P)
                            H_calc = Hm_approx_basic(T=T, V_over_F=VF)
                            return H_calc - Hm
                        return secant(to_solve, 300, xtol=None, ytol=10)

                    yield approx_H(Hm, P, MW, sv, self.Tcs, self.Pcs, self.omegas)
                elif i == 1:
                    yield 298.15

            # with Hvap and Cpl can get another guess - but is a little more complicated
            except Exception as e:
                print(e)
                pass
            i += 1

    def flash_P_HS_2P_N1(self, P, zs, Hm=None, Sm=None):
        # Solves only when two phase
        _, _, _, _, T = self.flash_PVF_zs(P, 1.0, zs)
        eos_l, eos_g = self.eos_l, self.eos_g
        if Hm is not None:
            H_l = self.enthalpy_eosmix(T, P, 0.0, zs, zs, zs, eos_l, eos_g, 'l')
            H_g = self.enthalpy_eosmix(T, P, 1.0, zs, zs, zs, eos_l, eos_g, 'g')
            VF = (Hm - H_l)/(H_g - H_l)
        else:
            S_l = self.entropy_eosmix(T, P, 0.0, zs, zs, zs, eos_l, eos_g, 'l')
            S_g = self.entropy_eosmix(T, P, 1.0, zs, zs, zs, eos_l, eos_g, 'g')
            VF = (Sm - S_l)/(S_g - S_l)

        return T, 'l/g', eos_l, VF

    def flash_P_HS_1P(self, P, zs, Hm=None, Sm=None, T_guess=None,
                      minimum_progress=0.3):
        r'''One phase direct solution to the pressure-enthalpy
        or pressure-entropy flash problem.

        The algorithm used is:
            * Obtain a temperature guess
            * Apply newton's method, calculating analytical derivatives
              (limit T steps to prevent negative temperatures (go half way);
              detect oscillations (normally when a phase change occurs))
            * If oscillations are detected, and the system is 1 component,
              solve the the saturation pressure - and try to solve the problem
              as a two-phase system
            * If oscillations are detected and the system is not 1 component
              (or the two-phase solution was not correct), use a bounded solver
              starting with the two closest bounding values from the newton
              iterations.

        Guesses are obtained from the method `PH_T_guesses_1P`; if one guess
        does not lead to success for any method, the next is tried until all
        available guesses have been attempted.

        '''
        solve_H = Hm is not None
        guesses = []
        if solve_H:
            guess_generator = self.PH_T_guesses_1P(P, Hm, zs, T_guess=T_guess)
        else:
            guess_generator = self.PS_T_guesses_1P(P, Sm, zs, T_guess=T_guess)

        info = []
        def to_solve(T):
            err_and_der = self.P_HS_error_and_der_1P(T, P, zs, Hm, Sm, info=info)
            return err_and_der
        to_solve, checker = oscillation_checking_wrapper(to_solve, full=True,
                                                         minimum_progress=minimum_progress)

        ans = None
        for T_trial in guess_generator:
            guesses.append(T_trial)
            try:
                ans = newton(to_solve, T_trial, fprime=True, require_eval=True,
                             damping_func=damping_maintain_sign, maxiter=100,
                             bisection=True)
                break
            except (OscillationError, Exception) as e:
                if not isinstance(e, (OscillationError, UnconvergedError)):
                    if solve_H:
                        msg = "Unexpected failure of newton's method at P=%s, Hm=%s; %s" %(P, Hm, e)
                    else:
                        msg = "Unexpected failure of newton's method at P=%s, Sm=%s; %s" %(P, Sm, e)
                    print(msg)

                wrapped_P_HS_error_1P, _, info_cache = caching_decorator(self.P_HS_error_1P, full=True)
                if self.N > 1:
                    wrapped_P_HS_error_1P = oscillation_checking_wrapper(wrapped_P_HS_error_1P, full=False,
                                                                         minimum_progress=minimum_progress)
                if self.N == 1 and P <= self.Pcs[0]:
                    try:
                        T, phase, eos, VF = self.flash_P_HS_2P_N1(P, zs, Hm, Sm)
                    except Exception as e:
                        print('2 Phase 1 component solver could not converge', e)
                        pass # Does not work often -
                    if VF < 0 or VF > 1:
                        # Use a bounded solver; will have data available in the oscillation checker
                        if VF < 0:
                            last_limit = checker.xs_neg[-1]
                            last_err = checker.ys_neg[-1]
                        else:
                            last_limit = checker.xs_pos[-1]
                            last_err = checker.ys_pos[-1]
                        try:
                            ans = brenth(wrapped_P_HS_error_1P, T, last_limit,
                                         fb=last_err, args=(P, zs, Hm, Sm),
                                         kwargs={'info': info})
                            info = info_cache[ans]
                        except Exception as e:
#                            print('1 Phase Bounded solver could not converge after oscillation', e)
                            pass
                        break
                    else:
                        return T, phase, eos, VF
                else:
                    try:
                        err_low = max(checker.ys_neg)
                        err_high = min(checker.ys_pos)
                        T_low = checker.xs_neg[checker.ys_neg.index(err_low)]
                        T_high = checker.xs_pos[checker.ys_pos.index(err_high)]


                        ans = brenth(wrapped_P_HS_error_1P, T_low, T_high,
                                     fa=err_low, fb=err_high, args=(P, zs, Hm, Sm),
                                     kwargs={'info': info})
                        info = info_cache[ans]
                        break
                    except Exception as e:
#                        print('1 Phase Bounded solver could not converge after oscillation', e)
                        break


                checker.clear()
                continue
        if ans is None:
            raise ValueError("Could not converge 1 phase with any of initial guesses %s" %(guesses))
        return ans, info[1], info[0], None


    def flash_PH_zs(self, P, Hm, zs, T_guess=None, xs_guess=None, ys_guess=None,
                 algorithms=[DIRECT_1P, DIRECT_2P, RIGOROUS_BISECTION],
                 maxiter=100, tol=1e-4, damping=0.5):
        '''Write some documentation
        '''
        eos_1P = None
        single_phase_data = None
        for algorithm in algorithms:
            try:
                if algorithm == DIRECT_1P:
                    T, phase, eos, V_over_F = self.flash_P_HS_1P(P=P, zs=zs, Hm=Hm, T_guess=T_guess)
                    if V_over_F is not None:
                        # Must be a two phase, 1 component solution
                        return phase, [1.0], [1.0], V_over_F, T
                    try:
                        if eos.G_dep_l < eos.G_dep_g:
                            xs, ys, V_over_F = zs, None, 0.0
                            self.eos_l = eos
                            self.eos_g = None
                        else:
                            xs, ys, V_over_F = None, zs, 1.0
                            self.eos_g = eos
                            self.eos_l = None
                    except:
                        if hasattr(eos, 'G_dep_g'):
                            xs, ys, V_over_F = None, zs, 1.0
                            self.eos_g = eos
                            self.eos_l = None
                        else:
                            xs, ys, V_over_F = zs, None, 0.0
                            self.eos_l = eos
                            self.eos_g = None
                    eos_1P = eos
                    single_phase_data = phase, xs, ys, V_over_F, T
#                    print('Single phase T solution', T)
                    if self.N > 1:
                        stable, _, _ = self.stability_test_VL(T, P, zs, eos=eos)
                    else:
                        stable = True
                    if not stable:
                        raise ValueError("One phase solution is unstable")
                    return phase, xs, ys, V_over_F, T


                    # A stability test is REQUIRED! Very important!
                    # Try to refactor some of the code from the PT flash.
                elif algorithm == DIRECT_2P:
                    guess_generator = self.PH_T_guesses_2P(P, Hm, zs, T_guess=T_guess)

                    if T_guess is None:
                        T_guess = next(guess_generator)
#                    print('T_guess', T_guess)
                    phase, xs, ys, V_over_F, T = self.PH_Michelson(T_guess, P, zs, Hm, maxiter=maxiter, tol=tol,
                                                                   VF_guess_init=None, damping=damping, analytical=True)
                    if eos_1P is not None:
                        H_1P = self.enthalpy_eosmix(single_phase_data[-1], P, single_phase_data[-2], zs, single_phase_data[-4], single_phase_data[-3], eos_1P, eos_1P, single_phase_data[0])
                        H_2P = self.enthalpy_eosmix(T, P, V_over_F, zs, xs, ys, self.eos_l, self.eos_g, phase)

                        S_1P = self.entropy_eosmix(single_phase_data[-1], P, single_phase_data[-2], zs, single_phase_data[-4], single_phase_data[-3], eos_1P, eos_1P, single_phase_data[0])
                        S_2P = self.entropy_eosmix(T, P, V_over_F, zs, xs, ys, self.eos_l, self.eos_g, phase)

                        G_1P = H_1P - single_phase_data[-1]*S_1P
                        G_2P = H_2P - T*S_2P

#                        print(T, V_over_F, S_1P, S_2P, G_1P, G_2P)
                        if (S_1P > S_2P and G_1P < G_2P) or V_over_F > 1 or V_over_F < 0:
#                        try:
#                            G_1P = eos_1P.G_dep_l if eos_1P.G_dep_l < eos_1P.G_dep_g else eos_1P.G_dep_g
#                        except:
#                            G_1P = eos_1P.G_dep_g if hasattr(eos, 'G_dep_g') else eos_1P.G_dep_l
#
#                        try:
#                            G_l = self.eos_l.G_dep_l if self.eos_l.G_dep_l < self.eos_l.G_dep_g else self.eos_l.G_dep_g
#                        except:
#                            G_l = self.eos_l.G_dep_g if hasattr(eos, 'G_dep_g') else self.eos_l.G_dep_l
#
#                        try:
#                            G_g = self.eos_g.G_dep_l if self.eos_g.G_dep_l < self.eos_g.G_dep_g else self.eos_g.G_dep_g
#                        except:
#                            G_g = self.eos_g.G_dep_g if hasattr(eos, 'G_dep_g') else self.eos_g.G_dep_l
#
#                        if G_1P < (1 - V_over_F)*G_l + V_over_F*G_g:
                            return single_phase_data

                    return phase, xs, ys, V_over_F, T



                    # should return phase, xs, ys, V_over_F, T
            except Exception as e:
#                print(e)
                pass

    def flash_PS_zs(self, P, Sm, zs, T_guess=None, xs_guess=None, ys_guess=None,
                 algorithms=[DIRECT_1P, RIGOROUS_BISECTION],
                 maxiter=100, tol=1e-4, damping=0.5):
        eos_1P = None
        single_phase_data = None
        for algorithm in algorithms:
            try:
                if algorithm == DIRECT_1P:
                    T, phase, eos, V_over_F = self.flash_P_HS_1P(P=P, zs=zs, Hm=None, Sm=Sm, T_guess=T_guess)
                    if V_over_F is not None:
                        # Must be a two phase, 1 component solution
                        return phase, [1.0], [1.0], V_over_F, T
                    try:
                        if eos.G_dep_l < eos.G_dep_g:
                            xs, ys, V_over_F = zs, None, 0.0
                            self.eos_l = eos
                            self.eos_g = None
                        else:
                            xs, ys, V_over_F = None, zs, 1.0
                            self.eos_g = eos
                            self.eos_l = None
                    except:
                        if hasattr(eos, 'G_dep_g'):
                            xs, ys, V_over_F = None, zs, 1.0
                            self.eos_g = eos
                            self.eos_l = None
                        else:
                            xs, ys, V_over_F = zs, None, 0.0
                            self.eos_l = eos
                            self.eos_g = None
                    eos_1P = eos
                    single_phase_data = phase, xs, ys, V_over_F, T
                    if self.N > 1:
                        stable, _, _ = self.stability_test_VL(T, P, zs, eos=eos)
                    else:
                        stable = True
                    if not stable:
                        raise ValueError("One phase solution is unstable")
                    return phase, xs, ys, V_over_F, T

            except Exception as e:
#                print(e)
                pass


    def bubble_T_Michelsen_Mollerup(self, T_guess, P, zs, maxiter=200,
                                    xtol=1E-10, info=None, ys_guess=None,
                                    max_step_damping=5.0, near_critical=False,
                                    trivial_solution_tol=1e-4):
        # ys_guess did not help convergence at all
        N = len(zs)
        cmps = range(N)

        ys = zs if ys_guess is None else ys_guess

        def lnphis_and_derivatives(T_guess):
            eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                                 zs=zs, kijs=self.kijs, T=T_guess, P=P,
                                 only_l=True, fugacities=False,
                                 **self.eos_kwargs)

            if near_critical:
                try:
                    ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_l)
                    dlnphis_dT_l = eos_l.dlnphis_dT('l')
                except AttributeError:
                    ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_g)
                    dlnphis_dT_l = eos_l.dlnphis_dT('g')
            else:
                ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_l)
                dlnphis_dT_l = eos_l.dlnphis_dT('l')

            # TODO: d alpha 1 only?
            eos_g = eos_l.to_TP_zs_fast(T=T_guess, P=P, zs=ys, full_alphas=True, only_g=True)

            if near_critical:
                try:
                    ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_g)
                    dlnphis_dT_g = eos_g.dlnphis_dT('g')
                except AttributeError:
                    ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_l)
                    dlnphis_dT_g = eos_g.dlnphis_dT('l')
            else:
                ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_g)
                dlnphis_dT_g = eos_g.dlnphis_dT('g')

            return ln_phis_l, ln_phis_g, dlnphis_dT_l, dlnphis_dT_g, eos_l, eos_g

        T_guess_old = None
        successive_fails = 0
        for i in range(maxiter):
            try:
                ln_phis_l, ln_phis_g, dlnphis_dT_l, dlnphis_dT_g, eos_l, eos_g = lnphis_and_derivatives(T_guess)
                successive_fails = 0
            except Exception as e:
#                print(e)
                if T_guess_old is None:
                    raise ValueError("Could not calculate liquid and vapor conditions at provided initial temperature %s K" %(T_guess))
                successive_fails += 1
                if successive_fails >= 2:
                    raise ValueError("Stopped convergence procedure after multiple bad steps")
                T_guess = T_guess_old + copysign(min(max_step_damping, abs(step)), step)
#                print('fail - new T guess', T_guess)
                ln_phis_l, ln_phis_g, dlnphis_dT_l, dlnphis_dT_g, eos_l, eos_g = lnphis_and_derivatives(T_guess)



            Ks = [exp(a - b) for a, b in zip(ln_phis_l, ln_phis_g)]
            f_k = sum([zs[i]*Ks[i] for i in cmps]) - 1.0

            dfk_dT = 0.0
            for i in cmps:
                dfk_dT += zs[i]*Ks[i]*(dlnphis_dT_l[i] - dlnphis_dT_g[i])

#            print('dfk_dT', dfk_dT)
            T_guess_old = T_guess
            step = -f_k/dfk_dT


            if near_critical:
                T_guess = T_guess + copysign(min(max_step_damping, abs(step)), step)
            else:
                T_guess = T_guess + step

            ys = [zs[i]*Ks[i] for i in cmps]

            if near_critical:
                comp_difference = sum([abs(zi - yi) for zi, yi in zip(zs, ys)])
                if comp_difference < trivial_solution_tol:
                    raise ValueError("Converged to trivial condition, compositions of both phases equal")

#            print(ys, 'ys raw')
            y_sum = sum(ys)
            ys = [y/y_sum for y in ys]

            if info is not None:
                info[:] = zs, ys, Ks, eos_l, eos_g, 0.0

#            print(ys, T_guess, abs(T_guess - T_guess_old), dfk_dT, ys)
            if abs(T_guess - T_guess_old) < xtol:
                T_guess = T_guess_old
                break




        if abs(T_guess - T_guess_old) > xtol:
            raise ValueError("Did not converge to specified tolerance")
        return T_guess



#    def _err_bubble_T2(self, T, P, zs, maxiter=200, xtol=1E-10, info=None,
#                      xs_guess=None, y_guess=None):
#        T = float(T)
#        if xs_guess is not None and y_guess is not None:
#            xs, ys = xs_guess, y_guess
#        else:
#            xs, ys = zs, zs
#
#        eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
#                             zs=zs, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)
#        eos_g = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
#                         zs=ys, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)
#        phis_l = eos_l.phis_l
#        phis_g = eos_g.phis_g
#        Ks = [K_value(phi_l=l, phi_g=g) for l, g in zip(phis_l, phis_g)]

    def _err_bubble_T(self, T, P, zs, maxiter=200, xtol=1E-10, info=None,
                      xs_guess=None, y_guess=None):
        '''Needs a better error function to handle azeotropes.
        '''
        T = float(T)
        if xs_guess is not None and y_guess is not None:
            xs, ys = xs_guess, y_guess
        else:
            xs, ys = zs, zs
#            Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
#            print('wilson starting', zs, Ks)
#            V_over_F, xs, ys = flash_inner_loop(zs, Ks)
#            print('Wilson done', V_over_F, xs, ys)

        eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=zs, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)
#        print('made eos l', eos_l.phis_l)

#        phis_l = eos_l.fugacity_coefficients(eos_l.Z_l, zs)
        phis_l = eos_l.phis_l
        ys_older, xs_older = None, None

        maxiter = 15
        for i in range(maxiter):
#            print('making eosg', ys)
            eos_g = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=ys, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)
#            print('made eos g', eos_g.phis_g)
#            phis_g = eos_g.fugacity_coefficients(eos_g.Z_g, ys)
            phis_g = eos_g.phis_g
            Ks = [K_value(phi_l=l, phi_g=g) for l, g in zip(phis_l, phis_g)]
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks)
            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))

            if ys == ys_older:
                raise ValueError("Stuck in loop")

            xs_older, ys_older = xs, ys
            xs, ys = xs_new, ys_new
            if any(y < 0.0 for y in ys):
                y_tot_abs = sum([abs(y) for y in ys])
                ys = [abs(y)/y_tot_abs for y in ys]

            if err < xtol:
                break
        if info is not None:
            info[:] = xs, ys, Ks, eos_l, eos_g, V_over_F
        return V_over_F


    def bubble_T_growth(self, T_guess, P, zs, maxiter=200,
                        xtol=1E-10, info=None, ys_guess=None,
                        factor=1.4, T_max=None, T_low_factor=0.25,
                        min_factor=1.05):
        if T_max is None:
            T_max = 2*max(self.Tcs)

        while factor > min_factor:
            T = T_guess*T_low_factor
#            print(factor, P, P_max)
            Ts = []
            count = 0
            while T < T_max:
                if count != 0:
                    T = Ts[-1]*factor
                Ts.append(T)
                count += 1
                try:
                    eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                                 zs=zs, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)
                    if not hasattr(eos_l, 'lnphis_l'):
#                        print('Did not continue with P', P)
                        continue
                except Exception as e:
                    print('Could not solve eos with P=%g' %P)
                    continue

                try:
                    # The root existed - try it!
                    ans = self.bubble_T_Michelsen_Mollerup(T, P=P, zs=zs,
                                                           ys_guess=ys_guess,
                                                           maxiter=maxiter,
                                                           xtol=xtol, info=info)
                    print('success')
                    return ans
                except Exception as e:
                    pass
#                    print('failed MM with P=%g'%(P), e)


            factor = factor - abs(factor - 1)*.35
        raise ValueError("Could not converge")


    def bubble_T_guess(self, P, zs, method):
        if method == 'Wilson':
            return flash_wilson(zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, P=P, VF=0)
        elif method == 'Tb_Tc_Pc':
            return flash_Tb_Tc_Pc(zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, Tbs=self.Tbs, P=P, VF=0)
        elif method == 'IdealEOS':
            return self.flash_PVF_zs_ideal(P=P, VF=0, zs=zs)

    def bubble_T_guesses(self, P, zs, T_guess=None):
        i = -1 if T_guess is not None else 0
        while i < 3:
            try:
                if i == -1:
                    yield T_guess, None, None
                if i == 1:
                    ans = self.bubble_T_guess(P=P, zs=zs, method='IdealEOS')
                    yield ans[4], ans[1], ans[2]
                if i == 0:
                    ans = self.bubble_T_guess(P=P, zs=zs, method='Wilson')
                    yield ans[0], ans[3], ans[4]
                if i == 2:
                    ans = self.bubble_T_guess(P=P, zs=zs, method='Tb_Tc_Pc')
                    yield ans[0], ans[3], ans[4]
            except Exception as e:
#                print(e, i)
                pass
            i += 1

    def bubble_T(self, P, zs, maxiter=200, xtol=1E-10, maxiter_initial=20, xtol_initial=1e-3,
                 T_guess=None):
        info = []

        for T_guess, xs, ys in self.bubble_T_guesses(P=P, zs=zs, T_guess=T_guess):
#            print('starting guess', T_guess, xs, ys)
            try:
                T = self.bubble_T_Michelsen_Mollerup(T_guess=T_guess, P=P, zs=zs,
                                                     info=info, xtol=self.FLASH_VF_TOL,
                                                     near_critical=True,
                                                     ys_guess=ys)
                return info[0], info[1], info[5], T, info[3], info[4]
            except Exception as e:
                import sys, os
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(e, 'bubble_T_Michelsen_Mollerup falure')
                pass

            try:
                P = self.bubble_T_growth(T_guess=T_guess, P=P, zs=zs,
                                                     info=info, xtol=self.FLASH_VF_TOL,
                                                     ys_guess=ys)
                return info[0], info[1], info[5], P, info[3], info[4]
            except Exception as e:
                print(e, 'bubble_T_Michelsen_Mollerup falure of new method')
                pass


            try:
#                print('bubble T guess', T_guess)
                # Simplest solution method
                if xs is not None and ys is not None:
                    args = (P, zs, maxiter, xtol, info, xs, ys)
                else:
                    args = (P, zs, maxiter, xtol, info)
                try:
                    T = newton(self._err_bubble_T, T_guess, args=args, ytol=self.FLASH_VF_TOL)
                except Exception as e:
                    print('bubble T - newton failed with initial guess (%g):' %(T_guess)  + str(e))
                    from scipy.optimize import fsolve
                    T = float(fsolve(self._err_bubble_T, T_guess, factor=.1, xtol=self.FLASH_VF_TOL, args=args))
    #            print(T, T_guess)
                return info[0], info[1], info[5], T, info[3], info[4]
            except Exception as e:
                print('bubble T - fsolve failed with initial guess (%g):' %(T_guess)  + str(e))
                pass

        raise ValueError("Overall bubble P loop could not find a convergent method")
        Tmin, Tmax = self._bracket_bubble_T(P=P, zs=zs, maxiter=maxiter_initial, xtol=xtol_initial)
        T = ridder(self._err_bubble_T, Tmin, Tmax, args=(P, zs, maxiter, xtol, info))
        return info[0], info[1], info[5], T, info[3], info[4]



    def _bracket_bubble_T(self, P, zs, maxiter, xtol):
        negative_VFs = []
        negative_Ts = []
        positive_VFs = []
        positive_Ts = []
        guess = flash_Tb_Tc_Pc(zs=zs, Tbs=self.Tbs, Tcs=self.Tcs, Pcs=self.Pcs, P=P, VF=0.0)[0]
        eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=zs, kijs=self.kijs, T=self.T_REF_IG, P=P, **self.eos_kwargs)

        limit_list = [(20, .9, 1.1), (10, .8, 1.2), (30, .7, 1.3), (50, .6, 1.4), (2500, .2, 5)]
        for limits in limit_list:
            pts, mult_min, mult_max = limits
            guess_Ts = [guess*i for i in np.linspace(mult_min, mult_max, pts).tolist()]
            shuffle(guess_Ts)

            for T in guess_Ts:
                try:
#                    print("Trying %f" %T)
                    ans = eos_l._V_over_F_bubble_T_inner(T=T, P=P, zs=zs, maxiter=maxiter, xtol=xtol)
#                    print(ans)
                    if ans < 0:
#                        if abs(abs) < 1:
                        negative_VFs.append(ans)
                        negative_Ts.append(T)
                    else:
                        # This is very important - but it reduces speed quite a bit
                        if abs(ans) < 1:
#                            diff = lambda T : eos_l._V_over_F_bubble_T_inner(T=T, P=P, zs=zs)
                            positive_VFs.append(ans)
                            positive_Ts.append(T)
                except:
                    pass
                if negative_Ts and positive_Ts:
                    break
            if negative_Ts and positive_Ts:
                break

        T_high = positive_Ts[positive_VFs.index(min(positive_VFs))]
        T_low = negative_Ts[negative_VFs.index(max(negative_VFs))]
        return T_high, T_low


    def _bracket_dew_T(self, P, zs, maxiter, xtol, check=False):
        negative_VFs = []
        negative_Ts = []
        positive_VFs = []
        positive_Ts = []
        guess = flash_Tb_Tc_Pc(zs=zs, Tbs=self.Tbs, Tcs=self.Tcs, Pcs=self.Pcs, P=P, VF=1.0)[0]

        eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=zs, kijs=self.kijs, T=self.T_REF_IG, P=P, **self.eos_kwargs)

        limit_list = [(30, .7, 1.3), (50, .6, 1.4), (250, .1, 3)]
        for limits in limit_list:
            pts, mult_min, mult_max = limits
            guess_Ts = [guess*i for i in np.linspace(mult_min, mult_max, pts).tolist()]
            shuffle(guess_Ts)

            for T in guess_Ts:
                try:
#                    print("Trying %f" %T)
                    ans = eos_l._V_over_F_dew_T_inner(T=T, P=P, zs=zs, maxiter=maxiter, xtol=xtol)
#                    print(ans)
                    if ans < 0:
#                        if abs(abs) < 1.0:
                        # Seems to be necessary to check the second derivative
                        # This should only be necessary if the first solution is not right
                        if check:
                            diff = lambda T : eos_l._V_over_F_dew_T_inner(T=T, P=P, zs=zs)
                            second_derivative = derivative(diff, T, n=2, order=3)
                            if second_derivative > 0 and second_derivative < 1:

                                negative_VFs.append(ans)
                                negative_Ts.append(T)
                        else:
                            negative_VFs.append(ans)
                            negative_Ts.append(T)
                    else:
                        if abs(ans) < 1:
                            positive_VFs.append(ans)
                            positive_Ts.append(T)
                except:
                    pass
                if negative_Ts and positive_Ts:
                    break
            if negative_Ts and positive_Ts:
                break

        T_high = positive_Ts[positive_VFs.index(min(positive_VFs))]
        T_low = negative_Ts[negative_VFs.index(max(negative_VFs))]
        return T_high, T_low


    def _err_dew_T(self, T, P, zs, maxiter=200, xtol=1E-10, info=None):
        T = float(T)
        Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        V_over_F, xs, ys = flash_inner_loop(zs, Ks)

        eos_g = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=zs, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)
        phis_g = eos_g.phis_g

        for i in range(maxiter):
            eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=xs, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)

            Ks = [K_value(phi_l=l, phi_g=g) for l, g in zip(eos_l.phis_l, phis_g)]
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks)
            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
            xs, ys = xs_new, ys_new
            if err < xtol:
                break

        if info is not None:
            info[:] = xs, ys, Ks, eos_l, eos_g, V_over_F
        return V_over_F - 1.0

    def dew_T_Michelsen_Mollerup(self, T_guess, P, zs, maxiter=200,
                                 xtol=1E-10, info=None, xs_guess=None,
                                 max_step_damping=100.0, near_critical=False,
                                 trivial_solution_tol=1e-4):
        # Does not have any formulation available
        # According to the following, convergence does not occur with newton's method near the critical point
        # It recommends some sort of substitution method
        # Accelerated successive substitution schemes for bubble-point and dew-point calculations
        N = len(zs)
        cmps = range(N)

        xs = zs if xs_guess is None else xs_guess

        def lnphis_and_derivatives(T_guess):
            eos_g = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                                 zs=zs, kijs=self.kijs, T=T_guess, P=P,
                                 fugacities=False, only_g=True,
                                 **self.eos_kwargs)
            if near_critical:
                try:
                    ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_g)
                    dlnphis_dT_g = eos_g.dlnphis_dT('g')
                except AttributeError:
                    ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_l)
                    dlnphis_dT_g = eos_g.dlnphis_dT('l')
            else:
                ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_g)
                dlnphis_dT_g = eos_g.dlnphis_dT('g')

            eos_l = eos_g.to_TP_zs_fast(T=T_guess, P=P, zs=xs, full_alphas=True, only_l=True)

            if near_critical:
                try:
                    ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_l)
                    dlnphis_dT_l = eos_l.dlnphis_dT('l')
                except AttributeError:
                    ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_g)
                    dlnphis_dT_l = eos_l.dlnphis_dT('g')
            else:
                ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_l)
                dlnphis_dT_l = eos_l.dlnphis_dT('l')

            return ln_phis_l, ln_phis_g, dlnphis_dT_l, dlnphis_dT_g, eos_l, eos_g

        T_guess_old = None
        successive_fails = 0
        for i in range(maxiter):
            try:
                ln_phis_l, ln_phis_g, dlnphis_dT_l, dlnphis_dT_g, eos_l, eos_g = lnphis_and_derivatives(T_guess)
                successive_fails = 0
            except:
                if T_guess_old is None:
                    raise ValueError("Could not calculate liquid and vapor conditions at provided initial temperature %s K" %(T_guess))
                successive_fails += 1
                if successive_fails >= 2:
                    raise ValueError("Stopped convergence procedure after multiple bad steps")
                T_guess = T_guess_old + copysign(min(max_step_damping, abs(step)), step)
#                print('fail - new T guess', T_guess)
                ln_phis_l, ln_phis_g, dlnphis_dT_l, dlnphis_dT_g, eos_l, eos_g = lnphis_and_derivatives(T_guess)

            Ks = [exp(a - b) for a, b in zip(ln_phis_l, ln_phis_g)]
            xs = [zs[i]/Ks[i] for i in cmps]
            f_k = sum([xs[i] for i in cmps]) - 1.0

            dfk_dT = 0.0
            for i in cmps:
                dfk_dT += xs[i]*(dlnphis_dT_g[i] - dlnphis_dT_l[i])

            T_guess_old = T_guess
            step = -f_k/dfk_dT

#            print(xs, T_guess, step, dfk_dT)

            if near_critical:
                T_guess = T_guess + copysign(min(max_step_damping, abs(step)), step)
            else:
                T_guess = T_guess + step

            if near_critical:
                comp_difference = sum([abs(zi - xi) for zi, xi in zip(zs, xs)])
                if comp_difference < trivial_solution_tol:
                    raise ValueError("Converged to trivial condition, compositions of both phases equal")

            x_sum = sum(xs)
            xs = [x/x_sum for x in xs]

            if info is not None:
                info[:] = xs, zs, Ks, eos_l, eos_g, 1.0
            if abs(T_guess - T_guess_old) < xtol:
                T_guess = T_guess_old
                break


        if abs(T_guess - T_guess_old) > xtol:
            raise ValueError("Did not converge to specified tolerance")

        return T_guess


    def dew_T_growth(self, T_guess, P, zs, maxiter=200,
                     xtol=1E-10, info=None, xs_guess=None,
                     T_max=None, T_low_factor=0.9,
                     T_step=4):
        if T_max is None:
            T_pseudo = sum([zi*Tci for zi, Tci in zip(zs, self.Tcs)])
            T_max = 1.25*T_pseudo
        T_points = np.arange(T_low_factor*T_guess, T_max, step=T_step).tolist()
        T_points.sort(key=lambda x: abs(x - T_guess))


        for T in T_points:
#        while factor > min_factor:
#            T = T_guess*T_low_factor
#            print(T)
#            Ts = []
#            count = 0
#            while T < T_max:
#                if count != 0:
#                    T = Ts[-1]*factor
#                Ts.append(T)
#                count += 1
            try:
                eos_g = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=zs, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)
                if not hasattr(eos_g, 'lnphis_g'):
#                    print('Did not continue with T', T)
                    continue
            except Exception as e:
#                print('Could not solve eos with P=%g' %P)
                continue

            try:
                # The root existed - try it!
                ans = self.dew_T_Michelsen_Mollerup(T, P=P, zs=zs,
                                                    xs_guess=xs_guess,
                                                    maxiter=maxiter,
                                                    xtol=xtol, info=info)
                print('success')
                return ans
            except Exception as e:
                pass
                print('failed MM with T=%g'%(T), e)


#            factor = factor - abs(factor - 1)*.35
        raise ValueError("Could not converge")


    def dew_T_guess(self, P, zs, method):
        if method == 'Wilson':
            return flash_wilson(zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, P=P, VF=1)
        elif method == 'Tb_Tc_Pc':
            return flash_Tb_Tc_Pc(zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, Tbs=self.Tbs, P=P, VF=1)
        elif method == 'IdealEOS':
            return self.flash_PVF_zs_ideal(P=P, VF=1, zs=zs)

    def dew_T_guesses(self, P, zs, T_guess=None):
        i = -1 if T_guess is not None else 0
        while i < 3:
            try:
                if i == -1:
                    yield T_guess, None, None
                if i == 1:
                    ans = self.dew_T_guess(P=P, zs=zs, method='IdealEOS')
                    yield ans[4], ans[1], ans[2]
                if i == 0:
                    ans = self.dew_T_guess(P=P, zs=zs, method='Wilson')
                    yield ans[0], ans[3], ans[4]
                if i == 2:
                    ans = self.dew_T_guess(P=P, zs=zs, method='Tb_Tc_Pc')
                    yield ans[0], ans[3], ans[4]
            except Exception as e:
                print(e, i)
                pass
            i += 1


    def dew_T(self, P, zs, maxiter=200, xtol=1E-10, maxiter_initial=20, xtol_initial=1e-3,
              T_guess=None):
        info = []
        for T_guess, xs, ys in self.dew_T_guesses(P=P, zs=zs, T_guess=T_guess):
            try:
                T = self.dew_T_Michelsen_Mollerup(T_guess=T_guess, P=P, zs=zs, info=info,
                                                  xtol=self.FLASH_VF_TOL, xs_guess=xs,
                                                  near_critical=True)
                return info[0], info[1], info[5], T, info[3], info[4]
            except Exception as e:
                print(e, 'dew_T_Michelsen_Mollerup falure')
                pass

            try:
                P = self.dew_T_growth(T_guess=T_guess, P=P, zs=zs,
                                      info=info, xtol=self.FLASH_VF_TOL,
                                      xs_guess=xs)
                return info[0], info[1], info[5], P, info[3], info[4]
            except Exception as e:
                print(e, 'dew_T_Michelsen_Mollerup falure of new method')
                pass

            try:
                try:
                    T = newton(self._err_dew_T, T_guess, ytol=self.FLASH_VF_TOL, args=(P, zs, maxiter, xtol, info))
                except Exception as e:
                    print('dew_T newton failed with %g K guess' %(T_guess), e)
                    from scipy.optimize import fsolve
                    try:
                        T = float(fsolve(self._err_dew_T, T_guess, factor=.1, xtol=self.FLASH_VF_TOL, args=(P, zs, maxiter, xtol, info)))
                    except Exception as e:
                        print('dew_T fsolve failed with %g K guess' %(T_guess), e)
                        continue

    #            print(T, T_guess)
                return info[0], info[1], info[5], T, info[3], info[4]
            except Exception as e:
                pass

        Tmin, Tmax = self._bracket_dew_T(P=P, zs=zs, maxiter=maxiter_initial, xtol=xtol_initial)
        T_calc = ridder(self._err_dew_T, Tmin, Tmax, args=(P, zs, maxiter, xtol, info))
        val = self._err_dew_T(T_calc, P, zs, maxiter, xtol)
        if abs(val) < 1:
            T = T_calc
        else:
            Tmin, Tmax = self._bracket_dew_T(P=P, zs=zs, maxiter=maxiter_initial, xtol=xtol_initial, check=True)
            T=  ridder(self._err_dew_T, Tmin, Tmax, args=(P, zs, maxiter, xtol, info))
        return info[0], info[1], info[5], T


    def _bracket_dew_P(self, T, zs, maxiter, xtol, check=False):
        negative_VFs = []
        negative_Ps = []
        positive_VFs = []
        positive_Ps = []
        guess = flash_Tb_Tc_Pc(zs=zs, Tbs=self.Tbs, Tcs=self.Tcs, Pcs=self.Pcs, T=T, VF=1.0)[1]

        eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=zs, kijs=self.kijs, T=T, P=self.P_REF_IG, **self.eos_kwargs)

        limit_list = [(30, .7, 1.3), (50, .6, 1.4), (250, .001, 3)]
        for limits in limit_list:
            pts, mult_min, mult_max = limits
            guess_Ps = [guess*10**(i) for i in np.linspace(mult_min, mult_max, pts).tolist()]
            shuffle(guess_Ps)

            for P in guess_Ps:
                try:
#                    print("Trying %f" %P)
                    ans = eos_l._V_over_F_dew_T_inner(T=T, P=P, zs=zs, maxiter=maxiter, xtol=xtol)
#                    print(ans)
                    if ans < 0:
                        if abs(ans) < 1.0:
                        # Seems to be necessary to check the second derivative
                        # This should only be necessary if the first solution is not right
#                        if check:
#                            diff = lambda T : eos_l._V_over_F_dew_T_inner(T=T, P=P, zs=zs)
#                            second_derivative = derivative(diff, T, n=2, order=3)
#                            if second_derivative > 0 and second_derivative < 1:
#
                            negative_VFs.append(ans)
                            negative_Ps.append(P)
#                        else:
#                        negative_VFs.append(ans)
#                        negative_Ps.append(P)
                    else:
#                        if abs(ans) < 1:
                        positive_VFs.append(ans)
                        positive_Ps.append(P)
                except:
                    pass
                if negative_Ps and positive_Ps:
                    break
            if negative_Ps and positive_Ps:
                break

        P_high = positive_Ps[positive_VFs.index(min(positive_VFs))]
        P_low = negative_Ps[negative_VFs.index(max(negative_VFs))]
        return P_high, P_low


    def _err_dew_P(self, P, T, zs, maxiter=200, xtol=1E-10, info=None):
        P = float(P)
        Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        V_over_F, xs, ys = flash_inner_loop(zs, Ks)

        eos_g = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=zs, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)
        phis_g = eos_g.phis_g

        for i in range(maxiter):
            eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=xs, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)

            Ks = [K_value(phi_l=l, phi_g=g) for l, g in zip(eos_l.phis_l, phis_g)]
            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks)
            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
            xs, ys = xs_new, ys_new
            if err < xtol:
                break

        if info is not None:
            info[:] = xs, ys, Ks, eos_l, eos_g, V_over_F
        return V_over_F - 1.0

    def dew_P_Michelsen_Mollerup(self, P_guess, T, zs, maxiter=200,
                                 xtol=1E-3, info=None, xs_guess=None,
                                 near_critical=False):
        N = len(zs)
        cmps = range(N)
        xs = zs if xs_guess is None else xs_guess

#        if xtol < 1e-4:
#            xtol = 1e-3
        eos_g_base = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=zs, kijs=self.kijs, T=T, P=P_guess, fugacities=False, only_g=True,
                             **self.eos_kwargs)

        def lnphis_and_derivatives(P_guess):
            if eos_g_base.P == P_guess:
                eos_g = eos_g_base
            else:
                eos_g = eos_g_base.to_TP_zs_fast(T=T, P=P_guess, zs=zs, full_alphas=False, only_g=True) #

            if near_critical:
                try:
                    ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_g)
                    d_lnphis_dP_g = eos_g.dlnphis_dP('g')
                except AttributeError:
                    ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_l)
                    d_lnphis_dP_g = eos_g.dlnphis_dP('l')
            else:
                ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_g)
                d_lnphis_dP_g = eos_g.dlnphis_dP('g')


            eos_l = eos_g_base.to_TP_zs_fast(T=T, P=P_guess, zs=xs, full_alphas=False, only_l=True)

            if near_critical:
                try:
                    ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_l)
                    d_lnphis_dP_l = eos_l.dlnphis_dP('l')
                except AttributeError:
                    ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_g)
                    d_lnphis_dP_l = eos_l.dlnphis_dP('g')
            else:
                ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_l)
                d_lnphis_dP_l = eos_l.dlnphis_dP('l')

            return ln_phis_l, ln_phis_g, d_lnphis_dP_l, d_lnphis_dP_g, eos_l, eos_g

        for i in range(maxiter):
            ln_phis_l, ln_phis_g, d_lnphis_dP_l, d_lnphis_dP_g, eos_l, eos_g = lnphis_and_derivatives(P_guess)

            Ks = [exp(a - b) for a, b in zip(ln_phis_l, ln_phis_g)]

            f_k = sum([zs[i]/Ks[i] for i in cmps]) - 1.0

            dfk_dP = 0.0
            for i in cmps:
                dfk_dP += zs[i]/Ks[i]*(d_lnphis_dP_g[i] - d_lnphis_dP_l[i])

            P_guess_old = P_guess

            step = - f_k/dfk_dP
            P_guess = P_guess + step
            xs = [zs[i]/Ks[i] for i in cmps]

            x_sum = sum(xs)
            xs = [x/x_sum for x in xs]

            if info is not None:
                info[:] = xs, zs, Ks, eos_l, eos_g, 1.0

#            print(xs, P_guess, step, P_guess - P_guess_old)
            if abs(P_guess - P_guess_old) < xtol:
                P_guess = P_guess_old # avoid new step which does not have eos's evaluated at it; just use a lower tolerance
                break

        if abs(P_guess - P_guess_old) > xtol:
            raise ValueError("Did not converge to specified tolerance")
        return P_guess

    def dew_P_growth(self, P_guess, T, zs, maxiter=200,
                     xtol=1E-4, info=None, xs_guess=None,
                     factor=1.4, P_max=None, P_low_factor=0.25,
                     min_factor=1.05):
        if P_max is None:
            P_max = 2*max(self.Pcs)

        while factor > min_factor:
            P = P_guess*P_low_factor
#            print(factor, P, P_max)
            Ps = []
            count = 0
            while P < P_max:
                if count != 0:
                    P = Ps[-1]*factor
                Ps.append(P)
                count += 1
                try:
                    eos_g = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                                 zs=zs, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)
                    if not hasattr(eos_g, 'lnphis_g'):
                        print('Did not continue with P', P)
                        continue
                except Exception as e:
                    print('Could not solve eos with P=%g' %P)
                    continue

                try:
                    # The root existed - try it!
                    ans = self.dew_P_Michelsen_Mollerup(P, T=T, zs=zs,
                                                        xs_guess=xs_guess,
                                                        maxiter=maxiter,
                                                        xtol=xtol, info=info,
                                                        near_critical=True)
                    print('success')
                    return ans
                except Exception as e:
                    pass
                    print('failed MM with P=%g'%(P), e)


            factor = factor - abs(factor - 1)*.35
        raise ValueError("Could not converge")


    def dew_P_guess(self, T, zs, method):
        if method == 'Wilson':
            return flash_wilson(zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, T=T, VF=1)
        elif method == 'Tb_Tc_Pc':
            return flash_Tb_Tc_Pc(zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, Tbs=self.Tbs, T=T, VF=1)
        elif method == 'IdealEOS':
            return self.flash_TVF_zs_ideal(T=T, VF=1, zs=zs)

    def dew_P_guesses(self, T, zs, P_guess=None):
        i = -1 if P_guess is not None else 0
        while i < 3:
            try:
                if i == -1:
                    yield P_guess, None, None
                if i == 0:
                    ans = self.dew_P_guess(T=T, zs=zs, method='IdealEOS')
                    yield ans[4], ans[1], ans[2]
                if i == 1:
                    ans = self.dew_P_guess(T=T, zs=zs, method='Wilson')
                    yield ans[0], ans[3], ans[4]
                if i == 2:
                    ans = self.dew_P_guess(T=T, zs=zs, method='Tb_Tc_Pc')
                    yield ans[0], ans[3], ans[4]
            except Exception as e:
                pass
            i += 1


    def dew_P(self, T, zs, maxiter=200, xtol=1E-10, maxiter_initial=20, xtol_initial=1e-3,
              P_guess=None):
        info = []
        for P_guess, xs, ys in self.dew_P_guesses(T=T, zs=zs, P_guess=P_guess):
            P = None
            try:
                P = self.dew_P_Michelsen_Mollerup(P_guess=P_guess, T=T, zs=zs,
                                                  info=info, xtol=self.FLASH_VF_TOL,
                                                  xs_guess=xs, near_critical=True,
                                                  )
                return info[0], info[1], info[5], P, info[3], info[4]
            except Exception as e:
                print(e, 'dew_P_Michelsen_Mollerup falure')
                pass

#            try:
#                P = self.dew_P_growth(P_guess=P_guess, T=T, zs=zs,
#                                      info=info, xtol=self.FLASH_VF_TOL,
#                                      xs_guess=xs)
#                return info[0], info[1], info[5], P, info[3], info[4]
#            except Exception as e:
#                print(e, 'dew_P_Michelsen_Mollerup falure of new method')
#                pass


            # Simplest solution method
            try:
                P = newton(self._err_dew_P, P_guess, ytol=self.FLASH_VF_TOL,
                           args=(T, zs, maxiter, xtol, info))
            except Exception as e:
                print('newton failed dew_P guess %g' %(P_guess), e)
                pass
            if P is None:
                try:
                    from scipy.optimize import fsolve
                    P = float(fsolve(self._err_dew_P, P_guess, xtol=self.FLASH_VF_TOL,
                                     factor=.1, args=(T, zs, maxiter, xtol, info)))
                except Exception as e:
                    print('fsolve failed dew_P %g' %(P_guess), e)
                    pass
#            print(P, P_guess_as_pure)
            if P is not None:
                return info[0], info[1], info[5], P, info[3], info[4]

        raise ValueError("Overall bubble P loop could not find a convergent method")
        Pmin, Pmax = self._bracket_dew_P(T=T, zs=zs, maxiter=maxiter_initial, xtol=xtol_initial)
        P = ridder(self._err_dew_P, Pmin, Pmax, args=(T, zs, maxiter, xtol, info))
        return info[0], info[1], info[5], P, info[3], info[4]


    def _bracket_bubble_P(self, T, zs, maxiter, xtol, check=False):
        negative_VFs = []
        negative_Ps = []
        positive_VFs = []
        positive_Ps = []
        guess = flash_Tb_Tc_Pc(zs=zs, Tbs=self.Tbs, Tcs=self.Tcs, Pcs=self.Pcs, T=T, VF=0.0)[1]

        eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=zs, kijs=self.kijs, T=T, P=self.P_REF_IG, **self.eos_kwargs)

        limit_list = [(30, .7, 1.3), (50, .6, 1.4), (250, .001, 3)]
        for limits in limit_list:
            pts, mult_min, mult_max = limits
            guess_Ps = [guess*10**(i) for i in np.linspace(mult_min, mult_max, pts).tolist()]
            shuffle(guess_Ps)

            for P in guess_Ps:
                try:
                    ans = eos_l._V_over_F_bubble_T_inner(T=T, P=P, zs=zs, maxiter=maxiter, xtol=xtol)
                    if ans < 0:
                        if abs(ans) < 1.0:
                            negative_VFs.append(ans)
                            negative_Ps.append(P)
                    else:
                        positive_VFs.append(ans)
                        positive_Ps.append(P)
                except:
                    pass
                if negative_Ps and positive_Ps:
                    break
            if negative_Ps and positive_Ps:
                break

        P_high = positive_Ps[positive_VFs.index(min(positive_VFs))]
        P_low = negative_Ps[negative_VFs.index(max(negative_VFs))]
        return P_high, P_low

    def _err_bubble_P(self, P, T, zs, maxiter=200, xtol=1E-10, info=None):
        P = float(P)
#        print('P', P)
        Ks = [Wilson_K_value(T, P, Tci, Pci, omega) for Pci, Tci, omega in zip(self.Pcs, self.Tcs, self.omegas)]
        V_over_F, xs, ys = flash_inner_loop(zs, Ks)

        eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=zs, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)

        for i in range(maxiter):
            eos_g = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=ys, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)

#            try:
            phis_g = eos_g.phis_g
#            except AttributeError:
##                print('using liquid phis to avoid failure')
#                phis_g = eos_g.phis_l
            Ks = [K_value(phi_l=l, phi_g=g) for l, g in zip(eos_l.phis_l, phis_g)]

            V_over_F, xs_new, ys_new = flash_inner_loop(zs, Ks)
            err = (sum([abs(x_new - x_old) for x_new, x_old in zip(xs_new, xs)]) +
                  sum([abs(y_new - y_old) for y_new, y_old in zip(ys_new, ys)]))
#            if not (any(i < 0.0 for i in xs_new) or any(i < 0.0 for i in ys_new)):
                # No point to this check - halts convergence
            xs, ys = xs_new, ys_new
#            print('err', err, 'xs, ys', xs, ys)
            if err < xtol:
                break

        if info is not None:
            info[:] = xs, ys, Ks, eos_l, eos_g, V_over_F
        return V_over_F


    def bubble_P_Michelsen_Mollerup(self, P_guess, T, zs, maxiter=200,
                                    xtol=1E-1, info=None, ys_guess=None,
                                    near_critical=False, max_step_damping=1e9,
                                    trivial_solution_tol=1e-4):
        N = len(zs)
        cmps = range(N)
        ys = zs if ys_guess is None else ys_guess


        eos_l_base = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                             zs=zs, kijs=self.kijs, T=T, P=P_guess,
                             fugacities=False, only_l=True,
                             **self.eos_kwargs)

        def lnphis_and_derivatives(P_guess):
            if eos_l_base.P == P_guess:
                eos_l = eos_l_base
            else:
                eos_l = eos_l_base.to_TP_zs_fast(T=T, P=P_guess, zs=zs, full_alphas=False, only_l=True)

            if near_critical:
                try:
                    ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_l)
                    d_lnphis_dP_l = eos_l.dlnphis_dP('l')
                except AttributeError:
                    ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_g)
                    d_lnphis_dP_l = eos_l.dlnphis_dP('g')
            else:
                ln_phis_l = eos_l.fugacity_coefficients(eos_l.Z_l)
                d_lnphis_dP_l = eos_l.dlnphis_dP('l')

            eos_g = eos_l_base.to_TP_zs_fast(T=T, P=P_guess, zs=ys, full_alphas=False, only_g=True)

            if near_critical:
                try:
                    ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_g)
                    d_lnphis_dP_g = eos_g.dlnphis_dP('g')
                except AttributeError:
                    ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_l)
                    d_lnphis_dP_g = eos_g.dlnphis_dP('l')
            else:
                ln_phis_g = eos_g.fugacity_coefficients(eos_g.Z_g)
                d_lnphis_dP_g = eos_g.dlnphis_dP('g')


            return ln_phis_l, ln_phis_g, d_lnphis_dP_l, d_lnphis_dP_g, eos_l, eos_g


        for i in range(maxiter):

            ln_phis_l, ln_phis_g, d_lnphis_dP_l, d_lnphis_dP_g, eos_l, eos_g = lnphis_and_derivatives(P_guess)
            Ks = [exp(a - b) for a, b in zip(ln_phis_l, ln_phis_g)]
            f_k = sum([zs[i]*Ks[i] for i in cmps]) - 1.0

            dfk_dP = 0.0
            for i in cmps:
                dfk_dP += zs[i]*Ks[i]*(d_lnphis_dP_l[i] - d_lnphis_dP_g[i])

            P_guess_old = P_guess

            step = - f_k/dfk_dP


            if near_critical:
                P_guess = P_guess + copysign(min(max_step_damping, abs(step)), step)
            else:
                P_guess = P_guess + step

            ys = [zs[i]*Ks[i] for i in cmps]
            y_sum = sum(ys)
            ys = [y/y_sum for y in ys]

#            if near_critical:
#                comp_difference = sum([abs(zi - yi) for zi, yi in zip(zs, ys)])
#                if comp_difference < trivial_solution_tol:
#                    raise ValueError("Converged to trivial condition, compositions of both phases equal")


            if info is not None:
                info[:] = zs, ys, Ks, eos_l, eos_g, 0.0
#            print(ys, P_guess, abs(P_guess - P_guess_old), dfk_dP)
            if abs(P_guess - P_guess_old) < xtol:
                P_guess = P_guess_old
                break

        if abs(P_guess - P_guess_old) > xtol:
            raise UnconvergedError("Did not converge to specified tolerance")
        return P_guess


    def bubble_P_growth(self, P_guess, T, zs, maxiter=200,
                        xtol=1E-4, info=None, ys_guess=None,
                        factor=1.4, P_max=None, P_low_factor=0.25,
                        min_factor=1.025, near_critical=True):
        if P_max is None:
            P_max = 2*max(self.Pcs)

        while factor > min_factor:
            P = P_guess*P_low_factor
#            print(factor, P, P_max)
            Ps = []
            count = 0
            while P < P_max:
                if count != 0:
                    P = Ps[-1]*factor
                Ps.append(P)
                count += 1
                try:
                    eos_l = self.eos_mix(Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas,
                                 zs=zs, kijs=self.kijs, T=T, P=P, **self.eos_kwargs)
                    if not hasattr(eos_l, 'lnphis_l'):
#                        print('Did not continue with P', P)
                        continue
                except Exception as e:
                    print('Could not solve eos with P=%g' %P)
                    continue

                try:
                    # The root existed - try it!
                    ans = self.bubble_P_Michelsen_Mollerup(P, T=T, zs=zs,
                                                           ys_guess=ys_guess,
                                                           maxiter=maxiter,
                                                           xtol=xtol, info=info,
                                                           near_critical=near_critical)
                    print('success')
                    return ans
                except Exception as e:
                    pass
#                    print('failed MM with P=%g'%(P), e)


            factor = factor - abs(factor - 1)*.35
        raise ValueError("Could not converge")


    def bubble_P_guess(self, T, zs, method):
        if method == 'Wilson':
            return flash_wilson(zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, omegas=self.omegas, T=T, VF=0)
        elif method == 'Tb_Tc_Pc':
            return flash_Tb_Tc_Pc(zs=zs, Tcs=self.Tcs, Pcs=self.Pcs, Tbs=self.Tbs, T=T, VF=0)
        elif method == 'IdealEOS':
            return self.flash_TVF_zs_ideal(T=T, VF=0, zs=zs)

    def bubble_P_guesses(self, T, zs, P_guess=None):
        i = -1 if P_guess is not None else 0
        while i < 3:
            try:
                if i == -1:
                    yield P_guess, None, None
                if i == 0:
                    ans = self.bubble_P_guess(T=T, zs=zs, method='IdealEOS')
                    yield ans[4], ans[1], ans[2]
                if i == 1:
                    ans = self.bubble_P_guess(T=T, zs=zs, method='Wilson')
                    yield ans[1], ans[3], ans[4]
                if i == 2:
                    ans = self.bubble_P_guess(T=T, zs=zs, method='Tb_Tc_Pc')
                    yield ans[1], ans[3], ans[4]
            except Exception as e:
                pass
            i += 1


    def bubble_P(self, T, zs, maxiter=200, xtol=1E-4, maxiter_initial=20,
                 xtol_initial=1e-3, P_guess=None, max_step_damping=1e6):



        info = []
        maxP = max(self.Pcs)*2
        for P_guess, xs, ys in self.bubble_P_guesses(T=T, zs=zs, P_guess=P_guess):
            P = None

            try:
#                print(P_guess, xs, ys, 'P_guess, xs, ys')
                P = self.bubble_P_Michelsen_Mollerup(P_guess=P_guess, T=T, zs=zs,
                                                     info=info, xtol=self.FLASH_VF_TOL,
                                                     ys_guess=ys, near_critical=True,
                                                     max_step_damping=max_step_damping)
                return info[0], info[1], info[5], P, info[3], info[4]
            except:
                import sys, os
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


                print(exc_obj, 'bubble_P_Michelsen_Mollerup falure')
                pass

            try:
                P = self.bubble_P_growth(P_guess=P_guess, T=T, zs=zs,
                                                     info=info, xtol=self.FLASH_VF_TOL,
                                                     ys_guess=ys)
                return info[0], info[1], info[5], P, info[3], info[4]
            except Exception as e:
                print(e, 'bubble_P_Michelsen_Mollerup falure of new method')
                pass


            try:
                P = float(newton(self._err_bubble_P, P_guess, ytol=self.FLASH_VF_TOL,
                                 args=(T, zs, maxiter, xtol, info)))
                if P > maxP:
                    1/0
            except Exception as e:
                print('bubble_P newton failure with guess %s' %(P_guess), e)
                try:
                    from scipy.optimize import fsolve
                    P = float(fsolve(self._err_bubble_P, P_guess, xtol=self.FLASH_VF_TOL,
                                     factor=.1, args=(T, zs, maxiter, xtol, info)))
                    if P > maxP:
                        1/0
                except Exception as e:
                    print('bubble_P fsolve failure with guess %s' %(P_guess), e)
                    continue
            return info[0], info[1], info[5], P, info[3], info[4]

        raise ValueError("Overall bubble P loop could not find a convergent method")
        Pmin, Pmax = self._bracket_bubble_P(T=T, zs=zs, maxiter=maxiter_initial, xtol=xtol_initial)
        P = ridder(self._err_bubble_P, Pmin, Pmax, args=(T, zs, maxiter, xtol, info))
        return info[0], info[1], info[5], P, info[3], info[4]


    def dew_T_envelope(self, zs, P_low=1e5, P_high=None, xtol=1E-10,
                       factor=1.02, max_step_damping=.05, min_step_termination=1,
                       min_factor_termination=1.0000001, max_P_step=1e5,
                       spec_points=None):
        factor_original = factor
        factor_power = 1.0
        info = []
        xs_known = []
        Ts_known = []
        P_points = []

        near_critical = False
        T_low = None
        for T_guess, xs, ys in self.dew_T_guesses(P=P_low, zs=zs):
            try:
                T_low = self.dew_T_Michelsen_Mollerup(T_guess=T_guess, P=P_low, zs=zs, info=info,
                                                  xtol=self.FLASH_VF_TOL, xs_guess=xs,
                                                  max_step_damping=max_step_damping)
                xs_low, _, _, _, _, _ = info

            except Exception as e:
#                print('dew_T_Michelsen_Mollerup falure on initialization', e)
                pass
        if T_low is None:
            raise ValueError("Could not solve initial point to begin dew T envelope")

        xs_known.append(xs_low)
        Ts_known.append(T_low)
        P_points.append(P_low)
        T_prev, xs_prev = T_low, xs_low

        if P_high is None:
            P_high = 1.5*max(self.Pcs)

        spec_point_working = 1e100 if spec_points is None else spec_points[0]
        if spec_point_working < P_low:
            raise ValueError("Cannot specify a point lower than the minimum pressure")
        if spec_point_working == P_low:
            spec_points = spec_points[1:]
            spec_point_working = spec_points[0]

        P_prev = P_working = min(P_low*factor, P_low + max_P_step, spec_point_working)
        if P_prev == spec_point_working:
            spec_point_working = spec_points[1]


        while P_working < P_high + P_working*(factor - 1):
            info = []
            try:
#                print('trying point', P_working, 'factor', factor)
                T = self.dew_T_Michelsen_Mollerup(T_guess=T_prev, P=P_working, zs=zs, info=info,
                                                  xtol=self.FLASH_VF_TOL, xs_guess=xs_prev,
                                                  max_step_damping=max_step_damping,
                                                  near_critical=near_critical)
                xs, _, _, _, _, _ = info
                xs_known.append(xs)
                Ts_known.append(T)
                P_points.append(P_working)
                xs_prev, T_prev, P_prev = xs, T, P_working
#                print('success on P', P_working)
            except Exception as e:
                factor = 1 + (factor - 1)*0.5
                if spec_points is not None and P_working in spec_points:
#                    print('failed spec point', P_working, spec_point_working, spec_points)
                    spec_point_working = P_working
#                print('failed dew T at P %g with xs %s, factor now %f' %(P_working, xs_known[-1], factor), e)
            P_working_prev = P_working
            P_working = min(P_prev*factor**factor_power, P_prev + max_P_step, spec_point_working)
            # Force a different pressure if the factor has changed but not enough - still spec_point_working

            if spec_points is not None:
                while P_working == P_working_prev:
                    factor = 1 + (factor - 1)*0.5
                    P_working = min(P_prev*factor**factor_power, P_prev + max_P_step, spec_point_working)

            if P_working == spec_point_working:
                try:
                    spec_point_working = spec_points[spec_points.index(spec_point_working)+1]
                except:
                    spec_point_working = 1e100

            if factor < min_factor_termination and P_points[-1]*(factor-1) < min_step_termination:
                if not near_critical:
                    factor = factor_original
                    near_critical = True
                else:
                    break
#                if factor_power < 1:
#                    break
#                else:
#                    factor_power = -1
#                    factor = 2.0*factor - 1.0
#                    factor = 2.0*factor - 1.0
#                    factor = 2.0*factor - 1.0
#                    factor = min(factor, factor_original)

                # After this fails, it would be ideal to try to keep running the envelope while decreasing pressure

                # Is is possible to come up with a step on zs or Ts to help?
#                P_prev = P_working = P_working + dP_skip_step
#                N = len(zs)
#                cmps = range(N)
#
#                dP_skipped = P_working - P_points[-1]
#
#                dP_prev = (P_points[-1] - P_points[-2])
#
#                dT_dP = (Ts_known[-1] - Ts_known[-2])/dP_prev
#                dx_dP = [(xs_known[-1][i] - xs_known[-2][i])/dP_prev for i in cmps]
#
#                xs_prev = [max(xs_known[-1][i] + dx_dP[i]*dP_skipped, 0) for i in cmps]
#                xs_prev_sum = sum(xs_prev)
#                xs_prev = [xi/xs_prev_sum for xi in xs_prev]
#
#                T_prev = T_prev + dT_dP*dP_skipped
#
#                print('dP_skipped', dP_skipped, 'dT_dP', dT_dP, 'dx_dP', dx_dP, 'xs_prev', xs_prev, 'T_prev', T_prev)


        return P_points, Ts_known, xs_known

    def bubble_T_envelope(self, zs, P_low=1e5, P_high=None, xtol=1E-10,
                          factor=1.02, max_step_damping=.05, min_step_termination=1,
                          min_factor_termination=1.0000001,
                          max_P_step=1e5, spec_points=None):
        factor_original = factor
        factor_power = 1.0
        info = []
        ys_known = []
        Ts_known = []
        P_points = []
        near_critical = False
        T_low = None
        for T_guess, xs, ys in self.bubble_T_guesses(P=P_low, zs=zs):
            try:
                T_low = self.bubble_T_Michelsen_Mollerup(T_guess=T_guess, P=P_low, zs=zs, info=info,
                                                  xtol=self.FLASH_VF_TOL, ys_guess=ys,
                                                  max_step_damping=max_step_damping)
                _, ys_low, _, _, _, _ = info

            except Exception as e:
                print('bubble_T_Michelsen_Mollerup falure on initialization', e)
                pass
        if T_low is None:
            raise ValueError("Could not solve initial point to begin bubble T envelope")

        ys_known.append(ys_low)
        Ts_known.append(T_low)
        P_points.append(P_low)
        T_prev, ys_prev = T_low, ys_low

        if P_high is None:
            P_high = 1.5*max(self.Pcs)

        spec_point_working = 1e100 if spec_points is None else spec_points[0]
        if spec_point_working < P_low:
            raise ValueError("Cannot specify a point lower than the minimum pressure")
        if spec_point_working == P_low:
            spec_points = spec_points[1:]
            spec_point_working = spec_points[0]

        P_prev = P_working = min(P_low*factor, P_low + max_P_step, spec_point_working)
        if P_prev == spec_point_working:
            spec_point_working = spec_points[1]



        while P_working < P_high + P_working*(factor - 1):
            info = []
            try:
#                print('trying P', P_working)
                if factor_power == -1:
                    T_prev_working = T_prev + 0.02
                else:
                    T_prev_working = T_prev
                T = self.bubble_T_Michelsen_Mollerup(T_guess=T_prev_working, P=P_working, zs=zs, info=info,
                                                  xtol=self.FLASH_VF_TOL, ys_guess=ys_prev,
                                                  max_step_damping=max_step_damping,
                                                  near_critical=near_critical)
                _, ys, _, _, _, _ = info
                ys_known.append(ys)
                Ts_known.append(T)
                P_points.append(P_working)
                ys_prev, T_prev, P_prev = ys, T, P_working
#                print('success on P', P_working)

                if factor_power == -1:
                    factor = min(2.0*factor - 1.0, factor_original)
                    # try a larger delta next time
            except Exception as e:
                factor = 1 + (factor - 1)*0.5
                if spec_points is not None and P_working in spec_points:
                    spec_point_working = P_working

#                print('failed bubble T at P %g with ys %s, factor now %f' %(P_working, ys_known[-1], factor), e)

#                import sys, os
#                exc_type, exc_obj, exc_tb = sys.exc_info()
#                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#                print(exc_type, fname, exc_tb.tb_lineno)
#
            P_working_prev = P_working
            P_working = min(P_prev*factor**factor_power, P_prev + max_P_step, spec_point_working)
            # Force a different pressure if the factor has changed but not enough - still spec_point_working
            while P_working == P_working_prev:
                factor = 1 + (factor - 1)*0.5
                P_working = min(P_prev*factor**factor_power, P_prev + max_P_step, spec_point_working)

            if P_working == spec_point_working:
                try:
                    spec_point_working = spec_points[spec_points.index(spec_point_working)+1]
                except:
                    spec_point_working = 1e100



            if factor < min_factor_termination and P_points[-1]*(factor-1) < min_step_termination or P_working < P_low:
                if not near_critical:
                    factor = factor_original
                    near_critical = True
                else:
                    break
#                elif factor_power < 1:
#                    break
#                else:
#                    factor_power = -1
##                    factor = 2.0*factor - 1.0
#                    factor = factor_original
##                    factor = min(factor, factor_original)

        return P_points, Ts_known, ys_known



    def bubble_P_envelope(self, zs, T_low=200, T_high=None, xtol=10,
                          factor=1.2, max_step_damping=1e7, min_step_termination=1e-3,
                          min_factor_termination=1.0000001,
                          max_T_step=20, spec_points=None):
        factor_original = factor
        factor_power = 1.0
        info = []
        ys_known = []
        Ps_known = []
        T_points = []
        near_critical = False
        P_low = None
        for P_guess, xs, ys in self.bubble_P_guesses(T=T_low, zs=zs):
            try:
                P_low = self.bubble_P_Michelsen_Mollerup(P_guess=P_guess, T=T_low, zs=zs, info=info,
                                                  xtol=xtol, ys_guess=ys,
                                                  max_step_damping=max_step_damping)
                _, ys_low, _, _, _, _ = info

            except Exception as e:
                print('bubble_P_Michelsen_Mollerup falure on initialization', e)
                pass
        if P_low is None:
            raise ValueError("Could not solve initial point to begin bubble T envelope")

        ys_known.append(ys_low)
        Ps_known.append(P_low)
        T_points.append(T_low)
        P_prev, ys_prev = P_low, ys_low

        if T_high is None:
            T_high = 1.5*max(self.Tcs)

        spec_point_working = 1e100 if spec_points is None else spec_points[0]
        if spec_point_working < T_low:
            raise ValueError("Cannot specify a point lower than the minimum temperature")
        if spec_point_working == T_low:
            spec_points = spec_points[1:]
            spec_point_working = spec_points[0]

        T_prev = T_working = min(T_low*factor, T_low + max_T_step, spec_point_working)
        if T_prev == spec_point_working:
            spec_point_working = spec_points[1]


        successes = 0
        while T_working < T_high + T_working*(factor - 1):
            info = []
            try:
#                print('trying T', T_working)
                P_prev_working = P_prev
                P = self.bubble_P_Michelsen_Mollerup(P_guess=P_prev_working, T=T_working, zs=zs, info=info,
                                                  xtol=xtol, ys_guess=ys_prev,
                                                  max_step_damping=max_step_damping,
                                                  near_critical=near_critical)
                _, ys, _, _, _, _ = info
                ys_known.append(ys)
                Ps_known.append(P)
                T_points.append(T_working)
                ys_prev, P_prev, T_prev = ys, P, T_working
#                print('success on P', T_working)
                successes += 1
                if successes > 3:
                    factor = 2.0*factor - 1.0

                if factor_power == -1:
                    factor = min(2.0*factor - 1.0, factor_original)
                    # try a larger delta next time
            except Exception as e:
#                print(e)
                successes = 0
                factor = 1 + (factor - 1)*0.5
                if spec_points is not None and T_working in spec_points:
                    spec_point_working = T_working

#                print('failed bubble P at T %g with ys %s, factor now %f' %(T_working, ys_known[-1], factor), e)

#                import sys, os
#                exc_type, exc_obj, exc_tb = sys.exc_info()
#                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#                print(exc_type, fname, exc_tb.tb_lineno)
#
            T_working_prev = T_working
            T_working = min(T_prev*factor**factor_power, T_prev + max_T_step, spec_point_working)
            # Force a different pressure if the factor has changed but not enough - still spec_point_working
            while T_working == T_working_prev:
                factor = 1 + (factor - 1)*0.5
                T_working = min(T_prev*factor**factor_power, T_prev + max_T_step, spec_point_working)

            if T_working == spec_point_working:
                try:
                    spec_point_working = spec_points[spec_points.index(spec_point_working)+1]
                except:
                    spec_point_working = 1e100



            if factor < min_factor_termination and T_points[-1]*(factor-1) < min_step_termination or T_working < T_low:
                if not near_critical:
                    factor = factor_original
                    near_critical = True
                else:
                    break
#                elif factor_power < 1:
#                    break
#                else:
#                    factor_power = -1
##                    factor = 2.0*factor - 1.0
#                    factor = factor_original
##                    factor = min(factor, factor_original)

        return T_points, Ps_known, ys_known

    def Ks_and_dKs_dP(self, eos_l, eos_g, xs, ys):
        import numpy as np
        eos_l.fugacities()
        eos_g.fugacities()

        try:
            lnphis_l = eos_l.lnphis_l
            dlnphis_l_dP = eos_l.dlnphis_dP('l')
        except:
            lnphis_l = eos_l.lnphis_g
            dlnphis_l_dP = eos_l.dlnphis_dP('g')
        try:
            lnphis_g = eos_g.lnphis_g
            dlnphis_g_dP = eos_g.dlnphis_dP('g')
        except:
            lnphis_g = eos_g.lnphis_l
            dlnphis_g_dP = eos_g.dlnphis_dP('l')



        Ks = np.exp(np.array(lnphis_l) - np.array(lnphis_g))
        dKs_dP = (np.array(dlnphis_l_dP) - np.array(dlnphis_g_dP))*Ks
#        dKs_dP = (np.array(dlnphis_g_dP) - np.array(dlnphis_l_dP))*Ks
        return Ks, dKs_dP

    def Ks_and_dKs_dT(self, eos_l, eos_g, xs, ys):

        try:
            try:
                lnphis_l = eos_l.lnphis_l
                dlnphis_l_dT = eos_l.dlnphis_dT('l')
            except:
                lnphis_l = eos_l.lnphis_g
                dlnphis_l_dT = eos_l.dlnphis_dT('g')
            try:
                lnphis_g = eos_g.lnphis_g
                dlnphis_g_dT = eos_g.dlnphis_dT('g')
            except:
                lnphis_g = eos_g.lnphis_l
                dlnphis_g_dT = eos_g.dlnphis_dT('l')
        except:
            eos_l.fugacities()
            eos_g.fugacities()
            return self.Ks_and_dKs_dT(eos_l, eos_g, xs, ys)

        Ks = [exp(l - g) for l, g in zip(lnphis_l, lnphis_g)]
        dKs_dT = [(l - g)*Ki for l, g, Ki in zip(dlnphis_l_dT, dlnphis_g_dT, Ks)]

#        dKs_dT = (np.array(dlnphis_l_dT) - np.array(dlnphis_g_dT))*Ks
#        dKs_dT = (np.array(dlnphis_g_dT) - np.array(dlnphis_l_dT))*Ks
        return Ks, dKs_dT


    def d_VF_dT(self, delta=1e-4, full=True):
        # Not accurate enough
        VF1 = self.V_over_F
        zs = self.zs
        if full:
            new_eos = self.to_TP_zs(T=self.T+delta, P=self.P, zs=zs)
            # flash it
            VF2, xs, ys, eos_l, eos_g = new_eos.sequential_substitution_VL(xs=self.xs, ys=self.ys)
            return (VF2 - VF1)/delta



        Ks, dKs_dT = self.Ks_and_dKs_dT(self.eos_l, self.eos_g, self.xs, self.ys)
        # Perturb the Ks
        Ks2 = [Ki + dKi*delta for Ki, dKi in zip(Ks, dKs_dT)]
        VF2, _, _ = flash_inner_loop(zs, Ks2, guess=VF1)
        return (VF2 - VF1)/delta

    def d_VF_dP(self, delta=1e-4, full=True):
        # Not accurate enough
        VF1 = self.V_over_F
        zs = self.zs
        if full:
            new_eos = self.to_TP_zs(T=self.T, P=self.P+delta, zs=zs)
            # flash it
            VF2, xs, ys, eos_l, eos_g = new_eos.sequential_substitution_VL(xs=self.xs, ys=self.ys)
            return (VF2 - VF1)/delta


        Ks, dKs_dP = self.Ks_and_dKs_dP(self.eos_l, self.eos_g, self.xs, self.ys)
        # Perturb the Ks
        Ks2 = [Ki + dKi*delta for Ki, dKi in zip(Ks, dKs_dP)]
        VF2, _, _ = flash_inner_loop(zs, Ks2, guess=VF1)
        return (VF2 - VF1)/delta
