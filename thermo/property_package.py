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


__all__ = ['PropertyPackage', 'Ideal',
           'IdealCaloric',
           'StabilityTester']

try:
    from random import seed, uniform
except:
    pass

from chemicals.flash_basic import K_value, Wilson_K_value
from chemicals.rachford_rice import Rachford_Rice_flash_error, flash_inner_loop
from chemicals.utils import normalize, remove_zeros
from fluids.constants import R
from fluids.numerics import UnconvergedError, brenth, derivative, exp, log, secant
from fluids.numerics import numpy as np

from thermo.serialize import JsonOptEncodable

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






class StabilityTester:
    json_version = 1
    obj_references = []
    non_json_attributes = []
    vectorized = False
    __full_path__ = f"{__module__}.{__qualname__}"

    def __init__(self, Tcs, Pcs, omegas, aqueous_check=False, CASs=None):
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.omegas = omegas
        self.N = len(Tcs)
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

        arr = -np.ones(len(guesses[0]) - 1)
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
        cmps = range(self.N)
        pure_guesses = [normalize([zero_fraction if j != k else 1 for j in cmps])
                       for k in cmps]
        return pure_guesses

    def Wilson_guesses(self, T, P, zs, powers=(1, -1, 1/3., -1/3.)):
        # First K is vapor-like phase; second, liquid like
        cmps = range(self.N)
        Ks_Wilson = [Wilson_K_value(T=T, P=P, Tc=self.Tcs[i], Pc=self.Pcs[i], omega=self.omegas[i]) for i in cmps]
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
        N = self.N
        cmps = range(N)
        Ks_Wilson = [Wilson_K_value(T=T, P=P, Tc=self.Tcs[i], Pc=self.Pcs[i], omega=self.omegas[i]) for i in cmps]
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
        N = self.N
        cmps = range(N)
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

    as_json = JsonOptEncodable.as_json
    from_json = JsonOptEncodable.from_json

class PropertyPackage:


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
                                 f'temperature bound {T_low:g} K has an enthalpy ({Hm_low:g} '
                                 f'J/mol) higher than that requested ({Hm:g} J/mol)')
            if Hm_high is None:
                pkg_high = self.to(T=T_high, P=P, zs=zs)
                pkg_high._post_flash()
                Hm_high = pkg_high.Hm
            if Hm > Hm_high:
                raise ValueError('The requested molar enthalpy cannot be found'
                                 ' with this bounded solver because the upper '
                                 f'temperature bound {T_high:g} K has an enthalpy ({Hm_high:g} '
                                 f'J/mol) lower than that requested ({Hm:g} J/mol)')


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
                                 f'temperature bound {T_low:g} K has an entropy ({Sm_low:g} '
                                 f'J/mol/K) higher than that requested ({Sm:g} J/mol/K)')
            if Sm_high is None:
                pkg_high = self.to(T=T_high, P=P, zs=zs)
                pkg_high._post_flash()
                Sm_high = pkg_high.Sm
            if Sm > Sm_high:
                raise ValueError('The requested molar entropy cannot be found'
                                 ' with this bounded solver because the upper '
                                 f'temperature bound {T_high:g} K has an entropy ({Sm_high:g} '
                                 f'J/mol/K) lower than that requested ({Sm:g} J/mol/K)')

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
                                 f'pressure bound {P_low:g} Pa has an entropy ({Sm_low:g} '
                                 f'J/mol/K) lower than that requested ({Sm:g} J/mol/K)')
            if Sm_high is None:
                pkg_high = self.to(T=T, P=P_high, zs=zs)
                pkg_high._post_flash()
                Sm_high = pkg_high.Sm
            if Sm < Sm_high:
                raise ValueError('The requested molar entropy cannot be found'
                                 ' with this bounded solver because the upper '
                                 f'pressure bound {P_high:g} Pa has an entropy ({Sm_high:g} '
                                 f'J/mol/K) upper than that requested ({Sm:g} J/mol/K)')
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
        """Cp_ideal, Cp_real, speed of sound -- or come up with a way for
        mixture to better make calls to the property package. Probably both.
        """

