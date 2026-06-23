# -*- coding: utf-8 -*-
'''
MOSCED model for calculating activity coefficients at infinite dilution
author: Edgar Sanchez
'''
import numpy as np

# Parameters from Lazzaroni, M.J., Bush, D., Eckert, C.A., Frank, T.C., Gupta, S. and 
# Olson, J.D., 2005. Revision of MOSCED parameters and extension to solid 
# solubility calculations. Industrial & engineering chemistry research, 
# 44(11), pp.4075-4083.
# Retrieved from DDBST

mosced_params_2005={
'UOCLXMDMGBRAIB-UHFFFAOYSA-N' : [100.3, 16.54, 3.15, 1.01, 1.05, 0.85] ,
'SCYULBFZEHDVBN-UHFFFAOYSA-N' : [84.7, 16.77, 6.22, 0.92, 3.28, 1.56] ,
'WSLDOOZREJYCGB-UHFFFAOYSA-N' : [79.4, 16.6, 6.58, 0.94, 2.42, 1.34] ,
'FILVIKOEJGORQS-UHFFFAOYSA-N' : [115.2, 16.5, 8.45, 1.0, 0.0, 22.66] ,
'LRHPLDYGYMQRHN-UHFFFAOYSA-N' : [92.0, 14.82, 1.86, 1.0, 8.44, 11.01] ,
'VFWCMGCRMGJXDK-UHFFFAOYSA-N' : [105.1, 15.49, 3.38, 1.0, 0.11, 1.17] ,
'ZFPGARUNNKGOBB-UHFFFAOYSA-N' : [114.1, 16.74, 8.31, 1.0, 0.0, 20.75] ,
'ZSIAUFGUXNUGDI-UHFFFAOYSA-N' : [125.2, 15.02, 1.27, 1.0, 7.56, 9.2] ,
'LIKMAJRDDDTEIG-UHFFFAOYSA-N' : [125.8, 15.23, 0.22, 0.93, 0.0, 0.29] ,
'JSZOAYXJRCEYSX-UHFFFAOYSA-N' : [89.5, 15.17, 8.62, 1.0, 0.28, 5.83] ,
'KBPLFHHGFOOTCA-UHFFFAOYSA-N' : [158.2, 15.08, 1.31, 1.0, 4.22, 9.35] ,
'KWKAKUADMBZCLK-UHFFFAOYSA-N' : [157.8, 15.39, 0.44, 0.95, 0.0, 0.51] ,
'AMQJEAYHLZJPGS-UHFFFAOYSA-N' : [108.5, 15.25, 1.46, 1.0, 8.1, 9.51] ,
'YWAKXRMUMFPDSH-UHFFFAOYSA-N' : [110.3, 14.64, 0.25, 0.9, 0.0, 0.24] ,
'FFSAXUULYPJSKH-UHFFFAOYSA-N' : [145.2, 16.46, 4.98, 1.0, 0.88, 6.54] ,
'BDERNNFJNOPAEC-UHFFFAOYSA-N' : [75.1, 14.93, 1.39, 1.0, 11.97, 10.35] ,
'NHTMVDHEPJAVLT-UHFFFAOYSA-N' : [165.5, 14.08, 0.0, 1.0, 0.0, 0.0] ,
'HNRMPXKDFBEGFZ-UHFFFAOYSA-N' : [133.7, 13.77, 0.0, 1.0, 0.0, 0.0] ,
'CXOWYJMDMMMMJO-UHFFFAOYSA-N' : [148.9, 14.26, 0.0, 1.0, 0.0, 0.0] ,
'RLPGDEORIPLBNF-UHFFFAOYSA-N' : [159.5, 14.94, 0.0, 1.0, 0.0, 0.0] ,
'ZFFMLCVRJBZUDZ-UHFFFAOYSA-N' : [131.2, 14.3, 0.0, 1.0, 0.0, 0.0] ,
'BZHMBWZPUJHVEE-UHFFFAOYSA-N' : [150.0, 14.29, 0.0, 1.0, 0.0, 0.0] ,
'UWNADWZGEHDQAB-UHFFFAOYSA-N' : [165.6, 14.74, 0.0, 1.0, 0.0, 0.0] ,
'OISVCGZHLKNMSJ-UHFFFAOYSA-N' : [116.7, 15.95, 4.16, 0.9, 0.73, 13.12] ,
'BTANRVKWQNVYAZ-UHFFFAOYSA-N' : [92.0, 14.5, 1.56, 1.0, 8.03, 10.21] ,
'ZWEHNKRNPOVVGH-UHFFFAOYSA-N' : [90.2, 14.74, 6.64, 1.0, 0.0, 9.7] ,
'ZNQVEEAIQZEUHB-UHFFFAOYSA-N' : [97.3, 15.12, 7.39, 1.0, 3.77, 16.84] ,
'CATSNJVOTSVZJV-UHFFFAOYSA-N' : [140.7, 14.72, 4.2, 1.0, 0.0, 6.08] ,
'ZXEKIIBDNHEJCQ-UHFFFAOYSA-N' : [92.9, 14.19, 1.85, 1.0, 8.3, 10.52] ,
'DKGAVHZHDRPRBM-UHFFFAOYSA-N' : [94.7, 14.47, 2.55, 1.0, 5.8, 11.93] ,
'AFABGHUZZDYHJO-UHFFFAOYSA-N' : [132.9, 14.4, 0.0, 1.0, 0.0, 0.0] ,
'FGLBSLMDCBOPQK-UHFFFAOYSA-N' : [90.6, 14.6, 8.3, 1.0, 0.55, 3.43] ,
'XNLICIUVMPYHGG-UHFFFAOYSA-N' : [107.3, 15.07, 5.49, 1.0, 0.0, 8.09] ,
'KFZMGEQAYNKOFK-UHFFFAOYSA-N' : [76.8, 13.95, 1.95, 1.0, 9.23, 11.86] ,
'HNJBEVLQSNELDL-UHFFFAOYSA-N' : [76.8, 16.72, 11.36, 1.0, 2.39, 27.59] ,
'VLJXXKKOSFGPHI-UHFFFAOYSA-N' : [146.4, 14.95, 0.0, 1.0, 0.0, 0.0] ,
'PFEOZHBOMNWTJB-UHFFFAOYSA-N' : [130.4, 14.68, 0.0, 1.0, 0.0, 0.0] ,
'NTIZESTWPVYFNL-UHFFFAOYSA-N' : [125.8, 15.27, 4.71, 1.0, 0.0, 6.34] ,
'[emin][(CF3SO2)2N]'          : [258.6, 15.18, 10.72, 0.9, 9.79, 4.75] ,
'[emmin][(CF3SO2)2N]'         : [275.9, 15.25, 10.83, 0.9, 7.2, 5.11] ,
'IKHGUXGNUITLKF-UHFFFAOYSA-N' : [56.5, 13.76, 8.48, 1.0, 0.0, 6.5] ,
'QTBSBXVTEAMEQO-UHFFFAOYSA-N' : [57.6, 14.96, 3.23, 1.0, 24.03, 7.5] ,
'CSCPPACGZOOCGX-UHFFFAOYSA-N' : [73.8, 13.71, 8.3, 1.0, 0.0, 11.14] ,
'WEVYAHXRMPXWCK-UHFFFAOYSA-N' : [52.9, 13.78, 11.51, 1.0, 3.49, 8.98] ,
'KWOLFJPFCHCOCG-UHFFFAOYSA-N' : [117.4, 16.16, 6.5, 0.9, 1.71, 7.12] ,
'GRWFGVWFFZKLTI-UHFFFAOYSA-N' : [159.0, 17.32, 0.15, 0.95, 0.0, 1.3] ,
'PAYRUJLWNCNPSJ-UHFFFAOYSA-N' : [91.6, 16.51, 9.41, 0.9, 6.51, 6.34] ,
'RDOXTESZEPMUJZ-UHFFFAOYSA-N' : [109.2, 16.54, 5.63, 0.9, 0.75, 3.93] ,
'XKRFYHLGVUSROY-UHFFFAOYSA-N' : [57.1, 9.84, 0.0, 1.0, 0.0, 0.0] ,
'UHOVQNZJYSORNB-UHFFFAOYSA-N' : [89.5, 16.71, 3.95, 0.9, 0.63, 2.24] ,
'JFDZBHWFFUWGJE-UHFFFAOYSA-N' : [103.0, 15.43, 8.21, 0.9, 0.15, 7.41] ,
'QUKGYYKBILRGFE-UHFFFAOYSA-N' : [142.9, 16.17, 6.84, 0.9, 0.54, 5.53] ,
'WVDDGKGOMKODPV-UHFFFAOYSA-N' : [103.8, 16.56, 5.03, 1.0, 15.01, 6.69] ,
'QARVLSVVCXYDNA-UHFFFAOYSA-N' : [105.6, 17.1, 4.29, 0.89, 0.0, 3.13] ,
'RDHPKYGYEGBMSE-UHFFFAOYSA-N' : [75.3, 15.72, 4.41, 1.0, 0.22, 1.56] ,
'ZTQSAGDEMFDKMZ-UHFFFAOYSA-N' : [90.4, 15.11, 5.97, 1.0, 0.0, 5.27] ,
'IJDNQMDRQITEOD-UHFFFAOYSA-N' : [96.5, 13.7, 0.0, 1.0, 0.0, 0.0] ,
'KVNRLNFWIYMESJ-UHFFFAOYSA-N' : [87.9, 14.95, 8.27, 1.0, 0.0, 8.57] ,
'DKPFZGUDAPQIHT-UHFFFAOYSA-N' : [132.0, 15.22, 4.16, 1.0, 0.0, 6.4] ,
'OCKPCBLVNKHBMX-UHFFFAOYSA-N' : [156.6, 17.1, 2.51, 0.9, 0.1, 1.83] ,
'CURLTUGMZLYLDI-UHFFFAOYSA-N' : [42.2, 8.72, 5.68, 1.0, 1.87, 0.0] ,
'QGJOPFRUJISHPQ-UHFFFAOYSA-N' : [60.6, 19.67, 1.04, 1.0, 0.59, 0.33] ,
'UGFAIRIUMAVXCW-UHFFFAOYSA-N' : [49.0, 8.15, 0.0, 1.0, 0.0, 0.0] ,
'VZGDMQKNWNREIO-UHFFFAOYSA-N' : [97.1, 16.54, 1.82, 1.01, 1.25, 0.64] ,
'MVPPADPHJFYWMZ-UHFFFAOYSA-N' : [102.3, 16.72, 4.17, 0.89, 0.0, 2.5] ,
'HEDRZPFGACZZDS-UHFFFAOYSA-N' : [80.5, 15.61, 4.5, 0.96, 5.8, 0.12] ,
'DMEGYFMYUHOHGS-UHFFFAOYSA-N' : [121.7, 17.2, 0.0, 1.0, 0.0, 0.0] ,
'XDTMQSROBMDMFD-UHFFFAOYSA-N' : [108.9, 16.74, 0.0, 1.0, 0.0, 0.0] ,
'JHIVVAPYMSGYDF-UHFFFAOYSA-N' : [104.1, 15.8, 6.4, 1.0, 0.0, 10.71] ,
'WJTCGQSWYFHTAC-UHFFFAOYSA-N' : [134.9, 17.41, 0.0, 1.0, 0.0, 0.0] ,
'RGSFGYAAUTVSQA-UHFFFAOYSA-N' : [94.6, 16.55, 0.0, 1.0, 0.0, 0.0] ,
'DIOQZVSQGTUSAI-UHFFFAOYSA-N' : [195.8, 15.7, 0.0, 1.0, 0.0, 0.0] ,
'DURPTKYDGMDSBL-UHFFFAOYSA-N' : [170.4, 15.13, 1.73, 1.0, 0.0, 5.29] ,
'YMWUJEATGCHHMB-UHFFFAOYSA-N' : [64.4, 15.94, 6.23, 0.96, 3.98, 0.92] ,
'RTZKZFJDLAIYFH-UHFFFAOYSA-N' : [104.7, 13.96, 2.79, 1.0, 0.0, 6.61] ,
'FLKPEMZONWLCSK-UHFFFAOYSA-N' : [199.7, 16.33, 6.14, 1.0, 1.07, 7.81] ,
'NZZFYRREKKOMAT-UHFFFAOYSA-N' : [81.0, 21.9, 5.19, 1.0, 2.4, 2.08] ,
'ZAFNJMIOTHYJRJ-UHFFFAOYSA-N' : [141.8, 14.72, 1.9, 1.0, 0.0, 6.39] ,
'IEJIGPNLZYLLBP-UHFFFAOYSA-N' : [84.7, 17.81, 8.05, 1.0, 0.0, 7.32] ,
'RYHBNJHYFVUHQT-UHFFFAOYSA-N' : [85.7, 16.96, 6.72, 1.0, 0.0, 10.39] ,
'POLCUAVZOMRGSN-UHFFFAOYSA-N' : [137.6, 15.2, 2.0, 1.0, 0.0, 5.25] ,
'ZMXDDKWLCZADIW-UHFFFAOYSA-N' : [77.4, 15.95, 9.51, 1.0, 1.22, 22.65] ,
'IAZDPXIOMUYVGZ-UHFFFAOYSA-N' : [71.3, 16.12, 13.36, 1.0, 0.0, 26.17] ,
'SNRUBQQJIBEYMU-UHFFFAOYSA-N' : [228.6, 16.0, 0.0, 1.0, 0.0, 0.0] ,
'PAPBSGBWRJIAAV-UHFFFAOYSA-N' : [106.8, 16.42, 9.65, 1.0, 0.43, 13.06] ,
'LFQSCWFLJHTTHZ-UHFFFAOYSA-N' : [58.6, 14.37, 2.53, 1.0, 12.58, 13.29] ,
'XEKOWRVHYACXOJ-UHFFFAOYSA-N' : [98.6, 14.51, 5.74, 1.0, 0.0, 7.25] ,
'MTZQAGJQAFMTAQ-UHFFFAOYSA-N' : [144.1, 16.48, 4.97, 1.0, 0.28, 2.4] ,
'YNQLUTRBYVCPMQ-UHFFFAOYSA-N' : [122.9, 16.78, 2.98, 0.9, 0.23, 1.83] ,
'IIEWJVIFRVWJOD-UHFFFAOYSA-N' : [143.0, 16.34, 0.0, 1.0, 0.0, 0.0] ,
'ZTOMUSMDRMJOTH-UHFFFAOYSA-N' : [95.8, 15.12, 12.59, 1.0, 3.76, 9.11] ,
'IMNFDUFMRHMDMM-UHFFFAOYSA-N' : [147.0, 15.2, 0.0, 1.0, 0.0, 0.0] ,
'DCAYPVUWAIABOU-UHFFFAOYSA-N' : [294.2, 16.2, 0.0, 1.0, 0.0, 0.0] ,
'VLKZOEOYAKHREP-UHFFFAOYSA-N' : [131.4, 14.9, 0.0, 1.0, 0.0, 0.0] ,
'HVTICUPFWKNHNG-UHFFFAOYSA-N' : [93.6, 17.39, 3.58, 1.0, 0.51, 1.96] ,
'INQOMBQAUSQDDS-UHFFFAOYSA-N' : [62.7, 19.13, 4.21, 1.0, 1.16, 0.83] ,
'QWTDNUCVQCZILF-UHFFFAOYSA-N' : [117.1, 13.87, 0.0, 1.0, 0.0, 0.0] ,
'RWGFKTVRMDUZSP-UHFFFAOYSA-N' : [139.9, 17.09, 3.23, 0.9, 0.2, 2.57] ,
'RLSSMJSEOOYNOY-UHFFFAOYSA-N' : [105.0, 17.86, 4.16, 0.9, 27.15, 2.17] ,
'OKKJLVBELUTLKV-UHFFFAOYSA-N' : [40.6, 14.43, 3.77, 1.0, 17.43, 14.49] ,
'KXKVLQRXCPHEJC-UHFFFAOYSA-N' : [79.8, 13.59, 7.54, 1.0, 0.0, 8.38] ,
'TZIHFWKZFHZASV-UHFFFAOYSA-N' : [62.1, 18.79, 8.29, 1.0, 0.37, 8.62] ,
'BZLVMXJERCGZMT-UHFFFAOYSA-N' : [119.9, 15.17, 2.48, 1.0, 0.0, 7.4] ,
'UAEPNZWRGJTJPN-UHFFFAOYSA-N' : [128.2, 16.06, 0.0, 1.0, 0.0, 0.0] ,
'GDOPTJXRTPNYNR-UHFFFAOYSA-N' : [113.0, 16.1, 0.0, 1.0, 0.0, 0.0] ,
'NZMAJUHVSZBJHL-UHFFFAOYSA-N' : [182.0, 15.99, 5.02, 1.0, 0.24, 14.07] ,
'AJFDBNQQDYLMJN-UHFFFAOYSA-N' : [124.5, 15.66, 6.71, 1.0, 0.25, 18.67] ,
'FXHOOIRPVKKKFG-UHFFFAOYSA-N' : [93.0, 15.86, 9.46, 1.0, 0.0, 21.0] ,
'PMDCZENCAXMSOU-UHFFFAOYSA-N' : [94.3, 16.07, 4.91, 1.0, 4.14, 22.45] ,
'LCEDQNDDFOCWGG-UHFFFAOYSA-N' : [100.6, 16.1, 10.91, 1.0, 2.42, 19.29] ,
'OHLUUHNLEMFGTQ-UHFFFAOYSA-N' : [76.9, 16.22, 5.9, 1.0, 5.28, 23.58] ,
'ATHHXGZTWNVVOU-UHFFFAOYSA-N' : [59.1, 15.55, 8.92, 1.0, 8.07, 22.01] ,
'SECXISVLQFMRJM-UHFFFAOYSA-N' : [96.6, 17.64, 9.34, 1.0, 0.0, 24.22] ,
'LQNUZADURLCDLV-UHFFFAOYSA-N' : [102.7, 16.06, 8.23, 0.9, 0.98, 3.29] ,
'MCSAJNNLRCFZED-UHFFFAOYSA-N' : [72.0, 14.68, 9.96, 1.0, 1.19, 4.72] ,
'IJGRMHOSHXDMSA-UHFFFAOYSA-N' : [50.0, 7.48, 0.0, 1.0, 0.0, 0.0] ,
'LYGJENNIWJXYER-UHFFFAOYSA-N' : [54.1, 13.48, 12.44, 1.0, 4.07, 4.01] ,
'BKIMMITUMNQMOS-UHFFFAOYSA-N' : [179.6, 15.6, 0.0, 1.0, 0.0, 0.0] ,
'TVMXDCGIABBOFY-UHFFFAOYSA-N' : [163.4, 15.4, 0.0, 1.0, 0.0, 0.0] ,
'MYMOFIZGZYHOMD-UHFFFAOYSA-N' : [52.9, 8.84, 0.0, 1.0, 0.0, 0.0] ,
'URLKBWYHVLBVBO-UHFFFAOYSA-N' : [123.9, 16.06, 2.7, 0.9, 0.27, 1.87] ,
'OFBQJSOFQDEBGM-UHFFFAOYSA-N' : [116.0, 14.4, 0.0, 1.0, 0.0, 0.0] ,
'ISWSIDIOOBJBQZ-UHFFFAOYSA-N' : [88.9, 16.66, 4.5, 0.9, 25.14, 5.35] ,
'ATUOYWHBWRKTHZ-UHFFFAOYSA-N' : [75.7, 13.1, 0.0, 1.0, 0.0, 0.0] ,
'FVSKHRXBFJPNKK-UHFFFAOYSA-N' : [70.9, 14.95, 9.82, 1.0, 1.08, 6.83] ,
'YKYONYBAUNKHLG-UHFFFAOYSA-N' : [115.8, 13.98, 5.45, 1.0, 0.0, 7.53] ,
'JUJWROOIHBZHMG-UHFFFAOYSA-N' : [80.9, 16.39, 6.13, 0.9, 1.61, 14.93] ,
'SMWDFEZZVXVKRB-UHFFFAOYSA-N' : [118.5, 16.84, 5.96, 0.9, 2.17, 12.1] ,
'PRAKJMSDJKAYCZ-UHFFFAOYSA-N' : [526.1, 14.49, 0.0, 1.0, 0.0, 0.0] ,
'HXJUTPCZVOIRIF-UHFFFAOYSA-N' : [95.3, 16.49, 12.16, 1.0, 1.36, 13.52] ,
'BGHCVCJVXZWKCC-UHFFFAOYSA-N' : [261.3, 16.1, 0.0, 1.0, 0.0, 0.0] ,
'ZUHZGEOKBKGPSW-UHFFFAOYSA-N' : [221.1, 16.08, 6.73, 1.0, 0.0, 13.53] ,
'WYURNTSHIVDZCO-UHFFFAOYSA-N' : [81.9, 15.78, 4.41, 1.0, 0.0, 10.43] ,
'YXFVVABEGXRONW-UHFFFAOYSA-N' : [106.7, 16.61, 3.22, 0.9, 0.57, 2.23] ,
'STCOOQWBFONSKY-UHFFFAOYSA-N' : [345.0, 15.05, 4.87, 1.0, 0.0, 14.06] ,
'XSTXAVWGXDQKEL-UHFFFAOYSA-N' : [90.1, 17.19, 2.96, 1.0, 2.07, 0.21] ,
'ZMANZCXQSJIPKH-UHFFFAOYSA-N' : [139.7, 14.49, 1.02, 1.0, 0.0, 7.7] ,
'XLYOFNOQVPJJNP-UHFFFAOYSA-N' : [36.0, 10.58, 10.48, 1.0, 52.78, 15.86] 
    }


class MOSCED():
    r'''Class for calculating activity coefficients at infinite dilution from 
    the solvation model Modified Separation of Cohesive Energy Density (MOSCED). 
    The implementation is based on the original paper [1]_.
    
    Note: The value 3.4 in the equation for xi is different from the value given 
    in the publication. The value in the publication (3.24) has been confirmed 
    to be a typing error.
    
    In the following equations 1 refers to the solvent and 2 refers to the solute 
    
    .. math::
        alpha^T = alpha \left( \frac{293}{T} \right)^{0.8}

    .. math::
        beta^T = beta \left( \frac{293}{T} \right)^{0.8}
        
    .. math::
        tau^T = tau \left( \frac{293}{T} \right)^{0.4}
        
    .. math::
        POL = q_1^4 \left[ 1.15 - 1.15 exp \left( -0.002337 (tau_1^T)^3 \right) \right] + 1
        
    .. math::
        xi = 0.68 (POL-1) + \left[3.4 - 2.4 exp \left(-0.002687 (alpha_1 beta_1)^{1.5} \right) \right]^{(\frac{293}{T})^2}
        
    .. math::
        psi = POL + 0.002629 alpha_1^T beta_1^T
        
    .. math::
        aa = 0.953 - 0.002314\left( (tau_2^T)^2 + alpha_2^T beta_2^T \right)
        
    .. math::
        d_{12} = ln \left(\left(\frac{v_2}{v_1} \right)^{aa} \right) + 1 - \left(\frac{v_2}{v_1} \right)^{aa}
    
    .. math::
        gamma_2^{\infty} = exp\left( \frac{v_2}{RT} \left[ (\lambda_1 - lambda_2)^2 + \frac{q_1^2 q_2^2(tau_1^T - tau_2^T)^2}{psi} + \frac{(alpha_1^T - alpha_2^T)(beta_1^T - beta_2^T)}{xi}  \right] + d_{12} \right)
        

    Examples
    --------
    The DDBST has published an online tool with the implementation of MOSCED here:
        http://ddbonline.ddbst.com/MOSCEDCalculation/MOSCEDCGI.exe
        
    Let's compare the following system at 298.15 K:
        - Component 1 (solvent): water
        - Component 2 (solute): 1,1,1-trichloroethane
        
    The corresponding InChikeys are:
        - Component 1: XLYOFNOQVPJJNP-UHFFFAOYSA-N
        - Component 2: UOCLXMDMGBRAIB-UHFFFAOYSA-N

    >>> from thermo.mosced import MOSCED
    >>> binary_sys = MOSCED()
    >>> solvent_params = binary_sys.get_mosced_params('XLYOFNOQVPJJNP-UHFFFAOYSA-N')
    >>> solute_params = binary_sys.get_mosced_params('UOCLXMDMGBRAIB-UHFFFAOYSA-N')
    >>> binary_sys.load_mosced_parameters(298.15, solvent_params, solute_params)
    >>> binary_sys.gamma_infinite_dilution()
    7971.723485660339

    The solution given by the DDB tool has the same value: 7972.02

    References
    ----------
    .. [1] Lazzaroni, M.J., Bush, D., Eckert, C.A., Frank, T.C., Gupta, S. and 
    Olson, J.D., 2005. Revision of MOSCED parameters and extension to solid 
    solubility calculations. Industrial & engineering chemistry research, 
    44(11), pp.4075-4083.
    
    '''
    
    
    def __init__(self):
        self.mosced_params_names = ['V', 'lambda', 'tau', 'q', 'alpha', 'beta']
        
    def get_mosced_params(self, inchikey, verbose=False):
        '''
        Retrieve MOSCED parameters from the database of the original paper [1]_.

        Parameters
        ----------
        inchikey : str
            InChI key of the component of interest.
        verbose : Bool, optional
            The default is False. 
            Whether to give notice in case that the passed inchikey is not 
            present in the database

        Returns
        -------
        params_dict : dict
            MOSCED parameters of the compound

        '''
        
        if inchikey in mosced_params_2005:
            params_dict={}
            for i, param in enumerate(mosced_params_2005[inchikey]):
                params_dict[self.mosced_params_names[i]] = param
            return params_dict
        else:
            if verbose:
                print('InChIkey is not present at the MOSCED database')
            return None
            
    def load_mosced_parameters(self, T, solvent_params, solute_params):
        '''
        Load MOSCED parameters of the solvent and solute, and temperature of
        the system.

        Parameters
        ----------
        T : float
            Temperature of the system in K
        solvent_params : dict
            MOSCED parameters for the solvent. The corresponding parameters
            need to be stored with the following keys:
                ['V', 'lambda', 'tau', 'q', 'alpha', 'beta']
        solute_params : dict
            MOSCED parameters for the solute. The corresponding parameters
            need to be stored with the following keys:
                ['V', 'lambda', 'tau', 'q', 'alpha', 'beta']

        '''
        self.T = T
        self.solvent_params = solvent_params
        self.solute_params = solute_params
        self.collect_params_sys()
        
    def gamma_infinite_dilution(self):
        '''
        Returns
        -------
        float
            Activity coefficient at infinite dilution for the specified system

        '''
        v2 = self.sys_params['V_2']
        R = 8.31446261815324
        T = self.T
        l1 = self.sys_params['lambda_1']
        l2 = self.sys_params['lambda_2']
        q1 = self.sys_params['q_1']
        q2 = self.sys_params['q_2']
        t1_T, t2_T = self.tau_T()
        psi = self.psi()
        a1_T, a2_T = self.alpha_T()
        b1_T, b2_T = self.beta_T()
        xi = self.xi()
        d12 = self.d12()
        
        part_a = (l1-l2)**2
        part_b = q1**2*q2**2*(t1_T - t2_T)**2/psi
        part_c = (a1_T - a2_T)*(b1_T - b2_T)/xi
        
        return np.exp(v2/(R*T)*(part_a + part_b + part_c) + d12)
        
    def collect_params_sys(self):
        '''
        Creates a joined dictionary to store the MOSCED parameters of the 
        solvent and solute using the identifiers 1 and 2, respectively. This is
        for simple readability.

        Raises
        ------
        KeyError
            In case one of the 6 MOSCED parameters has not been passed 

        '''
        solvent_id = 1
        solute_id = 2
        self.sys_params ={}
        for name in self.mosced_params_names:
            if name not in self.solvent_params.keys():
                raise KeyError(f'{name} key is missing on the solvent params dict')
            if name not in self.solute_params.keys():
                raise KeyError(f'{name} key is missing on the solute params dict')
            self.sys_params[name+'_'+str(solvent_id)] = self.solvent_params[name]
            self.sys_params[name+'_'+str(solute_id)] = self.solute_params[name]
        
    def alpha_T(self):
        a1_T = self.sys_params['alpha_1']*(293/self.T)**0.8
        a2_T = self.sys_params['alpha_2']*(293/self.T)**0.8
        return [a1_T, a2_T]
    
    def beta_T(self):
        b1_T = self.sys_params['beta_1']*(293/self.T)**0.8
        b2_T = self.sys_params['beta_2']*(293/self.T)**0.8
        return [b1_T, b2_T]
    
    def tau_T(self):
        t1_T = self.sys_params['tau_1']*(293/self.T)**0.4
        t2_T = self.sys_params['tau_2']*(293/self.T)**0.4
        return [t1_T, t2_T]
    
    def POL(self):
        q1 = self.sys_params['q_1']
        t1_T = self.tau_T()[0]
        return q1**4 * (1.15-1.15*np.exp(-.002337*t1_T**3)) + 1
    
    def xi(self):
        POL = self.POL()
        a1 = self.sys_params['alpha_1']
        b1 = self.sys_params['beta_1']
        xi_a = 0.68*(POL-1)
        xi_b = 3.4 - 2.4*np.exp(-0.002687*(a1*b1)**1.5)
        return xi_a + xi_b**((293/self.T)**2)
    
    def psi(self):
        POL = self.POL()
        a1_T = self.alpha_T()[0]
        b1_T = self.beta_T()[0]
        return POL + 0.002629*a1_T*b1_T
    
    def aa(self):
        t2_T = self.tau_T()[1]
        a2_T = self.alpha_T()[1]
        b2_T = self.beta_T()[1]
        return 0.953 - 0.002314*(t2_T**2 + a2_T*b2_T)
    
    def d12(self):
        v2 = self.sys_params['V_2']
        v1 = self.sys_params['V_1']
        aa = self.aa()
        return np.log((v2/v1)**aa) + 1 - (v2/v1)**aa
       
