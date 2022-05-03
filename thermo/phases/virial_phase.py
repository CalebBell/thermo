# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
__all__ = ['VirialCSP', 'VirialGas']

from fluids.constants import R, R_inv
from fluids.numerics import newton, numpy as np
from chemicals.utils import log, mixing_simple, dxs_to_dns, dxs_to_dn_partials, dns_to_dn_partials, hash_any_primitive
from thermo.heat_capacity import HeatCapacityGas
from thermo.phases.phase import Phase
from thermo.phases.ceos import CEOSGas


from chemicals.virial import (BVirial_Pitzer_Curl_vec,BVirial_Pitzer_Curl_mat,
                                BVirial_Abbott_vec, BVirial_Abbott_mat,
                                BVirial_Tsonopoulos_vec, BVirial_Tsonopoulos_mat,
                                BVirial_Tsonopoulos_extended_vec,
                                BVirial_Tsonopoulos_extended_mat,
                                Meng_virial_a,
                                BVirial_Meng_vec, BVirial_Meng_mat,
                                BVirial_Oconnell_Prausnitz_vec,
                                BVirial_Oconnell_Prausnitz_mat,
                                BVirial_Xiang_vec,
                                BVirial_Xiang_mat,
                              Z_from_virial_density_form, BVirial_mixture,
                              dBVirial_mixture_dzs, d2BVirial_mixture_dzizjs, d3BVirial_mixture_dzizjzks,
                              CVirial_mixture_Orentlicher_Prausnitz,
                              dCVirial_mixture_dT_Orentlicher_Prausnitz,
                              d2CVirial_mixture_dT2_Orentlicher_Prausnitz,
                              d3CVirial_mixture_dT3_Orentlicher_Prausnitz,
                              dCVirial_mixture_Orentlicher_Prausnitz_dzs,
                              d2CVirial_mixture_Orentlicher_Prausnitz_dzizjs,
                              d3CVirial_mixture_Orentlicher_Prausnitz_dzizjzks,
                              d2CVirial_mixture_Orentlicher_Prausnitz_dTdzs,
                              CVirial_Orbey_Vera_mat, CVirial_Liu_Xiang_mat,
                              CVirial_Orbey_Vera_vec, CVirial_Liu_Xiang_vec,
                              Tarakad_Danner_virial_CSP_kijs, Tarakad_Danner_virial_CSP_Tcijs,
                              Tarakad_Danner_virial_CSP_Pcijs, Lee_Kesler_virial_CSP_Vcijs,
                              Tarakad_Danner_virial_CSP_omegaijs,
                              dV_dzs_virial, d2V_dzizjs_virial)


try:
    array, zeros, ones, delete, npsum, nplog = np.array, np.zeros, np.ones, np.delete, np.sum, np.log
except (ImportError, AttributeError):
    pass

VIRIAL_B_ZERO = 'VIRIAL_B_ZERO'
VIRIAL_B_PITZER_CURL = 'VIRIAL_B_PITZER_CURL'
VIRIAL_B_ABBOTT = 'VIRIAL_B_ABBOTT'
VIRIAL_B_TSONOPOULOS = 'VIRIAL_B_TSONOPOULOS'
VIRIAL_B_TSONOPOULOS_EXTENDED = 'VIRIAL_B_TSONOPOULOS_EXTENDED' # requires `a` and `b` parameter
VIRIAL_B_OCONNELL_PRAUSNITZ = "VIRIAL_B_OCONNELL_PRAUSNITZ"
VIRIAL_B_XIANG = 'VIRIAL_B_XIANG'
VIRIAL_B_MENG = 'VIRIAL_B_MENG'

VIRIAL_B_MODELS = (VIRIAL_B_ZERO, 
                   VIRIAL_B_PITZER_CURL,
                   VIRIAL_B_ABBOTT,
                   VIRIAL_B_TSONOPOULOS,
                   VIRIAL_B_TSONOPOULOS_EXTENDED,
                   VIRIAL_B_OCONNELL_PRAUSNITZ,
                   VIRIAL_B_XIANG,
                   VIRIAL_B_MENG)

__all__.extend(VIRIAL_B_MODELS)
# reqiures an `a` parameter




VIRIAL_C_XIANG = 'VIRIAL_C_XIANG'
VIRIAL_C_ORBEY_VERA = 'VIRIAL_C_ORBEY_VERA'
VIRIAL_C_ZERO = 'VIRIAL_C_ZERO'

VIRIAL_C_MODELS = (VIRIAL_C_ZERO, VIRIAL_C_XIANG, VIRIAL_C_ORBEY_VERA)
__all__.extend(VIRIAL_C_MODELS)


VIRIAL_CROSS_B_ZEROS = VIRIAL_CROSS_C_ZEROS = 'Zeros'
VIRIAL_CROSS_B_TARAKAD_DANNER = 'Tarakad-Danner'
VIRIAL_CROSS_C_TARAKAD_DANNER = 'Tarakad-Danner'


class VirialCSP(object):
    r'''Class for calculating the `B` virial coefficients of pure components
    and their B interaction matrix, and the `C` virial coefficients of pure
    components and their mixtures. It is configurable which corresponding
    states model is used. Either the `B` or `C` model can be disabled;
    if both are off, this will revert to the ideal-gas equation of state.
    
    Parameters
    ----------
    Tcs : list[float]
        Critical temperatures of all components, [K]
    Pcs : list[float]
        Critical pressures of all components, [Pa]
    Vcs : list[float]
        Critical volumes of all components, [m^3/mol]
    omegas : list[float]
        Acentric factors of all components, [-]
    B_model : str, optional
        The model used to calculate the `B` pure component and interaction 
        virial coefficients, [-]
        
        * **VIRIAL_B_ZERO**: The B virial coefficient is always zero
        * **VIRIAL_B_PITZER_CURL** The model of [2]_, :obj:`chemicals.virial.BVirial_Pitzer_Curl`
        * **VIRIAL_B_ABBOTT** The model of [3]_, :obj:`chemicals.virial.BVirial_Abbott`
        * **VIRIAL_B_TSONOPOULOS** The model of [4]_, :obj:`chemicals.virial.BVirial_Tsonopoulos`
        * **VIRIAL_B_TSONOPOULOS_EXTENDED** The model of [5]_ and [6]_, :obj:`chemicals.virial.BVirial_Tsonopoulos_extended`
        * **VIRIAL_B_OCONNELL_PRAUSNITZ** The model of [1]_, :obj:`chemicals.virial.BVirial_Oconnell_Prausnitz`
        * **VIRIAL_B_XIANG** The model of [7]_, :obj:`chemicals.virial.BVirial_Xiang`
        * **VIRIAL_B_MENG** The model of [8]_, :obj:`chemicals.virial.BVirial_Meng`

    cross_B_model : str, optional
        The model used to calculate the `B` cross virial coefficient
        
        * **VIRIAL_CROSS_B_TARAKAD_DANNER** : This model uses the mixing rules
          for estimating interaction critical components according to the
          rules  :obj:`chemicals.virial.Tarakad_Danner_virial_CSP_Tcijs`,  
          :obj:`chemicals.virial.Tarakad_Danner_virial_CSP_Pcijs`,
          :obj:`chemicals.virial.Lee_Kesler_virial_CSP_Vcijs` and
          :obj:`chemicals.virial.Tarakad_Danner_virial_CSP_omegaijs`;
          note that this mixing rule has an interaction parameter for the
          interaction critical temperature, which defaults to zero and can be
          provided. :obj:`chemicals.virial.Meng_Duan_2005_virial_CSP_kijs`
          or  :obj:`chemicals.virial.Tarakad_Danner_virial_CSP_kijs` are two
          sample models for estimating these parameters; additional models are
          available in the literature and also the value can be regressed from
          experimental values.
    cross_B_model_kijs : list[list[float]], optional
        Cross parameters `kijs` for **VIRIAL_CROSS_B_TARAKAD_DANNER** cross
        rule; specified or set to zero [-]
    C_model : str, optional
        The model used to calculate the `C` pure component and interaction 
        virial coefficients, [-]
        
        * **VIRIAL_C_ZERO**: The C virial coefficient is always zero
        * **VIRIAL_C_ORBEY_VERA** The model of [9]_, :obj:`chemicals.virial.CVirial_Orbey_Vera`
        * **VIRIAL_C_XIANG** The model of [10]_, :obj:`chemicals.virial.CVirial_Liu_Xiang`
        
    B_model_Meng_as : list[list[float]], optional
        Meng `a` parameters; this is essentially a correction for polar
        behavior, and must be provided for all components as well as their
        interactions; see :obj:`chemicals.virial.Meng_virial_a`.
        This is used only for the model **VIRIAL_B_MENG** [-]
    B_model_Tsonopoulos_extended_as : list[list[float]], optional
        Tsonopoulos extended `a` parameters; this is essentially a correction for polar
        behavior, and must be provided for all components as well as their
        interactions; see :obj:`thermo.functional_groups.BVirial_Tsonopoulos_extended_ab`.
        This is used only for the model **VIRIAL_B_TSONOPOULOS_EXTENDED** [-]
    B_model_Tsonopoulos_extended_bs : list[list[float]], optional
        Meng `a` parameters; this is essentially a correction for polar
        behavior, and must be provided for all components as well as their
        interactions; see :obj:`thermo.functional_groups.BVirial_Tsonopoulos_extended_ab`.
        This is used only for the model **VIRIAL_B_TSONOPOULOS_EXTENDED** [-]
    T : float, optional
        The specified temperature for the model; the calculations are cached
        based only on temperature, use :obj:`VirialCSP.to` to obtain a new
        object at a different temperature, [K]

    Examples
    --------

    Notes
    -----

    References
    ----------
    .. [1] O’Connell, J. P., and J. M. Prausnitz. "Empirical Correlation of 
       Second Virial Coefficients for Vapor-Liquid Equilibrium Calculations."
       Industrial & Engineering Chemistry Process Design and Development 6, 
       no. 2 (April 1, 1967): 245-50. https://doi.org/10.1021/i260022a016.
    .. [2] Pitzer, Kenneth S., and R. F. Curl. "The Volumetric and
       Thermodynamic Properties of Fluids. III. Empirical Equation for the
       Second Virial Coefficient1." Journal of the American Chemical Society
       79, no. 10 (May 1, 1957): 2369-70. doi:10.1021/ja01567a007.
    .. [3] Smith, H. C. Van Ness Joseph M. Introduction to Chemical Engineering
       Thermodynamics 4E 1987.
    .. [4] Tsonopoulos, Constantine. "An Empirical Correlation of Second Virial
       Coefficients." AIChE Journal 20, no. 2 (March 1, 1974): 263-72.
       doi:10.1002/aic.690200209.
    .. [5] Tsonopoulos, C., and J. L. Heidman. "From the Virial to the Cubic
       Equation of State." Fluid Phase Equilibria 57, no. 3 (1990): 261-76.
       doi:10.1016/0378-3812(90)85126-U
    .. [6] Tsonopoulos, Constantine, and John H. Dymond. "Second Virial
       Coefficients of Normal Alkanes, Linear 1-Alkanols (and Water), Alkyl
       Ethers, and Their Mixtures." Fluid Phase Equilibria, International
       Workshop on Vapour-Liquid Equilibria and Related Properties in Binary
       and Ternary Mixtures of Ethers, Alkanes and Alkanols, 133, no. 1-2
       (June 1997): 11-34. doi:10.1016/S0378-3812(97)00058-7.
    .. [7] Xiang, H. W. "The New Simple Extended Corresponding-States 
       Principle: Vapor Pressure and Second Virial Coefficient." Chemical 
       Engineering Science 57, no. 8 (April 2002): 1439049. 
       https://doi.org/10.1016/S0009-2509(02)00017-9.
    .. [8] Meng, Long, Yuan-Yuan Duan, and Lei Li. "Correlations for Second and
       Third Virial Coefficients of Pure Fluids." Fluid Phase Equilibria 226
       (December 10, 2004): 109-20. https://doi.org/10.1016/j.fluid.2004.09.023.
    .. [9] Orbey, Hasan, and J. H. Vera. "Correlation for the Third Virial 
       Coefficient Using Tc, Pc and ω as Parameters." AIChE Journal 29, no. 1 
       (January 1, 1983): 107-13. https://doi.org/10.1002/aic.690290115.
    .. [10] Liu, D. X., and H. W. Xiang. "Corresponding-States Correlation and
       Prediction of Third Virial Coefficients for a Wide Range of Substances."
       International Journal of Thermophysics 24, no. 6 (November 1, 2003):
       1667-80. https://doi.org/10.1023/B:IJOT.0000004098.98614.38.
    '''
    cross_B_calculated = False
    cross_C_calculated = False
    pure_B_calculated = False
    pure_C_calculated = False

    nonstate_constants = ('Tcs', 'Pcs', 'Vcs', 'omegas', 'B_model', 'cross_B_model', 
                          'cross_B_model_kijs', 'C_model', 'B_model_Meng_as',
                          'B_model_Tsonopoulos_extended_as', 'B_model_Tsonopoulos_extended_bs')
        
    def __repr__(self):
        r'''Method to create a string representation of the VirialCSP object, with
        the goal of making it easy to obtain standalone code which reproduces
        the current state of the phase. This is extremely helpful in creating
        new test cases.

        Returns
        -------
        recreation : str
            String which is valid Python and recreates the current state of
            the object if ran, [-]

        Examples
        --------
        
        >>> from thermo import VirialCSP
        >>> model = VirialCSP(Tcs=[126.2, 154.58, 150.8], Pcs=[3394387.5, 5042945.25, 4873732.5], Vcs=[8.95e-05, 7.34e-05, 7.49e-05], omegas=[0.04, 0.021, -0.004], B_model='VIRIAL_B_PITZER_CURL', cross_B_model='Tarakad-Danner', C_model='VIRIAL_C_ORBEY_VERA')
        >>> model
        VirialCSP(Tcs=[126.2, 154.58, 150.8], Pcs=[3394387.5, 5042945.25, 4873732.5], Vcs=[8.95e-05, 7.34e-05, 7.49e-05], omegas=[0.04, 0.021, -0.004], B_model='VIRIAL_B_PITZER_CURL', cross_B_model='Tarakad-Danner', C_model='VIRIAL_C_ORBEY_VERA')
        '''
        try:
            Cpgs = ', '.join(str(o) for o in self.HeatCapacityGases)
        except:
            Cpgs = ''
        base = '%s('  %(self.__class__.__name__,)
        for s in self.nonstate_constants + ('T',):
            if hasattr(self, s) and getattr(self, s) is not None:
                val = getattr(self, s)
                if type(val) is str:
                    val = f"'{val}'"
                elif isinstance(val, (np.ndarray, list,)):
                    if not np.any(val):
                        continue
                base += '%s=%s, ' %(s, val)
        if base[-2:] == ', ':
            base = base[:-2]
        base += ')'
        return base


    def state_hash(self):
        r'''Basic method to calculate a hash of the state of the model and its
        model parameters.

        Note that the hashes should only be compared on the same system running
        in the same process!

        Returns
        -------
        state_hash : int
            Hash of the object's model parameters and state, [-]
        '''
        #print((self.model_hash(), self.T), 'state hash args')
        return hash_any_primitive((self.model_hash(), self.T))

    def model_hash(self):
        r'''Basic method to calculate a hash of the non-state parts of the model
        This is useful for comparing to models to
        determine if they are the same, i.e. in a VLL flash it is important to
        know if both liquids have the same model.

        Note that the hashes should only be compared on the same system running
        in the same process!

        Returns
        -------
        model_hash : int
            Hash of the object's model parameters, [-]
        '''
        try:
            return self._model_hash
        except AttributeError:
            pass
        h = hash(self.__class__.__name__)
        for s in self.nonstate_constants:
            try:
                #print(s,getattr(self, s), hash((h, s, hash_any_primitive(getattr(self, s)))))
                h = hash((h, s, hash_any_primitive(getattr(self, s))))
            except AttributeError:
                pass
        self._model_hash = h
        return h

    def __hash__(self):
        r'''Method to calculate and return a hash representing the exact state
        of the object.

        Returns
        -------
        hash : int
            Hash of the object, [-]
        '''
        d = self.__dict__
        ans = hash_any_primitive((self.__class__.__name__, self.state_hash(), self.model_hash()))
        return ans

    def __eq__(self, other):
        return self.__hash__() == hash(other)

    def __init__(self, Tcs, Pcs, Vcs, omegas,
                 B_model=VIRIAL_B_XIANG, 
                 
                 cross_B_model=VIRIAL_CROSS_B_TARAKAD_DANNER,
                 # always require kijs in this model
                 cross_B_model_kijs=None,
                 
                 C_model=VIRIAL_C_XIANG,
                 
                 B_model_Meng_as=None,
                 B_model_Tsonopoulos_extended_as=None,
                 B_model_Tsonopoulos_extended_bs=None,
                 T=None,
                 ):
        self.Tcs = Tcs
        self.Pcs = Pcs
        self.Vcs = Vcs
        self.omegas = omegas
        self.N = N = len(Tcs)
        self.scalar = scalar = type(Tcs) is list
        self.T = T

        self.B_model = B_model
        self.cross_B_model = cross_B_model
        
        if cross_B_model_kijs is None:
            if scalar:
                cross_B_model_kijs = [[0.0]*N for i in range(N)]
            else:
                cross_B_model_kijs = zeros((N, N))

        self.cross_B_model_kijs = cross_B_model_kijs
        
        # Parameters specific to `B` model
        if B_model_Meng_as is None:
            if scalar:
                B_model_Meng_as = [[0.0]*N for i in range(N)]
            else:
                B_model_Meng_as = zeros((N, N))
        B_model_Meng_as_pure = [B_model_Meng_as[i][i] for i in range(N)]

        if B_model_Tsonopoulos_extended_as is None:
            if scalar:
                B_model_Tsonopoulos_extended_as = [[0.0]*N for i in range(N)]
            else:
                B_model_Tsonopoulos_extended_as = zeros((N, N))
        B_model_Tsonopoulos_extended_as_pure = [B_model_Tsonopoulos_extended_as[i][i] for i in range(N)]

        if B_model_Tsonopoulos_extended_bs is None:
            if scalar:
                B_model_Tsonopoulos_extended_bs = [[0.0]*N for i in range(N)]
            else:
                B_model_Tsonopoulos_extended_bs = zeros((N, N))
        B_model_Tsonopoulos_extended_bs_pure = [B_model_Tsonopoulos_extended_bs[i][i] for i in range(N)]

        self.B_model_Meng_as = B_model_Meng_as
        self.B_model_Tsonopoulos_extended_as = B_model_Tsonopoulos_extended_as
        self.B_model_Tsonopoulos_extended_bs = B_model_Tsonopoulos_extended_bs
        self.B_model_Meng_as_pure = B_model_Meng_as_pure
        self.B_model_Tsonopoulos_extended_as_pure = B_model_Tsonopoulos_extended_as_pure
        self.B_model_Tsonopoulos_extended_bs_pure = B_model_Tsonopoulos_extended_bs_pure
        
        
        # Cross B coefficients
        self.cross_B_model_Tcijs = Tarakad_Danner_virial_CSP_Tcijs(Tcs, self.cross_B_model_kijs)
        self.cross_B_model_Pcijs = Tarakad_Danner_virial_CSP_Pcijs(Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, Tcijs=self.cross_B_model_Tcijs)
        self.cross_B_model_Vcijs = Lee_Kesler_virial_CSP_Vcijs(Vcs=Vcs)
        self.cross_B_model_omegaijs = Tarakad_Danner_virial_CSP_omegaijs(omegas=omegas)

        self.cross_C_model_Tcijs = self.cross_B_model_Tcijs
        self.cross_C_model_Pcijs = self.cross_B_model_Pcijs
        self.cross_C_model_Vcijs = self.cross_B_model_Vcijs
        self.cross_C_model_omegaijs = self.cross_B_model_omegaijs

        self.C_model = C_model
        self.C_zero = C_model == VIRIAL_C_ZERO
        
    def to(self, T=None):
        r'''Method to construct a new object at a new temperature.

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        obj : VirialCSP
            Object at new temperature, [-]

        Notes
        -----

        Examples
        --------

        '''

        new = self.__class__.__new__(self.__class__)
        new.Tcs = self.Tcs
        new.Pcs = self.Pcs
        new.Vcs = self.Vcs
        new.omegas = self.omegas
        new.N = self.N
        new.scalar = self.scalar
        new.B_model = self.B_model
        # Parameters specific to `B` model
        new.B_model_Meng_as = self.B_model_Meng_as
        new.B_model_Tsonopoulos_extended_as = self.B_model_Tsonopoulos_extended_as
        new.B_model_Tsonopoulos_extended_bs = self.B_model_Tsonopoulos_extended_bs
        new.B_model_Meng_as_pure = self.B_model_Meng_as_pure
        new.B_model_Tsonopoulos_extended_as_pure = self.B_model_Tsonopoulos_extended_as_pure
        new.B_model_Tsonopoulos_extended_bs_pure = self.B_model_Tsonopoulos_extended_bs_pure

        new.cross_B_model = self.cross_B_model
        new.cross_B_model_kijs = self.cross_B_model_kijs
        new.cross_B_model_Tcijs = self.cross_B_model_Tcijs
        new.cross_B_model_Pcijs = self.cross_B_model_Pcijs
        new.cross_B_model_Vcijs = self.cross_B_model_Vcijs
        new.cross_B_model_omegaijs = self.cross_B_model_omegaijs
        new.cross_C_model_Tcijs = self.cross_C_model_Tcijs
        new.cross_C_model_Pcijs = self.cross_C_model_Pcijs
        new.cross_C_model_Vcijs = self.cross_C_model_Vcijs
        new.cross_C_model_omegaijs = self.cross_C_model_omegaijs
        new.C_model = self.C_model
        new.C_zero = self.C_zero
        new.T = T
        return new
        
        
        
    def B_interactions_at_T(self, T):
        N = self.N
        Tcijs, Pcijs, Vcijs, omegaijs = self.cross_B_model_Tcijs, self.cross_B_model_Pcijs, self.cross_B_model_Vcijs, self.cross_B_model_omegaijs
        if self.scalar:
            Bs = [[0.0]*N for _ in range(N)]
            dB_dTs = [[0.0]*N for _ in range(N)]
            d2B_dT2s = [[0.0]*N for _ in range(N)]
            d3B_dT3s = [[0.0]*N for _ in range(N)]
        else:
            Bs = zeros((N, N))
            dB_dTs = zeros((N, N))
            d2B_dT2s = zeros((N, N))
            d3B_dT3s = zeros((N, N))
        

        if self.B_model == VIRIAL_B_ZERO:
            return Bs, dB_dTs, d2B_dT2s, d3B_dT3s
        elif self.B_model == VIRIAL_B_PITZER_CURL:
            Bs_interactions, dB_dTs_interactions, d2B_dT2s_interactions, d3B_dT3s_interactions = BVirial_Pitzer_Curl_mat(T=T, Tcs=Tcijs, Pcs=Pcijs, omegas=omegaijs,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_ABBOTT:
            Bs_interactions, dB_dTs_interactions, d2B_dT2s_interactions, d3B_dT3s_interactions = BVirial_Abbott_mat(T=T, Tcs=Tcijs, Pcs=Pcijs, omegas=omegaijs,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_TSONOPOULOS:
            Bs_interactions, dB_dTs_interactions, d2B_dT2s_interactions, d3B_dT3s_interactions = BVirial_Tsonopoulos_mat(T=T, Tcs=Tcijs, Pcs=Pcijs, omegas=omegaijs,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_TSONOPOULOS_EXTENDED:
            Bs_interactions, dB_dTs_interactions, d2B_dT2s_interactions, d3B_dT3s_interactions = BVirial_Tsonopoulos_extended_mat(T=T, Tcs=Tcijs, Pcs=Pcijs, omegas=omegaijs,
                                                                                                  ais=self.B_model_Tsonopoulos_extended_as,
                                                                                                  bs=self.B_model_Tsonopoulos_extended_bs,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_OCONNELL_PRAUSNITZ:
            Bs_interactions, dB_dTs_interactions, d2B_dT2s_interactions, d3B_dT3s_interactions = BVirial_Oconnell_Prausnitz_mat(T=T, Tcs=Tcijs, Pcs=Pcijs, omegas=omegaijs,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_XIANG:
            Bs_interactions, dB_dTs_interactions, d2B_dT2s_interactions, d3B_dT3s_interactions = BVirial_Xiang_mat(T=T, Tcs=Tcijs, Pcs=Pcijs, Vcs=Vcijs, omegas=omegaijs,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_MENG:
            Bs_interactions, dB_dTs_interactions, d2B_dT2s_interactions, d3B_dT3s_interactions = BVirial_Meng_mat(T=T, Tcs=Tcijs, Pcs=Pcijs, omegas=omegaijs,
                                                                                                  ais=self.B_model_Meng_as,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)

        return Bs_interactions, dB_dTs_interactions, d2B_dT2s_interactions, d3B_dT3s_interactions
    
    
    def _set_B_and_der_interactions(self):
        Bs_interactions, dB_dTs_interactions, d2B_dT2s_interactions, d3B_dT3s_interactions = self.B_interactions_at_T(self.T)
        
        self.Bs_interactions = Bs_interactions
        self.dB_dTs_interactions = dB_dTs_interactions
        self.d2B_dT2s_interactions = d2B_dT2s_interactions
        self.d3B_dT3s_interactions = d3B_dT3s_interactions
        self.cross_B_calculated = True

    
    def B_pures_at_T(self, T):
        N = self.N
        Tcs, Pcs, Vcs, omegas = self.Tcs, self.Pcs, self.Vcs, self.omegas
        if self.scalar:
            Bs = [0.0]*N
            dB_dTs = [0.0]*N
            d2B_dT2s = [0.0]*N
            d3B_dT3s = [0.0]*N
        else:
            Bs = zeros(N)
            dB_dTs = zeros(N)
            d2B_dT2s = zeros(N)
            d3B_dT3s = zeros(N)

        if self.B_model == VIRIAL_B_ZERO:
            return Bs, dB_dTs, d2B_dT2s, d3B_dT3s
        elif self.B_model == VIRIAL_B_PITZER_CURL:
            Bs_pure, dB_dTs_pure, d2B_dT2s_pure, d3B_dT3s_pure = BVirial_Pitzer_Curl_vec(T=T, Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_ABBOTT:
            Bs_pure, dB_dTs_pure, d2B_dT2s_pure, d3B_dT3s_pure = BVirial_Abbott_vec(T=T, Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_TSONOPOULOS:
            Bs_pure, dB_dTs_pure, d2B_dT2s_pure, d3B_dT3s_pure = BVirial_Tsonopoulos_vec(T=T, Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_TSONOPOULOS_EXTENDED:
            Bs_pure, dB_dTs_pure, d2B_dT2s_pure, d3B_dT3s_pure = BVirial_Tsonopoulos_extended_vec(T=T, Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                                                                                  ais=self.B_model_Tsonopoulos_extended_as_pure,
                                                                                                  bs=self.B_model_Tsonopoulos_extended_bs_pure,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_OCONNELL_PRAUSNITZ:
            Bs_pure, dB_dTs_pure, d2B_dT2s_pure, d3B_dT3s_pure = BVirial_Oconnell_Prausnitz_vec(T=T, Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_XIANG:
            Bs_pure, dB_dTs_pure, d2B_dT2s_pure, d3B_dT3s_pure = BVirial_Xiang_vec(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)
        elif self.B_model == VIRIAL_B_MENG:
            Bs_pure, dB_dTs_pure, d2B_dT2s_pure, d3B_dT3s_pure = BVirial_Meng_vec(T=T, Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                                                                                  ais=self.B_model_Meng_as_pure,
                                                                                   Bs=Bs, dB_dTs=dB_dTs, d2B_dT2s=d2B_dT2s, d3B_dT3s=d3B_dT3s)

        return Bs_pure, dB_dTs_pure, d2B_dT2s_pure, d3B_dT3s_pure
    
    def _set_B_and_der_pure(self):
        Bs_pure, dB_dTs_pure, d2B_dT2s_pure, d3B_dT3s_pure = self.B_pures_at_T(self.T)
        
        self.Bs_pure = Bs_pure
        self.dB_dTs_pure = dB_dTs_pure
        self.d2B_dT2s_pure = d2B_dT2s_pure
        self.d3B_dT3s_pure = d3B_dT3s_pure
        self.pure_B_calculated = True

    def B_pures(self):
        r'''Method to calculate and return the pure component virial coefficients
        at the specified temperature.
        
        Returns
        -------
        B_pures : list[float]
            Second `B` virial coefficients, [m^3/mol]
        '''
        if not self.pure_B_calculated:
            self._set_B_and_der_pure()
        return self.Bs_pure
        
    def dB_dT_pures(self):
        r'''Method to calculate and return the first temperature derivative of 
        pure component virial coefficients at the specified temperature.
        
        Returns
        -------
        dB_dT_pures : list[float]
            Second temperature derivative of second `B` virial coefficients, 
            [m^3/(mol*K)]
        '''
        if not self.pure_B_calculated:
            self._set_B_and_der_pure()
        return self.dB_dTs_pure

    def d2B_dT2_pures(self):
        r'''Method to calculate and return the second temperature derivative of 
        pure component virial coefficients at the specified temperature.
        
        Returns
        -------
        d2B_dT2_pures : list[float]
            Second temperature derivative of second `B` virial coefficients, 
            [m^3/(mol*K^2)]
        '''
        if not self.pure_B_calculated:
            self._set_B_and_der_pure()
        return self.d2B_dT2s_pure

    def d3B_dT3_pures(self):
        r'''Method to calculate and return the third temperature derivative of 
        pure component virial coefficients at the specified temperature.
        
        Returns
        -------
        d3B_dT3_pures : list[float]
            Third temperature derivative of second `B` virial coefficients, 
            [m^3/(mol*K^3)]
        '''
        if not self.pure_B_calculated:
            self._set_B_and_der_pure()
        return self.d3B_dT3s_pure
    
    def B_interactions(self):
        r'''Method to calculate and return the matrix of interaction component
        virial coefficients at the specified temperature.
        
        Returns
        -------
        B_interactions : list[list[float]]
            Second `B` virial coefficients interaction matrix, [m^3/mol]
        '''
        if not self.cross_B_calculated:
            self._set_B_and_der_interactions()
        return self.Bs_interactions
        
    def dB_dT_interactions(self):
        r'''Method to calculate and return the first temperature derivative of 
        the `B` virial interaction coefficients at the specified temperature.
        
        Returns
        -------
        dB_dT_interactions : list[list[float]]
            Second temperature derivative of second `B` virial interaction 
            coefficients, [m^3/(mol*K)]
        '''
        if not self.cross_B_calculated:
            self._set_B_and_der_interactions()
        return self.dB_dTs_interactions
    
    def d2B_dT2_interactions(self):
        r'''Method to calculate and return the second temperature derivative of 
        the `B` virial interaction coefficients at the specified temperature.
        
        Returns
        -------
        d2B_dT2_interactions : list[list[float]]
            Second temperature derivative of second `B` virial interaction 
            coefficients, [m^3/(mol*K^2)]
        '''
        if not self.cross_B_calculated:
            self._set_B_and_der_interactions()
        return self.d2B_dT2s_interactions
        
    def d3B_dT3_interactions(self):
        r'''Method to calculate and return the third temperature derivative of 
        the `B` virial interaction coefficients at the specified temperature.
        
        Returns
        -------
        d3B_dT3_interactions : list[list[float]]
            Third temperature derivative of second `B` virial interaction 
            coefficients, [m^3/(mol*K^3)]
        '''
        if not self.cross_B_calculated:
            self._set_B_and_der_interactions()
        return self.d3B_dT3s_interactions
    
    def C_interactions_at_T(self, T):
        N = self.N
        Tcijs, Pcijs, Vcijs, omegaijs = self.cross_C_model_Tcijs, self.cross_C_model_Pcijs, self.cross_C_model_Vcijs, self.cross_C_model_omegaijs
        if self.scalar:
            Cs = [[0.0]*N for _ in range(N)]
            dC_dTs = [[0.0]*N for _ in range(N)]
            d2C_dT2s = [[0.0]*N for _ in range(N)]
            d3C_dT3s = [[0.0]*N for _ in range(N)]
        else:
            Cs = zeros((N, N))
            dC_dTs = zeros((N, N))
            d2C_dT2s = zeros((N, N))
            d3C_dT3s = zeros((N, N))
            
        if self.C_model == VIRIAL_C_ZERO:
            return Cs, dC_dTs, d2C_dT2s, d3C_dT3s
            
        elif self.C_model == VIRIAL_C_XIANG:
            Cs_interactions, dC_dTs_interactions, d2C_dT2s_interactions, d3C_dT3s_interactions = CVirial_Liu_Xiang_mat(T=T, Tcs=Tcijs, Pcs=Pcijs, Vcs=Vcijs, omegas=omegaijs,
                                                                                   Cs=Cs, dC_dTs=dC_dTs, d2C_dT2s=d2C_dT2s, d3C_dT3s=d3C_dT3s)
        elif self.C_model == VIRIAL_C_ORBEY_VERA:
            Cs_interactions, dC_dTs_interactions, d2C_dT2s_interactions, d3C_dT3s_interactions = CVirial_Orbey_Vera_mat(T=T, Tcs=Tcijs, Pcs=Pcijs, omegas=omegaijs,
                                                                                   Cs=Cs, dC_dTs=dC_dTs, d2C_dT2s=d2C_dT2s, d3C_dT3s=d3C_dT3s)        
        
        return Cs_interactions, dC_dTs_interactions, d2C_dT2s_interactions, d3C_dT3s_interactions

    def C_pures_at_T(self, T):
        N = self.N
        Tcs, Pcs, Vcs, omegas = self.Tcs, self.Pcs, self.Vcs, self.omegas
        if self.scalar:
            Cs = [0.0]*N
            dC_dTs = [0.0]*N
            d2C_dT2s = [0.0]*N
            d3C_dT3s = [0.0]*N
        else:
            Cs = zeros(N)
            dC_dTs = zeros(N)
            d2C_dT2s = zeros(N)
            d3C_dT3s = zeros(N)

        if self.C_model == VIRIAL_C_ZERO:
            return Cs, dC_dTs, d2C_dT2s, d3C_dT3s
        elif self.C_model == VIRIAL_C_XIANG:
            Cs_pure, dC_dTs_pure, d2C_dT2s_pure, d3C_dT3s_pure = CVirial_Liu_Xiang_vec(T=T, Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas,
                                                                                   Cs=Cs, dC_dTs=dC_dTs, d2C_dT2s=d2C_dT2s, d3C_dT3s=d3C_dT3s)
        elif self.C_model == VIRIAL_C_ORBEY_VERA:
            Cs_pure, dC_dTs_pure, d2C_dT2s_pure, d3C_dT3s_pure = CVirial_Orbey_Vera_vec(T=T, Tcs=Tcs, Pcs=Pcs, omegas=omegas,
                                                                                   Cs=Cs, dC_dTs=dC_dTs, d2C_dT2s=d2C_dT2s, d3C_dT3s=d3C_dT3s)
        return Cs_pure, dC_dTs_pure, d2C_dT2s_pure, d3C_dT3s_pure
    

    def _set_C_and_der_pure(self):
        Cs_pure, dC_dTs_pure, d2C_dT2s_pure, d3C_dT3s_pure = self.C_pures_at_T(self.T)
        
        self.Cs_pure = Cs_pure
        self.dC_dTs_pure = dC_dTs_pure
        self.d2C_dT2s_pure = d2C_dT2s_pure
        self.d3C_dT3s_pure = d3C_dT3s_pure
        self.pure_C_calculated = True

    def _set_C_and_der_interactions(self):
        Cs_interactions, dC_dTs_interactions, d2C_dT2s_interactions, d3C_dT3s_interactions = self.C_interactions_at_T(self.T)
        
        self.Cs_interactions = Cs_interactions
        self.dC_dTs_interactions = dC_dTs_interactions
        self.d2C_dT2s_interactions = d2C_dT2s_interactions
        self.d3C_dT3s_interactions = d3C_dT3s_interactions
        self.cross_C_calculated = True

    def C_pures(self):
        r'''Method to calculate and return the pure component third virial 
        coefficients  at the specified temperature.
        
        Returns
        -------
        C_pures : list[float]
            Third `C` virial coefficients, [m^6/mol^2]
        '''
        if not self.pure_C_calculated:
            self._set_C_and_der_pure()
        return self.Cs_pure
        
    def dC_dT_pures(self):
        r'''Method to calculate and return the first temperature derivative of
        pure component third virial coefficients at the specified temperature.
        
        Returns
        -------
        dC_dT_pures : list[float]
            First temperature derivative of third `C` virial coefficients, 
            [m^6/(mol^2*K)]
        '''
        if not self.pure_C_calculated:
            self._set_C_and_der_pure()
        return self.dC_dTs_pure

    def d2C_dT2_pures(self):
        r'''Method to calculate and return the second temperature derivative of
        pure component third virial coefficients at the specified temperature.
        
        Returns
        -------
        d2C_dT2_pures : list[float]
            Second temperature derivative of third `C` virial coefficients, 
            [m^6/(mol^2*K^2)]
        '''
        if not self.pure_C_calculated:
            self._set_C_and_der_pure()
        return self.d2C_dT2s_pure

    def d3C_dT3_pures(self):
        r'''Method to calculate and return the third temperature derivative of
        pure component third virial coefficients at the specified temperature.
        
        Returns
        -------
        d3C_dT3_pures : list[float]
            Third temperature derivative of third `C` virial coefficients, 
            [m^6/(mol^2*K^3)]
        '''
        if not self.pure_C_calculated:
            self._set_C_and_der_pure()
        return self.d3C_dT3s_pure

    def C_interactions(self):
        r'''Method to calculate and return the matrix of interaction third  
        virial coefficients at the specified temperature.
        
        Returns
        -------
        C_interactions : list[list[float]]
            Interaction third `C` virial coefficients, [m^6/mol^2]
        '''
        if not self.cross_C_calculated:
            self._set_C_and_der_interactions()
        return self.Cs_interactions
    
    def dC_dT_interactions(self):
        r'''Method to calculate and return the matrix of first temperature 
        derivatives of interaction third virial coefficients at the specified 
        temperature.
        
        Returns
        -------
        dC_dT_interactions : list[list[float]]
            Interaction first temperature derivatives of third `C` virial 
            coefficients, [m^6/(mol^2*K)]
        '''
        if not self.cross_C_calculated:
            self._set_C_and_der_interactions()
        return self.dC_dTs_interactions
    
    def d2C_dT2_interactions(self):
        r'''Method to calculate and return the matrix of second temperature 
        derivatives of interaction third virial coefficients at the specified 
        temperature.
        
        Returns
        -------
        d2C_dT2_interactions : list[list[float]]
            Interaction second temperature derivatives of third `C` virial 
            coefficients, [m^6/(mol^2*K^2)]
        '''
        if not self.cross_C_calculated:
            self._set_C_and_der_interactions()
        return self.d2C_dT2s_interactions
    
    def d3C_dT3_interactions(self):
        r'''Method to calculate and return the matrix of third temperature 
        derivatives of interaction third virial coefficients at the specified 
        temperature.
        
        Returns
        -------
        d3C_dT3_interactions : list[list[float]]
            Interaction third temperature derivatives of third `C` virial 
            coefficients, [m^6/(mol^2*K^2)]
        '''
        if not self.cross_C_calculated:
            self._set_C_and_der_interactions()
        return self.d3C_dT3s_interactions
    


class VirialGas(Phase):
    r'''Class for representing a real gas defined by the virial equation of
    state (density form), as a phase object. The equation includes the `B`
    and `C` coefficients but not further coefficients as they cannot be 
    accurately estimated. Only limited experimental data for third virial
    coefficients is available.
    
    This model is generic, and allows any source of virial coefficients to be 
    plugged it, so long as it provides the right methods. See :obj:`VirialCSP`.

    .. math::
        Z = \frac{PV}{RT} = 1 + \frac{B}{V} + \frac{C}{V^2} 
    
    Parameters
    ----------
    model : object
        Object which provides pure component and interaction second and third
        virial coefficients; :obj:`VirialCSP`, [-]
    HeatCapacityGases : list[HeatCapacityGas]
        Objects proiding pure-component heat capacity correlations, [-]
    Hfs : list[float]
        Molar ideal-gas standard heats of formation at 298.15 K and 1 atm,
        [J/mol]
    Gfs : list[float]
        Molar ideal-gas standard Gibbs energies of formation at 298.15 K and
        1 atm, [J/mol]
    T : float, optional
        Temperature, [K]
    P : float, optional
        Pressure, [Pa]
    zs : list[float], optional
        Mole fractions of each component, [-]
    cross_B_model : str, optional
        The method used to combine the pure and/or interaction second `B` virial 
        coefficients into a single `B` coefficient.
        
        * 'linear': :math:`B = \sum_i y_i B_i`
        * 'theory': :math:` B = \sum_i \sum_j y_i y_j B(T)`
    cross_C_model : str, optional
        The method used to combine the pure and/or interaction third `C` virial 
        coefficients into a single `C` coefficient.
        
        * 'linear': :math:`C = \sum_i y_i C_i`; this is considerably faster 
        * 'Orentlicher-Prausnitz': :math:`C = \sum_i \sum_j \sum_k y_i y_j y_k C_{ijk}(T)`
          where :math:`C_{ijk} = \left(C_{ij}C_{jk}C_{ik}\right)^{1/3}`

    Examples
    --------
    T-P initialization for nitrogen, oxygen, and argon, using Poling's polynomial heat
    capacities:
    
    >>> Tcs=[126.2, 154.58, 150.8]
    >>> Pcs=[3394387.5, 5042945.25, 4873732.5]
    >>> Vcs=[8.95e-05, 7.34e-05, 7.49e-05]
    >>> omegas=[0.04, 0.021, -0.004]
    >>> model = VirialCSP(Tcs=Tcs, Pcs=Pcs, Vcs=Vcs, omegas=omegas, B_model='VIRIAL_B_PITZER_CURL', cross_B_model='Tarakad-Danner', C_model='VIRIAL_C_ORBEY_VERA')
    >>> HeatCapacityGases = [HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*1.79e-12, R*-6e-09, R*6.58e-06, R*-0.001794, R*3.63])),
    ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [R*-9.9e-13, R*1.57e-09, R*7e-08, R*-0.000261, R*3.539])),
    ...                      HeatCapacityGas(poly_fit=(50.0, 1000.0, [0,0,0,0, R*2.5]))]
    >>> phase = VirialGas(model=model, T=300, P=1e5, zs=[.78, .21, .01], HeatCapacityGases=HeatCapacityGases, cross_B_model='theory', cross_C_model='Orentlicher-Prausnitz')
    >>> phase.V(), phase.isothermal_compressibility(), phase.speed_of_sound()
    (0.02493687, 1.00025907e-05, 59.081947)
    >>> phase
    VirialGas(model=VirialCSP(Tcs=[126.2, 154.58, 150.8], Pcs=[3394387.5, 5042945.25, 4873732.5], Vcs=[8.95e-05, 7.34e-05, 7.49e-05], omegas=[0.04, 0.021, -0.004], B_model='VIRIAL_B_PITZER_CURL', cross_B_model='Tarakad-Danner', C_model='VIRIAL_C_ORBEY_VERA', T=300), HeatCapacityGases=[HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [1.48828880864943e-11, -4.9886775708919434e-08, 5.4709164027448316e-05, -0.014916145936966912, 30.18149930389626])), HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [-8.231317991971707e-12, 1.3053706310500586e-08, 5.820123832707268e-07, -0.0021700747433379955, 29.424883205644317])), HeatCapacityGas(extrapolation="linear", method="POLY_FIT", poly_fit=(50.0, 1000.0, [0, 0, 0, 0, 20.7861565453831]))], cross_B_model='theory', cross_C_model='Orentlicher-Prausnitz', T=300, P=100000.0, zs=[0.78, 0.21, 0.01])

    '''
    phase = 'g'
    force_phase = 'g'
    is_gas = True
    is_liquid = False
    ideal_gas_basis = True
    pure_references = ('HeatCapacityGases',)
    pure_reference_types = (HeatCapacityGas, )

    model_attributes = ('Hfs', 'Gfs', 'Sfs', 'model',
                        'cross_B_model', 'cross_C_model') + pure_references

    def __init__(self, model, HeatCapacityGases=None, Hfs=None, Gfs=None, 
                 T=None, P=None, zs=None,
                 cross_B_model='theory', cross_C_model='Orentlicher-Prausnitz'):
        self.model = model.to(T=T)
        self.HeatCapacityGases = HeatCapacityGases
        self.Hfs = Hfs
        self.Gfs = Gfs
        
        if cross_B_model not in ('theory', 'linear'):
            raise ValueError("Unsupported value for `cross_B_model`")
        if cross_C_model not in ('Orentlicher-Prausnitz', 'linear'):
            raise ValueError("Unsupported value for `cross_C_model`")
        self.cross_B_model = cross_B_model
        self.cross_C_model = cross_C_model
        
        # Store the virial cross model as a boolean
        # It is likely additional `C` models will be published, the current one is emperical
        self.cross_B_coefficients = cross_B_model == 'theory'
        self.cross_C_coefficients = cross_C_model == 'Orentlicher-Prausnitz'
        
        if Hfs is not None and Gfs is not None and None not in Hfs and None not in Gfs:
            self.Sfs = [(Hfi - Gfi)/298.15 for Hfi, Gfi in zip(Hfs, Gfs)]
        else:
            self.Sfs = None

        for i in (zs, HeatCapacityGases, Hfs, Gfs):
            if i is not None:
                self.N = len(i)
                break
        if zs is not None:
            self.zs = zs
            self.scalar = scalar = type(zs) is list
        if T is not None:
            self.T = T
            self.model.T = T
        if P is not None:
            self.P = P
        if T is not None and P is not None and zs is not None:
            Z = Z_from_virial_density_form(T, P, self.B(), self.C())
            self._V = Z*self.R*T/P
        
    def __repr__(self):
        r'''Method to create a string representation of the phase object, with
        the goal of making it easy to obtain standalone code which reproduces
        the current state of the phase. This is extremely helpful in creating
        new test cases.

        Returns
        -------
        recreation : str
            String which is valid Python and recreates the current state of
            the object if ran, [-]

        '''
        try:
            Cpgs = ', '.join(str(o) for o in self.HeatCapacityGases)
        except:
            Cpgs = ''
        base = '%s(model=%s, HeatCapacityGases=[%s], '  %(self.__class__.__name__, self.model, Cpgs)
        for s in ('cross_B_model', 'cross_C_model', 'Hfs', 'Gfs', 'T', 'P', 'zs'):
            if hasattr(self, s) and getattr(self, s) is not None:
                val = getattr(self, s)
                if type(val) is str:
                    val = f"'{val}'"
                base += '%s=%s, ' %(s, val)
        if base[-2:] == ', ':
            base = base[:-2]
        base += ')'
        return base


        
    def V(self):
        r'''Method to calculate and return the molar volume.
        
        Returns
        -------
        V : float
            Molar volume [m^3/mol]
        '''
        return self._V
    
    def dV_dzs(self):
        r'''Method to calculate and return the first mole fraction derivatives 
        of the molar volume. See :obj:`chemicals.virial.dV_dzs_virial` for 
        further details.
        
        Returns
        -------
        dV_dzs : list[float]
            First mole fraction derivatives of molar volume 
            [m^3/mol]
        '''
        try:
            return self._dV_dzs
        except:
            pass
        dB_dzs = self.dB_dzs()
        dC_dzs = self.dC_dzs()
        B = self.B()
        C = self.C()
        V = self._V
        N = self.N
        if self.scalar:
            dV_dzs = [0.0]*N
        else:
            dV_dzs = zeros(N)
        self._dV_dzs = dV_dzs_virial(B=B, C=C, V=V, dB_dzs=dB_dzs, dC_dzs=dC_dzs, dV_dzs=dV_dzs)
        return dV_dzs

    def d2V_dzizjs(self):
        r'''Method to calculate and return the second mole fraction derivatives 
        of the molar volume. See :obj:`chemicals.virial.d2V_dzizjs_virial` for 
        further details.
        
        Returns
        -------
        d2V_dzizjs : list[float]
            Second mole fraction derivatives of molar volume 
            [m^3/mol]
        '''
        try:
            return self._d2V_dzizjs
        except:
            pass
        dB_dzs = self.dB_dzs()
        dC_dzs = self.dC_dzs()
        d2B_dzizjs = self.d2B_dzizjs()
        d2C_dzizjs = self.d2C_dzizjs()
        B = self.B()
        C = self.C()
        V = self._V
        dV_dzs = self.dV_dzs()
        N = self.N
        if self.scalar:
            d2V_dzizjs = [[0.0]*N for _ in range(N)]
        else:
            d2V_dzizjs = zeros((N,N))
        self._d2V_dzizjs = d2V_dzizjs_virial(B=B, C=C, V=V, dB_dzs=dB_dzs, dC_dzs=dC_dzs, dV_dzs=dV_dzs,
                                             d2B_dzizjs=d2B_dzizjs, d2C_dzizjs=d2C_dzizjs, d2V_dzizjs=d2V_dzizjs)
        return d2V_dzizjs
    
    def dG_dep_dzs(self):
        r'''Method to calculate and return the first mole fraction derivatives 
        of the departure Gibbs energy. 
        
        Returns
        -------
        dG_dep_dzs : list[float]
            First mole fraction derivatives of departure Gibbs energy 
            [J/mol]
        '''
        try:
            return self._dG_dep_dzs
        except:
            pass
        T = self.T
        dB_dzs = self.dB_dzs()
        dB_dT = self.dB_dT()
        dC_dT = self.dC_dT()
        dC_dzs = self.dC_dzs()
        dV_dzs = self.dV_dzs()
        d2C_dTdzs = self.d2C_dTdzs()
        d2B_dTdzs = self.d2B_dTdzs()
        B = self.B()
        C = self.C()
        V = self._V
        N = self.N
        if self.scalar:
            dG_dep_dzs = [0.0]*N
        else:
            dG_dep_dzs = zeros(N)
        for i in range(N):
            x0 = V
            x1 = x0**2
            x2 = 1/x1
            x3 = C
            x4 = dC_dzs[i]#Derivative(x3, z1)
            x5 = B
            x6 = 2*T
            x7 = dV_dzs[i]#Derivative(x0, z1)
            x8 = 2*x7
            x9 = T*dB_dT#Derivative(x5, T)
            x10 = x0*x5 + x1 + x3
            x11 = log(x10*x2)
            x12 = x0*x8
            x13 = 1/x0
            dG_dep_dzs[i] = (R*T*x2*(T*d2C_dTdzs[i] + x0*x6*d2B_dTdzs[i]+ x1*(-x0*dB_dzs[i]
                            + 2*x10*x13*x7 - x12 - x4 - x5*x7)/x10 - x11*x12 - x13*x7*(4*x0*x9 - 2*x1*x11 - x3 + x6*dC_dT)
                                    - x4/2 + x8*x9))
        # self._dG_dep_dzs = dG_dep_dzs_virial(B=B, C=C, V=V, dB_dzs=dB_dzs, dC_dzs=dC_dzs, dG_dep_dzs=dG_dep_dzs)
        return dG_dep_dzs
    
    def dG_dep_dns(self):
        try:
            return self._dG_dep_dns
        except:
            pass
        N = self.N
        if self.scalar:
            dG_dep_dns = [0.0]*N
        else:
            dG_dep_dns = zeros(N)
        self._dG_dep_dns = dG_dep_dns = dxs_to_dns(dxs=self.dG_dep_dzs(), xs=self.zs, dns=dG_dep_dns)
        return dG_dep_dns 

    
        
    def dnG_dep_dns(self):
        try:
            return self._dnG_dep_dns
        except:
            pass
        N = self.N
        if self.scalar:
            dnG_dep_dns = [0.0]*N
        else:
            dnG_dep_dns = zeros(N)
        
        self._dnG_dep_dns = dnG_dep_dns = dxs_to_dn_partials(dxs=self.dG_dep_dzs(), xs=self.zs, F=self.G_dep(), partial_properties=dnG_dep_dns)
        return dnG_dep_dns 

    def lnphi(self):
        return self.G_dep()/(R*self.T)

    def lnphis(self):
        r'''Method to calculate and return the log fugacity coefficients of
        the phase.
        
        Returns
        -------
        lnphis : list[float]
            Log fugacity coefficients, [-] 
        '''
        # working!
        zs = self.zs
        T = self.T
        RT_inv = 1.0/(R*T)
        lnphi = self.G_dep()*RT_inv
        dG_dep_dns = self.dG_dep_dns()
        if self.scalar:
            dG_dep_dns_RT = [v*RT_inv for v in dG_dep_dns]
            
        else:
            dG_dep_dns_RT = RT_inv*dG_dep_dns
        
        log_phis = dns_to_dn_partials(dG_dep_dns_RT, lnphi)
        return log_phis if self.scalar else array(log_phis)

    
    
    def dP_dT(self):
        r'''Method to calculate and return the first derivative of pressure
        with respect to temperature.
        
        .. math::
            \left(\frac{\partial P}{\partial T}\right)_{V} = \frac{R \left(T
            \left(V \frac{d}{d T} B{\left(T \right)} + \frac{d}{d T} C{\left(T
            \right)}\right) + V^{2} + V B{\left(T \right)} + C{\left(T \right)}
            \right)}{V^{3}}

        Returns
        -------
        dP_dT : float
            First derivative of pressure with respect to temperature at constant
            volume [Pa/K]
        '''
        try:
            return self._dP_dT
        except:
            pass
        T, V = self.T, self._V
        self._dP_dT = dP_dT = self.R*(T*(V*self.dB_dT() + self.dC_dT()) + V*(V + self.B()) + self.C())/(V*V*V)
        return dP_dT

    def dP_dV(self):
        r'''Method to calculate and return the first derivative of pressure
        with respect to volume.
        
        .. math::
            \left(\frac{\partial P}{\partial V}\right)_{T} =
            - \frac{R T \left(V^{2} + 2 V B{\left(T \right)} + 3 C{\left(T
            \right)}\right)}{V^{4}}

        Returns
        -------
        dP_dV : float
            First derivative of pressure with respect to volume at constant
            temperature [Pa*mol/(m^3)]
        '''
        try:
            return self._dP_dV
        except:
            pass
        T, V = self.T, self._V
        self._dP_dV = dP_dV = -self.R*T*(V*V + 2.0*V*self.B() + 3.0*self.C())/(V*V*V*V)
        return dP_dV

    def d2P_dTdV(self):
        r'''Method to calculate and return the second derivative of pressure
        with respect to volume and temperature.
        
        .. math::
            \left(\frac{\partial^2 P}{\partial V\partial T}\right)_{T} =
            - \frac{R \left(2 T V \frac{d}{d T} B{\left(T \right)} + 3 T
            \frac{d}{d T} C{\left(T \right)} + V^{2} + 2 V B{\left(T \right)}
            + 3 C{\left(T \right)}\right)}{V^{4}}

        Returns
        -------
        d2P_dTdV : float
            Second derivative of pressure with respect to volume at and
            temperature [Pa*mol/(m^3*K)]
        '''
        try:
            return self._d2P_dTdV
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V
        self._d2P_dTdV = d2P_dTdV = -self.R*(2.0*T*V*self.dB_dT() + 3.0*T*self.dC_dT()
        + V2 + 2.0*V*self.B() + 3.0*self.C())/(V2*V2)

        return d2P_dTdV

    def d2P_dV2(self):
        r'''Method to calculate and return the second derivative of pressure
        with respect to volume.
        
        .. math::
            \left(\frac{\partial^2 P}{\partial V^2}\right)_{T} =
            \frac{2 R T \left(V^{2} + 3 V B{\left(T \right)}
            + 6 C{\left(T \right)}\right)}{V^{5}}

        Returns
        -------
        d2P_dV2 : float
            Second derivative of pressure with respect to volume at constant
            temperature [Pa*mol^2/(m^6)]
        '''
        try:
            return self._d2P_dV2
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V
        self._d2P_dV2 = d2P_dV2 = 2.0*self.R*T*(V2 + 3.0*V*self.B() + 6.0*self.C())/(V2*V2*V)
        return d2P_dV2

    def d2P_dT2(self):
        r'''Method to calculate and return the second derivative of pressure
        with respect to temperature.
        
        .. math::
            \left(\frac{\partial^2 P}{\partial T^2}\right)_{V} =
            \frac{R \left(T \left(V \frac{d^{2}}{d T^{2}} B{\left(T \right)}
            + \frac{d^{2}}{d T^{2}} C{\left(T \right)}\right) + 2 V \frac{d}{d T}
            B{\left(T \right)} + 2 \frac{d}{d T} C{\left(T \right)}\right)}{V^{3}}

        Returns
        -------
        d2P_dT2 : float
            Second derivative of pressure with respect to temperature at constant
            volume [Pa/K^2]
        '''
        try:
            return self._d2P_dT2
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V
        self._d2P_dT2 = d2P_dT2 = self.R*(T*(V*self.d2B_dT2() + self.d2C_dT2())
                              + 2.0*V*self.dB_dT() + 2.0*self.dC_dT())/(V*V*V)
        return d2P_dT2

    def H_dep(self):
        r'''Method to calculate and return the molar departure enthalpy.
        
        .. math::
           H_{dep} = \frac{R T^{2} \left(2 V \frac{d}{d T} B{\left(T \right)}
           + \frac{d}{d T} C{\left(T \right)}\right)}{2 V^{2}} - R T \left(-1
            + \frac{V^{2} + V B{\left(T \right)} + C{\left(T \right)}}{V^{2}}
            \right)

        Returns
        -------
        H_dep : float
            Departure enthalpy [J/mol]
            
        Notes
        -----
        
        '''
        '''
        from sympy import *
        Z, R, T, V, P = symbols('Z, R, T, V, P')
        B, C = symbols('B, C', cls=Function)
        base =Eq(P*V/(R*T), 1 + B(T)/V + C(T)/V**2)
        P_sln = solve(base, P)[0]
        Z = P_sln*V/(R*T)

        # Two ways to compute H_dep
        Hdep2 = R*T - P_sln*V + integrate(P_sln - T*diff(P_sln, T), (V, oo, V))
        Hdep = -R*T*(Z-1) -integrate(diff(Z, T)/V, (V, oo, V))*R*T**2
        '''
        try:
            return self._H_dep
        except:
            pass

        T, V = self.T, self._V
        V2 = V*V
        RT = self.R*T
        self._H_dep = H_dep = RT*(T*(2.0*V*self.dB_dT() + self.dC_dT())/(2.0*V2)
               - (-1.0 + (V2 + V*self.B() + self.C())/V2))
        return H_dep

    def dH_dep_dT(self):
        r'''Method to calculate and return the first temperature derivative of  
        molar departure enthalpy.
        
        .. math::
           \frac{\partial H_{dep}}{\partial T} = \frac{R \left(2 T^{2} V
           \frac{d^{2}}{d T^{2}} B{\left(T \right)} + T^{2} \frac{d^{2}}{d T^{2}}
           C{\left(T \right)} + 2 T V \frac{d}{d T} B{\left(T \right)}
           - 2 V B{\left(T \right)} - 2 C{\left(T \right)}\right)}{2 V^{2}}

        Returns
        -------
        dH_dep_dT : float
            First temperature derivative of departure enthalpy [J/(mol*K)]
        
        '''
        try:
            return self._dH_dep_dT
        except:
            pass
        T, V = self.T, self._V
        B = self.B()
        C = self.C()
        dB_dT = self.dB_dT()
        d2B_dT2 = self.d2B_dT2()
        d2C_dT2 = self.d2C_dT2()
        self._dH_dep_dT = dH_dep_dT = (self.R*(2.0*T*T*V*d2B_dT2 + T*T*d2C_dT2
            + 2.0*T*V*dB_dT - 2.0*V*B - 2.0*C)/(2.0*V*V))
        return dH_dep_dT
    
    Cp_dep = dH_dep_dT

    def S_dep(self):
        r'''Method to calculate and return the molar departure entropy.
        
        .. math::
           S_{dep} = \frac{R \left(- T \frac{d}{d T} C{\left(T \right)} + 2 V^{2}
           \ln{\left(\frac{V^{2} + V B{\left(T \right)} + C{\left(T \right)}}
           {V^{2}} \right)} - 2 V \left(T \frac{d}{d T} B{\left(T \right)}
            + B{\left(T \right)}\right) - C{\left(T \right)}\right)}{2 V^{2}}

        Returns
        -------
        S_dep : float
            Departure enthalpy [J/(mol*K)]
            
        Notes
        -----
        
        '''
        '''
        dP_dT = diff(P_sln, T)
        S_dep = integrate(dP_dT - R/V, (V, oo, V)) + R*log(Z)

        '''
        try:
            return self._S_dep
        except:
            pass

        T, V = self.T, self._V
        V2 = V*V
        self._S_dep = S_dep = (self.R*(-T*self.dC_dT() + 2*V2*log((V2 + V*self.B() + self.C())/V**2)
        - 2*V*(T*self.dB_dT() + self.B()) - self.C())/(2*V2))
        return S_dep

    def dS_dep_dT(self):
        r'''Method to calculate and return the first temperature derivative of 
        molar departure entropy.
        
        .. math::
           \frac{\partial S_{dep}}{\partial T} = \frac{R \left(2 V^{2} \left(V
           \frac{d}{d T} B{\left(T \right)} + \frac{d}{d T} C{\left(T \right)}
           \right) - \left(V^{2} + V B{\left(T \right)} + C{\left(T \right)}
           \right) \left(T \frac{d^{2}}{d T^{2}} C{\left(T \right)} + 2 V
           \left(T \frac{d^{2}}{d T^{2}} B{\left(T \right)} + 2 \frac{d}{d T}
           B{\left(T \right)}\right) + 2 \frac{d}{d T} C{\left(T \right)}
           \right)\right)}{2 V^{2} \left(V^{2} + V B{\left(T \right)}
           + C{\left(T \right)}\right)}

        Returns
        -------
        dS_dep_dT : float
            First temperature derivative of departure enthalpy [J/(mol*K^2)]
            
        '''
        try:
            return self._dS_dep_dT
        except:
            pass
        T, V = self.T, self._V
        V2 = V*V

        self._dS_dep_dT = dS_dep_dT = (self.R*(2.0*V2*(V*self.dB_dT() + self.dC_dT()) - (V2 + V*self.B() + self.C())*(T*self.d2C_dT2()
        + 2.0*V*(T*self.d2B_dT2() + 2.0*self.dB_dT()) + 2.0*self.dC_dT()))/(2.0*V2*(V2 + V*self.B() + self.C())))
        return dS_dep_dT

    def to_TP_zs(self, T, P, zs):
        new = self.__class__.__new__(self.__class__)
        new.T = T
        new.P = P
        new.zs = zs
        new.N = self.N
        new.scalar = self.scalar
        new.cross_B_coefficients = self.cross_B_coefficients
        new.cross_C_coefficients = self.cross_C_coefficients
        new.cross_B_model = self.cross_B_model
        new.cross_C_model = self.cross_C_model

        new.HeatCapacityGases = self.HeatCapacityGases
        new.model = self.model.to(T)
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        Z = Z_from_virial_density_form(T, P, new.B(), new.C())
        new._V = Z*self.R*T/P
        return new

    def to(self, zs, T=None, P=None, V=None):
        new = self.__class__.__new__(self.__class__)
        new.zs = zs
        new.scalar = self.scalar
        new.N = self.N
        new.cross_B_coefficients = self.cross_B_coefficients
        new.cross_C_coefficients = self.cross_C_coefficients
        new.cross_B_model = self.cross_B_model
        new.cross_C_model = self.cross_C_model

        new.HeatCapacityGases = self.HeatCapacityGases
        new.model = model = self.model.to(T=None)
        new.Hfs = self.Hfs
        new.Gfs = self.Gfs
        new.Sfs = self.Sfs
        if T is not None:
            new.T = T
            new.model.T = T
            if P is not None:
                new.P = P
                Z = Z_from_virial_density_form(T, P, new.B(), new.C())
                new._V = Z*self.R*T/P
            elif V is not None:
                P = new.P = self.R*T*(V*V + V*new.B() + new.C())/(V*V*V)
                new._V = V
        elif P is not None and V is not None:
            new.P = P
            # PV specified, solve for T
            def err(T):
                # Solve for P matching; probably there is a better solution here that does not
                # require the cubic solution but this works for now
                # TODO: instead of using self.to_TP_zs to allow calculating B and C,
                # they should be functional
                new_tmp = self.to_TP_zs(T=T, P=P, zs=zs)
                B = new_tmp.B()
                C = new_tmp.C()
                x2 = V*V + V*B + C
                x3 = self.R/(V*V*V)

                P_err = T*x2*x3 - P
                dP_dT = x3*(T*(V*new_tmp.dB_dT() + new_tmp.dC_dT()) + x2)
                return P_err, dP_dT

            T_ig = P*V/self.R # guess
            T = newton(err, T_ig, fprime=True, xtol=1e-15)
            new.T = T
            new.model.T = T
        else:
            raise ValueError("Two of T, P, or V are needed")

        return new

    def B(self):
        r'''Method to calculate and return the `B` second virial coefficient.
        
        Returns
        -------
        B : float
            Second molar virial coefficient [m^3/mol]
        '''
        try:
            return self._B
        except:
            pass
        N = self.N
        if N == 1:
            self._B = B = self.model.B_pures()[0]
            return B
        zs = self.zs
        if not self.cross_B_coefficients:
            Bs = self.model.B_pures()
            self._B = B = float(mixing_simple(zs, Bs))
            return B

        B_interactions = self.model.B_interactions()
        self._B = B = float(BVirial_mixture(zs, B_interactions))
        return B

    def dB_dT(self):
        r'''Method to calculate and return the first temperature derivative of
        the `B` second virial coefficient.
        
        Returns
        -------
        dB_dT : float
            First temperature derivative of second molar virial coefficient 
            [m^3/(mol*K)]
        '''
        try:
            return self._dB_dT
        except:
            pass
        N = self.N
        if N == 1:
            return self.model.dB_dT_pures()[0]
        zs = self.zs
        if not self.cross_B_coefficients:
            Bs = self.model.dB_dT_pures()
            self._dB_dT = dB_dT = float(mixing_simple(zs, Bs))
            return dB_dT
        dB_dT_interactions = self.model.dB_dT_interactions()
        self._dB_dT = dB_dT = float(BVirial_mixture(zs, dB_dT_interactions))
        return dB_dT

        

    def d2B_dT2(self):
        r'''Method to calculate and return the second temperature derivative of
        the `B` second virial coefficient.
        
        Returns
        -------
        d2B_dT2 : float
            Second temperature derivative of second molar virial coefficient 
            [m^3/(mol*K^2)]
        '''
        try:
            return self._d2B_dT2
        except:
            pass
        N = self.N
        if N == 1:
            return self.model.d2B_dT2_pures()[0]
        zs = self.zs
        if not self.cross_B_coefficients:
            Bs = self.model.d2B_dT2_pures()
            self._d2B_dT2 = d2B_dT2 = float(mixing_simple(zs, Bs))
            return d2B_dT2
        d2B_dT2_interactions = self.model.d2B_dT2_interactions()
        self._d2B_dT2 = d2B_dT2 = float(BVirial_mixture(zs, d2B_dT2_interactions))
        return d2B_dT2

    def d3B_dT3(self):
        r'''Method to calculate and return the third temperature derivative of
        the `B` second virial coefficient.
        
        Returns
        -------
        d3B_dT3 : float
            Third temperature derivative of second molar virial coefficient 
            [m^3/(mol*K^3)]
        '''
        try:
            return self._d3B_dT3
        except:
            pass
        N = self.N
        if N == 1:
            return self.model.d3B_dT3_pures()[0]
        zs = self.zs
        
        if not self.cross_B_coefficients:
            Bs = self.model.d3B_dT3_pures()
            self._d3B_dT3 = d3B_dT3 = float(mixing_simple(zs, Bs))
            return d3B_dT3
        d3B_dT3_interactions = self.model.d3B_dT3_interactions()
        self._d3B_dT3 = d3B_dT3 = float(BVirial_mixture(zs, d3B_dT3_interactions))
        return d3B_dT3


    def C(self):
        r'''Method to calculate and return the `C` third virial coefficient.
        
        Returns
        -------
        C : float
            Third molar virial coefficient [m^6/mol^2]
        '''
        try:
            return self._C
        except:
            pass
        T = self.T
        zs = self.zs
        N = self.N
        if self.model.C_zero:
            self._C = C = 0.0
            return C
        if not self.cross_C_coefficients:
            Cs = self.model.C_pures()
            self._C = C = float(mixing_simple(zs, Cs))
            return C
        else:
            Cijs = self.model.C_interactions()
            self._C = C = float(CVirial_mixture_Orentlicher_Prausnitz(zs, Cijs))
        return C

    def dC_dT(self):
        r'''Method to calculate and return the first temperature derivative of 
        the `C` third virial coefficient.
        
        Returns
        -------
        dC_dT : float
            First temperature derivative of third molar virial coefficient
            [m^6/(mol^2*K)]
        '''
        try:
            return self._dC_dT
        except:
            pass
        T = self.T
        zs = self.zs
        if self.model.C_zero:
            self._dC_dT = dC_dT = 0.0
            return dC_dT

        if not self.cross_C_coefficients:
            dC_dTs = self.model.dC_dT_pures()
            self._dC_dT = dC_dT = float(mixing_simple(zs, dC_dTs))
            return dC_dT

        Cijs = self.model.C_interactions()
        dCijs = self.model.dC_dT_interactions()
        # TODO
        '''
        from sympy import *
        Cij, Cik, Cjk = symbols('Cij, Cik, Cjk', cls=Function)
        T = symbols('T')
        # The derivative of this is messy
        expr = (Cij(T)*Cik(T)*Cjk(T))**Rational('1/3')
        # diff(expr, T, 3)
        '''
        self._dC_dT = dC_dT = float(dCVirial_mixture_dT_Orentlicher_Prausnitz(zs, Cijs, dCijs))
        return dC_dT

    def d2C_dT2(self):
        r'''Method to calculate and return the second temperature derivative of 
        the `C` third virial coefficient.
        
        Returns
        -------
        d2C_dT2 : float
            Second temperature derivative of third molar virial coefficient
            [m^6/(mol^2*K^2)]
        '''
        try:
            return self._d2C_dT2
        except:
            pass
        T = self.T
        zs = self.zs
        if self.model.C_zero:
            self._d2C_dT2 = d2C_dT2 = 0.0
            return d2C_dT2

        if not self.cross_C_coefficients:
            d2C_dT2s = self.model.d2C_dT2_pures()
            self._d2C_dT2 = d2C_dT2 = float(mixing_simple(zs, d2C_dT2s))
            return d2C_dT2

        Cijs = self.model.C_interactions()
        dCijs = self.model.dC_dT_interactions()
        d2C_dT2ijs = self.model.d2C_dT2_interactions()
        N = self.N
        self._d2C_dT2 = d2C_dT2 = float(d2CVirial_mixture_dT2_Orentlicher_Prausnitz(zs, Cijs, dCijs, d2C_dT2ijs))
        return d2C_dT2
    
    def d3C_dT3(self):
        r'''Method to calculate and return the third temperature derivative of 
        the `C` third virial coefficient.
        
        Returns
        -------
        d3C_dT3 : float
            Second temperature derivative of third molar virial coefficient
            [m^6/(mol^2*K^3)]
        '''
        try:
            return self._d3C_dT3
        except:
            pass
        T = self.T
        zs = self.zs
        if self.model.C_zero:
            self._d3C_dT3 = d3C_dT3 = 0.0
            return d3C_dT3
        if not self.cross_C_coefficients:
            d3C_dT3s = self.model.d3C_dT3_pures()
            self._d3C_dT3 = d3C_dT3 = float(mixing_simple(zs, d3C_dT3s))
            return d3C_dT3

        Cijs = self.model.C_interactions()
        dCijs = self.model.dC_dT_interactions()
        d2C_dT2ijs = self.model.d2C_dT2_interactions()
        d3C_dT3ijs = self.model.d3C_dT3_interactions()
        N = self.N
        self._d3C_dT3 = d3C_dT3 = float(d3CVirial_mixture_dT3_Orentlicher_Prausnitz(zs, Cijs, dCijs, d2C_dT2ijs, d3C_dT3ijs))
        return d3C_dT3

    def dB_dzs(self):
        r'''Method to calculate and return the first mole fraction derivatives 
        of the `B` second virial coefficient.
        
        Returns
        -------
        dB_dzs : list[float]
            First mole fraction derivatives of second molar virial coefficient 
            [m^3/(mol)]
        '''
        try:
            return self._dB_dzs
        except:
            pass

        zs = self.zs
        if not self.cross_B_coefficients:
            Bs = self.model.B_pures()
            self._dB_dzs = dB_dzs = Bs
            return dB_dzs
        N = self.N
        if self.scalar:
            dB_dzs = [0.0]*N
        else:
            dB_dzs = zeros(N)

        B_interactions = self.model.B_interactions()
        self._dB_dzs = dB_dzs = dBVirial_mixture_dzs(zs, B_interactions, dB_dzs)
        return dB_dzs

    def d2B_dTdzs(self):
        r'''Method to calculate and return the temperature derivative of the
        first mole fraction derivatives 
        of the `B` second virial coefficient.
        
        Returns
        -------
        d2B_dTdzs : list[float]
            First temperature derivative of first mole fraction derivatives of
            second molar virial coefficient [m^3/(mol*K)]
        '''
        try:
            return self._d2B_dTdzs
        except:
            pass

        zs = self.zs
        if not self.cross_B_coefficients:
            Bs = self.model.dB_dT_pures()
            self._d2B_dTdzs = d2B_dTdzs = Bs
            return d2B_dTdzs

        N = self.N
        if self.scalar:
            d2B_dTdzs = [0.0]*N
        else:
            d2B_dTdzs = zeros(N)
        B_interactions = self.model.dB_dT_interactions()
        self._d2B_dTdzs = d2B_dTdzs = dBVirial_mixture_dzs(zs, B_interactions, d2B_dTdzs)
        return d2B_dTdzs

    def d3B_dT2dzs(self):
        r'''Method to calculate and return the second temperature derivative of the
        first mole fraction derivatives 
        of the `B` second virial coefficient.
        
        Returns
        -------
        d3B_dT2dzs : list[float]
            Second temperature derivative of first mole fraction derivatives of
            second molar virial coefficient [m^3/(mol*K^2)]
        '''
        try:
            return self._d3B_dT2dzs
        except:
            pass

        zs = self.zs
        if not self.cross_B_coefficients:
            Bs = self.model.d2B_dT2_pures()
            self._d3B_dT2dzs = d3B_dT2dzs = Bs
            return d3B_dT2dzs

        N = self.N
        if self.scalar:
            d3B_dT2dzs = [0.0]*N
        else:
            d3B_dT2dzs = zeros(N)
        B_interactions = self.model.d2B_dT2_interactions()
        self._d3B_dT2dzs = d3B_dT2dzs = dBVirial_mixture_dzs(zs, B_interactions, d3B_dT2dzs)
        return d3B_dT2dzs

    def d4B_dT3dzs(self):
        r'''Method to calculate and return the third temperature derivative of the
        first mole fraction derivatives 
        of the `B` second virial coefficient.
        
        Returns
        -------
        d4B_dT3dzs : list[float]
            Third temperature derivative of first mole fraction derivatives of
            second molar virial coefficient [m^3/(mol*K^3)]
        '''
        try:
            return self._d4B_dT3dzs
        except:
            pass

        zs = self.zs
        if not self.cross_B_coefficients:
            Bs = self.model.d3B_dT3_pures()
            self._d4B_dT3dzs = d4B_dT3dzs = Bs
            return d4B_dT3dzs
        N = self.N
        if self.scalar:
            d4B_dT3dzs = [0.0]*N
        else:
            d4B_dT3dzs = zeros(N)

        B_interactions = self.model.d3B_dT3_interactions()
        self._d4B_dT3dzs = d4B_dT3dzs = dBVirial_mixture_dzs(zs, B_interactions, d4B_dT3dzs)
        return d4B_dT3dzs

    def d2B_dzizjs(self):
        r'''Method to calculate and return the second mole fraction derivatives 
        of the `B` second virial coefficient.
        
        Returns
        -------
        d2B_dzizjs : list[list[float]]
            Second mole fraction derivatives of second molar virial coefficient 
            [m^3/(mol)]
        '''
        try:
            return self._d2B_dzizjs
        except:
            pass

        N = self.N
        zs = self.zs
        if self.scalar:
            d2B_dzizjs = [[0.0]*N for _ in range(N)]
        else:
            d2B_dzizjs = zeros((N, N))
        if not self.cross_B_coefficients:
            self._d2B_dzizjs = d2B_dzizjs
            return d2B_dzizjs

        B_interactions = self.model.B_interactions()
        self._d2B_dzizjs = d2B_dzizjs = d2BVirial_mixture_dzizjs(zs, B_interactions, d2B_dzizjs)
        return d2B_dzizjs

    def d3B_dTdzizjs(self):
        r'''Method to calculate and return the first temperature derivative of
        the second mole fraction derivatives 
        of the `B` second virial coefficient.
        
        Returns
        -------
        d3B_dTdzizjs : list[list[float]]
            First temperature derivative of second mole fraction derivatives of
            second molar virial coefficient 
            [m^3/(mol*K)]
        '''
        try:
            return self._d3B_dTdzizjs
        except:
            pass

        N = self.N
        zs = self.zs
        if self.scalar:
            d3B_dTdzizjs = [[0.0]*N for _ in range(N)]
        else:
            d3B_dTdzizjs = zeros((N, N))
        if not self.cross_B_coefficients:
            self._d3B_dTdzizjs = d3B_dTdzizjs
            return d3B_dTdzizjs

        B_interactions = self.model.dB_dT_interactions()
        self._d3B_dTdzizjs = d3B_dTdzizjs = d2BVirial_mixture_dzizjs(zs, B_interactions, d3B_dTdzizjs)
        return d3B_dTdzizjs

    def d4B_dT2dzizjs(self):
        r'''Method to calculate and return the second temperature derivative of
        the second mole fraction derivatives 
        of the `B` second virial coefficient.
        
        Returns
        -------
        d4B_dT2dzizjs : list[list[float]]
            Second temperature derivative of second mole fraction derivatives of
            second molar virial coefficient 
            [m^3/(mol*K^2)]
        '''
        try:
            return self._d4B_dT2dzizjs
        except:
            pass

        N = self.N
        zs = self.zs
        if self.scalar:
            d4B_dT2dzizjs = [[0.0]*N for _ in range(N)]
        else:
            d4B_dT2dzizjs = zeros((N, N))
        if not self.cross_B_coefficients:
            self._d4B_dT2dzizjs = d4B_dT2dzizjs
            return d4B_dT2dzizjs

        B_interactions = self.model.d2B_dT2_interactions()
        self._d4B_dT2dzizjs = d4B_dT2dzizjs = d2BVirial_mixture_dzizjs(zs, B_interactions, d4B_dT2dzizjs)
        return d4B_dT2dzizjs

    def d3B_dzizjzks(self):
        r'''Method to calculate and return the third mole fraction derivatives 
        of the `B` second virial coefficient.
        
        Returns
        -------
        d3B_dzizjzks : list[list[list[float]]]
            Third mole fraction derivatives of second molar virial coefficient 
            [m^3/(mol)]
        '''
        try:
            return self._d3B_dzizjzks
        except:
            pass

        N = self.N
        zs = self.zs
        if self.scalar:
            d3B_dzizjzks = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            d3B_dzizjzks = zeros((N, N, N))
        if not self.cross_B_coefficients:
            self._d3B_dzizjzks = d3B_dzizjzks
            return d3B_dzizjzks

        B_interactions = self.model.B_interactions()
        self._d3B_dzizjzks = d3B_dzizjzks = d3BVirial_mixture_dzizjzks(zs, B_interactions, d3B_dzizjzks)
        return d3B_dzizjzks
    
    d4B_dTdzizjzks = d3B_dzizjzks
    d5B_dT2dzizjzks = d3B_dzizjzks
    d6B_dT3dzizjzks = d3B_dzizjzks

    def dB_dns(self):
        r'''Method to calculate and return the first mole number derivatives 
        of the `B` second virial coefficient.
        
        Returns
        -------
        dB_dns : list[float]
            First mole number derivatives of second molar virial coefficient 
            [m^3/(mol^2)]
        '''
        try:
            return self._dB_dns
        except:
            pass
        N = self.N
        if self.scalar:
            dB_dns = [0.0]*N
        else:
            dB_dns = zeros(N)
        self._dB_dns = dB_dns = dxs_to_dns(dxs=self.dB_dzs(), xs=self.zs, dns=dB_dns)
        return dB_dns 
        
    def dnB_dns(self):
        r'''Method to calculate and return the first partial mole number  
        derivatives of the `B` second virial coefficient.
        
        Returns
        -------
        dnB_dns : list[float]
            First partial mole number derivatives of second molar virial
            coefficient [m^3/(mol)]
        '''
        try:
            return self._dnB_dns
        except:
            pass
        N = self.N
        if self.scalar:
            dnB_dns = [0.0]*N
        else:
            dnB_dns = zeros(N)
        
        self._dnB_dns = dnB_dns = dxs_to_dn_partials(dxs=self.dB_dzs(), xs=self.zs, F=self.B(), partial_properties=dnB_dns)
        return dnB_dns 

    def dC_dzs(self):
        r'''Method to calculate and return the first mole fraction derivatives 
        of the `C` third virial coefficient.
        
        Returns
        -------
        dC_dzs : list[float]
            First mole fraction derivatives of third molar virial coefficient 
            [m^6/(mol^2)]
        '''
        try:
            return self._dC_dzs
        except:
            pass

        zs = self.zs
        if not self.cross_C_coefficients:
            Cs = self.model.C_pures()
            self._dC_dzs = dC_dzs = Cs
            return dC_dzs
        N = self.N
        if self.scalar:
            dC_dzs = [0.0]*N
        else:
            dC_dzs = zeros(N)
        self._dC_dzs = dC_dzs
            
        if not self.model.C_zero:
            C_interactions = self.model.C_interactions()
            dCVirial_mixture_Orentlicher_Prausnitz_dzs(zs, C_interactions, dC_dzs)
        return dC_dzs
    
    def d2C_dTdzs(self):
        r'''Method to calculate and return the first temperature derivative of
        the first mole fraction derivatives of the `C` third virial coefficient.
        
        Returns
        -------
        d2C_dTdzs : list[float]
            First temperature derivative of the first mole fraction derivatives
            of third molar virial coefficient [m^6/(mol^2*K)]
        '''
        try:
            return self._d2C_dTdzs
        except:
            pass
    
        zs = self.zs
        if not self.cross_C_coefficients:
            self._d2C_dTdzs = d2C_dTdzs = self.model.dC_dT_pures()
            return d2C_dTdzs
        N = self.N
        if self.scalar:
            d2C_dTdzs = [0.0]*N
        else:
            d2C_dTdzs = zeros(N)
        
        self._d2C_dTdzs = d2C_dTdzs
        if not self.model.C_zero:
            C_interactions = self.model.C_interactions()
            dC_dT_interactions = self.model.dC_dT_interactions()
            d2CVirial_mixture_Orentlicher_Prausnitz_dTdzs(zs, C_interactions, dC_dT_interactions, d2C_dTdzs)
        return d2C_dTdzs
    


    
    def d2C_dzizjs(self):
        r'''Method to calculate and return the second mole fraction derivatives 
        of the `C` third virial coefficient.
        
        Returns
        -------
        d2C_dzizjs : list[list[float]]
            Second mole fraction derivatives of third molar virial coefficient 
            [m^6/(mol^2)]
        '''
        try:
            return self._d2C_dzizjs
        except:
            pass

        N = self.N
        if self.scalar:
            d2C_dzizjs = [[0.0]*N for _ in range(N)]
        else:
            d2C_dzizjs = zeros((N, N))

        zs = self.zs
        if not self.cross_C_coefficients:
            # Cs = self.model.C_pures()
            # for i, C in enumerate(Cs):
                # d2C_dzizjs[i][i] = C
            self._d2C_dzizjs = d2C_dzizjs
            return d2C_dzizjs
        
        self._d2C_dzizjs = d2C_dzizjs
        if not self.model.C_zero:
            C_interactions = self.model.C_interactions()
            d2CVirial_mixture_Orentlicher_Prausnitz_dzizjs(zs, C_interactions, d2C_dzizjs)
        return d2C_dzizjs
        
    
    def d3C_dzizjzks(self):
        r'''Method to calculate and return the third mole fraction derivatives 
        of the `C` third virial coefficient.
        
        Returns
        -------
        d3C_dzizjzks : list[list[float]]
            Third mole fraction derivatives of third molar virial coefficient 
            [m^6/(mol^2)]
        '''
        try:
            return self._d3C_dzizjzks
        except:
            pass

        N = self.N
        if self.scalar:
            d3C_dzizjzks = [[[0.0]*N for _ in range(N)] for _ in range(N)]
        else:
            d3C_dzizjzks = zeros((N, N, N))

        zs = self.zs
        if not self.cross_C_coefficients:
            self._d3C_dzizjzks = d3C_dzizjzks
            return d3C_dzizjzks

        self._d3C_dzizjzks = d3C_dzizjzks

        if not self.model.C_zero:
            C_interactions = self.model.C_interactions()
            d3CVirial_mixture_Orentlicher_Prausnitz_dzizjzks(zs, C_interactions, d3C_dzizjzks)
        return d3C_dzizjzks

    def dC_dns(self):
        r'''Method to calculate and return the first mole number derivatives 
        of the `C` third virial coefficient.
        
        Returns
        -------
        dC_dns : list[float]
            First mole number derivatives of third molar virial coefficient 
            [m^6/(mol^3)]
        '''
        try:
            return self._dC_dns
        except:
            pass
        N = self.N
        if self.scalar:
            dC_dns = [0.0]*N
        else:
            dC_dns = zeros(N)
        self._dC_dns = dC_dns = dxs_to_dns(dxs=self.dC_dzs(), xs=self.zs, dns=dC_dns)
        return dC_dns 
        
    def dnC_dns(self):
        r'''Method to calculate and return the first partial mole number  
        derivatives of the `C` third virial coefficient.
        
        Returns
        -------
        dnC_dns : list[float]
            First partial mole number derivatives of third molar virial coefficient 
            [m^6/(mol^2)]
        '''
        try:
            return self._dnC_dns
        except:
            pass
        N = self.N
        if self.scalar:
            dnC_dns = [0.0]*N
        else:
            dnC_dns = zeros(N)
        
        self._dnC_dns = dnC_dns = dxs_to_dn_partials(dxs=self.dC_dzs(), xs=self.zs, F=self.C(), partial_properties=dnC_dns)
        return dnC_dns 


VirialGas.H = CEOSGas.H
VirialGas.S = CEOSGas.S
VirialGas.Cp = CEOSGas.Cp