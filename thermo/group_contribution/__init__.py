'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2022 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

This module contains base classes for temperature `T`, pressure `P`, and
composition `zs` dependent properties. These power the various interfaces for
each property.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/chemicals/>`_.

.. contents:: :local:


'''
from thermo.group_contribution.fedors import Fedors
from thermo.group_contribution.joback import J_BIGGS_JOBACK_SMARTS, J_BIGGS_JOBACK_SMARTS_id_dict, Joback
from thermo.group_contribution.wilson_jasperson import (
    Wilson_Jasperson,
    Wilson_Jasperson_Pc_groups,
    Wilson_Jasperson_Pc_increments,
    Wilson_Jasperson_Tc_groups,
    Wilson_Jasperson_Tc_increments,
)

__all__ = ('Wilson_Jasperson', 'Wilson_Jasperson_Tc_increments',
           'Wilson_Jasperson_Pc_increments',
           'Wilson_Jasperson_Tc_groups', 'Wilson_Jasperson_Pc_groups',
           'Joback', 'J_BIGGS_JOBACK_SMARTS',
           'J_BIGGS_JOBACK_SMARTS_id_dict',
           'Fedors',
           )
