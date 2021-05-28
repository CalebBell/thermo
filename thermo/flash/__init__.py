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

This module contains classes and functions for performing flash calculations.

For reporting bugs, adding feature requests, or submitting pull requests,
please use the `GitHub issue tracker <https://github.com/CalebBell/thermo/>`_.

.. contents:: :local:

Main Interfaces
===============

Pure Components
---------------
.. autoclass:: FlashPureVLS
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Vapor-Liquid Systems
--------------------
.. autoclass:: FlashVL
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Vapor and Multiple Liquid Systems
---------------------------------
.. autoclass:: FlashVLN
   :show-inheritance:
   :members: __init__
   :exclude-members: __init__

Base Flash Class
----------------
.. autoclass:: Flash
   :show-inheritance:
   :members: flash, plot_TP
   :exclude-members:


Specific Flash Algorithms
=========================
It is recommended to use the Flash classes, which are designed to have generic
interfaces. The implemented specific flash algorithms may be changed in the
future, but reading their source code may be helpful for instructive purposes.

'''
from . import flash_utils
from . import flash_base
from . import flash_vl
from . import flash_vln
from . import flash_pure_vls

from .flash_utils import *
from .flash_base import *
from .flash_vl import *
from .flash_vln import *
from .flash_pure_vls import *

__all__ = (flash_utils.__all__ + flash_base.__all__ + flash_vl.__all__
           + flash_vln.__all__ + flash_pure_vls.__all__)

