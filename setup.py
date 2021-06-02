# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from setuptools import setup

classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Manufacturing',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: BSD',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: Implementation :: CPython',
    'Programming Language :: Python :: Implementation :: PyPy',
    'Topic :: Education',
    'Topic :: Scientific/Engineering :: Atmospheric Science',
    'Topic :: Scientific/Engineering :: Chemistry',
    'Topic :: Scientific/Engineering :: Physics',
]


setup(
  name = 'thermo',
  packages = ['thermo'],
  license='MIT',
  version = '0.2.7',
  description = 'Chemical properties component of Chemical Engineering Design Library (ChEDL)',
  author = 'Caleb Bell',
  install_requires=['fluids>=1.0.8', 'scipy', 'pandas', 'chemicals>=1.0.10'],
  extras_require = {
      'Coverage documentation':  ['wsgiref>=0.1.2', 'coverage>=4.0.3']
  },
  long_description=open('README.rst').read(),
  platforms=["Windows", "Linux", "Mac OS", "Unix"],
  author_email = 'Caleb.Andrew.Bell@gmail.com',
  url = 'https://github.com/CalebBell/thermo',
  download_url = 'https://github.com/CalebBell/thermo/tarball/0.2.7',
  keywords = ['chemical engineering', 'chemistry', 'mechanical engineering',
  'thermodynamics', 'databases', 'cheminformatics', 'engineering','viscosity',
  'density', 'heat capacity', 'thermal conductivity', 'surface tension',
  'combustion', 'environmental engineering', 'solubility', 'vapor pressure',
  'equation of state', 'molecule'],
  classifiers = classifiers,
  package_data={'thermo': ['Critical Properties/*', 'Density/*',
  'Electrolytes/*', 'Environment/*', 'Heat Capacity/*', 'Identifiers/*',
  'Law/*', 'Misc/*', 'Phase Change/*', 'Reactions/*', 'Safety/*',
  'Solubility/*', 'Interface/*', 'Triple Properties/*',
  'Thermal Conductivity/*', 'Interaction Parameters/*',
  'flash/*', 'phases/*',
  'Interaction Parameters/ChemSep/*',
  'Vapor Pressure/*', 'Viscosity/*']}
)

