'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2020, 2021, 2022 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel
import os
import shutil
from pathlib import Path
import tempfile

class bdist_wheel_light(bdist_wheel):
    description = "Build a light wheel package without test files and large data files"
    
    def run(self):
        # Get absolute path to the project directory
        pkg_dir = Path(os.path.abspath('thermo'))
        root_dir = pkg_dir.parent
        
        # Files/directories to exclude (paths relative to project root)
        exclude_files = [
            'tests',
            'thermo/Interaction Parameters/PRTranslated_best_henry_T_dep.json',
            'thermo/Law',
            'thermo/Misc/constants dump.json',
            'thermo/Phase Change/DDBST UNIFAC assignments.tsv',
            '.pylintrc'
        ]
        
        # Also exclude any files with 'polyfits' in the name
        for path in pkg_dir.rglob('*polyfits*'):
            rel_path = path.relative_to(root_dir)
            exclude_files.append(str(rel_path))
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            moved_files = []
            
            try:
                # Move files to temporary location
                for rel_path in exclude_files:
                    orig_path = root_dir / rel_path
                    if orig_path.exists():
                        # Create path in temp dir maintaining structure
                        temp_path = Path(temp_dir) / rel_path
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        if orig_path.is_dir():
                            shutil.move(str(orig_path), str(temp_path))
                        else:
                            shutil.move(str(orig_path), str(temp_path))
                        moved_files.append((orig_path, temp_path))
                
                # Build the wheel
                super().run()
                
            finally:
                # Restore moved files
                for orig_path, temp_path in moved_files:
                    if temp_path.exists():
                        orig_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(temp_path), str(orig_path))

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
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
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
  version = '0.4.0',
  description = 'Chemical properties component of Chemical Engineering Design Library (ChEDL)',
  author = 'Caleb Bell',
  install_requires=['fluids>=1.1.0', "scipy>=1.6.0", 'pandas', 'chemicals>=1.3.0'],
  extras_require = {
      'Coverage documentation':  ['wsgiref>=0.1.2', 'coverage>=4.0.3']
  },
  long_description=open('README.rst').read(),
  platforms=["Windows", "Linux", "Mac OS", "Unix"],
  author_email = 'Caleb.Andrew.Bell@gmail.com',
  url = 'https://github.com/CalebBell/thermo',
  download_url = 'https://github.com/CalebBell/thermo/tarball/0.4.0',
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
  'flash/*', 'phases/*', 'utils/*','group_contribution/*',
  'Scalar Parameters/*',
  'Interaction Parameters/ChemSep/*',
  'Vapor Pressure/*', 'Viscosity/*']},
    cmdclass={
        'bdist_wheel_light': bdist_wheel_light,
    }  
)

