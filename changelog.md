# Changelog

## [Unreleased]

### Added

### Changed

### Removed

### Fixed

## [0.6.0] - 2025-10-26

### Added

- Project is now PEP 517 compliant and doesn't use deprecated setup.py commands anymore
- GitHub Actions workflow for publishing to PyPI using environment protection
- `uv` package manager integration across CI workflows for faster dependency resolution
- New consolidated `quality.yml` workflow for linting and testing
- New `build_third_party_packagers.yml` workflow consolidating cx_Freeze, PyInstaller, Nuitka, and py2exe testing
- Pre-commit hooks configuration
- Justfile with extensive development automation commands - github actions have been refactored use this where possible, making them locally debuggable

### Changed

- Consolidated GitHub Actions workflows for better organization
- Moved test_documentation.py from tests/ to docs/ directory
- Updated PyInstaller standalone check specification
- Migrated all configuration from separate files to pyproject.toml:
  - Ruff configuration (from .ruff.toml)
  - pytest configuration (from pytest.ini)
  - mypy configuration (from mypy.ini)
- Reorganized requirements files (security, docs, test)
- Improved build system configuration in pyproject.toml

### Removed

- Removed standalone packaging workflow files:
  - build_cxfreeze_library.yml
  - build_pyinstaller_library.yml
  - build_nuitka_library.yml
  - build_py2exe_library.yml (including py2exe builder script)
- Removed pre-commit.yml workflow (consolidated into quality.yml)
- Removed security.yml workflow (consolidated into quality.yml)
- Removed configuration files migrated to pyproject.toml:
  - .ruff.toml
  - pytest.ini
  - mypy.ini
- Removed separate requirements files:
  - requirements_test.txt
  - requirements_test_multiarch.txt
  - requirements_docs.txt

### Fixed

- Test execution issues on certain platforms

## [0.5.0] - 2025-10-19

### Added

- Python 3.13 and 3.13t (free-threaded) support with PYTHON_GIL=0 configuration
- Pre-commit configuration with Ruff, mdformat, and file validators
- New GitHub Actions workflows for pre-commit checks and security scanning
- Packaging compatibility workflows for cx_Freeze, PyInstaller, and py2exe
- Standalone test scripts and demo builders for verifying packaged distributions
- Coverage HTML artifact uploads to all test workflows
- Concurrency controls to workflows to cancel redundant builds
- Justfile for streamlined development tasks (setup, docs, test, typecheck, lint)
- Security scanning with pip-audit and bandit
- Modified UNIFAC 2.0 interaction parameters (DOUF2IP): Hayer, Nicolas, Hans Hasse, and Fabian Jirasek. "Modified UNIFAC 2.0-A Group-Contribution Method Completed with Machine Learning." Industrial & Engineering Chemistry Research 64, no. 20 (2025): 10304–13. https://doi.org/10.1021/acs.iecr.5c00077.

### Changed

- Minimum Python version raised from 3.6 to 3.8
- Updated actions to latest versions (setup-qemu v3, run-on-arch v3)
- Updated macOS CI runners (macos-13 → macos-15-intel, added macos-latest for ARM)
- Extensive code quality improvements with Ruff linting across entire codebase:
  - String quote normalization to double quotes
  - Removed unused imports and variables
  - Improved code formatting and PEP 8 compliance
  - Better type hints compatibility
- Begun process of integrating type hints into the codebase
- Updated copyright year to 2025
- Fixed numerous typos across documentation files
- Improved Sphinx configuration for Python 3.13 compatibility
- Enhanced docstring and markdown formatting
- Improved coveralls error handling (continue on failure)
- Updated README to reflect Python 3.8+ requirement

### Removed

- Dropped Python 3.6 and 3.7 support
- Removed obsolete platform-specific exclusions

### Security

- Added automated security scanning workflow documented in SECURITY.md

## [0.4.2] - 2025-03-16

### Added

- Unifac 2.0: Hayer, Nicolas, Thorsten Wendel, Stephan Mandt, Hans Hasse, and Fabian Jirasek. “Advancing Thermodynamic Group-Contribution Methods by Machine Learning: UNIFAC 2.0.” Chemical Engineering Journal 504 (January 15, 2025): 158667. https://doi.org/10.1016/j.cej.2024.158667.

- First pass implementation of Bondi group contribution method for estimating R and Q for UNIQUAC (and regressing new UNIFAC groups). Some definitions are unclear and/or hard to compute programatically.

- More UNIFAC groups have SMARTS groups patterns assigned to them

- Add functions for identifying which functional groups are in a chemical

- Molecule graph structure functions `count_rings_by_atom_counts`, `identify_functional_group_atoms`, `identify_conjugated_bonds`

- New Diky Joback implementation which improves predictions of ideal gas heat capacity, from: Elliott, J. Richard, Vladimir Diky, Thomas A. Knotts IV, and W. Vincent Wilding. The Properties of Gases and Liquids, Sixth Edition. 6th edition. New York: McGraw Hill, 2023.

- Activity coefficient models have `missing_interaction_parameters` function added to return a list of compound interactions that are missing

### Changed

- NISTUFSG CH=NOH previously had the group ID 309, but this duplicated another 309 group CH2(O)2, so it was reassigned 1309 in thermo https://github.com/CalebBell/thermo/issues/158
- Reorganized the teperature dependent chemical property data additions to TDependentProperty objects so that data is added through generic methods, instead of hardcoding the method and attributes on the classes that the attributes are stored on. This positions thermo for adding new data sets easier in the future.
- Renamed arguments `cross_B_model` to `B_mixing_rule` of `VirialGas` to avoid confusion with similar parameter in `VirialCSP`. and the same for `cross_C_model` to `C_mixing_rule`
- Fix missing negative sign on `H_dep` of Virial model - this had impact on enthalpy derivatives, `G_dep`, and fugacity related terms as well. This was a serious bug and any `VirialCSP` calculations should be repeated because of it.
- Clean up UNIFAC assignment tests
- Fix inconsistency of using IAPWS name for multiple TDependentProperty methods

## [0.4.0] - 2024-11-10

### Changed

- Fluids version dependency now >= 1.0.27
- Chemicals version dependency now >= 1.3.0
- General code cleanup and further documentation
- Add Flory Huggins and Hansen activity coefficient models
- Further progress on removing legacy scipy interp1d method
- Clean up in code of vapor pressure extrapolation

## [0.3.0] - 2024-07-26

### Changed

- Compatibility with NumPy 2.0 and SciPy 1.14. Note this causes somewhat different results when extrapolating tabular data (Temperature and pressure dependent - temperature dependent only behaves the same)
- Fluids version dependency now >= 1.0.26
- Chemicals version dependency now >= 1.2.0
- General code cleanup and further documentation

## [0.2.26] - 2023-09-17

### Changed

- Previously added accurate fits to pure-component temperature-dependent properties now have analytical integrals implemented so as to speed up enthalpy and entropy calculations.
- Creation of Flasher objects has been sped up
- Add some fits for pure metal solid and liquid heat capacities to the SGTE UNARY database. The fits are quite accurate but do not implement the same equations.
- Add base class for ThermalConductivitySolid
- Add element fits for thermal conductivities of solids from the source: Ho, C. Y., R. W. Powell, and P. E. Liley. "Thermal Conductivity of the Elements." Journal of Physical and Chemical Reference Data 1, no. 2 (April 1, 1972): 279-421. https://doi.org/10.1063/1.3253100.
- Add additional data for sublimation pressure
- Fix an issue with threading and sqlite lookups

## [0.2.25] - 2023-06-04

### Changed

- Code cleanup with ruff (experiment)
- Add accurate fits to pure-component temperature-dependent properties derived using REFPROP. This is the preferred method where available. As part of this effort, a way of adding new data to thermo using json files is being experimented with.
- Add liquid density and viscosity correlation for 8 elements (experiment)
