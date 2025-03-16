# Changelog

## [Unreleased]

### Added
- Unifac 2.0

### Changed
 - NISTUFSG CH=NOH previously had the group ID 309, but this duplicated another 309 group CH2(O)2, so it was reassigned 1309 in thermo https://github.com/CalebBell/thermo/issues/158

 - Renamed arguments `cross_B_model` to `B_mixing_rule` of `VirialGas` to avoid confusion with similar parameter in `VirialCSP`. and the same for `cross_C_model` to `C_mixing_rule`
 - Fix missing negative sign on `H_dep` of Virial model - this had impact on enthalpy derivatives, `G_dep`, and fugacity related terms as well. This was a serious bug and any `VirialCSP` calculations should be repeated because of it. 

### Removed

### Fixed


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