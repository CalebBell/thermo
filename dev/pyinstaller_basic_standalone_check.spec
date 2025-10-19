# -*- mode: python ; coding: utf-8 -*-
import os
import glob

# Collect all data files from thermo package (local)
datas = []
thermo_data_dirs = ["Critical Properties", "Density", "Electrolytes", "Environment",
                    "Heat Capacity", "Identifiers", "Law", "Misc", "Phase Change",
                    "Reactions", "Safety", "Solubility", "Interface", "Triple Properties",
                    "Thermal Conductivity", "Interaction Parameters", "Scalar Parameters",
                    "Vapor Pressure", "Viscosity"]

for data_dir in thermo_data_dirs:
    dir_path = os.path.join('..', 'thermo', data_dir)
    if os.path.exists(dir_path):
        datas.append((dir_path, os.path.join('thermo', data_dir)))

# Handle Interaction Parameters/ChemSep subdirectory for thermo
chemsep_dir = os.path.join('..', 'thermo', 'Interaction Parameters', 'ChemSep')
if os.path.exists(chemsep_dir):
    datas.append((chemsep_dir, os.path.join('thermo', 'Interaction Parameters', 'ChemSep')))

# Collect all data files from installed chemicals package
try:
    import chemicals
    chemicals_path = os.path.dirname(chemicals.__file__)
    chemicals_data_dirs = ["Critical Properties", "Density", "Electrolytes", "Environment",
                           "Heat Capacity", "Identifiers", "Law", "Misc", "Phase Change",
                           "Reactions", "Safety", "Solubility", "Interface", "Triple Properties",
                           "Thermal Conductivity", "Vapor Pressure", "Viscosity"]

    for data_dir in chemicals_data_dirs:
        dir_path = os.path.join(chemicals_path, data_dir)
        if os.path.exists(dir_path):
            datas.append((dir_path, os.path.join('chemicals', data_dir)))
except ImportError:
    print("Warning: chemicals package not found, data files may be missing")

a = Analysis(
    ['basic_standalone_thermo_check.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide2', 'PyQt6', 'PySide6', 'tkinter', 'matplotlib', 'IPython', 'notebook', 'jupyter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='basic_standalone_thermo_check',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
