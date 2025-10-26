"""Custom build backend for creating minified wheels for Pyodide"""
from setuptools import build_meta as _orig
from setuptools.build_meta import *
import tempfile
import shutil
from pathlib import Path


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel - minified if config_settings["light"] is set"""

    # Check if we should build a light wheel
    if config_settings and config_settings.get("light") == "true":
        return _build_wheel_light(wheel_directory, config_settings, metadata_directory)

    # Otherwise build normal wheel
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def _build_wheel_light(wheel_directory, config_settings, metadata_directory):
    """Build a minified wheel for Pyodide"""
    import python_minifier

    pkg_dir = Path.cwd() / "thermo"

    # Files to exclude (relative to thermo directory)
    exclude_files = [
        "data",
    ]

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        moved_files = []
        minified_files = []

        try:
            # Move files to temporary location
            for rel_path in exclude_files:
                orig_path = pkg_dir / rel_path
                if orig_path.exists():
                    # Create path in temp dir maintaining structure
                    temp_path = Path(temp_dir) / rel_path
                    temp_path.parent.mkdir(parents=True, exist_ok=True)

                    shutil.move(str(orig_path), str(temp_path))
                    moved_files.append((orig_path, temp_path))

            # Minify .py files
            for py_file in pkg_dir.rglob("*.py"):
                # Store original content and minify
                with open(py_file, encoding="utf-8") as f:
                    original_content = f.read()

                minified_content = python_minifier.minify(
                    original_content,
                    remove_annotations=True,
                    remove_pass=True,
                    remove_literal_statements=True
                )

                # Write minified content
                with open(py_file, "w", encoding="utf-8") as f:
                    f.write(minified_content)

                # Store original content for restoration
                minified_files.append((py_file, original_content))

            # Build the wheel
            result = _orig.build_wheel(wheel_directory, config_settings, metadata_directory)

        finally:
            # Restore original Python files
            for file_path, original_content in minified_files:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(original_content)

            # Restore moved files
            for orig_path, temp_path in moved_files:
                if temp_path.exists():
                    orig_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(temp_path), str(orig_path))

        return result
