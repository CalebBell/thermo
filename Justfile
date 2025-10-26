# Use a strict shell for more predictable recipe execution.
# The "-c" is crucial: it tells bash to treat the recipe lines as commands.
set shell := ["bash", "-euo", "pipefail", "-c"]

# --- Variables ---
# Define paths to executables within the virtual environment for clarity.
# This ensures we always use the tools installed in our project's venv.
VENV_PYTHON := ".venv/bin/python"
VENV_PYTEST := ".venv/bin/pytest"
VENV_MYPY   := ".venv/bin/mypy"
VENV_RUFF   := ".venv/bin/ruff"
VENV_PIP_AUDIT := ".venv/bin/pip-audit"
VENV_BANDIT := ".venv/bin/bandit"

# Cross-platform variables for third-party packager tests
VENV_BIN_DIR := if os_family() == "windows" { "Scripts" } else { "bin" }
PYTHON_EXE := if os_family() == "windows" { "python.exe" } else { "python" }
EXE_SUFFIX := if os_family() == "windows" { ".exe" } else { "" }

# --- Main Recipes ---

# The default recipe, run when you just type `just`. It lists available commands.
default:
    @just --list

## ⚙️  install: Create a uv virtual environment and install all dependencies.
install:
    @echo ">>> Creating virtual environment in ./.venv..."
    @uv venv
    @echo "\n>>> Installing 'thermo' in editable mode with dev dependencies..."
    @uv pip install -e .[dev]
    @echo "\n>>> Installing prek hooks..."
    @prek install
    @echo "\n✅ Environment setup complete! You can now run other commands."

## 📚 docs: Build the Sphinx documentation.
docs:
    @echo ">>> Building Sphinx docs..."
    # Note: -j auto (parallel build) is faster but less stable, can cause JSON decoding crashes
    @{{VENV_PYTHON}} -m sphinx -b html -d _build/doctrees docs _build/html
    @echo "✅ Docs built in _build/html"

## 🗄️  generate-databases: Generate required database files (UNIFAC assignments).
generate-databases:
    @if [ ! -f "thermo/Phase Change/DDBST_UNIFAC_assignments.sqlite" ]; then \
        echo ">>> Generating UNIFAC database files..."; \
        {{VENV_PYTHON}} dev/dump_UNIFAC_assignments_to_sqlite.py; \
        echo "✅ Database files generated."; \
    else \
        echo ">>> UNIFAC database already exists, skipping generation..."; \
    fi

## 🧪 test: Run the test suite with pytest.
test *ARGS:
    @just generate-databases
    @echo ">>> Running pytest..."
    @{{VENV_PYTEST}} -n auto -m "not online and not sympy and not numba and not CoolProp and not fuzz and not deprecated and not slow" {{ARGS}}

## 📊 test-cov: Run tests with coverage report.
test-cov:
    @just generate-databases
    @echo ">>> Running pytest with coverage..."
    @{{VENV_PYTEST}} -n auto -m "not online and not sympy and not numba and not CoolProp and not fuzz and not deprecated and not slow" --cov=thermo --cov-report=html --cov-report=term
    @echo "✅ Coverage report generated in htmlcov/"

## 🧐 typecheck: Check static types with mypy.
typecheck:
    @echo ">>> Running mypy..."
    @{{VENV_MYPY}} .

## ✨ lint: Check for code style issues and errors with Ruff.
lint:
    @echo ">>> Running Ruff..."
    @{{VENV_RUFF}} check .

## 🏁 check: Run all checks (linting and type checking).
check: lint typecheck

## 🔒 security: Run security scans with pip-audit and bandit.
security:
    @echo ">>> Running pip-audit..."
    @{{VENV_PIP_AUDIT}} -r requirements_security.txt
    @echo ">>> Running bandit..."
    @{{VENV_BANDIT}} -r thermo -ll
    @echo "✅ Security scans complete."

## 🪝 precommit: Run pre-commit hooks on all files.
precommit:
    @echo ">>> Running pre-commit hooks..."
    @prek run --all-files

## 🔌 hooks-install: Install prek hooks.
hooks-install:
    @echo ">>> Installing prek hooks..."
    @prek install
    @echo "✅ Hooks installed."

## 🗑️  hooks-remove: Remove prek hooks.
hooks-remove:
    @echo ">>> Removing prek hooks..."
    @prek uninstall
    @echo "✅ Hooks removed."

# asv is broken
# ## ⚡ bench: Run performance benchmarks.
# bench:
#     @echo ">>> Running benchmarks..."
#     @asv run

## 📦 build: Build wheel and source distributions.
build:
    @echo ">>> Building distributions..."
    @{{VENV_PYTHON}} -m build
    @echo "✅ Distributions built in dist/"

## 🔍 check-dist: Check built distributions with twine.
check-dist:
    @echo ">>> Checking distributions with twine..."
    @.venv/bin/twine check dist/*
    @echo "✅ Distributions are valid."

## 🚀 ci: Run all CI checks (lint, typecheck, test).
ci: lint typecheck test
    @echo "✅ All CI checks passed!"

## 🧊 test-cxfreeze: Test cx_Freeze compatibility (build executable and run it).
test-cxfreeze py="3.13":
    @echo ">>> Creating temporary virtual environment with Python {{py}}..."
    @uv venv .venv-cxfreeze-{{py}} --python {{py}}
    @echo "\n>>> Installing project and cx_Freeze in temporary environment..."
    @uv pip install --python .venv-cxfreeze-{{py}}/{{VENV_BIN_DIR}}/{{PYTHON_EXE}} -e .[test]
    @uv pip install --python .venv-cxfreeze-{{py}}/{{VENV_BIN_DIR}}/{{PYTHON_EXE}} cx_Freeze
    @echo "\n>>> Building cx_Freeze executable..."
    @cd dev/cx_freeze && ../../.venv-cxfreeze-{{py}}/{{VENV_BIN_DIR}}/{{PYTHON_EXE}} cx_freeze_basic_standalone_check_builder.py build && cd ../..
    @echo "\n>>> Testing executable..."
    @./dev/cx_freeze/build/exe.*/basic_standalone_thermo_check{{EXE_SUFFIX}}
    @echo "\n>>> Cleaning up temporary environment..."
    @rm -rf .venv-cxfreeze-{{py}}
    @echo "✅ cx_Freeze test complete and cleaned up!"

## 🔥 test-nuitka: Test Nuitka compatibility (compile module and import it).
test-nuitka py="3.13":
    @echo ">>> Creating temporary virtual environment with Python {{py}}..."
    @uv venv .venv-nuitka-{{py}} --python {{py}}
    @echo "\n>>> Installing project and Nuitka in temporary environment..."
    @uv pip install --python .venv-nuitka-{{py}}/{{VENV_BIN_DIR}}/{{PYTHON_EXE}} -e .[test,numba]
    @uv pip install --python .venv-nuitka-{{py}}/{{VENV_BIN_DIR}}/{{PYTHON_EXE}} nuitka
    @echo "\n>>> Preparing build directory..."
    @mkdir -p dev/nuitka/build
    @cp -r thermo dev/nuitka/build/
    @echo "\n>>> Building Nuitka module..."
    @cd dev/nuitka/build && ../../../.venv-nuitka-{{py}}/{{VENV_BIN_DIR}}/{{PYTHON_EXE}} -m nuitka --module thermo --include-package=thermo
    @echo "\n>>> Removing original thermo folder from build directory..."
    @rm -rf dev/nuitka/build/thermo/thermo
    @echo "\n>>> Testing compiled module can be imported..."
    @cd dev/nuitka/build && ../../../.venv-nuitka-{{py}}/{{VENV_BIN_DIR}}/{{PYTHON_EXE}} -c "import thermo; print('Version:', thermo.__version__)"
    @echo "\n>>> Cleaning up temporary environment..."
    @rm -rf .venv-nuitka-{{py}}
    @echo "✅ Nuitka test complete and cleaned up!"

## 📦 test-pyinstaller: Test PyInstaller compatibility (build executable and run it).
test-pyinstaller py="3.13":
    @echo ">>> Creating temporary virtual environment with Python {{py}}..."
    @uv venv .venv-pyinstaller-{{py}} --python {{py}}
    @echo "\n>>> Installing project and PyInstaller in temporary environment..."
    @uv pip install --python .venv-pyinstaller-{{py}}/{{VENV_BIN_DIR}}/{{PYTHON_EXE}} .[test]
    @uv pip install --python .venv-pyinstaller-{{py}}/{{VENV_BIN_DIR}}/{{PYTHON_EXE}} pyinstaller
    @rm -rf build
    @echo "\n>>> Preparing build directory..."
    @mkdir -p dev/pyinstaller/build
    @echo "\n>>> Building PyInstaller executable..."
    @.venv-pyinstaller-{{py}}/{{VENV_BIN_DIR}}/pyinstaller{{EXE_SUFFIX}} dev/pyinstaller/pyinstaller_basic_standalone_check.spec --distpath dev/pyinstaller/build/dist --workpath dev/pyinstaller/build/build
    @echo "\n>>> Testing executable..."
    @./dev/pyinstaller/build/dist/basic_standalone_thermo_check{{EXE_SUFFIX}}
    @echo "\n>>> Cleaning up temporary environment..."
    @rm -rf .venv-pyinstaller-{{py}}
    @echo "✅ PyInstaller test complete and cleaned up!"

## 🌍 qemu-setup: Register QEMU interpreters for multi-arch container support.
qemu-setup:
    @command -v podman >/dev/null 2>&1 || { echo "❌ Error: podman is not installed. Please install podman first."; exit 1; }
    @echo ">>> Registering QEMU interpreters with binfmt_misc..."
    @podman run --rm --privileged multiarch/qemu-user-static --reset -p yes
    @echo "✅ QEMU multi-arch support enabled."

## 🎯 prepare-multiarch-image: Build and cache a multiarch image with dependencies (use: just prepare-multiarch-image <arch> <distro>).
prepare-multiarch-image arch distro="trixie":
    #!/usr/bin/env bash
    set -euo pipefail

    # Check for podman
    command -v podman >/dev/null 2>&1 || { echo "❌ Error: podman is not installed. Please install podman first."; exit 1; }

    # Tag for cached image
    tag="localhost/thermo-test-{{arch}}-{{distro}}:latest"

    # Check if image already exists
    if podman image exists "$tag" 2>/dev/null; then
        echo "✅ Image $tag already exists, skipping build."
        exit 0
    fi

    echo ">>> Building cached image for {{arch}} with {{distro}}..."

    # Map architecture to platform
    case "{{arch}}" in
        armv6)   platform="linux/arm/v6" ;;
        armv7)   platform="linux/arm/v7" ;;
        aarch64) platform="linux/arm64" ;;
        riscv64) platform="linux/riscv64" ;;
        s390x)   platform="linux/s390x" ;;
        ppc64le) platform="linux/ppc64le" ;;
        *) echo "Unknown architecture: {{arch}}"; exit 1 ;;
    esac

    # Map distro to base image (using slim variants for Debian/Ubuntu)
    case "{{distro}}" in
        trixie)         image="debian:trixie-slim" ;;
        ubuntu_latest)  image="ubuntu:latest" ;;
        ubuntu_devel)   image="ubuntu:devel" ;;
        alpine_latest)  image="alpine:latest" ;;
        *) echo "Unknown distro: {{distro}}"; exit 1 ;;
    esac

    echo "Platform: $platform, Image: $image"

    # Determine package manager and install commands
    if [[ "{{distro}}" == "alpine_latest" ]]; then
        install_cmd="apk update && apk add bash python3 py3-pip py3-scipy py3-matplotlib py3-numpy py3-pandas"
    else
        install_cmd="apt-get update && apt-get install -y liblapack-dev gfortran libgmp-dev libmpfr-dev libsuitesparse-dev ccache libmpc-dev python3 python3-pip python3-scipy python3-matplotlib python3-numpy python3-pandas"
    fi

    # Create a temporary Containerfile
    cat > /tmp/Containerfile.thermo.{{arch}}.{{distro}} << EOF
    FROM $image
    RUN $install_cmd
    EOF

    # Build the image with the specified platform
    podman build --platform "$platform" -t "$tag" -f /tmp/Containerfile.thermo.{{arch}}.{{distro}}

    # Clean up
    rm /tmp/Containerfile.thermo.{{arch}}.{{distro}}

    echo "✅ Cached image $tag built successfully!"

## 🔄 prepare-all-multiarch-images: Build all cached images for multiarch testing in parallel.
prepare-all-multiarch-images:
    #!/usr/bin/env bash
    set -euo pipefail

    # Check for GNU parallel
    command -v parallel >/dev/null 2>&1 || { echo "❌ Error: GNU parallel is not installed. Please install it (e.g., apt install parallel)."; exit 1; }

    echo ">>> Building all cached multiarch images in parallel (this will take a while)..."

    # Define all arch/distro combinations
    # riscv64 ubuntu_devel fails often, on github actions with Illegal Instruction
    combinations=(
        "armv6 trixie"
        "armv7 trixie"
        "aarch64 trixie"
        "riscv64 trixie"
        "s390x trixie"
        "ppc64le trixie"
        "armv7 ubuntu_latest"
        "aarch64 ubuntu_latest"
        "s390x ubuntu_latest"
        "ppc64le ubuntu_latest"
        # "riscv64 ubuntu_devel"
        "armv6 alpine_latest"
        "armv7 alpine_latest"
        "aarch64 alpine_latest"
        "riscv64 alpine_latest"
        "s390x alpine_latest"
        "ppc64le alpine_latest"
    )

    # Get number of CPU cores
    ncores=$(nproc)
    echo ">>> Using $ncores parallel jobs"

    # Run all builds in parallel with line-buffered output and keep going on failures
    failed=0
    printf '%s\n' "${combinations[@]}" | \
        parallel --line-buffer --keep-order --jobs "$ncores" --colsep ' ' \
        'echo ">>> Starting {1}/{2}" && just prepare-multiarch-image {1} {2} && echo "✅ Completed {1}/{2}" || (echo "❌ Failed: {1}/{2}" && exit 1)' \
        || failed=1

    echo ""
    if [ $failed -eq 0 ]; then
        echo "✅ All cached multiarch images built successfully!"
    else
        echo "⚠️  Some images failed to build. Check output above for details."
        exit 1
    fi

## 🏗️  test-arch: Run tests on a specific architecture (use: just test-arch <arch> <distro>).
## Note: This uses cached images built with prepare-multiarch-image for faster execution.
test-arch arch distro="trixie":
    #!/usr/bin/env bash
    set -euo pipefail

    # Check for podman
    command -v podman >/dev/null 2>&1 || { echo "❌ Error: podman is not installed. Please install podman first."; exit 1; }

    echo ">>> Running tests on {{arch}} with {{distro}}..."

    # Map architecture to platform
    case "{{arch}}" in
        armv6)   platform="linux/arm/v6" ;;
        armv7)   platform="linux/arm/v7" ;;
        aarch64) platform="linux/arm64" ;;
        riscv64) platform="linux/riscv64" ;;
        s390x)   platform="linux/s390x" ;;
        ppc64le) platform="linux/ppc64le" ;;
        *) echo "Unknown architecture: {{arch}}"; exit 1 ;;
    esac

    # Use cached image
    image="localhost/thermo-test-{{arch}}-{{distro}}:latest"
    echo "Platform: $platform, Image: $image"

    # Build image if it doesn't exist
    if ! podman image exists "$image" 2>/dev/null; then
        echo ">>> Image $image not found, building it now..."
        just prepare-multiarch-image {{arch}} {{distro}}
    fi

    # Determine pip flags
    if [[ "{{distro}}" == "alpine_latest" ]]; then
        pip_flags="--break-system-packages"
    else
        pip_flags="--break-system-packages"
    fi

    # Run the container with files copied (not mounted)
    # Note: Removed -it flag for CI compatibility, removed :Z flag for broader compatibility
    podman run --rm \
        --platform "$platform" \
        -v "$(pwd):/src:ro" \
        "$image" \
        bash -c "
            mkdir -p /workspace && \
            cd /src && \
            find . -mindepth 1 -maxdepth 1 ! -name '.*' -exec cp -r {} /workspace/ \; && \
            cd /workspace && \
            python3 -m pip install wheel $pip_flags && \
            pip3 install -e .[test-multiarch] $pip_flags && \
            python3 dev/dump_UNIFAC_assignments_to_sqlite.py && \
            python3 -m pytest . -v -m 'not online and not sympy and not numba and not CoolProp and not fuzz and not deprecated and not slow'
        "

    echo "✅ Tests on {{arch}} with {{distro}} complete!"

## 🌐 test-multiarch: Run tests on all architectures from CI (requires time!).
test-multiarch:
    @echo ">>> Running multi-arch tests (this will take a while)..."
    @echo "\n=== Debian Trixie ==="
    @just test-arch armv6 trixie || echo "❌ armv6/trixie failed"
    @just test-arch armv7 trixie || echo "❌ armv7/trixie failed"
    @just test-arch aarch64 trixie || echo "❌ aarch64/trixie failed"
    @just test-arch riscv64 trixie || echo "❌ riscv64/trixie failed"
    @just test-arch s390x trixie || echo "❌ s390x/trixie failed"
    @just test-arch ppc64le trixie || echo "❌ ppc64le/trixie failed"
    @echo "\n=== Ubuntu Latest ==="
    @just test-arch armv7 ubuntu_latest || echo "❌ armv7/ubuntu_latest failed"
    @just test-arch aarch64 ubuntu_latest || echo "❌ aarch64/ubuntu_latest failed"
    @just test-arch s390x ubuntu_latest || echo "❌ s390x/ubuntu_latest failed"
    @just test-arch ppc64le ubuntu_latest || echo "❌ ppc64le/ubuntu_latest failed"
    # @echo "\n=== Ubuntu Devel ==="
    # @just test-arch riscv64 ubuntu_devel || echo "❌ riscv64/ubuntu_devel failed"
    @echo "\n=== Alpine Latest ==="
    @just test-arch armv6 alpine_latest || echo "❌ armv6/alpine_latest failed"
    @just test-arch armv7 alpine_latest || echo "❌ armv7/alpine_latest failed"
    @just test-arch aarch64 alpine_latest || echo "❌ aarch64/alpine_latest failed"
    @just test-arch riscv64 alpine_latest || echo "❌ riscv64/alpine_latest failed"
    @just test-arch s390x alpine_latest || echo "❌ s390x/alpine_latest failed"
    @just test-arch ppc64le alpine_latest || echo "❌ ppc64le/alpine_latest failed"
    @echo "\n✅ Multi-arch testing complete!"

## 🧬 test-multi-single: Test with specific Python/NumPy/SciPy versions (e.g., just test-multi-single 3.9 1.26.4 1.12.0).
## Set KEEP_VENV=1 to keep the virtual environment for debugging (e.g., KEEP_VENV=1 just test-multi-single 3.9 1.26.4 1.12.0).
test-multi-single py="3.10" numpy="2.0.1" scipy="1.14.0":
    @echo ">>> Testing Python {{py}}, NumPy {{numpy}}, SciPy {{scipy}}..."
    @echo ">>> Installing Python {{py}} if needed..."
    @uv python install {{py}} || true
    @echo ">>> Creating temporary virtual environment..."
    @uv venv .venv-test-python{{py}}-numpy{{numpy}}-scipy{{scipy}} --python {{py}}
    @echo ">>> Installing dependencies..."
    @uv pip install --python .venv-test-python{{py}}-numpy{{numpy}}-scipy{{scipy}}/bin/python -e .[test]
    @uv pip install --python .venv-test-python{{py}}-numpy{{numpy}}-scipy{{scipy}}/bin/python "numpy=={{numpy}}" "scipy=={{scipy}}"
    @echo ">>> Installing numba..."
    @uv pip install --python .venv-test-python{{py}}-numpy{{numpy}}-scipy{{scipy}}/bin/python -e .[numba] || echo "⚠️  Numba install failed, continuing..."
    @echo ">>> Generating UNIFAC database files..."
    @.venv-test-python{{py}}-numpy{{numpy}}-scipy{{scipy}}/bin/python dev/dump_UNIFAC_assignments_to_sqlite.py
    @echo ">>> Running tests (no coverage)..."
    @.venv-test-python{{py}}-numpy{{numpy}}-scipy{{scipy}}/bin/pytest . -m "not online and not sympy and not numba and not CoolProp and not fuzz and not deprecated and not slow"
    @if [ -z "$${KEEP_VENV}" ]; then \
        echo ">>> Cleaning up temporary environment..."; \
        rm -rf .venv-test-python{{py}}-numpy{{numpy}}-scipy{{scipy}}; \
    else \
        echo ">>> Keeping venv .venv-test-python{{py}}-numpy{{numpy}}-scipy{{scipy}} for debugging (KEEP_VENV is set)"; \
    fi
    @echo "✅ Test complete for Python {{py}}, NumPy {{numpy}}, SciPy {{scipy}}!"

## 🧬 test-multi: Run all Python/NumPy/SciPy combinations from CI locally.
test-multi:
    #!/usr/bin/env bash
    set -euo pipefail

    # Check for GNU parallel
    command -v parallel >/dev/null 2>&1 || { echo "❌ Error: GNU parallel is not installed. Please install it (e.g., apt install parallel)."; exit 1; }

    echo ">>> Running multi-version tests (this will take a while)..."
    echo ">>> This mirrors the CI matrix from build_multi_numpy_scipy.yml"

    # Define all Python/NumPy/SciPy combinations
    combinations=(
        "3.10 1.24.4 1.9.3"
        "3.10 1.24.4 1.12.0"
        "3.9 1.24.4 1.12.0"
        "3.9 1.26.4 1.10.1"
        "3.9 1.26.4 1.12.0"
        "3.10 1.26.4 1.14.0"
        "3.10 2.0.1 1.14.0"
    )

    # Get number of CPU cores
    ncores=$(nproc)
    echo ">>> Using $ncores parallel jobs"
    echo ""

    # Run all tests in parallel with line-buffered output and keep going on failures
    failed=0
    printf '%s\n' "${combinations[@]}" | \
        parallel --line-buffer --keep-order --jobs "$ncores" --colsep ' ' \
        'echo ">>> Starting Python {1}, NumPy {2}, SciPy {3}" && just test-multi-single {1} {2} {3} && echo "✅ Completed Python {1}, NumPy {2}, SciPy {3}" || (echo "❌ Failed: Python {1}, NumPy {2}, SciPy {3}" && exit 1)' \
        || failed=1

    echo ""
    if [ $failed -eq 0 ]; then
        echo "✅ All multi-version tests passed!"
    else
        echo "⚠️  Some tests failed. Check output above for details."
        exit 1
    fi

## 🧹 clean: Remove build artifacts and Python caches.
clean:
    @echo ">>> Cleaning up build artifacts and cache files..."
    @rm -rf _build .mypy_cache .pytest_cache dist *.egg-info htmlcov prof dev/cx_freeze/build dev/nuitka/build dev/pyinstaller/build .venv-cxfreeze-* .venv-nuitka-* .venv-pyinstaller-*
    @rm -rf .venv-test-*
    @rm -f thermo.*.so thermo.*.pyd
    @rm -f "thermo/Phase Change/DDBST_UNIFAC_assignments.sqlite"
    @find . -type d -name "__pycache__" -exec rm -rf {} +
    @echo "✅ Cleanup complete."

## 🐳 clean-multiarch-images: Remove all cached multiarch container images.
clean-multiarch-images:
    #!/usr/bin/env bash
    set -euo pipefail

    # Check for podman
    command -v podman >/dev/null 2>&1 || { echo "❌ Error: podman is not installed. Please install podman first."; exit 1; }

    echo ">>> Removing cached multiarch container images..."

    # Find all images matching our naming pattern
    images=$(podman images --format "{{{{.Repository}}}}:{{{{.Tag}}}}" | grep "^thermo-test-" || true)

    if [ -z "$images" ]; then
        echo "✅ No multiarch images found to remove."
        exit 0
    fi

    removed=0
    while IFS= read -r img; do
        echo "  Removing $img..."
        podman rmi "$img" 2>/dev/null || echo "  ⚠️  Failed to remove $img"
        ((removed++))
    done <<< "$images"

    echo ""
    echo "✅ Removed $removed multiarch image(s)."

## 💣 nuke: Remove the virtual environment and all build artifacts.
nuke: clean
    @echo ">>> Removing all virtual environments..."
    @rm -rf .venv*
    @echo "✅ Project completely cleaned."