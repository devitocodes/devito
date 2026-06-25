# Devito Copilot Instructions

## Repository Overview

Devito is a Python package for implementing optimized stencil computation (finite differences, image processing, machine learning) from high-level symbolic problem definitions. It builds on SymPy and employs automated code generation and just-in-time compilation to execute optimized computational kernels on CPUs, GPUs, and clusters.

**Key Facts:**
- Production-stable package (Development Status :: 5 - Production/Stable)
- Python 3.10-3.13 support with comprehensive CI testing
- ~290MB repository with extensive test coverage and benchmarking
- Primary languages: Python (core), C/C++ (generated code)
- Main dependencies: NumPy, SymPy, cgen, codepy, psutil
- Supports multiple compilation backends: C, OpenMP, OpenACC, MPI
- Active development with 600+ stars, 238+ forks, and regular releases

## Build and Development Environment

### Prerequisites and Installation

**System Dependencies (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y python3-dev python3-pip python3-venv build-essential
```

**Development Installation (preferred method):**
```bash
# Create and activate virtual environment
python3 -m venv devito-env
source devito-env/bin/activate

# Install with all dependencies - requires network access
pip install -e .[tests,extras]

# Alternative: conda environment (recommended for full development)
conda env create -f environment-dev.yml
conda activate devito
```

**Docker Development (if dependencies fail):**
```bash
docker-compose up devito  # Starts Jupyter on port 8888
# OR
docker build . --file docker/Dockerfile.devito --tag devito_img
```

### Environment Variables (Critical for Compilation)
```bash
export DEVITO_ARCH="gcc"           # gcc, clang, icx, custom
export DEVITO_LANGUAGE="openmp"    # C, openmp, CXX, CXXopenmp
export OMP_NUM_THREADS=2           # For parallel tests
```

### Build and Test Commands

**Linting (ALWAYS run before committing):**
```bash
flake8 --builtins=ArgumentError .
# Configuration in pyproject.toml: max-line-length=90, specific ignore rules
```

**Testing:**
```bash
# Core tests (required, ~10-15 minutes)
pytest -k "not adjoint" -m "not parallel" --maxfail=5 tests/

# Adjoint tests (advanced feature tests)
pytest -k "adjoint" -m "not parallel" tests/

# Example tests (integration validation)
pytest examples/

# Specific test file
pytest tests/test_operator.py -v
```

**Configuration Check:**
```bash
python3 -c "from devito import configuration; print(''.join(['%s: %s \n' % (k, v) for (k, v) in configuration.items()]))"
```

### Common Build Issues and Solutions

1. **Import errors for cgen/codepy:** Install via pip or use conda environment
2. **Compiler not found:** Set DEVITO_ARCH to match installed compiler
3. **OpenMP issues on macOS:** Install with `brew install llvm libomp`
4. **Network timeouts during pip install:** Use conda or Docker alternatives
5. **Permission issues:** Use `--user` or `--break-system-packages` flags

## Project Architecture and Layout

### Core Directory Structure
```
devito/                 # Main source package
├── __init__.py        # Package initialization, configuration setup
├── arch/              # Architecture-specific compilation backends
├── builtins/          # Built-in mathematical functions
├── checkpointing/     # Checkpointing and reverse-mode functionality
├── core/              # Core IR (Intermediate Representation) classes
├── data/              # Data structures and memory management
├── finite_differences/# Finite difference operators and stencils
├── ir/                # Intermediate representation passes
├── mpatches/          # Memory patches and optimizations
├── mpi/               # MPI parallelization support
├── operations/        # Mathematical operations and transformations
├── operator/          # Operator compilation and generation
├── passes/            # Compiler optimization passes
├── symbolics/         # Symbolic computation utilities
├── tools/             # Utility functions and helpers
├── types/             # Core types (Grid, Function, Equation, etc.)
├── deprecations.py    # Deprecation warnings and legacy support
└── warnings.py        # Warning system and configuration

tests/                 # Test suite (comprehensive coverage)
examples/              # Tutorials and demonstrations
├── seismic/          # Seismic modeling examples
├── cfd/              # Computational fluid dynamics
├── misc/             # Miscellaneous examples
└── userapi/          # User API demonstrations

scripts/               # Utility scripts
docker/               # Docker configurations  
benchmarks/           # Performance benchmarks and ASV configuration
binder/               # Jupyter Binder configuration
```

### Key Configuration Files
- `pyproject.toml`: Build configuration, dependencies, flake8 settings (max-line-length=90)
- `pytest.ini`: Test configuration and options
- `requirements*.txt`: Dependency specifications for different environments (mpi, nvidia, testing, optional)
- `environment-dev.yml`: Conda environment specification (recommended for development)
- `.github/workflows/`: CI/CD pipelines (13 comprehensive workflows)
- `MANIFEST.in`: Package distribution file inclusion/exclusion rules

## Continuous Integration and Validation

### GitHub Workflows (All Must Pass)
1. **pytest-core-nompi.yml**: Core functionality tests across Python 3.10-3.13, multiple GCC versions
2. **pytest-core-mpi.yml**: MPI parallel execution tests  
3. **pytest-gpu.yml**: GPU computation tests (OpenACC/CUDA)
4. **flake8.yml**: Code style validation (90-char line limit, specific ignore rules)
5. **examples.yml**: Integration tests via example execution
6. **examples-mpi.yml**: MPI-based example validation
7. **tutorials.yml**: Jupyter notebook validation
8. **asv.yml**: Performance benchmarking and regression detection
9. **docker-bases.yml**: Docker base image builds for testing environments
10. **docker-devito.yml**: Devito-specific Docker image builds
11. **pythonpublish.yml**: Package publishing to PyPI
12. **release-notes.yml**: Automated release documentation
13. **triggers.yml**: Workflow orchestration and triggers

### Pre-commit Validation Steps
```bash
# 1. Lint code
flake8 --builtins=ArgumentError .

# 2. Run core tests
pytest -k "not adjoint" -m "not parallel" --maxfail=5 tests/

# 3. Check configuration
python3 -c "from devito import configuration; print(configuration)"

# 4. Test example (optional but recommended)
python examples/seismic/acoustic/acoustic_example.py
```

### Test Execution Times
- Core tests: 10-15 minutes
- Full test suite: 30-45 minutes
- Individual test files: 1-5 minutes
- Linting: <1 minute
- Examples: 5-10 minutes each

## Development Guidelines

### Code Style (Enforced by CI)
- **Line length**: 90 characters maximum (configured in pyproject.toml)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Grouped (stdlib, third-party, devito) and alphabetically sorted
- **Docstrings**: NumPy-style documentation required for public APIs
- **Naming**: CamelCase classes, snake_case methods/variables

### Commit Messages and PR Titles
Use descriptive prefixes to categorize changes (based on recent repository patterns).

**Guidelines:**
- Capitalize the title after the prefix (e.g., `misc: Patch` not `misc: patch`)
- Follow the official guide: https://github.com/devitocodes/devito/wiki/Tags-for-commit-messages-and-PR-titles

**Common prefixes:**
- **compiler:** Changes to compilation system, IR passes, code generation
- **mpi:** MPI-related functionality and parallel execution
- **misc:** General improvements, cleanup, modernization
- **docs:** Documentation updates, docstring fixes
- **install:** Installation, packaging, dependencies
- **bug-py:** Python code bug fixes
- **bug-C:** Generated C code bug fixes

**Examples:**
- `compiler: Hotfix estimate memory for certain devices`
- `misc: switch to f-string throughout`
- `docs: fix mmin/mmax docstring`
- `install: Remove files and directories not needed for install`

### Common Labels (for PR categorization)
- **bug-py**, **bug-py-minor**, **bug-C**: Bug fixes by category
- **compiler**: Compilation system changes
- **MPI**: MPI and parallel execution
- **documentation**: Documentation improvements
- **installation**: Package installation and setup
- **misc**: General improvements
- **dependencies**: Dependency updates
- **API**: Application Programming Interface changes

### Making Changes
1. **Always run linting first**: `flake8 --builtins=ArgumentError .`
2. **Test specific areas**: Run relevant tests before committing
3. **Environment setup**: Ensure DEVITO_ARCH and DEVITO_LANGUAGE are set
4. **Documentation**: Update docstrings for API changes
5. **Examples**: Add/update examples for new features

### Critical Dependencies
- **SymPy**: Symbolic mathematics (versions 1.12+ supported)
- **NumPy**: Numerical arrays (version 2.x compatible)
- **cgen/codepy**: C code generation and compilation
- **psutil**: System monitoring and resource management
- **setuptools-scm**: Dynamic versioning from git tags

### Flake8 Configuration and Common Workarounds
The project uses flake8 with specific ignore rules (configured in pyproject.toml):
- **F403/F405**: Star imports are expected in `__init__.py` files due to API design
- **E731**: Lambda expressions are acceptable for configuration and simple cases
- **W503**: Line breaks before binary operators (older style preference)
- **E722**: Bare except clauses are sometimes necessary for compatibility
- **W605**: Invalid escape sequences in docstrings (legacy patterns)

**Import order**: Follow the pattern in `devito/__init__.py` for consistency

## Agent-Specific Instructions

**Trust these instructions** - they are validated and current. Only search/explore if:
1. Instructions appear outdated or incorrect
2. New functionality is not covered
3. Specific error messages require investigation

**Time-saving tips:**
1. Use Docker if pip installation fails due to network issues  
2. Set environment variables before running any compilation-dependent tests
3. Start with small test files when debugging (e.g., `pytest tests/test_operator.py::test_specific`)
4. Check configuration output to verify environment setup
5. Use `--maxfail=5` to stop test runs early on systematic failures
6. Use python venv for full development workflow when possible
7. Run flake8 first before making commits to catch style issues early

**File modification priorities:**
1. Core types and operators: `devito/types/`, `devito/operator/`
2. Mathematical operations: `devito/finite_differences/`, `devito/symbolics/`
3. Compilation backends: `devito/arch/`, `devito/passes/`
4. MPI functionality: `devito/mpi/`
5. Always update corresponding tests in `tests/`
6. Update documentation in examples/ if adding new features