# Devito Copilot Instructions

## Repository Overview

Devito is a Python package for implementing optimized stencil computation (finite differences, image processing, machine learning) from high-level symbolic problem definitions. It builds on SymPy and employs automated code generation and just-in-time compilation to execute optimized computational kernels on CPUs, GPUs, and clusters.

**Key Facts:**
- Production-stable package (Development Status 5) for scientific computing
- Python 3.10-3.13 support (current development uses 3.12.3)
- ~50MB repository with extensive test coverage
- Primary languages: Python (core), C/C++ (generated code)
- Main dependencies: NumPy, SymPy, cgen, codepy, psutil
- Supports multiple compilation backends: C, OpenMP, OpenACC, MPI

## Build and Development Environment

### Prerequisites and Installation

**System Dependencies (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y python3-numpy python3-sympy python3-psutil python3-pip
sudo apt install -y gcc g++ flake8
```

**Development Installation (preferred method):**
```bash
# Install with all dependencies - requires network access
pip install -e .[tests,extras] --break-system-packages

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
├── core/              # Core IR (Intermediate Representation) classes
├── data/              # Data structures and memory management
├── finite_differences/# Finite difference operators and stencils
├── ir/                # Intermediate representation passes
├── mpi/               # MPI parallelization support
├── operator/          # Operator compilation and generation
├── passes/            # Compiler optimization passes
├── symbolics/         # Symbolic computation utilities
├── tools/             # Utility functions and helpers
└── types/             # Core types (Grid, Function, Equation, etc.)

tests/                 # Test suite (comprehensive coverage)
examples/              # Tutorials and demonstrations
├── seismic/          # Seismic modeling examples
├── cfd/              # Computational fluid dynamics
├── misc/             # Miscellaneous examples
└── userapi/          # User API demonstrations

scripts/               # Utility scripts
docker/               # Docker configurations
benchmarks/           # Performance benchmarks
```

### Key Configuration Files
- `pyproject.toml`: Build configuration, dependencies, flake8 settings
- `pytest.ini`: Test configuration and options
- `requirements*.txt`: Dependency specifications for different environments
- `environment-dev.yml`: Conda environment specification
- `.github/workflows/`: CI/CD pipelines

## Continuous Integration and Validation

### GitHub Workflows (All Must Pass)
1. **pytest-core-nompi.yml**: Core functionality tests across Python 3.10-3.13, multiple GCC versions
2. **pytest-core-mpi.yml**: MPI parallel execution tests
3. **pytest-gpu.yml**: GPU computation tests (OpenACC/CUDA)
4. **flake8.yml**: Code style validation
5. **examples.yml**: Integration tests via example execution
6. **tutorials.yml**: Jupyter notebook validation

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
- **Line length**: 90 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Grouped (stdlib, third-party, devito) and alphabetically sorted
- **Docstrings**: NumPy-style documentation required for public APIs
- **Naming**: CamelCase classes, snake_case methods/variables

### Making Changes
1. **Always run linting first**: `flake8 --builtins=ArgumentError .`
2. **Test specific areas**: Run relevant tests before committing
3. **Environment setup**: Ensure DEVITO_ARCH and DEVITO_LANGUAGE are set
4. **Documentation**: Update docstrings for API changes
5. **Examples**: Add/update examples for new features

### Critical Dependencies
- **SymPy**: Symbolic mathematics (versions 1.12-1.14 supported)
- **NumPy**: Numerical arrays (version 2.x)
- **cgen/codepy**: C code generation and compilation
- **psutil**: System monitoring and resource management

### Common Workarounds
- **Star imports**: F403/F405 warnings are expected in `__init__.py` due to API design
- **Lambda expressions**: E731 warnings are sometimes acceptable for configuration
- **Import order**: Follow the pattern in `devito/__init__.py`

## Agent-Specific Instructions

**Trust these instructions** - they are validated and current. Only search/explore if:
1. Instructions appear outdated or incorrect
2. New functionality is not covered
3. Specific error messages require investigation

**Time-saving tips:**
1. Use Docker if pip installation fails due to network issues
2. Set environment variables before running any compilation-dependent tests
3. Start with small test files when debugging
4. Check configuration output to verify environment setup
5. Use `--maxfail=5` to stop test runs early on systematic failures

**File modification priorities:**
1. Core types and operators: `devito/types/`, `devito/operator/`
2. Mathematical operations: `devito/finite_differences/`, `devito/symbolics/`
3. Compilation backends: `devito/arch/`, `devito/passes/`
4. Always update corresponding tests in `tests/`