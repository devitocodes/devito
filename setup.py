import versioneer

import os
try:
    import importlib.metadata as metadata
    get_version = lambda x: metadata.version(x)
    PkgNotFound = metadata.PackageNotFoundError
    parse_version = lambda x: metadata.version(x)
except ImportError:
    import pkg_resources
    get_version = lambda x: pkg_resources.get_distribution(x).version
    PkgNotFound = pkg_resources.DistributionNotFound
    parse_version = lambda x: pkg_resources.parse_version(x)

from setuptools import setup, find_packages


def min_max(pkgs, pkg_name):
    pkg = [p for p in pkgs if pkg_name in p][0]
    minsign = '>=' if '>=' in pkg else '>'
    maxsign = '<=' if '<=' in pkg else '<'
    vmin = pkg.split(minsign)[1].split(',')[0]
    vmax = pkg.split(maxsign)[-1]
    return vmin, vmax


def numpy_compat(required):
    new_reqs = [r for r in required if "numpy" not in r and "sympy" not in r]
    sympy_lb, sympy_ub = min_max(required, "sympy")
    numpy_lb, numpy_ub = min_max(required, "numpy")

    # Due to api changes in numpy 2.0, it requires sympy 1.12.1 at the minimum
    # Check if sympy is installed and enforce numpy version accordingly.
    # If sympy isn't installed, enforce sympy>=1.12.1 and numpy>=2.0
    try:
        sympy_version = get_version("sympy")
        min_ver2 = parse_version("1.12.1")
        if parse_version(sympy_version) < min_ver2:
            new_reqs.extend([f"numpy>{numpy_lb},<2.0", f"sympy=={sympy_version}"])
        else:
            new_reqs.extend([f"numpy>=2.0,<{numpy_ub}", f"sympy=={sympy_version}"])
    except PkgNotFound:
        new_reqs.extend([f"sympy>=1.12.1,<{sympy_ub}", f"numpy>=2.0,<{numpy_ub}"])

    return new_reqs


with open('requirements.txt') as f:
    required = f.read().splitlines()
    required = numpy_compat(required)

with open('requirements-optional.txt') as f:
    optionals = f.read().splitlines()

with open('requirements-testing.txt') as f:
    testing = f.read().splitlines()

with open('requirements-mpi.txt') as f:
    mpis = f.read().splitlines()

with open('requirements-nvidia.txt') as f:
    nvidias = f.read().splitlines()

reqs = []
for ir in required:
    if ir[0:3] == 'git':
        name = ir.split('/')[-1]
        reqs += ['%s @ %s@master' % (name, ir)]
    else:
        reqs += [ir]

extras_require = {}
for mreqs, mode in (zip([optionals, mpis, nvidias, testing],
                        ['extras', 'mpi', 'nvidia', 'tests'])):
    opt_reqs = []
    for ir in mreqs:
        # For conditionals like pytest=2.1; python == 3.6
        if ';' in ir:
            entries = ir.split(';')
            extras_require[entries[1]] = entries[0]
        # Git repos, install master
        if ir[0:3] == 'git':
            name = ir.split('/')[-1]
            opt_reqs += ['%s @ %s@master' % (name, ir)]
        else:
            opt_reqs += [ir]
    extras_require[mode] = opt_reqs

# If interested in benchmarking devito, we need the `examples` too
exclude = ['docs', 'tests']
try:
    if not bool(int(os.environ.get('DEVITO_BENCHMARKS', 0))):
        exclude += ['examples']
    else:
        required += testing
except (TypeError, ValueError):
    exclude += ['examples']

setup(name='devito',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description="Finite Difference DSL for symbolic computation.",
      long_description="""
      Devito is a tool for performing optimised Finite Difference (FD)
      computation from high-level symbolic problem definitions. Devito
      performs automated code generation and Just-In-time (JIT) compilation
      based on symbolic equations defined in SymPy to create and execute highly
      optimised Finite Difference stencil kernels on multiple computer platforms.""",
      project_urls={
          'Documentation': 'https://www.devitoproject.org/devito/index.html',
          'Source Code': 'https://github.com/devitocodes/devito',
          'Issue Tracker': 'https://github.com/devitocodes/devito/issues',
      },
      url='http://www.devitoproject.org',
      platforms=["Linux", "Mac OS-X", "Unix"],
      python_requires=">=3.9",
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'License :: OSI Approved :: MIT License',
          'Operating System :: MacOS',
          'Operating System :: POSIX :: Linux',
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: Implementation',
          'Programming Language :: C',
          'Programming Language :: C++',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Software Development :: Code Generators',
          'Topic :: Software Development :: Compilers'],
      test_suite='pytest',
      author="Imperial College London",
      author_email='g.gorman@imperial.ac.uk',
      license='MIT',
      packages=find_packages(exclude=exclude),
      install_requires=reqs,
      extras_require=extras_require)
