import versioneer

import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('requirements-optional.txt') as f:
    optionals = f.read().splitlines()

reqs = []
for ir in required:
    if ir[0:3] == 'git':
        name = ir.split('/')[-1]
        reqs += ['%s @ %s@master' % (name, ir)]
    else:
        reqs += [ir]

opt_reqs = []
extras_require = {}
for ir in optionals:
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
extras_require['extras'] = opt_reqs

# If interested in benchmarking devito, we need the `examples` too
exclude = ['docs', 'tests']
try:
    if not bool(int(os.environ.get('DEVITO_BENCHMARKS', 0))):
        exclude += ['examples']
except (TypeError, ValueError):
    exclude += ['examples']

setup(name='devito',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description="Finite Difference DSL for symbolic computation.",
      long_description="""Devito is a new tool for performing
      optimised Finite Difference (FD) computation from high-level
      symbolic problem definitions. Devito performs automated code
      generation and Just-In-time (JIT) compilation based on symbolic
      equations defined in SymPy to create and execute highly
      optimised Finite Difference kernels on multiple computer
      platforms.""",
      url='http://www.devitoproject.org',
      author="Imperial College London",
      author_email='g.gorman@imperial.ac.uk',
      license='MIT',
      packages=find_packages(exclude=exclude),
      install_requires=reqs,
      extras_require=extras_require,
      test_suite='tests')
