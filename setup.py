import versioneer

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('requirements-optional.txt') as f:
    optionals = f.read().splitlines()

reqs = []
links = []
for ir in required:
    if ir[0:3] == 'git':
        links += [ir + '#egg=' + ir.split('/')[-1] + '-0']
        reqs += [ir.split('/')[-1]]
    else:
        reqs += [ir]

opt_reqs = []
opt_links = []
extras_require = {}
for ir in optionals:
    # For conditionals like pytest=2.1; python == 3.6
    if ';' in ir:
        entries = ir.split(';')
        extras_require[entries[1]] = entries[0]
    # Git repos
    # This is still a bit of an issue because it can only catch
    # version not master
    if ir[0:3] == 'git':
        opt_links += [ir + '#egg=' + ir.split('/')[-1] + '-0']
        opt_reqs += [ir.split('/')[-1]]
    else:
        opt_reqs += [ir]
extras_require['extras'] = opt_reqs

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
      author_email='opesci@imperial.ac.uk',
      license='MIT',
      packages=find_packages(exclude=['docs', 'tests', 'examples']),
      install_requires=reqs,
      extras_require=extras_require,
      dependency_links=links,
      test_suite='tests')
