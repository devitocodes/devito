import versioneer
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(name='devito',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description="Finite Difference DSL for symbolic computation.",
      long_descritpion="""Devito is a new tool for performing
      optimised Finite Difference (FD) computation from high-level
      symbolic problem definitions. Devito performs automated code
      generation and Just-In-time (JIT) compilation based on symbolic
      equations defined in SymPy to create and execute highly
      optimised Finite Difference kernels on multiple computer
      platforms.""",
      url='http://www.opesci.org/devito',
      author="Imperial College London",
      author_email='opesci@imperial.ac.uk',
      license='MIT',
      packages=find_packages(exclude=['docs', 'examples', 'tests']),
      install_requires=['numpy', 'sympy>=1.1', 'mpmath', 'cgen', 'codepy',
                        'psutil', 'py-cpuinfo', 'cached-property', 'multidict'],
      test_requires=['pytest', 'flake8', 'isort'])
