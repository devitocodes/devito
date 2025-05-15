import os
from setuptools import setup, find_packages
import versioneer


def load_requirements(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


reqs = load_requirements('requirements.txt')
extras_require = {
    'mpi': load_requirements('requirements-mpi.txt'),
    'nvidia': load_requirements('requirements-nvidia.txt'),
    'tests': load_requirements('requirements-testing.txt'),
    'extras': load_requirements('requirements-optional.txt'),
}

# If interested in benchmarking devito, we need the `examples` too
exclude = ['docs', 'tests']
try:
    if not bool(int(os.environ.get('DEVITO_BENCHMARKS', 0))):
        exclude += ['examples']
    else:
        reqs += extras_require['tests']
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
