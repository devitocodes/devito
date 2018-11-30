#!/usr/bin/env python

import os
import subprocess

err=[]

if os.environ.get('testWithPip') == 'true':
    err.append(subprocess.run("python setup.py test", shell=True).returncode)

if os.environ.get('testWithPip') != 'true':
    err.append(subprocess.run("source activate devito; py.test --cov devito tests/", shell=True).returncode)
    if os.environ.get('DEVITO_BACKEND') == 'core':
        err.append(subprocess.run("source activate devito; python examples/seismic/benchmark.py test -P tti -so 4 -a -d 20 20 20 -n 5", shell=True).returncode)
        err.append(subprocess.run("source activate devito; python examples/seismic/benchmark.py test -P acoustic -a", shell=True).returncode)
        err.append(subprocess.run("source activate devito; python examples/seismic/acoustic/acoustic_example.py --full", shell=True).returncode)
        err.append(subprocess.run("source activate devito; python examples/seismic/acoustic/acoustic_example.py --full --checkpointing", shell=True).returncode)
        err.append(subprocess.run("source activate devito; python examples/seismic/acoustic/acoustic_example.py --constant --full", shell=True).returncode)
        err.append(subprocess.run("source activate devito; python examples/misc/linalg.py mat-vec mat-mat-sum transpose-mat-vec", shell=True).returncode)
        err.append(subprocess.run("source activate devito; python examples/seismic/tti/tti_example.py -a", shell=True).returncode)
        err.append(subprocess.run("source activate devito; python examples/seismic/tti/tti_example.py -a --noazimuth", shell=True).returncode)
        err.append(subprocess.run("source activate devito; python examples/seismic/elastic/elastic_example.py", shell=True).returncode)
        err.append(subprocess.run("source activate devito; py.test --nbval examples/cfd", shell=True).returncode)
        err.append(subprocess.run("source activate devito; py.test --nbval examples/seismic/tutorials/0[1-3]*", shell=True).returncode)
        err.append(subprocess.run("source activate devito; py.test --nbval examples/compiler", shell=True).returncode)
        err.append(subprocess.run("source activate devito; codecov", shell=True).returncode)
    err.append(subprocess.run("source activate devito; sphinx-apidoc -f -o docs/ examples", shell=True).returncode)
    err.append(subprocess.run("source activate devito; sphinx-apidoc -f -o docs/ devito devito/yask/*", shell=True).returncode)
    err.append(subprocess.run("source activate devito; pushd docs; make html; popd", shell=True).returncode)

exit(sum(err))
