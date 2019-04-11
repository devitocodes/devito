#!/usr/bin/env python

import os
import subprocess

err = []


def runStep(command, *args, **kwargs):
    envCmd = kwargs.get('envCmd', "source activate devito; ")
    global err
    err.append(subprocess.run(
        (envCmd + command),
        shell=True).returncode)


if os.environ.get('testWithPip') == 'true':
    runStep("python setup.py test", envCmd="")

if os.environ.get('testWithPip') != 'true':
    runStep("flake8 --exclude .conda,.git --builtins=ArgumentError .")
    runStep("py.test --durations=20 --cov devito tests/")
    if os.environ.get('RUN_EXAMPLES') == 'true':
        runStep(("python benchmarks/user/benchmark.py test " +
                 "-P tti -so 4 -a -d 20 20 20 -n 5"))
        runStep("python benchmarks/user/benchmark.py test -P acoustic -a")
        runStep("python examples/seismic/acoustic/acoustic_example.py --full")
        runStep(("python examples/seismic/acoustic/acoustic_example.py " +
                "--full --checkpointing"))
        runStep("python examples/seismic/acoustic/acoustic_example.py --constant --full")
        runStep("python examples/misc/linalg.py mat-vec mat-mat-sum transpose-mat-vec")
        runStep("python examples/seismic/tti/tti_example.py -a")
        runStep("python examples/seismic/tti/tti_example.py -a --noazimuth")
        runStep("python examples/seismic/elastic/elastic_example.py")
        runStep("py.test --nbval examples/cfd")
        runStep("py.test --nbval examples/seismic/tutorials/0[1-3]*")
        runStep("py.test --nbval examples/compiler")
        runStep("codecov")
    runStep("pushd docs; make html; popd")

exit(sum(err))
