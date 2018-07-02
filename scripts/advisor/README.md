Example runs:

* python run_advisor.py --name isotropic --path <path-to-devito>/examples/seismic/acoustic/acoustic_example.py
* python run_advisor.py --name tti_so8 --path <path-to-devito>/examples/seismic/tti/tti_example.py --exec-args "-so 8"

Limitations:

* Support guaranteed only for Intel Advisor 2018 version 3; earlier years won't
  work; other 2018 versions, as well as later years, may or may not work.
* `numactl` must be available on the system.
* Untested with more complicated examples.
* Untested on Intel KNL (we might need to ask `numactl` to bind to MCDRAM).
* Running the `tripcounts` analysis takes a lot, despite starting in paused
  mode. This analysis, together with the `survey` analysis, is necessary to
  generate a roofline. Both are run by `run_advisor.py`.

TODO:

* Give a name to the points in the roofline, otherwise it's challenging to
  relate loops (code sections) to data.
* Emit a report summarizing the configuration used to run the analysis
  (threading, socket binding, ...).
* Intel Advisor's Python API currently (v 2018.3) only supports Python 2.7.
  When generating a roofline via `run_advisor.py` (by default, `run_advisor.py`
  runs with `--plot`), it is expected that `python2.7` be in `$PATH`.
