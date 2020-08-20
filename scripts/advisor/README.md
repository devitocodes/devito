Example runs:

* python3 run_advisor.py --name isotropic --path <path-to-devito>/examples/seismic/acoustic/acoustic_example.py
* python3 run_advisor.py --name tti_so8 --path <path-to-devito>/examples/seismic/tti/tti_example.py --exec-args "-so 8"
* python3 run_advisor.py --name iso_ac_so6 --path <path-to-devito>/benchmarks/user/benchmark.py --exec-args "bench -P acoustic -so 6 --tn 200 -d 100 100 100 --autotune off -x 1"

After the run has finished you should be able to plot a roofline with the results using:
* python3 roofline.py --name Roofline --project <advisor-project-name>

Prerequisites:
* Support guaranteed only for Intel Advisor 2020 Update 2; earlier years may not
  work; other 2020 versions, as well as later years, may or may not work.
* `numactl` must be available on the system. If not available, install with:
	`sudo apt-get install numactl`
* Install `pandas`. `pandas` is not included in the core Devito installation.

Limitations:

* Untested with more complicated examples.
* Untested on Intel KNL (we might need to ask `numactl` to bind to MCDRAM).
* Running the `tripcounts` analysis takes a lot, despite starting in paused
  mode. This analysis, together with the `survey` analysis, is necessary to
  generate a roofline. Both are run by `run_advisor.py`.
* Requires python3, untested in earlier versions of python

TODO:

* Give a name to the points in the roofline, otherwise it's challenging to
  relate loops (code sections) to data.
* Emit a report summarizing the configuration used to run the analysis
  (threading, socket binding, ...).
