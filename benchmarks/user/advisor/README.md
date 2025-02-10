Example runs:

* `python3 run_advisor.py --name isotropic --path <path-to-devito>/examples/seismic/acoustic/acoustic_example.py`
* `python3 run_advisor.py --name tti_so8 --path <path-to-devito>/examples/seismic/tti/tti_example.py --exec-args "-so 8"`
* `python3 run_advisor.py --name iso_ac_so4 --path <path-to-devito>/benchmarks/user/benchmark.py --exec-args "run -P acoustic -so 4 --tn 200 -d 100 100 100"`

After the run has finished you should be able to save a .json and plot the
roofline with the results:
* `python3 roofline.py --name Roofline --project <advisor-project-name>`

To create a read-only snapshot for use with Intel Advisor GUI, use:
* `advixe-cl --snapshot --project-dir=<advisor-project-name> pack -- /<new-snapshot-name>`

Prerequisites:
* Support is guaranteed only for Intel oneAPI 2025; earlier versions may not work.
You may download Intel oneAPI [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=apt).

* Add Advisor (advixe-cl) and compilers (icx) in the path. It should be along the lines of:
```sh
source /opt/intel/oneapi/advisor/latest/env/vars.sh
source /opt/intel/oneapi/compiler/latest/env/vars.sh
```
depending on where you installed oneAPI

* In Linux systems you may need to enable system-wide profiling by setting:

```sh
/proc/sys/kernel/yama/ptrace_scope to 0
/proc/sys/kernel/perf_event_paranoid to 1
```

* `numactl` must be available on the system. If not available, install using:
```sh
sudo apt-get install numactl
```
* Install `pandas` and `matplotlib`. They are not included in the core Devito installation.
```sh
pip install pandas matplotlib
```

Limitations:

* Untested with more complicated examples.
* Running the `tripcounts` analysis is time-consuming, despite starting in paused
  mode. This analysis, together with the `survey` analysis, is necessary to
  generate a roofline. Both are run by `run_advisor.py`.
* Requires Python3, untested in conda environments
* Currently requires download of repository and running `pip install .`, the scripts
  are currently not included as a package with the user installation of Devito

TODO:

* Give a name to the points in the roofline, otherwise it's challenging to
  relate loops (code sections) to data.
* Emit a report summarizing the configuration used to run the analysis
  (threading, socket binding, ...).

Useful links:
* [ Memory-Level Roofline Analysis in Intel® Advisor ](https://software.intel.com/content/www/us/en/develop/articles/memory-level-roofline-model-with-advisor.html " Memory-Level Roofline Analysis in Intel® Advisor ")
* [CPU / Memory Roofline Insights
Perspective](https://software.intel.com/content/www/us/en/develop/documentation/advisor-user-guide/top/optimize-cpu-usage/cpu-roofline-perspective.html "CPU / Memory Roofline Insights
Perspective")
* [ Roofline Resources for Intel® Advisor Users ](https://software.intel.com/content/www/us/en/develop/articles/advisor-roofline-resources.html " Roofline Resources for Intel® Advisor Users ")