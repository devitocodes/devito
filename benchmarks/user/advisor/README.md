# Intel Advisor roofline profiling on Devito

This README aims to help users derive rooflines through using Devito with [Intel Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/advisor.html).
We recommend going through tutorial [02_advisor_roofline.ipynb](https://github.com/devitocodes/devito/blob/main/examples/performance/02_advisor_roofline.ipynb) for a more detailed step-by-step guidance.

### Prerequisites:
* Support is guaranteed only for Intel oneAPI 2025; earlier versions may not work.
You may download Intel oneAPI [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=apt).

* Add Advisor (advixe-cl) and compilers (icx) in the path. The right env variables should be sourced along the lines of (depending on your isntallation folder):
```sh
source /opt/intel/oneapi/advisor/latest/env/vars.sh
source /opt/intel/oneapi/compiler/latest/env/vars.sh
```

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


### Example runs:

```bash
# The isotropic acoustic example
python3 run_advisor.py --name isotropic --path <path-to-devito>/examples/seismic/acoustic/acoustic_example.py
# The isotropic elastic example
python3 run_advisor.py --name iso_elastic --path <path-to-devito>/examples/seismic/elastic/elastic_example.py --exec-args "-so 4"
# The anisotropic acoustic (TTI) example
python3 run_advisor.py --name tti_so8 --path <path-to-devito>/examples/seismic/tti/tti_example.py --exec-args "-so 8"
```

After the run has finished you should be able to save a `.json` and plot the
roofline with the results:
```bash
python3 roofline.py --name Roofline --project <advisor-project-name>
```

To create a read-only snapshot for use with Intel Advisor GUI, use:
```bash
advixe-cl --snapshot --project-dir=<advisor-project-name> pack -- /<new-snapshot-name>
```
### Limitations:

* Not tested with all possible examples that Devito can support.
* Running the `tripcounts` analysis is time-consuming, despite starting in paused
  mode. This analysis, together with the `survey` analysis, is necessary to
  generate a roofline. Both are run by `run_advisor.py`.
* Requires Python 3.10 or later, untested in conda environments
* Currently requires download of repository and running `pip install .`, the scripts
  are currently not included as a package with the user installation of Devito

### TODO:

* Give a name to the points in the roofline, otherwise it's challenging to
  relate loops (code sections) to data.
* Emit a report summarizing the configuration used to run the analysis
  (threading, socket binding, ...).

### Useful links:

* [ Intel® Advisor Performance Optimization Cookbook ](https://www.intel.com/content/www/us/en/docs/advisor/cookbook/2024-2/overview.html " Intel® Advisor Performance Optimization Cookbook ")

* [ Intel® Advisor User Guide ](https://www.intel.com/content/www/us/en/docs/advisor/cookbook/2024-2/overview.html " Intel® Advisor User Guide ")

* [ Roofline Resources for Intel® Advisor Users ](https://software.intel.com/content/www/us/en/develop/articles/advisor-roofline-resources.html " Roofline Resources for Intel® Advisor Users ")

* [ Memory-Level Roofline Analysis in Intel® Advisor ](https://software.intel.com/content/www/us/en/develop/articles/memory-level-roofline-model-with-advisor.html " Memory-Level Roofline Analysis in Intel® Advisor ")

* [ Identify Bottlenecks Iteratively: Cache-Aware Roofline ](https://www.intel.com/content/www/us/en/docs/advisor/cookbook/2024-2/identify-bottlenecks-cache-aware-roofline.html " Identify Bottlenecks Iteratively: Cache-Aware Roofline ")

* [ Samuel Williams, Andrew Waterman, and David Patterson [2009]. Roofline: an insightful visual performance model for multicore architectures ](https://dl.acm.org/doi/10.1145/1498765.1498785 " Roofline: an insightful visual performance model for multicore architectures ")

* [ A. Ilic, F. Pratas and L. Sousa [2014]. Cache-aware Roofline model: Upgrading the loft ](https://ieeexplore.ieee.org/document/6506838 " Cache-aware Roofline model: Upgrading the loft ")

* [ Understanding the Roofline Model by Durganshu Mishra ](https://hackernoon.com/understanding-the-roofline-model " Understanding the Roofline Model ")

