# this file runs a specific benchmarking configuration on archer2
# and returns the performance data

from dataclasses import dataclass
import subprocess


@dataclass
class PerfReport:
    impl_name: str
    config: str
    size: list[int]
    ranks: int
    runs: int

    times: list[list[float]]
    """
    a list of per-rank execution times

    times[i][k] is the execution time of the k-th rank on the i-th run.
    """


def run_benchmark(nodes: int, name: str, flags: str, time_limit='01:00:00', env: str = ""):
    cmd = f"srun --nodes={nodes} --exclusive --time=${time_limit} --partition=standard --qos=standard --account=d011 -u" \
          f" ${env} python3 run_benchmark.py f{name} ${flags}"
    pr = subprocess.run(cmd, shell=True, capture_output=True)
    res = pr.stdout.decode('utf-8')
    print(res)
