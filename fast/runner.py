# this file runs a specific benchmarking configuration on archer2
# and returns the performance data

from dataclasses import dataclass
import subprocess


@dataclass
class PerfReport:
    impl_name: str
    bench_name: str
    config: str
    size: list[int]
    ranks: int
    runs: int

    times: list[list[float]]
    """
    a list of per-rank execution times

    times[i][k] is the execution time of the k-th rank on the i-th run.
    """


def run_benchmark(ranks: int, cpus_per_rank: int, name: str, flags: str, time_limit='01:00:00', env: str = ""):
    cmd = f"${env} srun --n={nodes} --cpus-per-task={cpus_per_rank} --exclusive --time=${time_limit} --partition=standard --qos=standard --account=d011 -u" \
          f" time python3 run_benchmark.py f{name} ${flags}"
    pr = subprocess.run(cmd, shell=True, capture_output=True)
    res = pr.stdout.decode('utf-8')
    print(res)
