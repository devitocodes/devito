# this file runs a specific benchmarking configuration on archer2
# and returns the performance data

from dataclasses import dataclass
import subprocess
import json
from collections import defaultdict
from typing import Iterable


@dataclass
class PerfReport:
    impl_name: str
    bench_name: str
    flags: str
    env: str
    shape: list[int]
    ranks: int
    cpus: int

    times: list[list[float]]
    """
    a list of per-rank execution times

    times[i][k] is the execution time of the k-th rank on the i-th run.
    """

    correctness: list[dict]

    def to_json(self):
        return json.dumps({
            'impl_name': self.impl_name,
            'bench_name': self.bench_name,
            'flags': self.flags,
            'env': self.env,
            'shape': self.shape,
            'ranks': self.ranks,
            'cpus': self.cpus,
            'times': self.times,
            'correctness': self.correctness,
        })

    @staticmethod
    def from_json(json_d: str | dict):
        if isinstance(json_d, str):
            json_d = json.loads(json_d)
        return PerfReport(**json_d)

    def report(self):
        avg_time = sum(self.times) / len(self.times)
        c_nums = []
        if self.correctness:
            for correctness_data in self.correctness:
                c_nums.append(correctness_data['abs_max_error'])
        max_err = '-' if not c_nums else max(c_nums)

        print(f"""Run info:
  impl:    {self.impl_name}
  bench:   {self.bench_name}
  grid:    {self.shape}
  time:    {avg_time}
  max err: {max_err}""")

def run_benchmark(ranks: int, cpus_per_rank: int, name: str, shape: list[int], flags: str, runs: int = 1, time_limit='01:00:00', env: str = "", qos="standard"):
    shape = ' '.join(str(x) for x in shape)
    cmd = f"{env} srun -n {ranks} --cpus-per-task {cpus_per_rank} --exclusive --time={time_limit} --partition=standard --qos=${qos} --account=d011 --distribution=block:block --hint=nomultithread -u" \
          f" python3 run_benchmark.py {name} {flags} -d {shape} --repeat {runs}"
    print(f"running: {cmd}")
    pr = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(pr.stderr)
    res = pr.stdout

    reports: list[dict] = []

    for line in res.splitlines():
        if line[0] != '{':
            continue
        try:
            d = json.loads(line)
            reports.append(d)
        except json.JSONDecodeError:
            pass

    try:
        return list(process_records(reports, flags, name, shape, ranks, cpus_per_rank, env))
    except Exception as ex:
        print("Error in processing records: ", ex)
        print("with output:")
        print(res)
        

def get_rank(d: dict):
    return d['rank']
def get_run(d: dict):
    return d['run']
def get_type(d: dict):
    return d['type']
def get_impl(d: dict):
    return d['impl']

def prune_dict(d: dict, *remove_fields: str):
    for f in remove_fields:
        d.pop(f, None)
    return d

def process_records(records: list[dict], flags: str, bench: str, shape: list[int], ranks: int, cpus: int, env: str) -> list[PerfReport]:
    for run, run_records in sorted(group_by(records, get_run), key=lambda x: x[0]):
        kinds = dict(
            group_by(run_records, get_type)
        )

        if 'correctness' in kinds:
            kinds['correctness'] = [
                prune_dict(d, 'rank', 'run', 'type')
                for d in sorted(kinds.get('correctness', []), key=get_rank)
            ]
        else:
            kinds['correctness'] = []

        assert 'runtime' in kinds, "'runtime' records not present in run output! Something is wrong!"
        assert bench == kinds['runtime'][0]['name'], "collected records not for same bench as promised!"
        for impl, perf_records in group_by(kinds['runtime'], get_impl):
            yield PerfReport(
                impl_name=impl,
                bench_name=bench,
                flags=flags,
                env=env,
                shape=shape,
                ranks=ranks,
                cpus=cpus,
                times=[
                    r['runtime'] for r in sorted(perf_records, key=get_rank)
                ],
                correctness=kinds['correctness']
            )


def group_by(things: list, key) -> Iterable[list] | dict:
    buckets = defaultdict(list)
    for thing in things:
        buckets[key(thing)].append(thing)
    yield from buckets.items()
