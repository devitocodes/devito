from runner import run_benchmark, PerfReport
import csv

dims = {"2d5pt": 2, "3d_diff": 3}

def thread_scale(name: str, qos: str = 'standard'):
    flags = '-nt 100 --xdsl --devito --openmp'
    shape = [20000, 20000]
    env_base = "OMP_PLACES=cores"
    runs = 10

    raw_results = []

    result_lines = []

    for num_threads in (1, 2, 4, 8, 16, 32, 64, 128):
        env = env_base + f' OMP_NUM_THREADS={num_threads}'
        results = run_benchmark(
            ranks=1, 
            cpus_per_rank=num_threads,
            name=name,
            shape=shape,
            flags=flags,
            runs=runs,
            env=env 
        )
        raw_results.extend(results)

        for res in results:
            res.report()
            assert len(res.times) == 1
            result_lines.append((
                num_threads, res.impl_name, res.times[0]
            ))
    # write csv file
    with open(f'{name}_grid_runtimes.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['num_threads', 'implementation', 'time'])
        w.writerows(result_lines)
    
    # write raw data as json lines
    with open(f'{name}_grid_runtimes.json', 'w') as f:
        for r in raw_results:
            f.write(r.to_json() + '\n')


def mpi_scale_2d5pt(qos="standard"):
    flags = '-nt 100 --xdsl --devito --openmp --mpi'
    cpus_per_rank=4
    env = f"OMP_PLACES=cores OMP_NUM_THREADS={cpus_per_rank} DEVITO_MPI=1"
    shape = [2000, 2000]
    runs = 5

    raw_results = []

    result_lines = []

    for ranks in (1, 2, 4, 8, 16, 32, 64, 128):
        results = run_benchmark(
            ranks=ranks, 
            cpus_per_rank=cpus_per_rank,
            name='2d5pt',
            shape=shape,
            flags=flags,
            runs=runs,
            env=env,
            qos=qos,
        )
        raw_results.extend(results)

        for res in results:
            res.report()
            result_lines.append((
                ranks, res.impl_name, sum(res.times) / len(res.times)
            ))
    # write csv file
    with open(f'2d5pt_mpi_runtimes.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['ranks', 'implementation', 'time'])
        w.writerows(result_lines)

    # write raw data as json lines
    with open(f'2d5pt_mpi_runtimes.json', 'w') as f:
        for r in raw_results:
            f.write(r.to_json() + '\n')

def mpi_thread_scale_2d5pt(qos="standard"):
    flags = '-nt 100 --xdsl --devito --openmp --mpi'
    ranks = 16
    env = f"OMP_PLACES=cores OMP_NUM_THREADS={cpus_per_rank} DEVITO_MPI=1"
    shape = [20000, 20000]
    runs = 5

    raw_results = []

    result_lines = []

    for cpus_per_rank in (1, 2, 4, 8, 16, 32):
        results = run_benchmark(
            ranks=ranks, 
            cpus_per_rank=cpus_per_rank,
            name='2d5pt',
            shape=shape,
            flags=flags,
            runs=runs,
            env=env,
            qos=qos,
        )
        raw_results.extend(results)

        for res in results:
            res.report()
            result_lines.append((
                ranks, res.impl_name, sum(res.times) / len(res.times)
            ))
    # write csv file
    with open(f'2d5pt_mpi_runtimes.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['ranks', 'implementation', 'time'])
        w.writerows(result_lines)

    # write raw data as json lines
    with open(f'2d5pt_mpi_runtimes.json', 'w') as f:
        for r in raw_results:
            f.write(r.to_json() + '\n')
