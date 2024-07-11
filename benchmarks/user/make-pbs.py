import os

import argparse
import pathlib

from devito.logger import log


def cleanup():
    for f in os.listdir():
        if f.endswith('.gen.sh'):
            os.remove(f)
    log("Cleaned up successfully")


def generate(**kwargs):

    join = lambda l: ' '.join('%d' % i for i in l)
    args = dict(kwargs)

    # Some postprocessing
    args['shape'] = join(args['shape'])
    args['load'] = ' '.join(args['load'])
    args['export'] = ' '.join(args['export'])

    template_header = """\
#!/bin/bash

#PBS -lselect=%(nn)s:ncpus=%(ncores)s:mem=120gb:mpiprocs=%(np)s:ompthreads=%(nt)s
#PBS -lwalltime=02:00:00

lscpu

module load %(load)s

cd %(home)s

source activate devito

export DEVITO_HOME=%(home)s
export DEVITO_ARCH=intel
export DEVITO_LANGUAGE=openmp
export DEVITO_LOGGING=DEBUG
export DEVITO_MPI=%(mpi)s

export %(export)s

cd benchmarks/user
"""  # noqa
    template_cmd = """\
mpiexec python benchmark.py run -P %(problem)s -d %(shape)s -so %(space_order)s --tn %(tn)s --arch %(arch)s
"""  # noqa

    # Generate one PBS file for each `np` value
    for nn in kwargs['nn']:
        args['nn'] = nn

        cmds = []
        for i in kwargs['mpi']:
            args['mpi'] = i
            cmds.append(template_cmd % args)
        cmds = ' \n'.join(cmds)

        body = ' \n'.join([template_header % args, cmds])

        with open('pbs_nn%d.gen.sh' % int(nn), 'w') as f:
            f.write(body)

    log(f"PBS script generated successfully: {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool with multiple commands.")

    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest='command')

    # Add a 'generate' command
    pbs_parser = subparsers.add_parser(name='generate', help='Generate something.')

    pbs_parser.add_argument('-nn', nargs='*', type=int, default=[1],
                            help='Number of nodes')
    pbs_parser.add_argument('-ncores', type=int, default=1,
                            help='Number of cores *per node*')
    pbs_parser.add_argument('-mem', type=int, default=120,
                            help='Requested DRAM *per node*')
    pbs_parser.add_argument('-np', type=int, default=1,
                            help='Number of MPI processes *per node*')
    pbs_parser.add_argument('-nt', type=int, default=1,
                            help='Number of OpenMP threads *per MPI process*')
    pbs_parser.add_argument('--mpi', nargs='*', default=['basic'],
                            help='Devito MPI mode(s)')
    pbs_parser.add_argument('--arch', type=str, default='unknown',
                            help='Test-bed architecture')
    pbs_parser.add_argument('-r', '--resultsdir', type=str, default='results',
                            help='Results directory')
    pbs_parser.add_argument('--load', nargs='*', default=[],
                            help='Modules to be loaded')
    pbs_parser.add_argument('--export', nargs='*', default=[],
                            help='Env vars to be exported')
    pbs_parser.add_argument("-d", "--shape", default=(50, 50, 50), type=int, nargs="+",
                            help="Number of grid points along each axis")
    pbs_parser.add_argument('-so', '--space_order', default=4, type=int,
                            help='Space order')
    pbs_parser.add_argument('-P', '--problem', default='acoustic', type=str,
                            help='Problem name')
    pbs_parser.add_argument('-tn', '--tn', default=10, type=int,
                            help='Number of timesteps')
    pbs_parser.add_argument('--home', default=pathlib.Path(__file__).parent.resolve(),
                            type=pathlib.Path, help='Home directory')

    # Add a 'generate' command
    clean_parser = subparsers.add_parser('cleanup', help='Clean something.')

    # Parse the arguments
    args = parser.parse_args()

    # Execute the corresponding function
    if args.command == 'generate':
        generate(**vars(args))
    elif args.command == 'cleanup':
        cleanup()
