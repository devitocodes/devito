import os
import argparse
import devito


def generate(args):
    join = lambda l: ' '.join('%d' % i for i in l)
    args = vars(args)
    args['shape'] = join(args['shape'])
    args['space_order'] = join(args['space_order'])

    args['home'] = os.path.dirname(os.path.dirname(devito.__file__))

    args['load'] = '\n'.join('module load %s' % i for i in args['load'])
    args['export'] = '\n'.join('export %s' % i for i in args['export'])

    template_header = """\
#!/bin/bash

#PBS -lselect=%(nn)s:ncpus=%(ncpus)s:mem=120gb:mpiprocs=%(np)s:ompthreads=%(nt)s
#PBS -lwalltime=02:00:00

lscpu

%(load)s

cd %(home)s

source activate devito

export DEVITO_HOME=%(home)s
export DEVITO_ARCH=intel
export DEVITO_LANGUAGE=openmp
export DEVITO_LOGGING=DEBUG

%(export)s

cd benchmarks/user
"""  # noqa
    template_cmd = """\
DEVITO_MPI=%(mpi)s mpiexec python benchmark.py bench -P %(problem)s -bm O2 -d %(shape)s -so %(space_order)s --tn %(tn)s -x 1 --arch %(arch)s -r %(resultsdir)s\
"""  # noqa

    # Generate one PBS file for each `np` value
    for nn in args['nn']:
        args['nn'] = nn

        cmds = []
        for i in args['mpi']:
            args['mpi'] = i
            cmds.append(template_cmd % args)
        cmds = ' \n'.join(cmds)

        body = ' \n'.join([template_header % args, cmds])

        with open('pbs_nn%d.gen.sh' % int(nn), 'w') as f:
            f.write(body)


def cleanup(args):
    for f in os.listdir():
        if f.endswith('.gen.sh'):
            os.remove(f)

def create_parser():
    parser = argparse.ArgumentParser(description='Generate PBS scripts')
    subparsers = parser.add_subparsers(dest='command', required=True)

    gen = subparsers.add_parser('generate')
    gen.add_argument('-P', '--problem', required=True)
    gen.add_argument('-d', '--shape', type=int, nargs='+', default=[50, 50, 50])
    gen.add_argument('-so', '--space-order', type=int, nargs='+', default=[2])
    gen.add_argument('--tn', type=float, default=250)
    gen.add_argument('-nn', type=int, nargs='+', default=[1])
    gen.add_argument('-ncpus', type=int, default=1)
    gen.add_argument('-mem', type=int, default=120)
    gen.add_argument('-np', type=int, default=1)
    gen.add_argument('-nt', type=int, default=1)
    gen.add_argument('--mpi', nargs='+', default=['basic'])
    gen.add_argument('--arch', default='unknown')
    gen.add_argument('-r', '--resultsdir', default='results')
    gen.add_argument('--load', nargs='*', default=[])
    gen.add_argument('--export', nargs='*', default=[])

    subparsers.add_parser('cleanup')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command == 'generate':
        generate(args)
    elif args.command == 'cleanup':
        cleanup(args)


if __name__ == "__main__":
    main()
