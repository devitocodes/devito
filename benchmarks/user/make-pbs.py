import os

import click

from benchmark import option_simulation
import devito


@click.group()
def menu():
    pass


@menu.command(name='generate')
@option_simulation
@click.option('-nn', '-nnodes', multiple=True, default=[1], help='Number of nodes')
@click.option('-ncpus', default=1, help='Number of cores *per node*')  # Should be ncores
@click.option('-mem', default=120, help='Requested DRAM *per node*')
@click.option('-np', default=1, help='Number of MPI processes *per node*')
@click.option('-nt', default=1, help='Number of OpenMP threads *per MPI process*')
@click.option('--mpi', default='basic', help='Devito MPI mode')
@click.option('--arch', default='unknown', help='Test-bed architecture')
@click.option('-r', '--resultsdir', default='results', help='Results directory')
@click.option('--load', multiple=True, default=[], help='Modules to be loaded')
def generate(**kwargs):
    join = lambda l: ' '.join('%d' % i for i in l)
    args = dict(kwargs)
    args['shape'] = join(args['shape'])
    args['space_order'] = join(args['space_order'])

    args['home'] = os.path.dirname(os.path.dirname(devito.__file__))

    args['load'] = '\n'.join('module load %s' % i for i in args['load'])

    template = """\
#!/bin/bash

#PBS -lselect=%(nnodes)s:ncpus=%(ncpus)s:mem=120gb:mpiprocs=%(np)s:ompthreads=%(nt)s
#PBS -lwalltime=02:00:00

%(load)s

cd %(home)s

source activate devito

export DEVITO_HOME=%(home)s
export DEVITO_ARCH=intel
export DEVITO_OPENMP=1
export DEVITO_MPI=%(mpi)s
export DEVITO_LOGGING=DEBUG

cd benchmarks/user

mpiexec python benchmark.py bench -P %(problem)s -bm O2 -d %(shape)s -so %(space_order)s --tn %(tn)s -x 1 --arch %(arch)s -r %(resultsdir)s
"""  # noqa

    # Generate one PBS file for each `np` value
    for nn in kwargs['nnodes']:
        args['nnodes'] = nn

        with open('pbs_nnodes%d.gen.sh' % int(nn), 'w') as f:
            f.write(template % args)


@menu.command(name='cleanup')
def cleanup():
    for f in os.listdir():
        if f.endswith('.gen.sh'):
            os.remove(f)


if __name__ == "__main__":
    menu()
