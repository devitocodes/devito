import os

import click

from benchmark import option_simulation
import devito


@click.group()
def menu():
    pass


@menu.command(name='generate')
@option_simulation
@click.option('-nn', multiple=True, default=[1], help='Number of nodes')
@click.option('-ncpus', default=1, help='Number of cores *per node*')  # Should be ncores
@click.option('-mem', default=120, help='Requested DRAM *per node*')
@click.option('-np', default=1, help='Number of MPI processes *per node*')
@click.option('-nt', default=1, help='Number of OpenMP threads *per MPI process*')
@click.option('--mpi', multiple=True, default=['basic'], help='Devito MPI mode(s)')
@click.option('--arch', default='unknown', help='Test-bed architecture')
@click.option('-r', '--resultsdir', default='results', help='Results directory')
@click.option('--load', multiple=True, default=[], help='Modules to be loaded')
@click.option('--export', multiple=True, default=[], help='Env vars to be exported')
def generate(**kwargs):
    join = lambda l: ' '.join('%d' % i for i in l)
    args = dict(kwargs)
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


@menu.command(name='cleanup')
def cleanup():
    for f in os.listdir():
        if f.endswith('.gen.sh'):
            os.remove(f)


if __name__ == "__main__":
    menu()
