import os

import click

from benchmark import option_simulation
import devito


@click.group()
def menu():
    pass


@menu.command(name='generate')
@option_simulation
@click.option('-id', default='bench', help='Job ID')
@click.option('-image', default='mloubout/devito-bench:v0.1', help='Devito docker image')
@click.option('-arch', default='gcc-8', help='Devito architecture')
@click.option('-nt', default='4', help='OpenMP number of threads')
@click.option('-runtime', default='mpich', help='MPI runtime')
@click.option('-ppn', default='1', help='Processes per node')
@click.option('-tn', default=50, help='Number of timesteps')
def generate(**kwargs):
    join = lambda l: ' '.join('%d' % i for i in l)
    args = dict(kwargs)
    args['shape'] = join(args['shape'])
    args['space_order'] = join(args['space_order'])

    args['home'] = os.path.dirname(os.path.dirname(devito.__file__))

    template_header = """\
job_specifications:
- id: %(id)s
  shared_data_volumes:
  - azureblob_vol
  tasks:
  - docker_image: %(image)s
    environment_variables:
      DEVITO_ARCH: %(arch)s
      OMP_NUM_THREADS: %(nt)s
      OMP_PROC_BIND: 'close'
      DEVITO_MPI: 1
      DEVITO_OPENMP: 1
      OMP_PLACES: 'cores'
      DEVITO_LOGGING: 'DEBUG'
      LC_ALL: 'C.UTF-8'
      LANG: 'C.UTF-8'
    default_working_dir: container
    multi_instance:
      num_instances: pool_current_dedicated
      mpi:
        runtime: mpich
        processes_per_node: %(ppn)s
        options:
          - --bind-to socket
    command: python3 devito/benchmarks/user/benchmark.py bench -P acoustic --tn %(tn)s -d %(shape)s -so %(space_order)s --resultsdir $AZ_BATCH_NODE_SHARED_DIR/results_bench -x 3
"""  # noqa

    # Generate one PBS file for each `np` value
    for nvm in kwargs['nt']:
        args['nt'] = nvm

        cmds = []
        cmds = ' \n'.join(cmds)

        body = ' \n'.join([template_header % args, cmds])

        with open('job_nn%d.gen.yaml' % int(nvm), 'w') as f:
            f.write(body)


@menu.command(name='cleanup')
def cleanup():
    for f in os.listdir():
        if f.endswith('.gen.yaml'):
            os.remove(f)


if __name__ == "__main__":
    menu()
