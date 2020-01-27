import os

import click

from benchmark import option_simulation
import devito


@click.group()
def menu():
    pass


@menu.command(name='generate')
@option_simulation
@click.option('-id', default='devito-bench', help='Pool ID')
@click.option('-dedvm', multiple=True, default=[1], help='Dedicated VMs')
@click.option('-lowvm', default='0', help='Low priority VMs')
@click.option('-vmsize', default='Standard_H16', help='VM size')
@click.option('-offer', default='UbuntuServer', help='Platform image offer')
@click.option('-pub', default='Canonical', help='Platform image publisher')
@click.option('-sku', default='18.04-LTS', help='Platform image SKU')
def generate(**kwargs):
    join = lambda l: ' '.join('%d' % i for i in l)
    args = dict(kwargs)
    args['shape'] = join(args['shape'])
    args['space_order'] = join(args['space_order'])

    args['home'] = os.path.dirname(os.path.dirname(devito.__file__))

    template_header = """\
pool_specification:
  id: %(id)s
  vm_configuration:
    platform_image:
      offer: %(offer)s
      publisher: %(pub)s
      sku: %(sku)s
      native: true
  vm_count:
    dedicated: %(dedvm)s
    low_priority: %(lowvm)s
  vm_size: %(vmsize)s
  inter_node_communication_enabled: true
  ssh:
    username: shipyard
"""  # noqa

    # Generate one PBS file for each `np` value
    for nvm in kwargs['dedvm']:
        args['dedvm'] = nvm

        cmds = []
        cmds = ' \n'.join(cmds)

        body = ' \n'.join([template_header % args, cmds])

        with open('pbs_nn%d.gen.yaml' % int(nvm), 'w') as f:
            f.write(body)


@menu.command(name='cleanup')
def cleanup():
    for f in os.listdir():
        if f.endswith('.gen.yaml'):
            os.remove(f)


if __name__ == "__main__":
    menu()
