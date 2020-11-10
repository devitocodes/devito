"""
Generate a roofline for the Intel Advisor ``project``.

This module has been partly extracted from the examples directory of Intel Advisor 2018.
"""
import click

import math
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt  # noqa
import sys
import os

from benchmarks.user.advisor.advisor_logging import check, err, log


try:
    import advisor
except ImportError:
    check(False, 'Error: Intel Advisor could not be found on the system,'
          ' make sure to source environment variables properly. Information can be'
          ' found at https://software.intel.com/content/www/us/en/develop/'
          'documentation/advisor-user-guide/top/launch-the-intel-advisor/'
          'intel-advisor-cli/setting-and-using-intel-advisor-environment-variables.html')
    sys.exit(1)


matplotlib.use('Agg')
# Use fancy plot colors
plt.style.use('seaborn-darkgrid')


@click.command()
# Required arguments
@click.option('--name', '-n', required=True, help='The name of the generated'
              'roofline png.')
@click.option('--project', '-o', required=True,
              help='The directory of the Intel Advisor project containing '
                   'a roofline analysis.')
# Optional arguments
@click.option('--scale', type=float, default=1.0,
              help='Specify by how much should the roofs be '
                   'scaled down due to using fewer cores than '
                   'available (e.g., when running on a single '
                   'socket).')
@click.option('--precision', type=click.Choice(['SP', 'DP', 'all']),
              help='Arithmetic precision.', default='SP')
@click.option('--mode', '-m', type=click.Choice(['overview', 'top-loops']),
              default='overview', required=True,
              help='overview: Display a single point with the total GFLOPS and '
                   'arithmetic intensity of the program.\n top-loops: Display all the '
                   'top time consuming loops within one order of magnitude (x10) from '
                   'the most time consuming loop.')
@click.option('--th', default=0, help='Percentage threshold (e.g. 95) such that loops '
                                      'under this value in execution time consumed will '
                                      'not be displayed. Only valid for --top-loops.')
def roofline(name, project, scale, precision, mode, th):
    pd.options.display.max_rows = 20

    log('Opening project...')
    project = advisor.open_project(str(project))

    if not project:
        err('Could not open project %s.' % project)
    log('Loading data...')

    data = project.load(advisor.SURVEY)
    rows = [{col: row[col] for col in row} for row in data.bottomup]
    roofs = data.get_roofs()

    full_df = pd.DataFrame(rows).replace('', np.nan)

    # Narrow down the columns to those of interest
    df = full_df[analysis_columns].copy()

    df.self_ai = df.self_ai.astype(float)
    df.self_gflops = df.self_gflops.astype(float)
    df.self_time = df.self_time.astype(float)

    # Add time weight column
    loop_total_time = df.self_time.sum()
    df['percent_weight'] = df.self_time / loop_total_time * 100

    fig, ax = plt.subplots()
    key = lambda roof: roof.bandwidth if 'bandwidth' not in roof.name.lower() else 0
    max_compute_roof = max(roofs, key=key)
    max_compute_bandwidth = max_compute_roof.bandwidth / math.pow(10, 9)  # as GByte/s
    max_compute_bandwidth /= scale  # scale down as requested by the user

    key = lambda roof: roof.bandwidth if 'bandwidth' in roof.name.lower() else 0
    max_memory_roof = max(roofs, key=key)
    max_memory_bandwidth = max_memory_roof.bandwidth / math.pow(10, 9)  # as GByte/s
    max_memory_bandwidth /= scale  # scale down as requested by the user

    # Parameters to center the chart
    ai_min = 2**-5
    ai_max = 2**5
    gflops_min = 2**0
    width = ai_max

    for roof in roofs:
        # by default drawing multi-threaded roofs only
        if 'single-thread' not in roof.name:
            # memory roofs
            if 'bandwidth' in roof.name.lower():
                bandwidth = roof.bandwidth / math.pow(10, 9)  # as GByte/s
                bandwidth /= scale  # scale down as requested by the user
                # y = bandwidth * x
                x1, x2 = 0, min(width, max_compute_bandwidth / bandwidth)
                y1, y2 = 0, x2*bandwidth
                label = '{} {:.0f} GB/s'.format(roof.name, bandwidth)
                ax.plot([x1, x2], [y1, y2], '-', label=label)

            # compute roofs
            elif precision == 'all' or precision in roof.name:
                bandwidth = roof.bandwidth / math.pow(10, 9)  # as GFlOPS
                bandwidth /= scale  # scale down as requested by the user
                x1, x2 = max(bandwidth / max_memory_bandwidth, 0), width
                y1, y2 = bandwidth, bandwidth
                label = '{} {:.0f} GFLOPS'.format(roof.name, bandwidth)
                ax.plot([x1, x2], [y1, y2], '-', label=label)

    # drawing points using the same ax
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(ScalarFormatter())

    if mode == 'overview':
        # Only display the overall GFLOPS and arithmetic intensity of the program
        ax.plot(data.metrics.total_ai, data.metrics.total_gflops, 'o', color='black')
    elif mode == 'top-loops':
        # Display the most costly loop followed by loops with same order of magnitude
        max_self_time = df.self_time.max()
        top_df = df[(max_self_time / df.self_time < 10) &
                    (max_self_time / df.self_time >= 1) & (df.percent_weight >= th)]
        for _, row in top_df.iterrows():
            ax.plot(row.self_ai, row.self_gflops, 'o', color='black')
            label_x = row.self_ai + (row.self_ai + ai_max - 2 * ai_min) * (2**0.005 - 1)
            label_y = row.self_gflops
            ax.text(label_x, label_y,
                    'Time: %.2fs\n'
                    'Incidence: %.0f%%' % (row.self_time, row.percent_weight),
                    bbox={'boxstyle': 'round', 'facecolor': 'white'}, fontsize=8)

    # make sure axes start at 1
    ax.set_ylim(ymin=gflops_min)
    ax.set_xlim(xmin=ai_min, xmax=ai_max)

    ax.set_xlabel('Arithmetic intensity (FLOP/Byte)')
    ax.set_ylabel('Performance (GFLOPS)')

    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                        prop={'size': 7}, title='Rooflines')

    # saving the chart in PNG format
    plt.savefig('%s.png' % name, bbox_extra_artists=(legend,), bbox_inches='tight')
    figpath = os.path.realpath(__file__).split(os.path.basename(__file__))[0]
    log('Figure saved in %s%s.png.' % (figpath, name))


analysis_columns = ['loop_name', 'self_ai', 'self_gflops', 'self_time']

if __name__ == '__main__':
    roofline()
