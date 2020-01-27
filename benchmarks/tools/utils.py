import sys
from datetime import datetime


__all__ = ['bench_print']


def bench_print(msg, pre=0, post=0, timestamp=False, comm=None):
    if sys.stdout.isatty() and sys.stderr.isatty():
        # Blue
        color = '\033[1;37;34m%s\033[0m'
    else:
        color = '%s'

    if comm is not None:
        mpi_rank = comm.rank
    else:
        # Emit on all ranks, or perhaps we're not even running over MPI
        mpi_rank = 0

    for i in range(pre):
        if mpi_rank == 0:
            print("")
    if msg:
        if mpi_rank == 0:
            now = datetime.now()
            ts = ' [%s]' % now.strftime("%H:%M:%S") if timestamp else ''
            print(color % ("DevitoBench%s: %s" % (ts, msg)))
    for i in range(post):
        if mpi_rank == 0:
            print("")
