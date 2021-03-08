from devito.ir.support import IterationSpace, SEQUENTIAL, PARALLEL, Interval
from devito.logger import warning
from devito.passes.clusters.utils import cluster_pass
from devito.symbolics import xreplace_indices

__all__ = ['skewing']


@cluster_pass
def skewing(cluster, *args):

    """
    Skew the accesses along a SEQUENTIAL Dimension.
    Reference:
    https://link.springer.com/article/10.1007%2FBF01407876
    Example:

    Transform

    for i = 2, n-1
        for j = 2, m-1
            a[i,j] = (a[a-1,j] + a[i,j-1] + a[i+1,j] + a[i,j+1]) / 4
        end for
    end for

    to

    for i = 2, n-1
        for j = 2+i, m-1+i
            a[i,j-i] = (a[a-1,j-i] + a[i,j-1-i] + a[i+1,j-i] + a[i,j+1-i]) / 4
        end for
    end for
    """

    # What dimensions should we skew against?
    # e.g. SEQUENTIAL Dimensions (Usually time in FD solvers, but not only limited
    # to this)
    # What dimensions should be skewed?
    # e.g. PARALLEL dimensions (x, y, z) but not blocking ones (x_blk0, y_blk0)
    # Iterate over the iteration space and assign dimensions to skew or skewable
    # depending on their properties

    skew_dims, skewable = [], []

    for i in cluster.ispace:
        if SEQUENTIAL in cluster.properties[i.dim]:
            skew_dims.append(i.dim)
        elif PARALLEL in cluster.properties[i.dim] and not i.dim.symbolic_incr.is_Symbol:
            skewable.append(i.dim)

    if len(skew_dims) > 1:
        raise warning("More than 1 dimensions that can be skewed.\
                      Skewing the first in the list")
    elif len(skew_dims) == 0:
        # No dimensions to skew against -> nothing to do, return
        return cluster

    # Skew dim will not be none here:
    # Initializing a default skewed dim index position in loop
    skew_index = 0
    skew_dim = skew_dims.pop()  # Skew first one

    mapper, intervals, processed = {}, [], []

    for i in cluster.ispace.intervals:
        # Skew dim if nested under time i.dim not in skew_dims:
        if skew_index < cluster.ispace.intervals.index(i) and i.dim in skewable:
            mapper[i.dim] = i.dim - skew_dim
            intervals.append(Interval(i.dim, skew_dim, skew_dim))
            skewable.remove(i.dim)
        # Do not touch otherwise
        else:
            intervals.append(i)

        processed = xreplace_indices(cluster.exprs, mapper)

    assert len(skewable) == 0

    ispace = IterationSpace(intervals, cluster.ispace.sub_iterators,
                            cluster.ispace.directions)

    rebuilt = cluster.rebuild(exprs=processed, ispace=ispace)

    return rebuilt
