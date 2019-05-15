"""
Shape- and Dimension-promotion optimizations, such as:

    * Contraction: Clusters may be invariant in one or more Dimensions; such
                   Dimensions can be contracted away.
    * Scalarization. Some Array temporaries may be turned into scalars.
"""

from devito.symbolics import xreplace_indices
from devito.types import Scalar

__all__ = ['scalarize']


def scalarize(clusters, template):
    """
    Turn local Arrays, that is Arrays that read only within a single Cluster,
    into scalars.
    """

    processed = []
    for c in clusters:
        # Get Arrays appearing only in one cluster
        arrays = ({i for i in c.accesses if i.is_Array} -
                  set().union(*[c2.accesses for c2 in clusters if c2 is not c]))

        # Turn them into scalars -- this will produce a new cluster
        processed.append(_bump_and_scalarize(arrays, c, template))

    return processed



def _bump_and_scalarize(arrays, cluster, template):
    """
    Scalarize local Arrays.

    Parameters
    ----------
    arrays : list of Array
        The Arrays that will be scalarized.
    cluster : Cluster
        The Cluster where the local Arrays are used.

    Examples
    --------
    This transformation consists of two steps, "index bumping" and the
    actual "scalarization".

    Index bumping. Given:

        r[x,y,z] = b[x,y,z]*2

    Produce:

        r[x,y,z] = b[x,y,z]*2
        r[x,y,z+1] = b[x,y,z+1]*2

    Scalarization. Given:

        r[x,y,z] = b[x,y,z]*2
        r[x,y,z+1] = b[x,y,z+1]*2

    Produce:

        t0 = b[x,y,z]*2
        t1 = b[x,y,z+1]*2

    An Array being scalarized could be Indexed multiple times. Before proceeding
    with scalarization, therefore, we perform index bumping. For example, given:

        r0[x,y,z] = b[x,y,z]*2
        r1[x,y,z] = ... r[x,y,z] ... r[x,y,z-1] ...

    This function will produce:

        t0 = b[x,y,z]*2
        t1 = b[x,y,z-1]*2
        r1[x,y,z] = ... t0 ... t1 ...]
    """
    if not arrays:
        return cluster

    mapper = {}
    processed = []
    for e in cluster.exprs:
        f = e.lhs.function
        if f in arrays:
            for i in cluster.accesses[f]:
                mapper[i] = Scalar(name=template(), dtype=f.dtype)

                # Index bumping
                assert len(f.indices) == len(e.lhs.indices) == len(i.indices)
                shifting = {idx: idx + (o2 - o1) for idx, o1, o2 in
                            zip(f.indices, e.lhs.indices, i.indices)}

                # Scalarization
                handle = e.func(mapper[i], e.rhs.xreplace(mapper))
                handle = xreplace_indices(handle, shifting)
                processed.append(handle)
        else:
            processed.append(e.func(e.lhs, e.rhs.xreplace(mapper)))

    return cluster.rebuild(processed)
