from devito.ir import cluster_pass
from devito.symbolics import unevaluate as _unevaluate

__all__ = ['unevaluate']


@cluster_pass(mode='all')
def unevaluate(cluster):
    exprs = [_unevaluate(e) for e in cluster.exprs]

    return cluster.rebuild(exprs=exprs)
