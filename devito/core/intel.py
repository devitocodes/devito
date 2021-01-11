from devito.core.cpu import CPU64Operator, CPU64OpenMPOperator
from devito.exceptions import InvalidOperator
from devito.passes.clusters import (Blocking, Lift, cire, cse, eliminate_arrays,
                                    extract_increments, factorize, fuse, optimize_pows)
from devito.tools import timed_pass

__all__ = ['Intel64Operator', 'Intel64OpenMPOperator', 'Intel64FSGOperator',
           'Intel64FSGOpenMPOperator']


Intel64Operator = CPU64Operator
Intel64OpenMPOperator = CPU64OpenMPOperator


class Intel64FSGOperator(Intel64Operator):

    """
    Operator with performance optimizations tailored "For Small Grids" (FSG).
    """

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        kwargs = super(Intel64FSGOperator, cls)._normalize_kwargs(**kwargs)

        if kwargs['options']['min-storage']:
            raise InvalidOperator('You should not use `min-storage` with `advanced-fsg '
                                  ' as they work in opposite directions')

        return kwargs

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = fuse(clusters, toposort=True)

        # Hoist and optimize Dimension-invariant sub-expressions
        clusters = cire(clusters, 'invariants', sregistry, options, platform)
        clusters = Lift().process(clusters)

        # Reduce flops (potential arithmetic alterations)
        clusters = extract_increments(clusters, sregistry)
        clusters = cire(clusters, 'sops', sregistry, options, platform)
        clusters = factorize(clusters)
        clusters = optimize_pows(clusters)

        # The previous passes may have created fusion opportunities, which in
        # turn may enable further optimizations
        clusters = fuse(clusters)
        clusters = eliminate_arrays(clusters)

        # Reduce flops (no arithmetic alterations)
        clusters = cse(clusters, sregistry)

        # Blocking to improve data locality
        clusters = Blocking(options).process(clusters)

        return clusters


class Intel64FSGOpenMPOperator(Intel64FSGOperator, CPU64OpenMPOperator):
    _specialize_iet = CPU64OpenMPOperator._specialize_iet
