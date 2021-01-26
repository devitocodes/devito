from devito.core.cpu import Cpu64AdvOperator, Cpu64AdvOmpOperator
from devito.exceptions import InvalidOperator
from devito.passes.clusters import (Blocking, Lift, cire, cse, eliminate_arrays,
                                    extract_increments, factorize, fuse, optimize_pows)
from devito.tools import timed_pass

__all__ = ['Intel64AdvOperator', 'Intel64AdvOmpOperator', 'Intel64FsgOperator',
           'Intel64FsgOmpOperator']


Intel64AdvOperator = Cpu64AdvOperator
Intel64AdvOmpOperator = Cpu64AdvOmpOperator


class Intel64FsgOperator(Intel64AdvOperator):

    """
    Operator with performance optimizations tailored "For Small Grids" (FSG).
    """

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        kwargs = super(Intel64FsgOperator, cls)._normalize_kwargs(**kwargs)

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


class Intel64FsgOmpOperator(Intel64FsgOperator, Cpu64AdvOmpOperator):
    _specialize_iet = Cpu64AdvOmpOperator._specialize_iet
