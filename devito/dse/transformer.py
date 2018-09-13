from __future__ import absolute_import

from devito.ir.clusters import ClusterGroup, groupby
from devito.dse.backends import (BasicRewriter, AdvancedRewriter, SpeculativeRewriter,
                                 AggressiveRewriter)
from devito.dse.manipulation import cross_cluster_cse
from devito.logger import dse_warning
from devito.parameters import configuration
from devito.tools import flatten

__all__ = ['rewrite']


modes = {
    'basic': BasicRewriter,
    'advanced': AdvancedRewriter,
    'speculative': SpeculativeRewriter,
    'aggressive': AggressiveRewriter
}
"""The DSE transformation modes."""

configuration.add('dse', 'advanced', list(modes))


def rewrite(clusters, mode='advanced'):
    """
    Transform N :class:`Cluster` objects of SymPy expressions into M
    :class:`Cluster` objects of SymPy expressions with reduced
    operation count, with M >= N.

    :param clusters: The clusters to be transformed.
    :param mode: drive the expression transformation

    The ``mode`` parameter recognises the following values: ::

         * 'noop': Do nothing.
         * 'basic': Apply common sub-expressions elimination.
         * 'advanced': Apply all transformations that will reduce the
                       operation count w/ minimum increase to the memory pressure,
                       namely 'basic', factorization, CIRE for time-invariants only.
         * 'speculative': Like 'advanced', but apply CIRE also to time-varying
                          sub-expressions, which might further increase the memory
                          pressure.
         * 'aggressive': Like 'speculative', but apply CIRE to any non-trivial
                         sub-expression (i.e., anything that is at least in a
                         sum-of-product form).
                         Further, seek and drop cross-cluster redundancies (this
                         is the only pass that attempts to optimize *across*
                         clusters, rather than within a cluster).
                         The 'aggressive' mode may substantially increase the
                         symbolic processing time; it may or may not reduce the
                         JIT-compilation time; it may or may not improve the
                         overall runtime performance.
    """
    if not (mode is None or isinstance(mode, str)):
        raise ValueError("Parameter 'mode' should be a string, not %s." % type(mode))

    if mode is None or mode == 'noop':
        return clusters
    elif mode not in modes:
        dse_warning("Unknown rewrite mode(s) %s" % mode)
        return clusters

    # 1) Local optimization
    # ---------------------
    # We use separate rewriters for dense and sparse clusters; sparse clusters have
    # non-affine index functions, thus making it basically impossible, in general,
    # to apply the more advanced DSE passes.
    # Note: the sparse rewriter uses the same template for temporaries as
    # the dense rewriter, thus temporaries are globally unique
    rewriter = modes[mode]()
    fallback = BasicRewriter(False, rewriter.template)

    processed = ClusterGroup(flatten(rewriter.run(c) if c.is_dense else fallback.run(c)
                                     for c in clusters))

    # 2) Cluster grouping
    # -------------------
    # Different clusters may have created new (smaller) clusters which are
    # potentially groupable within a single cluster
    processed = groupby(processed)

    # 3)Global optimization
    # ---------------------
    # After grouping, there may be redundancies in one or more clusters. This final
    # pass searches and drops such redundancies
    if mode == 'aggressive':
        processed = cross_cluster_cse(processed)

    return processed.finalize()
