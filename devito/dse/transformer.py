from devito.ir.clusters import ClusterGroup, groupby
from devito.dse.backends import (BasicRewriter, AdvancedRewriter, SpeculativeRewriter,
                                 AggressiveRewriter)
from devito.dse.manipulation import cross_cluster_cse
from devito.logger import dse_warning
from devito.tools import flatten

__all__ = ['dse_registry', 'rewrite']


dse_registry = ('basic', 'advanced', 'speculative', 'aggressive')

modes = {
    'basic': BasicRewriter,
    'advanced': AdvancedRewriter,
    'speculative': SpeculativeRewriter,
    'aggressive': AggressiveRewriter
}
"""The DSE transformation modes."""


def rewrite(clusters, mode='advanced'):
    """
    Given a sequence of N Clusters, produce a sequence of M Clusters with reduced
    operation count, with M >= N.

    Parameters
    ----------
    clusters : list of Cluster
        The Clusters to be transformed.
    mode : str, optional
        The aggressiveness of the rewrite. Accepted:
        - ``noop``: Do nothing.
        - ``basic``: Apply common sub-expressions elimination.
        - ``advanced``: Apply all transformations that will reduce the
                        operation count w/ minimum increase to the memory pressure,
                        namely 'basic', factorization, CIRE for time-invariants only.
        - ``speculative``: Like 'advanced', but apply CIRE also to time-varying
                           sub-expressions, which might further increase the memory
                           pressure.
         * ``aggressive``: Like 'speculative', but apply CIRE to any non-trivial
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
    elif mode not in dse_registry:
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
