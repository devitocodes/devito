from devito.ir.clusters import ClusterGroup, groupby
<<<<<<< HEAD
from devito.dse.backends import (BasicRewriter, AdvancedRewriter, SpeculativeRewriter,
                                 AggressiveRewriter, SkewingRewriter)
=======
from devito.dse.rewriters import BasicRewriter, AdvancedRewriter, AggressiveRewriter
<<<<<<< HEAD
>>>>>>> dse: Reorganise DSE directory structure; drop unused SpeculativeRewriter
from devito.dse.manipulation import cross_cluster_cse
=======
>>>>>>> dse: Drop cross_cluster_cse
from devito.logger import dse_warning
from devito.tools import flatten
from devito.parameters import configuration

__all__ = ['dse_registry', 'rewrite']

<<<<<<< HEAD
# Skewing rewriter
dse_registry = ('basic', 'advanced', 'skewing', 'speculative', 'aggressive')
=======

dse_registry = ('basic', 'advanced', 'aggressive')
>>>>>>> dse: Reorganise DSE directory structure; drop unused SpeculativeRewriter

modes = {
    'basic': BasicRewriter,
    'advanced': AdvancedRewriter,
<<<<<<< HEAD
    'skewing': SkewingRewriter,
    'speculative': SpeculativeRewriter,
=======
>>>>>>> dse: Reorganise DSE directory structure; drop unused SpeculativeRewriter
    'aggressive': AggressiveRewriter
}
"""The DSE transformation modes."""
# Possible needed FIX nsim
#configuration.add('dse', 'advanced', list(modes))
MAX_SKEW_FACTOR = 8
configuration.add('skew_factor', 0, range(MAX_SKEW_FACTOR))

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
<<<<<<< HEAD
                        namely 'basic', factorization, CIRE for time-invariants only.
        - ``skewing``: Apply skewing.
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
=======
                        namely 'basic', factorization, and cross-iteration redundancy
                        elimination ("CIRE") for time-invariants only.
        - ``aggressive``: Like 'advanced', but apply CIRE to time-varying
                          sub-expressions too.
                          Further, seek and drop cross-cluster redundancies (this
                          is the only pass that attempts to optimize *across*
                          clusters, rather than within a cluster).
                          The 'aggressive' mode may substantially increase the
                          symbolic processing time; it may or may not reduce the
                          JIT-compilation time; it may or may not improve the
                          overall runtime performance.
>>>>>>> dse: Reorganise DSE directory structure; drop unused SpeculativeRewriter
    """
    if not (mode is None or isinstance(mode, str)):
        raise ValueError("Parameter 'mode' should be a string, not %s." % type(mode))

    if mode is None or mode == 'noop':
        return clusters
    elif mode not in dse_registry:
        raise ValueError("Unknown rewrite 'mode' %s." % type(mode))
        #dse_warning("Unknown rewrite mode(s) %s" % mode)
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

    return processed.finalize()
