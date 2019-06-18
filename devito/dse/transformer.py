from devito.ir.clusters import ClusterGroup, optimize
from devito.dse.promotion import scalarize
from devito.dse.rewriters import (BasicRewriter, AdvancedRewriter, AggressiveRewriter,
                                  CustomRewriter)
from devito.logger import dse as log
from devito.parameters import configuration
from devito.tools import flatten

__all__ = ['dse_registry', 'rewrite']

dse_registry = ('basic', 'advanced', 'aggressive')

modes = {
    'basic': BasicRewriter,
    'advanced': AdvancedRewriter,
    'aggressive': AggressiveRewriter
}
"""The DSE transformation modes."""

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
                        namely 'basic', factorization, and cross-iteration redundancy
                        elimination ("CIRE") for time-invariants only.
        - ``aggressive``: Like 'advanced', but apply CIRE to time-varying
                          sub-expressions too.
                          Further, seek and drop cross-cluster redundancies (this
                          is the only pass that attempts to optimize *across*
                          Clusters, rather than within a Cluster).
                          The 'aggressive' mode may substantially increase the
                          symbolic processing time; it may or may not reduce the
                          JIT-compilation time; it may or may not improve the
                          overall runtime performance.
    """
    if not (mode is None or isinstance(mode, str)):
        raise ValueError("Parameter 'mode' should be a string, not %s." % type(mode))

    if mode is None or mode == 'noop':
        return clusters

    # We use separate rewriters for dense and sparse clusters; sparse clusters have
    # non-affine index functions, thus making it basically impossible, in general,
    # to apply the more advanced DSE passes.
    # Note: the sparse rewriter uses the same template for temporaries as
    # the dense rewriter, thus temporaries are globally unique

    try:
        rewriter = modes[mode]()
    except KeyError:
        rewriter = CustomRewriter(mode)

    fallback = BasicRewriter(False, rewriter.template)
    states = [rewriter.run(c) if c.is_dense else fallback.run(c) for c in clusters]

    # Print out profiling information
    print_profiling(states)

    # Schedule and optimize the Rewriters-produced clusters
    clusters = ClusterGroup(optimize(flatten(i.clusters for i in states)))

    # Turn unnecessary temporary Arrays into scalars
    clusters = scalarize(clusters, rewriter.template)

    return ClusterGroup(clusters)


def print_profiling(states):
    """
    Print a summary of the applied transformations.
    """
    # Drop unprofiled clusters/states
    states = [i for i in states if i.ops]

    if configuration['profiling'] == 'advanced':
        tot_elapsed = 0.
        row = "%s [flops: %d, elapsed: %.2f s]"
        for n, i in enumerate(states):
            log(" >>\n     ".join(row % ("".join(filter(lambda c: not c.isdigit(),
                                                        k[1:])), i.ops[k], v)
                                  for k, v in i.timings.items()))
            tot_elapsed += sum(i.timings.values())
        log("[Total elapsed: %.2f s]" % tot_elapsed)
    else:
        # Shorter summary
        tot_elapsed = 0.
        row = "flops: %d >> %d (elapsed %.2f s)"
        rows = []
        for i in states:
            elapsed = sum(i.timings.values())
            tot_elapsed += elapsed
            keys = list(i.timings)
            rows.append(row % (i.ops[keys[0]], i.ops[keys[-1]], elapsed))
        rows = "\n     ".join(rows)
        log("%s\n     [Total elapsed: %.2f s]" % (rows, tot_elapsed))
