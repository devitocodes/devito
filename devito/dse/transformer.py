from devito.dse.rewriters import (BasicRewriter, AdvancedRewriter, AggressiveRewriter,
                                  CustomRewriter)
from devito.parameters import configuration
from devito.symbolics import estimate_cost
from devito.tools import flatten

__all__ = ['dse_registry', 'rewrite']

dse_registry = ('basic', 'advanced', 'aggressive')

modes = {
    'basic': BasicRewriter,
    'advanced': AdvancedRewriter,
    'aggressive': AggressiveRewriter
}
"""The DSE transformation modes."""


def rewrite(clusters, template, **kwargs):
    """
    Given a sequence of N Clusters, produce a sequence of M Clusters with reduced
    operation count, with M >= N.

    Parameters
    ----------
    clusters : list of Cluster
        The Clusters to be transformed.
    template : callable
        A stateful function producing unique symbol names each time it is called.
    **kwargs
        * dse : str, optional
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
            Defaults to ``advanced``.
        * profiler : Profiler, optional
            User to record the impact of the transformations, including operation
            variation and turnaround time.
    """
    # Optional kwargs
    mode = kwargs.get('dse', 'advanced')
    platform = kwargs.get('platform', configuration['platform'])
    profiler = kwargs.get('profiler')

    if not (mode is None or isinstance(mode, str)):
        raise ValueError("Parameter 'mode' should be a string, not %s." % type(mode))

    if mode is None or mode == 'noop':
        return clusters

    # We use separate rewriters for dense and sparse clusters; sparse clusters have
    # non-affine index functions, thus making it basically impossible, in general,
    # to apply the more advanced DSE passes.
    try:
        rewriter = modes[mode](template, platform)
    except KeyError:
        rewriter = CustomRewriter(mode, template, platform)
    fallback = BasicRewriter(template, platform)

    processed = []
    for c in clusters:
        if c.is_dense:
            retval = rewriter.run(c)
            processed.extend(retval)

            profiler.record_ops_variation(estimate_cost(c.exprs),
                                          estimate_cost(flatten(i.exprs for i in retval)))
        else:
            processed.extend(fallback.run(c))

    return processed
