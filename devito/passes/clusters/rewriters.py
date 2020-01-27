import abc

from cached_property import cached_property

from devito.exceptions import InvalidOperator
from devito.parameters import configuration
from devito.passes.clusters import (dse_pass, cire, cse, factorize, extract_increments,
                                    extract_time_invariants, extract_sum_of_products)
from devito.symbolics import estimate_cost, freeze, pow_to_mul
from devito.tools import as_tuple, flatten

__all__ = ['dse_registry', 'rewrite']


class AbstractRewriter(object):

    """
    Transform a Cluster of SymPy expressions into one or more clusters with
    reduced operation count.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, template, platform):
        self.platform = platform

        assert callable(template)
        self.template = template

    def run(self, cluster):
        clusters = self._pipeline(as_tuple(cluster), self.template, self.platform)

        clusters = self._finalize(clusters)

        return clusters

    @abc.abstractmethod
    def _pipeline(self, clusters, *args):
        return

    @dse_pass
    def _finalize(self, cluster):
        """
        Finalize the DSE output: ::

            * Pow-->Mul. Convert integer powers in an expression to Muls,
              like a**2 => a*a.
            * Freezing. Make sure that subsequent SymPy operations applied to
              the expressions in ``cluster.exprs`` will not alter the effect of
              the DSE passes.
        """
        exprs = [pow_to_mul(e) for e in cluster.exprs]
        return cluster.rebuild([freeze(e) for e in exprs])


class BasicRewriter(AbstractRewriter):

    def _pipeline(self, clusters, *args):
        clusters = extract_increments(clusters, *args)

        return clusters


class AdvancedRewriter(BasicRewriter):

    def _pipeline(self, clusters, *args):
        clusters = extract_time_invariants(clusters, *args)
        clusters = cire(clusters, *args)
        clusters = cse(clusters, *args)
        clusters = factorize(clusters)

        return clusters


class AggressiveRewriter(AdvancedRewriter):

    def _pipeline(self, clusters, *args):
        clusters = extract_sum_of_products(clusters, *args)
        clusters = extract_time_invariants(clusters, *args)
        clusters = cire(clusters, *args)

        clusters = extract_sum_of_products(clusters, *args)
        clusters = cire(clusters, *args)
        clusters = extract_sum_of_products(clusters, *args)

        clusters = factorize(clusters)
        clusters = cse(clusters, *args)

        return clusters


class CustomRewriter(AggressiveRewriter):

    @cached_property
    def passes_mapper(self):
        return {
            'extract_sop': extract_sum_of_products,
            'factorize': factorize,
            'cire': cire,
            'cse': cse,
            'extract_invariants': extract_time_invariants,
            'extract_increments': extract_increments
        }

    def __init__(self, passes, template, platform):
        try:
            passes = passes.split(',')
        except AttributeError:
            # Already in tuple format
            if not all(i in self.passes_mapper for i in passes):
                raise InvalidOperator("Unknown passes `%s`" % str(passes))
        self.passes = passes
        super(CustomRewriter, self).__init__(template, platform)

    def _pipeline(self, clusters, *args):
        passes_mapper = self.passes_mapper

        for i in self.passes:
            clusters = passes_mapper[i](clusters, *args)

        return clusters


modes = {
    'basic': BasicRewriter,
    'advanced': AdvancedRewriter,
    'aggressive': AggressiveRewriter
}
"""The rewriter modes."""

dse_registry = tuple(modes)


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
