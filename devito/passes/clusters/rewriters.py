import abc
from functools import wraps

from cached_property import cached_property

from devito.exceptions import InvalidOperator
from devito.symbolics import freeze, pow_to_mul
from devito.tools import as_tuple, flatten, timed_pass

__all__ = ['BasicRewriter', 'AdvancedRewriter', 'AggressiveRewriter', 'CustomRewriter']


def dse_pass(func):
    @wraps(func)
    def wrapper(*args):
        if timed_pass.is_enabled():
            maybe_timed = lambda *_args: timed_pass(func, func.__name__)(*_args)
        else:
            maybe_timed = lambda *_args: func(*_args)
        args = list(args)
        maybe_self = args.pop(0)
        if isinstance(maybe_self, AbstractRewriter):
            # Instance method
            clusters = args.pop(0)
            processed = [maybe_timed(maybe_self, c, *args) for c in clusters]
        else:
            # Pure function
            processed = [maybe_timed(c, *args) for c in maybe_self]
        return flatten(processed)
    return wrapper


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
        from devito.passes.clusters import extract_increments  #TODO
        clusters = extract_increments(clusters, *args)

        return clusters


class AdvancedRewriter(BasicRewriter):

    def _pipeline(self, clusters, *args):
        from devito.passes.clusters import cire, cse, factorize, extract_time_invariants

        clusters = extract_time_invariants(clusters, *args)
        clusters = cire(clusters, *args)
        clusters = cse(clusters, *args)
        clusters = factorize(clusters)

        return clusters


class AggressiveRewriter(AdvancedRewriter):

    def _pipeline(self, clusters, *args):
        from devito.passes.clusters import (cire, cse, factorize, extract_time_invariants,
                                            extract_sum_of_products)

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
        from devito.passes.clusters import (cire, cse, factorize, extract_time_invariants,
                                            extract_sum_of_products, extract_increments)  #TODO

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
