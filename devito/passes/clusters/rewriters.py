import abc
from functools import wraps

from cached_property import cached_property
from sympy import cos, sin

from devito.exceptions import InvalidOperator
from devito.passes.clusters.utils import make_is_time_invariant
from devito.symbolics import (bhaskara_cos, bhaskara_sin, estimate_cost, freeze,
                              pow_to_mul, q_leaf, q_sum_of_product, q_terminalop,
                              yreplace)
from devito.tools import as_tuple, flatten, timed_pass
from devito.types import Scalar

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
        clusters = self._extract_increments(clusters, *args)

        return clusters

    @dse_pass
    def _extract_increments(self, cluster, template, *args):
        """
        Extract the RHS of non-local tensor expressions performing an associative
        and commutative increment, and assign them to temporaries.
        """
        processed = []
        for e in cluster.exprs:
            if e.is_Increment and e.lhs.function.is_Input:
                handle = Scalar(name=template(), dtype=e.dtype).indexify()
                if e.rhs.is_Number or e.rhs.is_Symbol:
                    extracted = e.rhs
                else:
                    extracted = e.rhs.func(*[i for i in e.rhs.args if i != e.lhs])
                processed.extend([e.func(handle, extracted, is_Increment=False),
                                  e.func(e.lhs, handle)])
            else:
                processed.append(e)

        return cluster.rebuild(processed)

    @dse_pass
    def _optimize_trigonometry(self, cluster, *args):
        """
        Rebuild ``exprs`` replacing trigonometric functions with Bhaskara
        polynomials.
        """
        processed = []
        for expr in cluster.exprs:
            handle = expr.replace(sin, bhaskara_sin)
            handle = handle.replace(cos, bhaskara_cos)
            processed.append(handle)

        return cluster.rebuild(processed)


class AdvancedRewriter(BasicRewriter):

    def _pipeline(self, clusters, *args):
        from devito.passes.clusters import cire, cse, factorize

        clusters = self._extract_time_invariants(clusters, *args)
        clusters = cire(clusters, *args)
        clusters = cse(clusters, *args)
        clusters = factorize(clusters)

        return clusters

    @dse_pass
    def _extract_time_invariants(self, cluster, template, *args):
        """
        Extract time-invariant subexpressions, and assign them to temporaries.
        """
        make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
        rule = make_is_time_invariant(cluster.exprs)
        costmodel = lambda e: estimate_cost(e, True) >= 50  #TODO
        processed, found = yreplace(cluster.exprs, make, rule, costmodel, eager=True)

        return cluster.rebuild(processed)


class AggressiveRewriter(AdvancedRewriter):

    def _pipeline(self, clusters, *args):
        from devito.passes.clusters import cire, cse, factorize

        clusters = self._extract_sum_of_products(clusters, *args)
        clusters = self._extract_time_invariants(clusters, *args)
        clusters = cire(clusters, *args)

        clusters = self._extract_sum_of_products(clusters, *args)
        clusters = cire(clusters, *args)
        clusters = self._extract_sum_of_products(clusters, *args)

        clusters = factorize(clusters)
        clusters = cse(clusters, *args)

        return clusters

    @dse_pass
    def _extract_sum_of_products(self, cluster, template, *args):
        """
        Extract sub-expressions in sum-of-product form, and assign them to temporaries.
        """
        make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
        rule = q_sum_of_product
        costmodel = lambda e: not (q_leaf(e) or q_terminalop(e))
        processed, _ = yreplace(cluster.exprs, make, rule, costmodel)

        return cluster.rebuild(processed)


class CustomRewriter(AggressiveRewriter):

    @cached_property
    def passes_mapper(self):
        from devito.passes.clusters import cire, cse, factorize  #TODO

        return {
            'extract_sop': self._extract_sum_of_products,
            'factorize': factorize,
            'cire': cire,
            'cse': cse,
            'extract_invariants': self._extract_time_invariants,
            'extract_increments': self._extract_increments,
            'opt_transcedentals': self._optimize_trigonometry
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
