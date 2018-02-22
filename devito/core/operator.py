from __future__ import absolute_import

from devito.core.autotuning import autotune
from devito.cgen_utils import printmark
from devito.ir.iet import List, Transformer, filter_iterations, retrieve_iteration_tree
from devito.ir.support import compute_domain_misalignment
from devito.operator import OperatorRunnable
from devito.symbolics import retrieve_indexed
from devito.tools import flatten

__all__ = ['Operator']


class OperatorCore(OperatorRunnable):

    def _specialize_exprs(self, expressions, subs):
        """
        Transform the SymPy expressions in input to the Operator into a
        backend-specific representation.

        Three tasks are carried out: ::

            * Indexification (:class:`Function` --> :class:`Indexed`).
            * Application of user-provided substitution rules.
            * Translation of array accesses w.r.t. the computational domain.
              This ensures that computational domain and dimension indices are
              aligned.

        The latter task requires some thought. One may think that adding the
        halo+padding region to the index functions would suffice. There is,
        however, a complication. Consider for example the function `u(t, x, y)`,
        with halo+padding given by ``((0, 0), (3, 3), (3, 3))`` -- see also
        :meth:`Function._offset_domain` for more info about the semantics of the
        halo and padding tuples. A user may express an iteration update in many
        different formats, such as ``u[t+1, x, y] = f(u[t, x, y], u[t-1, x, y])``
        or ``u[t+2, x, y] = f(u[t+1, x, y], u[t, x, y])``. Not only are these
        two formats mathematically meaningful, but they are also considered
        equivalent by Devito. Thus, if the user writes: ::

            .. code-block::
                op = Operator(...)
                op.apply(t_start=2, t_end=4)

        then 3 timesteps need to be computed -- in particular, ``u[2, ...],
        u[3, ...], u[4, ....]`` -- no matter whether the left hand side of the
        input equation is expressed as ``u[t+1, ...] = ...`` rather than as
        ``u[t+2, ...] = ...`` or ``u[t, ...] = ...``.

        Thus, for the above scenario, the compiler will eventually generate: ::

            .. code-block::
                for t = t_s to t_e
                  for x = x_s to x_e
                    for y = y_s to y_e
                      u[t, x+3, y+3] = f(u[t-1, x+3, y+3], u[t-2, x+3, y+3])
        """
        expressions = super(OperatorCore, self)._specialize_exprs(expressions, subs)

        # Determine Function-wise shifting
        constraints = compute_domain_misalignment(expressions)
        mapper = {}
        for e in expressions:
            for indexed in retrieve_indexed(e):
                f = indexed.base.function
                if not f.is_SymbolicFunction:
                    continue
                subs = {i: i + gap.left - constraints.get(f, {}).get(d, 0) for i, d, gap
                        in zip(indexed.indices, f.dimensions, f._offset_domain)}
                mapper[indexed] = indexed.xreplace(subs)

        # Apply shifting
        expressions = [e.xreplace(mapper) for e in expressions]

        return expressions

    def _autotune(self, args):
        """
        Use auto-tuning on this Operator to determine empirically the
        best block sizes when loop blocking is in use.
        """
        if self.dle_flags.get('blocking', False):
            return autotune(self, args, self.dle_args)
        else:
            return args


class OperatorDebug(OperatorCore):
    """
    Decorate the generated code with useful print statements.
    """

    def __init__(self, expressions, **kwargs):
        super(OperatorDebug, self).__init__(expressions, **kwargs)
        self._includes.append('stdio.h')

        # Minimize the trip count of the sequential loops
        iterations = set(flatten(retrieve_iteration_tree(self.body)))
        mapper = {i: i._rebuild(limits=(max(i.offsets) + 2))
                  for i in iterations if i.is_Sequential}
        self.body = Transformer(mapper).visit(self.body)

        # Mark entry/exit points of each non-sequential Iteration tree in the body
        iterations = [filter_iterations(i, lambda i: not i.is_Sequential, 'any')
                      for i in retrieve_iteration_tree(self.body)]
        iterations = [i[0] for i in iterations if i]
        mapper = {t: List(header=printmark('In nest %d' % i), body=t)
                  for i, t in enumerate(iterations)}
        self.body = Transformer(mapper).visit(self.body)


class Operator(object):

    def __new__(cls, *args, **kwargs):
        cls = OperatorDebug if kwargs.pop('debug', False) else OperatorCore
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        return obj
