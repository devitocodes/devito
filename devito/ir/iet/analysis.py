"""
A collection of algorithms to analyze and decorate :class:`Iteration` in an
Iteration/Expression tree. Decoration comes in the form of :class:`IterationProperty`
objects, attached to Iterations in the Iteration/Expression tree. The algorithms
perform actual data dependence analysis.
"""

from collections import OrderedDict
from functools import cmp_to_key

from devito.ir.iet import (HaloSpot, SEQUENTIAL, PARALLEL, PARALLEL_IF_ATOMIC,
                           VECTOR, WRAPPABLE, AFFINE, REDUNDANT, MapIteration, FindNodes,
                           Transformer, retrieve_iteration_tree)
from devito.ir.support import Scope
from devito.tools import as_mapper, as_tuple, filter_ordered, flatten

__all__ = ['iet_analyze']


class Analysis(object):

    def __init__(self, iet):
        self.iet = iet
        self.properties = OrderedDict()

        self.trees = retrieve_iteration_tree(iet, mode='superset')
        self.scopes = OrderedDict([(k, Scope([i.expr for i in v]))
                                   for k, v in MapIteration().visit(iet).items()])

    def update(self, properties):
        for k, v in properties.items():
            self.properties.setdefault(k, []).append(v)


def propertizer(func):
    def wrapper(arg):
        analysis = Analysis(arg) if not isinstance(arg, Analysis) else arg
        func(analysis)
        return analysis
    return wrapper


def iet_analyze(iet):
    """
    Attach :class:`IterationProperty` to :class:`Iteration` objects within
    ``nodes``. The recognized IterationProperty decorators are listed in
    ``nodes.IterationProperty._KNOWN``.
    """
    analysis = mark_parallel(iet)
    analysis = mark_vectorizable(analysis)
    analysis = mark_wrappable(analysis)
    analysis = mark_affine(analysis)
    analysis = mark_halospots(analysis)

    # Decorate the Iteration/Expression tree with the found properties
    mapper = OrderedDict()
    for k, v in list(analysis.properties.items()):
        args = k.args
        properties = as_tuple(args.pop('properties')) + as_tuple(v)
        mapper[k] = k._rebuild(properties=properties, **args)
    processed = Transformer(mapper, nested=True).visit(iet)

    return processed


@propertizer
def mark_parallel(analysis):
    """Update the ``analysis`` detecting the ``SEQUENTIAL`` and ``PARALLEL``
    Iterations within ``analysis.iet``."""
    properties = OrderedDict()
    for tree in analysis.trees:
        for depth, i in enumerate(tree):
            if properties.get(i) is SEQUENTIAL:
                # Speed-up analysis
                continue

            if i.uindices:
                # Only ++/-- increments of iteration variables are supported
                properties.setdefault(i, []).append(SEQUENTIAL)
                continue

            # Get all dimensions up to and including Iteration /i/, grouped by Iteration
            dims = [filter_ordered(j.dimensions) for j in tree[:depth + 1]]
            # Get all dimensions up to and including Iteration /i-1/
            prev = flatten(dims[:-1])
            # Get all dimensions up to and including Iteration /i/
            dims = flatten(dims)

            # The i-th Iteration is PARALLEL if for all dependences (d_1, ..., d_n):
            # test0 := (d_1, ..., d_{i-1}) > 0, OR
            # test1 := (d_1, ..., d_i) = 0
            is_parallel = True

            # The i-th Iteration is PARALLEL_IF_ATOMIC if for all dependeces:
            # test0 OR test1 OR the write is an associative and commutative increment
            is_atomic_parallel = True

            for dep in analysis.scopes[i].d_all:
                test0 = len(prev) > 0 and any(dep.is_carried(d) for d in prev)
                test1 = all(dep.is_indep(d) for d in dims)
                test2 = all(dep.is_reduce_atmost(d) for d in prev) and dep.is_indep(i.dim)
                if not (test0 or test1 or test2):
                    is_parallel = False
                    if not dep.is_increment:
                        is_atomic_parallel = False
                        break

            if is_parallel:
                properties.setdefault(i, []).append(PARALLEL)
            elif is_atomic_parallel:
                properties.setdefault(i, []).append(PARALLEL_IF_ATOMIC)
            else:
                properties.setdefault(i, []).append(SEQUENTIAL)

    # Reduction (e.g, SEQUENTIAL takes priority over PARALLEL)
    priorities = {PARALLEL: 0, PARALLEL_IF_ATOMIC: 1, SEQUENTIAL: 2}
    properties = OrderedDict([(k, max(v, key=lambda i: priorities[i]))
                              for k, v in properties.items()])

    analysis.update(properties)


@propertizer
def mark_vectorizable(analysis):
    """Update the ``analysis`` detecting the ``VECTOR`` Iterations within
    ``analysis.iet``. An Iteration is VECTOR iff:

        * it's the innermost in an Iteration tree, AND
        * it's got at least an outer PARALLEL Iteration, AND
        * it's been marked as PARALLEL or all the accesses along its dimension
          are unit-strided.
        """
    for tree in analysis.trees:
        if len(tree) == 1:
            continue
        else:
            outer, innermost = tree[-2], tree[-1]
            if PARALLEL not in analysis.properties.get(outer, []):
                continue
            elif PARALLEL in analysis.properties.get(innermost, []):
                analysis.properties[innermost].append(VECTOR)
            else:
                accesses = analysis.scopes[innermost].accesses
                if not accesses:
                    continue
                if all(accesses[0][innermost.dim] == i[innermost.dim] for i in accesses):
                    analysis.update({innermost: VECTOR})


@propertizer
def mark_wrappable(analysis):
    """Update the ``analysis`` detecting the ``WRAPPABLE`` Iterations within
    ``analysis.iet``."""
    for i, scope in analysis.scopes.items():
        if not i.dim.is_Time:
            continue

        accesses = [a for a in scope.accesses if a.function.is_TimeFunction]

        # If not using modulo-buffered iteration, then `i` is surely not WRAPPABLE
        if not accesses or any(not a.function._time_buffering_default for a in accesses):
            continue

        stepping = {a.function.time_dim for a in accesses}
        if len(stepping) > 1:
            # E.g., with ConditionalDimensions we may have `stepping={t, tsub}`
            continue
        stepping = stepping.pop()

        # All accesses must be affine in `stepping`
        if any(not a.affine_if_present(stepping._defines) for a in accesses):
            continue

        # Pick the `back` and `front` slots accessed
        try:
            compareto = cmp_to_key(lambda a0, a1: a0.distance(a1, stepping))
            accesses = sorted(accesses, key=compareto)
            back, front = accesses[0][stepping], accesses[-1][stepping]
        except TypeError:
            continue

        # Check we're not accessing (read, write) always the same slot
        if back == front:
            continue

        accesses_back = [a for a in accesses if a[stepping] == back]

        # There must be NO writes to the `back` timeslot
        if any(a.is_write for a in accesses_back):
            continue

        # There must be NO further accesses to the `back` timeslot after
        # any earlier timeslot is written
        # Note: potentially, this can be relaxed by replacing "any earlier timeslot"
        # with the `front timeslot`
        if not all(all(d.sink is not a or d.source.lex_ge(a) for d in scope.d_flow)
                   for a in accesses_back):
            continue

        analysis.update({i: WRAPPABLE})


@propertizer
def mark_affine(analysis):
    """Update the ``analysis`` detecting the ``AFFINE`` Iterations within
    ``analysis.iet``."""
    properties = OrderedDict()
    for tree in analysis.trees:
        for i in tree:
            if i in properties:
                continue
            arrays = [a for a in analysis.scopes[i].accesses if not a.is_scalar]
            if all(a.is_regular and a.affine_if_present(i.dim._defines) for a in arrays):
                properties[i] = AFFINE

    analysis.update(properties)


@propertizer
def mark_halospots(analysis):
    """Update the ``analysis`` detecting the ``REDUNDANT`` HaloSpots within
    ``analysis.iet``."""
    properties = OrderedDict()

    def analyze(fmapper, scope):
        for f, hse in fmapper.items():
            if any(dep.cause & set(hse.loc_indices) for dep in scope.d_anti.project(f)):
                return False
        return True

    for i, scope in analysis.scopes.items():
        mapper = as_mapper(FindNodes(HaloSpot).visit(i), lambda hs: hs.halo_scheme)
        for k, v in mapper.items():
            if len(v) == 1:
                continue
            if analyze(k.fmapper, scope):
                properties.update({i: REDUNDANT for i in v[1:]})

    analysis.update(properties)
