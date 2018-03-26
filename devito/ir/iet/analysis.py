"""
A collection of algorithms to analyze and decorate :class:`Iteration` in an
Iteration/Expression tree. Decoration comes in the form of :class:`IterationProperty`
objects, attached to Iterations in the Iteration/Expression tree. The algorithms
perform actual data dependence analysis.
"""

from collections import OrderedDict
from functools import cmp_to_key

from devito.ir.iet import (Iteration, SEQUENTIAL, PARALLEL, VECTOR, WRAPPABLE,
                           MapIteration, NestedTransformer, retrieve_iteration_tree)
from devito.ir.support import Scope
from devito.tools import as_tuple, filter_ordered

__all__ = ['iet_analyze']


class Analysis(object):

    def __init__(self, iet):
        self.iet = iet
        self.properties = OrderedDict()

        self.trees = retrieve_iteration_tree(iet)
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

    # Decorate the Iteration/Expression tree with the found properties
    mapper = OrderedDict()
    for k, v in list(analysis.properties.items()):
        args = k.args
        properties = as_tuple(args.pop('properties')) + as_tuple(v)
        mapper[k] = Iteration(properties=properties, **args)
    processed = NestedTransformer(mapper).visit(iet)

    return processed


@propertizer
def mark_parallel(analysis):
    """Update the ``analysis`` detecting the ``SEQUENTIAL`` and ``PARALLEL``
    Iterations within ``analysis.iet``."""
    properties = OrderedDict()
    for tree in analysis.trees:
        for depth, i in enumerate(tree):
            if i in properties:
                continue
            # Get all tree dimensions, including the unbounded indices
            dims = []
            for j in tree[:depth + 1]:
                dims.append(j.dim)
                dims.extend([k.dim for k in j.uindices])
            dims = filter_ordered(dims)
            # The i-th Iteration is PARALLEL if for all dependences (d_1, ..., d_n):
            # (d_1, ..., d_{i-1}) > 0, OR
            # (d_1, ..., d_i) = 0
            is_sequential = False
            for dep in analysis.scopes[i].d_all:
                test0 = dims[:-1] and any(dep.is_carried(d) for d in dims[:-1])
                test1 = all(dep.is_independent(d) for d in dims)
                if not (test0 or test1):
                    is_sequential = True
                    break
            properties[i] = SEQUENTIAL if is_sequential else PARALLEL
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
    # All potential WRAPPABLEs are Stepping dimensions
    stepper = None
    for iteration in analysis.scopes:
        if not iteration.dim.is_Stepping:
            continue
        stepper = iteration
    if not stepper:
        return
    stepping = stepper.dim
    accesses = [i for i in analysis.scopes[stepper].accesses if stepping in i.findices]
    if not accesses:
        return
    # Pick the /back/ and /front/ slots accessed
    try:
        accesses = sorted(accesses, key=cmp_to_key(lambda i, j: i.distance(j, stepping)))
        back, front = accesses[0][stepping], accesses[-1][stepping]
    except TypeError:
        return
    if back == front:
        return
    # Finally check that all data dependences would be honored by using the
    # /front/ index function in place of the /back/ index function
    # There must be NO writes to the /back/ timeslot
    for access in analysis.scopes[stepper].accesses:
        if access.is_write and access[stepping] == back:
            return
    # All reads from the /front/ timeslot must not cause dependences with
    # the writes in the /back/ timeslot along the /i/ dimension
    for dep in analysis.scopes[stepper].d_flow:
        if dep.source[stepping] != front or dep.sink[stepping] != back:
            continue
        if dep.sink.lex_gt(dep.source) and\
                dep.source.section(stepping) != dep.sink.section(stepping):
            return
    analysis.update({stepper: WRAPPABLE})
