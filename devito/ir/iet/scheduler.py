from collections import OrderedDict
from itertools import product

from devito.ir.iet import (Expression, Increment, Iteration, List, Conditional,
                           Section, HaloSpot, ExpressionBundle, FindNodes, FindSymbols,
                           Transformer, XSubs, make_efunc, retrieve_iteration_tree)
from devito.symbolics import IntDiv, xreplace_indices
from devito.tools import as_mapper, flatten, is_integer, split, timed_pass
from devito.types import ConditionalDimension

__all__ = ['iet_build', 'iet_lower_dims']


@timed_pass(name='lowering.IET.build')
def iet_build(stree):
    """
    Construct an Iteration/Expression tree(IET) from a ScheduleTree.
    """
    nsections = 0
    queues = OrderedDict()
    for i in stree.visit():
        if i == stree:
            # We hit this handle at the very end of the visit
            return List(body=queues.pop(i))

        elif i.is_Exprs:
            exprs = [Increment(e) if e.is_Increment else Expression(e) for e in i.exprs]
            body = ExpressionBundle(i.ispace, i.ops, i.traffic, body=exprs)

        elif i.is_Conditional:
            body = Conditional(i.guard, queues.pop(i))

        elif i.is_Iteration:
            # Order to ensure deterministic code generation
            uindices = sorted(i.sub_iterators, key=lambda d: d.name)
            # Generate Iteration
            body = Iteration(queues.pop(i), i.dim, i.limits, offsets=i.offsets,
                             direction=i.direction, properties=i.properties,
                             uindices=uindices)

        elif i.is_Section:
            body = Section('section%d' % nsections, body=queues.pop(i))
            nsections += 1

        elif i.is_Halo:
            body = HaloSpot(i.halo_scheme, body=queues.pop(i))

        queues.setdefault(i.parent, []).append(body)

    assert False


@timed_pass(name='lowering.IET.lower_dims')
def iet_lower_dims(iet):
    """
    Lower the DerivedDimensions in ``iet``.
    """
    iet = lower_stepping_dims(iet)
    iet = lower_conditional_dims(iet)
    iet, efuncs = lower_incr_dims(iet)

    return iet, efuncs


def lower_stepping_dims(iet):
    """
    Lower SteppingDimensions: index functions involving SteppingDimensions are
    turned into ModuloDimensions.

    Examples
    --------
    u[t+1, x] = u[t, x] + 1

    becomes

    u[t1, x] = u[t0, x] + 1
    """
    for i in FindNodes(Iteration).visit(iet):
        if not i.uindices:
            # Be quick: avoid uselessy reconstructing nodes
            continue
        # In an expression, there could be `u[t+1, ...]` and `v[t+1, ...]`, where
        # `u` and `v` are TimeFunction with circular time buffers (save=None) *but*
        # different modulo extent. The `t+1` indices above are therefore conceptually
        # different, so they will be replaced with the proper ModuloDimension through
        # two different calls to `xreplace`
        mindices = [d for d in i.uindices if d.is_Modulo]
        groups = as_mapper(mindices, lambda d: d.modulo)
        for k, v in groups.items():
            mapper = {d.origin: d for d in v}
            rule = lambda i: i.function.is_TimeFunction and i.function._time_size == k
            replacer = lambda i: xreplace_indices(i, mapper, rule)
            iet = XSubs(replacer=replacer).visit(iet)

    return iet


def lower_conditional_dims(iet):
    """
    Lower ConditionalDimensions: index functions involving ConditionalDimensions
    are turned into integer-division expressions.

    Examples
    --------
    u[t_sub, x] = u[time, x]

    becomes

    u[time / 4, x] = u[time, x]
    """
    cdims = [d for d in FindSymbols('free-symbols').visit(iet)
             if isinstance(d, ConditionalDimension)]
    mapper = {d: IntDiv(d.index, d.factor) for d in cdims}
    iet = XSubs(mapper).visit(iet)

    return iet


def lower_incr_dims(iet):
    """
    Lower IncrDimensions: Iterations over IncrDimensions are recast as
    ElementalFunctions; multiple ElementalCalls are inserted so as to iterate
    over both the "main" and any of the "remainder" regions induced by the
    IncrDimension's step (which, obviously, isn't known until runtime).
    """
    efuncs = []
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        iterations = [i for i in tree if i.dim.is_Incr]
        if not iterations:
            continue

        root = iterations[0]
        if root in mapper:
            continue

        outer, inner = split(iterations, lambda i: not i.dim.parent.is_Incr)

        # Compute the iteration ranges
        ranges = []
        for i in outer:
            maxb = i.symbolic_max - (i.symbolic_size % i.dim.step)
            ranges.append(((i.symbolic_min, maxb, i.dim.step),
                           (maxb + 1, i.symbolic_max, i.symbolic_max - maxb)))

        # Create the ElementalFunction
        dynamic_parameters = flatten((i.dim, i.step) for i in outer)
        dynamic_parameters.extend([i.step for i in inner if not is_integer(i.step)])
        efunc = make_efunc("bf%d" % len(mapper), root, dynamic_parameters)
        efuncs.append(efunc)

        # Create the ElementalCalls
        body = []
        for p in product(*ranges):
            dynamic_args_mapper = {}
            for i, (m, M, b) in zip(outer, p):
                dynamic_args_mapper[i.dim] = (m, M)
                dynamic_args_mapper[i.step] = (b,)
                for j in inner:
                    if j.dim.root is i.dim.root and not is_integer(j.step):
                        value = j.step if b is i.step else b
                        dynamic_args_mapper[j.step] = (value,)
            call = efunc.make_call(dynamic_args_mapper)
            body.append(List(body=call))

        mapper[root] = List(body=body)

    iet = Transformer(mapper).visit(iet)

    return iet, efuncs
