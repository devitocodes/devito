from collections import OrderedDict

from devito.ir.iet import (
    Conditional, Expression, ExpressionBundle, HaloSpot, Increment, Iteration, List,
    Section, Switch, SyncSpot
)
from devito.ir.support import GuardCaseSwitch, GuardSwitch
from devito.tools import as_mapper, timed_pass

__all__ = ['iet_build']


@timed_pass(name='build')
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
            exprs = []
            for e in i.exprs:
                if e.is_Increment:
                    exprs.append(Increment(e))
                else:
                    exprs.append(Expression(e, operation=e.operation))
            body = ExpressionBundle(i.ispace, i.ops, i.traffic, body=exprs)

        elif i.is_Conditional:
            if isinstance(i.guard, GuardSwitch):
                bundle, = queues.pop(i)
                cases, nodes = _unpack_switch_case(bundle)
                body = Switch(i.guard.arg, cases, nodes)
            else:
                body = Conditional(i.guard, queues.pop(i))

        elif i.is_Iteration:
            if i.dim.is_Virtual:
                body = List(body=queues.pop(i))
            else:
                body = Iteration(queues.pop(i), i.dim, i.limits,
                                 direction=i.direction, properties=i.properties,
                                 uindices=i.sub_iterators)

        elif i.is_Section:
            body = Section('section%d' % nsections, body=queues.pop(i))
            nsections += 1

        elif i.is_Halo:
            try:
                body = HaloSpot(queues.pop(i), i.halo_scheme)
            except KeyError:
                body = HaloSpot(None, i.halo_scheme)

        elif i.is_Sync:
            body = SyncSpot(i.sync_ops, body=queues.pop(i, None))

        queues.setdefault(i.parent, []).append(body)

    assert False


def _unpack_switch_case(bundle):
    """
    Helper to unpack an ExpressionBundle containing GuardCaseSwitch expressions
    into Switch cases and corresponding IET nodes.
    """
    assert bundle.is_ExpressionBundle
    assert all(isinstance(e.rhs, GuardCaseSwitch) for e in bundle.body)

    mapper = as_mapper(bundle.body, key=lambda e: e.rhs.case)

    cases = list(mapper)

    nodes = []
    for v in mapper.values():
        exprs = [e._rebuild(expr=e.expr._subs(e.rhs, e.rhs.arg)) for e in v]
        if len(exprs) > 1:
            nodes.append(List(body=exprs))
        else:
            nodes.append(*exprs)

    return cases, nodes
