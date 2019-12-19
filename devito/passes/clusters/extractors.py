from devito.passes.clusters.aliases import MIN_COST_ALIAS_INV
from devito.passes.clusters.utils import dse_pass, make_is_time_invariant
from devito.symbolics import (estimate_cost, q_leaf, q_sum_of_product, q_terminalop,
                              yreplace)
from devito.types import Scalar

__all__ = ['extract_increments', 'extract_time_invariants', 'extract_sum_of_products']


@dse_pass
def extract_increments(cluster, template, *args):
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
def extract_time_invariants(cluster, template, *args):
    """
    Extract time-invariant subexpressions, and assign them to temporaries.
    """
    make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
    rule = make_is_time_invariant(cluster.exprs)
    costmodel = lambda e: estimate_cost(e, True) >= MIN_COST_ALIAS_INV
    processed, found = yreplace(cluster.exprs, make, rule, costmodel, eager=True)

    return cluster.rebuild(processed)


@dse_pass
def extract_sum_of_products(cluster, template, *args):
    """
    Extract sub-expressions in sum-of-product form, and assign them to temporaries.
    """
    make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()
    rule = q_sum_of_product
    costmodel = lambda e: not (q_leaf(e) or q_terminalop(e))
    processed, _ = yreplace(cluster.exprs, make, rule, costmodel)

    return cluster.rebuild(processed)
