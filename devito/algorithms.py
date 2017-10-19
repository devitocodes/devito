from collections import OrderedDict

from devito.dse import as_symbol, retrieve_terminals
from devito.nodes import Iteration, SEQUENTIAL, PARALLEL, VECTOR
from devito.tools import as_tuple
from devito.visitors import FindSections, IsPerfectIteration, NestedTransformer


def analyze_iterations(nodes):
    """
    Attach :class:`IterationProperty` to :class:`Iteration` nodes within
    ``nodes`` that verify one or more of the following properties.

        * sequential (attach SEQUENTIAL): In no way the iterations can be
          executed in parallel, unless techniques such as skewing are applied.
        * fully-parallel (attach PARALLEL): As the name suggests, an Iteration
          of this kind has no dependencies across its iterations.
        * vectorizable (attach VECTOR): Innermost fully-parallel Iterations
          are also marked as vectorizable.
    """
    sections = FindSections().visit(nodes)

    # The analysis below may return "false positives" (ie, absence of fully-
    # parallel or OSIP trees when this is actually false), but this should
    # never be the case in practice, given the targeted stencil codes.
    mapper = OrderedDict()
    for tree, nexprs in sections.items():
        exprs = [e.expr for e in nexprs]

        # "Prefetch" objects to speed up the analsys
        terms = {e: tuple(retrieve_terminals(e.rhs)) for e in exprs}

        # Determine whether the Iteration tree ...
        is_FP = True  # ... is fully parallel (FP)
        is_OP = True  # ... has an outermost parallel dimension (OP)
        is_OSIP = True  # ... is outermost-sequential, inner-parallel (OSIP)
        is_US = True  # ... has a unit-strided innermost dimension (US)
        for lhs in [e.lhs for e in exprs if not e.lhs.is_Symbol]:
            for e in exprs:
                for i in [j for j in terms[e] if as_symbol(j) == as_symbol(lhs)]:
                    is_FP &= lhs.indices == i.indices

                    is_OP &= lhs.indices[0] == i.indices[0] and\
                        all(lhs.indices[0].free_symbols.isdisjoint(j.free_symbols)
                            for j in i.indices[1:])  # not A[x,y] = A[x,x+1]

                    is_US &= lhs.indices[-1] == i.indices[-1]

                    lhs_function, i_function = lhs.base.function, i.base.function
                    is_OSIP &= lhs_function.indices[0] == i_function.indices[0] and\
                        (lhs.indices[0] != i.indices[0] or len(lhs.indices) == 1 or
                         lhs.indices[1] == i.indices[1])

        # Build a node->property mapper
        if is_FP:
            for i in tree:
                mapper.setdefault(i, []).append(PARALLEL)
        elif is_OP:
            mapper.setdefault(tree[0], []).append(PARALLEL)
        elif is_OSIP:
            mapper.setdefault(tree[0], []).append(SEQUENTIAL)
            for i in tree[1:]:
                mapper.setdefault(i, []).append(PARALLEL)
        if IsPerfectIteration().visit(tree[-1]) and (is_FP or is_OSIP or is_US):
            # Vectorizable
            if len(tree) > 1 and SEQUENTIAL not in mapper.get(tree[-2], []):
                # Heuristic: there's at least an outer parallel Iteration
                mapper.setdefault(tree[-1], []).append(VECTOR)

    # Store the discovered properties in the Iteration/Expression tree
    for k, v in list(mapper.items()):
        args = k.args
        # SEQUENTIAL kills PARALLEL
        properties = SEQUENTIAL if (SEQUENTIAL in v or not k.is_Linear) else v
        properties = as_tuple(args.pop('properties')) + as_tuple(properties)
        mapper[k] = Iteration(properties=properties, **args)
    nodes = NestedTransformer(mapper).visit(nodes)

    return nodes
