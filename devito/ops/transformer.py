from devito.ir.iet.nodes import Callable, Expression, List
from devito.ir.iet.visitors import FindNodes, FindSymbols
from devito.ops.node_factory import OPSNodeFactory
from devito.ops.utils import namespace


def opsit(trees, count):
    """
    Given an affine tree, generate a Callable representing an OPS Kernel.

    Parameters
    ----------
    tree : IterationTree
        IterationTree containing the loop to extract into an OPS Kernel
    count : int
        Generated kernel counters
    """
    node_factory = OPSNodeFactory()
    expressions = []
    ops_expressions = []

    for tree in trees:
        expressions.extend(FindNodes(Expression).visit(tree.inner))

    for expr in expressions:
        ops_expressions.append(Expression(make_ops_ast(expr.expr, node_factory)))

    parameters = FindSymbols('symbolics').visit(List(body=ops_expressions))
    to_remove = FindSymbols('defines').visit(List(body=expressions))
    parameters = [p for p in parameters if p not in to_remove]
    parameters = sorted(parameters, key=lambda i: (i.is_Constant, i.name))

    ops_kernel = Callable(
        namespace['ops-kernel'](count),
        ops_expressions,
        "void",
        parameters
    )

    return ops_kernel


def make_ops_ast(expr, nfops, is_write=False):
    """
    Transform a devito expression into an OPS expression.
    Only the interested nodes are rebuilt.

    Parameters
    ----------
    expr : Node
        Initial tree node.
    nfops : OPSNodeFactory
        Generate OPS specific nodes.

    Returns
    -------
    Node
        Expression alredy translated to OPS syntax.
    """

    if expr.is_Symbol:
        if expr.is_Constant:
            return nfops.new_ops_gbl(expr)
        return expr
    if expr.is_Number:
        return expr
    elif expr.is_Indexed:
        return nfops.new_ops_arg(expr, is_write)
    elif expr.is_Equality:
        res = expr.func(
            make_ops_ast(expr.lhs, nfops, True),
            make_ops_ast(expr.rhs, nfops)
        )
        return res
    else:
        return expr.func(*[make_ops_ast(i, nfops) for i in expr.args])
