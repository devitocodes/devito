import numpy as np
import itertools

from devito import Eq
from devito.symbolics import ListInitializer, Literal
from devito.ir.equations import ClusterizedEq
from devito.ir.iet.nodes import Callable, Expression
from devito.ir.iet.visitors import FindNodes
from devito.types import Array, DefaultDimension, Symbol
from devito.ops.node_factory import OPSNodeFactory
from devito.ops.types import OpsAccessible, OpsStencil
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

    parameters = sorted(node_factory.ops_params, key=lambda i: (i.is_Constant, i.name))

    ops_kernel = Callable(
        namespace['ops_kernel'](count),
        ops_expressions,
        "void",
        parameters
    )

    stencil_arrays_initializations = itertools.chain(*[
        to_ops_stencil(p, node_factory.ops_args_accesses[p])
        for p in parameters if isinstance(p, OpsAccessible)
    ])

    pre_time_loop = stencil_arrays_initializations

    return pre_time_loop, ops_kernel


def to_ops_stencil(param, accesses):
    dims = len(accesses[0])
    pts = len(accesses)
    stencil_name = namespace['ops_stencil_name'](dims, param.name, pts)

    stencil_array = Array(
        name=stencil_name,
        dimensions=(DefaultDimension(name='len', default_value=dims * pts),),
        dtype=np.int32,
    )

    ops_stencil = OpsStencil(stencil_name.upper())

    return [
        Expression(ClusterizedEq(Eq(
            stencil_array,
            ListInitializer(list(itertools.chain(*accesses)))
        ))),
        Expression(ClusterizedEq(Eq(
            ops_stencil,
            namespace['ops_decl_stencil'](
                dims,
                pts,
                Symbol(stencil_array.name),
                Literal('"%s"' % stencil_name.upper())
            )
        )))
    ]


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
