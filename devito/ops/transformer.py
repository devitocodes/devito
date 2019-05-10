from devito.ir.iet import Expression, FindNodes
from devito.ops.node_factory import OPSNodeFactory
from devito.ops.nodes import OPSKernel
from devito.ops.utils import namespace


def opsit(trees, count):
    node_factory = OPSNodeFactory()
    expressions = []
    for tree in trees:
        expressions.extend([
            Expression(make_ops_ast(i.expr, node_factory))
            for i in FindNodes(Expression).visit(tree.inner)
        ])

    arguments = set()
    to_remove = []
    for exp in expressions:
        func = [f for f in exp.functions if f.name != "OPS_ACC_size"]
        for f in func:
            f.is_OPS = True
        arguments |= set(func)
        if exp.is_scalar_assign:
            to_remove.append(exp.write)

    arguments -= set(to_remove)

    callable_kernel = OPSKernel(
        namespace['ops_kernel'](count),
        expressions,
        "void",
        list(arguments)
    )

    return callable_kernel, None


def make_ops_ast(expr, nfops):
    """
    Transform a devito expression into an OPS expression.
    Only the interested nodes are rebuilt.

    Parameters
    ----------
    expr : :class:`Node`
        Initial tree node.
    nfops : :class:`OPSNodeFactory`
        Generate OPS specific nodes.

    Returns
    -------
    :class:`Node`
        Expression alredy translated to OPS syntax.
    """

    if expr.is_Symbol or expr.is_Number:
        return expr
    elif expr.is_Indexed:
        return nfops.new_ops_arg(expr)
    else:
        return expr.func(*[make_ops_ast(i, nfops) for i in expr.args])
