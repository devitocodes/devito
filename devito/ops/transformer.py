from devito.ir.iet import Callable, Expression, FindNodes
from devito.ops.node_factory import OPSNodeFactory


def opsit(trees, count):
    node_factory = OPSNodeFactory()
    expressions = []
    for tree in trees:
        expressions.extend([
            Expression(make_ops_ast(i.expr, node_factory))
            for i in FindNodes(Expression).visit(tree.inner)
        ])

    arguments = []
    for exp in expressions:
        func = [f for f in exp.functions if f.name != "OPS_ACC_size"]
        for f in func:
            f.is_OPS = True
            f.is_LocalObject = True
        arguments.extend(func)

    callable_kernel = Callable("Kernel{}".format(count), expressions, "void", arguments)

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
