from collections import defaultdict

from devito.ir.iet import Callable
from devito.ir.iet.nodes import Expression, ClusterizedEq
from devito.ir.iet.visitors import FindNodes

from devito.ops.node_factory import OPSNodeFactory
from devito.ops.utils import namespace


def opsit(trees):
    """
    Populates the tree with OPS instructions.

    Parameters
    ----------
    trees : :class:`IterationTree`
        A sequence of offloadable :class:`IterationTree`s, in which the
        expressions are searched.

    Returns
    -------
    list of :class:`Callable`
        Kernels to be inserted into the iet.
    """

    # New kernel functions to be added.
    new_functions = []

    # kernel_{id} : list of processed expressions.
    processed = defaultdict(list)

    # Translate all expressions for each tree.
    for index, tree in enumerate(trees):

        # All expressions whithin `tree`
        expressions = [i.expr for i in FindNodes(Expression).visit(tree.inner)]

        # OPS nodes factory.
        nfops = OPSNodeFactory()

        for expression in expressions:
            ops_expr = make_ops_ast(expression, nfops)
            processed[namespace['ops-kernel'](index)].append(ops_expr)

    # Each tree generates a new OPS kernel function.
    for name, expressions in processed.items():
        # FIXME : Arguments are empty
        new_functions.append(Callable(name,
                                      Expression(ClusterizedEq(*expressions)),
                                      namespace['ops-kernel-retval'],
                                      []))

    return new_functions


def make_ops_ast(expr, nfops):
    """
    Transform an devito expression into an OPS expression.
    Only the interested nodes are changed.

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
        return nfops.new_grid(expr)
    else:
        return expr.func(*[make_ops_ast(i, nfops) for i in expr.args])
