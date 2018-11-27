from devito.ir.iet import Callable
from devito.ir.iet.nodes import Expression, ClusterizedEq
from devito.ir.iet.visitors import FindNodes

from devito.ops.node_factory import OpsNodeFactory
from devito.ops.utils import namespace


def opsit(trees):
    """
    Populate the tree with OPS instructions.

    :param trees: A sequence of offloadable :class: `IterationTree`s, in which the
                  Expressions are searched.
    """
    # Track all OPS kernels created
    mapper = {}
    processed = []
    for tree in trees:
        # All expressions whithin `tree`
        expressions = [i.expr for i in FindNodes(Expression).visit(tree.inner)]

        # Attach conditional expression for sub-domains
        conditions = [(i, []) for i in expressions]

        # Only one node factory for all expression so we can keep track
        # of all kernels generated.
        nfops = OpsNodeFactory()

        count = 0
        for k, v in conditions:
            arguments = []
            ops_expr = make_ops_ast(k, nfops, mapper, arguments)
            ops_kernel = create_new_ops_kernel(count, ops_expr, arguments)
            count += 1

            processed.append(ops_kernel)

    return processed


def make_ops_ast(expr, nfops, mapper, arguments):

    def nary2binary(args, op):
        r = make_ops_ast(args[0], nfops, mapper, arguments)
        return r if len(args) == 1 else op(r, nary2binary(args[1:], op))

    if expr.is_Integer:
        return nfops.new_int_node(int(expr))
    elif expr.is_Float:
        return nfops.new_float_node(float(expr))
    elif expr.is_Rational:
        a, b = expr.as_numer_denom()
        return nfops.new_rational_node(float(a)/float(b))
    elif expr.is_Symbol:
        # FIXME Yask is adding this part to the mapper... should we?
        # TODO Should i differentiate dimensions from constants
        #      like yask? Need a more complex example for test.
        return nfops.new_symbol(expr.name)
    elif expr.is_Mul:
        return nary2binary(expr.args, nfops.new_mul_node)
    elif expr.is_Add:
        return nary2binary(expr.args, nfops.new_add_node)
    elif expr.is_Pow:
        base, exp = expr.as_base_exp()

        if not exp.is_integer:
            raise NotImplementedError("Non-integer powers unsupported in "
                                      "Devito-OPS translation")
        if int(exp) < 0:
            num, den = expr.as_numer_denom()
            return nfops.new_divide_node(make_ops_ast(num, nfops, mapper, arguments),
                                         make_ops_ast(den, nfops, mapper, arguments))
        elif int(exp) >= 1:
            return nary2binary([base] * exp, nfops.new_mul_node)
    elif expr.is_Equality:
        if expr.lhs.is_Symbol:
            function = expr.lhs.base.function
            mapper[function] = make_ops_ast(expr.rhs, nfops, mapper, arguments)
        else:
            return nfops.new_equation_node(*[make_ops_ast(i, nfops, mapper, arguments)
                                             for i in expr.args])
    elif expr.is_Indexed:
        dimensions = [make_ops_ast(i, nfops, mapper, arguments)
                      for i in expr.indices]

        grid_access = nfops.new_grid(expr, dimensions)

        return grid_access
    else:
        print(expr)
        raise NotImplementedError("Missing handler in Devito-OPS translation")


def create_new_ops_kernel(count, expr, arguments):
    """
        Creates a Callable object responsable for defining the ops kernel method.

        :param count: number of ops kernel being created.
        :param expr: OPS expression that will be inside the method.
        :param arguments: OPS method arguments.
    """

    return Callable(namespace['ops-kernel'](count),
                    Expression(ClusterizedEq(expr)),
                    namespace['ops-kernel-retval'],
                    arguments)
