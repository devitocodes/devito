from devito.dimension import LoweredDimension
from devito.ir.iet import FindNodes, Expression, retrieve_iteration_tree
from devito.logger import yask_warning as warning
from devito.symbolics import split_affine

from devito.yask import nfac

__all__ = ['yaskizer']


def yaskizer(root, yc_soln):
    """
    Convert a SymPy expression into a YASK abstract syntax tree and create the
    necessay YASK grids.

    :param root: The outermost space :class:`Iteration` defining an offloadable
                 loop nest.
    :param yc_soln: The YASK compiler solution to which the new YASK grids are
                    attached.
    """
    # To be populated by `handle`
    mapper = {}

    trees = retrieve_iteration_tree(root)
    assert len(trees) == 1
    tree = trees[0]

    for i in FindNodes(Expression).visit(root):
        # Build the YASK AST
        yask_expr = handle(i.expr, yc_soln, mapper)

        if yask_expr is None:
            # A temporary, go on
            continue

        # Maybe set a conditional to apply the equation to a sub-domain only
        conditions = []
        for i in tree:
            if not i.dim.is_Sub:
                continue
            main = i.dim.parent
            # Create a node for the first index value along /main/
            ydim = nfac.new_domain_index(main.name)
            # Condition for lower offset
            first_ydim = nfac.new_first_domain_index(ydim)
            const = nfac.new_const_number_node(int(i.dim.symbolic_ofs_lower))
            lower = nfac.new_add_node(first_ydim, const)
            conditions.append(nfac.new_not_less_than_node(ydim, lower))
            # Condition for upper offset
            last_ydim = nfac.new_last_domain_index(ydim)
            const = nfac.new_const_number_node(int(i.dim.symbolic_ofs_upper))
            upper = nfac.new_add_node(last_ydim, const)
            conditions.append(nfac.new_not_greater_than_node(ydim, upper))
        # Finally concatenate the conditions with the logical AND `&` (if any)
        try:
            condition = conditions.pop(0)
            for i in conditions:
                condition = nfac.new_and_node(condition, i)
            yask_expr.set_cond(condition)
        except IndexError:
            pass

    return mapper


def handle(expr, yc_soln, mapper):

    def nary2binary(args, op):
        r = handle(args[0], yc_soln, mapper)
        return r if len(args) == 1 else op(r, nary2binary(args[1:], op))

    if expr.is_Integer:
        return nfac.new_const_number_node(int(expr))
    elif expr.is_Float:
        return nfac.new_const_number_node(float(expr))
    elif expr.is_Rational:
        a, b = expr.as_numer_denom()
        return nfac.new_const_number_node(float(a)/float(b))
    elif expr.is_Symbol:
        function = expr.base.function
        if function.is_Constant:
            if function not in mapper:
                mapper[function] = yc_soln.new_grid(function.name, [])
            return mapper[function].new_relative_grid_point([])
        else:
            # A DSE-generated temporary, which must have already been
            # encountered as a LHS of a previous expression
            assert function in mapper
            return mapper[function]
    elif expr.is_Indexed:
        function = expr.base.function
        if function not in mapper:
            if function.is_TimeFunction:
                dimensions = [nfac.new_step_index(function.indices[0].name)]
                dimensions += [nfac.new_domain_index(i.name)
                               for i in function.indices[1:]]
            else:
                dimensions = [nfac.new_domain_index(i.name)
                              for i in function.indices]
            mapper[function] = yc_soln.new_grid(function.name, dimensions)
        # Detect offset from dimension. E.g., in `[x+3,y+4]`, detect `[3,4]`
        indices = []
        for i, j in zip(expr.indices, function.indices):
            if isinstance(i, LoweredDimension):
                access = i.origin
            else:
                # SubDimension require this
                af = split_affine(i)
                dim = af.var.parent if af.var.is_Derived else af.var
                access = dim + af.shift
            indices.append(int(access - j))
        return mapper[function].new_relative_grid_point(indices)
    elif expr.is_Add:
        return nary2binary(expr.args, nfac.new_add_node)
    elif expr.is_Mul:
        return nary2binary(expr.args, nfac.new_multiply_node)
    elif expr.is_Pow:
        base, exp = expr.as_base_exp()
        if not exp.is_integer:
            warning("non-integer powers unsupported in Devito-YASK translation")
            raise NotImplementedError

        if int(exp) < 0:
            num, den = expr.as_numer_denom()
            return nfac.new_divide_node(handle(num, yc_soln, mapper),
                                        handle(den, yc_soln, mapper))
        elif int(exp) >= 1:
            return nary2binary([base] * exp, nfac.new_multiply_node)
        else:
            warning("0-power found in Devito-YASK translation? setting to 1")
            return nfac.new_const_number_node(1)
    elif expr.is_Equality:
        if expr.lhs.is_Symbol:
            function = expr.lhs.base.function
            assert function not in mapper
            mapper[function] = handle(expr.rhs, yc_soln, mapper)
        else:
            return nfac.new_equation_node(*[handle(i, yc_soln, mapper)
                                            for i in expr.args])
    else:
        warning("Missing handler in Devito-YASK translation")
        raise NotImplementedError
