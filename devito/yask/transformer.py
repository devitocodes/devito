from devito.dimension import LoweredDimension
from devito.ir.iet import FindNodes, Expression
from devito.ir.support import Backward
from devito.logger import yask_warning as warning
from devito.symbolics import split_affine

from devito.yask import nfac

__all__ = ['yaskizer', 'make_yask_ast']


def yaskizer(trees, yc_soln):
    """
    Populate a YASK compiler solution with the :class:`Expression`s found in an IET.

    The necessary YASK grids are instantiated.

    :param trees: A sequence of offloadable :class:`IterationTree`s, in which the
                  Expressions are searched.
    :param yc_soln: The YASK compiler solution to be populated.
    """
    # Track all created YASK grids
    mapper = {}

    # It's up to Devito to organize the equations into a flow graph
    yc_soln.set_dependency_checker_enabled(False)

    processed = []
    for tree in trees:
        # All expressions within `tree`
        expressions = [i.expr for i in FindNodes(Expression).visit(tree.inner)]

        # Attach conditional expression for sub-domains
        conditions = [(i, []) for i in expressions]
        for i in tree:
            if not i.dim.is_Sub:
                continue

            # Can we express both Iteration extremes as
            # `FIRST(i.dim) + integer` OR `LAST(i.dim) + integer` ?
            # If not, one of the following lines will throw a TypeError exception
            lower_ofs, lower_sym = i.dim.offset_lower()
            upper_ofs, upper_sym = i.dim.offset_upper()

            if i.is_Parallel:
                # At this point, no issues are expected -- we should just be able to
                # build the YASK conditions under which to execute this parallel Iteration

                ydim = nfac.new_domain_index(i.dim.parent.name)

                # Handle lower extreme
                if lower_sym == i.dim.parent.symbolic_start:
                    node = nfac.new_first_domain_index(ydim)
                else:
                    node = nfac.new_last_domain_index(ydim)
                expr = nfac.new_add_node(node, nfac.new_const_number_node(lower_ofs))
                for _, v in conditions:
                    v.append(nfac.new_not_less_than_node(ydim, expr))

                # Handle upper extreme
                if upper_sym == i.dim.parent.symbolic_start:
                    node = nfac.new_first_domain_index(ydim)
                else:
                    node = nfac.new_last_domain_index(ydim)
                expr = nfac.new_add_node(node, nfac.new_const_number_node(upper_ofs))
                for _, v in conditions:
                    v.append(nfac.new_not_greater_than_node(ydim, expr))

            elif i.is_Sequential:
                # For sequential Iterations, the extent *must* be statically known,
                # otherwise we don't know how to handle this
                try:
                    int(i.extent())
                except TypeError:
                    raise NotImplementedError("Found sequential Iteration with "
                                              "statically unknown extent")
                assert lower_sym == upper_sym  # A corollary of getting up to this point
                n = lower_sym

                ydim = nfac.new_domain_index(i.dim.parent.name)
                if n == i.dim.parent.symbolic_start:
                    node = nfac.new_first_domain_index(ydim)
                else:
                    node = nfac.new_last_domain_index(ydim)

                if i.direction is Backward:
                    _range = range(upper_ofs, lower_ofs - 1, -1)
                else:
                    _range = range(lower_ofs, upper_ofs + 1)

                unwound = []
                for e, v in conditions:
                    for r in _range:
                        expr = nfac.new_add_node(node, nfac.new_const_number_node(r))
                        unwound.append((e, v + [nfac.new_equals_node(ydim, expr)]))
                conditions = unwound

        # Build the YASK equations as well as all necessary grids
        for k, v in conditions:
            yask_expr = make_yask_ast(k, yc_soln, mapper)

            if yask_expr is not None:
                processed.append(yask_expr)

                # Is there a sub-domain to attach ?
                if v:
                    condition = v.pop(0)
                    for i in v:
                        condition = nfac.new_and_node(condition, i)
                    yask_expr.set_cond(condition)

    # Add flow dependences to the offloaded equations
    # TODO: This can be improved by spotting supergroups ?
    for to, frm in zip(processed, processed[1:]):
        yc_soln.add_flow_dependency(frm, to)

    return mapper


def make_yask_ast(expr, yc_soln, mapper):

    def nary2binary(args, op):
        r = make_yask_ast(args[0], yc_soln, mapper)
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
            return mapper[function].new_grid_point([])
        elif function.is_Dimension:
            if expr.is_Time:
                return nfac.new_step_index(expr.name)
            elif expr.is_Space:
                return nfac.new_domain_index(expr.name)
            else:
                return nfac.new_misc_index(expr.name)
        else:
            # A DSE-generated temporary, which must have already been
            # encountered as a LHS of a previous expression
            assert function in mapper
            return mapper[function]
    elif expr.is_Indexed:
        # Create a YASK compiler grid if it's the first time we encounter a Function
        function = expr.base.function
        if function not in mapper:
            dimensions = [make_yask_ast(i, yc_soln, mapper) for i in function.indices]
            mapper[function] = yc_soln.new_grid(function.name, dimensions)
        # Convert the Indexed into a YASK grid access
        indices = []
        for i in expr.indices:
            if isinstance(i, LoweredDimension):
                indices.append(make_yask_ast(i.origin, yc_soln, mapper))
            elif i.is_integer:
                # Typically, if we end up here it's because we have a misc dimension
                indices.append(make_yask_ast(i, yc_soln, mapper))
            else:
                # We must always use the parent ("main") dimension when creating
                # YASK expressions
                af = split_affine(i)
                dim = af.var.parent if af.var.is_Derived else af.var
                indices.append(make_yask_ast(dim + af.shift, yc_soln, mapper))
        return mapper[function].new_grid_point(indices)
    elif expr.is_Add:
        return nary2binary(expr.args, nfac.new_add_node)
    elif expr.is_Mul:
        return nary2binary(expr.args, nfac.new_multiply_node)
    elif expr.is_Pow:
        base, exp = expr.as_base_exp()
        if not exp.is_integer:
            raise NotImplementedError("Non-integer powers unsupported in "
                                      "Devito-YASK translation")

        if int(exp) < 0:
            num, den = expr.as_numer_denom()
            return nfac.new_divide_node(make_yask_ast(num, yc_soln, mapper),
                                        make_yask_ast(den, yc_soln, mapper))
        elif int(exp) >= 1:
            return nary2binary([base] * exp, nfac.new_multiply_node)
        else:
            warning("0-power found in Devito-YASK translation? setting to 1")
            return nfac.new_const_number_node(1)
    elif expr.is_Equality:
        if expr.lhs.is_Symbol:
            function = expr.lhs.base.function
            assert function not in mapper
            mapper[function] = make_yask_ast(expr.rhs, yc_soln, mapper)
        else:
            return nfac.new_equation_node(*[make_yask_ast(i, yc_soln, mapper)
                                            for i in expr.args])
    else:
        raise NotImplementedError("Missing handler in Devito-YASK translation")
