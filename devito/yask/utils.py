from collections import OrderedDict
import ctypes

from devito.cgen_utils import INT
from devito.ir.iet import Expression, ForeignExpression, FindNodes, Transformer
from devito.symbolics import FunctionFromPointer, ListInitializer, retrieve_indexed
from devito.tools import ctypes_pointer

__all__ = ['make_grid_accesses', 'make_sharedptr_funcall', 'split_increment']


def make_sharedptr_funcall(call, params, sharedptr):
    return FunctionFromPointer(call, FunctionFromPointer('get', sharedptr), params)


def make_grid_accesses(node, yk_grid_objs):
    """
    Construct a new Iteration/Expression based on ``node``, in which all
    :class:`types.Indexed` accesses have been converted into YASK grid
    accesses.
    """

    def make_grid_gets(expr):
        mapper = {}
        indexeds = retrieve_indexed(expr)
        data_carriers = [i for i in indexeds if i.base.function.from_YASK]
        for i in data_carriers:
            args = [ListInitializer([INT(make_grid_gets(j)) for j in i.indices])]
            mapper[i] = make_sharedptr_funcall(namespace['code-grid-get'], args,
                                               yk_grid_objs[i.base.function.name])
        return expr.xreplace(mapper)

    mapper = {}
    for i, e in enumerate(FindNodes(Expression).visit(node)):
        if e.is_ForeignExpression:
            continue

        lhs, rhs = e.expr.args

        # RHS translation
        rhs = make_grid_gets(rhs)

        # LHS translation
        if e.write.from_YASK:
            args = [rhs]
            args += [ListInitializer([INT(make_grid_gets(i)) for i in lhs.indices])]
            call = namespace['code-grid-add' if e.is_Increment else 'code-grid-put']
            handle = make_sharedptr_funcall(call, args, yk_grid_objs[e.write.name])
            processed = ForeignExpression(handle, e.dtype, is_Increment=e.is_Increment)
        else:
            # Writing to a scalar temporary
            processed = e._rebuild(expr=e.expr.func(lhs, rhs))

        mapper.update({e: processed})

    return Transformer(mapper).visit(node)


def split_increment(expr):
    """
    Split an increment of type: ::

        u->set_element(v + u->get_element(indices), indices)

    into its three main components, namely the target grid ``u``, the increment
    value ``v``, and the :class:`ListInitializer` ``indices``.

    :raises ValueError: If ``expr`` is not an increment or does not appear in
                        the normal form above.
    """
    if not isinstance(expr, FunctionFromPointer) or len(expr.params) != 2:
        raise ValueError
    target = expr.pointer
    expr, indices = expr.params
    if not isinstance(indices, ListInitializer):
        raise ValueError
    if not expr.is_Add or len(expr.args) != 2:
        raise ValueError
    values = [i for i in expr.args if not isinstance(i, FunctionFromPointer)]
    if not len(values) == 1:
        raise ValueError
    return target, values[0], indices


# YASK conventions
namespace = OrderedDict()
namespace['jit-hook'] = lambda i: 'hook_%s' % i
namespace['jit-soln'] = lambda i: 'soln_%s' % i
namespace['code-soln-name'] = lambda i: 'soln_%s' % i
namespace['code-soln-run'] = 'run_solution'
namespace['code-grid-name'] = lambda i: "grid_%s" % str(i)
namespace['code-grid-get'] = 'get_element'
namespace['code-grid-put'] = 'set_element'
namespace['code-grid-add'] = 'add_to_element'
namespace['type-solution'] = ctypes.POINTER(ctypes_pointer('yask::yk_solution_ptr'))
namespace['type-grid'] = ctypes.POINTER(ctypes_pointer('yask::yk_grid_ptr'))
namespace['numa-put-local'] = -1
