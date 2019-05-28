from collections import OrderedDict

from devito.ir.iet import Expression, ForeignExpression, FindNodes, Transformer
from devito.symbolics import FunctionFromPointer, ListInitializer, ccode, retrieve_indexed
from devito.tools import ctypes_pointer

__all__ = ['Offloaded', 'make_var_accesses', 'make_sharedptr_funcall']


class Offloaded(ForeignExpression):

    """
    Represent offloaded computation.
    """

    pass


def make_sharedptr_funcall(call, params, sharedptr):
    return FunctionFromPointer(call, FunctionFromPointer('get', sharedptr), params)


def make_var_accesses(node, yk_var_objs):
    """
    Construct a new Iteration/Expression based on ``node``, in which all
    :class:`types.Indexed` accesses have been converted into YASK var
    accesses.
    """

    def make_var_gets(expr):
        mapper = {}
        indexeds = retrieve_indexed(expr)
        data_carriers = [i for i in indexeds if i.base.function.from_YASK]
        for i in data_carriers:
            args = [ListInitializer([INT(make_var_gets(j)) for j in i.indices])]
            mapper[i] = make_sharedptr_funcall(namespace['code-var-get'], args,
                                               yk_var_objs[i.base.function.name])
        return expr.xreplace(mapper)

    mapper = {}
    for i, e in enumerate(FindNodes(Expression).visit(node)):
        if e.is_ForeignExpression:
            continue

        lhs, rhs = e.expr.args

        # RHS translation
        rhs = make_var_gets(rhs)

        # LHS translation
        if e.write.from_YASK:
            args = [rhs]
            args += [ListInitializer([INT(make_var_gets(i)) for i in lhs.indices])]
            call = namespace['code-var-add' if e.is_Increment else 'code-var-put']
            handle = make_sharedptr_funcall(call, args, yk_var_objs[e.write.name])
            processed = ForeignExpression(handle, e.dtype, is_Increment=e.is_Increment)
        else:
            # Writing to a scalar temporary
            processed = e._rebuild(expr=e.expr.func(lhs, rhs))

        mapper.update({e: processed})

    return Transformer(mapper).visit(node)


# YASK conventions
namespace = OrderedDict()
namespace['jit-hook'] = lambda i: 'hook_%s' % i
namespace['jit-soln'] = lambda i: 'soln_%s' % i
namespace['code-soln-name'] = lambda i: 'soln_%s' % i
namespace['code-soln-run'] = 'run_solution'
namespace['code-var-name'] = lambda i: "var_%s" % str(i)
namespace['code-var-get'] = 'get_element'
namespace['code-var-put'] = 'set_element'
namespace['code-var-add'] = 'add_to_element'
namespace['type-solution'] = ctypes_pointer('yask::yk_solution_ptr')
namespace['type-var'] = ctypes_pointer('yask::yk_var_ptr')
namespace['numa-put-local'] = -1
