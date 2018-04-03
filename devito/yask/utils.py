import cgen as c
import ctypes

from devito.cgen_utils import INT, ccode
from devito.ir.iet import Element, Expression, FindNodes, Transformer
from devito.symbolics import FunctionFromPointer, ListInitializer, retrieve_indexed

from devito.yask import namespace


def make_sharedptr_funcall(call, params, sharedptr):
    return FunctionFromPointer(call, FunctionFromPointer('get', sharedptr), params)


def make_grid_accesses(node):
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
            name = namespace['code-grid-name'](i.base.function.name)
            args = [ListInitializer([INT(make_grid_gets(j)) for j in i.indices])]
            mapper[i] = make_sharedptr_funcall(namespace['code-grid-get'], args, name)
        return expr.xreplace(mapper)

    mapper = {}
    for i, e in enumerate(FindNodes(Expression).visit(node)):
        lhs, rhs = e.expr.args

        # RHS translation
        rhs = make_grid_gets(rhs)

        # LHS translation
        if e.write.from_YASK:
            name = namespace['code-grid-name'](e.write.name)
            args = [rhs]
            args += [ListInitializer([INT(make_grid_gets(i)) for i in lhs.indices])]
            handle = make_sharedptr_funcall(namespace['code-grid-put'], args, name)
            processed = Element(c.Statement(ccode(handle)))
        else:
            # Writing to a scalar temporary
            processed = Expression(e.expr.func(lhs, rhs))

        mapper.update({e: processed})

    return Transformer(mapper).visit(node)


def rawpointer(obj):
    """Return a :class:`ctypes.c_void_p` pointing to ``obj``."""
    return ctypes.cast(int(obj), ctypes.c_void_p)
