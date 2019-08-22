import numpy as np
import itertools

from sympy import sympify
from sympy.core.numbers import Zero

from devito import Eq
from devito.ir.equations import ClusterizedEq
from devito.ir.iet.nodes import Call, Callable, Expression, IterationTree
from devito.ir.iet.visitors import FindNodes
from devito.ops.node_factory import OPSNodeFactory
from devito.ops.types import Array, OpsAccessible, OpsDat, OpsStencil
from devito.ops.utils import namespace
from devito.symbolics import Byref, ListInitializer, Literal
from devito.tools import dtype_to_cstr
from devito.types import Constant, DefaultDimension, Symbol


def opsit(trees, count, name_to_ops_dat, block, dims):
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
        ops_expressions.append(Expression(
            make_ops_ast(expr.expr, node_factory)))

    parameters = sorted(node_factory.ops_params,
                        key=lambda i: (i.is_Constant, i.name))

    stencil_arrays_initializations = []
    par_to_ops_stencil = {}

    for p in parameters:
        if isinstance(p, OpsAccessible):
            stencil, initialization = to_ops_stencil(
                p, node_factory.ops_args_accesses[p])

            par_to_ops_stencil[p] = stencil
            stencil_arrays_initializations.append(initialization)

    ops_kernel = Callable(
        namespace['ops_kernel'](count),
        ops_expressions,
        "void",
        parameters
    )

    ops_par_loop_init, ops_par_loop_call = create_ops_par_loop(
        trees, ops_kernel, parameters, block,
        name_to_ops_dat, par_to_ops_stencil, dims
    )

    pre_time_loop = stencil_arrays_initializations + ops_par_loop_init

    return pre_time_loop, ops_kernel, ops_par_loop_call


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

    return ops_stencil, [
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


def create_ops_dat(f, name_to_ops_dat, block):
    ndim = f.ndim - (1 if f.is_TimeFunction else 0)

    dim = Array(
        name=namespace['ops_dat_dim'](f.name),
        dimensions=(DefaultDimension(name='dim', default_value=ndim),),
        dtype=np.int32,
        scope='stack'
    )
    base = Array(
        name=namespace['ops_dat_base'](f.name),
        dimensions=(DefaultDimension(name='base', default_value=ndim),),
        dtype=np.int32,
        scope='stack'
    )
    d_p = Array(
        name=namespace['ops_dat_d_p'](f.name),
        dimensions=(DefaultDimension(name='d_p', default_value=ndim),),
        dtype=np.int32,
        scope='stack'
    )
    d_m = Array(
        name=namespace['ops_dat_d_m'](f.name),
        dimensions=(DefaultDimension(name='d_m', default_value=ndim),),
        dtype=np.int32,
        scope='stack'
    )

    res = []
    base_val = [Zero() for i in range(ndim)]

    # If f is a TimeFunction we need to create a ops_dat for each time stepping
    # variable (eg: t1, t2)
    if f.is_TimeFunction:
        time_pos = f._time_position
        time_index = f.indices[time_pos]
        time_dims = f.shape[time_pos]

        dim_shape = sympify(f.shape[:time_pos] + f.shape[time_pos + 1:])
        padding = f.padding[:time_pos] + f.padding[time_pos + 1:]
        halo = f.halo[:time_pos] + f.halo[time_pos + 1:]
        d_p_val = tuple(sympify([p[0] + h[0] for p, h in zip(padding, halo)]))
        d_m_val = tuple(sympify([-(p[1] + h[1])
                                 for p, h in zip(padding, halo)]))

        ops_dat_array = Array(
            name=namespace['ops_dat_name'](f.name),
            dimensions=(DefaultDimension(name='dat', default_value=time_dims),),
            dtype=namespace['ops_dat_type'],            
            scope='stack'
        )

        dat_decls = []
        for i in range(time_dims):
            name = '%s%s%s' % (f.name, time_index, i)
            name_to_ops_dat[name] = ops_dat_array.indexify(
                [Symbol('%s%s' % (time_index, i))]
            )
            dat_decls.append(namespace['ops_decl_dat'](
                block,
                1,
                Symbol(dim.name),
                Symbol(base.name),
                Symbol(d_m.name),
                Symbol(d_p.name),
                Byref(f.indexify([i])),
                Literal('"%s"' % f._C_typedata),
                Literal('"%s"' % name)
            ))

        ops_decl_dat = Expression(ClusterizedEq(Eq(
            ops_dat_array,
            ListInitializer(dat_decls)
        )))
    else:
        ops_dat = OpsDat("%s_dat" % f.name)
        name_to_ops_dat[f.name] = ops_dat

        d_p_val = tuple(sympify([p[0] + h[0]
                                 for p, h in zip(f.padding, f.halo)]))
        d_m_val = tuple(sympify([-(p[1] + h[1])
                                 for p, h in zip(f.padding, f.halo)]))
        dim_shape = sympify(f.shape)

        ops_decl_dat = Expression(ClusterizedEq(Eq(
            ops_dat,
            namespace['ops_decl_dat'](
                block,
                1,
                Symbol(dim.name),
                Symbol(base.name),
                Symbol(d_m.name),
                Symbol(d_p.name),
                Byref(f.indexify([0])),
                Literal('"%s"' % f._C_typedata),
                Literal('"%s"' % f.name)
            )
        )))

    res.append(Expression(ClusterizedEq(Eq(dim, ListInitializer(dim_shape)))))
    res.append(Expression(ClusterizedEq(Eq(base, ListInitializer(base_val)))))
    res.append(Expression(ClusterizedEq(Eq(d_p, ListInitializer(d_p_val)))))
    res.append(Expression(ClusterizedEq(Eq(d_m, ListInitializer(d_m_val)))))
    res.append(ops_decl_dat)

    return res


def create_ops_fetch(f, name_to_ops_dat):

    if f.is_TimeFunction:
        ops_fetch = [Call(namespace['ops_dat_fetch_data'], [
            name_to_ops_dat['%s%s%d' % (f.name,
                                        f.indices[f._time_position],
                                        i)],
            0,
            Byref(f.indexify([i]))
        ]) for i in range(f.shape[f._time_position])]

    else:
        ops_fetch = Call(namespace['ops_dat_fetch_data'], [
            name_to_ops_dat[f.name], 0,  Byref(f.indexify([0]))
        ])

    return ops_fetch


def create_ops_par_loop(
        trees, ops_kernel, parameters, block, name_to_ops_dat, par_to_ops_stencil, dims):
    it_range = []
    for tree in trees:
        if isinstance(tree, IterationTree):
            for bounds in [it.bounds() for it in tree]:
                it_range.extend(bounds)

    range_array = Array(
        name='%s_range' % ops_kernel.name,
        dimensions=(DefaultDimension(
            name='range', default_value=len(it_range)),),
        dtype=np.int32,
        scope='stack'
    )

    range_array_init = Expression(ClusterizedEq(Eq(
        range_array,
        ListInitializer(it_range)
    )))

    ops_par_loop_call = Call(
        namespace['ops_par_loop'], [
            Literal(ops_kernel.name),
            Literal('"%s"' % ops_kernel.name),
            block,
            dims,
            range_array,
            *[create_ops_arg(p, name_to_ops_dat, par_to_ops_stencil) for p in parameters]
        ]
    )

    return [range_array_init], ops_par_loop_call


def create_ops_arg(p, name_to_ops_dat, par_to_ops_stencil):
    if p.is_Constant:
        return namespace['ops_arg_gbl'](
            Byref(Constant(name=p.name[1:])),
            1,
            Literal('"%s"' % dtype_to_cstr(p.dtype)),
            namespace['ops_read']
        )
    else:
        return namespace['ops_arg_dat'](
            name_to_ops_dat[p.name],
            1,
            par_to_ops_stencil[p],
            Literal('"%s"' % dtype_to_cstr(p.dtype)),
            namespace['ops_read'] if p.read_only else namespace['ops_write']
        )


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
